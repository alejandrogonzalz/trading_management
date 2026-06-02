import time

import binance_common.utils as binance_utils
from binance.client import Client as StandardClient
from binance_common.configuration import ConfigurationRestAPI

# Specialized Binance SDKs for Lead/Futures
from binance_sdk_copy_trading.rest_api import CopyTradingRestAPI
from binance_sdk_derivatives_trading_usds_futures.rest_api import (
    DerivativesTradingUsdsFuturesRestAPI,
)
from fastapi import HTTPException

from app.core.config import settings
from app.db.database import db_session
from app.db.models import LeadTrade
from app.services import audit_service
from app.services.binance_service import binance_client, round_step_size


class FuturesService:
    _time_offset = 0

    def __init__(self):
        self.api_key = settings.LEAD_API_KEY
        self.api_secret = settings.LEAD_API_SECRET

        if self.api_key and self.api_secret:
            # Standard python-binance client for high-reliability orders
            self.std_client = StandardClient(self.api_key, self.api_secret)

            # Copy Trading Configuration (uses standard API)
            self.copy_config = ConfigurationRestAPI(
                api_key=self.api_key,
                api_secret=self.api_secret,
                base_path="https://api.binance.com",
            )

            # Futures Configuration (uses fapi)
            self.futures_config = ConfigurationRestAPI(
                api_key=self.api_key,
                api_secret=self.api_secret,
                base_path="https://fapi.binance.com",
            )

            self.sync_time()

            self.copy_client = CopyTradingRestAPI(self.copy_config)
            self.lead_client = DerivativesTradingUsdsFuturesRestAPI(self.futures_config)
        else:
            self.copy_client = None
            self.lead_client = None

    def sync_time(self):
        """Syncs local time offset with Binance server time via monkeypatching."""
        try:
            res = binance_client.get_server_time()
            server_time = res["serverTime"]
            local_time = int(time.time() * 1000)

            # Calculate offset: server - local
            # Subtract 1000ms as a safety buffer to ensure we are never "ahead" of server
            FuturesService._time_offset = (server_time - local_time) - 1000

            if hasattr(self, "std_client"):
                self.std_client.timestamp_offset = FuturesService._time_offset

            # Monkeypatch the SDK's internal timestamp generator
            def patched_get_timestamp():
                return int(time.time() * 1000) + FuturesService._time_offset

            binance_utils.get_timestamp = patched_get_timestamp
            print(f"Lead Sync: Server Time Offset = {FuturesService._time_offset}ms (Buffer Applied)")
        except Exception as e:
            print(f"Lead Sync Time Error: {e}")

    def _ensure_client(self):
        if not self.lead_client or not self.copy_client or not self.std_client:
            raise HTTPException(
                status_code=400,
                detail="Lead Trading API keys not configured in backend .env",
            )

    def get_balances(self):
        """Retrieves USDS-M Futures wallet balances."""
        self._ensure_client()
        try:
            # v2 returns account info including balances
            res = self.lead_client.account_information_v2()
            return res.data()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Futures Balance Error: {str(e)}")

    def get_lead_history(self, symbol: str = None):
        """Retrieves closed Lead/Futures trades from SQLite."""
        try:
            query = db_session.query(LeadTrade).filter(LeadTrade.status == "CLOSED")
            if symbol:
                query = query.filter(LeadTrade.symbol == symbol.upper())

            trades_models = query.order_by(LeadTrade.close_time.desc()).all()
            trades = []
            for t in trades_models:
                doc = {
                    "id": t.id,
                    "symbol": t.symbol,
                    "side": t.side,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "quantity": t.quantity,
                    "leverage": t.leverage,
                    "tp": t.tp,
                    "sl": t.sl,
                    "status": t.status,
                    "close_time": t.close_time * 1000 if t.close_time else 0,
                    "entry_fees": t.entry_fees,
                    "exit_fees": t.exit_fees,
                    "exit_fee_asset": t.exit_fee_asset,
                    "close_reason": t.close_reason,
                    "timestamp": t.timestamp,
                }
                trades.append(doc)
            return trades
        finally:
            db_session.remove()

    def get_lead_status(self):
        """Returns the user's status. Silently handles missing keys."""
        if not self.copy_client:
            return {"status": "RESTRICTED", "message": "API Keys Not Configured"}
        try:
            res = self.copy_client.get_futures_lead_trader_status()
            return res.data()
        except Exception as e:
            return {"status": "ERROR", "message": str(e)}

    def get_tradable_symbols(self):
        """Returns whitelisted symbols. Returns empty list if keys missing."""
        if not self.copy_client:
            return []
        try:
            res = self.copy_client.get_futures_lead_trading_symbol_whitelist()
            return res.data()
        except Exception as e:
            print(f"Lead Whitelist Fetch Error: {e}")
            return []

    def set_leverage(self, symbol: str, leverage: int):
        """Sets leverage for a specific symbol."""
        self._ensure_client()
        try:
            params = {
                "symbol": symbol.upper(),
                "leverage": leverage,
                "recv_window": 60000,
            }
            res = self.lead_client.change_initial_leverage(**params)
            return res.data()
        except Exception as e:
            # Retry mechanism for timeout
            if "Read timed out" in str(e):
                print("Leverage set timed out, retrying...")
                time.sleep(2)
                res = self.lead_client.change_initial_leverage(**params)
                return res.data()
            raise HTTPException(status_code=400, detail=f"Leverage Error: {str(e)}")

    def create_lead_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float = None,
    ):
        """
        Places a USDS-M Futures order.
        If account is a Lead account, followers will copy this automatically.
        """
        self._ensure_client()
        try:
            params = {
                "symbol": symbol.upper(),
                "side": side.upper(),
                "type": order_type.upper(),
                "quantity": quantity,
            }
            if price:
                params["price"] = price
                params["timeInForce"] = "GTC"

            # Execute via specialized Lead Order Client
            res = self.lead_client.new_order(**params)
            response = res.data()

            # Log to audit
            audit_service.log_api_call("POST", "lead/order", params, response)

            return response
        except Exception as e:
            audit_service.log_api_call("POST", "lead/order-FAILED", {"symbol": symbol}, str(e), 400)
            raise HTTPException(status_code=400, detail=f"Order Failed: {str(e)}")

    def _wait_for_position(self, symbol: str, target_side: str, timeout: int = 15):
        """Polls until position matches the intended side and has non-zero quantity."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            positions = self.get_active_positions(symbol)
            pos = next((p for p in positions if p.symbol == symbol.upper()), None)
            if pos:
                amt = float(pos.position_amt or 0)
                # amt > 0 for LONG/BUY, amt < 0 for SHORT/SELL
                if (target_side == "BUY" and amt > 0) or (target_side == "SELL" and amt < 0):
                    print(f"Position Confirmed for {symbol}: {amt}")
                    return True
            time.sleep(1.5)
        return False

    def _wait_for_no_orders(self, symbol: str, timeout: int = 10):
        """Polls until no open orders exist for the symbol."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            res = self.lead_client.current_all_open_orders(symbol=symbol.upper())
            if not res.data():
                return True
            time.sleep(1)
        return False

    def _verify_order_exists(self, symbol: str, client_algo_id: str, timeout: int = 10):
        """Polls until a specific algo order ID is confirmed on the book."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Check current algo orders (SDK uses current_all_algo_open_orders)
                res = self.lead_client.current_all_algo_open_orders(symbol=symbol.upper())
                open_algos = res.data()
                found = any(getattr(o, "client_algo_id", "") == client_algo_id for o in open_algos)
                if found:
                    print(f"✅ Algo Order Verified: {client_algo_id}")
                    return True
            except Exception as e:
                print(f"Verification Check Error for {client_algo_id}: {e}")
            time.sleep(2)
        print(f"❌ Algo Order Verification FAILED for {client_algo_id}")
        return False

    def create_smart_lead_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        tp_price: float,
        sl_price: float,
        leverage: int = 10,
    ):
        """
        BULLETPROOF SMART TRADE:
        1. Place Market Entry
        2. Wait/Verify Position Confirmation
        3. Place Standard Protection Orders (Standard Engine)
        4. Rollback on failure (Cancel all + Market Close)
        """
        self._ensure_client()
        trade_id = f"LEAD_{int(time.time())}"

        try:
            # 1. SETUP & LEVERAGE
            self.set_leverage(symbol, leverage)
            exchange_info = binance_client.futures_exchange_info()
            symbol_info = next(
                (s for s in exchange_info["symbols"] if s["symbol"] == symbol.upper()),
                None,
            )
            if not symbol_info:
                raise HTTPException(status_code=400, detail=f"Symbol {symbol} not found.")

            tick_size = float(next(f for f in symbol_info["filters"] if f["filterType"] == "PRICE_FILTER")["tickSize"])
            step_size = float(next(f for f in symbol_info["filters"] if f["filterType"] == "LOT_SIZE")["stepSize"])

            rounded_qty = round_step_size(quantity, step_size)
            rounded_tp = round_step_size(tp_price, tick_size) if tp_price > 0 else 0
            rounded_sl = round_step_size(sl_price, tick_size) if sl_price > 0 else 0

            # 2. ENTRY (STATE: ENTRY_PLACED)
            trade = LeadTrade(
                id=trade_id,
                symbol=symbol.upper(),
                side=side.upper(),
                entry_price=0,
                quantity=rounded_qty,
                tp=rounded_tp,
                sl=rounded_sl,
                leverage=leverage,
                status="ENTRY_PLACED",
                timestamp=time.time(),
            )
            db_session.add(trade)
            db_session.commit()

            entry_params = {
                "symbol": symbol.upper(),
                "side": side.upper(),
                "type": "MARKET",
                "quantity": rounded_qty,
                "new_client_order_id": f"ENT_{trade_id}",
            }
            res = self.lead_client.new_order(**entry_params)
            entry_res = res.data()
            audit_service.log_api_call("POST", "lead/smart-entry", entry_params, entry_res)

            # 3. VERIFY POSITION & CAPTURE ACTUALS (STATE: ENTRY_FILLED)
            if not self._wait_for_position(symbol, side.upper()):
                raise Exception("Position verification timed out.")

            # Fetch execution details for entry fees and precise price
            history = self.std_client.futures_account_trades(symbol=symbol.upper(), limit=10)
            entry_fill = next(
                (h for h in history if h.get("clientOrderId") == f"ENT_{trade_id}"),
                None,
            )

            actual_entry_price = (
                float(entry_fill["price"])
                if entry_fill
                else float(getattr(entry_res, "avg_price", 0) or getattr(entry_res, "price", 0))
            )
            actual_qty = float(entry_fill["qty"]) if entry_fill else rounded_qty
            entry_fees = float(entry_fill["commission"]) if entry_fill else 0

            if actual_entry_price == 0:
                ticker = binance_client.futures_symbol_ticker(symbol=symbol.upper())
                actual_entry_price = float(ticker.get("price", 0))

            trade.status = "ENTRY_FILLED"
            trade.entry_price = actual_entry_price
            trade.quantity = actual_qty
            trade.entry_fees = entry_fees
            trade.entry_fee_asset = entry_fill.get("commissionAsset", "USDT") if entry_fill else "USDT"
            db_session.commit()

            # 4. PROTECTION (STATE: ACTIVE)
            exit_side = "SELL" if side.upper() == "BUY" else "BUY"
            protection_orders = []

            # Get current price to validate TP/SL distances
            try:
                ticker = binance_client.futures_symbol_ticker(symbol=symbol.upper())
                current_price = float(ticker.get("price", actual_entry_price))
            except:
                current_price = actual_entry_price

            # Helper function to place protection order with price adjustment retry
            def place_protection_order(order_type, target_price, role):
                max_retries = 3
                adjusted_price = target_price

                for attempt in range(max_retries):
                    try:
                        if order_type == "STOP_MARKET":
                            res = self.std_client.futures_create_order(
                                symbol=symbol.upper(),
                                side=exit_side,
                                type="STOP_MARKET",
                                stopPrice=adjusted_price,
                                closePosition=True,
                                workingType="MARK_PRICE",
                            )
                        else:  # TAKE_PROFIT_MARKET
                            res = self.std_client.futures_create_order(
                                symbol=symbol.upper(),
                                side=exit_side,
                                type="TAKE_PROFIT_MARKET",
                                stopPrice=adjusted_price,
                                closePosition=True,
                                workingType="MARK_PRICE",
                            )

                        cid = res.get("clientAlgoId") or res.get("clientOrderId")
                        if not self._verify_order_exists(symbol, cid):
                            raise Exception(f"{role} order verification failed.")

                        return {
                            "orderId": res.get("algoId") or res.get("orderId"),
                            "status": "NEW",
                            "type": order_type,
                            "role": role,
                            "clientOrderId": cid,
                        }, adjusted_price

                    except Exception as e:
                        error_msg = str(e)
                        print(
                            f"Protection order attempt {attempt + 1}/{max_retries} for {role} at {adjusted_price}: {error_msg}"
                        )

                        # Check if error is "Order would immediately trigger"
                        if "Order would immediately trigger" in error_msg and attempt < max_retries - 1:
                            # Adjust price based on position side and order type
                            adjustment_factor = 0.005  # 0.5% adjustment
                            if side.upper() == "BUY":  # LONG position
                                if role == "SL":  # Stop loss below entry
                                    # SL would trigger immediately if price below SL
                                    # Move SL further down (more room for price drop)
                                    adjusted_price = adjusted_price * (1 - adjustment_factor)
                                else:  # TP for LONG
                                    # TP would trigger immediately if price above TP
                                    # Move TP further up (more room for price rise)
                                    adjusted_price = adjusted_price * (1 + adjustment_factor)
                            else:  # SHORT position
                                if role == "SL":  # Stop loss above entry
                                    # SL would trigger immediately if price above SL
                                    # Move SL further up (more room for price rise)
                                    adjusted_price = adjusted_price * (1 + adjustment_factor)
                                else:  # TP for SHORT
                                    # TP would trigger immediately if price below TP
                                    # Move TP further down (more room for price drop)
                                    adjusted_price = adjusted_price * (1 - adjustment_factor)

                            # Re-round to tick size
                            adjusted_price = round_step_size(adjusted_price, tick_size)
                            print(f"Adjusted {role} price to {adjusted_price} and retrying...")
                            continue
                        else:
                            raise  # Re-raise if not retriable or max retries reached

            # Place SL order with retry logic
            if rounded_sl > 0:
                try:
                    sl_order, final_sl_price = place_protection_order("STOP_MARKET", rounded_sl, "SL")
                    protection_orders.append(sl_order)
                    audit_service.log_api_call(
                        "POST",
                        "lead/std-sl",
                        {"sl": final_sl_price},
                        {"status": "placed"},
                    )
                    # Update SL price in trade if it was adjusted
                    if final_sl_price != rounded_sl:
                        trade.sl = final_sl_price
                        db_session.commit()
                except Exception as sl_err:
                    print(f"Failed to place SL order after retries: {sl_err}")
                    raise Exception(f"SL order placement failed: {sl_err}")

            # Place TP order with retry logic
            if rounded_tp > 0:
                try:
                    tp_order, final_tp_price = place_protection_order("TAKE_PROFIT_MARKET", rounded_tp, "TP")
                    protection_orders.append(tp_order)
                    audit_service.log_api_call(
                        "POST",
                        "lead/std-tp",
                        {"tp": final_tp_price},
                        {"status": "placed"},
                    )
                    # Update TP price in trade if it was adjusted
                    if final_tp_price != rounded_tp:
                        trade.tp = final_tp_price
                        db_session.commit()
                except Exception as tp_err:
                    print(f"Failed to place TP order after retries: {tp_err}")
                    raise Exception(f"TP order placement failed: {tp_err}")

            # 5. CONFIRMATION
            trade.status = "ACTIVE"
            trade.protection_orders = protection_orders
            db_session.commit()
            return {
                "trade_id": trade_id,
                "status": "ACTIVE",
                "entry_price": actual_entry_price,
            }

        except Exception as e:
            error_msg = str(e)
            print(f"CRITICAL ERROR in Smart Trade {trade_id}: {error_msg}")

            # ROLLBACK: Atomic cleanup
            try:
                # 1. Cancel all protection attempts for this trade
                self.std_client.futures_cancel_all_open_orders(symbol=symbol.upper())
                self._wait_for_no_orders(symbol)

                # 2. Market Close Position
                self.close_position(symbol)

                # 3. Verify Position = 0
                positions = self.get_active_positions(symbol)
                pos = next((p for p in positions if p.symbol == symbol.upper()), None)
                if not pos or float(pos.position_amt or 0) == 0:
                    trade.status = "ROLLBACK_SUCCESS"
                    trade.error = error_msg
                    db_session.commit()
                    error_msg += " (Rollback successful)"
                else:
                    raise Exception(f"Position still exists after rollback: {pos.position_amt}")

            except Exception as rollback_err:
                error_msg += f" (ROLLBACK FAILED: {str(rollback_err)} - MANUAL INTERVENTION REQUIRED)"
                trade.status = "ROLLBACK_FAILED"
                trade.error = error_msg
                db_session.commit()

            audit_service.log_api_call("POST", "lead/smart-trade-FAILED", {"symbol": symbol}, error_msg, 400)
            raise HTTPException(status_code=400, detail=error_msg)
        finally:
            db_session.remove()

    def get_active_positions(self, symbol: str = None):
        """Retrieves currently open Futures positions."""
        self._ensure_client()
        try:
            params = {}
            if symbol:
                params["symbol"] = symbol.upper()

            res = self.lead_client.position_information_v2(**params)
            raw_positions = res.data()
            # Filter for non-zero positions. SDK uses snake_case attributes.
            return [p for p in raw_positions if float(p.position_amt or 0) != 0]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Position Error: {str(e)}")

    def get_open_orders(self, symbol: str = None):
        """Retrieves pending Lead orders (Standard + Algo) and merges with virtual Smart Trades."""
        self._ensure_client()
        try:
            params = {"recv_window": 60000}
            if symbol:
                params["symbol"] = symbol.upper()

            # 1. Fetch real orders from Binance (Standard)
            res = self.lead_client.current_all_open_orders(**params)
            raw_orders = res.data()

            # 2. Fetch Algo orders (SL/TP)
            algo_res = self.lead_client.current_all_algo_open_orders(**params)
            raw_algos = algo_res.data()

            # 3. Format Algos to match standard order structure for the frontend
            formatted_algos = []
            for a in raw_algos:
                formatted_algos.append(
                    {
                        "orderId": getattr(a, "algo_id", None),
                        "symbol": getattr(a, "symbol", None),
                        "side": getattr(a, "side", None),
                        "type": getattr(a, "order_type", None),
                        "origQty": getattr(a, "quantity", 0),
                        "price": getattr(a, "price", 0),
                        "stopPrice": getattr(a, "trigger_price", 0),
                        "clientOrderId": getattr(a, "client_algo_id", None),
                        "status": "NEW",
                    }
                )

            # 4. Fetch virtual setups from SQLite
            query = db_session.query(LeadTrade).filter(LeadTrade.status.in_(["ACTIVE", "ENTRY_FILLED"]))
            if symbol:
                query = query.filter(LeadTrade.symbol == symbol.upper())

            virtual_setups = query.all()

            # 5. Format virtual setups
            formatted_setups = []
            for vs in virtual_setups:
                formatted_setups.append(
                    {
                        "id": vs.id,
                        "orderId": vs.id,
                        "symbol": vs.symbol,
                        "side": vs.side,
                        "type": "POSITION",  # Virtual tag
                        "origQty": vs.quantity,
                        "price": vs.entry_price,
                        "tp": vs.tp,
                        "sl": vs.sl,
                        "clientOrderId": vs.id,
                        "needs_protection": vs.status == "ENTRY_FILLED",  # Flag for UI warning
                        "smart_meta": {
                            "entry_price": vs.entry_price,
                            "tp": vs.tp,
                            "sl": vs.sl,
                            "side": vs.side,
                            "quantity": vs.quantity,
                            "leverage": vs.leverage,
                            "status": vs.status,  # Include status in meta
                        },
                    }
                )

            return raw_orders + formatted_algos + formatted_setups
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Open Orders Error: {str(e)}")
        finally:
            db_session.remove()

    def cancel_order(self, symbol: str, order_id: int):
        """Cancels a pending Lead order."""
        self._ensure_client()
        try:
            res = self.lead_client.cancel_order(symbol=symbol.upper(), orderId=order_id)
            return res.data()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Cancel Failed: {str(e)}")

    def get_binance_trade_history(self, symbol: str = None):
        """Retrieves raw execution history from USDS-M Futures."""
        self._ensure_client()
        try:
            params = {}
            if symbol:
                params["symbol"] = symbol.upper()

            # Debugging
            print(f"DEBUG: Fetching Lead History for {symbol} with params: {params}")

            res = self.lead_client.account_trade_list(**params)
            response = res.data()

            # Log successful fetch
            audit_service.log_api_call("GET", "lead/binance-history", params, f"Fetched {len(response)} trades")

            return response
        except Exception as e:
            print(f"CRITICAL: History Fetch Failed: {e}")
            # Log failure
            audit_service.log_api_call("GET", "lead/binance-history-FAILED", params, str(e), 400)
            raise HTTPException(status_code=400, detail=f"History Error: {str(e)}")

    def close_position(self, symbol: str, quantity: float = None):
        """
        BULLETPROOF PANIC SELL:
        1. Cancel all open orders for symbol
        2. Market exit exact position size
        3. Verify position is zero
        4. Update DB
        """
        self._ensure_client()
        try:
            # 1. Clear the deck: Cancel all orders
            print(f"Panic Sell: Cancelling all orders for {symbol}...")
            self.std_client.futures_cancel_all_open_orders(symbol=symbol.upper())
            self._wait_for_no_orders(symbol)

            # 2. Get exact current position
            positions = self.get_active_positions(symbol)
            pos = next((p for p in positions if p.symbol == symbol.upper()), None)
            if not pos:
                return {"status": "ALREADY_CLOSED"}

            amt = float(pos.position_amt or 0)
            if amt == 0:
                return {"status": "ALREADY_CLOSED"}

            # 3. Determine exit side & exact quantity
            exit_side = "SELL" if amt > 0 else "BUY"
            exit_qty = abs(amt) if not quantity else quantity

            # 4. Execute Market Close
            exit_client_id = f"EXT_{int(time.time())}"
            params = {
                "symbol": symbol.upper(),
                "side": exit_side,
                "type": "MARKET",
                "quantity": exit_qty,
                "new_client_order_id": exit_client_id,
            }

            res = self.lead_client.new_order(**params)
            response = res.data()
            audit_service.log_api_call("POST", "lead/close-position", params, response)

            # 5. Verify Position is gone
            if not quantity:  # Only verify if we intended to close the WHOLE position
                start_v = time.time()
                while time.time() - start_v < 10:
                    check_pos = self.get_active_positions(symbol)
                    if not any(p.symbol == symbol.upper() for p in check_pos):
                        print(f"✅ Panic Sell Verified: Position for {symbol} is ZERO.")
                        break
                    time.sleep(1.5)

            # 6. Fetch execution details for exit price and fees
            exit_price = 0
            exit_fees = 0
            exit_fee_asset = "USDT"

            try:
                # Wait a moment for execution to hit history
                time.sleep(1)
                history = self.std_client.futures_account_trades(symbol=symbol.upper(), limit=5)
                exit_fill = next(
                    (h for h in history if h.get("clientOrderId") == exit_client_id),
                    None,
                )

                if exit_fill:
                    exit_price = float(exit_fill["price"])
                    exit_fees = float(exit_fill["commission"])
                    exit_fee_asset = exit_fill.get("commissionAsset", "USDT")
                else:
                    # Fallback to ticker if not found in recent history
                    ticker = binance_client.futures_symbol_ticker(symbol=symbol.upper())
                    exit_price = float(ticker.get("price", 0))
            except Exception as e:
                print(f"Error fetching exit execution details: {e}")
                ticker = binance_client.futures_symbol_ticker(symbol=symbol.upper())
                exit_price = float(ticker.get("price", 0))

            # 7. Mark associated SQLite trades as CLOSED
            db_session.query(LeadTrade).filter(
                LeadTrade.symbol == symbol.upper(),
                LeadTrade.status.in_(["ACTIVE", "ENTRY_FILLED", "ENTRY_PLACED"]),
            ).update(
                {
                    "status": "CLOSED",
                    "exit_price": exit_price,
                    "exit_fees": exit_fees,
                    "exit_fee_asset": exit_fee_asset,
                    "close_time": time.time(),
                    "close_reason": "PANIC_SELL",
                },
                synchronize_session=False,
            )
            db_session.commit()

            return response
        except Exception as e:
            db_session.rollback()
            raise HTTPException(status_code=400, detail=f"Panic Sell Failed: {str(e)}")
        finally:
            db_session.remove()

    def reconcile_lead_trades(self):
        """
        Background task to sync Lead/Futures trades with Binance reality.
        Runs every 30 seconds.
        """
        self._ensure_client()
        try:
            # 1. Fetch all trades that are not CLOSED or FAILED
            pending_trades = (
                db_session.query(LeadTrade)
                .filter(LeadTrade.status.in_(["ACTIVE", "ENTRY_PLACED", "ENTRY_FILLED"]))
                .all()
            )

            if not pending_trades:
                return

            print(f"🔄 Reconciling {len(pending_trades)} Lead trades...")

            for trade in pending_trades:
                tid = trade.id
                symbol = trade.symbol

                # A. Handle ENTRY_PLACED: Check if the entry order filled
                if trade.status == "ENTRY_PLACED":
                    try:
                        order = self.std_client.futures_get_order(symbol=symbol, origClientOrderId=f"ENT_{tid}")
                        if order["status"] == "FILLED":
                            print(f"Entry Filled for {tid}. Updating state...")
                            avg_price = float(order.get("avgPrice", 0))
                            trade.status = "ENTRY_FILLED"
                            trade.entry_price = avg_price

                            # Capture Entry Fees
                            try:
                                history = self.std_client.futures_account_trades(symbol=symbol, limit=10)
                                fill = next(
                                    (h for h in history if h.get("clientOrderId") == f"ENT_{tid}"),
                                    None,
                                )
                                if fill:
                                    trade.entry_fees = float(fill.get("commission", 0))
                                    trade.entry_fee_asset = fill.get("commissionAsset", "USDT")
                            except:
                                pass

                            db_session.commit()
                    except Exception as e:
                        print(f"Error checking entry for {tid}: {e}")

                # B. Handle ENTRY_FILLED
                elif trade.status == "ENTRY_FILLED":
                    positions = self.get_active_positions(symbol)
                    pos = next((p for p in positions if p.symbol == symbol), None)
                    if not pos or float(pos.position_amt or 0) == 0:
                        trade.status = "CLOSED"
                        trade.close_reason = "SYNC_POSITION_GONE"
                        # TODO: REVIEW - Using current ticker price as exit price is inaccurate if the app was down.
                        # Should query futures_account_trades for the actual historical fill that closed the position.
                        try:
                            ticker = self.std_client.futures_symbol_ticker(symbol=symbol)
                            trade.exit_price = float(ticker.get("price", 0))
                        except:
                            trade.exit_price = 0
                        trade.exit_fees = 0
                        trade.exit_fee_asset = "USDT"
                        trade.close_time = time.time()
                        db_session.commit()

                # C. Handle ACTIVE
                elif trade.status == "ACTIVE":
                    res = self.lead_client.current_all_open_orders(symbol=symbol)
                    open_orders = res.data()
                    open_client_ids = [getattr(o, "client_order_id", "") for o in open_orders]

                    protection_ids = [
                        p.get("clientOrderId") for p in (trade.protection_orders or []) if p.get("clientOrderId")
                    ]
                    missing_ids = [pid for pid in protection_ids if pid not in open_client_ids]

                    if missing_ids:
                        print(f"Detection: Protection order(s) {missing_ids} missing for {tid}. Checking fills...")
                        history = self.std_client.futures_account_trades(symbol=symbol, limit=20)
                        fill = next(
                            (h for h in history if h.get("clientOrderId") in missing_ids),
                            None,
                        )

                        if fill:
                            print(f"Trade {tid} closed via {fill['clientOrderId']} at {fill['price']}")
                            trade.status = "CLOSED"
                            trade.exit_price = float(fill["price"])
                            trade.exit_fees = float(fill.get("commission", 0))
                            trade.exit_fee_asset = fill.get("commissionAsset", "USDT")
                            trade.close_time = fill["time"] / 1000.0
                            trade.close_reason = "TAKE_PROFIT" if "TP" in fill.get("clientOrderId", "") else "STOP_LOSS"
                            db_session.commit()
                            self.std_client.futures_cancel_all_open_orders(symbol=symbol)
                        else:
                            positions = self.get_active_positions(symbol)
                            pos = next((p for p in positions if p.symbol == symbol), None)
                            if not pos or float(pos.position_amt or 0) == 0:
                                print(f"Trade {tid} has no position and no orders. Closing in DB.")
                                trade.status = "CLOSED"
                                trade.close_reason = "RECONCILED_EMPTY"
                                db_session.commit()

        except Exception as e:
            print(f"Reconciler Lead Error: {e}")
        finally:
            db_session.remove()


# Singleton instance
futures_service = FuturesService()
