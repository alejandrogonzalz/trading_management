import os
import time
import math
from typing import Optional, List, Dict, Any
from fastapi import HTTPException

# Specialized Binance SDKs for Lead/Futures
from binance_sdk_copy_trading.rest_api import CopyTradingRestAPI
from binance_sdk_derivatives_trading_usds_futures.rest_api import DerivativesTradingUsdsFuturesRestAPI
from binance_common.configuration import ConfigurationRestAPI
from binance.client import Client as StandardClient

from app.core.config import settings
from app.services import audit_service
from app.db.database import db
from app.services.binance_service import binance_client, round_step_size

import binance_common.utils as binance_utils

# Isolated collection for Lead/Futures trades
lead_trades_collection = db.lead_trades

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
                base_path="https://api.binance.com"
            )
            
            # Futures Configuration (uses fapi)
            self.futures_config = ConfigurationRestAPI(
                api_key=self.api_key,
                api_secret=self.api_secret,
                base_path="https://fapi.binance.com"
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
            server_time = res['serverTime']
            local_time = int(time.time() * 1000)
            
            # Calculate offset: server - local
            # Subtract 1000ms as a safety buffer to ensure we are never "ahead" of server
            FuturesService._time_offset = (server_time - local_time) - 1000
            
            if hasattr(self, 'std_client'):
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
                detail="Lead Trading API keys not configured in backend .env"
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
        """Retrieves closed Lead/Futures trades from MongoDB."""
        query = {"status": "CLOSED"}
        if symbol:
            query["symbol"] = symbol.upper()
        
        cursor = lead_trades_collection.find(query).sort("close_time", -1)
        trades = []
        for doc in cursor:
            doc["id"] = str(doc.pop("_id"))
            # Normalize for frontend (BottomPanel expects .price or .avgPrice)
            if "exit_price" in doc:
                doc["price"] = doc["exit_price"]
            if "close_time" in doc:
                doc["time"] = doc["close_time"] * 1000 # To ms
            trades.append(doc)
        return trades

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
                "leverage": leverage
            }
            res = self.lead_client.change_initial_leverage(**params)
            return res.data()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Leverage Error: {str(e)}")

    def create_lead_order(self, symbol: str, side: str, order_type: str, quantity: float, price: float = None):
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
                "quantity": quantity
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
                found = any(getattr(o, 'client_algo_id', '') == client_algo_id for o in open_algos)
                if found:
                    print(f"✅ Algo Order Verified: {client_algo_id}")
                    return True
            except Exception as e:
                print(f"Verification Check Error for {client_algo_id}: {e}")
            time.sleep(2)
        print(f"❌ Algo Order Verification FAILED for {client_algo_id}")
        return False

    def create_smart_lead_order(self, symbol: str, side: str, quantity: float, tp_price: float, sl_price: float, leverage: int = 10):
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
            symbol_info = next((s for s in exchange_info['symbols'] if s['symbol'] == symbol.upper()), None)
            if not symbol_info:
                raise HTTPException(status_code=400, detail=f"Symbol {symbol} not found.")

            tick_size = float(next(f for f in symbol_info['filters'] if f['filterType'] == 'PRICE_FILTER')['tickSize'])
            step_size = float(next(f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE')['stepSize'])
            
            rounded_qty = round_step_size(quantity, step_size)
            rounded_tp = round_step_size(tp_price, tick_size) if tp_price > 0 else 0
            rounded_sl = round_step_size(sl_price, tick_size) if sl_price > 0 else 0
            
            # 2. ENTRY (STATE: ENTRY_PLACED)
            metadata = {
                "_id": trade_id, "symbol": symbol.upper(), "side": side.upper(),
                "entry_price": 0, "quantity": rounded_qty, "tp": rounded_tp, "sl": rounded_sl,
                "leverage": leverage, "status": "ENTRY_PLACED", "timestamp": time.time()
            }
            lead_trades_collection.insert_one(metadata)

            entry_params = {
                "symbol": symbol.upper(), 
                "side": side.upper(), 
                "type": "MARKET", 
                "quantity": rounded_qty,
                "new_client_order_id": f"ENT_{trade_id}"
            }
            res = self.lead_client.new_order(**entry_params)
            entry_res = res.data()
            audit_service.log_api_call("POST", "lead/smart-entry", entry_params, entry_res)
            
            # 3. VERIFY POSITION (STATE: ENTRY_FILLED)
            if not self._wait_for_position(symbol, side.upper()):
                raise Exception("Position verification timed out. Entry might have failed or is lagging.")
            
            # SDK returns avg_price as String. Must cast.
            raw_avg = getattr(entry_res, 'avg_price', '0') or '0'
            actual_entry_price = float(raw_avg)
            if actual_entry_price == 0:
                # Fallback to current ticker if market order price not returned correctly
                ticker = binance_client.futures_symbol_ticker(symbol=symbol.upper())
                actual_entry_price = float(ticker.get('price', 0))

            lead_trades_collection.update_one({"_id": trade_id}, {"$set": {"status": "ENTRY_FILLED", "entry_price": actual_entry_price}})

            # 4. PROTECTION (STATE: ACTIVE)
            exit_side = "SELL" if side.upper() == "BUY" else "BUY"
            protection_orders = []

            # Using standard Client (std_client) for reliability and MARK_PRICE support
            if rounded_sl > 0:
                sl_res = self.std_client.futures_create_order(
                    symbol=symbol.upper(),
                    side=exit_side,
                    type="STOP_MARKET",
                    stopPrice=rounded_sl,
                    closePosition=True,
                    workingType="MARK_PRICE"
                    # NOTE: python-binance might auto-prefix clientOrderId, let's use the returned ID
                )
                cid = sl_res.get('clientAlgoId') or sl_res.get('clientOrderId')
                if not self._verify_order_exists(symbol, cid):
                    raise Exception("SL order verification failed.")
                
                protection_orders.append({
                    "orderId": sl_res.get('algoId') or sl_res.get('orderId'),
                    "status": "NEW",
                    "type": "STOP_MARKET",
                    "role": "SL",
                    "clientOrderId": cid
                })
                audit_service.log_api_call("POST", "lead/std-sl", {"sl": rounded_sl}, sl_res)

            if rounded_tp > 0:
                tp_res = self.std_client.futures_create_order(
                    symbol=symbol.upper(),
                    side=exit_side,
                    type="TAKE_PROFIT_MARKET",
                    stopPrice=rounded_tp,
                    closePosition=True,
                    workingType="MARK_PRICE"
                )
                cid = tp_res.get('clientAlgoId') or tp_res.get('clientOrderId')
                if not self._verify_order_exists(symbol, cid):
                    raise Exception("TP order verification failed.")
                
                protection_orders.append({
                    "orderId": tp_res.get('algoId') or tp_res.get('orderId'),
                    "status": "NEW",
                    "type": "TAKE_PROFIT_MARKET",
                    "role": "TP",
                    "clientOrderId": cid
                })
                audit_service.log_api_call("POST", "lead/std-tp", {"tp": rounded_tp}, tp_res)

            # 5. CONFIRMATION
            lead_trades_collection.update_one({"_id": trade_id}, {"$set": {"status": "ACTIVE", "protection_orders": protection_orders}})
            return {"trade_id": trade_id, "status": "ACTIVE", "entry_price": actual_entry_price}

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
                    lead_trades_collection.update_one({"_id": trade_id}, {"$set": {"status": "ROLLBACK_SUCCESS", "error": error_msg}})
                    error_msg += " (Rollback successful)"
                else:
                    raise Exception(f"Position still exists after rollback: {pos.position_amt}")
                    
            except Exception as rollback_err:
                error_msg += f" (ROLLBACK FAILED: {str(rollback_err)} - MANUAL INTERVENTION REQUIRED)"
                lead_trades_collection.update_one({"_id": trade_id}, {"$set": {"status": "ROLLBACK_FAILED", "error": error_msg}})
            
            audit_service.log_api_call("POST", "lead/smart-trade-FAILED", {"symbol": symbol}, error_msg, 400)
            raise HTTPException(status_code=400, detail=error_msg)

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
                formatted_algos.append({
                    "orderId": getattr(a, 'algo_id', None),
                    "symbol": getattr(a, 'symbol', None),
                    "side": getattr(a, 'side', None),
                    "type": getattr(a, 'order_type', None),
                    "origQty": getattr(a, 'quantity', 0),
                    "price": getattr(a, 'price', 0),
                    "stopPrice": getattr(a, 'trigger_price', 0),
                    "clientOrderId": getattr(a, 'client_algo_id', None),
                    "status": "NEW"
                })

            # 4. Fetch virtual setups from MongoDB
            query = {"status": "ACTIVE"}
            if symbol:
                query["symbol"] = symbol.upper()
            
            virtual_setups = list(lead_trades_collection.find(query))
            
            # 5. Format virtual setups
            formatted_setups = []
            for vs in virtual_setups:
                formatted_setups.append({
                    "id": vs["_id"],
                    "orderId": vs["_id"],
                    "symbol": vs["symbol"],
                    "side": vs["side"],
                    "type": "POSITION", # Virtual tag
                    "origQty": vs["quantity"],
                    "price": vs["entry_price"],
                    "tp": vs.get('tp'),
                    "sl": vs.get('sl'),
                    "clientOrderId": vs["_id"],
                    "smart_meta": {
                        "entry_price": vs["entry_price"],
                        "tp": vs["tp"],
                        "sl": vs["sl"],
                        "side": vs["side"],
                        "quantity": vs["quantity"],
                        "leverage": vs["leverage"]
                    }
                })

            return raw_orders + formatted_algos + formatted_setups
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Open Orders Error: {str(e)}")

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
            params = {
                "symbol": symbol.upper(),
                "side": exit_side,
                "type": "MARKET",
                "quantity": exit_qty
            }
            
            res = self.lead_client.new_order(**params)
            response = res.data()
            audit_service.log_api_call("POST", "lead/close-position", params, response)
            
            # 5. Verify Position is gone
            if not quantity: # Only verify if we intended to close the WHOLE position
                start_v = time.time()
                while time.time() - start_v < 10:
                    check_pos = self.get_active_positions(symbol)
                    if not any(p.symbol == symbol.upper() for p in check_pos):
                        print(f"✅ Panic Sell Verified: Position for {symbol} is ZERO.")
                        break
                    time.sleep(1.5)

            # 6. Mark associated MongoDB trades as CLOSED
            raw_avg = getattr(response, 'avg_price', '0') or '0'
            exit_price = float(raw_avg)
            if exit_price == 0:
                ticker = binance_client.futures_symbol_ticker(symbol=symbol.upper())
                exit_price = float(ticker.get('price', 0))

            lead_trades_collection.update_many(
                {"symbol": symbol.upper(), "status": {"$in": ["ACTIVE", "ENTRY_FILLED", "ENTRY_PLACED"]}},
                {"$set": {
                    "status": "CLOSED",
                    "exit_price": exit_price,
                    "close_time": time.time(),
                    "close_reason": "PANIC_SELL"
                }}
            )
            
            return response
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Panic Sell Failed: {str(e)}")

    def reconcile_lead_trades(self):
        """
        Background task to sync Lead/Futures trades with Binance reality.
        Runs every 30 seconds.
        """
        self._ensure_client()
        try:
            # 1. Fetch all trades that are not CLOSED or FAILED
            pending_trades = list(lead_trades_collection.find({"status": {"$in": ["ACTIVE", "ENTRY_PLACED", "ENTRY_FILLED"]}}))
            if not pending_trades:
                return

            print(f"🔄 Reconciling {len(pending_trades)} Lead trades...")
            
            for trade in pending_trades:
                tid = trade["_id"]
                symbol = trade["symbol"]
                
                # A. Handle ENTRY_PLACED: Check if the entry order filled
                if trade["status"] == "ENTRY_PLACED":
                    try:
                        order = self.std_client.futures_get_order(symbol=symbol, origClientOrderId=f"ENT_{tid}")
                        if order["status"] == "FILLED":
                            print(f"Entry Filled for {tid}. Updating state...")
                            avg_price = float(order.get("avgPrice", 0))
                            lead_trades_collection.update_one({"_id": tid}, {"$set": {"status": "ENTRY_FILLED", "entry_price": avg_price}})
                            # Next run will pick this up as ENTRY_FILLED
                    except Exception as e:
                        print(f"Error checking entry for {tid}: {e}")
                
                # B. Handle ENTRY_FILLED: Logic should already be handled by the main flow, but we can verify position
                elif trade["status"] == "ENTRY_FILLED":
                    # If it's stuck here, it means protection failed or was never placed
                    positions = self.get_active_positions(symbol)
                    pos = next((p for p in positions if p.symbol == symbol), None)
                    if not pos or float(pos.position_amt or 0) == 0:
                        # Position is gone but trade is marked FILLED? Likely manually closed or liquidated.
                        lead_trades_collection.update_one({"_id": tid}, {"$set": {"status": "CLOSED", "close_reason": "SYNC_POSITION_GONE"}})

                # C. Handle ACTIVE: Sync protection orders and detect closure
                elif trade["status"] == "ACTIVE":
                    # Check if SL or TP filled
                    res = self.lead_client.current_all_open_orders(symbol=symbol)
                    open_orders = res.data()
                    open_client_ids = [getattr(o, 'client_order_id', '') for o in open_orders]
                    
                    # If protection orders are missing, check if they were filled
                    protection_ids = [p.get('clientOrderId') for p in trade.get('protection_orders', []) if p.get('clientOrderId')]
                    missing_ids = [pid for pid in protection_ids if pid not in open_client_ids]
                    
                    if missing_ids:
                        # Something filled or was cancelled
                        print(f"Detection: Protection order(s) {missing_ids} missing for {tid}. Checking fills...")
                        
                        # Fetch recent trades for this symbol
                        history = self.std_client.futures_account_trades(symbol=symbol, limit=20)
                        fill = next((h for h in history if h.get('clientOrderId') in missing_ids), None)
                        
                        if fill:
                            print(f"Trade {tid} closed via {fill['clientOrderId']} at {fill['price']}")
                            lead_trades_collection.update_one({"_id": tid}, {"$set": {
                                "status": "CLOSED",
                                "exit_price": float(fill['price']),
                                "exit_fees": float(fill.get('commission', 0)),
                                "exit_fee_asset": fill.get('commissionAsset', 'USDT'),
                                "close_time": fill['time'] / 1000.0,
                                "close_reason": "TAKE_PROFIT" if "TP" in fill.get('clientOrderId','') else "STOP_LOSS"
                            }})
                            # Cancel remaining legs
                            self.std_client.futures_cancel_all_open_orders(symbol=symbol)
                        else:
                            # Not found in history? Check if position is actually closed
                            positions = self.get_active_positions(symbol)
                            pos = next((p for p in positions if p.symbol == symbol), None)
                            if not pos or float(pos.position_amt or 0) == 0:
                                print(f"Trade {tid} has no position and no orders. Closing in DB.")
                                lead_trades_collection.update_one({"_id": tid}, {"$set": {"status": "CLOSED", "close_reason": "RECONCILED_EMPTY"}})

        except Exception as e:
            print(f"Reconciler Lead Error: {e}")

# Singleton instance
futures_service = FuturesService()
