import math
import time
from decimal import ROUND_FLOOR, Decimal

from binance.client import Client
from binance.enums import *
from fastapi import HTTPException

from app.core.config import settings
from app.services import audit_service, trade_tracker

# Initialize Binance Client
binance_client = Client(
    settings.BINANCE_API_KEY.strip().replace('"', "").replace("'", ""),
    settings.BINANCE_API_SECRET.strip().replace('"', "").replace("'", ""),
)


def sync_binance_time():
    try:
        server_time = binance_client.get_server_time()["serverTime"]
        local_time = int(time.time() * 1000)
        binance_client.timestamp_offset = server_time - local_time
    except Exception as e:
        print(f"Error syncing time: {e}")


def round_step_size(quantity, step_size):
    """
    Rounds a quantity to the nearest step size using high-precision Decimal.
    """
    if step_size <= 0:
        return quantity
    q = Decimal(str(quantity))
    s = Decimal(str(step_size))
    # Round down to ensure we don't exceed balance
    rounded = (q / s).quantize(Decimal("1"), rounding=ROUND_FLOOR) * s
    return float(rounded.normalize())


sync_binance_time()


def get_balances():
    """Returns current wallet balances."""
    try:
        sync_binance_time()
        res = binance_client.get_account(recvWindow=60000)
        balances = [b for b in res["balances"] if b["asset"] in ["USDC", "USDT"] or float(b["free"]) > 0]
        return balances
    except Exception as e:
        print(f"Error fetching balances: {e}")
        return []


def get_open_orders():
    try:
        sync_binance_time()
        raw_orders = binance_client.get_open_orders(recvWindow=60000)
        if not isinstance(raw_orders, list):
            return []
        relevant_orders = [o for o in raw_orders if "USDC" in o["symbol"] or "USDT" in o["symbol"]]

        all_meta = trade_tracker._load_trades()
        processed_order_ids = set()
        final_list = []

        for tid, meta in all_meta.items():
            if meta.get("status") == "CLOSED":
                continue
            tracked_ids = [o["id"] for o in meta.get("orders", [])]
            list_id = meta.get("orderListId")
            trade_legs = [
                o
                for o in relevant_orders
                if o["orderId"] in tracked_ids or (list_id and o.get("orderListId") == list_id)
            ]

            if trade_legs:
                for leg in trade_legs:
                    leg["smart_meta"] = meta
                    leg["clientOrderId"] = tid
                    final_list.append(leg)
                    processed_order_ids.add(leg["orderId"])
            elif meta.get("status") in [
                "ACTIVE",
                "MANUAL_CONTROL",
                "PROTECTION_FAILED",
            ]:
                quote = "USDC" if meta["symbol"].endswith("USDC") else "USDT"
                asset = meta["symbol"].replace(quote, "")
                try:
                    info = binance_client.get_account(recvWindow=60000)
                    balances = {b["asset"]: float(b["free"]) + float(b["locked"]) for b in info["balances"]}
                    asset_balance = balances.get(asset, 0)
                    orig_qty = meta.get("quantity", 0)
                    if asset_balance >= (orig_qty * 0.1) and asset_balance > 0.000001:
                        formatted_qty = float(format_quantity(meta["symbol"], orig_qty))
                        final_list.append(
                            {
                                "symbol": meta["symbol"],
                                "orderId": f"POS_{tid}",
                                "clientOrderId": tid,
                                "price": meta.get("entry_price", 0),
                                "origQty": formatted_qty,
                                "status": "FILLED/HOLDING" if meta.get("status") == "ACTIVE" else meta.get("status"),
                                "side": meta["side"],
                                "type": "POSITION",
                                "smart_meta": meta,
                            }
                        )
                except:
                    pass

        for o in relevant_orders:
            if o["orderId"] not in processed_order_ids:
                final_list.append(o)
        return final_list
    except Exception:
        return []


def reconcile_trades():
    """ID-STRICT RECONCILER: Queries specific Order IDs for truth."""
    try:
        sync_binance_time()
        all_meta = trade_tracker._load_trades()
        # We only reconcile ACTIVE trades.
        # MANUAL_CONTROL trades need manual intervention or a script to force close.
        active_trades = {tid: m for tid, m in all_meta.items() if m.get("status") == "ACTIVE"}
        if not active_trades:
            return

        open_orders_res = binance_client.get_open_orders(recvWindow=60000)
        open_orders = open_orders_res if isinstance(open_orders_res, list) else []
        open_ids = [o["orderId"] for o in open_orders]

        for tid, meta in active_trades.items():
            strategy_legs = [o for o in meta.get("orders", []) if o["role"] in ["TP", "SL"]]
            if not strategy_legs:
                continue

            # A trade is "missing legs" if ANY of its protection orders are not in the open orders list
            missing_legs = [leg for leg in strategy_legs if leg["id"] not in open_ids]

            if missing_legs:
                print(f"🔍 Reconciler: Detected missing order for {tid} ({meta['symbol']}). Checking leg history...")

                final_exit_price = 0
                final_exit_fees = 0
                final_close_time = None
                final_fee_asset = None
                trade_was_filled = False

                # Check ALL legs. If ANY leg is FILLED, the whole trade is CLOSED.
                for leg in strategy_legs:
                    try:
                        order_info = binance_client.get_order(
                            symbol=meta["symbol"], orderId=leg["id"], recvWindow=60000
                        )
                        status = order_info.get("status")

                        if status == "FILLED":
                            trade_was_filled = True
                            exec_qty = float(order_info["executedQty"])
                            final_close_time = order_info.get("updateTime", 0) / 1000.0

                            if exec_qty > 0:
                                # Weighted average exit price
                                final_exit_price = float(order_info["cummulativeQuoteQty"]) / exec_qty
                            # Fetch exact trades for fees with startTime filter for reliability
                            try:
                                # Fetch trades from 1 minute before trade timestamp to now
                                start_ts = int(meta.get("timestamp", time.time() - 3600) * 1000) - 60000
                                my_trades = binance_client.get_my_trades(
                                    symbol=meta["symbol"], startTime=start_ts, limit=100
                                )
                                leg_trades = [t for t in my_trades if t["orderId"] == leg["id"]]

                                if leg_trades:
                                    print(f"  > Found {len(leg_trades)} fill legs for exit order {leg['id']}")
                                    final_exit_fees = sum(float(t["commission"]) for t in leg_trades)
                                    final_fee_asset = leg_trades[0].get("commissionAsset")

                                    # Normalized exit price from fills
                                    total_qty = sum(float(t["qty"]) for t in leg_trades)
                                    total_quote = sum(float(t["qty"]) * float(t["price"]) for t in leg_trades)
                                    if total_qty > 0:
                                        final_exit_price = total_quote / total_qty
                            except Exception as e:
                                print(f"  > Error fetching spot exit fees: {e}")
                            break  # Found the fill, no need to check other legs
                    except Exception as e:
                        print(f"  ! Error checking leg {leg['id']}: {e}")
                        continue

                if trade_was_filled:
                    print(f"  ✅ Leg Fill Found! Archiving trade {tid}")
                    _mark_trade_closed(
                        meta["symbol"],
                        meta.get("orderListId"),
                        tid,
                        meta["quantity"],
                        final_exit_price,
                        final_exit_fees,
                        final_close_time,
                        final_fee_asset,
                    )
                else:
                    # If legs are missing but NONE were filled, it means they were CANCELLED manually
                    print(f"  ⚠️ No fill found for missing legs of {tid}. Moving to MANUAL_CONTROL.")
                    from app.db.database import db_session
                    from app.db.models import SpotTrade

                    try:
                        db_session.query(SpotTrade).filter(SpotTrade.id == tid).update({"status": "MANUAL_CONTROL"})
                        db_session.commit()
                    except:
                        db_session.rollback()
                    finally:
                        db_session.remove()
    except Exception as e:
        print(f"Reconciler Error: {e}")


def create_smart_trade(
    symbol: str,
    quantity: float,
    buy_price: float | None,
    take_profit_price: float,
    stop_loss_price: float,
    side: str = "BUY",
    mode: str = "SPOT",
):
    formatted_qty = format_quantity(symbol, quantity)
    try:
        sync_binance_time()
        client_order_id = f"SMART_{int(time.time())}"

        # 1. Entry Order
        entry_params = {
            "symbol": symbol,
            "side": SIDE_BUY if side == "BUY" else SIDE_SELL,
            "type": ORDER_TYPE_MARKET if not buy_price else ORDER_TYPE_LIMIT,
            "quantity": formatted_qty,
            "recvWindow": 60000,
            "newClientOrderId": client_order_id,
        }
        if buy_price:
            entry_params["price"] = format_price(symbol, buy_price)
            entry_params["timeInForce"] = TIME_IN_FORCE_GTC

        entry_order = binance_client.create_order(**entry_params)
        audit_service.log_api_call("POST", "order", entry_params, entry_order)

        # 2. Extract Fill Data
        actual_fill_price, total_entry_fees, fee_asset = 0, 0, ""
        actual_received_qty = float(formatted_qty)
        if entry_order.get("status") == "FILLED":
            fills = entry_order.get("fills", [])
            if fills:
                total_qty = sum(float(f["qty"]) for f in fills)
                total_cost = sum(float(f["qty"]) * float(f["price"]) for f in fills)
                total_entry_fees = sum(float(f["commission"]) for f in fills)
                fee_asset = fills[0].get("commissionAsset", "")
                actual_fill_price = total_cost / total_qty
                asset_bought = symbol.replace("USDT", "").replace("USDC", "")
                if fee_asset == asset_bought:
                    actual_received_qty = total_qty - total_entry_fees
                # Format quantity to exchange step size
                actual_received_qty = float(format_quantity(symbol, actual_received_qty))

        save_price = actual_fill_price if actual_fill_price > 0 else (buy_price or 0)
        trade_tracker.save_trade_metadata(client_order_id, symbol, take_profit_price, stop_loss_price, side)
        all_trades = trade_tracker._load_trades()
        if client_order_id in all_trades:
            all_trades[client_order_id].update(
                {
                    "entry_price": save_price,
                    "quantity": actual_received_qty,
                    "status": "ACTIVE",
                    "entry_fees": total_entry_fees,
                    "fee_asset": fee_asset,
                }
            )
            trade_tracker._save_trades(all_trades)

        trade_tracker.add_order_to_trade(client_order_id, entry_order["orderId"], entry_order["type"], "ENTRY")

        # 3. Exit Strategy
        oco_qty = format_quantity(symbol, actual_received_qty * 0.999)
        has_tp, has_sl = take_profit_price > 0, stop_loss_price > 0

        try:
            if has_tp and has_sl:
                oco_params = {
                    "symbol": symbol,
                    "side": SIDE_SELL if side == "BUY" else SIDE_BUY,
                    "quantity": oco_qty,
                    "listClientOrderId": f"LIST_{client_order_id}",
                    "aboveType": "LIMIT_MAKER",
                    "belowType": "STOP_LOSS_LIMIT",
                    "abovePrice": format_price(symbol, take_profit_price),
                    "belowStopPrice": format_price(symbol, stop_loss_price),
                    "belowPrice": format_price(symbol, stop_loss_price),
                    "belowTimeInForce": TIME_IN_FORCE_GTC,
                    "recvWindow": 60000,
                }
                try:
                    oco_res = binance_client.create_oco_order(**oco_params)
                    audit_service.log_api_call("POST", "orderList/oco", oco_params, oco_res)
                    if oco_res and "orderReports" in oco_res:
                        for report in oco_res["orderReports"]:
                            role = "TP" if report["type"] == "LIMIT_MAKER" else "SL"
                            trade_tracker.add_order_to_trade(client_order_id, report["orderId"], report["type"], role)
                        all_trades = trade_tracker._load_trades()
                        if client_order_id in all_trades:
                            all_trades[client_order_id]["orderListId"] = oco_res["orderListId"]
                            trade_tracker._save_trades(all_trades)
                except Exception as e:
                    if "aboveType" in str(e):
                        oco_res = binance_client._post("orderList/oco", True, data=oco_params)
                    else:
                        raise e
            elif has_tp:
                exit_order = binance_client.create_order(
                    symbol=symbol,
                    side=SIDE_SELL if side == "BUY" else SIDE_BUY,
                    type=ORDER_TYPE_LIMIT,
                    timeInForce=TIME_IN_FORCE_GTC,
                    quantity=oco_qty,
                    price=format_price(symbol, take_profit_price),
                    newClientOrderId=f"TP_{client_order_id}",
                    recvWindow=60000,
                )
                trade_tracker.add_order_to_trade(client_order_id, exit_order["orderId"], "LIMIT", "TP")
            elif has_sl:
                exit_order = binance_client.create_order(
                    symbol=symbol,
                    side=SIDE_SELL if side == "BUY" else SIDE_BUY,
                    type=ORDER_TYPE_STOP_LOSS_LIMIT,
                    timeInForce=TIME_IN_FORCE_GTC,
                    quantity=oco_qty,
                    stopPrice=format_price(symbol, stop_loss_price),
                    price=format_price(symbol, stop_loss_price),
                    newClientOrderId=f"SL_{client_order_id}",
                    recvWindow=60000,
                )
                trade_tracker.add_order_to_trade(client_order_id, exit_order["orderId"], "STOP_LOSS_LIMIT", "SL")
        except Exception as e:
            from app.db.database import db_session
            from app.db.models import SpotTrade

            try:
                db_session.query(SpotTrade).filter(SpotTrade.id == client_order_id).update(
                    {"status": "PROTECTION_FAILED", "error_msg": str(e)}
                )
                db_session.commit()
            except:
                db_session.rollback()
            finally:
                db_session.remove()
            return {
                "entry": entry_order,
                "status": "PROTECTION_FAILED",
                "error": str(e),
            }

        return {"entry": entry_order, "status": "ACTIVE"}
    except Exception as e:
        audit_service.log_api_call("POST", "smart-trade-CRITICAL-FAILED", {"symbol": symbol}, str(e), 400)
        raise HTTPException(status_code=400, detail=str(e))


def market_close_position(
    symbol: str,
    quantity: float | None = None,
    order_list_id: int | None = None,
    client_order_id: str | None = None,
):
    try:
        sync_binance_time()
        if order_list_id:
            try:
                binance_client._delete(
                    "orderList",
                    True,
                    data={
                        "symbol": symbol,
                        "orderListId": order_list_id,
                        "recvWindow": 60000,
                    },
                )
            except:
                pass
        if client_order_id:
            meta = trade_tracker.get_trade_metadata(client_order_id)
            if meta:
                for o in meta.get("orders", []):
                    if o["role"] in ["TP", "SL"]:
                        try:
                            binance_client.cancel_order(symbol=symbol, orderId=o["id"], recvWindow=60000)
                        except:
                            pass
        time.sleep(1.5)
        info = binance_client.get_account(recvWindow=60000)
        asset = symbol.replace("USDC", "").replace("USDT", "")
        real_balance = next((float(b["free"]) for b in info["balances"] if b["asset"] == asset), 0)
        sell_qty = float(quantity) if (quantity and float(quantity) > 0) else real_balance
        if sell_qty > real_balance:
            sell_qty = real_balance
        price_info = binance_client.get_symbol_ticker(symbol=symbol)
        if (sell_qty * float(price_info["price"])) < 5.1:
            _mark_trade_closed(
                symbol,
                order_list_id,
                client_order_id,
                quantity,
                float(price_info["price"]),
                0,
            )
            return {"status": "ARCHIVED"}
        formatted_qty_str = format_quantity(symbol, sell_qty)
        sell_order = binance_client.create_order(
            symbol=symbol,
            side=SIDE_SELL,
            type=ORDER_TYPE_MARKET,
            quantity=formatted_qty_str,
            recvWindow=60000,
        )
        exit_price, total_exit_fees, actual_close_time, exit_fee_asset = (
            0,
            0,
            None,
            None,
        )
        fills = sell_order.get("fills", [])
        actual_close_time = sell_order.get("transactTime", 0) / 1000.0  # Use transactTime for market orders

        if fills:
            total_qty = sum(float(f["qty"]) for f in fills)
            total_cost = sum(float(f["qty"]) * float(f["price"]) for f in fills)
            total_exit_fees = sum(float(f["commission"]) for f in fills)
            exit_fee_asset = fills[0].get("commissionAsset")
            exit_price = total_cost / total_qty
        else:
            exit_price = float(price_info["price"])

        _mark_trade_closed(
            symbol,
            order_list_id,
            client_order_id,
            float(formatted_qty_str),
            exit_price,
            total_exit_fees,
            actual_close_time,
            exit_fee_asset,
        )
        return sell_order
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def _mark_trade_closed(
    symbol: str,
    order_list_id: int | None = None,
    client_order_id: str | None = None,
    quantity: float = 0,
    exit_price: float = 0,
    exit_fees: float = 0,
    close_time: float | None = None,
    exit_fee_asset: str | None = None,
):
    all_trades = trade_tracker._load_trades()
    final_close_time = close_time if close_time else time.time()

    for tid, tmeta in all_trades.items():
        if tmeta["symbol"] == symbol and tmeta.get("status") != "CLOSED":
            match = False
            if order_list_id and tmeta.get("orderListId") == order_list_id:
                match = True
            elif client_order_id and tid == client_order_id:
                match = True
            elif (
                not order_list_id
                and not client_order_id
                and abs(tmeta.get("quantity", 0) - quantity) < (quantity * 0.1)
            ):
                match = True

            if match:
                from app.db.database import db_session
                from app.db.models import SpotTrade

                try:
                    update_data = {
                        "status": "CLOSED",
                        "exit_price": exit_price,
                        "exit_fees": exit_fees,
                        "close_time": final_close_time,
                    }
                    if exit_fee_asset:
                        update_data["exit_fee_asset"] = exit_fee_asset

                    db_session.query(SpotTrade).filter(SpotTrade.id == tid).update(update_data)
                    db_session.commit()
                except:
                    db_session.rollback()
                finally:
                    db_session.remove()
                print(f"✅ Trade {tid} archived successfully with close_time {final_close_time}")
                break  # Only close one match per call


def format_quantity(symbol: str, quantity: float) -> str:
    try:
        info = binance_client.get_symbol_info(symbol)
        lot_size_filter = next(f for f in info["filters"] if f["filterType"] == "LOT_SIZE")
        step_size_str = lot_size_filter["stepSize"]
        step_size = float(step_size_str)
        # Calculate precision from string representation
        if "." in step_size_str:
            precision = len(step_size_str.split(".")[-1].rstrip("0"))
        else:
            precision = 0
        factor = 10**precision
        return "{:0.{}f}".format(math.floor(quantity * factor) / factor, precision)
    except:
        return str(quantity)


def format_price(symbol: str, price: float) -> str:
    try:
        info = binance_client.get_symbol_info(symbol)
        price_filter = next(f for f in info["filters"] if f["filterType"] == "PRICE_FILTER")
        tick_size_str = price_filter["tickSize"]
        tick_size = float(tick_size_str)
        # Calculate precision from string representation
        if "." in tick_size_str:
            precision = len(tick_size_str.split(".")[-1].rstrip("0"))
        else:
            precision = 0
        formatted = "{:0.{}f}".format(price, precision)
        # Ensure formatted price is not zero for non-zero input
        if float(formatted) == 0 and price != 0:
            # Use tick size as minimum increment
            if price > 0:
                formatted = "{:0.{}f}".format(tick_size, precision)
            else:
                formatted = "{:0.{}f}".format(-tick_size, precision)
        return formatted
    except:
        return f"{price:0.2f}"


def get_symbols():
    try:
        info = binance_client.get_exchange_info()
        return sorted(
            [
                s["symbol"]
                for s in info["symbols"]
                if (s["symbol"].endswith("USDC") or s["symbol"].endswith("USDT")) and s["status"] == "TRADING"
            ]
        )
    except Exception as e:
        print(f"Error fetching symbols: {e}")
        return ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def get_trade_history(symbol: str | None = None):
    try:
        sync_binance_time()
        active_symbol = symbol if symbol else "BTCUSDT"
        orders = binance_client.get_all_orders(symbol=active_symbol, limit=50, recvWindow=60000)
        history = [o for o in orders if o["status"] in ["FILLED", "CANCELED", "REJECTED"]]
        all_meta = trade_tracker._load_trades()
        for o in history:
            meta = trade_tracker.get_trade_metadata(o.get("clientOrderId") or o.get("listClientOrderId", ""))
            if meta:
                o["smart_meta"] = meta
        return sorted(history, key=lambda x: x["updateTime"], reverse=True)
    except:
        return []


def get_smart_history():
    all_trades = trade_tracker._load_trades()
    closed = [dict(t, id=tid) for tid, t in all_trades.items() if t.get("status") == "CLOSED"]
    return sorted(closed, key=lambda x: x.get("close_time", 0), reverse=True)


def cancel_order(symbol: str, order_id: int):
    try:
        sync_binance_time()
        return binance_client.cancel_order(symbol=symbol, orderId=order_id, recvWindow=60000)
    except:
        return None


def get_24hr_tickers():
    """Returns 24hr ticker price change statistics for all symbols."""
    try:
        return binance_client.get_ticker()
    except Exception as e:
        print(f"Error fetching 24hr tickers: {e}")
        return []
