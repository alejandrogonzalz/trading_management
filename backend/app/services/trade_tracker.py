import time
from typing import Any

from app.db.database import db_session
from app.db.models import SpotTrade


def save_trade_metadata(client_order_id: str, symbol: str, tp_price: float, sl_price: float, side: str):
    """Saves or updates trade metadata in SQLite."""
    try:
        trade = db_session.query(SpotTrade).filter(SpotTrade.id == client_order_id).first()
        if not trade:
            trade = SpotTrade(
                id=client_order_id,
                symbol=symbol,
                tp=tp_price,
                sl=sl_price,
                side=side,
                timestamp=time.time(),
                status="ACTIVE",
                orders=[],
            )
            db_session.add(trade)
        else:
            trade.symbol = symbol
            trade.tp = tp_price
            trade.sl = sl_price
            trade.side = side
            trade.status = "ACTIVE"

        db_session.commit()
    except Exception as e:
        db_session.rollback()
        print(f"Error saving trade metadata: {e}")
    finally:
        db_session.remove()


def add_order_to_trade(client_order_id: str, order_id: int, order_type: str, role: str):
    """Adds a specific Binance order ID to the trade's tracking list."""
    try:
        trade = db_session.query(SpotTrade).filter(SpotTrade.id == client_order_id).first()
        if trade:
            # SQLAlchemy mutable JSON doesn't always detect changes in list.append
            # We re-assign to ensure it detects the change.
            new_orders = list(trade.orders) if trade.orders else []
            new_orders.append(
                {
                    "id": order_id,
                    "type": order_type,
                    "role": role,  # 'TP', 'SL', or 'ENTRY'
                    "status": "NEW",
                }
            )
            trade.orders = new_orders
            db_session.commit()
    except Exception as e:
        db_session.rollback()
        print(f"Error adding order to trade: {e}")
    finally:
        db_session.remove()


def get_trade_metadata(client_order_id: str) -> dict[str, Any] | None:
    """Retrieves trade metadata from SQLite."""
    try:
        trade = db_session.query(SpotTrade).filter(SpotTrade.id == client_order_id).first()
        if not trade:
            # Check by searching inside the orders list for any order_id match
            # SQLite JSON search is possible but simple iteration over recent is fine here
            # Or use a LIKE query on the JSON column for better performance
            trade = db_session.query(SpotTrade).filter(SpotTrade.orders.contains([{"id": client_order_id}])).first()
            if not trade:
                # Manual fallback if .contains doesn't work as expected with this specific JSON structure
                all_active = db_session.query(SpotTrade).filter(SpotTrade.status != "CLOSED").all()
                for t in all_active:
                    if any(str(o.get("id")) == str(client_order_id) for o in t.orders):
                        trade = t
                        break

        if trade:
            return _model_to_dict(trade)
        return None
    finally:
        db_session.remove()


def _model_to_dict(trade: SpotTrade) -> dict[str, Any]:
    return {
        "id": trade.id,
        "symbol": trade.symbol,
        "side": trade.side,
        "entry_price": trade.entry_price,
        "quantity": trade.quantity,
        "tp": trade.tp,
        "sl": trade.sl,
        "status": trade.status,
        "orders": trade.orders,
        "orderListId": trade.orderListId,
        "entry_fees": trade.entry_fees,
        "fee_asset": trade.fee_asset,
        "exit_price": trade.exit_price,
        "exit_fees": trade.exit_fees,
        "exit_fee_asset": trade.exit_fee_asset,
        "close_time": trade.close_time,
        "timestamp": trade.timestamp,
        "error_msg": trade.error_msg,
    }


def _load_trades() -> dict[str, Any]:
    try:
        trades = db_session.query(SpotTrade).all()
        return {t.id: _model_to_dict(t) for t in trades}
    finally:
        db_session.remove()


def _save_trades(trades: dict[str, Any]):
    """Compatibility function to save bulk updates."""
    try:
        for tid, data in trades.items():
            trade = db_session.query(SpotTrade).filter(SpotTrade.id == tid).first()
            if not trade:
                trade = SpotTrade(id=tid)
                db_session.add(trade)

            # Update fields from data
            for key, value in data.items():
                if hasattr(trade, key) and key != "id":
                    setattr(trade, key, value)

        db_session.commit()
    except Exception as e:
        db_session.rollback()
        print(f"Error bulk saving trades: {e}")
    finally:
        db_session.remove()
