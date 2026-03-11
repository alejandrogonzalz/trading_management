import time
import os
import json
from typing import Dict, Any, Optional, List
from .database import trades_collection

def save_trade_metadata(client_order_id: str, symbol: str, tp_price: float, sl_price: float, side: str):
    """Saves or updates trade metadata in MongoDB."""
    data = {
        "symbol": symbol,
        "tp": tp_price,
        "sl": sl_price,
        "side": side,
        "timestamp": time.time(),
        "status": "ACTIVE",
        "orders": [] # Initialize empty orders list for ID-strict tracking
    }
    trades_collection.update_one(
        {"_id": client_order_id},
        {"$set": data},
        upsert=True
    )

def add_order_to_trade(client_order_id: str, order_id: int, order_type: str, role: str):
    """Adds a specific Binance order ID to the trade's tracking list."""
    trades_collection.update_one(
        {"_id": client_order_id},
        {"$push": {"orders": {
            "id": order_id,
            "type": order_type,
            "role": role, # 'TP', 'SL', or 'ENTRY'
            "status": "NEW"
        }}}
    )

def get_trade_metadata(client_order_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves trade metadata from MongoDB."""
    res = trades_collection.find_one({"_id": client_order_id})
    if res:
        res['client_order_id'] = res['_id']
        return res
    
    # Check by searching inside the orders list for any order_id match
    res = trades_collection.find_one({"orders.id": client_order_id})
    if res:
        return res
            
    return None

def _load_trades() -> Dict[str, Any]:
    cursor = trades_collection.find({})
    trades = {}
    for doc in cursor:
        tid = doc.pop('_id')
        trades[tid] = doc
    return trades

def _save_trades(trades: Dict[str, Any]):
    for tid, data in trades.items():
        trades_collection.update_one(
            {"_id": tid},
            {"$set": data},
            upsert=True
        )
