#!/usr/bin/env python3
"""
Fix corrupted PEPEUSDT trade with PROTECTION_FAILED status.
Options:
1. Update status to MANUAL_CONTROL (user manages manually)
2. Attempt to add protection orders (if price formatting fixed)
3. Close position via market sell
"""

import json
import sqlite3
from datetime import datetime


def list_corrupted_trades():
    conn = sqlite3.connect("data/trading.db")
    cursor = conn.cursor()

    # Get all trades with issues
    cursor.execute("""
        SELECT id, symbol, side, entry_price, quantity, tp, sl, status, 
               error_msg, orders, timestamp
        FROM trades 
        WHERE status IN ('PROTECTION_FAILED', 'MANUAL_CONTROL')
        ORDER BY timestamp DESC
    """)

    rows = cursor.fetchall()
    if not rows:
        print("No corrupted trades found.")
        return []

    print("\n=== CORRUPTED TRADES ===")
    trades = []
    for i, row in enumerate(rows):
        (
            id_val,
            symbol,
            side,
            entry_price,
            quantity,
            tp,
            sl,
            status,
            error_msg,
            orders_json,
            timestamp,
        ) = row
        orders = json.loads(orders_json) if orders_json else []
        print(f"\n{i + 1}. {id_val} - {symbol} - {side}")
        print(f"   Status: {status}")
        print(f"   Entry: ${entry_price:.8f}, Qty: {quantity:.2f}")
        print(f"   TP: ${tp:.8f}, SL: ${sl:.8f}")
        print(f"   Error: {error_msg}")
        print(f"   Orders: {len(orders)} (Roles: {[o.get('role') for o in orders]})")
        print(f"   Time: {datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
        trades.append(
            {
                "id": id_val,
                "symbol": symbol,
                "side": side,
                "entry_price": entry_price,
                "quantity": quantity,
                "tp": tp,
                "sl": sl,
                "status": status,
                "error_msg": error_msg,
                "orders": orders,
                "timestamp": timestamp,
            }
        )

    cursor.close()
    conn.close()
    return trades


def update_trade_status(trade_id, new_status, new_error=None):
    conn = sqlite3.connect("data/trading.db")
    cursor = conn.cursor()

    update_sql = "UPDATE trades SET status = ? WHERE id = ?"
    params = [new_status, trade_id]

    if new_error is not None:
        update_sql = "UPDATE trades SET status = ?, error_msg = ? WHERE id = ?"
        params = [new_status, new_error, trade_id]

    cursor.execute(update_sql, params)
    conn.commit()

    print(f"Updated trade {trade_id} to status '{new_status}'")
    cursor.close()
    conn.close()


def add_missing_orders_to_trade(trade_id, tp_price, sl_price):
    """Attempt to add missing TP/SL orders to a trade that only has ENTRY."""
    conn = sqlite3.connect("data/trading.db")
    cursor = conn.cursor()

    # Get trade details
    cursor.execute("SELECT orders FROM trades WHERE id = ?", (trade_id,))
    row = cursor.fetchone()
    if not row:
        print(f"Trade {trade_id} not found")
        return False

    orders = json.loads(row[0]) if row[0] else []
    # Check if already has TP/SL
    has_tp = any(o.get("role") == "TP" for o in orders)
    has_sl = any(o.get("role") == "SL" for o in orders)

    if has_tp and has_sl:
        print(f"Trade {trade_id} already has TP and SL orders")
        return False

    # For demo purposes - in reality would call Binance API
    print(f"Would add TP at ${tp_price:.8f} and SL at ${sl_price:.8f}")
    print("NOTE: This requires actual Binance API calls - implement if needed")

    cursor.close()
    conn.close()
    return False


def market_close_trade(trade_id):
    """Market close the position (call the existing API endpoint)."""
    import os

    import requests
    from dotenv import load_dotenv

    load_dotenv()
    API_BASE = os.getenv("API_BASE", "http://localhost:8001")

    # Get trade details first
    conn = sqlite3.connect("data/trading.db")
    cursor = conn.cursor()
    cursor.execute("SELECT symbol, quantity FROM trades WHERE id = ?", (trade_id,))
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if not row:
        print(f"Trade {trade_id} not found")
        return False

    symbol, quantity = row
    print(f"Market closing {quantity} {symbol} for trade {trade_id}")

    try:
        # Call the existing market-close endpoint
        response = requests.post(
            f"{API_BASE}/trades/market-close",
            json={"symbol": symbol, "quantity": quantity},
            timeout=10,
        )
        if response.status_code == 200:
            print(f"Successfully closed position: {response.json()}")
            return True
        else:
            print(f"Failed to close: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"Error closing position: {e}")
        return False


def main():
    trades = list_corrupted_trades()
    if not trades:
        return

    print("\n=== OPTIONS ===")
    print("1. Update status to MANUAL_CONTROL (recommended)")
    print("2. Attempt to add missing protection orders")
    print("3. Market close position")
    print("4. Exit")

    try:
        choice = input("\nSelect option (1-4): ").strip()
        if choice == "4":
            return

        if choice in ["1", "2", "3"]:
            trade_idx = input(f"Select trade number (1-{len(trades)}): ").strip()
            try:
                idx = int(trade_idx) - 1
                if idx < 0 or idx >= len(trades):
                    print("Invalid trade number")
                    return

                trade = trades[idx]
                print(f"\nSelected: {trade['id']} - {trade['symbol']}")

                if choice == "1":
                    update_trade_status(trade["id"], "MANUAL_CONTROL")
                    print("Trade set to MANUAL_CONTROL. You can now manage it manually in the UI.")

                elif choice == "2":
                    # Re-calculate TP/SL based on original values or current price
                    tp = trade["tp"]
                    sl = trade["sl"]
                    if tp <= 0 or sl <= 0:
                        print("TP or SL is zero, cannot add protection")
                    else:
                        add_missing_orders_to_trade(trade["id"], tp, sl)

                elif choice == "3":
                    confirm = input(f"Market close {trade['quantity']} {trade['symbol']}? (yes/no): ")
                    if confirm.lower() == "yes":
                        market_close_trade(trade["id"])
                    else:
                        print("Cancelled")

            except ValueError:
                print("Invalid input")

    except KeyboardInterrupt:
        print("\nCancelled")


if __name__ == "__main__":
    main()
