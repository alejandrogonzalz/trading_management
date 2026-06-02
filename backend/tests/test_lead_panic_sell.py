import asyncio
import os
import sys
import time

# Setup path to import app
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from app.db.database import db
from app.services.futures_service import futures_service


async def run_panic_test():
    symbol = "FORTHUSDT"
    quantity = 150.0
    leverage = 10

    print(f"🧨 Starting Panic Sell Integration Test for {symbol}")

    try:
        # 1. Ensure clean start
        print("Cleaning up existing positions/orders...")
        futures_service.close_position(symbol)
        time.sleep(2)

        # 2. Place a Smart Trade (Entry + SL + TP)
        print("Placing test trade with protection...")
        # Wide targets to ensure they don't fill during test
        res = futures_service.create_smart_lead_order(
            symbol=symbol, side="BUY", quantity=quantity, tp_price=0.80, sl_price=0.20, leverage=leverage
        )
        trade_id = res["trade_id"]
        print(f"Trade {trade_id} is ACTIVE.")
        time.sleep(2)

        # 3. Execute Panic Sell
        print(f"\n🔥 EXECUTING PANIC SELL for {symbol}...")
        close_res = futures_service.close_position(symbol)
        print(f"Panic Sell API Response: {close_res}")

        # 4. Verification
        print("\n🧐 Verifying Results...")

        # A. Position check
        positions = futures_service.get_active_positions(symbol)
        pos_exists = any(p.symbol == symbol.upper() for p in positions)

        # B. Orders check
        orders = futures_service.get_open_orders(symbol)
        # Filter for real orders (not the virtual smart meta)
        real_orders = [o for o in orders if o.get("type") != "POSITION"]

        # C. DB check
        trade = db.lead_trades.find_one({"_id": trade_id})

        print(f"Position exists? {pos_exists}")
        print(f"Open orders remaining? {len(real_orders)}")
        print(f"DB Status: {trade['status']}")
        print(f"DB Close Reason: {trade.get('close_reason')}")

        if not pos_exists and len(real_orders) == 0 and trade["status"] == "CLOSED":
            print("\n🌟 PANIC SELL TEST PASSED: Account is flat and cleaned.")
        else:
            print("\n❌ PANIC SELL TEST FAILED: Residual risk detected.")

    except Exception as e:
        print(f"\n❌ TEST ERROR: {str(e)}")


if __name__ == "__main__":
    asyncio.run(run_panic_test())
