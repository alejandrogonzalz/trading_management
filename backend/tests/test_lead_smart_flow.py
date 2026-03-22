import os
import sys
import time
import asyncio
from typing import Dict, Any

# Setup path to import app
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.services.futures_service import futures_service
from app.db.database import db

async def run_test():
    symbol = "FORTHUSDT"
    # Small quantity (~$7 at $0.05 price)
    quantity = 150.0 
    # FORTH is around 0.483
    # BUY SETUP
    tp = 0.650
    sl = 0.350
    leverage = 10
    
    print(f"🚀 Starting Bulletproof Smart Trade Test for {symbol}")
    print(f"Qty: {quantity} | TP: {tp} | SL: {sl} | Leverage: {leverage}")
    
    try:
        # 1. Check current position to ensure clean start
        pos = futures_service.get_active_positions(symbol)
        if pos:
            print(f"⚠️ Warning: Existing position found for {symbol}. Closing first...")
            futures_service.close_position(symbol)
            time.sleep(2)

        # 2. Execute Smart Trade
        result = futures_service.create_smart_lead_order(
            symbol=symbol,
            side="BUY",
            quantity=quantity,
            tp_price=tp,
            sl_price=sl,
            leverage=leverage
        )
        
        print("\n✅ SMART TRADE EXECUTION SUCCESS!")
        print(f"Result: {result}")
        
        # 3. Verify in DB
        trade_id = result["trade_id"]
        trade = db.lead_trades.find_one({"_id": trade_id})
        print(f"\n📊 DB Record for {trade_id}:")
        print(f"Status: {trade['status']}")
        print(f"Entry Price: {trade['entry_price']}")
        print(f"Protection Orders: {len(trade.get('protection_orders', []))}")
        
        # 4. Verify on Binance
        open_orders = futures_service.get_open_orders(symbol)
        sl_found = any(o.get('role') == 'SL' or 'SL_' in str(o.get('clientOrderId','')) for o in open_orders)
        tp_found = any(o.get('role') == 'TP' or 'TP_' in str(o.get('clientOrderId','')) for o in open_orders)
        
        print(f"\n📡 Binance Verification:")
        print(f"SL Found: {sl_found}")
        print(f"TP Found: {tp_found}")
        
        if sl_found and tp_found:
            print("\n🌟 TEST PASSED: Full Lifecycle Verified.")
            print("KEEPING ORDERS OPEN FOR UI INSPECTION.")
        else:
            print("\n❌ TEST FAILED: Protection orders missing on Binance.")

    except Exception as e:
        print(f"\n❌ TEST ERROR: {str(e)}")
        # Check DB for failure reason
        last_trade = list(db.lead_trades.find({"symbol": symbol}).sort("timestamp", -1).limit(1))
        if last_trade:
            print(f"Last DB Status: {last_trade[0]['status']}")
            print(f"Last DB Error: {last_trade[0].get('error')}")

if __name__ == "__main__":
    asyncio.run(run_test())
