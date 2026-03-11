import os
import time
import json
from . import binance_service
from . import trade_tracker

def run_integration_test():
    print("🚀 STARTING EXPANDED SMART TRADE TEST (3-BUY / 3-SELL)")
    
    # 0. Initial Cleanup
    trade_tracker._save_trades({})
    symbol = "BTCUSDT"
    test_amount_usdt = 6.00 
    
    try:
        ticker = binance_service.binance_client.get_symbol_ticker(symbol=symbol)
        price = float(ticker['price'])
        qty = test_amount_usdt / price
        
        print(f"📊 Market Price: ${price}, Target Qty: {qty}")

        # --- STAGE 1: TRIPLE CREATION ---
        print("\n1️⃣  Creating 3 Smart Trades...")
        trade_ids = []
        for i in range(1, 4):
            print(f"Creating Trade #{i}...")
            res = binance_service.create_smart_trade(
                symbol=symbol, quantity=qty, buy_price=None,
                take_profit_price=price * 1.05, stop_loss_price=price * 0.95, side="BUY"
            )
            if res.get('entry', {}).get('status') == 'FILLED':
                tid = list(trade_tracker._load_trades().keys())[-1]
                trade_ids.append(tid)
                print(f"✅ Trade {tid} Created & Protected.")
            time.sleep(1) # Small gap between orders

        # --- STAGE 2: VALIDATE INITIAL STATE ---
        all_trades = trade_tracker._load_trades()
        active_count = len([t for t in all_trades.values() if t['status'] == 'ACTIVE'])
        print(f"\n2️⃣  Validation: {active_count}/3 Trades are ACTIVE in JSON.")
        if active_count != 3:
            print("❌ Initial state validation failed.")
            return

        # --- STAGE 3: DOUBLE EXIT (TRADE 1 & 2) ---
        print("\n3️⃣  Executing Panic Sell for Trade #1 and #2...")
        for i in range(2):
            tid = trade_ids[i]
            meta = all_trades[tid]
            print(f"🔥 Closing {tid}...")
            binance_service.market_close_position(
                symbol=symbol, quantity=meta['quantity'], order_list_id=meta.get('orderListId')
            )
            
        # Verify Isolation
        all_trades = trade_tracker._load_trades()
        active = [tid for tid, t in all_trades.items() if t['status'] == 'ACTIVE']
        closed = [tid for tid, t in all_trades.items() if t['status'] == 'CLOSED']
        
        print(f"✅ Isolation Check: {len(closed)} Closed, {len(active)} Active.")
        if len(active) != 1 or active[0] != trade_ids[2]:
            print(f"❌ Isolation Failure! Remaining Active: {active}")
            return
        
        # Verify Binance OCOs for Trade 3 still exist
        open_ocos = binance_service.binance_client.get_open_oco_orders(recvWindow=60000)
        target_list_id = all_trades[trade_ids[2]].get('orderListId')
        matching_oco = [o for o in open_ocos if o['orderListId'] == target_list_id]
        
        if matching_oco:
            print(f"✅ Trade #3 OCO confirmed still live on Binance.")
        else:
            print(f"❌ Trade #3 OCO was accidentally cancelled!")
            return

        # --- STAGE 4: FINAL EXIT ---
        print("\n4️⃣  Closing Final Trade #3...")
        binance_service.market_close_position(
            symbol=symbol, 
            quantity=all_trades[trade_ids[2]]['quantity'], 
            order_list_id=all_trades[trade_ids[2]].get('orderListId')
        )
        
        all_trades = trade_tracker._load_trades()
        final_active = len([t for t in all_trades.values() if t['status'] == 'ACTIVE'])
        if final_active == 0:
            print("✅ All trades CLOSED. JSON is clean.")
        else:
            print(f"❌ Final clearance failed. {final_active} trades still active.")

        # Check for any leftovers on Binance
        final_ocos = binance_service.binance_client.get_open_oco_orders(recvWindow=60000)
        if not final_ocos:
            print("✅ Binance is clean (Zero orphaned OCOs).")
        else:
            print(f"⚠️  Leftovers found on Binance: {len(final_ocos)} OCOs remain!")

        print("\n🏁 EXPANDED INTEGRATION TEST COMPLETED SUCCESSFULLY")

    except Exception as e:
        print(f"💥 CRITICAL TEST FAILURE: {e}")

if __name__ == "__main__":
    run_integration_test()
