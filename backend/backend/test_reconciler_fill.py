import os
import time
import json
from . import binance_service
from . import trade_tracker

def run_fill_path_test():
    print("🚀 STARTING RECONCILER FILL-PATH TEST")
    
    # 0. Initial Cleanup
    trade_tracker._save_trades({})
    symbol = "ETHUSDT" 
    test_amount_usdt = 10.0
    
    try:
        # 1. Get current price
        ticker = binance_service.binance_client.get_symbol_ticker(symbol=symbol)
        price = float(ticker['price'])
        qty = test_amount_usdt / price
        
        # 2. CREATE TRADE WITH INSTANT-FILL TP
        # We set TP slightly BELOW market to ensure immediate execution by Binance
        print(f"\n1️⃣  Creating Smart Trade with Instant-Fill TP at ${price * 0.999}...")
        trade_res = binance_service.create_smart_trade(
            symbol=symbol,
            quantity=qty,
            buy_price=None, 
            take_profit_price=price * 0.999, # Sell immediately
            stop_loss_price=0, 
            side="BUY"
        )
        
        all_meta = trade_tracker._load_trades()
        tid = list(all_meta.keys())[0]
        print(f"✅ Trade {tid} created. Waiting for Binance to process fill...")
        
        # Give Binance 3 seconds to move the order from 'Open' to 'History'
        time.sleep(3)

        # 3. RUN RECONCILER
        print("\n2️⃣  Running Reconciler logic...")
        binance_service.reconcile_trades()
        
        # 4. VERIFY TRUTH
        print("\n3️⃣  Verifying MongoDB Result...")
        final_meta = trade_tracker.get_trade_metadata(tid)
        
        if final_meta['status'] == 'CLOSED':
            print(f"✅ SUCCESS: Reconciler detected the FILL ID and CLOSED the trade.")
            print(f"📊 Final Exit Price: ${final_meta.get('exit_price')}")
            print(f"💰 Total Fees Captured: ${final_meta.get('exit_fees')}")
        else:
            print(f"❌ FAILURE: Reconciler missed the fill. Status is still: {final_meta['status']}")
            
            # Debug: Check why it didn't fill
            order_id = final_meta['orders'][1]['id'] # The TP leg
            debug_info = binance_service.binance_client.get_order(symbol=symbol, orderId=order_id)
            print(f"📝 Debug Order Status on Binance: {debug_info['status']}")

        print("\n🏁 FILL-PATH TEST COMPLETED")

    except Exception as e:
        print(f"💥 CRITICAL TEST FAILURE: {e}")

if __name__ == "__main__":
    run_fill_path_test()
