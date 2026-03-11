import os
import time
import json
from . import binance_service
from . import trade_tracker
from . import market_service_utils
from .database import trades_collection

def run_top_10_test():
    print("🚀 STARTING TOP 10 OPPORTUNITY INTEGRATION TEST")
    
    # 0. Initial Cleanup
    trades_collection.delete_many({})
    
    try:
        # 1. Fetch Top 10 Pairs
        print("🔍 Discovering Top 10 Opportunity Pairs...")
        top_pairs = market_service_utils.get_top_opportunity_pairs(10)
        print(f"✅ Top 10 Pairs: {top_pairs}")

        results = []
        test_amount_usdt = 12.00 # Increased to ensure OCO legs pass Notional

        for symbol in top_pairs:
            try:
                print(f"\n--- Testing Symbol: {symbol} ---")
                
                # A. Get Price
                ticker = binance_service.binance_client.get_symbol_ticker(symbol=symbol)
                price = float(ticker['price'])
                qty = test_amount_usdt / price
                
                # B. Create Smart Trade
                print(f"1️⃣  Creating Smart Trade at ${price}...")
                trade_res = binance_service.create_smart_trade(
                    symbol=symbol,
                    quantity=qty,
                    buy_price=None, # Market
                    take_profit_price=price * 1.05, 
                    stop_loss_price=price * 0.95,   
                    side="BUY"
                )
                
                status = trade_res.get('status')
                if status == 'PROTECTION_FAILED':
                    print(f"⚠️  Protection failed for {symbol}: {trade_res.get('error')}")
                
                # Get the Master ID
                all_meta = trade_tracker._load_trades()
                active_tids = [tid for tid, m in all_meta.items() if m['symbol'] == symbol and m['status'] in ['ACTIVE', 'PROTECTION_FAILED']]
                
                if not active_tids:
                    print(f"❌ No active trade found in DB for {symbol}")
                    results.append({"symbol": symbol, "status": "FAIL", "reason": "No DB record"})
                    continue
                
                tid = active_tids[0]
                meta = all_meta[tid]
                print(f"✅ Trade {tid} ({status}) in MongoDB.")

                # C. Panic Sell
                print(f"2️⃣  Executing Panic Sell for {tid}...")
                time.sleep(2)
                
                close_res = binance_service.market_close_position(
                    symbol=symbol,
                    quantity=meta['quantity'],
                    order_list_id=meta.get('orderListId'),
                    client_order_id=tid
                )
                
                if close_res.get('status') == 'FILLED' or close_res.get('status') == 'ARCHIVED' or 'orderId' in close_res:
                    print(f"✅ Panic Sell SUCCESSFUL")
                else:
                    print(f"❌ Panic Sell FAILED: {close_res}")

                # D. Final Validation
                time.sleep(1)
                final_meta = trade_tracker.get_trade_metadata(tid)
                if final_meta['status'] == 'CLOSED':
                    pnl = ((final_meta.get('exit_price', 0) - final_meta.get('entry_price', 0)) / final_meta.get('entry_price', 1)) * 100
                    print(f"✅ Trade {tid} CLOSED. P&L: {pnl:.2f}%")
                    results.append({"symbol": symbol, "status": "PASS", "pnl": pnl})
                else:
                    print(f"❌ Trade {tid} still {final_meta['status']}!")
                    results.append({"symbol": symbol, "status": "FAIL", "reason": "Status not CLOSED"})

            except Exception as e:
                print(f"⚠️ Error during {symbol} loop: {e}")
                results.append({"symbol": symbol, "status": "ERROR", "reason": str(e)})
            
            time.sleep(1)

        print("\n" + "="*40)
        print("🏁 TOP 10 TEST SUMMARY")
        print("="*40)
        for r in results:
            pnl_str = f" | P&L: {r['pnl']:.2f}%" if "pnl" in r else ""
            reason_str = f" | Reason: {r['reason']}" if "reason" in r else ""
            print(f"{r['symbol']}: {r['status']}{pnl_str}{reason_str}")
        
        passed = len([r for r in results if r['status'] == 'PASS'])
        print(f"\n📊 Total: {passed}/{len(top_pairs)} Passed.")

    except Exception as e:
        print(f"💥 CRITICAL TEST ERROR: {e}")

if __name__ == "__main__":
    run_top_10_test()
