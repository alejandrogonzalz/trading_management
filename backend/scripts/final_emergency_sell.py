import time
from app.services import binance_service
from app.db.database import trades_collection, audit_collection

def emergency_sell_and_wipe():
    print("🚨 STARTING FINAL EMERGENCY SELL & WIPE...")
    symbol = "BTCUSDT"
    
    # 1. Cancel everything first
    try:
        binance_service.sync_binance_time()
        ocos = binance_service.binance_client.get_open_oco_orders(recvWindow=60000)
        for oco in ocos:
            if oco['symbol'] == symbol:
                print(f"Cancelling OCO {oco['orderListId']}...")
                binance_service.binance_client._delete("orderList", True, data={"symbol": symbol, "orderListId": oco['orderListId'], "recvWindow": 60000})
        
        binance_service.binance_client.cancel_all_open_orders(symbol=symbol, recvWindow=60000)
        print("✅ Orders cancelled.")
    except Exception as e:
        print(f"⚠️ Cancel step: {e}")

    time.sleep(2)

    # 2. Market Sell EVERYTHING BTC
    try:
        info = binance_service.binance_client.get_account(recvWindow=60000)
        asset = "BTC"
        balance = next((float(b['free']) for b in info['balances'] if b['asset'] == asset), 0)
        
        if balance > 0:
            qty_str = binance_service.format_quantity(symbol, balance)
            if float(qty_str) > 0:
                print(f"Market Selling {qty_str} BTC...")
                sell_res = binance_service.binance_client.create_order(
                    symbol=symbol, side="SELL", type="MARKET", quantity=qty_str, recvWindow=60000
                )
                print("✅ Market Sell Complete.")
            else:
                print("ℹ️ Balance is too small to sell (dust).")
        else:
            print("ℹ️ No BTC balance found.")
    except Exception as e:
        print(f"❌ Market Sell failed: {e}")

    # 3. Wipe Database
    trades_collection.delete_many({})
    audit_collection.delete_many({})
    print("✅ MongoDB Wiped.")
    
    # 4. Remove old JSON if exists
    import os
    if os.path.exists("smart_trades_metadata.json"):
        os.remove("smart_trades_metadata.json")
        print("✅ Legacy JSON removed.")

    print("✨ SYSTEM IS NOW CLEAN AND READY.")

if __name__ == "__main__":
    emergency_sell_and_wipe()
