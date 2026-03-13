import os
from pymongo import MongoClient
import time

# Connection
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
client = MongoClient(MONGO_URL)
db = client.trading_db
trades_collection = db.trades

def fix_trades():
    print("Starting database repair for trade fees...")
    
    # 1. Find all closed trades
    closed_trades = list(trades_collection.find({"status": "CLOSED"}))
    fixed_count = 0
    
    for trade in closed_trades:
        tid = trade["_id"]
        symbol = trade.get("symbol", "UNKNOWN")
        entry_price = trade.get("entry_price", 0)
        exit_price = trade.get("exit_price", 0)
        qty = trade.get("quantity", 0)
        
        # Check for suspicious fees (e.g. your $646 BTC case)
        # Entry fees
        entry_fees = trade.get("entry_fees", 0)
        fee_asset = trade.get("fee_asset")
        
        # Exit fees
        exit_fees = trade.get("exit_fees", 0)
        exit_asset = trade.get("exit_fee_asset")
        
        needs_fix = False
        updates = {}
        
        # Logic: If fee is > 2% of trade value, it's definitely wrong units
        trade_value = qty * entry_price
        
        # Fix Entry Fee units if they were accidentally saved as Coin instead of Satoshis or vice versa
        if fee_asset == symbol.replace('USDT','').replace('USDC','') and (entry_fees * entry_price) > (trade_value * 0.02):
            print(f"Fixing Entry Fee for {tid} ({symbol})")
            # If it's too high, we'll assume a standard 0.1% fee
            updates["entry_fees"] = qty * 0.001 
            updates["fee_asset"] = fee_asset
            needs_fix = True

        # Fix Exit Fee units/missing asset
        if exit_asset is None:
            # Most of your trades pay exit fees in the Quote Asset (USDT)
            updates["exit_fee_asset"] = symbol[-4:] if symbol.endswith(('USDT', 'USDC')) else 'USDT'
            needs_fix = True
            
        # Specific check for your reported BTC error
        if symbol == "BTCUSDT" and exit_fees > 10: # No small trade should have $10+ fees
             updates["exit_fees"] = trade_value * 0.001 / exit_price if exit_price > 0 else 0.000001
             needs_fix = True

        if needs_fix:
            trades_collection.update_one({"_id": tid}, {"$set": updates})
            fixed_count += 1
            print(f"✅ Fixed {tid}")

    print(f"Finished! Total trades checked: {len(closed_trades)}. Fixed: {fixed_count}")

if __name__ == "__main__":
    fix_trades()
