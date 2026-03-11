import time
from . import binance_service
from . import trade_tracker
from .database import trades_collection

def test_reconciler_step_by_step():
    print("🔍 UNIT TEST: RECONCILER LOGIC")
    
    # 1. SETUP: Mock an ACTIVE trade in DB that doesn't actually exist on Binance
    fake_tid = f"SMART_MOCK_{int(time.time())}"
    trade_tracker._save_trades({}) # Clean start
    
    print(f"1️⃣  Mocking active trade {fake_tid} in MongoDB...")
    trade_tracker.save_trade_metadata(fake_tid, "BTCUSDT", 80000, 60000, "BUY")
    trades_collection.update_one({"_id": fake_tid}, {"$set": {"status": "ACTIVE", "quantity": 0.001}})
    
    # Verify it is active
    meta = trade_tracker.get_trade_metadata(fake_tid)
    print(f"   Trade Status: {meta['status']}")

    # 2. RUN RECONCILER
    # It should see this trade is ACTIVE in DB, but has NO orders on Binance and NO balance.
    print(f"\n2️⃣  Running Reconciler logic...")
    binance_service.reconcile_trades()
    
    # 3. VERIFY OUTCOME
    print(f"\n3️⃣  Verifying database update...")
    final_meta = trade_tracker.get_trade_metadata(fake_tid)
    if final_meta['status'] == 'CLOSED':
        print(f"✅ SUCCESS: Reconciler detected the orphaned trade and CLOSED it.")
    else:
        print(f"❌ FAILURE: Trade is still {final_meta['status']}.")

if __name__ == "__main__":
    test_reconciler_step_by_step()
