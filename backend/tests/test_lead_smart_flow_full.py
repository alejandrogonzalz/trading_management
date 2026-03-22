import os
import sys
import time
import asyncio
from app.services.futures_service import futures_service
from app.db.database import db

# Ensure symbols and environment are correct for test
SYMBOL = "FORTHUSDT"
QUANTITY = 150.0
LEVERAGE = 5

async def test_full_lead_lifecycle(side, tp, sl):
    print(f"\n🧪 Testing {side} Lifecycle for {SYMBOL}...")
    
    # 1. Clean Start
    futures_service.close_position(SYMBOL)
    time.sleep(2)

    # 2. Execute Smart Trade
    print("Step 1: Execute Smart Trade...")
    result = futures_service.create_smart_lead_order(
        symbol=SYMBOL, side=side, quantity=QUANTITY, 
        tp_price=tp, sl_price=sl, leverage=LEVERAGE
    )
    trade_id = result["trade_id"]
    
    # 3. Validate Entry Fields
    trade = db.lead_trades.find_one({"_id": trade_id})
    print(f"Step 2: Validate Entry Fields...")
    assert trade['status'] == 'ACTIVE'
    assert trade['entry_price'] > 0
    assert trade['quantity'] >= QUANTITY * 0.99
    assert trade['side'] == side
    assert len(trade.get('protection_orders', [])) == 2
    
    # 4. Verify on Binance (Positions & Algo Orders)
    pos = next((p for p in futures_service.get_active_positions(SYMBOL) if p.symbol == SYMBOL), None)
    assert pos is not None, "Position not found on Binance!"
    assert float(pos.position_amt) != 0

    # 5. Execute Panic Sell
    print("Step 3: Execute Panic Sell...")
    futures_service.close_position(SYMBOL)
    time.sleep(10)
    
    # 6. Validate History Fields
    closed_trade = db.lead_trades.find_one({"_id": trade_id})
    print("Step 4: Validate History Fields...")
    assert closed_trade['status'] == 'CLOSED'
    assert 'exit_price' in closed_trade and closed_trade['exit_price'] > 0
    assert 'exit_fees' in closed_trade and closed_trade['exit_fees'] >= 0
    assert closed_trade['close_reason'] == 'PANIC_SELL'
    
    print(f"✅ {side} Lifecycle Test PASSED!")

async def main():
    try:
        # Test Long
        await test_full_lead_lifecycle("BUY", 0.60, 0.30)
        # Test Short
        await test_full_lead_lifecycle("SELL", 0.30, 0.60)
        print("\n🌟 ALL TESTS PASSED.")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
