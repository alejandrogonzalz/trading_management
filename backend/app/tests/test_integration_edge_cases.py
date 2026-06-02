import time

from app.services import binance_service, trade_tracker


def run_edge_case_test():
    print("🚀 STARTING EDGE CASE INTEGRATION TEST")

    # 0. Initial Cleanup
    trade_tracker._save_trades({})
    symbol = "BTCUSDT"
    reconcile_symbol = "ETHUSDT"
    test_amount_usdt = 10.0

    try:
        # --- TEST 1: NO STOP LOSS (SINGLE LEG TP) ---
        print("\n1️⃣  Testing Smart Trade with TP Only (No SL)...")
        ticker = binance_service.binance_client.get_symbol_ticker(symbol=symbol)
        price = float(ticker["price"])
        qty = test_amount_usdt / price

        trade_res = binance_service.create_smart_trade(
            symbol=symbol, quantity=qty, buy_price=None, take_profit_price=price * 1.05, stop_loss_price=0, side="BUY"
        )

        if trade_res.get("entry", {}).get("status") == "FILLED":
            print("✅ Entry Order FILLED")

        # --- TEST 2: PANIC SELL FOR SINGLE-LEG ---
        print("\n2️⃣  Testing Panic Sell for Single-Leg Trade...")
        all_meta = trade_tracker._load_trades()
        tid = [k for k, v in all_meta.items() if v["status"] == "ACTIVE" and v["symbol"] == symbol][0]

        close_res = binance_service.market_close_position(symbol=symbol, quantity=qty, client_order_id=tid)
        if close_res.get("status") in ["FILLED", "ARCHIVED"]:
            print("✅ Panic Sell for Single-Leg SUCCESSFUL")

        # --- TEST 3: RECONCILER VALIDATION (AUTO-CLOSE ON FILL) ---
        print("\n3️⃣  Testing Reconciler (Auto-Detect Fill)...")
        ticker_eth = binance_service.binance_client.get_symbol_ticker(symbol=reconcile_symbol)
        price_eth = float(ticker_eth["price"])
        qty_eth = test_amount_usdt / price_eth

        # 3a. Create trade
        binance_service.create_smart_trade(
            symbol=reconcile_symbol,
            quantity=qty_eth,
            buy_price=None,
            take_profit_price=price_eth * 1.05,
            stop_loss_price=0,
            side="BUY",
        )
        all_meta = trade_tracker._load_trades()
        tid_2 = [k for k, v in all_meta.items() if v["status"] == "ACTIVE" and v["symbol"] == reconcile_symbol][0]
        meta_2 = all_meta[tid_2]
        print(f"✅ Created simulation trade {tid_2} (ETH)")

        # 3b. Simulate manual Binance closure (Sell everything ETH properly)
        print("🛠  Simulating manual Market Sell for ETH...")
        binance_service.binance_client.cancel_all_open_orders(symbol=reconcile_symbol, recvWindow=60000)
        qty_str = binance_service.format_quantity(reconcile_symbol, meta_2["quantity"])

        # This will create a 'FILLED' record in history for the reconciler to find
        sell_res = binance_service.binance_client.create_order(
            symbol=reconcile_symbol, side="SELL", type="MARKET", quantity=qty_str, recvWindow=60000
        )
        print(f"✅ Sim: Market Sell ID {sell_res['orderId']} filled.")

        # 3c. Run Reconciler
        print("🔍 Running Reconciler...")
        time.sleep(3)  # Wait for Binance history sync
        binance_service.reconcile_trades()

        # 3d. Check MongoDB Status
        updated_meta = trade_tracker.get_trade_metadata(tid_2)
        if updated_meta["status"] == "CLOSED":
            print("✅ Reconciler successfully detected FILL and updated MongoDB.")
        else:
            print(f"❌ Reconciler failed. Status is still: {updated_meta['status']}")

        print("\n🏁 EDGE CASE INTEGRATION TEST COMPLETED")

    except Exception as e:
        print(f"💥 CRITICAL TEST FAILURE: {e}")


if __name__ == "__main__":
    run_edge_case_test()
