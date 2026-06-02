from app.db.database import audit_collection, trades_collection
from app.services import binance_service


def final_cleanup():
    print("🧹 STARTING FINAL CLEANUP...")

    # 1. Clear all BTCUSDT orders on Binance
    try:
        binance_service.sync_binance_time()
        # Cancel all OCOs specifically first
        ocos = binance_service.binance_client.get_open_oco_orders(recvWindow=60000)
        for oco in ocos:
            if oco["symbol"] == "BTCUSDT":
                print(f"Cancelling OCO {oco['orderListId']}...")
                binance_service.binance_client._delete(
                    "orderList",
                    True,
                    data={"symbol": "BTCUSDT", "orderListId": oco["orderListId"], "recvWindow": 60000},
                )

        # Cancel all remaining orders
        binance_service.binance_client.cancel_all_open_orders(symbol="BTCUSDT", recvWindow=60000)
        print("✅ Binance BTCUSDT orders cleared.")
    except Exception as e:
        print(f"⚠️ Binance cleanup notice: {e}")

    # 2. Clear MongoDB
    try:
        trades_collection.delete_many({})
        audit_collection.delete_many({})
        print("✅ MongoDB trades and audit logs wiped.")
    except Exception as e:
        print(f"❌ MongoDB wipe failed: {e}")

    print("✨ TERMINAL IS NOW 100% CLEAN")


if __name__ == "__main__":
    final_cleanup()
