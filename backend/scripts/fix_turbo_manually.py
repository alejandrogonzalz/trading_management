from motor.motor_asyncio import AsyncIOMotorClient

from app.services.binance_service import binance_client


async def fix_turbo():
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    db = client.trading_db

    tid = "SMART_1774205602"
    trade = await db.trades.find_one({"_id": tid})

    if not trade:
        print("Trade not found.")
        return

    print(f"Checking legs for {tid}...")
    for leg in trade["orders"]:
        if leg["role"] in ["TP", "SL"]:
            try:
                order = binance_client.get_order(symbol=trade["symbol"], orderId=leg["id"])
                print(f"Leg {leg['id']} ({leg['role']}) Status: {order['status']}")
                if order["status"] == "FILLED":
                    print("Found fill! Archiving...")
                    exec_qty = float(order["executedQty"])
                    exit_price = float(order["cummulativeQuoteQty"]) / exec_qty

                    # Manual call to mark closed since it's already in MANUAL_CONTROL
                    # and the standard reconciler ignores MANUAL_CONTROL
                    update_data = {
                        "status": "CLOSED",
                        "exit_price": exit_price,
                        "close_time": order.get("updateTime", 0) / 1000.0,
                        "close_reason": f"FIXED_{leg['role']}",
                    }
                    await db.trades.update_one({"_id": tid}, {"$set": update_data})
                    print(f"✅ Trade {tid} successfully archived.")
                    return
            except Exception as e:
                print(f"Error checking leg {leg['id']}: {e}")

    print("❌ No filled legs found. Trade remains in MANUAL_CONTROL.")


if __name__ == "__main__":
    # We need to run this within the app context or mock it
    # For now, I'll just provide the logic.
    pass
