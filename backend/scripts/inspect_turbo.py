import asyncio
import datetime
from pprint import pprint

from motor.motor_asyncio import AsyncIOMotorClient

# Connect to MongoDB
MONGO_URL = "mongodb://localhost:27017"
client = AsyncIOMotorClient(MONGO_URL)
db = client.trading_db


async def inspect():
    print("--- DEEP INSPECTION TURBOUSDT (ID: SMART_1774205602) ---")

    trade = await db.trades.find_one({"_id": "SMART_1774205602"})
    if trade:
        print("\n[TRADE DOCUMENT]")
        pprint(trade)

        # Check audit logs for this specific orderListId
        olid = trade.get("orderListId")
        print(f"\n[AUDIT LOGS FOR OCO ID: {olid}]")
        cursor = db.audit_logs.find({"response_body": {"$regex": str(olid)}}).sort("timestamp", -1)
        async for doc in cursor:
            dt = datetime.datetime.fromtimestamp(doc.get("timestamp"))
            print(f"{dt} | {doc.get('method')} {doc.get('url')} | Status: {doc.get('status_code')}")
            # If it's a GET to orderList, print the status
            if olid and str(olid) in doc.get("url", "") and doc.get("method") == "GET":
                try:
                    resp = doc.get("response_body")
                    if isinstance(resp, str):
                        import json

                        resp = json.loads(resp)
                    print(f"  > Binance Status: {resp.get('listOrderStatus')} | {resp.get('listClientOrderId')}")
                except:
                    pass


if __name__ == "__main__":
    asyncio.run(inspect())
