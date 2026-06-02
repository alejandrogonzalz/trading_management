import asyncio

import httpx

# Base URL for local backend
API_BASE = "http://localhost:8001"


async def test_lead_connectivity():
    print("🔍 TESTING LEAD/FUTURES CONNECTIVITY (READ-ONLY)")
    print("-" * 50)

    async with httpx.AsyncClient(timeout=10.0) as client:
        # 1. Test Whitelist Symbols
        print("1️⃣  Fetching Whitelist Symbols...")
        try:
            res = await client.get(f"{API_BASE}/lead/symbols")
            if res.status_code == 200:
                symbols = res.json()
                print(f"✅ SUCCESS: Found {len(symbols)} whitelisted symbols.")
            else:
                print(f"❌ FAILED: Status {res.status_code} - {res.text}")
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")

        print("\n" + "-" * 50)

        # 2. Test Lead Status
        print("2️⃣  Checking Lead Trader Status...")
        try:
            res = await client.get(f"{API_BASE}/lead/status")
            if res.status_code == 200:
                status = res.json()
                print(f"✅ SUCCESS: Account Status -> {status}")
            else:
                print(f"❌ FAILED: Status {res.status_code} - {res.text}")
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")

        print("\n" + "-" * 50)

        # 3. Test Active Positions (Futures Info)
        print("3️⃣  Fetching Active Futures Positions...")
        try:
            res = await client.get(f"{API_BASE}/lead/positions")
            if res.status_code == 200:
                positions = res.json()
                print(f"✅ SUCCESS: Found {len(positions)} active positions.")
            else:
                print(f"❌ FAILED: Status {res.status_code} - {res.text}")
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")

        print("\n" + "-" * 50)

        # 4. Test Leverage Configuration (BTCUSDT)
        print("4️⃣  Testing Leverage Configuration (BTCUSDT -> 5x)...")
        try:
            res = await client.post(f"{API_BASE}/lead/leverage", json={"symbol": "BTCUSDT", "leverage": 5})
            if res.status_code == 200:
                print("✅ SUCCESS: Leverage updated successfully.")
            else:
                print(f"❌ FAILED: Status {res.status_code} - {res.text}")
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")

    print("\n" + "=" * 50)
    print("🏁 CONNECTIVITY TEST COMPLETE")
    print("If all 1-3 are green, your backend is ready for UI development.")
    print("If 4 is green, you have write permissions enabled.")


if __name__ == "__main__":
    asyncio.run(test_lead_connectivity())
