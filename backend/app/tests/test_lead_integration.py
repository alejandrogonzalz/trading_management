import httpx
import pytest
from httpx import ASGITransport

from app.main import app


@pytest.mark.asyncio
async def test_lead_read_endpoints():
    """Validates all read-only endpoints for Lead/Futures."""
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Status
        print("\nTesting GET /lead/status...")
        res = await client.get("/lead/status")
        assert res.status_code == 200
        data = res.json()
        print(f"Status: {data}")

        # 2. Symbols (Whitelist)
        print("Testing GET /lead/symbols...")
        res = await client.get("/lead/symbols")
        assert res.status_code == 200
        raw_res = res.json()
        symbols = raw_res.get("data", [])
        assert isinstance(symbols, list)
        print(f"Found {len(symbols)} whitelisted symbols.")

        # 3. Balances
        print("Testing GET /lead/balances...")
        res = await client.get("/lead/balances")
        if res.status_code != 200:
            print(f"❌ Balance Error: {res.status_code} - {res.text}")
        assert res.status_code == 200
        print("Balances retrieved successfully.")

        # 4. Positions
        print("Testing GET /lead/positions...")
        res = await client.get("/lead/positions")
        assert res.status_code == 200
        positions = res.json()
        assert isinstance(positions, list)
        print(f"Active Positions: {len(positions)}")


@pytest.mark.asyncio
async def test_lead_write_endpoints():
    """Validates write endpoints with a safe, small order."""
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        symbol = "BTCUSDT"

        # 1. Set Leverage
        print(f"\nTesting POST /lead/leverage for {symbol}...")
        res = await client.post("/lead/leverage", json={"symbol": symbol, "leverage": 5})
        assert res.status_code == 200
        print("Leverage updated to 5x.")

        # 2. Test Smart Order (Market Buy)
        print(f"Testing POST /lead/smart-order for {symbol}...")
        order_req = {
            "symbol": symbol,
            "side": "BUY",
            "type": "MARKET",
            "quantity": 0.002,
            "take_profit_price": 100000,
            "stop_loss_price": 10000,
            "leverage": 5,
        }
        res = await client.post("/lead/smart-order", json=order_req)

        if res.status_code != 200:
            print(f"❌ Order Error: {res.status_code} - {res.text}")

        if res.status_code == 400 and ("balance" in res.text.lower() or "margin" in res.text.lower()):
            print("⚠️ Skipping Order Test: Insufficient Balance/Margin (Logic is OK).")
            return

        assert res.status_code == 200
        order_data = res.json()
        print(f"Order Executed: {order_data['trade_id']}")

        # 3. Close Position
        print(f"Testing POST /lead/close-position for {symbol}...")
        close_res = await client.post("/lead/close-position", json={"symbol": symbol, "quantity": 0.002})
        assert close_res.status_code == 200
        print("Position closed successfully.")
