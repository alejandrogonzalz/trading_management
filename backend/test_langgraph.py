import asyncio
import httpx
import json


async def test():
    async with httpx.AsyncClient(timeout=60.0) as http_client:
        payload = {
            "symbol": "TRUUSDT",
            "mode": "SPOT",
            "indicators": {
                "1h": {"close": 0.1, "heatmap": "NEUTRAL", "atr_ratio": 1.0}
            },
        }
        try:
            response = await http_client.post(
                "http://langgraph:2024/analyze", json=payload
            )
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}")
        except Exception as e:
            print(f"Exception: {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(test())
