import httpx
import json
import asyncio


async def test_live_langgraph():
    url = "http://localhost:2024/analyze"
    payload = {
        "symbol": "BTCUSDT",
        "mode": "FUTURES",
        "indicators": {
            "1h": {"price": 65000, "heatmap": "STRONG_BULLISH", "rsi": 65, "adx": 35, "atr_ratio": 1.2, "bb_pos": 0.8},
            "1d": {"price": 65000, "heatmap": "BULLISH", "rsi": 58, "adx": 22, "atr_ratio": 1.0, "bb_pos": 0.6},
        },
    }

    print(f"--- Sending Live LangGraph Request for {payload['symbol']} ---")
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(url, json=payload)
            if response.status_code == 200:
                print("\n[SUCCESS] Response from Actual LLM (Qwen 2.5 via LangGraph):")
                print(json.dumps(response.json(), indent=2))
            else:
                print(f"\n[ERROR] Status {response.status_code}: {response.text}")
        except Exception as e:
            print(f"\n[CONNECTION FAILED] Is the langgraph container running? {e}")


if __name__ == "__main__":
    asyncio.run(test_live_langgraph())
