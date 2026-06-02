import asyncio
import json
import os
import sys

from dotenv import load_dotenv

# Set PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load env
load_dotenv()

from app.services import llm_service


async def test_deep_analysis():
    print("--- Testing AI Deep Analysis ---")

    # Mock data for a single pair across multiple timeframes
    mock_pair_data = {
        "symbol": "BTCUSDC",
        "indicators": {
            "15m": {"price": 69500, "heatmap": "BULLISH", "rsi": 62, "adx": 28, "macd_hist": 5.2, "bb_pos": 0.7},
            "1h": {
                "price": 69500,
                "heatmap": "STRONG_BULLISH",
                "rsi": 58,
                "adx": 32,
                "macd_hist": 12.5,
                "bb_pos": 0.65,
            },
            "4h": {"price": 69500, "heatmap": "NEUTRAL", "rsi": 52, "adx": 18, "macd_hist": -2.1, "bb_pos": 0.5},
            "1d": {"price": 69500, "heatmap": "BEARISH", "rsi": 42, "adx": 22, "macd_hist": -45.0, "bb_pos": 0.3},
        },
    }

    try:
        print(f"Sending multi-TF data for {mock_pair_data['symbol']} to AI...")
        result = await llm_service.analyze_row(mock_pair_data)

        print("\n--- AI DEEP ANALYSIS RESULT ---")
        print(json.dumps(result, indent=2))

        if "trade_setup" in result:
            print("\nSUCCESS: AI generated a structured trade setup.")
        else:
            print("\nWARNING: Result missing 'trade_setup' field.")

    except Exception as e:
        print(f"Deep Analysis test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_deep_analysis())
