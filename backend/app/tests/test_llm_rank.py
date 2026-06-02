import asyncio
import os
import sys

from dotenv import load_dotenv

# Set PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load env
load_dotenv()

from app.services import llm_service


async def test_llm_ranking_mock():
    print("--- Testing AI Ranking with 20 Mock Pairs ---")

    mock_data = [
        {"pair": "BTCUSDC", "score": 8.5, "structure": "BULLISH"},
        {"pair": "ETHUSDC", "score": 7.2, "structure": "BULLISH"},
        {"pair": "SOLUSDC", "score": 9.1, "structure": "BREAKOUT"},
        {"pair": "XRPUSDC", "score": 4.5, "structure": "RANGE"},
        {"pair": "ADAUSDC", "score": 3.2, "structure": "BEARISH"},
        {"pair": "DOTUSDC", "score": 6.8, "structure": "BULLISH"},
        {"pair": "LINKUSDC", "score": 7.9, "structure": "BULLISH"},
        {"pair": "AVAXUSDC", "score": 5.5, "structure": "RANGE"},
        {"pair": "DOGEUSDC", "score": 2.1, "structure": "BEARISH"},
        {"pair": "MATICUSDC", "score": 6.2, "structure": "BULLISH"},
        {"pair": "UNIUSDC", "score": 4.8, "structure": "RANGE"},
        {"pair": "LTCUSDC", "score": 5.1, "structure": "RANGE"},
        {"pair": "SUIUSDC", "score": 8.8, "structure": "BREAKOUT"},
        {"pair": "APTUSDC", "score": 7.5, "structure": "BULLISH"},
        {"pair": "NEARUSDC", "score": 6.9, "structure": "BULLISH"},
        {"pair": "OPUSDC", "score": 5.8, "structure": "RANGE"},
        {"pair": "ARBUSDC", "score": 4.2, "structure": "BEARISH"},
        {"pair": "TIAUSDC", "score": 8.2, "structure": "BULLISH"},
        {"pair": "INJUSDC", "score": 7.7, "structure": "BULLISH"},
        {"pair": "SEIUSDC", "score": 9.4, "structure": "BREAKOUT"},
    ]

    try:
        print(f"Sending {len(mock_data)} mock pairs to LLM...")
        results = await llm_service.rank_setups(mock_data)

        print(f"\n--- AI RANKING RESULTS (Captured {len(results)} pairs) ---")
        sorted_ranks = sorted(results.values(), key=lambda x: x.get("rank", 999))

        for item in sorted_ranks:
            print(
                f"Rank #{item.get('rank')}: {item.get('pair'):<10} | Bias: {item.get('bias'):<10} | Reason: {item.get('reason')}"
            )

        if len(results) < 20:
            print(f"WARNING: Only {len(results)}/20 pairs were ranked.")
        else:
            print("SUCCESS: All 20 pairs ranked correctly.")

    except Exception as e:
        print(f"AI Ranking test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_llm_ranking_mock())
