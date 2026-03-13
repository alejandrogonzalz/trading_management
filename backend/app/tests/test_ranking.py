import asyncio
import os
from dotenv import load_dotenv
import sys

# Set PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load env
load_dotenv()

from app.utils import market_utils as market_service_utils

async def test_ranking():
    print("--- Testing Top 20 Opportunity Ranking ---")
    try:
        top_20 = market_service_utils.get_top_opportunity_pairs(20)
        
        print("\nRANKED TOP 20 PAIRS:")
        for i, symbol in enumerate(top_20):
            print(f"{i+1}. {symbol}")
            
        if len(top_20) == 0:
            print("No pairs found! check volume filter or API connection.")
            
    except Exception as e:
        print(f"Ranking test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_ranking())
