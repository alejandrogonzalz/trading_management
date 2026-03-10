import asyncio
import json
import os
from dotenv import load_dotenv

# Set PYTHONPATH to include the current directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load env before imports
load_dotenv()

from backend import scanner_service

async def test_full_scan():
    print("Starting Test Scan for top pairs...")
    test_pairs = ["BTCUSDC", "ETHUSDC", "SOLUSDC", "BNBUSDC", "XRPUSDC"]
    
    try:
        # We need to wrap the blocking call if it's not async
        # scanner_service.run_scan is currently synchronous (uses binance-connector)
        results = scanner_service.run_scan(test_pairs)
        
        print("\n--- SCAN RESULTS ---")
        print(f"{'Pair':<10} | {'Price':<10} | {'Score':<6} | {'RSI':<6} | {'ADX':<6} | {'VolR':<6} | {'Struct':<6}")
        print("-" * 70)
        
        for r in results:
            print(f"{r['pair']:<10} | {r.get('price', 0):<10.2f} | {r.get('score', 0):<6.2f} | {r.get('rsi', 0):<6.2f} | {r.get('adx', 0):<6.2f} | {r.get('volume_ratio', 0):<6.2f} | {r.get('structure', 'N/A'):<6}")
        
        print("\nDetailed Heatmap for BTCUSDC:")
        btc_result = next((r for r in results if r['pair'] == "BTCUSDC"), None)
        if btc_result:
            print(json.dumps(btc_result['heatmap_multi'], indent=2))

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_full_scan())
