import requests
import json
import os
import time

API_BASE = "http://localhost:8001"
LANGGRAPH_BASE = "http://localhost:2024"

def trigger_scan():
    print("--- 1. TRIGGERING SCAN (AND WARMUP) ---")
    start = time.time()
    try:
        # Trigger scan to fire warmup
        requests.post(f"{API_BASE}/scanner/run", json={"pairs": ["BTCUSDT"], "timeframe": "1m"}, timeout=5)
        print("Scan Triggered")
    except Exception:
        pass
    print(f"Time to return: {round((time.time() - start) * 1000, 2)}ms")

def run_analysis():
    print(f"\n--- 2. RUNNING ANALYSIS (After 30s Wait) ---")
    url = f"{LANGGRAPH_BASE}/analyze"
    scenario_path = os.path.join("langgraph", "tests", "scenarios", "strong_bullish.json")
    
    with open(scenario_path, "r") as f:
        payload = json.load(f)
    
    start = time.time()
    try:
        response = requests.post(url, json=payload, timeout=120)
        end = time.time()
        
        if response.status_code == 200:
            result = response.json()
            gen_time = next((x["process_time_ms"] for x in result["audit_trail"] if x["node"] == "generator"), 0)
            total_time = round((end - start) * 1000, 2)
            print(f"Total Request Time: {total_time}ms")
            print(f"Generator Node Time: {gen_time}ms")
            
            if total_time < 10000:
                print("✅ RESULT: FAST (WARM)")
            else:
                print("⚠️ RESULT: SLOW (COLD)")
        else:
            print(f"Error: {response.status_code}")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    trigger_scan()
    print("Waiting 30 seconds for model to load into VRAM...")
    time.sleep(30)
    run_analysis()
