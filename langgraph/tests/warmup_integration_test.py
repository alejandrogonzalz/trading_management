import json
import os
import time

import requests

API_BASE = "http://localhost:8001"
LANGGRAPH_BASE = "http://localhost:2024"


def trigger_scan():
    print("--- 1. TRIGGERING SCAN (AND WARMUP) ---")
    start = time.time()
    try:
        # We trigger a lightweight scan just to fire the warmup logic
        # Using a small timeframe to make the scan itself fast
        res = requests.post(f"{API_BASE}/scanner/run", json={"pairs": ["BTCUSDT"], "timeframe": "1m"}, timeout=5)
        print(f"Scan Triggered: {res.status_code}")
    except Exception as e:
        print(f"Scan Trigger Warning (Expected if async): {e}")
    print(f"Time to return: {round((time.time() - start) * 1000, 2)}ms")


def run_analysis(run_num):
    print(f"\n--- {run_num}. RUNNING ANALYSIS ---")
    url = f"{LANGGRAPH_BASE}/analyze"
    scenario_path = os.path.join("langgraph", "tests", "scenarios", "strong_bullish.json")

    with open(scenario_path) as f:
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
    # 1. Trigger the scan (which should trigger warmup)
    trigger_scan()

    # 2. Wait a moment to simulate the time a user spends looking at the scanner results
    # Realistically, a scan takes 2-3 seconds, and the user takes 2-3 seconds to click analysis.
    print("Waiting 5 seconds (Simulating scan duration + user reaction)...")
    time.sleep(5)

    # 3. Run Analysis #1 (Should be fast if warmup worked)
    run_analysis(2)

    # 4. Run Analysis #2 (Should definitely be fast)
    run_analysis(3)
