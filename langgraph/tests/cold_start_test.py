import requests
import json
import os
import time


def run_test(run_num):
    url = "http://localhost:2024/analyze"
    scenario_path = os.path.join("langgraph", "tests", "scenarios", "strong_bullish.json")

    with open(scenario_path, "r") as f:
        payload = json.load(f)

    print(f"\n--- RUN #{run_num}: STRONG_BULLISH ---")
    start = time.time()

    try:
        response = requests.post(url, json=payload, timeout=120)
        end = time.time()

        if response.status_code == 200:
            result = response.json()
            gen_time = next((x["process_time_ms"] for x in result["audit_trail"] if x["node"] == "generator"), 0)
            print(f"Total Time: {round((end - start) * 1000, 2)}ms")
            print(f"Generator Node: {gen_time}ms")
        else:
            print(f"Error: {response.status_code}")
    except Exception as e:
        print(f"Failed: {e}")


if __name__ == "__main__":
    run_test(1)
    print("Waiting 2 seconds...")
    time.sleep(2)
    run_test(2)
