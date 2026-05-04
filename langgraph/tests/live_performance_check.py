import requests
import json
import os
import time


def run_scenario(name):
    url = "http://localhost:2024/analyze"
    scenario_path = os.path.join("langgraph", "tests", "scenarios", f"{name}.json")

    with open(scenario_path, "r") as f:
        payload = json.load(f)

    print(f"\n--- SCENARIO: {name.upper()} ({payload['mode']}) ---")
    start_total = time.time()

    try:
        response = requests.post(url, json=payload, timeout=120)
        end_total = time.time()

        if response.status_code == 200:
            result = response.json()
            print(f"Run ID: {result['run_id']}")
            print(f"Confidence: {result['evaluation']['confidence']}/10")
            print(f"Issues: {len(result['issues'])} found")

            print("-" * 60)
            print(f"{'NODE NAME':<15} | {'TIME (MS)':<10} | {'STATUS'}")
            print("-" * 60)

            node_sum = 0
            for entry in result.get("audit_trail", []):
                node = entry["node"]
                timing = entry["process_time_ms"]
                node_sum += timing
                print(f"{node.upper():<15} | {timing:>10} | SUCCESS")

            print("-" * 60)
            print(f"{'SUM NODES':<15} | {round(node_sum, 2):>10}ms")
            print(f"{'TOTAL REQUEST':<15} | {round((end_total - start_total) * 1000, 2):>10}ms")

        else:
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")


if __name__ == "__main__":
    run_scenario("strong_bullish")  # Fast path (No Optimizer)
    run_scenario("risky_futures")  # Full path (With Optimizer)
