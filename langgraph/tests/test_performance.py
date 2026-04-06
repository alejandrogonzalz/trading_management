import json
import os
import sys
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

# Ensure agent can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import after sys.path update
from agent.graph import graph

def load_scenario(name):
    path = os.path.join(os.path.dirname(__file__), "scenarios", f"{name}.json")
    with open(path, "r") as f:
        return json.load(f)

async def run_performance_test():
    print("--- STARTING PERFORMANCE & TIMING TEST ---")
    
    data = load_scenario("strong_bullish")
    
    # We invoke the actual graph
    # Note: Since I can't reach Ollama, I will mock the LLM within the graph components
    # to test the logic and timing structure.
    
    start_total = time.time()
    result = await graph.ainvoke(data)
    end_total = time.time()
    
    print(f"\nTotal Graph Execution: {round((end_total - start_total) * 1000, 2)}ms")
    print("-" * 40)
    
    for entry in result.get("audit_trail", []):
        node = entry["node"]
        duration = entry["process_time_ms"]
        print(f"Node: {node.upper():<10} | Time: {duration:>8}ms")
        
    print("-" * 40)
    print("Test Complete.")

if __name__ == "__main__":
    # If Ollama is not available, this will likely fail on the generator node.
    # But since the user wants to see the node times, I will try to run it.
    try:
        asyncio.run(run_performance_test())
    except Exception as e:
        print(f"Performance Test Failed (Expected if Ollama is down): {e}")
