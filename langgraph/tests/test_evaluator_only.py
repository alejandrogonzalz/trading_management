import json
import os
import sys

# Ensure agent can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.graph import evaluator_node


def load_scenario(name):
    path = os.path.join(os.path.dirname(__file__), "scenarios", f"{name}.json")
    with open(path, "r") as f:
        return json.load(f)


def test_evaluator():
    print("--- STARTING MULTI-TF EVALUATOR VALIDATION ---")

    # CASE 1: Strong Bullish (Multi-TF Scenario)
    data = load_scenario("strong_bullish")
    state = {
        "symbol": data["symbol"],
        "mode": data["mode"],
        "indicators": data["indicators"],
        "original": {"bias": "LONG", "entry": 65000, "tp": 72000, "sl": 63000, "leverage": 10, "quality": "HIGH"},
    }
    res = evaluator_node(state)
    print(
        f"\n[Case: Strong Bullish Multi-TF] Confidence: {res['evaluation']['confidence']}, Quant: {res['evaluation']['quant_confidence']}"
    )
    assert res["evaluation"]["confidence"] >= 7
    assert res["evaluation"]["macro_alignment"] == "BULLISH"  # From the 1d tf in our scenario

    # CASE 2: High Volatility (Forcing 1h data)
    data = load_scenario("high_volatility_futures")
    # Wrap legacy single TF data into multi-TF structure for compatibility
    multi_tf_data = {"1h": data["indicators"], "1d": data["indicators"]}
    state = {
        "symbol": data["symbol"],
        "mode": data["mode"],
        "indicators": multi_tf_data,
        "original": {"bias": "LONG", "entry": 18.50, "tp": 22.0, "sl": 18.0, "leverage": 40, "quality": "MEDIUM"},
    }
    res = evaluator_node(state)
    print(
        f"\n[Case: High Volatility] Confidence: {res['evaluation']['confidence']}, Issues: {res['evaluation']['issues']}"
    )
    assert any("High Liq Risk" in i for i in res["evaluation"]["issues"])
    assert res["evaluation"]["liquidation_safe"] is False

    # CASE 3: Spot Short Restriction
    data = load_scenario("overbought_spot")
    multi_tf_data = {"1h": data["indicators"], "1d": data["indicators"]}
    state = {
        "symbol": data["symbol"],
        "mode": "SPOT",
        "indicators": multi_tf_data,
        "original": {"bias": "SHORT", "entry": 3500, "tp": 3000, "sl": 3600, "leverage": None, "quality": "LOW"},
    }
    res = evaluator_node(state)
    print(f"\n[Case: Spot Short] Issues: {res['evaluation']['issues']}")
    assert any("SHORT not allowed in SPOT mode" in i for i in res["evaluation"]["issues"])

    print("\n--- ALL MULTI-TF EVALUATOR TESTS PASSED ---")


if __name__ == "__main__":
    try:
        test_evaluator()
    except Exception as e:
        print(f"Test Failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
