import pytest
import json
import os
from agent.graph import graph


def load_scenario(name):
    path = os.path.join(os.path.dirname(__file__), "scenarios", f"{name}.json")
    with open(path, "r") as f:
        data = json.load(f)

    # Normalize indicator data to ensure it's multi-timeframe (Dict[str, Dict])
    if "indicators" in data and isinstance(data["indicators"], dict):
        indicators = data["indicators"]
        # Check if it's already multi-timeframe by seeing if values are dicts
        # Common timeframes: 1m, 5m, 15m, 1h, 4h, 1d, 1w, 1M
        is_multi = any(isinstance(v, dict) for v in indicators.values())
        if not is_multi and indicators:
            data["indicators"] = {"1h": indicators}

    return data


@pytest.mark.asyncio
async def test_structure_and_sanity():
    """Validates basic output structure and numerical sanity."""
    data = load_scenario("strong_bullish")
    result = await graph.ainvoke(data)

    # 📦 A. Output Structure
    assert "symbol" in result
    assert "original" in result
    assert "evaluation" in result
    # Assert that 'evaluation' is a dictionary before checking keys
    assert isinstance(result["evaluation"], dict)
    assert "confidence" in result["evaluation"]

    # 🧮 B. Numerical Sanity
    setup = result.get("original", {})
    assert isinstance(setup, dict)  # Ensure setup is a dict
    assert setup.get("entry", 0.0) > 0
    assert setup.get("tp", 0.0) > 0
    assert setup.get("sl", 0.0) > 0

    # 📈 C. Direction Consistency
    bias = setup.get("bias", "Neutral")
    entry = setup.get("entry", 0.0)
    tp = setup.get("tp", 0.0)
    sl = setup.get("sl", 0.0)

    if bias == "LONG":
        assert tp > entry > sl
    elif bias == "SHORT":
        assert tp < entry < sl


@pytest.mark.asyncio
async def test_futures_liquidation_safety():
    """Validates that SL is reached before liquidation with buffer."""
    data = load_scenario("high_volatility_futures")
    result = await graph.ainvoke(data)

    assert "evaluation" in result
    eval_data = result["evaluation"]
    assert isinstance(eval_data, dict)  # Ensure evaluation is a dict

    if "risk_pct" in eval_data and "liq_pct" in eval_data:
        # 💣 Liquidation Safety: risk_pct < liq_pct * 0.7
        assert eval_data["risk_pct"] < eval_data["liq_pct"] * 0.7
        assert eval_data["safety_margin"] > 0

    # Check leverage consistency
    original_setup = result.get("original", {})
    assert isinstance(original_setup, dict)  # Ensure original is a dict
    assert original_setup.get("leverage") is not None  # Check if leverage exists
    assert 1 <= original_setup.get("leverage", 1) <= 50


@pytest.mark.asyncio
async def test_spot_restrictions():
    """Validates that SPOT mode only allows LONG and no leverage."""
    data = load_scenario("overbought_spot")
    result = await graph.ainvoke(data)

    assert result["mode"] == "SPOT"
    assert result["original"].get("leverage") is None  # SPOT mode should not have leverage
    assert result["original"].get("bias", "Neutral") == "LONG"


@pytest.mark.asyncio
async def test_optimizer_improvement():
    """Validates that if issues exist, an optimized setup is produced."""
    # Using range_market which is likely to trigger issues (Low ADX/Trend)
    data = load_scenario("range_market")
    result = await graph.ainvoke(data)

    # If issues were detected, an optimized setup should be produced
    if len(result.get("issues", [])) > 0:
        assert result.get("optimized") is not None
        assert isinstance(result["optimized"], dict)
        # Confidence might still be low but optimized setup must exist
        assert "entry" in result["optimized"]


@pytest.mark.asyncio
async def test_confidence_scaling():
    """Validates confidence varies by scenario quality."""
    # 📈 Case 1: Strong Trend
    res_strong = await graph.ainvoke(load_scenario("strong_bullish"))

    # 📉 Case 2: Range Market
    res_range = await graph.ainvoke(load_scenario("range_market"))

    # Range should have lower confidence than Strong Trend
    assert res_strong.get("evaluation", {}).get("confidence", 0) >= res_range.get("evaluation", {}).get("confidence", 0)

    # ⚖️ Range: 1-10
    assert 1 <= res_strong.get("evaluation", {}).get("confidence", 0) <= 10
    assert 1 <= res_range.get("evaluation", {}).get("confidence", 0) <= 10


@pytest.mark.asyncio
async def test_determinism():
    """Validates that quant confidence is deterministic for same input."""
    data = load_scenario("strong_bullish")
    res1 = await graph.ainvoke(data)
    res2 = await graph.ainvoke(data)

    assert res1["evaluation"]["quant_confidence"] == res2["evaluation"]["quant_confidence"]
