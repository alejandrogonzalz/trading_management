import json
import time
from typing import TypedDict, List, Dict, Any, Literal, Optional
from langgraph.graph import StateGraph, END
from agent.llm_factory import get_llm_provider


# --- State Definition ---
class TradeState(TypedDict):
    run_id: str
    symbol: str
    mode: Literal["SPOT", "FUTURES"]
    indicators: Dict[str, Any]  # Multi-timeframe: {"1h": {...}, "1d": {...}}
    original: Optional[Dict[str, Any]]
    evaluation: Optional[Dict[str, Any]]
    optimized: Optional[Dict[str, Any]]
    issues: List[str]
    needs_optimization: bool
    audit_trail: List[Dict[str, Any]]


# --- Utilities ---
def clean_json(text: str) -> str:
    """Extract and clean JSON from LLM response text."""
    if not text or not isinstance(text, str):
        return "{}"

    # Remove common markdown code block markers
    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    # Try to find JSON object or array
    import re

    # Pattern for JSON object {...}
    obj_match = re.search(r"(\{.*\})", text, re.DOTALL)
    if obj_match:
        return obj_match.group(1).strip()

    # Pattern for JSON array [...]
    arr_match = re.search(r"(\[.*\])", text, re.DOTALL)
    if arr_match:
        return arr_match.group(1).strip()

    # If no JSON structure found, return empty object
    # but keep the text as-is for error reporting
    return text.strip()


async def safe_json_parse(response_text: str, max_attempts: int = 2) -> Dict[str, Any]:
    """Safely parse JSON from LLM response with retry and validation."""
    import json
    from json import JSONDecodeError

    if not response_text or not isinstance(response_text, str):
        return {"error": "Empty response from LLM"}

    last_error = None

    for attempt in range(max_attempts):
        try:
            # Clean the text
            cleaned = clean_json(response_text)

            # Try to parse
            parsed = json.loads(cleaned)

            # Validate basic structure
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")

            return parsed

        except (JSONDecodeError, ValueError) as e:
            last_error = e

            # If we have a JSONDecodeError, try to extract JSON more aggressively
            if attempt < max_attempts - 1:
                # Try to find any JSON-like structure more aggressively
                import re

                # Look for content between curly braces that looks like JSON
                brace_match = re.search(r"\{[^{}]*\}", response_text)
                if brace_match:
                    # Try with just the matched content
                    response_text = brace_match.group(0)
                    continue

                # No improvement possible, break
                break

    # All attempts failed
    error_msg = f"Failed to parse JSON after {max_attempts} attempts: {last_error}"
    print(f"JSON parse error: {error_msg}")

    # Return a minimal valid structure
    return {
        "error": error_msg,
        "bias": "NEUTRAL",
        "entry": 0,
        "tp": 0,
        "sl": 0,
        "leverage": None,
        "reasoning": f"JSON parsing failed: {last_error}",
    }


# --- Nodes ---


async def generator_node(state: TradeState) -> Dict[str, Any]:
    """Initial LLM analysis and setup generation."""
    start_time = time.time()

    # Ensure nested keys exist in case they weren't initialized in the dict
    issues = state.get("issues", [])
    audit_trail = state.get("audit_trail", [])

    llm = get_llm_provider()

    mode_context = (
        "SPOT (Long Only, No Leverage)"
        if state["mode"] == "SPOT"
        else "FUTURES (Long/Short, Leverage 1-50x)"
    )

    system_prompt = f"""
    You are a Senior Technical Analyst for a {mode_context} trading system.
    Analyze the provided multi-timeframe indicators and generate a high-confluence trade setup.
    
    RULES:
    - MODE: {state["mode"]}
    - If SPOT: Bias MUST be LONG. Leverage MUST be null.
    - If FUTURES: Bias can be LONG or SHORT. Recommend leverage (1-50) based on volatility.
    - Output MUST be valid JSON.
    """

    user_prompt = f"""
    Symbol: {state["symbol"]}
    Indicators: {json.dumps(state["indicators"])}
    
    Return valid JSON with these exact fields:
    {{
        "bias": "LONG" or "SHORT",
        "entry": float (price number),
        "tp": float (take profit price),
        "sl": float (stop loss price), 
        "leverage": integer or null,
        "reasoning": "2 sentences max explaining the setup",
        "quality": "HIGH", "MEDIUM", or "LOW"
    }}
    
    Example JSON response:
    {{
        "bias": "LONG",
        "entry": 100.50,
        "tp": 105.25,
        "sl": 98.75,
        "leverage": 5,
        "reasoning": "Bullish breakout on 1h with strong volume support.",
        "quality": "HIGH"
    }}
    """

    original = None
    try:
        response_text = await llm.generate_setup(system_prompt, user_prompt)
        setup = await safe_json_parse(response_text)

        # Check if parsing failed
        if "error" in setup:
            issues.append(f"Generator JSON Parse Error: {setup['error']}")
            original = setup
        else:
            # Enforce SPOT constraints if LLM hallucinated
            if state["mode"] == "SPOT":
                setup["bias"] = "LONG"
                setup["leverage"] = None

            original = setup
    except Exception as e:
        issues.append(f"Generator Error: {str(e)}")
        original = {"error": f"Failed to generate initial setup: {str(e)}"}

    # Update audit trail
    duration = round((time.time() - start_time) * 1000, 2)
    audit_trail.append({"node": "generator", "process_time_ms": duration})

    return {"original": original, "issues": issues, "audit_trail": audit_trail}


def evaluator_node(state: TradeState) -> Dict[str, Any]:
    """Quant validation and risk check (No LLM)."""
    start_time = time.time()
    setup = state["original"]
    ind_multi = state["indicators"]
    issues = state.get("issues", [])
    audit_trail = state.get("audit_trail", [])

    if not setup or "error" in setup:
        # Update audit trail
        duration = round((time.time() - start_time) * 1000, 2)
        audit_trail.append({"node": "evaluator", "process_time_ms": duration})
        return {
            "audit_trail": audit_trail,
            "needs_optimization": True,
            "issues": issues + ["Setup missing or contains errors"],
            "evaluation": {
                "confidence": 0,
                "quant_confidence": 0,
                "reasoning": "Generator failed to provide a valid setup.",
            },
        }

    # 1. Multi-TF Context Extraction
    base_tf = (
        "1h"
        if "1h" in ind_multi
        else next(iter(ind_multi.keys()))
        if isinstance(ind_multi, dict) and ind_multi
        else None
    )
    ind = ind_multi.get(base_tf, {}) if base_tf else {}
    macro_tf = "1d" if isinstance(ind_multi, dict) and "1d" in ind_multi else base_tf
    macro_ind_val = ind_multi.get(macro_tf) if isinstance(ind_multi, dict) else None
    macro_ind = macro_ind_val if isinstance(macro_ind_val, dict) else {}

    # 2. Extract Key Data
    rsi = ind.get("rsi", 50)
    adx = ind.get("adx", 20)
    atr_ratio = ind.get("atr_ratio", 1.0)
    bias = setup.get("bias")
    entry = setup.get("entry", 0)
    tp = setup.get("tp", 0)
    sl = setup.get("sl", 0)
    leverage = setup.get("leverage", 1) or 1

    # 3. Mode Validation
    if state["mode"] == "SPOT" and bias == "SHORT":
        issues.append("SHORT not allowed in SPOT mode")

    # 4. Logical Target Validation
    if bias and entry and tp and sl:
        if bias == "LONG" and not (tp > entry > sl):
            issues.append("LONG targets invalid (TP must > Entry > SL)")
        if bias == "SHORT" and not (tp < entry < sl):
            issues.append("SHORT targets invalid (TP must < Entry < SL)")
    else:
        if not entry or not tp or not sl:
            issues.append("Missing price targets from generator")

    # 5. Multi-TF Indicator Confluence (Quant Confidence)
    q_score = 4  # Base score

    # Trend Confluence across all timeframes
    trend_points = 0
    total_tfs = len(ind_multi) if isinstance(ind_multi, dict) else 1

    if isinstance(ind_multi, dict):
        for tf, tf_ind in ind_multi.items():
            heatmap = tf_ind.get("heatmap", "NEUTRAL")
            if bias == "LONG":
                if heatmap == "STRONG_BULLISH":
                    trend_points += 1.5
                elif heatmap == "BULLISH":
                    trend_points += 1.0
                elif heatmap == "BEARISH":
                    trend_points -= 1.0
                elif heatmap == "STRONG_BEARISH":
                    trend_points -= 2.0
            if bias == "SHORT":
                if heatmap == "STRONG_BEARISH":
                    trend_points += 1.5
                elif heatmap == "BEARISH":
                    trend_points += 1.0
                elif heatmap == "BULLISH":
                    trend_points -= 1.0
                elif heatmap == "STRONG_BULLISH":
                    trend_points -= 2.0

    # Normalize trend points to a max of 5
    q_score += min(max(trend_points, 0), 5)

    # 5.1 Strong Macro Bonus
    if macro_ind.get("heatmap") in ["STRONG_BULLISH", "BULLISH"] and bias == "LONG":
        q_score += 1
    if macro_ind.get("heatmap") in ["STRONG_BEARISH", "BEARISH"] and bias == "SHORT":
        q_score += 1

    # 5.2 Specific Overextended/Risk Checks (on Base TF)
    if bias == "LONG" and rsi > 75:
        issues.append("RSI Overbought (>75)")
    if bias == "SHORT" and rsi < 25:
        issues.append("RSI Oversold (<25)")
    if atr_ratio > 2.5:
        issues.append("High Volatility Spike (ATR Ratio > 2.5)")

    # 6. Futures Liquidation Safety
    liq_safe = True
    risk_pct, liq_pct = 0, 0
    if state["mode"] == "FUTURES":
        # Distance to SL %
        risk_pct = abs(entry - sl) / entry * 100 if entry > 0 else 0
        # Estimated Distance to Liquidation % (Simple formula: 100/Leverage)
        liq_pct = 100 / leverage

        # We want SL to be hit BEFORE liquidation with a safety buffer (30%)
        if risk_pct >= (liq_pct * 0.7):
            issues.append(
                f"High Liq Risk: SL distance ({risk_pct:.1f}%) too close to Liq ({liq_pct:.1f}%)"
            )
            liq_safe = False

    # 7. Macro Alignment
    macro_map = {
        "STRONG_BULLISH": "BULLISH",
        "BULLISH": "BULLISH",
        "BEARISH": "BEARISH",
        "STRONG_BEARISH": "BEARISH",
    }
    macro_alignment = macro_map.get(macro_ind.get("heatmap"), "NEUTRAL")

    # 8. Results
    final_confidence = setup.get("confidence", q_score)
    # If LLM didn't provide confidence, or if quant is stronger, nudge it
    if q_score > final_confidence:
        final_confidence = q_score

    evaluation = {
        "confidence": final_confidence,
        "quant_confidence": min(max(q_score, 1), 10),
        "macro_alignment": macro_alignment,
        "liquidation_safe": liq_safe,
        "risk_pct": round(risk_pct, 2),
        "liq_pct": round(liq_pct, 2),
        "safety_margin": round(liq_pct - risk_pct, 2),
        "rr": round(abs(tp - entry) / abs(entry - sl), 2) if abs(entry - sl) > 0 else 0,
        "issues": issues,
    }

    # Update audit trail
    duration = round((time.time() - start_time) * 1000, 2)
    audit_trail.append({"node": "evaluator", "process_time_ms": duration})

    return {
        "evaluation": evaluation,
        "issues": issues,
        "needs_optimization": len(issues) > 0,
        "audit_trail": audit_trail,
    }


async def optimizer_node(state: TradeState) -> Dict[str, Any]:
    """LLM-based refinement if issues were found."""
    start_time = time.time()
    issues = state.get("issues", [])
    audit_trail = state.get("audit_trail", [])

    if not state["needs_optimization"]:
        # Update audit trail
        duration = round((time.time() - start_time) * 1000, 2)
        audit_trail.append({"node": "optimizer", "process_time_ms": duration})
        return {"audit_trail": audit_trail}

    llm = get_llm_provider()

    system_prompt = f"""You are a Risk Manager. Fix the provided trade setup based on identified technical issues.
    
    CRITICAL: You MUST return valid JSON only. Do not include any explanatory text before or after the JSON.
    The JSON must contain exactly the fields: bias, entry, tp, sl, leverage, reasoning, changes.
    
    If you cannot optimize the setup, return a valid JSON with error field explaining why."""
    user_prompt = f"""
    Symbol: {state["symbol"]}
    Mode: {state["mode"]}
    Original Setup: {json.dumps(state["original"])}
    Issues Found: {json.dumps(state["issues"])}
    Indicators: {json.dumps(state["indicators"])}
    
    Provide an OPTIMIZED setup that resolves the issues. 
    - If mode is SPOT and bias was SHORT, flip it to LONG and find a bullish entry.
    - If liquidation risk is high, reduce leverage or tighten SL.
    
    Return valid JSON with these exact fields:
    {{
        "bias": "LONG" or "SHORT",
        "entry": float (adjusted entry price),
        "tp": float (adjusted take profit),
        "sl": float (adjusted stop loss),
        "leverage": integer or null,
        "reasoning": "Brief explanation of optimization",
        "changes": ["list", "of", "changes", "made"]
    }}
    
    Example JSON response:
    {{
        "bias": "LONG",
        "entry": 100.75,
        "tp": 104.50,
        "sl": 99.25,
        "leverage": 8,
        "reasoning": "Reduced leverage from 10x to 8x to improve liquidation safety.",
        "changes": ["Reduced leverage", "Tightened stop loss"]
    }}
    """

    optimized = None
    try:
        response_text = await llm.generate_setup(system_prompt, user_prompt)
        optimized = await safe_json_parse(response_text)

        # Check if parsing failed
        if "error" in optimized:
            issues.append(f"Optimizer JSON Parse Error: {optimized['error']}")
            # Keep optimized as None to indicate failure
            optimized = None
    except Exception as e:
        issues.append(f"Optimizer Error: {str(e)}")

    # Update audit trail
    duration = round((time.time() - start_time) * 1000, 2)
    audit_trail.append({"node": "optimizer", "process_time_ms": duration})

    return {"optimized": optimized, "issues": issues, "audit_trail": audit_trail}


# --- Graph Construction ---

workflow = StateGraph(TradeState)

# Add Nodes
workflow.add_node("generator", generator_node)
workflow.add_node("evaluator", evaluator_node)
workflow.add_node("optimizer", optimizer_node)

# Set Entry Point
workflow.set_entry_point("generator")

# Add Edges
workflow.add_edge("generator", "evaluator")


# Conditional Edge from Evaluator
def should_optimize(state: TradeState):
    return "optimizer" if state["needs_optimization"] else END


workflow.add_conditional_edges("evaluator", should_optimize)
workflow.add_edge("optimizer", END)

# Compile
graph = workflow.compile()
