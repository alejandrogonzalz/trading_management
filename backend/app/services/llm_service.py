import os
import json
import re
import asyncio
import httpx
from ollama import AsyncClient
from typing import List, Dict, Any, Optional
from app.core.config import settings
from app.services import scanner_service

OLLAMA_BASE_URL = settings.OLLAMA_BASE_URL
LANGGRAPH_URL = settings.LANGGRAPH_URL
LLM_MODEL = settings.LLM_MODEL

client = AsyncClient(host=OLLAMA_BASE_URL)


async def get_deep_langgraph_analysis(
    symbol: str, indicators: Dict[str, Any], mode: str = "SPOT"
) -> Dict[str, Any]:
    """
    Routes analysis to the LangGraph container for deep multi-node validation.
    """
    try:
        # Limit indicator data size to avoid oversized payloads
        # Keep only essential fields for each timeframe
        essential_keys = {
            "close",
            "price",
            "heatmap",
            "structure",
            "rsi",
            "adx",
            "atr_ratio",
            "volume_ratio",
            "macd_hist",
            "bb_pos",
        }
        filtered_indicators = {}
        for tf, tf_data in indicators.items():
            if isinstance(tf_data, dict):
                filtered_indicators[tf] = {
                    k: v for k, v in tf_data.items() if k in essential_keys
                }
                # Ensure there's at least a price field
                if (
                    "close" not in filtered_indicators[tf]
                    and "price" not in filtered_indicators[tf]
                ):
                    # Try to find any numeric value that could be price
                    for k, v in tf_data.items():
                        if isinstance(v, (int, float)) and v > 0:
                            filtered_indicators[tf]["close"] = v
                            break
            else:
                filtered_indicators[tf] = tf_data

        import json as json_module

        payload_size = len(json_module.dumps(filtered_indicators))
        print(
            f"LangGraph request payload size: {payload_size} bytes, timeframes: {list(filtered_indicators.keys())}"
        )

        async with httpx.AsyncClient(timeout=60.0) as http_client:
            payload = {
                "symbol": symbol,
                "mode": mode.upper(),
                "indicators": filtered_indicators,
            }

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = await http_client.post(
                        f"{LANGGRAPH_URL}/analyze", json=payload, timeout=60.0
                    )
                    if response.status_code == 200:
                        try:
                            result = response.json()
                            print(
                                f"LangGraph Raw Response (JSON) attempt {attempt + 1}: {result}"
                            )
                        except json.JSONDecodeError:
                            print(
                                f"LangGraph returned non-JSON response: {response.text}"
                            )
                            if attempt < max_retries - 1:
                                await asyncio.sleep(2**attempt)
                                continue
                            return {"error": "Invalid JSON response from LangGraph"}

                        # Format to match the legacy frontend structure
                        setup = result.get("optimized") or result.get("original", {})
                        eval_data = result.get("evaluation", {})

                        # Validate that we have at least some meaningful data
                        has_critical_data = (
                            setup.get("entry")
                            or setup.get("tp")
                            or setup.get("sl")
                            or setup.get("bias")
                            or eval_data.get("confidence")
                        )

                        if not has_critical_data:
                            print(
                                f"Empty response from LangGraph (attempt {attempt + 1}/{max_retries})"
                            )
                            if attempt < max_retries - 1:
                                await asyncio.sleep(2**attempt)
                                continue
                            return {
                                "error": "LangGraph returned empty analysis after retries"
                            }

                        # Helper to flatten lists/strings to float
                        def to_float(val):
                            if isinstance(val, list):
                                return float(val[0]) if val else 0.0
                            try:
                                return float(val)
                            except:
                                return 0.0

                        # FORCE FLATTEN: Ensure top-level keys match what the frontend/terminal expects
                        final_entry = to_float(setup.get("entry"))
                        final_tp = to_float(setup.get("tp"))
                        final_sl = to_float(setup.get("sl"))
                        final_lev = (
                            int(setup.get("leverage"))
                            if setup.get("leverage")
                            else None
                        )

                        return {
                            "symbol": symbol,
                            "bias": setup.get("bias", "Neutral"),
                            # CORE TERMINAL DATA
                            "entry": final_entry,
                            "tp": final_tp,
                            "sl": final_sl,
                            "leverage": final_lev,
                            "pair": symbol,  # Legacy compat
                            # METRICS
                            "confidence": eval_data.get("confidence", 5),
                            "quant_confidence": eval_data.get("quant_confidence", 5),
                            "llm_adjustment": eval_data.get("llm_adjustment", 0),
                            "risk_reward": f"1:{eval_data.get('rr', 'N/A')}",
                            "safety_margin": eval_data.get("safety_margin"),
                            "liquidation_safe": eval_data.get("liquidation_safe", True),
                            "risk_pct": eval_data.get("risk_pct"),
                            "liq_pct": eval_data.get("liq_pct"),
                            # UI TEXT
                            "trade_setup": f"LANGGRAPH {mode} SETUP",
                            "reasoning": setup.get(
                                "reasoning",
                                "Validated through LangGraph node workflow.",
                            ),
                            "issues": result.get("issues", []),
                            "changes": setup.get("changes", []),
                        }
                    else:  # Non-200 status code
                        print(
                            f"LangGraph Error: {response.status_code} - {response.text}"
                        )
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2**attempt)
                            continue
                        return {
                            "error": f"LangGraph service returned status {response.status_code}"
                        }
                except httpx.RequestError as exc:
                    print(
                        f"LangGraph Connection Error (attempt {attempt + 1}): {type(exc).__name__}: {exc}"
                    )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2**attempt)
                        continue
                    return {"error": f"LangGraph connection failed: {exc}"}
                except Exception as e:
                    print(
                        f"Unexpected error in LangGraph call (attempt {attempt + 1}): {e}"
                    )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2**attempt)
                        continue
                    return {"error": f"Unexpected error: {e}"}

            # Should not reach here, but just in case
            return {"error": "Max retries exceeded without success"}
    except httpx.RequestError as exc:  # Catch specific httpx errors
        print(f"LangGraph Connection Error: {type(exc).__name__}: {exc}")
        import traceback

        traceback.print_exc()
        return {"error": f"LangGraph connection failed: {exc}"}
    except Exception as e:  # Catch other unexpected errors
        print(f"An unexpected error occurred: {e}")
        # Ensure error message is not empty for clarity
        return {"error": str(e) or "An unknown error occurred"}


def clean_llm_json(content: str) -> str:
    content = re.sub(r"```json\s*|\s*```", "", content)
    s = content.find("{")
    e = content.rfind("}")
    if s != -1 and e != -1:
        return content[s : e + 1]
    return content.strip()
