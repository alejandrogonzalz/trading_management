import os
import json
import re
import asyncio
from ollama import AsyncClient
from typing import List, Dict, Any, Optional
from app.core.config import settings
from app.services import scanner_service

OLLAMA_BASE_URL = settings.OLLAMA_BASE_URL
LLM_MODEL = settings.LLM_MODEL

client = AsyncClient(host=OLLAMA_BASE_URL)

def clean_llm_json(content: str) -> str:
    content = re.sub(r'```json\s*|\s*```', '', content)
    s = content.find('{')
    e = content.rfind('}')
    if s != -1 and e != -1: return content[s:e+1]
    return content.strip()

async def analyze_row(pair_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Highly structured technical analysis with robust key mapping.
    """
    prompt = f"""
    Analyze technical setup for {pair_data.get('symbol')}.
    Data: {json.dumps(pair_data)}

    Return ONLY JSON with:
    - bias: Bullish/Bearish/Neutral
    - confidence: 0-10
    - entry: float
    - tp: float
    - sl: float
    - risk_reward: string
    - trade_setup: title
    - reasoning: 2 sentences
    """
    try:
        response = await client.chat(
            model=LLM_MODEL,
            messages=[
                {'role': 'system', 'content': 'You are an Expert Analyst. Output JSON only. Use exact keys: bias, confidence, entry, tp, sl, risk_reward, trade_setup, reasoning.'},
                {'role': 'user', 'content': prompt}
            ],
            options={'temperature': 0.2},
            format='json'
        )
        
        raw_json = json.loads(clean_llm_json(response['message']['content']))
        
        # Robust Mapping: Map variations to standard keys
        mapping = {
            "entry_price": "entry", "suggested_entry": "entry",
            "take_profit": "tp", "target_price": "tp", "exit_target": "tp",
            "stop_loss": "sl", "stop_price": "sl",
            "rr": "risk_reward", "risk_to_reward": "risk_reward",
            "setup": "trade_setup", "title": "trade_setup",
            "analysis": "reasoning", "reason": "reasoning"
        }
        
        final_data = {}
        for key, val in raw_json.items():
            standard_key = mapping.get(key.lower(), key.lower())
            final_data[standard_key] = val
            
        # Ensure numeric values are actually floats
        for num_key in ['entry', 'tp', 'sl']:
            if num_key in final_data and final_data[num_key] is not None:
                try:
                    final_data[num_key] = float(str(final_data[num_key]).replace(',', ''))
                except ValueError:
                    final_data[num_key] = None

        return final_data
    except Exception as e:
        print(f"Deep Analysis Parse Error: {e}")
        return {"error": str(e), "bias": "Neutral", "reasoning": "Failed to generate technical targets."}

# Rest of the functions stay the same...
async def get_individual_opinion(p: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"Technical setup for {p['pair']}: Score {p['score']}, Heatmap {p.get('heatmap')}, Struct {p['structure']}. Return JSON: {{'bias': 'Bullish/Bearish', 'reason': '...'}}"
    try:
        response = await client.chat(
            model=LLM_MODEL,
            messages=[
                {'role': 'system', 'content': 'You are a quant analyst. Output JSON only. 1 sentence max.'},
                {'role': 'user', 'content': prompt}
            ],
            options={'temperature': 0.1, 'num_predict': 150},
            format='json'
        )
        return json.loads(clean_llm_json(response['message']['content']))
    except Exception:
        return {"bias": "Neutral", "reason": "Quant confluence analysis."}

async def rank_setups(scanner_table_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not scanner_table_data: return {}
    sorted_pairs = sorted(scanner_table_data, key=lambda x: x.get('score', 0), reverse=True)[:20]
    semaphore = asyncio.Semaphore(10)
    async def task_wrapper(pair):
        async with semaphore:
            opinion = await get_individual_opinion(pair)
            return pair['pair'], opinion
    enrichment_results = await asyncio.gather(*[task_wrapper(p) for p in sorted_pairs])
    results = {}
    opinions_map = dict(enrichment_results)
    for i, p in enumerate(sorted_pairs):
        pair_name = p['pair']
        opinion = opinions_map.get(pair_name, {"bias": "Neutral", "reason": "Technical analysis."})
        entry = { "pair": pair_name, "rank": i + 1, "bias": opinion.get('bias', 'Neutral'), "reason": opinion.get('reason', 'Technical analysis.') }
        results[pair_name] = entry
        for main_row in scanner_service.latest_scan_results.get("results", []):
            if main_row['pair'] == pair_name:
                main_row['ai_rank'] = entry['rank']
                main_row['ai_bias'] = entry['bias']
                main_row['ai_reason'] = entry['reason']
    return results
