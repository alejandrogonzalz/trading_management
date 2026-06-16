# Next Steps — Thesis Comparison Plan

**Last updated:** 2026-06-16

---

## Current Status at a Glance

All measurement work is complete. The remaining steps are presentation and integration.

| Model | Test Accuracy | Status |
|-------|---------------|--------|
| QLoRA cloud (`qlora_cloud`) | **88.03%** | ✅ Done |
| QLoRA config3 (`qlora_config3`) | **87.87%** | ✅ Done |
| Zero-shot LLM (Qwen 2.5 7B) | **58.49%** | ✅ Done |
| Blending ensemble v2 | **63.59%** | ✅ Done |
| Soft Voting v2 | **61.79%** | ✅ Done |
| Stacking v2 | **61.60%** | ✅ Done |
| Bagging-LSTM v2 | **60.30%** | ✅ Done |
| LSTM individual | **50.21%** | ✅ Done |
| XGBoost v2 | **63.60%** | ✅ Done |
| Random Forest v2 | **64.24%** | ✅ Done |
| McNemar test (QLoRA vs zero-shot) | chi²=557, p≈0 | ✅ Done |
| `optimization-results.ipynb` (Parts 1–5) | Analysis complete | ✅ Done |

---

## What Is Done ✅

| Artifact | Result | Where |
|----------|--------|-------|
| Dataset — 56,161 labeled samples | 12 symbols, 18 months, 3 TFs | `backtest/data/labeled/dataset.jsonl` (DVC) |
| Leakage fix | Strict temporal holdout, global sort + embargo | `features._temporal_split` |
| QLoRA cloud training | 88.03% test acc, profit_factor=12.92, AUC=0.942 | `optimization/qlora/results/qlora_cloud/result.json` |
| QLoRA config3 validation | 87.87% test acc — confirms robustness | `optimization/qlora/results/qlora_config3/result.json` |
| Zero-shot backtest | 58.49% acc, 28.42% win rate, 98.47% drawdown | `backtest/data/results/zero-shot-qwen7b.json` |
| ML models (all) | XGB 63.6%, RF 64.24%, Blending 63.59% (best ML) | `optimization/results/*_v2_optimization.json` |
| McNemar test | chi²=557, p≈0 — fine-tuning effect is real | `optimization/stats_tests.py` |
| `optimization-results.ipynb` | Parts 1–5: loss curves, bar charts, McNemar, trade metrics | `langgraph/optimization-results.ipynb` |
| Discussion document | Full analysis + zero-shot explanation + config3 section | `docs/deliverables/DISCUSSION_RESULTS.md` |
| Ollama deployment guide | Local inference walkthrough | `docs/ops/OLLAMA_DEPLOYMENT.md` |
| DVC tracking | Dataset, candles, QLoRA models in S3 | Remote: `s3://trading-management-dvc/` |

---

## What Remains ⏳

### 1. Part 6 — Equity Curves (notebook)

The only analysis piece missing from `optimization-results.ipynb`. Add a cell that
plots the cumulative equity curve for the 4 main models (QLoRA cloud, zero-shot,
best ML, LSTM) over the test period. The data is already in the result JSONs under
`trade_results[].pnl_pct`.

```python
# Sketch — add to optimization-results.ipynb Part 6
import matplotlib.pyplot as plt

def equity_curve(trade_results):
    equity = [1.0]
    for t in trade_results:
        equity.append(equity[-1] * (1 + t["pnl_pct"] / 100))
    return equity

fig, ax = plt.subplots(figsize=(12, 5))
for label, r in results.items():
    eq = equity_curve(r.get("trade_results", []))
    ax.plot(eq, label=label)
ax.axhline(1.0, color="black", linestyle="--", alpha=0.4, label="Breakeven")
ax.set_xlabel("Trade #")
ax.set_ylabel("Cumulative equity (starting = 1.0)")
ax.set_title("Equity Curves — Test Set")
ax.legend()
plt.tight_layout()
plt.show()
```

---

### 2. Avance6.ipynb — Formal Thesis Deliverable

The notebook submitted as part of the MNA program. It should be a clean,
self-contained walkthrough of the full research — not a debugging notebook.
Minimum content:

1. Introduction: hypothesis, dataset description, leakage fix explanation
2. Results table: all 7+ models × accuracy, win rate, profit factor, Sharpe
3. QLoRA overfitting diagnostic (train/val/test bar + loss curve)
4. McNemar statistical test (QLoRA vs zero-shot)
5. Trade metrics comparison (win rate, profit factor, drawdown)
6. Equity curves (4 main models)
7. Conclusions: hypothesis confirmed, config3 robustness, zero-shot anomaly

Use `optimization-results.ipynb` as the source — copy and clean each part.
All cells must have output before committing.

---

### 3. LangGraph Integration

Deploy the fine-tuned model locally and wire it into the agent.

```bash
# Step 1: pull GGUF from S3 (DVC)
cd trading_management/
dvc pull langgraph/backtest/data/models/qlora_cloud.dvc

# Step 2: copy Modelfile and create Ollama model
cp langgraph/optimization/qlora/Modelfile \
   langgraph/backtest/data/models/qlora_cloud/gguf_gguf/Modelfile

cd langgraph/backtest/data/models/qlora_cloud/gguf_gguf/
ollama create trading-qwen-ft -f Modelfile

# Step 3: quick smoke test
LLM_PROVIDER=ollama LLM_MODEL=trading-qwen-ft \
  python -m cli run-backtest \
    --dataset backtest/data/labeled/dataset.jsonl \
    --provider ollama --max-samples 20 --tag qlora-smoke

# Step 4: wire into agent (set env vars)
# In langgraph/.env or shell:
export LLM_PROVIDER=ollama
export LLM_MODEL=trading-qwen-ft

# Test via LangGraph API
curl -s -X POST http://localhost:2024/analyze \
  -H "Content-Type: application/json" \
  -d '{"symbol":"BTCUSDT","mode":"SPOT","indicators":{"1h":{"rsi":48,"adx":22}}}' \
  | python3 -m json.tool
```

---

### 4. Thesis Write-up and Defense Prep

- Complete the technical report
- Build presentation slides
- Record video demo (LangGraph agent + fine-tuned model live)
- Defense ~Jun 26

---

## Timeline

| Date | Task |
|------|------|
| Jun 16 | Add Part 6 (equity curves) to optimization-results.ipynb |
| Jun 16–17 | Start Avance6.ipynb — copy + clean from optimization-results |
| Jun 17–18 | LangGraph integration + screenshots |
| Jun 18–20 | Technical report draft |
| Jun 20–22 | Presentation slides |
| Jun 23–25 | Buffer + rehearsal |
| ~Jun 26 | Defense |

---

## Key Files

| File | Role |
|------|------|
| `optimization/qlora/results/qlora_cloud/result.json` | QLoRA fine-tuned result (88.03%) |
| `optimization/qlora/results/qlora_config3/result.json` | Config3 robustness check (87.87%) |
| `backtest/data/results/zero-shot-qwen7b.json` | Zero-shot baseline (58.49%) |
| `optimization/results/blending_v2_optimization.json` | Best ML ensemble (63.59%) |
| `optimization/results/random_forest_v2_optimization.json` | Best ML individual (64.24%) |
| `optimization/stats_tests.py` | McNemar + paired t-test |
| `langgraph/optimization-results.ipynb` | Analysis notebook (Parts 1–5 done, Part 6 pending) |
| `docs/deliverables/DISCUSSION_RESULTS.md` | Full written analysis |
| `docs/ops/OLLAMA_DEPLOYMENT.md` | Local inference guide |
