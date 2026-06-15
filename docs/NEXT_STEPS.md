# Next Steps — Thesis Comparison Plan

Status document for the four-approach comparison required by the thesis proposal (§2.3).  
**Last updated:** 2026-06-15

---

## Current Status at a Glance

| Model | Test Accuracy | Status |
|-------|---------------|--------|
| QLoRA cloud (`qlora_cloud`) | **88.03%** | ✅ Done — result in `optimization/qlora/results/qlora_cloud/result.json` |
| QLoRA config3 (`qlora_config3`) | pending | 🔄 Training — A100, PID 235685, ~1h remaining |
| LSTM individual | 50.21% | ✅ Done — result in `backtest/data/results/ml-lstm.json` |
| LSTM v2 (expanded grid) | pending | 🔄 Running — `lstm_v2.yaml`, 50 configs, ~1-2h |
| Blending ensemble v2 | 63.59% | ✅ Done — `optimization/results/blending_v2_optimization.json` |
| Bagging-LSTM v2 | 60.30% | ✅ Done — `optimization/results/bagging_lstm_v2_optimization.json` |
| Soft Voting v2 | 61.79% | ✅ Done |
| Stacking v2 | 61.60% | ✅ Done |
| **Zero-shot LLM** | **NOT RUN** | ⛔ Highest priority next step |
| McNemar tests | NOT RUN | Blocked on zero-shot |

All models use the same fixed temporal split (`features._temporal_split`). See `docs/DISCUSSION_RESULTS.md` for the full analysis of results and why the pre-fix numbers (Avance 4–5) are invalid.

---

## 1. What Is Done

| Artifact | Notes |
|----------|-------|
| Dataset — 56,161 labeled samples | 12 symbols, 18 months, 3 TFs; DVC-tracked at `s3://trading-management-dvc/` |
| Data leakage fix | `_temporal_split` sorts globally by timestamp + 24h embargo. See `docs/AUDITORIA_QLORA_LEAKAGE_OVERFITTING.md` |
| ML optimization — individual + ensembles | Re-run on fixed split. Results in `optimization/results/*_v2_optimization.json` |
| QLoRA training pipeline | `train_qlora.py` with `EarlyStoppingCallback`, overfitting gap, loss curve, heuristic baseline |
| QLoRA cloud run | 88.03% test acc, negative gap (−17.5pp) → no overfitting. `sample_keys`: 3,000 |
| LSTM individual MLBacktestRunner | 50.21% test acc — temporal overfitting confirmed |
| Statistical comparison framework | `optimization/stats_tests.py`: McNemar + paired t-test, paired by `sample_keys` |
| LangGraph agent (3-node DAG) | Generator → Evaluator → Optimizer; port 2024 |
| DVC tracking | Dataset, candles, models in S3 |
| Unit tests for train_qlora.py | 28 tests in `tests/optimization/test_qlora.py` |

---

## 2. What Is Pending (ordered by dependency)

---

### STEP 1 — Zero-Shot LLM Backtest ⛔ HIGHEST PRIORITY

**What:** Run the base LLM (no fine-tuning) on the same test samples as QLoRA. This is the critical baseline — without it the thesis hypothesis ("fine-tuning improves over zero-shot") cannot be demonstrated.

**Important:** you do NOT need to deploy the fine-tuned model for this. Zero-shot uses the raw base model API. The fine-tuned model results already exist from `train_qlora.py`.

**What zero-shot means:** calling the LLM via Groq/Ollama with the same system+user prompt, but with the base model weights (no LoRA adapters from fine-tuning). The response is parsed the same way as fine-tuned — only the model weights differ.

#### Option A — Groq API (recommended, fastest)
Best model for zero-shot: `llama-3.3-70b-versatile` (strong reasoning) or `qwen-qwq-32b` (closest family to Qwen 2.5).

```bash
cd langgraph   # on RunPod or locally
export GROQ_API_KEY="your-key-here"

python -m cli run-backtest \
  --dataset backtest/data/labeled/dataset.jsonl \
  --provider groq \
  --tag zero-shot-llama70b
# Output: backtest/data/results/zero-shot-llama70b.json
```

Groq rate limits: ~30 RPM on free tier, ~6000 RPM on paid. With 8,425 samples at 1 req/sample, a free key takes ~5h. A paid key does it in minutes. Use `--max-samples 3000` for a quicker run that still produces valid paired statistics.

#### Option B — Qwen 2.5 7B zero-shot via Ollama (cleanest comparison)
This isolates the fine-tuning effect because it's the SAME model architecture, just without the LoRA adapters. Scientifically stronger for the thesis.

```bash
# On RunPod (install Ollama and pull base model)
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull qwen2.5:7b

python -m cli run-backtest \
  --dataset backtest/data/labeled/dataset.jsonl \
  --provider ollama \
  --tag zero-shot-qwen7b
# Output: backtest/data/results/zero-shot-qwen7b.json
```

**Recommended thesis setup:** run BOTH — Qwen 7B zero-shot (clean comparison) + Llama 70B zero-shot (frontier model upper bound). This gives you three LLM data points: zero-shot small, zero-shot large, fine-tuned small.

---

### STEP 2 — QLoRA config3 Result (happening now, ~1h)

Config3 is training: `lr=1e-5, rank=32, alpha=64, epochs=2`. Monitor:

```bash
# Check GPU (should be 100%)
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader

# Check process
ps aux | grep train_qlora | grep -v grep

# Log (only updates at end due to tqdm buffering)
tail -f optimization/qlora/logs/run_qlora_config3.log
```

Expected output: `optimization/qlora/results/qlora_config3/result.json`

If config3 > 88% → use it as primary result. If ≈88% → confirms robustness. If <88% → cloud config (lr=2e-5, rank=16) remains the best.

---

### STEP 3 — LSTM v2 Grid Search Result (happening now, ~1-2h)

Running 50 random configs from `lstm_v2.yaml` (wider hidden, longer seq_len, more LR options). After it finishes, check best_score:

```bash
python3 -c "
import json
r = json.load(open('optimization/results/lstm_optimization.json'))
print('best_score:', r.get('best_score'))
print('best_params:', r.get('best_params'))
"
```

Then re-run `MLBacktestRunner` with new params to get the TRUE temporal test accuracy:

```bash
OMP_NUM_THREADS=1 python -m cli train-ml \
  --model lstm \
  --dataset backtest/data/labeled/dataset.jsonl \
  --serialize \
  --tag ml-lstm-v2
```

**Expected outcome:** 55–65%. The fundamental temporal overfitting problem won't be solved by hyperparams — this run is to give the thesis a defensible "we searched broadly" statement.

---

### STEP 4 — XGBoost + RF True Test Accuracy (30 min, parallel)

XGBoost and RF only have CV scores (64.75% and 64.24%). Run them through `MLBacktestRunner` quickly:

```bash
OMP_NUM_THREADS=1 python -m cli train-ml \
  --model xgboost \
  --dataset backtest/data/labeled/dataset.jsonl \
  --serialize --tag ml-xgboost &

OMP_NUM_THREADS=1 python -m cli train-ml \
  --model random-forest \
  --dataset backtest/data/labeled/dataset.jsonl \
  --serialize --tag ml-rf &
```

---

### STEP 5 — McNemar + Paired t-test

Once zero-shot results exist, run all comparison pairs:

```bash
cd langgraph

# Core thesis pair: fine-tuned vs zero-shot
python -m cli compare-stats \
  --a optimization/qlora/results/qlora_cloud/result.json \
  --b backtest/data/results/zero-shot-qwen7b.json

# Best ML vs fine-tuned
python -m cli compare-stats \
  --a optimization/qlora/results/qlora_cloud/result.json \
  --b backtest/data/results/ml-lstm-v2.json   # or ml-lstm.json

# Zero-shot vs best ML
python -m cli compare-stats \
  --a backtest/data/results/zero-shot-qwen7b.json \
  --b backtest/data/results/ml-lstm-v2.json

# Llama 70B vs fine-tuned (frontier upper bound)
python -m cli compare-stats \
  --a optimization/qlora/results/qlora_cloud/result.json \
  --b backtest/data/results/zero-shot-llama70b.json
```

**Sanity check before running:** both JSONs must have `sample_keys`. If `compare-stats` prints "using positional fallback", one result is missing `sample_keys` and the test is not paired.

```bash
python3 -c "
import json
for f in ['optimization/qlora/results/qlora_cloud/result.json',
          'backtest/data/results/zero-shot-qwen7b.json']:
    r = json.load(open(f))
    print(f, '->', len(r.get('sample_keys', [])), 'keys')
"
```

---

### STEP 6 — Avance6.ipynb

Compile all results into the notebook. See `docs/DISCUSSION_RESULTS.md` §6 for the full list of required charts. Minimum content:

1. Comparison table: all models × accuracy, win rate, profit factor, Sharpe, F1
2. Before/after leakage fix bar chart
3. QLoRA overfitting diagnostic (train/val/test + loss curve)
4. McNemar heatmap (all comparison pairs)
5. Per-symbol accuracy (QLoRA)
6. Equity curves (4 main models)

---

### STEP 7 — LangGraph Integration (after Step 1–5)

Condition: QLoRA test accuracy > 55% AND profit_factor > 1.0 (both already met at 88% / 12.92).

```bash
# 1. Create Modelfile pointing to the GGUF
cat > backtest/data/models/qlora_cloud/Modelfile << 'EOF'
FROM ./gguf_gguf/Qwen2.5-7B-Instruct.Q4_K_M.gguf
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.1
EOF

# 2. Register in Ollama
ollama create trading-qwen-ft \
  -f backtest/data/models/qlora_cloud/Modelfile

# 3. Set env var and test
LLM_PROVIDER=ollama LLM_MODEL=trading-qwen-ft \
  python -m cli run-backtest \
    --dataset backtest/data/labeled/dataset.jsonl \
    --provider ollama \
    --tag qlora-integrated \
    --max-samples 50

# 4. Test via LangGraph agent
curl -s -X POST http://localhost:2024/analyze \
  -H "Content-Type: application/json" \
  -d '{"symbol":"BTCUSDT","mode":"SPOT","indicators":{"1h":{"rsi":48,"adx":22}}}' \
  | python3 -m json.tool
```

---

## 3. Timeline to Defense (~Jun 26)

| Date | Task |
|------|------|
| Jun 15 (today) | Config3 finishes, lstm_v2 finishes; launch zero-shot backtest overnight |
| Jun 16 | Zero-shot results in hand; run McNemar tests; run XGBoost/RF MLBacktestRunner |
| Jun 17 | All results final; start Avance6.ipynb |
| Jun 18–19 | Finish notebook + charts; LangGraph integration |
| Jun 20–22 | Technical report + presentation draft |
| Jun 23–25 | Buffer + rehearsal |
| ~Jun 26 | Defense |

---

## 4. Verification Checklist (before Avance 6 is final)

- [ ] `qlora_cloud/result.json` has `sample_keys` (count ~3,000) and `win_rate > 0`
- [ ] `zero-shot-*.json` exists with `sample_keys` (count ~8,425) and `win_rate > 0`
- [ ] `ml-lstm.json` or `ml-lstm-v2.json` exists with `sample_keys` (count ~8,425)
- [ ] `compare-stats` does NOT print "using positional fallback" for any pair
- [ ] `sample_keys` intersection between QLoRA and zero-shot is > 2,000 samples
- [ ] Avance6.ipynb runs clean (`Kernel → Restart & Run All`)
- [ ] All cells have output before committing
- [ ] Loss curve (`qlora_cloud/loss_curve.json`) referenced in notebook

---

## 5. Key Files

| File | Role |
|------|------|
| `optimization/qlora/results/qlora_cloud/result.json` | QLoRA fine-tuned result (88.03%) |
| `optimization/qlora/results/qlora_config3/result.json` | Config3 result (pending) |
| `backtest/data/results/ml-lstm.json` | LSTM temporal test (50.21%) |
| `backtest/data/results/zero-shot-*.json` | Zero-shot baseline (to be created) |
| `optimization/results/blending_v2_optimization.json` | Best ensemble (63.59%) |
| `optimization/stats_tests.py` | McNemar + paired t-test |
| `docs/DISCUSSION_RESULTS.md` | Full analysis of all results |
| `docs/AUDITORIA_QLORA_LEAKAGE_OVERFITTING.md` | Leakage fix audit trail |
