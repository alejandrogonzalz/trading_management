# Next Steps — Thesis Comparison Plan

Status document for the four-approach comparison required by the thesis proposal (§2.3).
Audience: anyone who needs to understand what is done, what remains, and in what order.

---

## 1. What Has Been Done

| Artifact | Status | Notes |
|----------|--------|-------|
| **Dataset — 56,161 labeled samples** | ✅ Done | 12 symbols, 18 months, 3 TFs; 49% LONG / 51% SHORT; DVC-tracked at `s3://trading-management-dvc/` |
| **Feature engineering** | ✅ Done | 9 indicators × 3 TFs + 3 cross-TF features = dynamic vector; `backtest/models/features.py` |
| **Data leakage fix** | ✅ Done | `_temporal_split` now sorts ALL 56,161 samples globally by `timestamp` + 24h embargo at boundaries; prior positional cut was a per-symbol split that leaked market regime. See `docs/AUDITORIA_QLORA_LEAKAGE_OVERFITTING.md` |
| **ML optimization — individual models** | ✅ Done ⚠️ | LSTM 81.5%, XGBoost 66.6%, SVM 70.8% test acc — **all numbers pre-fix; need re-measuring on the fixed split (Tarea 7)** |
| **ML optimization — ensembles** | ✅ Done ⚠️ | Bagging-LSTM 83.37% test acc (5 bags, AUC 0.9157), Blending 81.89% — **pre-fix; need re-measuring** |
| **QLoRA training pipeline** | ✅ Done | `train_qlora.py` instrumented with `EarlyStoppingCallback(patience=3)`, train/val/test accuracy gap, loss curve, heuristic baseline |
| **QLoRA cloud training run** | 🔄 In progress | RunPod A100 80GB (`unsloth/unsloth:latest` image, FA2 2.8.3 pre-installed). Config: `lr=2e-5, rank=16, alpha=32, epochs=2, batch=2, grad_accum=8`. Observed ~3.6s/step → estimated ~5h total, ~$7 on A100 (~$1.39/hr). Run started via `nohup bash optimization/qlora/run_cloud.sh`. Output: `optimization/qlora/results/qlora_optimization.json`. The `unsloth/unsloth:latest` Docker image was chosen because it ships with FA2 pre-installed — avoiding the ABI mismatch that plagued bare torch images. No venv needed; `/opt/venv` is pre-activated in the container. |
| **~~QLoRA config-1 (92%)~~** | ❌ Invalid | Trained on positional (per-symbol) split; eval on 2000 single-symbol (LINK) rows; `win_rate=0.0`. Archived under `optimization/qlora/results/archive/`. Do not cite. |
| **Statistical comparison framework** | ✅ Done | `optimization/stats_tests.py`: McNemar exact binomial + paired t-test; requires `sample_keys` in both result JSONs |
| **Shared temporal test set** | ✅ Done | All runners (`LLMBacktestRunner`, `MLBacktestRunner`, `train_qlora.py`) call `features._temporal_split` — same 8,424 test samples across all models; paired tests are valid |
| **Backtest framework CLI** | ✅ Done | 6 commands: `fetch-candles`, `prepare-dataset`, `run-backtest`, `train-ml`, `export-training-data`, `compare-stats` |
| **LangGraph agent (3-node ReAct DAG)** | ✅ Done | Generator → Evaluator → Optimizer; running on port 2024; tested with 7 LLM providers |
| **DVC tracking** | ✅ Done | `dataset.jsonl`, `candles/`, trained models tracked; remote `s3://trading-management-dvc/` |

---

## 2. What Is Pending (ordered by dependency)

### Step 1 — Collect QLoRA Result

**What:** Wait for the RunPod run to finish and verify the output file is valid.

**Why this unblocks everything:** the QLoRA result JSON is required for all four
statistical comparison pairs and for the Avance 6 comparison table.

**How to monitor:**
```bash
# On the RunPod pod
tail -f optimization/qlora/logs/run_cloud.log
python3 optimization/qlora/monitor_training.py --epochs 2 --tz -6
```

**Expected output:** `optimization/qlora/results/qlora_optimization.json`

Verify the result is valid (not from the contaminated run):
```bash
python3 -c "
import json
r = json.load(open('optimization/qlora/results/qlora_optimization.json'))
print('accuracy:', r['metrics']['direction_accuracy'])
print('win_rate:', r['metrics']['win_rate'])      # must be > 0 (candles were pulled via dvc)
print('sample_keys count:', len(r.get('sample_keys', [])))  # must be ~8424
"
```

**Success criteria:** `accuracy` > 0.51 (beats majority-class baseline), `win_rate` > 0.0,
`sample_keys` length ≈ 8,424.

**Back up to S3 after collecting:**
```bash
aws s3 cp --recursive backtest/data/models/qlora_cloud/ s3://trading-management-dvc/models/qlora_cloud/
dvc add backtest/data/models/qlora_cloud
git add backtest/data/models/qlora_cloud.dvc
git commit -m "feat(qlora): add cloud training result from fixed temporal split"
```

---

### Step 2 — Re-measure ML / Ensembles on the Fixed Split (Tarea 7)

**What:** Re-run Bagging-LSTM and individual LSTM through `MLBacktestRunner` on the
same `_temporal_split` used by the QLoRA run. All numbers from Avance 5 are stale
because they were computed on the per-symbol (positional) split.

**Why this unblocks:** without valid ML results on the fixed split, the comparison
table rows for approaches 1 and 4 cannot be filled in.

**Commands (run from `langgraph/` with venv active):**
```bash
source .venv/bin/activate
cd langgraph

# Re-measure individual LSTM on the fixed split
OMP_NUM_THREADS=1 python -m cli train-ml \
  --model lstm \
  --dataset backtest/data/labeled/dataset.jsonl \
  --serialize \
  --tag ml-lstm
# Output: backtest/data/results/ml-lstm.json

# Re-measure Bagging-LSTM (need bagging-lstm support in MLBacktestRunner)
# If EnsembleLSTMPredictor is implemented:
OMP_NUM_THREADS=1 python -m cli train-ml \
  --model bagging-lstm \
  --dataset backtest/data/labeled/dataset.jsonl \
  --serialize \
  --tag ml-bagging-lstm
# Output: backtest/data/results/ml-bagging-lstm.json

# Re-measure XGBoost
OMP_NUM_THREADS=1 python -m cli train-ml \
  --model xgboost \
  --dataset backtest/data/labeled/dataset.jsonl \
  --serialize \
  --tag ml-xgboost
# Output: backtest/data/results/ml-xgboost.json
```

**Important:** the `EnsembleLSTMPredictor` class may not yet exist in
`backtest/models/lstm.py` — check before running. If missing, use the individual LSTM
result for the primary ML comparison row.

**Output files:**
- `backtest/data/results/ml-lstm.json` — individual LSTM on fixed test
- `backtest/data/results/ml-bagging-lstm.json` — Bagging-LSTM on fixed test
- `backtest/data/results/ml-xgboost.json` — XGBoost on fixed test

---

### Step 3 — Zero-shot LLM Backtest (Groq Llama 3.3 70B)

**What:** Run `LLMBacktestRunner` with Groq over the fixed 8,424-sample test split.
This is Approach 2 from the proposal — the zero-shot baseline against which fine-tuning
is measured.

**Why this unblocks:** required for comparison pairs (b) and (c); establishes the
accuracy of the same base reasoning capability without domain adaptation.

**Precondition:** `GROQ_API_KEY` must be set; candles pulled via DVC (otherwise
`win_rate=0.0`).
```bash
dvc pull backtest/data/labeled/dataset.jsonl backtest/data/candles
```

**Command:**
```bash
cd langgraph && source .venv/bin/activate

LLM_PROVIDER=groq LLM_MODEL=llama-3.3-70b-versatile \
  python -m cli run-backtest \
    --dataset backtest/data/labeled/dataset.jsonl \
    --provider groq \
    --tag zero-shot-llama70b
```

> Groq rate limits: if the run hits limits, add `--max-samples 2000` for a
> representative subsample. Results will be annotated as partial in the comparison
> table. For the thesis, a full ~8K evaluation is preferred — run overnight if needed.

**Output:** `backtest/data/results/zero-shot-llama70b.json`

**Optional — zero-shot Qwen 7B base (to isolate the pure fine-tuning effect on the
same model family):**
```bash
LLM_PROVIDER=ollama LLM_MODEL=qwen2.5:7b \
  python -m cli run-backtest \
    --dataset backtest/data/labeled/dataset.jsonl \
    --provider ollama \
    --tag zero-shot-qwen7b
# Output: backtest/data/results/zero-shot-qwen7b.json
```

---

### Step 4 — Statistical Comparisons (McNemar + Paired t-test)

**What:** Run `compare-stats` for all required comparison pairs. Each pair answers
one scientific question from the proposal.

**Why this is the academic core:** the thesis hypothesis requires statistical evidence
that differences are not due to chance. A p-value < 0.05 in McNemar's test is the
threshold for significance.

**Precondition:** all four result JSONs must have `sample_keys` of length ≈ 8,424.
Check with `python -c "import json; r=json.load(open('<file>')); print(len(r.get('sample_keys',[])))"`.

**All commands run from `langgraph/` with venv active:**

```bash
cd langgraph && source .venv/bin/activate

# Pair (a): QLoRA vs Bagging-LSTM — the central thesis question
# "Does fine-tuning a 7B LLM reach or exceed the best ML classifier on this domain?"
python -m cli compare-stats \
  --a optimization/qlora/results/qlora_optimization.json \
  --b backtest/data/results/ml-bagging-lstm.json

# Pair (b): Zero-shot Llama 70B vs Bagging-LSTM
# "Does an LLM's general reasoning ability match ML trained directly on this data?"
python -m cli compare-stats \
  --a backtest/data/results/zero-shot-llama70b.json \
  --b backtest/data/results/ml-bagging-lstm.json

# Pair (c): QLoRA vs Zero-shot Llama 70B
# "What is the net effect of domain fine-tuning vs prompting a frontier model?"
python -m cli compare-stats \
  --a optimization/qlora/results/qlora_optimization.json \
  --b backtest/data/results/zero-shot-llama70b.json

# Pair (d): Zero-shot Llama 70B vs individual LSTM
# "Can LLM chain-of-thought reasoning beat a supervised classifier for direction prediction?"
python -m cli compare-stats \
  --a backtest/data/results/zero-shot-llama70b.json \
  --b backtest/data/results/ml-lstm.json
```

**What to look for in each output:**
- `mcnemar_p_value < 0.05` → the difference is statistically significant
- Even a 2pp difference in direction accuracy can flip the profit factor from below 1
  to above 1 in backtesting — so report the economic magnitude, not just p-values
- `positional_fallback: true` in the output → one JSON is missing `sample_keys`;
  the comparison is not paired and results are unreliable. Fix the runner first.
- For pairs where the LLM wins on accuracy but loses on profit factor: report both;
  this is scientifically interesting (structured output quality matters beyond direction)

---

### Step 5 — Notebook Avance6.ipynb

**What:** Create `langgraph/Avance6.ipynb` with the full comparison table, charts,
statistical tests, and answers to all 6 research questions.

**Sections to include:**

1. **Setup** — imports, paths to result JSONs, constants
2. **Comparison table** — all 4 approaches × all 9 metrics (accuracy, precision,
   recall, F1-macro, win rate, profit factor, Sharpe, max drawdown, cost/1000 preds)
   ```python
   # Pattern to adapt from optimization/avance5_report.py
   ```
3. **Bar charts by metric** — one chart per metric family (classification vs trading)
4. **Equity curves** — all 4 models superimposed
   ```python
   from backtest.evaluation.report import plot_equity_curve
   # feed trade_results[*].pnl_pct from each result JSON
   ```
5. **Confusion matrices** — QLoRA vs Bagging-LSTM side by side
   ```python
   from sklearn.metrics import ConfusionMatrixDisplay
   ```
6. **ROC curves** — all 4 models on one plot
   ```python
   from sklearn.metrics import RocCurveDisplay  # needs test_y_proba in result JSONs
   ```
7. **McNemar heatmap** — p-values for all 4 pairs as a seaborn heatmap
8. **Per-symbol accuracy** — `groupby("symbol")` on `predictions[*]` to detect
   which pairs each approach handles better
9. **QLoRA confidence calibration**
   ```python
   from backtest.evaluation.metrics import confidence_calibration
   # already implemented; used in Avance5.ipynb §7 — replicate here
   ```
10. **Loss curve and train/val/test gap** — from `optimization/qlora/results/<tag>_loss_curve.json`
11. **Market regime analysis** — split test set by ADX: high (>25, trending) vs
    low (<20, ranging); compare per-regime accuracy across approaches
12. **Research questions scorecard** — one paragraph per question with the specific
    metric that answers it (see Section 5 of this document)

---

### Step 5b — Local Docker Training (optional, $0 cost alternative)

**What:** Run `unsloth/unsloth:latest` locally on the RTX 5070 Ti via Docker Desktop +
WSL2 GPU passthrough to get FlashAttention-2 — the same image used on RunPod.

**When to use this:**
- Cloud result is still running and you want a local sanity check in parallel
- Cloud result was lost / corrupted and you need a recovery run
- You want a second data point for the thesis infrastructure comparison section (local
  GPU vs cloud GPU cost-accuracy trade-off)

**Why this is faster than the current Windows native path:**

| Setup | FA2 | Step time | Cost |
|-------|-----|-----------|------|
| Windows native (`.venv-finetuning`) | No | ~28s | $0 |
| **Docker local (RTX 5070 Ti)** | **Yes** | **~5–12s** | **$0** |
| RunPod A100 80GB | Yes | ~3.6s | ~$3.50/epoch |

**Full guide:** `docs/LOCAL_DOCKER_TRAINING.md`

**Quick start (PowerShell from `langgraph/`):**
```powershell
# 1. Confirm GPU is visible to Docker
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi

# 2. Pull the image (one-time, ~15 GB)
docker pull docker.io/unsloth/unsloth:latest

# 3. Launch training (smoke test runs first, then full 1-epoch run)
docker run --rm `
  --gpus all --ipc=host --ulimit memlock=-1 `
  -v "${PWD}:/workspace" -w /workspace `
  docker.io/unsloth/unsloth:latest `
  bash optimization/qlora/run_docker_local.sh
```

**Expected output:** `optimization/qlora/results/qlora_docker_local.json`
(distinct tag from the cloud run — both can coexist)

---

### Step 6 — LangGraph Integration (conditional on QLoRA result)

**What:** Deploy the fine-tuned model to Ollama and wire it into `generator_node()`.

**Condition:** QLoRA result must achieve `accuracy > 0.55` AND `profit_factor > 1.0`
on the fixed test (the proposal's minimum success criteria). Below these thresholds,
document the gap but skip the live integration — negative result is academically valid.

**Commands:**
```bash
# 1. Create an Ollama Modelfile pointing to the GGUF
cat > backtest/data/models/qlora_cloud/Modelfile << 'EOF'
FROM ./model_q8_0.gguf
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.1
EOF

# 2. Register the model in Ollama
ollama create trading-qwen-ft -f backtest/data/models/qlora_cloud/Modelfile

# 3. Set the env variable and test end-to-end
LLM_PROVIDER=ollama LLM_MODEL=trading-qwen-ft \
  python -m cli run-backtest \
    --dataset backtest/data/labeled/dataset.jsonl \
    --provider ollama \
    --tag qlora-integrated \
    --max-samples 50    # smoke test first
```

**Verify `generator_node()` works:**
```bash
curl -s -X POST http://localhost:2024/analyze \
  -H "Content-Type: application/json" \
  -d '{"symbol": "BTCUSDT", "mode": "SPOT", "indicators": {"1h": {"rsi": 48, "adx": 22}}}' \
  | python3 -m json.tool
```

---

## 3. Verification Checklist

All items must be true before the comparison table can be considered final.

**Data integrity**
- [ ] `backtest/data/labeled/dataset.jsonl` is the post-fix version (DVC-tracked;
  `dvc status` shows no changes)
- [ ] `backtest/data/candles/` is pulled (`dvc pull`); `win_rate` is non-zero in
  all LLM result JSONs — this confirms `simulate_trade()` had candles to walk

**Result JSON completeness**
- [ ] `optimization/qlora/results/qlora_optimization.json` exists with `sample_keys`
  length ≈ 8,424
- [ ] `backtest/data/results/zero-shot-llama70b.json` exists with `sample_keys`
  length ≈ 8,424
- [ ] `backtest/data/results/ml-bagging-lstm.json` (or `ml-lstm.json`) exists with
  `sample_keys` length ≈ 8,424
- [ ] All JSONs have `win_rate` > 0.0 (non-zero means candles were available during
  `simulate_trade()`)

**Paired comparison validity**
- [ ] `python -m cli compare-stats --a <file1> --b <file2>` does NOT print
  `"using positional fallback"` for any pair — if it does, one or both JSONs are
  missing `sample_keys` and the comparison is not truly paired
- [ ] `sample_keys` sets overlap by > 95% across all four result JSONs (confirm that
  all models were evaluated on the same test samples)

**Comparison table**
- [ ] Table in `Avance6.ipynb` has exactly 4 rows:
  `zero-shot-llama70b`, `qlora-cloud`, `ml-bagging-lstm` (or `ml-lstm`),
  and optionally `zero-shot-qwen7b`
- [ ] All 9 metric columns are populated (no `None` or `0.0` unexpectedly)
- [ ] ⚠️ pre-fix annotation is removed from the row(s) that have been re-measured
  on the fixed split

**Notebook**
- [ ] `Avance6.ipynb` runs `Kernel → Restart & Run All` without errors
- [ ] All cells have output (notebook was executed before committing)

---

## 4. Charts for the Final Deliverable

| Chart | Input data | Function / tool |
|-------|-----------|----------------|
| Comparison table (all 4 approaches × 9 metrics) | All 4 result JSONs | `pd.DataFrame` + `display()` in notebook; adapt `optimization/avance5_report.py` |
| Bar chart — classification metrics (acc, F1-macro) | `metrics.direction_accuracy`, `metrics.f1_macro` per JSON | `matplotlib.pyplot.bar` in `Avance6.ipynb` |
| Bar chart — trading metrics (win rate, profit factor, Sharpe) | `metrics.win_rate` etc. per JSON | `matplotlib.pyplot.bar` — group by metric |
| Equity curves (4 models superimposed) | `trade_results[*].pnl_pct` per JSON | `backtest.evaluation.report.plot_equity_curve()` |
| Confusion matrix — QLoRA vs Bagging-LSTM | `predictions`, `actuals` arrays in result JSONs | `sklearn.metrics.ConfusionMatrixDisplay.from_predictions()` |
| ROC curves (4 models on one plot) | `test_y_proba` + `actuals` | `sklearn.metrics.RocCurveDisplay` (needs probability arrays in result JSONs) |
| McNemar heatmap (p-values for all 4 pairs) | Output of `compare-stats` for each pair | `seaborn.heatmap` on a 4×4 matrix of p-values |
| QLoRA confidence calibration | `predictions[*].confidence` vs outcome | `backtest.evaluation.metrics.confidence_calibration()` |
| Per-symbol accuracy breakdown | `predictions[*].symbol` grouped | `groupby("symbol").mean()` + horizontal bar chart |
| Probability distribution — QLoRA vs Bagging-LSTM | `test_y_proba` arrays | Overlapping histogram (`plt.hist(..., alpha=0.5)`) |
| Training loss curve (QLoRA) | `optimization/qlora/results/<tag>_loss_curve.json` | `plt.plot(train_losses, label="train")` + `plt.plot(val_losses, label="val")` |
| Market regime accuracy (ADX high vs. low) | `predictions[*]` filtered by `indicators.1h.adx` | Split DataFrame + grouped bar chart |

---

## 5. Research Questions Scorecard

Maps each of the 6 research questions from the proposal (§2.4) to the specific metric
and comparison pair that provides the answer. Use this table when writing the Avance 6
conclusions.

| # | Research question | Answer metric | Comparison pair | Where |
|---|---|---|---|---|
| Q1 | Does the ReAct agent (LangGraph) outperform zero-shot without retraining, solely through pipeline design? | Direction accuracy + win rate | ReAct agent backtest vs `zero-shot-llama70b.json` | Requires running `LLMBacktestRunner` through the `/analyze` endpoint as provider; optional if time is short — note the gap in the thesis |
| Q2 | Does QLoRA fine-tuning significantly improve over zero-shot and over the ReAct agent? (target: >55% accuracy) | Direction accuracy, F1-macro, profit factor | Pair (c): `qlora_optimization.json` vs `zero-shot-llama70b.json`; McNemar p-value | `compare-stats` output + Q2 row in comparison table |
| Q3 | Which approach offers the best cost-efficiency (accuracy per dollar of inference)? | Accuracy / (cost per 1000 predictions) | All 4 approaches; cost figures from proposal §3.7 Table 5 | Cost column in comparison table + narrative in Avance 6 §3 |
| Q4 | Does the fine-tuned model generate coherent reasoning consistent with input indicators? | Confidence calibration; qualitative review of reasoning field | QLoRA result JSON `predictions[*].reasoning` | Confidence calibration chart + 5-10 sampled reasoning examples in Avance 6 |
| Q5 | Is there a statistically significant difference between the four approaches? | McNemar p-value (threshold: p < 0.05) | All 4 pairs from Step 4 | McNemar heatmap in Avance 6 |
| Q6 | Which approach is most robust across market regimes (trending vs. ranging)? | Direction accuracy segmented by ADX > 25 (trend) vs ADX < 20 (range) | All 4 models; per-regime accuracy table | Market regime analysis chart in Avance 6 |

**Success criteria summary** (from proposal §10):

| Criterion | Threshold | Status |
|-----------|-----------|--------|
| QLoRA accuracy beats zero-shot | any positive difference, confirmed by McNemar | pending |
| Profit factor > 1.0 | `metrics.profit_factor > 1.0` in QLoRA result | pending |
| Documented comparison with quantitative metrics | Avance 6 comparison table | pending |
| A negative result is valid if causes are analyzed | N/A — document the gap if QLoRA misses the threshold | — |

---

## Files Referenced

| File | Role |
|------|------|
| `optimization/qlora/results/qlora_optimization.json` | QLoRA result on fixed split (Step 1) |
| `backtest/data/results/zero-shot-llama70b.json` | Zero-shot Llama 3.3 70B (Step 3) |
| `backtest/data/results/ml-bagging-lstm.json` | Re-measured Bagging-LSTM (Step 2) |
| `backtest/data/results/ml-lstm.json` | Re-measured individual LSTM (Step 2) |
| `optimization/stats_tests.py` | McNemar + paired t-test implementation |
| `backtest/evaluation/runner.py` | `LLMBacktestRunner`, `MLBacktestRunner` |
| `backtest/evaluation/report.py` | `plot_equity_curve()` |
| `backtest/evaluation/metrics.py` | `confidence_calibration()` and all metric functions |
| `backtest/models/features.py` | `_temporal_split` — shared by all models |
| `agent/prompts.py` | Shared system/user prompt templates |
| `docs/DATA_PIPELINE.md` | How the dataset was built |
| `docs/QLORA_FINETUNING.md` | QLoRA pipeline status, runbook, constraints |
| `docs/HYPERPARAMETERS.md` | Hyperparameter explanations and sensitivity |
| `docs/PROJECT_DEFINITION.md` | Formal thesis proposal — do not modify |
| `docs/LOCAL_DOCKER_TRAINING.md` | Full guide for Docker-based local fine-tuning (Step 5b) |
| `optimization/qlora/run_docker_local.sh` | Script that runs inside the container (Step 5b) |
