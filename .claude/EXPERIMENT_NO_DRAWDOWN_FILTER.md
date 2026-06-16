# Experiment: Fine-Tuning Without the Drawdown Filter

> **Status:** PREPARED — code + docs ready, **no training has been run.**
> **Branch:** `experiment/no-drawdown-filter`
> **Model tag:** `qlora_no_drawdown_filter`
> **Dataset type:** `no_filter`
> Addresses **Critique #2** of the audit (`docs/qlora/AUDIT_QLORA_88PCT.md` §2 — Label Leakage / survivorship bias).

---

## 1. Context and Motivation

The labeler (`langgraph/backtest/ingestion/labeler.py`) applies a chain of quality
filters. One of them — the **drawdown-before-profit filter** — discards any sample
whose stop-loss would have been hit *before* its take-profit within the 24-candle
lookahead window. The audit flagged this as a survivorship bias:

> "This means: **only samples where the trade WOULD HAVE WORKED (TP hit before SL)
> are labeled.** Ambiguous or losing trades are discarded as `None`. This is not
> 'leakage' per se — it's a design choice that creates a biased dataset where every
> sample represents a historically profitable trade."
> — Audit §2

> "The 88% accuracy is on pre-filtered 'easy' samples, not on the full universe of
> market conditions."
> — Audit §2, Verdict

> "On unfiltered market data (all candles, not just clear signals), this accuracy
> would almost certainly drop to 60-70%."
> — Audit §5, Verdict

The auditor's explicit recommendation:

> "Clearly state the labeler's quality filters in the methodology — readers must
> understand that 88% is on pre-filtered high-quality setups, not on arbitrary
> market conditions."
> — Audit §9, point 6

**Hypothesis.** If the drawdown filter is removed, directional accuracy will drop,
because the model must now call direction even on noisy paths where price dipped to
the stop before resolving in the correct direction. The auditor predicts 60–70%.

**Internal vs external validity (the key distinction).** Removing the filter does
**not** invalidate the *comparison* between QLoRA, zero-shot, and the ML models —
all models always saw the same filtered data, so the comparison is internally
valid. What the filter affects is **external validity**: how the reported 88%
generalizes to arbitrary market conditions. This experiment measures exactly that.

---

## 2. Objectives

1. **Quantify the filter's effect on absolute accuracy.** How much does the
   drawdown filter inflate the headline number?
2. **Test whether the +29.5pp gain over zero-shot survives** on unfiltered data. If
   it shrinks a lot, the contribution is less robust; if it holds, it is stronger.
3. **Estimate "honest" real-world accuracy** — unfiltered data is closer to what a
   deployed model would actually face.
4. **Give the thesis a defensible framing** for the 88% claim (see
   `RESULT_INTERPRETATION.md` for the decision table per outcome).

This experiment is **scoped to the drawdown filter only**. It does NOT address the
circular TP/SL financial metrics (audit §6/§9, already documented as a limitation),
the information-asymmetry ML gap (§7), or the inverse overfitting gap (§4).

---

## 3. Code Changes (this branch)

All changes are **backward-compatible** — every default reproduces the original run
exactly. Only the new flags activate the ablation.

| File | Change | Why |
|------|--------|-----|
| `langgraph/backtest/ingestion/labeler.py` | `label_candle(..., apply_drawdown_filter=True)` and `generate_labeled_dataset(..., apply_drawdown_filter=True)`. The drawdown-before-profit block (the two `for c in future` loops, formerly lines 81–93) is now wrapped in `if apply_drawdown_filter:`. | Make the survivorship filter conditional; bypass it for the ablation. **All other filters (ADX≥15, volume≥0.5, R:R≥1.0, directional clarity, whipsaw) are untouched.** |
| `langgraph/backtest/pipeline.py` | `DataPipeline(..., apply_drawdown_filter=True)`; forwarded into `generate_labeled_dataset`. | Plumb the flag through the pipeline. |
| `langgraph/cli.py` | `prepare-dataset` gains `--no-drawdown-filter` and `--output`. With the flag and no `--output`, it writes to `data/labeled/dataset_no_drawdown_filter.jsonl` (never overwrites the production `dataset.jsonl`). | User-facing way to generate the unfiltered dataset. |
| `langgraph/optimization/qlora/train_qlora.py` | `--dataset-type {filtered,no_filter}` (default `filtered`) and `--dataset PATH`. `DATASET_PATH` / `TRAINING_DATA_DIR` are now per-run config (`dataset_path`, `training_data_dir`). `no_filter` reads `dataset_no_drawdown_filter.jsonl` and exports chat data to a separate `training_data_no_filter/` dir. `prepare_data()` raises a clear `FileNotFoundError` if the unfiltered dataset hasn't been generated yet. | Train + evaluate on the unfiltered dataset without touching the filtered pipeline. |

The before/after of the toggled block:

```python
# BEFORE (always applied):
if bias == "LONG":
    for c in future:
        if c["low"] <= sl:
            return None        # discard if SL hit before TP
        if c["high"] >= tp:
            break
else:
    ...

# AFTER (conditional):
if apply_drawdown_filter:      # default True → identical behaviour
    if bias == "LONG":
        for c in future:
            if c["low"] <= sl:
                return None
            if c["high"] >= tp:
                break
    else:
        ...
```

`run_cloud.sh` needs **no change**: it already forwards unknown flags to
`train_qlora.py`, so `--dataset-type no_filter` passes straight through.

---

## 4. How to Execute

> **Prerequisites on the GPU box** (RunPod A100 80GB recommended — see
> `optimization/qlora/RUNPOD_GUIDE.md`):
> ```bash
> cd trading_management/langgraph
> dvc pull backtest/data/candles            # candles are required for labeling + trade sim
> ```
> The labeled `dataset.jsonl` is **not** needed for this experiment — we build a new
> one from the candles.

### Step 1 — Generate the unfiltered dataset (CPU, ~5–15 min)

```bash
cd langgraph
python -m cli prepare-dataset \
  --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT,AVAXUSDT,DOTUSDT,DOGEUSDT,LINKUSDT,MATICUSDT,NEARUSDT \
  --timeframes "1h,4h,1d" \
  --no-drawdown-filter
# → backtest/data/labeled/dataset_no_drawdown_filter.jsonl
```

**CRITICAL:** use the **same symbols and timeframes that produced the original
`dataset.jsonl`** (12 symbols, base TF `1h`, TFs `1h,4h,1d`). The temporal split,
embargo, and lookahead (24) are identical by construction — the *only* difference is
the drawdown filter. The command prints `N labeled samples (X LONG / Y SHORT)` per
symbol; **sum those to get the exact dataset size immediately** (no training needed).

### Step 2 — Fine-tune (RunPod, recommended path)

```bash
cd langgraph
nohup bash optimization/qlora/run_cloud.sh \
  --tag qlora_no_drawdown_filter \
  --lr 0.00002 --rank 16 --alpha 32 --epochs 2 \
  --dataset-type no_filter \
  > optimization/qlora/logs/qlora_no_drawdown_filter.log 2>&1 &
echo "PID: $!"
```

Or call the trainer directly (equivalent, more explicit; tune batch sizes to the GPU):

```bash
python optimization/qlora/train_qlora.py \
  --dataset-type no_filter \
  --lr 0.00002 --rank 16 --alpha 32 --epochs 2 \
  --batch-size 8 --grad-accum 2 \
  --eval-batch-size 32 --max-eval 3000 --diagnostic-samples 200 \
  --tag qlora_no_drawdown_filter \
  --output-dir backtest/data/models/qlora_no_drawdown_filter
```

> **Hyperparameter choice — read this.** The recommended config above
> (`lr=2e-5, rank=16, alpha=32, epochs=2`) is **identical to the primary baseline
> `qlora_cloud` (88.03%)**. This is deliberate: a single-variable ablation must hold
> hyperparameters constant so the *only* thing that changes is the dataset (filter
> on → off). The original prompt mentioned a `config3`-style `lr=1e-5`; that is also
> valid **as long as the same config is used for both the baseline and the
> ablation**. Since the baseline we compare against is `qlora_cloud`, matching its
> hyperparameters is the cleaner comparison. Do not change two variables at once.

### Step 3 — (Optional) Evaluate only, from saved adapters

If training finished but you want to re-run evaluation:

```bash
python optimization/qlora/train_qlora.py \
  --eval-only backtest/data/models/qlora_no_drawdown_filter \
  --dataset-type no_filter \
  --eval-batch-size 32 \
  --tag qlora_no_drawdown_filter
```

### Step 4 — Compare against the baselines

The no-filter test set differs from the filtered one, so a clean **paired** McNemar
needs zero-shot re-run on the *same* unfiltered test set first:

```bash
# Re-run zero-shot on the unfiltered test split (paired comparison)
python -m cli run-backtest \
  --dataset backtest/data/labeled/dataset_no_drawdown_filter.jsonl \
  --provider ollama --split test --tag zero-shot-no-filter

# Paired McNemar + t-test on the unfiltered data
python -m cli compare-stats \
  --a optimization/qlora/results/qlora_no_drawdown_filter/result.json \
  --b backtest/data/results/zero-shot-no-filter.json
```

Then read the accuracy drop vs the filtered baseline (88.03%) and apply the decision
table in `RESULT_INTERPRETATION.md`.

---

## 5. Dataset Size, Time & Cost

**Dataset size — MEASURED** (2026-06-16, by labeling the DVC-pulled candles both
ways; the filtered total reproduces the canonical **56,161** exactly, which
validates the measurement):

| Quantity | Filtered (drawdown ON) | No-filter (drawdown OFF) | Δ |
|----------|------------------------|--------------------------|---|
| Total samples | 56,161 | **75,289** | **+19,128 (+34.1%)** |
| Train samples (≈70% temporal split) | 39,312 | **~52,700** | ≈ +13,400 |
| Steps / epoch (eff. batch 16) | 2,457 | **~3,290** | ≈ +830 |

The drawdown filter removes **19,128 samples (+34.1%)**. The effect is strikingly
**uniform across symbols (32–38%)** — it is not driven by one volatile coin, which
means it removes a consistent slice of "price dipped to the 1-ATR stop before TP"
setups everywhere. (MATICUSDT contributes 0 in both — no usable 1h candles locally,
same as the original 56,161.) Class balance stays ~49/51 LONG/SHORT.

> **Methodology.** Baseline `qlora_cloud` trained on 39,312 samples, effective batch
> 16 → 2,457 steps/epoch, ~3–4 h total on an A100/H100 (`run_cloud.sh` header). Step
> count scales linearly with dataset size (×1.34); eval at `--max-eval 3000` is
> constant (strided); GGUF export is ~constant.

**Training time & cost (2 epochs, ×1.34 on training only):**

| GPU (RunPod) | FA2 | Baseline time | No-filter (×1.34) | $/hr | No-filter cost |
|--------------|-----|---------------|-------------------|------|----------------|
| A100 80GB | yes | ~3–3.5 h total | **~4–4.7 h** | ~$1.39 | **~$6–8** |
| H100 80GB | yes | ~2.5–3 h | **~3.3–4 h** | ~$3.29 | **~$11–14** |
| L40S 48GB | yes | ~6–7 h train | **~8–9.5 h** | ~$0.86 | **~$7–9** |

- Eval (`--max-eval 3000`, batched): ~10–30 min, unchanged.
- GGUF export: ~15 min, unchanged.
- **Recommendation:** A100 80GB on RunPod — best cost/throughput (~$6–8 total).
- The optional zero-shot re-run (Step 4) on the unfiltered test set via Ollama adds
  separate inference time (depends on the local/GPU Ollama box, not the training GPU).

---

## 6. Results Location

| Artifact | Path |
|----------|------|
| Trained adapters + GGUF | `backtest/data/models/qlora_no_drawdown_filter/` (DVC-track after) |
| Result JSON (durable) | `optimization/qlora/results/qlora_no_drawdown_filter/result.json` |
| Loss curve | `optimization/qlora/results/qlora_no_drawdown_filter/loss_curve.{json,png}` |
| Canonical (latest) | `optimization/qlora/results/qlora_optimization.json` ⚠️ overwritten each run |
| Training log | `optimization/qlora/logs/qlora_no_drawdown_filter.log` |
| Unfiltered dataset | `backtest/data/labeled/dataset_no_drawdown_filter.jsonl` |

> ⚠️ `train_qlora.py` always rewrites `qlora_optimization.json` as the "latest"
> canonical. The **per-tag** `qlora_no_drawdown_filter/result.json` and the filtered
> baseline `qlora_cloud/result.json` are the durable artifacts; rely on those for
> the comparison, not on the canonical.

---

## 7. Agent Runbook — debug + launch on a GPU box

**Fresh instance (nothing installed yet)?** Use the bootstrap — it takes a bare GPU
box to "experiment running": health checks → AWS CLI → `dvc pull` candles → install
Claude Code → launch. AWS creds come from env vars (never hardcoded):

```bash
export AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=...  AWS_DEFAULT_REGION=us-east-1
export ANTHROPIC_API_KEY=sk-ant-...        # optional, for agent debugging
bash optimization/qlora/user_data.sh        # EC2 user-data / SageMaker lifecycle style
# Override defaults via env: REPO_URL, REPO_BRANCH, WORKDIR, AUTO_LAUNCH=0, SMOKE_ONLY=1
```

**Already provisioned** (repo + GPU stack + candles present)? Use the launcher
directly — preflight → dataset → smoke test → full run:

```bash
cd trading_management/langgraph
dvc pull backtest/data/candles      # once, if candles aren't on the box

# Validate the box WITHOUT committing to the long run (recommended first):
bash optimization/qlora/run_experiment_no_drawdown_filter.sh --smoke-only

# Then the real thing (survives disconnects, logs to a file):
nohup bash optimization/qlora/run_experiment_no_drawdown_filter.sh --skip-smoke \
  > optimization/qlora/logs/qlora_no_drawdown_filter.log 2>&1 & echo "PID: $!"
```

The launcher prints `OK —` / `FAIL —` per stage. Flags: `--smoke-only` (stop after
the 3-step test), `--skip-smoke`, `--skip-checks`.

**Debug decision tree** (failure → cause → fix):

| Stage / symptom | Likely cause | Fix |
|-----------------|--------------|-----|
| Preflight: "no CUDA GPU visible to torch" | wrong venv, or container not started with `--gpus all` | use the unsloth image's `python3` (`/opt/venv`); confirm `nvidia-smi` works |
| Preflight: "only N *_1h.json candle files" | candles not pulled | `dvc pull backtest/data/candles` (needs `aws configure`) |
| Dataset stage fails | candles corrupt / TA-Lib missing | check the printed traceback; verify `python3 -c "import talib"` |
| Smoke test OOM on step 0 | batch too big, or zombie process holding VRAM | retry lower `--batch-size`; `nvidia-smi` → `kill -9 <PID>` of stale runs |
| Smoke: "dataset not found (no_filter)" | dataset stage skipped/failed | re-run without `--skip-*`; confirm `dataset_no_drawdown_filter.jsonl` exists |
| Full run: `win_rate = 0` in result | candles missing during eval | `dvc pull backtest/data/candles` (trade sim needs them) |
| Full run dies mid-training | spot reclaim / disconnect | re-launch `run_cloud.sh --tag qlora_no_drawdown_filter --dataset-type no_filter --resume` |
| Result accuracy looks too low/high | read it, don't panic | this is the experiment's point — apply `RESULT_INTERPRETATION.md` |

**Monitoring while it runs:** `tail -f optimization/qlora/logs/qlora_no_drawdown_filter.log`,
`watch -n 10 nvidia-smi`. The `.claude/agents/backtest-monitor` agent can watch logs/results.

**When done:** run §4 Step 4 (paired zero-shot on the unfiltered test set →
`compare-stats`), then `dvc add` the model + dataset and apply `RESULT_INTERPRETATION.md`.

## 8. References

- `docs/qlora/EXPERIMENT_NO_DRAWDOWN_FILTER_GUIDE.md` — the human-facing, copy-paste
  start-to-finish guide (fresh GPU box → result → thesis). Start here to just run it.
- `docs/qlora/AUDIT_QLORA_88PCT.md` — §2 (drawdown filter / survivorship bias), §5
  (baseline, "60–70% on unfiltered data"), §9 point 6 (state the filters).
- `RESULT_INTERPRETATION.md` — outcome → paper-decision table.
- `SESSION_PERSISTENCE.md` — current state + pending steps.
- `.claude/rules/qlora-training.md` — hyperparameter reference + constraints.
- `optimization/qlora/RUNPOD_GUIDE.md` — GPU setup + cost analysis.
