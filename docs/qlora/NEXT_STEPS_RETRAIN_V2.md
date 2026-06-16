# QLoRA v2 Re-training Plan — Fix Label Leakage + Full Eval

**Objective**: Re-train with forward-looking TP/SL labels and evaluate on ALL 8,425 test samples

---

## Problem Summary

The current QLoRA model (88.03% accuracy) has two issues:

### 1. TP/SL Circularity (label leakage in financial metrics)
The training labels contain `tp` and `sl` values derived from **hindsight** — the labeler looks 24 candles into the future and sets `tp = entry * (1 + max_up * 0.7)`. The model learns to replicate these values, and the simulator confirms they "work" because they were derived from knowing the price DID reach that level.

**Impact**: Direction accuracy (88%) is **unaffected** — the model still correctly predicts LONG/SHORT. But PF=12.92 and win_rate=61.67% are artifacts of the circular TP/SL, not tradeable alpha.

### 2. Incomplete Evaluation  
Only 3,000 of 8,425 test samples were evaluated (strided). The CI is ±1.2pp instead of ±0.7pp.

### 3. Mild Overfitting (cosmetic — not severe)
Train loss continued falling after step ~500 while eval loss plateaued. Early stopping caught it at step 3000, but tighter settings would save ~80% of GPU time. The gap (0.017) is stable and small — this is diminishing returns, not catastrophic overfitting.

---

## The Fix

### Code Changes Made

| File | Change |
|------|--------|
| `backtest/export.py` | Added `_atr_based_tp_sl()` + `use_atr_tp_sl` flag to `build_training_example()` and `export_training_data()` |
| `optimization/qlora/train_qlora.py` | Added `--use-atr-tp-sl` CLI flag, changed `lora_dropout` from 0.0→0.05, changed `eval_steps` from 250→150, changed early stopping to `patience=4, threshold=0.001` |
| `optimization/qlora/run_cloud.sh` | `MAX_EVAL` default changed from 3000 to empty (eval all) |
| `optimization/qlora/run_retrain_v2.sh` | New one-command launcher for the v2 run |

### What the ATR fix does

**Before** (training label):
```json
{"bias": "LONG", "entry": 67500, "tp": 69200, "sl": 66800}
```
Where `tp=69200` was derived from knowing BTC actually reached $69,200 in the next 24h.

**After** (training label with `--use-atr-tp-sl`):
```json
{"bias": "LONG", "entry": 67500, "tp": 69300, "sl": 66300}
```
Where `tp = entry + 1.5 * ATR` and `sl = entry - 1.0 * ATR`. These values are knowable at decision time — no future data involved.

### Training configuration for v2

| Param | v1 (qlora_cloud) | v2 (qlora_v2_atr) | Why |
|-------|------------------|-------------------|-----|
| TP/SL source | Hindsight | ATR forward-looking | Breaks circularity |
| Epochs | 3 (early stop at 1.5) | 1 | Model converges in ~500 steps; 1 epoch = ~2,457 steps = plenty |
| eval_steps | 250 | 150 | Detect plateau faster |
| ES patience | 3 (needs 750 steps) | 4 (needs 600 steps) | More checks but each is cheaper |
| ES threshold | 0 (any improvement counts) | 0.001 | Ignore micro-improvements |
| lora_dropout | 0.0 | 0.05 | Mild regularization |
| MAX_EVAL | 3000 (strided) | ALL 8,425 | Full test, tighter CI |
| Effective batch | 16 | 16 | Unchanged |
| LR / rank / alpha | 2e-5 / 16 / 32 | Same | Unchanged |

---

## Expected Results

| Metric | v1 (current) | v2 (expected) | Reasoning |
|--------|--------------|---------------|-----------|
| Direction accuracy | 88.03% | ~85-88% | Direction is independent of TP/SL values |
| Win rate | 61.67% (inflated) | ~50-55% | ATR TP/SL is harder to hit than hindsight-optimal |
| Profit factor | 12.92 (artifact) | ~2-4 | Realistic range for a working system |
| Max drawdown | 12.3% | ~25-35% | More realistic without the circular advantage |
| Training time | ~3.5h (A100) | ~1-2h | 1 epoch + faster stopping |
| Cost | ~$5 | ~$2-3 | Less compute |

**Why direction accuracy should be preserved**: The model learns `indicators → direction` from the bias field. The TP/SL values in the label affect what the model outputs for trade structure, but the *direction prediction* (LONG vs SHORT) is learned from the indicator patterns → label bias mapping, which is independent of where TP/SL are set. Changing TP/SL from hindsight to ATR-based doesn't change which samples are labeled LONG vs SHORT.

---

## How to Run

```bash
cd trading_management/langgraph

# 1. Pull data (required — candles needed for ATR-based financial eval)
dvc pull backtest/data/labeled/dataset.jsonl backtest/data/candles

# 2. Launch (on RunPod A100 / similar GPU box)
nohup bash optimization/qlora/run_retrain_v2.sh \
  > optimization/qlora/logs/qlora_v2.log 2>&1 & echo "PID: $!"

# 3. Monitor
tail -f optimization/qlora/logs/qlora_v2.log
nvidia-smi

# 4. After completion — compare against v1
python -m cli compare-stats \
  --a optimization/qlora/results/qlora_v2_atr/result.json \
  --b optimization/qlora/results/qlora_cloud/result.json
```

---

## About Epochs — Why 1 Is Enough

From the v1 loss curve analysis:
- Model converged (eval loss plateau) at **~500 steps** (0.2 epochs)
- Best eval checkpoint was at step 3000, but only 0.008 better than step 500
- 1 epoch = ~2,457 steps = 5x the convergence point = generous safety margin

With `early_stopping_threshold=0.001` and `eval_steps=150`:
- Training will stop as soon as eval loss improvement < 0.001 for 4 consecutive checks (600 steps)
- Expected stopping point: ~step 600-900 (based on v1 convergence curve)
- If the model hasn't converged by step 2,457 (end of epoch 1), something is wrong

**2 epochs would be fine too** — the early stopping will catch it either way. But 1 epoch is sufficient given the empirical evidence, and saves time/cost.

---

## Thesis Narrative After v2

The v2 results strengthen the thesis by separating two claims:

1. **Direction prediction** (the core contribution): "Fine-tuning improves direction accuracy by +29pp over zero-shot" — this should hold at ~85-88%.

2. **Trade execution** (honest evaluation): "The fine-tuned model learns ATR-calibrated exit placement from training examples, producing realistic financial metrics (PF ~2-4, win rate ~50-55%)" — this replaces the inflated PF=12.92.

The v1 result is not invalidated — its direction accuracy was always the primary metric. v2 just makes the financial metrics defensible for the thesis defense.

---

## Validation After Training

1. **Paired McNemar test** (v2 vs zero-shot) — should still show chi² >> 100, p ≈ 0
2. **Per-symbol accuracy** — check all 12 symbols are above 80%  
3. **Loss curve** — confirm early stopping triggered before 1000 steps
4. **Confidence interval** — with n=8,425, CI at 88% is ±0.7pp (much tighter)
5. **Financial metrics** — confirm PF > 1.5 (profitable) and < 5 (realistic)
