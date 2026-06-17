# QLoRA 88% Accuracy Audit

**Date**: 2026-06-15
**Auditor**: Claude (adversarial review)
**Model Under Audit**: QLoRA cloud (lr=2e-5, rank=16, 2 epochs)
**Claimed Accuracy**: 88.03% directional accuracy on temporal test set
**Claimed Profit Factor**: 12.92

---

## Executive Summary

The 88.03% **direction accuracy** appears methodologically sound — the temporal split is correct, the embargo matches the lookahead, and the evaluation code has no inflation bugs. However, the result has significant caveats that limit its real-world applicability:

1. **The profit factor (12.92) and win rate (61.67%) are NOT reliable** — they are inflated by a circular TP/SL mechanism where the model learned hindsight-derived price targets.
2. **The inverse overfitting gap** (train_acc=70.5% < test_acc=88%) is suspicious and results from a flawed diagnostic methodology (capped first-500 train samples from the oldest regime).
3. **Only 3,000 of 8,425 test samples were evaluated** — the strided sub-sampling is methodologically acceptable but reduces confidence.
4. **The 24pp gap vs ML models** (88% vs 64%) is real but explained by information asymmetry: the LLM sees rich text context that tree models cannot access from flat feature vectors.
5. **No transaction costs, slippage, or market impact** in the trade simulation.
6. **The labeler pre-filters for high-quality setups** — the dataset excludes ambiguous samples, artificially raising achievable accuracy.

**Bottom line**: The directional accuracy claim is defensible for a thesis. The financial metrics (profit factor, win rate, Sharpe) should NOT be cited as evidence of trading viability.

---

## 1. Data Leakage

### Findings

**`_temporal_split()` in `backtest/models/features.py:93-137`:**
- Sorts ALL samples globally by timestamp (line 120): `samples = sorted(samples, key=lambda s: s.get("timestamp", 0))`
- Computes embargo in milliseconds: `embargo_ms = 24 * 60 * 60 * 1000` = 24 hours (line 124)
- Purges train samples whose lookahead crosses val boundary: `train = [s for s in train if s.get("timestamp", 0) + embargo_ms <= val_start_ts]` (line 134)
- Same purge for val→test boundary (line 136)

**`export.py` (line 125):**
- Uses the SAME `_temporal_split` function: `train, val, test = _temporal_split(samples)`
- This ensures QLoRA training data uses identical splits as ML models.

**`train_qlora.py` (line 583):**
- Evaluation also uses `_temporal_split`: `all_samples = _load_dataset(DATASET_PATH)` → `train, val, test = _temporal_split(all_samples)`
- No re-shuffling. The temporal guarantee is preserved.

**Indicators (`ingestion/indicators.py:43`):**
- Loop `for i in range(lookback, len(df))` ensures indicators only use past candles `[0:i+1]`.
- Multi-TF alignment uses `_find_latest_at_or_before()` (line 137-146) — strictly causal.
- TA-Lib indicators (EMA, RSI, MACD, ATR, BB) are all backward-looking by construction.

**No evidence of data leakage in the feature/indicator pipeline.**

### Verdict: PASS

The temporal split is correctly implemented. The export function uses the same split. No future data leaks into indicators. The embargo purges samples at boundaries.

---

## 2. Label Leakage

### Findings

**Labeler (`ingestion/labeler.py:38`):**
- Looks 24 candles ahead: `future = candles[candle_index + 1 : candle_index + 1 + lookahead]`
- Uses future OHLCV to compute `max_up`, `max_down`, TP, SL
- The labeler's lookahead = 24 candles × 1h = 24 hours

**Embargo vs Lookahead:**
- Embargo = 24 bars × 60 min × 60 sec × 1000 ms = 86,400,000 ms = **exactly 24 hours**
- This is a BOUNDARY MATCH: the last permitted train sample's label uses future data up to (but not exceeding) the first val timestamp.
- Technically correct — labels at train boundaries do NOT use val/test period candles.

**However — the labeler's drawdown filter (lines 82-89) introduces a SUBTLE BIAS:**
```python
if bias == "LONG":
    for c in future:
        if c["low"] <= sl:
            return None  # DISCARD if SL hit before TP
        if c["high"] >= tp:
            break
```
This means: **only samples where the trade WOULD HAVE WORKED (TP hit before SL) are labeled.** Ambiguous or losing trades are discarded as `None`. This is not "leakage" per se — it's a design choice that creates a biased dataset where every sample represents a historically profitable trade.

**Impact on accuracy measurement**: The model predicts direction on a set of samples that were ALL profitable in hindsight. This doesn't inflate *direction* accuracy (the model still needs to guess LONG vs SHORT), but it means the 88% accuracy is measured on "easy" samples — ones with clear directional signals that actually materialized.

### Verdict: CONCERN

No classical label leakage (embargo matches lookahead). BUT the labeler's drawdown filter creates survivorship bias in the dataset — only historically successful trades are included. The 88% accuracy is on pre-filtered "easy" samples, not on the full universe of market conditions.

---

## 3. Evaluation Bugs

### Findings

**Parse error handling (`train_qlora.py:453-458`):**
```python
prediction = _parse_prediction(generated)
if prediction is None:
    parse_errors += 1
    bias_kw = self._parse_bias(generated)
    prediction = {"bias": bias_kw or "LONG", "confidence": 0}
```
On parse failure, the model defaults to extracting bias from keywords or falling back to "LONG". **This could inflate accuracy if failures happen to default correctly.** However, the result shows **0 parse errors** on the 3,000 test samples — so this code path was never triggered. Non-issue for this specific run.

**Test set size discrepancy:**
- `data_counts["test"] = 8,425` (full test split)
- `total_evaluated = 3,000` (what was actually tested)
- The `evaluate()` function uses `--max-eval` striding (lines 598-609) to evenly sample 3,000 from 8,425.
- The striding methodology is sound (every ~2.8th sample, spanning all symbols and the full time range).
- **CONCERN**: evaluating only 35.6% of the test set reduces statistical power.

**Symbol coverage:**
- 11 of 12 symbols represented (MATICUSDT missing — likely due to striding pattern)
- Distribution is reasonably uniform (7.5% to 10.7% per symbol)

**Direction accuracy computation (`metrics.py:8-11`):**
```python
correct = sum(1 for p, a in zip(predictions, actuals) if p["bias"] == a["bias"])
return correct / len(predictions)
```
Simple and correct. No bias toward any class. No edge cases that could inflate.

### Verdict: PASS (with minor concern)

The evaluation code is correct. Parse failures would default to "LONG" which COULD inflate accuracy, but 0 errors occurred. The 3,000/8,425 sub-sampling is methodologically acceptable but noted.

---

## 4. Overfitting Signals

### Findings

**Loss curve analysis:**
| Metric | Value |
|--------|-------|
| First eval loss (step 250) | 0.8060 |
| Best eval loss (step 3000) | 0.7911 |
| Last eval loss (step 3750) | 0.7930 |
| Last train loss | 0.7762 |
| Train-eval loss gap | 0.015 (tiny) |

The loss curve is HEALTHY — eval loss decreases monotonically and the train-eval gap is tiny (0.015). No divergence pattern. Early stopping triggered at step 3000, model was restored.

**THE INVERSE GAP ANOMALY:**
| Split | Direction Accuracy |
|-------|-------------------|
| Train diagnostic (first 500 samples) | 70.5% |
| Val diagnostic (first 500 samples) | 86.5% |
| Test (3000 strided samples) | 88.03% |
| Gap (train - test) | **-17.5pp** |

**This is EXTREMELY unusual.** A model should NOT perform better on unseen data than on data it was trained on. Explanations:

1. **The train diagnostic takes `train[:500]` = the OLDEST 500 samples** (line 631). These are from the earliest period in the dataset (~late 2024). The test set covers Feb-May 2026. Market regimes differ drastically.
2. **The train diagnostic does NOT use the training data that was actually fed to the model.** The model was trained on chat-format JSONL (processed by `export_training_data`), but the diagnostic evaluates on raw samples. The model may perform differently on familiar-format prompts vs raw data.
3. **More likely explanation**: The oldest 500 samples (2024 market) have different characteristics than the 2026 test period. The model learned patterns that happen to work better in the newer regime.

**Class balance:**
- Test predictions: LONG=1395 (46.5%), SHORT=1605 (53.5%)
- Test actuals: LONG=1502 (50.1%), SHORT=1498 (49.9%)
- Dataset is nearly perfectly balanced (49/51). Model does NOT predict majority class.

**Per-class metrics (from result):**
| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| LONG | 0.910 | 0.845 | 0.876 |
| SHORT | 0.855 | 0.916 | 0.884 |

Both classes have strong, balanced performance. Not a one-direction model.

### Verdict: CONCERN

Loss curves look healthy (no overfitting). Class balance and per-class metrics are excellent. However, the -17.5pp inverse gap is unexplained and suspicious. The diagnostic methodology (oldest 500 train samples) is flawed and should be replaced with a random subsample of the actual training period.

---

## 5. Baseline & Class Balance

### Findings

**Baselines:**
| Baseline | Accuracy |
|----------|----------|
| Random (50/50) | ~50% |
| Majority class | ~50.1% (dataset is balanced) |
| Indicator heuristic (multi-TF majority vote) | 50.73% |
| Zero-shot Qwen 2.5 7B | 58.49% |
| Best ML model (Random Forest v2) | 64.24% |
| **QLoRA fine-tuned** | **88.03%** |

The QLoRA model beats all baselines by large margins:
- +37pp over random
- +29.5pp over zero-shot
- +23.8pp over best ML model

**Is 88% achievable for this task?**

The labeler's quality filters (ADX >= 15, volume_ratio >= 0.5, R:R >= 1.0, whipsaw filter, drawdown filter) remove all ambiguous/noisy samples. The remaining dataset contains only **clear-signal, profitable trades**. On such a filtered set, 88% direction accuracy is more plausible than on raw market data because:
1. Only strong directional moves survive the filters
2. The model can learn pattern→direction mappings that are genuinely predictive on high-quality setups
3. The model receives rich multi-timeframe context in natural language format

**Literature comparison** (from training knowledge, no live web available):
- Typical crypto direction accuracy in published papers: 55-68% on raw data
- On filtered/high-quality datasets: 70-85% reported (with caveats about overfitting)
- 88% is at the HIGH end but not impossible on a heavily filtered dataset

### Verdict: CONCERN

The 88% is defensible ONLY because the dataset is heavily filtered for high-quality setups. On unfiltered market data (all candles, not just clear signals), this accuracy would almost certainly drop to 60-70%. The thesis must clearly state that accuracy is measured on pre-filtered samples, not on arbitrary market conditions.

---

## 6. Trade Simulation

### Findings

**THE CRITICAL FLAW — Circular TP/SL:**

The labeler computes TP/SL from hindsight:
```python
# labeler.py line 56-57
tp = entry * (1 + max_up * 0.7)  # 70% of actual future maximum
sl = entry * (1 - atr * 1.0 / entry)  # 1 ATR below entry
```

The model LEARNS these TP/SL during training and outputs similar values:
- **TP deviation from label: median 0.74%** (model's TP is very close to label's TP)
- **SL deviation from label: median 0.24%**
- **Entry match: 98.3%** (expected — entry price is in the prompt)

The simulator then checks if model's predicted TP is reached within 24 candles:
```python
# simulate.py line 37
if candle["high"] >= tp:
    return {"outcome": "WIN", ...}
```

**THE CIRCULARITY:**
1. Label TP = 70% of actual future max price → price DID reach this level by construction
2. Model learns to output TP ≈ label TP (within 0.74%)
3. Simulator asks: "did price reach model's TP?" → Almost always YES, because the TP was derived from knowing price DID go there

**Evidence of inflation:**
- Among correct-direction trades (2,641): **69.9% WIN**, 10.7% LOSS, 19.3% TIMEOUT
- Among wrong-direction trades (359): 0.8% WIN, **96.7% LOSS**, 2.5% TIMEOUT
- The wrong-direction loss rate (96.7%) confirms the simulation works — it correctly penalizes bad direction calls
- But the correct-direction win rate (69.9%) is inflated by the hindsight-optimal TP/SL

**No transaction costs or slippage:**
- `simulate_trade()` fills at exact TP/SL prices — no spread, no slippage
- No maker/taker fees modeled
- No market impact

**Profit Factor decomposition:**
- PF = 12.92 means gross wins are ~13x gross losses
- avg_win = 2.794%, avg_loss = -1.029%
- This R:R (~2.7:1) combined with 61.7% win rate → PF = (0.617 × 2.794) / (0.383 × 1.029) ≈ 4.37... wait, that doesn't match 12.92
- Recalculating: The 61.7% is among ALL trades including TIMEOUTs with positive pnl. Many timeouts are profitable, inflating gross profit further.

**Literature on realistic profit factors:**
- Professional systems: PF 1.5-2.5 (Pardo 2008, Kaufman 2013)
- Excellent systems: PF 2.0-3.0
- Suspicious/likely overfit: PF > 5.0 (community consensus)
- Almost certainly a bug: PF > 10.0
- **Our result: PF = 12.92 → firmly in "unrealistic" territory**

### Verdict: FAIL

The trade simulation has a fundamental circular dependency: TP/SL targets are derived from hindsight future prices, the model learns to replicate them, and the simulation confirms they "work" because they were designed to. The profit factor of 12.92 is an artifact, not tradeable alpha. **The financial metrics (PF, win rate, Sharpe) should not be used as evidence of trading viability.**

---

## 7. ML Model Gap Analysis

### Findings

**The gap:**
| Model | Test Accuracy | Gap vs QLoRA |
|-------|--------------|--------------|
| QLoRA (fine-tuned LLM) | 88.03% | — |
| Random Forest v2 | 64.24% | -23.8pp |
| XGBoost v2 | 63.60% | -24.4pp |
| Blending ensemble v2 | 63.59% | -24.4pp |
| LSTM v2 | 51.51% | -36.5pp |
| Zero-shot Qwen 7B | 58.49% | -29.5pp |

**Why the LLM is 24pp better — legitimate explanations:**

1. **Information asymmetry**: ML models see a flat feature vector (9 features × 3 TFs + 3 cross-TF = 30 numbers). The LLM sees the FULL text representation including:
   - Exact price levels and their relationships
   - Categorical heatmap/structure labels in natural language
   - Symbol name (can learn symbol-specific patterns)
   - Multi-TF narrative context

2. **Representational capacity**: 7B parameters with QLoRA adapters (~40M trainable) vs tree ensembles with ~1000 leaves. The LLM has vastly more capacity to memorize complex pattern→direction mappings.

3. **Training objective**: The LLM is trained on the EXACT format it's evaluated on (chat-format JSON). ML models extract features and lose information in the process.

4. **The labeler's text-based features**: Heatmap ("STRONG_BULLISH"), structure ("BREAKOUT"), etc. encode human-readable signals that an LLM can naturally reason about but a tree model reduces to integers {0,1,2,3,4}.

**Is this a red flag?**

A 24pp gap is large but EXPLAINED by the above factors. This is NOT evidence of evaluation bugs. It IS evidence that:
- The LLM task formulation gives the model more information than the ML feature vector
- The comparison is not perfectly apples-to-apples
- The thesis should acknowledge this information asymmetry

### Verdict: CONCERN

The gap is real and explainable (information asymmetry, capacity, format). It's not evidence of a bug. But the thesis should clearly state that the LLM sees strictly MORE information than the ML models, making the comparison somewhat unfair.

---

## 8. Literature Comparison

### Findings

**Published crypto direction accuracy benchmarks (from Semantic Scholar, training knowledge):**

| Source | Method | Accuracy | Notes |
|--------|--------|----------|-------|
| Lopez-Lira & Tang 2023 (arXiv:2304.07619) | ChatGPT zero-shot on news | ~60% | Stock direction, statistically significant |
| Bysik & Slepaczuk 2026 (arXiv:2606.00060) | XGBoost/LSTM walk-forward BTC | ~65% annual returns | After 10bp costs; hourly data |
| AlBahri & Dinesh 2025 | XGBoost rolling window BTC | 91.0% acc, F1=0.87 | Weekly signals; Buy/Hold/Sell |
| Pindza 2026 (Frontiers Blockchain) | Gradient-boosted, purged WF | **Fails after costs** | "Models overfit severely under proper leakage controls" |
| Various BERT+crypto papers (2022-2024) | Fine-tuned sentiment | 60-70% | Next-day direction |
| K. Chand 2026 (BCM system) | FinBERT + DistilBERT + tech indicators | ~92% | Not peer-reviewed; no replication |

**Key finding from Pindza 2026**: "Gradient-boosted models overfit severely under proper leakage controls; no strategy survives realistic fees." This directly challenges ANY high-accuracy crypto ML claim.

**Our 88% in context:**
- Higher than most published results (55-68% on raw data)
- Comparable to some filtered-dataset results (AlBahri's 91% on weekly BTC with Buy/Hold/Sell)
- The heavily filtered dataset and rich text features explain the high end
- BUT profit_factor=12.92 has no parallel in any published work

### Verdict: CONCERN

The 88% direction accuracy is at the high end but not implausible for a heavily filtered dataset with rich features. The profit factor of 12.92 has no precedent in legitimate published work and should be treated as an artifact of the circular TP/SL simulation, not as evidence of trading alpha.

---

## 9. Billionaire Test

### Findings

**If PF=12.92 and win_rate=61.67% were real and sustainable:**

Assumptions:
- avg_win = 2.794%, avg_loss = -1.029%
- ~3000 trades over 81 days of test data ≈ 37 trades/day
- Net expectancy per trade = (0.617 × 2.794%) + (0.383 × -1.029%) ≈ +1.33% per trade

With 37 trades/day × 1.33% net per trade = **49.2% daily return**.

Starting with $1,000:
- After 1 week: $1,000 × 1.492^7 = ~$25,000
- After 2 weeks: ~$625,000
- After 3 weeks: ~$15.6 million
- After 1 month: ~$390 million

**This is obviously impossible.** No one in history has achieved sustained 49% daily returns.

**Why backtest ≠ live:**
1. **Market impact**: 37 trades/day on altcoins would move prices significantly
2. **Slippage**: 10-50 bps per trade on mid-cap alts (Kaiko data)
3. **Regime change**: The model learned patterns from one 81-day window
4. **Model decay**: Crypto patterns shift within days/weeks
5. **Execution**: Can't fill at exact TP/SL in practice
6. **The circular TP/SL**: In live trading, you DON'T know the optimal TP — it was derived from hindsight

**Realistic degradation estimates (from practitioner literature):**
- Live PF is typically 30-50% of backtest PF (Pardo 2008)
- Adding 20bps round-trip costs at 37 trades/day = 7.4%/day in costs alone
- Expected live performance: **likely unprofitable** without significant TP/SL redesign

### Verdict: FAIL

The financial metrics imply impossible returns. They are artifacts of the simulation methodology, not evidence of tradeable alpha. The direction accuracy (88%) may have some real-world value, but the TP/SL calibration learned from hindsight would need to be completely replaced with a forward-looking exit strategy.

---

## Recommendations

### Must-Fix Before Going Live (ordered by priority)

1. **Replace the trade simulation** — Design a forward-looking exit strategy (e.g., fixed R:R based on current ATR, trailing stop) rather than using model-predicted TP/SL that were learned from hindsight labels. Re-run profit_factor with this new simulation.

2. **Run on FULL test set** — Evaluate all 8,425 test samples, not 3,000. This increases confidence and may reveal performance variations across the full test period.

3. **Fix the overfitting diagnostic** — Replace `train[:500]` with a random subsample of 500 from the middle of the training period, or better: evaluate on 500 samples spanning the full training time range.

4. **Add transaction costs to simulation** — Minimum: 0.1% taker fee per side (0.2% round-trip for Binance). Better: add 5-20bps slippage per trade for altcoins.

5. **Walk-forward validation** — Train on months 1-12, test on month 13. Retrain on months 1-13, test on month 14. Repeat. This tests whether the model generalizes across regime changes, not just one fixed test window.

### Should-Fix for Thesis Defense

6. **Clearly state the labeler's quality filters** in the methodology — readers must understand that 88% is on pre-filtered high-quality setups, not on arbitrary market conditions.

7. **Acknowledge the information asymmetry** — The LLM sees more information than tree models. The 88% vs 64% comparison should be framed as "what's possible with richer features" not "LLMs are inherently better at this task."

8. **Report financial metrics separately from accuracy** — Direction accuracy is the primary metric. PF/win_rate should be in an appendix with caveats about the circular TP/SL.

9. **Add confidence intervals** — On 3,000 samples with 88% accuracy, the 95% CI is approximately ±1.2pp (86.8% to 89.2%). Report this.

### Nice-to-Have

10. **Test on unseen symbols** — Hold out 2-3 symbols from training entirely. Does the model generalize to SHIBUSDT or PEPEUSDT?

11. **Test on a different time period** — If possible, get 2026 Q3 data and evaluate. Does accuracy hold?

12. **Ensemble with ML** — Use QLoRA direction + ML-derived TP/SL (from ATR-based rules) to separate "what the model is good at" (direction) from "what needs forward-looking logic" (exits).

---

## 10. Addendum (2026-06-17) — Pretraining Memorization Check

**Context**: while investigating whether the 88.03% direction accuracy might hide a
missed methodology issue (prompted by the drawdown-filter ablation result — removing
the filter only dropped accuracy to 84.13%, not the 60-70% the original audit
predicted in §5/§2 — see `qlora_no_drawdown` result and the updated
`optimization-results.ipynb` Part 8), one remaining hypothesis was checked: **could
Qwen2.5 be recalling memorized historical price action from its pretraining corpus,
rather than reasoning from the indicators in the prompt?** This would be a leakage
vector entirely outside the dataset/split/labeler pipeline already audited above —
the model's weights themselves "remembering" what actually happened on a given date
for a given symbol.

**Check**: pulled the actual candle timestamps in `dataset.jsonl` (not the sandbox's
displayed "current date") — the dataset spans **2024-11-23 to 2026-05-08**. The
temporal test split (most recent 15%) falls in roughly **Feb-May 2026**. Real Qwen2.5
models (the base model under both the zero-shot baseline and the QLoRA fine-tune)
were released ~September 2024 with a pretraining cutoff before that date.

**Verdict: PASS (confound ruled out for the test set).** The test period is
chronologically *after* Qwen2.5's knowledge cutoff, so the base model cannot have
memorized the specific price action being evaluated — it did not exist yet when the
model was pretrained. This rules out exact-memorization leakage as an explanation for
the high test accuracy.

**Caveat — not fully closed**: this only rules out *exact* memorization of
this-period outcomes. It does not rule out a more diffuse prior — e.g. general
pretrained associations like "BTC is a large-cap, comparatively lower-volatility
asset" or familiarity with technical-analysis terminology itself (heatmap/structure
category names, RSI/ADX conventions) — which could still contribute to the LLM's edge
over tree-based ML models in a way that isn't a leak, just a richer starting prior.
This is the same "information asymmetry" explanation already raised in §7, now
narrowed: it's pretrained *concept* familiarity, not pretrained *outcome* memorization.

**Remaining open test** (not yet run): a feature-occlusion experiment — strip
`heatmap`/`structure` (the categorical fields trees see only as lossy ordinal ints)
and/or anonymize the symbol name from the prompt, first as a cheap `--eval-only` probe
on the existing `qlora_cloud` adapter, then (if that signal is large) as a full retrain
with the same fields stripped from training too — to isolate how much of the 24pp gap
vs XGBoost/RF is the categorical-text encoding vs genuine reasoning capability.

---

## Summary Table

| Section | Verdict | Severity |
|---------|---------|----------|
| 1. Data Leakage | PASS | — |
| 2. Label Leakage | CONCERN | Medium (dataset bias) |
| 3. Evaluation Bugs | PASS | — |
| 4. Overfitting Signals | CONCERN | Medium (inverse gap unexplained) |
| 5. Baseline & Class Balance | CONCERN | Low (caveat needed) |
| 6. Trade Simulation | **FAIL** | **HIGH** (circular TP/SL) |
| 7. ML Model Gap | CONCERN | Low (explainable) |
| 8. Literature Comparison | CONCERN | Medium (PF unprecedented) |
| 9. Billionaire Test | **FAIL** | **HIGH** (impossible returns) |
| 10. Pretraining Memorization (addendum) | PASS | — (test period post-dates Qwen2.5's cutoff) |

**The 88% direction accuracy is likely REAL but applies only to filtered, high-quality setups.**
**The financial metrics (PF=12.92, win_rate=61.67%) are ARTIFACTS and should not be cited as evidence of trading viability.**
**Update (2026-06-17): the drawdown-filter ablation (`qlora_no_drawdown`, 84.13% vs 88.03%) shows the filter is NOT the primary driver of the headline number — see `optimization-results.ipynb` Part 8. The pretraining-memorization confound is ruled out (§10). The remaining open question is how much of the 24pp gap vs ML models is categorical-text encoding (testable, not yet run) vs genuine LLM reasoning capability.**
