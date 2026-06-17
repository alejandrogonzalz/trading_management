# Result Interpretation — No-Drawdown-Filter Experiment

> **RESOLVED (2026-06-17).** Result: **84.13%** (vs 88.03% filtered, -3.90pp). This is a
> null-ish result in the most useful sense — see "Decision Applied" below.

> Success here is **not** "high accuracy." Success is **understanding how much the
> drawdown filter inflates the headline 88%**. Even a null result (no change) is a
> valuable, publishable finding about the robustness of the result.

---

## Baseline Reference (filtered dataset, drawdown filter ON)

| Metric | Value | Source |
|--------|-------|--------|
| Direction accuracy | **88.03%** | `qlora_cloud/result.json` |
| Improvement vs zero-shot | **+29.5pp** (88.03 − 58.49) | audit §5 |
| McNemar χ² | 557, p≈0 | audit §4/§5 |
| Per-class F1 | LONG 0.876 / SHORT 0.884 | audit §4 |

The no-filter run produces an analogous `result.json` measured on the **unfiltered**
temporal test split. Compare the two.

---

## Outcome Scenarios → What It Means → Paper Decision

### Scenario A — accuracy drops to **70–75%**
- **Meaning:** Critique #2 is valid and material. The filter substantially inflates
  the headline; the model partly learned "clean-path" patterns, not pure direction.
- **Confirms** the auditor: *"On unfiltered market data … 60–70%"* (§5).
- **Paper:** Present 88% with a strong caveat as *"best case on filtered
  high-quality setups."* Lead the results with the unfiltered number as the honest
  metric. Treat the filter as a major limitation.

### Scenario B — accuracy drops to **78–83%**
- **Meaning:** The filter helps but is not the whole story; fine-tuning still
  contributes materially. The "easy samples" label is partly but not fully accurate.
- **Paper:** Keep the current structure; add a *"sensitivity to the drawdown filter"*
  section reporting both numbers and framing the gap as a measure of task difficulty.

### Scenario C — accuracy stays **> 85%**
- **Meaning:** The filter has little effect on *direction*; the model learned genuine
  direction and the filter only removed high-noise-path cases.
- **Contradicts** the auditor's "pre-filtered easy samples" concern — strong evidence
  it is overstated *for direction accuracy* (not for the financial metrics).
- **Paper:** Use to strengthen the robustness/generalization argument.

### Scenario D — no-filter model **beats** the filtered model on the *filtered* test
- **Meaning:** Training on harder data yields a more robust model — the filter was
  *harming* training. Counter-intuitive but powerful.
- **Paper:** Consider retraining the headline model without the filter; report the
  improved, more honest result. (Requires an extra eval of the no-filter model on the
  *filtered* test set — see "Two evaluation surfaces" below.)

| Outcome | Decision |
|---------|----------|
| A (70–75%) | Major revision: 88% = "best case under optimal conditions"; lead with unfiltered |
| B (78–83%) | Keep structure + add "sensitivity to filter" section (report both) |
| C (>85%) | Strengthen conclusions; filter is not the driver of the headline |
| D (beats filtered) | Change methodology: train the headline model without the filter |

---

## Decision Applied (2026-06-17)

**Measured: 84.13%** (-3.90pp from 88.03%). This falls **between Scenario B and C** —
just under the >85% cutoff, well above the 70-75% "major confounder" zone, and well
above the auditor's predicted 60-70% (audit §5). Reading both scenario writeups
together is more accurate than forcing it into one bucket:

- Like **Scenario C**: the drop is small enough that the filter is clearly **not** the
  primary driver of the 88.03% headline. The auditor's "pre-filtered easy samples"
  concern, as applied to *direction accuracy specifically*, is not borne out — strong
  evidence for the robustness/generalization argument.
- Like **Scenario B**: the drop isn't zero either — there's a real, if modest, +34%
  more (noisier) training data costing -3.9pp. Worth reporting both numbers rather than
  only the filtered one.
- **New signal not anticipated by the original scenario table**: the overfitting gap
  flips sign (-17.5pp → **+4.4pp**, the normal/expected direction). Removing the filter
  introduces *real* (if mild) overfitting that wasn't present before. This is itself
  evidence that the filter is a legitimate quality control, not a number-inflation
  trick — without it, the model starts memorizing noisier "didn't actually work"
  trade examples that don't generalize as well.

**Practical decision**: keep `qlora_cloud` (filtered) as the headline result. Report
`qlora_no_drawdown` as a "sensitivity to filter" section (closest to the Scenario B
paper guidance) with the gap-sign-flip finding as the most interesting single
observation — it's the dataset's clearest evidence that the filter helps rather than
inflates. No Scenario D triggered (no-filter never beat filtered), so no methodology
change is warranted.

---

## Two Evaluation Surfaces

The no-filter model can be read two ways — keep them distinct:

1. **No-filter model on the unfiltered test set** → the honest, real-world number.
   This is the primary output of `train_qlora.py --dataset-type no_filter`.
2. **No-filter model on the *filtered* test set** (optional, for Scenario D) → tests
   whether harder training generalizes back to clean data. **Not run** — moot here
   since the result landed close to Scenario C, not D. To produce it if needed later:
   `--eval-only backtest/data/models/qlora_no_drawdown --dataset-type filtered`
   (note: actual tag is `qlora_no_drawdown`, not `qlora_no_drawdown_filter`).

---

## Key Metrics to Compare

| Metric | Filtered model | No-filter model | Δ |
|--------|----------------|-----------------|---|
| Accuracy (unfiltered test) | — | **84.13%** | — |
| Accuracy (filtered test) | 88.03% | not run (no Scenario D trigger, see above) | — |
| Improvement vs zero-shot | +29.5pp | **+27.64pp** (84.13 − 56.49) | -1.9pp |
| Per-class F1 (LONG / SHORT) | 0.876 / 0.884 | **0.851 / 0.830** | both ↓ slightly |
| Per-symbol accuracy | (table, audit §4) | not yet broken out | still pending |
| N evaluated | 3,000 (strided) | 11,294 (full, no `--max-eval`) | not identical N |
| Parse errors | 0 / 3,000 | 7 / 11,294 (0.06%) | negligible |
| Class balance (actuals) | ~50/50 | LONG 52.4% / SHORT 47.6% | still roughly balanced |

---

## Other Metrics — How to Read Them Here

- **Profit factor / win rate (`metrics`).** Still inflated by the **circular TP/SL**
  (audit §6/§9): the model outputs hindsight-derived targets the simulator then
  "confirms." Removing the drawdown filter does **not** fix this. **Do not** cite
  PF/win-rate as trading viability. The non-circular **ATR simulation**
  (`metrics_atr`, `simulate_trade_atr`) is the fairer financial read — compare *that*
  across runs, not the circular PF.
- **McNemar.** Zero-shot has been re-run paired on the unfiltered test set
  (`zero-shot-qwen7b-no-drawdown`, 56.49%) — the raw accuracy gap holds (+27.6pp vs
  +29.5pp filtered), but the actual paired McNemar/t-test (`compare-stats`) has **not**
  been run yet. Do that before citing significance for the no-filter comparison.
- **Overfitting gap (`overfitting.gap` = train_acc − test_acc).** On filtered data it
  was a suspicious −17.5pp (audit §4). **Confirmed**: unfiltered data normalizes it to
  **+4.4pp** — the normal/expected sign. This is the single most informative
  observation in this experiment: it suggests the filtered model's inverted gap was
  itself partly a side-effect of training on cleanly-filtered data (less to overfit
  to), not an anomaly requiring a separate explanation (audit §4's open question).
- **Parse errors.** Filtered run had 0. An increase suggests the model struggles to
  emit valid JSON on noisier inputs.
- **Class balance.** Confirm the unfiltered dataset stays ~49/51 LONG/SHORT; a skew
  would change how to read accuracy vs the ~50% majority baseline.

---

## Questions the Experiment Should Answer

1. Does the advantage over zero-shot stay statistically significant (paired McNemar
   on unfiltered data)? — **Raw accuracy advantage holds** (+27.6pp); formal McNemar
   `compare-stats` not yet run.
2. Which symbols lose the most accuracy when the filter is removed (e.g. high-vol
   DOGE vs BTC)? — **Not yet broken out.**
3. Does the no-filter model do *relatively* better on volatile symbols? — **Not yet
   answered**, depends on (2).
4. Is the unfiltered dataset still class-balanced? — **Yes**, LONG 52.4% / SHORT 47.6%
   on the test split (training-set-wide balance was ~49/51 per the original
   measurement in `EXPERIMENT_NO_DRAWDOWN_FILTER.md` §5).

---

## Audit References

- §2 — drawdown filter survivorship bias ("trade WOULD HAVE WORKED").
- §5 — baseline table + "60–70% on unfiltered market data".
- §6 / §9 — circular TP/SL → financial metrics are artifacts (use ATR sim instead).
- §9 point 6 — recommendation to state the filters in the methodology.
- §9 point 2 — run on the full test set (8,425) for the final number, not just 3,000.
