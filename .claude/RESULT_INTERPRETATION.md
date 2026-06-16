# Result Interpretation — No-Drawdown-Filter Experiment

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

## Two Evaluation Surfaces

The no-filter model can be read two ways — keep them distinct:

1. **No-filter model on the unfiltered test set** → the honest, real-world number.
   This is the primary output of `train_qlora.py --dataset-type no_filter`.
2. **No-filter model on the *filtered* test set** (optional, for Scenario D) → tests
   whether harder training generalizes back to clean data. To produce it, eval the
   saved adapters against the filtered dataset:
   `--eval-only backtest/data/models/qlora_no_drawdown_filter --dataset-type filtered`.

---

## Key Metrics to Compare

| Metric | Filtered model | No-filter model | Δ |
|--------|----------------|-----------------|---|
| Accuracy (unfiltered test) | — | **[fill]** | — |
| Accuracy (filtered test) | 88.03% | [optional, Scenario D] | Δ vs 88.03 |
| Improvement vs zero-shot | +29.5pp | **[fill]** | Δ improvement |
| Per-class F1 (LONG / SHORT) | 0.876 / 0.884 | [fill] | — |
| Per-symbol accuracy | (table) | [fill] | which symbols move most? |

---

## Other Metrics — How to Read Them Here

- **Profit factor / win rate (`metrics`).** Still inflated by the **circular TP/SL**
  (audit §6/§9): the model outputs hindsight-derived targets the simulator then
  "confirms." Removing the drawdown filter does **not** fix this. **Do not** cite
  PF/win-rate as trading viability. The non-circular **ATR simulation**
  (`metrics_atr`, `simulate_trade_atr`) is the fairer financial read — compare *that*
  across runs, not the circular PF.
- **McNemar.** Must be re-run **paired on the unfiltered test set** (re-run zero-shot
  there first — see EXPERIMENT doc §4 Step 4). A large drop in χ² means the advantage
  over zero-shot is less robust on real data.
- **Overfitting gap (`overfitting.gap` = train_acc − test_acc).** On filtered data it
  was a suspicious −17.5pp (audit §4). Watch whether unfiltered data normalizes it
  toward a small positive gap — that would itself partly explain the filtered anomaly.
- **Parse errors.** Filtered run had 0. An increase suggests the model struggles to
  emit valid JSON on noisier inputs.
- **Class balance.** Confirm the unfiltered dataset stays ~49/51 LONG/SHORT; a skew
  would change how to read accuracy vs the ~50% majority baseline.

---

## Questions the Experiment Should Answer

1. Does the advantage over zero-shot stay statistically significant (paired McNemar
   on unfiltered data)?
2. Which symbols lose the most accuracy when the filter is removed (e.g. high-vol
   DOGE vs BTC)?
3. Does the no-filter model do *relatively* better on volatile symbols?
4. Is the unfiltered dataset still class-balanced?

---

## Audit References

- §2 — drawdown filter survivorship bias ("trade WOULD HAVE WORKED").
- §5 — baseline table + "60–70% on unfiltered market data".
- §6 / §9 — circular TP/SL → financial metrics are artifacts (use ATR sim instead).
- §9 point 6 — recommendation to state the filters in the methodology.
- §9 point 2 — run on the full test set (8,425) for the final number, not just 3,000.
