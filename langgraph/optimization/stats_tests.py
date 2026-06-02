#!/usr/bin/env python3
"""Statistical significance tests for model comparison (thesis validation).

Implements the two tests required by the research design:
  - McNemar's test  — paired nominal test on per-sample correctness
                      (did model A and model B get the SAME samples right?)
  - Paired t-test   — on a per-sample continuous metric (e.g. trade PnL%)

Both are computed in pure Python (no scipy/statsmodels dependency) so the
module runs in any environment:
  - McNemar uses the exact binomial p-value (sum of binomial tail mass).
  - The paired t-test uses the normal approximation for the p-value, which is
    essentially exact for the large test sets used here (df in the thousands).

Pairing is the crux of a valid comparison: both models MUST be evaluated on the
SAME test samples in the SAME order. Results that carry a ``sample_keys`` list
are aligned by key intersection; otherwise alignment falls back to positional
(truncating to the shorter array) with a warning.

Usage:
    python optimization/stats_tests.py --a results/qlora.json --b results/ml-lstm.json
    # or via the CLI:  python -m cli compare-stats --a ... --b ...
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Core statistics (pure Python)
# --------------------------------------------------------------------------- #
def mcnemar(correct_a: List[int], correct_b: List[int]) -> Dict[str, Any]:
    """McNemar's test on paired binary correctness vectors.

    correct_a / correct_b: lists of 0/1, aligned sample-by-sample.
    Returns the discordant counts, the continuity-corrected chi-square
    statistic, and the exact two-sided binomial p-value.
    """
    if len(correct_a) != len(correct_b):
        raise ValueError("correctness vectors must be the same length")

    # b: A right, B wrong;  c: A wrong, B right (discordant pairs)
    b = sum(1 for a, bb in zip(correct_a, correct_b) if a == 1 and bb == 0)
    c = sum(1 for a, bb in zip(correct_a, correct_b) if a == 0 and bb == 1)
    n = b + c

    # Continuity-corrected chi-square statistic (df=1), reported for reference.
    chi2 = ((abs(b - c) - 1) ** 2) / n if n > 0 else 0.0

    # Exact two-sided binomial p-value under H0: P(discordant favours A) = 0.5.
    p_value = _binom_two_sided(min(b, c), n) if n > 0 else 1.0

    return {
        "b_a_right_b_wrong": b,
        "c_a_wrong_b_right": c,
        "n_discordant": n,
        "chi2_corrected": round(chi2, 4),
        "p_value": round(p_value, 6),
        "significant_at_0.05": p_value < 0.05,
    }


def _binom_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial p-value for k successes in n trials (p=0.5)."""
    if n == 0:
        return 1.0
    # one-sided tail mass at the smaller count, doubled and capped at 1.0
    tail = sum(math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(0, k + 1))
    return min(1.0, 2.0 * tail)


def paired_ttest(values_a: List[float], values_b: List[float]) -> Dict[str, Any]:
    """Paired t-test on a per-sample continuous metric (e.g. PnL%).

    p-value uses the normal approximation (exact in the large-df limit).
    """
    if len(values_a) != len(values_b):
        raise ValueError("value vectors must be the same length")
    n = len(values_a)
    if n < 2:
        return {"t_statistic": 0.0, "df": max(0, n - 1), "p_value": 1.0,
                "mean_diff": 0.0, "significant_at_0.05": False}

    diffs = [a - b for a, b in zip(values_a, values_b)]
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    sd_d = math.sqrt(var_d)

    if sd_d == 0:
        # No variance in the differences: deterministic result.
        t = math.inf if mean_d != 0 else 0.0
        p = 0.0 if mean_d != 0 else 1.0
    else:
        se = sd_d / math.sqrt(n)
        t = mean_d / se
        # Two-sided p-value via standard-normal survival (df large → t ≈ z).
        p = 2.0 * (1.0 - _normal_cdf(abs(t)))

    return {
        "t_statistic": round(t, 4) if math.isfinite(t) else t,
        "df": n - 1,
        "p_value": round(p, 6),
        "mean_diff": round(mean_d, 6),
        "significant_at_0.05": p < 0.05,
    }


def _normal_cdf(x: float) -> float:
    """Standard-normal CDF via the error function (pure Python)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# --------------------------------------------------------------------------- #
# Result-file alignment
# --------------------------------------------------------------------------- #
def _bias_of(pred: Any) -> Optional[str]:
    """Extract bias from a prediction entry (dict or raw string)."""
    if isinstance(pred, dict):
        b = str(pred.get("bias", "")).upper()
    else:
        b = str(pred).upper()
    return b if b in ("LONG", "SHORT") else None


def _pnl_of(trade: Any) -> float:
    return float(trade.get("pnl_pct", 0.0)) if isinstance(trade, dict) else 0.0


def align_results(
    res_a: Dict[str, Any], res_b: Dict[str, Any]
) -> Tuple[List[int], List[int], List[float], List[float]]:
    """Align two result dicts sample-by-sample.

    Prefers alignment by ``sample_keys`` intersection; falls back to positional
    (shorter length) with a printed warning. Returns four parallel lists:
    correctness_a, correctness_b, pnl_a, pnl_b.
    """
    pa, aa, ta = res_a["predictions"], res_a["actuals"], res_a.get("trade_results", [])
    pb, ab, tb = res_b["predictions"], res_b["actuals"], res_b.get("trade_results", [])
    ka, kb = res_a.get("sample_keys"), res_b.get("sample_keys")

    if ka and kb and len(ka) == len(pa) and len(kb) == len(pb):
        idx_b = {k: i for i, k in enumerate(kb)}
        pairs = [(i, idx_b[k]) for i, k in enumerate(ka) if k in idx_b]
        order = [(i, j) for i, j in pairs]
    else:
        print("⚠️  No usable sample_keys on both results — falling back to "
              "positional alignment (truncating to the shorter array). "
              "Pairing validity is NOT guaranteed.")
        m = min(len(pa), len(pb))
        order = [(i, i) for i in range(m)]

    ca, cb, va, vb = [], [], [], []
    for i, j in order:
        bias_a, bias_b = _bias_of(pa[i]), _bias_of(pb[j])
        actual = _bias_of(aa[i])  # actuals are label dicts with a "bias" field
        if actual is None or bias_a is None or bias_b is None:
            continue
        ca.append(1 if bias_a == actual else 0)
        cb.append(1 if bias_b == actual else 0)
        va.append(_pnl_of(ta[i]) if i < len(ta) else 0.0)
        vb.append(_pnl_of(tb[j]) if j < len(tb) else 0.0)

    return ca, cb, va, vb


# --------------------------------------------------------------------------- #
# Top-level comparison
# --------------------------------------------------------------------------- #
def compare(path_a: str, path_b: str) -> Dict[str, Any]:
    """Load two result JSONs and run McNemar + paired t-test."""
    res_a = json.loads(Path(path_a).read_text())
    res_b = json.loads(Path(path_b).read_text())

    name_a = res_a.get("model") or res_a.get("tag") or Path(path_a).stem
    name_b = res_b.get("model") or res_b.get("tag") or Path(path_b).stem

    ca, cb, va, vb = align_results(res_a, res_b)
    n = len(ca)
    acc_a = sum(ca) / n if n else 0.0
    acc_b = sum(cb) / n if n else 0.0

    mc = mcnemar(ca, cb)
    tt = paired_ttest(va, vb)

    summary = {
        "model_a": name_a,
        "model_b": name_b,
        "n_paired_samples": n,
        "accuracy_a": round(acc_a, 4),
        "accuracy_b": round(acc_b, 4),
        "mcnemar": mc,
        "paired_ttest_pnl": tt,
    }
    _print_summary(summary)
    return summary


def _print_summary(s: Dict[str, Any]) -> None:
    print(f"\n{'=' * 64}")
    print(f"  Statistical Comparison: {s['model_a']}  vs  {s['model_b']}")
    print(f"{'=' * 64}")
    print(f"  Paired samples:    {s['n_paired_samples']}")
    print(f"  Accuracy {s['model_a']:<16}: {s['accuracy_a']:.4f}")
    print(f"  Accuracy {s['model_b']:<16}: {s['accuracy_b']:.4f}")
    mc = s["mcnemar"]
    print(f"\n  McNemar's test (direction correctness)")
    print(f"    A right / B wrong: {mc['b_a_right_b_wrong']}   "
          f"A wrong / B right: {mc['c_a_wrong_b_right']}")
    print(f"    chi2 (corrected):  {mc['chi2_corrected']}")
    print(f"    p-value (exact):   {mc['p_value']}   "
          f"{'SIGNIFICANT' if mc['significant_at_0.05'] else 'not significant'} (α=0.05)")
    tt = s["paired_ttest_pnl"]
    print(f"\n  Paired t-test (per-sample PnL%)")
    print(f"    mean diff (A−B):   {tt['mean_diff']}")
    print(f"    t = {tt['t_statistic']}  (df={tt['df']})   p = {tt['p_value']}   "
          f"{'SIGNIFICANT' if tt['significant_at_0.05'] else 'not significant'} (α=0.05)")
    print(f"{'=' * 64}\n")


def main():
    parser = argparse.ArgumentParser(description="Statistical model comparison (McNemar + paired t-test)")
    parser.add_argument("--a", required=True, help="Path to result JSON for model A")
    parser.add_argument("--b", required=True, help="Path to result JSON for model B")
    parser.add_argument("--out", help="Optional path to write the summary JSON")
    args = parser.parse_args()

    summary = compare(args.a, args.b)
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"Summary written to {args.out}")


if __name__ == "__main__":
    main()
