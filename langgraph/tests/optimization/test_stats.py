"""Unit tests for optimization/stats_tests.py (pure-Python, no scipy needed)."""

import math

from optimization.stats_tests import (
    align_results,
    mcnemar,
    paired_ttest,
)


def test_mcnemar_balanced_discordant_not_significant():
    # 1 discordant each way → p-value should be 1.0 (no evidence of difference)
    a = [1, 1, 0, 0, 1]
    b = [1, 0, 1, 0, 1]
    res = mcnemar(a, b)
    assert res["b_a_right_b_wrong"] == 1
    assert res["c_a_wrong_b_right"] == 1
    assert res["n_discordant"] == 2
    assert res["p_value"] == 1.0
    assert res["significant_at_0.05"] is False


def test_mcnemar_strong_imbalance_is_significant():
    # 20 samples A-right/B-wrong vs 2 the other way → strongly significant
    a = [1] * 20 + [0] * 2
    b = [0] * 20 + [1] * 2
    res = mcnemar(a, b)
    assert res["b_a_right_b_wrong"] == 20
    assert res["c_a_wrong_b_right"] == 2
    assert res["p_value"] < 0.05
    assert res["significant_at_0.05"] is True


def test_mcnemar_no_discordant_pairs():
    a = [1, 1, 0, 0]
    b = [1, 1, 0, 0]
    res = mcnemar(a, b)
    assert res["n_discordant"] == 0
    assert res["p_value"] == 1.0


def test_paired_ttest_constant_positive_diff():
    # Every diff is +1, zero variance → deterministic significant result
    a = [2, 2, 2, 2]
    b = [1, 1, 1, 1]
    res = paired_ttest(a, b)
    assert res["mean_diff"] == 1.0
    assert res["t_statistic"] == math.inf
    assert res["p_value"] == 0.0
    assert res["significant_at_0.05"] is True


def test_paired_ttest_identical_not_significant():
    a = [1, 2, 3, 4, 5]
    b = [1, 2, 3, 4, 5]
    res = paired_ttest(a, b)
    assert res["mean_diff"] == 0.0
    assert res["p_value"] == 1.0
    assert res["significant_at_0.05"] is False


def test_paired_ttest_sign_and_bounds():
    a = [3, 1, 4, 1, 5, 9, 2, 6]
    b = [2, 0, 3, 0, 4, 8, 1, 5]  # a - b = +1 everywhere except keeps positive mean
    res = paired_ttest(a, b)
    assert res["mean_diff"] > 0
    assert res["t_statistic"] > 0
    assert 0.0 <= res["p_value"] <= 1.0


def test_align_results_by_sample_keys_reordered():
    res_a = {
        "predictions": [{"bias": "LONG"}, {"bias": "SHORT"}],
        "actuals": [{"bias": "LONG"}, {"bias": "LONG"}],
        "trade_results": [{"pnl_pct": 1.0}, {"pnl_pct": -1.0}],
        "sample_keys": ["X@1", "X@2"],
    }
    res_b = {  # same samples, reversed order
        "predictions": [{"bias": "SHORT"}, {"bias": "LONG"}],
        "actuals": [{"bias": "LONG"}, {"bias": "LONG"}],
        "trade_results": [{"pnl_pct": -2.0}, {"pnl_pct": 0.5}],
        "sample_keys": ["X@2", "X@1"],
    }
    ca, cb, va, vb = align_results(res_a, res_b)
    assert ca == [1, 0]   # A: LONG==LONG (✓), SHORT!=LONG (✗)
    assert cb == [1, 0]   # B aligned by key: LONG==LONG (✓), SHORT!=LONG (✗)
    assert va == [1.0, -1.0]
    assert vb == [0.5, -2.0]


def test_align_results_positional_fallback_when_no_keys():
    res_a = {
        "predictions": [{"bias": "LONG"}, {"bias": "LONG"}, {"bias": "SHORT"}],
        "actuals": [{"bias": "LONG"}, {"bias": "LONG"}, {"bias": "SHORT"}],
        "trade_results": [{"pnl_pct": 1.0}, {"pnl_pct": 1.0}, {"pnl_pct": 1.0}],
    }
    res_b = {  # shorter → truncates to 2
        "predictions": [{"bias": "LONG"}, {"bias": "SHORT"}],
        "actuals": [{"bias": "LONG"}, {"bias": "LONG"}],
        "trade_results": [{"pnl_pct": 0.0}, {"pnl_pct": 0.0}],
    }
    ca, cb, va, vb = align_results(res_a, res_b)
    assert len(ca) == len(cb) == 2
