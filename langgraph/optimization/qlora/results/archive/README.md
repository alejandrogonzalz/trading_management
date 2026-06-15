# Archived QLoRA results — DO NOT USE FOR THE THESIS

These result JSONs come from the original 3-config sweep and are **invalid**.
They are kept only for traceability of how the audit was reached.

## Why they are invalid

1. **Contaminated split (data leakage).** They were produced with the old
   `_temporal_split`, which cut the symbol-grouped `dataset.jsonl` *by position*,
   not by time. The "test" set was therefore a per-symbol slice, not a temporal
   holdout. See `docs/AUDITORIA_QLORA_LEAKAGE_OVERFITTING.md`.
2. **Partial, single-symbol evaluation.** The runs passed `--max-eval 2000`, so
   only the first 2000 test rows were scored — **100% LINKUSDT**. Not representative.
3. **Financial metrics are zero.** `win_rate` / `profit_factor` / `Sharpe` came out
   as 0 because the candles were never `dvc pull`ed on SageMaker, so trade
   simulation had no data.

The headline **92% direction accuracy** (`qlora_config1.json` /
`qlora_optimization.json`) is an optimistic number measured under leakage on a
single asset. It does **not** demonstrate generalization and must not be cited.

## What replaces them

A single re-trained model (cloud, optionally one local) on the **fixed** strict
temporal split + embargo, evaluated on the **full** test set, with the new
overfitting diagnostics (train/val/test accuracy + gap, loss curve, heuristic
baseline). See `optimization/qlora/run_cloud.sh` / `run_local.sh` and
`docs/GUIA_IMPLEMENTACION_FIX_QLORA.md`.

## Related stale artifact

`backtest/data/models/qlora_config1.dvc` points at the adapters/GGUF for the same
invalid config-1 run. Treat that model as **superseded** — re-train before using
it in any comparison or LangGraph integration.
