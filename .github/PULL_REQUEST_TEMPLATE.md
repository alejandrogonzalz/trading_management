## Summary

<!-- What does this PR change and why? 1-3 bullets. -->

- 
- 

## Type of change

<!-- Check all that apply -->

- [ ] `feat` — new feature
- [ ] `fix` — bug fix
- [ ] `refactor` — code restructuring, no functional change
- [ ] `docs` — documentation only
- [ ] `ml` — model training, dataset, or evaluation change
- [ ] `data` — DVC-tracked file updated (dataset, candles, model weights)

## Component

- [ ] Backend (FastAPI / Binance)
- [ ] Frontend (React / Vite)
- [ ] LangGraph agent
- [ ] Backtest / ML pipeline
- [ ] QLoRA fine-tuning
- [ ] Infrastructure / CI

---

## ML / Data checklist (skip if not applicable)

- [ ] Temporal split used — no positional/per-symbol split
- [ ] `_temporal_split` called with `sort=True` (default) or data pre-sorted
- [ ] DVC pointer (`.dvc`) committed, not the raw model/dataset file
- [ ] `dvc push` run after adding new tracked files
- [ ] Result JSON saved to `optimization/results/` or `optimization/qlora/results/`
- [ ] `sample_keys` present in result JSON (required for `compare-stats` pairing)
- [ ] No stale accuracy numbers cited without ⚠️ annotation

## Test plan

<!-- What did you run to verify this works? -->

- [ ] `python -m pytest tests/ -v` passes
- [ ] Smoke test command: <!-- e.g. python -m cli run-backtest --provider mock -->
- [ ] Checked for regressions in: <!-- list affected features -->

---

## Result snapshot (ML PRs only)

| Model | Val acc | Test acc | Profit factor |
|-------|---------|----------|---------------|
|       |         |          |               |

Baseline (heuristic majority): <!-- e.g. 50.73% -->

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
