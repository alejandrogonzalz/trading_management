# Master Action Plan

**Date**: April 30, 2026
**Project**: Trading Management System — Maestría en IA Aplicada, Tec de Monterrey
**Author**: Alejandro González Almazán

---

## Overview

Prioritized action plan for the academic project: fine-tuning an LLM for crypto technical analysis. The existing trading platform is the infrastructure — the academic work is the fine-tuning pipeline, evaluation, and comparison.

**Related docs**:
- [Backtest Framework](./backtest-framework.md) — evaluation system design
- [Fine-Tuning Strategy](./fine-tuning-strategy.md) — training data and model training

---

## Phase 1: Backtest Framework (Week 1)

**Goal**: Build the evaluation infrastructure so we can measure any model.

| # | Task | Est. | Deliverable |
|---|------|------|-------------|
| 1.1 | Implement `OpenAICompatibleProvider` in `llm_factory.py` | 0.5 day | Cloud LLM providers working with existing tests |
| 1.2 | Create `backtest/fetch_candles.py` — Binance public API downloader | 1 day | 6 months of 1h+4h candles for 20 pairs |
| 1.3 | Create `backtest/calculate_indicators.py` — batch wrapper around `indicator_service` | 0.5 day | Indicators for all downloaded candles |
| 1.4 | Create `backtest/label_data.py` — hindsight labeling with ATR thresholds | 1.5 days | Labeled JSONL dataset |
| 1.5 | Create `backtest/metrics.py` + `backtest/report.py` | 1 day | Metric calculations + terminal/chart output |
| 1.6 | Create `cli.py` with `fetch-candles`, `prepare-dataset` commands | 0.5 day | CLI entry point |

**Dependencies**: None — Binance public API needs no key, Groq free tier for testing.

**Validation**: Run `prepare-dataset` end-to-end, inspect labeled JSONL output. Verify class balance, filter rates, temporal split.

---

## Phase 2: Data Pipeline & Baseline Capture (Week 2)

**Goal**: Generate the training dataset and capture the zero-shot baseline.

| # | Task | Est. | Deliverable |
|---|------|------|-------------|
| 2.1 | Create `backtest/run_backtest.py` — feed dataset to LLM, simulate trades | 1.5 days | Backtest runner |
| 2.2 | Create `backtest/compare.py` — side-by-side comparison | 1 day | Comparison framework |
| 2.3 | Add `run-backtest` and `compare` CLI commands | 0.5 day | Full CLI |
| 2.4 | Run baseline backtest with Groq `llama-3.3-70b-versatile` | 0.5 day | `results/baseline-groq.json` |
| 2.5 | Run baseline backtest with DeepSeek `deepseek-chat` | 0.5 day | `results/baseline-deepseek.json` |
| 2.6 | Compare two baselines (validates the comparison framework) | 0.5 day | First comparison report |
| 2.7 | Construct JSONL training data from labeled dataset | 1 day | `train.jsonl`, `val.jsonl`, `test.jsonl` |

**Dependencies**: Phase 1 complete.

**Validation**: Two baseline comparison reports. Training JSONL passes format validation (parseable JSON, correct fields, temporal split verified).

---

## Phase 3: Fine-Tuning Execution (Weeks 3-4)

**Goal**: Train the model via both local QLoRA and Together AI.

| # | Task | Est. | Deliverable |
|---|------|------|-------------|
| 3.1 | Set up Unsloth environment, verify GPU access | 0.5 day | Working QLoRA setup |
| 3.2 | First QLoRA training run (1 epoch, default hyperparams) | 0.5 day | Initial model + loss curve |
| 3.3 | Hyperparameter search (lr, LoRA rank, epochs) | 2 days | Best hyperparameters identified |
| 3.4 | Final QLoRA training + export to GGUF for Ollama | 1 day | `trading-analyst-local` model in Ollama |
| 3.5 | Upload dataset to Together AI, launch fine-tuning | 0.5 day | Together AI job running |
| 3.6 | Monitor Together AI training, iterate if needed | 1-2 days | `alexglz/Qwen2.5-7B-Instruct-trading-v1` |

**Dependencies**: Phase 2 complete (training JSONL ready).

**Validation**: Both models produce valid JSON output on 10 test examples. Training loss curves show convergence without overfitting.

---

## Phase 4: Evaluation & Comparison (Week 5)

**Goal**: Rigorous comparison of baseline vs fine-tuned models.

| # | Task | Est. | Deliverable |
|---|------|------|-------------|
| 4.1 | Run test set through all 3 models (base, QLoRA, Together) | 1 day | Prediction files for each |
| 4.2 | Calculate classification metrics (accuracy, precision, recall, F1) | 0.5 day | Classification report |
| 4.3 | Run backtest simulation on all predictions | 0.5 day | Win rate, profit factor, equity curves |
| 4.4 | Confidence calibration analysis (reliability diagrams) | 0.5 day | Calibration plots |
| 4.5 | Statistical significance tests (McNemar's, paired t-test) | 0.5 day | p-values |
| 4.6 | Error analysis — 20 examples where fine-tuned model was wrong | 1 day | Failure mode categorization |
| 4.7 | Generate all charts (confusion matrix, equity curve, loss curve, per-pair breakdown) | 1 day | matplotlib outputs |

**Dependencies**: Phase 3 complete (trained models available).

**Validation**: All metrics computed, statistical tests show significance (or document why not). Comparison report generated.

---

## Phase 5: Integration & Demo (Week 6)

**Goal**: Load best model into production system, demonstrate improvement.

| # | Task | Est. | Deliverable |
|---|------|------|-------------|
| 5.1 | Load best model in Ollama, update `.env` | 0.5 day | Fine-tuned model serving |
| 5.2 | A/B comparison in live system (scanner with both models) | 1 day | Side-by-side output comparison |
| 5.3 | Record demo video (3-5 min) showing pipeline and results | 1 day | Demo video |
| 5.4 | Clean up code, remove API keys, add READMEs | 1 day | Clean repository |
| 5.5 | Write technical report (methodology, results, conclusions) | 3 days | Thesis chapter / report |
| 5.6 | Prepare presentation slides | 1 day | Final presentation |

**Dependencies**: Phase 4 complete.

---

## Dependency Graph

```
Phase 1: Backtest Framework
    │
    ▼
Phase 2: Data Pipeline & Baseline ──────────────┐
    │                                            │
    ▼                                            ▼
Phase 3: Fine-Tuning ──────────────────▶ Phase 4: Evaluation
                                            │
                                            ▼
                                     Phase 5: Integration & Demo
```

Phases 1-2 can be tested immediately with free-tier cloud models. Phase 3 needs the dataset. Phase 4 needs trained models. Phase 5 needs evaluation results.

---

## What Can Be Done NOW

| Task | Why Now |
|---|---|
| Provider changes (Phase 1.1) | No dependencies |
| Candle downloader (Phase 1.2) | Binance public API, no key |
| Indicator batch calculation (Phase 1.3) | Reuses existing `indicator_service` |
| Labeling algorithm (Phase 1.4) | Pure logic, no LLM needed |
| Unsloth environment setup (Phase 3.1) | No dependencies |
| Together AI account setup | No dependencies |

---

## What Needs the Fine-Tuned Model

| Task | Blocked By |
|---|---|
| Phase 4 evaluation | Phase 3 training |
| Phase 5 integration | Phase 4 results |
| DPO (advanced, optional) | Working SFT model + backtest results |

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|---|---|---|
| Dataset too small (< 1,500) | Fine-tune doesn't converge | Add more pairs, extend time range |
| Overfitting | Good on train, bad on test | Early stopping, validation monitoring, reduce LoRA rank |
| Market regime change | Model trained in bull fails in bear | Include diverse market conditions in training data |
| Together AI pricing changes | Budget exceeded | Fireworks AI as backup (same API) |
| Fine-tuned model worse than base | Negative result | Valid academically — document why |
| GPU issues (local training) | Can't train locally | Together AI as primary, local as comparison |

---

## Timeline Summary

```
Week 1:  Phase 1 — Backtest framework
Week 2:  Phase 2 — Data pipeline + baseline capture
Week 3:  Phase 3 — QLoRA training (local)
Week 4:  Phase 3 — Together AI training + hyperparameter iteration
Week 5:  Phase 4 — Full evaluation
Week 6:  Phase 5 — Integration, demo, documentation
Week 7:  Buffer — iteration, polish, presentation prep
```

**Critical path**: ~4.5 weeks. **Buffer**: ~2.5 weeks for iteration if initial results are disappointing.
