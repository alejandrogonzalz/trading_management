# No-Drawdown-Filter Experiment — Start-to-Finish Guide

A single, linear, copy-paste guide to run the experiment on a fresh GPU box and turn
the result into thesis text. For the *why* and deeper internals see
`../.claude/EXPERIMENT_NO_DRAWDOWN_FILTER.md`; for the audit see `AUDIT_QLORA_88PCT.md`.

---

## TL;DR

The labeled dataset is built with a **drawdown filter** that keeps only trades that
hit take-profit before stop-loss — i.e. only setups that *worked in hindsight*. That
inflates the reported **88.03%** accuracy. This experiment rebuilds the dataset
**without** that filter and re-trains the same model, to see the honest accuracy on
noisier, real-world-like data.

- Measured: the unfiltered dataset is **75,289 samples vs 56,161 (+34.1%)**.
- Cost/time: **~$6–8, ~4–5 h** on a RunPod A100 80GB.
- Model tag: **`qlora_no_drawdown_filter`**.

---

## Prerequisites

- A GPU box: **RunPod A100 80GB** with image `unsloth/unsloth:latest` (recommended),
  or any GPU instance that already has the unsloth/torch stack.
- **AWS credentials** with read access to `s3://trading-management-dvc/` (for `dvc pull`).
- Optional: an **Anthropic API key** if you want to debug with `claude` on the box.

---

## Step 1 — Set credentials (env vars)

```bash
export AWS_ACCESS_KEY_ID=AKIA...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1
export ANTHROPIC_API_KEY=sk-ant-...     # optional, for `claude` debugging
```

## Step 2 — Launch (one command)

From the repo's `langgraph/` directory:

```bash
bash optimization/qlora/user_data.sh
```

That bootstrap does **everything**: health checks → installs AWS CLI → `dvc pull`
candles → installs Claude Code → clones/updates the repo → then starts the experiment
(builds the unfiltered dataset → 3-step smoke test → full training run).

> **Already on a provisioned box** (repo + stack + candles present)? Skip the
> bootstrap and run the launcher directly:
> ```bash
> dvc pull backtest/data/candles
> nohup bash optimization/qlora/run_experiment_no_drawdown_filter.sh \
>   > optimization/qlora/logs/qlora_no_drawdown_filter.log 2>&1 & echo "PID: $!"
> ```
> Validate the box first without committing to the long run:
> `bash optimization/qlora/run_experiment_no_drawdown_filter.sh --smoke-only`

## Step 3 — Monitor

```bash
tail -f optimization/qlora/logs/qlora_no_drawdown_filter.log   # progress + loss
watch -n 10 nvidia-smi                                         # GPU utilization
```

**What healthy looks like:** dataset build prints `~75,289` samples; the smoke test
passes 3 steps in under a minute; training loss falls from ~1.8 toward ~0.5 over
~2 epochs; eval runs at the end; a `result.json` is written.

## Step 4 — Read the result

```bash
python3 - <<'PY'
import json
r = json.load(open("optimization/qlora/results/qlora_no_drawdown_filter/result.json"))
m, ov, b = r["metrics"], r["overfitting"], r.get("baseline_metrics", {})
atr = r.get("metrics_atr", {})
print(f"Direction accuracy : {m.get('direction_accuracy', r['test_metrics']['accuracy']):.4f}")
print(f"Heuristic baseline : {b.get('direction_accuracy', 0):.4f}")
print(f"Overfitting gap    : {ov.get('gap')}  (train {ov.get('train_acc')}, test {ov.get('test_acc')})")
print(f"Win rate / PF      : {m.get('win_rate',0):.4f} / {m.get('profit_factor',0):.2f}   <-- CIRCULAR, see note")
print(f"ATR-sim win / PF   : {atr.get('win_rate',0):.4f} / {atr.get('profit_factor',0):.2f}   <-- use THIS for finance")
print(f"Samples evaluated  : {r['test_metrics'].get('total_evaluated')}")
PY
```

> The `win_rate`/`profit_factor` in `metrics` are inflated by the circular TP/SL
> (audit §6) — **do not cite them**. Use `metrics_atr` (forward-looking ATR exits)
> as the fair financial read. The headline metric for this experiment is
> **direction accuracy**.

## Step 5 — Paired comparison vs zero-shot

The unfiltered test set differs from the filtered one, so re-run zero-shot on the
**same** unfiltered split before comparing (so McNemar is paired):

```bash
python3 -m cli run-backtest \
  --dataset backtest/data/labeled/dataset_no_drawdown_filter.jsonl \
  --provider ollama --split test --tag zero-shot-no-filter

python3 -m cli compare-stats \
  --a optimization/qlora/results/qlora_no_drawdown_filter/result.json \
  --b backtest/data/results/zero-shot-no-filter.json
```

## Step 6 — Interpret + write it up

Fill this in and apply the decision table in `../.claude/RESULT_INTERPRETATION.md`:

| Metric | Filtered baseline | No-filter (this run) |
|--------|-------------------|----------------------|
| Direction accuracy | 88.03% | **____** |
| Improvement vs zero-shot | +29.5pp | **____** |
| Overfitting gap (train − test) | −17.5pp | **____** |

- Accuracy **70–75%** → filter is a major confounder; lead the thesis with the
  unfiltered number, present 88% as "best case on filtered setups."
- Accuracy **78–83%** → add a "sensitivity to the drawdown filter" section, report both.
- Accuracy **>85%** → robust; the filter isn't what drives the headline.
- No-filter **beats** filtered on the filtered test → consider training the headline
  model without the filter.

## Step 7 — Save the model

```bash
# from the repo root (DVC root), not from langgraph/
cd ..
dvc add langgraph/backtest/data/models/qlora_no_drawdown_filter
dvc add langgraph/backtest/data/labeled/dataset_no_drawdown_filter.jsonl
dvc push
git add langgraph/backtest/data/models/qlora_no_drawdown_filter.dvc \
        langgraph/backtest/data/labeled/dataset_no_drawdown_filter.jsonl.dvc
git commit -m "data(qlora): no-drawdown-filter model + dataset (DVC)"
```

---

## Troubleshooting (top 5)

| Symptom | Fix |
|---------|-----|
| Bootstrap stops at "AWS credentials" | export the 3 `AWS_*` env vars (Step 1) |
| `dvc pull` fails | `aws sts get-caller-identity` — bad/expired creds or wrong region |
| Smoke test OOM on step 0 | lower `--batch-size`; `nvidia-smi` → `kill -9` a zombie process |
| `win_rate = 0` in result | candles missing: `dvc pull backtest/data/candles` |
| Run died mid-training | `bash run_cloud.sh --tag qlora_no_drawdown_filter --dataset-type no_filter --resume` |

Full failure→fix tree: `../.claude/EXPERIMENT_NO_DRAWDOWN_FILTER.md` §7 (Agent Runbook).

---

## File map

| File | Purpose |
|------|---------|
| `langgraph/optimization/qlora/user_data.sh` | Bare instance → running (bootstrap) |
| `langgraph/optimization/qlora/run_experiment_no_drawdown_filter.sh` | Preflight → dataset → smoke → full run |
| `langgraph/optimization/qlora/run_cloud.sh` | Generic trainer (called with the tag + `--dataset-type no_filter`) |
| `langgraph/cli.py` `prepare-dataset --no-drawdown-filter` | Builds the unfiltered dataset |
| `.claude/EXPERIMENT_NO_DRAWDOWN_FILTER.md` | Plan, agent runbook, measured numbers |
| `.claude/RESULT_INTERPRETATION.md` | Outcome → thesis-framing decision table |
| `docs/AUDIT_QLORA_88PCT.md` | The audit this experiment addresses (§2, §5) |
