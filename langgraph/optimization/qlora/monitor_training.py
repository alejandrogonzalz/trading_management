#!/usr/bin/env python3
"""Monitor QLoRA fine-tuning progress from any run_cloud.log.

Generic: infers total steps, checkpoint interval, and epoch count from the log.
Works for any Unsloth/HuggingFace tqdm training run.

Usage:
  python3 optimization/qlora/monitor_training.py
  python3 optimization/qlora/monitor_training.py --watch        # refresh every 30s
  python3 optimization/qlora/monitor_training.py --log path/to/log
  python3 optimization/qlora/monitor_training.py --epochs 3     # override epoch count
  python3 optimization/qlora/monitor_training.py --tz -6        # override timezone offset
"""

import argparse
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median

DEFAULT_LOG = Path(__file__).parent / "logs" / "run_cloud.log"
DEFAULT_TZ_OFFSET = -6  # UTC-6
GGUF_MINUTES = 30  # estimated post-training export time

# Warning thresholds
SPEED_SLOWDOWN_FACTOR = 1.5  # warn if speed > 1.5× median
STALL_MINUTES = 45  # warn if no new checkpoint in N minutes
LOSS_STAGNATION_ROUNDS = 3  # warn if train loss didn't drop in last N checkpoints
ETA_DRIFT_MINUTES = 45  # warn if ETA grew > N min vs first estimate

# Mirrors train_qlora.py's EarlyStoppingCallback(early_stopping_patience=4,
# early_stopping_threshold=0.001) so the monitor can show how close a run is
# to (or already past) the point where training stopped on its own.
EVAL_PLATEAU_ROUNDS = 4
EVAL_PLATEAU_THRESHOLD = 0.001

# tqdm progress bar: "25%|███| 1250/4904 [1:26:30<3:42:49, 3.66s/it]"
_STEP_RE = re.compile(
    r"(\d+)%"
    r"[^|]*\|[^|]*\|\s*"
    r"(\d+)/(\d+)"
    r"\s*\[([^<]+)<([^,\]]+),\s*([\d.]+)s/it\]"
)

_TS_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")

# HF trainer log line: {'loss': 1.34, 'grad_norm': 0.89, ..., 'epoch': 0.41}
_LOSS_RE = re.compile(
    r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]"
    r".*?\{'loss':\s*([\d.]+).*?'grad_norm':\s*([\d.]+).*?'epoch':\s*([\d.]+)"
)

# HF trainer eval line: {'eval_loss': 0.81, 'eval_runtime': 133.2, ..., 'epoch': 0.05}
_EVAL_LOSS_RE = re.compile(r"\{'eval_loss':\s*([\d.]+).*?'epoch':\s*([\d.]+)")

# train_qlora.py's own log lines around training/eval completion
_COMPLETED_RE = re.compile(r"Training completed in ([\d.]+)s")
_BEST_EVAL_RE = re.compile(r"Best eval loss:\s*([\d.]+)")
_TEST_PROGRESS_RE = re.compile(r"\[test\]\s+(\d+)/(\d+)\s+\((\d+) parse errors,\s*([\d.]+) samples/s, ETA (\d+)s\)")
_RESULT_ACC_RE = re.compile(r"RESULT: Test Accuracy = ([\d.]+)")
_RESULT_WR_RE = re.compile(r"Win Rate = ([\d.]+)\s+Profit Factor = ([\d.]+)")
_RESULT_GAP_RE = re.compile(r"Train = ([\d.]+)\s+Val = ([\d.]+)\s+Gap\(train-test\) = ([+\-\d.]+)")


def _parse_duration(s: str) -> timedelta:
    parts = s.strip().split(":")
    if len(parts) == 3:
        h, m, sec = int(parts[0]), int(parts[1]), float(parts[2])
    elif len(parts) == 2:
        h, m, sec = 0, int(parts[0]), float(parts[1])
    else:
        h, m, sec = 0, 0, float(parts[0])
    return timedelta(hours=h, minutes=m, seconds=sec)


def _fmt_td(td: timedelta) -> str:
    total = int(td.total_seconds())
    if total < 0:
        return "—"
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    return f"{m}m {s:02d}s"


def _eval_loss_plateau(raw_eval_losses):
    """Mirror EarlyStoppingCallback(patience, threshold): count consecutive
    evals without a >threshold improvement over the best eval_loss seen so far.
    """
    if not raw_eval_losses:
        return None
    ordered = sorted(raw_eval_losses, key=lambda x: x[0])
    best = ordered[0][1]
    counter = 0
    for _, val in ordered[1:]:
        if val < best - EVAL_PLATEAU_THRESHOLD:
            counter = 0
        else:
            counter += 1
        best = min(best, val)
    return {"best": best, "stagnant_rounds": counter}


def parse_log(log_path: Path, num_epochs: int = 1):
    # Read binary so Python's universal-newline mode doesn't convert \r → \n.
    # Each \n-line may embed \r-separated tqdm bar updates; the LAST segment
    # is the checkpoint step (HF writes a newline after each checkpoint save).
    text = log_path.read_bytes().decode("utf-8", errors="replace")

    # Collect all loss entries keyed by (step_approx based on epoch fraction)
    raw_losses = []  # list of (epoch_float, loss, grad_norm)
    for m in _LOSS_RE.finditer(text):
        raw_losses.append((float(m.group(4)), float(m.group(2)), float(m.group(3))))

    raw_eval_losses = []  # list of (epoch_float, eval_loss)
    for m in _EVAL_LOSS_RE.finditer(text):
        raw_eval_losses.append((float(m.group(2)), float(m.group(1))))

    # Parse checkpoint rows — detect checkpoint interval from step gaps
    best = {}  # step -> (ts, pct, total, elapsed, eta, speed)
    for raw_line in text.split("\n"):
        ts_m = _TS_RE.search(raw_line)
        if ts_m is None:
            continue
        ts = datetime.strptime(ts_m.group(1), "%Y-%m-%d %H:%M:%S")
        last_seg = raw_line.split("\r")[-1]
        m = _STEP_RE.search(last_seg)
        if not m:
            continue
        step = int(m.group(2))
        total = int(m.group(3))
        if step == 0:
            continue
        best[step] = (
            ts,
            int(m.group(1)),
            total,
            _parse_duration(m.group(4)),
            _parse_duration(m.group(5)),
            float(m.group(6)),
        )

    if not best:
        return [], {}

    # Infer checkpoint interval from gaps between seen steps
    sorted_steps = sorted(best)
    if len(sorted_steps) >= 2:
        gaps = [sorted_steps[i + 1] - sorted_steps[i] for i in range(len(sorted_steps) - 1)]
        ckpt_interval = int(median(gaps))
    else:
        ckpt_interval = sorted_steps[0]  # first checkpoint = interval

    # Keep only checkpoint-multiple steps
    rows = []
    total_steps = None
    for step in sorted_steps:
        if step % ckpt_interval != 0:
            continue
        ts, pct, total, elapsed, eta, speed = best[step]
        total_steps = total

        # Match nearest loss/eval_loss entry by epoch fraction. HF's logged
        # 'epoch' is fractional progress through ONE epoch, but `total` here
        # is the tqdm denominator across ALL epochs — dividing by `total`
        # alone understates the true epoch fraction by a factor of num_epochs.
        steps_per_epoch = total / num_epochs if total and num_epochs else total
        step_epoch = step / steps_per_epoch if steps_per_epoch else 0
        nearest = min(raw_losses, key=lambda x: abs(x[0] - step_epoch), default=None)
        loss = f"{nearest[1]:.4f}" if nearest else "—"
        grad = f"{nearest[2]:.3f}" if nearest else "—"

        nearest_eval = min(raw_eval_losses, key=lambda x: abs(x[0] - step_epoch), default=None)
        eval_loss = f"{nearest_eval[1]:.4f}" if nearest_eval else "—"

        rows.append(
            {
                "ts": ts,
                "step": step,
                "total": total,
                "pct": pct,
                "elapsed": elapsed,
                "eta": eta,
                "speed": speed,
                "loss": loss,
                "grad": grad,
                "eval_loss": eval_loss,
            }
        )

    completed_m = _COMPLETED_RE.search(text)
    best_eval_m = _BEST_EVAL_RE.findall(text)
    test_progress_m = _TEST_PROGRESS_RE.findall(text)

    test_progress = None
    if test_progress_m:
        done, tot, errs, sps, eta_s = test_progress_m[-1]
        test_progress = {
            "done": int(done),
            "total": int(tot),
            "parse_errors": int(errs),
            "samples_per_sec": float(sps),
            "eta_seconds": int(eta_s),
        }

    final_result = None
    result_acc_m = _RESULT_ACC_RE.search(text)
    if result_acc_m:
        wr_m = _RESULT_WR_RE.search(text)
        gap_m = _RESULT_GAP_RE.search(text)
        final_result = {
            "accuracy": float(result_acc_m.group(1)),
            "win_rate": float(wr_m.group(1)) if wr_m else None,
            "profit_factor": float(wr_m.group(2)) if wr_m else None,
            "train_acc": float(gap_m.group(1)) if gap_m else None,
            "val_acc": float(gap_m.group(2)) if gap_m else None,
            "gap": float(gap_m.group(3)) if gap_m else None,
        }

    meta = {
        "ckpt_interval": ckpt_interval,
        "total_steps": total_steps,
        "raw_losses": raw_losses,
        "raw_eval_losses": raw_eval_losses,
        "training_completed": completed_m is not None,
        "train_seconds": float(completed_m.group(1)) if completed_m else None,
        "best_eval_loss": float(best_eval_m[-1]) if best_eval_m else None,
        "test_progress": test_progress,
        "final_result": final_result,
    }
    return rows, meta


def _build_warnings(rows, meta, num_epochs, now_utc):
    warnings = []
    if not rows:
        return warnings

    last = rows[-1]
    speeds = [r["speed"] for r in rows]
    med_speed = median(speeds)

    # 1. Speed slowdown
    if last["speed"] > med_speed * SPEED_SLOWDOWN_FACTOR:
        warnings.append(
            f"SLOW  Current speed {last['speed']:.1f}s/it is "
            f"{last['speed'] / med_speed:.1f}× slower than median ({med_speed:.1f}s/it) "
            f"— GPU may be throttling or OOM swapping"
        )

    # 2. Training stall (no new checkpoint recently) — only meaningful while
    # still training; once training has completed this isn't a stall, it's
    # just sitting in the (often long) post-training test-set evaluation.
    if not meta.get("training_completed"):
        since_last = now_utc - last["ts"]
        if since_last.total_seconds() > STALL_MINUTES * 60:
            warnings.append(
                f"STALL  Last checkpoint was {_fmt_td(since_last)} ago "
                f"(threshold: {STALL_MINUTES} min) — process may have crashed"
            )

    # 3. Train-loss stagnation
    if len(rows) >= LOSS_STAGNATION_ROUNDS + 1:
        recent_losses = [r["loss"] for r in rows[-LOSS_STAGNATION_ROUNDS - 1 :]]
        numeric = [float(loss) for loss in recent_losses if loss != "—"]
        if len(numeric) == LOSS_STAGNATION_ROUNDS + 1 and numeric[-1] >= numeric[0] - 0.001:
            warnings.append(
                f"LOSS   No improvement in last {LOSS_STAGNATION_ROUNDS} checkpoints "
                f"({numeric[0]:.4f} → {numeric[-1]:.4f}) — possible plateau"
            )

    # 4. ETA drift vs first estimate
    if len(rows) >= 2 and not meta.get("training_completed"):
        first = rows[0]
        first_eta_abs = first["ts"] + first["eta"]
        last_eta_abs = last["ts"] + last["eta"]
        drift = (last_eta_abs - first_eta_abs).total_seconds() / 60
        if drift > ETA_DRIFT_MINUTES:
            warnings.append(
                f"DRIFT  ETA drifted +{drift:.0f} min vs first estimate — training is slower than initially projected"
            )

    # 5. Eval-loss plateau — mirrors the actual EarlyStoppingCallback trigger,
    # so this warning fires BEFORE training stops, not just after the fact.
    raw_eval_losses = meta.get("raw_eval_losses") or []
    if raw_eval_losses and not meta.get("training_completed"):
        plateau = _eval_loss_plateau(raw_eval_losses)
        if plateau["stagnant_rounds"] >= EVAL_PLATEAU_ROUNDS - 1:
            warnings.append(
                f"PLATEAU  eval_loss hasn't improved by >{EVAL_PLATEAU_THRESHOLD} for "
                f"{plateau['stagnant_rounds']}/{EVAL_PLATEAU_ROUNDS} consecutive evals "
                f"(best={plateau['best']:.4f}) — early stopping may trigger soon"
            )

    return warnings


def print_table(rows, meta, log_path: Path, num_epochs: int, tz_offset: int):
    TZ = timedelta(hours=tz_offset)
    tz_label = f"UTC{tz_offset:+d}"

    if not rows:
        print("  No checkpoint lines found yet — training may still be loading the model.")
        return

    last = rows[-1]
    total = last["total"]
    now_utc = datetime.now()
    now_local = now_utc + TZ

    steps_per_epoch = total // num_epochs if num_epochs else total
    current_epoch = last["step"] / steps_per_epoch if steps_per_epoch else 0

    warnings = _build_warnings(rows, meta, num_epochs, now_utc)

    header = (
        f"{'Step':>6}  {'%':>4}  {f'Timestamp ({tz_label})':>19}  "
        f"{'Elapsed':>8}  {'ETA':>8}  {f'ETA ({tz_label})':>16}  "
        f"{'Speed':>8}  {'Loss':>8}  {'EvalLoss':>8}  {'GradNorm':>9}"
    )
    sep = "─" * len(header)

    print(f"\n  QLoRA Training — {log_path.name}")
    print(
        f"  Checkpoint interval: every {meta['ckpt_interval']} steps  |  "
        f"Epochs: {num_epochs}  |  Steps/epoch: ~{steps_per_epoch}"
    )
    print(f"  {'─' * (len(header) - 2)}")
    print(f"  {header}")
    print(f"  {sep}")

    for r in rows:
        ts_local = r["ts"] + TZ
        eta_clock = (r["ts"] + r["eta"] + TZ).strftime("%m-%d %H:%M")
        print(
            f"  {r['step']:>6}  {r['pct']:>3}%  "
            f"{ts_local.strftime('%Y-%m-%d %H:%M:%S'):>19}  "
            f"{_fmt_td(r['elapsed']):>8}  "
            f"{_fmt_td(r['eta']):>8}  "
            f"{eta_clock:>16}  "
            f"{r['speed']:>6.2f}s/it  "
            f"{r['loss']:>8}  "
            f"{r['eval_loss']:>8}  "
            f"{r['grad']:>9}"
        )

    print(f"  {sep}")

    if meta.get("best_eval_loss") is not None:
        plateau = _eval_loss_plateau(meta["raw_eval_losses"])
        plateau_note = ""
        if plateau:
            plateau_note = (
                f"  (stagnant {plateau['stagnant_rounds']}/{EVAL_PLATEAU_ROUNDS} rounds vs early-stop patience)"
            )
        print(f"\n  Best eval_loss : {meta['best_eval_loss']:.4f}{plateau_note}")

    if meta.get("training_completed"):
        pct_of_plan = 100 * last["step"] / total if total else 0
        stop_kind = "full run" if last["step"] >= total else "early stop"
        mins = meta["train_seconds"] / 60 if meta.get("train_seconds") else None
        took = f", took {mins:.1f} min" if mins else ""
        print(
            f"  Status         : TRAINING COMPLETE ({stop_kind}) — stopped at step "
            f"{last['step']}/{total} ({pct_of_plan:.0f}% of plan, epoch {current_epoch:.2f}/{num_epochs}){took}"
        )

        final_result = meta.get("final_result")
        test_progress = meta.get("test_progress")
        if final_result:
            print("\n  FINAL RESULT")
            print(f"    Test Accuracy  : {final_result['accuracy']:.4f}")
            if final_result.get("win_rate") is not None:
                print(
                    f"    Win Rate       : {final_result['win_rate']:.4f}   "
                    f"Profit Factor: {final_result['profit_factor']:.3f}"
                )
            if final_result.get("gap") is not None:
                print(
                    f"    Train/Val/Test : {final_result['train_acc']:.4f} / "
                    f"{final_result['val_acc']:.4f} / {final_result['accuracy']:.4f}   "
                    f"Gap(train-test): {final_result['gap']:+.4f}"
                )
        elif test_progress:
            done, tot = test_progress["done"], test_progress["total"]
            pct = 100 * done / tot if tot else 0
            eta_min = test_progress["eta_seconds"] / 60
            print(
                f"\n  Final test eval: {done}/{tot} ({pct:.0f}%)  "
                f"{test_progress['samples_per_sec']:.2f} samples/s  "
                f"ETA ~{eta_min:.0f} min  parse_errors={test_progress['parse_errors']}"
            )
        else:
            print("\n  Final test evaluation starting / GGUF export pending...")
    else:
        epoch_etas = []
        for ep in range(1, num_epochs + 1):
            ep_step = steps_per_epoch * ep
            secs = (ep_step - last["step"]) * last["speed"]
            epoch_etas.append(last["ts"] + timedelta(seconds=secs) + TZ)

        remaining_full = epoch_etas[-1] + timedelta(minutes=GGUF_MINUTES) - now_local

        print(f"\n  Current : step {last['step']}/{total} ({last['pct']}%) @ {last['speed']:.2f}s/step")
        for ep, eta in enumerate(epoch_etas, 1):
            secs_left = (ep * steps_per_epoch - last["step"]) * last["speed"]
            label = "training ends" if ep == num_epochs else f"epoch {ep} ends"
            suffix = f"  ← {_fmt_td(eta - now_local)} left" if ep == num_epochs else ""
            print(
                f"  Epoch {ep} done  : ~{eta.strftime('%H:%M')} ({_fmt_td(timedelta(seconds=secs_left))} from now)  [{label}]{suffix}"
            )

        print(
            f"  + GGUF/eval   : ~{GGUF_MINUTES} min after → done "
            f"~{(epoch_etas[-1] + timedelta(minutes=GGUF_MINUTES)).strftime('%H:%M')}"
            f"  ← {_fmt_td(remaining_full)} left"
        )

    print(f"  Checked at    : {now_local.strftime('%Y-%m-%d %H:%M:%S')} ({tz_label})")

    if warnings:
        print(f"\n  {'─' * (len(sep) - 2)}")
        print("  ⚠  WARNINGS")
        print(f"  {'─' * (len(sep) - 2)}")
        for w in warnings:
            kind, msg = w.split("  ", 1)
            print(f"  [{kind}] {msg}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=str(DEFAULT_LOG), help="Path to log file")
    parser.add_argument("--watch", action="store_true", help="Refresh every 30s")
    parser.add_argument("--epochs", type=int, default=2, help="Total training epochs (default: 2)")
    parser.add_argument("--tz", type=int, default=DEFAULT_TZ_OFFSET, help="UTC offset, e.g. -6 (default: -6)")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Log not found: {log_path}")
        return

    if args.watch:
        try:
            while True:
                print("\033[2J\033[H", end="")
                rows, meta = parse_log(log_path, args.epochs)
                print_table(rows, meta, log_path, args.epochs, args.tz)
                print("  (Ctrl+C to stop — refreshes every 30s)")
                time.sleep(30)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        rows, meta = parse_log(log_path, args.epochs)
        print_table(rows, meta, log_path, args.epochs, args.tz)


if __name__ == "__main__":
    main()
