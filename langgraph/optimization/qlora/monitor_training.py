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
LOSS_STAGNATION_ROUNDS = 3  # warn if loss didn't drop in last N checkpoints
ETA_DRIFT_MINUTES = 45  # warn if ETA grew > N min vs first estimate

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


def parse_log(log_path: Path):
    # Read binary so Python's universal-newline mode doesn't convert \r → \n.
    # Each \n-line may embed \r-separated tqdm bar updates; the LAST segment
    # is the checkpoint step (HF writes a newline after each checkpoint save).
    text = log_path.read_bytes().decode("utf-8", errors="replace")

    # Collect all loss entries keyed by (step_approx based on epoch fraction)
    raw_losses = []  # list of (epoch_float, loss, grad_norm)
    for m in _LOSS_RE.finditer(text):
        raw_losses.append((float(m.group(4)), float(m.group(2)), float(m.group(3))))

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

        # Match nearest loss entry by epoch fraction
        step_epoch = step / total if total else 0
        nearest = min(raw_losses, key=lambda x: abs(x[0] - step_epoch), default=None)
        loss = f"{nearest[1]:.4f}" if nearest else "—"
        grad = f"{nearest[2]:.3f}" if nearest else "—"

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
            }
        )

    meta = {
        "ckpt_interval": ckpt_interval,
        "total_steps": total_steps,
        "raw_losses": raw_losses,
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

    # 2. Training stall (no new checkpoint recently)
    since_last = now_utc - last["ts"]
    if since_last.total_seconds() > STALL_MINUTES * 60:
        warnings.append(
            f"STALL  Last checkpoint was {_fmt_td(since_last)} ago "
            f"(threshold: {STALL_MINUTES} min) — process may have crashed"
        )

    # 3. Loss stagnation
    if len(rows) >= LOSS_STAGNATION_ROUNDS + 1:
        recent_losses = [r["loss"] for r in rows[-LOSS_STAGNATION_ROUNDS - 1 :]]
        numeric = [float(loss) for loss in recent_losses if loss != "—"]
        if len(numeric) == LOSS_STAGNATION_ROUNDS + 1 and numeric[-1] >= numeric[0] - 0.001:
            warnings.append(
                f"LOSS   No improvement in last {LOSS_STAGNATION_ROUNDS} checkpoints "
                f"({numeric[0]:.4f} → {numeric[-1]:.4f}) — possible plateau"
            )

    # 4. ETA drift vs first estimate
    if len(rows) >= 2:
        first = rows[0]
        first_eta_abs = first["ts"] + first["eta"]
        last_eta_abs = last["ts"] + last["eta"]
        drift = (last_eta_abs - first_eta_abs).total_seconds() / 60
        if drift > ETA_DRIFT_MINUTES:
            warnings.append(
                f"DRIFT  ETA drifted +{drift:.0f} min vs first estimate — training is slower than initially projected"
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

    # Infer steps per epoch
    steps_per_epoch = total // num_epochs
    epoch_etas = []
    for ep in range(1, num_epochs + 1):
        ep_step = steps_per_epoch * ep
        secs = (ep_step - last["step"]) * last["speed"]
        epoch_etas.append(last["ts"] + timedelta(seconds=secs) + TZ)

    remaining_full = epoch_etas[-1] + timedelta(minutes=GGUF_MINUTES) - now_local

    # Warnings
    warnings = _build_warnings(rows, meta, num_epochs, now_utc)

    # Table header
    header = (
        f"{'Step':>6}  {'%':>4}  {f'Timestamp ({tz_label})':>19}  "
        f"{'Elapsed':>8}  {'ETA':>8}  {f'ETA ({tz_label})':>16}  "
        f"{'Speed':>8}  {'Loss':>8}  {'GradNorm':>9}"
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
            f"{r['grad']:>9}"
        )

    print(f"  {sep}")
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
                rows, meta = parse_log(log_path)
                print_table(rows, meta, log_path, args.epochs, args.tz)
                print("  (Ctrl+C to stop — refreshes every 30s)")
                time.sleep(30)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        rows, meta = parse_log(log_path)
        print_table(rows, meta, log_path, args.epochs, args.tz)


if __name__ == "__main__":
    main()
