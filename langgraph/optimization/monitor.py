#!/usr/bin/env python3
"""Monitor running ensemble/optimization jobs.

Checks log files, result JSONs, and process status.

Usage:
    python optimization/monitor.py
    python optimization/monitor.py --watch 30   # refresh every 30s
"""

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def check_process():
    """Find running python optimization processes."""
    try:
        out = subprocess.check_output(["ps", "aux"], text=True)
        procs = [ln for ln in out.splitlines() if "python" in ln and "optim" in ln.lower() or "ensemble" in ln.lower()]
        return procs
    except Exception:
        return []


def check_logs():
    """Read last lines from log files."""
    logs = {}
    if not LOGS_DIR.exists():
        return logs
    for f in sorted(LOGS_DIR.glob("*.log")):
        try:
            lines = f.read_text().splitlines()
            logs[f.name] = {
                "size_kb": f.stat().st_size / 1024,
                "lines": len(lines),
                "last_5": lines[-5:] if lines else [],
                "modified": datetime.fromtimestamp(f.stat().st_mtime).strftime("%H:%M:%S"),
            }
        except Exception as e:
            logs[f.name] = {"error": str(e)}
    return logs


def check_results():
    """Scan result JSONs for completed models."""
    results = {}
    if not RESULTS_DIR.exists():
        return results
    for f in sorted(RESULTS_DIR.glob("*_optimization.json")):
        try:
            d = json.load(open(f))
            results[f.stem.replace("_optimization", "")] = {
                "best_score": d.get("best_score"),
                "test_acc": d.get("test_metrics", {}).get("accuracy"),
                "time": d.get("elapsed_seconds"),
                "timestamp": d.get("timestamp", "?"),
            }
        except Exception:
            pass
    return results


def print_status():
    """Print full status report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'═' * 60}")
    print(f"  OPTIMIZATION MONITOR — {now}")
    print(f"{'═' * 60}")

    # Processes
    procs = check_process()
    print(f"\n  Running processes: {len(procs)}")
    for p in procs[:5]:
        cols = p.split()
        pid, cpu, mem = cols[1], cols[2], cols[3]
        cmd = " ".join(cols[10:])[:60]
        print(f"    PID={pid} CPU={cpu}% MEM={mem}% {cmd}")

    if not procs:
        print("    (none found)")

    # Logs
    logs = check_logs()
    if logs:
        print("\n  Log files:")
        for name, info in logs.items():
            if "error" in info:
                print(f"    {name}: ERROR - {info['error']}")
                continue
            print(f"    {name} ({info['size_kb']:.1f}KB, {info['lines']} lines, last update: {info['modified']})")
            for line in info["last_5"]:
                print(f"      {line[-80:]}")
    else:
        print(f"\n  No log files in {LOGS_DIR}")

    # Results
    results = check_results()
    if results:
        print("\n  Completed models:")
        print(f"    {'Model':<25} {'Val Acc':<10} {'Test Acc':<10} {'Time':<8}")
        print(f"    {'─' * 53}")
        for name, info in sorted(results.items(), key=lambda x: x[1].get("test_acc") or 0, reverse=True):
            test = info.get("test_acc")
            val = info.get("best_score")
            t = info.get("time")
            print(f"    {name:<25} {val or 0:<10.4f} {test or 0:<10.4f} {t or 0:.0f}s")
    else:
        print(f"\n  No results yet in {RESULTS_DIR}")

    print(f"\n{'═' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="Monitor optimization jobs")
    parser.add_argument("--watch", type=int, default=0, help="Refresh interval in seconds (0=once)")
    args = parser.parse_args()

    if args.watch > 0:
        try:
            while True:
                os.system("clear" if os.name != "nt" else "cls")
                print_status()
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        print_status()


if __name__ == "__main__":
    main()
