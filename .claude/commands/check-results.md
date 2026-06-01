---
description: Quick summary of all ML experiment results on disk
---

Scan result JSONs and print a ranked summary table.

## Usage
`/check-results`

## Steps

1. Scan both result directories:
```bash
cd langgraph && .venv/bin/python -c "
import json
from pathlib import Path

print('=== Backtest Results ===')
print(f'{\"Tag\":<30} {\"Acc\":>6} {\"WinRate\":>8} {\"Sharpe\":>7} {\"PF\":>6} {\"Samples\":>8}')
print('-' * 70)
for f in sorted(Path('backtest/data/results').glob('*.json')):
    if 'comparison' in f.name:
        continue
    d = json.load(open(f))
    m = d.get('metrics', {})
    if not m:
        continue
    print(f'{f.stem:<30} {m.get(\"direction_accuracy\",0):>6.4f} {m.get(\"win_rate\",0):>8.4f} {m.get(\"sharpe_ratio\",0):>7.2f} {m.get(\"profit_factor\",0):>6.2f} {d.get(\"total_samples\",0):>8}')

print()
print('=== Optimization Results ===')
for f in sorted(Path('optimization/results').glob('*_optimization*.json')):
    d = json.load(open(f))
    score = d.get('best_score')
    if score:
        print(f'{f.stem:<35} best={score:.4f}  params={d.get(\"best_params\",{})}')
"
```

2. Report the summary to the user.
