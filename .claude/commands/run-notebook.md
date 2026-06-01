---
description: Execute a Jupyter notebook end-to-end and verify all cells ran
---

Execute a notebook sequentially using nbconvert. Useful before committing or submitting.

## Usage
`/run-notebook <notebook_path>`

## Steps

1. Verify the notebook exists
2. Execute with nbconvert:
```bash
cd langgraph && caffeinate -dims .venv/bin/python -m jupyter nbconvert \
  --to notebook --execute \
  --ExecutePreprocessor.timeout=1800 \
  --ExecutePreprocessor.kernel_name=python3 \
  "$NOTEBOOK" --output "$NOTEBOOK"
```

3. Verify execution succeeded:
```bash
python3 -c "
import json, sys
nb = json.load(open('$NOTEBOOK'))
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
empty = [i for i, c in enumerate(code_cells) if not c.get('outputs')]
if empty:
    print(f'WARNING: {len(empty)} cells without output: {empty[:5]}')
    sys.exit(1)
print(f'OK: {len(code_cells)} code cells all executed')
"
```

4. Report success or which cells failed.
