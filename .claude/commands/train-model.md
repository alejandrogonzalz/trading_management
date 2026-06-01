---
description: Launch hyperparameter optimization for a model
---

Run the optimization pipeline. Results are saved to `optimization/results/`.

## Usage
`/train-model <model>` where model is `xgboost`, `random_forest`, or `lstm`

## Steps

1. Identify the config file: `langgraph/optimization/configs/{model}.yaml`
2. Launch optimization:
```bash
cd langgraph && OMP_NUM_THREADS=1 .venv/bin/python optimization/optimize.py \
  --model $MODEL --config optimization/configs/$MODEL.yaml
```

3. If the user wants it in background:
```bash
cd langgraph && mkdir -p logs && \
  nohup .venv/bin/python optimization/optimize.py \
  --model $MODEL --config optimization/configs/$MODEL.yaml \
  > logs/${MODEL}_optimization.log 2>&1 &
echo "PID: $!"
```

4. Report best score and params when complete.
