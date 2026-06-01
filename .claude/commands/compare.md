---
description: Compare two backtest result files side by side
---

Load two result JSONs and display a comparison table.

## Usage
`/compare <result1> <result2>`

Arguments are tags (filenames without .json) from `langgraph/backtest/data/results/`.

## Steps

1. Read both JSON files from `langgraph/backtest/data/results/{tag}.json`
2. Extract metrics from each: direction_accuracy, win_rate, profit_factor, sharpe_ratio, max_drawdown, avg_win, avg_loss
3. Print a markdown comparison table
4. Highlight which model wins on each metric
5. Give a 2-sentence summary of which is better for production use
