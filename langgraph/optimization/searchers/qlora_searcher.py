import itertools
import random
from datetime import datetime

from .base import BaseSearcher


class QLoRASearcher(BaseSearcher):
    """Prints recommended configs for manual Unsloth training — no automated search."""

    def search(self, cfg: dict, dataset_path: str) -> dict:
        recommended = cfg.get("recommended_configs", [])
        if not recommended:
            grid = cfg["param_grid"]
            all_combos = list(itertools.product(*grid.values()))
            keys = list(grid.keys())
            random.seed(42)
            recommended = [dict(zip(keys, c)) for c in random.sample(all_combos, min(5, len(all_combos)))]

        print("\n" + "=" * 70)
        print("QLoRA — Recommended configs for Unsloth training")
        print("=" * 70)
        for i, c in enumerate(recommended, 1):
            print(f"\nConfig {i}:")
            for k, v in c.items():
                print(f"  {k}: {v}")
        print("\n" + "=" * 70)
        print("Run each config manually with Unsloth and record accuracy.")
        print("Save results to results/qlora_optimization.json in the standard format.")
        print("=" * 70 + "\n")

        return {
            "model": "qlora",
            "search_method": "manual",
            "recommended_configs": recommended,
            "best_params": None,
            "best_score": None,
            "all_results": [],
            "timestamp": datetime.now().isoformat(),
            "note": "Run each config manually with Unsloth. Update this file with results.",
        }
