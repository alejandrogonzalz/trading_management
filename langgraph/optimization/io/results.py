import json
from pathlib import Path

import numpy as np


def _serialize(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def save_result(result: dict, output_dir: Path) -> Path:
    """Save optimization result JSON and return the path."""
    model = result["model"]
    out_path = Path(output_dir) / f"{model}_optimization.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=_serialize)
    return out_path


def load_all_results(results_dir: Path) -> dict:
    """Load all *_optimization.json files from results_dir."""
    data = {}
    for f in sorted(Path(results_dir).glob("*_optimization.json")):
        model = f.stem.replace("_optimization", "")
        with open(f) as fh:
            data[model] = json.load(fh)
    return data
