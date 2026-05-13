"""OptimizerPipeline — orchestrates hyperparameter search, result saving, and plotting."""

import logging
from pathlib import Path

import yaml

from .io.plots import save_optimization_plot
from .io.results import save_result

log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_DATASET = (
    Path(__file__).resolve().parent.parent / "backtest" / "data" / "labeled" / "dataset.jsonl"
)


class OptimizerPipeline:
    """Run hyperparameter search for one model, then save results and plot."""

    def __init__(
        self,
        model: str,
        config_path: str,
        dataset_path: str = None,
        output_dir: str = None,
    ):
        self.model = model
        self.cfg = yaml.safe_load(Path(config_path).read_text())
        self.dataset_path = dataset_path or str(DEFAULT_DATASET)
        self.output_dir = Path(output_dir) if output_dir else RESULTS_DIR

    def run(self) -> dict:
        log.info(f"Model: {self.model} | Dataset: {self.dataset_path}")
        searcher = self._get_searcher()
        result = searcher.search(self.cfg, self.dataset_path)

        out_path = save_result(result, self.output_dir)
        log.info(f"Results saved to {out_path}")

        if result.get("all_results"):
            save_optimization_plot(result, self.output_dir / f"{self.model}_optimization.png")

        return result

    def _get_searcher(self):
        if self.model in ("xgboost", "random_forest"):
            from .searchers.sklearn_searcher import SklearnSearcher
            return SklearnSearcher(self.model)
        if self.model == "lstm":
            from .searchers.lstm_searcher import LSTMSearcher
            return LSTMSearcher()
        if self.model == "qlora":
            from .searchers.qlora_searcher import QLoRASearcher
            return QLoRASearcher()
        raise ValueError(f"Unknown model: {self.model}. Choose from: xgboost, random_forest, lstm, qlora")
