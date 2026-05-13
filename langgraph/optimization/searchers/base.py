from abc import ABC, abstractmethod


class BaseSearcher(ABC):
    """Abstract base for hyperparameter search strategies."""

    @abstractmethod
    def search(self, cfg: dict, dataset_path: str) -> dict:
        """Run search and return a results dict in the standard output format."""
        ...
