from abc import ABC, abstractmethod
import numpy as np

from TD.dataset import Dataset


class Classification(ABC):
    """An abstract class for defining classifiers.

    Attributes:
        dataset: Dataset -- the dataset classified by the classifier
        col_class: int   -- the index of the column to classify
    """

    def __init__(self, dataset: Dataset, col_class: int):
        super().__init__()
        self.dataset = dataset
        self.col_class = col_class

    @abstractmethod
    def estimate(self, x: np.ndarray, threshold: float = 0.5) -> int:
        """Classify data point x for the given threshold."""
        pass
