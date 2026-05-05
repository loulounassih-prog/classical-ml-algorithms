import numpy as np
from typing import Any

from TD.dataset import Dataset

class Regression:
    """
    Base Regression class for implementing regression models.
    """

    def __init__(self, dataset: Dataset, col_regr: int) -> None:
        """
        Initialize the Regression base class.
        
        Parameters:
            dataset (Dataset): The dataset used for regression.
            col_regr (int): The index of the target (regression) column.
        """
        self.m_dataset: Dataset = dataset
        self.m_col_regr: int = col_regr

    def estimate(self, x: np.ndarray) -> Any:
        """
        Estimate the target value for a given input feature vector.
        
        Parameters:
            x (np.ndarray): Feature vector (excluding the target column).
        
        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method")

    def get_col_regr(self) -> int:
        """
        Get the target regression column index.
        
        Returns:
            int: The target column index.
        """
        return self.m_col_regr

    def get_dataset(self) -> Dataset:
        """
        Get the dataset associated with this regression.
        
        Returns:
            Dataset: The dataset.
        """
        return self.m_dataset

