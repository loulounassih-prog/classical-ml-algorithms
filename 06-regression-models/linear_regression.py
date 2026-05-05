import numpy as np
from typing import Optional, Tuple
from TD.regression import Regression
from TD.dataset import Dataset


class LinearRegression(Regression):
    """
    Linear Regression class using the least squares method.
    """

    def __init__(self, dataset: Dataset, col_regr: int, fit_intercept: bool = True) -> None:
        """
        Initialize the LinearRegression instance.

        Parameters:
            dataset (Dataset): The dataset containing the samples.
            col_regr (int): The index of the target (regression) column.
            fit_intercept (bool): Whether to include an intercept term.
        """
        super().__init__(dataset, col_regr)
        self.m_beta: Optional[np.ndarray] = None  # Regression coefficients
        self.fit_intercept: bool = fit_intercept  # Flag for intercept term
        # Compute the coefficients using the implemented methods.
        self.set_coefficients()

    def construct_matrix(self) -> np.ndarray:
        """
        Construct the design matrix (feature matrix) from the dataset.

        Returns:
            np.ndarray: The design matrix X.
        """
        d = self.m_dataset.get_dim()
        assert d > 1, "Dataset must have more than one dimension."
        X = np.delete(self.m_dataset.data, self.m_col_regr, axis=1)
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        return X

    def construct_y(self) -> np.ndarray:
        """
        Construct the target vector from the dataset.

        Returns:
            np.ndarray: The target vector y.
        """
        d = self.m_dataset.get_dim()
        assert d > 1, "Dataset must have more than one dimension."
        return self.m_dataset.data[:, self.m_col_regr]

    def set_coefficients(self) -> None:
        """
        Compute and set the regression coefficients using the least squares solution.
        """
        X = self.construct_matrix()
        y = self.construct_y()
        self.m_beta = np.linalg.lstsq(X, y, rcond=None)[0]

    def get_coefficients(self) -> Optional[np.ndarray]:
        """
        Get a copy of the computed regression coefficients.

        Returns:
            Optional[np.ndarray]: The regression coefficients if computed, else None.
        """
        return self.m_beta.copy() if self.m_beta is not None else None

    def show_coefficients(self) -> None:
        """
        Print the regression coefficients.
        """
        if self.m_beta is None:
            print("Coefficients not computed.")
            return

        expected_size = (self.m_dataset.get_dim() if self.fit_intercept
                         else self.m_dataset.get_dim() - 1)
        if self.m_beta.size != expected_size:
            print(f"Warning: Coefficients size mismatch. "
                  f"Expected {expected_size}, got {self.m_beta.size}")

        print("beta = (" + " ".join(map(str, self.m_beta)) + ")")

    def print_raw_coefficients(self) -> None:
        """
        Print the regression coefficients in raw format.
        """
        if self.m_beta is None:
            print("{}")
            return
        print("{ " + ", ".join(map(str, self.m_beta)) + " }")

    def sum_of_squares(self, dataset: Dataset) -> Tuple[float, float, float]:
        """
        Compute the Total Sum of Squares (TSS), Explained Sum of Squares (ESS),
        and Residual Sum of Squares (RSS) for the given dataset.

        Parameters:
            dataset (Dataset): The dataset to compute the sums of squares.

        Returns:
            Tuple[float, float, float]: A tuple containing (TSS, ESS, RSS).
        """
        assert dataset.get_dim() == self.m_dataset.get_dim(), \
        "Datasets must have the same dimensions."
        y = dataset.data[:, self.m_col_regr]
        X = np.delete(dataset.data, self.m_col_regr, axis=1)
        y_hat = np.array([self.estimate(X[i]) for i in range(X.shape[0])])
        y_bar = np.mean(y)

        tss = float(np.sum((y - y_bar) ** 2))
        ess = float(np.sum((y_hat - y_bar) ** 2))
        rss = float(np.sum((y - y_hat) ** 2))

        return tss, ess, rss

    def estimate(self, x: np.ndarray) -> float:
        """
        Estimate the target value for a given input feature vector.

        Parameters:
            x (np.ndarray): Feature vector (excluding the target column).

        Returns:
            float: The predicted target value.
        """
        assert x.size == self.m_dataset.get_dim() - 1, \
        "Input dimension does not match expected feature size."
        assert self.m_beta is not None, "Coefficients not computed."
        if self.fit_intercept:
            return float(self.m_beta[0] + np.dot(x, self.m_beta[1:]))
        return float(np.dot(x, self.m_beta))
