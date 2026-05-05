from sklearn.neighbors import KDTree
import numpy as np

from TD.classification import Classification
from TD.dataset import Dataset


class KnnClassification(Classification):
    """A k-NN classifier.

    Attributes:
        k: int          -- the number of nearest neighbors to use for classification
        kd_tree: KDTree -- the kd-tree used for computing shortest distances quickly
    """

    def __init__(self, k: int, dataset: Dataset, col_class: int):
        super().__init__(dataset, col_class)
        self.k = k
        X = np.delete(dataset.instances, col_class, axis=1)
        self.kd_tree = KDTree(X)

    def estimate(self, x: np.ndarray, threshold: float = 0.5) -> int:
        """Classify data point x for the given threshold."""
        x = np.asarray(x).reshape(1, -1)
        _, indices = self.kd_tree.query(x, k=self.k)
        neighbor_labels = self.dataset.instances[indices[0], self.col_class]
        p = np.mean(neighbor_labels)
        return int(p >= threshold) 
    
