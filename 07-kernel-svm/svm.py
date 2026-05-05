import numpy as np
from qpsolvers import solve_qp
from TD.confusion_matrix import ConfusionMatrix
from TD.dataset import Dataset
from TD.kernel import Kernel

class SVM:
    """Support Vector Machine classifier using quadratic programming."""

    def __init__(self, dataset: Dataset, col_class: int, kernel: Kernel) -> None:
        """Initialize the SVM with dataset, target column, and kernel."""

        self.dataset: Dataset = dataset
        self.col_class: int = col_class
        self.kernel: Kernel = kernel
        self.alphas: np.ndarray | None = None
        self.beta_0: float = 0.0
        self.train_labels: np.ndarray = self.__get_labels()
        self.train_features: np.ndarray = self.__get_features()
        self.computed_kernel: np.ndarray = self.__compute_kernel_matrix()
        
    def __get_labels(self) -> np.ndarray:
        """Extract and transform labels into -1/+1."""
        
        raw_labels = self.dataset.get_column(self.col_class)
        labels = np.where(raw_labels == 1, 1, -1)
        return labels.astype(float)

    def __get_features(self) -> np.ndarray:
        """Extract features, removing the label column."""
        
        X = np.asarray([list(inst) for inst in self.dataset.get_instances()], dtype=float)
        return np.delete(X, self.col_class, axis=1)
        

    def __compute_kernel_matrix(self) -> np.ndarray:
        """Compute the full kernel matrix for the training data."""
        
        n = self.train_features.shape[0]
        K = np.zeros((n, n), dtype=float)

        for i in range(n):
            for j in range(n):
                K[i, j] = self.kernel.k(self.train_features[i], self.train_features[j])

        return K

    def train(self, C: float) -> None:
        """Train the SVM by solving the dual optimization problem."""
        
        n = self.train_features.shape[0]
        y = self.train_labels.astype(float)

        P = np.outer(y, y) * self.computed_kernel
        q = -np.ones(n, dtype=float)

        G = np.vstack((-np.eye(n), np.eye(n)))
        h = np.hstack((np.zeros(n), C * np.ones(n)))

        A = y.reshape(1, -1)
        b = np.array([0.0])

        P = 0.5 * (P + P.T)
        P = P + 1e-8 * np.eye(n)

        self.alphas = solve_qp(P, q, G, h, A, b, solver="cvxopt")

        self.alphas = np.asarray(self.alphas, dtype=float).reshape(-1)

        self.__compute_beta_0()

    def __compute_beta_0(self) -> None:
        """Compute the bias term (beta_0) from support vectors."""
        
        support_indices = np.where(self.alphas > 1e-5)[0]
        if len(support_indices) == 0:
            self.beta_0 = 0.0
            return

        values = []

        for i in support_indices:
            decision_without_bias = np.sum(self.alphas * self.train_labels * self.computed_kernel[:, i])
            values.append(self.train_labels[i] - decision_without_bias)

        self.beta_0 = float(np.mean(values))

    def predict(self, x: np.ndarray) -> int:
        """Predict the class (+1 or -1) for a given input vector."""
        
        decision_value = 0.0

        for i in range(self.train_features.shape[0]):
            decision_value += (self.alphas[i]* self.train_labels[i]* self.kernel.k(self.train_features[i], x))

        decision_value += self.beta_0
        return int(1 if decision_value >= 0 else -1)

    def test(self, test_dataset: Dataset) -> ConfusionMatrix:
        """Evaluate the SVM on a test set and produce a confusion matrix."""
        
        confusion_matrix = ConfusionMatrix()

        for instance in test_dataset.get_instances():
            true_raw_label = instance[self.col_class]
            true_label = 1 if true_raw_label == 1 else 0

            x = np.delete(instance, self.col_class)
            predicted_svm_label = self.predict(x)
            predicted_label = 1 if predicted_svm_label == 1 else 0

            confusion_matrix.add_prediction(true_label, predicted_label)

        return confusion_matrix


    @property
    def alphas_(self) -> np.ndarray:
        return self.alphas

    @property
    def beta_0_(self) -> float:
        return self.beta_0

    @property
    def kernel_type(self):
        return self.kernel.kernel_type

