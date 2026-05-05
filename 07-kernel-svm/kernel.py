import numpy as np
from enum import Enum

# Enumeration for different kernel types
class KernelType(Enum):
    LINEAR = 0
    POLY = 1
    RBF = 2
    SIGMOID = 3
    RATQUAD = 4

class Kernel:
    def __init__(self, kernel_type: KernelType, degree: int = 0, gamma: float = 0.0, coef0: float = 0.0) -> None:
        """
        Initializes a Kernel object with specific parameters.

        :param kernel_type: Type of the kernel (from KernelType enum)
        :param degree: Degree of the polynomial kernel
        :param gamma: Scale parameter for RBF, POLY, and SIGMOID kernels
        :param coef0: Independent term for POLY, SIGMOID, and RATQUAD kernels
        """
        self.kernel_type = kernel_type
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0


        if kernel_type == KernelType.LINEAR:
            self._kernel_fun = self._kernel_linear
        elif kernel_type == KernelType.POLY:
            self._kernel_fun = self._kernel_poly
        elif kernel_type == KernelType.RBF:
            self._kernel_fun = self._kernel_rbf
        elif kernel_type == KernelType.SIGMOID:
            self._kernel_fun = self._kernel_sigmoid
        elif kernel_type == KernelType.RATQUAD:
            self._kernel_fun = self._kernel_ratquad
        else:
            raise ValueError("Invalid kernel type")


    def k(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Computes the kernel function between x1 and x2.
        """
        return self._kernel_fun(x1, x2)

    def _kernel_linear(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Computes the linear kernel."""
        return float(np.dot(x1, x2))

    def _kernel_poly(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Computes the polynomial kernel."""
        return float((self.gamma * np.dot(x1, x2) + self.coef0) ** self.degree)

    def _kernel_rbf(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Computes the RBF (Gaussian) kernel."""
        diff = x1 - x2
        return float(np.exp(-self.gamma * np.dot(diff, diff)))

    def _kernel_sigmoid(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Computes the sigmoid kernel."""
        return float(np.tanh(self.gamma * np.dot(x1, x2) + self.coef0))

    def _kernel_ratquad(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Computes the rational quadratic kernel."""
        c = self.coef0 if self.coef0 != 0 else 1
        diff = x1 - x2
        squared_distance = np.dot(diff, diff)
        return float(c / (squared_distance + c))

    def get_kernel_type(self) -> KernelType:
        """Returns the kernel type."""
        return self.kernel_type

