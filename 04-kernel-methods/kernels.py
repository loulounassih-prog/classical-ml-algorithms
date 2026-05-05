import numpy as np
from typing import Self
from abc import ABC, abstractmethod
import math
from sklearn.neighbors import NearestNeighbors


class Kernel(ABC):
    """A class for kernel density estimation, which also stores the cloud of points.
    Attributes:
        d: int                 -- dimension of the ambient space
        data: list[np.ndarray] -- list of coordinates of the points (each of dimension self.d)
    """
    def __init__(self: Self, d: int, data: list[np.ndarray]):
        self.d = d
        self.data = data

    @abstractmethod
    def density(self: Self, x: np.ndarray) -> float:
        pass

class Radial(Kernel) :
    def __init__(self: Self, d: int, data: list[np.ndarray], bandwith: float):
        super().__init__(d, data)
        self.bandwidth = bandwith

    @abstractmethod
    def volume(self) -> float :
        pass

    @abstractmethod
    def profile(self, t:float) -> float :
        pass

    def density(self, x):
        res = 0
        n = len(self.data)
        for i in range(n) :
            dist = 0
            for j in range(self.d) :
                dist += (x[j] - self.data[i][j])**2
            dist = dist/(self.bandwidth**2)
            res += self.profile(dist)
        res = res/(n*((self.bandwidth)**(self.d))*(self.volume()))
        return res
    
class Flat(Radial) :
    def volume(self) -> float :
        return ((math.pi)**((self.d)/2))/math.gamma((self.d)/2+1)
    
    def profile(self, t: float) -> float :
        return 1 if t<=1 else 0
    
class Gaussian(Radial) :
    def volume(self) -> float :
        return (2*math.pi)**((self.d)/2)

    def profile(self, t: float) -> float :
        return math.exp(-t/2)
    
    def guess_bandwidth(self) -> None :
        X = np.asarray(self.data, dtype=float)
        n = len(X)
        mean = np.mean(X, axis=0)
        sigma = np.sqrt(np.sum((X - mean) ** 2) / (n - 1))
        self.bandwidth = sigma * ((n * (self.d + 2) / 4) ** (-1 / (self.d + 4)))
    
    
class Knn(Kernel):
    """A class for kernel density estimation with k-Nearest Neighbors
       derived from Kernel.
    Attributes not already in Kernel:
        k: int      -- parameter for k-NN
        V: float    -- "volume" constant appearing in density
        neigh:    sklearn.neighbors.NearestNeighbors   -- data structure and methods for efficient k-NN computations
    """
    def __init__(self: Self, d: int, data: list[np.ndarray], k: int, V: float):
        super().__init__(d,data)
        self.k, self.V = k, V
        self.neigh = NearestNeighbors(n_neighbors=self.k)
        self.fit_knn()

    def fit_knn(self):
        """Computes the inner data structure according to the data points."""
        self.neigh.fit(np.array(self.data))

    def knn(self, x: np.ndarray, vk:int):
        """The vk nearest-neighbors (vk can be different from self.k)."""
        return [
            np.array(self.data[i])
            for i in self.neigh.kneighbors([x], n_neighbors=vk)[1][0]
        ]

    def k_dist_knn(self, x: np.ndarray, vk: int) -> float:
        """The distance to vk-th nearest-neighbor."""
        return self.neigh.kneighbors([x], n_neighbors=vk)[0][0][vk-1]

    def density(self, x: np.ndarray) -> float :
        n = len(self.data)
        dk = self.k_dist_knn(x, self.k)
        return self.k/(2*n*self.V*dk)
    
    def meanshift(self, k: int) -> None:
        new_data = []
        for x in self.data:
            neighbors = self.knn(np.array(x), k)
            new_x = np.mean(np.array(neighbors), axis=0)
            new_data.append(new_x)
        self.data = new_data
        self.fit_knn()