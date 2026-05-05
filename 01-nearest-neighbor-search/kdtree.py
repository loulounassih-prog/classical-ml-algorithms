"""
kdtree module
"""

from dataclasses import dataclass
from typing import Self

import numpy as np
from TD.nearest_neighbor import NearestNeighborSearch


def median(X: np.ndarray, start: int, stop: int, c: int) -> float:
    """
    Returns median of array X between indices start and stop for coordinate c
    """
    assert stop - start >= 0, "Requested stop > start"
    assert X.ndim == 2, "2D required"
    
    coords = np.sort(X[start:stop,c].copy())

    return float(coords[(stop-start)//2])


def swap(X: np.ndarray, idx1, idx2) -> None:
    """Swaps two rows of a 2D numpy array"""
    X[[idx1, idx2]] = X[[idx2, idx1]]


def partition(X: np.ndarray, start: int, stop: int, c: int) -> int:
    """
    Partitions the array X between start and stop wrt to its median along a coordinate c
    """
    
    order = np.argsort(X[start:stop,c], kind = "mergesort")
    X[start:stop] = X[start:stop][order]
    return start + (stop - start) // 2


@dataclass
class Node:
    idx: int
    med: float = np.inf
    c: int = 0
    left: Self = (
        None  # Self denotes that left (and right) is of same type as self == Node
    )
    right: Self = None


class KDTree(NearestNeighborSearch):
    def __init__(self, X):
        """
        Contrary to LinearScan, KDTree's constructor must build the tree
        To that end, we will loop through the coordinates of X,
        hence the need for the `dim` attribute below.
        """
        super().__init__(X)
        self.dim = X.shape[1]
        self.build()

    def _build(self, start: int, stop: int, c: int) -> Node | None:
        """
        Builds a node with a correct index by partitioning X along c between start and stop,
        including left and right children nodes
        """
        assert stop >= start, "Indices issue"
        if stop == start:
            return
        if stop == start + 1:
            return Node(start, self.X[start][c], c)
        
        next_c = (c + 1) % self.dim
        # Ex5: iteratively partition, retrieve median and recurse (while creating correct Nodes)

        ind = partition(self.X, start, stop, c)
        medi = (self.X)[ind][c]

        return Node(
        idx = ind,
        med = medi,
        c = c,
        left = self._build(start, ind, next_c),
        right = self._build(ind+1, stop, next_c)
        )


    def reset(self):
        """
        Resets current estimation of distance to and index of nearest neighbor
        """
        self._current_dist = np.inf
        self._current_idx = -1

    def build(self):
        """
        Builds the kdtree
        """
        self.reset()
        self.root = self._build(0, len(self.X), 0)

    def _defeatist(self, node: Node | None, x: np.ndarray):
        """
        Defeatist search of nearest neighbor of x in node
        """
        if node is None:
            return
        
        def search_d(kdt : Node | None) :
            if kdt is None :
                return
            if self.metric(x, (self.X)[kdt.idx]) < self._current_dist :
                self._current_dist = self.metric(x, (self.X)[kdt.idx])
                self._current_idx = kdt.idx
            
            if x[kdt.c] <= kdt.med :
                search_d(kdt.left)
            else :
                search_d(kdt.right)

        search_d(node)



    def _backtracking(self, node: Node | None, x: np.ndarray):
        """
        Backtracking search of nearest neighbor of x in node
        """
        if node is None:
            return

        def search_b(kdt : Node | None) :
            if kdt is None :
                return

            if self.metric(x, (self.X)[kdt.idx]) < self._current_dist :
                self._current_dist = self.metric(x, (self.X)[kdt.idx])
                self._current_idx = kdt.idx
            
            if x[kdt.c] <= kdt.med :
                search_b(kdt.left)
                if abs(x[kdt.c] - kdt.med) <= self._current_dist :
                    search_b(kdt.right)

            else :
                search_b(kdt.right)
                if abs(x[kdt.c] - kdt.med) <= self._current_dist :
                    search_b(kdt.left)

        search_b(node)




    def query(self, x, mode: str = "backtracking"):
        """
        Queries given mode 'backtracking' or 'defeatist'
        """
        super().query(x)
        self.reset()
        if mode == "defeatist":
            self._defeatist(self.root, x)
        elif mode == "backtracking":
            self._backtracking(self.root, x)
        else:
            raise ValueError("Incorrect mode!")
        return self._current_dist, self._current_idx

    def set_xaggle_config(self):
        self.mode = None  # Choose search strategy for xaggle
        pass
