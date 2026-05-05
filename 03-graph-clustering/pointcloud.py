# module cloud
"""Minimal implementation of data points and clouds."""

import numpy as np

class Point:
    """A point in a dataset.

    coords is initialized with a list of floats (or any other iterable
    that can be used to initialize a numpy ndarray).

    Attributes:
        name: str
        coords: np.ndarray -- coordinates of the point
    """

    def __init__(self, name: str, coords: [float]):
        self.name = name
        self.coords = np.array(coords)

    def __str__(self):
        return f'{self.coords} ({self.name})'

    def __repr__(self):
        return f'Point({repr(self.name)}, {repr(self.coords)})'

    def update_coords(self, new_coords: np.ndarray) -> None:
        """Copy the values of new_coords to coords."""
        self.coords = new_coords.copy()

    def euclidean_distance(self, other) -> float:
        """Euclidean distance from the other Point."""
        return np.sqrt(np.sum((self.coords - other.coords)**2))


class Cloud:
    """
    A simple cloud of Points.
    """
    def __init__(self):
        self._points = []

    def __str__(self) -> str:
        top = f'Cloud:\n'
        index_width = len(str(len(self._points) - 1))  # Width of max index
        rest = ''.join(f' {i:>{index_width}} {p}\n'
                       for (i, p) in enumerate(self._points))
        return top + rest

    def __len__(self) -> int:
        return len(self._points)

    def __iter__(self):
        return iter(self._points)

    def __getitem__(self, i : int):
        return self._points[i]

    def add_point(self, p : Point) -> None:
        self._points.append(p)

