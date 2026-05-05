import numpy as np


class Dataset:
    """A dataset of points of the same dimension.

    Attributes:
        dim: int              -- dimension of the ambient space
        nsamples: int         -- the number of points of the dataset
        instances: np.ndarray -- an array of all the points of the dataset
    """

    def __init__(self, file_path: str = "", dataset: np.ndarray = np.array([])):
        if file_path != "":
            self.instances = np.genfromtxt(file_path, delimiter=",")
        else:
            self.instances = dataset
        shape = np.shape(self.instances)
        self.nsamples, self.dim = shape
