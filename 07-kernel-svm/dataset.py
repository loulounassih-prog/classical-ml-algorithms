import numpy as np

class Dataset:
    def __init__(self, filename):
        self.data = np.loadtxt(filename, delimiter=',')
        self.n_samples, self.n_features = self.data.shape

    def get_dim(self):
        return self.n_features

    def get_nbr_samples(self):
        return self.n_samples

    def get_instance(self, i):
        return self.data[i]

    def get_instances(self):
        return self.data
        
    def get_column(self, index):
        """Return the full column vector at given index."""
        return self.data[:, index]


    def show(self, verbose=False):
        print(f"Dataset with {self.n_samples} samples and {self.n_features} dimensions.")
        if verbose:
            print(self.data)
