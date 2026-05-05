import numpy as np


class Dataset:
    """
    Dataset class for loading and accessing CSV data.
    This version removes any trailing commas (empty fields) from each line.
    """

    def __init__(self, filename: str) -> None:
        """
        Initialize the Dataset by loading data from a CSV file.
        This method strips any trailing commas from each line before parsing.

        Parameters:
            filename (str): Path to the CSV file.
        """
        # Read the file line by line and remove any trailing commas and newline characters.
        with open(filename, 'r') as f:
            lines = [line.rstrip('\n') for line in f]

        # Optionally, check if the first line is a header.
        # If the first field cannot be converted to float, assume it's a header and remove it.
        try:
            float(lines[0].split(',')[0])
            header = False
        except ValueError:
            header = True

        # Remove trailing commas from all lines.
        if header:
            lines = [line.rstrip(',') for line in lines[1:]]
        else:
            lines = [line.rstrip(',') for line in lines]

        # Use np.genfromtxt on the cleaned lines.
        self.data = np.genfromtxt(lines, delimiter=',', filling_values=0)
        self.n_samples, self.n_features = self.data.shape

    def get_dim(self) -> int:
        """
        Get the number of dimensions (columns) in the dataset.

        Returns:
            int: Number of features (columns).
        """
        return self.n_features

    def get_nbr_samples(self) -> int:
        """
        Get the number of samples (rows) in the dataset.

        Returns:
            int: Number of samples.
        """
        return self.n_samples

    def get_instance(self, i: int) -> np.ndarray:
        """
        Get a specific instance (row) from the dataset.

        Parameters:
            i (int): Index of the instance.

        Returns:
            np.ndarray: The data row.
        """
        return self.data[i]

    def show(self, verbose: bool = False) -> None:
        """
        Display information about the dataset. If verbose, also print the data.

        Parameters:
            verbose (bool): Whether to print the full dataset.
        """
        print(f"Dataset with {self.n_samples} samples and {self.n_features} dimensions.")
        if verbose:
            print(self.data)

