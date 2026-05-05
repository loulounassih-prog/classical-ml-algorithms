import numpy as np


class ConfusionMatrix:
    """A confusion matrix

    Attributes:
        confusion_matrix: np.ndarray -- The actual 2×2 confusion matrix
    """

    def __init__(self) -> None:
        self.confusion_matrix = np.zeros((2,2))

    def add_prediction(self, true_label: int, predicted_label: int) -> None:
        """Add a labeled point to the matrix."""
        self.confusion_matrix[true_label, predicted_label] += 1


    @property
    def tp(self) -> int:
        """Return the number of true positives."""
        return int(self.confusion_matrix[1, 1])

    @property
    def tn(self) -> int:
        """Return the number of true negatives."""
        return int(self.confusion_matrix[0, 0])

    @property
    def fp(self) -> int:
        """Return the number of false positives."""
        return int(self.confusion_matrix[0, 1])

    @property
    def fn(self) -> int:
        """Return the number of false negatives."""
        return int(self.confusion_matrix[1, 0])

    def f_score(self) -> float:
        """Compute the F-score."""
        prec = self.precision()
        rec = self.detection_rate()
        return 2*prec*rec/(prec+rec)

    def precision(self) -> float:
        """Compute the precision."""
        return self.tp / (self.fp + self.tp)

    def error_rate(self) -> float:
        """Compute the error rate."""
        return (self.fp + self.fn)/(self.fp + self.fn + self.tp + self.tn)

    def detection_rate(self) -> float:
        """Compute the detection rate."""
        return self.tp / (self.fn + self.tp)

    def false_alarm_rate(self) -> float:
        """Compute the false-alarm rate."""
        return self.fp / (self.fp + self.tn)

    def print_evaluation(self) -> None:
        """Print a summary of the values of the matrix."""
        print("\t\tPredicted")
        print("\t\t0\t1")
        print(f"Actual\t0\t{self.tn}\t{self.fp}")
        print(f"\t1\t{self.fn}\t{self.tp}\n")

        print(f"Error rate\t\t{self.error_rate()}")
        print(f"False-alarm rate\t{self.false_alarm_rate()}")
        print(f"Detection rate\t\t{self.detection_rate()}")
        print(f"F-score\t\t\t{self.f_score()}")
        print(f"Precision\t\t{self.precision()}")
