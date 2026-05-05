"""
Exercise 7 - Grid-search cross-validation for SVM hyperparameters.

Given a training dataset, an RBF kernel, and a grid of candidate values for
(C, gamma), find the hyperparameter pair that maximizes the mean k-fold
cross-validation accuracy.  Return the best pair together with the
corresponding mean accuracy.

You may NOT use scikit-learn.  You must implement:
  * stratified-like k-fold splitting from scratch,
  * the outer grid search over (C, gamma),
  * the inner CV loop that fits your SVM on the k-1 training folds and
    scores it on the held-out fold,
  * mean-of-folds aggregation and arg-max selection.

The reference solution on ramen-train.csv with k=3 folds,
Cs=[0.1, 1.0, 10.0], gammas=[0.1, 1.0, 10.0] achieves mean CV accuracy
>= 0.60 on at least one grid cell.  The grader tests that your
implementation reaches this threshold and that the selected pair is
actually the arg-max of the grid you searched.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

from TD.dataset import Dataset
from TD.kernel import Kernel, KernelType
from TD.svm import SVM


class GridSearchCV:
    """Grid-search cross-validation for (C, gamma) on an RBF-kernel SVM."""

    def __init__(
        self,
        dataset: Dataset,
        col_class: int,
        Cs: Sequence[float],
        gammas: Sequence[float],
        n_folds: int = 3,
        seed: int = 0,
    ) -> None:
        """
        Parameters
        ----------
        dataset    : the full training Dataset.
        col_class  : index of the class column in the dataset.
        Cs         : grid of C values to search over.
        gammas     : grid of gamma values to search over (RBF kernel).
        n_folds    : number of CV folds (default 3).
        seed       : RNG seed for the shuffle before folding.
        """
        self.dataset = dataset
        self.col_class = col_class
        self.Cs = list(Cs)
        self.gammas = list(gammas)
        self.n_folds = int(n_folds)
        self.seed = int(seed)

        # Filled in by .search()
        self.best_C: float | None = None
        self.best_gamma: float | None = None
        self.best_score: float | None = None
        # Full grid of mean CV accuracies, shape (len(Cs), len(gammas)).
        self.scores_: np.ndarray | None = None

    # -----------------------------------------------------------------
    # Helpers: you are free to add more if needed.
    # -----------------------------------------------------------------

    def _make_folds(self, n_samples: int) -> List[np.ndarray]:
        """
        Return a list of `n_folds` numpy arrays of row indices that
        partition range(n_samples) (approximately equally).  The partition
        is obtained by shuffling the indices with `self.seed` and then
        splitting with np.array_split.
        """
        
        indices = np.arange(n_samples)
        rng = np.random.default_rng(self.seed)
        rng.shuffle(indices)

        folds = np.array_split(indices, self.n_folds)

        return folds

    def _subset_dataset(self, rows: np.ndarray) -> Dataset:
        """
        Build a *new* Dataset restricted to the given row indices.

        Since Dataset is loaded from a file, we create a temporary
        in-memory Dataset by copying and overwriting `data`, `n_samples`,
        and `n_features`.  This avoids touching the file system.
        """
        
        subset = Dataset.__new__(Dataset)
        subset.data = np.asarray(self.dataset.data[rows], dtype=float).copy()
        subset.n_samples, subset.n_features = subset.data.shape

        return subset

    def _score_on_fold(self, train_rows: np.ndarray, test_rows: np.ndarray,
                       C: float, gamma: float) -> float:
        """
        Train an SVM with RBF kernel on `train_rows`, score it on
        `test_rows`, and return the accuracy (#correct / #total).
        """
        
        train_dataset = self._subset_dataset(train_rows)
        test_dataset = self._subset_dataset(test_rows)
        
        kernel = Kernel(KernelType.RBF, gamma=gamma)

        svm = SVM(train_dataset, self.col_class, kernel)
        svm.train(C=C)

        confusion_matrix = svm.test(test_dataset)

        tp = confusion_matrix.get_tp()
        tn = confusion_matrix.get_tn()
        fp = confusion_matrix.get_fp()
        fn = confusion_matrix.get_fn()

        total = tp + tn + fp + fn

        if total == 0:
            return 0.0
    
        return (tp + tn) / total

    # -----------------------------------------------------------------
    # Main entry point.
    # -----------------------------------------------------------------
    def search(self) -> Tuple[float, float, float]:
        """
        Run the full grid search with k-fold CV and return
        (best_C, best_gamma, best_mean_cv_accuracy).

        Also populates:
          self.best_C, self.best_gamma, self.best_score
          self.scores_  (a (len(Cs), len(gammas)) array of mean CV accuracies)
        """
        
        n_samples = self.dataset.get_nbr_samples()
        folds = self._make_folds(n_samples)

        self.scores_ = np.zeros((len(self.Cs), len(self.gammas)), dtype=float)

        for i, C in enumerate(self.Cs):
            for j, gamma in enumerate(self.gammas):
                fold_scores = []

                for k in range(self.n_folds):
                    test_rows = folds[k]
                    train_rows = np.concatenate([folds[m] for m in range(self.n_folds) if m != k])

                    score = self._score_on_fold( train_rows=train_rows, test_rows=test_rows, C=C, gamma=gamma,)

                    fold_scores.append(score)

                self.scores_[i, j] = float(np.mean(fold_scores))
        
        best_index = np.unravel_index(np.argmax(self.scores_), self.scores_.shape)
        best_C_index = best_index[0]
        best_gamma_index = best_index[1]

        self.best_C = self.Cs[best_C_index]
        self.best_gamma = self.gammas[best_gamma_index]
        self.best_score = float(self.scores_[best_C_index, best_gamma_index])

        return self.best_C, self.best_gamma, self.best_score


