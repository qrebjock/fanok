from typing import Union

import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator

try:
    # In Scikit-Learn 0.23 SelectorMixin is moved to sklearn.feature_selection
    from sklearn.feature_selection import SelectorMixin
except ImportError:
    from sklearn.feature_selection.base import SelectorMixin

from fanok.scores import selection_fdp, selection_power


def adaptive_significance_threshold(w: np.ndarray, q: float, offset: float = 0):
    """
    Compute the data-dependent threshold \tau from the statistics
    of the original and the knockoffs features.

    :param w: Statistics satisfying the flip-sign property.
    :param q: Desired FDR
    :param offset: Offset added to the numerator of the threshold
    expression. Knockoff+ correspond to offset=1. Defaults to 0.
    """
    w_set = np.setdiff1d(np.abs(w), 0)
    # w_set = np.union1d(np.abs(w), 0)
    w = np.broadcast_to(w, (len(w_set), len(w))).T

    numerator = np.sum(w <= -w_set, axis=0) + offset
    denominator = np.sum(w >= w_set, axis=0)
    denominator = np.clip(denominator, a_min=1, a_max=None)
    quotient = numerator / denominator

    candidates = w_set[quotient <= q]
    if candidates.size > 0:
        return np.min(candidates)
    else:
        return np.inf


def knockoffs_selection_mask(
    X: np.ndarray,
    y: np.ndarray,
    knockoffs: Union[np.ndarray, callable],
    stats: Union[np.ndarray, callable],
    q: float = 0.1,
    threshold_offset: float = 0,
):
    """
    Computes the knockoff selection mask from the data samples,
    the knockoffs, and the statistics.
    """
    if callable(knockoffs):
        X_tilde = knockoffs(X)
    else:
        X_tilde = knockoffs

    if callable(stats):
        w = stats(X, X_tilde, y)
    else:
        w = stats

    threshold = adaptive_significance_threshold(w, q, offset=threshold_offset)

    return w >= threshold


class BaseKnockoffSelector(BaseEstimator, SelectorMixin):
    """
    Base class for knockoff selectors.
    Must implement the method fit.
    """

    def __init__(self):
        pass

    def fit(self, X, y):
        """
        :param X: Data samples
        :param y: Target vector
        """
        raise NotImplementedError

    def _get_support_mask(self):
        check_is_fitted(self, "mask_")
        return self.mask_

    def score(self, X, y, ground_truth):
        """
        Computes the FDP and statistical power scores
        after fitting the selector.

        :param X: Data samples
        :param y: Target vector
        :param ground_truth: Features that are truly in the model
        """
        self.fit(X, y)
        return (
            selection_fdp(self.mask_, ground_truth),
            selection_power(self.mask_, ground_truth),
        )

    def scores(self, X, y, ground_truth, repeats=10):
        fdps, powers = [], []
        for _ in range(repeats):
            fdp, power = self.score(X, y, ground_truth)
            fdps.append(fdp)
            powers.append(power)
        return fdps, powers


class KnockoffSelector(BaseKnockoffSelector):
    """
    Main knockoff selector class, implementing the straightforward
    knockoff selector procedure.

    :param knockoffs: KnockoffGenerator object that will be fitted
    :param statistics: KnockoffStatistics object
    :param alpha: Desired FDR level
    :param offset: Offset added to the numerator of the threshold
    expression. Knockoff+ correspond to offset=1. Defaults to 0.
    :param fit_generator: Whether or not the knockoff generator
    must be fitted. If it was already fitted before, this is not
    required and it may be set to False. Defaults to True.
    """

    def __init__(
        self,
        knockoffs,
        statistics,
        alpha: float = 0.1,
        offset: float = 0,
        fit_generator: bool = True,
    ):
        self.knockoffs = knockoffs
        self.statistics = statistics
        self.alpha = alpha
        self.offset = offset
        self.fit_generator = fit_generator

        super().__init__()

    def fit(self, X, y):
        """
        :param X: Data samples
        :param y: Target vector
        """
        if self.fit_generator:
            self.knockoffs.fit(X)

        self.mask_ = knockoffs_selection_mask(
            X,
            y,
            self.knockoffs,
            self.statistics,
            q=self.alpha,
            threshold_offset=self.offset,
        )

        return self
