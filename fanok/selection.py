from typing import Union

import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin

from fanok.scores import selection_fdr, selection_power


def adaptive_significance_threshold(w: np.ndarray, q: float, offset: float = 0):
    """
    Compute the data-dependent threshold \tau from the statistics
    of the original and the knockoffs features.
    """
    w_set = np.setdiff1d(np.abs(w), 0)
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


class BaseKnockoffsSelector(BaseEstimator, SelectorMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        raise NotImplementedError

    def _get_support_mask(self):
        check_is_fitted(self, "mask_")
        return self.mask_

    def score(self, X, y, ground_truth):
        # check_is_fitted(self, 'mask_')
        self.fit(X, y)
        return (
            selection_fdr(self.mask_, ground_truth),
            selection_power(self.mask_, ground_truth),
        )

    def scores(self, X, y, ground_truth, repeats=10):
        fdrs, powers = [], []
        for _ in range(repeats):
            fdr, power = self.score(X, y, ground_truth)
            fdrs.append(fdr)
            powers.append(power)
        return fdrs, powers


class KnockoffsSelector(BaseKnockoffsSelector):
    """

    """

    def __init__(
        self,
        knockoffs_generator,
        stats_fun,
        alpha: float = 0.1,
        offset: float = 0,
        fit_generator: bool = True,
    ):
        super().__init__()
        self.knockoffs_generator = knockoffs_generator
        self.stats_fun = stats_fun
        self.alpha = alpha
        self.offset = offset
        self.fit_generator = fit_generator

    def fit(self, X, y):
        raise NotImplementedError


class SimpleKnockoffsSelector(KnockoffsSelector):
    def __init__(
        self,
        knockoffs_generator,
        stats_fun,
        alpha: float = 0.1,
        offset: float = 0,
        fit_generator: bool = True,
    ):
        super().__init__(
            knockoffs_generator,
            stats_fun,
            alpha=alpha,
            offset=offset,
            fit_generator=fit_generator,
        )

    def fit(self, X, y):
        if self.fit_generator:
            self.knockoffs_generator.fit(X)

        self.mask_ = knockoffs_selection_mask(
            X,
            y,
            self.knockoffs_generator,
            self.stats_fun,
            q=self.alpha,
            threshold_offset=self.offset,
        )

        return self
