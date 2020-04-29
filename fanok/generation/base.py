import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


class KnockoffsGenerator(BaseEstimator, TransformerMixin):
    """
    Base class for all knockoff generators.
    They must implement the functions fit and transform.
    """

    def __init__(self):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        raise NotImplementedError

    def transform(self, X: np.ndarray, y: np.ndarray = None):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)
