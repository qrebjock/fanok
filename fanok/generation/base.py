import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


class KnockoffsGenerator(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, x: np.ndarray, y: np.ndarray = None):
        raise NotImplementedError

    def transform(self, x: np.ndarray, y: np.ndarray = None):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)
