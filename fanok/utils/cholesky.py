import numpy as np

from fanok.utils._cholesky import _cholupdate, _choldowndate


def cholupdate(R: np.ndarray, u: np.ndarray, upper: bool = True):
    _cholupdate(R, u, upper)


def choldowndate(R: np.ndarray, u: np.ndarray, upper: bool = True):
    return _choldowndate(R, u, upper)
