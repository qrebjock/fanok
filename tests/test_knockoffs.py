import numpy as np

from sklearn.datasets import make_regression

from fanok.generation import fixed_knockoffs, are_fixed_knockoffs_valid
from fanok.generation.gaussian import estimate_covariance


def test_estimate_covariance():
    n = 1_000_000
    p = 2
    X = np.random.randn(n, p)
    cov = estimate_covariance(X, mode="empirical", assume_centered=False)

    error = np.abs(cov - np.eye(p)).max()

    assert error < 0.1


def test_gaussian_kncokoffs():
    pass
