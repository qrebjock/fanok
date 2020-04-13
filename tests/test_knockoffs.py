import numpy as np

from sklearn.datasets import make_regression

from fanok.generation import fixed_knockoffs, are_fixed_knockoffs_valid
from fanok.generation.gaussian import estimate_covariance


def test_fixed_knockoffs():
    X, _ = make_regression(n_samples=250, n_features=125, n_informative=75)
    X_tilde = fixed_knockoffs(X, mode="equi", stack=False)

    assert are_fixed_knockoffs_valid(X, X_tilde)

    X_tilde = fixed_knockoffs(X, mode="sdp", stack=False)

    assert are_fixed_knockoffs_valid(X, X_tilde)


def test_estimate_covariance():
    n = 1_000_000
    p = 2
    X = np.random.randn(n, p)
    cov = estimate_covariance(X, mode="empirical", assume_centered=False)

    error = np.abs(cov - np.eye(p)).max()

    assert error < 0.1


def test_gaussian_kncokoffs():
    pass
