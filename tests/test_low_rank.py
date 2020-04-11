from fanok.generation.gaussian import sample_low_rank_gaussian
from sklearn.covariance import empirical_covariance
import numpy as np


def test_sample_low_rank_gaussian():
    n = 10_000_000
    p = 2
    rank = 1

    c = np.array([1, 2])
    Z = np.array([[3], [4]])
    Omega = np.diag(c) + Z @ Z.T

    X = sample_low_rank_gaussian(n, c, Z)
    cov = empirical_covariance(X)

    error = np.abs(cov - Omega).max()

    assert error < 0.1
