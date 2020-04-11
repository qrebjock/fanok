import numpy as np
from fanok.factor_model import (
    ledoit_wolf_shrinkage,
    single_step_factor_model,
    randomized_factor_model,
)
from sklearn.covariance import (
    empirical_covariance,
    ledoit_wolf,
    ledoit_wolf_shrinkage as sklearn_lw_shrinkage,
)


def test_shrinkage_coefficient():
    n, p = 100, 5
    mu = np.random.randn(p)
    Sigma = np.random.randn(p, p)
    Sigma = Sigma @ Sigma.T
    X = np.random.multivariate_normal(mu, Sigma, size=n)

    shrinkage, mu = ledoit_wolf_shrinkage(X)
    sklearn_shrinkage = sklearn_lw_shrinkage(X, assume_centered=True)

    assert np.isclose(shrinkage, sklearn_shrinkage)


def test_single_step_factor_model():
    n = 100
    p = 2

    X = np.random.randn(n, p)

    d, V, lam = single_step_factor_model(X, rank=2, mode="empirical")
    Sigma = empirical_covariance(X, assume_centered=True)
    Sigma2 = np.diag(d) + V @ np.diag(lam) @ V.T
    assert np.allclose(Sigma, Sigma2)

    Sigma = ledoit_wolf(X, assume_centered=True)[0]
    d, V, lam = single_step_factor_model(X, rank=2, mode="ledoit")
    Sigma2 = np.diag(d) + V @ np.diag(lam) @ V.T
    assert np.allclose(Sigma, Sigma2)


def test_randomized_factor_model():
    p = 2
    n = 1000
    Sigma = np.random.randn(p, p)
    Sigma = Sigma @ Sigma.T

    X = np.random.multivariate_normal(np.zeros(p), Sigma, size=n)

    # TODO
