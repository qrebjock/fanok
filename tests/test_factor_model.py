import numpy as np
from fanok.factor_model import (
    ledoit_wolf_shrinkage,
    single_step_factor_model,
    randomized_factor_model,
)
from fanok.factor_model.factor_model import randomized_symmetric_decomposition
from sklearn.covariance import (
    empirical_covariance,
    ledoit_wolf,
    ledoit_wolf_shrinkage as sklearn_lw_shrinkage,
)
from sklearn.utils.extmath import randomized_svd


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


def test_randomized_symmetric_decomposition():
    p = 10
    rank = 2
    A = np.random.randn(p, p)
    A = A.T @ A

    U, eigs, _ = randomized_svd(A, rank)
    V, eigs2 = randomized_symmetric_decomposition(
        lambda v: A @ v, p, rank, 10, over_sample=5
    )

    error1 = np.linalg.norm(A - U @ np.diag(eigs) @ U.T)
    error2 = np.linalg.norm(A - V @ np.diag(eigs2) @ V.T)

    assert np.allclose(error1, error2)


def test_randomized_factor_model():
    p = 2
    n = 1000
    Sigma = np.random.randn(p, p)
    Sigma = Sigma @ Sigma.T

    X = np.random.multivariate_normal(np.zeros(p), Sigma, size=n)

    d, U, eig = randomized_factor_model(X, rank=2, shrink=False)
    Sigma = empirical_covariance(X, assume_centered=True)
    Sigma2 = np.diag(d) + U @ np.diag(eig) @ U.T
    assert np.allclose(Sigma, Sigma2)

    d, U, eig = randomized_factor_model(
        X, rank=2, num_iterations=5, shrink=True, shrinkage_mode="full"
    )
    Sigma = ledoit_wolf(X, assume_centered=True)[0]
    Sigma2 = np.diag(d) + U @ np.diag(eig) @ U.T
    assert np.allclose(Sigma, Sigma2)
