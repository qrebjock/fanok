import numpy as np
from scipy.linalg import cholesky

from sklearn.datasets import make_regression

from fanok.generation import fixed_knockoffs, are_fixed_knockoffs_valid
from fanok.generation.gaussian import (
    estimate_covariance,
    sample_gaussian_knockoffs,
    gaussian_knockoffs_sampling_parameters,
    low_rank_gaussian_knockoffs_sampling_parameters,
    sample_low_rank_gaussian_knockoffs,
)
from fanok.sdp import solve_full_sdp


def test_estimate_covariance():
    n = 1_000_000
    p = 2
    X = np.random.randn(n, p)
    cov = estimate_covariance(X, mode="empirical", assume_centered=False)

    error = np.abs(cov - np.eye(p)).max()

    assert error < 0.1


def test_gaussian_kncokoffs():
    pass


def test_low_samples_same_as_full():
    n = 10_000_000
    p = 2
    k = 1

    d = np.random.rand(p)
    U = np.random.randn(p, k)
    eig = np.random.rand(k)
    Sigma = np.diag(d) + U @ np.diag(eig) @ U.T
    mu = np.random.rand(p)

    X = np.random.multivariate_normal(mu, Sigma, size=n)

    s = solve_full_sdp(Sigma, mode="sdp", eps=1e-5)

    mu_tilde, Sigma_tilde = gaussian_knockoffs_sampling_parameters(
        X, mu, Sigma, sdp_mode="sdp", sdp_kwargs={"eps": 1e-5}
    )
    X_t1 = sample_gaussian_knockoffs(mu_tilde, cholesky(Sigma_tilde))

    c, Z, P = low_rank_gaussian_knockoffs_sampling_parameters(d, U, eig, s)
    X_t2 = sample_low_rank_gaussian_knockoffs(X, d, U, eig, s, c, Z, P)

    assert np.allclose(
        np.mean(X_t1, axis=0), np.mean(X_t2, axis=0), atol=1e-2, rtol=1e-4
    )

    assert np.allclose(
        estimate_covariance(X_t1), estimate_covariance(X_t2), atol=1e-2, rtol=1e-4
    )
