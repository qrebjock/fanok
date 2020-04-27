import numpy as np
import cvxpy as cp

from fanok.sdp import cov_to_cor, cvx_sdp_full, sdp_equi, sdp_full, sdp_low_rank


def test_cov_to_cor():
    A = np.random.randn(3, 3)
    A = A.T @ A

    B = cov_to_cor(A)

    diagonal_roots = np.sqrt(np.diag(A))
    scales = np.outer(diagonal_roots, diagonal_roots)
    C = A / scales

    assert np.allclose(B, C)


def test_sdp_equi():
    Sigma = np.random.randn(2, 2)
    Sigma = Sigma.T @ Sigma
    Sigma = cov_to_cor(Sigma)

    min_eig = np.linalg.eigvals(2 * Sigma).min()
    assert np.allclose(sdp_equi(Sigma), min(min_eig, 1))


def test_cvx_sdp_full():
    Sigma = np.diag([2, 7])
    s = cvx_sdp_full(Sigma, solver=cp.SCS, eps=1e-10)

    assert np.allclose(s, np.array([2, 7]))


def test_sdp_full():
    Sigma = np.diag([3, 13])
    s = sdp_full(Sigma, max_iterations=1000, mu=0.99, tol=0)

    assert np.allclose(s, np.array([3, 13]), atol=1e-2, rtol=1e-4)

    Sigma = np.random.randn(2, 2)
    Sigma = Sigma.T @ Sigma
    s_cvx = cvx_sdp_full(Sigma, solver=cp.SCS, eps=1e-10)
    s_coo = sdp_full(Sigma, max_iterations=1000, mu=0.99, tol=0)
    assert np.allclose(s_cvx, s_coo, atol=1e-2, rtol=1e-4)


def test_low_rank():
    d = np.random.rand(2)
    U = np.random.randn(2, 1)
    Sigma = np.diag(d) + U @ U.T

    s_cvx = cvx_sdp_full(Sigma, solver=cp.SCS, eps=1e-10)
    s_coo = sdp_low_rank(
        d, U, singular_values=None, max_iterations=1000, mu=0.99, tol=0
    )

    assert np.allclose(s_cvx, s_coo, atol=1e-2, rtol=1e-4)

    d = np.random.rand(2)
    U = np.random.randn(2, 2)
    singular_values = np.random.rand(2)
    Sigma = np.diag(d) + U @ np.diag(singular_values ** 2) @ U.T

    s_cvx = cvx_sdp_full(Sigma, solver=cp.SCS, eps=1e-10)
    s_coo = sdp_low_rank(
        d, U, singular_values=singular_values, max_iterations=2000, mu=0.995, tol=0
    )

    assert np.allclose(s_cvx, s_coo, atol=1e-2, rtol=1e-4)
