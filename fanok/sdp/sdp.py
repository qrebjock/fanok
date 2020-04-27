import numpy as np
from scipy.linalg import eigh

from fanok.sdp._full_rank import _full_rank
from fanok.sdp._low_rank import _sdp_low_rank
import cvxpy as cp


def cov_to_cor(Sigma: np.ndarray):
    """
    Converts a covariance matrix to a correlation matrix.
    """
    d = 1 / np.sqrt(np.diag(Sigma))
    return d[:, None] * Sigma * d


def sdp_equi(Sigma: np.ndarray):
    """
    Returns the minimum eigenvalue of 2 * Sigma.
    This is a cheap way to find a feasible solution to the SDP
    but knockoffs generated with it might lead to a low statistical
    power.
    """
    if Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError("Sigma is not a square matrix")

    cor = cov_to_cor(Sigma)

    min_eigenvalue = eigh(cor, eigvals_only=True, eigvals=(0, 0))[0]

    if min_eigenvalue < 0:
        raise ValueError("Sigma is not psd")

    return min(1, 2 * min_eigenvalue) * np.diag(Sigma)


def cvx_sdp_full(Sigma: np.ndarray, solver=cp.SCS, clip: bool = True, **kwargs):
    """
    Solves the SDP with CVXPY.
    """
    p = Sigma.shape[0]
    if p != Sigma.shape[1]:
        raise ValueError("Sigma is not a square matrix")

    cor = cov_to_cor(Sigma)

    s = cp.Variable(p)
    objective = cp.Maximize(cp.sum(s))
    constraints = [cp.diag(s) << 2 * cor, s <= 1, s >= 0]
    problem = cp.Problem(objective, constraints)

    problem.solve(solver=solver, **kwargs)
    if clip:
        s.value = np.clip(s.value, a_min=0, a_max=1)
    s.value *= np.diag(Sigma)

    return s.value


# TODO: Approximated SDP


def sdp_full(
    Sigma, max_iterations=None, lam=None, mu=None, tol=5e-5, return_objectives=False
):
    """
    Wrapper of the efficient Cython implementation.
    """
    cor = cov_to_cor(Sigma)

    s = _full_rank(
        cor,
        max_iterations=max_iterations,
        lam=lam,
        mu=mu,
        tol=tol,
        return_objectives=return_objectives,
    )

    return s * np.diag(Sigma)


def sdp_low_rank(
    d,
    U,
    singular_values=None,
    max_iterations=None,
    lam=None,
    mu=None,
    tol=5e-5,
    return_objectives=False,
):
    """
    Solves the low-rank SDP with coordinate ascent.
    Wrapper of the efficient Cython implementation.
    """
    # TODO: Handle singular values
    ztz = np.sum(U * U, axis=1)
    diag_Sigma = d + ztz
    inv_sqrt = 1 / np.sqrt(diag_Sigma)
    d = d / diag_Sigma
    U = inv_sqrt[:, None] * U

    s = _sdp_low_rank(
        d,
        U,
        singular_values=singular_values,
        max_iterations=max_iterations,
        lam=lam,
        mu=mu,
        tol=tol,
        return_objectives=return_objectives,
    )

    return s * diag_Sigma


def solve_full_sdp(
    Sigma: np.ndarray, mode: str = "equi", return_diag: bool = False, **kwargs
):
    """
    """
    if mode == "equi":
        res = sdp_equi(Sigma, **kwargs)
    elif mode == "sdp":
        res = sdp_full(Sigma, **kwargs)
    elif mode == "cvx":
        res = cvx_sdp_full(Sigma, **kwargs)
    elif mode == "ones":
        res = np.ones(Sigma.shape[0]) * 1e-16
    elif mode == "zero":
        res = np.zeros(Sigma.shape[0])
    else:
        raise ValueError(f"Mode can be either 'equi', 'sdp' or 'zero'. Found {mode}")

    if return_diag:
        return np.diag(res)
    else:
        return res
