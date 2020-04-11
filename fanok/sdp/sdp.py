import numpy as np
from scipy.linalg import eigh

from fanok.sdp._full_rank import _full_rank
from fanok.sdp._low_rank import _sdp_low_rank
import cvxpy as cp


def sdp_equi(Sigma: np.ndarray, tol: float = 0, array: bool = True):
    """
    Returns the minimum eigenvalue of 2 * Sigma.
    This is a cheap way to find a feasible solution to the SDP
    but knockoffs generated with it might lead to a low statistical
    power.
    """
    if Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError("Sigma is not a square matrix")

    min_eigenvalue = eigh(Sigma, eigvals_only=True, eigvals=(0, 0))[0]

    if min_eigenvalue < 0:
        raise ValueError("Sigma is not psd")

    if array:
        return 2 * min_eigenvalue * np.ones(Sigma.shape[0])
    else:
        return 2 * min_eigenvalue


def cvx_sdp_full(Sigma: np.ndarray, solver=cp.SCS, clip: bool = True, **kwargs):
    """
    Solves the SDP with CVXPY.
    """
    s = cp.Variable(Sigma.shape[0])
    objective = cp.Maximize(cp.sum(s))
    constraints = [cp.diag(s) << 2 * Sigma, s >= 0]
    problem = cp.Problem(objective, constraints)

    problem.solve(solver=solver, **kwargs)

    if clip:
        return np.clip(s.value, a_min=0, a_max=None)
    else:
        return s.value


# TODO: Approximated SDP


def sdp_full(
    Sigma, max_iterations=None, lam=None, mu=None, tol=5e-5, return_objectives=False
):
    """
    Wrapper of the efficient Cython implementation.
    """
    return _full_rank(
        Sigma,
        max_iterations=max_iterations,
        lam=lam,
        mu=mu,
        tol=tol,
        return_objectives=return_objectives,
    )


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
    return _sdp_low_rank(
        d,
        U,
        singular_values=singular_values,
        max_iterations=max_iterations,
        lam=lam,
        mu=mu,
        tol=tol,
        return_objectives=return_objectives,
    )


def solve_sdp(
    Sigma: np.ndarray,
    mode: str = "equi",
    Sigma_approx: np.ndarray = None,
    return_diag: bool = False,
):
    if mode == "equi":
        res = sdp_equi(Sigma)
    elif mode == "sdp":
        res = sdp_full(Sigma)
    elif mode == "asdp":
        if Sigma_approx is None:
            raise ValueError(
                "Please provide an approximation of the matrix Sigma with the 'cap_sigma_approx' parameter"
                "when setting the mode to 'asdp'"
            )
        res = sdp_full_approx(Sigma, Sigma_approx)
    elif mode == "zero":
        res = np.zeros(Sigma.shape[0])
    else:
        raise ValueError(f"Mode can be either 'equi', 'sdp' or 'asdp'. Found {mode}")

    if return_diag:
        return np.diag(res)
    else:
        return res
