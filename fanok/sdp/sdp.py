import warnings

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, cut_tree

from fanok.sdp._full_rank import _full_rank
from fanok.sdp._low_rank import _sdp_low_rank

try:
    import cvxpy as cp
except ImportError:
    # CVXPY isn't installed
    cp = None


def cov_to_cor(Sigma: np.ndarray):
    """
    Converts a covariance matrix to a correlation matrix.

    :param Sigma: Covariance matrix
    """
    d = 1 / np.sqrt(np.diag(Sigma))
    return d[:, None] * Sigma * d


def sdp_equi(Sigma: np.ndarray):
    """
    Returns the minimum eigenvalue of 2 * Sigma.
    This is a cheap way to find a feasible solution to the SDP
    but knockoffs generated with it might lead to a low statistical
    power.

    :param Sigma: Covariance matrix. It is scaled to a correlation matrix
    and the result is scaled back at the end.
    """
    if Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError("Sigma is not a square matrix")

    cor = cov_to_cor(Sigma)

    min_eigenvalue = eigh(cor, eigvals_only=True, eigvals=(0, 0))[0]

    if min_eigenvalue < 0:
        raise ValueError("Sigma is not psd")

    return min(1, 2 * min_eigenvalue) * np.diag(Sigma)


def cvx_sdp_full(Sigma: np.ndarray, solver: str = None, clip: bool = True, **kwargs):
    """
    Solves the SDP with CVXPY.

    :param Sigma: Covariance matrix
    :param solver: Which solver to use. Defaults to SCS
    :param clip: Whether or not to clip the solution (on the correlation matrix)
    into [0, 1]. It is only supposed to fix eventual numerical approximations
    of the solver. Defaults to True
    :param kwargs: Extra keyword arguments passed to the solver
    """
    if cp is None:
        raise ImportError(
            f"CVXPY is not installed; you cannot solve the SDP with it."
            f"Instead, either solve the SDP with coordinate ascent or install CVXPY."
        )
    # Default solver; SCS is installed by default with CVXPY
    # and scales pretty well. CVXOPT is another strong option
    # but must be installed separately.
    if solver is None:
        solver = cp.SCS

    p = Sigma.shape[0]
    if p != Sigma.shape[1]:
        raise ValueError("Sigma is not a square matrix")

    cor = cov_to_cor(Sigma)

    s = cp.Variable(p)
    objective = cp.Maximize(cp.sum(s))
    constraints = [cp.diag(s) << 2 * cor, s <= 1, s >= 0]
    problem = cp.Problem(objective, constraints)

    problem.solve(solver=solver, **kwargs)
    if s.value is None:
        raise RuntimeError("CVX didn't converge")
    if clip:
        s.value = np.clip(s.value, a_min=0, a_max=1)
    s.value *= np.diag(Sigma)

    return s.value


def make_asdp_clusters(Sigma: np.ndarray, blocks: int = 2):
    """
    Approximates the covariance matrix Sigma with a block
    diagonal matrix.

    :param Sigma: Covariance matrix
    :param blocks: Number of clusters
    """
    cor = cov_to_cor(Sigma)
    dissimilarity = 1 - cor

    distances = pdist(dissimilarity)
    lkg = linkage(distances, method="ward")
    labels = np.squeeze(cut_tree(lkg, blocks))

    sub_Sigmas = []
    indices = []
    for label in range(blocks):
        ind = np.where(labels == label)[0]
        indices.append(ind)
        sub_Sigmas.append(Sigma[np.ix_(ind, ind)])

    return indices, sub_Sigmas


def asdp(Sigma: np.ndarray, blocks: int = 2, gamma_tol: float = 1e-5, **kwargs):
    """
    Solves the SDP in two steps. First the covariance is approximated with
    a block-diagonal matrix. Sub-SDPs are solved in these blocks.
    Then, a one-dimensional SDP is efficiently solved with bisection in
    order to make the solution feasible.

    :param Sigma: Covariance matrix
    :param blocks: Number of clusters
    :param gamma_tol: Tolerance threshold when solving the one-dimensional SDP
    :param kwargs: Extra keyword arguments given to the SDP solver
    """
    p = Sigma.shape[0]
    if p != Sigma.shape[1]:
        raise ValueError("Sigma is not a square matrix")

    # Clustering step and solving sub SDPs
    indices, sub_Sigmas = make_asdp_clusters(Sigma, blocks)
    s = np.zeros(p)
    for i, sub_Sigma in enumerate(sub_Sigmas):
        s[indices[i]] = solve_full_sdp(sub_Sigma, mode="sdp", **kwargs)

    gamma_min, gamma_max = 0, 1
    while gamma_max - gamma_min > gamma_tol:
        gamma = (gamma_max + gamma_min) / 2
        G = 2 * Sigma - gamma * np.diag(s)
        min_eigenvalue = eigh(G, eigvals_only=True, eigvals=(0, 0))[0]
        if min_eigenvalue >= 0:
            gamma_min = gamma
        else:
            gamma_max = gamma

    if gamma_min == 0:
        warnings.warn(
            "When solving the ASDP, found gamma = 0. "
            "The knockoffs won't have any power. "
            "Consider lowering the parameter gamma_tol",
        )
    s = s * gamma_min

    return s


def sdp_full(
    Sigma: np.ndarray,
    max_iterations: int = None,
    lam: float = None,
    mu: float = None,
    tol: float = -1,
    eps=1e-5,
    return_objectives: bool = False,
):
    """
    Solves the SDP with a fast coordinate ascent algorithm.
    Wrapper of the efficient Cython implementation.

    :param Sigma: Covariance matrix
    :param max_iterations: Maximum number of coordinate cycles
    :param lam: Initial barrier coefficient parameter. Most
    you don't need to change this parameter because the default
    value automatically adapts to the problem.
    :param mu: Barrier coefficient decay parameter. Should be comprised
    between 0 and 1. Lower values are ore aggressive but might not
    converge to the optimal value (machine precision is reached faster).
    Defaults to 0.8.
    """
    cor = cov_to_cor(Sigma)

    s, objectives = _full_rank(
        cor,
        max_iterations=max_iterations,
        lam=lam,
        mu=mu,
        tol=tol,
        eps=eps,
        return_objectives=True,
    )

    s = s * np.diag(Sigma)

    if return_objectives:
        return s, objectives
    return s


def sdp_low_rank(
    d: np.ndarray,
    U: np.ndarray,
    singular_values: np.ndarray = None,
    max_iterations: int = None,
    lam: float = None,
    mu: float = None,
    tol: float = -1,
    eps: float = 1e-5,
    return_objectives: bool = False,
):
    """
    Solves the low-rank SDP with coordinate ascent.
    The covariance Sigma is supposed to have the special structure
    Sigma = diag(d) + U * eigs * U^T
    Wrapper of the efficient Cython implementation.

    :param d: Positive diagonal term of the factor model
    :param U: Low-rank term of the factor model
    :param singular_values: Optional singular_values of the low-rank term
    :param max_iterations: Maximum number of coordinate cycles
    :param lam: Initial barrier coefficient parameter. Most
    you don't need to change this parameter because the default
    value automatically adapts to the problem.
    :param mu: Barrier coefficient decay parameter. Should be comprised
    between 0 and 1. Lower values are ore aggressive but might not
    converge to the optimal value (machine precision is reached faster).
    Defaults to 0.8.
    :param return_objectives: Whether or not the sequences of objectives
    should be returned. Defaults to False
    """
    if singular_values is not None:
        U = U * singular_values

    # Scale Sigma to a correlation matrix
    ztz = np.sum(U * U, axis=1)
    diag_Sigma = d + ztz
    inv_sqrt = 1 / np.sqrt(diag_Sigma)
    d = d / diag_Sigma
    U = inv_sqrt[:, None] * U

    s, objectives = _sdp_low_rank(
        d,
        U,
        max_iterations=max_iterations,
        lam=lam,
        mu=mu,
        tol=tol,
        eps=eps,
        return_objectives=True,
    )

    s = s * diag_Sigma

    if return_objectives:
        return s, objectives
    return s


def solve_full_sdp(
    Sigma: np.ndarray, mode: str = "equi", return_diag: bool = False, **kwargs
):
    """
    Solves the SDP with one of the available methods ("equi", "sdp",
    "cvx", "asdp").

    :param Sigma: Covariance matrix
    :param mode: Method to solve the SDP
    :param return_diag: Whether to return a diagonal matrix or not
    (just the solution vector). Defaults to False.
    """
    if mode == "equi":
        res = sdp_equi(Sigma, **kwargs)
    elif mode == "sdp":
        res = sdp_full(Sigma, **kwargs)
    elif mode == "cvx":
        res = cvx_sdp_full(Sigma, **kwargs)
    elif mode == "asdp":
        res = asdp(Sigma, **kwargs)
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
