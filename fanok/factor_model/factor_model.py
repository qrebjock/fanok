import numpy as np
from scipy.linalg import svd, eigh, qr

from fanok.factor_model._shrinkage import (
    _ledoit_wolf_shrinkage,
    _shrinkage_tr_sst,
    _stochastic_lanczos_quadrature,
)


def ledoit_wolf_shrinkage(
    X: np.ndarray, s: np.ndarray = None, mode: str = "exact", m: int = 20, n_v: int = 20
):
    """
    Computes the Ledoit-Wolf optimal shrinkage coefficient from the sample X.
    It doesn't evaluate the empirical covariance Sigma.

    :param X: Data samples
    :param s:
    :param mode: Method to compute the shrinkage.
    :param m:
    :param n_v:
    """
    n, p = X.shape

    if mode == "exact":
        if s is None:
            return _ledoit_wolf_shrinkage(X)
        else:
            tr2 = np.sum(s ** 4) / n / n
    elif mode == "random":
        tr2 = _stochastic_lanczos_quadrature(X, m, n_v)
    else:
        raise ValueError(
            f"Ledoit-Wolf shrinkage mode can be either 'exact' or 'random'. Found f{mode}."
        )

    tr, sst = _shrinkage_tr_sst(X)
    d = tr2 - tr ** 2 / p
    b = sst - n * tr2

    return np.clip(b / d / n / n, a_min=0, a_max=1), tr / p


def single_step_factor_model(X: np.ndarray, rank: int, mode: str = "ledoit"):
    """
    :param X: Data samples
    :param rank: Rank approximation
    """
    # TODO: Option to do diagonal first, low-rank after.
    n = X.shape[0]

    _, lam, Vt = svd(X, full_matrices=False)

    if mode == "ledoit":
        shrinkage, mu = ledoit_wolf_shrinkage(X, s=lam)
    elif mode == "empirical":
        shrinkage, mu = 0, 0
    else:
        raise ValueError(f"mode can only be 'empirical' or 'ledoit', found {mode}")

    lam = lam[:rank]

    # The eigenvalues are shrunk in case of Ledoit-Wolf estimation
    shrunk_lam = (1 - shrinkage) * lam * lam / n + shrinkage * mu
    V = Vt.T[:, :rank]

    # Diagonal term
    d = (
        (1 - shrinkage) / n * np.sum(X * X, axis=0)
        + shrinkage * mu
        - np.sum(V * shrunk_lam * V, axis=1)
    )
    # The coordinate ascent algorithms requires positive d
    d = np.clip(d, a_min=0, a_max=None)

    return d, V, shrunk_lam


def randomized_subspace_iteration(A_dot_v, p: int, rank: int, q: int):
    """
    For a real symmetric p*p matrix A, computes a low-rank p*r matrix Q
    whose range approximates the one of A.

    :param A_dot_v: Callable taking a matrix and returning its product with A
    :param p: Size of the matrix A
    :param rank: Rank of Q
    :param q: Number of iterations to perform
    """

    # Following algorithm 4 of the paper
    # "Finding Structure With Randomness: Probabilistic Algorithms For Constructing Approximate Matrix Decompositions"
    Omega = np.random.randn(p, rank)
    Q, R = qr(A_dot_v(Omega), mode="economic")

    for _ in range(q):
        Q, R = qr(A_dot_v(Q), mode="economic")

    return Q


def randomized_symmetric_decomposition(
    A_dot_v, p: int, rank: int, q: int, over_sample: int = 10
):
    """
    For a real symmetric matrix A, computes its eigenvalue decomposition
    with randomized algorithms.

    :param A_dot_v: Callable taking a matrix and returning its product with A
    :param p: Size of the matrix A
    :param rank: Rank of Q
    :param q: Number of iterations to perform
    :param over_sample: How many more vectors than the rank to use
    for the low-rank approximation. This is essentialy usefull for
    stability purposes when the rank is low. Defaults to 10.
    """
    Q = randomized_subspace_iteration(A_dot_v, p, rank + over_sample, q)
    B = Q.T @ A_dot_v(Q)
    eig, U = eigh(B, eigvals=(B.shape[1] - rank, B.shape[1] - 1))

    return Q @ U, eig


def randomized_factor_model(
    X: np.ndarray,
    rank: int = 5,
    over_sample: int = 10,
    num_iterations: int = 2,
    shrink: bool = True,
    shrinkage_mode: str = "random",
    q: int = None,
    m: int = 20,
    n_v: int = 20,
):
    """
    :param X: Data samples
    :param rank: Rank approximation. Defaults to 5
    :param over_sample: How many more vectors than the rank to use
    for the low-rank approximation. This is essentialy usefull for
    stability purposes when the rank is low. Defaults to 10.
    :param num_iterations: How many iterations to perform in
    the alternating minimization algorithms.
    :param shrink: Whether or not to shrink the covariance matrix
    (Ledoit-Wolf estimation). This is recommended in high dimension.
    Defaults to True.
    :param shrinkage_mode: How should the optimal shrinkage coefficient
    be computed, in case of Ledoit-Wolf estimation.
    """
    n, p = X.shape

    if shrink:
        delta, mu = ledoit_wolf_shrinkage(X, mode=shrinkage_mode, m=m, n_v=n_v)
    else:
        delta, mu = 0, 0

    if q is None:
        q = 14 if rank < 0.1 * p else 8

    diag_Sigma = np.sum(X * X, axis=0) / n  # diagonal of Sigma
    A_dot_v = lambda v: X.T @ (X @ (v / n))

    d = np.zeros(p)

    for _ in range(num_iterations):
        oracle = lambda v: (1 - delta) * A_dot_v(v) + delta * mu * v - v * d[:, None]
        U, s = randomized_symmetric_decomposition(
            oracle, X.shape[1], rank, q, over_sample=over_sample
        )

        diag_U = np.sum(U * s * U, axis=1)
        d = (1 - delta) * diag_Sigma + delta * mu - diag_U
        d = np.clip(d, a_min=0, a_max=None)

    return d, U, s


class FactorModel:
    """
    Abstraction of the covariance factor model.
    If the covariance (empirical or Ledoit-Wolf) is Sigma,
    computes a diagonal plus low-rank approximation of it.

    Sigma = diag(d) + U * U^T
    """

    def fit(self, X: np.ndarray):
        raise NotImplementedError

    def transform(self):
        raise NotImplementedError


class RandomizedLowRankFactorModel(FactorModel):
    """
    Randomized factor model estimation.
    Performs an alternating minimization scheme to converge
    to local optimality.

    :param rank: Rank of the approximation
    :param over_sample: How many more vectors than the rank to use
    for the low-rank approximation. This is essentialy usefull for
    stability purposes when the rank is low. Defaults to 10.
    :param num_iterations: How many iterations to perform in
    the alternating minimization algorithms.
    :param shrink: Whether or not to shrink the covariance matrix
    (Ledoit-Wolf estimation). This is recommended in high dimension.
    Defaults to True.
    :param shrinkage_mode: How should the optimal shrinkage coefficient
    be computed, in case of Ledoit-Wolf estimation.
    """

    def __init__(
        self,
        rank: int,
        over_sample: int = 10,
        num_iterations: int = 2,
        shrink: bool = True,
        shrinkage_mode: str = "random",
    ):
        self.rank = rank
        self.shrink = shrink
        self.num_iterations = num_iterations
        self.over_sample = over_sample
        self.shrinkage_mode = shrinkage_mode

    def fit(self, X: np.ndarray):
        """
        Computes the diagonal plus low-rank approximation of the
        covariance estimated from the data samples.

        :param X: Data samples
        """
        self.d_, self.U_, self.s_ = randomized_factor_model(
            X,
            rank=self.rank,
            over_sample=self.over_sample,
            num_iterations=self.num_iterations,
            shrink=self.shrink,
            shrinkage_mode=self.shrinkage_mode,
        )

    def transform(self):
        """
        Returns the covariance approximation.
        """
        return self.d_, self.U_, self.s_
