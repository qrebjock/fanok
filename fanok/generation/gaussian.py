import numpy as np

from scipy.linalg import cholesky, lstsq
from sklearn.covariance import empirical_covariance, ledoit_wolf, GraphicalLassoCV

from fanok.sdp import solve_full_sdp, sdp_low_rank
from fanok.factor_model import FactorModel
from .base import KnockoffsGenerator

from scipy.linalg import solve_triangular, sqrtm


def estimate_covariance(X, mode: str = "wolf", assume_centered: bool = False):
    """
    Estimates the covariance matrix from the samples X.
    The estimator is either empirical, Ledoit-Wolf or graphical Lasso.
    """
    if mode == "empirical":
        return empirical_covariance(X, assume_centered=assume_centered)
    elif mode == "wolf" or mode == "ledoit":
        return ledoit_wolf(X, assume_centered=assume_centered)[0]
    elif mode == "graphical":
        return GraphicalLassoCV(assume_centered=assume_centered).fit(X).covariance_
    else:
        raise ValueError(
            f"The parameter mode can only be 'empirical', 'wolf'/'ledoit' or 'graphical'; found {mode}"
        )


def gaussian_knockoffs_sampling_parameters(
    X: np.ndarray = None,
    mu: np.ndarray = None,
    Sigma: np.ndarray = None,
    sdp_mode: str = "sdp",
    covariance_mode: str = "wolf",
    assume_centered: bool = False,
    cov_tol: float = 1e-6,
):
    if (mu is None or Sigma is None) and X is None:
        raise ValueError(
            "mu/Sigma are None, and X is None. Feed X to estimate mu and Sigma"
        )
    if mu is None:
        mu = np.mean(X, axis=0)
    if Sigma is None:
        Sigma = estimate_covariance(
            X, mode=covariance_mode, assume_centered=assume_centered
        )

    S = solve_full_sdp(Sigma, mode=sdp_mode, return_diag=True)
    mul = lstsq(Sigma, S)[0]

    mu_tilde = X - (X - mu) @ mul
    Sigma_tilde = 2 * S - S @ mul

    return mu_tilde, Sigma_tilde + cov_tol * np.eye(X.shape[1])


def sample_gaussian_knockoffs(mean: np.ndarray, L: np.ndarray):
    """
    Sample Gaussian knockoffs given the mean parameter and
    the (lower) Cholesky factor of the covariance matrix.
    """
    z = np.random.normal(size=mean.shape)  # z ~ N(0, 1)
    return mean + z @ L  # ~ N(mean, L @ L.T)


def gaussian_knockoffs(
    X,
    covariance_mode: str = "wolf",
    assume_centered: bool = False,
    sdp_mode: str = "sdp",
):
    """
    Sample Gaussian knockoffs from the original samples.
    """
    mu_tilde, Sigma_tilde = gaussian_knockoffs_sampling_parameters(
        X,
        sdp_mode=sdp_mode,
        covariance_mode=covariance_mode,
        assume_centered=assume_centered,
    )
    return sample_gaussian_knockoffs(mu_tilde, cholesky(Sigma_tilde))


class GaussianKnockoffs(KnockoffsGenerator):
    """
    Gaussian knockoffs abstraction.
    """

    def __init__(
        self,
        sdp_mode: str = "sdp",
        covariance_mode: str = "wolf",
        assume_centered: bool = False,
    ):
        super().__init__()
        self.sdp_mode = sdp_mode
        self.covariance_mode = covariance_mode
        self.assume_centered = assume_centered

    def fit(
        self, X: np.ndarray = None, mu: np.ndarray = None, Sigma: np.ndarray = None,
    ):
        self.mu_tilde_, Sigma_tilde = gaussian_knockoffs_sampling_parameters(
            X=X,
            mu=mu,
            Sigma=Sigma,
            sdp_mode=self.sdp_mode,
            covariance_mode=self.covariance_mode,
            assume_centered=self.assume_centered,
        )
        self.cholesky_ = cholesky(Sigma_tilde)

        return self

    def transform(self, X: np.ndarray):
        # check_is_fitted(self, ['mu_tilde_', 'cholesky_'])
        return sample_gaussian_knockoffs(self.mu_tilde_, self.cholesky_)


class LowRankGaussianKnockoffs(KnockoffsGenerator):
    """
    Low-rank Gaussian knockoffs.
    Requires a factor-model for the covariance matrix.
    """

    def __init__(self, factor_model, fit_factor_model: bool = True):
        super().__init__()
        self.factor_model = factor_model
        self.fit_factor_model = fit_factor_model

    def fit(
        self,
        X: np.ndarray = None,
        mu: np.ndarray = None,
        d: np.ndarray = None,
        U: np.ndarray = None,
        singular_values: np.ndarray = None,
        sdp_kwargs: dict = None,
    ):
        if self.fit_factor_model:
            self.factor_model.fit(X)

        d, U, eig = self.factor_model.transform()
        singular_values = np.sqrt(eig)

        if sdp_kwargs is None:
            sdp_kwargs = {}
        s = sdp_low_rank(d, U, singular_values=singular_values, **sdp_kwargs)

        self.s_ = s
        self.d_ = d
        (self.c_, self.Z_, self.P_,) = low_rank_gaussian_knockoffs_sampling_parameters(
            d, U, singular_values * singular_values, s
        )

    def transform(self, X: np.ndarray):
        return sample_low_rank_gaussian_knockoffs(
            X, d=self.d_, s=self.s_, c=self.c_, Z=self.Z_, P=self.P_
        )


def low_rank_gaussian_knockoffs_sampling_parameters(
    d: np.ndarray, U: np.ndarray, lam: np.ndarray, s: np.ndarray
):
    N = cholesky(np.linalg.inv(np.diag(1 / lam) + (U.T / d) @ U), lower=True)
    P = (U.T / d).T @ N
    Z = (P.T * s).T

    c = 2 * s - s / d * s

    return c, Z, P


def sample_low_rank_gaussian_knockoffs(
    X: np.ndarray,
    d: np.ndarray = None,
    U: np.ndarray = None,
    lam: np.ndarray = None,
    s: np.ndarray = None,
    c: np.ndarray = None,
    Z: np.ndarray = None,
    P: np.ndarray = None,
):
    if c is None or Z is None or P is None:
        c, Z, P = low_rank_gaussian_knockoffs_sampling_parameters(d, U, lam, s)

    X_tilde = sample_low_rank_gaussian(X.shape[0], c, Z)
    mu_tilde = X - X / d * s + ((s * X) @ P) @ P.T

    return mu_tilde + X_tilde


def sample_low_rank_gaussian(n: int, c: np.ndarray, Z: np.ndarray):
    """
    Sample n vectors from the normal distribution N(0, Omega)
    where Omega = diag(c) + Z * Z^T (c is a p-dimensional vector
    and Z a p * k matrix).
    It doesn't build the matrix Omega and is much faster than computing
    the Cholesky decomposition of Omega when p is large (and k small).
    """
    p, rank = Z.shape
    X = np.random.randn(n, p)
    M = np.identity(rank)

    Q = np.zeros((X.shape[0], rank))
    Y = np.zeros((X.shape[0], p))

    for j in range(p):
        t = M @ Z[j, :]
        delta = c[j] + Z[j, :] @ t
        if delta > 0:
            b = t / np.sqrt(delta)
            M -= np.outer(b, b)

            Y[:, j] = np.sqrt(delta) * X[:, j] + Q @ Z[j, :]
            Q += np.outer(X[:, j], b)
        else:
            Y[:, j] = Q @ Z[j, :]

    return Y
