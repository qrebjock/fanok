import numpy as np

from scipy.linalg import cholesky, lstsq
from sklearn.covariance import empirical_covariance, ledoit_wolf, GraphicalLassoCV

from fanok.sdp import solve_full_sdp, sdp_low_rank
from .base import KnockoffsGenerator


def estimate_covariance(X, mode: str = "wolf", assume_centered: bool = False):
    """
    Estimates the covariance matrix from the samples X.
    The estimator is either empirical, Ledoit-Wolf or graphical Lasso.

    :param X: Data samples (rows are samples, columns are features)
    :param mode: Method to estimate the covariance, defaults to "wolf"
    :param assume_centered: Whehter or not the samples are assumed to have zero mean, defaults to True
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
    sdp_kwargs: dict = None,
    covariance_mode: str = "wolf",
    assume_centered: bool = False,
    cov_tol: float = 1e-10,
):
    """
    Estimates the distribution N(mu_tilde, Sigma_tilde) of the knockoffs
    from the distribution of the original features.

    :param X: Data samples
    :param mu: True mean of the underlying model, optional
    :param Sigma: True covariance of the underlying model, optional
    :param sdp_mode: Method to solve the SDP
    :param sdp_kwargs: Additional keyword arguments given to the SDP solver
    :param covariance_mode: Method used to estimate the covariance matrix from the samples
    :param assume_centered: Whehter or not the samples are assumed to have zero mean, defaults to True
    """
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
    if sdp_kwargs is None:
        sdp_kwargs = {}

    S = solve_full_sdp(Sigma, mode=sdp_mode, return_diag=True, **sdp_kwargs)
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

    :param sdp_mode: Defines which method is employed to solve the SDP
    :param sdp_kwargs: Optional dict which is fed to the SDP solver
    :param covariance_mode: Method used to estimate the covariance matrix from the samples
    :param assume_centered: Whether or not the samples should be assumed to be centered to estimate the covariance
    """

    def __init__(
        self,
        sdp_mode: str = "sdp",
        sdp_kwargs: dict = None,
        covariance_mode: str = "wolf",
        assume_centered: bool = False,
    ):
        super().__init__()
        self.sdp_mode = sdp_mode
        self.sdp_kwargs = {} if sdp_kwargs is None else sdp_kwargs
        self.covariance_mode = covariance_mode
        self.assume_centered = assume_centered

    def fit(
        self, X: np.ndarray = None, mu: np.ndarray = None, Sigma: np.ndarray = None,
    ):
        """
        Fits the knockoff generator against the samples.
        If only X is provided, the mean and the covariance of the random variable that generated
        the data will be estimated from the samples.
        If mu and Sigma are given, they will be considered as the true parameters of the model.

        :param X: Data samples (rows are samples, columns are features)
        :param mu: True mean of the underlying model
        :param Sigma: True covariance of the underlying model
        """
        self.mu_tilde_, Sigma_tilde = gaussian_knockoffs_sampling_parameters(
            X=X,
            mu=mu,
            Sigma=Sigma,
            sdp_mode=self.sdp_mode,
            sdp_kwargs=self.sdp_kwargs,
            covariance_mode=self.covariance_mode,
            assume_centered=self.assume_centered,
        )
        self.cholesky_ = cholesky(Sigma_tilde)

        return self

    def transform(self, X: np.ndarray):
        """
        Creates knockoff features based on the samples X.
        Should be used after the generator is fitted.

        :param X: Data samples
        :return: Gaussian knockoff samples
        """
        # check_is_fitted(self, ['mu_tilde_', 'cholesky_'])
        return sample_gaussian_knockoffs(self.mu_tilde_, self.cholesky_)


class LowRankGaussianKnockoffs(KnockoffsGenerator):
    """
    Low-rank Gaussian knockoffs.
    Requires a factor-model for the covariance matrix.

    :param factor_model: Object of type FactorModel capable of
    computing the low-rank approximation of the covariance Sigma.
    :param fit_factor_model: Whether or not the factor model should
    be fitted when the generator is fitted, defaults to True
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
        """
        Creates knockoff features based on the samples X.
        Should be used after the generator is fitted.

        :param X: Data samples
        :return: Gaussian knockoff samples
        """
        return sample_low_rank_gaussian_knockoffs(
            X, d=self.d_, s=self.s_, c=self.c_, Z=self.Z_, P=self.P_
        )


def low_rank_gaussian_knockoffs_sampling_parameters(
    d: np.ndarray, U: np.ndarray, lam: np.ndarray, s: np.ndarray
):
    """
    Computes parameters from which low-rank Gaussian knockoffs
    can be samples from the low-rank approximation of Sigma
    and the solution to the SDP.

    :param d: Diagonal term in the approximation of Sigma
    :param U: Low-rank term in the approximation of Sigma
    :param lam: Eigenvalues associated to the low-rank term
    :param s: Solution to the SDP
    """
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
    """
    Samples knockoffs from the low-rank parameters (c, Z, P).
    If they are not provided, they will be computed from (d, U, lam, s).

    :param X: Data samples
    :param d: Diagonal term in the approximation of Sigma
    :param U: Low-rank term in the approximation of Sigma.
    This parameter is not required if (c, Z, P) are given.
    :param lam: Eigenvalues associated to the low-rank term
    This parameter is not required if (c, Z, P) are given.
    :param s: Solution to the SDP
    :param c: Parameter given by sample_low_rank_gaussian_knockoffs
    :param Z: Parameter given by sample_low_rank_gaussian_knockoffs
    :param P: Parameter given by sample_low_rank_gaussian_knockoffs
    """
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
    the Cholesky decomposition of Omega when p is large and k small.

    :param n: How many samples
    :param c: Diagonal term in the expression of Omega
    :param Z: Low-rank term in the expression of Omega
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
