import numpy as np

from scipy.linalg import eigh, null_space, solve

from .base import KnockoffsGenerator
from fanok.sdp import solve_full_sdp


def fixed_knockoffs_parameters(
    X: np.ndarray, sdp_mode: str = "equi", assume_centered: bool = True
):
    """
    Computes the parameters allowing the create
    fixed knockoffs from the samples.
    """
    # TODO: not centered case
    n, p = X.shape[0], X.shape[1]
    if 2 * p > n:
        raise ValueError(
            f"The number of features cannot be larger than half the "
            f"number of data points. Found X with shape {(n, p)}"
        )

    Sigma = X.T @ X
    s = solve_full_sdp(Sigma, mode=sdp_mode)
    iS_t_s = solve(Sigma, np.diag(s))

    A = 2 * np.diag(s) - s[:, None] * iS_t_s

    eig, V = eigh(A)
    eig = np.clip(eig, a_min=0, a_max=None)  # Correcting computation errors
    C = np.diag(np.sqrt(eig)) @ V.T

    U_tilde = null_space(X.T)[:, : X.shape[1]]

    return np.eye(iS_t_s.shape[0]) - iS_t_s, U_tilde @ C


def sample_fixed_knockoffs_from_parameters(X: np.ndarray, A: np.ndarray, B: np.ndarray):
    """
    Sample fixed knockoffs from the matrix parameters A and B.
    """
    return X @ A + B


def natural_fixed_knockoffs(
    X: np.ndarray, A: np.ndarray = None, B: np.ndarray = None, sdp_mode: str = "equi"
):
    """
    Sample natural fixed knockoffs (case where n >= 2p).
    """
    if A is None or B is None:
        A, B = fixed_knockoffs_parameters(X, sdp_mode=sdp_mode)

    return sample_fixed_knockoffs_from_parameters(X, A, B)


def fixed_knockoffs_extend_samples(
    X: np.ndarray, y: np.ndarray = None, noise_estimate: float = None
):
    """
    Extends the samples X and/or y so that the feature dimension
    matches the sample size.
    """
    n, p = X.shape[0], X.shape[1]

    if X is not None:
        zeros = np.zeros((2 * p - n, p))
        X = np.vstack((X, zeros))

    if y is not None:
        if noise_estimate is None:
            raise ValueError(
                f"When 2p > n you need A noise estimation in order to extend the response vector y"
            )
        y = np.concatenate((y, np.random.normal(0, noise_estimate, size=2 * p - n)))

    return X, y


def extended_fixed_knockoffs(
    X: np.ndarray,
    y: np.ndarray = None,
    A: np.ndarray = None,
    B: np.ndarray = None,
    noise_estimate: float = None,
    sdp_mode: str = "equi",
):
    """
    Create fixed knockoffs in the regime where 2p > n >= p.
    """
    n, p = X.shape[0], X.shape[1]
    if n > 2 * p:
        return natural_fixed_knockoffs(X, A, B, sdp_mode=sdp_mode)
    elif p > n:
        raise ValueError(
            "Can't generated fixed knockoffs when the feature dimension is greater than the sample size"
        )

    # Augment X to get the right dimension
    X, y = fixed_knockoffs_extend_samples(X, y, noise_estimate=noise_estimate)
    X_tilde = natural_fixed_knockoffs(X, A, B, sdp_mode=sdp_mode)

    return X, X_tilde, y


def fixed_knockoffs(
    X: np.ndarray,
    y: np.ndarray = None,
    sdp_mode: str = "equi",
    noise_estimate: float = None,
    stack: bool = True,
    A: np.ndarray = None,
    B: np.ndarray = None,
):
    """
    Fixed X_tilde work better in the case where n >= 2p.
    They may be partially extended to the case where n >= p.
    In the high-dimension regime, use Gaussian X_tilde instead.
    """
    n, p = X.shape[0], X.shape[1]

    if p > n:
        raise ValueError(
            f"The number of features cannot be larger than the "
            f"number of data points. Found X with shape {(n, p)}"
        )

    if n >= 2 * p:
        X_tilde = natural_fixed_knockoffs(X, A, B, sdp_mode=sdp_mode)
    else:
        X, X_tilde, y = extended_fixed_knockoffs(
            X, y, A, B, noise_estimate=noise_estimate, sdp_mode=sdp_mode
        )
    if stack:
        X_tilde = np.hstack((X, X_tilde))
    if y is None:
        return X_tilde
    else:
        return X_tilde, y


def are_fixed_knockoffs_valid(X: np.ndarray, X_tilde: np.ndarray, tol: float = -1e-12):
    """
    Checks if the fixed knockoffs X_tilde are valid for the samples X.

    :param X: Data samples
    :param X_tilde: Knockoff samples
    :param tol: Tolerance threshold, defaults to -1e-12
    """
    Sigma = X.T @ X
    Sigma_tilde = X_tilde.T @ X_tilde

    difference = Sigma - X.T @ X_tilde
    diagonal = np.diagonal(difference)

    return (
        np.allclose(Sigma, Sigma_tilde)
        and np.allclose(difference - np.diag(diagonal), 0)
        and np.all(diagonal >= tol)
    )


class FixedKnockoffs(KnockoffsGenerator):
    """
    Abstraction of fixed knockoffs.
    In high-dimension (more features than samples),
    you should use Gaussian knockoffs instead.

    :param sdp_mode: Defines which method is employed to solve the SDP
    """

    def __init__(
        self, sdp_mode: str = "equi", noise_estimate: float = None, stack: bool = False
    ):
        super().__init__()
        self.sdp_mode = sdp_mode
        self.noise_estimate = noise_estimate
        self.stack = stack

        self.A_ = None
        self.B_ = None

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """
        Fits the knockoff generator against the samples.

        :param X: Data samples (rows are samples, columns are features)
        """
        self.A_, self.B_ = fixed_knockoffs_parameters(X, sdp_mode=self.sdp_mode)
        return self

    def transform(self, X: np.ndarray, y: np.ndarray = None):
        """
        Creates fixed knockoff features based on the samples X.
        Should be used after the generator is fitted.

        :param X: Data samples
        :return: Fixed knockoff
        """
        return fixed_knockoffs(
            X,
            sdp_mode=self.sdp_mode,
            y=y,
            noise_estimate=self.noise_estimate,
            stack=self.stack,
            A=self.A_,
            B=self.B_,
        )
