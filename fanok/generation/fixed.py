import numpy as np

from scipy.linalg import eigh, inv, null_space

from .base import KnockoffsGenerator
from fanok.sdp import solve_full_sdp


def compute_mat_a(inv_cap_sigma: np.ndarray, s: np.ndarray):
    mat_s = np.diag(s)
    return 2 * mat_s - mat_s @ inv_cap_sigma @ mat_s


def compute_mat_c(Sigma: np.ndarray):
    s, v = eigh(Sigma)
    s = np.clip(s, a_min=0, a_max=None)  # Correcting computation errors
    return np.diag(np.sqrt(s)) @ v.T


def compute_mat_u_tilde(x: np.ndarray):
    return null_space(x.T)[:, : x.shape[1]]


def fixed_x_parameters(x: np.ndarray, mode: str = "equi"):
    Sigma = x.T @ x
    inv_cap_sigma = inv(Sigma)
    s = solve_full_sdp(Sigma, mode=mode)
    mat_a = compute_mat_a(inv_cap_sigma, s)
    mat_c = compute_mat_c(mat_a)
    mat_u_tilde = compute_mat_u_tilde(x)

    return inv_cap_sigma @ np.diag(s), mat_u_tilde @ mat_c


def fixed_x_natural_knockoffs_from_parameters(
    x: np.ndarray, a: np.ndarray, b: np.ndarray
):
    return x @ (np.identity(x.shape[1]) - a) + b


def fixed_x_natural_knockoffs(
    x: np.ndarray, mode: str = "equi", a: np.ndarray = None, b: np.ndarray = None
):
    if a is None or b is None:
        a, b = fixed_x_parameters(x, mode=mode)
    return fixed_x_natural_knockoffs_from_parameters(x, a, b)


def fixed_x_extend_x(x: np.ndarray):
    n, p = x.shape[0], x.shape[1]
    zeros = np.zeros((2 * p - n, p))
    return np.vstack((x, zeros))


def fixed_x_extend_y(y: np.ndarray, noise_estimate: float, size: int):
    return np.concatenate((y, np.random.normal(0, noise_estimate, size=size)))


def fixed_x_extended_knockoffs(x: np.ndarray, mode: str = "equi"):
    return fixed_x_natural_knockoffs(fixed_x_extend_x(x), mode=mode)


def fixed_knockoffs(
    x: np.ndarray,
    mode: str = "equi",
    y: np.ndarray = None,
    noise_estimate: float = None,
    stack: bool = True,
    a: np.ndarray = None,
    b: np.ndarray = None,
):
    n, p = x.shape[0], x.shape[1]

    if p > n:
        raise ValueError(
            f"The number of features cannot be larger than the "
            f"number of data points. Found x with shape {(n, p)}"
        )

    if n >= 2 * p:
        knockoffs = fixed_x_natural_knockoffs(x, mode=mode, a=a, b=b)
    else:
        x = fixed_x_extend_x(x)
        knockoffs = fixed_x_extended_knockoffs(x, mode=mode)
        if y is not None:
            if noise_estimate is None:
                raise ValueError(
                    f"When 2p > n you need a noise estimation in order to extend the response vector y"
                )
            else:
                y = fixed_x_extend_y(y, noise_estimate, 2 * p - n)
    if stack:
        knockoffs = np.hstack((x, knockoffs))
    if y is None:
        return knockoffs
    else:
        return knockoffs, y


def are_fixed_knockoffs_valid(x: np.ndarray, x_tilde: np.ndarray):
    Sigma = x.T @ x
    cap_sigma_tilde = x_tilde.T @ x_tilde

    difference = Sigma - x.T @ x_tilde
    diagonal = np.diagonal(difference)

    return (
        np.allclose(Sigma, cap_sigma_tilde)
        and np.allclose(difference - np.diag(diagonal), 0)
        and (diagonal >= -1e-12).all()
    )


class FixedKnockoffs(KnockoffsGenerator):
    def __init__(
        self, mode: str = "equi", noise_estimate: float = None, stack: bool = False
    ):
        super().__init__()
        self.mode = mode
        self.noise_estimate = noise_estimate
        self.stack = stack

        self.a = None
        self.b = None

    def fit(self, x: np.ndarray, y: np.ndarray = None):
        self.a, self.b = fixed_x_parameters(x, mode=self.mode)
        return self

    def transform(self, x: np.ndarray, y: np.ndarray = None):
        return fixed_knockoffs(
            x,
            mode=self.mode,
            y=y,
            noise_estimate=self.noise_estimate,
            stack=self.stack,
            a=self.a,
            b=self.b,
        )
