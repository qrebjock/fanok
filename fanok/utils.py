import numpy as np

from scipy.linalg import eigh


def is_positive_definite(x: np.ndarray):
    return np.all(eigh(x, eigvals_only=True) > 0)


def idx_to_mask(idx: np.ndarray, size: int, dtype=bool):
    mask = np.zeros(size, dtype=dtype)
    mask[idx] = 1
    return mask


def cov_to_cor(sigma: np.ndarray):
    diagonal_root = np.sqrt(np.diag(sigma))
    scales = np.outer(diagonal_root, diagonal_root)
    return sigma / scales


def is_pos_def(x):
    return np.all(np.linalg.eigvalsh(x) > 0)
