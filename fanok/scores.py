import numpy as np


def selection_bool_fdp(mask: np.ndarray, true_mask: np.ndarray, q: float = 0) -> float:
    return (mask & ~true_mask).sum() / max(mask.sum() + (0 if q == 0 else 1 / q), 1)


def selection_fdp(a: np.ndarray, b: np.ndarray, q: float = 0) -> float:
    """
    Computes the false discovery proportion (FDP) of the subset
    of selected features.

    :param a: Mask array describing the selected features
    :param b: Mask array of the ground truth
    :param q: Denominator term of the modified FDP expression, defaults to 0
    """
    return selection_bool_fdp(a.astype(bool), b.astype(bool), q=q)


def selection_bool_power(mask: np.ndarray, true_mask: np.ndarray) -> float:
    return (mask & true_mask).sum() / max(true_mask.sum(), 1)


def selection_power(a: np.ndarray, b: np.ndarray) -> float:
    """
    Computes the statistical power of the subset of selected features.

    :param a: Mask array describing the selected features
    :param b: Mask array of the ground truth
    """
    return selection_bool_power(a.astype(bool), b.astype(bool))
