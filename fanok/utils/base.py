import numpy as np


def sdp_solution_quality(Sigma: np.ndarray, s: np.ndarray):
    return np.linalg.eigvals(2 * Sigma - np.diag(s)).min()
