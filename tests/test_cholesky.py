import numpy as np
from scipy.linalg import cholesky

from fanok.utils.cholesky import cholupdate, choldowndate


def test_cholupdate():
    p = 3
    A = np.random.randn(p, p)
    A = A.T @ A
    u = np.random.randn(p)

    R = cholesky(A, lower=True).T  # C-contiguous + upper
    cholupdate(R, u.copy())

    assert np.allclose(R, np.triu(R))
    assert np.allclose(R.T @ R, A + np.outer(u, u))


def test_choldowndate():
    p = 3
    u = np.random.randn(p)
    A = np.random.randn(p, p)
    A = A.T @ A + np.outer(u, u)

    R = cholesky(A, lower=True).T
    choldowndate(R, u.copy())

    assert np.allclose(R, np.triu(R))
    assert np.allclose(R.T @ R, A - np.outer(u, u))
