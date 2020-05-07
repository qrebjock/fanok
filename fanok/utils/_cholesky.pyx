import numpy as np

cimport cython

from scipy.linalg.cython_blas cimport drotg, drot, dtrsv, dscal, daxpy, ddot
from scipy.linalg.cython_lapack cimport dgesv

from libc.math cimport sqrt
from libc.math cimport abs as cabs


FLOAT_NUMPY_TYPE_MAP = {
    4 : np.float32,
    8 : np.float64
}
NP_DOUBLE_D_TYPE = FLOAT_NUMPY_TYPE_MAP[sizeof(double)]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void __cholupdate(int p, double[:, ::1] R, double[::1] u) nogil:
    """
    Positive Cholesky rank-1 update.
    R must be upper-triangular and C-contiguous.
    At the end, u is overwritten.
    """
    cdef:
        int inc = 1
        int size = 0
        double c = 0, s = 0
        int j = 0

    for j in range(p):
        drotg(&R[j, j], &u[j], &c, &s)
        size = p - j - 1
        if size > 0:
            drot(&size, &R[j, j + 1], &inc, &u[j + 1], &inc, &c, &s)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef bint __choldowndate(int p, double[:, ::1] R, double[::1] u) nogil:
    """
    Negative Cholesky rank-1 update.
    Returns False if the update failed (matrix not psd), True otherwise.
    R must be upper-triangular and C-contiguous.
    At the end, u is overwritten.
    """
    cdef:
        int inc = 1
        int size = 1
        int k = 0
        double r = 0
        double c1 = 0, c2 = 0

    for k in range(p):
        r = (R[k, k] - u[k]) * (R[k, k] + u[k])
        if r <= 0:
            return False
        r = sqrt(r)

        size = p - k - 1

        if size > 0:
            c1 = R[k, k] / r
            c2 = -u[k] / r
            dscal(&size, &c1, &R[k, k + 1], &inc)
            daxpy(&size, &c2, &u[k + 1], &inc, &R[k, k + 1], &inc)

            c1 = r / R[k, k]
            c2 = -u[k] / R[k, k]
            dscal(&size, &c1, &u[k + 1], &inc)
            daxpy(&size, &c2, &R[k, k + 1], &inc, &u[k + 1], &inc)

        R[k, k] = r

    return True


def _check_R_u(R: np.ndarray, u: np.ndarray, upper: bool = True):
    if not upper:
        R = R.T
    R = np.ascontiguousarray(R, dtype=NP_DOUBLE_D_TYPE)
    u = np.ascontiguousarray(u, dtype=NP_DOUBLE_D_TYPE)

    p = R.shape[0]
    if R.shape[1] != p:
        raise ValueError(f'R must be a square matrix')
    if u.shape[0] != p:
        raise ValueError(f'Dimensions of R and u must agree')

    return p, R, u


def _cholupdate(R: np.ndarray, u: np.ndarray, upper: bool = True):
    p, R, u = _check_R_u(R, u, upper)
    __cholupdate(p, R, u)


def _choldowndate(R: np.ndarray, u: np.ndarray, upper: bool = True):
    p, R, u = _check_R_u(R, u, upper)
    return __choldowndate(p, R, u)
