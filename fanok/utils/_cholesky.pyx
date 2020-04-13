import numpy as np

cimport cython

from scipy.linalg.cython_blas cimport drotg, drot, dtrsv, dscal, daxpy, ddot
from scipy.linalg.cython_lapack cimport dgesv

from libc.math cimport sqrt
from libc.math cimport abs as cabs


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void cholupdate(int p, double[:, ::1] R, double[::1] u) nogil:
    """
    Positive Cholesky rank-1 update.
    R must be upper-triangular and C-contiguous.
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
cdef bint choldowndate(int p, double[:, ::1] R, double[::1] z) nogil:
    """
    Negative Cholesky rank-1 update.
    Returns False if the update failed (matrix not psd), True otherwise.
    R must be upper-triangular and C-contiguous.
    """
    cdef:
        int inc = 1
        int size = 1
        int k = 0
        double r = 0
        double c1 = 0, c2 = 0

    for k in range(p):
        r = (R[k, k] - z[k]) * (R[k, k] + z[k])
        if r <= 0:
            return False
        r = sqrt(r)

        size = p - k - 1

        if size > 0:
            c1 = R[k, k] / r
            c2 = -z[k] / r
            dscal(&size, &c1, &R[k, k + 1], &inc)
            daxpy(&size, &c2, &z[k + 1], &inc, &R[k, k + 1], &inc)

            c1 = r / R[k, k]
            c2 = -z[k] / R[k, k]
            dscal(&size, &c1, &z[k + 1], &inc)
            daxpy(&size, &c2, &R[k, k + 1], &inc, &z[k + 1], &inc)

        R[k, k] = r

    return True
