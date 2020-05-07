import numpy as np

from scipy.linalg.cython_blas cimport drotg, drot, dgemv, daxpy, dscal

cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void reduced_hessenberg_qr(int p, double[::1, :] Q, double[:, ::1] R) nogil:
    cdef:
        int inc = 1
        int size = 0
        int j = 0
        double c = 0, s = 0

    for j in range(p - 1):
        drotg(&R[j, j], &R[j + 1, j], &c, &s)
        R[j + 1, j] = 0

        size = p - j - 1
        drot(&size, &R[j, j + 1], &inc, &R[j + 1, j + 1], &inc, &c, &s)

        size = j + 2
        drot(&size, &Q[0, j], &inc, &Q[0, j + 1], &inc, &c, &s)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void hessenberg_qr(int p, double[::1, :] Q, double[:, ::1] R) nogil:
    cdef:
        int inc = 1
        int size = 0
        int j = 0
        double c = 0, s = 0

    for j in range(p - 1):
        drotg(&R[j, j], &R[j + 1, j], &c, &s)
        R[j + 1, j] = 0

        size = p - j - 1
        drot(&size, &R[j, j + 1], &inc, &R[j + 1, j + 1], &inc, &c, &s)
        drot(&p, &Q[0, j], &inc, &Q[0, j + 1], &inc, &c, &s)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void hessenberg_qr_update(
    int p, double[::1, :] Q, double[:, ::1] R,
    double* kappa, double* u0, double* v0, double* w0
) nogil:
    """
    Updates the QR factorization of an upper Hessenberg matrix
    (upper triangular with potential non-null entries on the lower
    diagonal).

    :param p: Size of the matrix
    :param Q: Q factor
    :param R: R factor
    """
    cdef:
        int j = 0

        int inc = 1
        char* trans = 'T'
        double alpha = 1, beta = 0

        int size = 0
        double c = 0, s = 0

    dgemv(trans, &p, &p, &alpha, &Q[0, 0], &p, &u0[0], &inc, &beta, &w0[0], &inc)
    dscal(&p, kappa, &w0[0], &inc)

    for j in range(p - 2, -1, -1):
        drotg(&w0[j], &w0[j + 1], &c, &s)
        w0[j + 1] = 0

        size = p - j
        drot(&size, &R[j, j], &inc, &R[j + 1, j], &inc, &c, &s)
        drot(&p, &Q[0, j], &inc, &Q[0, j + 1], &inc, &c, &s)

    daxpy(&p, &w0[0], &v0[0], &inc, &R[0, 0], &inc)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void qr_update(
    int p, double[::1, :] Q, double[:, ::1] R,
    double* kappa, double* u0, double* v0, double* w0
) nogil:
    """
    Updates the QR decomposition of a matrix.
    M = QR
    M' = M + kappa * u * v^T
    Computes the factors M' = Q'*R'

    :param p: Size of the matrix
    :param Q: Q factor
    :param R: R factor
    :param kappa: Rank-1 scalar factor
    :param u: Rank-1 vector factor
    :param v: Rank-1 vector factor
    :param w0: Working array. Overwritten at the end of the function
    """
    hessenberg_qr_update(p, Q, R, kappa, u0, v0, w0)
    hessenberg_qr(p, Q, R)


cpdef void qqr_update(
    Q: np.ndarray, R: np.ndarray,
    double kappa, double[::1] u, double[::1] v, double[::1] w = None
):
    p = Q.shape[0]
    qr_update(p, Q, R, &kappa, &u[0], &v[0], &w[0])
