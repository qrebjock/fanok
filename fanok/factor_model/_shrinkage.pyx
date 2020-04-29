import numpy as np
from scipy.linalg import svd

cimport cython
cimport numpy as np

from cython.parallel import prange
from libc.stdlib cimport rand

from scipy.linalg.cython_blas cimport ddot, dgemv, dscal, dnrm2
from scipy.linalg.cython_lapack cimport dbdsdc

from fanok.utils._dtypes import NP_DOUBLE_D_TYPE


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef void golub_kahan(
        int n, int p, double[:, ::1] X, double[::1] u, double[::1] v,
        int max_iterations, double[::1] d_out, double[::1] b_out
) nogil:
    cdef:
        double alpha = 0
        double beta = 0

        char* trans = 'T'
        char* no_trans = 'N'
        double c = 1
        int inc = 1
        double kappa

    for k in range(max_iterations):
        kappa = -beta
        dgemv(trans, &p, &n, &c, &X[0, 0], &p, &v[0], &inc, &kappa, &u[0], &inc)  # u = X @ v - beta * u
        alpha = dnrm2(&n, &u[0], &inc)
        kappa = 1 / alpha
        dscal(&n, &kappa, &u[0], &inc)  # u /= alpha

        kappa = -alpha
        dgemv(no_trans, &p, &n, &c, &X[0, 0], &p, &u[0], &inc, &kappa, &v[0], &inc)  # v = X.T @ u - alpha * v
        beta = dnrm2(&p, &v[0], &inc)  # np.linalg.norm(v)
        kappa = 1 / beta
        dscal(&p, &kappa, &v[0], &inc)  # v /= beta

        d_out[k] = alpha
        if k < max_iterations - 1:
            b_out[k] = beta


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef __stochastic_lanczos_quadrature(
        int n, int p, double[:, ::1] X, int m, int n_v
):
    cdef:
        double trace = 0
        double sqrt_p = np.sqrt(p)
        double[::1] u = np.zeros(n, dtype=NP_DOUBLE_D_TYPE)
        double[::1] v = np.zeros(p, dtype=NP_DOUBLE_D_TYPE)
        double[::1] d = np.zeros(m, dtype=NP_DOUBLE_D_TYPE)
        double[::1] b = np.zeros(m - 1, dtype=NP_DOUBLE_D_TYPE)

        # SVD
        char* up = 'up'
        char* comp = 'I'
        double[::1] work = np.zeros(3 * m**2 + 4 * m, dtype=NP_DOUBLE_D_TYPE)
        int[::1] i_work = np.zeros(8 * m, dtype=np.intc)
        int info
        double[:, ::1] U = np.zeros((m, m), dtype=NP_DOUBLE_D_TYPE)
        double[:, ::1] Vt = np.zeros((m, m), dtype=NP_DOUBLE_D_TYPE)

    with nogil:
        for i in range(n_v):
            for j in range(p):
                if rand() % 2:
                    v[j] = 1. / sqrt_p
                else:
                    v[j] = -1. / sqrt_p

            golub_kahan(n, p, X, u, v, m, d, b)

            dbdsdc(up, comp, &m, &d[0], &b[0], &U[0, 0], &m, &Vt[0, 0], &m, NULL, NULL, &work[0], &i_work[0], &info)

            for k in range(m):
                trace += Vt[0, k]**2 * d[k]**4

    return p / n_v * trace / n / n


cpdef _stochastic_lanczos_quadrature(X, m, n_v):
    # Array must be C-contiguous, finite, and the data type C double
    X = np.ascontiguousarray(X, dtype=NP_DOUBLE_D_TYPE)
    if not np.all(np.isfinite(X)):
        raise ValueError(f'X contains NaNs or Infs')

    # Dimensions checks
    if X.ndim != 2:
        raise ValueError(f'X must be a two dimensional array')
    n, p = X.shape

    if m < 2:
        raise ValueError(f'm must be >= 2, found {m}')

    return __stochastic_lanczos_quadrature(n, p, X, m, n_v)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef __shrinkage_tr_sst(int n, int p, double[:, ::1] X):
    cdef double tr = 0
    cdef double sst = 0
    cdef double temp = 0
    cdef int inc = 1
    cdef int i, j

    with nogil:
        for i in prange(n):
            temp = ddot(&p, &X[i, 0], &inc, &X[i, 0], &inc)
            tr += temp
            sst += temp**2

    return tr / n, sst


cpdef _shrinkage_tr_sst(X):
    # Array must be C-contiguous, finite, and the data type C double
    X = np.ascontiguousarray(X, dtype=NP_DOUBLE_D_TYPE)
    if not np.all(np.isfinite(X)):
        raise ValueError(f'X contains NaNs or Infs')

    # Dimensions checks
    if X.ndim != 2:
        raise ValueError(f'X must be a two dimensional array')
    n, p = X.shape

    return __shrinkage_tr_sst(n, p, X)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef __ledoit_wolf_shrinkage(
    int n, int p, double[:, ::1] X
):
    cdef double tr = 0  # trace(Sigma)
    cdef double tr_2 = 0  # trace(Sigma^2)
    cdef double self_dot = 0
    cdef double temp
    cdef int inc = 1
    cdef double b, d
    cdef int i, j

    with nogil:
        for i in prange(n):
            for j in range(i):
                tr_2 += 2 * (ddot(&p, &X[i, 0], &inc, &X[j, 0], &inc))**2
            temp = ddot(&p, &X[i, 0], &inc, &X[i, 0], &inc)
            tr += temp
            self_dot += temp**2

    tr_2 += self_dot
    tr = tr / n
    tr_2 = tr_2 / n / n

    d = tr_2 - (tr * tr) / p
    b = (self_dot - n * tr_2) / n / n

    b = min(b, d)

    return b / d, tr / p


def _ledoit_wolf_shrinkage(
        X
):
    # Array must be C-contiguous, finite, and the data type C double
    X = np.ascontiguousarray(X, dtype=NP_DOUBLE_D_TYPE)
    if not np.all(np.isfinite(X)):
        raise ValueError(f'X contains NaNs or Infs')

    # Dimensions checks
    if X.ndim != 2:
        raise ValueError(f'X must be a two dimensional array')
    n, p = X.shape

    return __ledoit_wolf_shrinkage(n, p, X)
