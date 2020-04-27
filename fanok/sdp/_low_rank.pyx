import numpy as np

cimport cython
cimport numpy as np

from libc.math cimport abs

from scipy.linalg.cython_blas cimport dger, dsymv, ddot, dcopy
from scipy.linalg.cython_lapack cimport dgesv

from fanok.utils._dtypes import NP_DOUBLE_D_TYPE


cdef double _clip(double x, double amin, double amax) nogil:
    return min(max(x, amin), amax)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef sdp_rank_k(
        int p, int k,
        double[::1] d, double[:, ::1] U,
        int max_iterations, double lam, double mu, double tol
):
    cdef np.ndarray[double, ndim=1] s = np.zeros(p, dtype=NP_DOUBLE_D_TYPE)
    cdef double prev_s_sum = 0, current_s_sum = 0
    objectives = []

    cdef double[::1] dms = 1 / (np.multiply(2, d) - s)

    cdef double[::1] u_diag_term = np.sum(np.multiply(U, U), axis=1)

    cdef double[:, ::1] abc = np.multiply(U.T, dms) @ U
    cdef double[:, ::1] abc_buffer = np.zeros((k, k), dtype=NP_DOUBLE_D_TYPE)
    cdef double[::1] abc_t_u = np.zeros(k, dtype=NP_DOUBLE_D_TYPE)
    cdef double[::1] zbc = np.zeros(k, dtype=NP_DOUBLE_D_TYPE)
    cdef int[::1] piv = np.zeros(k, dtype=np.intc)  # np.intc contains the int C type at runtime

    cdef double z, t2, bmb, c
    cdef int i, j

    for i in range(max_iterations):
        prev_s_sum = current_s_sum
        current_s_sum = 0
        for j in range(p):
            update_abc(k, abc, -dms[j], j, U)
            abc_mul_v(k, abc, j, U, abc_t_u)

            t2 = 8 * compute_t2(k, abc, abc_buffer, abc_t_u, zbc, piv)
            bmb = 4 * compute_bmb(k, j, U, abc_t_u)
            c = bmb - t2

            z = 2 * (d[j] + u_diag_term[j]) - lam - c
            # if z > 1:
            #     z = 1
            z = _clip(z, 0, 1)
            current_s_sum += z
            s[j] = z

            dms[j] = 1 / (2 * d[j] - s[j])

            update_abc(k, abc, dms[j], j, U)

        objectives.append(current_s_sum)

        if current_s_sum == 0 or abs(current_s_sum - prev_s_sum) / current_s_sum < tol:
            break

        lam = mu * lam

    return s, objectives


def _sdp_low_rank(
        d, u, singular_values=None,
        max_iterations=None, lam=None, mu=None, tol=5e-5,
        return_objectives=False
):
    # Arrays must be C-contiguous, finite, and the data type C double
    d = np.ascontiguousarray(d, dtype=NP_DOUBLE_D_TYPE)
    u = np.ascontiguousarray(u, dtype=NP_DOUBLE_D_TYPE)
    if not np.all(np.isfinite(d)):
        raise ValueError(f'The diagonal D contains NaNs or Infs')
    if not np.all(np.isfinite(u)):
        raise ValueError(f'The low rank U contains NaNs or Infs')

    # Dimensions checks
    if d.ndim != 1:
        raise ValueError(f'The diagonal D must be a one-dimensional array')
    p = d.shape[0]
    if u.ndim == 1:
        k = 1
    elif u.ndim == 2:
        k = u.shape[1]
    else:
        raise ValueError(f'The low rank U must be a one or two dimensional array')
    if p != u.shape[0]:
        raise ValueError(f'Dimensions between D and U don\'t match')

    # Default lambda, mu and max_iterations
    if lam is None:
        lam = np.sqrt(k) / p / 2
    if mu is None:
        mu = 1 - 1 / (k + 1)
    if max_iterations is None:
        max_iterations = np.ceil(np.log(p * lam / tol) / np.log(1 / mu)).astype(int)

    if singular_values is not None:
        u = u * singular_values

    if u.ndim == 1:
        # not working yet
        # return sdp_rank_1(p, d, u, max_iterations, lam, mu, tol)
        s, objectives = sdp_rank_k(p, 1, d, u[:, None], max_iterations, lam, mu, tol)
    elif u.ndim == 2:
        s, objectives = sdp_rank_k(p, k, d, u, max_iterations, lam, mu, tol)

    if return_objectives:
        return s, objectives
    else:
        return s


@cython.boundscheck(False)
cdef void update_abc(int k, double[:, ::1] abc, double alpha, int j, double[:, ::1] mat_u) nogil:
    cdef:
        int inc_v = 1

        double *a0 = &abc[0, 0]
        double *u0 = &mat_u[j, 0]
    dger(&k, &k, &alpha, u0, &inc_v, u0, &inc_v, a0, &k)


@cython.boundscheck(False)
cdef void abc_mul_v(int k, double[:, ::1] abc, int j, double[:, ::1] mat, double[::1] out) nogil:
    cdef:
        char* uplo = 'L'
        double alpha = 1
        double beta = 0
        int inc = 1

        double *a0 = &abc[0, 0]
        double *v0 = &mat[j, 0]
        double *o0 = &out[0]

    dsymv(uplo, &k, &alpha, a0, &k, v0, &inc, &beta, o0, &inc)


@cython.boundscheck(False)
cdef double compute_bmb(int k, int j, double[:, ::1] mat, double[::1] v) nogil:
    cdef:
        int inc = 1

        double *u0 = &mat[j, 0]
        double *v0 = &v[0]

    return ddot(&k, u0, &inc, v0, &inc)


@cython.boundscheck(False)
cdef double compute_t2(
        int k, double[:, ::1] abc, double[:, ::1] abc_buffer, double[::1] zb, double[::1] zbc, int[::1] piv
) nogil:
    # Zb.T @ solve(np.identity(k) + 2 * abc, Zb)
    cdef:
        int inc = 1
        int info = 0

        double* a0 = &abc[0, 0]
        double* abu0 = &abc_buffer[0, 0]
        double* zb0 = &zb[0]
        double* zbc0 = &zbc[0]
        int* p0 = &piv[0]

    dcopy(&k, zb0, &inc, zbc0, &inc)
    for i in range(k):
        for j in range(k):
            abc_buffer[i, j] = 2 * abc[i, j]
        abc_buffer[i, i] += 1

    dgesv(&k, &inc, abu0, &k, p0, zbc0, &k, &info)
    return ddot(&k, zb0, &inc, zbc0, &inc)
