import numpy as np
from scipy.linalg import qr

cimport cython
cimport numpy as np

from libc.math cimport abs as cabs

from scipy.linalg.cython_blas cimport dcopy, dtrmv, dgemv, daxpy, ddot, dtrsv
# from scipy.linalg.cython_lapack cimport dgesv

from fanok.utils._dtypes import NP_DOUBLE_D_TYPE
from fanok.utils._qr cimport qr_update


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef sdp_rank_k(
    int p, int k,
    double[::1] d, double[:, ::1] U,
    double[::1] ztz, double[::1] diag_Sigma, double[::1, :] Q, double[:, ::1] R,
    int max_iterations, double lam, double mu, double tol, double lam_min
):
    objectives = [0]

    cdef:
        np.ndarray[double, ndim=1] s = np.zeros(p, dtype=NP_DOUBLE_D_TYPE)
        double prev_s_sum = 0, current_s_sum = 0

        double kappa = 0
        double[::1] qr_work = np.zeros(k, dtype=NP_DOUBLE_D_TYPE)
        double[::1] Qz = np.zeros(k, dtype=NP_DOUBLE_D_TYPE)
        double[::1] Rz = np.zeros(k, dtype=NP_DOUBLE_D_TYPE)
        double[::1] QTz = np.zeros(k, dtype=NP_DOUBLE_D_TYPE)
        double* z = NULL
        double ztAz = 0, ztiAz = 0
        double b = 0, c = 0, r = 0, f = 0, q = 0, sj = 0

        double zero = 0, one = 1, minus_one = -1
        int inc_1 = 1
        char* low = 'L'
        char* no_trans = 'N'
        char* trans = 'T'
        char* not_unit = 'N'

    for i in range(max_iterations):
        prev_s_sum = current_s_sum
        current_s_sum = 0
        for j in range(p):
            z = &U[j, 0]
            kappa = 1 / (s[j] - 2 * d[j])

            # Qz = Q @ z
            dgemv(no_trans, &k, &k, &one, &Q[0, 0], &k, z, &inc_1, &zero, &Qz[0], &inc_1)
            # Rz = R @ z
            dcopy(&k, z, &inc_1, &Rz[0], &inc_1)
            dtrmv(low, trans, not_unit, &k, &R[0, 0], &k, &Rz[0], &inc_1)

            ztAz = ddot(&k, &Qz[0], &inc_1, &Rz[0], &inc_1)

            # QTz = Q.T @ z
            dgemv(trans, &k, &k, &one, &Q[0, 0], &k, z, &inc_1, &zero, &QTz[0], &inc_1)
            # Solve triangular
            dtrsv(low, trans, not_unit, &k, &R[0, 0], &k, &QTz[0], &inc_1)
            ztiAz = ddot(&k, &QTz[0], &inc_1, z, &inc_1)

            b = (ztAz - ztz[j]) / 2 + kappa * ztz[j]**2
            c = (
                ztAz / 4
                + ztiAz * (kappa * ztz[j] * (kappa * ztz[j] - 1) + 0.25)
                + ztz[j] * (kappa * ztz[j] - 0.5)
            )
            r = ztz[j] / 2 - ztiAz / 2 + (kappa * ztz[j]) * ztiAz
            f = 2 * kappa / (1 + 2 * kappa * ztiAz)
            q = c - r * r * f
            sj = max(2 * diag_Sigma[j] - 4 * b - lam + 8 * q, 0)

            # if not(1e-9 < abs(ztAz) < 1e9) or not(1e-9 < abs(ztiAz) < 1e9):
            #     return s, objectives

            if sj != s[j]:
                kappa = 2 * (sj - s[j]) / ((2 * d[j] - s[j]) * (2 * d[j] - sj))
                s[j] = sj
                qr_update(k, Q, R, &kappa, z, z, &qr_work[0])

            current_s_sum += s[j]

        objectives.append(current_s_sum)

        lam = lam * mu
        if lam < lam_min:
            break

    return s, objectives


def _sdp_low_rank(
        d, U, singular_values=None,
        max_iterations=None, lam=None, mu=None, tol=5e-5, lam_min=1e-5,
        return_objectives=False
):
    # Arrays must be C-contiguous, finite, and the data type C double
    d = np.ascontiguousarray(d, dtype=NP_DOUBLE_D_TYPE)
    U = np.ascontiguousarray(U, dtype=NP_DOUBLE_D_TYPE)
    if not np.all(np.isfinite(d)):
        raise ValueError(f'The diagonal D contains NaNs or Infs')
    if not np.all(np.isfinite(U)):
        raise ValueError(f'The low rank U contains NaNs or Infs')

    # Dimensions checks
    if d.ndim != 1:
        raise ValueError(f'The diagonal D must be a one-dimensional array')
    p = d.shape[0]
    if U.ndim == 1:
        k = 1
    elif U.ndim == 2:
        k = U.shape[1]
    else:
        raise ValueError(f'The low rank U must be a one or two dimensional array')
    if p != U.shape[0]:
        raise ValueError(f'Dimensions between D and U don\'t match')

    # Default lambda, mu and max_iterations
    if lam is None:
        lam = np.sqrt(k) / p / 2
    if mu is None:
        mu = 1 - 1 / (k + 1)
    if max_iterations is None:
        max_iterations = np.ceil(np.log(p * lam / tol) / np.log(1 / mu)).astype(int)

    if singular_values is not None:
        U = U * singular_values

    ztz = np.sum(U * U, axis=1)
    diag_Sigma = d + ztz
    Q, R = qr(np.eye(k) + (U.T / d) @ U)

    if U.ndim == 1:
        # not working yet
        # return sdp_rank_1(p, d, U, max_iterations, lam, mu, tol)
        # s, objectives = sdp_rank_k(p, 1, d, U[:, None], max_iterations, lam, mu, tol)
        pass
    elif U.ndim == 2:
        s, objectives = sdp_rank_k(
            p, k, d, U, ztz, diag_Sigma, Q, R, max_iterations, lam, mu, tol, lam_min
        )

    if return_objectives:
        return s, objectives
    else:
        return s
