import numpy as np
from scipy.linalg import qr, solve_triangular

cimport cython
cimport numpy as np

from libc.math cimport abs as cabs

from scipy.linalg.cython_blas cimport dcopy, dtrmv, dgemv, daxpy, ddot, dtrsv
# from scipy.linalg.cython_lapack cimport dgesv

from fanok.utils._dtypes import NP_DOUBLE_D_TYPE
from fanok.utils._qr cimport qr_update


cdef double _clip(double x, double amin, double amax) nogil:
    return min(max(x, amin), amax)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef sdp_rank_k(
    int p,
    int k,
    double[::1] d,
    double[:, ::1] U,
    double[::1] diag_Sigma,
    double[::1, :] Q,
    double[:, ::1] R,
    int max_iterations,
    double lam,
    double mu,
    double tol,
    double lam_min
):
    objectives = [0]

    cdef:
        np.ndarray[double, ndim=1] s = np.zeros(p, dtype=NP_DOUBLE_D_TYPE)
        double prev_s_sum = 0, current_s_sum = 0

        double kappa = 0
        double[::1] qr_work = np.zeros(k, dtype=NP_DOUBLE_D_TYPE)
        double[::1] Rz = np.zeros(k, dtype=NP_DOUBLE_D_TYPE)
        double[::1] y = np.zeros(k, dtype=NP_DOUBLE_D_TYPE)
        double* z = NULL
        double b = 0, c = 0

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
            # z = U[j, :]
            z = &U[j, 0]

            kappa = 2 / (s[j] - 2 * d[j])
            qr_update(k, Q, R, &kappa, z, z, &qr_work[0])

            # Rz = R @ z
            dcopy(&k, z, &inc_1, &Rz[0], &inc_1)
            dtrmv(low, trans, not_unit, &k, &R[0, 0], &k, &Rz[0], &inc_1)

            # y = Q @ R @ z
            dgemv(no_trans, &k, &k, &one, &Q[0, 0], &k, &Rz[0], &inc_1, &zero, &y[0], &inc_1)
            # y = Q @ R @ z - z
            daxpy(&k, &minus_one, z, &inc_1, &y[0], &inc_1)

            # b = z @ y
            b = ddot(&k, z, &inc_1, &y[0], &inc_1)

            # Rz = Rz - Q.T @ z
            dgemv(trans, &k, &k, &minus_one, &Q[0, 0], &k, z, &inc_1, &one, &Rz[0], &inc_1)

            # Solve triangular
            dtrsv(low, trans, not_unit, &k, &R[0, 0], &k, &Rz[0], &inc_1)

            # c = y @ Rz
            c = ddot(&k, &y[0], &inc_1, &Rz[0], &inc_1)

            s[j] = _clip(2 * diag_Sigma[j] - 2 * b - lam + 2 * c, 0, 1)

            kappa = 2 / (2 * d[j] - s[j])
            qr_update(k, Q, R, &kappa, z, z, &qr_work[0])

            current_s_sum += s[j]

        objectives.append(current_s_sum)

        lam = lam * mu
        if lam < lam_min:
            break

    return s, objectives


def _sdp_low_rank(
    d,
    U,
    max_iterations=None,
    lam=None,
    mu=None,
    tol=-1,
    eps=1e-5,
    return_objectives=False
):
    # Arrays must be C-contiguous, finite, and the data type C double
    d = np.ascontiguousarray(d, dtype=NP_DOUBLE_D_TYPE)
    U = np.ascontiguousarray(U, dtype=NP_DOUBLE_D_TYPE)
    if not np.all(np.isfinite(d)):
        raise ValueError(f'The diagonal D contains NaNs or Infs')
    if not np.all(np.isfinite(U)):
        raise ValueError(f'The low rank U matrix contains NaNs or Infs')

    # Dimensions checks
    if d.ndim != 1:
        raise ValueError(f'The diagonal D must be a one-dimensional array')
    p = d.shape[0]
    if U.ndim == 1:
        U = np.reshape(U, (-1, 1))
    elif U.ndim != 2:
        raise ValueError(f'The low rank U matrix must be a one or two dimensional array')
    k = U.shape[1]
    if p != U.shape[0]:
        raise ValueError(f'Dimensions between D and U don\'t match')

    ztz = np.sum(U * U, axis=1)
    diag_Sigma = d + ztz
    Q, R = qr(np.eye(k) + (U.T / d) @ U)

    # Default lambda, mu and max_iterations
    if mu is None:
        mu = 0.8
    elif mu >= 1 or mu <= 0:
        raise ValueError(f"The decay parameter mu must be between 0 and 1. Found {mu}.")
    # Find the first value of lambda such that a coordinate of s will change
    if lam is None:
        diag_inv_2Sigma = 0.5 / d - 0.5 * np.sum((U @ solve_triangular(R, np.eye(k))) * (U @ Q), axis=1) / d / d
        lam = (1 / diag_inv_2Sigma).max()
        lam = mu * lam
    elif lam <= 0:
        raise ValueError(f"The barrier parameter lam must be positive. Found {lam}.")
    # if tol < 0:
    #     raise ValueError(f"The tolerance cannot be negative. Found {tol}.")
    if eps < 0:
        raise ValueError(f"eps cannot be negative. Found {eps}")
    if max_iterations is None:
        if eps == 0:
            raise ValueError(
                f"Can't automatically set the maximum number of iterations if eps is null"
            )
        max_iterations = np.ceil(np.log(eps / (3 * lam)) / np.log(mu)).astype(int) + 1
    lam_min = eps / 3  # Average eps. Default global one, divide by p too

    # Cython fast solver
    s, objectives = sdp_rank_k(
        p, k, d, U, diag_Sigma, Q, R, max_iterations, lam, mu, tol, lam_min
    )

    if return_objectives:
        return s, objectives
    else:
        return s
