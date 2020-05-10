import numpy as np
from scipy.linalg import cholesky, solve_triangular

cimport cython
cimport numpy as np

from libc.math cimport sqrt
from libc.math cimport abs as cabs

from scipy.linalg.cython_blas cimport dscal, ddot, dtrsv

from fanok.utils._dtypes import NP_DOUBLE_D_TYPE
from fanok.utils._cholesky cimport __cholupdate, __choldowndate


cdef double _clip(double x, double amin, double amax) nogil:
    return min(max(x, amin), amax)


@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef __full_rank(
    int p,
    double[:, ::1] Sigma,
    double[:, ::1] R,
    int max_iterations,
    double lam,
    double mu,
    double tol,
    double lam_min
):
    objectives = [0]

    cdef:
        double prev_s_sum = 0, current_s_sum = 0
        np.ndarray[double, ndim=1] s = np.zeros(p, dtype=NP_DOUBLE_D_TYPE)
        double[::1] y_tilde = np.empty(p, dtype=NP_DOUBLE_D_TYPE)
        int inc_1 = 1
        double[::1] choldate_buffer = np.zeros(p, dtype=NP_DOUBLE_D_TYPE)
        double zero = 0
        double beta, xi, c, sj, delta
        char* low = 'L'
        char* no_trans = 'N'
        char* diag = 'N'

    for n_iter in range(max_iterations):
        prev_s_sum = current_s_sum
        current_s_sum = 0
        for j in range(p):
            with nogil:
                # Build y_tilde
                for k in range(p):
                    y_tilde[k] = Sigma[j, k]
                y_tilde[j] = 0

                # Solve triangular
                dtrsv(low, no_trans, diag, &p, &R[0, 0], &p, &y_tilde[0], &inc_1)

                beta = 4 * ddot(&p, &y_tilde[0], &inc_1, &y_tilde[0], &inc_1)
                xi = 2 * Sigma[j, j] - s[j]
                c = beta * xi / (xi + beta)

                sj = _clip(2 * Sigma[j, j] - c - lam, 0, 1)
                delta = s[j] - sj

            if delta != 0:
                choldate_buffer[j] = sqrt(cabs(delta))
                if delta > 0:
                    __cholupdate(p, R, choldate_buffer)
                else:
                    if not __choldowndate(p, R, choldate_buffer):
                        return s, objectives

                # the buffer was overwritten, nullify it
                dscal(&p, &zero, &choldate_buffer[0], &inc_1)

                s[j] = sj

            current_s_sum += sj

        objectives.append(current_s_sum)

        if current_s_sum == 0 or abs(current_s_sum - prev_s_sum) / current_s_sum < tol:
            break

        if lam < lam_min:
            break

        lam = lam * mu

    return s, objectives


def _full_rank(
    Sigma,
    max_iterations=None,
    lam=None,
    mu=None,
    tol=-1,  # This parameter will likely disappear in the future
    eps=1e-5,
    return_objectives=False
):
    # Arrays must be C-contiguous, finite, and the data type C double
    Sigma = np.ascontiguousarray(Sigma, dtype=NP_DOUBLE_D_TYPE)
    if not np.all(np.isfinite(Sigma)):
        raise ValueError(f'Sigma contains NaNs or Infs')

    # Dimensions checks
    if Sigma.ndim != 2:
        raise ValueError(f'Sigma must be a one or two dimensional array')
    p = Sigma.shape[0]
    if p != Sigma.shape[1]:
        raise ValueError(f'Sigma must be a square matrix')

    R = cholesky(2 * Sigma, lower=True).T  # Upper triangular + C-contiguous

    # Default lambda, mu and max_iterations and sanity checks
    if mu is None:
        mu = 0.8
    elif mu >= 1 or mu <= 0:
        raise ValueError(f"The decay parameter mu must be between 0 and 1. Found {mu}.")
    # Find the first value of lambda such that a coordinate of s will change
    if lam is None:
        W = solve_triangular(R, np.identity(p))
        lam = (1 / np.sum(W * W, axis=1)).max()
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
    s, objectives = __full_rank(p, Sigma, R, max_iterations, lam, mu, tol, lam_min)

    if return_objectives:
        return s, objectives
    else:
        return s
