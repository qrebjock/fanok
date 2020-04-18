cdef void reduced_hessenberg_qr(int p, double[::1, :] Q, double[:, ::1] R) nogil
cdef void hessenberg_qr(int p, double[::1, :] Q, double[:, ::1] R) nogil
cdef void hessenberg_qr_update(
    int p, double[::1, :] Q, double[:, ::1] R,
    double* kappa, double* u, double* v, double* w
) nogil
cdef void qr_update(
    int p, double[::1, :] Q, double[:, ::1] R,
    double* kappa, double* u, double* v, double* w
) nogil
