cdef void cholupdate(int p, double[:, ::1] R, double[::1] u) nogil
cdef bint choldowndate(int p, double[:, ::1] R, double[::1] z) nogil
