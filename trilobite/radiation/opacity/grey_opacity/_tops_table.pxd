"""Cython declarations for C_TOPSTableOpacity."""

from trilobite.radiation.opacity.opacity_base cimport C_GreyOpacityBase

cdef class C_TOPSTableOpacity(C_GreyOpacityBase):
    # log10(T [K]) grid, shape (n1,)
    cdef double[::1]    _g1
    # log10(rho [g/cm3]) grid, shape (n2,)
    cdef double[::1]    _g2
    # log10(kappa [cm**2/g]) table, shape (n1, n2); NaN = invalid
    cdef double[:, ::1] _lk
    cdef int _n1
    cdef int _n2
    # out-of-bounds mode: 0 = raise, 1 = clamp, 2 = nan
    cdef int _oob
    # precomputed ln(10)
    cdef double _LN10

    # Internal helpers
    cdef int    _bisect(self, double[::1] arr, int n, double val) nogil
    cdef double _interp(self, double x1, double x2, int i, int j) nogil
    cdef void   _coords(self, double log_T, double log_rho,
                        double *x1_out, double *x2_out) nogil
    cdef int    _find_cell(self, double x1, double x2,
                           int *i_out, int *j_out) nogil
    cdef double _nearest_finite(self, int ci, int cj) nogil
    cdef double _handle_oob(self, double x1, double x2) nogil
