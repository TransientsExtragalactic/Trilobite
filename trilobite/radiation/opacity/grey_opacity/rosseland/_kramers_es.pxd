# ================================================================ #
# _kramers_es.pxd — C-level declaration for combined Kramers + ES  #
# ================================================================ #
#
# Single C class parameterised by (kappa0, kappa_es).
# The three Python wrappers (KramersFFESOpacity, KramersBFESOpacity,
# KramersESOpacity) all instantiate this class with different defaults.
#
# Cimport to hold a typed reference:
#
#   from trilobite.radiation.opacity.models._kramers_es cimport C_KramersESOpacity

from trilobite.radiation.opacity.opacity_base cimport C_GreyOpacityBase


cdef class C_KramersESOpacity(C_GreyOpacityBase):
    # ln(κ_0) and ln(κ_es) — both precomputed in __cinit__.
    cdef double _log_kappa0
    cdef double _log_kappa_es

    cdef double _opacity(self, double rho, double T) nogil
    cdef double _log_opacity(self, double log_T, double log_rho) nogil
    cdef double _dlogkappa_dlogrho(self, double log_T, double log_rho) nogil
    cdef double _dlogkappa_dlogT(self, double log_T, double log_rho) nogil
