# ================================================================ #
# _electron_scattering.pxd — C-level declaration                  #
# ================================================================ #
#
# Cimport to hold a typed reference to an ES opacity object:
#
#   from trilobite.radiation.opacity.models._electron_scattering \
#       cimport C_ElectronScatteringOpacity

from trilobite.radiation.opacity.opacity_base cimport C_GreyOpacityBase


cdef class C_ElectronScatteringOpacity(C_GreyOpacityBase):

    # log(κ_es) — precomputed once in __cinit__.
    # Every call to _log_opacity is a single memory read; no arithmetic.
    cdef double _log_kappa_es

    cdef double _opacity(self, double rho, double T) nogil
    cdef double _log_opacity(self, double log_T, double log_rho) nogil
    cdef double _dlogkappa_dlogrho(self, double log_T, double log_rho) nogil
    cdef double _dlogkappa_dlogT(self, double log_T, double log_rho) nogil
