# ================================================================ #
# _kramers.pxd — C-level declaration for the Kramers opacity class #
# ================================================================ #
#
# There is a single C class, C_KramersOpacity, parameterised by kappa0.
# The three Python wrappers (KramersFFOpacity, KramersBFOpacity,
# KramersOpacity) all instantiate this same C class with different
# default values of kappa0.
#
# Cimport to hold a typed reference:
#
#   from triceratops.radiation.opacity.models._kramers cimport C_KramersOpacity

from triceratops.radiation.opacity.opacity_base cimport C_GreyOpacityBase


cdef class C_KramersOpacity(C_GreyOpacityBase):
    # ln(κ_0) where κ_0 is in cm⁵ g⁻² K^{3.5}.
    # Stored in log space so _log_opacity is a fused-multiply-add with no
    # extra transcendental calls.
    cdef double _log_kappa0

    cdef double _opacity(self, double rho, double T) nogil
    cdef double _log_opacity(self, double log_T, double log_rho) nogil
    cdef double _dlogkappa_dlogrho(self, double log_T, double log_rho) nogil
    cdef double _dlogkappa_dlogT(self, double log_T, double log_rho) nogil
