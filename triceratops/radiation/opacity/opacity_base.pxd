# ============================================================== #
# opacity_base.pxd — C-level declarations for opacity bases      #
# ============================================================== #
#
# Two C-level base classes are declared here:
#
#   C_GreyOpacityBase   — grey (frequency-independent) opacity laws.
#                         Methods take (log_T, log_rho).
#                         Cimport for disk closures and other hot-path
#                         callers that only need grey opacities.
#
#   C_OpacityBase       — general frequency-dependent opacity laws.
#                         Methods take (log_T, log_rho, log_nu).
#                         Subclass this for multigroup or line opacities.
#
# Cimport usage:
#
#   from triceratops.radiation.opacity.opacity_base cimport C_GreyOpacityBase
#   cdef C_GreyOpacityBase _grey_kappa
#
#   from triceratops.radiation.opacity.opacity_base cimport C_OpacityBase
#   cdef C_OpacityBase _kappa
#
# Design
# ------
# The ``cdef`` methods are the C hot path, they accept bare C doubles and
# can be called at C speed via a typed pointer.  Concrete subclasses override
# the underscore-prefixed ``cdef`` methods; Python wrappers (plain ``def``)
# are defined in opacity_base.pyx and inherited for free.
#
# Argument order
# --------------
# Grey:        (log_T, log_rho)
# Non-grey:    (log_T, log_rho, log_nu)
#
# Units: kappa in cm^2 g^-1; rho in g cm^-3; T in K; nu in Hz.

cdef class C_GreyOpacityBase:

    # ---- C hot path (cdef) -----------------------------------------
    # _opacity default: exp(_log_opacity(log T, log rho)).
    # _log_opacity is the primary method to override.

    cdef double _opacity(self, double rho, double T) nogil

    cdef double _log_opacity(self, double log_T, double log_rho) nogil

    cdef double _dlogkappa_dlogrho(self, double log_T, double log_rho) nogil

    cdef double _dlogkappa_dlogT(self, double log_T, double log_rho) nogil


cdef class C_OpacityBase:

    # ---- C hot path (cdef) -----------------------------------------
    # _opacity default: exp(_log_opacity(log T, log rho, log nu)).
    # _log_opacity is the primary method to override.

    cdef double _opacity(self, double nu, double rho, double T) nogil

    cdef double _log_opacity(self, double log_nu, double log_T, double log_rho) nogil

    cdef double _dlogkappa_dlogrho(self, double log_nu, double log_T, double log_rho) nogil

    cdef double _dlogkappa_dlogT(self, double log_nu, double log_T, double log_rho) nogil

    cdef double _dlogkappa_dlognu(self, double log_nu, double log_T, double log_rho) nogil
