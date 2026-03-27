# ============================================================== #
# opacity_base.pxd — C-level declarations for C_GreyOpacityBase  #
# ============================================================== #
#
# Cimport this file to hold a typed reference to any opacity object:
#
#   from triceratops.radiation.opacity.opacity_base cimport C_GreyOpacityBase
#   cdef C_GreyOpacityBase _opacity
#
# Design
# ------
# The four ``cdef`` methods below are the C hot path — they accept bare
# C doubles and can be inlined/statically dispatched by the compiler.
# They carry an underscore prefix by convention: concrete subclasses
# override ``_log_opacity`` etc., while callers at the C level invoke
# the ``cdef`` methods directly via a typed pointer.
#
# The corresponding Python-accessible wrappers (``log_opacity``,
# ``dlogkappa_dlogrho``, ``dlogkappa_dlogT``) are plain ``def`` methods
# defined in opacity_base.pyx.  They simply delegate to the ``cdef``
# equivalents, adding negligible overhead.  Only the ``def`` methods need
# to appear in this .pxd; the cdef declarations suffice for Cython callers.
#
# Argument order
# --------------
# All log-space methods follow the convention (log_T, log_rho) — i.e.
# temperature first — matching the ``kappa_func_t`` signature used by
# the accretion-disk closures in
# ``triceratops.dynamics.accretion.one_zone.closure``.
#
# Units: κ in cm² g⁻¹; ρ in g cm⁻³; T in K.

cdef class C_GreyOpacityBase:

    # ---- C hot path (cdef) -----------------------------------------
    # Override these in every concrete subclass.  Callers that cimport
    # this type can call them directly as fast C method calls.

    cdef double _opacity(self, double rho, double T) except *

    # log_T first, log_rho second — matches kappa_func_t convention.
    cdef double _log_opacity(self, double log_T, double log_rho) except *

    cdef double _dlogkappa_dlogrho(self, double log_T, double log_rho) except *

    cdef double _dlogkappa_dlogT(self, double log_T, double log_rho) except *
