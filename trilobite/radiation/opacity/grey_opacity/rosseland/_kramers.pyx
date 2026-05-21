#cython: language_level=3, boundscheck=False
r"""
Kramers power-law opacity — Cython implementation.

A single C class, :class:`C_KramersOpacity`, implements this physics.
The three Python wrapper classes (:class:`~.KramersFFOpacity`,
:class:`~.KramersBFOpacity`, :class:`~.KramersOpacity`) all instantiate
this class with the appropriate physical default for ``kappa0``.

Standard coefficients (solar composition, no detailed Gaunt factor)
--------------------------------------------------------------------
These are textbook estimates; see the Python wrappers for the values used.
When composition and Gaunt-factor data are available, pass ``kappa0``
explicitly to the Python wrapper.

* Free-free  : :math:`\kappa_{\rm ff,0} \approx 3.68\times10^{22}\ \mathrm{cm^5\,g^{-2}\,K^{3.5}}`
* Bound-free : :math:`\kappa_{\rm bf,0} \approx 4.34\times10^{25}\ \mathrm{cm^5\,g^{-2}\,K^{3.5}}`
* Combined   : :math:`\kappa_{\rm ff+bf,0} = \kappa_{\rm ff,0} + \kappa_{\rm bf,0}`
  (valid because both terms share the same :math:`\rho` and :math:`T` power-law indices)

Units: :math:`\kappa` in :math:`\mathrm{cm^2\,g^{-1}}`, :math:`\rho` in
:math:`\mathrm{g\,cm^{-3}}`, T in K, :math:`\kappa_0` in
:math:`\mathrm{cm^5\,g^{-2}\,K^{3.5}}`.
"""
from libc.math cimport exp, log

from trilobite.radiation.opacity.opacity_base cimport C_GreyOpacityBase


cdef class C_KramersOpacity(C_GreyOpacityBase):
    r"""Kramers power-law opacity :math:`\kappa \propto \rho\,T^{-3.5}`.

    All three Python wrappers (FF, BF, combined) use this single C class;
    they differ only in the value of ``kappa0`` passed to ``__cinit__``.

    The ``def`` scalar and vectorized wrappers are **inherited** from
    :class:`~trilobite.radiation.opacity.opacity_base.C_GreyOpacityBase`.

    Parameters
    ----------
    kappa0 : double
        Normalisation :math:`\kappa_0` in :math:`\mathrm{cm^5\,g^{-2}\,K^{3.5}}`.
    """

    def __cinit__(self, double kappa0):
        # Store in log space — _log_opacity is then three FMAs, no log() call.
        self._log_kappa0 = log(kappa0)

    # ------------------------------------------------------------------ #
    #  C hot path                                                         #
    # ------------------------------------------------------------------ #

    cdef double _opacity(self, double rho, double T) nogil:
        r""":math:`\kappa_0\,\rho\,T^{-3.5}` in :math:`\mathrm{cm^2\,g^{-1}}`."""
        return exp(self._log_opacity(log(T), log(rho)))

    cdef double _log_opacity(self, double log_T, double log_rho) nogil:
        r""":math:`\ln\kappa_0 + \ln\rho - 3.5\,\ln T`."""
        return self._log_kappa0 + log_rho - 3.5 * log_T

    cdef double _dlogkappa_dlogrho(self, double log_T, double log_rho) nogil:
        r""":math:`\partial\ln\kappa/\partial\ln\rho` = +1 (Kramers)."""
        return 1.0

    cdef double _dlogkappa_dlogT(self, double log_T, double log_rho) nogil:
        r""":math:`\partial\ln\kappa/\partial\ln T` = -3.5 (Kramers)."""
        return -3.5
