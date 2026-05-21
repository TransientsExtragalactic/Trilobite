#cython: language_level=3, boundscheck=False
r"""
Combined Kramers + electron-scattering opacity — Cython implementation.

A single C class, :class:`C_KramersESOpacity`, implements the total opacity

.. math::

    \kappa(\rho, T) = \kappa_{\rm es} + \kappa_0\,\rho\,T^{-3.5},

where the first term is the constant Thomson contribution and the second is
the Kramers power-law.  The three Python wrapper classes
(:class:`~.KramersFFESOpacity`, :class:`~.KramersBFESOpacity`,
:class:`~.KramersESOpacity`) instantiate this class with the standard physical
defaults for ``kappa0``.

Implementation notes
--------------------
``_log_opacity`` uses the log-sum-exp identity

.. math::

    \ln(\kappa_{\rm es} + \kappa_{\rm KR})
        = \ln\kappa_> + \ln\!\left(1 + e^{\ln\kappa_< - \ln\kappa_>}\right)

(where :math:`\kappa_>` is whichever term is larger) to avoid numerical
overflow or catastrophic cancellation.

The partial derivatives follow from the chain rule.  With
:math:`w = \kappa_{\rm KR} / \kappa`:

.. math::

    \frac{\partial\ln\kappa}{\partial\ln\rho} = w, \qquad
    \frac{\partial\ln\kappa}{\partial\ln T}   = -3.5\,w.

Both require one evaluation of ``_log_opacity`` to obtain
:math:`\ln\kappa_{\rm KR}` and :math:`\ln\kappa`.
"""
from libc.math cimport exp, log

from trilobite.radiation.opacity.opacity_base cimport C_GreyOpacityBase


cdef class C_KramersESOpacity(C_GreyOpacityBase):
    r"""Combined Kramers + electron-scattering opacity.

    .. math::

        \kappa(\rho, T) = \kappa_{\rm es} + \kappa_0\,\rho\,T^{-3.5}

    Parameters
    ----------
    kappa0 : double
        Kramers normalisation in :math:`\mathrm{cm^5\,g^{-2}\,K^{3.5}}`.
    kappa_es : double
        Electron-scattering opacity in :math:`\mathrm{cm^2\,g^{-1}}`.
    """

    def __cinit__(self, double kappa0, double kappa_es=0.34):
        self._log_kappa0  = log(kappa0)
        self._log_kappa_es = log(kappa_es)

    # ------------------------------------------------------------------ #
    #  C hot path                                                         #
    # ------------------------------------------------------------------ #

    cdef double _opacity(self, double rho, double T) nogil:
        r""":math:`\kappa_{\rm es} + \kappa_0\,\rho\,T^{-3.5}` in :math:`\mathrm{cm^2\,g^{-1}}`."""
        return exp(self._log_opacity(log(T), log(rho)))

    cdef double _log_opacity(self, double log_T, double log_rho) nogil:
        r""":math:`\ln(\kappa_{\rm es} + \kappa_{\rm KR})` via log-sum-exp."""
        cdef double log_kappa_kr = self._log_kappa0 + log_rho - 3.5 * log_T
        cdef double a = self._log_kappa_es
        cdef double b = log_kappa_kr
        # log(exp(a) + exp(b)) = max + log(1 + exp(min - max))
        if a >= b:
            return a + log(1.0 + exp(b - a))
        else:
            return b + log(1.0 + exp(a - b))

    cdef double _dlogkappa_dlogrho(self, double log_T, double log_rho) nogil:
        r""":math:`\kappa_{\rm KR}/\kappa` — fraction of opacity from Kramers."""
        cdef double log_kappa_kr = self._log_kappa0 + log_rho - 3.5 * log_T
        cdef double log_kappa    = self._log_opacity(log_T, log_rho)
        return exp(log_kappa_kr - log_kappa)

    cdef double _dlogkappa_dlogT(self, double log_T, double log_rho) nogil:
        r""":math:`-3.5\,\kappa_{\rm KR}/\kappa`."""
        cdef double log_kappa_kr = self._log_kappa0 + log_rho - 3.5 * log_T
        cdef double log_kappa    = self._log_opacity(log_T, log_rho)
        return -3.5 * exp(log_kappa_kr - log_kappa)
