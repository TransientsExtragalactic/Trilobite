#cython: language_level=3, boundscheck=False
r"""
Electron-scattering (Thomson) opacity.


Electron scattering is the dominant opacity source in hot, fully ionised
plasma.  To leading order it is independent of density and temperature:

.. math::

    \kappa_{\rm es} = \frac{\sigma_T}{m_p} \frac{1+X}{2}
                    \approx 0.34\,{\rm cm^2\,g^{-1}}

for solar hydrogen mass fraction :math:`X \approx 0.70`.

This class stores :math:`\ln\kappa_{\rm es}` as a scalar in ``__cinit__``
so that ``_log_opacity`` is a single memory load with zero arithmetic —
the natural hot-path implementation for a constant opacity.

"""
from libc.math cimport exp, log

from triceratops.radiation.opacity.opacity_base cimport C_GreyOpacityBase


cdef class C_ElectronScatteringOpacity(C_GreyOpacityBase):
    r"""Constant electron-scattering opacity.

    ``_log_opacity`` returns ``self._log_kappa_es`` regardless of (``log_T``, ``log_rho``).
    The Python ``def`` wrappers are inherited from :class:`C_GreyOpacityBase`.
    """

    def __cinit__(self, double kappa_es=0.34):
        # Precompute ln(kappa_es) once.  All four cdef methods read this scalar.
        self._log_kappa_es = log(kappa_es)

    # ------------------------------------------------------------------ #
    #  C hot path                                                         #
    # ------------------------------------------------------------------ #

    cdef double _opacity(self, double rho, double T) nogil:
        r""":math:`\kappa_{\rm es}` (:math:`\mathrm{cm^2\,g^{-1}}`) — same for all (:math:`\rho`, T)."""
        return exp(self._log_kappa_es)

    cdef double _log_opacity(self, double log_T, double log_rho) nogil:
        r""":math:`\ln\kappa_{\rm es}` — independent of ``log_T`` and ``log_rho``."""
        return self._log_kappa_es

    cdef double _dlogkappa_dlogrho(self, double log_T, double log_rho) nogil:
        r"""0 — ES opacity has no density dependence."""
        return 0.0

    cdef double _dlogkappa_dlogT(self, double log_T, double log_rho) nogil:
        r"""0 — ES opacity has no temperature dependence."""
        return 0.0
