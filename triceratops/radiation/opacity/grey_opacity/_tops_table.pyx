#cython: language_level=3, boundscheck=False
r"""
C-level bilinear-interpolation opacity from a pre-tabulated 2-D TOPS grid.

:class:`C_TOPSTableOpacity` stores a single ``(n1 × n2)`` table of
:math:`\log_{10}\kappa` on a ``(\log_{10}T,\,\log_{10}\rho)`` grid and
evaluates opacity and its log-space partial derivatives at arbitrary
``(T, \rho)`` via bilinear interpolation.

All public log-space methods accept **(log_T, log_rho)** in *natural* log
(convention shared with every other Cython opacity class).

Out-of-bounds modes
-------------------
* ``0`` (raise) — :exc:`ValueError` raised (acquires GIL).
* ``1`` (clamp) — nearest valid boundary cell returned.
* ``2`` (nan)   — :data:`NAN` returned silently.
"""

import numpy as np
from libc.math cimport log, exp, isnan, NAN

cdef class C_TOPSTableOpacity(C_GreyOpacityBase):
    r"""Bilinear-interpolation opacity from a 2-D TOPS log-space table.

    Parameters
    ----------
    g1 : ndarray, float64, C-contiguous, shape (n1,)
        :math:`\log_{10}(T\,[\mathrm{K}])` grid values (strictly increasing).
    g2 : ndarray, float64, C-contiguous, shape (n2,)
        :math:`\log_{10}(\rho\,[\mathrm{g\,cm^{-3}}])` grid values (strictly increasing).
    lk : ndarray, float64, C-contiguous, shape (n1, n2)
        :math:`\log_{10}(\kappa\,[\mathrm{cm^2\,g^{-1}}])`.
        ``NaN`` marks out-of-range cells.
    oob : int
        Out-of-bounds mode: ``0`` raise, ``1`` clamp, ``2`` nan.
    """

    def __cinit__(
        self,
        double[::1] g1,
        double[::1] g2,
        double[:, ::1] lk,
        int oob,
    ):
        self._g1   = g1
        self._g2   = g2
        self._lk   = lk
        self._n1   = g1.shape[0]
        self._n2   = g2.shape[0]
        self._oob  = oob
        self._LN10 = log(10.0)

    # ================================================================ #
    #  Internal helpers                                                #
    # ================================================================ #

    cdef int _bisect(self, double[::1] arr, int n, double val) nogil:
        """Return i such that arr[i] <= val < arr[i+1]; clamped to [0, n-2]."""
        cdef int lo = 0, hi = n - 1, mid
        while lo < hi - 1:
            mid = (lo + hi) >> 1
            if arr[mid] <= val:
                lo = mid
            else:
                hi = mid
        return lo

    cdef double _interp(self, double x1, double x2, int i, int j) nogil:
        """Bilinear interpolation of _lk at (x1, x2) within cell (i, j).

        x1 is in [g1[i], g1[i+1]], x2 is in [g2[j], g2[j+1]].
        Returns NAN if any of the four corners is NaN.
        """
        cdef double k00, k10, k01, k11, tx, ty
        k00 = self._lk[i,   j  ]
        k10 = self._lk[i+1, j  ]
        k01 = self._lk[i,   j+1]
        k11 = self._lk[i+1, j+1]
        if isnan(k00) or isnan(k10) or isnan(k01) or isnan(k11):
            return NAN
        tx = (x1 - self._g1[i]) / (self._g1[i+1] - self._g1[i])
        ty = (x2 - self._g2[j]) / (self._g2[j+1] - self._g2[j])
        return (k00*(1.0-tx)*(1.0-ty)
              + k10*tx*(1.0-ty)
              + k01*(1.0-tx)*ty
              + k11*tx*ty)

    # ================================================================ #
    #  Coordinate conversion + cell lookup                             #
    # ================================================================ #

    cdef void _coords(
        self,
        double log_T,
        double log_rho,
        double *x1_out,
        double *x2_out,
    ) nogil:
        """Convert (ln T, ln rho) → (log10 T, log10 rho)."""
        x1_out[0] = log_T   / self._LN10
        x2_out[0] = log_rho / self._LN10

    cdef int _find_cell(
        self,
        double x1,
        double x2,
        int *i_out,
        int *j_out,
    ) nogil:
        """Locate grid cell.

        Returns 0 if in-bounds, 1 if out-of-bounds.
        Sets *i_out and *j_out to the lower-left corner.
        """
        cdef int i, j
        cdef double eps = 1e-9
        if x1 < self._g1[0] - eps or x1 > self._g1[self._n1-1] + eps:
            i_out[0] = 0
            j_out[0] = 0
            return 1
        if x2 < self._g2[0] - eps or x2 > self._g2[self._n2-1] + eps:
            i_out[0] = 0
            j_out[0] = 0
            return 1
        i = self._bisect(self._g1, self._n1, x1)
        j = self._bisect(self._g2, self._n2, x2)
        if i > self._n1 - 2:
            i = self._n1 - 2
        if j > self._n2 - 2:
            j = self._n2 - 2
        i_out[0] = i
        j_out[0] = j
        return 0

    cdef double _nearest_finite(self, int ci, int cj) nogil:
        """Return the nearest non-NaN cell value to (ci, cj) by Chebyshev distance."""
        cdef int k, i, j, i0, i1, j0, j1, max_k
        cdef double v
        max_k = self._n1 + self._n2
        for k in range(max_k):
            i0 = ci - k if ci - k > 0 else 0
            i1 = ci + k if ci + k < self._n1 - 1 else self._n1 - 1
            j0 = cj - k if cj - k > 0 else 0
            j1 = cj + k if cj + k < self._n2 - 1 else self._n2 - 1
            for i in range(i0, i1 + 1):
                for j in range(j0, j1 + 1):
                    if (ci - i == k or i - ci == k or
                            cj - j == k or j - cj == k):
                        v = self._lk[i, j]
                        if not isnan(v):
                            return v
        return NAN

    cdef double _handle_oob(self, double x1, double x2) nogil:
        """Apply out-of-bounds strategy and return log10(kappa) or NAN."""
        cdef int ci, cj
        if self._oob == 1:
            ci = self._bisect(self._g1, self._n1, x1)
            cj = self._bisect(self._g2, self._n2, x2)
            if ci > self._n1 - 2:
                ci = self._n1 - 2
            if cj > self._n2 - 2:
                cj = self._n2 - 2
            return self._nearest_finite(ci, cj)
        elif self._oob == 2:
            return NAN
        else:
            with gil:
                raise ValueError(
                    f"(log10_T={x1:.3f}, log10_rho={x2:.3f}) is outside the "
                    f"table domain "
                    f"[{self._g1[0]:.2f}, {self._g1[self._n1-1]:.2f}] x "
                    f"[{self._g2[0]:.2f}, {self._g2[self._n2-1]:.2f}]."
                )

    # ================================================================ #
    #  Hot-path cdef methods                                           #
    # ================================================================ #

    cdef double _log_opacity(self, double log_T, double log_rho) nogil:
        r"""Return :math:`\ln\kappa` (:math:`\mathrm{cm^2\,g^{-1}}`)."""
        cdef double x1, x2, log10_kappa
        cdef int i, j, oob
        self._coords(log_T, log_rho, &x1, &x2)
        oob = self._find_cell(x1, x2, &i, &j)
        if oob:
            log10_kappa = self._handle_oob(x1, x2)
        else:
            log10_kappa = self._interp(x1, x2, i, j)
            if isnan(log10_kappa) and self._oob == 1:
                log10_kappa = self._nearest_finite(i, j)
        return log10_kappa * self._LN10

    cdef double _opacity(self, double rho, double T) nogil:
        r"""Return :math:`\kappa` in linear space (:math:`\mathrm{cm^2\,g^{-1}}`)."""
        return exp(self._log_opacity(log(T), log(rho)))

    cdef double _dlogkappa_dlogrho(self, double log_T, double log_rho) nogil:
        r"""Return :math:`\partial\ln\kappa/\partial\ln\rho` (dimensionless).

        In the TOPS coordinate system the second axis is :math:`\log_{10}\rho`
        directly, so the chain rule gives the same result as OPAL:

        .. math::

            \frac{\partial\ln\kappa}{\partial\ln\rho}
              = \frac{\partial\log_{10}\kappa}{\partial\log_{10}\rho}
        """
        cdef double x1, x2, tx, k0j, k0j1, dkg2
        cdef int i, j, oob
        self._coords(log_T, log_rho, &x1, &x2)
        oob = self._find_cell(x1, x2, &i, &j)
        if oob:
            return NAN if self._oob == 2 else 0.0

        tx   = (x1 - self._g1[i]) / (self._g1[i+1] - self._g1[i])
        k0j  = self._lk[i,   j  ]*(1.0-tx) + self._lk[i+1, j  ]*tx
        k0j1 = self._lk[i,   j+1]*(1.0-tx) + self._lk[i+1, j+1]*tx
        if isnan(k0j) or isnan(k0j1):
            return NAN if self._oob == 2 else 0.0

        dkg2 = (k0j1 - k0j) / (self._g2[j+1] - self._g2[j])
        return dkg2

    cdef double _dlogkappa_dlogT(self, double log_T, double log_rho) nogil:
        r"""Return :math:`\partial\ln\kappa/\partial\ln T` (dimensionless).

        In the TOPS coordinate system :math:`\rho` and :math:`T` are independent
        axes, so there is no cross-term:

        .. math::

            \frac{\partial\ln\kappa}{\partial\ln T}\bigg|_\rho
              = \frac{\partial\log_{10}\kappa}{\partial\log_{10}T}\bigg|_\rho
        """
        cdef double x1, x2, ty, ki0, ki1, dkg1
        cdef int i, j, oob
        self._coords(log_T, log_rho, &x1, &x2)
        oob = self._find_cell(x1, x2, &i, &j)
        if oob:
            return NAN if self._oob == 2 else 0.0

        ty  = (x2 - self._g2[j]) / (self._g2[j+1] - self._g2[j])
        ki0 = self._lk[i,   j]*(1.0-ty) + self._lk[i,   j+1]*ty
        ki1 = self._lk[i+1, j]*(1.0-ty) + self._lk[i+1, j+1]*ty
        if isnan(ki0) or isnan(ki1):
            return NAN if self._oob == 2 else 0.0
        dkg1 = (ki1 - ki0) / (self._g1[i+1] - self._g1[i])

        return dkg1
