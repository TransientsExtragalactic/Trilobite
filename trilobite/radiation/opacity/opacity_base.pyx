#cython: language_level=3, boundscheck=False
r"""
C-level base classes for opacity laws.

This module contains the core low-level logic for the opacity class structure and will
provide the core structure for any Cython extension classes that implement specific opacity laws.  The two base
classes defined here are:

- :class:`C_GreyOpacityBase` defines a two-tier interface for grey (frequency-independent) opacities.
- :class:`C_OpacityBase` is the corresponding template for frequency-dependent (non-grey) opacities.

Notes
-----

There are generically two branches of implementation for each base class:

- ``cdef`` methods (underscore-prefixed) overridden in
  subclasses.  Called directly at C speed by disk and radiative-transfer loops.
- scalar ``def`` wrappers and vectorized
  ``*_array`` variants (``double[::1]`` memoryviews) inherited for free.

All log-space methods use the **(log_T, log_rho)** argument order, matching
the ``kappa_func_t`` function-pointer convention in the accretion-disk closures.
Units: :math:`\kappa` in :math:`\mathrm{cm^2\,g^{-1}}`, :math:`\rho` in
:math:`\mathrm{g\,cm^{-3}}`, T in K.
"""
from libc.math cimport exp, log

import numpy as np

# ======================================================================= #
#  C_OpacityBase — frequency-dependent (non-grey) opacity template        #
# ======================================================================= #
cdef class C_OpacityBase:
    r"""Abstract C-level base class for frequency-dependent opacity laws.

    Subclasses override the four ``cdef`` methods; scalar and vectorized
    Python wrappers are inherited and require no re-implementation.

    All log-space methods use the **(log_nu, log_T, log_rho)** argument order —
    frequency first.  Units: :math:`\kappa` in :math:`\mathrm{cm^2\,g^{-1}}`,
    :math:`\rho` in :math:`\mathrm{g\,cm^{-3}}`, T in K, :math:`\nu` in Hz.
    """

    # ================================================================ #
    #  C hot path — override these in every concrete subclass          #
    # ================================================================ #

    cdef double _opacity(self, double nu, double rho, double T) nogil:
        r""":math:`\kappa` (:math:`\mathrm{cm^2\,g^{-1}}`) in linear space.

        Default delegates to ``_log_opacity``; override only when a direct
        formula is cheaper than the log-space round-trip.
        """
        return exp(self._log_opacity(log(nu), log(T), log(rho)))

    cdef double _log_opacity(self, double log_nu, double log_T, double log_rho) nogil:
        r""":math:`\ln\kappa` (:math:`\mathrm{log\,cm^2\,g^{-1}}`).  *Primary method to implement.*"""
        with gil:
            raise NotImplementedError(
                f"{type(self).__name__} must implement _log_opacity(log_nu, log_T, log_rho)."
            )

    cdef double _dlogkappa_dlogrho(self, double log_nu, double log_T, double log_rho) nogil:
        r""":math:`\partial\ln\kappa/\partial\ln\rho` (dimensionless)."""
        with gil:
            raise NotImplementedError(
                f"{type(self).__name__} must implement _dlogkappa_dlogrho(log_nu, log_T, log_rho)."
            )

    cdef double _dlogkappa_dlogT(self, double log_nu, double log_T, double log_rho) nogil:
        r""":math:`\partial\ln\kappa/\partial\ln T` (dimensionless)."""
        with gil:
            raise NotImplementedError(
                f"{type(self).__name__} must implement _dlogkappa_dlogT(log_nu, log_T, log_rho)."
            )

    cdef double _dlogkappa_dlognu(self, double log_nu, double log_T, double log_rho) nogil:
        r""":math:`\partial\ln\kappa/\partial\ln\nu` (dimensionless)."""
        with gil:
            raise NotImplementedError(
                f"{type(self).__name__} must implement _dlogkappa_dlognu(log_nu, log_T, log_rho)."
            )

    # ================================================================ #
    #  Scalar Python wrappers — inherited; do not override             #
    # ================================================================ #

    def opacity(self, double nu, double rho, double T):
        r""":math:`\kappa` (:math:`\mathrm{cm^2\,g^{-1}}`) — scalar Python wrapper for ``_opacity``."""
        return self._opacity(nu, rho, T)

    def log_opacity(self, double log_nu, double log_T, double log_rho):
        r""":math:`\ln\kappa` — scalar Python wrapper for ``_log_opacity``."""
        return self._log_opacity(log_nu, log_T, log_rho)

    def dlogkappa_dlogrho(self, double log_nu, double log_T, double log_rho):
        r""":math:`\partial\ln\kappa/\partial\ln\rho` — scalar Python wrapper."""
        return self._dlogkappa_dlogrho(log_nu, log_T, log_rho)

    def dlogkappa_dlogT(self, double log_nu, double log_T, double log_rho):
        r""":math:`\partial\ln\kappa/\partial\ln T` — scalar Python wrapper."""
        return self._dlogkappa_dlogT(log_nu, log_T, log_rho)

    def dlogkappa_dlognu(self, double log_nu, double log_T, double log_rho):
        r""":math:`\partial\ln\kappa/\partial\ln\nu` — scalar Python wrapper."""
        return self._dlogkappa_dlognu(log_nu, log_T, log_rho)

    # ================================================================ #
    #  Vectorized Python wrappers — inherited; do not override         #
    # ================================================================ #

    def log_opacity_array(self, double[::1] log_nu, double[::1] log_T, double[::1] log_rho):
        r""":math:`\ln\kappa` evaluated over 1-D arrays *log_nu*, *log_T*, *log_rho*.

        Parameters
        ----------
        log_nu, log_T, log_rho : ndarray, dtype=float64, C-contiguous

        Returns
        -------
        ndarray of float64
        """
        cdef int n = log_nu.shape[0]
        cdef double[::1] out = np.empty(n, dtype=np.float64)
        for i in range(n):
            out[i] = self._log_opacity(log_nu[i], log_T[i], log_rho[i])
        return np.asarray(out)

    def dlogkappa_dlogrho_array(self, double[::1] log_nu, double[::1] log_T, double[::1] log_rho):
        r""":math:`\partial\ln\kappa/\partial\ln\rho` evaluated over 1-D arrays.

        Parameters
        ----------
        log_nu, log_T, log_rho : ndarray, dtype=float64, C-contiguous

        Returns
        -------
        ndarray of float64
        """
        cdef int n = log_nu.shape[0]
        cdef double[::1] out = np.empty(n, dtype=np.float64)
        for i in range(n):
            out[i] = self._dlogkappa_dlogrho(log_nu[i], log_T[i], log_rho[i])
        return np.asarray(out)

    def dlogkappa_dlogT_array(self, double[::1] log_nu, double[::1] log_T, double[::1] log_rho):
        r""":math:`\partial\ln\kappa/\partial\ln T` evaluated over 1-D arrays.

        Parameters
        ----------
        log_nu, log_T, log_rho : ndarray, dtype=float64, C-contiguous

        Returns
        -------
        ndarray of float64
        """
        cdef int n = log_nu.shape[0]
        cdef double[::1] out = np.empty(n, dtype=np.float64)
        for i in range(n):
            out[i] = self._dlogkappa_dlogT(log_nu[i], log_T[i], log_rho[i])
        return np.asarray(out)

    def dlogkappa_dlognu_array(self, double[::1] log_nu, double[::1] log_T, double[::1] log_rho):
        r""":math:`\partial\ln\kappa/\partial\ln\nu` evaluated over 1-D arrays.

        Parameters
        ----------
        log_nu, log_T, log_rho : ndarray, dtype=float64, C-contiguous

        Returns
        -------
        ndarray of float64
        """
        cdef int n = log_nu.shape[0]
        cdef double[::1] out = np.empty(n, dtype=np.float64)
        for i in range(n):
            out[i] = self._dlogkappa_dlognu(log_nu[i], log_T[i], log_rho[i])
        return np.asarray(out)

# ======================================================================= #
# Grey opacity base class — frequency-independent opacity template        #
# ======================================================================= #
cdef class C_GreyOpacityBase:
    r"""Abstract C-level base class for grey (frequency-independent) opacity laws.

    Subclasses override the four ``cdef`` methods; scalar and vectorized
    Python wrappers are inherited and require no re-implementation.
    """

    # ================================================================ #
    #  C hot path — override these in every concrete subclass          #
    # ================================================================ #

    cdef double _opacity(self, double rho, double T) nogil:
        r""":math:`\kappa` (:math:`\mathrm{cm^2\,g^{-1}}`) in linear space.

        Default delegates to ``_log_opacity``; override only when a direct
        formula is cheaper than the log-space round-trip.
        """
        return exp(self._log_opacity(log(T), log(rho)))

    cdef double _log_opacity(self, double log_T, double log_rho) nogil:
        r""":math:`\ln\kappa` (:math:`\mathrm{log\,cm^2\,g^{-1}}`).  *Primary method to implement.*

        Takes ``log_T`` :math:`= \ln T` and ``log_rho`` :math:`= \ln\rho`,
        returns :math:`\ln\kappa`.  Called in the hot path by disk closures
        and radiative-transfer loops.
        """
        with gil:
            raise NotImplementedError(
                f"{type(self).__name__} must implement _log_opacity(log_T, log_rho)."
            )

    cdef double _dlogkappa_dlogrho(self, double log_T, double log_rho) nogil:
        r""":math:`\partial\ln\kappa/\partial\ln\rho` (dimensionless).

        For power-law :math:`\kappa \propto \rho^a` returns *a*.  For
        :math:`\rho`-independent opacity returns 0.
        """
        with gil:
            raise NotImplementedError(
                f"{type(self).__name__} must implement _dlogkappa_dlogrho(log_T, log_rho)."
            )

    cdef double _dlogkappa_dlogT(self, double log_T, double log_rho) nogil:
        r""":math:`\partial\ln\kappa/\partial\ln T` (dimensionless).

        For power-law :math:`\kappa \propto T^b` returns *b*.  For
        T-independent opacity returns 0.
        """
        with gil:
            raise NotImplementedError(
                f"{type(self).__name__} must implement _dlogkappa_dlogT(log_T, log_rho)."
            )

    # ================================================================ #
    #  Scalar Python wrappers — inherited; do not override             #
    # ================================================================ #
    # Accept Python floats (or anything Cython coerces to double),
    # call the cdef hot path, return a Python float.  These are the
    # entry points used by OpacityLaw in base.py.

    def opacity(self, double rho, double T):
        r""":math:`\kappa` (:math:`\mathrm{cm^2\,g^{-1}}`) — scalar Python wrapper for ``_opacity``."""
        return self._opacity(rho, T)

    def log_opacity(self, double log_T, double log_rho):
        r""":math:`\ln\kappa` — scalar Python wrapper for ``_log_opacity``."""
        return self._log_opacity(log_T, log_rho)

    def dlogkappa_dlogrho(self, double log_T, double log_rho):
        r""":math:`\partial\ln\kappa/\partial\ln\rho` — scalar Python wrapper for ``_dlogkappa_dlogrho``."""
        return self._dlogkappa_dlogrho(log_T, log_rho)

    def dlogkappa_dlogT(self, double log_T, double log_rho):
        r""":math:`\partial\ln\kappa/\partial\ln T` — scalar Python wrapper for ``_dlogkappa_dlogT``."""
        return self._dlogkappa_dlogT(log_T, log_rho)

    # ================================================================ #
    #  Vectorized Python wrappers — inherited; do not override         #
    # ================================================================ #
    # Accept C-contiguous 1-D double arrays (pass a numpy array with
    # dtype=float64, order='C').  Each method iterates with the GIL held
    # but calls the cdef hot path per element — no Python object creation
    # inside the loop.  Returns a new numpy ndarray.
    #
    # Usage:
    #   log_T   = np.log(T_array)
    #   log_rho = np.log(rho_array)
    #   log_kap = opacity_obj.log_opacity_array(log_T, log_rho)

    def opacity_array(self, double[::1] rho, double[::1] T):
        r""":math:`\kappa` (:math:`\mathrm{cm^2\,g^{-1}}`) evaluated over 1-D arrays *rho* and *T*.

        Parameters
        ----------
        rho : ndarray, dtype=float64, C-contiguous
            Density values in :math:`\mathrm{g\,cm^{-3}}`.
        T : ndarray, dtype=float64, C-contiguous
            Temperature values in K.

        Returns
        -------
        ndarray of float64
            Opacity values in :math:`\mathrm{cm^2\,g^{-1}}`, same shape as inputs.
        """
        cdef int n = rho.shape[0]
        cdef double[::1] out = np.empty(n, dtype=np.float64)
        for i in range(n):
            out[i] = self._opacity(rho[i], T[i])
        return np.asarray(out)

    def log_opacity_array(self, double[::1] log_T, double[::1] log_rho):
        r""":math:`\ln\kappa` evaluated over 1-D arrays *log_T* and *log_rho*.

        Parameters
        ----------
        log_T : ndarray, dtype=float64, C-contiguous
            Natural log of temperature values.
        log_rho : ndarray, dtype=float64, C-contiguous
            Natural log of density values.

        Returns
        -------
        ndarray of float64
            :math:`\ln\kappa` values (:math:`\mathrm{log\,cm^2\,g^{-1}}`).
        """
        cdef int n = log_T.shape[0]
        cdef double[::1] out = np.empty(n, dtype=np.float64)
        for i in range(n):
            out[i] = self._log_opacity(log_T[i], log_rho[i])
        return np.asarray(out)

    def dlogkappa_dlogrho_array(self, double[::1] log_T, double[::1] log_rho):
        r""":math:`\partial\ln\kappa/\partial\ln\rho` evaluated over 1-D arrays.

        Parameters
        ----------
        log_T, log_rho : ndarray, dtype=float64, C-contiguous

        Returns
        -------
        ndarray of float64
        """
        cdef int n = log_T.shape[0]
        cdef double[::1] out = np.empty(n, dtype=np.float64)
        for i in range(n):
            out[i] = self._dlogkappa_dlogrho(log_T[i], log_rho[i])
        return np.asarray(out)

    def dlogkappa_dlogT_array(self, double[::1] log_T, double[::1] log_rho):
        r""":math:`\partial\ln\kappa/\partial\ln T` evaluated over 1-D arrays.

        Parameters
        ----------
        log_T, log_rho : ndarray, dtype=float64, C-contiguous

        Returns
        -------
        ndarray of float64
        """
        cdef int n = log_T.shape[0]
        cdef double[::1] out = np.empty(n, dtype=np.float64)
        for i in range(n):
            out[i] = self._dlogkappa_dlogT(log_T[i], log_rho[i])
        return np.asarray(out)
