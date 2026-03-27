"""Abstract base classes for Triceratops opacity laws."""

import numpy as np
from astropy import units as u

from triceratops.utils.misc_utils import ensure_in_units


class OpacityLaw:
    r"""Abstract base class for all Triceratops opacity laws.

    Subclasses override the three private log-space methods (``_log_opacity``,
    ``_dlogkappa_dlogrho``, ``_dlogkappa_dlogT``).  The public unit-bearing
    wrappers (``opacity``, ``dlogkappa_dlogrho``, ``dlogkappa_dlogT``) are
    provided here and must not be overridden.

    C-backed subclasses set ``IS_C_BACKED = True`` and override
    :meth:`_initialize_C_object` — the private methods then delegate to the
    Cython companion object automatically.
    """

    #: Set to ``True`` in subclasses that pair with a Cython extension type.
    IS_C_BACKED: bool = False

    def __init__(self, **parameters):
        # Build the C-level companion object when the subclass declares one.
        # Pure-Python subclasses leave _c_object as None and implement the
        # _log_opacity / derivative methods directly.
        self._c_object = self._initialize_C_object(**parameters) if self.IS_C_BACKED else None

    def _initialize_C_object(self, **parameters):
        """Return a :class:`~.opacity_base.C_GreyOpacityBase` instance.  Override in C-backed subclasses."""
        raise NotImplementedError(
            f"{type(self).__name__} sets IS_C_BACKED=True but does not override _initialize_C_object()."
        )

    # ================================================================ #
    # Low-level log-space interface (private)                          #
    # ================================================================ #
    # These methods receive and return plain Python floats or numpy
    # arrays in ln-CGS — no unit objects.  Subclasses that are NOT
    # C-backed override these directly.  C-backed subclasses inherit
    # the default implementations below, which forward to _c_object.

    def _log_opacity(self, log_T, log_rho):
        r"""Return :math:`\ln\kappa` (:math:`\mathrm{log\,cm^2\,g^{-1}}`).

        Convention: **(log_T, log_rho)** — temperature first.

        Parameters
        ----------
        log_T : float or ndarray
            :math:`\ln T` in K.
        log_rho : float or ndarray
            :math:`\ln\rho` in :math:`\mathrm{g\,cm^{-3}}`.
        """
        if self.IS_C_BACKED:
            return self._c_object.log_opacity(log_T, log_rho)
        raise NotImplementedError(f"{type(self).__name__} must implement _log_opacity().")

    def _dlogkappa_dlogrho(self, log_T, log_rho):
        r"""Return :math:`\partial\ln\kappa/\partial\ln\rho` (dimensionless)."""
        if self.IS_C_BACKED:
            return self._c_object.dlogkappa_dlogrho(log_T, log_rho)
        raise NotImplementedError(f"{type(self).__name__} must implement _dlogkappa_dlogrho().")

    def _dlogkappa_dlogT(self, log_T, log_rho):
        r"""Return :math:`\partial\ln\kappa/\partial\ln T` (dimensionless)."""
        if self.IS_C_BACKED:
            return self._c_object.dlogkappa_dlogT(log_T, log_rho)
        raise NotImplementedError(f"{type(self).__name__} must implement _dlogkappa_dlogT().")

    # ================================================================ #
    # High-level unit-bearing interface (public)                       #
    # ================================================================ #
    # These methods accept and return astropy Quantities.  They convert
    # to CGS, call the low-level log-space methods, and wrap the result.
    # Do NOT override these in subclasses — override the _log_* methods.

    def opacity(self, rho, T) -> u.Quantity:
        r"""Return opacity :math:`\kappa` (:math:`\mathrm{cm^2\,g^{-1}}`).

        Parameters
        ----------
        rho : `~astropy.units.Quantity`
            Mass density.
        T : `~astropy.units.Quantity`
            Temperature.

        Returns
        -------
        `~astropy.units.Quantity`
            Opacity in :math:`\mathrm{cm^2\,g^{-1}}`.
        """
        rho_cgs = ensure_in_units(rho, u.g / u.cm**3)
        T_cgs = ensure_in_units(T, u.K)
        log_kap = self._log_opacity(np.log(T_cgs), np.log(rho_cgs))
        return np.exp(log_kap) * u.cm**2 / u.g

    def dlogkappa_dlogrho(self, rho, T):
        r"""Return :math:`\partial\ln\kappa/\partial\ln\rho` (dimensionless).

        Parameters
        ----------
        rho : `~astropy.units.Quantity`
            Mass density.
        T : `~astropy.units.Quantity`
            Temperature.
        """
        rho_cgs = ensure_in_units(rho, u.g / u.cm**3)
        T_cgs = ensure_in_units(T, u.K)
        return self._dlogkappa_dlogrho(np.log(T_cgs), np.log(rho_cgs))

    def dlogkappa_dlogT(self, rho, T):
        r"""Return :math:`\partial\ln\kappa/\partial\ln T` (dimensionless).

        Parameters
        ----------
        rho : `~astropy.units.Quantity`
            Mass density.
        T : `~astropy.units.Quantity`
            Temperature.
        """
        rho_cgs = ensure_in_units(rho, u.g / u.cm**3)
        T_cgs = ensure_in_units(T, u.K)
        return self._dlogkappa_dlogT(np.log(T_cgs), np.log(rho_cgs))


class GreyOpacityLaw(OpacityLaw):
    r"""Concrete base for constant (frequency- and density-independent) opacities.

    Subclasses set the class attribute ``_LOG_KAPPA`` (:math:`\ln\kappa` in
    :math:`\mathrm{log\,cm^2\,g^{-1}}`); all partial derivatives are
    identically zero.  C-backed subclasses leave ``_LOG_KAPPA`` unset and
    instead pair with a Cython extension object via ``IS_C_BACKED = True``.
    """

    #: :math:`\ln\kappa` in :math:`\mathrm{log\,cm^2\,g^{-1}}`.  Override with ``np.log(your_kappa_value)``.
    _LOG_KAPPA: float = 0.0  # log(1.0) — subclasses should override this

    def _log_opacity(self, log_T, log_rho):
        # Grey: kappa is constant — return the precomputed log value.
        # If IS_C_BACKED is True the parent's _log_opacity would delegate
        # to _c_object instead, so this branch is only reached for pure-Python
        # grey laws.
        if self.IS_C_BACKED:
            return self._c_object.log_opacity(log_T, log_rho)
        return self._LOG_KAPPA

    def _dlogkappa_dlogrho(self, log_T, log_rho):
        # Grey opacity has no density dependence.
        if self.IS_C_BACKED:
            return self._c_object.dlogkappa_dlogrho(log_T, log_rho)
        return 0.0

    def _dlogkappa_dlogT(self, log_T, log_rho):
        # Grey opacity has no temperature dependence.
        if self.IS_C_BACKED:
            return self._c_object.dlogkappa_dlogT(log_T, log_rho)
        return 0.0
