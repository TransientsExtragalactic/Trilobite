r"""
Base classes for grey (frequency-averaged) opacity laws.

This module defines the interface for opacity models in which the
frequency dependence has been integrated out, yielding an effective
opacity :math:`\kappa(\rho, T)`.

.. rubric:: Module Design

:class:`GreyOpacityLaw` subclasses :class:`~trilobite.radiation.opacity.base.OpacityLaw`
for the common case where opacity has no frequency dependence.  Private methods
take ``(log_T, log_rho)``; public methods take ``(rho, T)`` — no ``nu`` anywhere.

:class:`ConstantGreyOpacity` is a concrete grey opacity with a fixed value
independent of temperature and density.  It accepts either a plain float
(interpreted as :math:`\mathrm{cm^2\,g^{-1}}`) or an
:class:`~astropy.units.Quantity` with compatible units.
"""

from typing import Optional, Union

import numpy as np
from astropy import units as u

from trilobite.radiation.opacity.base import OpacityLaw
from trilobite.utils.misc_utils import ensure_in_units


class GreyOpacityLaw(OpacityLaw):
    r"""Abstract base class for grey (frequency-averaged) opacity laws.

    Specialises :class:`~trilobite.radiation.opacity.base.OpacityLaw` for
    models where the frequency dependence has been integrated out.  Private
    methods use the 2-argument signature ``(log_T, log_rho)``; public methods
    use ``(rho, T)``.  Neither accepts a frequency argument.

    Subclasses must override :meth:`_log_opacity`.  The derivative methods
    default to zero (appropriate for constant opacities); C-backed subclasses
    override them via the Cython companion object.

    See Also
    --------
    ~trilobite.radiation.opacity.grey_opacity.base.ConstantGreyOpacity : Concrete constant-opacity implementation.
    trilobite.radiation.opacity.opacity_base.C_GreyOpacityBase : C-level companion.
    """

    # ================================================================ #
    # Low-level log-space interface — (log_T, log_rho)                 #
    # ================================================================ #

    def _log_opacity(
        self,
        log_T: Union[float, np.ndarray],
        log_rho: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        r"""Return :math:`\ln\kappa` (:math:`\mathrm{cm^2\,g^{-1}}`).

        Parameters
        ----------
        log_T : float or ndarray
            :math:`\ln T` [K].
        log_rho : float or ndarray
            :math:`\ln\rho` [:math:`\mathrm{g\,cm^{-3}}`].

        Returns
        -------
        float or ndarray
            :math:`\ln\kappa` [:math:`\mathrm{cm^2\,g^{-1}}`].
        """
        if self.IS_C_BACKED:
            if np.ndim(log_T) > 0:
                return self._c_object.log_opacity_array(
                    np.ascontiguousarray(log_T, dtype=np.float64),
                    np.ascontiguousarray(log_rho, dtype=np.float64),
                )
            return self._c_object.log_opacity(log_T, log_rho)
        raise NotImplementedError(f"{type(self).__name__} must implement _log_opacity().")

    def _dlogkappa_dlogrho(
        self,
        log_T: Union[float, np.ndarray],
        log_rho: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        r"""Return :math:`\partial\ln\kappa/\partial\ln\rho` (dimensionless)."""
        if self.IS_C_BACKED:
            if np.ndim(log_T) > 0:
                return self._c_object.dlogkappa_dlogrho_array(
                    np.ascontiguousarray(log_T, dtype=np.float64),
                    np.ascontiguousarray(log_rho, dtype=np.float64),
                )
            return self._c_object.dlogkappa_dlogrho(log_T, log_rho)
        return 0.0

    def _dlogkappa_dlogT(
        self,
        log_T: Union[float, np.ndarray],
        log_rho: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        r"""Return :math:`\partial\ln\kappa/\partial\ln T` (dimensionless)."""
        if self.IS_C_BACKED:
            if np.ndim(log_T) > 0:
                return self._c_object.dlogkappa_dlogT_array(
                    np.ascontiguousarray(log_T, dtype=np.float64),
                    np.ascontiguousarray(log_rho, dtype=np.float64),
                )
            return self._c_object.dlogkappa_dlogT(log_T, log_rho)
        return 0.0

    # ================================================================ #
    # Public unit-bearing interface — (rho, T), no nu                  #
    # ================================================================ #

    def opacity(self, rho: u.Quantity, T: u.Quantity, nu=None) -> u.Quantity:
        r"""Return opacity :math:`\kappa(\rho, T)` (:math:`\mathrm{cm^2\,g^{-1}}`).

        Parameters
        ----------
        rho : `~astropy.units.Quantity`
            Mass density, convertible to :math:`\mathrm{g\,cm^{-3}}`.
        T : `~astropy.units.Quantity`
            Temperature, convertible to K.
        nu : ignored
            Accepted for interface compatibility; grey opacity has no frequency
            dependence.

        Returns
        -------
        `~astropy.units.Quantity`
            Opacity in :math:`\mathrm{cm^2\,g^{-1}}`.
        """
        log_T = np.log(ensure_in_units(T, u.K))
        log_rho = np.log(ensure_in_units(rho, u.g / u.cm**3))
        return np.exp(self._log_opacity(log_T, log_rho)) * u.cm**2 / u.g

    def dlogkappa_dlogrho(self, rho: u.Quantity, T: u.Quantity, nu=None) -> Union[float, np.ndarray]:
        r"""Return :math:`\partial\ln\kappa/\partial\ln\rho` (dimensionless).

        Parameters
        ----------
        rho : `~astropy.units.Quantity`
            Mass density.
        T : `~astropy.units.Quantity`
            Temperature.
        nu : ignored
        """
        log_T = np.log(ensure_in_units(T, u.K))
        log_rho = np.log(ensure_in_units(rho, u.g / u.cm**3))
        return self._dlogkappa_dlogrho(log_T, log_rho)

    def dlogkappa_dlogT(self, rho: u.Quantity, T: u.Quantity, nu=None) -> Union[float, np.ndarray]:
        r"""Return :math:`\partial\ln\kappa/\partial\ln T` (dimensionless).

        Parameters
        ----------
        rho : `~astropy.units.Quantity`
            Mass density.
        T : `~astropy.units.Quantity`
            Temperature.
        nu : ignored
        """
        log_T = np.log(ensure_in_units(T, u.K))
        log_rho = np.log(ensure_in_units(rho, u.g / u.cm**3))
        return self._dlogkappa_dlogT(log_T, log_rho)

    def dlogkappa_dlognu(self, rho: u.Quantity, T: u.Quantity, nu) -> float:
        r"""Return :math:`\partial\ln\kappa/\partial\ln\nu` = 0 (grey law)."""
        return 0.0


class ConstantGreyOpacity(GreyOpacityLaw):
    r"""Grey opacity law with a fixed, composition- and state-independent value.

    Parameters
    ----------
    kappa : float or `~astropy.units.Quantity`
        Opacity value.  A plain float is interpreted as
        :math:`\mathrm{cm^2\,g^{-1}}`.  An
        :class:`~astropy.units.Quantity` is converted to
        :math:`\mathrm{cm^2\,g^{-1}}` on construction.
    mean_type : str or None, optional
        Averaging prescription (e.g. ``"rosseland"`` or ``"planck"``).
        ``None`` (default) leaves the type unspecified, which is appropriate
        when the opacity is used in a context that doesn't require a specific
        mean type.

    Examples
    --------
    .. code-block:: python

        import astropy.units as u
        from trilobite.radiation.opacity.grey_opacity import (
            ConstantGreyOpacity,
        )

        op = ConstantGreyOpacity(0.34)  # type unspecified
        op = ConstantGreyOpacity(
            0.34, mean_type="rosseland"
        )  # tagged as Rosseland mean
        op = ConstantGreyOpacity(
            0.034 * u.m**2 / u.kg
        )  # Quantity input
        op.opacity(1e-5 * u.g / u.cm**3, 1e7 * u.K)
    """

    def __init__(
        self,
        kappa: Union[float, u.Quantity],
        *,
        mean_type: Optional[str] = None,
    ) -> None:
        if isinstance(kappa, u.Quantity):
            kappa_cgs = ensure_in_units(kappa, u.cm**2 / u.g)
        else:
            kappa_cgs = float(kappa)
        if kappa_cgs <= 0.0:
            raise ValueError(f"kappa must be positive, got {kappa_cgs}.")
        self._log_kappa: float = float(np.log(kappa_cgs))
        self.mean_type = mean_type
        super().__init__()

    @property
    def kappa(self) -> u.Quantity:
        r"""Opacity value as a :class:`~astropy.units.Quantity` in :math:`\mathrm{cm^2\,g^{-1}}`."""
        return float(np.exp(self._log_kappa)) * u.cm**2 / u.g

    def _log_opacity(self, log_T, log_rho):
        return self._log_kappa

    def _dlogkappa_dlogrho(self, log_T, log_rho):
        return 0.0

    def _dlogkappa_dlogT(self, log_T, log_rho):
        return 0.0
