r"""
Equation of state (EOS) utilities for Triceratops models.

This module provides two APIs for computing thermodynamic quantities
(pressure, sound speed) under common astrophysical equations of state.

Module-level functions
----------------------
A lightweight functional API that operates without instantiating a class.
Private ``_log_*`` functions take bare log-CGS floats and return ``ln(quantity)``;
public wrappers accept :class:`~astropy.units.Quantity` and return
unit-bearing outputs.

+------------------------------------------+--------------------------------------------+
| Private (log-CGS)                        | Public (:class:`~astropy.units.Quantity`)  |
+==========================================+============================================+
| :func:`_log_ideal_gas_disk_sound_speed`  | :func:`ideal_gas_sound_speed`              |
+------------------------------------------+--------------------------------------------+
| :func:`_log_radiative_ideal_gas_         | :func:`radiative_ideal_gas_sound_speed`    |
| sound_speed`                             |                                            |
+------------------------------------------+--------------------------------------------+
| :func:`_log_radiative_ideal_gas_disk_    | :func:`radiative_ideal_gas_disk_           |
| sound_speed`                             | sound_speed`                               |
+------------------------------------------+--------------------------------------------+

Class hierarchy
---------------
For use in model objects that need to store EOS parameters:

.. code-block:: text

    EquationOfState           (abstract)
    ├── IdealGasEOS           P_gas = ρ k_B T / (μ m_p)
    └── RadiativeIdealGas     P_tot = P_gas + a_rad T⁴ / 3

Each concrete class exposes ``_compute_log_*`` methods (log-CGS, no unit
overhead) and ``compute_*`` wrappers (:class:`~astropy.units.Quantity`).
Both delegate to the module-level private functions above.

Notes
-----
*Isothermal vs. adiabatic*: disk vertical-structure calculations use the
**isothermal** sound speed :math:`c_s = \sqrt{k_B T / (\mu m_p)}`, which
matches the Cython EOS primitives in
:mod:`~triceratops.dynamics.accretion.one_zone.physics._eos`.  The
:meth:`~RadiativeIdealGas.compute_sound_speed` method uses the
**effective adiabatic** sound speed derived from :math:`\gamma_{\rm eff}`,
which is appropriate for general thermodynamic calculations where the
density is known directly.

See Also
--------
:mod:`.composition` : Mean molecular weight helpers used to set ``mu``.
:mod:`.constants` : CGS constants consumed internally.
:mod:`~triceratops.dynamics.accretion.one_zone.physics._eos` : Cython EOS
    primitives used in the hot-loop integrator.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u

from ..utils.misc_utils import ensure_in_units
from .constants import _log_a_rad_cgs, _log_k_B_cgs, _log_m_p_cgs

if TYPE_CHECKING:
    from .._typing import _ArrayLike, _UnitBearingArrayLike


# ================================================================= #
# Private log-space helpers                                         #
# ================================================================= #


def _log_ideal_gas_disk_sound_speed(
    log_T: "_ArrayLike",
    mu: float,
) -> "_ArrayLike":
    r"""
    Compute :math:`\ln c_s` for an isothermal ideal gas (disk use).

    The isothermal sound speed (no adiabatic :math:`\gamma` factor) is
    used for disk vertical-structure calculations:

    .. math::

        c_s = \sqrt{\frac{k_B\,T}{\mu\,m_p}}
        \implies
        \ln c_s = \tfrac{1}{2}\!\left(\ln k_B + \ln T - \ln\mu - \ln m_p\right)

    This mirrors the Cython primitive ``compute_ideal_gas_cs`` in
    :mod:`~triceratops.dynamics.accretion.one_zone.physics._eos`.

    Parameters
    ----------
    log_T : float or ndarray
        :math:`\ln T` [:math:`\ln(\text{K})`].
    mu : float
        Mean molecular weight (dimensionless).

    Returns
    -------
    float or ndarray
        :math:`\ln c_s` [:math:`\ln(\text{cm s}^{-1})`].
    """
    log_T = np.asarray(log_T)
    result = 0.5 * (_log_k_B_cgs + log_T - np.log(mu) - _log_m_p_cgs)
    return result if result.ndim > 0 else result.item()


def _log_radiative_ideal_gas_sound_speed(
    log_rho: "_ArrayLike",
    log_T: "_ArrayLike",
    mu: float,
) -> "_ArrayLike":
    r"""
    Compute :math:`\ln c_s` for a gas + radiation EOS given density directly.

    Uses the effective adiabatic index

    .. math::

        \gamma_{\rm eff} = \frac{P_{\rm gas} + 4\,P_{\rm rad}}
                                {P_{\rm gas} + 3\,P_{\rm rad}}

    so that

    .. math::

        c_s = \sqrt{\frac{\gamma_{\rm eff}\,P_{\rm tot}}{\rho}}
        \implies
        \ln c_s = \tfrac{1}{2}\!\left(\ln P_{\rm tot} - \ln\rho
                  + \ln\gamma_{\rm eff}\right)

    with :math:`P_{\rm tot}` computed via :func:`numpy.logaddexp` for
    numerical stability.

    Parameters
    ----------
    log_rho : float or ndarray
        :math:`\ln\rho` [:math:`\ln(\text{g cm}^{-3})`].
    log_T : float or ndarray
        :math:`\ln T` [:math:`\ln(\text{K})`].
    mu : float
        Mean molecular weight (dimensionless).

    Returns
    -------
    float or ndarray
        :math:`\ln c_s` [:math:`\ln(\text{cm s}^{-1})`].
    """
    log_T, log_rho = np.asarray(log_T), np.asarray(log_rho)

    log_P_gas = _log_k_B_cgs + log_T + log_rho - np.log(mu) - _log_m_p_cgs
    log_P_rad = _log_a_rad_cgs - np.log(3.0) + 4.0 * log_T
    log_P_tot = np.logaddexp(log_P_gas, log_P_rad)

    P_gas = np.exp(log_P_gas)
    P_rad = np.exp(log_P_rad)
    gamma_eff = (P_gas + 4.0 * P_rad) / (P_gas + 3.0 * P_rad)

    result = 0.5 * (log_P_tot - log_rho + np.log(gamma_eff))
    return result if result.ndim > 0 else result.item()


def _log_radiative_ideal_gas_disk_sound_speed(
    log_T: "_ArrayLike",
    mu: float,
    log_Sigma: "_ArrayLike",
    log_Omega: "_ArrayLike",
) -> "_ArrayLike":
    r"""
    Compute :math:`\ln c_s` for a gas + radiation EOS in disk geometry.

    Because the disk scale height :math:`H = c_s / \Omega` and the midplane
    density :math:`\rho = \Sigma\,\Omega / (\sqrt{2\pi}\,c_s)`, the combined
    EOS

    .. math::

        c_s^2 = \frac{k_B\,T}{\mu\,m_p} + \frac{a_{\rm rad}\,T^4}{3\,\rho}

    reduces to a quadratic in :math:`c_s`:

    .. math::

        c_s^2 - A\,c_s - B = 0,
        \quad
        A = \frac{a_{\rm rad}\,T^4\,\sqrt{2\pi}}{3\,\Sigma\,\Omega},
        \quad
        B = \frac{k_B\,T}{\mu\,m_p}.

    The positive physical root :math:`c_s = (A + \sqrt{A^2 + 4B}) / 2`
    is returned.  With :math:`A \geq 0` and :math:`B > 0` there is no
    catastrophic cancellation.

    Parameters
    ----------
    log_T : float or ndarray
        :math:`\ln T` [:math:`\ln(\text{K})`].
    mu : float
        Mean molecular weight (dimensionless).
    log_Sigma : float or ndarray
        :math:`\ln\Sigma` [:math:`\ln(\text{g cm}^{-2})`].
    log_Omega : float or ndarray
        :math:`\ln\Omega` [:math:`\ln(\text{s}^{-1})`].

    Returns
    -------
    float or ndarray
        :math:`\ln c_s` [:math:`\ln(\text{cm s}^{-1})`].
    """
    log_T = np.asarray(log_T)
    log_Sigma = np.asarray(log_Sigma)
    log_Omega = np.asarray(log_Omega)

    log_A = _log_a_rad_cgs + 4.0 * log_T + 0.5 * np.log(2.0 * np.pi) - np.log(3.0) - log_Sigma - log_Omega
    log_B = _log_k_B_cgs + log_T - np.log(mu) - _log_m_p_cgs

    A = np.exp(log_A)
    B = np.exp(log_B)
    cs = 0.5 * (A + np.sqrt(A * A + 4.0 * B))

    result = np.log(cs)
    return result if result.ndim > 0 else result.item()


# ================================================================= #
# Public module-level functions                                     #
# ================================================================= #


def ideal_gas_sound_speed(T: u.Quantity, mu: float) -> u.Quantity:
    r"""
    Isothermal ideal-gas sound speed.

    .. math::

        c_s = \sqrt{\frac{k_B\,T}{\mu\,m_p}}

    Parameters
    ----------
    T : `~astropy.units.Quantity`
        Temperature.
    mu : float
        Mean molecular weight (dimensionless).

    Returns
    -------
    `~astropy.units.Quantity`
        Isothermal sound speed [:math:`\text{cm s}^{-1}`].

    Examples
    --------
    .. code-block:: python

        import astropy.units as u
        from triceratops.physics_utils.eos import (
            ideal_gas_sound_speed,
        )

        ideal_gas_sound_speed(1e7 * u.K, mu=0.62)
    """
    log_T = np.log(ensure_in_units(T, "K"))
    return np.exp(_log_ideal_gas_disk_sound_speed(log_T, mu)) * u.cm / u.s


def radiative_ideal_gas_sound_speed(
    rho: u.Quantity,
    T: u.Quantity,
    mu: float,
) -> u.Quantity:
    r"""
    Effective adiabatic sound speed for a gas + radiation EOS.

    Uses the effective adiabatic index

    .. math::

        \gamma_{\rm eff} = \frac{P_{\rm gas} + 4\,P_{\rm rad}}
                                {P_{\rm gas} + 3\,P_{\rm rad}},
        \qquad
        c_s = \sqrt{\frac{\gamma_{\rm eff}\,P_{\rm tot}}{\rho}}.

    Parameters
    ----------
    rho : `~astropy.units.Quantity`
        Mass density :math:`\rho`.
    T : `~astropy.units.Quantity`
        Temperature.
    mu : float
        Mean molecular weight (dimensionless).

    Returns
    -------
    `~astropy.units.Quantity`
        Adiabatic sound speed [:math:`\text{cm s}^{-1}`].

    Examples
    --------
    .. code-block:: python

        import astropy.units as u
        from triceratops.physics_utils.eos import (
            radiative_ideal_gas_sound_speed,
        )

        radiative_ideal_gas_sound_speed(
            1e-3 * u.Unit("g/cm^3"), 1e7 * u.K, mu=0.62
        )
    """
    log_rho = np.log(ensure_in_units(rho, "g/cm^3"))
    log_T = np.log(ensure_in_units(T, "K"))
    return np.exp(_log_radiative_ideal_gas_sound_speed(log_rho, log_T, mu)) * u.cm / u.s


def radiative_ideal_gas_disk_sound_speed(
    T: u.Quantity,
    mu: float,
    Sigma: u.Quantity,
    Omega: u.Quantity,
) -> u.Quantity:
    r"""
    Isothermal sound speed for a gas + radiation EOS in disk geometry.

    Because the disk scale height :math:`H = c_s / \Omega` and the midplane
    density :math:`\rho = \Sigma\,\Omega / (\sqrt{2\pi}\,c_s)`, the combined
    EOS

    .. math::

        c_s^2 = \frac{k_B\,T}{\mu\,m_p} + \frac{a_{\rm rad}\,T^4}{3\,\rho}

    reduces to a quadratic in :math:`c_s`:

    .. math::

        c_s^2 - A\,c_s - B = 0,
        \qquad
        A = \frac{a_{\rm rad}\,T^4\,\sqrt{2\pi}}{3\,\Sigma\,\Omega},
        \quad
        B = \frac{k_B\,T}{\mu\,m_p}.

    The physically relevant positive root is returned.

    Parameters
    ----------
    T : `~astropy.units.Quantity`
        Midplane temperature.
    mu : float
        Mean molecular weight (dimensionless).
    Sigma : `~astropy.units.Quantity`
        Column / surface density.
    Omega : `~astropy.units.Quantity`
        Keplerian angular velocity.

    Returns
    -------
    `~astropy.units.Quantity`
        Isothermal sound speed [:math:`\text{cm s}^{-1}`].

    Examples
    --------
    .. code-block:: python

        import astropy.units as u
        from triceratops.physics_utils.eos import (
            radiative_ideal_gas_disk_sound_speed,
        )

        radiative_ideal_gas_disk_sound_speed(
            1e7 * u.K,
            0.62,
            1e3 * u.Unit("g/cm^2"),
            1e-3 / u.s,
        )
    """
    log_T = np.log(ensure_in_units(T, "K"))
    log_Sigma = np.log(ensure_in_units(Sigma, "g/cm^2"))
    log_Omega = np.log(ensure_in_units(Omega, "1/s"))
    return np.exp(_log_radiative_ideal_gas_disk_sound_speed(log_T, mu, log_Sigma, log_Omega)) * u.cm / u.s


# ================================================================== #
# Abstract Base                                                      #
# ================================================================== #


class EquationOfState(ABC):
    r"""
    Abstract base class for all equations of state.

    All concrete EOS classes inherit from this and must implement
    :meth:`_compute_log_pressure`, which operates on log-space CGS
    scalars / arrays.  The public :meth:`compute_pressure` wrapper
    handles unit conversion and is provided by the base class.

    Parameters
    ----------
    **params : dict
        EOS-specific parameters (e.g. mean molecular weight ``mu``).
        Stored as :attr:`params`.

    Notes
    -----
    The two-level API mirrors the rest of the Triceratops codebase:

    .. code-block:: text

        _compute_log_*()   ← operates on bare log-CGS floats, no unit overhead
        compute_*()        ← accepts astropy Quantities, returns Quantities

    Both levels delegate to the module-level private functions in this
    module (e.g. :func:`_log_ideal_gas_disk_sound_speed`), so the
    standalone functional API and the class API are always consistent.

    See Also
    --------
    :class:`IdealGasEOS`, :class:`RadiativeIdealGas`
    """

    def __init__(self, **params):
        """
        Instantiate the EOS with the given parameters.

        Parameters
        ----------
        **params : dict
            EOS-specific parameters (e.g. mean molecular weight ``mu``).
        """
        self.params = params

    # -------------------------------------------------------------- #
    # Abstract interface                                              #
    # -------------------------------------------------------------- #

    @abstractmethod
    def _compute_log_pressure(self, *args, **kwargs) -> "_ArrayLike":
        r"""
        Compute the natural log of the pressure in CGS units.

        Parameters
        ----------
        *args, **kwargs
            EOS-specific inputs (e.g. log density, log temperature).

        Returns
        -------
        float or ndarray
            :math:`\ln P` [:math:`\ln(\text{erg cm}^{-3})`].
        """
        ...

    # -------------------------------------------------------------- #
    # Public interface                                                #
    # -------------------------------------------------------------- #

    @abstractmethod
    def compute_pressure(self, *args, **kwargs) -> "_UnitBearingArrayLike":
        r"""
        Compute the pressure.

        Parameters
        ----------
        *args, **kwargs
            EOS-specific inputs as :class:`~astropy.units.Quantity` objects.

        Returns
        -------
        `~astropy.units.Quantity`
            Pressure [:math:`\text{erg cm}^{-3}`].
        """
        ...

    # -------------------------------------------------------------- #
    # Dunder methods                                                  #
    # -------------------------------------------------------------- #

    def __repr__(self) -> str:
        param_str = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({param_str})"


# ================================================================== #
# Ideal Gas                                                          #
# ================================================================== #


class IdealGasEOS(EquationOfState):
    r"""
    Ideal gas equation of state.

    The thermal pressure and **isothermal** sound speed are

    .. math::

        P_{\rm gas} = \frac{\rho\,k_B\,T}{\mu\,m_p},
        \qquad
        c_s = \sqrt{\frac{k_B\,T}{\mu\,m_p}}.

    The isothermal form (no :math:`\gamma` factor) is used throughout
    because the disk vertical-structure equations assume an isothermal
    atmosphere, consistent with the Cython EOS primitives in
    :mod:`~triceratops.dynamics.accretion.one_zone.physics._eos`.

    Parameters
    ----------
    mu : float, optional
        Mean molecular weight per particle (dimensionless).  For solar
        composition use :func:`~.composition.compute_mean_molecular_weight`
        or pass ``mu=0.62`` directly.  Default ``0.6``.

    Notes
    -----
    The mean molecular weight :math:`\mu` relates the mean particle mass
    to the proton mass:

    .. math::

        \mu = \frac{\bar{m}}{m_p} \equiv \frac{\rho}{n_{\rm tot}\,m_p}.

    For a fully ionised hydrogen-helium plasma with mass fractions
    :math:`X = 0.70,\, Y = 0.28`, :math:`\mu \approx 0.62`.

    See Also
    --------
    :func:`ideal_gas_sound_speed` : standalone functional equivalent.
    :func:`~.composition.compute_mean_molecular_weight` : derive ``mu`` from
        hydrogen/helium/metal mass fractions.

    Examples
    --------
    Compute the gas pressure at :math:`\rho = 10^{-3}` g cm⁻³,
    :math:`T = 10^7` K:

    .. code-block:: python

        import astropy.units as u

        eos = IdealGasEOS(mu=0.62)
        eos.compute_pressure(
            1e-3 * u.Unit("g/cm^3"), 1e7 * u.K
        )

    Compute the isothermal sound speed at :math:`T = 10^7` K:

    .. code-block:: python

        eos.compute_sound_speed(1e7 * u.K)
    """

    def __init__(self, mu: float = 0.6, **kwargs):
        r"""
        Initialize the ideal gas equation of state.

        Parameters
        ----------
        mu : float, optional
            Mean molecular weight of the gas (dimensionless).  Controls the
            conversion between temperature and thermal energy via the ideal
            gas law.  Default ``0.6`` is appropriate for a fully ionized
            plasma with primordial composition.
        **kwargs
            Additional keyword arguments passed to :class:`EquationOfState`.
        """
        super().__init__(mu=mu, **kwargs)
        self.mu = mu

    # -------------------------------------------------------------- #
    # Private (log-space CGS)                                        #
    # -------------------------------------------------------------- #

    def _compute_log_pressure(
        self,
        log_density: "_ArrayLike",
        log_temperature: "_ArrayLike",
    ) -> "_ArrayLike":
        r"""
        Compute :math:`\ln P_{\rm gas}` in CGS units.

        .. math::

            \ln P_{\rm gas} = \ln k_B + \ln T + \ln\rho - \ln\mu - \ln m_p

        Parameters
        ----------
        log_density : float or ndarray
            :math:`\ln\rho` [:math:`\ln(\text{g cm}^{-3})`].
        log_temperature : float or ndarray
            :math:`\ln T` [:math:`\ln(\text{K})`].

        Returns
        -------
        float or ndarray
            :math:`\ln P_{\rm gas}` [:math:`\ln(\text{erg cm}^{-3})`].
        """
        log_temperature = np.asarray(log_temperature)
        log_density = np.asarray(log_density)
        result = _log_k_B_cgs + log_temperature + log_density - np.log(self.mu) - _log_m_p_cgs
        return result if result.ndim > 0 else result.item()

    def _compute_log_sound_speed(self, log_temperature: "_ArrayLike") -> "_ArrayLike":
        r"""
        Compute :math:`\ln c_s` in CGS units.

        Delegates to :func:`_log_ideal_gas_disk_sound_speed`.

        .. math::

            \ln c_s = \tfrac{1}{2}\!\left(\ln k_B + \ln T - \ln\mu - \ln m_p\right)

        Parameters
        ----------
        log_temperature : float or ndarray
            :math:`\ln T` [:math:`\ln(\text{K})`].

        Returns
        -------
        float or ndarray
            :math:`\ln c_s` [:math:`\ln(\text{cm s}^{-1})`].
        """
        return _log_ideal_gas_disk_sound_speed(log_temperature, self.mu)

    # -------------------------------------------------------------- #
    # Public (unit-bearing)                                          #
    # -------------------------------------------------------------- #

    def compute_pressure(
        self,
        density: u.Quantity,
        temperature: u.Quantity,
    ) -> u.Quantity:
        r"""
        Compute the ideal gas pressure.

        Parameters
        ----------
        density : `~astropy.units.Quantity`
            Mass density :math:`\rho`.
        temperature : `~astropy.units.Quantity`
            Temperature :math:`T`.

        Returns
        -------
        `~astropy.units.Quantity`
            Gas pressure :math:`P_{\rm gas}` [:math:`\text{erg cm}^{-3}`].

        Examples
        --------
        .. code-block:: python

            import astropy.units as u

            IdealGasEOS(mu=0.62).compute_pressure(
                1e-3 * u.Unit("g/cm^3"), 1e7 * u.K
            )
        """
        log_density = np.log(ensure_in_units(density, "g/cm^3"))
        log_temperature = np.log(ensure_in_units(temperature, "K"))
        return np.exp(self._compute_log_pressure(log_density, log_temperature)) * u.erg / u.cm**3

    def compute_sound_speed(self, temperature: u.Quantity) -> u.Quantity:
        r"""
        Compute the isothermal sound speed.

        Parameters
        ----------
        temperature : `~astropy.units.Quantity`
            Temperature :math:`T`.

        Returns
        -------
        `~astropy.units.Quantity`
            Isothermal sound speed :math:`c_s` [:math:`\text{cm s}^{-1}`].

        Examples
        --------
        .. code-block:: python

            import astropy.units as u

            IdealGasEOS(mu=0.62).compute_sound_speed(
                1e7 * u.K
            )
        """
        log_temperature = np.log(ensure_in_units(temperature, "K"))
        return np.exp(self._compute_log_sound_speed(log_temperature)) * u.cm / u.s


# ================================================================== #
# Combined Ideal Gas + Radiation                                     #
# ================================================================== #


class RadiativeIdealGas(EquationOfState):
    r"""
    Combined ideal gas plus radiation equation of state.

    The total pressure is the sum of the gas and radiation contributions:

    .. math::

        P_{\rm tot} = P_{\rm gas} + P_{\rm rad}
        = \frac{\rho\,k_B\,T}{\mu\,m_p} + \frac{a_{\rm rad}}{3}\,T^4.

    Two sound-speed methods are provided:

    * :meth:`compute_sound_speed` — effective adiabatic sound speed
      :math:`c_s = \sqrt{\gamma_{\rm eff} P_{\rm tot}/\rho}`, requires
      the density directly.
    * :meth:`compute_disk_sound_speed` — isothermal sound speed for disk
      geometry, solving the quadratic that arises when
      :math:`\rho = \Sigma\,\Omega / (\sqrt{2\pi}\,c_s)`.

    Parameters
    ----------
    mu : float, optional
        Mean molecular weight per particle (dimensionless).  Default ``0.6``.

    Notes
    -----
    The radiation contribution dominates at high temperatures and low
    densities; the gas term dominates at high densities and moderate
    temperatures.  The crossover temperature at fixed density :math:`\rho` is

    .. math::

        T_{\rm eq} = \left(\frac{3\,\rho\,k_B}{a_{\rm rad}\,\mu\,m_p}\right)^{1/3}.

    The effective adiabatic index

    .. math::

        \gamma_{\rm eff} = \frac{P_{\rm gas} + 4\,P_{\rm rad}}
                                {P_{\rm gas} + 3\,P_{\rm rad}}

    interpolates between :math:`5/3` (gas-dominated) and :math:`4/3`
    (radiation-dominated).

    See Also
    --------
    :class:`IdealGasEOS` : gas-only limit.
    :func:`radiative_ideal_gas_sound_speed` : standalone functional equivalent
        for the adiabatic sound speed.
    :func:`radiative_ideal_gas_disk_sound_speed` : standalone functional
        equivalent for the disk sound speed.
    :func:`~.composition.compute_mean_molecular_weight` : derive ``mu`` from
        composition.

    Examples
    --------
    Compute the total pressure at :math:`\rho = 1` g cm⁻³, :math:`T = 10^7` K:

    .. code-block:: python

        import astropy.units as u

        eos = RadiativeIdealGas(mu=0.62)
        eos.compute_pressure(
            1.0 * u.Unit("g/cm^3"), 1e7 * u.K
        )

    Compute the disk isothermal sound speed:

    .. code-block:: python

        eos.compute_disk_sound_speed(
            1e7 * u.K,
            Sigma=1e3 * u.Unit("g/cm^2"),
            Omega=1e-3 / u.s,
        )
    """

    def __init__(self, mu: float = 0.6, **kwargs):
        r"""
        Initialize the combined ideal gas + radiation equation of state.

        Parameters
        ----------
        mu : float, optional
            Mean molecular weight per particle (dimensionless).  Controls the
            gas pressure contribution.  Default ``0.6``.
        **kwargs
            Additional keyword arguments passed to :class:`EquationOfState`.
        """
        super().__init__(mu=mu, **kwargs)
        self.mu = mu

    # -------------------------------------------------------------- #
    # Private (log-space CGS)                                        #
    # -------------------------------------------------------------- #

    def _compute_log_pressure(
        self,
        log_density: "_ArrayLike",
        log_temperature: "_ArrayLike",
    ) -> "_ArrayLike":
        r"""
        Compute :math:`\ln P_{\rm tot}` using :func:`numpy.logaddexp`.

        .. math::

            \ln P_{\rm tot} = \ln\!\left(e^{\ln P_{\rm gas}}
                              + e^{\ln P_{\rm rad}}\right)

        Parameters
        ----------
        log_density : float or ndarray
            :math:`\ln\rho` [:math:`\ln(\text{g cm}^{-3})`].
        log_temperature : float or ndarray
            :math:`\ln T` [:math:`\ln(\text{K})`].

        Returns
        -------
        float or ndarray
            :math:`\ln P_{\rm tot}` [:math:`\ln(\text{erg cm}^{-3})`].
        """
        log_temperature = np.asarray(log_temperature)
        log_density = np.asarray(log_density)

        log_P_gas = _log_k_B_cgs + log_temperature + log_density - np.log(self.mu) - _log_m_p_cgs
        log_P_rad = _log_a_rad_cgs - np.log(3.0) + 4.0 * log_temperature

        result = np.logaddexp(log_P_gas, log_P_rad)
        return result if result.ndim > 0 else result.item()

    def _compute_log_sound_speed(
        self,
        log_density: "_ArrayLike",
        log_temperature: "_ArrayLike",
    ) -> "_ArrayLike":
        r"""
        Compute :math:`\ln c_s` for the effective adiabatic sound speed.

        Delegates to :func:`_log_radiative_ideal_gas_sound_speed`.

        Parameters
        ----------
        log_density : float or ndarray
            :math:`\ln\rho` [:math:`\ln(\text{g cm}^{-3})`].
        log_temperature : float or ndarray
            :math:`\ln T` [:math:`\ln(\text{K})`].

        Returns
        -------
        float or ndarray
            :math:`\ln c_s` [:math:`\ln(\text{cm s}^{-1})`].
        """
        return _log_radiative_ideal_gas_sound_speed(log_density, log_temperature, self.mu)

    def _compute_log_disk_sound_speed(
        self,
        log_temperature: "_ArrayLike",
        log_Sigma: "_ArrayLike",
        log_Omega: "_ArrayLike",
    ) -> "_ArrayLike":
        r"""
        Compute :math:`\ln c_s` for the disk isothermal sound speed.

        Delegates to :func:`_log_radiative_ideal_gas_disk_sound_speed`.

        Parameters
        ----------
        log_temperature : float or ndarray
            :math:`\ln T` [:math:`\ln(\text{K})`].
        log_Sigma : float or ndarray
            :math:`\ln\Sigma` [:math:`\ln(\text{g cm}^{-2})`].
        log_Omega : float or ndarray
            :math:`\ln\Omega` [:math:`\ln(\text{s}^{-1})`].

        Returns
        -------
        float or ndarray
            :math:`\ln c_s` [:math:`\ln(\text{cm s}^{-1})`].
        """
        return _log_radiative_ideal_gas_disk_sound_speed(log_temperature, self.mu, log_Sigma, log_Omega)

    # -------------------------------------------------------------- #
    # Public (unit-bearing)                                          #
    # -------------------------------------------------------------- #

    def compute_pressure(
        self,
        density: u.Quantity,
        temperature: u.Quantity,
    ) -> u.Quantity:
        r"""
        Compute the total (gas + radiation) pressure.

        Parameters
        ----------
        density : `~astropy.units.Quantity`
            Mass density :math:`\rho`.
        temperature : `~astropy.units.Quantity`
            Temperature :math:`T`.

        Returns
        -------
        `~astropy.units.Quantity`
            Total pressure :math:`P_{\rm tot}` [:math:`\text{erg cm}^{-3}`].

        Examples
        --------
        .. code-block:: python

            import astropy.units as u

            RadiativeIdealGas(mu=0.62).compute_pressure(
                1.0 * u.Unit("g/cm^3"), 1e7 * u.K
            )
        """
        log_density = np.log(ensure_in_units(density, "g/cm^3"))
        log_temperature = np.log(ensure_in_units(temperature, "K"))
        return np.exp(self._compute_log_pressure(log_density, log_temperature)) * u.erg / u.cm**3

    def compute_sound_speed(
        self,
        density: u.Quantity,
        temperature: u.Quantity,
    ) -> u.Quantity:
        r"""
        Compute the effective adiabatic sound speed.

        .. math::

            c_s = \sqrt{\frac{\gamma_{\rm eff}\,P_{\rm tot}}{\rho}},
            \qquad
            \gamma_{\rm eff} = \frac{P_{\rm gas} + 4\,P_{\rm rad}}
                                    {P_{\rm gas} + 3\,P_{\rm rad}}.

        Parameters
        ----------
        density : `~astropy.units.Quantity`
            Mass density :math:`\rho`.
        temperature : `~astropy.units.Quantity`
            Temperature :math:`T`.

        Returns
        -------
        `~astropy.units.Quantity`
            Effective adiabatic sound speed [:math:`\text{cm s}^{-1}`].

        Examples
        --------
        .. code-block:: python

            import astropy.units as u

            RadiativeIdealGas(mu=0.62).compute_sound_speed(
                1.0 * u.Unit("g/cm^3"), 1e7 * u.K
            )
        """
        log_density = np.log(ensure_in_units(density, "g/cm^3"))
        log_temperature = np.log(ensure_in_units(temperature, "K"))
        return np.exp(self._compute_log_sound_speed(log_density, log_temperature)) * u.cm / u.s

    def compute_disk_sound_speed(
        self,
        temperature: u.Quantity,
        Sigma: u.Quantity,
        Omega: u.Quantity,
    ) -> u.Quantity:
        r"""
        Compute the isothermal sound speed in disk geometry.

        Solves the quadratic arising from
        :math:`\rho = \Sigma\,\Omega / (\sqrt{2\pi}\,c_s)`:

        .. math::

            c_s = \frac{A + \sqrt{A^2 + 4B}}{2},
            \quad
            A = \frac{a_{\rm rad}\,T^4\,\sqrt{2\pi}}{3\,\Sigma\,\Omega},
            \quad
            B = \frac{k_B\,T}{\mu\,m_p}.

        Parameters
        ----------
        temperature : `~astropy.units.Quantity`
            Midplane temperature :math:`T`.
        Sigma : `~astropy.units.Quantity`
            Column / surface density :math:`\Sigma`.
        Omega : `~astropy.units.Quantity`
            Keplerian angular velocity :math:`\Omega`.

        Returns
        -------
        `~astropy.units.Quantity`
            Isothermal sound speed :math:`c_s` [:math:`\text{cm s}^{-1}`].

        Examples
        --------
        .. code-block:: python

            import astropy.units as u

            RadiativeIdealGas(
                mu=0.62
            ).compute_disk_sound_speed(
                1e7 * u.K,
                Sigma=1e3 * u.Unit("g/cm^2"),
                Omega=1e-3 / u.s,
            )
        """
        log_T = np.log(ensure_in_units(temperature, "K"))
        log_Sigma = np.log(ensure_in_units(Sigma, "g/cm^2"))
        log_Omega = np.log(ensure_in_units(Omega, "1/s"))
        return np.exp(self._compute_log_disk_sound_speed(log_T, log_Sigma, log_Omega)) * u.cm / u.s
