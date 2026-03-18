r"""
Equation of state (EOS) utilities for Triceratops models.

This module provides a class hierarchy for computing thermodynamic quantities
(pressure, sound speed) under several common astrophysical equations of state.
All classes follow the same two-level API: private ``_compute_log_*`` methods
operate directly on log-space CGS floats for numerical efficiency, while public
``compute_*`` methods accept :class:`~astropy.units.Quantity` inputs and return
unit-bearing outputs.

See Also
--------
:mod:`.composition` : Mean molecular weight helpers used to set ``mu``.
:mod:`.constants` : CGS constants consumed internally.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u

from ..utils.misc_utils import ensure_in_units
from .constants import _log_a_rad_cgs, _log_c_cgs, _log_k_B_cgs, _log_m_p_cgs

if TYPE_CHECKING:
    from .._typing import _ArrayLike, _UnitBearingArrayLike


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

    See Also
    --------
    :class:`IdealGasEOS`, :class:`RadiationEOS`, :class:`IdealGasRadiationEOS`
    """

    def __init__(self, **params):
        """
        Instantiate the EOS with the given parameters.

        Parameters
        ----------
        params: dict
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
# Ideal Gas                                                           #
# ================================================================== #


class IdealGasEOS(EquationOfState):
    r"""
    Ideal (monatomic) gas equation of state.

    The thermal pressure and adiabatic sound speed are

    .. math::

        P_{\rm gas} = \frac{\rho k_B T}{\mu m_p},
        \qquad
        c_s = \sqrt{\frac{\gamma k_B T}{\mu m_p}},

    with adiabatic index :math:`\gamma = 5/3` for a monatomic ideal gas.

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

        \mu = \frac{\bar{m}}{m_p} \equiv \frac{\rho}{n_{\rm tot}\, m_p}.

    For a fully ionised hydrogen-helium plasma with mass fractions
    :math:`X = 0.70,\, Y = 0.28`, :math:`\mu \approx 0.62`.

    See Also
    --------
    :func:`~.composition.compute_mean_molecular_weight` : helper to derive
        ``mu`` from hydrogen/helium/metal mass fractions.

    Examples
    --------
    Compute the gas pressure at :math:`\rho = 10^{-3}` g/cm³, :math:`T = 10^7` K:

    .. code-block:: python

        import astropy.units as u

        eos = IdealGasEOS(mu=0.62)
        eos.compute_pressure(
            1e-3 * u.Unit("g/cm^3"), 1e7 * u.K
        )

    Compute the adiabatic sound speed at :math:`T = 10^7` K:

    .. code-block:: python

        eos.compute_sound_speed(1e7 * u.K)
    """

    def __init__(self, mu: float = 0.6, **kwargs):
        r"""
        Initialize the ideal gas equation of state.

        Parameters
        ----------
        mu : float, optional
            Mean molecular weight of the gas. This sets the conversion between
            temperature and internal energy via the ideal gas law. The default
            value of ``0.6`` is appropriate for a fully ionized plasma with
            primordial composition.

        **kwargs
            Additional keyword arguments passed to the base
            :class:`EquationOfState` initializer. These may include parameters
            such as the adiabatic index ``gamma`` or other model-specific
            quantities.

        Notes
        -----
        The mean molecular weight :math:`\mu` enters thermodynamic relations as

        .. math::

            P = \frac{\rho k_B T}{\mu m_p},

        and therefore controls the relationship between pressure, temperature,
        and internal energy in the gas.

        For ionized astrophysical plasmas, typical values are:

        - :math:`\mu \approx 0.6` for fully ionized gas (default)
        - :math:`\mu \approx 1.3` for neutral gas

        """
        super().__init__(mu=mu, **kwargs)
        self.mu = mu

    # -------------------------------------------------------------- #
    # Private (log-space CGS)                                        #
    # -------------------------------------------------------------- #

    def _compute_log_pressure(self, log_density: "_ArrayLike", log_temperature: "_ArrayLike") -> "_ArrayLike":
        r"""
        Compute :math:`\ln P_{\rm gas}` in CGS units.

        .. math::

            \ln P = \ln k_B + \ln T + \ln \rho - \ln \mu - \ln m_p

        Parameters
        ----------
        log_density : float or ndarray
            :math:`\ln \rho` [:math:`\ln(\text{g cm}^{-3})`].
        log_temperature : float or ndarray
            :math:`\ln T` [:math:`\ln(\text{K})`].

        Returns
        -------
        float or ndarray
            :math:`\ln P_{\rm gas}` [:math:`\ln(\text{erg cm}^{-3})`].
        """
        log_temperature, log_density = np.asarray(log_temperature), np.asarray(log_density)

        log_pressure = _log_k_B_cgs + log_temperature + log_density - np.log(self.mu) - _log_m_p_cgs

        return log_pressure if log_pressure.ndim > 0 else log_pressure.item()

    def _compute_log_sound_speed(self, log_temperature: "_ArrayLike") -> "_ArrayLike":
        r"""
        Compute :math:`\ln c_s` in CGS units.

        .. math::

            \ln c_s = \tfrac{1}{2}\left(
                \ln\gamma + \ln k_B + \ln T - \ln\mu - \ln m_p
            \right), \quad \gamma = \tfrac{5}{3}

        Parameters
        ----------
        log_temperature : float or ndarray
            :math:`\ln T` [:math:`\ln(\text{K})`].

        Returns
        -------
        float or ndarray
            :math:`\ln c_s` [:math:`\ln(\text{cm s}^{-1})`].
        """
        log_temperature = np.asarray(log_temperature)

        log_cs = 0.5 * (np.log(5.0 / 3.0) + _log_k_B_cgs + log_temperature - np.log(self.mu) - _log_m_p_cgs)

        return log_cs if log_cs.ndim > 0 else log_cs.item()

    # -------------------------------------------------------------- #
    # Public (unit-bearing)                                           #
    # -------------------------------------------------------------- #

    def compute_pressure(self, density: u.Quantity, temperature: u.Quantity) -> u.Quantity:
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
        Compute the adiabatic sound speed.

        Parameters
        ----------
        temperature : `~astropy.units.Quantity`
            Temperature :math:`T`.

        Returns
        -------
        `~astropy.units.Quantity`
            Adiabatic sound speed :math:`c_s` [:math:`\text{cm s}^{-1}`].

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
# Radiation                                                           #
# ================================================================== #


class RadiationEOS(EquationOfState):
    r"""
    Pure radiation equation of state.

    The radiation pressure and sound speed are

    .. math::

        P_{\rm rad} = \frac{a_{\rm rad}}{3} T^4,
        \qquad
        c_s = \frac{c}{\sqrt{3}},

    where :math:`a_{\rm rad} = 4\sigma_{\rm SB}/c` is the radiation energy
    density constant.

    Notes
    -----
    This EOS describes a photon gas or a radiation-dominated fluid.
    The sound speed :math:`c/\sqrt{3}` is the relativistic result for a
    gas with adiabatic index :math:`\gamma = 4/3`.

    See Also
    --------
    :class:`IdealGasRadiationEOS` : combined gas + radiation EOS.

    Examples
    --------
    Compute the radiation pressure at :math:`T = 10^8` K:

    .. code-block:: python

        import astropy.units as u

        RadiationEOS().compute_pressure(1e8 * u.K)

    Radiation sound speed (independent of temperature):

    .. code-block:: python

        RadiationEOS().compute_sound_speed()
    """

    def __init__(self, **kwargs):
        """
        Initialize the radiation equation of state.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to the base
            :class:`EquationOfState` initializer.
        """
        super().__init__(**kwargs)

    # -------------------------------------------------------------- #
    # Private (log-space CGS)                                        #
    # -------------------------------------------------------------- #

    def _compute_log_pressure(self, log_temperature: "_ArrayLike") -> "_ArrayLike":
        r"""
        Compute :math:`\ln P_{\rm rad}` in CGS units.

        .. math::

            \ln P_{\rm rad} = \ln\!\left(\frac{a_{\rm rad}}{3}\right) + 4\ln T

        Parameters
        ----------
        log_temperature : float or ndarray
            :math:`\ln T` [:math:`\ln(\text{K})`].

        Returns
        -------
        float or ndarray
            :math:`\ln P_{\rm rad}` [:math:`\ln(\text{erg cm}^{-3})`].
        """
        log_temperature = np.asarray(log_temperature)

        log_pressure = _log_a_rad_cgs - np.log(3.0) + 4.0 * log_temperature

        return log_pressure if log_pressure.ndim > 0 else log_pressure.item()

    def _compute_log_sound_speed(self) -> float:
        r"""
        Compute :math:`\ln c_s` for a radiation-dominated fluid.

        .. math::

            c_s = \frac{c}{\sqrt{3}}
            \implies
            \ln c_s = \ln c - \tfrac{1}{2}\ln 3

        Returns
        -------
        float
            :math:`\ln c_s` [:math:`\ln(\text{cm s}^{-1})`].
        """
        return _log_c_cgs - 0.5 * np.log(3.0)

    # -------------------------------------------------------------- #
    # Public (unit-bearing)                                           #
    # -------------------------------------------------------------- #

    def compute_pressure(self, temperature: u.Quantity) -> u.Quantity:
        r"""
        Compute the radiation pressure.

        Parameters
        ----------
        temperature : `~astropy.units.Quantity`
            Temperature :math:`T`.

        Returns
        -------
        `~astropy.units.Quantity`
            Radiation pressure :math:`P_{\rm rad}` [:math:`\text{erg cm}^{-3}`].

        Examples
        --------
        .. code-block:: python

            import astropy.units as u

            RadiationEOS().compute_pressure(1e8 * u.K)
        """
        log_temperature = np.log(ensure_in_units(temperature, "K"))
        return np.exp(self._compute_log_pressure(log_temperature)) * u.erg / u.cm**3

    def compute_sound_speed(self) -> u.Quantity:
        r"""
        Return the radiation sound speed :math:`c_s = c / \sqrt{3}`.

        This is independent of temperature and density.

        Returns
        -------
        `~astropy.units.Quantity`
            Sound speed [:math:`\text{cm s}^{-1}`].

        Examples
        --------
        .. code-block:: python

            RadiationEOS().compute_sound_speed()
        """
        return np.exp(self._compute_log_sound_speed()) * u.cm / u.s


# ================================================================== #
# Combined Ideal Gas + Radiation                                      #
# ================================================================== #


class IdealGasRadiationEOS(EquationOfState):
    r"""
    Combined ideal gas plus radiation equation of state.

    The total pressure is the sum of the gas and radiation contributions:

    .. math::

        P_{\rm tot} = P_{\rm gas} + P_{\rm rad}
        = \frac{\rho k_B T}{\mu m_p} + \frac{a_{\rm rad}}{3} T^4.

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

        T_{\rm eq} = \left(\frac{3 \rho k_B}{a_{\rm rad} \mu m_p}\right)^{1/3}.

    See Also
    --------
    :class:`IdealGasEOS` : gas-only limit.
    :class:`RadiationEOS` : radiation-only limit.
    :func:`~.composition.compute_mean_molecular_weight` : derive ``mu`` from
        composition.

    Examples
    --------
    Compute the total pressure at :math:`\rho = 1` g/cm³, :math:`T = 10^7` K:

    .. code-block:: python

        import astropy.units as u

        eos = IdealGasRadiationEOS(mu=0.62)
        eos.compute_pressure(
            1.0 * u.Unit("g/cm^3"), 1e7 * u.K
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
            Additional keyword arguments passed to the base
            :class:`EquationOfState` initializer.

        Notes
        -----
        The mean molecular weight :math:`\mu` enters the gas pressure term as

        .. math::

            P_{\rm gas} = \frac{\rho k_B T}{\mu m_p}.

        The radiation term :math:`P_{\rm rad} = a_{\rm rad} T^4 / 3` is
        independent of :math:`\mu`.  For ionized astrophysical plasmas, typical
        values are :math:`\mu \approx 0.6` (fully ionized) to
        :math:`\mu \approx 1.3` (neutral).
        """
        super().__init__(mu=mu, **kwargs)
        self.mu = mu

    # -------------------------------------------------------------- #
    # Private (log-space CGS)                                        #
    # -------------------------------------------------------------- #

    def _compute_log_pressure(self, log_density: "_ArrayLike", log_temperature: "_ArrayLike") -> "_ArrayLike":
        r"""
        Compute :math:`\ln P_{\rm tot}` using :func:`numpy.logaddexp`.

        .. math::

            P_{\rm tot} = P_{\rm gas} + P_{\rm rad}
            \implies
            \ln P_{\rm tot} = \ln\!\left(e^{\ln P_{\rm gas}} + e^{\ln P_{\rm rad}}\right)

        Parameters
        ----------
        log_density : float or ndarray
            :math:`\ln \rho` [:math:`\ln(\text{g cm}^{-3})`].
        log_temperature : float or ndarray
            :math:`\ln T` [:math:`\ln(\text{K})`].

        Returns
        -------
        float or ndarray
            :math:`\ln P_{\rm tot}` [:math:`\ln(\text{erg cm}^{-3})`].
        """
        log_temperature, log_density = np.asarray(log_temperature), np.asarray(log_density)

        log_P_gas = _log_k_B_cgs + log_temperature + log_density - np.log(self.mu) - _log_m_p_cgs
        log_P_rad = _log_a_rad_cgs - np.log(3.0) + 4.0 * log_temperature

        log_pressure = np.logaddexp(log_P_gas, log_P_rad)

        return log_pressure if log_pressure.ndim > 0 else log_pressure.item()

    # -------------------------------------------------------------- #
    # Public (unit-bearing)                                           #
    # -------------------------------------------------------------- #

    def compute_pressure(self, density: u.Quantity, temperature: u.Quantity) -> u.Quantity:
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

            IdealGasRadiationEOS(mu=0.62).compute_pressure(
                1.0 * u.Unit("g/cm^3"), 1e7 * u.K
            )
        """
        log_density = np.log(ensure_in_units(density, "g/cm^3"))
        log_temperature = np.log(ensure_in_units(temperature, "K"))
        return np.exp(self._compute_log_pressure(log_density, log_temperature)) * u.erg / u.cm**3
