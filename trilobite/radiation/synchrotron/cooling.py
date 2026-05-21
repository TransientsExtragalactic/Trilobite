"""
Cooling module for synchrotron radiation processes in trilobite.

This module provides functions to calculate the cooling rates of electrons
due to synchrotron radiation in magnetic fields. It includes functions to
compute the synchrotron cooling time and the energy loss rate for electrons
in a given magnetic field strength.
"""

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from astropy import constants
from astropy import units as u

from trilobite.radiation.constants import electron_rest_mass_cgs
from trilobite.utils.misc_utils import ensure_in_units

from .core import _opt_compute_synch_frequency

# ========================================================= #
# CGS Constants and Coefficients                            #
# ========================================================= #
_cooling_frequency_coefficient_cgs = (
    (18 * np.pi * constants.m_e * constants.c * constants.e.esu) / (constants.sigma_T**2)
).cgs.value
_synchrotron_cooling_rate_coefficient_cgs = (1 / 6 * np.pi) * constants.sigma_T.cgs.value * constants.c.cgs.value
_synchrotron_cooling_time_coefficient_cgs = (
    6 * np.pi * constants.m_e.cgs.value * constants.c.cgs.value
) / constants.sigma_T.cgs.value

_IC_cooling_rate_coefficient_cgs = constants.sigma_T.cgs.value / (3 * np.pi)
_IC_cooling_time_coefficient_cgs = (
    3 * constants.m_e.cgs.value * constants.c.cgs.value**3
) / constants.sigma_T.cgs.value

# ========================================================= #
# Optimized Low-Level Callables                             #
# ========================================================= #


# --- Generic Functions --- #
def _opt_compute_cooling_time(
    gamma: Union[float, np.ndarray],
    cooling_rate: Union[float, np.ndarray],
):
    """
    Compute the radiative cooling time from an energy loss rate.

    Parameters
    ----------
    gamma : float or array-like
        Electron Lorentz factor.

    cooling_rate : float or array-like
        Energy loss rate dE/dt in erg/s (CGS).

    Returns
    -------
    t_cool : float or array-like
        Cooling time in seconds (CGS).
    """
    return electron_rest_mass_cgs * gamma / cooling_rate


# --- Synchrotron Cooling --- #
def _opt_compute_synchrotron_cooling_rate(
    B: Union[float, np.ndarray],
    gamma: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    r"""
    Compute the synchrotron energy loss rate dE/dt (CGS, optimized).

    Assumes an isotropic pitch-angle distribution and the Thomson regime.

    Parameters
    ----------
    B : float or array-like
        Magnetic field strength in Gauss.

    gamma : float or array-like
        Electron Lorentz factor.

    Returns
    -------
    dEdt : float or array-like
        Synchrotron cooling rate in erg/s (CGS).
    """
    return _synchrotron_cooling_rate_coefficient_cgs * B**2 * gamma**2


def _opt_compute_synchrotron_cooling_time(
    B: Union[float, np.ndarray],
    gamma: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    r"""
    Compute the synchrotron cooling time t_cool (CGS, optimized).

    Parameters
    ----------
    B : float or array-like
        Magnetic field strength in Gauss.

    gamma : float or array-like
        Electron Lorentz factor.

    Returns
    -------
    t_cool : float or array-like
        Synchrotron cooling time in seconds (CGS).

    Notes
    -----
    This expression follows directly from the synchrotron cooling rate:

    .. math::

        t_{cool} = \frac{6\pi m_e c}{\sigma_T B^2 \gamma}
    """
    return _synchrotron_cooling_time_coefficient_cgs / (B**2 * gamma)


def _opt_compute_synchrotron_cooling_gamma(
    B: Union[float, np.ndarray],
    t: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    r"""
    Compute the synchrotron cooling Lorentz factor gamma_c (CGS, optimized).

    The cooling Lorentz factor is defined by the condition
    t_cool(gamma_c) = t.

    Parameters
    ----------
    B : float or array-like
        Magnetic field strength in Gauss.

    t : float or array-like
        Dynamical time in seconds.

    Returns
    -------
    gamma_c : float or array-like
        Synchrotron cooling Lorentz factor.

    Notes
    -----
    The analytic expression is

    .. math::

        \gamma_c = \frac{6\pi m_e c}{\sigma_T B^2 t}
    """
    return _synchrotron_cooling_time_coefficient_cgs / (B**2 * t)


# --- IC Cooling --- #
def _opt_compute_IC_cooling_rate(
    L_bol: Union[float, np.ndarray],
    R: Union[float, np.ndarray],
    gamma: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    r"""
    Compute the inverse Compton energy loss rate dE/dt (CGS, optimized).

    Assumes an isotropic radiation field and the Thomson regime.

    Parameters
    ----------
    L_bol : float or array-like
        Bolometric luminosity of the radiation field in erg/s.

    R : float or array-like
        Distance from the radiation source in cm.

    gamma : float or array-like
        Electron Lorentz factor.

    Returns
    -------
    dEdt : float or array-like
        Inverse Compton cooling rate in erg/s (CGS).
    """
    return _IC_cooling_rate_coefficient_cgs * (L_bol / R**2) * gamma**2


def _opt_compute_IC_cooling_time(
    L_bol: Union[float, np.ndarray],
    R: Union[float, np.ndarray],
    gamma: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    r"""
    Compute the inverse Compton cooling time t_cool (CGS, optimized).

    Parameters
    ----------
    L_bol : float or array-like
        Bolometric luminosity of the radiation field in erg/s.

    R : float or array-like
        Distance from the radiation source in cm.

    gamma : float or array-like
        Electron Lorentz factor.

    Returns
    -------
    t_cool : float or array-like
        Inverse Compton cooling time in seconds (CGS).

    Notes
    -----
    This expression follows directly from the IC cooling rate:

    .. math::

        t_{cool}
        = \frac{3 m_e c^3 R^2}{\sigma_T L_{bol} \gamma}
    """
    return _IC_cooling_time_coefficient_cgs * (R**2 / L_bol) / gamma


def _opt_compute_IC_cooling_gamma(
    L_bol: Union[float, np.ndarray],
    R: Union[float, np.ndarray],
    t: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    r"""
    Compute the inverse Compton cooling Lorentz factor gamma_IC (CGS, optimized).

    The cooling Lorentz factor is defined by the condition
    t_cool(gamma_IC) = t.

    Parameters
    ----------
    L_bol : float or array-like
        Bolometric luminosity of the radiation field in erg/s.

    R : float or array-like
        Distance from the radiation source in cm.

    t : float or array-like
        Dynamical time in seconds.

    Returns
    -------
    gamma_IC : float or array-like
        Inverse Compton cooling Lorentz factor.

    Notes
    -----
    The analytic expression is

    .. math::

        \gamma_{IC}
        = \frac{3 m_e c^3 R^2}{\sigma_T L_{bol} t}
    """
    return _IC_cooling_time_coefficient_cgs * (R**2 / L_bol) / t


# ======================================================== #
# Cooling Engines                                          #
# ======================================================== #
class SynchrotronCoolingEngine(ABC):
    r"""
    Abstract base class for radiative electron cooling engines.

    A cooling engine encapsulates a specific radiative cooling channel
    (e.g. synchrotron, inverse Compton) and provides methods to compute:

    - the radiative energy loss rate,
    - the corresponding cooling time,
    - the cooling Lorentz factor,
    - and a *characteristic synchrotron frequency* associated with that
      Lorentz factor when applicable.

    Cooling engines are **stateless with respect to physical parameters**.
    All quantities (magnetic fields, radiation fields, timescales, etc.)
    must be supplied explicitly at call time. This design is intentional
    and supports efficient inference workflows.

    Subclasses must implement the private ``_compute_*`` methods in CGS
    units. Public ``compute_*`` methods may perform unit coercion or
    validation before dispatching to these implementations.

    See :ref:`synchrotron_theory` for the physical context in which these
    engines are used.
    """

    def __init__(self):
        """
        Initialize the cooling engine.

        Notes
        -----
        Cooling engines should not store physical parameters internally.
        They serve as structured, documented dispatch layers over
        optimized low-level kernels.
        """
        pass

    # --------------------------------------------------------- #
    # Core Computation Interface (CGS, no units)                #
    # --------------------------------------------------------- #
    @abstractmethod
    def _compute_cooling_rate(self, *args, **kwargs):
        r"""
        Compute the radiative energy loss rate.

        Returns the instantaneous energy loss rate

        .. math::

            \left(\frac{dE}{dt}\right)

        in CGS units (erg/s).

        Subclasses must document the physical assumptions and required
        parameters (e.g. magnetic field strength, radiation field
        properties, Lorentz factor).
        """
        raise NotImplementedError

    @abstractmethod
    def _compute_cooling_time(self, *args, **kwargs):
        r"""
        Compute the radiative cooling time.

        Returns the characteristic cooling time

        .. math::

            t_{\rm cool}(\gamma)
            = \frac{E}{|dE/dt|}

        in CGS units (seconds).

        Subclasses must specify the regime and assumptions under which
        the expression is valid.
        """
        raise NotImplementedError

    @abstractmethod
    def _compute_cooling_gamma(self, *args, **kwargs):
        r"""
        Compute the cooling Lorentz factor.

        The cooling Lorentz factor :math:`\gamma_c` is defined implicitly by

        .. math::

            t_{\rm cool}(\gamma_c) = t,

        where :math:`t` is a supplied dynamical or observational timescale.

        Returns
        -------
        gamma_c : float or array-like
            Cooling Lorentz factor (dimensionless).
        """
        raise NotImplementedError

    @abstractmethod
    def _compute_characteristic_frequency(self, *args, **kwargs):
        r"""
        Compute a characteristic synchrotron frequency.

        This method returns the **synchrotron characteristic frequency**
        associated with a physically meaningful Lorentz factor
        (e.g. :math:`\gamma_c`, :math:`\gamma_m`, or :math:`\gamma_{\rm IC}`).

        Notes
        -----
        - Not all cooling channels define an intrinsic emission frequency.
        - For channels such as inverse Compton cooling, this method should
          return the synchrotron frequency corresponding to the cooled
          electrons, or explicitly raise ``NotImplementedError`` if such
          a mapping is undefined.
        """
        raise NotImplementedError

    # --------------------------------------------------------- #
    # Public Interface Methods                                  #
    # --------------------------------------------------------- #
    def compute_cooling_rate(self, *args, **kwargs):
        """
        Public wrapper for computing the radiative cooling rate.

        Subclasses may override this method to perform unit coercion
        or validation before dispatching to the core implementation.
        """
        return self._compute_cooling_rate(*args, **kwargs)

    def compute_cooling_time(self, *args, **kwargs):
        """
        Public wrapper for computing the radiative cooling time.

        Subclasses may override this method to perform unit coercion
        or validation before dispatching to the core implementation.
        """
        return self._compute_cooling_time(*args, **kwargs)

    def compute_cooling_gamma(self, *args, **kwargs):
        """
        Public wrapper for computing the cooling Lorentz factor.

        Subclasses may override this method to perform unit coercion
        or validation before dispatching to the core implementation.
        """
        return self._compute_cooling_gamma(*args, **kwargs)

    def compute_characteristic_frequency(self, *args, **kwargs):
        """
        Public wrapper for computing a characteristic synchrotron frequency.

        Subclasses may override this method to perform unit coercion,
        validation, or composition with other physical processes.
        """
        return self._compute_characteristic_frequency(*args, **kwargs)


class SynchrotronRadiativeCoolingEngine(SynchrotronCoolingEngine):
    r"""
    Synchrotron radiative cooling engine.

    This engine provides synchrotron cooling rates, cooling times,
    cooling Lorentz factors, and the associated characteristic
    synchrotron frequency for relativistic electrons in a magnetic field.

    Cooling rates and cooling times are always **ensemble-averaged**
    (isotropic pitch-angle distribution, Thomson regime).

    The pitch-angle treatment only affects the mapping from Lorentz
    factor :math:`\gamma` to the characteristic synchrotron frequency
    :math:`\nu`.

    See :ref:`synchrotron_theory` for the full physical derivation.
    """

    # --------------------------------------------------------- #
    # Initialization                                           #
    # --------------------------------------------------------- #
    def __init__(self, pitch_averaged: bool = True):
        r"""
        Initialize the synchrotron cooling engine.

        Parameters
        ----------
        pitch_averaged : bool, optional
            Whether to use the pitch-angle–averaged synchrotron
            characteristic frequency.

            If True (default), the mapping uses

            .. math::

                \langle \sin\alpha \rangle = \frac{2}{\pi}

            If False, a specific pitch angle must be supplied via
            ``sin_alpha`` when computing the characteristic frequency.

        Notes
        -----
        This flag **does not** affect cooling rates or cooling times,
        which are always ensemble-averaged.
        """
        super().__init__()
        self._pitch_average = pitch_averaged

    # --------------------------------------------------------- #
    # Core Computation Methods (CGS, no units)                  #
    # --------------------------------------------------------- #
    def _compute_cooling_rate(
        self,
        *,
        B: Union[float, np.ndarray],
        gamma: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        return _opt_compute_synchrotron_cooling_rate(B, gamma)

    def _compute_cooling_time(
        self,
        *,
        B: Union[float, np.ndarray],
        gamma: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        return _opt_compute_synchrotron_cooling_time(B, gamma)

    def _compute_cooling_gamma(
        self,
        *,
        B: Union[float, np.ndarray],
        t: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        return _opt_compute_synchrotron_cooling_gamma(B, t)

    def _compute_characteristic_frequency(
        self,
        *,
        B: Union[float, np.ndarray],
        gamma: Union[float, np.ndarray],
        sin_alpha: Union[float, np.ndarray] = np.pi / 2,
    ) -> Union[float, np.ndarray]:
        return _opt_compute_synch_frequency(
            gamma=gamma,
            B=B,
            sin_alpha=sin_alpha,
            pitch_average=self._pitch_average,
        )

    # --------------------------------------------------------- #
    # Public Interface Methods                                  #
    # --------------------------------------------------------- #
    def compute_cooling_rate(
        self,
        *,
        B: Union[float, np.ndarray, u.Quantity],
        gamma: Union[float, np.ndarray],
    ) -> u.Quantity:
        r"""
        Compute the synchrotron energy loss rate.

        Parameters
        ----------
        B : float, array-like, or astropy.units.Quantity
            Magnetic field strength. Default units are Gauss.

        gamma : float or array-like
            Electron Lorentz factor.

        Returns
        -------
        dEdt : astropy.units.Quantity
            Synchrotron cooling rate in erg/s.
        """
        B = ensure_in_units(B, u.G)
        rate = self._compute_cooling_rate(B=B, gamma=gamma)
        return rate * (u.erg / u.s)

    def compute_cooling_time(
        self,
        *,
        B: Union[float, np.ndarray, u.Quantity],
        gamma: Union[float, np.ndarray],
    ) -> u.Quantity:
        r"""
        Compute the synchrotron cooling time.

        Parameters
        ----------
        B : float, array-like, or astropy.units.Quantity
            Magnetic field strength. Default units are Gauss.

        gamma : float or array-like
            Electron Lorentz factor.

        Returns
        -------
        t_cool : astropy.units.Quantity
            Synchrotron cooling time in seconds.
        """
        B = ensure_in_units(B, u.G)
        t_cool = self._compute_cooling_time(B=B, gamma=gamma)
        return t_cool * u.s

    def compute_cooling_gamma(
        self,
        *,
        B: Union[float, np.ndarray, u.Quantity],
        t: Union[float, np.ndarray, u.Quantity],
    ) -> Union[float, np.ndarray]:
        r"""
        Compute the synchrotron cooling Lorentz factor.

        Parameters
        ----------
        B : float, array-like, or astropy.units.Quantity
            Magnetic field strength. Default units are Gauss.

        t : float, array-like, or astropy.units.Quantity
            Dynamical time. Default units are seconds.

        Returns
        -------
        gamma_c : float or array-like
            Synchrotron cooling Lorentz factor.
        """
        B = ensure_in_units(B, u.G)
        t = ensure_in_units(t, u.s)
        return self._compute_cooling_gamma(B=B, t=t)

    def compute_characteristic_frequency(
        self,
        *,
        B: Union[float, np.ndarray, u.Quantity],
        gamma: Union[float, np.ndarray],
        sin_alpha: Union[float, np.ndarray] = 1.0,
    ) -> u.Quantity:
        r"""
        Compute the synchrotron characteristic frequency.

        Parameters
        ----------
        B : float, array-like, or astropy.units.Quantity
            Magnetic field strength. Default units are Gauss.

        gamma : float or array-like
            Electron Lorentz factor.

        sin_alpha : float or array-like, optional
            Sine of the pitch angle. Only used if ``pitch_averaged=False``.
            Default corresponds to :math:`\alpha = \pi/2`.

        Returns
        -------
        nu : astropy.units.Quantity
            Synchrotron characteristic frequency in Hz.
        """
        B = ensure_in_units(B, u.G)
        nu = self._compute_characteristic_frequency(
            B=B,
            gamma=gamma,
            sin_alpha=sin_alpha,
        )
        return nu * u.Hz


class InverseComptonCoolingEngine(SynchrotronCoolingEngine):
    r"""
    Inverse Compton radiative cooling engine.

    This engine provides inverse Compton (IC) cooling rates, cooling times,
    and cooling Lorentz factors for relativistic electrons interacting with
    an ambient radiation field.

    The implementation assumes:

    - an isotropic radiation field,
    - the Thomson scattering regime,
    - no Klein–Nishina corrections.

    Unlike synchrotron cooling, inverse Compton cooling does **not**
    define a unique characteristic synchrotron frequency. Instead, IC
    cooling modifies the electron energy distribution, which subsequently
    affects the emitted radiation through other channels.

    See :ref:`synchrotron_theory` for discussion of radiative cooling
    processes and their role in shaping non-thermal electron populations.
    """

    # --------------------------------------------------------- #
    # Initialization                                           #
    # --------------------------------------------------------- #
    def __init__(self, pitch_averaged: bool = True):
        r"""
        Instantiate the inverse Compton cooling engine.

        Parameters
        ----------
        pitch_averaged : bool, optional
            Whether to use the pitch-angle–averaged synchrotron
            characteristic frequency.

            If True (default), the mapping uses

            .. math::

                \langle \sin\alpha \rangle = \frac{2}{\pi}

            If False, a specific pitch angle must be supplied via
            ``sin_alpha`` when computing the characteristic frequency.

        Notes
        -----
        This flag **does not** affect cooling rates or cooling times,
        which are always ensemble-averaged.
        """
        super().__init__()
        self._pitch_average = pitch_averaged

    # --------------------------------------------------------- #
    # Core Computation Methods (CGS, no units)                  #
    # --------------------------------------------------------- #
    def _compute_cooling_rate(
        self,
        *,
        L_bol: Union[float, np.ndarray],
        R: Union[float, np.ndarray],
        gamma: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        r"""
        Inverse Compton energy loss rate.

        Implements

        .. math::

            \left(\frac{dE}{dt}\right)_{\rm IC}
            = \frac{4}{3}\,\sigma_T c\,\gamma^2\,u_{\rm rad}

        with the radiation energy density

        .. math::

            u_{\rm rad} = \frac{L_{\rm bol}}{4\pi R^2 c}

        See :ref:`synchrotron_theory`.
        """
        return _opt_compute_IC_cooling_rate(L_bol, R, gamma)

    def _compute_cooling_time(
        self,
        *,
        L_bol: Union[float, np.ndarray],
        R: Union[float, np.ndarray],
        gamma: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        r"""
        Inverse Compton cooling time.

        Implements

        .. math::

            t_{\rm cool}(\gamma)
            = \frac{3 m_e c^3 R^2}{\sigma_T L_{\rm bol}\,\gamma}
        """
        return _opt_compute_IC_cooling_time(L_bol, R, gamma)

    def _compute_cooling_gamma(
        self,
        *,
        L_bol: Union[float, np.ndarray],
        R: Union[float, np.ndarray],
        t: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        r"""
        Inverse Compton cooling Lorentz factor.

        Defined implicitly by

        .. math::

            t_{\rm cool}(\gamma_{\rm IC}) = t

        giving

        .. math::

            \gamma_{\rm IC}
            = \frac{3 m_e c^3 R^2}{\sigma_T L_{\rm bol} t}
        """
        return _opt_compute_IC_cooling_gamma(L_bol, R, t)

    def _compute_characteristic_frequency(
        self,
        *,
        B: Union[float, np.ndarray],
        L_bol: Union[float, np.ndarray],
        R: Union[float, np.ndarray],
        t: Union[float, np.ndarray],
        sin_alpha: Union[float, np.ndarray] = 1.0,
    ) -> Union[float, np.ndarray]:
        r"""
        Synchrotron characteristic frequency of inverse-Compton–cooled electrons.

        This computes the synchrotron frequency associated with electrons
        at the inverse Compton cooling Lorentz factor :math:`\gamma_{\rm IC}`.

        Notes
        -----
        This is **not** an inverse Compton emission frequency. It is the
        synchrotron frequency corresponding to IC-cooled electrons:

        .. math::

            \gamma_{\rm IC}
            = \frac{3 m_e c^3 R^2}{\sigma_T L_{\rm bol} t}

            \nu_{\rm syn, IC}
            = \nu_{\rm syn}(\gamma_{\rm IC}, B)
        """
        gamma_IC = self._compute_cooling_gamma(
            L_bol=L_bol,
            R=R,
            t=t,
        )

        return _opt_compute_synch_frequency(
            gamma=gamma_IC,
            B=B,
            sin_alpha=sin_alpha,
            pitch_average=True,
        )

    # --------------------------------------------------------- #
    # Public Interface Methods                                  #
    # --------------------------------------------------------- #
    def compute_cooling_rate(
        self,
        *,
        L_bol: Union[float, np.ndarray, u.Quantity],
        R: Union[float, np.ndarray, u.Quantity],
        gamma: Union[float, np.ndarray],
    ) -> u.Quantity:
        r"""
        Compute the inverse Compton energy loss rate.

        Parameters
        ----------
        L_bol : float, array-like, or astropy.units.Quantity
            Bolometric luminosity of the radiation field.
            Default units are erg/s.

        R : float, array-like, or astropy.units.Quantity
            Distance from the radiation source.
            Default units are cm.

        gamma : float or array-like
            Electron Lorentz factor.

        Returns
        -------
        dEdt : astropy.units.Quantity
            Inverse Compton cooling rate in erg/s.
        """
        L_bol = ensure_in_units(L_bol, u.erg / u.s)
        R = ensure_in_units(R, u.cm)

        rate = self._compute_cooling_rate(
            L_bol=L_bol,
            R=R,
            gamma=gamma,
        )
        return rate * (u.erg / u.s)

    def compute_cooling_time(
        self,
        *,
        L_bol: Union[float, np.ndarray, u.Quantity],
        R: Union[float, np.ndarray, u.Quantity],
        gamma: Union[float, np.ndarray],
    ) -> u.Quantity:
        r"""
        Compute the inverse Compton cooling time.

        Parameters
        ----------
        L_bol : float, array-like, or astropy.units.Quantity
            Bolometric luminosity of the radiation field.
            Default units are erg/s.

        R : float, array-like, or astropy.units.Quantity
            Distance from the radiation source.
            Default units are cm.

        gamma : float or array-like
            Electron Lorentz factor.

        Returns
        -------
        t_cool : astropy.units.Quantity
            Inverse Compton cooling time in seconds.
        """
        L_bol = ensure_in_units(L_bol, u.erg / u.s)
        R = ensure_in_units(R, u.cm)

        t_cool = self._compute_cooling_time(
            L_bol=L_bol,
            R=R,
            gamma=gamma,
        )
        return t_cool * u.s

    def compute_cooling_gamma(
        self,
        *,
        L_bol: Union[float, np.ndarray, u.Quantity],
        R: Union[float, np.ndarray, u.Quantity],
        t: Union[float, np.ndarray, u.Quantity],
    ) -> Union[float, np.ndarray]:
        r"""
        Compute the inverse Compton cooling Lorentz factor.

        Parameters
        ----------
        L_bol : float, array-like, or astropy.units.Quantity
            Bolometric luminosity of the radiation field.
            Default units are erg/s.

        R : float, array-like, or astropy.units.Quantity
            Distance from the radiation source.
            Default units are cm.

        t : float, array-like, or astropy.units.Quantity
            Dynamical time.
            Default units are seconds.

        Returns
        -------
        gamma_IC : float or array-like
            Inverse Compton cooling Lorentz factor.
        """
        L_bol = ensure_in_units(L_bol, u.erg / u.s)
        R = ensure_in_units(R, u.cm)
        t = ensure_in_units(t, u.s)

        return self._compute_cooling_gamma(
            L_bol=L_bol,
            R=R,
            t=t,
        )

    def compute_characteristic_frequency(
        self,
        *,
        B: Union[float, np.ndarray, u.Quantity],
        L_bol: Union[float, np.ndarray, u.Quantity],
        R: Union[float, np.ndarray, u.Quantity],
        t: Union[float, np.ndarray, u.Quantity],
        sin_alpha: Union[float, np.ndarray] = 1.0,
    ) -> u.Quantity:
        r"""
        Compute the synchrotron characteristic frequency corresponding to inverse-Compton–cooled electrons.

        Parameters
        ----------
        B : float, array-like, or astropy.units.Quantity
            Magnetic field strength. Default units are Gauss.

        L_bol : float, array-like, or astropy.units.Quantity
            Bolometric luminosity of the radiation field.
            Default units are erg/s.

        R : float, array-like, or astropy.units.Quantity
            Distance from the radiation source.
            Default units are cm.

        t : float, array-like, or astropy.units.Quantity
            Dynamical time. Default units are seconds.

        sin_alpha : float or array-like, optional
            Sine of the pitch angle. Only relevant if pitch-angle
            averaging is disabled elsewhere.

        Returns
        -------
        nu : astropy.units.Quantity
            Synchrotron characteristic frequency of IC-cooled electrons.
        """
        B = ensure_in_units(B, u.G)
        L_bol = ensure_in_units(L_bol, u.erg / u.s)
        R = ensure_in_units(R, u.cm)
        t = ensure_in_units(t, u.s)

        nu = self._compute_characteristic_frequency(
            B=B,
            L_bol=L_bol,
            R=R,
            t=t,
            sin_alpha=sin_alpha,
        )
        return nu
