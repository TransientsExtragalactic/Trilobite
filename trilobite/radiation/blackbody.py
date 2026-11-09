"""
Planck (blackbody) radiation functions.

Provides both high-performance CGS float implementations for use inside
MCMC hot loops and unit-aware public wrappers for interactive use.

The module follows the two-level API convention used throughout Trilobite:

- ``_*_cgs`` functions accept plain NumPy arrays / Python floats in CGS and
  return the same.  They carry no unit overhead and are safe to call inside
  MCMC loops.
- Public wrappers (no underscore prefix) accept :class:`~astropy.units.Quantity`
  objects and return a :class:`~astropy.units.Quantity` with the correct unit.
"""

from typing import TYPE_CHECKING

import numpy as np
from astropy import cosmology as cosmo
from astropy import units as u

from trilobite.physics_utils.cosmology import get_cosmology, resolve_cosmological_distances
from trilobite.utils.misc_utils import ensure_in_units

from .constants import c_cgs, h_cgs, kB_cgs, sigma_sb_cgs

__all__ = [
    # Specific intensity (nu= or lam=)
    "planck_B",
    # Radiative flux from a blackbody surface (nu= or lam=)
    "planck_surface_flux",
    # Observed flux density from a spherical source (nu= or lam=, distance)
    "planck_flux",
    # Specific luminosity of a spherical source (nu= or lam=, R)
    "planck_luminosity",
    # UV-suppressed (modified) blackbody
    "planck_cutoff_fraction",
    # Bolometric fluxes
    "stefan_boltzmann_flux",
    "stefan_boltzmann_observed_flux",
    # Wien's displacement law
    "wien_peak_frequency",
    "wien_peak_wavelength",
    # Photospheric quantities
    "photospheric_radius",
    "photospheric_temperature",
    "photospheric_radius_from_flux",
    "photospheric_temperature_from_flux",
]

if TYPE_CHECKING:
    from trilobite._typing import _ArrayLike, _UnitBearingArrayLike, _UnitBearingScalarLike

# ---------------------------------------------------------------------------
# Pre-computed CGS coefficients (extracted once for MCMC performance)
# ---------------------------------------------------------------------------
_PLANCK_FNU_PREFACTOR_CGS: float = 2.0 * h_cgs / c_cgs**2
_PLANCK_FLAMBDA_PREFACTOR_CGS: float = 2.0 * h_cgs * c_cgs**2
_H_OVER_KB_CGS: float = h_cgs / kB_cgs
_HC_OVER_KB_CGS: float = h_cgs * c_cgs / kB_cgs
_WIEN_B_FREQ_CGS: float = 2.821439372122079 * kB_cgs / h_cgs  # Hz / K
_WIEN_B_WAV_CGS: float = _HC_OVER_KB_CGS / 4.965114231744276  # cm K
_LOG_4PI_SIGMA = np.log(4.0 * np.pi * sigma_sb_cgs)

# ---------------------------------------------------------------------------
# Low-Level (Private) Interface
# ---------------------------------------------------------------------------

# --- Specific Intensity Functions --- #


def _log_planck_fnu_cgs(log_nu: "_ArrayLike", log_T: "_ArrayLike") -> "_ArrayLike":
    r"""Evaluate :math:`\ln B_\nu` (Planck function) in CGS units.

    This is a numerically stable, log-space implementation suitable for use
    in performance-critical contexts (e.g., MCMC loops).

    Parameters
    ----------
    log_nu : float, int, or ~numpy.ndarray
        Natural logarithm of frequency in Hz.
    log_T : float, int, or ~numpy.ndarray
        Natural logarithm of temperature in K.

    Returns
    -------
    float or ~numpy.ndarray
        Logarithm of the Planck spectral radiance :math:`B_\nu` in
        :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}\,sr^{-1}}`.

    Notes
    -----
    Uses the identity

    .. math::

        \ln(e^x - 1) = x + \ln(1 - e^{-x})

    for numerical stability at large :math:`x`.
    """
    log_nu = np.asarray(log_nu)
    log_T = np.asarray(log_T)

    # x = h nu / (k_B T)
    x = _H_OVER_KB_CGS * np.exp(log_nu - log_T)

    # log(e^x - 1) in stable form
    log_denom = x + np.log1p(-np.exp(-x))

    res = np.log(_PLANCK_FNU_PREFACTOR_CGS) + 3.0 * log_nu - log_denom

    return res if res.ndim > 0 else float(res)


def _log_planck_flambda_cgs(log_lambda: "_ArrayLike", log_T: "_ArrayLike") -> "_ArrayLike":
    r"""Evaluate :math:`\ln B_\lambda` (Planck function) in CGS units.

    This is a numerically stable, log-space implementation suitable for use
    in performance-critical contexts (e.g., MCMC loops).

    Parameters
    ----------
    log_lambda : float, int, or ~numpy.ndarray
        Natural logarithm of wavelength in cm.
    log_T : float, int, or ~numpy.ndarray
        Natural logarithm of temperature in K.

    Returns
    -------
    float or ~numpy.ndarray
        Logarithm of the Planck spectral radiance :math:`B_\lambda` in
        :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,cm^{-1}\,sr^{-1}}`.

    Notes
    -----
    Uses the identity

    .. math::

        \ln(e^x - 1) = x + \ln(1 - e^{-x})

    for numerical stability at large :math:`x`.
    """
    log_lambda = np.asarray(log_lambda)
    log_T = np.asarray(log_T)

    # Dimensionless exponent: x = hc / (λ k_B T)
    x = _HC_OVER_KB_CGS * np.exp(-log_lambda - log_T)

    # Stable evaluation of log(e^x - 1)
    log_denom = x + np.log1p(-np.exp(-x))

    result = np.log(_PLANCK_FLAMBDA_PREFACTOR_CGS) - 5.0 * log_lambda - log_denom

    return result if result.ndim > 0 else float(result)


# --- Specific Flux Functions --- #
def _log_specific_rad_flux_fnu_cgs(
    log_nu: "_ArrayLike",
    log_T: "_ArrayLike",
) -> "_ArrayLike":
    r"""Evaluate :math:`\ln F_\nu` for a blackbody surface in CGS units.

    This returns the outward specific flux emitted per unit surface area of an
    isotropically emitting blackbody surface:

    .. math::

        F_\nu = \pi B_\nu

    Parameters
    ----------
    log_nu : float, int, or ~numpy.ndarray
        Natural logarithm of frequency in Hz.
    log_T : float, int, or ~numpy.ndarray
        Natural logarithm of temperature in K.

    Returns
    -------
    float or ~numpy.ndarray
        Logarithm of the specific radiative flux :math:`F_\nu` in
        :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}}`.
    """
    result = _log_planck_fnu_cgs(log_nu, log_T) + np.log(np.pi)
    return result if np.ndim(result) > 0 else float(result)


def _log_specific_flux_fnu_cgs(
    log_nu: "_ArrayLike",
    log_T: "_ArrayLike",
    log_R: "_ArrayLike",
    log_D: "_ArrayLike",
    z: "_ArrayLike" = 0,
    lambda_cut: float = 0.0,
) -> "_ArrayLike":
    r"""Evaluate :math:`\ln f_\nu` for a spherical blackbody observed at distance.

    For zero redshift, this uses

    .. math::

        f_\nu = \pi B_\nu \left(\frac{R}{D}\right)^2

    where :math:`R` is the emitter radius and :math:`D` is the distance.

    For nonzero redshift, this assumes `log_nu` is observer-frame frequency and
    `D` is luminosity distance, so that

    .. math::

        f_{\nu,\mathrm{obs}}(\nu_{\mathrm{obs}})
        =
        \frac{(1+z)\,L_{\nu,\mathrm{em}}((1+z)\nu_{\mathrm{obs}})}
             {4\pi D_L^2},

    which conserves the bolometric flux, :math:`\int f_\nu\,d\nu = L/4\pi D_L^2`.

    If ``lambda_cut`` is given, the spectrum is multiplied by the UV-suppression factor of
    :func:`_log_cutoff_factor_cgs` at the rest-frame wavelength.

    Parameters
    ----------
    log_nu : float, int, or ~numpy.ndarray
        Natural logarithm of observer-frame frequency in Hz.
    log_T : float, int, or ~numpy.ndarray
        Natural logarithm of temperature in K.
    log_R : float, int, or ~numpy.ndarray
        Natural logarithm of source radius in cm.
    log_D : float, int, or ~numpy.ndarray
        Natural logarithm of luminosity distance in cm.
    z : float, int, or ~numpy.ndarray, optional
        Redshift. Default is 0.
    lambda_cut : float, optional
        Rest-frame cutoff wavelength in cm below which the spectrum is suppressed. Default is 0 (no suppression).

    Returns
    -------
    float or ~numpy.ndarray
        Logarithm of the observed specific flux density :math:`f_\nu` in
        :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}}`.
    """
    log1pz = np.log1p(np.asarray(z))
    log_nu_rest = np.asarray(log_nu) + log1pz

    result = (
        _log_planck_fnu_cgs(log_nu_rest, log_T)
        + np.log(np.pi)
        + 2.0 * (np.asarray(log_R) - np.asarray(log_D))
        + log1pz
        + _log_cutoff_factor_cgs(np.log(c_cgs) - log_nu_rest, log_T, lambda_cut)
    )

    return result if np.ndim(result) > 0 else float(result)


# Terms in the series for the surviving fraction of a UV-suppressed blackbody. At a cutoff of 3000 Angstrom the relative
# error against a 30-digit reference is below 1e-14 for T <= 3e5 K, 3e-11 at 1e6 K, and 5e-6 at 1e7 K, where the series
# has not converged (the terms fall as exp(-n x_c), with x_c = hc / (lambda_cut k T)).
_CUTOFF_SERIES_N: np.ndarray = np.arange(1, 257, dtype=float)


def _cutoff_transmitted_fraction_cgs(T: "_ArrayLike", lambda_cut: float) -> "_ArrayLike":
    r"""Fraction of a blackbody's bolometric flux that survives linear suppression below ``lambda_cut``.

    The suppression multiplies the spectrum by :math:`\min(1, \lambda/\lambda_{\rm cut})`, so
    that the transmission is 100% at the cutoff and falls linearly to zero at :math:`\lambda = 0`. With
    :math:`x_c = hc/(\lambda_{\rm cut} k_B T)`, the surviving fraction

    .. math::

        f(T) = \frac{\int B_\lambda \min(1, \lambda/\lambda_{\rm cut})\,d\lambda}{\int B_\lambda\,d\lambda}
        = 1 - \frac{15}{\pi^4}\sum_{n=1}^{\infty} e^{-n x_c}
        \left(\frac{x_c^2}{n^2} + \frac{4 x_c}{n^3} + \frac{6}{n^4}\right)

    is evaluated exactly with the series, which converges geometrically.

    Parameters
    ----------
    T : float or ~numpy.ndarray
        Temperature in K.
    lambda_cut : float
        Cutoff wavelength in cm. A value of zero or less means no suppression and returns 1.

    Returns
    -------
    float or ~numpy.ndarray
        The surviving fraction, between 0 and 1.
    """
    T = np.asarray(T, dtype=float)
    if lambda_cut <= 0.0:
        return np.ones_like(T)
    x_c = (_HC_OVER_KB_CGS / (lambda_cut * T))[..., None]
    n = _CUTOFF_SERIES_N
    series = np.sum(np.exp(-n * x_c) * (x_c**2 / n**2 + 4.0 * x_c / n**3 + 6.0 / n**4), axis=-1)
    return 1.0 - (15.0 / np.pi**4) * series


def _log_cutoff_factor_cgs(log_wavelength: "_ArrayLike", log_T: "_ArrayLike", lambda_cut: float) -> "_ArrayLike":
    r"""Evaluate :math:`\ln[\min(1, \lambda/\lambda_{\rm cut})/f(T)]`, the log of the UV-suppression factor.

    The spectrum of a blackbody with a suppressed UV is the blackbody times this factor, which is the linear
    suppression below ``lambda_cut`` divided by the surviving fraction :math:`f(T)` of
    :func:`_cutoff_transmitted_fraction_cgs`, so that the bolometric flux is that of the plain blackbody.

    Parameters
    ----------
    log_wavelength : float, int, or ~numpy.ndarray
        Natural logarithm of the **rest-frame** wavelength in cm.
    log_T : float, int, or ~numpy.ndarray
        Natural logarithm of the temperature in K.
    lambda_cut : float
        Cutoff wavelength in cm. Zero or less means no suppression, and the factor is 1 (its log is 0).

    Returns
    -------
    float or ~numpy.ndarray
        The log of the factor, which broadcasts over its inputs.
    """
    if lambda_cut <= 0.0:
        return 0.0
    return np.log(np.minimum(np.exp(log_wavelength) / lambda_cut, 1.0)) - np.log(
        _cutoff_transmitted_fraction_cgs(np.exp(log_T), lambda_cut)
    )


def _log_specific_rad_flux_flambda_cgs(
    log_lambda: "_ArrayLike",
    log_T: "_ArrayLike",
) -> "_ArrayLike":
    r"""Evaluate :math:`\ln F_\lambda` for a blackbody surface in CGS units.

    This returns the outward specific flux emitted per unit surface area of an
    isotropically emitting blackbody surface:

    .. math::

        F_\lambda = \pi B_\lambda

    Parameters
    ----------
    log_lambda : float, int, or ~numpy.ndarray
        Natural logarithm of wavelength in cm.
    log_T : float, int, or ~numpy.ndarray
        Natural logarithm of temperature in K.

    Returns
    -------
    float or ~numpy.ndarray
        Logarithm of the specific radiative flux :math:`F_\lambda` in
        :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,cm^{-1}}`.
    """
    result = _log_planck_flambda_cgs(log_lambda, log_T) + np.log(np.pi)
    return result if np.ndim(result) > 0 else float(result)


def _log_specific_flux_flambda_cgs(
    log_lambda: "_ArrayLike",
    log_T: "_ArrayLike",
    log_R: "_ArrayLike",
    log_D: "_ArrayLike",
    z: "_ArrayLike" = 0,
    lambda_cut: float = 0.0,
) -> "_ArrayLike":
    r"""Evaluate :math:`\ln f_\lambda` for a spherical blackbody observed at distance.

    For zero redshift, this uses

    .. math::

        f_\lambda = \pi B_\lambda \left(\frac{R}{D}\right)^2

    where :math:`R` is the emitter radius and :math:`D` is the distance.

    For nonzero redshift, this assumes `log_lambda` is observer-frame wavelength
    and `D` is luminosity distance, so that

    .. math::

        f_{\lambda,\mathrm{obs}}(\lambda_{\mathrm{obs}})
        =
        \frac{L_{\lambda,\mathrm{em}}(\lambda_{\mathrm{obs}}/(1+z))}
             {4\pi D_L^2 (1+z)},

    which conserves the bolometric flux, :math:`\int f_\lambda\,d\lambda = L/4\pi D_L^2`.

    If ``lambda_cut`` is given, the spectrum is multiplied by the UV-suppression factor of
    :func:`_log_cutoff_factor_cgs` at the rest-frame wavelength.

    Parameters
    ----------
    log_lambda : float, int, or ~numpy.ndarray
        Natural logarithm of observer-frame wavelength in cm.
    log_T : float, int, or ~numpy.ndarray
        Natural logarithm of temperature in K.
    log_R : float, int, or ~numpy.ndarray
        Natural logarithm of source radius in cm.
    log_D : float, int, or ~numpy.ndarray
        Natural logarithm of luminosity distance in cm.
    z : float, int, or ~numpy.ndarray, optional
        Redshift. Default is 0.
    lambda_cut : float, optional
        Rest-frame cutoff wavelength in cm below which the spectrum is suppressed. Default is 0 (no suppression).

    Returns
    -------
    float or ~numpy.ndarray
        Logarithm of the observed specific flux density :math:`f_\lambda` in
        :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,cm^{-1}}`.
    """
    log1pz = np.log1p(np.asarray(z))
    log_lambda_rest = np.asarray(log_lambda) - log1pz

    result = (
        _log_planck_flambda_cgs(log_lambda_rest, log_T)
        + np.log(np.pi)
        + 2.0 * (np.asarray(log_R) - np.asarray(log_D))
        - log1pz
        + _log_cutoff_factor_cgs(log_lambda_rest, log_T, lambda_cut)
    )

    return result if np.ndim(result) > 0 else float(result)


# --- Specific Luminosity --- #
def _log_specific_luminosity_fnu_cgs(
    log_nu: "_ArrayLike",
    log_T: "_ArrayLike",
    log_R: "_ArrayLike",
    lambda_cut: float = 0.0,
) -> "_ArrayLike":
    r"""Evaluate :math:`\ln L_\nu` for a spherical blackbody in CGS units.

    Uses

    .. math::

        L_\nu = 4\pi^2 R^2 B_\nu

    If ``lambda_cut`` is given, the spectrum is multiplied by the UV-suppression factor of
    :func:`_log_cutoff_factor_cgs`.

    Parameters
    ----------
    log_nu : float, int, or ~numpy.ndarray
        Natural logarithm of frequency in Hz.
    log_T : float, int, or ~numpy.ndarray
        Natural logarithm of temperature in K.
    log_R : float, int, or ~numpy.ndarray
        Natural logarithm of source radius in cm.
    lambda_cut : float, optional
        Cutoff wavelength in cm below which the spectrum is suppressed. Default is 0 (no suppression).

    Returns
    -------
    float or ~numpy.ndarray
        Logarithm of the specific luminosity :math:`L_\nu` in
        :math:`\mathrm{erg\,s^{-1}\,Hz^{-1}}`.
    """
    result = (
        _log_planck_fnu_cgs(log_nu, log_T)
        + np.log(4.0 * np.pi**2)
        + 2.0 * np.asarray(log_R)
        + _log_cutoff_factor_cgs(np.log(c_cgs) - np.asarray(log_nu), log_T, lambda_cut)
    )
    return result if np.ndim(result) > 0 else float(result)


def _log_specific_luminosity_flambda_cgs(
    log_lambda: "_ArrayLike",
    log_T: "_ArrayLike",
    log_R: "_ArrayLike",
    lambda_cut: float = 0.0,
) -> "_ArrayLike":
    r"""Evaluate :math:`\ln L_\lambda` for a spherical blackbody in CGS units.

    Uses

    .. math::

        L_\lambda = 4\pi^2 R^2 B_\lambda

    If ``lambda_cut`` is given, the spectrum is multiplied by the UV-suppression factor of
    :func:`_log_cutoff_factor_cgs`.

    Parameters
    ----------
    log_lambda : float, int, or ~numpy.ndarray
        Natural logarithm of wavelength in cm.
    log_T : float, int, or ~numpy.ndarray
        Natural logarithm of temperature in K.
    log_R : float, int, or ~numpy.ndarray
        Natural logarithm of source radius in cm.
    lambda_cut : float, optional
        Cutoff wavelength in cm below which the spectrum is suppressed. Default is 0 (no suppression).

    Returns
    -------
    float or ~numpy.ndarray
        Logarithm of the specific luminosity :math:`L_\lambda` in
        :math:`\mathrm{erg\,s^{-1}\,cm^{-1}}`.
    """
    result = (
        _log_planck_flambda_cgs(log_lambda, log_T)
        + np.log(4.0 * np.pi**2)
        + 2.0 * np.asarray(log_R)
        + _log_cutoff_factor_cgs(log_lambda, log_T, lambda_cut)
    )
    return result if np.ndim(result) > 0 else float(result)


# --- Bolometric Fluxes --- #
def _log_bol_rad_flux_cgs(log_T: "_ArrayLike") -> "_ArrayLike":
    r"""Evaluate :math:`\ln F` (bolometric radiative flux) in CGS units.

    This is the surface flux emitted by a blackbody:

    .. math::

        F = \sigma_{\rm SB} T^4

    Parameters
    ----------
    log_T : float, int, or ~numpy.ndarray
        Natural logarithm of temperature in K.

    Returns
    -------
    float or ~numpy.ndarray
        Logarithm of the bolometric radiative flux in
        :math:`\mathrm{erg\,s^{-1}\,cm^{-2}}`.
    """
    log_T = np.asarray(log_T)

    result = np.log(sigma_sb_cgs) + 4.0 * log_T

    return result if result.ndim > 0 else float(result)


def _log_bol_flux_cgs(
    log_T: "_ArrayLike",
    log_R: "_ArrayLike",
    log_D: "_ArrayLike",
    z: "_ArrayLike" = 0,
) -> "_ArrayLike":
    r"""Evaluate :math:`\ln f` (bolometric flux) for a spherical blackbody.

    For zero redshift:

    .. math::

        f = \sigma_{\rm SB} T^4 \left(\frac{R}{D}\right)^2

    For nonzero redshift (assuming luminosity distance):

    .. math::

        f_{\mathrm{obs}} =
        \frac{L}{4\pi D_L^2 (1+z)^2}

    Parameters
    ----------
    log_T : float, int, or ~numpy.ndarray
        Natural logarithm of temperature in K.
    log_R : float, int, or ~numpy.ndarray
        Natural logarithm of source radius in cm.
    log_D : float, int, or ~numpy.ndarray
        Natural logarithm of luminosity distance in cm.
    z : float, int, or ~numpy.ndarray, optional
        Redshift. Default is 0.

    Returns
    -------
    float or ~numpy.ndarray
        Logarithm of the observed bolometric flux in
        :math:`\mathrm{erg\,s^{-1}\,cm^{-2}}`.
    """
    log_T = np.asarray(log_T)
    log_R = np.asarray(log_R)
    log_D = np.asarray(log_D)
    z = np.asarray(z)

    log1pz = np.log1p(z)

    result = np.log(sigma_sb_cgs) + 4.0 * log_T + 2.0 * (log_R - log_D) - 2.0 * log1pz

    return result if result.ndim > 0 else float(result)


# --- Wein's Law --- #


def _log_peak_frequency(log_T: "_ArrayLike") -> "_ArrayLike":
    r"""Evaluate :math:`\ln \nu_{\rm peak}` from Wien's displacement law.

    The peak frequency of the Planck function :math:`B_\nu` is given by

    .. math::

        \nu_{\rm peak} = b_\nu \, T,

    where :math:`b_\nu` is the Wien frequency constant.

    Parameters
    ----------
    log_T : float, int, or ~numpy.ndarray
        Natural logarithm of temperature in K.

    Returns
    -------
    float or ~numpy.ndarray
        Natural logarithm of the peak frequency in Hz.

    Notes
    -----
    This corresponds to the maximum of :math:`B_\nu`. The peak of
    :math:`B_\lambda` occurs at a different location.
    """
    log_T = np.asarray(log_T)

    result = np.log(_WIEN_B_FREQ_CGS) + log_T

    return result if result.ndim > 0 else float(result)


def _log_peak_wavelength(log_T: "_ArrayLike") -> "_ArrayLike":
    r"""Evaluate :math:`\ln \lambda_{\rm peak}` from Wien's displacement law.

    The peak wavelength of the Planck function :math:`B_\lambda` is given by

    .. math::

        \lambda_{\rm peak} = \frac{b_\lambda}{T},

    where :math:`b_\lambda` is the Wien wavelength constant.

    Parameters
    ----------
    log_T : float, int, or ~numpy.ndarray
        Natural logarithm of temperature in K.

    Returns
    -------
    float or ~numpy.ndarray
        Natural logarithm of the peak wavelength in cm.

    Notes
    -----
    This corresponds to the maximum of :math:`B_\lambda`. The peak of
    :math:`B_\nu` occurs at a different location.
    """
    log_T = np.asarray(log_T)

    result = np.log(_WIEN_B_WAV_CGS) - log_T

    return result if result.ndim > 0 else float(result)


# --- Photospheric Radius / Temperature / Luminosity --- #


def _log_photospheric_radius(
    log_L: "_ArrayLike",
    log_T: "_ArrayLike",
) -> "_ArrayLike":
    r"""Compute :math:`\ln R` from luminosity and temperature.

    This inverts the Stefan–Boltzmann relation

    .. math::

        L = 4\pi R^2 \sigma_{\rm SB} T^4,

    to solve for the photospheric radius.

    Parameters
    ----------
    log_L : float, int, or ~numpy.ndarray
        Natural logarithm of bolometric luminosity in
        :math:`\mathrm{erg\,s^{-1}}`.
    log_T : float, int, or ~numpy.ndarray
        Natural logarithm of temperature in K.

    Returns
    -------
    float or ~numpy.ndarray
        Natural logarithm of the photospheric radius in cm.
    """
    log_L = np.asarray(log_L)
    log_T = np.asarray(log_T)

    result = 0.5 * (log_L - _LOG_4PI_SIGMA - 4.0 * log_T)

    return result if result.ndim > 0 else float(result)


def _log_photospheric_temperature(
    log_L: "_ArrayLike",
    log_R: "_ArrayLike",
) -> "_ArrayLike":
    r"""Compute :math:`\ln T` from luminosity and radius.

    This inverts the Stefan–Boltzmann relation

    .. math::

        L = 4\pi R^2 \sigma_{\rm SB} T^4,

    to solve for the effective temperature.

    Parameters
    ----------
    log_L : float, int, or ~numpy.ndarray
        Natural logarithm of bolometric luminosity in
        :math:`\mathrm{erg\,s^{-1}}`.
    log_R : float, int, or ~numpy.ndarray
        Natural logarithm of photospheric radius in cm.

    Returns
    -------
    float or ~numpy.ndarray
        Natural logarithm of the effective temperature in K.
    """
    log_L = np.asarray(log_L)
    log_R = np.asarray(log_R)

    result = 0.25 * (log_L - _LOG_4PI_SIGMA - 2.0 * log_R)

    return result if result.ndim > 0 else float(result)


def _log_photospheric_radius_from_flux(
    log_F: "_ArrayLike",
    log_T: "_ArrayLike",
    log_D: "_ArrayLike",
    z: "_ArrayLike" = 0,
) -> "_ArrayLike":
    r"""Compute :math:`\ln R` from observed flux and temperature.

    This uses the relation

    .. math::

        f = \sigma_{\rm SB} T^4 \left(\frac{R}{D}\right)^2 (1+z)^{-2},

    where :math:`f` is the observed bolometric flux and :math:`D` is the
    luminosity distance.

    Parameters
    ----------
    log_F : float, int, or ~numpy.ndarray
        Natural logarithm of observed bolometric flux in
        :math:`\mathrm{erg\,s^{-1}\,cm^{-2}}`.
    log_T : float, int, or ~numpy.ndarray
        Natural logarithm of temperature in K.
    log_D : float, int, or ~numpy.ndarray
        Natural logarithm of luminosity distance in cm.
    z : float, int, or ~numpy.ndarray, optional
        Redshift. Default is 0.

    Returns
    -------
    float or ~numpy.ndarray
        Natural logarithm of the photospheric radius in cm.

    Notes
    -----
    Assumes that `D` is the luminosity distance. For nearby sources
    (:math:`z \approx 0`), the redshift correction reduces to unity.
    """
    log_F = np.asarray(log_F)
    log_T = np.asarray(log_T)
    log_D = np.asarray(log_D)
    z = np.asarray(z)

    log1pz = np.log1p(z)

    result = log_D + 0.5 * (log_F - np.log(sigma_sb_cgs) - 4.0 * log_T + 2.0 * log1pz)

    return result if result.ndim > 0 else float(result)


def _log_photospheric_temperature_from_flux(
    log_F: "_ArrayLike",
    log_R: "_ArrayLike",
    log_D: "_ArrayLike",
    z: "_ArrayLike" = 0,
) -> "_ArrayLike":
    r"""Compute :math:`\ln T` from observed flux and radius.

    This uses the relation

    .. math::

        f = \sigma_{\rm SB} T^4 \left(\frac{R}{D}\right)^2 (1+z)^{-2},

    where :math:`f` is the observed bolometric flux and :math:`D` is the
    luminosity distance.

    Parameters
    ----------
    log_F : float, int, or ~numpy.ndarray
        Natural logarithm of observed bolometric flux in
        :math:`\mathrm{erg\,s^{-1}\,cm^{-2}}`.
    log_R : float, int, or ~numpy.ndarray
        Natural logarithm of photospheric radius in cm.
    log_D : float, int, or ~numpy.ndarray
        Natural logarithm of luminosity distance in cm.
    z : float, int, or ~numpy.ndarray, optional
        Redshift. Default is 0.

    Returns
    -------
    float or ~numpy.ndarray
        Natural logarithm of the effective temperature in K.

    Notes
    -----
    Assumes that `D` is the luminosity distance. For nearby sources
    (:math:`z \approx 0`), the redshift correction reduces to unity.
    """
    log_F = np.asarray(log_F)
    log_R = np.asarray(log_R)
    log_D = np.asarray(log_D)
    z = np.asarray(z)

    log1pz = np.log1p(z)

    result = 0.25 * (log_F - np.log(sigma_sb_cgs) - 2.0 * (log_R - log_D) + 2.0 * log1pz)

    return result if result.ndim > 0 else float(result)


# ===========================================================================
# Public (unit-aware) Interface
# ===========================================================================


def _resolve_nu_lam(nu, lam):
    """Return (log_x_cgs, mode) where mode is 'nu' or 'lam'."""
    if (nu is None) == (lam is None):
        raise ValueError("Exactly one of nu or lam must be provided.")
    if nu is not None:
        return np.log(ensure_in_units(nu, u.Hz)), "nu"
    return np.log(ensure_in_units(lam, u.cm)), "lam"


def _resolve_lambda_cut(lambda_cut) -> float:
    """Cutoff wavelength in cm from a Quantity, a bare number (cm), or ``None`` (no cutoff: 0)."""
    return 0.0 if lambda_cut is None else float(ensure_in_units(lambda_cut, u.cm))


def planck_B(
    T: "_UnitBearingArrayLike",
    *,
    nu: "_UnitBearingArrayLike" = None,
    lam: "_UnitBearingArrayLike" = None,
) -> u.Quantity:
    r"""Planck spectral radiance :math:`B_\nu` or :math:`B_\lambda`.

    Parameters
    ----------
    T : ~astropy.units.Quantity
        Temperature.  Convertible to K.
    nu : ~astropy.units.Quantity, optional
        Frequency.  Convertible to Hz.  Mutually exclusive with *lam*.
    lam : ~astropy.units.Quantity, optional
        Wavelength.  Convertible to cm.  Mutually exclusive with *nu*.

    Returns
    -------
    ~astropy.units.Quantity
        :math:`B_\nu` in :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}\,sr^{-1}}` when *nu* is given,
        or :math:`B_\lambda` in :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,cm^{-1}\,sr^{-1}}` when *lam* is given.
    """
    log_x, mode = _resolve_nu_lam(nu, lam)
    log_T = np.log(ensure_in_units(T, u.K))
    if mode == "nu":
        return np.exp(_log_planck_fnu_cgs(log_x, log_T)) * u.Unit("erg s-1 cm-2 Hz-1 sr-1")
    return np.exp(_log_planck_flambda_cgs(log_x, log_T)) * u.Unit("erg s-1 cm-2 cm-1 sr-1")


def planck_surface_flux(
    T: "_UnitBearingArrayLike",
    *,
    nu: "_UnitBearingArrayLike" = None,
    lam: "_UnitBearingArrayLike" = None,
) -> u.Quantity:
    r"""Outward specific flux :math:`\pi B_\nu` or :math:`\pi B_\lambda` per unit area of a blackbody surface.

    Parameters
    ----------
    T : ~astropy.units.Quantity
        Temperature.  Convertible to K.
    nu : ~astropy.units.Quantity, optional
        Frequency.  Convertible to Hz.  Mutually exclusive with *lam*.
    lam : ~astropy.units.Quantity, optional
        Wavelength.  Convertible to cm.  Mutually exclusive with *nu*.

    Returns
    -------
    ~astropy.units.Quantity
        :math:`F_\nu` in :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}}` when *nu* is given,
        or :math:`F_\lambda` in :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,cm^{-1}}` when *lam* is given.
    """
    log_x, mode = _resolve_nu_lam(nu, lam)
    log_T = np.log(ensure_in_units(T, u.K))
    if mode == "nu":
        return np.exp(_log_specific_rad_flux_fnu_cgs(log_x, log_T)) * u.Unit("erg s-1 cm-2 Hz-1")
    return np.exp(_log_specific_rad_flux_flambda_cgs(log_x, log_T)) * u.Unit("erg s-1 cm-2 cm-1")


def planck_flux(
    T: "_UnitBearingArrayLike",
    R: "_UnitBearingArrayLike",
    *,
    nu: "_UnitBearingArrayLike" = None,
    lam: "_UnitBearingArrayLike" = None,
    redshift: float = None,
    luminosity_distance: "_UnitBearingArrayLike" = None,
    angular_diameter_distance: "_UnitBearingArrayLike" = None,
    proper_distance: "_UnitBearingArrayLike" = None,
    cosmology: cosmo.Cosmology = None,
    lambda_cut: "_UnitBearingScalarLike" = None,
) -> u.Quantity:
    r"""Observed specific flux density :math:`f_\nu` or :math:`f_\lambda` from a spherical blackbody.

    With ``lambda_cut`` the blackbody has a suppressed UV (see :func:`planck_cutoff_fraction` and
    :ref:`blackbody_cutoff`); the bolometric flux is the same as that of the plain blackbody.

    Parameters
    ----------
    T : ~astropy.units.Quantity
        Blackbody temperature.  Convertible to K.
    R : ~astropy.units.Quantity
        Source radius.  Convertible to cm.
    nu : ~astropy.units.Quantity, optional
        Observer-frame frequency.  Convertible to Hz.  Mutually exclusive with *lam*.
    lam : ~astropy.units.Quantity, optional
        Observer-frame wavelength.  Convertible to cm.  Mutually exclusive with *nu*.
    redshift : float, optional
        Cosmological redshift of the source.
    luminosity_distance : ~astropy.units.Quantity, optional
        Luminosity distance to the source.
    angular_diameter_distance : ~astropy.units.Quantity, optional
        Angular diameter distance to the source.
    proper_distance : ~astropy.units.Quantity, optional
        Proper (comoving radial) distance to the source.
    cosmology : ~astropy.cosmology.Cosmology, optional
        Cosmology used to convert between redshift and distance measures.
        If not provided, the configured default cosmology is used.
    lambda_cut : ~astropy.units.Quantity, optional
        **Rest-frame** wavelength below which the spectrum is suppressed linearly, as
        :math:`\min(1, \lambda/\lambda_{\rm cut})`, and then renormalised to conserve the luminosity. Convertible to cm.
        If not provided (default), the spectrum is a plain blackbody.

    Returns
    -------
    ~astropy.units.Quantity
        :math:`f_\nu` in :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}}` when *nu* is given,
        or :math:`f_\lambda` in :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,cm^{-1}}` when *lam* is given.

    Notes
    -----
    Exactly one of *nu* or *lam* and exactly one of *redshift*, *luminosity_distance*,
    *angular_diameter_distance*, or *proper_distance* must be provided.
    """
    log_x, mode = _resolve_nu_lam(nu, lam)
    cosmology = get_cosmology(cosmology=cosmology)
    distances = resolve_cosmological_distances(
        redshift=redshift,
        luminosity_distance=luminosity_distance,
        angular_diameter_distance=angular_diameter_distance,
        proper_distance=proper_distance,
        cosmology=cosmology,
    )
    z = distances["redshift"]
    log_D = np.log(ensure_in_units(distances["luminosity_distance"], u.cm))
    log_T = np.log(ensure_in_units(T, u.K))
    log_R = np.log(ensure_in_units(R, u.cm))
    cut = _resolve_lambda_cut(lambda_cut)
    if mode == "nu":
        return np.exp(_log_specific_flux_fnu_cgs(log_x, log_T, log_R, log_D, z, cut)) * u.Unit("erg s-1 cm-2 Hz-1")
    return np.exp(_log_specific_flux_flambda_cgs(log_x, log_T, log_R, log_D, z, cut)) * u.Unit("erg s-1 cm-2 cm-1")


def planck_cutoff_fraction(T: "_UnitBearingArrayLike", lambda_cut: "_UnitBearingScalarLike") -> np.ndarray:
    r"""Fraction of a blackbody's bolometric flux that survives a linear UV suppression.

    The suppression multiplies the spectrum by :math:`\min(1, \lambda/\lambda_{\rm cut})`: it is 100% at
    the cutoff wavelength and falls linearly to zero at :math:`\lambda = 0`. The surviving fraction

    .. math::

        f(T) = \frac{\int B_\lambda \min(1, \lambda/\lambda_{\rm cut})\,d\lambda}{\int B_\lambda\,d\lambda}

    is evaluated exactly (see :ref:`blackbody_cutoff`). A spectrum that has been renormalised to conserve the
    luminosity is the suppressed spectrum divided by this fraction.

    Parameters
    ----------
    T : ~astropy.units.Quantity
        Blackbody temperature.  Convertible to K.
    lambda_cut : ~astropy.units.Quantity
        Cutoff wavelength.  Convertible to cm. Zero (or less) means no suppression.

    Returns
    -------
    ~numpy.ndarray
        The surviving fraction, between 0 and 1. It falls from 1 as the temperature rises: the hotter the
        blackbody, the more of its flux is in the suppressed UV. It is accurate to :math:`10^{-14}` for
        :math:`T \lesssim 3\times10^5` K at :math:`\lambda_{\rm cut} = 3000` Angstrom.

    Examples
    --------
    >>> import astropy.units as u
    >>> from trilobite.radiation.blackbody import (
    ...     planck_cutoff_fraction,
    ... )
    >>> planck_cutoff_fraction(
    ...     [8000, 15000, 30000] * u.K, 3000 * u.AA
    ... )
    array([0.97..., 0.81..., 0.52...])
    """
    return _cutoff_transmitted_fraction_cgs(ensure_in_units(T, u.K), _resolve_lambda_cut(lambda_cut))


def planck_luminosity(
    T: "_UnitBearingArrayLike",
    R: "_UnitBearingArrayLike",
    *,
    nu: "_UnitBearingArrayLike" = None,
    lam: "_UnitBearingArrayLike" = None,
    lambda_cut: "_UnitBearingScalarLike" = None,
) -> u.Quantity:
    r"""Specific luminosity :math:`L_\nu = 4\pi^2 R^2 B_\nu` or :math:`L_\lambda = 4\pi^2 R^2 B_\lambda`.

    With ``lambda_cut`` the blackbody has a suppressed UV (see :func:`planck_cutoff_fraction` and
    :ref:`blackbody_cutoff`); the bolometric luminosity is :math:`4\pi R^2 \sigma_{\rm SB} T^4` either way.

    Parameters
    ----------
    T : ~astropy.units.Quantity
        Temperature.  Convertible to K.
    R : ~astropy.units.Quantity
        Source radius.  Convertible to cm.
    nu : ~astropy.units.Quantity, optional
        Frequency.  Convertible to Hz.  Mutually exclusive with *lam*.
    lam : ~astropy.units.Quantity, optional
        Wavelength.  Convertible to cm.  Mutually exclusive with *nu*.
    lambda_cut : ~astropy.units.Quantity, optional
        Wavelength below which the spectrum is suppressed linearly, as :math:`\min(1, \lambda/\lambda_{\rm cut})`, and
        then renormalised to conserve the luminosity. Convertible to cm. If not provided (default), the spectrum is a
        plain blackbody.

    Returns
    -------
    ~astropy.units.Quantity
        :math:`L_\nu` in :math:`\mathrm{erg\,s^{-1}\,Hz^{-1}}` when *nu* is given,
        or :math:`L_\lambda` in :math:`\mathrm{erg\,s^{-1}\,cm^{-1}}` when *lam* is given.
    """
    log_x, mode = _resolve_nu_lam(nu, lam)
    log_T = np.log(ensure_in_units(T, u.K))
    log_R = np.log(ensure_in_units(R, u.cm))
    cut = _resolve_lambda_cut(lambda_cut)
    if mode == "nu":
        return np.exp(_log_specific_luminosity_fnu_cgs(log_x, log_T, log_R, cut)) * u.Unit("erg s-1 Hz-1")
    return np.exp(_log_specific_luminosity_flambda_cgs(log_x, log_T, log_R, cut)) * u.Unit("erg s-1 cm-1")


def stefan_boltzmann_flux(T: "_UnitBearingArrayLike") -> u.Quantity:
    r"""Bolometric radiative flux :math:`F = \sigma_\mathrm{SB} T^4` per unit area of a blackbody surface.

    Parameters
    ----------
    T : ~astropy.units.Quantity
        Temperature.  Convertible to K.

    Returns
    -------
    ~astropy.units.Quantity
        :math:`F` in :math:`\mathrm{erg\,s^{-1}\,cm^{-2}}`.
    """
    T_cgs = ensure_in_units(T, u.K)
    return np.exp(_log_bol_rad_flux_cgs(np.log(T_cgs))) * u.Unit("erg s-1 cm-2")


def stefan_boltzmann_observed_flux(
    T: "_UnitBearingArrayLike",
    R: "_UnitBearingArrayLike",
    *,
    redshift: float = None,
    luminosity_distance: "_UnitBearingArrayLike" = None,
    angular_diameter_distance: "_UnitBearingArrayLike" = None,
    proper_distance: "_UnitBearingArrayLike" = None,
    cosmology: cosmo.Cosmology = None,
) -> u.Quantity:
    r"""Observed bolometric flux from a spherical blackbody.

    Parameters
    ----------
    T : ~astropy.units.Quantity
        Blackbody temperature.  Convertible to K.
    R : ~astropy.units.Quantity
        Source radius.  Convertible to cm.
    redshift : float, optional
        Cosmological redshift of the source.
    luminosity_distance : ~astropy.units.Quantity, optional
        Luminosity distance to the source.
    angular_diameter_distance : ~astropy.units.Quantity, optional
        Angular diameter distance to the source.
    proper_distance : ~astropy.units.Quantity, optional
        Proper (comoving radial) distance to the source.
    cosmology : ~astropy.cosmology.Cosmology, optional
        Cosmology used to convert between redshift and distance measures.
        If not provided, the configured default cosmology is used.

    Returns
    -------
    ~astropy.units.Quantity
        Observed bolometric flux in :math:`\mathrm{erg\,s^{-1}\,cm^{-2}}`.

    Notes
    -----
    Exactly one of *redshift*, *luminosity_distance*, *angular_diameter_distance*,
    or *proper_distance* must be provided.
    """
    cosmology = get_cosmology(cosmology=cosmology)
    distances = resolve_cosmological_distances(
        redshift=redshift,
        luminosity_distance=luminosity_distance,
        angular_diameter_distance=angular_diameter_distance,
        proper_distance=proper_distance,
        cosmology=cosmology,
    )
    z = distances["redshift"]
    log_D = np.log(ensure_in_units(distances["luminosity_distance"], u.cm))
    log_T = np.log(ensure_in_units(T, u.K))
    log_R = np.log(ensure_in_units(R, u.cm))
    return np.exp(_log_bol_flux_cgs(log_T, log_R, log_D, z)) * u.Unit("erg s-1 cm-2")


def wien_peak_frequency(T: "_UnitBearingArrayLike") -> u.Quantity:
    r"""Peak frequency :math:`\nu_\mathrm{peak}` from Wien's displacement law.

    Parameters
    ----------
    T : ~astropy.units.Quantity
        Temperature.  Convertible to K.

    Returns
    -------
    ~astropy.units.Quantity
        :math:`\nu_\mathrm{peak}` in Hz.
    """
    T_cgs = ensure_in_units(T, u.K)
    return np.exp(_log_peak_frequency(np.log(T_cgs))) * u.Hz


def wien_peak_wavelength(T: "_UnitBearingArrayLike") -> u.Quantity:
    r"""Peak wavelength :math:`\lambda_\mathrm{peak}` from Wien's displacement law.

    Parameters
    ----------
    T : ~astropy.units.Quantity
        Temperature.  Convertible to K.

    Returns
    -------
    ~astropy.units.Quantity
        :math:`\lambda_\mathrm{peak}` in cm.
    """
    T_cgs = ensure_in_units(T, u.K)
    return np.exp(_log_peak_wavelength(np.log(T_cgs))) * u.cm


def photospheric_radius(
    L: "_UnitBearingArrayLike",
    T: "_UnitBearingArrayLike",
) -> u.Quantity:
    r"""Photospheric radius from bolometric luminosity and effective temperature.

    Inverts :math:`L = 4\pi R^2 \sigma_\mathrm{SB} T^4`.

    Parameters
    ----------
    L : ~astropy.units.Quantity
        Bolometric luminosity.  Convertible to :math:`\mathrm{erg\,s^{-1}}`.
    T : ~astropy.units.Quantity
        Effective temperature.  Convertible to K.

    Returns
    -------
    ~astropy.units.Quantity
        Photospheric radius in cm.
    """
    log_L = np.log(ensure_in_units(L, u.Unit("erg s-1")))
    log_T = np.log(ensure_in_units(T, u.K))
    return np.exp(_log_photospheric_radius(log_L, log_T)) * u.cm


def photospheric_temperature(
    L: "_UnitBearingArrayLike",
    R: "_UnitBearingArrayLike",
) -> u.Quantity:
    r"""Effective temperature from bolometric luminosity and photospheric radius.

    Inverts :math:`L = 4\pi R^2 \sigma_\mathrm{SB} T^4`.

    Parameters
    ----------
    L : ~astropy.units.Quantity
        Bolometric luminosity.  Convertible to :math:`\mathrm{erg\,s^{-1}}`.
    R : ~astropy.units.Quantity
        Photospheric radius.  Convertible to cm.

    Returns
    -------
    ~astropy.units.Quantity
        Effective temperature in K.
    """
    log_L = np.log(ensure_in_units(L, u.Unit("erg s-1")))
    log_R = np.log(ensure_in_units(R, u.cm))
    return np.exp(_log_photospheric_temperature(log_L, log_R)) * u.K


def photospheric_radius_from_flux(
    F: "_UnitBearingArrayLike",
    T: "_UnitBearingArrayLike",
    *,
    redshift: float = None,
    luminosity_distance: "_UnitBearingArrayLike" = None,
    angular_diameter_distance: "_UnitBearingArrayLike" = None,
    proper_distance: "_UnitBearingArrayLike" = None,
    cosmology: cosmo.Cosmology = None,
) -> u.Quantity:
    r"""Photospheric radius inferred from observed bolometric flux and temperature.

    Parameters
    ----------
    F : ~astropy.units.Quantity
        Observed bolometric flux.  Convertible to :math:`\mathrm{erg\,s^{-1}\,cm^{-2}}`.
    T : ~astropy.units.Quantity
        Effective temperature.  Convertible to K.
    redshift : float, optional
        Cosmological redshift of the source.
    luminosity_distance : ~astropy.units.Quantity, optional
        Luminosity distance to the source.
    angular_diameter_distance : ~astropy.units.Quantity, optional
        Angular diameter distance to the source.
    proper_distance : ~astropy.units.Quantity, optional
        Proper (comoving radial) distance to the source.
    cosmology : ~astropy.cosmology.Cosmology, optional
        Cosmology used to convert between redshift and distance measures.
        If not provided, the configured default cosmology is used.

    Returns
    -------
    ~astropy.units.Quantity
        Photospheric radius in cm.

    Notes
    -----
    Exactly one of *redshift*, *luminosity_distance*, *angular_diameter_distance*,
    or *proper_distance* must be provided.
    """
    cosmology = get_cosmology(cosmology=cosmology)
    distances = resolve_cosmological_distances(
        redshift=redshift,
        luminosity_distance=luminosity_distance,
        angular_diameter_distance=angular_diameter_distance,
        proper_distance=proper_distance,
        cosmology=cosmology,
    )
    z = distances["redshift"]
    log_D = np.log(ensure_in_units(distances["luminosity_distance"], u.cm))
    log_F = np.log(ensure_in_units(F, u.Unit("erg s-1 cm-2")))
    log_T = np.log(ensure_in_units(T, u.K))
    return np.exp(_log_photospheric_radius_from_flux(log_F, log_T, log_D, z)) * u.cm


def photospheric_temperature_from_flux(
    F: "_UnitBearingArrayLike",
    R: "_UnitBearingArrayLike",
    *,
    redshift: float = None,
    luminosity_distance: "_UnitBearingArrayLike" = None,
    angular_diameter_distance: "_UnitBearingArrayLike" = None,
    proper_distance: "_UnitBearingArrayLike" = None,
    cosmology: cosmo.Cosmology = None,
) -> u.Quantity:
    r"""Effective temperature inferred from observed bolometric flux and photospheric radius.

    Parameters
    ----------
    F : ~astropy.units.Quantity
        Observed bolometric flux.  Convertible to :math:`\mathrm{erg\,s^{-1}\,cm^{-2}}`.
    R : ~astropy.units.Quantity
        Photospheric radius.  Convertible to cm.
    redshift : float, optional
        Cosmological redshift of the source.
    luminosity_distance : ~astropy.units.Quantity, optional
        Luminosity distance to the source.
    angular_diameter_distance : ~astropy.units.Quantity, optional
        Angular diameter distance to the source.
    proper_distance : ~astropy.units.Quantity, optional
        Proper (comoving radial) distance to the source.
    cosmology : ~astropy.cosmology.Cosmology, optional
        Cosmology used to convert between redshift and distance measures.
        If not provided, the configured default cosmology is used.

    Returns
    -------
    ~astropy.units.Quantity
        Effective temperature in K.

    Notes
    -----
    Exactly one of *redshift*, *luminosity_distance*, *angular_diameter_distance*,
    or *proper_distance* must be provided.
    """
    cosmology = get_cosmology(cosmology=cosmology)
    distances = resolve_cosmological_distances(
        redshift=redshift,
        luminosity_distance=luminosity_distance,
        angular_diameter_distance=angular_diameter_distance,
        proper_distance=proper_distance,
        cosmology=cosmology,
    )
    z = distances["redshift"]
    log_D = np.log(ensure_in_units(distances["luminosity_distance"], u.cm))
    log_F = np.log(ensure_in_units(F, u.Unit("erg s-1 cm-2")))
    log_R = np.log(ensure_in_units(R, u.cm))
    return np.exp(_log_photospheric_temperature_from_flux(log_F, log_R, log_D, z)) * u.K
