"""
Normalization functions for 1-zone synchrotron SEDs.

This module contains the low-level structure for the synchrotron normalization logic implemented
in :mod:`one_zone`.
"""

import numpy as np

from triceratops.radiation.synchrotron.core import _opt_compute_log_synch_frequency
from triceratops.radiation.synchrotron.microphysics import (
    _opt_normalize_BPL_from_magnetic_field,
    _opt_normalize_PL_from_magnetic_field,
)
from triceratops.radiation.synchrotron.utils import _log_chi_cgs, _log_chi_cgs_iso

from ._one_zone_ssa import (
    compute_ssa_frequencies_with_cooling,
    compute_ssa_frequencies_without_cooling,
    select_ssa_sed_regime_from_candidates_with_cooling,
    select_ssa_sed_regime_from_candidates_without_cooling,
)


# ---------------------------------------------------------
# Norm → Peak Correctors
# ---------------------------------------------------------
def _log_norm_to_log_peak_powerlaw_sbpl_sed(
    log_nu_m,
    log_F_norm,
    p,
    regime,
):
    r"""
    Convert SED normalization to peak flux for a pure power-law synchrotron spectrum.

    This case corresponds to a synchrotron spectrum with **no radiative cooling**
    and **no synchrotron self-absorption**. The spectrum therefore peaks at the
    injection frequency :math:`\nu_m`, and the normalization is defined at the
    peak itself.

    In this regime

    .. math::

        F_{\nu,\mathrm{peak}} = F_{\mathrm{norm}}.

    Parameters
    ----------
    log_nu_m : float or ndarray
        Natural logarithm of the injection frequency :math:`\nu_m`.
        Included for API consistency but unused in this regime.

    log_F_norm : float or ndarray
        Natural logarithm of the SED normalization.

    p : float or ndarray
        Electron power-law index :math:`p`. Unused in this regime.

    regime : int or ndarray
        Spectral regime identifier. Included for API consistency but
        ignored in this function.

    Returns
    -------
    float or ndarray
        Natural logarithm of the peak flux :math:`F_{\nu,\mathrm{peak}}`.
    """
    _ = np.asarray(log_nu_m)
    log_F_norm = np.asarray(log_F_norm)
    _ = np.asarray(p)
    _ = np.asarray(regime)

    log_F_peak = np.array(log_F_norm, copy=True)

    return log_F_peak


def _log_norm_to_log_peak_powerlaw_sbpl_sed_cool(
    log_nu_m,
    log_nu_c,
    log_F_norm,
    p,
    regime,
):
    r"""
    Convert SED normalization to peak flux for synchrotron spectra including radiative cooling but **without SSA**.

    In cooling spectra the optically thin peak occurs at either
    :math:`\nu_m` (slow cooling) or :math:`\nu_c` (fast cooling). The
    normalization used in this SED parameterization is defined directly
    at the peak frequency.

    Consequently

    .. math::

        F_{\nu,\mathrm{peak}} = F_{\mathrm{norm}}.

    Parameters
    ----------
    log_nu_m : float or ndarray
        Natural logarithm of the injection frequency :math:`\nu_m`.

    log_nu_c : float or ndarray
        Natural logarithm of the cooling frequency :math:`\nu_c`.

    log_F_norm : float or ndarray
        Natural logarithm of the SED normalization.

    p : float or ndarray
        Electron power-law index :math:`p`. Included for interface
        compatibility but unused in this conversion.

    regime : int or ndarray
        Cooling spectral regime identifier. Included for API consistency
        but ignored by this function.

    Returns
    -------
    float or ndarray
        Natural logarithm of the peak flux :math:`F_{\nu,\mathrm{peak}}`.
    """
    _ = np.asarray(log_nu_m)
    _ = np.asarray(log_nu_c)
    log_F_norm = np.asarray(log_F_norm)
    _ = np.asarray(p)
    _ = np.asarray(regime)

    log_F_peak = np.array(log_F_norm, copy=True)

    return log_F_peak


def _log_norm_to_log_peak_powerlaw_sbpl_sed_ssa(
    log_nu_m,
    log_nu_a,
    log_F_norm,
    p,
    regime,
):
    r"""
    Convert SED normalization to peak flux for synchrotron spectra including SSA but **without radiative cooling**.

    In SSA spectra the observed peak may occur at either the injection
    frequency :math:`\nu_m` or the self-absorption frequency :math:`\nu_a`,
    depending on their ordering.

    Parameters
    ----------
    log_nu_m : float or ndarray
        Natural logarithm of the injection frequency :math:`\nu_m`.

    log_nu_a : float or ndarray
        Natural logarithm of the synchrotron self-absorption frequency
        :math:`\nu_a`.

    log_F_norm : float or ndarray
        Natural logarithm of the optically thin normalization.

    p : float or ndarray
        Electron power-law index :math:`p`.

    regime : int or ndarray
        SSA spectral regime identifier

        * ``1`` — peak at :math:`\nu_m`
        * ``2`` — peak at :math:`\nu_a`

    Returns
    -------
    float or ndarray
        Natural logarithm of the peak flux.
    """
    log_nu_m = np.asarray(log_nu_m)
    log_nu_a = np.asarray(log_nu_a)
    log_F_norm = np.asarray(log_F_norm)
    p = np.asarray(p)
    regime = np.asarray(regime)

    log_F_peak = np.array(log_F_norm, copy=True)

    mask_2 = regime == "optically_thick"

    log_F_peak[mask_2] = log_F_norm[mask_2] - ((p[mask_2] - 1) / 2) * (log_nu_a[mask_2] - log_nu_m[mask_2])

    return log_F_peak


def _log_norm_to_log_peak_powerlaw_sbpl_sed_ssa_cool(
    log_nu_m,
    log_nu_c,
    log_nu_a,
    log_nu_max,
    log_F_norm,
    p,
):
    r"""
    Convert optically thin normalization to peak flux for synchrotron spectra including both SSA and radiative cooling.

    This routine determines the spectral regime from the ordering of the
    characteristic synchrotron frequencies and propagates the optically thin
    normalization to the spectral peak.

    The normalization :math:`F_{\rm norm}` corresponds to the optically thin
    flux normalization defined by the synchrotron emissivity. When the spectral
    peak occurs at a self-absorbed frequency, the peak flux is obtained by
    propagating the normalization along the appropriate power-law segment of
    the synchrotron spectrum.

    Parameters
    ----------
    log_nu_m : float or ndarray
        Natural logarithm of the injection frequency :math:`\nu_m`.

    log_nu_c : float or ndarray
        Natural logarithm of the cooling frequency :math:`\nu_c`.

    log_nu_a : float or ndarray
        Natural logarithm of the synchrotron self-absorption frequency
        :math:`\nu_a`.

    log_nu_max : float or ndarray
        Natural logarithm of the maximum synchrotron frequency
        :math:`\nu_{\max}`.

    log_F_norm : float or ndarray
        Natural logarithm of the optically thin normalization of the
        synchrotron spectrum.

    p : float or ndarray
        Electron energy power-law index :math:`p`.

    Returns
    -------
    float or ndarray
        Natural logarithm of the peak flux :math:`F_{\nu,\rm peak}`.

    Notes
    -----
    The spectral regime is determined from the ordering of the characteristic
    synchrotron frequencies :math:`\nu_a`, :math:`\nu_m`, :math:`\nu_c`, and
    :math:`\nu_{\max}`.

    The following spectral configurations are handled:

    * :math:`\nu_a < \nu_m < \nu_{\max}` and :math:`\nu_c > \nu_{\max}`
      No cooling with an optically thin peak.

    * :math:`\nu_m < \nu_a < \nu_{\max}` and :math:`\nu_c > \nu_{\max}`
      No cooling with a self-absorbed spectral peak.

    * :math:`\nu_a < \nu_m < \nu_c < \nu_{\max}`
      Slow cooling with an optically thin peak.

    * :math:`\nu_m < \nu_a < \nu_c < \nu_{\max}`
      Slow cooling with a self-absorbed peak.

    * :math:`\nu_a < \nu_c < \nu_m < \nu_{\max}`
      Fast cooling with an optically thin peak.

    * :math:`\nu_c < \nu_a < \nu_m < \nu_{\max}`
      Fast cooling with a self-absorbed peak.

    * :math:`\nu_m < \nu_a` and :math:`\nu_c < \nu_a < \nu_{\max}`
      The self-absorption frequency lies above both :math:`\nu_m`
      and :math:`\nu_c`, requiring propagation across two spectral
      segments.

    In optically thin regimes the normalization already corresponds to the
    peak flux. In self-absorbed regimes the peak flux is obtained by
    propagating the normalization along the appropriate synchrotron
    spectral slope.
    """
    log_nu_m = np.asarray(log_nu_m)
    log_nu_c = np.asarray(log_nu_c)
    log_nu_a = np.asarray(log_nu_a)
    log_nu_max = np.asarray(log_nu_max)
    log_F_norm = np.asarray(log_F_norm)
    p = np.asarray(p)

    log_F_peak = np.array(log_F_norm, copy=True)

    # ----------------------------------------
    # Masks for spectral orderings
    # ----------------------------------------
    # We construct masks for the cases that need to be propagated through. Because this only applies to
    # optically thick scenarios, we can ignore any orderings where the peak is optically thin (i.e. cases 1, 3, and 5).
    mask_2 = (log_nu_m < log_nu_a) & (log_nu_a < log_nu_max) & (log_nu_c > log_nu_max)  # thick, no cooling.
    mask_4 = (log_nu_m < log_nu_a) & (log_nu_a < log_nu_c) & (log_nu_c < log_nu_max)  # thick, slow cooling.
    mask_6 = (log_nu_c < log_nu_a) & (log_nu_a < log_nu_m) & (log_nu_m < log_nu_max)  # thick, fast cooling.
    mask_7 = (log_nu_m < log_nu_c) & (log_nu_c < log_nu_a) & (log_nu_a < log_nu_max)  # thick, fast, fully absorbed.
    mask_8 = (log_nu_c < log_nu_m) & (log_nu_m < log_nu_a) & (log_nu_a < log_nu_max)  # thick, slow, fully absorbed.

    # ----------------------------------------
    # Apply conversions
    # ----------------------------------------
    # for masks 1, 3, and 5, the peak is optically thin and we don't need a correction.
    # for masks 2, 4, and 6, the peak is self-absorbed and we need to propagate the normalization up to the peak.
    log_F_peak[mask_2] = log_F_norm[mask_2] - ((p[mask_2] - 1) / 2) * (log_nu_a[mask_2] - log_nu_m[mask_2])
    log_F_peak[mask_4] = log_F_norm[mask_4] - ((p[mask_4] - 1) / 2) * (log_nu_a[mask_4] - log_nu_m[mask_4])
    log_F_peak[mask_6] = log_F_norm[mask_6] - 0.5 * (log_nu_a[mask_6] - log_nu_c[mask_6])

    # In cases 7 and 8, we need to perform the double power-law propagation from the normalization to the peak.
    # This is actually identical between the two spectra, so we just need to apply the same correction to both masks.
    # See the documentation for a discussion of normalization for Spectrum 7 & 8.
    log_F_peak[mask_7 | mask_8] = (
        log_F_norm[mask_7 | mask_8]
        + ((p[mask_7 | mask_8] - 1) / 2) * log_nu_m[mask_7 | mask_8]
        + 0.5 * log_nu_c[mask_7 | mask_8]
        - (p[mask_7 | mask_8] / 2) * log_nu_a[mask_7 | mask_8]
    )

    return log_F_peak


# ---------------------------------------------------------
# Peak → Norm Correctors
# ---------------------------------------------------------
def _log_peak_to_log_norm_powerlaw_sbpl_sed(
    log_nu_m,
    log_F_peak,
    p,
    regime,
):
    r"""
    Convert peak flux to normalization for a pure power-law synchrotron spectrum.

    In this regime the normalization is defined at the spectral peak,
    therefore

    .. math::

        F_{\mathrm{norm}} = F_{\nu,\mathrm{peak}}.

    Parameters
    ----------
    log_nu_m : float or ndarray
        Natural logarithm of the injection frequency :math:`\nu_m`.
        Included for API consistency but unused.

    log_F_peak : float or ndarray
        Natural logarithm of the peak flux.

    p : float or ndarray
        Electron power-law index :math:`p`. Unused.

    regime : int or ndarray
        Spectral regime identifier. Ignored.

    Returns
    -------
    float or ndarray
        Natural logarithm of the SED normalization.
    """
    _ = np.asarray(log_nu_m)
    log_F_peak = np.asarray(log_F_peak)
    _ = np.asarray(p)
    _ = np.asarray(regime)

    log_F_norm = np.array(log_F_peak, copy=True)

    return log_F_norm


def _log_peak_to_log_norm_powerlaw_sbpl_sed_cool(
    log_nu_m,
    log_nu_c,
    log_F_peak,
    p,
    regime,
):
    r"""
    Convert peak flux to normalization for synchrotron spectra with radiative cooling but **without self-absorption**.

    In this regime the normalization is defined at the optically thin
    spectral peak, so

    .. math::

        F_{\mathrm{norm}} = F_{\nu,\mathrm{peak}}.

    Parameters
    ----------
    log_nu_m : float or ndarray
        Natural logarithm of the injection frequency :math:`\nu_m`.

    log_nu_c : float or ndarray
        Natural logarithm of the cooling frequency :math:`\nu_c`.

    log_F_peak : float or ndarray
        Natural logarithm of the peak flux.

    p : float or ndarray
        Electron power-law index :math:`p`. Unused.

    regime : int or ndarray
        Cooling regime identifier. Ignored.

    Returns
    -------
    float or ndarray
        Natural logarithm of the SED normalization.
    """
    _ = np.asarray(log_nu_m)
    _ = np.asarray(log_nu_c)
    log_F_peak = np.asarray(log_F_peak)
    _ = np.asarray(p)
    _ = np.asarray(regime)

    log_F_norm = np.array(log_F_peak, copy=True)

    return log_F_norm


def _log_peak_to_log_norm_powerlaw_sbpl_sed_ssa(
    log_nu_m,
    log_nu_a,
    log_F_peak,
    p,
    regime,
):
    r"""
    Convert peak flux to normalization for synchrotron spectra including SSA but **without radiative cooling**.

    Parameters
    ----------
    log_nu_m : float or ndarray
        Natural logarithm of the injection frequency :math:`\nu_m`.

    log_nu_a : float or ndarray
        Natural logarithm of the self-absorption frequency :math:`\nu_a`.

    log_F_peak : float or ndarray
        Natural logarithm of the peak flux.

    p : float or ndarray
        Electron power-law index :math:`p`.

    regime : int or ndarray
        SSA spectral regime identifier.

        * ``1`` — peak at :math:`\nu_m`
        * ``2`` — peak at :math:`\nu_a`

    Returns
    -------
    float or ndarray
        Natural logarithm of the SED normalization.
    """
    log_nu_m = np.asarray(log_nu_m)
    log_nu_a = np.asarray(log_nu_a)
    log_F_peak = np.asarray(log_F_peak)
    p = np.asarray(p)
    regime = np.asarray(regime)

    log_F_norm = np.array(log_F_peak, copy=True)

    mask_2 = regime == "optically_thick"
    log_F_norm[mask_2] = log_F_peak[mask_2] + ((p[mask_2] - 1) / 2) * (log_nu_a[mask_2] - log_nu_m[mask_2])

    return log_F_norm


def _log_peak_to_log_norm_powerlaw_sbpl_sed_ssa_cool(
    log_nu_m,
    log_nu_c,
    log_nu_a,
    log_nu_max,
    log_F_peak,
    p,
):
    r"""
    Convert peak flux to optically thin normalization for synchrotron spectra including both SSA and radiative cooling.

    This function is the inverse of
    :func:`_log_norm_to_log_peak_powerlaw_sbpl_sed_ssa_cool`.

    Parameters
    ----------
    log_nu_m : float or ndarray
        Natural logarithm of the injection frequency :math:`\nu_m`.

    log_nu_c : float or ndarray
        Natural logarithm of the cooling frequency :math:`\nu_c`.

    log_nu_a : float or ndarray
        Natural logarithm of the self-absorption frequency :math:`\nu_a`.

    log_nu_max : float or ndarray
        Natural logarithm of the maximum synchrotron frequency.

    log_F_peak : float or ndarray
        Natural logarithm of the spectral peak flux.

    p : float or ndarray
        Electron power-law index :math:`p`.

    Returns
    -------
    float or ndarray
        Natural logarithm of the optically thin normalization.
    """
    log_nu_m = np.asarray(log_nu_m)
    log_nu_c = np.asarray(log_nu_c)
    log_nu_a = np.asarray(log_nu_a)
    log_nu_max = np.asarray(log_nu_max)
    log_F_peak = np.asarray(log_F_peak)
    p = np.asarray(p)

    log_F_norm = np.array(log_F_peak, copy=True)

    # ----------------------------------------
    # Spectral ordering masks
    # ----------------------------------------
    # We construct masks for the spectral orderings that require corrections. Because the normalization
    # is defined at the peak, we only need to apply corrections in cases where the peak is self-absorbed
    # (i.e. cases 2, 4, 6, 7, and 8). Cases 1, 3, and 5 have an optically thin peak
    # and therefore require no correction.
    mask_2 = (log_nu_m < log_nu_a) & (log_nu_a < log_nu_max) & (log_nu_c > log_nu_max)
    mask_4 = (log_nu_m < log_nu_a) & (log_nu_a < log_nu_c) & (log_nu_c < log_nu_max)
    mask_6 = (log_nu_c < log_nu_a) & (log_nu_a < log_nu_m) & (log_nu_m < log_nu_max)
    mask_7 = (log_nu_m < log_nu_c) & (log_nu_c < log_nu_a) & (log_nu_a < log_nu_max)
    mask_8 = (log_nu_c < log_nu_m) & (log_nu_m < log_nu_a) & (log_nu_a < log_nu_max)

    # Invert SSA corrections
    log_F_norm[mask_2] = log_F_peak[mask_2] + ((p[mask_2] - 1) / 2) * (log_nu_a[mask_2] - log_nu_m[mask_2])

    log_F_norm[mask_4] = log_F_peak[mask_4] + ((p[mask_4] - 1) / 2) * (log_nu_a[mask_4] - log_nu_m[mask_4])

    log_F_norm[mask_6] = log_F_peak[mask_6] + 0.5 * (log_nu_a[mask_6] - log_nu_c[mask_6])

    log_F_norm[mask_7 | mask_8] = (
        log_F_peak[mask_7 | mask_8]
        - ((p[mask_7 | mask_8] - 1) / 2) * log_nu_m[mask_7 | mask_8]
        - 0.5 * log_nu_c[mask_7]
        + (p[mask_7 | mask_8] / 2) * log_nu_a[mask_7 | mask_8]
    )

    return log_F_norm


# ---------------------------------------------------------
# Normalization Routines
# ---------------------------------------------------------
def _log_normalize_powerlaw_sbpl_sed(
    log_B: float,
    log_R: float,
    log_D_L: float,
    log_gamma_min: float,
    log_gamma_max: float = np.inf,
    p: float = 2.5,
    epsilon_E: float = 0.1,
    epsilon_B: float = 0.1,
    f_V: float = 1.0,
    alpha: float = 1.0,
    gamma_bulk: float = 1.0,
    redshift: float = 0.0,
    pitch_average: bool = False,
):
    r"""
    Compute the normalization of a single-zone synchrotron SED.

    This routine converts **physical model parameters** describing the emitting
    region into the phenomenological parameters used by the SED model
    (break frequencies and flux normalization).

    The normalization assumes a homogeneous emission region populated by
    a power-law electron distribution

    .. math::

        N(\gamma) = N_0 \gamma^{-p}

    with normalization determined through the equipartition parameters
    :math:`\epsilon_E` and :math:`\epsilon_B`.

    Parameters
    ----------
    log_B : float
        Natural logarithm of the magnetic field strength (Gauss).

    log_R : float
        Natural logarithm of the effective emitting radius (cm).

    log_D_L : float
        Natural logarithm of the luminosity distance (cm).

    log_gamma_min : float
        Natural logarithm of the minimum electron Lorentz factor
        :math:`\gamma_{\min}`.

    log_gamma_max : float, optional
        Natural logarithm of the maximum Lorentz factor
        :math:`\gamma_{\max}`.

    p : float, optional
        Power-law index of the electron energy distribution.

    epsilon_E : float, optional
        Fraction of post-shock energy carried by electrons.

    epsilon_B : float, optional
        Fraction of post-shock energy stored in magnetic fields.

    f_V : float, optional
        Filling factor of the emitting region.

    alpha : float, optional
        Electron pitch angle in radians.

    gamma_bulk : float, optional
        Bulk Lorentz factor of the emitting region.

    redshift : float, optional
        Source redshift.

    pitch_average : bool, optional
        If True, use pitch-angle averaged emissivity.

    Returns
    -------
    dict
        Dictionary containing the SED parameters

        - ``log_F_norm``  : logarithm of flux normalization
        - ``log_F_peak``  : logarithm of peak flux
        - ``log_nu_m``    : logarithm of injection frequency
        - ``log_nu_max``  : logarithm of maximum synchrotron frequency

    Notes
    -----
    This routine performs **only the normalization step**. It does not enforce
    spectral ordering constraints (cooling, self-absorption, etc.), which are
    handled elsewhere in the SED implementation.
    """
    # ---------------------------------------------------------
    # Pitch-angle treatment
    # ---------------------------------------------------------

    if pitch_average:
        log_chi = _log_chi_cgs_iso
        sin_alpha = 1.0
    else:
        log_chi = _log_chi_cgs
        sin_alpha = np.sin(alpha)

    log_sin_alpha = np.log(sin_alpha)

    # ---------------------------------------------------------
    # Doppler and redshift corrections
    # ---------------------------------------------------------

    beta_bulk = np.sqrt(1.0 - gamma_bulk**-2)
    delta_bulk = gamma_bulk * (1.0 + beta_bulk)

    log_delta = np.log(delta_bulk)
    log_1pz = np.log1p(redshift)

    log_nu_boost = log_delta - log_1pz
    log_flux_boost = 3.0 * log_delta - log_1pz

    # ---------------------------------------------------------
    # Characteristic synchrotron frequencies
    # ---------------------------------------------------------
    log_nu_m = (
        _opt_compute_log_synch_frequency(
            log_gamma_min,
            log_B,
            sin_alpha=sin_alpha,
            pitch_average=pitch_average,
        )
        + log_nu_boost
    )
    log_nu_max = (
        _opt_compute_log_synch_frequency(
            log_gamma_max,
            log_B,
            sin_alpha=sin_alpha,
            pitch_average=pitch_average,
        )
        + log_nu_boost
    )

    # ---------------------------------------------------------
    # Electron distribution normalization
    # ---------------------------------------------------------

    N0 = _opt_normalize_PL_from_magnetic_field(
        np.exp(log_B),
        p=p,
        epsilon_E=epsilon_E,
        epsilon_B=epsilon_B,
        gamma_min=np.exp(log_gamma_min),
        gamma_max=np.exp(log_gamma_max),
    )
    log_N0 = np.log(N0)

    # ---------------------------------------------------------
    # Emitting volume
    # ---------------------------------------------------------
    log_V = np.log(4.0 * np.pi / 3.0) + np.log(f_V) + 3.0 * log_R

    # ---------------------------------------------------------
    # Flux normalization
    # ---------------------------------------------------------
    log_F_norm = (
        log_chi + log_B + log_sin_alpha + log_N0 + log_V - 2.0 * log_D_L + (1.0 - p) * log_gamma_min + log_flux_boost
    )

    return {
        "log_F_norm": log_F_norm,
        "log_F_peak": log_F_norm,
        "log_nu_m": log_nu_m,
        "log_nu_max": log_nu_max,
        "log_nu_peak": log_nu_m,
    }


def _log_normalize_powerlaw_sbpl_sed_ssa_cool(
    log_B: float,
    log_R: float,
    log_D_L: float,
    log_D_A: float,
    log_gamma_min: float,
    log_gamma_c: float,
    log_gamma_max: float = np.inf,
    p: float = 2.5,
    f_V: float = 1.0,
    f_A: float = 1.0,
    epsilon_E: float = 0.1,
    epsilon_B: float = 0.1,
    alpha: float = 1.0,
    gamma_bulk: float = 1.0,
    redshift: float = 0.0,
    pitch_average: bool = False,
):
    r"""
    Compute the parameters for a one-zone synchrotron SED with SSA and optional radiative cooling.

    This routine constructs the characteristic synchrotron frequencies,
    determines the appropriate electron normalization from the assumed
    microphysical closure, computes the optically thin normalization, and
    then derives the candidate SSA frequencies and selects the physically
    consistent spectral regime.

    Parameters
    ----------
    log_B : float
        Natural logarithm of the magnetic field strength.
    log_R : float
        Natural logarithm of the characteristic source radius.
    log_D_L : float
        Natural logarithm of the luminosity distance.
    log_D_A : float
        Natural logarithm of the angular diameter distance.
    log_gamma_min : float
        Natural logarithm of the minimum electron Lorentz factor.
    log_gamma_c : float
        Natural logarithm of the cooling Lorentz factor.
    log_gamma_max : float, optional
        Natural logarithm of the maximum electron Lorentz factor.
    p : float, optional
        Electron power-law index.
    f_V : float, optional
        Volume filling factor.
    f_A: float, optional
        Area filling factor for SSA calculations.
    epsilon_E : float, optional
        Fraction of post-shock energy in electrons.
    epsilon_B : float, optional
        Fraction of post-shock energy in magnetic fields.
    alpha : float, optional
        Pitch angle in radians when ``pitch_average=False``.
    gamma_bulk : float, optional
        Bulk Lorentz factor of the emitting region.
    redshift : float, optional
        Source redshift.
    pitch_average : bool, optional
        If ``True``, assume isotropic pitch-angle averaging.

    Returns
    -------
    dict
        Dictionary containing the logarithmic phenomenological parameters:

        - ``log_F_norm``
        - ``log_nu_m``
        - ``log_nu_c``
        - ``log_nu_a``
        - ``log_nu_max``
        - ``regime``
    """
    # ---------------------------------------------------------
    # Pitch-angle treatment
    # ---------------------------------------------------------
    if pitch_average:
        log_chi = _log_chi_cgs_iso
        sin_alpha = 1.0
    else:
        log_chi = _log_chi_cgs
        sin_alpha = np.sin(alpha)

    log_sin_alpha = np.log(sin_alpha)

    # ---------------------------------------------------------
    # Doppler and redshift corrections
    # ---------------------------------------------------------
    beta_bulk = np.sqrt(1.0 - gamma_bulk**-2)
    delta_bulk = gamma_bulk * (1.0 + beta_bulk)

    log_delta_bulk = np.log(delta_bulk)
    log_1pz = np.log1p(redshift)

    log_nu_correction = log_delta_bulk - log_1pz
    log_flux_correction = 3.0 * log_delta_bulk - log_1pz

    # ---------------------------------------------------------
    # Characteristic synchrotron frequencies
    # ---------------------------------------------------------
    log_nu_m = (
        _opt_compute_log_synch_frequency(
            log_gamma_min,
            log_B,
            sin_alpha=sin_alpha,
            pitch_average=pitch_average,
        )
        + log_nu_correction
    )

    log_nu_c = (
        _opt_compute_log_synch_frequency(
            log_gamma_c,
            log_B,
            sin_alpha=sin_alpha,
            pitch_average=pitch_average,
        )
        + log_nu_correction
    )

    log_nu_max = (
        _opt_compute_log_synch_frequency(
            log_gamma_max,
            log_B,
            sin_alpha=sin_alpha,
            pitch_average=pitch_average,
        )
        + log_nu_correction
    )

    # ---------------------------------------------------------
    # Geometric factors
    # ---------------------------------------------------------
    log_V = np.log(4.0 * np.pi / 3.0) + 3.0 * log_R + np.log(f_V)
    log_A = np.log(np.pi) + 2.0 * log_R + np.log(f_A)
    log_Omega = log_A - 2.0 * log_D_A

    # ---------------------------------------------------------
    # Electron normalization and optically thin flux normalization
    # ---------------------------------------------------------
    is_fast_cooling = log_nu_c < log_nu_m
    is_cooling = log_nu_c < log_nu_max

    if is_fast_cooling:
        log_electron_norm = np.log(
            _opt_normalize_BPL_from_magnetic_field(
                np.exp(log_B),
                -2.0,
                -(p + 1),
                gamma_b=np.exp(log_gamma_min),
                gamma_min=np.exp(log_gamma_c),
                gamma_max=np.exp(log_gamma_max),
                epsilon_E=epsilon_E,
                epsilon_B=epsilon_B,
            )
        )
        log_F_norm = (
            log_chi
            + log_B
            + log_sin_alpha
            + log_electron_norm
            + log_nu_m
            - log_nu_c
            + log_gamma_c
            + log_V
            - 2.0 * log_D_L
            + log_flux_correction
        )

    elif is_cooling:
        # This is now the slow (but not no) cooling regime, so we use the BPL normalization with
        # the break at gamma_c and the low-energy slope fixed at -p.
        log_electron_norm = np.log(
            _opt_normalize_BPL_from_magnetic_field(
                np.exp(log_B),
                -p,
                -(p + 1),
                gamma_b=np.exp(log_gamma_c),
                gamma_min=np.exp(log_gamma_min),
                gamma_max=np.exp(log_gamma_max),
                epsilon_E=epsilon_E,
                epsilon_B=epsilon_B,
            )
        )
        log_F_norm = (
            log_chi
            + log_B
            + log_sin_alpha
            + log_electron_norm
            + (1.0 - p) * log_gamma_min
            + log_V
            + p * log_gamma_c  # Added 3/12 to fix the normalization in slow vs. no cooling.
            - 2.0 * log_D_L
            + log_flux_correction
        )

    else:
        # This is the no-cooling case.
        log_electron_norm = np.log(
            _opt_normalize_PL_from_magnetic_field(
                np.exp(log_B),
                p,
                gamma_min=np.exp(log_gamma_min),
                gamma_max=np.exp(log_gamma_max),
                epsilon_E=epsilon_E,
                epsilon_B=epsilon_B,
            )
        )
        log_F_norm = (
            log_chi
            + log_B
            + log_sin_alpha
            + log_electron_norm
            + (1.0 - p) * log_gamma_min
            + log_V
            - 2.0 * log_D_L
            + log_flux_correction
        )

    # ---------------------------------------------------------
    # SSA candidates and regime selection
    # ---------------------------------------------------------

    log_nu_ssa_candidates, cooling_regime = compute_ssa_frequencies_with_cooling(
        log_F_norm=log_F_norm,
        log_omega=log_Omega,
        log_nu_m=log_nu_m,
        log_nu_c=log_nu_c,
        log_nu_max=log_nu_max,
        log_gamma_m=log_gamma_min,
        p=p,
    )

    regime, log_nu_a = select_ssa_sed_regime_from_candidates_with_cooling(
        log_nu_ssa=log_nu_ssa_candidates,
        cooling_regime=cooling_regime,
        log_nu_m=log_nu_m,
        log_nu_c=log_nu_c,
        log_nu_max=log_nu_max,
    )

    # ---------------------------------------------------------
    # Return logarithmic normalization parameters
    # ---------------------------------------------------------
    log_nu_peak = np.amax([log_nu_a, np.amin([log_nu_m, log_nu_c])])

    return {
        "log_F_norm": log_F_norm,
        "log_F_peak": _log_norm_to_log_peak_powerlaw_sbpl_sed_ssa_cool(
            log_nu_m,
            log_nu_c,
            log_nu_a,
            log_nu_max,
            log_F_norm,
            p,
        ),
        "log_nu_m": log_nu_m,
        "log_nu_c": log_nu_c,
        "log_nu_a": log_nu_a,
        "log_nu_max": log_nu_max,
        "log_nu_peak": log_nu_peak,
        "log_omega": log_Omega,
        "regime": regime,
    }


def _log_normalize_powerlaw_sbpl_sed_cool(
    log_B: float,
    log_R: float,
    log_D_L: float,
    log_gamma_min: float,
    log_gamma_c: float,
    log_gamma_max: float = np.inf,
    p: float = 2.5,
    f_V: float = 1.0,
    epsilon_E: float = 0.1,
    epsilon_B: float = 0.1,
    alpha: float = 1.0,
    gamma_bulk: float = 1.0,
    redshift: float = 0.0,
    pitch_average: bool = False,
):
    r"""
    Compute logarithmic phenomenological parameters for an optically thin synchrotron SED with radiative cooling.

    This low-level routine maps physical source parameters onto the
    phenomenological parameters required to evaluate a one-zone synchrotron
    spectrum with a cooling break and no synchrotron self-absorption.

    All calculations are performed in natural logarithmic CGS units.

    Parameters
    ----------
    log_B : float
        Natural logarithm of the magnetic field strength in Gauss.

    log_R : float
        Natural logarithm of the effective emitting radius in cm.

    log_D_L : float
        Natural logarithm of the luminosity distance in cm.

    log_gamma_min : float
        Natural logarithm of the minimum electron Lorentz factor
        :math:`\gamma_m`.

    log_gamma_c : float
        Natural logarithm of the cooling Lorentz factor
        :math:`\gamma_c`.

    log_gamma_max : float, optional
        Natural logarithm of the maximum electron Lorentz factor
        :math:`\gamma_{\max}`.

    p : float, optional
        Power-law index of the injected electron energy distribution.

    f_V : float, optional
        Volume filling factor of the emitting region.

    epsilon_E : float, optional
        Fraction of post-shock internal energy carried by relativistic
        electrons.

    epsilon_B : float, optional
        Fraction of post-shock internal energy stored in magnetic fields.

    alpha : float, optional
        Electron pitch angle in radians. Ignored if
        ``pitch_average=True``.

    gamma_bulk : float, optional
        Bulk Lorentz factor of the emitting region.

    redshift : float, optional
        Source redshift.

    pitch_average : bool, optional
        If ``True``, use pitch-angle-averaged synchrotron emissivity.

    Returns
    -------
    dict
        Dictionary containing logarithmic phenomenological parameters:

        - ``log_F_norm`` : logarithm of the optically thin normalization
        - ``log_F_peak`` : logarithm of the spectral peak flux
        - ``log_nu_m`` : logarithm of the injection frequency
        - ``log_nu_c`` : logarithm of the cooling frequency
        - ``log_nu_max`` : logarithm of the maximum synchrotron frequency
        - ``log_nu_peak`` : logarithm of the peak frequency
        - ``regime`` : cooling regime identifier

    Notes
    -----
    The cooling regime is determined globally from the ordering of
    :math:`\nu_m`, :math:`\nu_c`, and :math:`\nu_{\max}`:

    - ``"fast_cooling"`` if :math:`\nu_c < \nu_m`
    - ``"slow_cooling"`` if :math:`\nu_m \le \nu_c < \nu_{\max}`
    - ``"no_cooling"`` if :math:`\nu_c \ge \nu_{\max}`

    In the fast-cooling regime the spectral peak occurs at :math:`\nu_c`.
    In the slow-cooling and no-cooling regimes it occurs at :math:`\nu_m`.
    """
    # ---------------------------------------------------------
    # Pitch-angle treatment
    # ---------------------------------------------------------
    sin_alpha = np.sin(alpha)
    log_sin_alpha = 0.0 if pitch_average else np.log(sin_alpha)
    log_chi = _log_chi_cgs_iso if pitch_average else _log_chi_cgs

    # ---------------------------------------------------------
    # Doppler and redshift corrections
    # ---------------------------------------------------------
    beta_bulk = np.sqrt(1.0 - gamma_bulk**-2)
    delta_bulk = gamma_bulk * (1.0 + beta_bulk)

    log_delta_bulk = np.log(delta_bulk)
    log_1pz = np.log1p(redshift)

    log_nu_correction = log_delta_bulk - log_1pz
    log_flux_correction = 3.0 * log_delta_bulk - log_1pz

    # ---------------------------------------------------------
    # Characteristic synchrotron frequencies
    # ---------------------------------------------------------
    log_nu_m = (
        _opt_compute_log_synch_frequency(
            log_gamma_min,
            log_B,
            sin_alpha=sin_alpha,
            pitch_average=pitch_average,
        )
        + log_nu_correction
    )

    log_nu_c = (
        _opt_compute_log_synch_frequency(
            log_gamma_c,
            log_B,
            sin_alpha=sin_alpha,
            pitch_average=pitch_average,
        )
        + log_nu_correction
    )

    log_nu_max = (
        _opt_compute_log_synch_frequency(
            log_gamma_max,
            log_B,
            sin_alpha=sin_alpha,
            pitch_average=pitch_average,
        )
        + log_nu_correction
    )

    # ---------------------------------------------------------
    # Geometric factor
    # ---------------------------------------------------------
    log_V = np.log(4.0 * np.pi / 3.0) + 3.0 * log_R + np.log(f_V)

    # ---------------------------------------------------------
    # Cooling regime and electron normalization
    # ---------------------------------------------------------
    if log_nu_c < log_nu_m:
        regime = "fast_cooling"

        log_electron_norm = np.log(
            _opt_normalize_BPL_from_magnetic_field(
                np.exp(log_B),
                -2.0,
                -(p + 1.0),
                gamma_b=np.exp(log_gamma_min),
                gamma_min=np.exp(log_gamma_c),
                gamma_max=np.exp(log_gamma_max),
                epsilon_E=epsilon_E,
                epsilon_B=epsilon_B,
            )
        )

        log_F_norm = (
            log_chi
            + log_B
            + log_sin_alpha
            + log_electron_norm
            + log_nu_m
            - log_nu_c
            + log_gamma_c
            + log_V
            - 2.0 * log_D_L
            + log_flux_correction
        )

    elif (log_nu_c < log_nu_max) & (log_nu_m <= log_nu_c):
        regime = "slow_cooling"

        log_electron_norm = np.log(
            _opt_normalize_BPL_from_magnetic_field(
                np.exp(log_B),
                -p,
                -(p + 1.0),
                gamma_b=np.exp(log_gamma_c),
                gamma_min=np.exp(log_gamma_min),
                gamma_max=np.exp(log_gamma_max),
                epsilon_E=epsilon_E,
                epsilon_B=epsilon_B,
            )
        )
        log_F_norm = (
            log_chi
            + log_B
            + log_sin_alpha
            + log_electron_norm
            + (1.0 - p) * log_gamma_min
            + log_V
            + p * log_gamma_c  # Added 3/12 to fix the normalization in slow vs. no cooling.
            - 2.0 * log_D_L
            + log_flux_correction
        )

    else:
        regime = "no_cooling"

        log_electron_norm = np.log(
            _opt_normalize_PL_from_magnetic_field(
                np.exp(log_B),
                p,
                gamma_min=np.exp(log_gamma_min),
                gamma_max=np.exp(log_gamma_max),
                epsilon_E=epsilon_E,
                epsilon_B=epsilon_B,
            )
        )

        log_F_norm = (
            log_chi
            + log_B
            + log_sin_alpha
            + log_electron_norm
            + (1.0 - p) * log_gamma_min
            + log_V
            - 2.0 * log_D_L
            + log_flux_correction
        )

    # ---------------------------------------------------------
    # Peak frequency
    # ---------------------------------------------------------
    log_nu_peak = np.minimum(log_nu_m, log_nu_c)

    # ---------------------------------------------------------
    # Return parameters
    # ---------------------------------------------------------
    return {
        "log_F_norm": log_F_norm,
        "log_F_peak": _log_norm_to_log_peak_powerlaw_sbpl_sed_cool(
            log_nu_m,
            log_nu_c,
            log_F_norm,
            p,
            regime,
        ),
        "log_nu_m": log_nu_m,
        "log_nu_c": log_nu_c,
        "log_nu_max": log_nu_max,
        "log_nu_peak": log_nu_peak,
        "regime": regime,
    }


def _log_normalize_powerlaw_sbpl_sed_ssa(
    log_B: float,
    log_R: float,
    log_D_L: float,
    log_D_A: float,
    log_gamma_min: float,
    log_gamma_max: float = np.inf,
    p: float = 2.5,
    f_V: float = 1.0,
    f_A: float = 1.0,
    epsilon_E: float = 0.1,
    epsilon_B: float = 0.1,
    alpha: float = 1.0,
    gamma_bulk: float = 1.0,
    redshift: float = 0.0,
    pitch_average: bool = False,
):
    r"""
    Compute logarithmic normalization parameters for the SSA synchrotron SED without radiative cooling.

    This routine converts **physical model parameters** describing a homogeneous
    synchrotron emission region into the **phenomenological SED parameters**
    required by the smoothed broken power-law spectrum.

    The procedure performs the following steps:

    1. Compute Doppler and cosmological corrections.
    2. Determine characteristic synchrotron frequencies (:math:`\nu_m`, :math:`\nu_{\max}`).
    3. Compute the optically thin flux normalization.
    4. Solve for the SSA frequency :math:`\nu_a`.
    5. Determine the global SSA spectral regime.

    All internal calculations are performed in **natural logarithmic space**.

    Parameters
    ----------
    log_B : float
        Natural logarithm of the magnetic field strength :math:`\log B`.

    log_R : float
        Natural logarithm of the emission region radius :math:`\log R`.

    log_D_L : float
        Natural logarithm of the luminosity distance :math:`\log D_L`.

    log_D_A : float
        Natural logarithm of the angular diameter distance :math:`\log D_A`.

    log_gamma_min : float
        Natural logarithm of the minimum electron Lorentz factor
        :math:`\log \gamma_m`.

    log_gamma_max : float, optional
        Natural logarithm of the maximum electron Lorentz factor.

    p : float, optional
        Electron power-law index.

    f_V : float, optional
        Volume filling factor.

    f_A : float, optional
        Angular filling factor.

    epsilon_E : float, optional
        Fraction of post-shock energy in relativistic electrons.

    epsilon_B : float, optional
        Fraction of post-shock energy in magnetic fields.

    alpha : float, optional
        Electron pitch angle in radians.

    gamma_bulk : float, optional
        Bulk Lorentz factor of the emitting region.

    redshift : float, optional
        Source redshift.

    pitch_average : bool, optional
        If ``True``, use isotropic pitch-angle averaged synchrotron constants.

    Returns
    -------
    dict
        Dictionary containing the logarithmic SED parameters

        - ``log_F_norm``
        - ``log_F_peak``
        - ``log_nu_m``
        - ``log_nu_a``
        - ``log_nu_max``
        - ``log_nu_peak``
        - ``regime``
    """
    # ---------------------------------------------------------
    # Pitch-angle treatment
    # ---------------------------------------------------------
    if pitch_average:
        log_chi = _log_chi_cgs_iso
        sin_alpha = 1.0
    else:
        log_chi = _log_chi_cgs
        sin_alpha = np.sin(alpha)

    log_sin_alpha = np.log(sin_alpha)

    # ---------------------------------------------------------
    # Doppler and cosmological corrections
    # ---------------------------------------------------------
    beta_bulk = np.sqrt(1.0 - gamma_bulk**-2)
    delta_bulk = gamma_bulk * (1.0 + beta_bulk)

    log_delta_bulk = np.log(delta_bulk)
    log_1pz = np.log1p(redshift)

    log_nu_correction = log_delta_bulk - log_1pz
    log_flux_correction = 3.0 * log_delta_bulk - log_1pz

    # ---------------------------------------------------------
    # Characteristic synchrotron frequencies
    # ---------------------------------------------------------
    log_nu_m = (
        _opt_compute_log_synch_frequency(
            log_gamma_min,
            log_B,
            sin_alpha=sin_alpha,
            pitch_average=pitch_average,
        )
        + log_nu_correction
    )

    log_nu_max = (
        _opt_compute_log_synch_frequency(
            log_gamma_max,
            log_B,
            sin_alpha=sin_alpha,
            pitch_average=pitch_average,
        )
        + log_nu_correction
    )

    # ---------------------------------------------------------
    # Geometric factors
    # ---------------------------------------------------------
    log_V = np.log(4.0 * np.pi / 3.0) + 3.0 * log_R + np.log(f_V)
    log_A = np.log(np.pi) + 2.0 * log_R + np.log(f_A)

    log_Omega = log_A - 2.0 * log_D_A

    # ---------------------------------------------------------
    # Electron distribution normalization
    # ---------------------------------------------------------
    N0 = _opt_normalize_PL_from_magnetic_field(
        np.exp(log_B),
        p=p,
        epsilon_E=epsilon_E,
        epsilon_B=epsilon_B,
        gamma_min=np.exp(log_gamma_min),
        gamma_max=np.exp(log_gamma_max),
    )

    log_N0 = np.log(N0)
    log_F_norm = (
        log_chi + log_B + log_sin_alpha + log_N0 + (1 - p) * log_gamma_min + log_V - 2.0 * log_D_L + log_flux_correction
    )

    # ---------------------------------------------------------
    # SSA candidates and regime selection
    # ---------------------------------------------------------
    log_nu_ssa_candidates = compute_ssa_frequencies_without_cooling(
        log_F_norm=log_F_norm,
        log_omega=log_Omega,
        log_nu_m=log_nu_m,
        log_gamma_m=log_gamma_min,
        p=p,
    )

    regime, log_nu_a = select_ssa_sed_regime_from_candidates_without_cooling(
        log_nu_ssa=log_nu_ssa_candidates,
        log_nu_m=log_nu_m,
    )

    # ---------------------------------------------------------
    # Peak frequency
    # ---------------------------------------------------------
    log_nu_peak = np.maximum(log_nu_a, log_nu_m)

    # ---------------------------------------------------------
    # Return normalization parameters
    # ---------------------------------------------------------
    return {
        "log_F_norm": log_F_norm,
        "log_F_peak": _log_norm_to_log_peak_powerlaw_sbpl_sed_ssa(
            log_nu_m,
            log_nu_a,
            log_F_norm,
            p,
            regime=regime,
        ),
        "log_nu_m": log_nu_m,
        "log_nu_a": log_nu_a,
        "log_nu_max": log_nu_max,
        "log_nu_peak": log_nu_peak,
        "log_omega": log_Omega,
        "regime": regime,
    }
