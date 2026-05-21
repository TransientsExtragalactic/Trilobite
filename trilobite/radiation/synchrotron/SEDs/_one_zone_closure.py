"""
SED closure functions.

This module contains a number of low-level implementations of inverse functions for synchrotron SEDs. These are
used in the ``SEDs`` module to allow the user to go from the peak flux and frequency to the underlying physical
parameters R and B subject to some assumptions about the electron distribution and the relevant microphysics.
"""

import numpy as np

from trilobite.radiation.constants import (
    electron_rest_energy_cgs,
    electron_rest_mass_cgs,
)
from trilobite.radiation.synchrotron.cooling import (
    _synchrotron_cooling_time_coefficient_cgs,
)
from trilobite.radiation.synchrotron.microphysics import (
    _opt_compute_BPL_moment,
    _opt_compute_PL_moment,
)
from trilobite.radiation.synchrotron.utils import (
    _log_c_1_gamma_cgs,
    _log_c_1_gamma_iso_cgs,
    _log_chi_cgs,
    _log_chi_cgs_iso,
)


# ================================================================================ #
# Component Functions
# ================================================================================ #
# These functions provide computation of a few of the standard parameters which are used broadly in
# many of the inversion procedures.
def compute_log_Qm_no_cgs(
    filling_factor,
    log_luminosity_distance,
    gamma_min,
    p,
):
    r"""
    Compute :math:`\log Q_{m,0}` (natural log) for the *fixed pitch-angle* normalization constant.

    Parameters
    ----------
    filling_factor : float or ndarray
        Volume filling factor :math:`f_V` entering
        :math:`V_{\rm eff} = (4\pi/3) R^3 f_V`.

    log_luminosity_distance : float or ndarray
        Natural logarithm of the luminosity distance :math:`\log D_L` (cm).

    gamma_min : float or ndarray
        Minimum Lorentz factor :math:`\gamma_m`.

    p : float or ndarray
        Electron power-law index.

    Returns
    -------
    log_Qm : float or ndarray
        Natural logarithm of :math:`Q_{m,0}`.

    Notes
    -----
    Uses

    .. math::

        Q_{m,0}
        =
        \frac{4}{3}\pi f_V \chi
        \gamma_m^{1-p}
        D_L^{-2}.
    """
    log_qm = (
        np.log((4.0 / 3.0) * np.pi)
        + np.log(filling_factor)
        + _log_chi_cgs
        + (1.0 - p) * np.log(gamma_min)
        - 2.0 * log_luminosity_distance
    )

    return log_qm


def compute_log_Qm_no_cgs_iso(
    filling_factor,
    log_luminosity_distance,
    gamma_min,
    p,
):
    r"""
    Compute :math:`\log Q_{m,\mathrm{ISO}}` for isotropic pitch-angle synchrotron normalization.

    Parameters
    ----------
    filling_factor : float or ndarray
        Volume filling factor :math:`f_V`.

    log_luminosity_distance : float or ndarray
        Natural logarithm of luminosity distance :math:`\log D_L`.

    gamma_min : float or ndarray
        Minimum Lorentz factor :math:`\gamma_m`.

    p : float or ndarray
        Electron power-law index.

    Returns
    -------
    log_Qm : float or ndarray
        Natural logarithm of :math:`Q_{m,\mathrm{ISO}}`.

    Notes
    -----
    Uses

    .. math::

        Q_{m,\mathrm{ISO}}
        =
        \frac{4}{3}\pi f_V \chi_{\rm ISO}
        \gamma_m^{1-p}
        D_L^{-2}.
    """
    log_qm = (
        np.log((4.0 / 3.0) * np.pi)
        + np.log(filling_factor)
        + _log_chi_cgs_iso
        + (1.0 - p) * np.log(gamma_min)
        - 2.0 * log_luminosity_distance
    )

    return log_qm


def compute_log_Qm_slow_cgs(
    filling_factor,
    log_luminosity_distance,
    gamma_min,
    gamma_c,
    p,
):
    r"""
    Compute :math:`\log Q_{m,0}` (natural log) for the *fixed pitch-angle* normalization constant.

    Parameters
    ----------
    filling_factor : float or ndarray
        Volume filling factor :math:`f_V` entering
        :math:`V_{\rm eff} = (4\pi/3) R^3 f_V`.

    log_luminosity_distance : float or ndarray
        Natural logarithm of the luminosity distance :math:`\log D_L` (cm).

    gamma_min : float or ndarray
        Minimum Lorentz factor :math:`\gamma_m`.

    p : float or ndarray
        Electron power-law index.

    Returns
    -------
    log_Qm : float or ndarray
        Natural logarithm of :math:`Q_{m,0}`.

    Notes
    -----
    Uses

    .. math::

        Q_{m,0}
        =
        \frac{4}{3}\pi f_V \chi
        \gamma_m^{1-p}
        D_L^{-2}.
    """
    log_qm = (
        np.log((4.0 / 3.0) * np.pi)
        + np.log(filling_factor)
        + _log_chi_cgs
        + (1.0 - p) * np.log(gamma_min)
        - 2.0 * log_luminosity_distance
        + p * np.log(gamma_c)
    )

    return log_qm


def compute_log_Qm_slow_cgs_iso(
    filling_factor,
    log_luminosity_distance,
    gamma_min,
    gamma_c,
    p,
):
    r"""
    Compute :math:`\log Q_{m,\mathrm{ISO}}` for isotropic pitch-angle synchrotron normalization.

    Parameters
    ----------
    filling_factor : float or ndarray
        Volume filling factor :math:`f_V`.

    log_luminosity_distance : float or ndarray
        Natural logarithm of luminosity distance :math:`\log D_L`.

    gamma_min : float or ndarray
        Minimum Lorentz factor :math:`\gamma_m`.

    p : float or ndarray
        Electron power-law index.

    Returns
    -------
    log_Qm : float or ndarray
        Natural logarithm of :math:`Q_{m,\mathrm{ISO}}`.

    Notes
    -----
    Uses

    .. math::

        Q_{m,\mathrm{ISO}}
        =
        \frac{4}{3}\pi f_V \chi_{\rm ISO}
        \gamma_m^{1-p}
        D_L^{-2}.
    """
    log_qm = (
        np.log((4.0 / 3.0) * np.pi)
        + np.log(filling_factor)
        + _log_chi_cgs_iso
        + (1.0 - p) * np.log(gamma_min)
        + p * np.log(gamma_c)
        - 2.0 * log_luminosity_distance
    )

    return log_qm


def compute_log_Qc_cgs(
    filling_factor,
    log_luminosity_distance,
    gamma_min,
    gamma_c,
):
    r"""
    Compute :math:`\log Q_{c,0}` for the cooling-regime normalization constant.

    Parameters
    ----------
    filling_factor : float or ndarray
        Volume filling factor :math:`f_V`.

    log_luminosity_distance : float or ndarray
        Natural logarithm of luminosity distance :math:`\log D_L`.

    gamma_min : float or ndarray
        Minimum Lorentz factor :math:`\gamma_m`.

    gamma_c : float or ndarray
        Cooling Lorentz factor :math:`\gamma_c`.

    Returns
    -------
    log_Qc : float or ndarray
        Natural logarithm of :math:`Q_{c,0}`.

    Notes
    -----
    Uses

    .. math::

        Q_{c,0}
        =
        \frac{4}{3}\pi f_V \chi
        \gamma_m^{2}
        \gamma_c^{-1}
        D_L^{-2}.
    """
    log_qc = (
        np.log((4.0 / 3.0) * np.pi)
        + np.log(filling_factor)
        + _log_chi_cgs
        - 2.0 * log_luminosity_distance
        + 2.0 * np.log(gamma_min)
        - np.log(gamma_c)
    )

    return log_qc


def compute_log_Qc_cgs_iso(
    filling_factor,
    log_luminosity_distance,
    gamma_min,
    gamma_c,
):
    r"""
    Compute :math:`\log Q_{c,\mathrm{ISO}}` for isotropic pitch-angle cooling normalization.

    Parameters
    ----------
    filling_factor : float or ndarray
        Volume filling factor :math:`f_V`.

    log_luminosity_distance : float or ndarray
        Natural logarithm of luminosity distance :math:`\log D_L`.

    gamma_min : float or ndarray
        Minimum Lorentz factor :math:`\gamma_m`.

    gamma_c : float or ndarray
        Cooling Lorentz factor :math:`\gamma_c`.

    Returns
    -------
    log_Qc : float or ndarray
        Natural logarithm of :math:`Q_{c,\mathrm{ISO}}`.
    """
    log_qc = (
        np.log((4.0 / 3.0) * np.pi)
        + np.log(filling_factor)
        + _log_chi_cgs_iso
        - 2.0 * log_luminosity_distance
        + 2.0 * np.log(gamma_min)
        - np.log(gamma_c)
    )

    return log_qc


def compute_log_P0_cgs(
    area_factor,
    log_angular_diameter_distance,
    sin_pitch_angle=1.0,
):
    r"""
    Compute ``log(P0)`` (natural log) for the SSA blackbody-peak condition (fixed pitch angle).

    Parameters
    ----------
    area_factor : float or ndarray
        Angular filling factor :math:`f_A` entering
        :math:`\Omega = (\pi R^2 / D_A^2) f_A`.
    log_angular_diameter_distance : float or ndarray
        Log angular diameter distance :math:`D_A` in cm.
    sin_pitch_angle : float or ndarray,
        The sine of the pitch angle :math:`\sin\alpha` (dimensionless). Default is 1.0,
        corresponding to a pitch angle of 90 degrees.

    Returns
    -------
    log_P0 : float or ndarray
        Natural log of :math:`P_0` in CGS.

    Notes
    -----
    Uses
    :math:`P_0 = 2\pi m_e f_A\,D_A^{-2}\,c_1^{-1/2}\,\sin^{-1/2}\alpha`.

    This constant appears in the SSA inversion condition
    :math:`F_{\rm brk}\,\nu_{\rm brk}^{-5/2} = P_0 R^2 B^{-1/2}`.
    """
    log_fa = np.log(area_factor)
    log_sin = np.log(sin_pitch_angle)

    log_p0 = np.log(2.0 * np.pi * electron_rest_mass_cgs)
    log_p0 += log_fa - 2.0 * log_angular_diameter_distance
    log_p0 += -0.5 * _log_c_1_gamma_cgs - 0.5 * log_sin
    return log_p0


def compute_log_P0_cgs_iso(
    area_factor,
    log_angular_diameter_distance,
):
    r"""
    Compute ``log(P0)`` (natural log) for the SSA blackbody-peak condition (isotropic pitch angle).

    Parameters
    ----------
    area_factor : float or ndarray
        Angular filling factor :math:`f_A`.
    log_angular_diameter_distance : float or ndarray
        Log angular diameter distance :math:`D_A` in cm.

    Returns
    -------
    log_P0 : float or ndarray
        Natural log of :math:`P_{0,\mathrm{ISO}}` in CGS.

    Notes
    -----
    Uses
    :math:`P_{0,\mathrm{ISO}} = 2\pi m_e f_A\,D_A^{-2}\,c_{1,\mathrm{ISO}}^{-1/2}`.
    """
    log_fa = np.log(area_factor)

    log_p0 = np.log(2.0 * np.pi * electron_rest_mass_cgs)
    log_p0 += log_fa - 2.0 * log_angular_diameter_distance
    log_p0 += -0.5 * _log_c_1_gamma_iso_cgs
    return log_p0


def compute_log_N0_fast_cooling(
    gamma_min: float,
    gamma_c: float,
    p: float,
    epsilon_e: float,
    epsilon_B: float,
    gamma_max: float = np.inf,
):
    r"""
    Compute log(tilde{N_0}) for fast cooling.

    Fast cooling distribution:
        slope -2 for γ_c < γ < γ_min
        slope -p for γ > γ_min

    The first moment is computed in scaled units x = γ / γ_c.
    """
    # Dimensionless bounds
    x_min = gamma_c / gamma_min
    x_max = gamma_max / gamma_min

    # First moment of BPL
    log_moment = np.log(_opt_compute_BPL_moment(-2.0, -(p + 1), x_min, x_max)) + 2 * np.log(gamma_min)
    log_prefactor = np.log(epsilon_e) - np.log(epsilon_B) - np.log(8.0 * np.pi) - np.log(electron_rest_energy_cgs)

    return log_prefactor - log_moment


def compute_log_N0_slow_cooling(
    gamma_min: float,
    gamma_c: float,
    p: float,
    epsilon_e: float,
    epsilon_B: float,
    gamma_max: float = np.inf,
):
    r"""
    Compute log(tilde{N_0}) for slow cooling.

    Slow cooling distribution:
        slope -p for γ_min < γ < γ_c
        slope -(p+1) for γ > γ_c
    """
    x_min, x_max = gamma_min / gamma_c, gamma_max / gamma_c

    log_moment = np.log(_opt_compute_BPL_moment(-p, -(p + 1.0), x_min, x_max)) + 2 * np.log(gamma_c)
    log_prefactor = np.log(epsilon_e) - np.log(epsilon_B) - np.log(8.0 * np.pi) - np.log(electron_rest_energy_cgs)

    return log_prefactor - log_moment


def compute_log_N0_no_cooling(
    gamma_min: float,
    p: float,
    epsilon_e: float,
    epsilon_B: float,
    gamma_max: float = np.inf,
):
    r"""
    Compute log(tilde{N_0}) for pure power-law (no cooling).

    Distribution:
        slope -p for γ_min < γ < γ_max
    """
    log_moment = np.log(_opt_compute_PL_moment(p, gamma_min, gamma_max, order=1))

    log_prefactor = np.log(epsilon_e) - np.log(epsilon_B) - np.log(8.0 * np.pi) - np.log(electron_rest_energy_cgs)

    return log_prefactor - log_moment


# ================================================================================ #
# Inversion Functions                                                              #
# ================================================================================ #
# We now implement the relevant inversion functions so that we can go from the peak
# frequency to the underlying physical parameters. We do EVERYTHING in CGS in the low-level
# implementation available here. The SED classes in ``SEDs`` will handle the public interface.


# === Single Power-Law, No Cooling, No SSA === #
def _inv_log_powerlaw_sbpl_sed(
    log_nu_peak: float,
    log_F_nu_peak: float,
    log_DL: float,
    *,
    gamma_min: float = 1,
    gamma_max: float = np.inf,
    epsilon_e: float = 0.1,
    epsilon_B: float = 0.01,
    f_V: float = 0.5,
    p: float = 3,
    sin_alpha: float = None,
):
    # Determine if we are pitch averaging. Compute the relevant constants based on this choice
    # so that we can run with a single logical branch for the rest of the inversion procedure.
    if sin_alpha is None:
        _is_pitch_averaged = True
        log_Qm = compute_log_Qm_no_cgs_iso(
            f_V,
            log_DL,
            gamma_min,
            p,
        )
        log_c1 = _log_c_1_gamma_iso_cgs
    else:
        _is_pitch_averaged = False
        log_Qm = compute_log_Qm_no_cgs(
            f_V,
            log_DL,
            gamma_min,
            p,
        )
        log_c1 = _log_c_1_gamma_cgs

    # Compute the correct N0 for this scenario.
    log_N0 = compute_log_N0_no_cooling(gamma_min, p, epsilon_e, epsilon_B, gamma_max)
    # Compute the B-field first from the optically thin closure and the definition of the
    # peak frequency.
    log_B = log_nu_peak - 2 * np.log(gamma_min) - log_c1

    # Now compute the radius using the standard closure method.
    log_R = (1 / 3) * (log_F_nu_peak - log_Qm - (3 * log_B) - log_N0)

    return log_R, log_B


# === Single Power-Law, Cooling, No SSA === #
def _inv_log_powerlaw_sbpl_sed_cool_1(
    log_nu_peak: float,
    log_F_nu_peak: float,
    log_DL: float,
    gamma_min: float = 1,
    gamma_c: float = 1,
    gamma_max: float = np.inf,
    epsilon_e: float = 0.1,
    epsilon_B: float = 0.01,
    f_V: float = 0.5,
    p: float = 3,
    sin_alpha: float = None,
):
    # Determine if we are pitch averaging. Compute the relevant constants based on this choice
    # so that we can run with a single logical branch for the rest of the inversion procedure.
    if sin_alpha is None:
        _is_pitch_averaged = True
        log_Qc = compute_log_Qc_cgs_iso(
            f_V,
            log_DL,
            gamma_min,
            gamma_c,
        )
        log_c1 = _log_c_1_gamma_iso_cgs
    else:
        _is_pitch_averaged = False
        log_Qc = compute_log_Qc_cgs(
            f_V,
            log_DL,
            gamma_min,
            gamma_c,
        )
        log_c1 = _log_c_1_gamma_cgs

    # Compute the correct N0 for this scenario.
    log_N0 = compute_log_N0_fast_cooling(gamma_min, gamma_c, p, epsilon_e, epsilon_B, gamma_max)

    # Compute the B-field first from the optically thin closure and the definition of the
    # peak frequency.
    log_B = log_nu_peak - 2 * np.log(gamma_c) - log_c1

    # Now compute the radius using the standard closure method.
    log_R = (1 / 3) * (log_F_nu_peak - log_Qc - (3 * log_B) - log_N0)

    return log_R, log_B


def _inv_log_powerlaw_sbpl_sed_cool_2(
    log_nu_peak: float,
    log_F_nu_peak: float,
    log_DL: float,
    gamma_min: float = 1,
    gamma_c: float = 1,
    gamma_max: float = np.inf,
    epsilon_e: float = 0.1,
    epsilon_B: float = 0.01,
    f_V: float = 0.5,
    p: float = 3,
    sin_alpha: float = None,
):
    # Determine if we are pitch averaging. Compute the relevant constants based on this choice
    # so that we can run with a single logical branch for the rest of the inversion procedure.
    if sin_alpha is None:
        _is_pitch_averaged = True
        log_Qm = compute_log_Qm_slow_cgs_iso(
            f_V,
            log_DL,
            gamma_min,
            gamma_c,
            p,
        )
        log_c1 = _log_c_1_gamma_iso_cgs
    else:
        _is_pitch_averaged = False
        log_Qm = compute_log_Qm_slow_cgs(
            f_V,
            log_DL,
            gamma_min,
            gamma_c,
            p,
        )
        log_c1 = _log_c_1_gamma_cgs

    # Compute the correct N0 for this scenario.
    log_N0 = compute_log_N0_slow_cooling(gamma_min, gamma_c, p, epsilon_e, epsilon_B, gamma_max)

    # Compute the B-field first from the optically thin closure and the definition of the
    # peak frequency.
    log_B = log_nu_peak - 2 * np.log(gamma_min) - log_c1

    # Now compute the radius using the standard closure method.
    log_R = (1 / 3) * (log_F_nu_peak - log_Qm - (3 * log_B) - log_N0)

    return log_R, log_B


def _inv_log_powerlaw_sbpl_sed_cool_3(
    log_nu_peak: float,
    log_F_nu_peak: float,
    log_DL: float,
    gamma_min: float = 1,
    gamma_c: float = 1,
    gamma_max: float = np.inf,
    epsilon_e: float = 0.1,
    epsilon_B: float = 0.01,
    f_V: float = 0.5,
    p: float = 3,
    sin_alpha: float = None,
):
    # Ignore gamma_c since we aren't in the cooling regime.
    _ = gamma_c

    # Determine if we are pitch averaging. Compute the relevant constants based on this choice
    # so that we can run with a single logical branch for the rest of the inversion procedure.
    if sin_alpha is None:
        _is_pitch_averaged = True
        log_Qm = compute_log_Qm_no_cgs_iso(
            f_V,
            log_DL,
            gamma_min,
            p,
        )
        log_c1 = _log_c_1_gamma_iso_cgs
    else:
        _is_pitch_averaged = False
        log_Qm = compute_log_Qm_no_cgs(
            f_V,
            log_DL,
            gamma_min,
            p,
        )
        log_c1 = _log_c_1_gamma_cgs

    # Compute the correct N0 for this scenario.
    log_N0 = compute_log_N0_no_cooling(gamma_min, p, epsilon_e, epsilon_B, gamma_max)

    # Compute the B-field first from the optically thin closure and the definition of the
    # peak frequency.
    log_B = log_nu_peak - 2 * np.log(gamma_min) - log_c1

    # Now compute the radius using the standard closure method.
    log_R = (1 / 3) * (log_F_nu_peak - log_Qm - (3 * log_B) - log_N0)

    return log_R, log_B


# === Single Power-Law, No Cooling, SSA === #
def _inv_log_powerlaw_sbpl_sed_ssa_1(
    log_nu_peak: float,
    log_F_nu_peak: float,
    log_DL: float,
    gamma_min: float = 1,
    gamma_max: float = np.inf,
    epsilon_e: float = 0.1,
    epsilon_B: float = 0.01,
    f_V: float = 0.5,
    p: float = 3,
    sin_alpha: float = None,
):
    # Determine if we are pitch averaging. Compute the relevant constants based on this choice
    # so that we can run with a single logical branch for the rest of the inversion procedure.
    if sin_alpha is None:
        _is_pitch_averaged = True
        log_Qm = compute_log_Qm_no_cgs_iso(
            f_V,
            log_DL,
            gamma_min,
            p,
        )
        log_c1 = _log_c_1_gamma_iso_cgs
    else:
        _is_pitch_averaged = False
        log_Qm = compute_log_Qm_no_cgs(
            f_V,
            log_DL,
            gamma_min,
            p,
        )
        log_c1 = _log_c_1_gamma_cgs

    # Compute the correct N0 for this scenario.
    log_N0 = compute_log_N0_no_cooling(gamma_min, p, epsilon_e, epsilon_B, gamma_max)

    # Compute the B-field first from the optically thin closure and the definition of the
    # peak frequency.
    log_B = log_nu_peak - 2 * np.log(gamma_min) - log_c1

    # Now compute the radius using the standard closure method.
    log_R = (1 / 3) * (log_F_nu_peak - log_Qm - (3 * log_B) - log_N0)

    return log_R, log_B


def _inv_log_powerlaw_sbpl_sed_ssa_2(
    log_nu_peak: float,
    log_F_nu_peak: float,
    log_DL: float,
    log_DA: float,
    gamma_min: float = 1,
    gamma_max: float = np.inf,
    epsilon_e: float = 0.1,
    epsilon_B: float = 0.01,
    f_V: float = 0.5,
    f_A: float = 0.5,
    p: float = 3,
    sin_alpha: float = None,
):
    # Determine if we are pitch averaging. Compute the relevant constants based on this choice
    # so that we can run with a single logical branch for the rest of the inversion procedure.
    if sin_alpha is None:
        _is_pitch_averaged = True
        log_Qm = compute_log_Qm_no_cgs_iso(
            f_V,
            log_DL,
            gamma_min,
            p,
        )
        log_c1 = _log_c_1_gamma_iso_cgs

        log_P0 = compute_log_P0_cgs_iso(f_A, log_DA)

        # Compute A: we just do the angular component here so that
        # the rest can be done in a single branch for both the pitch-averaged and non-pitch-averaged cases.
        _A = ((p - 1) / 2) * (log_c1 + 2 * np.log(gamma_min)) + log_Qm
    else:
        _is_pitch_averaged = False
        log_Qm = compute_log_Qm_no_cgs(
            f_V,
            log_DL,
            gamma_min,
            p,
        )
        log_c1 = _log_c_1_gamma_cgs
        log_P0 = compute_log_P0_cgs(f_A, log_DA, sin_alpha)

        # Compute A: we just do the angular component here so that
        # the rest can be done in a single branch for both the pitch-averaged and non-pitch-averaged cases.
        _A = ((p - 1) / 2) * (log_c1 + 2 * np.log(gamma_min)) + log_Qm + ((p + 1) / 2) * np.log(sin_alpha)

    # Compute the correct N0 for this scenario.
    log_N0 = compute_log_N0_no_cooling(gamma_min, p, epsilon_e, epsilon_B, gamma_max)

    # Finalize the A variable
    _A += log_N0

    # Compute log R and log B. We will reduce out the exponents so that it's crystal clear
    # how these play out for easy debugging.
    ar_1, ar_2, ar_3, ar_4 = -1 / (2 * p + 13), -(p + 5) / (2 * p + 13), (p + 6) / (2 * p + 13), -1
    ab_1, ab_2, ab_3, ab_4 = -4 / (2 * p + 13), 6 / (2 * p + 13), -2 / (2 * p + 13), 1

    log_R = (ar_1 * _A) + (ar_2 * log_P0) + (ar_3 * log_F_nu_peak) + (ar_4 * log_nu_peak)
    log_B = (ab_1 * _A) + (ab_2 * log_P0) + (ab_3 * log_F_nu_peak) + (ab_4 * log_nu_peak)

    return log_R, log_B


# === Single Power-Law, Cooling, SSA === #
def _inv_log_powerlaw_sbpl_sed_ssa_cool_1(
    log_nu_peak: float,
    log_F_nu_peak: float,
    log_DL: float,
    gamma_min: float = 1,
    gamma_max: float = np.inf,
    epsilon_e: float = 0.1,
    epsilon_B: float = 0.01,
    f_V: float = 0.5,
    p: float = 3,
    sin_alpha: float = None,
):
    # Determine if we are pitch averaging. Compute the relevant constants based on this choice
    # so that we can run with a single logical branch for the rest of the inversion procedure.
    if sin_alpha is None:
        _is_pitch_averaged = True
        log_Qm = compute_log_Qm_no_cgs_iso(
            f_V,
            log_DL,
            gamma_min,
            p,
        )
        log_c1 = _log_c_1_gamma_iso_cgs
    else:
        _is_pitch_averaged = False
        log_Qm = compute_log_Qm_no_cgs(
            f_V,
            log_DL,
            gamma_min,
            p,
        )
        log_c1 = _log_c_1_gamma_cgs

    # Compute the correct N0 for this scenario.
    log_N0 = compute_log_N0_no_cooling(gamma_min, p, epsilon_e, epsilon_B, gamma_max)

    # Compute the B-field first from the optically thin closure and the definition of the
    # peak frequency.
    log_B = log_nu_peak - 2 * np.log(gamma_min) - log_c1

    # Now compute the radius using the standard closure method.
    log_R = (1 / 3) * (log_F_nu_peak - log_Qm - (3 * log_B) - log_N0)

    return log_R, log_B


def _inv_log_powerlaw_sbpl_sed_ssa_cool_2(
    log_nu_peak: float,
    log_F_nu_peak: float,
    log_DL: float,
    log_DA: float,
    gamma_min: float = 1,
    gamma_max: float = np.inf,
    epsilon_e: float = 0.1,
    epsilon_B: float = 0.01,
    f_V: float = 0.5,
    f_A: float = 0.5,
    p: float = 3,
    sin_alpha: float = None,
):
    # Determine if we are pitch averaging. Compute the relevant constants based on this choice
    # so that we can run with a single logical branch for the rest of the inversion procedure.
    if sin_alpha is None:
        _is_pitch_averaged = True
        log_Qm = compute_log_Qm_no_cgs_iso(
            f_V,
            log_DL,
            gamma_min,
            p,
        )
        log_c1 = _log_c_1_gamma_iso_cgs
        log_P0 = compute_log_P0_cgs_iso(f_A, log_DA)

        # Compute A: we just do the angular component here so that
        # the rest can be done in a single branch for both the pitch-averaged and non-pitch-averaged cases.
        _A = ((p - 1) / 2) * (log_c1 + 2 * np.log(gamma_min)) + log_Qm
    else:
        _is_pitch_averaged = False
        log_Qm = compute_log_Qm_no_cgs(
            f_V,
            log_DL,
            gamma_min,
            p,
        )
        log_c1 = _log_c_1_gamma_cgs
        log_P0 = compute_log_P0_cgs(f_A, log_DA, sin_alpha)

        # Compute A: we just do the angular component here so that
        # the rest can be done in a single branch for both the pitch-averaged and non-pitch-averaged cases.
        _A = ((p - 1) / 2) * (log_c1 + 2 * np.log(gamma_min)) + log_Qm + ((p + 1) / 2) * np.log(sin_alpha)

    # Compute the correct N0 for this scenario.
    log_N0 = compute_log_N0_no_cooling(gamma_min, p, epsilon_e, epsilon_B, gamma_max)

    # Finalize the A variable
    _A += log_N0

    # Compute log R and log B. We will reduce out the exponents so that it's crystal clear
    # how these play out for easy debugging.
    ar_1, ar_2, ar_3, ar_4 = -1 / (2 * p + 13), -(p + 5) / (2 * p + 13), (p + 6) / (2 * p + 13), -1
    ab_1, ab_2, ab_3, ab_4 = -4 / (2 * p + 13), 6 / (2 * p + 13), -2 / (2 * p + 13), 1

    log_R = (ar_1 * _A) + (ar_2 * log_P0) + (ar_3 * log_F_nu_peak) + (ar_4 * log_nu_peak)
    log_B = (ab_1 * _A) + (ab_2 * log_P0) + (ab_3 * log_F_nu_peak) + (ab_4 * log_nu_peak)

    return log_R, log_B


def _inv_log_powerlaw_sbpl_sed_ssa_cool_3(
    log_nu_peak: float,
    log_F_nu_peak: float,
    log_DL: float,
    gamma_min: float = 1,
    gamma_c: float = 1,
    gamma_max: float = np.inf,
    epsilon_e: float = 0.1,
    epsilon_B: float = 0.01,
    f_V: float = 0.5,
    p: float = 3,
    sin_alpha: float = None,
):
    # Determine if we are pitch averaging. Compute the relevant constants based on this choice
    # so that we can run with a single logical branch for the rest of the inversion procedure.
    if sin_alpha is None:
        _is_pitch_averaged = True
        log_Qm = compute_log_Qm_slow_cgs_iso(
            f_V,
            log_DL,
            gamma_min,
            gamma_c,
            p,
        )
        log_c1 = _log_c_1_gamma_iso_cgs
    else:
        _is_pitch_averaged = False
        log_Qm = compute_log_Qm_slow_cgs(
            f_V,
            log_DL,
            gamma_min,
            gamma_c,
            p,
        )
        log_c1 = _log_c_1_gamma_cgs

    # Compute the correct N0 for this scenario.
    log_N0 = compute_log_N0_slow_cooling(gamma_min, gamma_c, p, epsilon_e, epsilon_B, gamma_max)

    # Compute the B-field first from the optically thin closure and the definition of the
    # peak frequency.
    log_B = log_nu_peak - 2 * np.log(gamma_min) - log_c1

    # Now compute the radius using the standard closure method.
    log_R = (1 / 3) * (log_F_nu_peak - log_Qm - (3 * log_B) - log_N0)

    return log_R, log_B


def _inv_log_powerlaw_sbpl_sed_ssa_cool_4(
    log_nu_peak: float,
    log_F_nu_peak: float,
    log_DL: float,
    log_DA: float,
    gamma_min: float = 1,
    gamma_c: float = 1,
    gamma_max: float = np.inf,
    epsilon_e: float = 0.1,
    epsilon_B: float = 0.01,
    f_V: float = 0.5,
    f_A: float = 0.5,
    p: float = 3,
    sin_alpha: float = None,
):
    # Determine if we are pitch averaging. Compute the relevant constants based on this choice
    # so that we can run with a single logical branch for the rest of the inversion procedure.
    if sin_alpha is None:
        _is_pitch_averaged = True
        log_Qm = compute_log_Qm_slow_cgs_iso(
            f_V,
            log_DL,
            gamma_min,
            gamma_c,
            p,
        )
        log_c1 = _log_c_1_gamma_iso_cgs
        log_P0 = compute_log_P0_cgs_iso(f_A, log_DA)

        # Compute A: we just do the angular component here so that
        # the rest can be done in a single branch for both the pitch-averaged and non-pitch-averaged cases.
        _A = ((p - 1) / 2) * (log_c1 + 2 * np.log(gamma_min)) + log_Qm
    else:
        _is_pitch_averaged = False
        log_Qm = compute_log_Qm_slow_cgs(
            f_V,
            log_DL,
            gamma_min,
            gamma_c,
            p,
        )
        log_c1 = _log_c_1_gamma_cgs
        log_P0 = compute_log_P0_cgs(f_A, log_DA, sin_alpha)

        # Compute A: we just do the angular component here so that
        # the rest can be done in a single branch for both the pitch-averaged and non-pitch-averaged cases.
        _A = ((p - 1) / 2) * (log_c1 + 2 * np.log(gamma_min)) + log_Qm + ((p + 1) / 2) * np.log(sin_alpha)

    # Compute the correct N0 for this scenario.
    log_N0 = compute_log_N0_slow_cooling(gamma_min, gamma_c, p, epsilon_e, epsilon_B, gamma_max)

    # Finalize the A variable
    _A += log_N0

    # Compute log R and log B. We will reduce out the exponents so that it's crystal clear
    # how these play out for easy debugging.
    ar_1, ar_2, ar_3, ar_4 = -1 / (2 * p + 13), -(p + 5) / (2 * p + 13), (p + 6) / (2 * p + 13), -1
    ab_1, ab_2, ab_3, ab_4 = -4 / (2 * p + 13), 6 / (2 * p + 13), -2 / (2 * p + 13), 1

    log_R = (ar_1 * _A) + (ar_2 * log_P0) + (ar_3 * log_F_nu_peak) + (ar_4 * log_nu_peak)
    log_B = (ab_1 * _A) + (ab_2 * log_P0) + (ab_3 * log_F_nu_peak) + (ab_4 * log_nu_peak)

    return log_R, log_B


def _inv_log_powerlaw_sbpl_sed_ssa_cool_5(
    log_nu_peak: float,
    log_F_nu_peak: float,
    log_DL: float,
    gamma_min: float = 1,
    gamma_c: float = 1,
    gamma_max: float = np.inf,
    epsilon_e: float = 0.1,
    epsilon_B: float = 0.01,
    f_V: float = 0.5,
    p: float = 3,
    sin_alpha: float = None,
):
    # Determine if we are pitch averaging. Compute the relevant constants based on this choice
    # so that we can run with a single logical branch for the rest of the inversion procedure.
    if sin_alpha is None:
        _is_pitch_averaged = True
        log_Qc = compute_log_Qc_cgs_iso(
            f_V,
            log_DL,
            gamma_min,
            gamma_c,
        )
        log_c1 = _log_c_1_gamma_iso_cgs
    else:
        _is_pitch_averaged = False
        log_Qc = compute_log_Qc_cgs(
            f_V,
            log_DL,
            gamma_min,
            gamma_c,
        )
        log_c1 = _log_c_1_gamma_cgs

    # Compute the correct N0 for this scenario.
    log_N0 = compute_log_N0_fast_cooling(gamma_min, gamma_c, p, epsilon_e, epsilon_B, gamma_max)

    # Compute the B-field first from the optically thin closure and the definition of the
    # peak frequency.
    log_B = log_nu_peak - 2 * np.log(gamma_c) - log_c1

    # Now compute the radius using the standard closure method.
    log_R = (1 / 3) * (log_F_nu_peak - log_Qc - (3 * log_B) - log_N0)

    return log_R, log_B


def _inv_log_powerlaw_sbpl_sed_ssa_cool_6(
    log_nu_peak: float,
    log_F_nu_peak: float,
    log_DL: float,
    log_DA: float,
    gamma_min: float = 1,
    gamma_c: float = 1,
    gamma_max: float = np.inf,
    epsilon_e: float = 0.1,
    epsilon_B: float = 0.01,
    f_V: float = 0.5,
    f_A: float = 0.5,
    p: float = 3,
    sin_alpha: float = None,
):
    # Determine if we are pitch averaging. Compute the relevant constants based on this choice
    # so that we can run with a single logical branch for the rest of the inversion procedure.
    if sin_alpha is None:
        _is_pitch_averaged = True
        log_Qc = compute_log_Qc_cgs_iso(
            f_V,
            log_DL,
            gamma_min,
            gamma_c,
        )
        log_c1 = _log_c_1_gamma_iso_cgs
        log_P0 = compute_log_P0_cgs_iso(f_A, log_DA)

        # Compute A: we just do the angular component here so that
        # the rest can be done in a single branch for both the pitch-averaged and non-pitch-averaged cases.
        _A = (0.5) * (log_c1 + 2 * np.log(gamma_c)) + log_Qc
    else:
        _is_pitch_averaged = False
        log_Qc = compute_log_Qc_cgs(
            f_V,
            log_DL,
            gamma_min,
            gamma_c,
        )
        log_c1 = _log_c_1_gamma_cgs
        log_P0 = compute_log_P0_cgs(f_A, log_DA, sin_alpha)

        # Compute A: we just do the angular component here so that
        # the rest can be done in a single branch for both the pitch-averaged and non-pitch-averaged cases.
        _A = (0.5) * (log_c1 + 2 * np.log(gamma_c)) + log_Qc + (3 / 2) * np.log(sin_alpha)

    # Compute the correct N0 for this scenario.
    log_N0 = compute_log_N0_fast_cooling(gamma_min, gamma_c, p, epsilon_e, epsilon_B, gamma_max)

    # Finalize the A variable
    _A += log_N0

    # Compute log R and log B. We will reduce out the exponents so that it's crystal clear
    # how these play out for easy debugging.
    ar_1, ar_2, ar_3, ar_4 = -1 / 17, -7 / 17, 8 / 17, -1
    ab_1, ab_2, ab_3, ab_4 = -4 / 17, 6 / 17, -2 / 17, 1

    log_R = (ar_1 * _A) + (ar_2 * log_P0) + (ar_3 * log_F_nu_peak) + (ar_4 * log_nu_peak)
    log_B = (ab_1 * _A) + (ab_2 * log_P0) + (ab_3 * log_F_nu_peak) + (ab_4 * log_nu_peak)

    return log_R, log_B


def _inv_log_powerlaw_sbpl_sed_ssa_cool_7(
    log_nu_peak: float,
    log_F_nu_peak: float,
    log_DL: float,
    log_DA: float,
    gamma_min: float = 1,
    gamma_c: float = 1,
    gamma_max: float = np.inf,
    epsilon_e: float = 0.1,
    epsilon_B: float = 0.01,
    f_V: float = 0.5,
    f_A: float = 0.5,
    p: float = 3,
    sin_alpha: float = None,
):
    # Coerce everything to arrays so that we can work cleanly in the
    # array space without having to worry about scalars vs. arrays for the rest of the function.
    # This is corrected at the end to return scalars if scalars were input.
    log_nu_peak = np.asarray(log_nu_peak)
    log_F_nu_peak = np.asarray(log_F_nu_peak)
    log_DL = np.asarray(log_DL)
    log_DA = np.asarray(log_DA)

    # --- PITCH-ANGLE MANAGEMENT --- #
    if sin_alpha is None:
        # Calculate the necessary (agnostic) constants before moving into the branching for fast vs. slow cooling.
        log_P0 = compute_log_P0_cgs_iso(f_A, log_DA)

        # Compute the Q and A arrays.
        log_Q = compute_log_Qm_slow_cgs_iso(
            f_V,
            log_DL,
            gamma_min,
            gamma_c,
            p,
        )
        log_A = (p / 2) * _log_c_1_gamma_iso_cgs + (p - 1) * np.log(gamma_min) + np.log(gamma_c) + log_Q

    else:
        # Calculate the necessary (agnostic) constants before moving into the branching for fast vs. slow cooling.
        log_P0 = compute_log_P0_cgs(f_A, log_DA, sin_alpha)

        # Compute the Q and A arrays.
        log_Q = compute_log_Qm_slow_cgs(
            f_V,
            log_DL,
            gamma_min,
            gamma_c,
            p,
        )
        log_A = (
            (p / 2) * _log_c_1_gamma_cgs
            + (p - 1) * np.log(gamma_min)
            + ((p + 2) / 2) * np.log(sin_alpha)
            + np.log(gamma_c)
            + log_Q
        )

    # --- COOLING MANAGEMENT --- #
    log_N0 = compute_log_N0_slow_cooling(gamma_min, gamma_c, p, epsilon_e, epsilon_B, gamma_max)

    A = log_A + log_N0

    # Compute log R and log B. We will reduce out the exponents so that it's crystal clear
    # how these play out for easy debugging.
    ar_1, ar_2, ar_3, ar_4 = -1 / (2 * p + 15), -(p + 6) / (2 * p + 15), (p + 7) / (2 * p + 15), -1
    ab_1, ab_2, ab_3, ab_4 = -4 / (2 * p + 15), 6 / (2 * p + 15), -2 / (2 * p + 15), 1

    log_R = (ar_1 * A) + (ar_2 * log_P0) + (ar_3 * log_F_nu_peak) + (ar_4 * log_nu_peak)
    log_B = (ab_1 * A) + (ab_2 * log_P0) + (ab_3 * log_F_nu_peak) + (ab_4 * log_nu_peak)

    return log_R, log_B


def _inv_log_powerlaw_sbpl_sed_ssa_cool_8(
    log_nu_peak: float,
    log_F_nu_peak: float,
    log_DL: float,
    log_DA: float,
    gamma_min: float = 1,
    gamma_c: float = 1,
    gamma_max: float = np.inf,
    epsilon_e: float = 0.1,
    epsilon_B: float = 0.01,
    f_V: float = 0.5,
    f_A: float = 0.5,
    p: float = 3,
    sin_alpha: float = None,
):
    # Coerce everything to arrays so that we can work cleanly in the
    # array space without having to worry about scalars vs. arrays for the rest of the function.
    # This is corrected at the end to return scalars if scalars were input.
    log_nu_peak = np.asarray(log_nu_peak)
    log_F_nu_peak = np.asarray(log_F_nu_peak)
    log_DL = np.asarray(log_DL)
    log_DA = np.asarray(log_DA)

    # --- PITCH-ANGLE MANAGEMENT --- #
    if sin_alpha is None:
        # Calculate the necessary (agnostic) constants before moving into the branching for fast vs. slow cooling.
        log_P0 = compute_log_P0_cgs_iso(f_A, log_DA)

        # Compute the Q and A arrays.
        log_Q = compute_log_Qc_cgs_iso(
            f_V,
            log_DL,
            gamma_min,
            gamma_c,
        )
        log_A = (p / 2) * _log_c_1_gamma_iso_cgs + (p - 1) * np.log(gamma_min) + np.log(gamma_c) + log_Q

    else:
        # Calculate the necessary (agnostic) constants before moving into the branching for fast vs. slow cooling.
        log_P0 = compute_log_P0_cgs(f_A, log_DA, sin_alpha)

        # Compute the Q and A arrays.
        log_Q = compute_log_Qc_cgs(
            f_V,
            log_DL,
            gamma_min,
            gamma_c,
        )
        log_A = (
            (p / 2) * _log_c_1_gamma_cgs
            + (p - 1) * np.log(gamma_min)
            + ((p + 2) / 2) * np.log(sin_alpha)
            + np.log(gamma_c)
            + log_Q
        )

    # --- COOLING MANAGEMENT --- #
    log_N0 = compute_log_N0_fast_cooling(gamma_min, gamma_c, p, epsilon_e, epsilon_B, gamma_max)

    A = log_A + log_N0

    # Compute log R and log B. We will reduce out the exponents so that it's crystal clear
    # how these play out for easy debugging.
    ar_1, ar_2, ar_3, ar_4 = -1 / (2 * p + 15), -(p + 6) / (2 * p + 15), (p + 7) / (2 * p + 15), -1
    ab_1, ab_2, ab_3, ab_4 = -4 / (2 * p + 15), 6 / (2 * p + 15), -2 / (2 * p + 15), 1

    log_R = (ar_1 * A) + (ar_2 * log_P0) + (ar_3 * log_F_nu_peak) + (ar_4 * log_nu_peak)
    log_B = (ab_1 * A) + (ab_2 * log_P0) + (ab_3 * log_F_nu_peak) + (ab_4 * log_nu_peak)

    return log_R, log_B


# === Single Power-Law, Cooling, SSA (Implicit nu_c) === #
# These closures are based on Ho+22, which implicitly includes the gamma_c closure to allow
# for self-consistent synchrotron cooling. These are documented and described in detail in the
# ``reference/physics/radiation/synchrotron/synchrotron_cooling_closure.rst`` file in the documentation.
#
# We can only support 3 cases: the optically thin slow cooling SED, and spectra 4 and 7 in the SSA case.
def _inv_log_powerlaw_sbpl_sed_implicit_cool_2(
    log_nu_peak: float,
    log_F_nu_peak: float,
    log_t: float,
    log_DL: float,
    gamma_min: float = 1,
    gamma_max: float = np.inf,
    epsilon_e: float = 0.1,
    epsilon_B: float = 0.01,
    f_V: float = 0.5,
    p: float = 3,
    sin_alpha: float = None,
):
    log_nu_peak = np.asarray(log_nu_peak)
    log_F_nu_peak = np.asarray(log_F_nu_peak)
    log_t = np.asarray(log_t)
    log_DL = np.asarray(log_DL)

    # --------- Pitch Averaging --------- #
    # We need to compute specific Q_m dependent on the pitch averaging convention
    # and also get our specific c_1 value for the decided convention.
    if sin_alpha is None:
        _is_pitch_averaged = True
        log_Qm = compute_log_Qm_no_cgs_iso(
            f_V,
            log_DL,
            gamma_min,
            p,
        )
        log_c1 = _log_c_1_gamma_iso_cgs
    else:
        _is_pitch_averaged = False
        log_Qm = compute_log_Qm_no_cgs(
            f_V,
            log_DL,
            gamma_min,
            p,
        )
        log_c1 = _log_c_1_gamma_cgs

    # ---------- Compute the electron distribution normalization ---------- #
    # We compute the correct N0 for this scenario. To avoid allowing for implicit
    # gamma_c dependence in a non-linear manner, we assume a fully uncooled (no cooling) scenario
    # for the N0. This is a known limitation.
    log_N0 = compute_log_N0_no_cooling(gamma_min, p, epsilon_e, epsilon_B, gamma_max)

    # --- Compute log B, log gamma_c, and log R --- #
    # We now apply the closure relations to obtain the correct
    # B-field from the peak frequency.
    log_B = log_nu_peak - 2 * np.log(gamma_min) - log_c1

    # With log_B known, we can just use Theta_C and the definition of gamma_c to get log gamma_c.
    log_theta = np.log(
        _synchrotron_cooling_time_coefficient_cgs
    )  # This is just a constant that comes from the definition of the cooling time.

    log_gamma_c = log_theta - 2 * log_B - log_t

    # Now compute the radius using the standard closure method.
    log_R = (1 / 3) * (log_F_nu_peak - log_Qm - (3 * log_B) - log_N0)

    if np.abs(log_gamma_c - np.log(gamma_min)) < 1:
        raise ValueError()

    # Return everything
    return log_R, log_B, log_gamma_c


def _inv_log_powerlaw_sbpl_sed_ssa_implicit_cool_7(
    log_nu_peak: float,
    log_F_nu_peak: float,
    log_t: float,
    log_DL: float,
    log_DA: float,
    gamma_min: float = 1,
    gamma_max: float = np.inf,
    epsilon_e: float = 0.1,
    epsilon_B: float = 0.01,
    f_V: float = 0.5,
    f_A: float = 0.5,
    p: float = 3,
    sin_alpha: float = None,
):
    # Coerce everything to arrays so that we can work cleanly in the
    # array space without having to worry about scalars vs. arrays for the rest of the function.
    # This is corrected at the end to return scalars if scalars were input.
    log_nu_peak = np.asarray(log_nu_peak)
    log_F_nu_peak = np.asarray(log_F_nu_peak)
    log_DL = np.asarray(log_DL)
    log_DA = np.asarray(log_DA)
    log_t = np.asarray(log_t)

    # --- PITCH-ANGLE MANAGEMENT --- #
    if sin_alpha is None:
        # Calculate the necessary (agnostic) constants before moving into the branching for fast vs. slow cooling.
        log_P0 = compute_log_P0_cgs_iso(f_A, log_DA)

        # Compute the Q and A arrays.
        log_Q = compute_log_Qm_no_cgs_iso(
            f_V,
            log_DL,
            gamma_min,
            p,
        )
        log_A = (p / 2) * _log_c_1_gamma_iso_cgs + (p - 1) * np.log(gamma_min) + log_Q

    else:
        # Calculate the necessary (agnostic) constants before moving into the branching for fast vs. slow cooling.
        log_P0 = compute_log_P0_cgs(f_A, log_DA, sin_alpha)

        # Compute the Q and A arrays.
        log_Q = compute_log_Qm_no_cgs(
            f_V,
            log_DL,
            gamma_min,
            p,
        )
        log_A = (p / 2) * _log_c_1_gamma_cgs + (p - 1) * np.log(gamma_min) + ((p + 2) / 2) * np.log(sin_alpha) + log_Q

    # --- COOLING MANAGEMENT --- #
    log_N0 = compute_log_N0_no_cooling(gamma_min, p, epsilon_e, epsilon_B, gamma_max)
    log_A += log_N0

    # ------ INVERSION LOGIC ------ #
    # We now simply assemble to coefficients and start getting
    # the inversion done.
    log_THETA = np.log(_synchrotron_cooling_time_coefficient_cgs)

    # Compute log R and log B. We will reduce out the exponents so that it's crystal clear
    # how these play out for easy debugging.
    ar1, ar2, ar3, ar4, ar5, ar6 = (
        -1 / (2 * p + 7),
        -1 / (2 * p + 7),
        -(p + 2) / (2 * p + 7),
        (p + 3) / (2 * p + 7),
        -(2 * p + 5) / (2 * p + 7),
        1 / (2 * p + 7),
    )
    ab1, ab2, ab3, ab4, ab5, ab6 = (
        -4 / (2 * p + 7),
        -4 / (2 * p + 7),
        6 / (2 * p + 7),
        -2 / (2 * p + 7),
        (2 * p + 15) / (2 * p + 7),
        4 / (2 * p + 7),
    )
    ac1, ac2, ac3, ac4, ac5, ac6 = (
        (2 * p + 15) / (2 * p + 7),
        8 / (2 * p + 7),
        -12 / (2 * p + 7),
        4 / (2 * p + 7),
        -2 * (2 * p + 15) / (2 * p + 7),
        -(2 * p + 15) / (2 * p + 7),
    )

    log_R = (
        (ar1 * log_THETA) + (ar2 * log_A) + (ar3 * log_P0) + (ar4 * log_F_nu_peak) + (ar5 * log_nu_peak) + (ar6 * log_t)
    )
    log_B = (
        (ab1 * log_THETA) + (ab2 * log_A) + (ab3 * log_P0) + (ab4 * log_F_nu_peak) + (ab5 * log_nu_peak) + (ab6 * log_t)
    )
    log_gamma_c = (
        (ac1 * log_THETA) + (ac2 * log_A) + (ac3 * log_P0) + (ac4 * log_F_nu_peak) + (ac5 * log_nu_peak) + (ac6 * log_t)
    )

    return log_R, log_B, log_gamma_c


def _inv_log_powerlaw_sbpl_sed_ssa_implicit_cool_4(
    log_nu_peak: float,
    log_F_nu_peak: float,
    log_t: float,
    log_DL: float,
    log_DA: float,
    gamma_min: float = 1,
    gamma_max: float = np.inf,
    epsilon_e: float = 0.1,
    epsilon_B: float = 0.01,
    f_V: float = 0.5,
    f_A: float = 0.5,
    p: float = 3,
    sin_alpha: float = None,
):
    # Coerce everything to arrays so that we can work cleanly in the
    # array space without having to worry about scalars vs. arrays for the rest of the function.
    # This is corrected at the end to return scalars if scalars were input.
    log_nu_peak = np.asarray(log_nu_peak)
    log_F_nu_peak = np.asarray(log_F_nu_peak)
    log_DL = np.asarray(log_DL)
    log_DA = np.asarray(log_DA)
    log_t = np.asarray(log_t)

    # --- PITCH-ANGLE MANAGEMENT --- #
    if sin_alpha is None:
        _is_pitch_averaged = True
        log_Qm = compute_log_Qm_no_cgs_iso(
            f_V,
            log_DL,
            gamma_min,
            p,
        )
        log_c1 = _log_c_1_gamma_iso_cgs
        log_P0 = compute_log_P0_cgs_iso(f_A, log_DA)

        # Compute A: we just do the angular component here so that
        # the rest can be done in a single branch for both the pitch-averaged and non-pitch-averaged cases.
        log_A = ((p - 1) / 2) * (log_c1 + 2 * np.log(gamma_min)) + log_Qm
    else:
        _is_pitch_averaged = False
        log_Qm = compute_log_Qm_no_cgs(
            f_V,
            log_DL,
            gamma_min,
            p,
        )
        log_c1 = _log_c_1_gamma_cgs
        log_P0 = compute_log_P0_cgs(f_A, log_DA, sin_alpha)

        # Compute A: we just do the angular component here so that
        # the rest can be done in a single branch for both the pitch-averaged and non-pitch-averaged cases.
        log_A = ((p - 1) / 2) * (log_c1 + 2 * np.log(gamma_min)) + log_Qm + ((p + 1) / 2) * np.log(sin_alpha)

    # --- COOLING MANAGEMENT --- #
    log_N0 = compute_log_N0_no_cooling(gamma_min, p, epsilon_e, epsilon_B, gamma_max)
    log_A += log_N0

    # ------ INVERSION LOGIC ------ #
    # We now simply assemble to coefficients and start getting
    # the inversion done.
    log_THETA = np.log(_synchrotron_cooling_time_coefficient_cgs)

    # Compute log R and log B. We will reduce out the exponents so that it's crystal clear
    # how these play out for easy debugging.
    ar1, ar2, ar3, ar4, ar5, ar6 = (
        0,
        -1 / (2 * p + 13),
        (-p - 5) / (2 * p + 13),
        (p + 6) / (2 * p + 13),
        -1,
        0,
    )
    ab1, ab2, ab3, ab4, ab5, ab6 = (
        0,
        -4 / (2 * p + 13),
        6 / (2 * p + 13),
        -2 / (2 * p + 13),
        1,
        0,
    )
    ac1, ac2, ac3, ac4, ac5, ac6 = (
        1,
        8 / (2 * p + 13),
        -12 / (2 * p + 13),
        4 / (2 * p + 13),
        -2,
        -1,
    )

    log_R = (
        (ar1 * log_THETA) + (ar2 * log_A) + (ar3 * log_P0) + (ar4 * log_F_nu_peak) + (ar5 * log_nu_peak) + (ar6 * log_t)
    )
    log_B = (
        (ab1 * log_THETA) + (ab2 * log_A) + (ab3 * log_P0) + (ab4 * log_F_nu_peak) + (ab5 * log_nu_peak) + (ab6 * log_t)
    )
    log_gamma_c = (
        (ac1 * log_THETA) + (ac2 * log_A) + (ac3 * log_P0) + (ac4 * log_F_nu_peak) + (ac5 * log_nu_peak) + (ac6 * log_t)
    )
    return log_R, log_B, log_gamma_c


SSA_COOLING_INV_FUNCTION_REGISTRY = {
    "Spectrum1": _inv_log_powerlaw_sbpl_sed_ssa_cool_1,
    "Spectrum2": _inv_log_powerlaw_sbpl_sed_ssa_cool_2,
    "Spectrum3": _inv_log_powerlaw_sbpl_sed_ssa_cool_3,
    "Spectrum4": _inv_log_powerlaw_sbpl_sed_ssa_cool_4,
    "Spectrum5": _inv_log_powerlaw_sbpl_sed_ssa_cool_5,
    "Spectrum6": _inv_log_powerlaw_sbpl_sed_ssa_cool_6,
    "Spectrum7": _inv_log_powerlaw_sbpl_sed_ssa_cool_7,
    "Spectrum8": _inv_log_powerlaw_sbpl_sed_ssa_cool_8,
}
"""dict: Registry mapping SSA cooling regimes to their corresponding SED inversion functions."""

SSA_INV_FUNCTION_REGISTRY = {
    "optically_thin": _inv_log_powerlaw_sbpl_sed_ssa_1,
    "optically_thick": _inv_log_powerlaw_sbpl_sed_ssa_2,
}
"""dict: Registry mapping SSA regimes to their corresponding SED inversion functions."""

COOLING_INV_FUNCTION_REGISTRY = {
    "fast_cooling": _inv_log_powerlaw_sbpl_sed_cool_1,
    "slow_cooling": _inv_log_powerlaw_sbpl_sed_cool_2,
    "no_cooling": _inv_log_powerlaw_sbpl_sed,
}
"""dict: Registry mapping cooling regimes to their corresponding SED inversion functions."""
