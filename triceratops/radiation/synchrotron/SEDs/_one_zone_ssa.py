r"""
Self-absorption (SSA) utilities for one-zone synchrotron SEDs.

This module contains low-level analytic expressions for the synchrotron
self-absorption frequency :math:`\nu_a` under the various spectral
configurations that arise in synchrotron emission with power-law
electron populations.

Each function implements the closed-form solution for :math:`\nu_a`
assuming a specific spectral ordering of the characteristic frequencies:

    ν_a — self-absorption frequency
    ν_m — injection frequency
    ν_c — cooling frequency
    ν_max — maximum synchrotron frequency

These functions are intentionally **branch-free** and compute the SSA
frequency assuming that the corresponding spectral regime applies.

Regime consistency checks and final spectrum selection are handled
elsewhere in the closure logic.
"""

import numpy as np

from triceratops.radiation.constants import electron_rest_mass_cgs


# ---------------------------------------------------------------------
# Low-Level SSA Frequency Computations
# ---------------------------------------------------------------------
#
# Each function below evaluates the analytic expression for the
# self-absorption frequency ν_a assuming that a specific synchrotron
# spectral regime applies.
#
# The calculations are performed entirely in logarithmic CGS units.
def _log_nu_a_powerlaw_sbpl_sed_ssa_1(
    log_F_norm,
    log_omega,
    log_nu_m,
    p,
    log_gamma_m,
):
    r"""
    Compute the synchrotron self-absorption frequency :math:`\nu_a` for the first SSA regime.

    This regime corresponds to the ordering

    .. math::

        \nu_a < \nu_m ,

    where the spectrum is self-absorbed below the injection frequency.
    The absorption frequency is obtained by solving the SSA condition

    .. math::

        F_\nu(\nu_a)
        =
        \frac{2 m_e}{\Omega}
        \gamma_m
        \nu_a^2 ,

    using the optically thin normalization defined at :math:`\nu_m`.

    Parameters
    ----------
    log_F_norm : float or ndarray
        Natural logarithm of the optically thin flux normalization.

    log_omega : float or ndarray
        Natural logarithm of the angular source area :math:`\Omega`.

    log_nu_m : float or ndarray
        Natural logarithm of the injection frequency :math:`\nu_m`.

    p : float or ndarray
        Electron power-law index. Included for API consistency but
        not used in this regime.

    log_gamma_m : float or ndarray
        Natural logarithm of the minimum Lorentz factor
        :math:`\gamma_m`.

    Returns
    -------
    float or ndarray
        Natural logarithm of the self-absorption frequency
        :math:`\nu_a`.
    """
    _ = p
    log_Q = log_F_norm - np.log(2) - np.log(electron_rest_mass_cgs) - log_omega - log_gamma_m
    return (6 * log_Q / 13) + (log_nu_m / 13)


def _log_nu_a_powerlaw_sbpl_sed_ssa_2(
    log_F_norm,
    log_omega,
    log_nu_m,
    p,
    log_gamma_m,
):
    r"""
    Compute the synchrotron self-absorption frequency :math:`\nu_a` for the second SSA regime.

    This regime corresponds to the ordering

    .. math::

        \nu_m < \nu_a ,

    where the spectral peak occurs in the self-absorbed portion of the
    synchrotron spectrum.

    The absorption frequency is obtained by equating the optically thin
    synchrotron flux with the SSA blackbody limit,

    .. math::

        F_\nu(\nu_a)
        =
        \frac{2 m_e}{\Omega}
        \gamma_m
        \nu_a^2 ,

    while propagating the normalization from :math:`\nu_m` along the
    optically thin spectral slope :math:`F_\nu \propto \nu^{-(p-1)/2}`.

    Parameters
    ----------
    log_F_norm : float or ndarray
        Natural logarithm of the optically thin flux normalization.

    log_omega : float or ndarray
        Natural logarithm of the angular source area :math:`\Omega`.

    log_nu_m : float or ndarray
        Natural logarithm of the injection frequency :math:`\nu_m`.

    p : float or ndarray
        Electron power-law index :math:`p`.

    log_gamma_m : float or ndarray
        Natural logarithm of the minimum Lorentz factor
        :math:`\gamma_m`.

    Returns
    -------
    float or ndarray
        Natural logarithm of the synchrotron self-absorption frequency
        :math:`\nu_a`.
    """
    log_Q = log_F_norm - np.log(2) - np.log(electron_rest_mass_cgs) - log_omega - log_gamma_m
    return (2 * log_Q / (p + 4)) + (p * log_nu_m / (p + 4))


def _log_nu_a_powerlaw_sbpl_sed_ssa_cool_1(
    log_F_norm,
    log_omega,
    log_nu_m,
    p,
    log_gamma_m,
):
    r"""
    Self-absorption frequency for **Spectrum 1** (no cooling, optically thin peak).

    This regime corresponds to

    .. math::

        \nu_a < \nu_m < \nu_{\max}

    Parameters
    ----------
    log_F_norm : float
        Natural logarithm of the flux normalization.
    log_omega : float
        Natural logarithm of the effective emission solid angle.
    log_nu_m : float
        Natural logarithm of the injection frequency.
    p : float
        Electron power-law index.
    log_gamma_m : float
        Natural logarithm of the minimum Lorentz factor.

    Returns
    -------
    float
        Natural logarithm of the self-absorption frequency.
    """
    log_Q = log_F_norm - np.log(2) - np.log(electron_rest_mass_cgs) - log_omega - log_gamma_m
    return (2 * log_Q / (p + 4)) + (p * log_nu_m / (p + 4))


def _log_nu_a_powerlaw_sbpl_sed_ssa_cool_2(
    log_F_norm,
    log_omega,
    log_nu_m,
    p,
    log_gamma_m,
):
    r"""
    Self-absorption frequency for **Spectrum 2** (strong absorption).

    Corresponds to the regime

    .. math::

        \nu_m < \nu_a < \nu_{\max}
    """
    log_Q = log_F_norm - np.log(2) - np.log(electron_rest_mass_cgs) - log_omega - log_gamma_m
    return (2 * log_Q / (p + 4)) + (p * log_nu_m / (p + 4))


def _log_nu_a_powerlaw_sbpl_sed_ssa_cool_3(
    log_F_norm,
    log_omega,
    log_nu_m,
    log_gamma_m,
):
    r"""
    Self-absorption frequency for **Spectrum 3** (slow cooling, optically thin peak).

    Corresponds to

    .. math::

        \nu_a < \nu_m < \nu_c < \nu_{\max}
    """
    log_Q = log_F_norm - np.log(2) - np.log(electron_rest_mass_cgs) - log_omega - log_gamma_m
    return (3 * log_Q / 5) - (log_nu_m / 5)


def _log_nu_a_powerlaw_sbpl_sed_ssa_cool_4(
    log_F_norm,
    log_omega,
    log_nu_m,
    p,
    log_gamma_m,
):
    r"""
    Self-absorption frequency for **Spectrum 4** (slow cooling, moderate absorption).

    Corresponds to

    .. math::

        \nu_m < \nu_a < \nu_c < \nu_{\max}
    """
    log_Q = log_F_norm - np.log(2) - np.log(electron_rest_mass_cgs) - log_omega - log_gamma_m
    return (2 * log_Q / (p + 4)) + (p * log_nu_m / (p + 4))


def _log_nu_a_powerlaw_sbpl_sed_ssa_cool_5(
    log_F_norm,
    log_omega,
    log_nu_m,
    log_nu_c,
    log_gamma_m,
):
    r"""
    Self-absorption frequency for **Spectrum 5** (fast cooling, weak absorption).

    Corresponds to

    .. math::

        \nu_a < \nu_c < \nu_m < \nu_{\max}
    """
    log_Q = log_F_norm - np.log(2) - np.log(electron_rest_mass_cgs) - log_omega - log_gamma_m
    log_Q_c = log_Q + 0.5 * (log_nu_m - log_nu_c)

    return (6 * log_Q_c / 13) + (3 * log_nu_m / 13) - (2 * log_nu_c / 13)


def _log_nu_a_powerlaw_sbpl_sed_ssa_cool_6(
    log_F_norm,
    log_omega,
    log_nu_m,
    log_nu_c,
    log_gamma_m,
):
    r"""
    Self-absorption frequency for **Spectrum 6** (fast cooling, intermediate absorption).

    Corresponds to

    .. math::

        \nu_c < \nu_a < \nu_m < \nu_{\max}
    """
    log_Q = log_F_norm - np.log(2) - np.log(electron_rest_mass_cgs) - log_omega - log_gamma_m
    log_Q_c = log_Q + 0.5 * (log_nu_m - log_nu_c)

    return (log_Q_c / 3) + (log_nu_m / 6) + (log_nu_c / 6)


def _log_nu_a_powerlaw_sbpl_sed_ssa_cool_7(
    log_F_norm,
    log_omega,
    log_nu_m,
    log_nu_c,
    p,
    log_gamma_m,
):
    r"""
    Self-absorption frequency for **Spectrum 7** (extreme absorption).

    This is functionally identical
    to the same case with fast cooling below the absorption break, but is differentiated for clarity.

    Corresponds to

    .. math::

        \nu_m < \nu_c < \nu_a < \nu_{\max}
    """
    log_Q = log_F_norm - np.log(2) - np.log(electron_rest_mass_cgs) - log_omega - log_gamma_m

    return (2 * log_Q / (5 + p)) + (log_nu_c / (5 + p)) + (p * log_nu_m / (5 + p))


def _log_nu_a_powerlaw_sbpl_sed_ssa_cool_8(
    log_F_norm,
    log_omega,
    log_nu_m,
    log_nu_c,
    p,
    log_gamma_m,
):
    r"""
    Self-absorption frequency for **Spectrum 8** (extreme absorption).

    This is functionally identical
    to the same case with slow cooling below the absorption break, but is differentiated for clarity.

    Corresponds to

    .. math::

        \nu_c < \nu_m < \nu_a < \nu_{\max}
    """
    log_Q = log_F_norm - np.log(2) - np.log(electron_rest_mass_cgs) - log_omega - log_gamma_m

    return (2 * log_Q / (5 + p)) + (log_nu_c / (5 + p)) + (p * log_nu_m / (5 + p))


# ---------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------
SSA_FREQUENCY_W_COOLING_FUNCTION_REGISTRY = {
    "Spectrum1": _log_nu_a_powerlaw_sbpl_sed_ssa_cool_1,
    "Spectrum2": _log_nu_a_powerlaw_sbpl_sed_ssa_cool_2,
    "Spectrum3": _log_nu_a_powerlaw_sbpl_sed_ssa_cool_3,
    "Spectrum4": _log_nu_a_powerlaw_sbpl_sed_ssa_cool_4,
    "Spectrum5": _log_nu_a_powerlaw_sbpl_sed_ssa_cool_5,
    "Spectrum6": _log_nu_a_powerlaw_sbpl_sed_ssa_cool_6,
    "Spectrum7": _log_nu_a_powerlaw_sbpl_sed_ssa_cool_7,
    "Spectrum8": _log_nu_a_powerlaw_sbpl_sed_ssa_cool_8,
}
"""
Mapping between spectral regimes and the corresponding SSA
frequency evaluation functions.

Each function computes the candidate value of the self-absorption
frequency assuming that the corresponding regime applies.

Final regime selection must be performed by the closure logic.
"""
SSA_FREQUENCY_NO_COOLING_FUNCTION_REGISTRY = {
    "optically_thin": _log_nu_a_powerlaw_sbpl_sed_ssa_cool_1,
    "optically_thick": _log_nu_a_powerlaw_sbpl_sed_ssa_cool_2,
}
"""Mapping between spectral regimes and the corresponding SSA"""


# ---------------------------------------------------------------------
# DICTIONARY GENERATORS
# ---------------------------------------------------------------------
def compute_ssa_frequencies_with_cooling(
    log_F_norm: float,
    log_omega: float,
    log_nu_m: float,
    log_nu_c: float,
    log_nu_max: float,
    log_gamma_m: float,
    p: float,
) -> tuple[dict[str, float], int]:
    r"""
    Compute candidate synchrotron self-absorption frequencies.

    This function evaluates the analytic SSA frequency expressions
    corresponding to all spectral regimes compatible with the cooling
    state of the system.

    Parameters
    ----------
    log_F_norm : float
        Natural logarithm of the spectral flux normalization.
    log_omega : float
        Natural logarithm of the emission solid angle.
    log_nu_m : float
        Natural logarithm of the injection frequency :math:`\nu_m`.
    log_nu_c : float
        Natural logarithm of the cooling frequency :math:`\nu_c`.
    log_nu_max : float
        Natural logarithm of the maximum synchrotron frequency :math:`\nu_{\max}`.
    log_gamma_m : float
        Natural logarithm of the minimum electron Lorentz factor.
    p : float
        Electron power-law index.

    Returns
    -------
    tuple
        ``(log_nu_a_dict, cooling_regime)`` where

        log_nu_a_dict : dict
            Mapping of candidate spectral regimes to
            :math:`\log \nu_a`.

        cooling_regime : int
            Cooling classification flag

            * ``0`` → fast cooling
            * ``1`` → slow cooling
            * ``2`` → no cooling

    Notes
    -----
    These SSA frequencies are **candidate values** computed under the
    assumption that the corresponding regime applies.

    Final regime selection must be performed using the regime
    determination routine.
    """
    # -------------------------------------------------
    # FAST COOLING
    # ν_c < ν_m
    # -------------------------------------------------
    if log_nu_c < log_nu_m:
        return {
            "Spectrum5": _log_nu_a_powerlaw_sbpl_sed_ssa_cool_5(log_F_norm, log_omega, log_nu_m, log_nu_c, log_gamma_m),
            "Spectrum6": _log_nu_a_powerlaw_sbpl_sed_ssa_cool_6(log_F_norm, log_omega, log_nu_m, log_nu_c, log_gamma_m),
            "Spectrum8": _log_nu_a_powerlaw_sbpl_sed_ssa_cool_8(
                log_F_norm, log_omega, log_nu_m, log_nu_c, p, log_gamma_m
            ),
        }, 0

    # -------------------------------------------------
    # SLOW COOLING
    # ν_m < ν_c < ν_max
    # -------------------------------------------------
    elif log_nu_c < log_nu_max:
        return {
            "Spectrum3": _log_nu_a_powerlaw_sbpl_sed_ssa_cool_3(log_F_norm, log_omega, log_nu_m, log_gamma_m),
            "Spectrum4": _log_nu_a_powerlaw_sbpl_sed_ssa_cool_4(log_F_norm, log_omega, log_nu_m, p, log_gamma_m),
            "Spectrum7": _log_nu_a_powerlaw_sbpl_sed_ssa_cool_7(
                log_F_norm, log_omega, log_nu_m, log_nu_c, p, log_gamma_m
            ),
        }, 1

    # -------------------------------------------------
    # NO COOLING
    # ν_c > ν_max
    # -------------------------------------------------
    else:
        return {
            "Spectrum1": _log_nu_a_powerlaw_sbpl_sed_ssa_cool_1(log_F_norm, log_omega, log_nu_m, p, log_gamma_m),
            "Spectrum2": _log_nu_a_powerlaw_sbpl_sed_ssa_cool_2(log_F_norm, log_omega, log_nu_m, p, log_gamma_m),
        }, 2


def compute_ssa_frequencies_without_cooling(
    log_F_norm: float,
    log_omega: float,
    log_nu_m: float,
    log_gamma_m: float,
    p: float,
) -> dict[str, float]:
    r"""
    Compute candidate synchrotron self-absorption frequencies in the absence of cooling.

    This function evaluates the analytic SSA frequency expressions
    corresponding to the two spectral regimes compatible with the
    absence of cooling.

    Parameters
    ----------
    log_F_norm : float
        Natural logarithm of the spectral flux normalization.
    log_omega : float
        Natural logarithm of the emission solid angle.
    log_nu_m : float
        Natural logarithm of the injection frequency :math:`\nu_m`.
    log_gamma_m : float
        Natural logarithm of the minimum electron Lorentz factor.
    p: float
        Electron power-law index.

    Returns
    -------
    dict
        Mapping of candidate spectral regimes to :math:`\log \nu_a`.

    Notes
    -----
    These SSA frequencies are **candidate values** computed under the
    assumption that the corresponding regime applies.

    Final regime selection must be performed using the regime
    determination routine.
    """
    return {
        "optically_thin": _log_nu_a_powerlaw_sbpl_sed_ssa_cool_1(
            log_F_norm, log_omega, log_nu_m, p=p, log_gamma_m=log_gamma_m
        ),
        "optically_thick": _log_nu_a_powerlaw_sbpl_sed_ssa_cool_2(
            log_F_norm, log_omega, log_nu_m, p=p, log_gamma_m=log_gamma_m
        ),
    }


# ---------------------------------------------------------------------
# REGIME SELECTION
# ---------------------------------------------------------------------
def select_ssa_sed_regime_from_candidates_with_cooling(
    log_nu_ssa: dict[str, float],
    cooling_regime: int,
    log_nu_m: float,
    log_nu_c: float,
    log_nu_max: float,
) -> tuple[str, float]:
    """
    Determine the physically consistent SSA synchrotron spectral regime.

    This function selects the unique spectral regime whose candidate
    self-absorption frequency satisfies the required ordering relative
    to the characteristic synchrotron break frequencies.

    Parameters
    ----------
    log_nu_ssa : dict[str, float]
        Candidate SSA frequencies computed under the assumption that
        each spectral regime applies.
    cooling_regime : int
        Cooling classification flag

        * ``0`` — fast cooling (:math:`ν_c < ν_m`)
        * ``1`` — slow cooling (:math:`ν_m < ν_c < ν_{max}`)
        * ``2`` — no cooling (:math:`ν_c > ν_{max}`)
    log_nu_m : float
        Log injection frequency.
    log_nu_c : float
        Log cooling frequency.
    log_nu_max : float
        Log maximum synchrotron frequency.

    Returns
    -------
    tuple[str, float]
        ``(regime, log_nu_a)`` where

        * ``regime`` is the selected spectral regime identifier
        * ``log_nu_a`` is the SSA frequency consistent with that regime.

    Raises
    ------
    RuntimeError
        If none of the candidate regimes satisfy the required ordering.
    """
    # -------------------------------------------------
    # FAST COOLING
    # ν_c < ν_m
    # -------------------------------------------------
    if cooling_regime == 0:
        s5 = log_nu_ssa["Spectrum5"]
        s6 = log_nu_ssa["Spectrum6"]
        s8 = log_nu_ssa["Spectrum8"]

        if s5 < log_nu_c:
            return "Spectrum5", s5

        if s8 > log_nu_m:
            return "Spectrum8", s8

        if s6 < log_nu_m:
            return "Spectrum6", s6

        raise RuntimeError(
            "Unable to determine SSA regime in fast-cooling case.\n"
            f"log_nu_c={log_nu_c:.3f}, log_nu_m={log_nu_m:.3f}, log_nu_max={log_nu_max:.3f}\n"
            f"S5={s5:.3f}, S6={s6:.3f}, S8={s8:.3f}"
        )

    # -------------------------------------------------
    # SLOW COOLING
    # ν_m < ν_c < ν_max
    # -------------------------------------------------
    if cooling_regime == 1:
        s3 = log_nu_ssa["Spectrum3"]
        s4 = log_nu_ssa["Spectrum4"]
        s7 = log_nu_ssa["Spectrum7"]

        if s3 < log_nu_m:
            return "Spectrum3", s3

        if s4 < log_nu_c:
            return "Spectrum4", s4

        if s7 > log_nu_c:
            return "Spectrum7", s7

        raise RuntimeError(
            "Unable to determine SSA regime in slow-cooling case.\n"
            f"log_nu_m={log_nu_m:.3f}, log_nu_c={log_nu_c:.3f}, log_nu_max={log_nu_max:.3f}\n"
            f"S3={s3:.3f}, S4={s4:.3f}, S7={s7:.3f}"
        )

    # -------------------------------------------------
    # NO COOLING
    # ν_c > ν_max
    # -------------------------------------------------
    if cooling_regime == 2:
        s1 = log_nu_ssa["Spectrum1"]
        s2 = log_nu_ssa["Spectrum2"]

        if s1 < log_nu_m:
            return "Spectrum1", s1

        if s2 > log_nu_m:
            return "Spectrum2", s2

        raise RuntimeError(
            "Unable to determine SSA regime in no-cooling case.\n"
            f"log_nu_m={log_nu_m:.3f}, log_nu_max={log_nu_max:.3f}\n"
            f"S1={s1:.3f}, S2={s2:.3f}"
        )

    raise RuntimeError(f"Unrecognized cooling regime: {cooling_regime}")


def select_ssa_sed_regime_from_candidates_without_cooling(
    log_nu_ssa: dict[str, float],
    log_nu_m: float,
) -> tuple[str, float]:
    r"""
    Select the physically consistent SSA synchrotron spectral regime in the absence of radiative cooling.

    Two candidate self-absorption frequencies are computed assuming
    each possible spectral ordering. The physically valid regime is
    determined by enforcing the required ordering between the
    self-absorption frequency :math:`\nu_a` and the injection frequency
    :math:`\nu_m`.

    The allowed regimes are

    * ``"optically_thin"`` — :math:`\nu_a < \nu_m`
    * ``"optically_thick"`` — :math:`\nu_a > \nu_m`

    Parameters
    ----------
    log_nu_ssa : dict[str, float]
        Dictionary of candidate self-absorption frequencies
        computed under each regime assumption. Expected keys are

        * ``"optically_thin"``
        * ``"optically_thick"``

    log_nu_m : float
        Natural logarithm of the injection frequency :math:`\nu_m`.

    Returns
    -------
    tuple[str, float]
        ``(regime, log_nu_a)`` where

        * ``regime`` is the selected spectral regime identifier
        * ``log_nu_a`` is the SSA frequency consistent with that regime.

    Raises
    ------
    RuntimeError
        If neither candidate frequency satisfies the required ordering.
    """
    log_nu_a_thin = log_nu_ssa["optically_thin"]
    log_nu_a_thick = log_nu_ssa["optically_thick"]

    # ν_a < ν_m
    if log_nu_a_thin < log_nu_m:
        return "optically_thin", log_nu_a_thin

    # ν_a > ν_m
    if log_nu_a_thick > log_nu_m:
        return "optically_thick", log_nu_a_thick

    raise RuntimeError(
        "Unable to determine SSA regime in the no-cooling case.\n"
        f"log_nu_m = {log_nu_m}\n"
        f"candidate ν_a (thin)  = {log_nu_a_thin}\n"
        f"candidate ν_a (thick) = {log_nu_a_thick}"
    )
