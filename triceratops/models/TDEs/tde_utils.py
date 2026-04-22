"""
Utility functions specific to modeling TDEs.

These functions and classes are intended to provide convenient features / interfaces either within
the TDE modeling framework or for users to set up and run their TDE models with good initial conditions and
parameter choices.
"""

from typing import Union

import numpy as np
from astropy import units as u

from triceratops.utils.misc_utils import ensure_in_units


# ================================================================ #
# Low-Level (Private) API
# ================================================================ #
# These functions are intended for internal use within performance-critical code paths.
def _log_mdot_peak_guillochon13(
    log_M_bh: float,
    log_M_star: float,
    log_R_star: float,
    beta: float = 1.0,
    gamma: float = 5 / 3,
):
    beta = np.asarray(beta)
    log_M_bh = np.asarray(log_M_bh)
    log_M_star = np.asarray(log_M_star)
    log_R_star = np.asarray(log_R_star)

    # Use the fitting formula from Guillochon & Ramirez-Ruiz (2015) to compute the peak mass fallback rate.
    if np.isclose(gamma, 5 / 3, rtol=1e-8):
        log_A = (10.253 - (17.380 * beta) + (5.9988 * beta**2)) / (1 - (0.46573 * beta) - (4.5066 * beta**2))
    elif np.isclose(gamma, 4 / 3, rtol=1e-8):
        log_A = (27.261 - (27.516 * beta) + (3.8716 * beta**2)) / (1 - (3.2605 * beta) - (1.3865 * beta**2))
    else:
        raise ValueError("Unsupported gamma value. Only 5/3 and 4/3 are supported.")

    # Everything is input in CGS units, so we need to get the relative parameters sorted out nicely.
    log_mdot_peak = (
        log_A - 0.5 * (log_M_bh - 90.487947735) + 2 * (log_M_star - 76.672437177) - 1.5 * (log_R_star - 24.9655992769)
    )

    return log_mdot_peak if np.ndim(log_mdot_peak) > 0 else float(log_mdot_peak)


def _log_t_peak_guillochon13(
    log_M_bh: float,
    log_M_star: float,
    log_R_star: float,
    beta: float = 1.0,
    gamma: float = 5 / 3,
):
    beta = np.asarray(beta)
    log_M_bh = np.asarray(log_M_bh)
    log_M_star = np.asarray(log_M_star)
    log_R_star = np.asarray(log_R_star)

    # Use the fitting formula from Guillochon & Ramirez-Ruiz (2015) to compute the time of peak fallback.
    if np.isclose(gamma, 5 / 3, rtol=1e-8):
        log_B = (-0.30908 + (1.1804 * beta**0.5) - (1.1764 * beta)) / (1 + (1.3089 * beta**0.5) - (4.1940 * beta))
    elif np.isclose(gamma, 4 / 3, rtol=1e-8):
        log_B = (-0.38670 + (0.57291 * beta**0.5) - (0.31231 * beta)) / (1 - (1.2744 * beta**0.5) - (0.90053 * beta))
    else:
        raise ValueError("Unsupported gamma value. Only 5/3 and 4/3 are supported.")

    # Everything is input in CGS units, so we need to get the relative parameters sorted out nicely.
    log_t_peak = (
        log_B + 0.5 * (log_M_bh - 90.487947735) - 1 * (log_M_star - 76.672437177) + 1.5 * (log_R_star - 24.9655992769)
    )

    return log_t_peak if np.ndim(log_t_peak) > 0 else float(log_t_peak)


def _log_delta_M_guillochon13(
    log_M_star: float,
    beta: float = 1.0,
    gamma: float = 5 / 3,
):
    beta = np.asarray(beta)
    log_M_star = np.asarray(log_M_star)

    # Use the fitting formula from Guillochon & Ramirez-Ruiz (2015) to compute the time of peak fallback.
    if np.isclose(gamma, 5 / 3, rtol=1e-8):
        log_C = (3.1647 - (6.3777 * beta) + (3.1797 * beta**2)) / (1 - (3.4137 * beta) + (2.4616 * beta**2))
    elif np.isclose(gamma, 4 / 3, rtol=1e-8):
        log_C = (12.996 - (31.149 * beta) - (12.865 * beta**2)) / (1 - (5.3232 * beta) - (6.4262 * beta**2))
    else:
        raise ValueError("Unsupported gamma value. Only 5/3 and 4/3 are supported.")

    log_delta_m = log_C + (log_M_star - 76.672437177)

    return log_delta_m if np.ndim(log_delta_m) > 0 else float(log_delta_m)


def _log_n_index_guillochon13(
    beta: float = 1.0,
    gamma: float = 5 / 3,
):
    beta = np.asarray(beta)

    # Use the fitting formula from Guillochon & Ramirez-Ruiz (2015) to compute the time of peak fallback.
    if np.isclose(gamma, 5 / 3, rtol=1e-8):
        D = (-0.93653 + (11.109 * beta) - (38.161 * beta**2) + (50.418 * beta**3) - 22.965 * beta**4) / (
            1 - (2.3585 * beta) + (0.47593 * beta**2) + (0.96280 * beta**3) - (0.37996 * beta**4)
        )
    elif np.isclose(gamma, 4 / 3, rtol=1e-8):
        D = (-2.7332 + (6.94565 * beta) - (3.2743 * beta**2) - (0.84659 * beta**3) + 0.56254 * beta**4) / (
            1 - (2.3585 * beta) + (0.47593 * beta**2) + (0.96280 * beta**3) - (0.37996 * beta**4)
        )
    else:
        raise ValueError("Unsupported gamma value. Only 5/3 and 4/3 are supported.")

    return D if np.ndim(D) > 0 else float(D)


# ================================================================ #
# Public API
# ================================================================ #
def compute_mdot_peak_guillochon13(
    M_bh: Union[float, np.ndarray, u.Quantity],
    M_star: Union[float, np.ndarray, u.Quantity],
    R_star: Union[float, np.ndarray, u.Quantity],
    beta: float = 1.0,
    gamma: float = 5 / 3,
) -> u.Quantity:
    r"""Compute the peak mass fallback rate for a tidal disruption event.

    Uses the fitting formulae of :footcite:t:`2013ApJ...767...25G` (Appendix) to evaluate
    the peak rate at which disrupted stellar debris returns to pericentre as a
    function of black hole mass, stellar properties, and the penetration parameter.

    Parameters
    ----------
    M_bh : float, array-like, or astropy.units.Quantity
        Black hole mass.  Default units are grams.
    M_star : float, array-like, or astropy.units.Quantity
        Stellar mass.  Default units are grams.
    R_star : float, array-like, or astropy.units.Quantity
        Stellar radius.  Default units are cm.
    beta : float, optional
        Penetration parameter :math:`\beta = r_t / r_p`, the ratio of the tidal radius
        to the pericentre distance.  Default is ``1.0``.
    gamma : float, optional
        Stellar polytropic index.  Must be ``5/3`` or ``4/3``.  Default is ``5/3``.

    Returns
    -------
    mdot_peak : astropy.units.Quantity
        Peak mass fallback rate in :math:`M_\odot\,\mathrm{yr}^{-1}`.

    Notes
    -----
    The peak fallback rate is parameterised as

    .. math::

        \dot{M}_\mathrm{peak}\,\left[\frac{M_\odot}{\mathrm{yr}}\right] =
            A(\beta,\,\gamma)\,
            \left(\frac{M_\mathrm{bh}}{10^6\,M_\odot}\right)^{-1/2}
            \left(\frac{M_\star}{M_\odot}\right)^{2}
            \left(\frac{R_\star}{R_\odot}\right)^{-3/2}

    where the dimensionless amplitude :math:`A(\beta,\gamma)` is given by
    rational-polynomial fits to hydrodynamical simulations
    (:footcite:t:`2013ApJ...767...25G`, Appendix, Table A1).

    References
    ----------
    .. footbibliography::
    """
    M_bh_cgs = ensure_in_units(M_bh, u.g)
    M_star_cgs = ensure_in_units(M_star, u.g)
    R_star_cgs = ensure_in_units(R_star, u.cm)
    log_mdot = _log_mdot_peak_guillochon13(np.log(M_bh_cgs), np.log(M_star_cgs), np.log(R_star_cgs), beta, gamma)
    return np.exp(log_mdot) * u.M_sun / u.yr


def compute_t_peak_guillochon13(
    M_bh: Union[float, np.ndarray, u.Quantity],
    M_star: Union[float, np.ndarray, u.Quantity],
    R_star: Union[float, np.ndarray, u.Quantity],
    beta: float = 1.0,
    gamma: float = 5 / 3,
) -> u.Quantity:
    r"""Compute the time to peak mass fallback rate for a tidal disruption event.

    Uses the fitting formulae of :footcite:t:`2013ApJ...767...25G` (Appendix) to evaluate
    the time elapsed between the disruption and the peak of the fallback rate.

    Parameters
    ----------
    M_bh : float, array-like, or astropy.units.Quantity
        Black hole mass.  Default units are grams.
    M_star : float, array-like, or astropy.units.Quantity
        Stellar mass.  Default units are grams.
    R_star : float, array-like, or astropy.units.Quantity
        Stellar radius.  Default units are cm.
    beta : float, optional
        Penetration parameter :math:`\beta = r_t / r_p`.  Default is ``1.0``.
    gamma : float, optional
        Stellar polytropic index.  Must be ``5/3`` or ``4/3``.  Default is ``5/3``.

    Returns
    -------
    t_peak : astropy.units.Quantity
        Time to peak fallback rate in years.

    Notes
    -----
    The time to peak is parameterised as

    .. math::

        t_\mathrm{peak} = B(\beta,\,\gamma)\,
            \left(\frac{M_\mathrm{bh}}{10^6\,M_\odot}\right)^{1/2}
            \left(\frac{M_\star}{M_\odot}\right)^{-1}
            \left(\frac{R_\star}{R_\odot}\right)^{3/2}

    where :math:`B(\beta,\gamma)` is given by rational-polynomial fits
    (:footcite:t:`2013ApJ...767...25G`, Appendix, Table A1).

    References
    ----------
    .. footbibliography::
    """
    M_bh_cgs = ensure_in_units(M_bh, u.g)
    M_star_cgs = ensure_in_units(M_star, u.g)
    R_star_cgs = ensure_in_units(R_star, u.cm)
    log_t = _log_t_peak_guillochon13(np.log(M_bh_cgs), np.log(M_star_cgs), np.log(R_star_cgs), beta, gamma)
    return np.exp(log_t) * u.yr


def compute_delta_M_guillochon13(
    M_star: Union[float, np.ndarray, u.Quantity],
    beta: float = 1.0,
    gamma: float = 5 / 3,
) -> u.Quantity:
    r"""Compute the total mass stripped from the star in a tidal disruption event.

    Uses the fitting formulae of :footcite:t:`2013ApJ...767...25G` (Appendix) to evaluate
    the total mass that becomes unbound from the star and falls back to the black hole.
    For full disruptions (:math:`\beta` above the critical value) this equals
    :math:`M_\star / 2`; for partial disruptions it is less.

    Parameters
    ----------
    M_star : float, array-like, or astropy.units.Quantity
        Stellar mass.  Default units are grams.
    beta : float, optional
        Penetration parameter :math:`\beta = r_t / r_p`.  Default is ``1.0``.
    gamma : float, optional
        Stellar polytropic index.  Must be ``5/3`` or ``4/3``.  Default is ``5/3``.

    Returns
    -------
    delta_M : astropy.units.Quantity
        Total accreted mass in :math:`M_\odot`.

    Notes
    -----
    The stripped mass is parameterised as

    .. math::

        \Delta M = C(\beta,\,\gamma)\,\frac{M_\star}{M_\odot}

    where :math:`C(\beta,\gamma)` is given by rational-polynomial fits
    (:footcite:t:`2013ApJ...767...25G`, Appendix, Table A1).

    References
    ----------
    .. footbibliography::
    """
    M_star_cgs = ensure_in_units(M_star, u.g)
    log_delta_M = _log_delta_M_guillochon13(np.log(M_star_cgs), beta, gamma)
    return np.exp(log_delta_M) * u.M_sun


def compute_n_index_guillochon13(
    beta: float = 1.0,
    gamma: float = 5 / 3,
) -> Union[float, np.ndarray]:
    r"""Compute the late-time power-law index of the mass fallback rate.

    Uses the fitting formulae of :footcite:t:`2013ApJ...767...25G` (Appendix) to evaluate
    the asymptotic exponent :math:`n` such that
    :math:`\dot{M}(t) \propto t^n` for :math:`t \gg t_\mathrm{peak}`.
    The classical Rees (1988) value is :math:`n = -5/3`.

    Parameters
    ----------
    beta : float, optional
        Penetration parameter :math:`\beta = r_t / r_p`.  Default is ``1.0``.
    gamma : float, optional
        Stellar polytropic index.  Must be ``5/3`` or ``4/3``.  Default is ``5/3``.

    Returns
    -------
    n : float or ndarray
        Dimensionless power-law index of the late-time fallback rate.

    Notes
    -----
    The late-time fallback rate follows

    .. math::

        \dot{M}(t) \propto t^{n(\beta,\,\gamma)}

    where :math:`n(\beta,\gamma)` is given by rational-polynomial fits
    (:footcite:t:`2013ApJ...767...25G`, Appendix, Table A1).  For a full disruption
    of a :math:`\gamma = 5/3` polytrope at :math:`\beta = 1`, the result
    approaches the classical :math:`n = -5/3`.

    References
    ----------
    .. footbibliography::
    """
    return _log_n_index_guillochon13(beta, gamma)
