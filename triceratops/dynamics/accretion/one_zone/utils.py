r"""Accretion physics utility functions for one-zone disk models.

This module provides several levels of functionality:

1. **State conversion** — :func:`disk_state` converts any two of
   :math:`(M_D, R_D, J_D, \Sigma_D)` into the full disk state, including
   :math:`\Omega`.

2. **Heating/cooling curves** — :func:`compute_heating_curves` and
   :func:`compute_advective_heating_curves` evaluate the heating and cooling
   rates as functions of temperature for fixed disk parameters, returning
   individual rates as :class:`~astropy.units.Quantity` arrays.

3. **S-curves** — :func:`compute_advection_s_curve` traces the full thermal
   equilibrium locus in :math:`\Sigma`–:math:`T` space.

All public functions accept :class:`~astropy.units.Quantity` inputs and return
:class:`~astropy.units.Quantity` outputs.  Private functions (prefixed ``_``)
work directly in CGS floats.

Geometry
--------
The one-zone disk uses the Metzger+08 self-similar geometry constants:

.. math::

    A = 1.62, \quad B = 1.33, \quad \xi = B/A \approx 0.821

such that:

.. math::

    J_D = \xi\,M_D\,\sqrt{G\,M_{\rm BH}\,R_D}, \qquad
    \Sigma_D = \frac{M_D}{\pi\,A\,R_D^2}

Physics reference
-----------------
Heating/cooling balance:

.. math::

    \begin{aligned}
    q^+ &= \tfrac{9}{8}\,\alpha\,\Sigma\,\Omega\,c_s^2(T) \\
    q^- &= \frac{4\,\sigma_{\rm SB}\,T^4}{3\,\kappa\,\Sigma} \\
    q_{\rm adv} &= \frac{3}{2}\,\xi\,\alpha\,\Sigma\,\frac{c_s^4}{\Omega\,R^2}
    \end{aligned}

See Also
--------
:mod:`triceratops.physics_utils.eos` : sound-speed implementations.
:class:`~triceratops.dynamics.accretion.one_zone.core.FullPressureDisk` : model
    class whose Cython closure implements the same balance.
"""

from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from astropy import constants as const
from astropy import units as u
from scipy.optimize import brentq
from tqdm.auto import tqdm

from triceratops.physics_utils.eos import _log_radiative_ideal_gas_disk_sound_speed
from triceratops.radiation.opacity.base import GreyOpacityLaw
from triceratops.radiation.opacity.utils import get_opacity
from triceratops.utils.misc_utils import ensure_in_units

if TYPE_CHECKING:
    from triceratops._typing import _UnitBearingArrayLike, _UnitBearingScalarLike

# ================================================================ #
#  Physical constants (CGS)                                        #
# ================================================================ #
_SIGMA_SB: float = const.sigma_sb.cgs.value  # erg cm⁻² s⁻¹ K⁻⁴
_G_CGS: float = const.G.cgs.value  # cm³ g⁻¹ s⁻²
_DISK_F0: float = 1.6  # Metzger+08 disk correction factor (dimensionless)
_DISK_A: float = 1.62  # Metzger+08 geometry constant (dimensionless)
_DISK_B: float = 1.33  # Metzger+08 geometry constant (dimensionless)
_DISK_XI: float = _DISK_B / _DISK_A  # ≈ 0.821


# ================================================================ #
#  Low-level CGS helpers                                           #
# ================================================================ #


def _log_heating_curves(
    log_T: np.ndarray,
    log_Sigma: float,
    log_Omega: float,
    alpha: float,
    mu: float,
    opacity_obj: GreyOpacityLaw,
):
    r"""Viscous heating and radiative cooling rates in log-space (CGS).

    Parameters
    ----------
    log_T : array-like
        Natural log of midplane temperature (K).
    log_Sigma : float
        Natural log of surface density (g cm⁻²).
    log_Omega : float
        Natural log of Keplerian angular velocity (rad s⁻¹).
    alpha : float
        Shakura–Sunyaev viscosity parameter.
    mu : float
        Mean molecular weight.
    opacity_obj : GreyOpacityLaw
        Opacity law instance.

    Returns
    -------
    log_q_visc, log_q_rad : ndarray
        Natural logs of viscous heating and radiative cooling rates
        (erg cm⁻² s⁻¹).
    """
    log_T = np.asarray(log_T, dtype=float)
    log_cs = _log_radiative_ideal_gas_disk_sound_speed(log_T, mu, log_Sigma, log_Omega)
    log_rho = log_Sigma + log_Omega - log_cs - 0.5 * np.log(2.0 * np.pi)
    log_kappa = opacity_obj._log_opacity(log_T, log_rho)

    log_q_visc = np.log(9.0 / 8.0) + np.log(alpha) + log_Sigma + log_Omega + 2.0 * log_cs
    log_q_rad = np.log(4.0 * _SIGMA_SB / 3.0) + 4.0 * log_T - log_kappa - log_Sigma

    return log_q_visc, log_q_rad


def _log_advective_heating_curves(
    log_T: np.ndarray,
    log_Sigma: float,
    log_Omega: float,
    log_R: float,
    alpha: float,
    mu: float,
    xi: float,
    opacity_obj: GreyOpacityLaw,
):
    r"""Viscous heating, radiative cooling, and advective cooling rates in log-space (CGS).

    Parameters
    ----------
    log_T : array-like
        Natural log of midplane temperature (K).
    log_Sigma : float
        Natural log of surface density (g cm⁻²).
    log_Omega : float
        Natural log of Keplerian angular velocity (rad s⁻¹).
    log_R : float
        Natural log of disk radius (cm).
    alpha : float
        Shakura–Sunyaev viscosity parameter.
    mu : float
        Mean molecular weight.
    xi : float
        Entropy gradient parameter (dimensionless, > 0).
    opacity_obj : GreyOpacityLaw
        Opacity law instance.

    Returns
    -------
    log_q_visc, log_q_rad, log_q_adv : ndarray
        Natural logs of viscous heating, radiative cooling, and advective
        cooling rates (erg cm⁻² s⁻¹).
    """
    log_T = np.asarray(log_T, dtype=float)
    log_cs = _log_radiative_ideal_gas_disk_sound_speed(log_T, mu, log_Sigma, log_Omega)
    log_rho = log_Sigma + log_Omega - log_cs - 0.5 * np.log(2.0 * np.pi)
    log_kappa = opacity_obj._log_opacity(log_T, log_rho)

    log_q_visc = np.log(9.0 / 8.0) + np.log(alpha) + log_Sigma + log_Omega + 2.0 * log_cs
    log_q_rad = np.log(4.0 * _SIGMA_SB / 3.0) + 4.0 * log_T - log_kappa - log_Sigma
    log_q_adv = np.log(3.0 * xi * alpha / 2.0) + log_Sigma - log_Omega - 2.0 * log_R + 4.0 * log_cs

    return log_q_visc, log_q_rad, log_q_adv


def _log_heating_residual(
    log_T: float,
    log_Sigma: float,
    log_Omega: float,
    alpha: float,
    mu: float,
    opacity_obj: GreyOpacityLaw,
) -> float:
    r"""Thermal residual :math:`q^+ - q^-` (erg cm⁻² s⁻¹) for a non-advective disk.

    Positive when the disk is over-heated; zero at thermal equilibrium.
    """
    log_q_visc, log_q_rad = _log_heating_curves(log_T, log_Sigma, log_Omega, alpha, mu, opacity_obj)
    return float(np.exp(log_q_visc) - np.exp(log_q_rad))


def _log_advective_heating_residual(
    log_T: float,
    log_Sigma: float,
    log_Omega: float,
    log_R: float,
    alpha: float,
    mu: float,
    xi: float,
    opacity_obj: GreyOpacityLaw,
) -> float:
    r"""Thermal residual :math:`q^+ - q^- - q_{\rm adv}` (erg cm⁻² s⁻¹).

    Positive when the disk is over-heated; zero at thermal equilibrium.
    """
    log_q_visc, log_q_rad, log_q_adv = _log_advective_heating_curves(
        log_T, log_Sigma, log_Omega, log_R, alpha, mu, xi, opacity_obj
    )
    return float(np.exp(log_q_visc) - np.exp(log_q_rad) - np.exp(log_q_adv))


# ================================================================ #
#  Disk state conversion                                           #
# ================================================================ #


def disk_state(
    M_BH: "_UnitBearingScalarLike",
    M_D: "_UnitBearingScalarLike" = None,
    R_D: "_UnitBearingScalarLike" = None,
    J_D: "_UnitBearingScalarLike" = None,
    Sigma_D: "_UnitBearingScalarLike" = None,
) -> dict:
    r"""Compute the full one-zone disk state from any two of the four disk variables.

    Given the black hole mass and exactly **two** of :math:`(M_D, R_D, J_D, \Sigma_D)`,
    this function infers the remaining two variables and the Keplerian angular velocity
    :math:`\Omega` using the Metzger+08 one-zone geometry:

    .. math::

        J_D = \xi\,M_D\,\sqrt{G\,M_{\rm BH}\,R_D}, \qquad
        \Sigma_D = \frac{M_D}{\pi\,A\,R_D^2}, \qquad
        \Omega = \sqrt{\frac{G\,M_{\rm BH}}{R_D^3}}

    with :math:`A = 1.62`, :math:`B = 1.33`, :math:`\xi = B/A \approx 0.821`.

    Parameters
    ----------
    M_BH : Quantity [mass]
        Black hole mass.
    M_D : Quantity [mass], optional
        Disk mass.
    R_D : Quantity [length], optional
        Characteristic disk radius.
    J_D : Quantity [angular momentum], optional
        Disk angular momentum.
    Sigma_D : Quantity [surface density], optional
        Disk surface density.

    Returns
    -------
    dict
        Dictionary with keys ``"M_D"``, ``"R_D"``, ``"J_D"``, ``"Sigma_D"``,
        ``"Omega"``, all as :class:`~astropy.units.Quantity` in CGS units.

    Raises
    ------
    ValueError
        If not exactly two of the four disk variables are provided.
    """
    provided = {
        "M_D": M_D,
        "R_D": R_D,
        "J_D": J_D,
        "Sigma_D": Sigma_D,
    }
    given = {k: v for k, v in provided.items() if v is not None}
    if len(given) != 2:
        raise ValueError(f"Exactly 2 of (M_D, R_D, J_D, Sigma_D) must be provided; got {list(given.keys())}.")

    # Work in CGS floats internally.
    MBH = ensure_in_units(M_BH, u.g)
    G = _G_CGS
    xi = _DISK_XI
    A = _DISK_A

    def _cgs(val, unit):
        return ensure_in_units(val, unit)

    keys = set(given.keys())

    if keys == {"M_D", "R_D"}:
        M = _cgs(M_D, u.g)
        R = _cgs(R_D, u.cm)
        J = xi * M * np.sqrt(G * MBH * R)
        Sigma = M / (np.pi * A * R**2)

    elif keys == {"M_D", "J_D"}:
        M = _cgs(M_D, u.g)
        J = _cgs(J_D, u.g * u.cm**2 / u.s)
        # J = xi * M * sqrt(G * MBH * R)  =>  R = (J / (xi * M))^2 / (G * MBH)
        R = (J / (xi * M)) ** 2 / (G * MBH)
        Sigma = M / (np.pi * A * R**2)

    elif keys == {"M_D", "Sigma_D"}:
        M = _cgs(M_D, u.g)
        Sigma = _cgs(Sigma_D, u.g / u.cm**2)
        # Sigma = M / (pi * A * R^2)  =>  R = sqrt(M / (pi * A * Sigma))
        R = np.sqrt(M / (np.pi * A * Sigma))
        J = xi * M * np.sqrt(G * MBH * R)

    elif keys == {"R_D", "J_D"}:
        R = _cgs(R_D, u.cm)
        J = _cgs(J_D, u.g * u.cm**2 / u.s)
        # J = xi * M * sqrt(G * MBH * R)  =>  M = J / (xi * sqrt(G * MBH * R))
        M = J / (xi * np.sqrt(G * MBH * R))
        Sigma = M / (np.pi * A * R**2)

    elif keys == {"R_D", "Sigma_D"}:
        R = _cgs(R_D, u.cm)
        Sigma = _cgs(Sigma_D, u.g / u.cm**2)
        M = np.pi * A * Sigma * R**2
        J = xi * M * np.sqrt(G * MBH * R)

    else:  # keys == {"J_D", "Sigma_D"}
        J = _cgs(J_D, u.g * u.cm**2 / u.s)
        Sigma = _cgs(Sigma_D, u.g / u.cm**2)
        # J = xi * pi * A * Sigma * sqrt(G * MBH) * R^(5/2)
        # R = (J / (xi * pi * A * Sigma * sqrt(G * MBH)))^(2/5)
        R = (J / (xi * np.pi * A * Sigma * np.sqrt(G * MBH))) ** (2.0 / 5.0)
        M = np.pi * A * Sigma * R**2

    Omega = np.sqrt(G * MBH / R**3)

    return {
        "M_D": M * u.g,
        "R_D": R * u.cm,
        "J_D": J * u.g * u.cm**2 / u.s,
        "Sigma_D": Sigma * u.g / u.cm**2,
        "Omega": Omega * u.rad / u.s,
    }


# ================================================================ #
#  Heating/cooling diagnostic curves                               #
# ================================================================ #


def compute_heating_curves(
    M_disk: "_UnitBearingScalarLike" = None,
    J_disk: "_UnitBearingScalarLike" = None,
    R_disk: "_UnitBearingScalarLike" = None,
    Sigma_disk: "_UnitBearingScalarLike" = None,
    M_BH: "_UnitBearingScalarLike" = None,
    alpha: float = 0.1,
    mu: float = 0.615,
    opacity: Union[str, float, GreyOpacityLaw] = "electron_scattering",
    T_grid: Optional["_UnitBearingArrayLike"] = None,
    T_grid_n: int = 500,
    T_min: Optional["_UnitBearingScalarLike"] = 1e3 * u.K,
    T_max: Optional["_UnitBearingScalarLike"] = 1e8 * u.K,
):
    r"""Compute viscous heating and radiative cooling as functions of temperature.

    Exactly two of ``(M_disk, J_disk, R_disk, Sigma_disk)`` must be provided
    together with ``M_BH`` to fully specify the disk state.

    Parameters
    ----------
    M_disk : Quantity [mass], optional
        Disk mass.
    J_disk : Quantity [angular momentum], optional
        Disk angular momentum.
    R_disk : Quantity [length], optional
        Characteristic disk radius.
    Sigma_disk : Quantity [surface density], optional
        Disk surface density.
    M_BH : Quantity [mass]
        Black hole mass.
    alpha : float
        Shakura–Sunyaev viscosity parameter.
    mu : float
        Mean molecular weight.
    opacity : str, float, or GreyOpacityLaw
        Opacity specification.
    T_grid : Quantity [temperature], optional
        Explicit temperature grid.  If ``None``, a log-spaced grid between
        ``T_min`` and ``T_max`` with ``T_grid_n`` points is used.
    T_grid_n : int
        Number of temperature grid points (used only when ``T_grid`` is ``None``).
    T_min, T_max : Quantity [temperature]
        Temperature range for the default grid.

    Returns
    -------
    T : Quantity, shape (T_grid_n,) [K]
        Temperature grid.
    q_visc : Quantity, shape (T_grid_n,) [erg cm⁻² s⁻¹]
        Viscous heating rate.
    q_rad : Quantity, shape (T_grid_n,) [erg cm⁻² s⁻¹]
        Radiative cooling rate.
    """
    if M_BH is None:
        raise ValueError("M_BH must be provided.")

    state = disk_state(M_BH, M_D=M_disk, R_D=R_disk, J_D=J_disk, Sigma_D=Sigma_disk)
    Sigma = state["Sigma_D"].to(u.g / u.cm**2).value
    Omega = state["Omega"].to(u.rad / u.s).value

    T_min_val = ensure_in_units(T_min, u.K)
    T_max_val = ensure_in_units(T_max, u.K)

    if T_grid is not None:
        T_arr = ensure_in_units(T_grid, u.K)
    else:
        T_arr = np.geomspace(T_min_val, T_max_val, T_grid_n)

    log_T = np.log(T_arr)
    log_Sigma = np.log(Sigma)
    log_Omega = np.log(Omega)

    opacity_obj = get_opacity(opacity)
    log_q_visc, log_q_rad = _log_heating_curves(log_T, log_Sigma, log_Omega, alpha, mu, opacity_obj)

    erg_unit = u.erg / u.cm**2 / u.s
    return T_arr * u.K, np.exp(log_q_visc) * erg_unit, np.exp(log_q_rad) * erg_unit


def compute_advective_heating_curves(
    M_disk: "_UnitBearingScalarLike" = None,
    J_disk: "_UnitBearingScalarLike" = None,
    R_disk: "_UnitBearingScalarLike" = None,
    Sigma_disk: "_UnitBearingScalarLike" = None,
    M_BH: "_UnitBearingScalarLike" = None,
    alpha: float = 0.1,
    xi: float = 1.5,
    mu: float = 0.615,
    opacity: Union[str, float, GreyOpacityLaw] = "electron_scattering",
    T_grid: Optional["_UnitBearingArrayLike"] = None,
    T_grid_n: int = 500,
    T_min: Optional["_UnitBearingScalarLike"] = 1e3 * u.K,
    T_max: Optional["_UnitBearingScalarLike"] = 1e8 * u.K,
):
    r"""Compute viscous heating, radiative cooling, and advective cooling as functions of temperature.

    Exactly two of ``(M_disk, J_disk, R_disk, Sigma_disk)`` must be provided
    together with ``M_BH`` to fully specify the disk state.

    Parameters
    ----------
    M_disk : Quantity [mass], optional
        Disk mass.
    J_disk : Quantity [angular momentum], optional
        Disk angular momentum.
    R_disk : Quantity [length], optional
        Characteristic disk radius.
    Sigma_disk : Quantity [surface density], optional
        Disk surface density.
    M_BH : Quantity [mass]
        Black hole mass.
    alpha : float
        Shakura–Sunyaev viscosity parameter.
    xi : float
        Entropy gradient (advection) parameter.
    mu : float
        Mean molecular weight.
    opacity : str, float, or GreyOpacityLaw
        Opacity specification.
    T_grid : Quantity [temperature], optional
        Explicit temperature grid.
    T_grid_n : int
        Number of temperature grid points (used only when ``T_grid`` is ``None``).
    T_min, T_max : Quantity [temperature]
        Temperature range for the default grid.

    Returns
    -------
    T : Quantity, shape (T_grid_n,) [K]
        Temperature grid.
    q_visc : Quantity, shape (T_grid_n,) [erg cm⁻² s⁻¹]
        Viscous heating rate.
    q_rad : Quantity, shape (T_grid_n,) [erg cm⁻² s⁻¹]
        Radiative cooling rate.
    q_adv : Quantity, shape (T_grid_n,) [erg cm⁻² s⁻¹]
        Advective cooling rate.
    """
    if M_BH is None:
        raise ValueError("M_BH must be provided.")

    state = disk_state(M_BH, M_D=M_disk, R_D=R_disk, J_D=J_disk, Sigma_D=Sigma_disk)
    R = state["R_D"].to(u.cm).value
    Sigma = state["Sigma_D"].to(u.g / u.cm**2).value
    Omega = state["Omega"].to(u.rad / u.s).value

    T_min_val = ensure_in_units(T_min, u.K)
    T_max_val = ensure_in_units(T_max, u.K)

    if T_grid is not None:
        T_arr = ensure_in_units(T_grid, u.K)
    else:
        T_arr = np.geomspace(T_min_val, T_max_val, T_grid_n)

    log_T = np.log(T_arr)
    log_Sigma = np.log(Sigma)
    log_Omega = np.log(Omega)
    log_R = np.log(R)

    opacity_obj = get_opacity(opacity)
    log_q_visc, log_q_rad, log_q_adv = _log_advective_heating_curves(
        log_T, log_Sigma, log_Omega, log_R, alpha, mu, xi, opacity_obj
    )

    erg_unit = u.erg / u.cm**2 / u.s
    return (
        T_arr * u.K,
        np.exp(log_q_visc) * erg_unit,
        np.exp(log_q_rad) * erg_unit,
        np.exp(log_q_adv) * erg_unit,
    )


# ================================================================ #
#  S-curves                                                        #
# ================================================================ #


def compute_advection_s_curve(
    R_disk: "_UnitBearingScalarLike",
    M_BH: "_UnitBearingScalarLike",
    alpha: float = 0.1,
    xi: float = 1.5,
    mu: float = 0.615,
    opacity: Union[str, float, GreyOpacityLaw] = "electron_scattering",
    T_grid: Optional["_UnitBearingArrayLike"] = None,
    T_grid_n: int = 500,
    T_min: Optional["_UnitBearingScalarLike"] = 1e3 * u.K,
    T_max: Optional["_UnitBearingScalarLike"] = 1e8 * u.K,
    sigma_max: Optional["_UnitBearingScalarLike"] = 1e6 * u.g / u.cm**2,
    sigma_min: Optional["_UnitBearingScalarLike"] = 1e2 * u.g / u.cm**2,
):
    r"""Compute the thermal equilibrium S-curve for an advective disk.

    For each temperature in the grid, finds the surface density :math:`\Sigma`
    at which :math:`q^+ = q^- + q_{\rm adv}` using :func:`scipy.optimize.brentq`.

    Parameters
    ----------
    R_disk : Quantity [length]
        Characteristic disk radius.
    M_BH : Quantity [mass]
        Black hole mass.
    alpha : float
        Shakura–Sunyaev viscosity parameter.
    xi : float
        Entropy gradient (advection) parameter.
    mu : float
        Mean molecular weight.
    opacity : str, float, or GreyOpacityLaw
        Opacity specification.
    T_grid : Quantity [temperature], optional
        Explicit temperature grid.
    T_grid_n : int
        Number of temperature grid points (used only when ``T_grid`` is ``None``).
    T_min, T_max : Quantity [temperature]
        Temperature range for the default grid.
    sigma_min, sigma_max : Quantity [surface density]
        Surface density search bracket for brentq.

    Returns
    -------
    T : Quantity, shape (T_grid_n,) [K]
        Temperature grid (NaN entries where no equilibrium was found).
    Sigma : Quantity, shape (T_grid_n,) [g cm⁻²]
        Equilibrium surface density (NaN where no root exists).
    """
    R_disk = ensure_in_units(R_disk, u.cm)
    M_BH = ensure_in_units(M_BH, u.g)
    T_min_val = ensure_in_units(T_min, u.K)
    T_max_val = ensure_in_units(T_max, u.K)
    sigma_min_val = ensure_in_units(sigma_min, u.g / u.cm**2)
    sigma_max_val = ensure_in_units(sigma_max, u.g / u.cm**2)

    if T_grid is not None:
        T_arr = ensure_in_units(T_grid, u.K)
    else:
        T_arr = np.geomspace(T_min_val, T_max_val, T_grid_n)

    Omega = np.sqrt(_G_CGS * M_BH / R_disk**3)
    log_T = np.log(T_arr)
    log_Omega = np.log(Omega)
    log_R = np.log(R_disk)

    opacity_obj = get_opacity(opacity)

    sigma_equilibrium = np.full_like(T_arr, np.nan)

    for T_idx, _log_T in tqdm(enumerate(log_T), total=len(log_T)):
        try:
            log_sigma_root = brentq(
                lambda log_Sigma: _log_advective_heating_residual(
                    _log_T,  # noqa: B023
                    log_Sigma,
                    log_Omega,
                    log_R,
                    alpha=alpha,
                    mu=mu,
                    xi=xi,
                    opacity_obj=opacity_obj,
                ),
                np.log(sigma_min_val),
                np.log(sigma_max_val),
            )
            sigma_equilibrium[T_idx] = log_sigma_root
        except ValueError:
            pass  # No root in bracket — leave as NaN

    return T_arr * u.K, np.exp(sigma_equilibrium) * (u.g / u.cm**2)
