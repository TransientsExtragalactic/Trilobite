r"""Accretion physics utility functions for one-zone disk models.

This module provides three levels of functionality:

1. **Opacity resolution** — :func:`_resolve_kappa` converts a string name or a bare
   float into a CGS opacity value.

2. **Heating/cooling curves** — :func:`igP_es_trial_curves` and
   :func:`igP_es_adv_trial_curves` evaluate the heating and cooling rates as
   functions of temperature for fixed disk parameters, returning the individual
   rates and their residual.  Zero-crossings of the residual are thermal equilibrium
   points.  These are primarily diagnostic: plot :math:`q^+` and :math:`q^-` vs.
   :math:`T` to verify that a root exists before running the integrator.

3. **Equilibrium temperature** — :func:`compute_equilibrium_temperature` and
   :func:`compute_advective_equilibrium_temperature` find all thermal equilibrium
   temperatures for fixed :math:`\Sigma` and :math:`\Omega` using
   :func:`scipy.optimize.brentq` bracketed by sign changes on a coarse temperature
   grid.

4. **S-curves** — :func:`compute_standard_s_curve` and
   :func:`compute_advective_s_curve` trace the full thermal equilibrium locus in
   :math:`\Sigma`–:math:`T` space by sweeping a :math:`T`–:math:`\Sigma` grid and
   locating zero-crossings column by column.

All public functions accept :class:`~astropy.units.Quantity` inputs and return
:class:`~astropy.units.Quantity` outputs.  Private functions (prefixed ``_``) work
directly in CGS floats and may accept or return :class:`numpy.ndarray`.

Physics reference
-----------------
Heating/cooling balance:

.. math::

    q^+ &= \tfrac{9}{8}\,\alpha\,\Sigma\,\Omega\,c_s^2(T) \\
    q^- &= \frac{16\,\sigma_{\rm SB}\,T^4}{3\,\kappa\,\Sigma}  \\
    q_{\rm adv} &= q^+\,B\,c_s^2,
    \quad B = \frac{4\,\xi\,F_0\,\alpha\,M}{9\pi\,R^4\,\Omega^2\,\Sigma}

where :math:`F_0 = 1.6` is the Metzger+08 disk correction factor.

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

from triceratops.physics_utils.eos import _log_radiative_ideal_gas_disk_sound_speed
from triceratops.utils.misc_utils import ensure_in_units

if TYPE_CHECKING:
    from triceratops._typing import _UnitBearingArrayLike


# ================================================================ #
#  Physical constants (CGS)                                        #
# ================================================================ #
_SIGMA_SB: float = const.sigma_sb.cgs.value  # erg cm⁻² s⁻¹ K⁻⁴
_DISK_F0: float = 1.6  # Metzger+08 disk correction factor (dimensionless)
_KAPPA_ES_CGS: float = 0.34  # cm² g⁻¹ — electron scattering
_KNOWN_OPACITIES: dict = {"electron_scattering": _KAPPA_ES_CGS}


# ================================================================ #
#  Opacity resolution                                              #
# ================================================================ #
def _resolve_kappa(opacity: Union[str, float]) -> float:
    r"""Return opacity :math:`\kappa` in cm² g⁻¹ from a name or a bare float.

    Parameters
    ----------
    opacity : str or float
        Either a recognised string model name (``"electron_scattering"``) or a
        constant opacity value already in cm² g⁻¹.

    Returns
    -------
    float
        Opacity in cm² g⁻¹.

    Raises
    ------
    ValueError
        If *opacity* is a string that is not in the known table.
    """
    if isinstance(opacity, (int, float)):
        return float(opacity)
    if opacity not in _KNOWN_OPACITIES:
        raise ValueError(f"Unknown opacity model {opacity!r}.  Known models: {sorted(_KNOWN_OPACITIES)}.")
    return _KNOWN_OPACITIES[opacity]


# ================================================================ #
#  Low-level CGS helpers (private)                                 #
# ================================================================ #


def _cs_cgs(
    T: np.ndarray,
    Sigma: np.ndarray,
    Omega: np.ndarray,
    mu: float,
) -> np.ndarray:
    """Gas + radiation disk sound speed in CGS (cm s⁻¹).

    Thin wrapper around :func:`~triceratops.physics_utils.eos.\
_log_radiative_ideal_gas_disk_sound_speed` that works directly on CGS floats
    or arrays.
    """
    return np.exp(_log_radiative_ideal_gas_disk_sound_speed(np.log(T), mu, np.log(Sigma), np.log(Omega)))


def _standard_residual_cgs(
    T: np.ndarray,
    Sigma: float,
    Omega: float,
    alpha: float,
    mu: float,
    kappa: float,
) -> np.ndarray:
    r"""Net cooling rate :math:`q^- - q^+` (erg cm⁻² s⁻¹) for the standard closure.

    Positive means the disk is over-cooling; zero means thermal equilibrium.

    Parameters
    ----------
    T : array-like (K)
        Midplane temperature.
    Sigma : float (g cm⁻²)
        Surface density.
    Omega : float (s⁻¹)
        Keplerian angular velocity.
    alpha, mu, kappa : float
        Viscosity, mean molecular weight, opacity (cm² g⁻¹).
    """
    c_s = _cs_cgs(np.asarray(T, dtype=float), Sigma, Omega, mu)
    q_visc = (9.0 / 8.0) * alpha * Sigma * Omega * c_s**2
    q_rad = 16.0 * _SIGMA_SB * np.asarray(T, dtype=float) ** 4 / (3.0 * kappa * Sigma)
    return q_rad - q_visc


def _advective_residual_cgs(
    T: np.ndarray,
    Sigma: float,
    Omega: float,
    M: float,
    R: float,
    xi: float,
    alpha: float,
    mu: float,
    kappa: float,
) -> np.ndarray:
    r"""Net cooling rate :math:`(q^- + q_{\rm adv}) - q^+` (erg cm⁻² s⁻¹).

    Positive means the disk is over-cooling; zero means thermal equilibrium.

    Parameters
    ----------
    T : array-like (K)
        Midplane temperature.
    Sigma : float (g cm⁻²)
        Surface density.
    Omega : float (s⁻¹)
        Keplerian angular velocity.
    M : float (g)
        Disk mass.
    R : float (cm)
        Disk radius.
    xi : float
        Entropy gradient parameter (dimensionless, > 0).
    alpha, mu, kappa : float
        Viscosity, mean molecular weight, opacity (cm² g⁻¹).
    """
    T_arr = np.asarray(T, dtype=float)
    c_s = _cs_cgs(T_arr, Sigma, Omega, mu)
    q_visc = (9.0 / 8.0) * alpha * Sigma * Omega * c_s**2
    q_rad = 16.0 * _SIGMA_SB * T_arr**4 / (3.0 * kappa * Sigma)
    B = (4.0 / (9.0 * np.pi)) * xi * _DISK_F0 * alpha * M / (R**4 * Omega**2 * Sigma)
    q_adv = B * c_s**2 * q_visc
    return (q_rad + q_adv) - q_visc


# ================================================================ #
#  Heating/cooling diagnostic curves                               #
# ================================================================ #


def igP_es_trial_curves(
    T: "_UnitBearingArrayLike",
    alpha: float,
    Sigma: "_UnitBearingArrayLike",
    Omega: "_UnitBearingArrayLike",
    mu: float = 0.615,
    opacity: Union[str, float] = "electron_scattering",
) -> tuple[u.Quantity, u.Quantity, u.Quantity]:
    r"""Heating and cooling rates for the non-advective full-pressure closure.

    Evaluates the viscous heating rate :math:`q^+` and radiative cooling rate
    :math:`q^-` as functions of temperature for fixed disk parameters, using the
    same physics as the Cython ``igP`` closure.  A sign change in the residual
    :math:`q^- - q^+` indicates a thermal equilibrium root.

    .. math::

        q^+ &= \tfrac{9}{8}\,\alpha\,\Sigma\,\Omega\,c_s^2(T) \\
        q^- &= \frac{16\,\sigma_{\rm SB}\,T^4}{3\,\kappa\,\Sigma}

    where :math:`c_s(T)` is the gas + radiation isothermal sound speed.

    Parameters
    ----------
    T : `~astropy.units.Quantity`
        Array of midplane temperatures to evaluate.
    alpha : float
        Shakura–Sunyaev viscosity parameter (dimensionless).
    Sigma : `~astropy.units.Quantity`
        Disk surface density.
    Omega : `~astropy.units.Quantity`
        Keplerian angular velocity.
    mu : float, optional
        Mean molecular weight (default 0.615 for fully ionised solar composition).
    opacity : str or float, optional
        Opacity model name (``"electron_scattering"``) or constant opacity in
        cm² g⁻¹ (default ``"electron_scattering"`` → 0.34 cm² g⁻¹).

    Returns
    -------
    q_visc : `~astropy.units.Quantity`
        Viscous heating rate :math:`q^+` (erg cm⁻² s⁻¹).
    q_rad : `~astropy.units.Quantity`
        Radiative cooling rate :math:`q^-` (erg cm⁻² s⁻¹).
    residual : `~astropy.units.Quantity`
        Net cooling :math:`q^- - q^+` (erg cm⁻² s⁻¹).  A sign change indicates
        a thermal equilibrium root.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import matplotlib.pyplot as plt
        from astropy import units as u
        from triceratops.dynamics.accretion.one_zone.utils import igP_es_trial_curves

        T = np.logspace(4, 12, 500) * u.K
        q_visc, q_rad, residual = igP_es_trial_curves(
            T, alpha=0.1, Sigma=1e4 * u.g / u.cm**2, Omega=1e-3 / u.s
        )
        fig, ax = plt.subplots()
        ax.loglog(T, q_visc.value, label=r"$q^+$ (viscous)")
        ax.loglog(T, q_rad.value,  label=r"$q^-$ (radiative)")
        ax.set_xlabel("T (K)")
        ax.set_ylabel(r"$q$ (erg cm$^{-2}$ s$^{-1}$)")
        ax.legend()
    """
    kappa = _resolve_kappa(opacity)
    T_cgs = ensure_in_units(T, u.K)
    Sig_cgs = ensure_in_units(Sigma, u.g / u.cm**2)
    Omg_cgs = ensure_in_units(Omega, 1 / u.s)

    c_s = _cs_cgs(T_cgs, Sig_cgs, Omg_cgs, mu)
    q_visc = (9.0 / 8.0) * alpha * Sig_cgs * Omg_cgs * c_s**2
    q_rad = 16.0 * _SIGMA_SB * T_cgs**4 / (3.0 * kappa * Sig_cgs)

    flux_unit = u.erg / u.cm**2 / u.s
    return q_visc * flux_unit, q_rad * flux_unit, (q_rad - q_visc) * flux_unit


def igP_es_adv_trial_curves(
    T: "_UnitBearingArrayLike",
    alpha: float,
    Sigma: "_UnitBearingArrayLike",
    Omega: "_UnitBearingArrayLike",
    M: "_UnitBearingArrayLike",
    R: "_UnitBearingArrayLike",
    xi: float,
    mu: float = 0.615,
    opacity: Union[str, float] = "electron_scattering",
) -> tuple[u.Quantity, u.Quantity, u.Quantity, u.Quantity]:
    r"""Heating and cooling rates for the advective full-pressure closure.

    Extends :func:`igP_es_trial_curves` with an advective cooling term.  The
    thermal balance is :math:`q^+ = q^- + q_{\rm adv}`, giving the residual
    :math:`(q^- + q_{\rm adv}) - q^+`.

    .. math::

        q^+        &= \tfrac{9}{8}\,\alpha\,\Sigma\,\Omega\,c_s^2(T) \\
        q^-        &= \frac{16\,\sigma_{\rm SB}\,T^4}{3\,\kappa\,\Sigma} \\
        q_{\rm adv}&= q^+\,B\,c_s^2,
        \quad
        B = \frac{4\,\xi\,F_0\,\alpha\,M}{9\pi\,R^4\,\Omega^2\,\Sigma}

    where :math:`\xi` is the entropy gradient parameter and
    :math:`F_0 = 1.6` is the Metzger+08 disk correction factor.
    Setting :math:`\xi \to 0` recovers the non-advective limit.

    Parameters
    ----------
    T : `~astropy.units.Quantity`
        Array of midplane temperatures to evaluate.
    alpha : float
        Shakura–Sunyaev viscosity parameter (dimensionless).
    Sigma : `~astropy.units.Quantity`
        Disk surface density.
    Omega : `~astropy.units.Quantity`
        Keplerian angular velocity.
    M : `~astropy.units.Quantity`
        Disk mass.
    R : `~astropy.units.Quantity`
        Disk radius.
    xi : float
        Entropy gradient parameter (dimensionless, > 0).
    mu : float, optional
        Mean molecular weight (default 0.615).
    opacity : str or float, optional
        Opacity model name or constant opacity in cm² g⁻¹
        (default ``"electron_scattering"``).

    Returns
    -------
    q_visc : `~astropy.units.Quantity`
        Viscous heating rate :math:`q^+` (erg cm⁻² s⁻¹).
    q_rad : `~astropy.units.Quantity`
        Radiative cooling rate :math:`q^-` (erg cm⁻² s⁻¹).
    q_adv : `~astropy.units.Quantity`
        Advective cooling rate :math:`q_{\rm adv}` (erg cm⁻² s⁻¹).
    residual : `~astropy.units.Quantity`
        Net cooling :math:`(q^- + q_{\rm adv}) - q^+` (erg cm⁻² s⁻¹).

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import matplotlib.pyplot as plt
        from astropy import units as u
        from triceratops.dynamics.accretion.one_zone.utils import igP_es_adv_trial_curves

        T = np.logspace(4, 12, 500) * u.K
        q_visc, q_rad, q_adv, residual = igP_es_adv_trial_curves(
            T,
            alpha=0.1,
            Sigma=1e4 * u.g / u.cm**2,
            Omega=1e-3 / u.s,
            M=1e29 * u.g,
            R=1e11 * u.cm,
            xi=1.0,
        )
        fig, ax = plt.subplots()
        ax.loglog(T, q_visc.value, label=r"$q^+$ (viscous)")
        ax.loglog(T, q_rad.value,  label=r"$q^-$ (radiative)")
        ax.loglog(T, q_adv.value,  label=r"$q_{\rm adv}$")
        ax.set_xlabel("T (K)")
        ax.set_ylabel(r"$q$ (erg cm$^{-2}$ s$^{-1}$)")
        ax.legend()
    """
    kappa = _resolve_kappa(opacity)
    T_cgs = ensure_in_units(T, u.K)
    Sig_cgs = ensure_in_units(Sigma, u.g / u.cm**2)
    Omg_cgs = ensure_in_units(Omega, 1 / u.s)
    M_cgs = ensure_in_units(M, u.g)
    R_cgs = ensure_in_units(R, u.cm)

    c_s = _cs_cgs(T_cgs, Sig_cgs, Omg_cgs, mu)
    q_visc = (9.0 / 8.0) * alpha * Sig_cgs * Omg_cgs * c_s**2
    q_rad = 16.0 * _SIGMA_SB * T_cgs**4 / (3.0 * kappa * Sig_cgs)
    B = (4.0 / (9.0 * np.pi)) * xi * _DISK_F0 * alpha * M_cgs / (R_cgs**4 * Omg_cgs**2 * Sig_cgs)
    q_adv = B * c_s**2 * q_visc

    flux_unit = u.erg / u.cm**2 / u.s
    return (
        q_visc * flux_unit,
        q_rad * flux_unit,
        q_adv * flux_unit,
        (q_rad + q_adv - q_visc) * flux_unit,
    )


# ================================================================ #
#  Low-level T-equilibrium finders (private, CGS scalar inputs)    #
# ================================================================ #

_DEFAULT_T_LO: float = 1e1  # K
_DEFAULT_T_HI: float = 1e14  # K
_DEFAULT_N_BRACKET: int = 500


def _find_equilibrium_T_standard(
    Sigma_cgs: float,
    Omega_cgs: float,
    alpha: float,
    mu: float,
    kappa: float,
    T_lo: float = _DEFAULT_T_LO,
    T_hi: float = _DEFAULT_T_HI,
    n: int = _DEFAULT_N_BRACKET,
) -> list[float]:
    """Find all thermal equilibrium temperatures (K) for the standard closure.

    Evaluates the residual on a coarse log-spaced temperature grid, then refines
    each sign-change bracket with :func:`scipy.optimize.brentq`.

    Parameters
    ----------
    Sigma_cgs : float (g cm⁻²)
    Omega_cgs : float (s⁻¹)
    alpha, mu, kappa : float
    T_lo, T_hi : float (K)
        Search bounds.
    n : int
        Number of points in the coarse grid.

    Returns
    -------
    list of float
        Equilibrium temperature(s) in K (empty if none found in the range).
    """
    T_grid = np.geomspace(T_lo, T_hi, n)
    F = _standard_residual_cgs(T_grid, Sigma_cgs, Omega_cgs, alpha, mu, kappa)
    roots: list[float] = []
    for i in np.where(np.diff(np.sign(F)))[0]:
        root = brentq(
            lambda T: _standard_residual_cgs(T, Sigma_cgs, Omega_cgs, alpha, mu, kappa),
            T_grid[i],
            T_grid[i + 1],
        )
        roots.append(float(root))
    return roots


def _find_equilibrium_T_advective(
    Sigma_cgs: float,
    Omega_cgs: float,
    M_cgs: float,
    R_cgs: float,
    xi: float,
    alpha: float,
    mu: float,
    kappa: float,
    T_lo: float = _DEFAULT_T_LO,
    T_hi: float = _DEFAULT_T_HI,
    n: int = _DEFAULT_N_BRACKET,
) -> list[float]:
    """Find all thermal equilibrium temperatures (K) for the advective closure.

    Parameters
    ----------
    Sigma_cgs : float (g cm⁻²)
    Omega_cgs : float (s⁻¹)
    M_cgs : float (g)
    R_cgs : float (cm)
    xi : float
        Entropy gradient parameter.
    alpha, mu, kappa : float
    T_lo, T_hi : float (K)
        Search bounds.
    n : int
        Number of points in the coarse grid.

    Returns
    -------
    list of float
        Equilibrium temperature(s) in K.
    """
    T_grid = np.geomspace(T_lo, T_hi, n)
    F = _advective_residual_cgs(T_grid, Sigma_cgs, Omega_cgs, M_cgs, R_cgs, xi, alpha, mu, kappa)
    roots: list[float] = []
    for i in np.where(np.diff(np.sign(F)))[0]:
        root = brentq(
            lambda T: _advective_residual_cgs(T, Sigma_cgs, Omega_cgs, M_cgs, R_cgs, xi, alpha, mu, kappa),
            T_grid[i],
            T_grid[i + 1],
        )
        roots.append(float(root))
    return roots


# ================================================================ #
#  Public T-equilibrium functions                                  #
# ================================================================ #


def compute_equilibrium_temperature(
    Sigma: "_UnitBearingArrayLike",
    Omega: "_UnitBearingArrayLike",
    alpha: float,
    mu: float = 0.615,
    opacity: Union[str, float] = "electron_scattering",
    T_range: Optional[tuple] = None,
    n_bracket: int = _DEFAULT_N_BRACKET,
) -> list[u.Quantity]:
    r"""Find all thermal equilibrium temperatures for fixed :math:`\Sigma` and :math:`\Omega`.

    Solves :math:`q^-(T) = q^+(T)` for the non-advective full-pressure disk using
    :func:`scipy.optimize.brentq` applied to sign-change brackets on a log-spaced
    temperature grid.

    Parameters
    ----------
    Sigma : `~astropy.units.Quantity`
        Disk surface density.
    Omega : `~astropy.units.Quantity`
        Keplerian angular velocity.
    alpha : float
        Shakura–Sunyaev viscosity parameter.
    mu : float, optional
        Mean molecular weight (default 0.615).
    opacity : str or float, optional
        Opacity model name or constant opacity in cm² g⁻¹.
    T_range : (Quantity, Quantity), optional
        Lower and upper temperature search bounds.  Defaults to (10 K, 10¹⁴ K).
    n_bracket : int, optional
        Number of points in the initial coarse grid (default 500).

    Returns
    -------
    list of `~astropy.units.Quantity`
        Equilibrium temperature(s) in K.  Typically 1 (stable) or 3
        (cold, unstable, and hot branch) values depending on the disk regime.

    Examples
    --------
    .. code-block:: python

        from astropy import units as u
        from triceratops.dynamics.accretion.one_zone.utils import (
            compute_equilibrium_temperature,
        )

        roots = compute_equilibrium_temperature(
            Sigma=1e4 * u.g / u.cm**2,
            Omega=1e-3 / u.s,
            alpha=0.1,
        )
        for T_eq in roots:
            print(f"T_eq = {T_eq:.3e}")
    """
    kappa = _resolve_kappa(opacity)
    Sig_cgs = ensure_in_units(Sigma, u.g / u.cm**2)
    Omg_cgs = ensure_in_units(Omega, 1 / u.s)
    T_lo = ensure_in_units(T_range[0], u.K) if T_range is not None else _DEFAULT_T_LO
    T_hi = ensure_in_units(T_range[1], u.K) if T_range is not None else _DEFAULT_T_HI

    roots = _find_equilibrium_T_standard(Sig_cgs, Omg_cgs, alpha, mu, kappa, T_lo, T_hi, n_bracket)
    return [T * u.K for T in roots]


def compute_advective_equilibrium_temperature(
    Sigma: "_UnitBearingArrayLike",
    Omega: "_UnitBearingArrayLike",
    M: "_UnitBearingArrayLike",
    R: "_UnitBearingArrayLike",
    xi: float,
    alpha: float,
    mu: float = 0.615,
    opacity: Union[str, float] = "electron_scattering",
    T_range: Optional[tuple] = None,
    n_bracket: int = _DEFAULT_N_BRACKET,
) -> list[u.Quantity]:
    r"""Find all thermal equilibrium temperatures for the advective disk closure.

    Solves :math:`q^-(T) + q_{\rm adv}(T) = q^+(T)` using
    :func:`scipy.optimize.brentq` applied to sign-change brackets on a log-spaced
    temperature grid.

    Parameters
    ----------
    Sigma : `~astropy.units.Quantity`
        Disk surface density.
    Omega : `~astropy.units.Quantity`
        Keplerian angular velocity.
    M : `~astropy.units.Quantity`
        Disk mass.
    R : `~astropy.units.Quantity`
        Disk radius.
    xi : float
        Entropy gradient parameter (dimensionless, > 0).
    alpha : float
        Shakura–Sunyaev viscosity parameter.
    mu : float, optional
        Mean molecular weight (default 0.615).
    opacity : str or float, optional
        Opacity model name or constant opacity in cm² g⁻¹.
    T_range : (Quantity, Quantity), optional
        Temperature search bounds.  Defaults to (10 K, 10¹⁴ K).
    n_bracket : int, optional
        Number of points in the initial coarse grid (default 500).

    Returns
    -------
    list of `~astropy.units.Quantity`
        Equilibrium temperature(s) in K.
    """
    kappa = _resolve_kappa(opacity)
    Sig_cgs = ensure_in_units(Sigma, u.g / u.cm**2)
    Omg_cgs = ensure_in_units(Omega, 1 / u.s)
    M_cgs = ensure_in_units(M, u.g)
    R_cgs = ensure_in_units(R, u.cm)
    T_lo = ensure_in_units(T_range[0], u.K) if T_range is not None else _DEFAULT_T_LO
    T_hi = ensure_in_units(T_range[1], u.K) if T_range is not None else _DEFAULT_T_HI

    roots = _find_equilibrium_T_advective(Sig_cgs, Omg_cgs, M_cgs, R_cgs, xi, alpha, mu, kappa, T_lo, T_hi, n_bracket)
    return [T * u.K for T in roots]


# ================================================================ #
#  S-curve computation                                             #
# ================================================================ #

_DEFAULT_T_RANGE = (1e1 * u.K, 1e14 * u.K)
_DEFAULT_SIGMA_RANGE = (1e-2 * u.g / u.cm**2, 1e8 * u.g / u.cm**2)
_DEFAULT_N_GRID = 400


def compute_standard_s_curve(
    Omega: "_UnitBearingArrayLike",
    alpha: float,
    mu: float = 0.615,
    opacity: Union[str, float] = "electron_scattering",
    T_range: Optional[tuple] = None,
    Sigma_range: Optional[tuple] = None,
    n_T: int = _DEFAULT_N_GRID,
    n_Sigma: int = _DEFAULT_N_GRID,
) -> dict:
    r"""Trace the thermal equilibrium S-curve in :math:`\Sigma`–:math:`T` space.

    For each temperature in a log-spaced :math:`T` grid, sweeps a log-spaced
    :math:`\Sigma` grid and locates all :math:`\Sigma` roots of
    :math:`q^-(T, \Sigma) - q^+(T, \Sigma) = 0` using sign-change detection and
    :func:`scipy.optimize.brentq`.  The collected :math:`(T, \Sigma)` pairs form
    the equilibrium locus, which has the characteristic S-shape in log–log space.

    Parameters
    ----------
    Omega : `~astropy.units.Quantity`
        Keplerian angular velocity.
    alpha : float
        Shakura–Sunyaev viscosity parameter.
    mu : float, optional
        Mean molecular weight (default 0.615).
    opacity : str or float, optional
        Opacity model name or constant opacity in cm² g⁻¹.
    T_range : (Quantity, Quantity), optional
        Lower and upper temperature bounds.  Defaults to (10 K, 10¹⁴ K).
    Sigma_range : (Quantity, Quantity), optional
        Surface density bounds.  Defaults to (10⁻² g cm⁻², 10⁸ g cm⁻²).
    n_T : int, optional
        Number of temperature grid points (default 400).
    n_Sigma : int, optional
        Number of surface density grid points per temperature (default 400).

    Returns
    -------
    dict
        ``"T"`` : `~astropy.units.Quantity`
            Equilibrium temperatures (K).
        ``"Sigma"`` : `~astropy.units.Quantity`
            Equilibrium surface densities (g cm⁻²) at each temperature.

    Notes
    -----
    The multi-valued region (where three branches coexist) produces multiple
    entries at nearby temperatures, one per branch.  Sort by ``T`` and plot
    ``Sigma`` vs ``T`` to reproduce the classic S-shape.

    Examples
    --------
    .. code-block:: python

        import matplotlib.pyplot as plt
        from astropy import units as u
        from triceratops.dynamics.accretion.one_zone.utils import (
            compute_standard_s_curve,
        )

        sc = compute_standard_s_curve(
            Omega=1e-3 / u.s, alpha=0.1
        )
        fig, ax = plt.subplots()
        ax.loglog(sc["T"].value, sc["Sigma"].value, "k.")
        ax.set_xlabel("T (K)")
        ax.set_ylabel(r"$\Sigma$ (g cm$^{-2}$)")
    """
    kappa = _resolve_kappa(opacity)
    Omg_cgs = ensure_in_units(Omega, 1 / u.s)

    T_lo = ensure_in_units(T_range[0] if T_range is not None else _DEFAULT_T_RANGE[0], u.K)
    T_hi = ensure_in_units(T_range[1] if T_range is not None else _DEFAULT_T_RANGE[1], u.K)
    Sig_lo = ensure_in_units(
        Sigma_range[0] if Sigma_range is not None else _DEFAULT_SIGMA_RANGE[0],
        u.g / u.cm**2,
    )
    Sig_hi = ensure_in_units(
        Sigma_range[1] if Sigma_range is not None else _DEFAULT_SIGMA_RANGE[1],
        u.g / u.cm**2,
    )

    T_grid = np.geomspace(T_lo, T_hi, n_T)
    Sig_grid = np.geomspace(Sig_lo, Sig_hi, n_Sigma)

    T_pts: list[float] = []
    Sig_pts: list[float] = []

    for T_val in T_grid:
        F = _standard_residual_cgs(T_val, Sig_grid, Omg_cgs, alpha, mu, kappa)
        for i in np.where(np.diff(np.sign(F)))[0]:
            sig_root = brentq(
                lambda S: _standard_residual_cgs(T_val, S, Omg_cgs, alpha, mu, kappa),  # noqa: B023
                Sig_grid[i],
                Sig_grid[i + 1],
            )
            T_pts.append(float(T_val))
            Sig_pts.append(float(sig_root))

    return {
        "T": np.array(T_pts) * u.K,
        "Sigma": np.array(Sig_pts) * u.g / u.cm**2,
    }


def compute_advective_s_curve(
    Omega: "_UnitBearingArrayLike",
    M: "_UnitBearingArrayLike",
    R: "_UnitBearingArrayLike",
    xi: float,
    alpha: float,
    mu: float = 0.615,
    opacity: Union[str, float] = "electron_scattering",
    T_range: Optional[tuple] = None,
    Sigma_range: Optional[tuple] = None,
    n_T: int = _DEFAULT_N_GRID,
    n_Sigma: int = _DEFAULT_N_GRID,
) -> dict:
    r"""Trace the advective thermal equilibrium S-curve in :math:`\Sigma`–:math:`T` space.

    Extends :func:`compute_standard_s_curve` to include the advective cooling term
    :math:`q_{\rm adv}`.

    Parameters
    ----------
    Omega : `~astropy.units.Quantity`
        Keplerian angular velocity.
    M : `~astropy.units.Quantity`
        Disk mass.
    R : `~astropy.units.Quantity`
        Disk radius.
    xi : float
        Entropy gradient parameter (dimensionless, > 0).
    alpha : float
        Shakura–Sunyaev viscosity parameter.
    mu : float, optional
        Mean molecular weight (default 0.615).
    opacity : str or float, optional
        Opacity model name or constant opacity in cm² g⁻¹.
    T_range : (Quantity, Quantity), optional
        Temperature bounds.  Defaults to (10 K, 10¹⁴ K).
    Sigma_range : (Quantity, Quantity), optional
        Surface density bounds.  Defaults to (10⁻² g cm⁻², 10⁸ g cm⁻²).
    n_T : int, optional
        Number of temperature grid points (default 400).
    n_Sigma : int, optional
        Number of surface density grid points per temperature (default 400).

    Returns
    -------
    dict
        ``"T"`` : `~astropy.units.Quantity`
            Equilibrium temperatures (K).
        ``"Sigma"`` : `~astropy.units.Quantity`
            Equilibrium surface densities (g cm⁻²).
    """
    kappa = _resolve_kappa(opacity)
    Omg_cgs = ensure_in_units(Omega, 1 / u.s)
    M_cgs = ensure_in_units(M, u.g)
    R_cgs = ensure_in_units(R, u.cm)

    T_lo = ensure_in_units(T_range[0] if T_range is not None else _DEFAULT_T_RANGE[0], u.K)
    T_hi = ensure_in_units(T_range[1] if T_range is not None else _DEFAULT_T_RANGE[1], u.K)
    Sig_lo = ensure_in_units(
        Sigma_range[0] if Sigma_range is not None else _DEFAULT_SIGMA_RANGE[0],
        u.g / u.cm**2,
    )
    Sig_hi = ensure_in_units(
        Sigma_range[1] if Sigma_range is not None else _DEFAULT_SIGMA_RANGE[1],
        u.g / u.cm**2,
    )

    T_grid = np.geomspace(T_lo, T_hi, n_T)
    Sig_grid = np.geomspace(Sig_lo, Sig_hi, n_Sigma)

    T_pts: list[float] = []
    Sig_pts: list[float] = []

    for T_val in T_grid:
        F = _advective_residual_cgs(T_val, Sig_grid, Omg_cgs, M_cgs, R_cgs, xi, alpha, mu, kappa)
        for i in np.where(np.diff(np.sign(F)))[0]:
            sig_root = brentq(
                lambda S: _advective_residual_cgs(T_val, S, Omg_cgs, M_cgs, R_cgs, xi, alpha, mu, kappa),  # noqa: B023
                Sig_grid[i],
                Sig_grid[i + 1],
            )
            T_pts.append(float(T_val))
            Sig_pts.append(float(sig_root))

    return {
        "T": np.array(T_pts) * u.K,
        "Sigma": np.array(Sig_pts) * u.g / u.cm**2,
    }
