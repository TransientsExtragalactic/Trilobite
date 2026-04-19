r"""
Thin-disk analytical accretion disk models.

This module implements the canonical **geometrically-thin, optically-thick** accretion disk model
of :footcite:t:`1973A&A....24..337S`, which is one of the few analytically tractable disk models
and serves as a useful reference point for more complex models.

See Also
--------
:mod:`triceratops.dynamics.accretion.utils` : Low-level disk utility functions (T_eff profile, SED integrals, etc.)
:mod:`triceratops.dynamics.accretion.one_zone` : Time-dependent one-zone disk models.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union

import numpy as np
from astropy import constants as const
from astropy import units as u

from triceratops.physics_utils.constants import _log_G_cgs, _log_sigma_sb_cgs
from triceratops.utils.misc_utils import ensure_in_units

if TYPE_CHECKING:
    from triceratops._typing import _ArrayLike, _UnitBearingArrayLike

__all__ = [
    "ThinDiskBase",
    "AlphaDisk",
    # SS73 steady-state emission utilities
    "disk_effective_temperature",
    "disk_bolometric_luminosity",
    "disk_spectral_luminosity",
    "disk_spectral_luminosity_iso",
    "disk_flux_density",
]

# ============================================================= #
# Constants
# ============================================================= #
#: Solar mass in grams  (M ≡ M / M_sun)
_M_SUN_CGS: float = const.M_sun.cgs.value
_h_cgs: float = const.h.cgs.value  # erg s
_k_B_cgs: float = const.k_B.cgs.value  # erg K-1
_c_cgs: float = const.c.cgs.value  # cm s-1
_MDOT_NORM_CGS: float = 1.0e16
_R_NORM_CGS: float = 1.0e10

# Frank, King, and Raine (2002) scaling constants for the alpha-disk solution. These
# are used to avoid recomputing the relevant constants in
# the log-space computations of the disk structure.
_LOG_M_SUN: float = np.log(_M_SUN_CGS)
_LOG_MDOT_NORM: float = np.log(_MDOT_NORM_CGS)
_LOG_R_NORM: float = np.log(_R_NORM_CGS)
_LOG_C_SIGMA: float = np.log(5.2)
_LOG_C_TC: float = np.log(1.4e4)
_LOG_C_H: float = np.log(1.7e8)
_LOG_C_RHO: float = np.log(3.1e-8)
_LOG_C_TAU: float = np.log(190.0)
_LOG_C_NU: float = np.log(1.8e14)
_LOG_C_VR: float = np.log(2.7e4)

# Pre-computed log combination for the steady-state T_eff(r) profile.
# Combines the coefficient from FKR so that:
_log_T_disk_coef: float = np.log(3.0) + _log_G_cgs - np.log(8.0 * np.pi) - _log_sigma_sb_cgs


# ============================================================================= #
# Low-Level API
# ============================================================================= #
# This is the low-level interface for these disk models, which operate on raw
# cgs arrays without any checks.
def _log_disk_effective_temperature(
    log_r: "_ArrayLike",
    log_M_BH: float,
    log_M_dot: float,
    log_R_in: float,
) -> "_ArrayLike":
    r"""
    Compute :math:`\ln T_{\rm eff}(r)` for a steady-state Shakura-Sunyaev disk.

    .. math::

        T_{\rm eff}(r) = \left(\frac{3\,G\,M_{\rm BH}\,\dot{M}}
                               {8\pi\,\sigma_{\rm SB}\,r^3}
                         \left[1-\sqrt{\tfrac{R_{\rm in}}{r}}\right]
                         \right)^{1/4}

    At :math:`r = R_{\rm in}` the zero-torque inner boundary condition forces
    :math:`T_{\rm eff} \to 0`.

    Parameters
    ----------
    log_r : array-like
        :math:`\ln r` grid (natural log of radius in cm).
    log_M_BH, log_M_dot, log_R_in : float
        Natural logs of BH mass (g), accretion rate (g s⁻¹), and inner
        truncation radius (cm).

    Returns
    -------
    array-like
        :math:`\ln T_{\rm eff}(r)` [K].
    """
    log_r = np.asarray(log_r, dtype=float)
    log_coef = 0.25 * (_log_T_disk_coef + log_M_BH + log_M_dot)
    with np.errstate(divide="ignore"):
        log_f = np.log1p(-np.exp(0.5 * (log_R_in - log_r)))
    log_T = log_coef - 0.75 * log_r + 0.25 * log_f
    return log_T if log_T.ndim > 0 else log_T.item()


def _disk_planck_ring_integral(
    log_nu_arr: np.ndarray,
    log_M_BH: float,
    log_M_dot: float,
    log_R_in: float,
    log_R_out: float,
    N_r: int = 500,
) -> np.ndarray:
    r"""
    Evaluate the core radial integral of the Planck function over disk annuli.

    .. math::

        I(\nu) = \int_{R_{\rm in}}^{R_{\rm out}}
                 B_\nu\!\left(T_{\rm eff}(r)\right) r\,\mathrm{d}r

    evaluated on a log-spaced radial grid using the trapezoidal rule.

    Parameters
    ----------
    log_nu_arr : ndarray, shape (Nν,)
        :math:`\ln\nu` grid (natural log of frequency in Hz).
    log_M_BH, log_M_dot, log_R_in, log_R_out : float
        Natural logs of BH mass (g), accretion rate (g s⁻¹), inner and
        outer radii (cm).
    N_r : int
        Number of radial quadrature points.  Default ``500``.

    Returns
    -------
    ndarray, shape (Nν,)
        :math:`I(\nu)` in :math:`\mathrm{erg\,s^{-1}\,Hz^{-1}\,sr^{-1}}`.
    """
    # Obtain the log_r and log_T grids on the fly.
    log_r = np.linspace(log_R_in, log_R_out, N_r)
    log_T = np.asarray(_log_disk_effective_temperature(log_r, log_M_BH, log_M_dot, log_R_in))
    T = np.exp(log_T)
    r_sq = np.exp(2.0 * log_r)

    # Broadcast to (Nν, Nr) for vectorized Planck evaluation.  The Planck function is evaluated
    # on the fly here to avoid storing large intermediate arrays.  This is the main bottleneck in
    # the SED computation, so we want to avoid unnecessary overhead from storing large arrays of T.
    nu = np.exp(log_nu_arr)[:, None]  # (N_nu, 1)
    T_grid = T[None, :]  # (1, Nr)
    r2_grid = r_sq[None, :]  # (1, Nr)

    # At r = R_in, T → 0 → x → ∞ → B_ν → 0.  Suppress benign boundary warnings.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        x = (_h_cgs * nu) / (_k_B_cgs * T_grid)
        B_nu = (2.0 * _h_cgs * nu**3 / _c_cgs**2) / np.expm1(x)
        B_nu = np.where(np.isfinite(B_nu), B_nu, 0.0)

    return np.trapz(r2_grid * B_nu, x=log_r, axis=1)  # (N_nu,)


def _log_disk_bolometric_luminosity(
    log_M_BH: float,
    log_M_dot: float,
    log_R_in: float,
) -> float:
    r"""
    Analytic bolometric luminosity of a steady-state Shakura-Sunyaev disk.

    Integrating the viscous dissipation profile over both faces with a
    zero-torque inner boundary and :math:`R_{\rm out} \to \infty` gives:

    .. math::

        L_{\rm bol} = \frac{G\,M_{\rm BH}\,\dot{M}}{2\,R_{\rm in}}

    Returns
    -------
    float
        :math:`\ln L_{\rm bol}` [:math:`\mathrm{erg\,s^{-1}}`].
    """
    return _log_G_cgs + log_M_BH + log_M_dot - np.log(2.0) - log_R_in


def _log_disk_spectral_luminosity(
    log_nu_arr: np.ndarray,
    log_M_BH: float,
    log_M_dot: float,
    log_R_in: float,
    log_R_out: float,
    N_r: int = 500,
) -> np.ndarray:
    r"""
    Compute :math:`\ln L_\nu` — the angle-integrated spectral luminosity.

    Accounts for emission from both disk faces (Lambertian emitter):

    .. math::

        L_\nu = 4\pi^2
                \int_{R_{\rm in}}^{R_{\rm out}} B_\nu(T_{\rm eff}(r))\,r\,\mathrm{d}r

    Returns
    -------
    ndarray, shape (Nν,)
        :math:`\ln L_\nu` [:math:`\mathrm{erg\,s^{-1}\,Hz^{-1}}`].
    """
    integral = _disk_planck_ring_integral(log_nu_arr, log_M_BH, log_M_dot, log_R_in, log_R_out, N_r)
    return np.log(4.0 * np.pi**2 * integral)


def _log_disk_flux_density(
    log_nu_arr: np.ndarray,
    log_M_BH: float,
    log_M_dot: float,
    log_R_in: float,
    log_R_out: float,
    log_DL: float,
    cos_theta: float = 1.0,
    N_r: int = 500,
) -> np.ndarray:
    r"""
    Compute :math:`\ln F_\nu` for a multi-colour blackbody disk (one face).

    The observer sees one optically-thick face of the disk.  For a Lambertian
    surface at inclination :math:`\theta` from the disk normal:

    .. math::

        F_\nu = \frac{2\pi\cos\theta}{D_L^2}
                \int_{R_{\rm in}}^{R_{\rm out}} B_\nu(T_{\rm eff}(r))\,r\,\mathrm{d}r

    Returns
    -------
    ndarray, shape (Nν,)
        :math:`\ln F_\nu` [:math:`\mathrm{erg\,s^{-1}\,Hz^{-1}\,cm^{-2}}`].
    """
    integral = _disk_planck_ring_integral(log_nu_arr, log_M_BH, log_M_dot, log_R_in, log_R_out, N_r)
    F_nu = (2.0 * np.pi * cos_theta) * np.exp(-2.0 * log_DL) * integral
    return np.log(F_nu)


def _log_disk_spectral_luminosity_iso(
    log_nu_arr: np.ndarray,
    log_M_BH: float,
    log_M_dot: float,
    log_R_in: float,
    log_R_out: float,
    cos_theta: float = 1.0,
    N_r: int = 500,
) -> np.ndarray:
    r"""
    Compute :math:`\ln L_{\nu,{\rm iso}}` — the isotropic-equivalent spectral luminosity.

    This is the luminosity an observer infers by multiplying the observed flux
    (from one face) by :math:`4\pi D_L^2`, assuming isotropic emission:

    .. math::

        L_{\nu,{\rm iso}} = 4\pi D_L^2\,F_\nu
                          = 8\pi^2\cos\theta
                            \int_{R_{\rm in}}^{R_{\rm out}} B_\nu(T_{\rm eff}(r))\,r\,\mathrm{d}r

    Returns
    -------
    ndarray, shape (Nν,)
        :math:`\ln L_{\nu,{\rm iso}}` [:math:`\mathrm{erg\,s^{-1}\,Hz^{-1}}`].
    """
    integral = _disk_planck_ring_integral(log_nu_arr, log_M_BH, log_M_dot, log_R_in, log_R_out, N_r)
    return np.log(8.0 * np.pi**2 * cos_theta * integral)


# ============================================================================= #
# Public API
# ============================================================================= #
def disk_effective_temperature(
    R: u.Quantity,
    M_BH: u.Quantity,
    mdot: u.Quantity,
    R_in: u.Quantity,
) -> u.Quantity:
    r"""
    Effective temperature profile of a steady-state Shakura-Sunyaev disk.

    .. math::

        T_{\rm eff}(r) = \left(\frac{3\,G\,M_{\rm BH}\,\dot{M}}
                               {8\pi\,\sigma_{\rm SB}\,r^3}
                         \left[1-\sqrt{\tfrac{R_{\rm in}}{r}}\right]
                         \right)^{1/4}

    Parameters
    ----------
    R : `~astropy.units.Quantity`
        Radial coordinate(s).
    M_BH : `~astropy.units.Quantity`
        Black hole mass.
    mdot : `~astropy.units.Quantity`
        Mass accretion rate :math:`\dot{M}`.
    R_in : `~astropy.units.Quantity`
        Inner truncation radius.

    Returns
    -------
    `~astropy.units.Quantity`
        Effective temperature :math:`T_{\rm eff}` [K].
    """
    log_r = np.log(np.asarray(ensure_in_units(R, "cm"), dtype=float))
    log_M_BH = float(np.log(ensure_in_units(M_BH, "g")))
    log_M_dot = float(np.log(ensure_in_units(mdot, "g/s")))
    log_R_in = float(np.log(ensure_in_units(R_in, "cm")))

    log_T = np.asarray(_log_disk_effective_temperature(log_r, log_M_BH, log_M_dot, log_R_in))
    result = np.exp(log_T)
    return (result.item() if result.ndim == 0 else result) * u.K


def disk_bolometric_luminosity(
    M_BH: u.Quantity,
    mdot: u.Quantity,
    R_in: u.Quantity,
) -> u.Quantity:
    r"""
    Bolometric luminosity of a steady-state Shakura-Sunyaev disk.

    Analytic result from integrating the viscous dissipation profile over both
    faces with a zero-torque inner boundary and :math:`R_{\rm out} \to \infty`:

    .. math::

        L_{\rm bol} = \frac{G\,M_{\rm BH}\,\dot{M}}{2\,R_{\rm in}}

    Parameters
    ----------
    M_BH : `~astropy.units.Quantity`
        Black hole mass.
    mdot : `~astropy.units.Quantity`
        Mass accretion rate :math:`\dot{M}`.
    R_in : `~astropy.units.Quantity`
        Inner truncation radius (e.g. ISCO).

    Returns
    -------
    `~astropy.units.Quantity`
        :math:`L_{\rm bol}` [erg s⁻¹].

    Notes
    -----
    This is equivalent to an efficiency :math:`\eta = R_{\rm S} / (4 R_{\rm in})`
    where :math:`R_{\rm S} = 2 G M / c^2` is the Schwarzschild radius.
    For a Schwarzschild ISCO (:math:`R_{\rm in} = 3 R_{\rm S}`), :math:`\eta \approx 0.083`.
    """
    log_M_BH = float(np.log(ensure_in_units(M_BH, "g")))
    log_M_dot = float(np.log(ensure_in_units(mdot, "g/s")))
    log_R_in = float(np.log(ensure_in_units(R_in, "cm")))

    result = np.exp(_log_disk_bolometric_luminosity(log_M_BH, log_M_dot, log_R_in))
    return float(result) * u.erg / u.s


def disk_spectral_luminosity(
    nu: u.Quantity,
    M_BH: u.Quantity,
    mdot: u.Quantity,
    R_in: u.Quantity,
    R_out: u.Quantity,
    N_r: int = 500,
) -> u.Quantity:
    r"""
    Angle-integrated spectral luminosity of a multi-colour blackbody disk.

    Assumes Lambertian emission from both disk faces:

    .. math::

        L_\nu = 4\pi^2
                \int_{R_{\rm in}}^{R_{\rm out}}
                B_\nu\!\left(T_{\rm eff}(r)\right) r\,\mathrm{d}r

    Parameters
    ----------
    nu : `~astropy.units.Quantity`
        Frequency or frequency array.
    M_BH : `~astropy.units.Quantity`
        Black hole mass.
    mdot : `~astropy.units.Quantity`
        Mass accretion rate :math:`\dot{M}`.
    R_in : `~astropy.units.Quantity`
        Inner truncation radius.
    R_out : `~astropy.units.Quantity`
        Outer disk radius.
    N_r : int, optional
        Number of radial grid points.  Default ``500``.

    Returns
    -------
    `~astropy.units.Quantity`
        :math:`L_\nu` [erg s⁻¹ Hz⁻¹], same shape as *nu*.
    """
    nu_val = ensure_in_units(nu, "Hz")
    scalar_nu = np.ndim(nu_val) == 0
    log_nu = np.log(np.atleast_1d(np.asarray(nu_val, dtype=float)))

    log_M_BH = float(np.log(ensure_in_units(M_BH, "g")))
    log_M_dot = float(np.log(ensure_in_units(mdot, "g/s")))
    log_R_in = float(np.log(ensure_in_units(R_in, "cm")))
    log_R_out = float(np.log(ensure_in_units(R_out, "cm")))

    log_L = _log_disk_spectral_luminosity(log_nu, log_M_BH, log_M_dot, log_R_in, log_R_out, N_r)
    result = np.exp(log_L)
    if scalar_nu:
        result = result.item()
    return result * u.erg / u.s / u.Hz


def disk_spectral_luminosity_iso(
    nu: u.Quantity,
    M_BH: u.Quantity,
    mdot: u.Quantity,
    R_in: u.Quantity,
    R_out: u.Quantity,
    cos_theta: float = 1.0,
    N_r: int = 500,
) -> u.Quantity:
    r"""
    Isotropic-equivalent spectral luminosity of a multi-colour blackbody disk.

    The luminosity an observer infers by assuming isotropic emission
    (:math:`L_{\nu,{\rm iso}} = 4\pi D_L^2 F_\nu`) where :math:`F_\nu` is
    the observed flux from one optically-thick face:

    .. math::

        L_{\nu,{\rm iso}} = 8\pi^2\cos\theta
                            \int_{R_{\rm in}}^{R_{\rm out}}
                            B_\nu\!\left(T_{\rm eff}(r)\right) r\,\mathrm{d}r

    Parameters
    ----------
    nu : `~astropy.units.Quantity`
        Frequency or frequency array.
    M_BH : `~astropy.units.Quantity`
        Black hole mass.
    mdot : `~astropy.units.Quantity`
        Mass accretion rate :math:`\dot{M}`.
    R_in : `~astropy.units.Quantity`
        Inner truncation radius.
    R_out : `~astropy.units.Quantity`
        Outer disk radius.
    cos_theta : float, optional
        Cosine of the inclination angle.  ``1.0`` (default) = face-on.
    N_r : int, optional
        Number of radial grid points.  Default ``500``.

    Returns
    -------
    `~astropy.units.Quantity`
        :math:`L_{\nu,{\rm iso}}` [erg s⁻¹ Hz⁻¹], same shape as *nu*.
    """
    nu_val = ensure_in_units(nu, "Hz")
    scalar_nu = np.ndim(nu_val) == 0
    log_nu = np.log(np.atleast_1d(np.asarray(nu_val, dtype=float)))

    log_M_BH = float(np.log(ensure_in_units(M_BH, "g")))
    log_M_dot = float(np.log(ensure_in_units(mdot, "g/s")))
    log_R_in = float(np.log(ensure_in_units(R_in, "cm")))
    log_R_out = float(np.log(ensure_in_units(R_out, "cm")))

    log_L = _log_disk_spectral_luminosity_iso(log_nu, log_M_BH, log_M_dot, log_R_in, log_R_out, cos_theta, N_r)
    result = np.exp(log_L)
    if scalar_nu:
        result = result.item()
    return result * u.erg / u.s / u.Hz


def disk_flux_density(
    nu: u.Quantity,
    M_BH: u.Quantity,
    mdot: u.Quantity,
    R_in: u.Quantity,
    R_out: u.Quantity,
    D_L: u.Quantity,
    cos_theta: float = 1.0,
    N_r: int = 500,
) -> u.Quantity:
    r"""
    Observed flux density of a multi-colour blackbody disk (one face).

    For an optically thick disk the observer sees one face.  For a Lambertian
    surface at inclination :math:`\theta` from the disk normal:

    .. math::

        F_\nu = \frac{2\pi\cos\theta}{D_L^2}
                \int_{R_{\rm in}}^{R_{\rm out}}
                B_\nu\!\left(T_{\rm eff}(r)\right) r\,\mathrm{d}r

    Parameters
    ----------
    nu : `~astropy.units.Quantity`
        Observed frequency or frequency array.
    M_BH : `~astropy.units.Quantity`
        Black hole mass.
    mdot : `~astropy.units.Quantity`
        Mass accretion rate :math:`\dot{M}`.
    R_in : `~astropy.units.Quantity`
        Inner truncation radius (e.g. ISCO).
    R_out : `~astropy.units.Quantity`
        Outer disk radius.
    D_L : `~astropy.units.Quantity`
        Luminosity distance.
    cos_theta : float, optional
        Cosine of the inclination angle.  ``1.0`` (default) = face-on.
    N_r : int, optional
        Number of radial grid points.  Default ``500``.

    Returns
    -------
    `~astropy.units.Quantity`
        :math:`F_\nu` [erg s⁻¹ Hz⁻¹ cm⁻²], same shape as *nu*.
    """
    nu_val = ensure_in_units(nu, "Hz")
    scalar_nu = np.ndim(nu_val) == 0
    log_nu = np.log(np.atleast_1d(np.asarray(nu_val, dtype=float)))

    log_M_BH = float(np.log(ensure_in_units(M_BH, "g")))
    log_M_dot = float(np.log(ensure_in_units(mdot, "g/s")))
    log_R_in = float(np.log(ensure_in_units(R_in, "cm")))
    log_R_out = float(np.log(ensure_in_units(R_out, "cm")))
    log_DL = float(np.log(ensure_in_units(D_L, "cm")))

    log_F = _log_disk_flux_density(log_nu, log_M_BH, log_M_dot, log_R_in, log_R_out, log_DL, cos_theta, N_r)
    result = np.exp(log_F)
    if scalar_nu:
        result = result.item()
    return result * u.erg / u.s / u.Hz / u.cm**2


# =============================================================================
# Abstract base class
# =============================================================================
class ThinDiskBase(ABC):
    """
    Abstract base class for steady-state thin-disk accretion models.

    A :class:`ThinDiskBase` encapsulates the physical scalings needed to evaluate
    the local disk structure (temperature, surface density, scale height, etc.) at
    an arbitrary radius given the global disk parameters.  Subclasses implement the
    concrete scaling relations in :meth:`_compute_cgs`.

    The class also provides shared, physics-agnostic methods for:

    - :meth:`compute_effective_temperature` — T_eff(r) from the first-principles
      viscous dissipation formula (independent of the specific scaling relations).
    - :meth:`compute_sed` — multi-colour blackbody SED by numerically integrating
      the Planck function over the disk annuli.
    - :meth:`compute_bolometric_luminosity` — analytic L_bol = G M Ṁ / (2 R_in).

    Design Notes
    ------------
    The base class is **stateless** with respect to the disk physical parameters:
    ``M_BH``, ``mdot``, and ``R_in`` are passed at call time, not stored at
    construction.  Subclass ``__init__`` methods may cache model-level constants
    (e.g. the viscosity parameter α) but must not cache physical state.
    """

    # =========================================================================
    # Initialisation
    # =========================================================================
    @abstractmethod
    def __init__(self, **kwargs):
        """
        Instantiate the disk engine.

        Parameters
        ----------
        kwargs
            Model-level configuration (e.g. viscosity parameter ``alpha``).
            Physical disk state must **not** be stored here.
        """
        pass

    # =========================================================================
    # Core structural computation
    # =========================================================================
    def compute(
        self,
        radius: "_UnitBearingArrayLike",
        M_BH: "_UnitBearingArrayLike",
        mdot: "_UnitBearingArrayLike",
        R_in: "_UnitBearingArrayLike",
    ) -> "dict[str, Union[u.Quantity, _ArrayLike]]":
        r"""
        Evaluate the disk structure at the given radius (or radii).

        This is the high-level, unit-aware wrapper around :meth:`_compute_cgs`.
        If the inputs carry :class:`~astropy.units.Quantity` units they are
        converted internally; otherwise CGS units are assumed.

        Parameters
        ----------
        radius : array-like or `~astropy.units.Quantity`
            Radial coordinate(s) [cm or Quantity].
        M_BH : float or `~astropy.units.Quantity`
            Black-hole (central object) mass [g or Quantity].
        mdot : float or `~astropy.units.Quantity`
            Mass accretion rate :math:`\dot{M}` [g s⁻¹ or Quantity].
        R_in : float or `~astropy.units.Quantity`
            Inner truncation radius (e.g. ISCO) [cm or Quantity].

        Returns
        -------
        dict
            Keys and units:

            - ``"Sigma"``  : surface density [g cm⁻²]
            - ``"T_c"``    : midplane temperature [K]
            - ``"H"``      : pressure scale height [cm]
            - ``"rho"``    : midplane mass density [g cm⁻³]
            - ``"tau"``    : vertical optical depth [dimensionless]
            - ``"nu"``     : kinematic viscosity [cm² s⁻¹]
            - ``"v_R"``    : radial drift velocity [cm s⁻¹]
        """
        radius_cm = np.asarray(ensure_in_units(radius, "cm"), dtype=float)
        M_BH_g = float(ensure_in_units(M_BH, "g"))
        mdot_gs = float(ensure_in_units(mdot, "g/s"))
        R_in_cm = float(ensure_in_units(R_in, "cm"))

        raw = self._compute_cgs(radius_cm, M_BH_g, mdot_gs, R_in_cm)

        return {
            "Sigma": raw["Sigma"] * u.g / u.cm**2,
            "T_c": raw["T_c"] * u.K,
            "H": raw["H"] * u.cm,
            "rho": raw["rho"] * u.g / u.cm**3,
            "tau": raw["tau"],
            "nu": raw["nu"] * u.cm**2 / u.s,
            "v_R": raw["v_R"] * u.cm / u.s,
        }

    @abstractmethod
    def _compute_cgs(
        self,
        radius_cm: np.ndarray,
        M_BH_g: float,
        mdot_gs: float,
        R_in_cm: float,
    ) -> "dict[str, np.ndarray]":
        """
        Low-level CGS evaluation of the disk structure.

        All inputs and outputs are plain floats / NumPy arrays in CGS.
        Override this method in subclasses to implement the specific
        scaling relations.

        Parameters
        ----------
        radius_cm : ndarray
            Radial coordinate(s) in cm.
        M_BH_g : float
            Black-hole mass in grams.
        mdot_gs : float
            Accretion rate in g s⁻¹.
        R_in_cm : float
            Inner truncation radius in cm.

        Returns
        -------
        dict
            Keys: ``"Sigma"``, ``"T_c"``, ``"H"``, ``"rho"``, ``"tau"``,
            ``"nu"``, ``"v_R"`` — all CGS, no units attached.
        """
        pass

    # =========================================================================
    # Effective temperature profile
    # =========================================================================
    def compute_effective_temperature(
        self,
        radius: "_UnitBearingArrayLike",
        M_BH: "_UnitBearingArrayLike",
        mdot: "_UnitBearingArrayLike",
        R_in: "_UnitBearingArrayLike",
    ) -> u.Quantity:
        r"""
        Effective (surface) temperature profile of the steady-state disk.

        Computed from the first-principles viscous dissipation formula,
        independent of the specific scaling-relation prescription:

        .. math::

            T_{\rm eff}(r) = \left(\frac{3\,G\,M_{\rm BH}\,\dot{M}}
                                   {8\pi\,\sigma_{\rm SB}\,r^3}
                             \left[1-\sqrt{\frac{R_{\rm in}}{r}}\right]
                             \right)^{1/4}

        This differs from the midplane temperature :math:`T_c` returned by
        :meth:`compute`; the two are related by
        :math:`T_c = \bigl(\tfrac{3\tau}{4}\bigr)^{1/4} T_{\rm eff}`.

        Parameters
        ----------
        radius : array-like or `~astropy.units.Quantity`
            Radial coordinate(s).
        M_BH : float or `~astropy.units.Quantity`
            Black-hole mass.
        mdot : float or `~astropy.units.Quantity`
            Mass accretion rate.
        R_in : float or `~astropy.units.Quantity`
            Inner truncation radius.

        Returns
        -------
        `~astropy.units.Quantity`
            Effective temperature :math:`T_{\rm eff}(r)` in K.
        """
        radius_val = ensure_in_units(radius, "cm")
        scalar_radius = np.ndim(radius_val) == 0
        log_r = np.log(np.atleast_1d(np.asarray(radius_val, dtype=float)))
        log_M_BH = float(np.log(ensure_in_units(M_BH, "g")))
        log_M_dot = float(np.log(ensure_in_units(mdot, "g/s")))
        log_R_in = float(np.log(ensure_in_units(R_in, "cm")))

        log_T = np.asarray(_log_disk_effective_temperature(log_r, log_M_BH, log_M_dot, log_R_in))
        result = np.exp(log_T)
        if scalar_radius:
            result = result.item()
        return result * u.K

    # =========================================================================
    # SED computation
    # =========================================================================
    def compute_sed(
        self,
        nu: "_UnitBearingArrayLike",
        M_BH: "_UnitBearingArrayLike",
        mdot: "_UnitBearingArrayLike",
        R_in: "_UnitBearingArrayLike",
        R_out: "_UnitBearingArrayLike",
        D_L: "Union[_UnitBearingArrayLike, None]" = None,
        cos_theta: float = 1.0,
        N_r: int = 500,
    ) -> "dict[str, u.Quantity]":
        r"""
        Multi-colour blackbody spectral energy distribution of the disk.

        Integrates the Planck function weighted by the local effective temperature
        :math:`T_{\rm eff}(r)` over all annuli from :math:`R_{\rm in}` to
        :math:`R_{\rm out}`:

        .. math::

            L_\nu = 4\pi^2
                    \int_{R_{\rm in}}^{R_{\rm out}}
                    B_\nu\!\left[T_{\rm eff}(r)\right]\,r\,\mathrm{d}r

        If a luminosity distance ``D_L`` is supplied the observed flux density is
        also returned:

        .. math::

            F_\nu = \frac{4\pi\cos\theta}{D_L^2}
                    \int_{R_{\rm in}}^{R_{\rm out}}
                    B_\nu\!\left[T_{\rm eff}(r)\right]\,r\,\mathrm{d}r

        The radial integral is evaluated on a log-spaced grid using the
        trapezoidal rule (see :func:`_disk_planck_ring_integral`).

        Parameters
        ----------
        nu : array-like or `~astropy.units.Quantity`
            Frequency grid [Hz or Quantity].
        M_BH : float or `~astropy.units.Quantity`
            Black-hole mass.
        mdot : float or `~astropy.units.Quantity`
            Mass accretion rate.
        R_in : float or `~astropy.units.Quantity`
            Inner truncation radius.
        R_out : float or `~astropy.units.Quantity`
            Outer disk radius.
        D_L : float or `~astropy.units.Quantity`, optional
            Luminosity distance.  If provided, ``"F_nu"`` is included in the
            returned dict.
        cos_theta : float, optional
            Cosine of the inclination angle (1 = face-on).
        N_r : int, optional
            Number of radial quadrature points (default 500).

        Returns
        -------
        dict
            - ``"L_nu"`` : spectral luminosity [erg s⁻¹ Hz⁻¹] (always present)
            - ``"F_nu"`` : flux density [erg s⁻¹ Hz⁻¹ cm⁻²] (only if *D_L* given)
        """
        nu_val = ensure_in_units(nu, "Hz")
        scalar_nu = np.ndim(nu_val) == 0
        log_nu = np.log(np.atleast_1d(np.asarray(nu_val, dtype=float)))

        log_M_BH = float(np.log(ensure_in_units(M_BH, "g")))
        log_M_dot = float(np.log(ensure_in_units(mdot, "g/s")))
        log_R_in = float(np.log(ensure_in_units(R_in, "cm")))
        log_R_out = float(np.log(ensure_in_units(R_out, "cm")))

        log_L = _log_disk_spectral_luminosity(log_nu, log_M_BH, log_M_dot, log_R_in, log_R_out, N_r)
        L_nu_arr = np.exp(log_L)
        if scalar_nu:
            L_nu_arr = L_nu_arr.item()

        out: dict[str, u.Quantity] = {"L_nu": L_nu_arr * u.erg / u.s / u.Hz}

        if D_L is not None:
            log_DL = float(np.log(ensure_in_units(D_L, "cm")))
            log_F = _log_disk_flux_density(log_nu, log_M_BH, log_M_dot, log_R_in, log_R_out, log_DL, cos_theta, N_r)
            F_nu_arr = np.exp(log_F)
            if scalar_nu:
                F_nu_arr = F_nu_arr.item()
            out["F_nu"] = F_nu_arr * u.erg / u.s / u.Hz / u.cm**2

        return out

    # =========================================================================
    # Bolometric luminosity
    # =========================================================================
    def compute_bolometric_luminosity(
        self,
        M_BH: "_UnitBearingArrayLike",
        mdot: "_UnitBearingArrayLike",
        R_in: "_UnitBearingArrayLike",
    ) -> u.Quantity:
        r"""
        Analytic bolometric luminosity of the steady-state thin disk.

        Integrating the viscous dissipation profile :math:`Q^+(r)` over both
        disk faces with a zero-torque inner boundary and
        :math:`R_{\rm out} \to \infty` gives the exact result:

        .. math::

            L_{\rm bol} = \frac{G\,M_{\rm BH}\,\dot{M}}{2\,R_{\rm in}}

        This is equivalent to half the gravitational energy released as material
        falls from infinity to :math:`R_{\rm in}`.

        Parameters
        ----------
        M_BH : float or `~astropy.units.Quantity`
            Black-hole mass.
        mdot : float or `~astropy.units.Quantity`
            Mass accretion rate :math:`\dot{M}`.
        R_in : float or `~astropy.units.Quantity`
            Inner truncation radius.

        Returns
        -------
        `~astropy.units.Quantity`
            Bolometric luminosity :math:`L_{\rm bol}` [erg s⁻¹].
        """
        log_M_BH = float(np.log(ensure_in_units(M_BH, "g")))
        log_M_dot = float(np.log(ensure_in_units(mdot, "g/s")))
        log_R_in = float(np.log(ensure_in_units(R_in, "cm")))

        log_L = _log_disk_bolometric_luminosity(log_M_BH, log_M_dot, log_R_in)
        return float(np.exp(log_L)) * u.erg / u.s

    # =========================================================================
    # Dunder methods
    # =========================================================================
    def __call__(
        self,
        radius: "_UnitBearingArrayLike",
        M_BH: "_UnitBearingArrayLike",
        mdot: "_UnitBearingArrayLike",
        R_in: "_UnitBearingArrayLike",
    ) -> "dict[str, Union[u.Quantity, _ArrayLike]]":
        """
        Evaluate the disk structure (alias for :meth:`compute`).

        See :meth:`compute` for full parameter and return documentation.
        """
        return self.compute(radius, M_BH, mdot, R_in)


# =============================================================================
# Sunyaev-Shakura alpha disk
# =============================================================================
class AlphaDisk(ThinDiskBase):
    r"""
    Shakura-Sunyaev alpha-disk model (SS73 / FKR zone-C scalings).

    This model assumes a geometrically thin, optically thick accretion disk in
    which viscosity is parameterised by the dimensionless Shakura-Sunyaev
    :math:`\alpha` parameter.  The disk structure is determined by the balance
    of viscous heating and radiative cooling, yielding the following analytical
    scalings at radius :math:`R`:

    .. math::

        \begin{aligned}
        \Sigma &\;=\; 5.2\;
        \alpha^{-4/5}\,
        \dot{M}_{16}^{7/10}\,
        M_1^{1/4}\,
        R_{10}^{-3/4}\,
        f^{14/5}
        \quad [\mathrm{g\,cm^{-2}}],
        \\[4pt]
        T_c &\;=\; 1.4\times10^{4}\;
        \alpha^{-1/5}\,
        \dot{M}_{16}^{3/10}\,
        M_1^{1/4}\,
        R_{10}^{-3/4}\,
        f^{6/5}
        \quad [\mathrm{K}],
        \\[4pt]
        H &\;=\; 1.7\times10^{8}\;
        \alpha^{-1/10}\,
        \dot{M}_{16}^{3/20}\,
        M_1^{-3/8}\,
        R_{10}^{9/8}\,
        f^{3/5}
        \quad [\mathrm{cm}],
        \\[4pt]
        \rho &\;=\; 3.1\times10^{-8}\;
        \alpha^{-7/10}\,
        \dot{M}_{16}^{11/20}\,
        M_1^{5/8}\,
        R_{10}^{-15/8}\,
        f^{11/5}
        \quad [\mathrm{g\,cm^{-3}}],
        \\[4pt]
        \tau &\;=\; 190\;
        \alpha^{-4/5}\,
        \dot{M}_{16}^{1/5}\,
        f^{4/5},
        \\[4pt]
        \nu &\;=\; 1.8\times10^{14}\;
        \alpha^{4/5}\,
        \dot{M}_{16}^{3/10}\,
        M_1^{-1/4}\,
        R_{10}^{3/4}\,
        f^{6/5}
        \quad [\mathrm{cm^2\,s^{-1}}],
        \\[4pt]
        v_R &\;=\; 2.7\times10^{4}\;
        \alpha^{4/5}\,
        \dot{M}_{16}^{3/10}\,
        M_1^{-1/4}\,
        R_{10}^{-1/4}\,
        f^{-14/5}
        \quad [\mathrm{cm\,s^{-1}}],
        \end{aligned}

    where :math:`\dot{M}_{16} \equiv \dot{M}/10^{16}\ \mathrm{g\,s^{-1}}`,
    :math:`M_1 \equiv M_{\rm BH}/M_\odot`,
    :math:`R_{10} \equiv R/10^{10}\ \mathrm{cm}`, and

    .. math::

        f^4 \;\equiv\; 1 - \sqrt{\frac{R_{\rm in}}{R}}.

    Parameters
    ----------
    alpha : float
        Shakura-Sunyaev dimensionless viscosity parameter
        (:math:`0 < \alpha \lesssim 1`).

    Examples
    --------
    Evaluate the disk structure at three radii for a :math:`10\,M_\odot` black hole:

    .. code-block:: python

        import astropy.units as u
        from astropy import constants as const
        from triceratops.dynamics.accretion import (
            AlphaDisk,
        )

        disk = AlphaDisk(alpha=0.1)
        R = [1e9, 1e10, 1e11] * u.cm
        result = disk.compute(
            R,
            10 * const.M_sun,
            1e16 * u.g / u.s,
            3e6 * u.cm,
        )
        print(
            result["T_c"]
        )  # midplane temperature at each radius

    Compute the bolometric luminosity and a broadband SED:

    .. code-block:: python

        import numpy as np

        L_bol = disk.compute_bolometric_luminosity(
            10 * const.M_sun, 1e16 * u.g / u.s, 3e6 * u.cm
        )

        nu = np.geomspace(1e13, 1e17, 300) * u.Hz
        sed = disk.compute_sed(
            nu,
            10 * const.M_sun,
            1e16 * u.g / u.s,
            R_in=3e6 * u.cm,
            R_out=1e12 * u.cm,
            D_L=100 * u.Mpc,
        )
        print(sed["F_nu"])

    References
    ----------
    Shakura, N. I., & Sunyaev, R. A. 1973, A&A, 24, 337.
    """

    def __init__(self, alpha: float):
        """
        Instantiate an AlphaDisk with the given viscosity parameter.

        Parameters
        ----------
        alpha : float
            Shakura-Sunyaev viscosity parameter.
        """
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must satisfy 0 < alpha <= 1; got {alpha}.")
        self.alpha = float(alpha)
        self._log_alpha = float(np.log(alpha))

    # -------------------------------------------------------------------------
    # Low-level CGS implementation
    # -------------------------------------------------------------------------
    def _compute_cgs(
        self,
        radius_cm: np.ndarray,
        M_BH_g: float,
        mdot_gs: float,
        R_in_cm: float,
    ) -> "dict[str, np.ndarray]":
        """
        Evaluate the Shakura-Sunyaev scalings in CGS.

        Parameters
        ----------
        radius_cm : ndarray
            Radial coordinate(s) in cm.  Must satisfy ``radius_cm >= R_in_cm``.
        M_BH_g : float
            Black-hole mass in grams.
        mdot_gs : float
            Accretion rate in g s⁻¹.
        R_in_cm : float
            Inner truncation radius in cm.

        Returns
        -------
        dict
            Keys: ``"Sigma"``, ``"T_c"``, ``"H"``, ``"rho"``, ``"tau"``,
            ``"nu"``, ``"v_R"`` (all CGS, plain NumPy values).
        """
        # Work entirely in log-space: replace multiplications with additions
        # and power operations with scalar multiplications.
        log_r = np.log(radius_cm)
        log_Mdot_16 = np.log(mdot_gs) - _LOG_MDOT_NORM
        log_M1 = np.log(M_BH_g) - _LOG_M_SUN
        log_R10 = log_r - _LOG_R_NORM
        log_alpha = self._log_alpha

        # Inner-boundary correction: f⁴ = max(1 − √(R_in/R), 0),  log_f = ¼ log(f⁴)
        with np.errstate(invalid="ignore"):
            xi = 1.0 - np.exp(0.5 * (np.log(R_in_cm) - log_r))
        log_f = np.where(xi > 0.0, 0.25 * np.log(xi), -np.inf)

        # --- SS73 scalings as log-space fused multiply-adds ---
        log_Sigma = (
            _LOG_C_SIGMA
            + (-4.0 / 5.0) * log_alpha
            + (7.0 / 10.0) * log_Mdot_16
            + (1.0 / 4.0) * log_M1
            + (-3.0 / 4.0) * log_R10
            + (14.0 / 5.0) * log_f
        )  # noqa: E501
        log_Tc = (
            _LOG_C_TC
            + (-1.0 / 5.0) * log_alpha
            + (3.0 / 10.0) * log_Mdot_16
            + (1.0 / 4.0) * log_M1
            + (-3.0 / 4.0) * log_R10
            + (6.0 / 5.0) * log_f
        )  # noqa: E501
        log_H = (
            _LOG_C_H
            + (-1.0 / 10.0) * log_alpha
            + (3.0 / 20.0) * log_Mdot_16
            + (-3.0 / 8.0) * log_M1
            + (9.0 / 8.0) * log_R10
            + (3.0 / 5.0) * log_f
        )  # noqa: E501
        log_rho = (
            _LOG_C_RHO
            + (-7.0 / 10.0) * log_alpha
            + (11.0 / 20.0) * log_Mdot_16
            + (5.0 / 8.0) * log_M1
            + (-15.0 / 8.0) * log_R10
            + (11.0 / 5.0) * log_f
        )  # noqa: E501
        log_tau = _LOG_C_TAU + (-4.0 / 5.0) * log_alpha + (1.0 / 5.0) * log_Mdot_16 + (4.0 / 5.0) * log_f  # noqa: E501
        log_nu = (
            _LOG_C_NU
            + (4.0 / 5.0) * log_alpha
            + (3.0 / 10.0) * log_Mdot_16
            + (-1.0 / 4.0) * log_M1
            + (3.0 / 4.0) * log_R10
            + (6.0 / 5.0) * log_f
        )  # noqa: E501
        log_vR = (
            _LOG_C_VR
            + (4.0 / 5.0) * log_alpha
            + (3.0 / 10.0) * log_Mdot_16
            + (-1.0 / 4.0) * log_M1
            + (-1.0 / 4.0) * log_R10
            + (-14.0 / 5.0) * log_f
        )  # noqa: E501

        return {
            "Sigma": np.exp(log_Sigma),
            "T_c": np.exp(log_Tc),
            "H": np.exp(log_H),
            "rho": np.exp(log_rho),
            "tau": np.exp(log_tau),
            "nu": np.exp(log_nu),
            "v_R": np.exp(log_vR),
        }

    # -------------------------------------------------------------------------
    # Dunder methods
    # -------------------------------------------------------------------------
    def __str__(self) -> str:
        return f"AlphaDisk(alpha={self.alpha})"

    def __repr__(self) -> str:
        return f"<AlphaDisk(alpha={self.alpha})>"
