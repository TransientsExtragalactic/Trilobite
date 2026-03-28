r"""
Accretion disk utility functions.

This module provides low-level functions covering the core recurring calculations
in thin-disk accretion physics: orbital mechanics, viscous dissipation, thermal
structure, and disk geometry.  All functions follow the two-level API convention:

- Private ``_log_*`` functions operate directly on log-space CGS scalars / arrays
  for numerical efficiency.
- Public functions accept :class:`~astropy.units.Quantity` inputs, perform unit
  coercion internally via :func:`~triceratops.utils.misc_utils.ensure_in_units`,
  and return unit-bearing outputs.

See Also
--------
:mod:`.one_zone.core` : One-zone disk models that consume these utilities.
:ref:`one_zone_disk`, :ref:`one_zone_disk_theory`
"""

from typing import TYPE_CHECKING, Union

import numpy as np
from astropy import constants as const
from astropy import units as u

from triceratops.physics_utils.constants import _log_G_cgs, _log_sigma_sb_cgs
from triceratops.utils.misc_utils import ensure_in_units

if TYPE_CHECKING:
    from triceratops._typing import _ArrayLike


# ================================================================== #
# Private (log-space CGS)                                            #
# ================================================================== #
def _log_omega_K(log_M_BH: "_ArrayLike", log_R: "_ArrayLike") -> "_ArrayLike":
    r"""
    Compute :math:`\ln \Omega_K` from the BH mass and radius.

    .. math::

        \Omega_K = \sqrt{\frac{G M_{\rm BH}}{R^3}}
        \implies
        \ln \Omega_K = \tfrac{1}{2}\left(\ln G + \ln M_{\rm BH} - 3\ln R\right)
    """
    return 0.5 * (_log_G_cgs + log_M_BH - 3.0 * log_R)


def _log_visc_dissipation_rate(log_nu: "_ArrayLike", log_sigma: "_ArrayLike", log_omega: "_ArrayLike") -> "_ArrayLike":
    r"""
    Compute :math:`\ln Q^+` from viscosity, surface density, and angular velocity.

    .. math::

        Q^+ = \frac{9}{8} \nu \Sigma \Omega^2
    """
    return np.log(9.0 / 8.0) + log_nu + log_sigma + 2.0 * log_omega


def _compute_log_f(log_R: "_ArrayLike", log_R_in: "_ArrayLike") -> "_ArrayLike":
    r"""Compute :math:`\ln f(R)` where :math:`f(R) = 1 - \sqrt{R_{\rm in}/R}`."""
    return np.log(1.0 - np.sqrt(np.exp(log_R_in - log_R)))


def _log_visc_dissipation_rate_from_mdot(
    log_M_BH: "_ArrayLike",
    log_M_dot: "_ArrayLike",
    log_R: "_ArrayLike",
    log_R_in: "_ArrayLike",
) -> "_ArrayLike":
    r"""
    Compute :math:`\ln Q^+` from the accretion rate in steady state.

    .. math::

        Q^+ = \frac{3 G M_{\rm BH} \dot{M}}{8\pi R^3}
              \left(1 - \sqrt{\frac{R_{\rm in}}{R}}\right)
    """
    return (
        np.log(3.0 / (8.0 * np.pi)) + _log_G_cgs + log_M_BH + log_M_dot - 3.0 * log_R + _compute_log_f(log_R, log_R_in)
    )


def _log_T_eff_from_visc_dissipation_rate(log_Q_plus: "_ArrayLike") -> "_ArrayLike":
    r"""
    Compute :math:`\ln T_{\rm eff}` from the surface dissipation rate.

    .. math::

        T_{\rm eff} = \left(\frac{Q^+}{\sigma_{\rm SB}}\right)^{1/4}
    """
    return 0.25 * (log_Q_plus - _log_sigma_sb_cgs)


def _log_T_eff_from_nu(log_nu: "_ArrayLike", log_sigma: "_ArrayLike", log_omega: "_ArrayLike") -> "_ArrayLike":
    r"""Compute :math:`\ln T_{\rm eff}` directly from viscosity, surface density, and :math:`\Omega`."""
    log_Q_plus = _log_visc_dissipation_rate(log_nu, log_sigma, log_omega)
    return _log_T_eff_from_visc_dissipation_rate(log_Q_plus)


def _log_alpha_viscosity(log_cs_sq: "_ArrayLike", log_omega: "_ArrayLike", alpha: float) -> "_ArrayLike":
    r"""
    Compute :math:`\ln \nu` under the Shakura-Sunyaev alpha prescription.

    .. math::

        \nu = \alpha \frac{c_s^2}{\Omega}
    """
    return np.log(alpha) + log_cs_sq - log_omega


def _log_viscous_timescale(log_R: "_ArrayLike", log_nu: "_ArrayLike") -> "_ArrayLike":
    r"""Compute :math:`\ln t_{\rm visc}` where :math:`t_{\rm visc} = R^2 / \nu`."""
    return 2.0 * log_R - log_nu


def _log_scale_height(log_cs: "_ArrayLike", log_omega: "_ArrayLike") -> "_ArrayLike":
    r"""Compute :math:`\ln H` where :math:`H = c_s / \Omega`."""
    return log_cs - log_omega


def _log_midplane_density(log_sigma: "_ArrayLike", log_H: "_ArrayLike") -> "_ArrayLike":
    r"""Compute :math:`\ln \rho_{\rm mid}` where :math:`\rho_{\rm mid} = \Sigma / (2H)`."""
    return log_sigma - np.log(2.0) - log_H


def _log_optical_depth(log_sigma: "_ArrayLike", log_kappa: "_ArrayLike") -> "_ArrayLike":
    r"""Compute :math:`\ln \tau` where :math:`\tau = \Sigma \kappa`."""
    return log_sigma + log_kappa


def _log_central_temperature(log_T_eff: "_ArrayLike", log_tau: "_ArrayLike") -> "_ArrayLike":
    r"""
    Compute :math:`\ln T_c` for an optically thick blackbody disk midplane.

    .. math::

        T_c = \left(\frac{3\tau}{4}\right)^{1/4} T_{\rm eff}
        \implies
        \ln T_c = \tfrac{1}{4}\left(\ln\tfrac{3}{4} + \ln\tau\right) + \ln T_{\rm eff}
    """
    return 0.25 * (np.log(3.0 / 4.0) + log_tau) + log_T_eff


# ================================================================== #
# Public API                                                         #
# ================================================================== #
def alpha_viscosity(c_s: u.Quantity, Omega: u.Quantity, alpha: float) -> u.Quantity:
    r"""
    Compute the kinematic viscosity under the Shakura-Sunyaev alpha prescription.

    .. math::

        \nu = \alpha \frac{c_s^2}{\Omega}

    Parameters
    ----------
    c_s : `~astropy.units.Quantity`
        Isothermal sound speed :math:`c_s`.
    Omega : `~astropy.units.Quantity`
        Angular velocity :math:`\Omega`.
    alpha : float
        Shakura-Sunyaev viscosity parameter (dimensionless,
        :math:`0 < \alpha \leq 1`).

    Returns
    -------
    `~astropy.units.Quantity`
        Kinematic viscosity :math:`\nu` [:math:`\text{cm}^2\,\text{s}^{-1}`].

    Examples
    --------
    .. code-block:: python

        import astropy.units as u

        alpha_viscosity(
            1e6 * u.cm / u.s,
            1e-2 * u.rad / u.s,
            alpha=0.1,
        )
    """
    log_cs = np.log(ensure_in_units(c_s, "cm/s"))
    log_omega = np.log(ensure_in_units(Omega, "rad/s"))

    log_nu = np.asarray(_log_alpha_viscosity(2.0 * log_cs, log_omega, alpha))

    result = np.exp(log_nu)
    return (result.item() if result.ndim == 0 else result) * u.cm**2 / u.s


def viscous_timescale(R: u.Quantity, nu: u.Quantity) -> u.Quantity:
    r"""
    Compute the viscous (spreading) timescale.

    .. math::

        t_{\rm visc} = \frac{R^2}{\nu}

    Parameters
    ----------
    R : `~astropy.units.Quantity`
        Disk radius :math:`R`.
    nu : `~astropy.units.Quantity`
        Kinematic viscosity :math:`\nu`.

    Returns
    -------
    `~astropy.units.Quantity`
        Viscous timescale :math:`t_{\rm visc}` [s].

    Examples
    --------
    .. code-block:: python

        import astropy.units as u

        viscous_timescale(
            1e10 * u.cm, 1e14 * u.cm**2 / u.s
        )
    """
    log_R = np.log(ensure_in_units(R, "cm"))
    log_nu = np.log(ensure_in_units(nu, "cm^2/s"))

    log_t = np.asarray(_log_viscous_timescale(log_R, log_nu))

    result = np.exp(log_t)
    return (result.item() if result.ndim == 0 else result) * u.s


def keplerian_angular_velocity(M_BH: u.Quantity, R: u.Quantity) -> u.Quantity:
    r"""
    Compute the Keplerian angular velocity at radius :math:`R`.

    .. math::

        \Omega_K = \sqrt{\frac{G M_{\rm BH}}{R^3}}

    Parameters
    ----------
    M_BH : `~astropy.units.Quantity`
        Black hole (central object) mass.
    R : `~astropy.units.Quantity`
        Orbital radius.

    Returns
    -------
    `~astropy.units.Quantity`
        Keplerian angular velocity :math:`\Omega_K` [:math:`\text{rad s}^{-1}`].

    Examples
    --------
    .. code-block:: python

        import astropy.units as u
        from astropy import constants as const

        M_BH = 3 * const.M_sun
        R = 1e10 * u.cm
        keplerian_angular_velocity(M_BH, R)
    """
    log_M_BH = np.log(ensure_in_units(M_BH, "g"))
    log_R = np.log(ensure_in_units(R, "cm"))

    log_Omega = np.asarray(_log_omega_K(log_M_BH, log_R))

    result = np.exp(log_Omega)
    return (result.item() if result.ndim == 0 else result) / u.s


def inner_boundary_correction(R: u.Quantity, R_in: u.Quantity) -> "_ArrayLike":
    r"""
    Compute the dimensionless inner-boundary correction factor.

    .. math::

        f(R) = 1 - \sqrt{\frac{R_{\rm in}}{R}}

    This factor accounts for the zero-torque condition at the inner boundary
    and reduces the viscous dissipation near :math:`R \approx R_{\rm in}`.

    Parameters
    ----------
    R : `~astropy.units.Quantity`
        Disk radius at which to evaluate the correction.
    R_in : `~astropy.units.Quantity`
        Inner truncation radius (e.g. ISCO).  Must satisfy :math:`R \geq R_{\rm in}`.

    Returns
    -------
    float or ndarray
        Correction factor :math:`f(R) \in [0, 1)` (dimensionless).

    Raises
    ------
    ValueError
        If :math:`R < R_{\rm in}`.

    Examples
    --------
    .. code-block:: python

        import astropy.units as u

        inner_boundary_correction(1e10 * u.cm, 1e9 * u.cm)
    """
    log_R = np.log(np.asarray(ensure_in_units(R, "cm"), dtype=float))
    log_R_in = np.log(np.asarray(ensure_in_units(R_in, "cm"), dtype=float))

    if np.any(log_R < log_R_in):
        raise ValueError("R must be >= R_in everywhere.")

    result = np.asarray(np.exp(_compute_log_f(log_R, log_R_in)))
    return result.item() if result.ndim == 0 else result


def viscous_dissipation_rate(nu: u.Quantity, sigma: u.Quantity, Omega: u.Quantity) -> u.Quantity:
    r"""
    Compute the viscous surface dissipation rate.

    .. math::

        Q^+ = \frac{9}{8} \nu \Sigma \Omega^2

    Parameters
    ----------
    nu : `~astropy.units.Quantity`
        Kinematic viscosity :math:`\nu`.
    sigma : `~astropy.units.Quantity`
        Surface mass density :math:`\Sigma`.
    Omega : `~astropy.units.Quantity`
        Angular velocity :math:`\Omega`.

    Returns
    -------
    `~astropy.units.Quantity`
        Viscous dissipation rate :math:`Q^+` [:math:`\text{erg cm}^{-2}\,\text{s}^{-1}`].

    Examples
    --------
    .. code-block:: python

        import astropy.units as u

        viscous_dissipation_rate(
            1e14 * u.cm**2 / u.s,
            1e3 * u.Unit("g/cm^2"),
            1e3 * u.rad / u.s,
        )
    """
    log_nu = np.log(ensure_in_units(nu, "cm^2/s"))
    log_sigma = np.log(ensure_in_units(sigma, "g/cm^2"))
    log_omega = np.log(ensure_in_units(Omega, "rad/s"))

    log_Q = np.asarray(_log_visc_dissipation_rate(log_nu, log_sigma, log_omega))

    result = np.exp(log_Q)
    return (result.item() if result.ndim == 0 else result) * u.erg / u.cm**2 / u.s


def viscous_dissipation_rate_from_mdot(
    M_BH: u.Quantity,
    mdot: u.Quantity,
    R: u.Quantity,
    R_in: u.Quantity,
) -> u.Quantity:
    r"""
    Compute the steady-state viscous dissipation rate from the accretion rate.

    .. math::

        Q^+(R) = \frac{3 G M_{\rm BH} \dot{M}}{8\pi R^3}
                 \left(1 - \sqrt{\frac{R_{\rm in}}{R}}\right)

    Parameters
    ----------
    M_BH : `~astropy.units.Quantity`
        Black hole (central object) mass.
    mdot : `~astropy.units.Quantity`
        Mass accretion rate :math:`\dot{M}`.
    R : `~astropy.units.Quantity`
        Disk radius.
    R_in : `~astropy.units.Quantity`
        Inner truncation radius.

    Returns
    -------
    `~astropy.units.Quantity`
        Viscous dissipation rate :math:`Q^+` [:math:`\text{erg cm}^{-2}\,\text{s}^{-1}`].

    Examples
    --------
    .. code-block:: python

        import astropy.units as u
        from astropy import constants as const

        viscous_dissipation_rate_from_mdot(
            3 * const.M_sun,
            1e-8 * const.M_sun / u.yr,
            1e10 * u.cm,
            3e6 * u.cm,
        )
    """
    log_M_BH = np.log(ensure_in_units(M_BH, "g"))
    log_mdot = np.log(ensure_in_units(mdot, "g/s"))
    log_R = np.log(ensure_in_units(R, "cm"))
    log_R_in = np.log(ensure_in_units(R_in, "cm"))

    log_Q = np.asarray(_log_visc_dissipation_rate_from_mdot(log_M_BH, log_mdot, log_R, log_R_in))

    result = np.exp(log_Q)
    return (result.item() if result.ndim == 0 else result) * u.erg / u.cm**2 / u.s


def effective_temperature(Q_plus: u.Quantity) -> u.Quantity:
    r"""
    Compute the effective (blackbody) surface temperature from the dissipation rate.

    .. math::

        T_{\rm eff} = \left(\frac{Q^+}{\sigma_{\rm SB}}\right)^{1/4}

    Parameters
    ----------
    Q_plus : `~astropy.units.Quantity`
        Surface dissipation rate :math:`Q^+`
        [:math:`\text{erg cm}^{-2}\,\text{s}^{-1}`].

    Returns
    -------
    `~astropy.units.Quantity`
        Effective temperature :math:`T_{\rm eff}` [K].

    Examples
    --------
    .. code-block:: python

        import astropy.units as u

        effective_temperature(
            1e10 * u.erg / u.cm**2 / u.s
        )
    """
    log_Q = np.log(ensure_in_units(Q_plus, "erg / (cm^2 s)"))

    log_T = np.asarray(_log_T_eff_from_visc_dissipation_rate(log_Q))

    result = np.exp(log_T)
    return (result.item() if result.ndim == 0 else result) * u.K


def effective_temperature_from_viscosity(nu: u.Quantity, sigma: u.Quantity, Omega: u.Quantity) -> u.Quantity:
    r"""
    Compute the effective surface temperature directly from viscous parameters.

    Combines :func:`viscous_dissipation_rate` and :func:`effective_temperature`:

    .. math::

        T_{\rm eff} = \left(\frac{9\,\nu\,\Sigma\,\Omega^2}{8\,\sigma_{\rm SB}}\right)^{1/4}

    Parameters
    ----------
    nu : `~astropy.units.Quantity`
        Kinematic viscosity :math:`\nu`.
    sigma : `~astropy.units.Quantity`
        Surface mass density :math:`\Sigma`.
    Omega : `~astropy.units.Quantity`
        Angular velocity :math:`\Omega`.

    Returns
    -------
    `~astropy.units.Quantity`
        Effective temperature :math:`T_{\rm eff}` [K].

    Examples
    --------
    .. code-block:: python

        import astropy.units as u

        effective_temperature_from_viscosity(
            1e14 * u.cm**2 / u.s,
            1e3 * u.Unit("g/cm^2"),
            1e3 * u.rad / u.s,
        )
    """
    log_nu = np.log(ensure_in_units(nu, "cm^2/s"))
    log_sigma = np.log(ensure_in_units(sigma, "g/cm^2"))
    log_omega = np.log(ensure_in_units(Omega, "rad/s"))

    log_T = np.asarray(_log_T_eff_from_nu(log_nu, log_sigma, log_omega))

    result = np.exp(log_T)
    return (result.item() if result.ndim == 0 else result) * u.K


def disk_scale_height(c_s: u.Quantity, Omega: u.Quantity) -> u.Quantity:
    r"""
    Compute the disk pressure scale height.

    .. math::

        H = \frac{c_s}{\Omega}

    Parameters
    ----------
    c_s : `~astropy.units.Quantity`
        Isothermal sound speed :math:`c_s`.
    Omega : `~astropy.units.Quantity`
        Angular velocity :math:`\Omega`.

    Returns
    -------
    `~astropy.units.Quantity`
        Pressure scale height :math:`H` [cm].

    Notes
    -----
    For a Keplerian disk this is equivalent to setting :math:`H/R = c_s / v_K`,
    the thermal-to-orbital velocity ratio.  Thin-disk models assume
    :math:`H \ll R`.

    Examples
    --------
    .. code-block:: python

        import astropy.units as u

        disk_scale_height(
            1e6 * u.cm / u.s, 1e-2 * u.rad / u.s
        )
    """
    log_cs = np.log(ensure_in_units(c_s, "cm/s"))
    log_omega = np.log(ensure_in_units(Omega, "rad/s"))

    log_H = np.asarray(_log_scale_height(log_cs, log_omega))

    result = np.exp(log_H)
    return (result.item() if result.ndim == 0 else result) * u.cm


def disk_aspect_ratio(c_s: u.Quantity, Omega: u.Quantity, R: u.Quantity) -> "_ArrayLike":
    r"""
    Compute the disk aspect ratio :math:`h = H/R`.

    .. math::

        h \equiv \frac{H}{R} = \frac{c_s}{\Omega R} = \frac{c_s}{v_K}

    Parameters
    ----------
    c_s : `~astropy.units.Quantity`
        Isothermal sound speed :math:`c_s`.
    Omega : `~astropy.units.Quantity`
        Angular velocity :math:`\Omega`.
    R : `~astropy.units.Quantity`
        Disk radius :math:`R`.

    Returns
    -------
    float or ndarray
        Aspect ratio :math:`h = H/R` (dimensionless).

    Notes
    -----
    The thin-disk approximation is valid when :math:`h \ll 1`.  For a
    standard :math:`\alpha`-disk, :math:`h` is a slowly varying function of
    radius and accretion rate.

    Examples
    --------
    .. code-block:: python

        import astropy.units as u

        disk_aspect_ratio(
            1e6 * u.cm / u.s,
            1e-2 * u.rad / u.s,
            1e8 * u.cm,
        )
    """
    log_cs = np.log(ensure_in_units(c_s, "cm/s"))
    log_omega = np.log(ensure_in_units(Omega, "rad/s"))
    log_R = np.log(ensure_in_units(R, "cm"))

    log_h = np.asarray(_log_scale_height(log_cs, log_omega) - log_R)

    result = np.exp(log_h)
    return result.item() if result.ndim == 0 else result


def midplane_density(sigma: u.Quantity, H: u.Quantity) -> u.Quantity:
    r"""
    Compute the disk midplane mass density.

    .. math::

        \rho_{\rm mid} = \frac{\Sigma}{2 H}

    Parameters
    ----------
    sigma : `~astropy.units.Quantity`
        Surface mass density :math:`\Sigma`.
    H : `~astropy.units.Quantity`
        Pressure scale height :math:`H`.

    Returns
    -------
    `~astropy.units.Quantity`
        Midplane density :math:`\rho_{\rm mid}` [:math:`\text{g cm}^{-3}`].

    Notes
    -----
    This is the vertically-averaged midplane density for a Gaussian vertical
    profile :math:`\rho(z) = \rho_{\rm mid} \exp(-z^2 / 2H^2)`, evaluated at
    :math:`z = 0`.

    Examples
    --------
    .. code-block:: python

        import astropy.units as u

        midplane_density(
            1e3 * u.Unit("g/cm^2"), 1e6 * u.cm
        )
    """
    log_sigma = np.log(ensure_in_units(sigma, "g/cm^2"))
    log_H = np.log(ensure_in_units(H, "cm"))

    log_rho = np.asarray(_log_midplane_density(log_sigma, log_H))

    result = np.exp(log_rho)
    return (result.item() if result.ndim == 0 else result) * u.g / u.cm**3


def accretion_rate_from_viscous_timescale(
    M_D: u.Quantity,
    t_visc: u.Quantity,
    f_disk: float = 1.0,
) -> u.Quantity:
    r"""
    Estimate the accretion rate from the disk mass and viscous timescale.

    .. math::

        \dot{M} = f_{\rm disk} \frac{M_D}{t_{\rm visc}}

    Parameters
    ----------
    M_D : `~astropy.units.Quantity`
        Disk mass :math:`M_D`.
    t_visc : `~astropy.units.Quantity`
        Viscous timescale :math:`t_{\rm visc}`.
    f_disk : float, optional
        Geometric correction factor (dimensionless).  Default ``1.0``.

    Returns
    -------
    `~astropy.units.Quantity`
        Mass accretion rate :math:`\dot{M}` [:math:`\text{g s}^{-1}`].

    Notes
    -----
    This is the one-zone approximation for the accretion drain.  The factor
    :math:`f_{\rm disk}` encodes assumptions about the radial profile of the
    spreading ring (see :footcite:t:`metzgerTimeDependentModelsAccretion2008`).

    Examples
    --------
    .. code-block:: python

        import astropy.units as u
        from astropy import constants as const

        accretion_rate_from_viscous_timescale(
            0.1 * const.M_sun, 1e3 * u.s
        )
    """
    M_D_cgs = ensure_in_units(M_D, "g")
    t_visc_cgs = ensure_in_units(t_visc, "s")

    result = np.asarray(f_disk * M_D_cgs / t_visc_cgs)
    return (result.item() if result.ndim == 0 else result) * u.g / u.s


def optical_depth(sigma: u.Quantity, kappa: u.Quantity) -> "Union[float, np.ndarray]":
    r"""
    Compute the vertical optical depth.

    .. math::

        \tau = \Sigma \kappa

    Parameters
    ----------
    sigma : `~astropy.units.Quantity`
        Surface mass density :math:`\Sigma`.
    kappa : `~astropy.units.Quantity`
        Opacity :math:`\kappa`.

    Returns
    -------
    float or ndarray
        Optical depth :math:`\tau` (dimensionless).

    Examples
    --------
    .. code-block:: python

        import astropy.units as u

        optical_depth(
            1e3 * u.Unit("g/cm^2"), 0.34 * u.cm**2 / u.g
        )
    """
    log_sigma = np.log(ensure_in_units(sigma, "g/cm^2"))
    log_kappa = np.log(ensure_in_units(kappa, "cm^2/g"))

    result = np.asarray(np.exp(_log_optical_depth(log_sigma, log_kappa)))
    return result.item() if result.ndim == 0 else result


def central_temperature(T_eff: u.Quantity, tau: "Union[float, np.ndarray]") -> u.Quantity:
    r"""
    Compute the disk central (midplane) temperature for an optically thick disk.

    .. math::

        T_c = \left(\frac{3\tau}{4}\right)^{1/4} T_{\rm eff}

    Parameters
    ----------
    T_eff : `~astropy.units.Quantity`
        Effective surface temperature.
    tau : float or ndarray
        Optical depth :math:`\tau` (dimensionless).

    Returns
    -------
    `~astropy.units.Quantity`
        Midplane temperature :math:`T_c` [K].

    Examples
    --------
    .. code-block:: python

        import astropy.units as u

        central_temperature(1e4 * u.K, tau=100.0)
    """
    log_T_eff = np.log(ensure_in_units(T_eff, "K"))
    log_tau = np.log(np.asarray(tau, dtype=float))

    log_T_c = np.asarray(_log_central_temperature(log_T_eff, log_tau))
    result = np.exp(log_T_c)
    return (result.item() if result.ndim == 0 else result) * u.K


# ------------------------------------------------------------------- #
# Emission Features                                                   #
# ------------------------------------------------------------------- #

# Planck-function constants (CGS, module-level for performance)
_h_cgs: float = const.h.cgs.value  # erg s
_k_B_cgs: float = const.k_B.cgs.value  # erg K⁻¹
_c_cgs: float = const.c.cgs.value  # cm s⁻¹

# Pre-computed log combination for the T_eff(r) profile:
#   ln(3 G / (8π σ_SB))
_log_T_disk_coef: float = np.log(3.0) + _log_G_cgs - np.log(8.0 * np.pi) - _log_sigma_sb_cgs


def _log_disk_effective_temperature(
    log_r: "_ArrayLike",
    log_M_BH: float,
    log_M_dot: float,
    log_R_in: float,
) -> "_ArrayLike":
    r"""
    Compute :math:`\ln T_{\rm eff}(r)` for a steady-state thin disk.

    .. math::

        T_{\rm eff}(r) = \left(\frac{3\,G\,M_{\rm BH}\,\dot{M}}
                               {8\pi\,\sigma_{\rm SB}\,r^3}
                         \left[1-\sqrt{\tfrac{R_{\rm in}}{r}}\right]
                         \right)^{1/4}
    """
    log_r = np.asarray(log_r, dtype=float)
    log_coef = 0.25 * (_log_T_disk_coef + log_M_BH + log_M_dot)
    # At r = R_in the correction factor f(r) → 0; log1p(-1) = -∞ (T = 0).
    # Suppress the benign boundary warning.
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
    Evaluate the core radial integral of the Planck function.

    .. math::

        I(\nu) = \int_{R_{\rm in}}^{R_{\rm out}}
                 B_\nu\!\left(T_{\rm eff}(r)\right) r\,\mathrm{d}r
               = \int B_\nu\!\left(T_{\rm eff}(r)\right) r^2\,\mathrm{d}\ln r

    Parameters
    ----------
    log_nu_arr : ndarray, shape (Nν,)
        :math:`\ln\nu` grid (natural log of frequency in Hz).
    log_M_BH, log_M_dot, log_R_in, log_R_out : float
        Natural logs of BH mass (g), accretion rate (g s⁻¹), inner and
        outer radii (cm).
    N_r : int
        Number of radial grid points.

    Returns
    -------
    ndarray, shape (Nν,)
        :math:`I(\nu)` in :math:`\mathrm{erg\,s^{-1}\,Hz^{-1}\,sr^{-1}}`.
    """
    log_r = np.linspace(log_R_in, log_R_out, N_r)
    log_T = np.asarray(_log_disk_effective_temperature(log_r, log_M_BH, log_M_dot, log_R_in))
    T = np.exp(log_T)
    r_sq = np.exp(2.0 * log_r)

    # Broadcast: (Nν, 1) × (1, Nr)
    nu = np.exp(log_nu_arr)[:, None]  # (Nν, 1)
    T_grid = T[None, :]  # (1, Nr)
    r2_grid = r_sq[None, :]  # (1, Nr)

    # At r = R_in, T = 0 (zero-torque BC) → x → ∞ → B_nu → 0.
    # Suppress the resulting divide-by-zero and overflow; they are benign.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        x = (_h_cgs * nu) / (_k_B_cgs * T_grid)
        B_nu = (2.0 * _h_cgs * nu**3 / _c_cgs**2) / np.expm1(x)  # (Nν, Nr)
        B_nu = np.where(np.isfinite(B_nu), B_nu, 0.0)

    return np.trapz(r2_grid * B_nu, x=log_r, axis=1)  # (Nν,)


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
    Compute :math:`\ln F_\nu` for a multi-colour blackbody disk.

    .. math::

        F_\nu = \frac{4\pi\cos\theta}{D_L^2}
                \int_{R_{\rm in}}^{R_{\rm out}} B_\nu(T_{\rm eff}(r))\,r\,\mathrm{d}r

    Returns
    -------
    ndarray, shape (Nν,)
        :math:`\ln F_\nu` [:math:`\mathrm{erg\,s^{-1}\,Hz^{-1}\,cm^{-2}}`].
    """
    integral = _disk_planck_ring_integral(log_nu_arr, log_M_BH, log_M_dot, log_R_in, log_R_out, N_r)
    F_nu = (4.0 * np.pi * cos_theta) * np.exp(-2.0 * log_DL) * integral
    return np.log(F_nu)


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

    Both disk faces are included (Lambertian emitter):

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

    This is what an observer infers by multiplying the observed flux by
    :math:`4\pi D_L^2`, assuming isotropic emission:

    .. math::

        L_{\nu,{\rm iso}} = 4\pi D_L^2\,F_\nu
                          = 16\pi^2\cos\theta
                            \int_{R_{\rm in}}^{R_{\rm out}} B_\nu(T_{\rm eff}(r))\,r\,\mathrm{d}r

    Returns
    -------
    ndarray, shape (Nν,)
        :math:`\ln L_{\nu,{\rm iso}}` [:math:`\mathrm{erg\,s^{-1}\,Hz^{-1}}`].
    """
    integral = _disk_planck_ring_integral(log_nu_arr, log_M_BH, log_M_dot, log_R_in, log_R_out, N_r)
    return np.log(16.0 * np.pi**2 * cos_theta * integral)


def _log_disk_bolometric_luminosity(
    log_M_BH: float,
    log_M_dot: float,
    log_R_in: float,
) -> float:
    r"""
    Analytic bolometric luminosity for a steady-state thin disk.

    Integrating the viscous dissipation profile over both faces with a
    zero-torque inner boundary and :math:`R_{\rm out} \to \infty`:

    .. math::

        L_{\rm bol} = \frac{G\,M_{\rm BH}\,\dot{M}}{2\,R_{\rm in}}

    Returns
    -------
    float
        :math:`\ln L_{\rm bol}` [:math:`\mathrm{erg\,s^{-1}}`].
    """
    return _log_G_cgs + log_M_BH + log_M_dot - np.log(2.0) - log_R_in


# ================================================================== #
# Public emission API                                                 #
# ================================================================== #


def disk_effective_temperature(
    R: u.Quantity,
    M_BH: u.Quantity,
    mdot: u.Quantity,
    R_in: u.Quantity,
) -> u.Quantity:
    r"""
    Effective temperature profile of a steady-state thin disk.

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
        Effective temperature [K].
    """
    log_r = np.log(np.asarray(ensure_in_units(R, "cm"), dtype=float))
    log_M_BH = float(np.log(ensure_in_units(M_BH, "g")))
    log_M_dot = float(np.log(ensure_in_units(mdot, "g/s")))
    log_R_in = float(np.log(ensure_in_units(R_in, "cm")))

    log_T = np.asarray(_log_disk_effective_temperature(log_r, log_M_BH, log_M_dot, log_R_in))
    result = np.exp(log_T)
    return (result.item() if result.ndim == 0 else result) * u.K


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
    Observed flux density of a multi-colour blackbody disk.

    Integrates the Planck function over the disk face, accounting for
    inclination :math:`\cos\theta` and luminosity distance :math:`D_L`:

    .. math::

        F_\nu = \frac{4\pi\cos\theta}{D_L^2}
                \int_{R_{\rm in}}^{R_{\rm out}}
                B_\nu\!\left(T_{\rm eff}(r)\right) r\,\mathrm{d}r

    The radial integral is evaluated on a log-spaced grid using the
    trapezoidal rule.

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
        :math:`F_\nu` [:math:`\mathrm{erg\,s^{-1}\,Hz^{-1}\,cm^{-2}}`],
        same shape as *nu*.
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
        :math:`L_\nu` [:math:`\mathrm{erg\,s^{-1}\,Hz^{-1}}`],
        same shape as *nu*.
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

    This is the luminosity an observer infers by assuming isotropic emission
    (:math:`L_{\nu,{\rm iso}} = 4\pi D_L^2 F_\nu`):

    .. math::

        L_{\nu,{\rm iso}} = 16\pi^2\cos\theta
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
        :math:`L_{\nu,{\rm iso}}` [:math:`\mathrm{erg\,s^{-1}\,Hz^{-1}}`],
        same shape as *nu*.
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


def disk_bolometric_luminosity(
    M_BH: u.Quantity,
    mdot: u.Quantity,
    R_in: u.Quantity,
) -> u.Quantity:
    r"""
    Bolometric luminosity of a steady-state thin disk.

    Analytic result obtained by integrating the viscous dissipation
    profile over both faces with a zero-torque inner boundary and
    :math:`R_{\rm out} \to \infty`:

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
        :math:`L_{\rm bol}` [:math:`\mathrm{erg\,s^{-1}}`].

    Notes
    -----
    This is equivalent to an efficiency :math:`\eta = R_{\rm S} / (4 R_{\rm in})`
    where :math:`R_{\rm S} = 2 G M / c^2` is the Schwarzschild radius.
    For a Schwarzschild ISCO (:math:`R_{\rm in} = 3 R_{\rm S}`),
    :math:`\eta \approx 0.083`.
    """
    log_M_BH = float(np.log(ensure_in_units(M_BH, "g")))
    log_M_dot = float(np.log(ensure_in_units(mdot, "g/s")))
    log_R_in = float(np.log(ensure_in_units(R_in, "cm")))

    result = np.exp(_log_disk_bolometric_luminosity(log_M_BH, log_M_dot, log_R_in))
    return float(result) * u.erg / u.s
