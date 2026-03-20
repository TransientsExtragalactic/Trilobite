r"""
Free-free optical depth calculations for structured plasma environments.

This module provides routines for computing the **free-free (bremsstrahlung)
optical depth** :math:`\tau_{\rm ff}` by integrating the absorption coefficient
:math:`\alpha_\nu` along a line of sight through an ionised plasma:

.. math::

    \tau_{\rm ff}(\nu) = \int_r^{r_{\rm max}} \alpha_\nu(r')\, dr'

Five density-profile models are supported, each with both an **exact** and a
**Rayleigh–Jeans** (RJ) variant:

* **Quadrature** (``_from_quadrature``): arbitrary :math:`n_e(r)`,
  :math:`n_i(r)`, :math:`T(r)` profiles supplied as Python callables;
  integration performed via :func:`scipy.integrate.quad`.
* **Arrays** (``_from_arrays``): density profile as discrete :math:`(r, \rho)`
  arrays; integration via the trapezoidal rule.
* **Wind** (``_wind``): :math:`\rho(r) = \dot{M} / (4\pi v_{\rm w} r^2)`;
  closed-form analytic integral :math:`\propto r^{-3}`.
* **Shell** (``_shell``): uniform density :math:`\rho_0`; depth
  :math:`r_{\rm max} - r`.
* **Power-law** (``_powerlaw``): :math:`\rho(r) = \rho_0\, r^{-p}`; analytic
  integral in log-space, handling both converging (:math:`p > 1/2`) and
  diverging (:math:`p < 1/2`) cases.

Module structure
----------------
* **Private backend** (``_compute_*``): all inputs and outputs are in **natural
  logarithm CGS form**.  No unit conversion is performed.  These functions are
  optimised for performance and numerical stability.
* **Public API** (``compute_*``): unit-aware wrappers that accept
  :class:`float` or :class:`~astropy.units.Quantity` inputs, coerce units via
  :func:`~triceratops.utils.misc_utils.ensure_in_units`, delegate to the
  backend, and return a plain :class:`~numpy.ndarray` of dimensionless optical
  depths (one value per frequency).

.. note::

    Quadrature-based functions accept the density and temperature profiles as
    **pure-CGS Python callables** (``n_e(r_cgs) -> float`` etc.).  No unit
    conversion is applied inside the integrand.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Union

import numpy as np
from astropy import constants as const
from astropy import units as u
from scipy.integrate import quad

from triceratops.utils.misc_utils import ensure_in_units

from .core import _log_ff_absorption, _log_ff_RJ_absorption

if TYPE_CHECKING:
    from triceratops._typing import _ArrayLike, _UnitBearingArrayLike

# NumPy compatibility: np.trapezoid added in 2.0, np.trapz deprecated in 2.0.
_trapz = getattr(np, "trapezoid", np.trapz)

# ============================================== #
# Module-level CGS constants                     #
# ============================================== #
_proton_mass_cgs: float = const.m_p.cgs.value
"""float: Proton mass :math:`m_p` in CGS units (g)."""


# ============================================== #
# Private API — log-space CGS backends           #
# ============================================== #


def _compute_ff_optical_depth_from_quadrature(
    log_nu: "_ArrayLike",
    r: float,
    *,
    n_e: "Callable[[float], float]",
    n_i: "Callable[[float], float]",
    temperature: "Callable[[float], float]",
    r_max: float = np.inf,
    Z: float = 1.0,
    g_ff: float = 5.0,
) -> np.ndarray:
    r"""Exact free-free optical depth via adaptive quadrature (CGS log-space).

    For each frequency :math:`\nu_k = e^{\log\nu_k}`, integrates the
    absorption coefficient

    .. math::

        \tau_{\rm ff}(\nu_k) = \int_r^{r_{\rm max}} \alpha_\nu(r')\, dr'

    numerically using :func:`scipy.integrate.quad`, with :math:`\alpha_\nu`
    evaluated by :func:`~.core._log_ff_absorption` at each quadrature point.

    Parameters
    ----------
    log_nu : array-like
        Natural logarithm of photon frequencies :math:`\nu` [Hz].
    r : float
        Inner integration radius [cm].
    n_e : callable
        Electron number density profile ``n_e(r) -> float`` [cm\ :sup:`-3`].
        The argument ``r`` is in cm (CGS); no unit conversion is applied.
    n_i : callable
        Ion number density profile ``n_i(r) -> float`` [cm\ :sup:`-3`].
    temperature : callable
        Electron temperature profile ``temperature(r) -> float`` [K].
    r_max : float, optional
        Outer integration radius [cm].  Default is ``numpy.inf`` (integrate
        to infinity).
    Z : float, optional
        Mean ionic charge (dimensionless).  Default ``1.0``.
    g_ff : float, optional
        Free–free Gaunt factor (dimensionless).  Default ``5.0``.

    Returns
    -------
    tau : ~numpy.ndarray
        Free-free optical depths, shape ``(len(log_nu),)``.

    See Also
    --------
    _compute_RJ_ff_optical_depth_from_quadrature : Rayleigh–Jeans variant.
    _compute_ff_optical_depth_from_arrays : Discrete-profile trapezoidal variant.
    """
    nu = np.exp(np.asarray(log_nu, dtype=float))
    tau = np.zeros_like(nu)

    for i, nu_i in enumerate(nu):

        def integrand(rp, _nu_i=nu_i):
            """Free-free absorption coefficient at radius ``rp`` [cm]."""
            log_alpha = _log_ff_absorption(
                log_nu=np.log(_nu_i),
                log_n_e=np.log(n_e(rp)),
                log_n_i=np.log(n_i(rp)),
                log_Z=np.log(Z),
                log_T=np.log(temperature(rp)),
                g_ff=g_ff,
            )
            return np.exp(log_alpha)

        tau[i], _ = quad(integrand, r, r_max)

    return tau


def _compute_RJ_ff_optical_depth_from_quadrature(
    log_nu: "_ArrayLike",
    r: float,
    *,
    n_e: "Callable[[float], float]",
    n_i: "Callable[[float], float]",
    temperature: "Callable[[float], float]",
    r_max: float = np.inf,
    Z: float = 1.0,
    g_ff: float = 5.0,
) -> np.ndarray:
    r"""Rayleigh–Jeans free-free optical depth via adaptive quadrature (CGS log-space).

    Identical to :func:`_compute_ff_optical_depth_from_quadrature` except that
    the integrand uses the RJ absorption coefficient
    :func:`~.core._log_ff_RJ_absorption`, which replaces
    :math:`(1 - e^{-h\nu/k_BT}) \to h\nu / k_BT`.  Valid when
    :math:`h\nu \ll k_B T`.

    Parameters
    ----------
    log_nu : array-like
        Natural logarithm of photon frequencies :math:`\nu` [Hz].
    r : float
        Inner integration radius [cm].
    n_e : callable
        Electron number density profile ``n_e(r) -> float`` [cm\ :sup:`-3`].
    n_i : callable
        Ion number density profile ``n_i(r) -> float`` [cm\ :sup:`-3`].
    temperature : callable
        Electron temperature profile ``temperature(r) -> float`` [K].
    r_max : float, optional
        Outer integration radius [cm].  Default ``numpy.inf``.
    Z : float, optional
        Mean ionic charge (dimensionless).  Default ``1.0``.
    g_ff : float, optional
        Free–free Gaunt factor (dimensionless).  Default ``5.0``.

    Returns
    -------
    tau : ~numpy.ndarray
        Rayleigh–Jeans free-free optical depths, shape ``(len(log_nu),)``.

    See Also
    --------
    _compute_ff_optical_depth_from_quadrature : Exact (full Planck) variant.
    """
    nu = np.exp(np.asarray(log_nu, dtype=float))
    tau = np.zeros_like(nu)

    for i, nu_i in enumerate(nu):

        def integrand(rp, _nu_i=nu_i):
            """Rayleigh–Jeans free-free absorption coefficient at radius ``rp`` [cm]."""
            log_alpha = _log_ff_RJ_absorption(
                log_nu=np.log(_nu_i),
                log_n_e=np.log(n_e(rp)),
                log_n_i=np.log(n_i(rp)),
                log_Z=np.log(Z),
                log_T=np.log(temperature(rp)),
                g_ff=g_ff,
            )
            return np.exp(log_alpha)

        tau[i], _ = quad(integrand, r, r_max)

    return tau


def _compute_ff_optical_depth_from_arrays(
    log_nu: "_ArrayLike",
    r: "_ArrayLike",
    rho: "_ArrayLike",
    *,
    temperature: float,
    mu_e: float = 1.2,
    mu_i: float = 1.3,
    Z: float = 1.0,
    g_ff: float = 5.0,
) -> np.ndarray:
    r"""Exact free-free optical depth via trapezoidal integration over discrete arrays (CGS log-space).

    Converts mass density :math:`\rho` to number densities using mean molecular
    weights, evaluates :math:`\alpha_\nu` at each grid point, and integrates
    along the radial grid using :func:`numpy.trapz`.

    .. math::

        n_e = \frac{\rho}{\mu_e m_p}, \qquad
        n_i = \frac{\rho}{\mu_i m_p}

    Parameters
    ----------
    log_nu : array-like
        Natural logarithm of photon frequencies [Hz], shape ``(n_nu,)``.
    r : array-like
        Radial grid [cm], shape ``(n_r,)``.  Must be monotonically increasing.
    rho : array-like
        Mass density at each radial grid point [g cm\ :sup:`-3`], shape ``(n_r,)``.
    temperature : float
        Uniform electron temperature [K].
    mu_e : float, optional
        Mean molecular weight per electron.  Default ``1.2``.
    mu_i : float, optional
        Mean molecular weight per ion.  Default ``1.3``.
    Z : float, optional
        Mean ionic charge (dimensionless).  Default ``1.0``.
    g_ff : float, optional
        Free–free Gaunt factor (dimensionless).  Default ``5.0``.

    Returns
    -------
    tau : ~numpy.ndarray
        Free-free optical depths, shape ``(n_nu,)``.

    See Also
    --------
    _compute_RJ_ff_optical_depth_from_arrays : Rayleigh–Jeans variant.
    _compute_ff_optical_depth_from_quadrature : Arbitrary-profile quadrature variant.
    """
    r = np.asarray(r, dtype=float)
    rho = np.asarray(rho, dtype=float)

    log_ne = np.log(rho) - np.log(_proton_mass_cgs) - np.log(mu_e)
    log_ni = np.log(rho) - np.log(_proton_mass_cgs) - np.log(mu_i)

    log_alpha = _log_ff_absorption(
        log_nu=np.asarray(log_nu, dtype=float)[..., None],
        log_n_e=log_ne,
        log_n_i=log_ni,
        log_Z=np.log(Z),
        log_T=np.log(temperature),
        g_ff=g_ff,
    )

    return _trapz(np.exp(log_alpha), r, axis=-1)


def _compute_RJ_ff_optical_depth_from_arrays(
    log_nu: "_ArrayLike",
    r: "_ArrayLike",
    rho: "_ArrayLike",
    *,
    temperature: float,
    mu_e: float = 1.2,
    mu_i: float = 1.3,
    Z: float = 1.0,
    g_ff: float = 5.0,
) -> np.ndarray:
    r"""Rayleigh–Jeans free-free optical depth via trapezoidal integration over discrete arrays (CGS log-space).

    Identical to :func:`_compute_ff_optical_depth_from_arrays` except that the
    Rayleigh–Jeans absorption coefficient :func:`~.core._log_ff_RJ_absorption`
    is used in place of the exact form.

    Parameters
    ----------
    log_nu : array-like
        Natural logarithm of photon frequencies [Hz], shape ``(n_nu,)``.
    r : array-like
        Radial grid [cm], shape ``(n_r,)``.
    rho : array-like
        Mass density [g cm\ :sup:`-3`], shape ``(n_r,)``.
    temperature : float
        Uniform electron temperature [K].
    mu_e : float, optional
        Mean molecular weight per electron.  Default ``1.2``.
    mu_i : float, optional
        Mean molecular weight per ion.  Default ``1.3``.
    Z : float, optional
        Mean ionic charge (dimensionless).  Default ``1.0``.
    g_ff : float, optional
        Free–free Gaunt factor (dimensionless).  Default ``5.0``.

    Returns
    -------
    tau : ~numpy.ndarray
        Rayleigh–Jeans free-free optical depths, shape ``(n_nu,)``.

    See Also
    --------
    _compute_ff_optical_depth_from_arrays : Exact (full Planck) variant.
    """
    r = np.asarray(r, dtype=float)
    rho = np.asarray(rho, dtype=float)

    log_ne = np.log(rho) - np.log(_proton_mass_cgs) - np.log(mu_e)
    log_ni = np.log(rho) - np.log(_proton_mass_cgs) - np.log(mu_i)

    log_alpha = _log_ff_RJ_absorption(
        log_nu=np.asarray(log_nu, dtype=float)[..., None],
        log_n_e=log_ne,
        log_n_i=log_ni,
        log_Z=np.log(Z),
        log_T=np.log(temperature),
        g_ff=g_ff,
    )

    return _trapz(np.exp(log_alpha), r, axis=-1)


def _compute_ff_optical_depth_wind(
    log_nu: "_ArrayLike",
    log_r: float,
    *,
    log_mdot: float,
    log_wind_velocity: float,
    log_r_max: float = np.inf,
    log_temperature: float = np.log(1.0e4),
    mu_e: float = 1.2,
    mu_i: float = 1.3,
    Z: float = 1.0,
    g_ff: float = 5.0,
) -> np.ndarray:
    r"""Exact free-free optical depth through a steady stellar wind (CGS log-space).

    For a spherically symmetric wind with :math:`\rho(r) = \dot{M}/(4\pi v_{\rm w} r^2)`:

    .. math::

        n_e(r) = \frac{\dot{M}}{4\pi v_{\rm w}\,\mu_e m_p}\,r^{-2},
        \qquad
        n_i(r) = \frac{\dot{M}}{4\pi v_{\rm w}\,\mu_i m_p}\,r^{-2}

    so :math:`\alpha_\nu \propto r^{-4}` and the integral evaluates analytically:

    .. math::

        \tau_{\rm ff} = \frac{\alpha_\nu^{(1)}}{3}
        \left(r^{-3} - r_{\rm max}^{-3}\right)

    where :math:`\alpha_\nu^{(1)}` is the absorption coefficient evaluated with
    the density coefficients at :math:`r = 1` cm.  All inputs are in **natural
    logarithm CGS** form.

    Parameters
    ----------
    log_nu : array-like
        Natural logarithm of photon frequencies [Hz].
    log_r : float
        Natural logarithm of the inner integration radius [cm].
    log_mdot : float
        Natural logarithm of the mass-loss rate [g s\ :sup:`-1`].
    log_wind_velocity : float
        Natural logarithm of the wind terminal velocity [cm s\ :sup:`-1`].
    log_r_max : float, optional
        Natural logarithm of the outer integration radius [cm].
        Default ``numpy.inf`` (integrate to infinity).
    log_temperature : float, optional
        Natural logarithm of the uniform electron temperature [K].
        Default ``log(10^4)``.
    mu_e : float, optional
        Mean molecular weight per electron.  Default ``1.2``.
    mu_i : float, optional
        Mean molecular weight per ion.  Default ``1.3``.
    Z : float, optional
        Mean ionic charge (dimensionless).  Default ``1.0``.
    g_ff : float, optional
        Free–free Gaunt factor (dimensionless).  Default ``5.0``.

    Returns
    -------
    tau : ~numpy.ndarray
        Exact free-free optical depths, shape ``(len(log_nu),)``.

    See Also
    --------
    _compute_RJ_ff_optical_depth_wind : Rayleigh–Jeans variant.
    _compute_ff_optical_depth_powerlaw : General power-law density profile.
    """
    # Density coefficient: n = (mdot / 4π v_w) / (mu m_p) at r = 1 cm.
    log_wind_coeff = log_mdot - np.log(4 * np.pi) - log_wind_velocity
    log_ne_coeff = log_wind_coeff - np.log(_proton_mass_cgs) - np.log(mu_e)
    log_ni_coeff = log_wind_coeff - np.log(_proton_mass_cgs) - np.log(mu_i)

    # Absorption coefficient at the density coefficient (i.e. α at r = 1 cm).
    log_alpha_coeff = _log_ff_absorption(
        log_nu=log_nu,
        log_n_e=log_ne_coeff,
        log_n_i=log_ni_coeff,
        log_Z=np.log(Z),
        log_T=log_temperature,
        g_ff=g_ff,
    )

    # Analytic integral ∫_r^{r_max} r'^{-4} dr' = (r^{-3} − r_max^{-3}) / 3.
    log_r_term = -3.0 * log_r
    log_rmax_term = -3.0 * log_r_max
    log_integral = np.log(1.0 / 3.0) + log_r_term + np.log1p(-np.exp(log_rmax_term - log_r_term))

    return np.exp(log_alpha_coeff + log_integral)


def _compute_RJ_ff_optical_depth_wind(
    log_nu: "_ArrayLike",
    log_r: float,
    *,
    log_mdot: float,
    log_wind_velocity: float,
    log_r_max: float = np.inf,
    log_temperature: float = np.log(1.0e4),
    mu_e: float = 1.2,
    mu_i: float = 1.3,
    Z: float = 1.0,
    g_ff: float = 5.0,
) -> np.ndarray:
    r"""Rayleigh–Jeans free-free optical depth through a steady stellar wind (CGS log-space).

    Identical to :func:`_compute_ff_optical_depth_wind` except that the RJ
    absorption coefficient :func:`~.core._log_ff_RJ_absorption` is used.
    Refer to that function for the full mathematical derivation.

    Parameters
    ----------
    log_nu : array-like
        Natural logarithm of photon frequencies [Hz].
    log_r : float
        Natural logarithm of the inner integration radius [cm].
    log_mdot : float
        Natural logarithm of the mass-loss rate [g s\ :sup:`-1`].
    log_wind_velocity : float
        Natural logarithm of the wind terminal velocity [cm s\ :sup:`-1`].
    log_r_max : float, optional
        Natural logarithm of the outer integration radius [cm].
        Default ``numpy.inf``.
    log_temperature : float, optional
        Natural logarithm of the uniform electron temperature [K].
        Default ``log(10^4)``.
    mu_e : float, optional
        Mean molecular weight per electron.  Default ``1.2``.
    mu_i : float, optional
        Mean molecular weight per ion.  Default ``1.3``.
    Z : float, optional
        Mean ionic charge (dimensionless).  Default ``1.0``.
    g_ff : float, optional
        Free–free Gaunt factor (dimensionless).  Default ``5.0``.

    Returns
    -------
    tau : ~numpy.ndarray
        Rayleigh–Jeans free-free optical depths, shape ``(len(log_nu),)``.

    See Also
    --------
    _compute_ff_optical_depth_wind : Exact (full Planck) variant.
    """
    log_wind_coeff = log_mdot - np.log(4 * np.pi) - log_wind_velocity
    log_ne_coeff = log_wind_coeff - np.log(_proton_mass_cgs) - np.log(mu_e)
    log_ni_coeff = log_wind_coeff - np.log(_proton_mass_cgs) - np.log(mu_i)

    log_alpha_coeff = _log_ff_RJ_absorption(
        log_nu=log_nu,
        log_n_e=log_ne_coeff,
        log_n_i=log_ni_coeff,
        log_Z=np.log(Z),
        log_T=log_temperature,
        g_ff=g_ff,
    )

    log_r_term = -3.0 * log_r
    log_rmax_term = -3.0 * log_r_max
    log_integral = np.log(1.0 / 3.0) + log_r_term + np.log1p(-np.exp(log_rmax_term - log_r_term))

    return np.exp(log_alpha_coeff + log_integral)


def _compute_ff_optical_depth_shell(
    log_nu: "_ArrayLike",
    log_r: float,
    *,
    log_rho: float,
    log_r_max: float = np.inf,
    log_temperature: float = np.log(1.0e4),
    mu_e: float = 1.2,
    mu_i: float = 1.3,
    Z: float = 1.0,
    g_ff: float = 5.0,
) -> np.ndarray:
    r"""Exact free-free optical depth through a uniform-density shell (CGS log-space).

    For constant :math:`\rho_0`, both :math:`n_e` and :math:`n_i` are uniform,
    making :math:`\alpha_\nu` constant along the line of sight.  The optical
    depth reduces to

    .. math::

        \tau_{\rm ff} = \alpha_\nu \,(r_{\rm max} - r)

    which is evaluated in log-space as

    .. math::

        \log\tau = \log\alpha_\nu + \log r_{\rm max}
                   + \log\!\bigl(1 - e^{\log r - \log r_{\rm max}}\bigr)

    Parameters
    ----------
    log_nu : array-like
        Natural logarithm of photon frequencies [Hz].
    log_r : float
        Natural logarithm of the inner shell radius [cm].
    log_rho : float
        Natural logarithm of the uniform mass density [g cm\ :sup:`-3`].
    log_r_max : float, optional
        Natural logarithm of the outer shell radius [cm].
        Default ``numpy.inf`` (semi-infinite slab).
    log_temperature : float, optional
        Natural logarithm of the electron temperature [K].
        Default ``log(10^4)``.
    mu_e : float, optional
        Mean molecular weight per electron.  Default ``1.2``.
    mu_i : float, optional
        Mean molecular weight per ion.  Default ``1.3``.
    Z : float, optional
        Mean ionic charge (dimensionless).  Default ``1.0``.
    g_ff : float, optional
        Free–free Gaunt factor (dimensionless).  Default ``5.0``.

    Returns
    -------
    tau : ~numpy.ndarray
        Exact free-free optical depths, shape ``(len(log_nu),)``.

    See Also
    --------
    _compute_RJ_ff_optical_depth_shell : Rayleigh–Jeans variant.
    _compute_ff_optical_depth_wind : Wind density-profile variant.
    """
    log_ne = log_rho - np.log(_proton_mass_cgs) - np.log(mu_e)
    log_ni = log_rho - np.log(_proton_mass_cgs) - np.log(mu_i)

    log_alpha = _log_ff_absorption(
        log_nu=log_nu,
        log_n_e=log_ne,
        log_n_i=log_ni,
        log_Z=np.log(Z),
        log_T=log_temperature,
        g_ff=g_ff,
    )

    # log(r_max - r) = log_r_max + log1p(-exp(log_r - log_r_max))
    log_length = log_r_max + np.log1p(-np.exp(log_r - log_r_max))

    return np.exp(log_alpha + log_length)


def _compute_RJ_ff_optical_depth_shell(
    log_nu: "_ArrayLike",
    log_r: float,
    *,
    log_rho: float,
    log_r_max: float = np.inf,
    log_temperature: float = np.log(1.0e4),
    mu_e: float = 1.2,
    mu_i: float = 1.3,
    Z: float = 1.0,
    g_ff: float = 5.0,
) -> np.ndarray:
    r"""Rayleigh–Jeans free-free optical depth through a uniform-density shell (CGS log-space).

    Identical to :func:`_compute_ff_optical_depth_shell` except that the RJ
    absorption coefficient :func:`~.core._log_ff_RJ_absorption` is used.

    Parameters
    ----------
    log_nu : array-like
        Natural logarithm of photon frequencies [Hz].
    log_r : float
        Natural logarithm of the inner shell radius [cm].
    log_rho : float
        Natural logarithm of the uniform mass density [g cm\ :sup:`-3`].
    log_r_max : float, optional
        Natural logarithm of the outer shell radius [cm].
        Default ``numpy.inf``.
    log_temperature : float, optional
        Natural logarithm of the electron temperature [K].
        Default ``log(10^4)``.
    mu_e : float, optional
        Mean molecular weight per electron.  Default ``1.2``.
    mu_i : float, optional
        Mean molecular weight per ion.  Default ``1.3``.
    Z : float, optional
        Mean ionic charge (dimensionless).  Default ``1.0``.
    g_ff : float, optional
        Free–free Gaunt factor (dimensionless).  Default ``5.0``.

    Returns
    -------
    tau : ~numpy.ndarray
        Rayleigh–Jeans free-free optical depths, shape ``(len(log_nu),)``.

    See Also
    --------
    _compute_ff_optical_depth_shell : Exact (full Planck) variant.
    """
    log_ne = log_rho - np.log(_proton_mass_cgs) - np.log(mu_e)
    log_ni = log_rho - np.log(_proton_mass_cgs) - np.log(mu_i)

    log_alpha = _log_ff_RJ_absorption(
        log_nu=log_nu,
        log_n_e=log_ne,
        log_n_i=log_ni,
        log_Z=np.log(Z),
        log_T=log_temperature,
        g_ff=g_ff,
    )

    log_length = log_r_max + np.log1p(-np.exp(log_r - log_r_max))

    return np.exp(log_alpha + log_length)


def _compute_ff_optical_depth_powerlaw(
    log_nu: "_ArrayLike",
    log_r: float,
    *,
    log_rho: float,
    p: float,
    log_r_max: float = np.inf,
    log_temperature: float = np.log(1.0e4),
    mu_e: float = 1.2,
    mu_i: float = 1.3,
    Z: float = 1.0,
    g_ff: float = 5.0,
) -> np.ndarray:
    r"""Exact free-free optical depth for a power-law density profile (CGS log-space).

    For :math:`\rho(r) = \rho_0\, r^{-p}` (with :math:`\rho_0` the density
    coefficient at :math:`r = 1` cm), the absorption coefficient scales as
    :math:`\alpha_\nu \propto r^{-2p}` and the optical depth integral is

    .. math::

        \tau_{\rm ff} = \alpha_\nu^{(1)} \int_r^{r_{\rm max}} r'^{-2p}\, dr'

    where :math:`\alpha_\nu^{(1)}` is evaluated at :math:`n_e = \rho_0/(\mu_e m_p)`.

    Setting :math:`k = 1 - 2p`:

    * :math:`k \neq 0`: :math:`\int_r^{r_{\rm max}} r'^{k-1}\, dr' = (r_{\rm max}^k - r^k)/k`.
    * :math:`k = 0` (:math:`p = 0.5`): :math:`\int_r^{r_{\rm max}} r'^{-1}\, dr' = \ln(r_{\rm max}/r)`.

    The integral is always positive because the numerator and denominator of
    :math:`(r_{\rm max}^k - r^k)/k` share the same sign.  Log-space evaluation
    handles both converging (:math:`k < 0`, :math:`p > 0.5`) and diverging
    (:math:`k > 0`, :math:`p < 0.5`) cases without cancellation errors.

    Parameters
    ----------
    log_nu : array-like
        Natural logarithm of photon frequencies [Hz].
    log_r : float
        Natural logarithm of the inner integration radius [cm].
    log_rho : float
        Natural logarithm of the density coefficient :math:`\rho_0`
        [g cm\ :sup:`-3`], i.e.\ the density at :math:`r = 1` cm.
    p : float
        Power-law index (:math:`\rho \propto r^{-p}`).  The integral converges
        at :math:`r_{\rm max} \to \infty` for :math:`p > 0.5` and diverges for
        :math:`p \leq 0.5`.
    log_r_max : float, optional
        Natural logarithm of the outer integration radius [cm].
        Default ``numpy.inf``.
    log_temperature : float, optional
        Natural logarithm of the uniform electron temperature [K].
        Default ``log(10^4)``.
    mu_e : float, optional
        Mean molecular weight per electron.  Default ``1.2``.
    mu_i : float, optional
        Mean molecular weight per ion.  Default ``1.3``.
    Z : float, optional
        Mean ionic charge (dimensionless).  Default ``1.0``.
    g_ff : float, optional
        Free–free Gaunt factor (dimensionless).  Default ``5.0``.

    Returns
    -------
    tau : ~numpy.ndarray
        Exact free-free optical depths, shape ``(len(log_nu),)``.

    See Also
    --------
    _compute_RJ_ff_optical_depth_powerlaw : Rayleigh–Jeans variant.
    _compute_ff_optical_depth_wind : Wind profile (:math:`p = 2` special case).
    """
    log_ne_coeff = log_rho - np.log(_proton_mass_cgs) - np.log(mu_e)
    log_ni_coeff = log_rho - np.log(_proton_mass_cgs) - np.log(mu_i)

    log_alpha_coeff = _log_ff_absorption(
        log_nu=log_nu,
        log_n_e=log_ne_coeff,
        log_n_i=log_ni_coeff,
        log_Z=np.log(Z),
        log_T=log_temperature,
        g_ff=g_ff,
    )

    k = 1.0 - 2.0 * p

    if np.isclose(k, 0.0):
        # p = 0.5 exactly: ∫_r^{r_max} r'^{-1} dr' = ln(r_max / r)
        log_integral = np.log(log_r_max - log_r)
    elif k > 0:
        # r_max^k > r^k; use r_max^k as the leading term.
        log_rmax_term = k * log_r_max
        log_integral = np.log(1.0 / k) + log_rmax_term + np.log1p(-np.exp(k * log_r - log_rmax_term))
    else:
        # k < 0: r_max^k < r^k; use r^k as the leading term so the argument
        # of log1p lies in (-1, 0) and is numerically well-defined.
        log_r_term = k * log_r
        log_integral = np.log(-1.0 / k) + log_r_term + np.log1p(-np.exp(k * log_r_max - log_r_term))

    return np.exp(log_alpha_coeff + log_integral)


def _compute_RJ_ff_optical_depth_powerlaw(
    log_nu: "_ArrayLike",
    log_r: float,
    *,
    log_rho: float,
    p: float,
    log_r_max: float = np.inf,
    log_temperature: float = np.log(1.0e4),
    mu_e: float = 1.2,
    mu_i: float = 1.3,
    Z: float = 1.0,
    g_ff: float = 5.0,
) -> np.ndarray:
    r"""Rayleigh–Jeans free-free optical depth for a power-law density profile (CGS log-space).

    Identical to :func:`_compute_ff_optical_depth_powerlaw` except that the RJ
    absorption coefficient :func:`~.core._log_ff_RJ_absorption` is used.
    See that function for the full mathematical derivation.

    Parameters
    ----------
    log_nu : array-like
        Natural logarithm of photon frequencies [Hz].
    log_r : float
        Natural logarithm of the inner integration radius [cm].
    log_rho : float
        Natural logarithm of the density coefficient :math:`\rho_0`
        [g cm\ :sup:`-3`] at :math:`r = 1` cm.
    p : float
        Power-law index (:math:`\rho \propto r^{-p}`).
    log_r_max : float, optional
        Natural logarithm of the outer integration radius [cm].
        Default ``numpy.inf``.
    log_temperature : float, optional
        Natural logarithm of the uniform electron temperature [K].
        Default ``log(10^4)``.
    mu_e : float, optional
        Mean molecular weight per electron.  Default ``1.2``.
    mu_i : float, optional
        Mean molecular weight per ion.  Default ``1.3``.
    Z : float, optional
        Mean ionic charge (dimensionless).  Default ``1.0``.
    g_ff : float, optional
        Free–free Gaunt factor (dimensionless).  Default ``5.0``.

    Returns
    -------
    tau : ~numpy.ndarray
        Rayleigh–Jeans free-free optical depths, shape ``(len(log_nu),)``.

    See Also
    --------
    _compute_ff_optical_depth_powerlaw : Exact (full Planck) variant.
    """
    log_ne_coeff = log_rho - np.log(_proton_mass_cgs) - np.log(mu_e)
    log_ni_coeff = log_rho - np.log(_proton_mass_cgs) - np.log(mu_i)

    log_alpha_coeff = _log_ff_RJ_absorption(
        log_nu=log_nu,
        log_n_e=log_ne_coeff,
        log_n_i=log_ni_coeff,
        log_Z=np.log(Z),
        log_T=log_temperature,
        g_ff=g_ff,
    )

    k = 1.0 - 2.0 * p

    if np.isclose(k, 0.0):
        log_integral = np.log(log_r_max - log_r)
    elif k > 0:
        log_rmax_term = k * log_r_max
        log_integral = np.log(1.0 / k) + log_rmax_term + np.log1p(-np.exp(k * log_r - log_rmax_term))
    else:
        log_r_term = k * log_r
        log_integral = np.log(-1.0 / k) + log_r_term + np.log1p(-np.exp(k * log_r_max - log_r_term))

    return np.exp(log_alpha_coeff + log_integral)


# ============================================== #
# Public API — unit-aware wrappers               #
# ============================================== #


def compute_ff_optical_depth_from_quadrature(
    frequency: "_UnitBearingArrayLike",
    r: "Union[float, u.Quantity]",
    *,
    n_e: "Callable[[float], float]",
    n_i: "Callable[[float], float]",
    temperature: "Callable[[float], float]",
    r_max: "Union[float, u.Quantity]" = np.inf,
    Z: float = 1.0,
    g_ff: float = 5.0,
) -> np.ndarray:
    r"""Exact free-free optical depth for an arbitrary density profile via quadrature.

    Evaluates

    .. math::

        \tau_{\rm ff}(\nu) = \int_r^{r_{\rm max}} \alpha_\nu(r')\, dr'

    by numerical quadrature (:func:`scipy.integrate.quad`) at each frequency.
    The density and temperature are arbitrary callables, making this function
    suitable for structured or numerically-specified media.

    Parameters
    ----------
    frequency : float or `~astropy.units.Quantity` or array-like
        Photon frequencies :math:`\nu`.  Bare floats are assumed to be in **Hz**.
    r : float or `~astropy.units.Quantity`
        Inner integration radius.  Bare floats are assumed to be in **cm**.
    n_e : callable
        Electron number density profile.  Must accept a single CGS radius
        argument ``r_cgs`` [cm] and return :math:`n_e` [cm\ :sup:`-3`] as a
        float.  **No unit conversion is applied inside the integrand.**
    n_i : callable
        Ion number density profile, same calling convention as ``n_e``.
    temperature : callable
        Electron temperature profile.  Must accept ``r_cgs`` [cm] and return
        :math:`T` [K] as a float.
    r_max : float or `~astropy.units.Quantity`, optional
        Outer integration radius.  Bare floats are assumed to be in **cm**.
        Default is ``numpy.inf`` (integrate to infinity).
    Z : float, optional
        Mean ionic charge (dimensionless).  Default ``1.0``.
    g_ff : float, optional
        Free–free Gaunt factor (dimensionless).  Default ``5.0``.

    Returns
    -------
    tau : ~numpy.ndarray
        Dimensionless free-free optical depths, one entry per input frequency.

    Notes
    -----
    The quadrature uses the exact stimulated-emission factor
    :math:`(1 - e^{-h\nu/k_B T})`, making this function valid at all
    frequencies.  For radio frequencies where :math:`h\nu \ll k_B T`,
    :func:`compute_ff_RJ_optical_depth_from_quadrature` may be used instead.

    Each frequency is integrated independently, so runtime scales as
    :math:`\mathcal{O}(n_\nu \times N_{\rm quad})` where :math:`N_{\rm quad}`
    is the number of quadrature evaluations per frequency.  For smooth profiles
    and modest :math:`n_\nu`, this is fast; for very large arrays of frequencies
    consider the array-based or analytic alternatives.

    Examples
    --------
    Optical depth through a uniform HII region (constant :math:`n_e`, :math:`T`):

    .. code-block:: python

        import numpy as np
        from astropy import units as u
        from triceratops.radiation.free_free.absorption import (
            compute_ff_optical_depth_from_quadrature,
        )

        nu_arr = np.geomspace(1e8, 1e11, 50) * u.Hz
        tau = compute_ff_optical_depth_from_quadrature(
            frequency=nu_arr,
            r=1e16 * u.cm,
            n_e=lambda r: 1e3,  # cm^-3, constant
            n_i=lambda r: 1e3,  # cm^-3, constant
            temperature=lambda r: 1e4,  # K, constant
            r_max=1e17 * u.cm,
        )

    See Also
    --------
    compute_ff_RJ_optical_depth_from_quadrature : Rayleigh–Jeans variant.
    compute_ff_optical_depth_from_arrays : Trapezoidal variant for discrete profiles.
    compute_ff_optical_depth_wind : Analytic wind-profile variant.
    """
    nu_cgs = np.atleast_1d(np.asarray(ensure_in_units(frequency, u.Hz), dtype=float))
    r_cgs = float(ensure_in_units(r, u.cm))
    r_max_cgs = float(ensure_in_units(r_max, u.cm))

    return _compute_ff_optical_depth_from_quadrature(
        log_nu=np.log(nu_cgs),
        r=r_cgs,
        n_e=n_e,
        n_i=n_i,
        temperature=temperature,
        r_max=r_max_cgs,
        Z=Z,
        g_ff=g_ff,
    )


def compute_ff_RJ_optical_depth_from_quadrature(
    frequency: "_UnitBearingArrayLike",
    r: "Union[float, u.Quantity]",
    *,
    n_e: "Callable[[float], float]",
    n_i: "Callable[[float], float]",
    temperature: "Callable[[float], float]",
    r_max: "Union[float, u.Quantity]" = np.inf,
    Z: float = 1.0,
    g_ff: float = 5.0,
) -> np.ndarray:
    r"""Rayleigh–Jeans free-free optical depth for an arbitrary density profile via quadrature.

    Identical to :func:`compute_ff_optical_depth_from_quadrature` except that
    the RJ approximation

    .. math::

        (1 - e^{-h\nu/k_B T}) \approx \frac{h\nu}{k_B T}

    is used, giving :math:`\alpha_\nu^{\rm RJ} \propto \nu^{-2} T^{-3/2}`.
    Valid when :math:`h\nu \ll k_B T`, i.e.\ radio through sub-mm frequencies
    in plasmas with :math:`T \gtrsim 10^4` K.

    Parameters
    ----------
    frequency : float or `~astropy.units.Quantity` or array-like
        Photon frequencies :math:`\nu`.  Bare floats are assumed to be in **Hz**.
    r : float or `~astropy.units.Quantity`
        Inner integration radius.  Bare floats are assumed to be in **cm**.
    n_e : callable
        Electron number density profile ``n_e(r_cgs) -> float`` [cm\ :sup:`-3`].
        Argument is in CGS; no unit conversion is applied.
    n_i : callable
        Ion number density profile ``n_i(r_cgs) -> float`` [cm\ :sup:`-3`].
    temperature : callable
        Temperature profile ``temperature(r_cgs) -> float`` [K].
    r_max : float or `~astropy.units.Quantity`, optional
        Outer integration radius.  Bare floats are assumed to be in **cm**.
        Default ``numpy.inf``.
    Z : float, optional
        Mean ionic charge (dimensionless).  Default ``1.0``.
    g_ff : float, optional
        Free–free Gaunt factor (dimensionless).  Default ``5.0``.

    Returns
    -------
    tau : ~numpy.ndarray
        Dimensionless Rayleigh–Jeans optical depths, one entry per frequency.

    Notes
    -----
    The RJ approximation breaks down at :math:`\nu \gtrsim k_B T / h \approx
    2 \times 10^{10} (T / 1\,\mathrm{K})` Hz; use
    :func:`compute_ff_optical_depth_from_quadrature` for higher frequencies.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        from astropy import units as u
        from triceratops.radiation.free_free.absorption import (
            compute_ff_RJ_optical_depth_from_quadrature,
        )

        nu_arr = np.geomspace(1e8, 1e10, 30) * u.Hz
        tau_rj = (
            compute_ff_RJ_optical_depth_from_quadrature(
                frequency=nu_arr,
                r=1e16 * u.cm,
                n_e=lambda r: 1e3,
                n_i=lambda r: 1e3,
                temperature=lambda r: 1e4,
                r_max=1e17 * u.cm,
            )
        )

    See Also
    --------
    compute_ff_optical_depth_from_quadrature : Exact (full Planck) variant.
    compute_ff_RJ_optical_depth_from_arrays : Trapezoidal RJ variant.
    """
    nu_cgs = np.atleast_1d(np.asarray(ensure_in_units(frequency, u.Hz), dtype=float))
    r_cgs = float(ensure_in_units(r, u.cm))
    r_max_cgs = float(ensure_in_units(r_max, u.cm))

    return _compute_RJ_ff_optical_depth_from_quadrature(
        log_nu=np.log(nu_cgs),
        r=r_cgs,
        n_e=n_e,
        n_i=n_i,
        temperature=temperature,
        r_max=r_max_cgs,
        Z=Z,
        g_ff=g_ff,
    )


def compute_ff_optical_depth_from_arrays(
    frequency: "_UnitBearingArrayLike",
    r: "_UnitBearingArrayLike",
    rho: "_UnitBearingArrayLike",
    *,
    temperature: "Union[float, u.Quantity]",
    mu_e: float = 1.2,
    mu_i: float = 1.3,
    Z: float = 1.0,
    g_ff: float = 5.0,
) -> np.ndarray:
    r"""Exact free-free optical depth for a discrete density profile via trapezoidal integration.

    Converts mass density :math:`\rho` to electron and ion number densities
    using the mean molecular weights :math:`\mu_e` and :math:`\mu_i`:

    .. math::

        n_e = \frac{\rho}{\mu_e\, m_p},
        \qquad
        n_i = \frac{\rho}{\mu_i\, m_p}

    evaluates :math:`\alpha_\nu` at every grid point, and integrates radially
    via the trapezoidal rule.

    Parameters
    ----------
    frequency : float or `~astropy.units.Quantity` or array-like
        Photon frequencies :math:`\nu`, shape ``(n_\nu,)``.
        Bare floats are assumed to be in **Hz**.
    r : array-like or `~astropy.units.Quantity`
        Radial grid, shape ``(n_r,)``.  Must be monotonically increasing.
        Bare floats are assumed to be in **cm**.
    rho : array-like or `~astropy.units.Quantity`
        Mass density at each radial grid point, shape ``(n_r,)``.
        Bare floats are assumed to be in **g cm**\ :sup:`-3`.
    temperature : float or `~astropy.units.Quantity`
        Uniform electron temperature.  Bare floats are assumed to be in **K**.
    mu_e : float, optional
        Mean molecular weight per electron.  Default ``1.2`` (cosmic abundances).
    mu_i : float, optional
        Mean molecular weight per ion.  Default ``1.3`` (cosmic abundances).
    Z : float, optional
        Mean ionic charge (dimensionless).  Default ``1.0``.
    g_ff : float, optional
        Free–free Gaunt factor (dimensionless).  Default ``5.0``.

    Returns
    -------
    tau : ~numpy.ndarray
        Dimensionless free-free optical depths, shape ``(n_\nu,)``.

    Notes
    -----
    The trapezoidal approximation introduces discretisation error proportional
    to the square of the radial grid spacing.  For exponential or power-law
    profiles, a logarithmically-spaced grid substantially reduces this error.

    The absorption coefficient array has shape ``(n_\nu, n_r)`` and integration
    is performed along the last axis, so the returned array has shape
    ``(n_\nu,)``.

    Examples
    --------
    Optical depth through a numerically-specified ejecta profile:

    .. code-block:: python

        import numpy as np
        from astropy import units as u
        from triceratops.radiation.free_free.absorption import (
            compute_ff_optical_depth_from_arrays,
        )

        r_arr = np.geomspace(1e15, 1e17, 200) * u.cm
        # Power-law density profile ρ ∝ r^{-2}
        rho_arr = (
            1e-18
            * (r_arr.value / 1e16) ** -2
            * u.g
            / u.cm**3
        )
        nu_arr = np.geomspace(1e8, 1e11, 50) * u.Hz

        tau = compute_ff_optical_depth_from_arrays(
            frequency=nu_arr,
            r=r_arr,
            rho=rho_arr,
            temperature=1e4 * u.K,
        )

    See Also
    --------
    compute_ff_RJ_optical_depth_from_arrays : Rayleigh–Jeans variant.
    compute_ff_optical_depth_from_quadrature : Quadrature variant for callable profiles.
    compute_ff_optical_depth_wind : Analytic wind-profile alternative.
    """
    nu_cgs = np.atleast_1d(np.asarray(ensure_in_units(frequency, u.Hz), dtype=float))
    r_cgs = np.asarray(ensure_in_units(r, u.cm), dtype=float)
    rho_cgs = np.asarray(ensure_in_units(rho, u.g / u.cm**3), dtype=float)
    T_cgs = float(ensure_in_units(temperature, u.K))

    return _compute_ff_optical_depth_from_arrays(
        log_nu=np.log(nu_cgs),
        r=r_cgs,
        rho=rho_cgs,
        temperature=T_cgs,
        mu_e=mu_e,
        mu_i=mu_i,
        Z=Z,
        g_ff=g_ff,
    )


def compute_ff_RJ_optical_depth_from_arrays(
    frequency: "_UnitBearingArrayLike",
    r: "_UnitBearingArrayLike",
    rho: "_UnitBearingArrayLike",
    *,
    temperature: "Union[float, u.Quantity]",
    mu_e: float = 1.2,
    mu_i: float = 1.3,
    Z: float = 1.0,
    g_ff: float = 5.0,
) -> np.ndarray:
    r"""Rayleigh–Jeans free-free optical depth for a discrete density profile via trapezoidal integration.

    Identical to :func:`compute_ff_optical_depth_from_arrays` except that the
    Rayleigh–Jeans absorption coefficient is used.  Valid when
    :math:`h\nu \ll k_B T`.

    Parameters
    ----------
    frequency : float or `~astropy.units.Quantity` or array-like
        Photon frequencies :math:`\nu`, shape ``(n_\nu,)``.
        Bare floats are assumed to be in **Hz**.
    r : array-like or `~astropy.units.Quantity`
        Radial grid, shape ``(n_r,)``.  Bare floats are assumed to be in **cm**.
    rho : array-like or `~astropy.units.Quantity`
        Mass density, shape ``(n_r,)``.
        Bare floats are assumed to be in **g cm**\ :sup:`-3`.
    temperature : float or `~astropy.units.Quantity`
        Uniform electron temperature.  Bare floats are assumed to be in **K**.
    mu_e : float, optional
        Mean molecular weight per electron.  Default ``1.2``.
    mu_i : float, optional
        Mean molecular weight per ion.  Default ``1.3``.
    Z : float, optional
        Mean ionic charge (dimensionless).  Default ``1.0``.
    g_ff : float, optional
        Free–free Gaunt factor (dimensionless).  Default ``5.0``.

    Returns
    -------
    tau : ~numpy.ndarray
        Dimensionless Rayleigh–Jeans optical depths, shape ``(n_\nu,)``.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        from astropy import units as u
        from triceratops.radiation.free_free.absorption import (
            compute_ff_RJ_optical_depth_from_arrays,
        )

        r_arr = np.geomspace(1e15, 1e17, 200) * u.cm
        rho_arr = (
            1e-18
            * (r_arr.value / 1e16) ** -2
            * u.g
            / u.cm**3
        )
        nu_arr = np.geomspace(1e8, 1e10, 30) * u.Hz

        tau = compute_ff_RJ_optical_depth_from_arrays(
            frequency=nu_arr,
            r=r_arr,
            rho=rho_arr,
            temperature=1e4 * u.K,
        )

    See Also
    --------
    compute_ff_optical_depth_from_arrays : Exact (full Planck) variant.
    compute_ff_RJ_optical_depth_from_quadrature : Quadrature RJ variant.
    """
    nu_cgs = np.atleast_1d(np.asarray(ensure_in_units(frequency, u.Hz), dtype=float))
    r_cgs = np.asarray(ensure_in_units(r, u.cm), dtype=float)
    rho_cgs = np.asarray(ensure_in_units(rho, u.g / u.cm**3), dtype=float)
    T_cgs = float(ensure_in_units(temperature, u.K))

    return _compute_RJ_ff_optical_depth_from_arrays(
        log_nu=np.log(nu_cgs),
        r=r_cgs,
        rho=rho_cgs,
        temperature=T_cgs,
        mu_e=mu_e,
        mu_i=mu_i,
        Z=Z,
        g_ff=g_ff,
    )


def compute_ff_optical_depth_wind(
    frequency: "_UnitBearingArrayLike",
    r: "Union[float, u.Quantity]",
    mdot: "Union[float, u.Quantity]",
    wind_velocity: "Union[float, u.Quantity]",
    *,
    r_max: "Union[float, u.Quantity]" = np.inf,
    temperature: "Union[float, u.Quantity]" = 1.0e4,
    mu_e: float = 1.2,
    mu_i: float = 1.3,
    Z: float = 1.0,
    g_ff: float = 5.0,
) -> np.ndarray:
    r"""Exact free-free optical depth through a steady spherically-symmetric stellar wind.

    For a steady wind with mass-loss rate :math:`\dot{M}` and terminal velocity
    :math:`v_{\rm w}`, mass continuity gives

    .. math::

        \rho(r) = \frac{\dot{M}}{4\pi\, v_{\rm w}\, r^2}

    and hence

    .. math::

        n_e(r) = \frac{\dot{M}}{4\pi\, v_{\rm w}\,\mu_e m_p}\, r^{-2},
        \qquad
        n_i(r) = \frac{\dot{M}}{4\pi\, v_{\rm w}\,\mu_i m_p}\, r^{-2}

    The absorption coefficient therefore scales as :math:`\alpha_\nu \propto r^{-4}`,
    and the optical depth integral evaluates analytically:

    .. math::

        \tau_{\rm ff}(\nu) =
        \frac{\alpha_\nu^{(1)}}{3}
        \left(r^{-3} - r_{\rm max}^{-3}\right)

    where :math:`\alpha_\nu^{(1)}` is the absorption coefficient evaluated with
    the density coefficients at :math:`r = 1` cm.

    Parameters
    ----------
    frequency : float or `~astropy.units.Quantity` or array-like
        Photon frequencies :math:`\nu`.  Bare floats are assumed to be in **Hz**.
    r : float or `~astropy.units.Quantity`
        Inner integration radius (typically the photospheric or shock radius).
        Bare floats are assumed to be in **cm**.
    mdot : float or `~astropy.units.Quantity`
        Wind mass-loss rate :math:`\dot{M}`.
        Bare floats are assumed to be in **g s**\ :sup:`-1`.
    wind_velocity : float or `~astropy.units.Quantity`
        Wind terminal velocity :math:`v_{\rm w}`.
        Bare floats are assumed to be in **cm s**\ :sup:`-1`.
    r_max : float or `~astropy.units.Quantity`, optional
        Outer integration radius.  Bare floats are assumed to be in **cm**.
        Default ``numpy.inf`` (integrate to infinity; the integral converges
        since :math:`\alpha_\nu \propto r^{-4}`).
    temperature : float or `~astropy.units.Quantity`, optional
        Uniform electron temperature.  Bare floats are assumed to be in **K**.
        Default ``1e4``.
    mu_e : float, optional
        Mean molecular weight per electron.  Default ``1.2``.
    mu_i : float, optional
        Mean molecular weight per ion.  Default ``1.3``.
    Z : float, optional
        Mean ionic charge (dimensionless).  Default ``1.0``.
    g_ff : float, optional
        Free–free Gaunt factor (dimensionless).  Default ``5.0``.

    Returns
    -------
    tau : ~numpy.ndarray
        Dimensionless exact free-free optical depths, one entry per frequency.

    Notes
    -----
    The **turnover frequency** :math:`\nu_{\rm ff}` at which
    :math:`\tau_{\rm ff} = 1` scales as

    .. math::

        \nu_{\rm ff} \propto
        \left(\frac{\dot{M}^2}{v_{\rm w}^2\, T^{1/2}\, r^3}\right)^{1/3}

    (in the RJ limit).  This is the characteristic free-free absorption
    frequency of radio supernovae and stellar wind shells.

    Examples
    --------
    Optical depth through a :math:`10^{-6}\,M_\odot\,\rm yr^{-1}` wind at 10 GHz:

    .. code-block:: python

        import numpy as np
        from astropy import units as u
        from triceratops.radiation.free_free.absorption import (
            compute_ff_optical_depth_wind,
        )

        nu_arr = np.geomspace(1e8, 1e11, 100) * u.Hz
        tau = compute_ff_optical_depth_wind(
            frequency=nu_arr,
            r=1e15 * u.cm,
            mdot=(1e-6 * u.Msun / u.yr).to(u.g / u.s),
            wind_velocity=1e8 * u.cm / u.s,
            temperature=1e4 * u.K,
        )

    See Also
    --------
    compute_ff_RJ_optical_depth_wind : Rayleigh–Jeans variant.
    compute_ff_optical_depth_powerlaw : General power-law density profile.
    compute_ff_optical_depth_from_quadrature : Arbitrary-profile variant.
    """
    nu_cgs = np.atleast_1d(np.asarray(ensure_in_units(frequency, u.Hz), dtype=float))
    r_cgs = float(ensure_in_units(r, u.cm))
    mdot_cgs = float(ensure_in_units(mdot, u.g / u.s))
    v_cgs = float(ensure_in_units(wind_velocity, u.cm / u.s))
    r_max_cgs = float(ensure_in_units(r_max, u.cm))
    T_cgs = float(ensure_in_units(temperature, u.K))

    return _compute_ff_optical_depth_wind(
        log_nu=np.log(nu_cgs),
        log_r=np.log(r_cgs),
        log_mdot=np.log(mdot_cgs),
        log_wind_velocity=np.log(v_cgs),
        log_r_max=np.log(r_max_cgs),
        log_temperature=np.log(T_cgs),
        mu_e=mu_e,
        mu_i=mu_i,
        Z=Z,
        g_ff=g_ff,
    )


def compute_ff_RJ_optical_depth_wind(
    frequency: "_UnitBearingArrayLike",
    r: "Union[float, u.Quantity]",
    mdot: "Union[float, u.Quantity]",
    wind_velocity: "Union[float, u.Quantity]",
    *,
    r_max: "Union[float, u.Quantity]" = np.inf,
    temperature: "Union[float, u.Quantity]" = 1.0e4,
    mu_e: float = 1.2,
    mu_i: float = 1.3,
    Z: float = 1.0,
    g_ff: float = 5.0,
) -> np.ndarray:
    r"""Rayleigh–Jeans free-free optical depth through a steady stellar wind.

    Identical to :func:`compute_ff_optical_depth_wind` except that the RJ
    approximation :math:`(1 - e^{-h\nu/k_B T}) \approx h\nu/k_B T` is used.
    The scaling becomes :math:`\tau_{\rm ff}^{\rm RJ} \propto \nu^{-2} T^{-3/2}`.

    Parameters
    ----------
    frequency : float or `~astropy.units.Quantity` or array-like
        Photon frequencies :math:`\nu`.  Bare floats are assumed to be in **Hz**.
    r : float or `~astropy.units.Quantity`
        Inner integration radius.  Bare floats are assumed to be in **cm**.
    mdot : float or `~astropy.units.Quantity`
        Wind mass-loss rate :math:`\dot{M}`.
        Bare floats are assumed to be in **g s**\ :sup:`-1`.
    wind_velocity : float or `~astropy.units.Quantity`
        Wind terminal velocity :math:`v_{\rm w}`.
        Bare floats are assumed to be in **cm s**\ :sup:`-1`.
    r_max : float or `~astropy.units.Quantity`, optional
        Outer integration radius.  Bare floats are assumed to be in **cm**.
        Default ``numpy.inf``.
    temperature : float or `~astropy.units.Quantity`, optional
        Uniform electron temperature.  Bare floats are assumed to be in **K**.
        Default ``1e4``.
    mu_e : float, optional
        Mean molecular weight per electron.  Default ``1.2``.
    mu_i : float, optional
        Mean molecular weight per ion.  Default ``1.3``.
    Z : float, optional
        Mean ionic charge (dimensionless).  Default ``1.0``.
    g_ff : float, optional
        Free–free Gaunt factor (dimensionless).  Default ``5.0``.

    Returns
    -------
    tau : ~numpy.ndarray
        Dimensionless Rayleigh–Jeans optical depths, one entry per frequency.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        from astropy import units as u
        from triceratops.radiation.free_free.absorption import (
            compute_ff_RJ_optical_depth_wind,
        )

        nu_arr = np.geomspace(1e8, 1e10, 50) * u.Hz
        tau_rj = compute_ff_RJ_optical_depth_wind(
            frequency=nu_arr,
            r=1e15 * u.cm,
            mdot=(1e-6 * u.Msun / u.yr).to(u.g / u.s),
            wind_velocity=1e8 * u.cm / u.s,
            temperature=1e4 * u.K,
        )

    See Also
    --------
    compute_ff_optical_depth_wind : Exact (full Planck) variant.
    compute_ff_RJ_optical_depth_powerlaw : General power-law RJ variant.
    """
    nu_cgs = np.atleast_1d(np.asarray(ensure_in_units(frequency, u.Hz), dtype=float))
    r_cgs = float(ensure_in_units(r, u.cm))
    mdot_cgs = float(ensure_in_units(mdot, u.g / u.s))
    v_cgs = float(ensure_in_units(wind_velocity, u.cm / u.s))
    r_max_cgs = float(ensure_in_units(r_max, u.cm))
    T_cgs = float(ensure_in_units(temperature, u.K))

    return _compute_RJ_ff_optical_depth_wind(
        log_nu=np.log(nu_cgs),
        log_r=np.log(r_cgs),
        log_mdot=np.log(mdot_cgs),
        log_wind_velocity=np.log(v_cgs),
        log_r_max=np.log(r_max_cgs),
        log_temperature=np.log(T_cgs),
        mu_e=mu_e,
        mu_i=mu_i,
        Z=Z,
        g_ff=g_ff,
    )


def compute_ff_optical_depth_shell(
    frequency: "_UnitBearingArrayLike",
    r: "Union[float, u.Quantity]",
    rho: "Union[float, u.Quantity]",
    *,
    r_max: "Union[float, u.Quantity]" = np.inf,
    temperature: "Union[float, u.Quantity]" = 1.0e4,
    mu_e: float = 1.2,
    mu_i: float = 1.3,
    Z: float = 1.0,
    g_ff: float = 5.0,
) -> np.ndarray:
    r"""Exact free-free optical depth through a uniform-density spherical shell.

    For a shell of constant density :math:`\rho_0`, the absorption coefficient
    :math:`\alpha_\nu` is uniform and the optical depth reduces to

    .. math::

        \tau_{\rm ff}(\nu) = \alpha_\nu \,(r_{\rm max} - r)

    This is the simplest free-free optical depth model and is appropriate for
    a cold, dense circumstellar shell or a homogeneous HII region.

    Parameters
    ----------
    frequency : float or `~astropy.units.Quantity` or array-like
        Photon frequencies :math:`\nu`.  Bare floats are assumed to be in **Hz**.
    r : float or `~astropy.units.Quantity`
        Inner shell radius.  Bare floats are assumed to be in **cm**.
    rho : float or `~astropy.units.Quantity`
        Uniform mass density :math:`\rho_0`.
        Bare floats are assumed to be in **g cm**\ :sup:`-3`.
    r_max : float or `~astropy.units.Quantity`, optional
        Outer shell radius.  Bare floats are assumed to be in **cm**.
        Default ``numpy.inf`` (semi-infinite slab — optical depth diverges).
    temperature : float or `~astropy.units.Quantity`, optional
        Uniform electron temperature.  Bare floats are assumed to be in **K**.
        Default ``1e4``.
    mu_e : float, optional
        Mean molecular weight per electron.  Default ``1.2``.
    mu_i : float, optional
        Mean molecular weight per ion.  Default ``1.3``.
    Z : float, optional
        Mean ionic charge (dimensionless).  Default ``1.0``.
    g_ff : float, optional
        Free–free Gaunt factor (dimensionless).  Default ``5.0``.

    Returns
    -------
    tau : ~numpy.ndarray
        Dimensionless exact free-free optical depths, one entry per frequency.

    Notes
    -----
    The optical depth scales as :math:`\tau_{\rm ff} \propto \nu^{-3}` at high
    frequencies (Wien limit) and :math:`\nu^{-2}` in the Rayleigh–Jeans limit.
    The **turnover frequency** at :math:`\tau = 1` is

    .. math::

        \nu_{\rm ff} \propto
        \left(n_e^2\, (r_{\rm max} - r)\, T^{-1/2}\right)^{1/3}

    Examples
    --------
    Optical depth through a dense supernova ejecta shell at day 10:

    .. code-block:: python

        import numpy as np
        from astropy import units as u
        from triceratops.radiation.free_free.absorption import (
            compute_ff_optical_depth_shell,
        )

        nu_arr = np.geomspace(1e8, 1e12, 100) * u.Hz
        tau = compute_ff_optical_depth_shell(
            frequency=nu_arr,
            r=1e15 * u.cm,
            rho=1e-19 * u.g / u.cm**3,
            r_max=2e15 * u.cm,
            temperature=1e4 * u.K,
        )

    See Also
    --------
    compute_ff_RJ_optical_depth_shell : Rayleigh–Jeans variant.
    compute_ff_optical_depth_wind : Wind (1/r²) density-profile variant.
    compute_ff_optical_depth_powerlaw : General power-law density-profile variant.
    """
    nu_cgs = np.atleast_1d(np.asarray(ensure_in_units(frequency, u.Hz), dtype=float))
    r_cgs = float(ensure_in_units(r, u.cm))
    rho_cgs = float(ensure_in_units(rho, u.g / u.cm**3))
    r_max_cgs = float(ensure_in_units(r_max, u.cm))
    T_cgs = float(ensure_in_units(temperature, u.K))

    return _compute_ff_optical_depth_shell(
        log_nu=np.log(nu_cgs),
        log_r=np.log(r_cgs),
        log_rho=np.log(rho_cgs),
        log_r_max=np.log(r_max_cgs),
        log_temperature=np.log(T_cgs),
        mu_e=mu_e,
        mu_i=mu_i,
        Z=Z,
        g_ff=g_ff,
    )


def compute_ff_RJ_optical_depth_shell(
    frequency: "_UnitBearingArrayLike",
    r: "Union[float, u.Quantity]",
    rho: "Union[float, u.Quantity]",
    *,
    r_max: "Union[float, u.Quantity]" = np.inf,
    temperature: "Union[float, u.Quantity]" = 1.0e4,
    mu_e: float = 1.2,
    mu_i: float = 1.3,
    Z: float = 1.0,
    g_ff: float = 5.0,
) -> np.ndarray:
    r"""Rayleigh–Jeans free-free optical depth through a uniform-density spherical shell.

    Identical to :func:`compute_ff_optical_depth_shell` except that the RJ
    absorption coefficient :math:`\alpha_\nu^{\rm RJ} \propto \nu^{-2} T^{-3/2}`
    is used in place of the exact form.

    Parameters
    ----------
    frequency : float or `~astropy.units.Quantity` or array-like
        Photon frequencies :math:`\nu`.  Bare floats are assumed to be in **Hz**.
    r : float or `~astropy.units.Quantity`
        Inner shell radius.  Bare floats are assumed to be in **cm**.
    rho : float or `~astropy.units.Quantity`
        Uniform mass density :math:`\rho_0`.
        Bare floats are assumed to be in **g cm**\ :sup:`-3`.
    r_max : float or `~astropy.units.Quantity`, optional
        Outer shell radius.  Bare floats are assumed to be in **cm**.
        Default ``numpy.inf``.
    temperature : float or `~astropy.units.Quantity`, optional
        Uniform electron temperature.  Bare floats are assumed to be in **K**.
        Default ``1e4``.
    mu_e : float, optional
        Mean molecular weight per electron.  Default ``1.2``.
    mu_i : float, optional
        Mean molecular weight per ion.  Default ``1.3``.
    Z : float, optional
        Mean ionic charge (dimensionless).  Default ``1.0``.
    g_ff : float, optional
        Free–free Gaunt factor (dimensionless).  Default ``5.0``.

    Returns
    -------
    tau : ~numpy.ndarray
        Dimensionless Rayleigh–Jeans optical depths, one entry per frequency.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        from astropy import units as u
        from triceratops.radiation.free_free.absorption import (
            compute_ff_RJ_optical_depth_shell,
        )

        nu_arr = np.geomspace(1e8, 1e10, 50) * u.Hz
        tau_rj = compute_ff_RJ_optical_depth_shell(
            frequency=nu_arr,
            r=1e15 * u.cm,
            rho=1e-19 * u.g / u.cm**3,
            r_max=2e15 * u.cm,
            temperature=1e4 * u.K,
        )

    See Also
    --------
    compute_ff_optical_depth_shell : Exact (full Planck) variant.
    compute_ff_RJ_optical_depth_wind : Wind-profile RJ variant.
    """
    nu_cgs = np.atleast_1d(np.asarray(ensure_in_units(frequency, u.Hz), dtype=float))
    r_cgs = float(ensure_in_units(r, u.cm))
    rho_cgs = float(ensure_in_units(rho, u.g / u.cm**3))
    r_max_cgs = float(ensure_in_units(r_max, u.cm))
    T_cgs = float(ensure_in_units(temperature, u.K))

    return _compute_RJ_ff_optical_depth_shell(
        log_nu=np.log(nu_cgs),
        log_r=np.log(r_cgs),
        log_rho=np.log(rho_cgs),
        log_r_max=np.log(r_max_cgs),
        log_temperature=np.log(T_cgs),
        mu_e=mu_e,
        mu_i=mu_i,
        Z=Z,
        g_ff=g_ff,
    )


def compute_ff_optical_depth_powerlaw(
    frequency: "_UnitBearingArrayLike",
    r: "Union[float, u.Quantity]",
    rho: "Union[float, u.Quantity]",
    p: float,
    *,
    r_max: "Union[float, u.Quantity]" = np.inf,
    temperature: "Union[float, u.Quantity]" = 1.0e4,
    mu_e: float = 1.2,
    mu_i: float = 1.3,
    Z: float = 1.0,
    g_ff: float = 5.0,
) -> np.ndarray:
    r"""Exact free-free optical depth for a power-law density profile.

    For :math:`\rho(r) = \rho_0\, r^{-p}` (with :math:`\rho_0` the density
    coefficient evaluated at :math:`r = 1` cm), the absorption coefficient scales
    as :math:`\alpha_\nu \propto r^{-2p}` and the optical depth evaluates analytically.

    Setting :math:`k = 1 - 2p`:

    .. math::

        \tau_{\rm ff}(\nu) = \alpha_\nu^{(1)}
        \begin{cases}
            \dfrac{r_{\rm max}^k - r^k}{k} & k \neq 0 \\[6pt]
            \ln\!\left(\dfrac{r_{\rm max}}{r}\right) & k = 0\;\;(p = 0.5)
        \end{cases}

    The result is always positive: for :math:`k > 0` (diverging integral,
    :math:`p < 0.5`) and :math:`k < 0` (converging integral, :math:`p > 0.5`)
    the numerator and denominator of :math:`(r_{\rm max}^k - r^k)/k` share the
    same sign.  Both cases are evaluated in log-space without cancellation.

    Parameters
    ----------
    frequency : float or `~astropy.units.Quantity` or array-like
        Photon frequencies :math:`\nu`.  Bare floats are assumed to be in **Hz**.
    r : float or `~astropy.units.Quantity`
        Inner integration radius.  Bare floats are assumed to be in **cm**.
    rho : float or `~astropy.units.Quantity`
        Density coefficient :math:`\rho_0` at :math:`r = 1` cm.
        Bare floats are assumed to be in **g cm**\ :sup:`-3`.
    p : float
        Power-law index (:math:`\rho \propto r^{-p}`).  Common values:

        * :math:`p = 2` — steady stellar wind (same as :func:`compute_ff_optical_depth_wind`).
        * :math:`p = 7/4` — Chevalier (1982) SN ejecta–wind interaction.
        * :math:`p = 0` — uniform density (same as :func:`compute_ff_optical_depth_shell`).
    r_max : float or `~astropy.units.Quantity`, optional
        Outer integration radius.  Bare floats are assumed to be in **cm**.
        Default ``numpy.inf``.  The integral converges at infinity for
        :math:`p > 0.5` and diverges for :math:`p \leq 0.5`.
    temperature : float or `~astropy.units.Quantity`, optional
        Uniform electron temperature.  Bare floats are assumed to be in **K**.
        Default ``1e4``.
    mu_e : float, optional
        Mean molecular weight per electron.  Default ``1.2``.
    mu_i : float, optional
        Mean molecular weight per ion.  Default ``1.3``.
    Z : float, optional
        Mean ionic charge (dimensionless).  Default ``1.0``.
    g_ff : float, optional
        Free–free Gaunt factor (dimensionless).  Default ``5.0``.

    Returns
    -------
    tau : ~numpy.ndarray
        Dimensionless exact free-free optical depths, one entry per frequency.

    Notes
    -----
    The wind profile (:math:`p = 2`) is a special case of this function; prefer
    :func:`compute_ff_optical_depth_wind` in that case for clearer intent, as
    it accepts :math:`\dot{M}` and :math:`v_{\rm w}` directly.

    Examples
    --------
    Optical depth for a Chevalier wind-interaction shell (:math:`p = 2`):

    .. code-block:: python

        import numpy as np
        from astropy import units as u
        from triceratops.radiation.free_free.absorption import (
            compute_ff_optical_depth_powerlaw,
        )

        nu_arr = np.geomspace(1e8, 1e11, 100) * u.Hz
        tau = compute_ff_optical_depth_powerlaw(
            frequency=nu_arr,
            r=1e15 * u.cm,
            rho=1e-15
            * u.g
            / u.cm**3,  # rho_0 at r = 1 cm
            p=2.0,
            temperature=1e4 * u.K,
        )

    Effect of the power-law index on the optical depth profile:

    .. code-block:: python

        import numpy as np
        from astropy import units as u
        from triceratops.radiation.free_free.absorption import (
            compute_ff_optical_depth_powerlaw,
        )

        nu = np.array([1e9]) * u.Hz
        for p in [0.5, 1.0, 1.5, 2.0]:
            tau = compute_ff_optical_depth_powerlaw(
                frequency=nu,
                r=1e15 * u.cm,
                rho=1e-15 * u.g / u.cm**3,
                p=p,
                temperature=1e4 * u.K,
            )
            print(f"p={p:.1f}  tau={tau[0]:.3e}")

    See Also
    --------
    compute_ff_RJ_optical_depth_powerlaw : Rayleigh–Jeans variant.
    compute_ff_optical_depth_wind : Optimised variant for the :math:`p = 2` wind profile.
    compute_ff_optical_depth_shell : Uniform-density (:math:`p = 0`) variant.
    """
    nu_cgs = np.atleast_1d(np.asarray(ensure_in_units(frequency, u.Hz), dtype=float))
    r_cgs = float(ensure_in_units(r, u.cm))
    rho_cgs = float(ensure_in_units(rho, u.g / u.cm**3))
    r_max_cgs = float(ensure_in_units(r_max, u.cm))
    T_cgs = float(ensure_in_units(temperature, u.K))

    return _compute_ff_optical_depth_powerlaw(
        log_nu=np.log(nu_cgs),
        log_r=np.log(r_cgs),
        log_rho=np.log(rho_cgs),
        p=p,
        log_r_max=np.log(r_max_cgs),
        log_temperature=np.log(T_cgs),
        mu_e=mu_e,
        mu_i=mu_i,
        Z=Z,
        g_ff=g_ff,
    )


def compute_ff_RJ_optical_depth_powerlaw(
    frequency: "_UnitBearingArrayLike",
    r: "Union[float, u.Quantity]",
    rho: "Union[float, u.Quantity]",
    p: float,
    *,
    r_max: "Union[float, u.Quantity]" = np.inf,
    temperature: "Union[float, u.Quantity]" = 1.0e4,
    mu_e: float = 1.2,
    mu_i: float = 1.3,
    Z: float = 1.0,
    g_ff: float = 5.0,
) -> np.ndarray:
    r"""Rayleigh–Jeans free-free optical depth for a power-law density profile.

    Identical to :func:`compute_ff_optical_depth_powerlaw` except that the RJ
    absorption coefficient :math:`\alpha_\nu^{\rm RJ} \propto \nu^{-2} T^{-3/2}`
    is used.  Valid when :math:`h\nu \ll k_B T`.

    Parameters
    ----------
    frequency : float or `~astropy.units.Quantity` or array-like
        Photon frequencies :math:`\nu`.  Bare floats are assumed to be in **Hz**.
    r : float or `~astropy.units.Quantity`
        Inner integration radius.  Bare floats are assumed to be in **cm**.
    rho : float or `~astropy.units.Quantity`
        Density coefficient :math:`\rho_0` at :math:`r = 1` cm.
        Bare floats are assumed to be in **g cm**\ :sup:`-3`.
    p : float
        Power-law index (:math:`\rho \propto r^{-p}`).
    r_max : float or `~astropy.units.Quantity`, optional
        Outer integration radius.  Bare floats are assumed to be in **cm**.
        Default ``numpy.inf``.
    temperature : float or `~astropy.units.Quantity`, optional
        Uniform electron temperature.  Bare floats are assumed to be in **K**.
        Default ``1e4``.
    mu_e : float, optional
        Mean molecular weight per electron.  Default ``1.2``.
    mu_i : float, optional
        Mean molecular weight per ion.  Default ``1.3``.
    Z : float, optional
        Mean ionic charge (dimensionless).  Default ``1.0``.
    g_ff : float, optional
        Free–free Gaunt factor (dimensionless).  Default ``5.0``.

    Returns
    -------
    tau : ~numpy.ndarray
        Dimensionless Rayleigh–Jeans optical depths, one entry per frequency.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        from astropy import units as u
        from triceratops.radiation.free_free.absorption import (
            compute_ff_RJ_optical_depth_powerlaw,
        )

        nu_arr = np.geomspace(1e8, 1e10, 50) * u.Hz
        tau_rj = compute_ff_RJ_optical_depth_powerlaw(
            frequency=nu_arr,
            r=1e15 * u.cm,
            rho=1e-15 * u.g / u.cm**3,
            p=2.0,
            temperature=1e4 * u.K,
        )

    See Also
    --------
    compute_ff_optical_depth_powerlaw : Exact (full Planck) variant.
    compute_ff_RJ_optical_depth_wind : Wind-profile RJ variant.
    """
    nu_cgs = np.atleast_1d(np.asarray(ensure_in_units(frequency, u.Hz), dtype=float))
    r_cgs = float(ensure_in_units(r, u.cm))
    rho_cgs = float(ensure_in_units(rho, u.g / u.cm**3))
    r_max_cgs = float(ensure_in_units(r_max, u.cm))
    T_cgs = float(ensure_in_units(temperature, u.K))

    return _compute_RJ_ff_optical_depth_powerlaw(
        log_nu=np.log(nu_cgs),
        log_r=np.log(r_cgs),
        log_rho=np.log(rho_cgs),
        p=p,
        log_r_max=np.log(r_max_cgs),
        log_temperature=np.log(T_cgs),
        mu_e=mu_e,
        mu_i=mu_i,
        Z=Z,
        g_ff=g_ff,
    )
