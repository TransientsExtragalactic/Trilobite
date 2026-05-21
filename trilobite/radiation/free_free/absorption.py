r"""
Free-free optical depth calculations for structured plasma environments.

This module provides routines for computing the **free-free (bremsstrahlung)
optical depth** :math:`\tau_{\rm ff}` in the Rayleigh–Jeans limit
(:math:`h\nu \ll k_B T`) by integrating

.. math::

    \tau_{\rm ff}(\nu) = \int_{r_{\rm min}}^{r_{\rm max}} \alpha_\nu^{\rm RJ}(r')\, dr',
    \qquad
    \alpha_\nu^{\rm RJ} \propto Z^2\, n_e\, n_i\, T^{-3/2}\, \nu^{-2}\, g_{\rm ff}

Five density-profile models are supported:

* **Quadrature** (``_from_quadrature``): arbitrary :math:`n_e(r)`, :math:`n_i(r)`,
  :math:`T(r)` profiles as Python callables.
* **Arrays** (``_from_arrays``): discrete :math:`n_e` and :math:`n_i` on a radial
  grid; trapezoidal integration.
* **Wind** (``_wind``): :math:`n_e, n_i \propto r^{-2}`; analytic integral.
* **Shell** (``_shell``): uniform :math:`n_e, n_i`; depth :math:`r_{\rm max} - r_{\rm min}`.
* **Power-law** (``_powerlaw``): :math:`n_e, n_i \propto r^{-p}`; analytic integral.

Interface
~~~~~~~~~
All functions share three Gaunt-factor / composition parameters:

``Z``
    Ionic charge.  Scalar for a single species; 1-D array when ``X`` is provided.
``g_ff``
    Free–free Gaunt factor.  ``None`` triggers auto-computation via the Lu+
    prescription.  Array of per-species values when ``X`` is provided.
``X``
    Number fractions (1-D array).  When ``X is not None``, ``Z`` and ``g_ff``
    must be matching arrays and the function uses the composition-weighted
    effective Gaunt factor

    .. math::

        g_{\rm ff,eff} = \sum_i Z_i^2\, x_i\, g_{{\rm ff},i}

Thomson scattering
~~~~~~~~~~~~~~~~~~
Set ``thomson=True`` to include electron-scattering opacity.  Returns the
effective optical depth

.. math::

    \tau_{\rm eff} = \sqrt{\tau_{\rm ff}\,(\tau_{\rm ff} + \tau_T)},
    \qquad
    \tau_T = \sigma_T \int_{r_{\rm min}}^{r_{\rm max}} n_e(r')\, dr'

Module structure
----------------
* **Private backend** (``_compute_RJ_*``): all inputs are in **natural logarithm
  CGS form** (except callables and flags).
* **Public API** (``compute_ff_RJ_*``): unit-aware wrappers.

.. note::

    Quadrature-based functions accept density and temperature profiles as
    **pure-CGS Python callables** (``n_e(r_cgs) -> float`` etc.).
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from astropy import units as u
from scipy.integrate import quad

from trilobite.utils.misc_utils import ensure_in_units

from .core import _log_ff_RJ_absorption
from .gaunt_factor import _resolve_gff

if TYPE_CHECKING:
    from trilobite._typing import _ArrayLike, _UnitBearingArrayLike

# NumPy compatibility: np.trapezoid added in 2.0, np.trapz removed in 2.0.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

_sigma_T_cgs: float = 6.6524e-25
r"""float: Thomson cross-section :math:`\sigma_T` [cm\ :sup:`2`]."""

_m_p_cgs: float = 1.67262192e-24
r"""float: Proton mass :math:`m_p` [g]."""


# ============================================== #
# Private helpers                                #
# ============================================== #
def _powerlaw_integral(k: float, log_r_min: float, log_r_max: float) -> float:
    r"""Return :math:`\ln\!\int_{r_{\rm min}}^{r_{\rm max}} r^{k-1}\,dr` in log-space.

    * :math:`k \neq 0`: :math:`(r_{\rm max}^k - r_{\rm min}^k)/k`.
    * :math:`k = 0`: :math:`\ln(r_{\rm max}/r_{\rm min})`.
    """
    if np.isclose(k, 0.0):
        return np.log(log_r_max - log_r_min)
    elif k > 0:
        log_rmax_k = k * log_r_max
        return np.log(1.0 / k) + log_rmax_k + np.log1p(-np.exp(k * log_r_min - log_rmax_k))
    else:
        log_rmin_k = k * log_r_min
        return np.log(-1.0 / k) + log_rmin_k + np.log1p(-np.exp(k * log_r_max - log_rmin_k))


# ============================================== #
# Private API — log-space CGS backends           #
# ============================================== #
def _compute_RJ_ff_optical_depth_from_quadrature(
    log_nu: "_ArrayLike",
    r: float,
    *,
    n_e: "Callable[[float], float]",
    n_i: "Callable[[float], float]",
    temperature: "Callable[[float], float]",
    r_max: float = np.inf,
    Z: "Union[float, _ArrayLike]" = 1.0,
    X: "Optional[_ArrayLike]" = None,
    g_ff: "Optional[Union[float, _ArrayLike]]" = None,
    thomson: bool = False,
) -> np.ndarray:
    r"""Free-free optical depth via adaptive quadrature in the RJ limit (CGS log-space).

    Gaunt factor is resolved from ``temperature(r)`` when ``g_ff`` is ``None``.

    Parameters
    ----------
    log_nu : array-like
        Natural logarithm of photon frequencies [Hz].
    r : float
        Inner integration radius [cm].
    n_e : callable
        Electron number density ``n_e(r) -> float`` [cm\ :sup:`-3`].
    n_i : callable
        Ion number density ``n_i(r) -> float`` [cm\ :sup:`-3`].
    temperature : callable
        Electron temperature ``temperature(r) -> float`` [K].
    r_max : float, optional
        Outer integration radius [cm].  Default ``numpy.inf``.
    Z : float or array-like, optional
        Ionic charge (scalar) or per-species charges (array, when ``X`` is set).
        Default ``1.0``.
    X : array-like or None, optional
        Number fractions.  ``None`` (default) → single-species mode.
    g_ff : float, array-like, or None, optional
        Gaunt factor(s).  Auto-computed if ``None``.
    thomson : bool, optional
        Return :math:`\tau_{\rm eff}` if ``True``.  Default ``False``.

    Returns
    -------
    tau : ~numpy.ndarray
        Optical depths, shape ``(len(log_nu),)``.
    """
    log_T_ref = np.log(temperature(r))
    nu = np.exp(np.asarray(log_nu, dtype=float))
    tau_ff = np.zeros_like(nu)

    for i, nu_i in enumerate(nu):
        Z_eff, g_ff_eff = _resolve_gff(np.log(nu_i), log_T_ref, Z, X, g_ff)

        def _integrand(rp, _nu_i=nu_i, _Z=Z_eff, _g=g_ff_eff):
            log_alpha = _log_ff_RJ_absorption(
                np.log(_nu_i),
                np.log(n_e(rp)),
                np.log(n_i(rp)),
                _Z,
                np.log(temperature(rp)),
                _g,
            )
            return np.exp(log_alpha)

        tau_ff[i], _ = quad(_integrand, r, r_max)

    if not thomson:
        return tau_ff
    tau_T, _ = quad(n_e, r, r_max)
    tau_T *= _sigma_T_cgs
    return np.sqrt(tau_ff * (tau_ff + tau_T))


def _compute_RJ_ff_optical_depth_from_arrays(
    log_nu: "_ArrayLike",
    r: "_ArrayLike",
    log_n_e: "_ArrayLike",
    log_n_i: "_ArrayLike",
    *,
    log_T: float,
    Z: "Union[float, _ArrayLike]" = 1.0,
    X: "Optional[_ArrayLike]" = None,
    g_ff: "Optional[Union[float, _ArrayLike]]" = None,
    thomson: bool = False,
) -> np.ndarray:
    r"""Free-free optical depth via trapezoidal integration in the RJ limit (CGS log-space).

    Parameters
    ----------
    log_nu : array-like
        Natural logarithm of photon frequencies [Hz], shape ``(n_nu,)``.
    r : array-like
        Radial grid [cm], shape ``(n_r,)``.
    log_n_e : array-like
        Natural log of electron number density [cm\ :sup:`-3`], shape ``(n_r,)``.
    log_n_i : array-like
        Natural log of ion number density [cm\ :sup:`-3`], shape ``(n_r,)``.
    log_T : float
        Natural log of the uniform electron temperature [K].
    Z : float or array-like, optional
        Ionic charge(s).  Default ``1.0``.
    X : array-like or None, optional
        Number fractions.  ``None`` → single-species mode.
    g_ff : float, array-like, or None, optional
        Gaunt factor(s).  Auto-computed if ``None``.
    thomson : bool, optional
        Return :math:`\tau_{\rm eff}` if ``True``.  Default ``False``.

    Returns
    -------
    tau : ~numpy.ndarray
        Optical depths, shape ``(n_nu,)``.
    """
    log_nu_ref = float(np.asarray(log_nu, dtype=float).mean())
    Z_eff, g_ff_eff = _resolve_gff(log_nu_ref, log_T, Z, X, g_ff)
    r = np.asarray(r, dtype=float)
    log_n_e = np.asarray(log_n_e, dtype=float)
    log_n_i = np.asarray(log_n_i, dtype=float)

    log_alpha = _log_ff_RJ_absorption(
        np.asarray(log_nu, dtype=float)[..., None],
        log_n_e,
        log_n_i,
        Z_eff,
        log_T,
        g_ff_eff,
    )
    tau_ff = _trapz(np.exp(log_alpha), r, axis=-1)

    if not thomson:
        return tau_ff
    tau_T = _sigma_T_cgs * _trapz(np.exp(log_n_e), r)
    return np.sqrt(tau_ff * (tau_ff + tau_T))


def _compute_RJ_ff_optical_depth_wind(
    log_nu: "_ArrayLike",
    log_r_min: float,
    log_r_0: float,
    log_n_e_0: float,
    log_n_i_0: float,
    *,
    log_r_max: float = np.inf,
    log_T: float = np.log(1.0e4),
    Z: "Union[float, _ArrayLike]" = 1.0,
    X: "Optional[_ArrayLike]" = None,
    g_ff: "Optional[Union[float, _ArrayLike]]" = None,
    thomson: bool = False,
) -> np.ndarray:
    r"""Free-free optical depth through a steady wind in the RJ limit (CGS log-space).

    Profile: :math:`n_e(r) = n_{e,0}\,(r_0/r)^2`.

    Parameters
    ----------
    log_nu : array-like
        Natural log of photon frequencies [Hz].
    log_r_min : float
        Natural log of the inner radius [cm].
    log_r_0 : float
        Natural log of the reference radius :math:`r_0` [cm].
    log_n_e_0 : float
        Natural log of electron density at :math:`r_0` [cm\ :sup:`-3`].
    log_n_i_0 : float
        Natural log of ion density at :math:`r_0` [cm\ :sup:`-3`].
    log_r_max : float, optional
        Natural log of the outer radius [cm].  Default ``numpy.inf``.
    log_T : float, optional
        Natural log of electron temperature [K].  Default ``log(10^4)``.
    Z, X, g_ff, thomson
        See :func:`_compute_RJ_ff_optical_depth_from_arrays`.

    Returns
    -------
    tau : ~numpy.ndarray
        Optical depths, shape ``(len(log_nu),)``.
    """
    log_nu_ref = float(np.asarray(log_nu, dtype=float).mean())
    Z_eff, g_ff_eff = _resolve_gff(log_nu_ref, log_T, Z, X, g_ff)

    log_ne_coeff = log_n_e_0 + 2.0 * log_r_0
    log_ni_coeff = log_n_i_0 + 2.0 * log_r_0

    log_alpha_coeff = _log_ff_RJ_absorption(
        log_nu,
        log_ne_coeff,
        log_ni_coeff,
        Z_eff,
        log_T,
        g_ff_eff,
    )
    tau_ff = np.exp(log_alpha_coeff + _powerlaw_integral(-3.0, log_r_min, log_r_max))

    if not thomson:
        return tau_ff
    tau_T = np.exp(np.log(_sigma_T_cgs) + log_ne_coeff + _powerlaw_integral(-1.0, log_r_min, log_r_max))
    return np.sqrt(tau_ff * (tau_ff + tau_T))


def _compute_RJ_ff_optical_depth_shell(
    log_nu: "_ArrayLike",
    log_r_min: float,
    log_n_e_0: float,
    log_n_i_0: float,
    *,
    log_r_max: float = np.inf,
    log_T: float = np.log(1.0e4),
    Z: "Union[float, _ArrayLike]" = 1.0,
    X: "Optional[_ArrayLike]" = None,
    g_ff: "Optional[Union[float, _ArrayLike]]" = None,
    thomson: bool = False,
) -> np.ndarray:
    r"""Free-free optical depth through a uniform shell in the RJ limit (CGS log-space).

    Profile: :math:`n_e(r) = n_{e,0}` (constant).

    Parameters
    ----------
    log_nu : array-like
        Natural log of photon frequencies [Hz].
    log_r_min : float
        Natural log of the inner shell radius [cm].
    log_n_e_0 : float
        Natural log of the uniform electron density [cm\ :sup:`-3`].
    log_n_i_0 : float
        Natural log of the uniform ion density [cm\ :sup:`-3`].
    log_r_max : float, optional
        Natural log of the outer shell radius [cm].
        Default ``numpy.inf`` (semi-infinite slab).
    log_T : float, optional
        Natural log of the electron temperature [K].  Default ``log(10^4)``.
    Z, X, g_ff, thomson
        See :func:`_compute_RJ_ff_optical_depth_from_arrays`.

    Returns
    -------
    tau : ~numpy.ndarray
        Optical depths, shape ``(len(log_nu),)``.
    """
    log_nu_ref = float(np.asarray(log_nu, dtype=float).mean())
    Z_eff, g_ff_eff = _resolve_gff(log_nu_ref, log_T, Z, X, g_ff)

    log_alpha = _log_ff_RJ_absorption(
        log_nu,
        log_n_e_0,
        log_n_i_0,
        Z_eff,
        log_T,
        g_ff_eff,
    )
    log_length = log_r_max + np.log1p(-np.exp(log_r_min - log_r_max))
    tau_ff = np.exp(log_alpha + log_length)

    if not thomson:
        return tau_ff
    tau_T = np.exp(np.log(_sigma_T_cgs) + log_n_e_0 + log_length)
    return np.sqrt(tau_ff * (tau_ff + tau_T))


def _compute_RJ_ff_optical_depth_powerlaw(
    log_nu: "_ArrayLike",
    log_r_min: float,
    log_r_0: float,
    log_n_e_0: float,
    log_n_i_0: float,
    p: float,
    *,
    log_r_max: float = np.inf,
    log_T: float = np.log(1.0e4),
    Z: "Union[float, _ArrayLike]" = 1.0,
    X: "Optional[_ArrayLike]" = None,
    g_ff: "Optional[Union[float, _ArrayLike]]" = None,
    thomson: bool = False,
) -> np.ndarray:
    r"""Free-free optical depth for a power-law density profile in the RJ limit (CGS log-space).

    Profile: :math:`n_e(r) = n_{e,0}\,(r_0/r)^p`.  Setting :math:`k = 1 - 2p`:

    .. math::

        \tau_{\rm ff}^{\rm RJ} = \alpha_\nu^{\rm RJ,(1)}
        \int_{r_{\rm min}}^{r_{\rm max}} r^{-2p}\, dr

    Parameters
    ----------
    log_nu : array-like
        Natural log of photon frequencies [Hz].
    log_r_min : float
        Natural log of the inner integration radius [cm].
    log_r_0 : float
        Natural log of the reference radius :math:`r_0` [cm].
    log_n_e_0 : float
        Natural log of electron density at :math:`r_0` [cm\ :sup:`-3`].
    log_n_i_0 : float
        Natural log of ion density at :math:`r_0` [cm\ :sup:`-3`].
    p : float
        Power-law index (:math:`n \propto r^{-p}`).
    log_r_max : float, optional
        Natural log of the outer radius [cm].  Default ``numpy.inf``.
    log_T : float, optional
        Natural log of the electron temperature [K].  Default ``log(10^4)``.
    Z, X, g_ff, thomson
        See :func:`_compute_RJ_ff_optical_depth_from_arrays`.

    Returns
    -------
    tau : ~numpy.ndarray
        Optical depths, shape ``(len(log_nu),)``.
    """
    log_nu_ref = float(np.asarray(log_nu, dtype=float).mean())
    Z_eff, g_ff_eff = _resolve_gff(log_nu_ref, log_T, Z, X, g_ff)

    log_ne_coeff = log_n_e_0 + p * log_r_0
    log_ni_coeff = log_n_i_0 + p * log_r_0

    log_alpha_coeff = _log_ff_RJ_absorption(
        log_nu,
        log_ne_coeff,
        log_ni_coeff,
        Z_eff,
        log_T,
        g_ff_eff,
    )
    tau_ff = np.exp(log_alpha_coeff + _powerlaw_integral(1.0 - 2.0 * p, log_r_min, log_r_max))

    if not thomson:
        return tau_ff
    tau_T = np.exp(np.log(_sigma_T_cgs) + log_ne_coeff + _powerlaw_integral(1.0 - p, log_r_min, log_r_max))
    return np.sqrt(tau_ff * (tau_ff + tau_T))


def _compute_RJ_ff_optical_depth_EM(
    log_nu: "_ArrayLike",
    log_EM: float,
    *,
    log_T: float = np.log(1.0e4),
    Z: "Union[float, _ArrayLike]" = 1.0,
    X: "Optional[_ArrayLike]" = None,
    g_ff: "Optional[Union[float, _ArrayLike]]" = None,
) -> np.ndarray:
    r"""Free-free optical depth from emission measure in the RJ limit (CGS log-space).

    Evaluates

    .. math::

        \tau_{\rm ff}^{\rm RJ}(\nu) =
        C_\alpha\, Z^2\, {\rm EM}\, T^{-3/2}\, \nu^{-2}\, g_{\rm ff}

    where :math:`{\rm EM} = \int n_e\, n_i\, dr` [cm\ :sup:`-5`].  This is a
    direct rearrangement of :func:`~.core._log_ff_RJ_absorption` when the
    line-of-sight integral is already known.

    Parameters
    ----------
    log_nu : array-like
        Natural log of photon frequencies [Hz].
    log_EM : float
        Natural log of the emission measure
        :math:`{\rm EM} = \int n_e\, n_i\, dr` [cm\ :sup:`-5`].
    log_T : float, optional
        Natural log of the electron temperature [K].  Default ``log(10^4)``.
    Z : float or array-like, optional
        Ionic charge(s).  Default ``1.0``.
    X : array-like or None, optional
        Number fractions.  ``None`` → single-species mode.
    g_ff : float, array-like, or None, optional
        Gaunt factor(s).  Auto-computed via Lu+ if ``None``.

    Returns
    -------
    tau : ~numpy.ndarray
        Optical depths, shape ``(n_nu,)``.

    See Also
    --------
    _compute_RJ_ff_optical_depth_from_arrays : Integrates over a discrete profile.
    """
    log_nu_arr = np.asarray(log_nu, dtype=float)
    log_nu_ref = float(log_nu_arr.mean())
    Z_eff, g_ff_eff = _resolve_gff(log_nu_ref, log_T, Z, X, g_ff)

    # Pass log_EM as log_n_e and 0.0 as log_n_i so that n_e * n_i → EM * 1 = EM.
    log_tau = _log_ff_RJ_absorption(log_nu_arr, float(log_EM), 0.0, Z_eff, log_T, g_ff_eff)
    return np.exp(log_tau)


# ============================================== #
# Public API — unit-aware wrappers               #
# ============================================== #
def compute_ff_RJ_optical_depth_from_quadrature(
    frequency: "_UnitBearingArrayLike",
    r: "Union[float, u.Quantity]",
    *,
    n_e: "Callable[[float], float]",
    n_i: "Callable[[float], float]",
    temperature: "Callable[[float], float]",
    r_max: "Union[float, u.Quantity]" = np.inf,
    Z: "Union[float, _ArrayLike]" = 1.0,
    X: "Optional[_ArrayLike]" = None,
    g_ff: "Optional[Union[float, _ArrayLike]]" = None,
    thomson: bool = False,
) -> np.ndarray:
    r"""Free-free optical depth for an arbitrary density profile via quadrature (RJ limit).

    Evaluates :math:`\tau_{\rm ff}^{\rm RJ}(\nu) = \int_r^{r_{\rm max}} \alpha_\nu^{\rm RJ}(r')\, dr'`
    by numerical quadrature at each frequency.  Valid when :math:`h\nu \ll k_B T`.

    Parameters
    ----------
    frequency : float or `~astropy.units.Quantity` or array-like
        Photon frequencies :math:`\nu`.  Bare floats assumed to be in **Hz**.
    r : float or `~astropy.units.Quantity`
        Inner integration radius.  Bare floats assumed to be in **cm**.
    n_e : callable
        Electron number density ``n_e(r_cgs) -> float`` [cm\ :sup:`-3`].
        **No unit conversion inside the integrand.**
    n_i : callable
        Ion number density ``n_i(r_cgs) -> float`` [cm\ :sup:`-3`].
    temperature : callable
        Electron temperature ``temperature(r_cgs) -> float`` [K].
    r_max : float or `~astropy.units.Quantity`, optional
        Outer integration radius.  Bare floats assumed to be in **cm**.
        Default ``numpy.inf``.
    Z : float or array-like, optional
        Ionic charge (scalar) or per-species array when ``X`` is set.
        Default ``1.0``.
    X : array-like or None, optional
        Number fractions.  ``None`` (default) → single-species mode.
    g_ff : float, array-like, or None, optional
        Gaunt factor(s).  Auto-computed via Draine formula if ``None``.
    thomson : bool, optional
        If ``True``, return :math:`\tau_{\rm eff} = \sqrt{\tau_{\rm ff}(\tau_{\rm ff}+\tau_T)}`.
        Default ``False``.

    Returns
    -------
    tau : ~numpy.ndarray
        Optical depths (or :math:`\tau_{\rm eff}`), one entry per frequency.

    Notes
    -----
    Runtime scales as :math:`\mathcal{O}(n_\nu \times N_{\rm quad})`.
    For analytic density profiles prefer the wind/shell/powerlaw variants.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        from astropy import units as u
        from trilobite.radiation.free_free.absorption import (
            compute_ff_RJ_optical_depth_from_quadrature,
        )

        nu_arr = np.geomspace(1e8, 1e10, 30) * u.Hz
        tau = compute_ff_RJ_optical_depth_from_quadrature(
            frequency=nu_arr,
            r=1e16 * u.cm,
            n_e=lambda r: 1e3,
            n_i=lambda r: 1e3,
            temperature=lambda r: 1e4,
            r_max=1e17 * u.cm,
        )

    See Also
    --------
    compute_ff_RJ_optical_depth_from_arrays : Trapezoidal variant.
    compute_ff_RJ_optical_depth_wind : Analytic wind-profile variant.
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
        X=X,
        g_ff=g_ff,
        thomson=thomson,
    )


def compute_ff_RJ_optical_depth_from_arrays(
    frequency: "_UnitBearingArrayLike",
    r: "_UnitBearingArrayLike",
    n_e: "_UnitBearingArrayLike",
    n_i: "_UnitBearingArrayLike",
    *,
    temperature: "Union[float, u.Quantity]",
    Z: "Union[float, _ArrayLike]" = 1.0,
    X: "Optional[_ArrayLike]" = None,
    g_ff: "Optional[Union[float, _ArrayLike]]" = None,
    thomson: bool = False,
) -> np.ndarray:
    r"""Free-free optical depth for discrete density arrays via trapezoidal integration (RJ limit).

    Parameters
    ----------
    frequency : float or `~astropy.units.Quantity` or array-like
        Photon frequencies :math:`\nu`, shape ``(n_\nu,)``.
        Bare floats assumed to be in **Hz**.
    r : array-like or `~astropy.units.Quantity`
        Radial grid, shape ``(n_r,)``.  Must be monotonically increasing.
        Bare floats assumed to be in **cm**.
    n_e : array-like or `~astropy.units.Quantity`
        Electron number density, shape ``(n_r,)``.
        Bare floats assumed to be in **cm**\ :sup:`-3`.
    n_i : array-like or `~astropy.units.Quantity`
        Ion number density, shape ``(n_r,)``.
        Bare floats assumed to be in **cm**\ :sup:`-3`.
    temperature : float or `~astropy.units.Quantity`
        Uniform electron temperature.  Bare floats assumed to be in **K**.
    Z : float or array-like, optional
        Ionic charge(s).  Default ``1.0``.
    X : array-like or None, optional
        Number fractions.  ``None`` → single-species mode.
    g_ff : float, array-like, or None, optional
        Gaunt factor(s).  Auto-computed if ``None``.
    thomson : bool, optional
        Return :math:`\tau_{\rm eff}` if ``True``.  Default ``False``.

    Returns
    -------
    tau : ~numpy.ndarray
        Optical depths, shape ``(n_\nu,)``.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        from astropy import units as u
        from trilobite.radiation.free_free.absorption import (
            compute_ff_RJ_optical_depth_from_arrays,
        )

        r_arr = np.geomspace(1e15, 1e17, 200) * u.cm
        n_e_arr = (
            1e3
            * (r_arr.to_value(u.cm) / 1e16) ** -2
            / u.cm**3
        )
        nu_arr = np.geomspace(1e8, 1e10, 30) * u.Hz

        tau = compute_ff_RJ_optical_depth_from_arrays(
            frequency=nu_arr,
            r=r_arr,
            n_e=n_e_arr,
            n_i=n_e_arr,
            temperature=1e4 * u.K,
        )

    See Also
    --------
    compute_ff_RJ_optical_depth_from_quadrature : Quadrature variant.
    compute_ff_RJ_optical_depth_wind : Analytic alternative.
    """
    nu_cgs = np.atleast_1d(np.asarray(ensure_in_units(frequency, u.Hz), dtype=float))
    r_cgs = np.asarray(ensure_in_units(r, u.cm), dtype=float)
    n_e_cgs = np.asarray(ensure_in_units(n_e, u.cm**-3), dtype=float)
    n_i_cgs = np.asarray(ensure_in_units(n_i, u.cm**-3), dtype=float)
    T_cgs = float(ensure_in_units(temperature, u.K))

    return _compute_RJ_ff_optical_depth_from_arrays(
        log_nu=np.log(nu_cgs),
        r=r_cgs,
        log_n_e=np.log(n_e_cgs),
        log_n_i=np.log(n_i_cgs),
        log_T=np.log(T_cgs),
        Z=Z,
        X=X,
        g_ff=g_ff,
        thomson=thomson,
    )


def compute_ff_RJ_optical_depth_wind(
    frequency: "_UnitBearingArrayLike",
    r_min: "Union[float, u.Quantity]",
    r_0: "Union[float, u.Quantity]",
    n_e_0: "Union[float, u.Quantity]",
    n_i_0: "Union[float, u.Quantity]",
    *,
    r_max: "Union[float, u.Quantity]" = np.inf,
    temperature: "Union[float, u.Quantity]" = 1.0e4,
    Z: "Union[float, _ArrayLike]" = 1.0,
    X: "Optional[_ArrayLike]" = None,
    g_ff: "Optional[Union[float, _ArrayLike]]" = None,
    thomson: bool = False,
) -> np.ndarray:
    r"""Free-free optical depth through a steady spherically-symmetric wind (RJ limit).

    Assumes :math:`n_e(r) = n_{e,0}\,(r_0/r)^2`, giving
    :math:`\tau_{\rm ff}^{\rm RJ} \propto \nu^{-2} T^{-3/2}`.

    Parameters
    ----------
    frequency : float or `~astropy.units.Quantity` or array-like
        Photon frequencies.  Bare floats assumed to be in **Hz**.
    r_min : float or `~astropy.units.Quantity`
        Inner integration radius.  Bare floats assumed to be in **cm**.
    r_0 : float or `~astropy.units.Quantity`
        Reference radius where ``n_e_0``, ``n_i_0`` are specified.
        Bare floats assumed to be in **cm**.
    n_e_0 : float or `~astropy.units.Quantity`
        Electron number density at :math:`r_0`.
        Bare floats assumed to be in **cm**\ :sup:`-3`.
    n_i_0 : float or `~astropy.units.Quantity`
        Ion number density at :math:`r_0`.
        Bare floats assumed to be in **cm**\ :sup:`-3`.
    r_max : float or `~astropy.units.Quantity`, optional
        Outer integration radius.  Default ``numpy.inf``.
    temperature : float or `~astropy.units.Quantity`, optional
        Uniform electron temperature.  Bare floats assumed to be in **K**.
        Default ``1e4``.
    Z : float or array-like, optional
        Ionic charge(s).  Default ``1.0``.
    X : array-like or None, optional
        Number fractions.  ``None`` → single-species mode.
    g_ff : float, array-like, or None, optional
        Gaunt factor(s).  Auto-computed if ``None``.
    thomson : bool, optional
        Return :math:`\tau_{\rm eff}` if ``True``.  Default ``False``.

    Returns
    -------
    tau : ~numpy.ndarray
        Optical depths, one entry per frequency.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        from astropy import units as u
        from trilobite.radiation.free_free.absorption import (
            compute_ff_RJ_optical_depth_wind,
        )

        nu_arr = np.geomspace(1e8, 1e10, 50) * u.Hz
        tau = compute_ff_RJ_optical_depth_wind(
            frequency=nu_arr,
            r_min=1e15 * u.cm,
            r_0=1e16 * u.cm,
            n_e_0=1e4 / u.cm**3,
            n_i_0=1e4 / u.cm**3,
            temperature=1e4 * u.K,
        )

    See Also
    --------
    compute_ff_RJ_optical_depth_powerlaw : General power-law density profile.
    compute_ff_RJ_optical_depth_from_quadrature : Arbitrary-profile variant.
    """
    nu_cgs = np.atleast_1d(np.asarray(ensure_in_units(frequency, u.Hz), dtype=float))
    r_min_cgs = float(ensure_in_units(r_min, u.cm))
    r_0_cgs = float(ensure_in_units(r_0, u.cm))
    n_e_0_cgs = float(ensure_in_units(n_e_0, u.cm**-3))
    n_i_0_cgs = float(ensure_in_units(n_i_0, u.cm**-3))
    r_max_cgs = float(ensure_in_units(r_max, u.cm))
    T_cgs = float(ensure_in_units(temperature, u.K))

    return _compute_RJ_ff_optical_depth_wind(
        log_nu=np.log(nu_cgs),
        log_r_min=np.log(r_min_cgs),
        log_r_0=np.log(r_0_cgs),
        log_n_e_0=np.log(n_e_0_cgs),
        log_n_i_0=np.log(n_i_0_cgs),
        log_r_max=np.log(r_max_cgs),
        log_T=np.log(T_cgs),
        Z=Z,
        X=X,
        g_ff=g_ff,
        thomson=thomson,
    )


def compute_ff_RJ_optical_depth_wind_Mdot(
    frequency: "_UnitBearingArrayLike",
    Mdot: "Union[float, u.Quantity]",
    v_w: "Union[float, u.Quantity]",
    r_min: "Union[float, u.Quantity]",
    *,
    mu_e: float = 1.0,
    mu_i: float = 1.0,
    r_max: "Union[float, u.Quantity]" = np.inf,
    temperature: "Union[float, u.Quantity]" = 1.0e4,
    Z: "Union[float, _ArrayLike]" = 1.0,
    X: "Optional[_ArrayLike]" = None,
    g_ff: "Optional[Union[float, _ArrayLike]]" = None,
    thomson: bool = False,
) -> np.ndarray:
    r"""Free-free optical depth through a steady wind, parameterised by mass-loss rate (RJ limit).

    Converts the mass-loss rate :math:`\dot{M}` and wind velocity :math:`v_w` into
    electron and ion number densities at the reference radius :math:`r_0` via

    .. math::

        n_{e,0} = \frac{\dot{M}}{4\pi\,m_p\,\mu_e\,v_w\,r_0^2},
        \qquad
        n_{i,0} = \frac{\dot{M}}{4\pi\,m_p\,\mu_i\,v_w\,r_0^2},

    then evaluates the same analytic wind integral as
    :func:`compute_ff_RJ_optical_depth_wind`.

    Parameters
    ----------
    frequency : float or `~astropy.units.Quantity` or array-like
        Photon frequencies.  Bare floats assumed to be in **Hz**.
    Mdot : float or `~astropy.units.Quantity`
        Mass-loss rate :math:`\dot{M}`.  Bare floats assumed to be in **g/s**.
        Typical usage: ``1e-5 * u.Msun / u.yr``.
    v_w : float or `~astropy.units.Quantity`
        Wind terminal velocity :math:`v_w`.  Bare floats assumed to be in **cm/s**.
        Typical usage: ``1e3 * u.km / u.s``.
    r_min : float or `~astropy.units.Quantity`
        Inner integration radius.  Bare floats assumed to be in **cm**.
    mu_e : float, optional
        Mean molecular weight per electron :math:`\mu_e`.  Default ``1.0``
        (fully ionized hydrogen).  For a solar-composition H/He plasma use
        ``mu_e = 1.14``.
    mu_i : float, optional
        Mean molecular weight per ion :math:`\mu_i`.  Default ``1.0``
        (fully ionized hydrogen).  For a solar-composition H/He plasma use
        ``mu_i = 1.27``.
    r_max : float or `~astropy.units.Quantity`, optional
        Outer integration radius.  Default ``numpy.inf``.
    temperature : float or `~astropy.units.Quantity`, optional
        Uniform electron temperature.  Bare floats assumed to be in **K**.
        Default ``1e4``.
    Z : float or array-like, optional
        Ionic charge(s).  Default ``1.0``.
    X : array-like or None, optional
        Number fractions.  ``None`` → single-species mode.
    g_ff : float, array-like, or None, optional
        Gaunt factor(s).  Auto-computed if ``None``.
    thomson : bool, optional
        Return :math:`\tau_{\rm eff}` if ``True``.  Default ``False``.

    Returns
    -------
    tau : ~numpy.ndarray
        Optical depths, one entry per frequency.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        from astropy import units as u
        from trilobite.radiation.free_free import (
            compute_ff_RJ_optical_depth_wind_Mdot,
        )

        nu = np.geomspace(1e8, 1e10, 50) * u.Hz
        tau = compute_ff_RJ_optical_depth_wind_Mdot(
            frequency=nu,
            Mdot=1e-5 * u.Msun / u.yr,
            v_w=1e3 * u.km / u.s,
            r_min=1e15 * u.cm,
        )

    See Also
    --------
    compute_ff_RJ_optical_depth_wind : Same geometry parameterised by :math:`n_{e,0}`, :math:`n_{i,0}`.
    compute_ff_RJ_optical_depth_powerlaw : General power-law density profile.
    """
    nu_cgs = np.atleast_1d(np.asarray(ensure_in_units(frequency, u.Hz), dtype=float))
    Mdot_cgs = float(ensure_in_units(Mdot, u.g / u.s))
    v_w_cgs = float(ensure_in_units(v_w, u.cm / u.s))
    r_min_cgs = float(ensure_in_units(r_min, u.cm))
    r_max_cgs = float(ensure_in_units(r_max, u.cm))
    T_cgs = float(ensure_in_units(temperature, u.K))

    r_0_cgs = r_min_cgs

    prefactor = 4.0 * np.pi * _m_p_cgs * v_w_cgs * r_0_cgs**2
    n_e_0_cgs = Mdot_cgs / (mu_e * prefactor)
    n_i_0_cgs = Mdot_cgs / (mu_i * prefactor)

    return _compute_RJ_ff_optical_depth_wind(
        log_nu=np.log(nu_cgs),
        log_r_min=np.log(r_min_cgs),
        log_r_0=np.log(r_0_cgs),
        log_n_e_0=np.log(n_e_0_cgs),
        log_n_i_0=np.log(n_i_0_cgs),
        log_r_max=np.log(r_max_cgs),
        log_T=np.log(T_cgs),
        Z=Z,
        X=X,
        g_ff=g_ff,
        thomson=thomson,
    )


def compute_ff_RJ_optical_depth_shell(
    frequency: "_UnitBearingArrayLike",
    r_min: "Union[float, u.Quantity]",
    n_e_0: "Union[float, u.Quantity]",
    n_i_0: "Union[float, u.Quantity]",
    *,
    r_max: "Union[float, u.Quantity]" = np.inf,
    temperature: "Union[float, u.Quantity]" = 1.0e4,
    Z: "Union[float, _ArrayLike]" = 1.0,
    X: "Optional[_ArrayLike]" = None,
    g_ff: "Optional[Union[float, _ArrayLike]]" = None,
    thomson: bool = False,
) -> np.ndarray:
    r"""Free-free optical depth through a uniform-density spherical shell (RJ limit).

    :math:`\tau_{\rm ff}^{\rm RJ}(\nu) = \alpha_\nu^{\rm RJ}\,(r_{\rm max} - r_{\rm min})`,
    scaling as :math:`\nu^{-2} T^{-3/2}`.

    Parameters
    ----------
    frequency : float or `~astropy.units.Quantity` or array-like
        Photon frequencies.  Bare floats assumed to be in **Hz**.
    r_min : float or `~astropy.units.Quantity`
        Inner shell radius.  Bare floats assumed to be in **cm**.
    n_e_0 : float or `~astropy.units.Quantity`
        Uniform electron number density.
        Bare floats assumed to be in **cm**\ :sup:`-3`.
    n_i_0 : float or `~astropy.units.Quantity`
        Uniform ion number density.
        Bare floats assumed to be in **cm**\ :sup:`-3`.
    r_max : float or `~astropy.units.Quantity`, optional
        Outer shell radius.  Default ``numpy.inf``.
    temperature : float or `~astropy.units.Quantity`, optional
        Uniform electron temperature.  Bare floats assumed to be in **K**.
        Default ``1e4``.
    Z : float or array-like, optional
        Ionic charge(s).  Default ``1.0``.
    X : array-like or None, optional
        Number fractions.  ``None`` → single-species mode.
    g_ff : float, array-like, or None, optional
        Gaunt factor(s).  Auto-computed if ``None``.
    thomson : bool, optional
        Return :math:`\tau_{\rm eff}` if ``True``.  Default ``False``.

    Returns
    -------
    tau : ~numpy.ndarray
        Optical depths, one entry per frequency.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        from astropy import units as u
        from trilobite.radiation.free_free.absorption import (
            compute_ff_RJ_optical_depth_shell,
        )

        nu_arr = np.geomspace(1e8, 1e10, 50) * u.Hz
        tau = compute_ff_RJ_optical_depth_shell(
            frequency=nu_arr,
            r_min=1e15 * u.cm,
            n_e_0=1e4 / u.cm**3,
            n_i_0=1e4 / u.cm**3,
            r_max=2e15 * u.cm,
            temperature=1e4 * u.K,
        )

    See Also
    --------
    compute_ff_RJ_optical_depth_wind : Wind (r\ :sup:`-2`) variant.
    compute_ff_RJ_optical_depth_powerlaw : General power-law variant.
    """
    nu_cgs = np.atleast_1d(np.asarray(ensure_in_units(frequency, u.Hz), dtype=float))
    r_min_cgs = float(ensure_in_units(r_min, u.cm))
    n_e_0_cgs = float(ensure_in_units(n_e_0, u.cm**-3))
    n_i_0_cgs = float(ensure_in_units(n_i_0, u.cm**-3))
    r_max_cgs = float(ensure_in_units(r_max, u.cm))
    T_cgs = float(ensure_in_units(temperature, u.K))

    return _compute_RJ_ff_optical_depth_shell(
        log_nu=np.log(nu_cgs),
        log_r_min=np.log(r_min_cgs),
        log_n_e_0=np.log(n_e_0_cgs),
        log_n_i_0=np.log(n_i_0_cgs),
        log_r_max=np.log(r_max_cgs),
        log_T=np.log(T_cgs),
        Z=Z,
        X=X,
        g_ff=g_ff,
        thomson=thomson,
    )


def compute_ff_RJ_optical_depth_powerlaw(
    frequency: "_UnitBearingArrayLike",
    r_min: "Union[float, u.Quantity]",
    r_0: "Union[float, u.Quantity]",
    n_e_0: "Union[float, u.Quantity]",
    n_i_0: "Union[float, u.Quantity]",
    p: float,
    *,
    r_max: "Union[float, u.Quantity]" = np.inf,
    temperature: "Union[float, u.Quantity]" = 1.0e4,
    Z: "Union[float, _ArrayLike]" = 1.0,
    X: "Optional[_ArrayLike]" = None,
    g_ff: "Optional[Union[float, _ArrayLike]]" = None,
    thomson: bool = False,
) -> np.ndarray:
    r"""Free-free optical depth for a power-law density profile (RJ limit).

    Assumes :math:`n_e(r) = n_{e,0}\,(r_0/r)^p`.  Setting :math:`k = 1 - 2p`:

    .. math::

        \tau_{\rm ff}^{\rm RJ}(\nu) = \alpha_\nu^{\rm RJ,(1)}
        \begin{cases}
            (r_{\rm max}^k - r_{\rm min}^k)/k & k \neq 0 \\
            \ln(r_{\rm max}/r_{\rm min}) & k = 0
        \end{cases}

    Parameters
    ----------
    frequency : float or `~astropy.units.Quantity` or array-like
        Photon frequencies.  Bare floats assumed to be in **Hz**.
    r_min : float or `~astropy.units.Quantity`
        Inner integration radius.  Bare floats assumed to be in **cm**.
    r_0 : float or `~astropy.units.Quantity`
        Reference radius where ``n_e_0``, ``n_i_0`` are specified.
        Bare floats assumed to be in **cm**.
    n_e_0 : float or `~astropy.units.Quantity`
        Electron number density at :math:`r_0`.
        Bare floats assumed to be in **cm**\ :sup:`-3`.
    n_i_0 : float or `~astropy.units.Quantity`
        Ion number density at :math:`r_0`.
        Bare floats assumed to be in **cm**\ :sup:`-3`.
    p : float
        Power-law index (:math:`n \propto r^{-p}`).  Common values:
        :math:`p = 2` (wind), :math:`p = 7/4` (Chevalier), :math:`p = 0` (shell).
    r_max : float or `~astropy.units.Quantity`, optional
        Outer radius.  Default ``numpy.inf``.
    temperature : float or `~astropy.units.Quantity`, optional
        Uniform electron temperature.  Bare floats assumed to be in **K**.
        Default ``1e4``.
    Z : float or array-like, optional
        Ionic charge(s).  Default ``1.0``.
    X : array-like or None, optional
        Number fractions.  ``None`` → single-species mode.
    g_ff : float, array-like, or None, optional
        Gaunt factor(s).  Auto-computed if ``None``.
    thomson : bool, optional
        Return :math:`\tau_{\rm eff}` if ``True``.  Default ``False``.

    Returns
    -------
    tau : ~numpy.ndarray
        Optical depths, one entry per frequency.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        from astropy import units as u
        from trilobite.radiation.free_free.absorption import (
            compute_ff_RJ_optical_depth_powerlaw,
        )

        nu_arr = np.geomspace(1e8, 1e10, 50) * u.Hz
        tau = compute_ff_RJ_optical_depth_powerlaw(
            frequency=nu_arr,
            r_min=1e15 * u.cm, r_0=1e16 * u.cm,
            n_e_0=1e4 / u.cm**3, n_i_0=1e4 / u.cm**3,
            p=2.0, temperature=1e4 * u.K,
        )

    See Also
    --------
    compute_ff_RJ_optical_depth_wind : Optimised :math:`p = 2` variant.
    compute_ff_RJ_optical_depth_shell : Uniform-density (:math:`p = 0`) variant.
    """
    nu_cgs = np.atleast_1d(np.asarray(ensure_in_units(frequency, u.Hz), dtype=float))
    r_min_cgs = float(ensure_in_units(r_min, u.cm))
    r_0_cgs = float(ensure_in_units(r_0, u.cm))
    n_e_0_cgs = float(ensure_in_units(n_e_0, u.cm**-3))
    n_i_0_cgs = float(ensure_in_units(n_i_0, u.cm**-3))
    r_max_cgs = float(ensure_in_units(r_max, u.cm))
    T_cgs = float(ensure_in_units(temperature, u.K))

    return _compute_RJ_ff_optical_depth_powerlaw(
        log_nu=np.log(nu_cgs),
        log_r_min=np.log(r_min_cgs),
        log_r_0=np.log(r_0_cgs),
        log_n_e_0=np.log(n_e_0_cgs),
        log_n_i_0=np.log(n_i_0_cgs),
        p=p,
        log_r_max=np.log(r_max_cgs),
        log_T=np.log(T_cgs),
        Z=Z,
        X=X,
        g_ff=g_ff,
        thomson=thomson,
    )


def compute_ff_RJ_optical_depth_EM(
    frequency: "_UnitBearingArrayLike",
    EM: "Union[float, u.Quantity]",
    *,
    temperature: "Union[float, u.Quantity]" = 1.0e4,
    Z: "Union[float, _ArrayLike]" = 1.0,
    X: "Optional[_ArrayLike]" = None,
    g_ff: "Optional[Union[float, _ArrayLike]]" = None,
) -> np.ndarray:
    r"""Free-free optical depth from emission measure (RJ limit).

    Evaluates

    .. math::

        \tau_{\rm ff}^{\rm RJ}(\nu) =
        C_\alpha\, Z^2\, {\rm EM}\, T^{-3/2}\, \nu^{-2}\, g_{\rm ff},
        \qquad
        {\rm EM} = \int n_e\, n_i\, dr

    This is the preferred form when the emission measure is known directly
    (e.g.\ from spectral fitting or a model integral) rather than from an
    explicit density profile.

    Parameters
    ----------
    frequency : float or `~astropy.units.Quantity` or array-like
        Photon frequencies :math:`\nu`.  Bare floats assumed to be in **Hz**.
    EM : float or `~astropy.units.Quantity`
        Emission measure :math:`{\rm EM} = \int n_e\, n_i\, dr`.
        Bare floats assumed to be in **cm**\ :sup:`-5`.
    temperature : float or `~astropy.units.Quantity`, optional
        Uniform electron temperature.  Bare floats assumed to be in **K**.
        Default ``1e4``.
    Z : float or array-like, optional
        Ionic charge(s).  Default ``1.0``.
    X : array-like or None, optional
        Number fractions.  ``None`` → single-species mode.
    g_ff : float, array-like, or None, optional
        Gaunt factor(s).  Auto-computed via Lu+ if ``None``.

    Returns
    -------
    tau : ~numpy.ndarray
        Dimensionless optical depths, one entry per frequency.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        from astropy import units as u
        from trilobite.radiation.free_free.absorption import (
            compute_ff_RJ_optical_depth_EM,
        )

        nu_arr = np.geomspace(1e8, 1e10, 50) * u.Hz
        tau = compute_ff_RJ_optical_depth_EM(
            frequency=nu_arr,
            EM=1e24 * u.cm**-5,
            temperature=1e4 * u.K,
        )

    See Also
    --------
    compute_ff_RJ_optical_depth_wind : Analytic wind profile.
    compute_ff_RJ_optical_depth_from_arrays : Integrates over a discrete profile.
    """
    nu_cgs = np.atleast_1d(np.asarray(ensure_in_units(frequency, u.Hz), dtype=float))
    EM_cgs = float(ensure_in_units(EM, u.cm**-5))
    T_cgs = float(ensure_in_units(temperature, u.K))

    return _compute_RJ_ff_optical_depth_EM(
        log_nu=np.log(nu_cgs),
        log_EM=np.log(EM_cgs),
        log_T=np.log(T_cgs),
        Z=Z,
        X=X,
        g_ff=g_ff,
    )
