r"""
Upstream profile utilities for numerical shock engines.

This module provides factory functions and simple callables for constructing
the velocity and density profiles that numerical shock engines require as
upstream boundary conditions. Three groups of helpers are included.

**Velocity profiles** are simple callables ``u(r, t)`` representing the bulk
velocity field of the upstream gas. Stationary, homologous, and constant
variants are provided.

**Ejecta density profiles** model freely expanding supernova ejecta under the
homologous approximation :math:`\rho_{\rm ej}(r,t) = t^{-3}G(r/t)`, where
:math:`G(v)` is a time-independent velocity-space kernel. Broken-power-law
(Chevalier-style) and exponential kernels are supported, with optional
upper-velocity truncation and analytic normalization to supplied ejecta mass
and kinetic energy.

**CSM density profiles** cover the common circumstellar medium geometries
encountered in transient modeling: uniform, steady wind, power-law, top-hat
shell, Gaussian shell, wind with floor, and smooth-truncated wind.

All returned callables are unit-free (CGS) for efficient evaluation inside
ODE right-hand sides; unit handling is performed once at factory-call time.
"""

from collections.abc import Callable

import numpy as np
from astropy import units as u
from scipy.optimize import brentq

from triceratops._typing import _ArrayLike, _UnitBearingArrayLike, _UnitBearingScalarLike
from triceratops.utils.misc_utils import ensure_in_units


# =============================================== #
# Source-Function Factory                         #
# =============================================== #
def make_homologous_stationary_sources(
    G_ej: "Callable[['_ArrayLike'], '_ArrayLike']",
    rho_csm: "Callable[['_ArrayLike'], '_ArrayLike']",
) -> "tuple[Callable, Callable, Callable, Callable]":
    r"""
    Build the four upstream source callables for homologous ejecta in stationary CSM.

    This convenience factory constructs ``(rho_1, u_1, rho_4, u_4)`` — the four
    two-argument callables expected by numerical shock engines — for the standard
    case of freely expanding homologous ejecta running into a stationary
    circumstellar medium.

    The factory wraps the time-independent kernel and density profile into the
    two-argument ``(r, t)`` interface required by the ODE right-hand sides and
    initial-condition solvers:

    .. math::

        \rho_1(r,\,t) = t^{-3}\,G_{\rm ej}(r/t),
        \qquad
        u_1(r,\,t) = \frac{r}{t},
        \qquad
        \rho_4(r,\,t) = \rho_{\rm CSM}(r),
        \qquad
        u_4(r,\,t) = 0.

    Parameters
    ----------
    G_ej : callable
        Time-independent ejecta velocity kernel ``G_ej(v)`` in CGS, such that
        the physical ejecta density is

        .. math::

            \rho_{\rm ej}(r,\,t) = t^{-3}\,G_{\rm ej}(r/t).

        Suitable kernels are returned by :func:`get_bpl_ejecta_kernel` and
        :func:`get_exponential_ejecta_kernel`.
    rho_csm : callable
        Stationary CSM density profile ``rho_csm(r)`` in
        :math:`\mathrm{g\,cm^{-3}}`, with ``r`` in cm.  Any profile callable
        that accepts a scalar or array radius is accepted. For common profiles
        see :func:`get_wind_csm_density_func`,
        :func:`get_uniform_csm_density_func`, and related helpers.

    Returns
    -------
    rho_1 : callable
        Upstream ejecta density ``rho_1(r, t)`` in :math:`\mathrm{g\,cm^{-3}}`.
    u_1 : callable
        Upstream ejecta velocity ``u_1(r, t) = r/t`` in cm/s (homologous flow).
    rho_4 : callable
        Upstream CSM density ``rho_4(r, t)`` in :math:`\mathrm{g\,cm^{-3}}`
        (``t`` is accepted for API compatibility and ignored).
    u_4 : callable
        Upstream CSM velocity ``u_4(r, t) = 0`` in cm/s (stationary medium).
    """

    def rho_1(r: "_ArrayLike", t: "_ArrayLike") -> "_ArrayLike":
        r_arr = np.asarray(r, dtype=float)
        t_arr = np.asarray(t, dtype=float)
        return G_ej(r_arr / t_arr) / t_arr**3

    def u_1(r: "_ArrayLike", t: "_ArrayLike") -> "_ArrayLike":
        return np.asarray(r, dtype=float) / np.asarray(t, dtype=float)

    def rho_4(r: "_ArrayLike", _t: "_ArrayLike | None" = None) -> "_ArrayLike":
        return rho_csm(r)

    def u_4(r: "_ArrayLike", _t: "_ArrayLike | None" = None) -> "_ArrayLike":
        r_arr = np.asarray(r)
        if r_arr.ndim == 0:
            return 0.0
        return np.zeros_like(r_arr, dtype=float)

    return rho_1, u_1, rho_4, u_4


# =============================================== #
# Generic Profile Functions                       #
# =============================================== #
def stationary_velocity_profile(
    r: "_ArrayLike",
    _t: "_ArrayLike | None" = None,
) -> "_ArrayLike":
    r"""
    Return a zero velocity field in cm/s.

    Parameters
    ----------
    r : float or array-like
        Radius in cm. Used only to determine the output shape.
    _t : float or array-like, optional
        Time in s. Accepted for API compatibility and ignored.

    Returns
    -------
    v : float or numpy.ndarray
        Zero velocity in cm/s.
    """
    r_array = np.asarray(r)

    if r_array.ndim == 0:
        return 0.0

    return np.zeros_like(r_array, dtype=float)


def homologous_velocity_profile(r: "_ArrayLike", t: "_ArrayLike") -> "_ArrayLike":
    r"""
    Return the homologous velocity field :math:`u(r,t) = r/t` in cm/s.

    Parameters
    ----------
    r : float or array-like
        Radius in cm.
    t : float or array-like
        Time since explosion in s.

    Returns
    -------
    u : float or numpy.ndarray
        Homologous flow velocity in cm/s.
    """
    return np.asarray(r, dtype=float) / np.asarray(t, dtype=float)


def get_constant_velocity_profile(
    velocity: "_UnitBearingScalarLike",
) -> Callable[["_ArrayLike", "_ArrayLike | None"], "_ArrayLike"]:
    r"""
    Build a constant velocity profile callable.

    The returned function evaluates

    .. math::

        u(r,t) = u_0,

    in CGS units. The optional time argument is accepted for compatibility with
    shock-engine source functions and is ignored.

    Parameters
    ----------
    velocity : astropy.units.Quantity or float
        Constant velocity. Unit-bearing inputs are converted to cm/s; unit-free
        inputs are interpreted as cm/s.

    Returns
    -------
    velocity_profile : callable
        Callable ``velocity_profile(r, _t=None)`` returning velocity in cm/s.
        The input ``r`` is used only to determine the output shape.
    """
    velocity_cgs = ensure_in_units(velocity, u.cm / u.s)
    velocity_cgs = float(velocity_cgs)

    def _velocity_profile(
        r: "_ArrayLike",
        _t: "_ArrayLike | None" = None,
    ) -> "_ArrayLike":
        r_array = np.asarray(r)

        if r_array.ndim == 0:
            return velocity_cgs

        return np.full_like(r_array, velocity_cgs, dtype=float)

    return _velocity_profile


# =============================================== #
# Ejecta Properties and Generation                #
# =============================================== #
# These functions provide a number of utilities for computing properties of the ejecta in
# various shock scenarios.
def _normalize_BPL_ejecta(
    E_ej: "_ArrayLike",
    M_ej: "_ArrayLike",
    n: "_ArrayLike" = 10,
    delta: "_ArrayLike" = 0,
) -> tuple["_ArrayLike", "_ArrayLike"]:
    r"""
    Compute the broken-power-law ejecta transition velocity and normalization.

    This is the low-level, unit-free implementation used by
    :func:`normalize_bpl_ejecta`. Inputs are assumed to be in CGS units and
    may be scalars or broadcast-compatible ~numpy.ndarray objects.

    The ejecta density profile is assumed to be homologous and separable,

    .. math::

        \rho_{\rm ej}(r,t)
        =
        t^{-3} G(v),
        \qquad
        v = \frac{r}{t},

    with a broken power-law velocity profile

    .. math::

        G(v)
        =
        \begin{cases}
            K v^{-\delta}, & v < v_t, \\
            K v_t^{n-\delta} v^{-n}, & v \geq v_t,
        \end{cases}

    where :math:`v_t` is the transition velocity, :math:`K` is the inner-profile
    normalization, :math:`\delta` is the inner power-law index, and :math:`n` is
    the outer power-law index.

    The returned values are chosen so that the profile integrates to the supplied
    ejecta mass and kinetic energy. For the mass and energy integrals to
    converge, the usual broken-power-law constraints are

    .. math::

        \delta < 3,
        \qquad
        n > 5.

    Parameters
    ----------
    E_ej : float or ~numpy.ndarray
        Total kinetic energy of the ejecta in erg.
    M_ej : float or ~numpy.ndarray
        Total ejecta mass in g.
    n : float or ~numpy.ndarray, optional
        Outer ejecta density power-law index. This must satisfy :math:`n > 5`
        for a finite kinetic energy. Default is ``10``.
    delta : float or ~numpy.ndarray, optional
        Inner ejecta density power-law index. This must satisfy
        :math:`\delta < 3` for a finite mass. Default is ``0``.

    Returns
    -------
    v_t : float or ~numpy.ndarray
        Transition velocity between the inner and outer ejecta profiles in cm/s.
    K : float or ~numpy.ndarray
        Inner-profile normalization in CGS units. For

        .. math::

            G(v) = K v^{-\delta},

        the units of ``K`` are

        .. math::

            \mathrm{g\,cm^{\delta - 3}\,s^{3-\delta}}.

    Notes
    -----
    This helper intentionally performs no unit conversion or validation. The
    public wrapper should coerce units, check physical parameter ranges, and
    provide user-facing error messages.
    """
    if np.any(E_ej <= 0):
        raise ValueError("Ejecta kinetic energy `E_ej` must be positive.")
    if np.any(M_ej <= 0):
        raise ValueError("Ejecta mass `M_ej` must be positive.")

    # Twice the kinetic energy per unit mass sets the characteristic velocity
    # scale appearing in the transition velocity.
    specific_velocity_squared = 2.0 * E_ej / M_ej

    # Transition velocity fixed by simultaneous mass and kinetic-energy
    # normalization of the broken power law.
    v_t = np.sqrt(specific_velocity_squared * ((5.0 - delta) * (n - 5.0)) / ((3.0 - delta) * (n - 3.0)))

    # Inner-profile normalization. The outer branch is continuous when written
    # as K * v_t**(n - delta) * v**(-n).
    K = M_ej * v_t ** (delta - 3.0) * ((3.0 - delta) * (n - 3.0)) / (4.0 * np.pi * (n - delta))

    return v_t, K


def _normalize_truncated_bpl_ejecta(
    E_ej: float,
    M_ej: float,
    n: float = 10,
    delta: float = 0,
    v_max: float = np.inf,
) -> tuple[float, float]:
    r"""
    Normalize an upper-truncated broken-power-law ejecta profile in CGS.

    The kernel is

    .. math::

        G(v) =
        \begin{cases}
            K v^{-\delta}, & 0 \le v < v_t,\\
            K v_t^{n-\delta} v^{-n}, & v_t \le v \le v_{\max},\\
            0, & v > v_{\max}.
        \end{cases}

    The transition velocity :math:`v_t` is chosen so that the truncated profile
    has the requested kinetic energy per unit mass. Once :math:`v_t` is known,
    the mass constraint fixes :math:`K`.

    Parameters
    ----------
    E_ej : float
        Total ejecta kinetic energy in erg.
    M_ej : float
        Total ejecta mass in g.
    n : float, optional
        Outer ejecta density power-law index. Must satisfy ``n > 5``.
    delta : float, optional
        Inner ejecta density power-law index. Must satisfy ``delta < 3``.
    v_max : float, optional
        Maximum ejecta velocity in cm/s. Default is ``np.inf``, which recovers
        the untruncated BPL normalization.

    Returns
    -------
    v_t : float
        Transition velocity in cm/s.
    K : float
        Inner-branch normalization in
        :math:`\mathrm{g\,cm^{\delta - 3}\,s^{3-\delta}}`.
    """
    if E_ej <= 0:
        raise ValueError("Ejecta kinetic energy `E_ej` must be positive.")
    if M_ej <= 0:
        raise ValueError("Ejecta mass `M_ej` must be positive.")
    if delta >= 3:
        raise ValueError("Inner ejecta index `delta` must be < 3 for finite mass.")
    if n <= 5:
        raise ValueError("Outer ejecta index `n` must be > 5 for finite kinetic energy.")
    if n <= delta:
        raise ValueError("Outer index `n` must exceed inner index `delta`.")
    if v_max <= 0:
        raise ValueError("Maximum ejecta velocity `v_max` must be positive.")

    if np.isinf(v_max):
        return _normalize_BPL_ejecta(E_ej=E_ej, M_ej=M_ej, n=n, delta=delta)

    target = E_ej / M_ej

    if target >= 0.5 * v_max**2:
        raise ValueError("Requested E_ej / M_ej is incompatible with `v_max`: it must be less than 0.5 * v_max**2.")

    def _power_integral(a: float, b: float, p: float) -> float:
        if b <= a:
            return 0.0
        if a == 0.0 and p <= -1.0:
            raise ValueError("Power-law integral diverges at zero.")
        if np.isclose(p, -1.0):
            return np.log(b / a)
        return (b ** (p + 1.0) - a ** (p + 1.0)) / (p + 1.0)

    def _moment(v_t: float, q: int) -> float:
        inner_upper = min(v_t, v_max)

        I_inner = _power_integral(
            a=0.0,
            b=inner_upper,
            p=q - delta,
        )

        I_outer = 0.0
        if v_t < v_max:
            I_outer = v_t ** (n - delta) * _power_integral(
                a=v_t,
                b=v_max,
                p=q - n,
            )

        return I_inner + I_outer

    def _residual(log_v_t: float) -> float:
        v_t = np.exp(log_v_t)
        I2 = _moment(v_t, 2)
        I4 = _moment(v_t, 4)
        return 0.5 * I4 / I2 - target

    # v_t must be positive. In practice, the root lies below or near v_max for
    # physically sensible truncated profiles, but searching a broad interval is
    # safer.
    lower = v_max * 1.0e-12
    upper = v_max * (1.0 - 1.0e-12)

    grid = np.geomspace(lower, upper, 256)
    values = np.array([_residual(np.log(v)) for v in grid])

    bracket = None
    for left, right, f_left, f_right in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
        if f_left == 0.0:
            bracket = (left, left)
            break
        if np.sign(f_left) != np.sign(f_right):
            bracket = (left, right)
            break

    if bracket is None:
        raise RuntimeError(
            "Could not bracket the truncated BPL transition velocity. "
            "The requested E/M may be incompatible with the chosen `v_max`."
        )

    if bracket[0] == bracket[1]:
        v_t = bracket[0]
    else:
        v_t = np.exp(brentq(_residual, np.log(bracket[0]), np.log(bracket[1])))

    I2 = _moment(v_t, 2)
    K = M_ej / (4.0 * np.pi * I2)

    return v_t, K


def _normalize_exponential_ejecta(
    E_ej: "_ArrayLike",
    M_ej: "_ArrayLike",
) -> tuple["_ArrayLike", "_ArrayLike"]:
    r"""
    Normalize an untruncated exponential homologous ejecta profile in CGS.

    The kernel is

    .. math::

        G(v) = K\exp(-v/v_e),

    with :math:`\rho_{\rm ej}(r,t) = t^{-3}G(r/t)`. Enforcing the total mass
    and kinetic energy gives

    .. math::

        v_e = \sqrt{\frac{E_{\rm ej}}{6M_{\rm ej}}},
        \qquad
        K = \frac{M_{\rm ej}}{8\pi v_e^3}.

    Parameters
    ----------
    E_ej : float or array-like
        Total ejecta kinetic energy in erg.
    M_ej : float or array-like
        Total ejecta mass in g.

    Returns
    -------
    v_e : float or array-like
        Exponential velocity scale in cm/s.
    K : float or array-like
        Kernel normalization in :math:`\mathrm{g\,s^3\,cm^{-3}}`.
    """
    if np.any(E_ej <= 0):
        raise ValueError("Ejecta kinetic energy `E_ej` must be positive.")
    if np.any(M_ej <= 0):
        raise ValueError("Ejecta mass `M_ej` must be positive.")

    v_e = np.sqrt(E_ej / (6.0 * M_ej))
    K = M_ej / (8.0 * np.pi * v_e**3)

    return v_e, K


def _normalize_truncated_exponential_ejecta(
    E_ej: float,
    M_ej: float,
    v_min: float = 0.0,
    v_max: float = np.inf,
) -> tuple[float, float]:
    r"""
    Normalize a velocity-truncated exponential homologous ejecta profile in CGS.

    The kernel is

    .. math::

        G(v) =
        \begin{cases}
            K\exp(-v/v_e), & v_{\min} \le v \le v_{\max},\\
            0, & \mathrm{otherwise}.
        \end{cases}

    The scale velocity :math:`v_e` is chosen so that the truncated profile has
    the requested kinetic energy per unit mass. Once :math:`v_e` is known, the
    mass constraint fixes :math:`K`.

    Parameters
    ----------
    E_ej : float
        Total ejecta kinetic energy in erg.
    M_ej : float
        Total ejecta mass in g.
    v_min : float, optional
        Lower velocity cutoff in cm/s. Default is ``0``.
    v_max : float, optional
        Upper velocity cutoff in cm/s. Default is ``np.inf``.

    Returns
    -------
    v_e : float
        Exponential velocity scale in cm/s.
    K : float
        Kernel normalization in :math:`\mathrm{g\,s^3\,cm^{-3}}`.
    """
    if E_ej <= 0:
        raise ValueError("Ejecta kinetic energy `E_ej` must be positive.")
    if M_ej <= 0:
        raise ValueError("Ejecta mass `M_ej` must be positive.")
    if v_min < 0:
        raise ValueError("Lower velocity cutoff `v_min` must be non-negative.")
    if v_max <= v_min:
        raise ValueError("Velocity cutoffs must satisfy `v_max > v_min`.")

    if v_min == 0.0 and np.isinf(v_max):
        return _normalize_exponential_ejecta(E_ej=E_ej, M_ej=M_ej)

    target = E_ej / M_ej

    def moment(v_e: float, order: int) -> float:
        x0 = v_min / v_e
        x1 = v_max / v_e if np.isfinite(v_max) else np.inf

        if order == 2:
            p0 = x0**2 + 2.0 * x0 + 2.0
            term0 = np.exp(-x0) * p0
            term1 = 0.0 if np.isinf(x1) else np.exp(-x1) * (x1**2 + 2.0 * x1 + 2.0)
            return v_e**3 * (term0 - term1)

        if order == 4:
            p0 = x0**4 + 4.0 * x0**3 + 12.0 * x0**2 + 24.0 * x0 + 24.0
            term0 = np.exp(-x0) * p0
            term1 = 0.0 if np.isinf(x1) else np.exp(-x1) * (x1**4 + 4.0 * x1**3 + 12.0 * x1**2 + 24.0 * x1 + 24.0)
            return v_e**5 * (term0 - term1)

        raise ValueError("Only moments of order 2 and 4 are supported.")

    def residual(log_v_e: float) -> float:
        v_e = np.exp(log_v_e)
        I2 = moment(v_e, 2)
        I4 = moment(v_e, 4)
        return 0.5 * I4 / I2 - target

    # The characteristic velocity is a sensible center for the search.
    v_char = np.sqrt(2.0 * E_ej / M_ej)
    grid = np.geomspace(v_char * 1.0e-8, v_char * 1.0e8, 256)
    values = np.array([residual(np.log(v)) for v in grid])

    bracket = None
    for left, right, f_left, f_right in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
        if f_left == 0.0:
            bracket = (left, left)
            break
        if np.sign(f_left) != np.sign(f_right):
            bracket = (left, right)
            break

    if bracket is None:
        raise RuntimeError(
            "Could not bracket the truncated exponential velocity scale. "
            "The requested E/M may be incompatible with the velocity cutoffs."
        )

    if bracket[0] == bracket[1]:
        v_e = bracket[0]
    else:
        v_e = np.exp(brentq(residual, np.log(bracket[0]), np.log(bracket[1])))

    I2 = moment(v_e, 2)
    K = M_ej / (4.0 * np.pi * I2)

    return v_e, K


def normalize_exponential_ejecta(
    E_ej: "_UnitBearingArrayLike",
    M_ej: "_UnitBearingArrayLike",
) -> tuple[u.Quantity, u.Quantity]:
    r"""
    Compute the scale velocity and normalization for an exponential ejecta profile.

    This function normalizes the homologous exponential ejecta kernel

    .. math::

        G(v) = K \exp(-v/v_e),

    where the physical density is

    .. math::

        \rho_{\rm ej}(r,t) = t^{-3}G(r/t).

    The returned parameters are chosen so that the profile integrates to the
    supplied ejecta mass and kinetic energy.

    Parameters
    ----------
    E_ej : astropy.units.Quantity or array-like
        Total kinetic energy of the ejecta. Unit-bearing inputs are converted to
        erg; unit-free inputs are interpreted as erg.
    M_ej : astropy.units.Quantity or array-like
        Total ejecta mass. Unit-bearing inputs are converted to g; unit-free
        inputs are interpreted as g.

    Returns
    -------
    v_e : astropy.units.Quantity
        Exponential velocity scale in cm/s.
    K : astropy.units.Quantity
        Homologous ejecta-kernel normalization in
        :math:`\mathrm{g\,s^3\,cm^{-3}}`.
    """
    E_ej_cgs = ensure_in_units(E_ej, u.erg)
    M_ej_cgs = ensure_in_units(M_ej, u.g)

    v_e_cgs, K_cgs = _normalize_exponential_ejecta(
        E_ej=E_ej_cgs,
        M_ej=M_ej_cgs,
    )

    v_e = v_e_cgs * (u.cm / u.s)
    K = K_cgs * (u.g * u.s**3 / u.cm**3)

    return v_e, K


def normalize_truncated_exponential_ejecta(
    E_ej: "_UnitBearingScalarLike",
    M_ej: "_UnitBearingScalarLike",
    v_min: "_UnitBearingScalarLike" = 0.0 * u.cm / u.s,
    v_max: "_UnitBearingScalarLike" = np.inf * u.cm / u.s,
) -> tuple[u.Quantity, u.Quantity]:
    r"""
    Compute the normalization for a velocity-truncated exponential ejecta profile.

    This function normalizes the homologous ejecta kernel

    .. math::

        G(v) =
        \begin{cases}
            K\exp(-v/v_e), & v_{\min} \le v \le v_{\max},\\
            0, & \mathrm{otherwise}.
        \end{cases}

    The scale velocity :math:`v_e` is solved so that the truncated profile has
    the requested kinetic energy per unit mass. The mass constraint then fixes
    :math:`K`.

    Parameters
    ----------
    E_ej : astropy.units.Quantity or float
        Total ejecta kinetic energy. Unit-bearing inputs are converted to erg;
        unit-free inputs are interpreted as erg.
    M_ej : astropy.units.Quantity or float
        Total ejecta mass. Unit-bearing inputs are converted to g; unit-free
        inputs are interpreted as g.
    v_min : astropy.units.Quantity or float, optional
        Lower velocity cutoff. Unit-bearing inputs are converted to cm/s;
        unit-free inputs are interpreted as cm/s. Default is ``0 cm/s``.
    v_max : astropy.units.Quantity or float, optional
        Upper velocity cutoff. Unit-bearing inputs are converted to cm/s;
        unit-free inputs are interpreted as cm/s. Default is infinity.

    Returns
    -------
    v_e : astropy.units.Quantity
        Exponential velocity scale in cm/s.
    K : astropy.units.Quantity
        Homologous ejecta-kernel normalization in
        :math:`\mathrm{g\,s^3\,cm^{-3}}`.
    """
    E_ej_cgs = float(ensure_in_units(E_ej, u.erg))
    M_ej_cgs = float(ensure_in_units(M_ej, u.g))
    v_min_cgs = float(ensure_in_units(v_min, u.cm / u.s))
    v_max_cgs = float(ensure_in_units(v_max, u.cm / u.s))

    v_e_cgs, K_cgs = _normalize_truncated_exponential_ejecta(
        E_ej=E_ej_cgs,
        M_ej=M_ej_cgs,
        v_min=v_min_cgs,
        v_max=v_max_cgs,
    )

    v_e = v_e_cgs * (u.cm / u.s)
    K = K_cgs * (u.g * u.s**3 / u.cm**3)

    return v_e, K


def normalize_bpl_ejecta(
    E_ej: "_UnitBearingArrayLike",
    M_ej: "_UnitBearingArrayLike",
    n: float = 10,
    delta: float = 0,
) -> tuple[u.Quantity, u.Quantity]:
    r"""
    Compute the transition velocity and normalization constant for a Chevalier-style ejecta profile.

    Returns the transition velocity :math:`v_t` and inner-branch normalization :math:`K` of the
    broken power-law ejecta density profile of
    :footcite:t:`chevalierSelfsimilarSolutionsInteraction1982`, chosen so that the profile
    integrates to the supplied ejecta mass and kinetic energy.

    Parameters
    ----------
    E_ej: astropy.units.Quantity or ~numpy.ndarray
        The total kinetic energy of the ejecta. If units are specified, then they will be taken into
        account. Otherwise, CGS units are assumed.
    M_ej: astropy.units.Quantity or ~numpy.ndarray
        The total mass of the ejecta. If units are specified, then they will be taken into
        account. Otherwise, CGS units are assumed.
    n: float
        The outer ejecta density profile power-law index. By default, this is set to ``10``.
    delta: float, optional
        The inner ejecta density profile power-law index. By default, this is set to ``0``.

    Returns
    -------
    v_t : astropy.units.Quantity
        Transition velocity between the inner and outer ejecta profiles, in cm/s.
    K : astropy.units.Quantity
        Inner-branch normalization, in
        :math:`\mathrm{g\,cm^{\delta-3}\,s^{3-\delta}}`, such that
        :math:`G(v) = K v^{-\delta}` for :math:`v < v_t`.

    Notes
    -----
    As described in :footcite:t:`chevalierSelfsimilarSolutionsInteraction1982`, the ejecta velocity profile
    is well described by a broken power-law. During homologous expansion, :math:`r = vt` implies that the density
    of the ejecta is likewise a broken power-law in velocity space:

    .. math::

        \rho(r,t) = Kt^{-3} \begin{cases}
            v^{-\delta}, & v < v_t \\
            v_t^{n-\delta} v^{-n}, & v \geq v_t,
        \end{cases}

    where :math:`v_t` is the transition velocity between the inner and outer ejecta profiles. The total mass of
    the ejecta is given by :math:`M_{\rm ej}` and must be conserved:

    .. math::

        M_{\rm ej} = \int_0^{\infty} 4\pi r^2 \rho(r,t) dr = 4\pi K v_t^{3-\delta} \frac{n-\delta}{(3-\delta)(n-3)}.

    Similarly, the total kinetic energy of the ejecta is given by :math:`E_{\rm ej}` and must also be conserved:

    .. math::

        E_{\rm ej} = \int_0^{\infty} \frac{1}{2} 4\pi r^2 \rho(r,t) v^2 dr =
        2\pi K v_t^{5-\delta} \frac{n-\delta}{(5-\delta)(n-5)}.

    In terms of the energy per unit mass, :math:`E_{\rm ej}/M_{\rm ej}`, these two equations can be combined to
    solve for the transition velocity:

    .. math::

        v_t^2 = \frac{2(5-\delta)(n-5)}{(3-\delta)(n-3)} \frac{E_{\rm ej}}{M_{\rm ej}}.

    Finally, substituting this back into the mass equation allows us to solve for the normalization constant
    :math:`K`:

    .. math::

        K = \frac{1}{4\pi} \left(\frac{(3-\delta)(n-3)}{(n-\delta)}\right) \frac{M_{\rm ej}}{v_t^{3-\delta}}.

    References
    ----------
    .. footbibliography::
    """
    # Ensure that ``n`` and ``delta`` are valid values for convergence. This requires that delta < 3 and n > 5.
    if delta >= 3:
        raise ValueError("The inner ejecta density profile index `delta` must be less than 3 for convergence.")
    if n <= 5:
        raise ValueError("The outer ejecta density profile index `n` must be greater than 5 for convergence.")

    # Convert any unit-bearing inputs to CGS for internal calculations
    E_ej_cgs = ensure_in_units(E_ej, "erg")
    M_ej_cgs = ensure_in_units(M_ej, "g")

    # Call the optimized internal function for computation
    v_t_cgs, K_cgs = _normalize_BPL_ejecta(
        E_ej=E_ej_cgs,
        M_ej=M_ej_cgs,
        n=n,
        delta=delta,
    )

    # Attach units to the outputs. For v_t, CGS velocity is cm/s. For K, the units are g * cm^(2*delta-3)
    # * s^(3-delta).
    v_t = v_t_cgs * (u.cm / u.s)
    K = K_cgs * (u.g * u.cm ** (delta - 3) * u.s ** (3 - delta))

    return v_t, K


def normalize_truncated_bpl_ejecta(
    E_ej: "_UnitBearingScalarLike",
    M_ej: "_UnitBearingScalarLike",
    n: float = 10,
    delta: float = 0,
    v_max: "_UnitBearingScalarLike" = np.inf * u.cm / u.s,
) -> tuple[u.Quantity, u.Quantity]:
    r"""
    Compute the normalization for an upper-truncated broken-power-law ejecta profile.

    This function normalizes the homologous ejecta kernel

    .. math::

        G(v) =
        \begin{cases}
            K v^{-\delta}, & 0 \le v < v_t,\\
            K v_t^{n-\delta}v^{-n}, & v_t \le v \le v_{\max},\\
            0, & v > v_{\max},
        \end{cases}

    where the physical ejecta density is

    .. math::

        \rho_{\rm ej}(r,t) = t^{-3}G(r/t).

    The transition velocity :math:`v_t` is chosen so that the truncated profile
    has the requested kinetic energy per unit mass. Once :math:`v_t` is known,
    the mass constraint fixes the inner-branch normalization :math:`K`.

    Parameters
    ----------
    E_ej : astropy.units.Quantity or float
        Total ejecta kinetic energy. Unit-bearing inputs are converted to erg;
        unit-free inputs are interpreted as erg.
    M_ej : astropy.units.Quantity or float
        Total ejecta mass. Unit-bearing inputs are converted to g; unit-free
        inputs are interpreted as g.
    n : float, optional
        Outer ejecta density power-law index. Must satisfy :math:`n > 5`.
        Default is ``10``.
    delta : float, optional
        Inner ejecta density power-law index. Must satisfy :math:`\delta < 3`.
        Default is ``0``.
    v_max : astropy.units.Quantity or float, optional
        Maximum ejecta velocity. Unit-bearing inputs are converted to cm/s;
        unit-free inputs are interpreted as cm/s. Default is infinity, which
        recovers the untruncated BPL normalization.

    Returns
    -------
    v_t : astropy.units.Quantity
        Transition velocity between the inner and outer ejecta profiles in cm/s.
    K : astropy.units.Quantity
        Inner-branch kernel normalization. For the inner branch
        :math:`G(v)=Kv^{-\delta}`, the units are

        .. math::

            \mathrm{g\,cm^{\delta - 3}\,s^{3-\delta}}.
    """
    E_ej_cgs = float(ensure_in_units(E_ej, u.erg))
    M_ej_cgs = float(ensure_in_units(M_ej, u.g))
    v_max_cgs = float(ensure_in_units(v_max, u.cm / u.s))

    v_t_cgs, K_cgs = _normalize_truncated_bpl_ejecta(
        E_ej=E_ej_cgs,
        M_ej=M_ej_cgs,
        n=n,
        delta=delta,
        v_max=v_max_cgs,
    )

    v_t = v_t_cgs * (u.cm / u.s)
    K_units = u.g * u.cm ** (delta - 3) * u.s ** (3 - delta)
    K = K_cgs * K_units

    return v_t, K


def get_bpl_ejecta_profile(
    E_ej: "_UnitBearingScalarLike",
    M_ej: "_UnitBearingScalarLike",
    n: float = 10,
    delta: float = 0,
) -> Callable[["_ArrayLike", "_ArrayLike"], "_ArrayLike"]:
    r"""
    Generate a homologous broken-power-law ejecta density profile.

    This function constructs a callable ejecta density profile

    .. math::

        \rho_{\rm ej}(r,t) = t^{-3} G(r/t),

    where :math:`G(v)` is a broken power law normalized to the supplied ejecta
    mass and kinetic energy. The returned callable accepts radius and time in
    CGS units and returns density in :math:`\mathrm{g\,cm^{-3}}`.

    The velocity-space profile is

    .. math::

        G(v)
        =
        \begin{cases}
            K v^{-\delta}, & v < v_t, \\
            K v_t^{n-\delta} v^{-n}, & v \geq v_t,
        \end{cases}

    where :math:`v_t` and :math:`K` are computed by
    :func:`normalize_bpl_ejecta`.

    Parameters
    ----------
    E_ej : astropy.units.Quantity or float
        Total kinetic energy of the ejecta. Unit-bearing inputs are converted to
        erg; unit-free inputs are interpreted as erg.
    M_ej : astropy.units.Quantity or float
        Total ejecta mass. Unit-bearing inputs are converted to g; unit-free
        inputs are interpreted as g.
    n : float, optional
        Outer ejecta density power-law index. This must satisfy :math:`n > 5`
        for finite kinetic energy. Default is ``10``.
    delta : float, optional
        Inner ejecta density power-law index. This must satisfy
        :math:`\delta < 3` for finite mass. Default is ``0``.

    Returns
    -------
    rho_ej : callable
        Callable ``rho_ej(r, t)`` returning the ejecta density in
        :math:`\mathrm{g\,cm^{-3}}`. The inputs ``r`` and ``t`` should be in cm
        and s, respectively, and may be scalars or broadcast-compatible arrays.

    Notes
    -----
    This helper returns a low-level CGS callable suitable for passing directly
    into numerical shock engines. It does not attach units inside the returned
    function, so it remains inexpensive to evaluate inside ODE kernels.
    """
    # Normalize the ejecta so that we have them on deck for use in
    # the function. We also convert them to CGS units here for internal consistency, since the returned function
    # is designed to be unit-free and in CGS.
    v_t, K = normalize_bpl_ejecta(
        E_ej=E_ej,
        M_ej=M_ej,
        n=n,
        delta=delta,
    )

    v_t_cgs = v_t.to_value(u.cm / u.s)
    K_cgs = K.to_value(u.g * u.cm ** (delta - 3) * u.s ** (3 - delta))

    # Now define the density function.
    def _rho_ej(r: "_ArrayLike", t: "_ArrayLike") -> "_ArrayLike":
        v = np.asarray(r) / np.asarray(t)

        G = np.where(
            v < v_t_cgs,
            K_cgs * v ** (-delta),
            K_cgs * v_t_cgs ** (n - delta) * v ** (-n),
        )

        return G / np.asarray(t) ** 3

    return _rho_ej


def get_truncated_bpl_ejecta_profile(
    E_ej: "_UnitBearingScalarLike",
    M_ej: "_UnitBearingScalarLike",
    n: float = 10,
    delta: float = 0,
    v_max: "_UnitBearingScalarLike" = np.inf * u.cm / u.s,
) -> Callable[["_ArrayLike", "_ArrayLike"], "_ArrayLike"]:
    r"""
    Build an upper-truncated broken-power-law ejecta density profile.

    The returned callable evaluates

    .. math::

        \rho_{\rm ej}(r,t)
        =
        t^{-3}G(r/t),

    where ``G`` is the normalized upper-truncated BPL ejecta kernel.

    Parameters
    ----------
    E_ej : astropy.units.Quantity or float
        Total ejecta kinetic energy.
    M_ej : astropy.units.Quantity or float
        Total ejecta mass.
    n : float, optional
        Outer ejecta density power-law index. Default is ``10``.
    delta : float, optional
        Inner ejecta density power-law index. Default is ``0``.
    v_max : astropy.units.Quantity or float, optional
        Maximum ejecta velocity. Default is infinity.

    Returns
    -------
    rho_ej : callable
        Callable ``rho_ej(r, t)`` returning ejecta density in
        :math:`\mathrm{g\,cm^{-3}}`, with ``r`` in cm and ``t`` in s.
    """
    G_ej = get_truncated_bpl_ejecta_kernel(
        E_ej=E_ej,
        M_ej=M_ej,
        n=n,
        delta=delta,
        v_max=v_max,
    )

    def _rho_ej(r: "_ArrayLike", t: "_ArrayLike") -> "_ArrayLike":
        r_array = np.asarray(r, dtype=float)
        t_array = np.asarray(t, dtype=float)

        rho = G_ej(r_array / t_array) / t_array**3

        return float(rho) if r_array.ndim == 0 and t_array.ndim == 0 else rho

    return _rho_ej


def get_exponential_ejecta_profile(
    E_ej: "_UnitBearingScalarLike",
    M_ej: "_UnitBearingScalarLike",
) -> Callable[["_ArrayLike", "_ArrayLike"], "_ArrayLike"]:
    r"""
    Build an exponential homologous ejecta density profile.

    The returned callable evaluates

    .. math::

        \rho_{\rm ej}(r,t)
        =
        t^{-3}G(r/t),

    where ``G`` is the normalized exponential ejecta kernel.

    Parameters
    ----------
    E_ej : astropy.units.Quantity or float
        Total ejecta kinetic energy.
    M_ej : astropy.units.Quantity or float
        Total ejecta mass.

    Returns
    -------
    rho_ej : callable
        Callable ``rho_ej(r, t)`` returning density in
        :math:`\mathrm{g\,cm^{-3}}`, with ``r`` in cm and ``t`` in s.
    """
    G_ej = get_exponential_ejecta_kernel(E_ej=E_ej, M_ej=M_ej)

    def _rho_ej(r: "_ArrayLike", t: "_ArrayLike") -> "_ArrayLike":
        r_array = np.asarray(r, dtype=float)
        t_array = np.asarray(t, dtype=float)

        rho = G_ej(r_array / t_array) / t_array**3

        return float(rho) if r_array.ndim == 0 and t_array.ndim == 0 else rho

    return _rho_ej


def get_truncated_exponential_ejecta_profile(
    E_ej: "_UnitBearingScalarLike",
    M_ej: "_UnitBearingScalarLike",
    v_min: "_UnitBearingScalarLike" = 0.0 * u.cm / u.s,
    v_max: "_UnitBearingScalarLike" = np.inf * u.cm / u.s,
) -> Callable[["_ArrayLike", "_ArrayLike"], "_ArrayLike"]:
    r"""
    Build a velocity-truncated exponential homologous ejecta density profile.

    The returned callable evaluates

    .. math::

        \rho_{\rm ej}(r,t)
        =
        t^{-3}G(r/t),

    where ``G`` is zero outside the requested velocity interval.

    Parameters
    ----------
    E_ej : astropy.units.Quantity or float
        Total ejecta kinetic energy.
    M_ej : astropy.units.Quantity or float
        Total ejecta mass.
    v_min : astropy.units.Quantity or float, optional
        Lower velocity cutoff.
    v_max : astropy.units.Quantity or float, optional
        Upper velocity cutoff.

    Returns
    -------
    rho_ej : callable
        Callable ``rho_ej(r, t)`` returning density in
        :math:`\mathrm{g\,cm^{-3}}`, with ``r`` in cm and ``t`` in s.
    """
    G_ej = get_truncated_exponential_ejecta_kernel(
        E_ej=E_ej,
        M_ej=M_ej,
        v_min=v_min,
        v_max=v_max,
    )

    def _rho_ej(r: "_ArrayLike", t: "_ArrayLike") -> "_ArrayLike":
        r_array = np.asarray(r, dtype=float)
        t_array = np.asarray(t, dtype=float)

        rho = G_ej(r_array / t_array) / t_array**3

        return float(rho) if r_array.ndim == 0 and t_array.ndim == 0 else rho

    return _rho_ej


def get_truncated_bpl_ejecta_kernel(
    E_ej: "_UnitBearingScalarLike",
    M_ej: "_UnitBearingScalarLike",
    n: float = 10,
    delta: float = 0,
    v_max: "_UnitBearingScalarLike" = np.inf * u.cm / u.s,
) -> Callable[["_ArrayLike"], "_ArrayLike"]:
    r"""
    Build an upper-truncated broken-power-law homologous ejecta kernel ``G(v)``.

    The returned callable evaluates

    .. math::

        G(v) =
        \begin{cases}
            K v^{-\delta}, & 0 \le v < v_t,\\
            K v_t^{n-\delta}v^{-n}, & v_t \le v \le v_{\max},\\
            0, & v > v_{\max},
        \end{cases}

    in CGS units. The input velocity should be supplied in cm/s. The returned
    callable is unit-free for fast evaluation inside ODE right-hand sides.

    Parameters
    ----------
    E_ej : astropy.units.Quantity or float
        Total ejecta kinetic energy.
    M_ej : astropy.units.Quantity or float
        Total ejecta mass.
    n : float, optional
        Outer ejecta density power-law index. Default is ``10``.
    delta : float, optional
        Inner ejecta density power-law index. Default is ``0``.
    v_max : astropy.units.Quantity or float, optional
        Maximum ejecta velocity. Default is infinity.

    Returns
    -------
    G_ej : callable
        Callable ``G_ej(v)`` returning the truncated homologous ejecta kernel in
        CGS units.
    """
    v_t, K_inner = normalize_truncated_bpl_ejecta(
        E_ej=E_ej,
        M_ej=M_ej,
        n=n,
        delta=delta,
        v_max=v_max,
    )

    v_t_cgs = v_t.to_value(u.cm / u.s)
    K_inner_cgs = K_inner.to_value(u.g * u.cm ** (delta - 3) * u.s ** (3 - delta))
    v_max_cgs = float(ensure_in_units(v_max, u.cm / u.s))

    K_outer_cgs = K_inner_cgs * v_t_cgs ** (n - delta)

    def _G_ej(v: "_ArrayLike") -> "_ArrayLike":
        r"""
        Evaluate the upper-truncated BPL homologous ejecta kernel in CGS units.

        Parameters
        ----------
        v : float or array-like
            Homologous ejecta velocity in cm/s.

        Returns
        -------
        G : float or numpy.ndarray
            Velocity-space ejecta density kernel. The physical density is
            recovered as ``rho_ej(r, t) = t**-3 * G_ej(r / t)``.
        """
        v_array = np.asarray(v, dtype=float)

        G_raw = np.where(
            v_array < v_t_cgs,
            K_inner_cgs * v_array ** (-delta),
            K_outer_cgs * v_array ** (-n),
        )
        G = np.where(v_array <= v_max_cgs, G_raw, 0.0)

        return float(G) if v_array.ndim == 0 else G

    return _G_ej


def get_bpl_ejecta_kernel(
    E_ej: "_UnitBearingScalarLike",
    M_ej: "_UnitBearingScalarLike",
    n: float = 10,
    delta: float = 0,
) -> Callable[["_ArrayLike"], "_ArrayLike"]:
    r"""
    Build the homologous broken-power-law ejecta kernel ``G(v)``.

    Returns the time-independent velocity-space density kernel for freely
    expanding supernova ejecta under the homologous approximation
    :math:`\rho_{\rm ej}(r,t) = t^{-3}G(r/t)`.  The kernel is

    .. math::

        G(v) =
        \begin{cases}
            K\,v^{-\delta}, & v < v_t, \\
            K\,v_t^{n-\delta}v^{-n}, & v \ge v_t,
        \end{cases}

    with :math:`v_t` and :math:`K` chosen by :func:`normalize_bpl_ejecta` so
    that the profile integrates to the supplied ejecta mass and kinetic energy.

    Parameters
    ----------
    E_ej : ~astropy.units.Quantity or float
        Total kinetic energy of the ejecta. Unit-bearing inputs are converted to
        erg; unit-free inputs are interpreted as erg.
    M_ej : ~astropy.units.Quantity or float
        Total ejecta mass. Unit-bearing inputs are converted to g; unit-free
        inputs are interpreted as g.
    n : float, optional
        Outer ejecta density power-law index. Must satisfy :math:`n > 5`.
        Default is ``10``.
    delta : float, optional
        Inner ejecta density power-law index. Must satisfy :math:`\delta < 3`.
        Default is ``0``.

    Returns
    -------
    G_ej : callable
        Callable ``G_ej(v)`` returning the kernel in
        :math:`\mathrm{g\,s^3\,cm^{-3}}`, with ``v`` in cm/s. Accepts scalars
        or broadcast-compatible arrays; returns a scalar when the input is
        scalar.

    Raises
    ------
    ValueError
        If ``delta >= 3``, ``n <= 5``, or ``n <= delta``.

    Notes
    -----
    The returned callable is unit-free for efficient evaluation inside ODE
    right-hand sides. Unit handling is performed once at construction time.
    """
    # Validate n > delta for a well-ordered BPL (normalize_bpl_ejecta checks
    # delta < 3 and n > 5 but not the cross-constraint).
    if n <= delta:
        raise ValueError("Outer index `n` must exceed inner index `delta`.")

    # Normalize: fixes v_t and K_inner by enforcing the requested M_ej and E_ej.
    v_t, K_inner = normalize_bpl_ejecta(E_ej=E_ej, M_ej=M_ej, n=n, delta=delta)

    # Extract CGS scalars for use inside the kernel closure.
    v_t_cgs = v_t.to_value(u.cm / u.s)
    K_inner_cgs = K_inner.to_value(u.g * u.cm ** (delta - 3) * u.s ** (3 - delta))

    # The outer branch K_outer * v**(-n) is continuous at v_t when
    # K_outer = K_inner * v_t**(n - delta).
    K_outer_cgs = K_inner_cgs * v_t_cgs ** (n - delta)

    def _G_ej(v: "_ArrayLike") -> "_ArrayLike":
        v_array = np.asarray(v, dtype=float)

        G = np.where(
            v_array < v_t_cgs,
            K_inner_cgs * v_array ** (-delta),
            K_outer_cgs * v_array ** (-n),
        )

        return float(G) if v_array.ndim == 0 else G

    return _G_ej


def get_exponential_ejecta_kernel(
    E_ej: "_UnitBearingScalarLike",
    M_ej: "_UnitBearingScalarLike",
) -> Callable[["_ArrayLike"], "_ArrayLike"]:
    r"""
    Build an exponential homologous ejecta kernel ``G(v)``.

    The returned callable evaluates

    .. math::

        G(v) = K\exp(-v/v_e),

    in CGS units. The input velocity should be supplied in cm/s. The returned
    kernel is unit-free for fast evaluation inside ODE right-hand sides.

    Parameters
    ----------
    E_ej : astropy.units.Quantity or float
        Total ejecta kinetic energy.
    M_ej : astropy.units.Quantity or float
        Total ejecta mass.

    Returns
    -------
    G_ej : callable
        Callable ``G_ej(v)`` returning the homologous ejecta kernel in
        :math:`\mathrm{g\,s^3\,cm^{-3}}`.
    """
    v_e, K = normalize_exponential_ejecta(
        E_ej=E_ej,
        M_ej=M_ej,
    )

    v_e_cgs = v_e.to_value(u.cm / u.s)
    K_cgs = K.to_value(u.g * u.s**3 / u.cm**3)

    def _G_ej(v: "_ArrayLike") -> "_ArrayLike":
        r"""
        Evaluate the exponential homologous ejecta kernel in CGS units.

        Parameters
        ----------
        v : float or array-like
            Homologous ejecta velocity in cm/s.

        Returns
        -------
        G : float or numpy.ndarray
            Velocity-space ejecta density kernel.
        """
        v_array = np.asarray(v, dtype=float)
        G = K_cgs * np.exp(-v_array / v_e_cgs)

        return float(G) if v_array.ndim == 0 else G

    return _G_ej


def get_truncated_exponential_ejecta_kernel(
    E_ej: "_UnitBearingScalarLike",
    M_ej: "_UnitBearingScalarLike",
    v_min: "_UnitBearingScalarLike" = 0.0 * u.cm / u.s,
    v_max: "_UnitBearingScalarLike" = np.inf * u.cm / u.s,
) -> Callable[["_ArrayLike"], "_ArrayLike"]:
    r"""
    Build a velocity-truncated exponential homologous ejecta kernel ``G(v)``.

    The returned callable evaluates

    .. math::

        G(v) =
        \begin{cases}
            K\exp(-v/v_e), & v_{\min} \le v \le v_{\max},\\
            0, & \mathrm{otherwise},
        \end{cases}

    in CGS units.

    Parameters
    ----------
    E_ej : astropy.units.Quantity or float
        Total ejecta kinetic energy.
    M_ej : astropy.units.Quantity or float
        Total ejecta mass.
    v_min : astropy.units.Quantity or float, optional
        Lower velocity cutoff.
    v_max : astropy.units.Quantity or float, optional
        Upper velocity cutoff.

    Returns
    -------
    G_ej : callable
        Callable ``G_ej(v)`` returning the truncated homologous ejecta kernel in
        :math:`\mathrm{g\,s^3\,cm^{-3}}`.
    """
    v_e, K = normalize_truncated_exponential_ejecta(
        E_ej=E_ej,
        M_ej=M_ej,
        v_min=v_min,
        v_max=v_max,
    )

    v_e_cgs = v_e.to_value(u.cm / u.s)
    K_cgs = K.to_value(u.g * u.s**3 / u.cm**3)
    v_min_cgs = float(ensure_in_units(v_min, u.cm / u.s))
    v_max_cgs = float(ensure_in_units(v_max, u.cm / u.s))

    def _G_ej(v: "_ArrayLike") -> "_ArrayLike":
        r"""
        Evaluate the truncated exponential homologous ejecta kernel in CGS units.

        Parameters
        ----------
        v : float or array-like
            Homologous ejecta velocity in cm/s.

        Returns
        -------
        G : float or numpy.ndarray
            Velocity-space ejecta density kernel.
        """
        v_array = np.asarray(v, dtype=float)

        inside = (v_array >= v_min_cgs) & (v_array <= v_max_cgs)
        G_raw = K_cgs * np.exp(-v_array / v_e_cgs)
        G = np.where(inside, G_raw, 0.0)

        return float(G) if v_array.ndim == 0 else G

    return _G_ej


# =============================================== #
# CSM Properties and Generation                   #
# =============================================== #
# These functions provide utilities for constructing simple circumstellar
# medium (CSM) density profiles used by the shock engines.
def _compute_wind_csm_parameters(
    mass_loss_rate: "_ArrayLike",
    wind_velocity: "_ArrayLike",
) -> "_ArrayLike":
    r"""
    Compute the wind-density normalization in CGS units.

    This is the low-level, unit-free implementation used by
    :func:`compute_wind_csm_parameters`. Inputs are assumed to be in CGS units
    and may be scalars or broadcast-compatible array-like objects.

    For a steady, spherically symmetric wind with mass-loss rate
    :math:`\dot{M}` and wind velocity :math:`v_w`, mass conservation gives

    .. math::

        \dot{M}
        =
        4\pi r^2 \rho_{\rm CSM}(r) v_w.

    Therefore,

    .. math::

        \rho_{\rm CSM}(r)
        =
        \frac{\dot{M}}{4\pi v_w} r^{-2}
        =
        A r^{-2},

    where the wind normalization is

    .. math::

        A = \frac{\dot{M}}{4\pi v_w}.

    Parameters
    ----------
    mass_loss_rate : float or ~numpy.ndarray
        Progenitor mass-loss rate :math:`\dot{M}` in g/s.
    wind_velocity : float or ~numpy.ndarray
        Wind velocity :math:`v_w` in cm/s.

    Returns
    -------
    A : float or ~numpy.ndarray
        Wind-density normalization in g/cm. The physical CSM density is
        :math:`\rho_{\rm CSM}(r) = A r^{-2}`.

    Notes
    -----
    This helper performs no unit conversion or validation. The public wrapper
    should handle unit coercion and user-facing checks.
    """
    return mass_loss_rate / (4.0 * np.pi * wind_velocity)


def compute_wind_csm_parameters(
    mass_loss_rate: "_UnitBearingArrayLike",
    wind_velocity: "_UnitBearingArrayLike",
) -> u.Quantity:
    r"""
    Compute the normalization of a steady wind-like CSM density profile.

    A steady, spherically symmetric progenitor wind with mass-loss rate
    :math:`\dot{M}` and wind velocity :math:`v_w` produces the density profile

    .. math::

        \rho_{\rm CSM}(r)
        =
        \frac{\dot{M}}{4\pi r^2 v_w}
        =
        A r^{-2}.

    This function computes the wind-density normalization

    .. math::

        A = \frac{\dot{M}}{4\pi v_w},

    which has units of :math:`\mathrm{g\,cm^{-1}}`.

    Parameters
    ----------
    mass_loss_rate : astropy.units.Quantity or array-like
        Progenitor mass-loss rate. Unit-bearing inputs are converted to g/s.
        Unit-free inputs are interpreted as g/s.
    wind_velocity : astropy.units.Quantity or array-like
        Wind velocity. Unit-bearing inputs are converted to cm/s. Unit-free
        inputs are interpreted as cm/s.

    Returns
    -------
    A : astropy.units.Quantity
        Wind-density normalization in g/cm. The CSM density profile is recovered
        as :math:`\rho_{\rm CSM}(r) = A r^{-2}`.

    Raises
    ------
    ValueError
        If ``wind_velocity`` is non-positive.
    """
    mass_loss_rate_cgs = ensure_in_units(mass_loss_rate, u.g / u.s)
    wind_velocity_cgs = ensure_in_units(wind_velocity, u.cm / u.s)

    if np.any(wind_velocity_cgs <= 0):
        raise ValueError("Wind velocity `wind_velocity` must be positive.")

    A_cgs = _compute_wind_csm_parameters(
        mass_loss_rate=mass_loss_rate_cgs,
        wind_velocity=wind_velocity_cgs,
    )

    return A_cgs * (u.g / u.cm)


def get_wind_csm_density_func(
    mass_loss_rate: u.Quantity,
    wind_velocity: u.Quantity,
) -> Callable[["_ArrayLike", "_ArrayLike | None"], "_ArrayLike"]:
    r"""
    Build a steady wind-like CSM density profile callable.

    This helper returns a low-level CGS callable for the density profile of a
    steady, spherically symmetric circumstellar wind. For a constant mass-loss
    rate and constant wind speed, mass conservation gives

    .. math::

        \rho_{\rm CSM}(r)
        =
        \frac{\dot{M}}{4\pi r^2 v_w}
        =
        A r^{-2},

    where

    .. math::

        A = \frac{\dot{M}}{4\pi v_w}.

    The returned callable accepts an optional time argument, ``t``, for API
    compatibility with shock-engine source functions of the form ``rho(r, t)``.
    Since the wind profile is stationary, this argument is ignored.

    Parameters
    ----------
    mass_loss_rate : astropy.units.Quantity
        Progenitor mass-loss rate. This is converted to g/s when the density
        function is constructed.
    wind_velocity : astropy.units.Quantity
        Wind velocity. This is converted to cm/s when the density function is
        constructed.

    Returns
    -------
    rho_csm : callable
        Callable ``rho_csm(radius, t=None)`` returning the CSM density in
        :math:`\mathrm{g\,cm^{-3}}`. The input ``radius`` should be in cm and may
        be a scalar or array-like object. The optional ``t`` argument is ignored.

    Notes
    -----
    The returned callable does not accept or return
    :class:`astropy.units.Quantity` objects. Unit handling is performed once
    when this factory is called, so that the returned function remains cheap to
    evaluate inside ODE kernels.
    """
    A_cgs = compute_wind_csm_parameters(
        mass_loss_rate=mass_loss_rate,
        wind_velocity=wind_velocity,
    ).to_value(u.g / u.cm)

    def _rho_csm(
        radius: "_ArrayLike",
        _t: "_ArrayLike | None" = None,
    ) -> "_ArrayLike":
        radius_array = np.asarray(radius, dtype=float)
        rho = A_cgs * radius_array**-2

        return float(rho) if radius_array.ndim == 0 else rho

    return _rho_csm


def get_uniform_csm_density_func(
    density: "_UnitBearingScalarLike",
) -> Callable[["_ArrayLike", "_ArrayLike | None"], "_ArrayLike"]:
    r"""
    Build a uniform CSM density profile callable.

    The returned function evaluates

    .. math::

        \rho_{\rm CSM}(r,t) = \rho_0,

    in CGS units. The optional ``t`` argument is accepted for compatibility with
    two-argument shock-engine source functions and is ignored.

    Parameters
    ----------
    density : astropy.units.Quantity or float
        Uniform CSM density. Unit-bearing inputs are converted to
        :math:`\mathrm{g\,cm^{-3}}`; unit-free inputs are interpreted as
        :math:`\mathrm{g\,cm^{-3}}`.

    Returns
    -------
    rho_csm : callable
        Callable ``rho_csm(radius, t=None)`` returning density in
        :math:`\mathrm{g\,cm^{-3}}`.
    """
    density_cgs = ensure_in_units(density, u.g / u.cm**3)

    if np.any(density_cgs < 0):
        raise ValueError("Uniform CSM density `density` must be non-negative.")

    density_cgs = float(density_cgs)

    def _rho_csm(
        radius: "_ArrayLike",
        _t: "_ArrayLike | None" = None,
    ) -> "_ArrayLike":
        radius_array = np.asarray(radius)

        if radius_array.ndim == 0:
            return density_cgs

        return np.full_like(radius_array, density_cgs, dtype=float)

    return _rho_csm


def get_truncated_wind_csm_density_func(
    mass_loss_rate: "_UnitBearingScalarLike",
    wind_velocity: "_UnitBearingScalarLike",
    r_max: "_UnitBearingScalarLike",
    density_floor: "_UnitBearingScalarLike" = 0.0 * u.g / u.cm**3,
) -> Callable[["_ArrayLike", "_ArrayLike | None"], "_ArrayLike"]:
    r"""
    Build an outer-truncated wind-like CSM density profile callable.

    This helper returns a low-level CGS callable for a steady wind profile that
    transitions to a constant density floor outside a maximum radius,

    .. math::

        \rho_{\rm CSM}(r)
        =
        \begin{cases}
            A r^{-2}, & r \le r_{\max},\\
            \rho_{\rm floor}, & r > r_{\max},
        \end{cases}

    where

    .. math::

        A = \frac{\dot{M}}{4\pi v_w}.

    This is useful for representing a finite wind region embedded in a lower-
    density ambient medium. The returned callable accepts an optional time
    argument, ``_t``, for compatibility with shock-engine source functions of
    the form ``rho(r, t)``. Since the profile is stationary, this argument is
    ignored.

    Parameters
    ----------
    mass_loss_rate : astropy.units.Quantity or float
        Progenitor mass-loss rate. Unit-bearing inputs are converted to g/s;
        unit-free inputs are interpreted as g/s.
    wind_velocity : astropy.units.Quantity or float
        Wind velocity. Unit-bearing inputs are converted to cm/s; unit-free
        inputs are interpreted as cm/s.
    r_max : astropy.units.Quantity or float
        Outer edge of the wind region. Unit-bearing inputs are converted to cm;
        unit-free inputs are interpreted as cm.
    density_floor : astropy.units.Quantity or float, optional
        Density returned outside ``r_max``. Unit-bearing inputs are converted to
        :math:`\mathrm{g\,cm^{-3}}`; unit-free inputs are interpreted as
        :math:`\mathrm{g\,cm^{-3}}`. Default is zero.

    Returns
    -------
    rho_csm : callable
        Callable ``rho_csm(radius, _t=None)`` returning the CSM density in
        :math:`\mathrm{g\,cm^{-3}}`. The input ``radius`` should be in cm and may
        be a scalar or array-like object.

    Raises
    ------
    ValueError
        If ``wind_velocity`` is non-positive, ``r_max`` is non-positive, or
        ``density_floor`` is negative.

    Notes
    -----
    The returned callable does not accept or return
    :class:`astropy.units.Quantity` objects. Unit handling is performed once
    when this factory is called, so that the returned function remains cheap to
    evaluate inside ODE kernels.
    """
    A_cgs = compute_wind_csm_parameters(
        mass_loss_rate=mass_loss_rate,
        wind_velocity=wind_velocity,
    ).to_value(u.g / u.cm)

    r_max_cgs = ensure_in_units(r_max, u.cm)
    density_floor_cgs = ensure_in_units(density_floor, u.g / u.cm**3)

    if np.any(r_max_cgs <= 0):
        raise ValueError("Maximum wind radius `r_max` must be positive.")
    if np.any(density_floor_cgs < 0):
        raise ValueError("Density floor `density_floor` must be non-negative.")

    r_max_cgs = float(r_max_cgs)
    density_floor_cgs = float(density_floor_cgs)

    def _rho_csm(
        radius: "_ArrayLike",
        _t: "_ArrayLike | None" = None,
    ) -> "_ArrayLike":
        radius_array = np.asarray(radius, dtype=float)

        rho_wind = A_cgs * radius_array**-2
        rho = np.where(radius_array <= r_max_cgs, rho_wind, density_floor_cgs)

        return float(rho) if radius_array.ndim == 0 else rho

    return _rho_csm


def get_power_law_csm_density_func(
    density_ref: "_UnitBearingScalarLike",
    radius_ref: "_UnitBearingScalarLike",
    slope: float,
) -> Callable[["_ArrayLike", "_ArrayLike | None"], "_ArrayLike"]:
    r"""
    Build a power-law CSM density profile callable.

    The returned function evaluates

    .. math::

        \rho_{\rm CSM}(r)
        =
        \rho_{\rm ref}
        \left(\frac{r}{r_{\rm ref}}\right)^{-s}.

    Parameters
    ----------
    density_ref : astropy.units.Quantity or float
        Reference density. Unit-bearing inputs are converted to
        :math:`\mathrm{g\,cm^{-3}}`; unit-free inputs are interpreted as
        :math:`\mathrm{g\,cm^{-3}}`.
    radius_ref : astropy.units.Quantity or float
        Reference radius. Unit-bearing inputs are converted to cm; unit-free
        inputs are interpreted as cm.
    slope : float
        Power-law slope.

    Returns
    -------
    rho_csm : callable
        Callable ``rho_csm(radius, _t=None)`` returning density in
        :math:`\mathrm{g\,cm^{-3}}`.
    """
    density_ref_cgs = ensure_in_units(density_ref, u.g / u.cm**3)
    radius_ref_cgs = ensure_in_units(radius_ref, u.cm)

    if np.any(density_ref_cgs < 0):
        raise ValueError("Reference density `density_ref` must be non-negative.")
    if np.any(radius_ref_cgs <= 0):
        raise ValueError("Reference radius `radius_ref` must be positive.")

    density_ref_cgs = float(density_ref_cgs)
    radius_ref_cgs = float(radius_ref_cgs)

    def _rho_csm(
        radius: "_ArrayLike",
        _t: "_ArrayLike | None" = None,
    ) -> "_ArrayLike":
        radius_array = np.asarray(radius, dtype=float)
        rho = density_ref_cgs * (radius_array / radius_ref_cgs) ** (-slope)

        return float(rho) if radius_array.ndim == 0 else rho

    return _rho_csm


def get_wind_with_floor_csm_density_func(
    mass_loss_rate: "_UnitBearingScalarLike",
    wind_velocity: "_UnitBearingScalarLike",
    density_floor: "_UnitBearingScalarLike",
) -> Callable[["_ArrayLike", "_ArrayLike | None"], "_ArrayLike"]:
    r"""
    Build a wind-like CSM density profile with a constant density floor.

    The returned function evaluates

    .. math::

        \rho_{\rm CSM}(r) = A r^{-2} + \rho_{\rm floor},

    where :math:`A = \dot{M}/(4\pi v_w)`.

    Parameters
    ----------
    mass_loss_rate : astropy.units.Quantity or float
        Progenitor mass-loss rate. Unit-bearing inputs are converted to g/s.
    wind_velocity : astropy.units.Quantity or float
        Wind velocity. Unit-bearing inputs are converted to cm/s.
    density_floor : astropy.units.Quantity or float
        Constant background density. Unit-bearing inputs are converted to
        :math:`\mathrm{g\,cm^{-3}}`.

    Returns
    -------
    rho_csm : callable
        Callable ``rho_csm(radius, _t=None)`` returning density in
        :math:`\mathrm{g\,cm^{-3}}`.
    """
    A_cgs = compute_wind_csm_parameters(
        mass_loss_rate=mass_loss_rate,
        wind_velocity=wind_velocity,
    ).to_value(u.g / u.cm)

    density_floor_cgs = ensure_in_units(density_floor, u.g / u.cm**3)

    if np.any(density_floor_cgs < 0):
        raise ValueError("Density floor `density_floor` must be non-negative.")

    density_floor_cgs = float(density_floor_cgs)

    def _rho_csm(
        radius: "_ArrayLike",
        _t: "_ArrayLike | None" = None,
    ) -> "_ArrayLike":
        radius_array = np.asarray(radius, dtype=float)
        rho = A_cgs * radius_array**-2 + density_floor_cgs

        return float(rho) if radius_array.ndim == 0 else rho

    return _rho_csm


def get_shell_csm_density_func(
    r_inner: "_UnitBearingScalarLike",
    r_outer: "_UnitBearingScalarLike",
    shell_density: "_UnitBearingScalarLike",
    density_floor: "_UnitBearingScalarLike" = 0.0 * u.g / u.cm**3,
) -> Callable[["_ArrayLike", "_ArrayLike | None"], "_ArrayLike"]:
    r"""
    Build a top-hat shell CSM density profile callable.

    The returned function evaluates

    .. math::

        \rho_{\rm CSM}(r)
        =
        \begin{cases}
            \rho_{\rm shell}, & r_{\rm in} \le r \le r_{\rm out},\\
            \rho_{\rm floor}, & \mathrm{otherwise}.
        \end{cases}

    Parameters
    ----------
    r_inner : astropy.units.Quantity or float
        Inner shell radius. Unit-bearing inputs are converted to cm.
    r_outer : astropy.units.Quantity or float
        Outer shell radius. Unit-bearing inputs are converted to cm.
    shell_density : astropy.units.Quantity or float
        Density inside the shell. Unit-bearing inputs are converted to
        :math:`\mathrm{g\,cm^{-3}}`.
    density_floor : astropy.units.Quantity or float, optional
        Density outside the shell. Default is zero.

    Returns
    -------
    rho_csm : callable
        Callable ``rho_csm(radius, _t=None)`` returning density in
        :math:`\mathrm{g\,cm^{-3}}`.
    """
    r_inner_cgs = ensure_in_units(r_inner, u.cm)
    r_outer_cgs = ensure_in_units(r_outer, u.cm)
    shell_density_cgs = ensure_in_units(shell_density, u.g / u.cm**3)
    density_floor_cgs = ensure_in_units(density_floor, u.g / u.cm**3)

    if np.any(r_inner_cgs < 0):
        raise ValueError("Inner shell radius `r_inner` must be non-negative.")
    if np.any(r_outer_cgs <= r_inner_cgs):
        raise ValueError("Shell radii must satisfy `r_outer > r_inner`.")
    if np.any(shell_density_cgs < 0):
        raise ValueError("Shell density `shell_density` must be non-negative.")
    if np.any(density_floor_cgs < 0):
        raise ValueError("Density floor `density_floor` must be non-negative.")

    r_inner_cgs = float(r_inner_cgs)
    r_outer_cgs = float(r_outer_cgs)
    shell_density_cgs = float(shell_density_cgs)
    density_floor_cgs = float(density_floor_cgs)

    def _rho_csm(
        radius: "_ArrayLike",
        _t: "_ArrayLike | None" = None,
    ) -> "_ArrayLike":
        radius_array = np.asarray(radius, dtype=float)
        inside = (radius_array >= r_inner_cgs) & (radius_array <= r_outer_cgs)
        rho = np.where(inside, shell_density_cgs, density_floor_cgs)

        return float(rho) if radius_array.ndim == 0 else rho

    return _rho_csm


def get_gaussian_shell_csm_density_func(
    background_density: "_UnitBearingScalarLike",
    shell_density: "_UnitBearingScalarLike",
    shell_radius: "_UnitBearingScalarLike",
    shell_width: "_UnitBearingScalarLike",
) -> Callable[["_ArrayLike", "_ArrayLike | None"], "_ArrayLike"]:
    r"""
    Build a smooth Gaussian-shell CSM density profile callable.

    The returned function evaluates

    .. math::

        \rho_{\rm CSM}(r)
        =
        \rho_{\rm bg}
        +
        \rho_{\rm shell}
        \exp\left[
            -\frac{1}{2}
            \left(\frac{r - R_{\rm shell}}{\sigma_{\rm shell}}\right)^2
        \right].

    Parameters
    ----------
    background_density : astropy.units.Quantity or float
        Background CSM density.
    shell_density : astropy.units.Quantity or float
        Peak shell density above the background.
    shell_radius : astropy.units.Quantity or float
        Shell center radius.
    shell_width : astropy.units.Quantity or float
        Gaussian shell width.

    Returns
    -------
    rho_csm : callable
        Callable ``rho_csm(radius, _t=None)`` returning density in
        :math:`\mathrm{g\,cm^{-3}}`.
    """
    background_cgs = ensure_in_units(background_density, u.g / u.cm**3)
    shell_density_cgs = ensure_in_units(shell_density, u.g / u.cm**3)
    shell_radius_cgs = ensure_in_units(shell_radius, u.cm)
    shell_width_cgs = ensure_in_units(shell_width, u.cm)

    if np.any(background_cgs < 0):
        raise ValueError("Background density `background_density` must be non-negative.")
    if np.any(shell_density_cgs < 0):
        raise ValueError("Shell density `shell_density` must be non-negative.")
    if np.any(shell_radius_cgs < 0):
        raise ValueError("Shell radius `shell_radius` must be non-negative.")
    if np.any(shell_width_cgs <= 0):
        raise ValueError("Shell width `shell_width` must be positive.")

    background_cgs = float(background_cgs)
    shell_density_cgs = float(shell_density_cgs)
    shell_radius_cgs = float(shell_radius_cgs)
    shell_width_cgs = float(shell_width_cgs)

    def _rho_csm(
        radius: "_ArrayLike",
        _t: "_ArrayLike | None" = None,
    ) -> "_ArrayLike":
        radius_array = np.asarray(radius, dtype=float)
        x = (radius_array - shell_radius_cgs) / shell_width_cgs
        rho = background_cgs + shell_density_cgs * np.exp(-0.5 * x**2)

        return float(rho) if radius_array.ndim == 0 else rho

    return _rho_csm


def get_broken_power_law_csm_density_func(
    density_break: "_UnitBearingScalarLike",
    radius_break: "_UnitBearingScalarLike",
    slope_inner: float,
    slope_outer: float,
) -> Callable[["_ArrayLike", "_ArrayLike | None"], "_ArrayLike"]:
    r"""
    Build a continuous broken-power-law CSM density profile callable.

    The returned function evaluates

    .. math::

        \rho_{\rm CSM}(r)
        =
        \rho_b
        \begin{cases}
            (r/r_b)^{-s_{\rm in}}, & r < r_b,\\
            (r/r_b)^{-s_{\rm out}}, & r \ge r_b.
        \end{cases}

    Parameters
    ----------
    density_break : astropy.units.Quantity or float
        Density at the break radius.
    radius_break : astropy.units.Quantity or float
        Break radius.
    slope_inner : float
        Inner power-law slope.
    slope_outer : float
        Outer power-law slope.

    Returns
    -------
    rho_csm : callable
        Callable ``rho_csm(radius, _t=None)`` returning density in
        :math:`\mathrm{g\,cm^{-3}}`.
    """
    density_break_cgs = ensure_in_units(density_break, u.g / u.cm**3)
    radius_break_cgs = ensure_in_units(radius_break, u.cm)

    if np.any(density_break_cgs < 0):
        raise ValueError("Break density `density_break` must be non-negative.")
    if np.any(radius_break_cgs <= 0):
        raise ValueError("Break radius `radius_break` must be positive.")

    density_break_cgs = float(density_break_cgs)
    radius_break_cgs = float(radius_break_cgs)

    def _rho_csm(
        radius: "_ArrayLike",
        _t: "_ArrayLike | None" = None,
    ) -> "_ArrayLike":
        radius_array = np.asarray(radius, dtype=float)
        x = radius_array / radius_break_cgs
        rho = np.where(
            radius_array < radius_break_cgs,
            density_break_cgs * x ** (-slope_inner),
            density_break_cgs * x ** (-slope_outer),
        )

        return float(rho) if radius_array.ndim == 0 else rho

    return _rho_csm


def get_smooth_truncated_wind_csm_density_func(
    mass_loss_rate: "_UnitBearingScalarLike",
    wind_velocity: "_UnitBearingScalarLike",
    r_max: "_UnitBearingScalarLike",
    transition_width: "_UnitBearingScalarLike",
    density_floor: "_UnitBearingScalarLike" = 0.0 * u.g / u.cm**3,
) -> Callable[["_ArrayLike", "_ArrayLike | None"], "_ArrayLike"]:
    r"""
    Build a smoothly truncated wind-like CSM density profile callable.

    The returned function evaluates

    .. math::

        \rho_{\rm CSM}(r)
        =
        \rho_{\rm floor}
        +
        \left(A r^{-2} - \rho_{\rm floor}\right)
        \frac{1}{2}
        \left[
            1 - \tanh\left(\frac{r-r_{\max}}{\Delta r}\right)
        \right],

    where :math:`A = \dot{M}/(4\pi v_w)`.

    Parameters
    ----------
    mass_loss_rate : astropy.units.Quantity or float
        Progenitor mass-loss rate.
    wind_velocity : astropy.units.Quantity or float
        Wind velocity.
    r_max : astropy.units.Quantity or float
        Characteristic outer wind radius.
    transition_width : astropy.units.Quantity or float
        Width of the smooth transition.
    density_floor : astropy.units.Quantity or float, optional
        Density approached outside the wind region.

    Returns
    -------
    rho_csm : callable
        Callable ``rho_csm(radius, _t=None)`` returning density in
        :math:`\mathrm{g\,cm^{-3}}`.
    """
    A_cgs = compute_wind_csm_parameters(
        mass_loss_rate=mass_loss_rate,
        wind_velocity=wind_velocity,
    ).to_value(u.g / u.cm)

    r_max_cgs = ensure_in_units(r_max, u.cm)
    transition_width_cgs = ensure_in_units(transition_width, u.cm)
    density_floor_cgs = ensure_in_units(density_floor, u.g / u.cm**3)

    if np.any(r_max_cgs <= 0):
        raise ValueError("Maximum wind radius `r_max` must be positive.")
    if np.any(transition_width_cgs <= 0):
        raise ValueError("Transition width `transition_width` must be positive.")
    if np.any(density_floor_cgs < 0):
        raise ValueError("Density floor `density_floor` must be non-negative.")

    r_max_cgs = float(r_max_cgs)
    transition_width_cgs = float(transition_width_cgs)
    density_floor_cgs = float(density_floor_cgs)

    def _rho_csm(
        radius: "_ArrayLike",
        _t: "_ArrayLike | None" = None,
    ) -> "_ArrayLike":
        radius_array = np.asarray(radius, dtype=float)
        rho_wind = A_cgs * radius_array**-2
        switch = 0.5 * (1.0 - np.tanh((radius_array - r_max_cgs) / transition_width_cgs))
        rho = density_floor_cgs + (rho_wind - density_floor_cgs) * switch

        return float(rho) if radius_array.ndim == 0 else rho

    return _rho_csm
