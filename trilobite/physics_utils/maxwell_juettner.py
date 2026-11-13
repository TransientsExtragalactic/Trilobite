r"""
Maxwell-Juettner (MJ) distribution utilities.

For particles of mass :math:`m` at temperature :math:`T`, the distribution of the Lorentz factor is

.. math::

    f(\gamma) = \frac{\gamma^2 \beta}{\Theta K_2(1/\Theta)} e^{-\gamma/\Theta}, \qquad \gamma \geq 1,

where :math:`\beta = \sqrt{1 - \gamma^{-2}}`, :math:`K_2` is the modified Bessel function of the second kind, and
:math:`\Theta = k_B T / (m c^2)` is the dimensionless temperature. Nothing here is specific to electrons: ``mass``
defaults to the electron mass, but any particle works.

The ``compute_*`` functions are the public, validated interface. The ``_opt_mj_*`` functions are unvalidated backends
that take floats or ``float64`` arrays in CGS units.
"""

from typing import Union

import numpy as np
from astropy import units as u
from scipy.integrate import quad
from scipy.interpolate import CubicHermiteSpline
from scipy.special import kve

from trilobite._typing import _ArrayLike, _ScalarLike, _UnitBearingArrayLike, _UnitBearingScalarLike
from trilobite.physics_utils.constants import c_cgs, k_B_cgs, m_e_cgs
from trilobite.utils.misc_utils import ensure_in_units

__all__ = [
    "compute_dimensionless_temperature",
    "compute_temperature_from_dimensionless",
    "compute_mj_pdf",
    "compute_mj_cdf",
    "compute_mj_moment",
    "compute_mj_mean_lorentz_factor",
    "compute_mj_mean_kinetic_energy",
    "compute_mj_theta_from_kinetic_energy",
    "compute_mj_number_density_from_total_energy_density",
    "compute_mj_number_density_from_internal_energy_density",
]

_CDF_PANELS = 1000  # Simpson panels in the CDF grid
_CDF_TAIL_WIDTHS = 50.0  # the CDF grid ends at gamma - 1 = 50 Theta; <1e-18 of the probability lies beyond
_SERIES_THETA = 1e-4  # below this <gamma - 1> uses its series expansion
_KVE_ASYMPTOTIC_Z = 1e8  # above this kve(2, z) uses its asymptotic series (scipy returns NaN beyond ~1e10)
_EXP_UNDERFLOW = -746.0  # exp(x) is zero below this


# ============================================================================ #
# Backends                                                                     #
# ============================================================================ #
def _opt_mj_kve2(z):
    r"""Return :math:`e^z K_2(z)`, using the Hankel series (error :math:`O(z^{-3})`) where scipy fails."""
    asymptotic = np.sqrt(0.5 * np.pi / z) * (1.0 + 15.0 / (8.0 * z) + 105.0 / (128.0 * z**2))
    return np.where(z > _KVE_ASYMPTOTIC_Z, asymptotic, kve(2, z))


def _opt_mj_pdf_excess(x, Theta, norm=1.0):
    r"""MJ distribution in terms of :math:`x = \gamma - 1`: zero for :math:`x < 0` and :math:`x = \infty`, NaN for NaN.

    Working in :math:`x` keeps full precision near :math:`\gamma = 1`, where :math:`1 + x` rounds away information
    when :math:`\Theta \ll 1`.
    """
    x_safe = np.maximum(x, 0.0)  # propagates NaN
    gamma = 1.0 + x_safe

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        # sqrt(x (x + 2)) / gamma avoids the cancellation in 1 - gamma^-2 near gamma = 1; the direct form avoids
        # overflow in x (x + 2) for large gamma.
        beta = np.where(x_safe < 1.0, np.sqrt(x_safe * (x_safe + 2.0)) / gamma, np.sqrt(1.0 - gamma**-2))

        # K_2(1/Theta) underflows for Theta <~ 1e-3, so use kve(2, z) = e^z K_2(z):
        # exp(-gamma/Theta) / K_2(1/Theta) = exp(-x/Theta) / kve(2, 1/Theta).
        exponent = -x_safe / Theta
        shape = norm * gamma**2 * beta * np.exp(exponent) / (Theta * _opt_mj_kve2(1.0 / Theta))

    value = np.where(exponent < _EXP_UNDERFLOW, 0.0, shape)  # also avoids inf * 0 at x = inf
    return np.where(x < 0.0, 0.0, value)[()]


def _opt_mj_pdf(gamma, Theta, norm=1.0):
    return _opt_mj_pdf_excess(gamma - 1.0, Theta, norm)  # exact for gamma in [1/2, 2]


def _opt_mj_cdf(gamma, Theta, norm=1.0):
    # Scalar Theta only. The grid depends on Theta alone, so a point's value never depends on the rest of the array.
    # In t = sqrt((gamma - 1) / Theta) the pdf's square-root singularity at gamma = 1 becomes dF/dt ~ t^2, which is
    # smooth, so Simpson's rule and Hermite interpolation converge at full order.
    Theta = float(Theta)
    t_max = np.sqrt(_CDF_TAIL_WIDTHS)
    nodes = np.linspace(0.0, t_max, _CDF_PANELS + 1)
    mids = 0.5 * (nodes[:-1] + nodes[1:])

    def dF_dt(t):
        return _opt_mj_pdf_excess(Theta * t**2, Theta) * 2.0 * Theta * t

    f_nodes, f_mids = dF_dt(nodes), dF_dt(mids)
    F = np.concatenate([[0.0], np.cumsum(np.diff(nodes) / 6.0 * (f_nodes[:-1] + 4.0 * f_mids + f_nodes[1:]))])

    # Dividing by F[-1] removes the discretization error (the neglected tail is <1e-18). F and its exact derivative are
    # known at every node, so a cubic Hermite spline interpolates to fourth order; gamma beyond the grid gives 1.
    spline = CubicHermiteSpline(nodes, F / F[-1], f_nodes / F[-1])
    t = np.sqrt(np.maximum(np.asarray(gamma, dtype="f8") - 1.0, 0.0) / Theta)
    return norm * spline(np.minimum(t, t_max))


def _opt_mj_exact_mean_kinetic_energy(Theta):
    # Below _SERIES_THETA the subtraction <gamma> - 1 loses digits (and kve fails for Theta <~ 1e-9), so use
    # 3/2 Theta + 15/8 Theta^2 - 15/8 Theta^3 + O(Theta^4); both routes are accurate to ~1e-12 at the switch.
    z = 1.0 / Theta
    closed_form = kve(1, z) / kve(2, z) + 3.0 * Theta - 1.0
    series = Theta * (1.5 + Theta * (1.875 - 1.875 * Theta))
    return np.where(Theta < _SERIES_THETA, series, closed_form)[()]


def _opt_mj_exact_mean_lorentz_factor(Theta):
    return 1.0 + _opt_mj_exact_mean_kinetic_energy(Theta)


# Gammie & McKinney (1998) fits to the two means above.
def _opt_mj_approx_mean_lorentz_factor(Theta):
    return (15.0 * Theta**2 + 11.0 * Theta + 4.0) / (5.0 * Theta + 4.0)


def _opt_mj_approx_mean_kinetic_energy(Theta):
    return Theta * (6.0 + 15.0 * Theta) / (4.0 + 5.0 * Theta)


def _opt_mj_moment(order, Theta):
    if order == 0:
        return np.ones_like(Theta)[()]

    if order == 1:
        return _opt_mj_exact_mean_lorentz_factor(Theta)

    if order == 2:
        return 1.0 + 3.0 * Theta * (Theta + _opt_mj_exact_mean_lorentz_factor(Theta))

    # Integrate in s = (gamma - 1) / Theta, where the integrand ~ (1 + Theta s)^n s^(1/2) e^-s is O(1)-scaled for every
    # Theta, peaks near s = n - 1/Theta with width ~ sqrt(n), and is negligible beyond s_max. (Integrating over gamma in
    # [1, inf) misses the peak once Theta <~ 1e-5.)
    s_max = order + 60.0 + 12.0 * np.sqrt(order)

    def _quad(theta):
        def integrand(s):
            return theta * (1.0 + theta * s) ** order * _opt_mj_pdf_excess(theta * s, theta)

        return quad(integrand, 0.0, s_max, epsabs=0.0, epsrel=1e-12, limit=200)[0]

    return np.vectorize(_quad, otypes=["f8"])(Theta)[()]


def _opt_mj_theta_from_kinetic_energy(xi):
    discriminant = np.sqrt((6.0 - 5.0 * xi) ** 2 + 240.0 * xi)
    # For xi < 6/5 the textbook form (5 xi - 6 + sqrt(D)) / 30 cancels; the conjugate form 8 xi / (sqrt(D) + 6 - 5 xi)
    # is identical and does not.
    return np.where(
        xi < 1.2,
        8.0 * xi / (discriminant + 6.0 - 5.0 * xi),
        (5.0 * xi - 6.0 + discriminant) / 30.0,
    )[()]


# ============================================================================ #
# Validation                                                                   #
# ============================================================================ #
def _check(value, name, allow_zero=False) -> np.ndarray:
    ok = np.isfinite(value) & ((value >= 0.0) if allow_zero else (value > 0.0))
    if not np.all(ok):
        raise ValueError(f"`{name}` must be finite and {'non-negative' if allow_zero else 'positive'}.")
    return value


def _as_dimensionless(value, name) -> np.ndarray:
    # np.asarray would silently drop the units of a Quantity, e.g. a temperature in K passed as Theta.
    if isinstance(value, u.Quantity):
        try:
            value = value.to_value(u.dimensionless_unscaled)
        except u.UnitConversionError as err:
            raise ValueError(f"`{name}` must be dimensionless, got units of {value.unit}.") from err
    return np.asarray(value, dtype="f8")


def _validate_Theta(Theta) -> np.ndarray:
    return _check(_as_dimensionless(Theta, "Theta"), "Theta")


def _as_cgs(value, unit, name, allow_zero=False) -> np.ndarray:
    return _check(np.asarray(ensure_in_units(value, unit), dtype="f8"), name, allow_zero)


# ============================================================================ #
# Temperature                                                                  #
# ============================================================================ #
def compute_dimensionless_temperature(
    T: _UnitBearingArrayLike, mass: _UnitBearingScalarLike = m_e_cgs * u.g
) -> Union[_ScalarLike, np.ndarray]:
    r"""
    Compute the dimensionless temperature :math:`\Theta = k_B T / (m c^2)`.

    Parameters
    ----------
    T : float or ~astropy.units.Quantity
        Temperature. Bare floats are interpreted as kelvin.
    mass : float or ~astropy.units.Quantity, optional
        Particle mass. Bare floats are interpreted as grams. Default is the electron mass.

    Returns
    -------
    float or ~numpy.ndarray
        The dimensionless temperature :math:`\Theta`.

    Examples
    --------
    >>> import astropy.units as u
    >>> round(
    ...     float(
    ...         compute_dimensionless_temperature(
    ...             5.93e9 * u.K
    ...         )
    ...     ),
    ...     3,
    ... )
    1.0
    """
    T = _as_cgs(T, u.K, "T", allow_zero=True)
    mass = _as_cgs(mass, u.g, "mass")
    return (k_B_cgs * T / (mass * c_cgs**2))[()]


def compute_temperature_from_dimensionless(
    Theta: Union[_ArrayLike, _ScalarLike], mass: _UnitBearingScalarLike = m_e_cgs * u.g
) -> u.Quantity:
    r"""
    Compute the temperature corresponding to :math:`\Theta`; the inverse of :func:`compute_dimensionless_temperature`.

    Parameters
    ----------
    Theta : float or array-like
        Dimensionless temperature :math:`\Theta = k_B T / (m c^2)`. Must be positive.
    mass : float or ~astropy.units.Quantity, optional
        Particle mass. Bare floats are interpreted as grams. Default is the electron mass.

    Returns
    -------
    ~astropy.units.Quantity
        Temperature in kelvin.
    """
    Theta = _validate_Theta(Theta)
    mass = _as_cgs(mass, u.g, "mass")
    return (Theta * mass * c_cgs**2 / k_B_cgs)[()] * u.K


# ============================================================================ #
# Distribution                                                                 #
# ============================================================================ #
def compute_mj_pdf(
    gamma: Union[_ArrayLike, _ScalarLike], Theta: Union[_ArrayLike, _ScalarLike], norm: float = 1.0
) -> Union[_ScalarLike, np.ndarray]:
    r"""
    Evaluate the Maxwell-Juettner distribution :math:`N(\gamma) = N_0 f(\gamma)`.

    The distribution is zero for :math:`\gamma < 1`, and ``nan`` for ``nan`` input. For :math:`\Theta \lesssim 10^{-8}`
    the distribution is Maxwell-Boltzmann to better than :math:`10^{-7}`, and ``gamma`` can no longer resolve it.

    Parameters
    ----------
    gamma : float or array-like
        Lorentz factors at which to evaluate the distribution.
    Theta : float or array-like
        Dimensionless temperature :math:`\Theta = k_B T / (m c^2)`. Must be positive.
    norm : float, optional
        Amplitude :math:`N_0`. Since :math:`f` integrates to one, this is the total number density. Default is ``1.0``.

    Returns
    -------
    float or ~numpy.ndarray
        The distribution evaluated at ``gamma``.
    """
    return _opt_mj_pdf(_as_dimensionless(gamma, "gamma"), _validate_Theta(Theta), _as_dimensionless(norm, "norm"))


def compute_mj_cdf(
    gamma: Union[_ArrayLike, _ScalarLike], Theta: _ScalarLike, norm: float = 1.0
) -> Union[_ScalarLike, np.ndarray]:
    r"""
    Evaluate the cumulative distribution :math:`F(\gamma) = N_0 \int_1^\gamma f(\gamma')\,d\gamma'`.

    The integral uses Simpson's rule in :math:`t = \sqrt{(\gamma - 1)/\Theta}` out to :math:`\gamma - 1 = 50\Theta`,
    with a cubic Hermite spline in between (absolute error about :math:`10^{-10}`). Each value depends only on its own
    ``gamma``. The CDF is exactly ``norm`` beyond the grid and at ``inf``, 0 for :math:`\gamma \leq 1`, and ``nan``
    for ``nan``.

    Parameters
    ----------
    gamma : float or array-like
        Lorentz factors at which to evaluate the CDF.
    Theta : float
        Dimensionless temperature. Must be a positive scalar, since all ``gamma`` share one integration grid.
    norm : float, optional
        Amplitude :math:`N_0`. Default is ``1.0``.

    Returns
    -------
    float or ~numpy.ndarray
        The cumulative distribution evaluated at ``gamma``.
    """
    Theta = _validate_Theta(Theta)
    if Theta.ndim != 0:
        raise ValueError("`compute_mj_cdf` shares one integration grid and requires a scalar `Theta`.")

    return _opt_mj_cdf(_as_dimensionless(gamma, "gamma"), Theta, _as_dimensionless(norm, "norm"))


# ============================================================================ #
# Moments                                                                      #
# ============================================================================ #
def compute_mj_moment(order: float, Theta: Union[_ArrayLike, _ScalarLike]) -> Union[_ScalarLike, np.ndarray]:
    r"""
    Compute the raw moment :math:`\langle \gamma^n \rangle`.

    Orders 0 to 2 are exact closed forms,

    .. math::

        M_0 = 1, \qquad M_1 = \frac{K_1(1/\Theta)}{K_2(1/\Theta)} + 3\Theta, \qquad M_2 = 1 + 3\Theta(\Theta + M_1),

    and other orders use numerical quadrature.

    Parameters
    ----------
    order : float
        Moment order :math:`n \geq 0`.
    Theta : float or array-like
        Dimensionless temperature. Must be positive.

    Returns
    -------
    float or ~numpy.ndarray
        The raw moment.
    """
    if not np.isfinite(order) or order < 0:
        raise ValueError("Moment order `order` must be finite and non-negative.")

    return _opt_mj_moment(order, _validate_Theta(Theta))


def compute_mj_mean_lorentz_factor(
    Theta: Union[_ArrayLike, _ScalarLike], approximate: bool = False
) -> Union[_ScalarLike, np.ndarray]:
    r"""
    Compute the mean Lorentz factor :math:`\langle\gamma\rangle`, which includes rest-mass energy.

    The approximation :footcite:p:`1998ApJ...498..313G`

    .. math::

        \langle\gamma\rangle \approx \frac{15\Theta^2 + 11\Theta + 4}{4 + 5\Theta}

    is accurate to about 1.3%.

    Parameters
    ----------
    Theta : float or array-like
        Dimensionless temperature. Must be positive.
    approximate : bool, optional
        Use the approximation instead of the exact Bessel-function result. Default is ``False``.

    Returns
    -------
    float or ~numpy.ndarray
        The mean Lorentz factor.
    """
    backend = _opt_mj_approx_mean_lorentz_factor if approximate else _opt_mj_exact_mean_lorentz_factor
    return backend(_validate_Theta(Theta))


def compute_mj_mean_kinetic_energy(
    Theta: Union[_ArrayLike, _ScalarLike], approximate: bool = False
) -> Union[_ScalarLike, np.ndarray]:
    r"""
    Compute the mean kinetic energy per particle in units of :math:`m c^2`, :math:`\langle\gamma - 1\rangle`.

    This excludes the rest-mass energy that :func:`compute_mj_mean_lorentz_factor` includes. The approximation
    :footcite:p:`1998ApJ...498..313G`

    .. math::

        \langle\gamma - 1\rangle \approx \Theta \frac{6 + 15\Theta}{4 + 5\Theta}

    is accurate to about 1.6%. The exact result is accurate to about :math:`10^{-12}`; below :math:`\Theta = 10^{-4}`
    it uses the series :math:`\tfrac{3}{2}\Theta + \tfrac{15}{8}\Theta^2 - \tfrac{15}{8}\Theta^3`.

    Parameters
    ----------
    Theta : float or array-like
        Dimensionless temperature. Must be positive.
    approximate : bool, optional
        Use the approximation instead of the exact result. Default is ``False``.

    Returns
    -------
    float or ~numpy.ndarray
        The mean kinetic energy per particle in units of :math:`m c^2`.
    """
    backend = _opt_mj_approx_mean_kinetic_energy if approximate else _opt_mj_exact_mean_kinetic_energy
    return backend(_validate_Theta(Theta))


def compute_mj_theta_from_kinetic_energy(xi: Union[_ArrayLike, _ScalarLike]) -> Union[_ScalarLike, np.ndarray]:
    r"""
    Invert the approximate mean kinetic energy to obtain :math:`\Theta`.

    Given :math:`\xi = U_{\rm int} / (\rho c^2)`, the internal energy per unit rest-mass energy, this solves
    :math:`\xi = \Theta (6 + 15\Theta) / (4 + 5\Theta)` :footcite:p:`1998ApJ...498..313G`, a quadratic with solution

    .. math::

        \Theta = \frac{5\xi - 6 + \sqrt{(6 - 5\xi)^2 + 240\xi}}{30}.

    Parameters
    ----------
    xi : float or array-like
        Internal energy per unit rest-mass energy. Must be positive.

    Returns
    -------
    float or ~numpy.ndarray
        The dimensionless temperature :math:`\Theta = k_B T / (m c^2)` of the particle that carries the rest mass.
    """
    return _opt_mj_theta_from_kinetic_energy(_check(_as_dimensionless(xi, "xi"), "xi"))


# ============================================================================ #
# Normalization                                                                #
# ============================================================================ #
def _number_density(energy_density, Theta, mass, mean_energy) -> u.Quantity:
    """Return ``n = u / (m c^2 <E>)``, where ``mean_energy`` is a backend giving ``<E>`` in units of ``m c^2``."""
    energy_density = _as_cgs(energy_density, u.erg / u.cm**3, "energy_density", allow_zero=True)
    mass = _as_cgs(mass, u.g, "mass")
    n = energy_density / (mass * c_cgs**2 * mean_energy(_validate_Theta(Theta)))
    return n[()] * u.cm**-3


def compute_mj_number_density_from_total_energy_density(
    energy_density: _UnitBearingArrayLike,
    Theta: Union[_ArrayLike, _ScalarLike],
    mass: _UnitBearingScalarLike = m_e_cgs * u.g,
    approximate: bool = False,
) -> u.Quantity:
    r"""
    Compute the number density :math:`n = u_{\rm tot} / (m c^2 \langle\gamma\rangle)` from a *total* energy density.

    Use this when the energy density includes rest mass. Otherwise use
    :func:`compute_mj_number_density_from_internal_energy_density`; the two differ by
    :math:`\langle\gamma\rangle / \langle\gamma - 1\rangle`, about 1.4 at :math:`\Theta = 1` and :math:`2/(3\Theta)`
    for :math:`\Theta \ll 1`. Since :math:`f` integrates to one, :math:`n` is also the amplitude :math:`N_0`.

    Parameters
    ----------
    energy_density : float or ~astropy.units.Quantity
        Total energy density. Bare floats are interpreted as :math:`\mathrm{erg\,cm^{-3}}`.
    Theta : float or array-like
        Dimensionless temperature. Must be positive.
    mass : float or ~astropy.units.Quantity, optional
        Particle mass. Bare floats are interpreted as grams. Default is the electron mass.
    approximate : bool, optional
        Use the approximate :math:`\langle\gamma\rangle` (error up to 1.3%). Default is ``False``.

    Returns
    -------
    ~astropy.units.Quantity
        Number density in :math:`\mathrm{cm^{-3}}`.
    """
    backend = _opt_mj_approx_mean_lorentz_factor if approximate else _opt_mj_exact_mean_lorentz_factor
    return _number_density(energy_density, Theta, mass, backend)


def compute_mj_number_density_from_internal_energy_density(
    energy_density: _UnitBearingArrayLike,
    Theta: Union[_ArrayLike, _ScalarLike],
    mass: _UnitBearingScalarLike = m_e_cgs * u.g,
    approximate: bool = False,
) -> u.Quantity:
    r"""
    Compute the number density :math:`n = u_{\rm int} / (m c^2 \langle\gamma - 1\rangle)` from an *internal* energy.

    Use this when the energy density is thermal (kinetic) energy only, as for the post-shock internal energy density
    of the jump conditions. See :func:`compute_mj_number_density_from_total_energy_density` for the version that
    includes rest mass.

    Parameters
    ----------
    energy_density : float or ~astropy.units.Quantity
        Internal energy density. Bare floats are interpreted as :math:`\mathrm{erg\,cm^{-3}}`.
    Theta : float or array-like
        Dimensionless temperature. Must be positive.
    mass : float or ~astropy.units.Quantity, optional
        Particle mass. Bare floats are interpreted as grams. Default is the electron mass.
    approximate : bool, optional
        Use the approximate :math:`\langle\gamma - 1\rangle` (error up to 1.6%). Default is ``False``.

    Returns
    -------
    ~astropy.units.Quantity
        Number density in :math:`\mathrm{cm^{-3}}`.
    """
    backend = _opt_mj_approx_mean_kinetic_energy if approximate else _opt_mj_exact_mean_kinetic_energy
    return _number_density(energy_density, Theta, mass, backend)
