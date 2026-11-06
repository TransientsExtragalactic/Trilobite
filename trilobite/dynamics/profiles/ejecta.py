r"""
Ejecta density profile classes.

This module provides an abstract base class and concrete implementations for
homologous supernova ejecta density profiles. All profiles assume the
homologous expansion form

.. math::

    \\rho_{\\rm ej}(r, t) = t^{-3}\\,G(r/t),

where :math:`G(v)` is the time-independent velocity-space kernel. The kernel
:math:`G` is exposed directly through :meth:`~EjectaDensityProfile.kernel` and
:meth:`~EjectaDensityProfile.fkernel` for cases where the velocity-space
distribution is needed on its own.

Supported kernels:

- **Broken power law** (Chevalier-style): inner flat/power-law core joined to a
  steep outer envelope, optionally truncated at a maximum velocity.
- **Exponential**: smooth exponential fall-off, optionally truncated between a
  minimum and maximum velocity.

Both kernels can be analytically normalized to a supplied ejecta mass and
kinetic energy via :meth:`~BrokenPowerLawEjectaProfile.normalize`.
"""

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from astropy import units as u
from scipy.optimize import brentq

from trilobite.utils.misc_utils import ensure_in_units

from .core import _DynamicalProfile

# Argument types that take the scalar fast path of ``as_optimized_callable``.
_SCALAR_TYPES = (float, np.floating)

if TYPE_CHECKING:
    from trilobite._typing import _ArrayLike, _UnitBearingArrayLike, _UnitBearingScalarLike


# =========================================================================== #
# Private Normalization Helpers                                                #
# =========================================================================== #
def _normalize_bpl_cgs(
    E_ej: float,
    M_ej: float,
    n: float,
    delta: float,
    v_max: float,
) -> tuple[float, float]:
    r"""
    Normalize a (possibly truncated) broken-power-law ejecta kernel in CGS.

    The kernel is

    .. math::

        G(v) =
        \begin{cases}
            K v^{-\delta}, & 0 \le v < v_t, \\
            K v_t^{n-\delta} v^{-n}, & v_t \le v \le v_{\max}, \\
            0, & v > v_{\max}.
        \end{cases}

    When ``v_max`` is infinite the transition velocity has the analytic form

    .. math::

        v_t^2 = \frac{2(5-\delta)(n-5)}{(3-\delta)(n-3)}\frac{E_{\rm ej}}{M_{\rm ej}}.

    With a finite ``v_max`` the transition velocity is solved numerically.

    Parameters
    ----------
    E_ej : float
        Total ejecta kinetic energy in erg.
    M_ej : float
        Total ejecta mass in g.
    n : float
        Outer density power-law index. Must satisfy ``n > 5``.
    delta : float
        Inner density power-law index. Must satisfy ``delta < 3``.
    v_max : float
        Maximum ejecta velocity in cm/s. Use ``np.inf`` for the untruncated case.

    Returns
    -------
    K : float
        Inner-branch normalization in :math:`\mathrm{g\,cm^{\delta-3}\,s^{3-\delta}}`.
    v_t : float
        Transition velocity in cm/s.
    """
    if E_ej <= 0:
        raise ValueError("`E_ej` must be positive.")
    if M_ej <= 0:
        raise ValueError("`M_ej` must be positive.")
    if delta >= 3:
        raise ValueError("`delta` must be < 3 for finite mass.")
    if n <= 5:
        raise ValueError("`n` must be > 5 for finite kinetic energy.")
    if n <= delta:
        raise ValueError("`n` must exceed `delta`.")
    if v_max <= 0:
        raise ValueError("`v_max` must be positive.")

    if np.isinf(v_max):
        specific_velocity_sq = 2.0 * E_ej / M_ej
        v_t = np.sqrt(specific_velocity_sq * ((5.0 - delta) * (n - 5.0)) / ((3.0 - delta) * (n - 3.0)))
        K = M_ej * v_t ** (delta - 3.0) * ((3.0 - delta) * (n - 3.0)) / (4.0 * np.pi * (n - delta))
        return K, v_t

    target = E_ej / M_ej
    if target >= 0.5 * v_max**2:
        raise ValueError("`E_ej / M_ej` is incompatible with `v_max`: must be < 0.5 * v_max**2.")

    # The transition velocity is found from the condition E/M = (1/2) <v^4>/<v^2>, with the moments written as
    # integrals of the kernel. The moments accept an array of trial transition velocities so that the bracketing
    # scan below is evaluated in one vectorized pass.
    def _power_integral(a, b, p: float):
        """Integral of ``v**p`` over ``[a, b]``, zero where ``b <= a``."""
        if abs(p + 1.0) <= 1.0e-8 + 1.0e-5:  # p == -1 (np.isclose(p, -1) semantics)
            return np.where(b > a, np.log(np.where(b > a, b, 1.0) / np.where(b > a, a, 1.0)), 0.0)
        return np.where(b > a, (b ** (p + 1.0) - a ** (p + 1.0)) / (p + 1.0), 0.0)

    def _moment(v_t_, q: int):
        inner_upper = np.minimum(v_t_, v_max)
        I_inner = _power_integral(0.0, inner_upper, q - delta)
        I_outer = np.where(v_t_ < v_max, v_t_ ** (n - delta) * _power_integral(v_t_, v_max, q - n), 0.0)
        return I_inner + I_outer

    def _residual(log_v_t):
        v_t_ = np.exp(log_v_t)
        return 0.5 * _moment(v_t_, 4) / _moment(v_t_, 2) - target

    lower = v_max * 1.0e-12
    upper = v_max * (1.0 - 1.0e-12)
    grid = np.geomspace(lower, upper, 256)
    values = _residual(np.log(grid))

    # The first grid cell containing a root (an exact zero at its left edge counts as a root).
    exact_zero = values[:-1] == 0.0
    crossing = exact_zero | (np.sign(values[:-1]) != np.sign(values[1:]))
    if not np.any(crossing):
        raise RuntimeError(
            "Could not bracket the BPL transition velocity. The requested E/M may be incompatible with `v_max`."
        )
    i = int(np.argmax(crossing))

    if exact_zero[i]:
        v_t = grid[i]
    else:
        v_t = np.exp(brentq(lambda x: float(_residual(x)), np.log(grid[i]), np.log(grid[i + 1])))
    K = M_ej / (4.0 * np.pi * float(_moment(v_t, 2)))
    return K, float(v_t)


def _normalize_exponential_cgs(
    E_ej: float,
    M_ej: float,
    v_min: float,
    v_max: float,
) -> tuple[float, float]:
    r"""
    Normalize a (possibly truncated) exponential ejecta kernel in CGS.

    The kernel is

    .. math::

        G(v) =
        \begin{cases}
            K\exp(-v/v_e), & v_{\min} \le v \le v_{\max}, \\
            0, & \text{otherwise}.
        \end{cases}

    For the untruncated case (:math:`v_{\min}=0`, :math:`v_{\max}=\infty`),

    .. math::

        v_e = \sqrt{E_{\rm ej}/(6 M_{\rm ej})},
        \qquad
        K = M_{\rm ej}/(8\pi v_e^3).

    With truncation, :math:`v_e` is solved numerically.

    Parameters
    ----------
    E_ej : float
        Total ejecta kinetic energy in erg.
    M_ej : float
        Total ejecta mass in g.
    v_min : float
        Lower velocity cutoff in cm/s.
    v_max : float
        Upper velocity cutoff in cm/s.

    Returns
    -------
    K : float
        Kernel normalization in :math:`\mathrm{g\,s^3\,cm^{-3}}`.
    v_e : float
        Exponential velocity scale in cm/s.
    """
    if E_ej <= 0:
        raise ValueError("`E_ej` must be positive.")
    if M_ej <= 0:
        raise ValueError("`M_ej` must be positive.")
    if v_min < 0:
        raise ValueError("`v_min` must be non-negative.")
    if v_max <= v_min:
        raise ValueError("`v_max` must be greater than `v_min`.")

    if v_min == 0.0 and np.isinf(v_max):
        v_e = np.sqrt(E_ej / (6.0 * M_ej))
        K = M_ej / (8.0 * np.pi * v_e**3)
        return K, v_e

    target = E_ej / M_ej

    def _moment(v_e_: float, order: int) -> float:
        x0 = v_min / v_e_
        x1 = v_max / v_e_ if np.isfinite(v_max) else np.inf

        if order == 2:
            p0 = x0**2 + 2.0 * x0 + 2.0
            term0 = np.exp(-x0) * p0
            term1 = 0.0 if np.isinf(x1) else np.exp(-x1) * (x1**2 + 2.0 * x1 + 2.0)
            return v_e_**3 * (term0 - term1)

        if order == 4:
            p0 = x0**4 + 4.0 * x0**3 + 12.0 * x0**2 + 24.0 * x0 + 24.0
            term0 = np.exp(-x0) * p0
            term1 = 0.0 if np.isinf(x1) else np.exp(-x1) * (x1**4 + 4.0 * x1**3 + 12.0 * x1**2 + 24.0 * x1 + 24.0)
            return v_e_**5 * (term0 - term1)

        raise ValueError("Only moment orders 2 and 4 are supported.")

    def _residual(log_v_e: float) -> float:
        v_e_ = np.exp(log_v_e)
        return 0.5 * _moment(v_e_, 4) / _moment(v_e_, 2) - target

    v_char = np.sqrt(2.0 * E_ej / M_ej)
    grid = np.geomspace(v_char * 1.0e-8, v_char * 1.0e8, 256)
    values = np.array([_residual(np.log(v)) for v in grid])

    bracket = None
    for left, right, fl, fr in zip(grid[:-1], grid[1:], values[:-1], values[1:]):
        if fl == 0.0:
            bracket = (left, left)
            break
        if np.sign(fl) != np.sign(fr):
            bracket = (left, right)
            break

    if bracket is None:
        raise RuntimeError(
            "Could not bracket the exponential velocity scale. "
            "The requested E/M may be incompatible with the velocity cutoffs."
        )

    v_e = bracket[0] if bracket[0] == bracket[1] else np.exp(brentq(_residual, np.log(bracket[0]), np.log(bracket[1])))
    K = M_ej / (4.0 * np.pi * _moment(v_e, 2))
    return K, v_e


# =========================================================================== #
# Ejecta Profile ABC                                                          #
# =========================================================================== #
class EjectaDensityProfile(_DynamicalProfile):
    r"""
    Abstract base class for homologous ejecta density profiles.

    All ejecta profiles in this module assume the homologous factored form

    .. math::

        \rho_{\rm ej}(r, t) = t^{-3}\,G(r/t),

    where :math:`G(v)` is the time-independent velocity-space kernel. Subclasses
    implement :meth:`_opt_eval_kernel`; this ABC provides :meth:`_opt_eval` and
    the kernel access methods :meth:`kernel` and :meth:`fkernel`.

    Subclasses must implement:

    ``_opt_eval_kernel``
        Unit-free evaluation of :math:`G(v)` in :math:`\mathrm{g\,s^3\,cm^{-3}}`.

    ``_validate_and_process_parameters``
        Validate physical parameters and convert them to unit-free CGS values.

    See Also
    --------
    BrokenPowerLawEjectaProfile : Chevalier-style broken-power-law ejecta kernel.
    ExponentialEjectaProfile : Exponential ejecta kernel.
    """

    OUTPUT_UNITS: ClassVar = u.g / u.cm**3
    KERNEL_UNITS: ClassVar = u.g * u.s**3 / u.cm**3
    """Units of the velocity-space kernel :math:`G(v)`."""

    # ------------------------------------------------------------------ #
    # Kernel Interface                                                   #
    # ------------------------------------------------------------------ #
    @classmethod
    @abstractmethod
    def _opt_eval_kernel(cls, v: np.ndarray, **parameters: Any) -> np.ndarray:
        r"""
        Evaluate the ejecta kernel :math:`G(v)` using processed unit-free inputs.

        Parameters
        ----------
        v : ~numpy.ndarray
            Ejecta velocity in cm/s.
        **parameters
            Already-processed profile parameters.

        Returns
        -------
        G : ~numpy.ndarray
            Velocity-space ejecta density kernel in :math:`\mathrm{g\,s^3\,cm^{-3}}`.
        """
        raise NotImplementedError

    @classmethod
    def kernel(cls, v: "_UnitBearingArrayLike", **parameters: Any) -> u.Quantity:
        r"""
        Evaluate the velocity-space ejecta kernel :math:`G(v)` with unit handling.

        The physical density is recovered as

        .. math::

            \rho_{\rm ej}(r, t) = t^{-3}\,G(r/t).

        Parameters
        ----------
        v : float, array-like, or ~astropy.units.Quantity
            Ejecta velocity. Bare values are interpreted as cm/s.
        **parameters
            Physical parameters for the profile.

        Returns
        -------
        G : ~astropy.units.Quantity
            Kernel value in :math:`\mathrm{g\,s^3\,cm^{-3}}`.
        """
        v_cgs = np.asarray(ensure_in_units(v, u.cm / u.s))
        processed = cls._validate_and_process_parameters(**parameters)
        return cls._opt_eval_kernel(v_cgs, **processed) * cls.KERNEL_UNITS

    @classmethod
    def fkernel(cls, v: "_ArrayLike", **parameters: Any) -> np.ndarray:
        r"""
        Evaluate the velocity-space ejecta kernel :math:`G(v)` without unit handling.

        This calls :meth:`_opt_eval_kernel` directly. No unit conversion or
        validation is performed. The input ``v`` must be in cm/s and
        ``parameters`` must already be processed CGS values.

        Parameters
        ----------
        v : float or array-like
            Ejecta velocity in cm/s.
        **parameters
            Already-processed CGS profile parameters.

        Returns
        -------
        G : float or ~numpy.ndarray
            Kernel value in :math:`\mathrm{g\,s^3\,cm^{-3}}`.
        """
        return cls._opt_eval_kernel(np.asarray(v, dtype=float), **parameters)

    # ------------------------------------------------------------------ #
    # Density Profile Evaluation                                         #
    # ------------------------------------------------------------------ #
    @classmethod
    def _opt_eval(cls, r: np.ndarray, t: np.ndarray, **parameters: Any) -> np.ndarray:
        v = r / t
        G = cls._opt_eval_kernel(v, **parameters)
        return G / t**3

    @classmethod
    @abstractmethod
    def _validate_and_process_parameters(cls, **parameters: Any) -> dict[str, Any]:
        raise NotImplementedError


# =========================================================================== #
# Concrete Ejecta Profiles                                                    #
# =========================================================================== #
class BrokenPowerLawEjectaProfile(EjectaDensityProfile):
    r"""
    Broken-power-law homologous ejecta density profile.

    The velocity-space kernel is

    .. math::

        G(v) =
        \begin{cases}
            K v^{-\delta}, & 0 \le v < v_t, \\
            K v_t^{n-\delta} v^{-n}, & v_t \le v \le v_{\max}, \\
            0, & v > v_{\max},
        \end{cases}

    and the physical density is

    .. math::

        \rho_{\rm ej}(r, t) = t^{-3}\,G(r/t).

    Setting ``v_max=np.inf`` recovers the standard untruncated Chevalier profile
    :footcite:p:`chevalierSelfsimilarSolutionsInteraction1982`.

    Parameters
    ----------
    K : float or ~astropy.units.Quantity
        Inner-branch kernel normalization in
        :math:`\mathrm{g\,cm^{\delta-3}\,s^{3-\delta}}`.
    v_t : float or ~astropy.units.Quantity
        Transition velocity. Bare values interpreted as cm/s.
    n : float
        Outer density power-law index. Must satisfy ``n > 5``.
    delta : float
        Inner density power-law index. Must satisfy ``delta < 3``.
    v_max : float or ~astropy.units.Quantity, optional
        Maximum ejecta velocity. Bare values interpreted as cm/s.
        Default is ``np.inf`` (untruncated).

    See Also
    --------
    BrokenPowerLawEjectaProfile.normalize : Compute ``K`` and ``v_t`` from
        ejecta mass and kinetic energy.

    References
    ----------
    .. footbibliography::

    Examples
    --------
    Normalize to an ejecta mass and kinetic energy and evaluate the density:

    >>> import astropy.units as u
    >>> K, v_t = BrokenPowerLawEjectaProfile.normalize(
    ...     E_ej=1e51 * u.erg, M_ej=1.0 * u.Msun, n=10, delta=0
    ... )
    >>> rho = BrokenPowerLawEjectaProfile.eval(
    ...     1e16 * u.cm, 10 * u.day, K=K, v_t=v_t, n=10, delta=0
    ... )
    """

    @classmethod
    def normalize(
        cls,
        E_ej: "_UnitBearingScalarLike",
        M_ej: "_UnitBearingScalarLike",
        n: float = 10,
        delta: float = 0,
        v_max: "_UnitBearingScalarLike" = np.inf,
    ) -> tuple[u.Quantity, u.Quantity]:
        r"""
        Compute the kernel normalization and transition velocity.

        The returned parameters ``K`` and ``v_t`` are chosen so that the
        profile integrates to the supplied ejecta mass and kinetic energy.

        Parameters
        ----------
        E_ej : float or ~astropy.units.Quantity
            Total ejecta kinetic energy. Bare values interpreted as erg.
        M_ej : float or ~astropy.units.Quantity
            Total ejecta mass. Bare values interpreted as g.
        n : float, optional
            Outer density power-law index. Must satisfy ``n > 5``. Default is 10.
        delta : float, optional
            Inner density power-law index. Must satisfy ``delta < 3``. Default is 0.
        v_max : float or ~astropy.units.Quantity, optional
            Maximum ejecta velocity. Bare values interpreted as cm/s.
            Default is ``np.inf`` (untruncated).

        Returns
        -------
        K : ~astropy.units.Quantity
            Inner-branch normalization in
            :math:`\mathrm{g\,cm^{\delta-3}\,s^{3-\delta}}`.
        v_t : ~astropy.units.Quantity
            Transition velocity in cm/s.
        """
        E_cgs = float(ensure_in_units(E_ej, u.erg))
        M_cgs = float(ensure_in_units(M_ej, u.g))
        v_max_cgs = float(ensure_in_units(v_max, u.cm / u.s)) if not np.isinf(v_max) else np.inf

        K_cgs, v_t_cgs = _normalize_bpl_cgs(E_cgs, M_cgs, n=float(n), delta=float(delta), v_max=v_max_cgs)

        K = K_cgs * (u.g * u.cm ** (delta - 3) * u.s ** (3 - delta))
        v_t = v_t_cgs * (u.cm / u.s)
        return K, v_t

    @classmethod
    def _validate_and_process_parameters(cls, *, K, v_t, n, delta, v_max=np.inf, **_):
        delta_f = float(delta)
        n_f = float(n)
        K_cgs = float(ensure_in_units(K, u.g * u.cm ** (delta_f - 3) * u.s ** (3 - delta_f)))
        v_t_cgs = float(ensure_in_units(v_t, u.cm / u.s))
        v_max_cgs = float(ensure_in_units(v_max, u.cm / u.s)) if not np.isinf(v_max) else np.inf

        if K_cgs <= 0:
            raise ValueError("`K` must be positive.")
        if v_t_cgs <= 0:
            raise ValueError("`v_t` must be positive.")

        K_outer = K_cgs * v_t_cgs ** (n_f - delta_f)

        return {
            "K_inner": K_cgs,
            "K_outer": K_outer,
            "v_t": v_t_cgs,
            "n": n_f,
            "delta": delta_f,
            "v_max": v_max_cgs,
        }

    @classmethod
    def as_optimized_callable(cls, **parameters: Any):
        """
        Return a fast unit-free callable with fixed profile parameters.

        This behaves exactly like :meth:`~trilobite.dynamics.profiles.core._DynamicalProfile.as_optimized_callable`,
        but evaluates scalar ``(r, t)`` arguments with plain scalar arithmetic instead of building
        arrays. ODE right-hand sides call the density at a single point thousands of times, where the array
        machinery dominates the cost. Array arguments take the usual vectorized path.
        """
        p = cls._validate_and_process_parameters(**parameters)
        K_inner, K_outer, v_t, n, delta, v_max = (p[k] for k in ("K_inner", "K_outer", "v_t", "n", "delta", "v_max"))
        truncated = not np.isinf(v_max)

        def _frozen(r, t):
            # Anything but a positive scalar point (arrays, non-positive values) takes the vectorized path, whose
            # handling of the edge cases is the reference.
            if not (isinstance(r, _SCALAR_TYPES) and isinstance(t, _SCALAR_TYPES) and r > 0.0 and t > 0.0):
                return cls._opt_eval(r, t, **p)
            v = r / t
            if truncated and v > v_max:
                return 0.0 / t**3
            G = K_inner * v ** (-delta) if v < v_t else K_outer * v ** (-n)
            return G / t**3

        return _frozen

    @classmethod
    def _opt_eval_kernel(cls, v, *, K_inner, K_outer, v_t, n, delta, v_max, **_):
        v_arr = np.asarray(v, dtype=float)
        G_raw = np.where(
            v_arr < v_t,
            K_inner * v_arr ** (-delta),
            K_outer * v_arr ** (-n),
        )
        if not np.isinf(v_max):
            G_raw = np.where(v_arr <= v_max, G_raw, 0.0)
        return float(G_raw) if v_arr.ndim == 0 else G_raw


class ExponentialEjectaProfile(EjectaDensityProfile):
    r"""
    Exponential homologous ejecta density profile.

    The velocity-space kernel is

    .. math::

        G(v) =
        \begin{cases}
            K\exp(-v/v_e), & v_{\min} \le v \le v_{\max}, \\
            0, & \text{otherwise},
        \end{cases}

    and the physical density is

    .. math::

        \rho_{\rm ej}(r, t) = t^{-3}\,G(r/t).

    Setting ``v_min=0`` and ``v_max=np.inf`` recovers the standard untruncated
    exponential profile.

    Parameters
    ----------
    K : float or ~astropy.units.Quantity
        Kernel normalization in :math:`\mathrm{g\,s^3\,cm^{-3}}`.
    v_e : float or ~astropy.units.Quantity
        Exponential velocity scale. Bare values interpreted as cm/s.
    v_min : float or ~astropy.units.Quantity, optional
        Lower velocity cutoff. Bare values interpreted as cm/s. Default is 0.
    v_max : float or ~astropy.units.Quantity, optional
        Upper velocity cutoff. Bare values interpreted as cm/s.
        Default is ``np.inf`` (untruncated).

    See Also
    --------
    ExponentialEjectaProfile.normalize : Compute ``K`` and ``v_e`` from
        ejecta mass and kinetic energy.

    Examples
    --------
    >>> import astropy.units as u
    >>> K, v_e = ExponentialEjectaProfile.normalize(
    ...     E_ej=1e51 * u.erg, M_ej=1.0 * u.Msun
    ... )
    >>> rho = ExponentialEjectaProfile.eval(
    ...     1e16 * u.cm, 10 * u.day, K=K, v_e=v_e
    ... )
    """

    @classmethod
    def normalize(
        cls,
        E_ej: "_UnitBearingScalarLike",
        M_ej: "_UnitBearingScalarLike",
        v_min: "_UnitBearingScalarLike" = 0.0,
        v_max: "_UnitBearingScalarLike" = np.inf,
    ) -> tuple[u.Quantity, u.Quantity]:
        r"""
        Compute the kernel normalization and velocity scale.

        Parameters
        ----------
        E_ej : float or ~astropy.units.Quantity
            Total ejecta kinetic energy. Bare values interpreted as erg.
        M_ej : float or ~astropy.units.Quantity
            Total ejecta mass. Bare values interpreted as g.
        v_min : float or ~astropy.units.Quantity, optional
            Lower velocity cutoff. Bare values interpreted as cm/s. Default is 0.
        v_max : float or ~astropy.units.Quantity, optional
            Upper velocity cutoff. Bare values interpreted as cm/s.
            Default is ``np.inf``.

        Returns
        -------
        K : ~astropy.units.Quantity
            Kernel normalization in :math:`\mathrm{g\,s^3\,cm^{-3}}`.
        v_e : ~astropy.units.Quantity
            Exponential velocity scale in cm/s.
        """
        E_cgs = float(ensure_in_units(E_ej, u.erg))
        M_cgs = float(ensure_in_units(M_ej, u.g))
        v_min_cgs = float(ensure_in_units(v_min, u.cm / u.s)) if not np.isinf(v_min) else 0.0
        v_max_cgs = float(ensure_in_units(v_max, u.cm / u.s)) if not np.isinf(v_max) else np.inf

        K_cgs, v_e_cgs = _normalize_exponential_cgs(E_cgs, M_cgs, v_min=v_min_cgs, v_max=v_max_cgs)

        K = K_cgs * (u.g * u.s**3 / u.cm**3)
        v_e = v_e_cgs * (u.cm / u.s)
        return K, v_e

    @classmethod
    def _validate_and_process_parameters(cls, *, K, v_e, v_min=0.0, v_max=np.inf, **_):
        K_cgs = float(ensure_in_units(K, u.g * u.s**3 / u.cm**3))
        v_e_cgs = float(ensure_in_units(v_e, u.cm / u.s))
        v_min_cgs = float(ensure_in_units(v_min, u.cm / u.s)) if not np.isinf(v_min) else 0.0
        v_max_cgs = float(ensure_in_units(v_max, u.cm / u.s)) if not np.isinf(v_max) else np.inf

        if K_cgs <= 0:
            raise ValueError("`K` must be positive.")
        if v_e_cgs <= 0:
            raise ValueError("`v_e` must be positive.")
        if v_min_cgs < 0:
            raise ValueError("`v_min` must be non-negative.")
        if v_max_cgs <= v_min_cgs:
            raise ValueError("`v_max` must be greater than `v_min`.")

        return {"K": K_cgs, "v_e": v_e_cgs, "v_min": v_min_cgs, "v_max": v_max_cgs}

    @classmethod
    def _opt_eval_kernel(cls, v, *, K, v_e, v_min, v_max, **_):
        v_arr = np.asarray(v, dtype=float)
        G_raw = K * np.exp(-v_arr / v_e)

        inside = v_arr >= v_min
        if not np.isinf(v_max):
            inside = inside & (v_arr <= v_max)

        G = np.where(inside, G_raw, 0.0)
        return float(G) if v_arr.ndim == 0 else G
