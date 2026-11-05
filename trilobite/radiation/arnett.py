r"""
Arnett-type diffusion engines for radioactively / centrally powered transients.

The Arnett model describes the bolometric luminosity of an optically thick, homologously expanding ejecta that is
heated internally. The luminosity :math:`L` relaxes toward the instantaneously deposited power
:math:`P(t)` on a *diffusion clock* :math:`u(t)`,

.. math::

    \frac{dL}{du} = P(t) - L, \qquad L(0) = L_\mathrm{init},

whose formal solution

.. math::

    L(t) = L_\mathrm{init}\,e^{-u(t)} + \int_0^t P(t')\,\frac{du}{dt'}\,e^{-[u(t) - u(t')]}\,dt'

is evaluated by a compiled quadrature kernel (:func:`~trilobite.radiation.arnett.build_solver`).
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from functools import cache
from math import exp, sqrt
from typing import TYPE_CHECKING, NamedTuple, Optional

import numpy as np
from astropy import units as u
from numba import njit, prange
from numba.core.registry import CPUDispatcher

from trilobite.radiation.constants import c_cgs
from trilobite.utils.misc_utils import ensure_in_units

if TYPE_CHECKING:
    from trilobite._typing import _ArrayLike, _UnitBearingArrayLike

__all__ = [
    "QuadratureRule",
    "DiffusionClock",
    "HOMOLOGOUS_CLOCK",
    "FINITE_RADIUS_CLOCK",
    "Channel",
    "build_solver",
    "ArnettEngine",
    "NickelCobaltArnettEngine",
    "MagnetarArnettEngine",
    "MagnetarNickelArnettEngine",
]


# =================================== #
# Diffusion Clocks                    #
# =================================== #
# A clock u(t) with u(0) = 0 is described by three compiled functions. Each takes a parameter row ``p`` (the clock
# parameters of one realization):
#   u_diff(t_hi, dt, p) = u(t_hi) - u(t_hi - dt)   (accurate for dt << t_hi; u(t) is u_diff(t, t, p))
#   dudt(t, p)          = du/dt
#   t_of_u(u, p)        = the inverse t(u), used to place the integration window.


class DiffusionClock(NamedTuple):
    r"""
    The diffusion clock :math:`u(t)` of the relaxation equation :math:`dL/du = P - L`.

    Parameters
    ----------
    u_diff : numba dispatcher
        ``u_diff(t_hi, dt, p)`` returning :math:`u(t_\mathrm{hi}) - u(t_\mathrm{hi} - dt)`. Written in terms of
        ``dt`` so the difference keeps its digits when ``dt`` is small.
    dudt : numba dispatcher
        ``dudt(t, p)`` returning :math:`du/dt`.
    t_of_u : numba dispatcher
        ``t_of_u(u, p)`` returning the inverse :math:`t(u)`, for :math:`u > 0`.
    n_params : int
        Length of the parameter row ``p``.
    """

    u_diff: CPUDispatcher
    dudt: CPUDispatcher
    t_of_u: CPUDispatcher
    n_params: int


@njit(cache=True, inline="always")
def _homologous_u_diff(t_hi, dt, p):
    return dt * (2.0 * t_hi - dt) / (p[0] * p[0])


@njit(cache=True, inline="always")
def _homologous_dudt(t, p):
    return 2.0 * t / (p[0] * p[0])


@njit(cache=True, inline="always")
def _homologous_t_of_u(u_value, p):
    return p[0] * sqrt(u_value)


@njit(cache=True, inline="always")
def _finite_radius_u_diff(t_hi, dt, p):
    return dt * (2.0 * (t_hi + p[1]) - dt) / (p[0] * p[0])


@njit(cache=True, inline="always")
def _finite_radius_dudt(t, p):
    return 2.0 * (t + p[1]) / (p[0] * p[0])


@njit(cache=True, inline="always")
def _finite_radius_t_of_u(u_value, p):
    # Stable root of t^2 + 2 a t = u t_d^2 (no cancellation for small u or a).
    x = u_value * p[0] * p[0]
    return x / (sqrt(x + p[1] * p[1]) + p[1])


HOMOLOGOUS_CLOCK = DiffusionClock(_homologous_u_diff, _homologous_dudt, _homologous_t_of_u, 1)
r"""
:math:`u = (t/t_d)^2`. Parameter row: ``[t_d]`` (s).
"""

FINITE_RADIUS_CLOCK = DiffusionClock(_finite_radius_u_diff, _finite_radius_dudt, _finite_radius_t_of_u, 2)
r"""
:math:`u = (t^2 + 2 a t)/t_d^2` with :math:`a = R_0/v_\mathrm{sc}`, for ejecta with :math:`R = R_0 + v_\mathrm{sc} t`.
Parameter row: ``[t_d, a]`` (s, s).
"""

_CLOCKS = {"homologous": HOMOLOGOUS_CLOCK, "finite_radius": FINITE_RADIUS_CLOCK}
_TRAPPING_MODES = ("deposition", "emergence")


# =================================== #
# Kernel Formulation                  #
# =================================== #
# At compilation time, we need to build the JIT kernel for the Arnett solve.
# This is handled by the function defined here.


@njit(cache=True, inline="always")
def _unity(t, parameter_array):
    """Factor for an unused slot: 1, ignoring its (empty) parameter row."""
    return 1.0


@njit(cache=True, inline="always")
def _gamma_trapping(t, parameter_array):
    """Trapped fraction 1 - exp(-A/t^2), with A = row[0] in s^2."""
    if t <= 0.0:
        return 1.0
    return -np.expm1(-parameter_array[0] / (t * t))


def build_solver(clock, injection_function, *, deposition_function=None, emergence_function=None, window_depth=40.0):
    r"""
    Compile the Arnett solve for one clock and one source.

    For each requested time :math:`t_q` and parameter realization, the solver evaluates

    .. math::

        L(t_q) = \mathcal{E}(t_q)\left[L_\mathrm{init}\,e^{-u(t_q)} + \int_{t_\mathrm{min}}^{t_q} F(t')\,
                 \mathcal{D}(t')\,\dot u(t')\,e^{-[u(t_q) - u(t')]}\,dt'\right],
        \qquad t_\mathrm{min} = t\!\left(\max[u(t_q) - S,\ 0]\right),

    the formal solution of :math:`dL/du = F\mathcal{D} - L`, with a quadrature rule on [0, 1] mapped onto
    :math:`[t_\mathrm{min}, t_q]`. The window drops injection more than :math:`S` clock units in the past; the
    dropped part is exactly :math:`e^{-S} L(t_\mathrm{min})`.

    The equation is linear in the deposited power, so several sources are handled by solving each one and
    summing (see :class:`~trilobite.radiation.arnett.Channel`).

    The deposition and emergence functions are keyword-only: they share a signature, so passing
    them positionally in the wrong order would fail silently.

    Parameters
    ----------
    clock : ~trilobite.radiation.arnett.DiffusionClock
        The clock :math:`u(t)`.
    injection_function : numba dispatcher
        ``F(t, params) -> float``: injected power in erg/s.
    deposition_function : numba dispatcher, optional
        ``D(t, params) -> float``: fraction of the power injected at ``t`` that thermalizes. Default: 1.
    emergence_function : numba dispatcher, optional
        ``E(t, params) -> float``: factor on the diffused luminosity at the requested time. Default: 1.
    window_depth : float, optional
        :math:`S`. 40 drops a fraction :math:`e^{-40} \approx 4\times10^{-18}`.

    Returns
    -------
    numba dispatcher
        ``solver(eval_time, clock_params, injection_params, deposition_params, emergence_params,
        initial_luminosity, quadrature, out)``:

        - ``eval_time``: ``(n_times, n_realizations)``, in s.
        - ``clock_params``, ``injection_params``, ``deposition_params``, ``emergence_params``:
          ``(n_realizations, k)`` each (``k`` may be 0); row ``r`` is passed to the matching function.
        - ``initial_luminosity``: ``(n_realizations,)``, :math:`L_\mathrm{init}` in erg/s.
        - ``quadrature``: a ``QuadratureRule``.
        - ``out``: ``(n_times, n_realizations)``, filled in place.
    """
    if not isinstance(clock, DiffusionClock):
        raise TypeError(f"clock must be a DiffusionClock, got {type(clock).__name__}")
    deposition_function = _unity if deposition_function is None else deposition_function
    emergence_function = _unity if emergence_function is None else emergence_function
    for name, fn in (
        ("clock.u_diff", clock.u_diff),
        ("clock.dudt", clock.dudt),
        ("clock.t_of_u", clock.t_of_u),
        ("injection_function", injection_function),
        ("deposition_function", deposition_function),
        ("emergence_function", emergence_function),
    ):
        if not isinstance(fn, CPUDispatcher):
            raise TypeError(f"{name} must be decorated with @numba.njit")
    if not window_depth > 0:
        raise ValueError(f"window_depth must be positive, got {window_depth}")
    window_depth = float(window_depth)
    u_diff, dudt, t_of_u = clock.u_diff, clock.dudt, clock.t_of_u

    @njit(parallel=True)
    def _arnett_solver(
        eval_time,
        clock_params,
        injection_params,
        deposition_params,
        emergence_params,
        initial_luminosity,
        quadrature,
        out,
    ):
        n_times, n_realizations = eval_time.shape
        nodes, one_minus_nodes, weights = quadrature.nodes, quadrature.one_minus_nodes, quadrature.weights

        # Realizations are independent: parallelize over them. (Only the outermost prange runs in
        # parallel, so the time loop is a plain range.)
        for real_idx in prange(n_realizations):
            clk_pars = clock_params[real_idx]
            inj_pars = injection_params[real_idx]
            dep_pars = deposition_params[real_idx]
            em_pars = emergence_params[real_idx]
            l_init = initial_luminosity[real_idx]

            for time_idx in range(n_times):
                t_eval = eval_time[time_idx, real_idx]

                # No light before the explosion (or at it).
                if t_eval <= 0.0:
                    out[time_idx, real_idx] = 0.0
                    continue

                # Integration window [t_min, t_eval]: from the explosion, or, at late times, from
                # window_depth clock units before t_eval, so nodes stay where the kernel is.
                u_eval = u_diff(t_eval, t_eval, clk_pars)
                t_min = 0.0
                if u_eval > window_depth:
                    t_min = t_of_u(u_eval - window_depth, clk_pars)
                t_span = t_eval - t_min

                integral = 0.0
                for node_idx in range(nodes.size):
                    t = t_min + t_span * nodes[node_idx]
                    # Kernel exp(-[u(t_eval) - u(t)]), with t_eval - t taken from 1 - x so it keeps
                    # its digits next to t_eval.
                    kernel = exp(-u_diff(t_eval, t_span * one_minus_nodes[node_idx], clk_pars))
                    deposited = injection_function(t, inj_pars) * deposition_function(t, dep_pars)
                    integral += weights[node_idx] * deposited * dudt(t, clk_pars) * kernel

                # The [0, 1] rule is mapped onto [t_min, t_eval] (factor t_span).
                out[time_idx, real_idx] = (t_span * integral + l_init * exp(-u_eval)) * emergence_function(
                    t_eval, em_pars
                )

    return _arnett_solver


@cache
def _get_solver(clock, injection_function, deposition_function, emergence_function, window_depth):
    """Process-wide cache of compiled solvers, so engines sharing a source share one compilation."""
    return build_solver(
        clock,
        injection_function,
        deposition_function=deposition_function,
        emergence_function=emergence_function,
        window_depth=window_depth,
    )


# =================================== #
# Source Channels                     #
# =================================== #
# The relaxation equation is linear in the deposited power, so an engine with several sources is solved one
# channel at a time and the luminosities are summed. Each channel is just the two-slot kernel above: an
# injection function and, if it leaks, the trapping factor.


class Channel(NamedTuple):
    r"""
    One source of power: an injection function and optional incomplete trapping.

    An engine's luminosity is the sum over its channels, each solved on the same clock.

    Parameters
    ----------
    name : str
        Identifier; keys the channel's parameters in :meth:`ArnettEngine._solve`.
    injection : numba dispatcher
        ``F(t, params) -> float``: injected power in erg/s, receiving the channel's ``n_injection`` parameters.
    n_injection : int
        Number of injection parameters.
    opacity : str, optional
        Name of the opacity (a key of the ``opacities`` given to :meth:`ArnettEngine._solve`) governing the
        channel's trapping :math:`1 - e^{-A/t^2}`. ``None`` means the channel is fully thermalized.
    """

    name: str
    injection: CPUDispatcher
    n_injection: int
    opacity: Optional[str] = None


# =================================== #
# Quadrature Rules                    #
# =================================== #
# The JIT compiled kernel solves exactly on a grid, which is determined
# by a quadrature rule. There are a variety of good quadrature rules which
# may vary depending on the engine. We provide a number of them here, which
# allows different classes to have different default quadrature and to vary
# the quadratures that the users may choose to use.


def _check_nodes(n_nodes: int, minimum: int) -> None:
    if int(n_nodes) != n_nodes or n_nodes < minimum:
        raise ValueError(f"n_nodes must be an integer >= {minimum}, got {n_nodes}")


class QuadratureRule(NamedTuple):
    r"""
    A quadrature rule on [0, 1]: :math:`\int_0^1 f(x)\,dx \approx \sum_i w_i f(x_i)`.

    ``one_minus_nodes`` is :math:`1 - x_i` computed without cancellation, so integrands that depend
    on the distance to the upper end (the diffusion kernel next to :math:`t_q`) keep their digits.
    Arrays are float64, C-contiguous and read-only. A NamedTuple of arrays unpacks positionally
    and can be passed straight into ``@njit`` code.
    """

    nodes: np.ndarray
    one_minus_nodes: np.ndarray
    weights: np.ndarray

    @classmethod
    def gauss_legendre(cls, n_nodes: int) -> "QuadratureRule":
        """Plain Gauss-Legendre: best for smooth integrands and interior features, weak at endpoints."""
        return cls.gauss_legendre_power(n_nodes, power=1)

    @classmethod
    def gauss_legendre_power(cls, n_nodes: int, power: int = 3) -> "QuadratureRule":
        r"""
        Gauss-Legendre in :math:`y` with :math:`x = y^p`.

        Nodes are packed algebraically toward :math:`x = 0`, where fast engines (short spin-down times) live.
        ``power=1`` is plain GL.
        """
        _check_nodes(n_nodes, 1)
        if int(power) != power or power < 1:
            raise ValueError(f"power must be a positive integer, got {power}")
        power = int(power)
        g, gw = np.polynomial.legendre.leggauss(n_nodes)
        y, one_minus_y, wy = 0.5 * (g + 1.0), 0.5 * (1.0 - g), 0.5 * gw
        # 1 - y^p = (1 - y)(1 + y + ... + y^(p-1)), exact near y = 1
        one_minus_x = one_minus_y * np.polynomial.polynomial.polyval(y, np.ones(power))
        return cls._frozen(y**power, one_minus_x, wy * power * y ** (power - 1))

    @classmethod
    def tanh_sinh(cls, n_nodes: int, k_max: float = 3.2) -> "QuadratureRule":
        r"""
        Tanh-sinh (double exponential) quadrature.

        Uses :math:`x = \tfrac12(1 + \tanh a)`, :math:`a = \tfrac{\pi}{2}\sinh k`, on ``n_nodes`` equally spaced
        ``k`` in ``[-k_max, k_max]``. Very strong at both endpoints, sparse in the interior. At ``k_max = 3.2``
        the end weights are below double precision.
        """
        _check_nodes(n_nodes, 2)
        if not k_max > 0:
            raise ValueError(f"k_max must be positive, got {k_max}")
        k = np.linspace(-k_max, k_max, n_nodes)
        a = 0.5 * np.pi * np.sinh(k)
        nodes = 1.0 / (1.0 + np.exp(-2.0 * a))
        one_minus = 1.0 / (1.0 + np.exp(2.0 * a))
        weights = (k[1] - k[0]) * 0.25 * np.pi * np.cosh(k) / np.cosh(a) ** 2
        return cls._frozen(nodes, one_minus, weights)

    @classmethod
    def _frozen(cls, *arrays: np.ndarray) -> "QuadratureRule":
        arrays = tuple(np.array(a, dtype=np.float64, order="C") for a in arrays)  # owned copies
        for arr in arrays:
            arr.flags.writeable = False  # the tuple is immutable; make the arrays match
        return cls(*arrays)


# =================================== #
# Base Class                          #
# =================================== #
_COMMON_UNITS = {
    "M_ej": u.g,
    "E_ej": u.erg,
    "kappa": u.cm**2 / u.g,
    "R_0": u.cm,
    "E_th_0": u.erg,
}


class ArnettEngine(ABC):
    r"""Abstract base class for Arnett-type diffusion engines."""

    _PARAMETER_UNITS: dict = dict(_COMMON_UNITS)

    # =========================================== #
    # Initialization                              #
    # =========================================== #
    def __init__(
        self,
        gamma_trapping: bool = True,
        quadrature: Optional[QuadratureRule] = None,
        window_depth: float = 40.0,
        beta: float = 13.8,
        clock: str = "homologous",
        trapping_mode: str = "deposition",
    ):
        r"""
        Instantiate the engine.

        Parameters
        ----------
        gamma_trapping : bool, optional
            Include incomplete trapping of the high-energy photons (default ``True``). If ``False`` the source
            power is fully thermalized.
        quadrature : ~trilobite.radiation.arnett.QuadratureRule, optional
            Quadrature rule used by the solver. Defaults to the subclass's choice.
        window_depth : float, optional
            Number of clock units of past injection retained by the solver. See
            :func:`~trilobite.radiation.arnett.build_solver`.
        beta : float, optional
            Arnett's dimensionless opacity-geometry constant :math:`\beta`. Default ``13.8``.
        clock : {"homologous", "finite_radius"}, optional
            The diffusion clock. Default ``"homologous"``.
        trapping_mode : {"deposition", "emergence"}, optional
            Where the incomplete trapping :math:`1 - e^{-A/t^2}` of a leaky channel is applied. ``"deposition"``
            (default) applies it to the power at the time it is deposited, inside the diffusion integral, so that
            energy that leaks out is lost for good. ``"emergence"`` applies it to the diffused luminosity of the
            channel at the time of observation, outside the integral, which is how the MOSFiT ``slsn`` model does it.
            The two agree when the channel is fully trapped. See :ref:`arnett_trapping_mode`.
        """
        if quadrature is None:
            quadrature = self._default_quadrature()
        if not isinstance(quadrature, QuadratureRule):
            raise TypeError(f"quadrature must be a QuadratureRule, got {type(quadrature).__name__}")
        if not beta > 0:
            raise ValueError(f"beta must be positive, got {beta}")
        if clock not in _CLOCKS:
            raise ValueError(f"clock must be one of {sorted(_CLOCKS)}, got {clock!r}")
        if trapping_mode not in _TRAPPING_MODES:
            raise ValueError(f"trapping_mode must be one of {list(_TRAPPING_MODES)}, got {trapping_mode!r}")

        self._gamma_trapping = bool(gamma_trapping)
        self._quadrature = quadrature
        self._window_depth = float(window_depth)
        self._beta = float(beta)
        self._clock_name = clock
        self._trapping_mode = trapping_mode

        self._channels = tuple(self._get_channels())
        if not self._channels:
            raise ValueError("an engine needs at least one channel")
        if len({ch.name for ch in self._channels}) != len(self._channels):
            raise ValueError("channel names must be unique")

        # One solver per channel, compiled lazily on first use and shared process-wide between engines
        # (and channels) with the same physics. The trapping of a leaky channel is a deposition factor (applied inside
        # the integral) or an emergence factor (applied to the output); both are the same function of time.
        emergence = self._get_emergence_function()
        if trapping_mode == "emergence" and emergence is not None and any(ch.opacity for ch in self._channels):
            raise ValueError("trapping_mode='emergence' cannot be combined with a custom emergence function")
        self._solvers = tuple(
            _get_solver(
                _CLOCKS[clock],
                ch.injection,
                _gamma_trapping if (ch.opacity is not None and trapping_mode == "deposition") else None,
                _gamma_trapping if (ch.opacity is not None and trapping_mode == "emergence") else emergence,
                self._window_depth,
            )
            for ch in self._channels
        )

    # =========================================== #
    # Properties                                  #
    # =========================================== #
    @property
    def gamma_trapping(self) -> bool:
        """bool: Whether incomplete gamma-ray trapping is included."""
        return self._gamma_trapping

    @property
    def quadrature(self) -> QuadratureRule:
        """~trilobite.radiation.arnett.QuadratureRule: The quadrature rule used by the solver."""
        return self._quadrature

    @property
    def window_depth(self) -> float:
        """float: Number of clock units of past injection retained."""
        return self._window_depth

    @property
    def beta(self) -> float:
        r"""float: Arnett's constant :math:`\beta` in the diffusion time."""
        return self._beta

    @property
    def trapping_mode(self) -> str:
        """str: Where the incomplete trapping is applied, ``"deposition"`` or ``"emergence"``."""
        return self._trapping_mode

    @property
    def clock(self) -> str:
        """str: Name of the diffusion clock."""
        return self._clock_name

    @property
    def channels(self) -> "tuple[Channel, ...]":
        """Tuple of ~trilobite.radiation.arnett.Channel: The source channels."""
        return self._channels

    # =========================================== #
    # Physics hooks (subclass interface)          #
    # =========================================== #
    @abstractmethod
    def _get_channels(self) -> "Sequence[Channel]":
        """Return the source channels. Called once, at the end of ``__init__``."""

    def _get_emergence_function(self) -> Optional[CPUDispatcher]:
        """Return an ``@njit`` emergence factor ``E(t, params)``, or ``None`` for no correction."""
        return None

    @classmethod
    def _default_quadrature(cls) -> QuadratureRule:
        """Return the default quadrature rule; override for sources with unusual time structure."""
        return QuadratureRule.gauss_legendre(96)

    # =========================================== #
    # Generic physics                             #
    # =========================================== #
    def _compute_diffusion_time_cgs(self, M_ej, E_ej, kappa):
        r"""Arnett diffusion time :math:`t_d` in s from CGS ``M_ej`` (g), ``E_ej`` (erg), ``kappa`` (cm^2/g)."""
        M_ej, E_ej, kappa = np.asarray(M_ej, float), np.asarray(E_ej, float), np.asarray(kappa, float)
        v_sc = np.sqrt(10.0 * E_ej / (3.0 * M_ej))
        return np.sqrt(2.0 * kappa * M_ej / (self._beta * c_cgs * v_sc))

    @staticmethod
    def _compute_trapping_parameter_cgs(M_ej, E_ej, kappa_gamma):
        r"""Gamma-ray trapping parameter :math:`A = 9\kappa_\gamma M^2/(40\pi E)` in s^2 (CGS inputs)."""
        M_ej, E_ej, kappa_gamma = np.asarray(M_ej, float), np.asarray(E_ej, float), np.asarray(kappa_gamma, float)
        return 9.0 * kappa_gamma * M_ej**2 / (40.0 * np.pi * E_ej)

    def compute_diffusion_time(self, M_ej, E_ej, kappa) -> u.Quantity:
        r"""
        Arnett diffusion time :math:`t_d`.

        Parameters
        ----------
        M_ej : ~astropy.units.Quantity or float
            Ejecta mass. Bare numbers are in g.
        E_ej : ~astropy.units.Quantity or float
            Ejecta kinetic energy. Bare numbers are in erg.
        kappa : ~astropy.units.Quantity or float
            Gray opacity. Bare numbers are in :math:`\mathrm{cm^2\,g^{-1}}`.

        Returns
        -------
        ~astropy.units.Quantity
            :math:`t_d` in seconds.
        """
        return (
            self._compute_diffusion_time_cgs(
                ensure_in_units(M_ej, u.g), ensure_in_units(E_ej, u.erg), ensure_in_units(kappa, u.cm**2 / u.g)
            )
            * u.s
        )

    def _clock_columns(self, M_ej, E_ej, t_d, R_0, E_th_0):
        """``(clock parameter columns, initial luminosity)`` for the chosen clock."""
        if self._clock_name == "homologous":
            if np.any(np.asarray(R_0) != 0) or np.any(np.asarray(E_th_0) != 0):
                raise ValueError("R_0 and E_th_0 must be zero unless the engine is built with clock='finite_radius'")
            return [t_d], 0.0
        if np.any(np.asarray(R_0) < 0) or np.any(np.asarray(E_th_0) < 0):
            raise ValueError("R_0 and E_th_0 must be non-negative")
        v_sc = np.sqrt(10.0 * np.asarray(E_ej, float) / (3.0 * np.asarray(M_ej, float)))
        a = np.asarray(R_0, float) / v_sc
        # Initial stored heat drains at the initial rate: L_init = E_th_0 * du/dt(0) = E_th_0 / t_diff(0).
        return [t_d, a], np.asarray(E_th_0, float) * 2.0 * a / t_d**2

    # =========================================== #
    # Core solve                                  #
    # =========================================== #
    def _solve(
        self,
        time: "_ArrayLike",
        M_ej,
        E_ej,
        kappa,
        injection: dict,
        opacities: Optional[dict] = None,
        R_0=0.0,
        E_th_0=0.0,
    ) -> np.ndarray:
        """
        Pack arguments and run the compiled solver (CGS, no units).

        Parameters
        ----------
        time : float or array-like
            Times since explosion in s. Scalar or 1D (``n_times``).
        M_ej, E_ej, kappa : float or array-like
            Ejecta mass (g), kinetic energy (erg) and optical opacity (cm^2/g). Scalar or 1D (``n_realizations``).
        injection : dict
            Channel name -> tuple of that channel's injection parameters (scalars or 1D arrays).
        opacities : dict, optional
            Opacity name -> value (cm^2/g), for each trapped channel.
        R_0, E_th_0 : float or array-like, optional
            Initial radius (cm) and thermal energy (erg); nonzero only with ``clock='finite_radius'``.

        Returns
        -------
        numpy.ndarray
            Luminosity in erg/s with shape ``(n_times, n_realizations)``; the time axis is dropped for a scalar
            ``time`` and the realization axis when every parameter is a scalar.
        """
        opacities = {} if opacities is None else opacities
        scalar_time = np.ndim(time) == 0
        time = np.atleast_1d(np.asarray(time, dtype=np.float64))
        if time.ndim != 1:
            raise ValueError(f"time must be a scalar or 1D, got shape {time.shape}")

        t_d = self._compute_diffusion_time_cgs(M_ej, E_ej, kappa)
        clock_columns, l_init = self._clock_columns(M_ej, E_ej, t_d, R_0, E_th_0)

        # Each channel's columns: its injection parameters, then the trapping parameter A if it leaks.
        channel_columns = []
        for ch in self._channels:
            columns = list(injection[ch.name])
            if len(columns) != ch.n_injection:
                raise ValueError(f"channel '{ch.name}' takes {ch.n_injection} parameters, got {len(columns)}")
            if ch.opacity is not None:
                columns.append(self._compute_trapping_parameter_cgs(M_ej, E_ej, opacities[ch.opacity]))
            channel_columns.append(columns)

        # Broadcast every column against every other, then split them back up.
        flat = [*clock_columns, l_init, *(c for columns in channel_columns for c in columns)]
        if any(np.ndim(c) > 1 for c in flat):
            raise ValueError("physical parameters must be scalars or 1D arrays")
        scalar_params = all(np.ndim(c) == 0 for c in flat)
        try:
            flat = np.broadcast_arrays(*[np.atleast_1d(np.asarray(c, dtype=np.float64)) for c in flat])
        except ValueError as err:
            raise ValueError("array-valued physical parameters must all have the same length") from err
        n_real, n_clock = flat[0].shape[0], len(clock_columns)

        def pack(columns):  # (n_real, len(columns)), C-contiguous; zero columns allowed
            return np.ascontiguousarray(np.stack(columns, axis=1)) if len(columns) else np.empty((n_real, 0))

        clock_params, initial_luminosity = pack(flat[:n_clock]), flat[n_clock]
        eval_time = np.ascontiguousarray(np.broadcast_to(time[:, None], (time.size, n_real)))
        emergence_params = np.empty((n_real, 0))
        total, buffer = np.zeros_like(eval_time), np.empty_like(eval_time)

        # The problem is linear in the deposited power, so the channels simply add. The stored heat belongs to
        # the ejecta, not to a source, so it is given to the first channel only.
        start = n_clock + 1
        for i, (solver, ch, columns) in enumerate(zip(self._solvers, self._channels, channel_columns)):
            block = flat[start : start + len(columns)]
            start += len(columns)
            trapping = pack(block[ch.n_injection :])  # the trapping parameter A, if the channel leaks
            emergence_here = trapping if self._trapping_mode == "emergence" else emergence_params
            deposition_here = emergence_params if self._trapping_mode == "emergence" else trapping
            solver(
                eval_time,
                clock_params,
                pack(block[: ch.n_injection]),
                deposition_here,
                emergence_here,
                initial_luminosity if i == 0 else np.zeros(n_real),
                self._quadrature,
                buffer,
            )
            total += buffer

        if scalar_params:
            total = total[:, 0]
        if scalar_time:
            total = total[0]
        return total

    # =========================================== #
    # Public interface                            #
    # =========================================== #
    def compute_luminosity(self, time: "_UnitBearingArrayLike", **parameters) -> u.Quantity:
        r"""
        Bolometric luminosity at one or more times.

        Unit-aware wrapper around the private CGS backend. Each parameter may be a
        :class:`~astropy.units.Quantity` (converted) or a bare number (interpreted in CGS). Time may be a scalar
        or 1D array; each physical parameter may be a scalar or a 1D array (one entry per parameter
        realization), in which case the output has shape ``(n_times, n_realizations)``. The accepted parameters,
        and their defaults, are listed in each subclass's docstring.

        Parameters
        ----------
        time : ~astropy.units.Quantity or float or numpy.ndarray
            Time(s) since explosion. Bare numbers are in s.
        **parameters
            The engine's physical parameters.

        Returns
        -------
        ~astropy.units.Quantity
            :math:`L(t)` in :math:`\mathrm{erg\,s^{-1}}`.
        """
        cgs = {}
        for key, value in parameters.items():
            if key not in self._PARAMETER_UNITS:
                raise TypeError(
                    f"{type(self).__name__} got an unexpected parameter '{key}'; "
                    f"expected {sorted(self._PARAMETER_UNITS)}"
                )
            cgs[key] = ensure_in_units(value, self._PARAMETER_UNITS[key])
        return self._compute_luminosity_cgs(ensure_in_units(time, u.s), **cgs) * (u.erg / u.s)

    def _compute_luminosity_cgs(
        self, time: "_ArrayLike", M_ej, E_ej, kappa, R_0=0.0, E_th_0=0.0, **source
    ) -> np.ndarray:
        """
        Bolometric luminosity in erg/s from CGS inputs, with no unit handling.

        ``M_ej``, ``E_ej``, ``kappa``, ``R_0`` and ``E_th_0`` are common to every engine; the remaining keyword
        arguments are the engine's source parameters and go to :meth:`_source_parameters`.
        """
        injection, opacities = self._source_parameters(**source)
        return self._solve(time, M_ej, E_ej, kappa, injection, opacities, R_0, E_th_0)

    @abstractmethod
    def _source_parameters(self, **source) -> "tuple[dict, dict]":
        """
        Turn the engine's CGS source parameters into ``(injection, opacities)`` for :meth:`_solve`.

        ``injection`` maps each channel name to the tuple of that channel's injection parameters; ``opacities``
        maps each opacity name used by a channel (``Channel.opacity``) to its value. The arguments, and their
        defaults, are the engine's physical parameters other than the common ones; they must be listed (with
        units) in ``_PARAMETER_UNITS``.
        """

    def __call__(self, time: "_UnitBearingArrayLike", **parameters) -> u.Quantity:
        """Alias for :meth:`compute_luminosity`."""
        return self.compute_luminosity(time, **parameters)

    def _switches(self) -> str:
        return (
            f"gamma_trapping={self._gamma_trapping}, trapping_mode={self._trapping_mode!r}, "
            f"clock={self._clock_name!r}, beta={self._beta}"
        )

    def __str__(self):
        return f"{type(self).__name__}({self._switches()})"

    def __repr__(self):
        return f"<{self}>"


class _PositronSwitchMixin:
    """Adds the ``positron_trapping`` switch (must be set before ``ArnettEngine.__init__`` builds channels)."""

    def __init__(self, gamma_trapping: bool = True, positron_trapping: bool = False, **kwargs):
        self._positron_trapping = bool(positron_trapping)
        super().__init__(gamma_trapping=gamma_trapping, **kwargs)

    @property
    def positron_trapping(self) -> bool:
        """bool: Whether incomplete positron trapping is included."""
        return self._positron_trapping

    def _switches(self) -> str:
        return f"gamma_trapping={self._gamma_trapping}, positron_trapping={self._positron_trapping}, " + (
            f"trapping_mode={self._trapping_mode!r}, clock={self._clock_name!r}, beta={self._beta}"
        )


# =================================== #
# 56Ni / 56Co Decay                   #
# =================================== #
# Specific heating rates and e-folding times as widely used in SN light curve work (Nadyozhin 1994).
_EPS_NI = 3.9e10  # erg / s / g of 56Ni, all in gamma rays
_EPS_CO = 6.78e9  # erg / s / g of 56Co, total (gamma rays + positron kinetic energy)
_TAU_NI = 8.8 * 86400.0  # s
_TAU_CO = 111.3 * 86400.0  # s
_F_POSITRON = 0.032  # fraction of the 56Co decay energy carried by positron kinetic energy


@njit(cache=True, inline="always")
def _ni_co_power_per_gram(t):
    """(56Ni power, 56Co power) in erg/s per gram of initial 56Ni, from the exact Bateman solution."""
    e_ni = np.exp(-t / _TAU_NI)
    e_co = np.exp(-t / _TAU_CO)
    return _EPS_NI * e_ni, _EPS_CO * (_TAU_CO / (_TAU_CO - _TAU_NI)) * (e_co - e_ni)


@njit(cache=True)
def _ni_gamma_injection(t, parameter_array):
    """Gamma-ray power in erg/s (all of the Ni decay, the non-positron part of the Co decay); ``row = [M_Ni]`` g."""
    p_ni, p_co = _ni_co_power_per_gram(t)
    return parameter_array[0] * (p_ni + (1.0 - _F_POSITRON) * p_co)


@njit(cache=True)
def _ni_positron_injection(t, parameter_array):
    """Positron kinetic-energy power in erg/s; ``row = [M_Ni]`` g."""
    _, p_co = _ni_co_power_per_gram(t)
    return parameter_array[0] * _F_POSITRON * p_co


def _nickel_channels(engine):
    return [
        Channel("nickel_gamma", _ni_gamma_injection, 1, "kappa_gamma" if engine.gamma_trapping else None),
        Channel("nickel_positron", _ni_positron_injection, 1, "kappa_positron" if engine.positron_trapping else None),
    ]


_NICKEL_UNITS = {"M_Ni": u.g, "kappa_gamma": u.cm**2 / u.g, "kappa_positron": u.cm**2 / u.g}


def _nickel_parameters(M_Ni, kappa_gamma=0.03, kappa_positron=7.0):
    """``(injection, opacities)`` for the nickel channels, from CGS inputs."""
    return (
        {"nickel_gamma": (M_Ni,), "nickel_positron": (M_Ni,)},
        {"kappa_gamma": kappa_gamma, "kappa_positron": kappa_positron},
    )


class NickelCobaltArnettEngine(_PositronSwitchMixin, ArnettEngine):
    r"""
    Arnett engine powered by the :math:`^{56}\mathrm{Ni} \to {}^{56}\mathrm{Co} \to {}^{56}\mathrm{Fe}` chain.

    The decay power for a nickel mass :math:`M_\mathrm{Ni}` is

    .. math::

        F(t) = M_\mathrm{Ni}\left[\epsilon_\mathrm{Ni}\, e^{-t/\tau_\mathrm{Ni}}
        + \epsilon_\mathrm{Co}\,\frac{\tau_\mathrm{Co}}{\tau_\mathrm{Co} - \tau_\mathrm{Ni}}
        \left(e^{-t/\tau_\mathrm{Co}} - e^{-t/\tau_\mathrm{Ni}}\right)\right],

    with :math:`\epsilon_\mathrm{Ni} = 3.9\times10^{10}` and :math:`\epsilon_\mathrm{Co} = 6.78\times10^{9}`
    :math:`\mathrm{erg\,s^{-1}\,g^{-1}}`, :math:`\tau_\mathrm{Ni} = 8.8` d and :math:`\tau_\mathrm{Co} = 111.3` d
    (Nadyozhin 1994). It is split into two channels:

    - ``"nickel_gamma"``: all of the Ni decay power and 96.8% of the Co decay power, carried by gamma rays;
      with ``gamma_trapping=True`` it is trapped as :math:`1 - e^{-A_\gamma/t^2}` using ``kappa_gamma``.
    - ``"nickel_positron"``: the remaining 3.2% of the Co power, carried by positrons. By default
      (``positron_trapping=False``) it is deposited locally and completely. With ``positron_trapping=True`` it
      is trapped as :math:`1 - e^{-A_{e^+}/t^2}` using ``kappa_positron``. This is a gray, local-deposition
      caricature: :footcite:t:`Milne1999` show that real positron escape depends on the magnetic-field
      topology and ionization of the ejecta, so ``kappa_positron`` should be regarded as an effective
      parameter.

    Parameters
    ----------
    gamma_trapping : bool, optional
        As for :class:`~trilobite.radiation.arnett.ArnettEngine`.
    positron_trapping : bool, optional
        Include incomplete trapping of the positron channel. Default ``False``.
    **kwargs
        Forwarded to :class:`~trilobite.radiation.arnett.ArnettEngine`.

    Notes
    -----
    Arguments of :meth:`compute_luminosity` (after ``time``):

    ========================  ============================================  ===========================
    ``M_ej``, ``E_ej``        ejecta mass, kinetic energy                   g, erg
    ``kappa``                 optical opacity                               cm^2/g
    ``M_Ni``                  initial 56Ni mass                             g
    ``kappa_gamma``           gamma-ray opacity (default 0.03)              cm^2/g
    ``kappa_positron``        positron opacity (default 7)                  cm^2/g
    ``R_0``, ``E_th_0``       initial radius, thermal energy (default 0)    cm, erg
    ========================  ============================================  ===========================

    ``kappa_gamma`` is used only if ``gamma_trapping``, ``kappa_positron`` only if ``positron_trapping``, and
    ``R_0`` / ``E_th_0`` only with ``clock='finite_radius'``.

    Examples
    --------
    >>> import astropy.units as u
    >>> engine = NickelCobaltArnettEngine(
    ...     positron_trapping=True
    ... )
    >>> L = engine.compute_luminosity(
    ...     [10, 20, 50] * u.day,
    ...     M_ej=2 * u.Msun,
    ...     E_ej=1e51 * u.erg,
    ...     kappa=0.1 * u.cm**2 / u.g,
    ...     M_Ni=0.3 * u.Msun,
    ... )
    """

    _PARAMETER_UNITS = {**_COMMON_UNITS, **_NICKEL_UNITS}

    def _get_channels(self):
        return _nickel_channels(self)

    def _source_parameters(self, **source):
        return _nickel_parameters(**source)


# =================================== #
# Magnetar Spin-Down                  #
# =================================== #
@njit(cache=True)
def _magnetar_injection(t, parameter_array):
    """Spin-down power L_0 (1 + t/t_sd)^-q in erg/s; ``row = [L_0, t_sd, q]`` with q = (n + 1)/(n - 1)."""
    return parameter_array[0] * (1.0 + t / parameter_array[1]) ** (-parameter_array[2])


def _magnetar_spin_down_cgs(P_spin, B_dipole, braking_index, I_ns, R_ns):
    r"""
    ``(L_0, t_sd, q)`` in (erg/s, s, 1) from CGS inputs.

    The initial spin-down rate is fixed by the vacuum-dipole formula
    :math:`|\dot\Omega_0| = B^2R^6\Omega_0^3/(6Ic^3)` for any braking index :math:`n`, so that
    :math:`L_0 = I\Omega_0|\dot\Omega_0|`, :math:`t_\mathrm{sd} = \Omega_0/[(n-1)|\dot\Omega_0|]` and
    :math:`q = (n+1)/(n-1)`.
    """
    n = np.asarray(braking_index, float)
    if np.any(n <= 1.0):
        raise ValueError("braking_index must be greater than 1")
    omega = 2.0 * np.pi / np.asarray(P_spin, float)
    B2R6 = np.asarray(B_dipole, float) ** 2 * np.asarray(R_ns, float) ** 6
    I_ns = np.asarray(I_ns, float)
    omega_dot = B2R6 * omega**3 / (6.0 * I_ns * c_cgs**3)
    return I_ns * omega * omega_dot, omega / ((n - 1.0) * omega_dot), (n + 1.0) / (n - 1.0)


def _magnetar_channel(engine):
    return Channel("magnetar", _magnetar_injection, 3, "kappa_magnetar" if engine.gamma_trapping else None)


_MAGNETAR_UNITS = {
    "P_spin": u.s,
    "B_dipole": u.G,
    "braking_index": u.dimensionless_unscaled,
    "kappa_magnetar": u.cm**2 / u.g,
    "I_ns": u.g * u.cm**2,
    "R_ns": u.cm,
}


def _magnetar_parameters(P_spin, B_dipole, braking_index=3.0, kappa_magnetar=0.03, I_ns=1e45, R_ns=1e6):
    """``(injection, opacities)`` for the magnetar channel, from CGS inputs."""
    return (
        {"magnetar": _magnetar_spin_down_cgs(P_spin, B_dipole, braking_index, I_ns, R_ns)},
        {"kappa_magnetar": kappa_magnetar},
    )


class MagnetarArnettEngine(ArnettEngine):
    r"""
    Arnett engine powered by magnetar spin-down with a free braking index.

    A neutron star of moment of inertia :math:`I`, radius :math:`R`, initial spin period :math:`P` and field
    :math:`B` spins down as :math:`\dot\Omega = -k\Omega^n`, injecting :math:`L = I\Omega|\dot\Omega|`:

    .. math::

        F(t) = L_0\left(1 + \frac{t}{t_\mathrm{sd}}\right)^{-\frac{n+1}{n-1}}, \quad
        L_0 = \frac{B^2R^6\Omega_0^4}{6c^3}, \quad
        t_\mathrm{sd} = \frac{\Omega_0}{(n-1)|\dot\Omega_0|}, \quad \Omega_0 = \frac{2\pi}{P},

    with the initial rate set by the vacuum-dipole formula
    :math:`|\dot\Omega_0| = B^2R^6\Omega_0^3/(6Ic^3)`, so that :math:`B` is an *effective* field for
    :math:`n \neq 3`. The total energy released is :math:`L_0 t_\mathrm{sd}(n-1)/2 = \tfrac12 I\Omega_0^2`.
    :math:`n = 3` is magnetic dipole spin-down (:footcite:p:`Kasen2010`); :math:`n = 5` is spin-down dominated by
    gravitational-wave emission. :footcite:t:`Omand2024` show that the braking index can change the light curve
    substantially. Unlike their model, the ejecta here do not accelerate in response to the injected energy:
    :math:`E_\mathrm{ej}` is supplied by the caller, and should include any energy added by the magnetar.

    With ``gamma_trapping=True`` the injected power is treated as a hard-photon population that leaks as
    :math:`1 - e^{-A/t^2}` with opacity ``kappa_magnetar``. Spin-down is often fast compared to the diffusion
    time, so the default quadrature packs nodes toward the explosion.

    Notes
    -----
    Arguments of :meth:`compute_luminosity` (after ``time``):

    ========================  ============================================  ===========================
    ``M_ej``, ``E_ej``        ejecta mass, kinetic energy                   g, erg
    ``kappa``                 optical opacity                               cm^2/g
    ``P_spin``                initial spin period                           s
    ``B_dipole``              (effective) polar field                       G
    ``braking_index``         braking index :math:`n > 1` (default 3)       dimensionless
    ``kappa_magnetar``        hard-photon opacity (default 0.03)            cm^2/g
    ``I_ns``, ``R_ns``        moment of inertia, radius (1e45, 1e6)         g cm^2, cm
    ``R_0``, ``E_th_0``       initial radius, thermal energy (default 0)    cm, erg
    ========================  ============================================  ===========================
    """

    _PARAMETER_UNITS = {**_COMMON_UNITS, **_MAGNETAR_UNITS}

    @classmethod
    def _default_quadrature(cls):
        return QuadratureRule.gauss_legendre_power(128, power=3)

    def _get_channels(self):
        return [_magnetar_channel(self)]

    def _source_parameters(self, **source):
        return _magnetar_parameters(**source)


# =================================== #
# Magnetar + 56Ni                     #
# =================================== #
class MagnetarNickelArnettEngine(_PositronSwitchMixin, ArnettEngine):
    r"""
    Arnett engine powered by magnetar spin-down *and* :math:`^{56}\mathrm{Ni}` decay.

    The problem is linear in the deposited power, so this is the sum of a
    :class:`~trilobite.radiation.arnett.NickelCobaltArnettEngine` and a
    :class:`~trilobite.radiation.arnett.MagnetarArnettEngine` evolving on a common clock, except that each
    source keeps its own trapping: the magnetar leaks through ``kappa_magnetar``, the nickel gamma rays through
    ``kappa_gamma``, and the positrons (optionally) through ``kappa_positron``. Its parameters are the union of
    those of the two engines.

    Parameters
    ----------
    gamma_trapping : bool, optional
        As for :class:`~trilobite.radiation.arnett.ArnettEngine`; applies to the nickel gamma rays and to the
        magnetar.
    positron_trapping : bool, optional
        Include incomplete trapping of the positron channel. Default ``False``.
    **kwargs
        Forwarded to :class:`~trilobite.radiation.arnett.ArnettEngine`.
    """

    _PARAMETER_UNITS = {**_COMMON_UNITS, **_NICKEL_UNITS, **_MAGNETAR_UNITS}

    @classmethod
    def _default_quadrature(cls):
        return QuadratureRule.gauss_legendre_power(128, power=3)

    def _get_channels(self):
        return [*_nickel_channels(self), _magnetar_channel(self)]

    def _source_parameters(self, **source):
        # Each source keeps its own parameters and defaults; hand each its share.
        nickel = {k: v for k, v in source.items() if k in _NICKEL_UNITS}
        magnetar = {k: v for k, v in source.items() if k in _MAGNETAR_UNITS}
        if len(nickel) + len(magnetar) != len(source):
            unknown = sorted(source.keys() - nickel.keys() - magnetar.keys())
            raise TypeError(f"{type(self).__name__} got unexpected parameters {unknown}")
        (ni_injection, ni_opacities), (mag_injection, mag_opacities) = (
            _nickel_parameters(**nickel),
            _magnetar_parameters(**magnetar),
        )
        return {**ni_injection, **mag_injection}, {**ni_opacities, **mag_opacities}
