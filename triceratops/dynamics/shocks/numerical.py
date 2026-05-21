r"""
Numerical shock-evolution engines for transient dynamics.

This module provides ODE-based shock engines for modelling the interaction
between expanding ejecta and an external circumstellar medium (CSM). The
implementations are intended for **semi-analytic transient models** where the
hydrodynamics are reduced to a small set of coupled ordinary differential
equations rather than evolved on a full spatial grid.

The engines in this module cover several levels of physical closure. The
simplest closures treat the shocked material as a thin shell and evolve its
radius, velocity, and swept-up mass using algebraic post-shock pressure or
momentum-conservation assumptions. The more detailed mechanical closure evolves
the shocked ejecta and shocked CSM as separate regions, tracking their masses,
internal energies, effective widths, pressures, and forward/reverse shock
locations. This allows the model to retain a minimal two-shock structure while
remaining inexpensive enough for parameter studies and inference workflows.

All public interfaces accept unit-bearing inputs through :mod:`astropy.units`
and return unit-bearing quantities. Internally, the ODE systems are evaluated in
CGS units for numerical consistency. Low-level ``*_cgs`` methods are provided
for callers that already manage units externally or need to avoid unit overhead
inside tight loops.

See Also
--------
triceratops.dynamics.shocks.core.shock_engine.ShockEngine
    Base class for shock engines.
scipy.integrate.solve_ivp
    ODE integrator used by the numerical engines.

See Also
--------
:ref:`numerical_shocks_overview`: walkthrough of the available numerical engines and when to use each closure.

:ref:`numeric_shocks_theory`: derivation of the governing ODE systems for the thin-shell, pressure-driven,
    and mechanical closures.

:ref:`shock_engines`: overview of the shock engine interface shared by all engines.
"""

from abc import ABC
from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple, Union

import numpy as np
from astropy import units as u
from scipy.integrate import quad, solve_ivp

from triceratops.dynamics.shocks.core.rankine_hugoniot import StrongColdShockConditions
from triceratops.dynamics.shocks.core.relativistic_jump_conditions import _solve_strong_cold_shock_beta
from triceratops.dynamics.shocks.core.shock_engine import ShockEngine
from triceratops.physics_utils.constants import c_cgs, k_B_cgs, m_p_cgs
from triceratops.utils.misc_utils import ensure_in_units

# NumPy compatibility: np.trapezoid added in 2.0, np.trapz removed in 2.0.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


if TYPE_CHECKING:
    from triceratops._typing import (
        _ArrayLike,
        _UnitBearingArrayLike,
        _UnitBearingScalarLike,
    )


# =========================================================== #
# Snowplow Closure Engines                                    #
# =========================================================== #
# These are numerical closures for the snowplow phase of shock evolutions.
class MomentumConservingShockEngine(ShockEngine, ABC):
    """
    Momentum conserving shock engine.

    .. note::

        This engine is not yet implemented. It will be added in a future release.
    """

    pass


class RelMomentumConservingShockEngine(ShockEngine, ABC):
    """
    Relativistic momentum conserving shock engine.

    .. note::

        This engine is not yet implemented. It will be added in a future release.
    """

    pass


# ========================================================== #
# Chevalier / Pressure Driven Shock Engines                  #
# ========================================================== #
class ThinShellShockState(NamedTuple):
    r"""
    Time-dependent state returned by :class:`PressureDrivenThinShellShockEngine`.

    The low-level CGS interface returns plain :class:`numpy.ndarray` fields in
    CGS units. The public unit-aware interface returns the same structure with
    fields converted to :class:`astropy.units.Quantity`.
    """

    radius: Union[np.ndarray, u.Quantity]
    r"""
    Shell radius :math:`R_{\rm sh}` in cm.
    """

    velocity: Union[np.ndarray, u.Quantity]
    r"""
    Shell velocity :math:`v_{\rm sh}` in cm/s.
    """

    mass: Union[np.ndarray, u.Quantity]
    r"""
    Accumulated shell mass :math:`M_{\rm sh}` in g.
    """

    post_shock_density: Union[np.ndarray, u.Quantity]
    r"""
    Immediate post-shock density :math:`\rho_s` in :math:`\mathrm{g\,cm^{-3}}`,
    derived from the strong cold-shock Rankine--Hugoniot relation applied at the
    forward shock using the upstream CSM density :math:`\rho_4(R_{\rm sh},\,t)`.
    """

    post_shock_pressure: Union[np.ndarray, u.Quantity]
    r"""
    Immediate post-shock pressure :math:`p_s` in :math:`\mathrm{dyn\,cm^{-2}}`,
    derived from the strong cold-shock Rankine--Hugoniot relation applied at the
    forward shock.
    """

    post_shock_temperature: Union[np.ndarray, u.Quantity]
    r"""
    Immediate post-shock temperature :math:`T_s` in K, derived from the ideal-gas
    relation applied at the forward shock.
    """

    thermal_energy_density: Union[np.ndarray, u.Quantity]
    r"""
    Post-shock thermal energy density
    :math:`e_{\rm th} = p_s / (\gamma - 1)` in :math:`\mathrm{erg\,cm^{-3}}`,
    evaluated at the forward shock.
    """


class RelThinShellShockState(NamedTuple):
    r"""
    Time-dependent state returned by :class:`RelPressureDrivenThinShellShockEngine`.

    The low-level CGS interface returns plain :class:`numpy.ndarray` fields.
    The public unit-aware interface attaches :class:`astropy.units.Quantity`
    units to the dimensional fields; dimensionless fields (β, Γ, η) are always
    plain arrays.
    """

    # ------------------------------------------------------------------ #
    # Evolved four-momentum state                                         #
    # ------------------------------------------------------------------ #
    radius: Union[np.ndarray, u.Quantity]
    r"""Shell radius :math:`R_{\rm sh}` in cm."""

    mass: Union[np.ndarray, u.Quantity]
    r"""Swept baryonic rest mass :math:`M_{\rm sh}` in g."""

    energy: Union[np.ndarray, u.Quantity]
    r"""Lab-frame total energy :math:`E_{\rm sh}` in erg."""

    momentum: Union[np.ndarray, u.Quantity]
    r"""Lab-frame radial momentum :math:`\Pi_{\rm sh}` in :math:`\mathrm{g\,cm\,s^{-1}}`."""

    # ------------------------------------------------------------------ #
    # Derived shell kinematics                                            #
    # ------------------------------------------------------------------ #
    beta_sh: np.ndarray
    r"""Shell velocity :math:`\beta_{\rm sh} = v_{\rm sh}/c` (dimensionless)."""

    Gamma_sh: np.ndarray
    r"""Shell Lorentz factor :math:`\Gamma_{\rm sh}` (dimensionless)."""

    w_sh: np.ndarray
    r"""Shell proper velocity :math:`w_{\rm sh} = \Gamma_{\rm sh}\beta_{\rm sh}` (dimensionless)."""

    # ------------------------------------------------------------------ #
    # Shell-frame relative velocities and Lorentz factors                 #
    # ------------------------------------------------------------------ #
    beta_rel_rs: np.ndarray
    r"""
    Upstream ejecta speed in the shell (downstream) frame,
    :math:`\beta_{\rm rel,2}` (dimensionless).
    """

    Gamma_rel_rs: np.ndarray
    r"""
    Lorentz factor of the upstream ejecta in the shell frame,
    :math:`\Gamma_{\rm rel,2} = \Gamma_{u|d}` at the reverse shock (dimensionless).
    """

    beta_rel_fs: np.ndarray
    r"""
    Upstream CSM speed in the shell (downstream) frame,
    :math:`\beta_{\rm rel,3}` (dimensionless).
    """

    Gamma_rel_fs: np.ndarray
    r"""
    Lorentz factor of the upstream CSM in the shell frame,
    :math:`\Gamma_{\rm rel,3} = \Gamma_{u|d}` at the forward shock (dimensionless).
    """

    # ------------------------------------------------------------------ #
    # Post-shock pressures                                                #
    # ------------------------------------------------------------------ #
    pressure_2: Union[np.ndarray, u.Quantity]
    r"""
    Reverse-shock post-shock pressure
    :math:`P_2 = \rho_{1,\rm sh}c^2(\Gamma_{\rm rel,2}-1)(\hat\gamma\Gamma_{\rm rel,2}+1)`
    in :math:`\mathrm{dyn\,cm^{-2}}`.
    """

    pressure_3: Union[np.ndarray, u.Quantity]
    r"""
    Forward-shock post-shock pressure
    :math:`P_3 = \rho_{4,\rm sh}c^2(\Gamma_{\rm rel,3}-1)(\hat\gamma\Gamma_{\rm rel,3}+1)`
    in :math:`\mathrm{dyn\,cm^{-2}}`.
    """

    # ------------------------------------------------------------------ #
    # Energy-loading factors                                              #
    # ------------------------------------------------------------------ #
    eta_2: np.ndarray
    r"""
    Shell-frame energy per downstream rest-mass at the reverse shock,
    :math:`\eta_2 = e_2/(\rho_2 c^2) = \Gamma_{\rm rel,2}` (dimensionless).
    """

    eta_3: np.ndarray
    r"""
    Shell-frame energy per downstream rest-mass at the forward shock,
    :math:`\eta_3 = e_3/(\rho_3 c^2) = \Gamma_{\rm rel,3}` (dimensionless).
    """


class PressureDrivenThinShellShockEngine(ShockEngine):
    r"""
    Pressure-driven thin-shell shock engine for arbitrary ejecta and CSM profiles.

    This :class:`~triceratops.dynamics.shocks.shock_engine.ShockEngine` subclass
    implements a general thin-shell shock model that collapses the shocked
    interaction region to a single shell of mass :math:`M_{\rm sh}`, radius
    :math:`R_{\rm sh}`, and velocity :math:`v_{\rm sh}`. The shell acceleration
    is driven by the net post-shock pressure difference estimated from the
    instantaneous Rankine--Hugoniot jump conditions rather than evolved
    thermodynamic quantities.

    .. hint::

        For a detailed description of the theory behind this engine, see
        :ref:`pressure_driven_thin_shell_model`.

    Notes
    -----
    The engine integrates the 3-component state vector

    .. math::

        \mathbf{y} = (R_{\rm sh},\; v_{\rm sh},\; M_{\rm sh})

    governed by

    .. math::

        \begin{aligned}
        \frac{dR_{\rm sh}}{dt} &= v_{\rm sh}, \\[4pt]
        \frac{dv_{\rm sh}}{dt} &= \frac{4\pi R_{\rm sh}^2}{M_{\rm sh}}
            \left(1-\frac{1}{\chi}\right)
            \left[\rho_1(R_{\rm sh},t)\,\Delta^2
                  - \rho_4(R_{\rm sh},t)
                    \left(v_{\rm sh}-u_4(R_{\rm sh},t)\right)^2\right], \\[4pt]
        \frac{dM_{\rm sh}}{dt} &= 4\pi R_{\rm sh}^2
            \left[\rho_1(R_{\rm sh},t)\,\Delta
                  + \rho_4(R_{\rm sh},t)
                    \left(v_{\rm sh}-u_4(R_{\rm sh},t)\right)\right],
        \end{aligned}

    where :math:`\Delta \equiv u_1(R_{\rm sh},t) - v_{\rm sh}` is the
    ejecta--shell velocity lag and
    :math:`\chi = (\hat{\gamma}+1)/(\hat{\gamma}-1)` is the strong-shock
    compression ratio.

    See Also
    --------
    :class:`~triceratops.dynamics.shocks.numerical.MechanicalShockEngine` :
        More complete closure that evolves separate internal energies for each
        shocked layer.
    :ref:`pressure_driven_thin_shell_model` : Theory derivation.
    """

    _STATE_CLASS = ThinShellShockState

    def __init__(self, mu: float = 0.5, **kwargs):
        """
        Initialize the :class:`PressureDrivenThinShellShockEngine`.

        Parameters
        ----------
        mu : float, optional
            Mean molecular weight in units of the proton mass used for the
            post-shock temperature calculation. Default is ``0.5`` (fully
            ionized hydrogen).
        kwargs
            Passed to :class:`~triceratops.dynamics.shocks.shock_engine.ShockEngine`.
        """
        super().__init__(**kwargs)
        self._mu = float(mu)

    @staticmethod
    def generate_evaluation_kernel(
        rho_1: Callable,
        rho_4: Callable,
        u_1: Callable,
        u_4: Callable,
        gamma: float = 5 / 3,
    ) -> Callable:
        r"""
        Build the ODE right-hand side for the pressure-driven thin-shell model.

        Returns a function ``kernel(t, y)`` suitable for
        :func:`scipy.integrate.solve_ivp`.

        Parameters
        ----------
        rho_1 : callable
            Upstream ejecta density :math:`\rho_1(r,\,t)` in CGS
            (:math:`\mathrm{g\,cm^{-3}}`).
        rho_4 : callable
            Upstream CSM density :math:`\rho_4(r,\,t)` in CGS
            (:math:`\mathrm{g\,cm^{-3}}`).
        u_1 : callable
            Upstream ejecta velocity :math:`u_1(r,\,t)` in cm/s.
        u_4 : callable
            Upstream CSM velocity :math:`u_4(r,\,t)` in cm/s.
        gamma : float, optional
            Adiabatic index of the shocked gas. Default ``5/3``.

        Returns
        -------
        callable
            ``kernel(t, y) -> dy/dt`` where
            ``y = [R_sh, v_sh, M_sh]``.
        """
        chi = (gamma + 1) / (gamma - 1)

        def _kernel(t, y):
            R_sh, v_sh, m = y

            rho1 = rho_1(R_sh, t)
            rho4 = rho_4(R_sh, t)
            u1 = u_1(R_sh, t)
            u4 = u_4(R_sh, t)

            Delta = u1 - v_sh
            v_fwd = v_sh - u4

            coeff = 4.0 * np.pi * R_sh**2 / m * (1.0 - 1.0 / chi)

            dR_dt = v_sh
            dv_dt = coeff * (rho1 * Delta**2 - rho4 * v_fwd**2)
            dm_dt = 4.0 * np.pi * R_sh**2 * (rho1 * Delta + rho4 * v_fwd)

            return np.array([dR_dt, dv_dt, dm_dt])

        return _kernel

    def compute_shock_properties(
        self,
        time: "_UnitBearingArrayLike",
        rho_1: Callable = None,
        rho_4: Callable = None,
        u_1: Callable = None,
        u_4: Callable = None,
        R_0: "_UnitBearingScalarLike" = 1e11 * u.cm,
        v_0: "_UnitBearingScalarLike" = 1e7 * u.cm / u.s,
        M_0: "_UnitBearingScalarLike" = 1e28 * u.g,
        t_0: "_UnitBearingScalarLike" = 1.0 * u.s,
        gamma: float = 5 / 3,
        **kwargs,
    ):
        r"""
        Compute the shock properties at the given times.

        Parameters
        ----------
        time : ~astropy.units.Quantity or array-like
            Times at which to evaluate the solution. Unit-bearing inputs are
            converted to seconds; bare floats are assumed to be in seconds.
            Must be sorted and satisfy ``time >= t_0``.
        rho_1 : callable
            Upstream ejecta density :math:`\rho_1(r,\,t)` in CGS.
        rho_4 : callable
            Upstream CSM density :math:`\rho_4(r,\,t)` in CGS.
        u_1 : callable
            Upstream ejecta velocity :math:`u_1(r,\,t)` in cm/s.
        u_4 : callable
            Upstream CSM velocity :math:`u_4(r,\,t)` in cm/s.
        R_0 : ~astropy.units.Quantity or float
            Initial shock radius. Default ``1e11 cm``.
        v_0 : ~astropy.units.Quantity or float
            Initial shock velocity. Default ``1e7 cm/s``.
        M_0 : ~astropy.units.Quantity or float
            Initial shell mass. Default ``1e28 g``.
        t_0 : ~astropy.units.Quantity or float
            Initial time. Default ``1.0 s``.
        gamma : float, optional
            Adiabatic index of the shocked gas. Default ``5/3``.
        **kwargs
            Forwarded to :func:`scipy.integrate.solve_ivp`.
            ``method`` defaults to ``'Radau'``; ``rtol`` defaults to ``1e-10``.

        Returns
        -------
        dict of str, ~astropy.units.Quantity
            - ``'radius'``: shell radius in cm.
            - ``'velocity'``: shell velocity in cm/s.
            - ``'mass'``: shell mass in g.
        """
        if isinstance(time, u.Quantity):
            time = time.to(u.s).value
        if isinstance(R_0, u.Quantity):
            R_0 = R_0.to(u.cm).value
        if isinstance(v_0, u.Quantity):
            v_0 = v_0.to(u.cm / u.s).value
        if isinstance(M_0, u.Quantity):
            M_0 = M_0.to(u.g).value
        if isinstance(t_0, u.Quantity):
            t_0 = t_0.to(u.s).value

        cgs = self._compute_shock_properties_cgs(
            time=time,
            rho_1=rho_1,
            rho_4=rho_4,
            u_1=u_1,
            u_4=u_4,
            R_0=R_0,
            v_0=v_0,
            M_0=M_0,
            t_0=t_0,
            gamma=gamma,
            **kwargs,
        )

        return ThinShellShockState(
            radius=cgs.radius * u.cm,
            velocity=cgs.velocity * (u.cm / u.s),
            mass=cgs.mass * u.g,
            post_shock_density=cgs.post_shock_density * (u.g / u.cm**3),
            post_shock_pressure=cgs.post_shock_pressure * (u.dyn / u.cm**2),
            post_shock_temperature=cgs.post_shock_temperature * u.K,
            thermal_energy_density=cgs.thermal_energy_density * (u.erg / u.cm**3),
        )

    def _compute_shock_properties_cgs(
        self,
        time: "_ArrayLike",
        rho_1: Callable = None,
        rho_4: Callable = None,
        u_1: Callable = None,
        u_4: Callable = None,
        R_0: float = 1e11,
        v_0: float = 1e7,
        M_0: float = 1e28,
        t_0: float = 1.0,
        gamma: float = 5 / 3,
        **kwargs,
    ):
        r"""
        Integrate the pressure-driven thin-shell ODEs in CGS units.

        Parameters
        ----------
        time : array-like
            Evaluation times in seconds. Must be sorted and satisfy
            ``time >= t_0``.
        rho_1, rho_4, u_1, u_4 : callable
            Upstream density and velocity functions in CGS.
        R_0, v_0, M_0 : float
            Initial shock radius (cm), velocity (cm/s), and shell mass (g).
        t_0 : float
            Initial time (s).
        gamma : float
            Adiabatic index of the shocked gas.
        **kwargs
            Forwarded to :func:`scipy.integrate.solve_ivp`.

        Returns
        -------
        dict of str, numpy.ndarray
            - ``'radius'``: shell radius in cm.
            - ``'velocity'``: shell velocity in cm/s.
            - ``'mass'``: shell mass in g.
        """
        if rho_1 is None:
            raise ValueError("An upstream ejecta density function `rho_1` must be provided.")
        if rho_4 is None:
            raise ValueError("An upstream CSM density function `rho_4` must be provided.")
        if u_1 is None:
            raise ValueError("An upstream ejecta velocity function `u_1` must be provided.")
        if u_4 is None:
            raise ValueError("An upstream CSM velocity function `u_4` must be provided.")

        time = np.atleast_1d(np.asarray(time, dtype=float))

        kernel = self.generate_evaluation_kernel(
            rho_1=rho_1,
            rho_4=rho_4,
            u_1=u_1,
            u_4=u_4,
            gamma=gamma,
        )

        y0 = np.array([R_0, v_0, M_0])
        t_span = (t_0, float(np.amax(time)))

        solver_kwargs = dict(kwargs)
        rtol = solver_kwargs.pop("rtol", 1e-10)
        method = solver_kwargs.pop("method", "Radau")

        sol = solve_ivp(
            fun=kernel,
            t_span=t_span,
            y0=y0,
            t_eval=time,
            rtol=rtol,
            method=method,
            **solver_kwargs,
        )

        if sol.status < 0:
            raise RuntimeError(
                f"ODE solver failed to integrate the pressure-driven thin-shell equations:\n{sol.message}"
            )

        R_sh, v_sh, m = sol.y

        # Post-shock thermodynamics at the forward shock via strong cold-shock RH conditions.
        # Evaluated element-by-element because the upstream callables accept scalar (r, t).
        n_steps = len(time)
        rho_4_arr = np.array([float(rho_4(R_sh[i], time[i])) for i in range(n_steps)])
        u_4_arr = np.array([float(u_4(R_sh[i], time[i])) for i in range(n_steps)])
        rh = StrongColdShockConditions._solve(v_sh, rho_4_arr, u_4_arr, gamma=gamma, mu=self._mu)

        return ThinShellShockState(
            radius=R_sh,
            velocity=v_sh,
            mass=m,
            post_shock_density=rh["post_shock_density"],
            post_shock_pressure=rh["post_shock_pressure"],
            post_shock_temperature=rh["post_shock_temperature"],
            thermal_energy_density=rh["post_shock_thermal_energy_density"],
        )


class RelPressureDrivenThinShellShockEngine(ShockEngine):
    r"""
    Relativistic pressure-driven thin-shell shock engine for arbitrary ejecta and CSM profiles.

    This engine evolves the four-component lab-frame state

    .. math::

        \mathbf{y}(t) = \bigl(R_{\rm sh},\; M_{\rm sh},\; E_{\rm sh},\; \Pi_{\rm sh}\bigr),

    where :math:`M_{\rm sh}` is the swept baryonic rest mass, :math:`E_{\rm sh}` is the
    lab-frame total energy of the shell, and :math:`\Pi_{\rm sh}` is the lab-frame radial
    momentum.  The shell velocity follows algebraically from the four-momentum ratio

    .. math::

        \beta_{\rm sh} = \frac{\Pi_{\rm sh}\,c}{E_{\rm sh}}.

    At each shock face the post-shock pressure and energy-loading coefficient are
    evaluated from the cold-upstream relativistic Rankine--Hugoniot closure using the
    Lorentz factor of the upstream fluid in the shell (downstream) frame,

    .. math::

        \Gamma_{\rm rel,i}
        =
        \left[
            1 -
            \left(
                \frac{\beta_{u,i} - \beta_{\rm sh}}
                {1 - \beta_{u,i}\,\beta_{\rm sh}}
            \right)^{\!2}
        \right]^{-1/2},

    giving

    .. math::

        P_i = \rho_{u,i}\,c^2\,(\Gamma_{\rm rel,i}-1)\,(\hat\gamma\,\Gamma_{\rm rel,i}+1),
        \qquad
        \eta_i = \Gamma_{\rm rel,i}.

    The governing equations of motion are

    .. math::

        \begin{aligned}
            \frac{dR_{\rm sh}}{dt}
            &= c\,\beta_{\rm sh},\\[4pt]
            \frac{dM_{\rm sh}}{dt}
            &= \dot M_2 + \dot M_3,\\[4pt]
            \frac{dE_{\rm sh}}{dt}
            &= \Gamma_{\rm sh}\,c^2\!\left(\eta_2\dot M_2 + \eta_3\dot M_3\right),\\[4pt]
            \frac{d\Pi_{\rm sh}}{dt}
            &= w_{\rm sh}\,c\!\left(\eta_2\dot M_2 + \eta_3\dot M_3\right)
               + 4\pi R_{\rm sh}^2\!\left(P_2 - P_3\right),
        \end{aligned}

    with baryonic rest-mass fluxes

    .. math::

        \dot M_2 = 4\pi R_{\rm sh}^2\,\rho_{1,\rm sh}\,\Gamma_{1,\rm sh}\,c
                   \left(\beta_{1,\rm sh}-\beta_{\rm sh}\right),
        \qquad
        \dot M_3 = 4\pi R_{\rm sh}^2\,\rho_{4,\rm sh}\,\Gamma_{4,\rm sh}\,c
                   \left(\beta_{\rm sh}-\beta_{4,\rm sh}\right),

    clamped to be non-negative.

    Initial conditions are specified by the intuitive triple
    :math:`(R_0,\,\beta_0,\,M_0)`.  A cold-shell initialization is assumed:

    .. math::

        E_0 = M_0\,c^2\,\Gamma_0,
        \qquad
        \Pi_0 = M_0\,c\,w_0 = M_0\,c\,\Gamma_0\,\beta_0.

    .. hint::

        For a detailed derivation see :ref:`relativistic_pressure_driven_shells`.

    See Also
    --------
    :class:`~triceratops.dynamics.shocks.numerical.PressureDrivenThinShellShockEngine` :
        Non-relativistic analogue.
    :ref:`relativistic_pressure_driven_shells` : Theory derivation.
    """

    _STATE_CLASS = RelThinShellShockState

    def __init__(
        self,
        gamma_1: float = 4.0 / 3.0,
        gamma_4: float = 4.0 / 3.0,
        beta_grid: np.ndarray = None,
        beta_grid_min: float = 1e-10,
        beta_grid_max: float = 1 - 1e-10,
        beta_grid_size: int = 1000,
        **kwargs,
    ):
        """
        Initialize the :class:`RelPressureDrivenThinShellShockEngine`.

        Parameters
        ----------
        kwargs
            Passed to :class:`~triceratops.dynamics.shocks.core.shock_engine.ShockEngine`.
        """
        # Pass off the kwargs and the adiabatic indices to the parent class. The gammas are stored as
        # attributes on the parent class since they are needed for the kernel generation.
        super().__init__(gamma_1=gamma_1, gamma_4=gamma_4, **kwargs)

        # Assign the attributes.
        self._gamma_ad_1 = float(gamma_1)
        self._gamma_ad_4 = float(gamma_4)

        # Generate a grid of beta values
        if beta_grid is not None:
            self._beta_grid = beta_grid
        else:
            self._beta_grid = np.linspace(beta_grid_min, beta_grid_max, beta_grid_size)

        # Do some checks on the beta grid to ensure that it is valid for use generating the
        # shock properties. The grid must be strictly between -1 and 1, and it must be sorted.
        if np.any(self._beta_grid < 0) or np.any(self._beta_grid > 1):
            raise ValueError("The beta grid must be strictly between 0 and 1.")

        # Check that the beta grid is sorted in ascending order.
        if not np.all(np.diff(self._beta_grid) > 0):
            raise ValueError("The beta grid must be sorted in ascending order.")

        # Pre-compute the grids.
        self._beta_2_grid = [_solve_strong_cold_shock_beta(_beta, self._gamma_ad_1) for _beta in self._beta_grid]
        self._beta_3_grid = [_solve_strong_cold_shock_beta(_beta, self._gamma_ad_4) for _beta in self._beta_grid]

    def generate_evaluation_kernel(
        self,
        rho_1: Callable,
        rho_4: Callable,
        beta_1: Callable,
        beta_4: Callable,
    ) -> Callable:
        r"""
        Build the ODE right-hand side for the relativistic pressure-driven thin-shell model.

        Returns a function ``kernel(t, y)`` suitable for
        :func:`scipy.integrate.solve_ivp`.

        Parameters
        ----------
        rho_1 : callable
            Upstream ejecta comoving rest-mass density :math:`\rho_1(r,\,t)` in CGS
            (:math:`\mathrm{g\,cm^{-3}}`).
        rho_4 : callable
            Upstream CSM comoving rest-mass density :math:`\rho_4(r,\,t)` in CGS.
        beta_1 : callable
            Lab-frame ejecta velocity :math:`\beta_1(r,\,t) = v_1/c` (dimensionless).
        beta_4 : callable
            Lab-frame CSM velocity :math:`\beta_4(r,\,t) = v_4/c` (dimensionless).

        Returns
        -------
        callable
            ``kernel(t, y) -> dy/dt`` where
            ``y = [R_sh, M_sh, E_sh, Pi_sh]`` in CGS.
        """
        # Load important state variables into the local scope so that
        # the kernel has access to them.
        _beta_grid = self._beta_grid
        _beta_2_grid = self._beta_2_grid
        _beta_3_grid = self._beta_3_grid
        _gamma_ad_4 = self._gamma_ad_4
        _gamma_ad_1 = self._gamma_ad_1

        # Set the minimum shock-frame velocity and density for a side of the
        # shock to be considered active. Below this threshold, we do not attempt
        # to reconstruct the downstream state because the jump conditions become
        # numerically singular and the mass flux should vanish anyway.
        _MIN_SHOCK_BETA = _beta_grid[0]
        _MIN_RHO = 0.0

        def _kernel(t, y):
            # Extract the current state from the input array.
            R_sh, M_sh, E_sh, Pi_sh = y

            # Using the current shock properties, compute the observer-frame
            # shock properties.
            # We protect this to ensure that we do not end up in numerically
            # ill posed scenarios.
            SHOCK_BETA = np.clip(Pi_sh * c_cgs / E_sh, 1e-10, 1.0 - 1e-10)
            SHOCK_GAMMA = 1.0 / np.sqrt(1.0 - SHOCK_BETA**2)
            SHOCK_PROPER_VELOCITY = SHOCK_GAMMA * SHOCK_BETA

            # Evaluate the upstream fluid properties to compute the
            # properties for the jump conditions. The density is the **proper**
            # density and the velocity is the **lab-frame** velocity.
            rho_1_proper, rho_4_proper = rho_1(R_sh, t), rho_4(R_sh, t)
            beta_1_lab, beta_4_lab = beta_1(R_sh, t), beta_4(R_sh, t)

            gamma_1_lab = 1.0 / np.sqrt(1.0 - beta_1_lab**2)
            gamma_4_lab = 1.0 / np.sqrt(1.0 - beta_4_lab**2)

            # Convert the gammas and betas into the correct shock
            # frame so that they can be used in the jump conditions.
            beta_1_shock_frame = np.abs((beta_1_lab - SHOCK_BETA) / (1.0 - beta_1_lab * SHOCK_BETA))
            beta_4_shock_frame = np.abs((SHOCK_BETA - beta_4_lab) / (1.0 - SHOCK_BETA * beta_4_lab))

            gamma_1_shock_frame = 1.0 / np.sqrt(1.0 - beta_1_shock_frame**2)
            gamma_4_shock_frame = 1.0 / np.sqrt(1.0 - beta_4_shock_frame**2)

            # Determine whether the reverse and forward shocks are active.
            # If a side is inactive, we do not reconstruct its downstream state;
            # this avoids singular compression ratios and eta * dM = inf * 0
            # failures in the energy and momentum equations.
            reverse_shock_active = (
                (rho_1_proper > _MIN_RHO) and (beta_1_lab > SHOCK_BETA) and (beta_1_shock_frame > _MIN_SHOCK_BETA)
            )

            forward_shock_active = (
                (rho_4_proper > _MIN_RHO) and (SHOCK_BETA > beta_4_lab) and (beta_4_shock_frame > _MIN_SHOCK_BETA)
            )

            # Initialize the reverse-shock contribution. These neutral values are
            # used when there is no physical reverse shock / no ejecta inflow.
            P2 = 0.0
            eta_2 = 1.0
            dM2_dt = 0.0

            if reverse_shock_active:
                # Using the interpolator, compute the beta_2 value for the reverse shock.
                beta_2_shock_frame = np.interp(
                    beta_1_shock_frame,
                    _beta_grid,
                    _beta_2_grid,
                )

                # Protect the physical branch. The downstream shock-frame speed must
                # satisfy 0 < beta_2 < beta_1 for a compressive shock.
                beta_2_shock_frame = np.clip(
                    beta_2_shock_frame,
                    _beta_grid[0],
                    np.nextafter(beta_1_shock_frame, 0.0),
                )

                gamma_2_shock_frame = 1.0 / np.sqrt(1.0 - beta_2_shock_frame**2)

                # Apply the shock-frame conditions to determine the compression ratio:
                # rho_2 / rho_1 = (Gamma_1s * beta_1s) / (Gamma_2s * beta_2s)
                compression_ratio_2 = (gamma_1_shock_frame * beta_1_shock_frame) / (
                    gamma_2_shock_frame * beta_2_shock_frame
                )

                rho_2_proper = compression_ratio_2 * rho_1_proper

                # Compute the pressure in the post-shock ejecta region from the
                # shock-frame momentum jump.
                P2 = (
                    rho_1_proper
                    * c_cgs**2
                    * gamma_1_shock_frame**2
                    * beta_1_shock_frame**2
                    * (1.0 - beta_2_shock_frame / beta_1_shock_frame)
                )

                # Compute the shell-frame energy loading coefficient. This is the
                # downstream total energy per unit downstream rest-mass energy.
                eta_2 = 1.0 + P2 / ((_gamma_ad_1 - 1.0) * rho_2_proper * c_cgs**2)

                # Compute the baryonic rest-mass flux through the reverse shock.
                dM2_dt = 4.0 * np.pi * R_sh**2 * rho_1_proper * gamma_1_lab * c_cgs * (beta_1_lab - SHOCK_BETA)

            # Initialize the forward-shock contribution. These neutral values are
            # used when there is no physical forward shock / no CSM inflow.
            P3 = 0.0
            eta_3 = 1.0
            dM3_dt = 0.0

            if forward_shock_active:
                # Using the interpolator, compute the beta_3 value for the forward shock.
                beta_3_shock_frame = np.interp(
                    beta_4_shock_frame,
                    _beta_grid,
                    _beta_3_grid,
                )

                # Protect the physical branch. The downstream shock-frame speed must
                # satisfy 0 < beta_3 < beta_4 for a compressive shock.
                beta_3_shock_frame = np.clip(
                    beta_3_shock_frame,
                    _beta_grid[0],
                    np.nextafter(beta_4_shock_frame, 0.0),
                )

                gamma_3_shock_frame = 1.0 / np.sqrt(1.0 - beta_3_shock_frame**2)

                # Apply the shock-frame conditions to determine the compression ratio:
                # rho_3 / rho_4 = (Gamma_4s * beta_4s) / (Gamma_3s * beta_3s)
                compression_ratio_3 = (gamma_4_shock_frame * beta_4_shock_frame) / (
                    gamma_3_shock_frame * beta_3_shock_frame
                )

                rho_3_proper = compression_ratio_3 * rho_4_proper

                # Compute the pressure in the post-shock CSM region from the
                # shock-frame momentum jump.
                P3 = (
                    rho_4_proper
                    * c_cgs**2
                    * gamma_4_shock_frame**2
                    * beta_4_shock_frame**2
                    * (1.0 - beta_3_shock_frame / beta_4_shock_frame)
                )

                # Compute the shell-frame energy loading coefficient. This is the
                # downstream total energy per unit downstream rest-mass energy.
                eta_3 = 1.0 + P3 / ((_gamma_ad_4 - 1.0) * rho_3_proper * c_cgs**2)

                # Compute the baryonic rest-mass flux through the forward shock.
                dM3_dt = 4.0 * np.pi * R_sh**2 * rho_4_proper * gamma_4_lab * c_cgs * (SHOCK_BETA - beta_4_lab)

            # Clamp the mass-loading rates for safety. In normal operation the
            # active-shock guards above should already enforce non-negative fluxes.
            dM2_dt = np.maximum(dM2_dt, 0.0)
            dM3_dt = np.maximum(dM3_dt, 0.0)

            # Shell-frame energy loading rate.
            E_prime_dot = c_cgs**2 * (eta_2 * dM2_dt + eta_3 * dM3_dt)

            # Equations of motion.
            dR_dt = c_cgs * SHOCK_BETA
            dM_dt = dM2_dt + dM3_dt
            dE_dt = SHOCK_GAMMA * E_prime_dot
            dPi_dt = SHOCK_PROPER_VELOCITY * c_cgs * (eta_2 * dM2_dt + eta_3 * dM3_dt) + 4.0 * np.pi * R_sh**2 * (
                P2 - P3
            )

            return np.array([dR_dt, dM_dt, dE_dt, dPi_dt])

        return _kernel

    def compute_shock_properties(
        self,
        time: "_UnitBearingArrayLike",
        rho_1: Callable,
        rho_4: Callable,
        beta_1: Callable,
        beta_4: Callable,
        R_0: "_UnitBearingScalarLike" = 1e11 * u.cm,
        beta_0: float = 0.1,
        M_0: "_UnitBearingScalarLike" = 1e28 * u.g,
        t_0: "_UnitBearingScalarLike" = 1.0 * u.s,
        gamma: float = 4.0 / 3.0,
        **kwargs,
    ) -> RelThinShellShockState:
        r"""
        Compute the relativistic shock properties at the given times.

        Parameters
        ----------
        time : ~astropy.units.Quantity or array-like
            Times at which to evaluate the solution. Unit-bearing inputs are
            converted to seconds; bare floats are assumed to be in seconds.
            Must be sorted and satisfy ``time >= t_0``.
        rho_1 : callable
            Upstream ejecta comoving rest-mass density :math:`\rho_1(r,\,t)` in CGS.
        rho_4 : callable
            Upstream CSM comoving rest-mass density :math:`\rho_4(r,\,t)` in CGS.
        beta_1 : callable
            Lab-frame ejecta velocity :math:`\beta_1(r,\,t)` (dimensionless).
        beta_4 : callable
            Lab-frame CSM velocity :math:`\beta_4(r,\,t)` (dimensionless).
        R_0 : ~astropy.units.Quantity or float
            Initial shell radius. Default ``1e11 cm``.
        beta_0 : float
            Initial shell speed :math:`\beta_0 = v_0/c`. Default ``0.1``.
        M_0 : ~astropy.units.Quantity or float
            Initial swept baryonic rest mass. Default ``1e28 g``.
        t_0 : ~astropy.units.Quantity or float
            Initial time. Default ``1.0 s``.
        gamma : float, optional
            Adiabatic index :math:`\hat\gamma`. Default ``4/3``.
        **kwargs
            Forwarded to :func:`scipy.integrate.solve_ivp`.
            ``method`` defaults to ``'Radau'``; ``rtol`` defaults to ``1e-10``.

        Returns
        -------
        ~triceratops.dynamics.shocks.numerical.RelThinShellShockState
            Named tuple with unit-bearing fields.
        """
        if isinstance(time, u.Quantity):
            time = time.to(u.s).value
        if isinstance(R_0, u.Quantity):
            R_0 = R_0.to(u.cm).value
        if isinstance(M_0, u.Quantity):
            M_0 = M_0.to(u.g).value
        if isinstance(t_0, u.Quantity):
            t_0 = t_0.to(u.s).value

        cgs = self._compute_shock_properties_cgs(
            time=time,
            rho_1=rho_1,
            rho_4=rho_4,
            beta_1=beta_1,
            beta_4=beta_4,
            R_0=float(R_0),
            beta_0=float(beta_0),
            M_0=float(M_0),
            t_0=float(t_0),
            **kwargs,
        )

        return RelThinShellShockState(
            radius=cgs.radius * u.cm,
            mass=cgs.mass * u.g,
            energy=cgs.energy * u.erg,
            momentum=cgs.momentum * (u.g * u.cm / u.s),
            beta_sh=cgs.beta_sh,
            Gamma_sh=cgs.Gamma_sh,
            w_sh=cgs.w_sh,
            beta_rel_rs=cgs.beta_rel_rs,
            Gamma_rel_rs=cgs.Gamma_rel_rs,
            beta_rel_fs=cgs.beta_rel_fs,
            Gamma_rel_fs=cgs.Gamma_rel_fs,
            pressure_2=cgs.pressure_2 * (u.dyn / u.cm**2),
            pressure_3=cgs.pressure_3 * (u.dyn / u.cm**2),
            eta_2=cgs.eta_2,
            eta_3=cgs.eta_3,
        )

    def _compute_shock_properties_cgs(
        self,
        time: "_ArrayLike",
        rho_1: Callable,
        rho_4: Callable,
        beta_1: Callable,
        beta_4: Callable,
        R_0: float = 1e11,
        beta_0: float = 0.1,
        M_0: float = 1e28,
        t_0: float = 1.0,
        **kwargs,
    ) -> RelThinShellShockState:
        r"""
        Integrate the relativistic pressure-driven thin-shell ODEs in CGS units.

        Parameters
        ----------
        time : array-like
            Evaluation times in seconds.
        rho_1, rho_4 : callable
            Upstream density functions in CGS.
        beta_1, beta_4 : callable
            Upstream dimensionless velocity functions.
        R_0 : float
            Initial shell radius in cm.
        beta_0 : float
            Initial shell speed :math:`\beta_0 = v_0/c`.
        M_0 : float
            Initial swept baryonic rest mass in g.
        t_0 : float
            Initial time in seconds.
        gamma : float
            Adiabatic index :math:`\hat\gamma`.
        **kwargs
            Forwarded to :func:`scipy.integrate.solve_ivp`.

        Returns
        -------
        ~triceratops.dynamics.shocks.numerical.RelThinShellShockState
            CGS-valued named tuple.
        """
        time = np.atleast_1d(np.asarray(time, dtype=float))

        # Cold-shell initial four-momentum: E = M c² Γ₀, Π = M c w₀.
        Gamma_0 = 1.0 / np.sqrt(1.0 - beta_0**2)
        E_0 = M_0 * c_cgs**2 * Gamma_0
        Pi_0 = M_0 * c_cgs * Gamma_0 * beta_0

        kernel = self.generate_evaluation_kernel(rho_1=rho_1, rho_4=rho_4, beta_1=beta_1, beta_4=beta_4)

        y0 = np.array([R_0, M_0, E_0, Pi_0])
        t_span = (t_0, float(np.amax(time)))

        solver_kwargs = dict(kwargs)
        rtol = solver_kwargs.pop("rtol", 1e-10)
        method = solver_kwargs.pop("method", "Radau")

        sol = solve_ivp(
            fun=kernel,
            t_span=t_span,
            y0=y0,
            t_eval=time,
            rtol=rtol,
            method=method,
            **solver_kwargs,
        )

        if sol.status < 0:
            raise RuntimeError(f"ODE solver failed to integrate the relativistic thin-shell equations:\n{sol.message}")

        R_sh, M_sh, E_sh, Pi_sh = sol.y

        # Derived shell kinematics.
        beta_sh = np.clip(Pi_sh * c_cgs / E_sh, -(1.0 - 1e-10), 1.0 - 1e-10)
        Gamma_sh = 1.0 / np.sqrt(1.0 - beta_sh**2)
        w_sh = Gamma_sh * beta_sh

        # Post-processed shock diagnostics at each output time.
        n_steps = len(time)
        rho1_arr = np.array([float(rho_1(R_sh[i], time[i])) for i in range(n_steps)])
        rho4_arr = np.array([float(rho_4(R_sh[i], time[i])) for i in range(n_steps)])
        b1_arr = np.array([float(beta_1(R_sh[i], time[i])) for i in range(n_steps)])
        b4_arr = np.array([float(beta_4(R_sh[i], time[i])) for i in range(n_steps)])

        denom_rs = 1.0 - b1_arr * beta_sh
        denom_fs = 1.0 - beta_sh * b4_arr
        beta_rel_rs = np.abs((b1_arr - beta_sh) / denom_rs)
        beta_rel_fs = np.abs((beta_sh - b4_arr) / denom_fs)
        Gamma_rel_rs = 1.0 / np.sqrt(1.0 - beta_rel_rs**2)
        Gamma_rel_fs = 1.0 / np.sqrt(1.0 - beta_rel_fs**2)

        pressure_2 = rho1_arr * c_cgs**2 * (Gamma_rel_rs - 1.0) * (self._gamma_ad_1 * Gamma_rel_rs + 1.0)
        pressure_3 = rho4_arr * c_cgs**2 * (Gamma_rel_fs - 1.0) * (self._gamma_ad_4 * Gamma_rel_fs + 1.0)

        return RelThinShellShockState(
            radius=R_sh,
            mass=M_sh,
            energy=E_sh,
            momentum=Pi_sh,
            beta_sh=beta_sh,
            Gamma_sh=Gamma_sh,
            w_sh=w_sh,
            beta_rel_rs=beta_rel_rs,
            Gamma_rel_rs=Gamma_rel_rs,
            beta_rel_fs=beta_rel_fs,
            Gamma_rel_fs=Gamma_rel_fs,
            pressure_2=pressure_2,
            pressure_3=pressure_3,
            eta_2=Gamma_rel_rs,
            eta_3=Gamma_rel_fs,
        )


# =========================================================== #
# Mechanical Shock Engines                                    #
# =========================================================== #
# The mechanical shock engine classes evolve separate
# internal energies for the shocked ejecta and shocked CSM layers, allowing them to retain
# a minimal two-shock structure while remaining inexpensive
# enough for parameter studies and inference workflows.

# Fraction of M1_total at which the reverse shock is considered to have crossed all ejecta.
# At M2 >= (1 - _EJECTA_MASS_TOL) * M1_total the RS is treated as fully traversed.
_EJECTA_MASS_TOL = 1e-4


class MechanicalShockState(NamedTuple):
    r"""
    Time-dependent state returned by :class:`MechanicalShockEngine`.

    The state separates three conceptually distinct groups of quantities:

    1. The evolved dynamical variables integrated by the ODE solver.
    2. Region-averaged thermodynamic quantities derived from the evolved
       masses, energies, and effective volumes.
    3. Instantaneous shock-front jump diagnostics derived from the current
       Rankine--Hugoniot shock conditions.

    Region 2 denotes shocked ejecta, bounded externally by the contact
    discontinuity and internally by the reverse shock. Region 3 denotes shocked
    CSM, bounded internally by the contact discontinuity and externally by the
    forward shock.
    """

    # ------------------------------------------------------------------ #
    # Evolved dynamical state                                             #
    # ------------------------------------------------------------------ #
    radius: Union[np.ndarray, u.Quantity]
    r"""
    Contact-discontinuity radius :math:`R_{\rm cd}` in cm.
    """

    velocity: Union[np.ndarray, u.Quantity]
    r"""
    Contact-discontinuity velocity :math:`v_{\rm cd}` in cm/s.
    """

    mass_2: Union[np.ndarray, u.Quantity]
    r"""
    Shocked ejecta mass :math:`M_2` in g.
    """

    mass_3: Union[np.ndarray, u.Quantity]
    r"""
    Shocked CSM mass :math:`M_3` in g.
    """

    energy_2: Union[np.ndarray, u.Quantity]
    r"""
    Internal energy of the shocked ejecta, :math:`U_2`, in erg.
    """

    energy_3: Union[np.ndarray, u.Quantity]
    r"""
    Internal energy of the shocked CSM, :math:`U_3`, in erg.
    """

    width_2: Union[np.ndarray, u.Quantity]
    r"""
    Effective shocked-ejecta width :math:`\Delta_2 = R_{\rm cd}-R_{\rm rs}` in cm.
    """

    width_3: Union[np.ndarray, u.Quantity]
    r"""
    Effective shocked-CSM width :math:`\Delta_3 = R_{\rm fs}-R_{\rm cd}` in cm.
    """

    # ------------------------------------------------------------------ #
    # Shock geometry and kinematics                                       #
    # ------------------------------------------------------------------ #
    radius_rs: Union[np.ndarray, u.Quantity]
    r"""
    Reverse-shock radius :math:`R_{\rm rs} = R_{\rm cd}-\Delta_2` in cm.
    """

    radius_fs: Union[np.ndarray, u.Quantity]
    r"""
    Forward-shock radius :math:`R_{\rm fs} = R_{\rm cd}+\Delta_3` in cm.
    """

    velocity_rs: Union[np.ndarray, u.Quantity]
    r"""
    Reverse-shock speed :math:`D_{\rm rs}` in cm/s.
    """

    velocity_fs: Union[np.ndarray, u.Quantity]
    r"""
    Forward-shock speed :math:`D_{\rm fs}` in cm/s.
    """

    # ------------------------------------------------------------------ #
    # Region-averaged thermodynamics                                     #
    # ------------------------------------------------------------------ #
    volume_2: Union[np.ndarray, u.Quantity]
    r"""
    Effective shocked-ejecta volume :math:`V_2 \simeq 4\pi R_{\rm cd}^2\Delta_2`
    in cm³.
    """

    volume_3: Union[np.ndarray, u.Quantity]
    r"""
    Effective shocked-CSM volume :math:`V_3 \simeq 4\pi R_{\rm cd}^2\Delta_3`
    in cm³.
    """

    density_2: Union[np.ndarray, u.Quantity]
    r"""
    Volume-averaged shocked-ejecta density :math:`\bar{\rho}_2 = M_2/V_2`
    in g/cm³.
    """

    density_3: Union[np.ndarray, u.Quantity]
    r"""
    Volume-averaged shocked-CSM density :math:`\bar{\rho}_3 = M_3/V_3`
    in g/cm³.
    """

    pressure_2: Union[np.ndarray, u.Quantity]
    r"""
    Volume-averaged shocked-ejecta pressure
    :math:`P_2 = (\gamma_2-1)U_2/V_2` in dyn/cm².
    """

    pressure_3: Union[np.ndarray, u.Quantity]
    r"""
    Volume-averaged shocked-CSM pressure
    :math:`P_3 = (\gamma_3-1)U_3/V_3` in dyn/cm².
    """

    temperature_2: Union[np.ndarray, u.Quantity]
    r"""
    Volume-averaged shocked-ejecta temperature

    .. math::

        T_2 =
        \frac{\mu_2 m_p}{k_B}
        \frac{P_2}{\bar{\rho}_2}
        =
        \frac{\mu_2 m_p}{k_B}
        (\gamma_2-1)\frac{U_2}{M_2}.

    Returned in K.
    """

    temperature_3: Union[np.ndarray, u.Quantity]
    r"""
    Volume-averaged shocked-CSM temperature

    .. math::

        T_3 =
        \frac{\mu_3 m_p}{k_B}
        \frac{P_3}{\bar{\rho}_3}
        =
        \frac{\mu_3 m_p}{k_B}
        (\gamma_3-1)\frac{U_3}{M_3}.

    Returned in K.
    """

    thermal_energy_density_2: Union[np.ndarray, u.Quantity]
    r"""
    Volume-averaged shocked-ejecta thermal energy density
    :math:`e_2 = U_2/V_2` in erg/cm³.
    """

    thermal_energy_density_3: Union[np.ndarray, u.Quantity]
    r"""
    Volume-averaged shocked-CSM thermal energy density
    :math:`e_3 = U_3/V_3` in erg/cm³.
    """

    sound_speed_2: Union[np.ndarray, u.Quantity]
    r"""
    Volume-averaged shocked-ejecta sound speed

    .. math::

        c_{s,2} =
        \sqrt{\gamma_2 P_2/\bar{\rho}_2}
        =
        \sqrt{\gamma_2(\gamma_2-1)U_2/M_2}.

    Returned in cm/s.
    """

    sound_speed_3: Union[np.ndarray, u.Quantity]
    r"""
    Volume-averaged shocked-CSM sound speed

    .. math::

        c_{s,3} =
        \sqrt{\gamma_3 P_3/\bar{\rho}_3}
        =
        \sqrt{\gamma_3(\gamma_3-1)U_3/M_3}.

    Returned in cm/s.
    """

    # ------------------------------------------------------------------ #
    # Instantaneous reverse-shock jump diagnostics                        #
    # ------------------------------------------------------------------ #
    jump_density_rs: Union[np.ndarray, u.Quantity]
    r"""
    Immediate post-shock density at the reverse shock in g/cm³.

    This is an instantaneous Rankine--Hugoniot jump diagnostic, not the
    volume-averaged density of Region 2.
    """

    jump_pressure_rs: Union[np.ndarray, u.Quantity]
    r"""
    Immediate post-shock pressure at the reverse shock in dyn/cm².

    This is an instantaneous Rankine--Hugoniot jump diagnostic, not the
    volume-averaged pressure of Region 2.
    """

    jump_temperature_rs: Union[np.ndarray, u.Quantity]
    r"""
    Immediate post-shock temperature at the reverse shock in K.

    This is an instantaneous shock-front diagnostic. It should be interpreted as
    the injection temperature of newly shocked ejecta, not the evolved Region-2
    temperature.
    """

    jump_thermal_energy_density_rs: Union[np.ndarray, u.Quantity]
    r"""
    Immediate post-shock thermal energy density at the reverse shock in
    erg/cm³.
    """

    # ------------------------------------------------------------------ #
    # Instantaneous forward-shock jump diagnostics                        #
    # ------------------------------------------------------------------ #
    jump_density_fs: Union[np.ndarray, u.Quantity]
    r"""
    Immediate post-shock density at the forward shock in g/cm³.

    This is an instantaneous Rankine--Hugoniot jump diagnostic, not the
    volume-averaged density of Region 3.
    """

    jump_pressure_fs: Union[np.ndarray, u.Quantity]
    r"""
    Immediate post-shock pressure at the forward shock in dyn/cm².

    This is an instantaneous Rankine--Hugoniot jump diagnostic, not the
    volume-averaged pressure of Region 3.
    """

    jump_temperature_fs: Union[np.ndarray, u.Quantity]
    r"""
    Immediate post-shock temperature at the forward shock in K.

    This is an instantaneous shock-front diagnostic. It should be interpreted as
    the injection temperature of newly shocked CSM, not the evolved Region-3
    temperature.
    """

    jump_thermal_energy_density_fs: Union[np.ndarray, u.Quantity]
    r"""
    Immediate post-shock thermal energy density at the forward shock in
    erg/cm³.
    """


class MechanicalShockEngine(ShockEngine):
    r"""
    Non-relativistic mechanical shock engine.

    This class implements the non-relativistic form of the dynamical shock model of
    :footcite:t:`beloborodovMechanicalModelRelativistic2006a`, which self-consistently tracks both
    the forward and reverse shocks and the internal energy of the shocked material. We follow the simplifications
    described in :footcite:t:`wangVegasAfterglowHighperformanceFramework2026`.

    The engine integrates the 8-component state vector

    .. math::

        \mathbf{y} = (R_{\rm cd},\; v_{\rm cd},\; M_2,\; M_3,\; U_2,\; U_3,\; \Delta_2,\; \Delta_3),

    where subscript 2 denotes the shocked ejecta layer (between the reverse shock and the
    contact discontinuity) and subscript 3 denotes the shocked CSM layer (between the
    contact discontinuity and the forward shock).  The governing equations are

    .. math::

        \dot{R}_{\rm cd} &= v_{\rm cd}, \\
        \dot{v}_{\rm cd} &= \frac{4\pi R_{\rm cd}^2 (P_2 - P_3)}{M_2 + M_3}, \\
        \dot{M}_i       &= 4\pi R_{{\rm s},i}^2\,\rho_i^{\rm up}\,v_{{\rm rel},i}, \\
        \dot{U}_i       &= \dot{U}_{{\rm sh},i} + \dot{U}_{{\rm ad},i} + \dot{U}_{{\rm rad},i}, \\
        \dot{\Delta}_i  &= c_{s,i},

    with layer-averaged pressures :math:`P_i = (\gamma_i-1)U_i/V_i`
    (:math:`V_i = 4\pi R_{\rm cd}^2 \Delta_i`), forward-shock width closure
    :math:`c_{s,3} = \sqrt{\gamma_3(\gamma_3-1)U_3/M_3}`, reverse-shock width closure
    :math:`\dot{\Delta}_2 = (u_1(R_{\rm rs},t) - v_{\rm cd})/(\chi_2-1)`, shock speeds
    :math:`D_{\rm rs} = v_{\rm cd} - \dot{\Delta}_2` and :math:`D_{\rm fs} = v_{\rm cd} + c_{s,3}`,
    Rankine--Hugoniot shock heating

    .. math::

        \dot{U}_{{\rm sh},i} = \frac{1}{\gamma_i - 1}\frac{\chi_i - 1}{\chi_i^2}
                               v_{{\rm rel},i}^2\,\dot{M}_i
        \qquad (\chi_i \equiv \tfrac{\gamma_i+1}{\gamma_i-1}),

    and adiabatic expansion losses

    .. math::

        \dot{U}_{{\rm ad},i} = -(\gamma_i-1)\,U_i
        \!\left(\frac{2\,v_{\rm cd}}{R_{\rm cd}} + \frac{c_{s,i}}{\Delta_i}\right).

    .. note::

        For a more detailed description of the theory behind this engine, see :ref:`mechanical_internal_energy_model`.


    See Also
    --------
    :ref:`mechanical_internal_energy_model` : Theory derivation.
    :class:`~triceratops.dynamics.shocks.numerical.PressureDrivenThinShellShockEngine` :
        Simpler thin-shell closure with algebraic pressures.

    References
    ----------
    .. footbibliography::
    """

    _STATE_CLASS = MechanicalShockState

    # ------------------------------------------------------------------ #
    # Instantiation and Dunder Methods                                   #
    # ------------------------------------------------------------------ #
    def __init__(self, mu_2: float = 0.5, mu_3: float = 0.5, **kwargs):
        """
        Instantiate the :class:`MechanicalShockEngine`.

        Parameters
        ----------
        mu_2 : float, optional
            Mean molecular weight in units of the proton mass for the shocked
            ejecta (Region 2), used for the reverse-shock post-shock temperature.
            Default is ``0.5`` (fully ionized hydrogen).
        mu_3 : float, optional
            Mean molecular weight in units of the proton mass for the shocked
            CSM (Region 3), used for the forward-shock post-shock temperature.
            Default is ``0.5`` (fully ionized hydrogen).
        kwargs
            Passed to the base :class:`~triceratops.dynamics.shocks.core.shock_engine.ShockEngine`.
        """
        super().__init__(**kwargs)
        self._mu_2 = float(mu_2)
        self._mu_3 = float(mu_3)

    # ------------------------------------------------------------------ #
    # Initial Conditions and Evaluation Kernel Generation                #
    # ------------------------------------------------------------------ #
    @staticmethod
    def generate_initial_conditions(
        R_cd_0: "_UnitBearingScalarLike",
        v_cd_0: "_UnitBearingScalarLike",
        t_0: "_UnitBearingScalarLike",
        rho_1: "Callable",
        rho_4: "Callable",
        u_1: "Callable",
        u_4: "Callable",
        gamma_2: float = 5 / 3,
        gamma_3: float = 5 / 3,
        Delta_frac: float = 0.01,
    ) -> "tuple[float, ...]":
        r"""
        Derive a self-consistent 8-component initial-condition vector.

        This method is intended as a convenience for setting the initial conditions for simulation
        runs with the :class:`MechanicalShockEngine` based on the smaller set of intuitive initial
        conditions. This should not be considered the only way to set initial conditions for this engine,
        and users are free to construct their own initial conditions as long as they are consistent with
        the governing equations.

        Given the contact-discontinuity position and velocity at time
        :math:`t_0`, this method constructs
        :math:`(R_{{\rm cd},0},\,v_{{\rm cd},0},\,M_{2,0},\,M_{3,0},\,U_{2,0},\,U_{3,0},\,\Delta_{2,0},\,\Delta_{3,0})`
        such that the ODE kernel is free of artificial relaxation transients at
        :math:`t = t_0`.

        Both shocked layers are initialised to a small fraction of the contact
        radius,

        .. math::

            \Delta_{i,0} = f_{\Delta}\,R_{{\rm cd},0},

        placing the reverse-shock face at
        :math:`R_{{\rm rs},0} = R_{{\rm cd},0} - \Delta_{2,0}` and the
        forward-shock face at
        :math:`R_{{\rm fs},0} = R_{{\rm cd},0} + \Delta_{3,0}`.

        The swept-up masses are obtained by integrating the upstream density
        profiles at :math:`t_0` over the appropriate radial domains.  For Region
        2 (shocked ejecta) the integral runs outward from the reverse-shock face,
        capturing all ejecta that would already be moving faster than the shock:

        .. math::

            M_{2,0} = 4\pi\int_{R_{{\rm rs},0}}^{\infty} \rho_1(r,\,t_0)\,r^2\,dr.

        For Region 3 (shocked CSM) it sweeps inward from the forward-shock face:

        .. math::

            M_{3,0} = 4\pi\int_0^{R_{{\rm fs},0}} \rho_4(r,\,t_0)\,r^2\,dr.

        :math:`M_{2,0}` is evaluated in :math:`\ln r` space: the substitution
        :math:`r = e^s` transforms the integrand to
        :math:`4\pi e^{3s}\rho_1(e^s, t_0)`, which for a broken-power-law outer
        index :math:`n` scales as :math:`e^{(3-n)s}` — smooth and rapidly
        decaying for Gauss-Kronrod quadrature.  Integrating in the original
        linear variable instead would place the function peak at the lower bound
        with a steep falloff, causing QUADPACK's :math:`(1-t)/t` substitution to
        incorrectly estimate the integral through catastrophic cancellation.
        :math:`M_{3,0}` is integrated in the original variable over the finite
        interval :math:`[0, R_{{\rm fs},0}]`, where Gauss-Kronrod performs
        reliably for all standard CSM profiles.

        Naively setting :math:`v_{{\rm rel},i,0}` from the shock speeds and then
        computing internal energies from those speeds produces a sound speed that
        is inconsistent with the assumed shock speed, generating a sharp transient
        at the first ODE step.  To avoid this, each closure is substituted into
        the shock-frame relative velocity definition.

        For the reverse shock the velocity-ratio closure
        :math:`\dot{\Delta}_2 = (u_1 - v_{\rm cd})/(\chi_2-1)` gives
        :math:`D_{\rm rs} = v_{\rm cd} - \dot{\Delta}_2`, so:

        .. math::

            v_{{\rm rel},2,0}
            = u_1(R_{{\rm rs},0},\,t_0) - D_{{\rm rs},0}
            = \frac{\chi_2}{\chi_2 - 1}\bigl(u_1(R_{{\rm rs},0},\,t_0) - v_{{\rm cd},0}\bigr).

        For the forward shock the sound-speed closure
        :math:`D_{{\rm fs}} = v_{{\rm cd}} + c_{s,3}` is substituted into the
        definition of the shock-frame relative velocity:

        .. math::

            v_{{\rm rel},3,0}
            = \frac{v_{{\rm cd},0} - u_4(R_{{\rm fs},0},\,t_0)}{1 - \beta_3},

        where :math:`\chi_i = (\gamma_i+1)/(\gamma_i-1)` and
        :math:`\beta_3 = \sqrt{\gamma_3(\chi_3-1)/\chi_3^2}` is the ratio of
        the post-shock sound speed to the shock-frame upstream velocity in the
        strong-shock limit.  The resulting :math:`v_{{\rm rel},i,0}` are
        self-consistent with each layer's width closure.

        The Rankine--Hugoniot jump conditions give the specific internal energy
        deposited by a strong shock:

        .. math::

            U_{i,0}
            = \frac{1}{\gamma_i-1}\frac{\chi_i-1}{\chi_i^2}
            \,v_{{\rm rel},i,0}^2\,M_{i,0}.

        Parameters
        ----------
        R_cd_0 : ~astropy.units.Quantity or float
            Initial contact-discontinuity radius.  Unit-bearing inputs are
            converted to cm; bare floats are interpreted as cm.
        v_cd_0 : ~astropy.units.Quantity or float
            Initial contact-discontinuity velocity.  Unit-bearing inputs are
            converted to cm/s; bare floats are interpreted as cm/s.
        t_0 : ~astropy.units.Quantity or float
            Time at which the initial conditions are evaluated.  Unit-bearing
            inputs are converted to s; bare floats are interpreted as s.
        rho_1 : callable
            Upstream ejecta density :math:`\rho_1(r,\,t)` **in CGS**
            (:math:`\mathrm{g\,cm^{-3}}`).  Must accept array-like ``r`` (cm)
            and scalar ``t`` (s).
        rho_4 : callable
            Upstream CSM density :math:`\rho_4(r,\,t)` **in CGS**
            (:math:`\mathrm{g\,cm^{-3}}`).  Must accept array-like ``r`` and
            scalar ``t``.
        u_1 : callable
            Upstream ejecta velocity :math:`u_1(r,\,t)` **in CGS** (cm/s).
        u_4 : callable
            Upstream CSM velocity :math:`u_4(r,\,t)` **in CGS** (cm/s).
        gamma_2 : float, optional
            Adiabatic index of the shocked ejecta.  Default ``5/3``.
        gamma_3 : float, optional
            Adiabatic index of the shocked CSM.  Default ``5/3``.
        Delta_frac : float, optional
            Initial layer width as a fraction of :math:`R_{{\rm cd},0}`.
            Smaller values place the shock faces closer to the contact
            discontinuity, reducing the initial swept-up mass but potentially
            requiring finer ODE tolerances to resolve the early evolution.
            Default ``0.01``.

        Returns
        -------
        tuple of float
            ``(R_cd_0, v_cd_0, M2_0, M3_0, U2_0, U3_0, Delta2_0, Delta3_0)``
            — all values in CGS, ready to unpack directly as the ``y0`` argument
            to :meth:`compute_shock_properties`.

        Raises
        ------
        ValueError
            If the upstream conditions imply a non-positive shock-frame relative
            velocity for either shock, meaning no valid shock exists at the
            chosen initial position and velocity.

        Notes
        -----
        Both mass integrals are evaluated with :func:`scipy.integrate.quad`
        (adaptive Gauss--Kronrod quadrature, ``limit=200`` subintervals).  The
        :math:`M_2` upper limit is :math:`+\infty`; ``quad`` handles the tail
        via an internal change of variable and does not require a hardcoded
        truncation radius.  The :math:`M_3` lower limit is exactly 0; the
        integrand :math:`4\pi r^2 \rho_4` is integrable at the origin for all
        standard CSM profiles (wind: integrand is constant; uniform: quadratic;
        shell: zero interior to the shell).  Because ``quad`` evaluates the
        integrand at scalar :math:`r` values, each density callable is invoked
        with a 1-element NumPy array rather than a bare Python float, so
        callables that use boolean array masking internally remain compatible.
        """
        # --- strip units ---
        R_cd_0 = float(ensure_in_units(R_cd_0, u.cm))
        v_cd_0 = float(ensure_in_units(v_cd_0, u.cm / u.s))
        t_0 = float(ensure_in_units(t_0, u.s))

        chi_2 = (gamma_2 + 1) / (gamma_2 - 1)
        chi_3 = (gamma_3 + 1) / (gamma_3 - 1)

        # --- initial widths and shock face positions ---
        Delta2_0 = Delta_frac * R_cd_0
        Delta3_0 = Delta_frac * R_cd_0
        R_rs_0 = R_cd_0 - Delta2_0
        R_fs_0 = R_cd_0 + Delta3_0

        # --- shocked masses from density integrals ---
        # Callables may use boolean array masking internally, so scalars are
        # wrapped in a 1-element array and the result is extracted with flat[0].
        #
        # M2: ejecta at v > R_rs_0/t_0, swept up by the reverse shock.
        # Integrated in log-r: f(r) dr = f(e^s) e^s ds, so the log-space
        # integrand is 4π r³ ρ. For a BPL outer index n the physical integrand
        # scales as r^{2-n} (e.g. r^{-8} for n=10), which DQAGI's (1-t)/t
        # substitution handles poorly when the function is steep near the lower
        # bound — cancellation in the quadrature weights causes wrong signs and
        # magnitudes. In log-space the same profile becomes e^{(3-n)s}, smooth
        # for Gauss-Kronrod. The upper bound is capped at ln(R_rs_0)+100
        # (≈ 43 orders of magnitude above R_rs_0) so exp() never overflows;
        # no physical ejecta profile carries appreciable mass that far out.
        def _log_M2_integrand(log_r: float) -> float:
            r = np.exp(log_r)
            val = 4.0 * np.pi * r**3 * np.asarray(rho_1(np.array([r]), t_0)).flat[0]
            return val if np.isfinite(val) else 0.0

        M2_0, _ = quad(
            _log_M2_integrand,
            np.log(R_rs_0),
            np.log(R_rs_0) + 100.0,
            limit=200,
        )

        # M3: CSM swept up interior to the forward-shock face.
        # The lower limit is 0; r²ρ is integrable there for all standard CSM
        # profiles (wind integrand is constant, uniform is quadratic, etc.).
        # Gauss-Kronrod never evaluates at the exact endpoint so r=0 is safe.
        M3_0, _ = quad(
            lambda r: 4.0 * np.pi * r**2 * np.asarray(rho_4(np.array([r]), t_0)).flat[0],
            0.0,
            R_fs_0,
            limit=200,
        )

        # --- self-consistent shock-frame relative velocities ---
        # RS closure: dDelta2/dt = (u1 - v_cd) / (chi_2 - 1), D_rs = v_cd - dDelta2/dt.
        # Substituting: v_rel_2 = u1 - D_rs = (u1 - v_cd) * chi_2 / (chi_2 - 1).
        # FS closure: dDelta3/dt = c_s3 = beta_3 * v_rel_3.
        # Substituting into v_rel_3 = D_fs - u_4 gives the closed form below.
        beta_3 = np.sqrt(gamma_3 * (chi_3 - 1) / chi_3**2)

        v_rel_2_0 = (u_1(R_rs_0, t_0) - v_cd_0) * chi_2 / (chi_2 - 1.0)
        v_rel_3_0 = (v_cd_0 - u_4(R_fs_0, t_0)) / (1.0 - beta_3)

        if v_rel_2_0 <= 0:
            raise ValueError(
                f"Reverse shock has non-positive upstream relative velocity "
                f"({v_rel_2_0:.3e} cm/s). Ensure u_1(R_rs_0, t_0) > v_cd_0."
            )
        if v_rel_3_0 <= 0:
            raise ValueError(
                f"Forward shock has non-positive upstream relative velocity "
                f"({v_rel_3_0:.3e} cm/s). Ensure v_cd_0 > u_4(R_fs_0, t_0)."
            )

        # --- Rankine-Hugoniot specific internal energies ---
        eps_2 = (1.0 / (gamma_2 - 1)) * ((chi_2 - 1) / chi_2**2) * v_rel_2_0**2
        eps_3 = (1.0 / (gamma_3 - 1)) * ((chi_3 - 1) / chi_3**2) * v_rel_3_0**2

        U2_0 = eps_2 * M2_0
        U3_0 = eps_3 * M3_0

        return R_cd_0, v_cd_0, M2_0, M3_0, U2_0, U3_0, Delta2_0, Delta3_0

    @staticmethod
    def generate_evaluation_kernel(
        rho_1: Callable,
        rho_4: Callable,
        u_1: Callable,
        u_4: Callable,
        gamma_2: float = 5 / 3,
        gamma_3: float = 5 / 3,
        cooling_2: "Union[Callable, None]" = None,
        cooling_3: "Union[Callable, None]" = None,
        M1_total: "Union[float, None]" = None,
    ) -> Callable:
        r"""
        Build the ODE right-hand side for the mechanical shock model.

        Returns a function ``kernel(t, y)`` suitable for
        :func:`scipy.integrate.solve_ivp`.

        Parameters
        ----------
        rho_1 : callable
            Upstream ejecta density :math:`\rho_1(r,\,t)` in
            :math:`\mathrm{g\,cm^{-3}}`.
        rho_4 : callable
            Upstream CSM density :math:`\rho_4(r,\,t)` in
            :math:`\mathrm{g\,cm^{-3}}`.
        u_1 : callable
            Upstream ejecta velocity :math:`u_1(r,\,t)` in cm/s.
        u_4 : callable
            Upstream CSM velocity :math:`u_4(r,\,t)` in cm/s.
        gamma_2 : float, optional
            Adiabatic index of the shocked ejecta. Default ``5/3``.
        gamma_3 : float, optional
            Adiabatic index of the shocked CSM. Default ``5/3``.
        cooling_2 : callable or None, optional
            Radiative loss rate for Region 2,
            ``cooling_2(R_cd, v_cd, M2, U2, Delta2, t) -> dU2/dt`` in erg/s.
            Should return a **negative** value for energy losses. Default is
            no cooling.
        cooling_3 : callable or None, optional
            Radiative loss rate for Region 3,
            ``cooling_3(R_cd, v_cd, M3, U3, Delta3, t) -> dU3/dt`` in erg/s.
            Should return a **negative** value for energy losses. Default is
            no cooling.
        M1_total : float or None, optional
            Total ejecta mass in grams.  When ``M2`` reaches
            ``(1 - 1e-4) * M1_total`` the reverse shock is considered to have
            crossed all ejecta.  Three things are applied simultaneously:

            1. **Mass loading quenched**: :math:`\dot{M}_2 = 0`.
            2. **Shock heating quenched**: :math:`\dot{U}_{{\rm sh},2} = 0`.
            3. **Width closure switched**: :math:`\dot{\Delta}_2` changes from
               the velocity-ratio formula :math:`(u_1 - v_{\rm cd})/(\chi_2-1)`
               to the sonic closure :math:`c_{s,2}`, so Region 2 continues to
               expand adiabatically consistent with the Region-3 treatment.

            Default ``None`` (no limit; reverse shock evolves without bound).

        Returns
        -------
        callable
            ``kernel(t, y) -> dy/dt`` where
            ``y = [R_cd, v_cd, M2, M3, U2, U3, Delta2, Delta3]``.
        """
        chi_2 = (gamma_2 + 1) / (gamma_2 - 1)
        chi_3 = (gamma_3 + 1) / (gamma_3 - 1)

        _cooling_2 = cooling_2 if cooling_2 is not None else lambda *_: 0.0
        _cooling_3 = cooling_3 if cooling_3 is not None else lambda *_: 0.0

        # Capture the ejecta mass limit; treat None as no limit.
        _M1_total = np.inf if M1_total is None else float(M1_total)

        def _kernel(t, y):
            R_cd, v_cd, M2, M3, U2, U3, Delta2, Delta3 = y

            # --- Geometry ---
            R_rs = R_cd - Delta2
            R_fs = R_cd + Delta3
            V2 = 4.0 * np.pi * R_cd**2 * Delta2
            V3 = 4.0 * np.pi * R_cd**2 * Delta3

            # --- Layer-averaged pressures ---
            P2 = (gamma_2 - 1) * U2 / V2
            P3 = (gamma_3 - 1) * U3 / V3

            # --- Sound speeds ---
            # c_s3 drives the FS width closure (dDelta3/dt = c_s3) throughout.
            # c_s2 replaces the velocity-ratio RS closure once the ejecta is exhausted.
            c_s3 = np.sqrt(np.maximum(gamma_3 * (gamma_3 - 1) * U3 / M3, 0.0))
            c_s2 = np.sqrt(np.maximum(gamma_2 * (gamma_2 - 1) * U2 / M2, 0.0))

            # --- Upstream quantities at shock faces ---
            rho1_rs = rho_1(R_rs, t)
            rho4_fs = rho_4(R_fs, t)
            u1_rs = u_1(R_rs, t)
            u4_fs = u_4(R_fs, t)

            # --- Ejecta-crossing flag ---
            # True once the reverse shock has swept through (1 - tol) of all ejecta.
            # Three things happen at this transition:
            #   1. The RS width closure switches from velocity-ratio to sonic: dDelta2/dt = c_s2.
            #      This lets Region 2 expand freely at its own sound speed, consistent with
            #      the Region-3 (FS) closure. Without this switch dDelta2/dt would either
            #      stall at zero (if v_cd ≥ u1) or keep using a now-unphysical velocity ratio.
            #   2. Mass loading is quenched: dM2/dt = 0.
            #   3. Shock heating is quenched: dU_sh2 = 0.
            _is_ejecta_crossed = M2 >= (1.0 - _EJECTA_MASS_TOL) * _M1_total

            # --- Shock speeds ---
            # FS: sound-speed closure throughout.
            # RS: velocity-ratio closure while ejecta exists; sonic closure after crossing.
            #     The velocity-ratio closure is floored at zero so the shock cannot retreat.
            delta_v_2 = u1_rs - v_cd
            dDelta2_dt = np.where(
                _is_ejecta_crossed,
                c_s2,
                np.maximum(delta_v_2 / (chi_2 - 1.0), 0.0),
            )

            D_rs = v_cd - dDelta2_dt
            D_fs = v_cd + c_s3

            # --- Shock-frame relative velocities ---
            # Clamped to zero: swept mass is non-decreasing. A negative value means the
            # shock has stalled; no mass is un-swept in that case, and heating ceases.
            v_rel_2 = np.maximum(u1_rs - D_rs, 0.0)
            v_rel_3 = np.maximum(D_fs - u4_fs, 0.0)

            # --- Mass loading ---
            dM2_dt = np.where(_is_ejecta_crossed, 0.0, 4.0 * np.pi * R_rs**2 * rho1_rs * v_rel_2)
            dM3_dt = 4.0 * np.pi * R_fs**2 * rho4_fs * v_rel_3

            # --- Contact-discontinuity motion ---
            dR_cd_dt = v_cd
            dv_cd_dt = 4.0 * np.pi * R_cd**2 * (P2 - P3) / (M2 + M3)

            # --- Shock heating ---
            dU_sh2 = np.where(
                _is_ejecta_crossed,
                0.0,
                (1.0 / (gamma_2 - 1)) * ((chi_2 - 1) / chi_2**2) * v_rel_2**2 * dM2_dt,
            )
            dU_sh3 = (1.0 / (gamma_3 - 1)) * ((chi_3 - 1) / chi_3**2) * v_rel_3**2 * dM3_dt

            # --- Adiabatic expansion losses (P dV work via V_i = 4π R_cd² Δ_i) ---
            dU_ad2 = -(gamma_2 - 1) * U2 * (2.0 * v_cd / R_cd + dDelta2_dt / Delta2)
            dU_ad3 = -(gamma_3 - 1) * U3 * (2.0 * v_cd / R_cd + c_s3 / Delta3)

            # --- Radiative losses ---
            dU_rad2 = _cooling_2(R_cd, v_cd, M2, U2, Delta2, t)
            dU_rad3 = _cooling_3(R_cd, v_cd, M3, U3, Delta3, t)

            dU2_dt = dU_sh2 + dU_ad2 + dU_rad2
            dU3_dt = dU_sh3 + dU_ad3 + dU_rad3

            return np.array(
                [
                    dR_cd_dt,
                    dv_cd_dt,
                    dM2_dt,
                    dM3_dt,
                    dU2_dt,
                    dU3_dt,
                    dDelta2_dt,
                    c_s3,
                ]
            )

        return _kernel

    # ------------------------------------------------------------------ #
    # Core interface                                                       #
    # ------------------------------------------------------------------ #

    def compute_shock_properties(
        self,
        time: "_UnitBearingArrayLike",
        rho_1: Callable,
        rho_4: Callable,
        u_1: Callable,
        u_4: Callable,
        R_cd_0: "_UnitBearingScalarLike" = 1e14 * u.cm,
        v_cd_0: "_UnitBearingScalarLike" = 1e9 * u.cm / u.s,
        M2_0: "_UnitBearingScalarLike" = 1e26 * u.g,
        M3_0: "_UnitBearingScalarLike" = 1e26 * u.g,
        U2_0: "_UnitBearingScalarLike" = 1e45 * u.erg,
        U3_0: "_UnitBearingScalarLike" = 1e45 * u.erg,
        Delta2_0: "_UnitBearingScalarLike" = 1e12 * u.cm,
        Delta3_0: "_UnitBearingScalarLike" = 1e12 * u.cm,
        t_0: "_UnitBearingScalarLike" = 1.0 * u.s,
        gamma_2: float = 5 / 3,
        gamma_3: float = 5 / 3,
        cooling_2: "Union[Callable, None]" = None,
        cooling_3: "Union[Callable, None]" = None,
        M1_total: "Union[float, None]" = None,
        **kwargs,
    ) -> MechanicalShockState:
        r"""
        Compute the mechanical shock evolution and return a unit-bearing state.

        Parameters
        ----------
        time : ~astropy.units.Quantity or array-like
            Times at which to evaluate the solution. If a
            :class:`~astropy.units.Quantity`, units are converted to seconds;
            otherwise seconds are assumed. Must be sorted and lie in
            :math:`[t_0, \max(t)]`.
        rho_1 : callable
            :math:`\rho_1(r,\,t)` — upstream ejecta density in CGS.
        rho_4 : callable
            :math:`\rho_4(r,\,t)` — upstream CSM density in CGS.
        u_1 : callable
            :math:`u_1(r,\,t)` — upstream ejecta velocity in CGS.
        u_4 : callable
            :math:`u_4(r,\,t)` — upstream CSM velocity in CGS.
        R_cd_0 : ~astropy.units.Quantity or float
            Initial contact-discontinuity radius. Default ``1e14 cm``.
        v_cd_0 : ~astropy.units.Quantity or float
            Initial contact-discontinuity velocity. Default ``1e9 cm/s``.
        M2_0 : ~astropy.units.Quantity or float
            Initial shocked ejecta mass. Default ``1e26 g``.
        M3_0 : ~astropy.units.Quantity or float
            Initial shocked CSM mass. Default ``1e26 g``.
        U2_0 : ~astropy.units.Quantity or float
            Initial shocked ejecta internal energy. Default ``1e45 erg``.
        U3_0 : ~astropy.units.Quantity or float
            Initial shocked CSM internal energy. Default ``1e45 erg``.
        Delta2_0 : ~astropy.units.Quantity or float
            Initial shocked ejecta layer width. Default ``1e12 cm``.
        Delta3_0 : ~astropy.units.Quantity or float
            Initial shocked CSM layer width. Default ``1e12 cm``.
        t_0 : ~astropy.units.Quantity or float
            Initial time. Default ``1.0 s``.
        gamma_2 : float, optional
            Adiabatic index of the shocked ejecta. Default ``5/3``.
        gamma_3 : float, optional
            Adiabatic index of the shocked CSM. Default ``5/3``.
        cooling_2 : callable or None, optional
            Radiative loss rate for Region 2 (see
            :meth:`generate_evaluation_kernel`). Default is no cooling.
        cooling_3 : callable or None, optional
            Radiative loss rate for Region 3. Default is no cooling.
        M1_total : ~astropy.units.Quantity or float or None, optional
            Total ejecta mass.  Unit-bearing inputs are converted to g; bare floats
            are interpreted as g.  When the swept ejecta mass ``M2`` reaches
            ``(1 - 1e-4) * M1_total``, the reverse shock is treated as fully
            traversed: mass loading and shock heating for Region 2 are quenched and
            the width closure switches to the sonic form :math:`\dot{\Delta}_2 = c_{s,2}`.
            Default ``None`` (no limit).
        **kwargs
            Forwarded to :func:`scipy.integrate.solve_ivp`.
            ``method`` defaults to ``'Radau'``; ``rtol`` defaults to ``1e-10``.

        Returns
        -------
        ~triceratops.dynamics.shocks.numerical.MechanicalShockState
            Named tuple of :class:`~astropy.units.Quantity` arrays.
        """
        if isinstance(time, u.Quantity):
            time = time.to(u.s).value
        if isinstance(R_cd_0, u.Quantity):
            R_cd_0 = R_cd_0.to(u.cm).value
        if isinstance(v_cd_0, u.Quantity):
            v_cd_0 = v_cd_0.to(u.cm / u.s).value
        if isinstance(M2_0, u.Quantity):
            M2_0 = M2_0.to(u.g).value
        if isinstance(M3_0, u.Quantity):
            M3_0 = M3_0.to(u.g).value
        if isinstance(U2_0, u.Quantity):
            U2_0 = U2_0.to(u.erg).value
        if isinstance(U3_0, u.Quantity):
            U3_0 = U3_0.to(u.erg).value
        if isinstance(Delta2_0, u.Quantity):
            Delta2_0 = Delta2_0.to(u.cm).value
        if isinstance(Delta3_0, u.Quantity):
            Delta3_0 = Delta3_0.to(u.cm).value
        if isinstance(t_0, u.Quantity):
            t_0 = t_0.to(u.s).value
        if M1_total is None:
            M1_total = np.inf
        elif isinstance(M1_total, u.Quantity):
            M1_total = M1_total.to(u.g).value

        cgs = self._compute_shock_properties_cgs(
            time=time,
            rho_1=rho_1,
            rho_4=rho_4,
            u_1=u_1,
            u_4=u_4,
            R_cd_0=R_cd_0,
            v_cd_0=v_cd_0,
            M2_0=M2_0,
            M3_0=M3_0,
            U2_0=U2_0,
            U3_0=U3_0,
            Delta2_0=Delta2_0,
            Delta3_0=Delta3_0,
            t_0=t_0,
            gamma_2=gamma_2,
            gamma_3=gamma_3,
            cooling_2=cooling_2,
            cooling_3=cooling_3,
            M1_total=M1_total,
            **kwargs,
        )

        return MechanicalShockState(
            radius=cgs.radius * u.cm,
            velocity=cgs.velocity * (u.cm / u.s),
            mass_2=cgs.mass_2 * u.g,
            mass_3=cgs.mass_3 * u.g,
            energy_2=cgs.energy_2 * u.erg,
            energy_3=cgs.energy_3 * u.erg,
            width_2=cgs.width_2 * u.cm,
            width_3=cgs.width_3 * u.cm,
            radius_rs=cgs.radius_rs * u.cm,
            radius_fs=cgs.radius_fs * u.cm,
            velocity_rs=cgs.velocity_rs * (u.cm / u.s),
            velocity_fs=cgs.velocity_fs * (u.cm / u.s),
            volume_2=cgs.volume_2 * u.cm**3,
            volume_3=cgs.volume_3 * u.cm**3,
            density_2=cgs.density_2 * (u.g / u.cm**3),
            density_3=cgs.density_3 * (u.g / u.cm**3),
            pressure_2=cgs.pressure_2 * (u.dyn / u.cm**2),
            pressure_3=cgs.pressure_3 * (u.dyn / u.cm**2),
            temperature_2=cgs.temperature_2 * u.K,
            temperature_3=cgs.temperature_3 * u.K,
            thermal_energy_density_2=cgs.thermal_energy_density_2 * (u.erg / u.cm**3),
            thermal_energy_density_3=cgs.thermal_energy_density_3 * (u.erg / u.cm**3),
            sound_speed_2=cgs.sound_speed_2 * (u.cm / u.s),
            sound_speed_3=cgs.sound_speed_3 * (u.cm / u.s),
            jump_density_rs=cgs.jump_density_rs * (u.g / u.cm**3),
            jump_pressure_rs=cgs.jump_pressure_rs * (u.dyn / u.cm**2),
            jump_temperature_rs=cgs.jump_temperature_rs * u.K,
            jump_thermal_energy_density_rs=cgs.jump_thermal_energy_density_rs * (u.erg / u.cm**3),
            jump_density_fs=cgs.jump_density_fs * (u.g / u.cm**3),
            jump_pressure_fs=cgs.jump_pressure_fs * (u.dyn / u.cm**2),
            jump_temperature_fs=cgs.jump_temperature_fs * u.K,
            jump_thermal_energy_density_fs=cgs.jump_thermal_energy_density_fs * (u.erg / u.cm**3),
        )

    def _compute_shock_properties_cgs(
        self,
        time: "_ArrayLike",
        rho_1: Callable,
        rho_4: Callable,
        u_1: Callable,
        u_4: Callable,
        R_cd_0: float = 1e14,
        v_cd_0: float = 1e9,
        M2_0: float = 1e26,
        M3_0: float = 1e26,
        U2_0: float = 1e45,
        U3_0: float = 1e45,
        Delta2_0: float = 1e12,
        Delta3_0: float = 1e12,
        t_0: float = 1.0,
        gamma_2: float = 5 / 3,
        gamma_3: float = 5 / 3,
        cooling_2: "Union[Callable, None]" = None,
        cooling_3: "Union[Callable, None]" = None,
        M1_total: "Union[float, None]" = None,
        **kwargs,
    ) -> MechanicalShockState:
        r"""
        Integrate the mechanical shock ODEs in CGS units.

        This method returns the evolved dynamical state, region-averaged
        thermodynamic quantities, and instantaneous shock-front jump diagnostics.

        The region-averaged thermodynamic quantities are computed from the evolved
        masses, internal energies, and effective volumes. The ``jump_*`` quantities
        are computed from instantaneous Rankine--Hugoniot shock conditions and
        describe newly shocked material at the shock fronts, not the volume-averaged
        state of Regions 2 and 3.
        """
        time = np.atleast_1d(np.asarray(time, dtype=float))

        kernel = self.generate_evaluation_kernel(
            rho_1=rho_1,
            rho_4=rho_4,
            u_1=u_1,
            u_4=u_4,
            gamma_2=gamma_2,
            gamma_3=gamma_3,
            cooling_2=cooling_2,
            cooling_3=cooling_3,
            M1_total=M1_total,
        )

        y0 = np.array(
            [R_cd_0, v_cd_0, M2_0, M3_0, U2_0, U3_0, Delta2_0, Delta3_0],
            dtype=float,
        )
        t_span = (t_0, float(np.amax(time)))

        solver_kwargs = dict(kwargs)
        rtol = solver_kwargs.pop("rtol", 1e-10)
        method = solver_kwargs.pop("method", "Radau")

        sol = solve_ivp(
            fun=kernel,
            t_span=t_span,
            y0=y0,
            t_eval=time,
            rtol=rtol,
            method=method,
            **solver_kwargs,
        )

        if sol.status < 0:
            raise RuntimeError(f"ODE solver failed to integrate the mechanical shock equations:\n{sol.message}")

        R_cd, v_cd, M2, M3, U2, U3, Delta2, Delta3 = sol.y

        # ------------------------------------------------------------------ #
        # Geometry                                                           #
        # ------------------------------------------------------------------ #
        R_rs = R_cd - Delta2
        R_fs = R_cd + Delta3

        V2 = 4.0 * np.pi * R_cd**2 * Delta2
        V3 = 4.0 * np.pi * R_cd**2 * Delta3

        # ------------------------------------------------------------------ #
        # Region-averaged thermodynamics                                     #
        # ------------------------------------------------------------------ #
        density_2 = M2 / V2
        density_3 = M3 / V3

        pressure_2 = (gamma_2 - 1.0) * U2 / V2
        pressure_3 = (gamma_3 - 1.0) * U3 / V3

        thermal_energy_density_2 = U2 / V2
        thermal_energy_density_3 = U3 / V3

        temperature_2 = self._mu_2 * m_p_cgs / k_B_cgs * pressure_2 / density_2
        temperature_3 = self._mu_3 * m_p_cgs / k_B_cgs * pressure_3 / density_3

        sound_speed_2 = np.sqrt(np.maximum(gamma_2 * pressure_2 / density_2, 0.0))
        sound_speed_3 = np.sqrt(np.maximum(gamma_3 * pressure_3 / density_3, 0.0))

        # ------------------------------------------------------------------ #
        # Upstream quantities at shock faces                                 #
        # ------------------------------------------------------------------ #
        n_steps = len(time)

        rho_4_fs = np.array([float(rho_4(R_fs[i], time[i])) for i in range(n_steps)])
        u_4_fs = np.array([float(u_4(R_fs[i], time[i])) for i in range(n_steps)])

        rho_1_rs = np.array([float(rho_1(R_rs[i], time[i])) for i in range(n_steps)])
        u_1_rs = np.array([float(u_1(R_rs[i], time[i])) for i in range(n_steps)])

        # ------------------------------------------------------------------ #
        # Shock kinematics                                                   #
        # ------------------------------------------------------------------ #
        chi_2 = (gamma_2 + 1.0) / (gamma_2 - 1.0)

        # Ejecta-crossing flag: mirrors the identical check in the ODE kernel so
        # the post-processed kinematics stay consistent with the integrated trajectory.
        _ejecta_crossed = M2 >= (1.0 - _EJECTA_MASS_TOL) * (np.inf if M1_total is None else float(M1_total))

        # Reverse shock: velocity-ratio closure while ejecta exists; sonic closure
        # afterwards (same branch as the kernel).  active_rs is also False after
        # crossing because there is no longer a physical shock front to diagnose.
        delta_v_2_raw = u_1_rs - v_cd
        dDelta2_dt = np.where(
            _ejecta_crossed,
            sound_speed_2,
            np.maximum(delta_v_2_raw / (chi_2 - 1.0), 0.0),
        )
        D_rs = v_cd - dDelta2_dt
        active_rs = (delta_v_2_raw > 0.0) & ~_ejecta_crossed

        # Forward shock: sound-speed width closure.
        D_fs = v_cd + sound_speed_3
        v_rel_fs = D_fs - u_4_fs
        active_fs = v_rel_fs > 0.0

        # ------------------------------------------------------------------ #
        # Instantaneous jump diagnostics via Rankine-Hugoniot conditions      #
        # ------------------------------------------------------------------ #
        # Forward shock: usually active, but still mask if the relative speed is
        # non-positive.
        jump_density_fs = np.full_like(R_cd, np.nan)
        jump_pressure_fs = np.full_like(R_cd, np.nan)
        jump_temperature_fs = np.full_like(R_cd, np.nan)
        jump_thermal_energy_density_fs = np.full_like(R_cd, np.nan)

        if np.any(active_fs):
            rh_fs = StrongColdShockConditions._solve(
                D_fs[active_fs],
                rho_4_fs[active_fs],
                u_4_fs[active_fs],
                gamma=gamma_3,
                mu=self._mu_3,
            )

            jump_density_fs[active_fs] = rh_fs["post_shock_density"]
            jump_pressure_fs[active_fs] = rh_fs["post_shock_pressure"]
            jump_temperature_fs[active_fs] = rh_fs["post_shock_temperature"]
            jump_thermal_energy_density_fs[active_fs] = rh_fs["post_shock_thermal_energy_density"]

        # Reverse shock: mask inactive phases. During inactive phases, Region 2 can
        # remain hot, but there is no instantaneous reverse-shock injection state.
        jump_density_rs = np.full_like(R_cd, np.nan)
        jump_pressure_rs = np.full_like(R_cd, np.nan)
        jump_temperature_rs = np.full_like(R_cd, np.nan)
        jump_thermal_energy_density_rs = np.full_like(R_cd, np.nan)

        if np.any(active_rs):
            rh_rs = StrongColdShockConditions._solve(
                D_rs[active_rs],
                rho_1_rs[active_rs],
                u_1_rs[active_rs],
                gamma=gamma_2,
                mu=self._mu_2,
            )

            jump_density_rs[active_rs] = rh_rs["post_shock_density"]
            jump_pressure_rs[active_rs] = rh_rs["post_shock_pressure"]
            jump_temperature_rs[active_rs] = rh_rs["post_shock_temperature"]
            jump_thermal_energy_density_rs[active_rs] = rh_rs["post_shock_thermal_energy_density"]

        return MechanicalShockState(
            radius=R_cd,
            velocity=v_cd,
            mass_2=M2,
            mass_3=M3,
            energy_2=U2,
            energy_3=U3,
            width_2=Delta2,
            width_3=Delta3,
            radius_rs=R_rs,
            radius_fs=R_fs,
            velocity_rs=D_rs,
            velocity_fs=D_fs,
            volume_2=V2,
            volume_3=V3,
            density_2=density_2,
            density_3=density_3,
            pressure_2=pressure_2,
            pressure_3=pressure_3,
            temperature_2=temperature_2,
            temperature_3=temperature_3,
            thermal_energy_density_2=thermal_energy_density_2,
            thermal_energy_density_3=thermal_energy_density_3,
            sound_speed_2=sound_speed_2,
            sound_speed_3=sound_speed_3,
            jump_density_rs=jump_density_rs,
            jump_pressure_rs=jump_pressure_rs,
            jump_temperature_rs=jump_temperature_rs,
            jump_thermal_energy_density_rs=jump_thermal_energy_density_rs,
            jump_density_fs=jump_density_fs,
            jump_pressure_fs=jump_pressure_fs,
            jump_temperature_fs=jump_temperature_fs,
            jump_thermal_energy_density_fs=jump_thermal_energy_density_fs,
        )


class RelMechanicalShockEngine(ShockEngine, ABC):
    """
    Relativistic mechanical shock engine.

    .. note::

        This engine is not yet implemented. It will be added in a future release.
    """

    pass
