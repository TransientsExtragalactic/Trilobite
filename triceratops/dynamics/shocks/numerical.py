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
from triceratops.dynamics.shocks.core.shock_engine import ShockEngine
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
# State Classes                                              #
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


# ========================================================== #
# Chevalier / Pressure Driven Shock Engines                  #
# ========================================================== #
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


class RelPressureDrivenThinShellShockEngine(ShockEngine, ABC):
    """
    Relativistic pressure driven thin-shell shock engine.

    .. note::

        This engine is not yet implemented. It will be added in a future release.
    """

    pass


# =========================================================== #
# Mechanical Shock Engines                                    #
# =========================================================== #
# The mechanical shock engine classes evolve separate
# internal energies for the shocked ejecta and shocked CSM layers, allowing them to retain
# a minimal two-shock structure while remaining inexpensive
# enough for parameter studies and inference workflows.


class MechanicalShockState(NamedTuple):
    r"""
    Time-dependent state returned by :class:`MechanicalShockEngine`.

    The low-level CGS interface returns plain :class:`numpy.ndarray` fields in
    CGS units. The public unit-aware interface returns the same state structure
    with dimensional fields converted to :class:`astropy.units.Quantity`.

    Region 2 denotes shocked ejecta, bounded externally by the contact
    discontinuity and internally by the reverse shock. Region 3 denotes shocked
    CSM, bounded internally by the contact discontinuity and externally by the
    forward shock.
    """

    radius: np.ndarray | u.Quantity
    r"""
    Contact-discontinuity radius :math:`R_{\rm cd}` in cm.

    This is the radius of the surface separating shocked ejecta from shocked
    CSM.
    """

    velocity: np.ndarray | u.Quantity
    r"""
    Contact-discontinuity velocity :math:`v_{\rm cd}` in cm/s.

    This is the velocity of the contact discontinuity and therefore the bulk
    velocity assigned to the volume-averaged shocked region.
    """

    mass_2: np.ndarray | u.Quantity
    r"""
    Shocked ejecta mass :math:`M_2` in g.

    This is the accumulated mass swept into Region 2 by the reverse shock.
    """

    mass_3: np.ndarray | u.Quantity
    r"""
    Shocked CSM mass :math:`M_3` in g.

    This is the accumulated mass swept into Region 3 by the forward shock.
    """

    energy_2: np.ndarray | u.Quantity
    r"""
    Internal energy of the shocked ejecta, :math:`U_2`, in erg.

    This is the volume-integrated thermal/internal energy assigned to Region 2.
    It includes shock heating, adiabatic work, and any configured radiative loss
    term.
    """

    energy_3: np.ndarray | u.Quantity
    r"""
    Internal energy of the shocked CSM, :math:`U_3`, in erg.

    This is the volume-integrated thermal/internal energy assigned to Region 3.
    It includes shock heating, adiabatic work, and any configured radiative loss
    term.
    """

    width_2: np.ndarray | u.Quantity
    r"""
    Effective shocked-ejecta width :math:`\Delta_2` in cm.

    This width is defined by

    .. math::

        \Delta_2 = R_{\rm cd} - R_{\rm rs},

    and sets the approximate Region 2 volume through

    .. math::

        V_2 \simeq 4\pi R_{\rm cd}^2 \Delta_2.
    """

    width_3: np.ndarray | u.Quantity
    r"""
    Effective shocked-CSM width :math:`\Delta_3` in cm.

    This width is defined by

    .. math::

        \Delta_3 = R_{\rm fs} - R_{\rm cd},

    and sets the approximate Region 3 volume through

    .. math::

        V_3 \simeq 4\pi R_{\rm cd}^2 \Delta_3.
    """

    pressure_2: np.ndarray | u.Quantity
    r"""
    Volume-averaged shocked-ejecta pressure :math:`P_2` in dyn/cm².

    This diagnostic is computed from the Region 2 internal energy and effective
    volume using the adopted ideal-gas closure.
    """

    pressure_3: np.ndarray | u.Quantity
    r"""
    Volume-averaged shocked-CSM pressure :math:`P_3` in dyn/cm².

    This diagnostic is computed from the Region 3 internal energy and effective
    volume using the adopted ideal-gas closure.
    """

    radius_rs: np.ndarray | u.Quantity
    r"""
    Reverse-shock radius :math:`R_{\rm rs}` in cm.

    This derived diagnostic is computed from

    .. math::

        R_{\rm rs} = R_{\rm cd} - \Delta_2.
    """

    radius_fs: np.ndarray | u.Quantity
    r"""
    Forward-shock radius :math:`R_{\rm fs}` in cm.

    This derived diagnostic is computed from

    .. math::

        R_{\rm fs} = R_{\rm cd} + \Delta_3.
    """

    velocity_rs: np.ndarray | u.Quantity
    r"""
    Reverse-shock speed :math:`D_{\rm rs}` in cm/s.

    With the current sound-speed width closure, this diagnostic is computed as

    .. math::

        D_{\rm rs} = v_{\rm cd} - c_{s,2}.
    """

    velocity_fs: np.ndarray | u.Quantity
    r"""
    Forward-shock speed :math:`D_{\rm fs}` in cm/s.

    With the current sound-speed width closure, this diagnostic is computed as

    .. math::

        D_{\rm fs} = v_{\rm cd} + c_{s,3}.
    """

    post_shock_density_fs: np.ndarray | u.Quantity
    r"""
    Immediate post-shock density at the forward shock :math:`\rho_{s,3}` in
    :math:`\mathrm{g\,cm^{-3}}`, from the strong cold-shock Rankine--Hugoniot
    relation applied to the upstream CSM density :math:`\rho_4(R_{\rm fs},\,t)`.
    """

    post_shock_pressure_fs: np.ndarray | u.Quantity
    r"""
    Immediate post-shock pressure at the forward shock :math:`p_{s,3}` in
    :math:`\mathrm{dyn\,cm^{-2}}`, from the strong cold-shock Rankine--Hugoniot
    relation.
    """

    post_shock_temperature_fs: np.ndarray | u.Quantity
    r"""
    Immediate post-shock temperature at the forward shock :math:`T_{s,3}` in K,
    from the ideal-gas relation.
    """

    thermal_energy_density_fs: np.ndarray | u.Quantity
    r"""
    Post-shock thermal energy density at the forward shock
    :math:`e_{\rm th,3} = p_{s,3} / (\gamma_3 - 1)` in
    :math:`\mathrm{erg\,cm^{-3}}`.
    """

    post_shock_density_rs: np.ndarray | u.Quantity
    r"""
    Immediate post-shock density at the reverse shock :math:`\rho_{s,2}` in
    :math:`\mathrm{g\,cm^{-3}}`, from the strong cold-shock Rankine--Hugoniot
    relation applied to the upstream ejecta density :math:`\rho_1(R_{\rm rs},\,t)`.
    """

    post_shock_pressure_rs: np.ndarray | u.Quantity
    r"""
    Immediate post-shock pressure at the reverse shock :math:`p_{s,2}` in
    :math:`\mathrm{dyn\,cm^{-2}}`, from the strong cold-shock Rankine--Hugoniot
    relation.
    """

    post_shock_temperature_rs: np.ndarray | u.Quantity
    r"""
    Immediate post-shock temperature at the reverse shock :math:`T_{s,2}` in K,
    from the ideal-gas relation.
    """

    thermal_energy_density_rs: np.ndarray | u.Quantity
    r"""
    Post-shock thermal energy density at the reverse shock
    :math:`e_{\rm th,2} = p_{s,2} / (\gamma_2 - 1)` in
    :math:`\mathrm{erg\,cm^{-3}}`.
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
    (:math:`V_i = 4\pi R_{\rm cd}^2 \Delta_i`), sound-speed width closure
    :math:`c_{s,i} = \sqrt{\gamma_i(\gamma_i-1)U_i/M_i}`, shock speeds
    :math:`D_{\rm rs} = v_{\rm cd} - c_{s,2}` and :math:`D_{\rm fs} = v_{\rm cd} + c_{s,3}`,
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
        at the first ODE step.  To avoid this, the width closure
        :math:`D_{{\rm rs}} = v_{{\rm cd}} - c_{s,2}` (and analogously for the
        forward shock) is substituted into the definition of the shock-frame
        relative velocity:

        .. math::

            v_{{\rm rel},2,0}
            = \frac{u_1(R_{{\rm rs},0},\,t_0) - v_{{\rm cd},0}}{1 - \beta_2},
            \qquad
            v_{{\rm rel},3,0}
            = \frac{v_{{\rm cd},0} - u_4(R_{{\rm fs},0},\,t_0)}{1 - \beta_3},

        where :math:`\chi_i = (\gamma_i+1)/(\gamma_i-1)` and
        :math:`\beta_i = \sqrt{\gamma_i(\chi_i-1)/\chi_i^2}` is the ratio of
        the post-shock sound speed to the shock-frame upstream velocity in the
        strong-shock limit.  The resulting :math:`v_{{\rm rel},i,0}` are
        self-consistent with the layer sound speeds implied by the initial
        internal energies computed below.

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
        # Closure: D_rs = v_cd - c_s2, c_s2 = beta * v_rel_2.
        # Substituting into v_rel_2 = u_1(R_rs) - D_rs gives the closed form below.
        beta_2 = np.sqrt(gamma_2 * (chi_2 - 1) / chi_2**2)
        beta_3 = np.sqrt(gamma_3 * (chi_3 - 1) / chi_3**2)

        v_rel_2_0 = (u_1(R_rs_0, t_0) - v_cd_0) / (1.0 - beta_2)
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
        cooling_2: "Callable | None" = None,
        cooling_3: "Callable | None" = None,
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

            # --- Sound speeds (width closure: dDelta_i/dt = c_s_i) ---
            c_s2 = np.sqrt(np.maximum(gamma_2 * (gamma_2 - 1) * U2 / M2, 0.0))
            c_s3 = np.sqrt(np.maximum(gamma_3 * (gamma_3 - 1) * U3 / M3, 0.0))

            # --- Shock speeds ---
            D_rs = v_cd - c_s2
            D_fs = v_cd + c_s3

            # --- Upstream quantities at shock faces ---
            rho1_rs = rho_1(R_rs, t)
            rho4_fs = rho_4(R_fs, t)
            u1_rs = u_1(R_rs, t)
            u4_fs = u_4(R_fs, t)

            # --- Shock-frame relative velocities ---
            # Clamped to zero: swept mass is non-decreasing. A negative value
            # means the shock has stalled (closure breaks down); no mass is
            # un-swept in that case, and shock heating ceases naturally.
            v_rel_2 = np.maximum(u1_rs - D_rs, 0.0)
            v_rel_3 = np.maximum(D_fs - u4_fs, 0.0)

            # --- Mass loading ---
            dM2_dt = 4.0 * np.pi * R_rs**2 * rho1_rs * v_rel_2
            dM3_dt = 4.0 * np.pi * R_fs**2 * rho4_fs * v_rel_3

            # --- Contact-discontinuity motion ---
            dR_cd_dt = v_cd
            dv_cd_dt = 4.0 * np.pi * R_cd**2 * (P2 - P3) / (M2 + M3)

            # --- Shock heating ---
            dU_sh2 = (1.0 / (gamma_2 - 1)) * ((chi_2 - 1) / chi_2**2) * v_rel_2**2 * dM2_dt
            dU_sh3 = (1.0 / (gamma_3 - 1)) * ((chi_3 - 1) / chi_3**2) * v_rel_3**2 * dM3_dt

            # --- Adiabatic expansion (P dV work via V_i = 4pi R_cd^2 Delta_i) ---
            dU_ad2 = -(gamma_2 - 1) * U2 * (2.0 * v_cd / R_cd + c_s2 / Delta2)
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
                    c_s2,  # dDelta2/dt
                    c_s3,  # dDelta3/dt
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
        cooling_2: "Callable | None" = None,
        cooling_3: "Callable | None" = None,
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
            pressure_2=cgs.pressure_2 * (u.dyn / u.cm**2),
            pressure_3=cgs.pressure_3 * (u.dyn / u.cm**2),
            radius_rs=cgs.radius_rs * u.cm,
            radius_fs=cgs.radius_fs * u.cm,
            velocity_rs=cgs.velocity_rs * (u.cm / u.s),
            velocity_fs=cgs.velocity_fs * (u.cm / u.s),
            post_shock_density_fs=cgs.post_shock_density_fs * (u.g / u.cm**3),
            post_shock_pressure_fs=cgs.post_shock_pressure_fs * (u.dyn / u.cm**2),
            post_shock_temperature_fs=cgs.post_shock_temperature_fs * u.K,
            thermal_energy_density_fs=cgs.thermal_energy_density_fs * (u.erg / u.cm**3),
            post_shock_density_rs=cgs.post_shock_density_rs * (u.g / u.cm**3),
            post_shock_pressure_rs=cgs.post_shock_pressure_rs * (u.dyn / u.cm**2),
            post_shock_temperature_rs=cgs.post_shock_temperature_rs * u.K,
            thermal_energy_density_rs=cgs.thermal_energy_density_rs * (u.erg / u.cm**3),
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
        cooling_2: "Callable | None" = None,
        cooling_3: "Callable | None" = None,
        **kwargs,
    ) -> MechanicalShockState:
        r"""
        Integrate the mechanical shock ODEs in CGS units.

        Parameters
        ----------
        time : array-like
            Evaluation times in seconds. Must be sorted and satisfy
            ``time >= t_0``.
        rho_1, rho_4, u_1, u_4 : callable
            Upstream density and velocity functions in CGS.
        R_cd_0, v_cd_0 : float
            Initial contact radius (cm) and velocity (cm/s).
        M2_0, M3_0 : float
            Initial shocked masses (g).
        U2_0, U3_0 : float
            Initial internal energies (erg).
        Delta2_0, Delta3_0 : float
            Initial layer widths (cm).
        t_0 : float
            Initial time (s).
        gamma_2, gamma_3 : float
            Adiabatic indices of Regions 2 and 3.
        cooling_2, cooling_3 : callable or None
            Radiative loss rates (see :meth:`generate_evaluation_kernel`).
        **kwargs
            Forwarded to :func:`scipy.integrate.solve_ivp`.

        Returns
        -------
        ~triceratops.dynamics.shocks.numerical.MechanicalShockState
            Named tuple of plain numpy arrays in CGS units.
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
        )

        y0 = np.array([R_cd_0, v_cd_0, M2_0, M3_0, U2_0, U3_0, Delta2_0, Delta3_0])
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

        # Derived geometry
        R_rs = R_cd - Delta2
        R_fs = R_cd + Delta3

        # Layer volumes and pressures
        V2 = 4.0 * np.pi * R_cd**2 * Delta2
        V3 = 4.0 * np.pi * R_cd**2 * Delta3
        P2 = (gamma_2 - 1) * U2 / V2
        P3 = (gamma_3 - 1) * U3 / V3

        # Shock speeds from sound-speed width closure
        c_s2 = np.sqrt(np.maximum(gamma_2 * (gamma_2 - 1) * U2 / M2, 0.0))
        c_s3 = np.sqrt(np.maximum(gamma_3 * (gamma_3 - 1) * U3 / M3, 0.0))
        D_rs = v_cd - c_s2
        D_fs = v_cd + c_s3

        # Post-shock thermodynamics via strong cold-shock RH conditions.
        # Upstream callables accept scalar (r, t), so evaluate element-by-element.
        n_steps = len(time)
        rho_4_fs = np.array([float(rho_4(R_fs[i], time[i])) for i in range(n_steps)])
        u_4_fs = np.array([float(u_4(R_fs[i], time[i])) for i in range(n_steps)])
        rho_1_rs = np.array([float(rho_1(R_rs[i], time[i])) for i in range(n_steps)])
        u_1_rs = np.array([float(u_1(R_rs[i], time[i])) for i in range(n_steps)])

        rh_fs = StrongColdShockConditions._solve(D_fs, rho_4_fs, u_4_fs, gamma=gamma_3, mu=self._mu_3)
        rh_rs = StrongColdShockConditions._solve(D_rs, rho_1_rs, u_1_rs, gamma=gamma_2, mu=self._mu_2)

        return MechanicalShockState(
            radius=R_cd,
            velocity=v_cd,
            mass_2=M2,
            mass_3=M3,
            energy_2=U2,
            energy_3=U3,
            width_2=Delta2,
            width_3=Delta3,
            pressure_2=P2,
            pressure_3=P3,
            radius_rs=R_rs,
            radius_fs=R_fs,
            velocity_rs=D_rs,
            velocity_fs=D_fs,
            post_shock_density_fs=rh_fs["post_shock_density"],
            post_shock_pressure_fs=rh_fs["post_shock_pressure"],
            post_shock_temperature_fs=rh_fs["post_shock_temperature"],
            thermal_energy_density_fs=rh_fs["post_shock_thermal_energy_density"],
            post_shock_density_rs=rh_rs["post_shock_density"],
            post_shock_pressure_rs=rh_rs["post_shock_pressure"],
            post_shock_temperature_rs=rh_rs["post_shock_temperature"],
            thermal_energy_density_rs=rh_rs["post_shock_thermal_energy_density"],
        )


class RelMechanicalShockEngine(ShockEngine, ABC):
    """
    Relativistic mechanical shock engine.

    .. note::

        This engine is not yet implemented. It will be added in a future release.
    """

    pass
