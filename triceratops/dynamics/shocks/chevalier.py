r"""
Chevalier self-similar shock dynamics for supernova ejecta--CSM interaction.

This module implements analytic shock-dynamics engines based on the classical
self-similar solutions of :footcite:t:`chevalierSelfsimilarSolutionsInteraction1982`
for homologously expanding supernova ejecta interacting with a power-law
circumstellar medium (CSM). These models are useful for rapidly estimating the
radius and velocity evolution of the shocked interface in regimes where the
ejecta and external medium can be approximated by scale-free density profiles.

The ejecta are modeled as a broken power law in homologous velocity
:math:`v = r/t`,

.. math::

    \rho_{\rm ej}(r,t)
    =
    K_{\rm ej} t^{-3}
    \begin{cases}
        v^{-\delta}, & v < v_t, \\
        v_t^{n-\delta} v^{-n}, & v \ge v_t,
    \end{cases}

where :math:`v_t` is the transition velocity, :math:`K_{\rm ej}` is the ejecta
normalization, :math:`\delta` is the inner density index, and :math:`n` is the
outer density index. The helper function :func:`normalize_bpl_ejecta`
computes :math:`v_t` and :math:`K_{\rm ej}` from the total ejecta kinetic energy
and mass.

The CSM is modeled as a power-law density profile,

.. math::

    \rho_{\rm CSM}(r) = K_{\rm CSM} r^{-s},

with utilities for constructing both general power-law normalizations and the
special steady-wind case,

.. math::

    K_{\rm CSM} = \frac{\dot{M}}{4\pi v_w}.

The primary classes are:

- :class:`ChevalierSelfSimilarShockEngine`, which computes self-similar shock
  radius and velocity for arbitrary power-law CSM index :math:`s`.
- :class:`ChevalierSelfSimilarWindShockEngine`, a convenience specialization for
  wind-like media with :math:`s = 2`.

The module also provides helper functions for constructing callable ejecta and
CSM density profiles suitable for use by other shock engines.

Notes
-----
The Chevalier solution assumes homologous ejecta expansion, scale-free ejecta
and CSM density profiles, and a self-similar shocked structure. It is therefore
most appropriate as a fast analytic model or as a physically motivated
approximation for supernova interaction problems. More general density profiles
or non-self-similar behavior should be treated with numerical shock engines.

All internal calculations are performed in CGS units. Public functions and
methods accept :class:`astropy.units.Quantity` inputs where appropriate and
return unit-bearing quantities.

References
----------
.. footbibliography::
"""

from typing import NamedTuple, Union

import numpy as np
from astropy import units as u

from triceratops._typing import _ArrayLike, _UnitBearingArrayLike, _UnitBearingScalarLike
from triceratops.dynamics.shocks.core.rankine_hugoniot import StrongColdShockConditions
from triceratops.dynamics.shocks.core.shock_engine import ShockEngine
from triceratops.dynamics.shocks.utils import _normalize_BPL_ejecta

# ===================================================== #
# Utility Functions                                     #
# ===================================================== #
# These are functions made available here because they are directly associated
# with the Chevalier model, but they are not necessarily specific to the Chevalier model. They are
# intended to be accessible to other shock engines as well, but they are not necessarily general enough to be moved
# to a more general utilities module.


# ==================================================== #
# State Classes                                        #
# ==================================================== #
# ==================================================== #
# Utility Methods                                      #
# ==================================================== #
class ChevalierSelfSimilarFunctions(NamedTuple):
    r"""
    Dimensionless Chevalier self-similar two-shock structure.

    The global similarity coordinate is

    .. math::

        \xi = r / R_c,

    where :math:`R_c` is the contact-discontinuity radius. The returned grid
    spans

    .. math::

        R_{\rm rs}/R_c \le \xi \le R_{\rm fs}/R_c.

    The contact discontinuity is located at ``xi = 1``.
    """

    n: float
    s: float
    gamma: float
    expansion_index: float
    similarity_exponent: float

    A: float
    radius_fs_over_radius_cd: float
    radius_rs_over_radius_cd: float

    xi: np.ndarray
    U: np.ndarray
    Omega: np.ndarray
    P: np.ndarray

    density_hat: np.ndarray
    pressure_hat: np.ndarray
    velocity_hat: np.ndarray

    xi_inner: np.ndarray
    U_inner: np.ndarray
    Omega_inner: np.ndarray
    P_inner: np.ndarray

    xi_outer: np.ndarray
    U_outer: np.ndarray
    Omega_outer: np.ndarray
    P_outer: np.ndarray


class ChevalierSelfSimilarCriticalGrid(NamedTuple):
    r"""
    Critical Chevalier self-similar constants on an ``(n, s)`` grid.

    Arrays are indexed as ``[i, j]``, where ``i`` indexes ``n_values`` and
    ``j`` indexes ``s_values``.
    """

    n_values: np.ndarray
    s_values: np.ndarray
    gamma: float

    expansion_index: np.ndarray
    similarity_exponent: np.ndarray

    A: np.ndarray
    radius_fs_over_radius_cd: np.ndarray
    radius_rs_over_radius_cd: np.ndarray


def compute_self_similar_functions(
    n: float = 10.0,
    s: float = 2.0,
    gamma: float = 5.0 / 3.0,
    n_points: int = 512,
    contact_epsilon: float = 1.0e-6,
    rtol: float = 1.0e-9,
    atol: float = 1.0e-11,
) -> ChevalierSelfSimilarFunctions:
    r"""
    Compute the two-sided Chevalier self-similar shock structure.

    This solves the self-similar ODEs for the shocked ejecta and shocked CSM
    regions and returns the dimensionless profiles on a global grid in

    .. math::

        \xi = r / R_c.

    Parameters
    ----------
    n : float, optional
        Outer ejecta density power-law index. Must satisfy ``n > 5``.
    s : float, optional
        CSM density power-law index. Must satisfy ``s < 3``.
    gamma : float, optional
        Adiabatic index of the shocked gas.
    n_points : int, optional
        Number of global grid points between the reverse and forward shocks.
    contact_epsilon : float, optional
        Small offset used to stop before the contact singularity. The pressure
        and velocity remain finite at the contact, but density or temperature
        may be singular depending on ``n`` and ``s``.
    rtol, atol : float, optional
        Relative and absolute tolerances passed to ``scipy.integrate.solve_ivp``.

    Returns
    -------
    ChevalierSelfSimilarFunctions
        Dimensionless shock constants and self-similar profiles.

    Notes
    -----
    The physical scalings are

    .. math::

        u = \frac{r}{t} U,

    for both sides. For the outer shocked CSM,

    .. math::

        \rho = K_{\rm CSM} r^{-s} \Omega,
        \qquad
        p = K_{\rm CSM} t^{-2} r^{2-s} P.

    For the inner shocked ejecta,

    .. math::

        \rho = K_{\rm ej} t^{n-3} r^{-n} \Omega,
        \qquad
        p = K_{\rm ej} t^{n-5} r^{2-n} P.

    The returned ``density_hat`` and ``pressure_hat`` are scaled to the common
    contact-radius convention, up to the factors ``K_csm R_c^{-s}`` and
    ``K_csm R_c^{2-s} t^{-2}``, respectively.
    """
    from scipy.integrate import solve_ivp
    from scipy.interpolate import PchipInterpolator

    if n <= 5:
        raise ValueError("The outer ejecta index `n` must be greater than 5.")
    if s >= 3:
        raise ValueError("The CSM index `s` must be less than 3.")

    # Chevalier's contact-discontinuity expansion is R_c ∝ t^m.
    expansion_index = (n - 3.0) / (n - s)
    similarity_exponent = 1.0 / expansion_index

    def _powers(side):
        if side == "outer":
            return 0.0, -s
        if side == "inner":
            return n - 3.0, -n
        raise ValueError("side must be 'inner' or 'outer'.")

    def _shock_state(side):
        compression = (gamma + 1.0) / (gamma - 1.0)

        if side == "outer":
            # Forward shock into stationary, cold CSM.
            U = 2.0 * expansion_index / (gamma + 1.0)
            Omega = compression
            P = 2.0 * expansion_index**2 / (gamma + 1.0)
            return np.array([U, Omega, P], dtype=float)

        if side == "inner":
            # Reverse shock into homologously expanding ejecta.
            relative_speed = 1.0 - expansion_index
            U = expansion_index + (gamma - 1.0) / (gamma + 1.0) * relative_speed
            Omega = compression
            P = 2.0 * relative_speed**2 / (gamma + 1.0)
            return np.array([U, Omega, P], dtype=float)

        raise ValueError("side must be 'inner' or 'outer'.")

    def _rhs(x, y, side):
        U, Omega, P = y
        a, b = _powers(side)

        if Omega <= 0.0 or P <= 0.0:
            return np.array([np.nan, np.nan, np.nan])

        # x = r/R_shock on each side. The contact is where U -> m.
        D = x * (U - expansion_index)

        E = a - 2.0 - gamma * a + (b + 2.0 - gamma * b) * U

        matrix = np.array(
            [
                [D, x / Omega],
                [gamma * P * x, D],
            ],
            dtype=float,
        )

        rhs = np.array(
            [
                U - U**2 - (b + 2.0) * P / Omega,
                -P * (E + gamma * (a + (b + 3.0) * U)),
            ],
            dtype=float,
        )

        dU_dx, dP_dx = np.linalg.solve(matrix, rhs)

        dOmega_dx = Omega * (-x * dU_dx - a - (b + 3.0) * U) / D

        return np.array([dU_dx, dOmega_dx, dP_dx], dtype=float)

    def _solve_side(side):
        y0 = _shock_state(side)

        if side == "outer":
            # Start at R_fs and integrate inward to R_c.
            x_span = (1.0, 1.0e-8)
            target_U = expansion_index - contact_epsilon
            direction = 1.0
            max_step = 1.0e-3

        else:
            # Start at R_rs and integrate outward to R_c.
            x_span = (1.0, 100.0)
            target_U = expansion_index + contact_epsilon
            direction = -1.0
            max_step = 1.0e-3

        def event_contact(x, y):
            return y[0] - target_U

        event_contact.terminal = True
        event_contact.direction = direction

        solution = solve_ivp(
            lambda x, y: _rhs(x, y, side),
            x_span,
            y0,
            method="DOP853",
            events=event_contact,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
        )

        if solution.status != 1:
            raise RuntimeError(
                f"The {side} self-similar integration did not reach the "
                f"contact condition. Last x={solution.t[-1]:.6e}, "
                f"last U={solution.y[0, -1]:.6e}, target U={target_U:.6e}."
            )

        x_contact = solution.t_events[0][0]
        y_contact = solution.y_events[0][0]

        radius_shock_over_radius_cd = 1.0 / x_contact
        xi = solution.t * radius_shock_over_radius_cd

        a, b = _powers(side)
        pressure_contact_over_shock = x_contact ** (b + 2.0) * y_contact[2] / y0[2]

        order = np.argsort(xi)

        return {
            "x_contact": x_contact,
            "radius_shock_over_radius_cd": radius_shock_over_radius_cd,
            "pressure_contact_over_shock": pressure_contact_over_shock,
            "xi": xi[order],
            "U": solution.y[0][order],
            "Omega": solution.y[1][order],
            "P": solution.y[2][order],
            "U_shock": y0[0],
            "Omega_shock": y0[1],
            "P_shock": y0[2],
        }

    outer = _solve_side("outer")
    inner = _solve_side("inner")

    radius_fs_over_radius_cd = outer["radius_shock_over_radius_cd"]
    radius_rs_over_radius_cd = inner["radius_shock_over_radius_cd"]

    # C1 = p_c / p_1 and C2 = p_c / p_2.
    C1 = outer["pressure_contact_over_shock"]
    C2 = inner["pressure_contact_over_shock"]

    # Pressure matching fixes Chevalier's dimensionless normalization A.
    A = (
        (C2 / C1)
        * radius_rs_over_radius_cd ** (2.0 - n)
        * radius_fs_over_radius_cd ** (s - 2.0)
        * ((3.0 - s) / (n - 3.0)) ** 2
    )

    # Avoid evaluating exactly at the contact. The two one-sided solutions
    # are separately well behaved, but density/temperature can be singular.
    n_inner = max(2, n_points // 2)
    n_outer = max(2, n_points - n_inner)

    xi_inner = np.linspace(
        radius_rs_over_radius_cd,
        1.0 - contact_epsilon,
        n_inner,
    )
    xi_outer = np.linspace(
        1.0 + contact_epsilon,
        radius_fs_over_radius_cd,
        n_outer,
    )

    interp_inner_U = PchipInterpolator(inner["xi"], inner["U"], extrapolate=False)
    interp_inner_Omega = PchipInterpolator(inner["xi"], inner["Omega"], extrapolate=False)
    interp_inner_P = PchipInterpolator(inner["xi"], inner["P"], extrapolate=False)

    interp_outer_U = PchipInterpolator(outer["xi"], outer["U"], extrapolate=False)
    interp_outer_Omega = PchipInterpolator(outer["xi"], outer["Omega"], extrapolate=False)
    interp_outer_P = PchipInterpolator(outer["xi"], outer["P"], extrapolate=False)

    U_inner = interp_inner_U(xi_inner)
    Omega_inner = interp_inner_Omega(xi_inner)
    P_inner = interp_inner_P(xi_inner)

    U_outer = interp_outer_U(xi_outer)
    Omega_outer = interp_outer_Omega(xi_outer)
    P_outer = interp_outer_P(xi_outer)

    xi = np.concatenate([xi_inner, xi_outer])
    U = np.concatenate([U_inner, U_outer])
    Omega = np.concatenate([Omega_inner, Omega_outer])
    P = np.concatenate([P_inner, P_outer])

    # Contact-radius-scaled profiles. These are useful for plotting the full
    # solution without choosing a physical K_csm, K_ej, R_c, or time.
    density_hat_inner = (1.0 / A) * xi_inner ** (-n) * Omega_inner
    density_hat_outer = xi_outer ** (-s) * Omega_outer

    pressure_hat_inner = (1.0 / A) * xi_inner ** (2.0 - n) * P_inner
    pressure_hat_outer = xi_outer ** (2.0 - s) * P_outer

    velocity_hat_inner = xi_inner * U_inner
    velocity_hat_outer = xi_outer * U_outer

    density_hat = np.concatenate([density_hat_inner, density_hat_outer])
    pressure_hat = np.concatenate([pressure_hat_inner, pressure_hat_outer])
    velocity_hat = np.concatenate([velocity_hat_inner, velocity_hat_outer])

    return ChevalierSelfSimilarFunctions(
        n=float(n),
        s=float(s),
        gamma=float(gamma),
        expansion_index=float(expansion_index),
        similarity_exponent=float(similarity_exponent),
        A=float(A),
        radius_fs_over_radius_cd=float(radius_fs_over_radius_cd),
        radius_rs_over_radius_cd=float(radius_rs_over_radius_cd),
        xi=xi,
        U=U,
        Omega=Omega,
        P=P,
        density_hat=density_hat,
        pressure_hat=pressure_hat,
        velocity_hat=velocity_hat,
        xi_inner=xi_inner,
        U_inner=U_inner,
        Omega_inner=Omega_inner,
        P_inner=P_inner,
        xi_outer=xi_outer,
        U_outer=U_outer,
        Omega_outer=Omega_outer,
        P_outer=P_outer,
    )


def compute_self_similar_critical_grid(
    n_values: "_ArrayLike",
    s_values: "_ArrayLike",
    gamma: float = 5.0 / 3.0,
    *,
    show_progress: bool = True,
    contact_epsilon: float = 1.0e-6,
    rtol: float = 1.0e-9,
    atol: float = 1.0e-11,
) -> ChevalierSelfSimilarCriticalGrid:
    r"""
    Compute critical Chevalier self-similar constants on an ``(n, s)`` grid.

    This function evaluates :func:`compute_self_similar_functions` over a grid
    of outer ejecta indices ``n`` and CSM indices ``s``, retaining only the
    dimensionless constants needed for the two-shock kinematics:

    .. math::

        A,\qquad R_{\rm fs}/R_{\rm cd},\qquad R_{\rm rs}/R_{\rm cd}.

    Parameters
    ----------
    n_values : array-like
        Outer ejecta density indices. Each value must satisfy ``n > 5``.
    s_values : array-like
        CSM density indices. Each value must satisfy ``s < 3``.
    gamma : float, optional
        Adiabatic index of the shocked gas.
    show_progress : bool, optional
        If ``True``, use ``tqdm`` to show progress when available.
    contact_epsilon : float, optional
        Offset used by :func:`compute_self_similar_functions` to stop before
        the contact singularity.
    rtol, atol : float, optional
        Relative and absolute solver tolerances.

    Returns
    -------
    ChevalierSelfSimilarCriticalGrid
        Named tuple containing the grid values of ``A``, ``R_fs/R_cd``, and
        ``R_rs/R_cd``.
    """
    n_values = np.asarray(n_values, dtype=float)
    s_values = np.asarray(s_values, dtype=float)

    shape = (n_values.size, s_values.size)

    expansion_index = np.full(shape, np.nan, dtype=float)
    similarity_exponent = np.full(shape, np.nan, dtype=float)

    A = np.full(shape, np.nan, dtype=float)
    radius_fs_over_radius_cd = np.full(shape, np.nan, dtype=float)
    radius_rs_over_radius_cd = np.full(shape, np.nan, dtype=float)

    grid_indices = [(i, j) for i in range(n_values.size) for j in range(s_values.size)]

    if show_progress:
        try:
            from tqdm.auto import tqdm

            grid_indices = tqdm(
                grid_indices,
                desc="Computing Chevalier similarity constants",
            )
        except ImportError:
            pass

    for i, j in grid_indices:
        n = n_values[i]
        s = s_values[j]

        solution = compute_self_similar_functions(
            n=n,
            s=s,
            gamma=gamma,
            n_points=8,
            contact_epsilon=contact_epsilon,
            rtol=rtol,
            atol=atol,
        )

        expansion_index[i, j] = solution.expansion_index
        similarity_exponent[i, j] = solution.similarity_exponent

        A[i, j] = solution.A
        radius_fs_over_radius_cd[i, j] = solution.radius_fs_over_radius_cd
        radius_rs_over_radius_cd[i, j] = solution.radius_rs_over_radius_cd

    return ChevalierSelfSimilarCriticalGrid(
        n_values=n_values,
        s_values=s_values,
        gamma=float(gamma),
        expansion_index=expansion_index,
        similarity_exponent=similarity_exponent,
        A=A,
        radius_fs_over_radius_cd=radius_fs_over_radius_cd,
        radius_rs_over_radius_cd=radius_rs_over_radius_cd,
    )


# ==================================================== #
# Chevalier Self-Similar Shock Engine                  #
# ==================================================== #
class ChevalierShockState(NamedTuple):
    r"""
    Time-dependent state returned by Chevalier self-similar shock engines.

    The low-level CGS interface returns plain :class:`numpy.ndarray` fields in
    CGS units. The public unit-aware interface returns the same structure with
    fields converted to :class:`astropy.units.Quantity`.

    .. note::

        Post-shock thermodynamic quantities are evaluated using
        :class:`~triceratops.dynamics.shocks.core.rankine_hugoniot.StrongColdShockConditions`
        with the contact-discontinuity velocity :math:`v_{\rm cd}` as the shock velocity.
        This follows the standard convention in the supernova radio-afterglow literature
        but is an approximation: the true forward shock outruns the contact discontinuity
        by a factor that depends on :math:`n` and :math:`s`.  The full two-surface
        solution will be provided in a future implementation.
    """

    radius: Union[np.ndarray, u.Quantity]
    r"""
    Shock contact-discontinuity radius :math:`R(t)` in cm.
    """

    velocity: Union[np.ndarray, u.Quantity]
    r"""
    Shock contact-discontinuity velocity :math:`v_{\rm cd}(t) = \dot{R}(t)` in cm/s.
    Also used as the shock velocity for the Rankine--Hugoniot post-shock quantities.
    """

    post_shock_density: Union[np.ndarray, u.Quantity]
    r"""
    Immediate post-shock density :math:`\rho_s` in :math:`\mathrm{g\,cm^{-3}}`,
    from the strong cold-shock Rankine--Hugoniot relation applied at :math:`R(t)`.
    """

    post_shock_pressure: Union[np.ndarray, u.Quantity]
    r"""
    Immediate post-shock pressure :math:`p_s` in :math:`\mathrm{dyn\,cm^{-2}}`,
    from the strong cold-shock Rankine--Hugoniot relation applied at :math:`R(t)`.
    """

    post_shock_temperature: Union[np.ndarray, u.Quantity]
    r"""
    Immediate post-shock temperature :math:`T_s` in K,
    from the strong cold-shock Rankine--Hugoniot relation applied at :math:`R(t)`.
    """

    thermal_energy_density: Union[np.ndarray, u.Quantity]
    r"""
    Post-shock thermal energy density
    :math:`e_{\rm th} = p_s / (\gamma - 1)` in :math:`\mathrm{erg\,cm^{-3}}`.
    """


class ChevalierSelfSimilarShockEngine(ShockEngine):
    r"""
    Implementation of the "classical" Chevalier 1982 self-similar supernova shock model.

    This :class:`~triceratops.dynamics.shock_engine.ShockEngine` subclass implements the self-similar shock solutions
    described in :footcite:t:`chevalierSelfsimilarSolutionsInteraction1982` for the interaction between
    supernova ejecta and a surrounding circumstellar medium (CSM). The model assumes power-law density profiles
    for both the ejecta and the CSM, leading to a self-similar evolution of the shock structure over time.

    The engine calculates the position and velocity of the shock-interface as functions of time, based on
    the ejecta and CSM density profiles. It can be used to model the dynamical evolution of supernova remnants
    and their interaction with the surrounding medium.

    .. note::

        A derivation of this model can be found on the :ref:`chevalier_theory` guide. Much of the
        relevant detail omitted here can be found there.

    Notes
    -----

    .. important::

        This model is based off of the classical self-similar solutions for supernova ejecta interacting
        with a circumstellar medium as described in
        :footcite:t:`chevalierSelfsimilarSolutionsInteraction1982`, and a discussion of their relevance
        for synchrotron emission from supernovae may be found in
        :footcite:t:`ChevalierXRayRadioEmission1982`. The implementation here follows these sources quite closely
        with some moderate improvements to the formalism.

    **Model Assumptions:**

    This model assumes the following:

    - The supernova ejecta is expanding homologously such that :math:`r = vt`.
    - As required by homologous expansion, the ejecta density profile must be a function of the velocity of the
      form

      .. math::

            \rho(r,t) = t^{-3} f\left(\frac{r}{t}\right),

      for some, generally unknown, function :math:`f(v)`.

      In this model, the supernova ejecta density profile follows a broken power-law in velocity space such that

      .. math::

            \rho_{\rm ej}(r,t) = K_{\rm ej} t^{-3} \begin{cases}
                v^{-\delta}, & v < v_t \\
                v_t^{n-\delta} v^{-n}, & v \geq v_t,
            \end{cases}

      where :math:`v_t` is the transition velocity between the inner and outer ejecta profiles, and :math:`K` is
      the normalization constant.

    - The circumstellar medium (CSM) surrounding the supernova progenitor follows a power-law density profile of the
      form

      .. math::

            \rho_{\rm CSM}(r) = K_{\rm CSM} r^{-s},

      where :math:`\rho_0` is the normalization constant and :math:`s` is the CSM density power-law index.

    - The interaction between the ejecta and the CSM produces a forward and reverse shock structure that evolves
      self-similarly over time and maintains the shock within a region suitable for thin-shell approximation.

    **Model Features**

    Under the previously described assumptions, the position of the discontinuity surface between the forward and
    reverse shocks evolves self-similarly as :math:`R(t)` such that

    .. math::

        R(t) = \left(\frac{\zeta K_{\rm CSM}}{K_{\rm ej}}\right)^{\frac{1}{s-n}} t^{\frac{3-n}{s-n}},

    where :math:`A` is a dimensionless constant that depends on the ejecta and CSM density power-law indices
    :math:`n` and :math:`s`, respectively. From conservation of momentum, it can be derived that

    .. math::

        \zeta = \frac{3\lambda^2 +4\frac{ (\lambda -1)\lambda}{3-s} }
                    { 3(1-\lambda)^2-4\frac{ (\lambda  -1)\lambda}{n-3}}.

    is a factor generally of order unity.

    **Connection to Energetics**:

    The ejecta normalization constant :math:`K_{\rm ej}` and transition velocity :math:`v_t` can be computed
    from the total ejecta kinetic energy :math:`E_{\rm ej}` and ejecta mass :math:`M_{\rm ej}` instead of requiring
    them as direct inputs.

    See Also
    --------
    ChevalierSelfSimilarWindShockEngine
        Specialized version of :class:`ChevalierSelfSimilarShockEngine` for steady-wind CSM.
    NumericalThinShellShockEngine
        A numerical implementation of thin-shell shock dynamics for arbitrary ejecta and CSM profiles.

    References
    ----------
    .. footbibliography::

    """

    _STATE_CLASS = ChevalierShockState

    # =========================================== #
    # Initialization                              #
    # =========================================== #
    def __init__(self, gamma: float = 5.0 / 3.0, mu: float = 0.5, **kwargs):
        """
        Instantiate the :class:`ChevalierSelfSimilarShockEngine`.

        Parameters
        ----------
        gamma : float, optional
            Adiabatic index of the shocked gas. Default is :math:`5/3` (monatomic ideal gas).
        mu : float, optional
            Mean molecular weight in units of the proton mass used for the post-shock
            temperature calculation. Default is ``0.5`` (fully ionized hydrogen).
        **kwargs
            Additional keyword arguments forwarded to the parent
            :class:`~triceratops.dynamics.shocks.core.shock_engine.ShockEngine`.
        """
        super().__init__(**kwargs)
        self._gamma = float(gamma)
        self._mu = float(mu)

    # ============================================================= #
    # Supplementary Numerical Methods                               #
    # ============================================================= #
    @staticmethod
    def compute_scale_parameter(n: float, s: float):
        r"""
        Compute the radius scale parameter for the Chevalier self-similar shock solution.

        This function computes the dimensionless :math:`\zeta` parameter that appears in the
        radius normalization of the Chevalier self-similar shock solution. This parameter
        depends on the ejecta density power-law index :math:`n` and the CSM density power-law
        index :math:`s`.

        Parameters
        ----------
        n: float
            The outer ejecta density profile power-law index.
        s: float
            The CSM density profile power-law index.

        Returns
        -------
        float
            The computed scale parameter :math:`\zeta`.

        Notes
        -----
        The equation for :math:`\zeta` is

        .. math::

            \zeta = \frac{3\lambda^2 +4\frac{ (\lambda -1)\lambda}{3-s} }
                        { 3(1-\lambda)^2-4\frac{ (\lambda  -1)\lambda}{n-3}},

        where :math:`\lambda = \frac{3-n}{s-n}`. A derivation of this parameter can be found in
        :ref:`chevalier_theory`.
        """
        # Construct lambda from n and s.
        _lambda = (3 - n) / (s - n)

        # Compute the A, B, C, and D terms
        A = 4 * _lambda * (_lambda - 1) / (3 - s)
        B = 4 * _lambda * (_lambda - 1) / (n - 3)
        C = 3 * _lambda**2
        D = 3 * (1 - _lambda) ** 2

        return (C + A) / (D - B)

    # ============================================================ #
    # Core Shock Engine Methods                                    #
    # ============================================================ #
    def compute_shock_properties(
        self,
        time: "_UnitBearingArrayLike",
        E_ej: "_UnitBearingScalarLike" = 1e51 * u.erg,
        M_ej: "_UnitBearingScalarLike" = 10 * u.Msun,
        K_csm: "_UnitBearingScalarLike" = None,
        n: float = 10.0,
        s: float = 2.0,
        delta: float = 0.0,
    ) -> ChevalierShockState:
        r"""
        Compute the shock properties at a given time.

        This method calculates the shock radius and velocity as a function of time since the
        explosion. See the class documentation (:class:`ChevalierSelfSimilarShockEngine`) for details
        on the relevant theory and assumptions.

        Parameters
        ----------
        time: ~astropy.units.Quantity or float or numpy.ndarray
            The time(s) at which to evaluate the shock properties. If units are provided,
            they will be taken into account. Otherwise, CGS units (seconds) are assumed.
            If ``time`` is provided as an array of shape ``(N,)``, the results will all have
            corresponding shapes ``(N,)``.
        E_ej: ~astropy.units.Quantity or float
            The total energy in the ejecta from the explosion. If units are provided,
            they will be taken into account. Otherwise, CGS units (erg) are assumed.
        M_ej: ~astropy.units.Quantity or float
            The total mass in the ejecta from the explosion. If units are provided,
            they will be taken into account. Otherwise, CGS units (grams) are assumed.
        K_csm: ~astropy.units.Quantity or float, optional
            The scaling (:math:`K_{\rm CSM}`) for the CSM density profile of the form
            :math:`\rho_{\rm CSM}(r) = K_{\rm CSM} r^{-s}`. If units are provided, they will be
            taken into account. Otherwise, CGS units (``g * cm^{(s-3)}``) are assumed. If not provided,
            a default scaling based on a wind-like CSM with :math:`\dot{M} \sim 10^{-5} M_{\odot}/yr`
            and :math:`v_w \sim 1000 km/s` is used at a radius of :math:`r = 10^{16} cm`.

            .. note::

                For science scenarios, ``K_csm`` should always be provided explicitly to ensure
                physical accuracy. The default is only a placeholder.
        n: float, optional
            The outer ejecta density profile power-law index. Default is ``10.0``. Must be steeper than
            5 for convergence.
        s: float, optional
            The CSM density profile power-law index. Default is ``2.0``.
        delta: float, optional
            The inner ejecta density profile power-law index. Default is ``0.0``. Must be less than
            3 for convergence. In general, it is suitable to use ``0.0``.

        Returns
        -------
        ChevalierShockState
            Shock kinematics and post-shock thermodynamics at the requested time(s),
            with :class:`~astropy.units.Quantity` units attached to every field.

        """
        # Validate inputs and determine a scaling for K_csm if one
        # is not provided. To do this, we set a standard density based on a wind-like
        # density profile. THIS IS PURELY A MEANS FOR PICKING A DEFAULT, IT SHOULD
        # NOT BE USED IN SCIENCE RUNS.
        if K_csm is None:
            # Assume a generic wind-like CSM with M_dot ~ 1e-5 Msun/yr and v_w ~ 1000 km/s scaled
            # at r = 1e16 cm.
            K_csm = ((1e16 * u.cm) ** (s - 2)) * (1e-5 * u.Msun / u.yr) / (4.0 * np.pi * (1000 * u.km / u.s))

        # Scale everything down to CGS for internal computation.
        if isinstance(E_ej, u.Quantity):
            E_ej = E_ej.to(u.erg).value
        if isinstance(M_ej, u.Quantity):
            M_ej = M_ej.to(u.g).value
        if isinstance(time, u.Quantity):
            time = time.to(u.s).value
        if isinstance(K_csm, u.Quantity):
            K_csm = K_csm.to(u.g * u.cm ** (s - 3)).value

        # Perform checks on ``n``, ``s``, and ``delta`` to ensure convergence.
        if delta >= 3:
            raise ValueError("The inner ejecta density profile index `delta` must be less than 3 for convergence.")
        if n <= 5:
            raise ValueError("The outer ejecta density profile index `n` must be greater than 5 for convergence.")

        # Call the internal CGS computation method.
        shock_properties_cgs = self._compute_shock_properties_cgs(
            time=time, E_ej=E_ej, M_ej=M_ej, K_csm=K_csm, n=n, s=s, delta=delta
        )

        return ChevalierShockState(
            radius=shock_properties_cgs.radius * u.cm,
            velocity=shock_properties_cgs.velocity * (u.cm / u.s),
            post_shock_density=shock_properties_cgs.post_shock_density * (u.g / u.cm**3),
            post_shock_pressure=shock_properties_cgs.post_shock_pressure * (u.dyne / u.cm**2),
            post_shock_temperature=shock_properties_cgs.post_shock_temperature * u.K,
            thermal_energy_density=shock_properties_cgs.thermal_energy_density * (u.erg / u.cm**3),
        )

    def _compute_shock_properties_cgs(
        self,
        time: "_ArrayLike",
        E_ej: float = 1e51,
        M_ej: float = 1e34,
        K_csm: float = 5e11,
        n: float = 10.0,
        s: float = 2.0,
        delta: float = 0.0,
    ):
        r"""
        Compute the shock properties at a given time in CGS units.

        This method computes the shock radius and velocity at a given time based on the
        Chevalier self-similar solution for supernova ejecta interacting with a circumstellar
        medium (CSM). The ejecta and CSM are assumed to follow power-law density profiles.

        Parameters
        ----------
        time: array-like
            The time(s) at which to evaluate the shock properties in seconds. This can be a scalar or
            an array of times. The results will match the shape of the input time array.
        E_ej: float
            The total kinetic energy of the ejecta in erg. Default is ``1e51``.
        M_ej: float
            The total mass of the ejecta in grams. Default is ``1e34``.
        K_csm: float
            The normalization constant of the CSM density profile in CGS units ``g * cm^{(s-3}}``.
        n: float
            The outer ejecta density profile power-law index. Default is ``10.0``.
        s: float
            The CSM density profile power-law index. Default is ``2.0``.
        delta:
            The inner ejecta density profile power-law index. Default is ``0.0``.

        Returns
        -------
        ChevalierShockState
            All fields are plain :class:`numpy.ndarray` values in CGS units.

        """
        # Using the ``_compute_v_t_and_K_from_energetics_cgs`` static method to get v_t and K. We can
        # discard v_t, but K is necessary.
        v_t, K = _normalize_BPL_ejecta(
            E_ej=E_ej,
            M_ej=M_ej,
            n=n,
            delta=delta,
        )

        # Correct K since it needs to be multiplied by v_t^(n - delta) for the outer ejecta profile.
        K_EJ = K * v_t ** (n - delta)

        # Compute relevant factors: We of course need ``_lambda`` (the scaling in time of the radius),
        # ``_gamma`` appears in the radius scale factor, as does the ``_lambda_gamma_constant``.
        _lambda = (3 - n) / (s - n)
        SCALE_CONSTANT = self.compute_scale_parameter(n=n, s=s)
        R_0 = (SCALE_CONSTANT * K_csm / K_EJ) ** (1 / (s - n))

        # Compute the shock radius and velocity at the given time(s).
        shock_radius = R_0 * time**_lambda
        shock_velocity = _lambda * shock_radius / time

        # Upstream CSM density at the contact discontinuity.
        rho_upstream = K_csm * shock_radius ** (-s)

        # Post-shock thermodynamics via the strong cold-shock RH conditions.
        # v_cd is used as the shock velocity following the SN literature convention.
        rh = StrongColdShockConditions._solve(shock_velocity, rho_upstream, gamma=self._gamma, mu=self._mu)

        return ChevalierShockState(
            radius=shock_radius,
            velocity=shock_velocity,
            post_shock_density=rh["post_shock_density"],
            post_shock_pressure=rh["post_shock_pressure"],
            post_shock_temperature=rh["post_shock_temperature"],
            thermal_energy_density=rh["post_shock_thermal_energy_density"],
        )

    # =========================================== #
    # UTILITIES                                   #
    # =========================================== #
    @staticmethod
    def normalize_csm_density(
        rho_0: "_UnitBearingScalarLike",
        r_0: "_UnitBearingScalarLike",
        s: float,
    ) -> "_UnitBearingScalarLike":
        r"""
        Compute the CSM density normalization constant from a reference density.

        This function computes the normalization constant :math:`K_{\rm CSM}` for a power-law
        circumstellar medium (CSM) density profile of the form

        .. math::

            \rho_{\rm CSM}(r) = K_{\rm CSM} r^{-s} = \rho_0 \left(\frac{r}{r_0}\right)^{-s},

        given a reference density :math:`\rho_0` at a reference radius :math:`r_0`.

        Parameters
        ----------
        rho_0: astropy.units.Quantity or float
            The reference density of the CSM at radius :math:`r_0`. If units are provided,
            they will be taken into account. Otherwise, CGS units (``grams/cm^3``) are assumed.
        r_0: astropy.units.Quantity or float
            The reference radius at which the density is specified. If units are provided,
            they will be taken into account. Otherwise, CGS units (``cm``) are assumed.
        s: float
            The CSM density profile power-law index.

        Returns
        -------
        K_csm: astropy.units.Quantity
            The computed normalization constant :math:`K_{\rm CSM}` with units of
            ``grams * cm^(s-3)``.

        """
        # Convert inputs to CGS for internal computation.
        if isinstance(rho_0, u.Quantity):
            rho_0 = rho_0.to(u.g / u.cm**3).value
        if isinstance(r_0, u.Quantity):
            r_0 = r_0.to(u.cm).value

        # Compute K_csm in CGS.
        K_csm_cgs = rho_0 * r_0**s

        # Attach units to K_csm.
        K_csm_units = u.g * u.cm ** (s - 3)
        K_csm = K_csm_cgs * K_csm_units

        return K_csm

    @staticmethod
    def normalize_outer_ejecta_density(
        rho_0: "_UnitBearingScalarLike",
        v_0: "_UnitBearingScalarLike",
        t_0: "_UnitBearingScalarLike",
        n: float,
    ) -> "_UnitBearingScalarLike":
        r"""
        Compute the ejecta density normalization constant from a reference density.

        This function computes the normalization constant :math:`K_{\rm ej}` for a power-law
        outer ejecta density profile of the form

        .. math::

            \rho_{\rm ej}(r,t) = K_{\rm ej} t^{-3} \left(\frac{r}{t}\right)^{-n} =
            \rho_0 \left(\frac{r/t}{v_0}\right)^{-n} left(\frac{t}{t_0}\right)^{-3},

        given a reference density :math:`\rho_0` at a reference velocity :math:`v_0` and time :math:`t_0`.

        Parameters
        ----------
        rho_0: astropy.units.Quantity or float
            The reference density of the ejecta at velocity :math:`v_0` and time :math:`t_0`. If units are provided,
            they will be taken into account. Otherwise, CGS units (``grams/cm^3``) are assumed.
        v_0: astropy.units.Quantity or float
            The reference velocity at which the density is specified. If units are provided,
            they will be taken into account. Otherwise, CGS units (``cm/s``) are assumed.
        t_0: astropy.units.Quantity or float
            The reference time at which the density is specified. If units are provided,
            they will be taken into account. Otherwise, CGS units (``s``) are assumed.
        n: float
            The outer ejecta density profile power-law index.

        Returns
        -------
        K_ej: astropy.units.Quantity
            The computed normalization constant :math:`K_{\rm ej}` with units of
            ``grams * cm^(n-3) * s^(3-n)``.

        """
        # Convert inputs to CGS for internal computation.
        if isinstance(rho_0, u.Quantity):
            rho_0 = rho_0.to(u.g / u.cm**3).value
        if isinstance(v_0, u.Quantity):
            v_0 = v_0.to(u.cm / u.s).value
        if isinstance(t_0, u.Quantity):
            t_0 = t_0.to(u.s).value

        # Compute K_ej in CGS.
        K_ej_cgs = rho_0 * v_0**n * t_0**3

        # Attach units to K_ej.
        K_ej_units = u.g * u.cm ** (n - 3) * u.s ** (3 - n)
        K_ej = K_ej_cgs * K_ej_units

        return K_ej


class ChevalierSelfSimilarWindShockEngine(ChevalierSelfSimilarShockEngine):
    r"""
    Specialized version of :class:`ChevalierSelfSimilarShockEngine` for steady-wind CSM.

    In the case of a steady wind CSM with injection rate :math:`\dot{M}` and characteristic velocity
    :math:`v_{\rm wind}`, the corresponding CSM density profile is

    .. math::

        \rho_{\rm CSM}(r) = \frac{\dot{M}}{4\pi r^2 v_{\rm wind}}.

    This corresponds to a conventional :class:`ChevalierSelfSimilarShockEngine` with CSM density power-law index
    :math:`s = 2` and normalization constant

    .. math::

        K_{\rm CSM} = \frac{\dot{M}}{4\pi v_{\rm wind}}.

    .. note::

        A derivation of this model can be found on the :ref:`chevalier_theory` guide. Much of the
        relevant detail omitted here can be found there.

    """

    # =========================================== #
    # Initialization                              #
    # =========================================== #
    def __init__(self, gamma: float = 5.0 / 3.0, mu: float = 0.5, **kwargs):
        """
        Instantiate the :class:`ChevalierSelfSimilarWindShockEngine`.

        Parameters
        ----------
        gamma : float, optional
            Adiabatic index of the shocked gas. Forwarded to
            :class:`ChevalierSelfSimilarShockEngine`. Default is :math:`5/3`.
        mu : float, optional
            Mean molecular weight in units of the proton mass. Forwarded to
            :class:`ChevalierSelfSimilarShockEngine`. Default is ``0.5``.
        **kwargs
            Additional keyword arguments forwarded to the parent engine.
        """
        super().__init__(gamma=gamma, mu=mu, **kwargs)

    def compute_shock_properties(
        self,
        time: "_UnitBearingArrayLike",
        E_ej: "_UnitBearingScalarLike" = 1e51 * u.erg,
        M_ej: "_UnitBearingScalarLike" = 10 * u.Msun,
        M_dot: "_UnitBearingScalarLike" = 1e-5 * u.Msun / u.yr,
        v_wind: "_UnitBearingScalarLike" = 1000 * u.km / u.s,
        n: float = 10.0,
        delta: float = 0.0,
    ) -> ChevalierShockState:
        """
        Compute the shock properties at a given time.

        This method calculates the shock radius and velocity as a function of time since the
        explosion. See the class documentation (:class:`ChevalierSelfSimilarShockEngine`) for details
        on the relevant theory and assumptions.

        Parameters
        ----------
        time: ~astropy.units.Quantity or float or numpy.ndarray
            The time(s) at which to evaluate the shock properties. If units are provided,
            they will be taken into account. Otherwise, CGS units (seconds) are assumed.
            If ``time`` is provided as an array of shape ``(N,)``, the results will all have
            corresponding shapes ``(N,)``.
        E_ej: ~astropy.units.Quantity or float
            The total energy in the ejecta from the explosion. If units are provided,
            they will be taken into account. Otherwise, CGS units (erg) are assumed.
        M_ej: ~astropy.units.Quantity or float
            The total mass in the ejecta from the explosion. If units are provided,
            they will be taken into account. Otherwise, CGS units (grams) are assumed.
        M_dot: ~astropy.units.Quantity or float
            The mass-loss rate of the progenitor star's wind. If units are provided,
            they will be taken into account. Otherwise, CGS units (grams/second) are assumed.
            By default, this is set to ``1e-5 Msun/yr``.
        v_wind: ~astropy.units.Quantity or float
            The velocity of the progenitor star's wind. If units are provided,
            they will be taken into account. Otherwise, CGS units (cm/second) are assumed. By
            default, this is set to ``1000 km/s``.
        n: float, optional
            The outer ejecta density profile power-law index. Default is ``10.0``. Must be steeper than
            5 for convergence.
        delta: float, optional
            The inner ejecta density profile power-law index. Default is ``0.0``. Must be less than
            3 for convergence. In general, it is suitable to use ``0.0``.

        Returns
        -------
        ChevalierShockState
            Shock kinematics and post-shock thermodynamics at the requested time(s),
            with :class:`~astropy.units.Quantity` units attached to every field.

        """
        # Scale everything down to CGS for internal computation.
        if isinstance(E_ej, u.Quantity):
            E_ej = E_ej.to(u.erg).value
        if isinstance(M_ej, u.Quantity):
            M_ej = M_ej.to(u.g).value
        if isinstance(time, u.Quantity):
            time = time.to(u.s).value
        if isinstance(M_dot, u.Quantity):
            M_dot = M_dot.to(u.g / u.s).value
        if isinstance(v_wind, u.Quantity):
            v_wind = v_wind.to(u.cm / u.s).value

        # Perform checks on ``n``, ``s``, and ``delta`` to ensure convergence.
        if delta >= 3:
            raise ValueError("The inner ejecta density profile index `delta` must be less than 3 for convergence.")
        if n <= 5:
            raise ValueError("The outer ejecta density profile index `n` must be greater than 5 for convergence.")

        # Call the internal CGS computation method.
        shock_properties_cgs = self._compute_shock_properties_cgs(
            time=time, E_ej=E_ej, M_ej=M_ej, M_dot=M_dot, v_wind=v_wind, n=n, delta=delta
        )

        return ChevalierShockState(
            radius=shock_properties_cgs.radius * u.cm,
            velocity=shock_properties_cgs.velocity * (u.cm / u.s),
            post_shock_density=shock_properties_cgs.post_shock_density * (u.g / u.cm**3),
            post_shock_pressure=shock_properties_cgs.post_shock_pressure * (u.dyne / u.cm**2),
            post_shock_temperature=shock_properties_cgs.post_shock_temperature * u.K,
            thermal_energy_density=shock_properties_cgs.thermal_energy_density * (u.erg / u.cm**3),
        )

    def _compute_shock_properties_cgs(
        self,
        time: "_ArrayLike",
        E_ej: float = 1e51,
        M_ej: float = 1e34,
        M_dot: float = 6.3e20,
        v_wind: float = 1e8,
        n: float = 10.0,
        delta: float = 0.0,
    ):
        """
        Compute the shock properties at a given time in CGS units.

        This method computes the shock radius and velocity at a given time based on the
        Chevalier self-similar solution for supernova ejecta interacting with a circumstellar
        medium (CSM). The ejecta and CSM are assumed to follow power-law density profiles.

        Parameters
        ----------
        time: array-like
            The time(s) at which to evaluate the shock properties in seconds. This can be a scalar or
            an array of times. The results will match the shape of the input time array.
        E_ej: float
            The total kinetic energy of the ejecta in erg. Default is ``1e51``.
        M_ej: float
            The total mass of the ejecta in grams. Default is ``1e34``.
        M_dot: float
            The mass-loss rate of the progenitor star's wind in grams/second. Default is
            ``1e-5 Msun/yr`` in CGS units.
        v_wind: float
            The velocity of the progenitor star's wind in cm/second. Default is
            ``1000 km/s`` in CGS units.
        n: float
            The outer ejecta density profile power-law index. Default is ``10.0``.
        delta:
            The inner ejecta density profile power-law index. Default is ``0.0``.

        Returns
        -------
        ChevalierShockState
            All fields are plain :class:`numpy.ndarray` values in CGS units.

        """
        # Compute the correct K_CSM for the wind-like CSM
        K_csm = M_dot / (4.0 * np.pi * v_wind)

        # Now just pass off to the super-class
        return super()._compute_shock_properties_cgs(
            time=time,
            E_ej=E_ej,
            M_ej=M_ej,
            K_csm=K_csm,
            n=n,
            s=2.0,
            delta=delta,
        )


# ==================================================== #
# 2-Shock Chevalier Self-Similar Engines               #
# ==================================================== #
class ChevalierTwoShockState(NamedTuple):
    r"""
    Time-dependent two-shock state from :class:`ChevalierTwoShockSelfSimilarEngine`.

    All three surfaces --- contact discontinuity (cd), forward shock (fs), and
    reverse shock (rs) --- move self-similarly as :math:`R \propto t^\lambda` with

    .. math::

        \lambda = \frac{n - 3}{n - s}.

    Post-shock thermodynamic quantities follow from the strong cold-shock
    Rankine--Hugoniot relations applied at each surface with its correct
    upstream state:

    - **Forward shock**: upstream is stationary CSM at density
      :math:`\rho_{\rm CSM}(R_{\rm fs}) = K_{\rm CSM} R_{\rm fs}^{-s}`;
      shock velocity in the lab frame is :math:`v_{\rm fs} = \lambda R_{\rm fs}/t`.
    - **Reverse shock**: upstream ejecta flow outward at :math:`v_{\rm ej} = R_{\rm rs}/t`
      (homologous); the ejecta velocity relative to the shock surface is
      :math:`v_{\rm rel} = (1 - \lambda) R_{\rm rs}/t`, and the upstream ejecta density
      is :math:`\rho_{\rm ej}(R_{\rm rs}, t) = K_{\rm ej}\, t^{n-3} R_{\rm rs}^{-n}`.
    """

    radius_cd: Union[np.ndarray, u.Quantity]
    r"""Contact-discontinuity radius :math:`R_{\rm cd}(t)` in cm."""

    radius_fs: Union[np.ndarray, u.Quantity]
    r"""Forward-shock radius :math:`R_{\rm fs}(t) = \xi_{\rm fs} R_{\rm cd}` in cm."""

    radius_rs: Union[np.ndarray, u.Quantity]
    r"""Reverse-shock radius :math:`R_{\rm rs}(t) = \xi_{\rm rs} R_{\rm cd}` in cm."""

    velocity_cd: Union[np.ndarray, u.Quantity]
    r"""Lab-frame contact-discontinuity velocity :math:`v_{\rm cd} = \lambda R_{\rm cd}/t` in cm/s."""

    velocity_fs: Union[np.ndarray, u.Quantity]
    r"""Lab-frame forward-shock velocity :math:`v_{\rm fs} = \lambda R_{\rm fs}/t` in cm/s."""

    velocity_rs: Union[np.ndarray, u.Quantity]
    r"""Lab-frame reverse-shock velocity :math:`v_{\rm rs} = \lambda R_{\rm rs}/t` in cm/s."""

    thermal_energy_density_fs: Union[np.ndarray, u.Quantity]
    r"""Post-shock thermal energy density at the forward shock in :math:`\mathrm{erg\,cm^{-3}}`."""

    thermal_energy_density_rs: Union[np.ndarray, u.Quantity]
    r"""Post-shock thermal energy density at the reverse shock in :math:`\mathrm{erg\,cm^{-3}}`."""

    pressure_fs: Union[np.ndarray, u.Quantity]
    r"""Immediate post-shock pressure at the forward shock in :math:`\mathrm{dyn\,cm^{-2}}`."""

    pressure_rs: Union[np.ndarray, u.Quantity]
    r"""Immediate post-shock pressure at the reverse shock in :math:`\mathrm{dyn\,cm^{-2}}`."""

    temperature_fs: Union[np.ndarray, u.Quantity]
    r"""Immediate post-shock temperature at the forward shock in K."""

    temperature_rs: Union[np.ndarray, u.Quantity]
    r"""Immediate post-shock temperature at the reverse shock in K."""

    density_fs: Union[np.ndarray, u.Quantity]
    r"""Immediate post-shock density at the forward shock in :math:`\mathrm{g\,cm^{-3}}`."""

    density_rs: Union[np.ndarray, u.Quantity]
    r"""Immediate post-shock density at the reverse shock in :math:`\mathrm{g\,cm^{-3}}`."""


class ChevalierTwoShockSelfSimilarEngine(ShockEngine):
    r"""
    Two-shock Chevalier self-similar engine backed by a pre-tabulated ``(n, s)`` grid.

    Unlike :class:`ChevalierSelfSimilarShockEngine`, which applies Rankine--Hugoniot
    conditions only at the contact discontinuity, this engine resolves both the
    **forward shock** (into the CSM) and the **reverse shock** (into the ejecta)
    separately. The shock-radius ratios

    .. math::

        \xi_{\rm fs} = R_{\rm fs} / R_{\rm cd},
        \qquad
        \xi_{\rm rs} = R_{\rm rs} / R_{\rm cd}

    are read from a table of self-similar ODE solutions precomputed on a grid of
    ``(n, s)`` values at instantiation and bilinearly interpolated at runtime.
    The normalization constant :math:`A` (which determines the absolute contact-radius
    scale) is likewise tabulated and is more accurate than the analytic
    :meth:`ChevalierSelfSimilarShockEngine.compute_scale_parameter` approximation.

    The contact-discontinuity radius follows

    .. math::

        R_{\rm cd}(t) = \left(\frac{A\, K_{\rm ej}}{K_{\rm CSM}}\right)^{1/(n-s)}
                        t^{\lambda},

    where :math:`\lambda = (n-3)/(n-s)`. The Rankine--Hugoniot jump conditions are
    then applied at each shock surface with the appropriate upstream state
    (see :class:`ChevalierTwoShockState` for details).

    Parameters
    ----------
    gamma : float, optional
        Adiabatic index of the shocked gas. Default is :math:`5/3`.
    mu : float, optional
        Mean molecular weight in proton-mass units for the post-shock temperature.
        Default is ``0.5`` (fully ionized hydrogen).
    n_grid : array-like, optional
        Outer ejecta indices at which to tabulate the self-similar constants.
        All values must satisfy ``n > 5``. Defaults to ``np.arange(6.0, 15.0)``.
    s_grid : array-like, optional
        CSM indices at which to tabulate. All values must satisfy ``s < 3``.
        Defaults to ``[0.0, 0.5, 1.0, 1.5, 2.0, 2.5]``.
    contact_epsilon : float, optional
        Singularity offset forwarded to :func:`compute_self_similar_functions`.
    rtol, atol : float, optional
        ODE solver tolerances forwarded to :func:`compute_self_similar_functions`.
    show_progress : bool, optional
        Show a ``tqdm`` progress bar while building the table. Default ``False``.

    Notes
    -----
    Table construction calls :func:`compute_self_similar_functions` once per grid
    point. For the default 9 × 6 = 54-point grid this takes a few seconds on a
    modern machine.

    See Also
    --------
    ChevalierSelfSimilarShockEngine : Single-surface (contact-CD only) engine.
    ChevalierTwoShockSelfSimilarWindEngine : Wind-CSM specialization (``s = 2``).
    """

    _DEFAULT_N_GRID: np.ndarray = np.arange(6.0, 15.0, 1.0)
    _DEFAULT_S_GRID: np.ndarray = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    _STATE_CLASS = ChevalierTwoShockState

    # =========================================== #
    # Initialization                              #
    # =========================================== #
    def __init__(
        self,
        gamma: float = 5.0 / 3.0,
        mu: float = 0.5,
        n_grid: "_ArrayLike" = None,
        s_grid: "_ArrayLike" = None,
        contact_epsilon: float = 1.0e-6,
        rtol: float = 1.0e-9,
        atol: float = 1.0e-11,
        show_progress: bool = False,
        **kwargs,
    ):
        """
        Instantiate and pre-tabulate the two-shock Chevalier engine.

        Parameters
        ----------
        gamma : float, optional
            Adiabatic index.
        mu : float, optional
            Mean molecular weight.
        n_grid : array-like, optional
            Grid of outer ejecta indices (all must be ``> 5``).
        s_grid : array-like, optional
            Grid of CSM indices (all must be ``< 3``).
        contact_epsilon : float, optional
            Stop-before-contact offset for the ODE solver.
        rtol, atol : float, optional
            ODE solver tolerances.
        show_progress : bool, optional
            Display a progress bar during table construction.
        **kwargs
            Forwarded to :class:`~triceratops.dynamics.shocks.core.shock_engine.ShockEngine`.
        """
        super().__init__(**kwargs)
        self._gamma = float(gamma)
        self._mu = float(mu)

        n_values = np.asarray(n_grid if n_grid is not None else self._DEFAULT_N_GRID, dtype=float)
        s_values = np.asarray(s_grid if s_grid is not None else self._DEFAULT_S_GRID, dtype=float)

        self._table = compute_self_similar_critical_grid(
            n_values=n_values,
            s_values=s_values,
            gamma=self._gamma,
            show_progress=show_progress,
            contact_epsilon=contact_epsilon,
            rtol=rtol,
            atol=atol,
        )

        from scipy.interpolate import RegularGridInterpolator

        _pts = (n_values, s_values)
        self._interp_A = RegularGridInterpolator(_pts, self._table.A, method="linear", bounds_error=True)
        self._interp_xi_fs = RegularGridInterpolator(
            _pts, self._table.radius_fs_over_radius_cd, method="linear", bounds_error=True
        )
        self._interp_xi_rs = RegularGridInterpolator(
            _pts, self._table.radius_rs_over_radius_cd, method="linear", bounds_error=True
        )

    # =========================================== #
    # Table interpolation                         #
    # =========================================== #
    def _interpolate_constants(self, n: float, s: float):
        """Return ``(A, xi_fs, xi_rs)`` from the pre-tabulated grid.

        Raises
        ------
        ValueError
            If ``(n, s)`` falls outside the tabulated grid.
        """
        try:
            pt = np.array([[float(n), float(s)]])
            return (
                float(self._interp_A(pt)[0]),
                float(self._interp_xi_fs(pt)[0]),
                float(self._interp_xi_rs(pt)[0]),
            )
        except ValueError as exc:
            n_min, n_max = self._table.n_values[0], self._table.n_values[-1]
            s_min, s_max = self._table.s_values[0], self._table.s_values[-1]
            raise ValueError(
                f"(n, s) = ({n:.4g}, {s:.4g}) is outside the tabulated grid "
                f"n ∈ [{n_min:.4g}, {n_max:.4g}], s ∈ [{s_min:.4g}, {s_max:.4g}]. "
                "Pass a broader n_grid or s_grid when constructing the engine."
            ) from exc

    # =========================================== #
    # Unit attachment helper                      #
    # =========================================== #
    @staticmethod
    def _attach_units(cgs: "ChevalierTwoShockState") -> "ChevalierTwoShockState":
        """Wrap a plain-CGS :class:`ChevalierTwoShockState` in astropy units."""
        return ChevalierTwoShockState(
            radius_cd=cgs.radius_cd * u.cm,
            radius_fs=cgs.radius_fs * u.cm,
            radius_rs=cgs.radius_rs * u.cm,
            velocity_cd=cgs.velocity_cd * (u.cm / u.s),
            velocity_fs=cgs.velocity_fs * (u.cm / u.s),
            velocity_rs=cgs.velocity_rs * (u.cm / u.s),
            thermal_energy_density_fs=cgs.thermal_energy_density_fs * (u.erg / u.cm**3),
            thermal_energy_density_rs=cgs.thermal_energy_density_rs * (u.erg / u.cm**3),
            pressure_fs=cgs.pressure_fs * (u.dyne / u.cm**2),
            pressure_rs=cgs.pressure_rs * (u.dyne / u.cm**2),
            temperature_fs=cgs.temperature_fs * u.K,
            temperature_rs=cgs.temperature_rs * u.K,
            density_fs=cgs.density_fs * (u.g / u.cm**3),
            density_rs=cgs.density_rs * (u.g / u.cm**3),
        )

    # =========================================== #
    # Public interface                            #
    # =========================================== #
    def compute_shock_properties(
        self,
        time: "_UnitBearingArrayLike",
        E_ej: "_UnitBearingScalarLike" = 1e51 * u.erg,
        M_ej: "_UnitBearingScalarLike" = 10 * u.Msun,
        K_csm: "_UnitBearingScalarLike" = None,
        n: float = 10.0,
        s: float = 2.0,
        delta: float = 0.0,
    ) -> "ChevalierTwoShockState":
        r"""
        Compute the two-shock structure at one or more times.

        Parameters
        ----------
        time : ~astropy.units.Quantity or array-like
            Time since explosion. Bare float assumed to be in seconds.
        E_ej : ~astropy.units.Quantity or float
            Ejecta kinetic energy. Bare float assumed to be in erg.
        M_ej : ~astropy.units.Quantity or float
            Ejecta mass. Bare float assumed to be in grams.
        K_csm : ~astropy.units.Quantity or float, optional
            CSM normalization :math:`K_{\rm CSM}` in :math:`\mathrm{g\,cm^{s-3}}`.
            A wind-like default is used if omitted.
        n : float, optional
            Outer ejecta power-law index. Must lie within the tabulated ``n_grid``.
        s : float, optional
            CSM power-law index. Must lie within the tabulated ``s_grid``.
        delta : float, optional
            Inner ejecta power-law index. Must be less than 3. Default is ``0.0``.

        Returns
        -------
        ChevalierTwoShockState
            Shock positions, velocities, and post-shock thermodynamics at all
            requested times, with astropy units attached to every field.
        """
        if K_csm is None:
            K_csm = ((1e16 * u.cm) ** (s - 2)) * (1e-5 * u.Msun / u.yr) / (4.0 * np.pi * (1000 * u.km / u.s))

        if isinstance(E_ej, u.Quantity):
            E_ej = E_ej.to(u.erg).value
        if isinstance(M_ej, u.Quantity):
            M_ej = M_ej.to(u.g).value
        if isinstance(time, u.Quantity):
            time = time.to(u.s).value
        if isinstance(K_csm, u.Quantity):
            K_csm = K_csm.to(u.g * u.cm ** (s - 3)).value

        if delta >= 3:
            raise ValueError("The inner ejecta density profile index `delta` must be less than 3 for convergence.")
        if n <= 5:
            raise ValueError("The outer ejecta density profile index `n` must be greater than 5 for convergence.")

        cgs = self._compute_shock_properties_cgs(time=time, E_ej=E_ej, M_ej=M_ej, K_csm=K_csm, n=n, s=s, delta=delta)
        return self._attach_units(cgs)

    # =========================================== #
    # CGS backend                                 #
    # =========================================== #
    def _compute_shock_properties_cgs(
        self,
        time: "_ArrayLike",
        E_ej: float = 1e51,
        M_ej: float = 1e34,
        K_csm: float = 5e11,
        n: float = 10.0,
        s: float = 2.0,
        delta: float = 0.0,
    ) -> "ChevalierTwoShockState":
        r"""
        Compute the two-shock structure in CGS units.

        Parameters
        ----------
        time : array-like
            Time(s) in seconds.
        E_ej : float
            Ejecta kinetic energy in erg.
        M_ej : float
            Ejecta mass in grams.
        K_csm : float
            CSM normalization in :math:`\mathrm{g\,cm^{s-3}}`.
        n, s, delta : float
            Power-law indices.

        Returns
        -------
        ChevalierTwoShockState
            All fields are plain :class:`numpy.ndarray` values in CGS units.
        """
        # Ejecta normalization: K_ej in g * cm^{n-3} * s^{3-n}
        v_t, K_inner = _normalize_BPL_ejecta(E_ej=E_ej, M_ej=M_ej, n=n, delta=delta)
        K_ej = K_inner * v_t ** (n - delta)

        _lambda = (n - 3.0) / (n - s)

        # Tabulated self-similar constants
        A, xi_fs, xi_rs = self._interpolate_constants(n, s)

        # Contact discontinuity
        R_0 = (A * K_ej / K_csm) ** (1.0 / (n - s))
        R_cd = R_0 * time**_lambda
        v_cd = _lambda * R_cd / time

        # Forward and reverse shock surfaces
        R_fs = xi_fs * R_cd
        R_rs = xi_rs * R_cd
        v_fs = _lambda * R_fs / time  # lab-frame forward-shock velocity
        v_rs = _lambda * R_rs / time  # lab-frame reverse-shock velocity

        # ---- Forward shock: into stationary CSM ----
        # RH is evaluated in the shock frame; CSM is at rest so the
        # shock-frame upstream speed equals the lab-frame shock speed.
        rho_csm_fs = K_csm * R_fs ** (-s)
        rh_fs = StrongColdShockConditions._solve(v_fs, rho_csm_fs, gamma=self._gamma, mu=self._mu)

        # ---- Reverse shock: into homologously expanding ejecta ----
        # Ejecta velocity at R_rs (lab frame): R_rs / t.
        # Velocity of ejecta relative to the reverse-shock surface:
        #   v_rel = R_rs/t - lambda * R_rs/t = (1 - lambda) * R_rs / t.
        # We pass this as the "shock velocity" with upstream at rest so
        # that the RH module computes the correct shock-frame jump.
        v_rel_rs = (1.0 - _lambda) * R_rs / time
        rho_ej_rs = K_ej * time ** (n - 3.0) * R_rs ** (-n)
        rh_rs = StrongColdShockConditions._solve(v_rel_rs, rho_ej_rs, gamma=self._gamma, mu=self._mu)

        return ChevalierTwoShockState(
            radius_cd=R_cd,
            radius_fs=R_fs,
            radius_rs=R_rs,
            velocity_cd=v_cd,
            velocity_fs=v_fs,
            velocity_rs=v_rs,
            thermal_energy_density_fs=rh_fs["post_shock_thermal_energy_density"],
            thermal_energy_density_rs=rh_rs["post_shock_thermal_energy_density"],
            pressure_fs=rh_fs["post_shock_pressure"],
            pressure_rs=rh_rs["post_shock_pressure"],
            temperature_fs=rh_fs["post_shock_temperature"],
            temperature_rs=rh_rs["post_shock_temperature"],
            density_fs=rh_fs["post_shock_density"],
            density_rs=rh_rs["post_shock_density"],
        )


class ChevalierTwoShockSelfSimilarWindEngine(ChevalierTwoShockSelfSimilarEngine):
    r"""
    Two-shock Chevalier engine specialized for a steady-wind CSM (:math:`s = 2`).

    Converts the mass-loss rate :math:`\dot{M}` and wind velocity :math:`v_w` to the
    CSM normalization constant

    .. math::

        K_{\rm CSM} = \frac{\dot{M}}{4\pi v_w},

    then delegates to :class:`ChevalierTwoShockSelfSimilarEngine` with ``s = 2``.

    See Also
    --------
    ChevalierTwoShockSelfSimilarEngine : General power-law CSM version.
    ChevalierSelfSimilarWindShockEngine : Single-surface wind engine.
    """

    def compute_shock_properties(
        self,
        time: "_UnitBearingArrayLike",
        E_ej: "_UnitBearingScalarLike" = 1e51 * u.erg,
        M_ej: "_UnitBearingScalarLike" = 10 * u.Msun,
        M_dot: "_UnitBearingScalarLike" = 1e-5 * u.Msun / u.yr,
        v_wind: "_UnitBearingScalarLike" = 1000 * u.km / u.s,
        n: float = 10.0,
        delta: float = 0.0,
    ) -> "ChevalierTwoShockState":
        """
        Compute the two-shock structure for a wind-like CSM.

        Parameters
        ----------
        time : ~astropy.units.Quantity or array-like
            Time since explosion. Bare float assumed to be in seconds.
        E_ej : ~astropy.units.Quantity or float
            Ejecta kinetic energy. Bare float assumed to be in erg.
        M_ej : ~astropy.units.Quantity or float
            Ejecta mass. Bare float assumed to be in grams.
        M_dot : ~astropy.units.Quantity or float
            Wind mass-loss rate. Bare float assumed to be in g/s.
        v_wind : ~astropy.units.Quantity or float
            Wind velocity. Bare float assumed to be in cm/s.
        n : float, optional
            Outer ejecta power-law index.
        delta : float, optional
            Inner ejecta power-law index. Default is ``0.0``.

        Returns
        -------
        ChevalierTwoShockState
        """
        if isinstance(time, u.Quantity):
            time = time.to(u.s).value
        if isinstance(E_ej, u.Quantity):
            E_ej = E_ej.to(u.erg).value
        if isinstance(M_ej, u.Quantity):
            M_ej = M_ej.to(u.g).value
        if isinstance(M_dot, u.Quantity):
            M_dot = M_dot.to(u.g / u.s).value
        if isinstance(v_wind, u.Quantity):
            v_wind = v_wind.to(u.cm / u.s).value

        if delta >= 3:
            raise ValueError("The inner ejecta density profile index `delta` must be less than 3 for convergence.")
        if n <= 5:
            raise ValueError("The outer ejecta density profile index `n` must be greater than 5 for convergence.")

        cgs = self._compute_shock_properties_cgs(
            time=time, E_ej=E_ej, M_ej=M_ej, M_dot=M_dot, v_wind=v_wind, n=n, delta=delta
        )
        return self._attach_units(cgs)

    def _compute_shock_properties_cgs(
        self,
        time: "_ArrayLike",
        E_ej: float = 1e51,
        M_ej: float = 1e34,
        M_dot: float = 6.3e20,
        v_wind: float = 1e8,
        n: float = 10.0,
        delta: float = 0.0,
    ) -> "ChevalierTwoShockState":
        """CGS backend: convert wind parameters to K_csm then call the parent solver."""
        K_csm = M_dot / (4.0 * np.pi * v_wind)
        return super()._compute_shock_properties_cgs(
            time=time, E_ej=E_ej, M_ej=M_ej, K_csm=K_csm, n=n, s=2.0, delta=delta
        )
