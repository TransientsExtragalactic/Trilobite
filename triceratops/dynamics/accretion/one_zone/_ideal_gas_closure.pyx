#cython: language_level=3, boundscheck=False
r"""
Ideal gas closures for one-zone accretion disk models (Cython).

Provides :class:`GasPressureElectronScatteringClosure`, which wires three
C-level functions (closure, derivative, writer) into a
:class:`~._integrator.OneZoneClosure` object suitable for
:func:`~._integrator.run_one_zone_model`.

The closure assumes:

- **Pressure support**: gas pressure only (:math:`P = \rho k_B T_c / \mu m_p`).
- **Opacity**: electron-scattering dominated (:math:`\kappa = \kappa_{\rm es}`).

Under these assumptions the midplane temperature can be solved analytically
at each step (no iteration needed), making this the fastest available closure.
"""
from libc.math cimport exp, log, pi, sqrt

from triceratops.math_utils._bracket_root_finder cimport find_root

from ._integrator cimport (DISK_F0, K_B_CGS, LOG_G_CGS, LOG_K_B_CGS,
                           LOG_KAPPA_ES, LOG_M_P_CGS, LOG_SIGMA_SB_CGS,
                           M_P_CGS, RAD_A_CGS, ClosureResult, DiskDerived,
                           DiskParameters, DiskState, DiskStep, OneZoneClosure)

# ==================================================== #
# CONSTANTS                                            #
# ==================================================== #
DEF SQRT_2PI = 2.506628274631000

# ==================================================== #
# Utilities                                            #
# ==================================================== #
cdef inline double solve_gas_plus_radiation_eos(
    double T,
    double mu,
    double Sigma,
    double Omega
) nogil:
    """
    Solve for the isothermal sound speed c_s in a gas+radiation pressure equation of state.

    Because

        rho = Sigma * Omega / (sqrt(2*pi) * c_s)

    and

        c_s^2 = P / rho
              = k_B T / (mu m_p) + (a T^4 / 3) / rho

    which gives the quadratic

        c_s^2 - A c_s - B = 0

    with
        A = a T^4 sqrt(2*pi) / (3 Sigma Omega)
        B = k_B T / (mu m_p)

    Returns the positive physical root using a numerically stable form. This is used inside
    the gas+radiation closure to compute the sound speed when both pressure components are included.
    """
    # Declare the relevant constants in cdef scope for performance.
    cdef double A, B, D

    # Compute the A and B parameters.
    B = K_B_CGS * T / (mu * M_P_CGS)
    A = RAD_A_CGS * T * T * T * T * SQRT_2PI / (3.0 * Sigma * Omega)

    # Compute the positive root of c_s^2 - A c_s - B = 0.
    # The quadratic formula gives c_s = (A + sqrt(A^2 + 4B)) / 2.
    # Since A >= 0 and B > 0, both A and D = sqrt(A^2 + 4B) are positive and
    # we are ADDING them — no catastrophic cancellation occurs in any regime.
    # An alternative form (2B)/(D - A) avoids large-number addition but suffers
    # from D - A → 0 underflow when A is very large (radiation-dominated limit),
    # leading to division by zero. The direct form below is always safe.
    D = sqrt(A * A + 4.0 * B)
    return 0.5 * (A + D)


# ==================================================== #
# Gas Pressure + Electron Scattering Closure           #
# ==================================================== #

# Number of output fields written by gas_pressure_writer_func.
# Fields: index | t | M | J | R | Sigma | Omega | T_eff | T_c | tau |
#         cs | nu | q_visc | dM_dt | dJ_dt | dt | t_visc | H | H/R | rho
DEF GAS_PRESSURE_ES_N_RESULT_FIELDS = 20
# Python-visible constant so _source_terms.pyx can read it without a second cimport.
GAS_PRESSURE_ES_N_RESULT_FIELDS_PY = GAS_PRESSURE_ES_N_RESULT_FIELDS


cdef int gas_pressure_closure_func(
        const DiskState* state,
        const DiskDerived* derived,
        const DiskParameters* params,
        const ClosureResult* prev,
        ClosureResult* out,
) nogil:
    """
    Compute thermodynamic and viscous quantities for the gas-pressure /
    electron-scattering closure.

    The midplane temperature follows from vertical energy balance:

    .. math::

        T_c^3 = \\frac{27}{64}\\,\\frac{k_B}{m_p\\,\\mu}\\,
                \\alpha\\,\\kappa_{\\rm es}\\,\\Omega\\,\\Sigma^2.

    All remaining quantities (cs, nu, q_visc, tau, T_eff, t_visc) are then
    derived analytically with no iteration.
    """
    # log[(27/64) * k_B / m_p] — leading coefficient for T_c^3
    cdef double log_TC_coef = log(27.0 / 64.0) + LOG_K_B_CGS - LOG_M_P_CGS

    # ---- Midplane temperature ------------------------------------------- #
    # T_c^3 = [(27/64) k_B / m_p] mu^{-1} alpha kappa_es Omega Sigma^2
    cdef double log_T = (1.0 / 3.0) * (
        log_TC_coef
        - log(params.mu)
        + log(params.alpha)
        + LOG_KAPPA_ES
        + log(derived.Omega)
        + 2.0 * log(derived.Sigma)
    )
    out.log_T_c = log_T

    # ---- Optical depth ---------------------------------------------------- #
    # tau = kappa_es * Sigma / 2
    out.log_tau = log(0.5) + LOG_KAPPA_ES + log(derived.Sigma)

    # ---- Effective temperature -------------------------------------------- #
    # T_c^4 = (3/8) kappa_es Sigma T_eff^4  =>  T_eff = T_c (8 / (3 kappa Sigma))^(1/4)
    out.log_T_eff = log_T + 0.25 * (
        log(8.0 / 3.0) - LOG_KAPPA_ES - log(derived.Sigma)
    )

    # ---- Sound speed ------------------------------------------------------ #
    # cs^2 = k_B T_c / (mu m_p)
    out.log_cs = 0.5 * (log_T + LOG_K_B_CGS - LOG_M_P_CGS - log(params.mu))

    # ---- Kinematic viscosity --------------------------------------------- #
    # nu = alpha cs^2 / Omega
    out.log_nu = log(params.alpha) + 2.0 * out.log_cs - log(derived.Omega)

    # ---- Viscous heating rate -------------------------------------------- #
    # q_visc = (9/8) nu Sigma Omega^2
    out.log_q_visc = (
        log(9.0 / 8.0) + out.log_nu + log(derived.Sigma) + 2.0 * log(derived.Omega)
    )

    # ---- Viscous timescale ----------------------------------------------- #
    # t_visc = R^2 / nu
    out.t_visc = exp(2.0 * log(derived.R) - out.log_nu)

    return 0


cdef int gas_pressure_derivative_func(
    const DiskState* state,
    const DiskDerived* derived,
    const DiskParameters* params,
    const ClosureResult* closure,
    DiskStep* out,
) nogil:
    """
    Compute the explicit-Euler time step for the gas-pressure closure.

    Mass and angular momentum drain through the inner boundary only; the
    boundary correction factor is

    .. math::

        f_{\\rm corr} = \\frac{F_0}{1 - \\sqrt{R_{\\rm in}/R_D}}.
    """
    cdef double time_step_factor = 1e-3

    # Inner-boundary correction to the mass-drain rate.
    cdef double f_corr = DISK_F0 / (1.0 - sqrt(params.R_in / derived.R))

    # dM/dt: mass drains through the inner boundary (negative).
    out.dM_dt = -exp(log(state.M) - log(closure.t_visc) + log(f_corr))

    # dJ/dt: angular momentum carried away at the ISCO.
    out.dJ_dt = out.dM_dt * exp(0.5 * (LOG_G_CGS + log(params.MBH) + log(params.R_in)))

    # Adaptive time step: a fixed fraction of the viscous timescale.
    out.dt = time_step_factor * closure.t_visc

    return 0


cdef int gas_pressure_writer_func(
    const int step_index,
    const DiskState* state,
    const DiskDerived* derived,
    const DiskParameters* params,
    const ClosureResult* closure,
    const DiskStep* step,
    double* result_array,
    int n_steps,
) nogil:
    """
    Write one step of output into ``result_array``.

    ``result_array`` points to ``result[0, 0]`` of the pre-allocated
    ``(GAS_PRESSURE_ES_N_RESULT_FIELDS, n_steps)`` array.  Element
    ``[field, step_index]`` lives at offset ``field * n_steps + step_index``.

    Field layout (GAS_PRESSURE_ES_N_RESULT_FIELDS = 20)
    -----------------------------------------------------
    0  step_index        (dimensionless)
    1  t                 s
    2  M                 g
    3  J                 g cm² s⁻¹
    4  R                 cm
    5  Sigma             g cm⁻²
    6  Omega             s⁻¹
    7  T_eff             K
    8  T_c               K
    9  tau               (dimensionless)
    10 cs                cm s⁻¹
    11 nu                cm² s⁻¹
    12 q_visc            erg cm⁻² s⁻¹
    13 dM_dt             g s⁻¹
    14 dJ_dt             g cm² s⁻²
    15 dt                s
    16 t_visc            s
    17 H                 cm
    18 H/R               (dimensionless)
    19 rho               g cm⁻³
    """
    cdef double H = exp(closure.log_cs - log(derived.Omega))
    cdef double rho = derived.Sigma / (sqrt(2.0 * pi) * H)
    cdef double H_over_R = H / derived.R

    result_array[ 0 * n_steps + step_index] = step_index
    result_array[ 1 * n_steps + step_index] = state.t
    result_array[ 2 * n_steps + step_index] = state.M
    result_array[ 3 * n_steps + step_index] = state.J
    result_array[ 4 * n_steps + step_index] = derived.R
    result_array[ 5 * n_steps + step_index] = derived.Sigma
    result_array[ 6 * n_steps + step_index] = derived.Omega
    result_array[ 7 * n_steps + step_index] = exp(closure.log_T_eff)
    result_array[ 8 * n_steps + step_index] = exp(closure.log_T_c)
    result_array[ 9 * n_steps + step_index] = exp(closure.log_tau)
    result_array[10 * n_steps + step_index] = exp(closure.log_cs)
    result_array[11 * n_steps + step_index] = exp(closure.log_nu)
    result_array[12 * n_steps + step_index] = exp(closure.log_q_visc)
    result_array[13 * n_steps + step_index] = step.dM_dt
    result_array[14 * n_steps + step_index] = step.dJ_dt
    result_array[15 * n_steps + step_index] = step.dt
    result_array[16 * n_steps + step_index] = closure.t_visc
    result_array[17 * n_steps + step_index] = H
    result_array[18 * n_steps + step_index] = H_over_R
    result_array[19 * n_steps + step_index] = rho
    return 0

cdef class GasPressureElectronScatteringClosure(OneZoneClosure):
    """
    Closure with only gas pressure and electron scattering opacity.

    This is the simplest physically motivated closure and is the default for
    :func:`run_one_zone_model`.  All three function pointers are
    wired in ``__cinit__``; the object is immediately
    :meth:`~._integrator.OneZoneClosure.is_ready` after construction.

    """

    def __cinit__(self):
        self._closure_fn     = gas_pressure_closure_func
        self._derivative_fn  = gas_pressure_derivative_func
        self._writer_fn      = gas_pressure_writer_func
        self.n_result_fields = GAS_PRESSURE_ES_N_RESULT_FIELDS

# ==================================================== #
# Full Pressure (Gas + Radiation) + ES Closure         #
# ==================================================== #

# Number of result fields — identical layout to GasPressureElectronScatteringClosure.
DEF FULL_PRESSURE_ES_N_RESULT_FIELDS = 20
# Python-visible constant so _source_terms.pyx can read it without a second cimport.
FULL_PRESSURE_ES_N_RESULT_FIELDS_PY = FULL_PRESSURE_ES_N_RESULT_FIELDS


# Context forwarded to the energy-balance root-finding callback.
# Holds the constant part of the q+ = q- log-space residual and
# the EOS parameters needed to evaluate c_s(T_c).
cdef struct FullPressureRootData:
    double log_Q   # log((27/64) alpha kappa_es Omega Sigma^2 / sigma_SB)
    double mu      # mean molecular weight (dimensionless)
    double Sigma   # surface density (g cm^-2)
    double Omega   # Keplerian angular velocity (s^-1)


cdef int full_pressure_residual(
    double log_T,
    void* user_data,
    double* f_out,
) nogil:
    r"""Log-space residual of the :math:`q^+ = q^-` energy balance.

    Evaluates

    .. math::

        f(\log T_c) = \log Q + 2\log c_s(T_c) - 4\log T_c

    where :math:`c_s(T_c)` is the isothermal sound speed from the combined
    gas-plus-radiation EOS.  The root gives the midplane temperature
    satisfying vertical energy balance.

    Parameters
    ----------
    log_T : double
        Natural logarithm of the trial midplane temperature (K).
    user_data : void*
        Pointer to a :c:type:`FullPressureRootData` struct.
    f_out : double*
        On return, holds the residual value.

    Returns
    -------
    int
        Always 0 (SUCCESS).
    """
    cdef FullPressureRootData* d = <FullPressureRootData*>user_data
    cdef double c_s = solve_gas_plus_radiation_eos(exp(log_T), d.mu, d.Sigma, d.Omega)
    f_out[0] = d.log_Q + 2.0 * log(c_s) - 4.0 * log_T
    return 0


cdef int full_pressure_closure_func(
    const DiskState* state,
    const DiskDerived* derived,
    const DiskParameters* params,
    const ClosureResult* prev,
    ClosureResult* out,
) nogil:
    r"""Compute thermodynamic and viscous quantities for the full-pressure /
    electron-scattering closure.

    The midplane temperature :math:`T_c` is found by solving the vertical
    energy balance :math:`q^+(T_c) = q^-(T_c)`:

    .. math::

        \frac{9}{8}\,\alpha\,c_s^2\,\Sigma\,\Omega
        \;=\;
        \sigma_{\rm SB}\,\frac{8\,T_c^4}{3\,\kappa_{\rm es}\,\Sigma}.

    Rearranging gives the log-space residual

    .. math::

        f(\log T_c)
        = \log Q + 2\log c_s(T_c) - 4\log T_c
        = 0,
        \qquad
        Q = \frac{27}{64}\,\alpha\,\kappa_{\rm es}\,\Omega\,\Sigma^2,

    where :math:`c_s(T_c)` is the isothermal sound speed from the combined
    gas-plus-radiation equation of state
    (see :func:`solve_gas_plus_radiation_eos`).  The root is located with
    :func:`~triceratops.math_utils._bracket_root_finder.find_root` (geometric
    bracket expansion followed by Brent's method).

    A warm start from :attr:`~._integrator.ClosureResult.log_T_c` of the
    previous step is used when available; otherwise the analytical gas-only
    estimate serves as the initial guess.

    All downstream quantities (:math:`c_s`, :math:`\nu`, :math:`\tau`,
    :math:`T_{\rm eff}`, :math:`q_{\rm visc}`, :math:`t_{\rm visc}`) are
    computed from the converged :math:`T_c` using the full sound speed.

    Parameters
    ----------
    state : DiskState*
        Current disk state (:math:`t`, :math:`M`, :math:`J`).
    derived : DiskDerived*
        Derived quantities (:math:`R`, :math:`\Sigma`, :math:`\Omega`).
    params : DiskParameters*
        Fixed model parameters (:math:`M_{\rm BH}`, :math:`R_{\rm in}`,
        :math:`\alpha`, :math:`\mu`).
    prev : ClosureResult*
        Closure result from the previous step; ``log_T_c`` is used as a
        warm-start initial guess (sentinel 0.0 triggers fall-back).
    out : ClosureResult*
        On SUCCESS, all fields are filled.

    Returns
    -------
    int
        0 on SUCCESS; propagates the :func:`find_root` status code on
        failure (1 FUNC_ERROR, 2 EXPAND_FAIL, 4 MAX_ITER).
    """
    cdef double log_Q, log_T_guess, log_T, c_s
    cdef FullPressureRootData root_data
    cdef int status

    # ---- Constant part of the q+ = q- residual --------------------------- #
    # log Q = log(27/64) + log(alpha) + LOG_KAPPA_ES + 2 log(Sigma) + log(Omega) - LOG_SIGMA_SB
    # This is the physically-correct energy-balance constant (Shakura-Sunyaev
    # vertical structure with electron-scattering opacity).  The gas-pressure-
    # only closure omits sigma_SB (legacy convention), so the two closures
    # produce different T_c values in the gas-dominated limit; T_c_full is
    # always larger than T_c_gas_only.
    log_Q = (
        log(27.0 / 64.0)
        + log(params.alpha)
        + LOG_KAPPA_ES
        + 2.0 * log(derived.Sigma)
        + log(derived.Omega)
        - LOG_SIGMA_SB_CGS
    )

    # ---- Initial guess: warm-start or gas-only analytical estimate -------- #
    # When prev.log_T_c is non-zero (all steps after the first), use it
    # directly as the initial guess for the root finder — the temperature
    # changes smoothly between timesteps so the previous value is nearly
    # always inside the final bracket.
    if prev.log_T_c != 0.0:
        log_T_guess = prev.log_T_c
    else:
        # Gas-only analytical estimate: T_c^3 = (27/64)(k_B/m_p)/mu * alpha *
        # kappa * Omega * Sigma^2.  This is exact in the pure-gas limit and a
        # reasonable starting point in the radiation-dominated regime.
        log_T_guess = (1.0 / 3.0) * (
              log(27.0 / 64.0)
            + LOG_K_B_CGS
            + log(params.alpha)
            + LOG_KAPPA_ES
            - LOG_M_P_CGS
            - log(params.mu)
            - LOG_SIGMA_SB_CGS
            + log(derived.Omega)
            + 2.0 * log(derived.Sigma)
        )

    # ---- Root find: q+(T_c) = q-(T_c) in log(T_c) space ----------------- #
    root_data.log_Q = log_Q
    root_data.mu    = params.mu
    root_data.Sigma = derived.Sigma
    root_data.Omega = derived.Omega

    status = find_root(
        full_pressure_residual, &root_data,
        log_T_guess,
        0.5,   # step: initial half-width of bracket in log-T space (~factor 1.65)
        2.0,   # grow_factor: geometric expansion rate
        60,    # max_expand: up to 60 expansions covers the full physical domain
        log(1.0e1), log(1.0e14),  # domain: 10 K to 1e14 K
        1.0e-30,                  # tolerance: exact-root detection threshold
        1.0e-10, 1.0e-12,         # tol_x, tol_f: convergence thresholds
        100,                      # maxiter: Brent iterations
        &log_T,
    )
    if status != 0:
        return status

    out.log_T_c = log_T

    # ---- Sound speed from full (gas + radiation) EOS --------------------- #
    # Re-evaluate at the converged T_c.  solve_gas_plus_radiation_eos was last
    # called inside the root finder at log_T; evaluating it once more avoids
    # storing c_s in the root-data struct.
    c_s = solve_gas_plus_radiation_eos(exp(log_T), params.mu, derived.Sigma, derived.Omega)
    out.log_cs = log(c_s)

    # ---- Optical depth --------------------------------------------------- #
    # tau = kappa_es * Sigma / 2
    out.log_tau = log(0.5) + LOG_KAPPA_ES + log(derived.Sigma)

    # ---- Effective temperature ------------------------------------------- #
    # T_c^4 = (3/8) kappa_es Sigma T_eff^4
    out.log_T_eff = log_T + 0.25 * (log(8.0 / 3.0) - LOG_KAPPA_ES - log(derived.Sigma))

    # ---- Kinematic viscosity -------------------------------------------- #
    # nu = alpha c_s^2 / Omega
    out.log_nu = log(params.alpha) + 2.0 * out.log_cs - log(derived.Omega)

    # ---- Viscous heating rate ------------------------------------------- #
    # q_visc = (9/8) nu Sigma Omega^2
    out.log_q_visc = (
        log(9.0 / 8.0) + out.log_nu + log(derived.Sigma) + 2.0 * log(derived.Omega)
    )

    # ---- Viscous timescale ---------------------------------------------- #
    # t_visc = R^2 / nu
    out.t_visc = exp(2.0 * log(derived.R) - out.log_nu)

    return 0


cdef class FullPressureElectronScatteringClosure(OneZoneClosure):
    """Full-pressure (gas + radiation) closure with electron-scattering opacity.

    Extends :class:`GasPressureElectronScatteringClosure` by including radiation
    pressure in the equation of state.  The midplane temperature is solved
    iteratively at each step via :func:`~triceratops.math_utils._bracket_root_finder.find_root`
    (bracket expansion + Brent's method) rather than analytically, making this
    closure accurate in the radiation-dominated regime where the gas-only formula
    fails.

    The derivative and writer functions are identical to those of
    :class:`GasPressureElectronScatteringClosure`; only the closure function
    (temperature solve) differs.  The output field layout is therefore also
    identical (``FULL_PRESSURE_ES_N_RESULT_FIELDS = 20``).
    """

    def __cinit__(self):
        self._closure_fn     = full_pressure_closure_func
        self._derivative_fn  = gas_pressure_derivative_func   # same viscous physics
        self._writer_fn      = gas_pressure_writer_func       # same output layout
        self.n_result_fields = FULL_PRESSURE_ES_N_RESULT_FIELDS
