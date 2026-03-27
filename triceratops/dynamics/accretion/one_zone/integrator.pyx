#cython: language_level=3, boundscheck=False
r"""
[C-LEVEL] One-zone accretion disk integrator.

Implements an explicit Euler time-marching scheme for the coupled
:math:`(M_D, J_D)` ODE system described in
:mod:`dynamics.accretion.one_zone.base`.  The hot loop is
compiled to C and released from the GIL, making it safe to call from
threaded contexts.

The public entry point is :func:`run_one_zone_model`; concrete
:class:`~.closure.OneZoneClosure` subclasses supply the physics via C
function pointers with zero Python-layer overhead.

See Also
--------
:mod:`dynamics.accretion.one_zone.closure` :
    Abstract closure base: constants, structs, typedefs, ``OneZoneClosure``.
:mod:`dynamics.accretion.one_zone._eos` :
    Equations of state (closure, derivative, writer functions).
:mod:`dynamics.accretion.one_zone._sources` :
    Optional source-term functions.
:mod:`dynamics.accretion.one_zone.base` :
    Python reference implementation and closure pipeline semantics.
"""
import cython
cimport cython

from libc.math cimport exp, log
from cpython.ref cimport PyObject
from triceratops.radiation.opacity.opacity_base cimport C_GreyOpacityBase

from .closure cimport (
    LOG_DISK_XI, LOG_DISK_A, LOG_G_CGS, LOG_PI,
    ClosureResult, DiskDerived, DiskParameters, DiskState, DiskStep,
    OneZoneClosure,
    closure_func, derivative_func, writer_func, source_func,
)


# ---------------------------------------- #
# Explicit Marcher                         #
# ---------------------------------------- #

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int compute_one_zone_model(
        double[::1] initial_state,
        double[::1] parameters,
        double[:, ::1] result_array,
        double t_start,
        double t_end,
        int max_steps,
        double epsilon,
        closure_func closure_function,
        derivative_func derivative_function,
        writer_func writer_function,
        source_func source_function,
        void* opacity,
):
    """
    Compute the one-zone disk evolution using an explicit Euler scheme.

    Parameters
    ----------
    initial_state: double
        Initial disk state: ``[M_D (g), J_D (g cm^2 s^-1)]``. This acts as the initial conditions
        for the simulation.
    parameters: double
        Model parameters: ``[M_BH (g), R_in (cm), alpha, mu, ...]``. The first four elements are interpreted as
         the black hole mass, inner disk radius, viscosity parameter, and mean molecular weight, respectively.
         Elements beyond index 3 are passed to the closure as ``params.extra``; their interpretation is
         closure-specific.

         This is where implementations can add additional runtime parameters.

    result_array: double
        The array into which the results will be written.  It is pre-allocated by the caller as
        ``(n_result_fields, max_steps)`` and is field-major: field ``f`` at step ``i`` lives at offset
        ``f * max_steps + i``.  The integrator writes exactly one column per step, starting with column 0 for the
        initial state.

        The ``writer_function`` determines how data is packed into the array.

    t_start: double
        Start time of the integration in seconds.
    t_end: double
        End time of the integration in seconds.  The integrator will terminate early if this time is reached.
    max_steps:
        Maximum number of integration steps.  The integrator will terminate if this number of steps is taken, even if
        t_end is not reached.
    epsilon:
        Adaptive time-step fraction. For a relevant update timescale dt, we actually step epsilon * dt. The literature
        (Piro & Mockler 2025) suggest epsilon < 1e-3.
    closure_function:
        C function pointer for the thermodynamic closure.  This is a required argument; the integrator does not
        support a default closure.
    derivative_function:
        C function pointer for the ODE derivative.  This is a required argument; the integrator does not support a
        default derivative.
    writer_function:
        C function pointer for writing output.  This is a required argument; the integrator does not support a default
        writer.
    source_function:
        Optional C function pointer for source terms.  If NULL, no source term is applied.  If non-NULL, this function
        is called after the closure and before the derivative at each step, and can modify the derivative state to
        add source terms to the ODEs.
    opacity: C_GreyOpacityBase
        An opacity object pointer, threaded through to the closure and source functions.  This is passed as a void* to
        avoid Cython's type-checking overhead on the closure and source function pointers; the
        relevant opacity methods must be called from the C hot path in the closure and source functions,
        not from Python code.

    Returns
    -------
    int
        The number of steps actually taken (columns written to the result array).  This may be less than max_steps if
        t_end is reached first.
    """
    # --- Declare variables --- #
    cdef double log_m, log_j, log_MBH, log_r, log_sigma, log_omega
    cdef int actual_steps = 0
    cdef int fn_status
    cdef DiskDerived derived_disk_params
    cdef ClosureResult closure_result
    cdef ClosureResult prev_closure

    cdef DiskState state
    cdef DiskParameters params
    cdef DiskStep step

    # Populate the state.
    state.t = t_start
    state.M = initial_state[0]
    state.J = initial_state[1]

    # Populate the parameters.
    params.MBH     = parameters[0]
    params.R_in    = parameters[1]
    params.alpha   = parameters[2]
    params.mu      = parameters[3]
    params.epsilon = epsilon

    # Extra parameters (indices 4+) — zero-copy pointer into the buffer.
    params.n_extra = parameters.shape[0] - 4
    if params.n_extra > 0:
        params.extra = &parameters[4] # Start extra at index 4.
    else:
        params.extra = NULL

    # Thread the opacity object pointer into DiskParameters.
    params.opacity = opacity

    # Zero-initialise prev_closure (warm-start seed for step 0).
    # t_visc == 0 is the sentinel the closure uses to fall back to state.t.
    prev_closure.dM_dt      = 0.0
    prev_closure.dJ_dt      = 0.0
    prev_closure.t_visc     = 0.0
    prev_closure.log_T_eff  = 0.0
    prev_closure.log_T_c    = 0.0
    prev_closure.log_tau    = 0.0
    prev_closure.log_cs     = 0.0
    prev_closure.log_nu     = 0.0
    prev_closure.log_q_visc = 0.0
    prev_closure.log_rho    = 0.0

    # --- Hot loop (released from GIL) --- #
    # Column `actual_steps` is written at the START of each iteration (before the
    # state update), so column 0 holds the initial condition with its full
    # thermodynamic closure, and column k holds the state after k Euler steps.
    with nogil:
        while actual_steps < max_steps:
            # Canonical step order:
            # (1) compute log quantities → (2) pack DiskDerived → (3) call closure
            # → (4) call derivative → (5) write row → (6) update state
            # → (7) copy prev_closure → (8) increment actual_steps → (9) check t_end.

            # (1) Compute log quantities.
            log_m   = log(state.M)
            log_j   = log(state.J)
            log_MBH = log(params.MBH)

            # R_D = J^2 / (xi^2 * M^2 * G * M_BH)
            log_r     = 2.0 * (log_j - LOG_DISK_XI - log_m) - LOG_G_CGS - log_MBH
            # Sigma = M / (pi * A * R^2)
            log_sigma = log_m - 2.0 * log_r - LOG_PI - LOG_DISK_A
            # Omega = sqrt(G * M_BH / R^3)
            log_omega = 0.5 * (log_MBH + LOG_G_CGS - 3.0 * log_r)

            # (2) Pack DiskDerived.
            derived_disk_params.R     = exp(log_r)
            derived_disk_params.Sigma = exp(log_sigma)
            derived_disk_params.Omega = exp(log_omega)

            # (3) Call closure.
            fn_status = closure_function(&state, &derived_disk_params, &params,
                                        &prev_closure, &closure_result)
            if fn_status != 0:
                return -fn_status

            # (4b) Apply optional source term.
            if source_function != NULL:
                fn_status = source_function(&state, &derived_disk_params, &params,
                                            &closure_result, &step)
                if fn_status != 0:
                    return -30

            # (4) Call derivative.
            fn_status = derivative_function(&state, &derived_disk_params, &params,
                                            &closure_result, &step)
            if fn_status != 0:
                return -10

            # (5) Write column actual_steps.
            fn_status = writer_function(actual_steps, &state, &derived_disk_params, &params,
                                        &closure_result, &step, &result_array[0, 0], max_steps)
            if fn_status != 0:
                return -20

            # (6) Update state.
            state.t += step.dt
            state.M += step.dM_dt * step.dt
            state.J += step.dJ_dt * step.dt

            # Clear the current derivative state to prevent
            # accumulations in the source and derivative functions from affecting the next step.
            step.dM_dt = 0.0
            step.dJ_dt = 0.0
            step.dt = 0.0

            # (7) Copy prev_closure for next iteration.
            prev_closure = closure_result

            # (8) Increment counter.
            actual_steps += 1

            # (9) Terminate early if t_end reached.
            if state.t >= t_end:
                break

    return actual_steps


# ---------------------------------------- #
# Python Wrapper                           #
# ---------------------------------------- #
def run_one_zone_model(
    double[::1] initial_state not None,
    double[::1] parameters not None,
    double t_start,
    double t_end,
    int max_steps,
    OneZoneClosure closure not None,
    double epsilon = 1e-6,
):
    """
    Run the one-zone disk integrator; return the populated result array.

    Parameters
    ----------
    initial_state : ndarray, shape (2,)
        ``[M_D (g), J_D (g cm^2 s^-1)]``.
    parameters : ndarray, shape (>=4,)
        ``[M_BH (g), R_in (cm), alpha, mu, ...]``.  Elements beyond index 3
        are passed to the closure as ``params.extra``; their interpretation
        is closure-specific.
    t_start, t_end : float
        Integration interval in seconds.
    max_steps : int
        Maximum number of integration steps.
    closure : OneZoneClosure
        Closure with all three mandatory function pointers installed.
    epsilon : float, optional
        Adaptive time-step fraction: :math:`\\Delta t = \\epsilon\\,\\min(t_{\\rm visc},
        |M/\\dot{M}|, |J/\\dot{J}|)`.  Smaller values give finer time resolution at
        the cost of more steps.  Default is ``1e-6``.

    Returns
    -------
    ndarray, shape (n_result_fields, actual_steps)
        Column ``k`` holds the disk state and all derived quantities after
        ``k`` explicit-Euler steps; column 0 is the initial condition with
        its full thermodynamic closure computed.  Row ``i`` holds field ``i``
        across all steps.

    Raises
    ------
    ValueError
        Bad array shapes, ``t_end <= t_start``, or uninitialised closure.
    RuntimeError
        Integrator returned a negative status code.  The message identifies the
        failure mode; possible codes and their meanings are:

        * ``-1`` **CLOSURE / FUNC_ERROR** — the root-finding residual callback
          returned non-zero (NaN or Inf encountered in state or residual).
        * ``-2`` **CLOSURE / EXPAND_FAIL** — bracket-expansion could not find a
          sign change in ``[T_min, T_max]``.  The disk parameters may have driven
          the surface density or Ω outside the physical range of this closure.
        * ``-3`` **CLOSURE / NO_BRACKET** — the initial bracket had the same sign
          at both endpoints after expansion.
        * ``-4`` **CLOSURE / MAX_ITER** — Brent's method did not converge within
          the iteration limit.
        * ``-10`` **DERIVATIVE_FAIL** — the derivative function returned non-zero.
        * ``-20`` **WRITER_FAIL** — the writer function returned non-zero.
        * ``-30`` **SOURCE_FAIL** — the source function returned non-zero.
    """
    cdef double[:, ::1] result
    cdef int status

    import numpy as np

    if not closure.is_ready():
        raise ValueError("OneZoneClosure has uninitialised function pointers.")
    if initial_state.shape[0] != 2:
        raise ValueError(
            f"initial_state must have shape (2,), got shape ({initial_state.shape[0]},)."
        )
    if parameters.shape[0] < 4:
        raise ValueError(
            f"parameters must have shape (>=4,), got shape ({parameters.shape[0]},). "
            f"Expected at least [MBH, R_in, alpha, mu]."
        )
    if t_end <= t_start:
        raise ValueError(f"t_end ({t_end}) must be > t_start ({t_start}).")
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}.")
    if epsilon <= 0.0 or epsilon >= 1.0:
        raise ValueError(f"epsilon must be in (0, 1), got {epsilon}.")
    if closure._c_opacity is None:
        raise ValueError(
            "OneZoneClosure has no opacity installed.  "
            "Set closure.opacity before passing to the integrator."
        )

    result = np.zeros(
        (closure.n_result_fields, max_steps), dtype=np.float64
    )

    status = compute_one_zone_model(
        initial_state, parameters, result,
        t_start, t_end, max_steps, epsilon,
        closure._closure_fn,
        closure._derivative_fn,
        closure._writer_fn,
        closure._source_fn,
        <void*>(<PyObject*>closure._c_opacity),
    )

    if status < 0:
        _CLOSURE_MESSAGES = {
            -1: (
                "CLOSURE / FUNC_ERROR (-1): the root-finding residual callback returned "
                "non-zero — NaN or Inf encountered in the disk state or residual."
            ),
            -2: (
                "CLOSURE / EXPAND_FAIL (-2): bracket expansion could not find a sign "
                "change in [T_min, T_max].  The disk state (surface density, Ω) may be "
                "outside the physical range supported by this closure."
            ),
            -3: (
                "CLOSURE / NO_BRACKET (-3): both bracket endpoints have the same sign "
                "after expansion — no root detected in the search domain."
            ),
            -4: (
                "CLOSURE / MAX_ITER (-4): Brent's method did not converge within the "
                "iteration limit."
            ),
            -10: (
                "DERIVATIVE_FAIL (-10): the derivative function returned a non-zero "
                "status (internal error)."
            ),
            -20: (
                "WRITER_FAIL (-20): the writer function returned a non-zero status "
                "(internal error)."
            ),
            -30: (
                "SOURCE_FAIL (-30): the source function returned a non-zero status "
                "(internal error)."
            ),
        }
        msg = _CLOSURE_MESSAGES.get(
            status,
            f"Integrator returned unrecognised error code {status}.",
        )
        raise RuntimeError(msg)

    return np.ascontiguousarray(result[:, :status])
