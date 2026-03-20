#cython: language_level=3, boundscheck=False
r"""
One-zone accretion disk integrator (Cython).

Implements an explicit Euler time-marching scheme for the coupled
:math:`(M_D, J_D)` ODE system described in
:mod:`triceratops.dynamics.accretion.one_zone.base`.  The hot loop is
compiled to C and released from the GIL, making it safe to call from
threaded contexts.

The public entry point is :func:`run_one_zone_model`; the abstract
extension type :class:`OneZoneClosure` is provided so that
``_closure.pyx`` can supply concrete C function pointers without any
Python-layer overhead.

See Also
--------
:mod:`triceratops.dynamics.accretion.one_zone.base` :
    Python reference implementation and closure pipeline semantics.
:mod:`triceratops.dynamics.accretion.utils` :
    Log-space disk arithmetic utilities.
"""
import cython

cimport cython
from libc.math cimport exp, log, pi
from libc.stdio cimport printf


# ---------------------------------------- #
# Constants                                #
# ---------------------------------------- #
# Physical constants (defined here, declared in .pxd for cimport by other modules).
cdef double DISK_A   = 1.62                # Metzger+08 disk correction factor (dimensionless)
cdef double DISK_B   = 1.33                # Metzger+08 disk correction factor (dimensionless)
cdef double DISK_F0  = 1.6                 # Metzger+08 disk correction factor (dimensionless)
cdef double DISK_XI  = DISK_B / DISK_A    # Metzger+08 disk correction factor (dimensionless)
cdef double G_CGS    = 6.67430e-8          # Gravitational constant (cm^3 g^-1 s^-2)
cdef double K_B_CGS  = 1.380649e-16        # Boltzmann constant (erg K^-1)
cdef double M_P_CGS  = 1.67262192369e-24   # Proton mass (g)
cdef double KAPPA_ES = 0.34                # Electron scattering opacity (cm^2 g^-1)
cdef double RAD_A_CGS    = 7.5657e-15           # Radiation constant (erg cm^-3 K^-4)
cdef double SIGMA_SB_CGS = 5.6703744e-5          # Stefan-Boltzmann constant (erg cm^-2 s^-1 K^-4)

# Precomputed logs — log() is not a compile-time expression.
cdef double LOG_DISK_XI     = log(DISK_XI)       # log of xi (dimensionless)
cdef double LOG_DISK_A      = log(DISK_A)        # log of A  (dimensionless)
cdef double LOG_DISK_F0     = log(DISK_F0)       # log of F0 (dimensionless)
cdef double LOG_G_CGS       = log(G_CGS)         # log of G  (cm^3 g^-1 s^-2)
cdef double LOG_PI          = log(pi)            # log of pi (dimensionless)
cdef double LOG_K_B_CGS     = log(K_B_CGS)       # log of k_B (erg K^-1)
cdef double LOG_M_P_CGS     = log(M_P_CGS)       # log of m_p (g)
cdef double LOG_KAPPA_ES    = log(KAPPA_ES)      # log of kappa_es (cm^2 g^-1)
cdef double LOG_RAD_A_CGS   = log(RAD_A_CGS)     # log of a_rad (erg cm^-3 K^-4)
cdef double LOG_SIGMA_SB_CGS = log(SIGMA_SB_CGS) # log of sigma_SB (erg cm^-2 s^-1 K^-4)

# -------------------------------------- #
# OneZoneClosure implementation          #
# -------------------------------------- #
# For each implementation of a different disk scenario, we require
# the developer to generate a ``OneZoneClosure`` object with the appropriate function pointers set.  This is the
# only way to get the closure semantics into the Cython hot loop without any Python overhead.
cdef class OneZoneClosure:

    def __cinit__(self, int n_result_fields=0):
        self._closure_fn     = NULL
        self._derivative_fn  = NULL
        self._writer_fn      = NULL
        self.n_result_fields = n_result_fields

    cpdef bint is_ready(self):
        """True if all three function pointers are installed."""
        return (self._closure_fn    != NULL and
                self._derivative_fn != NULL and
                self._writer_fn     != NULL)

# -------------------------------------- #
# Explicit Marcher                       #
# -------------------------------------- #

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int compute_one_zone_model(
        double[::1] initial_state,
        double[::1] parameters,
        double[:, ::1] result_array,
        double t_start,
        double t_end,
        int max_steps,
        closure_func closure_function,
        derivative_func derivative_function,
        writer_func writer_function,
):
    # --- Declare variables --- #
    # Loop temporaries — declared here because Cython requires all cdef declarations before executable statements.
    cdef double log_m, log_j, log_MBH, log_r, log_sigma, log_omega
    cdef int actual_steps = 0
    cdef int fn_status           # captures per-call return code before negating
    cdef DiskDerived derived_disk_params
    cdef ClosureResult closure_result
    cdef ClosureResult prev_closure

    # --- Unpack initial conditions and validate --- #
    # Using the provided initial state and the parameters, pack them into
    # the appropriate C structures for use in the closure function.
    cdef DiskState state
    cdef DiskParameters params
    cdef DiskStep step

    # Populate the state.
    state.t = t_start
    state.M = initial_state[0]
    state.J = initial_state[1]

    # Populate the parameters.
    params.MBH   = parameters[0]
    params.R_in  = parameters[1]
    params.alpha = parameters[2]
    params.mu    = parameters[3]

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

    # --- Hot loop (released from GIL) --- #
    # Column `actual_steps` is written at the START of each iteration (before the
    # state update), so column 0 holds the initial condition with its full
    # thermodynamic closure, and column k holds the state after k Euler steps.
    with nogil:
        while actual_steps < max_steps:
            # This is the hot loop of the algorithm.  Canonical step order:
            # (1) compute log quantities → (2) pack DiskDerived → (3) call closure
            # → (4) call derivative → (5) write row → (6) update state
            # → (7) copy prev_closure → (8) increment actual_steps → (9) check t_end.

            # (1) Compute log quantities.
            log_m   = log(state.M)
            log_j   = log(state.J)
            log_MBH = log(params.MBH)

            # R_D = J^2 / (xi^2 * M^2 * G * M_BH)  =>  log_r = 2*(log_j - log_xi - log_m) - log_G - log_MBH
            log_r     = 2.0 * (log_j - LOG_DISK_XI - log_m) - LOG_G_CGS - log_MBH
            # Sigma = M / (pi * A * R^2)            =>  log_sigma = log_m - 2*log_r - log_pi - log_A
            log_sigma = log_m - 2.0 * log_r - LOG_PI - LOG_DISK_A
            # Omega = sqrt(G * M_BH / R^3)          =>  log_omega = 0.5*(log_MBH + log_G - 3*log_r)
            log_omega = 0.5 * (log_MBH + LOG_G_CGS - 3.0 * log_r)

            # (2) Pack DiskDerived.
            derived_disk_params.R     = exp(log_r)
            derived_disk_params.Sigma = exp(log_sigma)
            derived_disk_params.Omega = exp(log_omega)

            # (3) Call closure — prev and out are distinct structs (no aliasing).
            # On failure, negate the find_root status so the Python layer can
            # distinguish: -1 FUNC_ERROR, -2 EXPAND_FAIL, -3 NO_BRACKET, -4 MAX_ITER.
            fn_status = closure_function(&state, &derived_disk_params, &params,
                                        &prev_closure, &closure_result)
            if fn_status != 0:
                return -fn_status

            # (4) Call derivative.
            fn_status = derivative_function(&state, &derived_disk_params, &params,
                                            &closure_result, &step)
            if fn_status != 0:
                return -10

            # (5) Write column actual_steps (column 0 = IC, column k = state after k steps).
            fn_status = writer_function(actual_steps, &state, &derived_disk_params, &params,
                                        &closure_result, &step, &result_array[0, 0], max_steps)
            if fn_status != 0:
                return -20

            # (6) Update state.
            state.t += step.dt
            state.M += step.dM_dt * step.dt
            state.J += step.dJ_dt * step.dt

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
):
    """
    Run the one-zone disk integrator; return the populated result array.

    Parameters
    ----------
    initial_state : ndarray, shape (2,)
        ``[M_D (g), J_D (g cm^2 s^-1)]``.
    parameters : ndarray, shape (4,)
        ``[M_BH (g), R_in (cm), alpha, mu]``.
    t_start, t_end : float
        Integration interval in seconds.
    max_steps : int
        Maximum number of integration steps.
    closure : OneZoneClosure
        Closure with all three function pointers installed.

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
    if parameters.shape[0] != 4:
        raise ValueError(
            f"parameters must have shape (4,), got shape ({parameters.shape[0]},). "
            f"Expected [MBH, R_in, alpha, mu]."
        )
    if t_end <= t_start:
        raise ValueError(f"t_end ({t_end}) must be > t_start ({t_start}).")
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}.")

    result = np.zeros(
        (closure.n_result_fields, max_steps), dtype=np.float64
    )

    status = compute_one_zone_model(
        initial_state, parameters, result,
        t_start, t_end, max_steps,
        closure._closure_fn,
        closure._derivative_fn,
        closure._writer_fn,
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
        }
        msg = _CLOSURE_MESSAGES.get(
            status,
            f"Integrator returned unrecognised error code {status}.",
        )
        raise RuntimeError(msg)

    # Slice to populated columns only: column 0 (IC) + actual_steps integration columns.
    # Return a C-contiguous copy so callers can rely on row-major access patterns.
    return np.ascontiguousarray(result[:, :status])
