# ---------------------------------------- #
# Constants                                #
# ---------------------------------------- #
# All physical constants are cdef double so they can be cimported by any .pyx
# that needs them — single source of truth, no DEF duplication across files.
cdef double DISK_A    # 1.62     — Metzger+08 disk correction factor (dimensionless)
cdef double DISK_B    # 1.33     — Metzger+08 disk correction factor (dimensionless)
cdef double DISK_F0   # 1.6      — Metzger+08 disk correction factor (dimensionless)
cdef double DISK_XI   # B/A      — Metzger+08 disk correction factor (dimensionless)
cdef double G_CGS     # 6.674e-8 — Gravitational constant (cm^3 g^-1 s^-2)
cdef double K_B_CGS   # 1.381e-16 — Boltzmann constant (erg K^-1)
cdef double M_P_CGS   # 1.673e-24 — Proton mass (g)
cdef double KAPPA_ES  # 0.34     — Electron scattering opacity (cm^2 g^-1)
cdef double RAD_A_CGS    # 7.5657e-15  — Radiation constant (erg cm^-3 K^-4)
cdef double SIGMA_SB_CGS # 5.6703744e-5 — Stefan-Boltzmann constant (erg cm^-2 s^-1 K^-4)

# Precomputed logs (log() is not a compile-time expression).
cdef double LOG_DISK_XI
cdef double LOG_DISK_A
cdef double LOG_DISK_F0
cdef double LOG_G_CGS
cdef double LOG_PI
cdef double LOG_K_B_CGS
cdef double LOG_M_P_CGS
cdef double LOG_KAPPA_ES
cdef double LOG_RAD_A_CGS
cdef double LOG_SIGMA_SB_CGS

# ---------------------------------------- #
# Core Structures                          #
# ---------------------------------------- #

cdef struct DiskState:
    double t       # s
    double M       # g
    double J       # g cm^2 s^-1

cdef struct DiskStep:
    double dt      # s
    double dM_dt   # g s^-1
    double dJ_dt   # g cm^2 s^-2

cdef struct DiskParameters:
    double MBH     # g
    double R_in    # cm
    double alpha   # dimensionless
    double mu      # dimensionless (mean molecular weight)
    double* extra  # pointer into the parameters memoryview (may be NULL)
    int    n_extra # number of extra parameters

cdef struct DiskDerived:
    double R       # cm
    double Sigma   # g cm^-2
    double Omega   # s^-1

cdef struct ClosureResult:
    double dM_dt      # g s^-1        (negative: mass drain)
    double dJ_dt      # g cm^2 s^-2   (negative: angular-momentum drain)
    double t_visc     # s              (viscous timescale; warm-start for next step)
    double log_T_eff  # ln(K)
    double log_T_c    # ln(K)
    double log_tau    # ln(dimensionless)
    double log_cs     # ln(cm s^-1)
    double log_nu     # ln(cm^2 s^-1)
    double log_q_visc # ln(erg cm^-2 s^-1)

# ---------------------------------------- #
# Function Pointer Typedefs                #
# ---------------------------------------- #

ctypedef int (*closure_func)(
    const DiskState* state,
    const DiskDerived* derived,
    const DiskParameters* params,
    const ClosureResult* prev,
    ClosureResult* out
) nogil

ctypedef int (*derivative_func)(
    const DiskState* state,
    const DiskDerived* derived,
    const DiskParameters* params,
    const ClosureResult* closure,
    DiskStep* out
) nogil

ctypedef int (*writer_func)(
    const int step_index,
    const DiskState* state,
    const DiskDerived* derived,
    const DiskParameters* params,
    const ClosureResult* closure,
    const DiskStep* step,
    double* result_array,
    int n_steps
) nogil

ctypedef int (*source_func)(
    const DiskState* state,
    const DiskDerived* derived,
    const DiskParameters* params,
    const ClosureResult* closure,
    DiskStep* step
) nogil

# ---------------------------------------- #
# Integrator status codes                  #
# ---------------------------------------- #
# compute_one_zone_model / run_one_zone_model return conventions:
#
#   >= 0  SUCCESS — value is the number of steps actually written.
#   -1    CLOSURE / FUNC_ERROR    — root-finding callback returned non-zero
#                                   (NaN / Inf in state or residual).
#   -2    CLOSURE / EXPAND_FAIL   — bracket expansion failed; no sign change found
#                                   in [T_min, T_max] (disk state outside physical range).
#   -3    CLOSURE / NO_BRACKET    — initial bracket had the same sign at both endpoints.
#   -4    CLOSURE / MAX_ITER      — Brent's method did not converge within maxiter.
#   -10   DERIVATIVE_FAIL         — derivative function returned non-zero.
#   -20   WRITER_FAIL             — writer function returned non-zero.
#   -30   SOURCE_FAIL             — source function returned non-zero.
#
# The magnitude of a closure failure code equals the find_root status:
#   |status| == 1 → FUNC_ERROR, 2 → EXPAND_FAIL, 3 → NO_BRACKET, 4 → MAX_ITER.

# ---------------------------------------- #
# OneZoneClosure extension type            #
# ---------------------------------------- #

cdef class OneZoneClosure:
    """
    Extension type wrapping the C function pointers consumed by
    :func:`compute_one_zone_model`.

    Subclass in ``_closure.pyx``; set ``_closure_fn``, ``_derivative_fn``,
    ``_writer_fn`` (and optionally ``_source_fn``) from ``__cinit__`` to
    install concrete implementations.

    Attributes
    ----------
    n_result_fields : int
        Number of output columns per time step.  Read-only from Python.

    Notes
    -----
    ``_source_fn`` defaults to ``NULL``, meaning no source term is applied.
    The hot loop skips the source call entirely when it is ``NULL``, so
    existing closures with no source term pay zero overhead.
    """
    cdef closure_func    _closure_fn
    cdef derivative_func _derivative_fn
    cdef writer_func     _writer_fn
    cdef source_func     _source_fn      # NULL = no source term (default)
    cdef readonly int    n_result_fields

    cpdef bint is_ready(self)
