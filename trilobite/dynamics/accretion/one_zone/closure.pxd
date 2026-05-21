from cpython.ref cimport PyObject
from trilobite.radiation.opacity.opacity_base cimport C_GreyOpacityBase

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
# Physics Component Parameter Structs      #
# ---------------------------------------- #
# Named structs replace the old ``double* extra`` catch-all in DiskParameters.
# Each physics feature owns its own struct, which is pointed to from
# DiskParameters.  NULL means the feature is disabled.

cdef struct FallbackParams:
    double M_fb_0   # g s^-1  — fallback rate at the reference time t_fb
    double R_c      # cm      — circularisation radius
    double t_fb     # s       — reference (peak) time
    double beta_fb  # dimensionless — power-law index (typically 5/3)

cdef struct AdvectionParams:
    double xi       # dimensionless — entropy gradient parameter (> 0)

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
    double MBH          # g
    double R_in         # cm
    double alpha        # dimensionless
    double mu           # dimensionless (mean molecular weight)
    double epsilon      # dimensionless (adaptive time-step fraction)
    FallbackParams* fallback   # NULL = fallback disabled
    AdvectionParams* advection  # NULL = advection disabled
    void*  opacity      # raw PyObject* to a C_GreyOpacityBase instance (set by integrator)

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
    double log_rho    # ln(g cm^-3) — midplane density; filled by every closure
    double* extra     # pointer into the closure result memoryview (may be NULL)
    int     n_extra   # number of extra closure result fields

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
# run_one_zone_model return conventions:
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

# ---------------------------------------- #
# OneZoneClosure extension type            #
# ---------------------------------------- #

cdef class OneZoneClosure:
    """
    Abstract extension type wrapping the C function pointers consumed by
    :func:`~trilobite.dynamics.accretion.one_zone.integrator.run_one_zone_model`.

    Subclass in a concrete closure ``.pyx`` file; set ``_closure_fn``,
    ``_derivative_fn``, ``_writer_fn`` (and optionally ``_source_fn``) from
    ``__cinit__`` to install concrete implementations.

    Attributes
    ----------
    n_result_fields : int
        Number of output rows per time step.  Read-only from Python.

    Notes
    -----
    ``_source_fn`` defaults to ``NULL``, meaning no source term is applied.
    The hot loop skips the source call entirely when it is ``NULL``, so
    closures with no source term pay zero overhead.

    ``_opacity_ptr`` is a pre-extracted raw ``void*`` to the C-level opacity
    object. It is set automatically by the ``opacity`` property setter and used
    in ``_pack_params`` without acquiring the GIL.
    """
    cdef closure_func    _closure_fn
    cdef derivative_func _derivative_fn
    cdef writer_func     _writer_fn
    cdef source_func     _source_fn      # NULL = no source term (default)
    cdef C_GreyOpacityBase _c_opacity    # typed reference (keeps object alive for GC)
    cdef object          _opacity_obj    # original GreyOpacityLaw (for opacity getter)
    cdef void*           _opacity_ptr    # pre-extracted raw pointer (GIL-free in _pack_params)
    cdef readonly int    n_result_fields

    cpdef bint is_ready(self)

    cdef void _pack_params(self, DiskParameters* p) nogil
