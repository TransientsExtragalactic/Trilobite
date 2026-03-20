# ------------------------------------------ #
# Bracketed 1-D root finder — declarations   #
# ------------------------------------------ #

# Generic scalar callback: write f(x) to *f_out, return 0 on success.
# The opaque void* pointer is forwarded unchanged from the call site,
# allowing callers to pass a struct pointer carrying all necessary context.
ctypedef int (*scalar_func)(double x, void* user_data, double* f_out) nogil

# ---------------------------------------------------------------------- #
# Status codes (documented here; returned as plain int for nogil compat)  #
# ---------------------------------------------------------------------- #
#   0  SUCCESS      — root written to *root.
#   1  FUNC_ERROR   — the scalar_func callback returned a nonzero error code.
#   2  EXPAND_FAIL  — expand_bracket exhausted max_expand without a sign change.
#   3  NO_BRACKET   — brent_root called with an interval where f has equal signs.
#   4  MAX_ITER     — Brent iteration hit maxiter; best estimate written to *root.

cdef int expand_bracket(
    scalar_func f,
    void* user_data,
    double x_guess,
    double step,
    double grow_factor,
    int max_expand,
    double x_min,
    double x_max,
    double tolerance,
    double* x_lo_out,
    double* x_hi_out,
    double* f_lo_out,
    double* f_hi_out,
) nogil

cdef int brent_root(
    scalar_func f,
    void* user_data,
    double x_lo,
    double x_hi,
    double f_lo,
    double f_hi,
    double tol_x,
    double tol_f,
    int maxiter,
    double* root,
) nogil

cdef int find_root(
    scalar_func f,
    void* user_data,
    double x_guess,
    double step,
    double grow_factor,
    int max_expand,
    double x_min,
    double x_max,
    double tolerance,
    double tol_x,
    double tol_f,
    int maxiter,
    double* root,
) nogil
