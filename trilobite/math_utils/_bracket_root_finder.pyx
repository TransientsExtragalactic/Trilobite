#cython: language_level=3, boundscheck=False
r"""
Bracketed 1-D root finder (Cython).

Provides three ``nogil`` C-level functions for solving :math:`f(x) = 0`:

``expand_bracket``
    Starting from an initial guess :math:`x_0`, geometrically widens a
    symmetric search interval until a sign change is found.

``brent_root``
    Given an already-bracketed interval :math:`[a, b]` with
    :math:`f(a)\,f(b) < 0`, refines the root using **Brent's method** — a
    combination of inverse-quadratic interpolation (IQI), the secant method,
    and bisection that guarantees convergence and achieves near-quadratic
    rates on smooth functions.

``find_root``
    Combines both steps: expand bracket → Brent refinement.  This is the
    normal entry point; call the lower-level functions directly only when a
    tight warm-start bracket is already available.

**Callback interface**

All three functions accept a generic :c:type:`scalar_func` pointer::

    int f(double x, void *user_data, double *f_out) nogil

The callback should write :math:`f(x)` to ``*f_out`` and return 0 on
success or a nonzero error code on failure.  ``user_data`` is an opaque
pointer forwarded unchanged, allowing callers to thread arbitrary context
(e.g. a pointer to a struct holding the disk state and parameters) without
any Python overhead.

**Status codes** (returned by all three functions)

=  ==============  ========================================================
0  SUCCESS         Root written to ``*root``.
1  FUNC_ERROR      The ``scalar_func`` callback returned a nonzero code.
2  EXPAND_FAIL     ``expand_bracket`` hit ``max_expand`` without a sign change.
3  NO_BRACKET      ``brent_root`` called with :math:`f(a)\,f(b) > 0`.
4  MAX_ITER        Brent iteration hit ``maxiter``; best estimate written.
=  ==============  ========================================================
"""
from libc.math cimport fabs


# ====================================================================== #
# Bracket Finding                                                        #
# ====================================================================== #
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
) nogil:
    r"""Find a sign-changing bracket around *x_guess*.

    The search interval starts as
    :math:`[x_0 - \delta,\; x_0 + \delta]` with :math:`\delta` = *step*,
    and :math:`\delta` is multiplied by *grow_factor* on each iteration
    until :math:`f(x_{\rm lo})\,f(x_{\rm hi}) < 0`.  Both endpoints are
    clamped to ``[x_min, x_max]`` throughout.

    If an exact zero is found at an endpoint the two output pointers are set
    to the same value and the function returns 0 (SUCCESS); the caller
    should check ``x_lo == x_hi`` before passing to ``brent_root``.

    Parameters
    ----------
    f : scalar_func
        Callback ``f(x, user_data, f_out) -> int``.
    user_data : void*
        Opaque pointer forwarded to *f*.
    x_guess : double
        Centre of the initial search interval.
    step : double
        Initial half-width of the search interval (same units as *x*).
    grow_factor : double
        Multiplicative growth applied to the half-width each iteration.
        Must be > 1.  Typical value: ``2.0``.
    max_expand : int
        Maximum number of expansion iterations before giving up.
    x_min, x_max : double
        Hard bounds on the search domain.
    tolerance: double
        If the function values at the endpoints are within this absolute
        tolerance of zero, treat them as exact roots and return SUCCESS.
    x_lo_out, x_hi_out : double*
        On SUCCESS, the left and right bracket endpoints.
    f_lo_out, f_hi_out : double*
        On SUCCESS, the corresponding function values.

    Returns
    -------
    int
        0 SUCCESS, 1 FUNC_ERROR, 2 EXPAND_FAIL.
    """
    # Declare the variables for the root finder.
    cdef double x_lo, x_hi, f_lo, f_hi, half_width
    cdef int i

    # Setup up the initial bracket and clamp to the domain bounds.
    half_width = step
    x_lo = x_guess - half_width
    x_hi = x_guess + half_width
    if x_lo < x_min:
        x_lo = x_min
    if x_hi > x_max:
        x_hi = x_max

    # --- Prelim Check --- #
    # Before doing any expansion steps, check if we
    # already have a sign change or an exact root at the initial endpoints.

    # Check for failures at either edge of the bracket. If so, we
    # just fail.
    if f(x_lo, user_data, &f_lo) != 0:
        return 1
    if f(x_hi, user_data, &f_hi) != 0:
        return 1

    # If instead we are right at zero, we can just return that as the root.
    if fabs(f_lo) < tolerance:
        x_lo_out[0] = x_lo
        x_hi_out[0] = x_lo
        f_lo_out[0] = 0.0
        f_hi_out[0] = 0.0
        return 0

    if fabs(f_hi) < tolerance:
        x_lo_out[0] = x_hi
        x_hi_out[0] = x_hi
        f_lo_out[0] = 0.0
        f_hi_out[0] = 0.0
        return 0

    # Check if we have already found a valid bracket.  If so, return it immediately without doing
    # any expansion steps.
    if f_lo * f_hi < 0.0:
        x_lo_out[0] = x_lo
        x_hi_out[0] = x_hi
        f_lo_out[0] = f_lo
        f_hi_out[0] = f_hi
        return 0

    # --- Geometric Expansion --- #
    # We now cycle through the expansion steps, geometrically increasing the half-width of the search interval
    # until we find a sign change or exhaust the maximum number of expansions.  If we exhaust the maximum
    # number of expansions, we return EXPAND_FAIL.
    for i in range(max_expand):

        # Expand the grid.
        half_width *= grow_factor
        x_lo = x_guess - half_width
        x_hi = x_guess + half_width

        # Clamp to the domain bounds.
        if x_lo < x_min:
            x_lo = x_min
        if x_hi > x_max:
            x_hi = x_max

        # Check for errors in the function evaluation at either edge of the bracket. If so, we
        # just fail.
        if f(x_lo, user_data, &f_lo) != 0:
            return 1
        if f(x_hi, user_data, &f_hi) != 0:
            return 1

        # Check for exact roots at the endpoints.  If so, we can just return that as the root.
        if fabs(f_lo) < tolerance:
            x_lo_out[0] = x_lo
            x_hi_out[0] = x_lo
            f_lo_out[0] = 0.0
            f_hi_out[0] = 0.0
            return 0
        if fabs(f_hi) < tolerance:
            x_lo_out[0] = x_hi
            x_hi_out[0] = x_hi
            f_lo_out[0] = 0.0
            f_hi_out[0] = 0.0
            return 0

        # Check for a sign change.  If we have one, return the bracket and function values.
        if f_lo * f_hi < 0.0:
            x_lo_out[0] = x_lo
            x_hi_out[0] = x_hi
            f_lo_out[0] = f_lo
            f_hi_out[0] = f_hi
            return 0

        # No room left to expand — give up.
        if x_lo == x_min and x_hi == x_max:
            return 2

    return 2  # EXPAND_FAIL


# ====================================================================== #
# brent_root                                                              #
# ====================================================================== #

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
) nogil:
    r"""Refine a root within ``[x_lo, x_hi]`` using Brent's method.

    *f_lo* and *f_hi* must satisfy :math:`f_{\rm lo}\,f_{\rm hi} < 0`
    (a proper bracket).  Passing pre-evaluated endpoint values avoids
    re-evaluating them — callers that obtained the bracket from
    ``expand_bracket`` should forward ``f_lo_out`` / ``f_hi_out`` directly.

    **Algorithm** (Brent-Dekker)

    At each iteration the next candidate point *s* is computed by:

    1. **Inverse-quadratic interpolation (IQI)** if three distinct function
       values are available (*a*, *b*, *c* are distinct).
    2. **Secant method** otherwise.

    The candidate is accepted only if it falls inside the bracket and makes
    sufficient progress relative to the previous two steps; otherwise
    **bisection** is used.  This guarantees that the bracket halves at
    worst every two iterations (same asymptotic rate as bisection) while
    achieving near-quadratic convergence on smooth functions.

    Parameters
    ----------
    f : scalar_func
        Callback ``f(x, user_data, f_out) -> int``.
    user_data : void*
        Opaque pointer forwarded to *f*.
    x_lo, x_hi : double
        Bracket endpoints with ``f_lo * f_hi < 0``.
    f_lo, f_hi : double
        Pre-evaluated function values at the bracket endpoints.
    tol_x : double
        Bracket-width convergence threshold (absolute, same units as *x*).
    tol_f : double
        Function-value convergence threshold (absolute).
    maxiter : int
        Maximum number of function evaluations inside this call.
    root : double*
        On SUCCESS or MAX_ITER, the best root estimate.

    Returns
    -------
    int
        0 SUCCESS, 1 FUNC_ERROR, 3 NO_BRACKET, 4 MAX_ITER.
    """
    cdef double a, b, c, d, s
    cdef double fa, fb, fc, fs
    cdef double tmp, lo_bound, hi_bound
    cdef bint mflag
    cdef int i

    # Guard: require a proper bracket.
    if f_lo * f_hi > 0.0:
        return 3  # NO_BRACKET

    # Initialise so that b is the "better" endpoint (|f(b)| <= |f(a)|).
    a = x_lo;  fa = f_lo
    b = x_hi;  fb = f_hi
    if fabs(fa) < fabs(fb):
        tmp = a;  a = b;  b = tmp
        tmp = fa; fa = fb; fb = tmp

    c = a;  fc = fa
    d = 0.0      # read only when mflag is False; initialised to silence compiler
    mflag = True

    for i in range(maxiter):
        # ---- Convergence checks ------------------------------------------ #
        if fabs(fb) < tol_f:
            root[0] = b
            return 0
        if fabs(b - a) < tol_x:
            root[0] = b
            return 0

        # ---- Compute interpolated candidate s ----------------------------- #
        if fa != fc and fb != fc:
            # Inverse-quadratic interpolation.
            s = (a * fb * fc / ((fa - fb) * (fa - fc))
               + b * fa * fc / ((fb - fa) * (fb - fc))
               + c * fa * fb / ((fc - fa) * (fc - fb)))
        else:
            # Secant method.
            s = b - fb * (b - a) / (fb - fa)

        # ---- Decide whether to accept s or fall back to bisection --------- #
        # Bisection conditions (any one triggers fallback):
        #   (i)  s falls outside the open interval between (3a+b)/4 and b.
        #   (ii) Last step used bisection and |s−b| >= |b−c|/2.
        #  (iii) Last step used interpolation and |s−b| >= |c−d|/2.
        #   (iv) Last step used bisection and |b−c| < tol_x.
        #   (v)  Last step used interpolation and |c−d| < tol_x.
        lo_bound = 0.25 * (3.0 * a + b)
        hi_bound = b
        if lo_bound > hi_bound:
            tmp = lo_bound;  lo_bound = hi_bound;  hi_bound = tmp

        if (not (lo_bound < s < hi_bound)
                or (mflag     and fabs(s - b) >= 0.5 * fabs(b - c))
                or (not mflag and fabs(s - b) >= 0.5 * fabs(c - d))
                or (mflag     and fabs(b - c) < tol_x)
                or (not mflag and fabs(c - d) < tol_x)):
            s = 0.5 * (a + b)   # bisection step
            mflag = True
        else:
            mflag = False

        # ---- Evaluate f(s) ------------------------------------------------ #
        if f(s, user_data, &fs) != 0:
            return 1

        # ---- Update bracket ----------------------------------------------- #
        d = c
        c = b;  fc = fb
        if fa * fs < 0.0:
            b = s;  fb = fs
        else:
            a = s;  fa = fs

        # Keep b as the endpoint with smaller |f|.
        if fabs(fa) < fabs(fb):
            tmp = a;  a = b;  b = tmp
            tmp = fa; fa = fb; fb = tmp

    root[0] = b
    return 4  # MAX_ITER


# ====================================================================== #
# find_root                                                               #
# ====================================================================== #

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
) nogil:
    r"""Find a root of *f* starting from an initial guess.

    Calls ``expand_bracket`` to locate a sign-changing interval, then
    ``brent_root`` to refine it.  This is the normal entry point.

    Call ``expand_bracket`` + ``brent_root`` separately only if you already
    have a tight bracket from a previous timestep (e.g. a warm-started
    :math:`[\log T_{c,n-1} - \varepsilon,\; \log T_{c,n-1} + \varepsilon]`).

    Parameters
    ----------
    f : scalar_func
        Callback ``f(x, user_data, f_out) -> int``.
    user_data : void*
        Opaque pointer forwarded to *f*.
    x_guess : double
        Initial guess for the root (centre of the first search interval).
    step : double
        Initial half-width of the bracket search interval.
    grow_factor : double
        Geometric expansion factor for the bracket search.  Must be > 1.
    max_expand : int
        Maximum expansion iterations before returning EXPAND_FAIL.
    x_min, x_max : double
        Hard domain bounds for the bracket search.
    tolerance : double
        Absolute function-value threshold for treating an endpoint as an exact
        root during bracket expansion (forwarded to ``expand_bracket``).
    tol_x : double
        Bracket-width convergence tolerance (passed to ``brent_root``).
    tol_f : double
        Function-value convergence tolerance (passed to ``brent_root``).
    maxiter : int
        Maximum Brent iterations (passed to ``brent_root``).
    root : double*
        On SUCCESS or MAX_ITER, the best root estimate.

    Returns
    -------
    int
        0 SUCCESS, 1 FUNC_ERROR, 2 EXPAND_FAIL, 4 MAX_ITER.
    """
    cdef double x_lo, x_hi, f_lo, f_hi, f_guess
    cdef int status

    # Fast path: exact root at the initial guess.
    if f(x_guess, user_data, &f_guess) != 0:
        return 1
    if f_guess == 0.0:
        root[0] = x_guess
        return 0

    # Find a sign-changing bracket.
    status = expand_bracket(
        f, user_data,
        x_guess, step, grow_factor, max_expand, x_min, x_max,
        tolerance, &x_lo, &x_hi, &f_lo, &f_hi,
    )
    if status != 0:
        return status

    # Exact root found during expansion (expand_bracket sets x_lo == x_hi).
    if x_lo == x_hi:
        root[0] = x_lo
        return 0

    # Refine with Brent's method.
    return brent_root(
        f, user_data,
        x_lo, x_hi, f_lo, f_hi,
        tol_x, tol_f, maxiter, root,
    )


# ====================================================================== #
# Python-callable wrappers (for testing and Python-layer callers)        #
# ====================================================================== #

cdef int _py_trampoline(double x, void* user_data, double* f_out) nogil:
    """Call a Python callable stored as a borrowed reference in *user_data*."""
    with gil:
        try:
            f_out[0] = (<object>user_data)(x)
            return 0
        except Exception:
            return 1


def py_expand_bracket(
    object f,
    double x_guess,
    double step,
    double grow_factor,
    int max_expand,
    double x_min,
    double x_max,
    double tolerance,
):
    r"""Python wrapper around :func:`expand_bracket`.

    Accepts a plain Python callable ``f(x) -> float`` and searches for a
    sign-changing bracket around *x_guess* using geometric expansion.

    Parameters
    ----------
    f : callable
        Function ``f(x) -> float``.  Must be finite and continuous in the
        search region.
    x_guess : float
        Centre of the initial search interval.
    step : float
        Initial half-width of the search interval.
    grow_factor : float
        Multiplicative growth applied to the half-width each iteration (> 1).
    max_expand : int
        Maximum number of expansion iterations.
    x_min, x_max : float
        Hard bounds on the search domain.
    tolerance : float
        Absolute function-value threshold below which an endpoint is treated
        as an exact root (returns SUCCESS immediately).

    Returns
    -------
    status : int
        0 SUCCESS, 1 FUNC_ERROR, 2 EXPAND_FAIL.
    x_lo, x_hi : float
        Bracket endpoints (equal if an exact root was found).
    f_lo, f_hi : float
        Function values at the bracket endpoints.
    """
    cdef double x_lo, x_hi, f_lo, f_hi
    cdef int status

    status = expand_bracket(
        _py_trampoline, <void*>f,
        x_guess, step, grow_factor, max_expand, x_min, x_max,
        tolerance, &x_lo, &x_hi, &f_lo, &f_hi,
    )
    return status, x_lo, x_hi, f_lo, f_hi


def py_brent_root(
    object f,
    double x_lo,
    double x_hi,
    double f_lo,
    double f_hi,
    double tol_x,
    double tol_f,
    int maxiter,
):
    r"""Python wrapper around :func:`brent_root`.

    Refines a root within a pre-validated bracket ``[x_lo, x_hi]`` using
    Brent's method (IQI / secant / bisection).

    Parameters
    ----------
    f : callable
        Function ``f(x) -> float``.
    x_lo, x_hi : float
        Bracket endpoints.  ``f(x_lo) * f(x_hi)`` must be negative.
    f_lo, f_hi : float
        Pre-evaluated function values at *x_lo* and *x_hi*.
    tol_x : float
        Bracket-width convergence threshold (absolute, same units as *x*).
    tol_f : float
        Function-value convergence threshold (absolute).
    maxiter : int
        Maximum number of function evaluations.

    Returns
    -------
    status : int
        0 SUCCESS, 1 FUNC_ERROR, 3 NO_BRACKET, 4 MAX_ITER.
    root : float
        Best root estimate (written on SUCCESS and MAX_ITER).
    """
    cdef double root = 0.0
    cdef int status

    status = brent_root(
        _py_trampoline, <void*>f,
        x_lo, x_hi, f_lo, f_hi,
        tol_x, tol_f, maxiter, &root,
    )
    return status, root


def py_find_root(
    object f,
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
):
    r"""Python wrapper around :func:`find_root`.

    Combines bracket expansion and Brent refinement into a single call.
    This is the normal entry point for Python callers.

    Parameters
    ----------
    f : callable
        Function ``f(x) -> float``.
    x_guess : float
        Initial guess for the root.
    step : float
        Initial half-width of the bracket search interval.
    grow_factor : float
        Geometric expansion factor (> 1).
    max_expand : int
        Maximum expansion iterations before returning EXPAND_FAIL.
    x_min, x_max : float
        Hard domain bounds.
    tolerance : float
        Absolute function-value threshold for treating an endpoint as an
        exact root during bracket expansion.
    tol_x : float
        Bracket-width convergence tolerance for Brent refinement.
    tol_f : float
        Function-value convergence tolerance for Brent refinement.
    maxiter : int
        Maximum Brent iterations.

    Returns
    -------
    status : int
        0 SUCCESS, 1 FUNC_ERROR, 2 EXPAND_FAIL, 4 MAX_ITER.
    root : float
        Best root estimate (written on SUCCESS and MAX_ITER).
    """
    cdef double root = 0.0
    cdef int status

    status = find_root(
        _py_trampoline, <void*>f,
        x_guess, step, grow_factor, max_expand, x_min, x_max,
        tolerance, tol_x, tol_f, maxiter, &root,
    )
    return status, root
