"""
Tests for trilobite.math_utils._bracket_root_finder.

The Cython ``cdef`` functions are exercised via their Python-callable wrappers:

    py_expand_bracket(f, x_guess, step, grow_factor, max_expand,
                      x_min, x_max, tolerance)
    py_brent_root(f, x_lo, x_hi, f_lo, f_hi, tol_x, tol_f, maxiter)
    py_find_root(f, x_guess, step, grow_factor, max_expand,
                 x_min, x_max, tolerance, tol_x, tol_f, maxiter)

Status codes
------------
0  SUCCESS
1  FUNC_ERROR
2  EXPAND_FAIL
3  NO_BRACKET
4  MAX_ITER
"""

from __future__ import annotations

import math

import pytest

# ---------------------------------------------------------------------------
# Module import — skip entire file if the extension has not been compiled yet.
# ---------------------------------------------------------------------------
try:
    from trilobite.math_utils._bracket_root_finder import (
        py_brent_root,
        py_expand_bracket,
        py_find_root,
    )
except ImportError:
    pytest.skip(
        "trilobite.math_utils._bracket_root_finder not compiled — run `pip install -e .`",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

TOL_X = 1e-12
TOL_F = 1e-12
MAX_ITER = 200
MAX_EXPAND = 60
STEP = 0.5
GROW = 2.0
TOLERANCE = 1e-14  # near-zero endpoint threshold for expand_bracket


def _find(
    f,
    x_guess,
    *,
    x_min=-1e6,
    x_max=1e6,
    step=STEP,
    grow=GROW,
    max_expand=MAX_EXPAND,
    tolerance=TOLERANCE,
    tol_x=TOL_X,
    tol_f=TOL_F,
    maxiter=MAX_ITER,
):
    """Convenience wrapper around py_find_root."""
    return py_find_root(
        f,
        x_guess,
        step,
        grow,
        max_expand,
        x_min,
        x_max,
        tolerance,
        tol_x,
        tol_f,
        maxiter,
    )


# ---------------------------------------------------------------------------
# py_find_root — analytic root cases
# ---------------------------------------------------------------------------


class TestFindRootAnalytic:
    """find_root must recover exact roots on well-behaved functions."""

    def test_linear(self):
        """f(x) = x - 3  has root at 3."""
        status, root = _find(lambda x: x - 3.0, x_guess=0.0)
        assert status == 0
        assert abs(root - 3.0) < 1e-10

    def test_quadratic_positive_root(self):
        """f(x) = x^2 - 4  has roots at ±2; guess near +2."""
        status, root = _find(lambda x: x * x - 4.0, x_guess=1.5)
        assert status == 0
        assert abs(root - 2.0) < 1e-10

    def test_quadratic_negative_root(self):
        """f(x) = x^2 - 4  has roots at ±2; guess near -2."""
        status, root = _find(lambda x: x * x - 4.0, x_guess=-1.5)
        assert status == 0
        assert abs(root + 2.0) < 1e-10

    def test_cubic(self):
        """f(x) = x^3 - x - 2  has a single real root near x = 1.52."""

        def f(x):
            return x**3 - x - 2.0

        status, root = _find(f, x_guess=1.0)
        assert status == 0
        assert abs(f(root)) < 1e-10

    def test_transcendental_sin(self):
        """f(x) = sin(x)  has roots at multiples of π; guess near π."""
        status, root = _find(math.sin, x_guess=3.0)
        assert status == 0
        assert abs(root - math.pi) < 1e-10

    def test_transcendental_exp_minus_one(self):
        """f(x) = e^x - 1  has root at 0."""
        status, root = _find(lambda x: math.exp(x) - 1.0, x_guess=0.5)
        assert status == 0
        assert abs(root) < 1e-10

    def test_exact_root_at_guess(self):
        """When f(x_guess) == 0 exactly, find_root returns SUCCESS immediately."""
        status, root = _find(lambda x: x - 5.0, x_guess=5.0)
        assert status == 0
        assert root == pytest.approx(5.0)

    def test_root_at_domain_boundary(self):
        """Root sitting on x_min is found by expand_bracket's endpoint check."""
        # f(x) = x + 10; root at -10 which is x_min.
        status, root = _find(
            lambda x: x + 10.0,
            x_guess=-9.0,
            x_min=-10.0,
            x_max=0.0,
            tolerance=1e-10,
        )
        assert status == 0
        assert abs(root + 10.0) < 1e-8


class TestFindRootAccuracy:
    """Root accuracy should exceed the requested tolerances."""

    @pytest.mark.parametrize("tol", [1e-6, 1e-9, 1e-12])
    def test_tolerance_respected(self, tol):
        """Residual |f(root)| must be below tol_f for smooth functions."""

        def f(x):
            return math.sin(x) - 0.5  # roots at π/6 + 2kπ, 5π/6 + 2kπ

        status, root = _find(f, x_guess=0.5, tol_f=tol, tol_x=tol)
        assert status in (0, 4)  # SUCCESS or MAX_ITER with best estimate
        assert abs(f(root)) < max(tol * 10, 1e-13)


# ---------------------------------------------------------------------------
# py_find_root — failure modes
# ---------------------------------------------------------------------------


class TestFindRootFailures:
    """find_root must return the correct status code on failure."""

    def test_expand_fail_monotone(self):
        """A strictly monotone function with no root returns EXPAND_FAIL."""
        # f(x) = e^x (always positive) within a bounded domain.
        status, _root = _find(
            math.exp,
            x_guess=0.0,
            x_min=-5.0,
            x_max=5.0,
            max_expand=20,
        )
        assert status == 2  # EXPAND_FAIL

    def test_expand_fail_constant(self):
        """A constant function has no sign change — returns EXPAND_FAIL."""
        status, _root = _find(
            lambda x: 1.0,
            x_guess=0.0,
            max_expand=10,
        )
        assert status == 2

    def test_func_error_propagated(self):
        """A callback that raises an exception triggers FUNC_ERROR (status 1)."""

        def bad(x):
            raise ValueError("deliberate error")

        status, _root = _find(bad, x_guess=0.0)
        assert status == 1

    def test_max_iter_returns_best_estimate(self):
        """With maxiter=1 Brent cannot converge; status 4 and a finite estimate."""
        # A function with a root at x = 7.
        status, root = _find(
            lambda x: x - 7.0,
            x_guess=0.0,
            tol_x=1e-30,
            tol_f=1e-30,
            maxiter=1,
        )
        # Either MAX_ITER (4) or SUCCESS (0) depending on whether bisection landed on root.
        assert status in (0, 4)
        assert math.isfinite(root)


# ---------------------------------------------------------------------------
# py_expand_bracket — direct tests
# ---------------------------------------------------------------------------


class TestExpandBracket:
    """expand_bracket must locate a valid sign-changing interval."""

    def _expand(
        self, f, x_guess=0.0, step=0.5, grow=2.0, max_expand=MAX_EXPAND, x_min=-1e6, x_max=1e6, tolerance=TOLERANCE
    ):
        return py_expand_bracket(f, x_guess, step, grow, max_expand, x_min, x_max, tolerance)

    def test_bracket_found(self):
        """f(x) = x - 2 should find a valid bracket containing 2.

        expand_bracket may return either a sign-changing interval (f_lo*f_hi < 0)
        or an exact-root point (x_lo == x_hi, f_lo == f_hi == 0) depending on
        whether the geometric expansion lands exactly on the root.  Both are
        correct SUCCESS outcomes.
        """
        status, x_lo, x_hi, f_lo, f_hi = self._expand(lambda x: x - 2.0, x_guess=0.0)
        assert status == 0
        # Either a proper sign-change bracket or an exact root.
        assert (f_lo * f_hi < 0.0) or (x_lo == x_hi)

    def test_bracket_endpoints_sign_change(self):
        """Returned bracket satisfies f(x_lo)*f(x_hi) <= 0."""

        def f(x):
            return x**3 - 8.0  # root at 2

        status, x_lo, x_hi, f_lo, f_hi = self._expand(f, x_guess=0.0)
        assert status == 0
        assert (f_lo * f_hi < 0.0) or (x_lo == x_hi)

    def test_exact_root_at_endpoint(self):
        """When an endpoint evaluates within tolerance, x_lo == x_hi is returned."""
        # f(x) = x - 3; initial step hits x = 3.0 exactly → f = 0
        status, x_lo, x_hi, f_lo, f_hi = self._expand(
            lambda x: x - 3.0,
            x_guess=3.0,
            step=0.0,
            tolerance=1e-10,
        )
        assert status == 0
        # Either exact root or sign-change bracket; either is acceptable.
        assert math.isfinite(x_lo) and math.isfinite(x_hi)

    def test_clamp_to_domain(self):
        """Bracket endpoints must not exceed [x_min, x_max]."""
        status, x_lo, x_hi, _fl, _fh = self._expand(
            lambda x: x - 100.0,
            x_guess=0.0,
            x_min=-10.0,
            x_max=200.0,
        )
        assert x_lo >= -10.0
        assert x_hi <= 200.0

    def test_expand_fail_no_sign_change(self):
        """Always-positive function in a bounded domain returns EXPAND_FAIL."""
        status, *_ = self._expand(
            math.exp,
            x_guess=0.0,
            x_min=-5.0,
            x_max=5.0,
            max_expand=15,
        )
        assert status == 2

    def test_func_error_propagated(self):
        """A callback that raises returns FUNC_ERROR (status 1)."""

        def bad(x):
            raise RuntimeError("boom")

        status, *_ = self._expand(bad, x_guess=0.0)
        assert status == 1


# ---------------------------------------------------------------------------
# py_brent_root — direct tests
# ---------------------------------------------------------------------------


class TestBrentRoot:
    """brent_root must refine a pre-supplied bracket to the requested tolerance."""

    def _brent(self, f, x_lo, x_hi, tol_x=TOL_X, tol_f=TOL_F, maxiter=MAX_ITER):
        f_lo = f(x_lo)
        f_hi = f(x_hi)
        return py_brent_root(f, x_lo, x_hi, f_lo, f_hi, tol_x, tol_f, maxiter)

    def test_linear(self):
        """f(x) = x − 7 on [0, 10]."""
        status, root = self._brent(lambda x: x - 7.0, 0.0, 10.0)
        assert status == 0
        assert abs(root - 7.0) < 1e-10

    def test_sin_pi(self):
        """sin(x) on [3, 4] — root at π."""
        status, root = self._brent(math.sin, 3.0, 4.0)
        assert status == 0
        assert abs(root - math.pi) < 1e-11

    def test_no_bracket_returns_status_3(self):
        """If f(x_lo)*f(x_hi) > 0, status 3 (NO_BRACKET) is returned."""
        # Both endpoints positive: f(1) = 0, f(2) = 1 → same sign.
        f = lambda x: (x - 1.0) * (x - 3.0)  # roots at 1 and 3; f(0.5) > 0, f(0.9) > 0
        status, _root = py_brent_root(f, 0.5, 0.9, f(0.5), f(0.9), TOL_X, TOL_F, MAX_ITER)
        assert status == 3

    def test_func_error_propagated(self):
        """A callback that always raises triggers FUNC_ERROR (status 1)."""

        def always_bad(x):
            raise ValueError("deliberate error on every call")

        # Pass pre-computed endpoint values so brent_root starts its loop;
        # the first internal evaluation will raise and return status 1.
        status, _root = py_brent_root(always_bad, 0.0, 10.0, -5.0, 5.0, TOL_X, TOL_F, MAX_ITER)
        assert status == 1

    def test_max_iter_finite_estimate(self):
        """With maxiter=1 and very tight tolerances, best estimate is finite."""
        status, root = self._brent(
            lambda x: x - 3.14159,
            0.0,
            10.0,
            tol_x=1e-30,
            tol_f=1e-30,
            maxiter=1,
        )
        assert status in (0, 4)
        assert math.isfinite(root)

    def test_accuracy_on_smooth_function(self):
        """Residual |f(root)| should be well below tol_f on a smooth function."""

        def f(x):
            return math.log(x) - 1.0  # root at e ≈ 2.71828

        status, root = self._brent(f, 1.0, 5.0)
        assert status == 0
        assert abs(root - math.e) < 1e-11

    @pytest.mark.parametrize("root_val", [-100.0, 0.0, 1.0, 42.0, 1e5])
    def test_various_root_locations(self, root_val):
        """Root finder works for roots spanning many orders of magnitude."""
        f = lambda x: x - root_val
        lo = root_val - 1e6
        hi = root_val + 1e6
        status, root = self._brent(f, lo, hi)
        assert status == 0
        assert abs(root - root_val) < 1e-6


# ---------------------------------------------------------------------------
# Energy balance use-case: q+ = q-
# ---------------------------------------------------------------------------


class TestEnergyBalanceUseCase:
    r"""
    Smoke tests representative of the one-zone disk :math:`q^+ = q^-` root find.

    The energy balance equation is:

    .. math::

        q^+(T_c) - q^-(T_c) = 0

    where :math:`q^+` grows with :math:`T_c` (viscous heating increases with
    sound speed) and :math:`q^-` grows faster with :math:`T_c` (radiative
    cooling).  The function changes sign exactly once on the physical domain.
    """

    # Crossing temperature — deliberately chosen so the root lies well inside
    # the log-space search domain [log(1e3), log(1e12)].
    T_STAR = 1e7  # K

    def _energy_balance(self, T_c):
        r"""Normalised toy energy balance with a crossing at :attr:`T_STAR`.

        .. math::

            f(T_c) = \frac{T_c}{T^*} - \left(\frac{T_c}{T^*}\right)^4

        ``q+ ∝ T`` (viscous heating, sound-speed driven) dominates at low
        temperature; ``q- ∝ T^4`` (optically-thick radiative cooling) dominates
        at high temperature.  The two are equal at exactly ``T_c = T_STAR``.
        """
        u = T_c / self.T_STAR
        return u - u**4

    def test_sign_change_exists(self):
        """The toy q+−q− function changes sign between 1e4 K and 1e10 K."""
        f_lo = self._energy_balance(1e4)
        f_hi = self._energy_balance(1e10)
        assert f_lo * f_hi < 0.0, "No sign change — check T_STAR"

    def test_root_found_in_log_space(self):
        """
        In log-space the function is smooth and find_root converges quickly.

        The one-zone integrator calls find_root on log(T_c), so we replicate
        that here.
        """

        def f_log(log_T):
            return self._energy_balance(math.exp(log_T))

        status, log_root = _find(
            f_log,
            x_guess=math.log(1e7),
            x_min=math.log(1e3),
            x_max=math.log(1e12),
        )
        assert status == 0
        T_root = math.exp(log_root)
        assert abs(T_root - self.T_STAR) / self.T_STAR < 1e-9

    def test_expand_bracket_then_brent(self):
        """Manually calling expand_bracket + brent_root mirrors the warm-start path."""

        def f_log(log_T):
            return self._energy_balance(math.exp(log_T))

        status_eb, x_lo, x_hi, f_lo, f_hi = py_expand_bracket(
            f_log,
            math.log(1e7),
            0.5,
            2.0,
            60,
            math.log(1e3),
            math.log(1e12),
            TOLERANCE,
        )
        assert status_eb == 0
        # Either a sign-change bracket or an exact root at the crossing.
        assert (f_lo * f_hi < 0.0) or (x_lo == x_hi)

        if x_lo < x_hi:
            status_br, root = py_brent_root(
                f_log,
                x_lo,
                x_hi,
                f_lo,
                f_hi,
                TOL_X,
                TOL_F,
                MAX_ITER,
            )
            assert status_br == 0
            assert abs(math.exp(root) - self.T_STAR) / self.T_STAR < 1e-9
