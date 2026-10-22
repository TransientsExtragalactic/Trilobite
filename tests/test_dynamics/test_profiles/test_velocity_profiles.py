"""
Testing suite for velocity profiles.

This module provides unit testing for the velocity profiles in
:py:mod:`trilobite.dynamics.profiles.velocity`. Tests are organized into:

(a) API functionality — ``eval``, ``feval``, ``as_callable``,
    ``as_optimized_callable``, ``process_arguments``, and
    ``process_parameters`` all behave as documented.

(b) Physical accuracy — profile values match analytic expectations and
    the characteristic invariances of each profile are verified.
"""

import numpy as np
import pytest
from astropy import units as u

from trilobite.dynamics.profiles.velocity import (
    ConstantVelocityProfile,
    HomologousVelocityProfile,
    StaticVelocityProfile,
)

# Standard test grids
_R_CGS = np.geomspace(1e14, 1e18, 50)
_R_QTY = _R_CGS * u.cm
_T_SCALAR = 10.0 * 24.0 * 3600.0  # 10 days in seconds
_T_QTY = _T_SCALAR * u.s


# =========================================================================== #
# Base Test Class                                                             #
# =========================================================================== #
class BaseTestVelocityProfile:
    """
    Base test class for velocity profiles.

    Subclasses must set ``PROFILE_CLASS`` and ``PARAMETER_SETS``.
    """

    PROFILE_CLASS = None
    PARAMETER_SETS: list[dict] = []

    # ------------------------------------------- #
    # API / Structural Tests                      #
    # ------------------------------------------- #

    def test_eval_returns_quantity(self):
        """eval returns a Quantity in cm s^{-1}."""
        for params in self.PARAMETER_SETS:
            result = self.PROFILE_CLASS.eval(_R_QTY, _T_QTY, **params)
            assert isinstance(result, u.Quantity)
            assert result.unit.is_equivalent(u.cm / u.s), (
                f"eval unit is not cm/s for {params}: got {result.unit}"
            )

    def test_eval_shape_matches_input(self):
        """eval output shape matches the radius array shape."""
        for params in self.PARAMETER_SETS:
            result = self.PROFILE_CLASS.eval(_R_QTY, _T_QTY, **params)
            assert result.shape == _R_QTY.shape

    def test_eval_scalar_input(self):
        """eval handles scalar radius and time inputs."""
        r_s = 1e16 * u.cm
        for params in self.PARAMETER_SETS:
            result = self.PROFILE_CLASS.eval(r_s, _T_QTY, **params)
            assert isinstance(result, u.Quantity)
            assert result.isscalar or result.shape == ()

    def test_feval_matches_eval_value(self):
        """feval with pre-processed inputs matches eval numerically."""
        for params in self.PARAMETER_SETS:
            processed_args = self.PROFILE_CLASS.process_arguments(_R_CGS, _T_SCALAR)
            processed_params = self.PROFILE_CLASS.process_parameters(**params)
            feval_result = self.PROFILE_CLASS.feval(*processed_args, **processed_params)
            eval_value = self.PROFILE_CLASS.eval(_R_CGS, _T_SCALAR, **params).value
            assert np.allclose(feval_result, eval_value)

    def test_as_callable_returns_quantity(self):
        """as_callable returns a callable that produces Quantity output."""
        for params in self.PARAMETER_SETS:
            fn = self.PROFILE_CLASS.as_callable(**params)
            result = fn(_R_QTY, _T_QTY)
            assert isinstance(result, u.Quantity)

    def test_as_callable_matches_eval(self):
        """as_callable output matches direct eval on the same grid."""
        for params in self.PARAMETER_SETS:
            fn = self.PROFILE_CLASS.as_callable(**params)
            assert np.allclose(
                fn(_R_QTY, _T_QTY).value,
                self.PROFILE_CLASS.eval(_R_QTY, _T_QTY, **params).value,
            )

    def test_as_optimized_callable_matches_feval(self):
        """as_optimized_callable output matches feval on the same grid."""
        for params in self.PARAMETER_SETS:
            fn = self.PROFILE_CLASS.as_optimized_callable(**params)
            processed_args = self.PROFILE_CLASS.process_arguments(_R_CGS, _T_SCALAR)
            processed_params = self.PROFILE_CLASS.process_parameters(**params)
            assert np.allclose(
                fn(*processed_args),
                self.PROFILE_CLASS.feval(*processed_args, **processed_params),
            )

    def test_process_arguments_returns_tuple(self):
        """process_arguments returns a tuple."""
        assert isinstance(self.PROFILE_CLASS.process_arguments(_R_CGS, _T_SCALAR), tuple)

    def test_process_parameters_returns_dict(self):
        """process_parameters returns a dict for each parameter set."""
        for params in self.PARAMETER_SETS:
            assert isinstance(self.PROFILE_CLASS.process_parameters(**params), dict)


# =========================================================================== #
# Concrete Velocity Profile Test Classes                                      #
# =========================================================================== #

class TestStaticVelocityProfile(BaseTestVelocityProfile):
    PROFILE_CLASS = StaticVelocityProfile
    PARAMETER_SETS = [{}]

    def test_velocity_is_zero_everywhere(self):
        """Static velocity is zero at every point on the test grid."""
        result = StaticVelocityProfile.eval(_R_QTY, _T_QTY)
        assert np.all(result.value == 0.0)

    def test_velocity_is_zero_at_scalar(self):
        """Static velocity is zero at a single point."""
        result = StaticVelocityProfile.eval(1e16 * u.cm, _T_QTY)
        assert result.to(u.cm / u.s).value == 0.0

    def test_velocity_independent_of_radius(self):
        """Static velocity is the same (zero) regardless of r."""
        r_vals = np.array([1e14, 1e15, 1e16, 1e17, 1e18]) * u.cm
        result = StaticVelocityProfile.eval(r_vals, _T_QTY).value
        assert np.all(result == result[0])

    def test_velocity_independent_of_time(self):
        """Static velocity is the same (zero) at different times."""
        r = 1e16 * u.cm
        for t in [1e5 * u.s, 1e7 * u.s, 1e10 * u.s]:
            result = StaticVelocityProfile.eval(r, t)
            assert result.to(u.cm / u.s).value == 0.0


class TestHomologousVelocityProfile(BaseTestVelocityProfile):
    PROFILE_CLASS = HomologousVelocityProfile
    PARAMETER_SETS = [{}]

    def test_exact_value(self):
        """eval(r, t) == r/t in cm/s."""
        result = HomologousVelocityProfile.eval(_R_QTY, _T_QTY).to(u.cm / u.s).value
        expected = _R_CGS / _T_SCALAR
        assert np.allclose(result, expected, rtol=1e-10)

    def test_linear_in_radius(self):
        """eval(2r, t) == 2 * eval(r, t)."""
        r1 = np.array([1e15, 1e16, 1e17]) * u.cm
        v1 = HomologousVelocityProfile.eval(r1, _T_QTY).value
        v2 = HomologousVelocityProfile.eval(2 * r1, _T_QTY).value
        assert np.allclose(v2 / v1, 2.0, rtol=1e-10)

    def test_inverse_in_time(self):
        """eval(r, 2t) == 0.5 * eval(r, t)."""
        r = np.array([1e15, 1e16, 1e17]) * u.cm
        v1 = HomologousVelocityProfile.eval(r, _T_QTY).value
        v2 = HomologousVelocityProfile.eval(r, 2 * _T_QTY).value
        assert np.allclose(v2 / v1, 0.5, rtol=1e-10)

    def test_scale_invariance(self):
        """eval(λr, λt) == eval(r, t) — homologous velocity is scale-free."""
        lam = 3.7
        r = np.array([1e15, 1e16, 1e17]) * u.cm
        v_orig = HomologousVelocityProfile.eval(r, _T_QTY).value
        v_scaled = HomologousVelocityProfile.eval(lam * r, lam * _T_QTY).value
        assert np.allclose(v_orig, v_scaled, rtol=1e-10)

    def test_unit_input_conversion(self):
        """eval correctly converts unit-bearing r and t inputs."""
        r = 1.0 * u.au        # 1 AU in cm
        t = 1.0 * u.yr        # 1 yr in s
        result = HomologousVelocityProfile.eval(r, t).to(u.cm / u.s).value
        expected = r.to(u.cm).value / t.to(u.s).value
        assert result == pytest.approx(expected, rel=1e-10)


class TestConstantVelocityProfile(BaseTestVelocityProfile):
    PROFILE_CLASS = ConstantVelocityProfile

    PARAMETER_SETS = [
        {"v_0": 1e9},
        {"v_0": 3e10},
        {"v_0": 1e8},
    ]

    def test_spatially_uniform(self):
        """eval returns the same velocity at every radius."""
        for params in self.PARAMETER_SETS:
            result = ConstantVelocityProfile.eval(_R_QTY, _T_QTY, **params).value
            assert np.all(result == result[0]), (
                f"Velocity is not spatially uniform for {params}"
            )

    def test_matches_parameter_cgs(self):
        """eval returns exactly v_0 (in CGS) at every grid point."""
        for params in self.PARAMETER_SETS:
            result = ConstantVelocityProfile.eval(_R_QTY, _T_QTY, **params).value
            assert np.allclose(result, params["v_0"], rtol=1e-10)

    def test_time_independent(self):
        """eval returns the same velocity at any time."""
        params = {"v_0": 1e9}
        r = 1e16 * u.cm
        for t in [1e5 * u.s, 1e8 * u.s, 1e12 * u.s]:
            result = ConstantVelocityProfile.eval(r, t, **params).to(u.cm / u.s).value
            assert result == pytest.approx(params["v_0"], rel=1e-10)

    def test_unit_aware_v0(self):
        """eval accepts a unit-bearing v_0 and converts correctly."""
        result = ConstantVelocityProfile.eval(
            _R_QTY, _T_QTY, v_0=100.0 * u.km / u.s
        ).to(u.cm / u.s).value
        assert np.allclose(result, 1e7, rtol=1e-10)
