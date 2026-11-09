"""
Testing suite for CSM density profiles.

This module provides unit testing for the CSM density profiles in
:py:mod:`trilobite.dynamics.profiles.csm`. Tests are organized into:

(a) API functionality — ``eval``, ``feval``, ``as_callable``,
    ``as_optimized_callable``, ``process_arguments``, and
    ``process_parameters`` all behave as documented.

(b) Physical accuracy — profile values match analytic formulae and
    inter-method consistency is verified by direct comparison.
"""

import numpy as np
import pytest
from astropy import units as u

from trilobite.dynamics.profiles.csm import (
    AxisymmetricCSMDensityProfile,
    BrokenPowerLawCSMProfile,
    CoredPowerLawCSMProfile,
    ExponentialCSMProfile,
    FiniteWindCSMProfile,
    GaussianShellCSMProfile,
    LogNormalCSMProfile,
    PowerLawCSMProfile,
    ShellCSMProfile,
    SmoothBPLCSMProfile,
    SmoothTruncatedWindCSMProfile,
    TruncatedWindCSMProfile,
    UniformCSMProfile,
    WindBubbleCSMProfile,
    WindCSMProfile,
    WindWithFloorCSMProfile,
)

# NumPy compatibility: np.trapezoid added in 2.0, np.trapz removed in 2.0.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

# Shared test grid in CGS (bare cm values → accepted as cm by all stationary profiles)
_R_CGS = np.geomspace(1e14, 1e18, 50)
_R_QTY = _R_CGS * u.cm
_T_QTY = 1e7 * u.s  # dummy time; stationary profiles ignore it


# =========================================================================== #
# Base Test Class                                                             #
# =========================================================================== #
class BaseTestCSMProfile:
    """
    Base test class for stationary CSM density profiles.

    Subclasses must set ``PROFILE_CLASS`` and ``PARAMETER_SETS``.
    """

    PROFILE_CLASS = None
    PARAMETER_SETS: list[dict] = []

    # ------------------------------------------- #
    # API / Structural Tests                      #
    # ------------------------------------------- #

    def test_eval_returns_quantity(self):
        """eval returns an astropy Quantity in g cm^{-3}."""
        for params in self.PARAMETER_SETS:
            result = self.PROFILE_CLASS.eval(_R_QTY, **params)
            assert isinstance(result, u.Quantity), f"eval did not return Quantity for {params}"
            assert result.unit.is_equivalent(u.g / u.cm**3), f"eval unit is not g/cm^3 for {params}: got {result.unit}"

    def test_eval_shape_matches_input(self):
        """eval output shape matches the radius array shape."""
        for params in self.PARAMETER_SETS:
            result = self.PROFILE_CLASS.eval(_R_QTY, **params)
            assert result.shape == _R_QTY.shape, (
                f"Shape mismatch for {params}: expected {_R_QTY.shape}, got {result.shape}"
            )

    def test_eval_scalar_input(self):
        """eval handles a scalar radius input."""
        r_scalar = 1e16 * u.cm
        for params in self.PARAMETER_SETS:
            result = self.PROFILE_CLASS.eval(r_scalar, **params)
            assert isinstance(result, u.Quantity)
            assert result.isscalar or result.shape == (), f"Scalar eval did not return scalar-like for {params}"

    def test_feval_matches_eval_value(self):
        """feval with pre-processed inputs matches eval numerically."""
        for params in self.PARAMETER_SETS:
            processed_args = self.PROFILE_CLASS.process_arguments(_R_CGS)
            processed_params = self.PROFILE_CLASS.process_parameters(**params)
            feval_result = self.PROFILE_CLASS.feval(*processed_args, **processed_params)
            eval_value = self.PROFILE_CLASS.eval(_R_CGS, **params).value
            assert np.allclose(feval_result, eval_value), f"feval does not match eval for {params}"

    def test_as_callable_returns_quantity(self):
        """as_callable returns a callable that produces Quantity output."""
        for params in self.PARAMETER_SETS:
            fn = self.PROFILE_CLASS.as_callable(**params)
            result = fn(_R_QTY)
            assert isinstance(result, u.Quantity), f"as_callable did not return Quantity for {params}"

    def test_as_callable_matches_eval(self):
        """as_callable output matches direct eval on the same grid."""
        for params in self.PARAMETER_SETS:
            fn = self.PROFILE_CLASS.as_callable(**params)
            assert np.allclose(
                fn(_R_QTY).value,
                self.PROFILE_CLASS.eval(_R_QTY, **params).value,
            ), f"as_callable does not match eval for {params}"

    def test_as_optimized_callable_matches_feval(self):
        """as_optimized_callable output matches feval on the same grid."""
        for params in self.PARAMETER_SETS:
            fn = self.PROFILE_CLASS.as_optimized_callable(**params)
            processed_args = self.PROFILE_CLASS.process_arguments(_R_CGS)
            processed_params = self.PROFILE_CLASS.process_parameters(**params)
            assert np.allclose(
                fn(*processed_args),
                self.PROFILE_CLASS.feval(*processed_args, **processed_params),
            ), f"as_optimized_callable does not match feval for {params}"

    def test_process_arguments_returns_tuple(self):
        """process_arguments returns a tuple."""
        assert isinstance(self.PROFILE_CLASS.process_arguments(_R_QTY), tuple)

    def test_process_parameters_returns_dict(self):
        """process_parameters returns a dict for each parameter set."""
        for params in self.PARAMETER_SETS:
            result = self.PROFILE_CLASS.process_parameters(**params)
            assert isinstance(result, dict), f"process_parameters did not return dict for {params}"

    def test_output_is_nonnegative(self):
        """Density is non-negative everywhere on the test grid."""
        for params in self.PARAMETER_SETS:
            result = self.PROFILE_CLASS.eval(_R_QTY, **params)
            assert np.all(result.value >= 0), f"Negative density found for {params}"

    def test_time_argument_is_ignored(self):
        """Stationary profiles return the same density with or without a time argument."""
        for params in self.PARAMETER_SETS:
            rho_no_t = self.PROFILE_CLASS.eval(_R_QTY, **params)
            rho_with_t = self.PROFILE_CLASS.eval(_R_QTY, _T_QTY, **params)
            assert np.allclose(rho_no_t.value, rho_with_t.value), f"Profile is not time-independent for {params}"


# =========================================================================== #
# Concrete CSM Profile Test Classes                                           #
# =========================================================================== #


class TestWindCSMProfile(BaseTestCSMProfile):
    PROFILE_CLASS = WindCSMProfile

    # Exercise all three parameterization routes
    _MDOT_CGS = 1e-5 * 1.989e33 / 3.156e7  # ~1e-5 Msun/yr in g/s
    _VW_CGS = 10.0e5  # 10 km/s in cm/s

    PARAMETER_SETS = [
        {"A": 5e11},
        {"A_star": 1.0},
        {"A_star": 0.1},
        {"mass_loss_rate": _MDOT_CGS, "wind_velocity": _VW_CGS},
    ]

    def test_inverse_square_law(self):
        """rho(r) == A * r^{-2}."""
        A = 5e11
        result = WindCSMProfile.eval(_R_QTY, A=A).value
        assert np.allclose(result, A * _R_CGS**-2, rtol=1e-10)

    def test_doubling_radius_halves_density_squared(self):
        """rho(2r) / rho(r) == 1/4 for all r."""
        r1 = np.array([1e15, 1e16, 1e17]) * u.cm
        rho1 = WindCSMProfile.eval(r1, A=5e11).value
        rho2 = WindCSMProfile.eval(2 * r1, A=5e11).value
        assert np.allclose(rho2 / rho1, 0.25, rtol=1e-10)

    def test_normalization_routes_agree(self):
        """All three parameterization routes produce identical densities."""
        A_cgs = self._MDOT_CGS / (4 * np.pi * self._VW_CGS)
        A_star = A_cgs / 5e11

        rho1 = WindCSMProfile.eval(_R_QTY, A=A_cgs).value
        rho2 = WindCSMProfile.eval(_R_QTY, A_star=A_star).value
        rho3 = WindCSMProfile.eval(_R_QTY, mass_loss_rate=self._MDOT_CGS, wind_velocity=self._VW_CGS).value

        assert np.allclose(rho1, rho2, rtol=1e-10), "Routes 1 and 2 disagree"
        assert np.allclose(rho1, rho3, rtol=1e-10), "Routes 1 and 3 disagree"

    def test_normalize_formula(self):
        """normalize(Mdot, vw) == Mdot / (4 pi vw)."""
        A = WindCSMProfile.normalize(self._MDOT_CGS, self._VW_CGS)
        expected = self._MDOT_CGS / (4 * np.pi * self._VW_CGS)
        assert A.to(u.g / u.cm).value == pytest.approx(expected, rel=1e-10)

    def test_normalize_A_star_formula(self):
        """normalize_A_star returns A / (5e11 g/cm)."""
        A_star = WindCSMProfile.normalize_A_star(self._MDOT_CGS, self._VW_CGS)
        A_cgs = self._MDOT_CGS / (4 * np.pi * self._VW_CGS)
        assert A_star == pytest.approx(A_cgs / 5e11, rel=1e-10)

    def test_normalize_from_A_star(self):
        """normalize_from_A_star(A_star) == A_star * 5e11 g/cm."""
        for A_star in [0.1, 1.0, 10.0]:
            A = WindCSMProfile.normalize_from_A_star(A_star)
            assert A.to(u.g / u.cm).value == pytest.approx(A_star * 5e11, rel=1e-10)

    def test_enclosed_mass_formula(self):
        """enclosed_mass(r) == 4 pi A r."""
        A = 5e11
        r_vals = np.array([1e15, 1e16, 1e17])
        M = WindCSMProfile.enclosed_mass(r_vals * u.cm, A=A)
        assert np.allclose(M.to(u.g).value, 4 * np.pi * A * r_vals, rtol=1e-10)

    def test_shell_mass_formula(self):
        """shell_mass(r1, r2) == 4 pi A (r2 - r1)."""
        A, r_in, r_out = 5e11, 1e15, 1e16
        M = WindCSMProfile.shell_mass(r_in * u.cm, r_out * u.cm, A=A)
        assert M.to(u.g).value == pytest.approx(4 * np.pi * A * (r_out - r_in), rel=1e-10)

    def test_column_density_formula(self):
        """column_density(r) == A / r."""
        A = 5e11
        r_vals = np.array([1e15, 1e16, 1e17])
        Sigma = WindCSMProfile.column_density(r_vals * u.cm, A=A)
        assert np.allclose(Sigma.to(u.g / u.cm**2).value, A / r_vals, rtol=1e-10)

    def test_missing_route_raises(self):
        """Providing no parameterization route raises ValueError."""
        with pytest.raises(ValueError):
            WindCSMProfile.eval(_R_QTY)

    def test_multiple_routes_raises(self):
        """Providing two parameterization routes simultaneously raises ValueError."""
        with pytest.raises(ValueError):
            WindCSMProfile.eval(_R_QTY, A=5e11, A_star=1.0)


class TestUniformCSMProfile(BaseTestCSMProfile):
    PROFILE_CLASS = UniformCSMProfile

    PARAMETER_SETS = [
        {"rho_0": 1e-24},
        {"rho_0": 1e-20},
        {"rho_0": 0.0},
    ]

    def test_spatially_constant(self):
        """eval returns rho_0 at every radius."""
        for params in self.PARAMETER_SETS:
            result = UniformCSMProfile.eval(_R_QTY, **params).value
            assert np.allclose(result, params["rho_0"], atol=1e-40), f"Density is not constant for {params}"


class TestPowerLawCSMProfile(BaseTestCSMProfile):
    PROFILE_CLASS = PowerLawCSMProfile

    PARAMETER_SETS = [
        {"rho_ref": 1e-18, "r_ref": 1e16, "slope": 2.0},
        {"rho_ref": 1e-20, "r_ref": 1e15, "slope": 0.0},
        {"rho_ref": 5e-19, "r_ref": 1e17, "slope": 1.5},
    ]

    def test_density_at_reference_radius(self):
        """eval(r_ref) == rho_ref."""
        for params in self.PARAMETER_SETS:
            result = PowerLawCSMProfile.eval(params["r_ref"] * u.cm, **params).value
            assert result == pytest.approx(params["rho_ref"], rel=1e-10)

    def test_power_law_slope(self):
        """rho(2 r_ref) / rho(r_ref) == 2^{-slope}."""
        for params in self.PARAMETER_SETS:
            r = params["r_ref"] * u.cm
            rho_r = PowerLawCSMProfile.eval(r, **params).value
            rho_2r = PowerLawCSMProfile.eval(2 * r, **params).value
            assert rho_2r / rho_r == pytest.approx(2.0 ** (-params["slope"]), rel=1e-10)


class TestShellCSMProfile(BaseTestCSMProfile):
    PROFILE_CLASS = ShellCSMProfile

    PARAMETER_SETS = [
        {"r_inner": 3e15, "r_outer": 1e16, "shell_density": 1e-18, "density_floor": 1e-23},
        {"r_inner": 1e15, "r_outer": 5e15, "shell_density": 5e-19, "density_floor": 0.0},
    ]

    def test_density_inside_shell(self):
        """Density at shell midpoint equals shell_density."""
        for params in self.PARAMETER_SETS:
            r_mid = 0.5 * (params["r_inner"] + params["r_outer"]) * u.cm
            result = ShellCSMProfile.eval(r_mid, **params).value
            assert result == pytest.approx(params["shell_density"], rel=1e-10)

    def test_floor_density_outside(self):
        """Density far outside the shell equals density_floor."""
        for params in self.PARAMETER_SETS:
            r_far = params["r_outer"] * 10 * u.cm
            result = ShellCSMProfile.eval(r_far, **params).value
            assert result == pytest.approx(params["density_floor"], abs=1e-35)

    def test_floor_density_below_shell(self):
        """Density well inside r_inner equals density_floor."""
        for params in self.PARAMETER_SETS:
            if params["r_inner"] > 1e12:
                r_small = params["r_inner"] * 0.01 * u.cm
                result = ShellCSMProfile.eval(r_small, **params).value
                assert result == pytest.approx(params["density_floor"], abs=1e-35)


class TestGaussianShellCSMProfile(BaseTestCSMProfile):
    PROFILE_CLASS = GaussianShellCSMProfile

    PARAMETER_SETS = [
        {
            "background_density": 1e-23,
            "shell_density": 5e-19,
            "shell_radius": 1e16,
            "shell_width": 2e15,
        },
        {
            "background_density": 0.0,
            "shell_density": 1e-18,
            "shell_radius": 5e16,
            "shell_width": 5e15,
        },
    ]

    def test_peak_at_shell_radius(self):
        """eval(shell_radius) == background_density + shell_density."""
        for params in self.PARAMETER_SETS:
            result = GaussianShellCSMProfile.eval(params["shell_radius"] * u.cm, **params).value
            expected = params["background_density"] + params["shell_density"]
            assert result == pytest.approx(expected, rel=1e-10)

    def test_one_sigma_value(self):
        """eval at shell_radius + shell_width == bg + shell * exp(-0.5)."""
        for params in self.PARAMETER_SETS:
            r_sig = (params["shell_radius"] + params["shell_width"]) * u.cm
            result = GaussianShellCSMProfile.eval(r_sig, **params).value
            expected = params["background_density"] + params["shell_density"] * np.exp(-0.5)
            assert result == pytest.approx(expected, rel=1e-10)

    def test_approaches_background_far_away(self):
        """Far from the shell, density approaches background_density."""
        for params in self.PARAMETER_SETS:
            r_far = params["shell_radius"] * 100 * u.cm
            result = GaussianShellCSMProfile.eval(r_far, **params).value
            expected = params["background_density"]
            assert result == pytest.approx(expected, rel=1e-2, abs=1e-40)


class TestBrokenPowerLawCSMProfile(BaseTestCSMProfile):
    PROFILE_CLASS = BrokenPowerLawCSMProfile

    PARAMETER_SETS = [
        {"density_break": 1e-18, "radius_break": 1e16, "slope_inner": 2.0, "slope_outer": 0.0},
        {"density_break": 5e-19, "radius_break": 5e15, "slope_inner": 1.0, "slope_outer": 2.5},
    ]

    def test_density_at_break_radius(self):
        """eval(r_b) == density_break."""
        for params in self.PARAMETER_SETS:
            result = BrokenPowerLawCSMProfile.eval(params["radius_break"] * u.cm, **params).value
            assert result == pytest.approx(params["density_break"], rel=1e-10)

    def test_inner_slope(self):
        """For r < r_b, density scales as (r1/r2)^{-slope_inner}."""
        for params in self.PARAMETER_SETS:
            r_b = params["radius_break"]
            r1 = r_b * 0.1 * u.cm
            r2 = r_b * 0.5 * u.cm
            rho1 = BrokenPowerLawCSMProfile.eval(r1, **params).value
            rho2 = BrokenPowerLawCSMProfile.eval(r2, **params).value
            expected = (0.1 / 0.5) ** (-params["slope_inner"])
            assert rho1 / rho2 == pytest.approx(expected, rel=1e-10)

    def test_outer_slope(self):
        """For r > r_b, density scales as (r1/r2)^{-slope_outer}."""
        for params in self.PARAMETER_SETS:
            r_b = params["radius_break"]
            r1 = r_b * 2.0 * u.cm
            r2 = r_b * 10.0 * u.cm
            rho1 = BrokenPowerLawCSMProfile.eval(r1, **params).value
            rho2 = BrokenPowerLawCSMProfile.eval(r2, **params).value
            expected = (2.0 / 10.0) ** (-params["slope_outer"])
            assert rho1 / rho2 == pytest.approx(expected, rel=1e-10)

    def test_continuity_at_break(self):
        """Profile is continuous at r_b: inner and outer limits both equal density_break."""
        for params in self.PARAMETER_SETS:
            r_b = params["radius_break"]
            eps = r_b * 1e-6
            rho_in = BrokenPowerLawCSMProfile.eval((r_b - eps) * u.cm, **params).value
            rho_out = BrokenPowerLawCSMProfile.eval((r_b + eps) * u.cm, **params).value
            assert rho_in == pytest.approx(params["density_break"], rel=1e-4)
            assert rho_out == pytest.approx(params["density_break"], rel=1e-4)


class TestTruncatedWindCSMProfile(BaseTestCSMProfile):
    PROFILE_CLASS = TruncatedWindCSMProfile

    PARAMETER_SETS = [
        {"A": 5e11, "r_max": 3e16, "density_floor": 1e-23},
        {"A": 1e12, "r_max": 1e17, "density_floor": 0.0},
    ]

    def test_wind_profile_inside_rmax(self):
        """For r <= r_max, density == A * r^{-2}."""
        for params in self.PARAMETER_SETS:
            r_inner = np.geomspace(1e14, params["r_max"] * 0.9, 20) * u.cm
            result = TruncatedWindCSMProfile.eval(r_inner, **params).value
            expected = params["A"] * r_inner.value**-2
            assert np.allclose(result, expected, rtol=1e-10)

    def test_floor_density_outside_rmax(self):
        """For r > r_max, density == density_floor."""
        for params in self.PARAMETER_SETS:
            r_outer = np.geomspace(params["r_max"] * 1.1, 1e18, 10) * u.cm
            result = TruncatedWindCSMProfile.eval(r_outer, **params).value
            assert np.allclose(result, params["density_floor"], atol=1e-35)


class TestWindWithFloorCSMProfile(BaseTestCSMProfile):
    PROFILE_CLASS = WindWithFloorCSMProfile

    PARAMETER_SETS = [
        {"A": 5e11, "density_floor": 1e-23},
        {"A": 1e12, "density_floor": 0.0},
    ]

    def test_exact_formula(self):
        """eval(r) == A * r^{-2} + density_floor everywhere."""
        for params in self.PARAMETER_SETS:
            result = WindWithFloorCSMProfile.eval(_R_QTY, **params).value
            expected = params["A"] * _R_CGS**-2 + params["density_floor"]
            assert np.allclose(result, expected, rtol=1e-10)


class TestSmoothTruncatedWindCSMProfile(BaseTestCSMProfile):
    PROFILE_CLASS = SmoothTruncatedWindCSMProfile

    PARAMETER_SETS = [
        {"A": 5e11, "r_max": 3e16, "transition_width": 3e15, "density_floor": 1e-23},
        {"A": 1e12, "r_max": 1e17, "transition_width": 1e16, "density_floor": 0.0},
    ]

    def test_midpoint_value(self):
        """At r == r_max, tanh switch == 0 → density == (A*r_max^{-2} + floor)/2."""
        for params in self.PARAMETER_SETS:
            r_m = params["r_max"] * u.cm
            result = SmoothTruncatedWindCSMProfile.eval(r_m, **params).value
            wind_at_rmax = params["A"] * params["r_max"] ** -2
            expected = 0.5 * (wind_at_rmax + params["density_floor"])
            assert result == pytest.approx(expected, rel=1e-10)

    def test_approaches_floor_far_outside(self):
        """Far outside r_max, density approaches density_floor."""
        for params in self.PARAMETER_SETS:
            r_far = params["r_max"] * 1000 * u.cm
            result = SmoothTruncatedWindCSMProfile.eval(r_far, **params).value
            assert result == pytest.approx(params["density_floor"], rel=1e-3, abs=1e-40)


class TestExponentialCSMProfile(BaseTestCSMProfile):
    PROFILE_CLASS = ExponentialCSMProfile

    PARAMETER_SETS = [
        {"rho_0": 1e-18, "r_0": 0.0, "scale_height": 1e16, "density_floor": 1e-23},
        {"rho_0": 5e-19, "r_0": 1e15, "scale_height": 5e15, "density_floor": 0.0},
    ]

    def test_value_at_r0(self):
        """eval(r_0) == rho_0 + density_floor."""
        for params in self.PARAMETER_SETS:
            r0 = params["r_0"] * u.cm
            result = ExponentialCSMProfile.eval(r0, **params).value
            assert result == pytest.approx(params["rho_0"] + params["density_floor"], rel=1e-10)

    def test_scale_height_decay(self):
        """eval(r_0 + scale_height) == rho_0 * exp(-1) + density_floor."""
        for params in self.PARAMETER_SETS:
            r_H = (params["r_0"] + params["scale_height"]) * u.cm
            result = ExponentialCSMProfile.eval(r_H, **params).value
            expected = params["rho_0"] * np.exp(-1.0) + params["density_floor"]
            assert result == pytest.approx(expected, rel=1e-10)


class TestSmoothBPLCSMProfile(BaseTestCSMProfile):
    PROFILE_CLASS = SmoothBPLCSMProfile

    PARAMETER_SETS = [
        {
            "density_break": 1e-18,
            "radius_break": 1e16,
            "slope_inner": 2.0,
            "slope_outer": 0.0,
            "smoothness": 0.5,
        },
        {
            "density_break": 5e-19,
            "radius_break": 5e15,
            "slope_inner": 1.0,
            "slope_outer": 2.0,
            "smoothness": 0.1,
        },
    ]

    def test_density_at_break_radius(self):
        """eval(r_b) == density_break exactly."""
        for params in self.PARAMETER_SETS:
            result = SmoothBPLCSMProfile.eval(params["radius_break"] * u.cm, **params).value
            assert result == pytest.approx(params["density_break"], rel=1e-8)

    def test_asymptotic_inner_slope(self):
        """For r << r_b, effective log slope approaches slope_inner."""
        for params in self.PARAMETER_SETS:
            r_b = params["radius_break"]
            # Use r1 = 1e-4 * r_b so that (r/r_b)^{1/Delta} << 1
            r1 = r_b * 1e-4 * u.cm
            r2 = r_b * 2e-4 * u.cm
            rho1 = SmoothBPLCSMProfile.eval(r1, **params).value
            rho2 = SmoothBPLCSMProfile.eval(r2, **params).value
            measured = -np.log(rho2 / rho1) / np.log(2.0)
            assert measured == pytest.approx(params["slope_inner"], abs=0.01)

    def test_asymptotic_outer_slope(self):
        """For r >> r_b, effective log slope approaches slope_outer."""
        for params in self.PARAMETER_SETS:
            r_b = params["radius_break"]
            r1 = r_b * 1e4 * u.cm
            r2 = r_b * 2e4 * u.cm
            rho1 = SmoothBPLCSMProfile.eval(r1, **params).value
            rho2 = SmoothBPLCSMProfile.eval(r2, **params).value
            measured = -np.log(rho2 / rho1) / np.log(2.0)
            assert measured == pytest.approx(params["slope_outer"], abs=0.01)


class TestCoredPowerLawCSMProfile(BaseTestCSMProfile):
    PROFILE_CLASS = CoredPowerLawCSMProfile

    PARAMETER_SETS = [
        {"rho_0": 1e-18, "r_core": 1e15, "slope": 2.0},
        {"rho_0": 5e-19, "r_core": 1e16, "slope": 1.5},
    ]

    def test_flat_inner_core(self):
        """For r << r_core, density approaches rho_0."""
        for params in self.PARAMETER_SETS:
            r_tiny = params["r_core"] * 1e-5 * u.cm
            result = CoredPowerLawCSMProfile.eval(r_tiny, **params).value
            assert result == pytest.approx(params["rho_0"], rel=1e-4)

    def test_outer_power_law_slope(self):
        """For r >> r_core, rho(r1)/rho(r2) == (r1/r2)^{-slope}."""
        for params in self.PARAMETER_SETS:
            r_c = params["r_core"]
            r1 = r_c * 1e3 * u.cm
            r2 = r_c * 2e3 * u.cm
            rho1 = CoredPowerLawCSMProfile.eval(r1, **params).value
            rho2 = CoredPowerLawCSMProfile.eval(r2, **params).value
            expected = (r1.value / r2.value) ** (-params["slope"])
            assert rho1 / rho2 == pytest.approx(expected, rel=1e-3)


class TestFiniteWindCSMProfile(BaseTestCSMProfile):
    PROFILE_CLASS = FiniteWindCSMProfile

    PARAMETER_SETS = [
        {"A": 5e11, "r_min": 1e15, "r_max": 3e16, "density_floor": 1e-23},
        {"A": 1e12, "r_min": 0.0, "r_max": 1e17, "density_floor": 0.0},
    ]

    def test_wind_inside_window(self):
        """Inside [r_min, r_max], density == A * r^{-2}."""
        for params in self.PARAMETER_SETS:
            r_start = max(params["r_min"] * 1.01 + 1e12, 1e14)
            r_end = params["r_max"] * 0.9
            r_test = np.geomspace(r_start, r_end, 10) * u.cm
            result = FiniteWindCSMProfile.eval(r_test, **params).value
            expected = params["A"] * r_test.value**-2
            assert np.allclose(result, expected, rtol=1e-10)

    def test_floor_outside_window(self):
        """Outside [r_min, r_max], density == density_floor."""
        for params in self.PARAMETER_SETS:
            r_far = params["r_max"] * 10 * u.cm
            result = FiniteWindCSMProfile.eval(r_far, **params).value
            assert result == pytest.approx(params["density_floor"], abs=1e-35)

    def test_normalize_from_timing(self):
        """normalize_from_timing returns r = v_w * t boundaries."""
        v_w = 1e8  # cm/s
        t0, t1 = 1e10, 3e10  # s
        r_min, r_max = FiniteWindCSMProfile.normalize_from_timing(v_w, t0, t1)
        assert r_min.to(u.cm).value == pytest.approx(v_w * t0, rel=1e-10)
        assert r_max.to(u.cm).value == pytest.approx(v_w * t1, rel=1e-10)


class TestWindBubbleCSMProfile(BaseTestCSMProfile):
    PROFILE_CLASS = WindBubbleCSMProfile

    PARAMETER_SETS = [
        {
            "A": 5e11,
            "r_wind_termination": 3e16,
            "rho_bubble": 1e-25,
            "r_shell_inner": 3e17,
            "r_shell_outer": 4e17,
            "rho_shell": 1e-21,
            "rho_ism": 1e-24,
        },
    ]

    def _p(self):
        return self.PARAMETER_SETS[0]

    def test_wind_zone(self):
        """r <= r_ts: density == A * r^{-2}."""
        p = self._p()
        r_test = np.geomspace(1e14, p["r_wind_termination"] * 0.9, 10) * u.cm
        result = WindBubbleCSMProfile.eval(r_test, **p).value
        assert np.allclose(result, p["A"] * r_test.value**-2, rtol=1e-10)

    def test_bubble_zone(self):
        """r_ts < r <= r_sh_in: density == rho_bubble."""
        p = self._p()
        r_mid = 0.5 * (p["r_wind_termination"] + p["r_shell_inner"]) * u.cm
        result = WindBubbleCSMProfile.eval(r_mid, **p).value
        assert result == pytest.approx(p["rho_bubble"], rel=1e-10)

    def test_shell_zone(self):
        """r_sh_in < r <= r_sh_out: density == rho_shell."""
        p = self._p()
        r_mid = 0.5 * (p["r_shell_inner"] + p["r_shell_outer"]) * u.cm
        result = WindBubbleCSMProfile.eval(r_mid, **p).value
        assert result == pytest.approx(p["rho_shell"], rel=1e-10)

    def test_ism_zone(self):
        """r > r_sh_out: density == rho_ism."""
        p = self._p()
        r_far = p["r_shell_outer"] * 10 * u.cm
        result = WindBubbleCSMProfile.eval(r_far, **p).value
        assert result == pytest.approx(p["rho_ism"], rel=1e-10)


class TestLogNormalCSMProfile(BaseTestCSMProfile):
    PROFILE_CLASS = LogNormalCSMProfile

    PARAMETER_SETS = [
        {"rho_0": 1e-18, "r_peak": 1e16, "sigma": 0.5},
        {"rho_0": 5e-19, "r_peak": 5e15, "sigma": 1.0},
    ]

    def test_peak_at_r_peak(self):
        """eval(r_peak) == rho_0."""
        for params in self.PARAMETER_SETS:
            result = LogNormalCSMProfile.eval(params["r_peak"] * u.cm, **params).value
            assert result == pytest.approx(params["rho_0"], rel=1e-10)

    def test_log_symmetry(self):
        """Profile is symmetric in log(r/r_peak): rho at +sigma and -sigma are equal."""
        for params in self.PARAMETER_SETS:
            r_p, sig = params["r_peak"], params["sigma"]
            rho_plus = LogNormalCSMProfile.eval(r_p * np.exp(sig) * u.cm, **params).value
            rho_minus = LogNormalCSMProfile.eval(r_p * np.exp(-sig) * u.cm, **params).value
            assert rho_plus == pytest.approx(rho_minus, rel=1e-10)

    def test_one_sigma_value(self):
        """eval at r_peak * exp(±sigma) == rho_0 * exp(-0.5)."""
        for params in self.PARAMETER_SETS:
            r_sig = params["r_peak"] * np.exp(params["sigma"]) * u.cm
            result = LogNormalCSMProfile.eval(r_sig, **params).value
            assert result == pytest.approx(params["rho_0"] * np.exp(-0.5), rel=1e-10)


class TestAxisymmetricCSMDensityProfile(BaseTestCSMProfile):
    PROFILE_CLASS = AxisymmetricCSMDensityProfile

    _BASE = {
        "rho_int": 2e-16,
        "rho_ext": 1e-18,
        "r_ref": 1e15,
        "alpha_int": 1.5,
        "alpha_ext": 2.0,
        "r_trans": 5e15,
        "delta_trans": 0.3,
        "f_pole": 0.2,
        "aspect": 2.0,
    }
    PARAMETER_SETS = [
        {**_BASE, "theta": np.pi / 2},
        {**_BASE, "theta": 0.0},
        {**_BASE, "theta": 0.7, "aspect": 0.5, "f_pole": 0.0},
    ]

    @staticmethod
    def _sphere_average(func, n=4001):
        """Solid-angle average (1/2) * integral f(theta) sin(theta) dtheta, by the trapezoid rule."""
        theta = np.linspace(0.0, np.pi, n)
        return 0.5 * _trapz(func(theta) * np.sin(theta), theta)

    # ------------------------------------------- #
    # Angular structure                           #
    # ------------------------------------------- #
    @pytest.mark.parametrize("f_pole", [0.0, 0.2, 0.7, 1.0])
    @pytest.mark.parametrize("aspect", [0.1, 1.0, 5.0, 50.0])
    def test_angular_shape_has_unit_solid_angle_average(self, f_pole, aspect):
        avg = self._sphere_average(lambda th: AxisymmetricCSMDensityProfile.angular_shape(th, f_pole, aspect))
        assert avg == pytest.approx(1.0, rel=1e-4)

    def test_angular_norm_matches_direct_integration(self):
        f, b = 0.2, 2.0
        direct = self._sphere_average(lambda th: f + (1 - f) * np.exp(-np.abs(np.cos(th) / np.sin(th + 1e-300)) / b))
        assert AxisymmetricCSMDensityProfile.angular_norm(f, b) == pytest.approx(direct, rel=1e-4)

    def test_interior_component_averages_to_rho_int(self):
        """With rho_ext = 0, the sphere-averaged density equals rho_int(r) S(r)."""
        params = {**self._BASE, "rho_ext": 0.0}
        for r in (1e14, 5e15, 3e16):
            avg = self._sphere_average(lambda th: AxisymmetricCSMDensityProfile.eval(r, theta=th, **params).value)
            x = r / params["r_ref"]
            S = 1.0 / (1.0 + (r / params["r_trans"]) ** (1.0 / params["delta_trans"]))
            assert avg == pytest.approx(params["rho_int"] * x ** (-params["alpha_int"]) * S, rel=1e-4)

    def test_equator_denser_than_pole(self):
        eq = AxisymmetricCSMDensityProfile.eval(_R_QTY, theta=np.pi / 2, **self._BASE)
        pole = AxisymmetricCSMDensityProfile.eval(_R_QTY, theta=0.0, **self._BASE)
        assert np.all(eq.value >= pole.value)

    def test_pole_value_is_floor(self):
        """On the axis G = f_pole / norm exactly."""
        p = self._BASE
        norm = AxisymmetricCSMDensityProfile.angular_norm(p["f_pole"], p["aspect"])
        assert AxisymmetricCSMDensityProfile.angular_shape(0.0, p["f_pole"], p["aspect"]) == pytest.approx(
            p["f_pole"] / norm, rel=1e-12
        )

    def test_reflection_symmetry(self):
        th = np.array([0.1, 0.7, 1.3])
        a = AxisymmetricCSMDensityProfile.eval(1e16, theta=th, **self._BASE).value
        b = AxisymmetricCSMDensityProfile.eval(1e16, theta=np.pi - th, **self._BASE).value
        c = AxisymmetricCSMDensityProfile.eval(1e16, theta=-th, **self._BASE).value
        assert np.allclose(a, b, rtol=1e-12)
        assert np.allclose(a, c, rtol=1e-12)

    def test_angle_units(self):
        deg = AxisymmetricCSMDensityProfile.eval(1e16, theta=90 * u.deg, **self._BASE).value
        rad = AxisymmetricCSMDensityProfile.eval(1e16, theta=np.pi / 2, **self._BASE).value
        assert deg == pytest.approx(rad, rel=1e-12)

    def test_theta_broadcasts_against_radius(self):
        r = np.geomspace(1e14, 1e17, 7)[:, None]
        th = np.linspace(0.0, np.pi / 2, 5)[None, :]
        rho = AxisymmetricCSMDensityProfile.eval(r, theta=th, **self._BASE)
        assert rho.shape == (7, 5)

    # ------------------------------------------- #
    # Radial structure and limits                 #
    # ------------------------------------------- #
    def test_density_at_reference_radius_exterior_only(self):
        params = {**self._BASE, "rho_int": 0.0}
        rho = AxisymmetricCSMDensityProfile.eval(params["r_ref"], theta=1.0, **params)
        assert rho.value == pytest.approx(params["rho_ext"], rel=1e-12)

    def test_zero_interior_reduces_to_spherical_power_law(self):
        params = {**self._BASE, "rho_int": 0.0}
        rho = AxisymmetricCSMDensityProfile.eval(_R_QTY, theta=0.3, **params).value
        expected = PowerLawCSMProfile.eval(
            _R_QTY, rho_ref=params["rho_ext"], r_ref=params["r_ref"], slope=params["alpha_ext"]
        ).value
        assert np.allclose(rho, expected, rtol=1e-12)

    def test_unit_f_pole_is_spherical(self):
        """With f_pole = 1 the angular dependence vanishes."""
        params = {**self._BASE, "f_pole": 1.0}
        eq = AxisymmetricCSMDensityProfile.eval(_R_QTY, theta=np.pi / 2, **params).value
        pole = AxisymmetricCSMDensityProfile.eval(_R_QTY, theta=0.0, **params).value
        assert np.allclose(eq, pole, rtol=1e-12)

    def test_transition_half_at_r_trans(self):
        params = {**self._BASE, "rho_ext": 0.0}
        theta = np.pi / 2
        rho = AxisymmetricCSMDensityProfile.eval(params["r_trans"], theta=theta, **params).value
        G = AxisymmetricCSMDensityProfile.angular_shape(theta, params["f_pole"], params["aspect"])
        expected = 0.5 * params["rho_int"] * (params["r_trans"] / params["r_ref"]) ** (-params["alpha_int"]) * G
        assert rho == pytest.approx(expected, rel=1e-12)

    def test_sharp_transition_does_not_overflow(self):
        params = {**self._BASE, "delta_trans": 1e-3}
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            rho = AxisymmetricCSMDensityProfile.eval(_R_QTY, theta=np.pi / 2, **params).value
        assert np.all(np.isfinite(rho))
        # Far beyond the transition only the exterior component remains.
        assert rho[-1] == pytest.approx(params["rho_ext"] * (_R_CGS[-1] / params["r_ref"]) ** -2.0, rel=1e-12)

    def test_sightline_callable_matches_eval(self):
        fn = AxisymmetricCSMDensityProfile.as_optimized_callable(theta=0.4, **self._BASE)
        expected = AxisymmetricCSMDensityProfile.eval(_R_CGS, theta=0.4, **self._BASE).value
        assert np.allclose(fn(_R_CGS), expected, rtol=1e-12)

    # ------------------------------------------- #
    # Sector fractions                            #
    # ------------------------------------------- #
    @pytest.mark.parametrize("aspect", [0.2, 1.0, 2.0, 10.0])
    def test_sector_fractions_sum_to_one(self, aspect):
        eq = AxisymmetricCSMDensityProfile.equatorial_solid_angle_fraction(aspect)
        pol = AxisymmetricCSMDensityProfile.polar_solid_angle_fraction(aspect)
        assert eq + pol == pytest.approx(1.0, rel=1e-14)
        assert 0 < eq < 1

    @pytest.mark.parametrize("aspect", [0.2, 1.0, 2.0, 10.0])
    @pytest.mark.parametrize("f_pole", [0.0, 0.3])
    def test_disk_boundary_is_half_maximum(self, aspect, f_pole):
        """At the disk half-opening latitude G sits halfway between its polar floor and equatorial peak."""
        lat = AxisymmetricCSMDensityProfile.disk_half_opening_angle(aspect).to_value(u.rad)
        theta = np.pi / 2 - lat
        raw = AxisymmetricCSMDensityProfile._unnormalized_angular_shape
        mid = 0.5 * (raw(0.0, f_pole, aspect) + raw(np.pi / 2, f_pole, aspect))
        assert raw(theta, f_pole, aspect) == pytest.approx(mid, rel=1e-12)

    def test_equatorial_fraction_is_solid_angle_of_band(self):
        aspect = 2.0
        lat = AxisymmetricCSMDensityProfile.disk_half_opening_angle(aspect).to_value(u.rad)
        theta = np.linspace(np.pi / 2 - lat, np.pi / 2 + lat, 4001)
        band = 0.5 * _trapz(np.sin(theta), theta)
        assert AxisymmetricCSMDensityProfile.equatorial_solid_angle_fraction(aspect) == pytest.approx(band, rel=1e-6)

    # ------------------------------------------- #
    # Validation                                  #
    # ------------------------------------------- #
    @pytest.mark.parametrize(
        "override",
        [
            {"rho_int": -1e-18},
            {"rho_ext": -1e-18},
            {"r_ref": 0.0},
            {"r_trans": -1.0},
            {"delta_trans": 0.0},
            {"f_pole": 1.5},
            {"f_pole": -0.1},
            {"aspect": 0.0},
            {"aspect": -1.0},
            {"theta": np.nan},
        ],
    )
    def test_invalid_parameters_raise(self, override):
        params = {**self._BASE, "theta": np.pi / 2, **override}
        with pytest.raises(ValueError):
            AxisymmetricCSMDensityProfile.eval(1e16, **params)

    def test_missing_theta_raises(self):
        with pytest.raises(TypeError):
            AxisymmetricCSMDensityProfile.eval(1e16, **self._BASE)

    # ------------------------------------------------------------------ #
    # Inner radius                                                       #
    # ------------------------------------------------------------------ #
    def test_r_min_defaults_to_no_truncation(self):
        r = np.geomspace(1e10, 1e18, 50)
        params = {**self._BASE, "theta": np.pi / 2}
        default = AxisymmetricCSMDensityProfile.as_optimized_callable(**params)
        explicit = AxisymmetricCSMDensityProfile.as_optimized_callable(**params, r_min=0.0)
        assert np.array_equal(default(r, 0.0), explicit(r, 0.0))

    @pytest.mark.parametrize("scalar", [False, True])
    def test_density_vanishes_below_r_min_only(self, scalar):
        r_min = 3e14
        params = {**self._BASE, "theta": np.pi / 2}
        truncated = AxisymmetricCSMDensityProfile.as_optimized_callable(**params, r_min=r_min)
        full = AxisymmetricCSMDensityProfile.as_optimized_callable(**params)
        r = np.array([1e13, 2.9e14, 3e14, 3.1e14, 1e16])
        evaluate = (lambda f: np.array([f(float(x), 0.0) for x in r])) if scalar else (lambda f: f(r, 0.0))
        expected = np.where(r >= r_min, full(r, 0.0), 0.0)
        result = evaluate(truncated)
        assert np.array_equal(result == 0.0, expected == 0.0)
        assert np.allclose(result, expected, rtol=1e-12, atol=0.0)

    def test_r_min_accepts_units(self):
        a = AxisymmetricCSMDensityProfile.eval(1e14 * u.cm, theta=1.0, **self._BASE, r_min=1e15 * u.cm)
        b = AxisymmetricCSMDensityProfile.eval(1e14 * u.cm, theta=1.0, **self._BASE, r_min=1e13 * u.m)
        assert a.value == 0.0 and b.value == 0.0

    def test_negative_r_min_raises(self):
        with pytest.raises(ValueError):
            AxisymmetricCSMDensityProfile.eval(1e16, theta=1.0, **self._BASE, r_min=-1.0)

    # ------------------------------------------------------------------ #
    # Scalar fast path                                                   #
    # ------------------------------------------------------------------ #
    @pytest.mark.parametrize("params", PARAMETER_SETS)
    @pytest.mark.parametrize("scalar_type", [float, np.float64])
    def test_scalar_fast_path_matches_the_vectorized_evaluation(self, params, scalar_type):
        """Scalar radii take a dedicated path in the optimized callable; it must agree with the array path."""
        density = AxisymmetricCSMDensityProfile.as_optimized_callable(**params)
        # Many decades, including radii deep in the exponentially suppressed tail on both sides of r_trans.
        r = np.geomspace(1e10, 1e25, 200)
        reference = density(r, 0.0)  # array input: vectorized path
        scalar = np.array([density(scalar_type(ri), 0.0) for ri in r])
        assert np.all(np.isfinite(scalar))
        assert np.allclose(scalar, reference, rtol=1e-12, atol=0.0)

    def test_array_valued_theta_is_evaluated_by_the_vectorized_path(self):
        params = {**self._BASE, "theta": np.array([0.3, 1.2, np.pi / 2])}
        density = AxisymmetricCSMDensityProfile.as_optimized_callable(**params)
        processed = AxisymmetricCSMDensityProfile._validate_and_process_parameters(**params)
        expected = AxisymmetricCSMDensityProfile._opt_eval(1e16, 0.0, **processed)
        assert np.shape(expected) == (3,)
        assert np.allclose(density(1e16, 0.0), expected, rtol=1e-13)

    def test_nonpositive_radius_defers_to_the_vectorized_evaluation(self):
        params = {**self._BASE, "theta": np.pi / 2}
        density = AxisymmetricCSMDensityProfile.as_optimized_callable(**params)
        processed = AxisymmetricCSMDensityProfile._validate_and_process_parameters(**params)
        with np.errstate(all="ignore"):
            expected = AxisymmetricCSMDensityProfile._opt_eval(np.float64(0.0), 0.0, **processed)
            result = density(np.float64(0.0), 0.0)
        assert np.array_equal(result, expected, equal_nan=True)


# =========================================================================== #
# Axisymmetric profile: sector filling parameters                             #
# =========================================================================== #
class TestAxisymmetricFillingParameters:
    _GRIDS = (np.array([0.5, 1.0, 2.0]), np.array([0.0, np.pi / 4, np.pi / 2]), np.array([0.2, 0.5, 0.8]))

    @pytest.fixture(scope="class")
    def table(self):
        return AxisymmetricCSMDensityProfile.build_filling_parameter_table(*self._GRIDS, n_pix=80)

    def test_table_shape_and_range(self, table):
        assert table.shape == (2, 3, 3, 3)
        assert np.all((table > 0.0) & (table <= 1.0))

    def test_progress_bar_does_not_change_the_table(self, table):
        shown = AxisymmetricCSMDensityProfile.build_filling_parameter_table(*self._GRIDS, n_pix=80, progress=True)
        assert np.array_equal(shown, table)

    def test_interpolation_reproduces_grid_points(self, table):
        out = AxisymmetricCSMDensityProfile.compute_filling_parameters(0.5, np.pi / 4, 1.0, table=(*self._GRIDS, table))
        assert out["polar"]["f_A"] == pytest.approx(table[0, 1, 1, 1], rel=1e-12)
        assert out["equatorial"]["f_A"] == pytest.approx(table[1, 1, 1, 1], rel=1e-12)

    def test_volume_fractions_partition_the_filling_factor(self):
        out = AxisymmetricCSMDensityProfile.compute_filling_parameters([0.2, 0.6], 0.7, 1.3, n_pix=60)
        assert np.allclose(out["polar"]["f_V"] + out["equatorial"]["f_V"], [0.2, 0.6], rtol=1e-12)

    def test_path_length_is_volume_over_projected_area(self):
        out = AxisymmetricCSMDensityProfile.compute_filling_parameters(0.4, 0.7, 1.3, n_pix=60)
        for sector in out.values():
            assert sector["ell_over_R"] == pytest.approx(4.0 / 3.0 * sector["f_V"] / sector["f_A"])

    def test_inclination_is_folded_about_ninety_degrees(self):
        a = AxisymmetricCSMDensityProfile.compute_filling_parameters(0.4, 0.6, 1.0, n_pix=60)
        b = AxisymmetricCSMDensityProfile.compute_filling_parameters(0.4, np.pi - 0.6, 1.0, n_pix=60)
        assert a["polar"]["f_A"] == pytest.approx(b["polar"]["f_A"], rel=1e-12)

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1])
    def test_invalid_filling_factor_raises(self, bad):
        with pytest.raises(ValueError, match="filling_factor"):
            AxisymmetricCSMDensityProfile.compute_filling_parameters(bad, 0.5, 1.0)

    def test_mismatched_table_raises(self, table):
        with pytest.raises(ValueError, match="expected"):
            AxisymmetricCSMDensityProfile.compute_filling_parameters(0.5, 0.5, 1.0, table=(*self._GRIDS, table[:, :2]))
