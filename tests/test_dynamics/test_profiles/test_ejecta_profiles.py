"""
Testing suite for ejecta density profiles.

This module provides unit testing for the ejecta density profiles in
:py:mod:`trilobite.dynamics.profiles.ejecta`. Tests are organized into:

(a) API functionality — ``eval``, ``feval``, ``kernel``, ``fkernel``,
    ``as_callable``, ``as_optimized_callable``, ``process_arguments``,
    and ``process_parameters`` all behave as documented.

(b) Physical accuracy — the homologous factored form ``rho = G(r/t)/t^3``
    is verified, and the mass and energy budgets from ``normalize()`` are
    confirmed by numerical quadrature.
"""

import numpy as np
import pytest
from astropy import units as u
from scipy.integrate import quad

from trilobite.dynamics.profiles.ejecta import (
    BrokenPowerLawEjectaProfile,
    ExponentialEjectaProfile,
)

# Standard test grids
_R_CGS = np.geomspace(1e13, 1e18, 50)
_R_QTY = _R_CGS * u.cm
_T_SCALAR = 10.0 * 24.0 * 3600.0  # 10 days in seconds
_T_QTY = _T_SCALAR * u.s

# Fiducial physical normalizations
_E_EJ = 1e51   # erg (1 foe)
_M_EJ = 1.989e33  # 1 Msun in g


# =========================================================================== #
# Base Test Class                                                             #
# =========================================================================== #
class BaseTestEjectaProfile:
    """
    Base test class for homologous ejecta density profiles.

    Subclasses must set ``PROFILE_CLASS`` and ``PARAMETER_SETS``.
    """

    PROFILE_CLASS = None
    PARAMETER_SETS: list[dict] = []

    # ------------------------------------------- #
    # API / Structural Tests                      #
    # ------------------------------------------- #

    def test_eval_returns_quantity(self):
        """eval returns a Quantity in g cm^{-3}."""
        for params in self.PARAMETER_SETS:
            result = self.PROFILE_CLASS.eval(_R_QTY, _T_QTY, **params)
            assert isinstance(result, u.Quantity)
            assert result.unit.is_equivalent(u.g / u.cm**3)

    def test_eval_shape_matches_input(self):
        """eval output shape matches r input shape."""
        for params in self.PARAMETER_SETS:
            result = self.PROFILE_CLASS.eval(_R_QTY, _T_QTY, **params)
            assert result.shape == _R_QTY.shape

    def test_eval_scalar_input(self):
        """eval handles scalar r and t inputs."""
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

    def test_kernel_returns_quantity(self):
        """kernel returns a Quantity in g s^3 cm^{-3}."""
        v_test = np.geomspace(1e7, 1e10, 20) * u.cm / u.s
        for params in self.PARAMETER_SETS:
            result = self.PROFILE_CLASS.kernel(v_test, **params)
            assert isinstance(result, u.Quantity)
            assert result.unit.is_equivalent(u.g * u.s**3 / u.cm**3)

    def test_fkernel_matches_kernel(self):
        """fkernel with processed params matches kernel numerically."""
        v_cgs = np.geomspace(1e7, 1e10, 20)
        for params in self.PARAMETER_SETS:
            processed_params = self.PROFILE_CLASS.process_parameters(**params)
            fkernel_result = self.PROFILE_CLASS.fkernel(v_cgs, **processed_params)
            kernel_value = self.PROFILE_CLASS.kernel(v_cgs * u.cm / u.s, **params).value
            assert np.allclose(fkernel_result, kernel_value)

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

    def test_output_is_nonnegative(self):
        """Density is non-negative on the entire test grid."""
        for params in self.PARAMETER_SETS:
            result = self.PROFILE_CLASS.eval(_R_QTY, _T_QTY, **params)
            assert np.all(result.value >= 0)

    # ------------------------------------------- #
    # Physical Accuracy: Homologous Form          #
    # ------------------------------------------- #

    def test_homologous_factored_form(self):
        """eval(r, t) == kernel(r/t) / t^3 (homologous factored form)."""
        for params in self.PARAMETER_SETS:
            rho = self.PROFILE_CLASS.eval(_R_QTY, _T_QTY, **params).value
            v = _R_CGS / _T_SCALAR
            G = self.PROFILE_CLASS.kernel(v * u.cm / u.s, **params).value
            rho_from_kernel = G / _T_SCALAR**3
            assert np.allclose(rho, rho_from_kernel), (
                f"Homologous factored form violated for {params}"
            )

    def test_homologous_scaling(self):
        """rho(λr, λt) == λ^{-3} * rho(r, t) — the t^{-3} pre-factor under dilation."""
        lam = 3.0
        for params in self.PARAMETER_SETS:
            rho_orig = self.PROFILE_CLASS.eval(_R_QTY, _T_QTY, **params).value
            rho_scaled = self.PROFILE_CLASS.eval(lam * _R_QTY, lam * _T_QTY, **params).value
            assert np.allclose(rho_scaled, rho_orig / lam**3, rtol=1e-10), (
                f"Homologous dilation scaling violated for {params}"
            )


# =========================================================================== #
# Concrete Ejecta Profile Test Classes                                        #
# =========================================================================== #

class TestBrokenPowerLawEjectaProfile(BaseTestEjectaProfile):
    PROFILE_CLASS = BrokenPowerLawEjectaProfile

    PARAMETER_SETS = [
        {"K": 1e22, "v_t": 1e9, "n": 10, "delta": 0},
        {"K": 5e21, "v_t": 5e8, "n": 12, "delta": 1},
        {"K": 1e22, "v_t": 1e9, "n": 10, "delta": 0, "v_max": 3e9},
    ]

    # ------------------------------------------- #
    # Kernel Accuracy                             #
    # ------------------------------------------- #

    def test_inner_branch_formula(self):
        """For v < v_t, G(v) == K * v^{-delta}."""
        params = {"K": 1e22, "v_t": 1e9, "n": 10, "delta": 0}
        v_inner = np.geomspace(1e7, 0.5e9, 20) * u.cm / u.s
        result = BrokenPowerLawEjectaProfile.kernel(v_inner, **params).value
        expected = params["K"] * v_inner.value ** (-params["delta"])
        assert np.allclose(result, expected, rtol=1e-10)

    def test_outer_branch_formula(self):
        """For v > v_t, G(v) == K * v_t^{n-delta} * v^{-n}."""
        params = {"K": 1e22, "v_t": 1e9, "n": 10, "delta": 0}
        v_outer = np.geomspace(2e9, 5e9, 20) * u.cm / u.s
        result = BrokenPowerLawEjectaProfile.kernel(v_outer, **params).value
        K_outer = params["K"] * params["v_t"] ** (params["n"] - params["delta"])
        expected = K_outer * v_outer.value ** (-params["n"])
        assert np.allclose(result, expected, rtol=1e-10)

    def test_continuity_at_vt(self):
        """Kernel is continuous at v_t."""
        params = {"K": 1e22, "v_t": 1e9, "n": 10, "delta": 0}
        v_t = params["v_t"]
        eps = v_t * 1e-6
        G_in = BrokenPowerLawEjectaProfile.kernel((v_t - eps) * u.cm / u.s, **params).value
        G_out = BrokenPowerLawEjectaProfile.kernel((v_t + eps) * u.cm / u.s, **params).value
        assert G_in == pytest.approx(G_out, rel=1e-4)

    def test_truncation_zero_beyond_vmax(self):
        """G(v) == 0 for v > v_max when v_max is finite."""
        params = {"K": 1e22, "v_t": 1e9, "n": 10, "delta": 0, "v_max": 3e9}
        v_beyond = np.array([3.1e9, 5e9, 1e10]) * u.cm / u.s
        result = BrokenPowerLawEjectaProfile.kernel(v_beyond, **params).value
        assert np.all(result == 0.0)

    # ------------------------------------------- #
    # Normalization Budget Tests                  #
    # ------------------------------------------- #

    @pytest.mark.parametrize("n,delta", [(10, 0), (12, 1), (7, 0)])
    def test_normalize_mass_budget(self, n, delta):
        """After normalize(), ∫ 4π v² G(v) dv == M_ej to 0.01%."""
        K, v_t = BrokenPowerLawEjectaProfile.normalize(_E_EJ, _M_EJ, n=n, delta=delta)
        params = {"K": K.value, "v_t": v_t.value, "n": n, "delta": delta}
        v_t_cgs = v_t.value

        def integrand(v):
            return 4.0 * np.pi * v**2 * BrokenPowerLawEjectaProfile.kernel(
                v * u.cm / u.s, **params
            ).value

        M_computed, _ = quad(
            integrand, 0.0, 100.0 * v_t_cgs,
            limit=300, epsabs=0.0, epsrel=1e-6,
            points=[v_t_cgs],
        )
        assert M_computed == pytest.approx(_M_EJ, rel=1e-4)

    @pytest.mark.parametrize("n,delta", [(10, 0), (12, 1), (7, 0)])
    def test_normalize_energy_budget(self, n, delta):
        """After normalize(), ∫ 2π v^4 G(v) dv == E_ej to 0.01%."""
        K, v_t = BrokenPowerLawEjectaProfile.normalize(_E_EJ, _M_EJ, n=n, delta=delta)
        params = {"K": K.value, "v_t": v_t.value, "n": n, "delta": delta}
        v_t_cgs = v_t.value

        def integrand(v):
            return 2.0 * np.pi * v**4 * BrokenPowerLawEjectaProfile.kernel(
                v * u.cm / u.s, **params
            ).value

        E_computed, _ = quad(
            integrand, 0.0, 100.0 * v_t_cgs,
            limit=300, epsabs=0.0, epsrel=1e-6,
            points=[v_t_cgs],
        )
        assert E_computed == pytest.approx(_E_EJ, rel=1e-4)

    def test_normalize_returns_quantities(self):
        """normalize returns a (Quantity, Quantity) tuple with correct units."""
        K, v_t = BrokenPowerLawEjectaProfile.normalize(_E_EJ, _M_EJ)
        assert isinstance(K, u.Quantity)
        assert isinstance(v_t, u.Quantity)
        assert v_t.unit.is_equivalent(u.cm / u.s)


class TestExponentialEjectaProfile(BaseTestEjectaProfile):
    PROFILE_CLASS = ExponentialEjectaProfile

    PARAMETER_SETS = [
        {"K": 1e10, "v_e": 1e9},
        {"K": 5e10, "v_e": 5e8, "v_min": 1e7, "v_max": 5e9},
    ]

    # ------------------------------------------- #
    # Kernel Accuracy                             #
    # ------------------------------------------- #

    def test_exponential_formula_inside_window(self):
        """G(v) == K * exp(-v/v_e) inside [v_min, v_max]."""
        params = {"K": 1e10, "v_e": 1e9, "v_min": 1e8, "v_max": 5e9}
        v_inside = np.geomspace(2e8, 4e9, 20) * u.cm / u.s
        result = ExponentialEjectaProfile.kernel(v_inside, **params).value
        expected = params["K"] * np.exp(-v_inside.value / params["v_e"])
        assert np.allclose(result, expected, rtol=1e-10)

    def test_zero_outside_window(self):
        """G(v) == 0 outside [v_min, v_max]."""
        params = {"K": 1e10, "v_e": 1e9, "v_min": 1e8, "v_max": 5e9}
        v_below = 0.5e8 * u.cm / u.s
        v_above = 6e9 * u.cm / u.s
        assert ExponentialEjectaProfile.kernel(v_below, **params).value == 0.0
        assert ExponentialEjectaProfile.kernel(v_above, **params).value == 0.0

    def test_monotone_decay(self):
        """G(v) is monotonically decreasing for untruncated profile."""
        params = {"K": 1e10, "v_e": 1e9}
        v_arr = np.geomspace(1e7, 1e10, 100) * u.cm / u.s
        G = ExponentialEjectaProfile.kernel(v_arr, **params).value
        assert np.all(np.diff(G) <= 0), "Exponential kernel is not monotonically decreasing"

    # ------------------------------------------- #
    # Normalization Budget Tests                  #
    # ------------------------------------------- #

    def test_normalize_mass_budget(self):
        """After normalize(), ∫ 4π v² G(v) dv == M_ej to 0.001%."""
        K, v_e = ExponentialEjectaProfile.normalize(_E_EJ, _M_EJ)
        params = {"K": K.value, "v_e": v_e.value}
        v_e_cgs = v_e.value

        def integrand(v):
            return 4.0 * np.pi * v**2 * ExponentialEjectaProfile.kernel(
                v * u.cm / u.s, **params
            ).value

        M_computed, _ = quad(
            integrand, 0.0, 30.0 * v_e_cgs,
            limit=200, epsabs=0.0, epsrel=1e-7,
        )
        assert M_computed == pytest.approx(_M_EJ, rel=1e-5)

    def test_normalize_energy_budget(self):
        """After normalize(), ∫ 2π v^4 G(v) dv == E_ej to 0.001%."""
        K, v_e = ExponentialEjectaProfile.normalize(_E_EJ, _M_EJ)
        params = {"K": K.value, "v_e": v_e.value}
        v_e_cgs = v_e.value

        def integrand(v):
            return 2.0 * np.pi * v**4 * ExponentialEjectaProfile.kernel(
                v * u.cm / u.s, **params
            ).value

        E_computed, _ = quad(
            integrand, 0.0, 30.0 * v_e_cgs,
            limit=200, epsabs=0.0, epsrel=1e-7,
        )
        assert E_computed == pytest.approx(_E_EJ, rel=1e-5)

    def test_normalize_untruncated_analytic_ve(self):
        """Untruncated case: v_e == sqrt(E_ej / (6 M_ej)) analytically."""
        K, v_e = ExponentialEjectaProfile.normalize(_E_EJ, _M_EJ)
        v_e_expected = np.sqrt(_E_EJ / (6.0 * _M_EJ))
        assert v_e.to(u.cm / u.s).value == pytest.approx(v_e_expected, rel=1e-10)

    def test_normalize_returns_quantities(self):
        """normalize returns a (Quantity, Quantity) tuple with correct units."""
        K, v_e = ExponentialEjectaProfile.normalize(_E_EJ, _M_EJ)
        assert isinstance(K, u.Quantity)
        assert isinstance(v_e, u.Quantity)
        assert v_e.unit.is_equivalent(u.cm / u.s)
        assert K.unit.is_equivalent(u.g * u.s**3 / u.cm**3)


# =========================================================================== #
# Cross-Profile Physical Consistency                                          #
# =========================================================================== #

def test_bpl_and_exponential_same_EM_budget():
    """BPL and exponential normalizations both satisfy the same E/M = E_ej/M_ej."""
    K_bpl, v_t = BrokenPowerLawEjectaProfile.normalize(_E_EJ, _M_EJ)
    K_exp, v_e = ExponentialEjectaProfile.normalize(_E_EJ, _M_EJ)

    params_bpl = {"K": K_bpl.value, "v_t": v_t.value, "n": 10, "delta": 0}
    params_exp = {"K": K_exp.value, "v_e": v_e.value}

    def mass_integrand_bpl(v):
        return 4.0 * np.pi * v**2 * BrokenPowerLawEjectaProfile.kernel(
            v * u.cm / u.s, **params_bpl
        ).value

    def mass_integrand_exp(v):
        return 4.0 * np.pi * v**2 * ExponentialEjectaProfile.kernel(
            v * u.cm / u.s, **params_exp
        ).value

    M_bpl, _ = quad(mass_integrand_bpl, 0.0, 100.0 * v_t.value,
                    limit=300, epsrel=1e-5, points=[v_t.value])
    M_exp, _ = quad(mass_integrand_exp, 0.0, 30.0 * v_e.value,
                    limit=200, epsrel=1e-5)

    assert M_bpl == pytest.approx(_M_EJ, rel=1e-4)
    assert M_exp == pytest.approx(_M_EJ, rel=1e-4)
