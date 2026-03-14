"""Unit tests for the free-free optical depth module (absorption.py).

Organisation
------------
TestOutputContract       — shape, dtype, scalar-frequency handling
TestPositivity           — tau > 0 for all profile types
TestUnitCoercion         — bare float == astropy.Quantity for all wrappers
TestShellAnalytic        — tau = alpha_nu * L  (cross-check against core.py)
TestQuadratureVsShell    — constant-profile quadrature matches analytic shell
TestArraysVsShell        — uniform-rho trapezoidal matches analytic shell
TestWindRadialScaling    — tau ∝ r^-3 for r_max → ∞
TestPowerlawSpecialCases — p=0 matches shell; k=0 is finite; k<0 converges at infinity
TestPowerlawSign         — both k>0 and k<0 return positive tau
TestRmaxEffect           — larger r_max always increases tau
TestMonotonicity         — tau is decreasing with nu at radio frequencies
TestRJVariants           — RJ variants return positive arrays of the right shape
"""

import numpy as np
import pytest
from astropy import units as u

from triceratops.radiation.free_free.absorption import (
    _proton_mass_cgs,
    compute_ff_optical_depth_from_arrays,
    compute_ff_optical_depth_from_quadrature,
    compute_ff_optical_depth_powerlaw,
    compute_ff_optical_depth_shell,
    compute_ff_optical_depth_wind,
    compute_ff_RJ_optical_depth_from_arrays,
    compute_ff_RJ_optical_depth_from_quadrature,
    compute_ff_RJ_optical_depth_powerlaw,
    compute_ff_RJ_optical_depth_shell,
    compute_ff_RJ_optical_depth_wind,
)
from triceratops.radiation.free_free.core import compute_ff_absorption

# ------------------------------------------------------------------ #
# Shared test fixtures / constants                                    #
# ------------------------------------------------------------------ #
# Standard physical parameters used across most tests.
_R_CM = 1.0e15  # inner radius [cm]
_R_MAX_CM = 2.0e15  # outer radius [cm]
_T_K = 1.0e4  # electron temperature [K]
_RHO_GCM3 = 1.0e-19  # uniform mass density [g cm^-3]
_MDOT_GS = 1.0e20  # mass-loss rate [g s^-1]
_V_CMS = 1.0e8  # wind velocity [cm s^-1]
_MU_E = 1.2  # mean molecular weight per electron
_MU_I = 1.3  # mean molecular weight per ion
_Z = 1.0  # mean ionic charge
_G_FF = 5.0  # Gaunt factor

# Frequency arrays used for shape / monotonicity tests.
_NU_2 = np.array([1.0e9, 1.0e10])  # 2 radio frequencies [Hz]
_NU_5 = np.geomspace(1.0e8, 1.0e11, 5)  # wider sweep


@pytest.fixture(scope="module")
def nu_arr():
    return _NU_2


# Density profiles for quadrature tests — constant n_e, n_i, T.
_NE_CONST = _RHO_GCM3 / (_MU_E * _proton_mass_cgs)  # cm^-3
_NI_CONST = _RHO_GCM3 / (_MU_I * _proton_mass_cgs)  # cm^-3
_NE_FN = lambda r: _NE_CONST  # noqa: E731
_NI_FN = lambda r: _NI_CONST  # noqa: E731
_T_FN = lambda r: _T_K  # noqa: E731

# Dense radial grid for array-integration tests.
_R_GRID = np.linspace(_R_CM, _R_MAX_CM, 5000)
_RHO_GRID = np.full_like(_R_GRID, _RHO_GCM3)


# ================================================================== #
# TestOutputContract                                                  #
# ================================================================== #
class TestOutputContract:
    """All public wrappers must return a plain ndarray of shape (n_nu,)."""

    def test_shell_returns_ndarray(self):
        result = compute_ff_optical_depth_shell(
            _NU_2 * u.Hz, _R_CM * u.cm, _RHO_GCM3 * u.g / u.cm**3, r_max=_R_MAX_CM * u.cm, temperature=_T_K * u.K
        )
        assert isinstance(result, np.ndarray)

    def test_shell_shape(self):
        result = compute_ff_optical_depth_shell(
            _NU_2 * u.Hz, _R_CM * u.cm, _RHO_GCM3 * u.g / u.cm**3, r_max=_R_MAX_CM * u.cm, temperature=_T_K * u.K
        )
        assert result.shape == (2,)

    def test_wind_shape(self):
        result = compute_ff_optical_depth_wind(
            _NU_2 * u.Hz, _R_CM * u.cm, _MDOT_GS * u.g / u.s, _V_CMS * u.cm / u.s, temperature=_T_K * u.K
        )
        assert result.shape == (2,)

    def test_powerlaw_shape(self):
        result = compute_ff_optical_depth_powerlaw(
            _NU_2 * u.Hz, _R_CM * u.cm, _RHO_GCM3 * u.g / u.cm**3, p=2.0, r_max=_R_MAX_CM * u.cm, temperature=_T_K * u.K
        )
        assert result.shape == (2,)

    def test_arrays_shape(self):
        result = compute_ff_optical_depth_from_arrays(
            _NU_2 * u.Hz, _R_GRID * u.cm, _RHO_GRID * u.g / u.cm**3, temperature=_T_K * u.K
        )
        assert result.shape == (2,)

    def test_quadrature_shape(self):
        result = compute_ff_optical_depth_from_quadrature(
            _NU_2 * u.Hz, _R_CM * u.cm, n_e=_NE_FN, n_i=_NI_FN, temperature=_T_FN, r_max=_R_MAX_CM * u.cm
        )
        assert result.shape == (2,)

    def test_scalar_frequency_gives_length_one_array(self):
        """A scalar frequency should be wrapped into a length-1 array."""
        result = compute_ff_optical_depth_shell(
            1e9 * u.Hz, _R_CM * u.cm, _RHO_GCM3 * u.g / u.cm**3, r_max=_R_MAX_CM * u.cm, temperature=_T_K * u.K
        )
        assert result.shape == (1,)

    def test_bare_float_frequency(self):
        """Bare float frequency (no Quantity) should work."""
        result = compute_ff_optical_depth_shell(1e9, _R_CM, _RHO_GCM3, r_max=_R_MAX_CM, temperature=_T_K)
        assert result.shape == (1,)


# ================================================================== #
# TestPositivity                                                      #
# ================================================================== #
class TestPositivity:
    """Optical depths must be strictly positive for all profile types."""

    def test_shell_positive(self):
        tau = compute_ff_optical_depth_shell(
            _NU_5 * u.Hz, _R_CM * u.cm, _RHO_GCM3 * u.g / u.cm**3, r_max=_R_MAX_CM * u.cm, temperature=_T_K * u.K
        )
        assert np.all(tau > 0)

    def test_wind_positive(self):
        tau = compute_ff_optical_depth_wind(
            _NU_5 * u.Hz, _R_CM * u.cm, _MDOT_GS * u.g / u.s, _V_CMS * u.cm / u.s, temperature=_T_K * u.K
        )
        assert np.all(tau > 0)

    def test_powerlaw_p2_positive(self):
        tau = compute_ff_optical_depth_powerlaw(
            _NU_5 * u.Hz, _R_CM * u.cm, _RHO_GCM3 * u.g / u.cm**3, p=2.0, r_max=_R_MAX_CM * u.cm, temperature=_T_K * u.K
        )
        assert np.all(tau > 0)

    def test_powerlaw_k_negative_positive(self):
        """p = 1.0 → k = -1 (converging integral); should still be positive."""
        tau = compute_ff_optical_depth_powerlaw(
            _NU_5 * u.Hz, _R_CM * u.cm, _RHO_GCM3 * u.g / u.cm**3, p=1.0, r_max=_R_MAX_CM * u.cm, temperature=_T_K * u.K
        )
        assert np.all(tau > 0)

    def test_arrays_positive(self):
        tau = compute_ff_optical_depth_from_arrays(
            _NU_5 * u.Hz, _R_GRID * u.cm, _RHO_GRID * u.g / u.cm**3, temperature=_T_K * u.K
        )
        assert np.all(tau > 0)

    def test_quadrature_positive(self):
        tau = compute_ff_optical_depth_from_quadrature(
            _NU_2 * u.Hz, _R_CM * u.cm, n_e=_NE_FN, n_i=_NI_FN, temperature=_T_FN, r_max=_R_MAX_CM * u.cm
        )
        assert np.all(tau > 0)


# ================================================================== #
# TestUnitCoercion                                                    #
# ================================================================== #
class TestUnitCoercion:
    """Bare-float and Quantity inputs must give bitwise-identical results."""

    def test_shell_float_vs_quantity(self):
        tau_q = compute_ff_optical_depth_shell(
            _NU_2 * u.Hz, _R_CM * u.cm, _RHO_GCM3 * u.g / u.cm**3, r_max=_R_MAX_CM * u.cm, temperature=_T_K * u.K
        )
        tau_f = compute_ff_optical_depth_shell(_NU_2, _R_CM, _RHO_GCM3, r_max=_R_MAX_CM, temperature=_T_K)
        np.testing.assert_array_equal(tau_q, tau_f)

    def test_wind_float_vs_quantity(self):
        tau_q = compute_ff_optical_depth_wind(
            _NU_2 * u.Hz, _R_CM * u.cm, _MDOT_GS * u.g / u.s, _V_CMS * u.cm / u.s, temperature=_T_K * u.K
        )
        tau_f = compute_ff_optical_depth_wind(_NU_2, _R_CM, _MDOT_GS, _V_CMS, temperature=_T_K)
        np.testing.assert_array_equal(tau_q, tau_f)

    def test_powerlaw_float_vs_quantity(self):
        tau_q = compute_ff_optical_depth_powerlaw(
            _NU_2 * u.Hz, _R_CM * u.cm, _RHO_GCM3 * u.g / u.cm**3, p=2.0, r_max=_R_MAX_CM * u.cm, temperature=_T_K * u.K
        )
        tau_f = compute_ff_optical_depth_powerlaw(_NU_2, _R_CM, _RHO_GCM3, p=2.0, r_max=_R_MAX_CM, temperature=_T_K)
        np.testing.assert_array_equal(tau_q, tau_f)

    def test_arrays_float_vs_quantity(self):
        tau_q = compute_ff_optical_depth_from_arrays(
            _NU_2 * u.Hz, _R_GRID * u.cm, _RHO_GRID * u.g / u.cm**3, temperature=_T_K * u.K
        )
        tau_f = compute_ff_optical_depth_from_arrays(_NU_2, _R_GRID, _RHO_GRID, temperature=_T_K)
        np.testing.assert_array_equal(tau_q, tau_f)

    def test_quadrature_float_vs_quantity(self):
        tau_q = compute_ff_optical_depth_from_quadrature(
            _NU_2 * u.Hz, _R_CM * u.cm, n_e=_NE_FN, n_i=_NI_FN, temperature=_T_FN, r_max=_R_MAX_CM * u.cm
        )
        tau_f = compute_ff_optical_depth_from_quadrature(
            _NU_2, _R_CM, n_e=_NE_FN, n_i=_NI_FN, temperature=_T_FN, r_max=_R_MAX_CM
        )
        np.testing.assert_array_equal(tau_q, tau_f)

    def test_unit_conversion_applied(self):
        """Frequency in GHz and radius in pc should give the same result as CGS."""
        tau_cgs = compute_ff_optical_depth_shell(
            np.array([1.0]) * u.GHz,
            _R_CM * u.cm,
            _RHO_GCM3 * u.g / u.cm**3,
            r_max=_R_MAX_CM * u.cm,
            temperature=_T_K * u.K,
        )
        tau_alt = compute_ff_optical_depth_shell(
            np.array([1.0e9]) * u.Hz,
            _R_CM * u.cm,
            _RHO_GCM3 * u.g / u.cm**3,
            r_max=_R_MAX_CM * u.cm,
            temperature=_T_K * u.K,
        )
        np.testing.assert_allclose(tau_cgs, tau_alt, rtol=1e-12)


# ================================================================== #
# TestShellAnalytic                                                   #
# ================================================================== #
class TestShellAnalytic:
    """tau_shell = alpha_nu * (r_max - r).

    Cross-checks compute_ff_optical_depth_shell against
    compute_ff_absorption from core.py, for which the physics is
    independently tested.
    """

    def _expected_tau(self, nu_hz):
        """Compute expected tau = alpha_nu * L from core.py."""
        n_e = _RHO_GCM3 / (_MU_E * _proton_mass_cgs)
        n_i = _RHO_GCM3 / (_MU_I * _proton_mass_cgs)
        alpha = compute_ff_absorption(nu_hz, n_e, n_i, _Z, _T_K, _G_FF)  # cm^-1
        L = _R_MAX_CM - _R_CM
        return alpha.value * L

    def test_single_frequency(self):
        nu = 1.0e9  # Hz
        tau_actual = compute_ff_optical_depth_shell(
            np.array([nu]) * u.Hz,
            _R_CM * u.cm,
            _RHO_GCM3 * u.g / u.cm**3,
            r_max=_R_MAX_CM * u.cm,
            temperature=_T_K * u.K,
            mu_e=_MU_E,
            mu_i=_MU_I,
            Z=_Z,
            g_ff=_G_FF,
        )
        assert tau_actual[0] == pytest.approx(self._expected_tau(nu), rel=1e-10)

    def test_multiple_frequencies(self):
        tau_actual = compute_ff_optical_depth_shell(
            _NU_5 * u.Hz,
            _R_CM * u.cm,
            _RHO_GCM3 * u.g / u.cm**3,
            r_max=_R_MAX_CM * u.cm,
            temperature=_T_K * u.K,
            mu_e=_MU_E,
            mu_i=_MU_I,
            Z=_Z,
            g_ff=_G_FF,
        )
        expected = np.array([self._expected_tau(nu) for nu in _NU_5])
        np.testing.assert_allclose(tau_actual, expected, rtol=1e-10)

    def test_scales_linearly_with_depth(self):
        """Doubling the shell depth must double tau."""
        L = _R_MAX_CM - _R_CM
        tau1 = compute_ff_optical_depth_shell(
            _NU_2 * u.Hz,
            _R_CM * u.cm,
            _RHO_GCM3 * u.g / u.cm**3,
            r_max=(_R_CM + L) * u.cm,
            temperature=_T_K * u.K,
        )
        tau2 = compute_ff_optical_depth_shell(
            _NU_2 * u.Hz,
            _R_CM * u.cm,
            _RHO_GCM3 * u.g / u.cm**3,
            r_max=(_R_CM + 2 * L) * u.cm,
            temperature=_T_K * u.K,
        )
        np.testing.assert_allclose(tau2, 2.0 * tau1, rtol=1e-12)

    def test_scales_with_density_squared(self):
        """Doubling rho must quadruple tau (alpha ∝ n_e * n_i ∝ rho^2)."""
        tau1 = compute_ff_optical_depth_shell(
            _NU_2 * u.Hz,
            _R_CM * u.cm,
            _RHO_GCM3 * u.g / u.cm**3,
            r_max=_R_MAX_CM * u.cm,
            temperature=_T_K * u.K,
        )
        tau2 = compute_ff_optical_depth_shell(
            _NU_2 * u.Hz,
            _R_CM * u.cm,
            (2 * _RHO_GCM3) * u.g / u.cm**3,
            r_max=_R_MAX_CM * u.cm,
            temperature=_T_K * u.K,
        )
        np.testing.assert_allclose(tau2, 4.0 * tau1, rtol=1e-10)


# ================================================================== #
# TestQuadratureVsShell                                              #
# ================================================================== #
class TestQuadratureVsShell:
    """Quadrature with constant n_e, n_i, T equals the analytic shell result."""

    def test_matches_shell_single_frequency(self):
        nu = np.array([1.0e9])
        tau_shell = compute_ff_optical_depth_shell(
            nu * u.Hz,
            _R_CM * u.cm,
            _RHO_GCM3 * u.g / u.cm**3,
            r_max=_R_MAX_CM * u.cm,
            temperature=_T_K * u.K,
            mu_e=_MU_E,
            mu_i=_MU_I,
            Z=_Z,
            g_ff=_G_FF,
        )
        tau_quad = compute_ff_optical_depth_from_quadrature(
            nu * u.Hz,
            _R_CM * u.cm,
            n_e=_NE_FN,
            n_i=_NI_FN,
            temperature=_T_FN,
            r_max=_R_MAX_CM * u.cm,
            Z=_Z,
            g_ff=_G_FF,
        )
        np.testing.assert_allclose(tau_quad, tau_shell, rtol=1e-6)

    def test_matches_shell_multiple_frequencies(self):
        tau_shell = compute_ff_optical_depth_shell(
            _NU_2 * u.Hz,
            _R_CM * u.cm,
            _RHO_GCM3 * u.g / u.cm**3,
            r_max=_R_MAX_CM * u.cm,
            temperature=_T_K * u.K,
            mu_e=_MU_E,
            mu_i=_MU_I,
            Z=_Z,
            g_ff=_G_FF,
        )
        tau_quad = compute_ff_optical_depth_from_quadrature(
            _NU_2 * u.Hz,
            _R_CM * u.cm,
            n_e=_NE_FN,
            n_i=_NI_FN,
            temperature=_T_FN,
            r_max=_R_MAX_CM * u.cm,
            Z=_Z,
            g_ff=_G_FF,
        )
        np.testing.assert_allclose(tau_quad, tau_shell, rtol=1e-6)


# ================================================================== #
# TestArraysVsShell                                                  #
# ================================================================== #
class TestArraysVsShell:
    """Trapezoidal integration over uniform rho equals the analytic shell.

    For a constant integrand the trapezoidal rule is exact, so agreement
    should be near machine precision regardless of grid spacing.
    """

    def test_matches_shell_single_frequency(self):
        nu = np.array([1.0e9])
        tau_shell = compute_ff_optical_depth_shell(
            nu * u.Hz,
            _R_CM * u.cm,
            _RHO_GCM3 * u.g / u.cm**3,
            r_max=_R_MAX_CM * u.cm,
            temperature=_T_K * u.K,
            mu_e=_MU_E,
            mu_i=_MU_I,
            Z=_Z,
            g_ff=_G_FF,
        )
        tau_arr = compute_ff_optical_depth_from_arrays(
            nu * u.Hz,
            _R_GRID * u.cm,
            _RHO_GRID * u.g / u.cm**3,
            temperature=_T_K * u.K,
            mu_e=_MU_E,
            mu_i=_MU_I,
            Z=_Z,
            g_ff=_G_FF,
        )
        np.testing.assert_allclose(tau_arr, tau_shell, rtol=1e-6)

    def test_matches_shell_multiple_frequencies(self):
        tau_shell = compute_ff_optical_depth_shell(
            _NU_2 * u.Hz,
            _R_CM * u.cm,
            _RHO_GCM3 * u.g / u.cm**3,
            r_max=_R_MAX_CM * u.cm,
            temperature=_T_K * u.K,
            mu_e=_MU_E,
            mu_i=_MU_I,
            Z=_Z,
            g_ff=_G_FF,
        )
        tau_arr = compute_ff_optical_depth_from_arrays(
            _NU_2 * u.Hz,
            _R_GRID * u.cm,
            _RHO_GRID * u.g / u.cm**3,
            temperature=_T_K * u.K,
            mu_e=_MU_E,
            mu_i=_MU_I,
            Z=_Z,
            g_ff=_G_FF,
        )
        np.testing.assert_allclose(tau_arr, tau_shell, rtol=1e-6)


# ================================================================== #
# TestWindRadialScaling                                              #
# ================================================================== #
class TestWindRadialScaling:
    """For r_max → ∞, tau_wind ∝ r^{-3}."""

    def test_r_cubed_scaling(self):
        r1, r2 = 1.0e15, 2.0e15
        nu = np.array([1.0e9]) * u.Hz
        tau1 = compute_ff_optical_depth_wind(
            nu, r1 * u.cm, _MDOT_GS * u.g / u.s, _V_CMS * u.cm / u.s, temperature=_T_K * u.K
        )
        tau2 = compute_ff_optical_depth_wind(
            nu, r2 * u.cm, _MDOT_GS * u.g / u.s, _V_CMS * u.cm / u.s, temperature=_T_K * u.K
        )
        # tau1 / tau2 should equal (r2/r1)^3 = 8 for r_max = inf.
        np.testing.assert_allclose(tau1 / tau2, (r2 / r1) ** 3, rtol=1e-10)

    def test_r_max_infinity_larger_than_finite(self):
        nu = np.array([1.0e9]) * u.Hz
        tau_inf = compute_ff_optical_depth_wind(
            nu, _R_CM * u.cm, _MDOT_GS * u.g / u.s, _V_CMS * u.cm / u.s, temperature=_T_K * u.K
        )  # r_max = ∞
        tau_fin = compute_ff_optical_depth_wind(
            nu, _R_CM * u.cm, _MDOT_GS * u.g / u.s, _V_CMS * u.cm / u.s, r_max=_R_MAX_CM * u.cm, temperature=_T_K * u.K
        )
        assert tau_inf[0] > tau_fin[0]

    def test_higher_mdot_increases_tau(self):
        nu = np.array([1.0e9]) * u.Hz
        tau1 = compute_ff_optical_depth_wind(
            nu, _R_CM * u.cm, _MDOT_GS * u.g / u.s, _V_CMS * u.cm / u.s, temperature=_T_K * u.K
        )
        tau2 = compute_ff_optical_depth_wind(
            nu, _R_CM * u.cm, (10 * _MDOT_GS) * u.g / u.s, _V_CMS * u.cm / u.s, temperature=_T_K * u.K
        )
        assert tau2[0] > tau1[0]

    def test_mdot_squared_scaling(self):
        """tau ∝ mdot^2 (alpha ∝ n_e n_i ∝ rho^2 ∝ mdot^2)."""
        nu = np.array([1.0e9]) * u.Hz
        tau1 = compute_ff_optical_depth_wind(
            nu, _R_CM * u.cm, _MDOT_GS * u.g / u.s, _V_CMS * u.cm / u.s, temperature=_T_K * u.K
        )
        tau2 = compute_ff_optical_depth_wind(
            nu, _R_CM * u.cm, (2 * _MDOT_GS) * u.g / u.s, _V_CMS * u.cm / u.s, temperature=_T_K * u.K
        )
        np.testing.assert_allclose(tau2 / tau1, 4.0, rtol=1e-10)


# ================================================================== #
# TestPowerlawSpecialCases                                           #
# ================================================================== #
class TestPowerlawSpecialCases:
    """Edge cases and special values of the power-law index."""

    def test_p0_matches_shell(self):
        """p = 0 is a uniform density profile — must equal the shell result."""
        nu = _NU_2 * u.Hz
        tau_shell = compute_ff_optical_depth_shell(
            nu,
            _R_CM * u.cm,
            _RHO_GCM3 * u.g / u.cm**3,
            r_max=_R_MAX_CM * u.cm,
            temperature=_T_K * u.K,
            mu_e=_MU_E,
            mu_i=_MU_I,
            Z=_Z,
            g_ff=_G_FF,
        )
        tau_pl = compute_ff_optical_depth_powerlaw(
            nu,
            _R_CM * u.cm,
            _RHO_GCM3 * u.g / u.cm**3,
            p=0.0,
            r_max=_R_MAX_CM * u.cm,
            temperature=_T_K * u.K,
            mu_e=_MU_E,
            mu_i=_MU_I,
            Z=_Z,
            g_ff=_G_FF,
        )
        np.testing.assert_allclose(tau_pl, tau_shell, rtol=1e-10)

    def test_k_zero_p_half_positive(self):
        """p = 0.5 → k = 0 → logarithmic integral.  Result must be positive and finite."""
        tau = compute_ff_optical_depth_powerlaw(
            _NU_2 * u.Hz,
            _R_CM * u.cm,
            _RHO_GCM3 * u.g / u.cm**3,
            p=0.5,
            r_max=_R_MAX_CM * u.cm,
            temperature=_T_K * u.K,
        )
        assert np.all(tau > 0)
        assert np.all(np.isfinite(tau))

    def test_k_negative_converges_at_infinity(self):
        """p = 2.0 → k = -3 (convergent) — r_max = ∞ must give finite tau."""
        tau = compute_ff_optical_depth_powerlaw(
            _NU_2 * u.Hz,
            _R_CM * u.cm,
            _RHO_GCM3 * u.g / u.cm**3,
            p=2.0,
            r_max=np.inf,
            temperature=_T_K * u.K,
        )
        assert np.all(tau > 0)
        assert np.all(np.isfinite(tau))

    def test_k_positive_finite_rmax_positive(self):
        """p = 0.1 → k = 0.8 (divergent at ∞).  With finite r_max, tau is finite."""
        tau = compute_ff_optical_depth_powerlaw(
            _NU_2 * u.Hz,
            _R_CM * u.cm,
            _RHO_GCM3 * u.g / u.cm**3,
            p=0.1,
            r_max=_R_MAX_CM * u.cm,
            temperature=_T_K * u.K,
        )
        assert np.all(tau > 0)
        assert np.all(np.isfinite(tau))

    def test_larger_p_gives_smaller_tau_same_rmax(self):
        """Steeper profile puts less material in the line of sight beyond r."""
        nu = _NU_2 * u.Hz
        tau_shallow = compute_ff_optical_depth_powerlaw(
            nu,
            _R_CM * u.cm,
            _RHO_GCM3 * u.g / u.cm**3,
            p=0.5,
            r_max=_R_MAX_CM * u.cm,
            temperature=_T_K * u.K,
        )
        tau_steep = compute_ff_optical_depth_powerlaw(
            nu,
            _R_CM * u.cm,
            _RHO_GCM3 * u.g / u.cm**3,
            p=2.0,
            r_max=_R_MAX_CM * u.cm,
            temperature=_T_K * u.K,
        )
        assert np.all(tau_shallow > tau_steep)


# ================================================================== #
# TestPowerlawSign                                                    #
# ================================================================== #
class TestPowerlawSign:
    """k > 0 and k < 0 branches must both return positive tau."""

    @pytest.mark.parametrize("p", [0.0, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    def test_positive_for_various_p(self, p):
        tau = compute_ff_optical_depth_powerlaw(
            _NU_2 * u.Hz,
            _R_CM * u.cm,
            _RHO_GCM3 * u.g / u.cm**3,
            p=p,
            r_max=_R_MAX_CM * u.cm,
            temperature=_T_K * u.K,
        )
        assert np.all(tau > 0), f"p={p} gave non-positive tau: {tau}"

    @pytest.mark.parametrize("p", [0.0, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    def test_finite_for_various_p(self, p):
        tau = compute_ff_optical_depth_powerlaw(
            _NU_2 * u.Hz,
            _R_CM * u.cm,
            _RHO_GCM3 * u.g / u.cm**3,
            p=p,
            r_max=_R_MAX_CM * u.cm,
            temperature=_T_K * u.K,
        )
        assert np.all(np.isfinite(tau)), f"p={p} gave non-finite tau: {tau}"

    @pytest.mark.parametrize("p", [0.6, 0.75, 1.0, 1.5, 2.0, 3.0])
    def test_k_negative_with_inf_rmax(self, p):
        """For p > 0.5 the integral to infinity converges; result must be finite."""
        tau = compute_ff_optical_depth_powerlaw(
            _NU_2 * u.Hz,
            _R_CM * u.cm,
            _RHO_GCM3 * u.g / u.cm**3,
            p=p,
            r_max=np.inf,
            temperature=_T_K * u.K,
        )
        assert np.all(tau > 0)
        assert np.all(np.isfinite(tau))


# ================================================================== #
# TestRmaxEffect                                                     #
# ================================================================== #
class TestRmaxEffect:
    """A larger outer boundary always increases the optical depth."""

    def test_shell_larger_rmax(self):
        nu = _NU_2 * u.Hz
        tau1 = compute_ff_optical_depth_shell(
            nu, _R_CM * u.cm, _RHO_GCM3 * u.g / u.cm**3, r_max=_R_MAX_CM * u.cm, temperature=_T_K * u.K
        )
        tau2 = compute_ff_optical_depth_shell(
            nu, _R_CM * u.cm, _RHO_GCM3 * u.g / u.cm**3, r_max=(2 * _R_MAX_CM) * u.cm, temperature=_T_K * u.K
        )
        assert np.all(tau2 > tau1)

    def test_wind_larger_rmax(self):
        nu = _NU_2 * u.Hz
        tau1 = compute_ff_optical_depth_wind(
            nu, _R_CM * u.cm, _MDOT_GS * u.g / u.s, _V_CMS * u.cm / u.s, r_max=_R_MAX_CM * u.cm, temperature=_T_K * u.K
        )
        tau2 = compute_ff_optical_depth_wind(
            nu,
            _R_CM * u.cm,
            _MDOT_GS * u.g / u.s,
            _V_CMS * u.cm / u.s,
            r_max=(2 * _R_MAX_CM) * u.cm,
            temperature=_T_K * u.K,
        )
        assert np.all(tau2 > tau1)

    def test_powerlaw_larger_rmax(self):
        nu = _NU_2 * u.Hz
        tau1 = compute_ff_optical_depth_powerlaw(
            nu, _R_CM * u.cm, _RHO_GCM3 * u.g / u.cm**3, p=2.0, r_max=_R_MAX_CM * u.cm, temperature=_T_K * u.K
        )
        tau2 = compute_ff_optical_depth_powerlaw(
            nu, _R_CM * u.cm, _RHO_GCM3 * u.g / u.cm**3, p=2.0, r_max=(2 * _R_MAX_CM) * u.cm, temperature=_T_K * u.K
        )
        assert np.all(tau2 > tau1)

    def test_quadrature_larger_rmax(self):
        nu = _NU_2 * u.Hz
        tau1 = compute_ff_optical_depth_from_quadrature(
            nu, _R_CM * u.cm, n_e=_NE_FN, n_i=_NI_FN, temperature=_T_FN, r_max=_R_MAX_CM * u.cm
        )
        tau2 = compute_ff_optical_depth_from_quadrature(
            nu, _R_CM * u.cm, n_e=_NE_FN, n_i=_NI_FN, temperature=_T_FN, r_max=(2 * _R_MAX_CM) * u.cm
        )
        assert np.all(tau2 > tau1)


# ================================================================== #
# TestMonotonicity                                                   #
# ================================================================== #
class TestMonotonicity:
    """tau must be monotonically decreasing with nu at radio frequencies.

    At radio/microwave wavelengths (nu ~ 10^8-10^11 Hz, T ~ 10^4 K) the
    free-free opacity decreases with increasing frequency in all limits.
    """

    _NU_MONO = np.geomspace(1e8, 1e11, 20)  # 20 radio frequencies

    def test_shell_decreasing_with_nu(self):
        tau = compute_ff_optical_depth_shell(
            self._NU_MONO * u.Hz,
            _R_CM * u.cm,
            _RHO_GCM3 * u.g / u.cm**3,
            r_max=_R_MAX_CM * u.cm,
            temperature=_T_K * u.K,
        )
        assert np.all(np.diff(tau) < 0)

    def test_wind_decreasing_with_nu(self):
        tau = compute_ff_optical_depth_wind(
            self._NU_MONO * u.Hz,
            _R_CM * u.cm,
            _MDOT_GS * u.g / u.s,
            _V_CMS * u.cm / u.s,
            temperature=_T_K * u.K,
        )
        assert np.all(np.diff(tau) < 0)

    def test_powerlaw_decreasing_with_nu(self):
        tau = compute_ff_optical_depth_powerlaw(
            self._NU_MONO * u.Hz,
            _R_CM * u.cm,
            _RHO_GCM3 * u.g / u.cm**3,
            p=2.0,
            r_max=_R_MAX_CM * u.cm,
            temperature=_T_K * u.K,
        )
        assert np.all(np.diff(tau) < 0)

    def test_arrays_decreasing_with_nu(self):
        tau = compute_ff_optical_depth_from_arrays(
            self._NU_MONO * u.Hz,
            _R_GRID * u.cm,
            _RHO_GRID * u.g / u.cm**3,
            temperature=_T_K * u.K,
        )
        assert np.all(np.diff(tau) < 0)

    def test_quadrature_decreasing_with_nu(self):
        tau = compute_ff_optical_depth_from_quadrature(
            self._NU_MONO * u.Hz,
            _R_CM * u.cm,
            n_e=_NE_FN,
            n_i=_NI_FN,
            temperature=_T_FN,
            r_max=_R_MAX_CM * u.cm,
        )
        assert np.all(np.diff(tau) < 0)


# ================================================================== #
# TestRJVariants                                                     #
# ================================================================== #
class TestRJVariants:
    """All RJ public functions must return positive arrays of the right shape."""

    def test_rj_shell_shape_and_sign(self):
        tau = compute_ff_RJ_optical_depth_shell(
            _NU_5 * u.Hz,
            _R_CM * u.cm,
            _RHO_GCM3 * u.g / u.cm**3,
            r_max=_R_MAX_CM * u.cm,
            temperature=_T_K * u.K,
        )
        assert tau.shape == (5,)
        assert np.all(tau > 0)

    def test_rj_wind_shape_and_sign(self):
        tau = compute_ff_RJ_optical_depth_wind(
            _NU_5 * u.Hz,
            _R_CM * u.cm,
            _MDOT_GS * u.g / u.s,
            _V_CMS * u.cm / u.s,
            temperature=_T_K * u.K,
        )
        assert tau.shape == (5,)
        assert np.all(tau > 0)

    def test_rj_powerlaw_shape_and_sign(self):
        tau = compute_ff_RJ_optical_depth_powerlaw(
            _NU_5 * u.Hz,
            _R_CM * u.cm,
            _RHO_GCM3 * u.g / u.cm**3,
            p=2.0,
            r_max=_R_MAX_CM * u.cm,
            temperature=_T_K * u.K,
        )
        assert tau.shape == (5,)
        assert np.all(tau > 0)

    def test_rj_arrays_shape_and_sign(self):
        tau = compute_ff_RJ_optical_depth_from_arrays(
            _NU_5 * u.Hz,
            _R_GRID * u.cm,
            _RHO_GRID * u.g / u.cm**3,
            temperature=_T_K * u.K,
        )
        assert tau.shape == (5,)
        assert np.all(tau > 0)

    def test_rj_quadrature_shape_and_sign(self):
        tau = compute_ff_RJ_optical_depth_from_quadrature(
            _NU_2 * u.Hz,
            _R_CM * u.cm,
            n_e=_NE_FN,
            n_i=_NI_FN,
            temperature=_T_FN,
            r_max=_R_MAX_CM * u.cm,
        )
        assert tau.shape == (2,)
        assert np.all(tau > 0)

    def test_rj_shell_analytic(self):
        """tau_RJ_shell = alpha_RJ * L. Cross-check against compute_ff_RJ_absorption."""
        from triceratops.radiation.free_free.core import compute_ff_RJ_absorption

        nu = np.array([1.0e9])
        n_e = _RHO_GCM3 / (_MU_E * _proton_mass_cgs)
        n_i = _RHO_GCM3 / (_MU_I * _proton_mass_cgs)
        alpha_rj = compute_ff_RJ_absorption(nu * u.Hz, n_e, n_i, _Z, _T_K, _G_FF)
        L = _R_MAX_CM - _R_CM
        tau_expected = alpha_rj.value * L
        tau_actual = compute_ff_RJ_optical_depth_shell(
            nu * u.Hz,
            _R_CM * u.cm,
            _RHO_GCM3 * u.g / u.cm**3,
            r_max=_R_MAX_CM * u.cm,
            temperature=_T_K * u.K,
            mu_e=_MU_E,
            mu_i=_MU_I,
            Z=_Z,
            g_ff=_G_FF,
        )
        np.testing.assert_allclose(tau_actual, tau_expected, rtol=1e-10)

    def test_rj_unit_coercion(self):
        tau_q = compute_ff_RJ_optical_depth_shell(
            _NU_2 * u.Hz,
            _R_CM * u.cm,
            _RHO_GCM3 * u.g / u.cm**3,
            r_max=_R_MAX_CM * u.cm,
            temperature=_T_K * u.K,
        )
        tau_f = compute_ff_RJ_optical_depth_shell(
            _NU_2,
            _R_CM,
            _RHO_GCM3,
            r_max=_R_MAX_CM,
            temperature=_T_K,
        )
        np.testing.assert_array_equal(tau_q, tau_f)
