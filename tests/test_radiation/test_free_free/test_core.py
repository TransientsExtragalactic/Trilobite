"""Unit tests for the free-free core module (core.py).

Organisation
------------
TestPrivateEmissivity       — _log_ff_emissivity: formula, scalings, array I/O
TestPrivateAbsorption       — _log_ff_absorption: formula, scalings, log1p stability
TestPrivateRJEmissivity     — _log_ff_RJ_emissivity: freq-independence, T scaling
TestPrivateRJAbsorption     — _log_ff_RJ_absorption: nu^-2 and T^-3/2 scalings
TestPrivateWienEmissivity   — _log_ff_Wien_emissivity: identical to exact backend
TestPrivateWienAbsorption   — _log_ff_Wien_absorption: nu^-3, no stimulated-emission factor
TestPublicEmissivity        — compute_ff_emissivity: units, coercion, scalings, RJ limit
TestPublicAbsorption        — compute_ff_absorption: units, coercion, scalings, Wien limit
TestPublicRJEmissivity      — compute_ff_RJ_emissivity: freq-independence, units
TestPublicRJAbsorption      — compute_ff_RJ_absorption: nu^-2 scaling, units
"""

import numpy as np
import pytest
from astropy import units as u

from triceratops.radiation.constants import h_cgs, kB_cgs
from triceratops.radiation.free_free.core import (
    _ff_absorption_coefficient_cgs,
    _ff_emissivity_coefficient_cgs,
    _log_ff_absorption,
    _log_ff_emissivity,
    _log_ff_RJ_absorption,
    _log_ff_RJ_emissivity,
    _log_ff_Wien_absorption,
    _log_ff_Wien_emissivity,
    compute_ff_absorption,
    compute_ff_emissivity,
    compute_ff_RJ_absorption,
    compute_ff_RJ_emissivity,
)

# ------------------------------------------------------------------ #
# Shared reference conditions                                        #
# ------------------------------------------------------------------ #
_NU = 1.0e9  # Hz  — radio, deeply in RJ regime at T=1e4 K
_NE = 1.0e3  # cm^-3
_NI = 1.0e3  # cm^-3
_Z = 1.0  # hydrogen plasma
_T = 1.0e4  # K
_GFF = 5.0

# Natural-log forms used directly by the private backends.
_LOG_NU = np.log(_NU)
_LOG_NE = np.log(_NE)
_LOG_NI = np.log(_NI)
_LOG_Z = np.log(_Z)
_LOG_T = np.log(_T)

# Dimensionless frequency parameter at reference conditions.
_X = h_cgs * _NU / (kB_cgs * _T)  # hν / k_B T ≈ 4.8e-6  (deep RJ)


# ================================================================== #
# TestPrivateEmissivity                                              #
# ================================================================== #
class TestPrivateEmissivity:
    """_log_ff_emissivity implements log j_ν = log C_ff + 2 log Z
    + log n_e + log n_i - 0.5 log T - hν/(k_B T) + log g_ff."""

    def _reference(self, log_nu=_LOG_NU, log_ne=_LOG_NE, log_ni=_LOG_NI, log_Z=_LOG_Z, log_T=_LOG_T, g_ff=_GFF):
        """Direct evaluation of the R&L formula in log-space."""
        nu = np.exp(log_nu)
        T = np.exp(log_T)
        exp_term = -h_cgs * nu / (kB_cgs * T)
        return (
            np.log(_ff_emissivity_coefficient_cgs) + 2 * log_Z + log_ne + log_ni - 0.5 * log_T + exp_term + np.log(g_ff)
        )

    def test_scalar_matches_formula(self):
        result = _log_ff_emissivity(_LOG_NU, _LOG_NE, _LOG_NI, _LOG_Z, _LOG_T, _GFF)
        assert result == pytest.approx(self._reference(), rel=1e-12)

    def test_array_nu_matches_formula(self):
        log_nu_arr = np.log(np.geomspace(1e8, 1e12, 5))
        result = _log_ff_emissivity(log_nu_arr, _LOG_NE, _LOG_NI, _LOG_Z, _LOG_T, _GFF)
        expected = self._reference(log_nu=log_nu_arr)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_z_squared_scaling(self):
        """Doubling Z must increase log j by 2 log 2."""
        j1 = _log_ff_emissivity(_LOG_NU, _LOG_NE, _LOG_NI, _LOG_Z, _LOG_T, _GFF)
        j2 = _log_ff_emissivity(_LOG_NU, _LOG_NE, _LOG_NI, np.log(2 * _Z), _LOG_T, _GFF)
        assert j2 - j1 == pytest.approx(2 * np.log(2), rel=1e-12)

    def test_ne_linear_scaling(self):
        """Doubling n_e must increase log j by log 2."""
        j1 = _log_ff_emissivity(_LOG_NU, _LOG_NE, _LOG_NI, _LOG_Z, _LOG_T, _GFF)
        j2 = _log_ff_emissivity(_LOG_NU, np.log(2 * _NE), _LOG_NI, _LOG_Z, _LOG_T, _GFF)
        assert j2 - j1 == pytest.approx(np.log(2), rel=1e-12)

    def test_ni_linear_scaling(self):
        """Doubling n_i must increase log j by log 2."""
        j1 = _log_ff_emissivity(_LOG_NU, _LOG_NE, _LOG_NI, _LOG_Z, _LOG_T, _GFF)
        j2 = _log_ff_emissivity(_LOG_NU, _LOG_NE, np.log(2 * _NI), _LOG_Z, _LOG_T, _GFF)
        assert j2 - j1 == pytest.approx(np.log(2), rel=1e-12)

    def test_temperature_minus_half_scaling(self):
        """Quadrupling T must decrease log j by log 2  (T^{-1/2} factor)."""
        j1 = _log_ff_emissivity(_LOG_NU, _LOG_NE, _LOG_NI, _LOG_Z, _LOG_T, _GFF)
        j2 = _log_ff_emissivity(_LOG_NU, _LOG_NE, _LOG_NI, _LOG_Z, np.log(4 * _T), _GFF)
        # Δlog j from T factor alone: -0.5 * log(4) = -log(2)
        # The exp_term also changes, but for deep-RJ conditions the change is negligible.
        assert j2 - j1 == pytest.approx(-np.log(2), rel=1e-3)

    def test_gff_linear_scaling(self):
        """Doubling g_ff must increase log j by log 2."""
        j1 = _log_ff_emissivity(_LOG_NU, _LOG_NE, _LOG_NI, _LOG_Z, _LOG_T, _GFF)
        j2 = _log_ff_emissivity(_LOG_NU, _LOG_NE, _LOG_NI, _LOG_Z, _LOG_T, 2 * _GFF)
        assert j2 - j1 == pytest.approx(np.log(2), rel=1e-12)

    def test_output_is_finite(self):
        result = _log_ff_emissivity(_LOG_NU, _LOG_NE, _LOG_NI, _LOG_Z, _LOG_T, _GFF)
        assert np.isfinite(result)


# ================================================================== #
# TestPrivateAbsorption                                              #
# ================================================================== #
class TestPrivateAbsorption:
    """_log_ff_absorption implements log α_ν = log C_α + 2 log Z
    + log n_e + log n_i - 0.5 log T - 3 log ν
    + log(1 − exp(−hν/k_B T)) + log g_ff."""

    def _reference(self, log_nu=_LOG_NU, log_ne=_LOG_NE, log_ni=_LOG_NI, log_Z=_LOG_Z, log_T=_LOG_T, g_ff=_GFF):
        nu = np.exp(log_nu)
        T = np.exp(log_T)
        exp_term = -h_cgs * nu / (kB_cgs * T)
        return (
            np.log(_ff_absorption_coefficient_cgs)
            + 2 * log_Z
            + log_ne
            + log_ni
            - 0.5 * log_T
            - 3 * log_nu
            + np.log1p(-np.exp(exp_term))
            + np.log(g_ff)
        )

    def test_scalar_matches_formula(self):
        result = _log_ff_absorption(_LOG_NU, _LOG_NE, _LOG_NI, _LOG_Z, _LOG_T, _GFF)
        assert result == pytest.approx(self._reference(), rel=1e-12)

    def test_array_nu_matches_formula(self):
        log_nu_arr = np.log(np.geomspace(1e8, 1e12, 5))
        result = _log_ff_absorption(log_nu_arr, _LOG_NE, _LOG_NI, _LOG_Z, _LOG_T, _GFF)
        expected = self._reference(log_nu=log_nu_arr)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_z_squared_scaling(self):
        a1 = _log_ff_absorption(_LOG_NU, _LOG_NE, _LOG_NI, _LOG_Z, _LOG_T, _GFF)
        a2 = _log_ff_absorption(_LOG_NU, _LOG_NE, _LOG_NI, np.log(2 * _Z), _LOG_T, _GFF)
        assert a2 - a1 == pytest.approx(2 * np.log(2), rel=1e-12)

    def test_ne_linear_scaling(self):
        a1 = _log_ff_absorption(_LOG_NU, _LOG_NE, _LOG_NI, _LOG_Z, _LOG_T, _GFF)
        a2 = _log_ff_absorption(_LOG_NU, np.log(2 * _NE), _LOG_NI, _LOG_Z, _LOG_T, _GFF)
        assert a2 - a1 == pytest.approx(np.log(2), rel=1e-12)

    def test_nu_minus_three_scaling_in_wien_limit(self):
        """In the Wien limit (1 − e^{−x} ≈ 1), α ∝ ν^{−3}.
        Use very high ν / low T so the stimulated-emission factor is ~1."""
        log_T_cold = np.log(1.0)  # T = 1 K (extreme Wien)
        log_nu_hi = np.log(1e12)
        log_nu_hi2 = np.log(2e12)
        a1 = _log_ff_absorption(log_nu_hi, _LOG_NE, _LOG_NI, _LOG_Z, log_T_cold, 1.0)
        a2 = _log_ff_absorption(log_nu_hi2, _LOG_NE, _LOG_NI, _LOG_Z, log_T_cold, 1.0)
        # Δlog α = -3 log 2 (pure ν^{-3} scaling when stimulated emission ≈ 0)
        assert a2 - a1 == pytest.approx(-3 * np.log(2), rel=1e-4)

    def test_log1p_stability_at_small_x(self):
        """At very low ν the argument of log1p(-exp(-x)) is tiny; result must be finite."""
        log_nu_very_low = np.log(1.0)  # 1 Hz, hν/kT ~ 4.8e-23
        result = _log_ff_absorption(log_nu_very_low, _LOG_NE, _LOG_NI, _LOG_Z, _LOG_T, _GFF)
        assert np.isfinite(result)

    def test_output_is_finite_at_reference(self):
        result = _log_ff_absorption(_LOG_NU, _LOG_NE, _LOG_NI, _LOG_Z, _LOG_T, _GFF)
        assert np.isfinite(result)


# ================================================================== #
# TestPrivateRJEmissivity                                            #
# ================================================================== #
class TestPrivateRJEmissivity:
    """_log_ff_RJ_emissivity is independent of frequency and scales as T^{-1/2}."""

    def test_frequency_independent(self):
        """The RJ emissivity must not change when nu changes."""
        j1 = _log_ff_RJ_emissivity(_LOG_NU, _LOG_NE, _LOG_NI, _LOG_Z, _LOG_T, _GFF)
        j2 = _log_ff_RJ_emissivity(np.log(1e15), _LOG_NE, _LOG_NI, _LOG_Z, _LOG_T, _GFF)
        assert j1 == j2

    def test_array_frequency_all_equal(self):
        """Passing an array of frequencies must yield identical values."""
        log_nu_arr = np.log(np.geomspace(1e6, 1e18, 10))
        result = _log_ff_RJ_emissivity(log_nu_arr, _LOG_NE, _LOG_NI, _LOG_Z, _LOG_T, _GFF)
        assert np.all(result == result[0])

    def test_temperature_minus_half_scaling(self):
        """Quadrupling T must decrease log j_RJ by log 2."""
        j1 = _log_ff_RJ_emissivity(_LOG_NU, _LOG_NE, _LOG_NI, _LOG_Z, _LOG_T, _GFF)
        j2 = _log_ff_RJ_emissivity(_LOG_NU, _LOG_NE, _LOG_NI, _LOG_Z, np.log(4 * _T), _GFF)
        assert j2 - j1 == pytest.approx(-np.log(2), rel=1e-12)

    def test_z_squared_scaling(self):
        j1 = _log_ff_RJ_emissivity(_LOG_NU, _LOG_NE, _LOG_NI, _LOG_Z, _LOG_T, _GFF)
        j2 = _log_ff_RJ_emissivity(_LOG_NU, _LOG_NE, _LOG_NI, np.log(2 * _Z), _LOG_T, _GFF)
        assert j2 - j1 == pytest.approx(2 * np.log(2), rel=1e-12)

    def test_deep_rj_limit_matches_exact(self):
        """In the deep RJ regime (hν/kT ≈ 0), the exact and RJ backends agree."""
        log_nu_rj = np.log(1.0)  # 1 Hz — hν/kT ~ 4.8e-23
        j_exact = _log_ff_emissivity(log_nu_rj, _LOG_NE, _LOG_NI, _LOG_Z, _LOG_T, _GFF)
        j_rj = _log_ff_RJ_emissivity(log_nu_rj, _LOG_NE, _LOG_NI, _LOG_Z, _LOG_T, _GFF)
        assert j_exact == pytest.approx(j_rj, rel=1e-6)

    def test_matches_formula(self):
        expected = np.log(_ff_emissivity_coefficient_cgs) + 2 * _LOG_Z + _LOG_NE + _LOG_NI - 0.5 * _LOG_T + np.log(_GFF)
        result = _log_ff_RJ_emissivity(_LOG_NU, _LOG_NE, _LOG_NI, _LOG_Z, _LOG_T, _GFF)
        assert result == pytest.approx(expected, rel=1e-12)


# ================================================================== #
# TestPrivateRJAbsorption                                            #
# ================================================================== #
class TestPrivateRJAbsorption:
    """_log_ff_RJ_absorption scales as ν^{−2} and T^{−3/2}."""

    def test_nu_minus_two_scaling(self):
        """Doubling ν must decrease log α_RJ by 2 log 2."""
        a1 = _log_ff_RJ_absorption(_LOG_NU, _LOG_NE, _LOG_NI, _LOG_Z, _LOG_T, _GFF)
        a2 = _log_ff_RJ_absorption(np.log(2 * _NU), _LOG_NE, _LOG_NI, _LOG_Z, _LOG_T, _GFF)
        assert a2 - a1 == pytest.approx(-2 * np.log(2), rel=1e-12)

    def test_temperature_minus_three_half_scaling(self):
        """Raising T by 4× must decrease log α_RJ by 3/2 log 4 = 3 log 2."""
        a1 = _log_ff_RJ_absorption(_LOG_NU, _LOG_NE, _LOG_NI, _LOG_Z, _LOG_T, _GFF)
        a2 = _log_ff_RJ_absorption(_LOG_NU, _LOG_NE, _LOG_NI, _LOG_Z, np.log(4 * _T), _GFF)
        assert a2 - a1 == pytest.approx(-1.5 * np.log(4), rel=1e-12)

    def test_z_squared_scaling(self):
        a1 = _log_ff_RJ_absorption(_LOG_NU, _LOG_NE, _LOG_NI, _LOG_Z, _LOG_T, _GFF)
        a2 = _log_ff_RJ_absorption(_LOG_NU, _LOG_NE, _LOG_NI, np.log(2 * _Z), _LOG_T, _GFF)
        assert a2 - a1 == pytest.approx(2 * np.log(2), rel=1e-12)

    def test_matches_formula(self):
        expected = (
            np.log(_ff_absorption_coefficient_cgs)
            + 2 * _LOG_Z
            + _LOG_NE
            + _LOG_NI
            - 1.5 * _LOG_T
            - 2 * _LOG_NU
            + np.log(_GFF)
        )
        result = _log_ff_RJ_absorption(_LOG_NU, _LOG_NE, _LOG_NI, _LOG_Z, _LOG_T, _GFF)
        assert result == pytest.approx(expected, rel=1e-12)

    def test_array_input_produces_array(self):
        log_nu_arr = np.log(np.geomspace(1e8, 1e11, 4))
        result = _log_ff_RJ_absorption(log_nu_arr, _LOG_NE, _LOG_NI, _LOG_Z, _LOG_T, _GFF)
        assert result.shape == (4,)


# ================================================================== #
# TestPrivateWienEmissivity                                          #
# ================================================================== #
class TestPrivateWienEmissivity:
    """_log_ff_Wien_emissivity is mathematically identical to _log_ff_emissivity."""

    def test_identical_to_exact_at_reference(self):
        j_exact = _log_ff_emissivity(_LOG_NU, _LOG_NE, _LOG_NI, _LOG_Z, _LOG_T, _GFF)
        j_wien = _log_ff_Wien_emissivity(_LOG_NU, _LOG_NE, _LOG_NI, _LOG_Z, _LOG_T, _GFF)
        assert j_exact == pytest.approx(j_wien, rel=1e-12)

    def test_identical_across_frequency_range(self):
        log_nu_arr = np.log(np.geomspace(1e8, 1e18, 10))
        j_exact = _log_ff_emissivity(log_nu_arr, _LOG_NE, _LOG_NI, _LOG_Z, _LOG_T, _GFF)
        j_wien = _log_ff_Wien_emissivity(log_nu_arr, _LOG_NE, _LOG_NI, _LOG_Z, _LOG_T, _GFF)
        np.testing.assert_array_equal(j_exact, j_wien)


# ================================================================== #
# TestPrivateWienAbsorption                                          #
# ================================================================== #
class TestPrivateWienAbsorption:
    """_log_ff_Wien_absorption omits the (1 − e^{−hν/kT}) factor,
    giving pure ν^{−3} scaling and converging to the exact form at high ν."""

    def test_nu_minus_three_scaling(self):
        """Wien absorption has exact ν^{−3} dependence at all ν."""
        a1 = _log_ff_Wien_absorption(_LOG_NU, _LOG_NE, _LOG_NI, _LOG_Z, _LOG_T, _GFF)
        a2 = _log_ff_Wien_absorption(np.log(2 * _NU), _LOG_NE, _LOG_NI, _LOG_Z, _LOG_T, _GFF)
        assert a2 - a1 == pytest.approx(-3 * np.log(2), rel=1e-12)

    def test_matches_formula(self):
        expected = (
            np.log(_ff_absorption_coefficient_cgs)
            + 2 * _LOG_Z
            + _LOG_NE
            + _LOG_NI
            - 0.5 * _LOG_T
            - 3 * _LOG_NU
            + np.log(_GFF)
        )
        result = _log_ff_Wien_absorption(_LOG_NU, _LOG_NE, _LOG_NI, _LOG_Z, _LOG_T, _GFF)
        assert result == pytest.approx(expected, rel=1e-12)

    def test_converges_to_exact_in_wien_limit(self):
        """At hν/kT >> 1, (1 − e^{−hν/kT}) ≈ 1, so exact ≈ Wien."""
        log_nu_hi = np.log(1e17)  # hν/kT ≈ 480 for T = 1e4 K
        a_exact = _log_ff_absorption(log_nu_hi, _LOG_NE, _LOG_NI, _LOG_Z, _LOG_T, _GFF)
        a_wien = _log_ff_Wien_absorption(log_nu_hi, _LOG_NE, _LOG_NI, _LOG_Z, _LOG_T, _GFF)
        assert a_exact == pytest.approx(a_wien, rel=1e-6)

    def test_larger_than_exact_at_low_nu(self):
        """Without the (1 − e^{−x}) correction, Wien over-estimates α at low ν."""
        a_exact = _log_ff_absorption(_LOG_NU, _LOG_NE, _LOG_NI, _LOG_Z, _LOG_T, _GFF)
        a_wien = _log_ff_Wien_absorption(_LOG_NU, _LOG_NE, _LOG_NI, _LOG_Z, _LOG_T, _GFF)
        assert a_wien > a_exact  # Wien omits the < 1 factor so it's larger


# ================================================================== #
# TestPublicEmissivity                                               #
# ================================================================== #
class TestPublicEmissivity:
    """compute_ff_emissivity: unit-aware wrapper returning a Quantity."""

    _EMISSIVITY_UNIT = u.erg / (u.s * u.cm**3 * u.Hz * u.sr)

    def test_returns_quantity(self):
        result = compute_ff_emissivity(_NU * u.Hz, _NE / u.cm**3, _NI / u.cm**3, _Z, _T * u.K, _GFF)
        assert isinstance(result, u.Quantity)

    def test_correct_units(self):
        result = compute_ff_emissivity(_NU * u.Hz, _NE / u.cm**3, _NI / u.cm**3, _Z, _T * u.K, _GFF)
        result.to(self._EMISSIVITY_UNIT)  # raises if units are incompatible

    def test_positive(self):
        result = compute_ff_emissivity(_NU * u.Hz, _NE / u.cm**3, _NI / u.cm**3, _Z, _T * u.K, _GFF)
        assert result.value > 0

    def test_bare_float_inputs(self):
        """Bare floats (no Quantity) must work and give the same result."""
        j_q = compute_ff_emissivity(_NU * u.Hz, _NE / u.cm**3, _NI / u.cm**3, _Z, _T * u.K, _GFF)
        j_f = compute_ff_emissivity(_NU, _NE, _NI, _Z, _T, _GFF)
        assert j_q.value == pytest.approx(j_f.value, rel=1e-12)

    def test_unit_conversion_applied(self):
        """Frequency in GHz must give the same result as Hz."""
        j_hz = compute_ff_emissivity(1.0 * u.GHz, _NE / u.cm**3, _NI / u.cm**3, _Z, _T * u.K, _GFF)
        j_ghz = compute_ff_emissivity(1e9 * u.Hz, _NE / u.cm**3, _NI / u.cm**3, _Z, _T * u.K, _GFF)
        assert j_hz.value == pytest.approx(j_ghz.value, rel=1e-12)

    def test_array_vectorisation(self):
        nu_arr = np.geomspace(1e8, 1e12, 20) * u.Hz
        result = compute_ff_emissivity(nu_arr, _NE / u.cm**3, _NI / u.cm**3, _Z, _T * u.K, _GFF)
        assert result.shape == (20,)

    def test_z_squared_scaling(self):
        j1 = compute_ff_emissivity(_NU, _NE, _NI, 1.0, _T, 1.0)
        j2 = compute_ff_emissivity(_NU, _NE, _NI, 2.0, _T, 1.0)
        assert (j2 / j1).decompose().value == pytest.approx(4.0, rel=1e-10)

    def test_ne_linear_scaling(self):
        j1 = compute_ff_emissivity(_NU, 1e3, _NI, _Z, _T, 1.0)
        j2 = compute_ff_emissivity(_NU, 2e3, _NI, _Z, _T, 1.0)
        assert (j2 / j1).decompose().value == pytest.approx(2.0, rel=1e-10)

    def test_ni_linear_scaling(self):
        j1 = compute_ff_emissivity(_NU, _NE, 1e3, _Z, _T, 1.0)
        j2 = compute_ff_emissivity(_NU, _NE, 2e3, _Z, _T, 1.0)
        assert (j2 / j1).decompose().value == pytest.approx(2.0, rel=1e-10)

    def test_temperature_minus_half_scaling(self):
        """Quadrupling T reduces j by factor 2 (T^{−1/2} term dominates at radio ν)."""
        j1 = compute_ff_emissivity(_NU, _NE, _NI, _Z, 1e4, 1.0)
        j2 = compute_ff_emissivity(_NU, _NE, _NI, _Z, 4e4, 1.0)
        assert (j2 / j1).decompose().value == pytest.approx(0.5, rel=1e-3)

    def test_deep_rj_limit_matches_rj(self):
        """At 1 Hz (hν/kT ~ 5e-23) the exact and RJ emissivities must agree."""
        j_exact = compute_ff_emissivity(1.0 * u.Hz, _NE / u.cm**3, _NI / u.cm**3, _Z, _T * u.K, _GFF)
        j_rj = compute_ff_RJ_emissivity(1.0 * u.Hz, _NE / u.cm**3, _NI / u.cm**3, _Z, _T * u.K, _GFF)
        assert j_exact.value == pytest.approx(j_rj.value, rel=1e-6)

    def test_matches_private_backend(self):
        """Public function must agree with exp(private backend)."""
        log_j = _log_ff_emissivity(_LOG_NU, _LOG_NE, _LOG_NI, _LOG_Z, _LOG_T, _GFF)
        j_pub = compute_ff_emissivity(_NU, _NE, _NI, _Z, _T, _GFF)
        assert j_pub.value == pytest.approx(np.exp(log_j), rel=1e-12)


# ================================================================== #
# TestPublicAbsorption                                               #
# ================================================================== #
class TestPublicAbsorption:
    """compute_ff_absorption: unit-aware wrapper returning a Quantity."""

    _ABSORPTION_UNIT = u.cm**-1

    def test_returns_quantity(self):
        result = compute_ff_absorption(_NU * u.Hz, _NE / u.cm**3, _NI / u.cm**3, _Z, _T * u.K, _GFF)
        assert isinstance(result, u.Quantity)

    def test_correct_units(self):
        result = compute_ff_absorption(_NU * u.Hz, _NE / u.cm**3, _NI / u.cm**3, _Z, _T * u.K, _GFF)
        result.to(self._ABSORPTION_UNIT)

    def test_positive(self):
        result = compute_ff_absorption(_NU * u.Hz, _NE / u.cm**3, _NI / u.cm**3, _Z, _T * u.K, _GFF)
        assert result.value > 0

    def test_bare_float_inputs(self):
        a_q = compute_ff_absorption(_NU * u.Hz, _NE / u.cm**3, _NI / u.cm**3, _Z, _T * u.K, _GFF)
        a_f = compute_ff_absorption(_NU, _NE, _NI, _Z, _T, _GFF)
        assert a_q.value == pytest.approx(a_f.value, rel=1e-12)

    def test_unit_conversion_applied(self):
        a_hz = compute_ff_absorption(1.0 * u.GHz, _NE / u.cm**3, _NI / u.cm**3, _Z, _T * u.K, _GFF)
        a_ghz = compute_ff_absorption(1e9 * u.Hz, _NE / u.cm**3, _NI / u.cm**3, _Z, _T * u.K, _GFF)
        assert a_hz.value == pytest.approx(a_ghz.value, rel=1e-12)

    def test_array_vectorisation(self):
        nu_arr = np.geomspace(1e8, 1e12, 20) * u.Hz
        result = compute_ff_absorption(nu_arr, _NE / u.cm**3, _NI / u.cm**3, _Z, _T * u.K, _GFF)
        assert result.shape == (20,)

    def test_z_squared_scaling(self):
        a1 = compute_ff_absorption(_NU, _NE, _NI, 1.0, _T, 1.0)
        a2 = compute_ff_absorption(_NU, _NE, _NI, 2.0, _T, 1.0)
        assert (a2 / a1).decompose().value == pytest.approx(4.0, rel=1e-10)

    def test_ne_linear_scaling(self):
        a1 = compute_ff_absorption(_NU, 1e3, _NI, _Z, _T, 1.0)
        a2 = compute_ff_absorption(_NU, 2e3, _NI, _Z, _T, 1.0)
        assert (a2 / a1).decompose().value == pytest.approx(2.0, rel=1e-10)

    def test_nu_minus_three_scaling_in_wien_limit(self):
        """At very high ν, the stimulated-emission factor → 1 and α ∝ ν^{−3}."""
        nu1, nu2 = 1e17, 2e17  # hν/kT >> 1 for T = 1e4 K
        a1 = compute_ff_absorption(nu1, _NE, _NI, _Z, _T, 1.0)
        a2 = compute_ff_absorption(nu2, _NE, _NI, _Z, _T, 1.0)
        assert (a2 / a1).decompose().value == pytest.approx((nu1 / nu2) ** 3, rel=1e-4)

    def test_wien_limit_matches_wien_backend(self):
        """At high ν, the public exact function must agree with the Wien backend."""
        log_nu_hi = np.log(1e17)
        a_pub = compute_ff_absorption(1e17, _NE, _NI, _Z, _T, _GFF)
        a_wien = np.exp(_log_ff_Wien_absorption(log_nu_hi, _LOG_NE, _LOG_NI, _LOG_Z, _LOG_T, _GFF))
        assert a_pub.value == pytest.approx(a_wien, rel=1e-6)

    def test_matches_private_backend(self):
        log_a = _log_ff_absorption(_LOG_NU, _LOG_NE, _LOG_NI, _LOG_Z, _LOG_T, _GFF)
        a_pub = compute_ff_absorption(_NU, _NE, _NI, _Z, _T, _GFF)
        assert a_pub.value == pytest.approx(np.exp(log_a), rel=1e-12)


# ================================================================== #
# TestPublicRJEmissivity                                             #
# ================================================================== #
class TestPublicRJEmissivity:
    """compute_ff_RJ_emissivity: frequency-independent, T^{−1/2}, correct units."""

    def test_returns_quantity(self):
        result = compute_ff_RJ_emissivity(_NU * u.Hz, _NE / u.cm**3, _NI / u.cm**3, _Z, _T * u.K, _GFF)
        assert isinstance(result, u.Quantity)

    def test_correct_units(self):
        result = compute_ff_RJ_emissivity(_NU * u.Hz, _NE / u.cm**3, _NI / u.cm**3, _Z, _T * u.K, _GFF)
        result.to(u.erg / (u.s * u.cm**3 * u.Hz * u.sr))

    def test_positive(self):
        result = compute_ff_RJ_emissivity(_NU, _NE, _NI, _Z, _T, _GFF)
        assert result.value > 0

    def test_frequency_independent(self):
        """Two very different frequencies must give identical results."""
        j1 = compute_ff_RJ_emissivity(1e6 * u.Hz, _NE / u.cm**3, _NI / u.cm**3, _Z, _T * u.K, _GFF)
        j2 = compute_ff_RJ_emissivity(1e14 * u.Hz, _NE / u.cm**3, _NI / u.cm**3, _Z, _T * u.K, _GFF)
        assert j1.value == pytest.approx(j2.value, rel=1e-12)

    def test_temperature_minus_half_scaling(self):
        j1 = compute_ff_RJ_emissivity(_NU, _NE, _NI, _Z, 1e4, 1.0)
        j2 = compute_ff_RJ_emissivity(_NU, _NE, _NI, _Z, 4e4, 1.0)
        assert (j2 / j1).decompose().value == pytest.approx(0.5, rel=1e-12)

    def test_matches_private_backend(self):
        log_j = _log_ff_RJ_emissivity(_LOG_NU, _LOG_NE, _LOG_NI, _LOG_Z, _LOG_T, _GFF)
        j_pub = compute_ff_RJ_emissivity(_NU, _NE, _NI, _Z, _T, _GFF)
        assert j_pub.value == pytest.approx(np.exp(log_j), rel=1e-12)

    def test_array_vectorisation(self):
        nu_arr = np.geomspace(1e8, 1e12, 10) * u.Hz
        result = compute_ff_RJ_emissivity(nu_arr, _NE / u.cm**3, _NI / u.cm**3, _Z, _T * u.K, _GFF)
        assert result.shape == (10,)
        # All values equal (frequency-independent)
        np.testing.assert_allclose(result.value, result.value[0], rtol=1e-12)


# ================================================================== #
# TestPublicRJAbsorption                                             #
# ================================================================== #
class TestPublicRJAbsorption:
    """compute_ff_RJ_absorption: ν^{−2}, T^{−3/2}, correct units."""

    def test_returns_quantity(self):
        result = compute_ff_RJ_absorption(_NU * u.Hz, _NE / u.cm**3, _NI / u.cm**3, _Z, _T * u.K, _GFF)
        assert isinstance(result, u.Quantity)

    def test_correct_units(self):
        result = compute_ff_RJ_absorption(_NU * u.Hz, _NE / u.cm**3, _NI / u.cm**3, _Z, _T * u.K, _GFF)
        result.to(u.cm**-1)

    def test_positive(self):
        result = compute_ff_RJ_absorption(_NU, _NE, _NI, _Z, _T, _GFF)
        assert result.value > 0

    def test_nu_minus_two_scaling(self):
        """Doubling ν must halve α_RJ (ν^{−2})."""
        a1 = compute_ff_RJ_absorption(1e9, _NE, _NI, _Z, _T, 1.0)
        a2 = compute_ff_RJ_absorption(2e9, _NE, _NI, _Z, _T, 1.0)
        assert (a2 / a1).decompose().value == pytest.approx(0.25, rel=1e-10)

    def test_temperature_minus_three_half_scaling(self):
        """Raising T by 4× must reduce α_RJ by 4^{3/2} = 8."""
        a1 = compute_ff_RJ_absorption(_NU, _NE, _NI, _Z, 1e4, 1.0)
        a2 = compute_ff_RJ_absorption(_NU, _NE, _NI, _Z, 4e4, 1.0)
        assert (a2 / a1).decompose().value == pytest.approx(1.0 / 8.0, rel=1e-10)

    def test_z_squared_scaling(self):
        a1 = compute_ff_RJ_absorption(_NU, _NE, _NI, 1.0, _T, 1.0)
        a2 = compute_ff_RJ_absorption(_NU, _NE, _NI, 2.0, _T, 1.0)
        assert (a2 / a1).decompose().value == pytest.approx(4.0, rel=1e-10)

    def test_bare_float_inputs(self):
        a_q = compute_ff_RJ_absorption(_NU * u.Hz, _NE / u.cm**3, _NI / u.cm**3, _Z, _T * u.K, _GFF)
        a_f = compute_ff_RJ_absorption(_NU, _NE, _NI, _Z, _T, _GFF)
        assert a_q.value == pytest.approx(a_f.value, rel=1e-12)

    def test_matches_private_backend(self):
        log_a = _log_ff_RJ_absorption(_LOG_NU, _LOG_NE, _LOG_NI, _LOG_Z, _LOG_T, _GFF)
        a_pub = compute_ff_RJ_absorption(_NU, _NE, _NI, _Z, _T, _GFF)
        assert a_pub.value == pytest.approx(np.exp(log_a), rel=1e-12)

    def test_array_vectorisation(self):
        nu_arr = np.geomspace(1e8, 1e11, 15) * u.Hz
        result = compute_ff_RJ_absorption(nu_arr, _NE / u.cm**3, _NI / u.cm**3, _Z, _T * u.K, _GFF)
        assert result.shape == (15,)
