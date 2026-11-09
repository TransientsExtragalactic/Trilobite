"""
Tests for the synchrotron cooling engines in trilobite.radiation.synchrotron.cooling.

Design
------
A non-collected base class ``CoolingEngineTestBase`` holds shared physics-correctness
tests that apply to every ``SynchrotronCoolingEngine`` subclass:

  - Known analytic values (RATE_CASES, TIME_CASES, GAMMA_CASES)
  - Positivity of all outputs
  - Round-trip consistency:  t_cool(gamma_c(t)) ≈ t
  - Energy budget identity:  rate × time = gamma × m_e c²
  - Scaling laws:            rate ∝ γ²,  time ∝ 1/γ,  gamma_c ∝ 1/t

Concrete subclasses add engine-specific tests:

  TestSynchrotronRadiativeCoolingEngine
      - PA-averaged vs. fixed-pitch-angle branches
      - RuntimeWarning when gamma_c < 1
      - Characteristic frequency
      - Public compute_* unit return types and unit coercion

  TestInverseComptonCoolingEngine
      - L_bol / R² energy-density scaling
      - Characteristic frequency composition
      - RuntimeWarning when gamma_IC < 1
      - Public compute_* unit return types and unit coercion
"""

import numpy as np
import pytest
from astropy import constants
from astropy import units as u

from trilobite.radiation.synchrotron.cooling import (
    InverseComptonCoolingEngine,
    SynchrotronRadiativeCoolingEngine,
)
from trilobite.radiation.synchrotron.core import _opt_compute_synch_frequency

# ------------------------------------------------------------------ #
# Physics constants (independent of trilobite's own constant defs)   #
# ------------------------------------------------------------------ #
_sigma_T = constants.sigma_T.cgs.value
_c_cgs = constants.c.cgs.value
_m_e = constants.m_e.cgs.value
_m_e_c2 = _m_e * _c_cgs**2

# ------------------------------------------------------------------ #
# Expected-value coefficients derived from first principles          #
# ------------------------------------------------------------------ #
# Synchrotron — pitch-angle-averaged
_SYNCH_PA_RATE_COEFF = _sigma_T * _c_cgs / (9 * np.pi)
_SYNCH_PA_TIME_COEFF = 9 * np.pi * _m_e * _c_cgs / _sigma_T
# Synchrotron — specific pitch angle
_SYNCH_SA_RATE_COEFF = _sigma_T * _c_cgs / (6 * np.pi)
_SYNCH_SA_TIME_COEFF = 6 * np.pi * _m_e * _c_cgs / _sigma_T
# Inverse Compton
_IC_RATE_COEFF = _sigma_T / (3 * np.pi)
_IC_TIME_COEFF = 3 * np.pi * _m_e * _c_cgs**2 / _sigma_T

# ------------------------------------------------------------------ #
# Case lists for SynchrotronRadiativeCoolingEngine                   #
# ------------------------------------------------------------------ #
_SYNCH_RATE_CASES = [
    ({"B": 1.0, "gamma": 1000.0}, _SYNCH_PA_RATE_COEFF * 1.0**2 * 1000.0**2),
    ({"B": 0.1, "gamma": 500.0}, _SYNCH_PA_RATE_COEFF * 0.1**2 * 500.0**2),
]
_SYNCH_TIME_CASES = [
    ({"B": 1.0, "gamma": 1000.0}, _SYNCH_PA_TIME_COEFF / (1.0**2 * 1000.0)),
    ({"B": 0.1, "gamma": 500.0}, _SYNCH_PA_TIME_COEFF / (0.1**2 * 500.0)),
]
_SYNCH_GAMMA_CASES = [
    ({"B": 1.0, "t": _SYNCH_PA_TIME_COEFF / (1.0**2 * 1000.0)}, 1000.0),
    ({"B": 0.1, "t": _SYNCH_PA_TIME_COEFF / (0.1**2 * 500.0)}, 500.0),
]
_SYNCH_FIELD_KWARGS = {"B": 1.0}
_SYNCH_GAMMA_REF = 1000.0
_SYNCH_T_REF = _SYNCH_PA_TIME_COEFF / (1.0**2 * _SYNCH_GAMMA_REF)

# ------------------------------------------------------------------ #
# Case lists for InverseComptonCoolingEngine                         #
# ------------------------------------------------------------------ #
_IC_RATE_CASES = [
    ({"L_bol": 1e43, "R": 1e15, "gamma": 1000.0}, _IC_RATE_COEFF * (1e43 / 1e15**2) * 1000.0**2),
    ({"L_bol": 1e44, "R": 1e16, "gamma": 500.0}, _IC_RATE_COEFF * (1e44 / 1e16**2) * 500.0**2),
]
_IC_TIME_CASES = [
    ({"L_bol": 1e43, "R": 1e15, "gamma": 1000.0}, _IC_TIME_COEFF * (1e15**2 / 1e43) / 1000.0),
    ({"L_bol": 1e44, "R": 1e16, "gamma": 500.0}, _IC_TIME_COEFF * (1e16**2 / 1e44) / 500.0),
]
_IC_GAMMA_CASES = [
    ({"L_bol": 1e43, "R": 1e15, "t": _IC_TIME_COEFF * (1e15**2 / 1e43) / 1000.0}, 1000.0),
    ({"L_bol": 1e44, "R": 1e16, "t": _IC_TIME_COEFF * (1e16**2 / 1e44) / 500.0}, 500.0),
]
_IC_FIELD_KWARGS = {"L_bol": 1e43, "R": 1e15}
_IC_GAMMA_REF = 1000.0
_IC_T_REF = _IC_TIME_COEFF * (1e15**2 / 1e43) / _IC_GAMMA_REF

RTOL = 1e-6


# ================================================================== #
# Base test class (not collected by pytest — no "Test" prefix)       #
# ================================================================== #
class CoolingEngineTestBase:
    """
    Shared physics-correctness tests for all SynchrotronCoolingEngine subclasses.

    Subclasses must set:
      RATE_CASES   list of (kwargs, expected_rate)
      TIME_CASES   list of (kwargs, expected_time)
      GAMMA_CASES  list of (kwargs, expected_gamma_c)
      FIELD_KWARGS dict of field parameters without gamma or t
      GAMMA_REF    float — reference Lorentz factor
      T_REF        float — reference dynamical time (s)

    and define ``make_engine()`` returning a concrete engine instance.
    """

    RATE_CASES = []
    TIME_CASES = []
    GAMMA_CASES = []
    FIELD_KWARGS = {}
    GAMMA_REF = None
    T_REF = None

    @classmethod
    def make_engine(cls):
        raise NotImplementedError

    # ---- Known-value tests ---------------------------------------- #

    def test_rate_known_values(self):
        engine = self.make_engine()
        for kwargs, expected in self.RATE_CASES:
            result = engine._compute_cooling_rate(**kwargs)
            assert np.isclose(result, expected, rtol=RTOL), (
                f"_compute_cooling_rate(**{kwargs}) = {result:.6e}, expected {expected:.6e}"
            )

    def test_time_known_values(self):
        engine = self.make_engine()
        for kwargs, expected in self.TIME_CASES:
            result = engine._compute_cooling_time(**kwargs)
            assert np.isclose(result, expected, rtol=RTOL), (
                f"_compute_cooling_time(**{kwargs}) = {result:.6e}, expected {expected:.6e}"
            )

    def test_gamma_known_values(self):
        engine = self.make_engine()
        for kwargs, expected in self.GAMMA_CASES:
            result = engine._compute_cooling_gamma(**kwargs)
            assert np.isclose(result, expected, rtol=RTOL), (
                f"_compute_cooling_gamma(**{kwargs}) = {result:.6e}, expected {expected:.6e}"
            )

    # ---- Positivity ----------------------------------------------- #

    def test_positivity(self):
        engine = self.make_engine()
        for kwargs, _ in self.RATE_CASES:
            assert engine._compute_cooling_rate(**kwargs) > 0
        for kwargs, _ in self.TIME_CASES:
            assert engine._compute_cooling_time(**kwargs) > 0
        for kwargs, _ in self.GAMMA_CASES:
            assert engine._compute_cooling_gamma(**kwargs) > 0

    # ---- Round-trip consistency ------------------------------------ #

    def test_round_trip_time_gamma(self):
        """t_cool(gamma_c(t)) ≈ t for the canonical field configuration."""
        engine = self.make_engine()
        gamma_c = engine._compute_cooling_gamma(**self.FIELD_KWARGS, t=self.T_REF)
        t_back = engine._compute_cooling_time(**self.FIELD_KWARGS, gamma=gamma_c)
        assert np.isclose(t_back, self.T_REF, rtol=RTOL), (
            f"Round-trip failed: t_back={t_back:.6e}, T_REF={self.T_REF:.6e}"
        )

    # ---- Energy budget -------------------------------------------- #

    def test_energy_budget(self):
        """rate × time = gamma × m_e c², by definition of the cooling time."""
        engine = self.make_engine()
        rate = engine._compute_cooling_rate(**self.FIELD_KWARGS, gamma=self.GAMMA_REF)
        time = engine._compute_cooling_time(**self.FIELD_KWARGS, gamma=self.GAMMA_REF)
        expected = self.GAMMA_REF * _m_e_c2
        assert np.isclose(rate * time, expected, rtol=RTOL), (
            f"Energy budget: rate*time={rate * time:.6e}, gamma*m_e*c²={expected:.6e}"
        )

    # ---- Scaling laws --------------------------------------------- #

    def test_rate_scales_as_gamma_squared(self):
        engine = self.make_engine()
        r1 = engine._compute_cooling_rate(**self.FIELD_KWARGS, gamma=self.GAMMA_REF)
        r2 = engine._compute_cooling_rate(**self.FIELD_KWARGS, gamma=2.0 * self.GAMMA_REF)
        assert np.isclose(r2 / r1, 4.0, rtol=RTOL)

    def test_time_scales_as_inverse_gamma(self):
        engine = self.make_engine()
        t1 = engine._compute_cooling_time(**self.FIELD_KWARGS, gamma=self.GAMMA_REF)
        t2 = engine._compute_cooling_time(**self.FIELD_KWARGS, gamma=2.0 * self.GAMMA_REF)
        assert np.isclose(t2 / t1, 0.5, rtol=RTOL)

    def test_gamma_c_scales_as_inverse_t(self):
        engine = self.make_engine()
        g1 = engine._compute_cooling_gamma(**self.FIELD_KWARGS, t=self.T_REF)
        g2 = engine._compute_cooling_gamma(**self.FIELD_KWARGS, t=2.0 * self.T_REF)
        assert np.isclose(g2 / g1, 0.5, rtol=RTOL)


# ================================================================== #
# SynchrotronRadiativeCoolingEngine                                  #
# ================================================================== #
class TestSynchrotronRadiativeCoolingEngine(CoolingEngineTestBase):
    RATE_CASES = _SYNCH_RATE_CASES
    TIME_CASES = _SYNCH_TIME_CASES
    GAMMA_CASES = _SYNCH_GAMMA_CASES
    FIELD_KWARGS = _SYNCH_FIELD_KWARGS
    GAMMA_REF = _SYNCH_GAMMA_REF
    T_REF = _SYNCH_T_REF

    @classmethod
    def make_engine(cls):
        return SynchrotronRadiativeCoolingEngine()

    # ---- Pitch-angle branch --------------------------------------- #

    @pytest.mark.parametrize(
        "B,gamma,sin_alpha",
        [
            (1.0, 1000.0, 1.0),
            (0.5, 300.0, np.sqrt(0.5)),
        ],
    )
    def test_sin_alpha_rate_vs_pa(self, B, gamma, sin_alpha):
        """Fixed-pitch-angle rate = (3/2) × PA rate × sin²α, by coefficient ratio."""
        engine = self.make_engine()
        rate_pa = engine._compute_cooling_rate(B=B, gamma=gamma)
        rate_sa = engine._compute_cooling_rate(B=B, gamma=gamma, sin_alpha=sin_alpha)
        expected_ratio = (3.0 / 2.0) * sin_alpha**2
        assert np.isclose(rate_sa / rate_pa, expected_ratio, rtol=RTOL)

    @pytest.mark.parametrize(
        "B,gamma,sin_alpha",
        [
            (1.0, 1000.0, 1.0),
            (0.5, 300.0, np.sqrt(0.5)),
        ],
    )
    def test_sin_alpha_time_vs_pa(self, B, gamma, sin_alpha):
        """Fixed-pitch-angle time = (2/3) × PA time / sin²α, by coefficient ratio."""
        engine = self.make_engine()
        time_pa = engine._compute_cooling_time(B=B, gamma=gamma)
        time_sa = engine._compute_cooling_time(B=B, gamma=gamma, sin_alpha=sin_alpha)
        expected_ratio = (2.0 / 3.0) / sin_alpha**2
        assert np.isclose(time_sa / time_pa, expected_ratio, rtol=RTOL)

    @pytest.mark.parametrize(
        "B,gamma",
        [
            (1.0, 1000.0),
            (2.0, 500.0),
        ],
    )
    def test_sin_alpha_rate_known_values(self, B, gamma):
        """Pin the sin_alpha=1 rate against the analytic formula."""
        engine = self.make_engine()
        result = engine._compute_cooling_rate(B=B, gamma=gamma, sin_alpha=1.0)
        expected = _SYNCH_SA_RATE_COEFF * B**2 * gamma**2
        assert np.isclose(result, expected, rtol=RTOL)

    @pytest.mark.parametrize(
        "B,gamma",
        [
            (1.0, 1000.0),
            (2.0, 500.0),
        ],
    )
    def test_sin_alpha_time_known_values(self, B, gamma):
        """Pin the sin_alpha=1 cooling time against the analytic formula."""
        engine = self.make_engine()
        result = engine._compute_cooling_time(B=B, gamma=gamma, sin_alpha=1.0)
        expected = _SYNCH_SA_TIME_COEFF / (B**2 * gamma)
        assert np.isclose(result, expected, rtol=RTOL)

    # ---- Characteristic frequency --------------------------------- #

    def test_characteristic_frequency_positive(self):
        engine = self.make_engine()
        nu = engine._compute_characteristic_frequency(B=1.0, gamma=1000.0, sin_alpha=1.0)
        assert nu > 0

    def test_characteristic_frequency_scales_as_gamma_squared(self):
        engine = self.make_engine()
        nu1 = engine._compute_characteristic_frequency(B=1.0, gamma=1000.0)
        nu2 = engine._compute_characteristic_frequency(B=1.0, gamma=2000.0)
        assert np.isclose(nu2 / nu1, 4.0, rtol=RTOL)

    def test_characteristic_frequency_scales_as_B(self):
        engine = self.make_engine()
        nu1 = engine._compute_characteristic_frequency(B=1.0, gamma=1000.0)
        nu2 = engine._compute_characteristic_frequency(B=2.0, gamma=1000.0)
        assert np.isclose(nu2 / nu1, 2.0, rtol=RTOL)

    # ---- Unphysical gamma_c warning ------------------------------- #

    def test_compute_cooling_gamma_warns_when_unphysical(self):
        """Very strong field + long time drives gamma_c < 1; expect RuntimeWarning."""
        engine = self.make_engine()
        with pytest.warns(RuntimeWarning, match="gamma_c < 1"):
            engine.compute_cooling_gamma(B=1000.0, t=1e6)

    # ---- Public interface: unit output ---------------------------- #

    def test_compute_cooling_rate_returns_quantity(self):
        engine = self.make_engine()
        result = engine.compute_cooling_rate(B=1.0, gamma=1000.0)
        assert isinstance(result, u.Quantity)
        assert result.unit.is_equivalent(u.erg / u.s)

    def test_compute_cooling_time_returns_quantity(self):
        engine = self.make_engine()
        result = engine.compute_cooling_time(B=1.0, gamma=1000.0)
        assert isinstance(result, u.Quantity)
        assert result.unit.is_equivalent(u.s)

    def test_compute_cooling_gamma_returns_scalar(self):
        engine = self.make_engine()
        result = engine.compute_cooling_gamma(B=1.0, t=1e6)
        assert np.isscalar(result) or isinstance(result, np.ndarray)

    def test_compute_characteristic_frequency_returns_quantity(self):
        engine = self.make_engine()
        result = engine.compute_characteristic_frequency(B=1.0, gamma=1000.0)
        assert isinstance(result, u.Quantity)
        assert result.unit.is_equivalent(u.Hz)

    def test_B_unit_coercion(self):
        """B supplied as Tesla gives the same numeric result as Gauss."""
        engine = self.make_engine()
        rate_G = engine.compute_cooling_rate(B=1.0 * u.G, gamma=1000.0)
        rate_T = engine.compute_cooling_rate(B=1e-4 * u.T, gamma=1000.0)
        assert np.isclose(rate_G.value, rate_T.to(u.erg / u.s).value, rtol=RTOL)


# ================================================================== #
# InverseComptonCoolingEngine                                        #
# ================================================================== #
class TestInverseComptonCoolingEngine(CoolingEngineTestBase):
    RATE_CASES = _IC_RATE_CASES
    TIME_CASES = _IC_TIME_CASES
    GAMMA_CASES = _IC_GAMMA_CASES
    FIELD_KWARGS = _IC_FIELD_KWARGS
    GAMMA_REF = _IC_GAMMA_REF
    T_REF = _IC_T_REF

    @classmethod
    def make_engine(cls):
        return InverseComptonCoolingEngine()

    # ---- Radiation-field scaling ---------------------------------- #

    @pytest.mark.parametrize("factor", [2.0, 10.0])
    def test_rate_scales_linearly_with_L_bol(self, factor):
        engine = self.make_engine()
        r1 = engine._compute_cooling_rate(L_bol=1e43, R=1e15, gamma=1000.0)
        r2 = engine._compute_cooling_rate(L_bol=factor * 1e43, R=1e15, gamma=1000.0)
        assert np.isclose(r2 / r1, factor, rtol=RTOL)

    @pytest.mark.parametrize("factor", [2.0, 10.0])
    def test_rate_scales_as_inverse_R_squared(self, factor):
        engine = self.make_engine()
        r1 = engine._compute_cooling_rate(L_bol=1e43, R=1e15, gamma=1000.0)
        r2 = engine._compute_cooling_rate(L_bol=1e43, R=factor * 1e15, gamma=1000.0)
        assert np.isclose(r2 / r1, 1.0 / factor**2, rtol=RTOL)

    # ---- Characteristic frequency composition -------------------- #

    def test_characteristic_frequency_composition(self):
        """
        _compute_characteristic_frequency must equal
        _opt_compute_synch_frequency(gamma_IC(L, R, t), B).
        """
        engine = self.make_engine()
        B, L_bol, R, t = 1.0, 1e43, 1e15, 1e3
        nu = engine._compute_characteristic_frequency(B=B, L_bol=L_bol, R=R, t=t, sin_alpha=1.0)
        gamma_IC = engine._compute_cooling_gamma(L_bol=L_bol, R=R, t=t)
        nu_expected = _opt_compute_synch_frequency(gamma=gamma_IC, B=B, sin_alpha=1.0, pitch_average=True)
        assert np.isclose(nu, nu_expected, rtol=RTOL)

    def test_characteristic_frequency_positive(self):
        engine = self.make_engine()
        nu = engine._compute_characteristic_frequency(B=1.0, L_bol=1e43, R=1e15, t=1e3, sin_alpha=1.0)
        assert nu > 0

    # ---- Unphysical gamma_IC warning ------------------------------ #

    def test_compute_cooling_gamma_warns_when_unphysical(self):
        """Extreme luminosity + short time drives gamma_IC < 1; expect RuntimeWarning."""
        engine = self.make_engine()
        with pytest.warns(RuntimeWarning, match="gamma_IC < 1"):
            engine.compute_cooling_gamma(L_bol=1e55, R=1e15, t=1e6)

    # ---- Public interface: unit output ---------------------------- #

    def test_compute_cooling_rate_returns_quantity(self):
        engine = self.make_engine()
        result = engine.compute_cooling_rate(L_bol=1e43, R=1e15, gamma=1000.0)
        assert isinstance(result, u.Quantity)
        assert result.unit.is_equivalent(u.erg / u.s)

    def test_compute_cooling_time_returns_quantity(self):
        engine = self.make_engine()
        result = engine.compute_cooling_time(L_bol=1e43, R=1e15, gamma=1000.0)
        assert isinstance(result, u.Quantity)
        assert result.unit.is_equivalent(u.s)

    def test_compute_cooling_gamma_returns_scalar(self):
        engine = self.make_engine()
        result = engine.compute_cooling_gamma(L_bol=1e43, R=1e15, t=1e3)
        assert np.isscalar(result) or isinstance(result, np.ndarray)

    def test_compute_characteristic_frequency_returns_quantity(self):
        engine = self.make_engine()
        result = engine.compute_characteristic_frequency(B=1.0, L_bol=1e43, R=1e15, t=1e3)
        assert isinstance(result, u.Quantity)
        assert result.unit.is_equivalent(u.Hz)

    def test_unit_coercion_L_bol(self):
        """L_bol supplied as L_sun gives the same numeric result as erg/s."""
        engine = self.make_engine()
        L_cgs = 1e43
        L_sun_val = (L_cgs * (u.erg / u.s)).to(u.L_sun).value
        t1 = engine.compute_cooling_time(L_bol=L_cgs, R=1e15, gamma=1000.0)
        t2 = engine.compute_cooling_time(L_bol=L_sun_val * u.L_sun, R=1e15, gamma=1000.0)
        assert np.isclose(t1.value, t2.to(u.s).value, rtol=RTOL)
