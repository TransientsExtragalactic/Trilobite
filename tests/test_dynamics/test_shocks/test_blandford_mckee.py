r"""
Tests for the Blandford--McKee ultra-relativistic blastwave shock engine.

The tests are organized into four groups that validate different aspects of the
theory:

1. **Energy conservation** — Γ²R^{3−k} = (17−4k)E/(8πKc²) for k ∈ {0, 1, 2}.
2. **Self-similar time evolution** — Γ ∝ t^{−(3−k)/2} from the logarithmic
   time slope of the returned Lorentz factors.
3. **Post-shock field consistency** — γ₂_lab = Γ/√2, p₂ ≈ (2/3)ρ_ext c²Γ²
   (exact in the Γ→∞ limit, with a ±O(1/Γ) correction from the exact RH
   conditions), e_comoving = 3p₂ (ultra-relativistic EOS),
   ρ₂_lab = γ₂_lab × ρ₂_comoving, and T^{00} = (e_tot + p₂)γ₂² − p₂.
4. **Interface and validation** — k ≥ 3 raises ``ValueError``, low-Γ emits
   ``RuntimeWarning``, the wind specialization is consistent with the general
   engine, and ``normalize_csm_density`` inverts correctly.

**Choice of K_csm per k:**

The Blandford--McKee solution assumes Γ ≫ 1. In float64, β = √(1−1/Γ²)
rounds to 1.0 when Γ ≳ 1/√ε_mach ≈ 6.7×10⁷, which causes the RH solver to
produce NaN. Each test therefore uses a ``K_csm`` value tuned to keep
Γ ∈ [10, 10⁴] over the chosen time grid:

+------+---------------------------------------+------------------------------+
| k    | K_csm                                 | Physical interpretation      |
+======+=======================================+==============================+
| 0.0  | m_p ≈ 1.67×10⁻²⁴ g cm⁻³             | n = 1 cm⁻³ uniform ISM       |
+------+---------------------------------------+------------------------------+
| 1.0  | 1×10⁻⁴ g cm⁻²                        | Generic power-law medium     |
+------+---------------------------------------+------------------------------+
| 2.0  | 5×10¹¹ g cm⁻¹                        | Stellar-wind normalization   |
+------+---------------------------------------+------------------------------+
"""

import warnings

import astropy.constants as const
import numpy as np
import pytest
from astropy import units as u
from numpy.testing import assert_allclose

from triceratops.dynamics.shocks import BlandfordMcKeeShockEngine, BlandfordMcKeeWindShockEngine

# --------------------------------------------------------------------------- #
# Module-level constants (CGS)                                                #
# --------------------------------------------------------------------------- #
_C_CGS = const.c.cgs.value
_MP_CGS = const.m_p.cgs.value

# Standard explosion energy.
_E_ERG = 1e52

# k-specific (K_csm, times) pairs that keep Γ in [10, 1e4] over the grid.
# Each entry: (k, K_csm [CGS g cm^{k-3}], times [seconds], description).
_K_TABLE = {
    0.0: (_MP_CGS, np.logspace(3, 7, 40)),  # 1e3–1e7 s, Γ ~ 1.3e7–13
    1.0: (1e-4, np.logspace(2, 5, 40)),  # 1e2–1e5 s, Γ ~ 2.5e5–7900
    2.0: (5e11, np.logspace(1, 4, 40)),  # 1e1–1e4 s, Γ ~ 1.6e4–520
}


def _bm_const(k, E, K):
    """Expected value of Γ²R^{3-k} from BM energy conservation."""
    return (17.0 - 4.0 * k) * E / (8.0 * np.pi * K * _C_CGS**2)


# --------------------------------------------------------------------------- #
# Shared fixture                                                               #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def engine():
    return BlandfordMcKeeShockEngine()


# =========================================================================== #
# 1  Energy conservation: Γ²R^{3−k} = (17−4k)E/(8πKc²)                      #
# =========================================================================== #


class TestEnergyConservation:
    r"""Verify the BM energy integral Γ²R^{3−k} = C_E(k)×E/(Kc²) at all times.

    Since :math:`R = ct` exactly (the defining assumption of the ultra-relativistic
    limit), the relation reduces to an explicit algebraic identity that must hold
    to floating-point precision for any finite time.
    """

    @pytest.mark.parametrize("k", [0.0, 1.0, 2.0])
    def test_conservation_constant_all_times(self, engine, k):
        """Γ²R^{3-k} equals the analytic BM constant at every time step."""
        K, times = _K_TABLE[k]
        state = engine._compute_shock_properties_cgs(time=times, E=_E_ERG, K_csm=K, k=k)
        expected = _bm_const(k, _E_ERG, K)
        computed = state.lorentz_factor**2 * state.radius ** (3.0 - k)
        assert_allclose(computed, expected, rtol=1e-12)

    @pytest.mark.parametrize("k", [0.0, 1.0, 2.0])
    def test_normalization_coefficient(self, engine, k):
        """C_E(k) = 8π/(17-4k) is exactly recovered from the returned state."""
        K, times = _K_TABLE[k]
        state = engine._compute_shock_properties_cgs(time=times, E=_E_ERG, K_csm=K, k=k)
        C_E_analytic = 8.0 * np.pi / (17.0 - 4.0 * k)
        C_E_recovered = _E_ERG / (K * _C_CGS**2 * state.lorentz_factor**2 * state.radius ** (3 - k))
        assert_allclose(C_E_recovered, C_E_analytic, rtol=1e-12)

    def test_k0_exact_bm_formula(self, engine):
        """k=0: Γ²= 17E/(8πKc²R³), the BM paper result verbatim."""
        K, times = _K_TABLE[0.0]
        state = engine._compute_shock_properties_cgs(time=times, E=_E_ERG, K_csm=K, k=0.0)
        expected = 17.0 * _E_ERG / (8.0 * np.pi * K * _C_CGS**2 * state.radius**3)
        assert_allclose(state.lorentz_factor**2, expected, rtol=1e-12)

    def test_k2_exact_bm_formula(self, engine):
        """k=2: Γ²R = 9E/(8πKc²), the BM wind result."""
        K, times = _K_TABLE[2.0]
        state = engine._compute_shock_properties_cgs(time=times, E=_E_ERG, K_csm=K, k=2.0)
        expected = 9.0 * _E_ERG / (8.0 * np.pi * K * _C_CGS**2)
        assert_allclose(state.lorentz_factor**2 * state.radius, expected, rtol=1e-12)


# =========================================================================== #
# 2  Self-similar time evolution: Γ ∝ t^{−(3−k)/2}                           #
# =========================================================================== #


class TestTimePowerLaw:
    r"""Verify the BM deceleration law Γ ∝ t^{−(3−k)/2}.

    Because :math:`R = ct` exactly, the Lorentz factor satisfies

    .. math::

        \Gamma(t) \propto t^{-(3-k)/2}

    and its logarithmic slope must equal :math:`-(3-k)/2` at every interior
    grid point (edge points are excluded to avoid finite-difference artefacts).
    """

    @pytest.mark.parametrize("k", [0.0, 1.0, 2.0])
    def test_lorentz_factor_log_slope(self, engine, k):
        """d(lnΓ)/d(lnt) = −(3−k)/2 holds at all interior grid points."""
        K, times = _K_TABLE[k]
        state = engine._compute_shock_properties_cgs(time=times, E=_E_ERG, K_csm=K, k=k)
        log_slope = np.gradient(np.log(state.lorentz_factor), np.log(times))
        expected = -(3.0 - k) / 2.0
        assert_allclose(log_slope[2:-2], expected, rtol=1e-4)

    @pytest.mark.parametrize("k", [0.0, 1.0, 2.0])
    def test_radius_equals_ct(self, engine, k):
        """R = ct identically (the defining ultra-relativistic assumption)."""
        K, times = _K_TABLE[k]
        state = engine._compute_shock_properties_cgs(time=times, E=_E_ERG, K_csm=K, k=k)
        assert_allclose(state.radius, _C_CGS * times, rtol=1e-14)

    @pytest.mark.parametrize("k", [0.0, 1.0, 2.0])
    def test_velocity_equals_beta_c(self, engine, k):
        """v = βc identically."""
        K, times = _K_TABLE[k]
        state = engine._compute_shock_properties_cgs(time=times, E=_E_ERG, K_csm=K, k=k)
        assert_allclose(state.velocity, state.beta * _C_CGS, rtol=1e-14)

    @pytest.mark.parametrize("k", [0.0, 1.0, 2.0])
    def test_beta_strictly_between_0_and_1(self, engine, k):
        """0 < β < 1 at all times."""
        K, times = _K_TABLE[k]
        state = engine._compute_shock_properties_cgs(time=times, E=_E_ERG, K_csm=K, k=k)
        assert np.all(state.beta > 0.0)
        assert np.all(state.beta < 1.0)

    @pytest.mark.parametrize("k", [0.0, 1.0, 2.0])
    def test_lorentz_factor_decreasing(self, engine, k):
        """Γ decreases monotonically with time (blastwave is decelerating)."""
        K, times = _K_TABLE[k]
        state = engine._compute_shock_properties_cgs(time=times, E=_E_ERG, K_csm=K, k=k)
        assert np.all(np.diff(state.lorentz_factor) < 0.0)


# =========================================================================== #
# 3  Post-shock field consistency                                              #
# =========================================================================== #


class TestPostShockConsistency:
    r"""Verify post-shock fields are mutually consistent with BM and RH theory.

    At the shock front (:math:`\chi = 1`) the BM similarity functions satisfy
    :math:`f(1) = g(1) = h(1) = 1`, giving exact results for a pure
    ultra-relativistic shock:

    - :math:`\gamma_{2,\mathrm{lab}}^2 = \tfrac{1}{2}\Gamma^2`
    - :math:`p_2 \approx \tfrac{2}{3}\rho_{\mathrm{ext}}(R)\,c^2\Gamma^2`
      (asymptotically exact; O(1/Γ) corrections from the finite-Γ RH solution)
    - :math:`U_{\mathrm{int},2} = 3p_2` (:math:`\hat\gamma = 4/3` EOS)
    - :math:`\rho_{2,\mathrm{lab}} = \gamma_{2,\mathrm{lab}}\,\rho_{2,\mathrm{proper}}`
    - :math:`T^{00} = (e_2 + p_2)\,\gamma_{2,\mathrm{lab}}^2 - p_2`
    """

    @pytest.fixture(scope="class")
    def state_k0_early(self):
        r"""State at moderate–high Γ for the pressure formula test.

        Uses t ∈ [3×10⁴, 2×10⁵] s (8 h – 2.3 days) so that
        Γ ∈ [4 600, 27 000].  In this window:

        - The O(1/Γ) correction to p₂ = (2/3)ρc²Γ² is ≲ 5×10⁻⁵,
          so ``rtol=1e-4`` verifies the asymptotic formula without
          noise from the correction itself.
        - The RH solver's catastrophic-cancellation error in
          γ₁ = 1/√(1−β₁²) scales as ε_mach × Γ² ≈ 2×10⁻⁶,
          far below the tolerance threshold.

        Very early times (t ≲ 350 s) give Γ² > 1/ε_mach ≈ 4.5×10¹⁵,
        causing 1/Γ² to underflow to zero in float64 and producing NaN.
        Times in the intermediate window (t ~ 1 000–20 000 s) have
        ε_mach × Γ² ≳ 10⁻⁴ and should be avoided for tight tolerances.
        """
        eng = BlandfordMcKeeShockEngine()
        K = _MP_CGS
        # Γ ≈ 27 000, 17 000, 8 500, 4 600 at these four times.
        times = np.array([3e4, 5e4, 1e5, 2e5])  # seconds
        return eng._compute_shock_properties_cgs(time=times, E=_E_ERG, K_csm=K, k=0.0)

    @pytest.fixture(scope="class")
    def state_k0(self):
        K, times = _K_TABLE[0.0]
        eng = BlandfordMcKeeShockEngine()
        return eng._compute_shock_properties_cgs(time=times, E=_E_ERG, K_csm=K, k=0.0)

    # --- Exact relations (hold for any Γ > 1) --------------------------------

    def test_fluid_lorentz_factor_is_gamma_over_sqrt2(self, state_k0):
        """γ₂_lab = Γ/√2 exactly from BM profile g(χ=1) = 1."""
        assert_allclose(
            state_k0.fluid_lorentz_factor,
            state_k0.lorentz_factor / np.sqrt(2.0),
            rtol=1e-12,
        )

    def test_thermal_energy_density_comoving_is_3p(self, state_k0):
        """U_int = 3p₂ for the ultra-relativistic EOS (γ̂ = 4/3)."""
        assert_allclose(
            state_k0.thermal_energy_density_comoving,
            3.0 * state_k0.post_shock_pressure,
            rtol=1e-10,
        )

    def test_lab_density_is_gamma2_times_comoving(self, state_k0):
        """ρ₂_lab = γ₂_lab × ρ₂_proper by Lorentz transformation."""
        assert_allclose(
            state_k0.post_shock_lab_density,
            state_k0.fluid_lorentz_factor * state_k0.post_shock_comoving_density,
            rtol=1e-12,
        )

    def test_lab_energy_density_stress_tensor(self, state_k0):
        """T^{00} = (e_tot + p₂)γ₂² − p₂ holds to RH precision."""
        e_tot = state_k0.thermal_energy_density_comoving + state_k0.post_shock_comoving_density * _C_CGS**2
        expected = (
            e_tot + state_k0.post_shock_pressure
        ) * state_k0.fluid_lorentz_factor**2 - state_k0.post_shock_pressure
        assert_allclose(state_k0.thermal_energy_density_lab, expected, rtol=1e-10)

    # --- Asymptotic BM formula (valid for Γ >> 1) ----------------------------

    def test_post_shock_pressure_bm_formula_high_gamma(self, state_k0_early):
        r"""p₂ = (2/3)ρ_ext c²Γ² exactly.

        The BM engine computes pressure directly from Γ² rather than via the
        RH solver, so the result is exact to machine precision for all Γ.
        """
        state = state_k0_early
        rho_ext = _MP_CGS * state.radius**0.0  # k=0
        expected = (2.0 / 3.0) * rho_ext * _C_CGS**2 * state.lorentz_factor**2
        assert_allclose(state.post_shock_pressure, expected, rtol=1e-12)

    @pytest.mark.parametrize("k", [0.0, 1.0, 2.0])
    def test_post_shock_pressure_bm_formula_moderate_gamma(self, engine, k):
        r"""p₂ ≈ (2/3)ρ_ext c²Γ² to 5% at moderate Γ.

        The 1/Γ correction to the BM pressure formula is 3√2/(4Γ). For
        Γ ∼ 10, the relative deviation is ~10%; hence we only require 5%
        agreement here to confirm the asymptotic formula is approached.
        """
        K, times = _K_TABLE[k]
        # Use early times only, where Γ is largest.
        early_times = times[:10]
        state = engine._compute_shock_properties_cgs(time=early_times, E=_E_ERG, K_csm=K, k=k)
        rho_ext = K * state.radius ** (-k)
        expected = (2.0 / 3.0) * rho_ext * _C_CGS**2 * state.lorentz_factor**2
        assert_allclose(state.post_shock_pressure, expected, rtol=0.05)

    # --- Ordering and finiteness checks ---------------------------------------

    def test_lab_density_exceeds_comoving(self, state_k0):
        """ρ₂_lab > ρ₂_proper (Lorentz boost always increases rest-mass density)."""
        assert np.all(state_k0.post_shock_lab_density > state_k0.post_shock_comoving_density)

    def test_lab_energy_exceeds_comoving_thermal(self, state_k0):
        """T^{00} > U_int (lab-frame energy includes bulk kinetic energy)."""
        assert np.all(state_k0.thermal_energy_density_lab > state_k0.thermal_energy_density_comoving)

    def test_temperature_positive(self, state_k0):
        """Post-shock temperature is strictly positive."""
        assert np.all(state_k0.post_shock_temperature > 0.0)

    def test_all_fields_finite(self, state_k0):
        """All state fields are finite at every time step."""
        for field in state_k0._fields:
            vals = getattr(state_k0, field)
            assert np.all(np.isfinite(vals)), f"Field '{field}' contains non-finite values"

    @pytest.mark.parametrize("k", [0.0, 1.0, 2.0])
    def test_all_fields_finite_general_k(self, engine, k):
        """All fields are finite for each supported k value."""
        K, times = _K_TABLE[k]
        state = engine._compute_shock_properties_cgs(time=times, E=_E_ERG, K_csm=K, k=k)
        for field in state._fields:
            vals = getattr(state, field)
            assert np.all(np.isfinite(vals)), f"Field '{field}' non-finite for k={k}"


# =========================================================================== #
# 4  Interface, validation, and specializations                               #
# =========================================================================== #


class TestInterface:
    """Input handling, validation, unit conversions, and the wind subclass."""

    def test_k_geq_3_raises_value_error(self, engine):
        """Public interface raises ValueError for k ≥ 3."""
        K, _ = _K_TABLE[0.0]
        with pytest.raises(ValueError, match="k must be less than 3"):
            engine.compute_shock_properties(
                time=1.0 * u.day,
                E=_E_ERG * u.erg,
                K_csm=K * u.g / u.cm**3,
                k=3.0,
            )

    def test_k_above_3_also_raises(self, engine):
        """Public interface raises ValueError for k = 4 as well."""
        K, _ = _K_TABLE[0.0]
        with pytest.raises(ValueError):
            engine.compute_shock_properties(
                time=1.0 * u.day,
                E=_E_ERG * u.erg,
                K_csm=K * u.g / u.cm**3,
                k=4.0,
            )

    def test_low_lorentz_factor_warning(self):
        """RuntimeWarning emitted when Γ drops below lorentz_warn_threshold."""
        eng = BlandfordMcKeeShockEngine(lorentz_warn_threshold=5.0)
        with pytest.warns(RuntimeWarning, match="Lorentz factor"):
            eng._compute_shock_properties_cgs(
                time=np.array([1e10]),  # ~317 years — deep in non-UR regime
                E=1e44,
                K_csm=_MP_CGS,
                k=0.0,
            )

    def test_no_warning_when_gamma_high(self):
        """No RuntimeWarning when Γ is comfortably above threshold."""
        eng = BlandfordMcKeeShockEngine(lorentz_warn_threshold=2.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            K, _ = _K_TABLE[0.0]
            eng._compute_shock_properties_cgs(
                time=np.array([86400.0]),  # 1 day, Γ ~ 1.6e4
                E=_E_ERG,
                K_csm=K,
                k=0.0,
            )

    def test_scalar_time_input(self, engine):
        """Scalar time input is accepted without error."""
        K, _ = _K_TABLE[0.0]
        state = engine._compute_shock_properties_cgs(time=86400.0, E=_E_ERG, K_csm=K, k=0.0)
        assert np.ndim(state.radius) == 0

    def test_array_time_input_shape(self, engine):
        """Array time input returns arrays of matching shape."""
        K, _ = _K_TABLE[0.0]
        times = np.array([1.0, 2.0, 3.0]) * 86400.0
        state = engine._compute_shock_properties_cgs(time=times, E=_E_ERG, K_csm=K, k=0.0)
        assert state.radius.shape == (3,)
        assert state.lorentz_factor.shape == (3,)

    def test_unit_bearing_inputs_match_cgs(self, engine):
        """Astropy-unit and bare-float inputs agree to numerical precision."""
        K, _ = _K_TABLE[0.0]
        t_s = 10.0 * 86400.0

        state_cgs = engine._compute_shock_properties_cgs(time=t_s, E=_E_ERG, K_csm=K, k=0.0)
        state_q = engine.compute_shock_properties(
            time=10.0 * u.day,
            E=_E_ERG * u.erg,
            K_csm=K * u.g / u.cm**3,
            k=0.0,
        )

        assert_allclose(state_q.radius.to(u.cm).value, state_cgs.radius, rtol=1e-12)
        assert_allclose(state_q.lorentz_factor, state_cgs.lorentz_factor, rtol=1e-12)

    def test_public_interface_attaches_units(self, engine):
        """Physical fields have astropy units; dimensionless fields are bare."""
        K, _ = _K_TABLE[0.0]
        state = engine.compute_shock_properties(time=1.0 * u.day, E=_E_ERG * u.erg, K_csm=K * u.g / u.cm**3, k=0.0)
        for field in (
            "radius",
            "velocity",
            "post_shock_pressure",
            "post_shock_temperature",
            "post_shock_comoving_density",
            "post_shock_lab_density",
            "thermal_energy_density_comoving",
            "thermal_energy_density_lab",
        ):
            assert isinstance(getattr(state, field), u.Quantity), f"Field '{field}' should be a Quantity"
        for field in ("lorentz_factor", "beta", "fluid_lorentz_factor"):
            assert not isinstance(getattr(state, field), u.Quantity), (
                f"Field '{field}' should be dimensionless (bare float/array)"
            )


class TestWindEngine:
    """BlandfordMcKeeWindShockEngine consistency with the general engine at k=2."""

    @pytest.fixture(scope="class")
    def wind_params(self):
        M_dot_cgs = (1e-5 * u.Msun / u.yr).to(u.g / u.s).value
        v_wind_cgs = (1000.0 * u.km / u.s).to(u.cm / u.s).value
        K_csm_cgs = M_dot_cgs / (4.0 * np.pi * v_wind_cgs)
        return dict(M_dot=M_dot_cgs, v_wind=v_wind_cgs, K_csm=K_csm_cgs)

    def test_wind_matches_general_engine_at_k2(self, wind_params):
        """Wind engine gives identical kinematics to the general engine with k=2."""
        _, times = _K_TABLE[2.0]
        general = BlandfordMcKeeShockEngine()
        wind = BlandfordMcKeeWindShockEngine()

        state_g = general._compute_shock_properties_cgs(time=times, E=_E_ERG, K_csm=wind_params["K_csm"], k=2.0)
        state_w = wind._compute_shock_properties_cgs(
            time=times,
            E=_E_ERG,
            M_dot=wind_params["M_dot"],
            v_wind=wind_params["v_wind"],
        )

        assert_allclose(state_w.lorentz_factor, state_g.lorentz_factor, rtol=1e-12)
        assert_allclose(state_w.radius, state_g.radius, rtol=1e-12)
        assert_allclose(state_w.post_shock_pressure, state_g.post_shock_pressure, rtol=1e-12)

    def test_wind_public_interface_accepts_quantities(self, wind_params):
        """Wind public interface accepts Quantity inputs and returns Quantities."""
        wind = BlandfordMcKeeWindShockEngine()
        state = wind.compute_shock_properties(
            time=np.array([1.0, 10.0]) * u.day,
            E=1e52 * u.erg,
            M_dot=1e-5 * u.Msun / u.yr,
            v_wind=1000.0 * u.km / u.s,
        )
        assert isinstance(state.radius, u.Quantity)
        assert np.all(state.lorentz_factor > 1.0)

    def test_wind_energy_conservation_k2(self, wind_params):
        """Wind engine satisfies Γ²R = 9E/(8πKc²) (the k=2 BM identity)."""
        _, times = _K_TABLE[2.0]
        wind = BlandfordMcKeeWindShockEngine()
        state = wind._compute_shock_properties_cgs(
            time=times,
            E=_E_ERG,
            M_dot=wind_params["M_dot"],
            v_wind=wind_params["v_wind"],
        )
        K = wind_params["K_csm"]
        expected = 9.0 * _E_ERG / (8.0 * np.pi * K * _C_CGS**2)
        assert_allclose(state.lorentz_factor**2 * state.radius, expected, rtol=1e-12)


class TestNormalizeCsmDensity:
    """BlandfordMcKeeShockEngine.normalize_csm_density utility."""

    def test_k0_recovers_rho0(self):
        """For k=0, K = ρ₀ (r₀ drops out)."""
        rho0 = 1.67e-24 * u.g / u.cm**3
        r0 = 1e17 * u.cm
        K = BlandfordMcKeeShockEngine.normalize_csm_density(rho0, r0, k=0.0)
        assert_allclose(K.to(u.g / u.cm**3).value, rho0.to(u.g / u.cm**3).value, rtol=1e-14)

    @pytest.mark.parametrize("k", [0.0, 1.0, 2.0])
    def test_roundtrip_density(self, k):
        """ρ(r₀) = K × r₀^{−k} recovers ρ₀ exactly."""
        rho0 = 5e-24 * u.g / u.cm**3
        r0 = 1e16 * u.cm
        K = BlandfordMcKeeShockEngine.normalize_csm_density(rho0, r0, k=k)
        rho_recovered = K * r0 ** (-k)
        assert_allclose(
            rho_recovered.to(u.g / u.cm**3).value,
            rho0.to(u.g / u.cm**3).value,
            rtol=1e-14,
        )

    @pytest.mark.parametrize("k", [0.0, 2.0])
    def test_units_correct(self, k):
        """Returned K has units equivalent to g cm^{k-3}."""
        K = BlandfordMcKeeShockEngine.normalize_csm_density(1e-24 * u.g / u.cm**3, 1e16 * u.cm, k=k)
        assert K.unit.is_equivalent(u.g * u.cm ** (k - 3))
