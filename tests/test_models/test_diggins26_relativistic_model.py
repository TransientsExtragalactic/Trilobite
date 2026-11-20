"""Tests for :class:`~trilobite.models.LFBOTs.Diggins26_RelativisticTwoComponentModel`."""

import numpy as np
import pytest
from astropy import constants as const
from astropy import units as u

from trilobite.dynamics.shocks.numerical import RelPressureDrivenThinShellShockEngine
from trilobite.models.core.status import ModelStatus
from trilobite.models.LFBOTs import (
    Diggins26_RelativisticTwoComponentModel,
    Diggins26_TwoComponentModel,
)
from trilobite.physics_utils.constants import c_cgs, m_e_cgs, m_p_cgs
from trilobite.physics_utils.maxwell_juettner import _opt_mj_exact_mean_kinetic_energy, _opt_mj_log_pdf
from trilobite.physics_utils.thermal_partition import compute_ion_electron_theta

DAY = 86400.0
M_SUN = const.M_sun.cgs.value

TIMES = np.array([5.0, 20.0, 80.0, 300.0]) * DAY
FREQS = np.array([1e9, 5e9, 2e10, 1e11, 3e11])

# No free-free absorption (the ionized CSM ends inside the shock), so that the electrons set the spectrum.
TRANSPARENT = {"R_ion": 1e10}


@pytest.fixture(scope="module")
def model():
    return Diggins26_RelativisticTwoComponentModel(luminosity_distance=60 * u.Mpc)


@pytest.fixture(scope="module")
def parent():
    return Diggins26_TwoComponentModel(luminosity_distance=60 * u.Mpc)


# =========================================================================== #
# Construction                                                                #
# =========================================================================== #
class TestConstruction:
    def test_uses_the_relativistic_engine_with_the_adiabatic_index(self):
        m = Diggins26_RelativisticTwoComponentModel(redshift=0.01, adiabatic_index=1.4)
        assert isinstance(m._shock_engine, RelPressureDrivenThinShellShockEngine)
        assert m._shock_engine._gamma_ad_1 == m._shock_engine._gamma_ad_4 == 1.4

    def test_thermal_parameter_follows_epsilon_B(self, model):
        names = [p.name for p in model.PARAMETERS]
        assert names.index("epsilon_T") == names.index("epsilon_B") + 1
        # Every parameter of the parent is kept, in the same order, so the two models share a parameter vocabulary.
        assert [n for n in names if n != "epsilon_T"] == [p.name for p in Diggins26_TwoComponentModel.PARAMETERS]

    def test_status_is_declared(self):
        assert Diggins26_RelativisticTwoComponentModel.STATUS is ModelStatus.NOVEL
        assert "STATUS" in vars(Diggins26_RelativisticTwoComponentModel)

    def test_registered_for_serialization(self, model):
        rebuilt = Diggins26_RelativisticTwoComponentModel(**model._model_init_kwargs)
        assert isinstance(rebuilt, Diggins26_RelativisticTwoComponentModel)
        assert rebuilt.redshift == pytest.approx(model.redshift)
        assert rebuilt._integrator_kw == model._integrator_kw

    def test_integrator_kw_reaches_the_solver_without_gamma(self, monkeypatch):
        """The relativistic engine takes its adiabatic indices at construction and rejects a per-call ``gamma``."""
        m = Diggins26_RelativisticTwoComponentModel(redshift=0.01, integrator_kw={"atol": 1e-3})
        seen = {}
        original = m._shock_engine._compute_shock_properties_cgs

        def spy(*args, **kwargs):
            seen.update(kwargs)
            return original(*args, **kwargs)

        monkeypatch.setattr(m._shock_engine, "_compute_shock_properties_cgs", spy)
        m.forward_model({"time": 20 * DAY, "frequency": 5e9}, {})
        assert seen["atol"] == 1e-3 and seen["rtol"] == 1e-7 and "gamma" not in seen


# =========================================================================== #
# Shock dynamics                                                              #
# =========================================================================== #
class TestDynamics:
    def test_reports_the_relativistic_state(self, model):
        dyn = model.compute_dynamics({}, TIMES)
        for sector in (dyn.equatorial, dyn.polar):
            assert sector.lorentz_factor is not None and np.all(sector.lorentz_factor >= 1.0)
            assert sector.density.unit == u.g / u.cm**3
            assert np.all(sector.density.value > 0)

    def test_nonrelativistic_parent_has_no_relativistic_fields(self, parent):
        sector = parent.compute_dynamics({}, TIMES).equatorial
        assert sector.density is None and sector.lorentz_factor is None

    def test_agrees_with_the_nonrelativistic_model_for_slow_ejecta(self, model, parent):
        """At v_0 = 0.1c the two solvers differ at order beta^2 only."""
        rel = model.compute_dynamics({}, TIMES)
        nonrel = parent.compute_dynamics({}, TIMES)
        for sector in ("equatorial", "polar"):
            a, b = getattr(rel, sector), getattr(nonrel, sector)
            for field in ("radius", "velocity", "mass", "thermal_energy_density"):
                assert getattr(a, field).value == pytest.approx(getattr(b, field).value, rel=0.015)

    def test_departs_from_the_nonrelativistic_model_as_the_ejecta_speed_up(self, model, parent):
        """The relativistic correction grows with v_0 (here 0.1c, 0.3c and 0.6c)."""
        departures = []
        for M_ej in (0.1, 0.01, 0.0025):
            p = {"M_ej": M_ej * M_SUN}
            a = model.compute_dynamics(p, TIMES).equatorial.radius.value
            b = parent.compute_dynamics(p, TIMES).equatorial.radius.value
            departures.append(np.max(np.abs(a / b - 1)))
        assert departures[0] < departures[1] < departures[2]

    def test_fast_ejecta_are_mildly_relativistic(self, model):
        """A light-CSM shell with v_0 = 0.67c stays at beta ~ 0.6 (Gamma ~ 1.2) while it sweeps up little gas."""
        dyn = model.compute_dynamics({"M_ej": 0.0025 * M_SUN, "rho_int": 1e-18, "rho_ext": 1e-20}, TIMES)
        assert np.all(dyn.equatorial.lorentz_factor > 1.2)


# =========================================================================== #
# Electrons                                                                   #
# =========================================================================== #
class TestElectrons:
    @pytest.fixture(scope="class")
    def setup(self, model):
        p = model._coerce({"epsilon_T": 0.3, "gamma_min": 1.0})
        t_rest = model._rest_times(np.array([3.0, 12.0, 50.0]) * DAY)
        states, _ = model._dynamics_cgs(p, t_rest)
        return p, t_rest, states[0], model._gamma_grid(p, model._gamma_points_per_decade)

    def test_electron_energy_is_the_budgeted_fraction(self, model, setup):
        """Thermal plus non-thermal electrons carry (epsilon_e + thermal share) of the thermal energy density."""
        p, t_rest, state, gamma = setup
        micro = model._microphysics(p, state, t_rest, gamma)
        _, log_weights = model._quadrature(gamma)

        energy = (np.exp(micro.log_N) * (gamma - 1.0) * np.exp(log_weights)).sum(axis=1) * m_e_cgs * c_cgs**2
        thermal_share = p["epsilon_T"] * (1 - p["epsilon_B"] - p["epsilon_e"]) / (1 + p["epsilon_T"])
        assert energy == pytest.approx((p["epsilon_e"] + thermal_share) * state.eth, rel=5e-3)

    def test_thermal_electrons_have_the_expected_temperature_and_density(self, model, setup):
        p, t_rest, state, gamma = setup
        micro = model._microphysics(p, state, t_rest, gamma)

        _, theta = compute_ion_electron_theta(
            state.eth, state.density, model.mean_molecular_weight_per_electron, 0.3, p["epsilon_B"], p["epsilon_e"]
        )
        np.testing.assert_allclose(micro.theta_e, theta, rtol=1e-12)
        # All electrons are thermal in the thermal component: n = rho / (mu_e m_p).
        np.testing.assert_allclose(
            micro.n_thermal, state.density / (model.mean_molecular_weight_per_electron * m_p_cgs), rtol=1e-10
        )
        # The thermal electrons' mean kinetic energy reproduces their budget.
        thermal_share = 0.3 * (1 - p["epsilon_B"] - p["epsilon_e"]) / 1.3
        np.testing.assert_allclose(
            micro.n_thermal * _opt_mj_exact_mean_kinetic_energy(micro.theta_e) * m_e_cgs * c_cgs**2,
            thermal_share * state.eth,
            rtol=1e-8,
        )

    def test_zero_epsilon_T_is_the_parents_electron_distribution(self, model, parent, setup):
        p, t_rest, state, gamma = setup
        p = dict(p, epsilon_T=0.0)
        rel = model._microphysics(p, state, t_rest, gamma)
        assert rel.theta_e is None and rel.n_thermal is None

        # The non-thermal electrons are the parent's on the support, and absent outside it.
        parent_micro = parent._microphysics(p, state, t_rest, gamma)
        np.testing.assert_allclose(rel.N0, parent_micro.N0, rtol=1e-12)
        # (The two grids differ, so compare where the parent is defined: it can drop an edge node that rounds to just
        # outside the support, which the relativistic model keeps.)
        defined = np.isfinite(parent_micro.log_N)
        assert np.all(np.isfinite(rel.log_N[defined]))
        np.testing.assert_allclose(rel.log_N[defined], parent_micro.log_N[defined], rtol=1e-12)
        assert np.sum(~np.isfinite(rel.log_N)) <= np.sum(~defined)

    def test_thermal_electrons_add_to_the_power_law(self, model, setup):
        p, t_rest, state, gamma = setup
        with_thermal = model._microphysics(p, state, t_rest, gamma).log_N
        without = model._microphysics(dict(p, epsilon_T=0.0), state, t_rest, gamma).log_N
        assert np.all(with_thermal >= without - 1e-12)
        assert np.any(with_thermal > without + 1.0)

    def test_power_law_edge_is_resolved_and_thermal_electrons_extend_below_it(self, model):
        """With gamma_min > 1 the power law jumps on at gamma_min, but the thermal population fills gamma < gamma_min."""
        p = model._coerce({"gamma_min": 3.0, "epsilon_T": 1.0})
        t_rest = model._rest_times(np.array([5.0, 40.0]) * DAY)
        states, _ = model._dynamics_cgs(p, t_rest)
        gamma = model._gamma_grid(p, model._gamma_points_per_decade)

        below = gamma < 3.0
        assert below.sum() > 30  # the thermal population is resolved below the power law's edge
        assert np.any(np.isclose(gamma, 3.0, rtol=1e-12))  # an edge node
        assert np.any(np.isclose(gamma, 3.0 * (1 - 1e-9), rtol=1e-12))  # and the node just below the jump

        micro = model._microphysics(p, states[0], t_rest, gamma)
        pure_thermal = _opt_mj_log_pdf(gamma[None, :], micro.theta_e[:, None], micro.n_thermal[:, None])
        # Below gamma_min the power law is off, so the distribution is the thermal one alone; at the edge it is on.
        np.testing.assert_allclose(micro.log_N[:, below], pure_thermal[:, below], rtol=1e-6)
        edge = np.argmin(np.abs(gamma - 3.0))
        assert np.all(micro.log_N[:, edge] > pure_thermal[:, edge] + 1e-3)

    def test_grid_resolves_the_spectrum(self, model):
        """Quadrupling the Lorentz-factor resolution changes the flux by well under a percent."""
        fine = Diggins26_RelativisticTwoComponentModel(
            luminosity_distance=60 * u.Mpc, gamma_points_per_decade=4 * model._gamma_points_per_decade
        )
        p = {**TRANSPARENT, "epsilon_B": 0.1, "epsilon_T": 1.0}
        nu = np.array([1e8, 1e9, 1e10, 1e11])
        a = model.tabulate(TIMES[:2], nu, p).flux_density.value
        b = fine.tabulate(TIMES[:2], nu, p).flux_density.value
        np.testing.assert_allclose(a, b, rtol=5e-3)


# =========================================================================== #
# Radiation                                                                   #
# =========================================================================== #
class TestRadiation:
    def test_zero_epsilon_T_agrees_with_the_nonrelativistic_model(self, model, parent):
        """Without thermal electrons the two models differ only through the shock solver."""
        a = model.tabulate(TIMES, FREQS, {"epsilon_T": 0.0}).flux_density.value
        b = parent.tabulate(TIMES, FREQS, {}).flux_density.value
        # The earliest epochs at the highest frequencies are exponentially suppressed, so allow for a floor.
        np.testing.assert_allclose(a, b, rtol=0.03, atol=1e-20)

    def test_thermal_electrons_brighten_the_low_frequency_emission(self, model):
        p = {**TRANSPARENT, "epsilon_B": 0.1}
        nu = np.array([3e8])
        off = model.tabulate(TIMES[:2], nu, {**p, "epsilon_T": 0.0}).flux_density.value
        on = model.tabulate(TIMES[:2], nu, {**p, "epsilon_T": 1.0}).flux_density.value
        assert np.all(on > 1.2 * off)

    def test_tabulate_reports_the_thermal_electrons(self, model, parent):
        tab = model.tabulate(TIMES, FREQS, {"epsilon_T": 0.2})
        for sector in (tab.equatorial, tab.polar):
            assert sector.thermal_temperature.shape == TIMES.shape
            assert np.all(sector.thermal_temperature > 0)
            assert sector.thermal_electron_density.unit == u.cm**-3
        assert model.tabulate(TIMES, FREQS, {"epsilon_T": 0.0}).equatorial.thermal_temperature is None
        assert parent.tabulate(TIMES, FREQS, {}).equatorial.thermal_temperature is None

    def test_total_is_sum_of_attenuated_sectors(self, model):
        tab = model.tabulate(TIMES, FREQS, {})
        total = tab.equatorial.flux_density_attenuated + tab.polar.flux_density_attenuated
        np.testing.assert_allclose(tab.flux_density.value, total.value, rtol=1e-12)

    def test_paired_matches_tabulate(self, model):
        tab = model.tabulate(TIMES, FREQS, {})
        T, N = np.meshgrid(TIMES, FREQS, indexing="ij")
        paired = model.forward_model({"time": T.ravel(), "frequency": N.ravel()}, {}).flux_density.value
        np.testing.assert_allclose(paired.reshape(T.shape), tab.flux_density.value, rtol=1e-10)

    def test_reused_dynamics_allows_changed_thermal_microphysics(self, model):
        dyn = model.compute_dynamics({}, TIMES)
        a = model.tabulate(TIMES, FREQS, {"epsilon_T": 0.5}, dynamics=dyn).flux_density.value
        b = model.tabulate(TIMES, FREQS, {"epsilon_T": 0.5}).flux_density.value
        np.testing.assert_allclose(a, b, rtol=1e-12)

    def test_dynamics_of_the_nonrelativistic_model_are_rejected(self, model, parent):
        """The thermal electrons need the post-shock density, which only the relativistic solver reports."""
        with pytest.raises(ValueError, match="relativistic solver"):
            model.tabulate(TIMES, FREQS, {}, dynamics=parent.compute_dynamics({}, TIMES))

    def test_output_units_and_finiteness(self, model):
        out = model.forward_model({"time": TIMES, "frequency": np.full_like(TIMES, 5e9)}, {})
        assert out.flux_density.unit == u.Jy
        assert np.all(np.isfinite(out.flux_density.value)) and np.all(out.flux_density.value >= 0)


# =========================================================================== #
# Infeasible parameters                                                       #
# =========================================================================== #
class TestInfeasibility:
    def test_default_parameters_are_feasible(self, model):
        assert model.diagnose({}, TIMES) is None

    @pytest.mark.parametrize(
        "parameters,fragment",
        [
            ({"epsilon_B": 0.6, "epsilon_e": 0.5}, "epsilon_B + epsilon_e"),  # the ions would carry negative energy
            ({"M_ej": 1e29}, "max_initial_beta"),  # inherited from the parent
        ],
    )
    def test_nan_flux_and_diagnosis(self, model, parameters, fragment):
        out = model.forward_model({"time": TIMES, "frequency": np.full_like(TIMES, 5e9)}, parameters)
        assert np.all(np.isnan(out.flux_density.value))
        reason = model.diagnose(parameters, TIMES)
        assert reason is not None and fragment.lower() in reason.lower()

    def test_negative_epsilon_T_raises(self, model):
        with pytest.raises(ValueError):
            model.forward_model({"time": 20 * DAY, "frequency": 5e9}, {"epsilon_T": -0.1})
