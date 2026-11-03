"""Tests for :class:`~trilobite.models.LFBOTs.Diggins26_TwoComponentModel`."""

import numpy as np
import pytest
from astropy import constants as const
from astropy import units as u
from scipy.integrate import simpson

from trilobite.data.core import InferenceData, Observable
from trilobite.dynamics.profiles.csm import AxisymmetricCSMDensityProfile
from trilobite.inference import InferenceProblem, LogGaussianCensoredLikelihood
from trilobite.models.LFBOTs import (
    Diggins26_TwoComponentModel,
    Diggins26Dynamics,
    Diggins26Tabulation,
)

DAY = 86400.0
M_P = const.m_p.cgs.value
M_E = const.m_e.cgs.value
C = const.c.cgs.value
SIGMA_T = const.sigma_T.cgs.value


@pytest.fixture(scope="module")
def model():
    return Diggins26_TwoComponentModel(luminosity_distance=60 * u.Mpc)


# A modest grid keeps the tests quick: 4 epochs and 5 frequencies.
TIMES = np.array([5.0, 20.0, 80.0, 300.0]) * DAY
FREQS = np.array([1e9, 5e9, 2e10, 1e11, 3e11])


# =========================================================================== #
# Construction                                                                #
# =========================================================================== #
class TestConstruction:
    def test_requires_exactly_one_distance(self):
        with pytest.raises(ValueError, match="Exactly one"):
            Diggins26_TwoComponentModel()
        with pytest.raises(ValueError, match="Exactly one"):
            Diggins26_TwoComponentModel(redshift=0.01, luminosity_distance=60 * u.Mpc)
        with pytest.raises(ValueError, match="Exactly one"):
            Diggins26_TwoComponentModel(luminosity_distance=60 * u.Mpc, angular_diameter_distance=55 * u.Mpc)

    def test_angular_diameter_distance_constructor(self, model):
        m = Diggins26_TwoComponentModel(angular_diameter_distance=model.angular_diameter_distance)
        assert m.redshift == pytest.approx(model.redshift, rel=1e-6)
        assert m.luminosity_distance.to_value(u.cm) == pytest.approx(model.luminosity_distance.to_value(u.cm), rel=1e-6)
        bare = Diggins26_TwoComponentModel(angular_diameter_distance=model.angular_diameter_distance.to_value(u.Mpc))
        assert bare.redshift == pytest.approx(model.redshift, rel=1e-6)

    def test_distances_are_consistent(self, model):
        assert model.luminosity_distance.to_value(u.Mpc) == pytest.approx(60.0)
        z = model.redshift
        assert 0.012 < z < 0.015
        ratio = (model.luminosity_distance / model.angular_diameter_distance).to_value(u.dimensionless_unscaled)
        assert ratio == pytest.approx((1 + z) ** 2, rel=1e-8)

    def test_redshift_constructor(self):
        m = Diggins26_TwoComponentModel(redshift=0.1)
        assert m.redshift == 0.1

    def test_composition_sets_mean_molecular_weights(self):
        m = Diggins26_TwoComponentModel(redshift=0.01, hydrogen_mass_fraction=0.7, helium_mass_fraction=0.28)
        assert m.mean_molecular_weight == pytest.approx(1 / (2 * 0.7 + 0.75 * 0.28 + 0.5 * 0.02))
        assert m.mean_molecular_weight_per_electron == pytest.approx(1 / (0.7 + 0.5 * 0.28 + 0.5 * 0.02))
        # The shock engine must use the same mean molecular weight.
        assert m._shock_engine._mu == pytest.approx(m.mean_molecular_weight)

    def test_helium_defaults_to_the_remainder(self):
        m = Diggins26_TwoComponentModel(redshift=0.01, hydrogen_mass_fraction=0.6)
        assert m._Y == pytest.approx(0.4)
        assert m._Z_metals == 0.0

    def test_integrator_kw_defaults_and_overrides(self):
        assert Diggins26_TwoComponentModel(redshift=0.01)._integrator_kw == {"method": "LSODA", "rtol": 1e-7}
        m = Diggins26_TwoComponentModel(redshift=0.01, integrator_kw={"method": "Radau", "rtol": 1e-8})
        assert m._integrator_kw == {"method": "Radau", "rtol": 1e-8}

    def test_integrator_kw_reaches_the_solver(self, monkeypatch):
        m = Diggins26_TwoComponentModel(redshift=0.01, integrator_kw={"atol": 1e-3}, adiabatic_index=1.4)
        seen = {}
        original = m._shock_engine._compute_shock_properties_cgs

        def spy(*args, **kwargs):
            seen.update(kwargs)
            return original(*args, **kwargs)

        monkeypatch.setattr(m._shock_engine, "_compute_shock_properties_cgs", spy)
        m.forward_model({"time": 20 * DAY, "frequency": 5e9}, {})
        assert seen["atol"] == 1e-3 and seen["rtol"] == 1e-7 and seen["method"] == "LSODA" and seen["gamma"] == 1.4

    def test_seed_luminosity_must_be_callable(self):
        with pytest.raises(TypeError):
            Diggins26_TwoComponentModel(redshift=0.01, seed_photon_luminosity=1e43)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_initial_beta": 1.5},
            {"shock_start_time": 0.0},
            {"reference_radius": -1.0},
            {"gamma_points_per_decade": 0},
            {"integrator_kw": {"rtol": 0.0}},
            {"integrator_kw": {"t_eval": [1.0]}},
            {"adiabatic_index": 1.0},
            {"hydrogen_mass_fraction": 1.2},
            {"hydrogen_mass_fraction": 0.7, "helium_mass_fraction": 0.5},
        ],
    )
    def test_invalid_configuration_raises(self, kwargs):
        with pytest.raises(ValueError):
            Diggins26_TwoComponentModel(redshift=0.01, **kwargs)

    def test_registered_for_serialization(self, model):
        kwargs = model._model_init_kwargs
        assert kwargs["redshift"] == model.redshift
        rebuilt = Diggins26_TwoComponentModel(**kwargs)
        assert rebuilt.mean_molecular_weight == pytest.approx(model.mean_molecular_weight)
        assert rebuilt._integrator_kw == model._integrator_kw
        assert rebuilt.angular_diameter_distance.to_value(u.cm) == pytest.approx(
            model.angular_diameter_distance.to_value(u.cm), rel=1e-12
        )


# =========================================================================== #
# Forward model                                                               #
# =========================================================================== #
class TestForwardModel:
    def test_output_units_and_shape(self, model):
        out = model.forward_model({"time": TIMES, "frequency": np.full_like(TIMES, 5e9)}, {})
        assert out.flux_density.unit == u.Jy
        assert out.flux_density.shape == TIMES.shape
        assert np.all(np.isfinite(out.flux_density.value)) and np.all(out.flux_density.value >= 0)

    def test_units_in_inputs(self, model):
        a = model.forward_model({"time": TIMES * u.s, "frequency": np.full_like(TIMES, 5e9) * u.Hz}, {})
        b = model.forward_model({"time": TIMES / DAY * u.day, "frequency": np.full_like(TIMES, 5) * u.GHz}, {})
        assert np.allclose(a.flux_density.value, b.flux_density.value, rtol=1e-12)

    def test_scalar_inputs(self, model):
        out = model.forward_model({"time": 20 * DAY, "frequency": 5e9}, {})
        assert np.ndim(out.flux_density.value) == 0
        assert np.isfinite(out.flux_density.value)

    def test_broadcasting(self, model):
        out = model.forward_model({"time": TIMES[:, None], "frequency": FREQS[None, :]}, {})
        assert out.flux_density.shape == (TIMES.size, FREQS.size)

    def test_order_independent(self, model):
        """Paired evaluation must not depend on the ordering or repetition of the epochs."""
        t = np.array([80.0, 5.0, 80.0, 20.0, 5.0]) * DAY
        nu = np.array([2e10, 1e9, 5e9, 5e9, 1e11])
        out = model.forward_model({"time": t, "frequency": nu}, {}).flux_density.value
        for i in range(t.size):
            single = model.forward_model({"time": t[i], "frequency": nu[i]}, {}).flux_density.value
            # The adaptive ODE step history depends on the end of the integration span, so single-point and
            # grouped evaluations agree only to the solver tolerance (rtol = 1e-7).
            assert out[i] == pytest.approx(single, rel=1e-5)

    def test_paired_matches_tabulate(self, model):
        tab = model.tabulate(TIMES, FREQS, {})
        T, N = np.meshgrid(TIMES, FREQS, indexing="ij")
        paired = model.forward_model({"time": T.ravel(), "frequency": N.ravel()}, {}).flux_density.value
        assert np.allclose(paired.reshape(T.shape), tab.flux_density.value, rtol=1e-10)

    def test_epoch_before_t0_raises(self, model):
        with pytest.raises(ValueError, match="shock_start_time"):
            model.forward_model({"time": 0.1 * DAY, "frequency": 5e9}, {})

    def test_parameter_bounds_raise(self, model):
        with pytest.raises(ValueError):
            model.forward_model({"time": 20 * DAY, "frequency": 5e9}, {"epsilon_B": 2.0})

    def test_flux_scales_with_epsilon_e(self, model):
        """At high frequency the emission is optically thin, so the flux scales linearly with the electron energy."""
        lo = model.forward_model({"time": 80 * DAY, "frequency": 3e11}, {"epsilon_e": 0.05}).flux_density.value
        hi = model.forward_model({"time": 80 * DAY, "frequency": 3e11}, {"epsilon_e": 0.1}).flux_density.value
        assert hi > lo


# =========================================================================== #
# Redshift                                                                    #
# =========================================================================== #
class TestRedshift:
    def test_dynamics_are_evaluated_in_the_source_frame(self):
        z = 0.2
        m_z = Diggins26_TwoComponentModel(redshift=z)
        m_0 = Diggins26_TwoComponentModel(redshift=1e-9)
        t_obs = np.array([10.0, 50.0, 200.0]) * DAY
        d_z = m_z.compute_dynamics({}, t_obs)
        d_0 = m_0.compute_dynamics({}, t_obs / (1 + z))
        assert np.allclose(d_z.time_rest.to_value(u.s), t_obs / (1 + z))
        for sector in ("equatorial", "polar"):
            for field in ("radius", "velocity", "thermal_energy_density"):
                a = getattr(getattr(d_z, sector), field).value
                b = getattr(getattr(d_0, sector), field).value
                assert np.allclose(a, b, rtol=1e-7), (sector, field)

    def test_free_free_is_evaluated_at_the_source_frame_frequency(self):
        """tau_ff(nu_obs) at redshift z equals tau_ff at nu_obs * (1 + z) with the source-frame optical depth."""
        z = 0.2
        m_z = Diggins26_TwoComponentModel(redshift=z)
        m_0 = Diggins26_TwoComponentModel(redshift=1e-9)
        t_obs = np.array([20.0, 80.0]) * DAY
        nu = np.array([1e9, 4e9])
        tau_z = m_z.tabulate(t_obs, nu, {}).equatorial.tau_ff
        tau_0 = m_0.tabulate(t_obs / (1 + z), nu * (1 + z), {}).equatorial.tau_ff
        assert np.allclose(tau_z, tau_0, rtol=1e-6)

    def test_distance_dimming(self):
        """At fixed physics, the flux scales as the engine's (1 + z)-corrected 1 / D_A^2 transformation."""
        near = Diggins26_TwoComponentModel(luminosity_distance=60 * u.Mpc)
        far = Diggins26_TwoComponentModel(luminosity_distance=120 * u.Mpc)
        f_near = near.forward_model({"time": 80 * DAY, "frequency": 3e11}, {}).flux_density.value
        f_far = far.forward_model({"time": 80 * DAY, "frequency": 3e11}, {}).flux_density.value
        assert 2.0 < f_near / f_far < 6.0  # roughly D^-2 with a modest redshift correction


# =========================================================================== #
# Free-free emission measure                                                  #
# =========================================================================== #
class TestEmissionMeasure:
    _P = {
        "rho_int": 2e-16,
        "rho_ext": 1e-18,
        "alpha_int": 1.5,
        "alpha_ext": 2.0,
        "R_trans": 5e15,
        "delta_trans": 0.3,
        "f_pole": 0.2,
        "aspect": 2.0,
        "R_ion": 1e19,
    }

    def _direct(self, model, p, theta, radius):
        """Reference int rho^2 dr by dense Simpson integration in ln r (independent of the model's quadrature)."""
        csm = AxisymmetricCSMDensityProfile.as_optimized_callable(theta=theta, **model._csm_parameters(p))
        out = []
        for R in radius:
            # QUADPACK's adaptive quad is unreliable for this steep integrand (errors up to ~1e-3 even with a
            # breakpoint at the transition, with no warning), so a fixed dense grid is used instead.
            x = np.linspace(np.log(R), np.log(p["R_ion"]), 200001)
            out.append(simpson(csm(np.exp(x), 0.0) ** 2 * np.exp(x), x=x))
        return np.array(out)

    @pytest.mark.parametrize("theta", [np.pi / 2, 0.0])
    @pytest.mark.parametrize("X", [1.0, 0.7, 0.0])
    def test_matches_direct_quadrature(self, theta, X):
        model = Diggins26_TwoComponentModel(redshift=0.01, hydrogen_mass_fraction=X)
        p = self._P
        radius = np.geomspace(1e14, 3e18, 9)
        em = model._emission_measure(p, theta, radius)
        composition = (1 + X) * (1 + 3 * X) / 8
        assert np.allclose(em, composition * self._direct(model, p, theta, radius) / M_P**2, rtol=1e-8)

    def test_pure_hydrogen_is_n_squared(self):
        """For X = 1 the plasma is n_e = n_i = rho / m_p, so EM = int n^2 dr."""
        model = Diggins26_TwoComponentModel(redshift=0.01, hydrogen_mass_fraction=1.0)
        radius = np.array([1e15])
        em = model._emission_measure(self._P, np.pi / 2, radius)
        assert em[0] == pytest.approx(self._direct(model, self._P, np.pi / 2, radius)[0] / M_P**2, rel=1e-8)

    def test_zero_beyond_the_ionized_radius(self, model):
        em = model._emission_measure({**self._P, "R_ion": 1e16}, np.pi / 2, np.array([1e15, 2e16, 5e17]))
        assert em[0] > 0 and em[1] == 0 and em[2] == 0

    def test_decreases_outward(self, model):
        em = model._emission_measure(self._P, np.pi / 2, np.geomspace(1e14, 1e18, 20))
        assert np.all(np.diff(em) < 0)

    def test_sharp_transition_is_resolved(self, model):
        p = {**self._P, "delta_trans": 0.02}
        radius = np.geomspace(1e14, 1e18, 6)
        em = model._emission_measure(p, np.pi / 2, radius)
        composition = (1 + 0.7) * (1 + 3 * 0.7) / 8
        assert np.allclose(em, composition * self._direct(model, p, np.pi / 2, radius) / M_P**2, rtol=1e-6)


# =========================================================================== #
# Tabulation and dynamics reuse                                               #
# =========================================================================== #
class TestTabulate:
    def test_structure(self, model):
        tab = model.tabulate(TIMES, FREQS, {})
        assert isinstance(tab, Diggins26Tabulation)
        assert isinstance(tab.dynamics, Diggins26Dynamics)
        assert tab.flux_density.shape == (TIMES.size, FREQS.size)
        for sector in (tab.equatorial, tab.polar):
            assert sector.flux_density.shape == tab.flux_density.shape
            assert sector.tau_ff.shape == tab.flux_density.shape
            assert sector.emission_measure.shape == TIMES.shape
            assert sector.magnetic_field.shape == TIMES.shape
        assert tab.equatorial.area_fraction + tab.polar.area_fraction == pytest.approx(1.0)

    def test_total_is_sum_of_attenuated_sectors(self, model):
        tab = model.tabulate(TIMES, FREQS, {})
        total = tab.equatorial.flux_density_attenuated + tab.polar.flux_density_attenuated
        assert np.allclose(total.value, tab.flux_density.value, rtol=1e-12)

    def test_attenuation_never_increases_flux(self, model):
        tab = model.tabulate(TIMES, FREQS, {})
        for sector in (tab.equatorial, tab.polar):
            assert np.all(sector.flux_density_attenuated.value <= sector.flux_density.value * (1 + 1e-12))
            assert np.all(sector.tau_ff >= 0)

    def test_free_free_absorbs_low_frequencies_more(self, model):
        tab = model.tabulate(TIMES, FREQS, {})
        assert np.all(np.diff(tab.equatorial.tau_ff, axis=1) <= 0)

    def test_reused_dynamics_gives_identical_result(self, model):
        dyn = model.compute_dynamics({}, TIMES)
        a = model.tabulate(TIMES, FREQS, {}, dynamics=dyn)
        b = model.tabulate(TIMES, FREQS, {})
        assert np.allclose(a.flux_density.value, b.flux_density.value, rtol=1e-12)

    def test_reused_dynamics_allows_changed_microphysics(self, model):
        dyn = model.compute_dynamics({}, TIMES)
        a = model.tabulate(TIMES, FREQS, {"epsilon_B": 0.05}, dynamics=dyn)
        b = model.tabulate(TIMES, FREQS, {"epsilon_B": 0.05})
        assert np.allclose(a.flux_density.value, b.flux_density.value, rtol=1e-12)

    def test_mismatched_dynamics_raises(self, model):
        dyn = model.compute_dynamics({}, TIMES)
        with pytest.raises(ValueError, match="rho_int"):
            model.tabulate(TIMES, FREQS, {"rho_int": 1e-15}, dynamics=dyn)
        with pytest.raises(ValueError, match="different epochs"):
            model.tabulate(TIMES[:-1], FREQS, {}, dynamics=dyn)

    def test_unsorted_times_raise(self, model):
        with pytest.raises(ValueError, match="increasing"):
            model.tabulate(TIMES[::-1], FREQS, {})

    def test_no_interior_makes_sectors_identical_per_unit_area(self, model):
        """With rho_int = 0 the CSM is spherical, so both sectors radiate identically per unit solid angle."""
        tab = model.tabulate(TIMES, FREQS, {"rho_int": 0.0})
        eq = tab.equatorial.flux_density.value / tab.equatorial.area_fraction
        pol = tab.polar.flux_density.value / tab.polar.area_fraction
        assert np.allclose(eq, pol, rtol=1e-6)
        assert np.allclose(tab.dynamics.equatorial.radius.value, tab.dynamics.polar.radius.value, rtol=1e-6)

    def test_dense_equator_decelerates_the_shock(self, model):
        dyn = model.compute_dynamics({}, TIMES)
        assert np.all(dyn.equatorial.radius.value < dyn.polar.radius.value)


# =========================================================================== #
# Infeasible parameters                                                       #
# =========================================================================== #
class TestInfeasibility:
    def test_default_parameters_are_feasible(self, model):
        assert model.diagnose({}, TIMES) is None

    @pytest.mark.parametrize(
        "parameters,fragment",
        [
            ({"M_ej": 1e29}, "max_initial_beta"),  # v_0 ~ c
            ({"gamma_max": 1.005, "gamma_min": 1.0}, "cooling break"),
            ({"rho_int": 0.0, "rho_ext": 0.0}, "no CSM mass"),
        ],
    )
    def test_nan_flux_and_diagnosis(self, model, parameters, fragment):
        out = model.forward_model({"time": TIMES, "frequency": np.full_like(TIMES, 5e9)}, parameters)
        assert np.all(np.isnan(out.flux_density.value))
        reason = model.diagnose(parameters, TIMES)
        assert reason is not None and fragment.lower() in reason.lower()

    def test_ejecta_normalization_failure_is_infeasible(self):
        m = Diggins26_TwoComponentModel(redshift=0.01, max_initial_beta=0.99)
        reason = m.diagnose({"M_ej": 1e29}, 20 * DAY)
        assert reason is not None
        out = m.forward_model({"time": 20 * DAY, "frequency": 5e9}, {"M_ej": 1e29})
        assert np.isnan(out.flux_density.value)

    def test_tabulate_raises_with_reason(self, model):
        with pytest.raises(ValueError, match="infeasible"):
            model.tabulate(TIMES, FREQS, {"M_ej": 1e29})

    def test_unexpected_errors_propagate(self, model, monkeypatch):
        def broken(*args, **kwargs):
            raise KeyError("a genuine bug")

        monkeypatch.setattr(model, "_microphysics", broken)
        with pytest.raises(KeyError):
            model.forward_model({"time": 20 * DAY, "frequency": 5e9}, {})


# =========================================================================== #
# Vectorized radiation                                                        #
# =========================================================================== #
class TestVectorizedRadiation:
    """The batched microphysics and paired spectra must equal a plain per-epoch loop of the scalar library calls."""

    @pytest.fixture(scope="class")
    def setup(self, model):
        p = model._coerce({})
        t_rest = model._rest_times(np.unique(TIMES))
        states, reason = model._dynamics_cgs(p, t_rest)
        assert states is not None, reason
        gamma = model._gamma_grid(p, model._gamma_points_per_decade)
        return p, t_rest, states[0], gamma

    def test_microphysics_matches_the_scalar_distribution(self, model, setup):
        from trilobite.radiation.synchrotron.electron_distributions import BrokenPowerLaw

        p, t_rest, state, gamma = setup
        B, gamma_c, N0, log_N = model._microphysics(p, state, t_rest, gamma)

        for i in range(B.size):
            shape = {
                "p1": p["p"],
                "p2": p["p"] + 1.0,
                "gamma_min": p["gamma_min"],
                "gamma_max": p["gamma_max"],
                "gamma_c": gamma_c[i],
            }
            N0_i = BrokenPowerLaw._normalize_from_magnetic_field(B[i], p["epsilon_B"], p["epsilon_e"], **shape)
            assert N0[i] == pytest.approx(N0_i, rel=1e-12)
            with np.errstate(divide="ignore"):
                expected = np.log(BrokenPowerLaw.pdf(gamma, norm=N0_i, **shape))
            assert np.allclose(log_N[i], expected, rtol=1e-12, atol=0.0, equal_nan=True)

    def test_paired_spectra_match_per_epoch_engine_calls(self, model, setup):
        p, t_rest, state, gamma = setup
        micro = model._microphysics(p, state, t_rest, gamma)
        B, _, _, log_N = micro
        EM = model._emission_measure(p, np.pi / 2, state.radius)
        f_A = model._area_fractions(p["aspect"])[0]

        # a different number of frequencies at every epoch, including a single one
        rng = np.random.default_rng(0)
        nu_groups = [np.sort(rng.uniform(1e9, 3e11, size=1 + (i % 4))) for i in range(t_rest.size)]
        tau_per_EM = model._free_free_opacity(p, nu_groups)
        flux, tau = model._sector_spectra(p, state, micro, EM, f_A, gamma, nu_groups, tau_per_EM)

        log_gamma, log_weights = model._synchrotron_engine._build_gamma_grid(gamma)
        ell = p["f_V"] ** (1.0 / 3.0) * state.radius
        area = np.pi * state.radius**2 * f_A
        for i, nu in enumerate(nu_groups):
            with np.errstate(divide="ignore"):
                expected = np.exp(
                    model._synchrotron_engine._compute_log_pa_flux_density(
                        np.log(nu),
                        np.log(ell[i]),
                        np.log(B[i]),
                        log_N[i],
                        log_gamma,
                        log_weights,
                        np.log(area[i]),
                        model._log_D_A,
                        z=model._z,
                    )
                )
            assert np.allclose(flux[i], expected, rtol=1e-12, atol=0.0)
            assert np.allclose(tau[i], EM[i] * tau_per_EM[i], rtol=1e-14)

    def test_chunked_evaluation_is_identical(self, model, setup, monkeypatch):
        import trilobite.models.LFBOTs._diggins26 as module

        p, t_rest, state, gamma = setup
        micro = model._microphysics(p, state, t_rest, gamma)
        EM = model._emission_measure(p, np.pi / 2, state.radius)
        nu_groups = [np.array([1e9, 5e10, 2e11]) for _ in t_rest]
        tau_per_EM = model._free_free_opacity(p, nu_groups)
        args = (p, state, micro, EM, 0.5, gamma, nu_groups, tau_per_EM)

        whole, _ = model._sector_spectra(*args)
        monkeypatch.setattr(module, "_SPECTRUM_CHUNK", 5)  # forces several chunks that split epochs mid-way
        chunked, _ = model._sector_spectra(*args)
        assert all(np.array_equal(a, b) for a, b in zip(whole, chunked))


# =========================================================================== #
# Compton cooling                                                             #
# =========================================================================== #
class TestComptonCooling:
    @staticmethod
    def _constant(L):
        return lambda t: np.full_like(t, L)

    def test_no_seed_photons_is_synchrotron_cooling(self, model):
        B = np.array([0.1, 1.0])
        radius = np.array([1e16, 1e17])
        t = np.array([1e6, 1e7])
        assert np.allclose(model._cooling_gamma(B, radius, t), 9 * np.pi * M_E * C / (SIGMA_T * B**2 * t), rtol=1e-12)

    def test_inverse_compton_loss_rates_add(self):
        L = 1e44
        model = Diggins26_TwoComponentModel(redshift=0.01, seed_photon_luminosity=self._constant(L))
        B, radius, t = np.array([0.3]), np.array([3e16]), np.array([2e6])
        gamma_syn = 9 * np.pi * M_E * C / (SIGMA_T * B**2 * t)
        gamma_ic = 3 * np.pi * M_E * C**2 * radius**2 / (SIGMA_T * L * t)
        assert np.allclose(model._cooling_gamma(B, radius, t), 1 / (1 / gamma_syn + 1 / gamma_ic), rtol=1e-12)

    def test_zero_luminosity_recovers_synchrotron_only(self, model):
        dark = Diggins26_TwoComponentModel(luminosity_distance=60 * u.Mpc, seed_photon_luminosity=self._constant(0.0))
        assert np.allclose(
            dark.tabulate(TIMES, FREQS, {}).flux_density.value,
            model.tabulate(TIMES, FREQS, {}).flux_density.value,
            rtol=1e-12,
            atol=0.0,
        )

    def test_seed_photons_lower_the_cooling_break_and_leave_the_dynamics_alone(self, model):
        bright = Diggins26_TwoComponentModel(
            luminosity_distance=60 * u.Mpc, seed_photon_luminosity=self._constant(1e45)
        )
        plain, cooled = model.tabulate(TIMES, FREQS, {}), bright.tabulate(TIMES, FREQS, {})
        assert np.all(cooled.equatorial.gamma_c <= plain.equatorial.gamma_c)
        assert np.any(cooled.equatorial.gamma_c < plain.equatorial.gamma_c)
        assert np.allclose(cooled.dynamics.equatorial.radius.value, plain.dynamics.equatorial.radius.value, rtol=1e-10)

    def test_callable_receives_source_frame_seconds_and_may_return_a_quantity(self):
        seen = []

        def luminosity(t):
            seen.append(np.array(t))
            return 1e43 * (t / 1e6) ** -1.0 * u.erg / u.s

        m = Diggins26_TwoComponentModel(redshift=0.5, seed_photon_luminosity=luminosity)
        out = m.forward_model({"time": 30 * DAY, "frequency": 5e9}, {})
        assert np.isfinite(out.flux_density.value)
        assert seen[0][0] == pytest.approx(30 * DAY / 1.5)

    @pytest.mark.parametrize("bad", [-1.0, np.nan])
    def test_invalid_luminosity_raises(self, bad):
        m = Diggins26_TwoComponentModel(redshift=0.01, seed_photon_luminosity=self._constant(bad))
        with pytest.raises(ValueError, match="seed_photon_luminosity"):
            m.forward_model({"time": 20 * DAY, "frequency": 5e9}, {})


# =========================================================================== #
# Inference integration                                                       #
# =========================================================================== #
class TestInference:
    @pytest.fixture(scope="class")
    def problem(self):
        model = Diggins26_TwoComponentModel(luminosity_distance=60 * u.Mpc)
        t = np.repeat(np.array([10.0, 40.0, 160.0]) * DAY, 3)
        nu = np.tile(np.array([2e9, 1e10, 1e11]), 3)
        truth = model.forward_model({"time": t, "frequency": nu}, {}).flux_density.to_value(u.Jy)
        data = InferenceData(
            x={"time": t, "frequency": nu},
            observables={"flux_density": Observable(value=truth, error=0.1 * truth)},
        )
        like = LogGaussianCensoredLikelihood(model, data, sys_dex=0.1)
        problem = InferenceProblem(likelihood=like)
        problem.set_prior("rho_int", "uniform", lower=1e-17, upper=1e-14)
        problem.set_prior("epsilon_B", "uniform", lower=1e-4, upper=0.5)
        for name in problem.parameter_order:
            if name not in ("rho_int", "epsilon_B"):
                problem.parameters[name].freeze = True
        problem.parameters["rho_int"].initial_value = 2e-16
        problem.parameters["epsilon_B"].initial_value = 0.01
        return problem

    def test_posterior_is_finite_and_peaks_at_truth(self, problem):
        theta = problem.initial_theta
        at_truth = problem._log_free_posterior(theta)
        assert np.isfinite(at_truth)
        # Perturbing either free parameter lowers the posterior.
        for i in range(theta.size):
            moved = theta.copy()
            moved[i] = moved[i] * 3.0
            assert problem._log_free_posterior(moved) < at_truth

    def test_infeasible_draw_is_rejected_cleanly(self, problem):
        problem.parameters["M_ej"].freeze = True
        problem.parameters["M_ej"].initial_value = 1e29
        try:
            assert problem._log_free_posterior(problem.initial_theta) == -np.inf
        finally:
            problem.parameters["M_ej"].initial_value = 0.1 * const.M_sun.cgs.value
