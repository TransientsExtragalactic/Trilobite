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

    def test_distances_are_consistent(self, model):
        assert model.luminosity_distance.to_value(u.Mpc) == pytest.approx(60.0)
        z = model.redshift
        assert 0.012 < z < 0.015
        ratio = (model.luminosity_distance / model.angular_diameter_distance).to_value(u.dimensionless_unscaled)
        assert ratio == pytest.approx((1 + z) ** 2, rel=1e-8)

    def test_redshift_constructor(self):
        m = Diggins26_TwoComponentModel(redshift=0.1)
        assert m.redshift == 0.1

    @pytest.mark.parametrize(
        "kwargs",
        [{"beta_max": 1.5}, {"t_0": 0.0}, {"r_ref": -1.0}, {"n_gamma_per_decade": 0}, {"ode_rtol": 0.0}],
    )
    def test_invalid_configuration_raises(self, kwargs):
        with pytest.raises(ValueError):
            Diggins26_TwoComponentModel(redshift=0.01, **kwargs)

    def test_registered_for_serialization(self, model):
        kwargs = model._model_init_kwargs
        assert kwargs["redshift"] == model.redshift
        rebuilt = Diggins26_TwoComponentModel(**kwargs)
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
            # grouped evaluations agree only to the solver tolerance (rtol = 1e-6).
            assert out[i] == pytest.approx(single, rel=1e-5)

    def test_paired_matches_tabulate(self, model):
        tab = model.tabulate(TIMES, FREQS, {})
        T, N = np.meshgrid(TIMES, FREQS, indexing="ij")
        paired = model.forward_model({"time": T.ravel(), "frequency": N.ravel()}, {}).flux_density.value
        assert np.allclose(paired.reshape(T.shape), tab.flux_density.value, rtol=1e-10)

    def test_epoch_before_t0_raises(self, model):
        with pytest.raises(ValueError, match="t_0"):
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
        "X": 0.7,
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
    def test_matches_direct_quadrature(self, model, theta, X):
        p = {**self._P, "X": X}
        radius = np.geomspace(1e14, 3e18, 9)
        em = model._emission_measure(p, theta, radius)
        composition = (1 + X) * (1 + 3 * X) / 8
        assert np.allclose(em, composition * self._direct(model, p, theta, radius) / M_P**2, rtol=1e-8)

    def test_pure_hydrogen_is_n_squared(self, model):
        """For X = 1 the plasma is n_e = n_i = rho / m_p, so EM = int n^2 dr."""
        p = {**self._P, "X": 1.0}
        radius = np.array([1e15])
        em = model._emission_measure(p, np.pi / 2, radius)
        assert em[0] == pytest.approx(self._direct(model, p, np.pi / 2, radius)[0] / M_P**2, rel=1e-8)

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
            ({"M_ej": 1e29}, "beta_max"),  # v_0 ~ c
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
        m = Diggins26_TwoComponentModel(redshift=0.01, beta_max=0.99)
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
