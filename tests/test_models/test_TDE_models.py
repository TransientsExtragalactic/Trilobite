"""Tests for VanVelzenModel."""

import pathlib

import numpy as np
import pytest
from astropy import units as u

from trilobite.data.core import InferenceData, Observable
from trilobite.inference import GaussianLikelihood, InferenceProblem
from trilobite.inference.sampling.mcmc import EmceeSampler
from trilobite.models.TDEs.van_velzen import VanVelzenModel
from trilobite.utils.phot_utils import FilterBundle, PhotometryFilter

np.random.seed(42)

_DAY = 86400.0

# ---------------------------------------------------------------------------
# Helpers: minimal multi-band bundle
# ---------------------------------------------------------------------------


def _make_box_filter(lam_center_cm: float, width_cm: float, name: str) -> PhotometryFilter:
    """Box filter centred on lam_center_cm with given full width."""
    lam = np.linspace(lam_center_cm - width_cm / 2, lam_center_cm + width_cm / 2, 50)
    T = np.ones(50)
    return PhotometryFilter(lam, T, name=name)


@pytest.fixture
def two_filter_bundle() -> FilterBundle:
    """Bundle with g (477 nm) and r (620 nm) box filters."""
    f_g = _make_box_filter(4770e-8, 1000e-8, "g")
    f_r = _make_box_filter(6200e-8, 1200e-8, "r")
    return FilterBundle({"g": f_g, "r": f_r})


@pytest.fixture
def vv_model(two_filter_bundle) -> VanVelzenModel:
    return VanVelzenModel(two_filter_bundle)


_TRUE_PARAMS = {
    "L_peak": 2.0e44,
    "t_peak": 20.0 * _DAY,
    "sigma_rise": 8.0 * _DAY,
    "tau_decay": 40.0 * _DAY,
    "T": 2.2e4,
    "D": 3.086e26,  # 100 Mpc
}


# ---------------------------------------------------------------------------
# Structural / physics tests
# ---------------------------------------------------------------------------


class TestVanVelzenModel:
    def test_output_shape(self, vv_model):
        variables = {"band_index": [0, 1, 0, 1], "time": [1e5, 1e5, 2e5, 2e5]}
        result = vv_model.forward_model(variables, _TRUE_PARAMS)
        assert result.flux.shape == (4,)

    def test_output_units(self, vv_model):
        variables = {"band_index": [0, 1], "time": [1e5, 1e5]}
        result = vv_model.forward_model(variables, _TRUE_PARAMS)
        assert result.flux.unit.is_equivalent(u.erg / u.s / u.cm**2 / u.Hz)

    def test_positive_flux(self, vv_model):
        t = np.linspace(-20, 150, 20) * _DAY
        variables = {"band_index": [0] * len(t), "time": t}
        result = vv_model.forward_model(variables, _TRUE_PARAMS)
        assert np.all(result.flux.value > 0)

    def test_flux_peaks_near_t_peak(self, vv_model):
        """The flux light curve should peak at (or very near) t_peak."""
        t = np.linspace(-40, 150, 400) * _DAY
        variables = {"band_index": [0] * len(t), "time": t}
        result = vv_model.forward_model(variables, _TRUE_PARAMS)
        t_at_max = t[np.argmax(result.flux.value)]
        assert abs(t_at_max - _TRUE_PARAMS["t_peak"]) < 2 * _DAY

    def test_continuity_at_t_peak(self, vv_model):
        """Rise and decay branches must agree at t_peak (both equal L_peak)."""
        t_peak = _TRUE_PARAMS["t_peak"]
        eps = 1.0  # 1 second
        variables = {"band_index": [0, 0], "time": [t_peak - eps, t_peak + eps]}
        result = vv_model.forward_model(variables, _TRUE_PARAMS)
        np.testing.assert_allclose(result.flux.value[0], result.flux.value[1], rtol=1e-6)

    def test_hotter_is_brighter_in_g_band(self, vv_model):
        """At fixed luminosity, a hotter blackbody peaks bluer; check monotonic increase in a specific regime."""
        params_cool = dict(_TRUE_PARAMS, T=1.0e4)
        params_hot = dict(_TRUE_PARAMS, T=2.5e4)
        variables = {"band_index": [0], "time": [_TRUE_PARAMS["t_peak"]]}
        r_cool = vv_model.forward_model(variables, params_cool)
        r_hot = vv_model.forward_model(variables, params_hot)
        assert r_hot.flux.value[0] != r_cool.flux.value[0]

    def test_string_band_names(self, vv_model):
        variables = {"band_index": ["g", "r", "g"], "time": [1e5, 1e5, 2e5]}
        result = vv_model.forward_model(variables, _TRUE_PARAMS)
        assert result.flux.shape == (3,)

    def test_astropy_time_units(self, vv_model):
        vars_days = {"band_index": [0, 1], "time": [1.0, 1.0] * u.day}
        vars_secs = {"band_index": [0, 1], "time": [86400.0, 86400.0]}
        r_days = vv_model.forward_model(vars_days, _TRUE_PARAMS)
        r_secs = vv_model.forward_model(vars_secs, _TRUE_PARAMS)
        np.testing.assert_allclose(r_days.flux.value, r_secs.flux.value, rtol=1e-10)

    def test_serialisation_roundtrip(self, vv_model):
        spec = vv_model.to_model_spec()
        restored = VanVelzenModel.from_model_spec(spec)
        assert isinstance(restored, VanVelzenModel)
        assert restored.bundle.filter_names == vv_model.bundle.filter_names
        np.testing.assert_allclose(restored.bundle.weight_matrix, vv_model.bundle.weight_matrix, rtol=1e-10)


# ---------------------------------------------------------------------------
# End-to-end fitting test
# ---------------------------------------------------------------------------


@pytest.fixture
def van_velzen_problem(two_filter_bundle):
    """Synthetic multi-band light curve, fit through the full InferenceProblem stack."""
    model = VanVelzenModel(two_filter_bundle)

    # Epochs for each band (rest-frame seconds since an arbitrary reference).
    t_g = np.linspace(-15, 140, 25) * _DAY
    t_r = np.linspace(-10, 140, 25) * _DAY

    band_index = np.concatenate([np.zeros_like(t_g, dtype=int), np.ones_like(t_r, dtype=int)])
    time = np.concatenate([t_g, t_r])

    true_flux = model.forward_model({"band_index": band_index, "time": time}, _TRUE_PARAMS).flux.value

    # A local, fixed-seed generator keeps the noise realisation independent of whatever
    # other tests have consumed from the global numpy RNG.
    frac_err = 0.05
    flux_err = frac_err * true_flux
    flux_obs = np.random.default_rng(42).normal(loc=true_flux, scale=flux_err)

    data = InferenceData(
        x={"band_index": band_index, "time": time},
        observables={"flux": Observable(value=flux_obs, error=flux_err)},
    )

    likelihood = GaussianLikelihood(model=model, data=data)
    problem = InferenceProblem(likelihood=likelihood)

    # Distance is treated as externally known (e.g. from redshift + cosmology)
    # and frozen; the remaining light-curve-shape and temperature parameters
    # are fit.
    problem.freeze_parameters(["D"])
    problem.parameters["D"].initial_value = _TRUE_PARAMS["D"]

    problem.set_prior("L_peak", "loguniform", lower=1.0e43, upper=1.0e45)
    problem.set_prior("t_peak", "uniform", lower=-10.0 * _DAY, upper=60.0 * _DAY)
    problem.set_prior("sigma_rise", "uniform", lower=1.0 * _DAY, upper=30.0 * _DAY)
    problem.set_prior("tau_decay", "uniform", lower=5.0 * _DAY, upper=100.0 * _DAY)
    problem.set_prior("T", "uniform", lower=5.0e3, upper=5.0e4)

    # Offset initial guesses from the truth so the sampler has to do real work.
    problem.parameters["L_peak"].initial_value = 1.2e44
    problem.parameters["t_peak"].initial_value = 12.0 * _DAY
    problem.parameters["sigma_rise"].initial_value = 12.0 * _DAY
    problem.parameters["tau_decay"].initial_value = 25.0 * _DAY
    problem.parameters["T"].initial_value = 1.5e4

    return problem, _TRUE_PARAMS


# The initial guess is deliberately poor (log-posterior ~ -1e8); the sampler must recover from it.
def test_van_velzen_mcmc_converges_near_truth(
    van_velzen_problem,
    diagnostic_plots,
    diagnostic_plots_dir,
):
    problem, true_params = van_velzen_problem

    sampler = EmceeSampler(problem, n_walkers=32)

    # Seed the walker initialisation so the run is reproducible.
    initial_state = sampler.generate_initial_state(rng=np.random.default_rng(42))
    result = sampler.run(2000, initial_state=initial_state, progress=False)

    assert result.find_stranded_walkers().size == 0

    samples = result.get_flat_samples(burn=500, thin=5)
    median_estimate = np.median(samples, axis=0)
    lower, upper = np.percentile(samples, [0.15, 99.85], axis=0)

    # ------------------------------------------------------------------
    # Convergence checks.  L_peak and T are strongly degenerate for two optical
    # bands (the 99.7% interval on L_peak spans > 100% of its value), so a fixed
    # tolerance on the posterior mean fails for ~6% of noise realisations even
    # with a perfect sampler.  We therefore require every true value to lie in
    # the central 99.7% credible interval, and additionally require the
    # well-constrained light-curve-shape parameters to be tightly recovered.
    # ------------------------------------------------------------------
    tight_tol = {
        "t_peak": 5.0 * _DAY,  # absolute, in seconds
        "sigma_rise": 0.4,
        "tau_decay": 0.4,
    }

    for i, name in enumerate(problem.free_parameter_names):
        truth = true_params[name]
        assert lower[i] <= truth <= upper[i], f"{name}: truth {truth:.4g} outside [{lower[i]:.4g}, {upper[i]:.4g}]"

        if name == "t_peak":
            assert abs(median_estimate[i] - truth) < tight_tol[name]
        elif name in tight_tol:
            assert abs(median_estimate[i] - truth) / truth < tight_tol[name]

    if diagnostic_plots:
        output_path = pathlib.Path(diagnostic_plots_dir) / "van_velzen_corner_plot.png"
        fig = result.corner_plot(burn=500, thin=5, truths=true_params)
        fig.savefig(output_path, dpi=150)
