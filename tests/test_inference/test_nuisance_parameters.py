"""Tests for likelihood-level nuisance parameters and :class:`LogGaussianCensoredSysLikelihood`."""

import json

import numpy as np
import pytest

from trilobite.data.core import InferenceData, Observable
from trilobite.inference import (
    GaussianLikelihood,
    InferenceProblem,
    LogGaussianCensoredLikelihood,
    LogGaussianCensoredSysLikelihood,
)
from trilobite.inference.likelihood.base import Likelihood
from trilobite.inference.sampling.mcmc import EmceeSampler
from trilobite.inference.transform import Log10Transform
from trilobite.models.core.parameters import ModelParameter
from trilobite.models.generic.curves import LinearModel
from trilobite.parallel.mp import LikelihoodMPPool

X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
TRUE = {"m": 2.0, "b": 1.0}  # y = 3, 5, 7, 9, 11


def _data(value, error, upper=None, lower=None):
    return InferenceData(
        x={"x": X},
        observables={
            "y": Observable(
                value=np.asarray(value, dtype=float),
                error=np.asarray(error, dtype=float),
                upper=None if upper is None else np.asarray(upper, dtype=float),
                lower=None if lower is None else np.asarray(lower, dtype=float),
            )
        },
    )


@pytest.fixture
def model():
    return LinearModel()


@pytest.fixture
def truth(model):
    return np.asarray(model.forward_model({"x": X}, TRUE).y, dtype=float)


@pytest.fixture
def y_obs(truth):
    return truth * np.array([1.1, 0.9, 1.05, 0.8, 1.2])


@pytest.fixture
def problem(model, y_obs):
    problem = InferenceProblem(likelihood=LogGaussianCensoredSysLikelihood(model, _data(y_obs, 0.1 * y_obs)))
    problem.set_prior("m", "uniform", lower=0.0, upper=5.0)
    problem.set_prior("b", "uniform", lower=0.0, upper=5.0)
    problem.set_prior("sys_dex", "loguniform", lower=1e-3, upper=1.0)
    problem.parameters["m"].initial_value = 1.5
    problem.parameters["b"].initial_value = 1.0
    problem.parameters["sys_dex"].initial_value = 0.1
    return problem


# =================================================================== #
# Base class                                                          #
# =================================================================== #
class TestBaseClass:
    def test_existing_likelihoods_declare_no_nuisance_parameters(self, model, y_obs):
        like = GaussianLikelihood(model, _data(y_obs, 0.1 * y_obs))
        assert like.NUISANCE_PARAMETERS == ()
        assert like.nuisance_parameters == ()
        assert like.coerce_nuisance_parameters({"m": 1.0}) == {}

    def test_problem_without_nuisance_parameters_is_unchanged(self, model, y_obs):
        problem = InferenceProblem(likelihood=GaussianLikelihood(model, _data(y_obs, 0.1 * y_obs)))
        assert problem.parameter_order == ("m", "b")
        assert problem.nuisance_parameter_names == ()
        assert problem.model_parameter_names == ("m", "b")

    def test_single_argument_likelihoods_are_still_called_with_one_argument(self, model, y_obs):
        """A likelihood without nuisance parameters keeps the ``_log_likelihood(self, parameters)`` signature."""
        like = GaussianLikelihood(model, _data(y_obs, 0.1 * y_obs))
        assert like.log_likelihood(TRUE) == pytest.approx(like._log_likelihood(TRUE), rel=1e-14)

    def test_name_collision_with_a_model_parameter_raises(self, model, y_obs):
        class Colliding(GaussianLikelihood):
            NUISANCE_PARAMETERS = (ModelParameter("m", 1.0),)

        with pytest.raises(ValueError, match="collide"):
            Colliding(model, _data(y_obs, 0.1 * y_obs))

    def test_duplicate_nuisance_names_raise(self, model, y_obs):
        class Duplicated(GaussianLikelihood):
            NUISANCE_PARAMETERS = (ModelParameter("s", 1.0), ModelParameter("s", 2.0))

        with pytest.raises(ValueError, match="Duplicate"):
            Duplicated(model, _data(y_obs, 0.1 * y_obs))

    def test_non_model_parameter_entry_raises(self, model, y_obs):
        class Bad(GaussianLikelihood):
            NUISANCE_PARAMETERS = ("s",)

        with pytest.raises(TypeError, match="ModelParameter"):
            Bad(model, _data(y_obs, 0.1 * y_obs))

    def test_nuisance_values_reach_the_statistics_but_not_the_model(self, model, y_obs):
        """The model sees only its own parameters; the likelihood receives the coerced nuisance dict."""
        seen = {}

        class Scaled(GaussianLikelihood):
            NUISANCE_PARAMETERS = (ModelParameter("scale", 2.0, bounds=(0.0, None)),)

            def _log_likelihood(self, parameters, nuisance):
                seen["parameters"] = dict(parameters)
                seen["nuisance"] = dict(nuisance)
                return 0.0

        like = Scaled(model, _data(y_obs, 0.1 * y_obs))
        like.log_likelihood({**TRUE, "scale": 3.5})
        assert set(seen["parameters"]) == {"m", "b"}
        assert seen["nuisance"] == {"scale": 3.5}

        like.log_likelihood(TRUE)  # missing nuisance values take their declared default
        assert seen["nuisance"] == {"scale": 2.0}

    def test_nuisance_parameter_is_a_class_level_declaration_of_a_likelihood_subclass(self):
        assert issubclass(LogGaussianCensoredSysLikelihood, Likelihood)
        assert [p.name for p in LogGaussianCensoredSysLikelihood.NUISANCE_PARAMETERS] == ["sys_dex"]


# =================================================================== #
# Statistics                                                          #
# =================================================================== #
class TestSysLikelihood:
    def test_matches_fixed_floor_parent_at_the_same_value(self, model, y_obs):
        data = _data(y_obs, 0.1 * y_obs)
        fixed = LogGaussianCensoredLikelihood(model, data, sys_dex=0.15)
        sampled = LogGaussianCensoredSysLikelihood(model, data)
        assert sampled.log_likelihood({**TRUE, "sys_dex": 0.15}) == pytest.approx(fixed.log_likelihood(TRUE), rel=1e-13)

    def test_matches_fixed_floor_parent_with_censored_limits(self, model, truth):
        """The sampled term must widen censored limits exactly as the fixed floor does."""
        value = truth.copy()
        err = 0.1 * truth
        upper = np.full_like(truth, np.nan)
        lower = np.full_like(truth, np.nan)
        value[-1], upper[-1] = np.nan, 4.0  # violated upper limit
        value[0], lower[0] = np.nan, 1.0  # satisfied lower limit
        data = _data(value, err, upper=upper, lower=lower)

        fixed = LogGaussianCensoredLikelihood(model, data, sys_dex=0.2)
        sampled = LogGaussianCensoredSysLikelihood(model, data)
        assert sampled.log_likelihood({**TRUE, "sys_dex": 0.2}) == pytest.approx(fixed.log_likelihood(TRUE), rel=1e-13)

    def test_default_nuisance_value_is_used_when_not_given(self, model, y_obs):
        data = _data(y_obs, 0.1 * y_obs)
        sampled = LogGaussianCensoredSysLikelihood(model, data)
        fixed = LogGaussianCensoredLikelihood(model, data, sys_dex=0.1)
        assert sampled.log_likelihood(TRUE) == pytest.approx(fixed.log_likelihood(TRUE), rel=1e-13)

    def test_normalization_term_lets_the_error_scale_be_inferred(self, model, truth):
        """With sigma free, the likelihood peaks at the scatter actually present (no chi_r = 1 constraint)."""
        rng = np.random.default_rng(1)
        scatter = 0.12
        y = truth * 10 ** rng.normal(0.0, scatter, size=truth.size * 40)[: truth.size]
        like = LogGaussianCensoredSysLikelihood(model, _data(y, 1e-6 * y))

        grid = np.linspace(0.005, 0.6, 400)
        logl = np.array([like.log_likelihood({**TRUE, "sys_dex": s}) for s in grid])
        resid = np.log10(y) - np.log10(truth)
        rms = np.sqrt(np.mean(resid**2))  # maximum-likelihood sigma for known mean
        assert grid[np.argmax(logl)] == pytest.approx(rms, abs=2 * (grid[1] - grid[0]))

    @pytest.mark.parametrize("bad", [-0.1, np.nan, np.inf])
    def test_invalid_sys_dex_gives_neg_inf(self, model, y_obs, bad):
        like = LogGaussianCensoredSysLikelihood(model, _data(y_obs, 0.1 * y_obs))
        assert like.log_likelihood({**TRUE, "sys_dex": bad}) == -np.inf

    def test_zero_total_uncertainty_gives_neg_inf_not_nan(self, model, y_obs):
        like = LogGaussianCensoredSysLikelihood(model, _data(y_obs, np.zeros_like(y_obs)))
        assert like.log_likelihood({**TRUE, "sys_dex": 0.0}) == -np.inf
        assert np.isfinite(like.log_likelihood({**TRUE, "sys_dex": 0.1}))

    def test_invalid_model_output_still_gives_neg_inf(self, model, y_obs, monkeypatch):
        like = LogGaussianCensoredSysLikelihood(model, _data(y_obs, 0.1 * y_obs))
        monkeypatch.setattr(model, "_forward_model_tupled", lambda x, p: (np.full(5, np.nan),))
        assert like.log_likelihood({**TRUE, "sys_dex": 0.1}) == -np.inf

    def test_fixed_floor_option_is_rejected(self, model, y_obs):
        with pytest.raises(TypeError, match="sys_dex"):
            LogGaussianCensoredSysLikelihood(model, _data(y_obs, 0.1 * y_obs), sys_dex=0.1)

    def test_parent_behavior_is_unchanged(self, model, y_obs):
        """The refactor of the parent must not alter its fixed-floor likelihood."""
        err = 0.1 * y_obs
        like = LogGaussianCensoredLikelihood(model, _data(y_obs, err), sys_dex=0.15)
        sigma = np.sqrt((err / (y_obs * np.log(10.0))) ** 2 + 0.15**2)
        truth = np.asarray(model.forward_model({"x": X}, TRUE).y, dtype=float)
        resid = (np.log10(y_obs) - np.log10(truth)) / sigma
        assert like.log_likelihood(TRUE) == pytest.approx(
            -0.5 * np.sum(resid**2 + np.log(2 * np.pi * sigma**2)), rel=1e-12
        )
        np.testing.assert_allclose(like._y_err.reshape(sigma.shape), sigma, rtol=1e-14)


# =================================================================== #
# InferenceProblem plumbing                                           #
# =================================================================== #
class TestProblem:
    def test_nuisance_parameter_is_appended_after_model_parameters(self, problem):
        assert problem.parameter_order == ("m", "b", "sys_dex")
        assert problem.model_parameter_names == ("m", "b")
        assert problem.nuisance_parameter_names == ("sys_dex",)
        assert problem.n_parameters == 3
        assert problem["sys_dex"].model_parameter.name == "sys_dex"

    def test_pack_unpack_round_trip(self, problem):
        params = {"m": 1.5, "b": 1.0, "sys_dex": 0.2}
        theta = problem.pack_parameters(params)
        np.testing.assert_array_equal(theta, [1.5, 1.0, 0.2])
        assert problem.unpack_parameters(theta) == params

    def test_freezing_the_nuisance_parameter(self, problem):
        problem.freeze_parameters("sys_dex")
        assert problem.free_parameter_names == ("m", "b")
        assert problem.fixed_parameter_names == ("sys_dex",)
        z = problem.initial_z
        assert z.shape == (2,)
        assert np.isfinite(problem._log_free_posterior(z))

    def test_posterior_depends_on_the_nuisance_parameter(self, problem):
        base = problem.initial_theta
        shifted = base.copy()
        shifted[2] = 0.3
        assert problem._log_free_posterior(base) != problem._log_free_posterior(shifted)

    def test_public_log_methods_accept_nuisance_values(self, problem):
        params = {"m": 1.5, "b": 1.0, "sys_dex": 0.25}
        theta = problem.pack_parameters(params)
        assert problem.log_likelihood(params) == pytest.approx(problem._log_likelihood(theta), rel=1e-14)
        assert problem.log_prior(params) == pytest.approx(problem._log_prior(theta), rel=1e-14)
        assert problem.log_posterior(params) == pytest.approx(problem._log_free_posterior(theta), rel=1e-14)

    def test_public_log_methods_fill_in_the_default_nuisance_value(self, problem):
        with_default = problem.log_likelihood({"m": 1.5, "b": 1.0})
        explicit = problem.log_likelihood({"m": 1.5, "b": 1.0, "sys_dex": 0.1})
        assert with_default == pytest.approx(explicit, rel=1e-14)

    def test_log_transform_on_the_nuisance_parameter(self, problem):
        problem.parameters["sys_dex"].transform = Log10Transform()
        z = problem.initial_z
        assert z[2] == pytest.approx(np.log10(0.1))
        theta, _ = problem.transform_free_theta_inverse(z)
        assert theta[2] == pytest.approx(0.1)
        assert np.isfinite(problem._log_free_posterior(z))

    def test_prior_sampling_includes_the_nuisance_parameter(self, problem):
        draws = problem.sample_prior(50, rng=np.random.default_rng(0))
        assert draws.shape == (50, 3)
        assert np.all((draws[:, 2] >= 1e-3) & (draws[:, 2] <= 1.0))

    def test_validate_for_inference_requires_a_prior_on_the_nuisance_parameter(self, model, y_obs):
        problem = InferenceProblem(likelihood=LogGaussianCensoredSysLikelihood(model, _data(y_obs, 0.1 * y_obs)))
        problem.set_prior("m", "uniform", lower=0.0, upper=5.0)
        problem.set_prior("b", "uniform", lower=0.0, upper=5.0)
        with pytest.raises(RuntimeError, match="sys_dex"):
            problem.validate_for_inference()

        problem.set_prior("sys_dex", "loguniform", lower=1e-3, upper=1.0)
        problem.validate_for_inference()

    def test_summary_and_repr_mark_the_nuisance_parameter(self, problem):
        assert "sys_dex [nuisance]" in problem.summary()
        assert "sys_dex (nuisance)" in problem._repr_html_()

    def test_forward_model_accepts_the_full_parameter_dict(self, problem, model):
        """A problem's full parameter dict can be passed straight to the model, which ignores nuisance keys."""
        full = problem.initial_params
        assert "sys_dex" in full
        out = model.forward_model({"x": X}, full)
        np.testing.assert_allclose(np.asarray(out.y), 1.5 * X + 1.0)


# =================================================================== #
# Serialization                                                       #
# =================================================================== #
class TestSerialization:
    @pytest.mark.parametrize("via", ["dict", "json"])
    def test_round_trip_preserves_parameters_priors_and_posterior(self, problem, via):
        problem.parameters["sys_dex"].transform = Log10Transform()
        rebuilt = (
            InferenceProblem.from_dict(problem.to_dict())
            if via == "dict"
            else InferenceProblem.from_json(problem.to_json())
        )

        assert isinstance(rebuilt.likelihood, LogGaussianCensoredSysLikelihood)
        assert rebuilt.parameter_order == problem.parameter_order
        assert rebuilt.nuisance_parameter_names == ("sys_dex",)
        assert isinstance(rebuilt["sys_dex"].transform, Log10Transform)
        assert type(rebuilt["sys_dex"].prior) is type(problem["sys_dex"].prior)
        assert rebuilt["sys_dex"].initial_value == problem["sys_dex"].initial_value

        for z in (problem.initial_z, problem.initial_z + 0.05, problem.initial_z - 0.05):
            assert rebuilt._log_free_posterior(z) == pytest.approx(problem._log_free_posterior(z), rel=1e-12)

    def test_frozen_state_survives(self, problem):
        problem.freeze_parameters("sys_dex")
        rebuilt = InferenceProblem.from_dict(problem.to_dict())
        assert rebuilt.fixed_parameter_names == ("sys_dex",)

    def test_nuisance_entry_is_in_the_serialized_parameters(self, problem):
        spec = json.loads(problem.to_json())
        assert list(spec["parameters"]) == ["m", "b", "sys_dex"]
        assert spec["metadata"]["num_parameters"] == 3

    def test_specs_without_nuisance_parameters_still_load(self, model, y_obs):
        legacy = InferenceProblem(
            likelihood=LogGaussianCensoredLikelihood(model, _data(y_obs, 0.1 * y_obs), sys_dex=0.1)
        )
        rebuilt = InferenceProblem.from_dict(legacy.to_dict())
        assert rebuilt.parameter_order == ("m", "b")

    def test_unknown_parameter_name_raises(self, problem):
        spec = problem.to_dict()
        spec["parameters"]["not_a_parameter"] = spec["parameters"].pop("sys_dex") | {"name": "not_a_parameter"}
        with pytest.raises(TypeError, match="not_a_parameter"):
            InferenceProblem.from_dict(spec)

    def test_mp_pool_matches_serial(self, problem):
        thetas = [problem.initial_theta * f for f in ([1.0, 1.0, 1.0], [1.2, 0.9, 2.0], [0.8, 1.3, 0.5])]
        serial = [problem._log_free_posterior(t) for t in thetas]
        with LikelihoodMPPool(problem=problem, processes=2) as pool:
            parallel = pool.map(None, thetas)
        np.testing.assert_allclose(parallel, serial, rtol=1e-10)


# =================================================================== #
# Sampling                                                            #
# =================================================================== #
class TestSampling:
    def test_nuisance_parameter_is_sampled_and_reported(self, problem):
        sampler = EmceeSampler(problem, n_walkers=12)
        initial_state = sampler.generate_initial_state(from_prior=True, rng=np.random.default_rng(3))
        result = sampler.run(30, initial_state=initial_state, progress=False)

        assert result.parameter_names == ("m", "b", "sys_dex")
        flat = result.get_flat_samples()
        assert flat.shape[-1] == 3
        assert np.all(flat[:, 2] > 0)

    def test_physical_samples_map_the_transformed_nuisance_parameter(self, problem):
        problem.parameters["sys_dex"].transform = Log10Transform()
        sampler = EmceeSampler(problem, n_walkers=12)
        initial_state = sampler.generate_initial_state(from_prior=True, rng=np.random.default_rng(4))
        result = sampler.run(20, initial_state=initial_state, progress=False)

        physical = result.get_flat_samples(physical=True)
        raw = result.get_flat_samples(physical=False)
        np.testing.assert_allclose(physical[:, 2], 10.0 ** raw[:, 2], rtol=1e-12)
