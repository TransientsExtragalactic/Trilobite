"""Regression tests: likelihood configuration must survive InferenceProblem serialization (and so the MP pool)."""

import json

import numpy as np
import pytest

from trilobite.data.core import InferenceData, Observable
from trilobite.inference import GaussianLikelihood, InferenceProblem, LogGaussianCensoredLikelihood
from trilobite.inference.likelihood.utils import get_likelihood_config
from trilobite.models.generic.curves import LinearModel
from trilobite.parallel.mp import LikelihoodMPPool

X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
TRUE = {"m": 2.0, "b": 1.0}


def _problem(likelihood_cls=LogGaussianCensoredLikelihood, **like_kwargs):
    model = LinearModel()
    y = np.asarray(model.forward_model({"x": X}, TRUE).y, dtype=float)
    y_obs = y * np.array([1.1, 0.9, 1.05, 0.8, 1.2])
    data = InferenceData(x={"x": X}, observables={"y": Observable(value=y_obs, error=0.1 * y_obs)})
    problem = InferenceProblem(likelihood=likelihood_cls(model, data, **like_kwargs))
    problem.set_prior("m", "uniform", lower=0.0, upper=5.0)
    problem.set_prior("b", "uniform", lower=0.0, upper=5.0)
    problem.parameters["m"].initial_value = 1.5
    problem.parameters["b"].initial_value = 1.0
    return problem


def _thetas(problem):
    theta0 = problem.initial_theta
    return [theta0 * f for f in ([1.0, 1.0], [1.2, 0.9], [0.8, 1.3], [1.5, 0.5])]


@pytest.mark.parametrize("via", ["dict", "json"])
def test_sys_dex_survives_round_trip(via):
    problem = _problem(sys_dex=0.3)
    if via == "dict":
        rebuilt = InferenceProblem.from_dict(problem.to_dict())
    else:
        rebuilt = InferenceProblem.from_json(problem.to_json())

    assert rebuilt.likelihood.config == {"sys_dex": 0.3}
    assert rebuilt.likelihood._sys_dex == 0.3
    for theta in _thetas(problem):
        assert rebuilt._log_free_posterior(theta) == pytest.approx(problem._log_free_posterior(theta), rel=1e-12)


def test_likelihood_without_options_round_trips():
    problem = _problem(GaussianLikelihood)
    spec = problem.to_dict()
    assert spec["likelihood_kwargs"] == {}
    rebuilt = InferenceProblem.from_dict(spec)
    theta = problem.initial_theta
    assert rebuilt._log_free_posterior(theta) == pytest.approx(problem._log_free_posterior(theta), rel=1e-12)


def test_spec_written_before_likelihood_kwargs_still_loads():
    spec = _problem(sys_dex=0.0).to_dict()
    del spec["likelihood_kwargs"]
    rebuilt = InferenceProblem.from_dict(spec)
    assert rebuilt.likelihood.config == {}


def test_numpy_scalar_options_are_json_safe():
    problem = _problem(sys_dex=np.float64(0.25))
    spec = json.loads(problem.to_json())
    assert spec["likelihood_kwargs"] == {"sys_dex": 0.25}
    assert type(get_likelihood_config(problem.likelihood)["sys_dex"]) is float


def test_non_serializable_option_raises_instead_of_being_dropped():
    problem = _problem(sys_dex=0.1)
    problem.likelihood._config["callback"] = object()
    with pytest.raises(TypeError, match="callback"):
        problem.to_dict()


def test_config_property_returns_a_copy():
    like = _problem(sys_dex=0.1).likelihood
    like.config["sys_dex"] = 99.0
    assert like.config == {"sys_dex": 0.1}


def test_mp_pool_matches_serial_for_configured_likelihood():
    """The symptom reported in the issue: pool results must equal serial results when sys_dex != 0."""
    problem = _problem(sys_dex=0.3)
    thetas = _thetas(problem)
    serial = [problem._log_free_posterior(t) for t in thetas]
    with LikelihoodMPPool(problem=problem, processes=2) as pool:
        parallel = pool.map(None, thetas)
    np.testing.assert_allclose(parallel, serial, rtol=1e-10)
