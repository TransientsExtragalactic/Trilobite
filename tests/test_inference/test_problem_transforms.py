"""Regression tests: parameter transforms must be inverted exactly once in the log-posterior."""

import numpy as np
import pytest

from trilobite.data.core import InferenceData, Observable
from trilobite.inference import GaussianLikelihood, InferenceProblem
from trilobite.inference.sampling.mcmc import EmceeSampler
from trilobite.inference.transform import Log10Transform
from trilobite.models.generic.curves import LinearModel

TRUE = {"m": 2.0, "b": 1.0}


def _make_problem(transform_m=False, transform_b=False, freeze_b=False):
    model = LinearModel()
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = model.forward_model({"x": x}, TRUE).y
    rng = np.random.default_rng(0)
    y_obs = rng.normal(loc=y, scale=0.1 * y)
    data = InferenceData(x={"x": x}, observables={"y": Observable(value=y_obs, error=0.1 * y)})
    problem = InferenceProblem(likelihood=GaussianLikelihood(model=model, data=data))
    problem.set_prior("m", "uniform", lower=0.0, upper=5.0)
    problem.set_prior("b", "uniform", lower=0.0, upper=5.0)
    problem.parameters["m"].initial_value = 1.5
    problem.parameters["b"].initial_value = 1.0
    if transform_m:
        problem.parameters["m"].transform = Log10Transform()
    if transform_b:
        problem.parameters["b"].transform = Log10Transform()
    if freeze_b:
        problem.parameters["b"].freeze = True
    return problem


def _log10_jacobian(theta):
    """log|d theta / d z| for a Log10Transform, expressed in the physical value."""
    return np.log(theta * np.log(10.0))


def test_unpack_is_inverse_of_pack_with_transforms():
    problem = _make_problem(transform_m=True, transform_b=True)
    theta = problem.pack_parameters(TRUE)
    assert problem.unpack_parameters(theta) == pytest.approx(TRUE)


def test_log_likelihood_does_not_invert_transforms():
    """The likelihood must see the physical values, whatever transforms are set."""
    plain = _make_problem()
    transformed = _make_problem(transform_m=True, transform_b=True)
    theta = plain.pack_parameters(TRUE)
    assert transformed._log_likelihood(theta) == pytest.approx(plain._log_likelihood(theta), rel=1e-12)
    assert transformed.log_likelihood(TRUE) == pytest.approx(plain.log_likelihood(TRUE), rel=1e-12)


def test_log_posterior_inverts_once():
    plain = _make_problem()
    transformed = _make_problem(transform_m=True)
    z = np.array([np.log10(TRUE["m"]), TRUE["b"]])
    expected = plain._log_posterior(np.array([TRUE["m"], TRUE["b"]])) + _log10_jacobian(TRUE["m"])
    assert transformed._log_posterior(z) == pytest.approx(expected, rel=1e-12)
    # The previous behaviour evaluated the likelihood at m = 10**(10**z) = 100.
    assert np.isfinite(transformed._log_posterior(z))
    assert transformed._log_posterior(z) > -1e3


def test_log_free_posterior_all_free_matches_full():
    problem = _make_problem(transform_m=True, transform_b=True)
    z = np.array([np.log10(TRUE["m"]), np.log10(TRUE["b"])])
    assert problem._log_free_posterior(z) == pytest.approx(problem._log_posterior(z), rel=1e-12)


def test_frozen_parameters_stay_physical_and_have_no_jacobian():
    """A frozen (even transformed) parameter keeps its physical initial_value and adds no Jacobian term."""
    plain = _make_problem(freeze_b=True)
    transformed = _make_problem(transform_m=True, transform_b=True, freeze_b=True)
    z_free = np.array([np.log10(TRUE["m"])])
    expected = plain._log_free_posterior(np.array([TRUE["m"]])) + _log10_jacobian(TRUE["m"])
    assert transformed._log_free_posterior(z_free) == pytest.approx(expected, rel=1e-12)


def test_call_accepts_free_vector_with_frozen_parameter():
    problem = _make_problem(transform_m=True, freeze_b=True)
    z_free = problem.initial_z
    assert z_free.shape == (1,)
    assert problem(z_free) == pytest.approx(problem._log_free_posterior(z_free), rel=1e-12)
    assert problem.initial_log_posterior == pytest.approx(problem(z_free), rel=1e-12)


def test_public_dict_api_takes_physical_values():
    plain = _make_problem()
    transformed = _make_problem(transform_m=True)
    expected = plain.log_posterior(TRUE) + _log10_jacobian(TRUE["m"])
    assert transformed.log_posterior(TRUE) == pytest.approx(expected, rel=1e-12)
    assert transformed.log_free_posterior(TRUE) == pytest.approx(expected, rel=1e-12)
    assert transformed.log_prior(TRUE) == pytest.approx(plain.log_prior(TRUE), rel=1e-12)


def test_sampler_initial_state_is_in_sampling_space():
    problem = _make_problem(transform_m=True)
    sampler = EmceeSampler(problem, n_walkers=8)
    positions = sampler.generate_initial_state(optimize=False, scale=0.01, rng=np.random.default_rng(1))
    assert positions.shape == (8, 2)
    # First coordinate is log10(m): centred near log10(1.5), not near 1.5.
    assert np.allclose(positions[:, 0], np.log10(1.5), atol=0.1)
    assert np.all(np.isfinite([problem._log_free_posterior(p) for p in positions]))
