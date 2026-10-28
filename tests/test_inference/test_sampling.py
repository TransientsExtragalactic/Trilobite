import warnings

import numpy as np
import pytest

from trilobite.data.core import InferenceData, Observable
from trilobite.inference import GaussianLikelihood, InferenceProblem
from trilobite.inference.sampling.mcmc import EmceeSampler
from trilobite.inference.sampling.result import MCMCSamplingResult
from trilobite.inference.sampling.utils import find_posterior_maximum
from trilobite.models.generic.curves import LinearModel

# =================================================================== #
# Fixtures                                                            #
# =================================================================== #


@pytest.fixture
def linear_problem():
    """Noise-free linear fit with uniform priors on [0, 5] and a configurable start."""
    model = LinearModel()
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = model.forward_model({"x": x}, {"m": 2.0, "b": 1.0}).y

    data = InferenceData(x={"x": x}, observables={"y": Observable(value=y, error=0.1 * y)})
    problem = InferenceProblem(likelihood=GaussianLikelihood(model=model, data=data))
    problem.set_prior("m", "uniform", lower=0.0, upper=5.0)
    problem.set_prior("b", "uniform", lower=0.0, upper=5.0)
    problem.parameters["m"].initial_value = 1.0
    problem.parameters["b"].initial_value = 0.5
    return problem


# =================================================================== #
# Initial state                                                       #
# =================================================================== #


class TestGenerateInitialState:
    def test_default_ball_is_one_percent(self, linear_problem):
        sampler = EmceeSampler(linear_problem, n_walkers=4000)
        positions = sampler.generate_initial_state(optimize=False, rng=np.random.default_rng(0))

        log_ratio = np.log(positions / linear_problem.initial_theta)
        np.testing.assert_allclose(log_ratio.std(axis=0), 1e-2, rtol=0.1)

    def test_optimize_centres_ball_on_posterior_maximum(self, linear_problem):
        # The data are noise-free, so the posterior peaks at the true (m, b) = (2, 1), far from the start (1, 0.5).
        sampler = EmceeSampler(linear_problem, n_walkers=500)
        positions = sampler.generate_initial_state(rng=np.random.default_rng(0))

        np.testing.assert_allclose(np.median(positions, axis=0), [2.0, 1.0], rtol=1e-2)

    def test_optimize_false_keeps_ball_on_initial_point(self, linear_problem):
        sampler = EmceeSampler(linear_problem, n_walkers=500)
        positions = sampler.generate_initial_state(optimize=False, rng=np.random.default_rng(0))

        np.testing.assert_allclose(np.median(positions, axis=0), [1.0, 0.5], rtol=1e-2)

    def test_reproducible_with_seeded_rng(self, linear_problem):
        sampler = EmceeSampler(linear_problem, n_walkers=16)
        a = sampler.generate_initial_state(rng=np.random.default_rng(3))
        b = sampler.generate_initial_state(rng=np.random.default_rng(3))
        np.testing.assert_array_equal(a, b)

    def test_walkers_outside_prior_are_redrawn(self, linear_problem):
        # Start a hair below the upper bound so roughly half of the raw draws leave the prior.
        linear_problem.parameters["m"].initial_value = 4.99
        sampler = EmceeSampler(linear_problem, n_walkers=64)

        positions = sampler.generate_initial_state(optimize=False, scale=5e-2, rng=np.random.default_rng(1))

        assert np.all(positions[:, 0] < 5.0)
        assert all(np.isfinite(linear_problem._log_free_posterior(p)) for p in positions)

    def test_raises_when_walkers_cannot_be_placed(self, linear_problem):
        linear_problem.parameters["m"].initial_value = 4.99
        sampler = EmceeSampler(linear_problem, n_walkers=64)

        with pytest.raises(RuntimeError, match="finite log-posterior"):
            sampler.generate_initial_state(optimize=False, scale=5e-2, rng=np.random.default_rng(1), max_attempts=0)


# =================================================================== #
# Posterior maximisation                                              #
# =================================================================== #


class TestFindPosteriorMaximum:
    def test_finds_maximum_across_scales(self):
        # Anisotropic Gaussian whose centre spans many orders of magnitude.
        centre = np.array([3.0e-3, 4.0e5])
        width = 0.05 * centre

        def logp(theta):
            return -0.5 * np.sum(((theta - centre) / width) ** 2)

        found = find_posterior_maximum(logp, np.array([1.0e-3, 1.0e5]))
        np.testing.assert_allclose(found, centre, rtol=1e-4)

    def test_never_worse_than_start(self):
        # A flat posterior gives the optimiser nothing to improve on.
        theta0 = np.array([1.0, 2.0])
        np.testing.assert_array_equal(find_posterior_maximum(lambda t: 0.0, theta0), theta0)

    def test_survives_non_finite_regions(self):
        # Hard wall at theta > 2: the start is inside, the maximum is at the wall's edge.
        def logp(theta):
            return -np.inf if theta[0] > 2.0 else -((theta[0] - 5.0) ** 2)

        found = find_posterior_maximum(logp, np.array([1.0]))
        assert 1.99 < found[0] <= 2.0


# =================================================================== #
# Stranded walkers                                                    #
# =================================================================== #


def _make_result(problem, logp, n_steps=40):
    n_walkers = logp.shape[1]
    samples = np.random.default_rng(0).normal(size=(n_steps, n_walkers, problem.n_dim))
    return MCMCSamplingResult(samples=samples, problem=problem, log_posterior=logp)


class TestStrandedWalkers:
    def test_flags_only_the_stranded_walkers(self, linear_problem):
        rng = np.random.default_rng(0)
        logp = 100.0 + rng.normal(scale=1.0, size=(40, 8))
        logp[:, [2, 5]] = -1.0e5

        result = _make_result(linear_problem, logp)

        np.testing.assert_array_equal(result.find_stranded_walkers(), [2, 5])

    def test_healthy_ensemble_flags_nothing(self, linear_problem):
        logp = 100.0 + np.random.default_rng(0).normal(scale=1.0, size=(40, 8))
        assert _make_result(linear_problem, logp).find_stranded_walkers().size == 0

    def test_infinite_log_posterior_counts_as_stranded(self, linear_problem):
        logp = 100.0 + np.random.default_rng(0).normal(scale=1.0, size=(40, 8))
        logp[:, 3] = -np.inf

        np.testing.assert_array_equal(_make_result(linear_problem, logp).find_stranded_walkers(), [3])

    def test_no_log_posterior_returns_empty(self, linear_problem):
        samples = np.zeros((10, 8, linear_problem.n_dim))
        result = MCMCSamplingResult(samples=samples, problem=linear_problem)
        assert result.find_stranded_walkers().size == 0

    def test_exclude_walkers_in_flat_samples(self, linear_problem):
        logp = np.zeros((40, 8))
        result = _make_result(linear_problem, logp)

        flat = result.get_flat_samples(burn=10, exclude_walkers=[2, 5])
        assert flat.shape == (30 * 6, linear_problem.n_dim)

        expected = np.delete(result.samples[10:], [2, 5], axis=1).reshape(-1, linear_problem.n_dim)
        np.testing.assert_array_equal(flat, expected)

    def test_run_does_not_warn_on_a_healthy_ensemble(self, linear_problem):
        sampler = EmceeSampler(linear_problem, n_walkers=16)
        initial_state = sampler.generate_initial_state(rng=np.random.default_rng(2))

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = sampler.run(400, initial_state=initial_state, progress=False)

        assert result.find_stranded_walkers().size == 0
