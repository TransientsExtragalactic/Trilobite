import pathlib

import numpy as np
import pytest

from trilobite.data.core import InferenceData, Observable
from trilobite.inference import GaussianLikelihood, InferenceProblem, UniformPrior
from trilobite.inference.sampling.mcmc import EmceeSampler
from trilobite.models.generic.curves import LinearModel

np.random.seed(42)


# =================================================================== #
# Simple Problem Fixture                                              #
# =================================================================== #
@pytest.fixture
def simple_problem():
    # Create the data by adding a 10% error to a basic linear model.
    model = LinearModel()
    x = np.array([0, 1, 2, 3, 4])
    true_params = {"m": 2.0, "b": 1.0}
    y = model.forward_model({"x": x}, true_params).y
    y_obs = np.random.normal(loc=y, scale=0.1 * y)  # Add 10% Gaussian noise

    # Define the data
    data = InferenceData(
        x={"x": x},
        observables={
            "y": Observable(
                value=y_obs,
                error=0.1 * y,  # 10% error
            )
        },
    )

    # Define the likelihood
    likelihood = GaussianLikelihood(model=model, data=data)

    # Create the inference problem
    problem = InferenceProblem(likelihood=likelihood)

    # Set the priors on the parameters
    problem.set_prior("m", "uniform", lower=0.0, upper=5.0)
    problem.set_prior("b", "uniform", lower=0.0, upper=5.0)

    # Set initial values for the parameters
    problem.parameters["m"].initial_value = 1.0
    problem.parameters["b"].initial_value = 0.5

    # Return the problem and the true parameters for testing
    return problem, true_params


def test_inference_problem_roundtrip(simple_problem):
    problem, true_params = simple_problem

    # ---------------------------------------------------------------
    # Evaluate baseline posterior
    # ---------------------------------------------------------------
    theta0 = problem.initial_theta
    baseline_logp = problem._log_free_posterior(theta0)

    # ---------------------------------------------------------------
    # Serialize to dict
    # ---------------------------------------------------------------
    spec = problem.to_dict()

    # ---------------------------------------------------------------
    # Reconstruct
    # ---------------------------------------------------------------
    reconstructed = InferenceProblem.from_dict(spec)

    # ---------------------------------------------------------------
    # Basic structural checks
    # ---------------------------------------------------------------
    assert reconstructed.n_parameters == problem.n_parameters
    assert reconstructed.n_free_parameters == problem.n_free_parameters
    assert reconstructed.parameter_order == problem.parameter_order

    # ---------------------------------------------------------------
    # Check priors survived
    # ---------------------------------------------------------------
    for name in problem.free_parameter_names:
        assert reconstructed.parameters[name].prior is not None
        assert type(reconstructed.parameters[name].prior) is type(problem.parameters[name].prior)

    # ---------------------------------------------------------------
    # Check initial values survived
    # ---------------------------------------------------------------
    for name in problem.parameter_order:
        assert np.isclose(
            reconstructed.parameters[name].initial_value,
            problem.parameters[name].initial_value,
        )

    # ---------------------------------------------------------------
    # Posterior consistency
    # ---------------------------------------------------------------
    theta_new = reconstructed.initial_theta
    reconstructed_logp = reconstructed._log_free_posterior(theta_new)

    assert np.isclose(baseline_logp, reconstructed_logp)

    # ---------------------------------------------------------------
    # JSON round-trip
    # ---------------------------------------------------------------
    json_str = problem.to_json()
    reconstructed_json = InferenceProblem.from_json(json_str)

    theta_json = reconstructed_json.initial_theta
    logp_json = reconstructed_json._log_free_posterior(theta_json)

    assert np.isclose(baseline_logp, logp_json)


def test_mcmc_converges_near_truth(
    simple_problem,
    diagnostic_plots,
    diagnostic_plots_dir,
):
    problem, true_params = simple_problem

    sampler = EmceeSampler(problem, n_walkers=16)
    result = sampler.run(1000, progress=True)

    samples = result.get_flat_samples(burn=50, thin=2)
    mean_estimate = samples.mean(axis=0)

    # ------------------------------------------------------------------
    # Convergence check (loose but stable)
    # ------------------------------------------------------------------
    for i, name in enumerate(problem.free_parameter_names):
        assert np.isclose(mean_estimate[i], true_params[name], atol=0.2)

    # ------------------------------------------------------------------
    # Optional diagnostic plot
    # ------------------------------------------------------------------
    if diagnostic_plots:
        output_path = pathlib.Path(diagnostic_plots_dir) / "mcmc_corner_plot.png"

        fig = result.corner_plot(
            burn=500,
            thin=2,
            truths=true_params,
        )

        fig.savefig(output_path, dpi=150)
