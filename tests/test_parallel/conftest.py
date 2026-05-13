"""
Fixtures shared across parallel module tests.

The ``simple_problem`` fixture creates a minimal but complete
:class:`~triceratops.inference.problem.InferenceProblem` that is JSON-serializable and
suitable for exercising all pool backends.

The ``mpi4py_available`` fixture exposes whether ``mpi4py`` is installed so that
individual tests can skip gracefully when the optional dependency is absent.
"""

import numpy as np
import pytest

from triceratops.data.core import InferenceData, Observable
from triceratops.inference import GaussianLikelihood, InferenceProblem
from triceratops.models.generic.curves import LinearModel


@pytest.fixture(scope="module")
def simple_problem():
    """
    Return a fully-configured :class:`~triceratops.inference.problem.InferenceProblem`.

    Uses a :class:`~triceratops.models.generic.curves.LinearModel` (y = m*x + b) with
    synthetic Gaussian observations.  Both parameters have :class:`~triceratops.inference.prior.UniformPrior`
    priors so the problem is immediately ready for likelihood evaluation.

    Returns
    -------
    InferenceProblem
        A problem with free parameters ``m`` and ``b``.
    """
    rng = np.random.default_rng(42)

    model = LinearModel()
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    true_params = {"m": 2.0, "b": 1.0}
    y_true = model.forward_model({"x": x}, true_params).y
    y_obs = rng.normal(loc=y_true, scale=0.1 * np.abs(y_true) + 1e-3)

    data = InferenceData(
        x={"x": x},
        observables={
            "y": Observable(
                value=y_obs,
                error=0.1 * np.abs(y_true) + 1e-3,
            )
        },
    )

    likelihood = GaussianLikelihood(model=model, data=data)
    problem = InferenceProblem(likelihood=likelihood)
    problem.set_prior("m", "uniform", lower=0.0, upper=5.0)
    problem.set_prior("b", "uniform", lower=0.0, upper=5.0)
    problem.parameters["m"].initial_value = 2.0
    problem.parameters["b"].initial_value = 1.0

    return problem


@pytest.fixture(scope="session")
def mpi4py_available():
    """Return ``True`` if ``mpi4py`` is importable, ``False`` otherwise."""
    try:
        import mpi4py  # noqa: F401

        return True
    except ImportError:
        return False
