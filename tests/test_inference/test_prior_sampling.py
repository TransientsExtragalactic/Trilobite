"""Tests for ``InferenceProblem.sample_prior`` and the ``set_prior`` registry / unit handling."""

from collections import namedtuple

import numpy as np
import pytest
from astropy import units as u
from scipy import stats

from trilobite.data.core import InferenceData, Observable
from trilobite.inference import GaussianLikelihood, InferenceProblem
from trilobite.inference.prior import (
    PRIOR_REGISTRY,
    BetaPrior,
    CallablePrior,
    GammaPrior,
    LogNormalPrior,
    NormalPrior,
    TruncatedNormalPrior,
)
from trilobite.inference.transform import Log10Transform, LogTransform
from trilobite.models.core import Model, ModelParameter, ModelVariable
from trilobite.models.generic.curves import LinearModel

# =================================================================== #
# Fixtures                                                            #
# =================================================================== #
_Out = namedtuple("Out", ["y"])


class _FrequencyModel(Model):
    """y = A * x / nu, with ``nu`` in Hz and ``A`` dimensionless (for unit-handling tests)."""

    VARIABLES = (ModelVariable("x", base_units=None),)
    PARAMETERS = (
        ModelParameter("A", 1.0, base_units=None),
        ModelParameter("nu", 1.0e9, base_units=u.Hz),
    )
    OUTPUTS = _Out
    UNITS = OUTPUTS(y=u.dimensionless_unscaled)
    DESCRIPTION = "Test model with a frequency parameter."
    REFERENCE = "None"

    def __init__(self):
        self._register_init()

    def _forward_model(self, variables, parameters):
        x = np.asarray(variables["x"], dtype=float)
        return self.OUTPUTS(y=parameters["A"] * x / parameters["nu"])


def _make_problem(model=None, true_params=None):
    model = model or LinearModel()
    true_params = true_params or {"m": 2.0, "b": 1.0}
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = model.forward_model({"x": x}, true_params).y
    data = InferenceData(x={"x": x}, observables={"y": Observable(value=y, error=0.1 * np.abs(y))})
    return InferenceProblem(likelihood=GaussianLikelihood(model=model, data=data))


@pytest.fixture
def linear_problem():
    problem = _make_problem()
    problem.set_prior("m", "uniform", lower=0.0, upper=5.0)
    problem.set_prior("b", "loguniform", lower=1e-2, upper=1e2)
    return problem


@pytest.fixture
def freq_problem():
    problem = _make_problem(_FrequencyModel(), {"A": 1.0, "nu": 1.0e9})
    problem.set_prior("A", "uniform", lower=0.0, upper=5.0)
    problem.set_prior("nu", "uniform", lower=1.0e8, upper=1.0e10)
    return problem


# =================================================================== #
# sample_prior                                                        #
# =================================================================== #
class TestSamplePrior:
    def test_shape_and_support(self, linear_problem):
        z = linear_problem.sample_prior(300, rng=np.random.default_rng(0))

        assert z.shape == (300, 2)
        assert np.all((z[:, 0] >= 0.0) & (z[:, 0] <= 5.0))
        assert np.all((z[:, 1] >= 1e-2) & (z[:, 1] <= 1e2))

    def test_reproducible_with_seeded_rng(self, linear_problem):
        a = linear_problem.sample_prior(10, rng=np.random.default_rng(3))
        b = linear_problem.sample_prior(10, rng=np.random.default_rng(3))
        np.testing.assert_array_equal(a, b)

    def test_draws_are_in_sampling_space(self, linear_problem):
        linear_problem.parameters["b"].transform = Log10Transform()
        z = linear_problem.sample_prior(2000, rng=np.random.default_rng(0))

        # A log-uniform prior on [1e-2, 1e2] is uniform in log10 on [-2, 2].
        assert np.all((z[:, 1] >= -2.0) & (z[:, 1] <= 2.0))
        assert stats.kstest(z[:, 1], stats.uniform(-2.0, 4.0).cdf).pvalue > 1e-3

    def test_uniform_prior_with_log_transform_gives_jacobian_weighted_density(self, linear_problem):
        # theta ~ U(1, 10) with z = ln(theta) has density p(z) = e^z / 9 on [0, ln 10] -- the prior times |dtheta/dz|.
        linear_problem.set_prior("b", "uniform", lower=1.0, upper=10.0)
        linear_problem.parameters["b"].transform = LogTransform()
        z = linear_problem.sample_prior(5000, rng=np.random.default_rng(0))[:, 1]

        cdf = lambda v: (np.exp(v) - 1.0) / 9.0  # noqa: E731
        assert stats.kstest(z, cdf).pvalue > 1e-3

    def test_draws_have_finite_log_posterior(self, linear_problem):
        linear_problem.parameters["b"].transform = Log10Transform()
        for z in linear_problem.sample_prior(25, rng=np.random.default_rng(0)):
            assert np.isfinite(linear_problem._log_free_posterior(z))

    def test_frozen_parameters_are_excluded(self, linear_problem):
        linear_problem.parameters["b"].freeze = True
        z = linear_problem.sample_prior(7, rng=np.random.default_rng(0))

        assert z.shape == (7, 1)
        assert np.all((z[:, 0] >= 0.0) & (z[:, 0] <= 5.0))

    @pytest.mark.parametrize("n", [0, -1, 2.5])
    def test_invalid_n_raises(self, linear_problem, n):
        with pytest.raises(ValueError, match="positive integer"):
            linear_problem.sample_prior(n)

    def test_unsampleable_prior_raises_naming_parameter(self, linear_problem):
        linear_problem.parameters["m"].prior = CallablePrior(lambda x: 0.0)
        with pytest.raises(RuntimeError, match="'m'"):
            linear_problem.sample_prior(5)

        # A bare callable (no ``sample`` attribute at all) is also reported rather than leaking an AttributeError.
        linear_problem.parameters["m"].prior = lambda x: 0.0
        with pytest.raises(RuntimeError, match="'m'"):
            linear_problem.sample_prior(5)

    def test_missing_prior_raises(self, linear_problem):
        linear_problem.parameters["m"].prior = None
        with pytest.raises(RuntimeError, match="No prior"):
            linear_problem.sample_prior(5)

    def test_draws_outside_transform_domain_raise(self, linear_problem):
        linear_problem.set_prior("b", "normal", mean=0.0, sigma=1.0)
        linear_problem.parameters["b"].transform = Log10Transform()
        with pytest.raises(RuntimeError, match="'b'.*transform"):
            linear_problem.sample_prior(50, rng=np.random.default_rng(0))

    def test_scalar_only_transform_is_supported(self, linear_problem):
        class ScalarLog10(Log10Transform):
            def _make_forward(self):
                return lambda theta: float(np.log10(float(theta)))

        linear_problem.parameters["b"].transform = ScalarLog10()
        z = linear_problem.sample_prior(10, rng=np.random.default_rng(0))
        assert z.shape == (10, 2)


# =================================================================== #
# set_prior: registry coverage                                        #
# =================================================================== #
class TestSetPriorRegistry:
    def test_registry_is_shared_with_prior_module(self):
        assert InferenceProblem._PRIOR_REGISTRY is PRIOR_REGISTRY

    @pytest.mark.parametrize(
        "prior_type, kwargs, expected",
        [
            ("truncatednormal", {"mean": 1.0, "sigma": 0.5, "lower": 0.0, "upper": 2.0}, TruncatedNormalPrior),
            ("gamma", {"shape": 2.0, "scale": 1.5}, GammaPrior),
            ("beta", {"alpha": 2.0, "beta": 3.0}, BetaPrior),
        ],
    )
    def test_newly_available_priors(self, linear_problem, prior_type, kwargs, expected):
        linear_problem.set_prior("m", prior_type, **kwargs)
        prior = linear_problem.parameters["m"].prior

        assert isinstance(prior, expected)
        assert np.all(np.isfinite(linear_problem.sample_prior(20, rng=np.random.default_rng(0))))

    def test_unknown_prior_lists_available(self, linear_problem):
        with pytest.raises(ValueError, match="truncatednormal"):
            linear_problem.set_prior("m", "nonsense")


# =================================================================== #
# set_prior: units                                                    #
# =================================================================== #
class TestSetPriorUnits:
    def test_unit_bearing_bounds_are_converted(self, freq_problem):
        freq_problem.set_prior("nu", "uniform", lower=1 * u.GHz, upper=10 * u.GHz)
        assert freq_problem.parameters["nu"].prior._parameters == {"lower": 1e9, "upper": 1e10}

    def test_lognormal_median_with_units(self, freq_problem):
        freq_problem.set_prior("nu", "lognormal", median=3 * u.GHz, sigma=0.5)
        prior = freq_problem.parameters["nu"].prior

        assert isinstance(prior, LogNormalPrior)
        assert prior._parameters["mean"] == pytest.approx(np.log(3e9))
        assert prior._parameters["sigma"] == 0.5

    def test_lognormal_median_is_independent_of_the_unit_it_is_given_in(self, freq_problem):
        freq_problem.set_prior("nu", "lognormal", median=3 * u.GHz, sigma=0.5)
        a = freq_problem.parameters["nu"].prior._parameters["mean"]
        freq_problem.set_prior("nu", "lognormal", median=3000 * u.MHz, sigma=0.5)
        b = freq_problem.parameters["nu"].prior._parameters["mean"]

        assert a == pytest.approx(b)

    def test_lognormal_bare_median_is_in_base_units(self, freq_problem):
        freq_problem.set_prior("nu", "lognormal", median=3e9, sigma=0.5)
        assert freq_problem.parameters["nu"].prior._parameters["mean"] == pytest.approx(np.log(3e9))

    def test_lognormal_median_matches_sample_median(self, freq_problem):
        freq_problem.set_prior("nu", "lognormal", median=3 * u.GHz, sigma=0.3)
        draws = freq_problem.sample_prior(20000, rng=np.random.default_rng(0))[:, 1]
        assert np.median(draws) == pytest.approx(3e9, rel=0.02)

    def test_lognormal_mean_is_a_bare_log(self, freq_problem):
        freq_problem.set_prior("nu", "lognormal", mean=np.log(3e9), sigma=0.5)
        assert freq_problem.parameters["nu"].prior._parameters["mean"] == pytest.approx(np.log(3e9))

    def test_lognormal_mean_and_median_together_raise(self, freq_problem):
        with pytest.raises(ValueError, match="not both"):
            freq_problem.set_prior("nu", "lognormal", mean=1.0, median=3 * u.GHz, sigma=0.5)

    def test_lognormal_nonpositive_median_raises(self, freq_problem):
        with pytest.raises(ValueError, match="positive"):
            freq_problem.set_prior("nu", "lognormal", median=-1 * u.GHz, sigma=0.5)

    def test_incompatible_median_units_raise(self, freq_problem):
        with pytest.raises(u.UnitConversionError):
            freq_problem.set_prior("nu", "lognormal", median=3 * u.s, sigma=0.5)

    @pytest.mark.parametrize("key", ["mean", "sigma"])
    def test_dimensional_quantity_for_dimensionless_argument_raises(self, freq_problem, key):
        kwargs = {"mean": 1.0, "sigma": 0.5}
        kwargs[key] = 1 * u.GHz
        with pytest.raises(ValueError, match=key):
            freq_problem.set_prior("nu", "lognormal", **kwargs)

    def test_dimensionless_quantity_is_reduced_to_float(self, freq_problem):
        freq_problem.set_prior("nu", "lognormal", mean=1.0 * u.dimensionless_unscaled, sigma=50 * u.percent)
        params = freq_problem.parameters["nu"].prior._parameters
        assert params == {"mean": 1.0, "sigma": 0.5}
        assert all(type(v) is float for v in params.values())

    def test_normal_prior_still_converts_units(self, freq_problem):
        freq_problem.set_prior("nu", "normal", mean=5 * u.GHz, sigma=500 * u.MHz)
        prior = freq_problem.parameters["nu"].prior

        assert isinstance(prior, NormalPrior)
        assert prior._parameters == {"mean": 5e9, "sigma": 5e8}
