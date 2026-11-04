import inspect

import numpy as np
import pytest

from trilobite.inference.prior import *

# ============================================================================== #
# Registry and Fixture Creation
# ============================================================================== #
# We create a catalog of priors in the PRIOR_REGISTRY so that we can initialize
# all of them and perform relevant tests. We also have a test to check that the
# prior registry is complete.
TEST_PRIOR_REGISTRY = {
    UniformPrior: {"lower": 0.0, "upper": 10.0},
    LogUniformPrior: {"lower": 1e-3, "upper": 1e3},
    NormalPrior: {"mean": 0.0, "sigma": 1.0},
    TruncatedNormalPrior: {
        "mean": 0.0,
        "sigma": 1.0,
        "lower": -2.0,
        "upper": 2.0,
    },
    HalfNormalPrior: {"sigma": 1.0},
    LogNormalPrior: {"mean": 0.0, "sigma": 1.0},
    GammaPrior: {"shape": 2.0, "scale": 1.0},
    BetaPrior: {"alpha": 2.0, "beta": 5.0},
}


# ============================================================================== #
# Tooling
# ============================================================================== #
def _get_concrete_prior_subclasses():
    subclasses = set()

    work = list(Prior.__subclasses__())

    while work:
        cls = work.pop()
        work.extend(cls.__subclasses__())

        if inspect.isabstract(cls):
            continue

        # Skip intentionally non-reconstructible types
        if cls.__name__ == "CallablePrior":
            continue

        subclasses.add(cls)

    return subclasses


# ============================================================================== #
# TESTS
# ============================================================================== #


# --- Prior Registry Completeness -------------------------------------------------- #
def test_all_priors_registered():
    subclasses = _get_concrete_prior_subclasses()

    registered = set(TEST_PRIOR_REGISTRY.keys())

    missing = subclasses - registered
    extra = registered - subclasses

    assert not missing, f"Missing priors in registry: {missing}"
    assert not extra, f"Registry contains unknown priors: {extra}"


def test_all_priors_in_public_registry():
    subclasses = _get_concrete_prior_subclasses()

    registered = set([value["class"] for value in PRIOR_REGISTRY.values()])

    missing = subclasses - registered
    extra = registered - subclasses

    assert not missing, f"Missing priors in registry: {missing}"
    assert not extra, f"Registry contains unknown priors: {extra}"


# --- Serialization Tests -------------------------------------------------- #
@pytest.mark.parametrize(
    "prior_cls, params",
    TEST_PRIOR_REGISTRY.items(),
)
def test_prior_serialization_roundtrip(prior_cls, params):
    prior = prior_cls(**params)

    # Serialize
    prior_dict = prior.to_dict()

    # Reconstruct
    recovered = Prior.from_dict(prior_dict)

    # Type check
    assert isinstance(recovered, prior_cls)

    # Parameter equality
    assert recovered._parameters == prior._parameters

    # Behavioral equivalence
    test_values = np.linspace(-2.0, 2.0, 7)

    for x in test_values:
        assert recovered.logp(x) == prior.logp(x)


# --- Sampling ------------------------------------------------------------------------ #
# Each case pairs a prior with the equivalent ``scipy.stats`` frozen distribution. The scipy object is
# built independently from the prior's own parametrization, so that disagreement between a prior's
# ``sample`` and its ``logp`` (e.g. natural-log versus dex conventions) is caught.
from scipy import stats  # noqa: E402

SAMPLING_CASES = {
    "uniform": (UniformPrior(0.0, 10.0), stats.uniform(0.0, 10.0)),
    "loguniform": (LogUniformPrior(1e-3, 1e3), stats.loguniform(1e-3, 1e3)),
    "normal": (NormalPrior(1.5, 2.0), stats.norm(1.5, 2.0)),
    "truncatednormal": (TruncatedNormalPrior(0.0, 1.0, -2.0, 2.0), stats.truncnorm(-2.0, 2.0, loc=0.0, scale=1.0)),
    "truncatednormal_tail": (TruncatedNormalPrior(0.0, 1.0, 8.0, 9.0), stats.truncnorm(8.0, 9.0, loc=0.0, scale=1.0)),
    "truncatednormal_lower_tail": (
        TruncatedNormalPrior(0.0, 1.0, -9.0, -8.0),
        stats.truncnorm(-9.0, -8.0, loc=0.0, scale=1.0),
    ),
    "halfnormal": (HalfNormalPrior(2.0), stats.halfnorm(scale=2.0)),
    "lognormal": (LogNormalPrior(2.0, 0.5), stats.lognorm(s=0.5, scale=np.exp(2.0))),
    "gamma": (GammaPrior(2.0, 3.0), stats.gamma(a=2.0, scale=3.0)),
    "beta": (BetaPrior(2.0, 5.0), stats.beta(2.0, 5.0)),
}


@pytest.mark.parametrize("name", SAMPLING_CASES)
def test_sample_matches_reference_distribution(name):
    prior, ref = SAMPLING_CASES[name]
    x = prior.sample(size=20000, rng=np.random.default_rng(0))

    assert stats.kstest(x, ref.cdf).pvalue > 1e-3


@pytest.mark.parametrize("name", SAMPLING_CASES)
def test_sample_is_consistent_with_logp(name):
    prior, ref = SAMPLING_CASES[name]
    x = prior.sample(size=500, rng=np.random.default_rng(1))

    # Every draw lies in the support, and logp agrees with the reference density.
    logp = np.array([prior.logp(xi) for xi in x])
    assert np.all(np.isfinite(logp))
    np.testing.assert_allclose(logp, ref.logpdf(x), rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("prior_cls, params", TEST_PRIOR_REGISTRY.items())
def test_sample_shape_and_reproducibility(prior_cls, params):
    prior = prior_cls(**params)

    assert np.ndim(prior.sample()) == 0
    assert prior.sample(size=5).shape == (5,)
    assert prior.sample(size=(3, 4)).shape == (3, 4)

    a = prior.sample(size=10, rng=np.random.default_rng(7))
    b = prior.sample(size=10, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


def test_callable_prior_cannot_be_sampled():
    prior = CallablePrior(lambda x: 0.0)
    with pytest.raises(NotImplementedError):
        prior.sample(size=3)
