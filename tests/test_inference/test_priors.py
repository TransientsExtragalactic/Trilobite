import inspect

import numpy as np
import pytest

from triceratops.inference.prior import *

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
