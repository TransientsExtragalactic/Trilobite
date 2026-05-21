import inspect

import numpy as np
import pytest

from trilobite.inference.transform import *

# ============================================================================== #
# Registry and Fixture Creation
# ============================================================================== #
TRANSFORM_REGISTRY = {
    IdentityTransform: {
        "kwargs": {},
        "theta_values": np.array([-2.0, 0.0, 3.0]),
    },
    LogTransform: {
        "kwargs": {},
        "theta_values": np.array([0.1, 1.0, 10.0]),
    },
    Log10Transform: {
        "kwargs": {},
        "theta_values": np.array([0.1, 1.0, 10.0]),
    },
    LogisticTransform: {
        "kwargs": {"lower": 0.0, "upper": 1.0},
        "theta_values": np.array([0.1, 0.5, 0.9]),
    },
    SoftplusTransform: {
        "kwargs": {},
        "theta_values": np.array([0.1, 1.0, 2.0]),
    },
}


# ============================================================================== #
# Tooling
# ============================================================================== #
def _get_concrete_transform_subclasses():
    subclasses = set()

    work = list(ParameterTransform.__subclasses__())

    while work:
        cls = work.pop()
        work.extend(cls.__subclasses__())

        if inspect.isabstract(cls):
            continue

        subclasses.add(cls)

    return subclasses


# ============================================================================== #
# TESTS
# ============================================================================== #


# --- Registry Completeness -------------------------------------------------- #
def test_all_transforms_registered():
    subclasses = _get_concrete_transform_subclasses()
    registered = set(TRANSFORM_REGISTRY.keys())

    missing = subclasses - registered
    extra = registered - subclasses

    assert not missing, f"Missing transforms in registry: {missing}"
    assert not extra, f"Registry contains unknown transforms: {extra}"


# --- Forward / Inverse Consistency ----------------------------------------- #
@pytest.mark.parametrize(
    "transform_cls, config",
    TRANSFORM_REGISTRY.items(),
)
def test_forward_inverse_roundtrip(transform_cls, config):
    transform = transform_cls(**config["kwargs"])
    values = config["theta_values"]

    for theta in values:
        z = transform.forward(theta)
        recovered = transform.inverse(z)
        assert np.allclose(recovered, theta, atol=1e-8)


# --- Serialization Round-Trip ----------------------------------------------- #
@pytest.mark.parametrize(
    "transform_cls, config",
    TRANSFORM_REGISTRY.items(),
)
def test_transform_serialization_roundtrip(transform_cls, config):
    transform = transform_cls(**config["kwargs"])

    # Serialize
    transform_dict = transform.to_dict()

    # Reconstruct
    recovered = ParameterTransform.from_dict(transform_dict)

    assert isinstance(recovered, transform_cls)
    assert recovered._parameters == transform._parameters

    theta_values = config["theta_values"]

    for theta in theta_values:
        z1 = transform.forward(theta)
        z2 = recovered.forward(theta)
        assert np.allclose(z1, z2)

        inv1 = transform.inverse(z1)
        inv2 = recovered.inverse(z2)
        assert np.allclose(inv1, inv2)
