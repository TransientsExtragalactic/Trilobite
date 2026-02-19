import json

import numpy as np
import pytest

from triceratops.data.core import InferenceData, Observable

# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def fake_inference_data():
    """
    Construct a fully populated InferenceData object for testing
    serialization and round-trip integrity.
    """
    n = 12

    time = np.linspace(0.0, 10.0, n)
    frequency = np.linspace(1.0, 5.0, n)

    flux = np.random.normal(0.0, 1.0, n)
    flux_err = np.full(n, 0.1)

    # Include NaNs to test censoring preservation
    flux_upper = np.full(n, np.nan)
    flux_lower = np.full(n, np.nan)

    data = InferenceData(
        x={
            "time": time,
            "frequency": frequency,
        },
        observables={
            "flux": Observable(
                value=flux,
                error=flux_err,
                upper=flux_upper,
                lower=flux_lower,
            )
        },
        x_error={
            "time": np.full(n, 0.01),
            "frequency": np.full(n, 0.02),
        },
        x_upper=None,
        x_lower=None,
    )

    return data


# ======================================================================
# Tests
# ======================================================================


def test_inference_data_to_from_dict_roundtrip(tmp_path, fake_inference_data):
    """
    Full round-trip test:

    InferenceData → to_dict → JSON file → from_dict → equality
    """

    original = fake_inference_data

    # Serialize to dictionary
    as_dict = original.to_dict(test_meta="pytest")

    # Write to JSON
    file_path = tmp_path / "inference_data.json"
    with open(file_path, "w") as f:
        json.dump(as_dict, f)

    # Read back
    with open(file_path, "r") as f:
        loaded_dict = json.load(f)

    reconstructed = InferenceData.from_dict(loaded_dict)

    # Ensure equality
    assert reconstructed == original


def test_inference_data_json_roundtrip(tmp_path, fake_inference_data):
    """
    InferenceData → to_json → from_json → equality
    """
    original = fake_inference_data

    file_path = tmp_path / "data.json"

    original.to_json(file_path)
    reconstructed = InferenceData.from_json(file_path)

    assert reconstructed == original


def test_inference_data_hdf5_roundtrip(tmp_path, fake_inference_data):
    """
    InferenceData → to_hdf5 → from_hdf5 → equality
    """
    original = fake_inference_data

    file_path = tmp_path / "data.h5"

    original.to_hdf5(file_path)
    reconstructed = InferenceData.from_hdf5(file_path)

    assert reconstructed == original


@pytest.mark.parametrize("extension", [".json", ".h5"])
def test_inference_data_generic_file_roundtrip(tmp_path, fake_inference_data, extension):
    """
    InferenceData → to_file → from_file → equality
    """
    original = fake_inference_data

    file_path = tmp_path / f"data{extension}"

    original.to_file(file_path)
    reconstructed = InferenceData.from_file(file_path)

    assert reconstructed == original
