"""
Tests for RadioPhotometryContainer.

Mirrors the structure of test_optical_photometry.py for radio data.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
from astropy import units as u
from astropy.table import Table

from triceratops.data import RadioPhotometryContainer


# ============================================================
# Fixtures
# ============================================================


def _radio_table(n=10, n_upper=3, with_epochs=False):
    """Build a minimal valid RadioPhotometryContainer table."""
    flux = np.concatenate([np.linspace(1, 5, n - n_upper), np.full(n_upper, np.nan)])
    err = np.concatenate([np.full(n - n_upper, 0.1), np.full(n_upper, np.nan)])
    upper = np.concatenate([np.full(n - n_upper, np.nan), np.linspace(0.5, 2.0, n_upper)])
    t = Table(
        {
            "time": np.linspace(0, 100, n) * u.day,
            "freq": np.full(n, 5.5) * u.GHz,
            "flux_density": flux * u.Jy,
            "flux_density_error": err * u.Jy,
            "flux_upper_limit": upper * u.Jy,
        }
    )
    if with_epochs:
        t["epoch_id"] = np.zeros(n, dtype=int)
    return t


def _mock_variable(name, base_units):
    v = MagicMock()
    v.name = name
    v.base_units = base_units
    return v


def _mock_model(var_names=("time", "freq"), output_names=("flux_density",)):
    """Build a mock model matching what InferenceData.from_arrays expects."""
    model = MagicMock()
    model.variable_names = list(var_names)
    model.output_names = list(output_names)

    _unit_map = {
        "time": u.s,
        "freq": u.Hz,
        "frequency": u.Hz,
    }
    model.VARIABLES = [_mock_variable(n, _unit_map.get(n, u.dimensionless_unscaled)) for n in var_names]

    outputs = MagicMock()
    outputs._fields = list(output_names)
    model.OUTPUTS = outputs

    units = MagicMock()
    units.flux_density = u.Jy
    model.UNITS = units

    return model


# ============================================================
# Construction
# ============================================================


class TestConstruction:
    def test_valid_table(self):
        c = RadioPhotometryContainer(_radio_table())
        assert c.n_obs == 10

    def test_missing_required_column_raises(self):
        t = _radio_table()
        del t["freq"]
        with pytest.raises(ValueError):
            RadioPhotometryContainer(t)

    def test_wrong_unit_raises(self):
        t = _radio_table()
        t["flux_density"].unit = u.m  # wrong unit
        with pytest.raises((u.UnitsError, ValueError)):
            RadioPhotometryContainer(t)

    def test_from_table_classmethod(self):
        c = RadioPhotometryContainer.from_table(_radio_table())
        assert c.n_obs == 10

    def test_from_table_column_map(self):
        t = _radio_table()
        t.rename_column("flux_density", "S_nu")
        c = RadioPhotometryContainer.from_table(t, column_map={"S_nu": "flux_density"})
        assert c.n_obs == 10


# ============================================================
# Properties
# ============================================================


class TestProperties:
    def test_time_is_quantity(self):
        c = RadioPhotometryContainer(_radio_table())
        assert isinstance(c.time, u.Quantity)
        assert c.time.unit.is_equivalent(u.day)

    def test_freq_is_quantity(self):
        c = RadioPhotometryContainer(_radio_table())
        assert isinstance(c.freq, u.Quantity)
        assert c.freq.unit.is_equivalent(u.GHz)

    def test_flux_density_is_quantity(self):
        c = RadioPhotometryContainer(_radio_table())
        assert isinstance(c.flux_density, u.Quantity)

    def test_flux_density_error_is_quantity(self):
        c = RadioPhotometryContainer(_radio_table())
        assert isinstance(c.flux_density_error, u.Quantity)

    def test_n_obs(self):
        c = RadioPhotometryContainer(_radio_table(n=15))
        assert c.n_obs == 15


# ============================================================
# Detection Logic
# ============================================================


class TestDetectionLogic:
    def test_detection_count(self):
        c = RadioPhotometryContainer(_radio_table(n=10, n_upper=3))
        assert c.n_detections == 7

    def test_non_detection_count(self):
        c = RadioPhotometryContainer(_radio_table(n=10, n_upper=3))
        assert c.n_non_detections == 3

    def test_detection_mask_length(self):
        c = RadioPhotometryContainer(_radio_table(n=10))
        assert len(c.detection_mask) == 10

    def test_detection_table_rows(self):
        c = RadioPhotometryContainer(_radio_table(n=10, n_upper=3))
        assert len(c.detection_table) == 7

    def test_non_detection_table_rows(self):
        c = RadioPhotometryContainer(_radio_table(n=10, n_upper=3))
        assert len(c.non_detection_table) == 3

    def test_all_detections(self):
        c = RadioPhotometryContainer(_radio_table(n=8, n_upper=0))
        assert c.n_non_detections == 0
        assert c.n_detections == 8


# ============================================================
# Epoch Support
# ============================================================


class TestEpochs:
    def test_set_epochs_from_indices_equal_groups(self):
        c = RadioPhotometryContainer(_radio_table(n=9))
        # Pass a per-observation epoch ID array
        epoch_ids = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        c.set_epochs_from_indices(epoch_ids)
        assert c.n_epochs == 3

    def test_set_epochs_from_time_gaps(self):
        c = RadioPhotometryContainer(_radio_table(n=10))
        c.set_epochs_from_time_gaps(max_gap=20 * u.day)
        assert c.n_epochs >= 1

    def test_epoch_ids_preserved_through_mask(self):
        c = RadioPhotometryContainer(_radio_table(n=10, with_epochs=True))
        sub = c.apply_mask(c.detection_mask)
        assert "epoch_id" in sub.table.colnames


# ============================================================
# to_inference_data
# ============================================================


class TestToInferenceData:
    def test_default_mapping_time_freq(self):
        c = RadioPhotometryContainer(_radio_table(n=8, n_upper=0))
        model = _mock_model(var_names=("time", "freq"))
        id_ = c.to_inference_data(model)
        assert "time" in id_.x
        assert "freq" in id_.x

    def test_default_mapping_time_only(self):
        c = RadioPhotometryContainer(_radio_table(n=8, n_upper=0))
        model = _mock_model(var_names=("time",))
        id_ = c.to_inference_data(model)
        assert "time" in id_.x

    def test_unknown_variable_raises(self):
        c = RadioPhotometryContainer(_radio_table(n=5, n_upper=0))
        model = _mock_model(var_names=("wavelength",))
        with pytest.raises(ValueError, match="Cannot infer mapping"):
            c.to_inference_data(model)

    def test_multi_observable_requires_explicit(self):
        c = RadioPhotometryContainer(_radio_table(n=5, n_upper=0))
        model = _mock_model(var_names=("time",), output_names=("flux", "luminosity"))
        with pytest.raises(ValueError, match="observables"):
            c.to_inference_data(model)

    def test_error_inferred_for_upper_limits(self):
        c = RadioPhotometryContainer(_radio_table(n=10, n_upper=4))
        model = _mock_model(var_names=("time",))
        id_ = c.to_inference_data(model, infer_errors=True)
        obs = id_.observables["flux_density"]
        assert obs.error is not None

    def test_mask_reduces_rows(self):
        c = RadioPhotometryContainer(_radio_table(n=10, n_upper=0))
        model = _mock_model(var_names=("time",))
        mask = np.ones(10, dtype=bool)
        mask[7:] = False
        id_ = c.to_inference_data(model, mask=mask)
        assert id_.size == 7

    def test_wrong_mask_length_raises(self):
        c = RadioPhotometryContainer(_radio_table(n=10, n_upper=0))
        model = _mock_model(var_names=("time",))
        with pytest.raises(ValueError, match="same length"):
            c.to_inference_data(model, mask=np.ones(5, dtype=bool))
