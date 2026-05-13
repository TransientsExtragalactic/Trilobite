"""
Tests for RadioPhotometryEpoch (and its deprecated alias RadioPhotometryEpochContainer).
"""

import warnings
from unittest.mock import MagicMock

import numpy as np
import pytest
from astropy import units as u
from astropy.table import Table

from triceratops.data import RadioPhotometryEpoch


# ============================================================
# Fixtures
# ============================================================


def _epoch_table(n=6, n_upper=2):
    """Build a minimal valid RadioPhotometryEpoch table."""
    flux = np.concatenate([np.linspace(1, 5, n - n_upper), np.full(n_upper, np.nan)])
    err = np.concatenate([np.full(n - n_upper, 0.1), np.full(n_upper, np.nan)])
    upper = np.concatenate([np.full(n - n_upper, np.nan), np.linspace(0.5, 2.0, n_upper)])
    return Table(
        {
            "freq": np.linspace(1.4, 22, n) * u.GHz,
            "flux_density": flux * u.Jy,
            "flux_density_error": err * u.Jy,
            "flux_upper_limit": upper * u.Jy,
        }
    )


def _mock_variable(name, base_units):
    v = MagicMock()
    v.name = name
    v.base_units = base_units
    return v


def _mock_model(var_names=("freq",), output_names=("flux_density",)):
    """Build a mock model matching what InferenceData.from_table expects."""
    _unit_map = {
        "time": u.s,
        "freq": u.Hz,
        "frequency": u.Hz,
    }
    model = MagicMock()
    model.variable_names = list(var_names)
    model.output_names = list(output_names)
    model.VARIABLES = [_mock_variable(n, _unit_map.get(n, u.dimensionless_unscaled)) for n in var_names]

    outputs = MagicMock()
    outputs._fields = list(output_names)
    model.OUTPUTS = outputs

    units = MagicMock()
    units.flux_density = u.Jy
    model.UNITS = units

    return model


# ============================================================
# Rename / alias
# ============================================================


class TestRename:
    def test_class_is_named_correctly(self):
        assert RadioPhotometryEpoch.__name__ == "RadioPhotometryEpoch"

    def test_deprecated_alias_warns(self):
        from triceratops.data import RadioPhotometryEpochContainer

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            RadioPhotometryEpochContainer(_epoch_table())
            assert any(issubclass(x.category, DeprecationWarning) for x in w)

    def test_deprecated_alias_is_subclass(self):
        from triceratops.data import RadioPhotometryEpochContainer

        assert issubclass(RadioPhotometryEpochContainer, RadioPhotometryEpoch)


# ============================================================
# Construction
# ============================================================


class TestConstruction:
    def test_valid_table(self):
        c = RadioPhotometryEpoch(_epoch_table())
        assert c.n_obs == 6

    def test_missing_required_column_raises(self):
        t = _epoch_table()
        del t["freq"]
        with pytest.raises(ValueError):
            RadioPhotometryEpoch(t)

    def test_from_table_classmethod(self):
        c = RadioPhotometryEpoch.from_table(_epoch_table())
        assert c.n_obs == 6

    def test_from_table_column_map(self):
        t = _epoch_table()
        t.rename_column("flux_density", "S_nu")
        c = RadioPhotometryEpoch.from_table(t, column_map={"S_nu": "flux_density"})
        assert c.n_obs == 6


# ============================================================
# Properties
# ============================================================


class TestProperties:
    def test_freq_is_quantity(self):
        c = RadioPhotometryEpoch(_epoch_table())
        assert isinstance(c.freq, u.Quantity)
        assert c.freq.unit.is_equivalent(u.GHz)

    def test_flux_density_is_quantity(self):
        c = RadioPhotometryEpoch(_epoch_table())
        assert isinstance(c.flux_density, u.Quantity)

    def test_flux_density_error_is_quantity(self):
        c = RadioPhotometryEpoch(_epoch_table())
        assert isinstance(c.flux_density_error, u.Quantity)

    def test_n_obs(self):
        c = RadioPhotometryEpoch(_epoch_table(n=8))
        assert c.n_obs == 8


# ============================================================
# Detection Logic
# ============================================================


class TestDetectionLogic:
    def test_detection_count(self):
        c = RadioPhotometryEpoch(_epoch_table(n=6, n_upper=2))
        assert c.n_detections == 4

    def test_non_detection_count(self):
        c = RadioPhotometryEpoch(_epoch_table(n=6, n_upper=2))
        assert c.n_non_detections == 2

    def test_detection_mask(self):
        c = RadioPhotometryEpoch(_epoch_table(n=6, n_upper=2))
        assert c.detection_mask.sum() == 4

    def test_detection_table_rows(self):
        c = RadioPhotometryEpoch(_epoch_table(n=6, n_upper=2))
        assert len(c.detection_table) == 4

    def test_non_detection_table_rows(self):
        c = RadioPhotometryEpoch(_epoch_table(n=6, n_upper=2))
        assert len(c.non_detection_table) == 2


# ============================================================
# to_inference_data
# ============================================================


class TestToInferenceData:
    def test_default_freq_mapping(self):
        c = RadioPhotometryEpoch(_epoch_table(n_upper=0))
        model = _mock_model(var_names=("freq",))
        id_ = c.to_inference_data(model)
        assert "freq" in id_.x

    def test_frequency_alias_maps_to_freq_col(self):
        c = RadioPhotometryEpoch(_epoch_table(n_upper=0))
        model = _mock_model(var_names=("frequency",))
        id_ = c.to_inference_data(model)
        assert "frequency" in id_.x

    def test_unknown_variable_raises(self):
        c = RadioPhotometryEpoch(_epoch_table(n=4, n_upper=0))
        model = _mock_model(var_names=("time",))
        with pytest.raises(ValueError, match="Cannot infer mapping"):
            c.to_inference_data(model)

    def test_error_inferred_for_upper_limits(self):
        c = RadioPhotometryEpoch(_epoch_table(n=6, n_upper=3))
        model = _mock_model()
        id_ = c.to_inference_data(model, infer_errors=True)
        obs = id_.observables["flux_density"]
        assert obs.error is not None

    def test_mask_reduces_rows(self):
        c = RadioPhotometryEpoch(_epoch_table(n=6, n_upper=0))
        model = _mock_model()
        mask = np.array([True, True, True, False, False, False])
        id_ = c.to_inference_data(model, mask=mask)
        assert id_.size == 3

    def test_wrong_mask_raises(self):
        c = RadioPhotometryEpoch(_epoch_table(n=6, n_upper=0))
        model = _mock_model()
        with pytest.raises(ValueError, match="same length"):
            c.to_inference_data(model, mask=np.ones(3, dtype=bool))

    def test_observable_in_result(self):
        c = RadioPhotometryEpoch(_epoch_table(n=4, n_upper=0))
        model = _mock_model()
        id_ = c.to_inference_data(model)
        assert "flux_density" in id_.observables
