"""
Tests for RadioLightCurveContainer and OpticalLightCurveContainer.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
from astropy import units as u
from astropy.table import Table

from trilobite.data import OpticalLightCurveContainer, RadioLightCurveContainer


# ============================================================
# Shared helpers
# ============================================================


def _mock_variable(name, base_units):
    v = MagicMock()
    v.name = name
    v.base_units = base_units
    return v


def _mock_model(var_names=("time",), output_names=("flux_density",), filter_names=None):
    """Build a mock model matching what InferenceData.from_arrays expects."""
    _unit_map = {
        "time": u.s,
        "freq": u.Hz,
        "frequency": u.Hz,
        "band_idx": u.dimensionless_unscaled,
    }
    model = MagicMock()
    model.variable_names = list(var_names)
    model.output_names = list(output_names)
    model.VARIABLES = [_mock_variable(n, _unit_map.get(n, u.dimensionless_unscaled)) for n in var_names]

    outputs = MagicMock()
    outputs._fields = list(output_names)
    model.OUTPUTS = outputs

    units = MagicMock()
    units.flux_density = u.Unit("erg/(s cm2 Hz)")
    model.UNITS = units

    if filter_names is not None:
        model.bundle.filter_names = list(filter_names)

    return model


# ============================================================
# RadioLightCurveContainer
# ============================================================


def _radio_lc_table(n=8, n_upper=2):
    flux = np.concatenate([np.linspace(1, 5, n - n_upper), np.full(n_upper, np.nan)])
    err = np.concatenate([np.full(n - n_upper, 0.1), np.full(n_upper, np.nan)])
    upper = np.concatenate([np.full(n - n_upper, np.nan), np.linspace(0.3, 1.0, n_upper)])
    return Table(
        {
            "time": np.linspace(10, 500, n) * u.day,
            "flux_density": flux * u.Jy,
            "flux_density_error": err * u.Jy,
            "flux_upper_limit": upper * u.Jy,
        }
    )


class TestRadioLightCurveConstruction:
    def test_valid_table(self):
        c = RadioLightCurveContainer(_radio_lc_table(), frequency=5.5)
        assert c.n_obs == 8

    def test_frequency_stored_as_quantity(self):
        c = RadioLightCurveContainer(_radio_lc_table(), frequency=5.5)
        assert c.frequency.unit.is_equivalent(u.GHz)
        assert np.isclose(c.frequency.to_value(u.GHz), 5.5)

    def test_frequency_accepts_quantity(self):
        c = RadioLightCurveContainer(_radio_lc_table(), frequency=5500 * u.MHz)
        assert np.isclose(c.frequency.to_value(u.GHz), 5.5)

    def test_missing_required_column_raises(self):
        t = _radio_lc_table()
        del t["time"]
        with pytest.raises(ValueError):
            RadioLightCurveContainer(t, frequency=5.5)

    def test_from_table_classmethod(self):
        c = RadioLightCurveContainer.from_table(_radio_lc_table(), frequency=5.5)
        assert c.n_obs == 8

    def test_from_table_column_map(self):
        t = _radio_lc_table()
        t.rename_column("time", "mjd")
        c = RadioLightCurveContainer.from_table(t, frequency=5.5, column_map={"mjd": "time"})
        assert c.n_obs == 8


class TestRadioLightCurveProperties:
    def test_time_is_quantity(self):
        c = RadioLightCurveContainer(_radio_lc_table(), frequency=5.5)
        assert isinstance(c.time, u.Quantity)

    def test_flux_density_is_quantity(self):
        c = RadioLightCurveContainer(_radio_lc_table(), frequency=5.5)
        assert isinstance(c.flux_density, u.Quantity)

    def test_flux_density_error_is_quantity(self):
        c = RadioLightCurveContainer(_radio_lc_table(), frequency=5.5)
        assert isinstance(c.flux_density_error, u.Quantity)


class TestRadioLightCurveDetection:
    def test_detection_count(self):
        c = RadioLightCurveContainer(_radio_lc_table(n=8, n_upper=2), frequency=5.5)
        assert c.n_detections == 6

    def test_non_detection_count(self):
        c = RadioLightCurveContainer(_radio_lc_table(n=8, n_upper=2), frequency=5.5)
        assert c.n_non_detections == 2

    def test_detection_mask_length(self):
        c = RadioLightCurveContainer(_radio_lc_table(n=8), frequency=5.5)
        assert len(c.detection_mask) == 8


class TestRadioLightCurveToInferenceData:
    def test_default_mapping_time(self):
        c = RadioLightCurveContainer(_radio_lc_table(n_upper=0), frequency=5.5)
        model = _mock_model(var_names=("time",))
        id_ = c.to_inference_data(model)
        assert "time" in id_.x
        assert "flux_density" in id_.observables

    def test_unknown_variable_raises(self):
        c = RadioLightCurveContainer(_radio_lc_table(n_upper=0), frequency=5.5)
        model = _mock_model(var_names=("freq",))
        with pytest.raises(ValueError, match="Cannot infer mapping"):
            c.to_inference_data(model)

    def test_error_inferred_for_upper_limits(self):
        c = RadioLightCurveContainer(_radio_lc_table(n=8, n_upper=3), frequency=5.5)
        model = _mock_model(var_names=("time",))
        id_ = c.to_inference_data(model, infer_errors=True)
        obs = id_.observables["flux_density"]
        assert obs.error is not None

    def test_mask_reduces_rows(self):
        c = RadioLightCurveContainer(_radio_lc_table(n=8, n_upper=0), frequency=5.5)
        model = _mock_model(var_names=("time",))
        mask = np.array([True] * 5 + [False] * 3)
        id_ = c.to_inference_data(model, mask=mask)
        assert id_.size == 5

    def test_wrong_mask_raises(self):
        c = RadioLightCurveContainer(_radio_lc_table(n=8, n_upper=0), frequency=5.5)
        model = _mock_model(var_names=("time",))
        with pytest.raises(ValueError, match="same length"):
            c.to_inference_data(model, mask=np.ones(3, dtype=bool))

    def test_frequency_not_in_x(self):
        """Frequency is metadata, not a per-row variable."""
        c = RadioLightCurveContainer(_radio_lc_table(n=6, n_upper=0), frequency=5.5)
        model = _mock_model(var_names=("time",))
        id_ = c.to_inference_data(model)
        assert "freq" not in id_.x
        assert "frequency" not in id_.x


# ============================================================
# OpticalLightCurveContainer
# ============================================================


def _optical_lc_flux_table(n=8, n_upper=2):
    flux = np.concatenate([np.linspace(1e-27, 5e-27, n - n_upper), np.full(n_upper, np.nan)])
    err = np.concatenate([np.full(n - n_upper, 1e-28), np.full(n_upper, np.nan)])
    upper = np.concatenate([np.full(n - n_upper, np.nan), np.full(n_upper, 2e-27)])
    return Table(
        {
            "time": np.linspace(10, 500, n) * u.day,
            "flux_density": flux * u.Unit("erg/(s cm2 Hz)"),
            "flux_density_error": err * u.Unit("erg/(s cm2 Hz)"),
            "flux_upper_limit": upper * u.Unit("erg/(s cm2 Hz)"),
        }
    )


def _optical_lc_mag_table(n=8, n_upper=2):
    mag = np.concatenate([np.linspace(20, 22, n - n_upper), np.full(n_upper, np.nan)])
    err = np.concatenate([np.full(n - n_upper, 0.05), np.full(n_upper, np.nan)])
    upper = np.concatenate([np.full(n - n_upper, np.nan), np.full(n_upper, 22.5)])
    return Table(
        {
            "time": np.linspace(10, 500, n) * u.day,
            "mag_ab": mag,
            "mag_ab_error": err,
            "mag_ab_upper_limit": upper,
        }
    )


def _optical_mock_model(filter_names=("g", "r", "i")):
    model = _mock_model(
        var_names=("time", "band_idx"),
        output_names=("flux_density",),
        filter_names=filter_names,
    )
    return model


class TestOpticalLightCurveConstruction:
    def test_valid_flux_table(self):
        c = OpticalLightCurveContainer(_optical_lc_flux_table(), band="g")
        assert c.n_obs == 8

    def test_valid_mag_table(self):
        c = OpticalLightCurveContainer(_optical_lc_mag_table(), band="r")
        assert c.n_obs == 8

    def test_missing_both_columns_raises(self):
        t = Table({"time": [1, 2, 3] * u.day})
        with pytest.raises(ValueError, match="y-column group"):
            OpticalLightCurveContainer(t, band="g")

    def test_band_stored(self):
        c = OpticalLightCurveContainer(_optical_lc_flux_table(), band="g")
        assert c.band == "g"

    def test_from_table_classmethod(self):
        c = OpticalLightCurveContainer.from_table(_optical_lc_flux_table(), band="g")
        assert c.band == "g"

    def test_from_table_column_map(self):
        t = _optical_lc_flux_table()
        t.rename_column("time", "mjd")
        c = OpticalLightCurveContainer.from_table(t, band="g", column_map={"mjd": "time"})
        assert c.n_obs == 8


class TestOpticalLightCurveProperties:
    def test_time_is_quantity(self):
        c = OpticalLightCurveContainer(_optical_lc_flux_table(), band="g")
        assert isinstance(c.time, u.Quantity)

    def test_flux_from_flux_columns(self):
        c = OpticalLightCurveContainer(_optical_lc_flux_table(n_upper=0), band="g")
        assert isinstance(c.flux, u.Quantity)
        assert c.flux.unit.is_equivalent(u.Unit("erg/(s cm2 Hz)"))

    def test_flux_from_mag_columns(self):
        c = OpticalLightCurveContainer(_optical_lc_mag_table(n_upper=0), band="r")
        flux = c.flux
        assert isinstance(flux, u.Quantity)
        assert np.all(flux.value > 0)

    def test_mag_from_mag_columns(self):
        c = OpticalLightCurveContainer(_optical_lc_mag_table(n_upper=0), band="r")
        assert isinstance(c.mag, np.ndarray)

    def test_mag_from_flux_columns(self):
        c = OpticalLightCurveContainer(_optical_lc_flux_table(n_upper=0), band="g")
        mag = c.mag
        assert isinstance(mag, np.ndarray)
        assert np.all(np.isfinite(mag))

    def test_band_property(self):
        c = OpticalLightCurveContainer(_optical_lc_flux_table(), band="i")
        assert c.band == "i"


class TestOpticalLightCurveDetection:
    def test_detection_count_flux(self):
        c = OpticalLightCurveContainer(_optical_lc_flux_table(n=8, n_upper=2), band="g")
        assert c.n_detections == 6

    def test_non_detection_count_flux(self):
        c = OpticalLightCurveContainer(_optical_lc_flux_table(n=8, n_upper=2), band="g")
        assert c.n_non_detections == 2

    def test_detection_count_mag(self):
        c = OpticalLightCurveContainer(_optical_lc_mag_table(n=8, n_upper=3), band="r")
        assert c.n_detections == 5

    def test_non_detection_count_mag(self):
        c = OpticalLightCurveContainer(_optical_lc_mag_table(n=8, n_upper=3), band="r")
        assert c.n_non_detections == 3


class TestOpticalLightCurveToInferenceData:
    def test_band_resolved_to_index(self):
        c = OpticalLightCurveContainer(_optical_lc_flux_table(n_upper=0), band="g")
        model = _optical_mock_model(filter_names=("g", "r", "i"))
        id_ = c.to_inference_data(model)
        assert "time" in id_.x
        assert "band_idx" in id_.x
        assert np.all(id_.x["band_idx"] == 0)  # "g" is index 0

    def test_second_band_index(self):
        c = OpticalLightCurveContainer(_optical_lc_flux_table(n_upper=0), band="r")
        model = _optical_mock_model(filter_names=("g", "r", "i"))
        id_ = c.to_inference_data(model)
        assert np.all(id_.x["band_idx"] == 1)  # "r" is index 1

    def test_unknown_band_raises(self):
        c = OpticalLightCurveContainer(_optical_lc_flux_table(n_upper=0), band="z")
        model = _optical_mock_model(filter_names=("g", "r", "i"))
        with pytest.raises(KeyError, match="z"):
            c.to_inference_data(model)

    def test_model_without_bundle_raises(self):
        c = OpticalLightCurveContainer(_optical_lc_flux_table(n_upper=0), band="g")
        model = MagicMock(spec=[])  # no 'bundle' attribute
        model.variable_names = ["time", "band_idx"]
        model.output_names = ["flux_density"]
        with pytest.raises(AttributeError, match="bundle"):
            c.to_inference_data(model)

    def test_error_inferred_for_upper_limits(self):
        c = OpticalLightCurveContainer(_optical_lc_flux_table(n=8, n_upper=3), band="g")
        model = _optical_mock_model(filter_names=("g", "r"))
        id_ = c.to_inference_data(model, infer_errors=True)
        obs = id_.observables["flux_density"]
        assert obs.error is not None

    def test_mask_reduces_rows(self):
        c = OpticalLightCurveContainer(_optical_lc_flux_table(n=8, n_upper=0), band="g")
        model = _optical_mock_model(filter_names=("g", "r"))
        mask = np.array([True] * 5 + [False] * 3)
        id_ = c.to_inference_data(model, mask=mask)
        assert id_.size == 5

    def test_mag_converted_to_flux(self):
        """When only mag columns are present, output should be in flux units."""
        c = OpticalLightCurveContainer(_optical_lc_mag_table(n=5, n_upper=0), band="r")
        model = _optical_mock_model(filter_names=("g", "r"))
        id_ = c.to_inference_data(model)
        obs = id_.observables["flux_density"]
        assert np.all(np.isfinite(obs.value))
        assert np.all(obs.value > 0)
