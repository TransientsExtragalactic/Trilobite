"""
Tests for triceratops.data.optical_photometry.OpticalPhotometryContainer.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
from astropy import units as u
from astropy.table import Table

from triceratops.data.optical_photometry import OpticalPhotometryContainer
from triceratops.utils.phot_utils import ab_mag_to_flux, flux_to_ab_mag

# ======================================================================
# Constants
# ======================================================================

_AB_ZERO = 3.631e-20  # erg/s/cm²/Hz

# ======================================================================
# Helpers
# ======================================================================


def _flux_table(n=5, with_upper_limits=True):
    """Build a minimal valid table using flux columns."""
    flux = np.array([1e-27, 2e-27, 3e-27, 4e-27, 5e-27]) * u.Unit("erg/(s cm2 Hz)")
    err = flux * 0.1
    ul = np.full(n, np.nan) * u.Unit("erg/(s cm2 Hz)")
    if with_upper_limits:
        ul[-1] = 1e-26 * u.Unit("erg/(s cm2 Hz)")
        flux[-1] = np.nan * u.Unit("erg/(s cm2 Hz)")
        err[-1] = np.nan * u.Unit("erg/(s cm2 Hz)")

    return Table(
        {
            "time": np.arange(1, n + 1, dtype=float) * u.day,
            "band_name": ["g", "r", "i", "g", "r"],
            "flux_density": flux,
            "flux_density_error": err,
            "flux_upper_limit": ul,
        }
    )


def _mag_table(n=5, with_upper_limits=True):
    """Build a minimal valid table using magnitude columns."""
    mag = np.array([20.0, 21.0, 22.0, 20.5, np.nan])
    mag_err = np.array([0.05, 0.06, 0.07, 0.05, np.nan])
    mag_ul = np.array([np.nan, np.nan, np.nan, np.nan, 23.0])

    return Table(
        {
            "time": np.arange(1, n + 1, dtype=float) * u.day,
            "band_name": ["g", "r", "i", "g", "r"],
            "mag_ab": mag,
            "mag_ab_error": mag_err,
            "mag_ab_upper_limit": mag_ul,
        }
    )


def _mock_variable(name, base_units):
    v = MagicMock()
    v.name = name
    v.base_units = base_units
    return v


def _mock_model(filter_names=("g", "r", "i")):
    """Return a mock model with a FilterBundle-like bundle attribute.

    The model interface mirrors what InferenceData.from_arrays expects:
    - model.variable_names → list of str
    - model.VARIABLES → list of objects with .name and .base_units
    - model.OUTPUTS._fields → list of observable name strings
    - model.UNITS.<observable_name> → unit string
    - model.output_names → list of str
    """
    bundle = MagicMock()
    bundle.filter_names = list(filter_names)

    model = MagicMock()
    model.bundle = bundle
    model.variable_names = ["time", "band_idx"]
    model.output_names = ["flux_density"]

    # VARIABLES with proper .name and .base_units
    model.VARIABLES = [
        _mock_variable("time", u.s),
        _mock_variable("band_idx", u.dimensionless_unscaled),
    ]

    # OUTPUTS named tuple fields
    outputs = MagicMock()
    outputs._fields = ["flux_density"]
    model.OUTPUTS = outputs

    # UNITS
    units = MagicMock()
    units.flux_density = u.Unit("erg/(s cm2 Hz)")
    model.UNITS = units

    return model


# ======================================================================
# Construction
# ======================================================================


class TestConstruction:
    def test_construction_from_flux_table(self):
        c = OpticalPhotometryContainer.from_table(_flux_table())
        assert c.n_obs == 5

    def test_construction_from_mag_table(self):
        c = OpticalPhotometryContainer.from_table(_mag_table())
        assert c.n_obs == 5

    def test_construction_from_mixed_table(self):
        t = _flux_table()
        t["mag_ab"] = flux_to_ab_mag(t["flux_density"].data)
        c = OpticalPhotometryContainer.from_table(t)
        assert c.n_obs == 5

    def test_construction_missing_y_raises(self):
        t = Table(
            {
                "time": [1.0, 2.0] * u.day,
                "band_name": ["g", "r"],
            }
        )
        with pytest.raises(ValueError, match="flux_density.*mag_ab"):
            OpticalPhotometryContainer.from_table(t)

    def test_construction_missing_band_raises(self):
        t = Table(
            {
                "time": [1.0, 2.0] * u.day,
                "flux_density": [1e-27, 2e-27] * u.Unit("erg/(s cm2 Hz)"),
                "flux_density_error": [1e-28, 2e-28] * u.Unit("erg/(s cm2 Hz)"),
                "flux_upper_limit": [np.nan, np.nan] * u.Unit("erg/(s cm2 Hz)"),
            }
        )
        with pytest.raises(ValueError, match="band_name"):
            OpticalPhotometryContainer.from_table(t)

    def test_construction_incompatible_units_raises(self):
        t = _flux_table()
        t["flux_density"].unit = u.m  # metres are not spectral flux density
        with pytest.raises(u.UnitsError):
            OpticalPhotometryContainer.from_table(t)


# ======================================================================
# Properties — flux representation
# ======================================================================


class TestFluxProperties:
    def test_flux_from_flux_column(self):
        c = OpticalPhotometryContainer.from_table(_flux_table(with_upper_limits=False))
        assert c.flux.unit.is_equivalent(u.Unit("erg/(s cm2 Hz)"))
        np.testing.assert_allclose(c.flux.value[:3], [1e-27, 2e-27, 3e-27])

    def test_flux_from_mag_column(self):
        mags = np.array([0.0])
        t = Table(
            {
                "time": [1.0] * u.day,
                "band_name": ["g"],
                "mag_ab": mags,
                "mag_ab_error": [0.05],
                "mag_ab_upper_limit": [np.nan],
            }
        )
        c = OpticalPhotometryContainer.from_table(t)
        expected = ab_mag_to_flux(0.0)  # 3.631e-20
        np.testing.assert_allclose(c.flux.value[0], expected, rtol=1e-6)

    def test_flux_error_from_flux_column(self):
        c = OpticalPhotometryContainer.from_table(_flux_table(with_upper_limits=False))
        np.testing.assert_allclose(c.flux_error.value[:3], c.flux.value[:3] * 0.1, rtol=1e-6)

    def test_flux_error_propagation_from_mag_error(self):
        mag = 20.0
        mag_err = 0.1
        t = Table(
            {
                "time": [1.0] * u.day,
                "band_name": ["g"],
                "mag_ab": [mag],
                "mag_ab_error": [mag_err],
                "mag_ab_upper_limit": [np.nan],
            }
        )
        c = OpticalPhotometryContainer.from_table(t)
        expected_err = c.flux.value[0] * (mag_err / 1.0857)
        np.testing.assert_allclose(c.flux_error.value[0], expected_err, rtol=1e-6)

    def test_flux_upper_limit_from_flux_column(self):
        c = OpticalPhotometryContainer.from_table(_flux_table())
        assert np.isnan(c.flux_upper_limit.value[0])  # detection
        assert np.isfinite(c.flux_upper_limit.value[-1])  # upper limit

    def test_flux_upper_limit_from_mag_column(self):
        c = OpticalPhotometryContainer.from_table(_mag_table())
        # last row is non-detection with mag_ab_upper_limit = 23.0
        expected = ab_mag_to_flux(23.0)
        np.testing.assert_allclose(c.flux_upper_limit.value[-1], expected, rtol=1e-6)
        assert np.isnan(c.flux_upper_limit.value[0])  # detection → NaN


# ======================================================================
# Properties — magnitude representation
# ======================================================================


class TestMagnitudeProperties:
    def test_mag_from_mag_column(self):
        c = OpticalPhotometryContainer.from_table(_mag_table())
        np.testing.assert_allclose(c.mag[:4], [20.0, 21.0, 22.0, 20.5])

    def test_mag_from_flux_column(self):
        t = Table(
            {
                "time": [1.0] * u.day,
                "band_name": ["g"],
                "flux_density": [_AB_ZERO] * u.Unit("erg/(s cm2 Hz)"),
                "flux_density_error": [_AB_ZERO * 0.05] * u.Unit("erg/(s cm2 Hz)"),
                "flux_upper_limit": [np.nan] * u.Unit("erg/(s cm2 Hz)"),
            }
        )
        c = OpticalPhotometryContainer.from_table(t)
        np.testing.assert_allclose(c.mag[0], 0.0, atol=1e-6)

    def test_mag_error_from_mag_column(self):
        c = OpticalPhotometryContainer.from_table(_mag_table())
        np.testing.assert_allclose(c.mag_error[:4], [0.05, 0.06, 0.07, 0.05])

    def test_mag_error_propagation_from_flux_error(self):
        flux = _AB_ZERO
        flux_err = _AB_ZERO * 0.1
        t = Table(
            {
                "time": [1.0] * u.day,
                "band_name": ["g"],
                "flux_density": [flux] * u.Unit("erg/(s cm2 Hz)"),
                "flux_density_error": [flux_err] * u.Unit("erg/(s cm2 Hz)"),
                "flux_upper_limit": [np.nan] * u.Unit("erg/(s cm2 Hz)"),
            }
        )
        c = OpticalPhotometryContainer.from_table(t)
        expected = 1.0857 * flux_err / flux
        np.testing.assert_allclose(c.mag_error[0], expected, rtol=1e-6)

    def test_mag_upper_limit_from_mag_column(self):
        c = OpticalPhotometryContainer.from_table(_mag_table())
        assert np.isnan(c.mag_upper_limit[0])  # detection
        np.testing.assert_allclose(c.mag_upper_limit[-1], 23.0)

    def test_mag_upper_limit_from_flux_column(self):
        t = _flux_table()
        c = OpticalPhotometryContainer.from_table(t)
        expected = flux_to_ab_mag(1e-26)  # the upper limit value
        np.testing.assert_allclose(c.mag_upper_limit[-1], expected, rtol=1e-6)


# ======================================================================
# Detection logic
# ======================================================================


class TestDetectionLogic:
    def test_is_detection_mask_flux_based(self):
        c = OpticalPhotometryContainer.from_table(_flux_table())
        assert c.detection_mask[:-1].all()
        assert not c.detection_mask[-1]

    def test_is_detection_mask_mag_based(self):
        c = OpticalPhotometryContainer.from_table(_mag_table())
        assert c.detection_mask[:-1].all()
        assert not c.detection_mask[-1]

    def test_n_detections(self):
        c = OpticalPhotometryContainer.from_table(_flux_table())
        assert c.n_detections == 4

    def test_n_non_detections(self):
        c = OpticalPhotometryContainer.from_table(_flux_table())
        assert c.n_non_detections == 1

    def test_detections_property(self):
        c = OpticalPhotometryContainer.from_table(_flux_table())
        det = c.detections
        assert det.n_obs == 4
        assert det.detection_mask.all()

    def test_upper_limits_property(self):
        c = OpticalPhotometryContainer.from_table(_mag_table())
        ul = c.upper_limits
        assert ul.n_obs == 1
        assert not ul.detection_mask.any()


# ======================================================================
# from_table — time handling
# ======================================================================


class TestFromTable:
    def test_from_table_time_start_subtraction_float(self):
        t = _flux_table(with_upper_limits=False)
        c = OpticalPhotometryContainer.from_table(t, time_start=1.0)
        np.testing.assert_allclose(c.time.value[0], 0.0, atol=1e-10)

    def test_from_table_time_start_subtraction_quantity(self):
        t = _flux_table(with_upper_limits=False)
        c = OpticalPhotometryContainer.from_table(t, time_start=1.0 * u.day)
        np.testing.assert_allclose(c.time.value[0], 0.0, atol=1e-10)

    def test_from_table_column_map(self):
        t = _flux_table(with_upper_limits=False)
        t.rename_column("band_name", "filter_name")
        c = OpticalPhotometryContainer.from_table(t, column_map={"filter_name": "band_name"})
        assert c.n_obs == 5

    def test_from_table_roundtrip_via_to_file(self, tmp_path):
        t = _flux_table(with_upper_limits=False)
        c = OpticalPhotometryContainer.from_table(t)
        path = tmp_path / "opt_phot.fits"
        c.to_file(str(path))
        c2 = OpticalPhotometryContainer.from_file(str(path))
        np.testing.assert_allclose(c.time.value, c2.time.value)
        np.testing.assert_array_equal(c.band_name, c2.band_name)


# ======================================================================
# Epoch support
# ======================================================================


class TestEpochs:
    def test_set_epochs_from_indices(self):
        c = OpticalPhotometryContainer.from_table(_flux_table(with_upper_limits=False))
        c.set_epochs_from_indices([0, 0, 1, 1, 2])
        assert c.has_epochs
        assert c.n_epochs == 3

    def test_set_epochs_from_time_gaps(self):
        c = OpticalPhotometryContainer.from_table(_flux_table(with_upper_limits=False))
        # times are 1,2,3,4,5 days — gap of 1.5 day splits into 5 epochs
        c.set_epochs_from_time_gaps(0.5 * u.day)
        assert c.n_epochs == 5

    def test_epochs_propagated_through_apply_mask(self):
        c = OpticalPhotometryContainer.from_table(_flux_table(with_upper_limits=False))
        c.set_epochs_from_indices([0, 0, 1, 1, 2])
        sub = c.apply_mask(np.array([True, True, False, False, True]))
        assert sub.has_epochs


# ======================================================================
# to_inference_data
# ======================================================================


class TestToInferenceData:
    def _make_container_and_model(self, use_mag=False):
        t = _mag_table(with_upper_limits=False) if use_mag else _flux_table(with_upper_limits=False)
        c = OpticalPhotometryContainer.from_table(t)
        model = _mock_model(filter_names=("g", "r", "i"))
        return c, model

    def test_band_name_mapped_to_idx_via_model_bundle(self):
        c, model = self._make_container_and_model()
        idata = c.to_inference_data(model)
        expected = np.array([0, 1, 2, 0, 1])  # g→0, r→1, i→2, g→0, r→1
        np.testing.assert_array_equal(idata.x["band_idx"], expected)

    def test_unknown_band_name_raises(self):
        c, model = self._make_container_and_model()
        model.bundle.filter_names = ["x", "y"]  # "g", "r", "i" not in this list
        with pytest.raises(KeyError, match="not in the model"):
            c.to_inference_data(model)

    def test_model_without_bundle_raises(self):
        c, _ = self._make_container_and_model()
        bad_model = MagicMock(spec=[])  # no bundle attribute
        with pytest.raises(AttributeError, match="bundle"):
            c.to_inference_data(bad_model)

    def test_produces_flux_from_mag_input(self):
        c, model = self._make_container_and_model(use_mag=True)
        idata = c.to_inference_data(model)
        obs = idata.observables["flux_density"]
        expected = ab_mag_to_flux(np.array([20.0, 21.0, 22.0, 20.5]))
        np.testing.assert_allclose(obs.value[:4], expected, rtol=1e-5)
        assert np.isnan(obs.value[-1])  # non-detection: mag_ab was NaN

    def test_produces_flux_from_flux_input(self):
        c, model = self._make_container_and_model(use_mag=False)
        idata = c.to_inference_data(model)
        obs = idata.observables["flux_density"]
        np.testing.assert_allclose(obs.value[:3], [1e-27, 2e-27, 3e-27], rtol=1e-6)

    def test_mag_to_flux_error_propagation(self):
        c, model = self._make_container_and_model(use_mag=True)
        idata = c.to_inference_data(model)
        obs = idata.observables["flux_density"]
        # σ_F = F * σ_m / 1.0857 for detections (rows 0-3)
        mag_errs = np.array([0.05, 0.06, 0.07, 0.05])
        expected_err = obs.value[:4] * (mag_errs / 1.0857)
        np.testing.assert_allclose(obs.error[:4], expected_err, rtol=1e-5)
        # Row 4 is a non-detection — error is inferred from upper limit / 3
        np.testing.assert_allclose(obs.error[-1], obs.upper[-1] / 3.0, rtol=1e-5)

    def test_x_keys_match_model_variables(self):
        c, model = self._make_container_and_model()
        idata = c.to_inference_data(model)
        assert "time" in idata.x
        assert "band_idx" in idata.x

    def test_upper_limits_propagated(self):
        t = _flux_table(with_upper_limits=True)
        c = OpticalPhotometryContainer.from_table(t)
        model = _mock_model()
        idata = c.to_inference_data(model, infer_errors=True)
        obs = idata.observables["flux_density"]
        # Last row is non-detection: upper is finite, value is NaN
        assert np.isfinite(obs.upper[-1])
        assert np.isnan(obs.value[-1])

    def test_mask_parameter(self):
        c, model = self._make_container_and_model()
        mask = np.array([True, True, False, False, False])
        idata = c.to_inference_data(model, mask=mask)
        assert idata.x["time"].shape == (2,)

    def test_infer_errors_fills_nan_errors_for_non_detections(self):
        t = _mag_table(with_upper_limits=True)
        c = OpticalPhotometryContainer.from_table(t)
        model = _mock_model()
        idata = c.to_inference_data(model, infer_errors=True, detection_threshold=3.0)
        obs = idata.observables["flux_density"]
        # Last row is non-detection: inferred error = upper_limit / 3
        expected_err = obs.upper[-1] / 3.0
        np.testing.assert_allclose(obs.error[-1], expected_err, rtol=1e-5)


# ======================================================================
# OpticalPhotometryEpoch
# ======================================================================

from triceratops.data.optical_photometry import OpticalPhotometryEpoch


def _epoch_flux_table(n=4, n_upper=1):
    """Minimal valid table for OpticalPhotometryEpoch using flux columns."""
    bands = ["g", "r", "i", "z"][:n]
    flux = np.array([2e-28, 3e-28, 4e-28, 1e-28][:n]) * u.Unit("erg/(s cm2 Hz)")
    err = flux * 0.1
    ul = np.full(n, np.nan) * u.Unit("erg/(s cm2 Hz)")
    for i in range(n_upper):
        flux[-(i + 1)] = np.nan * u.Unit("erg/(s cm2 Hz)")
        err[-(i + 1)] = np.nan * u.Unit("erg/(s cm2 Hz)")
        ul[-(i + 1)] = 5e-29 * u.Unit("erg/(s cm2 Hz)")
    return Table(
        {
            "band_name": bands,
            "flux_density": flux,
            "flux_density_error": err,
            "flux_upper_limit": ul,
        }
    )


def _epoch_mag_table(n=3):
    """Minimal valid table for OpticalPhotometryEpoch using magnitude columns."""
    return Table(
        {
            "band_name": ["g", "r", "i"][:n],
            "mag_ab": [22.1, 21.5, 21.0][:n],
            "mag_ab_error": [0.05, 0.04, 0.05][:n],
            "mag_ab_upper_limit": [float("nan")] * n,
        }
    )


def _epoch_mock_model():
    model = MagicMock()
    model.variable_names = ["band_idx"]
    model.output_names = ["flux_density"]

    var = MagicMock()
    var.name = "band_idx"
    var.base_units = u.dimensionless_unscaled
    model.VARIABLES = [var]

    outputs = MagicMock()
    outputs._fields = ["flux_density"]
    model.OUTPUTS = outputs

    units = MagicMock()
    units.flux_density = u.Unit("erg/(s cm2 Hz)")
    model.UNITS = units

    bundle = MagicMock()
    bundle.filter_names = ["g", "r", "i", "z"]
    model.bundle = bundle

    return model


class TestOpticalPhotometryEpochConstruction:
    def test_valid_flux_table(self):
        e = OpticalPhotometryEpoch(_epoch_flux_table())
        assert e.n_obs == 4

    def test_valid_mag_table(self):
        e = OpticalPhotometryEpoch(_epoch_mag_table())
        assert e.n_obs == 3

    def test_missing_band_name_raises(self):
        t = _epoch_flux_table()
        del t["band_name"]
        with pytest.raises(ValueError):
            OpticalPhotometryEpoch(t)

    def test_no_flux_or_mag_raises(self):
        t = Table({"band_name": ["g", "r"]})
        with pytest.raises(ValueError):
            OpticalPhotometryEpoch(t)

    def test_from_table_classmethod(self):
        e = OpticalPhotometryEpoch.from_table(_epoch_flux_table())
        assert e.n_obs == 4


class TestOpticalPhotometryEpochDetection:
    def test_detection_count(self):
        e = OpticalPhotometryEpoch(_epoch_flux_table(n=4, n_upper=1))
        assert e.n_detections == 3
        assert e.n_non_detections == 1

    def test_all_detections(self):
        e = OpticalPhotometryEpoch(_epoch_flux_table(n=4, n_upper=0))
        assert e.n_non_detections == 0
        assert e.n_detections == 4

    def test_detection_table_rows(self):
        e = OpticalPhotometryEpoch(_epoch_flux_table(n=4, n_upper=2))
        assert len(e.apply_mask(e.detection_mask)) == 2

    def test_apply_mask(self):
        e = OpticalPhotometryEpoch(_epoch_flux_table(n=4, n_upper=0))
        mask = np.array([True, False, True, False])
        sub = e.apply_mask(mask)
        assert sub.n_obs == 2
        assert isinstance(sub, OpticalPhotometryEpoch)


class TestOpticalPhotometryEpochToInferenceData:
    def test_band_idx_in_x(self):
        e = OpticalPhotometryEpoch(_epoch_flux_table(n=4, n_upper=0))
        model = _epoch_mock_model()
        idata = e.to_inference_data(model)
        assert "band_idx" in idata.x

    def test_band_idx_values(self):
        e = OpticalPhotometryEpoch(_epoch_flux_table(n=4, n_upper=0))
        model = _epoch_mock_model()
        idata = e.to_inference_data(model)
        # bands are g, r, i, z → indices 0, 1, 2, 3
        np.testing.assert_array_equal(idata.x["band_idx"], [0, 1, 2, 3])

    def test_flux_density_in_observables(self):
        e = OpticalPhotometryEpoch(_epoch_flux_table(n=4, n_upper=0))
        model = _epoch_mock_model()
        idata = e.to_inference_data(model)
        assert "flux_density" in idata.observables

    def test_upper_limits_propagated(self):
        e = OpticalPhotometryEpoch(_epoch_flux_table(n=4, n_upper=1))
        model = _epoch_mock_model()
        idata = e.to_inference_data(model, infer_errors=True)
        obs = idata.observables["flux_density"]
        assert np.isfinite(obs.upper[-1])
        assert np.isnan(obs.value[-1])

    def test_mag_table_converts_to_flux(self):
        e = OpticalPhotometryEpoch(_epoch_mag_table(n=3))
        model = _epoch_mock_model()
        idata = e.to_inference_data(model)
        obs = idata.observables["flux_density"]
        assert obs.value is not None
        assert np.all(np.isfinite(obs.value))

    def test_unknown_band_raises(self):
        t = _epoch_flux_table(n=4, n_upper=0)
        t["band_name"] = ["g", "r", "i", "UNKNOWN"]
        e = OpticalPhotometryEpoch(t)
        model = _epoch_mock_model()
        with pytest.raises(KeyError):
            e.to_inference_data(model)
