"""Tests for BlackbodyOpticalModel and BlackbodyOpticalEpochModel."""

from unittest.mock import patch

import numpy as np
import pytest
from astropy import units as u

from trilobite.models.SEDs.blackbody_optical import (
    BlackbodyOpticalEpochModel,
    BlackbodyOpticalModel,
)
from trilobite.utils.phot_utils import FilterBundle, PhotometryFilter


# ---------------------------------------------------------------------------
# Helpers: minimal two-filter bundle
# ---------------------------------------------------------------------------


def _make_box_filter(lam_center_cm: float, width_cm: float, name: str) -> PhotometryFilter:
    """Box filter centred on lam_center_cm with given full width."""
    lam = np.linspace(lam_center_cm - width_cm / 2, lam_center_cm + width_cm / 2, 50)
    T = np.ones(50)
    return PhotometryFilter(lam, T, name=name)


@pytest.fixture
def two_filter_bundle() -> FilterBundle:
    """Bundle with V (550 nm) and R (700 nm) box filters."""
    f_V = _make_box_filter(550e-7, 80e-7, "V")  # 550 nm ± 40 nm
    f_R = _make_box_filter(700e-7, 100e-7, "R")  # 700 nm ± 50 nm
    return FilterBundle({"V": f_V, "R": f_R})


@pytest.fixture
def bb_model(two_filter_bundle) -> BlackbodyOpticalModel:
    return BlackbodyOpticalModel(two_filter_bundle)


@pytest.fixture
def bb_epoch_model(two_filter_bundle) -> BlackbodyOpticalEpochModel:
    return BlackbodyOpticalEpochModel(two_filter_bundle)


_PARAMS = {"T": 1e4, "R": 1e14, "D": 3.086e26}


# ---------------------------------------------------------------------------
# OpticalModel (time-evolving) tests
# ---------------------------------------------------------------------------


class TestBlackbodyOpticalModel:
    def test_output_shape(self, bb_model):
        """forward_model returns (N,) flux for N (band_index, time) pairs."""
        variables = {"band_index": [0, 1, 0, 1], "time": [1e5, 1e5, 2e5, 2e5]}
        result = bb_model.forward_model(variables, _PARAMS)
        assert result.flux.shape == (4,)

    def test_output_units(self, bb_model):
        """Output has correct F_nu units."""
        variables = {"band_index": [0, 1], "time": [1e5, 1e5]}
        result = bb_model.forward_model(variables, _PARAMS)
        assert result.flux.unit.is_equivalent(u.erg / u.s / u.cm**2 / u.Hz)

    def test_positive_flux(self, bb_model):
        """All returned fluxes are positive."""
        variables = {"band_index": [0, 1, 0, 1], "time": [1e5, 1e5, 2e5, 2e5]}
        result = bb_model.forward_model(variables, _PARAMS)
        assert np.all(result.flux.value > 0)

    def test_time_deduplication(self, bb_model):
        """_compute_sed is called with unique times only, not all N pairs."""
        variables = {"band_index": [0, 1, 0, 1], "time": [1e5, 1e5, 2e5, 2e5]}
        raw_vars = bb_model.coerce_model_variables(variables)
        raw_params = bb_model.coerce_model_parameters(_PARAMS)

        with patch.object(bb_model, "_compute_sed", wraps=bb_model._compute_sed) as mock_sed:
            bb_model._forward_model(raw_vars, raw_params)
            # Called exactly once; the t_unique arg should have length 2, not 4
            args = mock_sed.call_args[0]
            t_unique = args[1]
            assert len(t_unique) == 2

    def test_static_blackbody_same_flux_both_epochs(self, bb_model):
        """Static blackbody: same band, same flux at both epochs."""
        variables = {"band_index": [0, 0], "time": [1e5, 2e5]}
        result = bb_model.forward_model(variables, _PARAMS)
        np.testing.assert_allclose(result.flux.value[0], result.flux.value[1], rtol=1e-10)

    def test_string_band_names(self, bb_model):
        """Band names can be supplied as strings."""
        variables = {"band_index": ["V", "R", "V"], "time": [1e5, 1e5, 2e5]}
        result = bb_model.forward_model(variables, _PARAMS)
        assert result.flux.shape == (3,)

    def test_string_and_int_give_same_result(self, bb_model):
        """String and integer band_index produce identical output."""
        vars_str = {"band_index": ["V", "R"], "time": [1e5, 1e5]}
        vars_int = {"band_index": [0, 1], "time": [1e5, 1e5]}
        r_str = bb_model.forward_model(vars_str, _PARAMS)
        r_int = bb_model.forward_model(vars_int, _PARAMS)
        np.testing.assert_allclose(r_str.flux.value, r_int.flux.value, rtol=1e-12)

    def test_astropy_time_units(self, bb_model):
        """time variable accepts astropy Quantity (days → seconds conversion)."""
        vars_days = {"band_index": [0, 1], "time": [1, 1] * u.day}
        vars_secs = {"band_index": [0, 1], "time": [86400.0, 86400.0]}
        r_days = bb_model.forward_model(vars_days, _PARAMS)
        r_secs = bb_model.forward_model(vars_secs, _PARAMS)
        np.testing.assert_allclose(r_days.flux.value, r_secs.flux.value, rtol=1e-10)

    def test_invalid_band_name_raises(self, bb_model):
        """Unknown band name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown band name"):
            bb_model.forward_model({"band_index": ["X"], "time": [1e5]}, _PARAMS)

    def test_out_of_range_index_raises(self, bb_model):
        """Band index out of range raises ValueError."""
        with pytest.raises(ValueError, match="band_index values must be in"):
            bb_model.forward_model({"band_index": [5], "time": [1e5]}, _PARAMS)

    def test_bundle_property(self, bb_model, two_filter_bundle):
        """bundle property returns the attached FilterBundle."""
        assert bb_model.bundle is two_filter_bundle

    def test_add_filter_mutates_bundle(self, bb_model):
        """add_filter increases the number of filters."""
        n_before = len(bb_model.bundle.filter_names)
        f_B = _make_box_filter(440e-7, 80e-7, "B")
        bb_model.add_filter("B", f_B)
        assert len(bb_model.bundle.filter_names) == n_before + 1

    def test_serialisation_roundtrip(self, bb_model):
        """to_model_spec / from_model_spec reconstructs the model with identical filter set."""
        spec = bb_model.to_model_spec()
        restored = BlackbodyOpticalModel.from_model_spec(spec)
        assert isinstance(restored, BlackbodyOpticalModel)
        assert restored.bundle.filter_names == bb_model.bundle.filter_names
        np.testing.assert_allclose(restored.bundle.weight_matrix, bb_model.bundle.weight_matrix, rtol=1e-10)

    def test_serialisation_spec_is_json_serializable(self, bb_model):
        """The ModelSpec produced by to_model_spec can be round-tripped via JSON."""
        import json

        spec = bb_model.to_model_spec()
        json_str = spec.to_json()
        spec2 = spec.from_json(json_str)
        assert spec2.target == spec.target


# ---------------------------------------------------------------------------
# OpticalEpochModel tests
# ---------------------------------------------------------------------------


class TestBlackbodyOpticalEpochModel:
    def test_output_shape(self, bb_epoch_model):
        """forward_model returns (N,) for N band queries."""
        result = bb_epoch_model.forward_model({"band_index": [0, 1, 0]}, _PARAMS)
        assert result.flux.shape == (3,)

    def test_output_units(self, bb_epoch_model):
        result = bb_epoch_model.forward_model({"band_index": [0, 1]}, _PARAMS)
        assert result.flux.unit.is_equivalent(u.erg / u.s / u.cm**2 / u.Hz)

    def test_positive_flux(self, bb_epoch_model):
        result = bb_epoch_model.forward_model({"band_index": [0, 1]}, _PARAMS)
        assert np.all(result.flux.value > 0)

    def test_string_band_names(self, bb_epoch_model):
        result = bb_epoch_model.forward_model({"band_index": ["V", "R"]}, _PARAMS)
        assert result.flux.shape == (2,)

    def test_string_and_int_give_same_result(self, bb_epoch_model):
        r_str = bb_epoch_model.forward_model({"band_index": ["V", "R"]}, _PARAMS)
        r_int = bb_epoch_model.forward_model({"band_index": [0, 1]}, _PARAMS)
        np.testing.assert_allclose(r_str.flux.value, r_int.flux.value, rtol=1e-12)

    def test_hotter_is_brighter_optical(self, bb_epoch_model):
        """Higher T → higher flux in optical bands."""
        r_hot = bb_epoch_model.forward_model({"band_index": [0]}, {"T": 2e4, "R": 1e14, "D": 3.086e26})
        r_cool = bb_epoch_model.forward_model({"band_index": [0]}, {"T": 5e3, "R": 1e14, "D": 3.086e26})
        assert r_hot.flux.value[0] > r_cool.flux.value[0]
