"""The optical data containers feed real :class:`OpticalModel` / :class:`OpticalEpochModel` instances."""

import numpy as np
import pytest
from astropy import units as u
from astropy.table import Table

from trilobite.data.light_curve import OpticalLightCurveContainer
from trilobite.data.optical_photometry import OpticalPhotometryContainer, OpticalPhotometryEpoch
from trilobite.data.utils import band_variable_name
from trilobite.inference import GaussianCensoredLikelihood
from trilobite.models.SEDs.blackbody_optical import BlackbodyOpticalEpochModel, BlackbodyOpticalModel
from trilobite.models.supernovae import NickelCobaltArnettModel
from trilobite.utils.phot_utils import FilterBundle, PhotometryFilter


def _box_filter(lam_center_cm, width_cm, name):
    lam = np.linspace(lam_center_cm - width_cm / 2, lam_center_cm + width_cm / 2, 50)
    return PhotometryFilter(lam, np.ones(50), name=name)


@pytest.fixture
def bundle():
    return FilterBundle({"V": _box_filter(550e-7, 80e-7, "V"), "R": _box_filter(700e-7, 100e-7, "R")})


def _mag_table():
    return Table(
        {
            "time": [10.0, 10.0, 20.0, 20.0] * u.day,
            "band_name": ["V", "R", "V", "R"],
            "mag_ab": [20.0, 19.8, 20.3, 20.1],
            "mag_ab_error": [0.1, 0.1, 0.1, 0.1],
            "mag_ab_upper_limit": [np.nan] * 4,
        }
    )


class TestBandVariableName:
    def test_optical_models_use_band_index(self, bundle):
        assert band_variable_name(BlackbodyOpticalModel(bundle)) == "band_index"

    def test_legacy_models_use_band_idx(self):
        class Legacy:
            variable_names = ("time", "band_idx")

        assert band_variable_name(Legacy()) == "band_idx"

    def test_model_without_variable_names(self):
        assert band_variable_name(object()) == "band_idx"


class TestPhotometryContainer:
    def test_inference_data_for_a_time_dependent_model(self, bundle):
        model = BlackbodyOpticalModel(bundle)
        data = OpticalPhotometryContainer(_mag_table()).to_inference_data(model)
        assert set(data.x) == {"time", "band_index"}
        np.testing.assert_array_equal(data.x["band_index"], [0, 1, 0, 1])

    def test_inference_data_gives_a_finite_likelihood(self, bundle):
        model = NickelCobaltArnettModel(bundle, redshift=0.05)
        data = OpticalPhotometryContainer(_mag_table()).to_inference_data(model)
        likelihood = GaussianCensoredLikelihood(model=model, data=data)
        assert likelihood is not None
        flux = model.forward_model({"band_index": data.x["band_index"], "time": data.x["time"]}, {}).flux
        assert np.all(np.isfinite(flux.value))


class TestEpochContainer:
    def test_inference_data_for_an_epoch_model(self, bundle):
        table = Table(
            {
                "band_name": ["V", "R"],
                "mag_ab": [20.0, 19.8],
                "mag_ab_error": [0.1, 0.1],
                "mag_ab_upper_limit": [np.nan, np.nan],
            }
        )
        model = BlackbodyOpticalEpochModel(bundle)
        data = OpticalPhotometryEpoch(table).to_inference_data(model)
        assert set(data.x) == {"band_index"}
        np.testing.assert_array_equal(data.x["band_index"], [0, 1])


class TestLightCurveContainer:
    def test_inference_data_for_a_single_band(self, bundle):
        table = Table(
            {
                "time": [10.0, 20.0, 30.0] * u.day,
                "mag_ab": [20.0, 20.3, 20.6],
                "mag_ab_error": [0.1, 0.1, 0.1],
                "mag_ab_upper_limit": [np.nan] * 3,
            }
        )
        model = BlackbodyOpticalModel(bundle)
        data = OpticalLightCurveContainer(table, band="R").to_inference_data(model)
        assert set(data.x) == {"time", "band_index"}
        np.testing.assert_array_equal(data.x["band_index"], [1, 1, 1])
