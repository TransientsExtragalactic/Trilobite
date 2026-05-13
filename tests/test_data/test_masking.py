"""Tests for DataContainer.apply_mask() and the associated copy() fix."""

import numpy as np
import pytest
from astropy import units as u
from astropy.table import Table

from triceratops.data.light_curve import RadioLightCurveContainer
from triceratops.data.light_curve import OpticalLightCurveContainer
from triceratops.data.photometry import RadioPhotometryContainer, RadioPhotometryEpoch, RadioPhotometryEpochContainer


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def photometry_table():
    """Minimal RadioPhotometryContainer table with mixed detections/non-detections."""
    n = 10
    rng = np.random.default_rng(0)
    flux = rng.uniform(0.1, 2.0, n)

    # First 6 are detections (NaN upper limit), last 4 are upper limits
    flux_ul = np.full(n, np.nan)
    flux_ul[6:] = rng.uniform(0.5, 1.0, 4)

    table = Table(
        {
            "flux_density": flux * u.Jy,
            "flux_density_error": (flux * 0.1) * u.Jy,
            "flux_upper_limit": flux_ul * u.Jy,
            "time": np.linspace(1.0, 100.0, n) * u.day,
            "freq": np.tile([1.4, 5.0, 8.5, 15.0, 22.0], 2) * u.GHz,
        }
    )
    return table


@pytest.fixture
def photometry_container(photometry_table):
    return RadioPhotometryContainer.from_table(photometry_table)


@pytest.fixture
def light_curve_table():
    n = 8
    rng = np.random.default_rng(1)
    flux = rng.uniform(0.1, 1.0, n)
    flux_ul = np.full(n, np.nan)
    flux_ul[-2:] = rng.uniform(0.5, 1.0, 2)

    return Table(
        {
            "time": np.linspace(10.0, 200.0, n) * u.day,
            "flux_density": flux * u.Jy,
            "flux_density_error": (flux * 0.1) * u.Jy,
            "flux_upper_limit": flux_ul * u.Jy,
        }
    )


@pytest.fixture
def light_curve_container(light_curve_table):
    return RadioLightCurveContainer.from_table(light_curve_table, frequency=5.0 * u.GHz)


@pytest.fixture
def epoch_table():
    n = 6
    rng = np.random.default_rng(2)
    flux = rng.uniform(0.1, 2.0, n)
    flux_ul = np.full(n, np.nan)
    flux_ul[-1] = 0.5

    return Table(
        {
            "flux_density": flux * u.Jy,
            "flux_density_error": (flux * 0.1) * u.Jy,
            "flux_upper_limit": flux_ul * u.Jy,
            "freq": np.linspace(1.0, 20.0, n) * u.GHz,
        }
    )


@pytest.fixture
def epoch_container(epoch_table):
    return RadioPhotometryEpochContainer.from_table(epoch_table)


# ======================================================================
# RadioPhotometryContainer
# ======================================================================


class TestRadioPhotometryContainerMask:
    def test_boolean_mask_length(self, photometry_container):
        mask = np.array([True, False] * 5)
        masked = photometry_container.apply_mask(mask)
        assert len(masked) == 5

    def test_integer_index_mask(self, photometry_container):
        indices = np.array([0, 2, 4])
        masked = photometry_container.apply_mask(indices)
        assert len(masked) == 3

    def test_returns_same_type(self, photometry_container):
        mask = np.ones(len(photometry_container), dtype=bool)
        masked = photometry_container.apply_mask(mask)
        assert isinstance(masked, RadioPhotometryContainer)

    def test_detection_mask_matches_detection_table(self, photometry_container):
        det_mask = photometry_container.detection_mask
        masked = photometry_container.apply_mask(det_mask)

        # All rows in masked container should be detections
        assert masked.n_non_detections == 0
        assert masked.n_detections == photometry_container.n_detections

        # Flux values should match
        np.testing.assert_array_almost_equal(
            masked.flux_density.value,
            photometry_container.detection_table["flux_density"].data,
        )

    def test_all_mask_returns_full_container(self, photometry_container):
        mask = np.ones(len(photometry_container), dtype=bool)
        masked = photometry_container.apply_mask(mask)
        assert len(masked) == len(photometry_container)

    def test_epoch_ids_preserved(self, photometry_container):
        photometry_container.set_epochs_from_indices([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        # Select only epoch 0
        mask = photometry_container.get_epoch_mask(0)
        masked = photometry_container.apply_mask(mask)
        assert len(masked) == 5
        assert masked.has_epochs
        assert masked.n_epochs == 1
        assert set(masked.epoch_ids) == {0}

    def test_copy_via_full_mask(self, photometry_container):
        mask = np.ones(len(photometry_container), dtype=bool)
        masked = photometry_container.apply_mask(mask)
        # Should be a distinct object
        assert masked is not photometry_container


# ======================================================================
# RadioLightCurveContainer
# ======================================================================


class TestRadioLightCurveContainerMask:
    def test_frequency_preserved(self, light_curve_container):
        mask = light_curve_container.detection_mask
        masked = light_curve_container.apply_mask(mask)
        assert masked.frequency == light_curve_container.frequency

    def test_returns_same_type(self, light_curve_container):
        mask = np.ones(len(light_curve_container), dtype=bool)
        masked = light_curve_container.apply_mask(mask)
        assert isinstance(masked, RadioLightCurveContainer)

    def test_boolean_mask_row_count(self, light_curve_container):
        mask = np.array([True, False] * (len(light_curve_container) // 2))
        masked = light_curve_container.apply_mask(mask)
        assert len(masked) == mask.sum()

    def test_detection_only_mask(self, light_curve_container):
        det_mask = light_curve_container.detection_mask
        masked = light_curve_container.apply_mask(det_mask)
        assert masked.n_non_detections == 0
        assert masked.n_detections == light_curve_container.n_detections

    def test_copy_preserves_frequency(self, light_curve_container):
        copied = light_curve_container.copy()
        assert isinstance(copied, RadioLightCurveContainer)
        assert copied.frequency == light_curve_container.frequency

    def test_copy_is_independent(self, light_curve_container):
        copied = light_curve_container.copy()
        assert copied is not light_curve_container


# ======================================================================
# RadioPhotometryEpoch (canonical name)
# ======================================================================


@pytest.fixture
def epoch_container_new(epoch_table):
    """Same data but constructed via the canonical RadioPhotometryEpoch."""
    return RadioPhotometryEpoch.from_table(epoch_table)


class TestRadioPhotometryEpochMask:
    def test_boolean_mask(self, epoch_container_new):
        mask = np.array([True, False, True, False, True, False])
        masked = epoch_container_new.apply_mask(mask)
        assert len(masked) == 3

    def test_returns_same_type(self, epoch_container_new):
        mask = np.ones(len(epoch_container_new), dtype=bool)
        masked = epoch_container_new.apply_mask(mask)
        assert isinstance(masked, RadioPhotometryEpoch)

    def test_detection_mask(self, epoch_container_new):
        det_mask = epoch_container_new.detection_mask
        masked = epoch_container_new.apply_mask(det_mask)
        assert masked.n_non_detections == 0


# ======================================================================
# RadioPhotometryEpochContainer (deprecated alias — kept for compat tests)
# ======================================================================


class TestRadioPhotometryEpochContainerMask:
    def test_boolean_mask(self, epoch_container):
        mask = np.array([True, False, True, False, True, False])
        masked = epoch_container.apply_mask(mask)
        assert len(masked) == 3

    def test_returns_same_type(self, epoch_container):
        mask = np.ones(len(epoch_container), dtype=bool)
        masked = epoch_container.apply_mask(mask)
        assert isinstance(masked, RadioPhotometryEpochContainer)

    def test_detection_mask(self, epoch_container):
        det_mask = epoch_container.detection_mask
        masked = epoch_container.apply_mask(det_mask)
        assert masked.n_non_detections == 0


# ======================================================================
# OpticalLightCurveContainer
# ======================================================================


@pytest.fixture
def optical_lc_table():
    n = 8
    rng = np.random.default_rng(3)
    flux = rng.uniform(1e-29, 5e-29, n)
    flux_ul = np.full(n, np.nan)
    flux_ul[-2:] = rng.uniform(1e-29, 2e-29, 2)

    return Table(
        {
            "time": np.linspace(5.0, 150.0, n) * u.day,
            "flux_density": flux * u.Unit("erg/(s cm2 Hz)"),
            "flux_density_error": (flux * 0.1) * u.Unit("erg/(s cm2 Hz)"),
            "flux_upper_limit": flux_ul * u.Unit("erg/(s cm2 Hz)"),
        }
    )


@pytest.fixture
def optical_lc_container(optical_lc_table):
    return OpticalLightCurveContainer.from_table(optical_lc_table, band="g")


class TestOpticalLightCurveContainerMask:
    def test_boolean_mask_length(self, optical_lc_container):
        mask = np.array([True, False] * 4)
        masked = optical_lc_container.apply_mask(mask)
        assert len(masked) == 4

    def test_returns_same_type(self, optical_lc_container):
        mask = np.ones(len(optical_lc_container), dtype=bool)
        masked = optical_lc_container.apply_mask(mask)
        assert isinstance(masked, OpticalLightCurveContainer)

    def test_band_preserved(self, optical_lc_container):
        mask = optical_lc_container.detection_mask
        masked = optical_lc_container.apply_mask(mask)
        assert masked.band == optical_lc_container.band

    def test_detection_only_mask(self, optical_lc_container):
        det_mask = optical_lc_container.detection_mask
        masked = optical_lc_container.apply_mask(det_mask)
        assert masked.n_non_detections == 0
        assert masked.n_detections == optical_lc_container.n_detections
