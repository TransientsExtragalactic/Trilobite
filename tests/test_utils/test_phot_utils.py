"""
Tests for triceratops.utils.phot_utils.

Covers PhotometryFilter, FilterBundle, magnitude utilities, and loading helpers.
"""

import json
from pathlib import Path
from unittest import mock

import astropy.units as u
import numpy as np
import pytest

from triceratops.utils.phot_utils import (
    FilterBundle,
    PhotometryFilter,
    ab_mag_to_flux,
    filter_to_ab_mag,
    flux_lambda_to_st_mag,
    flux_to_ab_mag,
    load_filter_from_file,
    load_filter_from_speclite,
    load_filter_from_svo,
    load_filters_from_svo,
    list_svo_filters,
    st_mag_to_flux_lambda,
)


# ============================================================================== #
# Shared fixtures                                                                #
# ============================================================================== #


def _gaussian_filter(center_aa, width_aa, name="test", N=200, weighting="photon"):
    """Construct a Gaussian-shaped PhotometryFilter centred at center_aa Å."""
    lam = np.linspace(center_aa - 3 * width_aa, center_aa + 3 * width_aa, N) * u.AA
    T = np.exp(-0.5 * ((lam.value - center_aa) / width_aa) ** 2)
    return PhotometryFilter(lam, T, name=name, weighting=weighting)


@pytest.fixture
def r_filter():
    return _gaussian_filter(6231, 600, name="r")


@pytest.fixture
def g_filter():
    return _gaussian_filter(4770, 600, name="g")


@pytest.fixture
def i_filter():
    return _gaussian_filter(7625, 600, name="i")


@pytest.fixture
def bundle(g_filter, r_filter, i_filter):
    return FilterBundle({"g": g_filter, "r": r_filter, "i": i_filter})


# ============================================================================== #
# PhotometryFilter — construction                                                #
# ============================================================================== #


class TestPhotometryFilterConstruction:
    def test_construction_with_astropy_units(self):
        lam = np.linspace(5000, 7000, 100) * u.AA
        T = np.ones(100)
        filt = PhotometryFilter(lam, T, name="flat")
        assert len(filt) == 100
        assert filt.name == "flat"

    def test_construction_bare_array_assumed_cm(self):
        lam_cm = np.linspace(5e-5, 7e-5, 100)  # already in cm
        T = np.ones(100)
        filt = PhotometryFilter(lam_cm, T)
        assert filt.name == ""
        assert filt.wavelength[0] == pytest.approx(lam_cm[0])

    def test_shape_mismatch_raises(self):
        lam = np.linspace(5000, 7000, 100) * u.AA
        T = np.ones(50)
        with pytest.raises(ValueError, match="same shape"):
            PhotometryFilter(lam, T)

    def test_invalid_weighting_raises(self):
        lam = np.linspace(5000, 7000, 100) * u.AA
        T = np.ones(100)
        with pytest.raises(ValueError, match="weighting"):
            PhotometryFilter(lam, T, weighting="invalid")

    def test_photon_vs_energy_weights_differ(self):
        lam = np.linspace(5000, 7000, 200) * u.AA
        T = np.exp(-0.5 * ((lam.value - 6000) / 400) ** 2)
        filt_p = PhotometryFilter(lam, T, weighting="photon")
        filt_e = PhotometryFilter(lam, T, weighting="energy")
        # Weights must be normalised to 1
        assert np.sum(filt_p.weights) == pytest.approx(1.0, rel=1e-6)
        assert np.sum(filt_e.weights) == pytest.approx(1.0, rel=1e-6)
        # Photon and energy weights should differ
        assert not np.allclose(filt_p.weights, filt_e.weights)


# ============================================================================== #
# PhotometryFilter — properties                                                  #
# ============================================================================== #


class TestPhotometryFilterProperties:
    def test_len(self, r_filter):
        assert len(r_filter) == 200

    def test_name_property(self, r_filter):
        assert r_filter.name == "r"

    def test_repr_contains_name(self, r_filter):
        s = repr(r_filter)
        assert "'r'" in s
        assert "AA" in s

    def test_str_equals_repr(self, r_filter):
        assert str(r_filter) == repr(r_filter)

    def test_effective_wavelength_near_center(self):
        # Symmetric Gaussian: pivot wavelength should be close to center
        center_aa = 6231.0
        filt = _gaussian_filter(center_aa, 400, N=500)
        lam_eff_aa = filt.effective_wavelength * 1e8  # cm → Å
        assert lam_eff_aa == pytest.approx(center_aa, rel=1e-2)

    def test_effective_frequency_consistent(self, r_filter):
        c_cgs = 2.99792458e10
        assert r_filter.effective_frequency == pytest.approx(c_cgs / r_filter.effective_wavelength, rel=1e-10)

    def test_filter_width_lambda_positive(self, r_filter):
        assert r_filter.filter_width_lambda > 0.0

    def test_filter_width_nu_positive(self, r_filter):
        assert r_filter.filter_width_nu > 0.0

    def test_wavelength_bounds(self, r_filter):
        lo, hi = r_filter.wavelength_bounds
        assert lo < hi
        assert lo == pytest.approx(r_filter.wavelength.min())
        assert hi == pytest.approx(r_filter.wavelength.max())

    def test_frequency_bounds(self, r_filter):
        lo, hi = r_filter.frequency_bounds
        assert lo < hi

    def test_eq_identical_filters(self, r_filter):
        lam = r_filter.wavelength
        T = r_filter.transmission
        filt2 = PhotometryFilter(lam, T, name="r2")
        assert r_filter == filt2  # eq ignores name

    def test_eq_different_transmission(self, r_filter, g_filter):
        assert r_filter != g_filter

    def test_eq_wrong_type(self, r_filter):
        assert r_filter.__eq__("not a filter") is NotImplemented


# ============================================================================== #
# PhotometryFilter — apply / convolve                                            #
# ============================================================================== #


class TestPhotometryFilterConvolve:
    def test_apply_flat_spectrum(self, r_filter):
        F0 = 3.631e-20
        F_nu = np.full(len(r_filter), F0)
        result = r_filter.apply(F_nu)
        assert result == pytest.approx(F0, rel=1e-6)

    def test_apply_batched_shape(self, r_filter):
        N_t = 10
        F_nu = np.ones((N_t, len(r_filter))) * 3.631e-20
        result = r_filter.apply(F_nu)
        assert result.shape == (N_t,)

    def test_apply_wrong_size_raises(self, r_filter):
        with pytest.raises(ValueError, match="last dimension"):
            r_filter.apply(np.ones(50))

    def test_convolve_nu_flat_spectrum(self, r_filter):
        # Build a coarser grid covering the filter
        nu_lo, nu_hi = r_filter.frequency_bounds
        nu = np.linspace(nu_lo * 0.5, nu_hi * 2.0, 1000)
        F_nu = np.full(1000, 3.631e-20)
        result = r_filter.convolve_nu(nu, F_nu)
        assert result == pytest.approx(3.631e-20, rel=1e-4)

    def test_convolve_nu_batched(self, r_filter):
        nu_lo, nu_hi = r_filter.frequency_bounds
        nu = np.linspace(nu_lo * 0.5, nu_hi * 2.0, 500)
        F_nu = np.ones((5, 500)) * 3.631e-20
        result = r_filter.convolve_nu(nu, F_nu)
        assert result.shape == (5,)
        assert np.allclose(result, 3.631e-20, rtol=1e-4)

    def test_convolve_lambda_consistent_with_nu(self, r_filter):
        # Build a flat F_lambda spectrum and cross-check both paths
        lam_lo, lam_hi = r_filter.wavelength_bounds
        lam = np.linspace(lam_lo * 0.5, lam_hi * 2.0, 1000)  # cm
        c_cgs = 2.99792458e10
        # Flat F_nu spectrum → F_lambda = F_nu * c / lambda^2
        F_nu_flat = 3.631e-20
        F_lambda = F_nu_flat * c_cgs / lam**2

        res_lam = r_filter.convolve_lambda(lam, F_lambda)
        nu = c_cgs / lam
        res_nu = r_filter.convolve_nu(nu, np.full_like(lam, F_nu_flat))
        assert res_lam == pytest.approx(res_nu, rel=1e-4)

    def test_call_shorthand(self, r_filter):
        nu_lo, nu_hi = r_filter.frequency_bounds
        nu = np.linspace(nu_lo * 0.5, nu_hi * 2.0, 500)
        F_nu = np.ones(500) * 3.631e-20
        assert r_filter(nu, F_nu) == pytest.approx(r_filter.convolve_nu(nu, F_nu), rel=1e-12)


# ============================================================================== #
# PhotometryFilter — IO roundtrips                                               #
# ============================================================================== #


class TestPhotometryFilterIO:
    def test_dict_roundtrip(self, r_filter):
        d = r_filter.to_dict()
        filt2 = PhotometryFilter.from_dict(d)
        assert filt2 == r_filter
        assert filt2.name == r_filter.name
        assert filt2.weighting == r_filter.weighting

    def test_json_roundtrip(self, r_filter, tmp_path):
        path = tmp_path / "filter.json"
        r_filter.to_json(path)
        filt2 = PhotometryFilter.from_json(path)
        assert filt2 == r_filter
        assert filt2.name == r_filter.name

    def test_hdf5_roundtrip(self, r_filter, tmp_path):
        path = tmp_path / "filter.h5"
        r_filter.to_hdf5(path)
        filt2 = PhotometryFilter.from_hdf5(path)
        assert filt2 == r_filter
        assert filt2.name == r_filter.name
        assert filt2.weighting == r_filter.weighting

    def test_hdf5_custom_key(self, r_filter, tmp_path):
        path = tmp_path / "filter.h5"
        r_filter.to_hdf5(path, key="my_filter")
        filt2 = PhotometryFilter.from_hdf5(path, key="my_filter")
        assert filt2 == r_filter

    def test_table_roundtrip(self, r_filter):
        tbl = r_filter.to_table()
        assert "wavelength_cm" in tbl.colnames
        assert "transmission" in tbl.colnames
        filt2 = PhotometryFilter.from_table(tbl)
        assert filt2 == r_filter
        assert filt2.name == r_filter.name

    def test_array_roundtrip(self, r_filter):
        arr = r_filter.to_array()
        assert arr.shape == (2, len(r_filter))
        filt2 = PhotometryFilter.from_array(arr, name=r_filter.name)
        assert filt2 == r_filter

    def test_from_array_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            PhotometryFilter.from_array(np.ones((3, 100)))


# ============================================================================== #
# PhotometryFilter — plotting                                                    #
# ============================================================================== #


class TestPhotometryFilterPlot:
    def test_plot_returns_fig_axes(self, r_filter):
        import matplotlib

        matplotlib.use("Agg")
        fig, ax = r_filter.plot()
        assert fig is not None
        assert ax is not None

    def test_plot_diagnostic(self, r_filter, diagnostic_plots, diagnostic_plots_dir):
        if not diagnostic_plots:
            pytest.skip("Diagnostic plots not requested.")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = r_filter.plot(label="r-band")
        ax.set_title("r-band diagnostic")
        fig.savefig(diagnostic_plots_dir / "filter_r_band.png", dpi=100)
        plt.close(fig)


# ============================================================================== #
# FilterBundle — construction and properties                                     #
# ============================================================================== #


class TestFilterBundleConstruction:
    def test_construction(self, bundle):
        assert bundle.n_filters == 3

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            FilterBundle({})

    def test_len(self, bundle):
        assert len(bundle) == 3

    def test_repr(self, bundle):
        s = repr(bundle)
        assert "FilterBundle" in s
        assert "'g'" in s

    def test_str_equals_repr(self, bundle):
        assert str(bundle) == repr(bundle)

    def test_getitem(self, bundle, r_filter):
        assert bundle["r"] == r_filter

    def test_contains(self, bundle):
        assert "g" in bundle
        assert "z" not in bundle

    def test_iter(self, bundle):
        names = list(bundle)
        assert set(names) == {"g", "r", "i"}

    def test_filter_names_ordered(self, bundle):
        assert bundle.filter_names == ["g", "r", "i"]

    def test_frequency_grid_ascending(self, bundle):
        nu = bundle.frequency_grid
        assert np.all(np.diff(nu) >= 0.0)

    def test_weight_matrix_shape(self, bundle):
        W = bundle.weight_matrix
        assert W.shape[0] == 3
        assert W.shape[1] == len(bundle.frequency_grid)

    def test_weight_matrix_rows_sum_to_one(self, bundle):
        W = bundle.weight_matrix
        # Each row that has any non-zero entry should sum to 1
        for i in range(W.shape[0]):
            row_sum = W[i].sum()
            assert row_sum == pytest.approx(1.0, abs=1e-10)


# ============================================================================== #
# FilterBundle — convolution                                                     #
# ============================================================================== #


class TestFilterBundleConvolve:
    def test_apply_shape_1d(self, bundle):
        F_nu = np.ones(len(bundle.frequency_grid)) * 3.631e-20
        result = bundle.apply(F_nu)
        assert result.shape == (3,)

    def test_apply_shape_2d(self, bundle):
        N_t = 7
        F_nu = np.ones((N_t, len(bundle.frequency_grid))) * 3.631e-20
        result = bundle.apply(F_nu)
        assert result.shape == (N_t, 3)

    def test_apply_wrong_size_raises(self, bundle):
        with pytest.raises(ValueError, match="common grid"):
            bundle.apply(np.ones(10))

    def test_apply_flat_spectrum_near_zero_mag(self, bundle):
        F0 = 3.631e-20
        F_nu = np.full(len(bundle.frequency_grid), F0)
        result = bundle.apply(F_nu)
        assert np.allclose(result, F0, rtol=1e-4)

    def test_convolve_nu_consistent_with_individual_filters(self, bundle, g_filter, r_filter, i_filter):
        # Build a common test SED
        nu_range = np.linspace(2e14, 1e15, 2000)
        F_nu = np.ones_like(nu_range) * 3.631e-20

        bundle_result = bundle.convolve_nu(nu_range, F_nu)

        individual = np.array(
            [
                g_filter.convolve_nu(nu_range, F_nu),
                r_filter.convolve_nu(nu_range, F_nu),
                i_filter.convolve_nu(nu_range, F_nu),
            ]
        )
        assert np.allclose(bundle_result, individual, rtol=1e-4)

    def test_convolve_nu_batched(self, bundle):
        nu_range = np.linspace(2e14, 1e15, 500)
        F_nu = np.ones((10, 500)) * 3.631e-20
        result = bundle.convolve_nu(nu_range, F_nu)
        assert result.shape == (10, 3)

    def test_convolve_lambda(self, bundle):
        c_cgs = 2.99792458e10
        lam_range = np.linspace(3e-5, 1.2e-4, 2000)  # cm
        F_nu_flat = 3.631e-20
        F_lambda = F_nu_flat * c_cgs / lam_range**2
        result = bundle.convolve_lambda(lam_range, F_lambda)
        assert result.shape == (3,)
        assert np.allclose(result, F_nu_flat, rtol=1e-3)

    def test_call_shorthand(self, bundle):
        nu_range = np.linspace(2e14, 1e15, 500)
        F_nu = np.ones(500) * 3.631e-20
        assert np.allclose(bundle(nu_range, F_nu), bundle.convolve_nu(nu_range, F_nu))


# ============================================================================== #
# FilterBundle — mutation                                                        #
# ============================================================================== #


class TestFilterBundleMutation:
    def test_add_filter(self, bundle):
        z_filt = _gaussian_filter(9000, 800, name="z")
        bundle.add_filter("z", z_filt)
        assert "z" in bundle
        assert bundle.n_filters == 4
        # Weight matrix should be rebuilt
        assert bundle.weight_matrix.shape[0] == 4

    def test_add_duplicate_raises(self, bundle, r_filter):
        with pytest.raises(KeyError, match="already exists"):
            bundle.add_filter("r", r_filter)

    def test_remove_filter(self, bundle):
        removed = bundle.remove_filter("i")
        assert removed.name == "i"
        assert "i" not in bundle
        assert bundle.n_filters == 2

    def test_remove_nonexistent_raises(self, bundle):
        with pytest.raises(KeyError, match="No filter"):
            bundle.remove_filter("z")

    def test_remove_last_raises(self, r_filter):
        b = FilterBundle({"r": r_filter})
        with pytest.raises(ValueError, match="last filter"):
            b.remove_filter("r")


# ============================================================================== #
# FilterBundle — IO roundtrips                                                   #
# ============================================================================== #


class TestFilterBundleIO:
    def test_dict_roundtrip(self, bundle):
        d = bundle.to_dict()
        assert "filters" in d
        b2 = FilterBundle.from_dict(d)
        assert b2.filter_names == bundle.filter_names
        for name in bundle.filter_names:
            assert b2[name] == bundle[name]

    def test_json_roundtrip(self, bundle, tmp_path):
        path = tmp_path / "bundle.json"
        bundle.to_json(path)
        b2 = FilterBundle.from_json(path)
        assert b2.filter_names == bundle.filter_names

    def test_hdf5_roundtrip(self, bundle, tmp_path):
        path = tmp_path / "bundle.h5"
        bundle.to_hdf5(path)
        b2 = FilterBundle.from_hdf5(path)
        assert b2.filter_names == bundle.filter_names
        for name in bundle.filter_names:
            assert b2[name] == bundle[name]

    def test_hdf5_order_preserved(self, bundle, tmp_path):
        path = tmp_path / "bundle_order.h5"
        bundle.to_hdf5(path)
        b2 = FilterBundle.from_hdf5(path)
        assert b2.filter_names == bundle.filter_names


# ============================================================================== #
# FilterBundle — plotting                                                        #
# ============================================================================== #


class TestFilterBundlePlot:
    def test_plot_returns_fig_axes(self, bundle):
        import matplotlib

        matplotlib.use("Agg")
        fig, ax = bundle.plot()
        assert fig is not None
        assert ax is not None

    def test_plot_diagnostic(self, bundle, diagnostic_plots, diagnostic_plots_dir):
        if not diagnostic_plots:
            pytest.skip("Diagnostic plots not requested.")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = bundle.plot()
        fig.savefig(diagnostic_plots_dir / "filter_bundle_gri.png", dpi=100)
        plt.close(fig)


# ============================================================================== #
# Magnitude utilities                                                            #
# ============================================================================== #


class TestMagnitudeUtilities:
    def test_ab_zero_point(self):
        mag = flux_to_ab_mag(3.631e-20)
        assert float(mag) == pytest.approx(0.0, abs=1e-6)

    def test_ab_mag_roundtrip(self):
        F = np.array([1e-23, 3.631e-20, 1e-17])
        assert np.allclose(ab_mag_to_flux(flux_to_ab_mag(F)), F, rtol=1e-10)

    def test_ab_mag_nonpositive_flux(self):
        mag = flux_to_ab_mag(0.0)
        assert np.isinf(mag)
        mag_neg = flux_to_ab_mag(-1.0)
        assert np.isnan(mag_neg)

    def test_ab_mag_array_shape(self):
        F = np.ones((3, 4)) * 3.631e-20
        mag = flux_to_ab_mag(F)
        assert mag.shape == (3, 4)
        assert np.allclose(mag, 0.0, atol=1e-6)

    def test_ab_brighter_star_lower_mag(self):
        assert flux_to_ab_mag(1e-19) < flux_to_ab_mag(1e-23)

    def test_st_mag_roundtrip(self):
        F = np.array([1e-15, 1e-12, 1e-9])
        assert np.allclose(st_mag_to_flux_lambda(flux_lambda_to_st_mag(F)), F, rtol=1e-10)

    def test_st_mag_nonpositive(self):
        mag = flux_lambda_to_st_mag(0.0)
        assert np.isinf(mag)

    def test_filter_to_ab_mag_single_filter(self, r_filter):
        nu_lo, nu_hi = r_filter.frequency_bounds
        nu = np.linspace(nu_lo * 0.5, nu_hi * 2.0, 1000)
        F_nu = np.full(1000, 3.631e-20)
        mag = filter_to_ab_mag(r_filter, nu, F_nu)
        assert float(mag) == pytest.approx(0.0, abs=0.05)

    def test_filter_to_ab_mag_bundle(self, bundle):
        nu = np.linspace(2e14, 1e15, 2000)
        F_nu = np.full(2000, 3.631e-20)
        mags = filter_to_ab_mag(bundle, nu, F_nu)
        assert mags.shape == (3,)
        assert np.allclose(mags, 0.0, atol=0.05)


# ============================================================================== #
# Loading helpers                                                                #
# ============================================================================== #


class TestLoadingHelpers:
    def test_load_filter_from_file(self, tmp_path):
        # Write a simple 2-column ASCII filter file (wavelength in Å, transmission)
        path = tmp_path / "myfilter.dat"
        lam_aa = np.linspace(5500, 7000, 100)
        T = np.exp(-0.5 * ((lam_aa - 6200) / 300) ** 2)
        data = np.column_stack([lam_aa, T])
        np.savetxt(path, data)

        filt = load_filter_from_file(path, wavelength_unit="AA")
        assert filt.name == "myfilter"
        assert len(filt) == 100
        # Check wavelength was converted to cm
        assert filt.wavelength.max() == pytest.approx(7000e-8, rel=1e-4)

    def test_load_filter_from_file_custom_name(self, tmp_path):
        path = tmp_path / "anonymous.dat"
        data = np.column_stack([np.linspace(5000, 6000, 50), np.ones(50)])
        np.savetxt(path, data)
        filt = load_filter_from_file(path, name="custom-g")
        assert filt.name == "custom-g"

    def test_load_filter_from_file_nm(self, tmp_path):
        path = tmp_path / "nm_filter.dat"
        lam_nm = np.linspace(550, 700, 80)
        T = np.ones(80)
        np.savetxt(path, np.column_stack([lam_nm, T]))
        filt = load_filter_from_file(path, wavelength_unit="nm")
        # 550 nm = 550e-7 cm
        assert filt.wavelength.min() == pytest.approx(550e-7, rel=1e-4)

    def test_load_filter_from_speclite_missing_import(self):
        with mock.patch.dict("sys.modules", {"speclite": None, "speclite.filters": None}):
            with pytest.raises(ImportError, match="speclite"):
                load_filter_from_speclite("sdss-r")


# ============================================================================== #
# SVO loading helpers                                                            #
# ============================================================================== #


def _make_svo_transmission_table(center_aa=6000.0, width_aa=200.0, n=50):
    """Return a fake SVO transmission table (astropy Table with Wavelength/Transmission)."""
    from astropy.table import Table as ATable

    lam = np.linspace(center_aa - 2 * width_aa, center_aa + 2 * width_aa, n)
    trans = np.exp(-0.5 * ((lam - center_aa) / width_aa) ** 2)
    return ATable({"Wavelength": lam, "Transmission": trans})


def _make_svo_index_table(filter_ids):
    """Return a fake SVO filter list table."""
    from astropy.table import Table as ATable

    return ATable({"filterID": filter_ids, "WavelengthEff": [6000.0] * len(filter_ids)})


def _svo_modules(get_transmission_data=None, get_filter_list=None):
    """Build a sys.modules patch dict with a fake astroquery.svo_fps module."""
    svo_fps_mod = mock.MagicMock()
    if get_transmission_data is not None:
        svo_fps_mod.SvoFps.get_transmission_data.side_effect = get_transmission_data
    if get_filter_list is not None:
        svo_fps_mod.SvoFps.get_filter_list.side_effect = get_filter_list
    return {"astroquery": mock.MagicMock(), "astroquery.svo_fps": svo_fps_mod}, svo_fps_mod


class TestSVOLoadingHelpers:
    def test_load_filter_from_svo(self):
        table = _make_svo_transmission_table()
        mods, _ = _svo_modules(get_transmission_data=lambda *a, **kw: table)
        with mock.patch.dict("sys.modules", mods):
            filt = load_filter_from_svo("Kepler/Kepler.K")
        assert isinstance(filt, PhotometryFilter)
        assert filt.name == "Kepler/Kepler.K"
        assert len(filt) == 50
        assert filt.wavelength.max() == pytest.approx(6400e-8, rel=1e-3)

    def test_load_filter_from_svo_clips_negatives(self):
        table = _make_svo_transmission_table()
        table["Transmission"][0] = -0.1
        table["Transmission"][1] = float("nan")
        mods, _ = _svo_modules(get_transmission_data=lambda *a, **kw: table)
        with mock.patch.dict("sys.modules", mods):
            filt = load_filter_from_svo("Generic/Johnson.R")
        assert filt.transmission[0] == 0.0
        assert filt.transmission[1] == 0.0

    def test_load_filter_from_svo_not_found(self):
        from astropy.table import Table as ATable

        empty = ATable({"Wavelength": [], "Transmission": []})
        mods, _ = _svo_modules(get_transmission_data=lambda *a, **kw: empty)
        with mock.patch.dict("sys.modules", mods):
            with pytest.raises(ValueError, match="not found"):
                load_filter_from_svo("Nonexistent/Filter.X")

    def test_load_filter_from_svo_import_error(self):
        with mock.patch.dict("sys.modules", {"astroquery": None, "astroquery.svo_fps": None}):
            with pytest.raises(ImportError, match="astroquery"):
                load_filter_from_svo("Kepler/Kepler.K")

    def test_list_svo_filters(self):
        index = _make_svo_index_table(["Kepler/Kepler.K", "Kepler/Kepler.LP"])
        mods, _ = _svo_modules(get_filter_list=lambda *a, **kw: index)
        with mock.patch.dict("sys.modules", mods):
            result = list_svo_filters("Kepler")
        assert "filterID" in result.colnames
        assert list(result["filterID"]) == ["Kepler/Kepler.K", "Kepler/Kepler.LP"]

    def test_list_svo_filters_with_instrument(self):
        # The SVO server doesn't support server-side instrument filtering,
        # so list_svo_filters fetches the full facility list and filters client-side.
        index = _make_svo_index_table(["HST/WFC3_IR.F160W", "HST/ACS_WFC.F606W"])
        mods, svo_mod = _svo_modules(get_filter_list=lambda *a, **kw: index)
        with mock.patch.dict("sys.modules", mods):
            result = list_svo_filters("HST", instrument="WFC3_IR")
        # get_filter_list should be called WITHOUT an instrument kwarg
        svo_mod.SvoFps.get_filter_list.assert_called_once_with("HST", cache=True, timeout=60)
        # Client-side filter keeps only WFC3_IR entries
        assert list(result["filterID"]) == ["HST/WFC3_IR.F160W"]

    def test_load_filters_from_svo(self):
        index = _make_svo_index_table(["Kepler/Kepler.K", "Kepler/Kepler.LP"])
        table_k = _make_svo_transmission_table(center_aa=6000)
        table_lp = _make_svo_transmission_table(center_aa=8000)

        def fake_transmission(filter_id, **_kw):
            return table_k if filter_id == "Kepler/Kepler.K" else table_lp

        mods, _ = _svo_modules(
            get_filter_list=lambda *a, **kw: index,
            get_transmission_data=fake_transmission,
        )
        with mock.patch.dict("sys.modules", mods):
            result = load_filters_from_svo("Kepler")

        assert set(result.keys()) == {"Kepler/Kepler.K", "Kepler/Kepler.LP"}
        assert all(isinstance(v, PhotometryFilter) for v in result.values())
        bundle = FilterBundle(result)
        assert bundle.n_filters == 2

    def test_load_filters_from_svo_skips_empty(self):
        from astropy.table import Table as ATable

        index = _make_svo_index_table(["Kepler/Kepler.K", "Kepler/Kepler.Bad"])
        good_table = _make_svo_transmission_table()
        empty_table = ATable({"Wavelength": [], "Transmission": []})

        def fake_transmission(filter_id, **_kw):
            return good_table if filter_id == "Kepler/Kepler.K" else empty_table

        mods, _ = _svo_modules(
            get_filter_list=lambda *a, **kw: index,
            get_transmission_data=fake_transmission,
        )
        with mock.patch.dict("sys.modules", mods):
            result = load_filters_from_svo("Kepler")

        assert list(result.keys()) == ["Kepler/Kepler.K"]
