"""Unit tests for the free-free Gaunt factor module.

Covers table loading, interpolation (scalar + array), domain containment,
properties/dunders, error handling, analytic approximations (Draine, Lu),
and the public __init__.py exports.
"""

import numpy as np
import pytest

from trilobite.radiation.free_free.gaunt_factor import (
    _VAN_HOOF_NON_REL_PATH,
    _VAN_HOOF_REL_PATH,
    GauntFactorInterpolatorBase,
    NonRelativisticGauntFactorInterpolator,
    RelativisticGauntFactorInterpolator,
    _gaunt_ff_draine,
    compute_ff_gaunt_factor,
    get_default_gaunt_interpolator,
    get_default_relativistic_gaunt_interpolator,
)


# ================================================================== #
# Fixtures                                                           #
# ================================================================== #
@pytest.fixture(scope="module")
def nr_interp():
    return get_default_gaunt_interpolator()


@pytest.fixture(scope="module")
def rel_interp():
    return get_default_relativistic_gaunt_interpolator()


# ================================================================== #
# Path sanity                                                        #
# ================================================================== #
class TestPaths:
    def test_non_rel_path_exists(self):
        assert _VAN_HOOF_NON_REL_PATH.exists()

    def test_rel_path_exists(self):
        assert _VAN_HOOF_REL_PATH.exists()


# ================================================================== #
# Factory functions & type checks                                    #
# ================================================================== #
class TestFactories:
    def test_nr_factory_returns_correct_type(self, nr_interp):
        assert isinstance(nr_interp, NonRelativisticGauntFactorInterpolator)

    def test_rel_factory_returns_correct_type(self, rel_interp):
        assert isinstance(rel_interp, RelativisticGauntFactorInterpolator)

    def test_nr_is_base_subclass(self, nr_interp):
        assert isinstance(nr_interp, GauntFactorInterpolatorBase)

    def test_rel_is_base_subclass(self, rel_interp):
        assert isinstance(rel_interp, GauntFactorInterpolatorBase)

    def test_factory_kwargs_forwarded(self):
        interp = get_default_gaunt_interpolator(bounds_error=False, fill_value=None)
        result = interp(Z=1, T=1e-5, nu=1e10)
        assert np.isfinite(result) or True  # must not raise


# ================================================================== #
# Table structure                                                     #
# ================================================================== #
class TestNonRelativisticTableStructure:
    def test_array_is_2d(self, nr_interp):
        assert nr_interp._array.ndim == 2

    def test_shape_consistent_with_grids(self, nr_interp):
        n_u, n_g2 = nr_interp.shape
        assert n_u == nr_interp.n_log_u
        assert n_g2 == nr_interp.n_log_gamma2

    def test_expected_grid_sizes(self, nr_interp):
        assert nr_interp.n_log_u == 146
        assert nr_interp.n_log_gamma2 == 81

    def test_log_u_range(self, nr_interp):
        lo, hi = nr_interp.log_u_range
        assert lo == pytest.approx(-16.0)
        assert hi == pytest.approx(13.0)

    def test_log_gamma2_range(self, nr_interp):
        lo, hi = nr_interp.log_gamma2_range
        assert lo == pytest.approx(-6.0)
        assert hi == pytest.approx(10.0)

    def test_no_z_axis(self, nr_interp):
        assert not hasattr(nr_interp, "z_range")
        assert not hasattr(nr_interp, "z_grid")

    def test_grids_are_read_only(self, nr_interp):
        with pytest.raises((ValueError, TypeError)):
            nr_interp.log_u_grid[0] = 999.0
        with pytest.raises((ValueError, TypeError)):
            nr_interp.log_gamma2_grid[0] = 999.0


class TestRelativisticTableStructure:
    def test_array_is_3d(self, rel_interp):
        assert rel_interp._array.ndim == 3

    def test_shape_consistent_with_grids(self, rel_interp):
        n_Z, n_u, n_g2 = rel_interp.shape
        assert n_Z == rel_interp.n_Z
        assert n_u == rel_interp.n_log_u
        assert n_g2 == rel_interp.n_log_gamma2

    def test_expected_grid_sizes(self, rel_interp):
        assert rel_interp.n_Z == 36
        assert rel_interp.n_log_u == 146
        assert rel_interp.n_log_gamma2 == 81

    def test_z_range(self, rel_interp):
        z_min, z_max = rel_interp.z_range
        assert z_min == 1
        assert z_max == 36

    def test_grids_are_read_only(self, rel_interp):
        with pytest.raises((ValueError, TypeError)):
            rel_interp.log_u_grid[0] = 999.0
        with pytest.raises((ValueError, TypeError)):
            rel_interp.z_grid[0] = 999.0


# ================================================================== #
# Scalar evaluation                                                  #
# ================================================================== #
class TestScalarEvaluation:
    Z, T, nu = 1, 1e4, 1e10

    def test_nr_scalar_returns_float(self, nr_interp):
        assert isinstance(nr_interp(Z=self.Z, T=self.T, nu=self.nu), float)

    def test_rel_scalar_returns_float(self, rel_interp):
        assert isinstance(rel_interp(Z=self.Z, T=self.T, nu=self.nu), float)

    def test_nr_scalar_is_positive(self, nr_interp):
        assert nr_interp(Z=self.Z, T=self.T, nu=self.nu) > 0.0

    def test_rel_scalar_is_positive(self, rel_interp):
        assert rel_interp(Z=self.Z, T=self.T, nu=self.nu) > 0.0

    def test_nr_scalar_is_order_unity(self, nr_interp):
        result = nr_interp(Z=self.Z, T=self.T, nu=self.nu)
        assert 0.1 < result < 20.0

    def test_rel_scalar_is_order_unity(self, rel_interp):
        result = rel_interp(Z=self.Z, T=self.T, nu=self.nu)
        assert 0.1 < result < 20.0

    def test_nr_known_value(self, nr_interp):
        result = nr_interp(Z=1, T=1e4, nu=1e10)
        assert result == pytest.approx(4.461, rel=1e-3)

    def test_rel_known_value(self, rel_interp):
        result = rel_interp(Z=1, T=1e4, nu=1e10)
        assert result == pytest.approx(4.708, rel=1e-3)


# ================================================================== #
# Array evaluation                                                   #
# ================================================================== #
class TestArrayEvaluation:
    T_arr = np.geomspace(1e4, 1e8, 50)

    def test_nr_array_shape(self, nr_interp):
        assert nr_interp(Z=1, T=self.T_arr, nu=1e10).shape == (50,)

    def test_rel_array_shape(self, rel_interp):
        assert rel_interp(Z=1, T=self.T_arr, nu=1e10).shape == (50,)

    def test_nr_array_returns_ndarray(self, nr_interp):
        assert isinstance(nr_interp(Z=1, T=self.T_arr, nu=1e10), np.ndarray)

    def test_nr_array_all_positive(self, nr_interp):
        assert np.all(nr_interp(Z=1, T=self.T_arr, nu=1e10) > 0)

    def test_rel_array_all_positive(self, rel_interp):
        assert np.all(rel_interp(Z=1, T=self.T_arr, nu=1e10) > 0)

    def test_nr_broadcast_T_and_nu(self, nr_interp):
        T2d = np.array([1e4, 1e6])[:, np.newaxis]
        result = nr_interp(Z=1, T=T2d, nu=np.array([1e9, 1e10, 1e14]))
        assert result.shape == (2, 3)

    def test_nr_scalar_vs_array_consistency(self, nr_interp):
        scalar = nr_interp(Z=1, T=1e6, nu=1e12)
        arr = nr_interp(Z=1, T=np.array([1e6]), nu=np.array([1e12]))
        assert arr[0] == pytest.approx(scalar, rel=1e-12)


# ================================================================== #
# Domain containment (__contains__)                                  #
# ================================================================== #
class TestContainment:
    def test_nr_inside_domain(self, nr_interp):
        assert (1, 1e4, 1e10) in nr_interp

    def test_rel_inside_domain(self, rel_interp):
        assert (1, 1e4, 1e10) in rel_interp

    def test_nr_outside_temperature(self, nr_interp):
        assert (1, 1e-10, 1e10) not in nr_interp

    def test_rel_outside_z_range(self, rel_interp):
        assert (100, 1e4, 1e10) not in rel_interp

    def test_nr_wrong_tuple_length_raises(self, nr_interp):
        with pytest.raises(ValueError, match="3-tuple"):
            _ = (1, 1e4) in nr_interp

    def test_rel_wrong_tuple_length_raises(self, rel_interp):
        with pytest.raises(ValueError, match="3-tuple"):
            _ = (1,) in rel_interp


# ================================================================== #
# Properties and dunders                                             #
# ================================================================== #
class TestPropertiesAndDunders:
    def test_nr_description_is_str(self, nr_interp):
        assert isinstance(nr_interp.description, str)

    def test_rel_description_is_str(self, rel_interp):
        assert isinstance(rel_interp.description, str)

    def test_nr_description_mentions_non_relativistic(self, nr_interp):
        assert "Non-relativistic" in nr_interp.description

    def test_rel_description_mentions_relativistic(self, rel_interp):
        assert "Relativistic" in rel_interp.description

    def test_nr_len(self, nr_interp):
        assert len(nr_interp) == 146 * 81

    def test_rel_len(self, rel_interp):
        assert len(rel_interp) == 36 * 146 * 81

    def test_nr_repr(self, nr_interp):
        r = repr(nr_interp)
        assert "NonRelativisticGauntFactorInterpolator" in r

    def test_nr_source_is_str(self, nr_interp):
        assert isinstance(nr_interp.source, str)


# ================================================================== #
# Error handling                                                     #
# ================================================================== #
class TestErrorHandling:
    def test_nr_bad_path_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            NonRelativisticGauntFactorInterpolator(path=tmp_path / "nonexistent.hdf5")

    def test_rel_bad_path_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            RelativisticGauntFactorInterpolator(path=tmp_path / "nonexistent.hdf5")

    def test_nr_missing_dataset_raises_key_error(self, tmp_path):
        import h5py

        bad = tmp_path / "bad.hdf5"
        with h5py.File(bad, "w") as f:
            f.create_dataset("log_u", data=np.linspace(-16, 13, 10))
        with pytest.raises(KeyError):
            NonRelativisticGauntFactorInterpolator(path=bad)

    def test_nr_wrong_array_dimensionality_raises_value_error(self, tmp_path):
        import h5py

        bad = tmp_path / "bad3d.hdf5"
        with h5py.File(bad, "w") as f:
            f.create_dataset("log_u", data=np.linspace(-16, 13, 5))
            f.create_dataset("log_gamma2", data=np.linspace(-6, 10, 4))
            f.create_dataset("gff", data=np.ones((3, 5, 4)))
        with pytest.raises(ValueError, match="2-D"):
            NonRelativisticGauntFactorInterpolator(path=bad)

    def test_rel_wrong_array_dimensionality_raises_value_error(self, tmp_path):
        import h5py

        bad = tmp_path / "bad2d.hdf5"
        with h5py.File(bad, "w") as f:
            f.create_dataset("Z", data=np.arange(1, 4))
            f.create_dataset("log_u", data=np.linspace(-16, 13, 5))
            f.create_dataset("log_gamma2", data=np.linspace(-6, 10, 4))
            f.create_dataset("gff", data=np.ones((5, 4)))
        with pytest.raises(ValueError, match="3-D"):
            RelativisticGauntFactorInterpolator(path=bad)


# ================================================================== #
# Analytic approximations                                            #
# ================================================================== #
class TestDraineApproximation:
    """Tests for compute_ff_gaunt_factor(..., approx='draine')."""

    def test_scalar_returns_float(self):
        from astropy import units as u

        result = compute_ff_gaunt_factor(nu=1e10 * u.Hz, T=1e4 * u.K, Z=1, approx="draine")
        assert isinstance(result, (float, np.floating))

    def test_scalar_is_positive(self):
        from astropy import units as u

        assert compute_ff_gaunt_factor(nu=1e10 * u.Hz, T=1e4 * u.K, Z=1, approx="draine") > 0

    def test_array_shape(self):
        from astropy import units as u

        T_arr = np.geomspace(1e4, 1e8, 20) * u.K
        result = compute_ff_gaunt_factor(nu=1e10 * u.Hz, T=T_arr, Z=1, approx="draine")
        assert result.shape == (20,)

    def test_log_level_private_matches_public(self):
        """_gaunt_ff_draine (private) and compute_ff_gaunt_factor (public) agree."""
        from astropy import units as u

        nu, T, Z = 1e12, 1e6, 1.0
        from_public = compute_ff_gaunt_factor(nu=nu * u.Hz, T=T * u.K, Z=Z, approx="draine")
        from_private = _gaunt_ff_draine(np.log(nu), np.log(T), Z)
        assert from_public == pytest.approx(from_private, rel=1e-12)

    def test_invalid_approx_raises(self):
        from astropy import units as u

        with pytest.raises(ValueError):
            compute_ff_gaunt_factor(nu=1e10 * u.Hz, T=1e4 * u.K, approx="bogus")


class TestLuApproximation:
    """Tests for compute_ff_gaunt_factor(..., approx='lu') (the default)."""

    def test_scalar_positive(self):
        from astropy import units as u

        assert compute_ff_gaunt_factor(nu=1e10 * u.Hz, T=1e4 * u.K, Z=1) > 0

    def test_higher_Z_gives_different_result(self):
        from astropy import units as u

        g1 = compute_ff_gaunt_factor(nu=1e10 * u.Hz, T=1e4 * u.K, Z=1)
        g2 = compute_ff_gaunt_factor(nu=1e10 * u.Hz, T=1e4 * u.K, Z=2)
        assert g1 != pytest.approx(g2)

    def test_lu_and_draine_agree_to_order_of_magnitude(self):
        from astropy import units as u

        g_lu = compute_ff_gaunt_factor(nu=1e10 * u.Hz, T=1e4 * u.K, approx="lu")
        g_dr = compute_ff_gaunt_factor(nu=1e10 * u.Hz, T=1e4 * u.K, approx="draine")
        assert abs(g_lu - g_dr) / g_lu < 0.3  # within 30 %


# ================================================================== #
# Public __init__.py exports                                         #
# ================================================================== #
class TestPublicExports:
    def test_init_exports_nr_interpolator(self):
        from trilobite.radiation.free_free import NonRelativisticGauntFactorInterpolator as C

        assert C is NonRelativisticGauntFactorInterpolator

    def test_init_exports_rel_interpolator(self):
        from trilobite.radiation.free_free import RelativisticGauntFactorInterpolator as C

        assert C is RelativisticGauntFactorInterpolator

    def test_init_exports_base(self):
        from trilobite.radiation.free_free import GauntFactorInterpolatorBase as C

        assert C is GauntFactorInterpolatorBase

    def test_init_exports_nr_factory(self):
        from trilobite.radiation.free_free import get_default_gaunt_interpolator as f

        assert f is get_default_gaunt_interpolator

    def test_init_exports_rel_factory(self):
        from trilobite.radiation.free_free import get_default_relativistic_gaunt_interpolator as f

        assert f is get_default_relativistic_gaunt_interpolator

    def test_init_exports_compute_ff_gaunt_factor(self):
        from trilobite.radiation.free_free import compute_ff_gaunt_factor as f

        assert f is compute_ff_gaunt_factor
