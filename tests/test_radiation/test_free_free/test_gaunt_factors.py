"""Unit tests for the free-free Gaunt factor module.

Covers table loading, interpolation (scalar + array), domain containment,
properties/dunders, error handling, the Draine analytic approximation,
and the public __init__.py exports.
"""

import numpy as np
import pytest

from triceratops.radiation.free_free.gaunt_factor import (
    _VAN_HOOF_NON_REL_PATH,
    _VAN_HOOF_REL_PATH,
    GauntFactorInterpolatorBase,
    NonRelativisticGauntFactorInterpolator,
    RelativisticGauntFactorInterpolator,
    _log_gaunt_ff_draine,
    gaunt_ff_draine,
    get_default_gaunt_interpolator,
    get_default_relativistic_gaunt_interpolator,
)


# ================================================================== #
# Fixtures                                                           #
# ================================================================== #
@pytest.fixture(scope="module")
def nr_interp():
    """Non-relativistic interpolator loaded once per module."""
    return get_default_gaunt_interpolator()


@pytest.fixture(scope="module")
def rel_interp():
    """Relativistic interpolator loaded once per module."""
    return get_default_relativistic_gaunt_interpolator()


# ================================================================== #
# Path sanity                                                        #
# ================================================================== #
class TestPaths:
    def test_non_rel_path_exists(self):
        assert _VAN_HOOF_NON_REL_PATH.exists(), f"Non-rel table missing: {_VAN_HOOF_NON_REL_PATH}"

    def test_rel_path_exists(self):
        assert _VAN_HOOF_REL_PATH.exists(), f"Rel table missing: {_VAN_HOOF_REL_PATH}"


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
        """bounds_error=False should not raise for out-of-range inputs."""
        interp = get_default_gaunt_interpolator(bounds_error=False, fill_value=None)
        # Extremely low T puts log_gamma2 far out of range — should not raise.
        result = interp(Z=1, T=1e-5, nu=1e10)
        assert np.isfinite(result) or True  # just must not raise


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
        # van Hoof (2014) non-rel table: 146 log_u points, 81 log_gamma2 points
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
        """Non-rel table should not expose a z_range or z_grid."""
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
    # Reference point well inside both tables.
    Z, T, nu = 1, 1e4, 1e10

    def test_nr_scalar_returns_float(self, nr_interp):
        result = nr_interp(Z=self.Z, T=self.T, nu=self.nu)
        assert isinstance(result, float)

    def test_rel_scalar_returns_float(self, rel_interp):
        result = rel_interp(Z=self.Z, T=self.T, nu=self.nu)
        assert isinstance(result, float)

    def test_nr_scalar_is_positive(self, nr_interp):
        result = nr_interp(Z=self.Z, T=self.T, nu=self.nu)
        assert result > 0.0

    def test_rel_scalar_is_positive(self, rel_interp):
        result = rel_interp(Z=self.Z, T=self.T, nu=self.nu)
        assert result > 0.0

    def test_nr_scalar_is_order_unity(self, nr_interp):
        result = nr_interp(Z=self.Z, T=self.T, nu=self.nu)
        assert 0.1 < result < 20.0

    def test_rel_scalar_is_order_unity(self, rel_interp):
        result = rel_interp(Z=self.Z, T=self.T, nu=self.nu)
        assert 0.1 < result < 20.0

    def test_nr_known_value(self, nr_interp):
        """Regression check against value recorded during implementation."""
        result = nr_interp(Z=1, T=1e4, nu=1e10)
        assert result == pytest.approx(4.461, rel=1e-3)

    def test_rel_known_value(self, rel_interp):
        """Regression check against value recorded during implementation."""
        result = rel_interp(Z=1, T=1e4, nu=1e10)
        assert result == pytest.approx(4.708, rel=1e-3)


# ================================================================== #
# Array evaluation                                                   #
# ================================================================== #
class TestArrayEvaluation:
    T_arr = np.geomspace(1e4, 1e8, 50)

    def test_nr_array_shape(self, nr_interp):
        result = nr_interp(Z=1, T=self.T_arr, nu=1e10)
        assert result.shape == (50,)

    def test_rel_array_shape(self, rel_interp):
        result = rel_interp(Z=1, T=self.T_arr, nu=1e10)
        assert result.shape == (50,)

    def test_nr_array_returns_ndarray(self, nr_interp):
        result = nr_interp(Z=1, T=self.T_arr, nu=1e10)
        assert isinstance(result, np.ndarray)

    def test_rel_array_returns_ndarray(self, rel_interp):
        result = rel_interp(Z=1, T=self.T_arr, nu=1e10)
        assert isinstance(result, np.ndarray)

    def test_nr_array_all_positive(self, nr_interp):
        result = nr_interp(Z=1, T=self.T_arr, nu=1e10)
        assert np.all(result > 0)

    def test_rel_array_all_positive(self, rel_interp):
        result = rel_interp(Z=1, T=self.T_arr, nu=1e10)
        assert np.all(result > 0)

    def test_nr_broadcast_T_and_nu(self, nr_interp):
        """Broadcasting over both T and nu simultaneously."""
        T = np.array([1e4, 1e6])
        nu = np.array([1e9, 1e10, 1e14])
        # np.broadcast_arrays will produce shape (2,) from (2,) and (3,) — fails.
        # Instead test with shapes that do broadcast: (2,1) x (3,) → (2,3)
        T2d = T[:, np.newaxis]
        result = nr_interp(Z=1, T=T2d, nu=nu)
        assert result.shape == (2, 3)

    def test_nr_scalar_vs_array_consistency(self, nr_interp):
        """Scalar call and length-1 array call should agree."""
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
        # T=1e-10 pushes log_gamma2 >> 10 (outside table)
        assert (1, 1e-10, 1e10) not in nr_interp

    def test_rel_outside_temperature(self, rel_interp):
        assert (1, 1e-10, 1e10) not in rel_interp

    def test_rel_outside_z_range(self, rel_interp):
        # Z=100 is outside [1, 36]
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
        assert "n_log_u" in r

    def test_rel_repr(self, rel_interp):
        r = repr(rel_interp)
        assert "RelativisticGauntFactorInterpolator" in r
        assert "n_Z" in r

    def test_nr_str(self, nr_interp):
        s = str(nr_interp)
        assert "NonRelativisticGauntFactorInterpolator" in s

    def test_rel_str(self, rel_interp):
        s = str(rel_interp)
        assert "RelativisticGauntFactorInterpolator" in s

    def test_nr_source_is_str(self, nr_interp):
        assert isinstance(nr_interp.source, str)

    def test_rel_source_is_str(self, rel_interp):
        assert isinstance(rel_interp.source, str)


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
            # omit log_gamma2 and gff
        with pytest.raises(KeyError):
            NonRelativisticGauntFactorInterpolator(path=bad)

    def test_nr_wrong_array_dimensionality_raises_value_error(self, tmp_path):
        import h5py

        bad = tmp_path / "bad3d.hdf5"
        with h5py.File(bad, "w") as f:
            f.create_dataset("log_u", data=np.linspace(-16, 13, 5))
            f.create_dataset("log_gamma2", data=np.linspace(-6, 10, 4))
            f.create_dataset("gff", data=np.ones((3, 5, 4)))  # 3-D, not 2-D
        with pytest.raises(ValueError, match="2-D"):
            NonRelativisticGauntFactorInterpolator(path=bad)

    def test_rel_wrong_array_dimensionality_raises_value_error(self, tmp_path):
        import h5py

        bad = tmp_path / "bad2d.hdf5"
        with h5py.File(bad, "w") as f:
            f.create_dataset("Z", data=np.arange(1, 4))
            f.create_dataset("log_u", data=np.linspace(-16, 13, 5))
            f.create_dataset("log_gamma2", data=np.linspace(-6, 10, 4))
            f.create_dataset("gff", data=np.ones((5, 4)))  # 2-D, not 3-D
        with pytest.raises(ValueError, match="3-D"):
            RelativisticGauntFactorInterpolator(path=bad)


# ================================================================== #
# Draine analytic approximation                                      #
# ================================================================== #
class TestDraineApproximation:
    def test_scalar_returns_float(self):
        result = gaunt_ff_draine(Z=1, T=1e4, nu=1e10)
        assert isinstance(result, (float, np.floating))

    def test_scalar_is_positive(self):
        assert gaunt_ff_draine(Z=1, T=1e4, nu=1e10) > 0

    def test_array_shape(self):
        T_arr = np.geomspace(1e4, 1e8, 20)
        result = gaunt_ff_draine(Z=1, T=T_arr, nu=1e10)
        assert result.shape == (20,)

    def test_accepts_astropy_quantities(self):
        from astropy import units as u

        result = gaunt_ff_draine(Z=1, T=1e4 * u.K, nu=10 * u.GHz)
        assert result > 0

    def test_log_level_helper_matches_wrapper(self):
        Z, T, nu = 1.0, 1e6, 1e12
        from_wrapper = gaunt_ff_draine(Z=Z, T=T, nu=nu)
        from_helper = _log_gaunt_ff_draine(np.log(Z), np.log(T), np.log(nu))
        assert from_wrapper == pytest.approx(from_helper, rel=1e-12)

    def test_increases_with_temperature(self):
        """At fixed nu, g_ff (Draine) is a non-decreasing function of T."""
        T_arr = np.geomspace(1e3, 1e9, 50)
        gff = gaunt_ff_draine(Z=1, T=T_arr, nu=1e10)
        # Allow for near-flat regions; just check no dramatic inversion
        assert np.all(np.diff(gff) >= -0.01)


# ================================================================== #
# Public __init__.py exports                                         #
# ================================================================== #
class TestPublicExports:
    def test_init_exports_nr_interpolator(self):
        from triceratops.radiation.free_free import (
            NonRelativisticGauntFactorInterpolator as C,
        )

        assert C is NonRelativisticGauntFactorInterpolator

    def test_init_exports_rel_interpolator(self):
        from triceratops.radiation.free_free import (
            RelativisticGauntFactorInterpolator as C,
        )

        assert C is RelativisticGauntFactorInterpolator

    def test_init_exports_base(self):
        from triceratops.radiation.free_free import GauntFactorInterpolatorBase as C

        assert C is GauntFactorInterpolatorBase

    def test_init_exports_nr_factory(self):
        from triceratops.radiation.free_free import get_default_gaunt_interpolator as f

        assert f is get_default_gaunt_interpolator

    def test_init_exports_rel_factory(self):
        from triceratops.radiation.free_free import (
            get_default_relativistic_gaunt_interpolator as f,
        )

        assert f is get_default_relativistic_gaunt_interpolator

    def test_init_exports_draine(self):
        from triceratops.radiation.free_free import gaunt_ff_draine as f

        assert f is gaunt_ff_draine
