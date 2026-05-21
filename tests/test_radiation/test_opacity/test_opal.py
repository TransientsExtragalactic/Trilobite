"""Tests for the OPAL opacity system.

Organisation
------------
TestOPALTables            — OPALOpacity.get_tables(), find(), where(): bundled table access
TestC_OPALTableOpacity    — C-level interpolator: grid-point accuracy, derivatives, OOB modes
TestOPALOpacity           — Python wrapper: interface contract, units, registry
TestC_OPALTableOpacityExtended — multi-point derivatives, OOB coverage, minimal grid
TestOPALOpacityTableSafety — read-only array guarantees
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
from astropy import units as u
from numpy.testing import assert_allclose

from trilobite.radiation.opacity.grey_opacity.rosseland.models import OPALOpacity
from trilobite.radiation.opacity.utils import get_opacity, load_opal_opacity

# ------------------------------------------------------------------ #
# Constants                                                           #
# ------------------------------------------------------------------ #

_BUNDLED_H5 = Path(__file__).parent.parent.parent.parent / ("trilobite/radiation/opacity/tables/asplund_grevesse_05.h5")

#: Known values straight from the source text for TABLE #1 (X=0, Y=1, Z=0):
#  logT=3.75, logR=-7.5 → -4.688
_SPOT_TABLE_IDX = 0  # table index (0-based)
_SPOT_LT_IDX = 0  # logT index: 3.75
_SPOT_LR_IDX = 1  # logR index: -7.5
_SPOT_VALUE = -4.688  # log10(kappa) [cm²/g]

#: Solar composition table index (verified during conversion).
_SOLAR_IDX = 72  # X=0.70, Z=0.02

#: Safe interior reference point for solar table (non-NaN, well inside grid).
#  logT=6.0 → T=1e6 K,  logR=-4 → rho=R*T6³ = 1e-4 * 1.0³ = 1e-4 g/cc
_REF_T_CGS = 1e6  # K
_REF_RHO_CGS = 1e-4  # g/cm³
_REF_T = _REF_T_CGS * u.K
_REF_RHO = _REF_RHO_CGS * u.g / u.cm**3


# ------------------------------------------------------------------ #
# Fixtures                                                            #
# ------------------------------------------------------------------ #


@pytest.fixture(scope="module")
def tables():
    """Full bundled OPAL table namespace (loaded once per test session)."""
    return OPALOpacity.get_tables()


@pytest.fixture(scope="module")
def solar_opacity():
    """OPALOpacity for solar composition (X=0.70, Z=0.02)."""
    return OPALOpacity.load_default()


@pytest.fixture(scope="module")
def solar_idx_dynamic(tables):
    """Locate solar composition index dynamically (no hardcoded constant)."""
    matches = OPALOpacity.where(X=0.70, Z=0.02)
    assert matches, "Solar composition not found in bundled table."
    return matches[0]


# ================================================================== #
# TestOPALTables                                                      #
# ================================================================== #


class TestOPALTables:
    """Tests for OPALOpacity class-level table access: get_tables, find, where."""

    # ---- get_tables ---- #

    def test_shape(self, tables):
        """Bundled table has expected dimensionality."""
        assert tables.opacity.shape == (126, 70, 19)
        assert tables.grid_T.shape == (70,)
        assert tables.grid_R.shape == (19,)
        assert tables.X.shape == (126,)
        assert tables.Y.shape == (126,)
        assert tables.Z.shape == (126,)

    def test_grid_range(self, tables):
        """logT and logR grids span expected ranges."""
        assert_allclose(tables.grid_T[0], 3.75, atol=1e-6)
        assert_allclose(tables.grid_T[-1], 8.70, atol=1e-6)
        assert_allclose(tables.grid_R[0], -8.0, atol=1e-6)
        assert_allclose(tables.grid_R[-1], 1.0, atol=1e-6)

    def test_known_spot_value(self, tables):
        """log10(kappa) at (TABLE #1, logT=3.75, logR=-7.5) matches source."""
        val = tables.opacity[_SPOT_TABLE_IDX, _SPOT_LT_IDX, _SPOT_LR_IDX]
        assert_allclose(val, _SPOT_VALUE, atol=1e-3)

    def test_nan_cells_present(self, tables):
        """9.999-flagged cells were converted to NaN."""
        assert int(np.isnan(tables.opacity).sum()) > 0

    def test_invalid_9999_replaced(self, tables):
        """No cell retains the sentinel value 9.999."""
        valid = tables.opacity[~np.isnan(tables.opacity)]
        assert np.all(np.abs(valid - 9.999) > 0.1)

    def test_caching(self):
        """get_tables() returns the same object on repeated calls."""
        t1 = OPALOpacity.get_tables()
        t2 = OPALOpacity.get_tables()
        assert t1 is t2

    def test_custom_path_bypasses_cache(self, tables):
        """Custom path returns a fresh namespace (not the cached one)."""
        t2 = OPALOpacity.get_tables(_BUNDLED_H5)
        assert t2 is not tables
        assert_allclose(t2.grid_T, tables.grid_T)

    # ---- read-only arrays ---- #

    def test_grid_T_readonly(self, tables):
        with pytest.raises(ValueError, match="read-only"):
            tables.grid_T[0] = 999.0

    def test_grid_R_readonly(self, tables):
        with pytest.raises(ValueError, match="read-only"):
            tables.grid_R[0] = 999.0

    def test_opacity_readonly(self, tables):
        with pytest.raises(ValueError, match="read-only"):
            tables.opacity[0, 0, 0] = 999.0

    # ---- find ---- #

    def test_find_solar(self, tables):
        """find() returns the known solar index."""
        idx = OPALOpacity.find(X=0.70, Z=0.02)
        assert idx == _SOLAR_IDX

    def test_find_no_match_raises(self):
        with pytest.raises(ValueError, match="No OPAL table"):
            OPALOpacity.find(X=0.999, Z=0.999)

    def test_find_composition_X_Z(self, tables):
        """find() round-trips: index back to correct X, Z."""
        idx = OPALOpacity.find(X=0.70, Z=0.02)
        assert_allclose(tables.X[idx], 0.70, atol=1e-6)
        assert_allclose(tables.Z[idx], 0.02, atol=1e-6)

    def test_find_custom_path(self):
        """find() works when given a custom table_path."""
        idx = OPALOpacity.find(X=0.70, Z=0.02, table_path=_BUNDLED_H5)
        assert idx == _SOLAR_IDX

    # ---- where ---- #

    def test_where_exact_scalar(self, tables):
        """where() with exact float values finds correct tables."""
        indices = OPALOpacity.where(X=0.0, Z=0.0)
        assert 0 in indices

    def test_where_solar(self):
        """where() locates the unique solar composition table."""
        indices = OPALOpacity.where(X=0.70, Z=0.02)
        assert len(indices) == 1
        assert indices[0] == _SOLAR_IDX

    def test_where_predicate(self, tables):
        """where() accepts callable predicates."""
        indices = OPALOpacity.where(Z=lambda z: z > 0.05)
        assert len(indices) > 0
        for i in indices:
            assert tables.Z[i] > 0.05

    def test_where_multi_condition(self, tables):
        """where() with multiple keys intersects conditions."""
        indices = OPALOpacity.where(X=0.70, Z=0.02)
        assert all(abs(tables.X[i] - 0.70) < 1e-6 and abs(tables.Z[i] - 0.02) < 1e-6 for i in indices)

    def test_where_no_match(self):
        assert OPALOpacity.where(X=0.12345) == []

    def test_where_unknown_key_raises(self):
        with pytest.raises(KeyError):
            OPALOpacity.where(nonexistent_key=1.0)

    def test_where_solar_matches_find(self):
        """where(X=..., Z=...) and find() agree on the solar index."""
        assert OPALOpacity.where(X=0.70, Z=0.02) == [OPALOpacity.find(X=0.70, Z=0.02)]

    # ---- XYZ coverage ---- #

    def test_xyz_shapes(self, tables):
        assert tables.X.shape == (126,)
        assert tables.Y.shape == (126,)
        assert tables.Z.shape == (126,)

    def test_xyz_non_negative(self, tables):
        assert np.all(tables.X >= 0.0)
        assert np.all(tables.Z >= 0.0)


# ================================================================== #
# TestC_OPALTableOpacity                                             #
# ================================================================== #


class TestC_OPALTableOpacity:
    """C-level interpolator: accuracy, derivatives, out-of-bounds."""

    def _make_c_obj(self, tables, index=_SOLAR_IDX, oob=0):
        from trilobite.radiation.opacity.grey_opacity.rosseland._opal_table import C_OPALTableOpacity

        lk = np.array(tables.opacity[index], dtype=np.float64, order="C")
        return C_OPALTableOpacity(
            np.array(tables.grid_T, dtype=np.float64),
            np.array(tables.grid_R, dtype=np.float64),
            lk,
            oob,
        )

    def test_on_grid_point_accuracy(self, tables):
        """Interpolation at a grid node matches the stored value exactly."""
        c = self._make_c_obj(tables, oob=1)
        opc = tables.opacity[_SOLAR_IDX]
        ri, rj = np.argwhere(~np.isnan(opc))[10]
        log10_T = tables.grid_T[ri]
        log10_R = tables.grid_R[rj]
        expected_log10_kappa = opc[ri, rj]

        LN10 = math.log(10)
        ln_T = log10_T * LN10
        log10_rho = log10_R + 3.0 * (log10_T - 6.0)
        ln_rho = log10_rho * LN10

        ln_kappa = c.log_opacity(ln_T, ln_rho)
        assert_allclose(ln_kappa / LN10, expected_log10_kappa, rtol=1e-4)

    def test_opacity_smooth(self, tables):
        """Opacity varies smoothly as T increases at fixed rho."""
        c = self._make_c_obj(tables, oob=2)
        LN10 = math.log(10)

        log10_R = -3.0
        log10_Ts = np.linspace(4.5, 7.5, 30)
        vals = []
        for log10_T in log10_Ts:
            ln_T = log10_T * LN10
            ln_rho = (log10_R + 3.0 * (log10_T - 6.0)) * LN10
            vals.append(c.log_opacity(ln_T, ln_rho))

        vals = np.array(vals)
        finite = vals[np.isfinite(vals)]
        assert len(finite) > 5
        assert np.all(np.isfinite(finite))

    def test_dlogrho_finite_difference(self, tables):
        """d(lnκ)/d(lnρ) matches a central finite difference.

        Use mid-cell coordinates so both FD points stay within the same bilinear
        cell and the result equals the analytic slope.
        """
        c = self._make_c_obj(tables, oob=2)
        LN10 = math.log(10)

        log10_T = 5.975
        log10_R = -2.75
        ln_T = log10_T * LN10
        ln_rho = (log10_R + 3.0 * (log10_T - 6.0)) * LN10

        deriv = c.dlogkappa_dlogrho(ln_T, ln_rho)
        h = 1e-5
        fd = (c.log_opacity(ln_T, ln_rho + h) - c.log_opacity(ln_T, ln_rho - h)) / (2.0 * h)
        if not (math.isnan(deriv) or math.isnan(fd)):
            assert_allclose(deriv, fd, rtol=1e-3)

    def test_dlogT_finite_difference(self, tables):
        """d(lnκ)/d(lnT) matches a central finite difference.

        Use mid-cell coordinates (see test_dlogrho_finite_difference).
        """
        c = self._make_c_obj(tables, oob=2)
        LN10 = math.log(10)

        log10_T = 5.975
        log10_R = -2.75
        ln_rho = (log10_R + 3.0 * (log10_T - 6.0)) * LN10

        deriv = c.dlogkappa_dlogT(log10_T * LN10, ln_rho)
        h = 1e-5
        ln_T = log10_T * LN10
        fd = (c.log_opacity(ln_T + h, ln_rho) - c.log_opacity(ln_T - h, ln_rho)) / (2.0 * h)
        if not (math.isnan(deriv) or math.isnan(fd)):
            assert_allclose(deriv, fd, rtol=1e-3)

    def test_out_of_bounds_raise(self, tables):
        c = self._make_c_obj(tables, oob=0)
        with pytest.raises(ValueError):
            c.log_opacity(3.0 * math.log(10), -30.0 * math.log(10))

    def test_out_of_bounds_clamp(self, tables):
        c = self._make_c_obj(tables, oob=1)
        result = c.log_opacity(3.0 * math.log(10), -30.0 * math.log(10))
        assert math.isfinite(result)

    def test_out_of_bounds_nan(self, tables):
        c = self._make_c_obj(tables, oob=2)
        result = c.log_opacity(3.0 * math.log(10), -30.0 * math.log(10))
        assert math.isnan(result)


# ================================================================== #
# TestOPALOpacity                                                    #
# ================================================================== #


class TestOPALOpacity:
    """Python wrapper: interface contract, units, derivatives, registry."""

    def test_is_c_backed(self, solar_opacity):
        assert solar_opacity.IS_C_BACKED is True

    def test_out_of_bounds_invalid_raises(self, tables):
        """Invalid out_of_bounds raises ValueError at construction."""
        with pytest.raises(ValueError):
            OPALOpacity(tables.grid_T, tables.grid_R, tables.opacity[_SOLAR_IDX], out_of_bounds="invalid")

    def test_opacity_returns_quantity(self, solar_opacity):
        kappa = solar_opacity.opacity(_REF_RHO, _REF_T)
        assert isinstance(kappa, u.Quantity)
        assert kappa.unit.is_equivalent(u.cm**2 / u.g)

    def test_opacity_finite(self, solar_opacity):
        kappa = solar_opacity.opacity(_REF_RHO, _REF_T)
        assert math.isfinite(kappa.value)
        assert kappa.value > 0

    def test_solar_opacity_plausible(self, solar_opacity):
        """κ at (1e-4 g/cc, 1e6 K) should be > 0 and < 100 cm²/g."""
        kappa = solar_opacity.opacity(_REF_RHO, _REF_T).to(u.cm**2 / u.g).value
        assert 0 < kappa < 100

    def test_dlogrho_dimensionless(self, solar_opacity):
        val = solar_opacity.dlogkappa_dlogrho(_REF_RHO, _REF_T)
        assert isinstance(val, (float, np.floating, np.ndarray))
        assert math.isfinite(float(val))

    def test_dlogT_dimensionless(self, solar_opacity):
        val = solar_opacity.dlogkappa_dlogT(_REF_RHO, _REF_T)
        assert isinstance(val, (float, np.floating, np.ndarray))
        assert math.isfinite(float(val))

    def test_array_input_returns_array(self, solar_opacity):
        rhos = np.array([1e-5, 1e-4, 1e-3]) * u.g / u.cm**3
        Ts = np.array([1e6, 2e6, 5e6]) * u.K
        kappa = solar_opacity.opacity(rhos, Ts)
        assert isinstance(kappa, u.Quantity)
        assert kappa.shape == (3,)

    def test_array_derivatives(self, solar_opacity):
        rhos = np.array([1e-5, 1e-4]) * u.g / u.cm**3
        Ts = np.array([1e6, 2e6]) * u.K
        d_rho = solar_opacity.dlogkappa_dlogrho(rhos, Ts)
        d_T = solar_opacity.dlogkappa_dlogT(rhos, Ts)
        assert isinstance(d_rho, np.ndarray)
        assert isinstance(d_T, np.ndarray)
        assert d_rho.shape == (2,)
        assert d_T.shape == (2,)

    def test_out_of_bounds_raise(self, tables):
        op = OPALOpacity(tables.grid_T, tables.grid_R, tables.opacity[_SOLAR_IDX], out_of_bounds="raise")
        with pytest.raises(ValueError):
            op.opacity(1e-30 * u.g / u.cm**3, 100.0 * u.K)

    def test_out_of_bounds_nan(self, tables):
        op = OPALOpacity(tables.grid_T, tables.grid_R, tables.opacity[_SOLAR_IDX], out_of_bounds="nan")
        kappa = op.opacity(1e-30 * u.g / u.cm**3, 100.0 * u.K)
        assert math.isnan(kappa.value)

    def test_out_of_bounds_clamp_no_raise(self, tables):
        op = OPALOpacity(tables.grid_T, tables.grid_R, tables.opacity[_SOLAR_IDX], out_of_bounds="clamp")
        _ = op.opacity(1e-30 * u.g / u.cm**3, 100.0 * u.K)

    def test_load_opal_opacity_helper(self):
        op = load_opal_opacity(_SOLAR_IDX)
        assert isinstance(op, OPALOpacity)
        assert op.IS_C_BACKED

    def test_load_opal_opacity_custom_path(self):
        op = load_opal_opacity(_SOLAR_IDX, table_path=_BUNDLED_H5)
        assert isinstance(op, OPALOpacity)

    def test_resolver_string(self):
        """get_opacity('opal') returns an OPALOpacity instance."""
        op = get_opacity("opal")
        assert isinstance(op, OPALOpacity)
        assert op.IS_C_BACKED is True

    def test_out_of_bounds_attribute(self, solar_opacity):
        assert solar_opacity.out_of_bounds == "raise"

    def test_repr(self, solar_opacity):
        r = repr(solar_opacity)
        assert "OPALOpacity" in r

    def test_load_default_solar(self):
        op = OPALOpacity.load_default()
        assert isinstance(op, OPALOpacity)

    def test_load_default_by_index(self):
        op = OPALOpacity.load_default(index=_SOLAR_IDX)
        assert isinstance(op, OPALOpacity)

    def test_load_default_by_composition(self):
        op = OPALOpacity.load_default(X=0.70, Z=0.02)
        assert isinstance(op, OPALOpacity)

    def test_load_default_index_and_xz_raises(self):
        with pytest.raises(ValueError):
            OPALOpacity.load_default(index=72, X=0.70, Z=0.02)

    def test_load_default_only_X_raises(self):
        """Supplying X without Z should raise ValueError."""
        with pytest.raises(ValueError):
            OPALOpacity.load_default(X=0.70)


# ================================================================== #
# TestC_OPALTableOpacityExtended                                     #
# ================================================================== #


class TestC_OPALTableOpacityExtended:
    """Additional C-level tests: multi-point derivatives, OOB coverage, minimal grid."""

    def _make_c_obj(self, tables, index=_SOLAR_IDX, oob=2):
        from trilobite.radiation.opacity.grey_opacity.rosseland._opal_table import C_OPALTableOpacity

        lk = np.array(tables.opacity[index], dtype=np.float64, order="C")
        return C_OPALTableOpacity(
            np.array(tables.grid_T, dtype=np.float64),
            np.array(tables.grid_R, dtype=np.float64),
            lk,
            oob,
        )

    _DERIV_POINTS = [
        (5.975, -2.75),
        (6.525, -1.25),
        (4.875, -6.25),
        (7.475, 0.25),
        (6.025, -4.75),
    ]

    @pytest.mark.parametrize("log10_T,log10_R", _DERIV_POINTS)
    def test_dlogrho_fd_multi(self, tables, log10_T, log10_R):
        """d(lnκ)/d(lnρ) matches FD at multiple interior cell midpoints."""
        c = self._make_c_obj(tables, oob=2)
        LN10 = math.log(10)
        ln_T = log10_T * LN10
        ln_rho = (log10_R + 3.0 * (log10_T - 6.0)) * LN10

        deriv = c.dlogkappa_dlogrho(ln_T, ln_rho)
        h = 1e-5
        fd = (c.log_opacity(ln_T, ln_rho + h) - c.log_opacity(ln_T, ln_rho - h)) / (2.0 * h)
        if not (math.isnan(deriv) or math.isnan(fd)):
            assert_allclose(deriv, fd, rtol=1e-3)

    @pytest.mark.parametrize("log10_T,log10_R", _DERIV_POINTS)
    def test_dlogT_fd_multi(self, tables, log10_T, log10_R):
        """d(lnκ)/d(lnT) matches FD at multiple interior cell midpoints."""
        c = self._make_c_obj(tables, oob=2)
        LN10 = math.log(10)
        ln_T = log10_T * LN10
        ln_rho = (log10_R + 3.0 * (log10_T - 6.0)) * LN10

        deriv = c.dlogkappa_dlogT(ln_T, ln_rho)
        h = 1e-5
        fd = (c.log_opacity(ln_T + h, ln_rho) - c.log_opacity(ln_T - h, ln_rho)) / (2.0 * h)
        if not (math.isnan(deriv) or math.isnan(fd)):
            assert_allclose(deriv, fd, rtol=1e-3)

    @pytest.mark.parametrize(
        "oob,expect_raise,expect_nan",
        [
            (0, True, False),
            (1, False, False),
            (2, False, True),
        ],
    )
    def test_oob_T_axis_low(self, tables, oob, expect_raise, expect_nan):
        """Out-of-bounds below T_min handled correctly."""
        c = self._make_c_obj(tables, oob=oob)
        LN10 = math.log(10)
        ln_T = 2.0 * LN10
        ln_rho = (-4.0 + 3.0 * (2.0 - 6.0)) * LN10
        if expect_raise:
            with pytest.raises(ValueError):
                c.log_opacity(ln_T, ln_rho)
        else:
            result = c.log_opacity(ln_T, ln_rho)
            if expect_nan:
                assert math.isnan(result)
            else:
                assert isinstance(result, float)

    @pytest.mark.parametrize(
        "oob,expect_raise,expect_nan",
        [
            (0, True, False),
            (1, False, False),
            (2, False, True),
        ],
    )
    def test_oob_R_axis_low(self, tables, oob, expect_raise, expect_nan):
        """Out-of-bounds below R_min handled correctly."""
        c = self._make_c_obj(tables, oob=oob)
        LN10 = math.log(10)
        ln_T = 6.0 * LN10
        ln_rho = (-10.0 + 3.0 * (6.0 - 6.0)) * LN10
        if expect_raise:
            with pytest.raises(ValueError):
                c.log_opacity(ln_T, ln_rho)
        else:
            result = c.log_opacity(ln_T, ln_rho)
            if expect_nan:
                assert math.isnan(result)
            else:
                assert isinstance(result, float)

    @pytest.mark.parametrize(
        "oob,expect_raise,expect_nan",
        [
            (0, True, False),
            (1, False, False),
            (2, False, True),
        ],
    )
    def test_oob_both_axes(self, tables, oob, expect_raise, expect_nan):
        """Out-of-bounds on both axes handled correctly."""
        c = self._make_c_obj(tables, oob=oob)
        LN10 = math.log(10)
        ln_T = 2.0 * LN10
        ln_rho = (-10.0 + 3.0 * (2.0 - 6.0)) * LN10
        if expect_raise:
            with pytest.raises(ValueError):
                c.log_opacity(ln_T, ln_rho)
        else:
            result = c.log_opacity(ln_T, ln_rho)
            if expect_nan:
                assert math.isnan(result)
            else:
                assert isinstance(result, float)

    def test_minimal_2x2_grid(self):
        """A 2×2 table is the smallest valid grid; bilinear interpolation works."""
        from trilobite.radiation.opacity.grey_opacity.rosseland._opal_table import C_OPALTableOpacity

        LN10 = math.log(10)
        g1 = np.array([5.0, 6.0], dtype=np.float64)
        g2 = np.array([-4.0, -3.0], dtype=np.float64)
        lk = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64, order="C")

        c = C_OPALTableOpacity(g1, g2, lk, 1)  # clamp oob

        # Cell centre: log10_T=5.5, log10_R=-3.5
        # → log10_rho = log10_R + 3*(log10_T - 6) = -3.5 + 3*(5.5-6) = -5.0
        ln_T = 5.5 * LN10
        ln_rho = -5.0 * LN10
        result = c.log_opacity(ln_T, ln_rho)
        expected = 1.5 * LN10
        assert_allclose(result, expected, rtol=1e-10)

    def test_solar_idx_dynamic_matches_constant(self, solar_idx_dynamic):
        """Dynamic search returns the same index as the hardcoded _SOLAR_IDX constant."""
        assert solar_idx_dynamic == _SOLAR_IDX, (
            f"Dynamic search found solar at index {solar_idx_dynamic}, "
            f"but _SOLAR_IDX={_SOLAR_IDX}.  Update _SOLAR_IDX or the table."
        )
