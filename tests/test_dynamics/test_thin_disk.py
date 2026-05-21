"""
Unit tests for the Sunyaev-Shakura alpha-disk model.

Tests cover:
- Output keys and units from ``AlphaDisk.compute()``
- Radial power-law slopes of the SS73 scalings (Σ, T_c, H, ρ)
- Alpha-parameter scaling of Σ
- Analytic bolometric luminosity
- Multi-colour blackbody SED (positivity, Wien peak)
- ``compute_effective_temperature`` monotonicity
- ``__call__`` aliasing
"""

import numpy as np
import pytest
from astropy import constants as const
from astropy import units as u
from numpy.testing import assert_allclose

from trilobite.dynamics.accretion import AlphaDisk


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


class TestAlphaDisk:
    """Tests for :class:`~trilobite.dynamics.accretion.AlphaDisk`."""

    @pytest.fixture(scope="class")
    def disk(self):
        """Fiducial AlphaDisk with alpha = 0.1."""
        return AlphaDisk(alpha=0.1)

    @pytest.fixture(scope="class")
    def fiducial_params(self):
        """Fiducial physical parameters for a stellar-mass BH disk."""
        return {
            "M_BH": 10.0 * const.M_sun,
            "mdot": 1e16 * u.g / u.s,
            "R_in": 3.0e6 * u.cm,  # ~ 6 Schwarzschild radii for 10 M_sun BH
        }

    @pytest.fixture(scope="class")
    def radii_far(self, fiducial_params):
        """Log-spaced radius grid well outside R_in (f → 1 regime)."""
        R_in_cm = fiducial_params["R_in"].to_value(u.cm)
        return np.geomspace(100.0 * R_in_cm, 1e4 * R_in_cm, 200) * u.cm

    @pytest.fixture(scope="class")
    def result(self, disk, fiducial_params, radii_far):
        """Pre-computed disk structure on the far-field radius grid."""
        return disk.compute(radii_far, **fiducial_params)

    # -----------------------------------------------------------------------
    # Output contract
    # -----------------------------------------------------------------------

    def test_compute_keys(self, result):
        """compute() must return all seven disk-structure quantities."""
        expected = {"Sigma", "T_c", "H", "rho", "tau", "nu", "v_R"}
        assert set(result.keys()) == expected

    def test_compute_units_sigma(self, result):
        assert result["Sigma"].unit.is_equivalent(u.g / u.cm**2)

    def test_compute_units_T_c(self, result):
        assert result["T_c"].unit.is_equivalent(u.K)

    def test_compute_units_H(self, result):
        assert result["H"].unit.is_equivalent(u.cm)

    def test_compute_units_rho(self, result):
        assert result["rho"].unit.is_equivalent(u.g / u.cm**3)

    def test_compute_units_nu(self, result):
        assert result["nu"].unit.is_equivalent(u.cm**2 / u.s)

    def test_compute_units_v_R(self, result):
        assert result["v_R"].unit.is_equivalent(u.cm / u.s)

    def test_compute_tau_dimensionless(self, result):
        """tau is dimensionless (plain ndarray, no astropy unit)."""
        assert isinstance(result["tau"], np.ndarray)

    def test_all_positive(self, result):
        """All quantities must be positive on the far-field grid."""
        for key, val in result.items():
            arr = val.value if hasattr(val, "value") else val
            assert np.all(arr > 0), f"{key} has non-positive values"

    # -----------------------------------------------------------------------
    # Radial power-law slopes  (R >> R_in  ⟹  f → 1)
    # -----------------------------------------------------------------------

    def _log_slope(self, radii, values):
        """d log val / d log R via central differences (interior points only)."""
        log_R = np.log10(radii.to_value(u.cm))
        log_V = np.log10(values.value if hasattr(values, "value") else values)
        grad = np.gradient(log_V, log_R)
        # Use middle 50% of the grid to avoid boundary effects
        lo, hi = len(grad) // 4, 3 * len(grad) // 4
        return np.median(grad[lo:hi])

    def test_sigma_radial_slope(self, radii_far, result):
        """Σ ∝ R^{-3/4}  (SS73 scaling)."""
        slope = self._log_slope(radii_far, result["Sigma"])
        assert_allclose(slope, -0.75, rtol=0.02, err_msg="Σ radial slope")

    def test_T_c_radial_slope(self, radii_far, result):
        """T_c ∝ R^{-3/4}  (SS73 scaling)."""
        slope = self._log_slope(radii_far, result["T_c"])
        assert_allclose(slope, -0.75, rtol=0.02, err_msg="T_c radial slope")

    def test_H_radial_slope(self, radii_far, result):
        """H ∝ R^{9/8}  (SS73 scaling)."""
        slope = self._log_slope(radii_far, result["H"])
        assert_allclose(slope, 9.0 / 8.0, rtol=0.02, err_msg="H radial slope")

    def test_rho_radial_slope(self, radii_far, result):
        """ρ ∝ R^{-15/8}  (SS73 scaling)."""
        slope = self._log_slope(radii_far, result["rho"])
        assert_allclose(slope, -15.0 / 8.0, rtol=0.02, err_msg="ρ radial slope")

    # -----------------------------------------------------------------------
    # Alpha-parameter scaling
    # -----------------------------------------------------------------------

    def test_alpha_scaling_sigma(self, fiducial_params, radii_far):
        """Σ ∝ α^{-4/5} — test two alpha values at the same radius."""
        alpha1, alpha2 = 0.1, 0.3
        R = radii_far[len(radii_far) // 2]  # single midpoint radius

        Sigma1 = AlphaDisk(alpha=alpha1).compute(R, **fiducial_params)["Sigma"].value
        Sigma2 = AlphaDisk(alpha=alpha2).compute(R, **fiducial_params)["Sigma"].value

        ratio_computed = Sigma1 / Sigma2
        ratio_expected = (alpha1 / alpha2) ** (-4.0 / 5.0)
        assert_allclose(ratio_computed, ratio_expected, rtol=1e-6, err_msg="Σ alpha scaling")

    # -----------------------------------------------------------------------
    # Bolometric luminosity
    # -----------------------------------------------------------------------

    def test_bolometric_luminosity_analytic(self, disk, fiducial_params):
        """L_bol should equal G M Ṁ / (2 R_in) within 0.1 %."""
        M_BH = fiducial_params["M_BH"]
        mdot = fiducial_params["mdot"]
        R_in = fiducial_params["R_in"]

        L_computed = disk.compute_bolometric_luminosity(M_BH, mdot, R_in).to_value(u.erg / u.s)
        L_expected = (const.G * M_BH * mdot / (2.0 * R_in)).to_value(u.erg / u.s)

        assert_allclose(L_computed, L_expected, rtol=1e-4, err_msg="L_bol analytic")

    # -----------------------------------------------------------------------
    # SED
    # -----------------------------------------------------------------------

    @pytest.fixture(scope="class")
    def nu_grid(self):
        return np.geomspace(1e12, 1e18, 200) * u.Hz

    @pytest.fixture(scope="class")
    def sed(self, disk, fiducial_params, nu_grid):
        return disk.compute_sed(
            nu_grid,
            fiducial_params["M_BH"],
            fiducial_params["mdot"],
            fiducial_params["R_in"],
            R_out=1e12 * u.cm,
            D_L=10.0 * u.kpc,
        )

    def test_sed_keys(self, sed):
        assert "L_nu" in sed
        assert "F_nu" in sed

    def test_sed_L_nu_positive(self, sed):
        assert np.all(sed["L_nu"].value > 0)

    def test_sed_F_nu_positive(self, sed):
        assert np.all(sed["F_nu"].value > 0)

    def test_sed_L_nu_units(self, sed):
        assert sed["L_nu"].unit.is_equivalent(u.erg / u.s / u.Hz)

    def test_sed_F_nu_units(self, sed):
        assert sed["F_nu"].unit.is_equivalent(u.erg / u.s / u.Hz / u.cm**2)

    def test_sed_no_DL_omits_F_nu(self, disk, fiducial_params, nu_grid):
        """If D_L is not given, F_nu must not be in the output."""
        sed = disk.compute_sed(
            nu_grid,
            fiducial_params["M_BH"],
            fiducial_params["mdot"],
            fiducial_params["R_in"],
            R_out=1e12 * u.cm,
        )
        assert "L_nu" in sed
        assert "F_nu" not in sed

    def test_sed_wien_peak_order(self, disk, fiducial_params, nu_grid):
        """
        The SED peak frequency should be in rough agreement with Wien's law
        evaluated at the peak effective temperature of the disk.

        The peak T_eff occurs near r ≈ (49/36) R_in ≈ 1.36 R_in.  We just
        check that the SED peak lies within 1 decade of the Wien estimate.
        """
        sed = disk.compute_sed(
            nu_grid,
            fiducial_params["M_BH"],
            fiducial_params["mdot"],
            fiducial_params["R_in"],
            R_out=1e12 * u.cm,
        )
        nu_peak_sed = nu_grid[np.argmax(sed["L_nu"].value)]

        # Wien peak of T_eff at r_peak = (49/36) * R_in
        r_peak = (49.0 / 36.0) * fiducial_params["R_in"]
        T_peak = disk.compute_effective_temperature(r_peak, **fiducial_params)
        nu_wien = (2.82 * const.k_B * T_peak / const.h).to(u.Hz)

        # The SED integrates many temperatures, so we allow an order-of-magnitude margin
        assert_allclose(
            np.log10(nu_peak_sed.to_value(u.Hz)),
            np.log10(nu_wien.to_value(u.Hz)),
            atol=1.0,
            err_msg="SED peak vs Wien peak",
        )

    # -----------------------------------------------------------------------
    # Effective temperature
    # -----------------------------------------------------------------------

    def test_effective_temperature_units(self, disk, fiducial_params, radii_far):
        T = disk.compute_effective_temperature(radii_far, **fiducial_params)
        assert T.unit.is_equivalent(u.K)

    def test_effective_temperature_peaks_near_isco(self, disk, fiducial_params):
        """T_eff peaks at r ≈ (49/36) R_in, i.e. close to but outside R_in."""
        R_in = fiducial_params["R_in"]
        r_grid = np.geomspace(1.01, 100.0, 500) * R_in.to_value(u.cm) * u.cm
        T = disk.compute_effective_temperature(r_grid, **fiducial_params)
        r_peak_idx = np.argmax(T.value)
        r_peak = r_grid[r_peak_idx]
        # Peak should be within a factor of 5 of the theoretical (49/36) R_in
        assert (r_peak / R_in).decompose().value < 5.0

    # -----------------------------------------------------------------------
    # Scalar input
    # -----------------------------------------------------------------------

    def test_scalar_radius(self, disk, fiducial_params):
        """compute() should work with a scalar (non-array) radius."""
        R = 1e10 * u.cm
        result = disk.compute(R, **fiducial_params)
        for key, val in result.items():
            arr = val.value if hasattr(val, "value") else val
            assert np.ndim(arr) == 0 or (np.ndim(arr) == 1 and len(arr) == 1)

    # -----------------------------------------------------------------------
    # __call__ alias
    # -----------------------------------------------------------------------

    def test_call_alias(self, disk, fiducial_params, radii_far):
        """disk(r, ...) should return identical results to disk.compute(r, ...)."""
        r1 = disk.compute(radii_far, **fiducial_params)
        r2 = disk(radii_far, **fiducial_params)
        for key in r1:
            v1 = r1[key].value if hasattr(r1[key], "value") else r1[key]
            v2 = r2[key].value if hasattr(r2[key], "value") else r2[key]
            assert_allclose(v1, v2, err_msg=f"__call__ vs compute for {key}")

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    def test_invalid_alpha_zero(self):
        with pytest.raises(ValueError):
            AlphaDisk(alpha=0.0)

    def test_invalid_alpha_gt1(self):
        with pytest.raises(ValueError):
            AlphaDisk(alpha=1.5)

    # -----------------------------------------------------------------------
    # String representation
    # -----------------------------------------------------------------------

    def test_str(self, disk):
        assert "AlphaDisk" in str(disk)
        assert "0.1" in str(disk)

    def test_repr(self, disk):
        assert "AlphaDisk" in repr(disk)
