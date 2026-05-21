"""
Framework compliance tests for GRB models.

All concrete ``Model`` subclasses in Trilobite **must**
be accompanied by a compliance test inheriting from
``BaseModelTest``. This ensures that every model satisfies
the core framework contract:

- The model initializes successfully.
- The forward evaluation executes without error.
- The returned outputs match the declared ``OUTPUTS``.
- Units are handled consistently and correctly.
- Optional diagnostic plots can be generated.

This test module may contain multiple model-specific
test classes, each inheriting from ``BaseModelTest``.

No new model implementation should be merged into the
codebase without a corresponding compliance test here.
"""

import numpy as np
import pytest
from astropy import units as u
from test_model_base import BaseModelTest

from trilobite.models.GRBs.band import BandFunctionModel


class TestBandFunctionModel(BaseModelTest):
    """
    Structural, diagnostic, and physics tests for BandFunctionModel.
    """

    MODEL = BandFunctionModel

    # Log–log spectrum plotting
    LOG_X = True
    LOG_Y = True

    VARIABLES = {
        "energy": np.logspace(0, 4, 500) * u.keV,  # 1 – 10 000 keV
    }

    PARAMETERS = {
        "alpha": -1.0,
        "beta": -2.5,
        "E_peak": 300.0 * u.keV,
        "E_piv": 100.0 * u.keV,
        "A": 1e-3 * u.Unit("1/(cm**2 s keV)"),
    }

    # ------------------------------------------------------------------ #
    # Band-specific physics tests                                          #
    # ------------------------------------------------------------------ #

    def test_band_continuity_at_break(self):
        """Photon flux must be continuous across the break energy E_c."""
        model = self.MODEL()
        alpha, beta = -1.0, -2.5
        E_peak = 300.0  # keV
        E_c = (alpha - beta) / (2.0 + alpha) * E_peak

        eps = 1e-6 * E_c  # tiny offset on either side of E_c

        result_low = model(
            {"energy": (E_c - eps) * u.keV},
            {"alpha": alpha, "beta": beta, "E_peak": E_peak * u.keV, "E_piv": 100.0 * u.keV, "A": 1.0},
        )
        result_high = model(
            {"energy": (E_c + eps) * u.keV},
            {"alpha": alpha, "beta": beta, "E_peak": E_peak * u.keV, "E_piv": 100.0 * u.keV, "A": 1.0},
        )

        np.testing.assert_allclose(
            result_low.photon_flux.value,
            result_high.photon_flux.value,
            rtol=1e-4,
            err_msg="Band function is not continuous at the break energy E_c.",
        )

    def test_band_low_energy_slope(self):
        """In the low-energy regime, the log-slope must match alpha."""
        model = self.MODEL()
        alpha, beta = -0.5, -2.5
        E_c_val = (alpha - beta) / (2.0 + alpha) * 300.0  # keV

        # Evaluate deep below E_c so the exponential factor is effectively 1.
        E_lo = np.array([1e-4, 2e-4]) * E_c_val * u.keV

        result = model(
            {"energy": E_lo},
            {"alpha": alpha, "beta": beta, "E_peak": 300.0 * u.keV, "E_piv": 100.0 * u.keV, "A": 1.0},
        )

        flux = result.photon_flux.value
        measured_slope = np.log(flux[1] / flux[0]) / np.log((E_lo[1] / E_lo[0]).value)
        np.testing.assert_allclose(measured_slope, alpha, rtol=0.01)

    def test_band_high_energy_slope(self):
        """In the high-energy regime, the log-slope must match beta."""
        model = self.MODEL()
        alpha, beta = -1.0, -2.5
        E_c_val = (alpha - beta) / (2.0 + alpha) * 300.0  # keV

        # Evaluate well above E_c.
        E_hi = np.array([100.0, 200.0]) * E_c_val * u.keV

        result = model(
            {"energy": E_hi},
            {"alpha": alpha, "beta": beta, "E_peak": 300.0 * u.keV, "E_piv": 100.0 * u.keV, "A": 1.0},
        )

        flux = result.photon_flux.value
        measured_slope = np.log(flux[1] / flux[0]) / np.log((E_hi[1] / E_hi[0]).value)
        np.testing.assert_allclose(measured_slope, beta, rtol=0.01)

    def test_band_amplitude_scaling(self):
        """Doubling A must double the output photon flux everywhere."""
        model = self.MODEL()
        energies = np.logspace(1, 3, 50) * u.keV
        base = {"alpha": -1.0, "beta": -2.5, "E_peak": 300.0 * u.keV, "E_piv": 100.0 * u.keV}

        r1 = model({"energy": energies}, {**base, "A": 1.0})
        r2 = model({"energy": energies}, {**base, "A": 2.0})

        np.testing.assert_allclose(r2.photon_flux.value, 2.0 * r1.photon_flux.value, rtol=1e-10)

    def test_band_alpha_beta_constraint(self):
        """alpha <= beta must raise ValueError (E_c would be non-positive).

        Uses alpha=-1.0 and beta=-0.5 so that alpha > -2 (valid individual
        bound) but alpha < beta (invalid model-level constraint).
        """
        model = self.MODEL()
        with pytest.raises(ValueError, match="alpha > beta"):
            model(
                {"energy": np.array([100.0]) * u.keV},
                {"alpha": -1.0, "beta": -0.5, "E_peak": 300.0 * u.keV, "E_piv": 100.0 * u.keV, "A": 1e-3},
            )

    def test_band_alpha_below_minus2_constraint(self):
        """alpha <= -2 must raise ValueError (unphysical exponential cut-off)."""
        model = self.MODEL()
        with pytest.raises(ValueError, match="alpha > -2"):
            model(
                {"energy": np.array([100.0]) * u.keV},
                {"alpha": -2.0, "beta": -3.0, "E_peak": 300.0 * u.keV, "E_piv": 100.0 * u.keV, "A": 1e-3},
            )
