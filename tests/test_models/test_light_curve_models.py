"""
Framework compliance tests for physical models.

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

from trilobite.models.generic.light_curve import (
    FRED,
    BrokenPowerLawTime,
    ExponentialRisePowerLawDecay,
    GaussianPulse,
    GeneralizedFRED,
    LogisticPulse,
    LogNormalPulse,
    NorrisPulse,
    SmoothedBrokenPowerLawTime,
    WeibullPulse,
)

# ============================================================
# Shared Inputs
# ============================================================

_TIME_GRID = np.linspace(0.1, 10.0, 500) * u.s


# ============================================================
# Individual Model Tests
# ============================================================


class TestFRED(BaseModelTest):
    MODEL = FRED
    VARIABLES = {"t": _TIME_GRID}
    PARAMETERS = {}
    LOG_X = True


class TestGeneralizedFRED(BaseModelTest):
    MODEL = GeneralizedFRED
    VARIABLES = {"t": _TIME_GRID}
    PARAMETERS = {}
    LOG_X = True


class TestGaussianPulse(BaseModelTest):
    MODEL = GaussianPulse
    VARIABLES = {"t": _TIME_GRID}
    PARAMETERS = {}
    LOG_X = True


class TestLogNormalPulse(BaseModelTest):
    MODEL = LogNormalPulse
    VARIABLES = {"t": _TIME_GRID}
    PARAMETERS = {}
    LOG_X = True


class TestBrokenPowerLawTime(BaseModelTest):
    MODEL = BrokenPowerLawTime
    VARIABLES = {"t": _TIME_GRID}
    PARAMETERS = {}
    LOG_X = True


class TestSmoothedBrokenPowerLawTime(BaseModelTest):
    MODEL = SmoothedBrokenPowerLawTime
    VARIABLES = {"t": _TIME_GRID}
    PARAMETERS = {}
    LOG_X = True


class TestSmoothedBrokenPowerLawTimeShape:
    """Break normalization and convexity of the smoothed temporal break."""

    @staticmethod
    def _flux(t, **params):
        flux = SmoothedBrokenPowerLawTime().forward_model({"t": np.asarray(t, dtype=float)}, params).flux
        return np.asarray(getattr(flux, "value", flux), dtype=float)

    @pytest.mark.parametrize("s", [-1.0, -0.3, 0.3, 1.0])
    def test_flux_at_break_equals_A(self, s):
        assert self._flux([5.0], A=2.5, t_b=5.0, alpha_1=2.0, alpha_2=-1.0, s=s, C=0.0)[0] == pytest.approx(2.5)

    def test_negative_s_gives_rise_then_fall(self):
        t = np.array([1e-3, 1e-2, 1e2, 1e3])
        F = self._flux(t, A=1.0, t_b=1.0, alpha_1=2.0, alpha_2=-1.0, s=-0.1, C=0.0)
        slopes = np.diff(np.log10(F)) / np.diff(np.log10(t))
        assert slopes[0] == pytest.approx(2.0, abs=1e-2)
        assert slopes[-1] == pytest.approx(-1.0, abs=1e-2)
        assert F.max() <= 1.0 + 1e-12

    def test_positive_s_gives_convex_break(self):
        t = np.array([1e-3, 1e-2, 1e2, 1e3])
        F = self._flux(t, A=1.0, t_b=1.0, alpha_1=-2.0, alpha_2=-0.5, s=0.1, C=0.0)
        slopes = np.diff(np.log10(F)) / np.diff(np.log10(t))
        assert slopes[0] == pytest.approx(-2.0, abs=1e-2)
        assert slopes[-1] == pytest.approx(-0.5, abs=1e-2)

    def test_small_smoothing_does_not_overflow(self):
        F = self._flux([1e-3, 1e3], A=1.0, t_b=1.0, alpha_1=3.0, alpha_2=-3.0, s=-1e-3, C=0.0)
        assert np.all(np.isfinite(F))

    def test_zero_s_raises(self):
        with pytest.raises(ValueError):
            self._flux([1.0], A=1.0, t_b=1.0, alpha_1=2.0, alpha_2=-1.0, s=0.0, C=0.0)


class TestExponentialRisePowerLawDecay(BaseModelTest):
    MODEL = ExponentialRisePowerLawDecay
    VARIABLES = {"t": _TIME_GRID}
    PARAMETERS = {}
    LOG_X = True


class TestNorrisPulse(BaseModelTest):
    MODEL = NorrisPulse
    VARIABLES = {"t": _TIME_GRID}
    PARAMETERS = {}
    LOG_X = True


class TestWeibullPulse(BaseModelTest):
    MODEL = WeibullPulse
    VARIABLES = {"t": _TIME_GRID}
    PARAMETERS = {}
    LOG_X = True


class TestLogisticPulse(BaseModelTest):
    MODEL = LogisticPulse
    VARIABLES = {"t": _TIME_GRID}
    PARAMETERS = {}
    LOG_X = True
