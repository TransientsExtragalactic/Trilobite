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
