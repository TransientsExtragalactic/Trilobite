"""
Framework compliance tests for physical models.

All concrete ``Model`` subclasses in Triceratops **must**
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

from triceratops.models.generic.curves import (
    ConstantModel,
    ExponentialModel,
    LinearModel,
    PowerLawModel,
    QuadraticModel,
)

# ============================================================
# Shared Inputs
# ============================================================

_TIME_GRID = np.linspace(0.1, 10.0, 500) * u.s


# ============================================================
# Individual Model Tests
# ============================================================
class TestConstantModel(BaseModelTest):
    MODEL = ConstantModel
    VARIABLES = {"x": np.linspace(0, 1, 100)}
    PARAMETERS = {}
    LOG_X = False
    LOG_Y = False


class TestLinearModel(BaseModelTest):
    MODEL = LinearModel
    VARIABLES = {"x": np.linspace(0, 1, 100)}
    PARAMETERS = {"m": 2.0, "b": 1.0}
    LOG_X = False
    LOG_Y = False


class TestQuadraticModel(BaseModelTest):
    MODEL = QuadraticModel
    VARIABLES = {"x": np.linspace(0, 1, 100)}
    PARAMETERS = {"a": 1.0, "b": 0.0, "c": 0.0}
    LOG_X = False
    LOG_Y = False


class TestPowerLawModel(BaseModelTest):
    MODEL = PowerLawModel
    VARIABLES = {"x": np.linspace(0.1, 1, 100)}
    PARAMETERS = {"A": 1.0, "alpha": 2.0}
    LOG_X = True
    LOG_Y = True


class TestExponentialModel(BaseModelTest):
    MODEL = ExponentialModel
    VARIABLES = {"x": np.linspace(0, 1, 100)}
    PARAMETERS = {"A": 1.0, "k": 1.0}
    LOG_X = False
    LOG_Y = False
