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

from triceratops.models.SEDs.evolving_phenom import PL_Evolving_SSA_SED_Model
from triceratops.models.SEDs.synchrotron import (
    Cooling_SynchrotronSEDModel,
    SSA_Cooling_SynchrotronSEDModel,
    SSA_SynchrotronSEDModel,
    SynchrotronSEDModel,
)


# ============================================================
# Individual Model Tests
# ============================================================
class TestPL_Evolving_SSA_SED_Model(BaseModelTest):
    MODEL = PL_Evolving_SSA_SED_Model
    VARIABLES = {"frequency": np.logspace(8, 11, 200) * u.Hz, "time": 10 * u.day}
    PARAMETERS = {
        "alpha_1": 5 / 2,  # self-absorbed slope
        "alpha_2": -1.0,  # optically thin slope
        "beta": 1.0,  # nu_brk evolution
        "gamma": 0.0,  # F_brk evolution
        "nu_brk_0": 1e10 * u.Hz,
        "F_brk_0": 1.0 * u.Jy,
        "t_0": 10 * u.day,
        "s": 0.3,
    }
    LOG_X = True
    LOG_Y = True


class TestSSA_Cooling_SynchrotronSEDModel(BaseModelTest):
    MODEL = SSA_Cooling_SynchrotronSEDModel
    VARIABLES = {"log_nu": np.linspace(8, 11, 200)}
    PARAMETERS = {}
    LOG_X = True
    LOG_Y = True


class TestSynchrotronSEDModel(BaseModelTest):
    MODEL = SynchrotronSEDModel
    VARIABLES = {"log_nu": np.linspace(8, 11, 200)}
    PARAMETERS = {}
    LOG_X = True
    LOG_Y = True


class TestSSA_SynchrotronSEDModel(BaseModelTest):
    MODEL = SSA_SynchrotronSEDModel
    VARIABLES = {"log_nu": np.linspace(8, 11, 200)}
    PARAMETERS = {}
    LOG_X = True
    LOG_Y = True


class TestCooling_SynchrotronSEDModel(BaseModelTest):
    MODEL = Cooling_SynchrotronSEDModel
    VARIABLES = {"log_nu": np.linspace(8, 11, 200)}
    PARAMETERS = {}
    LOG_X = True
    LOG_Y = True
