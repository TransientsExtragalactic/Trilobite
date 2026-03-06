import numpy as np

from tests.test_models.test_model_base import BaseModelTest
from triceratops.models import (
    Cooling_SynchrotronSEDModel,
    SSA_Cooling_SynchrotronSEDModel,
    SSA_SynchrotronSEDModel,
    SynchrotronSEDModel,
)


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
