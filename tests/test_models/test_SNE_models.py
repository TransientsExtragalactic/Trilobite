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

from trilobite.models import ChevalierShockModel


class TestChevalierShockModel(BaseModelTest):
    """
    Structural and diagnostic tests for ChevalierShockModel.
    """

    MODEL = ChevalierShockModel

    # Log–log spectrum plotting
    LOG_X = True
    LOG_Y = True

    VARIABLES = {
        "frequency": np.logspace(8, 12, 300) * u.Hz,
        "time": 1e6 * u.s,
    }

    PARAMETERS = {
        "E_ej": 1e51 * u.erg,
        "M_ej": 1.4 * u.Msun,
        "rho_0": 1e-18 * u.g / u.cm**3,
    }
