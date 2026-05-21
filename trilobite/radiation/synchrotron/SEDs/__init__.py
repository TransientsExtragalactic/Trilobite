"""
Synchrotron spectral energy distribution models.

This module implements the various spectral energy distributions relevant to synchrotron emission from transients.
For documentation on synchrotron emission modeling in Trilobite, see :ref:`radiation_overview`.
"""

__all__ = ["one_zone", "one_zone_closure"]
from . import one_zone, one_zone_closure
from .one_zone import *
from .one_zone_closure import *

__all__ += one_zone.__all__
__all__ += one_zone_closure.__all__
