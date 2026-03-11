"""
Synchrotron spectral energy distribution models.

This module implements the various spectral energy distributions relevant to synchrotron emission from transients.
For documentation on synchrotron emission modeling in Triceratops, see :ref:`radiation_overview`.
"""

__all__ = [
    "PowerLaw_SynchrotronSED",
    "PowerLaw_Cooling_SSA_SynchrotronSED",
    "PowerLaw_Cooling_SynchrotronSED",
    "PowerLaw_SSA_SynchrotronSED",
    "SSA_SED_PowerLaw",
]
from .one_zone import (
    PowerLaw_Cooling_SSA_SynchrotronSED,
    PowerLaw_Cooling_SynchrotronSED,
    PowerLaw_SSA_SynchrotronSED,
    PowerLaw_SynchrotronSED,
    SSA_SED_PowerLaw,
)
