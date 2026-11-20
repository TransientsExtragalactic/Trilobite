"""
Models of luminous fast blue optical transients (LFBOTs).

This subpackage collects models of the circumstellar interaction of LFBOTs, including the two-component
equatorial and polar synchrotron model of Diggins et al. (2026), its relativistic counterpart with a thermal
electron component, and their N-segment generalizations that resolve the angular structure of the CSM.
"""

__all__ = [
    "diggins26",
    "diggins26_relativistic",
    "diggins26_n_component",
    "sectors",
]

from . import diggins26, diggins26_n_component, diggins26_relativistic, sectors
from .diggins26 import *
from .diggins26_n_component import *
from .diggins26_relativistic import *

__all__.extend(diggins26.__all__)
__all__.extend(diggins26_relativistic.__all__)
__all__.extend(diggins26_n_component.__all__)
