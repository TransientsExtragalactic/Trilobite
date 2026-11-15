"""
Models of luminous fast blue optical transients (LFBOTs).

This subpackage collects models of the circumstellar interaction of LFBOTs, including the two-component
equatorial and polar synchrotron model of Diggins et al. (2026) and its relativistic counterpart with a thermal
electron component.
"""

__all__ = [
    "diggins26",
    "diggins26_relativistic",
]

from . import diggins26, diggins26_relativistic
from .diggins26 import *
from .diggins26_relativistic import *

__all__.extend(diggins26.__all__)
__all__.extend(diggins26_relativistic.__all__)
