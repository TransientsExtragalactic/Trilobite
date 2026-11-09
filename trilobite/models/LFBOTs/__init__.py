"""
Models of luminous fast blue optical transients (LFBOTs).

This subpackage collects models of the circumstellar interaction of LFBOTs, including the two-component
equatorial and polar synchrotron model of Diggins et al. (2026).
"""

__all__ = [
    "diggins26",
]

from . import diggins26
from .diggins26 import *

__all__.extend(diggins26.__all__)
