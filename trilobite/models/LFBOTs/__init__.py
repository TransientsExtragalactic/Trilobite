"""
Models of luminous fast blue optical transients (LFBOTs).

This subpackage collects models of the circumstellar interaction of LFBOTs, including the two-component
equatorial and polar synchrotron model of Diggins et al. (2026).
"""

from . import _diggins26
from ._diggins26 import *

__all__ = list(_diggins26.__all__)
