"""
Generic models of mathematical curves.

These models can be used to perform generic curve fitting tasks.
"""

__all__ = ["bpl", "light_curve"]

from . import bpl, light_curve
from .bpl import *
from .light_curve import *

__all__.extend(bpl.__all__)
__all__.extend(light_curve.__all__)
