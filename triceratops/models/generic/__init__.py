"""
Generic models of mathematical curves.

These models can be used to perform generic curve fitting tasks.
"""

__all__ = ["bpl", "light_curve", "evolving_sed"]

from . import bpl, evolving_sed, light_curve
from .bpl import *
from .evolving_sed import *
from .light_curve import *

__all__.extend(bpl.__all__)
__all__.extend(light_curve.__all__)
__all__.extend(evolving_sed.__all__)
