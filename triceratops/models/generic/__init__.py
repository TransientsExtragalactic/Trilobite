"""
Generic models of mathematical curves.

These models can be used to perform generic curve fitting tasks.
"""

__all__ = ["light_curve", "evolving_seds", "curves", "optical_photometry"]

from . import curves, evolving_seds, light_curve, optical_photometry
from .curves import *
from .evolving_seds import *
from .light_curve import *
from .optical_photometry import *

__all__.extend(curves.__all__)
__all__ += evolving_seds.__all__
__all__.extend(light_curve.__all__)
__all__.extend(optical_photometry.__all__)
