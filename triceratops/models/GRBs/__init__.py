"""
Models of Gamma-Ray Bursts (GRBs) for Triceratops.

This module contains models specifically designed to simulate and analyze gamma-ray
burst (GRB) events. These models incorporate the unique physical characteristics and
behaviors of GRBs, allowing researchers to study their radio emissions, jet dynamics,
and other relevant phenomena.
"""

__all__ = [
    "band",
]

from . import band
from .band import *

__all__.extend(band.__all__)
