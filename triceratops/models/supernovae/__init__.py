"""

Triceratops models for supernovae transients.

This module contains models specifically designed to simulate and analyze supernova
events. These models incorporate the unique physical characteristics and behaviors
of supernovae, allowing researchers to study their radio emissions, shock dynamics,
and other relevant phenomena.
"""

__all__ = [
    "chevalier_shock",
]

from . import chevalier_shock
from .chevalier_shock import *

__all__.extend(chevalier_shock.__all__)
