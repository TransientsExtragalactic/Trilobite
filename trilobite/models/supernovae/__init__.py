"""

Trilobite models for supernovae transients.

This module contains models specifically designed to simulate and analyze supernova
events. These models incorporate the unique physical characteristics and behaviors
of supernovae, allowing researchers to study their radio emissions, shock dynamics,
and other relevant phenomena. It includes radio models of the shock interaction
(:mod:`~trilobite.models.supernovae.chevalier_shock`) and optical light-curve models of radioactively or
centrally powered supernovae (:mod:`~trilobite.models.supernovae.arnett`).
"""

__all__ = [
    "chevalier_shock",
    "arnett",
]

from . import arnett, chevalier_shock
from .arnett import *
from .chevalier_shock import *

__all__.extend(chevalier_shock.__all__)
__all__.extend(arnett.__all__)
