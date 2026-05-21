"""
Models of radio emission from astrophysical transients.

This module includes various models for simulating and analyzing the spectral energy distributions (SEDs
of astrophysical transient events such as supernovae, gamma-ray bursts, and tidal disruption events. Models
are distributed across several submodules for better organization and maintainability.
"""

__all__ = ["core", "generic", "GRBs", "supernovae", "TDEs", "SEDs"]

# Import the core submodules.
from . import GRBs, SEDs, TDEs, core, generic, supernovae

# Add all of the individual models to the top-level namespace for easier access.
from .generic import *
from .GRBs import *
from .SEDs import *
from .supernovae import *
from .TDEs import *

# Add everything to __all__
__all__.extend(core.__all__)
__all__.extend(generic.__all__)
__all__.extend(GRBs.__all__)
__all__.extend(supernovae.__all__)
__all__.extend(TDEs.__all__)
__all__.extend(SEDs.__all__)
