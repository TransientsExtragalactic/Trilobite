"""
Models of synchrotron SEDs.

This includes both phenomenological models (e.g., smoothed broken power laws) and physical
models (e.g., synchrotron emission from a population of electrons with a power-law distribution of energies).
These models are commonly used to describe the SEDs of radio supernovae, GRB afterglows, and other
synchrotron-dominated transients.
"""

from . import core_SED, generic_time_evolving

__all__ = [
    "core_SED",
    "generic_time_evolving",
]

from .core_SED import *
from .generic_time_evolving import *
