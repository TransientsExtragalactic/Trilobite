"""
Generic models of time evolving SEDs.

This module contains a number of phenomenological (not physics-based) models of time-evolving
SEDs that can be used for flexible fitting of multi-epoch spectral data. These models are not
intended to be physically self-consistent, but rather to provide flexible parameterizations of
observed spectral evolution. They can be used for:

- Empirical SED fitting
- Multi-epoch spectral evolution studies
- Exploratory transient modeling
- Situations where physical interpretation is secondary
"""

__all__ = ["base", "evolving_sbpl"]

from . import base, evolving_sbpl
from .base import *
from .evolving_sbpl import *

__all__ += base.__all__
__all__ += evolving_sbpl.__all__
