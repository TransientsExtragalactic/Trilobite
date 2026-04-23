"""Opacity laws for Triceratops radiative physics.

This subpackage provides various utilities for computing opacities in the various
modeling contexts supported by Triceratops. This module provides access to a number of different types
of opacities, including

- **grey opacities** (frequency-averaged, e.g. Rosseland or Planck mean) implemented as pure-Python classes
  or Cython extensions;
- **frequency-dependent opacities** (e.g. multigroup) implemented as pure-Python classes or Cython extensions
  (not yet implemented);
- **analytic opacity laws** (e.g. Kramers power laws) implemented as pure-Python classes or Cython extensions;
- **table-based opacities** (e.g. OPAL) implemented as Cython extensions that perform bilinear
  interpolation on preloaded tables.

These are designed to be quick to evaluate and easy to use in the various contexts where they arise
in Triceratops.
"""

from .base import OpacityLaw
from .grey_opacity import (
    ConstantGreyOpacity,
    ElectronScatteringOpacity,
    GreyOpacityLaw,
    KramersBFESOpacity,
    KramersBFOpacity,
    KramersESOpacity,
    KramersFFESOpacity,
    KramersFFOpacity,
    KramersOpacity,
    OPALOpacity,
    TOPSOpacity,
)
from .utils import get_opacity, load_opal_opacity, load_tops_opacity

__all__ = [
    # Resolver
    "get_opacity",
    # Abstract / base
    "OpacityLaw",
    "GreyOpacityLaw",
    # Analytic opacity laws
    "ElectronScatteringOpacity",
    "KramersFFOpacity",
    "KramersBFOpacity",
    "KramersOpacity",
    "KramersFFESOpacity",
    "KramersBFESOpacity",
    "KramersESOpacity",
    "ConstantGreyOpacity",
    # Table-based opacity
    "OPALOpacity",
    "TOPSOpacity",
    "load_opal_opacity",
    "load_tops_opacity",
]
