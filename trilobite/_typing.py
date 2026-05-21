"""
Typing definitions for the Trilobite library.

The ``_trilobite._typing`` module provides type aliases and definitions used throughout the Trilobite
library. These types help ensure consistency and clarity in function signatures, class definitions, and
data structures.
"""

from typing import Union

import numpy as np
from astropy.units import Quantity

# Type aliases for unit bearing vectors, scalars, etc.
_UnitBearingArrayLike = Union[Quantity, np.ndarray, list[float], tuple[float, ...]]
_EnforcedUnitBearingArrayLike = Quantity
_UnitBearingScalarLike = Union[Quantity, float, int]
_EnforcedUnitBearingScalarLike = Quantity
_ArrayLike = Union[np.ndarray, list[float], tuple[float, ...]]
_ScalarLike = Union[float, int]
