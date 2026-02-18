"""
Data management infrastructure for observational radio datasets in Triceratops.

The :mod:`triceratops.data` package provides structured, unit-aware containers
for working with radio observations. These containers form the boundary between
raw observational data and the modeling and inference layers of the library.

Overview
--------
The data layer is responsible for:

- Enforcing well-defined column schemas for observational tables,
- Preserving and validating physical units,
- Providing convenient accessors for detections, non-detections,
  and grouped observations,
- Translating observational data into :class:`InferenceData`,
  the numerical format required by likelihood classes.

.. note::

    For detailed documentation, see :ref:`data_overview`.
"""

__all__ = ["light_curve", "spectra", "photometry", "core"]
from . import core, light_curve, photometry, spectra
from .core import *
from .light_curve import *
from .photometry import *
from .spectra import *

__all__.extend(light_curve.__all__)
__all__.extend(photometry.__all__)
__all__.extend(core.__all__)
