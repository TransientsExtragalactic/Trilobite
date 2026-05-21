"""
Data management infrastructure for observational datasets in Trilobite.

The :mod:`trilobite.data` package provides structured, unit-aware containers
for working with multi-wavelength observations — from radio photometry to optical
survey data. These containers form the boundary between raw observational data
and the modeling and inference layers of the library.

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

__all__ = ["light_curve", "photometry", "optical_photometry", "core"]
from . import core, light_curve, optical_photometry, photometry
from .core import *
from .light_curve import *
from .optical_photometry import *
from .photometry import *

__all__.extend(light_curve.__all__)
__all__.extend(photometry.__all__)
__all__.extend(optical_photometry.__all__)
__all__.extend(core.__all__)
