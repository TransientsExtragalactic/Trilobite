"""
Synchrotron spectral energy distribution models.

This package implements the various spectral energy distributions relevant to synchrotron
emission from astrophysical transients. It is organized into two subpackages:

- :mod:`~trilobite.radiation.synchrotron.SEDs.one_zone`: analytic power-law one-zone SED
  models and inversion closures.
- :mod:`~trilobite.radiation.synchrotron.SEDs.numerical`: numerical SED engines for
  arbitrary electron distributions.

For documentation on synchrotron emission modeling in Trilobite, see :ref:`radiation_overview`.
"""

from . import numerical, one_zone
from .numerical import *
from .one_zone import *

__all__ = [
    "numerical",
    "one_zone",
]

__all__.extend(one_zone.__all__)
__all__.extend(numerical.__all__)
