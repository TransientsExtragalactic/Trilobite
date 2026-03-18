"""
Accretion disk dynamics for Triceratops.

This sub-package contains time-dependent accretion disk models for use in
astrophysical transient modeling.  Currently implemented:

- :mod:`.one_zone` — one-zone (vertically-integrated) disk models following
  :footcite:t:`metzgerTimeDependentModelsAccretion2008`.
"""

from . import one_zone
from .one_zone import GasPressureElectronScatteringDisk, OneZoneAccretionDiskBase

__all__ = [
    "one_zone",
    "OneZoneAccretionDiskBase",
    "GasPressureElectronScatteringDisk",
]
