"""
Accretion disk dynamics for Trilobite.

This sub-package contains time-dependent accretion disk models for use in
astrophysical transient modeling.  Currently implemented:

- :mod:`.one_zone` — one-zone (vertically-integrated) disk models following
  :footcite:t:`metzgerTimeDependentModelsAccretion2008`.
"""

from ...physics_utils.general_relativity import (
    compute_ISCO,
    compute_schwarzschild_radius,
)
from . import one_zone
from .one_zone import (
    AdvectiveDisk,
    FullPressureDisk,
    # Canonical model classes
    GasPressureDisk,
    OneZoneAccretionDiskBase,
    OneZoneAccretionResult,
    compute_advection_s_curve,
    gP_es_fbDisk,
    # Backward-compat aliases
    gP_esDisk,
    igP_es_adv_fbDisk,
    igP_es_advDisk,
    igP_es_fbDisk,
    # Utilities
    igP_esDisk,
)
from .thin_disk import (
    AlphaDisk,
    ThinDiskBase,
    disk_bolometric_luminosity,
    disk_effective_temperature,
    disk_flux_density,
    disk_spectral_luminosity,
    disk_spectral_luminosity_iso,
)

__all__ = [
    "one_zone",
    "OneZoneAccretionDiskBase",
    "OneZoneAccretionResult",
    "GasPressureDisk",
    "FullPressureDisk",
    "AdvectiveDisk",
    "gP_esDisk",
    "igP_esDisk",
    "igP_es_advDisk",
    "gP_es_fbDisk",
    "igP_es_fbDisk",
    "igP_es_adv_fbDisk",
    "compute_advection_s_curve",
    "compute_ISCO",
    "compute_schwarzschild_radius",
    # Thin-disk (Shakura-Sunyaev) models
    "ThinDiskBase",
    "AlphaDisk",
    # SS73 steady-state emission utilities
    "disk_effective_temperature",
    "disk_bolometric_luminosity",
    "disk_spectral_luminosity",
    "disk_spectral_luminosity_iso",
    "disk_flux_density",
]
