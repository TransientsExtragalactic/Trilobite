"""
Accretion disk dynamics for Triceratops.

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
    compute_advective_equilibrium_temperature,
    compute_advective_s_curve,
    compute_equilibrium_temperature,
    compute_standard_s_curve,
    gP_es_fbDisk,
    # Backward-compat aliases
    gP_esDisk,
    igP_es_adv_fbDisk,
    igP_es_adv_trial_curves,
    igP_es_advDisk,
    igP_es_fbDisk,
    # Utilities
    igP_es_trial_curves,
    igP_esDisk,
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
    "igP_es_trial_curves",
    "igP_es_adv_trial_curves",
    "compute_equilibrium_temperature",
    "compute_advective_equilibrium_temperature",
    "compute_standard_s_curve",
    "compute_advective_s_curve",
    "compute_ISCO",
    "compute_schwarzschild_radius",
]
