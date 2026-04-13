"""
Physics utility functions and classes for Triceratops models.

This sub-package collects low-level, reusable physical utilities that are
shared across multiple Triceratops modules.  It is organised by topic:

- :mod:`triceratops.physics_utils.constants` — CGS physical constants and their logs.
- :mod:`.eos` — Equation of state classes (:class:`~.eos.IdealGasEOS`,
  :class:`~.eos.RadiativeIdealGas`) and standalone sound-speed functions.
- :mod:`.general_relativity` — Schwarzschild/Kerr radii, ISCO, spin, and
  orbital precession.
- :mod:`.composition` — Mean molecular weight helpers.
- :mod:`.special_relativity` — Lorentz / Doppler transformation utilities.
- :mod:`.cosmology` — Redshift, distance, and look-back time helpers.
"""

from .composition import (
    compute_mean_molecular_weight,
    compute_mean_molecular_weight_per_electron,
    compute_mean_molecular_weight_per_electron_primordial,
    compute_mean_molecular_weight_per_ion,
    compute_mean_molecular_weight_primordial,
)
from .cosmology import (
    age_to_redshift,
    angular_to_physical,
    get_cosmology,
    physical_to_angular,
    redshift_to_age,
    redshift_to_lookback_time,
    resolve_cosmological_distances,
)
from .eos import (
    EquationOfState,
    IdealGasEOS,
    RadiativeIdealGas,
    ideal_gas_sound_speed,
    radiative_ideal_gas_disk_sound_speed,
    radiative_ideal_gas_sound_speed,
)
from .general_relativity import (
    compute_gravitational_radius,
    compute_ISCO,
    compute_kerr_angular_momentum,
    compute_kerr_horizon_radius,
    compute_kerr_spin,
    compute_precession_per_orbit,
    compute_schwarzschild_radius,
)
from .gravity import (
    compute_bondi_radius,
    compute_hill_radius,
    compute_roche_lobe_radius,
    compute_tidal_radius,
)
from .special_relativity import (
    compute_beta_from_gamma,
    compute_doppler_factor,
    compute_lab_bolometric_intensity,
    compute_lab_frequency,
    compute_lab_specific_intensity,
    compute_lorentz_factor,
    compute_rest_bolometric_intensity,
    compute_rest_frequency,
    compute_rest_specific_intensity,
)

__all__ = [
    # EOS — classes
    "EquationOfState",
    "IdealGasEOS",
    "RadiativeIdealGas",
    # EOS — standalone functions
    "ideal_gas_sound_speed",
    "radiative_ideal_gas_sound_speed",
    "radiative_ideal_gas_disk_sound_speed",
    # Newtonian gravity
    "compute_tidal_radius",
    "compute_hill_radius",
    "compute_roche_lobe_radius",
    "compute_bondi_radius",
    # General relativity
    "compute_gravitational_radius",
    "compute_schwarzschild_radius",
    "compute_ISCO",
    "compute_kerr_horizon_radius",
    "compute_kerr_spin",
    "compute_kerr_angular_momentum",
    "compute_precession_per_orbit",
    # Composition
    "compute_mean_molecular_weight",
    "compute_mean_molecular_weight_per_electron",
    "compute_mean_molecular_weight_per_ion",
    "compute_mean_molecular_weight_primordial",
    "compute_mean_molecular_weight_per_electron_primordial",
    # Special relativity
    "compute_lorentz_factor",
    "compute_beta_from_gamma",
    "compute_doppler_factor",
    "compute_lab_frequency",
    "compute_rest_frequency",
    "compute_lab_specific_intensity",
    "compute_rest_specific_intensity",
    "compute_lab_bolometric_intensity",
    "compute_rest_bolometric_intensity",
    # Cosmology
    "get_cosmology",
    "resolve_cosmological_distances",
    "redshift_to_age",
    "redshift_to_lookback_time",
    "age_to_redshift",
    "angular_to_physical",
    "physical_to_angular",
]
