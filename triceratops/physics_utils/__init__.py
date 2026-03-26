"""
Physics utility functions and classes for Triceratops models.

This sub-package collects low-level, reusable physical utilities that are
shared across multiple Triceratops modules.  It is organised by topic:

- :mod:`.constants` — CGS physical constants and their logs.
- :mod:`.eos` — Equation of state classes (:class:`~.eos.IdealGasEOS`,
  :class:`~.eos.RadiativeIdealGas`) and standalone sound-speed functions.
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
