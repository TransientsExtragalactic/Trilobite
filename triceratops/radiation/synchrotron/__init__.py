"""
Synchrotron Radiation Module.

This module provides tools and functions to model and analyze synchrotron radiation
processes in astrophysical contexts. It includes implementations for calculating
synchrotron emissivity, absorption, and related phenomena based on physical parameters
such as magnetic fields, electron energy distributions, and observational frequencies.
These tools can be integrated with dynamical models of astrophysical transients to
produce comprehensive simulations of radiation from events like supernovae and gamma-ray bursts.
"""

__all__ = [
    "compute_gyrofrequency",
    "compute_nu_critical",
    "compute_synchrotron_frequency",
    "compute_synchrotron_gamma",
    "compute_electron_gamma_BPL_moment",
    "compute_electron_gamma_PL_moment",
    "compute_electron_energy_BPL_moment",
    "compute_electron_energy_PL_moment",
    "compute_mean_gamma_PL",
    "compute_mean_energy_PL",
    "compute_mean_energy_BPL",
    "compute_mean_gamma_BPL",
    "compute_PL_total_number_density",
    "compute_BPL_total_number_density",
    "compute_PL_effective_number_density",
    "compute_BPL_effective_number_density",
    "compute_PL_norm_from_thermal_energy_density",
    "compute_PL_norm_from_magnetic_field",
    "compute_BPL_norm_from_magnetic_field",
    "compute_BPL_norm_from_thermal_energy_density",
    "compute_equipartition_magnetic_field",
    "compute_bol_emissivity",
    "compute_bol_emissivity_BPL",
    "compute_bol_emissivity_from_thermal_energy_density",
    "compute_bol_emissivity_BPL_from_thermal_energy_density",
    "compute_electron_gamma_BPL_moment",
    "compute_electron_gamma_PL_moment",
    "compute_electron_energy_BPL_moment",
    "compute_electron_energy_PL_moment",
    "compute_mean_gamma_PL",
    "compute_mean_energy_PL",
    "compute_mean_energy_BPL",
    "compute_mean_gamma_BPL",
    "compute_PL_total_number_density",
    "compute_BPL_total_number_density",
    "compute_PL_effective_number_density",
    "compute_BPL_effective_number_density",
    "compute_PL_norm_from_thermal_energy_density",
    "compute_PL_norm_from_magnetic_field",
    "compute_BPL_norm_from_magnetic_field",
    "compute_BPL_norm_from_thermal_energy_density",
    "compute_equipartition_magnetic_field",
    "compute_bol_emissivity",
    "compute_bol_emissivity_BPL",
    "compute_bol_emissivity_from_thermal_energy_density",
    "compute_bol_emissivity_BPL_from_thermal_energy_density",
    "PowerLaw_SSA_SynchrotronSED",
    "PowerLaw_Cooling_SSA_SynchrotronSED",
    "PowerLaw_Cooling_SynchrotronSED",
    "InverseComptonCoolingEngine",
    "SynchrotronCoolingEngine",
]

# Import the various core module items.
from triceratops.radiation.synchrotron.SEDs import (
    PowerLaw_Cooling_SSA_SynchrotronSED,
    PowerLaw_Cooling_SynchrotronSED,
    PowerLaw_SSA_SynchrotronSED,
)

from .cooling import InverseComptonCoolingEngine, SynchrotronCoolingEngine
from .core import (
    compute_gyrofrequency,
    compute_nu_critical,
    compute_synchrotron_frequency,
    compute_synchrotron_gamma,
)
from .microphysics import (
    compute_bol_emissivity,
    compute_bol_emissivity_BPL,
    compute_bol_emissivity_BPL_from_thermal_energy_density,
    compute_bol_emissivity_from_thermal_energy_density,
    compute_BPL_effective_number_density,
    compute_BPL_norm_from_magnetic_field,
    compute_BPL_norm_from_thermal_energy_density,
    compute_BPL_total_number_density,
    compute_electron_energy_BPL_moment,
    compute_electron_energy_PL_moment,
    compute_electron_gamma_BPL_moment,
    compute_electron_gamma_PL_moment,
    compute_equipartition_magnetic_field,
    compute_mean_energy_BPL,
    compute_mean_energy_PL,
    compute_mean_gamma_BPL,
    compute_mean_gamma_PL,
    compute_PL_effective_number_density,
    compute_PL_norm_from_magnetic_field,
    compute_PL_norm_from_thermal_energy_density,
    compute_PL_total_number_density,
)
