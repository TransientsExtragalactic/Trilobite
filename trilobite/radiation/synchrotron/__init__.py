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
    "compute_averaged_first_synchrotron_kernel",
    "compute_first_synchrotron_kernel",
    "InverseComptonCoolingEngine",
    "SynchrotronCoolingEngine",
    "equipartition_magnetic_field",
    "equipartition_electron_energy",
    "ElectronDistribution",
    "PowerLaw",
    "BrokenPowerLaw",
    "MaxwellJuettner",
    "MixedThermalNonThermal",
    "MaxwellJuettnerPowerLaw",
    "MaxwellJuettnerBrokenPowerLaw",
    "invert_ssa_Diggins26",
    "invert_ssa_synchrotron_cooled_Diggins26",
    "invert_ssa_ic_cooled_Diggins26",
    "invert_ssa_combined_cooled_Diggins26",
    "invert_ssa_with_cooling_Diggins26",
    "invert_ssa_DeMarchi22",
]

# Import the various core module items.
from trilobite.radiation.synchrotron import SEDs
from trilobite.radiation.synchrotron.SEDs import *

from .cooling import InverseComptonCoolingEngine, SynchrotronCoolingEngine
from .core import (
    compute_averaged_first_synchrotron_kernel,
    compute_first_synchrotron_kernel,
    compute_gyrofrequency,
    compute_nu_critical,
    compute_synchrotron_frequency,
    compute_synchrotron_gamma,
)
from .electron_distributions import (
    BrokenPowerLaw,
    ElectronDistribution,
    MaxwellJuettner,
    MaxwellJuettnerBrokenPowerLaw,
    MaxwellJuettnerPowerLaw,
    MixedThermalNonThermal,
    PowerLaw,
    equipartition_electron_energy,
    equipartition_magnetic_field,
)
from .spectral_inversion import (
    invert_ssa_combined_cooled_Diggins26,
    invert_ssa_DeMarchi22,
    invert_ssa_Diggins26,
    invert_ssa_ic_cooled_Diggins26,
    invert_ssa_synchrotron_cooled_Diggins26,
    invert_ssa_with_cooling_Diggins26,
)

__all__ += SEDs.__all__
