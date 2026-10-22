"""
Spherical, one-zone synchrotron SEDs for power-law electron distributions.

This module contains synchrotron SED engines for the classic case of a
spherical emitting region with a single set of physical conditions. The engines
are designed to be suitable for MCMC parameter inference and are capable of performing
both forward and inverse computations.

See Also
--------
:ref:`synchrotron_seds` -- one-zone SED models and spectrum inversion
:ref:`synchrotron_theory` -- synchrotron theory and kernel functions
"""

__all__ = [
    "PowerLaw_SynchrotronSED",
    "PowerLaw_Cooling_SSA_SynchrotronSED",
    "PowerLaw_Cooling_SynchrotronSED",
    "PowerLaw_SSA_SynchrotronSED",
    "SSA_SED_PowerLaw",
    "Numerical_PowerLaw_SSA_SynchrotronSED",
    "Numerical_Thermal_SSA_SynchrotronSED",
    "Numerical_Thermal_PowerLaw_SSA_SynchrotronSED",
    "Numerical_PowerLaw_Cooling_SSA_SynchrotronSED",
    "invert_powerlaw_sed",
    "invert_powerlaw_cooling_ssa_sed",
    "invert_powerlaw_cooling_sed",
    "invert_powerlaw_ssa_sed",
    "invert_powerlaw_implicit_cooling_sed",
    "invert_powerlaw_implicit_cooling_ssa_sed",
    "invert_powerlaw_ssa_sed_demarchi",
]

from .closure import (
    invert_powerlaw_cooling_sed,
    invert_powerlaw_cooling_ssa_sed,
    invert_powerlaw_implicit_cooling_sed,
    invert_powerlaw_implicit_cooling_ssa_sed,
    invert_powerlaw_sed,
    invert_powerlaw_ssa_sed,
    invert_powerlaw_ssa_sed_demarchi,
)
from .seds import (
    Numerical_PowerLaw_Cooling_SSA_SynchrotronSED,
    Numerical_PowerLaw_SSA_SynchrotronSED,
    Numerical_Thermal_PowerLaw_SSA_SynchrotronSED,
    Numerical_Thermal_SSA_SynchrotronSED,
    PowerLaw_Cooling_SSA_SynchrotronSED,
    PowerLaw_Cooling_SynchrotronSED,
    PowerLaw_SSA_SynchrotronSED,
    PowerLaw_SynchrotronSED,
    SSA_SED_PowerLaw,
)
