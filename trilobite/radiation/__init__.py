"""
Physics tools and methods for use in radio modeling.

This module contains most of the pure-physics related tools for Trilobite. This includes
the radiative processes, shock dynamics, and other physical calculations needed for
modeling synchrotron emission from supernova shocks and similar phenomena.

.. note::

    For separation of concerns reasons, this module does **NOT** include any modeling. The methods here
    are simply physics building-blocks which can be combined to generate models elsewhere in the library.
"""

__all__ = [
    "blackbody",
    "constants",
    "synchrotron",
    # blackbody public API
    "planck_B",
    "planck_surface_flux",
    "planck_flux",
    "planck_luminosity",
    "stefan_boltzmann_flux",
    "stefan_boltzmann_observed_flux",
    "wien_peak_frequency",
    "wien_peak_wavelength",
    "photospheric_radius",
    "photospheric_temperature",
    "photospheric_radius_from_flux",
    "photospheric_temperature_from_flux",
]
from . import blackbody, constants, synchrotron
from .blackbody import (
    photospheric_radius,
    photospheric_radius_from_flux,
    photospheric_temperature,
    photospheric_temperature_from_flux,
    planck_B,
    planck_flux,
    planck_luminosity,
    planck_surface_flux,
    stefan_boltzmann_flux,
    stefan_boltzmann_observed_flux,
    wien_peak_frequency,
    wien_peak_wavelength,
)
