"""
Physics tools and methods for use in radio modeling.

This module contains most of the pure-physics related tools for Triceratops. This includes
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
    "planck_fnu",
    "planck_flambda",
    "wien_peak_frequency",
    "wien_peak_wavelength",
    "stefan_boltzmann_flux",
]
from . import blackbody, constants, synchrotron
from .blackbody import (
    planck_flambda,
    planck_fnu,
    stefan_boltzmann_flux,
    wien_peak_frequency,
    wien_peak_wavelength,
)
