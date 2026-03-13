"""
Module supporting free-free radiation calculations for Triceratops.

In dev.
"""

__all__ = [
    "core",
    "gaunt_factor",
    "compute_ff_absorption",
    "compute_ff_emissivity",
    "compute_ff_RJ_absorption",
    "compute_ff_RJ_emissivity",
    "GauntFactorInterpolatorBase",
    "NonRelativisticGauntFactorInterpolator",
    "RelativisticGauntFactorInterpolator",
    "get_default_gaunt_interpolator",
    "get_default_relativistic_gaunt_interpolator",
    "gaunt_ff_draine",
]

from . import core, gaunt_factor
from .core import (
    compute_ff_absorption,
    compute_ff_emissivity,
    compute_ff_RJ_absorption,
    compute_ff_RJ_emissivity,
)
from .gaunt_factor import (
    GauntFactorInterpolatorBase,
    NonRelativisticGauntFactorInterpolator,
    RelativisticGauntFactorInterpolator,
    gaunt_ff_draine,
    get_default_gaunt_interpolator,
    get_default_relativistic_gaunt_interpolator,
)
