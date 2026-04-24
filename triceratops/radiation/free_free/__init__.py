"""
Free–free (bremsstrahlung) radiation utilities.

This module provides tools for thermal bremsstrahlung (free-free) emission in
Triceratops transient models. This includes functions for computing emissivities and
absorption coefficients, emission measures, and optical depths for a variety of geometries and assumptions.

Additionally, this module includes several utilities for prescribing the
Gaunt factor corrections to the classical free–free formulae, including both
interpolators based on tabulated data and analytic approximations.
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
    "compute_ff_gaunt_factor",
    "compute_ff_gaunt_factor_comp",
    "compute_ff_optical_depth_from_arrays",
    "compute_ff_RJ_optical_depth_from_arrays",
    "compute_ff_RJ_optical_depth_from_quadrature",
    "compute_ff_optical_depth_from_quadrature",
    "compute_ff_RJ_optical_depth_powerlaw",
    "compute_ff_RJ_optical_depth_shell",
    "compute_ff_optical_depth_shell",
    "compute_ff_optical_depth_wind",
    "compute_ff_optical_depth_powerlaw",
    "compute_ff_RJ_optical_depth_wind",
]

from . import core, gaunt_factor
from .absorption import (
    compute_ff_optical_depth_from_arrays,
    compute_ff_optical_depth_from_quadrature,
    compute_ff_optical_depth_powerlaw,
    compute_ff_optical_depth_shell,
    compute_ff_optical_depth_wind,
    compute_ff_RJ_optical_depth_from_arrays,
    compute_ff_RJ_optical_depth_from_quadrature,
    compute_ff_RJ_optical_depth_powerlaw,
    compute_ff_RJ_optical_depth_shell,
    compute_ff_RJ_optical_depth_wind,
)
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
    compute_ff_gaunt_factor,
    compute_ff_gaunt_factor_comp,
    get_default_gaunt_interpolator,
    get_default_relativistic_gaunt_interpolator,
)
