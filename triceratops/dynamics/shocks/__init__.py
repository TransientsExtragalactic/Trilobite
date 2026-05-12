"""
Shock physics utilities for astrophysical transients.

This subpackage provides the core shock abstractions and utilities used throughout
the :mod:`triceratops.dynamics` module, including the abstract :class:`ShockEngine` base class,
Rankine-Hugoniot jump condition utilities, and general-purpose numerical shock engines.
"""

__all__ = [
    "rankine_hugoniot",
    "relativistic_jump_conditions",
    "numerical",
    "NumericalThinShellShockEngine",
    "RelativisticJumpConditions",
    "RelativisticShockConditions",
    "RelativisticColdShockConditions",
    "UltraRelativisticShockConditions",
    "UltraRelativisticColdShockConditions",
    "ChevalierSelfSimilarShockEngine",
    "ChevalierSelfSimilarWindShockEngine",
    "SedovTaylorShockEngine",
    "sedov_taylor_beta",
    "normalize_supernova_ejecta",
    "get_broken_power_law_ejecta_kernel_func",
    "get_wind_csm_density_func",
]
from triceratops.dynamics.shocks import numerical
from triceratops.dynamics.shocks.chevalier import (
    ChevalierSelfSimilarShockEngine,
    ChevalierSelfSimilarWindShockEngine,
    get_broken_power_law_ejecta_kernel_func,
    get_wind_csm_density_func,
    normalize_supernova_ejecta,
)
from triceratops.dynamics.shocks.core import rankine_hugoniot, relativistic_jump_conditions
from triceratops.dynamics.shocks.core.relativistic_jump_conditions import (
    RelativisticColdShockConditions,
    RelativisticJumpConditions,
    RelativisticShockConditions,
    UltraRelativisticColdShockConditions,
    UltraRelativisticShockConditions,
)
from triceratops.dynamics.shocks.numerical import NumericalThinShellShockEngine
from triceratops.dynamics.shocks.sedov_taylor import SedovTaylorShockEngine, sedov_taylor_beta
