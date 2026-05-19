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
    "ChevalierShockState",
    "ChevalierSelfSimilarShockEngine",
    "ChevalierSelfSimilarWindShockEngine",
    "SedovTaylorShockState",
    "SedovTaylorShockEngine",
    "sedov_taylor_beta",
    "ThinShellShockState",
    "MechanicalShockState",
    "MechanicalShockEngine",
    "PressureDrivenThinShellShockEngine",
    "RelativisticJumpConditions",
    "RelativisticShockConditions",
    "RelativisticColdShockConditions",
    "UltraRelativisticShockConditions",
    "UltraRelativisticColdShockConditions",
    "normalize_bpl_ejecta",
    "get_bpl_ejecta_kernel",
    "get_wind_csm_density_func",
    "make_homologous_stationary_sources",
]
from triceratops.dynamics.shocks import numerical
from triceratops.dynamics.shocks.chevalier import (
    ChevalierSelfSimilarShockEngine,
    ChevalierSelfSimilarWindShockEngine,
    ChevalierShockState,
)
from triceratops.dynamics.shocks.core import rankine_hugoniot, relativistic_jump_conditions
from triceratops.dynamics.shocks.core.relativistic_jump_conditions import (
    RelativisticColdShockConditions,
    RelativisticJumpConditions,
    RelativisticShockConditions,
    UltraRelativisticColdShockConditions,
    UltraRelativisticShockConditions,
)
from triceratops.dynamics.shocks.numerical import (
    MechanicalShockEngine,
    MechanicalShockState,
    PressureDrivenThinShellShockEngine,
    ThinShellShockState,
)
from triceratops.dynamics.shocks.sedov_taylor import SedovTaylorShockEngine, SedovTaylorShockState, sedov_taylor_beta
from triceratops.dynamics.shocks.utils import (
    get_bpl_ejecta_kernel,
    get_wind_csm_density_func,
    make_homologous_stationary_sources,
    normalize_bpl_ejecta,
)
