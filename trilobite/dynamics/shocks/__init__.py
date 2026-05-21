"""
Shock physics utilities for astrophysical transients.

This subpackage provides the core shock abstractions and utilities used throughout
the :mod:`trilobite.dynamics` module, including the abstract :class:`ShockEngine` base class,
Rankine-Hugoniot jump condition utilities, and general-purpose numerical shock engines.
"""

__all__ = [
    "rankine_hugoniot",
    "relativistic_jump_conditions",
    "numerical",
    "BlandfordMcKeeShockState",
    "BlandfordMcKeeShockEngine",
    "BlandfordMcKeeWindShockEngine",
    "ChevalierSelfSimilarFunctions",
    "ChevalierSelfSimilarCriticalGrid",
    "ChevalierShockState",
    "ChevalierSelfSimilarShockEngine",
    "ChevalierSelfSimilarWindShockEngine",
    "ChevalierTwoShockState",
    "ChevalierTwoShockSelfSimilarEngine",
    "ChevalierTwoShockSelfSimilarWindEngine",
    "compute_self_similar_functions",
    "compute_self_similar_critical_grid",
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
from trilobite.dynamics.shocks import numerical
from trilobite.dynamics.shocks.blandford_mckee import (
    BlandfordMcKeeShockEngine,
    BlandfordMcKeeShockState,
    BlandfordMcKeeWindShockEngine,
)
from trilobite.dynamics.shocks.chevalier import (
    ChevalierSelfSimilarCriticalGrid,
    ChevalierSelfSimilarFunctions,
    ChevalierSelfSimilarShockEngine,
    ChevalierSelfSimilarWindShockEngine,
    ChevalierShockState,
    ChevalierTwoShockSelfSimilarEngine,
    ChevalierTwoShockSelfSimilarWindEngine,
    ChevalierTwoShockState,
    compute_self_similar_critical_grid,
    compute_self_similar_functions,
)
from trilobite.dynamics.shocks.core import rankine_hugoniot, relativistic_jump_conditions
from trilobite.dynamics.shocks.core.relativistic_jump_conditions import (
    RelativisticColdShockConditions,
    RelativisticJumpConditions,
    RelativisticShockConditions,
    UltraRelativisticColdShockConditions,
    UltraRelativisticShockConditions,
)
from trilobite.dynamics.shocks.numerical import (
    MechanicalShockEngine,
    MechanicalShockState,
    PressureDrivenThinShellShockEngine,
    ThinShellShockState,
)
from trilobite.dynamics.shocks.sedov_taylor import SedovTaylorShockEngine, SedovTaylorShockState, sedov_taylor_beta
from trilobite.dynamics.shocks.utils import (
    get_bpl_ejecta_kernel,
    get_wind_csm_density_func,
    make_homologous_stationary_sources,
    normalize_bpl_ejecta,
)
