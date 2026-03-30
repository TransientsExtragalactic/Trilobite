"""
Shock physics utilities for astrophysical transients.

This subpackage provides the core shock abstractions and utilities used throughout
the :mod:`triceratops.dynamics` module, including the abstract :class:`ShockEngine` base class,
Rankine-Hugoniot jump condition utilities, and general-purpose numerical shock engines.
"""

__all__ = [
    "shock_engine",
    "rankine_hugoniot",
    "numerical",
    "ShockEngine",
    "NumericalThinShellShockEngine",
]

from triceratops.dynamics.shocks import numerical, rankine_hugoniot, shock_engine
from triceratops.dynamics.shocks.numerical import NumericalThinShellShockEngine
from triceratops.dynamics.shocks.shock_engine import ShockEngine
