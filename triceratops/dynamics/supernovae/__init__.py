"""
Self-similar dynamical models for astrophysical transients.

These modules provide tools for computing the self-similar dynamical evolution of
astrophysical transients, including supernovae and gamma-ray bursts. These can be coupled with
radiation processes in the :mod:`triceratops.radiation` module to produce self-consistent physical models in
the :mod:`triceratops.models` module.
"""

__all__ = [
    "shock_dynamics",
    "ChevalierSelfSimilarWindShockEngine",
    "ChevalierSelfSimilarShockEngine",
]

from . import shock_dynamics
from .shock_dynamics import (
    ChevalierSelfSimilarShockEngine,
    ChevalierSelfSimilarWindShockEngine,
)
