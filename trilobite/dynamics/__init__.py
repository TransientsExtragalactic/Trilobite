"""
Dynamical modeling functions for use in Trilobite models.

These modules provide tools for computing the dynamical evolution of
astrophysical transients, including supernovae and gamma-ray bursts. These can be coupled with
radiation processes in the :mod:`trilobite.radiation` module to produce self-consistent physical models in
the :mod:`trilobite.models` module.
"""

__all__ = [
    "shocks",
    "accretion",
]
from trilobite.dynamics import accretion, shocks
