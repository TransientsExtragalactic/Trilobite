"""
Dynamical modeling functions for use in Triceratops models.

These modules provide tools for computing the dynamical evolution of
astrophysical transients, including supernovae and gamma-ray bursts. These can be coupled with
radiation processes in the :mod:`triceratops.radiation` module to produce self-consistent physical models in
the :mod:`triceratops.models` module.
"""

__all__ = [
    "shocks",
    "supernovae",
    "accretion",
]
from triceratops.dynamics import accretion, shocks, supernovae
