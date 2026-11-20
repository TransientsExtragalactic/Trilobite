"""
Numerical synchrotron SED engines.

This subpackage provides numerical engines for computing synchrotron SEDs from
arbitrary electron distributions via direct kernel integration. Supports
synchrotron self-absorption, relativistic bulk motion, and multi-zone geometries.

See Also
--------
:mod:`trilobite.radiation.synchrotron.SEDs.numerical.core`
    Base engine class providing kernel tabulation and low-level radiative-transfer machinery.
:mod:`trilobite.radiation.synchrotron.SEDs.numerical.aspherical`
    On-axis axisymmetric (aspherical) multi-zone engine.
:mod:`trilobite.radiation.synchrotron.SEDs.numerical.one_zone`
    Non-relativistic and ultra-relativistic spherical geometry wrappers.
:mod:`trilobite.radiation.synchrotron.SEDs.one_zone`
    Analytic and numerical one-zone SED models.
"""

from .aspherical import OffAxisAsymmetricSynchrotronEngine, OnAxisAsymmetricSynchrotronEngine
from .core import NumericalSynchrotronEngine
from .inhomogeneous import InhomogeneousCylinderSynchrotronEngine, InhomogeneousSphereSynchrotronEngine
from .one_zone import (
    NonRelativisticSphericalSynchrotronEngine,
    UltraRelativisticSphericalSynchrotronEngine,
)

__all__ = [
    "NumericalSynchrotronEngine",
    "NonRelativisticSphericalSynchrotronEngine",
    "UltraRelativisticSphericalSynchrotronEngine",
    "OnAxisAsymmetricSynchrotronEngine",
    "OffAxisAsymmetricSynchrotronEngine",
    "InhomogeneousCylinderSynchrotronEngine",
    "InhomogeneousSphereSynchrotronEngine",
]
