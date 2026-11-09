r"""
Dynamical profile models.

This subpackage provides reusable profile classes for dynamical calculations in
Trilobite. Profiles represent analytic source functions such as circumstellar
medium (CSM) densities, homologous ejecta densities, and velocity fields. They
are used by shock engines and other dynamical models to describe upstream media,
freely expanding ejecta, and background flow fields.
"""

from .core import _DynamicalProfile
from .csm import (
    AxisymmetricCSMDensityProfile,
    BrokenPowerLawCSMProfile,
    CoredPowerLawCSMProfile,
    CSMDensityProfile,
    ExponentialCSMProfile,
    FiniteWindCSMProfile,
    GaussianShellCSMProfile,
    LogNormalCSMProfile,
    PowerLawCSMProfile,
    ShellCSMProfile,
    SmoothBPLCSMProfile,
    SmoothTruncatedWindCSMProfile,
    StationaryCSMDensityProfile,
    TruncatedWindCSMProfile,
    UniformCSMProfile,
    WindBubbleCSMProfile,
    WindCSMProfile,
    WindWithFloorCSMProfile,
)
from .ejecta import (
    BrokenPowerLawEjectaProfile,
    EjectaDensityProfile,
    ExponentialEjectaProfile,
)
from .velocity import (
    ConstantVelocityProfile,
    HomologousVelocityProfile,
    StaticVelocityProfile,
    VelocityProfile,
)

__all__ = [
    "_DynamicalProfile",
    # CSM
    "CSMDensityProfile",
    "StationaryCSMDensityProfile",
    "WindCSMProfile",
    "UniformCSMProfile",
    "PowerLawCSMProfile",
    "ShellCSMProfile",
    "GaussianShellCSMProfile",
    "BrokenPowerLawCSMProfile",
    "TruncatedWindCSMProfile",
    "WindWithFloorCSMProfile",
    "SmoothTruncatedWindCSMProfile",
    "ExponentialCSMProfile",
    "SmoothBPLCSMProfile",
    "CoredPowerLawCSMProfile",
    "FiniteWindCSMProfile",
    "WindBubbleCSMProfile",
    "LogNormalCSMProfile",
    "AxisymmetricCSMDensityProfile",
    # Ejecta
    "EjectaDensityProfile",
    "BrokenPowerLawEjectaProfile",
    "ExponentialEjectaProfile",
    # Velocity
    "VelocityProfile",
    "StaticVelocityProfile",
    "HomologousVelocityProfile",
    "ConstantVelocityProfile",
]
