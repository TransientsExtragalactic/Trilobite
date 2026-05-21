r"""
Rosseland mean opacity laws.

This subpackage provides opacity models based on the Rosseland mean,
a frequency-averaged opacity appropriate for optically thick,
diffusion-dominated radiative transport.
"""

from .models import (
    KAPPA_BF_0,
    KAPPA_FF_0,
    KAPPA_KR_0,
    ElectronScatteringOpacity,
    KramersBFESOpacity,
    KramersBFOpacity,
    KramersESOpacity,
    KramersFFESOpacity,
    KramersFFOpacity,
    KramersOpacity,
    OPALOpacity,
)

__all__ = [
    # Constants
    "KAPPA_FF_0",
    "KAPPA_BF_0",
    "KAPPA_KR_0",
    # Analytic Rosseland mean opacity laws
    "ElectronScatteringOpacity",
    "KramersFFOpacity",
    "KramersBFOpacity",
    "KramersOpacity",
    "KramersFFESOpacity",
    "KramersBFESOpacity",
    "KramersESOpacity",
    # Table-based Rosseland mean opacity
    "OPALOpacity",
]
