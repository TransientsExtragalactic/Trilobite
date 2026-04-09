"""Concrete opacity law implementations.

Classes are defined in :mod:`.core`; this module re-exports them for
convenience.  See :mod:`.core` for full docstrings and examples.
"""

from .core import (
    KAPPA_BF_0,
    KAPPA_FF_0,
    KAPPA_KR_0,
    ConstantOpacity,
    ElectronScatteringOpacity,
    KramersBFESOpacity,
    KramersBFOpacity,
    KramersESOpacity,
    KramersFFESOpacity,
    KramersFFOpacity,
    KramersOpacity,
    OPALOpacity,
    load_opal_opacity,
)

__all__ = [
    # Constants
    "KAPPA_FF_0",
    "KAPPA_BF_0",
    "KAPPA_KR_0",
    # Analytic opacity laws
    "ConstantOpacity",
    "ElectronScatteringOpacity",
    "KramersFFOpacity",
    "KramersBFOpacity",
    "KramersOpacity",
    "KramersFFESOpacity",
    "KramersBFESOpacity",
    "KramersESOpacity",
    # Table-based opacity
    "OPALOpacity",
    "load_opal_opacity",
]
