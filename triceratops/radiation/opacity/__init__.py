"""Opacity laws for Triceratops radiative physics."""

from .base import GreyOpacityLaw, OpacityLaw
from .models import (
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
from .tables import OpacityTable, OPALOpacityTable
from .utils import get_opacity

__all__ = [
    # Resolver
    "get_opacity",
    # Abstract / base
    "OpacityLaw",
    "GreyOpacityLaw",
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
    # Table containers
    "OpacityTable",
    "OPALOpacityTable",
]
