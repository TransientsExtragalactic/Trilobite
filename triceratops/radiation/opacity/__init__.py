"""Grey opacity laws for Triceratops radiative physics."""

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
)
from .utils import get_opacity

__all__ = [
    # Resolver
    "get_opacity",
    # Abstract / base
    "OpacityLaw",
    "GreyOpacityLaw",
    # Concrete implementations
    "ConstantOpacity",
    "ElectronScatteringOpacity",
    "KramersFFOpacity",
    "KramersBFOpacity",
    "KramersOpacity",
    "KramersFFESOpacity",
    "KramersBFESOpacity",
    "KramersESOpacity",
]
