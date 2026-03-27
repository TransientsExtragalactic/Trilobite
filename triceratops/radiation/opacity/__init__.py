"""Grey opacity laws for Triceratops radiative physics."""

from .base import GreyOpacityLaw, OpacityLaw
from .models import (
    ConstantOpacity,
    ElectronScatteringOpacity,
    KramersBFOpacity,
    KramersFFOpacity,
    KramersOpacity,
)

__all__ = [
    # Abstract / base
    "OpacityLaw",
    "GreyOpacityLaw",
    # Concrete implementations
    "ConstantOpacity",
    "ElectronScatteringOpacity",
    "KramersFFOpacity",
    "KramersBFOpacity",
    "KramersOpacity",
]
