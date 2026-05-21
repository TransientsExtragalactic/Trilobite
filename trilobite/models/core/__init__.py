"""
Abstract base classes and core structures for building models.

This module contains the foundational classes and utilities that serve as the backbone for
various modeling approaches in the trilobite package. It provides abstract base classes that define
the essential interfaces and behaviors that all specific model implementations must adhere to. In general,
these should not be used directly, but rather extended by other modules to create concrete models.
"""

__all__ = ["Model", "ModelParameter", "ModelVariable", "OpticalModel", "OpticalEpochModel"]
from .base import Model
from .optical import OpticalEpochModel, OpticalModel
from .parameters import ModelParameter, ModelVariable
