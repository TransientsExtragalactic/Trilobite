"""
Type aliases for the one-zone accretion disk module.

Centralises the recurring type patterns so that ``base.py`` and ``core.py``
import from a single source of truth rather than inlining the same
``Union[...]`` expressions everywhere.
"""

from pathlib import Path
from typing import Any, Union

import numpy as np
from astropy import units as u

# ---------------------------------------------------------------------- #
# Parameter dicts                                                         #
# ---------------------------------------------------------------------- #

_ParamDict = dict[str, Any]
"""User-supplied parameter or initial-condition dict.

Values may be plain Python scalars, :class:`~astropy.units.Quantity` objects,
or :class:`numpy.ndarray` instances.  Keys are the names declared in
:attr:`~OneZoneAccretionDiskBase.RUNTIME_PARAMETERS` or
:attr:`~OneZoneAccretionDiskBase.INITIAL_CONDITIONS`.
"""

_RunParams = dict[str, float]
"""Processed per-solve parameter dict, as returned by
:meth:`~OneZoneAccretionDiskBase.process_runtime_parameters`.

All values are plain Python :class:`float`.  Log-transformed parameters are
stored under ``"log_{key}"``; others under ``"{key}"``.
"""

_SpecDict = dict[str, Any]
"""JSON-serialisable model specification dict.

Must contain a ``"target"`` key of the form ``"module.path:ClassName"`` and
any constructor keyword arguments needed to reconstruct the model.
"""

# ---------------------------------------------------------------------- #
# Field values and data dicts                                             #
# ---------------------------------------------------------------------- #

_FieldValue = Union[u.Quantity, np.ndarray]
"""A single result-field value: a unit-bearing :class:`~astropy.units.Quantity`
for dimensional quantities, or a plain :class:`numpy.ndarray` for
dimensionless ones."""

_DataDict = dict[str, _FieldValue]
"""The full data dict returned by :attr:`OneZoneAccretionResult.data`.

Maps field names (``"t"``, ``"M_D"``, ``"J_D"``, and every key in
:attr:`~OneZoneAccretionDiskBase.RESULT_FIELDS`) to their corresponding
:data:`_FieldValue` arrays.
"""

# ---------------------------------------------------------------------- #
# Miscellaneous                                                           #
# ---------------------------------------------------------------------- #

_FilePath = Union[str, Path]
"""Acceptable types for HDF5 file-path arguments."""
