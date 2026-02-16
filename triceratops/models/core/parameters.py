"""
Model parameter declarations for Triceratops models.

This module defines the :class:`ModelParameter` class, which encapsulates metadata
and validation logic for parameters used in Triceratops physical and phenomenological
models. Model parameters are unit-aware, optionally bounded, and designed to integrate
cleanly with inference frameworks such as MCMC samplers.
"""

from collections.abc import Callable
from typing import Optional, Union

import numpy as np
from astropy import units as u

# Type alias for parameter bounds
BoundType = Union[
    tuple[Optional[float], Optional[float]],
    Callable[[float], bool],
]


# ----------------------------------------------------------------------
# ModelParameter class
# ----------------------------------------------------------------------
class ModelParameter:
    r"""
    Declarative container for a model parameter.

    A :class:`ModelParameter` represents a single scalar parameter in a Triceratops
    model, along with metadata describing its physical meaning, units, default value,
    and validity constraints. It does **not** store parameter state during inference;
    rather, it defines how values *should* be interpreted and validated.

    The model is largely used for bookkeeping at the model level, but is used for validation
    during inference. For example, when sampling model parameters with MCMC, the
    :class:`ModelParameter`'s bounds can be used to reject invalid samples.

    Parameters
    ----------
    name : str
        Name of the parameter. This must match the keyword expected by the model's
        forward function.
    default : float or astropy.units.Quantity
        Default value of the parameter. If a Quantity is provided, it will be coerced
        into ``base_units``.
    base_units : str or astropy.units.Unit, optional
        Base units assumed by the model's internal (low-level) physics implementation.
        Defaults to dimensionless.
    description : str, optional
        Human-readable description of the parameter.
    latex : str, optional
        LaTeX representation of the parameter symbol (e.g. ``r"$\epsilon_B$"``).
    bounds : tuple or callable, optional
        Either:

        - A ``(min, max)`` tuple of floats in base units, or
        - A callable ``f(value) -> bool`` that returns whether the value is valid.

    Notes
    -----
    - This class is intentionally lightweight and immutable.
    - Bounds are **not** enforced automatically during model evaluation.
      They are intended for use by inference layers (e.g. likelihood evaluation).
    - More complex, multi-parameter constraints should live at the *model* level,
      not on individual parameters.
    """

    # ===================================================
    # Initialization
    # ===================================================
    def __init__(
        self,
        name: str,
        default: Union[float, u.Quantity],
        *,
        base_units: Union[str, u.Unit] = u.dimensionless_unscaled,
        description: str = "",
        latex: str = "",
        bounds: BoundType = (None, None),
    ):
        r"""
        Initialize a :class:`ModelParameter` instance.

        Parameters
        ----------
        name : str
            Name of the parameter. This must match the keyword expected by the model's
            forward function.
        default : float or astropy.units.Quantity
            Default value of the parameter. If a Quantity is provided, it will be coerced
            into ``base_units``.
        base_units : str or astropy.units.Unit, optional
            Base units assumed by the model's internal (low-level) physics implementation.
            Defaults to dimensionless.
        description : str, optional
            Human-readable description of the parameter.
        latex : str, optional
            LaTeX representation of the parameter symbol (e.g. ``r"$\epsilon_B$"``).
        bounds : tuple or callable, optional
            Either:

            - A ``(min, max)`` tuple of floats in base units, or
            - A callable ``f(value) -> bool`` that returns whether the value is valid.

        """
        # Declare the core attributes.
        self.name: str = name
        """str: Name of the parameter."""
        self.description: str = description
        """str: Human-readable description of the parameter."""
        self.latex: str = latex
        """str: LaTeX representation of the parameter symbol."""

        # Handle the units and the default value. We handle this in separate
        # methods for extensibility of subclasses when / if relevant / necessary.
        self.base_units: u.Unit = self._parse_base_units(base_units)
        """astropy.units.Unit: Base units for this parameter."""
        self.default: float = self._parse_default(default)
        """float: Default value in base units."""
        self.default_quantity: u.Quantity = self.default * self.base_units
        """astropy.units.Quantity: Default value as a Quantity."""

        # Handle the bounds definition. Again, we do this in a separate
        # method for extensibility.
        self.bounds: BoundType = bounds
        """BoundType: Bounds definition for this parameter."""
        self._check_bounds_base: Callable[[float], bool] = self._build_base_bounds_checker(bounds)
        """callable: Low-level bounds-checking function in base units."""

    # ===================================================
    # Initialization Helpers
    # ===================================================
    def _parse_base_units(self, base_units: Union[str, u.Unit]) -> u.Unit:
        """Parse and validate base units."""
        if base_units is None:
            return u.Unit("")
        if isinstance(base_units, str):
            try:
                return u.Unit(base_units)
            except Exception as exc:
                raise ValueError(f"Invalid unit string '{base_units}' for parameter '{self.name}'.") from exc
        elif isinstance(base_units, u.UnitBase):
            return base_units
        else:
            raise TypeError(f"base_units must be a str or astropy Unit, got {type(base_units)}.")

    def _parse_default(self, default: Union[float, u.Quantity]) -> float:
        """Parse and coerce the default value into base units."""
        if isinstance(default, u.Quantity):
            try:
                return default.to(self.base_units).value
            except Exception as exc:
                raise ValueError(f"Default value for parameter '{self.name}' has incompatible units.") from exc
        elif isinstance(default, (int, float)):
            return float(default)
        else:
            raise TypeError(f"Default value for parameter '{self.name}' must be a float or Quantity.")

    def _build_base_bounds_checker(self, bounds: BoundType) -> Callable[[float], bool]:
        """Construct a bounds-checking function operating in base units."""
        if callable(bounds):
            return bounds

        if not isinstance(bounds, tuple) or len(bounds) != 2:
            raise ValueError(f"Bounds for parameter '{self.name}' must be a (min, max) tuple or callable.")

        lower, upper = bounds

        if lower is not None and not isinstance(lower, (int, float)):
            raise TypeError("Lower bound must be a float or None.")
        if upper is not None and not isinstance(upper, (int, float)):
            raise TypeError("Upper bound must be a float or None.")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("Lower bound cannot exceed upper bound.")

        def _check(value: float) -> bool:
            if lower is not None and value < lower:
                return False
            if upper is not None and value > upper:
                return False
            return True

        return _check

    # ===================================================
    # Public Methods
    # ===================================================
    def check_bounds(self, value: Union[float, np.ndarray, u.Quantity]) -> bool:
        r"""
        Check whether a value satisfies this parameter's bounds.

        Given a particular ``value`` determine if the value satisfies the bounds
        defined for this parameter. If the parameter has no bounds, this will always
        return ``True``. This is a wrapper around the low-level ``_check_bounds_base``, which
        assumes all inputs are already coerced to the base unit (:attr:`ModelParameter.base_unit`).

        Parameters
        ----------
        value : float or astropy.units.Quantity
            Value to check. If a Quantity is provided, it will be converted
            to base units before checking.

        Returns
        -------
        bool
            ``True`` if the value is valid, ``False`` otherwise.
        """
        base_value = self.to_base_value(value)
        return bool(self._check_bounds_base(base_value))

    def to_base_value(self, value: Union[float, u.Quantity]) -> float:
        """Convert a value into base units (float)."""
        # Catch the unitless case.
        if self.base_units == u.dimensionless_unscaled:
            if isinstance(value, u.Quantity):
                return value.value
            return value

        # Otherwise, convert to base units if it's a Quantity.
        if isinstance(value, u.Quantity):
            return value.to(self.base_units).value
        return float(value)

    def to_quantity(self, value: Union[float, u.Quantity]) -> u.Quantity:
        """Convert a value into an astropy Quantity with base units."""
        if isinstance(value, u.Quantity):
            return value.to(self.base_units)
        return value * self.base_units

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"ModelParameter(name='{self.name}', default={self.default_quantity}, bounds={self.bounds})"

    def __str__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other) -> bool:
        if not isinstance(other, ModelParameter):
            return False
        return self.name == other.name

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    @property
    def has_bounds(self) -> bool:
        """Whether this parameter has any effective bounds defined."""
        if callable(self.bounds):
            return True
        if self.bounds is None:
            return False
        lower, upper = self.bounds
        return (lower is not None) or (upper is not None)

    @property
    def is_dimensionless(self) -> bool:
        """Whether this parameter is dimensionless."""
        return self.base_units == u.dimensionless_unscaled


class ModelVariable:
    r"""
    Declarative container for a model independent variable.

    A :class:`ModelVariable` represents an independent variable supplied to a
    Triceratops model (e.g., frequency, time, radius). Variables are *not*
    inferred, do not have defaults, and do not carry bounds. They exist purely
    to describe how model inputs should be interpreted.

    Unlike :class:`ModelParameter`, variables are expected to be provided
    explicitly at evaluation time, typically by observational data.

    Parameters
    ----------
    name : str
        Name of the variable. This must match the keyword expected by the model's
        forward function.
    base_units : str or astropy.units.Unit
        Base units assumed by the model's internal (low-level) physics
        implementation.
    description : str, optional
        Human-readable description of the variable.
    latex : str, optional
        LaTeX representation of the variable symbol (e.g. ``r"$\nu$"``).

    Notes
    -----
    - Variables are not validated for bounds or physical plausibility.
    - Variables may be scalar or array-like; broadcasting behavior is defined
      by the model, not the variable.
    - Unit coercion is handled here for consistency with model internals.
    """

    # ===================================================
    # Initialization
    # ===================================================
    def __init__(
        self,
        name: str,
        *,
        base_units: Union[str, u.Unit],
        description: str = "",
        latex: str = "",
    ):
        self.name: str = name
        """str: Name of the variable."""

        self.description: str = description
        """str: Human-readable description of the variable."""

        self.latex: str = latex
        """str: LaTeX representation of the variable symbol."""

        self.base_units: u.Unit = self._parse_base_units(base_units)
        """astropy.units.Unit: Base units for this variable."""

    # ===================================================
    # Initialization helpers
    # ===================================================
    def _parse_base_units(self, base_units: Union[str, u.Unit]) -> u.Unit:
        """Parse and validate base units."""
        if isinstance(base_units, str):
            try:
                return u.Unit(base_units)
            except Exception as exc:
                raise ValueError(f"Invalid unit string '{base_units}' for variable '{self.name}'.") from exc
        elif isinstance(base_units, u.UnitBase):
            return base_units
        elif base_units is None:
            return u.Unit("")
        else:
            raise TypeError(f"base_units must be a str or astropy Unit, got {type(base_units)}.")

    # ===================================================
    # Public helpers
    # ===================================================
    def to_base_value(self, value: Union[float, np.ndarray, u.Quantity]):
        """
        Convert a value into base units (float or ndarray).

        Parameters
        ----------
        value : float, array-like, or astropy.units.Quantity
            Value to convert.

        Returns
        -------
        float or numpy.ndarray
            Value expressed in base units.
        """
        # Catch the unitless case.
        if self.base_units == u.dimensionless_unscaled:
            if isinstance(value, u.Quantity):
                return value.value
            return value

        # Otherwise, convert to base units if it's a Quantity.
        if isinstance(value, u.Quantity):
            return value.to(self.base_units).value

        return value

    def to_quantity(self, value: Union[float, np.ndarray, u.Quantity]) -> u.Quantity:
        """
        Convert a value into an astropy Quantity with base units.

        Parameters
        ----------
        value : float, array-like, or astropy.units.Quantity
            Value to convert.

        Returns
        -------
        astropy.units.Quantity
            Value expressed as a Quantity in base units.
        """
        if isinstance(value, u.Quantity):
            return value.to(self.base_units)
        return value * self.base_units

    # ===================================================
    # Dunder methods
    # ===================================================
    def __repr__(self) -> str:
        return f"ModelVariable(name='{self.name}', base_units='{self.base_units}')"

    def __str__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other) -> bool:
        if not isinstance(other, ModelVariable):
            return False
        return self.name == other.name

    # ===================================================
    # Introspection
    # ===================================================
    @property
    def is_dimensionless(self) -> bool:
        """Whether this variable is dimensionless."""
        return self.base_units == u.dimensionless_unscaled
