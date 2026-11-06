"""
Base classes for dynamical profiles.

This module defines the shared interface for profile objects used by the
dynamics package. A dynamical profile represents a scalar or vector field that
can be evaluated as a function of radius and time given a set of physical
parameters.

These profiles can then be used for various purposes throughout the dynamical
code base where they are needed.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar, Optional

import numpy as np
from astropy import units as u

from trilobite.utils.misc_utils import ensure_in_units

if TYPE_CHECKING:
    from trilobite._typing import _UnitBearingArrayLike


class _DynamicalProfile(ABC):
    r"""
    Abstract base class for parameterized dynamical profiles.

    A dynamical profile is a callable-like object representing a physical field
    such as density, velocity, pressure, or another source term. All concrete
    profiles are **stateless**: no parameters are stored on the class or its
    instances. Instead, parameters are passed at evaluation time, making the
    class suitable for inference and MCMC workflows where the same profile form
    is evaluated many times with different parameter values.

    Subclasses must implement two classmethods:

    ``_validate_and_process_parameters``
        Validate physical parameters, convert unit-bearing inputs to CGS, and
        return a clean dict of unit-free values ready for ``_opt_eval``.

    ``_opt_eval``
        Evaluate the profile using already-processed, unit-free inputs.

    Subclasses may also override:

    ``_validate_and_process_arguments``
        Validate the profile's independent variables (default: ``r``, ``t``
        converted to cm and s).

    Notes
    -----
    The distinction between ``eval`` and ``feval`` is important:

    - ``eval`` is safe and user-facing: validates inputs, converts units,
      attaches ``OUTPUT_UNITS`` to the result.
    - ``feval`` is fast and solver-facing: calls ``_opt_eval`` directly with
      no validation, unit conversion, or output-unit attachment. Use it only
      when inputs are already in the processed CGS form expected by the concrete
      profile.

    ``as_callable`` and ``as_optimized_callable`` freeze profile parameters at
    construction time, returning lightweight closures suitable for passing to
    ODE right-hand sides and other hot loops.

    See Also
    --------
    ~trilobite.dynamics.profiles.csm.CSMDensityProfile : ABC for circumstellar-medium density fields.
    ~trilobite.dynamics.profiles.ejecta.EjectaDensityProfile : ABC for homologous ejecta density fields.
    ~trilobite.dynamics.profiles.velocity.VelocityProfile : ABC for upstream velocity fields.
    """

    OUTPUT_UNITS: ClassVar[Optional[u.Unit]] = None
    """~astropy.units.Unit or None: Physical units of the profile output.
    If ``None``, the profile is dimensionless and ``eval`` returns a bare array.
    """

    # ------------------------------------------------------------------ #
    # Argument and Parameter Processing                                  #
    # ------------------------------------------------------------------ #
    @classmethod
    def _validate_and_process_arguments(
        cls,
        r: "_UnitBearingArrayLike",
        t: "_UnitBearingArrayLike",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Validate and convert the independent variables to CGS.

        The default implementation converts ``r`` to cm and ``t`` to s. Subclasses
        with a different argument signature should override this method.

        Parameters
        ----------
        r : float, array-like, or ~astropy.units.Quantity
            Radius. Unit-bearing inputs are converted to cm; bare values are
            interpreted as cm.
        t : float, array-like, or ~astropy.units.Quantity
            Time. Unit-bearing inputs are converted to s; bare values are
            interpreted as s.

        Returns
        -------
        r_cgs : ~numpy.ndarray
            Radius in cm.
        t_cgs : ~numpy.ndarray
            Time in s.
        """
        r_cgs = np.asarray(ensure_in_units(r, u.cm))
        t_cgs = np.asarray(ensure_in_units(t, u.s))
        return r_cgs, t_cgs

    @classmethod
    @abstractmethod
    def _validate_and_process_parameters(cls, **parameters: Any) -> dict[str, Any]:
        """
        Validate and convert profile parameters to CGS.

        Subclasses should use this method to:

        - check that all required parameters were provided,
        - validate physical constraints (e.g., positivity),
        - convert unit-bearing quantities to CGS values,
        - precompute derived quantities needed by ``_opt_eval``.

        Parameters
        ----------
        **parameters
            User-provided physical parameters.

        Returns
        -------
        processed_parameters : dict
            Unit-free processed parameters suitable for passing directly to
            :meth:`_opt_eval`.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Low-Level Evaluation                                               #
    # ------------------------------------------------------------------ #
    @classmethod
    @abstractmethod
    def _opt_eval(cls, *args: Any, **parameters: Any) -> Any:
        """
        Evaluate the profile using processed, unit-free inputs.

        This is the low-level numerical implementation. It should assume that
        all arguments and parameters have already been validated and converted to
        the expected unit-free CGS representation.

        Parameters
        ----------
        *args
            Processed independent variables (typically ``r_cgs``, ``t_cgs``).
        **parameters
            Processed physical parameters or derived quantities.

        Returns
        -------
        ~numpy.ndarray
            Unit-free profile value in the units represented by
            ``OUTPUT_UNITS``.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Public Evaluation                                                  #
    # ------------------------------------------------------------------ #
    @classmethod
    def eval(cls, *args: Any, **parameters: Any) -> Any:
        """
        Evaluate the profile with input validation, unit conversion, and output units.

        This is the safe user-facing evaluation path. Arguments are validated and
        converted to CGS, parameters are validated and processed, the optimized
        numerical evaluator is called, and ``OUTPUT_UNITS`` is attached when defined.

        Parameters
        ----------
        *args
            Independent variables (typically ``r`` and ``t``). Unit-bearing
            inputs are converted to CGS by :meth:`_validate_and_process_arguments`.
        **parameters
            Physical parameters required by the profile.

        Returns
        -------
        ~numpy.ndarray or ~astropy.units.Quantity
            Profile value. If ``OUTPUT_UNITS`` is not ``None``, the result is
            an :class:`~astropy.units.Quantity`; otherwise, a bare array or scalar.
        """
        processed_args = cls._validate_and_process_arguments(*args)
        processed_params = cls._validate_and_process_parameters(**parameters)
        result = cls._opt_eval(*processed_args, **processed_params)

        if cls.OUTPUT_UNITS is None:
            return result
        return result * cls.OUTPUT_UNITS

    @classmethod
    def feval(cls, *args: Any, **parameters: Any) -> Any:
        """
        Fast unit-free profile evaluation.

        This method calls :meth:`_opt_eval` directly. No validation, unit
        conversion, or output-unit attachment is performed. It is intended for
        performance-sensitive solver code where all inputs have already been
        converted to the processed CGS form expected by the concrete profile.

        Parameters
        ----------
        *args
            Already-processed independent variables in CGS.
        **parameters
            Already-processed physical parameters in CGS.

        Returns
        -------
        ~numpy.ndarray
            Raw unit-free profile value.
        """
        return cls._opt_eval(*args, **parameters)

    # ------------------------------------------------------------------ #
    # Public Parameter Processing                                        #
    # ------------------------------------------------------------------ #
    @classmethod
    def process_arguments(cls, *args: Any) -> tuple:
        """
        Public wrapper for argument validation and CGS conversion.

        Useful when a caller wants to process arguments once and then reuse
        the processed values in repeated fast evaluations via :meth:`feval`.

        Parameters
        ----------
        *args
            User-provided independent variables.

        Returns
        -------
        processed_args : tuple
            Processed unit-free arguments.
        """
        return cls._validate_and_process_arguments(*args)

    @classmethod
    def process_parameters(cls, **parameters: Any) -> dict[str, Any]:
        """
        Public wrapper for parameter validation and CGS conversion.

        Useful for inference workflows where model parameters change between
        likelihood evaluations but should be pre-converted before entering an
        ODE solver or hot loop.

        Parameters
        ----------
        **parameters
            User-provided physical parameters.

        Returns
        -------
        processed_parameters : dict
            Processed unit-free parameters suitable for :meth:`feval`.
        """
        return cls._validate_and_process_parameters(**parameters)

    # ------------------------------------------------------------------ #
    # Frozen Callables                                                   #
    # ------------------------------------------------------------------ #
    @classmethod
    def as_callable(cls, **parameters: Any) -> Callable[..., Any]:
        """
        Return a unit-aware callable with fixed profile parameters.

        Parameters are validated and processed once when this method is called.
        The returned callable accepts the profile's independent variables
        (``r``, ``t``), applies the usual argument validation, and returns the
        result with ``OUTPUT_UNITS`` attached.

        Parameters
        ----------
        **parameters
            Physical parameters to freeze into the callable.

        Returns
        -------
        profile_callable : callable
            Callable accepting ``(r, t)`` (or the profile's native argument
            signature) and returning the profile value with units.
        """
        processed_params = cls._validate_and_process_parameters(**parameters)

        def _frozen(*args: Any) -> Any:
            processed_args = cls._validate_and_process_arguments(*args)
            result = cls._opt_eval(*processed_args, **processed_params)
            if cls.OUTPUT_UNITS is None:
                return result
            return result * cls.OUTPUT_UNITS

        return _frozen

    @classmethod
    def as_optimized_callable(cls, **parameters: Any) -> Callable[..., Any]:
        """
        Return a fast unit-free callable with fixed profile parameters.

        Parameters are validated and processed once when this method is called.
        The returned callable calls :meth:`_opt_eval` directly with no argument
        validation, unit conversion, or output-unit attachment.

        This is the appropriate callable to pass into ODE right-hand sides or
        other hot-loop solver code that already works in CGS units.

        Parameters
        ----------
        **parameters
            Physical parameters to freeze into the callable.

        Returns
        -------
        profile_callable : callable
            Callable accepting already-processed unit-free independent variables
            and returning a raw unit-free result.
        """
        processed_params = cls._validate_and_process_parameters(**parameters)

        def _frozen(*args: Any) -> Any:
            return cls._opt_eval(*args, **processed_params)

        return _frozen
