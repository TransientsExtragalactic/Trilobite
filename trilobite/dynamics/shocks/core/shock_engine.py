"""
Shock engine module for shock computation in astrophysical transients.

The :class:`ShockEngine` is an abstract base class which encapsulates all of the necessary
physical tooling to compute the dynamical evolution of shocks in astrophysical transients,
including supernovae and gamma-ray bursts. Specific implementations of shock engines can be
found in submodules of the :mod:`trilobite.dynamics` and will vary in their structure.

These shock engines can then be instantiated by :class:`~trilobite.models.core.base.Model` classes to provide
an engine for forward modeling of astrophysical transients.
"""

from abc import ABC, abstractmethod
from typing import Union

from trilobite._typing import _UnitBearingArrayLike


class ShockEngine(ABC):
    """
    Base class for shock engine implementations.

    A :class:`ShockEngine` encapsulates the necessary physical tooling to compute the dynamics
    of a transient shock given a set of input parameters. In the most generic shock engine, these properties
    are the shock radius :math:`R(t)` and the shock velocity :math:`v(t)`; however, subclasses may add
    additional / different properties as necessary. Specific subclasses may implement various
    dynamical prescriptions from self-similar models to numerical hydrodynamics.

    The central callable method is :meth:`_compute_shock_properties_cgs(self, time, **parameters)`, which
    should compute the shock properties in CGS units given a time and a set of parameters. This method is
    intended to be called internally by :meth:`compute_shock_properties(self, time, **parameters)`, which may handle
    unit conversions and other pre- and post-processing steps while also leaving the core numerical method available
    for performance-critical code paths (i.e. inside JIT-compiled functions).

    The :class:`ShockEngine` is designed to be stateless, meaning that it does not maintain any internal
    parameter state between calls. Instead, all necessary parameters must be provided as inputs to its methods.
    This design allows for easy integration with various modeling frameworks and ensures that the
    shock engine can be reused across different models without unintended side effects. Instantiations may
    still cache intermediate results for performance optimization, but these caches should not affect the
    external behavior of the engine.

    .. note::

        The :class:`ShockEngine` is *extremely flexible* in its design. The hope is that, by introducing
        this abstraction layer, users and developers can implement a wide variety of shock dynamics
        models without needing to modify the core modeling framework. This could include, but is not limited to:

        - Self-similar solutions (e.g., Sedov-Taylor, Chevalier)
        - Numerical hydrodynamics solvers
        - Semi-analytic models
        - Machine learning-based emulators

        all of which have very different internal structures but can be interfaced in a consistent manner via
        the :class:`ShockEngine` abstraction.
    """

    _STATE_CLASS: "Union[type, None]" = None

    # =========================================== #
    # Initialization                              #
    # =========================================== #
    # The shock engine is a state-less object which simply provides methods for computing
    # shock properties given input parameters and (if necessary) caching of intermediate
    # results. The __init__ MUST reflect this design choice when subclassed. It IS NOT
    # re-initialized on every call to its host Model's forward call.
    @abstractmethod
    def __init__(self, **kwargs):
        """
        Instantiate the shock engine instance.

        Parameters
        ----------
        kwargs
            Parameters to be used for initializing the shock engine. These may vary
            depending on the specific implementation and its requirements. By default, the
            parameters are not used.
        """
        # DEVELOPER NOTE: ``**kwargs`` here should be instance-level parameters for things like
        # the cache size, numerical tolerances, etc. These are NOT the physical parameters
        # of the shock model itself, which should be passed to the compute methods. IN NO CASES
        # should the ShockEngine host a physical state as this would require replicate instantiation
        # in core modeling calls.
        pass

    # =========================================== #
    # CORE NUMERICAL METHODS                      #
    # =========================================== #
    @abstractmethod
    def compute_shock_properties(self, time: _UnitBearingArrayLike, **parameters) -> dict[str, _UnitBearingArrayLike]:
        """
        Evaluate the shock properties at a given dynamical time.

        This method is a high-level wrapper around the low-level ``._compute_shock_properties_cgs``, which handles
        additional verification and unit handling relative to the low-level API. The exact details of the computation
        are specific to the subclass implementing the dynamics.

        Parameters
        ----------
        time: float, astropy.units.Quantity, np.ndarray
            An array-like object representing the time at which to evaluate the shock properties. If
            ``time`` possesses units, they will be accounted for. Otherwise, CGS units are assumed and
            the ``time`` is treated in seconds. If a vector quantity is passed, it will be broadcast following
            standard numpy conventions.
        parameters:
            Shock parameters necessary for evaluating the dynamics. These are made
            explicit in subclass implementations.

        Returns
        -------
        dict of str, float, astropy.units.Quantity, or numpy.ndarray
            The results of the computation. The various calculated quantities are encapsulated in the
            returned dictionary with specific keys.
        """
        pass

    @abstractmethod
    def _compute_shock_properties_cgs(self, time: float, **parameters):
        pass

    # =========================================== #
    # DUNDER METHODS                              #
    # =========================================== #
    def __str__(self):
        return "ShockEngine()"

    def __repr__(self):
        return "<ShockEngine()>"

    def __call__(self, time: _UnitBearingArrayLike, **parameters) -> dict[str, _UnitBearingArrayLike]:
        """
        Evaluate the shock properties at a given dynamical time.

        This is an alias to the :meth:`compute_shock_properties` function.

        Parameters
        ----------
        time: float, astropy.units.Quantity, np.ndarray
            An array-like object representing the time at which to evaluate the shock properties. If
            ``time`` possesses units, they will be accounted for. Otherwise, CGS units are assumed and
            the ``time`` is treated in seconds. If a vector quantity is passed, it will be broadcast following
            standard numpy conventions.
        parameters:
            Shock parameters necessary for evaluating the dynamics. These are made
            explicit in subclass implementations.

        Returns
        -------
        dict of str, float, astropy.units.Quantity, or numpy.ndarray
            The results of the computation. The various calculated quantities are encapsulated in the
            returned dictionary with specific keys.
        """
        return self.compute_shock_properties(time, **parameters)
