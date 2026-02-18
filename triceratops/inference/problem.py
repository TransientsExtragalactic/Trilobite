"""
Abstract base class for inference problem definitions.

This module defines the :class:`InferenceProblem` abstract base class, which serves as a
template for defining specific inference problems in the Triceratops library. It takes, as input, a
:class:`~inference.likelihood.base.Likelihood` instance and allows the user to handling things like fixed and
free parameters, priors, and other problem-specific settings.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
from astropy import units as u

from triceratops.utils.misc_utils import ensure_in_units

from .prior import (
    HalfNormalPrior,
    LogNormalPrior,
    LogUniformPrior,
    NormalPrior,
    UniformPrior,
)

# Handle type checking and special aliases.
if TYPE_CHECKING:
    from triceratops._typing import _UnitBearingScalarLike
    from triceratops.inference.likelihood.base import Likelihood
    from triceratops.models._typing import _ModelParametersInput
    from triceratops.models.core.base import Model
    from triceratops.models.core.parameters import BoundType, ModelParameter

    from .prior import Prior


__all__ = [
    "InferenceProblem",
    "InferenceParameter",
]


# ------------------------------------------------- #
# INFERENCE PARAMETER CLASS                         #
# ------------------------------------------------- #
# A simple dataclass to hold information about each parameter.
@dataclass
class InferenceParameter:
    """
    Dataclass to hold information about an inference parameter.

    When an :class:`InferenceProblem` is initialized, it creates an :class:`InferenceParameter`
    object for each of the parameters defined in the corresponding :class:`~models.core.base.Model` provided
    as part of the input :class:`~inference.likelihood.base.Likelihood` object.

    Parameters
    ----------
    model_parameter: `~models.core.parameters.ModelParameter`
        The model parameter associated with this inference parameter. This is where we
        get the name, default value, bounds, and other metadata about the parameter. This should be
        selected from the model's :attr:`~models.core.base.Model.PARAMETERS` list. Because :class:`InferenceParameter`
        instances are generated automatically during :class:`InferenceProblem` initialization, users typically
        do not need to create them manually.
    freeze: bool, optional
        Whether to freeze this parameter during inference. If `True`, the parameter
        will not be varied. Default is `False`.
    prior: Callable, optional
        A callable representing the prior distribution for this parameter. If not provided,
        a default prior based on the parameter's bounds will be used. If a prior is provided, it
        should be a callable which takes a single float argument (the parameter value) and returns
        the log-prior probability (a float).

        .. code-block:: python

            def prior(x: float) -> float:

    transform: Callable, optional
        A callable to transform the parameter value (e.g., log-transform). If not provided,
        no transformation will be applied. If a transform is provided, an `inverse_transform` should
        also be provided.
    inverse_transform: Callable, optional
        A callable to inverse-transform the parameter value back to its original scale. If not provided,
        no inverse transformation will be applied. If `inverse_transform` is provided, a `transform` should
        also be provided.
    transform_jacobian: Callable, optional
        A callable to compute the Jacobian of the transform, if needed. This is useful for
        certain sampling algorithms that require knowledge of the transformation's Jacobian. If a transform
        is provided, this should also be provided. Likewise, if no transform is provided, this should be `None`.

        The transform jacobian should be provided as a callable which takes a single float argument (the parameter
        value) and returns the log-Jacobian (a float).
    initial_value: float, optional
        An initial value for the parameter to start inference from. If not provided, the model's default
        value will be used. When performing MCMC, chains will be distributed around this initial value.
    """

    model_parameter: "ModelParameter"
    """`~models.core.parameters.ModelParameter`: The model parameter associated with this inference parameter.

    This is where we get the name, default value, bounds, and other metadata about the parameter.
    """
    freeze: bool = False
    """bool: Whether to freeze this parameter during inference."""
    prior: "Prior" = None
    """~inference.prior.Prior: A callable representing the prior distribution for this parameter."""
    transform: Callable = None
    """Callable: A callable to transform the parameter value (e.g., log-transform)."""
    inverse_transform: Callable = None
    """Callable: A callable to inverse-transform the parameter value back to its original scale."""
    transform_jacobian: Callable = None
    """Callable: A callable to compute the Jacobian of the transform, if needed."""

    # --- Private Attributes --- #
    _initial_value: float = None
    """float: An initial value for the parameter to start inference from."""

    # ------------------------------------------------- #
    # Initialization and Post-Initialization            #
    # ------------------------------------------------- #
    def __init__(
        self,
        model_parameter: "ModelParameter",
        freeze: bool = False,
        prior: "Prior" = None,
        transform: Callable = None,
        inverse_transform: Callable = None,
        transform_jacobian: Callable = None,
        initial_value: float = None,
    ):
        # Set all of the attributes.
        self.model_parameter = model_parameter
        self.freeze = freeze
        self.prior = prior
        self.transform = transform
        self.inverse_transform = inverse_transform
        self.transform_jacobian = transform_jacobian

        # Handle the initial value so that it (a) defaults
        # to the model parameter's default and (b) performs
        # unit coercion to the model's base unit.
        if initial_value is None:
            self._initial_value = model_parameter.default
        else:
            try:
                self._initial_value = ensure_in_units(initial_value, model_parameter.base_units)
            except Exception as exp:
                raise ValueError(
                    f"Initial value for parameter '{model_parameter.name}' could not be "
                    f"coerced to required units '{model_parameter.base_units}': {exp}"
                ) from exp

        # Now handle the post-initialization checks for
        # the transforms.
        # Ensure that the transform and inverse_transform are both provided or both None.
        if (self.transform is None) != (self.inverse_transform is None):
            raise ValueError("Both 'transform' and 'inverse_transform' must be provided together.")

    @property
    def name(self) -> str:
        """str: The name of the parameter."""
        return self.model_parameter.name

    @property
    def bounds(self) -> "BoundType":
        """Tuple or callable: The bounds of the parameter."""
        return self.model_parameter.bounds

    @property
    def model_default(self):
        """The default value of the parameter from the model."""
        return self.model_parameter.default

    @property
    def model_base_unit(self) -> u.Unit:
        """`astropy.units.Unit`: The base unit of the parameter from the model."""
        return self.model_parameter.base_units

    @property
    def initial_value(self) -> float:
        """float: The initial value for the parameter to start inference from.

        This is in the model's base units.
        """
        return self._initial_value

    @initial_value.setter
    def initial_value(self, value: "_UnitBearingScalarLike"):
        """
        Set the initial value of this parameter.

        Parameters
        ----------
        value: float or `astropy.units.Quantity` or array-like
            The new initial value for the parameter. This will be coerced to the model's base
            units.

        Returns
        -------
        None
        """
        try:
            self._initial_value = ensure_in_units(value, self.model_base_unit)
        except Exception as exp:
            raise ValueError(
                f"Initial value for parameter '{self.name}' could not be "
                f"coerced to required units '{self.model_base_unit}': {exp}"
            ) from exp

    @property
    def initial_quantity(self) -> u.Quantity:
        """`astropy.units.Quantity`: The initial value as an Astropy Quantity."""
        return self.initial_value * self.model_base_unit

    # ------------------------------------------------- #
    # Serialization Methods                             #
    # ------------------------------------------------- #
    @staticmethod
    def _callable_repr(func: Optional[Callable]) -> Optional[str]:
        """Return a safe, non-reconstructive representation of a callable."""
        if func is None:
            return None
        return repr(func)

    def to_metadata(self) -> dict:
        """
        Serialize recoverable metadata for this inference parameter.

        Returns
        -------
        dict
            Metadata dictionary suitable for storage in HDF5 / JSON.

        Notes
        -----
        This does NOT allow full reconstruction. The owning
        ``InferenceProblem`` must supply the ``ModelParameter``.
        """
        return {
            "name": self.name,
            "freeze": self.freeze,
            "initial_value": self.initial_value,
            "prior": self.prior.to_dict() if self.prior else None,
            "transform": self._callable_repr(self.transform),
            "inverse_transform": self._callable_repr(self.inverse_transform),
            "transform_jacobian": self._callable_repr(self.transform_jacobian),
        }

    @classmethod
    def from_metadata(cls, metadata: dict) -> dict:
        """
        Construct a format compliant metadata dictionary from serialized metadata.

        Parameters
        ----------
        metadata : dict
            Metadata dictionary produced by ``to_metadata``.

        Returns
        -------
        dict
            Parsed metadata suitable for reconstruction by InferenceProblem.
        """
        data = dict(metadata)

        if data.get("prior") is not None:
            data["prior"] = Prior.from_dict(data["prior"])

        # NOTE:
        # Transforms are NOT reconstructed — metadata only.
        return data

    def to_metadata_string(self) -> str:
        """Serialize metadata to a JSON string."""
        import json

        return json.dumps(self.to_metadata())

    @classmethod
    def from_metadata_string(cls, metadata_string: str) -> dict:
        """Deserialize metadata from a JSON string."""
        import json

        return cls.from_metadata(json.loads(metadata_string))


# ------------------------------------------------- #
# INFERENCE PROBLEM BASE CLASS                      #
# ------------------------------------------------- #
class InferenceProblem:
    r"""
    Abstract base class representing a statistical inference problem.

    An :class:`InferenceProblem` defines the **glue layer** between a physical
    :class:`~models.core.base.Model`, observational data, and a statistical
    inference engine. It combines a likelihood function with a structured
    parameter management system, enabling consistent evaluation of priors,
    likelihoods, and posteriors across a wide range of inference scenarios.

    This class is intentionally **model-agnostic** and **sampler-agnostic**.
    It does not implement any specific inference algorithm (e.g., MCMC, nested
    sampling), but instead provides a standardized interface and bookkeeping
    layer that samplers can rely on.

    Core responsibilities
    ---------------------
    - **Parameter management**
        Each model parameter is wrapped in an :class:`InferenceParameter`,
        which tracks whether the parameter is free or fixed, its prior,
        optional transformations, and initial value. Parameters retain a
        deterministic ordering inherited from the underlying model to ensure
        unambiguous vector representations.

    - **Prior handling**
        The inference problem evaluates log-priors for free parameters,
        including optional transformations and Jacobian corrections. Convenience
        helpers are provided to attach common parametric priors in a unit-aware
        manner.

    - **Likelihood evaluation**
        The inference problem delegates likelihood computation to the associated
        :class:`~inference.likelihood.base.Likelihood` object, which encapsulates
        the model–data comparison logic.

    - **Posterior evaluation**
        The log-posterior is computed as the sum of the log-prior and
        log-likelihood, with appropriate short-circuiting for invalid parameter
        regions.

    - **Packing and unpacking**
        Utilities are provided to pack parameter dictionaries into NumPy arrays
        (and vice versa), both for full parameter sets and for free parameters
        only. This enables seamless integration with numerical optimizers and
        samplers.

    - **Validation**
        A comprehensive validation routine ensures that the inference problem
        is internally consistent and ready for execution before sampling begins.

    Design philosophy
    -----------------
    The :class:`InferenceProblem` enforces a clear separation of concerns:

    - **Models** define physical parameters and predictions.
    - **Likelihoods** define how predictions are compared to data.
    - **Inference problems** define *which parameters are varied*, *how they are
      constrained*, and *how probabilities are computed*.
    - **Samplers** explore the resulting posterior distribution.

    By isolating parameter bookkeeping and probability evaluation in this class,
    Triceratops ensures that inference logic remains transparent, reproducible,
    and extensible.

    Intended usage
    --------------
    Users typically do not instantiate :class:`InferenceProblem` directly.
    Instead, concrete subclasses may:

    - Customize parameter freezing or transformations
    - Define problem-specific validation logic
    - Expose convenience constructors for common inference setups

    Samplers and optimizers interact with instances of this class by calling
    them as functions (to evaluate the log-posterior) or by using the provided
    ``log_prior``, ``log_likelihood``, and ``log_posterior`` interfaces.

    Notes
    -----
    - All probability-returning methods return **log-probabilities**.
    - Parameter transformations are assumed to operate in unconstrained
      sampling space; inverse transforms are applied before evaluating priors.
    - This class does not store sampling results; results are handled by
      classes in :mod:`inference.sampling`.

    See Also
    --------
    inference.likelihood.base.Likelihood
        Defines the model–data comparison used by the inference problem.
    inference.prior
        Collection of prior distributions usable with inference parameters.
    inference.sampling
        Samplers that operate on :class:`InferenceProblem` instances.
    """

    # ------------------------------------------------- #
    # Initialization                                    #
    # ------------------------------------------------- #
    # Initialize the InferenceProblem with a Likelihood object.
    def __init__(self, likelihood: "Likelihood", *_, **__):
        """
        Initialize the :class:`InferenceProblem` object.

        Parameters
        ----------
        likelihood: `~inference.likelihood.base.Likelihood`
            The likelihood object that defines the model and data for this inference problem.
        args:
            Additional positional arguments. (Not used in this base class.)
        kwargs:
            Additional keyword arguments. (Not used in this base class.)

        Notes
        -----
        This constructor sets up the inference problem by storing the likelihood,
        generating the parameter dictionary, and initializing the lock state. Further
        initialization can be performed in subclasses.
        """
        # Assign the likelihood to ``self.__likelihood__`` so that we can access the
        # model and data from it. Properties are provided to give easy access to
        # all 3.
        self.__likelihood__ = likelihood

        # Specify an explicit ordering for the parameters based on their order in
        # the underlying model. This ensures that there is no ambiguity about parameter
        # ordering when packing/unpacking parameter vectors.
        self.__parameter_order__ = tuple([p.name for p in self.__likelihood__._model.PARAMETERS])

        # Generate the parameter dictionary with InferenceParameter objects so that
        # we can manage them easily. These are basically empty shells of parameters
        # that we can later fill in with priors, transforms, etc.
        self.__parameters__ = {
            model_parameter.name: InferenceParameter(
                model_parameter=model_parameter,
                freeze=False,
                prior=None,
                transform=None,
                inverse_transform=None,
                initial_value=None,
            )
            for model_parameter in self.__likelihood__._model.PARAMETERS
        }

        # Further initialization can be performed in subclasses.

    # ------------------------------------------------- #
    # Dunder Methods                                    #
    # ------------------------------------------------- #
    def __call__(self, theta: np.ndarray) -> float:
        """
        Evaluate the log-posterior at the given parameter vector.

        Parameters
        ----------
        theta: np.ndarray
            Array of free parameters in the order defined by the model.

        Returns
        -------
        float
            The computed log-posterior value.
        """
        return self._log_posterior(theta)

    def __len__(self) -> int:
        """
        Return the number of free parameters in the inference problem.

        Returns
        -------
        int
            The number of free parameters.
        """
        return self.n_free_parameters

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model={self.model.__class__.__name__}, "
            f"n_free={self.n_free_parameters}, "
            f"n_fixed={self.n_fixed_parameters})"
        )

    def __str__(self) -> str:
        return self.summary()

    def __contains__(self, item: str) -> bool:
        """
        Check if a parameter with the given name exists in the inference problem.

        Parameters
        ----------
        item: str
            The name of the parameter to check.

        Returns
        -------
        bool
            `True` if the parameter exists, `False` otherwise.
        """
        return item in self.__parameters__

    def __getitem__(self, item: str) -> "InferenceParameter":
        if item in self.__parameters__:
            return self.__parameters__[item]
        else:
            raise KeyError(f"Parameter '{item}' not found in InferenceProblem.")

    def __iter__(self):
        """Iterate over parameter names."""
        return iter(self.__parameters__)

    def _repr_html_(self) -> str:
        """Rich HTML representation for Jupyter notebooks."""

        def row(cols):
            return "<tr>" + "".join(f"<td>{c}</td>" for c in cols) + "</tr>"

        rows = []

        for name, p in self.parameters.items():
            status = "Free" if not p.freeze else "Fixed"
            prior = p.prior.__class__.__name__ if p.prior else "—"
            rows.append(
                row(
                    [
                        name,
                        status,
                        p.initial_value,
                        prior,
                    ]
                )
            )

        html = f"""
        <div style="font-family: monospace">
          <h3>InferenceProblem</h3>
          <b>Model:</b> {self.model.__class__.__name__}<br>
          <b>Likelihood:</b> {self.likelihood.__class__.__name__}<br>
          <b>Data points:</b> {self.num_data_points}<br>
          <b>Free parameters:</b> {self.n_free_parameters}<br>
          <br>
          <table style="border-collapse: collapse;">
            <thead>
              <tr>
                <th style="border-bottom: 1px solid black;">Parameter</th>
                <th style="border-bottom: 1px solid black;">Status</th>
                <th style="border-bottom: 1px solid black;">Initial Value</th>
                <th style="border-bottom: 1px solid black;">Prior</th>
              </tr>
            </thead>
            <tbody>
              {"".join(rows)}
            </tbody>
          </table>
        </div>
        """
        return html

    def summary(self) -> str:
        """Return a formatted summary of the inference problem."""
        lines = [
            "InferenceProblem",
            f"Model: {self.model.__class__.__name__}",
            f"Likelihood: {self.likelihood.__class__.__name__}",
            "",
            f"Number of data points: {self.num_data_points}",
            f"Free parameters ({self.n_free_parameters}):",
        ]

        for name, p in self.free_parameters.items():
            prior = p.prior.__class__.__name__ if p.prior else "None"
            lines.append(f"  - {name}: init={p.initial_value}, prior={prior}")

        if self.fixed_parameters:
            lines.append("")
            lines.append(f"Fixed parameters ({self.n_fixed_parameters}):")
            for name, p in self.fixed_parameters.items():
                lines.append(f"  - {name}: value={p.initial_value}")

        return "\n".join(lines)

    # ------------------------------------------------- #
    # Basic Access Utilities                            #
    # ------------------------------------------------- #
    def keys(self) -> list[str]:
        """Return a list of parameter names in the inference problem."""
        return list(self.__parameters__.keys())

    def values(self) -> list["InferenceParameter"]:
        """Return a list of inference parameters in the inference problem."""
        return list(self.__parameters__.values())

    def items(self) -> list[tuple[str, "InferenceParameter"]]:
        """Return a list of (name, inference parameter) tuples in the inference problem."""
        return list(self.__parameters__.items())

    # ------------------------------------------------- #
    # Likelihood / Inference Methods                    #
    # ------------------------------------------------- #
    # These methods provide the final log-likelihood, log-prior, and log-posterior calculations
    # by combining the model, data, and parameter information.

    # Packing and unpacking of parameters.
    def pack_parameters(self, parameters: dict[str, float]) -> np.ndarray:
        """
        Pack a dictionary of parameter values into a NumPy array in the model-defined parameter order.

        Parameters
        ----------
        parameters : dict
            Dictionary mapping parameter names to values.

        Returns
        -------
        np.ndarray
            One-dimensional array of parameter values ordered according
            to the model definition.
        """
        # Generate the packed parameter array.
        theta = np.empty(len(self.__parameter_order__), dtype=float)

        # Cycle through the parameters and assign them.
        for i, name in enumerate(self.__parameter_order__):
            try:
                theta[i] = parameters[name]
            except KeyError as e:
                raise KeyError(f"Missing value for parameter '{name}' when packing parameters.") from e

        return theta

    def unpack_parameters(self, theta: np.ndarray) -> dict[str, float]:
        """
        Unpack a NumPy array of parameter values into a dictionary keyed by parameter name.

        Parameters
        ----------
        theta : np.ndarray
            One-dimensional array of parameter values ordered according
            to the model definition.

        Returns
        -------
        dict
            Dictionary mapping parameter names to values.
        """
        theta = np.asarray(theta, dtype=float)

        return {name: theta[i] for i, name in enumerate(self.__parameter_order__)}

    def pack_free_parameters(self, parameters: dict[str, float]) -> np.ndarray:
        """
        Pack only the free parameters into a NumPy array in the model-defined parameter order.

        Parameters
        ----------
        parameters : dict
            Dictionary mapping parameter names to values.

        Returns
        -------
        np.ndarray
            One-dimensional array of free parameter values ordered according
            to the model definition.
        """
        # Generate the packed parameter array for free parameters only.
        theta = np.empty(self.n_free_parameters, dtype=float)

        # Cycle through the parameters and assign them if they are free.
        j = 0
        for name in self.__parameter_order__:
            p = self.__parameters__[name]
            if not p.freeze:
                theta[j] = parameters[name]
                j += 1

        return theta

    def unpack_free_parameters(self, theta_free: np.ndarray) -> dict[str, float]:
        """
        Expand a free-parameter vector into a full parameter dictionary, inserting frozen values where appropriate.

        Parameters
        ----------
        theta_free : np.ndarray
            One-dimensional array of free parameter values ordered according
            to the model definition.

        Returns
        -------
        dict
            Dictionary mapping parameter names to values, including frozen parameters.
        """
        theta_free = np.asarray(theta_free, dtype=float)

        out = {}
        i = 0
        for name in self.__parameter_order__:
            p = self.__parameters__[name]
            if p.freeze:
                out[name] = p.initial_value
            else:
                out[name] = theta_free[i]
                i += 1

        if i != len(theta_free):
            raise ValueError("Mismatch between free parameters and theta_free length.")

        return out

    # Prior callables.
    def _log_prior(self, theta: np.ndarray) -> float:
        """
        Compute the log-prior for a given set of parameters.

        Parameters
        ----------
        theta : np.ndarray
            Array of parameter values in the order defined by the model.

        Returns
        -------
        float
            The computed log-prior value.
        """
        # Declare the log-prior variable.
        logp = 0.0

        # Cycle through each of the parameters and compute the log-prior, adding
        # it to the total.
        for i, name in enumerate(self.__parameter_order__):
            p = self.__parameters__[name]

            # if the parameter is frozen, skip it. If the prior is None, then we can
            # also skip, although this should be caught when preparing for inference since
            # there should be a prior defined for each free parameter.
            if p.freeze or p.prior is None:
                continue

            # Get the parameter value, applying the inverse transform if necessary.
            x = theta[i]
            if p.inverse_transform:
                x = p.inverse_transform(x)

            # Compute the log-prior for this parameter.
            lp = p.prior(x)

            # Check if the log-prior is finite; if not, return -inf immediately.
            if not np.isfinite(lp):
                return -np.inf

            logp += lp

        return logp

    def _log_likelihood(self, theta: np.ndarray) -> float:
        """
        Compute the log-likelihood for a given set of parameters.

        Parameters
        ----------
        theta : np.ndarray
            Array of parameter values in the order defined by the model.

        Returns
        -------
        float
            The computed log-likelihood value.
        """
        # Unpack the parameters into a dictionary.
        params = self.unpack_parameters(theta)

        # Compute the log-likelihood using the likelihood object.
        logl = self.__likelihood__.log_likelihood(params)

        return logl

    def _log_posterior(self, theta: np.ndarray) -> float:
        """
        Compute the log-posterior for a given set of parameters.

        Parameters
        ----------
        theta : np.ndarray
            Array of parameter values in the order defined by the model.

        Returns
        -------
        float
            The computed log-posterior value.
        """
        logp = self._log_prior(theta)
        if not np.isfinite(logp):
            return -np.inf

        logl = self._log_likelihood(theta)
        if not np.isfinite(logl):
            return -np.inf

        return logp + logl

    def _log_free_prior(self, theta_free: np.ndarray) -> float:
        """
        Compute the log-prior for a given set of free parameters.

        Parameters
        ----------
        theta_free : np.ndarray
            Array of free parameter values in the order defined by the model.

        Returns
        -------
        float
            The computed log-prior value.
        """
        # Expand the free parameters into a full parameter vector.
        params = self.unpack_free_parameters(theta_free)
        theta = self.pack_parameters(params)
        return self._log_prior(theta)

    def _log_free_likelihood(self, theta_free: np.ndarray) -> float:
        """
        Compute the log-likelihood for a given set of free parameters.

        Parameters
        ----------
        theta_free : np.ndarray
            Array of free parameter values in the order defined by the model.

        Returns
        -------
        float
            The computed log-likelihood value.
        """
        # Expand the free parameters into a full parameter vector.
        params = self.unpack_free_parameters(theta_free)
        theta = self.pack_parameters(params)
        return self._log_likelihood(theta)

    def _log_free_posterior(self, theta_free: np.ndarray) -> float:
        """
        Compute the log-posterior for a given set of free parameters.

        Parameters
        ----------
        theta_free : np.ndarray
            Array of free parameter values in the order defined by the model.

        Returns
        -------
        float
            The computed log-posterior value.
        """
        # Expand the free parameters into a full parameter vector.
        params = self.unpack_free_parameters(theta_free)
        theta = self.pack_parameters(params)
        return self._log_posterior(theta)

    def log_prior(self, params: "_ModelParametersInput") -> float:
        """
        Compute the log-prior for a given set of parameters.

        Parameters
        ----------
        params : dict
            Dictionary of parameter names and their corresponding values.

        Returns
        -------
        float
            The computed log-prior value.

        Notes
        -----
        This is a public wrapper around the internal :meth:`_log_prior` method which
        handles packing the parameters from a dictionary into the required NumPy array format.
        """
        params = self.__likelihood__._model.coerce_model_parameters(params)
        theta = self.pack_parameters(params)
        return self._log_prior(theta)

    def log_likelihood(self, params: "_ModelParametersInput") -> float:
        """
        Compute the log-likelihood for a given set of parameters.

        Parameters
        ----------
        params : dict
            Dictionary of parameter names and their corresponding values.

        Returns
        -------
        float
            The computed log-likelihood value.

        Notes
        -----
        This is a public wrapper around the internal :meth:`_log_likelihood` method which
        handles packing the parameters from a dictionary into the required NumPy array format.
        """
        params = self.__likelihood__._model.coerce_model_parameters(params)
        theta = self.pack_parameters(params)
        return self._log_likelihood(theta)

    def log_posterior(self, params: "_ModelParametersInput") -> float:
        """
        Compute the log-posterior for a given set of parameters.

        Parameters
        ----------
        params : dict
            Dictionary of parameter names and their corresponding values.

        Returns
        -------
        float
            The computed log-posterior value.

        Notes
        -----
        This is a public wrapper around the internal :meth:`_log_posterior` method which
        handles packing the parameters from a dictionary into the required NumPy array format.
        """
        params = self.__likelihood__._model.coerce_model_parameters(params)
        theta = self.pack_parameters(params)
        return self._log_posterior(theta)

    def log_free_prior(self, params: "_ModelParametersInput") -> float:
        """
        Compute the log-prior for a given set of free parameters.

        Parameters
        ----------
        params : dict
            Dictionary of free parameter names and their corresponding values.

        Returns
        -------
        float
            The computed log-prior value.

        Notes
        -----
        This is a public wrapper around the internal :meth:`_log_free_prior` method which
        handles packing the free parameters from a dictionary into the required NumPy array format.
        """
        params = self.__likelihood__._model.coerce_model_parameters(params)
        theta_free = self.pack_free_parameters(params)
        return self._log_free_prior(theta_free)

    def log_free_likelihood(self, params: "_ModelParametersInput") -> float:
        """
        Compute the log-likelihood for a given set of free parameters.

        Parameters
        ----------
        params : dict
            Dictionary of free parameter names and their corresponding values.

        Returns
        -------
        float
            The computed log-likelihood value.

        Notes
        -----
        This is a public wrapper around the internal :meth:`_log_free_likelihood` method which
        handles packing the free parameters from a dictionary into the required NumPy array format.

        """
        params = self.__likelihood__._model.coerce_model_parameters(params)
        theta_free = self.pack_free_parameters(params)
        return self._log_free_likelihood(theta_free)

    def log_free_posterior(self, params: "_ModelParametersInput") -> float:
        """
        Compute the log-posterior for a given set of free parameters.

        Parameters
        ----------
        params : dict
            Dictionary of free parameter names and their corresponding values.

        Returns
        -------
        float
            The computed log-posterior value.

        Notes
        -----
        This is a public wrapper around the internal :meth:`_log_free_posterior` method which
        handles packing the free parameters from a dictionary into the required NumPy array format.
        """
        params = self.__likelihood__._model.coerce_model_parameters(params)
        theta_free = self.pack_free_parameters(params)
        return self._log_free_posterior(theta_free)

    # ------------------------------------------------- #
    # Validation Methods                                #
    # ------------------------------------------------- #
    def validate_for_inference(self):
        """
        Validate that the inference problem is fully and consistently specified and ready for execution by a sampler.

        This method performs a final, comprehensive validation pass before inference.
        It should be called explicitly by the user or implicitly by sampler interfaces.

        Validation checks include:

        - Every parameter has a unique name
        - Every *free* parameter has a prior defined
        - Parameter transforms and inverse transforms are consistent
        - Initial values are finite and compatible with priors
        - Packed parameter dimensionality is consistent
        - The likelihood can be evaluated at the initial parameter point

        Raises
        ------
        RuntimeError
            If any validation check fails.
        """
        # ------------------------------------------------- #
        # Parameter bookkeeping                             #
        # ------------------------------------------------- #
        # Ensure that we have parameters defined, that they are unique etc.
        if len(self.__parameters__) == 0:
            raise RuntimeError("InferenceProblem has no parameters defined.")

        names = self.__parameter_order__
        if len(set(names)) != len(names):
            raise RuntimeError("Duplicate parameter names detected in inference problem.")

        # ------------------------------------------------- #
        # Validate free / frozen parameters                 #
        # ------------------------------------------------- #
        free_params = self.free_parameters

        if len(free_params) == 0:
            raise RuntimeError("All parameters are frozen. Inference requires at least one free parameter.")

        # ------------------------------------------------- #
        # Validate priors                                   #
        # ------------------------------------------------- #
        for p in free_params.values():
            if p.prior is None:
                raise RuntimeError(f"No prior specified for free parameter '{p.name}'.")

            # Test prior at initial value
            x0 = p.initial_value
            if p.inverse_transform:
                try:
                    x0 = p.inverse_transform(x0)
                except Exception as e:
                    raise RuntimeError(f"Inverse transform failed for parameter '{p.name}': {e}") from e

            try:
                lp = p.prior(x0)
            except Exception as e:
                raise RuntimeError(f"Prior evaluation failed for parameter '{p.name}': {e}") from e

            if not np.isfinite(lp):
                raise RuntimeError(f"Prior for parameter '{p.name}' evaluates to non-finite value at initial point.")

        # ------------------------------------------------- #
        # Validate transforms                               #
        # ------------------------------------------------- #
        for p in free_params.values():
            if p.transform is not None:
                try:
                    y = p.transform(p.initial_value)
                    x = p.inverse_transform(y)
                except Exception as e:
                    raise RuntimeError(f"Transform/inverse_transform failed for parameter '{p.name}': {e}") from e

                if not np.isfinite(x):
                    raise RuntimeError(f"Inverse transform produced non-finite value for '{p.name}'.")

        # ------------------------------------------------- #
        # Validate parameter packing                        #
        # ------------------------------------------------- #
        try:
            theta0 = self.pack_parameters({name: p.initial_value for name, p in self.__parameters__.items()})
        except Exception as e:
            raise RuntimeError(f"Failed to pack initial parameters: {e}") from e

        if not isinstance(theta0, np.ndarray):
            raise RuntimeError("Packed parameters must be returned as a NumPy array.")

        if theta0.ndim != 1:
            raise RuntimeError("Packed parameter vector must be one-dimensional.")

        # ------------------------------------------------- #
        # Validate likelihood evaluation                    #
        # ------------------------------------------------- #
        try:
            logl = self._log_likelihood(theta0)
        except Exception as e:
            raise RuntimeError(f"Likelihood evaluation failed at initial point: {e}") from e

        if not np.isfinite(logl):
            raise RuntimeError("Log-likelihood evaluates to non-finite value at initial parameters.")

        # ------------------------------------------------- #
        # Validate posterior evaluation                     #
        # ------------------------------------------------- #
        try:
            logp = self._log_posterior(theta0)
        except Exception as e:
            raise RuntimeError(f"Posterior evaluation failed at initial point: {e}") from e

        if not np.isfinite(logp):
            raise RuntimeError("Log-posterior evaluates to non-finite value at initial parameters.")

    # ------------------------------------------------- #
    # Properties                                        #
    # ------------------------------------------------- #
    # Properties to access the likelihood, model, data, and parameters.
    @property
    def likelihood(self) -> "Likelihood":
        """`~inference.likelihood.base.Likelihood`: The likelihood associated with this inference problem.

        The :class:`~inference.likelihood.base.Likelihood` object defines the model and data for the problem and,
        most importantly, determines how the model predictions are compared to the data via the likelihood function.
        """
        return self.__likelihood__

    @property
    def model(self) -> "Model":
        """`~models.core.base.Model`: The model associated with this inference problem."""
        return self.__likelihood__._model

    @property
    def data(self):
        """The data associated with this inference problem.

        This data container is that data which is being fit by the problem. It is provided to the likelihood object
        when it is initialized and is used during likelihood evaluations. This may be any number of data container types
        defined in the :mod:`data` module.
        """
        return self.__likelihood__._data_container

    # --- Various Parameter focused properties --- #
    @property
    def parameter_order(self) -> tuple[str, ...]:
        """Tuple of str: The ordered names of the parameters in this inference problem."""
        return self.__parameter_order__

    @property
    def parameters(self) -> dict[str, InferenceParameter]:
        """dict: Dictionary of inference parameters keyed by parameter name."""
        return self.__parameters__

    @property
    def free_parameters(self) -> dict[str, InferenceParameter]:
        """dict: Dictionary of free (unfrozen) inference parameters keyed by parameter name."""
        return {name: p for name, p in self.__parameters__.items() if not p.freeze}

    @property
    def fixed_parameters(self) -> dict[str, InferenceParameter]:
        """dict: Dictionary of fixed (frozen) inference parameters keyed by parameter name."""
        return {name: p for name, p in self.__parameters__.items() if p.freeze}

    @property
    def free_parameter_names(self) -> tuple[str, ...]:
        """Tuple of str: The names of the free (unfrozen) parameters in this inference problem."""
        return tuple(self.free_parameters.keys())

    @property
    def fixed_parameter_names(self) -> tuple[str, ...]:
        """Tuple of str: The names of the fixed (frozen) parameters in this inference problem."""
        return tuple(self.fixed_parameters.keys())

    @property
    def n_parameters(self) -> int:
        """int: The total number of parameters in this inference problem."""
        return len(self.__parameters__)

    @property
    def n_free_parameters(self) -> int:
        """int: The number of free (unfrozen) parameters in this inference problem."""
        return len([p for p in self.__parameters__.values() if not p.freeze])

    @property
    def n_fixed_parameters(self) -> int:
        """int: The number of fixed (frozen) parameters in this inference problem."""
        return len([p for p in self.__parameters__.values() if p.freeze])

    @property
    def n_dim(self) -> int:
        """int: Alias for :attr:`n_free_parameters`."""
        return self.n_free_parameters

    @property
    def num_data_points(self) -> int:
        """int: The number of data points in the likelihood's data."""
        return len(self.data)

    # --- Sampling Utility Properties --- #
    @property
    def initial_theta(self) -> np.ndarray:
        """np.ndarray: The initial parameter vector of free parameters.

        This is the packed array of initial values for all free parameters in the inference problem. It
        is generally the starting point for samplers and optimizers. Note that this will only include free parameters;
        fixed parameters are not included in this vector.
        """
        return self.pack_free_parameters({name: p.initial_value for name, p in self.free_parameters.items()})

    @property
    def initial_full_theta(self) -> np.ndarray:
        """np.ndarray: The initial full parameter vector including fixed parameters.

        This is the packed array of initial values for all parameters in the inference problem, including both
        free and fixed parameters. It can be useful for evaluating the likelihood or posterior at the initial point.
        """
        return self.pack_parameters({name: p.initial_value for name, p in self.__parameters__.items()})

    @property
    def initial_params(self) -> dict[str, float]:
        """dict: The initial parameter dictionary of all parameters.

        This is a dictionary mapping free parameter names to their initial values for all parameters
        in the inference problem.
        """
        return {name: p.initial_value for name, p in self.free_parameters.items()}

    @property
    def initial_full_params(self) -> dict[str, float]:
        """dict: The initial full parameter dictionary including fixed parameters.

        This is a dictionary mapping all parameter names (free and fixed) to their initial values
        in the inference problem.
        """
        return {name: p.initial_value for name, p in self.__parameters__.items()}

    @property
    def initial_log_posterior(self) -> float:
        """float: The log-posterior evaluated at the initial free parameter values."""
        return self._log_free_posterior(self.initial_theta)

    # ------------------------------------------------- #
    # Prior Setting Helpers                             #
    # ------------------------------------------------- #
    # These methods allow us to set up some basic priors quickly without
    # requiring the user to manually define prior callables. We can also do
    # some unit-handled priors this way, which is a nice feature.
    _PRIOR_REGISTRY = {
        "uniform": {
            "class": UniformPrior,
            "unit_params": ("lower", "upper"),
        },
        "loguniform": {
            "class": LogUniformPrior,
            "unit_params": ("lower", "upper"),
        },
        "normal": {
            "class": NormalPrior,
            "unit_params": ("mean", "sigma"),
        },
        "lognormal": {
            "class": LogNormalPrior,
            "unit_params": ("mean", "sigma"),
        },
        "halfnormal": {
            "class": HalfNormalPrior,
            "unit_params": ("sigma",),
        },
    }

    def _coerce_to_base_units(
        self,
        parameter: str,
        values: dict[str, "_UnitBearingScalarLike"],
    ) -> dict[str, float]:
        """Coerce unit-bearing values into the model parameter's base units."""
        p = self.__parameters__.get(parameter)
        if p is None:
            raise ValueError(f"Parameter '{parameter}' not found in inference problem.")

        base_unit = p.model_base_unit

        out = {}
        for name, val in values.items():
            out[name] = u.Quantity(val, unit=base_unit).to_value(base_unit)

        return out

    def set_prior(self, parameter: str, prior_type: str, **parameters):
        """
        Set a prior distribution on a parameter.

        Parameters
        ----------
        parameter : str
            Name of the model parameter.
        prior_type : str
            Type of prior ('uniform', 'loguniform', 'normal', etc.).
        parameters
            Parameters defining the prior.
        """
        prior_type = prior_type.lower()

        if prior_type not in self.__class__._PRIOR_REGISTRY:
            raise ValueError(
                f"Unknown prior type '{prior_type}'. "
                f"Available priors: {list(self.__class__._PRIOR_REGISTRY)}.\n "
                f"Additional priors can be found in the `inference.prior` module."
            )

        entry = self.__class__._PRIOR_REGISTRY[prior_type]
        prior_cls = entry["class"]
        unit_params = entry["unit_params"]

        # Split parameters into unit-bearing and unitless
        unit_values = {k: parameters[k] for k in unit_params if k in parameters}

        # Coerce unit-bearing parameters
        coerced = self._coerce_to_base_units(parameter, unit_values)

        # Merge back
        final_params = dict(parameters)
        final_params.update(coerced)

        # Assign prior
        self.__parameters__[parameter].prior = prior_cls(**final_params)

    # ------------------------------------------------- #
    # Freezing and Unfreezing Parameters                #
    # ------------------------------------------------- #
    def freeze_parameters(self, parameters):
        """
        Freeze (fix) specified parameters in the inference problem.

        Parameters
        ----------
        parameters : list or str
            Parameter name or list of parameter names to freeze.
        """
        if isinstance(parameters, str):
            parameters = [parameters]

        for name in parameters:
            if name not in self.__parameters__:
                raise KeyError(f"Parameter '{name}' not found in inference problem.")
            self.__parameters__[name].freeze = True

    def unfreeze_parameters(self, parameters):
        """
        Unfreeze (free) specified parameters in the inference problem.

        Parameters
        ----------
        parameters : list or str
            Parameter name or list of parameter names to unfreeze.
        """
        if isinstance(parameters, str):
            parameters = [parameters]

        for name in parameters:
            if name not in self.__parameters__:
                raise KeyError(f"Parameter '{name}' not found in inference problem.")
            self.__parameters__[name].freeze = False
