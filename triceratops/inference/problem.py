"""
Abstract base class for inference problem definitions.

This module defines the :class:`InferenceProblem` abstract base class, which serves as a
template for defining specific inference problems in the Triceratops library. It takes, as input, a
:class:`~inference.likelihood.base.Likelihood` instance and allows the user to handling things like fixed and
free parameters, priors, and other problem-specific settings.
"""

import warnings
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Optional, Union

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

# ============================================================= #
# Package versioning and Type checking
# ============================================================= #=
try:
    __pkg_version__ = version("triceratops")
except PackageNotFoundError:
    __pkg_version__ = "unknown"


# Handle type checking and special aliases.
if TYPE_CHECKING:
    from triceratops._typing import _UnitBearingScalarLike
    from triceratops.inference.likelihood.base import Likelihood
    from triceratops.models._typing import _ModelParametersInput
    from triceratops.models.core.base import Model
    from triceratops.models.core.parameters import BoundType, ModelParameter

    from .prior import Prior
    from .transform import ParameterTransform


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
    r"""~inference.prior.Prior: Prior distribution for this parameter.

    The prior defines the log-probability density
    :math:`\log p(\theta)` in **physical parameter space**.

    During inference, the posterior is computed as:

    .. math::

        \log p(\theta \mid \mathcal{D})
        =
        \log \mathcal{L}(\theta)
        +
        \log p(\theta)

    If a :class:`~triceratops.inference.prior.Prior` instance is provided,
    its :meth:`~triceratops.inference.prior.Prior.logp` method is used to
    evaluate the prior contribution.

    If ``None`` (default), a prior will be automatically constructed based
    on the parameter's bounds:

    - If finite bounds are defined, a uniform prior over the bounds is used.
    - If no bounds are defined, the prior defaults to a constant (improper)
      prior unless explicitly overridden.

    Notes
    -----
    - The prior is always evaluated in physical parameter space,
      even if a transformation is applied for sampling.
    - If a transform is present, the log-Jacobian correction is
      applied separately and is **not** part of the prior.
    - Custom priors must be importable and JSON-serializable
      in order for :class:`InferenceProblem` serialization
      to remain fully reconstructible.
    """
    transform: "ParameterTransform" = None
    r"""~inference.transform.ParameterTransform: Optional transformation applied
    to this parameter during sampling.

    If provided, the sampler operates in a transformed space
    :math:`z = f(\theta)` rather than directly in the physical parameter
    space :math:`\theta`.

    The transformation must be a smooth, bijective mapping with a well-defined
    Jacobian determinant. The inference workflow proceeds as:

    1. The sampler proposes a value in sampling space :math:`z`.
    2. The inverse transform is applied:

       .. math::

           \theta = f^{-1}(z)

    3. The likelihood and prior are evaluated in physical parameter space.
    4. The log-Jacobian correction

       .. math::

           \log \left| \frac{d\theta}{dz} \right|

       is added to ensure proper change-of-variables accounting.

    If ``None`` (default), the identity transformation is assumed and the
    parameter is sampled directly in physical space.

    Common use cases include:

    - Enforcing positivity via a log transform
    - Mapping bounded intervals to :math:`\mathbb{R}` via logistic transforms
    - Improving sampling geometry for highly skewed posteriors

    Notes
    -----
    - The prior is always defined in physical parameter space.
    - The transform must be fully importable and JSON-serializable.
    - The transform must be deterministic and reconstructible.
    """

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
        transform: "ParameterTransform" = None,
        initial_value: float = None,
    ):
        # Set all of the attributes.
        self.model_parameter = model_parameter
        self.freeze = freeze
        self.prior = prior
        self.transform = transform

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
    # Serialization
    # ------------------------------------------------- #
    def to_dict(self) -> dict:
        """
        Serialize the :class:`InferenceParameter` to a JSON-safe dictionary.

        This method can be used to transfer an inference parameter either to
        disk or, in multiprocessing scenarios, between processes. The
        resulting dictionary is JSON-serializable, meaning it can be safely passed through
        multiprocessing queues or saved to disk in JSON format.

        .. important::

            If either the prior or transform for a particular parameter are not
            self-consistently serializable (i.e. they are not importable), then
            this method will raise a `TypeError`. This is because the serialization of
            the inference parameter relies on the ability to serialize both the prior and
            transform (if they are present) in a self-consistent way. If either of these
            components cannot be serialized, then the entire inference parameter cannot be
            serialized, and a `TypeError` is raised to indicate this issue.

        Returns
        -------
        dict
            JSON-serializable representation.

        Raises
        ------
        TypeError
            If prior or transform are not serializable.
        """
        import json

        # Serialize prior
        prior_dict = None
        if self.prior is not None:
            try:
                prior_dict = self.prior.to_dict()
            except Exception as exp:
                raise TypeError(
                    f"Failed to serialize prior for parameter '{self.name}'!\n"
                    f"This may indicate that the prior object is not self-consistently serializable...\n"
                    f"Original error: {exp}"
                ) from exp

        # Serialize transform
        transform_dict = None
        if self.transform is not None:
            try:
                transform_dict = self.transform.to_dict()
            except Exception as exp:
                raise TypeError(
                    f"Failed to serialize transform for parameter '{self.name}'!\n"
                    f"This may indicate that the transform object is not self-consistently serializable...\n"
                    f"Original error: {exp}"
                ) from exp

        data = {
            "format": "InferenceParameter",
            "name": self.name,
            "freeze": self.freeze,
            "initial_value": float(self.initial_value),
            "prior": prior_dict,
            "transform": transform_dict,
        }

        # Ensure JSON-safe
        try:
            json.dumps(data)
        except TypeError as exc:
            raise TypeError(f"InferenceParameter '{self.name}' is not JSON-serializable.") from exc

        return data

    @classmethod
    def from_dict(
        cls,
        dict_rep: dict,
        model: "Model",
    ) -> "InferenceParameter":
        """
        Reconstruct an :class:`InferenceParameter` from a serialized dictionary.

        Parameters
        ----------
        dict_rep : dict
            Dictionary produced by :meth:`to_dict`.

        model : Model
            The model instance that owns the corresponding
            :class:`~models.core.parameters.ModelParameter`. This is required in order
            to resolve the parameter definition (name, bounds,
            units, default value, etc.).

        Returns
        -------
        InferenceParameter
            Reconstructed inference parameter.

        Raises
        ------
        TypeError
            If inputs are invalid.
        ValueError
            If dictionary structure is malformed.
        KeyError
            If the referenced parameter does not exist on the model.
        """
        from triceratops.inference.prior import Prior
        from triceratops.inference.transform import ParameterTransform

        if not isinstance(dict_rep, dict):
            raise TypeError("dict_rep must be a dictionary.")

        if dict_rep.get("format") != "InferenceParameter":
            raise ValueError("Dictionary does not represent an InferenceParameter.")

        if model is None:
            raise TypeError("A valid Model instance must be provided.")

        try:
            name = dict_rep["name"]
        except KeyError as exc:
            raise ValueError("Serialized InferenceParameter missing 'name'.") from exc

        # ------------------------------------------------------------------
        # Resolve ModelParameter from model
        # ------------------------------------------------------------------
        try:
            model_parameter = next(p for p in model.PARAMETERS if p.name == name)
        except StopIteration as exc:
            raise KeyError(f"Model does not define a parameter named '{name}'.") from exc

        freeze = bool(dict_rep.get("freeze", False))
        initial_value = dict_rep.get("initial_value", None)

        # ------------------------------------------------------------------
        # Reconstruct prior
        # ------------------------------------------------------------------
        prior_dict = dict_rep.get("prior")
        prior = None
        if prior_dict is not None:
            try:
                prior = Prior.from_dict(prior_dict)
            except Exception as exp:
                raise TypeError(f"Failed to reconstruct prior for parameter '{name}'.") from exp

        # ------------------------------------------------------------------
        # Reconstruct transform
        # ------------------------------------------------------------------
        transform_dict = dict_rep.get("transform")
        transform = None
        if transform_dict is not None:
            try:
                transform = ParameterTransform.from_dict(transform_dict)
            except Exception as exp:
                raise TypeError(f"Failed to reconstruct transform for parameter '{name}'.") from exp

        return cls(
            model_parameter=model_parameter,
            freeze=freeze,
            prior=prior,
            transform=transform,
            initial_value=initial_value,
        )


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

        out = {}

        for i, name in enumerate(self.__parameter_order__):
            p = self.__parameters__[name]
            z = theta[i]

            if p.transform is not None:
                try:
                    x = p.transform.inverse(z)
                except Exception:
                    return None  # caller handles -inf
            else:
                x = z

            out[name] = x

        return out

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

    def transform_theta_forward(
        self,
        theta: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        r"""
        Transform a full physical parameter vector into sampling space.

        Returns
        -------
        z : np.ndarray
            Sampling-space parameter vector.
        log_det : float
            Log absolute determinant of the transformation :math:`|dz/d\theta|`.
        """
        theta = np.asarray(theta, dtype=float)

        if theta.ndim != 1:
            raise ValueError("Parameter vector must be one-dimensional.")

        if len(theta) != self.n_parameters:
            raise ValueError("Parameter vector length mismatch.")

        z = np.empty_like(theta)
        log_det = 0.0

        for i, name in enumerate(self.__parameter_order__):
            p = self.__parameters__[name]
            value = theta[i]

            if p.transform is not None:
                try:
                    z_i = p.transform.forward(value)

                    # convert stored log|dθ/dz| to log|dz/dθ|
                    log_det -= p.transform.log_abs_det_jacobian(z_i)

                except Exception as exc:
                    raise ValueError(f"Forward transform failed for parameter '{name}'.") from exc
            else:
                z_i = value

            z[i] = z_i

        return z, log_det

    def transform_theta_inverse(
        self,
        z: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        r"""
        Convert sampling-space vector z to physical space theta and compute total log-Jacobian.

        Returns
        -------
        theta : np.ndarray
            Physical-space parameter vector.
        log_det : float
            Log absolute determinant :math:`|d\theta/dz|`.
        """
        z = np.asarray(z, dtype=float)

        if z.ndim != 1:
            raise ValueError("Parameter vector must be one-dimensional.")

        if len(z) != self.n_parameters:
            raise ValueError("Parameter vector length mismatch.")

        theta = np.empty_like(z)
        log_det = 0.0

        for i, name in enumerate(self.__parameter_order__):
            p = self.__parameters__[name]
            zi = z[i]

            if p.transform is not None:
                try:
                    theta_i = p.transform.inverse(zi)
                    log_det += p.transform.log_abs_det_jacobian(zi)
                except Exception as exc:
                    raise ValueError(f"Inverse transform failed for parameter '{name}'.") from exc
            else:
                theta_i = zi

            theta[i] = theta_i

        return theta, log_det

    def transform_free_theta_forward(
        self,
        theta_free: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        r"""
        Transform free physical-space parameters into sampling space.

        Returns
        -------
        z_free : np.ndarray
            Free parameters in sampling space.
        log_det : float
            Log absolute determinant of the transformation
            :math:`|dz/d\theta|`.
        """
        theta_free = np.asarray(theta_free, dtype=float)

        if len(theta_free) != self.n_free_parameters:
            raise ValueError("Free parameter vector length mismatch.")

        z_free = np.empty_like(theta_free)
        log_det = 0.0

        j = 0
        for name in self.__parameter_order__:
            p = self.__parameters__[name]

            if not p.freeze:
                value = theta_free[j]

                if p.transform is not None:
                    try:
                        z_free[j] = p.transform.forward(value)

                        # we store log |dθ/dz|
                        log_det -= p.transform.log_abs_det_jacobian(z_free[j])

                    except Exception as exc:
                        raise ValueError(f"Forward transform failed for parameter '{name}'.") from exc
                else:
                    z_free[j] = value

                j += 1

        return z_free, log_det

    def transform_free_theta_inverse(
        self,
        z_free: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        r"""
        Transform free sampling-space parameters into physical space.

        Returns
        -------
        theta_free : np.ndarray
            Free parameters in physical space.
        log_det : float
            Log absolute determinant of the transformation
            :math:`|d\theta/dz|`.
        """
        z_free = np.asarray(z_free, dtype=float)

        if len(z_free) != self.n_free_parameters:
            raise ValueError("Free parameter vector length mismatch.")

        theta_free = np.empty_like(z_free)
        log_det = 0.0

        j = 0
        for name in self.__parameter_order__:
            p = self.__parameters__[name]

            if not p.freeze:
                value = z_free[j]

                if p.transform is not None:
                    try:
                        theta_free[j] = p.transform.inverse(value)

                        log_det += p.transform.log_abs_det_jacobian(value)

                    except Exception as exc:
                        raise ValueError(f"Inverse transform failed for parameter '{name}'.") from exc
                else:
                    theta_free[j] = value

                j += 1

        return theta_free, log_det

    # Prior callables.
    def _log_prior(self, theta: np.ndarray) -> float:
        """Compute the log-prior in sampling space, including Jacobian corrections for transformed parameters."""
        logp = 0.0

        for i, name in enumerate(self.__parameter_order__):
            p = self.__parameters__[name]

            if p.freeze or p.prior is None:
                continue

            lp = p.prior(theta[i])

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
            Array of parameter values in the order defined by the model. These
            will be in the sampling space and therefore must be transformed back
            to physical space before being passed to the likelihood.

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

    def _log_posterior(self, z: np.ndarray) -> float:
        """Compute log-posterior in sampling space."""
        # Convert sampling → physical once
        try:
            theta, log_jac = self.transform_theta_inverse(z)
        except Exception as exp:
            raise ValueError("Inverse transform failed.") from exp

        # Prior in physical space
        logp = self._log_prior(theta)
        if not np.isfinite(logp):
            return -np.inf

        # Likelihood in physical space
        logl = self._log_likelihood(theta)
        if not np.isfinite(logl):
            return -np.inf

        return logp + logl + log_jac

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
                    y = p.transform.forward(p.initial_value)
                    x = p.transform.inverse(y)
                    _ = p.transform.log_abs_det_jacobian(y)
                except Exception as e:
                    raise RuntimeError(f"Transform/inverse_transform failed for parameter '{p.name}': {e}") from e

                if not np.isfinite(x):
                    raise RuntimeError(f"Inverse transform produced non-finite value for '{p.name}'.")

        # ------------------------------------------------- #
        # Validate parameter packing                        #
        # ------------------------------------------------- #
        try:
            theta0 = self.pack_parameters({name: p.initial_value for name, p in self.__parameters__.items()})
            theta0_transformed, _ = self.transform_theta_forward(theta0)
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
            logp = self._log_posterior(theta0_transformed)
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
    def initial_z(self) -> np.ndarray:
        """np.ndarray: The initial parameter vector of free parameters transformed into sampling space.

        This is the initial free parameter vector transformed into sampling space using the defined parameter
        transforms.
        It is generally the starting point for samplers and optimizers that operate in sampling space.
        """
        z, _ = self.transform_free_theta_forward(self.initial_theta)
        return z

    @property
    def initial_full_z(self) -> np.ndarray:
        """np.ndarray: The initial full parameter vector including fixed parameters, transformed into sampling space.

        This is the initial full parameter vector (including both free and fixed parameters) transformed into
        sampling space
        using the defined parameter transforms. It can be useful for evaluating the likelihood or posterior at
        the initial point
        in sampling space.
        """
        z, _ = self.transform_theta_forward(self.initial_full_theta)
        return z

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

    # ------------------------------------------------- #
    # Serialization Methods                             #
    # ------------------------------------------------- #
    def to_dict(self, **extra_metadata) -> dict:
        """
        Serialize the :class:`InferenceProblem` to a fully reconstructible dictionary.

        This method produces a JSON-safe representation of the complete inference
        configuration, including:

        - Model specification
        - Likelihood target
        - Data container
        - All inference parameters (including priors and transforms)
        - Metadata describing the problem state

        The resulting dictionary is guaranteed to be JSON-serializable and may be:

        - Saved to disk
        - Transferred between processes (e.g., MPI workers)
        - Logged for reproducibility
        - Used to exactly reconstruct the inference problem via
          :meth:`InferenceProblem.from_dict`

        Parameters
        ----------
        **extra_metadata
            Optional additional metadata fields to attach to the serialized
            representation. All provided values must be JSON-serializable.

        Returns
        -------
        dict
            JSON-safe representation of the inference problem.

        Raises
        ------
        TypeError
            If any component (model, likelihood, data, priors, transforms,
            or metadata) is not serializable.

        Notes
        -----
        - All priors and transforms must themselves be importable and
          serializable.
        - The model must implement :meth:`to_model_spec`.
        - The data container must implement :meth:`to_dict`.
        - The likelihood must be importable via its fully-qualified module path.
        - The returned dictionary contains a ``format`` field for validation
          during reconstruction.
        - A version tag is embedded to allow compatibility checks during loading.
        """
        import json

        from .likelihood.utils import get_likelihood_target

        # ---------------------------------------------------------
        # Base output template
        # ---------------------------------------------------------
        output = {
            "format": "InferenceProblem",
            "metadata": {},
            "model": None,
            "likelihood": None,
            "data": None,
            "parameters": {},
        }

        # ---------------------------------------------------------
        # Serialize parameters
        # ---------------------------------------------------------
        for name, p in self.__parameters__.items():
            try:
                output["parameters"][name] = p.to_dict()
            except Exception as e:
                raise TypeError(f"Failed to serialize parameter '{name}': {e}") from e

        # ---------------------------------------------------------
        # Serialize likelihood target
        # ---------------------------------------------------------
        try:
            output["likelihood"] = get_likelihood_target(self.__likelihood__)
        except Exception as e:
            raise TypeError(f"Failed to serialize likelihood: {e}") from e

        # ---------------------------------------------------------
        # Serialize model
        # ---------------------------------------------------------
        try:
            output["model"] = self.model.to_model_spec().to_dict()
        except Exception as e:
            raise TypeError(f"Failed to serialize model: {e}") from e

        # ---------------------------------------------------------
        # Serialize data
        # ---------------------------------------------------------
        try:
            output["data"] = self.data.to_dict()
        except Exception as e:
            raise TypeError(f"Failed to serialize data: {e}") from e

        # ---------------------------------------------------------
        # Construct metadata (JSON-safe only)
        # ---------------------------------------------------------
        metadata_block = {
            "num_parameters": self.n_parameters,
            "num_free_parameters": self.n_free_parameters,
            "num_fixed_parameters": self.n_fixed_parameters,
            "num_data_points": self.num_data_points,
            "likelihood_target": output["likelihood"],
            "model_target": output["model"].get("target"),
            "triceratops_version": __pkg_version__,
        }

        # Merge user-provided metadata safely
        if extra_metadata:
            metadata_block.update(extra_metadata)

        output["metadata"] = metadata_block

        # ---------------------------------------------------------
        # Final JSON validation
        # ---------------------------------------------------------
        try:
            json.dumps(output)
        except TypeError as exc:
            raise TypeError("Serialized InferenceProblem contains non-JSON-serializable objects.") from exc

        return output

    @classmethod
    def from_dict(cls, dict_spec: dict) -> "InferenceProblem":
        """
        Reconstruct an :class:`InferenceProblem` from a serialized dictionary.

        This method reverses the operation performed by
        :meth:`InferenceProblem.to_dict` and rebuilds:

        - The model
        - The data container
        - The likelihood
        - All inference parameters (including priors and transforms)

        Reconstruction is deterministic provided that:

        - All referenced classes are importable
        - The serialized objects are compatible with the current package version

        Parameters
        ----------
        dict_spec : dict
            Dictionary produced by :meth:`InferenceProblem.to_dict`.

        Returns
        -------
        InferenceProblem
            Fully reconstructed inference problem instance.

        Raises
        ------
        ValueError
            If required fields are missing or malformed.
        TypeError
            If reconstruction of model, data, likelihood, priors,
            or transforms fails.

        Warnings
        --------
        A :class:`UserWarning` is issued if the serialized object was created
        using a different version of ``triceratops`` than the currently
        installed version.

        Notes
        -----
        - Reconstruction assumes that the model, likelihood, data,
          prior, and transform classes are importable via their
          fully-qualified module paths.
        - The parameter ordering is restored exactly as defined by
          the underlying model.
        - Metadata is preserved but does not affect functional behavior.
        """
        from triceratops.data.core import InferenceData
        from triceratops.models.core.serial import ModelSpec

        from .likelihood.utils import build_likelihood

        # --- Validation ------------------------
        # Check that all of the expected keys are present.
        expected_keys = {"format", "metadata", "model", "likelihood", "data", "parameters"}
        missing_keys = expected_keys - dict_spec.keys()
        if missing_keys:
            raise ValueError(f"Missing keys in InferenceProblem specification: {missing_keys}")

        # Verify the format
        if dict_spec["format"] != "InferenceProblem":
            raise ValueError(f"Invalid format '{dict_spec['format']}' for InferenceProblem specification.")

        # Extract the triceratops version and check it against current. If
        # it is not the same version, we raise a warning.
        spec_version = dict_spec["metadata"].get("triceratops_version")
        if spec_version != __pkg_version__:
            warnings.warn(
                f"Version mismatch: InferenceProblem specification was created with triceratops "
                f"version {spec_version}, "
                f"but current version is {__pkg_version__}. Attempting to load, but be aware that there may be "
                f"incompatibilities or missing features. Consider regenerating the specification with the current"
                f" version.",
                UserWarning,
                stacklevel=2,
            )
        # --- Data Reconstruction ------------------------
        try:
            data = InferenceData.from_dict(dict_spec["data"])
        except Exception as e:
            raise TypeError(f"Failed to reconstruct data from specification: {e}") from e

        # --- Model Reconstruction ------------------------
        try:
            model_spec = ModelSpec.from_dict(dict_spec["model"])
            model = model_spec.build()
        except Exception as e:
            raise TypeError(f"Failed to reconstruct model from specification: {e}") from e

        # --- Likelihood Reconstruction ------------------------
        try:
            likelihood_target = dict_spec["likelihood"]
            likelihood = build_likelihood(likelihood_target, model=model, data=data)
        except Exception as e:
            raise TypeError(f"Failed to reconstruct likelihood from specification: {e}") from e

        # --- Parameter Reconstruction ------------------------
        parameters = {}
        for name, param_spec in dict_spec["parameters"].items():
            try:
                parameters[name] = InferenceParameter.from_dict(param_spec, model)
            except Exception as e:
                raise TypeError(f"Failed to reconstruct parameter '{name}' from specification: {e}") from e

        # --- Construct InferenceProblem ------------------------
        problem = cls(likelihood=likelihood)

        # --- Process each of the parameter -------------------------
        problem.__parameters__ = parameters

        return problem

    def to_json(self, path: Optional[str] = None, **extra_metadata) -> str:
        """
        Serialize the :class:`InferenceProblem` to a JSON string or file.

        This method is a thin wrapper around :meth:`to_dict`. It converts
        the fully reconstructible dictionary representation into a
        JSON-formatted string.

        Parameters
        ----------
        path : str, optional
            If provided, the JSON representation will be written to this file.
            If omitted, the JSON string is returned.
        **extra_metadata
            Optional additional metadata fields passed to :meth:`to_dict`.

        Returns
        -------
        str
            JSON string representation of the inference problem
            (also written to disk if ``path`` is provided).

        Raises
        ------
        TypeError
            If serialization fails.
        """
        import json

        spec = self.to_dict(**extra_metadata)
        json_str = json.dumps(spec, indent=2)

        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(json_str)

        return json_str

    @classmethod
    def from_json(cls, source: Union[str, bytes]) -> "InferenceProblem":
        """
        Reconstruct an :class:`InferenceProblem` from a JSON string or file.

        Parameters
        ----------
        source : str or bytes
            Either:

            - A JSON string representation of the problem
            - A file path pointing to a JSON file

        Returns
        -------
        InferenceProblem
            Reconstructed inference problem instance.

        Raises
        ------
        ValueError
            If the JSON structure is invalid.
        TypeError
            If reconstruction fails.
        """
        import json
        import os

        # ---------------------------------------------------------
        # Determine if source is a file path
        # ---------------------------------------------------------
        if isinstance(source, str) and os.path.exists(source):
            with open(source, encoding="utf-8") as f:
                spec = json.load(f)
        else:
            spec = json.loads(source)

        return cls.from_dict(spec)

    def to_yaml(self, path: Optional[str] = None, **extra_metadata) -> str:
        """
        Serialize the :class:`InferenceProblem` to a YAML string or file.

        This method uses :meth:`to_dict` internally and converts the
        resulting dictionary to YAML format.

        Parameters
        ----------
        path : str, optional
            If provided, the YAML representation will be written to this file.
            If omitted, the YAML string is returned.
        **extra_metadata
            Optional additional metadata fields passed to :meth:`to_dict`.

        Returns
        -------
        str
            YAML string representation of the inference problem
            (also written to disk if ``path`` is provided).

        Raises
        ------
        ImportError
            If PyYAML is not installed.
        TypeError
            If serialization fails.
        """
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required for YAML serialization. Install with `pip install pyyaml`.") from exc

        spec = self.to_dict(**extra_metadata)
        yaml_str = yaml.safe_dump(spec, sort_keys=False)

        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(yaml_str)

        return yaml_str

    @classmethod
    def from_yaml(cls, source: Union[str, bytes]) -> "InferenceProblem":
        """
        Reconstruct an :class:`InferenceProblem` from a YAML string or file.

        Parameters
        ----------
        source : str or bytes
            Either:

            - A YAML string representation of the problem
            - A file path pointing to a YAML file

        Returns
        -------
        InferenceProblem
            Reconstructed inference problem instance.

        Raises
        ------
        ImportError
            If PyYAML is not installed.
        ValueError
            If the YAML structure is invalid.
        TypeError
            If reconstruction fails.
        """
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for YAML deserialization. Install with `pip install pyyaml`."
            ) from exc

        import os

        # ---------------------------------------------------------
        # Determine if source is a file path
        # ---------------------------------------------------------
        if isinstance(source, str) and os.path.exists(source):
            with open(source, encoding="utf-8") as f:
                spec = yaml.safe_load(f)
        else:
            spec = yaml.safe_load(source)

        return cls.from_dict(spec)
