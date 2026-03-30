"""Abstract base class for likelihood models."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from triceratops.data.core import InferenceData
from triceratops.models.core.base import Model

from .gaussian import censored_gaussian_loglikelihood, gaussian_loglikelihood

if TYPE_CHECKING:
    from triceratops.models._typing import (
        _ModelParametersInput,
        _ModelParametersInputRaw,
    )

# Defining the __all__ variable.
__all__ = ["Likelihood", "GaussianLikelihood", "GaussianCensoredLikelihood"]


# ============================================================ #
# Likelihood Base Class                                        #
# ============================================================ #
class Likelihood(ABC):
    """
    Abstract base class for statistical likelihood functions.

    The :class:`~triceratops.inference.likelihood.base.Likelihood` class defines the interface and execution
    semantics for evaluating the probability of observational data
    given a physical model.

    This class operates strictly on:

    - A validated :class:`~triceratops.data.core.InferenceData` object
    - A concrete subclass of :class:`~triceratops.models.core.base.Model`

    It performs **no unit coercion, column resolution, or table parsing**.
    Those responsibilities belong to the data ingestion layer
    (e.g. :class:`~triceratops.data.core.DataContainer` → :class:`~triceratops.data.core.InferenceData`).

    Architectural Role
    ------------------
    The inference stack is intentionally layered:

    1. **DataContainer**
       Handles column resolution and unit coercion.

    2. **InferenceData**
       Stores validated, immutable NumPy arrays.

    3. **Likelihood**
       Implements statistical assumptions and evaluates
       log-likelihood values using numerical arrays.

    4. **Model**
       Encodes forward physics.

    The :class:`~triceratops.inference.likelihood.base.Likelihood` layer is therefore a **pure statistical engine**
    operating on already-validated numerical data.

    Design Principles
    -----------------
    - The likelihood should contain *only statistical logic*.
    - All unit validation and shape checking must already be complete.
    - Model–data compatibility must be explicitly validated.
    - The likelihood evaluation itself must be stateless with respect
      to model parameters.

    Extending Likelihood
    --------------------
    Concrete likelihood implementations must implement:

    - :meth:`_configure`
    - :meth:`_validate_model_and_data`
    - :meth:`_log_likelihood`

    Subclasses are responsible for:

    - Enforcing model–data compatibility
    - Extracting required arrays from :class:`InferenceData`
    - Calling the appropriate numerical backend
    - Implementing the statistical form of the likelihood

    Notes
    -----
    - This class makes no assumptions about priors or sampling strategies.
    - Parameter coercion is delegated to the model.
    - Data validation is delegated to :class:`InferenceData`.
    """

    # ============================================================ #
    # Initialization                                               #
    # ============================================================ #
    def __init__(self, model: Model, data: InferenceData, **kwargs):
        """
        Initialize the likelihood object.

        Parameters
        ----------
        model : Model
            The forward model used to generate predicted observables.
            Must be an instance of :class:`~triceratops.models.core.base.Model`.

        data : InferenceData
            Fully validated, immutable dataset containing
            numerical arrays required for likelihood evaluation.
            All unit coercion and shape validation must already
            have been performed.

        **kwargs
            Additional keyword arguments forwarded to :meth:`_configure`.

        Raises
        ------
        TypeError
            If `model` is not a :class:`~triceratops.models.core.base.Model` instance.
        TypeError
            If `data` is not an :class:`InferenceData` instance.
        """
        if not isinstance(model, Model):
            raise TypeError("`model` must be an instance of Model.")

        if not isinstance(data, InferenceData):
            raise TypeError("`data` must be an instance of InferenceData.")

        self._model: Model = model
        self._data_container: InferenceData = data

        # Allow subclasses to configure internal state
        self._configure(**kwargs)

        # Validate model–data compatibility
        self._validate_model_and_data()

    # ============================================================ #
    # Configuration & Validation                                   #
    # ============================================================ #
    @abstractmethod
    def _configure(self, **kwargs):
        """
        Configure subclass-specific behavior.

        Called immediately after model and data assignment.
        Subclasses may use this hook to define flags, options,
        or internal parameters prior to validation.

        Parameters
        ----------
        **kwargs
            Arbitrary configuration options provided at initialization.
        """
        raise NotImplementedError

    @abstractmethod
    def _validate_model_and_data(self):
        """
        Validate compatibility between the model and data.

        This method must verify that:

        - The model structure is compatible with this likelihood
        - The data structure satisfies required statistical constraints

        It must raise an exception if incompatibility is detected.
        """
        raise NotImplementedError

    # ============================================================ #
    # Likelihood Evaluation                                        #
    # ============================================================ #
    @abstractmethod
    def _log_likelihood(
        self,
        parameters: dict[str, "_ModelParametersInputRaw"],
    ) -> float:
        """
        Compute the log-likelihood for raw model parameters.

        Parameters
        ----------
        parameters : dict
            Mapping of model parameter names to values in the
            model's raw, unit-consistent internal format.

        Returns
        -------
        float
            Log-likelihood value.
        """
        raise NotImplementedError

    def log_likelihood(
        self,
        parameters: dict[str, "_ModelParametersInput"],
    ) -> float:
        """
        Evaluate the log-likelihood for model parameters.

        This is the public entry point for likelihood evaluation.
        It accepts parameters in any format supported by the model
        (e.g., quantities with units, scalars, array-like objects)
        and coerces them into the model’s raw internal representation
        before delegating to :meth:`_log_likelihood`.

        Parameters
        ----------
        parameters : dict
            Mapping of model parameter names to values in any format
            accepted by :meth:`~triceratops.models.core.base.Model.coerce_model_parameters`.

        Returns
        -------
        float
            Log-likelihood value.

        Notes
        -----
        For performance-critical inference backends, prefer calling
        :meth:`_log_likelihood` directly with parameters already
        in raw format to avoid repeated coercion overhead.
        """
        raw = self._model.coerce_model_parameters(parameters)
        return self._log_likelihood(raw)


class GaussianLikelihood(Likelihood):
    r"""
    Multivariate Gaussian log-likelihood for uncensored observations.

    This likelihood assumes that all observed data points are detections
    with symmetric Gaussian uncertainties. Let the forward model predict observables

    .. math::

        \mathbf{y}_{\mathrm{model}}(\theta)

    at parameter vector :math:`\theta`, evaluated at fixed independent
    variables :math:`\mathbf{x}`.

    Given observed data :math:`\mathbf{y}` with associated 1σ
    uncertainties :math:`\boldsymbol{\sigma}`, the Gaussian log-likelihood is

    .. math::

        \log \mathcal{L}(\theta)
        =
        -\frac{1}{2}
        \sum_{i}
        \left[
            \frac{
                (y_i - y_{\mathrm{model},i})^2
            }{
                \sigma_i^2
            }
            +
            \log(2\pi \sigma_i^2)
        \right].

    The summation is taken over **all data points and all observables**.
    Observables are stacked along the final axis.

    Assumptions
    -----------

    - All observations are detections (no censoring).
    - Symmetric Gaussian uncertainties are provided.
    - Model output shape matches observed data shape.
    - Errors are independent (diagonal covariance).

    Shape Conventions
    -----------------

    If there are:

    - `N` data points
    - `P` observables

    then:

    - ``data_y`` has shape `(N, P)`
    - ``model_y`` has shape `(N, P)`
    - ``y_err`` has shape `(N, P)`

    These arrays are cached during initialization for efficiency.

    Performance Notes
    -----------------

    - All data extraction occurs once during initialization.
    - The hot loop in :meth:`_log_likelihood` performs only:
      1. Forward model evaluation
      2. Stacking of model outputs
      3. A call to the numerical backend

    This makes the class suitable for MCMC or nested sampling.

    See Also
    --------
    ~triceratops.inference.likelihood.gaussian.gaussian_loglikelihood : Low-level numerical backend.
    """

    # ============================================================
    # Configuration
    # ============================================================
    def _configure(self, **kwargs):
        """
        Extract and cache numerical arrays from :class:`InferenceData`.

        This method is called exactly once during initialization.

        Extracted attributes
        --------------------
        _data_y : ndarray, shape (N, P)
            Observed values.

        _y_err : ndarray, shape (N, P)
            1σ uncertainties.

        _x : dict[str, ndarray]
            Independent variable arrays used by the forward model.
        """
        # Cache independent variables for faster access
        self._x = self._data_container.x

        # Stack observed values across observables
        self._data_y = self._data_container.get_y_array(flatten=False)

        # Stack uncertainties
        self._y_err = self._data_container.get_y_error_array(flatten=False)

    # ============================================================
    # Validation
    # ============================================================
    def _validate_model_and_data(self):
        """
        Validate structural compatibility between model and data.

        Ensures:
        - Uncertainties are present
        - Shapes are consistent
        """
        # Ensure that shapes are consistent.
        shape = self._data_y.shape

        if self._y_err.shape != shape:
            raise ValueError(
                "Provided data container must have equal shapes for observed values and"
                f" uncertainties. Found shapes: y: {shape}, y_err: {self._y_err.shape}"
            )

        # Ensure that we have EITHER a detection, an upper limit, or a lower limit
        # for EVERY entry.
        if np.any(np.isnan(self._data_y)):
            raise ValueError("GaussianCensoredLikelihood requires that every entry has a detection.")

        # Ensure that an error is provided for EVERY entry.
        if np.any(np.isnan(self._y_err)):
            raise ValueError("GaussianCensoredLikelihood requires defined uncertainties for all entries.")

    # ============================================================
    # Likelihood evaluation
    # ============================================================
    def _log_likelihood(self, parameters):
        """
        Evaluate the Gaussian log-likelihood.

        Parameters
        ----------
        parameters : dict[str, Any]
            Raw model parameters in unit-consistent internal format.

        Returns
        -------
        float
            Log-likelihood value.
        """
        # --------------------------------------------------------
        # Forward model evaluation
        # --------------------------------------------------------
        # Returns a tuple of length P, each of shape (N,)
        model_tuple = self._model._forward_model_tupled(
            self._x,
            parameters,
        )

        # Stack into shape (N, P)
        model_y = np.stack(model_tuple, axis=-1)

        # --------------------------------------------------------
        # Delegate statistical computation
        # --------------------------------------------------------
        return gaussian_loglikelihood(
            data_y=self._data_y,
            model_y=model_y,
            y_err=self._y_err,
        )


class GaussianCensoredLikelihood(Likelihood):
    r"""
    Multivariate Gaussian log-likelihood with support for censored data.

    This likelihood extends the standard Gaussian likelihood to support:

    - Uncensored detections
    - Upper limits (right-censoring)
    - Lower limits (left-censoring)

    For uncensored observations, the likelihood contribution is:

    .. math::

        \log \mathcal{L}_i
        =
        -\frac{1}{2}
        \left[
            \frac{(y_i - \mu_i)^2}{\sigma_i^2}
            +
            \log(2\pi \sigma_i^2)
        \right]

    For upper-censored observations (upper limits):

    .. math::

        \log \mathcal{L}_i
        =
        \log \Phi\left(
            \frac{y_{\mathrm{upper}, i} - \mu_i}{\sigma_i}
        \right)

    For lower-censored observations (lower limits):

    .. math::

        \log \mathcal{L}_i
        =
        \log \left[
            1 - \Phi\left(
                \frac{y_{\mathrm{lower}, i} - \mu_i}{\sigma_i}
            \right)
        \right]

    where :math:`\Phi` is the standard normal CDF.

    The total log-likelihood is the sum over all data points and observables.

    Assumptions
    -----------

    - Symmetric 1-sigma uncertainties are defined for all entries.
    - Censoring is indicated via `upper` and/or `lower` arrays.
    - Non-censored entries contain `np.nan` in the corresponding limit arrays.
    - Observables are independent (diagonal covariance).

    Shape Conventions
    -----------------
    If there are:

    - `N` data points
    - `P` observables

    then:

    - ``data_y`` has shape `(N, P)`
    - ``model_y`` has shape `(N, P)`
    - ``y_err`` has shape `(N, P)`
    - ``y_upper`` has shape `(N, P)`
    - ``y_lower`` has shape `(N, P)`
    - mask arrays have shape `(N, P)`

    Performance Notes
    -----------------

    - All data extraction and mask construction occurs once
      during initialization.
    - The hot loop only evaluates the forward model and
      calls the numerical backend.

    See Also
    --------
    ~triceratops.inference.likelihood.gaussian.censored_gaussian_loglikelihood : Low-level numerical backend.
    """

    # ============================================================
    # Configuration
    # ============================================================
    def _configure(self, **kwargs):
        """
        Extract and cache numerical arrays and censoring masks.

        Executed once during initialization.
        """
        # Cache independent variables
        self._x = self._data_container.x

        # Extract observed values
        self._data_y = self._data_container.get_y_array(flatten=False)

        # Extract uncertainties
        self._y_err = self._data_container.get_y_error_array(flatten=False)

        # Extract censoring arrays
        self._y_upper = self._data_container.get_y_upper_array(flatten=False)
        self._y_lower = self._data_container.get_y_lower_array(flatten=False)

        # Construct masks
        if self._y_upper is not None:
            self._y_upper_mask = ~np.isnan(self._y_upper)
        else:
            self._y_upper = np.full_like(self._data_y, np.nan)
            self._y_upper_mask = np.zeros_like(self._data_y, dtype=bool)

        if self._y_lower is not None:
            self._y_lower_mask = ~np.isnan(self._y_lower)
        else:
            self._y_lower = np.full_like(self._data_y, np.nan)
            self._y_lower_mask = np.zeros_like(self._data_y, dtype=bool)

    # ============================================================
    # Validation
    # ============================================================
    def _validate_model_and_data(self):
        """Validate structural compatibility and required fields."""
        # Ensure that shapes are consistent.
        shape = self._data_y.shape

        if self._y_err.shape != shape:
            raise ValueError(
                "Provided data container must have equal shapes for observed values and"
                f" uncertainties. Found shapes: y: {shape}, y_err: {self._y_err.shape}"
            )
        if self._y_upper.shape != shape:
            raise ValueError(
                "Provided data container must have equal shapes for observed values and"
                f" upper limits. Found shapes: y: {shape}, y_upper: {self._y_upper.shape}"
            )
        if self._y_lower.shape != shape:
            raise ValueError(
                "Provided data container must have equal shapes for observed values and"
                f" lower limits. Found shapes: y: {shape}, y_lower: {self._y_lower.shape}"
            )

        # Ensure that we have EITHER a detection, an upper limit, or a lower limit
        # for EVERY entry.
        if np.any(np.isnan(self._y_upper) & np.isnan(self._y_lower) & np.isnan(self._data_y)):
            raise ValueError(
                "GaussianCensoredLikelihood requires that every entry has either a detection"
                ", an upper limit, or a lower limit."
            )

        # Ensure that an error is provided for EVERY entry.
        if np.any(np.isnan(self._y_err)):
            raise ValueError("GaussianCensoredLikelihood requires defined uncertainties for all entries.")

    # ============================================================
    # Likelihood evaluation
    # ============================================================
    def _log_likelihood(self, parameters):
        """
        Evaluate censored Gaussian log-likelihood.

        Parameters
        ----------
        parameters : dict[str, Any]
            Raw model parameters in unit-consistent internal format.

        Returns
        -------
        float
            Log-likelihood value.
        """
        # --------------------------------------------------------
        # Forward model evaluation
        # --------------------------------------------------------
        model_tuple = self._model._forward_model_tupled(
            self._x,
            parameters,
        )

        model_y = np.stack(model_tuple, axis=-1)
        # --------------------------------------------------------
        # Delegate to numerical backend
        # --------------------------------------------------------
        return censored_gaussian_loglikelihood(
            data_y=self._data_y,
            model_y=model_y,
            y_err=self._y_err,
            y_upper=self._y_upper,
            y_lower=self._y_lower,
            y_upper_mask=self._y_upper_mask,
            y_lower_mask=self._y_lower_mask,
        )
