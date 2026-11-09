"""Abstract base class for likelihood models."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Optional

import numpy as np

from trilobite.data.core import InferenceData
from trilobite.models.core.base import Model
from trilobite.models.core.parameters import ModelParameter

from .gaussian import censored_gaussian_loglikelihood, gaussian_loglikelihood

if TYPE_CHECKING:
    from trilobite.models._typing import (
        _ModelParametersInput,
        _ModelParametersInputRaw,
    )

# Defining the __all__ variable.
__all__ = [
    "Likelihood",
    "GaussianLikelihood",
    "GaussianCensoredLikelihood",
    "LogGaussianCensoredLikelihood",
    "LogGaussianCensoredSysLikelihood",
]


# ============================================================ #
# Likelihood Base Class                                        #
# ============================================================ #
class Likelihood(ABC):
    """
    Abstract base class for statistical likelihood functions.

    The :class:`~trilobite.inference.likelihood.base.Likelihood` class defines the interface and execution
    semantics for evaluating the probability of observational data
    given a physical model.

    This class operates strictly on:

    - A validated :class:`~trilobite.data.core.InferenceData` object
    - A concrete subclass of :class:`~trilobite.models.core.base.Model`

    It performs **no unit coercion, column resolution, or table parsing**.
    Those responsibilities belong to the data ingestion layer
    (e.g. :class:`~trilobite.data.core.DataContainer` → :class:`~trilobite.data.core.InferenceData`).

    .. rubric:: Architectural Role

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

    The :class:`~trilobite.inference.likelihood.base.Likelihood` layer is therefore a **pure statistical engine**
    operating on already-validated numerical data.

    .. rubric:: Design Principles

    - The likelihood should contain *only statistical logic*.
    - All unit validation and shape checking must already be complete.
    - Model–data compatibility must be explicitly validated.
    - The likelihood evaluation itself must be stateless with respect
      to model parameters.

    .. rubric:: Extending Likelihood

    Concrete likelihood implementations must implement:

    - :meth:`_configure`
    - :meth:`_validate_model_and_data`
    - :meth:`_log_likelihood`

    Subclasses are responsible for:

    - Enforcing model–data compatibility
    - Extracting required arrays from :class:`InferenceData`
    - Calling the appropriate numerical backend
    - Implementing the statistical form of the likelihood

    .. rubric:: Nuisance Parameters

    A likelihood may own parameters of its own, such as a free systematic error term, which are sampled
    alongside the model parameters but belong to the statistical description of the data rather than to the
    physics. They are declared in the class-level :attr:`NUISANCE_PARAMETERS` tuple, which mirrors
    :attr:`~trilobite.models.core.base.Model.PARAMETERS` and is empty by default. An
    :class:`~trilobite.inference.problem.InferenceProblem` appends them after the model parameters, so priors,
    freezing and transforms behave exactly as they do for model parameters.

    :meth:`log_likelihood` accepts one dictionary holding both groups, coerces the model parameters through the
    model and the nuisance parameters through their own :class:`~trilobite.models.core.parameters.ModelParameter`
    definitions, and calls ``_log_likelihood(parameters)`` or, when nuisance parameters are declared,
    ``_log_likelihood(parameters, nuisance)``. The model therefore never sees nuisance values.

    Parameters
    ----------
    model : ~trilobite.models.core.base.Model
        The forward model used to generate predicted observables. Must be an instance of
        :class:`~trilobite.models.core.base.Model`.
    data : InferenceData
        Fully validated, immutable dataset containing numerical arrays required for likelihood evaluation. All unit
        coercion and shape validation must already have been performed.
    **kwargs :
        Additional keyword arguments forwarded to :meth:`_configure`.

    Notes
    -----
    - This class makes no assumptions about priors or sampling strategies.
    - Parameter coercion is delegated to the model.
    - Data validation is delegated to :class:`InferenceData`.
    """

    NUISANCE_PARAMETERS: ClassVar[tuple] = ()
    """tuple of ~trilobite.models.core.parameters.ModelParameter: Parameters owned by the likelihood (default: none).

    Their names must be unique and must not coincide with any parameter of the model.
    """

    # ============================================================ #
    # Initialization                                               #
    # ============================================================ #
    def __init__(self, model: Model, data: InferenceData, **kwargs):
        """
        Initialize the likelihood object.

        Parameters
        ----------
        model : ~trilobite.models.core.base.Model
            The forward model used to generate predicted observables.
            Must be an instance of :class:`~trilobite.models.core.base.Model`.

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
            If `model` is not a :class:`~trilobite.models.core.base.Model` instance.
        TypeError
            If `data` is not an :class:`InferenceData` instance.
        ValueError
            If the declared nuisance parameters have duplicate names or share a name with a model parameter.
        """
        if not isinstance(model, Model):
            raise TypeError("`model` must be an instance of Model.")

        if not isinstance(data, InferenceData):
            raise TypeError("`data` must be an instance of InferenceData.")

        self._model: Model = model
        self._data_container: InferenceData = data

        # Nuisance parameters share the parameter namespace with the model's, so names must not collide.
        self._validate_nuisance_parameters()

        # Keep the configuration so that the likelihood can be serialized and rebuilt (see
        # `InferenceProblem.to_dict`); `model` and `data` are serialized separately.
        self._config: dict = dict(kwargs)

        # Allow subclasses to configure internal state
        self._configure(**kwargs)

        # Validate model–data compatibility
        self._validate_model_and_data()

    @property
    def config(self) -> dict:
        """dict: The keyword arguments this likelihood was constructed with (excluding ``model`` and ``data``).

        These are exactly the options forwarded to :meth:`_configure` (e.g. ``sys_dex`` for
        :class:`~trilobite.inference.likelihood.base.LogGaussianCensoredLikelihood`). They are stored when the
        :class:`~trilobite.inference.problem.InferenceProblem` is serialized, and must therefore be JSON-serializable.
        """
        return dict(self._config)

    @property
    def nuisance_parameters(self) -> tuple:
        """Tuple of ~trilobite.models.core.parameters.ModelParameter: The parameters owned by this likelihood.

        Empty unless the subclass declares :attr:`NUISANCE_PARAMETERS`.
        """
        return tuple(self.NUISANCE_PARAMETERS)

    def _validate_nuisance_parameters(self):
        """Check that the nuisance parameters are well-formed and do not collide with the model's parameters."""
        for parameter in self.NUISANCE_PARAMETERS:
            if not isinstance(parameter, ModelParameter):
                raise TypeError(
                    f"`{type(self).__name__}.NUISANCE_PARAMETERS` must contain ModelParameter instances, "
                    f"got {type(parameter).__name__}."
                )

        names = [p.name for p in self.NUISANCE_PARAMETERS]
        duplicated = {name for name in names if names.count(name) > 1}
        if duplicated:
            raise ValueError(f"Duplicate nuisance parameter names in `{type(self).__name__}`: {sorted(duplicated)}.")

        collisions = set(names) & {p.name for p in self._model.PARAMETERS}
        if collisions:
            raise ValueError(
                f"Nuisance parameter name(s) {sorted(collisions)} of `{type(self).__name__}` collide with "
                f"parameters of the model `{type(self._model).__name__}`."
            )

    def coerce_nuisance_parameters(self, parameters: dict) -> dict:
        """
        Coerce the likelihood's nuisance parameters into raw numerical values in base units.

        Mirrors :meth:`~trilobite.models.core.base.Model.coerce_model_parameters`: missing entries take the
        declared default, and keys that are not nuisance parameters are ignored.

        Parameters
        ----------
        parameters : dict
            Mapping of parameter names to values (floats, arrays or :class:`~astropy.units.Quantity`). May
            also contain model parameters, which are ignored here.

        Returns
        -------
        dict
            Mapping of each nuisance parameter name to its value in base units.

        Raises
        ------
        ValueError
            If a value cannot be converted to the parameter's base units.
        """
        coerced = {}
        for param in self.NUISANCE_PARAMETERS:
            try:
                coerced[param.name] = param.to_base_value(parameters.get(param.name, param.default))
            except Exception as exc:
                raise ValueError(f"Error coercing nuisance parameter '{param.name}': {exc}") from exc

        return coerced

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
        nuisance: Optional[dict] = None,
    ) -> float:
        """
        Compute the log-likelihood for raw model parameters.

        Parameters
        ----------
        parameters : dict
            Mapping of model parameter names to values in the
            model's raw, unit-consistent internal format.
        nuisance : dict, optional
            Mapping of nuisance parameter names to raw values in base units. Passed only when the likelihood
            declares :attr:`NUISANCE_PARAMETERS`, in which case the implementation must accept it; likelihoods
            without nuisance parameters may keep the single-argument signature ``_log_likelihood(self, parameters)``.

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
            Mapping of parameter names to values in any format accepted by
            :meth:`~trilobite.models.core.base.Model.coerce_model_parameters`. Nuisance parameters, if the
            likelihood declares any, are read from the same dictionary and coerced by
            :meth:`coerce_nuisance_parameters`.

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

        if not self.NUISANCE_PARAMETERS:
            return self._log_likelihood(raw)

        return self._log_likelihood(raw, self.coerce_nuisance_parameters(parameters))


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

    .. rubric:: Assumptions

    - All observations are detections (no censoring).
    - Symmetric Gaussian uncertainties are provided.
    - Model output shape matches observed data shape.
    - Errors are independent (diagonal covariance).

    .. rubric:: Shape Conventions

    If there are:

    - `N` data points
    - `P` observables

    then:

    - ``data_y`` has shape `(N, P)`
    - ``model_y`` has shape `(N, P)`
    - ``y_err`` has shape `(N, P)`

    These arrays are cached during initialization for efficiency.

    .. rubric:: Performance Notes

    - All data extraction occurs once during initialization.
    - The hot loop in :meth:`_log_likelihood` performs only:
      1. Forward model evaluation
      2. Stacking of model outputs
      3. A call to the numerical backend

    This makes the class suitable for MCMC or nested sampling.

    See Also
    --------
    :obj:`~trilobite.inference.likelihood.gaussian.gaussian_loglikelihood` : Low-level numerical backend.
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

    .. rubric:: Assumptions

    - Symmetric 1-sigma uncertainties are defined for all entries.
    - Censoring is indicated via `upper` and/or `lower` arrays.
    - Non-censored entries contain `np.nan` in the corresponding limit arrays.
    - Observables are independent (diagonal covariance).

    .. rubric:: Shape Conventions

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

    .. rubric:: Performance Notes

    - All data extraction and mask construction occurs once
      during initialization.
    - The hot loop only evaluates the forward model and
      calls the numerical backend.

    See Also
    --------
    :obj:`~trilobite.inference.likelihood.gaussian.censored_gaussian_loglikelihood` : Low-level numerical backend.
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


class LogGaussianCensoredLikelihood(GaussianCensoredLikelihood):
    r"""
    Censored Gaussian log-likelihood in :math:`\log_{10}` of the observables.

    This likelihood compares model and data in logarithmic space, which is the natural choice
    for strictly positive observables (e.g. flux densities) that span several orders of
    magnitude. In linear space, the brightest points dominate the fit and a model that
    under-predicts a point by a large factor is penalized far less than one that over-predicts
    it by the same factor. In log space both directions are penalized equally.

    For detections, the likelihood contribution is

    .. math::

        \log \mathcal{L}_i
        =
        -\frac{1}{2}
        \left[
            \frac{(\log_{10} y_i - \log_{10} \mu_i)^2}{\sigma_{i}^2}
            +
            \log(2\pi \sigma_i^2)
        \right],

    and upper (lower) limits contribute :math:`\log\Phi` (:math:`\log[1-\Phi]`) of the standardized
    :math:`\log_{10}` limit, exactly as in
    :class:`~trilobite.inference.likelihood.base.GaussianCensoredLikelihood`. The uncertainty
    in log space combines the reported measurement error, propagated to log space at the
    observed value (or at the limit value for censored entries), with an optional systematic
    floor added in quadrature:

    .. math::

        \sigma_i^2
        =
        \left(\frac{\sigma_{y,i}}{y_i \ln 10}\right)^2
        +
        \sigma_{\rm sys}^2 .

    .. rubric:: Parameters

    model : ~trilobite.models.core.base.Model
        The forward model. All of its outputs must be strictly positive.
    data : ~trilobite.data.core.InferenceData
        The dataset. Every detection and every limit must be strictly positive and finite.
    sys_dex : float, optional
        Systematic uncertainty floor in dex (:math:`\sigma_{\rm sys}`), added in quadrature to the
        propagated measurement error of every entry, including the censoring width of limits.
        Default is ``0``. If ``0``, every entry must have a nonzero reported error.

    See Also
    --------
    :obj:`~trilobite.inference.likelihood.base.GaussianCensoredLikelihood` : The linear-space version.

    Notes
    -----
    - **Invalid model output.** If any model output is NaN, infinite, or negative, the
      log-likelihood is ``-inf`` (this is how the infeasible-physics convention of models, which
      return NaN, is carried through). A model output of exactly zero is floored at
      ``1e-300`` before taking the logarithm, so it is penalized very strongly by detections but
      remains consistent with upper limits.
    - **Normalization.** This is a density in :math:`\log_{10} y`, not in :math:`y`; it omits the
      constant Jacobian :math:`1/(y\ln 10)`. Values of :math:`\ln\mathcal{L}` are therefore not
      comparable to those of the linear-space likelihoods (the parameter dependence, and hence
      all posteriors, are unaffected).
    - **Positivity.** Every observable in the dataset is assumed to be strictly positive.
    """

    #: Floor applied to non-negative model output before taking the logarithm.
    _MODEL_FLOOR = 1e-300

    def _configure(self, sys_dex: float = 0.0, **kwargs):
        """
        Cache the log10-transformed data, errors and limits.

        Executed once during initialization.
        """
        sys_dex = float(sys_dex)
        if not np.isfinite(sys_dex) or sys_dex < 0:
            raise ValueError(f"`sys_dex` must be finite and non-negative, got {sys_dex}.")
        self._sys_dex = sys_dex

        self._configure_log_space(**kwargs)

        sigma_dex = np.sqrt(self._sigma_meas_dex**2 + sys_dex**2)
        if not (np.all(np.isfinite(sigma_dex)) and np.all(sigma_dex > 0)):
            raise ValueError(
                "LogGaussianCensoredLikelihood requires a finite, nonzero uncertainty for every entry. "
                "Provide nonzero measurement errors or set `sys_dex` > 0."
            )

        self._y_err = sigma_dex

    def _configure_log_space(self, **kwargs):
        """
        Cache the log10-transformed data and limits, and the measurement-only uncertainty in dex.

        Sets ``_sigma_meas_dex`` (the reported error converted to dex, without any systematic term) and
        replaces ``_data_y``, ``_y_upper`` and ``_y_lower`` by their base-10 logarithms. Subclasses that
        treat the systematic term differently (for example as a sampled parameter) build on this.
        """
        # Cache the linear-space arrays and masks via the parent.
        super()._configure(**kwargs)

        upper_mask, lower_mask = self._y_upper_mask, self._y_lower_mask
        det_mask = ~upper_mask & ~lower_mask

        # The value at which each entry's reported error is converted to dex: the measured
        # value for detections and the limit value for censored entries.
        reference = np.where(det_mask, self._data_y, np.where(upper_mask, self._y_upper, self._y_lower))
        if not (np.all(np.isfinite(reference)) and np.all(reference > 0)):
            raise ValueError(
                f"{type(self).__name__} requires every detection and limit to be finite and strictly positive."
            )

        self._sigma_meas_dex = self._y_err / (reference * np.log(10.0))

        # NaN entries (non-detections in `value`, detections in the limit arrays) stay NaN.
        self._data_y = np.log10(self._data_y)
        self._y_upper = np.log10(self._y_upper)
        self._y_lower = np.log10(self._y_lower)

    def _log_likelihood(self, parameters):
        """
        Evaluate the censored Gaussian log-likelihood in log10 space.

        Parameters
        ----------
        parameters : dict[str, Any]
            Raw model parameters in unit-consistent internal format.

        Returns
        -------
        float
            Log-likelihood value, ``-inf`` if the model output is non-finite or negative.
        """
        return self._log_likelihood_with_errors(parameters, self._y_err)

    def _log_likelihood_with_errors(self, parameters, sigma_dex):
        """
        Evaluate the log10-space log-likelihood for the given per-entry uncertainties.

        Parameters
        ----------
        parameters : dict[str, Any]
            Raw model parameters in unit-consistent internal format.
        sigma_dex : numpy.ndarray
            Total one-sigma uncertainty of every entry in dex, shape ``(N, P)``.
        """
        model_tuple = self._model._forward_model_tupled(
            self._x,
            parameters,
        )
        model_y = np.stack(model_tuple, axis=-1)

        if not np.all(np.isfinite(model_y)) or np.any(model_y < 0):
            return -np.inf

        return censored_gaussian_loglikelihood(
            data_y=self._data_y,
            model_y=np.log10(np.maximum(model_y, self._MODEL_FLOOR)),
            y_err=sigma_dex,
            y_upper=self._y_upper,
            y_lower=self._y_lower,
            y_upper_mask=self._y_upper_mask,
            y_lower_mask=self._y_lower_mask,
        )


# ============================================================ #
# Likelihoods with Nuisance Parameters                         #
# ============================================================ #
class LogGaussianCensoredSysLikelihood(LogGaussianCensoredLikelihood):
    r"""
    Censored Gaussian likelihood in log10 space with a **sampled** systematic error term.

    This is :class:`~trilobite.inference.likelihood.base.LogGaussianCensoredLikelihood` with the fixed
    systematic floor ``sys_dex`` promoted to a nuisance parameter. The total uncertainty of entry :math:`i` is

    .. math::

        \sigma_i^2 = \sigma_{{\rm meas},i}^2 + \sigma_{\rm sys}^2,

    where :math:`\sigma_{{\rm meas},i}` is the reported error converted to dex and :math:`\sigma_{\rm sys}`
    is the nuisance parameter ``sys_dex``. It is added in quadrature to the width of detections and of censored
    limits alike. Because :math:`\sigma_i` now depends on a parameter, the :math:`\ln(2\pi\sigma_i^2)`
    normalization term is retained in the likelihood: the sampler therefore finds the error scale that the
    data support, without any reduced-:math:`\chi^2 = 1` constraint.

    The parameter appears in the :class:`~trilobite.inference.problem.InferenceProblem` after the model
    parameters, under the name ``sys_dex``, and has a default of 0.1 dex and a lower bound of 0. Priors,
    freezing and transforms (for example a :class:`~trilobite.inference.transform.Log10Transform`) are
    configured exactly as for a model parameter. Its starting value is set through
    ``problem.parameters["sys_dex"].initial_value``.

    .. rubric:: Parameters

    model : ~trilobite.models.core.base.Model
        Forward model producing strictly positive predictions.
    data : ~trilobite.data.core.InferenceData
        Validated dataset. Every detection and limit must be finite and strictly positive.

    Raises
    ------
    TypeError
        If the fixed ``sys_dex`` option of the parent class is passed. Use
        :class:`~trilobite.inference.likelihood.base.LogGaussianCensoredLikelihood` for a fixed floor.

    Notes
    -----
    If ``sys_dex`` is sampled to exactly zero while some entry has a zero reported error, that entry would have
    no width at all and the log-likelihood is ``-inf``. A negative ``sys_dex`` is likewise rejected with ``-inf``.
    """

    NUISANCE_PARAMETERS = (
        ModelParameter(
            "sys_dex",
            0.1,
            description="Systematic uncertainty added in quadrature to every entry, in dex.",
            latex=r"$\sigma_{\rm sys}$",
            bounds=(0.0, None),
        ),
    )

    def _configure(self, **kwargs):
        """Cache the log10-space data and the measurement-only uncertainty in dex."""
        if "sys_dex" in kwargs:
            raise TypeError(
                "`LogGaussianCensoredSysLikelihood` samples `sys_dex` as a nuisance parameter and does not accept a "
                "fixed `sys_dex` option; use `LogGaussianCensoredLikelihood` for a fixed floor."
            )

        self._configure_log_space(**kwargs)

        if not np.all(np.isfinite(self._sigma_meas_dex) & (self._sigma_meas_dex >= 0)):
            raise ValueError(
                "LogGaussianCensoredSysLikelihood requires a finite, non-negative reported uncertainty for every entry."
            )

        # Squared once here; the sampled term is added on every evaluation.
        self._sigma_meas_dex_sq = self._sigma_meas_dex**2

    def _log_likelihood(self, parameters, nuisance):
        """
        Evaluate the censored Gaussian log-likelihood in log10 space with a sampled systematic term.

        Parameters
        ----------
        parameters : dict[str, Any]
            Raw model parameters in unit-consistent internal format.
        nuisance : dict[str, float]
            Raw nuisance parameters; must contain ``sys_dex`` (dex).

        Returns
        -------
        float
            Log-likelihood value, ``-inf`` if the model output is non-finite or negative, if ``sys_dex`` is
            negative or non-finite, or if some entry has a zero total uncertainty.
        """
        sys_dex = float(nuisance["sys_dex"])
        if not np.isfinite(sys_dex) or sys_dex < 0.0:
            return -np.inf

        sigma_dex = np.sqrt(self._sigma_meas_dex_sq + sys_dex**2)
        if not np.all(sigma_dex > 0.0):
            return -np.inf

        return self._log_likelihood_with_errors(parameters, sigma_dex)
