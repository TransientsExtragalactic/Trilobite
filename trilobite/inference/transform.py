"""
Transformation functions for use in inference pipelines.

Transformations may be provided for any of the parameters in an
:class:`~trilobite.inference.problem.InferenceProblem`, and are applied to the parameters before they are
passed to the likelihood and prior functions. This allows for flexible reparameterization of
inference problems, which can be useful for improving convergence or handling constraints.
"""

from abc import ABC, abstractmethod
from importlib.metadata import PackageNotFoundError, version

import numpy as np

# ============================================================= #
# Package versioning and Type checking
# ============================================================= #=
try:
    __pkg_version__ = version("trilobite")
except PackageNotFoundError:
    __pkg_version__ = "unknown"

__all__ = [
    "ParameterTransform",
    "IdentityTransform",
    "LogTransform",
    "Log10Transform",
    "LogisticTransform",
    "SoftplusTransform",
]


class ParameterTransform(ABC):
    r"""
    Abstract base class for parameter transformations used in inference.

    A :class:`ParameterTransform` defines a smooth, bijective mapping between:

    - **Physical parameter space** :math:`\theta \in \Theta`
    - **Sampling space** :math:`z \in \mathbb{R}^n`

    via a forward transformation

    .. math::

        z = f(\theta)

    and its inverse

    .. math::

        \theta = f^{-1}(z).

    In addition, the absolute value of the determinant of the Jacobian

    .. math::

        J_f(\theta) =
        \left|
        \det \frac{\partial f}{\partial \theta}
        \right|

    must be available in order to correctly account for changes of
    variables in probability densities.

    .. rubric:: Role in Inference

    Parameter transformations are used to improve sampling behavior
    while preserving the physical interpretation of model parameters.

    The typical inference flow is:

    1. The sampler proposes parameters in **sampling space** :math:`z`.
    2. The transform maps to physical space:

       .. math::
           \theta = f^{-1}(z)

    3. The model and likelihood are evaluated using :math:`\theta`.
    4. The prior is evaluated in physical space.
    5. The log-Jacobian correction

       .. math::
           \log \left| \frac{d\theta}{dz} \right|

       is added to ensure a mathematically correct posterior.

    This separation cleanly decouples:

    - Sampling geometry
    - Physical model interpretation
    - Prior specification

    .. rubric:: Design Requirements

    Concrete subclasses must:

    - Implement `_make_forward`
    - Implement `_make_inverse`
    - Implement `_make_jacobian`
    - Be smooth and bijective on their domain
    - Be fully importable (not defined in `__main__`)
    - Have JSON-serializable constructor parameters

    Transformations must be deterministic and reconstructible, as they
    may cross process boundaries (e.g., MPI workers) or be written to disk.

    Notes
    -----
    - The prior is always defined in physical parameter space.
    - The Jacobian correction is required when sampling in transformed space.
    - The default transformation for unconstrained parameters is the identity map.
    - Multi-dimensional transforms may be implemented provided the Jacobian
      determinant is correctly defined.
    """

    def __init__(self, **parameters):
        """
        Initialize the transformation.

        Parameters
        ----------
        parameters:
            Keyword arguments specifying the parameters of the transformation. The specific parameters
            depend on the type of transformation and are defined by the concrete subclasses.
        """
        # Store parameters for later use
        self._parameters = parameters

        # Generate the callables
        self._forward = self._make_forward()
        self._inverse = self._make_inverse()
        self._log_abs_det_jacobian = self._make_log_abs_det_jacobian()

    # ------------------------------------------------------------ #
    # Generator Methods
    # ------------------------------------------------------------ #
    @abstractmethod
    def _make_forward(self):
        """
        Generate the forward transformation function :math:`f`.

        Returns
        -------
        Callable
            A function that takes a parameter value and returns the transformed value.
        """
        pass

    @abstractmethod
    def _make_inverse(self):
        """
        Generate the inverse transformation function :math:`f^{-1}`.

        Returns
        -------
        Callable
            A function that takes a transformed parameter value and returns the original value.
        """
        pass

    @abstractmethod
    def _make_log_abs_det_jacobian(self):
        r"""
        Generate the log abs of the jacobian function :math:`J_f(\theta)`.

        Returns
        -------
        Callable
            A function that takes a parameter value and returns the jacobian of the transformation at that point.
        """
        pass

    # ------------------------------------------------------------ #
    # Public evaluation interface                                  #
    # ------------------------------------------------------------ #
    def forward(self, theta):
        """
        Apply the forward transformation to a parameter value.

        Parameters
        ----------
        theta:
            The parameter value to transform.

        Returns
        -------
        The transformed parameter value.
        """
        return self._forward(theta)

    def inverse(self, z):
        """
        Apply the inverse transformation to a transformed parameter value.

        Parameters
        ----------
        z:
            The transformed parameter value to invert.

        Returns
        -------
        The original parameter value.
        """
        return self._inverse(z)

    def log_abs_det_jacobian(self, theta):
        """
        Evaluate the jacobian of the transformation at a given parameter value.

        Parameters
        ----------
        theta:
            The parameter value at which to evaluate the jacobian.

        Returns
        -------
        The jacobian of the transformation at the given parameter value.
        """
        return self._log_abs_det_jacobian(theta)

    # ------------------------------------------------------------ #
    # Dunder methods                                               #
    # ------------------------------------------------------------ #
    def __call__(self, x: float) -> float:
        """
        Evaluate the forward transformation at a given parameter value.

        Parameters
        ----------
        x: float
            The parameter value to transform.

        Returns
        -------
        float
            The transformed parameter value.
        """
        return self.forward(x)

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in self._parameters.items())
        return f"{self.__class__.__name__}({params})"

    def __str__(self) -> str:
        if self._parameters:
            params = ", ".join(f"{k}={v}" for k, v in self._parameters.items())
            return f"{self.__class__.__name__} with parameters ({params})"
        return f"{self.__class__.__name__}"

    # ------------------------------------------------------------ #
    # Serialization support                                        #
    # ------------------------------------------------------------ #
    def to_dict(self) -> dict:
        """
        Serialize the transform to a JSON-safe dictionary.

        This requires that the transform class be importable via a fully-
        qualified module path of the form:

            module.submodule:ClassName

        All constructor parameters must be JSON-serializable.

        Raises
        ------
        TypeError
            If the transform class is not importable.
        TypeError
            If constructor parameters are not JSON-serializable.
        """
        import importlib
        import json

        module_name = self.__class__.__module__
        class_name = self.__class__.__name__

        if module_name == "__main__":
            raise TypeError(
                f"Transform '{class_name}' is defined in __main__ and "
                "cannot be serialized. Define it in an importable module."
            )

        # Verify module importability
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise TypeError(f"Module '{module_name}' for transform '{class_name}' is not importable.") from exc

        if not hasattr(module, class_name):
            raise TypeError(f"Transform class '{class_name}' not found in module '{module_name}'.")

        # Ensure JSON serializable parameters
        try:
            json.dumps(self._parameters)
        except TypeError as exc:
            raise TypeError(
                f"Transform '{class_name}' contains non-serializable parameters. "
                "Ensure constructor arguments are JSON-safe."
            ) from exc

        return {
            "format": "ParameterTransform",
            "version": __pkg_version__,
            "target": f"{module_name}:{class_name}",
            "parameters": self._parameters,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ParameterTransform":
        """
        Reconstruct a ParameterTransform from a serialized dictionary.

        Parameters
        ----------
        data : dict
            Dictionary produced by :meth:`to_dict`.

        Returns
        -------
        ParameterTransform
            Reconstructed transform instance.

        Raises
        ------
        TypeError
            If input is not a dictionary.
        ValueError
            If dictionary format is invalid.
        ImportError
            If module or class cannot be imported.
        TypeError
            If imported object is not a ParameterTransform subclass.
        """
        import importlib

        if not isinstance(data, dict):
            raise TypeError("Transform serialization must be a dictionary.")

        if data.get("format") != "ParameterTransform":
            raise ValueError("Dictionary does not represent a ParameterTransform.")

        try:
            target = data["target"]
        except KeyError as exc:
            raise ValueError("Serialized transform missing required 'target' field.") from exc

        parameters = data.get("parameters", {})

        if ":" not in target:
            raise ValueError(f"Invalid target '{target}'. Must be 'module.submodule:ClassName'.")

        module_name, class_name = target.split(":", 1)

        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise ImportError(
                f"Could not import module '{module_name}' while reconstructing ParameterTransform '{class_name}'."
            ) from exc

        try:
            transform_cls = getattr(module, class_name)
        except AttributeError as exc:
            raise ImportError(f"Class '{class_name}' not found in module '{module_name}'.") from exc

        if not issubclass(transform_cls, ParameterTransform):
            raise TypeError(f"Imported class '{class_name}' is not a subclass of ParameterTransform.")

        return transform_cls(**parameters)


class IdentityTransform(ParameterTransform):
    r"""
    Identity transformation.

    This transform leaves the parameter unchanged:

    .. math::

        z = \theta

        \theta = z

    The Jacobian determinant is unity:

    .. math::

        \left| \frac{d\theta}{dz} \right| = 1.

    Use Cases
    ---------
    - Unconstrained parameters
    - Parameters already well-behaved in sampling space
    - Default transform when no reparameterization is needed

    Notes
    -----
    This is the default transform applied when no explicit
    transformation is specified.
    """

    def _make_forward(self):
        return lambda theta: theta

    def _make_inverse(self):
        return lambda z: z

    def _make_log_abs_det_jacobian(self):
        return lambda z: 0.0


class LogTransform(ParameterTransform):
    r"""
    Natural logarithmic transformation for strictly positive parameters.

    This transform maps a positive parameter :math:`\theta > 0`
    to the real line via:

    .. math::

        z = \log \theta

    The inverse mapping is:

    .. math::

        \theta = e^{z}

    The Jacobian determinant of the inverse transformation is:

    .. math::

        \left| \frac{d\theta}{dz} \right|
        =
        \theta.

    Role in Inference
    -----------------
    This transform is commonly used for:

    - Scale parameters
    - Positive-definite quantities
    - Parameters spanning multiple orders of magnitude

    Sampling in log-space typically improves convergence
    when the posterior is broad in linear space.

    Domain
    ------
    :math:`\theta > 0`

    Notes
    -----
    The prior should be defined in physical space (:math:`\theta`),
    and the log-Jacobian correction must be added during inference.
    """

    def _make_forward(self):
        return lambda theta: np.log(theta)

    def _make_inverse(self):
        return lambda z: np.exp(z)

    def _make_log_abs_det_jacobian(self):
        return lambda z: z


class Log10Transform(ParameterTransform):
    r"""
    Base-10 logarithmic transformation for strictly positive parameters.

    This transform maps :math:`\theta > 0` via:

    .. math::

        z = \log_{10} \theta

    The inverse mapping is:

    .. math::

        \theta = 10^{z}

    The Jacobian determinant of the inverse transformation is:

    .. math::

        \left| \frac{d\theta}{dz} \right|
        =
        \theta \ln(10).

    Role in Inference
    -----------------
    Useful when parameters are naturally interpreted
    in decades (e.g., luminosities, masses, densities).

    Domain
    ------
    :math:`\theta > 0`

    Notes
    -----
    This differs from :class:`LogTransform` only by a constant scaling.
    The Jacobian includes the factor :math:`\ln(10)`.
    """

    def _make_forward(self):
        return lambda theta: np.log10(theta)

    def _make_inverse(self):
        return lambda z: 10.0**z

    def _make_log_abs_det_jacobian(self):
        ln10 = np.log(10.0)
        return lambda z: z * ln10 + np.log(ln10)


class LogisticTransform(ParameterTransform):
    r"""
    Logistic transformation mapping a bounded interval to the real line.

    This transform maps a parameter constrained to an interval
    :math:`(a, b)` onto the real line via:

    .. math::

        z = \log\left(\frac{\theta - a}{b - \theta}\right)

    The inverse mapping is:

    .. math::

        \theta
        =
        a + \frac{b - a}{1 + e^{-z}}.

    The Jacobian determinant of the inverse transformation is:

    .. math::

        \left| \frac{d\theta}{dz} \right|
        =
        \frac{(b - a) e^{-z}}{(1 + e^{-z})^{2}}.

    Parameters
    ----------
    lower : float
        Lower bound :math:`a`.
    upper : float
        Upper bound :math:`b`.

    Role in Inference
    -----------------
    Useful for:

    - Parameters constrained to finite intervals
    - Efficiencies
    - Fractions
    - Probabilities

    Domain
    ------
    :math:`a < \theta < b`

    Notes
    -----
    The bounds must satisfy :math:`b > a`.

    This transformation ensures that unconstrained
    sampling in :math:`z \in \mathbb{R}` respects
    the physical bounds in :math:`\theta`.
    """

    def __init__(self, lower: float, upper: float):
        if upper <= lower:
            raise ValueError("LogisticTransform requires upper > lower.")
        super().__init__(lower=float(lower), upper=float(upper))

    def _make_forward(self):
        lower = self._parameters["lower"]
        upper = self._parameters["upper"]

        def forward(theta):
            return np.log((theta - lower) / (upper - theta))

        return forward

    def _make_inverse(self):
        lower = self._parameters["lower"]
        upper = self._parameters["upper"]

        def inverse(z):
            return lower + (upper - lower) / (1.0 + np.exp(-z))

        return inverse

    def _make_log_abs_det_jacobian(self):
        lower = self._parameters["lower"]
        upper = self._parameters["upper"]
        delta = upper - lower

        def _log_jac(z):
            ez = np.exp(-z)
            return np.log(delta) + np.log(ez) - 2.0 * np.log1p(ez)

        return _log_jac


class SoftplusTransform(ParameterTransform):
    r"""
    Softplus transformation for strictly positive parameters.

    The forward transform is:

    .. math::

        z = \log\left(e^{\theta} - 1\right)

    The inverse transform is:

    .. math::

        \theta = \log\left(1 + e^{z}\right).

    The Jacobian determinant of the inverse transformation is:

    .. math::

        \left| \frac{d\theta}{dz} \right|
        =
        \frac{e^{z}}{1 + e^{z}}.

    Role in Inference
    -----------------
    The softplus transform is smoother near zero than the logarithmic
    transform and may improve numerical stability in some cases.

    Domain
    ------
    :math:`\theta > 0`

    Notes
    -----
    Compared to :class:`LogTransform`, this transform does not
    diverge as :math:`\theta \to 0`.
    """

    def _make_forward(self):
        return lambda theta: np.log(np.exp(theta) - 1.0)

    def _make_inverse(self):
        return lambda z: np.log1p(np.exp(z))

    def _make_log_abs_det_jacobian(self):
        return lambda z: z - np.log1p(np.exp(z))
