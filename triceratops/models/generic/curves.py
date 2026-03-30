"""
Elementary curve models.

This module provides a collection of simple analytic curve models
intended primarily for:

- Testing inference infrastructure
- Validating likelihood implementations
- Demonstrating prior and transform behavior
- Writing deterministic unit tests
- Educational examples

These models implement elementary functional forms such as:

- Linear relations
- Polynomials
- Power laws
- Exponentials
- Constant functions

All models follow the standard :class:`~triceratops.models.core.base.Model`
interface and are fully compatible with:

- :class:`~triceratops.inference.likelihood.base.Likelihood`
- :class:`~triceratops.inference.problem.InferenceProblem`
- Parameter priors and transforms
- Serialization via model specifications

Notes
-----
- All models are dimensionless by default.
- Units may be extended in future versions if required.
- These models are not intended to represent physical systems,
  but rather serve as analytic test functions.
"""

from collections import namedtuple

import numpy as np
from astropy import units as u

from triceratops.models.core import Model, ModelParameter, ModelVariable

__all__ = ["LinearModel", "QuadraticModel", "PowerLawModel", "ExponentialModel", "ConstantModel"]

_CurveOutputs = namedtuple("CurveOutputs", ["y"])


class LinearModel(Model):
    r"""
    Linear function model.

    .. math::

        y(x) = m x + b

    .. dropdown:: Parameters

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``m``
             - :math:`m`
             - Slope of the linear relation.
           * - ``b``
             - :math:`b`
             - Intercept of the linear relation.

    .. dropdown:: Variables

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``x``
             - :math:`x`
             - Independent variable.

    .. dropdown:: Returns

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``y``
             - :math:`y(x)`
             - Evaluated model value.
    """

    VARIABLES = (ModelVariable("x", base_units=None),)

    PARAMETERS = (
        ModelParameter("m", 1.0, base_units=None),
        ModelParameter("b", 0.0, base_units=None),
    )

    OUTPUTS = _CurveOutputs
    UNITS = OUTPUTS(y=u.dimensionless_unscaled)

    DESCRIPTION = "Simple linear model."
    REFERENCE = "Elementary linear function."

    def __init__(self):
        self._register_init()

    def _forward_model(self, variables, parameters):
        x = np.asarray(variables["x"], dtype=float)
        m = parameters["m"]
        b = parameters["b"]
        return self.OUTPUTS(y=m * x + b)


class QuadraticModel(Model):
    r"""
    Quadratic polynomial model.

    .. math::

        y(x) = a x^2 + b x + c

    .. dropdown:: Parameters

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``a``
             - :math:`a`
             - Quadratic coefficient.
           * - ``b``
             - :math:`b`
             - Linear coefficient.
           * - ``c``
             - :math:`c`
             - Constant term.

    .. dropdown:: Variables

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``x``
             - :math:`x`
             - Independent variable.

    .. dropdown:: Returns

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``y``
             - :math:`y(x)`
             - Evaluated model value.
    """

    VARIABLES = (ModelVariable("x", base_units=None),)

    PARAMETERS = (
        ModelParameter("a", 1.0, base_units=None),
        ModelParameter("b", 0.0, base_units=None),
        ModelParameter("c", 0.0, base_units=None),
    )

    OUTPUTS = _CurveOutputs
    UNITS = OUTPUTS(y=u.dimensionless_unscaled)

    DESCRIPTION = "Quadratic polynomial model."
    REFERENCE = "Elementary quadratic function."

    def __init__(self):
        self._register_init()

    def _forward_model(self, variables, parameters):
        x = np.asarray(variables["x"], dtype=float)
        a = parameters["a"]
        b = parameters["b"]
        c = parameters["c"]
        return self.OUTPUTS(y=a * x**2 + b * x + c)


class PowerLawModel(Model):
    r"""
    Power-law model.

    .. math::

        y(x) = A x^{\alpha}

    .. dropdown:: Parameters

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``A``
             - :math:`A`
             - Amplitude normalization.
           * - ``alpha``
             - :math:`\alpha`
             - Power-law index.

    .. dropdown:: Variables

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``x``
             - :math:`x`
             - Independent variable.

    .. dropdown:: Returns

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``y``
             - :math:`y(x)`
             - Evaluated model value.

    Notes
    -----
    The amplitude parameter is constrained to be non-negative.
    """

    VARIABLES = (ModelVariable("x", base_units=None),)

    PARAMETERS = (
        ModelParameter("A", 1.0, base_units=None, bounds=(0.0, None)),
        ModelParameter("alpha", -1.0, base_units=None),
    )

    OUTPUTS = _CurveOutputs
    UNITS = OUTPUTS(y=u.dimensionless_unscaled)

    DESCRIPTION = "Simple power-law model."
    REFERENCE = "Elementary scaling relation."

    def __init__(self):
        self._register_init()

    def _forward_model(self, variables, parameters):
        x = np.asarray(variables["x"], dtype=float)
        A = parameters["A"]
        alpha = parameters["alpha"]
        return self.OUTPUTS(y=A * x**alpha)


class ExponentialModel(Model):
    r"""
    Exponential model.

    .. math::

        y(x) = A e^{k x}

    .. dropdown:: Parameters

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``A``
             - :math:`A`
             - Amplitude normalization.
           * - ``k``
             - :math:`k`
             - Exponential rate constant.

    .. dropdown:: Variables

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``x``
             - :math:`x`
             - Independent variable.

    .. dropdown:: Returns

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``y``
             - :math:`y(x)`
             - Evaluated model value.
    """

    VARIABLES = (ModelVariable("x", base_units=None),)

    PARAMETERS = (
        ModelParameter("A", 1.0, base_units=None),
        ModelParameter("k", -1.0, base_units=None),
    )

    OUTPUTS = _CurveOutputs
    UNITS = OUTPUTS(y=u.dimensionless_unscaled)

    DESCRIPTION = "Simple exponential model."
    REFERENCE = "Elementary exponential function."

    def __init__(self):
        self._register_init()

    def _forward_model(self, variables, parameters):
        x = np.asarray(variables["x"], dtype=float)
        A = parameters["A"]
        k = parameters["k"]
        return self.OUTPUTS(y=A * np.exp(k * x))


class ConstantModel(Model):
    r"""
    Constant model.

    .. math::

        y(x) = C

    .. dropdown:: Parameters

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``C``
             - :math:`C`
             - Constant value.

    .. dropdown:: Variables

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``x``
             - :math:`x`
             - Independent variable (ignored).

    .. dropdown:: Returns

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``y``
             - :math:`C`
             - Constant model value.
    """

    VARIABLES = (ModelVariable("x", base_units=None),)

    PARAMETERS = (ModelParameter("C", 1.0, base_units=None),)

    OUTPUTS = _CurveOutputs
    UNITS = OUTPUTS(y=u.dimensionless_unscaled)

    DESCRIPTION = "Constant function."
    REFERENCE = "Zero-order polynomial."

    def __init__(self):
        self._register_init()

    def _forward_model(self, variables, parameters):
        x = np.asarray(variables["x"], dtype=float)
        C = parameters["C"]
        return self.OUTPUTS(y=np.full_like(x, C))


_BPLOutputs = namedtuple("BPLOutputs", ["y"])


class BrokenPowerLaw(Model):
    r"""
    Sharp two-segment broken power-law model.

    This model describes a continuous but non-differentiable
    two-regime power-law transition at a single break location
    :math:`x_b`.

    The functional form is

    .. math::

        y(x) =
        \begin{cases}
            A \left( \frac{x}{x_b} \right)^{\alpha_1}, & x < x_b \\
            A \left( \frac{x}{x_b} \right)^{\alpha_2}, & x \ge x_b
        \end{cases}

    where:

    - :math:`A` is defined at :math:`x = x_b`,
    - :math:`\alpha_1` is the low-:math:`x` slope,
    - :math:`\alpha_2` is the high-:math:`x` slope.

    This model is purely phenomenological and does not encode
    any physical emission mechanism.

    It is commonly used for:

    - Synchrotron spectral segments
    - Afterglow modeling
    - Multi-phase scaling behavior
    - Empirical curve fitting

    .. dropdown:: Parameters

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``A``
             - :math:`A`
             - Normalization at the break location.
           * - ``x_b``
             - :math:`x_b`
             - Break position.
           * - ``alpha_1``
             - :math:`\alpha_1`
             - Slope for :math:`x < x_b`.
           * - ``alpha_2``
             - :math:`\alpha_2`
             - Slope for :math:`x \ge x_b`.

    .. dropdown:: Variables

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``x``
             - :math:`x`
             - Independent variable.

    .. dropdown:: Returns

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``y``
             - :math:`y(x)`
             - Model value evaluated at :math:`x`.

    Notes
    -----
    - The model is continuous at :math:`x_b`.
    - The first derivative is discontinuous.
    - For smooth transitions, use :class:`SmoothedBrokenPowerLaw`.

    See Also
    --------
    :class:`SmoothedBrokenPowerLaw`: Smooth transition version.
    :class:`TripleBrokenPowerLaw`: Three-segment extension.
    :class:`SmoothedTripleBrokenPowerLaw`: Smooth three-segment version.


    Examples
    --------

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        from triceratops.models.generic.bpl import BrokenPowerLaw

        model = BrokenPowerLaw()

        x = np.logspace(-2, 2, 500)
        y = model({"x": x}, {}).y

        plt.loglog(x, y)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Broken Power Law Example")
        plt.show()
    """

    VARIABLES = (
        ModelVariable(
            name="x",
            base_units=u.dimensionless_unscaled,
            description="Independent variable.",
            latex=r"$x$",
        ),
    )

    PARAMETERS = (
        ModelParameter("A", 1.0, base_units=u.dimensionless_unscaled, bounds=(0.0, None)),
        ModelParameter("x_b", 1.0, base_units=u.dimensionless_unscaled, bounds=(1e-12, None)),
        ModelParameter("alpha_1", -1.0, base_units=u.dimensionless_unscaled, bounds=(-20, 20)),
        ModelParameter("alpha_2", -2.0, base_units=u.dimensionless_unscaled, bounds=(-20, 20)),
    )

    OUTPUTS = _BPLOutputs
    UNITS = OUTPUTS(y=u.dimensionless_unscaled)

    DESCRIPTION = "Sharp two-segment broken power law."
    REFERENCE = "Standard phenomenological broken power law."

    def __init__(self):
        """Initialize the model with given parameters and variables."""
        self._register_init()

    def _forward_model(self, variables, parameters):
        x = np.asarray(variables["x"], dtype=float)

        if np.any(x <= 0):
            raise ValueError("BrokenPowerLaw requires x > 0.")

        A = parameters["A"]
        xb = parameters["x_b"]
        a1 = parameters["alpha_1"]
        a2 = parameters["alpha_2"]

        log_x = np.log(x)
        log_xb = np.log(xb)
        log_ratio = log_x - log_xb

        log_y = np.empty_like(log_ratio)

        mask = x < xb
        log_y[mask] = np.log(A) + a1 * log_ratio[mask]
        log_y[~mask] = np.log(A) + a2 * log_ratio[~mask]

        return self.OUTPUTS(y=np.exp(log_y))


class SmoothedBrokenPowerLaw(Model):
    r"""
    Smooth two-segment broken power-law model.

    This model replaces the sharp transition at :math:`x_b`
    with a differentiable break controlled by a smoothing
    parameter :math:`s`.

    The functional form is

    .. math::

        y(x) =
        A \left[
            \left( \frac{x}{x_b} \right)^{\alpha_1 / s}
            +
            \left( \frac{x}{x_b} \right)^{\alpha_2 / s}
        \right]^s

    where:

    - :math:`A` is defined at :math:`x = x_b`,
    - :math:`x_b` is the break location,
    - :math:`\alpha_1`, :math:`\alpha_2` are asymptotic slopes,
    - :math:`s > 0` controls transition sharpness.

    This model is purely phenomenological.

    It is commonly used when gradient-based inference
    requires differentiability.

    .. dropdown:: Parameters

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``A``
             - :math:`A`
             - Normalization at the break.
           * - ``x_b``
             - :math:`x_b`
             - Break position.
           * - ``alpha_1``
             - :math:`\alpha_1`
             - Low-:math:`x` slope.
           * - ``alpha_2``
             - :math:`\alpha_2`
             - High-:math:`x` slope.
           * - ``s``
             - :math:`s`
             - Smoothing parameter.

    .. dropdown:: Variables

        .. list-table::
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``x``
             - :math:`x`
             - Independent variable.

    .. dropdown:: Returns

        .. list-table::
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``y``
             - :math:`y(x)`
             - Model value.

    Notes
    -----
    - Continuous and differentiable everywhere.
    - As :math:`s \to 0`, approaches :class:`BrokenPowerLaw`.

    See Also
    --------
    :class:`BrokenPowerLaw`
    :class:`SmoothedTripleBrokenPowerLaw`

    Examples
    --------

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        from triceratops.models.generic.bpl import SmoothedBrokenPowerLaw

        model = SmoothedBrokenPowerLaw()

        x = np.logspace(-2, 2, 500)
        y = model({"x": x}, {}).y

        plt.loglog(x, y)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Smoothed Broken Power Law Example")
        plt.show()
    """

    VARIABLES = (ModelVariable("x", base_units=u.dimensionless_unscaled),)

    PARAMETERS = (
        ModelParameter("A", 1.0, base_units=u.dimensionless_unscaled, bounds=(0.0, None)),
        ModelParameter("x_b", 1.0, base_units=u.dimensionless_unscaled, bounds=(1e-12, None)),
        ModelParameter("alpha_1", -1.0, base_units=u.dimensionless_unscaled, bounds=(-20, 20)),
        ModelParameter("alpha_2", -2.0, base_units=u.dimensionless_unscaled, bounds=(-20, 20)),
        ModelParameter("s", 0.1, base_units=u.dimensionless_unscaled, bounds=(None, None)),
    )

    OUTPUTS = _BPLOutputs
    UNITS = OUTPUTS(y=u.dimensionless_unscaled)

    DESCRIPTION = "Smooth two-segment broken power law."
    REFERENCE = "Standard smooth break prescription."

    def __init__(self):
        """Initialize the model with given parameters and variables."""
        self._register_init()

    def _forward_model(self, variables, parameters):
        x = np.asarray(variables["x"], dtype=float)

        if np.any(x <= 0):
            raise ValueError("SmoothedBrokenPowerLaw requires x > 0.")

        A = parameters["A"]
        xb = parameters["x_b"]
        a1 = parameters["alpha_1"]
        a2 = parameters["alpha_2"]
        s = parameters["s"]

        if s == 0:
            raise ValueError("Smoothing parameter s must be non-zero.")

        log_x = np.log(x)
        log_ratio = log_x - np.log(xb)

        log_core = s * np.logaddexp(
            (a1 / s) * log_ratio,
            (a2 / s) * log_ratio,
        )

        log_y = np.log(A) + log_core

        return self.OUTPUTS(y=np.exp(log_y))


class TripleBrokenPowerLaw(Model):
    r"""
    Sharp three-segment broken power-law model.

    This model introduces two break locations,
    :math:`x_{b,1}` and :math:`x_{b,2}`, producing
    three power-law regimes.

    The piecewise form is

    .. math::

        y(x) =
        \begin{cases}
            A \left( \frac{x}{x_{b,1}} \right)^{\alpha_1},
            & x < x_{b,1} \\
            A \left( \frac{x}{x_{b,1}} \right)^{\alpha_2},
            & x_{b,1} \le x < x_{b,2} \\
            A \left( \frac{x_{b,2}}{x_{b,1}} \right)^{\alpha_2}
            \left( \frac{x}{x_{b,2}} \right)^{\alpha_3},
            & x \ge x_{b,2}
        \end{cases}

    This model is phenomenological and allows modeling
    of multi-phase scaling behavior.

    .. dropdown:: Parameters

        .. list-table::
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``A``
             - :math:`A`
             - Normalization.
           * - ``x_b1``
             - :math:`x_{b,1}`
             - First break.
           * - ``x_b2``
             - :math:`x_{b,2}`
             - Second break.
           * - ``alpha_1``
             - :math:`\alpha_1`
             - First slope.
           * - ``alpha_2``
             - :math:`\alpha_2`
             - Second slope.
           * - ``alpha_3``
             - :math:`\alpha_3`
             - Third slope.

    .. dropdown:: Variables

        .. list-table::
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``x``
             - :math:`x`
             - Independent variable.

    .. dropdown:: Returns

        .. list-table::
           :header-rows: 1

           * - ``y``
             - :math:`y`
             - Independent variable.

    Notes
    -----
    - Continuous at both break locations.
    - Derivative is discontinuous.
    - Useful for modeling multi-regime spectra.

    See Also
    --------
    :class:`BrokenPowerLaw`
    :class:`SmoothedTripleBrokenPowerLaw`

    Examples
    --------

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        from triceratops.models.generic.bpl import TripleBrokenPowerLaw

        model = TripleBrokenPowerLaw()

        x = np.logspace(-2, 3, 500)
        y = model({"x": x}, {}).y

        plt.loglog(x, y)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Triple Broken Power Law Example")
        plt.show()
    """

    VARIABLES = (ModelVariable("x", base_units=u.dimensionless_unscaled),)

    PARAMETERS = (
        ModelParameter("A", 1.0, base_units=u.dimensionless_unscaled, bounds=(0.0, None)),
        ModelParameter("x_b1", 1.0, base_units=u.dimensionless_unscaled, bounds=(1e-12, None)),
        ModelParameter("x_b2", 10.0, base_units=u.dimensionless_unscaled, bounds=(1e-12, None)),
        ModelParameter("alpha_1", -0.5, base_units=u.dimensionless_unscaled, bounds=(-20, 20)),
        ModelParameter("alpha_2", -1.5, base_units=u.dimensionless_unscaled, bounds=(-20, 20)),
        ModelParameter("alpha_3", -2.5, base_units=u.dimensionless_unscaled, bounds=(-20, 20)),
    )

    OUTPUTS = _BPLOutputs
    UNITS = OUTPUTS(y=u.dimensionless_unscaled)

    DESCRIPTION = "Sharp three-segment broken power law."
    REFERENCE = "Standard multi-break phenomenological model."

    def __init__(self):
        """Initialize the model with given parameters and variables."""
        self._register_init()

    def _forward_model(self, variables, parameters):
        x = np.asarray(variables["x"], dtype=float)

        if np.any(x <= 0):
            raise ValueError("TripleBrokenPowerLaw requires x > 0.")

        A = parameters["A"]
        b1 = parameters["x_b1"]
        b2 = parameters["x_b2"]
        a1 = parameters["alpha_1"]
        a2 = parameters["alpha_2"]
        a3 = parameters["alpha_3"]

        log_x = np.log(x)
        log_b1 = np.log(b1)
        log_b2 = np.log(b2)

        log_ratio1 = log_x - log_b1
        log_ratio2 = log_x - log_b2

        log_y = np.empty_like(log_x)

        m1 = x < b1
        m2 = (x >= b1) & (x < b2)
        m3 = x >= b2

        log_A = np.log(A)

        log_y[m1] = log_A + a1 * log_ratio1[m1]
        log_y[m2] = log_A + a2 * log_ratio1[m2]
        log_y[m3] = log_A + a2 * (log_b2 - log_b1) + a3 * log_ratio2[m3]

        return self.OUTPUTS(y=np.exp(log_y))


class SmoothedTripleBrokenPowerLaw(Model):
    r"""
    Smooth three-segment broken power-law model.

    This model introduces two smooth transitions using
    a shared smoothing parameter :math:`s`.

    It is equivalent to nesting two smooth broken
    power laws, ensuring differentiability across
    all regimes.

    .. dropdown:: Parameters

        - ``A`` – Normalization
        - ``x_b1`` – First break
        - ``x_b2`` – Second break
        - ``alpha_1`` – First slope
        - ``alpha_2`` – Second slope
        - ``alpha_3`` – Third slope
        - ``s`` – Smoothing parameter

    .. dropdown:: Variables

        - ``x`` – Independent variable

    .. dropdown:: Returns

        - ``y`` – Model value

    Notes
    -----
    - Continuous and differentiable everywhere.
    - Approaches :class:`TripleBrokenPowerLaw`
      as :math:`s \to 0`.
    - Recommended for inference workflows.

    See Also
    --------
    :class:`TripleBrokenPowerLaw`
    :class:`SmoothedBrokenPowerLaw`

    Examples
    --------

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        from triceratops.models.generic.bpl import SmoothedTripleBrokenPowerLaw

        model = SmoothedTripleBrokenPowerLaw()

        x = np.logspace(-2, 3, 500)
        y = model({"x": x}, {}).y

        plt.loglog(x, y)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Smoothed Triple Broken Power Law Example")
        plt.show()
    """

    VARIABLES = (ModelVariable("x", base_units=u.dimensionless_unscaled),)

    PARAMETERS = (
        ModelParameter("A", 1.0, base_units=u.dimensionless_unscaled, bounds=(0.0, None)),
        ModelParameter("x_b1", 1.0, base_units=u.dimensionless_unscaled, bounds=(1e-12, None)),
        ModelParameter("x_b2", 10.0, base_units=u.dimensionless_unscaled, bounds=(1e-12, None)),
        ModelParameter("alpha_1", -0.5, base_units=u.dimensionless_unscaled, bounds=(-20, 20)),
        ModelParameter("alpha_2", -1.5, base_units=u.dimensionless_unscaled, bounds=(-20, 20)),
        ModelParameter("alpha_3", -2.5, base_units=u.dimensionless_unscaled, bounds=(-20, 20)),
        ModelParameter("s", 0.1, base_units=u.dimensionless_unscaled, bounds=(None, None)),
    )

    OUTPUTS = _BPLOutputs
    UNITS = OUTPUTS(y=u.dimensionless_unscaled)

    DESCRIPTION = "Smooth three-segment broken power law."
    REFERENCE = "Nested smooth break prescription."

    def __init__(self):
        """Initialize the model with given parameters and variables."""
        self._register_init()

    def _forward_model(self, variables, parameters):
        x = np.asarray(variables["x"], dtype=float)

        if np.any(x <= 0):
            raise ValueError("SmoothedTripleBrokenPowerLaw requires x > 0.")

        A = parameters["A"]
        b1 = parameters["x_b1"]
        b2 = parameters["x_b2"]
        a1 = parameters["alpha_1"]
        a2 = parameters["alpha_2"]
        a3 = parameters["alpha_3"]
        s = parameters["s"]

        if s == 0:
            raise ValueError("Smoothing parameter s must be non-zero.")

        log_x = np.log(x)
        log_b1 = np.log(b1)
        log_b2 = np.log(b2)

        log_ratio1 = log_x - log_b1
        log_ratio2 = log_x - log_b2

        # First transition
        log_t12 = s * np.logaddexp(
            (a1 / s) * log_ratio1,
            (a2 / s) * log_ratio1,
        )

        # Second transition
        log_t23 = s * np.logaddexp(
            (a2 / s) * log_ratio2,
            (a3 / s) * log_ratio2,
        )

        # Remove double-counted middle slope
        log_middle = a2 * log_ratio2

        log_y = np.log(A) + log_t12 + (log_t23 - log_middle)

        return self.OUTPUTS(y=np.exp(log_y))
