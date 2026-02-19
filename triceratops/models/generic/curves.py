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

All models follow the standard :class:`~triceratops.models.core.Model`
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
