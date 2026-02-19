"""
Broken power-law models and associated utilities.

This module provides generic phenomenological curve models commonly used in
astrophysical spectral and time-domain analysis. Implementations include:

- Sharp two-segment broken power laws
- Smoothly broken two-segment power laws
- Sharp three-segment broken power laws
- Smooth three-segment broken power laws

All models:

- Map a single independent variable ``x`` to a single dependent variable ``y``.
- Are compatible with generic XY likelihood frameworks.
- Support unit-aware modeling via the ``Model`` base class.

These forms are widely used for modeling:

- Synchrotron spectra
- GRB afterglows
- TDE fallback curves
- Non-thermal emission processes
- Multi-phase lightcurves
"""

from collections import namedtuple

import numpy as np
from astropy import units as u

from triceratops.models.core import Model, ModelParameter, ModelVariable

__all__ = [
    "BrokenPowerLaw",
    "SmoothedBrokenPowerLaw",
    "TripleBrokenPowerLaw",
    "SmoothedTripleBrokenPowerLaw",
]

# ============================================================
# Shared Output Definitions
# ============================================================
_BPLOutputs = namedtuple("BPLOutputs", ["y"])


# ============================================================
# Sharp Broken Power Law
# ============================================================
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

        A = parameters["A"]
        xb = parameters["x_b"]
        a1 = parameters["alpha_1"]
        a2 = parameters["alpha_2"]

        y = np.empty_like(x)

        mask = x < xb
        y[mask] = A * (x[mask] / xb) ** a1
        y[~mask] = A * (x[~mask] / xb) ** a2

        return self.OUTPUTS(y=y)


# ============================================================
# Smoothed Broken Power Law
# ============================================================
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

        A = parameters["A"]
        xb = parameters["x_b"]
        a1 = parameters["alpha_1"]
        a2 = parameters["alpha_2"]
        s = parameters["s"]

        term = ((x / xb) ** (a1 / s) + (x / xb) ** (a2 / s)) ** s
        return self.OUTPUTS(y=A * term)


# ============================================================
# Triple Broken Power Law
# ============================================================
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
             — :math:`A`
             — Normalization.
           * - ``x_b1``
             — :math:`x_{b,1}`
             — First break.
           * - ``x_b2``
             — :math:`x_{b,2}`
             — Second break.
           * - ``alpha_1``
             — :math:`\alpha_1`
             — First slope.
           * - ``alpha_2``
             — :math:`\alpha_2`
             — Second slope.
           * - ``alpha_3``
             — :math:`\alpha_3`
             — Third slope.

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

        A = parameters["A"]
        b1 = parameters["x_b1"]
        b2 = parameters["x_b2"]
        a1 = parameters["alpha_1"]
        a2 = parameters["alpha_2"]
        a3 = parameters["alpha_3"]

        y = np.empty_like(x)

        m1 = x < b1
        m2 = (x >= b1) & (x < b2)
        m3 = x >= b2

        y[m1] = A * (x[m1] / b1) ** a1
        y[m2] = A * (x[m2] / b1) ** a2
        y[m3] = A * (b2 / b1) ** a2 * (x[m3] / b2) ** a3

        return self.OUTPUTS(y=y)


# ============================================================
# Smoothed Triple Broken Power Law
# ============================================================
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

        A = parameters["A"]
        b1 = parameters["x_b1"]
        b2 = parameters["x_b2"]
        a1 = parameters["alpha_1"]
        a2 = parameters["alpha_2"]
        a3 = parameters["alpha_3"]
        s = parameters["s"]

        t12 = ((x / b1) ** (a1 / s) + (x / b1) ** (a2 / s)) ** s
        t23 = ((x / b2) ** (a2 / s) + (x / b2) ** (a3 / s)) ** s

        y = A * t12 * (t23 / (x / b2) ** a2)

        return self.OUTPUTS(y=y)
