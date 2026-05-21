r"""
Phenomenological flare models for time-domain lightcurve analysis.

This module provides lightweight, analytic models for asymmetric transient
pulses commonly observed in high-energy and time-domain astrophysics.
In particular, it implements variants of the Fast Rise, Exponential Decay
(FRED) functional form, which is widely used in gamma-ray burst (GRB)
prompt emission modeling, X-ray flares, and other transient phenomena.

The models defined here are **purely phenomenological**. They are intended
to provide flexible curve-fitting templates rather than enforce physical
self-consistency or dynamical constraints. No assumptions are made about
radiative transfer, shock physics, energy injection, or emission mechanisms.

Both models:

- Map a single time variable ``t`` to a single output ``flux``.
- Support vectorized evaluation and NumPy broadcasting.
- Include optional constant background components.
- Are compatible with generic XY likelihood frameworks.

These models are particularly useful for:

- GRB pulse decomposition,
- Flare timing analysis,
- Exploratory transient fitting,
- Template-based inference workflows.

They should not be interpreted as physically predictive models.
When physical interpretation is required, more detailed dynamical
or radiative models should be used.

Notes
-----
Because these models are analytic and smooth (except at the onset time),
they are generally well-behaved under gradient-free inference methods
such as MCMC and nested sampling.

Care should be taken when fitting overlapping pulses, as parameter
degeneracies between amplitude, rise scale, and background level
can become significant.
"""

from collections import namedtuple

import numpy as np
from astropy import units as u

from ..core import Model, ModelParameter, ModelVariable

__all__ = [
    "FRED",
    "GeneralizedFRED",
    "GaussianPulse",
    "LogNormalPulse",
    "BrokenPowerLawTime",
    "SmoothedBrokenPowerLawTime",
    "ExponentialRisePowerLawDecay",
    "NorrisPulse",
    "WeibullPulse",
    "LogisticPulse",
]

_FredOutputs = namedtuple("FREDOutputs", ["flux"])


class FRED(Model):
    r"""
    Fast Rise, Exponential Decay (FRED) lightcurve model.

    This model describes a phenomenological flare profile commonly used
    in gamma-ray burst (GRB) pulse modeling and time-domain astrophysics (e.g. :footcite:t:`2009ApJ...698..417P`).
    The functional form consists of a rapid exponential rise followed
    by an exponential decay.

    The lightcurve is defined as

    .. math::

        F(t) =
        \begin{cases}
            C, & t < t_0 \\
            A \left(1 - e^{-(t - t_0)/\tau_r}\right)
            e^{-(t - t_0)/\tau_d} + C,
            & t \ge t_0
        \end{cases}

    where:

    - :math:`A` controls the flare amplitude,
    - :math:`t_0` defines the onset time,
    - :math:`\tau_r` sets the rise timescale,
    - :math:`\tau_d` sets the decay timescale,
    - :math:`C` is a constant background level.

    This model is purely phenomenological and does not encode
    physical emission mechanisms.

    It is most appropriate for:

    - isolated transient pulses,
    - GRB prompt emission pulse fitting,
    - supernova shock breakout lightcurve approximations,
    - exploratory flare modeling.

    .. dropdown:: Parameters

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``A``
             - :math:`A`
             - Flare amplitude scale.
           * - ``t_0``
             - :math:`t_0`
             - Flare onset time.
           * - ``tau_r``
             - :math:`\tau_r`
             - Rise timescale.
           * - ``tau_d``
             - :math:`\tau_d`
             - Decay timescale.
           * - ``C``
             - :math:`C`
             - Constant background level.

    .. dropdown:: Variables

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``t``
             - :math:`t`
             - Time at which the lightcurve is evaluated.

    .. dropdown:: Returns

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``flux``
             - :math:`F(t)`
             - Flux evaluated at time :math:`t`.

    Notes
    -----
    The peak time depends non-trivially on the ratio :math:`\tau_r/\tau_d`.
    The model reduces to a simple exponential decay when
    :math:`\tau_r \rightarrow 0`.

    Examples
    --------

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        from trilobite.models.generic.light_curve import FRED

        rng = np.random.default_rng(42)

        model = FRED()

        t = np.linspace(-1, 10, 400)
        flux = model({"t": t}, {}).flux

        noise = 0.05 * np.max(flux) * rng.normal(size=t.size)
        synthetic = flux + noise

        plt.plot(t, flux, label="Model", lw=2)
        plt.scatter(t, synthetic, s=8, alpha=0.5, label="Synthetic Data")

        plt.xlabel("Time")
        plt.ylabel("Flux")
        plt.title("FRED Model Example")
        plt.legend()
        plt.show()

    See Also
    --------
    :class:`GeneralizedFRED`
    :class:`GaussianPulse`
    :class:`NorrisPulse`

    References
    ----------
    .. footbibliography::
    """

    VARIABLES = (
        ModelVariable(
            name="t",
            base_units=None,
            description="Time.",
            latex=r"$t$",
        ),
    )

    PARAMETERS = (
        ModelParameter("A", 1.0, base_units=None, bounds=(0.0, None)),
        ModelParameter("t_0", 0.0, base_units=None),
        ModelParameter("tau_r", 0.1, base_units=None, bounds=(1e-6, None)),
        ModelParameter("tau_d", 1.0, base_units=None, bounds=(1e-6, None)),
        ModelParameter("C", 0.0, base_units=None, bounds=(None, None)),
    )

    OUTPUTS = _FredOutputs
    UNITS = OUTPUTS(flux=u.dimensionless_unscaled)

    DESCRIPTION = "Fast rise, exponential decay flare model with constant background."
    REFERENCE = "Common phenomenological GRB pulse model."

    def __init__(self):
        """Initialize the model with given parameters and variables."""
        self._register_init()

    def _forward_model(self, variables, parameters):
        t = np.asarray(variables["t"], dtype=float)

        A = parameters["A"]
        t0 = parameters["t_0"]
        tau_r = parameters["tau_r"]
        tau_d = parameters["tau_d"]
        C = parameters["C"]

        y = np.full_like(t, C, dtype=float)

        mask = t >= t0
        dt = t[mask] - t0

        rise = 1.0 - np.exp(-dt / tau_r)
        decay = np.exp(-dt / tau_d)

        y[mask] += A * rise * decay

        return self.OUTPUTS(flux=y)


class GeneralizedFRED(Model):
    r"""
    Generalized Fast Rise, Exponential Decay (FRED) lightcurve model.

    This model extends the classical FRED pulse profile by replacing the
    exponential rise with a power-law rise of index :math:`n`. It provides
    additional flexibility in modeling asymmetric transient pulses commonly
    observed in gamma-ray bursts (GRBs), X-ray flares, and other time-domain
    astrophysical phenomena (see e.g. :footcite:t:`ryde1999shape`).

    The lightcurve is defined as

    .. math::

        F(t) =
        \begin{cases}
            C, & t < t_0 \\
            A \left(\frac{t - t_0}{\tau_r}\right)^n
            \exp\left(-\frac{t - t_0}{\tau_d}\right) + C,
            & t \ge t_0
        \end{cases}

    where :math:`A` sets the amplitude scale and :math:`C` is a constant
    background level.

    This model is purely phenomenological and does not assume any
    specific emission mechanism or dynamical framework.

    It is most appropriate for:

    - flexible GRB pulse fitting,
    - X-ray flare modeling,
    - asymmetric transient pulse decomposition,
    - exploratory time-domain curve fitting.

    Compared to the standard FRED model, this formulation allows the
    early-time rise to vary from shallow (:math:`n \approx 1`) to sharply
    peaked (:math:`n \gg 1`) behavior.

    .. dropdown:: Parameters

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``A``
             - :math:`A`
             - Flare amplitude normalization.
           * - ``t_0``
             - :math:`t_0`
             - Flare onset time.
           * - ``tau_r``
             - :math:`\tau_r`
             - Characteristic rise scale.
           * - ``tau_d``
             - :math:`\tau_d`
             - Exponential decay timescale.
           * - ``n``
             - :math:`n`
             - Power-law rise index controlling early-time steepness.
           * - ``C``
             - :math:`C`
             - Constant background level.

    .. dropdown:: Variables

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``t``
             - :math:`t`
             - Time at which the lightcurve is evaluated.

    .. dropdown:: Returns

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``flux``
             - :math:`F(t)`
             - Flux evaluated at time :math:`t`.

    Notes
    -----
    - The peak time depends on both :math:`n` and the ratio
      :math:`\tau_r / \tau_d`.
    - For :math:`n = 1`, the rise becomes linear in time prior to decay.
    - Large :math:`n` produces sharper, more impulsive flares.
    - As :math:`\tau_d \rightarrow \infty`, the model reduces to a pure
      power-law rise plus constant background.

    Caution should be exercised when interpreting fitted parameters
    physically, as this model does not enforce energy conservation or
    radiative transfer consistency.

    Examples
    --------

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        from trilobite.models.generic.light_curve import GeneralizedFRED

        rng = np.random.default_rng(42)

        model = GeneralizedFRED()

        t = np.linspace(-1, 10, 400)
        flux = model({"t": t}, {}).flux

        noise = 0.05 * np.max(flux) * rng.normal(size=t.size)
        synthetic = flux + noise

        plt.plot(t, flux, lw=2, label="Model")
        plt.scatter(t, synthetic, s=8, alpha=0.5, label="Synthetic Data")

        plt.xlabel("Time")
        plt.ylabel("Flux")
        plt.title("Generalized FRED Example")
        plt.legend()
        plt.show()

    See Also
    --------
    :class:`FRED`
    :class:`NorrisPulse`
    :class:`LogNormalPulse`

    References
    ----------
    .. footbibliography::
    """

    VARIABLES = (ModelVariable("t", base_units=None),)

    PARAMETERS = (
        ModelParameter("A", 1.0, base_units=None, bounds=(0.0, None)),
        ModelParameter("t_0", 0.0, base_units=None),
        ModelParameter("tau_r", 0.1, base_units=None, bounds=(1e-6, None)),
        ModelParameter("tau_d", 1.0, base_units=None, bounds=(1e-6, None)),
        ModelParameter("n", 2.0, base_units=None, bounds=(0.0, 10.0)),
        ModelParameter("C", 0.0, base_units=None),
    )

    OUTPUTS = namedtuple("GenFREDOutputs", ["flux"])
    UNITS = OUTPUTS(flux=u.dimensionless_unscaled)

    DESCRIPTION = "Generalized FRED with power-law rise and constant background."
    REFERENCE = "Extended phenomenological GRB pulse model."

    def __init__(self):
        """Initialize the model with given parameters and variables."""
        self._register_init()

    def _forward_model(self, variables, parameters):
        t = np.asarray(variables["t"], dtype=float)

        A = parameters["A"]
        t0 = parameters["t_0"]
        tau_r = parameters["tau_r"]
        tau_d = parameters["tau_d"]
        n = parameters["n"]
        C = parameters["C"]

        y = np.full_like(t, C, dtype=float)

        mask = t >= t0
        dt = t[mask] - t0

        rise = (dt / tau_r) ** n
        decay = np.exp(-dt / tau_d)

        y[mask] += A * rise * decay

        return self.OUTPUTS(flux=y)


_PulseOutputs = namedtuple("PulseOutputs", ["flux"])


class GaussianPulse(Model):
    r"""
    Symmetric Gaussian pulse model with constant background.

    This model describes a symmetric transient pulse centered at
    :math:`t_0`, with characteristic width :math:`\sigma`. The profile
    is purely Gaussian in time and therefore symmetric about the peak.

    The lightcurve is defined as

    .. math::

        F(t) =
        A \exp\left[
            -\frac{(t - t_0)^2}{2\sigma^2}
        \right] + C,

    where:

    - :math:`A` sets the peak amplitude,
    - :math:`t_0` is the peak time,
    - :math:`\sigma` controls the pulse width,
    - :math:`C` is a constant background level.

    This model is analytic and infinitely differentiable.

    It is most appropriate for:

    - Symmetric flare fitting,
    - Approximate injection-like variability,
    - Situations where asymmetry is negligible,
    - Baseline comparisons to asymmetric pulse models.

    .. dropdown:: Parameters

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``A``
             - :math:`A`
             - Peak amplitude above background.
           * - ``t_0``
             - :math:`t_0`
             - Time of pulse maximum.
           * - ``sigma``
             - :math:`\sigma`
             - Characteristic width of the pulse.
           * - ``C``
             - :math:`C`
             - Constant background level.

    .. dropdown:: Variables

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``t``
             - :math:`t`
             - Time at which the pulse is evaluated.

    .. dropdown:: Returns

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``flux``
             - :math:`F(t)`
             - Flux evaluated at time :math:`t`.

    Notes
    -----
    The full width at half maximum (FWHM) is given by

    .. math::

        \mathrm{FWHM} = 2\sqrt{2\ln 2}\,\sigma.

    Because this model is symmetric, it cannot capture the fast-rise,
    slow-decay morphology typical of GRB pulses or shock-powered flares.

    Examples
    --------

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        from trilobite.models.generic.light_curve import GaussianPulse

        rng = np.random.default_rng(42)

        model = GaussianPulse()

        t = np.linspace(-5, 5, 400)
        flux = model({"t": t}, {}).flux

        noise = 0.05 * np.max(flux) * rng.normal(size=t.size)
        synthetic = flux + noise

        plt.plot(t, flux, lw=2, label="Model")
        plt.scatter(t, synthetic, s=8, alpha=0.5, label="Synthetic Data")

        plt.xlabel("Time")
        plt.ylabel("Flux")
        plt.title("Gaussian Pulse Example")
        plt.legend()
        plt.show()

    See Also
    --------
    :class:`FRED`
    :class:`LogNormalPulse`

    """

    VARIABLES = (ModelVariable("t", base_units=None),)

    PARAMETERS = (
        ModelParameter("A", 1.0, base_units=None, bounds=(0.0, None)),
        ModelParameter("t_0", 0.0, base_units=None),
        ModelParameter("sigma", 1.0, base_units=None, bounds=(1e-6, None)),
        ModelParameter("C", 0.0, base_units=None),
    )

    OUTPUTS = _PulseOutputs
    UNITS = OUTPUTS(flux=u.dimensionless_unscaled)

    DESCRIPTION = "Symmetric Gaussian pulse."
    REFERENCE = "Standard analytic Gaussian."

    def __init__(self):
        """Initialize the model with given parameters and variables."""
        self._register_init()

    def _forward_model(self, variables, parameters):
        t = np.asarray(variables["t"], dtype=float)
        A, t0, sigma, C = (
            parameters["A"],
            parameters["t_0"],
            parameters["sigma"],
            parameters["C"],
        )

        y = A * np.exp(-0.5 * ((t - t0) / sigma) ** 2) + C
        return self.OUTPUTS(flux=y)


class LogNormalPulse(Model):
    r"""
    Log-normal asymmetric pulse model with constant background.

    This model describes an intrinsically asymmetric pulse
    characterized by a log-normal shape in time relative to
    an onset time :math:`t_0`.

    The lightcurve is defined as

    .. math::

        F(t) =
        \begin{cases}
            C, & t \le t_0 \\
            A \exp\left[
                -\frac{(\ln(t - t_0) - \mu)^2}{2\sigma^2}
            \right] + C,
            & t > t_0
        \end{cases}

    where:

    - :math:`\mu` controls the location of the peak in logarithmic time,
    - :math:`\sigma` sets the width and asymmetry of the pulse.

    This form naturally produces a rapid rise and slower decay,
    making it well suited for GRB pulse modeling and solar flare
    lightcurves.

    .. dropdown:: Parameters

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``A``
             - :math:`A`
             - Amplitude scale.
           * - ``t_0``
             - :math:`t_0`
             - Onset time.
           * - ``mu``
             - :math:`\mu`
             - Logarithmic location parameter.
           * - ``sigma``
             - :math:`\sigma`
             - Logarithmic width parameter.
           * - ``C``
             - :math:`C`
             - Constant background.

    .. dropdown:: Variables

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``t``
             - :math:`t`
             - Evaluation time.

    .. dropdown:: Returns

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``flux``
             - :math:`F(t)`
             - Flux at time :math:`t`.

    Notes
    -----
    The peak time occurs at

    .. math::

        t_{\rm peak} = t_0 + \exp(\mu - \sigma^2).

    This model produces strong asymmetry and is widely used in GRB
    prompt emission pulse fitting.

    Examples
    --------

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        from trilobite.models.generic.light_curve import LogNormalPulse

        rng = np.random.default_rng(42)

        model = LogNormalPulse()

        t = np.linspace(0.01, 10, 400)
        flux = model({"t": t}, {}).flux

        noise = 0.05 * np.max(flux) * rng.normal(size=t.size)
        synthetic = flux + noise

        plt.plot(t, flux, lw=2, label="Model")
        plt.scatter(t, synthetic, s=8, alpha=0.5, label="Synthetic Data")

        plt.xlabel("Time")
        plt.ylabel("Flux")
        plt.title("Log-Normal Pulse Example")
        plt.legend()
        plt.show()

    See Also
    --------
    :class:`FRED`
    :class:`NorrisPulse`

    """

    VARIABLES = (ModelVariable("t", base_units=None),)

    PARAMETERS = (
        ModelParameter("A", 1.0, base_units=None, bounds=(0.0, None)),
        ModelParameter("t_0", 0.0, base_units=None),
        ModelParameter("mu", 0.0, base_units=None),
        ModelParameter("sigma", 1.0, base_units=None, bounds=(1e-6, None)),
        ModelParameter("C", 0.0, base_units=None),
    )

    OUTPUTS = _PulseOutputs
    UNITS = OUTPUTS(flux=u.dimensionless_unscaled)

    DESCRIPTION = "Asymmetric log-normal pulse."
    REFERENCE = "Common GRB pulse model."

    def __init__(self):
        """Initialize the model with given parameters and variables."""
        self._register_init()

    def _forward_model(self, variables, parameters):
        t = np.asarray(variables["t"], dtype=float)
        A, t0, mu, sigma, C = (
            parameters["A"],
            parameters["t_0"],
            parameters["mu"],
            parameters["sigma"],
            parameters["C"],
        )

        y = np.full_like(t, C, dtype=float)
        mask = t > t0
        dt = t[mask] - t0

        y[mask] += A * np.exp(-((np.log(dt) - mu) ** 2) / (2 * sigma**2))
        return self.OUTPUTS(flux=y)


class BrokenPowerLawTime(Model):
    r"""
    Sharp broken power-law temporal model with constant background.

    This model describes a piecewise power-law transition at
    time :math:`t_b`.

    The lightcurve is defined as

    .. math::

        F(t) =
        \begin{cases}
            A \left(\frac{t}{t_b}\right)^{\alpha_1} + C,
            & t < t_b \\
            A \left(\frac{t}{t_b}\right)^{\alpha_2} + C,
            & t \ge t_b
        \end{cases}

    This form is commonly used in:

    - GRB afterglow fitting,
    - Tidal disruption event lightcurves,
    - Shock-powered transient evolution.

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
           * - ``t_b``
             - :math:`t_b`
             - Break time.
           * - ``alpha_1``
             - :math:`\alpha_1`
             - Early-time slope.
           * - ``alpha_2``
             - :math:`\alpha_2`
             - Late-time slope.
           * - ``C``
             - :math:`C`
             - Background level.

    .. dropdown:: Variables

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``t``
             - :math:`t`
             - Evaluation time.

    .. dropdown:: Returns

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``flux``
             - :math:`F(t)`
             - Flux value.

    Notes
    -----
    The derivative is discontinuous at :math:`t_b`. This model is therefore
    not differentiable at the break.

    Examples
    --------

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        from trilobite.models.generic.light_curve import BrokenPowerLawTime

        rng = np.random.default_rng(42)

        model = BrokenPowerLawTime()

        t = np.logspace(-1, 2, 400)
        flux = model({"t": t}, {}).flux

        noise = 0.1 * flux * rng.normal(size=t.size)
        synthetic = flux + noise

        plt.loglog(t, flux, lw=2, label="Model")
        plt.scatter(t, synthetic, s=8, alpha=0.5, label="Synthetic Data")

        plt.xlabel("Time")
        plt.ylabel("Flux")
        plt.title("Broken Power Law Time Example")
        plt.legend()
        plt.show()

    See Also
    --------
    :class:`SmoothedBrokenPowerLawTime`

    """

    VARIABLES = (ModelVariable("t", base_units=None),)

    PARAMETERS = (
        ModelParameter("A", 1.0, base_units=None),
        ModelParameter("t_b", 1.0, base_units=None, bounds=(1e-6, None)),
        ModelParameter("alpha_1", -1.0, base_units=None),
        ModelParameter("alpha_2", -2.0, base_units=None),
        ModelParameter("C", 0.0, base_units=None),
    )
    OUTPUTS = _PulseOutputs
    UNITS = OUTPUTS(flux=u.dimensionless_unscaled)

    DESCRIPTION = "Sharp broken power-law in time."
    REFERENCE = "Standard afterglow fitting form."

    def __init__(self):
        """Initialize the model with given parameters and variables."""
        self._register_init()

    def _forward_model(self, variables, parameters):
        t = np.asarray(variables["t"], dtype=float)
        A, tb, a1, a2, C = (
            parameters["A"],
            parameters["t_b"],
            parameters["alpha_1"],
            parameters["alpha_2"],
            parameters["C"],
        )

        y = np.full_like(t, C)
        mask = t < tb
        y[mask] += A * (t[mask] / tb) ** a1
        y[~mask] += A * (t[~mask] / tb) ** a2
        return self.OUTPUTS(flux=y)


class SmoothedBrokenPowerLawTime(Model):
    r"""
    Smoothly broken power-law temporal model.

    This model provides a differentiable transition between two
    temporal power-law regimes.

    The lightcurve is defined as

    .. math::

        F(t) =
        A \left[
            \left(\frac{t}{t_b}\right)^{\alpha_1/s}
            +
            \left(\frac{t}{t_b}\right)^{\alpha_2/s}
        \right]^s + C,

    where :math:`s` controls the smoothness of the transition.

    - Small :math:`s` → sharp break
    - Large :math:`s` → smooth transition

    This form is widely used in GRB afterglow modeling.

    .. dropdown:: Parameters

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``A``
             - :math:`A`
             - Normalization factor at the break time.
           * - ``t_b``
             - :math:`t_b`
             - Characteristic break time between temporal regimes.
           * - ``alpha_1``
             - :math:`\alpha_1`
             - Early-time power-law slope.
           * - ``alpha_2``
             - :math:`\alpha_2`
             - Late-time power-law slope.
           * - ``s``
             - :math:`s`
             - Smoothness parameter controlling the sharpness of the break. Smaller values approach a sharp break.
           * - ``C``
             - :math:`C`
             - Constant background level.

    .. dropdown:: Variables

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``t``
             - :math:`t`
             - Time at which the model is evaluated.

    .. dropdown:: Returns

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``flux``
             - :math:`F(t)`
             - Flux evaluated at time :math:`t`.

    Notes
    -----
    This model is continuous and differentiable for all
    :math:`t > 0`.

    Examples
    --------

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        from trilobite.models.generic.light_curve import SmoothedBrokenPowerLawTime

        rng = np.random.default_rng(42)

        model = SmoothedBrokenPowerLawTime()

        t = np.logspace(-1, 2, 400)
        flux = model({"t": t}, {}).flux

        noise = 0.1 * flux * rng.normal(size=t.size)
        synthetic = flux + noise

        plt.loglog(t, flux, lw=2, label="Model")
        plt.scatter(t, synthetic, s=8, alpha=0.5, label="Synthetic Data")

        plt.xlabel("Time")
        plt.ylabel("Flux")
        plt.title("Smoothed Broken Power Law Time Example")
        plt.legend()
        plt.show()

    See Also
    --------
    :class:`BrokenPowerLawTime`

    """

    VARIABLES = (ModelVariable("t", base_units=None),)

    PARAMETERS = (
        ModelParameter("A", 1.0, base_units=None),
        ModelParameter("t_b", 1.0, base_units=None, bounds=(1e-6, None)),
        ModelParameter("alpha_1", -1.0, base_units=None),
        ModelParameter("alpha_2", -2.0, base_units=None),
        ModelParameter("s", 0.1, base_units=None, bounds=(1e-3, None)),
        ModelParameter("C", 0.0, base_units=None),
    )

    OUTPUTS = _PulseOutputs
    UNITS = OUTPUTS(flux=u.dimensionless_unscaled)

    DESCRIPTION = "Smooth broken power-law in time."
    REFERENCE = "Standard smooth break prescription."

    def __init__(self):
        """Initialize the model with given parameters and variables."""
        self._register_init()

    def _forward_model(self, variables, parameters):
        t = np.asarray(variables["t"], dtype=float)
        A, tb, a1, a2, s, C = (
            parameters["A"],
            parameters["t_b"],
            parameters["alpha_1"],
            parameters["alpha_2"],
            parameters["s"],
            parameters["C"],
        )

        term = ((t / tb) ** (a1 / s) + (t / tb) ** (a2 / s)) ** s
        return self.OUTPUTS(flux=A * term + C)


class ExponentialRisePowerLawDecay(Model):
    r"""
    Exponential rise with power-law decay model.

    This model describes a transient with a smooth exponential
    turn-on followed by power-law decay.

    .. math::

        F(t) =
        \begin{cases}
            C, & t \le t_0 \\
            A \left(1 - e^{-(t - t_0)/\tau_r}\right)
            (t - t_0)^{-\alpha} + C,
            & t > t_0
        \end{cases}

    Used in:

    - Shock interaction models,
    - Supernova circumstellar interaction,
    - Late-time transient decay modeling.

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
           * - ``t_0``
             - :math:`t_0`
             - Onset time of the transient.
           * - ``tau_r``
             - :math:`\tau_r`
             - Characteristic rise timescale.
           * - ``alpha``
             - :math:`\alpha`
             - Power-law decay index.
           * - ``C``
             - :math:`C`
             - Constant background level.

    .. dropdown:: Variables

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``t``
             - :math:`t`
             - Time at which the model is evaluated.

    .. dropdown:: Returns

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``flux``
             - :math:`F(t)`
             - Flux evaluated at time :math:`t`.


    Notes
    -----
    For large times, the decay approaches
    :math:`F(t) \propto t^{-\alpha}`.

    Examples
    --------

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        from trilobite.models.generic.light_curve import ExponentialRisePowerLawDecay

        rng = np.random.default_rng(42)

        model = ExponentialRisePowerLawDecay()

        t = np.logspace(-2, 2, 400)
        flux = model({"t": t}, {}).flux

        noise = 0.1 * flux * rng.normal(size=t.size)
        synthetic = flux + noise

        plt.loglog(t, flux, lw=2, label="Model")
        plt.scatter(t, synthetic, s=8, alpha=0.5, label="Synthetic Data")

        plt.xlabel("Time")
        plt.ylabel("Flux")
        plt.title("Exponential Rise Power-Law Decay Example")
        plt.legend()
        plt.show()

    See Also
    --------
    :class:`BrokenPowerLawTime`

    """

    VARIABLES = (ModelVariable("t", base_units=None),)

    PARAMETERS = (
        ModelParameter("A", 1.0, base_units=None),
        ModelParameter("t_0", 0.0, base_units=None),
        ModelParameter("tau_r", 1.0, base_units=None, bounds=(1e-6, None)),
        ModelParameter("alpha", 2.0, base_units=None),
        ModelParameter("C", 0.0, base_units=None),
    )

    OUTPUTS = _PulseOutputs
    UNITS = OUTPUTS(flux=u.dimensionless_unscaled)

    DESCRIPTION = "Exponential rise with power-law decay."
    REFERENCE = "Common transient decay model."

    def __init__(self):
        """Initialize the model with given parameters and variables."""
        self._register_init()

    def _forward_model(self, variables, parameters):
        t = np.asarray(variables["t"], dtype=float)
        A, t0, tau_r, alpha, C = (
            parameters["A"],
            parameters["t_0"],
            parameters["tau_r"],
            parameters["alpha"],
            parameters["C"],
        )

        y = np.full_like(t, C)
        mask = t > t0
        dt = t[mask] - t0

        rise = 1 - np.exp(-dt / tau_r)
        decay = dt ** (-alpha)

        y[mask] += A * rise * decay
        return self.OUTPUTS(flux=y)


class NorrisPulse(Model):
    r"""
    Norris GRB pulse model.

    This model was introduced to describe asymmetric GRB
    prompt emission pulses in :footcite:t:`norris1996attributes` and :footcite:t:`norris2005long`.

    .. math::

        F(t) =
        \begin{cases}
            C, & t \le t_s \\
            A \exp\left(
                -\frac{\tau_1}{t - t_s}
                -\frac{t - t_s}{\tau_2}
            \right) + C,
            & t > t_s
        \end{cases}

    It naturally produces rapid rise and slow decay behavior.

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
           * - ``t_s``
             - :math:`t_s`
             - Pulse start time.
           * - ``tau_1``
             - :math:`\tau_1`
             - Controls rise behavior (inverse-rise term).
           * - ``tau_2``
             - :math:`\tau_2`
             - Controls exponential decay.
           * - ``C``
             - :math:`C`
             - Constant background level.

    .. dropdown:: Variables

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``t``
             - :math:`t`
             - Time at which the pulse is evaluated.

    .. dropdown:: Returns

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``flux``
             - :math:`F(t)`
             - Flux evaluated at time :math:`t`.


    Notes
    -----
    The peak time is

    .. math::

        t_{\rm peak} = t_s + \sqrt{\tau_1 \tau_2}.

    Examples
    --------

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        from trilobite.models.generic.light_curve import NorrisPulse

        rng = np.random.default_rng(42)

        model = NorrisPulse()

        t = np.linspace(0, 10, 400)
        flux = model({"t": t}, {}).flux

        noise = 0.05 * np.max(flux) * rng.normal(size=t.size)
        synthetic = flux + noise

        plt.plot(t, flux, lw=2, label="Model")
        plt.scatter(t, synthetic, s=8, alpha=0.5, label="Synthetic Data")

        plt.xlabel("Time")
        plt.ylabel("Flux")
        plt.title("Norris Pulse Example")
        plt.legend()
        plt.show()

    See Also
    --------
    :class:`FRED`
    :class:`LogNormalPulse`

    References
    ----------
    .. footbibliography::
    """

    VARIABLES = (ModelVariable("t", base_units=None),)

    PARAMETERS = (
        ModelParameter("A", 1.0, base_units=None),
        ModelParameter("t_s", 0.0, base_units=None),
        ModelParameter("tau_1", 1.0, base_units=None, bounds=(1e-6, None)),
        ModelParameter("tau_2", 1.0, base_units=None, bounds=(1e-6, None)),
        ModelParameter("C", 0.0, base_units=None),
    )

    OUTPUTS = _PulseOutputs
    UNITS = OUTPUTS(flux=u.dimensionless_unscaled)

    DESCRIPTION = "Classic Norris GRB pulse."
    REFERENCE = "Norris et al. (1996, 2005)."

    def __init__(self):
        """Initialize the model with given parameters and variables."""
        self._register_init()

    def _forward_model(self, variables, parameters):
        t = np.asarray(variables["t"], dtype=float)
        A, ts, tau1, tau2, C = (
            parameters["A"],
            parameters["t_s"],
            parameters["tau_1"],
            parameters["tau_2"],
            parameters["C"],
        )

        y = np.full_like(t, C)
        mask = t > ts
        dt = t[mask] - ts

        y[mask] += A * np.exp(-tau1 / dt - dt / tau2)
        return self.OUTPUTS(flux=y)


class WeibullPulse(Model):
    r"""
    Weibull pulse profile with background.

    This model uses the Weibull distribution to generate
    asymmetric pulse shapes.

    .. math::

        F(t) =
        \begin{cases}
            C, & t \le t_0 \\
            A \frac{k}{\lambda}
            \left(\frac{t - t_0}{\lambda}\right)^{k-1}
            \exp\left[-\left(\frac{t - t_0}{\lambda}\right)^k\right]
            + C,
            & t > t_0
        \end{cases}

    - :math:`k` controls skewness.
    - :math:`\lambda` sets temporal scale.

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
           * - ``t_0``
             - :math:`t_0`
             - Pulse onset time.
           * - ``k``
             - :math:`k`
             - Shape parameter controlling skewness.
           * - ``lambda_``
             - :math:`\lambda`
             - Scale parameter controlling temporal width.
           * - ``C``
             - :math:`C`
             - Constant background level.

    .. dropdown:: Variables

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``t``
             - :math:`t`
             - Time at which the model is evaluated.

    .. dropdown:: Returns

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``flux``
             - :math:`F(t)`
             - Flux evaluated at time :math:`t`.


    Notes
    -----
    For :math:`k = 1`, this reduces to an exponential distribution.

    Examples
    --------

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        from trilobite.models.generic.light_curve import WeibullPulse

        rng = np.random.default_rng(42)

        model = WeibullPulse()

        t = np.linspace(0, 10, 400)
        flux = model({"t": t}, {}).flux

        noise = 0.05 * np.max(flux) * rng.normal(size=t.size)
        synthetic = flux + noise

        plt.plot(t, flux, lw=2, label="Model")
        plt.scatter(t, synthetic, s=8, alpha=0.5, label="Synthetic Data")

        plt.xlabel("Time")
        plt.ylabel("Flux")
        plt.title("Weibull Pulse Example")
        plt.legend()
        plt.show()

    See Also
    --------
    :class:`NorrisPulse`
    :class:`LogNormalPulse`

    """

    VARIABLES = (ModelVariable("t", base_units=None),)

    PARAMETERS = (
        ModelParameter("A", 1.0, base_units=None),
        ModelParameter("t_0", 0.0, base_units=None),
        ModelParameter("k", 2.0, base_units=None, bounds=(1e-6, None)),
        ModelParameter("lambda_", 1.0, base_units=None, bounds=(1e-6, None)),
        ModelParameter("C", 0.0, base_units=None),
    )

    OUTPUTS = _PulseOutputs
    UNITS = OUTPUTS(flux=u.dimensionless_unscaled)

    DESCRIPTION = "Weibull pulse profile."
    REFERENCE = "Generalized statistical pulse form."

    def __init__(self):
        """Initialize the model with given parameters and variables."""
        self._register_init()

    def _forward_model(self, variables, parameters):
        t = np.asarray(variables["t"], dtype=float)
        A, t0, k, lam, C = (
            parameters["A"],
            parameters["t_0"],
            parameters["k"],
            parameters["lambda_"],
            parameters["C"],
        )

        y = np.full_like(t, C)
        mask = t > t0
        dt = t[mask] - t0

        y[mask] += A * (k / lam) * (dt / lam) ** (k - 1) * np.exp(-((dt / lam) ** k))
        return self.OUTPUTS(flux=y)


class LogisticPulse(Model):
    r"""
    Logistic sigmoid transition model with background.

    This model describes a smooth monotonic transition
    from background to plateau.

    .. math::

        F(t) =
        \frac{A}{1 + e^{-(t - t_0)/\tau}} + C

    Unlike other models in this module, this form does not
    include a decay phase and therefore represents a
    turn-on or plateau transition rather than a true pulse.

    .. dropdown:: Parameters

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``A``
             - :math:`A`
             - Transition amplitude above background.
           * - ``t_0``
             - :math:`t_0`
             - Midpoint (inflection point) of the sigmoid transition.
           * - ``tau``
             - :math:`\tau`
             - Characteristic transition timescale.
           * - ``C``
             - :math:`C`
             - Constant baseline level.

    .. dropdown:: Variables

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``t``
             - :math:`t`
             - Time at which the model is evaluated.

    .. dropdown:: Returns

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``flux``
             - :math:`F(t)`
             - Flux evaluated at time :math:`t`.


    Notes
    -----
    This model is useful for modeling:

    - Lightcurve turn-on phases,
    - Activation transitions,
    - Gradual state changes.

    It is not suitable for isolated pulse modeling.

    Examples
    --------

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        from trilobite.models.generic.light_curve import LogisticPulse

        rng = np.random.default_rng(42)

        model = LogisticPulse()

        t = np.linspace(-5, 10, 400)
        flux = model({"t": t}, {}).flux

        noise = 0.05 * np.max(flux) * rng.normal(size=t.size)
        synthetic = flux + noise

        plt.plot(t, flux, lw=2, label="Model")
        plt.scatter(t, synthetic, s=8, alpha=0.5, label="Synthetic Data")

        plt.xlabel("Time")
        plt.ylabel("Flux")
        plt.title("Logistic Transition Example")
        plt.legend()
        plt.show()

    See Also
    --------
    :class:`GaussianPulse`

    """

    VARIABLES = (ModelVariable("t", base_units=None),)

    PARAMETERS = (
        ModelParameter("A", 1.0, base_units=None),
        ModelParameter("t_0", 0.0, base_units=None),
        ModelParameter("tau", 1.0, base_units=None, bounds=(1e-6, None)),
        ModelParameter("C", 0.0, base_units=None),
    )

    OUTPUTS = _PulseOutputs
    UNITS = OUTPUTS(flux=u.dimensionless_unscaled)

    DESCRIPTION = "Logistic rise pulse."
    REFERENCE = "Sigmoid transient model."

    def __init__(self):
        """Initialize the model with given parameters and variables."""
        self._register_init()

    def _forward_model(self, variables, parameters):
        t = np.asarray(variables["t"], dtype=float)
        A, t0, tau, C = (
            parameters["A"],
            parameters["t_0"],
            parameters["tau"],
            parameters["C"],
        )

        y = A / (1 + np.exp(-(t - t0) / tau)) + C
        return self.OUTPUTS(flux=y)
