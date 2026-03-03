r"""
Phenomenological models of evolving spectral energy distributions (SEDs).

This module provides flexible analytic models describing time-evolving
spectral energy distributions without imposing detailed dynamical
or radiative transfer physics.

The models here assume that:

- Each epoch's SED is described by a smooth broken power law.
- The break frequency evolves as a power law in time.
- The break flux density evolves as a power law in time.

These models are intended for:

- Multi-epoch radio SED fitting
- Self-similar shock evolution approximations
- GRB afterglow phenomenology
- Radio supernova lightcurve modeling
- Exploratory transient spectral analysis

They do **not** enforce microphysical closure relations,
shock dynamics, energy conservation, or synchrotron
self-consistency conditions.

For physically motivated models with dynamical coupling,
see the shock and synchrotron modules.

Notes
-----
Because these models operate in log-space internally,
they are numerically stable across many decades in
frequency and time.

Care should be taken when interpreting fitted parameters
physically, as degeneracies between spectral slopes and
temporal evolution indices may arise.
"""

from collections import namedtuple

import numpy as np
from astropy import units as u

from ...core import ModelParameter, ModelVariable
from .base import EvolvingSEDModel

__all__ = [
    "PL_Evolving_SBPL_Model",
    "BPL_Evolving_SBPL_Model",
    "TripleBPL_Evolving_SBPL_Model",
]


# ============================================================
# Curve Functions
# ============================================================
def log_smooth_broken_power_law(
    log_x: np.ndarray,
    log_xb: float,
    a1: float,
    a2: float,
    s: float,
) -> np.ndarray:
    r"""
    Logarithm of a normalized smooth broken power law.

    This function evaluates

    .. math::

        \phi(x)
        =
        2^{-s}
        \left[
            \left(\frac{x}{x_b}\right)^{a_1/s}
            +
            \left(\frac{x}{x_b}\right)^{a_2/s}
        \right]^s,

    computed entirely in log-space for numerical stability.

    The normalization ensures

    .. math::

        \phi(x_b) = 1.

    Asymptotically,

    .. math::

        \phi(x) \propto x^{a_1}, \quad x \ll x_b

    .. math::

        \phi(x) \propto x^{a_2}, \quad x \gg x_b.

    Parameters
    ----------
    log_x : ndarray
        Natural logarithm of the independent variable :math:`x`.
    log_xb : float
        Natural logarithm of the break location :math:`x_b`.
    a1 : float
        Power-law slope below the break.
    a2 : float
        Power-law slope above the break.
    s : float
        Smoothness parameter.

        Larger values of ``s`` produce a sharper transition.
        In the limit :math:`s \to \infty`, the function approaches
        a sharp broken power law.

    Returns
    -------
    ndarray
        Natural logarithm of :math:`\phi(x)`.

    Notes
    -----
    The implementation uses :func:`numpy.logaddexp` to maintain
    numerical stability over a wide dynamic range.
    """
    dx = log_x - log_xb

    return s * np.log(2.0) - s * np.logaddexp(
        (-a1 / s) * dx,
        (-a2 / s) * dx,
    )


def log_broken_power_law(
    log_x: np.ndarray,
    log_xb: float,
    slope_lo: float,
    slope_hi: float,
) -> np.ndarray:
    r"""
    Logarithm of a sharp broken power law.

    The function is defined piecewise as

    .. math::

        \phi(x)
        =
        \begin{cases}
        \left(\frac{x}{x_b}\right)^{slope_{lo}},
        & x < x_b \\
        \left(\frac{x}{x_b}\right)^{slope_{hi}},
        & x \ge x_b
        \end{cases}

    and is normalized such that

    .. math::

        \phi(x_b) = 1.

    Parameters
    ----------
    log_x : ndarray
        Natural logarithm of :math:`x`.
    log_xb : float
        Natural logarithm of the break location.
    slope_lo : float
        Slope below the break.
    slope_hi : float
        Slope above the break.

    Returns
    -------
    ndarray
        Natural logarithm of :math:`\phi(x)`.

    Notes
    -----
    The function is continuous at :math:`x = x_b`.
    """
    log_x = np.asarray(log_x)
    dx = log_x - log_xb

    result = np.empty_like(log_x)

    mask_lo = log_x < log_xb
    mask_hi = ~mask_lo

    result[mask_lo] = slope_lo * dx[mask_lo]
    result[mask_hi] = slope_hi * dx[mask_hi]

    return result


def log_triple_power_law(
    log_x: np.ndarray,
    log_xb1: float,
    log_xb2: float,
    slope_1: float,
    slope_2: float,
    slope_3: float,
) -> np.ndarray:
    r"""
    Logarithm of a sharp triple power law.

    The function is defined piecewise as

    .. math::

        \phi(x)
        =
        \begin{cases}
        \left(\frac{x}{x_{b1}}\right)^{slope_1},
        & x < x_{b1} \\
        \left(\frac{x}{x_{b1}}\right)^{slope_2},
        & x_{b1} \le x < x_{b2} \\
        \left(\frac{x_{b2}}{x_{b1}}\right)^{slope_2}
        \left(\frac{x}{x_{b2}}\right)^{slope_3},
        & x \ge x_{b2}
        \end{cases}

    The third branch includes a continuity factor so that the function
    is continuous at :math:`x = x_{b2}`.

    Parameters
    ----------
    log_x : ndarray
        Natural logarithm of :math:`x`.
    log_xb1 : float
        Natural logarithm of the first break.
    log_xb2 : float
        Natural logarithm of the second break.
        Must satisfy ``log_xb2 > log_xb1``.
    slope_1 : float
        Low-x slope.
    slope_2 : float
        Intermediate slope.
    slope_3 : float
        High-x slope.

    Returns
    -------
    ndarray
        Natural logarithm of :math:`\phi(x)`.

    Notes
    -----
    The function is continuous at both break points.
    """
    log_x = np.asarray(log_x)
    result = np.empty_like(log_x)

    mask_1 = log_x < log_xb1
    mask_2 = (log_x >= log_xb1) & (log_x < log_xb2)
    mask_3 = log_x >= log_xb2

    result[mask_1] = slope_1 * (log_x[mask_1] - log_xb1)
    result[mask_2] = slope_2 * (log_x[mask_2] - log_xb1)

    continuity = slope_2 * (log_xb2 - log_xb1)
    result[mask_3] = continuity + slope_3 * (log_x[mask_3] - log_xb2)

    return result


# ============================================================
# Output Definition
# ============================================================
_PLEvolvingOutputs = namedtuple("PLEvolvingSSAOutputs", ["flux_density"])


# ============================================================
# Model Definition
# ============================================================
class PL_Evolving_SBPL_Model(EvolvingSEDModel):
    r"""
    A generic model of synchrotron SED evolution assuming a time-evolving smoothed broken power law.

    This model provides a useful parameterization for first-look analysis of evolving SEDs in radio
    supernovae, GRB afterglows, and other synchrotron-dominated transients. The model assumes that the
    SED at each epoch can be described by a smoothed broken power law, with the break frequency and break
    flux density evolving as power laws in time. For an example of the model's use, see
    :footcite:t:`2019ApJ...872...18M`.

    In this model, we assume that the temporal evolution of the SED can be described by the break frequency:

    .. math::

        \nu_{\rm brk}(t) =
        \nu_{\rm brk,0}
        \left(\frac{t}{t_0}\right)^{\beta}

    and the break flux:

    .. math::

        F_{\rm brk}(t) =
        F_{\rm brk,0}
        \left(\frac{t}{t_0}\right)^{\gamma}.

    At each epoch, the SED is described by a smoothed broken power law:

    .. math::

        F_\nu(\nu,t) =
        F_{\rm brk}(t)
        \left[
            \left(\frac{\nu}{\nu_{\rm brk}(t)}\right)^{\alpha_1/s}
            +
            \left(\frac{\nu}{\nu_{\rm brk}(t)}\right)^{\alpha_2/s}
        \right]^s

    where:

    - :math:`\alpha_1` is the low-frequency spectral index,
    - :math:`\alpha_2` is the high-frequency spectral index,
    - :math:`s > 0` controls the smoothness of the transition.

    This model is purely phenomenological.

    It is suitable for:

    - Self-similar radio shock evolution
    - Multi-epoch GRB afterglow SED fitting
    - Radio supernova monitoring
    - Generic evolving SSA spectra

    .. dropdown:: Parameters

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``F_brk_0``
             - :math:`F_{\rm brk,0}`
             - Break flux density at reference time :math:`t_0`.
           * - ``nu_brk_0``
             - :math:`\nu_{\rm brk,0}`
             - Break frequency at reference time :math:`t_0`.
           * - ``alpha_1``
             - :math:`\alpha_1`
             - Low-frequency spectral index.
           * - ``alpha_2``
             - :math:`\alpha_2`
             - High-frequency spectral index.
           * - ``beta``
             - :math:`\beta`
             - Temporal decay index of break frequency.
           * - ``gamma``
             - :math:`\gamma`
             - Temporal decay index of break flux.
           * - ``s``
             - :math:`s`
             - Smoothness parameter.
           * - ``t_0``
             - :math:`t_0`
             - Reference time for evolution.

    .. dropdown:: Variables

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``frequency``
             - :math:`\nu`
             - Observing frequency.
           * - ``time``
             - :math:`t`
             - Time since explosion.

    .. dropdown:: Returns

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``flux_density``
             - :math:`F_\nu(\nu,t)`
             - Flux density at frequency :math:`\nu` and time :math:`t`.

    Notes
    -----
    - For :math:`\alpha_1 > 0` and :math:`\alpha_2 < 0`, the SED is concave down.
    - Smaller :math:`s` produces sharper breaks.
    - Larger :math:`s` produces broader transitions.
    - The model reduces to a single power law when
      :math:`\alpha_1 = \alpha_2`.

    Examples
    --------

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        from astropy import units as u

        from triceratops.models.SEDs.evolving_sed import PL_Evolving_SSA_SED_Model

        # --------------------------------------------------
        # Reproducibility
        # --------------------------------------------------

        rng = np.random.default_rng(42)

        # --------------------------------------------------
        # Instantiate model
        # --------------------------------------------------

        model = PL_Evolving_SSA_SED_Model()

        # --------------------------------------------------
        # Frequency grid
        # --------------------------------------------------

        nu = np.logspace(8, 11, 200) * u.Hz

        # Choose several epochs
        times = np.geomspace(1, 1000, 5) * u.day

        # --------------------------------------------------
        # Model parameters
        # --------------------------------------------------

        parameters = {
            "alpha_1": 5 / 2,  # self-absorbed slope
            "alpha_2": -1.0,  # optically thin slope
            "beta": 1.0,  # nu_brk evolution
            "gamma": 0.0,  # F_brk evolution
            "nu_brk_0": 1e10 * u.Hz,
            "F_brk_0": 1.0 * u.Jy,
            "t_0": 10 * u.day,
            "s": 0.3,
        }

        # --------------------------------------------------
        # Plot
        # --------------------------------------------------

        plt.figure(figsize=(8, 6))

        for t in times:
            flux = model(
                {"frequency": nu, "time": t},
                parameters
            ).flux_density

            # Add 10% Gaussian noise
            noise = 0.1 * flux * rng.normal(size=flux.size)
            synthetic = flux + noise

            plt.loglog(nu, flux, lw=2, label=f"t = {t.value:.0f} d")
            plt.scatter(nu, synthetic, s=8, alpha=0.4)

        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Flux Density [Jy]")
        plt.title("Time-Evolving Smoothed Broken Power-Law SED")
        plt.legend()
        plt.tight_layout()
        plt.show()

    References
    ----------
    .. footbibliography::
    """

    VARIABLES = (
        ModelVariable("frequency", base_units=u.Hz, description="Observing frequency."),
        ModelVariable("time", base_units=u.s, description="Time since explosion."),
    )

    PARAMETERS = (
        ModelParameter("F_brk_0", 1.0, base_units=u.Jy, bounds=(0.0, None)),
        ModelParameter("nu_brk_0", 1e9, base_units=u.Hz, bounds=(1e-3, None)),
        ModelParameter("alpha_1", -1, base_units=None),
        ModelParameter("alpha_2", 5 / 2, base_units=None),
        ModelParameter("beta", 1.0, base_units=None),
        ModelParameter("gamma", 0.0, base_units=None),
        ModelParameter("s", 0.2, base_units=None, bounds=(1e-4, None)),
        ModelParameter("t_0", 1.0, base_units=u.s, bounds=(1e-6, None)),
    )

    OUTPUTS = _PLEvolvingOutputs
    UNITS = OUTPUTS(flux_density=u.Jy)

    DESCRIPTION = "Time-evolving smoothed broken power-law SSA SED."
    REFERENCE = "Generic self-similar synchrotron evolution model."

    def __init__(self):
        self._register_init()

    # ------------------------------------------------------------
    # Log normalization
    # ------------------------------------------------------------

    def _compute_log_norm(self, log_t, parameters):
        log_F0 = np.log(parameters["F_brk_0"])
        log_t0 = np.log(parameters["t_0"])

        log_tau = log_t - log_t0
        return log_F0 + parameters["gamma"] * log_tau

    # ------------------------------------------------------------
    # Log breaks
    # ------------------------------------------------------------

    def _compute_log_breaks(self, log_t, parameters):
        log_nu0 = np.log(parameters["nu_brk_0"])
        log_t0 = np.log(parameters["t_0"])

        log_tau = log_t - log_t0

        log_nu_brk = log_nu0 + parameters["beta"] * log_tau

        return {"nu_brk": log_nu_brk}

    # ------------------------------------------------------------
    # Log shape
    # ------------------------------------------------------------

    def _compute_log_shape(self, log_nu, log_t, log_breaks, parameters):
        log_nu_brk = log_breaks["nu_brk"]

        return log_smooth_broken_power_law(
            log_nu,
            log_nu_brk,
            parameters["alpha_1"],
            parameters["alpha_2"],
            parameters["s"],
        )


class BPL_Evolving_SBPL_Model(EvolvingSEDModel):
    r"""
    A generic model of synchrotron SED evolution assuming a time-evolving broken power law in time.

    This model extends :class:`PL_Evolving_SSA_SED_Model` by allowing the
    break frequency and break flux density to evolve as broken power laws
    in time with a single temporal break at :math:`t_0`.

    In this model, the temporal evolution of the break frequency is

    .. math::

        \nu_{\rm brk}(t)
        =
        \nu_{\rm brk,0}
        \times
        \begin{cases}
        \left(\frac{t}{t_0}\right)^{\beta_1}, & t < t_0 \\
        \left(\frac{t}{t_0}\right)^{\beta_2}, & t \ge t_0
        \end{cases}

    and the break flux:

    .. math::

        F_{\rm brk}(t)
        =
        F_{\rm brk,0}
        \times
        \begin{cases}
        \left(\frac{t}{t_0}\right)^{\gamma_1}, & t < t_0 \\
        \left(\frac{t}{t_0}\right)^{\gamma_2}, & t \ge t_0
        \end{cases}

    At each epoch, the SED is described by a smoothed broken power law:

    .. math::

        F_\nu(\nu,t) =
        F_{\rm brk}(t)
        \left[
            \left(\frac{\nu}{\nu_{\rm brk}(t)}\right)^{\alpha_{1i}/s_i}
            +
            \left(\frac{\nu}{\nu_{\rm brk}(t)}\right)^{\alpha_{2i}/s_i}
        \right]^{s_i}

    where:

    - :math:`\alpha_{1i}` is the low-frequency spectral index in temporal segment :math:`i`,
    - :math:`\alpha_{2i}` is the high-frequency spectral index in temporal segment :math:`i`,
    - :math:`s_i > 0` controls the smoothness of the transition.

    This model is purely phenomenological.

    It is suitable for:

    - Radio light curves with a single dynamical transition
    - GRB afterglows with energy injection
    - Shock interaction models with density structure changes
    - Multi-stage transient evolution

    .. dropdown:: Parameters

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``F_brk_0``
             - :math:`F_{\rm brk,0}`
             - Break flux density normalization.
           * - ``nu_brk_0``
             - :math:`\nu_{\rm brk,0}`
             - Break frequency normalization.
           * - ``alpha_11``, ``alpha_21``
             - :math:`\alpha_{11}`, :math:`\alpha_{21}`
             - Spectral slopes before the temporal break.
           * - ``alpha_12``, ``alpha_22``
             - :math:`\alpha_{12}`, :math:`\alpha_{22}`
             - Spectral slopes after the temporal break.
           * - ``beta1``, ``beta2``
             - :math:`\beta_1`, :math:`\beta_2`
             - Temporal slopes of break frequency.
           * - ``gamma1``, ``gamma2``
             - :math:`\gamma_1`, :math:`\gamma_2`
             - Temporal slopes of break flux.
           * - ``s1``, ``s2``
             - :math:`s_1`, :math:`s_2`
             - Spectral smoothness parameters.
           * - ``t_0``
             - :math:`t_0`
             - Temporal break time.

    .. dropdown:: Variables

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``frequency``
             - :math:`\nu`
             - Observing frequency.
           * - ``time``
             - :math:`t`
             - Time since explosion.

    .. dropdown:: Returns

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``flux_density``
             - :math:`F_\nu(\nu,t)`
             - Flux density at frequency :math:`\nu` and time :math:`t`.

    Notes
    -----
    - The model is continuous in time by construction.
    - Spectral curvature in each segment is controlled by ``s1`` and ``s2``.
    - This model does **not** enforce microphysical closure relations.
    """

    VARIABLES = (
        ModelVariable("frequency", base_units=u.Hz, description="Observing frequency."),
        ModelVariable("time", base_units=u.s, description="Time since explosion."),
    )

    PARAMETERS = (
        ModelParameter("F_brk_0", 1.0, base_units=u.Jy, bounds=(0.0, None)),
        ModelParameter("nu_brk_0", 1e9, base_units=u.Hz, bounds=(1e-3, None)),
        ModelParameter("alpha_11", -1, base_units=None),
        ModelParameter("alpha_21", 5 / 2, base_units=None),
        ModelParameter("alpha_12", -1, base_units=None),
        ModelParameter("alpha_22", 5 / 2, base_units=None),
        ModelParameter("beta1", 1.0, base_units=None),
        ModelParameter("gamma1", 0.0, base_units=None),
        ModelParameter("beta2", 1.0, base_units=None),
        ModelParameter("gamma2", 0.0, base_units=None),
        ModelParameter("s1", 0.2, base_units=None, bounds=(1e-4, None)),
        ModelParameter("s2", 0.2, base_units=None, bounds=(1e-4, None)),
        ModelParameter("t_0", 1.0, base_units=u.s, bounds=(1e-6, None)),
    )

    OUTPUTS = _PLEvolvingOutputs
    UNITS = OUTPUTS(flux_density=u.Jy)

    DESCRIPTION = "Time-evolving broken power-law SSA SED."
    REFERENCE = "Generic multi-stage synchrotron evolution model."

    def __init__(self):
        self._register_init()

    def _compute_log_norm(self, log_t, parameters):
        log_F0 = np.log(parameters["F_brk_0"])
        log_t0 = np.log(parameters["t_0"])

        return log_F0 + log_broken_power_law(
            log_t,
            log_t0,
            parameters["gamma1"],
            parameters["gamma2"],
        )

    def _compute_log_breaks(self, log_t, parameters):
        log_nu0 = np.log(parameters["nu_brk_0"])
        log_t0 = np.log(parameters["t_0"])

        log_nu_brk = log_nu0 + log_broken_power_law(
            log_t,
            log_t0,
            parameters["beta1"],
            parameters["beta2"],
        )

        return {"nu_brk": log_nu_brk}

    def _compute_log_shape(self, log_nu, log_t, log_breaks, parameters):
        log_nu_brk = log_breaks["nu_brk"]
        log_t0 = np.log(parameters["t_0"])

        regime1 = log_t < log_t0
        regime2 = ~regime1

        log_phi = np.empty_like(log_nu)

        if np.any(regime1):
            log_phi[regime1] = log_smooth_broken_power_law(
                log_nu[regime1],
                log_nu_brk[regime1],
                parameters["alpha_11"],
                parameters["alpha_21"],
                parameters["s1"],
            )

        if np.any(regime2):
            log_phi[regime2] = log_smooth_broken_power_law(
                log_nu[regime2],
                log_nu_brk[regime2],
                parameters["alpha_12"],
                parameters["alpha_22"],
                parameters["s2"],
            )

        return log_phi


class TripleBPL_Evolving_SBPL_Model(EvolvingSEDModel):
    r"""
    A generic model of synchrotron SED evolution assuming a time-evolving triple broken power law in time.

    This model extends :class:`PL_Evolving_SSA_SED_Model` by allowing the
    break frequency and break flux to evolve as *triple* broken power laws
    in time with two temporal break points.

    In this model, the temporal evolution of the break frequency is

    .. math::

        \nu_{\rm brk}(t)
        =
        \nu_{\rm brk,0}
        \, \Phi_\nu(t),

    where :math:`\Phi_\nu(t)` is a sharp triple power law with temporal
    breaks at :math:`t_0` and :math:`t_1`.

    Similarly, the break flux evolves as

    .. math::

        F_{\rm brk}(t)
        =
        F_{\rm brk,0}
        \, \Phi_F(t),

    where :math:`\Phi_F(t)` is also a sharp triple power law.

    At each epoch, the SED is described by a smoothed broken power law:

    .. math::

        F_\nu(\nu,t) =
        F_{\rm brk}(t)
        \left[
            \left(\frac{\nu}{\nu_{\rm brk}(t)}\right)^{\alpha_{1i}/s_i}
            +
            \left(\frac{\nu}{\nu_{\rm brk}(t)}\right)^{\alpha_{2i}/s_i}
        \right]^{s_i}

    where the spectral slopes and smoothness parameter may differ in
    each temporal segment.

    This model is purely phenomenological.

    It is suitable for:

    - Radio light curves with multiple dynamical transitions
    - GRB afterglows with energy injection phases
    - Shock interaction models with multiple density regimes
    - Complex multi-stage transient evolution

    .. dropdown:: Parameters

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``F_brk_0``
             - :math:`F_{\rm brk,0}`
             - Break flux density normalization.
           * - ``nu_brk_0``
             - :math:`\nu_{\rm brk,0}`
             - Break frequency normalization.
           * - ``alpha_11``, ``alpha_21``
             - :math:`\alpha_{11}`, :math:`\alpha_{21}`
             - Spectral slopes in first temporal segment.
           * - ``alpha_12``, ``alpha_22``
             - :math:`\alpha_{12}`, :math:`\alpha_{22}`
             - Spectral slopes in second temporal segment.
           * - ``alpha_13``, ``alpha_23``
             - :math:`\alpha_{13}`, :math:`\alpha_{23}`
             - Spectral slopes in third temporal segment.
           * - ``beta1``, ``beta2``, ``beta3``
             - :math:`\beta_1`, :math:`\beta_2`, :math:`\beta_3`
             - Temporal slopes of break frequency.
           * - ``gamma1``, ``gamma2``, ``gamma3``
             - :math:`\gamma_1`, :math:`\gamma_2`, :math:`\gamma_3`
             - Temporal slopes of break flux.
           * - ``s1``, ``s2``, ``s3``
             - :math:`s_1`, :math:`s_2`, :math:`s_3`
             - Spectral smoothness parameters in each segment.
           * - ``t_0``, ``t_1``
             - :math:`t_0`, :math:`t_1`
             - Temporal break times.

    .. dropdown:: Variables

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``frequency``
             - :math:`\nu`
             - Observing frequency.
           * - ``time``
             - :math:`t`
             - Time since explosion.

    .. dropdown:: Returns

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``flux_density``
             - :math:`F_\nu(\nu,t)`
             - Flux density at frequency :math:`\nu` and time :math:`t`.

    Notes
    -----
    - Each temporal segment may have independent spectral slopes.
    - The model remains continuous in time by construction.
    - Spectral curvature is controlled by ``s1``, ``s2``, and ``s3``.
    - This model does **not** enforce physical closure relations.
    """

    VARIABLES = (
        ModelVariable("frequency", base_units=u.Hz, description="Observing frequency."),
        ModelVariable("time", base_units=u.s, description="Time since explosion."),
    )

    PARAMETERS = (
        ModelParameter("F_brk_0", 1.0, base_units=u.Jy, bounds=(0.0, None)),
        ModelParameter("nu_brk_0", 1e9, base_units=u.Hz, bounds=(1e-3, None)),
        ModelParameter("alpha_11", -1, base_units=None),
        ModelParameter("alpha_21", 5 / 2, base_units=None),
        ModelParameter("alpha_12", -1, base_units=None),
        ModelParameter("alpha_22", 5 / 2, base_units=None),
        ModelParameter("alpha_13", -1, base_units=None),
        ModelParameter("alpha_23", 5 / 2, base_units=None),
        ModelParameter("beta1", 1.0, base_units=None),
        ModelParameter("gamma1", 0.0, base_units=None),
        ModelParameter("beta2", 1.0, base_units=None),
        ModelParameter("gamma2", 0.0, base_units=None),
        ModelParameter("beta3", 1.0, base_units=None),
        ModelParameter("gamma3", 0.0, base_units=None),
        ModelParameter("s1", 0.2, base_units=None, bounds=(1e-4, None)),
        ModelParameter("s2", 0.2, base_units=None, bounds=(1e-4, None)),
        ModelParameter("s3", 0.2, base_units=None, bounds=(1e-4, None)),
        ModelParameter("t_0", 1.0, base_units=u.s, bounds=(1e-6, None)),
        ModelParameter("t_1", 1.0, base_units=u.s, bounds=(1e-6, None)),
    )

    OUTPUTS = _PLEvolvingOutputs
    UNITS = OUTPUTS(flux_density=u.Jy)

    DESCRIPTION = "Time-evolving triple broken power-law SSA SED."
    REFERENCE = "Generic multi-stage synchrotron evolution model."

    def __init__(self):
        self._register_init()

    def _compute_log_norm(self, log_t, parameters):
        log_F0 = np.log(parameters["F_brk_0"])
        log_t0 = np.log(parameters["t_0"])
        log_t1 = np.log(parameters["t_1"])

        return log_F0 + log_triple_power_law(
            log_t,
            log_t0,
            log_t1,
            parameters["gamma1"],
            parameters["gamma2"],
            parameters["gamma3"],
        )

    def _compute_log_breaks(self, log_t, parameters):
        log_nu0 = np.log(parameters["nu_brk_0"])
        log_t0 = np.log(parameters["t_0"])
        log_t1 = np.log(parameters["t_1"])

        log_nu_brk = log_nu0 + log_triple_power_law(
            log_t,
            log_t0,
            log_t1,
            parameters["beta1"],
            parameters["beta2"],
            parameters["beta3"],
        )
        return {"nu_brk": log_nu_brk}

    def _compute_log_shape(self, log_nu, log_t, log_breaks, parameters):
        log_nu_brk = log_breaks["nu_brk"]
        log_t0 = np.log(parameters["t_0"])
        log_t1 = np.log(parameters["t_1"])

        regime1 = log_t < log_t0
        regime2 = (log_t >= log_t0) & (log_t < log_t1)
        regime3 = log_t >= log_t1

        log_phi = np.empty_like(log_nu)

        if np.any(regime1):
            log_phi[regime1] = log_smooth_broken_power_law(
                log_nu[regime1],
                log_nu_brk[regime1],
                parameters["alpha_11"],
                parameters["alpha_21"],
                parameters["s1"],
            )

        if np.any(regime2):
            log_phi[regime2] = log_smooth_broken_power_law(
                log_nu[regime2],
                log_nu_brk[regime2],
                parameters["alpha_12"],
                parameters["alpha_22"],
                parameters["s2"],
            )

        if np.any(regime3):
            log_phi[regime3] = log_smooth_broken_power_law(
                log_nu[regime3],
                log_nu_brk[regime3],
                parameters["alpha_13"],
                parameters["alpha_23"],
                parameters["s3"],
            )

        return log_phi
