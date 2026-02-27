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

from triceratops.models.core import Model, ModelParameter, ModelVariable

__all__ = [
    "PL_Evolving_SSA_SED_Model",
]

# ============================================================
# Output Definition
# ============================================================
_PLEvolvingOutputs = namedtuple("PLEvolvingSSAOutputs", ["flux_density"])


# ============================================================
# Model Definition
# ============================================================
class PL_Evolving_SSA_SED_Model(Model):
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
        \left(\frac{t}{t_0}\right)^{-\beta}

    and the break flux:

    .. math::

        F_{\rm brk}(t) =
        F_{\rm brk,0}
        \left(\frac{t}{t_0}\right)^{-\gamma}.

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
        """Initialize the model with given parameters and variables."""
        self._register_init()

    def _forward_model(self, variables, parameters):
        nu = np.asarray(variables["frequency"], dtype=float)
        t = np.asarray(variables["time"], dtype=float)

        nu, t = np.broadcast_arrays(nu, t)

        log_nu = np.log(nu)
        log_t = np.log(t)

        F0 = parameters["F_brk_0"]
        nu0 = parameters["nu_brk_0"]
        a1 = parameters["alpha_1"]
        a2 = parameters["alpha_2"]
        beta = parameters["beta"]
        gamma = parameters["gamma"]
        s = parameters["s"]
        t0 = parameters["t_0"]

        log_F0 = np.log(F0)
        log_nu0 = np.log(nu0)
        log_t0 = np.log(t0)

        x = log_t - log_t0

        log_F_brk = log_F0 - gamma * x
        log_nu_brk = log_nu0 - beta * x

        log_xi = log_nu - log_nu_brk

        log_F = log_F_brk - s * np.logaddexp((-a1 / s) * log_xi, (-a2 / s) * log_xi)

        return self.OUTPUTS(flux_density=np.exp(log_F))
