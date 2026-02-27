r"""
Physical synchrotron spectral energy distribution (SED) models.

This module provides physically motivated single-zone synchrotron
SED models built on top of the low-level implementations in
:mod:`radiation.synchrotron.SEDs`.

The classes defined here wrap the optimized synchrotron backends
and expose them through the generic :class:`~models.core.base.Model`
interface, making them suitable for:

- Forward modeling
- Parameter inference
- Synthetic data generation
- Multi-frequency fitting pipelines

All models:

- Operate internally in logarithmic space for numerical stability.
- Use equipartition-based normalization.
- Apply relativistic Doppler boosting when ``gamma_bulk > 1``.
- Include cosmological redshift corrections.
- Support either pitch-angle averaged or fixed pitch-angle emission.

Model Hierarchy
---------------

The models differ only in which spectral breaks are included:

+----------------------------------------------+-----------+-----------+
| Model                                        | Cooling   | SSA       |
+==============================================+===========+===========+
| :class:`SynchrotronSEDModel`                 | ✗         | ✗         |
| :class:`Cooling_SynchrotronSEDModel`         | ✓         | ✗         |
| :class:`SSA_SynchrotronSEDModel`             | ✗         | ✓         |
| :class:`SSA_Cooling_SynchrotronSEDModel`     | ✓         | ✓         |
+----------------------------------------------+-----------+-----------+

Implementation Notes
--------------------

These models enforce microphysical closure relations
through the underlying synchrotron normalization routines.
Break frequencies are computed
self-consistently from:

- Magnetic field strength
- Effective emitting volume
- Electron Lorentz factor bounds
- Energy partition fractions
- Source geometry
- Doppler boosting
- Redshift

Because normalization is performed analytically in log-space,
these models remain numerically stable across many decades
in frequency.

For purely phenomenological evolving SEDs without
microphysical closure, see:

:mod:`~models.generic.SEDs.evolving_sed.PL_Evolving_SSA_SED_Model`

For theoretical background and implementation details, see:

- :ref:`synchrotron_theory`
- :ref:`synch_sed_theory`
"""

from collections import namedtuple

import numpy as np
from astropy import units as u

from triceratops.radiation.synchrotron.SEDs import (
    PowerLaw_Cooling_SSA_SynchrotronSED,
    PowerLaw_Cooling_SynchrotronSED,
    PowerLaw_SSA_SynchrotronSED,
    PowerLaw_SynchrotronSED,
)

from ...radiation.synchrotron._sed_functions import smoothed_BPL
from .._typing import _ModelParametersInputRaw, _ModelVariablesInputRaw
from ..core import Model, ModelParameter, ModelVariable

__all__ = []

# ------------------------------------------------- #
# Define the Output object                          #
# ------------------------------------------------- #
_SynchrotronSEDOutput = namedtuple("SynchrotronSEDOutput", ["flux"])
_SynchrotronSEDOutputUnits = _SynchrotronSEDOutput(flux=None)


class SSA_Cooling_SynchrotronSEDModel(Model):
    r"""
    Physical synchrotron spectral energy distribution including radiative cooling and synchrotron self-absorption (SSA).

    This model provides a physically motivated single-zone synchrotron
    emission model with:

    - A power-law electron distribution
    - Radiative cooling
    - Synchrotron self-absorption
    - Equipartition-based normalization
    - Optional relativistic Doppler boosting
    - Cosmological redshift corrections

    The model wraps
    :class:`~radiation.synchrotron.SEDs.PowerLaw_Cooling_SSA_SynchrotronSED`
    and exposes it through the generic :class:`~models.core.base.Model`
    interface for inference and forward modeling.

    All internal calculations are performed in log-space for numerical
    stability.

    .. important::

        A detailed description of this model's assumptions, parameters, and implementation methodology
        may be found in :ref:`synchrotron_theory` and :ref:`synch_sed_theory`.

    .. dropdown:: Parameters

        .. list-table::
           :widths: 30 30 40
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``p``
             - :math:`p`
             - Electron power-law index.
           * - ``epsilon_E``
             - :math:`\epsilon_e`
             - Fraction of post-shock energy in electrons.
           * - ``epsilon_B``
             - :math:`\epsilon_B`
             - Fraction of post-shock energy in magnetic fields.
           * - ``log_gamma_min``
             - :math:`\log \gamma_{\min}`
             - Log minimum electron Lorentz factor.
           * - ``log_gamma_c``
             - :math:`\log \gamma_c`
             - Log cooling Lorentz factor.
           * - ``log_gamma_max``
             - :math:`\log \gamma_{\max}`
             - Log maximum electron Lorentz factor.
           * - ``gamma_bulk``
             - :math:`\Gamma_{\rm bulk}`
             - Bulk Lorentz factor of emitting region.
           * - ``alpha``
             - :math:`\alpha`
             - Pitch angle (ignored if pitch-averaged).
           * - ``log_B``
             - :math:`\log B`
             - Log magnetic field strength (cgs).
           * - ``log_V_eff``
             - :math:`\log V_{\rm eff}`
             - Log effective emitting volume.
           * - ``log_Omega``
             - :math:`\log \Omega`
             - Log solid angle of emitting region.
           * - ``log_D_L``
             - :math:`\log D_L`
             - Log luminosity distance (cgs).
           * - ``redshift``
             - :math:`z`
             - Cosmological redshift.

    .. dropdown:: Variables

        .. list-table::
           :widths: 30 30 40
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``log_nu``
             - :math:`\log \nu`
             - Log observing frequency (Hz, observer frame).

    .. dropdown:: Returns

        .. list-table::
           :widths: 30 30 40
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``flux``
             - :math:`\log F_\nu`
             - Log flux density (cgs units).

    Notes
    -----
    - All break frequencies are computed self-consistently.
    - Doppler boosting is applied as:

      .. math::

          \nu_{\rm obs} = \frac{\delta}{1+z} \nu'

          F_{\nu,\rm obs} = \delta^3 F'_{\nu'}

    - Assumes a homogeneous single-zone emission region.
    - Equipartition closure is used for normalization.
    - During initialization, one may choose to either use pitch-angle averaged
      evaluation or specify a fixed pitch angle. If pitch-averaged, the model integrates over an
      isotropic distribution of pitch angles to compute the SED. If a fixed pitch
      angle is specified, the model evaluates the SED at that specific angle.

    Examples
    --------

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt

        from triceratops.models.SEDs.synchrotron import (
            SSA_Cooling_SynchrotronSEDModel
        )

        rng = np.random.default_rng(42)

        model = SSA_Cooling_SynchrotronSEDModel(pitch_averaged=True)

        nu = np.logspace(8, 16, 400)

        parameters = {
            "p": 2.5,
            "epsilon_E": 0.1,
            "epsilon_B": 0.01,
            "log_gamma_min": np.log(10.0),
            "log_gamma_c": np.log(1e3),
            "log_gamma_max": np.log(1e4),
            "gamma_bulk": 2.0,
            "alpha": np.pi / 2,
            "log_B": np.log(1.0),
            "log_V_eff": np.log(1e55),
            "log_Omega": np.log(1e-10),
            "log_D_L": np.log(1e27),
            "redshift": 0.01,
        }

        # Evaluate model
        output = model({"log_nu": np.log(nu)}, parameters)
        flux = np.exp(output.flux)

        # Compute break frequencies
        norm = model._sed._opt_from_physics_to_params(
            log_B=parameters["log_B"],
            log_V=parameters["log_V_eff"],
            log_D_L=parameters["log_D_L"],
            log_Omega=parameters["log_Omega"],
            log_gamma_min=parameters["log_gamma_min"],
            log_gamma_c=parameters["log_gamma_c"],
            log_gamma_max=parameters["log_gamma_max"],
            p=parameters["p"],
            epsilon_E=parameters["epsilon_E"],
            epsilon_B=parameters["epsilon_B"],
            alpha=parameters["alpha"],
            gamma_bulk=parameters["gamma_bulk"],
            redshift=parameters["redshift"],
            pitch_average=True,
        )

        # Synthetic data
        synthetic = flux + 0.1 * flux * rng.normal(size=flux.size)

        plt.figure(figsize=(8,6))
        plt.loglog(nu, flux, lw=2, label="Model")
        plt.scatter(nu, synthetic, s=10, alpha=0.4)

        # Mark breaks
        for key, label in [
            ("nu_a", r"$\nu_a$"),
            ("nu_m", r"$\nu_m$"),
            ("nu_c", r"$\nu_c$"),
            ("nu_max", r"$\nu_{\max}$"),
        ]:
            plt.axvline(np.exp(norm[key]), ls="--", alpha=0.6)
            plt.text(np.exp(norm[key]), np.amax(flux)*0.3, label, rotation=90)

        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Flux Density [cgs]")
        plt.title("Synchrotron SED (Cooling + SSA)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    See Also
    --------
    :class:`SSA_SynchrotronSEDModel`
        Synchrotron model including self-absorption but without cooling.

    :class:`Cooling_SynchrotronSEDModel`
        Synchrotron model including cooling but without self-absorption.

    :class:`SynchrotronSEDModel`
        Optically thin synchrotron model without cooling or absorption.

    :mod:`radiation.synchrotron.SEDs`
        Low-level SED implementations and normalization routines.

    :ref:`synchrotron_theory`
        Detailed theoretical description of synchrotron radiation.

    :ref:`synch_sed_theory`
        Detailed documentation of SED construction and normalization.

    """

    # -------------------------------------------------- #
    # Output Definition                                  #
    # -------------------------------------------------- #
    OUTPUTS = _SynchrotronSEDOutput
    UNITS = _SynchrotronSEDOutputUnits

    # -------------------------------------------------- #
    # Parameters and Variables                           #
    # -------------------------------------------------- #
    PARAMETERS = (
        # Microphysical Parameters.
        ModelParameter("p", 2.5, description="Electron power law index", base_units=None, latex="p", bounds=(0, None)),
        ModelParameter(
            "epsilon_E",
            0.1,
            description="Electron energy fraction",
            base_units=None,
            latex=r"\epsilon_e",
            bounds=(0, 1),
        ),
        ModelParameter(
            "epsilon_B",
            0.01,
            description="Magnetic energy fraction",
            base_units=None,
            latex=r"\epsilon_B",
            bounds=(0, 1),
        ),
        ModelParameter(
            "log_gamma_min",
            0.0,
            description="Logarithm of the minimum electron Lorentz factor",
            base_units=None,
            latex=r"\log \gamma_{\rm min}",
            bounds=(None, None),
        ),
        ModelParameter(
            "log_gamma_max",
            8.0,
            description="Logarithm of the maximum electron Lorentz factor",
            base_units=None,
            latex=r"\log \gamma_{\rm max}",
            bounds=(None, None),
        ),
        ModelParameter(
            "gamma_bulk",
            1.0,
            description="Bulk Lorentz factor of the emitting region",
            base_units=None,
            latex=r"\gamma_{\rm bulk}",
            bounds=(1, None),
        ),
        ModelParameter(
            "alpha",
            np.pi / 2,
            description="Pitch angle of the electrons (if not pitch-averaged)",
            base_units=None,
            latex=r"\alpha",
            bounds=(0, None),
        ),
        ModelParameter(
            "log_gamma_c",
            0.0,
            description="Logarithm of the cooling Lorentz factor",
            base_units=None,
            latex=r"\log \gamma_c",
            bounds=(None, None),
        ),
        # Geometric Parameters.
        ModelParameter(
            "log_D_L",
            27.0,
            description="Logarithm of the luminosity distance to the source (cgs).",
            base_units=None,
            latex=r"\log_D_L",
            bounds=(0, None),
        ),
        ModelParameter(
            "log_Omega",
            -5,
            description="Logarithm of the solid angle subtended by the emitting region (steradians).",
            base_units=None,
            latex=r"\log \Omega",
            bounds=(None, None),
        ),
        # Core Parameters.
        ModelParameter(
            "log_B",
            0.0,
            description="Logarithm of the magnetic field strength (cgs).",
            base_units=None,
            latex=r"\log B",
            bounds=(None, None),
        ),
        ModelParameter(
            "log_V_eff",
            57.0,
            description="Logarithm of the effective volume of the emitting region (cgs).",
            base_units=None,
            latex=r"\log V_{\rm eff}",
            bounds=(0, None),
        ),
        ModelParameter(
            "redshift", 0.0, description="Redshift of the source", base_units=None, latex="z", bounds=(0, None)
        ),
    )
    VARIABLES = (
        ModelVariable(
            "log_nu", base_units=None, description="Logarithm of the observing frequency (Hz).", latex=r"\log \nu"
        ),
    )

    # -------------------------------------------------- #
    # Initialization                                     #
    # -------------------------------------------------- #
    def __init__(
        self,
        pitch_averaged: bool = False,
    ):
        r"""
        Initialize the SSA + Cooling synchrotron SED model.

        Parameters
        ----------
        pitch_averaged : bool, optional
            If ``True``, the synchrotron emissivity is computed assuming
            an isotropic pitch-angle distribution, using the pitch-angle
            averaged formulation of the synchrotron kernel.

            If ``False``, a fixed pitch angle ``alpha`` must be supplied
            as a model parameter and the emissivity is computed using
            :math:`\sin\alpha` explicitly.

        Notes
        -----
        - The pitch-averaged formulation removes explicit dependence
          on ``alpha`` and is generally appropriate for turbulent
          post-shock magnetic fields.
        - The fixed-angle formulation may be useful for idealized
          ordered magnetic field configurations.
        - This choice affects the normalization and break frequencies
          through factors of :math:`\sin\alpha`.

        Internally, this constructor instantiates the low-level
        synchrotron SED backend:

        :class:`~radiation.synchrotron.SEDs.PowerLaw_Cooling_SSA_SynchrotronSED`.
        """
        super().__init__()

        self._sed = PowerLaw_Cooling_SSA_SynchrotronSED()
        self._pitch_averaged = pitch_averaged

        self._register_init(pitch_averaged=pitch_averaged)

    # -------------------------------------------------- #
    # Forward Model Definition                           #
    # -------------------------------------------------- #
    def _forward_model(
        self,
        variables: _ModelVariablesInputRaw,
        parameters: _ModelParametersInputRaw,
    ) -> "OUTPUTS":
        # Use the SED object created at initialization to construct the
        # correct normalization data. The resulting norm dictionary provides
        # - F_peak, the log of the peak flux density (in cgs units)
        # - nu_m, the log of the characteristic synchrotron frequency (in Hz)
        # - nu_c, the log of the cooling frequency (in Hz)
        # - nu_max, the log of the maximum synchrotron frequency (in Hz)
        # - regime: an Enum indicating the spectral regime (e.g., "fast_cooling"
        #           or "slow_cooling")
        norm = self._sed._opt_from_physics_to_params(
            log_B=parameters["log_B"],
            log_V=parameters["log_V_eff"],
            log_D_L=parameters["log_D_L"],
            log_Omega=parameters["log_Omega"],
            log_gamma_min=parameters["log_gamma_min"],
            log_gamma_c=parameters["log_gamma_c"],
            log_gamma_max=parameters["log_gamma_max"],
            p=parameters["p"],
            epsilon_E=parameters["epsilon_E"],
            epsilon_B=parameters["epsilon_B"],
            alpha=parameters["alpha"],
            gamma_bulk=parameters["gamma_bulk"],
            redshift=parameters["redshift"],
            pitch_average=self._pitch_averaged,
        )

        # Now we can use the SED to compute the correct flux density at the given frequencies.
        log_F_nu = self._sed._log_opt_sed_from_regime(
            regime=norm["regime"],
            log_nu=variables["log_nu"],
            log_nu_m=norm["nu_m"],
            log_nu_a=norm["nu_a"],
            log_nu_c=norm["nu_c"],
            log_nu_max=norm["nu_max"],
            log_F_norm=norm["F_norm"],
            p=parameters["p"],
            s=-1.0,  # We can set s=1 for the smoothing parameter since we're not modeling a smoothed SED here.
        )

        return self.OUTPUTS(flux=log_F_nu)


class SSA_SynchrotronSEDModel(Model):
    r"""
    Physical synchrotron spectral energy distribution including SSA but without radiative cooling.

    This model provides a single-zone synchrotron emission model with:

    - A power-law electron distribution
    - Synchrotron self-absorption
    - Equipartition-based normalization
    - Optional relativistic Doppler boosting
    - Cosmological redshift corrections

    Radiative cooling is not included in this formulation.

    The model wraps
    :class:`~radiation.synchrotron.SEDs.PowerLaw_SSA_SynchrotronSED`
    and exposes it through the generic :class:`~models.core.base.Model`
    interface.

    All internal calculations are performed in log-space for numerical
    stability.

    .. important::

        A detailed theoretical description may be found in
        :ref:`synchrotron_theory` and :ref:`synch_sed_theory`.

    .. dropdown:: Parameters

        .. list-table::
           :widths: 30 30 40
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``p``
             - :math:`p`
             - Electron power-law index.
           * - ``epsilon_E``
             - :math:`\epsilon_e`
             - Fraction of post-shock energy in electrons.
           * - ``epsilon_B``
             - :math:`\epsilon_B`
             - Fraction of post-shock energy in magnetic fields.
           * - ``log_gamma_min``
             - :math:`\log \gamma_{\min}`
             - Log minimum electron Lorentz factor.
           * - ``log_gamma_max``
             - :math:`\log \gamma_{\max}`
             - Log maximum electron Lorentz factor.
           * - ``gamma_bulk``
             - :math:`\Gamma_{\rm bulk}`
             - Bulk Lorentz factor of emitting region.
           * - ``alpha``
             - :math:`\alpha`
             - Pitch angle (ignored if pitch-averaged).
           * - ``log_B``
             - :math:`\log B`
             - Log magnetic field strength.
           * - ``log_V_eff``
             - :math:`\log V_{\rm eff}`
             - Log effective emitting volume.
           * - ``log_Omega``
             - :math:`\log \Omega`
             - Log solid angle of emitting region.
           * - ``log_D_L``
             - :math:`\log D_L`
             - Log luminosity distance.
           * - ``redshift``
             - :math:`z`
             - Cosmological redshift.

    .. dropdown:: Variables

        ``log_nu`` — Log observing frequency (Hz).

    .. dropdown:: Returns

        ``flux`` — :math:`\log F_\nu` (cgs units).

    Notes
    -----
    - No cooling break is computed.
    - SSA break is solved self-consistently.
    - Doppler corrections follow

      .. math::

          \nu_{\rm obs} = \frac{\delta}{1+z} \nu' ,
          \quad
          F_{\nu,\rm obs} = \delta^3 F'_{\nu'}.

    See Also
    --------
    :class:`SSA_Cooling_SynchrotronSEDModel`
    :class:`Cooling_SynchrotronSEDModel`
    :class:`SynchrotronSEDModel`

    Examples
    --------
    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt

        from triceratops.models.SEDs.synchrotron import (
            SSA_SynchrotronSEDModel
        )

        rng = np.random.default_rng(42)

        model = SSA_SynchrotronSEDModel(pitch_averaged=True)

        nu = np.logspace(8, 16, 400)

        parameters = {
            "p": 2.5,
            "epsilon_E": 0.1,
            "epsilon_B": 0.1,
            "log_gamma_min": np.log(10.0),
            "log_gamma_c": np.log(1e3),
            "log_gamma_max": np.log(1e4),
            "gamma_bulk": 2.0,
            "alpha": np.pi / 2,
            "log_B": np.log(1.0),
            "log_V_eff": np.log(1e55),
            "log_Omega": np.log(1e-15),
            "log_D_L": np.log(1e27),
            "redshift": 0.01,
        }

        # Evaluate model
        output = model({"log_nu": np.log(nu)}, parameters)
        flux = np.exp(output.flux)

        # Compute break frequencies
        norm = model._sed._opt_from_physics_to_params(
            log_B=parameters["log_B"],
            log_V=parameters["log_V_eff"],
            log_D_L=parameters["log_D_L"],
            log_Omega=parameters["log_Omega"],
            log_gamma_min=parameters["log_gamma_min"],
            log_gamma_max=parameters["log_gamma_max"],
            p=parameters["p"],
            epsilon_E=parameters["epsilon_E"],
            epsilon_B=parameters["epsilon_B"],
            alpha=parameters["alpha"],
            gamma_bulk=parameters["gamma_bulk"],
            redshift=parameters["redshift"],
            pitch_average=True,
        )

        # Synthetic data
        synthetic = flux + 0.1 * flux * rng.normal(size=flux.size)

        plt.figure(figsize=(8,6))
        plt.loglog(nu, flux, lw=2, label="Model")
        plt.scatter(nu, synthetic, s=10, alpha=0.4)

        # Mark breaks
        for key, label in [
            ("nu_a", r"$\nu_a$"),
            ("nu_m", r"$\nu_m$"),
            ("nu_max", r"$\nu_{\max}$"),
        ]:
            plt.axvline(np.exp(norm[key]), ls="--", alpha=0.6)
            plt.text(np.exp(norm[key]), np.amax(flux)*0.3, label, rotation=90)

        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Flux Density [cgs]")
        plt.title("Synchrotron SED (Cooling + SSA)")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    """

    # -------------------------------------------------- #
    # Output Definition                                  #
    # -------------------------------------------------- #
    OUTPUTS = _SynchrotronSEDOutput
    UNITS = _SynchrotronSEDOutputUnits

    # -------------------------------------------------- #
    # Parameters and Variables                           #
    # -------------------------------------------------- #
    PARAMETERS = (
        # Microphysical Parameters.
        ModelParameter("p", 2.5, description="Electron power law index", base_units=None, latex="p", bounds=(0, None)),
        ModelParameter(
            "epsilon_E",
            0.1,
            description="Electron energy fraction",
            base_units=None,
            latex=r"\epsilon_e",
            bounds=(0, 1),
        ),
        ModelParameter(
            "epsilon_B",
            0.01,
            description="Magnetic energy fraction",
            base_units=None,
            latex=r"\epsilon_B",
            bounds=(0, 1),
        ),
        ModelParameter(
            "log_gamma_min",
            0.0,
            description="Logarithm of the minimum electron Lorentz factor",
            base_units=None,
            latex=r"\log \gamma_{\rm min}",
            bounds=(None, None),
        ),
        ModelParameter(
            "log_gamma_max",
            8.0,
            description="Logarithm of the maximum electron Lorentz factor",
            base_units=None,
            latex=r"\log \gamma_{\rm max}",
            bounds=(None, None),
        ),
        ModelParameter(
            "gamma_bulk",
            1.0,
            description="Bulk Lorentz factor of the emitting region",
            base_units=None,
            latex=r"\gamma_{\rm bulk}",
            bounds=(1, None),
        ),
        ModelParameter(
            "alpha",
            np.pi / 2,
            description="Pitch angle of the electrons (if not pitch-averaged)",
            base_units=None,
            latex=r"\alpha",
            bounds=(0, None),
        ),
        # Geometric Parameters.
        ModelParameter(
            "log_D_L",
            27.0,
            description="Logarithm of the luminosity distance to the source (cgs).",
            base_units=None,
            latex=r"\log_D_L",
            bounds=(0, None),
        ),
        ModelParameter(
            "log_Omega",
            -5,
            description="Logarithm of the solid angle subtended by the emitting region (steradians).",
            base_units=None,
            latex=r"\log \Omega",
            bounds=(None, None),
        ),
        # Core Parameters.
        ModelParameter(
            "log_B",
            0.0,
            description="Logarithm of the magnetic field strength (cgs).",
            base_units=None,
            latex=r"\log B",
            bounds=(None, None),
        ),
        ModelParameter(
            "log_V_eff",
            57.0,
            description="Logarithm of the effective volume of the emitting region (cgs).",
            base_units=None,
            latex=r"\log V_{\rm eff}",
            bounds=(0, None),
        ),
        ModelParameter(
            "redshift", 0.0, description="Redshift of the source", base_units=None, latex="z", bounds=(0, None)
        ),
    )
    VARIABLES = (
        ModelVariable(
            "log_nu", base_units=None, description="Logarithm of the observing frequency (Hz).", latex=r"\log \nu"
        ),
    )

    def __init__(
        self,
        pitch_averaged: bool = False,
    ):
        r"""
        Initialize the synchrotron SED model including synchrotron self-absorption (SSA), without cooling.

        Parameters
        ----------
        pitch_averaged : bool, optional
            If ``True``, the synchrotron emissivity is computed assuming
            an isotropic pitch-angle distribution using the pitch-angle
            averaged synchrotron kernel.

            If ``False``, the model evaluates the emission at a fixed
            pitch angle ``alpha`` provided as a parameter.

        Notes
        -----
        - The pitch-averaged formulation is appropriate for
          turbulent or disordered magnetic fields.
        - The fixed-angle formulation may be useful for ordered
          magnetic field geometries.
        - This choice modifies the normalization and characteristic
          break frequencies through factors of :math:`\sin\alpha`.

        Internally, this constructor instantiates:

        :class:`~radiation.synchrotron.SEDs.PowerLaw_SSA_SynchrotronSED`
        """
        super().__init__()

        self._sed = PowerLaw_SSA_SynchrotronSED()
        self._pitch_averaged = pitch_averaged

        self._register_init(pitch_averaged=pitch_averaged)

    # -------------------------------------------------- #
    # Forward Model Definition                           #
    # -------------------------------------------------- #
    def _forward_model(
        self,
        variables: _ModelVariablesInputRaw,
        parameters: _ModelParametersInputRaw,
    ) -> "OUTPUTS":
        # Use the SED object created at initialization to construct the
        # correct normalization data. The resulting norm dictionary provides
        # - F_peak, the log of the peak flux density (in cgs units)
        # - nu_m, the log of the characteristic synchrotron frequency (in Hz)
        # - nu_c, the log of the cooling frequency (in Hz)
        # - nu_max, the log of the maximum synchrotron frequency (in Hz)
        # - regime: an Enum indicating the spectral regime (e.g., "fast_cooling" or "slow_cooling")
        norm = self._sed._opt_from_physics_to_params(
            log_B=parameters["log_B"],
            log_V=parameters["log_V_eff"],
            log_D_L=parameters["log_D_L"],
            log_Omega=parameters["log_Omega"],
            log_gamma_min=parameters["log_gamma_min"],
            log_gamma_max=parameters["log_gamma_max"],
            p=parameters["p"],
            epsilon_E=parameters["epsilon_E"],
            epsilon_B=parameters["epsilon_B"],
            alpha=parameters["alpha"],
            gamma_bulk=parameters["gamma_bulk"],
            redshift=parameters["redshift"],
            pitch_average=self._pitch_averaged,
        )

        # Now we can use the SED to compute the correct flux density at the given frequencies.
        log_F_nu = self._sed._log_opt_sed_from_regime(
            variables["log_nu"],
            regime=norm["regime"],
            log_nu_m=norm["nu_m"],
            log_nu_a=norm["nu_a"],
            log_nu_max=norm["nu_max"],
            log_F_norm=norm["F_norm"],
            p=parameters["p"],
            s=-1.0,  # We can set s=1 for the smoothing parameter since we're not modeling a smoothed SED here.
        )

        return self.OUTPUTS(flux=log_F_nu)


class Cooling_SynchrotronSEDModel(Model):
    r"""
    Physical synchrotron spectral energy distribution including cooling but without synchrotron self-absorption.

    This model includes:

    - Power-law electron injection
    - Radiative cooling break
    - Equipartition normalization
    - Optional Doppler boosting
    - Cosmological redshift corrections

    Synchrotron self-absorption is not included.

    The model wraps
    :class:`~radiation.synchrotron.SEDs.PowerLaw_Cooling_SynchrotronSED`.

    All computations are performed in log-space.

    .. important::

        See :ref:`synchrotron_theory` and :ref:`synch_sed_theory`
        for theoretical background.

    .. dropdown:: Parameters

        Includes cooling Lorentz factor ``log_gamma_c`` but
        does not include ``log_Omega``.

    .. dropdown:: Variables

        ``log_nu`` — Log observing frequency (Hz).

    .. dropdown:: Returns

        ``flux`` — :math:`\log F_\nu`.

    Notes
    -----
    - Cooling regime (fast/slow) is determined self-consistently.
    - No absorption break is computed.
    - Observer-frame corrections applied internally.

    See Also
    --------
    :class:`SSA_Cooling_SynchrotronSEDModel`
    :class:`SSA_SynchrotronSEDModel`
    :class:`SynchrotronSEDModel`


    Examples
    --------

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt

        from triceratops.models.SEDs.synchrotron import (
            Cooling_SynchrotronSEDModel
        )

        rng = np.random.default_rng(42)

        # --------------------------------------------------
        # Instantiate model (Cooling only)
        # --------------------------------------------------

        model = Cooling_SynchrotronSEDModel(pitch_averaged=True)

        # --------------------------------------------------
        # Frequency grid
        # --------------------------------------------------

        nu = np.logspace(8, 16, 400)

        # --------------------------------------------------
        # Model parameters
        # --------------------------------------------------

        parameters = {
            "p": 2.5,
            "epsilon_E": 0.1,
            "epsilon_B": 0.1,
            "log_gamma_min": np.log(10.0),
            "log_gamma_c": np.log(1e3),
            "log_gamma_max": np.log(1e6),
            "gamma_bulk": 2.0,
            "alpha": np.pi / 2,
            "log_B": np.log(1.0),
            "log_V_eff": np.log(1e55),
            "log_D_L": np.log(1e27),
            "redshift": 0.01,
        }

        # --------------------------------------------------
        # Evaluate model
        # --------------------------------------------------

        output = model({"log_nu": np.log(nu)}, parameters)
        flux = np.exp(output.flux)

        # --------------------------------------------------
        # Compute break frequencies
        # --------------------------------------------------

        norm = model._sed._opt_from_physics_to_params(
            log_B=parameters["log_B"],
            log_V=parameters["log_V_eff"],
            log_D_L=parameters["log_D_L"],
            log_gamma_min=parameters["log_gamma_min"],
            log_gamma_c=parameters["log_gamma_c"],
            log_gamma_max=parameters["log_gamma_max"],
            p=parameters["p"],
            epsilon_E=parameters["epsilon_E"],
            epsilon_B=parameters["epsilon_B"],
            alpha=parameters["alpha"],
            gamma_bulk=parameters["gamma_bulk"],
            redshift=parameters["redshift"],
            pitch_average=True,
        )

        # --------------------------------------------------
        # Synthetic data (10% noise)
        # --------------------------------------------------

        synthetic = flux + 0.1 * flux * rng.normal(size=flux.size)

        # --------------------------------------------------
        # Plot
        # --------------------------------------------------

        plt.figure(figsize=(8,6))

        plt.loglog(nu, flux, lw=2, label="Model")
        plt.scatter(nu, synthetic, s=10, alpha=0.4, label="Synthetic data")

        # Mark breaks
        for key, label in [
            ("nu_m", r"$\nu_m$"),
            ("nu_c", r"$\nu_c$"),
            ("nu_max", r"$\nu_{\max}$"),
        ]:
            plt.axvline(np.exp(norm[key]), ls="--", alpha=0.6)
            plt.text(
                np.exp(norm[key]),
                np.amax(flux) * 0.3,
                label,
                rotation=90,
                verticalalignment="center"
            )

        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Flux Density [cgs]")
        plt.title("Synchrotron SED (Cooling Only)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    """

    # -------------------------------------------------- #
    # Output Definition                                  #
    # -------------------------------------------------- #
    OUTPUTS = _SynchrotronSEDOutput
    UNITS = _SynchrotronSEDOutputUnits

    # -------------------------------------------------- #
    # Parameters and Variables                           #
    # -------------------------------------------------- #
    PARAMETERS = (
        # Microphysical Parameters.
        ModelParameter("p", 2.5, description="Electron power law index", base_units=None, latex="p", bounds=(0, None)),
        ModelParameter(
            "epsilon_E",
            0.1,
            description="Electron energy fraction",
            base_units=None,
            latex=r"\epsilon_e",
            bounds=(0, 1),
        ),
        ModelParameter(
            "epsilon_B",
            0.01,
            description="Magnetic energy fraction",
            base_units=None,
            latex=r"\epsilon_B",
            bounds=(0, 1),
        ),
        ModelParameter(
            "log_gamma_min",
            0.0,
            description="Logarithm of the minimum electron Lorentz factor",
            base_units=None,
            latex=r"\log \gamma_{\rm min}",
            bounds=(None, None),
        ),
        ModelParameter(
            "log_gamma_max",
            8.0,
            description="Logarithm of the maximum electron Lorentz factor",
            base_units=None,
            latex=r"\log \gamma_{\rm max}",
            bounds=(None, None),
        ),
        ModelParameter(
            "gamma_bulk",
            1.0,
            description="Bulk Lorentz factor of the emitting region",
            base_units=None,
            latex=r"\gamma_{\rm bulk}",
            bounds=(1, None),
        ),
        ModelParameter(
            "alpha",
            np.pi / 2,
            description="Pitch angle of the electrons (if not pitch-averaged)",
            base_units=None,
            latex=r"\alpha",
            bounds=(0, None),
        ),
        ModelParameter(
            "log_gamma_c",
            0.0,
            description="Logarithm of the cooling Lorentz factor",
            base_units=None,
            latex=r"\log \gamma_c",
            bounds=(None, None),
        ),
        # Geometric Parameters.
        ModelParameter(
            "log_D_L",
            27.0,
            description="Logarithm of the luminosity distance to the source (cgs).",
            base_units=None,
            latex=r"\log_D_L",
            bounds=(0, None),
        ),
        # Core Parameters.
        ModelParameter(
            "log_B",
            0.0,
            description="Logarithm of the magnetic field strength (cgs).",
            base_units=None,
            latex=r"\log B",
            bounds=(None, None),
        ),
        ModelParameter(
            "log_V_eff",
            57.0,
            description="Logarithm of the effective volume of the emitting region (cgs).",
            base_units=None,
            latex=r"\log V_{\rm eff}",
            bounds=(0, None),
        ),
        ModelParameter(
            "redshift", 0.0, description="Redshift of the source", base_units=None, latex="z", bounds=(0, None)
        ),
    )
    VARIABLES = (
        ModelVariable(
            "log_nu", base_units=None, description="Logarithm of the observing frequency (Hz).", latex=r"\log \nu"
        ),
    )

    def __init__(
        self,
        pitch_averaged: bool = False,
    ):
        r"""
        Initialize the synchrotron SED model including radiative cooling but without synchrotron self-absorption.

        Parameters
        ----------
        pitch_averaged : bool, optional
            If ``True``, the synchrotron emissivity is computed assuming
            an isotropic pitch-angle distribution.

            If ``False``, a fixed pitch angle ``alpha`` must be supplied
            and the emissivity is computed explicitly using
            :math:`\sin\alpha`.

        Notes
        -----
        - Cooling introduces a spectral break at the cooling
          Lorentz factor :math:`\gamma_c`.
        - The cooling regime (fast or slow) is determined
          self-consistently.
        - No self-absorption break is computed in this model.
        - The pitch-angle treatment affects normalization and
          characteristic frequencies.

        Internally, this constructor instantiates:

        :class:`~radiation.synchrotron.SEDs.PowerLaw_Cooling_SynchrotronSED`
        """
        super().__init__()

        self._sed = PowerLaw_Cooling_SynchrotronSED()
        self._pitch_averaged = pitch_averaged

        self._register_init(pitch_averaged=pitch_averaged)

    # -------------------------------------------------- #
    # Forward Model Definition                           #
    # -------------------------------------------------- #
    def _forward_model(
        self,
        variables: _ModelVariablesInputRaw,
        parameters: _ModelParametersInputRaw,
    ) -> "OUTPUTS":
        # Use the SED object created at initialization to construct the
        # correct normalization data. The resulting norm dictionary provides
        # - F_peak, the log of the peak flux density (in cgs units)
        # - nu_m, the log of the characteristic synchrotron frequency (in Hz)
        # - nu_c, the log of the cooling frequency (in Hz)
        # - nu_max, the log of the maximum synchrotron frequency (in Hz)
        # - regime: an Enum indicating the spectral regime (e.g., "fast_cooling" or "slow_cooling")
        norm = self._sed._opt_from_physics_to_params(
            log_B=parameters["log_B"],
            log_V=parameters["log_V_eff"],
            log_D_L=parameters["log_D_L"],
            log_gamma_min=parameters["log_gamma_min"],
            log_gamma_c=parameters["log_gamma_c"],
            log_gamma_max=parameters["log_gamma_max"],
            p=parameters["p"],
            epsilon_E=parameters["epsilon_E"],
            epsilon_B=parameters["epsilon_B"],
            alpha=parameters["alpha"],
            gamma_bulk=parameters["gamma_bulk"],
            redshift=parameters["redshift"],
            pitch_average=self._pitch_averaged,
        )

        # Now we can use the SED to compute the correct flux density at the given frequencies.
        log_F_nu = self._sed._log_opt_sed_from_regime(
            variables["log_nu"],
            regime=norm["regime"],
            log_nu_m=norm["nu_m"],
            log_nu_c=norm["nu_c"],
            log_nu_max=norm["nu_max"],
            log_F_norm=norm["F_norm"],
            p=parameters["p"],
            s=-1.0,  # We can set s=1 for the smoothing parameter since we're not modeling a smoothed SED here.
        )

        return self.OUTPUTS(flux=log_F_nu)


class SynchrotronSEDModel(Model):
    r"""
    Optically thin synchrotron spectral energy distribution without radiative cooling or self-absorption.

    This is the simplest physically motivated synchrotron model,
    including:

    - Power-law electron distribution
    - Equipartition normalization
    - Optional Doppler boosting
    - Cosmological redshift corrections

    No cooling or absorption breaks are included.

    The model wraps
    :class:`~radiation.synchrotron.SEDs.PowerLaw_SynchrotronSED`.

    All computations are performed in log-space.

    .. important::

        Detailed implementation methodology may be found in
        :ref:`synch_sed_theory`.

    .. dropdown:: Parameters

        Includes microphysical parameters and geometry,
        but no cooling Lorentz factor and no absorption geometry.

    .. dropdown:: Variables

        ``log_nu`` — Log observing frequency (Hz).

    .. dropdown:: Returns

        ``flux`` — :math:`\log F_\nu`.

    Notes
    -----
    - Single power-law SED above injection frequency.
    - No cooling break.
    - No SSA turnover.
    - Observer-frame corrections applied internally.

    See Also
    --------
    :class:`SSA_Cooling_SynchrotronSEDModel`
    :class:`SSA_SynchrotronSEDModel`
    :class:`Cooling_SynchrotronSEDModel`

    Examples
    --------

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt

        from triceratops.models.SEDs.synchrotron import (
            SynchrotronSEDModel
        )

        rng = np.random.default_rng(42)

        # --------------------------------------------------
        # Instantiate model (Optically Thin Only)
        # --------------------------------------------------

        model = SynchrotronSEDModel(pitch_averaged=True)

        # --------------------------------------------------
        # Frequency grid
        # --------------------------------------------------

        nu = np.logspace(6, 13.1, 400)

        # --------------------------------------------------
        # Model parameters
        # --------------------------------------------------

        parameters = {
            "p": 2.5,
            "epsilon_E": 0.1,
            "epsilon_B": 0.1,
            "log_gamma_min": np.log(10.0),
            "log_gamma_max": np.log(1e6),
            "gamma_bulk": 2.0,
            "alpha": np.pi / 2,
            "log_B": np.log(1.0),
            "log_V_eff": np.log(1e55),
            "log_D_L": np.log(1e27),
            "redshift": 0.01,
        }

        # --------------------------------------------------
        # Evaluate model
        # --------------------------------------------------

        output = model({"log_nu": np.log(nu)}, parameters)
        flux = np.exp(output.flux)

        # --------------------------------------------------
        # Compute break frequencies
        # --------------------------------------------------

        norm = model._sed._opt_from_physics_to_params(
            log_B=parameters["log_B"],
            log_V=parameters["log_V_eff"],
            log_D_L=parameters["log_D_L"],
            log_gamma_min=parameters["log_gamma_min"],
            log_gamma_max=parameters["log_gamma_max"],
            p=parameters["p"],
            epsilon_E=parameters["epsilon_E"],
            epsilon_B=parameters["epsilon_B"],
            alpha=parameters["alpha"],
            gamma_bulk=parameters["gamma_bulk"],
            redshift=parameters["redshift"],
            pitch_average=True,
        )

        # --------------------------------------------------
        # Synthetic data (10% Gaussian noise)
        # --------------------------------------------------

        synthetic = flux + 0.1 * flux * rng.normal(size=flux.size)

        # --------------------------------------------------
        # Plot
        # --------------------------------------------------

        plt.figure(figsize=(8,6))

        plt.loglog(nu, flux, lw=2, label="Model")
        plt.scatter(nu, synthetic, s=10, alpha=0.4, label="Synthetic data")

        # Mark breaks
        for key, label in [
            ("log_nu_m", r"$\nu_m$"),
            ("log_nu_max", r"$\nu_{\max}$"),
        ]:
            plt.axvline(np.exp(norm[key]), ls="--", alpha=0.6)
            plt.text(
                np.exp(norm[key]),
                np.amax(flux) * 0.3,
                label,
                rotation=90,
                verticalalignment="center"
            )

        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Flux Density [cgs]")
        plt.title("Synchrotron SED (Optically Thin Only)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
    """

    # -------------------------------------------------- #
    # Output Definition                                  #
    # -------------------------------------------------- #
    OUTPUTS = _SynchrotronSEDOutput
    UNITS = _SynchrotronSEDOutputUnits

    # -------------------------------------------------- #
    # Parameters and Variables                           #
    # -------------------------------------------------- #
    PARAMETERS = (
        # Microphysical Parameters.
        ModelParameter("p", 2.5, description="Electron power law index", base_units=None, latex="p", bounds=(0, None)),
        ModelParameter(
            "epsilon_E",
            0.1,
            description="Electron energy fraction",
            base_units=None,
            latex=r"\epsilon_e",
            bounds=(0, 1),
        ),
        ModelParameter(
            "epsilon_B",
            0.01,
            description="Magnetic energy fraction",
            base_units=None,
            latex=r"\epsilon_B",
            bounds=(0, 1),
        ),
        ModelParameter(
            "log_gamma_min",
            0.0,
            description="Logarithm of the minimum electron Lorentz factor",
            base_units=None,
            latex=r"\log \gamma_{\rm min}",
            bounds=(None, None),
        ),
        ModelParameter(
            "log_gamma_max",
            8.0,
            description="Logarithm of the maximum electron Lorentz factor",
            base_units=None,
            latex=r"\log \gamma_{\rm max}",
            bounds=(None, None),
        ),
        ModelParameter(
            "gamma_bulk",
            1.0,
            description="Bulk Lorentz factor of the emitting region",
            base_units=None,
            latex=r"\gamma_{\rm bulk}",
            bounds=(1, None),
        ),
        ModelParameter(
            "alpha",
            np.pi / 2,
            description="Pitch angle of the electrons (if not pitch-averaged)",
            base_units=None,
            latex=r"\alpha",
            bounds=(0, None),
        ),
        ModelParameter(
            "log_gamma_c",
            0.0,
            description="Logarithm of the cooling Lorentz factor",
            base_units=None,
            latex=r"\log \gamma_c",
            bounds=(None, None),
        ),
        # Geometric Parameters.
        ModelParameter(
            "log_D_L",
            27.0,
            description="Logarithm of the luminosity distance to the source (cgs).",
            base_units=None,
            latex=r"\log_D_L",
            bounds=(0, None),
        ),
        # Core Parameters.
        ModelParameter(
            "log_B",
            0.0,
            description="Logarithm of the magnetic field strength (cgs).",
            base_units=None,
            latex=r"\log B",
            bounds=(None, None),
        ),
        ModelParameter(
            "log_V_eff",
            57.0,
            description="Logarithm of the effective volume of the emitting region (cgs).",
            base_units=None,
            latex=r"\log V_{\rm eff}",
            bounds=(0, None),
        ),
        ModelParameter(
            "redshift", 0.0, description="Redshift of the source", base_units=None, latex="z", bounds=(0, None)
        ),
    )
    VARIABLES = (
        ModelVariable(
            "log_nu", base_units=None, description="Logarithm of the observing frequency (Hz).", latex=r"\log \nu"
        ),
    )

    def __init__(
        self,
        pitch_averaged: bool = False,
    ):
        r"""
        Initialize the optically thin synchrotron SED model without radiative cooling or self-absorption.

        Parameters
        ----------
        pitch_averaged : bool, optional
            If ``True``, emission is computed assuming an isotropic
            pitch-angle distribution.

            If ``False``, a fixed pitch angle ``alpha`` must be provided
            and the synchrotron kernel is evaluated at that angle.

        Notes
        -----
        - This is the simplest physical synchrotron SED model.
        - Only the injection Lorentz factor determines the break.
        - No cooling or absorption breaks are included.
        - Pitch-angle treatment affects normalization and
          characteristic frequency scaling.

        Internally, this constructor instantiates:

        :class:`~radiation.synchrotron.SEDs.PowerLaw_SynchrotronSED`
        """
        self._sed = PowerLaw_SynchrotronSED()
        self._pitch_averaged = pitch_averaged

        self._register_init(pitch_averaged=pitch_averaged)

    # -------------------------------------------------- #
    # Forward Model Definition                           #
    # -------------------------------------------------- #
    def _forward_model(
        self,
        variables: _ModelVariablesInputRaw,
        parameters: _ModelParametersInputRaw,
    ) -> "OUTPUTS":
        # Use the SED object created at initialization to construct the
        # correct normalization data. The resulting norm dictionary provides
        # - F_peak, the log of the peak flux density (in cgs units)
        # - nu_m, the log of the characteristic synchrotron frequency (in Hz)
        # - nu_c, the log of the cooling frequency (in Hz)
        # - nu_max, the log of the maximum synchrotron frequency (in Hz)
        # - regime: an Enum indicating the spectral regime (e.g., "fast_cooling" or "slow_cooling")
        norm = self._sed._opt_from_physics_to_params(
            log_B=parameters["log_B"],
            log_V=parameters["log_V_eff"],
            log_D_L=parameters["log_D_L"],
            log_gamma_min=parameters["log_gamma_min"],
            log_gamma_max=parameters["log_gamma_max"],
            p=parameters["p"],
            epsilon_E=parameters["epsilon_E"],
            epsilon_B=parameters["epsilon_B"],
            alpha=parameters["alpha"],
            gamma_bulk=parameters["gamma_bulk"],
            redshift=parameters["redshift"],
            pitch_average=self._pitch_averaged,
        )

        # Now we can use the SED to compute the correct flux density at the given frequencies.
        log_F_nu = self._sed._log_opt_sed(
            variables["log_nu"],
            log_F_norm=norm["log_F_norm"],
            log_nu_m=norm["log_nu_m"],
            log_nu_max=norm["log_nu_max"],
            p=parameters["p"],
            s=-1.0,  # We can set s=1 for the smoothing parameter since we're not modeling a smoothed SED here.
        )

        return self.OUTPUTS(flux=log_F_nu)


class Synchrotron_SSA_SBPL_Model(Model):
    r"""
    Phenomenological synchrotron self-absorbed broken power-law spectral energy distribution (SED).

    This model provides a **time-independent**, smoothly broken power-law approximation to a
    synchrotron self-absorbed (SSA) radio spectrum. It is intended as a lightweight,
    phenomenological description of synchrotron emission from transient astrophysical
    sources such as supernovae (SNe) and gamma-ray bursts (GRBs), without enforcing
    dynamical self-consistency or detailed radiative transfer.

    The SED is parameterized as

    .. math::

       F_{\nu} = F_{\nu,0}
       \left[
           \left( \frac{\nu}{\nu_{\rm break}} \right)^{\alpha_{\rm thick}/s}
           +
           \left( \frac{\nu}{\nu_{\rm break}} \right)^{\alpha_{\rm thin}/s}
       \right]^s,

    where :math:`F_{\nu,0}` is the normalization at the break frequency
    :math:`\nu_{\rm break}`. The spectral indices are tied to the electron energy
    distribution power-law index :math:`p` via

    .. math::

       \alpha_{\rm thick} = \frac{5}{2},
       \qquad
       \alpha_{\rm thin} = -\frac{p - 1}{2}.

    This choice reproduces the canonical optically thick and optically thin synchrotron
    spectral slopes expected for a homogeneous emitting region with a power-law electron
    population.

    This model is **phenomenological** in nature: it does not compute or require physical
    quantities such as shock radius, magnetic field strength, or particle density, and it
    does not include time evolution. It is therefore most appropriate for:

    - single-epoch radio SED fitting,
    - exploratory or diagnostic spectral modeling,
    - comparison against physically motivated, time-dependent synchrotron models.

    For detailed discussions of synchrotron spectral shapes and their physical
    interpretation, see :footcite:t:`rybickilightman` and
    :footcite:t:`demarchiRadioAnalysisSN2004C2022`.

    .. dropdown:: Parameters

       .. list-table::
          :widths: 25 25 50
          :header-rows: 1

          * - **Name**
            - **Symbol**
            - **Description**
          * - ``norm``
            - :math:`F_{\nu,0}`
            - Flux density normalization at the break frequency.
          * - ``nu_break``
            - :math:`\nu_{\rm break}`
            - Synchrotron self-absorption turnover (break) frequency.
          * - ``p``
            - :math:`p`
            - Power-law index of the relativistic electron energy distribution.
          * - ``s``
            - :math:`s`
            - Smoothness parameter controlling the sharpness of the spectral break.

    .. dropdown:: Variables

       .. list-table::
          :widths: 25 25 50
          :header-rows: 1

          * - **Name**
            - **Symbol**
            - **Description**
          * - ``frequency``
            - :math:`\nu`
            - Observing frequency at which the SED is evaluated.

    .. dropdown:: Returns

       .. list-table::
          :widths: 25 25 50
          :header-rows: 1

          * - **Name**
            - **Symbol**
            - **Description**
          * - ``flux_density``
            - :math:`F_{\nu}`
            - Flux density evaluated at the observing frequency.

    Notes
    -----
    In most scenarios, the break frequency will be ill-constrained without broadband sampling of
    the radio SED. Care should be taken when interpreting fit results, especially when using
    sparse data.
    """

    # ============================================== #
    # Model Initialization                           #
    # ============================================== #
    def __init__(self):
        """Initialize the model with given parameters and variables."""
        self._register_init()

    # =============================================== #
    # Parameter and Variable Declarations             #
    # =============================================== #
    # Each model must declare its parameters and variables as class-level attributes. These
    # must each be instances of `ModelParameter` and `ModelVariable`, respectively.
    PARAMETERS: tuple["ModelParameter", ...] = (
        ModelParameter(
            "norm",
            1.0,
            description="Normalization constant of the SED.",
            base_units="Jy",
            bounds=(0.0, None),
            latex=r"F_{\rm nu,0}",
        ),
        ModelParameter(
            "nu_break",
            5.0,
            description="Break frequency of the SED.",
            base_units="GHz",
            bounds=(0.0, None),
            latex=r"\nu_{\rm break}",
        ),
        ModelParameter(
            "p",
            3.0,
            description="Power-law index for the electron energy distribution.",
            base_units="",
            bounds=(0, None),
            latex=r"p",
        ),
        ModelParameter(
            "s",
            -0.5,
            description="The smoothing parameter of the broken power-law.",
            base_units="",
            bounds=(None, 0),
            latex=r"s",
        ),
    )
    """tuple of :class:`ModelParameter`: The model's parameters.

    Each element of :attr:`PARAMETERS` is a :class:`ModelParameter` instance that defines a single
    parameter of the model. These parameters are used to configure the model and control its behavior. Each
    parameter contains information about the base units, default value, and valid range for that parameter.
    """
    VARIABLES: tuple["ModelVariable", ...] = (
        ModelVariable(
            "frequency", description="Observing frequency at which to evaluate the SED.", base_units="GHz", latex=r"\nu"
        ),
    )
    """tuple of :class:`ModelVariable`: The model's variables.

    Each element of :attr:`VARIABLES` is a :class:`ModelVariable` instance that defines a single
    variable of the model. These variables represent the inputs to the model that can vary during
    evaluation. Each variable contains information about the base units, name, etc. of the variable. Notably,
    variables differ from the parameters (:attr:`PARAMETERS`) in that variables are do **NOT** have default
    values, do **NOT** have validity ranges, and are expected to be provided at model evaluation time.
    """

    # =============================================== #
    # Model Metadata Declarations                     #
    # =============================================== #
    # Each model must declare its parameters and variables as class-level attributes.
    OUTPUTS: tuple[str, ...] = namedtuple("Synchrotron_SSA_SBPL_SEDOutput", ["flux_density"])
    """tuple of str: The names of the model's outputs.

    Each element of :attr:`OUTPUTS` is a string that defines the name of a single output of the model.
    These names correspond to the keys in the dictionary returned by the model's evaluation method.
    """
    UNITS = OUTPUTS(flux_density=u.Jy)
    """tuple of :class:`astropy.units.Unit`: The units of the model's outputs.

    Each element of :attr:`UNITS` is an :class:`astropy.units.Unit` instance that defines the units of a single
    output of the model. The order of the units in :attr:`UNITS` corresponds to the order of the output names
    in :attr:`OUTPUTS`.
    """
    DESCRIPTION: str = ""
    """str: A brief description of the model."""
    REFERENCE: str = ""
    """str: A reference for the model, e.g., a journal article or textbook."""

    # =============================================== #
    # Model Evaluation Method                         #
    # =============================================== #
    def _forward_model(
        self,
        variables: _ModelVariablesInputRaw,
        parameters: _ModelParametersInputRaw,
    ) -> "OUTPUTS":
        # Construct the indices from the value of p.
        p = parameters["p"]
        alpha_2 = 5 / 2
        alpha_1 = -(p - 1) / 2

        result = smoothed_BPL(
            variables["frequency"], parameters["norm"], parameters["nu_break"], alpha_1, alpha_2, parameters["s"]
        )
        return self.OUTPUTS(flux_density=result)
