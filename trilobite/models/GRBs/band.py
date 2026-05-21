r"""
Band function spectral model for gamma-ray burst prompt emission.

The Band function :footcite:t:`1993ApJ...413..281B` is the
canonical empirical spectral model for GRB prompt emission. It describes
a smoothly broken power law with a low-energy exponential cut-off,
defined entirely by four physically interpretable parameters: the
low-energy photon index, the high-energy photon index, the peak energy
of the :math:`nu F_nu` spectrum, and a normalisation amplitude.

References
----------
.. footbibliography::
"""

from collections import namedtuple

import numpy as np
from astropy import units as u

from trilobite.models.core.base import Model
from trilobite.models.core.parameters import ModelParameter, ModelVariable

__all__ = [
    "BandFunctionModel",
]

_BandOutputs = namedtuple("BandOutputs", ["photon_flux"])


class BandFunctionModel(Model):
    r"""
    Empirical Band function model for GRB prompt-emission spectra.

    The Band function :footcite:t:`1993ApJ...413..281B` is a
    smoothly joined broken power law defined as

    .. math::

        N(E) = A \begin{cases}
            \left(\dfrac{E}{E_{\rm piv}}\right)^{\!\alpha}
            \exp\!\left(-\dfrac{(2+\alpha)\,E}{E_{\rm peak}}\right)
            & E \leq E_c \\[6pt]
            \left(\dfrac{E}{E_{\rm piv}}\right)^{\!\beta}
            \exp(\beta - \alpha)
            \left(\dfrac{(\alpha-\beta)\,E_{\rm peak}}
                        {(2+\alpha)\,E_{\rm piv}}\right)^{\!\alpha-\beta}
            & E > E_c
        \end{cases}

    where the break energy is

    .. math::

        E_c = \frac{(\alpha - \beta)}{2 + \alpha}\,E_{\rm peak}.

    Continuity of :math:`N(E)` at :math:`E = E_c` is guaranteed
    analytically.  The function reduces to a pure power law with an
    exponential cut-off (the "Comptonized" model) when
    :math:`\beta \to -\infty`.

    .. dropdown:: Parameters

        .. list-table::
           :widths: 20 20 20 40
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Default**
             - **Description**
           * - ``alpha``
             - :math:`\alpha`
             - :math:`-1`
             - Low-energy photon spectral index.
           * - ``beta``
             - :math:`\beta`
             - :math:`-2.5`
             - High-energy photon spectral index.
           * - ``E_peak``
             - :math:`E_{\rm peak}`
             - :math:`300\ \rm keV`
             - Peak energy of the :math:`\nu F_\nu` (SED) spectrum.
           * - ``E_piv``
             - :math:`E_{\rm piv}`
             - :math:`100\ \rm keV`
             - Pivot (reference normalisation) energy.
           * - ``A``
             - :math:`A`
             - :math:`10^{-3}\ \rm ph\,cm^{-2}\,s^{-1}\,keV^{-1}`
             - Differential photon flux amplitude.

    .. dropdown:: Variables

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``energy``
             - :math:`E`
             - Photon energy at which the spectrum is evaluated.

    .. dropdown:: Returns

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``photon_flux``
             - :math:`N(E)`
             - Differential photon flux
               :math:`[\rm ph\,cm^{-2}\,s^{-1}\,keV^{-1}]`.

    Notes
    -----
    The constraint :math:`\alpha > \beta` is required to ensure
    :math:`E_c > 0`.  The constraint :math:`\alpha > -2` is required to
    ensure that the exponential cut-off has the correct sign.
    Both constraints are enforced at evaluation time via
    :meth:`_check_model_bounds_base`.

    The pivot energy :math:`E_{\rm piv}` is conventionally fixed to
    :math:`100\ \rm keV` and acts purely as a reference normalisation
    point; its value does not affect the spectral *shape*, only the
    absolute normalisation of :math:`A`.
    """

    # =============================================== #
    # Parameter and Variable Declarations             #
    # =============================================== #
    PARAMETERS = (
        ModelParameter(
            "alpha",
            default=-1.0,
            description="Low-energy photon spectral index.",
            bounds=(-2, None),
            latex=r"\alpha",
            base_units="",
        ),
        ModelParameter(
            "beta",
            default=-2.5,
            description="High-energy photon spectral index.",
            bounds=(None, None),
            latex=r"\beta",
            base_units="",
        ),
        ModelParameter(
            "E_peak",
            default=300.0,
            description="Peak energy of the nu*F_nu (SED) spectrum.",
            bounds=(0, None),
            latex=r"E_{\rm peak}",
            base_units="keV",
        ),
        ModelParameter(
            "E_piv",
            default=100.0,
            description="Pivot (reference normalisation) energy.",
            bounds=(0, None),
            latex=r"E_{\rm piv}",
            base_units="keV",
        ),
        ModelParameter(
            "A",
            default=1e-3,
            description="Differential photon flux amplitude.",
            bounds=(0, None),
            latex=r"A",
            base_units="1/(cm**2 s keV)",
        ),
    )
    VARIABLES = (
        ModelVariable(
            "energy",
            description="Photon energy at which the spectrum is evaluated.",
            base_units="keV",
            latex=r"E",
        ),
    )

    # =============================================== #
    # Model Metadata Declarations                     #
    # =============================================== #
    OUTPUTS = _BandOutputs
    """tuple of str: The names of the model's outputs.

    Each element of :attr:`OUTPUTS` is a string that defines the name of a single output of the model.
    These names correspond to the keys in the dictionary returned by the model's evaluation method.
    """
    UNITS = OUTPUTS(photon_flux=u.Unit("1/(cm**2 s keV)"))
    """tuple of :class:`astropy.units.Unit`: The units of the model's outputs.

    Each element of :attr:`UNITS` is an :class:`astropy.units.Unit` instance that defines the units of a single
    output of the model. The order of the units in :attr:`UNITS` corresponds to the order of the output names
    in :attr:`OUTPUTS`.
    """
    DESCRIPTION: str = "Empirical Band function for GRB prompt-emission spectra."
    """str: A brief description of the model."""
    REFERENCE: str = "Band et al. (1993)"
    """str: A reference for the model, e.g., a journal article or textbook."""

    # =============================================== #
    # Initialization Method                           #
    # =============================================== #
    def __init__(self):
        """Initialize the model."""
        super().__init__()
        self._register_init()

    # =============================================== #
    # Model-Level Bounds Checking                     #
    # =============================================== #
    def _check_model_bounds_base(self, variables, parameters):
        """Enforce alpha > beta and alpha > -2.

        Parameters
        ----------
        variables : dict of str to float or numpy.ndarray
            Model variables in base units (unused here).
        parameters : dict of str to float or numpy.ndarray
            Model parameters in base units.

        Raises
        ------
        ValueError
            If ``alpha <= beta`` or ``alpha <= -2``.
        """
        alpha = parameters["alpha"]
        beta = parameters["beta"]

        if np.any(alpha <= beta):
            raise ValueError(
                f"BandFunctionModel requires alpha > beta to ensure E_c > 0, but got alpha={alpha}, beta={beta}."
            )
        if np.any(alpha <= -2):
            raise ValueError(
                f"BandFunctionModel requires alpha > -2 so that the exponential "
                f"cut-off has the correct sign, but got alpha={alpha}."
            )

    # =============================================== #
    # Core Model Evaluation Method                    #
    # =============================================== #
    def _forward_model(self, variables, parameters):
        r"""Evaluate the Band function.

        Parameters
        ----------
        variables : dict of str to float or numpy.ndarray
            Must contain ``'energy'`` in keV.
        parameters : dict of str to float or numpy.ndarray
            Must contain ``'alpha'``, ``'beta'``, ``'E_peak'``,
            ``'E_piv'``, and ``'A'`` in their declared base units.

        Returns
        -------
        BandOutputs
            Named tuple with field ``photon_flux`` in
            :math:`rm ph,cm^{-2},s^{-1},keV^{-1}`.
        """
        E = np.asarray(variables["energy"], dtype=float)
        alpha = parameters["alpha"]
        beta = parameters["beta"]
        E_peak = parameters["E_peak"]
        E_piv = parameters["E_piv"]
        A = parameters["A"]

        # Break (transition) energy separating the two power-law regimes.
        E_c = (alpha - beta) / (2.0 + alpha) * E_peak

        # Low-energy segment: power law with exponential cut-off.
        low = A * (E / E_piv) ** alpha * np.exp(-(2.0 + alpha) * E / E_peak)

        # High-energy segment: pure power law.
        # The normalisation factor ensures continuity at E = E_c.
        high_norm = np.exp(beta - alpha) * ((alpha - beta) * E_peak / ((2.0 + alpha) * E_piv)) ** (alpha - beta)
        high = A * (E / E_piv) ** beta * high_norm

        return self.OUTPUTS(photon_flux=np.where(E <= E_c, low, high))
