"""
Supernova shock model based on the work of Chevalier (1982).

This model describes the evolution of a supernova shock wave as it expands into the surrounding medium using
the self-similar solution set introduced in :footcite:t:`ChevalierXRayRadioEmission1982`.

References
----------
.. footbibliography::
"""

from collections import namedtuple

import numpy as np
from astropy import units as u

from triceratops.dynamics.rankine_hugoniot import _compute_s_c_shock_magnetic_field_cgs
from triceratops.dynamics.supernovae.shock_dynamics import (
    ChevalierSelfSimilarShockEngine,
)
from triceratops.models.core.base import Model
from triceratops.models.core.parameters import ModelParameter, ModelVariable
from triceratops.radiation.synchrotron.SEDs import SSA_SED_PowerLaw

__all__ = [
    "ChevalierShockModel",
]


# noinspection PyProtectedMember
class ChevalierShockModel(Model):
    r"""
    Supernova shock synchrotron emission model based on Chevalier (1982).

    This model simulates the synchrotron emission from a supernova shock wave expanding into a circumstellar medium
    (CSM) using the self-similar solutions presented in :footcite:t:`ChevalierXRayRadioEmission1982`.
    The model computes the shock dynamics and resulting synchrotron spectral energy distribution (SED)
    based on a set of macroscopic and microscopic parameters.

    """

    # =============================================== #
    # Parameter and Variable Declarations             #
    # =============================================== #
    PARAMETERS = (
        # ------------- Microscopic Shock Parameters ------------- #
        ModelParameter(
            "epsilon_e",
            0.1,
            description="Fraction of shock energy in relativistic electrons.",
            bounds=(0, None),
            latex=r"\epsilon_e",
            base_units="",
        ),
        ModelParameter(
            "epsilon_B",
            0.1,
            description="Fraction of shock energy in magnetic fields.",
            bounds=(0, None),
            latex=r"\epsilon_B",
            base_units="",
        ),
        ModelParameter(
            "p",
            3.0,
            description="Power-law index of the electron energy distribution.",
            bounds=(2, None),
            latex=r"p",
            base_units="",
        ),
        ModelParameter(
            "gamma_max",
            1e7,
            description="The maximum Lorentz factor of the electron population. Only used if p < 2.",
            bounds=(1, None),
            latex=r"\gamma_{\rm max}",
            base_units="",
        ),
        ModelParameter(
            "gamma_min",
            1,
            description="The minimum Lorentz factor of the electron population.",
            bounds=(0, None),
            latex=r"\gamma_{\rm max}",
            base_units="",
        ),
        # ----------- Geometric Parameters ------------ #
        ModelParameter(
            "f",
            0.5,
            description="Filling factor of the emitting region.",
            bounds=(0, 1),
            latex=r"f",
            base_units="",
        ),
        ModelParameter(
            "theta",
            np.pi / 2,
            description="The pitch angle of the electron population.",
            bounds=(-np.pi, np.pi),
            latex=r"\theta",
            base_units="",
        ),
        ModelParameter("D", 150, description="Distance to the source.", base_units="Mpc", latex=r"D", bounds=(0, None)),
        ModelParameter(
            "smoothing_s",
            default=-0.5,
            description="The smoothing parameter of the broken power-law.",
            bounds=(None, 0),
            latex=r"s",
            base_units="",
        ),
        # ----------- Macroscopic Shock Parameters ------------ #
        ModelParameter(
            "E_ej",
            default=1e51,
            description="The kinetic energy of the supernova ejecta.",
            bounds=(0, None),
            latex=r"E_{\rm ej}",
            base_units="erg",
        ),
        ModelParameter(
            "M_ej",
            default=1.4,
            description="The mass of the supernova ejecta.",
            bounds=(0, None),
            latex=r"M_{\rm ej}",
            base_units="g",
        ),
        ModelParameter(
            "n",
            default=10,
            description="The power-law index of the outer ejecta density profile.",
            bounds=(5, None),
            latex=r"n",
            base_units="",
        ),
        ModelParameter(
            "s",
            default=2,
            description="The power-law index of the circumstellar medium density profile.",
            bounds=(0, 3),
            latex=r"s",
            base_units="",
        ),
        ModelParameter(
            "rho_0",
            default=1,
            description="The CSM density at a fiducial radius of 1e14 cm.",
            bounds=(0, None),
            latex=r"\rho_0",
            base_units="g/cm**3",
        ),
        ModelParameter(
            "delta",
            default=0,
            description="The power-law index of the inner ejecta density profile.",
            bounds=(0, 3),
            latex=r"\delta",
            base_units="",
        ),
    )
    VARIABLES = (
        ModelVariable(
            "frequency", description="Observing frequency at which to evaluate the SED.", base_units="Hz", latex=r"\nu"
        ),
        ModelVariable("time", description="Time since explosion.", base_units="s", latex=r"t"),
    )
    # =============================================== #
    # Model Metadata Declarations                     #
    # =============================================== #
    # Each model must declare its parameters and variables as class-level attributes.
    OUTPUTS = namedtuple("ChevalierShockOutputs", ["flux_density"])
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
    REFERENCE: str = "Chevalier (1982), deMarchi et. al. (2022)"
    """str: A reference for the model, e.g., a journal article or textbook."""

    # =============================================== #
    # Initialization Method                           #
    # =============================================== #
    def __init__(self):
        """Initialize the model."""
        # Initialize the base Model class
        super().__init__()

        # Now generate the shock-engine. This allows us to avoid having
        # to re-instantiate it on every forward model call.
        self.shock_engine = ChevalierSelfSimilarShockEngine()

        # Set up the SED engine
        self.sed = SSA_SED_PowerLaw()

        self._register_init()

    # =============================================== #
    # Core Model Evaluation Method                    #
    # =============================================== #
    def _forward_model(
        self,
        variables,
        parameters,
    ):
        # First, use the shock engine to compute the shock properties at a given time.
        shock_properties = self.shock_engine._compute_shock_properties_cgs(
            variables["time"],
            parameters["E_ej"],
            parameters["M_ej"],
            parameters["rho_0"] * (1e14) ** parameters["s"],
            parameters["n"],
            parameters["s"],
            parameters["delta"],
        )
        _shock_radius, _shock_velocity = shock_properties["radius"], shock_properties["velocity"]

        # Now use the shock radius and velocity to compute the
        # corresponding magnetic field strength.
        _shock_B = _compute_s_c_shock_magnetic_field_cgs(
            _shock_velocity,
            parameters["rho_0"] * (_shock_radius / 1e14) ** -parameters["s"],
            gamma=5 / 3,
            epsilon_B=parameters["epsilon_B"],
        )

        # First use the parameters to compute the BPL parameters from the
        # core physics parameters.
        nu_brk, F_brk = self.sed._opt_from_physics_to_params(
            _shock_B,
            _shock_radius,
            parameters["D"],  # cm
            p=parameters["p"],
            f=parameters["f"],
            theta=parameters["theta"],
            epsilon_B=parameters["epsilon_B"],
            epsilon_E=parameters["epsilon_e"],
            gamma_min=parameters["gamma_min"],
            gamma_max=parameters["gamma_max"],
        )

        # Return the flux density in Jy
        # 1e-23 erg/s/cm^2/Hz = 1 Jy
        return self.OUTPUTS(
            flux_density=1e-23
            * self.sed._log_opt_sed(
                variables["frequency"],
                nu_brk,
                F_brk,
                parameters["p"],
                parameters["smoothing_s"],
            )
        )
