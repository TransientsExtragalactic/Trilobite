r"""
Two-component synchrotron model for LFBOTs based on the work of Diggins+2026.

This module implements a radio synchrotron model for the circumstellar-medium (CSM) interaction of luminous fast
blue optical transients (LFBOTs). The CSM is axisymmetric (a dense, disk-confined interior component plus a
spherical exterior component; see
:class:`~trilobite.dynamics.profiles.csm.AxisymmetricCSMDensityProfile`), and the emission is modeled along exactly
two sightlines, the equator and the pole, each of which stands in for its solid-angle sector of the sphere:

1. A pressure-driven thin-shell shock is solved independently along each sightline
   (:class:`~trilobite.dynamics.shocks.numerical.PressureDrivenThinShellShockEngine`).
2. The post-shock thermal energy density sets the magnetic field (:math:`\epsilon_B`) and the normalization of a
   cooled broken power-law electron distribution (:math:`\epsilon_e`).
3. The numerical synchrotron engine computes each sector's spectrum.
4. Free-free absorption by the CSM exterior to the shock attenuates each sector.
5. The observed flux density is the sum of the attenuated sectors, weighted by their solid-angle fractions.
"""

from collections import namedtuple
from typing import NamedTuple, Optional, Union

import numpy as np
from astropy import constants as const
from astropy import units as u
from scipy.integrate import quad

from trilobite.dynamics.profiles.csm import AxisymmetricCSMDensityProfile
from trilobite.dynamics.profiles.ejecta import BrokenPowerLawEjectaProfile
from trilobite.dynamics.shocks.numerical import PressureDrivenThinShellShockEngine
from trilobite.dynamics.shocks.utils import make_homologous_stationary_sources
from trilobite.models.core.base import Model
from trilobite.models.core.parameters import ModelParameter, ModelVariable
from trilobite.physics_utils.cosmology import get_cosmology, resolve_cosmological_distances
from trilobite.radiation.free_free.absorption import _compute_RJ_ff_optical_depth_EM
from trilobite.radiation.synchrotron.cooling import _opt_compute_synchrotron_cooling_gamma
from trilobite.radiation.synchrotron.electron_distributions import BrokenPowerLaw
from trilobite.radiation.synchrotron.SEDs.numerical.core import NumericalSynchrotronEngine
from trilobite.utils.misc_utils import ensure_in_units

__all__ = [
    "Diggins26_TwoComponentModel",
    "Diggins26SectorDynamics",
    "Diggins26Dynamics",
    "Diggins26SectorRadiation",
    "Diggins26Tabulation",
]

_C_CGS = const.c.cgs.value
_M_P_CGS = const.m_p.cgs.value
_M_SUN_CGS = const.M_sun.cgs.value
_DAY = 86400.0

# Sightlines standing in for the equatorial (disk) and polar sectors.
_THETA_EQUATOR = np.pi / 2
_THETA_POLE = 0.0

# Half-width (in ln r) beyond which the emission-measure quadrature panels are subdivided, and the
# minimum panel width regardless of how sharp the CSM transition is.
_EM_MAX_PANEL = 0.4
_EM_MIN_PANEL = 0.02
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(8)

# The parameters on which the shock dynamics depend.
_DYNAMICS_PARAMETERS = (
    "rho_int",
    "rho_ext",
    "alpha_int",
    "alpha_ext",
    "R_trans",
    "delta_trans",
    "f_pole",
    "aspect",
    "E_ej",
    "M_ej",
    "n",
    "delta",
)


# =========================================================================== #
# Result Containers                                                           #
# =========================================================================== #
class Diggins26SectorDynamics(NamedTuple):
    """Shock state of one sightline at the requested epochs (all :class:`~astropy.units.Quantity`)."""

    radius: u.Quantity
    """Shock radius."""
    velocity: u.Quantity
    """Shock velocity."""
    mass: u.Quantity
    """Shell mass."""
    thermal_energy_density: u.Quantity
    """Post-shock thermal energy density."""


class Diggins26Dynamics(NamedTuple):
    """
    Shock dynamics of both sightlines.

    Returned by :meth:`~trilobite.models.LFBOTs._diggins26.Diggins26_TwoComponentModel.compute_dynamics`, and
    accepted by :meth:`~trilobite.models.LFBOTs._diggins26.Diggins26_TwoComponentModel.tabulate` so that many
    frequencies can be evaluated without re-solving the shock.
    """

    time: u.Quantity
    """Unique, sorted observer-frame epochs."""
    time_rest: u.Quantity
    """The same epochs in the source frame, ``time / (1 + z)``."""
    equatorial: Diggins26SectorDynamics
    """Equatorial sightline."""
    polar: Diggins26SectorDynamics
    """Polar sightline."""
    parameters: dict
    """The dynamics parameters (CGS floats) this state was computed with."""


class Diggins26SectorRadiation(NamedTuple):
    """Radiation from one sector on the ``(n_t, n_nu)`` tabulation grid."""

    flux_density: u.Quantity
    """Intrinsic observer-frame flux density of the sector (before free-free absorption), weighted by its area."""
    flux_density_attenuated: u.Quantity
    """Flux density after free-free absorption by the CSM exterior to the shock."""
    tau_ff: np.ndarray
    """Free-free optical depth to the shock, evaluated at the source-frame frequency."""
    emission_measure: u.Quantity
    """Emission measure of the CSM exterior to the shock, :math:`\\int n_e n_i\\,dr`, shape ``(n_t,)``."""
    magnetic_field: u.Quantity
    """Post-shock magnetic field, shape ``(n_t,)``."""
    gamma_c: np.ndarray
    """Cooling Lorentz factor (after clipping to the electron-distribution support), shape ``(n_t,)``."""
    electron_normalization: u.Quantity
    """Electron distribution amplitude :math:`N_0`, shape ``(n_t,)``."""
    area_fraction: float
    """The sector's solid-angle fraction, used as the area filling factor."""


class Diggins26Tabulation(NamedTuple):
    """Full model output on independent time and frequency axes, returned by ``tabulate``."""

    time: u.Quantity
    """Observer-frame epochs, shape ``(n_t,)``."""
    frequency: u.Quantity
    """Observer-frame frequencies, shape ``(n_nu,)``."""
    flux_density: u.Quantity
    """Total attenuated flux density, shape ``(n_t, n_nu)``."""
    equatorial: Diggins26SectorRadiation
    """Equatorial sector."""
    polar: Diggins26SectorRadiation
    """Polar sector."""
    dynamics: Diggins26Dynamics
    """The shock dynamics used."""


class _SectorState(NamedTuple):
    """CGS arrays for one sightline (internal)."""

    radius: np.ndarray
    velocity: np.ndarray
    mass: np.ndarray
    eth: np.ndarray


# =========================================================================== #
# Model Definition                                                            #
# =========================================================================== #
class Diggins26_TwoComponentModel(Model):
    r"""
    Two-component (equatorial and polar) synchrotron model for the radio emission of LFBOTs.

    The CSM is axisymmetric (see :class:`~trilobite.dynamics.profiles.csm.AxisymmetricCSMDensityProfile`),

    .. math::

        \rho(r, \theta) = \rho_{\rm ext}(r) + \rho_{\rm int}(r)\,S(r)\,G(\theta),

    and the interaction of freely expanding, broken power-law ejecta with it is followed along two sightlines,
    the equator (:math:`\theta = \pi/2`) and the pole (:math:`\theta = 0`). Along each, a pressure-driven thin
    shell is integrated from :math:`t_0`; the post-shock thermal energy density :math:`e_{\rm th}` sets the
    magnetic field :math:`B = \sqrt{8\pi\epsilon_B e_{\rm th}}` and the amplitude of a synchrotron-cooled broken
    power-law electron distribution (:math:`N(\gamma) \propto \gamma^{-p}` below the cooling break and
    :math:`\gamma^{-(p+1)}` above) carrying a fraction :math:`\epsilon_e` of :math:`e_{\rm th}`. The spectrum of each
    sector comes from the numerical synchrotron engine (pitch-angle averaged, including self-absorption) with geometry

    .. math::

        A_{\rm eff} = \pi R^2 f_A, \qquad \ell_{\rm eff} = f_V^{1/3} R,

    where :math:`f_A` is the sector's solid-angle fraction (derived from the CSM geometry, not a parameter) and
    :math:`f_V` is a shared volume filling factor. Free-free absorption by the ionized CSM exterior to the shock
    (a hydrogen and helium plasma with hydrogen mass fraction :math:`X` and electron temperature :math:`T_e`)
    attenuates each sector by :math:`e^{-\tau_{\rm ff}}`. The model output is the sum of the sectors.

    .. rubric:: Redshift

    Redshift and distance are fixed model configuration (constructor arguments), never parameters. The model takes
    **observer-frame** ``time`` and ``frequency``. Internally, the dynamics are solved at
    :math:`t = t_{\rm obs}/(1+z)`, the free-free optical depth is evaluated at :math:`\nu = \nu_{\rm obs}(1+z)`, and
    the synchrotron engine applies the cosmological flux transformation using :math:`D_A`.

    .. rubric:: Evaluation paths

    :meth:`forward_model`
        The inference path. ``time`` and ``frequency`` are paired (one flux per pair); the shock is solved once per
        sector at the unique rest-frame epochs and each epoch's spectrum is evaluated only at its own frequencies.

    :meth:`tabulate`
        The analysis and plotting path. ``time`` and ``frequency`` are independent axes, and the result is a
        :class:`~trilobite.models.LFBOTs._diggins26.Diggins26Tabulation` containing the total and the per-sector
        flux densities, optical depths, magnetic fields, and so on, on an ``(n_t, n_nu)`` grid. The shock can be solved
        once with :meth:`compute_dynamics` and reused.

    .. rubric:: Infeasible parameters

    Instead of raising exceptions, :meth:`forward_model` returns NaN flux for a short, enumerated list of
    infeasible parameter combinations (so that samplers can reject them), and lets every other exception
    propagate:

    - the ejecta profile cannot be normalized to :math:`(E_{\rm ej}, M_{\rm ej})` with :math:`v_{\max} = c`,
    - :math:`v_0 = \sqrt{2E_{\rm ej}/M_{\rm ej}}` exceeds ``beta_max`` times :math:`c`,
    - :math:`\gamma_{\max}` is too close to :math:`\gamma_{\min}` to place a cooling break,
    - there is no CSM mass inside the starting radius, or
    - the shock ODE integration fails.

    :meth:`diagnose` reports which condition rejected a parameter set. Hard parameter-bound violations raise
    ``ValueError`` as for every Trilobite model.

    .. dropdown:: Parameters

        .. list-table::
           :widths: 20 20 60
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``rho_int``
             - :math:`\rho_{\rm int}`
             - Solid-angle-averaged interior CSM density at ``r_ref``.
           * - ``rho_ext``
             - :math:`\rho_{\rm ext}`
             - Exterior (spherical) CSM density at ``r_ref``.
           * - ``alpha_int``, ``alpha_ext``
             - :math:`\alpha_{\rm int}`, :math:`\alpha_{\rm ext}`
             - Power-law slopes of the interior and exterior components.
           * - ``R_trans``, ``delta_trans``
             - :math:`r_{\rm trans}`, :math:`\delta_{\rm trans}`
             - Radius and sharpness of the interior component's radial cutoff.
           * - ``f_pole``, ``aspect``
             - :math:`f_{\rm pole}`, :math:`b`
             - Polar floor and angular thickness of the interior component.
           * - ``E_ej``, ``M_ej``
             - :math:`E_{\rm ej}`, :math:`M_{\rm ej}`
             - Ejecta kinetic energy and mass.
           * - ``n``, ``delta``
             - :math:`n`, :math:`\delta`
             - Outer and inner power-law indices of the ejecta density.
           * - ``epsilon_e``, ``epsilon_B``
             - :math:`\epsilon_e`, :math:`\epsilon_B`
             - Fractions of the post-shock thermal energy density in electrons and magnetic fields.
           * - ``p``
             - :math:`p`
             - Electron index below the cooling break (it steepens to :math:`p+1` above).
           * - ``gamma_min``, ``gamma_max``
             - :math:`\gamma_{\min}`, :math:`\gamma_{\max}`
             - Electron Lorentz factor limits.
           * - ``f_V``
             - :math:`f_V`
             - Volume filling factor shared by both sectors.
           * - ``T_e``
             - :math:`T_e`
             - Electron temperature of the ionized CSM.
           * - ``X``
             - :math:`X`
             - Hydrogen mass fraction of the CSM (the remainder is helium).
           * - ``R_ion``
             - :math:`R_{\rm ion}`
             - Outer radius of the ionized CSM (the free-free integration cutoff).

    .. dropdown:: Variables

        .. list-table::
           :widths: 20 20 60
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``frequency``
             - :math:`\nu`
             - Observer-frame frequency.
           * - ``time``
             - :math:`t`
             - Observer-frame time since explosion.

    Parameters
    ----------
    redshift : float, optional
        Source redshift. Exactly one of ``redshift`` and ``luminosity_distance`` must be given.
    luminosity_distance : float or ~astropy.units.Quantity, optional
        Source luminosity distance. Bare values are interpreted as Mpc.
    cosmology : ~astropy.cosmology.FLRW, optional
        Cosmology used to convert between redshift and distance. Defaults to the Trilobite configuration cosmology.
    r_ref : float or ~astropy.units.Quantity, optional
        Reference radius at which ``rho_int`` and ``rho_ext`` are defined. Bare values are cm. Default ``1e15 cm``.
    t_0 : float or ~astropy.units.Quantity, optional
        Source-frame time at which the shock integration starts (at :math:`R_0 = v_0 t_0`). Every requested epoch must
        satisfy :math:`t_{\rm obs}/(1+z) \ge t_0`. Bare values are seconds. Default ``0.3 d``. The shock solution
        forgets its start as :math:`t_0 \to 0`; ``0.3 d`` biases the thermal energy density by roughly 0.1% at 5 d.
    beta_max : float, optional
        Largest permitted initial velocity :math:`v_0/c`. The shock and radiation calculations are non-relativistic
        (the radiation uses a static plasma, :math:`\beta = 0`), so both degrade as :math:`v_0` approaches
        :math:`c`. Parameter sets with a larger :math:`v_0` are rejected as infeasible. Default ``0.8``.
    n_gamma_per_decade : int, optional
        Electron Lorentz-factor grid resolution. About 20 per decade converges the flux to ~0.01 dex. Default ``20``.
    ode_rtol : float, optional
        Relative tolerance of the shock integration. ``1e-6`` is several times faster than the engine's default
        ``1e-10`` with errors of order :math:`10^{-8}`. Default ``1e-6``.

    Notes
    -----
    - Serialization stores the redshift but not a custom ``cosmology`` object; a model rebuilt from its
      specification uses the configuration cosmology.
    - The emission measure of a hydrogen-helium plasma is
      :math:`\int n_e n_i\,dr = \frac{(1+X)(1+3X)}{8 m_p^2}\int\rho^2\,dr`, and the free-free opacity uses
      the corresponding hydrogen and helium ion fractions.
    - The cooling Lorentz factor is the pitch-angle averaged :math:`\gamma_c = 9\pi m_e c/(\sigma_T B^2 t)` evaluated
      at the source-frame time.

    References
    ----------
    Diggins et al. (2026, in preparation).
    """

    # =============================================== #
    # Parameter and Variable Declarations             #
    # =============================================== #
    PARAMETERS = (
        # ------------------ CSM ------------------ #
        ModelParameter(
            "rho_int",
            2e-16,
            base_units="g/cm**3",
            description="Solid-angle-averaged interior CSM density at the reference radius.",
            latex=r"\rho_{\rm int}",
            bounds=(0, None),
        ),
        ModelParameter(
            "rho_ext",
            1e-18,
            base_units="g/cm**3",
            description="Exterior (spherical) CSM density at the reference radius.",
            latex=r"\rho_{\rm ext}",
            bounds=(0, None),
        ),
        ModelParameter(
            "alpha_int",
            1.5,
            description="Power-law slope of the interior CSM component.",
            latex=r"\alpha_{\rm int}",
            bounds=(0, 3),
        ),
        ModelParameter(
            "alpha_ext",
            2.0,
            description="Power-law slope of the exterior CSM component.",
            latex=r"\alpha_{\rm ext}",
            bounds=(0, 3),
        ),
        ModelParameter(
            "R_trans",
            5e15,
            base_units="cm",
            description="Radius at which the interior CSM component has faded to half.",
            latex=r"R_{\rm trans}",
            bounds=(0, None),
        ),
        ModelParameter(
            "delta_trans",
            0.3,
            description="Dimensionless sharpness of the interior component's radial cutoff.",
            latex=r"\delta_{\rm trans}",
            bounds=(0, None),
        ),
        ModelParameter(
            "f_pole",
            0.2,
            description="Polar density floor of the interior CSM component relative to its equatorial peak.",
            latex=r"f_{\rm pole}",
            bounds=(0, 1),
        ),
        ModelParameter(
            "aspect",
            2.0,
            description="Angular thickness of the interior CSM component (larger is a thicker disk).",
            latex=r"b",
            bounds=(0, None),
        ),
        # ----------------- Ejecta ----------------- #
        ModelParameter(
            "E_ej",
            1e51,
            base_units="erg",
            description="Kinetic energy of the ejecta.",
            latex=r"E_{\rm ej}",
            bounds=(0, None),
        ),
        ModelParameter(
            "M_ej",
            0.1 * _M_SUN_CGS,
            base_units="g",
            description="Mass of the ejecta.",
            latex=r"M_{\rm ej}",
            bounds=(0, None),
        ),
        ModelParameter(
            "n",
            10.0,
            description="Power-law index of the outer ejecta density profile.",
            latex=r"n",
            bounds=(5, None),
        ),
        ModelParameter(
            "delta",
            0.0,
            description="Power-law index of the inner ejecta density profile.",
            latex=r"\delta",
            bounds=(0, 3),
        ),
        # ------------- Microphysics -------------- #
        ModelParameter(
            "epsilon_e",
            0.1,
            description="Fraction of the post-shock thermal energy density in relativistic electrons.",
            latex=r"\epsilon_e",
            bounds=(0, 1),
        ),
        ModelParameter(
            "epsilon_B",
            0.01,
            description="Fraction of the post-shock thermal energy density in the magnetic field.",
            latex=r"\epsilon_B",
            bounds=(0, 1),
        ),
        ModelParameter(
            "p",
            3.0,
            description="Electron power-law index below the cooling break.",
            latex=r"p",
            bounds=(2, None),
        ),
        ModelParameter(
            "gamma_min",
            1.0,
            description="Minimum electron Lorentz factor.",
            latex=r"\gamma_{\min}",
            bounds=(1, None),
        ),
        ModelParameter(
            "gamma_max",
            1e8,
            description="Maximum electron Lorentz factor.",
            latex=r"\gamma_{\max}",
            bounds=(1, None),
        ),
        ModelParameter(
            "f_V",
            0.5,
            description="Volume filling factor of the emitting region, shared by both sectors.",
            latex=r"f_V",
            bounds=(0, 1),
        ),
        # --------------- Free-free --------------- #
        ModelParameter(
            "T_e",
            1e6,
            base_units="K",
            description="Electron temperature of the ionized CSM.",
            latex=r"T_e",
            bounds=(0, None),
        ),
        ModelParameter(
            "X",
            0.7,
            description="Hydrogen mass fraction of the CSM (the remainder is helium).",
            latex=r"X",
            bounds=(0, 1),
        ),
        ModelParameter(
            "R_ion",
            1e20,
            base_units="cm",
            description="Outer radius of the ionized CSM, the free-free integration cutoff.",
            latex=r"R_{\rm ion}",
            bounds=(0, None),
        ),
    )
    VARIABLES = (
        ModelVariable(
            "frequency",
            description="Observer-frame frequency at which to evaluate the flux density.",
            base_units="Hz",
            latex=r"\nu",
        ),
        ModelVariable("time", description="Observer-frame time since explosion.", base_units="s", latex=r"t"),
    )

    # =============================================== #
    # Model Metadata Declarations                     #
    # =============================================== #
    OUTPUTS = namedtuple("Diggins26Outputs", ["flux_density"])
    UNITS = OUTPUTS(flux_density=u.Jy)
    DESCRIPTION = (
        "Two-sector (equatorial and polar) thin-shell shock interaction with an axisymmetric CSM, producing "
        "free-free attenuated synchrotron emission."
    )
    REFERENCE = "Diggins et al. (2026, in preparation)."

    # =============================================== #
    # Initialization Method                           #
    # =============================================== #
    def __init__(
        self,
        redshift: Optional[float] = None,
        luminosity_distance: Optional[Union[float, u.Quantity]] = None,
        cosmology=None,
        *,
        r_ref: Union[float, u.Quantity] = 1e15,
        t_0: Union[float, u.Quantity] = 0.3 * u.day,
        beta_max: float = 0.8,
        n_gamma_per_decade: int = 20,
        ode_rtol: float = 1e-6,
    ):
        super().__init__()

        if (redshift is None) == (luminosity_distance is None):
            raise ValueError("Exactly one of `redshift` and `luminosity_distance` must be provided.")

        self._cosmology = get_cosmology(cosmology)
        if luminosity_distance is not None:
            luminosity_distance = ensure_in_units(luminosity_distance, u.Mpc) * u.Mpc
        distances = resolve_cosmological_distances(
            redshift=redshift,
            luminosity_distance=luminosity_distance,
            cosmology=self._cosmology,
        )
        self._z = float(u.Quantity(distances["redshift"]).value)
        self._D_L = distances["luminosity_distance"].to(u.cm)
        self._D_A = distances["angular_diameter_distance"].to(u.cm)
        if not self._z >= 0:
            raise ValueError(f"The redshift must be non-negative, got {self._z}.")

        self._r_ref = float(ensure_in_units(r_ref, u.cm))
        self._t_0 = float(ensure_in_units(t_0, u.s))
        self._beta_max = float(beta_max)
        self._n_gamma_per_decade = int(n_gamma_per_decade)
        self._ode_rtol = float(ode_rtol)
        if self._r_ref <= 0:
            raise ValueError("`r_ref` must be positive.")
        if self._t_0 <= 0:
            raise ValueError("`t_0` must be positive.")
        if not 0 < self._beta_max < 1:
            raise ValueError("`beta_max` must lie in (0, 1).")
        if self._n_gamma_per_decade < 1:
            raise ValueError("`n_gamma_per_decade` must be a positive integer.")
        if not self._ode_rtol > 0:
            raise ValueError("`ode_rtol` must be positive.")

        # The engines are built once so that they are not re-instantiated on every call. The averaged
        # synchrotron kernel is loaded here for the same reason.
        self._shock_engine = PressureDrivenThinShellShockEngine(mu=0.6)
        self._synchrotron_engine = NumericalSynchrotronEngine()
        self._synchrotron_engine.load_avg_first_kernel()
        self._log_D_A = float(np.log(self._D_A.value))

        self._register_init(
            redshift=self._z,
            r_ref=self._r_ref,
            t_0=self._t_0,
            beta_max=self._beta_max,
            n_gamma_per_decade=self._n_gamma_per_decade,
            ode_rtol=self._ode_rtol,
        )

    # =============================================== #
    # Configuration properties                        #
    # =============================================== #
    @property
    def redshift(self) -> float:
        """float: Source redshift."""
        return self._z

    @property
    def luminosity_distance(self) -> u.Quantity:
        """~astropy.units.Quantity: Luminosity distance."""
        return self._D_L

    @property
    def angular_diameter_distance(self) -> u.Quantity:
        """~astropy.units.Quantity: Angular diameter distance."""
        return self._D_A

    @property
    def t_0(self) -> u.Quantity:
        """~astropy.units.Quantity: Source-frame start time of the shock integration."""
        return self._t_0 * u.s

    # =============================================== #
    # Preparation and Feasibility                     #
    # =============================================== #
    def _rest_times(self, t_obs: np.ndarray) -> np.ndarray:
        """Convert observer-frame times to the source frame and check them against ``t_0``."""
        t_rest = t_obs / (1.0 + self._z)
        if not np.all(np.isfinite(t_rest)):
            raise ValueError("All times must be finite.")
        if np.min(t_rest) < self._t_0 * (1.0 - 1e-12):
            raise ValueError(
                f"The model requires t_obs / (1 + z) >= t_0 = {self._t_0 / _DAY:.4g} d, "
                f"but the earliest epoch is {np.min(t_rest) / _DAY:.4g} d in the source frame. "
                "Lower `t_0` or drop the earlier epochs."
            )
        if np.max(t_rest) <= self._t_0 * (1.0 + 1e-12):
            raise ValueError("At least one epoch must lie beyond `t_0`.")
        return t_rest

    def _csm_parameters(self, p: dict) -> dict:
        """Map model parameters onto the CSM profile's parameters (without the sightline angle)."""
        return {
            "rho_int": p["rho_int"],
            "rho_ext": p["rho_ext"],
            "r_ref": self._r_ref,
            "alpha_int": p["alpha_int"],
            "alpha_ext": p["alpha_ext"],
            "r_trans": p["R_trans"],
            "delta_trans": p["delta_trans"],
            "f_pole": p["f_pole"],
            "aspect": p["aspect"],
        }

    def _prepare(self, p: dict):
        """
        Check the enumerated infeasibility conditions that precede the shock solve.

        Returns
        -------
        prep : dict or None
            Ejecta callable and starting conditions, or ``None`` if infeasible.
        reason : str or None
            Why the parameters are infeasible.
        """
        v_0 = float(np.sqrt(2.0 * p["E_ej"] / p["M_ej"]))
        if v_0 > self._beta_max * _C_CGS:
            return None, (f"v_0 = sqrt(2 E_ej / M_ej) = {v_0 / _C_CGS:.3f} c exceeds beta_max = {self._beta_max:g} c.")
        if not p["gamma_max"] > 1.0201 * p["gamma_min"]:
            return None, "gamma_max must exceed 1.0201 * gamma_min to place a cooling break."

        try:
            K, v_t = BrokenPowerLawEjectaProfile.normalize(
                E_ej=p["E_ej"], M_ej=p["M_ej"], n=p["n"], delta=p["delta"], v_max=_C_CGS
            )
        except ValueError as exc:
            return None, f"The ejecta profile cannot be normalized: {exc}"

        rho_ej = BrokenPowerLawEjectaProfile.as_optimized_callable(
            K=K.cgs.value, v_t=v_t.cgs.value, n=p["n"], delta=p["delta"], v_max=_C_CGS
        )
        return {"v_0": v_0, "R_0": v_0 * self._t_0, "rho_ej": rho_ej}, None

    def _sector_density(self, p: dict, theta: float):
        """Unit-free ``rho(r, t)`` callable along a fixed sightline."""
        return AxisymmetricCSMDensityProfile.as_optimized_callable(theta=theta, **self._csm_parameters(p))

    @staticmethod
    def _swept_mass(rho_csm, radius: float) -> float:
        r"""CSM mass inside ``radius`` along a spherical sightline, :math:`4\pi\int_0^R \rho r^2 dr`."""

        def integrand(r):
            return float(rho_csm(r, 0.0)) * r * r

        return 4.0 * np.pi * quad(integrand, 0.0, radius, epsabs=0.0, epsrel=1e-8, limit=200)[0]

    def _solve_dynamics(self, p: dict, prep: dict, t_rest: np.ndarray):
        """
        Solve the shock along both sightlines at the unique source-frame epochs.

        Returns
        -------
        states : tuple of _SectorState or None
            ``(equatorial, polar)``, or ``None`` if infeasible.
        reason : str or None
            Why the shock could not be solved.
        """
        states = []
        for theta in (_THETA_EQUATOR, _THETA_POLE):
            rho_csm = self._sector_density(p, theta)
            rho_1, u_1, rho_4, u_4 = make_homologous_stationary_sources(rho_ej=prep["rho_ej"], rho_csm=rho_csm)

            # The starting shell holds the CSM swept up inside R_0 along this sightline.
            M_0 = self._swept_mass(rho_csm, prep["R_0"])
            if not (np.isfinite(M_0) and M_0 > 0):
                return None, "There is no CSM mass inside the starting radius R_0 = v_0 t_0."

            try:
                shock = self._shock_engine._compute_shock_properties_cgs(
                    time=t_rest,
                    rho_1=rho_1,
                    rho_4=rho_4,
                    u_1=u_1,
                    u_4=u_4,
                    R_0=prep["R_0"],
                    v_0=prep["v_0"],
                    M_0=M_0,
                    t_0=self._t_0,
                    rtol=self._ode_rtol,
                )
            except RuntimeError as exc:
                return None, f"The shock ODE integration failed: {exc}"

            state = _SectorState(
                radius=np.asarray(shock.radius),
                velocity=np.asarray(shock.velocity),
                mass=np.asarray(shock.mass),
                eth=np.asarray(shock.thermal_energy_density),
            )
            if not all(np.all(np.isfinite(a)) for a in state) or np.any(state.radius <= 0):
                return None, "The shock solution is non-finite."
            states.append(state)
        return tuple(states), None

    def _dynamics_cgs(self, p: dict, t_rest: np.ndarray):
        prep, reason = self._prepare(p)
        if prep is None:
            return None, reason
        return self._solve_dynamics(p, prep, t_rest)

    # =============================================== #
    # Radiation                                       #
    # =============================================== #
    def _area_fractions(self, aspect: float) -> tuple[float, float]:
        eq = AxisymmetricCSMDensityProfile.equatorial_solid_angle_fraction(aspect)
        return eq, 1.0 - eq

    @staticmethod
    def _gamma_grid(p: dict, n_per_decade: int):
        n_gamma = max(int(np.ceil(n_per_decade * np.log10(p["gamma_max"] / p["gamma_min"]))) + 1, 16)
        gamma = np.geomspace(p["gamma_min"], p["gamma_max"], n_gamma)
        return gamma

    def _microphysics(self, p: dict, state: _SectorState, t_rest: np.ndarray, gamma: np.ndarray):
        """Magnetic field, cooling break, and electron distribution at every epoch."""
        B = np.sqrt(8.0 * np.pi * p["epsilon_B"] * state.eth)
        gamma_c = np.clip(
            _opt_compute_synchrotron_cooling_gamma(B, t_rest),
            1.01 * p["gamma_min"],
            p["gamma_max"] / 1.01,
        )
        N0 = np.empty_like(B)
        N_gamma = np.empty((B.size, gamma.size))
        for i in range(B.size):
            shape = {
                "p1": p["p"],
                "p2": p["p"] + 1.0,
                "gamma_min": p["gamma_min"],
                "gamma_max": p["gamma_max"],
                "gamma_c": gamma_c[i],
            }
            N0[i] = BrokenPowerLaw._normalize_from_magnetic_field(B[i], p["epsilon_B"], p["epsilon_e"], **shape)
            N_gamma[i] = BrokenPowerLaw.pdf(gamma, norm=N0[i], **shape)
        with np.errstate(divide="ignore"):
            log_N = np.log(N_gamma)
        return B, gamma_c, N0, log_N

    def _emission_measure(self, p: dict, theta: float, radius: np.ndarray) -> np.ndarray:
        r"""
        Emission measure :math:`\int_R^{R_{\rm ion}} n_e n_i\,dr` of the CSM exterior to the shock at each radius.

        The integral is evaluated in :math:`\ln r` by composite Gauss-Legendre quadrature whose panels are never wider
        than the CSM transition, with panel edges placed on the shock radii so that all epochs share one cumulative sum.
        """
        EM = np.zeros_like(radius)
        outside = radius < p["R_ion"]
        if not np.any(outside):
            return EM

        ln_R = np.log(radius[outside])
        ln_R_ion = np.log(p["R_ion"])
        width = min(_EM_MAX_PANEL, max(p["delta_trans"], _EM_MIN_PANEL))
        n_panels = max(int(np.ceil((ln_R_ion - ln_R.min()) / width)), 1)
        edges = np.unique(np.concatenate([np.linspace(ln_R.min(), ln_R_ion, n_panels + 1), ln_R]))

        mid = 0.5 * (edges[1:] + edges[:-1])
        half = 0.5 * (edges[1:] - edges[:-1])
        r = np.exp(mid[:, None] + half[:, None] * _GL_NODES[None, :])
        rho = self._sector_density(p, theta)(r, 0.0)
        panel = half * np.sum(rho**2 * r * _GL_WEIGHTS[None, :], axis=1)
        tail = np.concatenate([np.cumsum(panel[::-1])[::-1], [0.0]])  # integral from each edge outward

        X = p["X"]
        composition = (1.0 + X) * (1.0 + 3.0 * X) / 8.0
        EM[outside] = composition * tail[np.searchsorted(edges, ln_R)] / _M_P_CGS**2
        return EM

    def _ion_composition(self, X: float):
        """Charges and number fractions of the ions of a fully ionized hydrogen-helium plasma."""
        return np.array([1.0, 2.0]), np.array([4.0 * X, 1.0 - X]) / (1.0 + 3.0 * X)

    def _sector_spectra(
        self,
        p: dict,
        state: _SectorState,
        micro: tuple,
        EM: np.ndarray,
        f_A: float,
        gamma: np.ndarray,
        nu_groups: list,
    ):
        """
        Intrinsic flux density and free-free optical depth of one sector.

        ``nu_groups[i]`` holds the observer-frame frequencies to evaluate at epoch ``i``. Returns lists of arrays
        (one per epoch) in CGS flux units and dimensionless optical depth.
        """
        B, _, _, log_N = micro
        log_gamma, log_weights = self._synchrotron_engine._build_gamma_grid(gamma)
        Z, X_ions = self._ion_composition(p["X"])
        log_T = np.log(p["T_e"])
        ell = p["f_V"] ** (1.0 / 3.0) * state.radius
        area = np.pi * state.radius**2 * f_A

        # The free-free optical depth is exactly linear in the emission measure, but the routine evaluates its Gaunt
        # factor at the mean of the frequencies it is given. Evaluating tau per unit EM one frequency at a time makes
        # each result independent of which other frequencies are requested.
        nu_unique, nu_inverse = np.unique(np.concatenate(nu_groups), return_inverse=True)
        tau_per_EM = np.array(
            [
                _compute_RJ_ff_optical_depth_EM(np.log([nu_j * (1.0 + self._z)]), 0.0, log_T=log_T, Z=Z, X=X_ions)[0]
                for nu_j in nu_unique
            ]
        )[nu_inverse]
        tau_per_EM = np.split(tau_per_EM, np.cumsum([nu.size for nu in nu_groups])[:-1])

        flux, tau = [], []
        for i, nu in enumerate(nu_groups):
            with np.errstate(divide="ignore"):
                log_F = self._synchrotron_engine._compute_log_pa_flux_density(
                    np.log(nu),
                    np.log(ell[i]),
                    np.log(B[i]),
                    log_N[i],
                    log_gamma,
                    log_weights,
                    np.log(area[i]),
                    self._log_D_A,
                    z=self._z,
                )
            flux.append(np.exp(log_F))
            tau.append(EM[i] * tau_per_EM[i])
        return flux, tau

    def _total_flux(self, p: dict, states: tuple, t_rest: np.ndarray, nu_groups: list):
        """Total attenuated flux density (CGS) per epoch group, summed over the two sectors."""
        gamma = self._gamma_grid(p, self._n_gamma_per_decade)
        f_As = self._area_fractions(p["aspect"])
        total = [np.zeros_like(nu) for nu in nu_groups]
        for theta, state, f_A in zip((_THETA_EQUATOR, _THETA_POLE), states, f_As):
            micro = self._microphysics(p, state, t_rest, gamma)
            EM = self._emission_measure(p, theta, state.radius)
            flux, tau = self._sector_spectra(p, state, micro, EM, f_A, gamma, nu_groups)
            for i in range(len(nu_groups)):
                total[i] += flux[i] * np.exp(-tau[i])
        return total

    # =============================================== #
    # Core Model Evaluation Method                    #
    # =============================================== #
    def _forward_model(self, variables, parameters):
        t_obs, nu_obs = np.broadcast_arrays(
            np.asarray(variables["time"], dtype=float), np.asarray(variables["frequency"], dtype=float)
        )
        shape = t_obs.shape
        t_flat, nu_flat = t_obs.ravel(), nu_obs.ravel()

        t_unique, inverse = np.unique(t_flat, return_inverse=True)
        t_rest = self._rest_times(t_unique)

        states, _ = self._dynamics_cgs(parameters, t_rest)
        if states is None:
            return self.OUTPUTS(flux_density=np.full(shape, np.nan))

        # Group the paired frequencies by epoch so that each epoch is evaluated only at its own frequencies.
        order = np.argsort(inverse, kind="stable")
        splits = np.cumsum(np.bincount(inverse, minlength=t_unique.size))[:-1]
        nu_groups = np.split(nu_flat[order], splits)

        total = self._total_flux(parameters, states, t_rest, nu_groups)

        flux = np.empty_like(nu_flat)
        flux[order] = np.concatenate(total)
        return self.OUTPUTS(flux_density=(1e23 * flux).reshape(shape))

    # =============================================== #
    # Analysis Methods                                #
    # =============================================== #
    def _coerce(self, parameters):
        """Coerce user parameters to raw base units and enforce hard bounds."""
        p = self.coerce_model_parameters({} if parameters is None else parameters)
        self._check_parameter_bounds_base(p)
        return {k: float(v) for k, v in p.items()}

    def _coerce_axes(self, time, frequency):
        v = self.coerce_model_variables({"time": time, "frequency": frequency})
        return np.atleast_1d(np.asarray(v["time"], dtype=float)), np.atleast_1d(np.asarray(v["frequency"], dtype=float))

    def diagnose(self, parameters=None, time=None) -> Optional[str]:
        """
        Explain why a parameter set is rejected as infeasible.

        Parameters
        ----------
        parameters : dict, optional
            Model parameters (missing entries take their defaults).
        time : float, array-like, or ~astropy.units.Quantity, optional
            Observer-frame epochs. Needed to detect shock integration failures; if omitted, only the conditions
            that do not require integrating the shock are checked.

        Returns
        -------
        str or None
            A description of the infeasibility, or ``None`` if the parameters are feasible.
        """
        p = self._coerce(parameters)
        prep, reason = self._prepare(p)
        if prep is None or time is None:
            return reason
        t_obs, _ = self._coerce_axes(time, 1.0)
        _, reason = self._solve_dynamics(p, prep, self._rest_times(np.unique(t_obs)))
        return reason

    def compute_dynamics(self, parameters=None, time=None) -> Diggins26Dynamics:
        """
        Solve the shock dynamics along both sightlines.

        The result can be passed to :meth:`tabulate` to evaluate many frequencies without re-solving the shock.

        Parameters
        ----------
        parameters : dict, optional
            Model parameters (missing entries take their defaults).
        time : array-like or ~astropy.units.Quantity
            Observer-frame epochs. Bare values are seconds.

        Returns
        -------
        Diggins26Dynamics
            The dynamics at the unique, sorted epochs.

        Raises
        ------
        ValueError
            If the parameters are infeasible (see :meth:`diagnose`) or an epoch precedes ``t_0``.
        """
        if time is None:
            raise ValueError("`time` is required.")
        p = self._coerce(parameters)
        t_obs, _ = self._coerce_axes(time, 1.0)
        t_unique = np.unique(t_obs)
        t_rest = self._rest_times(t_unique)

        states, reason = self._dynamics_cgs(p, t_rest)
        if states is None:
            raise ValueError(f"The parameters are infeasible: {reason}")

        def _wrap(s: _SectorState) -> Diggins26SectorDynamics:
            return Diggins26SectorDynamics(
                radius=s.radius * u.cm,
                velocity=s.velocity * (u.cm / u.s),
                mass=s.mass * u.g,
                thermal_energy_density=s.eth * (u.erg / u.cm**3),
            )

        return Diggins26Dynamics(
            time=t_unique * u.s,
            time_rest=t_rest * u.s,
            equatorial=_wrap(states[0]),
            polar=_wrap(states[1]),
            parameters={k: p[k] for k in _DYNAMICS_PARAMETERS},
        )

    def tabulate(self, time, frequency, parameters=None, *, dynamics: Optional[Diggins26Dynamics] = None):
        """
        Evaluate the model on independent time and frequency axes, keeping all intermediate quantities.

        This is the analysis and plotting counterpart of :meth:`forward_model`, which instead pairs each time
        with its own frequency.

        Parameters
        ----------
        time : array-like or ~astropy.units.Quantity
            Observer-frame epochs, shape ``(n_t,)``. Bare values are seconds.
        frequency : array-like or ~astropy.units.Quantity
            Observer-frame frequencies, shape ``(n_nu,)``. Bare values are Hz.
        parameters : dict, optional
            Model parameters (missing entries take their defaults).
        dynamics : Diggins26Dynamics, optional
            A previous result of :meth:`compute_dynamics` computed with the same dynamics parameters at the same
            epochs, to avoid re-solving the shock.

        Returns
        -------
        Diggins26Tabulation
            The total and per-sector flux densities and diagnostics on the ``(n_t, n_nu)`` grid.

        Raises
        ------
        ValueError
            If the parameters are infeasible, an epoch precedes ``t_0``, or ``dynamics`` does not match.
        """
        p = self._coerce(parameters)
        t_obs, nu_obs = self._coerce_axes(time, frequency)
        t_rest = self._rest_times(t_obs)
        if np.any(np.diff(t_obs) <= 0):
            raise ValueError("`time` must be strictly increasing.")

        if dynamics is None:
            dynamics = self.compute_dynamics(p, t_obs)
        else:
            if dynamics.time.shape != t_obs.shape or not np.allclose(dynamics.time.to_value(u.s), t_obs, rtol=1e-12):
                raise ValueError("`dynamics` was computed at different epochs than `time`.")
            for key in _DYNAMICS_PARAMETERS:
                if not np.isclose(dynamics.parameters[key], p[key], rtol=1e-12, atol=0.0):
                    raise ValueError(f"`dynamics` was computed with a different `{key}`.")

        states = tuple(
            _SectorState(
                radius=d.radius.to_value(u.cm),
                velocity=d.velocity.to_value(u.cm / u.s),
                mass=d.mass.to_value(u.g),
                eth=d.thermal_energy_density.to_value(u.erg / u.cm**3),
            )
            for d in (dynamics.equatorial, dynamics.polar)
        )

        gamma = self._gamma_grid(p, self._n_gamma_per_decade)
        nu_groups = [nu_obs] * t_obs.size
        results, total = [], 0.0
        for theta, state, f_A in zip((_THETA_EQUATOR, _THETA_POLE), states, self._area_fractions(p["aspect"])):
            micro = self._microphysics(p, state, t_rest, gamma)
            EM = self._emission_measure(p, theta, state.radius)
            flux, tau = self._sector_spectra(p, state, micro, EM, f_A, gamma, nu_groups)
            flux, tau = np.array(flux), np.array(tau)
            attenuated = flux * np.exp(-tau)
            total = total + attenuated
            B, gamma_c, N0, _ = micro
            results.append(
                Diggins26SectorRadiation(
                    flux_density=(1e23 * flux) * u.Jy,
                    flux_density_attenuated=(1e23 * attenuated) * u.Jy,
                    tau_ff=tau,
                    emission_measure=EM * u.cm**-5,
                    magnetic_field=B * u.G,
                    gamma_c=gamma_c,
                    electron_normalization=N0 * u.cm**-3,
                    area_fraction=f_A,
                )
            )

        return Diggins26Tabulation(
            time=t_obs * u.s,
            frequency=nu_obs * u.Hz,
            flux_density=(1e23 * total) * u.Jy,
            equatorial=results[0],
            polar=results[1],
            dynamics=dynamics,
        )
