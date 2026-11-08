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
4. The emitting geometry of each sector (sky-projected area, volume and mean path length) follows from the CSM
   opening angle, the observer's inclination and a shell volume filling factor
   (:meth:`~trilobite.dynamics.profiles.csm.AxisymmetricCSMDensityProfile.compute_filling_parameters`).
5. Free-free absorption by the CSM exterior to the shock attenuates each sector, integrated along the actual line of
   sight from the sector to the observer.
6. The observed flux density is the sum of the attenuated sectors.
"""

import warnings
from collections import namedtuple
from collections.abc import Callable
from typing import NamedTuple, Optional, Union

import numpy as np
from astropy import constants as const
from astropy import units as u
from scipy.integrate import IntegrationWarning, quad

from trilobite.dynamics.profiles.csm import AxisymmetricCSMDensityProfile
from trilobite.dynamics.profiles.ejecta import BrokenPowerLawEjectaProfile, _normalize_bpl_cgs
from trilobite.dynamics.shocks.numerical import PressureDrivenThinShellShockEngine
from trilobite.dynamics.shocks.utils import make_homologous_stationary_sources
from trilobite.models.core.base import Model
from trilobite.models.core.parameters import ModelParameter, ModelVariable
from trilobite.physics_utils.composition import (
    compute_mean_molecular_weight,
    compute_mean_molecular_weight_per_electron,
    compute_mean_molecular_weight_per_ion,
)
from trilobite.physics_utils.cosmology import get_cosmology, resolve_cosmological_distances
from trilobite.radiation.free_free.absorption import _compute_RJ_ff_optical_depth_EM
from trilobite.radiation.synchrotron.cooling import (
    _opt_compute_IC_cooling_gamma,
    _opt_compute_synchrotron_cooling_gamma,
)
from trilobite.radiation.synchrotron.electron_distributions import BrokenPowerLaw
from trilobite.radiation.synchrotron.SEDs.numerical.core import NumericalSynchrotronEngine
from trilobite.utils.misc_utils import ensure_in_units

from ..core.status import ModelStatus

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

# Panels of the line-of-sight emission-measure quadrature are ``_EM_PANEL_PER_DELTA * delta_trans`` wide in the
# quadrature variable, but never wider than ``_EM_MAX_PANEL`` or narrower than ``_EM_MIN_PANEL``. With 6-point
# Gauss-Legendre panels this reproduces a converged integral to about 1e-4, far below the model's other errors.
_EM_MAX_PANEL = 0.6
_EM_MIN_PANEL = 0.02
_EM_PANEL_PER_DELTA = 1.5
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(6)

# The line-of-sight emission-measure quadrature runs in x = ln(q / R + _LOS_EPSILON), where q is the path length
# beyond the shock. The offset lets the grid start exactly at q = 0 while resolving the near-shock decade.
_LOS_EPSILON = 1e-2
# Cap on the number of quadrature nodes evaluated at once, which bounds the size of the work arrays.
_LOS_MAX_NODES = 2_000_000

# Grid of the sky-projected area-fraction table: (aspect, inclination [rad], volume filling factor). The aspect range
# keeps both sectors wider than about 4 degrees; outside the grid the table is clamped to its edge values.
_FILLING_GRID = (np.geomspace(0.1, 20.0, 20), np.linspace(0.0, 0.5 * np.pi, 16), np.linspace(0.02, 0.98, 7))
_FILLING_N_PIX = 200
# Tables already built in this process, shared between model instances.
_FILLING_TABLE_CACHE: dict = {}

# Maximum number of (epoch, frequency) pairs sent to the synchrotron engine at once; this bounds the size of its
# (pairs x Lorentz-factor) work arrays on large tabulation grids.
_SPECTRUM_CHUNK = 2048

# Keyword arguments of the shock integration that the model sets itself and which `integrator_kw` may not override.
_RESERVED_INTEGRATOR_KW = frozenset({"fun", "t_span", "y0", "t_eval", "dense_output", "gamma"})

# The parameters on which the shock dynamics depend. (The composition and adiabatic index, which also enter the
# dynamics, are fixed hyperparameters rather than parameters.)
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

    Returned by :meth:`~trilobite.models.LFBOTs.diggins26.Diggins26_TwoComponentModel.compute_dynamics`, and
    accepted by :meth:`~trilobite.models.LFBOTs.diggins26.Diggins26_TwoComponentModel.tabulate` so that many
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
    """
    Effective free-free optical depth, :math:`-\\ln\\langle e^{-\\tau}\\rangle` with the average taken over the
    sector's sightlines, evaluated at the source-frame frequency.
    """
    emission_measure: u.Quantity
    """
    Emission measure along the line of sight through the CSM exterior to the shock, :math:`\\int n_e n_i\\,ds`,
    averaged over the sector's sightlines, shape ``(n_t,)``.
    """
    magnetic_field: u.Quantity
    """Post-shock magnetic field, shape ``(n_t,)``."""
    gamma_c: np.ndarray
    """Cooling Lorentz factor (after clipping to the electron-distribution support), shape ``(n_t,)``."""
    electron_normalization: u.Quantity
    """Electron distribution amplitude :math:`N_0`, shape ``(n_t,)``."""
    area_fraction: float
    """The sector's sky-projected area fraction :math:`f_A`, with :math:`A_\\perp = f_A \\pi R^2`."""
    volume_fraction: float
    """The sector's volume fraction :math:`f_V`, with :math:`V = f_V\\,4\\pi R^3/3`."""
    path_length_ratio: float
    """The sector's mean path length through the emitting shell in units of its radius, :math:`\\ell / R`."""


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

    Ejecta interact with an axisymmetric CSM (a dense, disk-confined interior component plus a spherical exterior
    component; see :class:`~trilobite.dynamics.profiles.csm.AxisymmetricCSMDensityProfile`). The interaction is
    followed along two sightlines, the equator and the pole, each standing in for its solid-angle sector. Along each
    sightline a pressure-driven thin shell sets the post-shock thermal energy density, which is partitioned into a
    magnetic field and a cooled broken power-law electron population that radiates synchrotron emission. Free-free
    absorption by the ionized CSM exterior to the shock attenuates each sector along its line of sight to the
    observer, and the observed flux density is the sum of the sectors. The observer's inclination to the symmetry axis
    sets the sectors' projected areas and the lines of sight. The technical details are given in the Notes below.

    .. important::

        The first evaluation builds a table of the sectors' sky-projected area fractions (a few seconds, with a
        progress bar), which is then cached on the model and shared between instances. Call
        :meth:`build_filling_table` beforehand when evaluating in parallel so that workers do not each repeat it.
        Keep ``aspect`` within 0.1 to 20, and ``f_V`` within 0.02 to 0.98: outside the table's range the geometry is
        held at its edge values.

    .. dropdown:: Parameters

        Parameters are varied at evaluation time (and are therefore what an inference samples).

        .. list-table::
           :widths: 20 20 60
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``rho_int``
             - :math:`\rho_{\rm int}`
             - Solid-angle-averaged interior CSM density at ``reference_radius``.
           * - ``rho_ext``
             - :math:`\rho_{\rm ext}`
             - Exterior (spherical) CSM density at ``reference_radius``.
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
           * - ``inclination``
             - :math:`i`
             - Angle between the CSM symmetry axis and the line of sight (0 is pole-on, :math:`\pi/2` is edge-on).
           * - ``f_V``
             - :math:`f`
             - Volume filling factor of the emitting shell, shared by both sectors.
           * - ``T_e``
             - :math:`T_e`
             - Electron temperature of the ionized CSM.
           * - ``R_ion``
             - :math:`R_{\rm ion}`
             - Outer radius of the ionized CSM (the free-free integration cutoff).

    .. dropdown:: Hyperparameters

        Hyperparameters are fixed when the model is constructed and are never sampled.

        .. list-table::
           :widths: 25 15 60
           :header-rows: 1

           * - **Name**
             - **Default**
             - **Description**
           * - ``redshift``, ``luminosity_distance``, ``angular_diameter_distance``
             - (one required)
             - Source distance. Exactly one is given and the others follow from ``cosmology``.
           * - ``cosmology``
             - configuration
             - Cosmology relating redshift and distance.
           * - ``hydrogen_mass_fraction``, ``helium_mass_fraction``
             - ``0.7``, ``1 - X``
             - CSM mass fractions :math:`X` and :math:`Y`. They set the mean molecular weights (and therefore the
               shock temperature and free-free emission measure) and the ion mix of the free-free opacity.
           * - ``adiabatic_index``
             - ``5/3``
             - Adiabatic index of the shocked gas.
           * - ``reference_radius``
             - ``1e15 cm``
             - Radius at which ``rho_int`` and ``rho_ext`` are defined.
           * - ``shock_start_time``
             - ``0.3 d``
             - Source-frame time :math:`t_0` at which the shock integration starts.
           * - ``max_initial_beta``
             - ``0.8``
             - Largest permitted initial velocity :math:`v_0/c`; faster ejecta are rejected as infeasible.
           * - ``gamma_points_per_decade``
             - ``20``
             - Resolution of the electron Lorentz-factor grid.
           * - ``n_azimuth``
             - ``4``
             - Number of azimuths at which the equatorial sector's line of sight is sampled when averaging the
               free-free transmission (see the Notes).
           * - ``integrator_kw``
             - ``{"method": "LSODA", "rtol": 1e-7}``
             - Keyword arguments for the shock integration (:func:`scipy.integrate.solve_ivp`), such as ``method``,
               ``rtol``, ``atol`` or ``max_step``.
           * - ``seed_photon_luminosity``
             - ``None``
             - Optional :math:`L(t)` of the Compton seed photons. If given, inverse-Compton losses are added to
               the synchrotron cooling.

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
        Source redshift. Exactly one of ``redshift``, ``luminosity_distance`` and ``angular_diameter_distance`` must
        be given.
    luminosity_distance : float or ~astropy.units.Quantity, optional
        Source luminosity distance. Bare values are interpreted as Mpc.
    angular_diameter_distance : float or ~astropy.units.Quantity, optional
        Source angular diameter distance. Bare values are interpreted as Mpc.
    cosmology : ~astropy.cosmology.FLRW, optional
        Cosmology used to convert between redshift and distance. Defaults to the Trilobite configuration cosmology.
    hydrogen_mass_fraction : float, optional
        Hydrogen mass fraction of the CSM. Default ``0.7``.
    helium_mass_fraction : float, optional
        Helium mass fraction of the CSM. Defaults to ``1 - hydrogen_mass_fraction`` (no metals).
    adiabatic_index : float, optional
        Adiabatic index of the shocked gas. Default ``5/3``.
    reference_radius : float or ~astropy.units.Quantity, optional
        Radius at which ``rho_int`` and ``rho_ext`` are defined. Bare values are cm. Default ``1e15 cm``.
    inner_radius : float or ~astropy.units.Quantity, optional
        Inner radius of the CSM, below which both components have zero density. Bare values are cm. Default ``0``.
    shock_start_time : float or ~astropy.units.Quantity, optional
        Source-frame time at which the shock integration starts. Bare values are seconds. Default ``0.3 d``.
    max_initial_beta : float, optional
        Largest permitted initial velocity :math:`v_0/c`; parameter sets above it are rejected as infeasible.
        Default ``0.8``.
    gamma_points_per_decade : int, optional
        Electron Lorentz-factor grid resolution. Default ``20``.
    n_azimuth : int, optional
        Number of azimuths sampled for the equatorial free-free transmission. Default ``4``.
    integrator_kw : dict, optional
        Keyword arguments forwarded to :func:`scipy.integrate.solve_ivp` for the shock integration.
    seed_photon_luminosity : callable, optional
        Luminosity :math:`L(t)` of the radiation field that Compton-scatters off the electrons. If ``None``
        (default), only synchrotron cooling is included.

    Notes
    -----
    .. rubric:: Geometry and the shock

    The CSM is axisymmetric,

    .. math::

        \rho(r, \theta) = \rho_{\rm ext}(r) + \rho_{\rm int}(r)\,S(r)\,G(\theta),

    and the interaction of freely expanding, broken power-law ejecta with it is followed along the equator
    (:math:`\theta = \pi/2`) and the pole (:math:`\theta = 0`). Along each, a pressure-driven thin shell is
    integrated from :math:`t_0` (see
    :class:`~trilobite.dynamics.shocks.numerical.PressureDrivenThinShellShockEngine`). The starting shell holds the
    CSM swept up inside :math:`R_0 = v_0 t_0`, with :math:`v_0 = \sqrt{2E_{\rm ej}/M_{\rm ej}}`.

    .. rubric:: Electrons and radiation

    The post-shock thermal energy density :math:`e_{\rm th}` sets the magnetic field
    :math:`B = \sqrt{8\pi\epsilon_B e_{\rm th}}` and the amplitude of a synchrotron-cooled broken power-law electron
    distribution (:math:`N(\gamma) \propto \gamma^{-p}` below the cooling break and :math:`\gamma^{-(p+1)}` above)
    carrying a fraction :math:`\epsilon_e` of :math:`e_{\rm th}`. The spectrum of each sector comes from the
    numerical synchrotron engine (pitch-angle averaged, including self-absorption) with geometry

    .. math::

        A_{\rm eff} = \pi R^2 f_A, \qquad \ell_{\rm eff} = \frac{4}{3}\frac{f_{V,k}}{f_A} R.

    The two sectors are split at the disk half-maximum latitude :math:`\arctan(b\ln 2)`. Each is a shell of
    fractional thickness :math:`\eta = 1 - (1-f)^{1/3}` set by the shared filling factor :math:`f` (``f_V``), so that
    sector :math:`k` holds a fraction :math:`f_{V,k} = f\,\Omega_k/4\pi` of the full-sphere volume. The sky-projected
    area fraction :math:`f_A` of each sector depends on the aspect :math:`b`, the inclination :math:`i` and
    :math:`f`; it is interpolated from a table built with
    :meth:`~trilobite.dynamics.profiles.csm.AxisymmetricCSMDensityProfile.build_filling_parameter_table`, and
    :math:`\ell_{\rm eff}` is the mean path length :math:`V/A_\perp`. None of these are free parameters apart from
    :math:`f`, :math:`b` and :math:`i`. The shock dynamics do not depend on the inclination.

    .. rubric:: Cooling

    The cooling Lorentz factor :math:`\gamma_c` is the one whose cooling time equals the source-frame age
    :math:`t`. For synchrotron losses alone (pitch-angle averaged),

    .. math::

        \gamma_{c,\rm syn} = \frac{9\pi m_e c}{\sigma_T B^2 t}.

    If ``seed_photon_luminosity`` is given, inverse-Compton losses on a point source of luminosity :math:`L(t)` at
    the center, seen by the shell at radius :math:`R`, are added in the Thomson regime without Klein-Nishina
    corrections,

    .. math::

        \gamma_{c,\rm IC} = \frac{3\pi m_e c^2 R^2}{\sigma_T L\, t}.

    The loss rates add, so the cooling times (and hence the inverse Lorentz factors) add:
    :math:`1/\gamma_c = 1/\gamma_{c,\rm syn} + 1/\gamma_{c,\rm IC}`. In either case :math:`\gamma_c` is clipped
    to lie strictly inside :math:`(\gamma_{\min}, \gamma_{\max})`. The same :math:`L(t)` is used for both sectors,
    and it only changes the electron distribution, not the shock dynamics.

    .. rubric:: Free-free absorption

    The CSM exterior to the shock (a fully ionized plasma at temperature :math:`T_e`) attenuates each sector. The
    emission measure along a line of sight leaving the shock at radius :math:`R` is

    .. math::

        \int n_e n_i\,ds = \frac{1}{\mu_e \mu_i m_p^2}\int\rho^2(r(s), \theta(s))\,ds,

    integrated along the straight path :math:`s` toward the observer, through the axisymmetric density, from where
    the path leaves the shock sphere to :math:`R_{\rm ion}`. A sector is not a single line of sight, so the model
    averages the transmission over a few representative ones,

    .. math::

        T_{\rm ff}(\nu) = \left\langle e^{-\tau_{\nu,j}} \right\rangle_j,

    and not the optical depth. A sector that is transparent along some sightlines and opaque along others therefore
    transmits the flux of the transparent ones. The equatorial sector is a ring and is sampled at ``n_azimuth``
    azimuths; the polar sector is sampled along its near and far lobes. Both are weighted equally. The path
    interior to the shock is not included: every sightline is treated as passing through the same CSM exterior to
    the sector's own shock radius.

    where the mean molecular weights per electron and per ion :math:`\mu_e` and :math:`\mu_i` follow from the
    composition (see :mod:`~trilobite.physics_utils.composition`); the same composition sets the mean molecular
    weight :math:`\mu` that converts the post-shock pressure to a temperature. The free-free opacity is evaluated
    with the hydrogen and helium ion fractions, :math:`n_{\rm H} : n_{\rm He} = X : Y/4`. A metal fraction
    :math:`1 - X - Y` enters :math:`\mu`, :math:`\mu_e` and :math:`\mu_i` but is not included in the ion mix of the
    opacity, which is accurate when the metals are a small contribution.

    .. rubric:: Redshift

    Redshift and distance are fixed model configuration (constructor arguments), never parameters. The model takes
    **observer-frame** ``time`` and ``frequency``. Internally, the dynamics are solved at
    :math:`t = t_{\rm obs}/(1+z)`, the free-free optical depth is evaluated at :math:`\nu = \nu_{\rm obs}(1+z)`, and
    the synchrotron engine applies the cosmological flux transformation using :math:`D_A`. The seed photon
    luminosity is a function of the **source-frame** time.

    .. rubric:: Evaluation paths

    :meth:`forward_model`
        The inference path. ``time`` and ``frequency`` are paired (one flux per pair); the shock is solved once per
        sector at the unique rest-frame epochs and each epoch's spectrum is evaluated only at its own frequencies.

    :meth:`tabulate`
        The analysis and plotting path. ``time`` and ``frequency`` are independent axes, and the result is a
        :class:`~trilobite.models.LFBOTs.diggins26.Diggins26Tabulation` containing the total and the per-sector
        flux densities, optical depths, magnetic fields, and so on, on an ``(n_t, n_nu)`` grid. The shock can be solved
        once with :meth:`compute_dynamics` and reused.

    .. rubric:: Infeasible parameters

    Instead of raising exceptions, :meth:`forward_model` returns NaN flux for a short, enumerated list of
    infeasible parameter combinations (so that samplers can reject them), and lets every other exception
    propagate:

    - the ejecta profile cannot be normalized to :math:`(E_{\rm ej}, M_{\rm ej})` with :math:`v_{\max} = c`,
    - :math:`v_0 = \sqrt{2E_{\rm ej}/M_{\rm ej}}` exceeds ``max_initial_beta`` times :math:`c`,
    - :math:`\gamma_{\max}` is too close to :math:`\gamma_{\min}` to place a cooling break,
    - there is no finite CSM mass inside the starting radius (including a slope :math:`\alpha \ge 3` with no
      ``inner_radius``), or
    - the shock ODE integration fails.

    :meth:`diagnose` reports which condition rejected a parameter set. Hard parameter-bound violations raise
    ``ValueError`` as for every Trilobite model.

    .. rubric:: Serialization

    Serialization stores the resolved redshift and the hyperparameters, but not a custom ``cosmology`` object or the
    ``seed_photon_luminosity`` callable; a model rebuilt from its specification uses the configuration cosmology and
    has no Compton cooling unless the callable is supplied again.

    References
    ----------
    Diggins et al. (2026, in preparation).
    """

    STATUS = ModelStatus.UNVALIDATED

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
            bounds=(0, None),
        ),
        ModelParameter(
            "alpha_ext",
            2.0,
            description="Power-law slope of the exterior CSM component.",
            latex=r"\alpha_{\rm ext}",
            bounds=(0, None),
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
            "inclination",
            np.pi / 3,
            base_units="rad",
            description="Angle between the CSM symmetry axis and the line of sight (0 is pole-on).",
            latex=r"i",
            bounds=(0, np.pi / 2),
        ),
        ModelParameter(
            "f_V",
            0.5,
            description="Volume filling factor of the emitting shell, shared by both sectors.",
            latex=r"f",
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
        angular_diameter_distance: Optional[Union[float, u.Quantity]] = None,
        cosmology=None,
        *,
        hydrogen_mass_fraction: float = 0.7,
        helium_mass_fraction: Optional[float] = None,
        adiabatic_index: float = 5.0 / 3.0,
        reference_radius: Union[float, u.Quantity] = 1e15,
        inner_radius: Union[float, u.Quantity] = 0.0,
        shock_start_time: Union[float, u.Quantity] = 0.3 * u.day,
        max_initial_beta: float = 0.8,
        gamma_points_per_decade: int = 20,
        n_azimuth: int = 4,
        integrator_kw: Optional[dict] = None,
        seed_photon_luminosity: Optional[Callable] = None,
    ):
        r"""
        Initialize the model.

        Parameters
        ----------
        redshift : float, optional
            Source redshift. Exactly one of ``redshift``, ``luminosity_distance`` and ``angular_diameter_distance`` must
            be given.
        luminosity_distance : float or ~astropy.units.Quantity, optional
            Source luminosity distance :math:`D_L`. Bare values are interpreted as Mpc.
        angular_diameter_distance : float or ~astropy.units.Quantity, optional
            Source angular diameter distance :math:`D_A`. Bare values are interpreted as Mpc.
        cosmology : ~astropy.cosmology.FLRW, optional
            Cosmology used to convert between redshift and distance. Defaults to the Trilobite configuration cosmology.
        hydrogen_mass_fraction : float, optional
            Hydrogen mass fraction :math:`X` of the CSM. Default ``0.7``.
        helium_mass_fraction : float, optional
            Helium mass fraction :math:`Y` of the CSM. Defaults to ``1 - hydrogen_mass_fraction`` (no metals). Any
            remainder :math:`1 - X - Y` is treated as metals; see the class Notes.
        adiabatic_index : float, optional
            Adiabatic index of the shocked gas. Default ``5/3``.
        reference_radius : float or ~astropy.units.Quantity, optional
            Reference radius at which ``rho_int`` and ``rho_ext`` are defined. Bare values are cm. Default ``1e15 cm``.
        inner_radius : float or ~astropy.units.Quantity, optional
            Inner radius :math:`r_{\rm min}` of the CSM, passed to the density profile as ``r_min``: both CSM
            components have zero density below it. It sets the lower limit of the CSM mass swept up inside
            :math:`R_0`. With the default ``0`` the CSM extends to the origin, and that mass diverges for slopes
            :math:`\alpha \ge 3`, so ``inner_radius`` must be positive if the model is used with such slopes. It must
            be smaller than :math:`R_0 = v_0 t_0` for a parameter set to be feasible. Bare values are cm. Default ``0``.
        shock_start_time : float or ~astropy.units.Quantity, optional
            Source-frame time at which the shock integration starts (at :math:`R_0 = v_0 t_0`). Every requested
            epoch must satisfy :math:`t_{\rm obs}/(1+z) \ge t_0`. Bare values are seconds. Default ``0.3 d``. The
            shock solution forgets its start as :math:`t_0 \to 0`; ``0.3 d`` biases the thermal energy density by
            roughly 0.1% at 5 d.
        max_initial_beta : float, optional
            Largest permitted initial velocity :math:`v_0/c`. The shock and radiation calculations are non-relativistic
            (the radiation uses a static plasma, :math:`\beta = 0`), so both degrade as :math:`v_0` approaches
            :math:`c`. Parameter sets with a larger :math:`v_0` are rejected as infeasible. Default ``0.8``.
        gamma_points_per_decade : int, optional
            Electron Lorentz-factor grid resolution. About 20 per decade converges the flux to ~0.01 dex.
            Default ``20``.
        n_azimuth : int, optional
            Number of azimuths at which the equatorial ring's line of sight is sampled when averaging the free-free
            transmission. The transmission is a smooth, periodic function of azimuth, so a few samples converge
            quickly. Default ``4``.
        integrator_kw : dict, optional
            Keyword arguments forwarded to :func:`scipy.integrate.solve_ivp` for the shock integration. The model
            defaults to ``method="LSODA"`` and ``rtol=1e-7``, which keeps the log-likelihood within about
            :math:`10^{-3}` of a ``Radau`` solution at ``rtol=1e-10`` while being roughly twice as fast as the
            engine's own defaults (``Radau``, ``rtol=1e-10``) or as ``Radau`` at ``rtol=1e-6``. Anything given here
            takes precedence. The integration grid and the adiabatic index cannot be overridden (use
            ``adiabatic_index``).
        seed_photon_luminosity : callable, optional
            The luminosity :math:`L(t)` of the radiation field that Compton-scatters off the electrons, called with the
            source-frame time since explosion (an array in seconds) and returning erg/s (an array, or a
            :class:`~astropy.units.Quantity`). If ``None`` (default), only synchrotron cooling is included. See the
            Notes of the class.
        """
        # Instantiate the superclass.
        super().__init__()

        # ---- Distance and redshift ------------------------------------------------------------------------------ #
        if all(x is None for x in (redshift, luminosity_distance, angular_diameter_distance)):
            raise ValueError(
                "Exactly one of `redshift`, `luminosity_distance` and `angular_diameter_distance` must be provided."
            )

        self._cosmology = get_cosmology(cosmology)
        if luminosity_distance is not None:
            luminosity_distance = ensure_in_units(luminosity_distance, u.Mpc) * u.Mpc
        if angular_diameter_distance is not None:
            angular_diameter_distance = ensure_in_units(angular_diameter_distance, u.Mpc) * u.Mpc
        distances = resolve_cosmological_distances(
            redshift=redshift,
            luminosity_distance=luminosity_distance,
            angular_diameter_distance=angular_diameter_distance,
            cosmology=self._cosmology,
        )
        self._z = float(u.Quantity(distances["redshift"]).value)
        self._D_L = distances["luminosity_distance"].to(u.cm)
        self._D_A = distances["angular_diameter_distance"].to(u.cm)
        if not self._z >= 0:
            raise ValueError(f"The redshift must be non-negative, got {self._z}.")
        self._log_D_A = float(np.log(self._D_A.value))
        self._log_D_L = float(np.log(self._D_L.value))

        # ---- Composition ---------------------------------------------------------------------------------------- #
        # The mean molecular weights are derived once here; they convert the post-shock pressure to a temperature
        # (mu) and set the electron and ion densities of the free-free emission measure (mu_e, mu_i).
        self._X = float(hydrogen_mass_fraction)
        self._Y = float(1.0 - self._X if helium_mass_fraction is None else helium_mass_fraction)
        self._Z_metals = max(1.0 - self._X - self._Y, 0.0)
        if not (0.0 <= self._X <= 1.0 and 0.0 <= self._Y <= 1.0 and self._X + self._Y <= 1.0 + 1e-9):
            raise ValueError(
                "The hydrogen and helium mass fractions must each lie in [0, 1] and sum to at most 1, "
                f"got X = {self._X} and Y = {self._Y}."
            )
        if self._X + 0.25 * self._Y <= 0.0:
            raise ValueError("The CSM must contain hydrogen or helium.")

        self._mu = compute_mean_molecular_weight(self._X, self._Y, self._Z_metals)
        self._mu_e = compute_mean_molecular_weight_per_electron(self._X, self._Y, self._Z_metals)
        self._mu_i = compute_mean_molecular_weight_per_ion(self._X, self._Y, self._Z_metals)

        # Free-free ion mix: hydrogen (Z=1) and helium (Z=2), with number ratio n_H : n_He = X : Y/4.
        n_H, n_He = self._X, 0.25 * self._Y
        self._ion_charges = np.array([1.0, 2.0])
        self._ion_fractions = np.array([n_H, n_He]) / (n_H + n_He)

        # ---- Shock and numerical configuration ------------------------------------------------------------------ #
        self._adiabatic_index = float(adiabatic_index)
        self._r_ref = float(ensure_in_units(reference_radius, u.cm))
        self._r_min = float(ensure_in_units(inner_radius, u.cm))
        self._t_0 = float(ensure_in_units(shock_start_time, u.s))
        self._max_initial_beta = float(max_initial_beta)
        self._gamma_points_per_decade = int(gamma_points_per_decade)
        self._n_azimuth = int(n_azimuth)
        self._filling_table = None  # built on first use; see `filling_table`

        # The model's defaults are applied first so that anything the user passes takes precedence.
        self._integrator_kw = {"method": "LSODA", "rtol": 1e-7, **(integrator_kw or {})}
        reserved = _RESERVED_INTEGRATOR_KW & self._integrator_kw.keys()
        if reserved:
            raise ValueError(
                f"`integrator_kw` may not set {sorted(reserved)}: the model controls the integration grid, "
                "and the adiabatic index is set by `adiabatic_index`."
            )

        if not self._adiabatic_index > 1:
            raise ValueError("`adiabatic_index` must exceed 1.")
        if self._r_ref <= 0:
            raise ValueError("`reference_radius` must be positive.")
        if not self._r_min >= 0:
            raise ValueError("`inner_radius` must be non-negative.")
        if self._t_0 <= 0:
            raise ValueError("`shock_start_time` must be positive.")
        if not 0 < self._max_initial_beta < 1:
            raise ValueError("`max_initial_beta` must lie in (0, 1).")
        if self._gamma_points_per_decade < 1:
            raise ValueError("`gamma_points_per_decade` must be a positive integer.")
        if self._n_azimuth < 1:
            raise ValueError("`n_azimuth` must be a positive integer.")
        if not self._integrator_kw["rtol"] > 0:
            raise ValueError("The integrator's `rtol` must be positive.")

        # ---- Compton seed photons ------------------------------------------------------------------------------- #
        if seed_photon_luminosity is not None and not callable(seed_photon_luminosity):
            raise TypeError("`seed_photon_luminosity` must be a callable L(t) or None.")
        self._seed_luminosity = seed_photon_luminosity

        # ---- Engines -------------------------------------------------------------------------------------------- #
        # The engines are built once so that they are not re-instantiated on every call. The averaged
        # synchrotron kernel is loaded here for the same reason.
        self._shock_engine = PressureDrivenThinShellShockEngine(mu=self._mu)
        self._synchrotron_engine = NumericalSynchrotronEngine()
        self._synchrotron_engine.load_avg_first_kernel()

        # The seed photon callable and the cosmology object are not serializable and are omitted.
        self._register_init(
            redshift=self._z,
            hydrogen_mass_fraction=self._X,
            helium_mass_fraction=self._Y,
            adiabatic_index=self._adiabatic_index,
            reference_radius=self._r_ref,
            inner_radius=self._r_min,
            shock_start_time=self._t_0,
            max_initial_beta=self._max_initial_beta,
            gamma_points_per_decade=self._gamma_points_per_decade,
            n_azimuth=self._n_azimuth,
            integrator_kw=dict(self._integrator_kw),
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
    def mean_molecular_weight(self) -> float:
        r"""float: Mean molecular weight :math:`\mu` of the CSM (in units of :math:`m_p`)."""
        return self._mu

    @property
    def mean_molecular_weight_per_electron(self) -> float:
        r"""float: Mean molecular weight per free electron :math:`\mu_e` of the CSM (in units of :math:`m_p`)."""
        return self._mu_e

    @property
    def seed_photon_luminosity(self) -> Optional[Callable]:
        """Callable or None: The Compton seed photon luminosity :math:`L(t)`, or ``None`` (synchrotron cooling only)."""
        return self._seed_luminosity

    @property
    def shock_start_time(self) -> u.Quantity:
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
                f"The model requires t_obs / (1 + z) >= shock_start_time = {self._t_0 / _DAY:.4g} d, "
                f"but the earliest epoch is {np.min(t_rest) / _DAY:.4g} d in the source frame. "
                "Lower `shock_start_time` or drop the earlier epochs."
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
            "r_min": self._r_min,
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
        if v_0 > self._max_initial_beta * _C_CGS:
            return None, (
                f"v_0 = sqrt(2 E_ej / M_ej) = {v_0 / _C_CGS:.3f} c "
                f"exceeds max_initial_beta = {self._max_initial_beta:g} c."
            )
        if not p["gamma_max"] > 1.0201 * p["gamma_min"]:
            return None, "gamma_max must exceed 1.0201 * gamma_min to place a cooling break."

        if self._r_min == 0.0:
            # With no inner cutoff, the mass inside R_0 is the integral of rho r^2 from the origin, which diverges
            # for any present component whose slope is >= 3.
            for amplitude, slope in ((p["rho_int"], p["alpha_int"]), (p["rho_ext"], p["alpha_ext"])):
                if amplitude > 0 and slope >= 3.0:
                    return None, (
                        f"A CSM slope of {slope:g} >= 3 has divergent mass at small radius; "
                        "set a positive `inner_radius`."
                    )

        # The private CGS normalization is used directly: the public `normalize` wraps its result in astropy units,
        # which would only be converted straight back (a measurable cost at every posterior evaluation).
        try:
            K, v_t = _normalize_bpl_cgs(p["E_ej"], p["M_ej"], n=p["n"], delta=p["delta"], v_max=_C_CGS)
        except (ValueError, RuntimeError) as exc:
            # ValueError: invalid inputs. RuntimeError: no transition velocity reproduces E_ej / M_ej under the
            # finite v_max (the kernel's largest attainable mean-square velocity is below the requested one), which
            # can happen for a v_0 that is still below max_initial_beta.
            return None, f"The ejecta profile cannot be normalized: {exc}"

        rho_ej = BrokenPowerLawEjectaProfile.as_optimized_callable(
            K=K, v_t=v_t, n=p["n"], delta=p["delta"], v_max=_C_CGS
        )
        return {"v_0": v_0, "R_0": v_0 * self._t_0, "rho_ej": rho_ej}, None

    def _sector_density(self, p: dict, theta: float):
        """Unit-free ``rho(r, t)`` callable along a fixed sightline."""
        return AxisymmetricCSMDensityProfile.as_optimized_callable(theta=theta, **self._csm_parameters(p))

    @staticmethod
    def _swept_mass(rho_csm, radius: float, r_min: float = 0.0) -> float:
        r"""
        CSM mass between ``r_min`` and ``radius`` along a spherical sightline.

        This is :math:`4\pi\int_{r_{\min}}^R \rho r^2 dr`.

        Returns ``nan`` if the integral diverges or does not converge (for ``r_min = 0`` and a slope
        :math:`\alpha \ge 3`), so that the parameter set is rejected as infeasible rather than carrying an
        unconverged value.
        """

        def integrand(r):
            return float(rho_csm(r, 0.0)) * r * r

        if radius <= r_min:
            return 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=IntegrationWarning)
            try:
                return 4.0 * np.pi * quad(integrand, r_min, radius, epsabs=0.0, epsrel=1e-8, limit=200)[0]
            except IntegrationWarning:
                return float("nan")

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
            M_0 = self._swept_mass(rho_csm, prep["R_0"], self._r_min)
            if not (np.isfinite(M_0) and M_0 > 0):
                return None, (
                    "There is no CSM mass (or it is not finite) inside the starting radius "
                    f"R_0 = v_0 t_0 = {prep['R_0']:.4g} cm for inner_radius = {self._r_min:.4g} cm."
                )

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
                    gamma=self._adiabatic_index,
                    **self._integrator_kw,
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
    def build_filling_table(self, *, progress: bool = True) -> tuple:
        """
        Build (or fetch) the table of the sectors' sky-projected area fractions and store it on the model.

        The table is computed once per process for a fixed grid in aspect, inclination and filling factor (a few
        seconds) and shared by every model instance. It is built automatically on the first evaluation, but calling
        this method beforehand keeps parallel workers from each repeating the computation.

        Parameters
        ----------
        progress : bool, optional
            Show a progress bar while the table is built. Default ``True``.

        Returns
        -------
        tuple
            ``(aspect_grid, inclination_grid, filling_factor_grid, table)``, as accepted by
            :meth:`~trilobite.dynamics.profiles.csm.AxisymmetricCSMDensityProfile.compute_filling_parameters`.
        """
        key = tuple(g.tobytes() for g in _FILLING_GRID) + (_FILLING_N_PIX,)
        if key not in _FILLING_TABLE_CACHE:
            table = AxisymmetricCSMDensityProfile.build_filling_parameter_table(
                *_FILLING_GRID, n_pix=_FILLING_N_PIX, progress=progress
            )
            _FILLING_TABLE_CACHE[key] = (*_FILLING_GRID, table)
        self._filling_table = _FILLING_TABLE_CACHE[key]
        return self._filling_table

    @property
    def filling_table(self) -> tuple:
        """tuple: The area-fraction table, built on first access (see :meth:`build_filling_table`)."""
        if self._filling_table is None:
            self.build_filling_table()
        return self._filling_table

    def _sector_geometry(self, p: dict) -> tuple[dict, dict]:
        """``(equatorial, polar)`` dicts of ``f_V``, ``f_A`` and ``ell_over_R`` (floats) at the current parameters."""
        # The filling factor is clipped just inside (0, 1), where the geometry is defined.
        f = float(np.clip(p["f_V"], 1e-6, 1.0 - 1e-6))
        geometry = AxisymmetricCSMDensityProfile.compute_filling_parameters(
            f, p["inclination"], p["aspect"], table=self.filling_table
        )
        return tuple({k: float(v) for k, v in geometry[kind].items()} for kind in ("equatorial", "polar"))

    @staticmethod
    def _gamma_grid(p: dict, n_per_decade: int) -> np.ndarray:
        """Log-spaced electron Lorentz-factor grid spanning ``[gamma_min, gamma_max]`` (at least 16 points)."""
        n_gamma = max(int(np.ceil(n_per_decade * np.log10(p["gamma_max"] / p["gamma_min"]))) + 1, 16)
        return np.geomspace(p["gamma_min"], p["gamma_max"], n_gamma)

    # ----- Electrons ---------------------------------------------------------------------------------------------- #
    def _seed_luminosity_cgs(self, t_rest: np.ndarray) -> np.ndarray:
        """Evaluate the user's seed photon luminosity L(t) at the source-frame epochs, in erg/s."""
        L = self._seed_luminosity(t_rest)
        if isinstance(L, u.Quantity):
            L = L.to_value(u.erg / u.s)
        L = np.broadcast_to(np.asarray(L, dtype=float), t_rest.shape)
        if not np.all(np.isfinite(L)) or np.any(L < 0):
            raise ValueError("`seed_photon_luminosity` must return finite, non-negative luminosities.")
        return L

    def _cooling_gamma(self, B: np.ndarray, radius: np.ndarray, t_rest: np.ndarray) -> np.ndarray:
        r"""
        Cooling Lorentz factor, at which the cooling time equals the source-frame age.

        Synchrotron losses are always included. If a seed photon luminosity was given, inverse-Compton losses are
        added: the loss rates add, so the inverse cooling Lorentz factors add,
        :math:`1/\gamma_c = 1/\gamma_{c,\rm syn} + 1/\gamma_{c,\rm IC}`.
        """
        inv_gamma_c = 1.0 / _opt_compute_synchrotron_cooling_gamma(B, t_rest)

        if self._seed_luminosity is not None:
            L_seed = self._seed_luminosity_cgs(t_rest)
            with np.errstate(divide="ignore"):  # L = 0 gives gamma_IC = inf, i.e. no Compton cooling
                inv_gamma_c = inv_gamma_c + 1.0 / _opt_compute_IC_cooling_gamma(L_seed, radius, t_rest)

        return 1.0 / inv_gamma_c

    def _microphysics(self, p: dict, state: _SectorState, t_rest: np.ndarray, gamma: np.ndarray):
        """
        Magnetic field, cooling break, and electron distribution at every epoch.

        Returns ``(B, gamma_c, N0, log_N)``, where ``log_N`` has shape ``(n_epochs, n_gamma)``.
        """
        # Equipartition magnetic field from the post-shock thermal energy density.
        B = np.sqrt(8.0 * np.pi * p["epsilon_B"] * state.eth)

        # The cooling break must lie strictly inside the electron distribution's support.
        gamma_c = np.clip(
            self._cooling_gamma(B, state.radius, t_rest),
            1.01 * p["gamma_min"],
            p["gamma_max"] / 1.01,
        )

        # Cooled broken power law at every epoch at once, each normalized to carry a fraction epsilon_e of the
        # thermal energy (equipartition with the magnetic field).
        shape = {
            "p1": p["p"],
            "p2": p["p"] + 1.0,
            "gamma_min": p["gamma_min"],
            "gamma_max": p["gamma_max"],
        }
        N0 = BrokenPowerLaw._normalize_from_magnetic_field(B, p["epsilon_B"], p["epsilon_e"], gamma_c=gamma_c, **shape)
        with np.errstate(divide="ignore"):  # ln(0) = -inf outside the support
            log_N = np.log(BrokenPowerLaw._opt_pdf(gamma[None, :], N0[:, None], gamma_c=gamma_c[:, None], **shape))
        return B, gamma_c, N0, log_N

    # ----- Free-free ---------------------------------------------------------------------------------------------- #
    def _sightlines(self, inclination: float) -> tuple:
        r"""
        Return representative lines of sight of the two sectors as ``(cos_psi, z_hat)`` pairs ``(equatorial, polar)``.

        A sightline leaves the sector's midpoint, which lies in the direction :math:`\hat p` from the center, along the
        direction :math:`\hat n` to the observer. ``cos_psi`` is :math:`\hat p\cdot\hat n` and ``z_hat`` is the
        component of :math:`\hat p` along the symmetry axis. The equatorial ring is sampled at ``n_azimuth``
        azimuths (the transmission is even in azimuth, so the half circle suffices), the polar sector along its
        near and far lobes.
        """
        phi = (np.arange(self._n_azimuth) + 0.5) * np.pi / self._n_azimuth
        equatorial = (np.sin(inclination) * np.cos(phi), np.zeros_like(phi))
        polar = (np.array([np.cos(inclination), -np.cos(inclination)]), np.array([1.0, -1.0]))
        return equatorial, polar

    def _csm_profile_parameters(self, p: dict) -> dict:
        """Return the processed CSM profile parameters, with the angle left to be set at evaluation."""
        return AxisymmetricCSMDensityProfile._validate_and_process_parameters(theta=0.0, **self._csm_parameters(p))

    def _line_of_sight_emission_measure(
        self, p: dict, radius: np.ndarray, cos_psi: np.ndarray, z_hat: np.ndarray, csm: dict
    ) -> np.ndarray:
        r"""
        Emission measure :math:`\int n_e n_i\,ds` along lines of sight from the shock to the observer.

        A sightline starts at the shock radius :math:`R` in the direction :math:`\hat p` and runs along the direction
        :math:`\hat n` to the observer, so that :math:`r^2 = R^2 + s^2 + 2Rs\cos\psi` and
        :math:`\cos\theta = (R\hat p_z + s\cos i)/r`. Only the part of the path outside the shock sphere
        :math:`r > R` is integrated, out to :math:`R_{\rm ion}`; a path that starts by running inward therefore
        begins where it re-emerges, at :math:`s = -2R\cos\psi`. Outside that point :math:`r` increases
        monotonically with :math:`s`.

        The integral is evaluated by composite Gauss-Legendre quadrature in :math:`x = \ln(q/R + \epsilon)`, where
        :math:`q` is the path length beyond the start. Panels are never wider than the CSM transition.

        Parameters
        ----------
        p : dict
            Model parameters (``inclination``, ``R_ion`` and the CSM parameters are used).
        radius : numpy.ndarray
            Shock radii, shape ``(n_t,)``, in cm.
        cos_psi, z_hat : numpy.ndarray
            Direction cosine between the sightline's start and the observer, and the start's axial component, one
            per sightline, shape ``(n_s,)``.
        csm : dict
            Processed CSM profile parameters from :meth:`_csm_profile_parameters`.

        Returns
        -------
        numpy.ndarray
            The emission measure in cm^-5, shape ``(n_s, n_t)``. Zero for shocks beyond :math:`R_{\rm ion}`.
        """
        EM = np.zeros((cos_psi.size, radius.size))
        inside = np.flatnonzero(radius < p["R_ion"])  # only shocks inside the ionized region are absorbed by it
        if inside.size == 0:
            return EM

        cos_i, R_ion = np.cos(p["inclination"]), p["R_ion"]
        width = min(_EM_MAX_PANEL, max(_EM_PANEL_PER_DELTA * p["delta_trans"], _EM_MIN_PANEL))
        norm = 1.0 / (self._mu_e * self._mu_i * _M_P_CGS**2)
        n_nodes = _GL_NODES.size

        # Chunk the epochs so that the node arrays stay bounded when the transition is sharp.
        x_lo = np.log(_LOS_EPSILON)
        cp = cos_psi[:, None]
        span = np.max(self._los_path(radius[inside][None, :], cp, R_ion)[1] - x_lo)
        per_epoch = cos_psi.size * max(int(np.ceil(span / width)), 1) * n_nodes
        size = max(_LOS_MAX_NODES // per_epoch, 1)
        for chunk in (inside[lo : lo + size] for lo in range(0, inside.size, size)):
            R = radius[chunk][None, :]  # (1, m)

            s0, x_hi = self._los_path(R, cp, R_ion)

            n_panels = max(int(np.ceil(np.max(x_hi - x_lo) / width)), 1)
            frac = np.linspace(0.0, 1.0, n_panels + 1)
            edges = x_lo + (x_hi - x_lo)[..., None] * frac  # (n_s, m, n_panels + 1)
            half = 0.5 * np.diff(edges, axis=-1)
            mid = 0.5 * (edges[..., 1:] + edges[..., :-1])
            x = (mid[..., None] + half[..., None] * _GL_NODES).reshape(*mid.shape[:2], -1)
            w = (half[..., None] * _GL_WEIGHTS).reshape(x.shape)

            Rb, s0b = R[..., None], s0[..., None]
            s = s0b + Rb * (np.exp(x) - _LOS_EPSILON)
            r = np.sqrt(Rb**2 + s**2 + 2.0 * Rb * s * cp[..., None])
            cos_theta = np.clip((Rb * z_hat[:, None, None] + s * cos_i) / r, -1.0, 1.0)
            rho = AxisymmetricCSMDensityProfile._opt_eval(r, None, **{**csm, "theta": np.arccos(cos_theta)})

            # ds = R e^x dx
            EM[:, chunk] = norm * np.sum(w * rho**2 * Rb * np.exp(x), axis=-1)
        return EM

    @staticmethod
    def _los_path(R: np.ndarray, cos_psi: np.ndarray, R_ion: float) -> tuple:
        """Start ``s0`` of the path outside the shock sphere and the quadrature variable at its end ``R_ion``."""
        s0 = np.maximum(0.0, -2.0 * R * cos_psi)
        s_end = -R * cos_psi + np.sqrt(R_ion**2 - R**2 * (1.0 - cos_psi**2))
        return s0, np.log((s_end - s0) / R + _LOS_EPSILON)

    @staticmethod
    def _transmission(EM: np.ndarray, tau_per_EM: list) -> list:
        r"""Return the free-free transmission :math:`\langle e^{-\tau}\rangle` over sightlines, one array per epoch."""
        return [np.mean(np.exp(-EM[:, i, None] * tau_per_EM[i][None, :]), axis=0) for i in range(len(tau_per_EM))]

    # ----- Spectra ------------------------------------------------------------------------------------------------ #
    def _free_free_opacity(self, p: dict, nu_groups: list) -> list:
        """
        Free-free optical depth per unit emission measure at each requested frequency (one array per epoch).

        This depends on the frequency and the plasma only, not on the sector, so it is computed once per evaluation.
        """
        log_T = np.log(p["T_e"])

        # The optical depth is exactly linear in the emission measure, but the routine evaluates its Gaunt factor at
        # the mean of the frequencies it is given. Evaluating tau per unit EM one frequency at a time makes each
        # result independent of which other frequencies are requested. The optical depth is a source-frame
        # quantity, so the frequency is shifted to the source frame.
        nu_unique, nu_inverse = np.unique(np.concatenate(nu_groups), return_inverse=True)
        tau_unique = np.array(
            [
                _compute_RJ_ff_optical_depth_EM(
                    np.log([nu_j * (1.0 + self._z)]), 0.0, log_T=log_T, Z=self._ion_charges, X=self._ion_fractions
                )[0]
                for nu_j in nu_unique
            ]
        )
        return np.split(tau_unique[nu_inverse], np.cumsum([nu.size for nu in nu_groups])[:-1])

    def _sector_spectra(
        self,
        p: dict,
        state: _SectorState,
        micro: tuple,
        geometry: dict,
        gamma: np.ndarray,
        nu_groups: list,
    ):
        """
        Intrinsic flux density of one sector, before free-free absorption.

        ``nu_groups[i]`` holds the observer-frame frequencies to evaluate at epoch ``i``. ``geometry`` holds the
        sector's ``f_A`` and ``ell_over_R``. Returns a list of arrays (one per epoch) in CGS flux units.
        """
        B, _, _, log_N = micro
        log_gamma, log_weights = self._synchrotron_engine._build_gamma_grid(gamma)

        # A sector with no projected area does not radiate.
        if not geometry["f_A"] > 0.0:
            return [np.zeros_like(nu) for nu in nu_groups]

        # Emitting geometry of the sector, per epoch.
        log_ell = np.log(geometry["ell_over_R"] * state.radius)
        log_area = np.log(np.pi * state.radius**2 * geometry["f_A"])
        log_B = np.log(B)

        # Flatten the (epoch, frequency) pairs so that the engine is called on whole batches rather than epoch by
        # epoch. Each pair carries the plasma state of its epoch.
        sizes = [nu.size for nu in nu_groups]
        epoch = np.repeat(np.arange(len(nu_groups)), sizes)
        log_nu = np.log(np.concatenate(nu_groups))

        log_F = np.empty(log_nu.size)
        for lo in range(0, log_nu.size, _SPECTRUM_CHUNK):
            sl = slice(lo, lo + _SPECTRUM_CHUNK)
            e = epoch[sl]
            with np.errstate(divide="ignore"):  # log N = -inf outside the electron distribution's support
                log_F[sl] = self._synchrotron_engine._compute_log_pa_flux_density(
                    log_nu[sl],
                    log_ell[e],
                    log_B[e],
                    log_N[e],
                    log_gamma,
                    log_weights,
                    log_area[e],
                    self._log_D_A,
                    z=self._z,
                    paired=True,
                )

        return np.split(np.exp(log_F), np.cumsum(sizes)[:-1])

    def _sector_radiation(
        self,
        p: dict,
        state: _SectorState,
        sightlines: tuple,
        geometry: dict,
        csm: dict,
        t_rest: np.ndarray,
        gamma: np.ndarray,
        nu_groups: list,
        tau_per_EM: list,
    ):
        """
        Microphysics, intrinsic spectrum and free-free transmission of one sector.

        Returns ``(micro, EM, flux, transmission)``, where ``micro`` is the output of :meth:`_microphysics`, ``EM``
        the emission measure along each of the sector's sightlines (shape ``(n_s, n_t)``), and ``flux`` and
        ``transmission`` are lists of arrays, one per epoch, in CGS flux units and dimensionless.
        """
        micro = self._microphysics(p, state, t_rest, gamma)
        EM = self._line_of_sight_emission_measure(p, state.radius, *sightlines, csm)
        flux = self._sector_spectra(p, state, micro, geometry, gamma, nu_groups)
        return micro, EM, flux, self._transmission(EM, tau_per_EM)

    def _total_flux(self, p: dict, states: tuple, t_rest: np.ndarray, nu_groups: list):
        """Total attenuated flux density (CGS) per epoch group, summed over the two sectors."""
        gamma = self._gamma_grid(p, self._gamma_points_per_decade)
        tau_per_EM = self._free_free_opacity(p, nu_groups)  # shared by both sectors
        csm = self._csm_profile_parameters(p)
        total = [np.zeros_like(nu) for nu in nu_groups]

        for state, sightlines, geometry in zip(states, self._sightlines(p["inclination"]), self._sector_geometry(p)):
            _, _, flux, transmission = self._sector_radiation(
                p, state, sightlines, geometry, csm, t_rest, gamma, nu_groups, tau_per_EM
            )
            for i in range(len(nu_groups)):
                total[i] += flux[i] * transmission[i]
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

        gamma = self._gamma_grid(p, self._gamma_points_per_decade)
        nu_groups = [nu_obs] * t_obs.size
        tau_per_EM = self._free_free_opacity(p, nu_groups)  # shared by both sectors
        csm = self._csm_profile_parameters(p)
        results, total = [], 0.0
        for state, sightlines, geometry in zip(states, self._sightlines(p["inclination"]), self._sector_geometry(p)):
            micro, EM, flux, transmission = self._sector_radiation(
                p, state, sightlines, geometry, csm, t_rest, gamma, nu_groups, tau_per_EM
            )
            flux, transmission = np.array(flux), np.array(transmission)
            attenuated = flux * transmission
            with np.errstate(divide="ignore"):
                tau = -np.log(transmission)
            total = total + attenuated
            B, gamma_c, N0, _ = micro
            results.append(
                Diggins26SectorRadiation(
                    flux_density=(1e23 * flux) * u.Jy,
                    flux_density_attenuated=(1e23 * attenuated) * u.Jy,
                    tau_ff=tau,
                    emission_measure=EM.mean(axis=0) * u.cm**-5,
                    magnetic_field=B * u.G,
                    gamma_c=gamma_c,
                    electron_normalization=N0 * u.cm**-3,
                    area_fraction=geometry["f_A"],
                    volume_fraction=geometry["f_V"],
                    path_length_ratio=geometry["ell_over_R"],
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
