r"""
Relativistic two-component synchrotron model for LFBOTs based on the work of Diggins+2026.

This module implements the relativistic counterpart of
:class:`~trilobite.models.LFBOTs.diggins26.Diggins26_TwoComponentModel`. The geometry, the free-free absorption, the
cooling and the redshift treatment are identical. Two ingredients differ:

1. The shock along each sightline is solved with the relativistic pressure-driven thin-shell engine
   (:class:`~trilobite.dynamics.shocks.numerical.RelPressureDrivenThinShellShockEngine`).
2. The post-shock electrons are a thermal Maxwell-Juettner population in addition to the cooled broken power law.
"""

from collections.abc import Callable
from typing import NamedTuple, Optional, Union

import numpy as np
from astropy import units as u

from trilobite.dynamics.shocks.numerical import RelPressureDrivenThinShellShockEngine
from trilobite.dynamics.shocks.utils import make_homologous_stationary_sources
from trilobite.models.core.parameters import ModelParameter
from trilobite.physics_utils.maxwell_juettner import _opt_mj_log_pdf
from trilobite.physics_utils.thermal_partition import _opt_thermal_electron_energy_fraction, compute_ion_electron_theta
from trilobite.radiation.synchrotron.electron_distributions import MaxwellJuettner
from trilobite.radiation.synchrotron.SEDs.one_zone.seds import _log_trapezoid_weights, _step_grid

from ..core.status import ModelStatus
from .diggins26 import _C_CGS, Diggins26_TwoComponentModel, _SectorState

__all__ = ["Diggins26_RelativisticTwoComponentModel"]

# The ejecta are truncated at v = c, but the homologous velocity r/t exceeds c beyond that, where the density is zero.
# The speed handed to the relativistic shock engine is held just below c so that its Lorentz factor stays finite there.
_BETA_MAX = 1.0 - 1e-12

# Smallest and largest gamma - 1 of the log-spaced points that resolve the thermal electrons. A Maxwell-Juettner
# population sits at gamma - 1 ~ Theta, which a grid log-spaced in gamma (starting at gamma_min) cannot resolve for
# Theta below about 0.3.
_THERMAL_X_MIN = 1e-6
_THERMAL_X_MAX = 1e2

# Relative tolerance used to decide whether a grid node lies on the support of the non-thermal component, since the
# edges are nodes only up to the rounding of exp(log(edge)).
_EDGE_RTOL = 1e-12


class _RelMicrophysics(NamedTuple):
    """Electron microphysics at every epoch of one sector (internal)."""

    B: np.ndarray
    gamma_c: np.ndarray
    N0: np.ndarray
    log_N: np.ndarray
    theta_e: Optional[np.ndarray] = None
    n_thermal: Optional[np.ndarray] = None


def _with_thermal_parameter(parameters: tuple) -> tuple:
    """Insert ``epsilon_T`` after ``epsilon_B`` in the parent's parameters."""
    names = [parameter.name for parameter in parameters]
    position = names.index("epsilon_B") + 1
    epsilon_T = ModelParameter(
        "epsilon_T",
        0.1,
        description="Ratio of the thermal-electron energy to the ion energy in the post-shock gas.",
        latex=r"\epsilon_T",
        bounds=(0, None),
    )
    return parameters[:position] + (epsilon_T,) + parameters[position:]


class Diggins26_RelativisticTwoComponentModel(Diggins26_TwoComponentModel):
    r"""
    Relativistic two-component (equatorial and polar) synchrotron model for the radio emission of LFBOTs.

    This is the relativistic version of :class:`~trilobite.models.LFBOTs.diggins26.Diggins26_TwoComponentModel`, and
    is otherwise the same model: ejecta interact with an axisymmetric CSM along an equatorial and a polar sightline,
    each standing in for its solid-angle sector, and the free-free attenuated synchrotron emission of the sectors
    is summed. The CSM geometry, the sectors' projected areas, the free-free absorption, the cooling of the
    non-thermal electrons, the redshift treatment, the evaluation paths (:meth:`forward_model` and :meth:`tabulate`)
    and the infeasibility policy are all described in the parent class. Two things change.

    .. rubric:: Relativistic shock

    Each sightline's shock is solved with
    :class:`~trilobite.dynamics.shocks.numerical.RelPressureDrivenThinShellShockEngine`, which evolves the shell's
    lab-frame energy and momentum and closes the post-shock state with the relativistic Rankine-Hugoniot conditions,
    instead of the non-relativistic
    :class:`~trilobite.dynamics.shocks.numerical.PressureDrivenThinShellShockEngine`. The shell starts at
    :math:`R_0 = v_0 t_0` with :math:`\beta_0 = v_0/c` and holds the CSM swept up inside :math:`R_0`, as in the
    parent. The two solvers agree to :math:`O(\beta^2)` (see the engine's non-relativistic-limit tests). The
    post-shock thermal energy density :math:`e_{\rm th} = P/(\hat\gamma - 1)` and the proper rest-mass density
    :math:`\rho_d` of the shocked CSM are those of the engine's forward-shock closure, evaluated in the shell frame.

    .. rubric:: Thermal electrons

    In addition to the cooled broken power law of the parent, the post-shock electrons include a Maxwell-Juettner
    population. The thermal energy density is shared as

    .. math::

        e_{\rm th} = u_i + u_{e,\rm th} + u_{e,\rm nt} + u_B,
        \qquad u_B = \epsilon_B\, e_{\rm th},
        \quad u_{e,\rm nt} = \epsilon_e\, e_{\rm th},
        \quad u_{e,\rm th} = \epsilon_T\, u_i,

    (see :mod:`~trilobite.physics_utils.thermal_partition`), so that ``epsilon_e`` and ``epsilon_B`` keep their
    meaning in the parent model and the thermal electrons carry
    :math:`\epsilon_T(1 - \epsilon_B - \epsilon_e)/(1 + \epsilon_T)` of :math:`e_{\rm th}`. The temperature
    :math:`\Theta_e = k_B T_e/(m_e c^2)` of the thermal electrons follows from their energy per electron,
    :math:`\langle\gamma - 1\rangle(\Theta_e) = u_{e,\rm th}\,\mu_e m_p/(\rho_d\, m_e c^2)`, solved exactly for the
    Maxwell-Juettner distribution. The electron distribution is the sum of the thermal population and the
    non-thermal broken power law, each normalized to its own energy. ``epsilon_T = 0`` removes the thermal electrons
    and leaves the parent's electron distribution.

    .. note::

        The following are carried over from the parent model unchanged and are not made relativistic:

        - The ejecta profile is normalized to :math:`(E_{\rm ej}, M_{\rm ej})` with the Newtonian relations, and its
          density is passed to the shock engine as the proper density. The starting speed is
          :math:`v_0 = \sqrt{2E_{\rm ej}/M_{\rm ej}}`.
        - The radiation is that of a static plasma in the shell frame: there is no Doppler boosting or
          aberration of the emission, and the observer time is :math:`t_{\rm obs} = (1+z)\,t`, with no
          light-travel-time compression.
        - The thermal electrons do not cool.

        These matter more as :math:`\Gamma - 1` grows, so ``max_initial_beta`` keeps its default of ``0.8``; raise
        it to explore faster shocks.

    The parameters, hyperparameters and variables are those of the parent class, plus the parameter

    .. list-table::
       :widths: 20 20 60
       :header-rows: 1

       * - **Name**
         - **Symbol**
         - **Description**
       * - ``epsilon_T``
         - :math:`\epsilon_T`
         - Ratio of the thermal-electron energy to the ion energy, :math:`u_{e,\rm th}/u_i`.

    The parameters ``epsilon_B`` and ``epsilon_e`` must now satisfy :math:`\epsilon_B + \epsilon_e < 1`, so that the
    ions carry the remaining energy; a parameter set that violates this is infeasible (see
    :meth:`~trilobite.models.LFBOTs.diggins26.Diggins26_TwoComponentModel.diagnose`). The adiabatic index
    ``adiabatic_index`` is used for both shocked ejecta and shocked CSM, and unlike the parent it must be set when
    the model is constructed because the engine tabulates its jump conditions for it.

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
        Electron Lorentz-factor grid resolution of the non-thermal component. Default ``20``.
    n_azimuth : int, optional
        Number of azimuths sampled for the equatorial free-free transmission. Default ``4``.
    integrator_kw : dict, optional
        Keyword arguments forwarded to :func:`scipy.integrate.solve_ivp` for the shock integration.
    seed_photon_luminosity : callable, optional
        Luminosity :math:`L(t)` of the radiation field that Compton-scatters off the electrons. If ``None``
        (default), only synchrotron cooling is included.

    References
    ----------
    Diggins et al. (2026, in preparation).
    """

    STATUS = ModelStatus.NOVEL
    STATUS_NOTE = (
        "The relativistic extension of the Diggins et al. (2026) model has no published counterpart to verify "
        "against. The ejecta normalization and the radiation are not made relativistic."
    )

    PARAMETERS = _with_thermal_parameter(Diggins26_TwoComponentModel.PARAMETERS)

    DESCRIPTION = (
        "Two-sector (equatorial and polar) relativistic thin-shell shock interaction with an axisymmetric CSM, "
        "producing free-free attenuated synchrotron emission from thermal and non-thermal electrons."
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

        The arguments are those of :class:`~trilobite.models.LFBOTs.diggins26.Diggins26_TwoComponentModel`.

        Parameters
        ----------
        redshift : float, optional
            Source redshift. Exactly one of ``redshift``, ``luminosity_distance`` and ``angular_diameter_distance``
            must be given.
        luminosity_distance : float or ~astropy.units.Quantity, optional
            Source luminosity distance. Bare values are interpreted as Mpc.
        angular_diameter_distance : float or ~astropy.units.Quantity, optional
            Source angular diameter distance. Bare values are interpreted as Mpc.
        cosmology : ~astropy.cosmology.FLRW, optional
            Cosmology used to convert between redshift and distance. Defaults to the Trilobite configuration
            cosmology.
        hydrogen_mass_fraction : float, optional
            Hydrogen mass fraction of the CSM. Default ``0.7``.
        helium_mass_fraction : float, optional
            Helium mass fraction of the CSM. Defaults to ``1 - hydrogen_mass_fraction`` (no metals).
        adiabatic_index : float, optional
            Adiabatic index of the shocked ejecta and CSM, for which the shock engine tabulates its jump conditions.
            Default ``5/3``.
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
            Electron Lorentz-factor grid resolution of the non-thermal component. Default ``20``.
        n_azimuth : int, optional
            Number of azimuths sampled for the equatorial free-free transmission. Default ``4``.
        integrator_kw : dict, optional
            Keyword arguments forwarded to :func:`scipy.integrate.solve_ivp` for the shock integration.
        seed_photon_luminosity : callable, optional
            Luminosity :math:`L(t)` of the radiation field that Compton-scatters off the electrons. If ``None``
            (default), only synchrotron cooling is included.
        """
        super().__init__(
            redshift=redshift,
            luminosity_distance=luminosity_distance,
            angular_diameter_distance=angular_diameter_distance,
            cosmology=cosmology,
            hydrogen_mass_fraction=hydrogen_mass_fraction,
            helium_mass_fraction=helium_mass_fraction,
            adiabatic_index=adiabatic_index,
            reference_radius=reference_radius,
            inner_radius=inner_radius,
            shock_start_time=shock_start_time,
            max_initial_beta=max_initial_beta,
            gamma_points_per_decade=gamma_points_per_decade,
            n_azimuth=n_azimuth,
            integrator_kw=integrator_kw,
            seed_photon_luminosity=seed_photon_luminosity,
        )

        # The relativistic engine pre-tabulates its jump conditions for the adiabatic indices fixed here.
        self._shock_engine = RelPressureDrivenThinShellShockEngine(
            gamma_1=self._adiabatic_index, gamma_4=self._adiabatic_index
        )

    # =============================================== #
    # Preparation and Dynamics                        #
    # =============================================== #
    def _prepare(self, p: dict):
        prep, reason = super()._prepare(p)
        if prep is None:
            return prep, reason
        if not p["epsilon_B"] + p["epsilon_e"] < 1.0:
            return None, "epsilon_B + epsilon_e must be less than 1: the ions carry the remaining energy."
        return prep, None

    def _integrate_sector(self, prep: dict, rho_csm, M_0: float, t_rest: np.ndarray) -> _SectorState:
        rho_1, u_1, rho_4, u_4 = make_homologous_stationary_sources(rho_ej=prep["rho_ej"], rho_csm=rho_csm)

        def beta_1(r, t):
            return min(u_1(r, t) / _C_CGS, _BETA_MAX)

        def beta_4(r, t):
            return min(u_4(r, t) / _C_CGS, _BETA_MAX)

        # The engine's jump conditions are tabulated for the adiabatic index set at construction, so `gamma` is not
        # passed.
        shock = self._shock_engine._compute_shock_properties_cgs(
            time=t_rest,
            rho_1=rho_1,
            rho_4=rho_4,
            beta_1=beta_1,
            beta_4=beta_4,
            R_0=prep["R_0"],
            beta_0=prep["v_0"] / _C_CGS,
            M_0=M_0,
            t_0=self._t_0,
            **self._integrator_kw,
        )
        return _SectorState(
            radius=np.asarray(shock.radius),
            velocity=np.asarray(shock.beta_sh) * _C_CGS,
            mass=np.asarray(shock.mass),
            eth=np.asarray(shock.thermal_energy_density_3),
            density=np.asarray(shock.density_3),
            lorentz_factor=np.asarray(shock.Gamma_sh),
        )

    # =============================================== #
    # Electrons                                       #
    # =============================================== #
    @staticmethod
    def _gamma_grid(p: dict, n_per_decade: int) -> np.ndarray:
        r"""
        Electron Lorentz-factor grid for the thermal and non-thermal components.

        The union of a grid log-spaced in :math:`\gamma` over ``[gamma_min, gamma_max]`` (for the power law) and one
        log-spaced in :math:`\gamma - 1` from ``1e-6`` to ``100`` (for the thermal population), with a pair of nodes
        resolving each edge of the power law's support, where the distribution jumps.
        """
        gamma_min, gamma_max = p["gamma_min"], p["gamma_max"]
        n_gamma = max(int(np.ceil(n_per_decade * np.log10(gamma_max / gamma_min))) + 1, 16)

        x_max = min(_THERMAL_X_MAX, gamma_max - 1.0)
        thermal = 1.0 + np.geomspace(_THERMAL_X_MIN, x_max, max(n_gamma // 2, 16))
        grid = np.unique(np.concatenate([np.geomspace(gamma_min, gamma_max, n_gamma), thermal]))

        log_nodes, _ = _step_grid(np.log(grid), np.log(gamma_min), np.log(gamma_max))
        return np.exp(log_nodes)

    def _quadrature(self, gamma: np.ndarray) -> tuple:
        log_gamma = np.log(gamma)
        return log_gamma, _log_trapezoid_weights(log_gamma)

    def _microphysics(self, p: dict, state: _SectorState, t_rest: np.ndarray, gamma: np.ndarray):
        """
        Magnetic field, cooling break, and electron distribution (thermal plus non-thermal) at every epoch.

        Returns a :class:`_RelMicrophysics`, where ``log_N`` has shape ``(n_epochs, n_gamma)``.
        """
        if state.density is None:
            raise ValueError(
                "The shock state has no post-shock density: it was not computed by the relativistic solver."
            )

        # The non-thermal electrons are the parent's cooled broken power law. It is evaluated on the grid clipped to its
        # support (so that the edge nodes, which are only equal to the edges up to rounding, are not rejected) and
        # then switched off outside it.
        inside = (gamma >= p["gamma_min"] * (1.0 - _EDGE_RTOL)) & (gamma <= p["gamma_max"] * (1.0 + _EDGE_RTOL))
        B, gamma_c, N0, log_N = super()._microphysics(p, state, t_rest, np.clip(gamma, p["gamma_min"], p["gamma_max"]))
        log_N = np.where(inside[None, :], log_N, -np.inf)

        if not p["epsilon_T"] > 0.0:
            return _RelMicrophysics(B, gamma_c, N0, log_N)

        # Thermal electrons: Maxwell-Juettner at the temperature set by their share of the thermal energy.
        _, theta_e = compute_ion_electron_theta(
            state.eth, state.density, self._mu_e, p["epsilon_T"], p["epsilon_B"], p["epsilon_e"]
        )
        fraction = _opt_thermal_electron_energy_fraction(p["epsilon_T"], p["epsilon_B"], p["epsilon_e"])
        n_thermal = MaxwellJuettner._normalize_from_energy_density(state.eth, fraction, Theta=theta_e)
        log_N_thermal = _opt_mj_log_pdf(gamma[None, :], theta_e[:, None], n_thermal[:, None])
        return _RelMicrophysics(B, gamma_c, N0, np.logaddexp(log_N_thermal, log_N), theta_e, n_thermal)
