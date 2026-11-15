r"""
N-component synchrotron models for LFBOTs based on the work of Diggins+2026.

These are the N-segment generalizations of
:class:`~trilobite.models.LFBOTs.diggins26.Diggins26_TwoComponentModel` and
:class:`~trilobite.models.LFBOTs.diggins26_relativistic.Diggins26_RelativisticTwoComponentModel`. Instead of
standing in the whole equatorial and polar regions with one sightline each, the sphere is split into ``N`` segments
in colatitude, each with its own shock, synchrotron spectrum and free-free attenuation, so that the angular structure
of the CSM is resolved smoothly as ``N`` grows. The disk half-maximum latitude is always a segment edge, which keeps
the polar and equatorial regions distinct at any ``N``, and ``N = 2`` is the two-component model. The segmentation
and the emitting geometry are described in :mod:`~trilobite.models.LFBOTs.sectors`.
"""

import warnings
from collections.abc import Callable
from typing import NamedTuple, Optional, Union

import numpy as np
from astropy import units as u

from ..core.status import ModelStatus
from .diggins26 import (
    _DYNAMICS_PARAMETERS,
    _FILLING_GRID,
    _FILLING_N_PIX,
    _FILLING_TABLE_CACHE,
    Diggins26_TwoComponentModel,
    _sector_dynamics_from_state,
    _state_from_sector_dynamics,
)
from .diggins26_relativistic import Diggins26_RelativisticTwoComponentModel
from .sectors import (
    AREA_MODES,
    Segmentation,
    build_segment_filling_table,
    compute_segmentation,
    interpolate_segment_filling_table,
    segment_volume_fractions,
)

__all__ = [
    "Diggins26_NComponentModel",
    "Diggins26_RelativisticNComponentModel",
    "Diggins26NDynamics",
    "Diggins26NTabulation",
]

_EQUATOR_TOL = 1e-12


# =========================================================================== #
# Result Containers                                                           #
# =========================================================================== #
class Diggins26NDynamics(NamedTuple):
    """
    Shock dynamics of every segment.

    Returned by :meth:`~trilobite.models.LFBOTs.diggins26_n_component.Diggins26_NComponentModel.compute_dynamics`, and
    accepted by :meth:`~trilobite.models.LFBOTs.diggins26_n_component.Diggins26_NComponentModel.tabulate` so that many
    frequencies can be evaluated without re-solving the shocks.
    """

    time: u.Quantity
    """Unique, sorted observer-frame epochs."""
    time_rest: u.Quantity
    """The same epochs in the source frame, ``time / (1 + z)``."""
    sectors: tuple
    """The shock state of every segment (:class:`~trilobite.models.LFBOTs.diggins26.Diggins26SectorDynamics`), ordered
    from the pole to the equator."""
    colatitude: np.ndarray
    """Colatitude of every segment's representative sightline in radians, shape ``(N,)``."""
    colatitude_edges: np.ndarray
    """Colatitudes of the ``N + 1`` segment edges in radians, from ``0`` (pole) to ``pi / 2`` (equator)."""
    n_polar: int
    """Number of polar segments: ``sectors[:n_polar]`` are polar and the rest equatorial."""
    parameters: dict
    """The dynamics parameters (CGS floats) this state was computed with."""


class Diggins26NTabulation(NamedTuple):
    """Full model output on independent time and frequency axes, returned by ``tabulate``."""

    time: u.Quantity
    """Observer-frame epochs, shape ``(n_t,)``."""
    frequency: u.Quantity
    """Observer-frame frequencies, shape ``(n_nu,)``."""
    flux_density: u.Quantity
    """Total attenuated flux density, shape ``(n_t, n_nu)``."""
    sectors: tuple
    """The radiation of every segment (:class:`~trilobite.models.LFBOTs.diggins26.Diggins26SectorRadiation`), ordered
    from the pole to the equator."""
    dynamics: Diggins26NDynamics
    """The shock dynamics used."""


def _segment_sightlines(inclination: float, colatitude: float, n_azimuth: int) -> tuple:
    r"""
    Lines of sight representing one segment, as ``(cos_psi, z_hat)`` (see ``Diggins26_TwoComponentModel._sightlines``).

    A sightline leaves the point of the segment at colatitude :math:`\theta` (in either hemisphere) and azimuth
    :math:`\varphi` from the observer's meridian, and the transmission is even in :math:`\varphi`, so the half circle
    is sampled at ``n_azimuth`` azimuths. The equatorial segment's two hemispheres coincide and the polar segment's
    azimuth is irrelevant, so each of those is sampled once; this is the sampling of the two-component model.
    """
    sin_i, cos_i = np.sin(inclination), np.cos(inclination)
    phi = (np.arange(n_azimuth) + 0.5) * np.pi / n_azimuth
    if abs(colatitude - 0.5 * np.pi) < _EQUATOR_TOL:
        return sin_i * np.cos(phi), np.zeros_like(phi)
    if colatitude < _EQUATOR_TOL:
        return np.array([cos_i, -cos_i]), np.array([1.0, -1.0])
    sin_t, cos_t = np.sin(colatitude), np.cos(colatitude)
    cos_psi = np.concatenate([sin_t * sin_i * np.cos(phi) + cos_t * cos_i, sin_t * sin_i * np.cos(phi) - cos_t * cos_i])
    return cos_psi, np.concatenate([np.full_like(phi, cos_t), np.full_like(phi, -cos_t)])


# =========================================================================== #
# Model Definition                                                            #
# =========================================================================== #
class Diggins26_NComponentModel(Diggins26_TwoComponentModel):
    r"""
    N-component synchrotron model for the radio emission of LFBOTs.

    This is the N-segment generalization of :class:`~trilobite.models.LFBOTs.diggins26.Diggins26_TwoComponentModel`.
    The sphere is split into ``n_sectors`` segments in colatitude (each mirrored about the equator), and every segment
    is modeled as the two-component model models its equatorial and polar regions: a pressure-driven thin shell
    along a representative sightline sets the post-shock state, a cooled broken power-law electron population radiates
    synchrotron emission from the segment's share of the shell, free-free absorption by the CSM exterior to the shock
    attenuates it, and the observed flux density is the sum of the segments. The parameters, variables and all other
    hyperparameters are those of the parent class, which describes the physics. The model adds two hyperparameters,
    ``n_sectors`` and ``area_mode``.

    .. important::

        The first evaluation builds a table of the segments' projected areas, which takes longer than for the parent
        (about 5 s at ``N = 2`` and a minute at ``N = 16``), and is then cached on the model and shared between
        instances for the same ``n_sectors`` and ``area_mode``. Call :meth:`build_filling_table` beforehand when
        evaluating in parallel. The cost of an evaluation grows linearly with ``n_sectors``, since a shock is solved
        along every segment's sightline.

    .. rubric:: Segmentation

    The disk half-maximum latitude :math:`\phi_e = \arctan(b\ln 2)` is always a segment edge, so the polar region is
    never lost at low ``N`` or large aspect :math:`b`. The segments are split between the equatorial and polar regions
    by a rule that does not depend on the parameters, so that the likelihood is continuous: ``n_sectors // 2`` are
    polar and the rest equatorial. Within each region the edges are evenly spaced in the density variable
    :math:`u = |\cot\theta|/b` (equatorial) or :math:`e^{-u}` (polar), which concentrates segments where the disk
    density changes. The innermost polar and equatorial segments are represented by the pole and the equator, and
    every other segment by its midpoint in that variable. The segments are indexed from the pole to the equator. See
    :mod:`~trilobite.models.LFBOTs.sectors`.

    .. rubric:: Geometry and area modes

    Segment :math:`k` holds the fraction :math:`f_{V,k} = f\,(\cos\theta_a - \cos\theta_b)` of the full-sphere volume,
    which sums to the filling factor :math:`f` exactly, and has the projected area :math:`A_\perp = f_A\,\pi R_k^2`
    and path length :math:`\ell = \tfrac{4}{3}(f_{V,k}/f_A)R_k` in the synchrotron calculation. The area fraction
    :math:`f_A` depends on ``area_mode``:

    ``"chord"`` (default)
        Every sky pixel is split between the segments in proportion to their path lengths through the shell, so the
        areas partition the shell's projected area. Because the segments' fluxes are added, a sightline that crosses
        several segments would otherwise be counted once per segment, and the optically thick flux would overcount
        roughly in proportion to the sum of the segments' areas. This definition does not, and the flux converges as
        ``n_sectors`` grows.
    ``"hit"``
        Every pixel whose sightline meets the segment's shell counts fully toward it, which is the definition of the
        two-component model. With ``n_sectors = 2`` the model is then identical to
        :class:`~trilobite.models.LFBOTs.diggins26.Diggins26_TwoComponentModel`. Larger ``n_sectors`` overcounts the
        optically thick flux, and a warning is issued.

    Both are computed in units of each segment's own shock radius, so neither captures the depth ordering of segments
    whose radii differ.

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
    n_sectors : int, optional
        Number of segments, at least 2 (the two-component model). Default 8.
    area_mode : {"chord", "hit"}, optional
        Definition of the segments' projected areas. Default ``"chord"``.
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
        Number of azimuths sampled per hemisphere for each segment's free-free transmission. Default ``4``.
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
        "The N-segment generalization of the Diggins et al. (2026) model has no published counterpart to verify "
        "against. With n_sectors=2 and area_mode='hit' it is the two-component model."
    )

    PARAMETERS = Diggins26_TwoComponentModel.PARAMETERS

    DESCRIPTION = (
        "N-segment (polar to equatorial) thin-shell shock interaction with an axisymmetric CSM, producing "
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
        n_sectors: int = 8,
        area_mode: str = "chord",
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

        The arguments are those of :class:`~trilobite.models.LFBOTs.diggins26.Diggins26_TwoComponentModel`, plus
        ``n_sectors`` and ``area_mode``.

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
        n_sectors : int, optional
            Number of segments, at least 2. ``n_sectors // 2`` are polar and the rest equatorial. Default 8.
        area_mode : {"chord", "hit"}, optional
            Definition of the segments' projected areas (see the class description). Default ``"chord"``.
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
            Largest permitted initial velocity :math:`v_0/c`. Default ``0.8``.
        gamma_points_per_decade : int, optional
            Electron Lorentz-factor grid resolution. Default ``20``.
        n_azimuth : int, optional
            Number of azimuths sampled per hemisphere for each segment's free-free transmission. Default ``4``.
        integrator_kw : dict, optional
            Keyword arguments forwarded to :func:`scipy.integrate.solve_ivp` for the shock integration.
        seed_photon_luminosity : callable, optional
            Luminosity :math:`L(t)` of the radiation field that Compton-scatters off the electrons. If ``None``
            (default), only synchrotron cooling is included.
        """
        n_sectors = int(n_sectors)
        compute_segmentation(1.0, n_sectors)  # validates n_sectors
        if area_mode not in AREA_MODES:
            raise ValueError(f"`area_mode` must be one of {AREA_MODES}, not {area_mode!r}.")
        if area_mode == "hit" and n_sectors > 2:
            warnings.warn(
                "area_mode='hit' counts every pixel a segment's sightlines touch, so the optically thick flux "
                f"overcounts roughly in proportion to the number of segments (n_sectors={n_sectors}). "
                "Use area_mode='chord' for n_sectors > 2.",
                UserWarning,
                stacklevel=2,
            )

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
        self._n_sectors = n_sectors
        self._area_mode = area_mode
        self._register_init(**self._model_init_kwargs, n_sectors=n_sectors, area_mode=area_mode)

    # =============================================== #
    # Configuration properties                        #
    # =============================================== #
    @property
    def n_sectors(self) -> int:
        """int: Number of segments."""
        return self._n_sectors

    @property
    def area_mode(self) -> str:
        """str: Definition of the segments' projected areas, ``"chord"`` or ``"hit"``."""
        return self._area_mode

    def segmentation(self, aspect: float) -> Segmentation:
        """
        Return the segment edges and representative sightlines for a disk of angular thickness ``aspect``.

        Parameters
        ----------
        aspect : float
            Angular thickness :math:`b` of the CSM disk.

        Returns
        -------
        ~trilobite.models.LFBOTs.sectors.Segmentation
            Edges and representative colatitudes in radians, ordered from the pole to the equator.
        """
        return compute_segmentation(aspect, self._n_sectors)

    # =============================================== #
    # Sector hooks                                    #
    # =============================================== #
    def _sector_thetas(self, p: dict) -> tuple:
        return tuple(float(theta) for theta in self.segmentation(p["aspect"]).colatitude)

    def _sector_sightlines(self, p: dict) -> tuple:
        return tuple(_segment_sightlines(p["inclination"], theta, self._n_azimuth) for theta in self._sector_thetas(p))

    def build_filling_table(self, *, progress: bool = True) -> tuple:
        """
        Build (or fetch) the table of the segments' sky-projected area fractions and store it on the model.

        The table is computed once per process for a fixed grid in aspect, inclination and filling factor, for the
        model's ``n_sectors`` and ``area_mode``, and shared by every model instance with the same settings. It is
        built automatically on the first evaluation, but calling this method beforehand keeps parallel workers from
        each repeating the computation.

        Parameters
        ----------
        progress : bool, optional
            Show a progress bar while the table is built. Default ``True``.

        Returns
        -------
        tuple
            ``(aspect_grid, inclination_grid, filling_factor_grid, table)``, where ``table`` has shape
            ``(n_sectors, n_aspect, n_inclination, n_filling_factor)``.
        """
        key = ("sectors", self._n_sectors, self._area_mode, _FILLING_N_PIX) + tuple(g.tobytes() for g in _FILLING_GRID)
        if key not in _FILLING_TABLE_CACHE:
            table = build_segment_filling_table(
                *_FILLING_GRID, self._n_sectors, mode=self._area_mode, n_pix=_FILLING_N_PIX, progress=progress
            )
            _FILLING_TABLE_CACHE[key] = (*_FILLING_GRID, table)
        self._filling_table = _FILLING_TABLE_CACHE[key]
        return self._filling_table

    def _sector_geometry(self, p: dict) -> tuple:
        """One dict of ``f_V``, ``f_A`` and ``ell_over_R`` per segment, from the pole to the equator."""
        # The filling factor is clipped just inside (0, 1), where the geometry is defined.
        f = float(np.clip(p["f_V"], 1e-6, 1.0 - 1e-6))
        edges = self.segmentation(p["aspect"]).edges
        f_V = segment_volume_fractions(edges, f)
        f_A = interpolate_segment_filling_table(self.filling_table, f, p["inclination"], p["aspect"])
        return tuple(
            {"f_V": float(v), "f_A": float(a), "ell_over_R": (4.0 / 3.0) * float(v) / float(a) if a > 0.0 else 0.0}
            for v, a in zip(f_V, f_A)
        )

    def _build_dynamics(self, p: dict, t_unique: np.ndarray, t_rest: np.ndarray, states: tuple) -> Diggins26NDynamics:
        segmentation = self.segmentation(p["aspect"])
        return Diggins26NDynamics(
            time=t_unique * u.s,
            time_rest=t_rest * u.s,
            sectors=tuple(_sector_dynamics_from_state(s) for s in states),
            colatitude=segmentation.colatitude,
            colatitude_edges=segmentation.edges,
            n_polar=segmentation.n_polar,
            parameters={k: p[k] for k in _DYNAMICS_PARAMETERS},
        )

    def _dynamics_states(self, dynamics: Diggins26NDynamics) -> tuple:
        if len(dynamics.sectors) != self._n_sectors:
            raise ValueError(
                f"`dynamics` has {len(dynamics.sectors)} segments but the model has n_sectors={self._n_sectors}."
            )
        return tuple(_state_from_sector_dynamics(d) for d in dynamics.sectors)

    def _build_tabulation(self, t_obs, nu_obs, total, results: list, dynamics) -> Diggins26NTabulation:
        return Diggins26NTabulation(
            time=t_obs * u.s,
            frequency=nu_obs * u.Hz,
            flux_density=(1e23 * total) * u.Jy,
            sectors=tuple(results),
            dynamics=dynamics,
        )

    # =============================================== #
    # Analysis Methods                                #
    # =============================================== #
    def compute_dynamics(self, parameters=None, time=None) -> Diggins26NDynamics:
        """
        Solve the shock dynamics along every segment's sightline.

        The result can be passed to :meth:`tabulate` to evaluate many frequencies without re-solving the shocks.

        Parameters
        ----------
        parameters : dict, optional
            Model parameters (missing entries take their defaults).
        time : array-like or ~astropy.units.Quantity
            Observer-frame epochs. Bare values are seconds.

        Returns
        -------
        Diggins26NDynamics
            The dynamics at the unique, sorted epochs.

        Raises
        ------
        ValueError
            If the parameters are infeasible (see :meth:`diagnose`) or an epoch precedes ``t_0``.
        """
        return super().compute_dynamics(parameters, time)

    def tabulate(self, time, frequency, parameters=None, *, dynamics: Optional[Diggins26NDynamics] = None):
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
        dynamics : Diggins26NDynamics, optional
            A previous result of :meth:`compute_dynamics` computed with the same dynamics parameters at the same
            epochs, to avoid re-solving the shocks.

        Returns
        -------
        Diggins26NTabulation
            The total and per-segment flux densities and diagnostics on the ``(n_t, n_nu)`` grid.

        Raises
        ------
        ValueError
            If the parameters are infeasible, an epoch precedes ``t_0``, or ``dynamics`` does not match.
        """
        return super().tabulate(time, frequency, parameters, dynamics=dynamics)


class Diggins26_RelativisticNComponentModel(Diggins26_NComponentModel, Diggins26_RelativisticTwoComponentModel):
    r"""
    Relativistic N-component synchrotron model for the radio emission of LFBOTs.

    This is the relativistic counterpart of
    :class:`~trilobite.models.LFBOTs.diggins26_n_component.Diggins26_NComponentModel`, and the N-segment
    generalization of
    :class:`~trilobite.models.LFBOTs.diggins26_relativistic.Diggins26_RelativisticTwoComponentModel`: the segmentation,
    area modes and evaluation paths are those of the former, and the relativistic shock, thermal electron population
    (``epsilon_T``) and their caveats are those of the latter. With ``n_sectors=2`` and ``area_mode="hit"`` it is the
    relativistic two-component model.

    The constructor is that of :class:`~trilobite.models.LFBOTs.diggins26_n_component.Diggins26_NComponentModel`,
    with the same ``n_sectors`` (default 8) and ``area_mode`` (default ``"chord"``) hyperparameters.

    References
    ----------
    Diggins et al. (2026, in preparation).
    """

    STATUS = ModelStatus.NOVEL
    STATUS_NOTE = (
        "The relativistic N-segment extension of the Diggins et al. (2026) model has no published counterpart to "
        "verify against. The ejecta normalization and the radiation are not made relativistic."
    )

    PARAMETERS = Diggins26_RelativisticTwoComponentModel.PARAMETERS

    DESCRIPTION = (
        "N-segment (polar to equatorial) relativistic thin-shell shock interaction with an axisymmetric CSM, "
        "producing free-free attenuated synchrotron emission from thermal and non-thermal electrons."
    )
    REFERENCE = "Diggins et al. (2026, in preparation)."
