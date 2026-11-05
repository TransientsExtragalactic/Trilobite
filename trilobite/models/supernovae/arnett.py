r"""
Optical light-curve models for Arnett-type supernovae.

This module turns the bolometric luminosity computed by the one-zone diffusion engines in
:mod:`trilobite.radiation.arnett` into multi-band optical photometry. Each model is an
:class:`~trilobite.models.core.optical.OpticalModel`: it is built with a
:class:`~trilobite.utils.phot_utils.FilterBundle` and returns filter-convolved flux densities at the requested
``(band_index, time)`` pairs.

The models differ only in what powers the ejecta:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Model
     - Power source
   * - :class:`~trilobite.models.supernovae.arnett.NickelCobaltArnettModel`
     - :math:`^{56}\mathrm{Ni} \to {}^{56}\mathrm{Co} \to {}^{56}\mathrm{Fe}` decay (stripped-envelope SNe)
   * - :class:`~trilobite.models.supernovae.arnett.MagnetarArnettModel`
     - Magnetar spin-down with a free braking index (SLSNe)
   * - :class:`~trilobite.models.supernovae.arnett.MagnetarNickelArnettModel`
     - Both of the above

Each of these radiates from a blackbody photosphere. Each also has a variant, with ``Cutoff`` in its name, that
suppresses the ultraviolet below a cutoff wavelength, as observed in superluminous supernovae and as in MOSFiT
(:footcite:t:`Nicholl_2017`):

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Model
     - Differs from
   * - :class:`~trilobite.models.supernovae.arnett.NickelCobaltArnettCutoffModel`
     - :class:`~trilobite.models.supernovae.arnett.NickelCobaltArnettModel`
   * - :class:`~trilobite.models.supernovae.arnett.MagnetarArnettCutoffModel`
     - :class:`~trilobite.models.supernovae.arnett.MagnetarArnettModel`
   * - :class:`~trilobite.models.supernovae.arnett.MagnetarNickelArnettCutoffModel`
     - :class:`~trilobite.models.supernovae.arnett.MagnetarNickelArnettModel`

All of them are subclasses of :class:`~trilobite.models.supernovae.arnett.ArnettOpticalModel`, which supplies the
photosphere and the spectrum and can be subclassed with any
:class:`~trilobite.radiation.arnett.ArnettEngine` to build a new model. The physics of the engines is described in
:ref:`arnett`.

.. footbibliography::
"""

from typing import TYPE_CHECKING, NamedTuple, Optional, Union

import numpy as np
from astropy import units as u

from trilobite.models.core.optical import OpticalModel
from trilobite.models.core.parameters import ModelParameter
from trilobite.physics_utils.cosmology import get_cosmology, resolve_cosmological_distances
from trilobite.radiation.arnett import (
    ArnettEngine,
    MagnetarArnettEngine,
    MagnetarNickelArnettEngine,
    NickelCobaltArnettEngine,
)
from trilobite.radiation.blackbody import _log_specific_flux_fnu_cgs
from trilobite.radiation.constants import sigma_sb_cgs
from trilobite.utils.misc_utils import ensure_in_units

if TYPE_CHECKING:
    from trilobite._typing import _ModelParametersInputRaw, _UnitBearingArrayLike
    from trilobite.utils.phot_utils import FilterBundle

__all__ = [
    "ArnettPhotosphere",
    "ArnettOpticalModel",
    "NickelCobaltArnettModel",
    "MagnetarArnettModel",
    "MagnetarNickelArnettModel",
    "NickelCobaltArnettCutoffModel",
    "MagnetarArnettCutoffModel",
    "MagnetarNickelArnettCutoffModel",
]

_FOUR_PI_SIGMA = 4.0 * np.pi * sigma_sb_cgs
_CM2_PER_G = u.cm**2 / u.g

# =========================================================================== #
# Parameter Declarations                                                      #
# =========================================================================== #
_COMMON_PARAMETERS = (
    ModelParameter(
        "M_ej",
        3.0 * u.Msun,
        base_units=u.g,
        description="Ejecta mass.",
        latex=r"$M_{\rm ej}$",
        bounds=(0.0, None),
    ),
    ModelParameter(
        "E_ej",
        1.0e51 * u.erg,
        base_units=u.erg,
        description="Ejecta kinetic energy (including any energy added by the central source).",
        latex=r"$E_{\rm ej}$",
        bounds=(0.0, None),
    ),
    ModelParameter(
        "kappa",
        0.1 * _CM2_PER_G,
        base_units=_CM2_PER_G,
        description="Gray optical opacity of the ejecta.",
        latex=r"$\kappa$",
        bounds=(0.0, None),
    ),
    ModelParameter(
        "T_floor",
        6.0e3 * u.K,
        base_units=u.K,
        description="Photospheric temperature floor.",
        latex=r"$T_{\rm floor}$",
        bounds=(0.0, None),
    ),
    ModelParameter(
        "t_exp",
        0.0,
        base_units=u.s,
        description="Explosion epoch on the time axis of the data (observer frame).",
        latex=r"$t_{\rm exp}$",
    ),
)

_NICKEL_PARAMETERS = (
    ModelParameter(
        "M_Ni",
        0.1 * u.Msun,
        base_units=u.g,
        description="Initial mass of 56Ni.",
        latex=r"$M_{\rm Ni}$",
        bounds=(0.0, None),
    ),
    ModelParameter(
        "kappa_gamma",
        0.03 * _CM2_PER_G,
        base_units=_CM2_PER_G,
        description="Gamma-ray opacity (used when gamma_trapping=True).",
        latex=r"$\kappa_\gamma$",
        bounds=(0.0, None),
    ),
    ModelParameter(
        "kappa_positron",
        7.0 * _CM2_PER_G,
        base_units=_CM2_PER_G,
        description="Positron opacity (used when positron_trapping=True).",
        latex=r"$\kappa_{e^+}$",
        bounds=(0.0, None),
    ),
)

_MAGNETAR_PARAMETERS = (
    ModelParameter(
        "P_spin",
        2.0e-3 * u.s,
        base_units=u.s,
        description="Initial spin period of the magnetar.",
        latex=r"$P_{\rm spin}$",
        bounds=(0.0, None),
    ),
    ModelParameter(
        "B_dipole",
        1.0e14 * u.G,
        base_units=u.G,
        description="(Effective) polar dipole field of the magnetar.",
        latex=r"$B_{\rm dipole}$",
        bounds=(0.0, None),
    ),
    ModelParameter(
        "braking_index",
        3.0,
        description="Braking index n of the spin-down (3 for a magnetic dipole). Must exceed 1.",
        latex=r"$n$",
        bounds=(1.0, None),
    ),
    ModelParameter(
        "kappa_magnetar",
        0.03 * _CM2_PER_G,
        base_units=_CM2_PER_G,
        description="Opacity to the magnetar's hard radiation (used when gamma_trapping=True).",
        latex=r"$\kappa_{\rm mag}$",
        bounds=(0.0, None),
    ),
    ModelParameter(
        "I_ns",
        1.0e45 * u.g * u.cm**2,
        base_units=u.g * u.cm**2,
        description="Moment of inertia of the neutron star.",
        latex=r"$I_{\rm ns}$",
        bounds=(0.0, None),
    ),
    ModelParameter(
        "R_ns",
        1.0e6 * u.cm,
        base_units=u.cm,
        description="Radius of the neutron star.",
        latex=r"$R_{\rm ns}$",
        bounds=(0.0, None),
    ),
)


_CUTOFF_PARAMETER = ModelParameter(
    "lambda_cut",
    3000.0,
    base_units=u.AA,
    description="Rest-frame wavelength below which the blackbody is suppressed.",
    latex=r"$\lambda_{\rm cut}$",
    bounds=(0.0, None),
)
_ANGSTROM_CM = 1.0e-8


# =========================================================================== #
# Result Container                                                            #
# =========================================================================== #
class ArnettPhotosphere(NamedTuple):
    """The bolometric luminosity and blackbody photosphere of an Arnett model at the requested epochs."""

    time: u.Quantity
    """Observer-frame epochs, as given."""
    luminosity: u.Quantity
    """Source-frame bolometric luminosity :math:`L(t/(1+z))`."""
    temperature: u.Quantity
    """Source-frame photospheric temperature."""
    radius: u.Quantity
    """Source-frame photospheric radius."""


# =========================================================================== #
# Base Class                                                                  #
# =========================================================================== #
class ArnettOpticalModel(OpticalModel):
    r"""
    Abstract base class for optical photometry models built on an Arnett diffusion engine.

    The bolometric luminosity :math:`L(t)` is computed by an
    :class:`~trilobite.radiation.arnett.ArnettEngine` and is radiated by a blackbody photosphere, whose spectrum
    is convolved with the filters of the model's :class:`~trilobite.utils.phot_utils.FilterBundle`. A concrete
    subclass sets ``ENGINE_CLASS`` (the engine class), ``PARAMETERS``, and ``_SOURCE_PARAMETERS`` (the names of the
    parameters that are forwarded to the engine as source parameters).

    Parameters
    ----------
    bundle : ~trilobite.utils.phot_utils.FilterBundle
        The filter set defining the observable bands.
    redshift : float, optional
        Source redshift. Exactly one of ``redshift`` and ``luminosity_distance`` must be given.
    luminosity_distance : float or ~astropy.units.Quantity, optional
        Source luminosity distance :math:`D_L`. Bare values are interpreted as Mpc. Because this fixes
        :math:`D_L` only through ``cosmology``, a source at :math:`z = 0` must be given its distance this way.
    cosmology : ~astropy.cosmology.FLRW, optional
        Cosmology relating redshift and distance. Defaults to the Trilobite configuration cosmology.
    **engine_kwargs
        Options forwarded to the engine constructor (for example ``gamma_trapping`` or ``beta``).

    Notes
    -----
    **The photosphere.** The ejecta are a uniform sphere expanding at the scale velocity
    :math:`v_{\rm sc} = \sqrt{10E_{\rm ej}/3M_{\rm ej}}`, and the photosphere is taken to be the surface of that
    sphere, :math:`R_{\rm ph} = v_{\rm sc} t`. The effective temperature then follows from the Stefan-Boltzmann law,

    .. math::

        T_{\rm eff} = \left(\frac{L}{4\pi\sigma_{\rm SB} R_{\rm ph}^2}\right)^{1/4}.

    Real ejecta cannot cool below the temperature at which they recombine, so the temperature is floored at
    :math:`T_{\rm floor}` (a model parameter). Once :math:`T_{\rm eff} < T_{\rm floor}` the photosphere recedes into
    the ejecta so as to radiate the luminosity at :math:`T_{\rm floor}`:

    .. math::

        T = \max(T_{\rm eff}, T_{\rm floor}), \qquad R = \sqrt{\frac{L}{4\pi\sigma_{\rm SB}T^4}}.

    This is the ``temperature_floor`` photosphere of MOSFiT (:footcite:p:`Nicholl_2017`, Sec. III.6): a constant
    expansion velocity, an effective temperature from the Stefan-Boltzmann law, and a recession at the floor. The one
    difference is the velocity: MOSFiT takes the velocity of the photosphere as a parameter, whereas here it is
    :math:`v_{\rm sc}`, the velocity that the ejecta kinetic energy implies for a uniform sphere, which is also the
    velocity in the diffusion time. The bolometric luminosity is conserved by construction.

    The spectrum of the photosphere is a blackbody. Real supernovae are fainter than a blackbody in the near UV,
    because of line blanketing; the ``*CutoffModel`` subclasses (for example
    :class:`~trilobite.models.supernovae.arnett.MagnetarArnettCutoffModel`) suppress it.

    **Trapping.** A leaky power source (the gamma rays of :math:`^{56}\mathrm{Ni}`, or the radiation of a magnetar)
    thermalizes only the fraction :math:`1 - e^{-A/t^2}` of its power, with
    :math:`A = 3\kappa_\gamma M_{\rm ej}/(4\pi v_{\rm sc}^2)`. The hyperparameter ``trapping_mode`` chooses where the
    fraction is applied. With ``"deposition"`` (the default) it is the fraction at the time the power is deposited,
    which lies inside the diffusion integral, so that energy that leaks out is gone for good. With ``"emergence"`` it
    is the fraction at the time of observation, applied to the diffused luminosity, :math:`L(t) = (1 -
    e^{-A/t^2})\,L_{\rm diffused}(t)`. This is how MOSFiT does it (:footcite:p:`Nicholl_2017`, Eq. 5). The two agree
    when nothing leaks, and differ when the source leaks while it is still being diffused.

    **Explosion epoch.** ``time`` is measured on whatever axis the data use, and the parameter ``t_exp`` is the
    explosion epoch on that axis, so the model evaluates the luminosity at the time since explosion
    :math:`t - t_{\rm exp}`. When the explosion epoch is known, measure ``time`` from it (as for the radio models, with
    the ``time_start`` of the data containers) and leave ``t_exp`` at its default of zero. When it is not, fit
    ``t_exp``; observations earlier than ``t_exp`` have no flux.

    **Redshift.** Redshift and distance are fixed configuration, never parameters. ``time`` is the
    **observer-frame** time and the filters are in the observer frame. The luminosity is evaluated
    at the source-frame time :math:`(t - t_{\rm exp})/(1+z)`, and the observed flux density is

    .. math::

        F_\nu(\nu) = (1+z)\,\frac{\pi R^2 B_{\nu'}(T)}{D_L^2}, \qquad \nu' = (1+z)\nu,

    which conserves the bolometric flux, :math:`\int F_\nu d\nu = L/4\pi D_L^2`.

    **Parameter bounds.** The bounds declared on the parameters (for example :math:`M_{\rm ej} > 0` and
    :math:`n > 1`) are enforced by :meth:`~trilobite.models.core.base.Model.forward_model`, which raises a
    ``ValueError`` for a violation.

    Serialization stores the filter curves, the resolved redshift (or the luminosity distance, if that is what was
    given), and the engine options, but not a custom ``cosmology`` object; a model rebuilt from its specification
    uses the configuration cosmology.

    .. footbibliography::
    """

    ENGINE_CLASS: Optional[type] = None
    """type: The :class:`~trilobite.radiation.arnett.ArnettEngine` subclass that supplies the luminosity."""

    _SOURCE_PARAMETERS: tuple = ()
    """tuple of str: Names of the parameters that are passed to the engine as source parameters."""

    PARAMETERS = _COMMON_PARAMETERS
    DESCRIPTION = "Blackbody photosphere radiating an Arnett diffusion light curve, observed through optical filters."
    REFERENCE = "Arnett 1982."

    def __init__(
        self,
        bundle: "FilterBundle",
        redshift: Optional[float] = None,
        luminosity_distance: Optional[Union[float, u.Quantity]] = None,
        cosmology=None,
        **engine_kwargs,
    ) -> None:
        if self.ENGINE_CLASS is None:
            raise TypeError(f"{type(self).__name__} does not define an ENGINE_CLASS; subclass it to choose one.")
        super().__init__(bundle)

        # ---- Distance and redshift ------------------------------------------------------------------------------ #
        if redshift is None and luminosity_distance is None:
            raise ValueError("Exactly one of `redshift` and `luminosity_distance` must be provided.")
        if luminosity_distance is not None:
            luminosity_distance = ensure_in_units(luminosity_distance, u.Mpc) * u.Mpc
        self._cosmology = get_cosmology(cosmology)
        distances = resolve_cosmological_distances(
            redshift=redshift, luminosity_distance=luminosity_distance, cosmology=self._cosmology
        )
        self._z = float(u.Quantity(distances["redshift"]).value)
        self._D_L = distances["luminosity_distance"].to(u.cm)
        if not self._z >= 0:
            raise ValueError(f"The redshift must be non-negative, got {self._z}.")
        if not self._D_L.value > 0:
            raise ValueError(
                "The luminosity distance must be positive; give `luminosity_distance` for a nearby source."
            )
        self._log_D_L = float(np.log(self._D_L.value))

        # ---- Engine --------------------------------------------------------------------------------------------- #
        self._engine = self.ENGINE_CLASS(**engine_kwargs)

        # The cosmology object is not serializable and is omitted. Whatever fixes the distance is stored.
        self._register_init(
            bundle=bundle,
            redshift=None if luminosity_distance is not None else self._z,
            luminosity_distance=None if luminosity_distance is None else float(luminosity_distance.to_value(u.Mpc)),
            **engine_kwargs,
        )

    # =============================================== #
    # Configuration properties                        #
    # =============================================== #
    @property
    def engine(self) -> ArnettEngine:
        """~trilobite.radiation.arnett.ArnettEngine: The diffusion engine that supplies the luminosity."""
        return self._engine

    @property
    def redshift(self) -> float:
        """float: Source redshift."""
        return self._z

    @property
    def luminosity_distance(self) -> u.Quantity:
        """~astropy.units.Quantity: Source luminosity distance."""
        return self._D_L

    # =============================================== #
    # Physics                                         #
    # =============================================== #
    def _compute_photosphere_cgs(self, t_obs: np.ndarray, parameters: "_ModelParametersInputRaw"):
        """``(L, T, R)`` in CGS at the observer-frame times ``t_obs`` (s, on the data's time axis), all ``(N_t,)``."""
        t_obs = np.atleast_1d(np.asarray(t_obs, dtype=float))
        t_rest = (t_obs - parameters["t_exp"]) / (1.0 + self._z)
        M_ej, E_ej = parameters["M_ej"], parameters["E_ej"]

        L = self._engine._compute_luminosity_cgs(
            t_rest,
            M_ej=M_ej,
            E_ej=E_ej,
            kappa=parameters["kappa"],
            **{name: parameters[name] for name in self._SOURCE_PARAMETERS},
        )

        # The ejecta surface, R = v_sc t, sets the effective temperature; the floor then moves the photosphere in.
        R_surface = np.sqrt(10.0 * E_ej / (3.0 * M_ej)) * t_rest
        T_eff = np.zeros_like(L)
        valid = R_surface > 0
        T_eff[valid] = (L[valid] / (_FOUR_PI_SIGMA * R_surface[valid] ** 2)) ** 0.25
        T = np.maximum(T_eff, parameters["T_floor"])
        R = np.sqrt(L / (_FOUR_PI_SIGMA * T**4))
        return L, T, R

    def _compute_sed(
        self,
        nu_grid: np.ndarray,
        t_unique: np.ndarray,
        parameters: "_ModelParametersInputRaw",
    ) -> np.ndarray:
        L, T, R = self._compute_photosphere_cgs(t_unique, parameters)
        sed = np.zeros((len(t_unique), len(nu_grid)))

        # No photosphere (R = 0) gives no flux. A NaN (a numerical failure upstream) is propagated, not hidden as zero.
        shining = (R > 0) | np.isnan(R)
        if not np.any(shining):
            return sed
        sed[shining] = np.exp(self._compute_log_spectrum_cgs(nu_grid, T[shining], R[shining], parameters))
        return sed

    def _compute_log_spectrum_cgs(
        self, nu_grid: np.ndarray, T: np.ndarray, R: np.ndarray, parameters: "_ModelParametersInputRaw"
    ) -> np.ndarray:
        r"""
        :math:`\ln F_\nu` observed on ``nu_grid`` (Hz) for photospheres of temperature ``T`` (K) and radius ``R`` (cm).

        Returns shape ``(len(T), len(nu_grid))``. This is the hook that decides what the photosphere radiates: here
        a blackbody, and a blackbody with a suppressed UV in the ``*CutoffModel`` subclasses.
        """
        return _log_specific_flux_fnu_cgs(
            np.log(nu_grid)[None, :], np.log(T)[:, None], np.log(R)[:, None], self._log_D_L, self._z, 0.0
        )

    # =============================================== #
    # Analysis                                        #
    # =============================================== #
    def compute_photosphere(
        self, time: "_UnitBearingArrayLike", parameters: Optional[dict] = None
    ) -> ArnettPhotosphere:
        r"""
        Bolometric luminosity and photosphere at the given epochs.

        Parameters
        ----------
        time : ~astropy.units.Quantity or float or ~numpy.ndarray
            Observer-frame time(s), on the same axis as the ``time`` variable of the model (so measured from the
            explosion when ``t_exp`` is zero). Bare numbers are in s.
        parameters : dict, optional
            Model parameters (``Quantity`` or bare numbers in base units). Missing parameters take their defaults.

        Returns
        -------
        ArnettPhotosphere
            The epochs and the source-frame luminosity, temperature, and radius at each. The luminosity is
            evaluated at the source-frame time :math:`(t - t_{\rm exp})/(1+z)`.
        """
        raw = self.coerce_model_parameters({} if parameters is None else parameters)
        t = np.atleast_1d(ensure_in_units(time, u.s))
        L, T, R = self._compute_photosphere_cgs(t, raw)
        return ArnettPhotosphere(time=t * u.s, luminosity=L * (u.erg / u.s), temperature=T * u.K, radius=R * u.cm)


# =========================================================================== #
# Concrete Models                                                             #
# =========================================================================== #
class NickelCobaltArnettModel(ArnettOpticalModel):
    r"""
    Optical light curves of :math:`^{56}\mathrm{Ni}`-powered supernovae.

    The luminosity is that of :class:`~trilobite.radiation.arnett.NickelCobaltArnettEngine`, the one-zone
    diffusion of the :math:`^{56}\mathrm{Ni} \to {}^{56}\mathrm{Co} \to {}^{56}\mathrm{Fe}` decay power, radiated by
    the blackbody photosphere of :class:`~trilobite.models.supernovae.arnett.ArnettOpticalModel`. It is the
    standard model for the diffusion peak of stripped-envelope supernovae.

    .. dropdown:: Parameters

        .. list-table::
           :widths: 20 20 60
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``M_ej``, ``E_ej``
             - :math:`M_{\rm ej}`, :math:`E_{\rm ej}`
             - Ejecta mass and kinetic energy.
           * - ``kappa``
             - :math:`\kappa`
             - Gray optical opacity of the ejecta. Sets the diffusion time.
           * - ``T_floor``
             - :math:`T_{\rm floor}`
             - Photospheric temperature floor.
           * - ``t_exp``
             - :math:`t_{\rm exp}`
             - Explosion epoch on the time axis of the data. Zero if ``time`` is measured from the explosion.
           * - ``M_Ni``
             - :math:`M_{\rm Ni}`
             - Initial mass of :math:`^{56}\mathrm{Ni}`.
           * - ``kappa_gamma``
             - :math:`\kappa_\gamma`
             - Gamma-ray opacity. Unused if ``gamma_trapping=False``.
           * - ``kappa_positron``
             - :math:`\kappa_{e^+}`
             - Positron opacity. Unused unless ``positron_trapping=True``.

    .. dropdown:: Hyperparameters

        .. list-table::
           :widths: 25 15 60
           :header-rows: 1

           * - **Name**
             - **Default**
             - **Description**
           * - ``bundle``
             - (required)
             - The :class:`~trilobite.utils.phot_utils.FilterBundle` defining the bands.
           * - ``redshift``, ``luminosity_distance``
             - (one required)
             - Source distance. The other follows from ``cosmology``.
           * - ``cosmology``
             - configuration
             - Cosmology relating redshift and distance.
           * - ``gamma_trapping``
             - ``True``
             - Include incomplete trapping of the gamma rays.
           * - ``positron_trapping``
             - ``False``
             - Include incomplete trapping of the positrons.
           * - ``trapping_mode``
             - ``"deposition"``
             - Where the incomplete trapping is applied: ``"deposition"`` to the power as it is deposited, or
               ``"emergence"`` to the diffused luminosity at the time of observation, as MOSFiT does. See the Notes of
               :class:`~trilobite.models.supernovae.arnett.ArnettOpticalModel`.
           * - ``beta``
             - ``13.8``
             - Arnett's constant :math:`\beta`.
           * - ``window_depth``
             - ``40``
             - Clock units of past injection retained by the engine.

    .. dropdown:: Variables

        .. list-table::
           :widths: 20 20 60
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``band_index``
             - :math:`b`
             - Filter, as an index into the bundle or a band name.
           * - ``time``
             - :math:`t`
             - Observer-frame time, measured from the explosion if ``t_exp`` is zero.

    Notes
    -----
    See :class:`~trilobite.models.supernovae.arnett.ArnettOpticalModel` for the photosphere, the temperature floor,
    and the treatment of redshift, and :ref:`arnett` for the engine.

    Examples
    --------
    >>> model = NickelCobaltArnettModel(
    ...     bundle, redshift=0.02
    ... )
    >>> flux = model.forward_model(
    ...     {
    ...         "band_index": ["V", "R"],
    ...         "time": [20, 20] * u.day,
    ...     },
    ...     {
    ...         "M_ej": 2 * u.Msun,
    ...         "E_ej": 1e51 * u.erg,
    ...         "M_Ni": 0.3 * u.Msun,
    ...     },
    ... )
    """

    ENGINE_CLASS = NickelCobaltArnettEngine
    _SOURCE_PARAMETERS = ("M_Ni", "kappa_gamma", "kappa_positron")
    PARAMETERS = _COMMON_PARAMETERS + _NICKEL_PARAMETERS
    DESCRIPTION = "56Ni-powered Arnett diffusion light curve observed through optical filters."
    REFERENCE = "Arnett 1982."

    def __init__(
        self,
        bundle: "FilterBundle",
        redshift: Optional[float] = None,
        luminosity_distance: Optional[Union[float, u.Quantity]] = None,
        cosmology=None,
        *,
        gamma_trapping: bool = True,
        positron_trapping: bool = False,
        trapping_mode: str = "deposition",
        beta: float = 13.8,
        window_depth: float = 40.0,
    ) -> None:
        super().__init__(
            bundle,
            redshift,
            luminosity_distance,
            cosmology,
            gamma_trapping=gamma_trapping,
            positron_trapping=positron_trapping,
            trapping_mode=trapping_mode,
            beta=beta,
            window_depth=window_depth,
        )


class MagnetarArnettModel(ArnettOpticalModel):
    r"""
    Optical light curves of magnetar-powered supernovae.

    The luminosity is that of :class:`~trilobite.radiation.arnett.MagnetarArnettEngine`, the one-zone diffusion of
    the spin-down power of a magnetar with a free braking index, radiated by the blackbody photosphere of
    :class:`~trilobite.models.supernovae.arnett.ArnettOpticalModel`. It is the standard model for superluminous
    supernovae.

    .. dropdown:: Parameters

        .. list-table::
           :widths: 20 20 60
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``M_ej``, ``E_ej``
             - :math:`M_{\rm ej}`, :math:`E_{\rm ej}`
             - Ejecta mass and kinetic energy. ``E_ej`` should include the energy added by the magnetar.
           * - ``kappa``
             - :math:`\kappa`
             - Gray optical opacity of the ejecta. Sets the diffusion time.
           * - ``T_floor``
             - :math:`T_{\rm floor}`
             - Photospheric temperature floor.
           * - ``t_exp``
             - :math:`t_{\rm exp}`
             - Explosion epoch on the time axis of the data. Zero if ``time`` is measured from the explosion.
           * - ``P_spin``
             - :math:`P_{\rm spin}`
             - Initial spin period.
           * - ``B_dipole``
             - :math:`B_{\rm dipole}`
             - (Effective) polar dipole field.
           * - ``braking_index``
             - :math:`n`
             - Braking index, greater than 1. ``3`` is magnetic dipole spin-down.
           * - ``kappa_magnetar``
             - :math:`\kappa_{\rm mag}`
             - Opacity to the magnetar's hard radiation. Unused if ``gamma_trapping=False``.
           * - ``I_ns``, ``R_ns``
             - :math:`I_{\rm ns}`, :math:`R_{\rm ns}`
             - Moment of inertia and radius of the neutron star. Usually held fixed.

    .. dropdown:: Hyperparameters

        .. list-table::
           :widths: 25 15 60
           :header-rows: 1

           * - **Name**
             - **Default**
             - **Description**
           * - ``bundle``
             - (required)
             - The :class:`~trilobite.utils.phot_utils.FilterBundle` defining the bands.
           * - ``redshift``, ``luminosity_distance``
             - (one required)
             - Source distance. The other follows from ``cosmology``.
           * - ``cosmology``
             - configuration
             - Cosmology relating redshift and distance.
           * - ``gamma_trapping``
             - ``True``
             - Include incomplete trapping of the magnetar's radiation.
           * - ``trapping_mode``
             - ``"deposition"``
             - Where the incomplete trapping is applied: ``"deposition"`` to the power as it is deposited, or
               ``"emergence"`` to the diffused luminosity at the time of observation, as MOSFiT does. See the Notes of
               :class:`~trilobite.models.supernovae.arnett.ArnettOpticalModel`.
           * - ``beta``
             - ``13.8``
             - Arnett's constant :math:`\beta`.
           * - ``window_depth``
             - ``40``
             - Clock units of past injection retained by the engine.

    .. dropdown:: Variables

        .. list-table::
           :widths: 20 20 60
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``band_index``
             - :math:`b`
             - Filter, as an index into the bundle or a band name.
           * - ``time``
             - :math:`t`
             - Observer-frame time, measured from the explosion if ``t_exp`` is zero.

    Notes
    -----
    See :class:`~trilobite.models.supernovae.arnett.ArnettOpticalModel` for the photosphere, the temperature floor,
    and the treatment of redshift, and :ref:`arnett` for the engine.
    """

    ENGINE_CLASS = MagnetarArnettEngine
    _SOURCE_PARAMETERS = ("P_spin", "B_dipole", "braking_index", "kappa_magnetar", "I_ns", "R_ns")
    PARAMETERS = _COMMON_PARAMETERS + _MAGNETAR_PARAMETERS
    DESCRIPTION = "Magnetar spin-down powered Arnett diffusion light curve observed through optical filters."
    REFERENCE = "Arnett 1982; Kasen & Bildsten 2010; Omand & Sarin 2024."

    def __init__(
        self,
        bundle: "FilterBundle",
        redshift: Optional[float] = None,
        luminosity_distance: Optional[Union[float, u.Quantity]] = None,
        cosmology=None,
        *,
        gamma_trapping: bool = True,
        trapping_mode: str = "deposition",
        beta: float = 13.8,
        window_depth: float = 40.0,
    ) -> None:
        super().__init__(
            bundle,
            redshift,
            luminosity_distance,
            cosmology,
            gamma_trapping=gamma_trapping,
            trapping_mode=trapping_mode,
            beta=beta,
            window_depth=window_depth,
        )


class MagnetarNickelArnettModel(ArnettOpticalModel):
    r"""
    Optical light curves of supernovae powered by both a magnetar and :math:`^{56}\mathrm{Ni}`.

    The luminosity is that of :class:`~trilobite.radiation.arnett.MagnetarNickelArnettEngine`: the two power
    sources diffuse on a common clock, each with its own trapping, and are radiated by the blackbody photosphere of
    :class:`~trilobite.models.supernovae.arnett.ArnettOpticalModel`. Its parameters are the union of those of
    :class:`~trilobite.models.supernovae.arnett.NickelCobaltArnettModel` and
    :class:`~trilobite.models.supernovae.arnett.MagnetarArnettModel`, which are tabulated there. The hyperparameters
    are those of the nickel model.

    .. dropdown:: Parameters

        .. list-table::
           :widths: 20 20 60
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``M_ej``, ``E_ej``, ``kappa``, ``T_floor``, ``t_exp``
             - :math:`M_{\rm ej}`, :math:`E_{\rm ej}`, :math:`\kappa`, :math:`T_{\rm floor}`, :math:`t_{\rm exp}`
             - Ejecta, opacity, photosphere, and explosion-epoch parameters.
           * - ``M_Ni``, ``kappa_gamma``, ``kappa_positron``
             - :math:`M_{\rm Ni}`, :math:`\kappa_\gamma`, :math:`\kappa_{e^+}`
             - Nickel parameters.
           * - ``P_spin``, ``B_dipole``, ``braking_index``, ``kappa_magnetar``, ``I_ns``, ``R_ns``
             - :math:`P_{\rm spin}`, :math:`B_{\rm dipole}`, :math:`n`, :math:`\kappa_{\rm mag}`,
               :math:`I_{\rm ns}`, :math:`R_{\rm ns}`
             - Magnetar parameters.

    Notes
    -----
    See :class:`~trilobite.models.supernovae.arnett.ArnettOpticalModel` for the photosphere, the temperature floor,
    and the treatment of redshift, and :ref:`arnett` for the engine.
    """

    ENGINE_CLASS = MagnetarNickelArnettEngine
    _SOURCE_PARAMETERS = NickelCobaltArnettModel._SOURCE_PARAMETERS + MagnetarArnettModel._SOURCE_PARAMETERS
    PARAMETERS = _COMMON_PARAMETERS + _NICKEL_PARAMETERS + _MAGNETAR_PARAMETERS
    DESCRIPTION = "Magnetar and 56Ni powered Arnett diffusion light curve observed through optical filters."
    REFERENCE = "Arnett 1982; Kasen & Bildsten 2010; Omand & Sarin 2024."

    def __init__(
        self,
        bundle: "FilterBundle",
        redshift: Optional[float] = None,
        luminosity_distance: Optional[Union[float, u.Quantity]] = None,
        cosmology=None,
        *,
        gamma_trapping: bool = True,
        positron_trapping: bool = False,
        trapping_mode: str = "deposition",
        beta: float = 13.8,
        window_depth: float = 40.0,
    ) -> None:
        super().__init__(
            bundle,
            redshift,
            luminosity_distance,
            cosmology,
            gamma_trapping=gamma_trapping,
            positron_trapping=positron_trapping,
            trapping_mode=trapping_mode,
            beta=beta,
            window_depth=window_depth,
        )


# =========================================================================== #
# UV-Suppressed (Modified) Blackbody Models                                   #
# =========================================================================== #
class _CutoffSpectrumMixin:
    r"""
    Replaces the blackbody of an :class:`ArnettOpticalModel` with a blackbody whose UV is suppressed.

    The photosphere (the temperature and radius) is unchanged. Only the spectrum differs: the flux is multiplied by
    :math:`\min(1, \lambda/\lambda_{\rm cut})`, where :math:`\lambda` is the rest-frame wavelength, so that the
    transmission is 100% at :math:`\lambda_{\rm cut}` and falls linearly to zero at :math:`\lambda = 0`. The spectrum is
    then renormalised so that it integrates to the bolometric luminosity, and the flux that is suppressed in the UV
    reappears at longer wavelengths. This is the ``blackbody_cutoff`` SED of MOSFiT (Nicholl et al. 2017), which found
    :math:`\lambda_{\rm cut} \simeq 3000` Angstrom to reproduce the UV spectra and the Swift photometry of SLSNe.
    The correction is small for photospheres that are cooler than about 8000 K, and grows with temperature.
    """

    def _compute_log_spectrum_cgs(self, nu_grid, T, R, parameters):
        return _log_specific_flux_fnu_cgs(
            np.log(nu_grid)[None, :],
            np.log(T)[:, None],
            np.log(R)[:, None],
            self._log_D_L,
            self._z,
            parameters["lambda_cut"] * _ANGSTROM_CM,
        )


class NickelCobaltArnettCutoffModel(_CutoffSpectrumMixin, NickelCobaltArnettModel):
    r"""
    :math:`^{56}\mathrm{Ni}`-powered optical light curves with a UV-suppressed blackbody.

    This is :class:`~trilobite.models.supernovae.arnett.NickelCobaltArnettModel` with the spectrum changed. The
    luminosity and the photosphere are the same, and so are the parameters, the hyperparameters, and the variables,
    which are tabulated there; the spectrum is a blackbody whose flux is suppressed linearly below the rest-frame
    cutoff wavelength :math:`\lambda_{\rm cut}` and renormalised to conserve the luminosity (see
    :ref:`blackbody_cutoff`). There is one additional parameter.

    .. dropdown:: Additional parameter

        .. list-table::
           :widths: 20 20 60
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``lambda_cut``
             - :math:`\lambda_{\rm cut}`
             - Rest-frame cutoff wavelength, in Angstrom. Default ``3000``. Fix it unless the data reach the far UV.
               ``0`` gives the plain blackbody.

    Notes
    -----
    Stripped-envelope supernovae are cooler than superluminous supernovae, and the correction is small for a
    photosphere that is cooler than about 8000 K (the surviving fraction is 0.97), so that this model differs from
    :class:`~trilobite.models.supernovae.arnett.NickelCobaltArnettModel` mainly near and before peak, and in the UV
    filters. The spectrum is

    .. math::

        F_\lambda = B_\lambda(T)\,\frac{\min(1, \lambda/\lambda_{\rm cut})}{f(T)},

    where :math:`f(T)` is the fraction of the blackbody flux that survives, which makes :math:`\int F_\lambda d\lambda
    = L` (see :func:`~trilobite.radiation.blackbody.planck_cutoff_fraction`).
    """

    PARAMETERS = NickelCobaltArnettModel.PARAMETERS + (_CUTOFF_PARAMETER,)
    DESCRIPTION = "56Ni-powered Arnett diffusion light curve with a UV-suppressed blackbody, through optical filters."
    REFERENCE = "Arnett 1982; Nicholl et al. 2017."


class MagnetarArnettCutoffModel(_CutoffSpectrumMixin, MagnetarArnettModel):
    r"""
    Magnetar-powered optical light curves with a UV-suppressed blackbody: the model of superluminous supernovae.

    This is :class:`~trilobite.models.supernovae.arnett.MagnetarArnettModel` with the spectrum changed. The luminosity
    and the photosphere are the same, and so are the parameters, the hyperparameters, and the variables, which are
    tabulated there; the spectrum is a blackbody whose flux is suppressed linearly below the rest-frame cutoff
    wavelength :math:`\lambda_{\rm cut}` and renormalised to conserve the luminosity (see :ref:`blackbody_cutoff`).
    There is one additional parameter.

    .. dropdown:: Additional parameter

        .. list-table::
           :widths: 20 20 60
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``lambda_cut``
             - :math:`\lambda_{\rm cut}`
             - Rest-frame cutoff wavelength, in Angstrom. Default ``3000``. Fix it unless the data reach the far UV.
               ``0`` gives the plain blackbody.

    Notes
    -----
    The spectrum is

    .. math::

        F_\lambda = B_\lambda(T)\,\frac{\min(1, \lambda/\lambda_{\rm cut})}{f(T)},

    where :math:`f(T)` is the fraction of the blackbody flux that survives, which makes :math:`\int F_\lambda d\lambda
    = L` (see :func:`~trilobite.radiation.blackbody.planck_cutoff_fraction`).

    .. rubric:: Reproducing the MOSFiT ``slsn`` model

    With ``trapping_mode="emergence"`` this is the magnetar model of MOSFiT (:footcite:p:`Nicholl_2017`): the same
    engine, diffusion equation, leakage, temperature-floor photosphere, and UV-suppressed spectrum with a cutoff of
    3000 Angstrom.

    .. code-block:: python

        model = MagnetarArnettCutoffModel(
            bundle,
            redshift=0.1,
            trapping_mode="emergence",
        )

    The differences are in the parameters. MOSFiT takes the velocity of the photosphere, :math:`v_{\rm phot}`, as a
    parameter, with a kinetic energy of :math:`\frac12 M_{\rm ej} v_{\rm phot}^2`; here the parameter is the kinetic
    energy, and the velocity that follows from it for a uniform sphere is :math:`v_{\rm sc} = \sqrt{10E_{\rm
    ej}/3M_{\rm ej}}`, which plays the role of :math:`v_{\rm phot}` in the diffusion time, the leakage, and the radius
    of the photosphere. So compare :math:`v_{\rm sc}` with :math:`v_{\rm phot}`, and do not compare the kinetic
    energies. MOSFiT's neutron star mass sets the moment of inertia, which is here the parameter ``I_ns``, and MOSFiT
    also fits a host-galaxy extinction, which is not part of this model.

    .. footbibliography::
    """

    PARAMETERS = MagnetarArnettModel.PARAMETERS + (_CUTOFF_PARAMETER,)
    DESCRIPTION = (
        "Magnetar-powered Arnett diffusion light curve with a UV-suppressed blackbody, through optical filters."
    )
    REFERENCE = "Arnett 1982; Kasen & Bildsten 2010; Nicholl et al. 2017; Omand & Sarin 2024."


class MagnetarNickelArnettCutoffModel(_CutoffSpectrumMixin, MagnetarNickelArnettModel):
    r"""
    Optical light curves powered by a magnetar and :math:`^{56}\mathrm{Ni}`, with a UV-suppressed blackbody.

    This is :class:`~trilobite.models.supernovae.arnett.MagnetarNickelArnettModel` with the spectrum changed. The
    luminosity and the photosphere are the same, and so are the parameters, the hyperparameters, and the variables,
    which are tabulated there; the spectrum is a blackbody whose flux is suppressed linearly below the rest-frame
    cutoff wavelength :math:`\lambda_{\rm cut}` and renormalised to conserve the luminosity (see
    :ref:`blackbody_cutoff` and :class:`~trilobite.models.supernovae.arnett.MagnetarArnettCutoffModel`). There is one
    additional parameter.

    .. dropdown:: Additional parameter

        .. list-table::
           :widths: 20 20 60
           :header-rows: 1

           * - **Name**
             - **Symbol**
             - **Description**
           * - ``lambda_cut``
             - :math:`\lambda_{\rm cut}`
             - Rest-frame cutoff wavelength, in Angstrom. Default ``3000``. Fix it unless the data reach the far UV.
               ``0`` gives the plain blackbody.
    """

    PARAMETERS = MagnetarNickelArnettModel.PARAMETERS + (_CUTOFF_PARAMETER,)
    DESCRIPTION = (
        "Magnetar and 56Ni powered Arnett light curve with a UV-suppressed blackbody, through optical filters."
    )
    REFERENCE = "Arnett 1982; Kasen & Bildsten 2010; Nicholl et al. 2017; Omand & Sarin 2024."
