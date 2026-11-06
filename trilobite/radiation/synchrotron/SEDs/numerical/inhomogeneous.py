r"""
Inhomogeneous numerical synchrotron emission models.

This module provides numerical synchrotron engines for spatially inhomogeneous
emitting regions. These models extend the one-zone synchrotron slab machinery to
multi-zone geometries by evaluating the local radiative-transfer solution across
a collection of zones and summing their projected contributions to the observed
flux density.
"""

from collections.abc import Callable
from typing import Union

import numpy as np
from astropy import units as u
from scipy.special import logsumexp

from trilobite.physics_utils import resolve_cosmological_distances
from trilobite.radiation.synchrotron.SEDs.numerical.core import NumericalSynchrotronEngine
from trilobite.utils.misc_utils import ensure_in_units

_LOG_2PI = np.log(2.0 * np.pi)


class InhomogeneousCylinderSynchrotronEngine(NumericalSynchrotronEngine):
    r"""
    Numerical synchrotron engine for an inhomogeneous nested-cylinder geometry.

    Models synchrotron emission from a set of concentric, edge-on cylindrical
    annuli, each with its own magnetic field strength :math:`B(r)`, electron
    distribution :math:`N(\gamma, r)`, and line-of-sight slab depth
    :math:`\ell(r)`. The observed flux density is obtained by integrating the
    specific intensity :math:`I_\nu(r)` weighted by the annular projected area
    over the radial grid:

    .. math::

        F_\nu \approx \sum_j \frac{2\pi r_j\,\Delta r_j}{D_A^2}\,I_\nu(r_j).

    Each per-ring radiative transfer calculation is performed using the full
    one-zone slab solution from the parent
    :class:`~trilobite.radiation.synchrotron.SEDs.numerical.core.NumericalSynchrotronEngine`.
    All rings are evaluated simultaneously in a single vectorized pass by
    treating the radial index as the zone batch axis, making the model efficient
    enough for MCMC hot-loop use.

    The private API follows a three-level chain:

    1. ``_compute_log_[pa_]ring_specific_intensity`` — comoving-frame
       :math:`\ln I_\nu(r_j)`.
    2. ``_compute_log_[pa_]ring_contributions`` — observer-frame log flux density
       per ring :math:`\ln[(2\pi r_j\,\Delta r_j / D_A^2)\,I_\nu(r_j)]`, shape
       ``(*nu_shape, n_R)``. Summing these in linear space gives :math:`F_\nu`.
    3. ``_compute_log_[pa_]cylinder_flux_density`` — total :math:`\ln F_\nu`,
       shape ``(*nu_shape,)``.

    The class follows the same two-level public/private API pattern as the parent
    engine: private ``_compute_log_*`` methods accept pre-computed log-space CGS
    arrays with no unit validation, while the public wrappers handle unit
    conversion, grid construction, and distance resolution.

    .. warning::

        **Velocity restriction.** This engine enforces :math:`\theta = \pi/2`
        (bulk velocity **perpendicular** to the line of sight) for all rings.
        This models radially expanding cylindrical shells observed edge-on, so
        the Doppler factor reduces to :math:`\mathcal{D} = 1/\Gamma(\beta)`
        independent of ring radius. A **single scalar** :math:`\beta` applies
        uniformly to all rings. This is a deliberate simplification: the model
        is not appropriate for jet-like geometries, oblique outflows, or
        scenarios where the expansion velocity varies between shells.

    See Also
    --------
    :class:`~trilobite.radiation.synchrotron.SEDs.numerical.core.NumericalSynchrotronEngine`
        Base engine providing radiative transfer and kernel interpolation.

    References
    ----------
    .. footbibliography::
    """

    # ------------------------------------------ #
    # Grid Helper                                #
    # ------------------------------------------ #
    @staticmethod
    def _build_radius_grid(
        r: Union[np.ndarray, None],
        r_min: float = 1e14,
        r_max: float = 1e17,
        n_r: int = 50,
        spacing: str = "log",
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Build a radial quadrature grid for the cylinder model.

        When ``r`` is ``None``, constructs a grid of ``n_r`` points spanning
        ``[r_min, r_max]`` with the requested spacing. When ``r`` is provided,
        uses it directly and computes per-point spacings via central differences.

        Parameters
        ----------
        r : ~numpy.ndarray or None
            Explicit radius grid in CGS (cm). If ``None``, built from ``r_min``,
            ``r_max``, and ``n_r``.
        r_min : float, optional
            Lower bound of the grid in CGS (cm). Ignored when ``r`` is provided.
            Default ``1e14``.
        r_max : float, optional
            Upper bound of the grid in CGS (cm). Ignored when ``r`` is provided.
            Default ``1e17``.
        n_r : int, optional
            Number of grid points. Ignored when ``r`` is provided. Default ``50``.
        spacing : {"log", "linear"}, optional
            Grid spacing when ``r`` is not provided. Default ``"log"``.

        Returns
        -------
        log_r : ~numpy.ndarray
            Natural log of the radius grid in CGS (cm).
        log_area_weights : ~numpy.ndarray
            Log of the annular projected area weights
            :math:`2\pi r_j\,\Delta r_j` in CGS (:math:`\mathrm{cm^2}`),
            where :math:`\Delta r_j` is the central-difference spacing at
            point :math:`j`.
        """
        if r is not None:
            r_grid = np.asarray(r, dtype="f8")
        elif spacing == "log":
            r_grid = np.geomspace(r_min, r_max, n_r)
        elif spacing == "linear":
            r_grid = np.linspace(r_min, r_max, n_r)
        else:
            raise ValueError(f"Invalid spacing '{spacing}'. Must be 'log' or 'linear'.")

        log_r = np.log(r_grid)
        dr = np.gradient(r_grid)
        log_area_weights = _LOG_2PI + log_r + np.log(dr)
        return log_r, log_area_weights

    # ------------------------------------------ #
    # Private API — Level 1: per-ring I_nu       #
    # ------------------------------------------ #
    def _compute_log_pa_ring_specific_intensity(
        self,
        log_nu: np.ndarray,
        log_slab_depth: np.ndarray,
        log_B: np.ndarray,
        log_N: np.ndarray,
        log_gamma: np.ndarray,
        log_weights: np.ndarray,
        z: float,
        beta: float,
    ) -> np.ndarray:
        r"""
        Compute the pitch-angle-averaged log specific intensity :math:`\ln I_\nu(r_j)` per ring.

        Thin wrapper around :meth:`_compute_log_pa_specific_intensity` that enforces
        :math:`\cos\theta = 0` (bulk velocity perpendicular to the line of sight).
        All inputs are pre-computed natural-log CGS quantities; no unit conversion or
        grid construction is performed. This is the intended entry point for MCMC
        hot-loop evaluations using the pitch-angle-averaged kernel.

        Parameters
        ----------
        log_nu : ~numpy.ndarray
            Natural log of the observer-frame frequency in CGS (Hz).
        log_slab_depth : ~numpy.ndarray or scalar
            Natural log of the comoving-frame slab depth :math:`\ell(r)` in CGS (cm).
        log_B : ~numpy.ndarray or scalar
            Natural log of the comoving-frame magnetic field strength in CGS (G).
        log_N : ~numpy.ndarray
            Natural log of the comoving-frame electron number density distribution
            :math:`N(\gamma, r)` in :math:`\mathrm{cm^{-3}}`. The last axis is
            the gamma integration axis; the leading axis indexes the rings.
        log_gamma : ~numpy.ndarray
            Natural log of the Lorentz factor grid (shared across all rings).
        log_weights : ~numpy.ndarray
            Log of the quadrature weights :math:`w_i = \gamma_i\,\Delta\ln\gamma_i`.
        z : float
            Cosmological redshift.
        beta : float
            Bulk plasma velocity in units of :math:`c`, applied uniformly to all rings.

        Returns
        -------
        log_I : ~numpy.ndarray
            Natural log of the observer-frame specific intensity per ring, in
            :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}\,sr^{-1}}`.
        """
        return self._compute_log_pa_specific_intensity(
            log_nu,
            log_slab_depth,
            log_B,
            log_N,
            log_gamma,
            log_weights,
            z=z,
            beta=beta,
            cos_theta=0.0,
        )

    def _compute_log_ring_specific_intensity(
        self,
        log_nu: np.ndarray,
        log_slab_depth: np.ndarray,
        log_B: np.ndarray,
        log_N: np.ndarray,
        log_gamma: np.ndarray,
        log_weights: np.ndarray,
        z: float,
        beta: float,
        sin_alpha: Union[float, np.ndarray],
    ) -> np.ndarray:
        r"""
        Compute the fixed-pitch-angle log specific intensity :math:`\ln I_\nu(r_j)` per ring.

        Thin wrapper around :meth:`_compute_log_specific_intensity` that enforces
        :math:`\cos\theta = 0` (bulk velocity perpendicular to the line of sight).
        All inputs are pre-computed natural-log CGS quantities; no unit conversion or
        grid construction is performed. This is the intended entry point for MCMC
        hot-loop evaluations with a fixed (or per-ring) pitch angle.

        Parameters
        ----------
        log_nu : ~numpy.ndarray
            Natural log of the observer-frame frequency in CGS (Hz).
        log_slab_depth : ~numpy.ndarray or scalar
            Natural log of the comoving-frame slab depth :math:`\ell(r)` in CGS (cm).
        log_B : ~numpy.ndarray or scalar
            Natural log of the comoving-frame magnetic field strength in CGS (G).
        log_N : ~numpy.ndarray
            Natural log of the comoving-frame electron number density distribution
            :math:`N(\gamma, r)` in :math:`\mathrm{cm^{-3}}`. The last axis is
            the gamma integration axis; the leading axis indexes the rings.
        log_gamma : ~numpy.ndarray
            Natural log of the Lorentz factor grid (shared across all rings).
        log_weights : ~numpy.ndarray
            Log of the quadrature weights :math:`w_i = \gamma_i\,\Delta\ln\gamma_i`.
        z : float
            Cosmological redshift.
        beta : float
            Bulk plasma velocity in units of :math:`c`, applied uniformly to all rings.
        sin_alpha : float or ~numpy.ndarray or scalar
            Sine of the comoving-frame pitch angle :math:`\sin\alpha`. May be
            per-ring or a single value broadcast to all rings.

        Returns
        -------
        log_I : ~numpy.ndarray
            Natural log of the observer-frame specific intensity per ring, in
            :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}\,sr^{-1}}`.
        """
        return self._compute_log_specific_intensity(
            log_nu,
            log_slab_depth,
            log_B,
            log_N,
            log_gamma,
            log_weights,
            z=z,
            beta=beta,
            cos_theta=0.0,
            sin_alpha=sin_alpha,
        )

    # ------------------------------------------ #
    # Private API — Level 2: per-ring dF_nu      #
    # ------------------------------------------ #
    def _compute_log_pa_ring_contributions(
        self,
        log_nu: np.ndarray,
        log_slab_depth: np.ndarray,
        log_B: np.ndarray,
        log_N: np.ndarray,
        log_gamma: np.ndarray,
        log_weights: np.ndarray,
        log_area_weights: np.ndarray,
        log_D_A: float,
        z: float,
        beta: float,
    ) -> np.ndarray:
        r"""
        Compute the pitch-angle-averaged log flux density contribution per ring.

        Evaluates

        .. math::

            \ln\!\left[\frac{2\pi r_j\,\Delta r_j}{D_A^2}\,I_\nu(r_j)\right]

        for each ring :math:`j`. Summing these contributions in linear space
        over the ring axis yields the total flux density :math:`F_\nu`. All
        inputs are pre-computed natural-log CGS quantities.

        Parameters
        ----------
        log_nu : ~numpy.ndarray
            Natural log of the observer-frame frequency in CGS (Hz).
        log_slab_depth : ~numpy.ndarray or scalar
            Natural log of the comoving-frame slab depth :math:`\ell(r)` in CGS (cm).
        log_B : ~numpy.ndarray or scalar
            Natural log of the comoving-frame magnetic field strength in CGS (G).
        log_N : ~numpy.ndarray
            Natural log of the comoving-frame electron number density distribution
            :math:`N(\gamma, r)` in :math:`\mathrm{cm^{-3}}`.
        log_gamma : ~numpy.ndarray
            Natural log of the Lorentz factor grid (shared across all rings).
        log_weights : ~numpy.ndarray
            Log of the quadrature weights :math:`w_i = \gamma_i\,\Delta\ln\gamma_i`.
        log_area_weights : ~numpy.ndarray
            Log of the annular projected area weights :math:`2\pi r_j\,\Delta r_j`
            in CGS (:math:`\mathrm{cm^2}`). Pre-computed by :meth:`_build_radius_grid`.
        log_D_A : float
            Natural log of the angular diameter distance in CGS (cm).
        z : float
            Cosmological redshift.
        beta : float
            Bulk plasma velocity in units of :math:`c`, applied uniformly to all rings.

        Returns
        -------
        log_dF : ~numpy.ndarray
            Log of the flux density contribution from each ring, in
            :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}}`. Summing
            :math:`\exp(\mathrm{log\_dF})` over the last axis gives :math:`F_\nu`.
        """
        log_I = self._compute_log_pa_ring_specific_intensity(
            log_nu,
            log_slab_depth,
            log_B,
            log_N,
            log_gamma,
            log_weights,
            z,
            beta,
        )
        return log_I + log_area_weights - 2.0 * log_D_A

    def _compute_log_ring_contributions(
        self,
        log_nu: np.ndarray,
        log_slab_depth: np.ndarray,
        log_B: np.ndarray,
        log_N: np.ndarray,
        log_gamma: np.ndarray,
        log_weights: np.ndarray,
        log_area_weights: np.ndarray,
        log_D_A: float,
        z: float,
        beta: float,
        sin_alpha: Union[float, np.ndarray],
    ) -> np.ndarray:
        r"""
        Compute the fixed-pitch-angle log flux density contribution per ring.

        Evaluates

        .. math::

            \ln\!\left[\frac{2\pi r_j\,\Delta r_j}{D_A^2}\,I_\nu(r_j)\right]

        for each ring :math:`j`. Summing these contributions in linear space
        over the ring axis yields the total flux density :math:`F_\nu`. All
        inputs are pre-computed natural-log CGS quantities.

        Parameters
        ----------
        log_nu : ~numpy.ndarray
            Natural log of the observer-frame frequency in CGS (Hz).
        log_slab_depth : ~numpy.ndarray or scalar
            Natural log of the comoving-frame slab depth :math:`\ell(r)` in CGS (cm).
        log_B : ~numpy.ndarray or scalar
            Natural log of the comoving-frame magnetic field strength in CGS (G).
        log_N : ~numpy.ndarray
            Natural log of the comoving-frame electron number density distribution
            :math:`N(\gamma, r)` in :math:`\mathrm{cm^{-3}}`.
        log_gamma : ~numpy.ndarray
            Natural log of the Lorentz factor grid (shared across all rings).
        log_weights : ~numpy.ndarray
            Log of the quadrature weights :math:`w_i = \gamma_i\,\Delta\ln\gamma_i`.
        log_area_weights : ~numpy.ndarray
            Log of the annular projected area weights :math:`2\pi r_j\,\Delta r_j`
            in CGS (:math:`\mathrm{cm^2}`). Pre-computed by :meth:`_build_radius_grid`.
        log_D_A : float
            Natural log of the angular diameter distance in CGS (cm).
        z : float
            Cosmological redshift.
        beta : float
            Bulk plasma velocity in units of :math:`c`, applied uniformly to all rings.
        sin_alpha : float or ~numpy.ndarray or scalar
            Sine of the comoving-frame pitch angle :math:`\sin\alpha`.

        Returns
        -------
        log_dF : ~numpy.ndarray
            Log of the flux density contribution from each ring, in
            :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}}`. Summing
            :math:`\exp(\mathrm{log\_dF})` over the last axis gives :math:`F_\nu`.
        """
        log_I = self._compute_log_ring_specific_intensity(
            log_nu,
            log_slab_depth,
            log_B,
            log_N,
            log_gamma,
            log_weights,
            z,
            beta,
            sin_alpha,
        )
        return log_I + log_area_weights - 2.0 * log_D_A

    # ------------------------------------------ #
    # Private API — Level 3: total F_nu          #
    # ------------------------------------------ #
    def _compute_log_pa_cylinder_flux_density(
        self,
        log_nu: np.ndarray,
        log_slab_depth: np.ndarray,
        log_B: np.ndarray,
        log_N: np.ndarray,
        log_gamma: np.ndarray,
        log_weights: np.ndarray,
        log_area_weights: np.ndarray,
        log_D_A: float,
        z: float,
        beta: float,
    ) -> np.ndarray:
        r"""
        Compute the pitch-angle-averaged log spectral flux density :math:`\ln F_\nu`.

        Sums the per-ring flux density contributions from
        :meth:`_compute_log_pa_ring_contributions` via
        :func:`~scipy.special.logsumexp`. All inputs are pre-computed
        natural-log CGS quantities; this is the intended MCMC hot-loop entry
        point for the pitch-angle-averaged kernel.

        Parameters
        ----------
        log_nu : ~numpy.ndarray
            Natural log of the observer-frame frequency in CGS (Hz).
        log_slab_depth : ~numpy.ndarray or scalar
            Natural log of the comoving-frame slab depth :math:`\ell(r)` in CGS (cm).
        log_B : ~numpy.ndarray or scalar
            Natural log of the comoving-frame magnetic field strength in CGS (G).
        log_N : ~numpy.ndarray
            Natural log of the comoving-frame electron number density distribution
            :math:`N(\gamma, r)` in :math:`\mathrm{cm^{-3}}`.
        log_gamma : ~numpy.ndarray
            Natural log of the Lorentz factor grid (shared across all rings).
        log_weights : ~numpy.ndarray
            Log of the quadrature weights :math:`w_i = \gamma_i\,\Delta\ln\gamma_i`.
        log_area_weights : ~numpy.ndarray
            Log of the annular projected area weights :math:`2\pi r_j\,\Delta r_j`
            in CGS (:math:`\mathrm{cm^2}`). Pre-computed by :meth:`_build_radius_grid`.
        log_D_A : float
            Natural log of the angular diameter distance in CGS (cm).
        z : float
            Cosmological redshift.
        beta : float
            Bulk plasma velocity in units of :math:`c`, applied uniformly to all rings.

        Returns
        -------
        log_F : ~numpy.ndarray
            Natural log of the observer-frame spectral flux density in
            :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}}`.
        """
        log_dF = self._compute_log_pa_ring_contributions(
            log_nu,
            log_slab_depth,
            log_B,
            log_N,
            log_gamma,
            log_weights,
            log_area_weights,
            log_D_A,
            z,
            beta,
        )
        return logsumexp(log_dF, axis=-1)

    def _compute_log_cylinder_flux_density(
        self,
        log_nu: np.ndarray,
        log_slab_depth: np.ndarray,
        log_B: np.ndarray,
        log_N: np.ndarray,
        log_gamma: np.ndarray,
        log_weights: np.ndarray,
        log_area_weights: np.ndarray,
        log_D_A: float,
        z: float,
        beta: float,
        sin_alpha: Union[float, np.ndarray],
    ) -> np.ndarray:
        r"""
        Compute the fixed-pitch-angle log spectral flux density :math:`\ln F_\nu`.

        Sums the per-ring flux density contributions from
        :meth:`_compute_log_ring_contributions` via
        :func:`~scipy.special.logsumexp`. All inputs are pre-computed
        natural-log CGS quantities; this is the intended MCMC hot-loop entry
        point for the fixed-pitch-angle kernel.

        Parameters
        ----------
        log_nu : ~numpy.ndarray
            Natural log of the observer-frame frequency in CGS (Hz).
        log_slab_depth : ~numpy.ndarray or scalar
            Natural log of the comoving-frame slab depth :math:`\ell(r)` in CGS (cm).
        log_B : ~numpy.ndarray or scalar
            Natural log of the comoving-frame magnetic field strength in CGS (G).
        log_N : ~numpy.ndarray
            Natural log of the comoving-frame electron number density distribution
            :math:`N(\gamma, r)` in :math:`\mathrm{cm^{-3}}`.
        log_gamma : ~numpy.ndarray
            Natural log of the Lorentz factor grid (shared across all rings).
        log_weights : ~numpy.ndarray
            Log of the quadrature weights :math:`w_i = \gamma_i\,\Delta\ln\gamma_i`.
        log_area_weights : ~numpy.ndarray
            Log of the annular projected area weights :math:`2\pi r_j\,\Delta r_j`
            in CGS (:math:`\mathrm{cm^2}`). Pre-computed by :meth:`_build_radius_grid`.
        log_D_A : float
            Natural log of the angular diameter distance in CGS (cm).
        z : float
            Cosmological redshift.
        beta : float
            Bulk plasma velocity in units of :math:`c`, applied uniformly to all rings.
        sin_alpha : float or ~numpy.ndarray or scalar
            Sine of the comoving-frame pitch angle :math:`\sin\alpha`.

        Returns
        -------
        log_F : ~numpy.ndarray
            Natural log of the observer-frame spectral flux density in
            :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}}`.
        """
        log_dF = self._compute_log_ring_contributions(
            log_nu,
            log_slab_depth,
            log_B,
            log_N,
            log_gamma,
            log_weights,
            log_area_weights,
            log_D_A,
            z,
            beta,
            sin_alpha,
        )
        return logsumexp(log_dF, axis=-1)

    # ------------------------------------------ #
    # Public API (unit-bearing wrappers)         #
    # ------------------------------------------ #
    def _resolve_inputs(
        self,
        nu,
        r,
        slab_depth,
        B,
        N,
        gamma,
        gamma_min,
        gamma_max,
        n_gamma,
        r_min,
        r_max,
        n_r,
        r_spacing,
        luminosity_distance,
        angular_diameter_distance,
        proper_distance,
        z,
        cosmology,
    ):
        """Shared unit-conversion and grid-building logic for both public methods."""
        nu_cgs = ensure_in_units(nu, u.Hz)
        slab_depth_cgs = ensure_in_units(slab_depth, u.cm)
        B_cgs = ensure_in_units(B, u.G)

        r_cgs = None if r is None else ensure_in_units(r, u.cm)
        _, log_area_weights = self._build_radius_grid(r_cgs, r_min, r_max, n_r, r_spacing)
        n_r_actual = log_area_weights.size

        log_gamma, log_weights = self._build_gamma_grid(gamma, gamma_min, gamma_max, n_gamma)
        log_N = self._resolve_log_N(N, log_gamma)

        # Homogeneous callable (or 1-D array) → broadcast to (n_R, n_gamma) so that
        # the base engine sees zone_shape = (n_r_actual,) for all rings.
        if log_N.ndim == 1:
            log_N = np.broadcast_to(log_N[np.newaxis, :], (n_r_actual, log_gamma.size))

        dist = resolve_cosmological_distances(
            redshift=z if z != 0 else None,
            luminosity_distance=luminosity_distance,
            angular_diameter_distance=angular_diameter_distance,
            proper_distance=proper_distance,
            cosmology=cosmology,
        )
        log_D_A = np.log(dist["angular_diameter_distance"].to(u.cm).value)

        return (
            np.asarray(np.log(nu_cgs), dtype="f8"),
            np.log(slab_depth_cgs),
            np.log(B_cgs),
            log_N,
            log_gamma,
            log_weights,
            log_area_weights,
            log_D_A,
        )

    def compute_ring_specific_intensity(
        self,
        nu: Union[float, np.ndarray, u.Quantity],
        r: Union[np.ndarray, u.Quantity, None],
        slab_depth: Union[float, np.ndarray, u.Quantity],
        B: Union[float, np.ndarray, u.Quantity],
        N: Union[np.ndarray, Callable],
        gamma: Union[np.ndarray, None] = None,
        luminosity_distance: Union[u.Quantity, None] = None,
        angular_diameter_distance: Union[u.Quantity, None] = None,
        proper_distance: Union[u.Quantity, None] = None,
        z: float = 0,
        cosmology=None,
        beta: float = 0,
        alpha: Union[float, u.Quantity, np.ndarray, None] = None,
        *,
        r_min: float = 1e14,
        r_max: float = 1e17,
        n_r: int = 50,
        r_spacing: str = "log",
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> u.Quantity:
        r"""
        Compute the area-weighted flux density contribution :math:`dF_{\nu,j}` for each ring.

        Returns the per-ring quantity

        .. math::

            dF_{\nu,j} = \frac{2\pi r_j\,\Delta r_j}{D_A^2}\,I_\nu(r_j),

        in units of :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}}`.
        Summing over the last (ring) axis recovers the total flux density:

        .. code-block:: python

            F_nu = engine.compute_ring_specific_intensity(
                ...
            ).sum(axis=-1)

        This method is intended for exploration and debugging: inspecting the
        per-ring contributions reveals which annuli dominate the emission and
        how the radial profile shapes the SED.

        .. note::

            This method enforces :math:`\theta = \pi/2` (bulk velocity
            perpendicular to the line of sight). See the class-level warning for
            the implications of this restriction.

        Parameters
        ----------
        nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Observer-frame frequency grid. Bare values
            are treated as Hz.
        r : ~numpy.ndarray, ~astropy.units.Quantity, or None
            Radial grid of cylindrical annuli. Bare values
            are treated as cm. If ``None``, a grid is built from ``r_min``,
            ``r_max``, ``n_r``, and ``r_spacing``.
        slab_depth : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame line-of-sight slab depth :math:`\ell(r)`, shape
            ``(n_R,)`` or scalar (broadcast to all rings). Bare values are
            treated as cm.
        B : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame magnetic field strength :math:`B(r)`, shape
            ``(n_R,)`` or scalar. Bare values are treated as Gauss.
        N : ~numpy.ndarray or callable
            Comoving-frame electron distribution :math:`N(\gamma, r)` in
            :math:`\mathrm{cm^{-3}}`. If callable,
            called as ``N(gamma)`` and must return shape ``(n_R, n_gamma)``
            or broadcastable to it. A callable returning ``(n_gamma,)`` is
            automatically broadcast to all rings.
        gamma : ~numpy.ndarray or None, optional
            Explicit Lorentz factor grid (shared across all rings). If ``None``,
            built from ``gamma_min``, ``gamma_max``, ``n_gamma``.
        luminosity_distance : ~astropy.units.Quantity or None, optional
            Source luminosity distance. Exactly one of ``luminosity_distance``,
            ``angular_diameter_distance``, ``proper_distance``, or a non-zero
            ``z`` must be provided.
        angular_diameter_distance : ~astropy.units.Quantity or None, optional
            Source angular diameter distance.
        proper_distance : ~astropy.units.Quantity or None, optional
            Source comoving line-of-sight distance.
        z : float, optional
            Cosmological redshift. Default ``0``.
        cosmology : ~astropy.cosmology.FLRW or None, optional
            Cosmological model for distance conversion.
        beta : float, optional
            Bulk plasma velocity in units of :math:`c`. Default ``0``.
        alpha : float, ~numpy.ndarray, ~astropy.units.Quantity, or None, optional
            Comoving-frame pitch angle or scalar. Bare values
            are treated as radians. If ``None``, the pitch-angle-averaged kernel
            is used.
        r_min : float, optional
            Radius grid lower bound in CGS (cm). Ignored if ``r`` is provided.
            Default ``1e14``.
        r_max : float, optional
            Radius grid upper bound in CGS (cm). Ignored if ``r`` is provided.
            Default ``1e17``.
        n_r : int, optional
            Number of radius grid points. Ignored if ``r`` is provided.
            Default ``50``.
        r_spacing : {"log", "linear"}, optional
            Radius grid spacing. Ignored if ``r`` is provided. Default ``"log"``.
        gamma_min : float, optional
            Gamma grid lower bound. Ignored if ``gamma`` is provided. Default ``1.0``.
        gamma_max : float, optional
            Gamma grid upper bound. Ignored if ``gamma`` is provided. Default ``1e8``.
        n_gamma : int, optional
            Number of gamma grid points. Ignored if ``gamma`` is provided.
            Default ``200``.

        Returns
        -------
        dF_nu : ~astropy.units.Quantity
            Per-ring flux density contributions in
            :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}}`. Summing over the
            last axis gives the total :math:`F_\nu`.
        """
        log_nu, log_slab_depth, log_B, log_N, log_gamma, log_weights, log_area_weights, log_D_A = self._resolve_inputs(
            nu,
            r,
            slab_depth,
            B,
            N,
            gamma,
            gamma_min,
            gamma_max,
            n_gamma,
            r_min,
            r_max,
            n_r,
            r_spacing,
            luminosity_distance,
            angular_diameter_distance,
            proper_distance,
            z,
            cosmology,
        )

        if alpha is None:
            self.ensure_avg_first_kernel_loaded()
            log_dF = self._compute_log_pa_ring_contributions(
                log_nu,
                log_slab_depth,
                log_B,
                log_N,
                log_gamma,
                log_weights,
                log_area_weights,
                log_D_A,
                z=z,
                beta=beta,
            )
        else:
            self.ensure_first_kernel_loaded()
            sin_alpha = np.sin(ensure_in_units(alpha, u.rad))
            log_dF = self._compute_log_ring_contributions(
                log_nu,
                log_slab_depth,
                log_B,
                log_N,
                log_gamma,
                log_weights,
                log_area_weights,
                log_D_A,
                z=z,
                beta=beta,
                sin_alpha=sin_alpha,
            )

        return np.exp(log_dF) * (u.erg / (u.s * u.cm**2 * u.Hz))

    def compute_flux_density(
        self,
        nu: Union[float, np.ndarray, u.Quantity],
        r: Union[np.ndarray, u.Quantity, None],
        slab_depth: Union[float, np.ndarray, u.Quantity],
        B: Union[float, np.ndarray, u.Quantity],
        N: Union[np.ndarray, Callable],
        gamma: Union[np.ndarray, None] = None,
        luminosity_distance: Union[u.Quantity, None] = None,
        angular_diameter_distance: Union[u.Quantity, None] = None,
        proper_distance: Union[u.Quantity, None] = None,
        z: float = 0,
        cosmology=None,
        beta: float = 0,
        alpha: Union[float, u.Quantity, np.ndarray, None] = None,
        *,
        r_min: float = 1e14,
        r_max: float = 1e17,
        n_r: int = 50,
        r_spacing: str = "log",
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> u.Quantity:
        r"""
        Compute the observer-frame spectral flux density :math:`F_\nu`.

        Integrates the per-ring specific intensity over the cylinder's radial
        profile:

        .. math::

            F_\nu \approx \sum_j \frac{2\pi r_j\,\Delta r_j}{D_A^2}\,I_\nu(r_j),

        where :math:`\Delta r_j` is the central-difference spacing at ring
        :math:`j` and :math:`D_A` is the angular diameter distance. The sum is
        evaluated in log-space via :func:`~scipy.special.logsumexp` for
        numerical stability.

        .. note::

            This method enforces :math:`\theta = \pi/2` (bulk velocity
            perpendicular to the line of sight). See the class-level warning for
            the implications of this restriction.

        Parameters
        ----------
        nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Observer-frame frequency grid. Bare values
            are treated as Hz.
        r : ~numpy.ndarray, ~astropy.units.Quantity, or None
            Radial grid of cylindrical annuli. Bare values
            are treated as cm. If ``None``, a grid is built from ``r_min``,
            ``r_max``, ``n_r``, and ``r_spacing``.
        slab_depth : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame line-of-sight slab depth :math:`\ell(r)`, shape
            ``(n_R,)`` or scalar (broadcast to all rings). Bare values are
            treated as cm.
        B : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame magnetic field strength :math:`B(r)`, shape
            ``(n_R,)`` or scalar. Bare values are treated as Gauss.
        N : ~numpy.ndarray or callable
            Comoving-frame electron distribution :math:`N(\gamma, r)` in
            :math:`\mathrm{cm^{-3}}`. If callable,
            called as ``N(gamma)`` and must return shape ``(n_R, n_gamma)``
            or broadcastable to it. A callable returning ``(n_gamma,)`` is
            automatically broadcast to all rings.
        gamma : ~numpy.ndarray or None, optional
            Explicit Lorentz factor grid shared across all rings. If ``None``,
            built from ``gamma_min``, ``gamma_max``, ``n_gamma``.
        luminosity_distance : ~astropy.units.Quantity or None, optional
            Source luminosity distance. Exactly one of ``luminosity_distance``,
            ``angular_diameter_distance``, ``proper_distance``, or a non-zero
            ``z`` must be provided.
        angular_diameter_distance : ~astropy.units.Quantity or None, optional
            Source angular diameter distance.
        proper_distance : ~astropy.units.Quantity or None, optional
            Source comoving line-of-sight distance.
        z : float, optional
            Cosmological redshift. Default ``0``.
        cosmology : ~astropy.cosmology.FLRW or None, optional
            Cosmological model for distance conversion.
        beta : float, optional
            Bulk plasma velocity in units of :math:`c`. Default ``0``.
        alpha : float, ~numpy.ndarray, ~astropy.units.Quantity, or None, optional
            Comoving-frame pitch angle or scalar. Bare values
            are treated as radians. If ``None``, the pitch-angle-averaged kernel
            is used.
        r_min : float, optional
            Radius grid lower bound in CGS (cm). Ignored if ``r`` is provided.
            Default ``1e14``.
        r_max : float, optional
            Radius grid upper bound in CGS (cm). Ignored if ``r`` is provided.
            Default ``1e17``.
        n_r : int, optional
            Number of radius grid points. Ignored if ``r`` is provided.
            Default ``50``.
        r_spacing : {"log", "linear"}, optional
            Radius grid spacing. Ignored if ``r`` is provided. Default ``"log"``.
        gamma_min : float, optional
            Gamma grid lower bound. Ignored if ``gamma`` is provided.
            Default ``1.0``.
        gamma_max : float, optional
            Gamma grid upper bound. Ignored if ``gamma`` is provided.
            Default ``1e8``.
        n_gamma : int, optional
            Number of gamma grid points. Ignored if ``gamma`` is provided.
            Default ``200``.

        Returns
        -------
        F_nu : ~astropy.units.Quantity
            Observer-frame spectral flux density in
            :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}}`.
        """
        log_nu, log_slab_depth, log_B, log_N, log_gamma, log_weights, log_area_weights, log_D_A = self._resolve_inputs(
            nu,
            r,
            slab_depth,
            B,
            N,
            gamma,
            gamma_min,
            gamma_max,
            n_gamma,
            r_min,
            r_max,
            n_r,
            r_spacing,
            luminosity_distance,
            angular_diameter_distance,
            proper_distance,
            z,
            cosmology,
        )

        if alpha is None:
            self.ensure_avg_first_kernel_loaded()
            log_F = self._compute_log_pa_cylinder_flux_density(
                log_nu,
                log_slab_depth,
                log_B,
                log_N,
                log_gamma,
                log_weights,
                log_area_weights,
                log_D_A,
                z=z,
                beta=beta,
            )
        else:
            self.ensure_first_kernel_loaded()
            sin_alpha = np.sin(ensure_in_units(alpha, u.rad))
            log_F = self._compute_log_cylinder_flux_density(
                log_nu,
                log_slab_depth,
                log_B,
                log_N,
                log_gamma,
                log_weights,
                log_area_weights,
                log_D_A,
                z=z,
                beta=beta,
                sin_alpha=sin_alpha,
            )

        return np.exp(log_F) * (u.erg / (u.s * u.cm**2 * u.Hz))


class InhomogeneousSphereSynchrotronEngine(NumericalSynchrotronEngine):
    r"""
    Numerical synchrotron engine for an inhomogeneous spherical geometry.

    Models synchrotron emission from a set of concentric spherical shells,
    each with its own magnetic field strength :math:`B(r)` and electron
    distribution :math:`N(\gamma, r)`. The observed flux density is obtained
    by integrating the specific intensity :math:`I_\nu(b)` weighted by the
    annular projected area over impact parameters :math:`b_j = r_j` drawn from
    the radial grid:

    .. math::

        F_\nu \approx \sum_j \frac{2\pi b_j\,\Delta b_j}{D_A^2}\,I_\nu(b_j).

    For each ray at impact parameter :math:`b_j`, the specific intensity is
    computed by a two-pass shell-by-shell radiative transfer procedure. The ray
    traverses the sphere from the far edge inward (far pass) and then back
    outward to the observer (near pass), with the standard slab update

    .. math::

        I_{\nu,k+1} = I_{\nu,k}\,e^{-\tau_k}
                      + S_{\nu,k}\!\left(1 - e^{-\tau_k}\right)

    applied at each shell segment :math:`k`. Here
    :math:`\tau_k = \alpha_{\nu,k}\,\ell_k` is the segment optical depth,
    :math:`S_{\nu,k} = j_{\nu,k}/\alpha_{\nu,k}` is the source function, and
    :math:`\ell_k` is the half-path of the ray through shell :math:`k` on the
    relevant side of the sphere. Shell half-path lengths are pre-computed from the
    grid geometry using geometric-mean boundaries.

    Shell :math:`i` spans from its inner boundary

    .. math::

        r_{i,\mathrm{inner}} =
        \begin{cases}
            0 & i = 0 \\
            \sqrt{r_{i-1}\,r_i} & i > 0
        \end{cases}

    to its outer boundary

    .. math::

        r_{i,\mathrm{outer}} =
        \begin{cases}
            \sqrt{r_i\,r_{i+1}} & i < n_R - 1 \\
            r_{n_R-1}^2 / r_{n_R-2,\mathrm{outer}} & i = n_R - 1.
        \end{cases}

    The half-path of ray :math:`j` through shell :math:`i` is then

    .. math::

        \ell_{j,i} =
        \sqrt{\max(r_{i,\mathrm{outer}}^2 - b_j^2,\,0)}
        - \sqrt{\max(r_{i,\mathrm{inner}}^2 - b_j^2,\,0)},

    which is zero whenever shell :math:`i` lies entirely below the impact
    parameter. Shells with zero half-path contribute nothing to the intensity
    update and are handled automatically by the vectorised RT loop.

    The private API follows a three-level chain:

    1. ``_compute_log_pa_ray_specific_intensity`` — observer-frame
       :math:`\ln I_\nu(b_j)`.
    2. ``_compute_log_pa_ray_contributions`` — observer-frame log flux density
       per ray :math:`\ln[(2\pi b_j\,\Delta b_j / D_A^2)\,I_\nu(b_j)]`, shape
       ``(*nu_shape, n_R)``. Summing these in linear space gives :math:`F_\nu`.
    3. ``_compute_log_pa_sphere_flux_density`` — total :math:`\ln F_\nu`,
       shape ``(*nu_shape,)``.

    .. warning::

        **Non-relativistic restriction.** This engine enforces
        :math:`\beta = 0`; relativistic Doppler corrections are not applied.
        Cosmological redshift :math:`z` is still handled correctly through the
        standard :math:`(1 + z)^3` intensity scaling and frequency shift.

    .. warning::

        **Pitch-angle averaging.** This engine uses only the pitch-angle-averaged
        synchrotron kernel :math:`\bar{F}(x)`, appropriate for an isotropically
        expanding sphere with no preferred emission axis. Fixed pitch-angle
        calculations are not supported. For fixed :math:`\sin\alpha`, use
        :class:`~trilobite.radiation.synchrotron.SEDs.numerical.inhomogeneous.InhomogeneousCylinderSynchrotronEngine`
        which exposes both pitch-angle-averaged and fixed-pitch-angle paths.

    See Also
    --------
    :class:`~trilobite.radiation.synchrotron.SEDs.numerical.inhomogeneous.InhomogeneousCylinderSynchrotronEngine`
        Analogous engine for edge-on cylindrical annuli with full pitch-angle
        and relativistic-motion support.
    :class:`~trilobite.radiation.synchrotron.SEDs.numerical.core.NumericalSynchrotronEngine`
        Base engine providing kernel tabulation and single-zone radiative transfer.

    References
    ----------
    .. footbibliography::
    """

    # ------------------------------------------ #
    # Grid Helper                                #
    # ------------------------------------------ #
    @staticmethod
    def _build_sphere_grid(
        r: Union[np.ndarray, None],
        r_min: float = 1e14,
        r_max: float = 1e17,
        n_r: int = 50,
        spacing: str = "log",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Build the radial grid, area weights, and shell half-path matrix.

        Constructs a radial grid of :math:`n_R` points, computes the annular
        area weights :math:`2\pi r_j\,\Delta r_j` for flux integration, and
        pre-computes the half-path length :math:`\ell_{j,i}` of each ray
        (impact parameter :math:`b_j = r_j`) through each shell :math:`i`.

        Shell boundaries are defined by geometric means of adjacent cell
        centers. The innermost shell extends to :math:`r = 0`; the outermost
        boundary is geometrically extrapolated beyond the last cell center,
        ensuring the outermost ray has a non-zero path through its own shell.

        Parameters
        ----------
        r : ~numpy.ndarray or None
            Explicit radius grid in CGS (cm). If ``None``, built from
            ``r_min``, ``r_max``, and ``n_r``.
        r_min : float, optional
            Lower bound of the grid in CGS (cm). Ignored when ``r`` is
            provided. Default ``1e14``.
        r_max : float, optional
            Upper bound of the grid in CGS (cm). Ignored when ``r`` is
            provided. Default ``1e17``.
        n_r : int, optional
            Number of grid points. Ignored when ``r`` is provided.
            Default ``50``.
        spacing : {"log", "linear"}, optional
            Grid spacing when ``r`` is not provided. Default ``"log"``.

        Returns
        -------
        log_r : ~numpy.ndarray
            Natural log of the radius grid in CGS (cm).
        log_area_weights : ~numpy.ndarray
            Log of the annular projected area weights
            :math:`2\pi r_j\,\Delta r_j` in CGS (:math:`\mathrm{cm^2}`),
            where :math:`\Delta r_j` is the central-difference spacing at
            point :math:`j`.
        half_path : ~numpy.ndarray
            Half-path lengths :math:`\ell_{j,i}` in CGS (cm). Entry
            ``[j, i]`` is the path of ray :math:`j` through shell :math:`i`
            on one side of the sphere midplane. The full path through shell
            :math:`i` for ray :math:`j` is :math:`2\,\ell_{j,i}`.

        Raises
        ------
        ValueError
            If the grid has fewer than 2 points or ``spacing`` is not
            ``"log"`` or ``"linear"``.
        """
        if r is not None:
            r_grid = np.asarray(r, dtype="f8")
        elif spacing == "log":
            r_grid = np.geomspace(r_min, r_max, n_r)
        elif spacing == "linear":
            r_grid = np.linspace(r_min, r_max, n_r)
        else:
            raise ValueError(f"Invalid spacing '{spacing}'. Must be 'log' or 'linear'.")

        if r_grid.size < 2:
            raise ValueError("Sphere grid requires at least 2 radial points.")

        n_r_actual = r_grid.size
        log_r = np.log(r_grid)
        log_area_weights = _LOG_2PI + log_r + np.log(np.gradient(r_grid))

        # Shell boundaries via geometric means.
        r_inner = np.empty(n_r_actual)
        r_outer = np.empty(n_r_actual)

        r_inner[0] = 0.0
        r_inner[1:] = np.sqrt(r_grid[:-1] * r_grid[1:])
        r_outer[:-1] = np.sqrt(r_grid[:-1] * r_grid[1:])
        r_outer[-1] = r_grid[-1] ** 2 / r_outer[-2]

        # half_path[j, i]: half-path of ray j (b=r_grid[j]) through shell i.
        b2 = r_grid[:, np.newaxis] ** 2  # (n_R, 1)
        r_out2 = r_outer[np.newaxis, :] ** 2  # (1, n_R)
        r_in2 = r_inner[np.newaxis, :] ** 2  # (1, n_R)

        half_path = np.sqrt(np.maximum(r_out2 - b2, 0.0)) - np.sqrt(np.maximum(r_in2 - b2, 0.0))
        return log_r, log_area_weights, half_path

    # ------------------------------------------ #
    # Private API — Level 1: per-ray I_nu       #
    # ------------------------------------------ #
    def _compute_log_pa_ray_specific_intensity(
        self,
        log_nu: np.ndarray,
        log_B: np.ndarray,
        log_N: np.ndarray,
        log_gamma: np.ndarray,
        log_weights: np.ndarray,
        z: float,
        half_path: np.ndarray,
    ) -> np.ndarray:
        r"""
        Compute the pitch-angle-averaged log specific intensity per ray.

        Runs the two-pass shell-by-shell radiative transfer for all rays in
        a single vectorised loop over shells. Shells with zero half-path for
        a given impact parameter contribute nothing to the intensity update
        (``exp(-0) = 1``, ``1 - exp(-0) = 0``). Cosmological redshift is
        applied by shifting the frequency to the comoving frame before
        evaluating emissivities, and scaling the resulting intensity by
        :math:`(1 + z)^{-3}`.

        Parameters
        ----------
        log_nu : ~numpy.ndarray
            Natural log of the observer-frame frequency in CGS (Hz).
        log_B : ~numpy.ndarray
            Natural log of the comoving-frame magnetic field per shell in
            CGS (G).
        log_N : ~numpy.ndarray
            Natural log of the comoving-frame electron distribution per
            shell in :math:`\mathrm{cm^{-3}}`. Last axis is the gamma
            integration axis.
        log_gamma : ~numpy.ndarray
            Natural log of the Lorentz factor grid.
        log_weights : ~numpy.ndarray
            Log of the quadrature weights
            :math:`w_i = \gamma_i\,\Delta\!\log\gamma_i`.
        z : float
            Cosmological redshift.
        half_path : ~numpy.ndarray
            Pre-computed half-path matrix from :meth:`_build_sphere_grid`.

        Returns
        -------
        log_I : ~numpy.ndarray
            Natural log of the observer-frame specific intensity per ray
            in :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}\,sr^{-1}}`.
        """
        log_nu = np.asarray(log_nu, dtype="f8")
        n_R = log_B.size

        log_nu_rf = log_nu + np.log1p(z)

        # zone_shape = (n_R,) → log_j, log_alpha have shape (*nu_shape, n_R).
        log_j, log_alpha = self._compute_log_pa_emissivity_and_absorption(
            log_nu_rf, log_B, log_N, log_gamma, log_weights
        )

        alpha = np.exp(log_alpha)
        S = np.exp(log_j - log_alpha)

        intensity = np.zeros(log_nu.shape + (n_R,))

        # Far-side pass: outermost shell to innermost.
        for i in range(n_R - 1, -1, -1):
            hp_i = half_path[:, i]  # (n_R,)
            tau = alpha[..., i][..., np.newaxis] * hp_i  # (*nu_shape, n_R)
            intensity = intensity * np.exp(-tau) - S[..., i][..., np.newaxis] * np.expm1(-tau)

        # Near-side pass: innermost shell to outermost.
        for i in range(n_R):
            hp_i = half_path[:, i]
            tau = alpha[..., i][..., np.newaxis] * hp_i
            intensity = intensity * np.exp(-tau) - S[..., i][..., np.newaxis] * np.expm1(-tau)

        log_I_rf = np.log(np.maximum(intensity, np.finfo("f8").tiny))
        return log_I_rf - 3.0 * np.log1p(z)

    # ------------------------------------------ #
    # Private API — Level 2: per-ray dF_nu      #
    # ------------------------------------------ #
    def _compute_log_pa_ray_contributions(
        self,
        log_nu: np.ndarray,
        log_B: np.ndarray,
        log_N: np.ndarray,
        log_gamma: np.ndarray,
        log_weights: np.ndarray,
        log_area_weights: np.ndarray,
        log_D_A: float,
        z: float,
        half_path: np.ndarray,
    ) -> np.ndarray:
        r"""
        Compute the pitch-angle-averaged log flux density contribution per ray.

        Evaluates

        .. math::

            \ln\!\left[\frac{2\pi b_j\,\Delta b_j}{D_A^2}\,I_\nu(b_j)\right]

        for each ray :math:`j`. Summing these contributions in linear space
        over the ray axis yields the total flux density :math:`F_\nu`. All
        inputs are pre-computed natural-log CGS quantities.

        Parameters
        ----------
        log_nu : ~numpy.ndarray
            Natural log of the observer-frame frequency in CGS (Hz).
        log_B : ~numpy.ndarray
            Natural log of the comoving-frame magnetic field in CGS (G).
        log_N : ~numpy.ndarray
            Natural log of the comoving-frame electron distribution.
        log_gamma : ~numpy.ndarray
            Natural log of the Lorentz factor grid.
        log_weights : ~numpy.ndarray
            Log of the quadrature weights.
        log_area_weights : ~numpy.ndarray
            Log of the annular projected area weights
            :math:`2\pi b_j\,\Delta b_j` in CGS (:math:`\mathrm{cm^2}`).
        log_D_A : float
            Natural log of the angular diameter distance in CGS (cm).
        z : float
            Cosmological redshift.
        half_path : ~numpy.ndarray
            Pre-computed half-path matrix.

        Returns
        -------
        log_dF : ~numpy.ndarray
            Log of the flux density contribution from each ray in
            :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}}`.
        """
        log_I = self._compute_log_pa_ray_specific_intensity(log_nu, log_B, log_N, log_gamma, log_weights, z, half_path)
        return log_I + log_area_weights - 2.0 * log_D_A

    # ------------------------------------------ #
    # Private API — Level 3: total F_nu         #
    # ------------------------------------------ #
    def _compute_log_pa_sphere_flux_density(
        self,
        log_nu: np.ndarray,
        log_B: np.ndarray,
        log_N: np.ndarray,
        log_gamma: np.ndarray,
        log_weights: np.ndarray,
        log_area_weights: np.ndarray,
        log_D_A: float,
        z: float,
        half_path: np.ndarray,
    ) -> np.ndarray:
        r"""
        Compute the pitch-angle-averaged log spectral flux density :math:`\ln F_\nu`.

        Sums the per-ray flux density contributions from
        :meth:`_compute_log_pa_ray_contributions` via
        :func:`~scipy.special.logsumexp`. This is the intended MCMC hot-loop
        entry point.

        Parameters
        ----------
        log_nu : ~numpy.ndarray
            Natural log of the observer-frame frequency in CGS (Hz).
        log_B : ~numpy.ndarray
            Natural log of the comoving-frame magnetic field in CGS (G).
        log_N : ~numpy.ndarray
            Natural log of the comoving-frame electron distribution.
        log_gamma : ~numpy.ndarray
            Natural log of the Lorentz factor grid.
        log_weights : ~numpy.ndarray
            Log of the quadrature weights.
        log_area_weights : ~numpy.ndarray
            Log of the annular projected area weights.
        log_D_A : float
            Natural log of the angular diameter distance in CGS (cm).
        z : float
            Cosmological redshift.
        half_path : ~numpy.ndarray
            Pre-computed half-path matrix.

        Returns
        -------
        log_F : ~numpy.ndarray
            Natural log of the observer-frame spectral flux density in
            :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}}`.
        """
        log_dF = self._compute_log_pa_ray_contributions(
            log_nu,
            log_B,
            log_N,
            log_gamma,
            log_weights,
            log_area_weights,
            log_D_A,
            z,
            half_path,
        )
        return logsumexp(log_dF, axis=-1)

    # ------------------------------------------ #
    # Shared input resolution                   #
    # ------------------------------------------ #
    def _resolve_inputs(
        self,
        nu,
        r,
        B,
        N,
        gamma,
        gamma_min,
        gamma_max,
        n_gamma,
        r_min,
        r_max,
        n_r,
        r_spacing,
        luminosity_distance,
        angular_diameter_distance,
        proper_distance,
        z,
        cosmology,
    ):
        """Shared unit-conversion and grid-building logic for both public methods."""
        nu_cgs = ensure_in_units(nu, u.Hz)
        B_cgs = ensure_in_units(B, u.G)
        r_cgs = None if r is None else ensure_in_units(r, u.cm)

        _, log_area_weights, half_path = self._build_sphere_grid(r_cgs, r_min, r_max, n_r, r_spacing)
        n_r_actual = log_area_weights.size

        log_gamma, log_weights = self._build_gamma_grid(gamma, gamma_min, gamma_max, n_gamma)
        log_N = self._resolve_log_N(N, log_gamma)

        if log_N.ndim == 1:
            log_N = np.broadcast_to(log_N[np.newaxis, :], (n_r_actual, log_gamma.size))

        dist = resolve_cosmological_distances(
            redshift=z if z != 0 else None,
            luminosity_distance=luminosity_distance,
            angular_diameter_distance=angular_diameter_distance,
            proper_distance=proper_distance,
            cosmology=cosmology,
        )
        log_D_A = np.log(dist["angular_diameter_distance"].to(u.cm).value)

        return (
            np.asarray(np.log(nu_cgs), dtype="f8"),
            np.log(np.asarray(B_cgs, dtype="f8")),
            log_N,
            log_gamma,
            log_weights,
            log_area_weights,
            log_D_A,
            half_path,
        )

    # ------------------------------------------ #
    # Public API (unit-bearing wrappers)        #
    # ------------------------------------------ #
    def compute_ray_specific_intensity(
        self,
        nu: Union[float, np.ndarray, u.Quantity],
        r: Union[np.ndarray, u.Quantity, None],
        B: Union[float, np.ndarray, u.Quantity],
        N: Union[np.ndarray, Callable],
        gamma: Union[np.ndarray, None] = None,
        luminosity_distance: Union[u.Quantity, None] = None,
        angular_diameter_distance: Union[u.Quantity, None] = None,
        proper_distance: Union[u.Quantity, None] = None,
        z: float = 0,
        cosmology=None,
        *,
        r_min: float = 1e14,
        r_max: float = 1e17,
        n_r: int = 50,
        r_spacing: str = "log",
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> u.Quantity:
        r"""
        Compute the area-weighted flux density contribution :math:`dF_{\nu,j}` for each ray.

        Returns the per-ray quantity

        .. math::

            dF_{\nu,j} = \frac{2\pi b_j\,\Delta b_j}{D_A^2}\,I_\nu(b_j),

        in units of :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}}`.
        Summing over the last (ray) axis recovers the total flux density:

        .. code-block:: python

            F_nu = engine.compute_ray_specific_intensity(
                ...
            ).sum(axis=-1)

        This method is intended for exploration and debugging: inspecting the
        per-ray contributions reveals which lines of sight dominate the emission
        and how the radial profile shapes the SED.

        .. note::

            This method enforces :math:`\beta = 0` and uses the pitch-angle-averaged
            kernel. See the class-level warnings for the implications of these
            restrictions.

        Parameters
        ----------
        nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Observer-frame frequency grid. Bare values
            are treated as Hz.
        r : ~numpy.ndarray, ~astropy.units.Quantity, or None
            Radial grid. Bare values are treated as cm. If
            ``None``, a grid is built from ``r_min``, ``r_max``, ``n_r``, and
            ``r_spacing``. The grid points serve as both shell centers and
            impact parameters.
        B : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame magnetic field :math:`B(r)` or
            scalar. Bare values are treated as Gauss.
        N : ~numpy.ndarray or callable
            Comoving-frame electron distribution :math:`N(\gamma, r)` in
            :math:`\mathrm{cm^{-3}}`. If callable,
            called as ``N(gamma)`` and broadcast to all shells if it returns
            shape ``(n_gamma,)``.
        gamma : ~numpy.ndarray or None, optional
            Explicit Lorentz factor grid. If ``None``, built from
            ``gamma_min``, ``gamma_max``, ``n_gamma``.
        luminosity_distance : ~astropy.units.Quantity or None, optional
            Source luminosity distance. Exactly one of ``luminosity_distance``,
            ``angular_diameter_distance``, ``proper_distance``, or a non-zero
            ``z`` must be provided.
        angular_diameter_distance : ~astropy.units.Quantity or None, optional
            Source angular diameter distance.
        proper_distance : ~astropy.units.Quantity or None, optional
            Source comoving line-of-sight distance.
        z : float, optional
            Cosmological redshift. Default ``0``.
        cosmology : ~astropy.cosmology.FLRW or None, optional
            Cosmological model for distance conversion.
        r_min : float, optional
            Radius grid lower bound in CGS (cm). Ignored if ``r`` is provided.
            Default ``1e14``.
        r_max : float, optional
            Radius grid upper bound in CGS (cm). Ignored if ``r`` is provided.
            Default ``1e17``.
        n_r : int, optional
            Number of radius grid points. Ignored if ``r`` is provided.
            Default ``50``.
        r_spacing : {"log", "linear"}, optional
            Radius grid spacing. Ignored if ``r`` is provided. Default ``"log"``.
        gamma_min : float, optional
            Gamma grid lower bound. Ignored if ``gamma`` is provided.
            Default ``1.0``.
        gamma_max : float, optional
            Gamma grid upper bound. Ignored if ``gamma`` is provided.
            Default ``1e8``.
        n_gamma : int, optional
            Number of gamma grid points. Ignored if ``gamma`` is provided.
            Default ``200``.

        Returns
        -------
        dF_nu : ~astropy.units.Quantity
            Per-ray flux density contributions in
            :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}}`. Summing over
            the last axis gives the total :math:`F_\nu`.
        """
        log_nu, log_B, log_N, log_gamma, log_weights, log_area_weights, log_D_A, half_path = self._resolve_inputs(
            nu,
            r,
            B,
            N,
            gamma,
            gamma_min,
            gamma_max,
            n_gamma,
            r_min,
            r_max,
            n_r,
            r_spacing,
            luminosity_distance,
            angular_diameter_distance,
            proper_distance,
            z,
            cosmology,
        )
        self.ensure_avg_first_kernel_loaded()
        log_dF = self._compute_log_pa_ray_contributions(
            log_nu,
            log_B,
            log_N,
            log_gamma,
            log_weights,
            log_area_weights,
            log_D_A,
            z,
            half_path,
        )
        return np.exp(log_dF) * (u.erg / (u.s * u.cm**2 * u.Hz))

    def compute_flux_density(
        self,
        nu: Union[float, np.ndarray, u.Quantity],
        r: Union[np.ndarray, u.Quantity, None],
        B: Union[float, np.ndarray, u.Quantity],
        N: Union[np.ndarray, Callable],
        gamma: Union[np.ndarray, None] = None,
        luminosity_distance: Union[u.Quantity, None] = None,
        angular_diameter_distance: Union[u.Quantity, None] = None,
        proper_distance: Union[u.Quantity, None] = None,
        z: float = 0,
        cosmology=None,
        *,
        r_min: float = 1e14,
        r_max: float = 1e17,
        n_r: int = 50,
        r_spacing: str = "log",
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> u.Quantity:
        r"""
        Compute the observer-frame spectral flux density :math:`F_\nu`.

        Integrates the per-ray specific intensity over the sphere's impact
        parameter profile:

        .. math::

            F_\nu \approx \sum_j \frac{2\pi b_j\,\Delta b_j}{D_A^2}\,I_\nu(b_j),

        where :math:`b_j = r_j` are the radial grid points, :math:`\Delta b_j`
        is the central-difference spacing at point :math:`j`, and :math:`D_A`
        is the angular diameter distance. The sum is evaluated in log-space via
        :func:`~scipy.special.logsumexp` for numerical stability.

        .. note::

            This method enforces :math:`\beta = 0` and uses the pitch-angle-averaged
            kernel. See the class-level warnings for the implications of these
            restrictions.

        Parameters
        ----------
        nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Observer-frame frequency grid. Bare values
            are treated as Hz.
        r : ~numpy.ndarray, ~astropy.units.Quantity, or None
            Radial grid. Bare values are treated as cm. If
            ``None``, a grid is built from ``r_min``, ``r_max``, ``n_r``, and
            ``r_spacing``.
        B : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame magnetic field :math:`B(r)` or
            scalar. Bare values are treated as Gauss.
        N : ~numpy.ndarray or callable
            Comoving-frame electron distribution :math:`N(\gamma, r)` in
            :math:`\mathrm{cm^{-3}}`. If callable,
            called as ``N(gamma)`` and broadcast to all shells.
        gamma : ~numpy.ndarray or None, optional
            Explicit Lorentz factor grid. If ``None``, built from
            ``gamma_min``, ``gamma_max``, ``n_gamma``.
        luminosity_distance : ~astropy.units.Quantity or None, optional
            Source luminosity distance. Exactly one of ``luminosity_distance``,
            ``angular_diameter_distance``, ``proper_distance``, or a non-zero
            ``z`` must be provided.
        angular_diameter_distance : ~astropy.units.Quantity or None, optional
            Source angular diameter distance.
        proper_distance : ~astropy.units.Quantity or None, optional
            Source comoving line-of-sight distance.
        z : float, optional
            Cosmological redshift. Default ``0``.
        cosmology : ~astropy.cosmology.FLRW or None, optional
            Cosmological model for distance conversion.
        r_min : float, optional
            Radius grid lower bound in CGS (cm). Ignored if ``r`` is provided.
            Default ``1e14``.
        r_max : float, optional
            Radius grid upper bound in CGS (cm). Ignored if ``r`` is provided.
            Default ``1e17``.
        n_r : int, optional
            Number of radius grid points. Ignored if ``r`` is provided.
            Default ``50``.
        r_spacing : {"log", "linear"}, optional
            Radius grid spacing. Ignored if ``r`` is provided. Default ``"log"``.
        gamma_min : float, optional
            Gamma grid lower bound. Ignored if ``gamma`` is provided.
            Default ``1.0``.
        gamma_max : float, optional
            Gamma grid upper bound. Ignored if ``gamma`` is provided.
            Default ``1e8``.
        n_gamma : int, optional
            Number of gamma grid points. Ignored if ``gamma`` is provided.
            Default ``200``.

        Returns
        -------
        F_nu : ~astropy.units.Quantity
            Observer-frame spectral flux density in
            :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}}`.
        """
        log_nu, log_B, log_N, log_gamma, log_weights, log_area_weights, log_D_A, half_path = self._resolve_inputs(
            nu,
            r,
            B,
            N,
            gamma,
            gamma_min,
            gamma_max,
            n_gamma,
            r_min,
            r_max,
            n_r,
            r_spacing,
            luminosity_distance,
            angular_diameter_distance,
            proper_distance,
            z,
            cosmology,
        )
        self.ensure_avg_first_kernel_loaded()
        log_F = self._compute_log_pa_sphere_flux_density(
            log_nu,
            log_B,
            log_N,
            log_gamma,
            log_weights,
            log_area_weights,
            log_D_A,
            z,
            half_path,
        )
        return np.exp(log_F) * (u.erg / (u.s * u.cm**2 * u.Hz))
