"""
Non-relativistic and ultra-relativistic spherical synchrotron engines.

Geometry-specializing wrappers around
:class:`~trilobite.radiation.synchrotron.SEDs.numerical.core.NumericalSynchrotronEngine`
for one-zone spherical emission regions. Two approximations are provided:

- :class:`NonRelativisticSphericalSynchrotronEngine`: isotropic expansion,
  bulk velocity approximately zero.
- :class:`UltraRelativisticSphericalSynchrotronEngine`: beamed-cap
  approximation for ultra-relativistic bulk Lorentz factors.

See Also
--------
:class:`~trilobite.radiation.synchrotron.SEDs.numerical.core.NumericalSynchrotronEngine`
    Base engine providing all low-level numerical synchrotron machinery.
"""

from collections.abc import Callable
from typing import Union

import numpy as np
from astropy import units as u

from trilobite.radiation.synchrotron.SEDs.numerical.core import NumericalSynchrotronEngine
from trilobite.utils.misc_utils import ensure_in_units


class NonRelativisticSphericalSynchrotronEngine(NumericalSynchrotronEngine):
    r"""
    Non-relativistic spherical synchrotron engine.

    A thin geometry-specializing wrapper around
    :class:`~trilobite.radiation.synchrotron.SEDs.numerical.core.NumericalSynchrotronEngine`
    for non-relativistic, spherically symmetric emission regions. Users supply a
    physical radius ``R`` and dimensionless volume and area filling factors ``f_V``
    and ``f_A`` in place of raw ``slab_depth`` and ``A_eff`` arguments. Bulk-motion
    parameters are fixed at zero and removed from every call signature.

    The effective radiative-transfer geometry is:

    .. math::

        A_\mathrm{eff} = \pi R^2 f_A, \qquad
        \ell_\mathrm{eff} = f_V^{1/3}\,R,

    where :math:`f_V` controls the emitting volume
    :math:`V = \tfrac{4}{3}\pi R^3 f_V` and :math:`f_A` modulates the
    projected emitting area.

    See Also
    --------
    :class:`~trilobite.radiation.synchrotron.SEDs.numerical.core.NumericalSynchrotronEngine`
        Base class providing all low-level numerical synchrotron machinery.
    """

    @staticmethod
    def _spherical_geometry(
        R: Union[float, u.Quantity],
        f_V: float,
        f_A: float,
    ):
        """Return ``(slab_depth_cgs, A_eff_cgs)`` from sphere parameters."""
        R_cgs = ensure_in_units(R, u.cm)
        slab_depth_cgs = f_V ** (1.0 / 3.0) * R_cgs
        A_eff_cgs = np.pi * R_cgs**2 * f_A
        return slab_depth_cgs, A_eff_cgs

    # ------------------------------------------------------------------ #
    # Public compute methods                                               #
    # ------------------------------------------------------------------ #

    def compute_specific_intensity(
        self,
        nu: Union[float, np.ndarray, u.Quantity],
        R: Union[float, u.Quantity],
        B: Union[float, np.ndarray, u.Quantity],
        N: Union[np.ndarray, Callable],
        gamma: Union[np.ndarray, None] = None,
        f_V: float = 1.0,
        z: float = 0,
        alpha: Union[float, u.Quantity, None] = None,
        *,
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> u.Quantity:
        r"""
        Compute the observer-frame specific intensity :math:`I_\nu` for a non-relativistic spherical source.

        Parameters
        ----------
        nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Observer-frame frequency grid. Bare values
            are treated as Hz.
        R : float or ~astropy.units.Quantity
            Physical radius of the emission region. Bare values are treated as
            cm.
        B : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame magnetic field strength. Bare values are treated as
            Gauss.
        N : ~numpy.ndarray or callable
            Electron distribution :math:`dN/d\gamma` in
            :math:`\mathrm{cm^{-3}}`. If callable, called as ``N(gamma)``.
        gamma : ~numpy.ndarray or None, optional
            Explicit Lorentz factor grid. If ``None``, built from
            ``gamma_min``, ``gamma_max``, ``n_gamma``.
        f_V : float, optional
            Volume filling factor. Sets the effective slab depth
            :math:`\ell_\mathrm{eff} = f_V^{1/3} R`. Default ``1.0``.
        z : float, optional
            Source redshift. Default ``0``.
        alpha : float, ~astropy.units.Quantity, or None, optional
            Pitch angle. If ``None``, the pitch-angle-averaged kernel is used.
        gamma_min : float, optional
            Grid lower bound. Default ``1.0``.
        gamma_max : float, optional
            Grid upper bound. Default ``1e8``.
        n_gamma : int, optional
            Number of grid points. Default ``200``.

        Returns
        -------
        I_nu : ~astropy.units.Quantity
            Observer-frame specific intensity in
            :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}\,sr^{-1}}`.
        """
        slab_depth, _ = self._spherical_geometry(R, f_V, 1.0)
        return super().compute_specific_intensity(
            nu,
            slab_depth,
            B,
            N,
            gamma=gamma,
            z=z,
            beta=0,
            theta=0,
            alpha=alpha,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            n_gamma=n_gamma,
        )

    def compute_flux_density(
        self,
        nu: Union[float, np.ndarray, u.Quantity],
        R: Union[float, u.Quantity],
        B: Union[float, np.ndarray, u.Quantity],
        N: Union[np.ndarray, Callable],
        gamma: Union[np.ndarray, None] = None,
        f_V: float = 1.0,
        f_A: float = 1.0,
        luminosity_distance: Union[u.Quantity, None] = None,
        angular_diameter_distance: Union[u.Quantity, None] = None,
        proper_distance: Union[u.Quantity, None] = None,
        z: float = 0,
        cosmology=None,
        alpha: Union[float, u.Quantity, None] = None,
        *,
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> u.Quantity:
        r"""
        Compute the observer-frame spectral flux density :math:`F_\nu` for a non-relativistic spherical source.

        Parameters
        ----------
        nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Observer-frame frequency grid. Bare values
            are treated as Hz.
        R : float or ~astropy.units.Quantity
            Physical radius of the emission region. Bare values are treated as
            cm.
        B : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame magnetic field strength. Bare values are treated as
            Gauss.
        N : ~numpy.ndarray or callable
            Electron distribution :math:`dN/d\gamma` in
            :math:`\mathrm{cm^{-3}}`. If callable, called as ``N(gamma)``.
        gamma : ~numpy.ndarray or None, optional
            Explicit Lorentz factor grid. If ``None``, built from
            ``gamma_min``, ``gamma_max``, ``n_gamma``.
        f_V : float, optional
            Volume filling factor; sets
            :math:`\ell_\mathrm{eff} = f_V^{1/3} R`. Default ``1.0``.
        f_A : float, optional
            Area filling factor; sets
            :math:`A_\mathrm{eff} = \pi R^2 f_A`. Default ``1.0``.
        luminosity_distance : ~astropy.units.Quantity or None, optional
            Source luminosity distance.
        angular_diameter_distance : ~astropy.units.Quantity or None, optional
            Source angular diameter distance.
        proper_distance : ~astropy.units.Quantity or None, optional
            Source comoving line-of-sight distance.
        z : float, optional
            Source redshift. Default ``0``.
        cosmology : ~astropy.cosmology.FLRW or None, optional
            Cosmology used to convert distances.
        alpha : float, ~astropy.units.Quantity, or None, optional
            Pitch angle. If ``None``, the pitch-angle-averaged kernel is used.
        gamma_min : float, optional
            Grid lower bound. Default ``1.0``.
        gamma_max : float, optional
            Grid upper bound. Default ``1e8``.
        n_gamma : int, optional
            Number of grid points. Default ``200``.

        Returns
        -------
        F_nu : ~astropy.units.Quantity
            Observer-frame spectral flux density in
            :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}}`.
        """
        slab_depth, A_eff = self._spherical_geometry(R, f_V, f_A)
        return super().compute_flux_density(
            nu,
            slab_depth,
            B,
            N,
            gamma=gamma,
            A_eff=A_eff,
            luminosity_distance=luminosity_distance,
            angular_diameter_distance=angular_diameter_distance,
            proper_distance=proper_distance,
            z=z,
            cosmology=cosmology,
            beta=0,
            theta=0,
            alpha=alpha,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            n_gamma=n_gamma,
        )

    def compute_brightness_temperature(
        self,
        nu: Union[float, np.ndarray, u.Quantity],
        R: Union[float, u.Quantity],
        B: Union[float, np.ndarray, u.Quantity],
        N: Union[np.ndarray, Callable],
        gamma: Union[np.ndarray, None] = None,
        f_V: float = 1.0,
        z: float = 0,
        alpha: Union[float, u.Quantity, None] = None,
        *,
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> u.Quantity:
        r"""
        Compute the observer-frame brightness temperature :math:`T_B` for a non-relativistic spherical source.

        Parameters
        ----------
        nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Observer-frame frequency grid. Bare values are treated as Hz.
        R : float or ~astropy.units.Quantity
            Physical radius of the emission region. Bare values are treated as
            cm.
        B : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame magnetic field strength. Bare values are treated as
            Gauss.
        N : ~numpy.ndarray or callable
            Electron distribution :math:`dN/d\gamma` in
            :math:`\mathrm{cm^{-3}}`.
        gamma : ~numpy.ndarray or None, optional
            Explicit Lorentz factor grid.
        f_V : float, optional
            Volume filling factor; sets
            :math:`\ell_\mathrm{eff} = f_V^{1/3} R`. Default ``1.0``.
        z : float, optional
            Source redshift. Default ``0``.
        alpha : float, ~astropy.units.Quantity, or None, optional
            Pitch angle. If ``None``, pitch-angle-averaged kernel is used.
        gamma_min : float, optional
            Grid lower bound. Default ``1.0``.
        gamma_max : float, optional
            Grid upper bound. Default ``1e8``.
        n_gamma : int, optional
            Number of grid points. Default ``200``.

        Returns
        -------
        T_B : ~astropy.units.Quantity
            Observer-frame brightness temperature in Kelvin.
        """
        slab_depth, _ = self._spherical_geometry(R, f_V, 1.0)
        return super().compute_brightness_temperature(
            nu,
            slab_depth,
            B,
            N,
            gamma=gamma,
            z=z,
            beta=0,
            theta=0,
            alpha=alpha,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            n_gamma=n_gamma,
        )

    def compute_rest_frame_luminosity_density(
        self,
        nu: Union[float, np.ndarray, u.Quantity],
        R: Union[float, u.Quantity],
        B: Union[float, np.ndarray, u.Quantity],
        N: Union[np.ndarray, Callable],
        gamma: Union[np.ndarray, None] = None,
        f_V: float = 1.0,
        f_A: float = 1.0,
        alpha: Union[float, u.Quantity, None] = None,
        *,
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> u.Quantity:
        r"""
        Compute the comoving-frame spectral luminosity density :math:`L'_{\nu'}` for a NR spherical source.

        Parameters
        ----------
        nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame frequency grid. Bare values are treated as Hz.
        R : float or ~astropy.units.Quantity
            Physical radius of the emission region. Bare values are treated as
            cm.
        B : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame magnetic field strength. Bare values are treated as
            Gauss.
        N : ~numpy.ndarray or callable
            Electron distribution :math:`dN/d\gamma` in
            :math:`\mathrm{cm^{-3}}`.
        gamma : ~numpy.ndarray or None, optional
            Explicit Lorentz factor grid.
        f_V : float, optional
            Volume filling factor; sets
            :math:`\ell_\mathrm{eff} = f_V^{1/3} R`. Default ``1.0``.
        f_A : float, optional
            Area filling factor; sets
            :math:`A_\mathrm{eff} = \pi R^2 f_A`. Default ``1.0``.
        alpha : float, ~astropy.units.Quantity, or None, optional
            Pitch angle. If ``None``, pitch-angle-averaged kernel is used.
        gamma_min : float, optional
            Grid lower bound. Default ``1.0``.
        gamma_max : float, optional
            Grid upper bound. Default ``1e8``.
        n_gamma : int, optional
            Number of grid points. Default ``200``.

        Returns
        -------
        L_nu : ~astropy.units.Quantity
            Comoving-frame spectral luminosity density in
            :math:`\mathrm{erg\,s^{-1}\,Hz^{-1}}`.
        """
        slab_depth, A_eff = self._spherical_geometry(R, f_V, f_A)
        return super().compute_rest_frame_luminosity_density(
            nu,
            slab_depth,
            B,
            N,
            gamma=gamma,
            A_eff=A_eff,
            alpha=alpha,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            n_gamma=n_gamma,
        )

    def compute_luminosity_density(
        self,
        nu: Union[float, np.ndarray, u.Quantity],
        R: Union[float, u.Quantity],
        B: Union[float, np.ndarray, u.Quantity],
        N: Union[np.ndarray, Callable],
        gamma: Union[np.ndarray, None] = None,
        f_V: float = 1.0,
        f_A: float = 1.0,
        z: float = 0,
        alpha: Union[float, u.Quantity, None] = None,
        *,
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> u.Quantity:
        r"""
        Compute the observer-frame spectral luminosity density :math:`L_\nu` for a non-relativistic spherical source.

        Parameters
        ----------
        nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Observer-frame frequency grid. Bare values are treated as Hz.
        R : float or ~astropy.units.Quantity
            Physical radius of the emission region. Bare values are treated as
            cm.
        B : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame magnetic field strength. Bare values are treated as
            Gauss.
        N : ~numpy.ndarray or callable
            Electron distribution :math:`dN/d\gamma` in
            :math:`\mathrm{cm^{-3}}`.
        gamma : ~numpy.ndarray or None, optional
            Explicit Lorentz factor grid.
        f_V : float, optional
            Volume filling factor; sets
            :math:`\ell_\mathrm{eff} = f_V^{1/3} R`. Default ``1.0``.
        f_A : float, optional
            Area filling factor; sets
            :math:`A_\mathrm{eff} = \pi R^2 f_A`. Default ``1.0``.
        z : float, optional
            Source redshift. Default ``0``.
        alpha : float, ~astropy.units.Quantity, or None, optional
            Pitch angle. If ``None``, pitch-angle-averaged kernel is used.
        gamma_min : float, optional
            Grid lower bound. Default ``1.0``.
        gamma_max : float, optional
            Grid upper bound. Default ``1e8``.
        n_gamma : int, optional
            Number of grid points. Default ``200``.

        Returns
        -------
        L_nu : ~astropy.units.Quantity
            Observer-frame spectral luminosity density in
            :math:`\mathrm{erg\,s^{-1}\,Hz^{-1}}`.
        """
        slab_depth, A_eff = self._spherical_geometry(R, f_V, f_A)
        return super().compute_luminosity_density(
            nu,
            slab_depth,
            B,
            N,
            gamma=gamma,
            A_eff=A_eff,
            z=z,
            beta=0,
            theta=0,
            alpha=alpha,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            n_gamma=n_gamma,
        )

    def compute_isotropic_luminosity(
        self,
        nu: Union[float, np.ndarray, u.Quantity],
        R: Union[float, u.Quantity],
        B: Union[float, np.ndarray, u.Quantity],
        N: Union[np.ndarray, Callable],
        gamma: Union[np.ndarray, None] = None,
        f_V: float = 1.0,
        f_A: float = 1.0,
        luminosity_distance: Union[u.Quantity, None] = None,
        angular_diameter_distance: Union[u.Quantity, None] = None,
        proper_distance: Union[u.Quantity, None] = None,
        z: float = 0,
        cosmology=None,
        alpha: Union[float, u.Quantity, None] = None,
        *,
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> u.Quantity:
        r"""
        Compute the isotropic-equivalent spectral luminosity :math:`L_{\nu,\mathrm{iso}}` for a NR spherical source.

        Parameters
        ----------
        nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Observer-frame frequency grid. Bare values are treated as Hz.
        R : float or ~astropy.units.Quantity
            Physical radius of the emission region. Bare values are treated as
            cm.
        B : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame magnetic field strength. Bare values are treated as
            Gauss.
        N : ~numpy.ndarray or callable
            Electron distribution :math:`dN/d\gamma` in
            :math:`\mathrm{cm^{-3}}`.
        gamma : ~numpy.ndarray or None, optional
            Explicit Lorentz factor grid.
        f_V : float, optional
            Volume filling factor; sets
            :math:`\ell_\mathrm{eff} = f_V^{1/3} R`. Default ``1.0``.
        f_A : float, optional
            Area filling factor; sets
            :math:`A_\mathrm{eff} = \pi R^2 f_A`. Default ``1.0``.
        luminosity_distance : ~astropy.units.Quantity or None, optional
            Source luminosity distance.
        angular_diameter_distance : ~astropy.units.Quantity or None, optional
            Source angular diameter distance.
        proper_distance : ~astropy.units.Quantity or None, optional
            Source comoving line-of-sight distance.
        z : float, optional
            Source redshift. Default ``0``.
        cosmology : ~astropy.cosmology.FLRW or None, optional
            Cosmology used to convert distances.
        alpha : float, ~astropy.units.Quantity, or None, optional
            Pitch angle. If ``None``, pitch-angle-averaged kernel is used.
        gamma_min : float, optional
            Grid lower bound. Default ``1.0``.
        gamma_max : float, optional
            Grid upper bound. Default ``1e8``.
        n_gamma : int, optional
            Number of grid points. Default ``200``.

        Returns
        -------
        L_iso : ~astropy.units.Quantity
            Isotropic-equivalent spectral luminosity in
            :math:`\mathrm{erg\,s^{-1}\,Hz^{-1}}`.
        """
        slab_depth, A_eff = self._spherical_geometry(R, f_V, f_A)
        return super().compute_isotropic_luminosity(
            nu,
            slab_depth,
            B,
            N,
            gamma=gamma,
            A_eff=A_eff,
            luminosity_distance=luminosity_distance,
            angular_diameter_distance=angular_diameter_distance,
            proper_distance=proper_distance,
            z=z,
            cosmology=cosmology,
            beta=0,
            theta=0,
            alpha=alpha,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            n_gamma=n_gamma,
        )


class UltraRelativisticSphericalSynchrotronEngine(NumericalSynchrotronEngine):
    r"""
    Ultra-relativistic spherical synchrotron engine (beamed-cap approximation).

    A thin geometry-specializing wrapper around
    :class:`~trilobite.radiation.synchrotron.SEDs.numerical.core.NumericalSynchrotronEngine`
    for ultra-relativistic, spherically expanding outflows observed on-axis.
    Equal-arrival-time-surface (EATS) corrections are neglected; only the
    relativistically beamed cap contributes.

    The bulk Lorentz factor :math:`\Gamma` sets both the apparent emitting geometry
    and the comoving slab depth via

    .. math::

        A_\mathrm{eff} = \frac{\pi R^2 f_A}{\Gamma^2}, \qquad
        \ell' = \frac{f_V\,R}{\Gamma},

    and the bulk velocity is derived self-consistently as
    :math:`\beta = \sqrt{1 - \Gamma^{-2}}`. The line-of-sight angle is fixed at
    :math:`\theta = 0` (on-axis observer).

    See Also
    --------
    :class:`~trilobite.radiation.synchrotron.SEDs.numerical.core.NumericalSynchrotronEngine`
        Base class providing all low-level numerical synchrotron machinery.
    """

    @staticmethod
    def _ultra_rel_geometry(
        R: Union[float, u.Quantity],
        Gamma: float,
        f_V: float,
        f_A: float,
    ):
        """Return ``(slab_depth_cgs, A_eff_cgs, beta)`` from blast-wave parameters."""
        R_cgs = ensure_in_units(R, u.cm)
        slab_depth_cgs = f_V * R_cgs / Gamma
        A_eff_cgs = np.pi * R_cgs**2 * f_A / Gamma**2
        beta = float(np.sqrt(1.0 - 1.0 / Gamma**2))
        return slab_depth_cgs, A_eff_cgs, beta

    # ------------------------------------------------------------------ #
    # Public compute methods                                               #
    # ------------------------------------------------------------------ #

    def compute_specific_intensity(
        self,
        nu: Union[float, np.ndarray, u.Quantity],
        R: Union[float, u.Quantity],
        B: Union[float, np.ndarray, u.Quantity],
        N: Union[np.ndarray, Callable],
        Gamma: float,
        gamma: Union[np.ndarray, None] = None,
        f_V: float = 1.0,
        z: float = 0,
        alpha: Union[float, u.Quantity, None] = None,
        *,
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> u.Quantity:
        r"""
        Compute the observer-frame specific intensity :math:`I_\nu` for an UR spherical source.

        Parameters
        ----------
        nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Observer-frame frequency grid. Bare values are treated as Hz.
        R : float or ~astropy.units.Quantity
            Lab-frame radius of the blast wave. Bare values are treated as cm.
        B : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame magnetic field strength. Bare values are treated as
            Gauss.
        N : ~numpy.ndarray or callable
            Electron distribution :math:`dN/d\gamma` in
            :math:`\mathrm{cm^{-3}}`.
        Gamma : float
            Bulk Lorentz factor :math:`\Gamma \geq 1`.
        gamma : ~numpy.ndarray or None, optional
            Explicit electron Lorentz factor grid.
        f_V : float, optional
            Volume filling factor; sets
            :math:`\ell' = f_V R / \Gamma`. Default ``1.0``.
        z : float, optional
            Source redshift. Default ``0``.
        alpha : float, ~astropy.units.Quantity, or None, optional
            Pitch angle. If ``None``, pitch-angle-averaged kernel is used.
        gamma_min : float, optional
            Grid lower bound. Default ``1.0``.
        gamma_max : float, optional
            Grid upper bound. Default ``1e8``.
        n_gamma : int, optional
            Number of grid points. Default ``200``.

        Returns
        -------
        I_nu : ~astropy.units.Quantity
            Observer-frame specific intensity in
            :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}\,sr^{-1}}`.
        """
        slab_depth, _, beta = self._ultra_rel_geometry(R, Gamma, f_V, 1.0)
        return super().compute_specific_intensity(
            nu,
            slab_depth,
            B,
            N,
            gamma=gamma,
            z=z,
            beta=beta,
            theta=0,
            alpha=alpha,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            n_gamma=n_gamma,
        )

    def compute_flux_density(
        self,
        nu: Union[float, np.ndarray, u.Quantity],
        R: Union[float, u.Quantity],
        B: Union[float, np.ndarray, u.Quantity],
        N: Union[np.ndarray, Callable],
        Gamma: float,
        gamma: Union[np.ndarray, None] = None,
        f_V: float = 1.0,
        f_A: float = 1.0,
        luminosity_distance: Union[u.Quantity, None] = None,
        angular_diameter_distance: Union[u.Quantity, None] = None,
        proper_distance: Union[u.Quantity, None] = None,
        z: float = 0,
        cosmology=None,
        alpha: Union[float, u.Quantity, None] = None,
        *,
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> u.Quantity:
        r"""
        Compute the observer-frame spectral flux density :math:`F_\nu` for an UR spherical source.

        Parameters
        ----------
        nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Observer-frame frequency grid. Bare values are treated as Hz.
        R : float or ~astropy.units.Quantity
            Lab-frame radius of the blast wave. Bare values are treated as cm.
        B : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame magnetic field strength. Bare values are treated as
            Gauss.
        N : ~numpy.ndarray or callable
            Electron distribution :math:`dN/d\gamma` in
            :math:`\mathrm{cm^{-3}}`.
        Gamma : float
            Bulk Lorentz factor :math:`\Gamma \geq 1`.
        gamma : ~numpy.ndarray or None, optional
            Explicit electron Lorentz factor grid.
        f_V : float, optional
            Volume filling factor; sets
            :math:`\ell' = f_V R / \Gamma`. Default ``1.0``.
        f_A : float, optional
            Area filling factor; sets
            :math:`A_\mathrm{eff} = \pi R^2 f_A / \Gamma^2`. Default ``1.0``.
        luminosity_distance : ~astropy.units.Quantity or None, optional
            Source luminosity distance.
        angular_diameter_distance : ~astropy.units.Quantity or None, optional
            Source angular diameter distance.
        proper_distance : ~astropy.units.Quantity or None, optional
            Source comoving line-of-sight distance.
        z : float, optional
            Source redshift. Default ``0``.
        cosmology : ~astropy.cosmology.FLRW or None, optional
            Cosmology used to convert distances.
        alpha : float, ~astropy.units.Quantity, or None, optional
            Pitch angle. If ``None``, pitch-angle-averaged kernel is used.
        gamma_min : float, optional
            Grid lower bound. Default ``1.0``.
        gamma_max : float, optional
            Grid upper bound. Default ``1e8``.
        n_gamma : int, optional
            Number of grid points. Default ``200``.

        Returns
        -------
        F_nu : ~astropy.units.Quantity
            Observer-frame spectral flux density in
            :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}}`.
        """
        slab_depth, A_eff, beta = self._ultra_rel_geometry(R, Gamma, f_V, f_A)
        return super().compute_flux_density(
            nu,
            slab_depth,
            B,
            N,
            gamma=gamma,
            A_eff=A_eff,
            luminosity_distance=luminosity_distance,
            angular_diameter_distance=angular_diameter_distance,
            proper_distance=proper_distance,
            z=z,
            cosmology=cosmology,
            beta=beta,
            theta=0,
            alpha=alpha,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            n_gamma=n_gamma,
        )

    def compute_brightness_temperature(
        self,
        nu: Union[float, np.ndarray, u.Quantity],
        R: Union[float, u.Quantity],
        B: Union[float, np.ndarray, u.Quantity],
        N: Union[np.ndarray, Callable],
        Gamma: float,
        gamma: Union[np.ndarray, None] = None,
        f_V: float = 1.0,
        z: float = 0,
        alpha: Union[float, u.Quantity, None] = None,
        *,
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> u.Quantity:
        r"""
        Compute the observer-frame brightness temperature :math:`T_B` for an UR spherical source.

        Parameters
        ----------
        nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Observer-frame frequency grid. Bare values are treated as Hz.
        R : float or ~astropy.units.Quantity
            Lab-frame radius of the blast wave. Bare values are treated as cm.
        B : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame magnetic field strength. Bare values are treated as
            Gauss.
        N : ~numpy.ndarray or callable
            Electron distribution :math:`dN/d\gamma` in
            :math:`\mathrm{cm^{-3}}`.
        Gamma : float
            Bulk Lorentz factor :math:`\Gamma \geq 1`.
        gamma : ~numpy.ndarray or None, optional
            Explicit electron Lorentz factor grid.
        f_V : float, optional
            Volume filling factor; sets
            :math:`\ell' = f_V R / \Gamma`. Default ``1.0``.
        z : float, optional
            Source redshift. Default ``0``.
        alpha : float, ~astropy.units.Quantity, or None, optional
            Pitch angle. If ``None``, pitch-angle-averaged kernel is used.
        gamma_min : float, optional
            Grid lower bound. Default ``1.0``.
        gamma_max : float, optional
            Grid upper bound. Default ``1e8``.
        n_gamma : int, optional
            Number of grid points. Default ``200``.

        Returns
        -------
        T_B : ~astropy.units.Quantity
            Observer-frame brightness temperature in Kelvin.
        """
        slab_depth, _, beta = self._ultra_rel_geometry(R, Gamma, f_V, 1.0)
        return super().compute_brightness_temperature(
            nu,
            slab_depth,
            B,
            N,
            gamma=gamma,
            z=z,
            beta=beta,
            theta=0,
            alpha=alpha,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            n_gamma=n_gamma,
        )

    def compute_rest_frame_luminosity_density(
        self,
        nu: Union[float, np.ndarray, u.Quantity],
        R: Union[float, u.Quantity],
        B: Union[float, np.ndarray, u.Quantity],
        N: Union[np.ndarray, Callable],
        Gamma: float,
        gamma: Union[np.ndarray, None] = None,
        f_V: float = 1.0,
        f_A: float = 1.0,
        alpha: Union[float, u.Quantity, None] = None,
        *,
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> u.Quantity:
        r"""
        Compute the comoving-frame spectral luminosity density :math:`L'_{\nu'}` for an UR spherical source.

        Parameters
        ----------
        nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame frequency grid. Bare values are treated as Hz.
        R : float or ~astropy.units.Quantity
            Lab-frame radius of the blast wave. Bare values are treated as cm.
        B : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame magnetic field strength. Bare values are treated as
            Gauss.
        N : ~numpy.ndarray or callable
            Electron distribution :math:`dN/d\gamma` in
            :math:`\mathrm{cm^{-3}}`.
        Gamma : float
            Bulk Lorentz factor :math:`\Gamma \geq 1`.
        gamma : ~numpy.ndarray or None, optional
            Explicit electron Lorentz factor grid.
        f_V : float, optional
            Volume filling factor; sets
            :math:`\ell' = f_V R / \Gamma`. Default ``1.0``.
        f_A : float, optional
            Area filling factor; sets
            :math:`A_\mathrm{eff} = \pi R^2 f_A / \Gamma^2`. Default ``1.0``.
        alpha : float, ~astropy.units.Quantity, or None, optional
            Pitch angle. If ``None``, pitch-angle-averaged kernel is used.
        gamma_min : float, optional
            Grid lower bound. Default ``1.0``.
        gamma_max : float, optional
            Grid upper bound. Default ``1e8``.
        n_gamma : int, optional
            Number of grid points. Default ``200``.

        Returns
        -------
        L_nu : ~astropy.units.Quantity
            Comoving-frame spectral luminosity density in
            :math:`\mathrm{erg\,s^{-1}\,Hz^{-1}}`.
        """
        slab_depth, A_eff, _ = self._ultra_rel_geometry(R, Gamma, f_V, f_A)
        return super().compute_rest_frame_luminosity_density(
            nu,
            slab_depth,
            B,
            N,
            gamma=gamma,
            A_eff=A_eff,
            alpha=alpha,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            n_gamma=n_gamma,
        )

    def compute_luminosity_density(
        self,
        nu: Union[float, np.ndarray, u.Quantity],
        R: Union[float, u.Quantity],
        B: Union[float, np.ndarray, u.Quantity],
        N: Union[np.ndarray, Callable],
        Gamma: float,
        gamma: Union[np.ndarray, None] = None,
        f_V: float = 1.0,
        f_A: float = 1.0,
        z: float = 0,
        alpha: Union[float, u.Quantity, None] = None,
        *,
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> u.Quantity:
        r"""
        Compute the observer-frame spectral luminosity density :math:`L_\nu` for an ultra-relativistic spherical source.

        Parameters
        ----------
        nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Observer-frame frequency grid. Bare values are treated as Hz.
        R : float or ~astropy.units.Quantity
            Lab-frame radius of the blast wave. Bare values are treated as cm.
        B : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame magnetic field strength. Bare values are treated as
            Gauss.
        N : ~numpy.ndarray or callable
            Electron distribution :math:`dN/d\gamma` in
            :math:`\mathrm{cm^{-3}}`.
        Gamma : float
            Bulk Lorentz factor :math:`\Gamma \geq 1`.
        gamma : ~numpy.ndarray or None, optional
            Explicit electron Lorentz factor grid.
        f_V : float, optional
            Volume filling factor; sets
            :math:`\ell' = f_V R / \Gamma`. Default ``1.0``.
        f_A : float, optional
            Area filling factor; sets
            :math:`A_\mathrm{eff} = \pi R^2 f_A / \Gamma^2`. Default ``1.0``.
        z : float, optional
            Source redshift. Default ``0``.
        alpha : float, ~astropy.units.Quantity, or None, optional
            Pitch angle. If ``None``, pitch-angle-averaged kernel is used.
        gamma_min : float, optional
            Grid lower bound. Default ``1.0``.
        gamma_max : float, optional
            Grid upper bound. Default ``1e8``.
        n_gamma : int, optional
            Number of grid points. Default ``200``.

        Returns
        -------
        L_nu : ~astropy.units.Quantity
            Observer-frame spectral luminosity density in
            :math:`\mathrm{erg\,s^{-1}\,Hz^{-1}}`.
        """
        slab_depth, A_eff, beta = self._ultra_rel_geometry(R, Gamma, f_V, f_A)
        return super().compute_luminosity_density(
            nu,
            slab_depth,
            B,
            N,
            gamma=gamma,
            A_eff=A_eff,
            z=z,
            beta=beta,
            theta=0,
            alpha=alpha,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            n_gamma=n_gamma,
        )

    def compute_isotropic_luminosity(
        self,
        nu: Union[float, np.ndarray, u.Quantity],
        R: Union[float, u.Quantity],
        B: Union[float, np.ndarray, u.Quantity],
        N: Union[np.ndarray, Callable],
        Gamma: float,
        gamma: Union[np.ndarray, None] = None,
        f_V: float = 1.0,
        f_A: float = 1.0,
        luminosity_distance: Union[u.Quantity, None] = None,
        angular_diameter_distance: Union[u.Quantity, None] = None,
        proper_distance: Union[u.Quantity, None] = None,
        z: float = 0,
        cosmology=None,
        alpha: Union[float, u.Quantity, None] = None,
        *,
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> u.Quantity:
        r"""
        Compute the spectral luminosity :math:`L_{\nu,\mathrm{iso}}` for an ultra-relativistic spherical source.

        Parameters
        ----------
        nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Observer-frame frequency grid. Bare values are treated as Hz.
        R : float or ~astropy.units.Quantity
            Lab-frame radius of the blast wave. Bare values are treated as cm.
        B : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame magnetic field strength. Bare values are treated as
            Gauss.
        N : ~numpy.ndarray or callable
            Electron distribution :math:`dN/d\gamma` in
            :math:`\mathrm{cm^{-3}}`.
        Gamma : float
            Bulk Lorentz factor :math:`\Gamma \geq 1`.
        gamma : ~numpy.ndarray or None, optional
            Explicit electron Lorentz factor grid.
        f_V : float, optional
            Volume filling factor; sets
            :math:`\ell' = f_V R / \Gamma`. Default ``1.0``.
        f_A : float, optional
            Area filling factor; sets
            :math:`A_\mathrm{eff} = \pi R^2 f_A / \Gamma^2`. Default ``1.0``.
        luminosity_distance : ~astropy.units.Quantity or None, optional
            Source luminosity distance.
        angular_diameter_distance : ~astropy.units.Quantity or None, optional
            Source angular diameter distance.
        proper_distance : ~astropy.units.Quantity or None, optional
            Source comoving line-of-sight distance.
        z : float, optional
            Source redshift. Default ``0``.
        cosmology : ~astropy.cosmology.FLRW or None, optional
            Cosmology used to convert distances.
        alpha : float, ~astropy.units.Quantity, or None, optional
            Pitch angle. If ``None``, pitch-angle-averaged kernel is used.
        gamma_min : float, optional
            Grid lower bound. Default ``1.0``.
        gamma_max : float, optional
            Grid upper bound. Default ``1e8``.
        n_gamma : int, optional
            Number of grid points. Default ``200``.

        Returns
        -------
        L_iso : ~astropy.units.Quantity
            Isotropic-equivalent spectral luminosity in
            :math:`\mathrm{erg\,s^{-1}\,Hz^{-1}}`.
        """
        slab_depth, A_eff, beta = self._ultra_rel_geometry(R, Gamma, f_V, f_A)
        return super().compute_isotropic_luminosity(
            nu,
            slab_depth,
            B,
            N,
            gamma=gamma,
            A_eff=A_eff,
            luminosity_distance=luminosity_distance,
            angular_diameter_distance=angular_diameter_distance,
            proper_distance=proper_distance,
            z=z,
            cosmology=cosmology,
            beta=beta,
            theta=0,
            alpha=alpha,
            gamma_min=gamma_min,
            gamma_max=gamma_max,
            n_gamma=n_gamma,
        )
