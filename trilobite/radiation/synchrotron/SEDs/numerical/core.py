r"""
Numerical synchrotron SED engines for complex emission scenarios.

This module provides numerical engines for efficient computation of synchrotron
radiation in regimes where analytical synchrotron prescriptions are either
inaccurate, overly restrictive, or entirely inapplicable. These engines perform
direct numerical integration over the synchrotron kernel and therefore support
arbitrary electron energy distributions, relativistic bulk motion, synchrotron
self-absorption, and non-trivial source geometries.

Unlike the analytic one-zone synchrotron models provided elsewhere in
:mod:`trilobite.radiation.synchrotron.SEDs`, the numerical engines are designed
to operate on general distributions :math:`N(\gamma)` and arbitrary emitting
configurations. This includes, for example:

- cooling-modified electron populations,
- thermal or hybrid distributions,
- multi-zone outflows,
- structured jets and angular profiles,
- relativistically moving emission regions,
- and spatially varying magnetic fields or transfer depths.

The engines rely on pre-computed spline interpolators for the synchrotron kernel
and its logarithmic derivative, allowing expensive Bessel-function integrations
to be amortized across many spectral evaluations. Radiative quantities are
evaluated primarily in log-space for numerical stability and efficient vectorized
execution over large frequency and Lorentz-factor grids.

Both fixed-pitch-angle and pitch-angle-averaged synchrotron formalisms are
supported. Synchrotron self-absorption is computed using a numerically stable
integration-by-parts formulation that avoids explicitly differentiating the
electron distribution.

Observed-frame quantities are generated from comoving-frame radiative transfer
solutions through relativistic Doppler boosting and cosmological redshift
transformations. The resulting architecture is suitable for both simple one-zone
models and more sophisticated multi-zone synchrotron calculations.

For details on the implementation and underlying numerical methods, see
:ref:`synch_numerical_sed_theory`. For the user guide and high-level API
documentation, see :ref:`synch_numerical_sed`.

See Also
--------
:mod:`trilobite.radiation.synchrotron.SEDs.one_zone`
    Analytic synchrotron SED models for idealized one-zone electron populations.

:mod:`trilobite.radiation.synchrotron.core`
    Core synchrotron kernel implementations and asymptotic approximations.

:ref:`synch_numerical_sed_theory`
    Detailed discussion of the numerical synchrotron formalism implemented by
    these engines.

:ref:`synch_numerical_sed`
    User guide and examples for the numerical synchrotron SED API.
"""

from collections.abc import Callable
from typing import Union

import numpy as np
from astropy import constants as consts
from astropy import units as u
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.special import logsumexp

from trilobite.physics_utils import resolve_cosmological_distances
from trilobite.radiation.synchrotron.core import _log_averaged_first_synchrotron_kernel, _log_first_synchrotron_kernel
from trilobite.radiation.synchrotron.utils import _log_c_1_gamma_cgs, _log_chi_abs_cgs, _log_chi_cgs
from trilobite.utils.misc_utils import ensure_in_units

# ================================================ #
# CGS Constants                                    #
# ================================================ #
_log_c_cgs = np.log(consts.c.cgs.value)
_log_k_B_cgs = np.log(consts.k_B.cgs.value)
_FLOAT64_TINY = np.finfo("f8").tiny
_LOG2 = np.log(2.0)
_LOG_4PI = np.log(4.0 * np.pi)


# ================================================ #
# Core Synchrotron Engine Class                    #
# ================================================ #
class NumericalSynchrotronEngine:
    r"""
    Numerical synchrotron engine base class.

    This class provides the foundational numerical machinery used throughout the
    Trilobite synchrotron framework for evaluating synchrotron emissivities,
    absorption coefficients, specific intensities, and spectral flux densities
    from arbitrary electron energy distributions.

    Unlike analytic synchrotron prescriptions that assume idealized power-law or
    thermal populations, :class:`NumericalSynchrotronEngine` performs direct
    numerical integration over the synchrotron kernel and therefore supports
    completely general electron distributions :math:`N(\gamma)`. This includes
    tabulated distributions, cooling-modified populations, hybrid thermal/nonthermal
    models, and arbitrary user-defined callable distributions.

    The engine operates primarily in log-space to maintain numerical stability over
    the extremely large dynamic ranges encountered in synchrotron calculations.
    Internally, synchrotron kernel functions and their logarithmic derivatives are
    precomputed on interpolation grids and cached as spline interpolators. This
    allows expensive Bessel-function integrations to be amortized over many
    subsequent spectral evaluations, dramatically improving performance in iterative
    applications such as MCMC and nested sampling.

    Two synchrotron formalisms are supported:

    - fixed pitch-angle synchrotron emission using the kernel

      .. math::

          F(x) = x \int_x^\infty K_{5/3}(y)\,dy,

    - and pitch-angle-averaged synchrotron emission using the averaged kernel
      :math:`\bar{F}(x)` appropriate for isotropic electron distributions.

    The engine additionally implements synchrotron self-absorption using an
    integration-by-parts formulation of the absorption coefficient. This avoids
    explicit differentiation of the electron distribution and instead rewrites the
    absorption kernel in terms of the logarithmic derivative of the synchrotron
    kernel itself. The resulting formulation is numerically stable, sign-definite,
    and naturally compatible with log-space quadrature.

    Radiative transfer is solved in the comoving frame through

    .. math::

        I_\nu = S_\nu\left(1 - e^{-\tau_\nu}\right),

    where :math:`S_\nu = j_\nu/\alpha_\nu` is the source function and
    :math:`\tau_\nu = \alpha_\nu \ell_\mathrm{eff}` is the optical depth.

    Observed-frame quantities are obtained through relativistic Doppler boosting
    and cosmological redshift transformations using the Lorentz invariance of

    .. math::

        \frac{I_\nu}{\nu^3}.

    The engine is heavily vectorized and designed to broadcast efficiently across:

    - frequency grids,
    - Lorentz-factor grids,
    - batches of independent electron populations,
    - magnetic field arrays,
    - and relativistic outflow parameters.

    This architecture allows the same low-level engine to power both simple
    one-zone synchrotron calculations and more sophisticated multi-zone or
    structured outflow models.

    See Also
    --------
    :mod:`trilobite.radiation.synchrotron.core`
        Low-level synchrotron kernel implementations and asymptotic approximations.

    :mod:`trilobite.radiation.synchrotron.SEDs.one_zone`
        Analytic one-zone synchrotron SED models.

    scipy.interpolate.InterpolatedUnivariateSpline
        Spline interpolator used for synchrotron kernel tabulation.

    Notes
    -----
    All local radiative quantities are assumed to be defined in the comoving frame
    of the emitting plasma. In particular:

    - magnetic field strengths :math:`B`,
    - electron distributions :math:`N(\gamma)`,
    - emissivities :math:`j_\nu`,
    - absorption coefficients :math:`\alpha_\nu`,
    - and transfer depths :math:`\ell_\mathrm{eff}`

    are interpreted as comoving-frame quantities unless otherwise stated.

    Relativistic Doppler boosting and cosmological redshift effects are applied
    only during transformation to observer-frame intensities and flux densities.

    The engine itself is geometry-agnostic. Higher-level SED classes are responsible
    for specifying emitting areas, angular structure, viewing geometry, and
    multi-zone integration strategies.

    References
    ----------
    .. footbibliography::
    """

    # ------------------------------------------ #
    # Initialization and Convenience Methods     #
    # ------------------------------------------ #
    def __init__(self):
        """Initialize the numerical synchrotron engine and declare internal caches for kernel interpolators."""
        # Declare the cache for the first kernel interpolator. This will be populated by load_first_kernel().
        self._log_x_first_kernel = None
        self._log_first_kernel = None
        self._log_dfirst_kernel_dlog_x = None
        self._interp_first_kernel = None
        self._interp_dfirst_kernel_dlog_x = None

        # Declare the cache for the pitch-angle averaged first kernel interpolator. This will be populated by
        # load_averaged_first_kernel().
        self._log_x_avg_first_kernel = None
        self._log_avg_first_kernel = None
        self._log_davg_first_kernel_dlog_x = None
        self._interp_avg_first_kernel = None
        self._interp_davg_first_kernel_dlog_x = None

    # --- Dunder Methods --- #
    def __repr__(self) -> str:
        """
        Return a concise developer-facing representation of the engine state.

        The representation summarizes which synchrotron kernel interpolators are
        currently loaded and therefore available for numerical evaluation.

        Returns
        -------
        str
            String representation of the engine.
        """
        return (
            f"{self.__class__.__name__}("
            f"first_kernel_loaded={self.is_first_kernel_loaded}, "
            f"avg_first_kernel_loaded={self.is_avg_first_kernel_loaded}"
            f")"
        )

    def __str__(self) -> str:
        """
        Return a human-readable summary of the numerical synchrotron engine.

        Returns
        -------
        str
            Human-readable engine summary.
        """
        kernels = []

        if self.is_first_kernel_loaded:
            kernels.append("first kernel")

        if self.is_avg_first_kernel_loaded:
            kernels.append("pitch-angle averaged kernel")

        if kernels:
            kernel_str = ", ".join(kernels)
        else:
            kernel_str = "no kernels loaded"

        return f"{self.__class__.__name__} with {kernel_str}"

    def __len__(self) -> int:
        """
        Return the number of loaded synchrotron kernel interpolators.

        Returns
        -------
        int
            Number of loaded kernel interpolation tables.
        """
        return int(self.is_first_kernel_loaded) + int(self.is_avg_first_kernel_loaded)

    def __enter__(self):
        """
        Enter the numerical synchrotron engine context.

        Returns
        -------
        ~trilobite.radiation.synchrotron.SEDs.numerical.core.NumericalSynchrotronEngine
            The engine itself.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the numerical synchrotron engine context and clear interpolator caches.

        Parameters
        ----------
        exc_type : type or None
            Type of the exception raised in the context, if any.
        exc_val : BaseException or None
            The exception raised in the context, if any.
        exc_tb : traceback or None
            Traceback of the exception raised in the context, if any.

        Returns
        -------
        bool
            Always ``False``, so exceptions raised in the context are not suppressed.
        """
        self.clear()
        return False

    def __copy__(self):
        """
        Return a shallow copy of the synchrotron engine.

        Returns
        -------
        ~trilobite.radiation.synchrotron.SEDs.numerical.core.NumericalSynchrotronEngine
            A shallow copy of the engine that shares the loaded kernel caches.
        """
        cls = self.__class__
        new = cls.__new__(cls)
        new.__dict__.update(self.__dict__)
        return new

    __deepcopy__ = __copy__

    # --- Properties --- #
    @property
    def is_first_kernel_loaded(self) -> bool:
        """bool: Whether the first synchrotron kernel interpolator is loaded."""
        return self._interp_first_kernel is not None

    @property
    def is_avg_first_kernel_loaded(self) -> bool:
        """bool: Whether the pitch-angle-averaged kernel interpolator is loaded."""
        return self._interp_avg_first_kernel is not None

    @property
    def log_first_kernel_grid(self) -> np.ndarray:
        """~numpy.ndarray: Natural-log x-grid for the first synchrotron kernel."""
        self.ensure_first_kernel_loaded()
        return self._log_x_first_kernel

    @property
    def first_kernel_grid(self) -> np.ndarray:
        """~numpy.ndarray: Linear x-grid for the first synchrotron kernel."""
        self.ensure_first_kernel_loaded()
        return np.exp(self._log_x_first_kernel)

    @property
    def log_first_kernel_values(self) -> np.ndarray:
        """~numpy.ndarray: Natural-log values of the first synchrotron kernel."""
        self.ensure_first_kernel_loaded()
        return self._log_first_kernel

    @property
    def first_kernel_values(self) -> np.ndarray:
        """~numpy.ndarray: Linear values of the first synchrotron kernel."""
        self.ensure_first_kernel_loaded()
        return np.exp(self._log_first_kernel)

    @property
    def log_first_kernel_derivative(self) -> np.ndarray:
        """~numpy.ndarray: Logarithmic derivative of the first synchrotron kernel."""
        self.ensure_first_kernel_loaded()
        return self._log_dfirst_kernel_dlog_x

    @property
    def first_kernel_interpolator(self) -> InterpolatedUnivariateSpline:
        """~scipy.interpolate.InterpolatedUnivariateSpline: Spline interpolator for the first synchrotron kernel."""
        self.ensure_first_kernel_loaded()
        return self._interp_first_kernel

    @property
    def first_kernel_derivative_interpolator(self) -> InterpolatedUnivariateSpline:
        """~scipy.interpolate.InterpolatedUnivariateSpline: Spline interpolator for the first kernel log-derivative."""
        self.ensure_first_kernel_loaded()
        return self._interp_dfirst_kernel_dlog_x

    @property
    def first_kernel_min_x(self) -> float:
        """float: Minimum x-value tabulated for the first synchrotron kernel."""
        self.ensure_first_kernel_loaded()
        return float(np.exp(self._log_x_first_kernel[0]))

    @property
    def first_kernel_max_x(self) -> float:
        """float: Maximum x-value tabulated for the first synchrotron kernel."""
        self.ensure_first_kernel_loaded()
        return float(np.exp(self._log_x_first_kernel[-1]))

    @property
    def first_kernel_size(self) -> int:
        """int: Number of tabulated points in the first synchrotron kernel."""
        self.ensure_first_kernel_loaded()
        return len(self._log_x_first_kernel)

    @property
    def log_avg_first_kernel_grid(self) -> np.ndarray:
        """~numpy.ndarray: Natural-log x-grid for the averaged synchrotron kernel."""
        self.ensure_avg_first_kernel_loaded()
        return self._log_x_avg_first_kernel

    @property
    def avg_first_kernel_grid(self) -> np.ndarray:
        """~numpy.ndarray: Linear x-grid for the averaged synchrotron kernel."""
        self.ensure_avg_first_kernel_loaded()
        return np.exp(self._log_x_avg_first_kernel)

    @property
    def log_avg_first_kernel_values(self) -> np.ndarray:
        """~numpy.ndarray: Natural-log values of the averaged synchrotron kernel."""
        self.ensure_avg_first_kernel_loaded()
        return self._log_avg_first_kernel

    @property
    def avg_first_kernel_values(self) -> np.ndarray:
        """~numpy.ndarray: Linear values of the averaged synchrotron kernel."""
        self.ensure_avg_first_kernel_loaded()
        return np.exp(self._log_avg_first_kernel)

    @property
    def log_avg_first_kernel_derivative(self) -> np.ndarray:
        """~numpy.ndarray: Logarithmic derivative of the averaged synchrotron kernel."""
        self.ensure_avg_first_kernel_loaded()
        return self._log_davg_first_kernel_dlog_x

    @property
    def avg_first_kernel_interpolator(self) -> InterpolatedUnivariateSpline:
        """~scipy.interpolate.InterpolatedUnivariateSpline: Spline interpolator for the averaged kernel."""
        self.ensure_avg_first_kernel_loaded()
        return self._interp_avg_first_kernel

    @property
    def avg_first_kernel_derivative_interpolator(self) -> InterpolatedUnivariateSpline:
        """~scipy.interpolate.InterpolatedUnivariateSpline: Spline for the averaged kernel log-derivative."""
        self.ensure_avg_first_kernel_loaded()
        return self._interp_davg_first_kernel_dlog_x

    @property
    def avg_first_kernel_min_x(self) -> float:
        """float: Minimum x-value tabulated for the averaged synchrotron kernel."""
        self.ensure_avg_first_kernel_loaded()
        return float(np.exp(self._log_x_avg_first_kernel[0]))

    @property
    def avg_first_kernel_max_x(self) -> float:
        """float: Maximum x-value tabulated for the averaged synchrotron kernel."""
        self.ensure_avg_first_kernel_loaded()
        return float(np.exp(self._log_x_avg_first_kernel[-1]))

    @property
    def avg_first_kernel_size(self) -> int:
        """int: Number of tabulated points in the averaged synchrotron kernel."""
        self.ensure_avg_first_kernel_loaded()
        return len(self._log_x_avg_first_kernel)

    # ------------------------------------------ #
    # Utility Functions                          #
    # ------------------------------------------ #
    def clear(self) -> None:
        """Clear all cached synchrotron kernel tables and interpolators."""
        # First kernel caches.
        self._log_x_first_kernel = None
        self._log_first_kernel = None
        self._log_dfirst_kernel_dlog_x = None
        self._interp_first_kernel = None
        self._interp_dfirst_kernel_dlog_x = None

        # Averaged first kernel caches.
        self._log_x_avg_first_kernel = None
        self._log_avg_first_kernel = None
        self._log_davg_first_kernel_dlog_x = None
        self._interp_avg_first_kernel = None
        self._interp_davg_first_kernel_dlog_x = None

    def load_first_kernel(
        self,
        x: Union[np.ndarray, None] = None,
        x_min: float = 1e-5,
        x_max: float = 1e2,
        num_points: int = 1000,
        spacing: str = "log",
        method: str = "exact",
        **kwargs,
    ) -> None:
        r"""
        Build and cache the interpolator for the first synchrotron kernel :math:`F(x)`.

        Calling this method a second time overwrites the existing grid.

        Parameters
        ----------
        x : array-like, optional
            Explicit :math:`x` grid. If provided, ``x_min``, ``x_max``, ``num_points``,
            and ``spacing`` are ignored.
        x_min : float, optional
            Lower bound of the :math:`x` grid. Default is ``1e-5``.
        x_max : float, optional
            Upper bound of the :math:`x` grid. Default is ``1e2``.
        num_points : int, optional
            Number of grid points. Default is ``1000``.
        spacing : {"log", "linear"}, optional
            Grid spacing when ``x`` is not provided. Default is ``"log"``.
        method : {"exact", "lu"}, optional
            Kernel evaluation method passed to :func:`_log_first_synchrotron_kernel`.
            Default is ``"exact"``.
        **kwargs : dict
            Additional keyword arguments passed to :class:`scipy.interpolate.InterpolatedUnivariateSpline`.
        """
        if x is not None:
            x_grid = np.asarray(x, dtype=float)
        elif spacing == "log":
            x_grid = np.geomspace(x_min, x_max, num_points)
        elif spacing == "linear":
            x_grid = np.linspace(x_min, x_max, num_points)
        else:
            raise ValueError(f"Invalid spacing '{spacing}'. Must be 'log' or 'linear'.")

        log_x = np.log(x_grid)
        log_F, dlogF_dlogx = _log_first_synchrotron_kernel(log_x, method=method, derivative=True)

        self._log_x_first_kernel = log_x
        self._log_first_kernel = log_F
        self._log_dfirst_kernel_dlog_x = dlogF_dlogx
        self._interp_first_kernel = InterpolatedUnivariateSpline(log_x, log_F, **kwargs)
        self._interp_dfirst_kernel_dlog_x = InterpolatedUnivariateSpline(log_x, dlogF_dlogx, **kwargs)

    def load_avg_first_kernel(
        self,
        x: Union[np.ndarray, None] = None,
        x_min: float = 1e-5,
        x_max: float = 1e2,
        num_points: int = 1000,
        spacing: str = "log",
        method: str = "exact",
        **kwargs,
    ) -> None:
        r"""
        Build and cache the interpolator for the pitch-angle-averaged synchrotron kernel :math:`\bar{F}(x)`.

        Calling this method a second time overwrites the existing grid.

        Parameters
        ----------
        x : array-like, optional
            Explicit :math:`x` grid. If provided, ``x_min``, ``x_max``, ``num_points``,
            and ``spacing`` are ignored.
        x_min : float, optional
            Lower bound of the :math:`x` grid. Default is ``1e-5``.
        x_max : float, optional
            Upper bound of the :math:`x` grid. Default is ``1e2``.
        num_points : int, optional
            Number of grid points. Default is ``1000``.
        spacing : {"log", "linear"}, optional
            Grid spacing when ``x`` is not provided. Default is ``"log"``.
        method : {"exact", "lu"}, optional
            Kernel evaluation method passed to
            :func:`_log_averaged_first_synchrotron_kernel`. Default is ``"exact"``.
        **kwargs : dict
            Additional keyword arguments passed to :class:`scipy.interpolate.InterpolatedUnivariateSpline`.
        """
        if x is not None:
            x_grid = np.asarray(x, dtype=float)
        elif spacing == "log":
            x_grid = np.geomspace(x_min, x_max, num_points)
        elif spacing == "linear":
            x_grid = np.linspace(x_min, x_max, num_points)
        else:
            raise ValueError(f"Invalid spacing '{spacing}'. Must be 'log' or 'linear'.")

        log_x = np.log(x_grid)
        log_F, dlogF_dlogx = _log_averaged_first_synchrotron_kernel(log_x, method=method, derivative=True)

        self._log_x_avg_first_kernel = log_x
        self._log_avg_first_kernel = log_F
        self._log_davg_first_kernel_dlog_x = dlogF_dlogx
        self._interp_avg_first_kernel = InterpolatedUnivariateSpline(log_x, log_F, **kwargs)
        self._interp_davg_first_kernel_dlog_x = InterpolatedUnivariateSpline(log_x, dlogF_dlogx, **kwargs)

    def _compute_first_kernel(self, log_x: Union[float, np.ndarray], **kwargs) -> Union[float, np.ndarray]:
        r"""
        Evaluate :math:`\ln F(x)` by spline interpolation.

        Parameters
        ----------
        log_x : float or array-like
            Natural log of the dimensionless frequency ratio :math:`x`.

        Returns
        -------
        log_F : float or ~numpy.ndarray
            :math:`\ln F(x)` at each input point.
        """
        self.ensure_first_kernel_loaded()
        return self._interp_first_kernel(log_x, **kwargs)

    def _compute_first_kernel_derivative(self, log_x, **kwargs):
        r"""
        Evaluate :math:`d\ln F/d\ln x` by spline interpolation.

        Parameters
        ----------
        log_x : float or array-like
            Natural log of the dimensionless frequency ratio :math:`x`.

        Returns
        -------
        dlogF_dlogx : float or ~numpy.ndarray
            Logarithmic derivative :math:`d\ln F/d\ln x` at each input point.
        """
        self.ensure_first_kernel_loaded()
        return self._interp_dfirst_kernel_dlog_x(log_x, **kwargs)

    def _compute_avg_first_kernel(self, log_x: Union[float, np.ndarray], **kwargs) -> Union[float, np.ndarray]:
        r"""
        Evaluate :math:`\ln \bar{F}(x)` by spline interpolation.

        Parameters
        ----------
        log_x : float or array-like
            Natural log of the dimensionless frequency ratio :math:`x`.

        Returns
        -------
        log_F_avg : float or ~numpy.ndarray
            :math:`\ln \bar{F}(x)` at each input point.
        """
        self.ensure_avg_first_kernel_loaded()
        return self._interp_avg_first_kernel(log_x, **kwargs)

    def _compute_avg_first_kernel_derivative(self, log_x, **kwargs):
        r"""
        Evaluate :math:`d\ln \bar{F}/d\ln x` by spline interpolation.

        Parameters
        ----------
        log_x : float or array-like
            Natural log of the dimensionless frequency ratio :math:`x`.

        Returns
        -------
        dlogFavg_dlogx : float or ~numpy.ndarray
            Logarithmic derivative :math:`d\ln \bar{F}/d\ln x` at each input point.
        """
        self.ensure_avg_first_kernel_loaded()
        return self._interp_davg_first_kernel_dlog_x(log_x, **kwargs)

    def ensure_first_kernel_loaded(self) -> None:
        r"""
        Ensure that the first kernel interpolator is loaded. Raises if not.

        Raises
        ------
        RuntimeError
            If the first kernel is not loaded.
        """
        if self._interp_first_kernel is None:
            raise RuntimeError("First kernel has not been loaded. Call load_first_kernel() first.")

    def ensure_avg_first_kernel_loaded(self) -> None:
        r"""
        Ensure that the pitch-angle averaged first kernel interpolator is loaded. Raises if not.

        Raises
        ------
        RuntimeError
            If the pitch-angle averaged first kernel is not loaded.
        """
        if self._interp_avg_first_kernel is None:
            raise RuntimeError(
                "Pitch-angle averaged first kernel has not been loaded. Call load_avg_first_kernel() first."
            )

    def compute_first_kernel(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""
        Evaluate the first synchrotron kernel :math:`F(x)`.

        Inside the loaded domain, evaluates by spline interpolation. Outside the domain,
        falls back to the analytic asymptotic expressions:

        - :math:`F(x) \approx C\,x^{1/3}` for :math:`x` below the grid
        - :math:`F(x) \approx \sqrt{\pi x/2}\,e^{-x}` for :math:`x` above the grid

        Parameters
        ----------
        x : float or ~numpy.ndarray
            Dimensionless frequency ratio :math:`x = \nu/\nu_c`, where
            :math:`\nu_c` is the synchrotron critical frequency.

        Returns
        -------
        float or ~numpy.ndarray
            :math:`F(x)` at each input point.

        Raises
        ------
        RuntimeError
            If :meth:`load_first_kernel` has not been called.
        """
        # Ensure the first kernel is loaded before attempting to evaluate.
        self.ensure_first_kernel_loaded()

        # Now cast down x to an array to ensure that we are evaluating efficiently.
        scalar = np.ndim(x) == 0
        x = np.atleast_1d(np.asarray(x, dtype=float))
        log_x = np.log(x)
        log_F = np.empty_like(log_x)

        # Set up the masks.
        lo, hi = self._log_x_first_kernel[0], self._log_x_first_kernel[-1]
        lmsk = log_x < lo
        rmsk = log_x > hi
        imsk = ~(lmsk | rmsk)

        if np.any(lmsk):
            log_F[lmsk] = np.log(2.1495282415344786) + (1.0 / 3.0) * log_x[lmsk]

        if np.any(rmsk):
            x_r = np.exp(log_x[rmsk])
            log_F[rmsk] = 0.5 * (np.log(np.pi) - np.log(2.0)) + 0.5 * log_x[rmsk] - x_r

        if np.any(imsk):
            log_F[imsk] = self._interp_first_kernel(log_x[imsk])

        # Return.
        F = np.exp(log_F)
        return F.item() if scalar else F

    def compute_first_kernel_derivative(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""
        Evaluate the logarithmic derivative of the first synchrotron kernel.

        Computes

        .. math::

            \frac{d\log F}{d\log x}

        for the first synchrotron kernel :math:`F(x)`.

        Inside the loaded interpolation domain, values are obtained from the cached
        spline interpolator. Outside the domain, analytic asymptotic limits are used:

        - :math:`d\log F/d\log x \to 1/3` for :math:`x \ll 1`
        - :math:`d\log F/d\log x \to 1/2 - x` for :math:`x \gg 1`

        Parameters
        ----------
        x : float or ~numpy.ndarray
            Dimensionless frequency ratio :math:`x = \nu/\nu_c`.

        Returns
        -------
        float or ~numpy.ndarray
            Logarithmic derivative :math:`d\log F/d\log x`.

        Raises
        ------
        RuntimeError
            If :meth:`load_first_kernel` has not been called.
        """
        self.ensure_first_kernel_loaded()

        scalar = np.ndim(x) == 0
        x = np.atleast_1d(np.asarray(x, dtype=float))
        log_x = np.log(x)

        dlogF_dlogx = np.empty_like(log_x)

        lo, hi = self._log_x_first_kernel[0], self._log_x_first_kernel[-1]

        lmsk = log_x < lo
        rmsk = log_x > hi
        imsk = ~(lmsk | rmsk)

        if np.any(lmsk):
            dlogF_dlogx[lmsk] = 1.0 / 3.0

        if np.any(rmsk):
            x_r = np.exp(log_x[rmsk])
            dlogF_dlogx[rmsk] = 0.5 - x_r

        if np.any(imsk):
            dlogF_dlogx[imsk] = self._interp_dfirst_kernel_dlog_x(log_x[imsk])

        return dlogF_dlogx.item() if scalar else dlogF_dlogx

    def compute_avg_first_kernel(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""
        Evaluate the pitch-angle-averaged synchrotron kernel :math:`\bar{F}(x)`.

        Inside the loaded domain, evaluates by spline interpolation. Outside the domain,
        falls back to analytic asymptotic expressions:

        - :math:`\bar{F}(x) \approx C\,x^{1/3}` for :math:`x \ll 1`
        - :math:`\bar{F}(x) \approx (\pi/2)e^{-x}` for :math:`x \gg 1`

        Parameters
        ----------
        x : float or ~numpy.ndarray
            Dimensionless frequency ratio :math:`x = \nu/\nu_c`.

        Returns
        -------
        float or ~numpy.ndarray
            Pitch-angle-averaged synchrotron kernel :math:`\bar{F}(x)`.

        Raises
        ------
        RuntimeError
            If :meth:`load_avg_first_kernel` has not been called.
        """
        self.ensure_avg_first_kernel_loaded()

        scalar = np.ndim(x) == 0
        x = np.atleast_1d(np.asarray(x, dtype=float))
        log_x = np.log(x)

        log_F = np.empty_like(log_x)

        lo, hi = self._log_x_avg_first_kernel[0], self._log_x_avg_first_kernel[-1]

        lmsk = log_x < lo
        rmsk = log_x > hi
        imsk = ~(lmsk | rmsk)

        if np.any(lmsk):
            log_F[lmsk] = 0.5924524416080822 + (1.0 / 3.0) * log_x[lmsk]

        if np.any(rmsk):
            x_r = np.exp(log_x[rmsk])
            eps = 99.0 / (162.0 + x_r)
            log_F[rmsk] = np.log(np.pi / 2.0) + np.log1p(-eps) - x_r

        if np.any(imsk):
            log_F[imsk] = self._interp_avg_first_kernel(log_x[imsk])

        F_avg = np.exp(log_F)

        return F_avg.item() if scalar else F_avg

    def compute_avg_first_kernel_derivative(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        r"""
        Evaluate the logarithmic derivative of the pitch-angle-averaged synchrotron kernel.

        Computes

        .. math::

            \frac{d\log \bar{F}}{d\log x}

        for the pitch-angle-averaged synchrotron kernel :math:`\bar{F}(x)`.

        Inside the loaded interpolation domain, values are obtained from the cached
        spline interpolator. Outside the domain, analytic asymptotic limits are used:

        - :math:`d\log \bar{F}/d\log x \to 1/3` for :math:`x \ll 1`
        - :math:`d\log \bar{F}/d\log x \to 1/2 - x` for :math:`x \gg 1`

        Parameters
        ----------
        x : float or ~numpy.ndarray
            Dimensionless frequency ratio :math:`x = \nu/\nu_c`.

        Returns
        -------
        float or ~numpy.ndarray
            Logarithmic derivative :math:`d\log \bar{F}/d\log x`.

        Raises
        ------
        RuntimeError
            If :meth:`load_avg_first_kernel` has not been called.
        """
        self.ensure_avg_first_kernel_loaded()

        scalar = np.ndim(x) == 0
        x = np.atleast_1d(np.asarray(x, dtype=float))
        log_x = np.log(x)

        dlogF_dlogx = np.empty_like(log_x)

        lo, hi = self._log_x_avg_first_kernel[0], self._log_x_avg_first_kernel[-1]

        lmsk = log_x < lo
        rmsk = log_x > hi
        imsk = ~(lmsk | rmsk)

        if np.any(lmsk):
            dlogF_dlogx[lmsk] = 1.0 / 3.0

        if np.any(rmsk):
            x_r = np.exp(log_x[rmsk])
            dlogF_dlogx[rmsk] = 0.5 - x_r

        if np.any(imsk):
            dlogF_dlogx[imsk] = self._interp_davg_first_kernel_dlog_x(log_x[imsk])

        return dlogF_dlogx.item() if scalar else dlogF_dlogx

    # ------------------------------------------ #
    # Radiative Quantities API (Private)         #
    # ------------------------------------------ #
    def _compute_log_emissivity(
        self,
        log_nu: Union[float, np.ndarray],
        log_B: Union[float, np.ndarray],
        log_N: Union[float, np.ndarray],
        log_gamma: Union[float, np.ndarray],
        log_weights: Union[float, np.ndarray],
        sin_alpha: Union[float, np.ndarray],
    ) -> np.ndarray:
        r"""
        Compute :math:`\log j_\nu`, the log of the synchrotron emissivity per unit volume.

        Evaluates the integral

        .. math::

            j_\nu = \chi\,B\sin\alpha
                    \int_0^\infty F(x)\,N(\gamma)\,d\gamma

        in log-space via :func:`~scipy.special.logsumexp`, where
        :math:`x = \nu / (c_{1\gamma} B \sin\alpha \gamma^2)` and :math:`\chi` is the
        CGS synchrotron emissivity constant.

        Parameters
        ----------
        log_nu : float or ~numpy.ndarray
            Natural log of the frequency grid in CGS (Hz). These are the
            rest frame frequencies for evaluation.
        log_B : float or ~numpy.ndarray or broadcastable to it
            Natural log of the (rest frame) magnetic field strength in CGS (G).
        log_N : float or ~numpy.ndarray
            Natural log of the (rest frame) electron number density evaluated on the Lorentz factor
            grid. The last axis is the gamma integration axis; leading axes define the zone batch.
        log_gamma : float or ~numpy.ndarray
            Natural log of the (rest frame) Lorentz factor grid.
        log_weights : float or ~numpy.ndarray
            Quadrature weights :math:`w_i = \gamma_i\,\Delta\!\log\gamma_i` in log-space,
            used to convert the sum over :math:`d\gamma` into a Riemann sum.
        sin_alpha : float or ~numpy.ndarray or broadcastable to it
            Sine of the (rest frame) pitch angle.

        Returns
        -------
        log_emissivity : ~numpy.ndarray
            :math:`\log j_\nu` in CGS (:math:`\mathrm{erg\,s^{-1}\,cm^{-3}\,Hz^{-1}\,sr^{-1}}`).
            This is the **rest-frame** emissivity; transformation to the observer
            frame is the responsibility of the caller.

        Notes
        -----
        ``log_nu`` defines the spectral axis ``(*nu_shape,)``; ``log_B``, ``sin_alpha``,
        and ``log_N`` share a common zone axis ``(*zone_shape,)``. Output shape is
        ``(*nu_shape, *zone_shape)``. Scalar zone quantities correspond to
        ``zone_shape = ()`` and collapse the output to ``(*nu_shape,)``.
        """
        # Cast everything to an array to ensure that broadcasting
        # behaves properly.
        log_nu = np.asarray(log_nu, dtype="f8")
        log_B = np.asarray(log_B, dtype="f8")
        log_gamma = np.asarray(log_gamma, dtype="f8")
        log_N = np.asarray(log_N, dtype="f8")
        log_weights = np.asarray(log_weights, dtype="f8")
        log_sin_alpha = np.log(np.asarray(sin_alpha, dtype="f8"))

        # Extract the shapes and set up the broadcasting. N sets the zone shape,
        # so we extract the elements up to the last to represent the different
        # batches.
        nu_shape = log_nu.shape
        zone_shape = log_N.shape[:-1]
        n_gamma = log_gamma.size
        n_nu_dims = len(nu_shape)
        n_zone_dims = len(zone_shape)

        # Ensure that B and sin_alpha are all self-consistent with the
        # zone shape.
        log_B = np.broadcast_to(log_B, zone_shape)
        log_sin_alpha = np.broadcast_to(log_sin_alpha, zone_shape)

        # log_x: (*nu_shape, *zone_shape, n_gamma)
        log_nu_v = log_nu.reshape(nu_shape + (1,) * n_zone_dims + (1,))
        log_B_v = log_B.reshape((1,) * n_nu_dims + zone_shape + (1,))
        log_sa_v = log_sin_alpha.reshape((1,) * n_nu_dims + zone_shape + (1,))
        log_g_v = log_gamma.reshape((1,) * (n_nu_dims + n_zone_dims) + (n_gamma,))

        log_x = np.ascontiguousarray(log_nu_v - 2.0 * log_g_v - log_B_v - _log_c_1_gamma_cgs - log_sa_v)

        # Perform the interpolation step.
        log_F = np.interp(log_x.ravel(), self._log_x_first_kernel, self._log_first_kernel)
        log_F = log_F.reshape(nu_shape + zone_shape + (n_gamma,))

        # Now perform the integration.
        log_N_v = log_N.reshape((1,) * n_nu_dims + zone_shape + (n_gamma,))
        log_w_v = log_weights.reshape((1,) * (n_nu_dims + n_zone_dims) + (n_gamma,))
        log_B_bc = log_B.reshape((1,) * n_nu_dims + zone_shape)
        log_sa_bc = log_sin_alpha.reshape((1,) * n_nu_dims + zone_shape)

        return logsumexp(log_F + log_N_v + log_w_v, axis=-1) + _log_chi_cgs + log_B_bc + log_sa_bc

    def _compute_log_absorption_coefficient(
        self,
        log_nu: Union[float, np.ndarray],
        log_B: Union[float, np.ndarray],
        log_N: Union[float, np.ndarray],
        log_gamma: Union[float, np.ndarray],
        log_weights: Union[float, np.ndarray],
        sin_alpha: Union[float, np.ndarray],
    ) -> np.ndarray:
        r"""
        Compute the log of the synchrotron self-absorption coefficient :math:`|\alpha_\nu|`.

        Uses integration by parts (IBP) to avoid requiring the derivative of the electron
        distribution as an input. Starting from

        .. math::

            \alpha_\nu = \frac{\sqrt{3}\,e^3 B\sin\alpha}{8\pi m_e^2 c^2 \nu^2}
                         \int_0^\infty F(x)\,\gamma^2
                         \frac{d}{d\gamma}\!\left(\frac{N(\gamma)}{\gamma^2}\right) d\gamma,

        IBP converts the integral to

        .. math::

            -2 \int_0^\infty \bigl(F(x) - x F'(x)\bigr)\, N(\gamma)\, d\!\log\gamma,

        where :math:`F - xF' = F\,(1 - d\!\log F/d\!\log x) > 0` everywhere, so the
        integrand has a guaranteed sign and can be evaluated entirely in log-space via
        :func:`~scipy.special.logsumexp`. The IBP factor of 2 cancels the 8π → 4π
        prefactor change, giving

        .. math::

            |\alpha_\nu| = \frac{\chi}{m_e}\,\frac{B\sin\alpha}{\nu^2}
                           \int_0^\infty (F - xF')\,N\,d\!\log\gamma.

        The returned value is :math:`\log|\alpha_\nu|`. The sign is always negative for
        physical distributions (:math:`\alpha_\nu < 0` as written in the source formula;
        the caller is responsible for applying the correct sign convention).

        Parameters
        ----------
        log_nu : float or ~numpy.ndarray
            Natural log of the frequency grid in CGS (Hz).
        log_B : float or ~numpy.ndarray or broadcastable to it
            Natural log of the magnetic field strength in CGS (G).
        log_N : float or ~numpy.ndarray
            Natural log of the electron number density on the Lorentz factor grid.
            The last axis is the gamma integration axis; leading axes define the zone batch.
        log_gamma : float or ~numpy.ndarray
            Natural log of the Lorentz factor grid.
        log_weights : float or ~numpy.ndarray
            Quadrature weights :math:`w_i = \gamma_i\,\Delta\!\log\gamma_i` in log-space.
        sin_alpha : float or ~numpy.ndarray or broadcastable to it
            Sine of the pitch angle.

        Returns
        -------
        log_alpha : ~numpy.ndarray
            :math:`\log|\alpha_\nu|` in CGS (:math:`\mathrm{cm^{-1}}`).

        Notes
        -----
        ``log_nu`` defines the spectral axis ``(*nu_shape,)``; ``log_B``, ``sin_alpha``,
        and ``log_N`` share a common zone axis ``(*zone_shape,)``. Output shape is
        ``(*nu_shape, *zone_shape)``. Scalar zone quantities correspond to
        ``zone_shape = ()`` and collapse the output to ``(*nu_shape,)``.
        """
        # Cast everything to an array to ensure that broadcasting
        # behaves properly.
        log_nu = np.asarray(log_nu, dtype="f8")
        log_B = np.asarray(log_B, dtype="f8")
        log_gamma = np.asarray(log_gamma, dtype="f8")
        log_N = np.asarray(log_N, dtype="f8")
        log_weights = np.asarray(log_weights, dtype="f8")
        log_sin_alpha = np.log(np.asarray(sin_alpha, dtype="f8"))

        # Extract the shapes and set up the broadcasting. N sets the zone shape,
        # so we extract the elements up to the last to represent the different
        # batches.
        nu_shape = log_nu.shape
        zone_shape = log_N.shape[:-1]
        n_gamma = log_gamma.size
        n_nu_dims = len(nu_shape)
        n_zone_dims = len(zone_shape)

        # Ensure that B and sin_alpha are all self-consistent with the
        # zone shape.
        log_B = np.broadcast_to(log_B, zone_shape)
        log_sin_alpha = np.broadcast_to(log_sin_alpha, zone_shape)

        # log_x: (*nu_shape, *zone_shape, n_gamma)
        log_nu_v = log_nu.reshape(nu_shape + (1,) * n_zone_dims + (1,))
        log_B_v = log_B.reshape((1,) * n_nu_dims + zone_shape + (1,))
        log_sa_v = log_sin_alpha.reshape((1,) * n_nu_dims + zone_shape + (1,))
        log_g_v = log_gamma.reshape((1,) * (n_nu_dims + n_zone_dims) + (n_gamma,))

        log_x = np.ascontiguousarray(log_nu_v - 2.0 * log_g_v - log_B_v - _log_c_1_gamma_cgs - log_sa_v)

        # Perform the interpolation step.
        log_F = np.interp(log_x.ravel(), self._log_x_first_kernel, self._log_first_kernel)
        log_F = log_F.reshape(nu_shape + zone_shape + (n_gamma,))
        d_log_F_d_log_x = np.interp(log_x.ravel(), self._log_x_first_kernel, self._log_dfirst_kernel_dlog_x)
        d_log_F_d_log_x = d_log_F_d_log_x.reshape(nu_shape + zone_shape + (n_gamma,))

        log_kernel = log_F + np.log(np.maximum(1.0 - d_log_F_d_log_x, _FLOAT64_TINY))

        # log_weights - log_gamma = log(d_log_gamma); integrand uses d_log_gamma measure.
        log_dloggamma = (log_weights - log_gamma).reshape((1,) * (n_nu_dims + n_zone_dims) + (n_gamma,))
        log_N_v = log_N.reshape((1,) * n_nu_dims + zone_shape + (n_gamma,))
        log_integrand = log_kernel + log_N_v + log_dloggamma

        log_nu_bc = log_nu.reshape(nu_shape + (1,) * n_zone_dims)
        log_B_bc = log_B.reshape((1,) * n_nu_dims + zone_shape)
        log_sa_bc = log_sin_alpha.reshape((1,) * n_nu_dims + zone_shape)

        return _log_chi_abs_cgs + log_B_bc + log_sa_bc - 2.0 * log_nu_bc + logsumexp(log_integrand, axis=-1)

    def _compute_log_emissivity_and_absorption(
        self,
        log_nu: Union[float, np.ndarray],
        log_B: Union[float, np.ndarray],
        log_N: Union[float, np.ndarray],
        log_gamma: Union[float, np.ndarray],
        log_weights: Union[float, np.ndarray],
        sin_alpha: Union[float, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Compute :math:`\log j_\nu` and :math:`\log|\alpha_\nu|` in a single pass.

        Fused version of :meth:`_compute_log_emissivity` and
        :meth:`_compute_log_absorption_coefficient` that builds ``log_x`` and
        evaluates both kernel interpolations once, eliminating the redundant array
        construction relative to calling the two methods separately. Parameters and
        return semantics match those of the constituent methods.

        Returns
        -------
        log_emissivity : ~numpy.ndarray
            :math:`\log j_\nu` in CGS.
        log_absorption : ~numpy.ndarray
            :math:`\log|\alpha_\nu|` in CGS.
        """
        log_nu = np.asarray(log_nu, dtype="f8")
        log_B = np.asarray(log_B, dtype="f8")
        log_gamma = np.asarray(log_gamma, dtype="f8")
        log_N = np.asarray(log_N, dtype="f8")
        log_weights = np.asarray(log_weights, dtype="f8")
        log_sin_alpha = np.log(np.asarray(sin_alpha, dtype="f8"))

        nu_shape = log_nu.shape
        zone_shape = log_N.shape[:-1]
        n_gamma = log_gamma.size
        n_nu_dims = len(nu_shape)
        n_zone_dims = len(zone_shape)

        log_B = np.broadcast_to(log_B, zone_shape)
        log_sin_alpha = np.broadcast_to(log_sin_alpha, zone_shape)

        log_nu_v = log_nu.reshape(nu_shape + (1,) * n_zone_dims + (1,))
        log_B_v = log_B.reshape((1,) * n_nu_dims + zone_shape + (1,))
        log_sa_v = log_sin_alpha.reshape((1,) * n_nu_dims + zone_shape + (1,))
        log_g_v = log_gamma.reshape((1,) * (n_nu_dims + n_zone_dims) + (n_gamma,))

        log_x = np.ascontiguousarray(log_nu_v - 2.0 * log_g_v - log_B_v - _log_c_1_gamma_cgs - log_sa_v)
        log_x_flat = log_x.ravel()

        log_F = np.interp(log_x_flat, self._log_x_first_kernel, self._log_first_kernel).reshape(
            nu_shape + zone_shape + (n_gamma,)
        )
        d_log_F_d_log_x = np.interp(log_x_flat, self._log_x_first_kernel, self._log_dfirst_kernel_dlog_x).reshape(
            nu_shape + zone_shape + (n_gamma,)
        )

        log_N_v = log_N.reshape((1,) * n_nu_dims + zone_shape + (n_gamma,))
        log_w_v = log_weights.reshape((1,) * (n_nu_dims + n_zone_dims) + (n_gamma,))
        log_B_bc = log_B.reshape((1,) * n_nu_dims + zone_shape)
        log_sa_bc = log_sin_alpha.reshape((1,) * n_nu_dims + zone_shape)

        log_emissivity = logsumexp(log_F + log_N_v + log_w_v, axis=-1) + _log_chi_cgs + log_B_bc + log_sa_bc

        log_kernel = log_F + np.log(np.maximum(1.0 - d_log_F_d_log_x, _FLOAT64_TINY))
        log_dloggamma = (log_weights - log_gamma).reshape((1,) * (n_nu_dims + n_zone_dims) + (n_gamma,))
        log_nu_bc = log_nu.reshape(nu_shape + (1,) * n_zone_dims)
        log_absorption = (
            _log_chi_abs_cgs
            + log_B_bc
            + log_sa_bc
            - 2.0 * log_nu_bc
            + logsumexp(log_kernel + log_N_v + log_dloggamma, axis=-1)
        )

        return log_emissivity, log_absorption

    def _compute_log_rf_specific_intensity(
        self,
        log_nu: Union[float, np.ndarray],
        log_slab_depth: Union[float, np.ndarray],
        log_B: Union[float, np.ndarray],
        log_N: Union[float, np.ndarray],
        log_gamma: Union[float, np.ndarray],
        log_weights: Union[float, np.ndarray],
        sin_alpha: Union[float, np.ndarray],
    ) -> np.ndarray:
        r"""
        Compute the comoving-frame synchrotron specific intensity :math:`\log I'_{\nu'}`.

        This method solves the homogeneous radiative transfer equation in the
        **rest frame of the emitting plasma** (equivalently the comoving frame)
        using

        .. math::

            I'_{\nu'} = S'_{\nu'}\left(1 - e^{-\tau'_{\nu'}}\right),

        where

        .. math::

            S'_{\nu'} = \frac{j'_{\nu'}}{\alpha'_{\nu'}}

        is the synchrotron source function and

        .. math::

            \tau'_{\nu'} = \alpha'_{\nu'} \ell'

        is the optical depth through the emitting slab.

        All quantities supplied to this function are interpreted as **comoving-frame**
        quantities. No relativistic Doppler boosting, aberration, Lorentz contraction,
        or cosmological redshift corrections are applied here. Observer-frame
        transformations are handled separately by
        :meth:`_compute_log_specific_intensity` and
        :meth:`_compute_log_flux_density`.

        Parameters
        ----------
        log_nu : float or ~numpy.ndarray
            Natural log of the **comoving-frame frequency**
            :math:`\nu'` in CGS (Hz).

        log_slab_depth : float or ~numpy.ndarray or broadcastable to it
            Natural log of the **comoving-frame line-of-sight path length**
            :math:`\ell'` through the emitting region in CGS (cm).

            This quantity should represent the physical transfer depth in the plasma
            rest frame. Lorentz contraction effects must therefore already be accounted
            for by the caller if the emitting geometry is defined in another frame.

        log_B : float or ~numpy.ndarray or broadcastable to it
            Natural log of the **comoving-frame magnetic field strength**
            :math:`B'` in CGS (G).

            The synchrotron kernel and emissivity formalism assume that the magnetic
            field is evaluated in the plasma rest frame.

        log_N : float or ~numpy.ndarray
            Natural log of the **comoving-frame electron number density distribution**
            :math:`N'(\gamma)` evaluated on the Lorentz-factor grid.

            The final axis corresponds to the electron Lorentz-factor integration axis;
            leading axes define independent emitting zones or batches.

        log_gamma : float or ~numpy.ndarray
            Natural log of the electron Lorentz-factor grid.

            Electron Lorentz factors are always defined relative to the plasma
            rest frame.

        log_weights : float or ~numpy.ndarray
            Quadrature weights used for numerical integration over the Lorentz-factor
            distribution.

        sin_alpha : float or ~numpy.ndarray or broadcastable to it
            Sine of the **comoving-frame pitch angle**
            :math:`\alpha'` between the electron velocity and magnetic field.

        Returns
        -------
        log_intensity : ~numpy.ndarray
            Natural log of the **comoving-frame specific intensity**
            :math:`I'_{\nu'}` in CGS units

            .. math::

                \mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}\,sr^{-1}}.

        Notes
        -----
        The returned intensity satisfies the Lorentz-invariant relation

        .. math::

            \frac{I_\nu}{\nu^3}
            =
            \frac{I'_{\nu'}}{\nu'^3},

        which is used by the observer-frame transformation routines.

        ``log_nu`` defines the spectral axis ``(*nu_shape,)`` while
        ``log_B``, ``log_slab_depth``, ``sin_alpha``, and ``log_N`` define the
        emitting-zone axes ``(*zone_shape,)``. The returned array therefore has
        shape

        ``(*nu_shape, *zone_shape)``.

        Scalar zone quantities correspond to ``zone_shape = ()`` and collapse
        the output to ``(*nu_shape,)``.
        """
        log_nu = np.asarray(log_nu, dtype="f8")
        log_slab_depth = np.asarray(log_slab_depth, dtype="f8")

        log_emissivity, log_absorption = self._compute_log_emissivity_and_absorption(
            log_nu, log_B, log_N, log_gamma, log_weights, sin_alpha
        )

        # log_slab_depth is zone-shaped; insert leading nu axes to broadcast against (*nu_shape, *zone_shape).
        log_sd_v = log_slab_depth[(np.newaxis,) * log_nu.ndim]

        tau = np.exp(np.clip(log_absorption + log_sd_v, -np.inf, 500))
        return log_emissivity - log_absorption + np.log(-np.expm1(-tau))

    def _compute_log_specific_intensity(
        self,
        log_nu: Union[float, np.ndarray],
        log_slab_depth: Union[float, np.ndarray],
        log_B: Union[float, np.ndarray],
        log_N: Union[float, np.ndarray],
        log_gamma: Union[float, np.ndarray],
        log_weights: Union[float, np.ndarray],
        z: float,
        beta: float,
        cos_theta: float,
        sin_alpha: Union[float, np.ndarray],
    ) -> np.ndarray:
        r"""
        Compute the observed synchrotron specific intensity :math:`\log I_\nu`.

        This method transforms observer-frame frequencies into the comoving frame
        of the emitting plasma, evaluates the synchrotron radiative transfer solution
        there, and then Lorentz-transforms the resulting intensity back into the
        observer frame.

        The transformation is based on the Lorentz invariance of

        .. math::

            \frac{I_\nu}{\nu^3},

        together with cosmological redshift. Frequencies are related by

        .. math::

            \nu'
            =
            \frac{(1+z)\,\nu}{\mathcal{D}},

        where

        .. math::

            \mathcal{D}
            =
            \frac{1}{\Gamma(1 - \beta\cos\theta)}

        is the Doppler factor and

        .. math::

            \Gamma = \frac{1}{\sqrt{1-\beta^2}}

        is the bulk Lorentz factor.

        The observer-frame intensity is then

        .. math::

            I_\nu
            =
            \left(\frac{\mathcal{D}}{1+z}\right)^3
            I'_{\nu'}.

        Parameters
        ----------
        log_nu : float or ~numpy.ndarray
            Natural log of the **observer-frame frequency**
            :math:`\nu` in CGS (Hz).

        log_slab_depth : float or ~numpy.ndarray or broadcastable to it
            Natural log of the **comoving-frame transfer depth**
            :math:`\ell'` through the emitting region in CGS (cm).

            This quantity is assumed to already include any Lorentz contraction
            appropriate to the emitting geometry.

        log_B : float or ~numpy.ndarray or broadcastable to it
            Natural log of the **comoving-frame magnetic field strength**
            :math:`B'` in CGS (G).

        log_N : float or ~numpy.ndarray
            Natural log of the **comoving-frame electron number density distribution**
            :math:`N'(\gamma)` evaluated on the Lorentz-factor grid.

            The final axis corresponds to the Lorentz-factor integration axis;
            leading axes define independent emitting zones or batches.

        log_gamma : float or ~numpy.ndarray
            Natural log of the electron Lorentz-factor grid.

            Electron Lorentz factors are defined in the plasma rest frame.

        log_weights : float or ~numpy.ndarray
            Quadrature weights used for numerical integration over the
            electron distribution.

        z : float
            Cosmological redshift of the source.

        beta : float
            Bulk plasma velocity in units of :math:`c`.

        cos_theta : float
            Cosine of the angle between the bulk velocity vector and the
            observer line of sight.

        sin_alpha : float or ~numpy.ndarray or broadcastable to it
            Sine of the **comoving-frame pitch angle**
            :math:`\alpha'`.

        Returns
        -------
        log_intensity : ~numpy.ndarray
            Natural log of the observer-frame specific intensity
            :math:`I_\nu` in CGS units

            .. math::

                \mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}\,sr^{-1}}.

        Notes
        -----
        This method assumes that all local plasma quantities are already expressed
        in the comoving frame. Only the frequency and intensity transformations
        associated with relativistic bulk motion and cosmological redshift are
        applied here.

        ``log_nu`` defines the spectral axis ``(*nu_shape,)`` while
        ``log_B``, ``log_slab_depth``, ``sin_alpha``, and ``log_N`` define the
        emitting-zone axes ``(*zone_shape,)``. The returned array therefore has
        shape

        ``(*nu_shape, *zone_shape)``.

        Scalar zone quantities correspond to ``zone_shape = ()`` and collapse
        the output to ``(*nu_shape,)``.
        """
        log_gamma_bulk = np.log(1.0 / np.sqrt(1 - beta**2))
        log_Doppler = -log_gamma_bulk - np.log1p(-beta * cos_theta)
        log_correction_factor = log_Doppler - np.log1p(z)

        log_nu_rf = log_nu - log_correction_factor

        log_rf_intensity = self._compute_log_rf_specific_intensity(
            log_nu_rf, log_slab_depth, log_B, log_N, log_gamma, log_weights, sin_alpha
        )

        return log_rf_intensity + 3 * log_correction_factor

    def _compute_log_flux_density(
        self,
        log_nu: Union[float, np.ndarray],
        log_slab_depth: Union[float, np.ndarray],
        log_B: Union[float, np.ndarray],
        log_N: Union[float, np.ndarray],
        log_gamma: Union[float, np.ndarray],
        log_weights: Union[float, np.ndarray],
        log_A: Union[float, np.ndarray],
        log_D_A: Union[float, np.ndarray],
        z: float = 0,
        beta: float = 0,
        cos_theta: float = 1.0,
        sin_alpha: Union[float, np.ndarray] = 1.0,
    ) -> np.ndarray:
        r"""
        Compute the observed synchrotron spectral flux density :math:`\log F_\nu`.

        This method evaluates the observed spectral flux density from a homogeneous
        synchrotron-emitting region by:

        1. transforming observer-frame frequencies into the plasma comoving frame,
        2. solving the synchrotron radiative transfer equation in the comoving frame,
        3. applying relativistic Doppler boosting and cosmological redshift corrections,
        4. and integrating the resulting specific intensity over the apparent emitting area.

        The observed flux density is computed from

        .. math::

            F_\nu
            =
            \frac{A_\mathrm{eff}}{D_A^2}
            I_\nu,

        where

        .. math::

            I_\nu
            =
            \left(\frac{\mathcal D}{1+z}\right)^3
            I'_{\nu'}

        is the observer-frame specific intensity obtained from the Lorentz invariance of

        .. math::

            \frac{I_\nu}{\nu^3},

        and

        .. math::

            \nu'
            =
            \frac{(1+z)\nu}{\mathcal D}.

        The relativistic Doppler factor is

        .. math::

            \mathcal D
            =
            \frac{1}{\Gamma(1-\beta\cos\theta)},

        with

        .. math::

            \Gamma
            =
            \frac{1}{\sqrt{1-\beta^2}}.

        Parameters
        ----------
        log_nu : float or ~numpy.ndarray
            Natural log of the **observer-frame frequency**
            :math:`\nu` in CGS (Hz).

        log_slab_depth : float or ~numpy.ndarray or broadcastable to it
            Natural log of the **comoving-frame transfer depth**
            :math:`\ell'` through the emitting region in CGS (cm).

            This quantity is interpreted as the effective line-of-sight radiative
            transfer depth in the plasma rest frame and should therefore already
            include any Lorentz-contraction corrections appropriate to the emitting
            geometry.

        log_B : float or ~numpy.ndarray or broadcastable to it
            Natural log of the **comoving-frame magnetic field strength**
            :math:`B'` in CGS (G).

        log_N : float or ~numpy.ndarray
            Natural log of the **comoving-frame electron number density distribution**
            :math:`N'(\gamma)` evaluated on the Lorentz-factor grid.

            The final axis corresponds to the Lorentz-factor integration axis while
            any leading axes define independent emitting zones or batches.

        log_gamma : float or ~numpy.ndarray
            Natural log of the electron Lorentz-factor grid.

            Electron Lorentz factors are always defined in the plasma rest frame.

        log_weights : float or ~numpy.ndarray
            Quadrature weights used for numerical integration over the electron
            distribution.

        log_A : float or ~numpy.ndarray or broadcastable to it
            Natural log of the projected effective emitting area
            :math:`A_\mathrm{eff}` in CGS (:math:`\mathrm{cm^2}`).

            This quantity is interpreted in the observer frame and represents the
            apparent projected area contributing to the observed emission.

        log_D_A : float or ~numpy.ndarray
            Natural log of the angular-diameter distance :math:`D_A`
            in CGS (cm).

        z : float, optional
            Cosmological redshift of the source. Default is ``0``.

        beta : float, optional
            Bulk plasma velocity in units of :math:`c`. Default is ``0``.

        cos_theta : float, optional
            Cosine of the angle between the bulk velocity vector and the observer
            line of sight. Default is ``1``.

        sin_alpha : float or ~numpy.ndarray or broadcastable to it, optional
            Sine of the **comoving-frame pitch angle**
            :math:`\alpha'`. Default is ``1``.

        Returns
        -------
        log_flux : ~numpy.ndarray
            Natural log of the observer-frame spectral flux density
            :math:`F_\nu` in CGS units

            .. math::

                \mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}}.

        Notes
        -----
        All local plasma quantities are assumed to be specified in the comoving
        frame of the emitting material. In particular:

        - magnetic field strengths :math:`B`,
        - electron distributions :math:`N(\gamma)`,
        - transfer depths :math:`\ell`,
        - emissivities :math:`j_\nu`,
        - and absorption coefficients :math:`\alpha_\nu`

        are interpreted as comoving-frame quantities.

        Only the frequency and intensity transformations associated with relativistic
        bulk motion and cosmological redshift are applied here.

        ``log_nu`` defines the spectral axis ``(*nu_shape,)`` while
        ``log_A``, ``log_B``, ``log_slab_depth``, ``sin_alpha``, and ``log_N`` define
        the emitting-zone axes ``(*zone_shape,)``. The returned array therefore has
        shape

        ``(*nu_shape, *zone_shape)``.

        Scalar zone quantities correspond to ``zone_shape = ()`` and collapse the
        output to ``(*nu_shape,)``.
        """
        log_nu = np.asarray(log_nu, dtype="f8")
        log_A = np.asarray(log_A, dtype="f8")
        log_D_A = np.asarray(log_D_A, dtype="f8")

        log_gamma_bulk = np.log(1.0 / np.sqrt(1 - beta**2))
        log_Doppler = -log_gamma_bulk - np.log1p(-beta * cos_theta)
        log_correction_factor = log_Doppler - np.log1p(z)

        log_nu_rf = log_nu - log_correction_factor

        log_specific_intensity = (
            self._compute_log_rf_specific_intensity(
                log_nu_rf, log_slab_depth, log_B, log_N, log_gamma, log_weights, sin_alpha
            )
            + 3 * log_correction_factor
        )

        # log_A_eff is zone-shaped; insert leading nu axes to broadcast against (*nu_shape, *zone_shape).
        log_A_eff_v = log_A[(np.newaxis,) * log_nu.ndim]

        return log_specific_intensity + log_A_eff_v - 2 * log_D_A

    def _compute_log_brightness_temperature(
        self,
        log_nu: Union[float, np.ndarray],
        log_slab_depth: Union[float, np.ndarray],
        log_B: Union[float, np.ndarray],
        log_N: Union[float, np.ndarray],
        log_gamma: Union[float, np.ndarray],
        log_weights: Union[float, np.ndarray],
        z: float = 0,
        beta: float = 0,
        cos_theta: float = 1.0,
        sin_alpha: Union[float, np.ndarray] = 1.0,
    ) -> np.ndarray:
        r"""
        Compute the observer-frame brightness temperature :math:`\log T_B`.

        The brightness temperature is defined through the Rayleigh--Jeans relation

        .. math::

            T_B
            =
            \frac{c^2 I_\nu}{2 k_B \nu^2},

        where :math:`I_\nu` is the observer-frame specific intensity.

        Parameters
        ----------
        log_nu : float or ~numpy.ndarray
            Natural log of the observer-frame frequency :math:`\nu`
            in CGS (Hz).

        log_slab_depth : float or ~numpy.ndarray
            Natural log of the comoving-frame transfer depth.

        log_B : float or ~numpy.ndarray
            Natural log of the comoving-frame magnetic field strength.

        log_N : float or ~numpy.ndarray
            Natural log of the comoving-frame electron distribution.

        log_gamma : float or ~numpy.ndarray
            Natural log of the electron Lorentz-factor grid.

        log_weights : float or ~numpy.ndarray
            Quadrature weights for gamma integration.

        z : float, optional
            Cosmological redshift.

        beta : float, optional
            Bulk plasma velocity in units of :math:`c`.

        cos_theta : float, optional
            Cosine of the observer angle relative to the bulk velocity.

        sin_alpha : float or ~numpy.ndarray, optional
            Sine of the comoving-frame pitch angle.

        Returns
        -------
        log_T_B : ~numpy.ndarray
            Natural log of the observer-frame brightness temperature
            in Kelvin.
        """
        log_nu = np.asarray(log_nu, dtype="f8")

        log_intensity = self._compute_log_specific_intensity(
            log_nu,
            log_slab_depth,
            log_B,
            log_N,
            log_gamma,
            log_weights,
            z,
            beta,
            cos_theta,
            sin_alpha,
        )

        return 2.0 * _log_c_cgs + log_intensity - _LOG2 - _log_k_B_cgs - 2.0 * log_nu

    def _compute_log_rf_brightness_temperature(
        self,
        log_nu: Union[float, np.ndarray],
        log_slab_depth: Union[float, np.ndarray],
        log_B: Union[float, np.ndarray],
        log_N: Union[float, np.ndarray],
        log_gamma: Union[float, np.ndarray],
        log_weights: Union[float, np.ndarray],
        sin_alpha: Union[float, np.ndarray] = 1.0,
    ) -> np.ndarray:
        r"""
        Compute the comoving-frame brightness temperature :math:`\log T'_B`.

        The brightness temperature is defined through the Rayleigh--Jeans relation

        .. math::

            T'_B
            =
            \frac{c^2 I'_{\nu'}}{2 k_B \nu'^2},

        where :math:`I'_{\nu'}` is the comoving-frame synchrotron specific intensity.

        All quantities supplied to this method are interpreted in the
        comoving frame of the emitting plasma. No relativistic Doppler
        boosting or cosmological redshift corrections are applied here.

        Parameters
        ----------
        log_nu : float or ~numpy.ndarray
            Natural log of the comoving-frame frequency
            :math:`\nu'` in CGS (Hz).

        log_slab_depth : float or ~numpy.ndarray or broadcastable to it
            Natural log of the comoving-frame transfer depth
            :math:`\ell'` through the emitting region in CGS (cm).

        log_B : float or ~numpy.ndarray or broadcastable to it
            Natural log of the comoving-frame magnetic field strength
            :math:`B'` in CGS (G).

        log_N : float or ~numpy.ndarray
            Natural log of the comoving-frame electron number density
            distribution :math:`N'(\gamma)`.

        log_gamma : float or ~numpy.ndarray
            Natural log of the electron Lorentz-factor grid.

        log_weights : float or ~numpy.ndarray
            Quadrature weights for gamma integration.

        sin_alpha : float or ~numpy.ndarray or broadcastable to it, optional
            Sine of the comoving-frame pitch angle
            :math:`\alpha'`. Default is ``1``.

        Returns
        -------
        log_T_B : ~numpy.ndarray
            Natural log of the comoving-frame brightness temperature
            in Kelvin.

        Notes
        -----
        The brightness temperature satisfies the Lorentz transformation

        .. math::

            \frac{T_B}{\nu}
            =
            \frac{T'_B}{\nu'},

        implying

        .. math::

            T_B
            =
            \frac{\mathcal D}{1+z} T'_B.

        Observer-frame transformations are therefore handled separately by
        :meth:`_compute_log_brightness_temperature`.
        """
        log_nu = np.asarray(log_nu, dtype="f8")

        log_rf_intensity = self._compute_log_rf_specific_intensity(
            log_nu,
            log_slab_depth,
            log_B,
            log_N,
            log_gamma,
            log_weights,
            sin_alpha,
        )

        return 2.0 * _log_c_cgs + log_rf_intensity - _LOG2 - _log_k_B_cgs - 2.0 * log_nu

    def _compute_log_pa_emissivity(
        self,
        log_nu: Union[float, np.ndarray],
        log_B: Union[float, np.ndarray],
        log_N: Union[float, np.ndarray],
        log_gamma: Union[float, np.ndarray],
        log_weights: Union[float, np.ndarray],
    ) -> np.ndarray:
        r"""
        Compute :math:`\log j_\nu`, the log of the pitch-angle-averaged synchrotron emissivity.

        Evaluates the integral

        .. math::

            j_\nu = \chi\,B
                    \int_0^\infty \bar{F}(x)\,N(\gamma)\,d\gamma

        in log-space via :func:`~scipy.special.logsumexp`, where
        :math:`x = \nu / (c_{1\gamma} B \gamma^2)`, :math:`\bar{F}(x)` is the
        pitch-angle-averaged synchrotron kernel, and :math:`\chi` is the CGS synchrotron
        emissivity constant. This formulation is appropriate for isotropic pitch-angle
        distributions, where :math:`\bar{F}` absorbs the average over :math:`\sin\alpha`.

        Parameters
        ----------
        log_nu : float or ~numpy.ndarray
            Natural log of the **comoving-frame frequency** :math:`\nu'` in CGS (Hz).
        log_B : float or ~numpy.ndarray or broadcastable to it
            Natural log of the **comoving-frame magnetic field strength** :math:`B'`
            in CGS (G).
        log_N : float or ~numpy.ndarray
            Natural log of the **comoving-frame electron number density distribution**
            :math:`N'(\gamma)` evaluated on the Lorentz-factor grid. The last axis is
            the gamma integration axis; leading axes define the zone batch.
        log_gamma : float or ~numpy.ndarray
            Natural log of the electron Lorentz-factor grid.
        log_weights : float or ~numpy.ndarray
            Quadrature weights :math:`w_i = \gamma_i\,\Delta\!\log\gamma_i` in log-space,
            used to convert the sum over :math:`d\gamma` into a Riemann sum.

        Returns
        -------
        log_emissivity : ~numpy.ndarray
            :math:`\log j_\nu` in CGS (:math:`\mathrm{erg\,s^{-1}\,cm^{-3}\,Hz^{-1}\,sr^{-1}}`).
            This is the **comoving-frame** emissivity; transformation to the observer
            frame is the responsibility of the caller.

        Notes
        -----
        ``log_nu`` defines the spectral axis ``(*nu_shape,)``; ``log_B`` and ``log_N``
        share a common zone axis ``(*zone_shape,)``. Output shape is
        ``(*nu_shape, *zone_shape)``. Scalar zone quantities correspond to
        ``zone_shape = ()`` and collapse the output to ``(*nu_shape,)``.
        """
        log_nu = np.asarray(log_nu, dtype="f8")
        log_B = np.asarray(log_B, dtype="f8")
        log_gamma = np.asarray(log_gamma, dtype="f8")
        log_N = np.asarray(log_N, dtype="f8")
        log_weights = np.asarray(log_weights, dtype="f8")

        nu_shape = log_nu.shape
        zone_shape = log_N.shape[:-1]
        n_gamma = log_gamma.size
        n_nu_dims = len(nu_shape)
        n_zone_dims = len(zone_shape)

        log_B = np.broadcast_to(log_B, zone_shape)

        # log_x: (*nu_shape, *zone_shape, n_gamma)
        log_nu_v = log_nu.reshape(nu_shape + (1,) * n_zone_dims + (1,))
        log_B_v = log_B.reshape((1,) * n_nu_dims + zone_shape + (1,))
        log_g_v = log_gamma.reshape((1,) * (n_nu_dims + n_zone_dims) + (n_gamma,))

        log_x = np.ascontiguousarray(log_nu_v - 2.0 * log_g_v - log_B_v - _log_c_1_gamma_cgs)

        log_F = np.interp(log_x.ravel(), self._log_x_avg_first_kernel, self._log_avg_first_kernel)
        log_F = log_F.reshape(nu_shape + zone_shape + (n_gamma,))

        log_N_v = log_N.reshape((1,) * n_nu_dims + zone_shape + (n_gamma,))
        log_w_v = log_weights.reshape((1,) * (n_nu_dims + n_zone_dims) + (n_gamma,))

        log_B_bc = log_B.reshape((1,) * n_nu_dims + zone_shape)

        return logsumexp(log_F + log_N_v + log_w_v, axis=-1) + _log_chi_cgs + log_B_bc

    def _compute_log_pa_absorption_coefficient(
        self,
        log_nu: Union[float, np.ndarray],
        log_B: Union[float, np.ndarray],
        log_N: Union[float, np.ndarray],
        log_gamma: Union[float, np.ndarray],
        log_weights: Union[float, np.ndarray],
    ) -> np.ndarray:
        r"""
        Compute :math:`\log|\alpha_\nu|`, the log of the pitch-angle-averaged synchrotron self-absorption coefficient.

        Uses integration by parts (IBP) to avoid requiring the derivative of the electron
        distribution as an input. Starting from the pitch-angle-averaged form

        .. math::

            \alpha_\nu
            =
            \frac{\sqrt{3}\,e^3 B}{8\pi m_e^2 c^2 \nu^2}
            \int_0^\infty \bar{F}(x)\,\gamma^2
            \frac{d}{d\gamma}\!\left(\frac{N(\gamma)}{\gamma^2}\right) d\gamma,

        IBP converts the integral to

        .. math::

            -2 \int_0^\infty \bigl(\bar{F}(x) - x\bar{F}'(x)\bigr)\,N(\gamma)\,d\!\log\gamma,

        where :math:`\bar{F} - x\bar{F}' = \bar{F}(1 - d\!\log\bar{F}/d\!\log x) > 0`
        everywhere, so the integrand has a guaranteed sign and can be evaluated entirely
        in log-space via :func:`~scipy.special.logsumexp`. The IBP factor of 2 cancels
        the :math:`8\pi \to 4\pi` prefactor change, giving

        .. math::

            |\alpha_\nu|
            =
            \chi_\mathrm{abs}\,\frac{B}{\nu^2}
            \int_0^\infty \bigl(\bar{F} - x\bar{F}'\bigr)\,N\,d\!\log\gamma.

        The returned value is :math:`\log|\alpha_\nu|`. The sign is always negative for
        physical distributions; the caller is responsible for applying the correct sign
        convention.

        Parameters
        ----------
        log_nu : float or ~numpy.ndarray
            Natural log of the **comoving-frame frequency** :math:`\nu'` in CGS (Hz).
        log_B : float or ~numpy.ndarray or broadcastable to it
            Natural log of the **comoving-frame magnetic field strength** :math:`B'`
            in CGS (G).
        log_N : float or ~numpy.ndarray
            Natural log of the **comoving-frame electron number density distribution**
            :math:`N'(\gamma)` evaluated on the Lorentz-factor grid. The last axis is
            the gamma integration axis; leading axes define the zone batch.
        log_gamma : float or ~numpy.ndarray
            Natural log of the electron Lorentz-factor grid.
        log_weights : float or ~numpy.ndarray
            Quadrature weights :math:`w_i = \gamma_i\,\Delta\!\log\gamma_i` in log-space.

        Returns
        -------
        log_alpha : ~numpy.ndarray
            :math:`\log|\alpha_\nu|` in CGS (:math:`\mathrm{cm^{-1}}`).

        Notes
        -----
        ``log_nu`` defines the spectral axis ``(*nu_shape,)``; ``log_B`` and ``log_N``
        share a common zone axis ``(*zone_shape,)``. Output shape is
        ``(*nu_shape, *zone_shape)``. Scalar zone quantities correspond to
        ``zone_shape = ()`` and collapse the output to ``(*nu_shape,)``.
        """
        log_nu = np.asarray(log_nu, dtype="f8")
        log_B = np.asarray(log_B, dtype="f8")
        log_gamma = np.asarray(log_gamma, dtype="f8")
        log_N = np.asarray(log_N, dtype="f8")
        log_weights = np.asarray(log_weights, dtype="f8")

        nu_shape = log_nu.shape
        zone_shape = log_N.shape[:-1]
        n_gamma = log_gamma.size
        n_nu_dims = len(nu_shape)
        n_zone_dims = len(zone_shape)

        log_B = np.broadcast_to(log_B, zone_shape)

        # log_x: (*nu_shape, *zone_shape, n_gamma)
        log_nu_v = log_nu.reshape(nu_shape + (1,) * n_zone_dims + (1,))
        log_B_v = log_B.reshape((1,) * n_nu_dims + zone_shape + (1,))
        log_g_v = log_gamma.reshape((1,) * (n_nu_dims + n_zone_dims) + (n_gamma,))

        log_x = np.ascontiguousarray(log_nu_v - 2.0 * log_g_v - log_B_v - _log_c_1_gamma_cgs)

        log_F = np.interp(log_x.ravel(), self._log_x_avg_first_kernel, self._log_avg_first_kernel)
        log_F = log_F.reshape(nu_shape + zone_shape + (n_gamma,))
        d_log_F_d_log_x = np.interp(log_x.ravel(), self._log_x_avg_first_kernel, self._log_davg_first_kernel_dlog_x)
        d_log_F_d_log_x = d_log_F_d_log_x.reshape(nu_shape + zone_shape + (n_gamma,))

        log_kernel = log_F + np.log(np.maximum(1.0 - d_log_F_d_log_x, _FLOAT64_TINY))

        log_dloggamma = (log_weights - log_gamma).reshape((1,) * (n_nu_dims + n_zone_dims) + (n_gamma,))
        log_N_v = log_N.reshape((1,) * n_nu_dims + zone_shape + (n_gamma,))
        log_integrand = log_kernel + log_N_v + log_dloggamma

        log_nu_bc = log_nu.reshape(nu_shape + (1,) * n_zone_dims)
        log_B_bc = log_B.reshape((1,) * n_nu_dims + zone_shape)

        return _log_chi_abs_cgs + log_B_bc - 2.0 * log_nu_bc + logsumexp(log_integrand, axis=-1)

    def _compute_log_pa_emissivity_and_absorption(
        self,
        log_nu: Union[float, np.ndarray],
        log_B: Union[float, np.ndarray],
        log_N: Union[float, np.ndarray],
        log_gamma: Union[float, np.ndarray],
        log_weights: Union[float, np.ndarray],
        paired: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Compute pitch-angle-averaged :math:`\log j_\nu` and :math:`\log|\alpha_\nu|` in a single pass.

        Fused version of :meth:`_compute_log_pa_emissivity` and
        :meth:`_compute_log_pa_absorption_coefficient` that builds ``log_x`` and
        evaluates both averaged-kernel interpolations once, eliminating the redundant
        array construction relative to calling the two methods separately. Parameters
        and return semantics match those of the constituent methods.

        Parameters
        ----------
        paired : bool, optional
            If ``False`` (default), every frequency is evaluated in every zone and the output has shape
            ``(*nu_shape, *zone_shape)``. If ``True``, ``log_nu`` is broadcast against the zone axes and each
            zone is evaluated only at its own frequency, giving output shape ``zone_shape``. Pairing avoids
            computing the unused cross terms when every zone has a different frequency.

        Returns
        -------
        log_emissivity : ~numpy.ndarray
            :math:`\log j_\nu` in CGS.
        log_absorption : ~numpy.ndarray
            :math:`\log|\alpha_\nu|` in CGS.
        """
        log_nu = np.asarray(log_nu, dtype="f8")
        log_B = np.asarray(log_B, dtype="f8")
        log_gamma = np.asarray(log_gamma, dtype="f8")
        log_N = np.asarray(log_N, dtype="f8")
        log_weights = np.asarray(log_weights, dtype="f8")

        zone_shape = log_N.shape[:-1]
        n_zone_dims = len(zone_shape)
        n_gamma = log_gamma.size

        # In paired mode the frequency is a per-zone quantity rather than its own axis.
        if paired:
            log_nu = np.broadcast_to(log_nu, zone_shape)
            nu_shape = ()
        else:
            nu_shape = log_nu.shape
        n_nu_dims = len(nu_shape)

        log_B = np.broadcast_to(log_B, zone_shape)

        if paired:
            log_nu_v = log_nu[..., np.newaxis]
        else:
            log_nu_v = log_nu.reshape(nu_shape + (1,) * n_zone_dims + (1,))
        log_B_v = log_B.reshape((1,) * n_nu_dims + zone_shape + (1,))
        log_g_v = log_gamma.reshape((1,) * (n_nu_dims + n_zone_dims) + (n_gamma,))

        log_x = np.ascontiguousarray(log_nu_v - 2.0 * log_g_v - log_B_v - _log_c_1_gamma_cgs)
        log_x_flat = log_x.ravel()

        log_F = np.interp(log_x_flat, self._log_x_avg_first_kernel, self._log_avg_first_kernel).reshape(
            nu_shape + zone_shape + (n_gamma,)
        )
        d_log_F_d_log_x = np.interp(
            log_x_flat, self._log_x_avg_first_kernel, self._log_davg_first_kernel_dlog_x
        ).reshape(nu_shape + zone_shape + (n_gamma,))

        log_N_v = log_N.reshape((1,) * n_nu_dims + zone_shape + (n_gamma,))
        log_w_v = log_weights.reshape((1,) * (n_nu_dims + n_zone_dims) + (n_gamma,))
        log_B_bc = log_B.reshape((1,) * n_nu_dims + zone_shape)

        log_emissivity = logsumexp(log_F + log_N_v + log_w_v, axis=-1) + _log_chi_cgs + log_B_bc

        log_kernel = log_F + np.log(np.maximum(1.0 - d_log_F_d_log_x, _FLOAT64_TINY))
        log_dloggamma = (log_weights - log_gamma).reshape((1,) * (n_nu_dims + n_zone_dims) + (n_gamma,))
        log_nu_bc = log_nu if paired else log_nu.reshape(nu_shape + (1,) * n_zone_dims)
        log_absorption = (
            _log_chi_abs_cgs + log_B_bc - 2.0 * log_nu_bc + logsumexp(log_kernel + log_N_v + log_dloggamma, axis=-1)
        )

        return log_emissivity, log_absorption

    def _compute_log_pa_rf_specific_intensity(
        self,
        log_nu: Union[float, np.ndarray],
        log_slab_depth: Union[float, np.ndarray],
        log_B: Union[float, np.ndarray],
        log_N: Union[float, np.ndarray],
        log_gamma: Union[float, np.ndarray],
        log_weights: Union[float, np.ndarray],
        paired: bool = False,
    ) -> np.ndarray:
        r"""
        Compute the comoving-frame pitch-angle-averaged synchrotron specific intensity :math:`\log I'_{\nu'}`.

        Solves the homogeneous radiative transfer equation in the **rest frame of the
        emitting plasma** using

        .. math::

            I'_{\nu'} = S'_{\nu'}\left(1 - e^{-\tau'_{\nu'}}\right),

        where

        .. math::

            S'_{\nu'} = \frac{j'_{\nu'}}{\alpha'_{\nu'}}

        is the synchrotron source function and

        .. math::

            \tau'_{\nu'} = \alpha'_{\nu'} \ell'

        is the optical depth through the emitting slab. The emissivity and absorption
        coefficient are evaluated using the pitch-angle-averaged kernel :math:`\bar{F}(x)`,
        appropriate for an isotropic distribution of electron pitch angles.

        All quantities are interpreted as **comoving-frame** quantities. No relativistic
        Doppler boosting, aberration, Lorentz contraction, or cosmological redshift
        corrections are applied here; those are handled by
        :meth:`_compute_log_pa_specific_intensity` and :meth:`_compute_log_pa_flux_density`.

        Parameters
        ----------
        log_nu : float or ~numpy.ndarray
            Natural log of the **comoving-frame frequency** :math:`\nu'` in CGS (Hz).
        log_slab_depth : float or ~numpy.ndarray or broadcastable to it
            Natural log of the **comoving-frame line-of-sight path length** :math:`\ell'`
            through the emitting region in CGS (cm). Lorentz contraction effects must be
            accounted for by the caller if the geometry is defined in another frame.
        log_B : float or ~numpy.ndarray or broadcastable to it
            Natural log of the **comoving-frame magnetic field strength** :math:`B'`
            in CGS (G).
        log_N : float or ~numpy.ndarray
            Natural log of the **comoving-frame electron number density distribution**
            :math:`N'(\gamma)` evaluated on the Lorentz-factor grid. The last axis is
            the gamma integration axis; leading axes define the zone batch.
        log_gamma : float or ~numpy.ndarray
            Natural log of the electron Lorentz-factor grid, defined in the plasma rest frame.
        log_weights : float or ~numpy.ndarray
            Quadrature weights used for numerical integration over the Lorentz-factor
            distribution.

        Returns
        -------
        log_intensity : ~numpy.ndarray
            Natural log of the **comoving-frame specific intensity** :math:`I'_{\nu'}`
            in CGS units

            .. math::

                \mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}\,sr^{-1}}.

        Notes
        -----
        The returned intensity satisfies the Lorentz-invariant relation
        :math:`I_\nu/\nu^3 = I'_{\nu'}/\nu'^3`, which is used by the observer-frame
        transformation routines.

        ``log_nu`` defines the spectral axis ``(*nu_shape,)`` while ``log_B``,
        ``log_slab_depth``, and ``log_N`` define the emitting-zone axes ``(*zone_shape,)``.
        Output shape is ``(*nu_shape, *zone_shape)``. Scalar zone quantities correspond to
        ``zone_shape = ()`` and collapse the output to ``(*nu_shape,)``.
        """
        log_nu = np.asarray(log_nu, dtype="f8")
        log_slab_depth = np.asarray(log_slab_depth, dtype="f8")

        log_emissivity, log_absorption = self._compute_log_pa_emissivity_and_absorption(
            log_nu, log_B, log_N, log_gamma, log_weights, paired=paired
        )

        log_sd_v = log_slab_depth if paired else log_slab_depth[(np.newaxis,) * log_nu.ndim]

        tau = np.exp(np.clip(log_absorption + log_sd_v, -np.inf, 500))
        return log_emissivity - log_absorption + np.log(-np.expm1(-tau))

    def _compute_log_pa_specific_intensity(
        self,
        log_nu: Union[float, np.ndarray],
        log_slab_depth: Union[float, np.ndarray],
        log_B: Union[float, np.ndarray],
        log_N: Union[float, np.ndarray],
        log_gamma: Union[float, np.ndarray],
        log_weights: Union[float, np.ndarray],
        z: float,
        beta: float,
        cos_theta: float,
    ) -> np.ndarray:
        r"""
        Compute the observed pitch-angle-averaged synchrotron specific intensity :math:`\log I_\nu`.

        Transforms observer-frame frequencies into the comoving frame, evaluates the
        radiative transfer solution there using the pitch-angle-averaged kernel
        :math:`\bar{F}`, and Lorentz-transforms the resulting intensity back into the
        observer frame via the Lorentz invariance of :math:`I_\nu/\nu^3`.

        Frequencies are related by

        .. math::

            \nu'
            =
            \frac{(1+z)\,\nu}{\mathcal{D}},
            \qquad
            \mathcal{D}
            =
            \frac{1}{\Gamma(1 - \beta\cos\theta)},
            \qquad
            \Gamma = \frac{1}{\sqrt{1-\beta^2}},

        and the observer-frame intensity is

        .. math::

            I_\nu
            =
            \left(\frac{\mathcal{D}}{1+z}\right)^3
            I'_{\nu'}.

        Parameters
        ----------
        log_nu : float or ~numpy.ndarray
            Natural log of the **observer-frame frequency** :math:`\nu` in CGS (Hz).
        log_slab_depth : float or ~numpy.ndarray or broadcastable to it
            Natural log of the **comoving-frame transfer depth** :math:`\ell'` through the
            emitting region in CGS (cm). Lorentz contraction effects must be accounted for
            by the caller if the emitting geometry is defined in another frame.
        log_B : float or ~numpy.ndarray or broadcastable to it
            Natural log of the **comoving-frame magnetic field strength** :math:`B'`
            in CGS (G).
        log_N : float or ~numpy.ndarray
            Natural log of the **comoving-frame electron number density distribution**
            :math:`N'(\gamma)` evaluated on the Lorentz-factor grid. The last axis is
            the gamma integration axis; leading axes define the zone batch.
        log_gamma : float or ~numpy.ndarray
            Natural log of the electron Lorentz-factor grid, defined in the plasma rest frame.
        log_weights : float or ~numpy.ndarray
            Quadrature weights used for numerical integration over the electron distribution.
        z : float
            Cosmological redshift of the source.
        beta : float
            Bulk plasma velocity in units of :math:`c`.
        cos_theta : float
            Cosine of the angle between the bulk velocity vector and the observer line of sight.

        Returns
        -------
        log_intensity : ~numpy.ndarray
            Natural log of the observer-frame specific intensity :math:`I_\nu` in CGS units

            .. math::

                \mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}\,sr^{-1}}.

        Notes
        -----
        All local plasma quantities are assumed to be expressed in the comoving frame.
        Only the frequency and intensity transformations associated with relativistic
        bulk motion and cosmological redshift are applied here.

        ``log_nu`` defines the spectral axis ``(*nu_shape,)`` while ``log_B``,
        ``log_slab_depth``, and ``log_N`` define the emitting-zone axes ``(*zone_shape,)``.
        Output shape is ``(*nu_shape, *zone_shape)``. Scalar zone quantities correspond to
        ``zone_shape = ()`` and collapse the output to ``(*nu_shape,)``.
        """
        log_gamma_bulk = np.log(1.0 / np.sqrt(1 - beta**2))
        log_Doppler = -log_gamma_bulk - np.log1p(-beta * cos_theta)
        log_correction_factor = log_Doppler - np.log1p(z)

        log_nu_rf = log_nu - log_correction_factor

        log_rf_intensity = self._compute_log_pa_rf_specific_intensity(
            log_nu_rf, log_slab_depth, log_B, log_N, log_gamma, log_weights
        )

        return log_rf_intensity + 3 * log_correction_factor

    def _compute_log_pa_flux_density(
        self,
        log_nu: Union[float, np.ndarray],
        log_slab_depth: Union[float, np.ndarray],
        log_B: Union[float, np.ndarray],
        log_N: Union[float, np.ndarray],
        log_gamma: Union[float, np.ndarray],
        log_weights: Union[float, np.ndarray],
        log_A_eff: Union[float, np.ndarray],
        log_D_A: Union[float, np.ndarray],
        z: float = 0,
        beta: float = 0,
        cos_theta: float = 1.0,
        paired: bool = False,
    ) -> np.ndarray:
        r"""
        Compute the observed pitch-angle-averaged synchrotron spectral flux density :math:`\log F_\nu`.

        Evaluates the observed spectral flux density from a homogeneous synchrotron-emitting
        region using the pitch-angle-averaged kernel :math:`\bar{F}`, by transforming
        observer-frame frequencies into the comoving frame, solving the radiative transfer
        equation there, applying relativistic and cosmological corrections, and integrating
        over the apparent emitting area:

        .. math::

            F_\nu
            =
            \frac{A_\mathrm{eff}}{D_A^2}\,I_\nu,
            \qquad
            I_\nu
            =
            \left(\frac{\mathcal{D}}{1+z}\right)^3 I'_{\nu'},
            \qquad
            \nu' = \frac{(1+z)\,\nu}{\mathcal{D}},

        where :math:`\mathcal{D} = [\Gamma(1-\beta\cos\theta)]^{-1}` is the Doppler factor.

        Parameters
        ----------
        log_nu : float or ~numpy.ndarray
            Natural log of the **observer-frame frequency** :math:`\nu` in CGS (Hz).
        log_slab_depth : float or ~numpy.ndarray or broadcastable to it
            Natural log of the **comoving-frame transfer depth** :math:`\ell'` through the
            emitting region in CGS (cm). Lorentz contraction effects must be accounted for
            by the caller if the emitting geometry is defined in another frame.
        log_B : float or ~numpy.ndarray or broadcastable to it
            Natural log of the **comoving-frame magnetic field strength** :math:`B'`
            in CGS (G).
        log_N : float or ~numpy.ndarray
            Natural log of the **comoving-frame electron number density distribution**
            :math:`N'(\gamma)` evaluated on the Lorentz-factor grid. The last axis is
            the gamma integration axis; leading axes define the zone batch.
        log_gamma : float or ~numpy.ndarray
            Natural log of the electron Lorentz-factor grid, defined in the plasma rest frame.
        log_weights : float or ~numpy.ndarray
            Quadrature weights used for numerical integration over the electron distribution.
        log_A_eff : float or ~numpy.ndarray or broadcastable to it
            Natural log of the projected effective emitting area :math:`A_\mathrm{eff}`
            in CGS (:math:`\mathrm{cm^2}`). Interpreted as an observer-frame apparent area.
        log_D_A : float
            Natural log of the angular-diameter distance :math:`D_A` in CGS (cm).
        z : float, optional
            Cosmological redshift of the source. Default is ``0``.
        beta : float, optional
            Bulk plasma velocity in units of :math:`c`. Default is ``0``.
        cos_theta : float, optional
            Cosine of the angle between the bulk velocity vector and the observer
            line of sight. Default is ``1``.
        paired : bool, optional
            If ``True``, ``log_nu`` is paired with the zone axes: each zone is evaluated only at its own
            frequency and the output has shape ``zone_shape`` instead of ``(*nu_shape, *zone_shape)``. This
            avoids the unused cross terms when every zone has a different frequency. Default ``False``.

        Returns
        -------
        log_flux : ~numpy.ndarray
            Natural log of the observer-frame spectral flux density :math:`F_\nu`
            in CGS units

            .. math::

                \mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}}.

        Notes
        -----
        All local plasma quantities are assumed to be specified in the comoving frame.
        Only the frequency and intensity transformations associated with relativistic
        bulk motion and cosmological redshift are applied here.

        ``log_nu`` defines the spectral axis ``(*nu_shape,)`` while ``log_A_eff``,
        ``log_B``, ``log_slab_depth``, and ``log_N`` define the emitting-zone axes
        ``(*zone_shape,)``. Output shape is ``(*nu_shape, *zone_shape)``. Scalar zone
        quantities correspond to ``zone_shape = ()`` and collapse the output to
        ``(*nu_shape,)``.
        """
        log_nu = np.asarray(log_nu, dtype="f8")
        log_A_eff = np.asarray(log_A_eff, dtype="f8")
        log_D_A = np.asarray(log_D_A, dtype="f8")

        log_gamma_bulk = np.log(1.0 / np.sqrt(1 - beta**2))
        log_Doppler = -log_gamma_bulk - np.log1p(-beta * cos_theta)
        log_correction_factor = log_Doppler - np.log1p(z)

        log_nu_rf = log_nu - log_correction_factor

        log_specific_intensity = (
            self._compute_log_pa_rf_specific_intensity(
                log_nu_rf, log_slab_depth, log_B, log_N, log_gamma, log_weights, paired=paired
            )
            + 3 * log_correction_factor
        )

        log_A_eff_v = log_A_eff if paired else log_A_eff[(np.newaxis,) * log_nu.ndim]

        return log_specific_intensity + log_A_eff_v - 2 * log_D_A

    def _compute_log_pa_rf_brightness_temperature(
        self,
        log_nu: Union[float, np.ndarray],
        log_slab_depth: Union[float, np.ndarray],
        log_B: Union[float, np.ndarray],
        log_N: Union[float, np.ndarray],
        log_gamma: Union[float, np.ndarray],
        log_weights: Union[float, np.ndarray],
    ) -> np.ndarray:
        r"""
        Compute the comoving-frame pitch-angle-averaged brightness temperature.

        The brightness temperature is defined through the Rayleigh--Jeans relation

        .. math::

            T'_B =
            \frac{c^2 I'_{\nu'}}{2 k_B \nu'^2},

        where :math:`I'_{\nu'}` is the comoving-frame pitch-angle-averaged
        synchrotron specific intensity.

        Parameters
        ----------
        log_nu : float or ~numpy.ndarray
            Natural log of the comoving-frame frequency :math:`\nu'` in CGS (Hz).
        log_slab_depth : float or ~numpy.ndarray
            Natural log of the comoving-frame transfer depth in CGS (cm).
        log_B : float or ~numpy.ndarray
            Natural log of the comoving-frame magnetic field strength in CGS (G).
        log_N : float or ~numpy.ndarray
            Natural log of the comoving-frame electron distribution.
        log_gamma : float or ~numpy.ndarray
            Natural log of the electron Lorentz-factor grid.
        log_weights : float or ~numpy.ndarray
            Quadrature weights for gamma integration.

        Returns
        -------
        log_T_B : ~numpy.ndarray
            Natural log of the comoving-frame brightness temperature in Kelvin.
        """
        log_nu = np.asarray(log_nu, dtype="f8")

        log_rf_intensity = self._compute_log_pa_rf_specific_intensity(
            log_nu,
            log_slab_depth,
            log_B,
            log_N,
            log_gamma,
            log_weights,
        )

        return 2.0 * _log_c_cgs + log_rf_intensity - _LOG2 - _log_k_B_cgs - 2.0 * log_nu

    def _compute_log_pa_brightness_temperature(
        self,
        log_nu: Union[float, np.ndarray],
        log_slab_depth: Union[float, np.ndarray],
        log_B: Union[float, np.ndarray],
        log_N: Union[float, np.ndarray],
        log_gamma: Union[float, np.ndarray],
        log_weights: Union[float, np.ndarray],
        z: float = 0,
        beta: float = 0,
        cos_theta: float = 1.0,
    ) -> np.ndarray:
        r"""
        Compute the observer-frame pitch-angle-averaged brightness temperature.

        The brightness temperature is defined by

        .. math::

            T_B =
            \frac{c^2 I_\nu}{2 k_B \nu^2},

        where :math:`I_\nu` is the observed pitch-angle-averaged synchrotron
        specific intensity.

        Parameters
        ----------
        log_nu : float or ~numpy.ndarray
            Natural log of the observer-frame frequency :math:`\nu` in CGS (Hz).
        log_slab_depth : float or ~numpy.ndarray
            Natural log of the comoving-frame transfer depth in CGS (cm).
        log_B : float or ~numpy.ndarray
            Natural log of the comoving-frame magnetic field strength in CGS (G).
        log_N : float or ~numpy.ndarray
            Natural log of the comoving-frame electron distribution.
        log_gamma : float or ~numpy.ndarray
            Natural log of the electron Lorentz-factor grid.
        log_weights : float or ~numpy.ndarray
            Quadrature weights for gamma integration.
        z : float, optional
            Cosmological redshift. Default is ``0``.
        beta : float, optional
            Bulk plasma velocity in units of :math:`c`. Default is ``0``.
        cos_theta : float, optional
            Cosine of the angle between the bulk velocity and line of sight.
            Default is ``1``.

        Returns
        -------
        log_T_B : ~numpy.ndarray
            Natural log of the observer-frame brightness temperature in Kelvin.
        """
        log_nu = np.asarray(log_nu, dtype="f8")

        log_intensity = self._compute_log_pa_specific_intensity(
            log_nu,
            log_slab_depth,
            log_B,
            log_N,
            log_gamma,
            log_weights,
            z,
            beta,
            cos_theta,
        )

        return 2.0 * _log_c_cgs + log_intensity - _LOG2 - _log_k_B_cgs - 2.0 * log_nu

    # ------------------------------------------ #
    # Grid and Distribution Helpers
    # ------------------------------------------ #
    def _build_gamma_grid(
        self,
        gamma: Union[np.ndarray, None],
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Build a Lorentz-factor quadrature grid.

        When ``gamma`` is ``None``, constructs a log-uniform grid of ``n_gamma``
        points spanning ``[gamma_min, gamma_max]``. When ``gamma`` is provided,
        uses it directly and computes per-point spacings via central differences.

        Parameters
        ----------
        gamma : ~numpy.ndarray or None
            Explicit Lorentz factor grid. If ``None``, the grid is built from
            ``gamma_min``, ``gamma_max``, and ``n_gamma``.
        gamma_min : float, optional
            Lower bound of the grid. Ignored when ``gamma`` is provided. Default ``1.0``.
        gamma_max : float, optional
            Upper bound of the grid. Ignored when ``gamma`` is provided. Default ``1e8``.
        n_gamma : int, optional
            Number of grid points. Ignored when ``gamma`` is provided. Default ``200``.

        Returns
        -------
        log_gamma : ~numpy.ndarray
            Natural log of the Lorentz factor grid.
        log_weights : ~numpy.ndarray
            Log of the quadrature weights :math:`w_i = \gamma_i\,\Delta\!\log\gamma_i`.
        """
        if gamma is not None:
            log_gamma = np.log(np.asarray(gamma, dtype="f8"))
            log_weights = log_gamma + np.log(np.gradient(log_gamma))
        else:
            log_gamma = np.linspace(np.log(gamma_min), np.log(gamma_max), n_gamma)
            log_weights = log_gamma + np.log(log_gamma[1] - log_gamma[0])
        return log_gamma, log_weights

    def _resolve_log_N(
        self,
        N: Union[np.ndarray, Callable],
        log_gamma: np.ndarray,
    ) -> np.ndarray:
        r"""
        Evaluate an electron distribution on the gamma grid and return its log.

        Parameters
        ----------
        N : ~numpy.ndarray or callable
            Electron number density :math:`dN/d\gamma` in :math:`\mathrm{cm^{-3}}`.
            If callable, called as ``N(gamma)`` where ``gamma = exp(log_gamma)``.
            If an array, used directly; its last axis must match ``log_gamma.size``.
        log_gamma : ~numpy.ndarray
            Natural log of the Lorentz factor grid.

        Returns
        -------
        log_N : ~numpy.ndarray
            :math:`\ln N(\gamma)` evaluated on the grid.
        """
        if callable(N):
            N_values = ensure_in_units(N(np.exp(log_gamma)), u.cm**-3)
        else:
            N_values = ensure_in_units(N, u.cm**-3)
        return np.log(np.asarray(N_values, dtype="f8"))

    # ------------------------------------------ #
    # Public Radiative Quantities API
    # ------------------------------------------ #
    def compute_emissivity(
        self,
        nu: Union[float, np.ndarray, u.Quantity],
        B: Union[float, np.ndarray, u.Quantity],
        N: Union[np.ndarray, Callable],
        gamma: Union[np.ndarray, None] = None,
        alpha: Union[float, u.Quantity, None] = None,
        *,
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> u.Quantity:
        r"""
        Compute the comoving-frame synchrotron emissivity :math:`j_\nu`.

        Parameters
        ----------
        nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame frequency grid. Bare values are
            treated as Hz.
        B : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame magnetic field strength or
            broadcastable to the leading dimensions of ``N``. Bare values are
            treated as Gauss.
        N : ~numpy.ndarray or callable
            Comoving-frame electron distribution :math:`dN/d\gamma` in
            :math:`\mathrm{cm^{-3}}`. If callable,
            called as ``N(gamma)`` and must return shape ``(*zone_shape, n_gamma)``
            or broadcastable to it.
        gamma : ~numpy.ndarray or None, optional
            Explicit Lorentz factor grid. If ``None``, built from ``gamma_min``,
            ``gamma_max``, ``n_gamma``.
        alpha : float, ~astropy.units.Quantity, ~numpy.ndarray, or None, optional
            Comoving-frame pitch angle or broadcastable to
            it. Bare values are treated as radians. If ``None``, the
            pitch-angle-averaged kernel is used (requires
            :meth:`load_avg_first_kernel`).
        gamma_min : float, optional
            Grid lower bound (ignored if ``gamma`` is provided). Default ``1.0``.
        gamma_max : float, optional
            Grid upper bound (ignored if ``gamma`` is provided). Default ``1e8``.
        n_gamma : int, optional
            Number of grid points (ignored if ``gamma`` is provided). Default ``200``.

        Returns
        -------
        ~astropy.units.Quantity
            Comoving-frame emissivity, in
            :math:`\mathrm{erg\,s^{-1}\,cm^{-3}\,Hz^{-1}\,sr^{-1}}`.
        """
        nu_cgs = ensure_in_units(nu, u.Hz)
        B_cgs = ensure_in_units(B, u.G)
        log_gamma, log_weights = self._build_gamma_grid(gamma, gamma_min, gamma_max, n_gamma)
        log_N = self._resolve_log_N(N, log_gamma)

        if alpha is None:
            self.ensure_avg_first_kernel_loaded()
            log_j = self._compute_log_pa_emissivity(
                np.log(nu_cgs),
                np.log(B_cgs),
                log_N,
                log_gamma,
                log_weights,
            )
        else:
            self.ensure_first_kernel_loaded()
            sin_alpha = np.sin(ensure_in_units(alpha, u.rad))
            log_j = self._compute_log_emissivity(
                np.log(nu_cgs),
                np.log(B_cgs),
                log_N,
                log_gamma,
                log_weights,
                sin_alpha,
            )

        return np.exp(log_j) * (u.erg / (u.s * u.cm**3 * u.Hz * u.sr))

    def compute_absorption(
        self,
        nu: Union[float, np.ndarray, u.Quantity],
        B: Union[float, np.ndarray, u.Quantity],
        N: Union[np.ndarray, Callable],
        gamma: Union[np.ndarray, None] = None,
        alpha: Union[float, u.Quantity, None] = None,
        *,
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> u.Quantity:
        r"""
        Compute the comoving-frame synchrotron self-absorption coefficient :math:`|\alpha_\nu|`.

        Parameters
        ----------
        nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame frequency grid. Bare values are
            treated as Hz.
        B : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame magnetic field strength or
            broadcastable to the leading dimensions of ``N``. Bare values are
            treated as Gauss.
        N : ~numpy.ndarray or callable
            Comoving-frame electron distribution :math:`dN/d\gamma` in
            :math:`\mathrm{cm^{-3}}`. If callable,
            called as ``N(gamma)`` and must return shape ``(*zone_shape, n_gamma)``
            or broadcastable to it.
        gamma : ~numpy.ndarray or None, optional
            Explicit Lorentz factor grid. If ``None``, built from ``gamma_min``,
            ``gamma_max``, ``n_gamma``.
        alpha : float, ~astropy.units.Quantity, ~numpy.ndarray, or None, optional
            Comoving-frame pitch angle or broadcastable to
            it. Bare values are treated as radians. If ``None``, pitch-angle-averaged
            kernel is used.
        gamma_min : float, optional
            Grid lower bound (ignored if ``gamma`` is provided). Default ``1.0``.
        gamma_max : float, optional
            Grid upper bound (ignored if ``gamma`` is provided). Default ``1e8``.
        n_gamma : int, optional
            Number of grid points (ignored if ``gamma`` is provided). Default ``200``.

        Returns
        -------
        ~astropy.units.Quantity
            Comoving-frame absorption coefficient,
            in :math:`\mathrm{cm^{-1}}`.
        """
        nu_cgs = ensure_in_units(nu, u.Hz)
        B_cgs = ensure_in_units(B, u.G)
        log_gamma, log_weights = self._build_gamma_grid(gamma, gamma_min, gamma_max, n_gamma)
        log_N = self._resolve_log_N(N, log_gamma)

        if alpha is None:
            self.ensure_avg_first_kernel_loaded()
            log_alpha = self._compute_log_pa_absorption_coefficient(
                np.log(nu_cgs),
                np.log(B_cgs),
                log_N,
                log_gamma,
                log_weights,
            )
        else:
            self.ensure_first_kernel_loaded()
            sin_alpha = np.sin(ensure_in_units(alpha, u.rad))
            log_alpha = self._compute_log_absorption_coefficient(
                np.log(nu_cgs),
                np.log(B_cgs),
                log_N,
                log_gamma,
                log_weights,
                sin_alpha,
            )

        return np.exp(log_alpha) * u.cm**-1

    def compute_source_function(
        self,
        nu: Union[float, np.ndarray, u.Quantity],
        B: Union[float, np.ndarray, u.Quantity],
        N: Union[np.ndarray, Callable],
        gamma: Union[np.ndarray, None] = None,
        alpha: Union[float, u.Quantity, None] = None,
        *,
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> u.Quantity:
        r"""
        Compute the comoving-frame synchrotron source function :math:`S_\nu = j_\nu / \alpha_\nu`.

        Parameters
        ----------
        nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame frequency grid. Bare values are
            treated as Hz.
        B : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame magnetic field strength or
            broadcastable to the leading dimensions of ``N``. Bare values are
            treated as Gauss.
        N : ~numpy.ndarray or callable
            Comoving-frame electron distribution :math:`dN/d\gamma` in
            :math:`\mathrm{cm^{-3}}`. If callable,
            called as ``N(gamma)`` and must return shape ``(*zone_shape, n_gamma)``
            or broadcastable to it.
        gamma : ~numpy.ndarray or None, optional
            Explicit Lorentz factor grid. If ``None``, built from ``gamma_min``,
            ``gamma_max``, ``n_gamma``.
        alpha : float, ~astropy.units.Quantity, ~numpy.ndarray, or None, optional
            Comoving-frame pitch angle or broadcastable to
            it. Bare values are treated as radians. If ``None``, pitch-angle-averaged
            kernel is used.
        gamma_min : float, optional
            Grid lower bound (ignored if ``gamma`` is provided). Default ``1.0``.
        gamma_max : float, optional
            Grid upper bound (ignored if ``gamma`` is provided). Default ``1e8``.
        n_gamma : int, optional
            Number of grid points (ignored if ``gamma`` is provided). Default ``200``.

        Returns
        -------
        ~astropy.units.Quantity
            Comoving-frame source function, in
            :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}\,sr^{-1}}`.
        """
        nu_cgs = ensure_in_units(nu, u.Hz)
        B_cgs = ensure_in_units(B, u.G)
        log_gamma, log_weights = self._build_gamma_grid(gamma, gamma_min, gamma_max, n_gamma)
        log_N = self._resolve_log_N(N, log_gamma)

        if alpha is None:
            self.ensure_avg_first_kernel_loaded()
            log_j, log_a = self._compute_log_pa_emissivity_and_absorption(
                np.log(nu_cgs),
                np.log(B_cgs),
                log_N,
                log_gamma,
                log_weights,
            )
        else:
            self.ensure_first_kernel_loaded()
            sin_alpha = np.sin(ensure_in_units(alpha, u.rad))
            log_j, log_a = self._compute_log_emissivity_and_absorption(
                np.log(nu_cgs),
                np.log(B_cgs),
                log_N,
                log_gamma,
                log_weights,
                sin_alpha,
            )

        return np.exp(log_j - log_a) * (u.erg / (u.s * u.cm**2 * u.Hz * u.sr))

    def compute_rest_frame_specific_intensity(
        self,
        nu: Union[float, np.ndarray, u.Quantity],
        slab_depth: Union[float, np.ndarray, u.Quantity],
        B: Union[float, np.ndarray, u.Quantity],
        N: Union[np.ndarray, Callable],
        gamma: Union[np.ndarray, None] = None,
        alpha: Union[float, u.Quantity, None] = None,
        *,
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> u.Quantity:
        r"""
        Compute the comoving-frame specific intensity :math:`I'_{\nu'} = S'_{\nu'}(1 - e^{-\tau'_{\nu'}})`.

        Parameters
        ----------
        nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame frequency grid. Bare values are
            treated as Hz.
        slab_depth : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame line-of-sight transfer depth :math:`\ell'`, shape
            ``(*zone_shape)`` or broadcastable to the leading dimensions of ``N``.
            Bare values are treated as cm. If the source geometry is defined in the
            lab frame, divide by :math:`\Gamma` (for motion along the line of sight)
            before passing.
        B : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame magnetic field strength or
            broadcastable to the leading dimensions of ``N``. Bare values are
            treated as Gauss.
        N : ~numpy.ndarray or callable
            Comoving-frame electron distribution :math:`dN/d\gamma` in
            :math:`\mathrm{cm^{-3}}`. If callable,
            called as ``N(gamma)`` and must return shape ``(*zone_shape, n_gamma)``
            or broadcastable to it.
        gamma : ~numpy.ndarray or None, optional
            Explicit Lorentz factor grid. If ``None``, built from ``gamma_min``,
            ``gamma_max``, ``n_gamma``.
        alpha : float, ~astropy.units.Quantity, ~numpy.ndarray, or None, optional
            Comoving-frame pitch angle or broadcastable to
            it. Bare values are treated as radians. If ``None``, pitch-angle-averaged
            kernel is used.
        gamma_min : float, optional
            Grid lower bound (ignored if ``gamma`` is provided). Default ``1.0``.
        gamma_max : float, optional
            Grid upper bound (ignored if ``gamma`` is provided). Default ``1e8``.
        n_gamma : int, optional
            Number of grid points (ignored if ``gamma`` is provided). Default ``200``.

        Returns
        -------
        ~astropy.units.Quantity
            Comoving-frame specific intensity,
            in :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}\,sr^{-1}}`.
        """
        nu_cgs = ensure_in_units(nu, u.Hz)
        slab_depth_cgs = ensure_in_units(slab_depth, u.cm)
        B_cgs = ensure_in_units(B, u.G)
        log_gamma, log_weights = self._build_gamma_grid(gamma, gamma_min, gamma_max, n_gamma)
        log_N = self._resolve_log_N(N, log_gamma)

        if alpha is None:
            self.ensure_avg_first_kernel_loaded()
            log_I = self._compute_log_pa_rf_specific_intensity(
                np.log(nu_cgs),
                np.log(slab_depth_cgs),
                np.log(B_cgs),
                log_N,
                log_gamma,
                log_weights,
            )
        else:
            self.ensure_first_kernel_loaded()
            sin_alpha = np.sin(ensure_in_units(alpha, u.rad))
            log_I = self._compute_log_rf_specific_intensity(
                np.log(nu_cgs),
                np.log(slab_depth_cgs),
                np.log(B_cgs),
                log_N,
                log_gamma,
                log_weights,
                sin_alpha,
            )

        return np.exp(log_I) * (u.erg / (u.s * u.cm**2 * u.Hz * u.sr))

    def compute_specific_intensity(
        self,
        nu: Union[float, np.ndarray, u.Quantity],
        slab_depth: Union[float, np.ndarray, u.Quantity],
        B: Union[float, np.ndarray, u.Quantity],
        N: Union[np.ndarray, Callable],
        gamma: Union[np.ndarray, None] = None,
        z: float = 0,
        beta: float = 0,
        theta: Union[float, u.Quantity] = 0,
        alpha: Union[float, u.Quantity, None] = None,
        *,
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> u.Quantity:
        r"""
        Compute the observer-frame specific intensity including Doppler boost and redshift.

        Parameters
        ----------
        nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Observer-frame frequency grid. Bare values are
            treated as Hz.
        slab_depth : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame line-of-sight transfer depth :math:`\ell'`, shape
            ``(*zone_shape)`` or broadcastable to the leading dimensions of ``N``.
            Bare values are treated as cm. If the source geometry is defined in the
            lab frame, divide by :math:`\Gamma` (for motion along the line of sight)
            before passing.
        B : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame magnetic field strength or
            broadcastable to the leading dimensions of ``N``. Bare values are
            treated as Gauss.
        N : ~numpy.ndarray or callable
            Comoving-frame electron distribution :math:`dN/d\gamma` in
            :math:`\mathrm{cm^{-3}}`. If callable,
            called as ``N(gamma)`` and must return shape ``(*zone_shape, n_gamma)``
            or broadcastable to it.
        gamma : ~numpy.ndarray or None, optional
            Explicit Lorentz factor grid. If ``None``, built from ``gamma_min``,
            ``gamma_max``, ``n_gamma``.
        z : float, optional
            Source redshift. Default ``0``.
        beta : float, optional
            Bulk plasma velocity in units of :math:`c`. Default ``0``.
        theta : float or ~astropy.units.Quantity, optional
            Angle between the bulk velocity and the line of sight. Bare values are
            treated as radians. Default ``0``.
        alpha : float, ~astropy.units.Quantity, ~numpy.ndarray, or None, optional
            Comoving-frame pitch angle or broadcastable to
            it. Bare values are treated as radians. If ``None``, pitch-angle-averaged
            kernel is used.
        gamma_min : float, optional
            Grid lower bound (ignored if ``gamma`` is provided). Default ``1.0``.
        gamma_max : float, optional
            Grid upper bound (ignored if ``gamma`` is provided). Default ``1e8``.
        n_gamma : int, optional
            Number of grid points (ignored if ``gamma`` is provided). Default ``200``.

        Returns
        -------
        ~astropy.units.Quantity
            Observer-frame specific intensity,
            in :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}\,sr^{-1}}`.
        """
        nu_cgs = ensure_in_units(nu, u.Hz)
        slab_depth_cgs = ensure_in_units(slab_depth, u.cm)
        B_cgs = ensure_in_units(B, u.G)
        cos_theta = float(np.cos(ensure_in_units(theta, u.rad)))
        log_gamma, log_weights = self._build_gamma_grid(gamma, gamma_min, gamma_max, n_gamma)
        log_N = self._resolve_log_N(N, log_gamma)

        if alpha is None:
            self.ensure_avg_first_kernel_loaded()
            log_I = self._compute_log_pa_specific_intensity(
                np.log(nu_cgs),
                np.log(slab_depth_cgs),
                np.log(B_cgs),
                log_N,
                log_gamma,
                log_weights,
                z=z,
                beta=beta,
                cos_theta=cos_theta,
            )
        else:
            self.ensure_first_kernel_loaded()
            sin_alpha = np.sin(ensure_in_units(alpha, u.rad))
            log_I = self._compute_log_specific_intensity(
                np.log(nu_cgs),
                np.log(slab_depth_cgs),
                np.log(B_cgs),
                log_N,
                log_gamma,
                log_weights,
                z=z,
                beta=beta,
                cos_theta=cos_theta,
                sin_alpha=sin_alpha,
            )

        return np.exp(log_I) * (u.erg / (u.s * u.cm**2 * u.Hz * u.sr))

    def compute_flux_density(
        self,
        nu: Union[float, np.ndarray, u.Quantity],
        slab_depth: Union[float, np.ndarray, u.Quantity],
        B: Union[float, np.ndarray, u.Quantity],
        N: Union[np.ndarray, Callable],
        gamma: Union[np.ndarray, None] = None,
        A_eff: Union[float, np.ndarray, u.Quantity, None] = None,
        luminosity_distance: Union[u.Quantity, None] = None,
        angular_diameter_distance: Union[u.Quantity, None] = None,
        proper_distance: Union[u.Quantity, None] = None,
        z: float = 0,
        cosmology=None,
        beta: float = 0,
        theta: Union[float, u.Quantity] = 0,
        alpha: Union[float, u.Quantity, None] = None,
        *,
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> u.Quantity:
        r"""
        Compute the observer-frame spectral flux density :math:`F_\nu`.

        Parameters
        ----------
        nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Observer-frame frequency grid. Bare values are
            treated as Hz.
        slab_depth : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame line-of-sight transfer depth :math:`\ell'`, shape
            ``(*zone_shape)`` or broadcastable to the leading dimensions of ``N``.
            Bare values are treated as cm. If the source geometry is defined in the
            lab frame, divide by :math:`\Gamma` (for motion along the line of sight)
            before passing.
        B : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame magnetic field strength or
            broadcastable to the leading dimensions of ``N``. Bare values are
            treated as Gauss.
        N : ~numpy.ndarray or callable
            Comoving-frame electron distribution :math:`dN/d\gamma` in
            :math:`\mathrm{cm^{-3}}`. If callable,
            called as ``N(gamma)`` and must return shape ``(*zone_shape, n_gamma)``
            or broadcastable to it.
        gamma : ~numpy.ndarray or None, optional
            Explicit Lorentz factor grid. If ``None``, built from ``gamma_min``,
            ``gamma_max``, ``n_gamma``.
        A_eff : float, ~numpy.ndarray, or ~astropy.units.Quantity, optional
            Observer-frame projected emitting area or
            broadcastable to it. Bare values are treated as :math:`\mathrm{cm^2}`.
            Defaults to :math:`\pi \ell'^2` when ``None``.
        luminosity_distance : ~astropy.units.Quantity or None, optional
            Source luminosity distance. Exactly one of ``luminosity_distance``,
            ``angular_diameter_distance``, ``proper_distance``, or a non-zero
            ``z`` must be provided.
        angular_diameter_distance : ~astropy.units.Quantity or None, optional
            Source angular diameter distance.
        proper_distance : ~astropy.units.Quantity or None, optional
            Source comoving line-of-sight distance.
        z : float, optional
            Source redshift. Used for Doppler–redshift corrections and, when no
            distance is given explicitly, to compute :math:`D_A` from ``cosmology``.
            Default ``0``.
        cosmology : ~astropy.cosmology.FLRW or None, optional
            Cosmological model used to convert distances. Defaults to the
            trilobite configuration cosmology.
        beta : float, optional
            Bulk plasma velocity in units of :math:`c`. Default ``0``.
        theta : float or ~astropy.units.Quantity, optional
            Angle between the bulk velocity and the line of sight. Bare values are
            treated as radians. Default ``0``.
        alpha : float, ~astropy.units.Quantity, ~numpy.ndarray, or None, optional
            Comoving-frame pitch angle or broadcastable to
            it. Bare values are treated as radians. If ``None``, pitch-angle-averaged
            kernel is used.
        gamma_min : float, optional
            Grid lower bound (ignored if ``gamma`` is provided). Default ``1.0``.
        gamma_max : float, optional
            Grid upper bound (ignored if ``gamma`` is provided). Default ``1e8``.
        n_gamma : int, optional
            Number of grid points (ignored if ``gamma`` is provided). Default ``200``.

        Returns
        -------
        ~astropy.units.Quantity
            Observer-frame spectral flux density,
            in :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}}`.
        """
        nu_cgs = ensure_in_units(nu, u.Hz)
        slab_depth_cgs = ensure_in_units(slab_depth, u.cm)
        B_cgs = ensure_in_units(B, u.G)
        cos_theta = float(np.cos(ensure_in_units(theta, u.rad)))
        log_gamma, log_weights = self._build_gamma_grid(gamma, gamma_min, gamma_max, n_gamma)
        log_N = self._resolve_log_N(N, log_gamma)

        # Resolve angular diameter distance.
        dist = resolve_cosmological_distances(
            redshift=z if z != 0 else None,
            luminosity_distance=luminosity_distance,
            angular_diameter_distance=angular_diameter_distance,
            proper_distance=proper_distance,
            cosmology=cosmology,
        )
        D_A_cgs = dist["angular_diameter_distance"].to(u.cm).value

        if A_eff is None:
            A_eff_cgs = np.pi * slab_depth_cgs**2
        else:
            A_eff_cgs = ensure_in_units(A_eff, u.cm**2)

        if alpha is None:
            self.ensure_avg_first_kernel_loaded()
            log_F = self._compute_log_pa_flux_density(
                np.log(nu_cgs),
                np.log(slab_depth_cgs),
                np.log(B_cgs),
                log_N,
                log_gamma,
                log_weights,
                np.log(A_eff_cgs),
                np.log(D_A_cgs),
                z=z,
                beta=beta,
                cos_theta=cos_theta,
            )
        else:
            self.ensure_first_kernel_loaded()
            sin_alpha = np.sin(ensure_in_units(alpha, u.rad))
            log_F = self._compute_log_flux_density(
                np.log(nu_cgs),
                np.log(slab_depth_cgs),
                np.log(B_cgs),
                log_N,
                log_gamma,
                log_weights,
                np.log(A_eff_cgs),
                np.log(D_A_cgs),
                z=z,
                beta=beta,
                cos_theta=cos_theta,
                sin_alpha=sin_alpha,
            )

        return np.exp(log_F) * (u.erg / (u.s * u.cm**2 * u.Hz))

    def compute_brightness_temperature(
        self,
        nu: Union[float, np.ndarray, u.Quantity],
        slab_depth: Union[float, np.ndarray, u.Quantity],
        B: Union[float, np.ndarray, u.Quantity],
        N: Union[np.ndarray, Callable],
        gamma: Union[np.ndarray, None] = None,
        z: float = 0,
        beta: float = 0,
        theta: Union[float, u.Quantity] = 0,
        alpha: Union[float, u.Quantity, None] = None,
        *,
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> u.Quantity:
        r"""
        Compute the observer-frame synchrotron brightness temperature :math:`T_B`.

        The brightness temperature is defined through the Rayleigh--Jeans relation

        .. math::

            T_B
            =
            \frac{c^2 I_\nu}{2 k_B \nu^2},

        where :math:`I_\nu` is the observer-frame synchrotron specific intensity.

        Parameters
        ----------
        nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Observer-frame frequency grid. Bare values are
            treated as Hz.

        slab_depth : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame line-of-sight transfer depth :math:`\ell'`, shape
            ``(*zone_shape)`` or broadcastable to the leading dimensions of ``N``.
            Bare values are treated as cm.

        B : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame magnetic field strength or
            broadcastable to the leading dimensions of ``N``. Bare values are
            treated as Gauss.

        N : ~numpy.ndarray or callable
            Comoving-frame electron distribution :math:`dN/d\gamma` in
            :math:`\mathrm{cm^{-3}}`. If callable,
            called as ``N(gamma)`` and must return shape
            ``(*zone_shape, n_gamma)`` or broadcastable to it.

        gamma : ~numpy.ndarray or None, optional
            Explicit Lorentz factor grid. If ``None``, built from
            ``gamma_min``, ``gamma_max``, and ``n_gamma``.

        z : float, optional
            Source redshift. Default ``0``.

        beta : float, optional
            Bulk plasma velocity in units of :math:`c`. Default ``0``.

        theta : float or ~astropy.units.Quantity, optional
            Angle between the bulk velocity and the line of sight.
            Bare values are treated as radians. Default ``0``.

        alpha : float, ~astropy.units.Quantity, ~numpy.ndarray, or None, optional
            Comoving-frame pitch angle or
            broadcastable to it. Bare values are treated as radians.
            If ``None``, the pitch-angle-averaged kernel is used.

        gamma_min : float, optional
            Grid lower bound (ignored if ``gamma`` is provided).
            Default ``1.0``.

        gamma_max : float, optional
            Grid upper bound (ignored if ``gamma`` is provided).
            Default ``1e8``.

        n_gamma : int, optional
            Number of grid points (ignored if ``gamma`` is provided).
            Default ``200``.

        Returns
        -------
        ~astropy.units.Quantity
            Observer-frame brightness temperature, shape
            ``(*nu_shape, *zone_shape)``, in Kelvin.
        """
        nu_cgs = ensure_in_units(nu, u.Hz)
        slab_depth_cgs = ensure_in_units(slab_depth, u.cm)
        B_cgs = ensure_in_units(B, u.G)

        cos_theta = float(np.cos(ensure_in_units(theta, u.rad)))

        log_gamma, log_weights = self._build_gamma_grid(
            gamma,
            gamma_min,
            gamma_max,
            n_gamma,
        )

        log_N = self._resolve_log_N(N, log_gamma)

        if alpha is None:
            self.ensure_avg_first_kernel_loaded()

            log_T_B = self._compute_log_pa_brightness_temperature(
                np.log(nu_cgs),
                np.log(slab_depth_cgs),
                np.log(B_cgs),
                log_N,
                log_gamma,
                log_weights,
                z=z,
                beta=beta,
                cos_theta=cos_theta,
            )

        else:
            self.ensure_first_kernel_loaded()

            sin_alpha = np.sin(ensure_in_units(alpha, u.rad))

            log_T_B = self._compute_log_brightness_temperature(
                np.log(nu_cgs),
                np.log(slab_depth_cgs),
                np.log(B_cgs),
                log_N,
                log_gamma,
                log_weights,
                z=z,
                beta=beta,
                cos_theta=cos_theta,
                sin_alpha=sin_alpha,
            )

        return np.exp(log_T_B) * u.K

    def compute_rest_frame_luminosity_density(
        self,
        nu: Union[float, np.ndarray, u.Quantity],
        slab_depth: Union[float, np.ndarray, u.Quantity],
        B: Union[float, np.ndarray, u.Quantity],
        N: Union[np.ndarray, Callable],
        gamma: Union[np.ndarray, None] = None,
        A_eff: Union[float, np.ndarray, u.Quantity, None] = None,
        alpha: Union[float, u.Quantity, None] = None,
        *,
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> u.Quantity:
        r"""
        Compute the comoving-frame spectral luminosity density :math:`L'_{\nu'}`.

        Evaluates

        .. math::

            L'_{\nu'} = 4\pi\,A_\mathrm{eff}\,I'_{\nu'},

        where :math:`I'_{\nu'}` is the comoving-frame specific intensity obtained from
        the full radiative transfer solution, including synchrotron self-absorption.

        Parameters
        ----------
        nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame frequency grid. Bare values are
            treated as Hz.
        slab_depth : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame line-of-sight transfer depth :math:`\ell'`, shape
            ``(*zone_shape)`` or broadcastable to the leading dimensions of ``N``.
            Bare values are treated as cm.
        B : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame magnetic field strength or
            broadcastable to the leading dimensions of ``N``. Bare values are
            treated as Gauss.
        N : ~numpy.ndarray or callable
            Comoving-frame electron distribution :math:`dN/d\gamma` in
            :math:`\mathrm{cm^{-3}}`. If callable,
            called as ``N(gamma)`` and must return shape ``(*zone_shape, n_gamma)``
            or broadcastable to it.
        gamma : ~numpy.ndarray or None, optional
            Explicit Lorentz factor grid. If ``None``, built from ``gamma_min``,
            ``gamma_max``, ``n_gamma``.
        A_eff : float, ~numpy.ndarray, or ~astropy.units.Quantity, optional
            Effective emitting area :math:`A_\mathrm{eff}` in the comoving frame,
            shape ``(*zone_shape)`` or broadcastable to it. Bare values are treated
            as :math:`\mathrm{cm^2}`. Defaults to :math:`\pi \ell'^2` when ``None``.
        alpha : float, ~astropy.units.Quantity, ~numpy.ndarray, or None, optional
            Comoving-frame pitch angle or broadcastable to
            it. Bare values are treated as radians. If ``None``, the
            pitch-angle-averaged kernel is used.
        gamma_min : float, optional
            Grid lower bound (ignored if ``gamma`` is provided). Default ``1.0``.
        gamma_max : float, optional
            Grid upper bound (ignored if ``gamma`` is provided). Default ``1e8``.
        n_gamma : int, optional
            Number of grid points (ignored if ``gamma`` is provided). Default ``200``.

        Returns
        -------
        ~astropy.units.Quantity
            Comoving-frame spectral luminosity density, shape
            ``(*nu_shape, *zone_shape)``, in :math:`\mathrm{erg\,s^{-1}\,Hz^{-1}}`.
        """
        nu_cgs = ensure_in_units(nu, u.Hz)
        slab_depth_cgs = ensure_in_units(slab_depth, u.cm)
        B_cgs = ensure_in_units(B, u.G)
        log_gamma, log_weights = self._build_gamma_grid(gamma, gamma_min, gamma_max, n_gamma)
        log_N = self._resolve_log_N(N, log_gamma)

        if A_eff is None:
            A_eff_cgs = np.pi * slab_depth_cgs**2
        else:
            A_eff_cgs = ensure_in_units(A_eff, u.cm**2)

        log_nu = np.asarray(np.log(nu_cgs))
        log_A = np.asarray(np.log(A_eff_cgs))

        if alpha is None:
            self.ensure_avg_first_kernel_loaded()
            log_I = self._compute_log_pa_rf_specific_intensity(
                log_nu,
                np.log(slab_depth_cgs),
                np.log(B_cgs),
                log_N,
                log_gamma,
                log_weights,
            )
        else:
            self.ensure_first_kernel_loaded()
            sin_alpha = np.sin(ensure_in_units(alpha, u.rad))
            log_I = self._compute_log_rf_specific_intensity(
                log_nu,
                np.log(slab_depth_cgs),
                np.log(B_cgs),
                log_N,
                log_gamma,
                log_weights,
                sin_alpha,
            )

        log_A_v = log_A[(np.newaxis,) * log_nu.ndim]
        log_L = _LOG_4PI + log_A_v + log_I
        return np.exp(log_L) * (u.erg / (u.s * u.Hz))

    def compute_luminosity_density(
        self,
        nu: Union[float, np.ndarray, u.Quantity],
        slab_depth: Union[float, np.ndarray, u.Quantity],
        B: Union[float, np.ndarray, u.Quantity],
        N: Union[np.ndarray, Callable],
        gamma: Union[np.ndarray, None] = None,
        A_eff: Union[float, np.ndarray, u.Quantity, None] = None,
        z: float = 0,
        beta: float = 0,
        theta: Union[float, u.Quantity] = 0,
        alpha: Union[float, u.Quantity, None] = None,
        *,
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> u.Quantity:
        r"""
        Compute the observer-frame spectral luminosity density :math:`L_\nu`.

        Applies the relativistic Doppler and cosmological redshift correction to the
        comoving-frame spectral luminosity:

        .. math::

            L_\nu = \left(\frac{\mathcal{D}}{1+z}\right)^3 L'_{\nu'},

        where

        .. math::

            \mathcal{D} = \frac{1}{\Gamma(1-\beta\cos\theta)}

        is the Doppler factor and :math:`L'_{\nu'}` is the comoving-frame spectral
        luminosity density computed by :meth:`compute_rest_frame_luminosity_density`.

        Parameters
        ----------
        nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Observer-frame frequency grid. Bare values are
            treated as Hz.
        slab_depth : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame line-of-sight transfer depth :math:`\ell'`, shape
            ``(*zone_shape)`` or broadcastable to the leading dimensions of ``N``.
            Bare values are treated as cm.
        B : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame magnetic field strength or
            broadcastable to the leading dimensions of ``N``. Bare values are
            treated as Gauss.
        N : ~numpy.ndarray or callable
            Comoving-frame electron distribution :math:`dN/d\gamma` in
            :math:`\mathrm{cm^{-3}}`. If callable,
            called as ``N(gamma)`` and must return shape ``(*zone_shape, n_gamma)``
            or broadcastable to it.
        gamma : ~numpy.ndarray or None, optional
            Explicit Lorentz factor grid. If ``None``, built from ``gamma_min``,
            ``gamma_max``, ``n_gamma``.
        A_eff : float, ~numpy.ndarray, or ~astropy.units.Quantity, optional
            Effective emitting area in the comoving frame
            or broadcastable to it. Bare values are treated as
            :math:`\mathrm{cm^2}`. Defaults to :math:`\pi \ell'^2` when ``None``.
        z : float, optional
            Cosmological redshift of the source. Default ``0``.
        beta : float, optional
            Bulk plasma velocity in units of :math:`c`. Default ``0``.
        theta : float or ~astropy.units.Quantity, optional
            Angle between the bulk velocity and the line of sight. Bare values are
            treated as radians. Default ``0``.
        alpha : float, ~astropy.units.Quantity, ~numpy.ndarray, or None, optional
            Comoving-frame pitch angle or broadcastable to
            it. Bare values are treated as radians. If ``None``, the
            pitch-angle-averaged kernel is used.
        gamma_min : float, optional
            Grid lower bound (ignored if ``gamma`` is provided). Default ``1.0``.
        gamma_max : float, optional
            Grid upper bound (ignored if ``gamma`` is provided). Default ``1e8``.
        n_gamma : int, optional
            Number of grid points (ignored if ``gamma`` is provided). Default ``200``.

        Returns
        -------
        ~astropy.units.Quantity
            Observer-frame spectral luminosity density, shape
            ``(*nu_shape, *zone_shape)``, in :math:`\mathrm{erg\,s^{-1}\,Hz^{-1}}`.
        """
        nu_cgs = ensure_in_units(nu, u.Hz)
        slab_depth_cgs = ensure_in_units(slab_depth, u.cm)
        B_cgs = ensure_in_units(B, u.G)
        cos_theta = float(np.cos(ensure_in_units(theta, u.rad)))
        log_gamma, log_weights = self._build_gamma_grid(gamma, gamma_min, gamma_max, n_gamma)
        log_N = self._resolve_log_N(N, log_gamma)

        if A_eff is None:
            A_eff_cgs = np.pi * slab_depth_cgs**2
        else:
            A_eff_cgs = ensure_in_units(A_eff, u.cm**2)

        log_gamma_bulk = np.log(1.0 / np.sqrt(1 - beta**2))
        log_Doppler = -log_gamma_bulk - np.log1p(-beta * cos_theta)
        log_correction_factor = log_Doppler - np.log1p(z)

        log_nu = np.asarray(np.log(nu_cgs))
        log_nu_rf = log_nu - log_correction_factor
        log_A = np.asarray(np.log(A_eff_cgs))

        if alpha is None:
            self.ensure_avg_first_kernel_loaded()
            log_I_rf = self._compute_log_pa_rf_specific_intensity(
                log_nu_rf,
                np.log(slab_depth_cgs),
                np.log(B_cgs),
                log_N,
                log_gamma,
                log_weights,
            )
        else:
            self.ensure_first_kernel_loaded()
            sin_alpha = np.sin(ensure_in_units(alpha, u.rad))
            log_I_rf = self._compute_log_rf_specific_intensity(
                log_nu_rf,
                np.log(slab_depth_cgs),
                np.log(B_cgs),
                log_N,
                log_gamma,
                log_weights,
                sin_alpha,
            )

        log_A_v = log_A[(np.newaxis,) * log_nu.ndim]
        log_L_rf = _LOG_4PI + log_A_v + log_I_rf
        log_L = log_L_rf + 3.0 * log_correction_factor
        return np.exp(log_L) * (u.erg / (u.s * u.Hz))

    def compute_isotropic_luminosity(
        self,
        nu: Union[float, np.ndarray, u.Quantity],
        slab_depth: Union[float, np.ndarray, u.Quantity],
        B: Union[float, np.ndarray, u.Quantity],
        N: Union[np.ndarray, Callable],
        gamma: Union[np.ndarray, None] = None,
        A_eff: Union[float, np.ndarray, u.Quantity, None] = None,
        luminosity_distance: Union[u.Quantity, None] = None,
        angular_diameter_distance: Union[u.Quantity, None] = None,
        proper_distance: Union[u.Quantity, None] = None,
        z: float = 0,
        cosmology=None,
        beta: float = 0,
        theta: Union[float, u.Quantity] = 0,
        alpha: Union[float, u.Quantity, None] = None,
        *,
        gamma_min: float = 1.0,
        gamma_max: float = 1e8,
        n_gamma: int = 200,
    ) -> u.Quantity:
        r"""
        Compute the isotropic-equivalent spectral luminosity :math:`L_{\nu,\mathrm{iso}}`.

        Defined from the observed spectral flux density as

        .. math::

            L_{\nu,\mathrm{iso}} = 4\pi\,D_L^2\,F_\nu,

        where :math:`D_L` is the luminosity distance. This quantity equals the true
        luminosity only for an isotropic, non-relativistic source; in general it
        is the luminosity that an observer would infer assuming isotropy.

        Parameters
        ----------
        nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Observer-frame frequency grid. Bare values are
            treated as Hz.
        slab_depth : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame line-of-sight transfer depth :math:`\ell'`, shape
            ``(*zone_shape)`` or broadcastable to the leading dimensions of ``N``.
            Bare values are treated as cm.
        B : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Comoving-frame magnetic field strength or
            broadcastable to the leading dimensions of ``N``. Bare values are
            treated as Gauss.
        N : ~numpy.ndarray or callable
            Comoving-frame electron distribution :math:`dN/d\gamma` in
            :math:`\mathrm{cm^{-3}}`. If callable,
            called as ``N(gamma)`` and must return shape ``(*zone_shape, n_gamma)``
            or broadcastable to it.
        gamma : ~numpy.ndarray or None, optional
            Explicit Lorentz factor grid. If ``None``, built from ``gamma_min``,
            ``gamma_max``, ``n_gamma``.
        A_eff : float, ~numpy.ndarray, or ~astropy.units.Quantity, optional
            Observer-frame projected emitting area or
            broadcastable to it. Bare values are treated as :math:`\mathrm{cm^2}`.
            Defaults to :math:`\pi \ell'^2` when ``None``.
        luminosity_distance : ~astropy.units.Quantity or None, optional
            Source luminosity distance :math:`D_L`. Exactly one of
            ``luminosity_distance``, ``angular_diameter_distance``,
            ``proper_distance``, or a non-zero ``z`` must be provided.
        angular_diameter_distance : ~astropy.units.Quantity or None, optional
            Source angular diameter distance.
        proper_distance : ~astropy.units.Quantity or None, optional
            Source comoving line-of-sight distance.
        z : float, optional
            Source redshift. Used for Doppler–redshift corrections and, when no
            distance is given explicitly, to compute :math:`D_L` from ``cosmology``.
            Default ``0``.
        cosmology : ~astropy.cosmology.FLRW or None, optional
            Cosmological model used to convert distances. Defaults to the
            trilobite configuration cosmology.
        beta : float, optional
            Bulk plasma velocity in units of :math:`c`. Default ``0``.
        theta : float or ~astropy.units.Quantity, optional
            Angle between the bulk velocity and the line of sight. Bare values are
            treated as radians. Default ``0``.
        alpha : float, ~astropy.units.Quantity, ~numpy.ndarray, or None, optional
            Comoving-frame pitch angle or broadcastable to
            it. Bare values are treated as radians. If ``None``, the
            pitch-angle-averaged kernel is used.
        gamma_min : float, optional
            Grid lower bound (ignored if ``gamma`` is provided). Default ``1.0``.
        gamma_max : float, optional
            Grid upper bound (ignored if ``gamma`` is provided). Default ``1e8``.
        n_gamma : int, optional
            Number of grid points (ignored if ``gamma`` is provided). Default ``200``.

        Returns
        -------
        ~astropy.units.Quantity
            Isotropic-equivalent spectral luminosity, shape
            ``(*nu_shape, *zone_shape)``, in :math:`\mathrm{erg\,s^{-1}\,Hz^{-1}}`.
        """
        nu_cgs = ensure_in_units(nu, u.Hz)
        slab_depth_cgs = ensure_in_units(slab_depth, u.cm)
        B_cgs = ensure_in_units(B, u.G)
        cos_theta = float(np.cos(ensure_in_units(theta, u.rad)))
        log_gamma, log_weights = self._build_gamma_grid(gamma, gamma_min, gamma_max, n_gamma)
        log_N = self._resolve_log_N(N, log_gamma)

        dist = resolve_cosmological_distances(
            redshift=z if z != 0 else None,
            luminosity_distance=luminosity_distance,
            angular_diameter_distance=angular_diameter_distance,
            proper_distance=proper_distance,
            cosmology=cosmology,
        )
        D_A_cgs = dist["angular_diameter_distance"].to(u.cm).value
        D_L_cgs = dist["luminosity_distance"].to(u.cm).value

        if A_eff is None:
            A_eff_cgs = np.pi * slab_depth_cgs**2
        else:
            A_eff_cgs = ensure_in_units(A_eff, u.cm**2)

        if alpha is None:
            self.ensure_avg_first_kernel_loaded()
            log_F = self._compute_log_pa_flux_density(
                np.log(nu_cgs),
                np.log(slab_depth_cgs),
                np.log(B_cgs),
                log_N,
                log_gamma,
                log_weights,
                np.log(A_eff_cgs),
                np.log(D_A_cgs),
                z=z,
                beta=beta,
                cos_theta=cos_theta,
            )
        else:
            self.ensure_first_kernel_loaded()
            sin_alpha = np.sin(ensure_in_units(alpha, u.rad))
            log_F = self._compute_log_flux_density(
                np.log(nu_cgs),
                np.log(slab_depth_cgs),
                np.log(B_cgs),
                log_N,
                log_gamma,
                log_weights,
                np.log(A_eff_cgs),
                np.log(D_A_cgs),
                z=z,
                beta=beta,
                cos_theta=cos_theta,
                sin_alpha=sin_alpha,
            )

        log_L_iso = _LOG_4PI + 2.0 * np.log(D_L_cgs) + log_F
        return np.exp(log_L_iso) * (u.erg / (u.s * u.Hz))
