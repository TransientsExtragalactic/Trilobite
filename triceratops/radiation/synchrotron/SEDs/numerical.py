r"""
Numerical synchrotron SED engine based on kernel interpolation.

This module provides :class:`NumericalSynchrotronEngine`, which evaluates synchrotron
emissivity, self-absorption, and spectral flux density for **arbitrary electron energy
distributions** :math:`N(\gamma)`.  Unlike the analytic power-law closure in
:mod:`~triceratops.radiation.synchrotron.SEDs.one_zone`, this engine integrates the
full synchrotron kernel numerically, making it suitable for cooling-modified,
thermal, or other non-power-law distributions.
"""

from collections.abc import Callable
from typing import Union

import numpy as np
from astropy import units as u
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.special import logsumexp

from triceratops.physics_utils import resolve_cosmological_distances
from triceratops.radiation.synchrotron.core import _log_averaged_first_synchrotron_kernel, _log_first_synchrotron_kernel
from triceratops.radiation.synchrotron.utils import _log_c_1_gamma_cgs, _log_chi_abs_cgs, _log_chi_cgs
from triceratops.utils.misc_utils import ensure_in_units


class NumericalSynchrotronEngine:
    r"""
    Numerical engine for evaluating synchrotron kernel functions via interpolation.

    Kernels must be explicitly loaded before use. Call :meth:`load_first_kernel` to
    build and cache the interpolator for :math:`F(x)`. Re-call to swap to a different
    grid. Kernel evaluation is then a fast spline lookup rather than a numerical integral.

    """

    def __init__(self):
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

    # ------------------------------------------ #
    # Loading
    # ------------------------------------------ #
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
        derivative : bool, optional
            Whether to compute and cache the derivative of the kernel with respect to log(x).
            Passed to :func:`_log_first_synchrotron_kernel`. Default is True.
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
        derivative : bool, optional
            Whether to compute and cache the derivative of the kernel with respect to log(x).
            Passed to :func:`_log_first_synchrotron_kernel`. Default is True.
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

    # ------------------------------------------ #
    # Private Kernel Evaluation
    # ------------------------------------------ #
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
        self.ensure_first_kernel_loaded()
        return self._interp_dfirst_kernel_dlog_x(log_x, **kwargs)

    def _compute_avg_first_kernel(self, log_x: Union[float, np.ndarray], **kwargs) -> Union[float, np.ndarray]:
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
        self.ensure_avg_first_kernel_loaded()
        return self._interp_avg_first_kernel(log_x, **kwargs)

    def _compute_avg_first_kernel_derivative(self, log_x, **kwargs):
        self.ensure_avg_first_kernel_loaded()
        return self._interp_davg_first_kernel_dlog_x(log_x, **kwargs)

    # ------------------------------------------ #
    # Public Kernel Evaluation
    # ------------------------------------------ #
    def is_first_kernel_loaded(self) -> bool:
        r"""
        Check if the first kernel interpolator is loaded.

        Returns
        -------
        bool
            True if the first kernel is loaded, False otherwise.
        """
        return self._interp_first_kernel is not None

    def is_avg_first_kernel_loaded(self) -> bool:
        r"""
        Check if the pitch-angle averaged first kernel interpolator is loaded.

        Returns
        -------
        bool
            True if the pitch-angle averaged first kernel is loaded, False otherwise.
        """
        return self._interp_avg_first_kernel is not None

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
        F : float or ~numpy.ndarray
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

    def compute_first_kernel_derivative(self, x):
        pass

    def compute_avg_first_kernel(self, x):
        pass

    def compute_avg_first_kernel_derivative(self, x):
        pass

    # ------------------------------------------ #
    # Radiative Quantities API
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
        log_nu : float or ~numpy.ndarray, shape ``(*nu_shape)``
            Natural log of the frequency grid in CGS (Hz).
        log_B : float or ~numpy.ndarray
            Natural log of the magnetic field strength in CGS (G). Must be broadcastable
            against ``log_nu``; scalar is the common case.
        log_N : float or ~numpy.ndarray, shape ``(*batch_shape, n_gamma)``
            Natural log of the electron number density evaluated on the Lorentz factor
            grid. The last axis is the gamma integration axis; any leading axes define
            an independent batch of distributions.
        log_gamma : float or ~numpy.ndarray, shape ``(n_gamma,)``
            Natural log of the Lorentz factor grid.
        log_weights : float or ~numpy.ndarray, shape ``(n_gamma,)``
            Quadrature weights :math:`w_i = \gamma_i\,\Delta\!\log\gamma_i` in log-space,
            used to convert the sum over :math:`d\gamma` into a Riemann sum.
        sin_alpha : float or ~numpy.ndarray
            Sine of the pitch angle. Must be broadcastable against ``log_nu``; scalar is
            the common case.

        Returns
        -------
        log_emissivity : ~numpy.ndarray, shape ``(*nu_shape, *batch_shape)``
            :math:`\log j_\nu` in CGS (:math:`\mathrm{erg\,s^{-1}\,cm^{-3}\,Hz^{-1}\,sr^{-1}}`).

        Notes
        -----
        **Broadcasting.** ``log_B`` and ``sin_alpha`` must broadcast against ``log_nu``
        and define the *spectral* axis. ``log_N`` adds a separate *batch* axis: its
        leading dimensions ``(*batch_shape,)`` are independent of the spectral
        dimensions, so the output has shape ``(*nu_shape, *batch_shape)``. Passing
        ``log_N`` with shape ``(n_gamma,)`` (no batch dimensions) returns output with
        shape ``(*nu_shape,)``.
        """
        log_nu = np.asarray(log_nu, dtype="f8")
        log_B = np.asarray(log_B, dtype="f8")
        log_gamma = np.asarray(log_gamma, dtype="f8")
        log_N = np.asarray(log_N, dtype="f8")
        log_weights = np.asarray(log_weights, dtype="f8")
        log_sin_alpha = np.log(np.asarray(sin_alpha, dtype="f8"))

        nu_shape = log_nu.shape
        n_gamma = log_gamma.size
        batch_shape = log_N.shape[:-1]
        n_nu = log_nu.size
        n_batch = log_N.size // n_gamma

        # Broadcast B and sin_alpha to nu_shape, then flatten to 1D.
        log_B_flat = np.broadcast_to(log_B, nu_shape).ravel()
        log_sin_alpha_flat = np.broadcast_to(log_sin_alpha, nu_shape).ravel()
        log_nu_flat = log_nu.ravel()

        # log_x: (n_nu, n_gamma) — the dimensionless frequency ratio for every (nu, gamma) pair.
        log_x = (
            log_nu_flat[:, np.newaxis]
            - 2.0 * log_gamma[np.newaxis, :]
            - log_B_flat[:, np.newaxis]
            - _log_c_1_gamma_cgs
            - log_sin_alpha_flat[:, np.newaxis]
        )

        # log F(x): (n_nu, n_gamma) — linear interpolation on the tabulated kernel.
        log_F = np.interp(log_x.ravel(), self._log_x_first_kernel, self._log_first_kernel).reshape(n_nu, n_gamma)

        # Integrand: (n_nu, n_batch, n_gamma).
        log_N_2d = log_N.reshape(n_batch, n_gamma)
        log_integrand = log_F[:, np.newaxis, :] + log_N_2d[np.newaxis, :, :] + log_weights[np.newaxis, np.newaxis, :]

        # Integrate over gamma, add prefactors: (n_nu, n_batch).
        log_emissivity_2d = (
            logsumexp(log_integrand, axis=-1)
            + _log_chi_cgs
            + log_B_flat[:, np.newaxis]
            + log_sin_alpha_flat[:, np.newaxis]
        )

        return log_emissivity_2d.reshape(*nu_shape, *batch_shape)

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
        log_nu : float or ~numpy.ndarray, shape ``(*nu_shape)``
            Natural log of the frequency grid in CGS (Hz).
        log_B : float or ~numpy.ndarray
            Natural log of the magnetic field strength in CGS (G). Must be broadcastable
            against ``log_nu``; scalar is the common case.
        log_N : float or ~numpy.ndarray, shape ``(*batch_shape, n_gamma)``
            Natural log of the electron number density on the Lorentz factor grid.
            The last axis is the gamma integration axis; any leading axes define a batch
            of independent distributions.
        log_gamma : float or ~numpy.ndarray, shape ``(n_gamma,)``
            Natural log of the Lorentz factor grid.
        log_weights : float or ~numpy.ndarray, shape ``(n_gamma,)``
            Quadrature weights :math:`w_i = \gamma_i\,\Delta\!\log\gamma_i` in log-space.
        sin_alpha : float or ~numpy.ndarray
            Sine of the pitch angle. Must be broadcastable against ``log_nu``; scalar is
            the common case.

        Returns
        -------
        log_alpha : ~numpy.ndarray, shape ``(*nu_shape, *batch_shape)``
            :math:`\log|\alpha_\nu|` in CGS (:math:`\mathrm{cm^{-1}}`).

        Notes
        -----
        **Broadcasting.** ``log_B`` and ``sin_alpha`` must broadcast against ``log_nu``
        and define the *spectral* axis. ``log_N`` adds a separate *batch* axis: its
        leading dimensions ``(*batch_shape,)`` are independent of the spectral
        dimensions, so the output has shape ``(*nu_shape, *batch_shape)``. Passing
        ``log_N`` with shape ``(n_gamma,)`` (no batch dimensions) returns output with
        shape ``(*nu_shape,)``.
        """
        log_nu = np.asarray(log_nu, dtype="f8")
        log_B = np.asarray(log_B, dtype="f8")
        log_gamma = np.asarray(log_gamma, dtype="f8")
        log_N = np.asarray(log_N, dtype="f8")
        log_weights = np.asarray(log_weights, dtype="f8")
        log_sin_alpha = np.log(np.asarray(sin_alpha, dtype="f8"))

        nu_shape = log_nu.shape
        n_gamma = log_gamma.size
        batch_shape = log_N.shape[:-1]
        n_nu = log_nu.size
        n_batch = log_N.size // n_gamma

        # Broadcast B and sin_alpha to nu_shape, then flatten to 1D.
        log_B_flat = np.broadcast_to(log_B, nu_shape).ravel()
        log_sin_alpha_flat = np.broadcast_to(log_sin_alpha, nu_shape).ravel()
        log_nu_flat = log_nu.ravel()

        # log_x: (n_nu, n_gamma)
        log_x = (
            log_nu_flat[:, np.newaxis]
            - 2.0 * log_gamma[np.newaxis, :]
            - log_B_flat[:, np.newaxis]
            - _log_c_1_gamma_cgs
            - log_sin_alpha_flat[:, np.newaxis]
        )

        # Evaluate log F and d(log F)/d(log x): (n_nu, n_gamma).
        log_F = np.interp(log_x.ravel(), self._log_x_first_kernel, self._log_first_kernel).reshape(n_nu, n_gamma)
        d_log_F_d_log_x = np.interp(log_x.ravel(), self._log_x_first_kernel, self._log_dfirst_kernel_dlog_x).reshape(
            n_nu, n_gamma
        )

        # IBP kernel: log(F - xF') = log F + log(1 - d log F / d log x): (n_nu, n_gamma).
        log_kernel = log_F + np.log(np.maximum(1.0 - d_log_F_d_log_x, np.finfo("f8").tiny))

        # Integrand: (n_nu, n_batch, n_gamma). The - log_gamma term converts d gamma to d log gamma.
        log_N_2d = log_N.reshape(n_batch, n_gamma)
        log_integrand = (
            log_kernel[:, np.newaxis, :]
            + log_N_2d[np.newaxis, :, :]
            + log_weights[np.newaxis, np.newaxis, :]
            - log_gamma[np.newaxis, np.newaxis, :]
        )

        # Integrate over gamma, add prefactors: (n_nu, n_batch).
        log_alpha_2d = (
            _log_chi_abs_cgs
            + log_B_flat[:, np.newaxis]
            + log_sin_alpha_flat[:, np.newaxis]
            - 2.0 * log_nu_flat[:, np.newaxis]
            + logsumexp(log_integrand, axis=-1)
        )

        return log_alpha_2d.reshape(*nu_shape, *batch_shape)

    def _compute_log_rf_specific_intensity(
        self,
        log_nu: Union[float, np.ndarray],
        log_R: Union[float, np.ndarray],
        log_B: Union[float, np.ndarray],
        log_N: Union[float, np.ndarray],
        log_gamma: Union[float, np.ndarray],
        log_weights: Union[float, np.ndarray],
        sin_alpha: Union[float, np.ndarray],
    ) -> np.ndarray:
        r"""
        Compute :math:`\log I_\nu` in the rest frame of the source.

        This function computes the rest-frame specific intensity via the radiative transfer solution
        :math:`I_\nu = S_\nu (1 - e^{-\tau_\nu})`, where :math:`S_\nu = j_\nu / \alpha_\nu` is the
        source function and :math:`\tau_\nu = \alpha_\nu R` is the optical depth.

        Parameters
        ----------
        log_nu : float or ~numpy.ndarray, shape ``(*nu_shape)``
            Natural log of the observed frequency grid in CGS (Hz).
        log_R : float or ~numpy.ndarray
            Natural log of the source radius in CGS (cm). Must be broadcastable against
            ``log_nu``; scalar is the common case.
        log_B : float or ~numpy.ndarray
            Natural log of the magnetic field strength in CGS (G). Must be broadcastable
            against ``log_nu``; scalar is the common case.
        log_N : float or ~numpy.ndarray, shape ``(*batch_shape, n_gamma)``
            Natural log of the electron number density on the Lorentz factor grid.
        log_gamma : float or ~numpy.ndarray, shape ``(n_gamma,)``
            Natural log of the Lorentz factor grid.
        log_weights : float or ~numpy.ndarray, shape ``(n_gamma,)``
            Quadrature weights :math:`w_i = \gamma_i\,\Delta\!\log\gamma_i` in log-space.
        sin_alpha : float or ~numpy.ndarray
            Sine of the pitch angle. Must be broadcastable against ``log_nu``; scalar is
            the common case.

        Returns
        -------
        log_intensity : ~numpy.ndarray, shape ``(*nu_shape, *batch_shape)``
            :math:`\log I_\nu` in CGS (:math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}\,sr^{-1}}`).
        """
        log_nu = np.asarray(log_nu, dtype="f8")
        log_R = np.asarray(log_R, dtype="f8")

        log_emissivity = self._compute_log_emissivity(log_nu, log_B, log_N, log_gamma, log_weights, sin_alpha)
        log_absorption = self._compute_log_absorption_coefficient(
            log_nu, log_B, log_N, log_gamma, log_weights, sin_alpha
        )

        # Expand log_R (nu-shaped) to broadcast against the trailing batch dims of the output.
        n_batch_dims = log_emissivity.ndim - log_nu.ndim
        log_R_view = log_R.reshape(log_R.shape + (1,) * n_batch_dims)

        tau = np.exp(np.clip(log_absorption + log_R_view, -np.inf, 500))
        return log_emissivity - log_absorption + np.log(-np.expm1(-tau))

    def _compute_log_specific_intensity(
        self,
        log_nu: Union[float, np.ndarray],
        log_R: Union[float, np.ndarray],
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
        Compute :math:`\log I_\nu` as observed, including bulk Doppler boost and cosmological redshift.

        Transforms the observed frequency to the source rest frame, evaluates the rest-frame
        specific intensity via :meth:`_compute_log_rf_specific_intensity`, then applies the
        :math:`\mathcal{D}^3` Doppler–redshift correction.

        Parameters
        ----------
        log_nu : float or ~numpy.ndarray, shape ``(*nu_shape)``
            Natural log of the observed frequency grid in CGS (Hz).
        log_R : float or ~numpy.ndarray
            Natural log of the source radius in CGS (cm). Must be broadcastable against
            ``log_nu``; scalar is the common case.
        log_B : float or ~numpy.ndarray
            Natural log of the magnetic field strength in CGS (G). Must be broadcastable
            against ``log_nu``; scalar is the common case.
        log_N : float or ~numpy.ndarray, shape ``(*batch_shape, n_gamma)``
            Natural log of the electron number density on the Lorentz factor grid.
        log_gamma : float or ~numpy.ndarray, shape ``(n_gamma,)``
            Natural log of the Lorentz factor grid.
        log_weights : float or ~numpy.ndarray, shape ``(n_gamma,)``
            Quadrature weights :math:`w_i = \gamma_i\,\Delta\!\log\gamma_i` in log-space.
        z : float
            Source redshift.
        beta : float
            Bulk velocity as a fraction of :math:`c`.
        cos_theta : float
            Cosine of the angle between the bulk velocity and the line of sight.
        sin_alpha : float or ~numpy.ndarray
            Sine of the pitch angle. Must be broadcastable against ``log_nu``; scalar is
            the common case.

        Returns
        -------
        log_intensity : ~numpy.ndarray, shape ``(*nu_shape, *batch_shape)``
            :math:`\log I_\nu` in CGS (:math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}\,sr^{-1}}`).
        """
        # Compute the doppler-factor corrections.
        log_gamma_bulk = np.log(1.0 / np.sqrt(1 - beta**2))
        log_Doppler = -log_gamma_bulk - np.log1p(beta * cos_theta)
        log_correction_factor = log_Doppler - np.log1p(z)

        # Now compute the corrected log_nu for the rest frame. In the rest frame, we have
        # nu_rf = (1+z)/D * nu_obs, so log_nu_rf = log_nu - log_correction_factor.
        log_nu_rf = log_nu - log_correction_factor

        # Compute the rest-frame specific intensity first.
        log_rf_intensity = self._compute_log_rf_specific_intensity(
            log_nu_rf, log_R, log_B, log_N, log_gamma, log_weights, sin_alpha
        )

        # Now apply the Doppler and redshift corrections to get the observed specific intensity.
        log_specific_intensity = log_rf_intensity + 3 * log_correction_factor

        return log_specific_intensity

    def _compute_log_flux_density(
        self,
        log_nu: Union[float, np.ndarray],
        log_R: Union[float, np.ndarray],
        log_B: Union[float, np.ndarray],
        log_N: Union[float, np.ndarray],
        log_gamma: Union[float, np.ndarray],
        log_weights: Union[float, np.ndarray],
        log_A_eff: Union[float, np.ndarray],
        log_D_A: Union[float, np.ndarray],
        z: float = 0,
        beta: float = 0,
        cos_theta: float = 1.0,
        sin_alpha: Union[float, np.ndarray] = 1.0,
    ) -> np.ndarray:
        r"""
        Compute :math:`\log F_\nu`, the log of the observed spectral flux density.

        Applies bulk Doppler and cosmological redshift corrections to the rest-frame
        specific intensity, then integrates over the source solid angle:

        .. math::

            F_\nu = \frac{A_\mathrm{eff}}{D_A^2}\,I_{\nu,\mathrm{obs}}

        where :math:`I_{\nu,\mathrm{obs}} = \mathcal{D}^3\,I_{\nu_\mathrm{rf}}` and
        :math:`\mathcal{D} = [\Gamma(1 + \beta\cos\theta)(1+z)]^{-1}` is the combined
        Doppler–redshift factor.

        Parameters
        ----------
        log_nu : float or ~numpy.ndarray, shape ``(*nu_shape)``
            Natural log of the observed frequency grid in CGS (Hz).
        log_R : float or ~numpy.ndarray
            Natural log of the source radius in CGS (cm). Must be broadcastable against
            ``log_nu``; scalar is the common case.
        log_B : float or ~numpy.ndarray
            Natural log of the magnetic field strength in CGS (G). Must be broadcastable
            against ``log_nu``; scalar is the common case.
        log_N : float or ~numpy.ndarray, shape ``(*batch_shape, n_gamma)``
            Natural log of the electron number density on the Lorentz factor grid.
        log_gamma : float or ~numpy.ndarray, shape ``(n_gamma,)``
            Natural log of the Lorentz factor grid.
        log_weights : float or ~numpy.ndarray, shape ``(n_gamma,)``
            Quadrature weights :math:`w_i = \gamma_i\,\Delta\!\log\gamma_i` in log-space.
        log_A_eff : float or ~numpy.ndarray
            Natural log of the effective emitting area in CGS (cm\ :sup:`2`). Must be
            broadcastable against ``log_nu``; scalar is the common case.
        log_D_A : float or ~numpy.ndarray
            Natural log of the angular diameter distance in CGS (cm). Must be
            broadcastable against ``log_nu``; scalar is the common case.
        z : float, optional
            Source redshift. Default is ``0``.
        beta : float, optional
            Bulk velocity as a fraction of :math:`c`. Default is ``0``.
        cos_theta : float, optional
            Cosine of the angle between the bulk velocity and the line of sight. Default is ``1``.
        sin_alpha : float or ~numpy.ndarray, optional
            Sine of the pitch angle. Must be broadcastable against ``log_nu``. Default is ``1``.

        Returns
        -------
        log_flux : ~numpy.ndarray, shape ``(*nu_shape, *batch_shape)``
            :math:`\log F_\nu` in CGS (:math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}}`).
        """
        log_nu = np.asarray(log_nu, dtype="f8")
        log_A_eff = np.asarray(log_A_eff, dtype="f8")
        log_D_A = np.asarray(log_D_A, dtype="f8")

        log_gamma_bulk = np.log(1.0 / np.sqrt(1 - beta**2))
        log_Doppler = -log_gamma_bulk - np.log1p(beta * cos_theta)
        log_correction_factor = log_Doppler - np.log1p(z)

        log_nu_rf = log_nu - log_correction_factor

        log_specific_intensity = (
            self._compute_log_rf_specific_intensity(log_nu_rf, log_R, log_B, log_N, log_gamma, log_weights, sin_alpha)
            + 3 * log_correction_factor
        )

        # Expand nu-shaped scalars to broadcast against trailing batch dims.
        n_batch_dims = log_specific_intensity.ndim - log_nu.ndim
        log_A_eff_view = log_A_eff.reshape(log_A_eff.shape + (1,) * n_batch_dims)
        log_D_A_view = log_D_A.reshape(log_D_A.shape + (1,) * n_batch_dims)

        return log_specific_intensity + log_A_eff_view - 2 * log_D_A_view

    def _compute_log_pa_emissivity(
        self,
        log_nu: Union[float, np.ndarray],
        log_B: Union[float, np.ndarray],
        log_N: Union[float, np.ndarray],
        log_gamma: Union[float, np.ndarray],
        log_weights: Union[float, np.ndarray],
    ) -> np.ndarray:
        r"""
        Compute :math:`\log j_\nu` for a pitch-angle-averaged electron distribution.

        Identical to :meth:`_compute_log_emissivity` but uses the pitch-angle-averaged
        kernel :math:`\bar{F}(x)` so that no ``sin_alpha`` is required. This is the
        appropriate emissivity for an isotropic distribution of pitch angles.

        Parameters
        ----------
        log_nu : float or ~numpy.ndarray, shape ``(*nu_shape)``
            Natural log of the frequency grid in CGS (Hz).
        log_B : float or ~numpy.ndarray
            Natural log of the magnetic field strength in CGS (G). Must be broadcastable
            against ``log_nu``; scalar is the common case.
        log_N : float or ~numpy.ndarray, shape ``(*batch_shape, n_gamma)``
            Natural log of the electron number density on the Lorentz factor grid.
            The last axis is the gamma integration axis; any leading axes define a batch
            of independent distributions.
        log_gamma : float or ~numpy.ndarray, shape ``(n_gamma,)``
            Natural log of the Lorentz factor grid.
        log_weights : float or ~numpy.ndarray, shape ``(n_gamma,)``
            Quadrature weights :math:`w_i = \gamma_i\,\Delta\!\log\gamma_i` in log-space.

        Returns
        -------
        log_emissivity : ~numpy.ndarray, shape ``(*nu_shape, *batch_shape)``
            :math:`\log j_\nu` in CGS (:math:`\mathrm{erg\,s^{-1}\,cm^{-3}\,Hz^{-1}\,sr^{-1}}`).

        Notes
        -----
        **Broadcasting.** ``log_B`` must broadcast against ``log_nu`` and defines the
        *spectral* axis. ``log_N`` adds a separate *batch* axis: its leading dimensions
        ``(*batch_shape,)`` are independent of the spectral dimensions, so the output
        has shape ``(*nu_shape, *batch_shape)``. Passing ``log_N`` with shape
        ``(n_gamma,)`` (no batch dimensions) returns output with shape ``(*nu_shape,)``.
        """
        log_nu = np.asarray(log_nu, dtype="f8")
        log_B = np.asarray(log_B, dtype="f8")
        log_gamma = np.asarray(log_gamma, dtype="f8")
        log_N = np.asarray(log_N, dtype="f8")
        log_weights = np.asarray(log_weights, dtype="f8")

        nu_shape = log_nu.shape
        n_gamma = log_gamma.size
        batch_shape = log_N.shape[:-1]
        n_nu = log_nu.size
        n_batch = log_N.size // n_gamma

        log_B_flat = np.broadcast_to(log_B, nu_shape).ravel()
        log_nu_flat = log_nu.ravel()

        # log_x: (n_nu, n_gamma)
        log_x = (
            log_nu_flat[:, np.newaxis] - 2.0 * log_gamma[np.newaxis, :] - log_B_flat[:, np.newaxis] - _log_c_1_gamma_cgs
        )

        # log F_avg(x): (n_nu, n_gamma).
        log_F = np.interp(log_x.ravel(), self._log_x_avg_first_kernel, self._log_avg_first_kernel).reshape(
            n_nu, n_gamma
        )

        # Integrand: (n_nu, n_batch, n_gamma).
        log_N_2d = log_N.reshape(n_batch, n_gamma)
        log_integrand = log_F[:, np.newaxis, :] + log_N_2d[np.newaxis, :, :] + log_weights[np.newaxis, np.newaxis, :]

        # Integrate over gamma, add prefactors: (n_nu, n_batch).
        log_emissivity_2d = logsumexp(log_integrand, axis=-1) + _log_chi_cgs + log_B_flat[:, np.newaxis]

        return log_emissivity_2d.reshape(*nu_shape, *batch_shape)

    def _compute_log_pa_absorption_coefficient(
        self,
        log_nu: Union[float, np.ndarray],
        log_B: Union[float, np.ndarray],
        log_N: Union[float, np.ndarray],
        log_gamma: Union[float, np.ndarray],
        log_weights: Union[float, np.ndarray],
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
        log_nu : float or ~numpy.ndarray, shape ``(*nu_shape)``
            Natural log of the frequency grid in CGS (Hz).
        log_B : float or ~numpy.ndarray
            Natural log of the magnetic field strength in CGS (G). Must be broadcastable
            against ``log_nu``; scalar is the common case.
        log_N : float or ~numpy.ndarray, shape ``(*batch_shape, n_gamma)``
            Natural log of the electron number density on the Lorentz factor grid.
            The last axis is the gamma integration axis; any leading axes define a batch
            of independent distributions.
        log_gamma : float or ~numpy.ndarray, shape ``(n_gamma,)``
            Natural log of the Lorentz factor grid.
        log_weights : float or ~numpy.ndarray, shape ``(n_gamma,)``
            Quadrature weights :math:`w_i = \gamma_i\,\Delta\!\log\gamma_i` in log-space.

        Returns
        -------
        log_alpha : ~numpy.ndarray, shape ``(*nu_shape, *batch_shape)``
            :math:`\log|\alpha_\nu|` in CGS (:math:`\mathrm{cm^{-1}}`).

        Notes
        -----
        **Broadcasting.** ``log_B`` must broadcast against ``log_nu`` and defines the
        *spectral* axis. ``log_N`` adds a separate *batch* axis: its leading dimensions
        ``(*batch_shape,)`` are independent of the spectral dimensions, so the output
        has shape ``(*nu_shape, *batch_shape)``. Passing ``log_N`` with shape
        ``(n_gamma,)`` (no batch dimensions) returns output with shape ``(*nu_shape,)``.
        """
        log_nu = np.asarray(log_nu, dtype="f8")
        log_B = np.asarray(log_B, dtype="f8")
        log_gamma = np.asarray(log_gamma, dtype="f8")
        log_N = np.asarray(log_N, dtype="f8")
        log_weights = np.asarray(log_weights, dtype="f8")

        nu_shape = log_nu.shape
        n_gamma = log_gamma.size
        batch_shape = log_N.shape[:-1]
        n_nu = log_nu.size
        n_batch = log_N.size // n_gamma

        log_B_flat = np.broadcast_to(log_B, nu_shape).ravel()
        log_nu_flat = log_nu.ravel()

        # log_x: (n_nu, n_gamma)
        log_x = (
            log_nu_flat[:, np.newaxis] - 2.0 * log_gamma[np.newaxis, :] - log_B_flat[:, np.newaxis] - _log_c_1_gamma_cgs
        )

        # Evaluate log F_avg and d(log F_avg)/d(log x): (n_nu, n_gamma).
        log_F = np.interp(log_x.ravel(), self._log_x_avg_first_kernel, self._log_avg_first_kernel).reshape(
            n_nu, n_gamma
        )
        d_log_F_d_log_x = np.interp(
            log_x.ravel(), self._log_x_avg_first_kernel, self._log_davg_first_kernel_dlog_x
        ).reshape(n_nu, n_gamma)

        # IBP kernel: (n_nu, n_gamma).
        log_kernel = log_F + np.log(np.maximum(1.0 - d_log_F_d_log_x, np.finfo("f8").tiny))

        # Integrand: (n_nu, n_batch, n_gamma).
        log_N_2d = log_N.reshape(n_batch, n_gamma)
        log_integrand = (
            log_kernel[:, np.newaxis, :]
            + log_N_2d[np.newaxis, :, :]
            + log_weights[np.newaxis, np.newaxis, :]
            - log_gamma[np.newaxis, np.newaxis, :]
        )

        # Integrate over gamma, add prefactors: (n_nu, n_batch).
        log_alpha_2d = (
            _log_chi_abs_cgs
            + log_B_flat[:, np.newaxis]
            - 2.0 * log_nu_flat[:, np.newaxis]
            + logsumexp(log_integrand, axis=-1)
        )

        return log_alpha_2d.reshape(*nu_shape, *batch_shape)

    def _compute_log_pa_rf_specific_intensity(
        self,
        log_nu: Union[float, np.ndarray],
        log_R: Union[float, np.ndarray],
        log_B: Union[float, np.ndarray],
        log_N: Union[float, np.ndarray],
        log_gamma: Union[float, np.ndarray],
        log_weights: Union[float, np.ndarray],
    ) -> np.ndarray:
        r"""
        Compute :math:`\log I_\nu` in the rest frame for a pitch-angle-averaged distribution.

        Identical to :meth:`_compute_log_rf_specific_intensity` but calls the pitch-angle-averaged
        emissivity and absorption coefficient, so no ``sin_alpha`` is required.

        Parameters
        ----------
        log_nu : float or ~numpy.ndarray, shape ``(*nu_shape)``
            Natural log of the observed frequency grid in CGS (Hz).
        log_R : float or ~numpy.ndarray
            Natural log of the source radius in CGS (cm). Must be broadcastable against
            ``log_nu``; scalar is the common case.
        log_B : float or ~numpy.ndarray
            Natural log of the magnetic field strength in CGS (G). Must be broadcastable
            against ``log_nu``; scalar is the common case.
        log_N : float or ~numpy.ndarray, shape ``(*batch_shape, n_gamma)``
            Natural log of the electron number density on the Lorentz factor grid.
        log_gamma : float or ~numpy.ndarray, shape ``(n_gamma,)``
            Natural log of the Lorentz factor grid.
        log_weights : float or ~numpy.ndarray, shape ``(n_gamma,)``
            Quadrature weights :math:`w_i = \gamma_i\,\Delta\!\log\gamma_i` in log-space.

        Returns
        -------
        log_intensity : ~numpy.ndarray, shape ``(*nu_shape, *batch_shape)``
            :math:`\log I_\nu` in CGS (:math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}\,sr^{-1}}`).
        """
        log_nu = np.asarray(log_nu, dtype="f8")
        log_R = np.asarray(log_R, dtype="f8")

        log_emissivity = self._compute_log_pa_emissivity(log_nu, log_B, log_N, log_gamma, log_weights)
        log_absorption = self._compute_log_pa_absorption_coefficient(log_nu, log_B, log_N, log_gamma, log_weights)

        n_batch_dims = log_emissivity.ndim - log_nu.ndim
        log_R_view = log_R.reshape(log_R.shape + (1,) * n_batch_dims)

        tau = np.exp(np.clip(log_absorption + log_R_view, -np.inf, 500))
        return log_emissivity - log_absorption + np.log(-np.expm1(-tau))

    def _compute_log_pa_specific_intensity(
        self,
        log_nu: Union[float, np.ndarray],
        log_R: Union[float, np.ndarray],
        log_B: Union[float, np.ndarray],
        log_N: Union[float, np.ndarray],
        log_gamma: Union[float, np.ndarray],
        log_weights: Union[float, np.ndarray],
        z: float,
        beta: float,
        cos_theta: float,
    ) -> np.ndarray:
        r"""
        Compute :math:`\log I_\nu` as observed for a pitch-angle-averaged distribution.

        Identical to :meth:`_compute_log_specific_intensity` but uses the pitch-angle-averaged
        kernel, so no ``sin_alpha`` parameter is accepted.

        Parameters
        ----------
        log_nu : float or ~numpy.ndarray, shape ``(*nu_shape)``
            Natural log of the observed frequency grid in CGS (Hz).
        log_R : float or ~numpy.ndarray
            Natural log of the source radius in CGS (cm). Must be broadcastable against
            ``log_nu``; scalar is the common case.
        log_B : float or ~numpy.ndarray
            Natural log of the magnetic field strength in CGS (G). Must be broadcastable
            against ``log_nu``; scalar is the common case.
        log_N : float or ~numpy.ndarray, shape ``(*batch_shape, n_gamma)``
            Natural log of the electron number density on the Lorentz factor grid.
        log_gamma : float or ~numpy.ndarray, shape ``(n_gamma,)``
            Natural log of the Lorentz factor grid.
        log_weights : float or ~numpy.ndarray, shape ``(n_gamma,)``
            Quadrature weights :math:`w_i = \gamma_i\,\Delta\!\log\gamma_i` in log-space.
        z : float
            Source redshift.
        beta : float
            Bulk velocity as a fraction of :math:`c`.
        cos_theta : float
            Cosine of the angle between the bulk velocity and the line of sight.

        Returns
        -------
        log_intensity : ~numpy.ndarray, shape ``(*nu_shape, *batch_shape)``
            :math:`\log I_\nu` in CGS (:math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}\,sr^{-1}}`).
        """
        # Compute the doppler-factor corrections.
        log_gamma_bulk = np.log(1.0 / np.sqrt(1 - beta**2))
        log_Doppler = -log_gamma_bulk - np.log1p(beta * cos_theta)
        log_correction_factor = log_Doppler - np.log1p(z)

        # Now compute the corrected log_nu for the rest frame. In the rest frame, we have
        # nu_rf = (1+z)/D * nu_obs, so log_nu_rf = log_nu - log_correction_factor.
        log_nu_rf = log_nu - log_correction_factor

        # Compute the rest-frame specific intensity first.
        log_rf_intensity = self._compute_log_pa_rf_specific_intensity(
            log_nu_rf,
            log_R,
            log_B,
            log_N,
            log_gamma,
            log_weights,
        )

        # Now apply the Doppler and redshift corrections to get the observed specific intensity.
        log_specific_intensity = log_rf_intensity + 3 * log_correction_factor

        return log_specific_intensity

    def _compute_log_pa_flux_density(
        self,
        log_nu: Union[float, np.ndarray],
        log_R: Union[float, np.ndarray],
        log_B: Union[float, np.ndarray],
        log_N: Union[float, np.ndarray],
        log_gamma: Union[float, np.ndarray],
        log_weights: Union[float, np.ndarray],
        log_A_eff: Union[float, np.ndarray],
        log_D_A: Union[float, np.ndarray],
        z: float = 0,
        beta: float = 0,
        cos_theta: float = 1.0,
    ) -> np.ndarray:
        r"""
        Compute :math:`\log F_\nu` for a pitch-angle-averaged distribution.

        Identical to :meth:`_compute_log_flux_density` but uses the pitch-angle-averaged
        kernel throughout. No ``sin_alpha`` parameter is accepted.

        Parameters
        ----------
        log_nu : float or ~numpy.ndarray, shape ``(*nu_shape)``
            Natural log of the observed frequency grid in CGS (Hz).
        log_R : float or ~numpy.ndarray
            Natural log of the source radius in CGS (cm). Must be broadcastable against
            ``log_nu``; scalar is the common case.
        log_B : float or ~numpy.ndarray
            Natural log of the magnetic field strength in CGS (G). Must be broadcastable
            against ``log_nu``; scalar is the common case.
        log_N : float or ~numpy.ndarray, shape ``(*batch_shape, n_gamma)``
            Natural log of the electron number density on the Lorentz factor grid.
        log_gamma : float or ~numpy.ndarray, shape ``(n_gamma,)``
            Natural log of the Lorentz factor grid.
        log_weights : float or ~numpy.ndarray, shape ``(n_gamma,)``
            Quadrature weights :math:`w_i = \gamma_i\,\Delta\!\log\gamma_i` in log-space.
        log_A_eff : float or ~numpy.ndarray
            Natural log of the effective emitting area in CGS (cm\ :sup:`2`). Must be
            broadcastable against ``log_nu``; scalar is the common case.
        log_D_A : float or ~numpy.ndarray
            Natural log of the angular diameter distance in CGS (cm). Must be
            broadcastable against ``log_nu``; scalar is the common case.
        z : float, optional
            Source redshift. Default is ``0``.
        beta : float, optional
            Bulk velocity as a fraction of :math:`c`. Default is ``0``.
        cos_theta : float, optional
            Cosine of the angle between the bulk velocity and the line of sight. Default is ``1``.

        Returns
        -------
        log_flux : ~numpy.ndarray, shape ``(*nu_shape, *batch_shape)``
            :math:`\log F_\nu` in CGS (:math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}}`).
        """
        log_nu = np.asarray(log_nu, dtype="f8")
        log_A_eff = np.asarray(log_A_eff, dtype="f8")
        log_D_A = np.asarray(log_D_A, dtype="f8")

        log_gamma_bulk = np.log(1.0 / np.sqrt(1 - beta**2))
        log_Doppler = -log_gamma_bulk - np.log1p(beta * cos_theta)
        log_correction_factor = log_Doppler - np.log1p(z)

        log_nu_rf = log_nu - log_correction_factor

        log_specific_intensity = (
            self._compute_log_pa_rf_specific_intensity(log_nu_rf, log_R, log_B, log_N, log_gamma, log_weights)
            + 3 * log_correction_factor
        )

        n_batch_dims = log_specific_intensity.ndim - log_nu.ndim
        log_A_eff_view = log_A_eff.reshape(log_A_eff.shape + (1,) * n_batch_dims)
        log_D_A_view = log_D_A.reshape(log_D_A.shape + (1,) * n_batch_dims)

        return log_specific_intensity + log_A_eff_view - 2 * log_D_A_view

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
        log_gamma : ~numpy.ndarray, shape ``(n_gamma,)``
            Natural log of the Lorentz factor grid.
        log_weights : ~numpy.ndarray, shape ``(n_gamma,)``
            Log of the quadrature weights :math:`w_i = \gamma_i\,\Delta\!\log\gamma_i`.
        """
        if gamma is not None:
            log_gamma = np.log(np.asarray(gamma, dtype="f8"))
        else:
            log_gamma = np.linspace(np.log(gamma_min), np.log(gamma_max), n_gamma)
        log_weights = log_gamma + np.log(np.gradient(log_gamma))
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
        log_gamma : ~numpy.ndarray, shape ``(n_gamma,)``
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
        Compute the synchrotron emissivity :math:`j_\nu`.

        Parameters
        ----------
        nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Frequency grid. Bare values are treated as Hz.
        B : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Magnetic field strength. Bare values are treated as Gauss.
        N : ~numpy.ndarray or callable
            Electron distribution :math:`dN/d\gamma` in :math:`\mathrm{cm^{-3}}`.
            If callable, evaluated as ``N(gamma)`` on the grid.
        gamma : ~numpy.ndarray or None, optional
            Explicit Lorentz factor grid. If ``None``, built from ``gamma_min``,
            ``gamma_max``, ``n_gamma``.
        alpha : float, ~astropy.units.Quantity, or None, optional
            Pitch angle. Bare values are treated as radians. If ``None``, the
            pitch-angle-averaged kernel is used (requires :meth:`load_avg_first_kernel`).
        gamma_min : float, optional
            Grid lower bound (ignored if ``gamma`` is provided). Default ``1.0``.
        gamma_max : float, optional
            Grid upper bound (ignored if ``gamma`` is provided). Default ``1e8``.
        n_gamma : int, optional
            Number of grid points (ignored if ``gamma`` is provided). Default ``200``.

        Returns
        -------
        j_nu : ~astropy.units.Quantity
            Emissivity in :math:`\mathrm{erg\,s^{-1}\,cm^{-3}\,Hz^{-1}\,sr^{-1}}`.
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
        Compute the synchrotron self-absorption coefficient :math:`|\alpha_\nu|`.

        Parameters
        ----------
        nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Frequency grid. Bare values are treated as Hz.
        B : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Magnetic field strength. Bare values are treated as Gauss.
        N : ~numpy.ndarray or callable
            Electron distribution :math:`dN/d\gamma` in :math:`\mathrm{cm^{-3}}`.
            If callable, evaluated as ``N(gamma)`` on the grid.
        gamma : ~numpy.ndarray or None, optional
            Explicit Lorentz factor grid. If ``None``, built from ``gamma_min``,
            ``gamma_max``, ``n_gamma``.
        alpha : float, ~astropy.units.Quantity, or None, optional
            Pitch angle in radians. If ``None``, pitch-angle-averaged kernel is used.
        gamma_min : float, optional
            Grid lower bound (ignored if ``gamma`` is provided). Default ``1.0``.
        gamma_max : float, optional
            Grid upper bound (ignored if ``gamma`` is provided). Default ``1e8``.
        n_gamma : int, optional
            Number of grid points (ignored if ``gamma`` is provided). Default ``200``.

        Returns
        -------
        alpha_nu : ~astropy.units.Quantity
            Absorption coefficient in :math:`\mathrm{cm^{-1}}`.
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
        Compute the synchrotron source function :math:`S_\nu = j_\nu / \alpha_\nu`.

        Parameters
        ----------
        nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Frequency grid. Bare values are treated as Hz.
        B : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Magnetic field strength. Bare values are treated as Gauss.
        N : ~numpy.ndarray or callable
            Electron distribution :math:`dN/d\gamma` in :math:`\mathrm{cm^{-3}}`.
            If callable, evaluated as ``N(gamma)`` on the grid.
        gamma : ~numpy.ndarray or None, optional
            Explicit Lorentz factor grid. If ``None``, built from ``gamma_min``,
            ``gamma_max``, ``n_gamma``.
        alpha : float, ~astropy.units.Quantity, or None, optional
            Pitch angle in radians. If ``None``, pitch-angle-averaged kernel is used.
        gamma_min : float, optional
            Grid lower bound (ignored if ``gamma`` is provided). Default ``1.0``.
        gamma_max : float, optional
            Grid upper bound (ignored if ``gamma`` is provided). Default ``1e8``.
        n_gamma : int, optional
            Number of grid points (ignored if ``gamma`` is provided). Default ``200``.

        Returns
        -------
        S_nu : ~astropy.units.Quantity
            Source function in :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}\,sr^{-1}}`.
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
            log_a = self._compute_log_pa_absorption_coefficient(
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
            log_a = self._compute_log_absorption_coefficient(
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
        R: Union[float, np.ndarray, u.Quantity],
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
        Compute the rest-frame specific intensity :math:`I_\nu = S_\nu(1 - e^{-\tau_\nu})`.

        Parameters
        ----------
        nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Frequency grid. Bare values are treated as Hz.
        R : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Line-of-sight depth of the source. Bare values are treated as cm.
        B : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Magnetic field strength. Bare values are treated as Gauss.
        N : ~numpy.ndarray or callable
            Electron distribution :math:`dN/d\gamma` in :math:`\mathrm{cm^{-3}}`.
            If callable, evaluated as ``N(gamma)`` on the grid.
        gamma : ~numpy.ndarray or None, optional
            Explicit Lorentz factor grid. If ``None``, built from ``gamma_min``,
            ``gamma_max``, ``n_gamma``.
        alpha : float, ~astropy.units.Quantity, or None, optional
            Pitch angle in radians. If ``None``, pitch-angle-averaged kernel is used.
        gamma_min : float, optional
            Grid lower bound (ignored if ``gamma`` is provided). Default ``1.0``.
        gamma_max : float, optional
            Grid upper bound (ignored if ``gamma`` is provided). Default ``1e8``.
        n_gamma : int, optional
            Number of grid points (ignored if ``gamma`` is provided). Default ``200``.

        Returns
        -------
        I_nu : ~astropy.units.Quantity
            Specific intensity in :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}\,sr^{-1}}`.
        """
        nu_cgs = ensure_in_units(nu, u.Hz)
        R_cgs = ensure_in_units(R, u.cm)
        B_cgs = ensure_in_units(B, u.G)
        log_gamma, log_weights = self._build_gamma_grid(gamma, gamma_min, gamma_max, n_gamma)
        log_N = self._resolve_log_N(N, log_gamma)

        if alpha is None:
            self.ensure_avg_first_kernel_loaded()
            log_I = self._compute_log_pa_rf_specific_intensity(
                np.log(nu_cgs),
                np.log(R_cgs),
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
                np.log(R_cgs),
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
        R: Union[float, np.ndarray, u.Quantity],
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
        Compute the observed specific intensity including bulk Doppler boost and redshift.

        Parameters
        ----------
        nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Observed frequency grid. Bare values are treated as Hz.
        R : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Line-of-sight depth of the source. Bare values are treated as cm.
        B : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Magnetic field strength. Bare values are treated as Gauss.
        N : ~numpy.ndarray or callable
            Electron distribution :math:`dN/d\gamma` in :math:`\mathrm{cm^{-3}}`.
            If callable, evaluated as ``N(gamma)`` on the grid.
        gamma : ~numpy.ndarray or None, optional
            Explicit Lorentz factor grid. If ``None``, built from ``gamma_min``,
            ``gamma_max``, ``n_gamma``.
        z : float, optional
            Source redshift. Default ``0``.
        beta : float, optional
            Bulk velocity as a fraction of :math:`c`. Default ``0``.
        theta : float or ~astropy.units.Quantity, optional
            Angle between the bulk velocity and the line of sight. Bare values
            are treated as radians. Default ``0``.
        alpha : float, ~astropy.units.Quantity, or None, optional
            Pitch angle in radians. If ``None``, pitch-angle-averaged kernel is used.
        gamma_min : float, optional
            Grid lower bound (ignored if ``gamma`` is provided). Default ``1.0``.
        gamma_max : float, optional
            Grid upper bound (ignored if ``gamma`` is provided). Default ``1e8``.
        n_gamma : int, optional
            Number of grid points (ignored if ``gamma`` is provided). Default ``200``.

        Returns
        -------
        I_nu : ~astropy.units.Quantity
            Observed specific intensity in
            :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}\,sr^{-1}}`.
        """
        nu_cgs = ensure_in_units(nu, u.Hz)
        R_cgs = ensure_in_units(R, u.cm)
        B_cgs = ensure_in_units(B, u.G)
        cos_theta = float(np.cos(ensure_in_units(theta, u.rad)))
        log_gamma, log_weights = self._build_gamma_grid(gamma, gamma_min, gamma_max, n_gamma)
        log_N = self._resolve_log_N(N, log_gamma)

        if alpha is None:
            self.ensure_avg_first_kernel_loaded()
            log_I = self._compute_log_pa_specific_intensity(
                np.log(nu_cgs),
                np.log(R_cgs),
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
                np.log(R_cgs),
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
        R: Union[float, np.ndarray, u.Quantity],
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
        Compute the observed spectral flux density :math:`F_\nu`.

        Parameters
        ----------
        nu : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Observed frequency grid. Bare values are treated as Hz.
        R : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Line-of-sight depth of the source (sets optical depth). Bare values
            are treated as cm.
        B : float, ~numpy.ndarray, or ~astropy.units.Quantity
            Magnetic field strength. Bare values are treated as Gauss.
        N : ~numpy.ndarray or callable
            Electron distribution :math:`dN/d\gamma` in :math:`\mathrm{cm^{-3}}`.
            If callable, evaluated as ``N(gamma)`` on the grid.
        gamma : ~numpy.ndarray or None, optional
            Explicit Lorentz factor grid. If ``None``, built from ``gamma_min``,
            ``gamma_max``, ``n_gamma``.
        A_eff : float, ~numpy.ndarray, ~astropy.units.Quantity, or None, optional
            Effective projected area of the source on the sky. Bare values are
            treated as :math:`\mathrm{cm^2}`. Defaults to :math:`\pi R^2` when
            ``None``.
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
            triceratops configuration cosmology.
        beta : float, optional
            Bulk velocity as a fraction of :math:`c`. Default ``0``.
        theta : float or ~astropy.units.Quantity, optional
            Angle between the bulk velocity and the line of sight. Bare values
            are treated as radians. Default ``0``.
        alpha : float, ~astropy.units.Quantity, or None, optional
            Pitch angle in radians. If ``None``, pitch-angle-averaged kernel is used.
        gamma_min : float, optional
            Grid lower bound (ignored if ``gamma`` is provided). Default ``1.0``.
        gamma_max : float, optional
            Grid upper bound (ignored if ``gamma`` is provided). Default ``1e8``.
        n_gamma : int, optional
            Number of grid points (ignored if ``gamma`` is provided). Default ``200``.

        Returns
        -------
        F_nu : ~astropy.units.Quantity
            Spectral flux density in :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}}`.
        """
        nu_cgs = ensure_in_units(nu, u.Hz)
        R_cgs = ensure_in_units(R, u.cm)
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

        # Effective sky area defaults to pi R^2.
        if A_eff is None:
            A_eff_cgs = np.pi * R_cgs**2
        else:
            A_eff_cgs = ensure_in_units(A_eff, u.cm**2)

        if alpha is None:
            self.ensure_avg_first_kernel_loaded()
            log_F = self._compute_log_pa_flux_density(
                np.log(nu_cgs),
                np.log(R_cgs),
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
                np.log(R_cgs),
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
