r"""
Gaunt factor support for free–free emission and absorption.

The **Gaunt factor** :math:`g_{\rm ff}(Z, T, \nu)` is a dimensionless quantum-mechanical
correction of order unity to the classical free–free (bremsstrahlung) emission and absorption
coefficients.  It accounts for the quantum nature of electron–ion Coulomb interactions and
depends on three physical parameters:

* The **ion charge** :math:`Z` (integer, :math:`Z \geq 1`).
* The **electron temperature** :math:`T` (Kelvin), through the dimensionless parameter

  .. math::

      \log_{10}\gamma^2 = \log_{10}\!\left(\frac{Z^2 R_y}{k_B T}\right)

  where :math:`R_y = 13.6\,{\rm eV}` is the Rydberg energy.

* The **photon frequency** :math:`\nu` (Hz), through the dimensionless parameter

  .. math::

      \log_{10} u = \log_{10}\!\left(\frac{h\nu}{k_B T}\right).

This module provides a number of convenient methods ranging from analytical approximations to
tabulated interpolators for accurate Gaunt factor evaluation.

.. rubric:: Available Gaunt Factor Methods


.. list-table::
   :header-rows: 1
   :widths: 15 20 20 45

   * - Key
     - Type
     - Regime
     - Notes

   * - ``"lu"``
     - Analytic
     - Non-relativistic (general)
     - Default. Smooth, fast, and accurate across a wide parameter range.
       Based on :footcite:t:`lu_2026_18603474`.

   * - ``"draine"``
     - Analytic
     - Classical non-relativistic
     - Simpler classical approximation from
       :footcite:t:`draine2011physics`. Slightly less accurate than ``"lu"``.

   * - ``"vanhoof"``
     - Tabulated (2D)
     - Non-relativistic
     - Bilinear interpolation on the van Hoof et al. (2014) table.
       More accurate than analytic forms; slower due to interpolation.

   * - ``"vanhoof_rel"``
     - Tabulated (3D)
     - Relativistic (high :math:`T`)
     - Trilinear interpolation including explicit :math:`Z` dependence.
       Use for :math:`T \gtrsim 10^8\,\mathrm{K}` where relativistic
       corrections are important.


"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import h5py
import numpy as np
from astropy import units as u
from scipy.interpolate import RegularGridInterpolator

from trilobite.radiation.constants import Ry_cgs, h_cgs, kB_cgs
from trilobite.utils.log import trilobite_logger
from trilobite.utils.misc_utils import ensure_in_units

if TYPE_CHECKING:
    from trilobite._typing import _ArrayLike, _UnitBearingArrayLike

__all__ = [
    # Interpolator classes
    "GauntFactorInterpolatorBase",
    "NonRelativisticGauntFactorInterpolator",
    "RelativisticGauntFactorInterpolator",
    # Interpolator factories
    "get_default_gaunt_interpolator",
    "get_default_relativistic_gaunt_interpolator",
    # Public compute functions
    "compute_ff_gaunt_factor",
    "compute_ff_gaunt_factor_comp",
    "compute_mean_ff_gaunt_factor",
    "compute_mean_ff_gaunt_factor_comp",
]

# ============================================ #
# PATHS                                        #
# ============================================ #
_VAN_HOOF_NON_REL_PATH: Path = Path(__file__).parent / "gaunt_tables" / "gauntff_vanhoof_2014.hdf5"
_VAN_HOOF_REL_PATH: Path = Path(__file__).parent / "gaunt_tables" / "gauntff_rel_vanhoof_2014.hdf5"


# ================================================ #
# Abstract Base Class                              #
# ================================================ #
class GauntFactorInterpolatorBase(ABC):
    """Abstract base class for tabulated free–free Gaunt factor interpolators.

    Defines the shared interface for all Gaunt factor interpolators, including
    HDF5 loading and the callable/containment protocol.

    Concrete subclasses must implement :meth:`__call__`, :meth:`__contains__`,
    and the :attr:`description` property.
    """

    @classmethod
    def _load_hdf5(cls, path: Union[str, Path]) -> dict:
        """Load Gaunt factor datasets from an HDF5 file.

        Parameters
        ----------
        path : str or ~pathlib.Path
            Path to the HDF5 file.

        Returns
        -------
        dict
            Dictionary containing all datasets found in the file.  Guaranteed
            keys are ``'log_gamma2'``, ``'log_u'``, and ``'gff'``; ``'Z'`` is
            included when present.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist on disk.
        KeyError
            If the HDF5 file is missing one of the required datasets
            (``'log_gamma2'``, ``'log_u'``, ``'gff'``).
        """
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Gaunt factor table not found at {path}")

        trilobite_logger.info("Loading Gaunt factor table from %s.", path)
        datasets = {}
        with h5py.File(path, "r") as f:
            for key in ("log_gamma2", "log_u", "gff"):
                if key not in f:
                    raise KeyError(f"Gaunt factor table at {path} is missing required dataset '{key}'.")
                datasets[key] = f[key][:]
            if "Z" in f:
                datasets["Z"] = f["Z"][:]
        trilobite_logger.debug("Retrieved Gaunt factor table from %s.", path)
        return datasets

    @abstractmethod
    def __call__(
        self,
        Z: Union[int, "np.ndarray"],
        T: Union[float, "np.ndarray"],
        nu: Union[float, "np.ndarray"],
    ) -> Union[float, "np.ndarray"]:
        """Evaluate the free–free Gaunt factor at ``(Z, T, nu)``."""

    @abstractmethod
    def __contains__(self, item: tuple) -> bool:
        """Check whether ``(Z, T, nu)`` lies within the interpolation domain."""

    @property
    @abstractmethod
    def description(self) -> str:
        """str: Human-readable one-line description of the table."""


# ================================================ #
# Interpolated Gaunt Factors                       #
# ================================================ #
class NonRelativisticGauntFactorInterpolator(GauntFactorInterpolatorBase):
    r"""Bilinear interpolator for non-relativistic tabulated free–free Gaunt factors.

    Wraps a :class:`~scipy.interpolate.RegularGridInterpolator` loaded from
    the 2-D van Hoof et al. (2014) HDF5 table.  The table layout is::

        gff.shape == (n_log_u, n_log_gamma2)

    The ion charge :math:`Z` enters only through the coordinate transform

    .. math::

        \log_{10}\gamma^2 = \log_{10}\!\left(\frac{Z^2 R_y}{k_B T}\right)

    and is **not** an explicit table axis.

    Parameters
    ----------
    path : str or ~pathlib.Path, optional
        Path to the HDF5 file.  Defaults to the bundled
        ``gauntff_vanhoof_2014.hdf5`` table.
    **kwargs
        Additional keyword arguments forwarded to
        :class:`~scipy.interpolate.RegularGridInterpolator`.

    Examples
    --------
    .. code-block:: python

        from trilobite.radiation.free_free.gaunt_factor import (
            get_default_gaunt_interpolator,
        )

        interp = get_default_gaunt_interpolator()
        gff = interp(Z=1, T=1e4, nu=1e10)
        print(f"gff = {gff:.4f}")
    """

    def __init__(self, path: Union[str, Path] = _VAN_HOOF_NON_REL_PATH, **kwargs) -> None:
        self.path = Path(path).expanduser().resolve()
        datasets = self._load_hdf5(self.path)

        self._log_gamma2_grid: np.ndarray = datasets["log_gamma2"]
        self._log_u_grid: np.ndarray = datasets["log_u"]
        self._array: np.ndarray = datasets["gff"]

        if self._array.ndim != 2:
            raise ValueError(
                f"Non-relativistic Gaunt table at {self.path} has shape {self._array.shape}; expected 2-D."
            )
        if self._array.shape[0] != len(self._log_u_grid):
            raise ValueError(
                f"Gaunt table at {self.path}: axis-0 length {self._array.shape[0]} "
                f"does not match log_u grid length {len(self._log_u_grid)}."
            )
        if self._array.shape[1] != len(self._log_gamma2_grid):
            raise ValueError(
                f"Gaunt table at {self.path}: axis-1 length {self._array.shape[1]} "
                f"does not match log_gamma2 grid length {len(self._log_gamma2_grid)}."
            )

        self._interpolator = RegularGridInterpolator(
            (self._log_u_grid, self._log_gamma2_grid),
            self._array,
            **kwargs,
        )
        trilobite_logger.info("Loading Gaunt factor table from %s. [DONE]", path)

    # ------------------------------------------------------------------ #
    # Calling interface                                                    #
    # ------------------------------------------------------------------ #
    def __call__(
        self,
        Z: Union[int, "np.ndarray"],
        T: Union[float, "np.ndarray"],
        nu: Union[float, "np.ndarray"],
    ) -> Union[float, "np.ndarray"]:
        r"""Evaluate the non-relativistic free–free Gaunt factor at ``(Z, T, nu)``.

        Parameters
        ----------
        Z : int or array-like
            Ion charge number (:math:`Z \geq 1`).  Enters only through the
            :math:`\log_{10}\gamma^2` coordinate transform.
        T : float or array-like
            Electron temperature in Kelvin.
        nu : float or array-like
            Photon frequency in Hz.

        Returns
        -------
        float or ~numpy.ndarray
            The dimensionless Gaunt factor :math:`g_{\rm ff}(Z, T, \nu)`.
        """
        Z_arr = np.asarray(Z, dtype=float)
        T_arr = np.asarray(T, dtype=float)
        nu_arr = np.asarray(nu, dtype=float)

        scalar_input = Z_arr.ndim == 0 and T_arr.ndim == 0 and nu_arr.ndim == 0

        log_gamma2 = np.log10(Z_arr**2 * Ry_cgs / (kB_cgs * T_arr))
        log_u = np.log10(h_cgs * nu_arr / (kB_cgs * T_arr))

        log_u_b, log_g2_b = np.broadcast_arrays(log_u, log_gamma2)
        points = np.column_stack([log_u_b.ravel(), log_g2_b.ravel()])

        result = self._interpolator(points).reshape(log_u_b.shape)

        if scalar_input:
            return float(result)
        return result

    # ------------------------------------------------------------------ #
    # Dunder helpers                                                       #
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        return f"{type(self).__name__}(n_log_u={self.n_log_u}, n_log_gamma2={self.n_log_gamma2}, path='{self.path}')"

    def __str__(self) -> str:
        return (
            f"NonRelativisticGauntFactorInterpolator\n"
            f"  Source  : {self.path.name}\n"
            f"  log_u   : {self._log_u_grid[0]:.1f} – {self._log_u_grid[-1]:.1f}  "
            f"({self.n_log_u} points)\n"
            f"  log_γ²  : {self._log_gamma2_grid[0]:.1f} – {self._log_gamma2_grid[-1]:.1f}  "
            f"({self.n_log_gamma2} points)"
        )

    def __len__(self) -> int:
        """Return the total number of tabulated Gaunt factor values."""
        return int(np.prod(self._array.shape))

    def __contains__(self, item: tuple) -> bool:
        r"""Check whether ``(Z, T, nu)`` lies within the interpolation domain.

        Parameters
        ----------
        item : tuple of (Z, T, nu)
            A 3-tuple of scalar physical parameters.

        Returns
        -------
        bool
            ``True`` if the dimensionless coordinates fall inside the table
            domain, ``False`` otherwise.  Z is not range-checked (it enters
            only through :math:`\log_{10}\gamma^2`).

        Examples
        --------
        .. code-block:: python

            (1, 1e4, 1e10) in interp  # True or False
        """
        if len(item) != 3:
            raise ValueError("__contains__ expects a 3-tuple (Z, T, nu).")
        Z, T, nu = item
        log_g2 = np.log10(float(Z) ** 2 * Ry_cgs / (kB_cgs * float(T)))
        log_u = np.log10(h_cgs * float(nu) / (kB_cgs * float(T)))
        g2_ok = self._log_gamma2_grid[0] <= log_g2 <= self._log_gamma2_grid[-1]
        u_ok = self._log_u_grid[0] <= log_u <= self._log_u_grid[-1]
        return bool(g2_ok and u_ok)

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #
    @property
    def shape(self) -> tuple:
        """tuple: Shape of the underlying Gaunt factor array ``(n_log_u, n_log_gamma2)``."""
        return self._array.shape

    @property
    def log_gamma2_range(self) -> tuple:
        r"""tuple: Minimum and maximum :math:`\log_{10}\gamma^2` in the table."""
        return float(self._log_gamma2_grid[0]), float(self._log_gamma2_grid[-1])

    @property
    def log_u_range(self) -> tuple:
        r"""tuple: Minimum and maximum :math:`\log_{10} u` in the table."""
        return float(self._log_u_grid[0]), float(self._log_u_grid[-1])

    @property
    def n_log_gamma2(self) -> int:
        r"""int: Number of :math:`\log_{10}\gamma^2` grid points."""
        return len(self._log_gamma2_grid)

    @property
    def n_log_u(self) -> int:
        r"""int: Number of :math:`\log_{10} u` grid points."""
        return len(self._log_u_grid)

    @property
    def source(self) -> str:
        """str: Absolute path to the HDF5 file backing this interpolator."""
        return str(self.path)

    @property
    def description(self) -> str:
        """str: Human-readable one-line description of the table."""
        return (
            f"Non-relativistic free-free Gaunt factors "
            f"(log_u={self.log_u_range[0]:.1f}–{self.log_u_range[1]:.1f}, "
            f"log_γ²={self.log_gamma2_range[0]:.1f}–{self.log_gamma2_range[1]:.1f})"
        )

    @property
    def log_gamma2_grid(self) -> np.ndarray:
        r"""~numpy.ndarray: Read-only view of the :math:`\log_{10}\gamma^2` grid."""
        view = self._log_gamma2_grid.view()
        view.flags.writeable = False
        return view

    @property
    def log_u_grid(self) -> np.ndarray:
        r"""~numpy.ndarray: Read-only view of the :math:`\log_{10} u` grid."""
        view = self._log_u_grid.view()
        view.flags.writeable = False
        return view


class RelativisticGauntFactorInterpolator(GauntFactorInterpolatorBase):
    """Trilinear interpolator for relativistic tabulated free–free Gaunt factors.

    Wraps a :class:`~scipy.interpolate.RegularGridInterpolator` loaded from
    the 3-D van Hoof et al. (2014) relativistic HDF5 table.  The table layout
    is::

        gff.shape == (n_Z, n_log_u, n_log_gamma2)

    Unlike the non-relativistic case, the ion charge :math:`Z` appears as an
    explicit table axis, capturing residual Z-dependence beyond the coordinate
    transform.

    Parameters
    ----------
    path : str or ~pathlib.Path, optional
        Path to the HDF5 file.  Defaults to the bundled
        ``gauntff_rel_vanhoof_2014.hdf5`` table.
    **kwargs
        Additional keyword arguments forwarded to
        :class:`~scipy.interpolate.RegularGridInterpolator`.

    Examples
    --------
    .. code-block:: python

        from trilobite.radiation.free_free.gaunt_factor import (
            get_default_relativistic_gaunt_interpolator,
        )

        interp = (
            get_default_relativistic_gaunt_interpolator()
        )
        gff = interp(Z=1, T=1e8, nu=1e18)
        print(f"gff = {gff:.4f}")
    """

    def __init__(self, path: Union[str, Path] = _VAN_HOOF_REL_PATH, **kwargs) -> None:
        self.path = Path(path).expanduser().resolve()
        datasets = self._load_hdf5(self.path)

        if "Z" not in datasets:
            raise KeyError(f"Relativistic Gaunt factor table at {self.path} is missing required dataset 'Z'.")

        self._z_grid: np.ndarray = datasets["Z"]
        self._log_gamma2_grid: np.ndarray = datasets["log_gamma2"]
        self._log_u_grid: np.ndarray = datasets["log_u"]
        self._array: np.ndarray = datasets["gff"]

        if self._array.ndim != 3:
            raise ValueError(f"Relativistic Gaunt table at {self.path} has shape {self._array.shape}; expected 3-D.")
        if self._array.shape[0] != len(self._z_grid):
            raise ValueError(
                f"Gaunt table at {self.path}: axis-0 length {self._array.shape[0]} "
                f"does not match Z grid length {len(self._z_grid)}."
            )
        if self._array.shape[1] != len(self._log_u_grid):
            raise ValueError(
                f"Gaunt table at {self.path}: axis-1 length {self._array.shape[1]} "
                f"does not match log_u grid length {len(self._log_u_grid)}."
            )
        if self._array.shape[2] != len(self._log_gamma2_grid):
            raise ValueError(
                f"Gaunt table at {self.path}: axis-2 length {self._array.shape[2]} "
                f"does not match log_gamma2 grid length {len(self._log_gamma2_grid)}."
            )

        self._interpolator = RegularGridInterpolator(
            (self._z_grid, self._log_u_grid, self._log_gamma2_grid),
            self._array,
            **kwargs,
        )
        trilobite_logger.info("Loading Gaunt factor table from %s. [DONE]", path)

    # ------------------------------------------------------------------ #
    # Calling interface                                                    #
    # ------------------------------------------------------------------ #
    def __call__(
        self,
        Z: Union[int, "np.ndarray"],
        T: Union[float, "np.ndarray"],
        nu: Union[float, "np.ndarray"],
    ) -> Union[float, "np.ndarray"]:
        r"""Evaluate the relativistic free–free Gaunt factor at ``(Z, T, nu)``.

        Parameters
        ----------
        Z : int or array-like
            Ion charge number (:math:`Z \geq 1`).  Used both in the
            coordinate transform and as an explicit interpolation axis.
        T : float or array-like
            Electron temperature in Kelvin.
        nu : float or array-like
            Photon frequency in Hz.

        Returns
        -------
        float or ~numpy.ndarray
            The dimensionless Gaunt factor :math:`g_{\rm ff}(Z, T, \nu)`.
        """
        Z_arr = np.asarray(Z, dtype=float)
        T_arr = np.asarray(T, dtype=float)
        nu_arr = np.asarray(nu, dtype=float)

        scalar_input = Z_arr.ndim == 0 and T_arr.ndim == 0 and nu_arr.ndim == 0

        log_gamma2 = np.log10(Z_arr**2 * Ry_cgs / (kB_cgs * T_arr))
        log_u = np.log10(h_cgs * nu_arr / (kB_cgs * T_arr))

        Z_b, log_u_b, log_g2_b = np.broadcast_arrays(Z_arr, log_u, log_gamma2)
        points = np.column_stack([Z_b.ravel(), log_u_b.ravel(), log_g2_b.ravel()])

        result = self._interpolator(points).reshape(Z_b.shape)

        if scalar_input:
            return float(result)
        return result

    # ------------------------------------------------------------------ #
    # Dunder helpers                                                       #
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"n_Z={self.n_Z}, "
            f"n_log_u={self.n_log_u}, "
            f"n_log_gamma2={self.n_log_gamma2}, "
            f"path='{self.path}')"
        )

    def __str__(self) -> str:
        return (
            f"RelativisticGauntFactorInterpolator\n"
            f"  Source  : {self.path.name}\n"
            f"  Z range : {int(self._z_grid[0])} – {int(self._z_grid[-1])}\n"
            f"  log_u   : {self._log_u_grid[0]:.1f} – {self._log_u_grid[-1]:.1f}  "
            f"({self.n_log_u} points)\n"
            f"  log_γ²  : {self._log_gamma2_grid[0]:.1f} – {self._log_gamma2_grid[-1]:.1f}  "
            f"({self.n_log_gamma2} points)"
        )

    def __len__(self) -> int:
        """Return the total number of tabulated Gaunt factor values."""
        return int(np.prod(self._array.shape))

    def __contains__(self, item: tuple) -> bool:
        """Check whether ``(Z, T, nu)`` lies within the interpolation domain.

        Parameters
        ----------
        item : tuple of (Z, T, nu)
            A 3-tuple of scalar physical parameters.

        Returns
        -------
        bool
            ``True`` if the point is inside the table domain, ``False`` otherwise.

        Examples
        --------
        .. code-block:: python

            (1, 1e4, 1e10) in interp  # True or False
        """
        if len(item) != 3:
            raise ValueError("__contains__ expects a 3-tuple (Z, T, nu).")
        Z, T, nu = item
        log_g2 = np.log10(float(Z) ** 2 * Ry_cgs / (kB_cgs * float(T)))
        log_u = np.log10(h_cgs * float(nu) / (kB_cgs * float(T)))
        z_ok = self._z_grid[0] <= float(Z) <= self._z_grid[-1]
        g2_ok = self._log_gamma2_grid[0] <= log_g2 <= self._log_gamma2_grid[-1]
        u_ok = self._log_u_grid[0] <= log_u <= self._log_u_grid[-1]
        return bool(z_ok and g2_ok and u_ok)

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #
    @property
    def shape(self) -> tuple:
        """tuple: Shape of the underlying Gaunt factor array ``(n_Z, n_log_u, n_log_gamma2)``."""
        return self._array.shape

    @property
    def z_range(self) -> tuple:
        """tuple: Minimum and maximum ion charge ``(Z_min, Z_max)``."""
        return int(self._z_grid[0]), int(self._z_grid[-1])

    @property
    def log_gamma2_range(self) -> tuple:
        r"""tuple: Minimum and maximum :math:`\log_{10}\gamma^2` in the table."""
        return float(self._log_gamma2_grid[0]), float(self._log_gamma2_grid[-1])

    @property
    def log_u_range(self) -> tuple:
        r"""tuple: Minimum and maximum :math:`\log_{10} u` in the table."""
        return float(self._log_u_grid[0]), float(self._log_u_grid[-1])

    @property
    def n_Z(self) -> int:
        """int: Number of ion charge values in the table."""
        return len(self._z_grid)

    @property
    def n_log_gamma2(self) -> int:
        r"""int: Number of :math:`\log_{10}\gamma^2` grid points."""
        return len(self._log_gamma2_grid)

    @property
    def n_log_u(self) -> int:
        r"""int: Number of :math:`\log_{10} u` grid points."""
        return len(self._log_u_grid)

    @property
    def source(self) -> str:
        """str: Absolute path to the HDF5 file backing this interpolator."""
        return str(self.path)

    @property
    def description(self) -> str:
        """str: Human-readable one-line description of the table."""
        return (
            f"Relativistic free-free Gaunt factors "
            f"(Z={self.z_range[0]}–{self.z_range[1]}, "
            f"log_u={self.log_u_range[0]:.1f}–{self.log_u_range[1]:.1f}, "
            f"log_γ²={self.log_gamma2_range[0]:.1f}–{self.log_gamma2_range[1]:.1f})"
        )

    @property
    def z_grid(self) -> np.ndarray:
        """~numpy.ndarray: Read-only view of the ion charge grid."""
        view = self._z_grid.view()
        view.flags.writeable = False
        return view

    @property
    def log_gamma2_grid(self) -> np.ndarray:
        r"""~numpy.ndarray: Read-only view of the :math:`\log_{10}\gamma^2` grid."""
        view = self._log_gamma2_grid.view()
        view.flags.writeable = False
        return view

    @property
    def log_u_grid(self) -> np.ndarray:
        r"""~numpy.ndarray: Read-only view of the :math:`\log_{10} u` grid."""
        view = self._log_u_grid.view()
        view.flags.writeable = False
        return view


# =============================================== #
# Analytic Approximations                         #
# =============================================== #
# These gaunt factors are taken broadly from the literature to provide
# analytic defaults for the relevant gaunt factors.


# ----------- Private Functions ------------------- #
def _gaunt_ff_lu(log_nu, log_T, Z):
    r"""
    Free–free Gaunt factor using the W. Lu textbook approximation.

    Parameters
    ----------
    log_nu : array_like
        Natural logarithm of the frequency in Hz.
    log_T : array_like
        Natural logarithm of the temperature in Kelvin.
    Z : array_like
        Ionic charge (atomic number) of the species. Must be positive.

    Returns
    -------
    gff : float or ~numpy.ndarray
        Free–free Gaunt factor :math:`g_{\rm ff}`. Shape is the broadcasted
        shape of the inputs.

    Notes
    -----
    Expressed in terms of :math:`\nu_9 = \nu / 10^9\,\mathrm{Hz}` and
    :math:`T_4 = T / 10^4\,\mathrm{K}`:

    .. math::

        g_{\rm ff} \approx \ln\!\left(e + \exp(Q)\right),\quad
        Q = 6 - \frac{\sqrt{3}}{\pi}\ln A,

    where

    .. math::

        A = \frac{\nu_9}{T_4} \cdot
            \max\!\left(0.25,\;\frac{Z}{T_4^{1/2}}\right)^{-1}.

    Note that :math:`\ln(e + \exp(Q))` is the Gaunt factor itself, not its
    logarithm; ``np.logaddexp`` is used purely for numerical stability.

    References
    ----------
    See :footcite:t:`lu_2026_18603474`, equation 6.79.
    """
    # Ensure everything is correctly case as logarithmic and in CGS-compatible units.
    log_nu = np.asarray(log_nu, dtype=float)
    log_T = np.asarray(log_T, dtype=float)
    Z = np.asarray(Z, dtype=float)

    # Convert the quantities to the log_10 space and correct them to
    # that they are in 10^4 K and 10^9 Hz, as per the original formula.
    log10 = np.log(10.0)
    log_nu9 = log_nu - 9.0 * log10
    log_T4 = log_T - 4.0 * log10
    log_Z = np.log(Z)

    # Compute the first term inside the logarithm.
    log_A = log_nu9 - log_T4 + np.maximum(np.log(0.25), log_Z - 0.5 * log_T4)

    # Q term
    Q = 6.0 - (np.sqrt(3.0) / np.pi) * log_A

    # Final stable log(e + exp(Q))
    res = np.logaddexp(1.0, Q)

    return res if res.ndim > 0 else float(res)


def _gaunt_ff_lu_comp(log_nu, log_T, Zs, Xs):
    r"""
    Composition-weighted effective free–free Gaunt factor (W. Lu textbook approximation).

    Parameters
    ----------
    log_nu : array_like
        Natural logarithm of the frequency in Hz.
    log_T : array_like
        Natural logarithm of the temperature in Kelvin.
    Zs : (N,) array_like
        Ionic charges of each species.
    Xs : (N,) array_like
        Number fractions of each species. Must sum to unity.

    Returns
    -------
    gff_eff : float or ~numpy.ndarray
        Effective Gaunt factor
        :math:`g_{\rm ff,eff} = \sum_i Z_i^2\,x_i\,g_{{\rm ff},i}`.
        Shape matches broadcasted ``log_nu`` and ``log_T``.

    Raises
    ------
    ValueError
        If ``Zs`` and ``Xs`` do not have matching shapes or are not 1-D.

    Notes
    -----
    Assumes a fully ionised plasma where free–free emission scales as
    :math:`Z_i^2`.  Weighting uses **number fractions**, not mass fractions.

    References
    ----------
    See :footcite:t:`lu_2026_18603474`, equation 6.79.
    """
    log_nu = np.asarray(log_nu, dtype=float)
    log_T = np.asarray(log_T, dtype=float)
    Zs = np.asarray(Zs, dtype=float)
    Xs = np.asarray(Xs, dtype=float)

    if Zs.shape != Xs.shape:
        raise ValueError(f"Zs and Xs must have the same shape; got {Zs.shape} and {Xs.shape}.")
    if Zs.ndim != 1:
        raise ValueError(f"Zs and Xs must be 1-D arrays; got shape {Zs.shape}.")

    gaunt_factors = _gaunt_ff_lu(log_nu[..., np.newaxis], log_T[..., np.newaxis], Zs[np.newaxis, :])
    weights = Zs**2 * Xs * gaunt_factors
    return np.sum(weights, axis=-1)


def _gaunt_ff_draine(log_nu, log_T, Z):
    r"""
    Free–free Gaunt factor using the Draine (2011) approximation.

    Parameters
    ----------
    log_nu : array_like
        Natural logarithm of the frequency in Hz.
    log_T : array_like
        Natural logarithm of the temperature in Kelvin.
    Z : array_like
        Ionic charge of the species. Must be positive.

    Returns
    -------
    float or ~numpy.ndarray
        Free–free Gaunt factor :math:`g_{\rm ff}`.

    Notes
    -----
    Expressed in terms of :math:`\nu_9 = \nu / 10^9\,\mathrm{Hz}` and
    :math:`T_4 = T / 10^4\,\mathrm{K}`:

    .. math::

        g_{\rm ff} \approx \ln\!\left(e + \exp(Q)\right), \quad
        Q = 5.960 - \frac{\sqrt{3}}{\pi}
            \ln\!\left(\nu_9\,T_4^{-3/2}\,Z\right).

    References
    ----------
    See :footcite:t:`draine2011physics`, Chapter 10, and
    :footcite:t:`2023MNRAS.519.3432T`.
    """
    log_nu = np.asarray(log_nu, dtype=float)
    log_T = np.asarray(log_T, dtype=float)
    Z = np.asarray(Z, dtype=float)

    log10 = np.log(10.0)
    log_nu9 = log_nu - 9.0 * log10
    log_T4 = log_T - 4.0 * log10
    log_Z = np.log(Z)

    log_A = log_nu9 - 1.5 * log_T4 + log_Z
    Q = 5.960 - (np.sqrt(3.0) / np.pi) * log_A
    res = np.logaddexp(1.0, Q)

    return res if res.ndim > 0 else float(res)


def _gaunt_ff_draine_comp(log_nu, log_T, Zs, Xs):
    r"""
    Composition-weighted effective free–free Gaunt factor (Draine 2011 approximation).

    Parameters
    ----------
    log_nu : array_like
        Natural logarithm of the frequency in Hz.
    log_T : array_like
        Natural logarithm of the temperature in Kelvin.
    Zs : (N,) array_like
        Ionic charges of each species.
    Xs : (N,) array_like
        Number fractions of each species.

    Returns
    -------
    float or ~numpy.ndarray
        Effective Gaunt factor
        :math:`g_{\rm ff,eff} = \sum_i Z_i^2\,x_i\,g_{{\rm ff},i}`.

    Raises
    ------
    ValueError
        If ``Zs`` and ``Xs`` do not have matching 1-D shapes.

    References
    ----------
    See :footcite:t:`draine2011physics`, Chapter 10, and
    :footcite:t:`2023MNRAS.519.3432T`.
    """
    log_nu = np.asarray(log_nu, dtype=float)
    log_T = np.asarray(log_T, dtype=float)
    Zs = np.asarray(Zs, dtype=float)
    Xs = np.asarray(Xs, dtype=float)

    if Zs.shape != Xs.shape:
        raise ValueError(f"Zs and Xs must have the same shape; got {Zs.shape} and {Xs.shape}.")
    if Zs.ndim != 1:
        raise ValueError(f"Zs and Xs must be 1-D arrays; got shape {Zs.shape}.")

    gaunt_factors = _gaunt_ff_draine(log_nu[..., np.newaxis], log_T[..., np.newaxis], Zs[np.newaxis, :])
    weights = Zs**2 * Xs * gaunt_factors
    return np.sum(weights, axis=-1)


def _average_gaunt_ff_draine(log_T, Z):
    r"""
    Frequency-averaged free–free Gaunt factor (Draine 2011 approximation).

    Computes the thermally averaged (frequency-integrated) Gaunt factor
    :math:`\bar{g}_{\rm ff}(T, Z)` using the analytic fit from
    :footcite:t:`draine2011physics`.  This quantity appears in **bolometric**
    free–free emissivity expressions where the spectrum has been integrated
    over frequency.

    The approximation is expressed as a function of temperature and ionic
    charge:

    .. math::

        \bar{g}_{\rm ff} \approx
        1 + \frac{0.44}{1 + 0.058\,Q^2},

    where

    .. math::

        Q = \ln\!\left(\frac{T}{10^{5.4}\,Z^2}\right).

    Parameters
    ----------
    log_T : float or array-like
        Natural logarithm of the electron temperature :math:`T` [K].
    Z : float or array-like
        Ionic charge number (:math:`Z \geq 1`).

    Returns
    -------
    float or ~numpy.ndarray
        Frequency-averaged Gaunt factor :math:`\bar{g}_{\rm ff}`.
        Shape follows broadcasting of *log_T* and *Z*.

    Notes
    -----
    This quantity differs from the frequency-dependent Gaunt factor
    :math:`g_{\rm ff}(\nu, T)` in that it is **integrated over frequency**
    with the appropriate thermal weighting.  It is therefore suitable for
    computing total (bolometric) bremsstrahlung emission:

    .. math::

        \epsilon_{\rm ff} \propto n_e n_i Z^2 T^{1/2} \bar{g}_{\rm ff}.

    The approximation is accurate to within a few percent over a wide
    temperature range relevant for astrophysical plasmas.

    See Also
    --------
    _gaunt_ff_draine :
        Frequency-dependent Gaunt factor.
    _average_gaunt_ff_draine_comp :
        Composition-weighted averaged Gaunt factor.
    """
    # Compute the Q factor ln(T/K/(10^5.4 Z^2))
    log_Q = log_T - 5.4 * np.log(10) - 2 * np.log(Z)

    res = 1 + (0.44 / (1 + (0.058 * log_Q**2)))

    return res if res.ndim > 0 else float(res)


def _average_gaunt_ff_draine_comp(log_T, Zs, Xs):
    r"""
    Composition-weighted frequency-averaged free–free Gaunt factor.

    Computes the effective averaged Gaunt factor for a multi-species plasma:

    .. math::

        \bar{g}_{\rm ff,eff}(T) =
        \sum_i Z_i^2\,x_i\,\bar{g}_{{\rm ff},i}(T, Z_i),

    where:

    * :math:`Z_i` is the ionic charge of species *i*,
    * :math:`x_i` is the number fraction of species *i*,
    * :math:`\bar{g}_{{\rm ff},i}` is the single-species averaged Gaunt factor
      from :func:`_average_gaunt_ff_draine`.

    Parameters
    ----------
    log_T : float or array-like
        Natural logarithm of the electron temperature :math:`T` [K].
    Zs : (N,) array-like
        Ionic charge numbers of each species (:math:`Z_i \geq 1`).
    Xs : (N,) array-like
        Number fractions of each species (:math:`\sum_i x_i = 1`).

    Returns
    -------
    float or ~numpy.ndarray
        Effective frequency-averaged Gaunt factor
        :math:`\bar{g}_{\rm ff,eff}`.
        Shape follows broadcasting of *log_T*.

    Raises
    ------
    ValueError
        If *Zs* and *Xs* do not have matching shapes or are not 1-D.

    Notes
    -----
    In a fully ionised plasma, the total free–free emissivity scales as

    .. math::

        \epsilon_{\rm ff} \propto
        n_e \sum_i n_i Z_i^2 \bar{g}_{{\rm ff},i},

    so the natural weighting is :math:`Z_i^2 x_i`, not simply :math:`x_i`.

    This function assumes:

    * fully ionised plasma,
    * number fractions (not mass fractions),
    * independent evaluation of each species' Gaunt factor.

    See Also
    --------
    _average_gaunt_ff_draine :
        Single-species averaged Gaunt factor.
    _gaunt_ff_draine_comp :
        Frequency-dependent composition-weighted Gaunt factor.
    """
    Zs = np.asarray(Zs, dtype=float)
    Xs = np.asarray(Xs, dtype=float)

    if Zs.shape != Xs.shape:
        raise ValueError(f"Zs and Xs must have the same shape; got {Zs.shape} and {Xs.shape}.")
    if Zs.ndim != 1:
        raise ValueError(f"Zs and Xs must be 1-D arrays; got shape {Zs.shape}.")

    gaunt_factors = _average_gaunt_ff_draine(log_T[..., np.newaxis], Zs[np.newaxis, :])
    weights = Zs**2 * Xs * gaunt_factors
    return np.sum(weights, axis=-1)


# =============================================== #
# Tabulated Wrappers (Van Hoof 2014)              #
# =============================================== #
# Lazily-loaded module-level interpolator instances.  These are created on the
# first call to a 'vanhoof' or 'vanhoof_rel' approximation and then reused.
# bounds_error=False / fill_value=NaN means out-of-domain queries return NaN
# rather than raising, consistent with the opacity module convention.

_DEFAULT_NON_REL_INTERP: "Optional[NonRelativisticGauntFactorInterpolator]" = None
_DEFAULT_REL_INTERP: "Optional[RelativisticGauntFactorInterpolator]" = None


def _get_non_rel_interp() -> "NonRelativisticGauntFactorInterpolator":
    global _DEFAULT_NON_REL_INTERP
    if _DEFAULT_NON_REL_INTERP is None:
        _DEFAULT_NON_REL_INTERP = NonRelativisticGauntFactorInterpolator(bounds_error=False, fill_value=np.nan)
    return _DEFAULT_NON_REL_INTERP


def _get_rel_interp() -> "RelativisticGauntFactorInterpolator":
    global _DEFAULT_REL_INTERP
    if _DEFAULT_REL_INTERP is None:
        _DEFAULT_REL_INTERP = RelativisticGauntFactorInterpolator(bounds_error=False, fill_value=np.nan)
    return _DEFAULT_REL_INTERP


def _gaunt_ff_vanhoof(log_nu, log_T, Z):
    r"""
    Free–free Gaunt factor via bilinear interpolation on the van Hoof et al. (2014) non-relativistic table.

    Parameters
    ----------
    log_nu : array_like
        Natural logarithm of the frequency in Hz.
    log_T : array_like
        Natural logarithm of the temperature in Kelvin.
    Z : array_like
        Ionic charge. Enters only through the coordinate transform
        :math:`\log_{10}\gamma^2 = \log_{10}(Z^2 R_y / k_B T)`.

    Returns
    -------
    float or ~numpy.ndarray
        Free–free Gaunt factor :math:`g_{\rm ff}`.  Points outside the table
        domain are returned as ``NaN``.

    Notes
    -----
    The table covers :math:`-16 \le \log_{10} u \le +13` and
    :math:`-6 \le \log_{10}\gamma^2 \le +10`.  The interpolator is
    lazy-loaded and cached on the first call.

    References
    ----------
    See :footcite:t:`2014MNRAS.444..420V`.
    """
    nu = np.exp(np.asarray(log_nu, dtype=float))
    T = np.exp(np.asarray(log_T, dtype=float))
    return _get_non_rel_interp()(Z=Z, T=T, nu=nu)


def _gaunt_ff_vanhoof_comp(log_nu, log_T, Zs, Xs):
    r"""
    Composition-weighted effective free–free Gaunt factor using the van Hoof et al. (2014) non-relativistic table.

    Parameters
    ----------
    log_nu : array_like
        Natural logarithm of the frequency in Hz.
    log_T : array_like
        Natural logarithm of the temperature in Kelvin.
    Zs : (N,) array_like
        Ionic charges of each species.
    Xs : (N,) array_like
        Number fractions of each species. Must sum to unity.

    Returns
    -------
    float or ~numpy.ndarray
        Effective Gaunt factor
        :math:`g_{\rm ff,eff} = \sum_i Z_i^2\,x_i\,g_{{\rm ff},i}`.
        Points where any species falls outside the table return ``NaN``.

    Raises
    ------
    ValueError
        If ``Zs`` and ``Xs`` do not have matching 1-D shapes.

    Notes
    -----
    Each per-species Gaunt factor is evaluated independently via
    :func:`_gaunt_ff_vanhoof`; the results are then combined as a
    number-fraction-weighted sum.  The interpolator is lazy-loaded and cached.

    References
    ----------
    See :footcite:t:`2014MNRAS.444..420V`.
    """
    Zs = np.asarray(Zs, dtype=float)
    Xs = np.asarray(Xs, dtype=float)
    if Zs.shape != Xs.shape:
        raise ValueError(f"Zs and Xs must have the same shape; got {Zs.shape} and {Xs.shape}.")
    if Zs.ndim != 1:
        raise ValueError(f"Zs and Xs must be 1-D arrays; got shape {Zs.shape}.")

    nu = np.exp(np.asarray(log_nu, dtype=float))
    T = np.exp(np.asarray(log_T, dtype=float))
    interp = _get_non_rel_interp()

    effective_gff = sum(Zi**2 * Xi * interp(Z=float(Zi), T=T, nu=nu) for Zi, Xi in zip(Zs, Xs))
    result = np.asarray(effective_gff)
    return float(result) if result.ndim == 0 else result


def _gaunt_ff_vanhoof_rel(log_nu, log_T, Z):
    r"""
    Free–free Gaunt factor via trilinear interpolation on the van Hoof et al. (2014) relativistic table.

    Parameters
    ----------
    log_nu : array_like
        Natural logarithm of the frequency in Hz.
    log_T : array_like
        Natural logarithm of the temperature in Kelvin.
    Z : array_like
        Ionic charge.  Used both through the coordinate transform and as an
        explicit interpolation axis (:math:`Z = 1, 2, \ldots, 36`).

    Returns
    -------
    float or ~numpy.ndarray
        Free–free Gaunt factor :math:`g_{\rm ff}`.  Points outside the table
        domain are returned as ``NaN``.

    Notes
    -----
    The table covers :math:`Z = 1\text{–}36`,
    :math:`-16 \le \log_{10} u \le +13`, and
    :math:`-6 \le \log_{10}\gamma^2 \le +10`.  The interpolator is
    lazy-loaded and cached on the first call.  Use this approximation when
    relativistic corrections to the Gaunt factor are significant
    (:math:`T \gtrsim 10^8\,\mathrm{K}`).

    References
    ----------
    See :footcite:t:`2014MNRAS.444..420V`.
    """
    nu = np.exp(np.asarray(log_nu, dtype=float))
    T = np.exp(np.asarray(log_T, dtype=float))
    return _get_rel_interp()(Z=Z, T=T, nu=nu)


def _gaunt_ff_vanhoof_rel_comp(log_nu, log_T, Zs, Xs):
    r"""
    Composition-weighted effective free–free Gaunt factor using the van Hoof et al. (2014) relativistic table.

    Parameters
    ----------
    log_nu : array_like
        Natural logarithm of the frequency in Hz.
    log_T : array_like
        Natural logarithm of the temperature in Kelvin.
    Zs : (N,) array_like
        Ionic charges of each species.  Each must lie within
        :math:`Z = 1, 2, \ldots, 36`.
    Xs : (N,) array_like
        Number fractions of each species. Must sum to unity.

    Returns
    -------
    float or ~numpy.ndarray
        Effective Gaunt factor
        :math:`g_{\rm ff,eff} = \sum_i Z_i^2\,x_i\,g_{{\rm ff},i}`.
        Points outside the table (including :math:`Z > 36`) return ``NaN``.

    Raises
    ------
    ValueError
        If ``Zs`` and ``Xs`` do not have matching 1-D shapes.

    Notes
    -----
    Each per-species Gaunt factor is evaluated independently via
    :func:`_gaunt_ff_vanhoof_rel`; the results are combined as a
    number-fraction-weighted sum.  The interpolator is lazy-loaded and cached.

    References
    ----------
    See :footcite:t:`2014MNRAS.444..420V`.
    """
    Zs = np.asarray(Zs, dtype=float)
    Xs = np.asarray(Xs, dtype=float)
    if Zs.shape != Xs.shape:
        raise ValueError(f"Zs and Xs must have the same shape; got {Zs.shape} and {Xs.shape}.")
    if Zs.ndim != 1:
        raise ValueError(f"Zs and Xs must be 1-D arrays; got shape {Zs.shape}.")

    nu = np.exp(np.asarray(log_nu, dtype=float))
    T = np.exp(np.asarray(log_T, dtype=float))
    interp = _get_rel_interp()

    effective_gff = sum(Zi**2 * Xi * interp(Z=float(Zi), T=T, nu=nu) for Zi, Xi in zip(Zs, Xs))
    result = np.asarray(effective_gff)
    return float(result) if result.ndim == 0 else result


# ----------- Public Functions ------------------- #

_APPROX_REGISTRY = {
    "lu": (_gaunt_ff_lu, _gaunt_ff_lu_comp),
    "draine": (_gaunt_ff_draine, _gaunt_ff_draine_comp),
    "vanhoof": (_gaunt_ff_vanhoof, _gaunt_ff_vanhoof_comp),
    "vanhoof_rel": (_gaunt_ff_vanhoof_rel, _gaunt_ff_vanhoof_rel_comp),
}

_AVERAGE_APPROX_REGISTRY = {
    "draine": (_average_gaunt_ff_draine, _average_gaunt_ff_draine_comp),
}


def compute_ff_gaunt_factor(
    nu: "_UnitBearingArrayLike",
    T: "_UnitBearingArrayLike",
    Z: "_ArrayLike" = 1,
    approx: str = "lu",
) -> "Union[float, np.ndarray]":
    r"""Evaluate the free–free Gaunt factor :math:`g_{\rm ff}(\nu, T, Z)` for a single ion species.

    Parameters
    ----------
    nu : float, int, ~numpy.ndarray, or ~astropy.units.Quantity
        Photon frequency.  Plain floats are assumed to be in Hz;
        unit-bearing Quantities are converted automatically.
    T : float, int, ~numpy.ndarray, or ~astropy.units.Quantity
        Electron temperature.  Plain floats are assumed to be in K;
        unit-bearing Quantities are converted automatically.
    Z : int, float, or ~numpy.ndarray, optional
        Ionic charge number (:math:`Z \geq 1`).  Dimensionless. Default is 1 (proton).
    approx : {'lu', 'draine', 'vanhoof', 'vanhoof_rel'}, optional
        Approximation method.  Default is ``'lu'``. See notes regarding the available
        approximation methods.

    Returns
    -------
    float or ~numpy.ndarray
        Dimensionless free–free Gaunt factor :math:`g_{\rm ff}`.  Shape is
        the broadcasted shape of *nu*, *T*, and *Z*.  Tabulated methods
        return ``NaN`` for queries outside the table domain.

    Raises
    ------
    ValueError
        If *approx* is not one of the recognised keys.

    Notes
    -----
    Four methods are available:

    ``'lu'`` (default)
        Analytic extension of the Draine (2011) formula by
        :footcite:t:`lu_2026_18603474` (equation 6.79), expressed in terms of
        :math:`\nu_9 \equiv \nu / (10^9\,\mathrm{Hz})` and
        :math:`T_4 \equiv T / (10^4\,\mathrm{K})`:

        .. math::

            g_{\rm ff} \approx \ln\!\left(e + \exp(Q)\right),\quad
            Q = 6 - \frac{\sqrt{3}}{\pi}\ln A,\quad
            A = \frac{\nu_9}{T_4}
                \max\!\left(0.25,\;\frac{Z}{T_4^{1/2}}\right)^{-1}.

        Covers both the classical and semi-quantum non-relativistic regimes.
        Valid for arbitrary :math:`Z`.

    ``'draine'``
        Classical analytic approximation from
        :footcite:t:`draine2011physics` (Chapter 10) and
        :footcite:t:`2023MNRAS.519.3432T`:

        .. math::

            g_{\rm ff} \approx \ln\!\left(e + \exp(Q)\right),\quad
            Q = 5.960 - \frac{\sqrt{3}}{\pi}
                \ln\!\left(\nu_9\,T_4^{-3/2}\,Z\right).

        Suitable when only the classical non-relativistic regime is needed.

    ``'vanhoof'``
        Bilinear interpolation on the 2-D non-relativistic table of
        :footcite:t:`2014MNRAS.444..420V`.  Covers
        :math:`-16 \le \log_{10} u \le +13` and
        :math:`-6 \le \log_{10}\gamma^2 \le +10`.
        The interpolator is lazy-loaded from disk and cached on the first
        call; subsequent calls are fast.  Recommended when accuracy matters
        and relativistic corrections are negligible.

    ``'vanhoof_rel'``
        Trilinear interpolation on the 3-D relativistic table of
        :footcite:t:`2014MNRAS.444..420V` with :math:`Z = 1\text{–}36`.
        Covers the same :math:`(u, \gamma^2)` domain as ``'vanhoof'``.
        Appropriate for high-temperature plasmas
        (:math:`T \gtrsim 10^8\,\mathrm{K}`) where relativistic corrections
        are significant.

    See Also
    --------
    compute_ff_gaunt_factor_comp :
        Composition-weighted effective Gaunt factor for a multi-species plasma.
    get_default_gaunt_interpolator :
        Direct access to the non-relativistic van Hoof interpolator.
    get_default_relativistic_gaunt_interpolator :
        Direct access to the relativistic van Hoof interpolator.

    References
    ----------
    .. footbibliography::

    Examples
    --------
    .. code-block:: python

        from astropy import units as u
        from trilobite.radiation.free_free.gaunt_factor import (
            compute_ff_gaunt_factor,
        )

        # Analytic (default)
        gff = compute_ff_gaunt_factor(
            nu=1e10 * u.Hz, T=1e4 * u.K, Z=1
        )
        print(f"g_ff (lu)      = {gff:.4f}")

        # Tabulated non-relativistic
        gff_tab = compute_ff_gaunt_factor(
            nu=1e10 * u.Hz,
            T=1e4 * u.K,
            Z=1,
            approx="vanhoof",
        )
        print(f"g_ff (vanhoof) = {gff_tab:.4f}")
    """
    if approx not in _APPROX_REGISTRY:
        raise ValueError(f"approx must be one of {list(_APPROX_REGISTRY)}; got {approx!r}.")

    nu_cgs = ensure_in_units(nu, u.Hz)
    T_cgs = ensure_in_units(T, u.K)

    _single, _ = _APPROX_REGISTRY[approx]
    return _single(np.log(nu_cgs), np.log(T_cgs), Z)


def compute_ff_gaunt_factor_comp(
    nu: "_UnitBearingArrayLike",
    T: "_UnitBearingArrayLike",
    Zs: "_ArrayLike",
    Xs: "_ArrayLike",
    approx: str = "lu",
) -> "Union[float, np.ndarray]":
    r"""Evaluate the composition-weighted effective free–free Gaunt factor.

    Parameters
    ----------
    nu : float, int, ~numpy.ndarray, or ~astropy.units.Quantity
        Photon frequency.  Plain floats are assumed to be in Hz;
        unit-bearing Quantities are converted automatically.
    T : float, int, ~numpy.ndarray, or ~astropy.units.Quantity
        Electron temperature.  Plain floats are assumed to be in K;
        unit-bearing Quantities are converted automatically.
    Zs : ~numpy.ndarray
        Ionic charge numbers of each species (:math:`Z_i \geq 1`).
        Dimensionless.
    Xs : ~numpy.ndarray
        Number fractions of each species (:math:`\sum_i x_i = 1`).
        Must match the shape of *Zs*.
    approx : {'lu', 'draine', 'vanhoof', 'vanhoof_rel'}, optional
        Method used to evaluate the per-species Gaunt factor.
        Default is ``'lu'``.

    Returns
    -------
    float or ~numpy.ndarray
        Dimensionless effective Gaunt factor :math:`g_{\rm ff,eff}`.  Shape
        matches the broadcasted shape of *nu* and *T*.  Tabulated methods
        return ``NaN`` for species or frequencies outside the table domain.

    Raises
    ------
    ValueError
        If *approx* is not recognised, or if *Zs* and *Xs* have incompatible
        shapes or are not 1-D.

    Notes
    -----
    For a fully ionised multi-species plasma the free–free volume emissivity
    scales as :math:`j_\nu \propto n_e \sum_i n_i Z_i^2 g_{{\rm ff},i}`.
    Expressing ion densities as number fractions :math:`x_i = n_i / n_{\rm ion}`
    yields the effective Gaunt factor

    .. math::

        g_{\rm ff,eff}(\nu, T) = \sum_i Z_i^2\,x_i\,g_{{\rm ff},i}(\nu, T, Z_i),

    where :math:`g_{{\rm ff},i}` is evaluated by the method selected through
    *approx*.  See :func:`compute_ff_gaunt_factor` for a description of each
    available method; all four (``'lu'``, ``'draine'``, ``'vanhoof'``,
    ``'vanhoof_rel'``) are supported here.

    For the tabulated methods (``'vanhoof'``, ``'vanhoof_rel'``), the
    per-species evaluations iterate over species in a Python loop; the
    interpolator is lazy-loaded and cached on the first call.

    See Also
    --------
    compute_ff_gaunt_factor :
        Single-species Gaunt factor; documents all available *approx* methods.
    get_default_gaunt_interpolator :
        Direct access to the non-relativistic van Hoof interpolator.

    References
    ----------
    .. footbibliography::

    Examples
    --------
    .. code-block:: python

        import numpy as np
        from astropy import units as u
        from trilobite.radiation.free_free.gaunt_factor import (
            compute_ff_gaunt_factor_comp,
        )

        # Cosmic number fractions: ~90% H, ~10% He
        Zs = np.array([1, 2])
        Xs = np.array([0.90, 0.10])
        nu_arr = np.logspace(8, 15, 200) * u.Hz

        # Analytic (default)
        gff_eff = compute_ff_gaunt_factor_comp(
            nu=nu_arr,
            T=1e4 * u.K,
            Zs=Zs,
            Xs=Xs,
        )

        # Tabulated non-relativistic
        gff_eff_tab = compute_ff_gaunt_factor_comp(
            nu=nu_arr,
            T=1e4 * u.K,
            Zs=Zs,
            Xs=Xs,
            approx="vanhoof",
        )
    """
    if approx not in _APPROX_REGISTRY:
        raise ValueError(f"approx must be one of {list(_APPROX_REGISTRY)}; got {approx!r}.")

    nu_cgs = ensure_in_units(nu, u.Hz)
    T_cgs = ensure_in_units(T, u.K)

    _, _comp = _APPROX_REGISTRY[approx]
    return _comp(np.log(nu_cgs), np.log(T_cgs), Zs, Xs)


def compute_mean_ff_gaunt_factor(
    T: "_UnitBearingArrayLike",
    Z: "_ArrayLike" = 1,
    approx: str = "draine",
) -> "Union[float, np.ndarray]":
    r"""Evaluate the frequency-averaged free–free Gaunt factor for a single ion species.

    Returns the thermally averaged (frequency-integrated) Gaunt factor
    :math:`\bar{g}_{\rm ff}(T, Z)` used in bolometric free–free luminosity
    calculations.  Unlike :func:`compute_ff_gaunt_factor`, this quantity has no
    frequency dependence.

    Parameters
    ----------
    T : float, int, ~numpy.ndarray, or ~astropy.units.Quantity
        Electron temperature.  Plain floats are assumed to be in K;
        unit-bearing Quantities are converted automatically.
    Z : int, float, or ~numpy.ndarray, optional
        Ionic charge number (:math:`Z \geq 1`).  Dimensionless. Default is 1.
    approx : {'draine'}, optional
        Approximation method.  Currently only ``'draine'`` is available.
        Default is ``'draine'``.

    Returns
    -------
    float or ~numpy.ndarray
        Dimensionless frequency-averaged Gaunt factor :math:`\bar{g}_{\rm ff}`.
        Shape is the broadcasted shape of *T* and *Z*.

    Raises
    ------
    ValueError
        If *approx* is not one of the recognised keys.

    See Also
    --------
    compute_ff_gaunt_factor : Frequency-specific Gaunt factor.
    compute_mean_ff_gaunt_factor_comp : Composition-weighted variant.
    """
    if approx not in _AVERAGE_APPROX_REGISTRY:
        raise ValueError(f"approx must be one of {list(_AVERAGE_APPROX_REGISTRY)}; got {approx!r}.")

    T_cgs = ensure_in_units(T, u.K)

    _single, _ = _AVERAGE_APPROX_REGISTRY[approx]
    return _single(np.log(T_cgs), Z)


def compute_mean_ff_gaunt_factor_comp(
    T: "_UnitBearingArrayLike",
    Zs: "_ArrayLike",
    Xs: "_ArrayLike",
    approx: str = "draine",
) -> "Union[float, np.ndarray]":
    r"""Evaluate the composition-weighted frequency-averaged free–free Gaunt factor.

    Returns :math:`\sum_i Z_i^2\, x_i\, \bar{g}_{{\rm ff},i}(T, Z_i)` where
    :math:`\bar{g}_{{\rm ff},i}` is the frequency-averaged Gaunt factor for
    species *i*.  Used in bolometric free–free luminosity calculations for
    multi-species plasmas.

    Parameters
    ----------
    T : float, int, ~numpy.ndarray, or ~astropy.units.Quantity
        Electron temperature.  Plain floats are assumed to be in K;
        unit-bearing Quantities are converted automatically.
    Zs : ~numpy.ndarray
        Ionic charge numbers of each species (:math:`Z_i \geq 1`).
        Dimensionless.
    Xs : ~numpy.ndarray
        Number fractions of each species (:math:`\sum_i x_i = 1`).
        Must match the shape of *Zs*.
    approx : {'draine'}, optional
        Method used to evaluate the per-species averaged Gaunt factor.
        Currently only ``'draine'`` is available.  Default is ``'draine'``.

    Returns
    -------
    float or ~numpy.ndarray
        Dimensionless effective frequency-averaged Gaunt factor
        :math:`\sum_i Z_i^2\, x_i\, \bar{g}_{{\rm ff},i}`.
        Shape matches the broadcasted shape of *T*.

    Raises
    ------
    ValueError
        If *approx* is not recognised, or if *Zs* and *Xs* have incompatible
        shapes or are not 1-D.

    See Also
    --------
    compute_ff_gaunt_factor_comp : Frequency-specific composition-weighted variant.
    compute_mean_ff_gaunt_factor : Single-species variant.
    """
    if approx not in _AVERAGE_APPROX_REGISTRY:
        raise ValueError(f"approx must be one of {list(_AVERAGE_APPROX_REGISTRY)}; got {approx!r}.")

    T_cgs = ensure_in_units(T, u.K)

    _, _comp = _AVERAGE_APPROX_REGISTRY[approx]
    return _comp(np.log(T_cgs), Zs, Xs)


# ================================================ #
# Convenience Factories                            #
# ================================================ #
def get_default_gaunt_interpolator(**kwargs) -> NonRelativisticGauntFactorInterpolator:
    r"""Return a :class:`NonRelativisticGauntFactorInterpolator` loaded from the bundled table.

    The bundled table is the non-relativistic free–free Gaunt factor grid of
    :footcite:t:`2014MNRAS.444..420V`. The table covers:

    * :math:`\log_{10}\gamma^2` from :math:`-6` to :math:`+10` (spacing 0.2 dex,
      81 points).
    * :math:`\log_{10} u` from :math:`-16` to :math:`+13` (spacing 0.2 dex,
      146 points).

    Parameters
    ----------
    **kwargs
        Keyword arguments forwarded verbatim to
        :class:`~scipy.interpolate.RegularGridInterpolator`.

    Returns
    -------
    NonRelativisticGauntFactorInterpolator
        A pre-loaded, callable interpolator.

    Examples
    --------
    .. code-block:: python

        from trilobite.radiation.free_free.gaunt_factor import (
            get_default_gaunt_interpolator,
        )

        interp = get_default_gaunt_interpolator()
        gff = interp(Z=1, T=1e4, nu=1e10)
        print(f"g_ff = {gff:.4f}")

    See Also
    --------
    NonRelativisticGauntFactorInterpolator : The underlying interpolator class.
    get_default_relativistic_gaunt_interpolator : Relativistic counterpart.

    References
    ----------
    .. footbibliography::
    """
    return NonRelativisticGauntFactorInterpolator(**kwargs)


def get_default_relativistic_gaunt_interpolator(**kwargs) -> RelativisticGauntFactorInterpolator:
    r"""Return a :class:`RelativisticGauntFactorInterpolator` loaded from the bundled table.

    The bundled table is the relativistic free–free Gaunt factor grid of
    :footcite:t:`2014MNRAS.444..420V`. The table covers:

    * Ion charges :math:`Z = 1, 2, \ldots, 36`.
    * :math:`\log_{10}\gamma^2` from :math:`-6` to :math:`+10` (spacing 0.2 dex,
      81 points).
    * :math:`\log_{10} u` from :math:`-16` to :math:`+13` (spacing 0.2 dex,
      146 points).

    Parameters
    ----------
    **kwargs
        Keyword arguments forwarded verbatim to
        :class:`~scipy.interpolate.RegularGridInterpolator`.

    Returns
    -------
    RelativisticGauntFactorInterpolator
        A pre-loaded, callable interpolator.

    Examples
    --------
    .. code-block:: python

        from trilobite.radiation.free_free.gaunt_factor import (
            get_default_relativistic_gaunt_interpolator,
        )

        interp = (
            get_default_relativistic_gaunt_interpolator()
        )
        gff = interp(Z=1, T=1e8, nu=1e18)
        print(f"g_ff = {gff:.4f}")

    See Also
    --------
    RelativisticGauntFactorInterpolator : The underlying interpolator class.
    get_default_gaunt_interpolator : Non-relativistic counterpart.

    References
    ----------
    .. footbibliography::
    """
    return RelativisticGauntFactorInterpolator(**kwargs)


def _resolve_gff(
    log_nu: float,
    log_T: float,
    Z: "Union[float, _ArrayLike]",
    X: "Optional[_ArrayLike]",
    g_ff: "Optional[Union[float, _ArrayLike]]",
) -> "tuple[float, float]":
    r"""Resolve ``(Z_eff, g_ff_eff)`` for free-free emission calculations.

    Parameters
    ----------
    log_nu : float
        Natural logarithm of the photon frequency [Hz].
    log_T : float
        Natural logarithm of the electron temperature [K].
    Z : float or array-like
        Ionic charge number(s).  Scalar for single-species; 1-D array when
        *X* is provided.
    X : array-like or None
        Number fractions of each species (:math:`\sum_i x_i = 1`).  When
        ``None``, the single-species path is taken.  Must have the same shape
        as *Z* when provided.
    g_ff : float, array-like, or None
        Free–free Gaunt factor(s).  When ``None``, computed automatically via
        the Lu+ prescription (:func:`_gaunt_ff_lu`).  Must match the shape of
        *Z* when *X* is provided.

    Returns
    -------
    Z_eff : float
        Effective ionic charge: the input *Z* (single-species) or ``1.0``
        (composition, since :math:`Z^2` is folded into *g_ff_eff*).
    g_ff_eff : float
        Effective Gaunt factor: the input *g_ff* (single-species) or the
        composition-weighted sum

        .. math::

            g_{\rm ff,eff} = \sum_i Z_i^2\, x_i\, g_{{\rm ff},i}

    Notes
    -----
    When *X* is provided and *g_ff* is ``None``, the per-species Gaunt factors
    are evaluated at the given ``(log_nu, log_T)`` using :func:`_gaunt_ff_lu`.
    When *X* is ``None`` and *g_ff* is ``None``, the single-species Gaunt
    factor is similarly computed via :func:`_gaunt_ff_lu`.
    """
    if X is not None:
        Z = np.asarray(Z, dtype=float)
        X = np.asarray(X, dtype=float)
        if g_ff is None:
            g_ff = np.array([float(_gaunt_ff_lu(log_nu, log_T, Zi)) for Zi in Z])
        else:
            g_ff = np.asarray(g_ff, dtype=float)
        return 1.0, float(np.sum(Z**2 * X * g_ff))
    else:
        if g_ff is None:
            g_ff = float(_gaunt_ff_lu(log_nu, log_T, float(Z)))
        return float(Z), float(g_ff)
