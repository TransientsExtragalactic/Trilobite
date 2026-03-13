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

Relevant API
------------
- :class:`~radiation.free_free.gaunt_factor.NonRelativisticGauntFactorInterpolator`
- :class:`~radiation.free_free.gaunt_factor.RelativisticGauntFactorInterpolator`
- :func:`~radiation.free_free.gaunt_factor.get_default_gaunt_interpolator`
- :func:`~radiation.free_free.gaunt_factor.get_default_relativistic_gaunt_interpolator`
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Union

import h5py
import numpy as np
from astropy import units as u
from scipy.interpolate import RegularGridInterpolator

from triceratops.utils.misc_utils import ensure_in_units

from ...utils.log import triceratops_logger
from ..constants import Ry_cgs, h_cgs, kB_cgs

if TYPE_CHECKING:
    pass  # Reserved for future type-only imports.

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

        triceratops_logger.info("Loading Gaunt factor table from %s.", path)
        datasets = {}
        with h5py.File(path, "r") as f:
            for key in ("log_gamma2", "log_u", "gff"):
                if key not in f:
                    raise KeyError(f"Gaunt factor table at {path} is missing required dataset '{key}'.")
                datasets[key] = f[key][:]
            if "Z" in f:
                datasets["Z"] = f["Z"][:]
        triceratops_logger.debug("Retrieved Gaunt factor table from %s.", path)
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

        from triceratops.radiation.free_free.gaunt_factor import (
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
        triceratops_logger.info("Loading Gaunt factor table from %s. [DONE]", path)

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

        from triceratops.radiation.free_free.gaunt_factor import (
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
        triceratops_logger.info("Loading Gaunt factor table from %s. [DONE]", path)

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

        from triceratops.radiation.free_free.gaunt_factor import (
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

        from triceratops.radiation.free_free.gaunt_factor import (
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


# =============================================== #
# Analytic Approximations                         #
# =============================================== #
def _log_gaunt_ff_draine(log_Z, log_T, log_nu):
    r"""
    Evaluate the Draine analytic approximation to the free–free Gaunt factor.

    This is the low-level implementation of the Draine approximation for the
    thermal free–free Gaunt factor. It operates entirely on **
    logarithmic inputs in CGS-compatible units** and returns the Gaunt factor
    in **linear space**.

    The approximation is

    .. math::

        g_{\rm ff} \approx
        \frac{\sqrt{3}}{\pi}
        \ln\!\left[
            \exp\!\left(
                5.960 - \frac{\sqrt{3}}{\pi}\ln Z
            \right)
            +
            e \frac{T^{3/2}}{\nu}
        \right].

    Parameters
    ----------
    log_Z : float or array-like
        logarithm of the ion charge :math:`Z`.
    log_T : float or array-like
        logarithm of the electron temperature :math:`T` in Kelvin.
    log_nu : float or array-like
        logarithm of the photon frequency :math:`\nu` in Hz.

    Returns
    -------
    float or ndarray
        Approximate free–free Gaunt factor :math:`g_{\rm ff}` in linear space.

    Notes
    -----
    This function assumes that the inputs are already valid and expressed in the
    expected logarithmic form. It performs no unit handling or physical
    validation beyond the requirements of the numerical expression itself.

    See Also
    --------
    gaunt_ff_draine
        Public, unit-aware wrapper around this low-level implementation.
    """
    log_Z = np.asarray(log_Z, dtype=float)
    log_T = np.asarray(log_T, dtype=float)
    log_nu = np.asarray(log_nu, dtype=float)

    pref = np.sqrt(3.0) / np.pi

    term1 = np.exp(5.960 - pref * log_Z)
    term2 = np.exp(1.0) * np.exp(1.5 * log_T - log_nu)

    return pref * np.log(term1 + term2)


def gaunt_ff_draine(Z, T, nu):
    r"""
    Analytic approximation to the thermal free–free Gaunt factor.

    This function evaluates the Draine free–free Gaunt factor approximation
    using unit-aware inputs. Inputs are coerced to the expected units using
    :func:`~triceratops.utils.misc_utils.ensure_in_units`, converted to
    base-10 logarithmic form, and passed to
    :func:`_log_gaunt_ff_draine`.

    Parameters
    ----------
    Z : float or array-like
        Ion charge :math:`Z`. This quantity is dimensionless.
    T : float, array-like, or `~astropy.units.Quantity`
        Electron temperature. If unitless, interpreted as Kelvin.
    nu : float, array-like, or `~astropy.units.Quantity`
        Photon frequency. If unitless, interpreted as Hz.

    Returns
    -------
    float or ndarray
        Approximate free–free Gaunt factor :math:`g_{\rm ff}` in linear space.

    Notes
    -----
    The approximation used is

    .. math::

        g_{\rm ff} \approx
        \frac{\sqrt{3}}{\pi}
        \ln\!\left[
            \exp\!\left(
                5.960 - \frac{\sqrt{3}}{\pi}\ln Z
            \right)
            +
            e \frac{T^{3/2}}{\nu}
        \right].

    This approximation is fast and useful for quick estimates, especially in
    radio and infrared regimes, but it is generally less accurate than a
    dedicated tabulated interpolator.

    Examples
    --------
    .. code-block:: python

        gff = gaunt_ff_draine(Z=1, T=1e4, nu=1e10)

    .. code-block:: python

        from astropy import units as u

        gff = gaunt_ff_draine(
            Z=1, T=1e6 * u.K, nu=5.0 * u.GHz
        )

    See Also
    --------
    _log_gaunt_ff_draine
        Low-level log-space implementation.
    NonRelativisticGauntFactorInterpolator
        Table-based, higher-accuracy interpolator for free–free Gaunt factors.
    """
    Z = np.asarray(Z, dtype=float)
    T = np.asarray(ensure_in_units(T, u.K), dtype=float)
    nu = np.asarray(ensure_in_units(nu, u.Hz), dtype=float)

    return _log_gaunt_ff_draine(
        np.log(Z),
        np.log(T),
        np.log(nu),
    )
