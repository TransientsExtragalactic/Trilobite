"""
Photometry utilities for interacting with and processing photometric data in Trilobite.

This module provides tools for filter-based photometry, including individual filter
response curves, batched filter bundles optimised for MCMC hot loops, and conversions
between flux density and standard magnitude systems (AB, ST).
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import astropy.units as u
import h5py
import numpy as np
from astropy import constants as const
from astropy.table import Table

# numpy <2.0 uses np.trapz; numpy >=2.0 renamed it to np.trapezoid
try:
    _trapz = np.trapezoid  # type: ignore[attr-defined]
except AttributeError:
    _trapz = np.trapz  # type: ignore[attr-defined]

from trilobite.utils.misc_utils import ensure_in_units

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

    from trilobite._typing import _UnitBearingArrayLike

# ============================================================================== #
# Magnitude System Zero Points                                                   #
# ============================================================================== #

#: AB magnitude zero point in erg/s/cm²/Hz.  F_nu at AB magnitude = 0.
_AB_ZERO_POINT_CGS: float = 3.631e-20

#: ST magnitude zero point offset constant.  m_ST = -2.5*log10(F_lambda) - 21.1
#: where F_lambda is in erg/s/cm²/Å.
_ST_ZERO_POINT_OFFSET: float = 21.1


# ============================================================================== #
# Filters and Photometric Systems                                                #
# ============================================================================== #
class PhotometryFilter:
    r"""
    A photometric bandpass filter with precomputed integration weights.

    Stores a filter transmission curve on a wavelength grid and precomputes
    normalised integration weights so that filter convolution reduces to a
    single dot product ``np.dot(F_nu, weights)``. The weights correctly
    account for the Jacobian of the wavelength-to-frequency transformation.

    Parameters
    ----------
    wavelength : array-like with units or bare ndarray in cm
        Wavelength grid of the filter. If an :class:`astropy.units.Quantity`
        is provided it is converted to cm. A bare array is assumed to be in cm.
    transmission : array-like
        Dimensionless transmission values on the wavelength grid. Must have
        the same shape as ``wavelength``. Values are not required to be
        normalised; normalisation is applied internally when computing weights.
    name : str, optional
        Human-readable name for the filter (e.g. ``"sdss-r"``). Stored for
        display and IO purposes only.
    weighting : {"photon", "energy"}, optional
        Photon-counting detectors (CCDs) should use ``"photon"`` (default).
        Bolometers and energy-sensitive detectors should use ``"energy"``.

    Raises
    ------
    ValueError
        If ``wavelength`` and ``transmission`` have different shapes, or if
        ``weighting`` is not one of ``"photon"`` or ``"energy"``.

    Notes
    -----
    **Photon weighting** (default for CCDs):

    .. math::

        langle F_nu rangle = frac{int F_nu(lambda) , T(lambda) ,
        frac{lambda}{hc} , dlambda}{int T(lambda) ,
        frac{lambda}{hc} , dlambda}

    **Energy weighting** (bolometers):

    .. math::

        langle F_nu rangle = frac{int F_nu(lambda) , T(lambda) ,
        frac{c}{lambda^2} , dlambda}{int T(lambda) ,
        frac{c}{lambda^2} , dlambda}

    The precomputed ``weights`` array encodes these integrals discretely via
    :func:`numpy.gradient`, so that ``np.dot(F_nu_on_filter_grid, weights)``
    gives the band-averaged flux density in the same units as ``F_nu``.

    Examples
    --------

    .. code-block:: python

        import numpy as np
        import astropy.units as u

        lam = np.linspace(5500, 7000, 300) * u.AA
        T = np.exp(-0.5 * ((lam.value - 6200) / 400) ** 2)
        filt = PhotometryFilter(lam, T, name="r-band")
        F_nu = (
            np.ones(300) * 3.631e-20
        )  # flat 0-mag spectrum
        filt.apply(F_nu)  # doctest: +SKIP
        3.631e-20
    """

    def __init__(
        self,
        wavelength: "_UnitBearingArrayLike",
        transmission: np.ndarray,
        name: Optional[str] = None,
        weighting: str = "photon",
    ):
        # ------------------------------------------------------------------ #
        # Validate and coerce inputs                                          #
        # ------------------------------------------------------------------ #
        wavelength = np.asarray(ensure_in_units(wavelength, "cm"))
        transmission = np.asarray(transmission, dtype=float)

        if wavelength.shape != transmission.shape:
            raise ValueError(
                "wavelength and transmission must have the same shape. "
                f"Got {wavelength.shape} and {transmission.shape}."
            )

        if weighting not in ("energy", "photon"):
            raise ValueError(f"weighting must be 'photon' or 'energy', got '{weighting}'.")

        self._name = name if name is not None else ""
        self._weighting = weighting
        self._wavelength = wavelength
        self._transmission = transmission

        # Frequency grid (same ordering as wavelength grid)
        self._frequency = const.c.cgs.value / wavelength  # Hz

        # ------------------------------------------------------------------ #
        # Precompute integration weights                                      #
        # ------------------------------------------------------------------ #
        # delta_lambda via central differences (units: cm)
        delta_lambda = np.gradient(wavelength)

        if weighting == "photon":
            # Photon-weighted: T(lambda) * (1/lambda) * |dlambda|
            # (hc cancels in normalisation)
            raw = transmission * (1.0 / wavelength) * np.abs(delta_lambda)
        else:
            # Energy-weighted: T(lambda) * |dnu/dlambda| * |dlambda|
            #                 = T(lambda) * (c/lambda^2) * |dlambda|
            raw = transmission * (const.c.cgs.value / wavelength**2) * np.abs(delta_lambda)

        total = raw.sum()
        self._weights = raw / total if total > 0.0 else raw

        # ------------------------------------------------------------------ #
        # Derived metadata                                                    #
        # ------------------------------------------------------------------ #
        self._wavelength_bounds = (float(np.amin(wavelength)), float(np.amax(wavelength)))
        self._frequency_bounds = (float(np.amin(self._frequency)), float(np.amax(self._frequency)))

    # ====================================================================== #
    # Properties                                                             #
    # ====================================================================== #

    @property
    def name(self) -> str:
        """Human-readable filter name (empty string if not set)."""
        return self._name

    @property
    def wavelength(self) -> np.ndarray:
        """Wavelength grid in cm."""
        return self._wavelength

    @property
    def frequency(self) -> np.ndarray:
        """Frequency grid in Hz (same ordering as :attr:`wavelength`)."""
        return self._frequency

    @property
    def transmission(self) -> np.ndarray:
        """Raw transmission values (not normalised)."""
        return self._transmission

    @property
    def weights(self) -> np.ndarray:
        """
        Precomputed, normalised integration weights (on the wavelength grid).

        The weights satisfy ``np.dot(F_nu_on_filter_grid, weights) == <F_nu>``
        where ``<F_nu>`` is the band-averaged flux density.
        """
        return self._weights

    @property
    def weighting(self) -> str:
        """Weighting scheme, either ``'photon'`` or ``'energy'``."""
        return self._weighting

    @property
    def wavelength_bounds(self) -> tuple:
        """``(lambda_min, lambda_max)`` of the filter grid in cm."""
        return self._wavelength_bounds

    @property
    def frequency_bounds(self) -> tuple:
        """``(nu_min, nu_max)`` of the filter grid in Hz."""
        return self._frequency_bounds

    @property
    def effective_wavelength(self) -> float:
        r"""
        Pivot (effective) wavelength of the filter in cm.

        Defined as the photon-weighted pivot wavelength independent of the
        chosen weighting scheme:

        .. math::

            lambda_{rm pivot} = sqrt{
                frac{int T(lambda),dlambda}{int T(lambda)/lambda^2,dlambda}
            }
        """
        numerator = _trapz(self._transmission, self._wavelength)
        denominator = _trapz(self._transmission / self._wavelength**2, self._wavelength)
        if denominator == 0.0:
            return float(np.average(self._wavelength, weights=self._transmission))
        return float(np.sqrt(abs(numerator / denominator)))

    @property
    def effective_frequency(self) -> float:
        """Pivot frequency in Hz, derived from :attr:`effective_wavelength`."""
        return const.c.cgs.value / self.effective_wavelength

    @property
    def filter_width_lambda(self) -> float:
        """Equivalent width of the filter in wavelength space (cm)."""
        return float(_trapz(self._transmission, self._wavelength))

    @property
    def filter_width_nu(self) -> float:
        """Equivalent width of the filter in frequency space (Hz)."""
        return float(abs(_trapz(self._transmission, self._frequency)))

    # ====================================================================== #
    # Dunders                                                                #
    # ====================================================================== #

    def __repr__(self) -> str:
        name_part = f"'{self._name}'" if self._name else "unnamed"
        lam_eff_aa = self.effective_wavelength * 1e8  # cm → Å
        return (
            f"PhotometryFilter({name_part}, "
            f"lam_eff={lam_eff_aa:.1f} AA, "
            f"weighting='{self._weighting}', "
            f"N={len(self._wavelength)})"
        )

    def __str__(self) -> str:
        return self.__repr__()

    def __len__(self) -> int:
        """Determine the number of grid points in the filter."""
        return len(self._wavelength)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PhotometryFilter):
            return NotImplemented
        return (
            np.array_equal(self._wavelength, other._wavelength)
            and np.array_equal(self._transmission, other._transmission)
            and self._weighting == other._weighting
        )

    def __call__(self, nu: np.ndarray, F_nu: np.ndarray) -> np.ndarray:
        """
        Convolve an SED with this filter (shorthand for :meth:`convolve_nu`).

        Parameters
        ----------
        nu : array-like
            Frequency grid in Hz.
        F_nu : array-like
            Flux density in erg/s/cm²/Hz. Shape ``(N_nu,)`` or ``(N_t, N_nu)``.

        Returns
        -------
        ndarray
            Band-averaged flux density.
        """
        return self.convolve_nu(nu, F_nu)

    # ====================================================================== #
    # Core functionality                                                     #
    # ====================================================================== #

    def apply(self, F_nu: np.ndarray) -> np.ndarray:
        """
        Apply filter weights to an SED already sampled on the filter grid.

        This is the fastest path — it assumes ``F_nu`` is evaluated exactly
        on :attr:`wavelength` (equivalently :attr:`frequency`). If your SED
        is on an arbitrary grid, use :meth:`convolve_nu` or
        :meth:`convolve_lambda` instead.

        Parameters
        ----------
        F_nu : array-like
            Flux density in erg/s/cm²/Hz. Last axis must have the same
            length as the filter grid. Shape ``(N_nu,)`` or ``(N_t, N_nu)``.

        Returns
        -------
        ndarray
            Band-averaged flux density. Scalar if 1-D input; ``(N_t,)``
            if 2-D input.

        Raises
        ------
        ValueError
            If the last dimension of ``F_nu`` does not match the filter grid.
        """
        F_nu = np.asarray(F_nu)
        if F_nu.shape[-1] != self._weights.shape[0]:
            raise ValueError(
                f"F_nu last dimension ({F_nu.shape[-1]}) must match filter grid size ({self._weights.shape[0]})."
            )
        return np.sum(F_nu * self._weights, axis=-1)

    def convolve_nu(self, nu: np.ndarray, F_nu: np.ndarray) -> np.ndarray:
        """
        Convolve an SED defined on an arbitrary frequency grid.

        Linearly interpolates ``F_nu`` onto the filter's frequency grid
        then calls :meth:`apply`.

        Parameters
        ----------
        nu : array-like
            Frequency grid in Hz. Need not be sorted.
        F_nu : array-like
            Flux density in erg/s/cm²/Hz. Shape ``(N_nu,)`` or
            ``(N_t, N_nu)``.

        Returns
        -------
        ndarray
            Band-averaged flux density. Scalar or ``(N_t,)``.
        """
        nu = np.asarray(nu, dtype=float)
        F_nu = np.asarray(F_nu, dtype=float)

        # np.interp requires ascending x
        sort_idx = np.argsort(nu)
        nu_sorted = nu[sort_idx]

        # Also sort the filter frequency grid for interp
        filt_nu = self._frequency
        filt_sort = np.argsort(filt_nu)
        filt_nu_sorted = filt_nu[filt_sort]

        if F_nu.ndim == 1:
            F_sorted = F_nu[sort_idx]
            F_interp_sorted = np.interp(filt_nu_sorted, nu_sorted, F_sorted, left=0.0, right=0.0)
            # Re-order to original filter grid ordering
            F_interp = np.empty_like(F_interp_sorted)
            F_interp[filt_sort] = F_interp_sorted
        else:
            F_sorted = F_nu[..., sort_idx]
            F_interp_sorted = np.apply_along_axis(
                lambda row: np.interp(filt_nu_sorted, nu_sorted, row, left=0.0, right=0.0),
                axis=-1,
                arr=F_sorted,
            )
            F_interp = np.empty_like(F_interp_sorted)
            F_interp[..., filt_sort] = F_interp_sorted

        return self.apply(F_interp)

    def convolve_lambda(self, wavelength: np.ndarray, F_lambda: np.ndarray) -> np.ndarray:
        """
        Convolve an SED defined in wavelength space (F_lambda).

        Converts ``F_lambda`` → ``F_nu`` using ``F_nu = (lambda^2/c)*F_lambda``,
        then delegates to :meth:`convolve_nu`.

        Parameters
        ----------
        wavelength : array-like
            Wavelength grid in cm.
        F_lambda : array-like
            Flux density in erg/s/cm²/cm. Shape ``(N_lambda,)`` or
            ``(N_t, N_lambda)``.

        Returns
        -------
        ndarray
            Band-averaged flux density in erg/s/cm²/Hz.
        """
        wavelength = np.asarray(wavelength, dtype=float)
        F_lambda = np.asarray(F_lambda, dtype=float)
        F_nu = (wavelength**2 / const.c.cgs.value) * F_lambda
        nu = const.c.cgs.value / wavelength
        return self.convolve_nu(nu, F_nu)

    # ====================================================================== #
    # Plotting                                                               #
    # ====================================================================== #

    def plot(
        self,
        fig: "Optional[matplotlib.figure.Figure]" = None,
        axes: "Optional[matplotlib.axes.Axes]" = None,
        wavelength_unit: str = "AA",
        **kwargs,
    ) -> tuple:
        """
        Plot the filter transmission curve.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            Existing figure. Creates a new one if ``None``.
        axes : matplotlib.axes.Axes, optional
            Existing axes. Creates new axes if ``None``.
        wavelength_unit : str, optional
            Astropy unit string for the x-axis (default ``"AA"`` = Ångström).
        **kwargs
            Additional keyword arguments forwarded to
            :func:`matplotlib.axes.Axes.plot`.

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : matplotlib.axes.Axes
        """
        from trilobite.utils.plot_utils import resolve_fig_axes, set_plot_style

        set_plot_style()
        fig, ax = resolve_fig_axes(fig, axes)

        scale = u.cm.to(wavelength_unit)
        ax.plot(self._wavelength * scale, self._transmission, **kwargs)
        ax.set_xlabel(f"Wavelength [{wavelength_unit}]")
        ax.set_ylabel("Transmission")
        if self._name:
            ax.set_title(self._name)
        ax.set_ylim(bottom=0.0)

        return fig, ax

    # ====================================================================== #
    # IO                                                                     #
    # ====================================================================== #

    def to_dict(self) -> dict:
        """
        Serialise the filter to a plain Python dictionary.

        Returns
        -------
        dict
            Keys: ``"name"``, ``"wavelength_cm"``, ``"transmission"``,
            ``"weighting"``.
        """
        return {
            "name": self._name,
            "wavelength_cm": self._wavelength.tolist(),
            "transmission": self._transmission.tolist(),
            "weighting": self._weighting,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PhotometryFilter":
        """
        Reconstruct a :class:`PhotometryFilter` from a dictionary.

        Parameters
        ----------
        d : dict
            Dictionary as produced by :meth:`to_dict`.

        Returns
        -------
        PhotometryFilter
        """
        return cls(
            wavelength=np.asarray(d["wavelength_cm"]),
            transmission=np.asarray(d["transmission"]),
            name=d.get("name") or None,
            weighting=d.get("weighting", "photon"),
        )

    def to_json(self, path: Union[str, Path]) -> None:
        """
        Write the filter to a JSON file.

        Parameters
        ----------
        path : str or Path
            Output file path.
        """
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "PhotometryFilter":
        """
        Load a :class:`PhotometryFilter` from a JSON file.

        Parameters
        ----------
        path : str or Path
            Path to a JSON file written by :meth:`to_json`.

        Returns
        -------
        PhotometryFilter
        """
        with open(path) as fh:
            d = json.load(fh)
        return cls.from_dict(d)

    def to_hdf5(self, path: Union[str, Path], key: str = "filter") -> None:
        """
        Write the filter to an HDF5 file.

        Parameters
        ----------
        path : str or Path
            Output HDF5 file path.
        key : str, optional
            HDF5 group name under which the filter is stored (default
            ``"filter"``).
        """
        with h5py.File(path, "a") as fh:
            if key in fh:
                del fh[key]
            grp = fh.create_group(key)
            grp.create_dataset("wavelength_cm", data=self._wavelength)
            grp.create_dataset("transmission", data=self._transmission)
            grp.attrs["name"] = self._name
            grp.attrs["weighting"] = self._weighting

    @classmethod
    def from_hdf5(cls, path: Union[str, Path], key: str = "filter") -> "PhotometryFilter":
        """
        Load a :class:`PhotometryFilter` from an HDF5 file.

        Parameters
        ----------
        path : str or Path
            Path to an HDF5 file written by :meth:`to_hdf5`.
        key : str, optional
            HDF5 group name (default ``"filter"``).

        Returns
        -------
        PhotometryFilter
        """
        with h5py.File(path, "r") as fh:
            grp = fh[key]
            wavelength = grp["wavelength_cm"][()]
            transmission = grp["transmission"][()]
            name = str(grp.attrs.get("name", "")) or None
            weighting = str(grp.attrs.get("weighting", "photon"))
        return cls(wavelength=wavelength, transmission=transmission, name=name, weighting=weighting)

    def to_table(self) -> Table:
        """
        Serialise the filter to an :class:`astropy.table.Table`.

        Returns
        -------
        astropy.table.Table
            Two columns: ``wavelength_cm`` and ``transmission``. Filter
            metadata (``name``, ``weighting``) are stored in ``table.meta``.
        """
        tbl = Table(
            {
                "wavelength_cm": self._wavelength,
                "transmission": self._transmission,
            }
        )
        tbl.meta["name"] = self._name
        tbl.meta["weighting"] = self._weighting
        return tbl

    @classmethod
    def from_table(
        cls,
        table: Table,
        name: Optional[str] = None,
        weighting: Optional[str] = None,
    ) -> "PhotometryFilter":
        """
        Reconstruct a :class:`PhotometryFilter` from an :class:`astropy.table.Table`.

        Parameters
        ----------
        table : astropy.table.Table
            Table with columns ``wavelength_cm`` and ``transmission``.
            Metadata keys ``name`` and ``weighting`` are used if present and
            the corresponding keyword arguments are not provided.
        name : str, optional
            Override the name stored in ``table.meta``.
        weighting : str, optional
            Override the weighting stored in ``table.meta``.

        Returns
        -------
        PhotometryFilter
        """
        resolved_name = name if name is not None else table.meta.get("name") or None
        resolved_weighting = weighting if weighting is not None else table.meta.get("weighting", "photon")
        return cls(
            wavelength=np.asarray(table["wavelength_cm"]),
            transmission=np.asarray(table["transmission"]),
            name=resolved_name,
            weighting=resolved_weighting,
        )

    def to_array(self) -> np.ndarray:
        """
        Serialise the filter to a 2-D NumPy array.

        Returns
        -------
        ndarray, shape (2, N)
            Row 0 is the wavelength grid in cm; row 1 is the transmission.
        """
        return np.vstack([self._wavelength, self._transmission])

    @classmethod
    def from_array(
        cls,
        arr: np.ndarray,
        name: Optional[str] = None,
        weighting: str = "photon",
    ) -> "PhotometryFilter":
        """
        Reconstruct a :class:`PhotometryFilter` from a 2-D NumPy array.

        Parameters
        ----------
        arr : ndarray, shape (2, N)
            Row 0 must be wavelength in cm; row 1 must be transmission.
        name : str, optional
            Filter name.
        weighting : str, optional
            ``"photon"`` (default) or ``"energy"``.

        Returns
        -------
        PhotometryFilter

        Raises
        ------
        ValueError
            If ``arr`` does not have shape ``(2, N)``.
        """
        arr = np.asarray(arr)
        if arr.ndim != 2 or arr.shape[0] != 2:
            raise ValueError(f"arr must have shape (2, N), got {arr.shape}.")
        return cls(wavelength=arr[0], transmission=arr[1], name=name, weighting=weighting)


# ============================================================================== #
# FilterBundle                                                                   #
# ============================================================================== #
class FilterBundle:
    """
    A collection of :class:`PhotometryFilter` objects optimised for batched convolution via matrix multiplication.

    At construction time all individual filter grids are merged into a single
    common frequency grid and a weight matrix ``W`` of shape
    ``(N_filters, N_common)`` is precomputed. The hot-loop method
    :meth:`apply` then evaluates all band-averaged fluxes simultaneously as
    a single matrix–vector or matrix–matrix product:

    .. code-block:: python

        F_bands = bundle.apply(F_nu_on_common_grid)
        # 1-D: W @ F_nu        → (N_filters,)
        # 2-D: F_nu @ W.T      → (N_t, N_filters)

    This avoids per-filter loops during MCMC sampling and is typically an
    order of magnitude faster than calling individual filters in a loop.

    Parameters
    ----------
    filters : dict[str, PhotometryFilter]
        Mapping of filter name to :class:`PhotometryFilter` instance. At
        least one entry is required.

    Raises
    ------
    ValueError
        If ``filters`` is empty.

    Examples
    --------

    .. code-block:: python

        import numpy as np, astropy.units as u
        from trilobite.utils.phot_utils import (
            PhotometryFilter,
            FilterBundle,
        )


        def gauss_filter(
            center_aa, width_aa, name, N=200
        ):
            lam = (
                np.linspace(
                    center_aa - 3 * width_aa,
                    center_aa + 3 * width_aa,
                    N,
                )
                * u.AA
            )
            T = np.exp(
                -0.5
                * ((lam.value - center_aa) / width_aa)
                ** 2
            )
            return PhotometryFilter(lam, T, name=name)


        bundle = FilterBundle(
            {
                "g": gauss_filter(4770, 600, "g"),
                "r": gauss_filter(6231, 600, "r"),
                "i": gauss_filter(7625, 600, "i"),
            }
        )
        nu = bundle.frequency_grid
        F_nu = (
            np.ones_like(nu) * 3.631e-20
        )  # 0-mag flat spectrum
        F_bands = bundle.apply(
            F_nu
        )  # shape (3,)  doctest: +SKIP
    """

    def __init__(self, filters: dict):
        if not filters:
            raise ValueError("FilterBundle requires at least one filter.")

        self._filters: dict = dict(filters)
        self._filter_names: list = list(filters.keys())
        self._filter_list: list = list(filters.values())

        self._build_common_grid_and_weights()

    # ------------------------------------------------------------------ #
    # Internal builder                                                    #
    # ------------------------------------------------------------------ #

    def _build_common_grid_and_weights(self) -> None:
        """Rebuild the common frequency grid and weight matrix."""
        # Union of all individual filter frequency grids, ascending order
        all_freqs = np.concatenate([f.frequency for f in self._filter_list])
        self._frequency_grid = np.sort(np.unique(all_freqs))

        n_common = len(self._frequency_grid)
        n_filters = len(self._filter_list)
        self._weight_matrix = np.zeros((n_filters, n_common))

        for i, filt in enumerate(self._filter_list):
            nu_lo, nu_hi = filt.frequency_bounds
            mask = (self._frequency_grid >= nu_lo) & (self._frequency_grid <= nu_hi)
            if not mask.any():
                continue

            # Sort filter's frequency to ascending for np.interp
            filt_nu = filt.frequency
            filt_weights = filt.weights
            sort_idx = np.argsort(filt_nu)
            filt_nu_asc = filt_nu[sort_idx]
            filt_w_asc = filt_weights[sort_idx]

            w_interp = np.interp(
                self._frequency_grid[mask],
                filt_nu_asc,
                filt_w_asc,
                left=0.0,
                right=0.0,
            )

            # Re-normalise so the row represents a proper weighted average
            row_sum = w_interp.sum()
            if row_sum > 0.0:
                w_interp /= row_sum

            self._weight_matrix[i, mask] = w_interp

    # ====================================================================== #
    # Properties                                                             #
    # ====================================================================== #

    @property
    def filters(self) -> dict:
        """Dictionary mapping filter name → :class:`PhotometryFilter`."""
        return dict(self._filters)

    @property
    def filter_names(self) -> list:
        """Ordered list of filter names."""
        return list(self._filter_names)

    @property
    def frequency_grid(self) -> np.ndarray:
        """Common frequency grid in Hz (ascending order)."""
        return self._frequency_grid

    @property
    def wavelength_grid(self) -> np.ndarray:
        """Common wavelength grid in cm (descending, = c/frequency_grid)."""
        return const.c.cgs.value / self._frequency_grid

    @property
    def weight_matrix(self) -> np.ndarray:
        """
        Precomputed weight matrix, shape ``(N_filters, N_common)``.

        Row ``i`` contains the normalised weights for ``filter_names[i]``
        interpolated onto the common frequency grid.
        """
        return self._weight_matrix

    @property
    def n_filters(self) -> int:
        """Number of filters in the bundle."""
        return len(self._filters)

    # ====================================================================== #
    # Dunders                                                                #
    # ====================================================================== #

    def __len__(self) -> int:
        return len(self._filters)

    def __repr__(self) -> str:
        names_str = ", ".join(f"'{n}'" for n in self._filter_names)
        return f"FilterBundle([{names_str}], N_common={len(self._frequency_grid)})"

    def __str__(self) -> str:
        return self.__repr__()

    def __getitem__(self, key: str) -> PhotometryFilter:
        return self._filters[key]

    def __iter__(self):
        return iter(self._filters)

    def __contains__(self, key: str) -> bool:
        return key in self._filters

    def __call__(self, nu: np.ndarray, F_nu: np.ndarray) -> np.ndarray:
        """
        Convolve an SED through all filters (shorthand for :meth:`convolve_nu`).

        Parameters
        ----------
        nu : array-like
            Frequency grid in Hz.
        F_nu : array-like
            Flux density in erg/s/cm²/Hz. Shape ``(N_nu,)`` or
            ``(N_t, N_nu)``.

        Returns
        -------
        ndarray
            Shape ``(N_filters,)`` or ``(N_t, N_filters)``.
        """
        return self.convolve_nu(nu, F_nu)

    # ====================================================================== #
    # Core functionality                                                     #
    # ====================================================================== #

    def apply(self, F_nu: np.ndarray) -> np.ndarray:
        """
        Apply all filter weights to an SED on the common frequency grid.

        This is the **hot-loop method** for MCMC: it assumes ``F_nu`` is
        already sampled on :attr:`frequency_grid` and performs a pure
        matrix multiply with no interpolation overhead.

        Parameters
        ----------
        F_nu : array-like
            Flux density in erg/s/cm²/Hz sampled on :attr:`frequency_grid`.
            Shape ``(N_common,)`` → returns ``(N_filters,)``.
            Shape ``(N_t, N_common)`` → returns ``(N_t, N_filters)``.

        Returns
        -------
        ndarray
            Band-averaged flux densities, one per filter.

        Raises
        ------
        ValueError
            If the last dimension of ``F_nu`` does not match the common grid.
        """
        F_nu = np.asarray(F_nu, dtype=float)
        if F_nu.shape[-1] != self._weight_matrix.shape[1]:
            raise ValueError(
                f"F_nu last dimension ({F_nu.shape[-1]}) must match the common grid size "
                f"({self._weight_matrix.shape[1]}). "
                "Use convolve_nu() for SEDs on arbitrary grids."
            )
        return F_nu @ self._weight_matrix.T

    def convolve_nu(self, nu: np.ndarray, F_nu: np.ndarray) -> np.ndarray:
        """
        Convolve an SED on an arbitrary frequency grid through all filters.

        Interpolates ``F_nu`` onto the common grid then calls :meth:`apply`.

        Parameters
        ----------
        nu : array-like
            Frequency grid in Hz. Need not be sorted or span all filters.
        F_nu : array-like
            Flux density in erg/s/cm²/Hz. Shape ``(N_nu,)`` or
            ``(N_t, N_nu)``.

        Returns
        -------
        ndarray
            Shape ``(N_filters,)`` or ``(N_t, N_filters)``.
        """
        nu = np.asarray(nu, dtype=float)
        F_nu = np.asarray(F_nu, dtype=float)

        sort_idx = np.argsort(nu)
        nu_sorted = nu[sort_idx]

        if F_nu.ndim == 1:
            F_sorted = F_nu[sort_idx]
            F_interp = np.interp(self._frequency_grid, nu_sorted, F_sorted, left=0.0, right=0.0)
        else:
            F_sorted = F_nu[..., sort_idx]
            F_interp = np.apply_along_axis(
                lambda row: np.interp(self._frequency_grid, nu_sorted, row, left=0.0, right=0.0),
                axis=-1,
                arr=F_sorted,
            )

        return self.apply(F_interp)

    def convolve_lambda(self, wavelength: np.ndarray, F_lambda: np.ndarray) -> np.ndarray:
        """
        Convolve an SED defined in wavelength space (F_lambda) through all filters.

        Converts ``F_lambda`` → ``F_nu`` then delegates to :meth:`convolve_nu`.

        Parameters
        ----------
        wavelength : array-like
            Wavelength grid in cm.
        F_lambda : array-like
            Flux density in erg/s/cm²/cm. Shape ``(N_lambda,)`` or
            ``(N_t, N_lambda)``.

        Returns
        -------
        ndarray
            Shape ``(N_filters,)`` or ``(N_t, N_filters)``.
        """
        wavelength = np.asarray(wavelength, dtype=float)
        F_lambda = np.asarray(F_lambda, dtype=float)
        F_nu = (wavelength**2 / const.c.cgs.value) * F_lambda
        nu = const.c.cgs.value / wavelength
        return self.convolve_nu(nu, F_nu)

    # ====================================================================== #
    # Mutation                                                               #
    # ====================================================================== #

    def add_filter(self, name: str, filt: PhotometryFilter) -> None:
        """
        Add a filter and rebuild the common grid and weight matrix.

        Parameters
        ----------
        name : str
            Key for the new filter. Must not already exist.
        filt : PhotometryFilter
            The filter to add.

        Raises
        ------
        KeyError
            If a filter with the same name already exists.
        """
        if name in self._filters:
            raise KeyError(f"A filter named '{name}' already exists. Remove it first.")
        self._filters[name] = filt
        self._filter_names.append(name)
        self._filter_list.append(filt)
        self._build_common_grid_and_weights()

    def remove_filter(self, name: str) -> PhotometryFilter:
        """
        Remove a filter and rebuild the common grid and weight matrix.

        Parameters
        ----------
        name : str
            Key of the filter to remove.

        Returns
        -------
        PhotometryFilter
            The removed filter.

        Raises
        ------
        KeyError
            If no filter with that name exists.
        ValueError
            If removing the filter would leave the bundle empty.
        """
        if name not in self._filters:
            raise KeyError(f"No filter named '{name}' in the bundle.")
        if len(self._filters) == 1:
            raise ValueError("Cannot remove the last filter from a FilterBundle.")
        filt = self._filters.pop(name)
        idx = self._filter_names.index(name)
        self._filter_names.pop(idx)
        self._filter_list.pop(idx)
        self._build_common_grid_and_weights()
        return filt

    # ====================================================================== #
    # Plotting                                                               #
    # ====================================================================== #

    def plot(
        self,
        fig: "Optional[matplotlib.figure.Figure]" = None,
        axes: "Optional[matplotlib.axes.Axes]" = None,
        wavelength_unit: str = "AA",
        legend: bool = True,
        **kwargs,
    ) -> tuple:
        """
        Plot all filter transmission curves on a single set of axes.

        Each filter is drawn in a distinct colour sampled from the default
        colourmap.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
        axes : matplotlib.axes.Axes, optional
        wavelength_unit : str, optional
            Astropy unit string for the x-axis (default ``"AA"``).
        legend : bool, optional
            Whether to add a legend (default ``True``).
        **kwargs
            Additional keyword arguments forwarded to
            :func:`matplotlib.axes.Axes.plot` for every filter curve.

        Returns
        -------
        fig : matplotlib.figure.Figure
        axes : matplotlib.axes.Axes
        """
        from trilobite.utils.plot_utils import get_default_cmap, resolve_fig_axes, set_plot_style

        set_plot_style()
        fig, ax = resolve_fig_axes(fig, axes)
        cmap = get_default_cmap()
        n = len(self._filters)
        colors = [cmap(i / max(n - 1, 1)) for i in range(n)]
        scale = u.cm.to(wavelength_unit)

        for color, (name, filt) in zip(colors, self._filters.items()):
            ax.plot(
                filt.wavelength * scale,
                filt.transmission,
                color=color,
                label=name,
                **kwargs,
            )

        ax.set_xlabel(f"Wavelength [{wavelength_unit}]")
        ax.set_ylabel("Transmission")
        ax.set_ylim(bottom=0.0)

        if legend:
            ax.legend()

        return fig, ax

    # ====================================================================== #
    # IO                                                                     #
    # ====================================================================== #

    def to_dict(self) -> dict:
        """
        Serialise the bundle to a plain Python dictionary.

        Returns
        -------
        dict
            ``{"filters": {name: filter_dict, ...}}``.
        """
        return {"filters": {name: filt.to_dict() for name, filt in self._filters.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "FilterBundle":
        """
        Reconstruct a :class:`FilterBundle` from a dictionary.

        Parameters
        ----------
        d : dict
            Dictionary as produced by :meth:`to_dict`.

        Returns
        -------
        FilterBundle
        """
        return cls({name: PhotometryFilter.from_dict(fd) for name, fd in d["filters"].items()})

    def to_json(self, path: Union[str, Path]) -> None:
        """
        Write the bundle to a JSON file.

        Parameters
        ----------
        path : str or Path
            Output file path.
        """
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "FilterBundle":
        """
        Load a :class:`FilterBundle` from a JSON file.

        Parameters
        ----------
        path : str or Path
            Path to a JSON file written by :meth:`to_json`.

        Returns
        -------
        FilterBundle
        """
        with open(path) as fh:
            d = json.load(fh)
        return cls.from_dict(d)

    def to_hdf5(self, path: Union[str, Path]) -> None:
        """
        Write the bundle to an HDF5 file.

        Each filter is stored as a separate HDF5 group. The original filter
        order is preserved via a ``filter_names`` root attribute.

        Parameters
        ----------
        path : str or Path
            Output HDF5 file path.
        """
        with h5py.File(path, "w") as fh:
            fh.attrs["filter_names"] = json.dumps(self._filter_names)
            for name, filt in self._filters.items():
                grp = fh.create_group(name)
                grp.create_dataset("wavelength_cm", data=filt.wavelength)
                grp.create_dataset("transmission", data=filt.transmission)
                grp.attrs["name"] = filt.name
                grp.attrs["weighting"] = filt.weighting

    @classmethod
    def from_hdf5(cls, path: Union[str, Path]) -> "FilterBundle":
        """
        Load a :class:`FilterBundle` from an HDF5 file.

        Parameters
        ----------
        path : str or Path
            Path to an HDF5 file written by :meth:`to_hdf5`.

        Returns
        -------
        FilterBundle
        """
        filters: dict = {}
        with h5py.File(path, "r") as fh:
            if "filter_names" in fh.attrs:
                names = json.loads(str(fh.attrs["filter_names"]))
            else:
                names = list(fh.keys())
            for name in names:
                grp = fh[name]
                wavelength = grp["wavelength_cm"][()]
                transmission = grp["transmission"][()]
                filt_name = str(grp.attrs.get("name", "")) or None
                weighting = str(grp.attrs.get("weighting", "photon"))
                filters[name] = PhotometryFilter(
                    wavelength=wavelength,
                    transmission=transmission,
                    name=filt_name,
                    weighting=weighting,
                )
        return cls(filters)


# ============================================================================== #
# Magnitude Systems and Conversions                                              #
# ============================================================================== #


def flux_to_ab_mag(F_nu: np.ndarray) -> np.ndarray:
    r"""
    Convert flux density to AB magnitude.

    The AB system is defined such that a constant spectrum with
    ``F_nu = 3.631e-20 erg/s/cm²/Hz`` has magnitude zero at all frequencies.

    Parameters
    ----------
    F_nu : array-like
        Flux density in erg/s/cm²/Hz. Non-positive values silently return
        ``np.inf``.

    Returns
    -------
    ndarray
        AB magnitudes. Same shape as input.

    Notes
    -----
    .. math::

        m_{rm AB} = -2.5 log_{10}!left(frac{F_nu}{3.631
        times 10^{-20},{rm erg,s^{-1},cm^{-2},Hz^{-1}}}right)

    Examples
    --------

    .. code-block:: python

        flux_to_ab_mag(3.631e-20)
        0.0
        flux_to_ab_mag(0.0)
        inf
    """
    F_nu = np.asarray(F_nu, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = -2.5 * np.log10(F_nu / _AB_ZERO_POINT_CGS)
    return mag


def ab_mag_to_flux(mag: np.ndarray) -> np.ndarray:
    """
    Convert AB magnitude to flux density.

    Parameters
    ----------
    mag : array-like
        AB magnitudes.

    Returns
    -------
    ndarray
        Flux density in erg/s/cm²/Hz. Same shape as input.

    Examples
    --------
    .. code-block:: python

        ab_mag_to_flux(0.0)
        3.631e-20
    """
    return _AB_ZERO_POINT_CGS * 10.0 ** (-0.4 * np.asarray(mag, dtype=float))


def flux_lambda_to_st_mag(F_lambda: np.ndarray) -> np.ndarray:
    r"""
    Convert F_lambda flux density to ST magnitude.

    The ST system is defined by ``m_ST = -2.5 log10(F_lambda) - 21.1``
    where ``F_lambda`` is in erg/s/cm²/Å.

    Parameters
    ----------
    F_lambda : array-like
        Flux density per unit wavelength in erg/s/cm²/Å. Non-positive values
        silently return ``np.inf``.

    Returns
    -------
    ndarray
        ST magnitudes. Same shape as input.

    Notes
    -----
    .. math::

        m_{rm ST} = -2.5 log_{10}(F_lambda) - 21.1

    where ``F_lambda`` is in erg/s/cm²/Å.
    """
    F_lambda = np.asarray(F_lambda, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = -2.5 * np.log10(F_lambda) - _ST_ZERO_POINT_OFFSET
    return mag


def st_mag_to_flux_lambda(mag: np.ndarray) -> np.ndarray:
    """
    Convert ST magnitude to F_lambda flux density.

    Parameters
    ----------
    mag : array-like
        ST magnitudes.

    Returns
    -------
    ndarray
        Flux density in erg/s/cm²/Å. Same shape as input.
    """
    return 10.0 ** (-0.4 * (np.asarray(mag, dtype=float) + _ST_ZERO_POINT_OFFSET))


def filter_to_ab_mag(
    filt: Union[PhotometryFilter, FilterBundle],
    nu: np.ndarray,
    F_nu: np.ndarray,
) -> np.ndarray:
    """
    Compute AB magnitude(s) from filter convolution of an SED.

    Convolves ``F_nu`` through ``filt`` then converts the resulting
    band-averaged flux density to AB magnitudes.

    Parameters
    ----------
    filt : PhotometryFilter or FilterBundle
        Filter or bundle of filters.
    nu : array-like
        Frequency grid in Hz.
    F_nu : array-like
        Flux density in erg/s/cm²/Hz. Shape ``(N_nu,)`` or ``(N_t, N_nu)``.

    Returns
    -------
    ndarray
        AB magnitudes. Shape:

        - :class:`PhotometryFilter` + 1-D → scalar
        - :class:`PhotometryFilter` + 2-D → ``(N_t,)``
        - :class:`FilterBundle` + 1-D → ``(N_filters,)``
        - :class:`FilterBundle` + 2-D → ``(N_t, N_filters)``
    """
    F_band = filt.convolve_nu(nu, F_nu)
    return flux_to_ab_mag(F_band)


# ============================================================================== #
# Filter Loading Helpers                                                         #
# ============================================================================== #


def load_filter_from_file(
    file_path: Union[str, Path],
    name: Optional[str] = None,
    weighting: str = "photon",
    wavelength_unit: str = "AA",
) -> PhotometryFilter:
    """
    Load a photometric filter from a two-column ASCII file.

    Parameters
    ----------
    file_path : str or Path
        Path to the file. Must contain two whitespace-separated columns:
        wavelength and transmission.
    name : str, optional
        Filter name. Defaults to the file stem if not provided.
    weighting : str, optional
        ``"photon"`` (default) or ``"energy"``.
    wavelength_unit : str, optional
        Astropy unit string for the wavelength column (default ``"AA"``).

    Returns
    -------
    PhotometryFilter
    """
    file_path = Path(file_path)
    data = np.loadtxt(file_path)
    wavelength = data[:, 0] * u.Unit(wavelength_unit)
    transmission = data[:, 1]
    if name is None:
        name = file_path.stem
    return PhotometryFilter(wavelength, transmission, name=name, weighting=weighting)


def load_filter_from_speclite(name: str, weighting: str = "photon") -> PhotometryFilter:
    """
    Load a photometric filter from the ``speclite`` filter registry.

    Requires the optional ``speclite`` package::

        pip install trilobite[optical]

    Parameters
    ----------
    name : str
        Filter name in speclite notation, e.g. ``"sdss-r"``, ``"ztf-r"``,
        ``"lsst2016-r"``. Call :func:`list_speclite_filters` to enumerate
        available filters.
    weighting : str, optional
        ``"photon"`` (default) or ``"energy"``.

    Returns
    -------
    PhotometryFilter

    Raises
    ------
    ImportError
        If ``speclite`` is not installed.
    """
    try:
        import speclite.filters as sf
    except ImportError:
        raise ImportError(
            "speclite is required to load filters by name. Install it with: pip install trilobite[optical]"
        ) from None

    filt = sf.load_filter(name)
    wavelength = np.asarray(filt.wavelength) * u.Angstrom
    transmission = np.asarray(filt.response, dtype=float)
    # Clip NaN / negative values that occasionally appear at filter edges
    transmission = np.where(np.isfinite(transmission) & (transmission >= 0.0), transmission, 0.0)
    return PhotometryFilter(wavelength, transmission, name=name, weighting=weighting)


def load_filter_from_name(name: str, weighting: str = "photon") -> PhotometryFilter:
    """
    Load a photometric filter by name from the ``speclite`` registry.

    This is an alias for :func:`load_filter_from_speclite` and exists for
    backward compatibility.

    Parameters
    ----------
    name : str
        Filter name (e.g. ``"sdss-r"``).
    weighting : str, optional
        ``"photon"`` (default) or ``"energy"``.

    Returns
    -------
    PhotometryFilter
    """
    return load_filter_from_speclite(name, weighting=weighting)


def list_speclite_filters(group: Optional[str] = None) -> list:
    """
    List filter names available in the ``speclite`` registry.

    Parameters
    ----------
    group : str, optional
        If provided, only filters whose name starts with ``group`` are
        returned (e.g. ``"sdss"`` to list all SDSS filters).

    Returns
    -------
    list of str
        Sorted list of filter names.

    Raises
    ------
    ImportError
        If ``speclite`` is not installed.
    """
    try:
        import speclite.filters as sf
    except ImportError:
        raise ImportError("speclite is required. Install it with: pip install trilobite[optical]") from None

    # speclite >=0.17 exposes filter_names(); older builds may not.
    try:
        names = list(sf.filter_names())
    except AttributeError:
        # Fallback: speclite stores built-in data files; we can't enumerate
        # them portably without knowing the version, so return an empty list.
        names = []

    if group is not None:
        names = [n for n in names if n.startswith(group)]
    return sorted(names)


def load_filter_from_svo(
    filter_id: str,
    weighting: str = "photon",
    cache: bool = True,
    timeout: int = 60,
) -> PhotometryFilter:
    """
    Load a photometric filter from the SVO Filter Profile Service.

    Fetches the transmission curve for *filter_id* from the Spanish Virtual
    Observatory (SVO) Filter Profile Service via ``astroquery.svo_fps``.
    Results are cached locally by astroquery for one week, so repeated calls
    for the same filter do not require a network round-trip.

    Requires the optional ``astroquery`` package::

        pip install trilobite[optical]

    Parameters
    ----------
    filter_id : str
        SVO filter identifier in ``facility/instrument.band`` notation, e.g.
        ``"Kepler/Kepler.K"``, ``"2MASS/2MASS.H"``, ``"Generic/Johnson.R"``.
        Browse available IDs at
        https://svo2.cab.inta-csic.es/svo/theory/fps/.
    weighting : str, optional
        ``"photon"`` (default, appropriate for CCDs) or ``"energy"``.
    cache : bool, optional
        Whether to use astroquery's on-disk cache (default ``True``).
    timeout : int, optional
        HTTP request timeout in seconds (default 60).

    Returns
    -------
    PhotometryFilter

    Raises
    ------
    ImportError
        If ``astroquery`` is not installed.
    ValueError
        If *filter_id* is not found in the SVO catalogue.
    """
    try:
        from astroquery.svo_fps import SvoFps
    except ImportError:
        raise ImportError(
            "astroquery is required to load SVO filters. Install it with: pip install trilobite[optical]"
        ) from None

    table = SvoFps.get_transmission_data(filter_id, cache=cache, timeout=timeout)
    if len(table) == 0:
        raise ValueError(
            f"Filter '{filter_id}' was not found in the SVO Filter Profile Service. "
            "Check the filter ID at https://svo2.cab.inta-csic.es/svo/theory/fps/."
        )

    wavelength = np.asarray(table["Wavelength"], dtype=float) * u.Angstrom
    transmission = np.asarray(table["Transmission"], dtype=float)
    transmission = np.where(np.isfinite(transmission) & (transmission >= 0.0), transmission, 0.0)
    return PhotometryFilter(wavelength, transmission, name=filter_id, weighting=weighting)


def list_svo_filters(
    facility: str,
    instrument: Optional[str] = None,
    cache: bool = True,
    timeout: int = 60,
) -> Table:
    """
    List filters available in the SVO Filter Profile Service for a given facility.

    Returns the full metadata table so callers can inspect effective
    wavelengths, zero-points, and FWHM values before deciding which filters
    to load.

    Requires the optional ``astroquery`` package::

        pip install trilobite[optical]

    Parameters
    ----------
    facility : str
        Telescope or observatory name as registered in SVO, e.g. ``"Kepler"``,
        ``"HST"``, ``"SDSS"``, ``"2MASS"``.
    instrument : str, optional
        Restrict results to a specific instrument (e.g. ``"WFC3_IR"``).
        If ``None`` (default) all instruments for the facility are returned.
    cache : bool, optional
        Whether to use astroquery's on-disk cache (default ``True``).
    timeout : int, optional
        HTTP request timeout in seconds (default 60).

    Returns
    -------
    astropy.table.Table
        Table with columns including ``filterID``, ``WavelengthEff``,
        ``WavelengthMin``, ``WavelengthMax``, ``FWHM``, ``ZeroPoint``,
        ``MagSys``, and others. Pass ``table["filterID"]`` to
        :func:`load_filter_from_svo` to fetch individual transmission curves.

    Raises
    ------
    ImportError
        If ``astroquery`` is not installed.
    """
    try:
        from astroquery.svo_fps import SvoFps
    except ImportError:
        raise ImportError(
            "astroquery is required to query the SVO. Install it with: pip install trilobite[optical]"
        ) from None

    table = SvoFps.get_filter_list(facility, cache=cache, timeout=timeout)

    if instrument is not None:
        # The SVO server does not reliably filter by instrument via URL params,
        # so we filter client-side on the filterID column ("facility/instrument.band").
        mask = np.array([f"/{instrument}." in str(fid) for fid in table["filterID"]])
        table = table[mask]

    return table


def load_filters_from_svo(
    facility: str,
    instrument: Optional[str] = None,
    weighting: str = "photon",
    cache: bool = True,
    timeout: int = 60,
) -> dict:
    """
    Load all SVO filters for a facility as a dict ready for :class:`FilterBundle`.

    Convenience wrapper around :func:`list_svo_filters` and
    :func:`load_filter_from_svo`. Each filter's transmission curve is fetched
    in a separate astroquery call; results are cached locally after the first
    request.

    Requires the optional ``astroquery`` package::

        pip install trilobite[optical]

    Parameters
    ----------
    facility : str
        Telescope or observatory name (e.g. ``"Kepler"``, ``"2MASS"``).
    instrument : str, optional
        Restrict to a specific instrument. If ``None`` all instruments are
        included.
    weighting : str, optional
        ``"photon"`` (default) or ``"energy"``.
    cache : bool, optional
        Whether to use astroquery's on-disk cache (default ``True``).
    timeout : int, optional
        HTTP request timeout in seconds per request (default 60).

    Returns
    -------
    dict[str, PhotometryFilter]
        Mapping of SVO ``filterID`` strings to :class:`PhotometryFilter`
        objects. Pass directly to ``FilterBundle(filters)`` to build a bundle.

    Raises
    ------
    ImportError
        If ``astroquery`` is not installed.

    Examples
    --------
    >>> filters = load_filters_from_svo("Kepler")
    >>> from trilobite.utils.phot_utils import (
    ...     FilterBundle,
    ... )
    >>> bundle = FilterBundle(filters)
    """
    from trilobite.utils.log import trilobite_logger

    index = list_svo_filters(facility, instrument=instrument, cache=cache, timeout=timeout)
    filter_ids = list(index["filterID"])
    trilobite_logger.debug("Loading %d SVO filters for facility '%s'", len(filter_ids), facility)

    result: dict = {}
    for fid in filter_ids:
        try:
            result[fid] = load_filter_from_svo(fid, weighting=weighting, cache=cache, timeout=timeout)
        except ValueError:
            trilobite_logger.warning("SVO filter '%s' has no transmission data; skipping.", fid)
    return result
