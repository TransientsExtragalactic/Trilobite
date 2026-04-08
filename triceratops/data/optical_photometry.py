"""
Data container for optical photometric observations.

This module provides :class:`OpticalPhotometryContainer`, a schema-validated,
unit-aware container for optical photometry from surveys such as ZTF, LSST/Rubin,
and DECam. Observations may be supplied as AB magnitudes, physical flux densities
(F_ν in erg/s/cm²/Hz), or a mix of both. The container exposes either
representation on-the-fly via properties and always converts to physical flux units
when building an :class:`~triceratops.data.core.InferenceData` object for likelihood
evaluation.
"""

import warnings
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from astropy import units as u
from astropy.table import Table
from astropy.time import Time

from triceratops.utils.log import triceratops_logger
from triceratops.utils.phot_utils import ab_mag_to_flux, flux_to_ab_mag

from .core import DataContainer, InferenceData

if TYPE_CHECKING:
    from triceratops.models.core.base import Model  # noqa: F401


__all__ = ["OpticalPhotometryContainer"]


class OpticalPhotometryContainer(DataContainer):
    r"""
    Immutable container for optical photometric observations.

    Observations may be expressed as **AB magnitudes**, **flux densities**
    (F_ν, erg/s/cm²/Hz), or both.  The container stores whichever columns are
    present and provides :attr:`flux`, :attr:`mag`, and related properties that
    compute the complementary representation on-the-fly when it is absent from
    the underlying table.

    Detections are identified by a ``NaN`` value in whichever upper-limit column
    is present (``flux_upper_limit`` takes priority over ``mag_ab_upper_limit``).

    Band identification uses human-readable **band names** (e.g. ``"g"``, ``"r"``)
    rather than integer indices.  Integer indices are resolved at
    :meth:`to_inference_data` time by looking up each name in the optical model's
    :attr:`FilterBundle.filter_names` list.  This keeps the container independent
    of any particular model or filter ordering convention.

    Schema
    ------
    Required columns:

    - ``time`` [day] — relative time of each observation.
    - ``band_name`` [str] — survey band identifier.

    At least one of the following y-column groups must be present:

    - flux group: ``flux_density`` [erg/(s cm² Hz)],
      ``flux_density_error`` [erg/(s cm² Hz)],
      ``flux_upper_limit`` [erg/(s cm² Hz)]
    - magnitude group: ``mag_ab``, ``mag_ab_error``,
      ``mag_ab_upper_limit`` (dimensionless)

    Optional columns:

    - ``obs_name`` [str]
    - ``epoch_id`` [int]

    Notes
    -----
    - All properties return :class:`astropy.units.Quantity` objects for physical
      columns and plain NumPy arrays for dimensionless or string columns.
    - Magnitude → flux conversion uses the AB zero-point
      :math:`F_0 = 3.631 \times 10^{-20}` erg/s/cm²/Hz.
    - Flux error → magnitude error: :math:`\sigma_m = 1.0857 \cdot \sigma_F / F`
    - Magnitude error → flux error: :math:`\sigma_F = F \cdot \sigma_m / 1.0857`

    See Also
    --------
    triceratops.utils.phot_utils.FilterBundle
        The filter bundle whose ``filter_names`` list defines the band→index mapping.
    triceratops.data.photometry.RadioPhotometryContainer
        Equivalent container for radio photometry.
    """

    # ========================= SCHEMA DEFINITION ========================= #
    COLUMNS = [
        # Time axis — always required
        {
            "name": "time",
            "dtype": float,
            "unit": "day",
            "description": "Relative time of each observation.",
            "required": True,
        },
        # Band identification — band_name is the canonical identifier;
        # band_idx is intentionally absent and only appears in InferenceData.x
        {
            "name": "band_name",
            "dtype": str,
            "description": "Survey band identifier (e.g. 'g', 'r', 'i').",
            "required": True,
        },
        # Flux representation (F_nu in erg/s/cm²/Hz)
        {
            "name": "flux_density",
            "dtype": float,
            "unit": "erg/(s cm2 Hz)",
            "description": "Measured flux density F_nu for detections.",
            "required": False,
        },
        {
            "name": "flux_density_error",
            "dtype": float,
            "unit": "erg/(s cm2 Hz)",
            "description": "1-sigma uncertainty on flux_density.",
            "required": False,
        },
        {
            "name": "flux_upper_limit",
            "dtype": float,
            "unit": "erg/(s cm2 Hz)",
            "description": "Upper limit on flux density for non-detections.",
            "required": False,
        },
        # Magnitude representation (dimensionless AB)
        {
            "name": "mag_ab",
            "dtype": float,
            "description": "AB magnitude for detections.",
            "required": False,
        },
        {
            "name": "mag_ab_error",
            "dtype": float,
            "description": "1-sigma uncertainty on mag_ab.",
            "required": False,
        },
        {
            "name": "mag_ab_upper_limit",
            "dtype": float,
            "description": "AB magnitude upper limit for non-detections.",
            "required": False,
        },
        # Optional metadata
        {
            "name": "obs_name",
            "dtype": str,
            "description": "Observation identifier.",
            "required": False,
        },
        {
            "name": "epoch_id",
            "dtype": int,
            "description": "Integer epoch identifier for grouping observations.",
            "required": False,
        },
    ]

    # ========================= Initialization ========================= #
    def __init__(self, table: Table):
        """
        Instantiate the container, validate the schema, and build detection masks.

        Parameters
        ----------
        table : astropy.table.Table
            Input table conforming to the :attr:`COLUMNS` schema.
        """
        super().__init__(table)

        # Determine the upper-limit column to use for detection logic.
        # flux_upper_limit takes priority; fall back to mag_ab_upper_limit.
        if "flux_upper_limit" in self.__table__.colnames:
            upper_col = np.asarray(self.__table__["flux_upper_limit"].data, dtype=float)
        elif "mag_ab_upper_limit" in self.__table__.colnames:
            upper_col = np.asarray(self.__table__["mag_ab_upper_limit"].data, dtype=float)
        else:
            # No upper-limit column: treat all rows as detections
            upper_col = np.full(len(self.__table__), np.nan)

        self.__detection_mask__: np.ndarray = np.isnan(upper_col)
        self.__non_detection_mask__: np.ndarray = ~self.__detection_mask__

        # Epoch support (same pattern as RadioPhotometryContainer)
        self.__epoch_ids__: Optional[np.ndarray] = None
        if "epoch_id" in self.__table__.colnames:
            self.__epoch_ids__ = np.asarray(self.__table__["epoch_id"].data, dtype=int)

    # ========================= Schema Validation ========================= #
    def _validate_table(self, table: Table) -> Table:
        """
        Validate schema and enforce optical-specific invariants.

        Extends the base :meth:`DataContainer._validate_table` with checks that
        ensure at least one y-column group (flux or magnitude) is present.
        """
        # Run base schema validation (dtype coercion + unit checking)
        table = super()._validate_table(table)

        # At least one y-column group must be present
        has_flux = "flux_density" in table.colnames
        has_mag = "mag_ab" in table.colnames

        if not has_flux and not has_mag:
            raise ValueError(
                "OpticalPhotometryContainer requires at least one of 'flux_density' "
                "or 'mag_ab' to be present in the input table."
            )

        # Warn about incomplete pairs
        if has_flux and "flux_density_error" not in table.colnames:
            triceratops_logger.warning("Column 'flux_density' present but 'flux_density_error' is missing.")
        if has_mag and "mag_ab_error" not in table.colnames:
            triceratops_logger.warning("Column 'mag_ab' present but 'mag_ab_error' is missing.")

        return table

    # ========================= Core Properties ========================= #
    @property
    def n_obs(self) -> int:
        """Return the total number of observations."""
        return len(self.__table__)

    @property
    def time(self) -> u.Quantity:
        """Observation times as an Astropy Quantity (days)."""
        return self.__table__["time"].quantity

    @property
    def band_name(self) -> np.ndarray:
        """Survey band names (string array)."""
        return np.asarray(self.__table__["band_name"].data)

    # ========================= Detection Properties ========================= #
    @property
    def detection_mask(self) -> np.ndarray:
        """Boolean mask selecting detections (rows where upper limit is NaN)."""
        return self.__detection_mask__.copy()

    @property
    def non_detection_mask(self) -> np.ndarray:
        """Boolean mask selecting non-detections / upper limits."""
        return self.__non_detection_mask__.copy()

    @property
    def n_detections(self) -> int:
        """Number of detections."""
        return int(self.__detection_mask__.sum())

    @property
    def n_non_detections(self) -> int:
        """Number of non-detections (upper limits)."""
        return int(self.__non_detection_mask__.sum())

    @property
    def detections(self) -> "OpticalPhotometryContainer":
        """Return a new container containing only detections."""
        return self.apply_mask(self.__detection_mask__)

    @property
    def upper_limits(self) -> "OpticalPhotometryContainer":
        """Return a new container containing only non-detections (upper limits)."""
        return self.apply_mask(self.__non_detection_mask__)

    # ========================= Flux Properties ========================= #
    @property
    def _has_flux_column(self) -> bool:
        return "flux_density" in self.__table__.colnames

    @property
    def _has_mag_column(self) -> bool:
        return "mag_ab" in self.__table__.colnames

    @property
    def flux(self) -> u.Quantity:
        """
        Flux density F_ν in erg/s/cm²/Hz.

        Returns the ``flux_density`` column if present.  Otherwise converts
        ``mag_ab`` to flux using the AB zero-point.
        """
        if self._has_flux_column:
            return self.__table__["flux_density"].quantity
        mag_vals = np.asarray(self.__table__["mag_ab"].data, dtype=float)
        return ab_mag_to_flux(mag_vals) * u.Unit("erg/(s cm2 Hz)")

    @property
    def flux_error(self) -> u.Quantity:
        r"""
        1-sigma uncertainty on F_ν in erg/s/cm²/Hz.

        Returns ``flux_density_error`` if present.  Otherwise propagates from
        ``mag_ab_error`` using :math:`\sigma_F = F \cdot \sigma_m / 1.0857`.
        """
        if self._has_flux_column and "flux_density_error" in self.__table__.colnames:
            return self.__table__["flux_density_error"].quantity
        mag_err = np.asarray(self.__table__["mag_ab_error"].data, dtype=float)
        return self.flux * (mag_err / 1.0857)

    @property
    def flux_upper_limit(self) -> u.Quantity:
        """
        Upper limit on F_ν in erg/s/cm²/Hz (NaN for detections).

        Returns ``flux_upper_limit`` if present.  Otherwise converts
        ``mag_ab_upper_limit`` to flux.
        """
        if "flux_upper_limit" in self.__table__.colnames:
            return self.__table__["flux_upper_limit"].quantity
        if "mag_ab_upper_limit" in self.__table__.colnames:
            mag_ul = np.asarray(self.__table__["mag_ab_upper_limit"].data, dtype=float)
            return ab_mag_to_flux(mag_ul) * u.Unit("erg/(s cm2 Hz)")
        # No upper-limit column present — return NaN array
        return np.full(self.n_obs, np.nan) * u.Unit("erg/(s cm2 Hz)")

    # ========================= Magnitude Properties ========================= #
    @property
    def mag(self) -> np.ndarray:
        """
        AB magnitude (dimensionless array).

        Returns ``mag_ab`` if present.  Otherwise converts ``flux_density``
        using the AB zero-point.
        """
        if self._has_mag_column:
            return np.asarray(self.__table__["mag_ab"].data, dtype=float)
        flux_vals = self.flux.value
        return flux_to_ab_mag(flux_vals)

    @property
    def mag_error(self) -> np.ndarray:
        r"""
        1-sigma uncertainty on AB magnitude.

        Returns ``mag_ab_error`` if present.  Otherwise propagates from
        ``flux_density_error`` using :math:`\sigma_m = 1.0857 \cdot \sigma_F / F`.
        """
        if self._has_mag_column and "mag_ab_error" in self.__table__.colnames:
            return np.asarray(self.__table__["mag_ab_error"].data, dtype=float)
        flux_err = self.flux_error.value
        flux_vals = self.flux.value
        return 1.0857 * flux_err / flux_vals

    @property
    def mag_upper_limit(self) -> np.ndarray:
        """
        AB magnitude upper limit (NaN for detections).

        Returns ``mag_ab_upper_limit`` if present.  Otherwise converts
        ``flux_upper_limit`` to magnitudes.
        """
        if "mag_ab_upper_limit" in self.__table__.colnames:
            return np.asarray(self.__table__["mag_ab_upper_limit"].data, dtype=float)
        return flux_to_ab_mag(self.flux_upper_limit.value)

    # ========================= Optional Metadata Properties ========================= #
    @property
    def obs_name(self) -> np.ndarray:
        """Observation names (raises if column absent)."""
        if "obs_name" not in self.__table__.colnames:
            raise AttributeError("Column 'obs_name' not found in the table.")
        return np.asarray(self.__table__["obs_name"].data)

    # ========================= Epoch Support ========================= #
    @property
    def has_epochs(self) -> bool:
        """Return True if epoch IDs have been assigned."""
        return self.__epoch_ids__ is not None

    @property
    def epoch_ids(self) -> np.ndarray:
        """Return a copy of the epoch ID array (raises if epochs not set)."""
        if self.__epoch_ids__ is None:
            raise AttributeError("No epochs defined.")
        return self.__epoch_ids__.copy()

    @property
    def n_epochs(self) -> Optional[int]:
        """Number of unique epochs, or None if epochs are not defined."""
        if self.__epoch_ids__ is None:
            return None
        return int(len(np.unique(self.__epoch_ids__)))

    def get_epoch_mask(self, epoch: int) -> np.ndarray:
        """Return a boolean mask for observations in *epoch*."""
        if self.__epoch_ids__ is None:
            raise AttributeError("No epochs defined.")
        return self.__epoch_ids__ == epoch

    def set_epochs_from_indices(self, indices) -> None:
        """
        Assign epoch IDs explicitly, one integer per observation.

        Parameters
        ----------
        indices : array-like of int
            Epoch identifier for each observation.  Need not start at zero or
            be contiguous; they are remapped to ``[0, n_epochs-1]`` internally.
        """
        indices = np.asarray(indices)
        if indices.shape != (self.n_obs,):
            raise ValueError("Epoch indices must have shape (n_obs,).")
        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("Epoch indices must be integers.")
        unique = np.unique(indices)
        remap = {old: new for new, old in enumerate(unique)}
        self.__epoch_ids__ = np.vectorize(remap.get)(indices)

    def set_epochs_from_time_gaps(self, max_gap: u.Quantity) -> None:
        """
        Group observations into epochs separated by time gaps larger than *max_gap*.

        Parameters
        ----------
        max_gap : astropy.units.Quantity
            Maximum time gap (must be convertible to days) within an epoch.
        """
        times = self.time.to(max_gap.unit).value
        order = np.argsort(times)
        epoch_ids = np.zeros(len(times), dtype=int)
        current = 0
        for i in range(1, len(order)):
            if times[order[i]] - times[order[i - 1]] > max_gap.value:
                current += 1
            epoch_ids[order[i]] = current
        self.__epoch_ids__ = epoch_ids

    # ========================= Masking ========================= #
    def apply_mask(self, mask) -> "OpticalPhotometryContainer":
        """
        Return a new container containing only the rows selected by *mask*.

        Parameters
        ----------
        mask : array-like of bool or int
            Boolean or index mask of the same length as this container.

        Returns
        -------
        OpticalPhotometryContainer
        """
        sliced = self.table[mask]
        if self.has_epochs and "epoch_id" not in sliced.colnames:
            sliced["epoch_id"] = self.__epoch_ids__[mask]
        return self.__class__.from_table(sliced)

    # ========================= IO Methods ========================= #
    @classmethod
    def from_table(
        cls,
        table: Table,
        column_map: Optional[dict] = None,
        time_start: Optional[Union[u.Quantity, float]] = None,
        internal_time_format: str = "relative",
        internal_time_scale: str = "utc",
    ) -> "OpticalPhotometryContainer":
        """
        Construct an :class:`OpticalPhotometryContainer` from an Astropy Table.

        Parameters
        ----------
        table : astropy.table.Table
            Input table containing optical photometry data.

        column_map : dict, optional
            Mapping from existing column names → expected schema column names.
            Use this to rename non-standard column names before validation.

        time_start : float or astropy.units.Quantity, optional
            Reference time subtracted from the ``time`` column to convert absolute
            times to relative times.  When ``internal_time_format="relative"``
            this is interpreted as days (float) or an Astropy Quantity.

        internal_time_format : str, default ``"relative"``
            Format of the ``time`` column in the input table.

            - ``"relative"`` — time is already elapsed days since some reference.
            - Any valid Astropy time format (e.g. ``"mjd"``, ``"jd"``,
              ``"iso"``, ``"unix"``).

        internal_time_scale : str, default ``"utc"``
            Astropy time scale used when interpreting absolute time formats.
            Ignored when ``internal_time_format="relative"``.

        Returns
        -------
        OpticalPhotometryContainer

        Examples
        --------
        From relative times:

        .. code-block:: python

            c = OpticalPhotometryContainer.from_table(table)

        From MJD with a reference epoch:

        .. code-block:: python

            from astropy.time import Time

            c = OpticalPhotometryContainer.from_table(
                table,
                internal_time_format="mjd",
                time_start=Time(59000.0, format="mjd"),
            )
        """
        table = table.copy()

        # Optional column renaming
        if column_map is not None:
            table.rename_columns(list(column_map.keys()), list(column_map.values()))

        # Time handling
        if internal_time_format != "relative":
            # Parse as absolute time and convert to relative days
            t_abs = Time(table["time"].data, format=internal_time_format, scale=internal_time_scale)
            if time_start is None:
                raise ValueError("time_start must be supplied when internal_time_format != 'relative'.")
            if not isinstance(time_start, Time):
                raise TypeError(
                    "time_start must be an astropy.time.Time object when "
                    "internal_time_format is an absolute time format."
                )
            t_rel = (t_abs - time_start).to(u.day)
            table["time"] = t_rel.value * u.day
        elif time_start is not None:
            # Relative times with an optional offset subtracted
            if isinstance(time_start, u.Quantity):
                offset = time_start.to(u.day).value
            else:
                offset = float(time_start)
            col = table["time"]
            vals = np.asarray(col.data, dtype=float) - offset
            table["time"] = vals * (col.unit if col.unit is not None else u.day)

        return cls(table)

    # ========================= Inference Data Conversion ========================= #
    def to_inference_data(
        self,
        model: "Model",
        variables: Optional[dict] = None,
        observables: Optional[dict] = None,
        infer_errors: bool = True,
        detection_threshold: float = 3.0,
        mask: Optional[np.ndarray] = None,
    ) -> "InferenceData":
        r"""
        Convert this container to :class:`~triceratops.data.core.InferenceData`.

        The independent variable ``band_idx`` is resolved at call time by
        looking up each observation's ``band_name`` in
        ``model.bundle.filter_names``.  The model is therefore the single
        source of truth for the band → index mapping, and the container remains
        independent of any particular filter ordering convention.

        Magnitudes are converted to physical flux densities (F_ν) before the
        :class:`InferenceData` is constructed.  The resulting object always
        contains flux in erg/s/cm²/Hz as the observable, regardless of how the
        input data were expressed.

        Parameters
        ----------
        model : Model
            Optical model instance.  Must expose a ``bundle`` attribute
            (:class:`~triceratops.utils.phot_utils.FilterBundle`) whose
            ``filter_names`` list maps band names to integer indices.

        variables : dict, optional
            Override the default variable mapping.  By default:

            - ``"time"`` → ``time`` column (converted to model units)
            - ``"band_idx"`` → resolved from ``band_name`` via
              ``model.bundle.filter_names``

        observables : dict, optional
            Override the default observable mapping.  By default, the single
            model observable is mapped to
            ``("flux_density", "flux_density_error", "flux_upper_limit", None)``.

        infer_errors : bool, default True
            If True, infer 1-sigma uncertainties for non-detections from their
            upper limits: :math:`\sigma = F_\mathrm{upper} / N_\sigma`.

        detection_threshold : float, default 3.0
            :math:`N_\sigma` used in error inference above.

        mask : np.ndarray of bool, optional
            Optional row mask applied before building the InferenceData.

        Returns
        -------
        InferenceData

        Raises
        ------
        AttributeError
            If ``model`` does not have a ``bundle`` attribute.
        KeyError
            If a ``band_name`` present in the data is not found in
            ``model.bundle.filter_names``.
        """
        # ------------------------------------------------------------------
        # Resolve band_name → band_idx via model.bundle
        # ------------------------------------------------------------------
        if not hasattr(model, "bundle"):
            raise AttributeError(
                f"Model '{type(model).__name__}' does not have a 'bundle' attribute. "
                "Optical photometry requires the model to expose a FilterBundle "
                "via 'model.bundle'."
            )

        filter_names = list(model.bundle.filter_names)
        name_to_idx = {name: idx for idx, name in enumerate(filter_names)}

        band_names = self.band_name
        try:
            band_idx = np.array([name_to_idx[n] for n in band_names], dtype=int)
        except KeyError as exc:
            bad = str(exc).strip("'")
            raise KeyError(
                f"Band name '{bad}' is not in the model's FilterBundle. Available bands: {filter_names}"
            ) from exc

        # ------------------------------------------------------------------
        # Build a working copy of the data (and optionally apply a mask)
        # ------------------------------------------------------------------
        if mask is not None:
            if mask.shape != (self.n_obs,):
                raise ValueError("Mask must have the same length as the number of observations.")
        else:
            mask = np.ones(self.n_obs, dtype=bool)

        t_vals = self.time.to(u.day).value[mask]
        idx_vals = band_idx[mask]

        # ------------------------------------------------------------------
        # Resolve flux values — convert from mag if needed
        # ------------------------------------------------------------------
        flux_vals = self.flux.to(u.Unit("erg/(s cm2 Hz)")).value[mask]
        flux_err_vals = self.flux_error.to(u.Unit("erg/(s cm2 Hz)")).value[mask]
        flux_ul_vals = self.flux_upper_limit.to(u.Unit("erg/(s cm2 Hz)")).value[mask]

        # ------------------------------------------------------------------
        # Infer errors for non-detections where they are missing
        # ------------------------------------------------------------------
        if infer_errors:
            nan_err_mask = np.isnan(flux_err_vals)
            ul_mask = ~np.isnan(flux_ul_vals)
            infer_mask = nan_err_mask & ul_mask
            if np.any(infer_mask):
                flux_err_vals[infer_mask] = flux_ul_vals[infer_mask] / detection_threshold

        if np.any(np.isnan(flux_err_vals)):
            warnings.warn(
                "flux_density_error contains NaN values after error inference. "
                "This may cause errors during likelihood evaluation. "
                "Use infer_errors=True to fill missing errors from upper limits.",
                stacklevel=2,
            )

        # ------------------------------------------------------------------
        # Build x and y dicts with astropy Quantities so from_arrays can
        # coerce to model-declared units
        # ------------------------------------------------------------------
        x = {
            "time": t_vals * u.day,
            "band_idx": idx_vals,
        }
        y = {model.output_names[0]: flux_vals * u.Unit("erg/(s cm2 Hz)")}
        y_err = {model.output_names[0]: flux_err_vals * u.Unit("erg/(s cm2 Hz)")}
        y_upper = {model.output_names[0]: flux_ul_vals * u.Unit("erg/(s cm2 Hz)")}

        return InferenceData.from_arrays(
            model=model,
            x=x,
            y=y,
            y_err=y_err,
            y_upper=y_upper,
        )
