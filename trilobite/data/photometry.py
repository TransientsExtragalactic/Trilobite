"""
Data container classes for radio photometry.

This module provides the :class:`RadioPhotometryContainer` class,
which encapsulates radio photometric observations stored in FITS tables and read into memory as
Astropy Tables. The container enforces a standardized schema, supports detection / non-detection logic,
and provides unit-aware accessors for time-, frequency-, and flux-related quantities.
"""

import warnings
import warnings as _warnings
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from astropy import units as u
from astropy.table import Table
from astropy.time import Time

from trilobite.utils.log import trilobite_logger

from .core import DataContainer, InferenceData, XYDataContainer
from .utils import infer_err_from_non_detection

if TYPE_CHECKING:
    from trilobite.models.core.base import Model

    from .light_curve import RadioLightCurveContainer


__all__ = [
    "RadioPhotometryContainer",
    "RadioPhotometryEpoch",
    "RadioPhotometryEpochContainer",  # deprecated alias
]


class RadioPhotometryContainer(DataContainer):
    """
    Immutable container for radio photometric observations.

    This class provides a **validated, unit-aware, read-only interface** to radio
    photometry data, typically originating from FITS tables. Internally, the data
    are stored as an :class:`astropy.table.Table`, but access is mediated through a
    standardized schema that enforces consistent column names, dtypes, and units.

    The container is designed to serve as a **clean boundary object** between raw
    observational data and downstream modeling, inference, and visualization
    routines. It supports explicit handling of detections versus non-detections
    (upper limits), optional epoch grouping, and safe conversion to NumPy arrays
    for numerical backends.

    Key features
    ------------
    - **Schema enforcement**
        Input tables are validated against a predefined schema specifying required
        and optional columns, expected dtypes, and physical units. Missing or
        incompatible columns raise informative errors.

    - **Unit awareness**
        All physical quantities are exposed as :class:`astropy.units.Quantity`
        objects. Unitless input columns are automatically assigned expected units
        (with a warning), while incompatible units raise an exception.

    - **Detection / non-detection logic**
        Observations are automatically classified as detections or upper limits
        based on the presence of ``flux_upper_limit`` values, with boolean masks
        and filtered table views provided.

    - **Epoch support**
        Observations may be grouped into epochs via an explicit ``epoch_id`` column
        or generated dynamically using time gaps or fixed time bins. Epochs are
        treated as metadata and do not modify the underlying data.

    - **Immutability**
        The container does not permit in-place mutation of the underlying table.
        All table accessors return copies, ensuring reproducibility and preventing
        accidental side effects during analysis.

    - **Numerical backend compatibility**
        The container can be coerced into dense NumPy arrays, either in schema units
        (via ``np.asarray(container)``) or explicitly converted to CGS base units
        (via :meth:`to_cgs_array`), for use in likelihoods, samplers, or compiled
        backends.

    Intended use
    ------------
    This class is intended to represent **observed radio photometry**, not model
    predictions or simulated data. It deliberately avoids embedding any source-
    specific physical interpretation (e.g., supernovae, jets, or shocks), and
    instead focuses on providing a robust, transparent data interface for
    higher-level emission models and inference pipelines.

    Typical workflows include:
    - Reading and validating radio photometry from FITS files
    - Grouping observations into epochs for joint modeling
    - Passing data to likelihood functions or samplers
    - Quick-look visualization of multi-frequency light curves

    Notes
    -----
    - Detection status is inferred from ``flux_upper_limit`` being NaN for
      detections and finite for non-detections.
    - Equality comparisons between containers compare the underlying tables.
    - This class intentionally does **not** subclass :class:`astropy.table.Table`
      in order to control mutability and enforce invariants.

    **Time Conventions**:

    In :class:`RadioPhotometryContainer`, the ``time`` is **always** a *relative time* in days since some
    specified zero point time. When reading from a table, using :meth:`from_table` or :meth:`from_file`,
    the user can specify a `time_starts` parameter, which is subtracted from the input time column
    to convert absolute times (e.g. JD) to relative times. This ensures that all internal handling
    of time is consistent and relative, which is important for modeling and inference. If the input table
    already contains relative times, the user can simply omit the `time_starts` parameter.

    See Also
    --------
    astropy.table.Table
        Underlying data structure used for storage.
    trilobite.inference.likelihood
        Likelihood implementations that consume this container.
    """

    # ========================= SCHEMA DEFINITION ========================= #
    # This ``COLUMNS`` dictionary contains the core schema requirements for the input
    # table in order to ensure that the radio photometry input is valid. This is then
    # enforced in ``_validate_table``.
    COLUMNS = [
        {
            "name": "obs_name",
            "dtype": str,
            "description": "Observation identifier (e.g. telescope + epoch).",
            "required": False,
        },
        {
            "name": "flux_density",
            "dtype": float,
            "unit": u.Jy,
            "description": "Measured flux density for detections.",
            "required": True,
        },
        {
            "name": "flux_density_error",
            "dtype": float,
            "unit": u.Jy,
            "description": "1-Sigma uncertainty on ``flux_density``.",
            "required": True,
        },
        {
            "name": "flux_upper_limit",
            "dtype": float,
            "unit": u.Jy,
            "description": "Upper limit on flux density for non-detections.",
            "required": True,
        },
        {
            "name": "time",
            "dtype": float,
            "unit": u.day,
            "description": "The *relative* time of each data point.",
            "required": True,
        },
        {
            "name": "obs_time",
            "dtype": float,
            "unit": u.day,
            "description": "Total integration time of the observation.",
            "required": False,
        },
        {
            "name": "freq",
            "dtype": float,
            "unit": u.GHz,
            "description": "Central observing frequency.",
            "required": True,
        },
        {
            "name": "band",
            "dtype": int,
            "description": "Integer band identifier (instrument-specific).",
            "required": False,
        },
        {
            "name": "comments",
            "dtype": str,
            "description": "Free-form comments or metadata.",
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
        Initialize the RadioPhotometryContainer.

        Parameters
        ----------
        table: astropy.table.Table
            Input table containing radio photometry data. The input data must conform
            to the schema defined in ``COLUMNS``.
        """
        # Instantiate at the super class level.
        super().__init__(table)

        # With the table validation complete, identify the detection and non-detection masks
        self.__detection_mask__ = np.isnan(np.asarray(self.__table__["flux_upper_limit"].data))
        self.__non_detection_mask__ = ~self.__detection_mask__

        # Generate the internal epochs object. If the epoch_id column is not present,
        # we set ``self.__epochs__`` to ``None``, otherwise, we extract the unique epoch IDs, order them,
        # and then store the indices of each row corresponding to each epoch ID.
        self.__epoch_ids__ = None
        if "epoch_id" in self.__table__.colnames:
            self.__epoch_ids__ = np.asarray(self.__table__["epoch_id"].data, dtype=int)

    def _validate_table(self, table: Table):
        """Ensure that the input table conforms to the required schema."""
        for column in self.__class__.COLUMNS:
            name = column["name"]
            required = column.get("required", False)

            if required and name not in table.colnames:
                raise ValueError(f"Missing required column '{name}' in input table.")

            if name not in table.colnames:
                continue

            # --- dtype coercion ---
            expected_dtype = np.dtype(column["dtype"])
            try:
                table[name] = table[name].astype(expected_dtype)
            except Exception as e:
                raise TypeError(
                    f"Column '{name}' has dtype '{table[name].dtype}', expected '{expected_dtype}'. Failed to cast: {e}"
                ) from e

            # --- unit validation ---
            expected_unit = column.get("unit", None)
            if expected_unit is not None:
                expected_unit = u.Unit(expected_unit)
                col = table[name]
                if col.unit is None:
                    trilobite_logger.warning(f"Column '{name}' has no unit. Assigning expected unit '{expected_unit}'.")
                    col.unit = expected_unit
                elif not col.unit.is_equivalent(expected_unit):
                    raise u.UnitsError(
                        f"Column '{name}' has unit '{col.unit}', "
                        f"which is not compatible with expected unit '{expected_unit}'."
                    )
                else:
                    table[name] = col.to(expected_unit)

        return table

    # ========================= Core Properties ========================= #
    # These are core properties and basic task methods.
    @property
    def detection_table(self) -> Table:
        """Return a table containing only detections."""
        return self.__table__[self.__detection_mask__].copy()

    @property
    def non_detection_table(self) -> Table:
        """Return a table containing only non-detections (upper limits)."""
        return self.__table__[self.__non_detection_mask__].copy()

    @property
    def detection_mask(self) -> np.ndarray:
        """Boolean mask selecting detections (i.e. non-upper-limits)."""
        return self.__detection_mask__

    @property
    def non_detection_mask(self) -> np.ndarray:
        """Boolean mask selecting upper limits."""
        return self.__non_detection_mask__

    @property
    def n_obs(self) -> int:
        """Return the number of observations."""
        return len(self.__table__)

    @property
    def has_epochs(self) -> bool:
        """Return True if epochs have been defined."""
        return self.__epoch_ids__ is not None

    @property
    def epoch_ids(self) -> np.ndarray:
        """Return a copy of the epoch IDs array."""
        if self.__epoch_ids__ is None:
            raise AttributeError("No epochs defined.")
        return self.__epoch_ids__.copy()

    @property
    def n_epochs(self) -> Optional[int]:
        """Return the number of unique epochs, or None if epochs are not defined."""
        if self.__epoch_ids__ is None:
            return None
        return int(len(np.unique(self.__epoch_ids__)))

    def get_epoch_mask(self, epoch: int) -> np.ndarray:
        """Return a boolean mask selecting observations in the specified epoch."""
        if self.__epoch_ids__ is None:
            raise AttributeError("No epochs defined.")
        return self.__epoch_ids__ == epoch

    def get_epoch_table(self, epoch: int) -> Table:
        """Return a table containing only observations in the specified epoch."""
        return self.__table__[self.get_epoch_mask(epoch)].copy()

    def get_epoch(self, epoch: int) -> "RadioPhotometryEpochContainer":
        """Return a RadioPhotometryContainer for the specified epoch."""
        epoch_table = self.get_epoch_table(epoch)
        return RadioPhotometryEpochContainer(epoch_table)

    @property
    def n_detections(self) -> int:
        """Return the number of detections."""
        return int(self.__detection_mask__.sum())

    @property
    def n_non_detections(self) -> int:
        """Return the number of non-detections (upper limits)."""
        return int(self.__non_detection_mask__.sum())

    @property
    def time(self) -> u.Quantity:
        """Return observation times as an Astropy Quantity array."""
        return self.__table__["time"].quantity

    @property
    def freq(self) -> u.Quantity:
        """Return observation frequencies as an Astropy Quantity array."""
        return self.__table__["freq"].quantity

    @property
    def flux_density(self) -> u.Quantity:
        """Return flux densities as an Astropy Quantity array."""
        return self.__table__["flux_density"].quantity

    @property
    def flux_density_error(self) -> u.Quantity:
        """Return flux density errors as an Astropy Quantity array."""
        return self.__table__["flux_density_error"].quantity

    @property
    def flux_upper_limit(self) -> u.Quantity:
        """Return flux upper limits as an Astropy Quantity array."""
        return self.__table__["flux_upper_limit"].quantity

    @property
    def obs_name(self) -> np.ndarray:
        """Return observation names."""
        if "obs_name" not in self.__table__.colnames:
            raise AttributeError("Column 'obs_name' not found in the table.")
        return self.__table__["obs_name"].data

    @property
    def obs_time(self) -> u.Quantity:
        """Return observation durations as an Astropy Quantity array."""
        if "obs_time" not in self.__table__.colnames:
            raise AttributeError("Column 'obs_time' not found in the table.")
        return self.__table__["obs_time"].quantity

    @property
    def band(self) -> np.ndarray:
        """Return observation band identifiers."""
        if "band" not in self.__table__.colnames:
            raise AttributeError("Column 'band' not found in the table.")
        return self.__table__["band"].data

    @property
    def comments(self) -> np.ndarray:
        """Return observation comments."""
        if "comments" not in self.__table__.colnames:
            raise AttributeError("Column 'comments' not found in the table.")
        return self.__table__["comments"].data

    # ========================= Epoch Generation ========================= #
    def set_epochs_from_indices(self, indices):
        """
        Define epochs explicitly by assigning an epoch index to each observation.

        This method sets the internal epoch structure using a user-provided array
        of integer epoch identifiers, one per observation. The provided indices
        need not be contiguous or start at zero; they will be internally normalized
        to a contiguous range ``[0, n_epochs - 1]`` while preserving grouping.

        Parameters
        ----------
        indices : array-like of int, shape (n_obs,)
            Integer epoch identifiers for each observation. Observations with the
            same value are grouped into the same epoch.

        Raises
        ------
        ValueError
            If the shape of ``indices`` does not match the number of observations.
        TypeError
            If ``indices`` does not contain integer values.

        Notes
        -----
        This method overwrites any existing epoch definition. After calling this
        method, epoch-based accessors such as ``n_epochs``, ``epoch_ids``,
        ``epoch_table``, and ``epoch_times`` become available.
        """
        indices = np.asarray(indices)
        if indices.shape != (self.n_obs,):
            raise ValueError("Epoch indices must have shape (n_obs,)")

        if not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("Epoch indices must be integers")

        # Normalize to 0..N-1
        unique = np.unique(indices)
        remap = {old: new for new, old in enumerate(unique)}
        self.__epoch_ids__ = np.vectorize(remap.get)(indices)

    def set_epochs_from_time_gaps(self, max_gap: u.Quantity):
        """
        Define epochs by grouping observations separated by time gaps.

        Observations are first sorted by observation time. A new epoch is started
        whenever the time difference between consecutive observations exceeds
        ``max_gap``. All observations separated by gaps smaller than or equal to
        ``max_gap`` are grouped into the same epoch.

        Parameters
        ----------
        max_gap : astropy.units.Quantity
            Maximum allowed time gap between consecutive observations within the
            same epoch. Must be convertible to the units of the ``time`` column.

        Raises
        ------
        astropy.units.UnitsError
            If ``max_gap`` is not compatible with the time units of the data.

        Notes
        -----
        - Epoch assignment is invariant under reordering of the input table.
        - Epoch IDs are assigned in increasing order of time.
        - This method overwrites any existing epoch definition.
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

    def set_epochs_from_bins(self, bins):
        """
        Define epochs by binning observations into fixed time intervals.

        Each observation is assigned to an epoch based on which time bin it falls
        into. Epoch IDs correspond to bin indices in the range
        ``[0, len(bins) - 2]``.

        Parameters
        ----------
        bins : array-like or astropy.units.Quantity
            Monotonically increasing array of bin edges. If a Quantity is provided,
            it must be convertible to the units of the ``time`` column.

        Raises
        ------
        ValueError
            If ``bins`` is not one-dimensional, has fewer than two edges, or if any
            observations fall outside the bin range.

        Notes
        -----
        - Bins follow NumPy's ``digitize`` convention: bins are left-inclusive and
          right-exclusive, except for the final bin.
        - This method overwrites any existing epoch definition.
        """
        if isinstance(bins, u.Quantity):
            bins = bins.to(self.time.unit).value
        bins = np.asarray(bins)

        if bins.ndim != 1 or len(bins) < 2:
            raise ValueError("bins must be 1D with at least two edges")

        epoch_ids = np.digitize(self.time.value, bins) - 1
        if np.any(epoch_ids < 0) or np.any(epoch_ids >= len(bins) - 1):
            raise ValueError("Some observations fall outside bin range")

        self.__epoch_ids__ = epoch_ids

    # ========================= Inference Data Conversion ========================= #
    def to_inference_data(
        self,
        model: "Model",
        variables: Optional[dict[str, tuple[str, ...]]] = None,
        observables: Optional[dict[str, tuple[str, ...]]] = None,
        infer_errors: bool = True,
        detection_threshold: float = 3.0,
        mask: Optional[np.ndarray] = None,
    ) -> "InferenceData":
        r"""
        Convert this :class:`RadioPhotometryContainer` into an :class:`~trilobite.data.core.InferenceData` object.

        This method provides a convenient bridge between validated radio
        photometry data and the inference pipeline. It maps table columns
        onto the independent variables and observables expected by the
        supplied model and constructs a fully validated
        :class:`InferenceData` instance.

        Default behavior
        ----------------
        If ``variables`` and/or ``observables`` are not provided, sensible
        defaults are inferred from the model's declared interface:

        - Independent variables are inferred from
          :attr:`model.variable_names`.

          - ``"time"`` → maps to the ``"time"`` column.
          - ``"freq"`` or ``"frequency"`` → maps to the ``"freq"`` column.

          If the model declares any variable name that cannot be
          automatically mapped, a :class:`ValueError` is raised and the user
          must provide an explicit mapping.

        - Observables are inferred only if the model declares exactly one
          observable in :attr:`model.observable_names`.

          In this case, that observable is mapped to the full radio
          photometry tuple:

          ``("flux_density", "flux_density_error", "flux_upper_limit", None)``

          This ensures compatibility with both standard Gaussian and
          censored likelihoods.

        If the model declares multiple observables, the mapping must be
        provided explicitly via the ``observables`` argument.

        Parameters
        ----------
        model : Model
            The model instance for which inference will be performed.
            Its declared ``variable_names`` and ``observable_names`` are
            used to determine default mappings.

        variables : dict of str to tuple of str, optional
            Mapping from model variable names to table column names.

            Each key must correspond to a name in
            ``model.variable_names``. Each value is a tuple of column names
            in this photometry table used to construct that variable.

            If not provided, defaults are inferred as described above.

        observables : dict of str to tuple of str, optional
            Mapping from model observable names to photometry table columns.

            Each tuple should follow the convention expected by
            :meth:`InferenceData.from_table`, typically:

            ``(value_column, error_column, upper_column, lower_column)``

            If not provided and the model declares exactly one observable,
            a default mapping to the flux density columns is inferred.
        infer_errors : bool, optional
            If ``True``, missing uncertainties for non-detections will be inferred
            from their reported upper limits under the assumption that the upper
            limit corresponds to an :math:`N\sigma` detection threshold.

            Specifically, for an upper limit :math:`F_{\rm upper}` and detection
            threshold :math:`N`, the inferred uncertainty is

            .. math::

                \sigma = \frac{F_{\rm upper}}{N}.

            This allows censored Gaussian likelihoods to be evaluated even when
            the original dataset does not explicitly report 1σ uncertainties for
            non-detections.

            This inference is applied **only** where uncertainties are missing.
            Any explicitly provided ``flux_density_error`` values are preserved
            and never overwritten.

            If ``False`` (default), the likelihood will raise an error if
            non-detections lack associated uncertainties.

        detection_threshold : float, optional
            The number of standard deviations corresponding to the reported
            upper limits when ``infer_errors=True``.

            For example, if upper limits are quoted at the 3σ level, set

            .. code-block:: python

                detection_threshold = 3.0

            The inferred uncertainty is then

            .. math::

                \sigma = \frac{F_{\rm upper}}{\text{detection_threshold}}.

            This parameter has no effect unless ``infer_errors=True``.
        mask: np.ndarray, optional
            Optional boolean mask to apply to the data before conversion. Only rows where
            the mask is True will be included in the resulting InferenceData. This allows
            users to easily subset the data (e.g. by epoch, frequency, or detection status
            before conversion.

        Returns
        -------
        InferenceData
            A validated, immutable inference data object ready to be
            consumed by a likelihood.

        Raises
        ------
        ValueError
            If automatic mapping fails because the model declares unknown
            variables or multiple observables without explicit mapping.

        Notes
        -----
        This method does not perform any statistical validation. It
        strictly handles structural mapping between this container and the
        inference interface.

        Advanced workflows may bypass this method and construct
        :class:`InferenceData` objects manually via
        :meth:`InferenceData.from_table` for full control.

        See Also
        --------
        InferenceData.from_table
            Lower-level constructor used internally.
        trilobite.inference.likelihood
            Likelihood classes that consume :class:`InferenceData`.
        """
        # ------------------------------------------------------------
        # Default variable mapping
        # ------------------------------------------------------------
        if variables is None:
            variables = {}

            for var_name in model.variable_names:
                if var_name == "time":
                    variables[var_name] = "time"
                elif var_name in {"freq", "frequency"}:
                    variables[var_name] = "freq"
                else:
                    raise ValueError(
                        f"Cannot infer mapping for model variable '{var_name}'. Please specify `variables=` explicitly."
                    )

        # ------------------------------------------------------------
        # Default observable mapping
        # ------------------------------------------------------------
        if observables is None:
            if len(model.output_names) != 1:
                raise ValueError(
                    "RadioPhotometryContainer can only infer defaults for "
                    "single-observable models. Please specify `observables=`."
                )

            observables = {
                model.output_names[0]: (
                    "flux_density",
                    "flux_density_error",
                    "flux_upper_limit",
                    None,
                )
            }

        # ------------------------------------------------------------
        # Infer errors for non-detections if requested and if any are missing
        # ------------------------------------------------------------
        table = self.__table__.copy()

        if infer_errors:
            nan_errors_mask = np.isnan(self.flux_density_error)
            upper_limits_mask = self.non_detection_mask

            _mask = upper_limits_mask & nan_errors_mask

            _masked_upper_limits = self.flux_upper_limit[_mask]
            inferred_error = infer_err_from_non_detection(
                flux_limit=_masked_upper_limits.value,
                sigma=detection_threshold,
            )

            # Update the table with the inferred errors for non-detections where errors are missing
            table["flux_density_error"][_mask] = inferred_error * self.flux_upper_limit.unit

        if np.any(np.isnan(table["flux_density_error"])):
            warnings.warn(
                "Flux density error contains NaN values!"
                " This will likely cause errors in likelihood evaluation.\n"
                " Use the `infer_errors` option to automatically infer errors"
                " for non-detections based on their upper limits.",
                stacklevel=2,
            )

        # ------------------------------------------------------------
        # Construct inference data
        # ------------------------------------------------------------
        if mask is not None:
            if mask.shape != (len(table),):
                raise ValueError("Mask must have the same length as the number of observations.")
            table = table[mask]

        return InferenceData.from_table(
            model=model,
            table=table,
            variables=variables,
            observables=observables,
        )

    # ========================= Masking ========================= #
    def apply_mask(self, mask) -> "RadioPhotometryContainer":
        """Return a new container containing only the rows selected by *mask*.

        Parameters
        ----------
        mask : array-like of bool or int
            Boolean array of the same length as this container, or an integer
            index array.

        Returns
        -------
        RadioPhotometryContainer
            A new container with the selected rows and schema validation
            re-applied.  If epoch IDs were defined on this container (whether
            via an ``epoch_id`` column or via :meth:`set_epochs_from_indices`),
            they are propagated to the returned container.
        """
        sliced = self.table[mask]

        # If epoch IDs are defined but not yet stored as a table column, inject
        # them so the reconstructed container retains the epoch structure.
        if self.has_epochs and "epoch_id" not in sliced.colnames:
            sliced["epoch_id"] = self.__epoch_ids__[mask]

        return self.__class__.from_table(sliced)

    def extract_lightcurve(
        self,
        band: Union[float, u.Quantity],
        bandwidth: Union[float, u.Quantity],
    ) -> "RadioLightCurveContainer":
        """Extract a single-frequency light curve from this container.

        Selects all observations within ``[band - bandwidth/2, band + bandwidth/2]``
        and returns them as a :class:`~trilobite.data.light_curve.RadioLightCurveContainer`.

        Parameters
        ----------
        band : float or ~astropy.units.Quantity
            Center frequency. A bare float is interpreted as GHz.
        bandwidth : float or ~astropy.units.Quantity
            Total frequency window. A bare float is interpreted as GHz.

        Returns
        -------
        ~trilobite.data.light_curve.RadioLightCurveContainer
            Light curve at the specified frequency band, ordered by time.

        Raises
        ------
        ValueError
            If no observations fall within the specified frequency window.
        """
        from .light_curve import RadioLightCurveContainer

        if not isinstance(band, u.Quantity):
            band = band * u.GHz
        if not isinstance(bandwidth, u.Quantity):
            bandwidth = bandwidth * u.GHz

        band_hz = band.to(u.Hz).value
        half_hz = (bandwidth / 2).to(u.Hz).value
        freq_hz = self.freq.to(u.Hz).value

        mask = np.abs(freq_hz - band_hz) <= half_hz
        if not np.any(mask):
            raise ValueError(
                f"No observations found within {band} ± {bandwidth / 2}. "
                f"Available frequencies span {self.freq.min():.3f} – {self.freq.max():.3f}."
            )

        sliced = self.table[mask]
        lc_cols = ["time", "flux_density", "flux_density_error", "flux_upper_limit"]
        optional = ["obs_name", "comments"]
        keep = lc_cols + [c for c in optional if c in sliced.colnames]
        lc_table = sliced[keep]

        sort_order = np.argsort(lc_table["time"])
        return RadioLightCurveContainer(lc_table[sort_order], frequency=band)

    # ========================= IO Methods ========================= #
    @classmethod
    def from_table(
        cls,
        table: Table,
        column_map: Optional[dict] = None,
        time_starts: Optional[Union[u.Quantity, Time, float]] = None,
        internal_time_format: str = "relative",
        internal_time_scale: str = "utc",
    ):
        """
        Construct a :class:`RadioPhotometryContainer` from an Astropy table.

        This constructor provides flexible handling of time columns, allowing
        both relative and absolute time representations. Internally, all times
        are converted to **relative days**.

        Parameters
        ----------
        table : astropy.table.Table
            Input table containing radio photometry data. Must contain a ``time``
            column after optional renaming.

        column_map : dict, optional
            Mapping from existing column names → expected column names.
            Useful when loading heterogeneous datasets.

        time_starts : float, astropy.units.Quantity, or astropy.time.Time, optional
            Reference start time used to convert absolute times into relative time.

            Interpretation depends on ``internal_time_format``:

            - If ``internal_time_format="relative"``, then ``time_starts`` must be
              a float (assumed days) or Quantity.
            - Otherwise, ``time_starts`` must be an ``astropy.time.Time`` object.

        internal_time_format : str, default="relative"
            Format used to interpret the input time column.

            Options:

            - ``"relative"`` → time column already represents elapsed time.
            - Any valid Astropy time format (e.g., ``"mjd"``, ``"jd"``,
              ``"iso"``, ``"unix"``).

        internal_time_scale : str, default="utc"
            Astropy time scale to use when interpreting absolute times.
            Only used if ``internal_time_format != "relative"``.

        Returns
        -------
        RadioPhotometryContainer
            Validated container with internal times stored as relative days.

        Notes
        -----
        - Internally, all times are stored as relative days (``astropy.units.Quantity``).
        - Absolute time formats are converted via ``astropy.time.Time``.
        - If the input time column has no units, days are assumed (with a warning).
        - If the time column is already an ``astropy.time.Time`` object,
          its format and scale are preserved.

        Examples
        --------
        Case 1 — Already relative times (days since explosion)

        .. code-block:: python

            container = RadioPhotometryContainer.from_table(
                table,
                internal_time_format="relative",
            )


        Case 2 — Absolute times in MJD

        .. code-block:: python

            from astropy.time import Time

            t0 = Time(58285.0, format="mjd")

            container = RadioPhotometryContainer.from_table(
                table,
                internal_time_format="mjd",
                time_starts=t0,
            )


        Case 3 — Absolute times in JD

        .. code-block:: python

            from astropy.time import Time

            t0 = Time(2458285.5, format="jd")

            container = RadioPhotometryContainer.from_table(
                table,
                internal_time_format="jd",
                time_starts=t0,
            )


        Case 4 — Relative times with offset

        .. code-block:: python

            from astropy import units as u

            container = RadioPhotometryContainer.from_table(
                table,
                internal_time_format="relative",
                time_starts=5.0 * u.day,
            )
        """
        # --------------------------------------------------
        # Column Renaming
        # --------------------------------------------------
        if column_map is not None:
            table = table.copy()
            table.rename_columns(list(column_map.keys()), list(column_map.values()))

        # --------------------------------------------------
        # Time Handling
        # --------------------------------------------------
        if time_starts is None:
            return cls(table)

        if "time" not in table.colnames:
            raise ValueError("Column 'time' not found in table.")

        # Ensure that the time column has a valid dtype or attempt to coerce it.
        if not np.issubdtype(table["time"].dtype, np.number):
            try:
                table["time"] = table["time"].astype(float)
            except Exception as e:
                raise TypeError(
                    f"Column 'time' has dtype '{table['time'].dtype}', expected a numerical dtype. Failed to cast: {e}"
                ) from e

        time_col = table["time"]

        # If time column is already a Time object
        if isinstance(time_col, Time):
            t_obs = time_col
        else:
            # Ensure unit exists
            if getattr(time_col, "unit", None) is None:
                trilobite_logger.warning("Column 'time' has no unit. Assuming days.")
                time_col.unit = u.day

            # ----------------------------------------------
            # CASE 1: Relative time
            # ----------------------------------------------
            if internal_time_format == "relative":
                if isinstance(time_starts, Time):
                    raise TypeError("time_starts must be float or Quantity when internal_time_format='relative'.")

                if isinstance(time_starts, u.Quantity):
                    t0 = time_starts.to(u.day)
                elif isinstance(time_starts, (int, float)):
                    t0 = time_starts * u.day
                else:
                    raise TypeError("time_starts must be float or Quantity when internal_time_format='relative'.")

                t_days = time_col.quantity.to(u.day)

                table["time"] = (t_days - t0).to(u.day)
                return cls(table)

            # ----------------------------------------------
            # CASE 2: Absolute time
            # ----------------------------------------------
            if not isinstance(time_starts, Time):
                raise TypeError("time_starts must be astropy.time.Time when using absolute time formats.")

            time_values = time_col.quantity.to_value(u.day)

            try:
                t_obs = Time(
                    time_values,
                    format=internal_time_format,
                    scale=internal_time_scale,
                )
            except Exception as exc:
                raise ValueError(f"Invalid internal_time_format '{internal_time_format}'.") from exc

        # ----------------------------------------------
        # Convert absolute → relative days
        # ----------------------------------------------
        delta = (t_obs - time_starts).to(u.day)
        table["time"] = delta

        return cls(table)

    @classmethod
    def from_file(
        cls,
        path: Union[str, Path],
        column_map: Optional[dict] = None,
        time_starts: u.Quantity = None,
        internal_time_format: str = "relative",
        internal_time_scale: str = "utc",
        **kwargs,
    ):
        """
        Create a :class:`RadioPhotometryContainer` from a FITS file.

        This method reads a FITS table from the specified file path and constructs
        a RadioPhotometryContainer. Optional column renaming can be performed via
        the `column_map` parameter.

        Parameters
        ----------
        path : str or pathlib.Path
            The file path to the FITS file containing the radio photometry data.
        column_map : dict, optional
            A mapping from existing column names (key) to expected column names (value).
            If provided, the table's columns will be renamed accordingly before validation. If
            a column is not present in the mapping, it will retain its original name.
        time_starts : astropy.units.Quantity, optional
            If provided, this time will be subtracted from the 'time' column to convert
            absolute times to relative times.
        internal_time_format : str, default="relative"
            Format used to interpret the input time column.

            Options:

            - ``"relative"`` → time column already represents elapsed time.
            - Any valid Astropy time format (e.g., ``"mjd"``, ``"jd"``,
              ``"iso"``, ``"unix"``).

        internal_time_scale : str, default="utc"
            Astropy time scale to use when interpreting absolute times.
            Only used if ``internal_time_format != "relative"``.
        **kwargs:
            Additional keyword arguments to pass to `astropy.table.Table.read`.

        Returns
        -------
        RadioPhotometryContainer
            The constructed RadioPhotometryContainer instance.
        """
        table = Table.read(path, **kwargs)
        return cls.from_table(
            table,
            column_map=column_map,
            time_starts=time_starts,
            internal_time_format=internal_time_format,
            internal_time_scale=internal_time_scale,
        )

    def to_file(self, path: Union[str, Path], **kwargs):
        """
        Write the RadioPhotometryContainer's table to a FITS file.

        Parameters
        ----------
        path : str or pathlib.Path
            The file path where the FITS file will be saved.
        **kwargs:
            Additional keyword arguments to pass to `astropy.table.Table.write`.
        """
        self.__table__.write(path, **kwargs)

    # ======================== QUICK VISUALIZATION ======================== #
    def plot_photometry(
        self,
        fig=None,
        axes=None,
        show_upper_limits=True,
        color=None,
        cmap=None,
        show_colorbar=True,
        **kwargs,
    ):
        """
        Plot the radio photometry data.

        Parameters
        ----------
        fig: matplotlib.figure.Figure, optional
            Figure to plot on. If None, a new figure will be created.
        axes: matplotlib.axes.Axes, optional
            Axes to plot on. If None, axes will be created or retrieved from the figure
        show_upper_limits: bool, optional
            Whether to show upper limits in the plot. Default is True.
        color: str, optional
            If provided, use this color for all points instead of a colormap.
        cmap: str, optional
            Colormap to use for coloring points by time. Ignored if `color` is provided.
        show_colorbar: bool, optional
            Whether to show a colorbar when using a colormap. Default is True.
        kwargs:
            Additional keyword arguments for customizing the plot appearance.
            Supported kwargs include:

            - ``detection_fmt``: Format string for detection markers (default: 'o').
            - ``detection_capsize``: Capsize for detection error bars (default: 3).
            - ``detection_ms``: Marker size for detections (default: 5).
            - ``detection_mec``: Marker edge color for detections (default: 'none').
            - ``upper_limit_fmt``: Format string for upper limit markers (default: 'v').
            - ``upper_limit_capsize``: Capsize for upper limit error bars (default: 3).
            - ``upper_limit_ms``: Marker size for upper limits (default: 5).
            - ``upper_limit_mec``: Marker edge color for upper limits (default: 'none').
            - ``colorbar_label``: Label for the colorbar (default: 'Observation Time (days)').
            - ``xlabel``: Label for the x-axis (default: 'Frequency (GHz)').
            - ``ylabel``: Label for the y-axis (default: 'Flux Density (mJy)').

        Returns
        -------
        fig: matplotlib.figure.Figure
            The figure containing the plot.
        axes: matplotlib.axes.Axes
            The axes containing the plot.
        """
        # Import matplotlib and our various plotting utilities.
        import matplotlib.pyplot as plt

        from trilobite.utils.plot_utils import (
            get_cmap,
            get_default_cmap,
            resolve_fig_axes,
            set_plot_style,
        )

        # Set the plot style and resolve the figure and the axes.
        set_plot_style()
        fig, axes = resolve_fig_axes(fig=fig, axes=axes, fig_size=(8, 6))

        # --- Color Management --- #
        # We handle color with two mechanisms: ``cmap`` and ``color``. If
        # ``color`` is provided, we use that color for all points. If ``color`` is
        # not provided, we default to the color map and color by time.
        if color is not None:
            # We're going to run with a solid color for all points.
            show_colorbar = False

            # set colors to a single color array for all of the
            # points.
            colors = np.array([color] * self.n_obs)
        else:
            # We're going to use a colormap to color by whichever field
            # is specified (default is time).
            cmap = cmap if cmap is not None else get_default_cmap()
            cmap = get_cmap(cmap)

            # Extract the time field to use as the cmap values. If we don't
            # have a norm provided, we create one. Always use days.
            norm = kwargs.get("norm", None)
            if norm is None:
                times = self.time.to(u.day).value
                min_time, max_time = times.min(), times.max()
                norm = plt.Normalize(min_time, max_time)

            # create the color array.
            colors = cmap(norm(self.time.to(u.day).value))

        # Now plot the detections with the specified markers and colors.
        detection_mask = self.detection_mask

        for i in np.where(detection_mask)[0]:
            axes.errorbar(
                self.freq[i].to(u.GHz).value,
                self.flux_density[i].to(u.mJy).value,
                yerr=self.flux_density_error[i].to(u.mJy).value,
                fmt=kwargs.get("detection_fmt", "o"),
                color=colors[i],
                capsize=kwargs.get("detection_capsize", 3),
                ms=kwargs.get("detection_ms", 5),
                mec=kwargs.get("detection_mec", "none"),
            )
        # Plot the upper limits if requested.
        if show_upper_limits:
            non_detection_mask = self.non_detection_mask
            for i in np.where(non_detection_mask)[0]:
                axes.errorbar(
                    self.freq[i].to(u.GHz).value,
                    self.flux_upper_limit[i].to(u.mJy).value,
                    yerr=0.2 * self.flux_upper_limit[i].to(u.mJy).value,
                    uplims=True,
                    fmt=kwargs.get("upper_limit_fmt", "v"),
                    color=colors[i],
                    capsize=kwargs.get("upper_limit_capsize", 3),
                    ms=kwargs.get("upper_limit_ms", 5),
                    mec=kwargs.get("upper_limit_mec", "none"),
                )

        # configure the colorbar.
        if show_colorbar:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            fig.colorbar(sm, ax=axes, label=kwargs.get("colorbar_label", r"Observation Time (days)"))

        # Configure the axes and the labels.
        axes.set_xscale("log")
        axes.set_yscale("log")
        axes.set_xlabel(kwargs.get("xlabel", r"Frequency (GHz)"))
        axes.set_ylabel(kwargs.get("ylabel", r"Flux Density (mJy)"))

        return fig, axes

    def plot_epoch(
        self,
        epoch_id,
        *,
        fig=None,
        axes=None,
        label=None,
        color=None,
        show_upper_limits=True,
        detection_style=None,
        upper_limit_style=None,
    ):
        """
        Plot photometry for a single epoch onto an existing axes.

        This method plots detections and (optionally) upper limits for a given
        epoch using a single color and adds a single legend entry.

        Parameters
        ----------
        epoch_id : int
            Epoch identifier to plot.
        fig : matplotlib.figure.Figure, optional
            Figure to plot on.
        axes : matplotlib.axes.Axes, optional
            Axes to plot on.
        label : str, optional
            Label for the legend entry corresponding to this epoch.
        color : str or tuple, optional
            Color to use for this epoch. If None, Matplotlib will assign one.
        show_upper_limits : bool, optional
            Whether to plot upper limits.
        detection_style : dict, optional
            Matplotlib style kwargs for detections.
        upper_limit_style : dict, optional
            Matplotlib style kwargs for upper limits.
        """
        import matplotlib.pyplot as plt

        from trilobite.utils.plot_utils import resolve_fig_axes, set_plot_style

        # Set the plot style and resolve the figure and axes.
        set_plot_style()
        fig, axes = resolve_fig_axes(fig=fig, axes=axes)

        # Defaults
        detection_style = detection_style or {}
        upper_limit_style = upper_limit_style or {}
        color = color or plt.rcParams["axes.prop_cycle"].by_key()["color"][0]

        # Select epoch
        mask = self.get_epoch_mask(epoch_id)
        det_mask = mask & self.detection_mask
        ul_mask = mask & self.non_detection_mask

        # Plot detections
        if det_mask.any():
            for i in np.where(det_mask)[0]:
                axes.errorbar(
                    self.freq[i].to_value(self.freq.unit),
                    self.flux_density[i].to_value(self.flux_density.unit),
                    yerr=self.flux_density_error[i].to_value(self.flux_density_error.unit),
                    color=color,
                    label=label if i == np.where(det_mask)[0][0] else None,
                    **detection_style,
                )

        # Plot upper limits
        if show_upper_limits and ul_mask.any():
            for i in np.where(ul_mask)[0]:
                axes.errorbar(
                    self.freq[i].to_value(self.freq.unit),
                    self.flux_upper_limit[i].to_value(self.flux_upper_limit.unit),
                    yerr=0.2 * self.flux_upper_limit[i].to_value(self.flux_upper_limit.unit),
                    uplims=True,
                    color=color,
                    **upper_limit_style,
                )

        return fig, axes


class RadioPhotometryEpoch(XYDataContainer):
    """
    Container for single-epoch, multi-frequency radio photometry.

    A :class:`RadioPhotometryEpochContainer` represents a collection of radio
    flux density measurements taken at a *single epoch in time* but potentially
    across multiple observing frequencies or bands. Each row corresponds to one
    frequency-channel measurement and may represent either a detection or a
    non-detection (upper limit).

    This container is designed to support **spectral modeling and inference**
    (e.g., fitting synchrotron spectra, SEDs, or phenomenological spectral models)
    where frequency is treated as the independent variable and flux density as
    the dependent variable.

    The container inherits from :class:`~trilobite.data.core.XYDataContainer`, exposing a standardized
    ``(x, y)`` interface with optional uncertainties and censoring while retaining
    domain-specific column semantics relevant to radio astronomy.

    Key features
    ------------
    - **Frequency-domain data**:
      The independent variable is observing frequency (``freq``), exposed via
      the :attr:`x` and :attr:`frequency` properties.

    - **Detection-aware**:
      Non-detections are represented via a flux density upper-limit column
      (``flux_upper_limit``). Detection and non-detection masks are computed
      automatically and exposed through convenience properties.

    - **Unit-aware and validated**:
      All physical quantities are validated and stored with Astropy units at
      construction time.

    - **Immutable by design**:
      The underlying data table cannot be modified in place. All accessors return
      copies to ensure reproducibility during inference.

    Intended use
    ------------
    This container is intended to be consumed by likelihood classes that compare
    model-predicted spectra to observed radio photometry, including:

    - synchrotron spectral energy distribution (SED) fitting,
    - broadband spectral index estimation,
    - forward modeling of physical emission models,
    - inference with censored data (upper limits).

    Notes
    -----
    - All rows are assumed to correspond to the same observing epoch.
      Time information, if needed, should be handled by a higher-level container
      (e.g., a light curve or epoch-grouping container).
    - Detection status is inferred from finite values in
      ``flux_upper_limit``; rows with finite upper limits are treated as
      non-detections.
    - This class makes no assumptions about the statistical treatment of
      detections or upper limits; such logic is handled entirely by the
      corresponding likelihood.

    See Also
    --------
    XYDataContainer
        Base class providing generic ``(x, y)`` data access and censoring logic.
    RadioLightCurveContainer
        Container for single-frequency, time-domain radio observations.
    trilobite.inference.likelihood
        Likelihood implementations that operate on photometry containers.
    """

    # ========================= SCHEMA DEFINITION ========================= #
    # This ``COLUMNS`` dictionary contains the core schema requirements for the input
    # table in order to ensure that the radio photometry input is valid. This is then
    # enforced in ``_validate_table``.
    COLUMNS = [
        {
            "name": "obs_name",
            "dtype": str,
            "description": "Observation identifier (e.g. telescope + epoch).",
            "required": False,
        },
        {
            "name": "flux_density",
            "dtype": float,
            "unit": u.Jy,
            "description": "Measured flux density for detections.",
            "required": True,
        },
        {
            "name": "flux_density_error",
            "dtype": float,
            "unit": u.Jy,
            "description": "1-Sigma uncertainty on ``flux_density``.",
            "required": True,
        },
        {
            "name": "flux_upper_limit",
            "dtype": float,
            "unit": u.Jy,
            "description": "Upper limit on flux density for non-detections.",
            "required": True,
        },
        {
            "name": "freq",
            "dtype": float,
            "unit": u.GHz,
            "description": "Central observing frequency.",
            "required": True,
        },
        {
            "name": "band",
            "dtype": int,
            "description": "Integer band identifier (instrument-specific).",
            "required": False,
        },
        {
            "name": "comments",
            "dtype": str,
            "description": "Free-form comments or metadata.",
            "required": False,
        },
    ]

    # ========================= SPECIAL COLUMNS ========================= #
    X_COLUMN = "freq"
    Y_COLUMN = "flux_density"
    Y_ERROR_COLUMN = "flux_density_error"
    Y_UPPER_LIMIT_COLUMN = "flux_upper_limit"

    # ========================= METADATA ========================= #
    @property
    def freq(self) -> u.Quantity:
        """Observing frequency of the light curve."""
        return self.x

    # ========================= DATA ACCESS ========================= #
    @property
    def flux_density(self) -> u.Quantity:
        """Flux density measurements."""
        return self.table["flux_density"].quantity

    @property
    def flux_density_error(self) -> u.Quantity:
        """Flux density uncertainties."""
        return self.table["flux_density_error"].quantity

    @property
    def flux_upper_limit(self) -> u.Quantity:
        """Flux density upper limits."""
        return self.table["flux_upper_limit"].quantity

    # ========================= DETECTION LOGIC ========================= #
    @property
    def detection_mask(self) -> np.ndarray:
        """Boolean mask selecting detections."""
        return ~self.y_lim_mask

    @property
    def non_detection_mask(self) -> np.ndarray:
        """Boolean mask selecting upper limits."""
        return self.y_lim_mask

    @property
    def n_detections(self) -> int:
        """Return the number of detections."""
        return int(self.detection_mask.sum())

    @property
    def n_non_detections(self) -> int:
        """Return the number of non-detections (upper limits)."""
        return int(self.non_detection_mask.sum())

    @property
    def detection_table(self) -> Table:
        """Return a table containing only detections."""
        return self.__table__[self.detection_mask].copy()

    @property
    def non_detection_table(self) -> Table:
        """Return a table containing only non-detections (upper limits)."""
        return self.__table__[self.non_detection_mask].copy()

    @property
    def n_obs(self) -> int:
        """Number of observations."""
        return len(self)

    # ========================= CONSTRUCTORS ========================= #
    @classmethod
    def from_table(
        cls,
        table: Table,
        *,
        column_map: Optional[dict] = None,
    ):
        """Construct a :class:`RadioPhotometryEpoch` from an Astropy table."""
        if column_map is not None:
            table = table.copy()
            table.rename_columns(list(column_map.keys()), list(column_map.values()))

        return cls(table)

    @classmethod
    def from_file(
        cls,
        path: Union[str, Path],
        **kwargs,
    ):
        """Construct a :class:`RadioPhotometryEpoch` from a file on disk."""
        table = Table.read(path, **kwargs)
        return cls(
            table,
        )

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
        Convert this :class:`RadioPhotometryEpoch` into an :class:`~trilobite.data.core.InferenceData` object.

        For a single-epoch SED container, frequency is the independent variable
        and flux density is the observable. The default mapping is:

        - ``variables``: ``{"freq": "freq"}`` (or the model's declared frequency variable)
        - ``observables``: the single model output mapped to the radio flux tuple

        Parameters
        ----------
        model : Model
            The model instance. Its ``variable_names`` and ``output_names`` are
            used to determine default mappings.
        variables : dict, optional
            Override the default variable mapping.
        observables : dict, optional
            Override the default observable mapping.
        infer_errors : bool, default True
            If True, infer 1-sigma uncertainties for non-detections from their
            upper limits: :math:`\sigma = F_\mathrm{upper} / N_\sigma`.
        detection_threshold : float, default 3.0
            :math:`N_\sigma` used in error inference.
        mask : np.ndarray of bool, optional
            Optional row mask applied before building the InferenceData.

        Returns
        -------
        InferenceData
        """
        # ------------------------------------------------------------
        # Default variable mapping
        # ------------------------------------------------------------
        if variables is None:
            variables = {}
            for var_name in model.variable_names:
                if var_name in {"freq", "frequency"}:
                    variables[var_name] = "freq"
                else:
                    raise ValueError(
                        f"Cannot infer mapping for model variable '{var_name}'. "
                        "RadioPhotometryEpoch only maps 'freq'/'frequency'. "
                        "Please specify `variables=` explicitly."
                    )

        # ------------------------------------------------------------
        # Default observable mapping
        # ------------------------------------------------------------
        if observables is None:
            if len(model.output_names) != 1:
                raise ValueError(
                    "RadioPhotometryEpoch can only infer defaults for "
                    "single-observable models. Please specify `observables=`."
                )
            observables = {
                model.output_names[0]: (
                    "flux_density",
                    "flux_density_error",
                    "flux_upper_limit",
                    None,
                )
            }

        # ------------------------------------------------------------
        # Infer errors for non-detections if requested
        # ------------------------------------------------------------
        table = self.__table__.copy()

        if infer_errors:
            nan_errors_mask = np.isnan(self.flux_density_error)
            upper_limits_mask = self.non_detection_mask
            _mask = upper_limits_mask & nan_errors_mask
            _masked_upper_limits = self.flux_upper_limit[_mask]
            inferred_error = infer_err_from_non_detection(
                flux_limit=_masked_upper_limits.value,
                sigma=detection_threshold,
            )
            table["flux_density_error"][_mask] = inferred_error * self.flux_upper_limit.unit

        if np.any(np.isnan(table["flux_density_error"])):
            warnings.warn(
                "Flux density error contains NaN values! "
                "This will likely cause errors in likelihood evaluation.\n"
                "Use the `infer_errors` option to automatically infer errors "
                "for non-detections based on their upper limits.",
                stacklevel=2,
            )

        # ------------------------------------------------------------
        # Construct inference data
        # ------------------------------------------------------------
        if mask is not None:
            if mask.shape != (len(table),):
                raise ValueError("Mask must have the same length as the number of observations.")
            table = table[mask]

        return InferenceData.from_table(
            model=model,
            table=table,
            variables=variables,
            observables=observables,
        )

    # ======================== PLOTTING METHODS ======================== #
    def plot(
        self,
        *,
        fig=None,
        axes=None,
        label=None,
        color=None,
        show_upper_limits=True,
        detection_style=None,
        upper_limit_style=None,
    ):
        """
        Plot this photometry epoch onto an existing axes.

        This method plots detections and (optionally) upper limits for a given
        epoch using a single color and adds a single legend entry.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            Figure to plot on.
        axes : matplotlib.axes.Axes, optional
            Axes to plot on.
        label : str, optional
            Label for the legend entry corresponding to this epoch.
        color : str or tuple, optional
            Color to use for this epoch. If None, Matplotlib will assign one.
        show_upper_limits : bool, optional
            Whether to plot upper limits.
        detection_style : dict, optional
            Matplotlib style kwargs for detections.
        upper_limit_style : dict, optional
            Matplotlib style kwargs for upper limits.
        """
        import matplotlib.pyplot as plt

        from trilobite.utils.plot_utils import resolve_fig_axes, set_plot_style

        # Set the plot style and resolve the figure and axes.
        set_plot_style()
        fig, axes = resolve_fig_axes(fig=fig, axes=axes)

        # Defaults
        detection_style = detection_style or {}
        upper_limit_style = upper_limit_style or {}
        color = color or plt.rcParams["axes.prop_cycle"].by_key()["color"][0]

        # Select epoch
        det_mask = self.detection_mask
        ul_mask = self.non_detection_mask

        # Plot detections
        if det_mask.any():
            for i in np.where(det_mask)[0]:
                axes.errorbar(
                    self.freq[i].to_value(self.freq.unit),
                    self.flux_density[i].to_value(self.flux_density.unit),
                    yerr=self.flux_density_error[i].to_value(self.flux_density_error.unit),
                    color=color,
                    label=label if i == np.where(det_mask)[0][0] else None,
                    **detection_style,
                )

        # Plot upper limits
        if show_upper_limits and ul_mask.any():
            for i in np.where(ul_mask)[0]:
                axes.errorbar(
                    self.freq[i].to_value(self.freq.unit),
                    self.flux_upper_limit[i].to_value(self.flux_upper_limit.unit),
                    yerr=0.2 * self.flux_upper_limit[i].to_value(self.flux_upper_limit.unit),
                    uplims=True,
                    color=color,
                    **upper_limit_style,
                )

        return fig, axes


# ---------------------------------------------------------------------------
# Deprecated alias — will be removed in a future release
# ---------------------------------------------------------------------------


class RadioPhotometryEpochContainer(RadioPhotometryEpoch):
    """Deprecated alias for :class:`RadioPhotometryEpoch`. Use that name instead."""

    def __init__(self, *args, **kwargs):
        _warnings.warn(
            "RadioPhotometryEpochContainer is deprecated and will be removed in a future release. "
            "Use RadioPhotometryEpoch instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
