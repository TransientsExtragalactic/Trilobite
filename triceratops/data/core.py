"""Generic / core data container structures for Triceratops.

These containers form the backbone of data handling in Triceratops, providing
a structured, validated interface to observational and synthetic datasets.
They enforce schema compliance, unit consistency, and immutability, ensuring
robustness and reproducibility throughout the modeling and inference pipeline.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from astropy import units as u
from astropy.table import Table

from triceratops.utils.log import triceratops_logger

if TYPE_CHECKING:
    from triceratops.models.core.base import Model  # noqa: F401

__all__ = ["InferenceData", "DataContainer", "Observable", "XYDataContainer"]


# ====================================================================== #
# Base Data Container
# ====================================================================== #
class DataContainer(ABC):
    """
    Abstract base class for validated, immutable data containers in Triceratops.

    A :class:`DataContainer` provides a **structured, unit-aware, read-only interface**
    to observational or synthetic datasets used throughout the Triceratops modeling
    and inference pipeline. It acts as a *boundary object* between raw input data
    (typically stored as :class:`astropy.table.Table` instances) and downstream
    numerical, statistical, and physical modeling components.

    The core responsibilities of a DataContainer are to:

    - enforce a **well-defined schema** via the class-level :attr:`COLUMNS` specification,
    - validate column presence, dtypes, and physical units,
    - provide safe, immutable access to the underlying data,
    - and support conversion to NumPy arrays for performance-critical backends
      such as likelihood evaluation and samplers.

    Unlike raw Astropy tables, DataContainer instances are **immutable by design**: all
    accessors return copies of the underlying data to prevent accidental modification
    during analysis or inference. This design choice ensures reproducibility and avoids
    subtle bugs in long-running sampling workflows.

    Subclasses must define a concrete schema by overriding the :attr:`COLUMNS` class
    attribute and implement the :meth:`from_table` constructor to handle any
    domain-specific preprocessing (e.g., detection logic, epoch grouping, masking,
    or derived quantities).

    Design principles
    -----------------
    - **Schema-driven**:
      Each subclass declares a formal column schema describing required and optional
      columns, expected dtypes, and physical units. Validation occurs eagerly at
      construction time.

    - **Unit-aware**:
      Physical quantities are validated and exposed using Astropy units. Missing but
      expected units are assigned with a warning; incompatible units raise errors.

    - **Immutable**:
      The underlying table cannot be modified in place. All slicing and accessors
      return copies.

    - **Inference-friendly**:
      Containers support NumPy coercion (``np.asarray(container)``) and explicit CGS
      conversion via :meth:`to_cgs_array` for efficient numerical evaluation.

    Role in the Triceratops pipeline
    --------------------------------
    DataContainer objects are typically consumed by likelihood classes, which
    define how model predictions are statistically compared to data. Models and
    samplers **never interact with raw tables directly**—all data access flows
    through a validated container.

    This abstraction allows Triceratops to support a wide range of data modalities
    (e.g., photometry, spectra, time series, generic ``(x, y)`` datasets) while
    maintaining a consistent and robust inference API.

    Notes
    -----
    - This class is abstract and cannot be instantiated directly.
    - Subclasses should call ``super().__init__`` to ensure schema validation.
    - Equality comparisons compare the underlying tables.
    - This class deliberately does **not** subclass :class:`astropy.table.Table`
      in order to strictly control mutability and invariants.

    See Also
    --------
    triceratops.inference.likelihood.base.Likelihood
        Likelihood classes that consume DataContainer instances.
    astropy.table.Table
        Underlying data structure used for storage.
    """

    # ========================= SCHEMA DEFINITION ========================= #
    # This ``COLUMNS`` dictionary contains the core schema requirements for the input
    # table in order to ensure that the radio photometry input is valid. This is then
    # enforced in ``_validate_table``.
    #
    # DEVELOPERS: YOU NEED TO OVERWRITE THIS!
    COLUMNS = [
        {
            "name": "COLUMN_NAME",
            "dtype": str,
            "description": "Description of the column.",
            "unit": None,
            "required": True,
        }
    ]

    # ========================= SPECIAL COLUMNS =========================== #
    # In subclasses in this module, it is common to see this section filled by
    # declarations about certain special columns (i.e. X_COL = ..., Y_COL = ...).
    # This is to facilitate easy access to these columns in downstream code.
    #
    # In this base class, we do not define any special columns.

    # ====================================================================== #
    # Initialization and validation methods
    # ====================================================================== #
    def __init__(self, table: Table, **kwargs):
        """
        Instantiate the data container.

        This method should be overridden in subclasses to ensure proper behavior
        downstream. The core responsibilities of this class are to

        1. Assign the ``self.__table__`` attribute to a :class:`~astropy.table.Table` instance.
        2. Validate that the table is consistent with the schema.
        3. Any additional configuration / setup required by the subclass.

        In the default implementation, we assign the table and offload to the
        ``self._validate_table`` method for validation. New implementations should
        call ``super().__init__()`` to ensure these responsibilities are fulfilled and
        then add any additional setup required.

        Parameters
        ----------
        table: ~astropy.table.Table
            The data table to be contained. This must conform to the schema defined in the
            :attr:`COLUMNS`` class attribute.
        kwargs:
            Additional keyword arguments for subclass-specific configuration.
        """
        # Assign the table to the data class.
        if not isinstance(table, Table):
            raise TypeError("The 'table' parameter must be an instance of astropy.table.Table.")

        # Hand off to the validation method. DEVELOPERS: you should modify validation by
        # intercepting in your subclass's ``_validate_table`` method.
        self.__table__: Table = self._validate_table(table.copy())

        # Proceed with any additional management.

    def _validate_table(self, table: Table) -> Table:
        """
        Validate that the provided table conforms to the schema defined in the :attr:`COLUMNS` class attribute.

        This method should be overridden in subclasses to implement specific validation
        logic. The default implementation checks for the presence of required columns.

        Parameters
        ----------
        table: ~astropy.table.Table
            The data table to validate.

        Returns
        -------
        ~astropy.table.Table
            The validated data table.

        Raises
        ------
        ValueError
            If the table does not conform to the schema.
        """
        for column in self.__class__.COLUMNS:
            # Extract necessary fields.
            name = column["name"]
            required = column.get("required", False)

            # Check for required columns.
            if required and name not in table.colnames:
                raise ValueError(f"Missing required column '{name}' in input table.")

            # If this column just isn't in the table, we can break
            # out now. This can happen for non-required columns.
            if name not in table.colnames:
                continue

            # Now ensure that we can coerce to the correct dtype
            # before we proceed
            expected_dtype = column.get("dtype", None)
            if expected_dtype is not None:
                try:
                    table[name] = table[name].astype(expected_dtype)
                except Exception as e:
                    raise ValueError(
                        f"Column '{name}' cannot be coerced to expected dtype '{expected_dtype}': {e}"
                    ) from e

            # Now handle the units. For units, a "" is a specifically dimensionless unit,
            # while "None" means we just don't care what unit is on the data.
            expected_unit = column.get("unit", None)
            if expected_unit is not None:
                # We have an expected unit that we're now going to check
                # for compatibility with the underlying data.
                expected_unit = u.Unit(expected_unit)

                col = table[name]
                if col.unit is None:
                    triceratops_logger.warning(
                        f"Column '{name}' has no unit. Assigning expected unit '{expected_unit}'."
                    )
                    col.unit = expected_unit
                elif not col.unit.is_equivalent(expected_unit):
                    raise u.UnitsError(
                        f"Column '{name}' has unit '{col.unit}', "
                        f"which is not compatible with expected unit '{expected_unit}'."
                    )

                # We have no reason to believe the conversion will fail. Attempt
                # to execute the conversion.
                try:
                    table[name] = table[name].to(expected_unit)
                except Exception as e:
                    raise ValueError(f"Column '{name}' cannot be assigned expected unit '{expected_unit}': {e}") from e

        return table

    # ====================================================================== #
    # Dunder Methods / Specialty Methods
    # ====================================================================== #
    # The scheme for dunder implementation in this class is as follows:
    #
    # - Effectively, we delegate everything to the underlying Astropy Table, this
    #   ensures users are not caught off-guard by missing functionality.
    # - However, we need to set a high __array_priority__ to ensure that
    #   operations involving both RadioPhotometryContainer and numpy arrays
    #   defer to the container's methods.
    # - We also implement __eq__ to allow for equality comparisons between
    #   two RadioPhotometryContainer instances.
    # - Finally, we are now an immutable container, so we do not implement
    #   any methods that would allow mutation of the underlying table.
    __array_priority__ = 1000

    def __len__(self):
        return self.__table__.__len__()

    def __getitem__(self, item):
        result = self.__table__.__getitem__(item)
        if isinstance(result, Table):
            return result.copy()
        else:
            return result.copy()

    def __iter__(self):
        return iter(self.__table__.copy())

    def __str__(self):
        return str(self.__table__)

    def __repr__(self):
        return f"<{self.__class__.__name__} n_obs={len(self)}>"

    def __contains__(self, item):
        return self.__table__.__contains__(item)

    def __bool__(self):
        return len(self) > 0

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.__table__ == other.__table__

    def __array__(self):
        """
        Convert the container to a NumPy structured array.

        This method is invoked by NumPy when coercing the container via
        ``np.asarray(container)``. Only columns marked as ``required=True`` in the
        schema and having numerical dtypes are included. Values are returned in the
        units specified by the schema (i.e. *not* automatically converted to CGS).

        The returned array is a copy and does not share writable memory with the
        internal table.

        Returns
        -------
        numpy.ndarray
            Structured NumPy array containing required numerical columns in schema
            units.
        """
        # Allocate a list to store each of the subarrays as we process them.
        _array_items = []

        # For each column in the dataset, we're going to go and
        # process it to a numerical array if it's required.
        for colspec in self.__class__.COLUMNS:
            if not colspec.get("required", False):
                continue

            name = colspec["name"]
            dtype = colspec["dtype"]

            # Skip non-numerical required columns (e.g. strings, identifiers)
            if not np.issubdtype(np.dtype(dtype), np.number):
                continue

            col = self.__table__[name]

            if col.unit is not None:
                values = col.quantity.to_value(col.unit)
            else:
                values = np.asarray(col)

            _array_items.append(values)

        return np.stack(_array_items, axis=-1)

    # ====================================================================== #
    # Properties
    # ====================================================================== #
    @property
    def table(self) -> Table:
        """
        Return a copy of the underlying Astropy table.

        This prevents mutation of the internal container state.
        """
        return self.__table__.copy()

    @property
    def size(self) -> int:
        """int: The number of observations in the container."""
        return len(self)

    # ====================================================================== #
    # Utility Methods
    # ====================================================================== #
    def copy(self):
        """Create a copy of this container."""
        return self.__class__(self.__table__.copy())

    def to_cgs_array(self):
        """
        Convert the container to a NumPy array, converted to CGS base units.

        This method mirrors ``__array__`` but explicitly converts all unit-bearing
        columns to their CGS equivalents. The output is intended for numerical backends such as
        likelihood evaluation and samplers.

        The returned array is a copy and does not share writable memory with the
        internal table.

        Returns
        -------
        numpy.ndarray
            Dense NumPy array of shape (n_obs, n_required_numeric_columns) with all
            values expressed in CGS base units.
        """
        _array_items = []

        for colspec in self.__class__.COLUMNS:
            if not colspec.get("required", False):
                continue

            name = colspec["name"]
            dtype = colspec["dtype"]

            # Skip non-numerical required columns
            if not np.issubdtype(np.dtype(dtype), np.number):
                continue

            col = self.__table__[name]

            if col.unit is not None:
                # Convert to CGS base units
                values = col.quantity.to(u.Unit(col.unit).cgs).value
            else:
                values = np.asarray(col)

            _array_items.append(np.asarray(values))

        return np.stack(_array_items, axis=-1)

    @classmethod
    @abstractmethod
    def from_table(cls, table: Table, **kwargs):
        """
        Create a DataContainer from an Astropy Table.

        This is an abstract method that must be implemented by subclasses.

        Parameters
        ----------
        table : ~astropy.table.Table
            The input table containing the data.
        kwargs:
            Additional keyword arguments for subclass-specific configuration.

        Returns
        -------
        DataContainer
            An instance of the subclass containing the data from the table.
        """
        pass

    @classmethod
    def from_file(cls, path: Union[str, Path], read_kws: dict = None, **kwargs):
        """
        Create a DataContainer from a FITS file.

        Parameters
        ----------
        path : str or pathlib.Path
            The file path to the FITS file containing the data.
        read_kws : dict
            Additional keyword arguments to pass to :meth:`astropy.table.Table.read`.
        kwargs:
            Additional keyword arguments to pass to :meth:`from_table`.

        Returns
        -------
        DataContainer
            An instance of the subclass containing the data from the FITS file.
        """
        # Coerce the read kwargs.
        if read_kws is None:
            read_kws = {}

        # Read the table from the file.
        table = Table.read(path, **read_kws)
        return cls.from_table(table, **kwargs)

    def to_file(self, path: Union[str, Path], write_kws: dict = None, **_):
        """
        Write the RadioPhotometryContainer's table to a file.

        Parameters
        ----------
        path : str or pathlib.Path
            The file path where the FITS file will be saved.
        write_kws : dict
            Additional keyword arguments to pass to :meth:`astropy.table.Table.write`.
        **kwargs:
            Additional keyword arguments to be used in subclasses.
        """
        if write_kws is None:
            write_kws = {}
        self.__table__.write(path, **write_kws)

    # ====================================================================== #
    # Converter Methods
    # ====================================================================== #
    def to_inference_data(
        self,
        model: "Model",
        variables: Optional[dict[str, tuple[str, ...]]] = None,
        observables: Optional[dict[str, tuple[str, ...]]] = None,
    ) -> "InferenceData":
        """
        Convert this :class:`DataContainer` into an :class:`InferenceData` instance.

        This method provides a high-level interface for preparing data for use
        in the inference pipeline. Internally, it delegates all validation,
        column resolution, and unit coercion to
        :meth:`InferenceData.from_table`.

        The conversion process is guided entirely by the provided ``model``:

        - Model-declared variable names determine required independent variables.
        - Model-declared output names determine required observables.
        - Model-declared units are used for automatic unit coercion.
        - Column names are matched by default using exact name equality.

        Optional overrides may be supplied to manually map model variables
        or observables to specific table columns.

        Parameters
        ----------
        model : ~models.core.base.Model
            The model against which inference will be performed.
            This determines:

            - Required independent variable names
            - Required observable names
            - Expected units for each quantity

        variables : dict[str, tuple[str, ...]] or str, optional
            Manual overrides for independent variable column resolution.

            Keys must match model variable names.

            Each value may be specified as:

            - ``"column_name"``
            - ``("column", "error")``
            - ``("column", "error", "upper", "lower")``

            If not provided, default column names are assumed to match
            the model variable names directly.

        observables : dict[str, tuple[str, ...]] or str, optional
            Manual overrides for observable column resolution.

            Keys must match model observable names.

            Same tuple formats as ``variables``.

        Returns
        -------
        InferenceData
            A fully validated, immutable, inference-ready dataset.

        Raises
        ------
        ValueError
            If required columns are missing or shapes are inconsistent.

        UnitsError
            If table column units cannot be coerced to model-declared units.

        Notes
        -----
        - No transformations (e.g., log-scaling) are applied.
        - No statistical assumptions are imposed.
        - All validation logic is centralized in :class:`InferenceData`.
        - This method is intentionally thin and deterministic.

        See Also
        --------
        InferenceData.from_table :
            Lower-level table ingestion logic.
        InferenceData.from_arrays :
            Direct array-based ingestion.
        """
        return InferenceData.from_table(
            model=model,
            table=self.table,
            variables=variables,
            observables=observables,
        )


# ====================================================================== #
# Inference Data Structures
# ====================================================================== #
@dataclass(frozen=True)
class Observable:
    """
    Immutable numerical representation of a single dependent variable used in statistical inference.

    An :class:`Observable` encapsulates a measured quantity (e.g. flux,
    magnitude, polarization) along with optional uncertainty and
    censoring information. This object is tied into a single
    :class:`InferenceData` instance, which may contain multiple named observables
    (e.g., flux and polarization) that are jointly modeled.

    This object is *purely numerical* and contains no units or
    domain-specific semantics. All unit conversion and validation
    should be performed upstream in reference to the generating
    model or data container. This allows likelihood implementations to
    interpret the data according to the needs of the problem at hand while
    being agnostic about the physical meaning of the observable.

    Parameters
    ----------
    value : np.ndarray
        Measured values of the observable. This should be provided
        as a ``(n,)`` array where ``n`` is the number of observations.
    error : np.ndarray, optional
        Symmetric 1-sigma uncertainties. Must be finite. Should be
        the same shape as ``value``. Use ``None`` if not provided.
    upper : np.ndarray, optional
        Upper limit values. Use NaN where not applicable.
    lower : np.ndarray, optional
        Lower limit values. Use NaN where not applicable.

    Notes
    -----
    - To indicate non-detections, provide finite values in the ``upper`` column
      and NaN in the ``value`` column. The likelihood can then interpret
      these as upper limits. Likewise, detections should be represented as
      finite values in the ``value`` column and NaN in the ``upper`` column.
    """

    value: np.ndarray
    error: Optional[np.ndarray] = None
    upper: Optional[np.ndarray] = None
    lower: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Initialization / Validation
    # ------------------------------------------------------------------
    # At this stage, we simply need to ensure that the
    # resulting object is correctly generated and does not
    # have any foundational issues.
    def __post_init__(self):
        # Coerce the input object to an array so that we
        # are guaranteed to have a consistent interface for the rest of the code.
        value = np.asarray(self.value)
        object.__setattr__(self, "value", value)

        # Compute the shape of ``value`` and use it to coerce
        # the shapes of the other arrays. This ensures that we have a consistent shape across all
        # arrays and that we can easily check for consistency.
        shape = value.shape

        if self.error is not None:
            err = np.asarray(self.error)
            if err.shape != shape:
                raise ValueError(f"error must have shape {shape}, got {err.shape}")
            if not np.all(np.isfinite(err)):
                raise ValueError(
                    "Error must contain only finite values. For non-detections, use the upper limit column."
                )
            object.__setattr__(self, "error", err)

        if self.upper is not None:
            upper = np.asarray(self.upper)
            if upper.shape != shape:
                raise ValueError(f"upper must have shape {shape}, got {upper.shape}")
            object.__setattr__(self, "upper", upper)

        if self.lower is not None:
            lower = np.asarray(self.lower)
            if lower.shape != shape:
                raise ValueError(f"lower must have shape {shape}, got {lower.shape}")
            object.__setattr__(self, "lower", lower)

        # Ensure that logical consistency is maintained throughout the
        # dataset. This includes checking for detection / non-detection overlap
        # and ensuring that upper limits do not exceed lower limits.

        # Check for upper limit and lower limit overlap
        if self.upper is not None:
            if np.any(~np.isnan(self.upper) & ~np.isnan(self.value)):
                raise ValueError(
                    "Upper limits cannot be finite where values are finite. Use NaN in the upper column for detections."
                )
        elif self.lower is not None:
            if np.any(~np.isnan(self.lower) & ~np.isnan(self.value)):
                raise ValueError(
                    "Lower limits cannot be finite where values are finite. Use NaN in the lower column for detections."
                )

        # Check that the upper and lower limits do not exceed one
        # another.
        if self.upper is not None and self.lower is not None:
            both = np.isfinite(self.upper) & np.isfinite(self.lower)
            if np.any(both & (self.lower > self.upper)):
                raise ValueError("Lower limit cannot exceed upper limit.")

    # ------------------------------------------------------------------
    # Dunder Methods
    # ------------------------------------------------------------------
    __array_priority__ = 1000  # ensure numpy defers properly

    def __repr__(self) -> str:
        return f"<Observable shape={self.shape} error={self.has_error} censoring={self.has_censoring}>"

    def __str__(self) -> str:
        return (
            f"Observable(n={self.size}, "
            f"error={self.has_error}, "
            f"upper_limits={self.has_upper_limits}, "
            f"lower_limits={self.has_lower_limits})"
        )

    def __len__(self) -> int:
        return self.size

    def __bool__(self) -> bool:
        """Truthiness defined as non-empty."""
        return self.size > 0

    def __array__(self, dtype=None) -> np.ndarray:
        """
        NumPy coercion returns the observable values.

        This allows:
            np.asarray(observable)
        """
        if dtype is not None:
            return np.asarray(self.value, dtype=dtype)
        return np.asarray(self.value)

    def __iter__(self):
        """Iterate over values."""
        return iter(self.value)

    def __getitem__(self, item):
        """
        Return a sliced Observable.

        This preserves error and censoring information.
        """
        return Observable(
            value=self.value[item],
            error=None if self.error is None else self.error[item],
            upper=None if self.upper is None else self.upper[item],
            lower=None if self.lower is None else self.lower[item],
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, Observable):
            return NotImplemented

        def _arr_equal(a, b):
            if a is None and b is None:
                return True
            if a is None or b is None:
                return False
            return np.array_equal(a, b, equal_nan=True)

        return (
            _arr_equal(self.value, other.value)
            and _arr_equal(self.error, other.error)
            and _arr_equal(self.upper, other.upper)
            and _arr_equal(self.lower, other.lower)
        )

    # ------------------------------------------------------------------
    # Derived Masks
    # ------------------------------------------------------------------
    @property
    def upper_mask(self) -> Optional[np.ndarray]:
        """Boolean mask of upper-censored points."""
        if self.upper is None:
            return None
        return np.isfinite(self.upper)

    @property
    def lower_mask(self) -> Optional[np.ndarray]:
        """Boolean mask of lower-censored points."""
        if self.lower is None:
            return None
        return np.isfinite(self.lower)

    # ------------------------------------------------------------------
    # Convenience Properties
    # ------------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the observable."""
        return self.value.shape

    @property
    def size(self) -> int:
        """Number of observations."""
        return self.value.size

    @property
    def has_error(self) -> bool:
        """True if uncertainties are provided."""
        return self.error is not None

    @property
    def has_upper_limits(self) -> bool:
        """True if any upper-censored points exist."""
        return self.upper is not None and np.any(self.upper_mask)

    @property
    def has_lower_limits(self) -> bool:
        """True if any lower-censored points exist."""
        return self.lower is not None and np.any(self.lower_mask)

    @property
    def has_censoring(self) -> bool:
        """True if any censoring exists."""
        return self.has_upper_limits or self.has_lower_limits


@dataclass(frozen=True)
class InferenceData:
    """
    An immutable, validated representation of data for use in the inference pipeline.

    Given any inference workflow with a model (:mod:`models`), and a data container
    from :mod:`data`, the :class:`InferenceData` class serves as the standardized interface
    which combines the two into an object universally understood by the inference
    pipeline.

    The :class:`InferenceData` object contains a set of
    dependent variables (Observables) and independent variables (x) that are used in the inference process.

    Parameters
    ----------
    x : dict[str, np.ndarray]
        Mapping from model variable names to arrays.
        All arrays must share the same shape.

    observables : dict[str, Observable]
        Mapping from observable names to :class:`Observable` objects.
        Each observable must match the shape of `x`.

    x_error : dict[str, np.ndarray], optional
        Uncertainties on independent variables.
        Must match shape of corresponding x arrays.

    x_upper : dict[str, np.ndarray], optional
        Upper limits on independent variables.

    x_lower : dict[str, np.ndarray], optional
        Lower limits on independent variables.

    Notes
    -----
    Not all likelihoods are compatible with all specifications for the :class:`InferenceData`
    class. It is worthwhile to check your likelihood's documentation to ensure that your data
    is complete before attempting to run inference. For example, some likelihoods may not be able to handle
    independent variable uncertainties or censoring, while others may require them for proper operation.
    """

    # ----------------------------------------------------------------
    # Parameters
    # ----------------------------------------------------------------
    x: dict[str, np.ndarray]
    observables: dict[str, Observable]
    x_error: Optional[dict[str, np.ndarray]] = None
    x_upper: Optional[dict[str, np.ndarray]] = None
    x_lower: Optional[dict[str, np.ndarray]] = None

    # ------------------------------------------------------------------
    # Initialization / Validation
    # ------------------------------------------------------------------
    def __post_init__(self):
        # Validate the independent variable exists and is in the
        # correct format before doing anything else.
        if not isinstance(self.observables, dict):
            raise TypeError(f"Parameter `x` of InferenceData must be a dictionary, not type {type(self.x)}.")
        elif len(self.observables) < 1:
            raise ValueError("InferenceData must contain at least one independent variable in the `x` parameter.")

        # Coerce all of the x-arrays so that they are guaranteed to be numpy arrays and have a consistent shape.
        # This also allows us to easily check for consistency across the different x-arrays and ensure that they all
        # have the same shape, which is a requirement for the inference process to work properly.
        x_clean = {}
        shapes = []

        for name, arr in self.x.items():
            arr = np.asarray(arr)
            x_clean[name] = arr
            shapes.append(arr.shape)

        # Check that everything is the same shape.
        if len(set(shapes)) > 1:
            raise ValueError(f"All independent variable arrays in `x` must have the same shape. Found shapes: {shapes}")
        base_shape = shapes[0]

        # We now validate that the x_err, x_lower, and x_upper are properly managed
        # and are all the same shape.
        # Validate optional x_error
        if self.x_error is not None:
            x_error_clean = {}
            for name, arr in self.x_error.items():
                arr = np.asarray(arr)

                if name not in x_clean:
                    raise ValueError(f"x_error provided for unknown variable '{name}'.")

                if arr.shape != base_shape:
                    raise ValueError(f"x_error for '{name}' must have shape {base_shape}.")

                x_error_clean[name] = arr
        else:
            x_error_clean = None

        # Validate optional x_upper
        if self.x_upper is not None:
            x_upper_clean = {}
            for name, arr in self.x_upper.items():
                arr = np.asarray(arr)

                if name not in x_clean:
                    raise ValueError(f"x_upper provided for unknown variable '{name}'.")

                if arr.shape != base_shape:
                    raise ValueError(f"x_upper for '{name}' must have shape {base_shape}.")

                x_upper_clean[name] = arr
        else:
            x_upper_clean = None

        # Validate optional x_lower
        if self.x_lower is not None:
            x_lower_clean = {}
            for name, arr in self.x_lower.items():
                arr = np.asarray(arr)

                if name not in x_clean:
                    raise ValueError(f"x_lower provided for unknown variable '{name}'.")

                if arr.shape != base_shape:
                    raise ValueError(f"x_lower for '{name}' must have shape {base_shape}.")

                x_lower_clean[name] = arr
        else:
            x_lower_clean = None

        # Set all the attributes
        object.__setattr__(self, "x", MappingProxyType(x_clean))
        object.__setattr__(self, "x_error", None if x_error_clean is None else MappingProxyType(x_error_clean))
        object.__setattr__(self, "x_upper", None if x_upper_clean is None else MappingProxyType(x_upper_clean))
        object.__setattr__(self, "x_lower", None if x_lower_clean is None else MappingProxyType(x_lower_clean))

        # Ensure that the observables are all in the correct format. These must all be Observable
        # objects and we require that their shape matches that of the base shape.
        if not self.observables:
            raise ValueError("InferenceData must contain at least one observable.")

        obs_clean = {}

        for name, obs in self.observables.items():
            if not isinstance(obs, Observable):
                raise TypeError(f"Observable '{name}' must be an Observable instance.")

            if obs.shape != base_shape:
                raise ValueError(
                    f"Observable '{name}' shape {obs.shape} does not match independent variable shape {base_shape}."
                )

            obs_clean[name] = obs

        # Replace internal state with read-only versions
        object.__setattr__(self, "observables", MappingProxyType(obs_clean))

    # ------------------------------------------------------------------
    # Dunder Methods
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return f"<InferenceData shape={self.shape} n_observables={self.n_observables}>"

    def __str__(self) -> str:
        return (
            f"InferenceData(n={self.size}, "
            f"variables={list(self.x.keys())}, "
            f"observables={list(self.observables.keys())})"
        )

    def __len__(self) -> int:
        return self.size

    def __eq__(self, other):
        if not isinstance(other, InferenceData):
            return NotImplemented
        return self.x == other.x and self.observables == other.observables

    # ------------------------------------------------------------------
    # Convenience Properties
    # ------------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the dataset."""
        return next(iter(self.x.values())).shape

    @property
    def size(self) -> int:
        """Total number of data points."""
        return int(np.prod(self.shape))

    @property
    def variable_names(self) -> list[str]:
        """List of independent variable names."""
        return list(self.x.keys())

    @property
    def observable_names(self) -> list[str]:
        """List of observable names."""
        return list(self.observables.keys())

    @property
    def n_observables(self) -> int:
        """Number of dependent variables."""
        return len(self.observables)

    # ------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------
    def get_observable(self, name: str) -> Observable:
        """Return a specific observable by name."""
        try:
            return self.observables[name]
        except KeyError as exp:
            raise KeyError(f"Observable '{name}' not found.") from exp

    def has_x_uncertainty(self) -> bool:
        """
        Determine if independent variable uncertainties are provided.

        Returns
        -------
        bool
            True if independent variable uncertainties are provided, False otherwise.
        """
        return self.x_error is not None

    def get_x_array(self, flatten: bool = False) -> np.ndarray:
        """
        Return stacked independent-variable values.

        This method stacks all independent-variable arrays stored in
        :attr:`x` into a single NumPy array.

        The stacking order follows :attr:`variable_names`, which must
        match the model's expected input ordering.

        Parameters
        ----------
        flatten : bool, optional
            If False (default), returns array with shape:
                (*base_shape, n_variables)

            If True, reshapes to:
                (n_points, n_variables),
            where n_points = product(base_shape).

        Returns
        -------
        np.ndarray
            Stacked independent-variable array.

        Notes
        -----
        - Always returns a numeric array.
        - Raises if any required variable is missing.
        - No unit conversion is performed.
        """
        arrays = [self.x[name] for name in self.variable_names]
        stacked = np.stack(arrays, axis=-1)

        if flatten:
            stacked = stacked.reshape(-1, stacked.shape[-1])

        return stacked

    def get_y_array(self, flatten: bool = False) -> np.ndarray:
        """
        Return stacked observable values.

        All observable values are required and must be numeric.

        Parameters
        ----------
        flatten : bool, optional
            If False (default), returns:
                (*base_shape, n_observables)

            If True, reshapes to:
                (n_points, n_observables)

        Returns
        -------
        np.ndarray
            Stacked observable array.

        Notes
        -----
        - NaN values are preserved.
        - Observable order matches :attr:`observable_names`.
        """
        arrays = [self.observables[name].value for name in self.observable_names]

        stacked = np.stack(arrays, axis=-1)

        if flatten:
            stacked = stacked.reshape(-1, stacked.shape[-1])

        return stacked

    def get_y_error_array(self, flatten: bool = False) -> np.ndarray:
        """
        Return stacked observable uncertainties.

        All observables must define symmetric uncertainties.
        Raises if any observable lacks uncertainty information.

        Parameters
        ----------
        flatten : bool, optional
            Same reshaping semantics as :meth:`get_y_array`.

        Returns
        -------
        np.ndarray
            Stacked uncertainty array.

        Raises
        ------
        ValueError
            If any observable lacks uncertainties.
        """
        arrays = []

        for name in self.observable_names:
            err = self.observables[name].error
            if err is None:
                raise ValueError(f"Observable '{name}' lacks uncertainty.")
            arrays.append(err)

        stacked = np.stack(arrays, axis=-1)

        if flatten:
            stacked = stacked.reshape(-1, stacked.shape[-1])

        return stacked

    def get_y_lower_array(self, flatten: bool = False) -> np.ndarray:
        """
        Return stacked observable lower limits.

        Missing lower limits are represented as np.nan.

        Parameters
        ----------
        flatten : bool, optional
            Same reshaping semantics as :meth:`get_y_array`.

        Returns
        -------
        np.ndarray
            Lower-limit array.
        """
        base_shape = self.get_y_array(flatten=False).shape[:-1]

        arrays = []

        for name in self.observable_names:
            lower = self.observables[name].lower
            if lower is None:
                arrays.append(np.full(base_shape, np.nan))
            else:
                arrays.append(lower)

        stacked = np.stack(arrays, axis=-1)

        if flatten:
            stacked = stacked.reshape(-1, stacked.shape[-1])

        return stacked

    def get_y_upper_array(self, flatten: bool = False) -> np.ndarray:
        """
        Return stacked observable upper limits.

        Missing upper limits are represented as np.nan.

        Parameters
        ----------
        flatten : bool, optional
            Same reshaping semantics as :meth:`get_y_array`.

        Returns
        -------
        np.ndarray
            Upper-limit array.
        """
        base_shape = self.get_y_array(flatten=False).shape[:-1]

        arrays = []

        for name in self.observable_names:
            upper = self.observables[name].upper
            if upper is None:
                arrays.append(np.full(base_shape, np.nan))
            else:
                arrays.append(upper)

        stacked = np.stack(arrays, axis=-1)

        if flatten:
            stacked = stacked.reshape(-1, stacked.shape[-1])

        return stacked

    def get_x_error_array(self, flatten: bool = False) -> np.ndarray:
        """
        Return stacked independent-variable uncertainties.

        All independent variables must define symmetric uncertainties.
        If any variable lacks uncertainty information, a ValueError is raised.

        Parameters
        ----------
        flatten : bool, optional
            If False (default), returns shape:
                (*base_shape, n_variables)

            If True, reshapes to:
                (n_points, n_variables)

        Returns
        -------
        np.ndarray
            Stacked uncertainty array.

        Raises
        ------
        ValueError
            If uncertainties are not defined for all variables.
        """
        if self.x_error is None:
            raise ValueError("Independent-variable uncertainties are not defined.")

        arrays = []

        for name in self.variable_names:
            if name not in self.x_error or self.x_error[name] is None:
                raise ValueError(f"Independent-variable '{name}' lacks uncertainty.")
            arrays.append(self.x_error[name])

        stacked = np.stack(arrays, axis=-1)

        if flatten:
            stacked = stacked.reshape(-1, stacked.shape[-1])

        return stacked

    def get_x_upper_array(self, flatten: bool = False) -> np.ndarray:
        """
        Return stacked independent-variable upper limits.

        Missing upper limits are represented as np.nan.

        Parameters
        ----------
        flatten : bool, optional
            Same reshaping semantics as :meth:`get_x_array`.

        Returns
        -------
        np.ndarray
            Upper-limit array with shape:
                (*base_shape, n_variables)
        """
        base_shape = self.get_x_array(flatten=False).shape[:-1]

        arrays = []

        for name in self.variable_names:
            if self.x_upper is None or name not in self.x_upper or self.x_upper[name] is None:
                arrays.append(np.full(base_shape, np.nan))
            else:
                arrays.append(self.x_upper[name])

        stacked = np.stack(arrays, axis=-1)

        if flatten:
            stacked = stacked.reshape(-1, stacked.shape[-1])

        return stacked

    def get_x_lower_array(self, flatten: bool = False) -> np.ndarray:
        """
        Return stacked independent-variable lower limits.

        Missing lower limits are represented as np.nan.

        Parameters
        ----------
        flatten : bool, optional
            Same reshaping semantics as :meth:`get_x_array`.

        Returns
        -------
        np.ndarray
            Lower-limit array.
        """
        base_shape = self.get_x_array(flatten=False).shape[:-1]

        arrays = []

        for name in self.variable_names:
            if self.x_lower is None or name not in self.x_lower or self.x_lower[name] is None:
                arrays.append(np.full(base_shape, np.nan))
            else:
                arrays.append(self.x_lower[name])

        stacked = np.stack(arrays, axis=-1)

        if flatten:
            stacked = stacked.reshape(-1, stacked.shape[-1])

        return stacked

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    @classmethod
    def from_arrays(
        cls,
        model: "Model",
        x: dict[str, Union[np.ndarray, u.Quantity]],
        y: dict[str, Union[np.ndarray, u.Quantity]],
        y_err: Optional[dict[str, Union[np.ndarray, u.Quantity]]] = None,
        y_upper: Optional[dict[str, Union[np.ndarray, u.Quantity]]] = None,
        y_lower: Optional[dict[str, Union[np.ndarray, u.Quantity]]] = None,
        x_err: Optional[dict[str, Union[np.ndarray, u.Quantity]]] = None,
        x_upper: Optional[dict[str, Union[np.ndarray, u.Quantity]]] = None,
        x_lower: Optional[dict[str, Union[np.ndarray, u.Quantity]]] = None,
    ):
        """
        Construct an :class:`InferenceData` instance directly from raw array inputs.

        This is a convenience constructor for workflows where data are already
        available as numerical arrays (or `astropy.units.Quantity` objects)
        and are intended to be used directly for inference against a given model.

        The method performs:

        1. **Name validation**
           - All model-declared variables and observables must be present.
           - Extra keys are ignored.
           - Missing required entries raise immediately.

        2. **Unit coercion**
           - If a value is provided as a `Quantity`, it is converted to the
             units declared by the model.
           - If a value is a plain NumPy array, it is assumed to already be in
             model-expected units.
           - Unit incompatibilities raise a `ValueError`.

        3. **Shape validation**
           - All independent variables must share identical shape.
           - All observables must match the independent-variable shape.
           - Optional uncertainty/limit arrays must match the same shape.

        4. **Construction**
           - Observable objects are created internally.
           - The resulting :class:`InferenceData` instance is fully validated
             and immutable.

        Parameters
        ----------
        model : Model
            The model defining:

            - Required independent variable names
            - Required observable names
            - Expected base units for each variable and observable

        x : dict[str, np.ndarray or Quantity]
            Mapping from model variable names to arrays.
            All arrays must share the same shape.

        y : dict[str, np.ndarray or Quantity]
            Mapping from model observable names to arrays.
            All arrays must match the shape of `x`.

        y_err : dict[str, np.ndarray or Quantity], optional
            Symmetric uncertainties for observables.
            Must match observable shape.

        y_upper : dict[str, np.ndarray or Quantity], optional
            Upper limits for observables.
            Must match observable shape.

        y_lower : dict[str, np.ndarray or Quantity], optional
            Lower limits for observables.
            Must match observable shape.

        x_err : dict[str, np.ndarray or Quantity], optional
            Symmetric uncertainties for independent variables.
            Must match independent-variable shape.

        x_upper : dict[str, np.ndarray or Quantity], optional
            Upper limits for independent variables.

        x_lower : dict[str, np.ndarray or Quantity], optional
            Lower limits for independent variables.

        Returns
        -------
        InferenceData
            Fully validated, immutable inference-ready dataset.

        Raises
        ------
        ValueError
            If:
            - Required variables or observables are missing
            - Shapes are inconsistent
            - Units cannot be coerced to model-declared units

        Notes
        -----
        - No statistical interpretation is applied.
        - No transformations (e.g., log-space conversions) are performed.
        - All arrays must already represent physical values in model units.
        - This method is intended for low-level or programmatic data ingestion.
        - For table-based workflows with automatic unit coercion and column
          resolution, use :meth:`DataContainer.to_inference_data` instead.

        Examples
        --------

        .. code-block:: python

            x = {"time": np.linspace(0, 10, 100)}
            y = {"flux": np.random.normal(0, 1, 100)}
            data = InferenceData.from_arrays(model, x=x, y=y)

        Using quantities:

        .. code-block:: python

            x = {"time": np.linspace(0, 10, 100) * u.day}
            y = {"flux": np.random.normal(0, 1, 100) * u.mJy}
            data = InferenceData.from_arrays(model, x=x, y=y)
        """
        variable_names = model.variable_names
        observable_names = list(model.OUTPUTS._fields)

        variable_units = {v.name: v.base_units for v in model.VARIABLES}
        observable_units = {o: getattr(model.UNITS, o) for o in observable_names}

        # -------------------------------------------------
        # Helper: validate dictionary of arrays
        # -------------------------------------------------
        def _validate_group(
            data_dict,
            expected_names,
            unit_map,
            group_name,
            required=True,
            expected_shape=None,
        ):
            # Check if the data object is required.
            if data_dict is None:
                if required:
                    raise ValueError(f"{group_name} must be provided.")
                return None

            # Begin the validation loop.
            validated = {}
            shapes = []

            for name in expected_names:
                if name not in data_dict:
                    raise ValueError(f"{group_name} missing required entry '{name}'.")

                unit = unit_map[name]

                try:
                    value = data_dict[name]
                    if isinstance(value, u.Quantity):
                        arr = np.asarray(value.to_value(unit))
                    else:
                        arr = np.asarray(value)
                except Exception as exc:
                    raise ValueError(f"Error converting '{name}' in {group_name} to expected unit '{unit}'.") from exc

                validated[name] = arr
                shapes.append(arr.shape)

            # All shapes must match
            if len(set(shapes)) > 1:
                raise ValueError(f"All arrays in {group_name} must share shape. Found shapes: {shapes}")

            _base_shape = shapes[0]

            if expected_shape is not None and _base_shape != expected_shape:
                raise ValueError(f"{group_name} must have shape {expected_shape}, got {_base_shape}")

            return validated

        # -------------------------------------------------
        # Validate required groups
        # -------------------------------------------------
        validated_x = _validate_group(
            x,
            variable_names,
            variable_units,
            "x",
            required=True,
        )

        base_shape = next(iter(validated_x.values())).shape

        validated_y = _validate_group(
            y,
            observable_names,
            observable_units,
            "y",
            required=True,
            expected_shape=base_shape,
        )

        # -------------------------------------------------
        # Validate optional groups
        # -------------------------------------------------
        validated_x_err = _validate_group(
            x_err,
            variable_names,
            variable_units,
            "x_err",
            required=False,
            expected_shape=base_shape,
        )

        validated_x_upper = _validate_group(
            x_upper,
            variable_names,
            variable_units,
            "x_upper",
            required=False,
            expected_shape=base_shape,
        )

        validated_x_lower = _validate_group(
            x_lower,
            variable_names,
            variable_units,
            "x_lower",
            required=False,
            expected_shape=base_shape,
        )

        validated_y_err = _validate_group(
            y_err,
            observable_names,
            observable_units,
            "y_err",
            required=False,
            expected_shape=base_shape,
        )

        validated_y_upper = _validate_group(
            y_upper,
            observable_names,
            observable_units,
            "y_upper",
            required=False,
            expected_shape=base_shape,
        )

        validated_y_lower = _validate_group(
            y_lower,
            observable_names,
            observable_units,
            "y_lower",
            required=False,
            expected_shape=base_shape,
        )

        # -------------------------------------------------
        # Build Observable objects
        # -------------------------------------------------
        observables = {
            name: Observable(
                value=validated_y[name],
                error=None if validated_y_err is None else validated_y_err[name],
                upper=None if validated_y_upper is None else validated_y_upper[name],
                lower=None if validated_y_lower is None else validated_y_lower[name],
            )
            for name in observable_names
        }

        # -------------------------------------------------
        # Construct InferenceData
        # -------------------------------------------------
        return cls(
            x=validated_x,
            observables=observables,
            x_error=validated_x_err,
            x_upper=validated_x_upper,
            x_lower=validated_x_lower,
        )

    @classmethod
    def from_table(
        cls,
        model: "Model",
        table,
        variables: Optional[dict[str, tuple[str, ...]]] = None,
        observables: Optional[dict[str, tuple[str, ...]]] = None,
    ):
        """
        Construct an :class:`InferenceData` object from an Astropy Table.

        This method extracts columns from a table, performs name resolution
        and unit coercion, and delegates validation to
        :meth:`InferenceData.from_arrays`.

        Parameters
        ----------
        model : Model
            Model defining required variable and observable names and units.

        table : astropy.table.Table
            Input data table.

        variables : dict[str, tuple or str], optional
            Manual column specifications for independent variables.

            Each key is a model variable name.

            Values may be:
            - "column_name"
            - ("column", "error")
            - ("column", "error", "upper", "lower")

        observables : dict[str, tuple or str], optional
            Manual column specifications for observables.
            Same format as `variables`.

        Returns
        -------
        InferenceData

        Raises
        ------
        ValueError
            If required columns are missing or incompatible.

        Notes
        -----
        - Column names default to model names if not overridden.
        - Columns must contain Quantity data or unit-compatible values.
        - All unit conversion and shape validation is handled by
          :meth:`from_arrays`.
        """
        variables = variables or {}
        observables = observables or {}

        variable_names = model.variable_names
        observable_names = list(model.OUTPUTS._fields)

        # ---------------------------------------------
        # Helper: normalize column specification
        # ---------------------------------------------
        def _coerce_spec(spec):
            if isinstance(spec, str):
                return spec, None, None, None
            elif isinstance(spec, tuple):
                if len(spec) == 2:
                    return spec[0], spec[1], None, None
                elif len(spec) == 4:
                    return spec
            raise ValueError(f"Invalid column specification: {spec}. Expected str, 2-tuple, or 4-tuple.")

        # ---------------------------------------------
        # Extract X columns
        # ---------------------------------------------
        x = {}
        x_err = {}
        x_upper = {}
        x_lower = {}

        for name in variable_names:
            col, err, upper, lower = _coerce_spec(
                variables.get(
                    name,
                    (name, f"{name}_err", f"{name}_upper", f"{name}_lower"),
                )
            )

            if col not in table.colnames:
                raise ValueError(f"Column '{col}' not found for variable '{name}'.")

            x[name] = table[col].quantity if hasattr(table[col], "quantity") else table[col]

            if err and err in table.colnames:
                x_err[name] = table[err].quantity if hasattr(table[err], "quantity") else table[err]

            if upper and upper in table.colnames:
                x_upper[name] = table[upper].quantity if hasattr(table[upper], "quantity") else table[upper]

            if lower and lower in table.colnames:
                x_lower[name] = table[lower].quantity if hasattr(table[lower], "quantity") else table[lower]

        # ---------------------------------------------
        # Extract Y columns
        # ---------------------------------------------
        y = {}
        y_err = {}
        y_upper = {}
        y_lower = {}

        for name in observable_names:
            col, err, upper, lower = _coerce_spec(
                observables.get(
                    name,
                    (name, f"{name}_err", f"{name}_upper", f"{name}_lower"),
                )
            )

            if col not in table.colnames:
                raise ValueError(f"Column '{col}' not found for observable '{name}'.")

            y[name] = table[col].quantity if hasattr(table[col], "quantity") else table[col]

            if err and err in table.colnames:
                y_err[name] = table[err].quantity if hasattr(table[err], "quantity") else table[err]

            if upper and upper in table.colnames:
                y_upper[name] = table[upper].quantity if hasattr(table[upper], "quantity") else table[upper]

            if lower and lower in table.colnames:
                y_lower[name] = table[lower].quantity if hasattr(table[lower], "quantity") else table[lower]

        # ---------------------------------------------
        # Delegate to canonical constructor
        # ---------------------------------------------
        return cls.from_arrays(
            model=model,
            x=x,
            y=y,
            x_err=x_err or None,
            x_upper=x_upper or None,
            x_lower=x_lower or None,
            y_err=y_err or None,
            y_upper=y_upper or None,
            y_lower=y_lower or None,
        )


# ====================================================================== #
# Generic Containers for (x, y) Data
# ====================================================================== #
class XYDataContainer(DataContainer, ABC):
    """
    Abstract base class for two-dimensional ``(x, y)`` datasets with optional uncertainties and censoring.

    An :class:`XYDataContainer` represents observational or synthetic data that
    can be expressed as pairs of an independent variable ``x`` and a dependent
    variable ``y``. This abstraction is intended to support a wide range of
    inference problems, including curve fitting, spectral modeling, and
    phenomenological regression, while remaining agnostic about the specific
    physical meaning of the axes.

    In addition to core ``(x, y)`` values, this container supports optional:

    - uncertainties on ``x`` and/or ``y``,
    - upper and lower limits (censoring) on either axis,
    - detection and non-detection masking derived from limit columns.

    All axis semantics are defined *declaratively* via class-level column
    attributes (e.g., :attr:`X_COLUMN`, :attr:`Y_ERROR_COLUMN`), allowing
    subclasses to specialize behavior without modifying downstream likelihood
    logic.

    This class does **not** define how uncertainties or limits are treated
    statistically; it only exposes them in a standardized, unit-aware form.
    Interpretation of errors and censoring is entirely delegated to likelihood
    implementations.

    Key features
    ------------
    - **Axis abstraction**:
    Independent and dependent variables are identified via class attributes
    rather than hard-coded column names.

    - **Optional uncertainties**:
    Support for symmetric uncertainties on ``x`` and/or ``y`` without
    imposing a statistical model.

    - **Censoring-aware**:
    Upper and lower limits on either axis are detected automatically and
    exposed through boolean masks.

    - **Unit-safe**:
    All values are validated and returned as Astropy quantities.

    - **Immutable**:
    Like all :class:`DataContainer` subclasses, this container is read-only.

    Detection and censoring model
    -----------------------------
    Censoring is inferred purely from the presence of *finite values* in
    declared limit columns:

    - Upper limits correspond to finite values in ``*_UPPER_LIMIT_COLUMN``.
    - Lower limits correspond to finite values in ``*_LOWER_LIMIT_COLUMN``.

    The container constructs independent boolean masks for:

    - x-axis upper limits,
    - x-axis lower limits,
    - y-axis upper limits,
    - y-axis lower limits,

    as well as composite masks indicating whether any limit applies to a given
    observation.

    No assumptions are made about how these limits should enter a likelihood
    (e.g., one-sided Gaussian, survival analysis, hard truncation).

    Intended use
    ------------
    :class:`XYDataContainer` is intended to serve as a *common interface* between
    diverse data products and inference machinery, including:

    - spectral energy distributions,
    - light curve slices at fixed frequency,
    - generic ``(x, y)`` curve-fitting problems,
    - censored regression with upper or lower limits.

    Subclasses should specialize this container by:

    - defining a concrete :attr:`COLUMNS` schema,
    - setting axis column attributes (``X_COLUMN``, ``Y_COLUMN``, etc.),
    - adding domain-specific convenience properties if needed.

    Notes
    -----
    - This class is abstract and cannot be instantiated directly.
    - Axis uncertainties are assumed to be symmetric unless otherwise
    interpreted by the likelihood.
    - If both upper and lower limits are present for the same axis and
    observation, the data point is considered interval-censored.

    See Also
    --------
    DataContainer
     Base class providing schema validation and immutability.
    triceratops.inference.likelihood.base.Likelihood
     Likelihood classes that interpret uncertainties and censoring.
    """

    # ========================= SCHEMA DEFINITION ========================= #
    # This ``COLUMNS`` dictionary contains the core schema requirements for the input
    # table in order to ensure that the radio photometry input is valid. This is then
    # enforced in ``_validate_table``.
    #
    # DEVELOPERS: YOU NEED TO OVERWRITE THIS!
    COLUMNS = [
        {
            "name": "COLUMN_NAME",
            "dtype": str,
            "description": "Description of the column.",
            "unit": None,
            "required": True,
        }
    ]

    # ========================= SPECIAL COLUMNS =========================== #
    # In subclasses in this module, it is common to see this section filled by
    # declarations about certain special columns (i.e. X_COL = ..., Y_COL = ...).
    # This is to facilitate easy access to these columns in downstream code.
    #
    # In this base class, we do not define any special columns.
    X_COLUMN = None
    """str: Name of the x-axis column.

    For :class:`XYDataContainer` subclasses, this should be overridden to specify
    the name of the column representing the x-axis data. In upstream processing,
    this is how the independent variable is identified.
    """
    Y_COLUMN = None
    """str: Name of the y-axis column.

    For :class:`XYDataContainer` subclasses, this should be overridden to specify
    the name of the column representing the y-axis data. In upstream processing,
    this is how the dependent variable is identified.
    """
    Y_ERROR_COLUMN = None
    """str: Name of the y-axis error column.

    This should be overridden to specify the name of the column representing
    the y-axis error data. This is generally the 1-sigma error, but that is
    left for the Likelihood function. If ``None``, then it is assumed that
    no y-axis errors are provided.
    """
    X_ERROR_COLUMN = None
    """str: Name of the x-axis error column.

    This should be overridden to specify the name of the column representing
    the x-axis error data. This is generally the 1-sigma error, but that is
    left for the Likelihood function. If ``None``, then it is assumed that
    no y-axis errors are provided.
    """
    Y_UPPER_LIMIT_COLUMN = None
    """str: Name of the y-axis upper limit column.

    This should be overridden to specify the name of the column representing
    the y-axis upper limit data. This is generally used to indicate non-detections.
    """
    Y_LOWER_LIMIT_COLUMN = None
    """str: Name of the y-axis lower limit column.

    This should be overridden to specify the name of the column representing
    the y-axis lower limit data. This is generally used to indicate non-detections.
    """
    X_UPPER_LIMIT_COLUMN = None
    """str: Name of the x-axis upper limit column.

    This should be overridden to specify the name of the column representing
    the x-axis upper limit data. This is generally used to indicate non-detections.
    """
    X_LOWER_LIMIT_COLUMN = None
    """str: Name of the x-axis lower limit column.

    This should be overridden to specify the name of the column representing
    the x-axis lower limit data. This is generally used to indicate non-detections.
    """

    # ====================================================================== #
    # Initialization and validation methods
    # ====================================================================== #
    def __init__(self, table: Table, **kwargs):
        """
        Instantiate the data container.

        This method should be overridden in subclasses to ensure proper behavior
        downstream. The core responsibilities of this class are to

        1. Assign the ``self.__table__`` attribute to a :class:`~astropy.table.Table` instance.
        2. Validate that the table is consistent with the schema.
        3. Generate limit masks.
        4. Any additional configuration / setup required by the subclass.

        In the default implementation, we assign the table and offload to the
        ``self._validate_table`` method for validation. New implementations should
        call ``super().__init__()`` to ensure these responsibilities are fulfilled and
        then add any additional setup required.

        Parameters
        ----------
        table: ~astropy.table.Table
            The data table to be contained. This must conform to the schema defined in the
            :attr:`COLUMNS`` class attribute.
        kwargs:
            Additional keyword arguments for subclass-specific configuration.
        """
        # Pass off to the super class to instantiate the
        # core data container functionality.
        super().__init__(table, **kwargs)

        # Now address the potential detection / non-detection logic.
        # This is done by checking for upper limit columns.
        (
            self.__x_ul_mask__,
            self.__x_ll_mask__,
            self.__y_ul_mask__,
            self.__y_ll_mask__,
        ) = self._construct_detection_mask()

    def _construct_detection_mask(self) -> np.ndarray:
        """
        Construct the detection mask based on upper limit columns.

        Returns
        -------
        numpy.ndarray
            Boolean array where True indicates a detection and False indicates
            a non-detection.

        Notes
        -----
        For each of X and Y, we first (a) check if we even have a column for the upper
        limit defined. If we do, we (b) check if that column is in the table. If it is, we (c)
        check if the value is finite. If it is finite, that indicates a non-detection
        in that axis. We combine the non-detection masks for both axes to get the final
        detection mask.
        """
        n_obs = len(self)
        _xumask, _xlmask, _yumask, _ylmask = (
            np.zeros(n_obs, dtype=bool),
            np.zeros(n_obs, dtype=bool),
            np.zeros(n_obs, dtype=bool),
            np.zeros(n_obs, dtype=bool),
        )

        # For each of the potential upper limit columns, we need to
        # check if they are defined and present in the table.
        if self.X_UPPER_LIMIT_COLUMN is not None:
            if self.X_UPPER_LIMIT_COLUMN in self.__table__.colnames:
                x_ul_col = self.__table__[self.X_UPPER_LIMIT_COLUMN]
                _xumask = np.isfinite(x_ul_col)

        if self.X_LOWER_LIMIT_COLUMN is not None:
            if self.X_LOWER_LIMIT_COLUMN in self.__table__.colnames:
                x_ll_col = self.__table__[self.X_LOWER_LIMIT_COLUMN]
                _xlmask = np.isfinite(x_ll_col)

        if self.Y_UPPER_LIMIT_COLUMN is not None:
            if self.Y_UPPER_LIMIT_COLUMN in self.__table__.colnames:
                y_ul_col = self.__table__[self.Y_UPPER_LIMIT_COLUMN]
                _yumask = np.isfinite(y_ul_col)

        if self.Y_LOWER_LIMIT_COLUMN is not None:
            if self.Y_LOWER_LIMIT_COLUMN in self.__table__.colnames:
                y_ll_col = self.__table__[self.Y_LOWER_LIMIT_COLUMN]
                _ylmask = np.isfinite(y_ll_col)

        return _xumask, _xlmask, _yumask, _ylmask

    # ====================================================================== #
    # Properties
    # ====================================================================== #
    @property
    def x(self) -> u.Quantity:
        """astropy.units.Quantity: The x-axis data."""
        if self.X_COLUMN is None:
            raise NotImplementedError(f"X_COLUMN is not defined in {self.__class__.__name__}.")

        col = self.__table__[self.X_COLUMN]
        return col.quantity.copy()

    @property
    def y(self) -> u.Quantity:
        """astropy.units.Quantity: The y-axis data."""
        if self.Y_COLUMN is None:
            raise NotImplementedError(f"Y_COLUMN is not defined in {self.__class__.__name__}.")

        col = self.__table__[self.Y_COLUMN]
        return col.quantity.copy()

    @property
    def y_error(self) -> Union[u.Quantity, None]:
        """astropy.units.Quantity or None: The y-axis error data, if provided."""
        if self.Y_ERROR_COLUMN is None:
            return None

        col = self.__table__[self.Y_ERROR_COLUMN]
        return col.quantity.copy()

    @property
    def y_upper_limit(self) -> Union[u.Quantity, None]:
        """astropy.units.Quantity or None: The y-axis upper limit data, if provided."""
        if self.Y_UPPER_LIMIT_COLUMN is None:
            return None

        col = self.__table__[self.Y_UPPER_LIMIT_COLUMN]
        return col.quantity.copy()

    def y_lower_limit(self) -> Union[u.Quantity, None]:
        """astropy.units.Quantity or None: The y-axis lower limit data, if provided."""
        if self.Y_LOWER_LIMIT_COLUMN is None:
            return None

        col = self.__table__[self.Y_LOWER_LIMIT_COLUMN]
        return col.quantity.copy()

    @property
    def x_error(self) -> Union[u.Quantity, None]:
        """astropy.units.Quantity or None: The x-axis error data, if provided."""
        if self.X_ERROR_COLUMN is None:
            return None

        col = self.__table__[self.X_ERROR_COLUMN]
        return col.quantity.copy()

    @property
    def x_upper_limit(self) -> Union[u.Quantity, None]:
        """astropy.units.Quantity or None: The x-axis upper limit data, if provided."""
        if self.X_UPPER_LIMIT_COLUMN is None:
            return None

        col = self.__table__[self.X_UPPER_LIMIT_COLUMN]
        return col.quantity.copy()

    @property
    def x_lower_limit(self) -> Union[u.Quantity, None]:
        """astropy.units.Quantity or None: The x-axis lower limit data, if provided."""
        if self.X_LOWER_LIMIT_COLUMN is None:
            return None

        col = self.__table__[self.X_LOWER_LIMIT_COLUMN]
        return col.quantity.copy()

    @property
    def x_upper_lim_mask(self) -> np.ndarray:
        """numpy.ndarray: Boolean mask indicating x-axis upper limits."""
        return self.__x_ul_mask__.copy()

    @property
    def x_lower_lim_mask(self) -> np.ndarray:
        """numpy.ndarray: Boolean mask indicating x-axis lower limits."""
        return self.__x_ll_mask__.copy()

    @property
    def x_lim_mask(self) -> np.ndarray:
        """numpy.ndarray: Boolean mask indicating x-axis limits."""
        return self.__x_ul_mask__.copy() | self.__x_ll_mask__.copy()

    @property
    def y_upper_lim_mask(self) -> np.ndarray:
        """numpy.ndarray: Boolean mask indicating y-axis upper limits."""
        return self.__y_ul_mask__.copy()

    @property
    def y_lower_lim_mask(self) -> np.ndarray:
        """numpy.ndarray: Boolean mask indicating y-axis lower limits."""
        return self.__y_ll_mask__.copy()

    @property
    def y_lim_mask(self) -> np.ndarray:
        """numpy.ndarray: Boolean mask indicating y-axis limits."""
        return self.__y_ul_mask__.copy() | self.__y_ll_mask__.copy()

    @property
    def limit_mask(self) -> np.ndarray:
        """numpy.ndarray: Boolean mask indicating any limits."""
        return (
            self.__x_ul_mask__.copy()
            | self.__x_ll_mask__.copy()
            | self.__y_ul_mask__.copy()
            | self.__y_ll_mask__.copy()
        )

    # ====================================================================== #
    # Utility Functions
    # ====================================================================== #
    @classmethod
    def class_has_x_column(cls) -> bool:
        """Check if the class defines an x-axis column."""
        return cls.X_COLUMN is not None

    @classmethod
    def class_has_y_column(cls) -> bool:
        """Check if the class defines a y-axis column."""
        return cls.Y_COLUMN is not None

    @classmethod
    def class_has_y_error_column(cls) -> bool:
        """Check if the class defines a y-axis error column."""
        return cls.Y_ERROR_COLUMN is not None

    @classmethod
    def class_has_y_upper_limit_column(cls) -> bool:
        """Check if the class defines a y-axis upper limit column."""
        return cls.Y_UPPER_LIMIT_COLUMN is not None

    @classmethod
    def class_has_x_error_column(cls) -> bool:
        """Check if the class defines an x-axis error column."""
        return cls.X_ERROR_COLUMN is not None

    @classmethod
    def class_has_x_upper_limit_column(cls) -> bool:
        """Check if the class defines an x-axis upper limit column."""
        return cls.X_UPPER_LIMIT_COLUMN is not None

    # ===================================================================== #
    # Conversion Functions
    # ===================================================================== #
