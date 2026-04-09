"""Light curve data structures and utilities."""

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from astropy import units as u
from astropy.table import Table

from .core import InferenceData, XYDataContainer
from .utils import infer_err_from_non_detection

if TYPE_CHECKING:
    from triceratops.models.core.base import Model

__all__ = [
    "RadioLightCurveContainer",
    "OpticalLightCurveContainer",
]


class RadioLightCurveContainer(XYDataContainer):
    """
    Immutable container for single-band radio light curve data.

    This container represents time-series radio flux density measurements
    taken at a *fixed observing frequency or band*. Unlike heterogeneous
    photometry containers, all observations are assumed to belong to a
    single frequency and are intended to be modeled as a single light curve.

    The container provides a validated, unit-aware, read-only interface
    suitable for likelihood evaluation, inference, and visualization.

    Notes
    -----
    - Detection status is inferred from ``flux_upper_limit`` being NaN
      for detections and finite for non-detections.
    - The observing frequency is stored as metadata, not as a column.
    """

    # ========================= SCHEMA ========================= #
    COLUMNS = [
        {"name": "time", "dtype": float, "unit": u.day, "required": True},
        {"name": "flux_density", "dtype": float, "unit": u.Jy, "required": True},
        {"name": "flux_density_error", "dtype": float, "unit": u.Jy, "required": True},
        {"name": "flux_upper_limit", "dtype": float, "unit": u.Jy, "required": True},
        {"name": "obs_name", "dtype": str, "required": False},
        {"name": "comments", "dtype": str, "required": False},
    ]

    # ========================= SPECIAL COLUMNS ========================= #
    X_COLUMN = "time"
    Y_COLUMN = "flux_density"
    Y_ERROR_COLUMN = "flux_density_error"
    Y_UPPER_LIMIT_COLUMN = "flux_upper_limit"

    # ========================= INIT ========================= #
    def __init__(
        self,
        table: Table,
        *,
        frequency: Union[float, u.Quantity],
    ):
        """
        Instantiate a radio light curve container.

        Parameters
        ----------
        table : astropy.table.Table
            Table containing the light curve data.
        frequency : float or astropy.units.Quantity
            Observing frequency. If unitless, GHz is assumed.
        """
        super().__init__(table)

        # --- metadata ---
        if isinstance(frequency, u.Quantity):
            self.__frequency__ = frequency.to(u.GHz)
        else:
            self.__frequency__ = frequency * u.GHz

    # ========================= METADATA ========================= #
    def _copy_kwargs(self) -> dict:
        """Return the frequency metadata required to reconstruct this container."""
        return {"frequency": self.frequency}

    @property
    def frequency(self) -> u.Quantity:
        """Observing frequency of the light curve."""
        return self.__frequency__

    # ========================= DATA ACCESS ========================= #
    @property
    def time(self) -> u.Quantity:
        """Observation times."""
        return self.table["time"].quantity

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

    # ========================= NUMERICAL BACKENDS ========================= #
    def to_cgs_array(self) -> np.ndarray:
        """
        Convert the light curve to a dense NumPy array in CGS units.

        Returns
        -------
        ndarray
            Array of shape (n_obs, 4) containing
            [time, flux, flux_err, flux_ul] in CGS units.
        """
        return np.vstack(
            [
                self.time.to(u.s).value,
                self.flux_density.to(u.erg / u.s / u.cm**2 / u.Hz).value,
                self.flux_density_error.to(u.erg / u.s / u.cm**2 / u.Hz).value,
                self.flux_upper_limit.to(u.erg / u.s / u.cm**2 / u.Hz).value,
            ]
        ).T

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
        Convert this :class:`RadioLightCurveContainer` into an :class:`~triceratops.data.core.InferenceData` object.

        For a single-frequency light curve, time is the independent variable and
        flux density is the observable. The container's fixed observing frequency
        is metadata and is not included as a per-row variable.

        Default mapping:

        - ``variables``: ``{"time": "time"}``
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
                if var_name == "time":
                    variables[var_name] = "time"
                else:
                    raise ValueError(
                        f"Cannot infer mapping for model variable '{var_name}'. "
                        "RadioLightCurveContainer only maps 'time'. "
                        "Please specify `variables=` explicitly."
                    )

        # ------------------------------------------------------------
        # Default observable mapping
        # ------------------------------------------------------------
        if observables is None:
            if len(model.output_names) != 1:
                raise ValueError(
                    "RadioLightCurveContainer can only infer defaults for "
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

    # ========================= CONSTRUCTORS ========================= #
    @classmethod
    def from_table(
        cls,
        table: Table,
        *,
        frequency: Union[float, u.Quantity],
        column_map: Optional[dict] = None,
    ):
        """Construct a RadioLightCurveContainer from an Astropy table."""
        if column_map is not None:
            table = table.copy()
            table.rename_columns(list(column_map.keys()), list(column_map.values()))

        return cls(
            table,
            frequency=frequency,
        )

    @classmethod
    def from_file(
        cls,
        path: Union[str, Path],
        *,
        frequency: Union[float, u.Quantity],
        **kwargs,
    ):
        """Construct a RadioLightCurveContainer from a file on disk."""
        table = Table.read(path, **kwargs)
        return cls(
            table,
            frequency=frequency,
        )


class OpticalLightCurveContainer(XYDataContainer):
    r"""
    Immutable container for single-band optical light curve data.

    Represents time-series optical flux density or AB magnitude measurements
    taken in a *single fixed optical band*. The band is stored as metadata
    (not a column) and is resolved to a model band index at
    :meth:`to_inference_data` time via the model's
    :class:`~triceratops.utils.phot_utils.FilterBundle`.

    Observations may be expressed as **flux densities** (F_ν in erg/s/cm²/Hz),
    **AB magnitudes**, or both. The container exposes both representations
    via :attr:`flux`, :attr:`mag`, and related properties.

    Detection status is inferred from ``flux_upper_limit`` being NaN
    for detections and finite for non-detections. If only magnitude columns
    are present, ``mag_ab_upper_limit`` is used instead.

    Schema
    ------
    Required:

    - ``time`` [day]

    At least one y-column group must be present:

    - flux group: ``flux_density``, ``flux_density_error``, ``flux_upper_limit``
      [erg/(s cm² Hz)]
    - magnitude group: ``mag_ab``, ``mag_ab_error``, ``mag_ab_upper_limit``
      (dimensionless)

    Optional:

    - ``obs_name`` [str]
    - ``comments`` [str]

    See Also
    --------
    RadioLightCurveContainer
        Equivalent container for single-frequency radio observations.
    OpticalPhotometryContainer
        Multi-band optical photometry container.
    """

    from triceratops.utils.phot_utils import ab_mag_to_flux as _ab_mag_to_flux
    from triceratops.utils.phot_utils import flux_to_ab_mag as _flux_to_ab_mag

    # ========================= SCHEMA ========================= #
    COLUMNS = [
        {"name": "time", "dtype": float, "unit": u.day, "required": True},
        {"name": "flux_density", "dtype": float, "unit": u.Unit("erg/(s cm2 Hz)"), "required": False},
        {"name": "flux_density_error", "dtype": float, "unit": u.Unit("erg/(s cm2 Hz)"), "required": False},
        {"name": "flux_upper_limit", "dtype": float, "unit": u.Unit("erg/(s cm2 Hz)"), "required": False},
        {"name": "mag_ab", "dtype": float, "required": False},
        {"name": "mag_ab_error", "dtype": float, "required": False},
        {"name": "mag_ab_upper_limit", "dtype": float, "required": False},
        {"name": "obs_name", "dtype": str, "required": False},
        {"name": "comments", "dtype": str, "required": False},
    ]

    # ========================= SPECIAL COLUMNS ========================= #
    X_COLUMN = "time"
    Y_COLUMN = "flux_density"
    Y_ERROR_COLUMN = "flux_density_error"
    Y_UPPER_LIMIT_COLUMN = "flux_upper_limit"

    # ========================= INIT ========================= #
    def __init__(self, table: Table, *, band: str):
        """
        Instantiate an optical light curve container.

        Parameters
        ----------
        table : astropy.table.Table
            Table containing the light curve data.
        band : str
            Band name (e.g. ``"g"``, ``"r"``). Must match a name in the
            model's FilterBundle at inference time.
        """
        self.__band__ = str(band)
        super().__init__(table)

    # ========================= VALIDATION ========================= #
    def _validate_table(self, table: Table) -> Table:
        table = super()._validate_table(table)
        has_flux = "flux_density" in table.colnames
        has_mag = "mag_ab" in table.colnames
        if not has_flux and not has_mag:
            raise ValueError(
                "OpticalLightCurveContainer requires at least one y-column group: "
                "either 'flux_density' (+ 'flux_density_error' + 'flux_upper_limit') "
                "or 'mag_ab' (+ 'mag_ab_error' + 'mag_ab_upper_limit')."
            )
        return table

    # ========================= METADATA ========================= #
    def _copy_kwargs(self) -> dict:
        return {"band": self.band}

    @property
    def band(self) -> str:
        """Band name for this light curve."""
        return self.__band__

    # ========================= HELPERS ========================= #
    @property
    def _has_flux_column(self) -> bool:
        return "flux_density" in self.__table__.colnames

    @property
    def _has_mag_column(self) -> bool:
        return "mag_ab" in self.__table__.colnames

    # ========================= DATA ACCESS ========================= #
    @property
    def time(self) -> u.Quantity:
        """Observation times in days."""
        return self.__table__["time"].quantity

    @property
    def flux(self) -> u.Quantity:
        """Flux density F_ν in erg/s/cm²/Hz (converted from mag if needed)."""
        if self._has_flux_column:
            return self.__table__["flux_density"].quantity
        from triceratops.utils.phot_utils import ab_mag_to_flux

        mag_vals = np.asarray(self.__table__["mag_ab"].data, dtype=float)
        return ab_mag_to_flux(mag_vals) * u.Unit("erg/(s cm2 Hz)")

    @property
    def flux_error(self) -> u.Quantity:
        r"""1-sigma uncertainty on F_ν (propagated from mag error if needed)."""
        if self._has_flux_column and "flux_density_error" in self.__table__.colnames:
            return self.__table__["flux_density_error"].quantity
        mag_err = np.asarray(self.__table__["mag_ab_error"].data, dtype=float)
        return self.flux * (mag_err / 1.0857)

    @property
    def flux_upper_limit(self) -> u.Quantity:
        """Upper limit on F_ν (NaN for detections, converted from mag if needed)."""
        if "flux_upper_limit" in self.__table__.colnames:
            return self.__table__["flux_upper_limit"].quantity
        if "mag_ab_upper_limit" in self.__table__.colnames:
            from triceratops.utils.phot_utils import ab_mag_to_flux

            mag_ul = np.asarray(self.__table__["mag_ab_upper_limit"].data, dtype=float)
            return ab_mag_to_flux(mag_ul) * u.Unit("erg/(s cm2 Hz)")
        return np.full(len(self), np.nan) * u.Unit("erg/(s cm2 Hz)")

    @property
    def mag(self) -> np.ndarray:
        """AB magnitude (dimensionless; converted from flux if needed)."""
        if self._has_mag_column:
            return np.asarray(self.__table__["mag_ab"].data, dtype=float)
        from triceratops.utils.phot_utils import flux_to_ab_mag

        return flux_to_ab_mag(self.flux.value)

    @property
    def mag_error(self) -> np.ndarray:
        r"""1-sigma uncertainty on AB magnitude (propagated from flux if needed)."""
        if self._has_mag_column and "mag_ab_error" in self.__table__.colnames:
            return np.asarray(self.__table__["mag_ab_error"].data, dtype=float)
        return 1.0857 * self.flux_error.value / self.flux.value

    @property
    def mag_upper_limit(self) -> np.ndarray:
        """AB magnitude upper limit (NaN for detections; converted from flux if needed)."""
        if "mag_ab_upper_limit" in self.__table__.colnames:
            return np.asarray(self.__table__["mag_ab_upper_limit"].data, dtype=float)
        from triceratops.utils.phot_utils import flux_to_ab_mag

        return flux_to_ab_mag(self.flux_upper_limit.value)

    # ========================= DETECTION LOGIC ========================= #
    @property
    def detection_mask(self) -> np.ndarray:
        """Boolean mask: True for detections."""
        return np.isnan(self.flux_upper_limit.value)

    @property
    def non_detection_mask(self) -> np.ndarray:
        """Boolean mask: True for upper limits (non-detections)."""
        return ~self.detection_mask

    @property
    def n_detections(self) -> int:
        return int(self.detection_mask.sum())

    @property
    def n_non_detections(self) -> int:
        return int(self.non_detection_mask.sum())

    @property
    def n_obs(self) -> int:
        return len(self)

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
        Convert this :class:`OpticalLightCurveContainer` to an :class:`~triceratops.data.core.InferenceData` object.

        The container's band name is resolved to an integer ``band_idx`` via
        ``model.bundle.filter_names``. The band index is broadcast to a constant
        array (same value for every time point) and injected into ``x``.

        Default mapping:

        - ``variables``: ``{"time": ..., "band_idx": <constant>}``
        - ``observables``: the single model output mapped to flux columns

        Parameters
        ----------
        model : Model
            Optical model instance exposing a ``bundle`` attribute
            (:class:`~triceratops.utils.phot_utils.FilterBundle`).
        variables : dict, optional
            Override the default variable mapping.
        observables : dict, optional
            Override the default observable mapping.
        infer_errors : bool, default True
            Infer 1-sigma errors for non-detections from upper limits.
        detection_threshold : float, default 3.0
            N-sigma threshold used when inferring errors.
        mask : np.ndarray of bool, optional
            Optional row mask applied before building InferenceData.

        Returns
        -------
        InferenceData

        Raises
        ------
        AttributeError
            If ``model`` does not expose a ``bundle`` attribute.
        KeyError
            If ``self.band`` is not in ``model.bundle.filter_names``.
        """
        if not hasattr(model, "bundle"):
            raise AttributeError(
                f"Model '{type(model).__name__}' does not have a 'bundle' attribute. "
                "Optical light curve containers require the model to expose a "
                "FilterBundle via 'model.bundle'."
            )

        filter_names = list(model.bundle.filter_names)
        if self.band not in filter_names:
            raise KeyError(f"Band '{self.band}' is not in the model's FilterBundle. Available bands: {filter_names}")
        band_idx = filter_names.index(self.band)

        # Apply optional row mask
        if mask is not None:
            if mask.shape != (self.n_obs,):
                raise ValueError("Mask must have the same length as the number of observations.")
            mask_arr = mask
        else:
            mask_arr = np.ones(self.n_obs, dtype=bool)

        t_vals = self.time.to(u.day).value[mask_arr]
        idx_vals = np.full(t_vals.shape, band_idx, dtype=int)
        flux_vals = self.flux.to(u.Unit("erg/(s cm2 Hz)")).value[mask_arr]
        flux_err_vals = self.flux_error.to(u.Unit("erg/(s cm2 Hz)")).value[mask_arr]
        flux_ul_vals = self.flux_upper_limit.to(u.Unit("erg/(s cm2 Hz)")).value[mask_arr]

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

        if variables is None:
            variables = {
                "time": t_vals * u.day,
                "band_idx": idx_vals,
            }
        if observables is None:
            if len(model.output_names) != 1:
                raise ValueError(
                    "OpticalLightCurveContainer can only infer defaults for "
                    "single-observable models. Please specify `observables=`."
                )
            obs_name = model.output_names[0]
            y = {obs_name: flux_vals * u.Unit("erg/(s cm2 Hz)")}
            y_err = {obs_name: flux_err_vals * u.Unit("erg/(s cm2 Hz)")}
            y_upper = {obs_name: flux_ul_vals * u.Unit("erg/(s cm2 Hz)")}
        else:
            # User-supplied observables — fall back to from_table path
            y = y_err = y_upper = None

        if y is not None:
            return InferenceData.from_arrays(
                model=model,
                x=variables,
                y=y,
                y_err=y_err,
                y_upper=y_upper,
            )

        # User supplied custom observables — build a scratch table and use from_table
        scratch = self.__table__.copy()
        if mask is not None:
            scratch = scratch[mask_arr]
        scratch["band_idx"] = idx_vals
        return InferenceData.from_table(
            model=model,
            table=scratch,
            variables=variables if not isinstance(list(variables.values())[0], u.Quantity) else None,
            observables=observables,
        )

    # ========================= CONSTRUCTORS ========================= #
    @classmethod
    def from_table(
        cls,
        table: Table,
        *,
        band: str,
        column_map: Optional[dict] = None,
    ):
        """Construct an OpticalLightCurveContainer from an Astropy table."""
        if column_map is not None:
            table = table.copy()
            table.rename_columns(list(column_map.keys()), list(column_map.values()))
        return cls(table, band=band)

    @classmethod
    def from_file(
        cls,
        path: Union[str, Path],
        *,
        band: str,
        **kwargs,
    ):
        """Construct an OpticalLightCurveContainer from a file on disk."""
        table = Table.read(path, **kwargs)
        return cls(table, band=band)
