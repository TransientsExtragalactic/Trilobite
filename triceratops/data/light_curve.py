"""Light curve data structures and utilities."""

from pathlib import Path
from typing import Optional, Union

import numpy as np
from astropy import units as u
from astropy.table import Table

from .core import XYDataContainer

__all__ = [
    "RadioLightCurveContainer",
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
