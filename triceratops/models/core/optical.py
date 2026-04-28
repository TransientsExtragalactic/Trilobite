"""
Abstract base classes for optical photometry models.

Defines :class:`OpticalModel` (time-evolving, multi-band) and
:class:`OpticalEpochModel` (single-epoch, multi-band), both built on top of
:class:`~triceratops.models.core.base.Model`.

Subclasses implement :meth:`_compute_sed` to supply a spectral energy
distribution on the bundle's common frequency grid.  All filter convolution,
time deduplication, and band-index gathering are handled by the base classes.
"""

import importlib
import json
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import TYPE_CHECKING, Union

import numpy as np
from astropy import units as u

from .base import Model
from .parameters import ModelVariable
from .serial import ModelSpec

if TYPE_CHECKING:
    from triceratops._typing import _ModelParametersInputRaw, _ModelVariablesInput, _ModelVariablesInputRaw
    from triceratops.utils.phot_utils import FilterBundle, PhotometryFilter

__all__ = ["OpticalModel", "OpticalEpochModel"]

# ============================================================
# Output NamedTuples
# ============================================================
_OpticalOutputs = namedtuple("OpticalOutputs", ["flux"])
_OpticalEpochOutputs = namedtuple("OpticalEpochOutputs", ["flux"])


# ============================================================
# Shared FilterBundle mixin (not exported)
# ============================================================
class _FilterBundleMixin:
    """Owns a FilterBundle and exposes setup-time mutation helpers."""

    def __init_bundle__(self, bundle: "FilterBundle") -> None:
        self._bundle = bundle

    @property
    def bundle(self) -> "FilterBundle":
        """The attached :class:`~triceratops.utils.phot_utils.FilterBundle`."""
        return self._bundle

    def add_filter(self, name: str, f: "PhotometryFilter") -> None:
        """Add a named filter to the bundle (rebuilds weight matrix). Call before inference."""
        self._bundle.add_filter(name, f)

    def remove_filter(self, name: str) -> None:
        """Remove a filter from the bundle by name. Call before inference."""
        self._bundle.remove_filter(name)

    def _resolve_band_index(self, band_index: Union[np.ndarray, list, int, str]) -> np.ndarray:
        """Resolve string band names or integer indices to a validated int array."""
        bi = np.asarray(band_index)
        if bi.dtype.kind in ("U", "S", "O"):
            name_to_idx = {name: i for i, name in enumerate(self._bundle.filter_names)}
            try:
                bi = np.array([name_to_idx[str(n)] for n in bi.flat], dtype=int).reshape(bi.shape)
            except KeyError as exc:
                raise ValueError(f"Unknown band name {exc}. Available: {self._bundle.filter_names}.") from exc
        else:
            bi = bi.astype(int)

        n_filters = len(self._bundle.filter_names)
        if bi.size > 0 and (np.any(bi < 0) or np.any(bi >= n_filters)):
            raise ValueError(
                f"band_index values must be in [0, {n_filters}); got range [{int(bi.min())}, {int(bi.max())}]."
            )
        return bi

    # ------------------------------------------------------------------
    # Serialization overrides: store full filter curves in the spec so
    # the model can be reconstructed without any external file or registry.
    # ------------------------------------------------------------------

    def to_model_spec(self) -> ModelSpec:
        """Serialize to a self-contained ModelSpec with full filter curves embedded."""
        if not hasattr(self, "_model_init_kwargs"):
            raise RuntimeError(
                f"{self.__class__.__name__} did not register constructor arguments. "
                "Models must call `_register_init(...)` inside __init__."
            )

        kwargs = dict(self._model_init_kwargs)
        kwargs["bundle"] = kwargs["bundle"].to_dict()

        try:
            json.dumps(kwargs)
        except TypeError as exc:
            raise TypeError(
                f"Model '{self.__class__.__name__}' contains non-serializable constructor "
                "arguments beyond 'bundle'. Override to_model_spec() to handle them."
            ) from exc

        module_name = self.__class__.__module__
        class_name = self.__class__.__name__
        try:
            module = importlib.import_module(module_name)
            getattr(module, class_name)
        except Exception as exc:
            raise ImportError(f"Model class '{class_name}' is not importable from '{module_name}'.") from exc

        return ModelSpec(target=f"{module_name}:{class_name}", kwargs=kwargs)

    @classmethod
    def from_model_spec(cls, spec: ModelSpec):
        """Reconstruct a model from a ModelSpec, rebuilding the FilterBundle from stored curves."""
        from triceratops.utils.phot_utils import FilterBundle

        if not isinstance(spec, ModelSpec):
            raise TypeError("Expected ModelSpec.")

        kwargs = dict(spec.kwargs)
        kwargs["bundle"] = FilterBundle.from_dict(kwargs["bundle"])

        model = cls(**kwargs)
        if not isinstance(model, Model):
            raise TypeError("Constructed object is not a Model instance.")
        return model


# ============================================================
# OpticalModel — time-evolving, multi-band
# ============================================================
class OpticalModel(_FilterBundleMixin, Model, ABC):
    r"""
    Abstract base class for time-evolving optical photometry models.

    Models inheriting from this class take ``(band_index, time)`` as variables
    and return filter-convolved flux densities.  Subclasses implement
    :meth:`_compute_sed`, which returns :math:`F_\nu` on the bundle's common
    frequency grid for a batch of deduplicated times; the ABC handles the rest.

    Parameters
    ----------
    bundle : FilterBundle
        Collection of optical filters defining the band set.  Mutable after
        construction via :meth:`add_filter` / :meth:`remove_filter`, but weight
        matrices must be finalised before any inference run.

    Notes
    -----
    Variable ``band_index`` accepts either integer indices into *bundle* or
    string band names, which are resolved via :meth:`coerce_model_variables`.

    ``_compute_sed`` is called with deduplicated times, so each unique epoch's
    SED is computed exactly once regardless of how many bands share that epoch.
    Subclasses may work in log-space internally for numerical stability.
    """

    VARIABLES = (
        ModelVariable(
            "band_index",
            base_units=None,
            description="Integer index into the FilterBundle (or string band name).",
            latex=r"$b$",
        ),
        ModelVariable(
            "time",
            base_units=u.s,
            description="Observation time.",
            latex=r"$t$",
        ),
    )
    OUTPUTS = _OpticalOutputs
    UNITS = OUTPUTS(flux=u.erg / u.s / u.cm**2 / u.Hz)
    DESCRIPTION = "Abstract base for time-evolving multi-band optical photometry models."

    def __init__(self, bundle: "FilterBundle") -> None:
        self.__init_bundle__(bundle)
        self._register_init(bundle=bundle)

    def coerce_model_variables(self, variables: "_ModelVariablesInput") -> "_ModelVariablesInputRaw":
        """Coerce variables; resolve band name strings to integer indices."""
        variables = dict(variables)
        if "band_index" in variables:
            variables["band_index"] = self._resolve_band_index(variables["band_index"])
        return super().coerce_model_variables(variables)

    def _forward_model(
        self,
        variables: "_ModelVariablesInputRaw",
        parameters: "_ModelParametersInputRaw",
    ) -> _OpticalOutputs:
        band_index = np.asarray(variables["band_index"], dtype=int)
        time = np.asarray(variables["time"], dtype=float)

        # Evaluate SED only at unique times (avoids N_filters-fold recomputation).
        unique_t, inv_idx = np.unique(time, return_inverse=True)

        # Subclass computes F_nu on the bundle grid: (N_unique, N_nu)
        sed = self._compute_sed(self._bundle.frequency_grid, unique_t, parameters)

        # Filter convolution via precomputed weight-matrix matmul: (N_unique, N_filters)
        band_fluxes = self._bundle.apply(sed)

        # Gather the requested (time, band) pairs: (N,)
        return self.OUTPUTS(flux=band_fluxes[inv_idx, band_index])

    @abstractmethod
    def _compute_sed(
        self,
        nu_grid: np.ndarray,
        t_unique: np.ndarray,
        parameters: "_ModelParametersInputRaw",
    ) -> np.ndarray:
        r"""
        Compute flux density on the bundle's frequency grid at unique times.

        Parameters
        ----------
        nu_grid : ndarray, shape (N_nu,)
            Frequency grid in Hz; equals ``self.bundle.frequency_grid``.
        t_unique : ndarray, shape (N_t,)
            Deduplicated observation times in seconds.
        parameters : dict
            Coerced model parameters in base units (no astropy Quantities).

        Returns
        -------
        ndarray, shape (N_t, N_nu)
            :math:`F_\nu` in :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}}`.

        Notes
        -----
        Subclasses may compute internally in log-space for numerical stability;
        the return value must be in linear flux units.
        """
        raise NotImplementedError


# ============================================================
# OpticalEpochModel — single-epoch, multi-band
# ============================================================
class OpticalEpochModel(_FilterBundleMixin, Model, ABC):
    r"""
    Abstract base class for single-epoch optical photometry (SED) models.

    Takes only ``band_index`` as a variable (no time axis).  Subclasses
    implement :meth:`_compute_sed` to return :math:`F_\nu` on the bundle's
    frequency grid; the ABC applies filter convolution and band-index gather.

    Parameters
    ----------
    bundle : FilterBundle
        Collection of optical filters defining the band set.
    """

    VARIABLES = (
        ModelVariable(
            "band_index",
            base_units=None,
            description="Integer index into the FilterBundle (or string band name).",
            latex=r"$b$",
        ),
    )
    OUTPUTS = _OpticalEpochOutputs
    UNITS = OUTPUTS(flux=u.erg / u.s / u.cm**2 / u.Hz)
    DESCRIPTION = "Abstract base for single-epoch multi-band optical photometry models."

    def __init__(self, bundle: "FilterBundle") -> None:
        self.__init_bundle__(bundle)
        self._register_init(bundle=bundle)

    def coerce_model_variables(self, variables: "_ModelVariablesInput") -> "_ModelVariablesInputRaw":
        """Coerce variables; resolve band name strings to integer indices."""
        variables = dict(variables)
        if "band_index" in variables:
            variables["band_index"] = self._resolve_band_index(variables["band_index"])
        return super().coerce_model_variables(variables)

    def _forward_model(
        self,
        variables: "_ModelVariablesInputRaw",
        parameters: "_ModelParametersInputRaw",
    ) -> _OpticalEpochOutputs:
        band_index = np.asarray(variables["band_index"], dtype=int)

        # Subclass computes F_nu on the bundle grid: (N_nu,)
        sed = self._compute_sed(self._bundle.frequency_grid, parameters)

        # Filter convolution: (N_filters,)
        band_fluxes = self._bundle.apply(sed)

        # Gather requested bands: (N,)
        return self.OUTPUTS(flux=band_fluxes[band_index])

    @abstractmethod
    def _compute_sed(
        self,
        nu_grid: np.ndarray,
        parameters: "_ModelParametersInputRaw",
    ) -> np.ndarray:
        r"""
        Compute flux density on the bundle's frequency grid.

        Parameters
        ----------
        nu_grid : ndarray, shape (N_nu,)
            Frequency grid in Hz; equals ``self.bundle.frequency_grid``.
        parameters : dict
            Coerced model parameters in base units.

        Returns
        -------
        ndarray, shape (N_nu,)
            :math:`F_\nu` in :math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}}`.
        """
        raise NotImplementedError
