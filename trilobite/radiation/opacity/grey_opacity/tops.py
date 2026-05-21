r"""
TOPS (LANL) tabulated grey opacity models.

This module provides grey opacity laws constructed from tabulated
TOPS (Los Alamos National Laboratory Opacity Code) data, including
both Rosseland and Planck mean opacities.

Averaging conventions
---------------------

TOPS tables include both:

- **Rosseland mean** — appropriate for optically thick, diffusive regimes
- **Planck mean** — appropriate for emission and optically thin regimes

The desired mean is selected at construction time via ``mean_type``.

Implementation
--------------

The primary class :class:`TOPSOpacity` wraps a tabulated dataset and
evaluates the opacity using a Cython-backed bilinear interpolation
kernel in :math:`(\log_{10} T, \log_{10} \rho)` space.

Tables are typically constructed via:

- :meth:`TOPSOpacity.load_default` — bundled solar-composition table
- :meth:`TOPSOpacity.from_tops` — from a parsed TOPS dataset

Notes
-----
- All data are internally converted to log-space for numerical stability.
- Invalid or out-of-domain values are represented by ``NaN``.
- This module provides a concrete tabulated implementation of
  :class:`~trilobite.radiation.opacity.grey_opacity.base.GreyOpacityLaw`.
"""

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Optional

import numpy as np

from trilobite.radiation.opacity.grey_opacity.base import GreyOpacityLaw

if TYPE_CHECKING:
    from types import SimpleNamespace

_OOB_MODES = {"raise": 0, "clamp": 1, "nan": 2}

_TABLES_DIR = Path(__file__).parent.parent / "tables"


class TOPSOpacity(GreyOpacityLaw):
    r"""
    TOPS (LANL) tabulated grey opacity.

    This class represents a grey opacity law constructed from tabulated
    TOPS data, evaluated via bilinear interpolation on a
    :math:`(\log_{10} T, \log_{10} \rho)` grid.

    This class implements the standard
    :class:`~trilobite.radiation.opacity.grey_opacity.base.GreyOpacityLaw`
    interface:

    - :meth:`opacity` — evaluate :math:`\kappa(\rho, T)`
    - :meth:`dlogkappa_dlogrho` — logarithmic density derivative
    - :meth:`dlogkappa_dlogT` — logarithmic temperature derivative

    The frequency dependence has been integrated out; the specific
    averaging convention is determined by :attr:`mean_type`.

    Instances are typically constructed via:

    - :meth:`load_default` — bundled solar-composition table
    - :meth:`from_tops` — from a parsed TOPS dataset

    Direct construction from arrays is supported but less common.

    Parameters
    ----------
    grid_T : array_like, shape (n_T,)
        :math:`\log_{10}(T\,[\mathrm{K}])` grid (strictly increasing).
    grid_rho : array_like, shape (n_\rho,)
        :math:`\log_{10}(\rho\,[\mathrm{g\,cm^{-3}}])` grid (strictly increasing).
    kappa : array_like, shape (n_T, n_\rho)
        :math:`\log_{10}(\kappa\,[\mathrm{cm^2\,g^{-1}}])`. ``NaN`` marks
        invalid or out-of-domain cells.
    mean_type : {'rosseland', 'planck'}
        Frequency-averaging convention represented by the table.
    out_of_bounds : {'raise', 'clamp', 'nan'}
        Behaviour for queries outside the tabulated domain.

    Notes
    -----
    - Evaluation is performed using a Cython-backed bilinear interpolation kernel.
    - All inputs are expected in log-space.
    - Missing or invalid table entries should be encoded as ``NaN``.

    See Also
    --------
    ~trilobite.radiation.opacity.opacity_io.read_tops
        Parse a TOPS text file into a namespace.
    TOPSOpacity.from_tops
        Construct from a parsed TOPS dataset.
    TOPSOpacity.load_default
        Load the bundled solar-composition TOPS table.
    """

    IS_C_BACKED = True

    _DEFAULT_PATH: ClassVar[Path] = _TABLES_DIR / "tops_solar.dat"
    _DEFAULT_NS: ClassVar[Optional["SimpleNamespace"]] = None

    def __init__(
        self,
        grid_T,
        grid_rho,
        kappa,
        *,
        mean_type: str = "rosseland",
        out_of_bounds: str = "raise",
    ):
        if out_of_bounds not in _OOB_MODES:
            raise ValueError(f"out_of_bounds must be one of {list(_OOB_MODES)}.")
        if mean_type not in ("rosseland", "planck"):
            raise ValueError("mean_type must be 'rosseland' or 'planck'.")
        self.mean_type = mean_type
        self.out_of_bounds = out_of_bounds
        super().__init__(
            grid_T=grid_T,
            grid_rho=grid_rho,
            kappa=kappa,
            out_of_bounds=out_of_bounds,
        )

    def _initialize_C_object(self, *, grid_T, grid_rho, kappa, out_of_bounds, **_):
        from ._tops_table import C_TOPSTableOpacity

        return C_TOPSTableOpacity(
            np.array(grid_T, dtype=np.float64),
            np.array(grid_rho, dtype=np.float64),
            np.array(kappa, dtype=np.float64, order="C"),
            _OOB_MODES[out_of_bounds],
        )

    @classmethod
    def get_default_ns(cls) -> "SimpleNamespace":
        """Return the bundled solar-composition TOPS namespace, loading it on first call.

        The file ``tables/tops_solar.dat`` (relative to the package) is parsed
        with :func:`~trilobite.radiation.opacity.opacity_io.read_tops` and the result
        is cached at the class level for subsequent calls.

        Returns
        -------
        SimpleNamespace
            Namespace as returned by :func:`~trilobite.radiation.opacity.opacity_io.read_tops`.
        """
        if cls._DEFAULT_NS is None:
            from trilobite.radiation.opacity.opacity_io import read_tops

            cls._DEFAULT_NS = read_tops(cls._DEFAULT_PATH)
        return cls._DEFAULT_NS

    @classmethod
    def load_default(
        cls,
        *,
        mean_type: str = "rosseland",
        out_of_bounds: str = "raise",
        table_path: Optional[Path] = None,
    ) -> "TOPSOpacity":
        r"""Load the bundled solar-composition TOPS table.

        Parameters
        ----------
        mean_type : {'rosseland', 'planck'}
            Which mean opacity to use.
        out_of_bounds : {'raise', 'clamp', 'nan'}
            Behaviour for queries outside the table domain.
        table_path : path-like, optional
            Override the bundled file.  If ``None``, ``tops_solar.dat`` is used.

        Returns
        -------
        TOPSOpacity
        """
        if table_path is not None:
            from trilobite.radiation.opacity.opacity_io import read_tops

            ns = read_tops(table_path)
        else:
            ns = cls.get_default_ns()
        return cls.from_tops(ns, mean_type=mean_type, out_of_bounds=out_of_bounds)

    @classmethod
    def from_tops(
        cls,
        ns: "SimpleNamespace",
        *,
        mean_type: str = "rosseland",
        out_of_bounds: str = "raise",
    ) -> "TOPSOpacity":
        r"""Construct a :class:`TOPSOpacity` tops data namespace.

        Parameters
        ----------
        ns : SimpleNamespace
            Output of :func:`~trilobite.radiation.opacity.opacity_io.read_tops`.
            Must have ``grid_T`` (K), ``grid_rho`` (g cm⁻³), ``rosseland``
            and ``planck`` arrays.
        mean_type : {'rosseland', 'planck'}
            Which opacity table to use.
        out_of_bounds : {'raise', 'clamp', 'nan'}
            Behaviour for out-of-range ``(T, rho)`` queries.

        Returns
        -------
        TOPSOpacity
        """
        if mean_type == "rosseland":
            kappa_raw = ns.rosseland
        elif mean_type == "planck":
            kappa_raw = ns.planck
        else:
            raise ValueError("mean_type must be 'rosseland' or 'planck'.")

        grid_T = np.log10(ns.grid_T)
        grid_rho = np.log10(ns.grid_rho)
        kappa = np.log10(kappa_raw)  # NaN propagates for invalid cells
        return cls(grid_T, grid_rho, kappa, mean_type=mean_type, out_of_bounds=out_of_bounds)

    # ------------------------------------------------------------------ #
    # Log-space interface (dispatches to Cython)                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _broadcast(log_T, log_rho):
        a, b = np.broadcast_arrays(
            np.asarray(log_T, dtype=np.float64),
            np.asarray(log_rho, dtype=np.float64),
        )
        return np.ascontiguousarray(a), np.ascontiguousarray(b)

    def _log_opacity(self, log_T, log_rho):
        r"""Return :math:`\ln\kappa` via bilinear interpolation in the table."""
        if np.ndim(log_T) > 0 or np.ndim(log_rho) > 0:
            return self._c_object.log_opacity_array(*self._broadcast(log_T, log_rho))
        return self._c_object.log_opacity(log_T, log_rho)

    def _dlogkappa_dlogrho(self, log_T, log_rho):
        r"""Return :math:`\partial\ln\kappa/\partial\ln\rho` from the table gradient."""
        if np.ndim(log_T) > 0 or np.ndim(log_rho) > 0:
            return self._c_object.dlogkappa_dlogrho_array(*self._broadcast(log_T, log_rho))
        return self._c_object.dlogkappa_dlogrho(log_T, log_rho)

    def _dlogkappa_dlogT(self, log_T, log_rho):
        r"""Return :math:`\partial\ln\kappa/\partial\ln T` from the table gradient."""
        if np.ndim(log_T) > 0 or np.ndim(log_rho) > 0:
            return self._c_object.dlogkappa_dlogT_array(*self._broadcast(log_T, log_rho))
        return self._c_object.dlogkappa_dlogT(log_T, log_rho)

    def __repr__(self) -> str:
        return f"TOPSOpacity(mean_type={self.mean_type!r}, out_of_bounds={self.out_of_bounds!r})"
