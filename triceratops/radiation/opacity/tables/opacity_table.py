"""Generic opacity table container with HDF5 I/O and OPAL text parser.

The primary persistent format is HDF5.  OPAL-format fixed-width text files
can be imported via :meth:`OpacityTable.read_from_opal_txt`.  Additional
formats can be added as further ``read_from_*`` class methods.

HDF5 schema
-----------
::

    / attrs: format_version='1.0', coord_system='T_R'|'T_rho',
             source='', reference=''

    /grid/
        grid_1   float64[n1]    log10(T [K])
        grid_2   float64[n2]    log10(R) or log10(rho [g/cm**3])

    /opacity   float64[N, n1, n2]   log10(kappa [cm**2/g]); NaN = invalid

    /metadata/                    one dataset per key, float64[N] or bytes_[N]
        <key_1>  float64[N]
        ...

All files — single- or multi-table — use the same schema; N=1 for a single
composition.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Union

import h5py
import numpy as np

if TYPE_CHECKING:
    pass

#: Current schema version written to every new file.
_FORMAT_VERSION = "1.0"

#: Sentinel value in OPAL text files that marks an out-of-range cell.
_OPAL_INVALID = 9.999

#: Tolerance used by :meth:`OpacityTable.where` when matching scalar metadata values.
_SCALAR_MATCH_TOL = 1e-6


class OpacityTable:
    r"""Generic container for *N* two-dimensional opacity tables on a shared grid.

    Parameters
    ----------
    grid_1 : ndarray, shape (n1,)
        First-axis grid values.  By convention this is always
        :math:`\log_{10}(T\,[\mathrm{K}])`.
    grid_2 : ndarray, shape (n2,)
        Second-axis grid values.  Its physical meaning is fixed by
        *coord_system*.
    opacity : ndarray, shape (N, n1, n2) or (n1, n2)
        :math:`\log_{10}(\kappa\,[\mathrm{cm^2\,g^{-1}}])`.
        ``NaN`` marks cells that are outside the valid table domain.
        A shape of ``(n1, n2)`` is accepted and stored as ``(1, n1, n2)``.
    coord_system : {'T_R', 'T_rho'}
        Parameterisation of *grid_2*:

        * ``'T_R'``   — :math:`\log_{10}(R)` where
          :math:`R = \rho / T_6^3`,  :math:`T_6 = 10^{-6}\,T\,[\mathrm{K}]`
          (OPAL convention).
        * ``'T_rho'`` — :math:`\log_{10}(\rho\,[\mathrm{g\,cm^{-3}}])`.
    metadata : dict of {str: array-like of length N}, optional
        Arbitrary per-table arrays.  Keys must be valid Python identifiers.
        Common keys for OPAL tables: ``X``, ``Y``, ``Z``, ``dX1``,
        ``dX2``, ``table_id``.
    source : str, optional
        Free-text description of the data origin.
    reference : str, optional
        Bibliographic reference string.
    """

    _VALID_COORD_SYSTEMS = frozenset({"T_R", "T_rho"})

    def __init__(
        self,
        grid_1: np.ndarray,
        grid_2: np.ndarray,
        opacity: np.ndarray,
        coord_system: str = "T_R",
        metadata: Union[dict, None] = None,
        source: str = "",
        reference: str = "",
    ):
        grid_1 = np.asarray(grid_1, dtype=np.float64)
        grid_2 = np.asarray(grid_2, dtype=np.float64)
        opacity = np.asarray(opacity, dtype=np.float64)

        if opacity.ndim == 2:
            opacity = opacity[np.newaxis, :, :]

        if coord_system not in self._VALID_COORD_SYSTEMS:
            raise ValueError(f"coord_system must be one of {sorted(self._VALID_COORD_SYSTEMS)}, got {coord_system!r}.")
        if opacity.ndim != 3:
            raise ValueError(f"opacity must be 2-D or 3-D, got shape {opacity.shape}.")
        n_tables, n1, n2 = opacity.shape
        if grid_1.shape != (n1,):
            raise ValueError(f"grid_1 shape {grid_1.shape} does not match opacity axis 1 ({n1}).")
        if grid_2.shape != (n2,):
            raise ValueError(f"grid_2 shape {grid_2.shape} does not match opacity axis 2 ({n2}).")

        if grid_1.size < 2 or not np.all(np.diff(grid_1) > 0):
            raise ValueError("grid_1 must be strictly increasing with at least 2 points.")
        if grid_2.size < 2 or not np.all(np.diff(grid_2) > 0):
            raise ValueError("grid_2 must be strictly increasing with at least 2 points.")

        self._grid_1 = grid_1
        self._grid_2 = grid_2
        self._opacity = opacity
        self._coord_system = coord_system
        self._source = source
        self._reference = reference

        # Normalise metadata: values → float64 or bytes_ arrays of length N.
        self._metadata: dict[str, np.ndarray] = {}
        if metadata:
            for key, val in metadata.items():
                arr = np.asarray(val)
                if arr.shape != (n_tables,):
                    raise ValueError(f"metadata[{key!r}] has shape {arr.shape}; expected ({n_tables},).")
                self._metadata[key] = arr

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def grid_1(self) -> np.ndarray:
        r"""First-axis grid: :math:`\log_{10}(T\,[\mathrm{K}])`, shape ``(n1,)``.  Read-only."""
        view = self._grid_1.view()
        view.flags.writeable = False
        return view

    @property
    def grid_2(self) -> np.ndarray:
        r"""Second-axis grid, shape ``(n2,)``.  Meaning set by :attr:`coord_system`.  Read-only."""
        view = self._grid_2.view()
        view.flags.writeable = False
        return view

    @property
    def opacity(self) -> np.ndarray:
        r"""Opacity array, shape ``(N, n1, n2)``, :math:`\log_{10}(\kappa)`.  Read-only."""
        view = self._opacity.view()
        view.flags.writeable = False
        return view

    @property
    def coord_system(self) -> str:
        """Coordinate system of the second axis (``'T_R'`` or ``'T_rho'``)."""
        return self._coord_system

    @property
    def metadata(self) -> MappingProxyType:
        """Per-table metadata.  Returns a read-only view of the internal dict."""
        return MappingProxyType(self._metadata)

    @property
    def source(self) -> str:
        """Free-text data origin description."""
        return self._source

    @property
    def reference(self) -> str:
        """Bibliographic reference string."""
        return self._reference

    @property
    def n_tables(self) -> int:
        """Number of tables *N*."""
        return self._opacity.shape[0]

    # ------------------------------------------------------------------ #
    # Selection                                                            #
    # ------------------------------------------------------------------ #

    def select(self, index: int) -> OpacityTable:
        """Return a single-table :class:`OpacityTable` at *index*.

        Parameters
        ----------
        index : int
            Zero-based table index.

        Returns
        -------
        OpacityTable
            A new instance with ``n_tables == 1`` and ``opacity`` shape
            ``(1, n1, n2)``.  The returned object is the same class as
            ``self`` (subclass-safe).
        """
        n = self.n_tables
        if not (-n <= index < n):
            raise IndexError(f"index {index} out of range for table with {n} entries.")
        # Normalise negative index.
        idx = index % n
        sliced_opacity = self._opacity[idx : idx + 1]
        sliced_meta = {k: v[idx : idx + 1] for k, v in self._metadata.items()}
        return type(self)(
            self._grid_1,
            self._grid_2,
            sliced_opacity,
            coord_system=self._coord_system,
            metadata=sliced_meta,
            source=self._source,
            reference=self._reference,
        )

    def where(self, **conditions) -> list[int]:
        """Return indices of tables whose metadata satisfies *all* conditions.

        Each keyword argument must match a key in :attr:`metadata`.  The value
        can be:

        * a **scalar float** — matched with absolute tolerance
          :data:`_SCALAR_MATCH_TOL`;
        * a **callable** ``f(values: ndarray) -> bool ndarray`` — applied
          element-wise.

        Parameters
        ----------
        **conditions
            ``key=value`` pairs (see above).

        Returns
        -------
        list of int
            Zero-based indices of matching tables.  May be empty.
            If called with **no conditions**, all indices are returned
            (equivalent to ``list(range(n_tables))``).

        Raises
        ------
        KeyError
            If any condition key is not present in :attr:`metadata`.

        Notes
        -----
        Scalar values are compared using an absolute tolerance of
        :data:`_SCALAR_MATCH_TOL` (``1e-6``) to accommodate floating-point
        storage round-trips.

        Examples
        --------
        >>> tbl.where(X=0.70, Z=0.02)
        [72]
        >>> tbl.where(Z=lambda z: z > 0.05)
        [10, 11, 12, ...]
        >>> tbl.where()  # returns all indices
        [0, 1, 2, ..., N-1]
        """
        if not conditions:
            return list(range(self.n_tables))

        mask = np.ones(self.n_tables, dtype=bool)
        for key, val in conditions.items():
            if key not in self._metadata:
                raise KeyError(f"Metadata key {key!r} not found.  Available keys: {sorted(self._metadata)}.")
            arr = self._metadata[key]
            if callable(val):
                mask &= val(arr).astype(bool)
            else:
                # Scalar: try numeric comparison; fall back to element equality.
                try:
                    mask &= np.abs(arr.astype(float) - float(val)) <= _SCALAR_MATCH_TOL
                except (TypeError, ValueError):
                    mask &= arr == val
        return list(np.where(mask)[0])

    # ------------------------------------------------------------------ #
    # I/O                                                                  #
    # ------------------------------------------------------------------ #

    @classmethod
    def read(cls, path: Union[str, Path]) -> OpacityTable:
        """Read an :class:`OpacityTable` from an HDF5 file.

        Parameters
        ----------
        path : str or Path
            Path to an HDF5 file written by :meth:`write`.

        Returns
        -------
        OpacityTable
            The loaded table.  Returns a plain :class:`OpacityTable`; use
            :meth:`OPALOpacityTable.read` to get the OPAL subclass.
        """
        return cls._read_hdf5(path)

    @classmethod
    def _read_hdf5(cls, path: Union[str, Path]) -> OpacityTable:
        path = Path(path)
        with h5py.File(path, "r") as f:
            _required = {"grid/grid_1", "grid/grid_2", "opacity"}
            _missing = [k for k in _required if k not in f]
            if _missing:
                raise ValueError(
                    f"Malformed HDF5 file {path}: missing required datasets "
                    f"{_missing}.  Expected keys: {sorted(_required)}."
                )

            _fmt = f.attrs.get("format_version", "unknown")
            coord_system = f.attrs.get("coord_system", "T_R")
            source = f.attrs.get("source", "")
            reference = f.attrs.get("reference", "")

            grid_1 = f["grid/grid_1"][:]
            grid_2 = f["grid/grid_2"][:]
            opacity = f["opacity"][:]

            metadata = {}
            if "metadata" in f:
                for key in f["metadata"]:
                    raw = f[f"metadata/{key}"][:]
                    # h5py returns bytes arrays for string datasets.
                    if raw.dtype.kind in ("S", "O"):
                        raw = raw.astype(str)
                    metadata[key] = raw

        return cls(
            grid_1,
            grid_2,
            opacity,
            coord_system=str(coord_system),
            metadata=metadata,
            source=str(source),
            reference=str(reference),
        )

    def write(self, path: Union[str, Path]) -> None:
        """Write this table to an HDF5 file.

        Parameters
        ----------
        path : str or Path
            Output path.  Existing files are overwritten.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            f.attrs["format_version"] = _FORMAT_VERSION
            f.attrs["coord_system"] = self._coord_system
            f.attrs["source"] = self._source
            f.attrs["reference"] = self._reference

            grp = f.create_group("grid")
            grp.create_dataset("grid_1", data=self._grid_1)
            grp.create_dataset("grid_2", data=self._grid_2)

            f.create_dataset("opacity", data=self._opacity)

            if self._metadata:
                mgrp = f.create_group("metadata")
                for key, arr in self._metadata.items():
                    if arr.dtype.kind in ("U", "S", "O"):
                        # Store strings as variable-length UTF-8.
                        dt = h5py.string_dtype()
                        mgrp.create_dataset(key, data=arr.astype(object), dtype=dt)
                    else:
                        mgrp.create_dataset(key, data=arr)

    @classmethod
    def read_from_opal_txt(cls, path: Union[str, Path]) -> OpacityTable:
        r"""Parse an OPAL-format fixed-width text file.

        Handles both the OP Rosseland opacity table format (Badnel et al.
        2005) and single-table files downloaded from the OPAL web server.

        The file is expected to follow the layout::

            ...preamble...
            Table Summaries — There are N tables
             TABLE #  1  <id>  X=...  Y=...  Z=...  dX1=...  dX2=...
             TABLE #  2  ...
             ...
            *...* Tables ****
            TABLE #  1  <id>  X=...  Y=...  Z=...  dX1=...  dX2=...
            ...header...
            logT  -8.0  -7.5  ...  1.0
            <rows of data>

        Parameters
        ----------
        path : str or Path
            Path to the OPAL text file.

        Returns
        -------
        OpacityTable
            A new table with ``coord_system='T_R'`` and metadata keys
            ``X``, ``Y``, ``Z``, ``dX1``, ``dX2``, ``table_id``.
        """
        return cls._parse_opal_txt(path)

    @classmethod
    def _parse_opal_txt(cls, path: Union[str, Path]) -> OpacityTable:
        """Parse OPAL text table.  Returns an ``OpacityTable`` (or subclass)."""
        path = Path(path)
        lines = path.read_text().splitlines()

        # --- Locate the data section ("----- Tables -----" separator) ---
        data_start = None
        for i, line in enumerate(lines):
            if re.match(r"^\*+\s+Tables\s+\*+", line.strip()):
                data_start = i + 1
                break
        if data_start is None:
            raise ValueError(f"Could not find '*** Tables ***' separator in {path}.")

        # --- Parse the standardised log_R header from the first data table ---
        # The column header line looks like:
        #   logT  -8.0   -7.5   ...   1.0
        log_r_values = None
        log_t_values = None

        # --- Parse all table blocks ---
        table_header_re = re.compile(
            r"TABLE\s*#\s*\d+\s+(\S+)\s+X=([0-9.]+)\s+Y=([0-9.]+)\s+"
            r"Z=([0-9.]+)\s+dX1=([0-9.]+)\s+dX2=([0-9.]+)"
        )
        col_header_re = re.compile(r"logT\s+(.*)")

        # Collect blocks: list of (table_id, X, Y, Z, dX1, dX2, rows)
        blocks: list[tuple] = []
        i = data_start
        n_lines = len(lines)

        while i < n_lines:
            line = lines[i]
            m = table_header_re.search(line)
            if m:
                tid, X, Y, Z, dX1, dX2 = (
                    m.group(1),
                    float(m.group(2)),
                    float(m.group(3)),
                    float(m.group(4)),
                    float(m.group(5)),
                    float(m.group(6)),
                )
                # Skip lines until we hit the "logT ..." header.
                i += 1
                col_header_line = None
                while i < n_lines:
                    m2 = col_header_re.match(lines[i].strip())
                    if m2:
                        col_header_line = m2
                        break
                    i += 1

                if col_header_line is None:
                    raise ValueError(f"Missing 'logT ...' column header after TABLE block in {path}.")

                if log_r_values is None:
                    log_r_values = np.array(
                        [float(x) for x in col_header_line.group(1).split()],
                        dtype=np.float64,
                    )

                i += 1  # move past column header

                # Read data rows until blank or next TABLE
                rows_logT = []
                rows_data = []
                while i < n_lines:
                    dline = lines[i].strip()
                    if not dline:
                        i += 1
                        continue
                    if table_header_re.search(dline) or re.match(r"^\*+", dline):
                        break
                    parts = dline.split()
                    try:
                        t_val = float(parts[0])
                    except (ValueError, IndexError):
                        i += 1
                        continue
                    kappa_row = np.array([float(x) for x in parts[1:]], dtype=np.float64)
                    # Pad or trim to n2 columns if necessary.
                    rows_logT.append(t_val)
                    rows_data.append(kappa_row)
                    i += 1

                if not rows_logT:
                    continue  # empty block — skip

                # Build data array; replace 9.999 sentinel with NaN.
                n2 = len(log_r_values)
                data_arr = np.full((len(rows_logT), n2), np.nan, dtype=np.float64)
                for ri, row in enumerate(rows_data):
                    n_cols = min(len(row), n2)
                    data_arr[ri, :n_cols] = row[:n_cols]
                data_arr[np.abs(data_arr - _OPAL_INVALID) < 1e-3] = np.nan

                if log_t_values is None:
                    log_t_values = np.array(rows_logT, dtype=np.float64)

                blocks.append((tid, X, Y, Z, dX1, dX2, data_arr))
            else:
                i += 1

        if not blocks:
            raise ValueError(f"No TABLE blocks found in {path}.")

        # Ensure all tables have the same logT grid (use first as reference).
        n1 = len(log_t_values)
        n2 = len(log_r_values)
        n_tables = len(blocks)

        opacity = np.full((n_tables, n1, n2), np.nan, dtype=np.float64)
        table_ids = []
        X_arr = np.zeros(n_tables)
        Y_arr = np.zeros(n_tables)
        Z_arr = np.zeros(n_tables)
        dX1_arr = np.zeros(n_tables)
        dX2_arr = np.zeros(n_tables)

        for ti, (tid, X, Y, Z, dX1, dX2, data_arr) in enumerate(blocks):
            r1 = min(data_arr.shape[0], n1)
            r2 = min(data_arr.shape[1], n2)
            opacity[ti, :r1, :r2] = data_arr[:r1, :r2]
            table_ids.append(tid)
            X_arr[ti] = X
            Y_arr[ti] = Y
            Z_arr[ti] = Z
            dX1_arr[ti] = dX1
            dX2_arr[ti] = dX2

        metadata = {
            "X": X_arr,
            "Y": Y_arr,
            "Z": Z_arr,
            "dX1": dX1_arr,
            "dX2": dX2_arr,
            "table_id": np.array(table_ids, dtype=object),
        }

        return cls(
            log_t_values,
            log_r_values,
            opacity,
            coord_system="T_R",
            metadata=metadata,
            source="OPAL",
            reference="Badnel et al. (2005) MNRAS 360:458",
        )

    # ------------------------------------------------------------------ #
    # Dunder                                                               #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return self.n_tables

    def __repr__(self) -> str:
        parts = [
            f"n_tables={self.n_tables}",
            f"grid_1=[{self._grid_1[0]:.2f}, {self._grid_1[-1]:.2f}]",
            f"grid_2=[{self._grid_2[0]:.2f}, {self._grid_2[-1]:.2f}]",
            f"coord_system={self._coord_system!r}",
        ]
        # For single-table instances, show any composition metadata inline.
        if self.n_tables == 1:
            comp_keys = [k for k in ("X", "Y", "Z") if k in self._metadata]
            parts.extend([f"{k}={self._metadata[k][0]:.4g}" for k in comp_keys])
        return f"{type(self).__name__}({', '.join(parts)})"


class OPALOpacityTable(OpacityTable):
    r"""Subclass of :class:`OpacityTable` for OPAL Rosseland mean opacity tables.

    Adds convenience accessors for the standard OPAL composition metadata
    (``X``, ``Y``, ``Z``) and a :meth:`select_composition` helper.

    All :meth:`~OpacityTable.read`, :meth:`~OpacityTable.write`, and
    :meth:`~OpacityTable.read_from_opal_txt` methods are inherited unchanged.

    Raises
    ------
    ValueError
        At construction time if the metadata does not contain the required keys
        ``'X'``, ``'Y'``, and ``'Z'``.
    """

    _REQUIRED_METADATA = ("X", "Y", "Z")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        missing = [k for k in self._REQUIRED_METADATA if k not in self._metadata]
        if missing:
            raise ValueError(
                f"OPALOpacityTable requires metadata keys {self._REQUIRED_METADATA!r}; "
                f"missing: {missing}.  Available keys: {sorted(self._metadata)}."
            )

    # ---- Composition accessors ---------------------------------------- #

    @property
    def X(self) -> np.ndarray:
        r"""Hydrogen mass fraction array, shape ``(N,)``."""
        return self._metadata["X"]

    @property
    def Y(self) -> np.ndarray:
        r"""Helium mass fraction array, shape ``(N,)``."""
        return self._metadata["Y"]

    @property
    def Z(self) -> np.ndarray:
        r"""Metal mass fraction array, shape ``(N,)``."""
        return self._metadata["Z"]

    # ---- Composition selection ----------------------------------------- #

    def select_composition(self, X: float, Z: float) -> OPALOpacityTable:
        r"""Return the single table matching *X* and *Z* exactly.

        Parameters
        ----------
        X : float
            Hydrogen mass fraction.
        Z : float
            Metal mass fraction.

        Returns
        -------
        OPALOpacityTable
            A single-table instance (``n_tables == 1``).

        Raises
        ------
        ValueError
            If no table matches, or if more than one table matches (ambiguous).
        """
        indices = self.where(X=X, Z=Z)
        if len(indices) == 0:
            raise ValueError(f"No table found with X={X}, Z={Z}.  Use .where() to search available compositions.")
        if len(indices) > 1:
            raise ValueError(
                f"Ambiguous: {len(indices)} tables match X={X}, Z={Z} "
                f"(indices {indices}).  Refine with .where() or select by index."
            )
        return self.select(indices[0])
