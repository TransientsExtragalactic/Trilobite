"""Standard opacity law implementations for Triceratops."""

from pathlib import Path

import numpy as np

from triceratops.radiation.opacity.base import GreyOpacityLaw, OpacityLaw

#: Path to the bundled OPAL Rosseland opacity table (Asplund & Grevesse 2005).
_BUNDLED_OPAL_TABLE = Path(__file__).parent.parent / "tables" / "asplund_grevesse_05.h5"

# ------------------------------------------------------------------ #
#  Standard Kramers normalisation constants                           #
# ------------------------------------------------------------------ #

#: Free-free Kramers :math:`\kappa_0` (solar composition, no Gaunt-factor correction).
KAPPA_FF_0: float = 3.68e22
#: Bound-free Kramers :math:`\kappa_0` (solar composition, no Gaunt-factor correction).
KAPPA_BF_0: float = 4.34e25
#: Combined (FF + BF) Kramers :math:`\kappa_0`.  Valid because both terms share the
#: same :math:`\rho` and :math:`T` power-law indices, so they add as a single
#: effective normalisation.
KAPPA_KR_0: float = KAPPA_FF_0 + KAPPA_BF_0


# ================================================================ #
#  Opacities                                                       #
# ================================================================ #
class ConstantOpacity(GreyOpacityLaw):
    r"""Opacity law with a fixed, user-specified :math:`\kappa`.

    Pure-Python (no Cython extension).

    Parameters
    ----------
    kappa : float
        Opacity in :math:`\mathrm{cm^2\,g^{-1}}`.
    """

    def __init__(self, kappa: float):
        # Set _LOG_KAPPA as an instance attribute, shadowing the class default.
        # GreyOpacityLaw._log_opacity returns self._LOG_KAPPA, so this is all
        # that is needed to fix the constant.
        self._LOG_KAPPA = np.log(float(kappa))
        super().__init__()

    @property
    def kappa(self) -> float:
        r"""Opacity value in :math:`\mathrm{cm^2\,g^{-1}}`."""
        return float(np.exp(self._LOG_KAPPA))


class ElectronScatteringOpacity(GreyOpacityLaw):
    r"""Constant electron-scattering (Thomson) opacity.

    .. math::

        \kappa_{\rm es} = \frac{\sigma_T}{m_p}\,\frac{1+X}{2}
                        \approx 0.34\,\mathrm{cm^2\,g^{-1}}

    for fully ionised solar composition.

    Parameters
    ----------
    kappa_es : float, optional
        Electron-scattering opacity in :math:`\mathrm{cm^2\,g^{-1}}`.  Default 0.34.
    """

    IS_C_BACKED = True

    #: Exposed so callers can read back the value without going through the C object.
    kappa_es: float

    def __init__(self, kappa_es: float = 0.34):
        self.kappa_es = kappa_es
        # Keep _LOG_KAPPA in sync for introspection (not reached via _log_opacity
        # when IS_C_BACKED=True, but makes the object inspectable without Cython).
        self._LOG_KAPPA = np.log(kappa_es)
        super().__init__(kappa_es=kappa_es)

    def _initialize_C_object(self, kappa_es=0.34):
        from triceratops.radiation.opacity.models._electron_scattering import (
            C_ElectronScatteringOpacity,
        )

        # noinspection PyArgumentList
        return C_ElectronScatteringOpacity(kappa_es=kappa_es)


class KramersFFOpacity(GreyOpacityLaw):
    r"""Free-free (bremsstrahlung) Kramers opacity.

    .. math::

        \kappa_{\rm ff}(\rho, T) = \kappa_0\,\rho\,T^{-3.5}

    with the standard solar-composition default:

    .. math::

        \kappa_{\rm ff,0} \approx 3.68\times10^{22}\ {\rm cm^5\,g^{-2}\,K^{3.5}}

    Parameters
    ----------
    kappa0 : float, optional
        Normalisation constant in :math:`\mathrm{cm^5\,g^{-2}\,K^{3.5}}`.  Defaults to
        :data:`~triceratops.radiation.opacity.models.core.KAPPA_FF_0`.  Override with a composition- or
        Gaunt-factor-corrected value when available.
    """

    IS_C_BACKED = True

    def __init__(self, kappa0: float = KAPPA_FF_0):
        self.kappa0 = kappa0
        super().__init__(kappa0=kappa0)

    def _initialize_C_object(self, kappa0):
        from triceratops.radiation.opacity.models._kramers import C_KramersOpacity

        # noinspection PyArgumentList
        return C_KramersOpacity(kappa0=kappa0)


class KramersBFOpacity(GreyOpacityLaw):
    r"""Bound-free (photoionisation) Kramers opacity.

    .. math::

        \kappa_{\rm bf}(\rho, T) = \kappa_0\,\rho\,T^{-3.5}

    with the standard solar-composition default:

    .. math::

        \kappa_{\rm bf,0} \approx 4.34\times10^{25}\ {\rm cm^5\,g^{-2}\,K^{3.5}}

    Parameters
    ----------
    kappa0 : float, optional
        Normalisation constant in :math:`\mathrm{cm^5\,g^{-2}\,K^{3.5}}`.  Defaults to
        :data:`~triceratops.radiation.opacity.models.core.KAPPA_BF_0`.
    """

    IS_C_BACKED = True

    def __init__(self, kappa0: float = KAPPA_BF_0):
        self.kappa0 = kappa0
        super().__init__(kappa0=kappa0)

    def _initialize_C_object(self, kappa0):
        from triceratops.radiation.opacity.models._kramers import C_KramersOpacity

        # noinspection PyArgumentList
        return C_KramersOpacity(kappa0=kappa0)


class KramersOpacity(GreyOpacityLaw):
    r"""Combined (free-free + bound-free) Kramers opacity.

    Because both terms share the same power-law dependence on :math:`\rho` and
    :math:`T`, they combine into a single effective normalisation:

    .. math::

        \kappa_{\rm ff+bf}(\rho, T) = (\kappa_{\rm ff,0} + \kappa_{\rm bf,0})\,\rho\,T^{-3.5}

    with the standard solar-composition default:

    .. math::

        \kappa_{\rm ff+bf,0} \approx 4.34\times10^{25}\ {\rm cm^5\,g^{-2}\,K^{3.5}}

    (dominated by the bound-free term).

    Parameters
    ----------
    kappa0 : float, optional
        Combined normalisation in :math:`\mathrm{cm^5\,g^{-2}\,K^{3.5}}`.  Defaults to
        :data:`~triceratops.radiation.opacity.models.core.KAPPA_KR_0` = ``KAPPA_FF_0 + KAPPA_BF_0``.
    """

    IS_C_BACKED = True

    def __init__(self, kappa0: float = KAPPA_KR_0):
        self.kappa0 = kappa0
        super().__init__(kappa0=kappa0)

    def _initialize_C_object(self, kappa0):
        from triceratops.radiation.opacity.models._kramers import C_KramersOpacity

        # noinspection PyArgumentList
        return C_KramersOpacity(kappa0=kappa0)


# ================================================================ #
#  Kramers + electron-scattering (combined) opacities             #
# ================================================================ #


class KramersFFESOpacity(GreyOpacityLaw):
    r"""Free-free Kramers opacity plus electron scattering.

    .. math::

        \kappa(\rho, T) = \kappa_{\rm es} + \kappa_{\rm ff,0}\,\rho\,T^{-3.5}

    Parameters
    ----------
    kappa0 : float, optional
        Free-free normalisation in :math:`\mathrm{cm^5\,g^{-2}\,K^{3.5}}`.
        Defaults to :data:`~triceratops.radiation.opacity.models.core.KAPPA_FF_0`.
    kappa_es : float, optional
        Electron-scattering opacity in :math:`\mathrm{cm^2\,g^{-1}}`.
        Default 0.34.
    """

    IS_C_BACKED = True

    def __init__(self, kappa0: float = KAPPA_FF_0, kappa_es: float = 0.34):
        self.kappa0 = kappa0
        self.kappa_es = kappa_es
        super().__init__(kappa0=kappa0, kappa_es=kappa_es)

    def _initialize_C_object(self, kappa0, kappa_es=0.34):
        from triceratops.radiation.opacity.models._kramers_es import C_KramersESOpacity

        return C_KramersESOpacity(kappa0=kappa0, kappa_es=kappa_es)


class KramersBFESOpacity(GreyOpacityLaw):
    r"""Bound-free Kramers opacity plus electron scattering.

    .. math::

        \kappa(\rho, T) = \kappa_{\rm es} + \kappa_{\rm bf,0}\,\rho\,T^{-3.5}

    Parameters
    ----------
    kappa0 : float, optional
        Bound-free normalisation in :math:`\mathrm{cm^5\,g^{-2}\,K^{3.5}}`.
        Defaults to :data:`~triceratops.radiation.opacity.models.core.KAPPA_BF_0`.
    kappa_es : float, optional
        Electron-scattering opacity in :math:`\mathrm{cm^2\,g^{-1}}`.
        Default 0.34.
    """

    IS_C_BACKED = True

    def __init__(self, kappa0: float = KAPPA_BF_0, kappa_es: float = 0.34):
        self.kappa0 = kappa0
        self.kappa_es = kappa_es
        super().__init__(kappa0=kappa0, kappa_es=kappa_es)

    def _initialize_C_object(self, kappa0, kappa_es=0.34):
        from triceratops.radiation.opacity.models._kramers_es import C_KramersESOpacity

        return C_KramersESOpacity(kappa0=kappa0, kappa_es=kappa_es)


class KramersESOpacity(GreyOpacityLaw):
    r"""Combined (free-free + bound-free) Kramers opacity plus electron scattering.

    .. math::

        \kappa(\rho, T) = \kappa_{\rm es} + (\kappa_{\rm ff,0} + \kappa_{\rm bf,0})\,\rho\,T^{-3.5}

    The most physically complete grey opacity for a hot, fully ionised plasma
    where both Thomson scattering and thermal bremsstrahlung/photoionisation
    contribute.

    Parameters
    ----------
    kappa0 : float, optional
        Combined Kramers normalisation in :math:`\mathrm{cm^5\,g^{-2}\,K^{3.5}}`.
        Defaults to :data:`~triceratops.radiation.opacity.models.core.KAPPA_KR_0` = ``KAPPA_FF_0 + KAPPA_BF_0``.
    kappa_es : float, optional
        Electron-scattering opacity in :math:`\mathrm{cm^2\,g^{-1}}`.
        Default 0.34.
    """

    IS_C_BACKED = True

    def __init__(self, kappa0: float = KAPPA_KR_0, kappa_es: float = 0.34):
        self.kappa0 = kappa0
        self.kappa_es = kappa_es
        super().__init__(kappa0=kappa0, kappa_es=kappa_es)

    def _initialize_C_object(self, kappa0, kappa_es=0.34):
        from triceratops.radiation.opacity.models._kramers_es import C_KramersESOpacity

        return C_KramersESOpacity(kappa0=kappa0, kappa_es=kappa_es)


# ================================================================ #
#  OPAL Rosseland mean opacity                                     #
# ================================================================ #

#: Out-of-bounds mode codes passed to C_OPALTableOpacity.
_OOB_MODES = {"raise": 0, "clamp": 1, "nan": 2}
#: Coordinate-system codes passed to C_OPALTableOpacity.
_COORD_MODES = {"T_R": 0, "T_rho": 1}


class OPALOpacity(OpacityLaw):
    r"""Rosseland mean opacity evaluated by bilinear interpolation on a 2-D OPAL table.

    The table must represent a *single composition*; pass a single-table
    :class:`~triceratops.radiation.opacity.tables.opacity_table.OpacityTable`
    (i.e. one with ``n_tables == 1``).  Use
    :meth:`~triceratops.radiation.opacity.tables.opacity_table.OPALOpacityTable.select`
    or
    :func:`load_opal_opacity` to extract a single table from a multi-table file.

    Parameters
    ----------
    table : OpacityTable
        A single-table opacity table (``n_tables == 1``).  Both
        ``'T_R'`` and ``'T_rho'`` coordinate systems are supported.
    out_of_bounds : {'raise', 'clamp', 'nan'}
        Behaviour when a query ``(T, rho)`` falls outside the table's valid
        domain (including NaN-flagged cells):

        * ``'raise'`` — :exc:`ValueError` with the out-of-range coordinates.
        * ``'clamp'`` — return the nearest boundary value silently.
        * ``'nan'``   — return ``NaN`` silently.

    Examples
    --------
    Load solar composition from the bundled Asplund & Grevesse (2005) table:

    >>> from triceratops.radiation.opacity.models.core import (
    ...     load_opal_opacity,
    ... )
    >>> op = load_opal_opacity(
    ...     index=72
    ... )  # solar: X=0.70, Z=0.02
    >>> import astropy.units as u
    >>> op.opacity(1e-5 * u.g / u.cm**3, 1e7 * u.K)
    """

    IS_C_BACKED = True

    def __init__(self, table, *, out_of_bounds: str = "raise"):
        if out_of_bounds not in _OOB_MODES:
            raise ValueError(f"out_of_bounds must be one of {list(_OOB_MODES)}, got {out_of_bounds!r}.")
        if table.n_tables != 1:
            raise ValueError(
                f"OPALOpacity requires a single-table OpacityTable "
                f"(n_tables == 1), got {table.n_tables}.  "
                f"Use table.select(index) first."
            )
        if table.coord_system not in _COORD_MODES:
            raise ValueError(f"Unsupported coord_system {table.coord_system!r}; expected one of {list(_COORD_MODES)}.")
        self.table = table
        self.out_of_bounds = out_of_bounds
        super().__init__(table=table, out_of_bounds=out_of_bounds)

    def _initialize_C_object(self, *, table, out_of_bounds):
        from triceratops.radiation.opacity.models._opal_table import C_OPALTableOpacity

        # np.array always copies, giving the writable C-contiguous buffer Cython requires.
        # (table.grid_1 / table.opacity are read-only views; we need writable copies here.)
        lk = np.array(table.opacity[0], dtype=np.float64, order="C")
        return C_OPALTableOpacity(
            np.array(table.grid_1, dtype=np.float64),
            np.array(table.grid_2, dtype=np.float64),
            lk,
            _COORD_MODES[table.coord_system],
            _OOB_MODES[out_of_bounds],
        )

    def _log_opacity(self, log_T, log_rho):
        if np.ndim(log_T) > 0:
            log_T_c = np.ascontiguousarray(log_T, dtype=np.float64)
            log_rho_c = np.ascontiguousarray(log_rho, dtype=np.float64)
            return self._c_object.log_opacity_array(log_T_c, log_rho_c)
        return self._c_object.log_opacity(log_T, log_rho)

    def _dlogkappa_dlogrho(self, log_T, log_rho):
        if np.ndim(log_T) > 0:
            log_T_c = np.ascontiguousarray(log_T, dtype=np.float64)
            log_rho_c = np.ascontiguousarray(log_rho, dtype=np.float64)
            return self._c_object.dlogkappa_dlogrho_array(log_T_c, log_rho_c)
        return self._c_object.dlogkappa_dlogrho(log_T, log_rho)

    def _dlogkappa_dlogT(self, log_T, log_rho):
        if np.ndim(log_T) > 0:
            log_T_c = np.ascontiguousarray(log_T, dtype=np.float64)
            log_rho_c = np.ascontiguousarray(log_rho, dtype=np.float64)
            return self._c_object.dlogkappa_dlogT_array(log_T_c, log_rho_c)
        return self._c_object.dlogkappa_dlogT(log_T, log_rho)

    def __repr__(self) -> str:
        tbl = self.table
        meta = tbl.metadata
        comp_parts = [f"{k}={meta[k][0]:.4g}" for k in ("X", "Y", "Z") if k in meta]
        comp_str = ", ".join(comp_parts) if comp_parts else "unknown composition"
        return (
            f"OPALOpacity({comp_str}, "
            f"n_T={tbl.grid_1.size}, n_R={tbl.grid_2.size}, "
            f"out_of_bounds={self.out_of_bounds!r})"
        )


def _load_solar_opal_opacity(
    *,
    table_path=None,
    out_of_bounds: str = "raise",
    X: float = 0.70,
    Z: float = 0.02,
) -> "OPALOpacity":
    """Load solar-composition OPAL opacity by searching the bundled table.

    Uses :meth:`~triceratops.radiation.opacity.tables.opacity_table.OPALOpacityTable.where`
    to locate the composition dynamically rather than relying on a hardcoded index,
    so the result is stable even if the bundled HDF5 file is rebuilt.

    Parameters
    ----------
    table_path : str or Path, optional
        Override the default bundled table path.
    out_of_bounds : {'raise', 'clamp', 'nan'}, optional
    X : float
        Hydrogen mass fraction to search for (default 0.70).
    Z : float
        Metal mass fraction to search for (default 0.02).

    Returns
    -------
    OPALOpacity
    """
    from triceratops.radiation.opacity.tables.opacity_table import OPALOpacityTable

    path = table_path if table_path is not None else _BUNDLED_OPAL_TABLE
    tbl = OPALOpacityTable.read(path)
    matches = tbl.where(X=X, Z=Z)
    if not matches:
        raise RuntimeError(f"Solar composition (X={X}, Z={Z}) not found in bundled OPAL table {path}.")
    return OPALOpacity(tbl.select(matches[0]), out_of_bounds=out_of_bounds)


def load_opal_opacity(
    index: int,
    *,
    table_path=None,
    out_of_bounds: str = "raise",
) -> OPALOpacity:
    """Load a single-composition :class:`OPALOpacity` from an HDF5 table file.

    Parameters
    ----------
    index : int
        Zero-based index of the table to load.
    table_path : str or Path, optional
        Path to an HDF5 OPAL table file.  Defaults to the bundled
        Asplund & Grevesse (2005) table.
    out_of_bounds : {'raise', 'clamp', 'nan'}, optional
        Out-of-bounds behaviour forwarded to :class:`OPALOpacity`.

    Returns
    -------
    OPALOpacity

    Notes
    -----
    In the bundled Asplund & Grevesse (2005) table, index 72 corresponds to
    solar composition (X = 0.70, Z = 0.02).  You can verify this with::

        tbl = OPALOpacityTable.read(path)
        print(tbl.where(X=0.70, Z=0.02))  # → [72]

    To load solar composition without hardcoding the index, use
    :func:`_load_solar_opal_opacity` (or ``get_opacity("opal")``).

    Examples
    --------
    >>> from triceratops.radiation.opacity.models.core import (
    ...     load_opal_opacity,
    ... )
    >>> op = load_opal_opacity(
    ...     72
    ... )  # solar composition (X=0.70, Z=0.02)
    """
    from triceratops.radiation.opacity.tables.opacity_table import OPALOpacityTable

    path = table_path if table_path is not None else _BUNDLED_OPAL_TABLE
    tbl = OPALOpacityTable.read(path)
    return OPALOpacity(tbl.select(index), out_of_bounds=out_of_bounds)
