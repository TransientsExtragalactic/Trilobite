r"""
Opacity resolution and utility functions.

This module provides helper functions for constructing and working with
opacity laws from flexible user input. It defines a registry-based
mechanism for resolving string identifiers into concrete
:class:`~triceratops.radiation.opacity.base.OpacityLaw` instances, along
with convenience wrappers for common opacity models.
"""

from typing import Optional, Union

from astropy import units as u

from triceratops.radiation.opacity.base import OpacityLaw
from triceratops.radiation.opacity.grey_opacity.base import ConstantGreyOpacity
from triceratops.radiation.opacity.grey_opacity.rosseland.models import (
    ElectronScatteringOpacity,
    KramersBFESOpacity,
    KramersBFOpacity,
    KramersESOpacity,
    KramersFFESOpacity,
    KramersFFOpacity,
    KramersOpacity,
    OPALOpacity,
)
from triceratops.radiation.opacity.grey_opacity.tops import TOPSOpacity

_OPACITY_REGISTRY: dict[str, type[OpacityLaw]] = {
    # ---- Rosseland mean: analytic grey approximations -------------------
    "electron_scattering": ElectronScatteringOpacity,
    "kramers_ff": KramersFFOpacity,
    "kramers_bf": KramersBFOpacity,
    "kramers": KramersOpacity,
    "kramers_ff_es": KramersFFESOpacity,
    "kramers_bf_es": KramersBFESOpacity,
    "kramers_es": KramersESOpacity,
    # ---- Rosseland mean: numerically exact OPAL table -------------------
    "opal": OPALOpacity,
    # ---- TOPS table (Rosseland or Planck; pass mean_type kwarg) ---------
    "tops": TOPSOpacity,
}

# ================================================================ #
# Opacity Locators
# ================================================================ #
# These functions all revolve around finding / loading opacities in
# user-friendly fashion.


def get_opacity(
    opacity: Union[str, float, u.Quantity, OpacityLaw],
    *,
    force_type: Optional[str] = None,
    **kwargs,
) -> OpacityLaw:
    r"""
    Resolve an opacity specification into an :class:`~triceratops.radiation.opacity.base.OpacityLaw` instance.

    This function provides a unified entry point for constructing opacity
    objects from flexible user input. The *opacity* argument may be a
    registered string key, a constant value, or an existing instance.

    Parameters
    ----------
    opacity : str, float, ~astropy.units.Quantity or ~triceratops.radiation.opacity.base.OpacityLaw
        The opacity to generate. The input may be any of the following:

        - :class:`str`: The name of a specific opacity type (see table below). In this case, the
          corresponding class is looked up in the internal registry and initialized. Additional
          keyword arguments are forwarded to the class's
          :meth:`~triceratops.radiation.opacity.base.OpacityLaw.load_default` method.
        - :class:`float`/:class:`int`/:class:`~astropy.units.Quantity`: Generate a constant opacity
          law with the specified
          value. Unit-bearing :class:`~astropy.units.Quantity` inputs are converted to
          :math:`\mathrm{cm^2\,g^{-1}}`; bare scalars are assumed to already be in those units.
          The result will be an instance of
          :class:`~triceratops.radiation.opacity.grey_opacity.base.ConstantGreyOpacity`,
          tagged with ``mean_type=force_type`` when *force_type* is supplied.
        - :class:`~triceratops.radiation.opacity.base.OpacityLaw`: Returned unchanged.

    force_type : str, optional
        If specified, the function will require that the resolved opacity match the specified type.
        Options are ``"rosseland"`` or ``"planck"``. If the resolved opacity's ``mean_type`` does not match, a
        ValueError is raised.

    **kwargs
        If the *opacity* argument is a string key, any additional keyword arguments are passed to the
        corresponding class's :meth:`~triceratops.radiation.opacity.base.OpacityLaw.load_default` method
        for initialization. This allows for flexible configuration of opacity instances based on user input.


    Returns
    -------
    ~triceratops.radiation.opacity.base.OpacityLaw
        A concrete opacity instance.

    Raises
    ------
    ValueError
        If *opacity* is an unknown string, if ``force_type`` does not match, or
        if a :class:`~astropy.units.Quantity` cannot be converted to :math:`\mathrm{cm^2\,g^{-1}}`.
    TypeError
        If *opacity* is not one of the accepted types.

    Notes
    -----
    Registered string names:

    =========================  ======================================================
    ``"electron_scattering"``  Constant Thomson opacity (default: 0.34 :math:`\mathrm{cm^2\,g^{-1}}`)
    ``"kramers_ff"``           Free-free Kramers :math:`\kappa \propto \rho T^{-3.5}`
    ``"kramers_bf"``           Bound-free Kramers
    ``"kramers"``              Combined free-free + bound-free Kramers
    ``"kramers_ff_es"``        Free-free Kramers + electron scattering
    ``"kramers_bf_es"``        Bound-free Kramers + electron scattering
    ``"kramers_es"``           Combined Kramers + electron scattering
    ``"opal"``                 OPAL Rosseland mean (default: X = 0.70, Z = 0.02)
    ``"tops"``                 TOPS solar table; pass ``mean_type="rosseland"`` (default)
                               or ``mean_type="planck"``
    =========================  ======================================================

    Examples
    --------
    .. code-block:: python

        from triceratops.radiation.opacity import (
            get_opacity,
        )

        op = get_opacity("kramers_es", kappa_es=0.20)
        op = get_opacity("opal", out_of_bounds="nan")
        op = get_opacity(0.34)
        op = get_opacity(0.34 * u.cm**2 / u.g)
        op = get_opacity(
            "kramers_es", force_type="rosseland"
        )
    """
    if isinstance(opacity, OpacityLaw):
        result = opacity
    elif isinstance(opacity, u.Quantity):
        try:
            result = ConstantGreyOpacity(float(opacity.to(u.cm**2 / u.g).value), mean_type=force_type)
        except u.UnitConversionError as exc:
            raise ValueError(f"Cannot convert opacity units {opacity.unit!r} to cm² g⁻¹.") from exc
    elif isinstance(opacity, (int, float)):
        result = ConstantGreyOpacity(float(opacity), mean_type=force_type)
    elif isinstance(opacity, str):
        if opacity not in _OPACITY_REGISTRY:
            raise ValueError(f"Unknown opacity {opacity!r}.  Available: {sorted(_OPACITY_REGISTRY)}.")
        result = _OPACITY_REGISTRY[opacity].load_default(**kwargs)
    else:
        raise TypeError(f"opacity must be a str, float, Quantity, or OpacityLaw, got {type(opacity).__name__}.")

    if force_type is not None and result.mean_type != force_type:
        raise ValueError(
            f"Resolved opacity has mean_type={result.mean_type!r}, but force_type={force_type!r} was requested."
        )
    return result


def load_opal_opacity(
    index: Optional[int] = None,
    *,
    X: Optional[float] = None,
    Z: Optional[float] = None,
    table_path=None,
    out_of_bounds: str = "raise",
):
    r"""
    Load an OPAL Rosseland mean opacity for a specified composition.

    The OPAL opacities provided by default in Triceratops are those from
    :footcite:t:`2005ASPC..336...25A`, which provide a number of different
    Rosseland mean opacity tables for various compositions. This function provides
    convenient access to these tables, allowing users to select a specific table by
    index or by specifying the hydrogen and metal mass fractions. The selected table is then
    loaded and returned as an instance of
    :class:`~triceratops.radiation.opacity.grey_opacity.rosseland.models.OPALOpacity`,
    which can be used to evaluate opacities and their derivatives at arbitrary points in the density-temperature plane.

    .. note::

        This is a convenience wrapper around
        :meth:`~triceratops.radiation.opacity.grey_opacity.rosseland.models.OPALOpacity.load_default`,
        providing a simple interface
        for selecting a table by index or composition.

    Parameters
    ----------
    index : int, optional
        If provided, selects the OPAL table by its zero-based index into the set of tables. Mutually exclusive with
        ``X`` and ``Z``.
    X : float, optional
        If provided, the correct OPAL table is identified based on the
        Hydrogen mass fraction and metallicity. Must be provided together with ``Z``.
    Z : float, optional
        If provided, the correct OPAL table is identified based on the
        Hydrogen mass fraction and metallicity. Must be provided together with ``X``.
    table_path : str, ~pathlib.Path, optional
        Path to an alternative HDF5 OPAL table. Defaults to the bundled
        :footcite:t:`2005ASPC..336...25A` dataset.
    out_of_bounds : {"raise", "clamp", "nan"}, optional
        The behavior of the resulting opacity object when evaluated at points outside the tabulated domain.
        The options are:

        - ``"raise"`` (default): Raise a ValueError if evaluated outside the table domain.
        - ``"clamp"``: Return the opacity at the nearest valid point within the table.
        - ``"nan"``: Return NaN for points outside the table domain.

    Returns
    -------
    ~triceratops.radiation.opacity.grey_opacity.rosseland.models.OPALOpacity
        A fully initialised OPAL opacity instance.

    Notes
    -----
    Exactly one selection method must be used:

    - ``index`` selects a table directly
    - ``X`` and ``Z`` select a table by composition

    If no arguments are provided, the default (solar composition)
    table is returned.

    Examples
    --------
    .. code-block:: python

        from triceratops.radiation.opacity import (
            load_opal_opacity,
        )

        # Solar composition (default)
        op = load_opal_opacity()

        # By table index
        op = load_opal_opacity(72)

        # By composition
        op = load_opal_opacity(X=0.74, Z=0.016)
    """
    return OPALOpacity.load_default(index=index, X=X, Z=Z, table_path=table_path, out_of_bounds=out_of_bounds)


def load_tops_opacity(
    *,
    mean_type: str = "rosseland",
    table_path=None,
    out_of_bounds: str = "raise",
):
    r"""Load a TOPS opacity from the bundled solar-composition table.

    Convenience wrapper around
    :meth:`~triceratops.radiation.opacity.grey_opacity.tops.TOPSOpacity.load_default`.

    Parameters
    ----------
    mean_type : {'rosseland', 'planck'}, optional
        Which mean opacity to use.  Default ``'rosseland'``.
    table_path : str or ~pathlib.Path, optional
        Path to an alternative TOPS text file parsed by
        :func:`~triceratops.radiation.opacity.opacity_io.read_tops`.
        If ``None``, the bundled ``tops_solar.dat`` is used.
    out_of_bounds : {'raise', 'clamp', 'nan'}, optional
        Behaviour when evaluated outside the tabulated domain.
        Default ``'raise'``.

    Returns
    -------
    ~triceratops.radiation.opacity.grey_opacity.tops.TOPSOpacity
        A fully initialised TOPS opacity instance.

    Examples
    --------
    .. code-block:: python

        from triceratops.radiation.opacity import (
            load_tops_opacity,
        )

        # Bundled solar table, Rosseland mean (default)
        op = load_tops_opacity()

        # Planck mean
        op = load_tops_opacity(mean_type="planck")

        # Custom file
        op = load_tops_opacity(
            table_path="my_tops_output.txt"
        )
    """
    return TOPSOpacity.load_default(mean_type=mean_type, table_path=table_path, out_of_bounds=out_of_bounds)
