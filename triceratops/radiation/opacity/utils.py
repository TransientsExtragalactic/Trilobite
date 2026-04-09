"""Opacity resolution utilities."""

from typing import Union

from triceratops.radiation.opacity.base import OpacityLaw

# Registry of string names → zero-argument callables that return OpacityLaw instances.
# Analytic laws map to class names in triceratops.radiation.opacity.models.core (resolved
# lazily via getattr); "opal" uses _load_solar_opal_opacity to locate solar composition
# dynamically rather than relying on a hardcoded table index.
# This is the single source of truth for the opacity string interface used across
# the codebase (disk models, diagnostic utils, etc.).
_OPACITY_REGISTRY: dict[str, str] = {
    "electron_scattering": "ElectronScatteringOpacity",
    "kramers_ff": "KramersFFOpacity",
    "kramers_bf": "KramersBFOpacity",
    "kramers": "KramersOpacity",
    "kramers_ff_es": "KramersFFESOpacity",
    "kramers_bf_es": "KramersBFESOpacity",
    "kramers_es": "KramersESOpacity",
    # OPAL Rosseland mean opacity — resolved dynamically by composition search.
    "opal": "opal",
}


def get_opacity(opacity: Union[str, float, OpacityLaw]) -> OpacityLaw:
    r"""Resolve an opacity specification to an :class:`~triceratops.radiation.opacity.base.OpacityLaw` instance.

    Parameters
    ----------
    opacity : str, float, or OpacityLaw
        - **str**: a registered name (see table below).
        - **float** or **int**: a constant opacity value in
          :math:`\mathrm{cm^2\,g^{-1}}`, wrapped in
          :class:`~.models.core.ConstantOpacity`.
        - **OpacityLaw**: returned unchanged.

    Returns
    -------
    OpacityLaw
        A concrete opacity instance.

    Raises
    ------
    ValueError
        If *opacity* is a string not in the registry.
    TypeError
        If *opacity* is none of the accepted types.

    Notes
    -----
    Registered string names:

    =========================  ======================================================
    ``"electron_scattering"``  Constant Thomson opacity (default: 0.34 cm² g⁻¹)
    ``"kramers_ff"``           Free-free Kramers :math:`\kappa \propto \rho T^{-3.5}`
    ``"kramers_bf"``           Bound-free Kramers
    ``"kramers"``              Combined FF + BF Kramers
    ``"kramers_ff_es"``        Free-free Kramers + electron scattering
    ``"kramers_bf_es"``        Bound-free Kramers + electron scattering
    ``"kramers_es"``           Combined Kramers + electron scattering
    ``"opal"``                 OPAL bilinear-interpolated Rosseland mean
                               (solar composition: X = 0.70, Z = 0.02)
    =========================  ======================================================

    .. note::

       ``"opal"`` defaults to solar composition (X = 0.70, Z = 0.02) from the
       bundled Asplund & Grevesse (2005) table.  The composition is located by
       searching the table metadata rather than relying on a hardcoded index.
       To load a different composition, use
       :func:`~triceratops.radiation.opacity.models.core.load_opal_opacity` directly::

           from triceratops.radiation.opacity.models.core import (
               load_opal_opacity,
           )

           op = load_opal_opacity(index=50)
    """
    if isinstance(opacity, OpacityLaw):
        return opacity
    if isinstance(opacity, (int, float)):
        from triceratops.radiation.opacity.models.core import ConstantOpacity

        return ConstantOpacity(float(opacity))
    if isinstance(opacity, str):
        if opacity not in _OPACITY_REGISTRY:
            raise ValueError(f"Unknown opacity {opacity!r}.  Available: {sorted(_OPACITY_REGISTRY)}.")
        if opacity == "opal":
            from triceratops.radiation.opacity.models.core import _load_solar_opal_opacity

            return _load_solar_opal_opacity()
        from triceratops.radiation.opacity import models as _models

        return getattr(_models, _OPACITY_REGISTRY[opacity])()
    raise TypeError(f"opacity must be a str, float, or OpacityLaw, got {type(opacity).__name__}.")
