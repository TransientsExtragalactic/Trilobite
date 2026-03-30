"""Opacity resolution utilities."""

from typing import Union

from triceratops.radiation.opacity.base import GreyOpacityLaw

# Registry of string names → class names in triceratops.radiation.opacity.models.core.
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
}


def get_opacity(opacity: Union[str, float, GreyOpacityLaw]) -> GreyOpacityLaw:
    r"""Resolve an opacity specification to a :class:`~triceratops.radiation.opacity.base.GreyOpacityLaw` instance.

    Parameters
    ----------
    opacity : str, float, or GreyOpacityLaw
        - **str**: a registered name (see below).
        - **float**: a constant opacity value in :math:`\\mathrm{cm^2\\,g^{-1}}`,
          wrapped in :class:`~.models.core.ConstantOpacity`.
        - **GreyOpacityLaw**: returned unchanged.

    Returns
    -------
    GreyOpacityLaw
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

    =====================  ======================================================
    ``"electron_scattering"``  Constant Thomson opacity (default: 0.34 cm² g⁻¹)
    ``"kramers_ff"``           Free-free Kramers :math:`\\kappa \\propto \\rho T^{-3.5}`
    ``"kramers_bf"``           Bound-free Kramers
    ``"kramers"``              Combined FF + BF Kramers
    ``"kramers_ff_es"``        Free-free Kramers + electron scattering
    ``"kramers_bf_es"``        Bound-free Kramers + electron scattering
    ``"kramers_es"``           Combined Kramers + electron scattering
    =====================  ======================================================
    """
    if isinstance(opacity, GreyOpacityLaw):
        return opacity
    if isinstance(opacity, (int, float)):
        from triceratops.radiation.opacity.models.core import ConstantOpacity

        return ConstantOpacity(float(opacity))
    if isinstance(opacity, str):
        if opacity not in _OPACITY_REGISTRY:
            raise ValueError(f"Unknown opacity {opacity!r}.  Available: {sorted(_OPACITY_REGISTRY)}.")
        from triceratops.radiation.opacity import models as _models

        return getattr(_models, _OPACITY_REGISTRY[opacity])()
    raise TypeError(f"opacity must be a str, float, or GreyOpacityLaw, got {type(opacity).__name__}.")
