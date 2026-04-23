r"""
Grey (frequency-averaged) opacity laws.

This subpackage provides opacity models in which the frequency dependence
of the opacity :math:`\kappa_\nu` has been integrated out, yielding an
effective opacity

.. math::

    \kappa = \kappa(\rho, T),

that depends only on density and temperature.

Grey opacities are used when the radiation field is sufficiently close
to thermal equilibrium that its detailed spectral structure can be
replaced by a single effective coupling between matter and radiation.
They are widely employed in radiative diffusion, stellar structure,
and simplified radiative transfer calculations.

Frequency averaging is not unique. Different physical regimes motivate
different definitions of the mean opacity:

**Rosseland mean**

    The Rosseland mean is a harmonic average weighted by the temperature
    derivative of the Planck function:

    .. math::

        \frac{1}{\kappa_R} =
        \frac{\displaystyle \int_0^\infty
              \frac{1}{\kappa_\nu}\,
              \frac{\partial B_\nu}{\partial T}\, d\nu}
             {\displaystyle \int_0^\infty
              \frac{\partial B_\nu}{\partial T}\, d\nu}.

    This weighting emphasizes the most transparent frequency channels,
    making :math:`\kappa_R` the appropriate opacity for radiative
    diffusion in optically thick media.

**Planck mean**

    The Planck mean is an emission-weighted (arithmetic) average:

    .. math::

        \kappa_P =
        \frac{\displaystyle \int_0^\infty
              \kappa_\nu\, B_\nu(T)\, d\nu}
             {\displaystyle \int_0^\infty
              B_\nu(T)\, d\nu}.

    This definition emphasizes the most emissive (and therefore most
    opaque) frequencies and is relevant for energy exchange and cooling
    in optically thin regimes.

Organization
------------
Opacity laws are grouped by their averaging prescription:

- :mod:`.rosseland` — Rosseland mean opacity laws
- :mod:`.tops` — TOPS table-based opacity (Rosseland and Planck mean)
- :mod:`.planck` — standalone analytic Planck mean laws (planned)

All classes in this subpackage inherit from
:class:`~triceratops.radiation.opacity.grey_opacity.base.GreyOpacityLaw` and
share a common interface for evaluating :math:`\kappa(\rho, T)` and its
logarithmic derivatives.
"""

from .base import ConstantGreyOpacity, GreyOpacityLaw
from .rosseland import (
    KAPPA_BF_0,
    KAPPA_FF_0,
    KAPPA_KR_0,
    ElectronScatteringOpacity,
    KramersBFESOpacity,
    KramersBFOpacity,
    KramersESOpacity,
    KramersFFESOpacity,
    KramersFFOpacity,
    KramersOpacity,
    OPALOpacity,
)
from .tops import TOPSOpacity

__all__ = [
    # Base
    "GreyOpacityLaw",
    "ConstantGreyOpacity",
    # Constants
    "KAPPA_FF_0",
    "KAPPA_BF_0",
    "KAPPA_KR_0",
    # Rosseland mean opacity laws
    "ElectronScatteringOpacity",
    "KramersFFOpacity",
    "KramersBFOpacity",
    "KramersOpacity",
    "KramersFFESOpacity",
    "KramersBFESOpacity",
    "KramersESOpacity",
    "OPALOpacity",
    "TOPSOpacity",
]
