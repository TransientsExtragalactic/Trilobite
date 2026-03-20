"""Accretion physics utility functions for one-zone models."""

from typing import TYPE_CHECKING, Optional

import numpy as np
from astropy import constants as const
from astropy import units as u

from triceratops.utils.misc_utils import ensure_in_units

if TYPE_CHECKING:
    from triceratops._typing import _UnitBearingArrayLike


# =============================================== #
# Physics Utilities                               #
# =============================================== #
def compute_ISCO(
    mass: "_UnitBearingArrayLike",
    spin: Optional[float] = None,
    J: Optional["_UnitBearingArrayLike"] = None,
    prograde: bool = True,
) -> u.Quantity:
    r"""
    Compute the innermost stable circular orbit (ISCO) radius of a Kerr black hole.

    The ISCO radius depends on the black hole mass and dimensionless spin
    parameter :math:`a_\star \in [-1, 1]`. This function supports either direct
    specification of the spin or derivation from the angular momentum.

    The ISCO radius is computed using the standard Kerr metric formula
    (Bardeen et al. 1972):

    .. math::

        r_{\rm ISCO} = r_g \left[ 3 + Z_2 \mp \sqrt{(3 - Z_1)(3 + Z_1 + 2Z_2)} \right]

    where:

    .. math::

        r_g &= \frac{GM}{c^2} \\
        Z_1 &= 1 + (1 - a_\star^2)^{1/3}
              \left[ (1 + a_\star)^{1/3} + (1 - a_\star)^{1/3} \right] \\
        Z_2 &= \sqrt{3 a_\star^2 + Z_1^2}

    The minus sign corresponds to **prograde** orbits, and the plus sign to
    **retrograde** orbits.

    The dimensionless spin parameter is defined as:

    .. math::

        a_\star = \frac{c J}{G M^2}

    Parameters
    ----------
    mass : `~astropy.units.Quantity`
        Black hole mass. Must have mass units.

    spin : float, optional
        Dimensionless Kerr spin parameter :math:`a_\star \in [-1, 1]`.
        If not provided, it will be computed from ``J`` if available.

    J : `~astropy.units.Quantity`, optional
        Black hole angular momentum. Used to compute the spin if
        ``spin`` is not provided.

    prograde : bool, optional
        If ``True`` (default), compute the ISCO for prograde orbits.
        If ``False``, compute for retrograde orbits.

    Returns
    -------
    `~astropy.units.Quantity`
        ISCO radius in units of length.

    Raises
    ------
    ValueError
        If both ``spin`` and ``J`` are provided.
        If the computed or supplied spin lies outside [-1, 1].

    Notes
    -----
    - For a Schwarzschild black hole (:math:`a_\star = 0`):

      .. math::

          r_{\rm ISCO} = 6 \frac{GM}{c^2}

    - For a maximally spinning Kerr black hole:

      - Prograde: :math:`r_{\rm ISCO} = GM/c^2`
      - Retrograde: :math:`r_{\rm ISCO} = 9 GM/c^2`

    Examples
    --------
    Compute ISCO for a non-spinning black hole:

    .. code-block:: python

        from astropy import units as u
        from triceratops.physics import compute_ISCO

        r_isco = compute_ISCO(10 * u.Msun)
        print(r_isco)

    Compute ISCO for a spinning black hole:

    .. code-block:: python

        r_isco = compute_ISCO(10 * u.Msun, spin=0.9)
        print(r_isco)

    Compute ISCO from angular momentum:

    .. code-block:: python

        from astropy import constants as const

        M = 10 * u.Msun
        J = 0.9 * const.G * M**2 / const.c

        r_isco = compute_ISCO(M, J=J)
        print(r_isco)
    """
    # --- Convert mass to CGS --- #
    M = ensure_in_units(mass, u.g)

    # --- Determine spin --- #
    if spin is not None and J is not None:
        raise ValueError("Provide either 'spin' or 'J', not both.")

    if spin is None:
        if J is not None:
            J_cgs = ensure_in_units(J, u.g * u.cm**2 / u.s)
            spin = (const.c * J_cgs / (const.G * M**2)).decompose().value
        else:
            spin = 0.0

    spin = float(spin)

    if not (-1.0 <= spin <= 1.0):
        raise ValueError(f"Spin must be in [-1, 1], got {spin}.")

    # --- Kerr ISCO formula --- #
    Z1 = 1 + (1 - spin**2) ** (1 / 3) * ((1 + spin) ** (1 / 3) + (1 - spin) ** (1 / 3))
    Z2 = np.sqrt(3 * spin**2 + Z1**2)

    if prograde:
        r_hat = 3 + Z2 - np.sqrt((3 - Z1) * (3 + Z1 + 2 * Z2))
    else:
        r_hat = 3 + Z2 + np.sqrt((3 - Z1) * (3 + Z1 + 2 * Z2))

    # --- Convert to physical units --- #
    r_g = (const.G * M / const.c**2).to(u.cm)
    r_isco = r_hat * r_g

    return r_isco
