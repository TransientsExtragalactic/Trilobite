"""Constants for various physical calculations across the physics module."""

from astropy import constants as const
from astropy import units as u

# ========================================= #
# Synchrotron Radiation Constants           #
# ========================================= #
electron_rest_energy: u.Quantity = const.m_e * const.c**2
r"""astropy.units.Quantity: Electron rest energy :math:`E_0`."""
electron_rest_energy_cgs: float = electron_rest_energy.cgs.value
"""float: Electron rest energy :math:`E_0` in CGS units."""
sigma_T_cgs: float = const.sigma_T.cgs.value
r"""float: Thomson cross-section :math:`\sigma_T` in CGS units."""
c_cgs: float = const.c.cgs.value
"""float: Speed of light :math:`c` in CGS units."""
electron_rest_mass_cgs: float = const.m_e.cgs.value
"""float: Electron rest mass :math:`m_e` in CGS units."""

# ========================================= #
# Free Free Emission Constants              #
# ========================================= #
h_cgs: float = const.h.cgs.value
"""float: Planck constant :math:`h` in CGS units."""
kB_cgs: float = const.k_B.cgs.value
"""float: Boltzmann constant :math:`k_B` in CGS units."""
Ry_cgs: float = (const.Ryd * const.h * const.c).cgs.value
"""float: Rydberg energy :math:`R_y = h c R_{\\infty}` in CGS units (erg)."""
