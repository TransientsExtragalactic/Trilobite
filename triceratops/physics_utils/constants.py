"""
Hard-coded physical constants in CGS units and their logarithms.

This module defines physical constants in CGS units and their logarithms for use across the physics module.  The
constants are sourced from `astropy.constants` and converted to CGS units.  The logarithms are pre-computed for
efficiency in calculations that involve logarithmic expressions.
"""

import numpy as np
from astropy import constants as const

__all__ = [
    "G_cgs",
    "electron_scattering_opacity_cgs",
    "sigma_sb_cgs",
    "k_B_cgs",
    "m_p_cgs",
    "c_cgs",
    "a_rad_cgs",
    "_log_G_cgs",
    "_log_kappa_es_cgs",
    "_log_sigma_sb_cgs",
    "_log_k_B_cgs",
    "_log_m_p_cgs",
    "_log_c_cgs",
    "_log_a_rad_cgs",
]

# ========================================= #
# Physical Constants in CGS Units           #
# ========================================= #
G_cgs = const.G.cgs.value  # Gravitational constant  [cm^3 g^-1 s^-2]
_log_G_cgs = np.log(G_cgs)

electron_scattering_opacity_cgs = 0.34  # Electron-scattering opacity for fully ionized H  [cm^2 g^-1]
_log_kappa_es_cgs = np.log(electron_scattering_opacity_cgs)

sigma_sb_cgs = const.sigma_sb.cgs.value  # Stefan-Boltzmann constant  [erg cm^-2 s^-1 K^-4]
_log_sigma_sb_cgs = np.log(sigma_sb_cgs)

k_B_cgs = const.k_B.cgs.value  # Boltzmann constant  [erg K^-1]
_log_k_B_cgs = np.log(k_B_cgs)

m_p_cgs = const.m_p.cgs.value  # Proton mass  [g]
_log_m_p_cgs = np.log(m_p_cgs)

c_cgs = const.c.cgs.value  # Speed of light  [cm s^-1]
_log_c_cgs = np.log(c_cgs)

# Radiation (energy) density constant: a_rad = 4 sigma_sb / c  [erg cm^-3 K^-4]
a_rad_cgs = 4.0 * sigma_sb_cgs / c_cgs
_log_a_rad_cgs = np.log(a_rad_cgs)
