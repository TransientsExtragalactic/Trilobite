import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy import constants as const

from triceratops.radiation.synchrotron.SEDs.numerical import NumericalSynchrotronEngine
from triceratops.radiation.synchrotron.microphysics import (
    compute_MJD_and_PL_norm_from_magnetic_field,
    get_maxwell_juttner_distribution,
    get_power_law_distribution,
)

# ---------------------------------------------------------------------
# Initialize synchrotron engine
# ---------------------------------------------------------------------
engine = NumericalSynchrotronEngine()
engine.load_avg_first_kernel()

# ---------------------------------------------------------------------
# Physical parameters
# ---------------------------------------------------------------------
B = 1.0 * u.G
R = 1e17 * u.cm

epsilon_e = 0.1
epsilon_B = 0.01
delta = 0.99999   # fraction of energy in thermal component

# Thermal population
T = 1e10 * u.K
Theta = (const.k_B * T / (const.m_e * const.c**2)).decompose().value

# Power-law population
p = 3.0
gamma_min = 50.0
gamma_max = 1e10

# ---------------------------------------------------------------------
# Normalize electron distributions (equipartition closure)
# ---------------------------------------------------------------------
N_therm, N_pl = compute_MJD_and_PL_norm_from_magnetic_field(
    B=B,
    Theta=Theta,
    p=p,
    delta=delta,
    epsilon_E=epsilon_e,
    epsilon_B=epsilon_B,
    gamma_min=gamma_min,
    gamma_max=gamma_max,
)

mjd = get_maxwell_juttner_distribution(Theta, norm=N_therm)
pl  = get_power_law_distribution(p=p, norm=N_pl,
                                 gamma_min=gamma_min, gamma_max=gamma_max)

# ---------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------
gamma = np.geomspace(1, gamma_max, 200)
nu = np.geomspace(1e-2, 1e4, 200) * u.GHz

# ---------------------------------------------------------------------
# Compute synchrotron specific intensity
# ---------------------------------------------------------------------
I_mjd = engine.compute_specific_intensity(nu, R, B, mjd(gamma), gamma)
I_pl  = engine.compute_specific_intensity(nu, R, B, pl(gamma), gamma)
I_tot = engine.compute_specific_intensity(nu, R, B,
                                          mjd(gamma) + pl(gamma), gamma)

# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))

ax.loglog(nu, I_mjd, label='Thermal (Maxwell–Jüttner)', lw=2)
ax.loglog(nu, I_pl,  label='Power-law', lw=2)
ax.loglog(nu, I_tot, '--', label='Combined', lw=2)

ax.set_xlabel('Frequency [GHz]')
ax.set_ylabel(r'$I_\nu$ [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$ sr$^{-1}$]')
ax.set_title('Synchrotron emission from thermal and non-thermal electrons')

ax.legend()
ax.grid(True, which='both', ls='--', alpha=0.4)
ax.set_ylim(1e-11, 1e-6)

plt.tight_layout()
plt.show()