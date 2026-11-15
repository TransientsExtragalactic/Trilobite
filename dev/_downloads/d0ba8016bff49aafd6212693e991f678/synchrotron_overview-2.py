import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy import constants as const

from trilobite.radiation.synchrotron.SEDs.numerical import NumericalSynchrotronEngine
from trilobite.radiation.synchrotron.electron_distributions import (
    MaxwellJuettner,
    PowerLaw,
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

epsilon_B = 0.01
epsilon_T = 0.1   # ratio of the thermal-electron to the ion energy
epsilon_E = 1e-6  # non-thermal electrons: a small share of the energy

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
N_therm = MaxwellJuettner.normalize_from_magnetic_field(
    B=B, epsilon_B=epsilon_B, epsilon_T=epsilon_T, epsilon_E=epsilon_E, Theta=Theta
)
N_pl = PowerLaw.normalize_from_magnetic_field(
    B=B, epsilon_B=epsilon_B, epsilon_E=epsilon_E,
    p=p, gamma_min=gamma_min, gamma_max=gamma_max,
)

mjd = MaxwellJuettner.as_callable(norm=N_therm.value, Theta=Theta)
pl  = PowerLaw.as_callable(norm=N_pl.value, p=p,
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