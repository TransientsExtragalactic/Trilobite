import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from trilobite.radiation.synchrotron.SEDs.numerical import NumericalSynchrotronEngine
from trilobite.radiation.synchrotron.microphysics import (
    compute_PL_norm_from_magnetic_field,
    compute_BPL_norm_from_magnetic_field,
    get_power_law_distribution,
    get_broken_power_law_distribution,
)

# ---------------------------------------------------------------------
# Initialize synchrotron engine
# ---------------------------------------------------------------------
engine = NumericalSynchrotronEngine()
engine.load_avg_first_kernel()

# ---------------------------------------------------------------------
# Physical parameters (two emitting regions)
# ---------------------------------------------------------------------
B1, R1 = 1.0 * u.G, 1e16 * u.cm
B2, R2 = 10.0 * u.G, 1e13 * u.cm

epsilon_e = 0.1
epsilon_B = 0.01

# Electron distribution parameters
p = 2.5
gamma_c = 1e2
gamma_min = 1.0
gamma_max = 1e10

# ---------------------------------------------------------------------
# Normalize electron distributions (equipartition closure)
# ---------------------------------------------------------------------
N0_pl = compute_PL_norm_from_magnetic_field(
    B1, p, epsilon_B, epsilon_e,
    gamma_min=gamma_min, gamma_max=gamma_max
)

N0_bpl = compute_BPL_norm_from_magnetic_field(
    B2, -p, -(p + 1), gamma_c,
    epsilon_B, epsilon_e,
    gamma_min=gamma_min, gamma_max=gamma_max
)

# Distribution functions
pl  = get_power_law_distribution(
    p, norm=N0_pl,
    gamma_min=gamma_min, gamma_max=gamma_max
)

bpl = get_broken_power_law_distribution(
    -p, -(p + 1), gamma_c,
    norm=N0_bpl,
    gamma_min=gamma_min, gamma_max=gamma_max
)

# ---------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------
gamma = np.geomspace(gamma_min, gamma_max, 500)
nu = np.geomspace(1e-1, 1e5, 500) * u.GHz

# Evaluate distributions
N_pl  = pl(gamma)
N_bpl = bpl(gamma)

# ---------------------------------------------------------------------
# Compute synchrotron intensities
# ---------------------------------------------------------------------
I_pl  = engine.compute_specific_intensity(nu, R1, B1, N_pl, gamma)
I_bpl = engine.compute_specific_intensity(nu, R2, B2, N_bpl, gamma)

I_total = I_pl + I_bpl

# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))

ax.loglog(nu, I_pl,  label='Component 1 (Power-law)', lw=2)
ax.loglog(nu, I_bpl, label='Component 2 (Broken power-law)', lw=2)
ax.loglog(nu, I_total, '--', label='Combined', lw=2)

ax.set_xlabel('Frequency [GHz]')
ax.set_ylabel(r'$I_\nu$ [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$ sr$^{-1}$]')
ax.set_title('Multi-component synchrotron SED')

ax.legend()
ax.grid(True, which='both', ls='--', alpha=0.4)
ax.set_ylim(1e-7, 1e-3)

plt.tight_layout()
plt.show()