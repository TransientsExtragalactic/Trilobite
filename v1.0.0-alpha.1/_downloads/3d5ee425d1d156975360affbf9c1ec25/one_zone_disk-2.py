import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as const
from astropy import units as u

from trilobite.dynamics.accretion.one_zone import GasPressureDisk
from trilobite.utils.plot_utils import set_plot_style

# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------
set_plot_style()

# ---------------------------------------------------------------------------
# Disk and black hole parameters
# ---------------------------------------------------------------------------
M_BH = 1e6 * const.M_sun   # Black hole mass
M_D_0 = 30 * const.M_sun   # Initial disk mass
R_D_0 = 3.0e13 * u.cm      # Initial disk radius

ALPHA = 0.1                 # Shakura-Sunyaev viscosity parameter
R_IN = 3.0e6 * u.cm         # Inner truncation radius (≈ 3 R_S for 10 M_sun BH)
T_END = 1e3 * u.yr          # Total integration time
T_SNAP = 1e2 * u.yr         # Snapshot time for the second SED

# ---------------------------------------------------------------------------
# Build and evolve the disk
# ---------------------------------------------------------------------------
disk = GasPressureDisk(mu=0.62)  # mu: mean molecular weight (fully ionised H/He mix)

initial_conditions = disk.generate_initial_conditions(
    M_BH=M_BH,
    M_D_0=M_D_0,
    R_D_0=R_D_0,
)

result = disk.solve(
    initial_conditions=initial_conditions,
    runtime_parameters={
        "M_BH": M_BH,
        "R_in": R_IN,
        "alpha": ALPHA,
    },
    t_span=(0, T_END),
    max_steps=1_000_000,
)

# ---------------------------------------------------------------------------
# Compute spectral energy distributions
# ---------------------------------------------------------------------------
nu = np.geomspace(1e9, 1e18, 400) * u.Hz   # Frequency grid: radio → hard X-ray

L_nu_initial = result.compute_spectral_luminosity(nu, 0 * u.yr).to_value("erg / (s Hz)")
L_nu_snap    = result.compute_spectral_luminosity(nu, T_SNAP).to_value("erg / (s Hz)")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots()

ax.loglog(nu, L_nu_initial, label=r"$t = 0$ yr")
ax.loglog(nu, L_nu_snap,    label=rf"$t = {T_SNAP.value:.0f}$ yr")

ax.set_xlim(nu[[0, -1]].value)
ax.set_ylim(1e10, 1e30)
ax.set_xlabel(r"Frequency $\nu$ (Hz)")
ax.set_ylabel(r"Spectral luminosity $L_\nu$ (erg s$^{-1}$ Hz$^{-1}$)")
ax.set_title("Gas Pressure Disk — SED Evolution")
ax.legend()

fig.tight_layout()
plt.show()