import matplotlib.pyplot as plt
from astropy import constants as const
from astropy import units as u
from triceratops.dynamics.accretion.one_zone import GasPressureDisk
from triceratops.utils.plot_utils import set_plot_style

# 1. Setup Environment
set_plot_style()

# 2. Configure Disk Parameters
M_BH = 1e6 * const.M_sun
M_D_0 = 3 * const.M_sun
R_D_0 = 3.0e15 * u.cm

disk = GasPressureDisk(mu=0.62)
ic = disk.generate_initial_conditions(M_BH=M_BH, M_D_0=M_D_0, R_D_0=R_D_0)

# 3. Solve the System
result = disk.solve(
initial_conditions=ic,
runtime_parameters={
    "M_BH": M_BH,
    "R_in": 3.0e6 * u.cm,
    "alpha": 0.1
},
t_span=(0, 1e7 * u.yr),
max_steps=100_000,
)

# 4. Extract Data
data = result.data
t_yr = data["t"].to(u.yr).value
m_disk = data["M_D"].to(u.M_sun).value

# 5. Visualization
fig, ax = plt.subplots(figsize=(8, 5))

ax.semilogx(t_yr, m_disk, label=r"$M_D$ (Disk Mass)", lw=2)

ax.set_xlabel("Time [yr]")
ax.set_ylabel(r"Disk Mass [$M_\odot$]")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()