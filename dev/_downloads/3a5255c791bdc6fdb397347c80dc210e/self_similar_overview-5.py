import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from triceratops.dynamics.shocks import (
    BlandfordMcKeeShockEngine,
    BlandfordMcKeeWindShockEngine,
)
from triceratops.utils.plot_utils import set_plot_style

set_plot_style()
fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

time = np.geomspace(0.1, 300, 400) * u.day
E    = 1e52 * u.erg

# --- ISM (k = 0) ---
ism = BlandfordMcKeeShockEngine()
K_ism = BlandfordMcKeeShockEngine.normalize_csm_density(
    rho_0=1.67e-24 * u.g / u.cm**3,
    r_0=1.0 * u.cm,
    k=0.0,
)
s_ism = ism.compute_shock_properties(time=time, E=E, K_csm=K_ism, k=0.0)

# --- Wind (k = 2) ---
wind = BlandfordMcKeeWindShockEngine()
s_wind = wind.compute_shock_properties(
    time=time, E=E,
    M_dot=1e-5 * u.Msun / u.yr,
    v_wind=1000.0 * u.km / u.s,
)

t_day = time.to_value(u.day)

axes[0].loglog(t_day, s_ism.lorentz_factor,  label=r"ISM ($k=0$)")
axes[0].loglog(t_day, s_wind.lorentz_factor, label=r"Wind ($k=2$)", ls="--")
axes[0].axhline(2, color="gray", lw=0.8, ls=":", label=r"$\Gamma = 2$")
axes[0].set_ylabel(r"Shock Lorentz factor $\Gamma$")
axes[0].legend()

axes[1].loglog(t_day, s_ism.radius.to_value(u.cm),  label=r"ISM ($k=0$)")
axes[1].loglog(t_day, s_wind.radius.to_value(u.cm), label=r"Wind ($k=2$)", ls="--")
axes[1].set_ylabel("Radius (cm)")
axes[1].set_xlabel("Time (days)")

plt.tight_layout()
plt.show()