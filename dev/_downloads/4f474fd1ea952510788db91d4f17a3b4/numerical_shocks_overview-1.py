import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from triceratops.dynamics.shocks import (
    PressureDrivenThinShellShockEngine,
    get_bpl_ejecta_kernel,
    get_wind_csm_density_func,
    make_homologous_stationary_sources,
)
from triceratops.utils.plot_utils import set_plot_style

G_ej    = get_bpl_ejecta_kernel(1e51 * u.erg, 5.0 * u.Msun, n=10.0, delta=1.0)
rho_csm = get_wind_csm_density_func(1e-5 * u.Msun / u.yr, 100.0 * u.km / u.s)
rho_1, u_1, rho_4, u_4 = make_homologous_stationary_sources(G_ej, rho_csm)

engine = PressureDrivenThinShellShockEngine()
time   = np.geomspace(1, 1000, 500) * u.day

state = engine.compute_shock_properties(
    time=time,
    rho_1=rho_1, rho_4=rho_4,
    u_1=u_1,     u_4=u_4,
    R_0=1e14 * u.cm,
    v_0=1e9  * u.cm / u.s,
    M_0=1e26 * u.g,
    t_0=1.0  * u.day,
)

set_plot_style()
fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
t_days = time.to_value(u.day)

axes[0].loglog(t_days, state.radius.to_value(u.cm))
axes[0].set_ylabel("Radius (cm)")

axes[1].loglog(t_days, state.velocity.to_value(u.km / u.s))
axes[1].set_ylabel(r"Velocity (km s$^{-1}$)")

axes[2].loglog(t_days, state.mass.to_value(u.Msun))
axes[2].set_xlabel("Time (days)")
axes[2].set_ylabel(r"Shell mass ($M_\odot$)")

plt.tight_layout()
plt.show()