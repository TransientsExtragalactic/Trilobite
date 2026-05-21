import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from trilobite.dynamics.shocks import SedovTaylorShockEngine
from trilobite.utils.plot_utils import set_plot_style

engine = SedovTaylorShockEngine()
time   = np.geomspace(100, 1e6, 500) * u.yr

state = engine.compute_shock_properties(
    time=time,
    E=1e51 * u.erg,
    rho_0=1.67e-24 * u.g / u.cm**3,   # approximately 1 H atom per cm^3
)

set_plot_style()
fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)

t_yr = time.to_value(u.yr)

axes[0].loglog(t_yr, state.radius.to_value(u.pc))
axes[0].set_ylabel("Radius (pc)")

axes[1].loglog(t_yr, state.velocity.to_value(u.km / u.s))
axes[1].set_ylabel(r"Velocity (km s$^{-1}$)")

axes[2].loglog(t_yr, state.post_shock_temperature.to_value(u.K))
axes[2].set_ylabel("Post-shock temperature (K)")
axes[2].set_xlabel("Time (yr)")

plt.tight_layout()
plt.show()