import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from trilobite.dynamics.shocks import ChevalierSelfSimilarWindShockEngine
from trilobite.utils.plot_utils import set_plot_style

engine = ChevalierSelfSimilarWindShockEngine()
time   = np.geomspace(1, 1000, 500) * u.day

state = engine.compute_shock_properties(
    time=time,
    E_ej=1e51 * u.erg,
    M_ej=5.0 * u.Msun,
    M_dot=1e-5 * u.Msun / u.yr,
    v_wind=100.0 * u.km / u.s,
    n=10.0,
    delta=1.0,
)

set_plot_style()
fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)

axes[0].loglog(time.to_value(u.day), state.radius.to_value(u.cm))
axes[0].set_ylabel("Radius (cm)")

axes[1].loglog(time.to_value(u.day), state.velocity.to_value(u.km / u.s))
axes[1].set_ylabel(r"Velocity (km s$^{-1}$)")
axes[1].set_xlabel("Time (days)")

plt.tight_layout()
plt.show()