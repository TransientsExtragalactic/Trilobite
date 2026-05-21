import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from trilobite.dynamics.shocks import ChevalierSelfSimilarShockEngine
from trilobite.utils.plot_utils import set_plot_style

engine = ChevalierSelfSimilarShockEngine()
time   = np.geomspace(1, 1000, 500) * u.day

K_csm = ChevalierSelfSimilarShockEngine.normalize_csm_density(
    rho_0=1e-20 * u.g / u.cm**3,
    r_0=1e16 * u.cm,
    s=2.0,
)

state = engine.compute_shock_properties(
    time=time,
    E_ej=1e51 * u.erg,
    M_ej=5.0 * u.Msun,
    K_csm=K_csm,
    n=10.0,
    s=2.0,
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