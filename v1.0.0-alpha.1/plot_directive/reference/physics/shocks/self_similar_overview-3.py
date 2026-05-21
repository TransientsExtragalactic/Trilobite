import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from trilobite.dynamics.shocks import ChevalierTwoShockSelfSimilarEngine
from trilobite.utils.plot_utils import set_plot_style

engine = ChevalierTwoShockSelfSimilarEngine()
time   = np.geomspace(1, 1000, 300) * u.day

K_csm = (5e-16 * u.g / u.cm**3) * (1e15 * u.cm)**2

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
fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
t_day = time.to_value(u.day)

axes[0].loglog(t_day, state.radius_fs.to_value(u.cm), label=r"$R_{\rm fs}$")
axes[0].loglog(t_day, state.radius_cd.to_value(u.cm), ls="--", label=r"$R_{\rm cd}$")
axes[0].loglog(t_day, state.radius_rs.to_value(u.cm), label=r"$R_{\rm rs}$")
axes[0].set_ylabel("Radius (cm)")
axes[0].legend()

axes[1].loglog(t_day, state.velocity_fs.to_value(u.km / u.s), label=r"$v_{\rm fs}$")
axes[1].loglog(t_day, state.velocity_cd.to_value(u.km / u.s), ls="--", label=r"$v_{\rm cd}$")
axes[1].loglog(t_day, state.velocity_rs.to_value(u.km / u.s), label=r"$v_{\rm rs}$")
axes[1].set_ylabel(r"Velocity (km s$^{-1}$)")
axes[1].legend()

axes[2].loglog(t_day, state.temperature_fs.to_value(u.K), label=r"$T_{\rm fs}$")
axes[2].loglog(t_day, state.temperature_rs.to_value(u.K), label=r"$T_{\rm rs}$")
axes[2].set_ylabel("Post-shock temperature (K)")
axes[2].set_xlabel("Time (days)")
axes[2].legend()

plt.tight_layout()
plt.show()