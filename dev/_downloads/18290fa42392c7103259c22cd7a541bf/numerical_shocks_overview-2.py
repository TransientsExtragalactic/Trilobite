import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u

from triceratops.dynamics.shocks import (
    MechanicalShockEngine,
    get_bpl_ejecta_kernel,
    get_wind_csm_density_func,
    make_homologous_stationary_sources,
)
from triceratops.utils.plot_utils import set_plot_style

G_ej    = get_bpl_ejecta_kernel(1e51 * u.erg, 5.0 * u.Msun, n=10.0, delta=1.0)
rho_csm = get_wind_csm_density_func(1e-5 * u.Msun / u.yr, 100.0 * u.km / u.s)
rho_1, u_1, rho_4, u_4 = make_homologous_stationary_sources(G_ej, rho_csm)

engine = MechanicalShockEngine()
t_0    = 1.0 * u.day
R0, v0, M2_0, M3_0, U2_0, U3_0, Dlt2_0, Dlt3_0 = (
    MechanicalShockEngine.generate_initial_conditions(
        R_cd_0=1e14 * u.cm, v_cd_0=1e9 * u.cm / u.s,
        t_0=t_0, rho_1=rho_1, rho_4=rho_4, u_1=u_1, u_4=u_4,
    )
)

time  = np.geomspace(1, 1000, 300) * u.day
state = engine.compute_shock_properties(
    time=time,
    rho_1=rho_1, rho_4=rho_4, u_1=u_1, u_4=u_4,
    R_cd_0=R0 * u.cm, v_cd_0=v0 * u.cm / u.s,
    M2_0=M2_0 * u.g, M3_0=M3_0 * u.g,
    U2_0=U2_0 * u.erg, U3_0=U3_0 * u.erg,
    Delta2_0=Dlt2_0 * u.cm, Delta3_0=Dlt3_0 * u.cm,
    t_0=t_0,
)

set_plot_style()
fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
t_days = time.to_value(u.day)

axes[0].loglog(t_days, state.post_shock_temperature_fs.to_value(u.K), label="Forward shock")
axes[0].loglog(t_days, state.post_shock_temperature_rs.to_value(u.K), label="Reverse shock")
axes[0].set_ylabel("Post-shock temperature (K)")
axes[0].legend()

axes[1].loglog(t_days, state.post_shock_pressure_fs.to_value(u.dyn / u.cm**2), label="Forward shock")
axes[1].loglog(t_days, state.post_shock_pressure_rs.to_value(u.dyn / u.cm**2), label="Reverse shock")
axes[1].set_ylabel(r"Post-shock pressure (dyn cm$^{-2}$)")
axes[1].set_xlabel("Time (days)")
axes[1].legend()

plt.tight_layout()
plt.show()