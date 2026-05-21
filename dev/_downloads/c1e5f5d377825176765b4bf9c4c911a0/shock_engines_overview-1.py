import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from triceratops.dynamics.shocks import ChevalierSelfSimilarWindShockEngine

# Instantiate the engine.  No physical parameters are stored at construction time.
engine = ChevalierSelfSimilarWindShockEngine()

# ── Physical parameters ────────────────────────────────────────────────────
E_ej   = 1e51 * u.erg                  # typical core-collapse SN explosion energy
M_ej   = 10.0 * u.Msun                 # ejecta mass
M_dot  = 1e-5 * u.Msun / u.yr          # progenitor wind mass-loss rate
v_wind = 1000.0 * u.km / u.s           # progenitor wind velocity
n      = 10.0                           # outer ejecta density power-law index

# ── Time grid: 10 days to ~10 years ───────────────────────────────────────
t = np.logspace(1, np.log10(3650), 300) * u.day

# ── Evaluate the shock engine ──────────────────────────────────────────────
# The engine is callable; engine(t, ...) is equivalent to
# engine.compute_shock_properties(t, ...).
props = engine(t, E_ej=E_ej, M_ej=M_ej, M_dot=M_dot, v_wind=v_wind, n=n)

# ── Plot radius and velocity ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].loglog(
    t.to(u.day).value,
    props.radius.to(u.cm).value / 1e16,
    color='steelblue', lw=2,
)
axes[0].set_xlabel('Time [days]')
axes[0].set_ylabel(r'Shock radius [$10^{16}$ cm]')
axes[0].set_title('Shock Radius')
axes[0].grid(True, which='both', ls='--', alpha=0.4)

axes[1].loglog(
    t.to(u.day).value,
    props.velocity.to(u.km / u.s).value,
    color='darkorange', lw=2,
)
axes[1].set_xlabel('Time [days]')
axes[1].set_ylabel(r'Shock velocity [km s$^{-1}$]')
axes[1].set_title('Shock Velocity')
axes[1].grid(True, which='both', ls='--', alpha=0.4)

plt.suptitle(
    r'Chevalier wind-CSM shock: '
    r'$E_{\rm ej} = 10^{51}\ {\rm erg},\; M_{\rm ej} = 10\,M_\odot$',
    fontsize=11, y=1.02,
)
plt.tight_layout()