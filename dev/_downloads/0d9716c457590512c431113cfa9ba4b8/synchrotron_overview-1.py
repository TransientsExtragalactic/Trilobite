import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from trilobite.radiation.synchrotron import PowerLaw_SSA_SynchrotronSED

# ── Instantiate the SED model ─────────────────────────────────────────────
# SED objects are stateless: physical parameters are passed at call time,
# not stored on the object.  This makes them safe to reuse in parameter
# sweeps and inference loops without hidden state.
sed = PowerLaw_SSA_SynchrotronSED()

# ── Physical source parameters ────────────────────────────────────────────
# These are the quantities a user would typically know from the source model
# or shock dynamics:
B         = 0.5 * u.G           # post-shock magnetic field strength
R         = 1e16 * u.cm         # characteristic emitting-region radius
gamma_min = 100.0               # minimum electron Lorentz factor
p         = 3.0                 # power-law index of the electron distribution
eps_E     = 0.1                 # fraction of shock energy in electrons
eps_B     = 0.1                 # fraction of shock energy in magnetic field
D_L       = 100 * u.Mpc         # luminosity distance to the source

# ── Convert physical parameters to phenomenological SED parameters ────────
# from_physics_to_params applies the equipartition closure: given B and R
# it computes the injection frequency nu_m, the SSA frequency nu_a, and the
# normalization F_norm, together with a solid-angle factor omega.
params = sed.from_physics_to_params(
    B=B, R=R,
    gamma_min=gamma_min,
    p=p,
    epsilon_E=eps_E,
    epsilon_B=eps_B,
    luminosity_distance=D_L,
    pitch_average=True,   # average over electron pitch angles
)

# ── Frequency grid: radio through soft X-ray ─────────────────────────────
nu = np.logspace(8, 15, 500) * u.Hz

# ── Evaluate the SED ──────────────────────────────────────────────────────
# The sed() method assembles the piecewise power-law spectrum.  Note that
# nu_a (the SSA turnover) is determined internally from nu_m, omega, p, and
# gamma_m -- it does not need to be supplied explicitly.
Fnu = sed.sed(
    nu,
    nu_m=params['nu_m'],       # injection break frequency
    F_norm=params['F_norm'],   # flux normalization at nu_m
    nu_max=params['nu_max'],   # high-frequency cutoff
    omega=params['omega'],     # effective solid angle (encodes R and D_L)
    gamma_m=gamma_min,         # minimum Lorentz factor (sets nu_a position)
    p=p,
)

# ── Plot ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))

ax.loglog(
    nu.to(u.GHz).value,
    Fnu.to(u.mJy).value,
    color='steelblue', lw=2,
)

# Mark the characteristic frequencies returned by from_physics_to_params
ax.axvline(
    params['nu_a'].to(u.GHz).value,
    ls='--', color='firebrick', lw=1.2, label=r'$\nu_a$ (SSA turnover)',
)
ax.axvline(
    params['nu_m'].to(u.GHz).value,
    ls='--', color='darkorange', lw=1.2, label=r'$\nu_m$ (injection break)',
)

ax.set_xlabel('Frequency [GHz]', fontsize=12)
ax.set_ylabel(r'$F_\nu$ [mJy]', fontsize=12)
ax.set_title(
    r'One-zone SSA synchrotron SED (no cooling), $D_L = 100$ Mpc',
    fontsize=11,
)
ax.legend(fontsize=10)
ax.grid(True, which='both', ls='--', alpha=0.4)
plt.tight_layout()