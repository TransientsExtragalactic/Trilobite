import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as const
from astropy import units as u
from triceratops.dynamics.accretion import AlphaDisk

# ── Physical parameters ───────────────────────────────────────────────────
M_BH  = 10 * const.M_sun           # black hole mass
mdot  = 1e16 * u.g / u.s          # mass accretion rate (Eddington-ish for 10 M_sun)
alpha = 0.1                        # Shakura-Sunyaev viscosity parameter

# Inner truncation radius: 6 R_g (Schwarzschild ISCO for a non-spinning BH)
R_g   = (const.G * M_BH / const.c**2).to(u.cm)
R_in  = 6.0 * R_g

# Outer radius: where the disk effectively truncates
R_out = 1e3 * R_in

# Luminosity distance — needed to convert L_nu to F_nu (observed flux density)
D_L   = 10 * u.kpc

# ── Instantiate the disk model ────────────────────────────────────────────
# AlphaDisk implements the SS73 / FKR analytical scalings.
# The alpha parameter is the only model-level constant; physical parameters
# (M_BH, mdot, R_in) are passed at evaluation time.
disk = AlphaDisk(alpha=alpha)

# ── Frequency grid — far-UV through soft X-ray ───────────────────────────
# 500 log-spaced points from 1e13 Hz (~12000 Angstrom, NIR) to 1e18 Hz (~4 keV)
nu = np.geomspace(1e14, 1e18, 500) * u.Hz

# ── Compute the multi-colour blackbody SED ────────────────────────────────
# compute_sed integrates the Planck function B_nu(T_eff(r)) over all annuli.
# Setting D_L causes the returned dict to include both L_nu and F_nu.
sed = disk.compute_sed(
    nu,
    M_BH,
    mdot,
    R_in,
    R_out=R_out,
    D_L=D_L,
    cos_theta=1.0,   # face-on inclination (maximises observed flux)
    N_r=500,         # number of radial quadrature points
)

# sed["L_nu"] -- spectral luminosity [erg/s/Hz], always returned
# sed["F_nu"] -- flux density [erg/s/Hz/cm^2], returned when D_L is set
nu_F_nu = (nu * sed["F_nu"]).to(u.erg / u.s / u.cm**2)

# ── Plot: nu * F_nu vs nu (standard SED presentation) ────────────────────
fig, ax = plt.subplots(figsize=(7, 4))

ax.loglog(nu.value, nu_F_nu.value, color="steelblue", lw=2)

ax.set_xlabel(r"Frequency  $\nu$  [Hz]", fontsize=12)
ax.set_ylabel(r"$\nu F_\nu$  [erg s$^{-1}$ cm$^{-2}$]", fontsize=12)
ax.set_title(
    r"Multi-colour blackbody SED — "
    r"$10\,M_\odot$ BH, $\dot{M}=10^{16}$ g s$^{-1}$, $D_L=10$ kpc",
    fontsize=11,
)
ax.grid(True, which="both", ls="--", alpha=0.4)
plt.tight_layout()