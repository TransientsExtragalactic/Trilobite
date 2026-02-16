import numpy as np
import matplotlib.pyplot as plt
from triceratops.dynamics.supernovae.shock_dynamics import NumericalThinShellShockEngine
from triceratops.dynamics.supernovae.profiles import get_broken_power_law_ejecta_kernel_func
from triceratops.utils.plot_utils import set_plot_style
from astropy import units as u

# Define parameters for the CSM and ejecta profiles.
M_dot = 1e-5 * u.M_sun / u.yr  # Mass-loss rate
v_wind = 1e5 * u.cm / u.s  # Wind velocity
t_wind = 1e8 * u.s  # Duration of wind phase
rho_ism = 1e-21 * u.g / u.cm ** 3  # ISM density
E_ej = 1e48 * u.erg  # Ejecta energy
M_ej = 10 * u.M_sun  # Ejecta mass
R_wind = v_wind * t_wind  # Radius of wind termination shock

# Derive CGS parameters for the wind profile.
rho_0_cgs = (M_dot / (4 * np.pi * R_wind ** 2 * v_wind)).to_value(u.g / u.cm ** 3)
R_wind_cgs = R_wind.to_value(u.cm)
rho_ISM_cgs = rho_ism.to_value(u.g / u.cm ** 3)


# Create the rho_csm function so that it can be passed to the
# integrator.
def rho_csm(r):
    """Broken CSM density profile: wind-like close in, uniform ISM far out."""
    return np.where(
        r < R_wind_cgs,
        rho_0_cgs * r ** -2,
        rho_ISM_cgs
    )

# Get the ejecta density kernel from the pre-built Chevalier-style broken power-law
# scenario.
G_v = get_broken_power_law_ejecta_kernel_func(E_ej,M_ej,n=10,delta=0)

# Define the shock engine
shock_engine = NumericalThinShellShockEngine()

# Define parameters
params = {
    "rho_csm": rho_csm,
    "G_ej": G_v,
    "R_0": 1e10 * u.cm,
    "v_0": 1e9 * u.cm / u.s,
    "t_0": 1e1 * u.s,
    "M_0": 1e-4 * u.M_sun,
}

# Time array (days)
time = np.geomspace(1, 1000, 500) * u.day

# Compute the shock properties.
shock_properties = shock_engine.compute_shock_properties(
    time,
    **params)

# Calculate the homologous expansion rate.
R_homologous = (1e9 * u.cm/u.s) * time

# Create the plot of the radius and velocity of the shock as a function
# of time.
set_plot_style()
fig, ax = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
ax[0].loglog(time.to_value(u.day), shock_properties["radius"].to_value(u.cm), label="Numerical Shock Radius")
ax[0].loglog(time.to_value(u.day), R_homologous.to_value(u.cm), ls="--", label="Homologous Expansion")
ax[0].set_ylabel("Shock Radius (cm)")
ax[0].grid(True, which="both", ls="--", alpha=0.5)
ax[1].loglog(time.to_value(u.day), shock_properties["velocity"].to_value(u.cm / u.s))
ax[1].set_xlabel("Time (days)")
ax[1].set_ylabel("Shock Velocity (cm/s)")
ax[1].grid(True, which="both", ls="--", alpha=0.5)
ax[2].loglog(time.to_value(u.day), shock_properties["mass"].to_value(u.M_sun))
ax[2].set_xlabel("Time (days)")
ax[2].set_ylabel(r"Swept-up Mass (M$_\odot$)")
ax[2].grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.show()