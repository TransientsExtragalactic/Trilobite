import numpy as np
import matplotlib.pyplot as plt
from triceratops.dynamics.supernovae.shock_dynamics import ChevalierSelfSimilarWindShockEngine
from triceratops.utils.plot_utils import set_plot_style
from astropy import units as u

# Define the shock engine
shock_engine = ChevalierSelfSimilarWindShockEngine()

# Define parameters
params = {
    "E_ej": 1e51 * u.erg,
    "M_ej": 1.4 * u.M_sun,
    "M_dot": 1e-5 * u.M_sun / u.yr,
    "v_w": 1e6 * u.cm / u.s,
    "n": 10,
    "delta": 0,
}

# Time array (days)
time = np.geomspace(1, 1000, 500) * u.day

# Compute the shock properties.
shock_properties = shock_engine.compute_shock_properties(
    time,
    E_ej=params["E_ej"],
    M_ej=params["M_ej"],
    M_dot=params["M_dot"],
    v_wind=params["v_w"],
    n=params["n"],
    delta=params["delta"],
)

# Extract the shock radius and velocity
shock_radius = shock_properties["radius"]
shock_velocity = shock_properties["velocity"]

# Plotting
set_plot_style()

fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# Plot Shock Radius
axes[0].loglog(time.to_value(u.day), shock_radius.to_value(u.pc), label="Shock Radius", color="blue")
axes[0].set_xlabel("Time since explosion (days)")
axes[0].set_ylabel("Shock Radius (pc)")
axes[0].set_title("Shock Radius and Velocity")
axes[0].legend()

# Plot Shock Velocity
axes[1].loglog(time.to_value(u.day), shock_velocity.to_value(u.km / u.s), label="Shock Velocity", color="red")
axes[1].set_xlabel("Time since explosion (days)")
axes[1].set_ylabel("Shock Velocity (km/s)")
axes[1].legend()
plt.tight_layout()
plt.show()