r"""
Chevalier Self-Similar Shock in a Steady Wind
=============================================

This example computes the self-similar shock evolution for homologously
expanding supernova ejecta interacting with a steady circumstellar wind.

The calculation uses the Chevalier self-similar solution specialized to a
wind-like circumstellar medium (CSM),

.. math::

    \rho_{\rm CSM}(r)
    =
    \frac{\dot M}{4\pi v_{\rm wind}} r^{-2},

where :math:`\dot M` is the progenitor mass-loss rate and :math:`v_{\rm wind}`
is the wind velocity.

We compute the shock radius and post-shock temperature as a function of time.
"""

# %%
# Imports
# -------
# We start by importing the numerical and plotting dependencies, along with the
# wind-specialized Chevalier shock engine.

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

from trilobite.dynamics.shocks import ChevalierSelfSimilarWindShockEngine


# %%
# Supernova Parameters
# --------------------
# The outer supernova ejecta are described by a power-law density profile. The
# index ``n`` controls how steeply the outer ejecta density falls with radius.

n = 7.0

E_ej = 1.0e51 * u.erg
M_ej = 1.0 * u.Msun


# %%
# Wind Parameters
# ---------------
# The CSM is assumed to be a steady wind. The wind density normalization is
# determined by the mass-loss rate and wind velocity.

M_dot = 1.0e-5 * u.Msun / u.yr
v_wind = 100.0 * u.km / u.s


# %%
# Gas Parameters
# --------------
# The adiabatic index and mean molecular weight are used when computing derived
# post-shock thermodynamic quantities such as the temperature.

gamma = 5.0 / 3.0
mu = 0.6


# %%
# Build the Shock Engine
# ----------------------
# The engine stores the equation-of-state information needed to evaluate the
# self-similar solution and derived post-shock quantities.

engine = ChevalierSelfSimilarWindShockEngine(
    gamma=gamma,
    mu=mu,
)


# %%
# Compute the Shock Evolution
# ---------------------------
# We evaluate the shock properties over a logarithmically spaced time grid.

time = np.geomspace(1.0e1, 1.0e4, 100) * u.day

shock = engine.compute_shock_properties(
    time,
    E_ej=E_ej,
    M_ej=M_ej,
    n=n,
    M_dot=M_dot,
    v_wind=v_wind,
)


# %%
# Plot the Shock Radius
# ---------------------
# The shock expands as a power law in time, as expected for a self-similar
# ejecta-wind interaction.

fig, ax = plt.subplots(figsize=(6, 4))

ax.loglog(
    time.to_value(u.day),
    shock.radius.to_value(u.cm),
)

ax.set_xlabel("Time [day]")
ax.set_ylabel("Shock radius [cm]")
ax.set_title("Chevalier wind shock radius")

plt.show()


# %%
# Plot the Post-Shock Temperature
# -------------------------------
# The post-shock temperature is set by the shock velocity and therefore evolves
# as the self-similar shock decelerates.

fig, ax = plt.subplots(figsize=(6, 4))

ax.loglog(
    time.to_value(u.day),
    shock.post_shock_temperature.to_value(u.K),
)

ax.set_xlabel("Time [day]")
ax.set_ylabel("Post-shock temperature [K]")
ax.set_title("Chevalier wind post-shock temperature")

plt.show()
