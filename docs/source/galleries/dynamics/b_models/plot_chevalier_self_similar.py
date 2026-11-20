r"""
Chevalier Self-Similar Shock Evolution
======================================

This example computes the self-similar shock evolution for homologously
expanding supernova ejecta interacting with a power-law circumstellar medium
(CSM). The calculation uses the Chevalier self-similar solution, which applies
when the outer ejecta density and the CSM density are both described by power
laws.

We compute the shock radius and post-shock temperature as a function of time for
a simple wind-like CSM profile.
"""

# %%
# Imports
# -------
# We start by importing the numerical and plotting dependencies, along with the
# Chevalier shock engine.

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

from trilobite.dynamics.shocks import ChevalierSelfSimilarShockEngine


# %%
# Supernova and CSM Parameters
# ---------------------------
# The Chevalier solution assumes an outer ejecta density profile of the form
#
# .. math::
#
#     \rho_{\rm ej} \propto r^{-n},
#
# and a CSM density profile of the form
#
# .. math::
#
#     \rho_{\rm CSM}(r) = K_{\rm CSM} r^{-s}.
#
# Here we choose a wind-like CSM with :math:`s = 2`. In this example the
# exponent is stored as ``alpha_csm = -2`` because the normalization is written
# using
#
# .. math::
#
#     \rho_{\rm CSM}(r) = K_{\rm CSM} r^{\alpha_{\rm CSM}}.
#
# Therefore, ``alpha_csm = -s``.

n = 7.0
gamma = 5.0 / 3.0
mu = 0.6

E_ej = 1.0e51 * u.erg
M_ej = 1.0 * u.Msun

R_0 = 1.0e15 * u.cm
rho_0 = 1.0e-20 * u.g / u.cm**3
alpha_csm = -2.0

K_csm = rho_0 * R_0**alpha_csm


# %%
# Build the Shock Engine
# ----------------------
# The shock engine stores the equation-of-state information needed to evaluate
# the self-similar solution and derived post-shock quantities.

engine = ChevalierSelfSimilarShockEngine(
    gamma=gamma,
    mu=mu,
)


# %%
# Compute the Evolution
# ---------------------
# We evaluate the shock properties over a logarithmically spaced time grid.

time = np.geomspace(1.0e1, 1.0e4, 100) * u.day

shock = engine.compute_shock_properties(
    time,
    E_ej=E_ej,
    M_ej=M_ej,
    K_csm=K_csm,
    n=n,
    s=alpha_csm,
)


# %%
# Plot the Shock Radius
# ---------------------
# The shock radius follows a power-law in time, as expected for a self-similar
# solution.

fig, ax = plt.subplots(figsize=(6, 4))

ax.loglog(time.to_value(u.day), shock.radius.to_value(u.cm))
ax.set_xlabel("Time [day]")
ax.set_ylabel("Shock radius [cm]")
ax.set_title("Chevalier self-similar shock radius")

plt.show()


# %%
# Plot the Post-Shock Temperature
# -------------------------------
# The post-shock temperature decreases as the shock decelerates.

fig, ax = plt.subplots(figsize=(6, 4))

ax.loglog(
    time.to_value(u.day),
    shock.post_shock_temperature.to_value(u.K),
)

ax.set_xlabel("Time [day]")
ax.set_ylabel("Post-shock temperature [K]")
ax.set_title("Post-shock temperature evolution")

plt.show()
