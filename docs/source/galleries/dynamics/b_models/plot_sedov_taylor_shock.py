r"""
Chevalier and Sedov-Taylor Shock Evolution
==========================================

This example compares two classic self-similar shock solutions for supernova
remnant evolution in a uniform circumstellar medium (CSM):

- the Chevalier solution, which describes the ejecta-dominated interaction
  between freely expanding power-law ejecta and an external medium;
- the Sedov-Taylor solution, which describes the later energy-conserving blast
  wave once the swept-up ambient mass dominates over the ejecta structure.

Both calculations are evaluated for the same explosion energy and ambient
density. The comparison is useful for illustrating the different temporal
scalings of the ejecta-dominated and Sedov-Taylor limits.
"""

# %%
# Imports
# -------
# We start by importing the numerical and plotting dependencies, along with the
# two shock engines.

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

from trilobite.dynamics.shocks import (
    ChevalierSelfSimilarShockEngine,
    SedovTaylorShockEngine,
)


# %%
# Physical Parameters
# -------------------
# We compare the two solutions in a uniform CSM. For the Chevalier solution,
# the CSM profile is written as
#
# .. math::
#
#     \rho_{\rm CSM}(r) = K_{\rm CSM} r^s.
#
# A uniform medium therefore corresponds to ``s = 0`` and
# ``K_csm = rho_0``.
#
# The ejecta-dominated Chevalier solution additionally requires an ejecta mass
# and an outer ejecta power-law index ``n``.

gamma = 5.0 / 3.0
mu = 0.6

E_ej = 1.0e51 * u.erg
M_ej = 1.0 * u.Msun

rho_0 = 1.0e-20 * u.g / u.cm**3

n = 7.0
s = 0.0
K_csm = rho_0


# %%
# Build the Shock Engines
# -----------------------
# The Chevalier engine represents the ejecta-dominated phase, while the
# Sedov-Taylor engine represents the energy-conserving phase.

chevalier_engine = ChevalierSelfSimilarShockEngine(
    gamma=gamma,
    mu=mu,
)

sedov_taylor_engine = SedovTaylorShockEngine(
    gamma=gamma,
    mu=mu,
)


# %%
# Compute the Evolution
# ---------------------
# Both solutions are evaluated on the same time grid. This is a formal
# comparison of the two limiting self-similar solutions rather than a stitched
# evolutionary model.

time = np.geomspace(1.0e1, 1.0e4, 100) * u.day

chevalier_shock = chevalier_engine.compute_shock_properties(
    time,
    E_ej=E_ej,
    M_ej=M_ej,
    K_csm=K_csm,
    n=n,
    s=s,
)

sedov_taylor_shock = sedov_taylor_engine.compute_shock_properties(
    time,
    E=E_ej,
    rho_0=rho_0,
)


# %%
# Compare Shock Radius
# --------------------
# In a uniform medium, the Sedov-Taylor blast wave follows
# :math:`R \propto t^{2/5}`. The Chevalier ejecta-dominated solution generally
# expands faster because the outer ejecta are still driving the shock.

fig, ax = plt.subplots(figsize=(6, 4))

ax.loglog(
    time.to_value(u.day),
    chevalier_shock.radius.to_value(u.cm),
    label="Chevalier",
)

ax.loglog(
    time.to_value(u.day),
    sedov_taylor_shock.radius.to_value(u.cm),
    label="Sedov-Taylor",
)

ax.set_xlabel("Time [day]")
ax.set_ylabel("Shock radius [cm]")
ax.set_title("Shock radius: Chevalier vs. Sedov-Taylor")
ax.legend()

plt.show()


# %%
# Compare Post-Shock Temperature
# ------------------------------
# The post-shock temperature tracks the square of the shock velocity. Since the
# two solutions decelerate differently, their temperature evolutions also differ.

fig, ax = plt.subplots(figsize=(6, 4))

ax.loglog(
    time.to_value(u.day),
    chevalier_shock.post_shock_temperature.to_value(u.K),
    label="Chevalier",
)

ax.loglog(
    time.to_value(u.day),
    sedov_taylor_shock.post_shock_temperature.to_value(u.K),
    label="Sedov-Taylor",
)

ax.set_xlabel("Time [day]")
ax.set_ylabel("Post-shock temperature [K]")
ax.set_title("Post-shock temperature: Chevalier vs. Sedov-Taylor")
ax.legend()

plt.show()
