r"""
Two-Shock Chevalier Self-Similar Evolution in a Steady Wind
===========================================================

This example computes the self-similar interaction between homologously
expanding supernova ejecta and a steady circumstellar wind.

The calculation uses the Chevalier self-similar solution specialized to a
wind-like circumstellar medium (CSM),

.. math::

    \rho_{\rm CSM}(r)
    =
    \frac{\dot M}{4\pi v_{\rm wind}} r^{-2},

where :math:`\dot M` is the progenitor mass-loss rate and :math:`v_{\rm wind}`
is the wind velocity.

Unlike a single-shock description, this engine returns the full two-shock
structure: the reverse shock, contact discontinuity, and forward shock. We plot
the evolution of these three radii, along with the post-shock temperatures
behind the forward and reverse shocks.
"""

# %%
# Imports
# -------
# We start by importing the numerical and plotting dependencies, along with the
# wind-specialized two-shock Chevalier engine.

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

from trilobite.dynamics.shocks import ChevalierTwoShockSelfSimilarWindEngine


# %%
# Supernova Ejecta Parameters
# ---------------------------
# The outer supernova ejecta are modeled as a homologously expanding power-law
# density profile. The index ``n`` controls how steeply the outer ejecta density
# falls with velocity or radius.

n = 7.0

E_ej = 1.0e51 * u.erg
M_ej = 1.0 * u.Msun


# %%
# Circumstellar Wind Parameters
# -----------------------------
# The CSM is assumed to be a steady wind. Its density normalization is set by
# the progenitor mass-loss rate and wind velocity.

M_dot = 1.0e-5 * u.Msun / u.yr
v_wind = 100.0 * u.km / u.s


# %%
# Gas Parameters
# --------------
# The adiabatic index and mean molecular weight are used when computing derived
# post-shock thermodynamic quantities such as the forward- and reverse-shock
# temperatures.

gamma = 5.0 / 3.0
mu = 0.6


# %%
# Build the Shock Engine
# ----------------------
# The engine stores the equation-of-state information needed to evaluate the
# self-similar shock structure and derived post-shock quantities.

engine = ChevalierTwoShockSelfSimilarWindEngine(
    gamma=gamma,
    mu=mu,
)


# %%
# Compute the Shock Evolution
# ---------------------------
# We evaluate the solution over a logarithmically spaced time grid.

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
# Plot the Shock Radii
# --------------------
# The solution contains three characteristic radii:
#
# - the reverse-shock radius,
# - the contact-discontinuity radius,
# - the forward-shock radius.
#
# These radii maintain fixed ratios in the self-similar phase.

fig, ax = plt.subplots(figsize=(6, 4))

ax.loglog(
    time.to_value(u.day),
    shock.radius_rs.to_value(u.cm),
    label="Reverse shock",
)
ax.loglog(
    time.to_value(u.day),
    shock.radius_cd.to_value(u.cm),
    label="Contact discontinuity",
)
ax.loglog(
    time.to_value(u.day),
    shock.radius_fs.to_value(u.cm),
    label="Forward shock",
)

ax.set_xlabel("Time [day]")
ax.set_ylabel("Radius [cm]")
ax.set_title("Two-shock Chevalier wind structure")
ax.legend()

plt.show()


# %%
# Plot the Post-Shock Temperatures
# --------------------------------
# The forward-shock temperature is set by the shock speed relative to the
# upstream wind. The reverse-shock temperature is set by the shock speed relative
# to the freely expanding ejecta.

fig, ax = plt.subplots(figsize=(6, 4))

ax.loglog(
    time.to_value(u.day),
    shock.temperature_rs.to_value(u.K),
    label="Reverse shock",
)
ax.loglog(
    time.to_value(u.day),
    shock.temperature_fs.to_value(u.K),
    label="Forward shock",
)

ax.set_xlabel("Time [day]")
ax.set_ylabel("Post-shock temperature [K]")
ax.set_title("Post-shock temperature evolution")
ax.legend()

plt.show()
