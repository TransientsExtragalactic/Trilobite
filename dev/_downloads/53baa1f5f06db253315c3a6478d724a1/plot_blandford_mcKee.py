r"""
Blandford-McKee Ultra-Relativistic Blast Wave
=============================================

This example computes the Blandford-McKee self-similar solution for an
ultra-relativistic blast wave expanding into a uniform external medium.

The Blandford-McKee solution describes an adiabatic, energy-conserving blast
wave in the limit :math:`\Gamma \gg 1`. For an external density profile

.. math::

    \rho_{\rm ext}(r) = K r^{-k},

the shock Lorentz factor satisfies

.. math::

    \Gamma^2 R^{3-k}
    =
    \frac{(17 - 4k)E}{8\pi K c^2}.

In this example we use a uniform medium, :math:`k = 0`, and choose parameters
that keep the shock relativistic while avoiding unnecessarily extreme Lorentz
factors.
"""

# %%
# Imports
# -------
# We start by importing the numerical and plotting dependencies, along with the
# Blandford-McKee shock engine.

import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const
from astropy import units as u

from trilobite.dynamics.shocks import BlandfordMcKeeShockEngine
from trilobite.physics_utils import compute_ion_electron_theta, compute_mean_molecular_weight_per_electron


# %%
# External Medium
# ---------------
# The general Blandford-McKee engine assumes a power-law external density,
#
# .. math::
#
#     \rho_{\rm ext}(r) = K_{\rm csm} r^{-k}.
#
# A uniform external medium corresponds to ``k = 0`` and
# ``K_csm = rho_0``.

rho_0 = 1.0e-24 * u.g / u.cm**3

k = 0.0
K_csm = rho_0


# %%
# Explosion Energy
# ----------------
# The Blandford-McKee energy is usually interpreted as an isotropic-equivalent
# energy. Here we choose a relatively modest value for visualization. The goal
# is not to model a particular event, but to keep the Lorentz factor in a useful
# range for plotting.

E_iso = 1.0e49 * u.erg


# %%
# Gas Parameters
# --------------
# The shocked gas is modeled with an ultra-relativistic equation of state.

gamma_hat = 4.0 / 3.0


# %%
# Build the Shock Engine
# ----------------------
# The engine emits a warning if the Lorentz factor falls below the configured
# threshold. We set the threshold to :math:`\Gamma = 2`.

engine = BlandfordMcKeeShockEngine(
    gamma_hat=gamma_hat,
    lorentz_warn_threshold=2.0,
)


# %%
# Compute the Shock Evolution
# ---------------------------
# We evaluate the solution over a time interval where the shock remains
# relativistic. In the BM approximation, the shock radius is approximately
# :math:`R \simeq ct`.

time = np.geomspace(1.0e5, 3.0e6, 200) * u.s

shock = engine.compute_shock_properties(
    time,
    E=E_iso,
    K_csm=K_csm,
    k=k,
)


# %%
# Shock Radius
# ------------
# In the ultra-relativistic limit, the shock radius is nearly linear in time.
# We plot both the BM radius and the light-travel reference radius :math:`ct`.

fig, ax = plt.subplots(figsize=(6, 4))

ax.loglog(
    time.to_value(u.s),
    shock.radius.to_value(u.cm),
    label="BM shock radius",
)

ax.loglog(
    time.to_value(u.s),
    (const.c * time).to_value(u.cm),
    linestyle="--",
    label=r"$ct$",
)

ax.set_xlabel("Time [s]")
ax.set_ylabel("Shock radius [cm]")
ax.set_title("Blandford-McKee shock radius")
ax.legend()

plt.show()


# %%
# Lorentz Factor Decay
# --------------------
# In a uniform medium, the BM Lorentz factor decreases as
# :math:`\Gamma \propto t^{-3/2}`. The horizontal line marks a rough threshold
# below which the ultra-relativistic approximation begins to break down.

fig, ax = plt.subplots(figsize=(6, 4))

ax.loglog(
    time.to_value(u.s),
    shock.lorentz_factor,
    label=r"$\Gamma$",
)

ax.axhline(
    2.0,
    linestyle="--",
    label=r"$\Gamma = 2$",
)

ax.set_xlabel("Time [s]")
ax.set_ylabel(r"Shock Lorentz factor $\Gamma$")
ax.set_title("Blandford-McKee Lorentz factor decay")
ax.legend()

plt.show()


# %%
# Relativistic Velocity Offset
# ----------------------------
# In the ultra-relativistic limit, :math:`\beta = v/c` is extremely close to
# one. Plotting :math:`\beta` directly is usually uninformative because the
# interesting evolution is hidden in the small difference :math:`1-\beta`.

fig, ax = plt.subplots(figsize=(6, 4))

ax.loglog(
    time.to_value(u.s),
    1.0 - shock.beta,
    label=r"$1-\beta$",
)

ax.set_xlabel("Time [s]")
ax.set_ylabel(r"$1 - v/c$")
ax.set_title("Deviation from light speed")
ax.legend()

plt.show()


# %%
# Post-Shock Pressure
# -------------------
# The proper post-shock pressure decreases as the blast wave decelerates and
# expands into the external medium.

fig, ax = plt.subplots(figsize=(6, 4))

ax.loglog(
    time.to_value(u.s),
    shock.post_shock_pressure.to_value(u.dyne / u.cm**2),
    label="Pressure",
)

ax.set_xlabel("Time [s]")
ax.set_ylabel(r"Post-shock pressure [$\mathrm{dyne\,cm^{-2}}$]")
ax.set_title("Post-shock pressure")
ax.legend()

plt.show()


# %%
# Post-Shock Temperatures
# -----------------------
# The jump conditions fix the total thermal energy density :math:`u_{\rm th}`, not how it is shared between ions
# and electrons. We assign the thermal electrons an energy :math:`\epsilon_T u_i` relative to the ions with
# :func:`~trilobite.physics_utils.thermal_partition.compute_ion_electron_theta`, which returns the dimensionless
# temperature :math:`\Theta = k_B T / (m c^2)` of each species. The mean molecular weight per electron comes from
# :func:`~trilobite.physics_utils.composition.compute_mean_molecular_weight_per_electron`.

mu_e = compute_mean_molecular_weight_per_electron(X=0.70, Y=0.28, Z=0.02)

fig, ax = plt.subplots(figsize=(6, 4))

for epsilon_T, color in zip([0.01, 0.1, 1.0], ["C0", "C1", "C2"]):
    Theta_i, Theta_e = compute_ion_electron_theta(
        shock.thermal_energy_density_comoving, shock.post_shock_comoving_density, mu_e, epsilon_T
    )
    ax.loglog(time.to_value(u.s), Theta_e, color=color, label=rf"$\Theta_e$, $\epsilon_T = {epsilon_T:g}$")
    ax.loglog(time.to_value(u.s), Theta_i, color=color, ls="--", label=rf"$\Theta_i$, $\epsilon_T = {epsilon_T:g}$")

ax.set_xlabel("Time [s]")
ax.set_ylabel(r"$\Theta = k_B T / m c^2$")
ax.set_title("Post-shock temperatures")
ax.legend(fontsize=8)

plt.show()
