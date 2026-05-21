r"""
Weak vs Strong Shock Jump Conditions (Cold Medium)
=================================================

.. admonition:: What this example does

   This example compares weak-shock and strong-shock Rankine–Hugoniot
   solvers in a cold upstream medium. By sweeping the shock velocity
   from subsonic to highly supersonic values, it shows how the weak-shock
   solution smoothly transitions into the strong-shock limit.

Shock jump conditions relate the **upstream (pre-shock)** state to the **downstream
(post-shock)** state via conservation of mass, momentum, and energy.

In the Newtonian regime, two important limits emerge:

- **Weak shocks** (:math:`\mathcal{M}_1 \gtrsim 1`):
  small perturbations where compression, pressure, and temperature
  change gradually with Mach number.

- **Strong shocks** (:math:`\mathcal{M}_1 \gg 1`):
  asymptotic limit where the compression ratio becomes constant
  and thermodynamic quantities scale simply with velocity.

In this example, we compare:

- :class:`~triceratops.dynamics.shocks.core.rankine_hugoniot.WeakShockConditions`
- :class:`~triceratops.dynamics.shocks.core.rankine_hugoniot.StrongColdShockConditions`

to show how the weak-shock solution approaches the strong-shock limit
as the Mach number increases.

----
"""

# %%
# Physical Setup
# --------------
#
# We consider a shock propagating into a cold, stationary medium.
#
# **Key assumptions:**
#
# - Upstream medium is cold: :math:`P_1 \approx 0`
# - Flow is initially at rest: :math:`v_1 = 0`
# - Ideal gas with :math:`\gamma = 5/3`
#
# Unlike the relativistic example, the relevant control parameter here
# is the **Mach number**:
#
# .. math::
#     \mathcal{M}_1 = \frac{v_{\rm sh}}{c_s}
#
# where :math:`c_s` is the upstream sound speed.

import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u

from triceratops.dynamics.shocks.core import (
    WeakShockConditions,
    StrongColdShockConditions,
)
from triceratops.utils.plot_utils import set_plot_style

GAMMA = 5.0 / 3.0

# Cold upstream density
rho_csm = 1e-24 * u.g / u.cm**3

# Small but finite upstream temperature (needed for weak shocks)
T_upstream = 1e4 * u.K


# %%
# Sweeping Shock Strength
# -----------------------
#
# We vary the shock strength over a wide range of Mach numbers:
#
# .. math::
#     \mathcal{M}_1 \in [1.01, 100]
#
# This spans:
#
# - Near-sonic regime (weak shocks)
# - Intermediate regime (transition)
# - Strong-shock limit
#
# The strong-shock solution does not depend on Mach number explicitly,
# so deviations between the two highlight where the strong-shock
# approximation breaks down.

mach_numbers = np.logspace(0.01, 2, 300)

# Convert Mach number → velocity using sound speed
k_B = 1.380649e-16  # CGS
m_p = 1.6726219e-24
mu = 0.61

c_s = np.sqrt(GAMMA * k_B * T_upstream.value / (mu * m_p)) * u.cm / u.s
shock_velocities = mach_numbers * c_s


weak_results = [
    WeakShockConditions.solve(
        shock_velocity=v,
        flow_density=rho_csm,
        flow_temperature=T_upstream,
        gamma=GAMMA,
    )
    for v in shock_velocities
]

strong_results = [
    StrongColdShockConditions.solve(
        shock_velocity=v,
        flow_density=rho_csm,
        gamma=GAMMA,
    )
    for v in shock_velocities
]


weak_R = np.array([r.compression_ratio for r in weak_results])
strong_R = np.array([r.compression_ratio for r in strong_results])

weak_T = np.array([r.post_shock_temperature.to_value(u.K) for r in weak_results])
strong_T = np.array([r.post_shock_temperature.to_value(u.K) for r in strong_results])

weak_P = np.array([r.post_shock_pressure.to_value(u.dyn / u.cm**2) for r in weak_results])
strong_P = np.array([r.post_shock_pressure.to_value(u.dyn / u.cm**2) for r in strong_results])


# %%
# Compression Ratio
# -----------------
#
# Weak shocks produce only small compressions near :math:`\mathcal{M}_1 \approx 1`,
# while strong shocks asymptote to:
#
# .. math::
#     R = \frac{\gamma + 1}{\gamma - 1}
#
# The weak-shock solution smoothly approaches this limit at high Mach number.

set_plot_style()
fig, ax = plt.subplots(figsize=(5, 4))

ax.semilogx(mach_numbers, weak_R, lw=2.2, color="#4C72B0", label="Weak shock")
ax.semilogx(mach_numbers, strong_R, ls="--", color="#DD8452", label="Strong limit")

ax.set_xlabel(r"Mach number $\mathcal{M}_1$")
ax.set_ylabel("Compression ratio $R$")
ax.set_title("Compression Ratio")

ax.legend(frameon=False)
ax.grid(alpha=0.25)

plt.tight_layout()
plt.show()


# %%
# Post-shock Temperature
# ----------------------
#
# Temperature increases gradually for weak shocks, but transitions
# to the familiar quadratic scaling in the strong-shock limit.

fig, ax = plt.subplots(figsize=(5, 4))

ax.loglog(mach_numbers, weak_T, lw=2.2, color="#4C72B0", label="Weak shock")
ax.loglog(mach_numbers, strong_T, ls="--", color="#DD8452", label="Strong limit")

ax.set_xlabel(r"Mach number $\mathcal{M}_1$")
ax.set_ylabel(r"$T_2\ (\mathrm{K})$")
ax.set_title("Post-shock Temperature")

ax.legend(frameon=False)
ax.grid(alpha=0.25, which="both")

plt.tight_layout()
plt.show()


# %%
# Post-shock Pressure
# -------------------
#
# Pressure behaves similarly, with weak shocks producing only modest
# increases until the strong-shock scaling dominates.

fig, ax = plt.subplots(figsize=(5, 4))

ax.loglog(mach_numbers, weak_P, lw=2.2, color="#4C72B0", label="Weak shock")
ax.loglog(mach_numbers, strong_P, ls="--", color="#DD8452", label="Strong limit")

ax.set_xlabel(r"Mach number $\mathcal{M}_1$")
ax.set_ylabel(r"$P_2\ (\mathrm{dyn\,cm^{-2}})$")
ax.set_title("Post-shock Pressure")

ax.legend(frameon=False)
ax.grid(alpha=0.25, which="both")

plt.tight_layout()
plt.show()
