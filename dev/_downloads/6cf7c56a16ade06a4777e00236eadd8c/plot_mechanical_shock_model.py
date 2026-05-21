r"""
Mechanical Shock Model: Forward and Reverse Shock Evolution
===========================================================

.. admonition:: What this example does

   This example walks through the full setup of the
   :class:`~triceratops.dynamics.shocks.numerical.MechanicalShockEngine` for a
   canonical core-collapse supernova expanding into a red-supergiant wind.
   It covers ejecta and CSM profile construction, source-function wrapping,
   self-consistent initial conditions, and interpretation of the two-shock
   structure in the outputs.

**When to use the mechanical engine.** Most semi-analytic transient models
collapse the shocked region to a thin shell and evolve only its radius and
velocity. This works well when the shocked ejecta and shocked CSM are in
pressure balance and the distinction between the two layers is unimportant
for the observable of interest. The
:class:`~triceratops.dynamics.shocks.numerical.MechanicalShockEngine`
instead tracks the two regions separately:

- **Region 2** — shocked ejecta, bounded by the reverse shock and the
  contact discontinuity (CD).
- **Region 3** — shocked CSM, bounded by the CD and the forward shock.

Each region carries its own mass, internal energy, and effective width,
evolved through a coupled 8-component ODE system. This retains the
distinction between forward and reverse shock speeds, lets the CD
accelerate or decelerate under the pressure difference between the two
layers, and allows independent radiative cooling terms. The cost is a
slightly more involved setup compared to
:class:`~triceratops.dynamics.shocks.numerical.PressureDrivenThinShellShockEngine`.

.. seealso::

   :ref:`mechanical_internal_energy_model`
       Derivation of the governing ODE system.

   :ref:`numeric_shocks_theory`
       Overview of all numerical shock closures and when to prefer each one.

   :class:`~triceratops.dynamics.shocks.numerical.PressureDrivenThinShellShockEngine`
       Simpler thin-shell alternative for when the two-shock structure is
       not needed.

----

"""

# %%
# Setup
# -----
#
# We import the engine, the ejecta-kernel factory, the source-function helper,
# and the Triceratops plotting utilities.

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

from triceratops.dynamics.shocks import (
    MechanicalShockEngine,
    get_bpl_ejecta_kernel,
    make_homologous_stationary_sources,
)
from triceratops.utils.plot_utils import set_plot_style

# %%
# Physical Parameters
# -------------------
#
# We adopt parameters representative of a Type IIP supernova interacting with
# a red-supergiant wind:
#
# - Ejecta energy :math:`E_{\rm ej} = 10^{51}` erg and mass
#   :math:`M_{\rm ej} = 5\,M_\odot`.  The outer velocity index :math:`n = 10`
#   and inner index :math:`\delta = 1` follow the Chevalier (1982) broken
#   power-law commonly inferred for Type II supernovae.
# - A steady red-supergiant wind with mass-loss rate
#   :math:`\dot{M} = 10^{-5}\,M_\odot\,\mathrm{yr}^{-1}` and wind speed
#   :math:`v_w = 100\;\mathrm{km\,s^{-1}}`.  These fix the CSM density
#   normalization :math:`A = \dot{M}/(4\pi v_w)`, so that
#   :math:`\rho_{\rm CSM}(r) = A\,r^{-2}`.

E_ej = 1e50 * u.erg
M_ej = 5.0 * u.M_sun
M_dot = 1e-3 * u.M_sun / u.yr
v_wind = 100.0 * u.km / u.s
v_ej = np.sqrt(2 * E_ej / M_ej)

# %%
# Ejecta Profile
# --------------
#
# The engine works with the **homologous ejecta kernel** :math:`G(v)`, defined
# so that the physical density at any time is recovered as
# :math:`\rho_{\rm ej}(r,t) = t^{-3}\,G(r/t)`.  Separating the time
# dependence this way makes the kernel time-independent, which is both
# conceptually clean and numerically efficient inside ODE right-hand sides.
#
# :func:`~triceratops.dynamics.shocks.utils.get_bpl_ejecta_kernel` normalises
# the Chevalier broken-power-law kernel to the requested mass and energy.
# For an exponential ejecta profile, use
# :func:`~triceratops.dynamics.shocks.utils.get_exponential_ejecta_kernel`
# instead; a velocity-truncated variant is available via
# :func:`~triceratops.dynamics.shocks.utils.get_truncated_bpl_ejecta_kernel`.

G_ej = get_bpl_ejecta_kernel(E_ej, M_ej, n=10, delta=1)

# %%
# CSM Profile
# -----------
#
# For a steady, spherically symmetric wind the density follows
# :math:`\rho_{\rm CSM}(r) = A\,r^{-2}`, where the normalization is set by
# mass conservation:
#
# .. math::
#
#     A = \frac{\dot{M}}{4\pi\,v_w}.
#
# We define this as a simple one-argument callable.  A ready-made factory for
# this profile is also provided by
# :func:`~triceratops.dynamics.shocks.utils.get_wind_csm_density_func`, which
# additionally handles unit conversion.  Other common profiles (uniform ISM,
# top-hat shell, smooth-truncated wind) are available in the same module.

A_cgs = (M_dot / (4 * np.pi * v_wind)).to_value(u.g / u.cm)


def rho_csm(r):
    return A_cgs / r**2


# %%
# Source Functions
# ----------------
#
# The engine ODE kernel requires four upstream callables at each timestep:
#
# - :math:`\rho_1(r,t)` — ejecta density just ahead of the reverse shock
# - :math:`u_1(r,t)` — ejecta velocity just ahead of the reverse shock
# - :math:`\rho_4(r,t)` — CSM density just ahead of the forward shock
# - :math:`u_4(r,t)` — CSM velocity just ahead of the forward shock
#
# For homologous ejecta and a stationary CSM the velocity fields are
# :math:`u_1 = r/t` and :math:`u_4 = 0`.
# :func:`~triceratops.dynamics.shocks.utils.make_homologous_stationary_sources`
# wraps the kernel :math:`G(v)` and the CSM profile into these four
# two-argument callables automatically.

rho_1, u_1, rho_4, u_4 = make_homologous_stationary_sources(
    G_ej=G_ej,
    rho_csm=rho_csm,
)

# %%
# Initial Conditions
# ------------------
#
# The engine integrates an 8-component state vector
# :math:`(R_{\rm cd}, v_{\rm cd}, M_2, M_3, U_2, U_3, \Delta_2, \Delta_3)`.
# We start at :math:`t_0 = 1` day, placing the CD at
# :math:`R_0 = 10^{14}` cm with velocity :math:`v_0 = 10^9` cm/s — values
# consistent with the Chevalier :math:`n=10`, :math:`s=2` self-similar
# solution at that epoch.
#
# :meth:`~triceratops.dynamics.shocks.numerical.MechanicalShockEngine.generate_initial_conditions`
# derives the remaining six components self-consistently from these two
# inputs, eliminating the initial transient that would otherwise arise from
# an inconsistency between the assumed shock speed and the sound-speed width
# closure.

n = 10.0
s = 2.0
m = (n - 3.0) / (n - s)

t_0_cgs = (0.1 * u.day).to_value(u.s)

v_coord_0 = 5.0 * v_ej.cgs.value  # position coordinate, not CD speed
R_cd_0 = t_0_cgs * v_coord_0
v_cd_0 = m * v_coord_0

print(R_cd_0 / 1e14, v_cd_0 / 3e10)

R0, v0, M2_0, M3_0, U2_0, U3_0, Dlt2_0, Dlt3_0 = MechanicalShockEngine.generate_initial_conditions(
    R_cd_0=R_cd_0,
    v_cd_0=v_cd_0,
    t_0=t_0_cgs,
    rho_1=rho_1,
    rho_4=rho_4,
    u_1=u_1,
    u_4=u_4,
)

# %%
# Time Grid and Integration
# -------------------------
#
# We evolve the system over three decades, from 1 to 1000 days.  The engine
# returns a :class:`~triceratops.dynamics.shocks.numerical.MechanicalShockState`
# named tuple whose fields are :class:`~astropy.units.Quantity` arrays.

time = np.geomspace(1e-1, 1000000, 4000) * u.day

engine = MechanicalShockEngine()

state = engine.compute_shock_properties(
    time=time,
    rho_1=rho_1,
    rho_4=rho_4,
    u_1=u_1,
    u_4=u_4,
    R_cd_0=R0,
    v_cd_0=v0,
    M2_0=M2_0,
    M3_0=M3_0,
    U2_0=U2_0,
    U3_0=U3_0,
    Delta2_0=Dlt2_0,
    Delta3_0=Dlt3_0,
    t_0=t_0_cgs,
    M1_total=M_ej,
)

# %%
# Visualization
# -------------
#
# We use a consistent color scheme across all three panels:
#
# - **Orange** (#DD8452): reverse shock and shocked-ejecta region (Region 2)
# - **Blue** (#4C72B0): contact discontinuity
# - **Green** (#55A868): forward shock and shocked-CSM region (Region 3)
#
# **Radii.** The forward shock sweeps up CSM and expands fastest; the reverse
# shock moves outward in the lab frame but at a speed lower than the CD,
# so it gradually falls behind the ejecta bulk.  Their separation encodes the
# thickness of each shocked layer.
#
# **Speeds.** All three boundaries follow rough power laws set by the
# self-similar solution, with deviations driven by the evolving pressure
# balance between the two regions.
#
# **Shocked masses.** Region 2 accumulates mass from the ejecta (falling
# density profile), while Region 3 sweeps up CSM (wind :math:`r^{-2}`, giving
# :math:`M_3 \propto R_{\rm fs}`).

C_RS = "#DD8452"  # orange — ejecta / reverse-shock side
C_CD = "#4C72B0"  # blue   — contact discontinuity
C_FS = "#55A868"  # green  — CSM / forward-shock side
LW = 1.8

set_plot_style()

t_days = time.to_value(u.day)

fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)

# --- Panel 1: Radii ---
axes[0].loglog(t_days, state.radius_rs.to_value(u.cm), color=C_RS, lw=LW, label="Reverse shock")
axes[0].loglog(t_days, state.radius.to_value(u.cm), color=C_CD, lw=LW, label="Contact discontinuity")
axes[0].loglog(t_days, state.radius_fs.to_value(u.cm), color=C_FS, lw=LW, label="Forward shock")
axes[0].set_ylabel("Radius (cm)")
axes[0].legend(frameon=False, fontsize=9)
axes[0].grid(alpha=0.2, which="both")

# --- Panel 2: Speeds ---
axes[1].loglog(t_days, np.abs(state.velocity_rs.to_value(u.cm / u.s)), color=C_RS, lw=LW, label="Reverse shock")
axes[1].loglog(t_days, state.velocity.to_value(u.cm / u.s), color=C_CD, lw=LW, label="Contact discontinuity")
axes[1].loglog(t_days, state.velocity_fs.to_value(u.cm / u.s), color=C_FS, lw=LW, label="Forward shock")
axes[1].set_ylabel("Speed (cm/s)")
axes[1].legend(frameon=False, fontsize=9)
axes[1].grid(alpha=0.2, which="both")

# --- Panel 3: Shocked masses ---
axes[2].loglog(t_days, state.mass_2.to_value(u.M_sun), color=C_RS, lw=LW, label=r"Region 2 (shocked ejecta)")
axes[2].loglog(t_days, state.mass_3.to_value(u.M_sun), color=C_FS, lw=LW, label=r"Region 3 (shocked CSM)")
axes[2].axhline(M_ej.to_value(u.M_sun), ls="--", color="k", lw=1, label="Total ejecta mass")
axes[2].set_xlabel("Time (days)")
axes[2].set_ylabel(r"Mass ($M_\odot$)")
axes[2].legend(frameon=False, fontsize=9)
axes[2].grid(alpha=0.2, which="both")

plt.tight_layout()
plt.show()
