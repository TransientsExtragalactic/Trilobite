r"""
Momentum-Conserving Snowplow Shock Evolution
============================================

.. admonition:: What this example does

   This example demonstrates the
   :class:`~trilobite.dynamics.shocks.numerical.MomentumConservingShockEngine` for a
   canonical core-collapse supernova expanding into a red-supergiant wind.  It covers
   profile construction, source-function assembly, and visualization of the shell
   kinematics — including a comparison with the pressure-driven thin-shell closure to
   illustrate when the two models diverge.

**When to use the momentum-conserving engine.** The snowplow closure assumes that
post-shock pressure is negligible: all kinetic energy deposited by swept-up material
is radiated away immediately, leaving the shell cold.  Shell acceleration is driven
entirely by the momentum flux from swept-up ejecta and CSM, with no pressure gradient
term.  This is the appropriate limit for the *radiative snowplow* phase of an older
supernova remnant.  It converges to :math:`R \propto t^{1/4}` in uniform CSM — a
distinctly softer deceleration than the Sedov-Taylor :math:`R \propto t^{2/5}` or the
pressure-driven thin-shell :math:`R \propto t^{4/13}`.

.. seealso::

   :ref:`conservative_snowplow_model`
       Derivation of the governing ODE system.

   :ref:`momentum_conserving_shock_engine`
       User-guide description with state-field reference.

   :class:`~trilobite.dynamics.shocks.numerical.PressureDrivenThinShellShockEngine`
       Adiabatic analogue that retains post-shock pressure.

----

"""

# %%
# Setup
# -----
#
# We import the engines, profile classes, source-function helper, and plotting utilities.

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

from trilobite.dynamics.profiles import BrokenPowerLawEjectaProfile, WindCSMProfile
from trilobite.dynamics.shocks import (
    MomentumConservingShockEngine,
    PressureDrivenThinShellShockEngine,
    make_homologous_stationary_sources,
)
from trilobite.utils.plot_utils import set_plot_style

# %%
# Physical Parameters
# -------------------
#
# We adopt parameters representative of a core-collapse supernova expanding into
# a red-supergiant wind:
#
# - Ejecta energy :math:`E_{\rm ej} = 10^{51}` erg, mass
#   :math:`M_{\rm ej} = 5\,M_\odot`, outer power-law index :math:`n = 10`,
#   inner index :math:`\delta = 1`.
# - Wind mass-loss rate :math:`\dot{M} = 10^{-5}\,M_\odot\,\mathrm{yr}^{-1}`,
#   wind speed :math:`v_w = 100\;\mathrm{km\,s^{-1}}`.

E_ej = 1e51 * u.erg
M_ej = 5.0 * u.Msun
M_dot = 1e-5 * u.Msun / u.yr
v_wind = 100.0 * u.km / u.s

# %%
# Profile Construction
# --------------------
#
# :class:`~trilobite.dynamics.profiles.ejecta.BrokenPowerLawEjectaProfile` normalizes
# the Chevalier broken-power-law kernel to the requested mass and energy and
# returns a fast unit-free callable.
#
# :class:`~trilobite.dynamics.profiles.csm.WindCSMProfile` constructs the
# :math:`\rho \propto r^{-2}` wind density callable.

K, v_t = BrokenPowerLawEjectaProfile.normalize(E_ej=E_ej, M_ej=M_ej, n=10, delta=1)
rho_ej = BrokenPowerLawEjectaProfile.as_optimized_callable(K=K, v_t=v_t, n=10, delta=1)
rho_csm = WindCSMProfile.as_optimized_callable(mass_loss_rate=M_dot, wind_velocity=v_wind)

# %%
# Source Functions
# ----------------
#
# Both engines require four upstream callables:
#
# - :math:`\rho_1(r,t)` — ejecta density ahead of the reverse shock.
# - :math:`u_1(r,t)` — ejecta velocity ahead of the reverse shock.
# - :math:`\rho_4(r,t)` — CSM density ahead of the forward shock.
# - :math:`u_4(r,t)` — CSM velocity ahead of the forward shock (zero for stationary CSM).
#
# :func:`~trilobite.dynamics.shocks.utils.make_homologous_stationary_sources`
# assembles all four from the ejecta and CSM density callables under the homologous
# expansion assumption :math:`u_1 = r/t`, :math:`u_4 = 0`.

rho_1, u_1, rho_4, u_4 = make_homologous_stationary_sources(rho_ej, rho_csm)

# %%
# Engine Initialization
# ---------------------
#
# Both engines are stateless; a single instance can be reused across multiple
# parameter surveys.

engine_sp = MomentumConservingShockEngine()
engine_pd = PressureDrivenThinShellShockEngine()

# %%
# Initial Conditions and Time Grid
# ---------------------------------
#
# We place the shell at :math:`R_0 = 10^{14}` cm with velocity
# :math:`v_0 = 10^9` cm/s and a nominal initial mass consistent with the swept-up
# CSM at that radius.  Both engines use the same initial conditions so that their
# divergence is purely due to the closure assumption.

R_0 = 1e14 * u.cm
v_0 = 1e9 * u.cm / u.s
M_0 = 1e26 * u.g
t_0 = 1.0 * u.day

time = np.geomspace(1, 2000, 600) * u.day

# %%
# Shock Evolution
# ---------------
#
# We integrate both closures over the same time grid.  The
# :class:`~trilobite.dynamics.shocks.numerical.MomentumConservingShockState` and
# :class:`~trilobite.dynamics.shocks.numerical.ThinShellShockState` named tuples
# carry unit-bearing :class:`~astropy.units.Quantity` fields for all shock diagnostics.

state_sp = engine_sp.compute_shock_properties(
    time=time,
    rho_1=rho_1,
    rho_4=rho_4,
    u_1=u_1,
    u_4=u_4,
    R_0=R_0,
    v_0=v_0,
    M_0=M_0,
    t_0=t_0,
)

state_pd = engine_pd.compute_shock_properties(
    time=time,
    rho_1=rho_1,
    rho_4=rho_4,
    u_1=u_1,
    u_4=u_4,
    R_0=R_0,
    v_0=v_0,
    M_0=M_0,
    t_0=t_0,
)

# %%
# Visualization: Shell Kinematics
# --------------------------------
#
# We compare the radius and velocity histories from the two closures.  The
# momentum-conserving engine decelerates more aggressively because it has no
# pressure support to resist the added inertia of swept-up material.

set_plot_style()

t_days = time.to_value(u.day)

fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

axes[0].loglog(t_days, state_sp.radius.to_value(u.cm), label="Momentum-conserving (snowplow)")
axes[0].loglog(t_days, state_pd.radius.to_value(u.cm), ls="--", label="Pressure-driven thin-shell")
axes[0].set_ylabel("Radius (cm)")
axes[0].legend(frameon=False, fontsize=9)
axes[0].grid(alpha=0.2, which="both")

axes[1].loglog(t_days, state_sp.velocity.to_value(u.km / u.s), label="Momentum-conserving (snowplow)")
axes[1].loglog(t_days, state_pd.velocity.to_value(u.km / u.s), ls="--", label="Pressure-driven thin-shell")
axes[1].set_xlabel("Time (days)")
axes[1].set_ylabel(r"Velocity (km s$^{-1}$)")
axes[1].legend(frameon=False, fontsize=9)
axes[1].grid(alpha=0.2, which="both")

plt.tight_layout()
plt.show()

# %%
# Visualization: Post-Shock Diagnostics
# ---------------------------------------
#
# Both engines compute forward-shock post-shock thermodynamics as a diagnostic
# from the instantaneous Rankine--Hugoniot conditions.  Here we compare the
# forward-shock post-shock pressure and temperature as a function of time.

fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

axes[0].loglog(
    t_days,
    state_sp.post_shock_pressure.to_value(u.dyn / u.cm**2),
    label="Momentum-conserving (snowplow)",
)
axes[0].loglog(
    t_days,
    state_pd.post_shock_pressure.to_value(u.dyn / u.cm**2),
    ls="--",
    label="Pressure-driven thin-shell",
)
axes[0].set_ylabel(r"Post-shock pressure (dyn cm$^{-2}$)")
axes[0].legend(frameon=False, fontsize=9)
axes[0].grid(alpha=0.2, which="both")

axes[1].loglog(
    t_days,
    state_sp.post_shock_temperature.to_value(u.K),
    label="Momentum-conserving (snowplow)",
)
axes[1].loglog(
    t_days,
    state_pd.post_shock_temperature.to_value(u.K),
    ls="--",
    label="Pressure-driven thin-shell",
)
axes[1].set_xlabel("Time (days)")
axes[1].set_ylabel("Post-shock temperature (K)")
axes[1].legend(frameon=False, fontsize=9)
axes[1].grid(alpha=0.2, which="both")

plt.tight_layout()
plt.show()
# sphinx_gallery_thumbnail_number = 1
