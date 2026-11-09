"""
SNe Shocks In Different CSM Environments
===================================================================

This example explores how a pressure-driven supernova shock evolves when the
same ejecta model is launched into several different circumstellar medium
(CSM) density profiles.  It uses the
:class:`~trilobite.dynamics.shocks.numerical.PressureDrivenThinShellShockEngine`
together with the CSM profile classes in
:mod:`~trilobite.dynamics.profiles`.

The goal is not to fit a specific transient, but to build intuition for how
density structure imprints itself on the shock dynamics. We compare shocks
propagating into:

- a low-density uniform ISM (:class:`~trilobite.dynamics.profiles.csm.UniformCSMProfile`),
- a dense uniform medium (:class:`~trilobite.dynamics.profiles.csm.UniformCSMProfile`),
- a steady wind (:class:`~trilobite.dynamics.profiles.csm.WindCSMProfile`),
- a wind with an outer ISM floor (:class:`~trilobite.dynamics.profiles.csm.WindWithFloorCSMProfile`),
- a sharply truncated wind (:class:`~trilobite.dynamics.profiles.csm.TruncatedWindCSMProfile`),
- a smoothly truncated wind (:class:`~trilobite.dynamics.profiles.csm.SmoothTruncatedWindCSMProfile`),
- a dense top-hat shell (:class:`~trilobite.dynamics.profiles.csm.ShellCSMProfile`),
- a smooth Gaussian shell (:class:`~trilobite.dynamics.profiles.csm.GaussianShellCSMProfile`),
- a broken power-law CSM (:class:`~trilobite.dynamics.profiles.csm.BrokenPowerLawCSMProfile`),
- a general power-law CSM (:class:`~trilobite.dynamics.profiles.csm.PowerLawCSMProfile`),
- an exponential CSM (:class:`~trilobite.dynamics.profiles.csm.ExponentialCSMProfile`),
- a smooth broken power-law CSM (:class:`~trilobite.dynamics.profiles.csm.SmoothBPLCSMProfile`),
- a cored power-law CSM (:class:`~trilobite.dynamics.profiles.csm.CoredPowerLawCSMProfile`),
- a finite-duration wind (:class:`~trilobite.dynamics.profiles.csm.FiniteWindCSMProfile`),
- a stellar wind bubble (:class:`~trilobite.dynamics.profiles.csm.WindBubbleCSMProfile`),
- and a log-normal shell (:class:`~trilobite.dynamics.profiles.csm.LogNormalCSMProfile`).

All models use the same homologous broken-power-law ejecta profile and the
same initial shock conditions. This isolates the dynamical effect of the CSM
structure.

.. seealso::

    :ref:`numerical_shocks_overview` — full API reference for numerical shock
    engines, including the pressure-driven and mechanical formulations.

    :ref:`numeric_shocks_theory` — derivation of the thin-shell equations of
    motion.

    :ref:`shock_engines` — overview of all shock engines and guidance on
    choosing between self-similar and numerical approaches.
"""

# %%
# Setup
# -----
#
# We begin with the numerical, unit, plotting, and Trilobite imports.

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

from trilobite.dynamics.profiles import (
    BrokenPowerLawCSMProfile,
    BrokenPowerLawEjectaProfile,
    CoredPowerLawCSMProfile,
    ExponentialCSMProfile,
    FiniteWindCSMProfile,
    GaussianShellCSMProfile,
    LogNormalCSMProfile,
    PowerLawCSMProfile,
    ShellCSMProfile,
    SmoothBPLCSMProfile,
    SmoothTruncatedWindCSMProfile,
    TruncatedWindCSMProfile,
    UniformCSMProfile,
    WindBubbleCSMProfile,
    WindCSMProfile,
    WindWithFloorCSMProfile,
)
from trilobite.dynamics.shocks.numerical import PressureDrivenThinShellShockEngine
from trilobite.dynamics.shocks.utils import make_homologous_stationary_sources
from trilobite.utils.plot_utils import set_plot_style


# %%
# Scenario Container
# ------------------
#
# To keep the gallery clean, we define a small container for each CSM scenario.
# Each scenario stores a human-readable label, the CSM density callable, and
# optional marker radii that are useful for interpreting sharp features such as
# shell edges or wind termination radii.


@dataclass
class CSMScenario:
    label: str
    rho_csm: callable
    color: str
    marker_radii: tuple[u.Quantity, ...] = ()


# %%
# Shared Physical Parameters
# --------------------------
#
# We use the same ejecta and initial shock setup for every CSM profile. These
# parameters are broadly representative of a mildly energetic core-collapse
# supernova.
#
# The ejecta are modeled as a homologously expanding broken power law,
#
# .. math::
#
#     \rho_{\rm ej}(r,t) = t^{-3} G_{\rm ej}(r/t),
#
# with an outer velocity-space slope ``n=10`` and an inner slope ``delta=0``.
# :class:`~trilobite.dynamics.profiles.csm.BrokenPowerLawEjectaProfile` normalizes
# the kernel to the explosion energy and ejecta mass and returns a fast
# unit-free :math:`\rho_{\rm ej}(r,t)` callable via
# :meth:`~trilobite.dynamics.profiles.csm.BrokenPowerLawEjectaProfile.as_optimized_callable`.
#
# A single stateless
# :class:`~trilobite.dynamics.shocks.numerical.PressureDrivenThinShellShockEngine`
# instance is shared across all scenarios; see
# :ref:`pressure_driven_thin_shell_engine` for the equations it integrates.

E_ej = 1e50 * u.erg
M_ej = 5.0 * u.M_sun

K_ej, v_t_ej = BrokenPowerLawEjectaProfile.normalize(E_ej=E_ej, M_ej=M_ej, n=10, delta=0)
G_ej = BrokenPowerLawEjectaProfile.as_optimized_callable(K=K_ej, v_t=v_t_ej, n=10, delta=0)

engine = PressureDrivenThinShellShockEngine()

time = np.geomspace(1e-1, 300000.0, 500) * u.day

R_0 = 1e10 * u.cm
v_0 = 1e9 * u.cm / u.s
t_0 = 1e1 * u.s
M_0 = 1e-4 * u.M_sun


# %%
# CSM Profile Gallery
# -------------------
#
# We now define the CSM environments using the profile classes in
# :mod:`~trilobite.dynamics.profiles`. Each profile's
# :meth:`~trilobite.dynamics.profiles.csm.StationaryCSMDensityProfile.as_optimized_callable`
# classmethod freezes the physical parameters once and returns a unit-free
# ``(r, t)`` callable suitable for use inside the ODE integrator.
#
# The profiles span the full range of CSM morphologies available in
# Trilobite:
#
# - **Power-law family**: uniform (:math:`s=0`), general power-law, wind
#   (:math:`s=2`), broken power-law (sharp and smooth), and cored power-law.
# - **Wind variants**: steady wind, wind with ISM floor, truncated (sharp and
#   smooth), finite-duration, and WR wind-bubble (four-zone).
# - **Shell structures**: top-hat shell, Gaussian shell, and log-normal shell.
# - **Exponential**: falling exponential from a reference radius (AGB
#   atmosphere or extended stellar envelope).
#
# Sharp structures — such as a top-hat shell or a sharply truncated wind —
# can produce abrupt features in the shock velocity, while smooth profiles
# produce gentler transitions.

rho_ism = 1e-24 * u.g / u.cm**3
rho_dense = 1e-21 * u.g / u.cm**3

M_dot_rsg = 1e-5 * u.M_sun / u.yr
v_wind_rsg = 10.0 * u.km / u.s
A_wind = WindCSMProfile.normalize(mass_loss_rate=M_dot_rsg, wind_velocity=v_wind_rsg)

R_wind = 3e17 * u.cm
dR_wind = 5e16 * u.cm

R_shell_inner = 5e16 * u.cm
R_shell_outer = 1.2e17 * u.cm
R_shell_center = 8e16 * u.cm
sigma_shell = 1.5e16 * u.cm

R_break = 1e17 * u.cm
R_core = 1e15 * u.cm

R_finite_wind_min = 1e15 * u.cm
R_finite_wind_max = 2e17 * u.cm

R_wind_termination = 3e16 * u.cm
R_bubble_shell_inner = 3e17 * u.cm
R_bubble_shell_outer = 4e17 * u.cm

scenarios = [
    CSMScenario(
        label="Uniform ISM",
        rho_csm=UniformCSMProfile.as_optimized_callable(rho_0=rho_ism),
        color="C0",
    ),
    CSMScenario(
        label="Dense uniform CSM",
        rho_csm=UniformCSMProfile.as_optimized_callable(rho_0=rho_dense),
        color="C1",
    ),
    CSMScenario(
        label="RSG-like wind",
        rho_csm=WindCSMProfile.as_optimized_callable(
            mass_loss_rate=M_dot_rsg,
            wind_velocity=v_wind_rsg,
        ),
        color="C2",
    ),
    CSMScenario(
        label="Wind + ISM floor",
        rho_csm=WindWithFloorCSMProfile.as_optimized_callable(
            mass_loss_rate=M_dot_rsg,
            wind_velocity=v_wind_rsg,
            density_floor=rho_ism,
        ),
        color="C3",
    ),
    CSMScenario(
        label="Truncated wind",
        rho_csm=TruncatedWindCSMProfile.as_optimized_callable(
            mass_loss_rate=M_dot_rsg,
            wind_velocity=v_wind_rsg,
            r_max=R_wind,
            density_floor=rho_ism,
        ),
        color="C4",
        marker_radii=(R_wind,),
    ),
    CSMScenario(
        label="Smooth truncated wind",
        rho_csm=SmoothTruncatedWindCSMProfile.as_optimized_callable(
            mass_loss_rate=M_dot_rsg,
            wind_velocity=v_wind_rsg,
            r_max=R_wind,
            transition_width=dR_wind,
            density_floor=rho_ism,
        ),
        color="C5",
        marker_radii=(R_wind,),
    ),
    CSMScenario(
        label="Dense top-hat shell",
        rho_csm=ShellCSMProfile.as_optimized_callable(
            r_inner=R_shell_inner,
            r_outer=R_shell_outer,
            shell_density=3e-19 * u.g / u.cm**3,
            density_floor=rho_ism,
        ),
        color="C6",
        marker_radii=(R_shell_inner, R_shell_outer),
    ),
    CSMScenario(
        label="Gaussian shell",
        rho_csm=GaussianShellCSMProfile.as_optimized_callable(
            background_density=rho_ism,
            shell_density=3e-19 * u.g / u.cm**3,
            shell_radius=R_shell_center,
            shell_width=sigma_shell,
        ),
        color="C7",
        marker_radii=(R_shell_center,),
    ),
    CSMScenario(
        label="Broken power-law CSM",
        rho_csm=BrokenPowerLawCSMProfile.as_optimized_callable(
            density_break=1e-20 * u.g / u.cm**3,
            radius_break=R_break,
            slope_inner=0.0,
            slope_outer=2.5,
        ),
        color="C8",
        marker_radii=(R_break,),
    ),
    CSMScenario(
        label=r"Power-law CSM ($s=1.5$)",
        rho_csm=PowerLawCSMProfile.as_optimized_callable(
            rho_ref=1e-20 * u.g / u.cm**3,
            r_ref=1e16 * u.cm,
            slope=1.5,
        ),
        color="C9",
    ),
    CSMScenario(
        label="Exponential CSM",
        rho_csm=ExponentialCSMProfile.as_optimized_callable(
            rho_0=1e-19 * u.g / u.cm**3,
            r_0=0.0 * u.cm,
            scale_height=1e16 * u.cm,
            density_floor=rho_ism,
        ),
        color="#e377c2",
    ),
    CSMScenario(
        label="Smooth BPL CSM",
        rho_csm=SmoothBPLCSMProfile.as_optimized_callable(
            density_break=1e-20 * u.g / u.cm**3,
            radius_break=R_break,
            slope_inner=0.0,
            slope_outer=2.5,
            smoothness=0.5,
        ),
        color="#7f7f7f",
        marker_radii=(R_break,),
    ),
    CSMScenario(
        label="Cored power-law CSM",
        rho_csm=CoredPowerLawCSMProfile.as_optimized_callable(
            rho_0=5e-22 * u.g / u.cm**3,
            r_core=R_core,
            slope=2.0,
        ),
        color="#bcbd22",
        marker_radii=(R_core,),
    ),
    CSMScenario(
        label="Finite wind",
        rho_csm=FiniteWindCSMProfile.as_optimized_callable(
            A=A_wind,
            r_min=R_finite_wind_min,
            r_max=R_finite_wind_max,
            density_floor=rho_ism,
        ),
        color="#17becf",
        marker_radii=(R_finite_wind_min, R_finite_wind_max),
    ),
    CSMScenario(
        label="Wind bubble (WR)",
        rho_csm=WindBubbleCSMProfile.as_optimized_callable(
            A=A_wind,
            r_wind_termination=R_wind_termination,
            rho_bubble=1e-25 * u.g / u.cm**3,
            r_shell_inner=R_bubble_shell_inner,
            r_shell_outer=R_bubble_shell_outer,
            rho_shell=1e-21 * u.g / u.cm**3,
            rho_ism=rho_ism,
        ),
        color="#aec7e8",
        marker_radii=(R_wind_termination, R_bubble_shell_inner, R_bubble_shell_outer),
    ),
    CSMScenario(
        label="Log-normal shell",
        rho_csm=LogNormalCSMProfile.as_optimized_callable(
            rho_0=5e-19 * u.g / u.cm**3,
            r_peak=R_shell_center,
            sigma=0.5,
        ),
        color="#ffbb78",
        marker_radii=(R_shell_center,),
    ),
]


# %%
# Visualize the CSM Density Profiles
# ----------------------------------
#
# Before running the shock model, we plot the density structures themselves.
# This is an important sanity check: it makes it much easier to attribute
# features in the shock evolution — velocity drops, kinks, or plateaus — to
# the density jump or transition that caused them.

set_plot_style()

radius_grid = np.geomspace(1e14, 3e19, 800) * u.cm
radius_cgs = radius_grid.to_value(u.cm)

fig, ax = plt.subplots(figsize=(8, 5))

for scenario in scenarios:
    rho_vals = scenario.rho_csm(radius_cgs, 0.0)
    ax.loglog(
        radius_cgs,
        rho_vals,
        color=scenario.color,
        lw=1.8,
        label=scenario.label,
    )

for scenario in scenarios:
    for radius_marker in scenario.marker_radii:
        ax.axvline(
            radius_marker.to_value(u.cm),
            color=scenario.color,
            ls=":",
            lw=1.0,
            alpha=0.5,
        )

ax.set_xlabel("Radius (cm)")
ax.set_ylabel(r"CSM Density ($\mathrm{g\,cm^{-3}}$)")
ax.set_ylim([1e-25, 1e-14])
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.show()


# %%
# Compute Shock Evolution
# -----------------------
#
# :func:`~trilobite.dynamics.shocks.utils.make_homologous_stationary_sources`
# assembles the four upstream source callables expected by the engine from a
# single ejecta kernel :math:`G_{\rm ej}` and a one-argument CSM density
# function:
#
# .. math::
#
#     \rho_1(r,t) = t^{-3}G_{\rm ej}(r/t),
#     \qquad
#     u_1(r,t) = r/t,
#     \qquad
#     \rho_4(r,t) = \rho_{\rm CSM}(r),
#     \qquad
#     u_4(r,t) = 0.
#
# We then call
# :meth:`~trilobite.dynamics.shocks.numerical.PressureDrivenThinShellShockEngine.compute_shock_properties`
# with the same time grid and initial conditions for every CSM environment.
# Each call returns a
# :class:`~trilobite.dynamics.shocks.numerical.ThinShellShockState` with
# the shell radius, velocity, mass, and post-shock thermodynamic quantities as
# :class:`~astropy.units.Quantity` arrays.

states = {}

for scenario in scenarios:
    rho_1, u_1, rho_4, u_4 = make_homologous_stationary_sources(G_ej, scenario.rho_csm)

    states[scenario.label] = engine.compute_shock_properties(
        time=time,
        rho_1=rho_1,
        rho_4=rho_4,
        u_1=u_1,
        u_4=u_4,
        R_0=R_0,
        v_0=v_0,
        t_0=t_0,
        M_0=M_0,
    )


# %%
# Shock Radius Comparison
# -----------------------
#
# The shock radius shows how quickly each blast wave propagates through its
# environment. Low-density environments allow nearly free expansion for longer,
# while dense CSM profiles cause earlier deceleration.

t_days = time.to_value(u.day)

fig, ax = plt.subplots(figsize=(8, 5))

for scenario in scenarios:
    state = states[scenario.label]
    ax.loglog(
        t_days,
        state.radius.to_value(u.cm),
        color=scenario.color,
        lw=2,
        label=scenario.label,
    )

ax.set_xlabel("Time (days)")
ax.set_ylabel("Shock Radius (cm)")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.show()


# %%
# Shock Velocity Comparison
# -------------------------
#
# The shock velocity is usually the clearest way to see the dynamical response
# to CSM structure.
#
# - Dense uniform media cause rapid, monotonic deceleration.
# - Wind profiles produce smoother evolution because swept-up mass grows
#   differently with radius (:math:`M_{\rm sw} \propto R` for a wind vs.
#   :math:`M_{\rm sw} \propto R^3` for a uniform medium).
# - Shells produce localized velocity drops when the shock crosses the
#   overdense structure; the log-normal and Gaussian variants show smoother
#   versions of this feature compared with the sharp top-hat shell.
# - A truncated wind produces a clear inflection when the shock exits the wind
#   and enters the lower-density ambient medium; the finite wind shows a
#   similar re-acceleration after the outer truncation radius.
# - The wind-bubble profile produces a two-stage signature: deceleration in
#   the free-wind zone, acceleration through the low-density bubble, then
#   deceleration again in the dense swept-up shell.
# - The smooth BPL and cored power-law profiles produce gentler, more gradual
#   transitions compared with their sharp-break counterparts.

fig, ax = plt.subplots(figsize=(8, 5))

v0_kms = v_0.to_value(u.km / u.s)

for scenario in scenarios:
    state = states[scenario.label]
    ax.loglog(
        t_days,
        state.velocity.to_value(u.km / u.s),
        color=scenario.color,
        lw=2,
        label=scenario.label,
    )

ax.axhline(
    v0_kms,
    color="0.4",
    ls="--",
    lw=1.2,
    label=r"Initial velocity $v_0$",
)

ax.set_xlabel("Time (days)")
ax.set_ylabel(r"Shock Velocity ($\mathrm{km\,s^{-1}}$)")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.show()


# %%
# Shock Velocity as a Function of Radius
# --------------------------------------
#
# Plotting velocity against radius connects the dynamical response directly to
# the density structures shown above. This is especially useful for shell and
# truncated-wind profiles, because the dynamically relevant feature is tied to
# a physical radius rather than a particular time.

fig, ax = plt.subplots(figsize=(8, 5))

for scenario in scenarios:
    state = states[scenario.label]
    ax.loglog(
        state.radius.to_value(u.cm),
        state.velocity.to_value(u.km / u.s),
        color=scenario.color,
        lw=2,
        label=scenario.label,
    )

    for radius_marker in scenario.marker_radii:
        ax.axvline(
            radius_marker.to_value(u.cm),
            color=scenario.color,
            ls=":",
            lw=1.0,
            alpha=0.5,
        )

ax.set_xlabel("Shock Radius (cm)")
ax.set_ylabel(r"Shock Velocity ($\mathrm{km\,s^{-1}}$)")
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.show()


# %%
# Density Sampled by the Shock
# ----------------------------
#
# Finally, we evaluate the ambient density at the shock radius itself. This
# maps each dynamical trajectory onto the density actually encountered by the
# forward shock and is useful for understanding which portion of the CSM
# profile is dynamically active at each moment.

fig, ax = plt.subplots(figsize=(8, 5))

for scenario in scenarios:
    state = states[scenario.label]
    shock_radius = state.radius.to_value(u.cm)
    sampled_density = scenario.rho_csm(shock_radius, 0.0)

    ax.loglog(
        t_days,
        sampled_density,
        color=scenario.color,
        lw=2,
        label=scenario.label,
    )

ax.set_xlabel("Time (days)")
ax.set_ylabel(r"CSM Density at Shock ($\mathrm{g\,cm^{-3}}$)")
ax.set_ylim([1e-25, 1e-14])
ax.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.show()


# %%
# Summary
# -------
#
# This gallery demonstrates the qualitative behavior expected from launching
# the same supernova ejecta into every CSM profile available in
# :mod:`~trilobite.dynamics.profiles`. The most informative diagnostics
# are:
#
# - **Shock velocity vs. time**: highlights when and how rapidly deceleration
#   occurs.
# - **Shock velocity vs. radius**: ties the dynamics directly to CSM features
#   at specific radii.
# - **Density sampled by the shock**: shows which part of the CSM profile is
#   dynamically active at each time.
#
# This example is therefore a useful sanity check before fitting real
# transients: if the model cannot reproduce the qualitative shock behavior
# expected from a given CSM structure, the chosen profile or parameter range
# is probably not appropriate.
#
# .. seealso::
#
#     :ref:`numerical_shocks_overview` — API reference and worked examples for
#     the numerical shock engines, including the
#     :ref:`mechanical engine <mechanical_shock_engine>` which additionally
#     tracks separate forward and reverse shock temperatures.
#
#     :ref:`shock_engines` — choosing between self-similar and numerical
#     shock engines for a given modeling problem.
#
# sphinx_gallery_thumbnail_number = -1
