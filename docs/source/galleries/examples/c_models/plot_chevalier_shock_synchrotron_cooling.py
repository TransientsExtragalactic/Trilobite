r"""
=================================================
Synchrotron Emission from Fast-Cooling SN Shocks
=================================================

In realistic supernova environments, synchrotron emission arises from shocks
propagating through dense circumstellar media (CSM), where both **radiative
cooling** of relativistic electrons and **synchrotron self-absorption (SSA)**
can play an essential role in shaping the observed spectrum.

This example demonstrates a complete **forward-modeling pipeline** for
synchrotron spectral energy distributions (SEDs) produced by a supernova shock
expanding into a wind-like CSM, following the classical self-similar interaction
framework introduced by Chevalier (see e.g. :footcite:t:`chevalierSelfsimilarSolutionsInteraction1982` and
:footcite:t:`ChevalierXRayRadioEmission1982`)

Unlike purely phenomenological demonstrations of spectral shapes, this example
connects **shock dynamics**, **microphysical assumptions**, and **radiative
processes** into a single, self-consistent workflow.

.. hint::

    For a detailed discussion of the theoretical background and its numerical
    implementation in Triceratops, see :ref:`synchrotron_theory` and
    :ref:`synch_sed_theory`.

Overview
--------

In this example, we're going to showcase the forward modeling power of the TRICERATOPS
physics backend by combining shock physics with synchrotron emission processes
to produce a family of evolving SEDs for a supernova shock interacting with a wind-like CSM.

We combine the following physical ingredients:

1. **Shock dynamics**
   using a Chevalier self-similar solution for a supernova ejecta profile
   interacting with a wind-stratified CSM.

2. **Microphysical closure**
   through standard energy-partition parameters
   :math:`(\epsilon_e, \epsilon_B)`, an electron power-law index :math:`p`,
   and a minimum Lorentz factor :math:`\gamma_{\min}`.

3. **Synchrotron cooling**
   to compute the cooling Lorentz factor :math:`\gamma_c` and the associated
   break frequency :math:`\nu_c`.

4. **Synchrotron self-absorption (SSA)**
   to compute the self-absorption frequency :math:`\nu_a` self-consistently
   from the emitting region geometry and normalization.

These ingredients are then combined to construct the **full synchrotron SED**
at a sequence of epochs using
:class:`~radiation.synchrotron.SEDs.PowerLaw_Cooling_SSA_SynchrotronSED`.

The Model
---------

The modeling pipeline implemented in this example follows a clear separation of
responsibilities: **dynamics**, **microphysics**, and **radiative processes**.
Each component is implemented as a modular engine within Triceratops, allowing
the full synchrotron SED to be constructed in a transparent and physically
interpretable way.

Dynamics
^^^^^^^^

The shock dynamics are modeled using a Chevalier self-similar solution for the
interaction of supernova ejecta with a wind-stratified circumstellar medium.
This framework assumes:

- Power-law ejecta density profiles,
- A steady wind CSM with :math:`\rho_{\rm w} \propto r^{-2}`,
- A strong, non-relativistic forward shock.

In Triceratops, this evolution is handled by
:class:`~dynamics.supernovae.shock_dynamics.ChevalierSelfSimilarWindShockEngine`,
which provides the forward-shock radius and velocity as functions of time. These
quantities set the geometric scale of the emitting region and determine the
post-shock energy density that feeds directly into the microphysical and
radiative calculations.

Given the shock velocity and upstream density, we compute the downstream
magnetic field strength assuming a strong, cold shock and a fixed magnetic
energy fraction :math:`\epsilon_B`, using
:func:`~dynamics.rankine_hugoniot.compute_strong_cold_shock_magnetic_field`.

.. hint::

    Other shock engines could be easily substituted here, including the numerical
    shock engines for generic density profiles or even relativistic shocks, depending on the physical scenario of
    interest. See :mod:`dynamics` for an API description. User-Guide discussion of dynamics can be
    found in :ref:`shock_overview`.

Microphysics
^^^^^^^^^^^^

The non-thermal electron population is described by a power-law distribution

.. math::

    \frac{dN}{d\gamma} \propto \gamma^{-p},
    \qquad \gamma_{\min} \le \gamma \le \gamma_{\max},

with energy injected into relativistic electrons and magnetic fields according
to fixed fractions :math:`\epsilon_e` and :math:`\epsilon_B` of the post-shock
internal energy density.

In this example we treat the following as *microphysical closure parameters*:

- :math:`\epsilon_e`: electron energy fraction,
- :math:`\epsilon_B`: magnetic energy fraction,
- :math:`p`: electron power-law index,
- :math:`\gamma_{\min}` and :math:`\gamma_{\max}`: Lorentz-factor bounds of the
  accelerated population.

These parameters determine both the *shape* and *normalization* of the optically
thin synchrotron spectrum once the magnetic field strength and emitting geometry
are specified. For background on the physical interpretation and typical ranges
of these parameters, see :ref:`synchrotron_theory`.

Radiative processes
^^^^^^^^^^^^^^^^^^^

Two radiative effects are essential in dense CSM environments and are included
self-consistently in this pipeline:

**Synchrotron cooling**
    Radiative losses steepen the electron distribution above a cooling Lorentz
    factor :math:`\gamma_c`, introducing a corresponding cooling break frequency
    :math:`\nu_c`. We compute :math:`\gamma_c(t)` using
    :class:`~radiation.synchrotron.cooling.SynchrotronRadiativeCoolingEngine`,
    given the time-dependent magnetic field and shock age.

    .. hint::

        One could easily use a different cooling engine (e.g. including inverse Compton losses)
        from :mod:`~radiation.synchrotron.cooling`
        or even a custom cooling prescription by implementing a new engine.

**Synchrotron self-absorption (SSA)**
    At low frequencies, the emitting region becomes optically thick and the
    spectrum turns over near :math:`\nu_a`, defined implicitly by
    :math:`\tau_\nu \sim 1`. In Triceratops, :math:`\nu_a` is computed inside the
    SED implementation by enforcing consistency between the optically thick
    (Rayleigh--Jeans) limit and the optically thin synchrotron spectrum, given
    the emitting geometry (solid angle) and normalization.

Once the dynamical evolution and microphysical closure are specified, the full
set of break frequencies :math:`(\nu_m, \nu_c, \nu_a)` and the flux normalization
are computed and passed to
:class:`~radiation.synchrotron.SEDs.PowerLaw_Cooling_SSA_SynchrotronSED`
to generate a time series of evolving spectra.

For this example, we neglect free-free absorption, which can be important in very dense CSM environments at low
frequencies. Triceratops does include a free-free absorption engine that can be easily integrated into this pipeline if
desired. See :mod:`~radiation.free_free`.

Goals of This Example
---------------------

The aim is to provide a reproducible, end-to-end modeling pipeline that produces:

- Shock radius and velocity evolution,
- Magnetic field evolution behind the shock,
- Time-dependent break frequencies
  :math:`(\nu_m, \nu_c, \nu_a)`,
- A family of evolving synchrotron SEDs across multiple epochs.

This mirrors the structure of practical radio and millimeter modeling of
supernovae, where observed spectra are interpreted through physically motivated
shock models rather than ad hoc spectral fits.
"""

# %%
# Setup
# -----
# We start by importing the necessary libraries and setting up the shock dynamics and synchrotron SED engines.
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.constants import c, k_B, m_p

from triceratops.dynamics.supernovae import ChevalierSelfSimilarWindShockEngine
from triceratops.radiation.synchrotron import PowerLaw_Cooling_SSA_SynchrotronSED
from triceratops.utils.plot_utils import set_plot_style

# Set the plot style to the standard for TRICERATOPS.
set_plot_style()

# Generate the shock engine and the SED engine.
shock_engine = ChevalierSelfSimilarWindShockEngine()
sed = PowerLaw_Cooling_SSA_SynchrotronSED()

# %%
# Parameter selection
# -------------------
#
# We now specify the physical parameters that define the supernova shock, the
# circumstellar medium, and the synchrotron-emitting electron population. These
# parameters serve as inputs to the shock dynamics engine and the synchrotron
# SED model and are chosen to be broadly representative of a core-collapse
# supernova interacting with a wind-like CSM.
#
# Shock and CSM parameters
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
# The Chevalier self-similar wind solution is parameterized by the properties of
# the supernova ejecta and the surrounding circumstellar medium:
#
# - :math:`E_{\rm ej}`
#   Total kinetic energy of the ejecta.
#
# - :math:`M_{\rm ej}`
#   Total ejecta mass.
#
# - :math:`n`
#   Power-law index of the outer ejecta density profile,
#   :math:`\rho_{\rm ej} \propto r^{-n}`.
#
# - :math:`\dot{M}` and :math:`v_{\rm w}`
#   Mass-loss rate and wind velocity of the progenitor star, which together
#   define a wind-stratified CSM with
#   :math:`\rho_{\rm w}(r) = \dot{M} / (4\pi r^2 v_{\rm w})`.
#
# In this example we adopt parameters characteristic of a red supergiant–like
# wind. Much faster winds (e.g. Wolf–Rayet progenitors) can be modeled by
# adjusting :math:`v_{\rm w}` accordingly.
#
# Microphysical parameters
# ^^^^^^^^^^^^^^^^^^^^^^^^
#
# The synchrotron-emitting electron population is specified using standard
# microphysical closure parameters:
#
# - :math:`\epsilon_e`
#   Fraction of the post-shock internal energy placed into relativistic electrons.
#
# - :math:`\epsilon_B`
#   Fraction of the post-shock internal energy placed into magnetic fields.
#
# - :math:`p`
#   Power-law index of the injected electron energy distribution.
#
# - :math:`\gamma_{\min}` and :math:`\gamma_{\max}`
#   Minimum and maximum electron Lorentz factors.
#
# These parameters control both the normalization and the shape of the optically
# thin synchrotron spectrum and enter directly into the computation of the break
# frequencies and flux normalization.
#
# Geometry and distance
# ^^^^^^^^^^^^^^^^^^^^^
#
# To relate the intrinsic synchrotron emissivity to an observed flux density, we
# must specify the emitting geometry and the source distance:
#
# - :math:`f_V`
#   Fraction of the spherical post-shock volume that contributes to synchrotron
#   emission (e.g. a thin emitting shell).
#
# - :math:`f_A`
#   Fraction of the projected shock area that is radiating.
#
# - :math:`D_L`
#   Luminosity distance to the source, used to convert intrinsic luminosities
#   into observed flux densities.
#
# Time sampling
# ^^^^^^^^^^^^^
#
# Finally, we choose a set of epochs spanning :math:`1`–:math:`10^3` days after
# explosion, logarithmically spaced to capture the rapid early-time evolution
# as well as the slower late-time behavior.

# --------------------------------------------------
# Shock and circumstellar medium parameters
# --------------------------------------------------
# Canonical core-collapse SN ejecta interacting with
# a wind-stratified CSM (rho ∝ r^-2).
shock_params = {
    "E_ej": 1.0e51 * u.erg,  # Ejecta kinetic energy
    "M_ej": 2.0 * u.Msun,  # Ejecta mass
    "n": 10,  # Outer ejecta density index
    "M_dot": 1.0e-5 * u.Msun / u.yr,  # Progenitor mass-loss rate
    "v_wind": 10.0 * u.km / u.s,  # Wind speed (RSG-like; WR winds are ~10^3 km/s)
}


# --------------------------------------------------
# Microphysical parameters
# --------------------------------------------------
# Energy partition and electron distribution properties.
epsilon_e = 0.3  # Fraction of post-shock energy in electrons
epsilon_B = 0.3  # Fraction of post-shock energy in magnetic fields
p = 3.0  # Electron power-law index
mu = 0.61  # Mean molecular weight (ionized plasma)
gamma_min = 1.0  # Minimum electron Lorentz factor
gamma_max = 1.0e9  # Maximum electron Lorentz factor


# --------------------------------------------------
# Emitting geometry and distance
# --------------------------------------------------
# Thin-shell approximation with partial volume filling.
f_V = 0.5  # Fraction of spherical volume emitting
f_A = 1.0  # Fraction of projected area emitting
D_L = 35.0 * u.Mpc  # Luminosity distance to the source

# --------------------------------------------------
# Temporal sampling
# --------------------------------------------------
# Logarithmically spaced epochs to capture early and late evolution.
times = np.geomspace(1.0, 1.0e3, 50) * u.day

# Add a smoothing parameter to control the sharpness of spectral breaks in the SED.
smoothing = 1


# %%
# Shock Dynamics and Post-Shock Conditions
# ----------------------------------------
#
# The synchrotron emission in this model is powered by the forward shock driven
# into the circumstellar medium by the expanding supernova ejecta. The properties
# of this shock — its radius, velocity, post-shock temperature, and magnetic field —
# set the fundamental physical scales that control particle acceleration and
# radiative emission.
#
# We compute the shock evolution using a Chevalier self-similar solution for
# ejecta expanding into a wind-stratified circumstellar medium
# (:math:`\rho_{\rm w} \propto r^{-2}`). This solution provides the forward-shock
# radius :math:`R_{\rm sh}(t)` and velocity :math:`v_{\rm sh}(t)` as explicit
# functions of time.
#
# From these dynamical quantities, we derive key post-shock conditions:
#
# - The **post-shock temperature**, assuming a strong, non-relativistic shock,
#   which sets the thermal energy scale of the shocked plasma.
# - The **downstream magnetic field strength**, computed by assuming that a
#   fixed fraction :math:`\epsilon_B` of the post-shock energy density resides
#   in magnetic fields.
#
# These quantities are tightly coupled: the shock velocity controls both the
# post-shock temperature and the magnetic field strength, while the shock radius
# determines the emitting volume and projected area used in the radiative
# normalization.
#
# To make these relationships explicit, we visualize the shock evolution using
# four diagnostic panels showing the time dependence of:
#
# - Forward-shock radius,
# - Forward-shock velocity,
# - Post-shock temperature,
# - Downstream magnetic field strength.
from triceratops.dynamics.rankine_hugoniot import (
    compute_strong_cold_shock_magnetic_field,
    compute_strong_cold_shock_temperature,
)

# --- Compute the shock evolution ---
shock_outputs = shock_engine.compute_shock_properties(
    times,
    **shock_params,
)

r_sh = shock_outputs["radius"].to(u.cm)
v_sh = shock_outputs["velocity"].to(u.cm / u.s)

# Compute the upstream density at the shock radius (wind profile).
rho_up = (shock_params["M_dot"] / (4 * np.pi * r_sh**2 * shock_params["v_wind"])).to(u.g / u.cm**3)

# Calculate the temperature.
T_sh = compute_strong_cold_shock_temperature(
    shock_velocity=v_sh,
    mu=mu,
)

# Calculate the magnetic field.
B = compute_strong_cold_shock_magnetic_field(
    shock_velocity=v_sh,
    upstream_density=rho_up,
    epsilon_B=epsilon_B,
).to(u.G)

# --------------------------------------------------
# Diagnostic plots
# --------------------------------------------------
fig, axes = plt.subplots(
    2,
    2,
    figsize=(10, 8),
    sharex=True,
)

# Shock radius
axes[0, 0].loglog(times, r_sh)
axes[0, 0].set_ylabel(r"$R_{\rm sh}\;[\mathrm{cm}]$")
axes[0, 0].set_title("Shock Radius")
axes[0, 0].grid(True, which="both", ls="--", alpha=0.4)

# Shock velocity
axes[0, 1].loglog(times, (v_sh / c).to_value(u.dimensionless_unscaled))
axes[0, 1].set_ylabel(r"$v_{\rm sh}/c$")
axes[0, 1].set_title("Shock Velocity")
axes[0, 1].grid(True, which="both", ls="--", alpha=0.4)

# Post-shock temperature
axes[1, 0].loglog(times, T_sh)
axes[1, 0].set_ylabel(r"$T_{\rm sh}\;[\mathrm{K}]$")
axes[1, 0].set_xlabel("Time [days]")
axes[1, 0].set_title("Post-Shock Temperature")
axes[1, 0].grid(True, which="both", ls="--", alpha=0.4)

# Magnetic field
axes[1, 1].loglog(times, B)
axes[1, 1].set_ylabel(r"$B\;[\mathrm{G}]\;(\epsilon_B=0.3)$")
axes[1, 1].set_xlabel("Time [days]")
axes[1, 1].set_title("Downstream Magnetic Field")
axes[1, 1].grid(True, which="both", ls="--", alpha=0.4)

plt.tight_layout()
plt.show()

# %%
# Synchrotron Cooling and the Cooling Break
# -----------------------------------------
#
# Relativistic electrons accelerated at the shock lose energy through synchrotron
# radiation as they propagate downstream. These radiative losses modify the
# electron energy distribution and introduce an additional spectral break,
# commonly referred to as the **cooling break**.
#
# The characteristic Lorentz factor at which synchrotron losses become important
# is the *cooling Lorentz factor* :math:`\gamma_c`, defined implicitly by the
# condition that the synchrotron cooling time equals the age of the system:
#
# .. math::
#
#     t_{\rm cool}(\gamma_c) = t.
#
# Electrons with :math:`\gamma \gg \gamma_c` cool efficiently over the dynamical
# timescale, while those with :math:`\gamma \ll \gamma_c` retain their injected
# power-law distribution. This transition steepens the synchrotron spectrum above
# the corresponding **cooling break frequency** :math:`\nu_c`.
#
# In Triceratops, synchrotron cooling is handled by a dedicated cooling engine that
# integrates the synchrotron loss rate using the time-dependent magnetic field.
# Given the downstream magnetic field :math:`B(t)` and the system age, the engine
# computes :math:`\gamma_c(t)` self-consistently.
#
# For this example, we compute the cooling Lorentz factor using
# :class:`~triceratops.radiation.synchrotron.cooling.SynchrotronRadiativeCoolingEngine`
# and visualize its temporal evolution below.
#
# The behavior of :math:`\gamma_c(t)` provides immediate physical insight:
#
# - A **decreasing** :math:`\gamma_c` indicates increasingly efficient cooling as
#   the magnetic field strengthens or the shock slows.
# - If :math:`\gamma_c < \gamma_{\min}`, the system enters the **fast-cooling**
#   regime.
# - If :math:`\gamma_c > \gamma_{\min}`, the system remains in the
#   **slow-cooling** regime.
#
# This distinction directly controls the ordering of synchrotron break
# frequencies and determines which spectral branches are active in the final SED.

from triceratops.radiation.synchrotron.cooling import SynchrotronRadiativeCoolingEngine

# Instantiate the cooling engine and compute the cooling Lorentz factor.
cooling_engine = SynchrotronRadiativeCoolingEngine()
gamma_c = cooling_engine.compute_cooling_gamma(B=B, t=times)

# Plot the cooling Lorentz factor evolution.
fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(times, gamma_c, color="C2")
ax.set_xlabel("Time [days]")
ax.set_ylabel(r"Cooling Lorentz Factor $\gamma_c$")
ax.grid(True, which="both", ls="--", alpha=0.4)
plt.tight_layout()
plt.show()


# %%
# Break Frequencies and SED Normalization
# ---------------------------------------
#
# At this stage, all physical ingredients required to construct the synchrotron
# spectrum are available. Specifically, we have:
#
# - **Shock dynamics**, which determine the emitting geometry
#   (volume :math:`V` and solid angle :math:`\Omega`) and the post-shock energy
#   density;
# - **Microphysical closure parameters**
#   :math:`(\epsilon_e, \epsilon_B, p, \gamma_{\min}, \gamma_{\max})`, which specify
#   how the shock energy is partitioned into relativistic electrons and magnetic
#   fields;
# - **Radiative cooling**, which introduces a cooling Lorentz factor
#   :math:`\gamma_c(t)` and the associated break frequency :math:`\nu_c`.
#
# Given these inputs, the synchrotron SED model computes the complete set of
# phenomenological parameters that uniquely define the spectrum at each epoch:
#
# .. math::
#
#     \left\{
#         F_{\nu,\mathrm{norm}},
#         \nu_m,
#         \nu_c,
#         \nu_a,
#         \nu_{\max}
#     \right\},
#
# along with the electron index :math:`p` and smoothing parameter controlling the
# sharpness of spectral breaks.
#
# The mapping from physical parameters to SED parameters is performed by
# :meth:`~radiation.synchrotron.SEDs.PowerLaw_Cooling_SSA_SynchrotronSED.from_physics_to_params`,
# which implements an equipartition-based normalization under the assumptions of a
# single-zone, homogeneous emitting region.
#
# In particular, the synchrotron self-absorption frequency :math:`\nu_a` is
# computed **self-consistently** by enforcing equality between:
#
# - the optically thick (Rayleigh–Jeans) synchrotron spectrum, and
# - the optically thin synchrotron spectrum,
#
# at the frequency where the optical depth satisfies
#
# .. math::
#
#     \tau_\nu \sim 1.
#
# This procedure ensures that the low-frequency turnover is physically tied to
# the emitting geometry and normalization rather than introduced as an ad hoc
# break. A detailed discussion of this construction can be found in
# :ref:`synch_sed_theory`.
#
# We now evaluate these parameters as a function of time and track the evolution
# of the characteristic synchrotron frequencies :math:`(\nu_m, \nu_c, \nu_a)`.

# --------------------------------------------------
# Compute break frequencies as a function of time
# --------------------------------------------------

parameters = [
    sed.from_physics_to_params(
        B=B_i,
        R=r_i,
        luminosity_distance=D_L,
        epsilon_E=epsilon_e,
        epsilon_B=epsilon_B,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
        gamma_c=gamma_c_i,
        f_V=f_V,
        f_A=f_A,
        p=p,
        pitch_average=True,
    )
    for B_i, r_i, gamma_c_i in zip(B, r_sh, gamma_c)
]

# Extract break frequencies (unit-safe)
nu_m = u.Quantity([par["nu_m"] for par in parameters])
nu_c = u.Quantity([par["nu_c"] for par in parameters])
nu_a = u.Quantity([par["nu_a"] for par in parameters])

# --------------------------------------------------
# Plot break frequency evolution
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 4))

ax.loglog(times, nu_m, label=r"$\nu_m$")
ax.loglog(times, nu_c, label=r"$\nu_c$")
ax.loglog(times, nu_a, label=r"$\nu_a$")

ax.set_xlabel("Time [days]")
ax.set_ylabel("Frequency [Hz]")
ax.legend()
ax.grid(True, which="both", ls="--", alpha=0.4)

plt.tight_layout()
plt.show()

# %%
# Interpretation of the Break Frequencies
# ---------------------------------------
#
# The evolution of the characteristic synchrotron frequencies
# :math:`(\nu_m, \nu_c, \nu_a)` encapsulates the changing physical conditions
# behind the shock.
#
# - The injection frequency :math:`\nu_m` decreases with time as the shock
#   decelerates and the characteristic Lorentz factor of newly accelerated
#   electrons remains fixed while the magnetic field weakens.
#
# - The cooling frequency :math:`\nu_c` typically increases with time, reflecting
#   the declining synchrotron loss rate as the post-shock magnetic field decays.
#
# - The self-absorption frequency :math:`\nu_a` traces the competition between
#   decreasing density, expanding geometry, and the evolving normalization of the
#   optically thin spectrum. Its behavior is especially sensitive to the emitting
#   volume and projected area.
#
# The relative ordering of these frequencies determines the **global spectral
# regime** at each epoch and controls both the shape and normalization of the
# observed spectrum.
#
# With the time-dependent break frequencies in hand, we now turn to the full
# synchrotron spectral energy distributions and examine how the broadband spectrum
# evolves across epochs.

from matplotlib.colors import LogNorm

frequencies = np.logspace(7, 12, 1000) * u.Hz

fig, ax = plt.subplots(figsize=(8, 6))

# Log-scaled time normalization (in days)
t_days = times.to_value(u.day)
norm = LogNorm(vmin=t_days.min(), vmax=t_days.max())
cmap = plt.cm.viridis

for par, r_i, t_day in zip(parameters, r_sh, t_days):
    F_nu = sed.sed(
        frequencies,
        nu_m=par["nu_m"],
        nu_c=par["nu_c"],
        F_norm=par["F_norm"],
        gamma_m=gamma_min,
        omega=np.pi * (r_i / D_L) ** 2 * f_A,  # Projected solid angle of the emitting region
        p=p,
        s=-smoothing,  # effectively piecewise
    )

    ax.loglog(
        frequencies.to_value(u.GHz),
        F_nu.to_value(u.uJy),
        color=cmap(norm(t_day)),
        lw=1.2,
    )

# --------------------------------------------------
# Colorbar: explicitly shows (log) time
# --------------------------------------------------

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

cbar = plt.colorbar(sm, ax=ax)
cbar.set_label(r"Time since explosion [days]")

# --------------------------------------------------
# Plot cosmetics
# --------------------------------------------------
# sphinx_gallery_thumbnail_number = -1
ax.set_xlabel(r"Frequency [GHz]")
ax.set_ylabel(r"Flux Density [$\mu$Jy]")
ax.set_title("Synchrotron SED Evolution (Cooling + SSA)")
ax.grid(True, which="both", ls="--", alpha=0.4)
ax.set_ylim(1e-3, None)

plt.tight_layout()
plt.show()
