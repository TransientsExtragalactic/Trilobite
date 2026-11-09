r"""
Synchrotron SEDs from an Inhomogeneous Sphere
==============================================

.. currentmodule:: trilobite.radiation.synchrotron.SEDs.numerical.inhomogeneous

Radio-bright transients — supernovae interacting with a dense circumstellar
medium, GRB afterglows, and tidal disruption event outflows — are often
modelled as expanding spherical blast waves.  In the simplest picture the
entire emitting volume is assigned a single uniform magnetic field and a
single electron density; in reality both decline outward as the shocked shell
expands.

:class:`InhomogeneousSphereSynchrotronEngine` captures this radial structure
by decomposing the source into :math:`n_R` concentric spherical shells, each
with its own field :math:`B(r)` and electron distribution :math:`N(\gamma, r)`.
The specific intensity at impact parameter :math:`b_j = r_j` is built by a
two-pass shell-by-shell radiative transfer:

.. math::

    I_{\nu,k+1} = I_{\nu,k}\,e^{-\tau_k}
                  + S_{\nu,k}\!\left(1 - e^{-\tau_k}\right),

where :math:`\tau_k = \alpha_{\nu,k}\,\ell_k` and :math:`\ell_k` is the
half-path of the ray through shell :math:`k` (computed purely from the grid
geometry).  The total flux density is then

.. math::

    F_\nu \approx \sum_j \frac{2\pi b_j\,\Delta b_j}{D_A^2}\,I_\nu(b_j).

In this example we:

1. Build a power-law magnetic field :math:`B(r) = B_0(r/r_0)^{-\alpha_B}`
   with several values of the slope index :math:`\alpha_B`.
2. Assign each shell an equipartition power-law electron distribution whose
   *shape* (:math:`p`, :math:`\gamma_{\min}`, :math:`\gamma_{\max}`) is
   identical at every radius but whose *amplitude* :math:`K(r)` scales
   consistently with the local magnetic energy density
   :math:`u_B = B^2/8\pi`.
3. Decompose the SED into per-ray contributions to see which impact
   parameters dominate.
4. Compare the total SEDs produced by different :math:`\alpha_B` values.
"""
# sphinx_gallery_thumbnail_number = 3

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from matplotlib.colors import LogNorm

from trilobite.radiation.synchrotron import InhomogeneousSphereSynchrotronEngine
from trilobite.radiation.synchrotron.electron_distributions import PowerLaw
from trilobite.utils.plot_utils import set_plot_style

# %%
# Engine Setup
# ------------
# :class:`InhomogeneousSphereSynchrotronEngine` must pre-tabulate the
# pitch-angle-averaged synchrotron kernel before use.  Unlike the cylinder
# engine, the sphere enforces :math:`\beta = 0` (non-relativistic) and
# uses only the pitch-angle-averaged kernel, appropriate for an isotropically
# expanding shell.

engine = InhomogeneousSphereSynchrotronEngine()
engine.load_avg_first_kernel()

# %%
# Radial and Lorentz-Factor Grids
# --------------------------------
# We model a radio supernova shell extending from
# :math:`r_{\min} = 10^{16}\ \mathrm{cm}` to
# :math:`r_{\max} = 10^{17}\ \mathrm{cm}`.  A logarithmically-spaced grid
# ensures that the geometric half-path lengths computed internally by the
# engine are resolved equally in log-radius.

r_min, r_max = 1e16, 1e17
n_r, n_gamma = 60, 150

radii = np.geomspace(r_min, r_max, n_r) * u.cm
gammas = np.geomspace(1.0, 1e8, n_gamma)

# %%
# Magnetic Field Profiles
# -----------------------
# We adopt the family of power-law profiles
#
# .. math::
#
#     B(r) = B_0 \left(\frac{r}{r_0}\right)^{-\alpha_B},
#
# with :math:`B_0 = 0.3\ \mathrm{G}` at :math:`r_0 = r_{\min}`.
# The slope index :math:`\alpha_B` controls how steeply the field declines
# outward:
#
# - :math:`\alpha_B = 0.5`: very shallow decline — the sphere is nearly
#   homogeneous.
# - :math:`\alpha_B = 1.0`: typical of a toroidal field advected by a
#   uniformly expanding shell.
# - :math:`\alpha_B = 1.5`: intermediate compression with field gradient.
# - :math:`\alpha_B = 2.0`: steep decline; emission strongly concentrated
#   near the inner boundary.

B0 = 0.3 * u.G
r0 = r_min * u.cm

alpha_B_values = [0.5, 1.0, 1.5, 2.0]
colors = ["C0", "C1", "C2", "C3"]

set_plot_style()

fig, ax = plt.subplots(figsize=(7, 4))

B_profiles = {}
for alpha_B, color in zip(alpha_B_values, colors):
    B = B0 * (radii / r0) ** (-alpha_B)
    B_profiles[alpha_B] = B
    ax.loglog(
        radii.to_value(u.cm),
        B.to_value(u.G),
        color=color,
        lw=2,
        label=rf"$\alpha_B = {alpha_B}$",
    )

ax.set_xlabel(r"$r\ [\mathrm{cm}]$")
ax.set_ylabel(r"$B\ [\mathrm{G}]$")
ax.set_title(r"Magnetic Field Profiles $B(r) = B_0\,(r/r_0)^{-\alpha_B}$")
ax.legend()
ax.grid(True, which="both", ls="--", alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Consistent Electron Distributions
# ----------------------------------
# At each shell we assign an equipartition power-law distribution
#
# .. math::
#
#     N(\gamma, r) = K(r)\,\gamma^{-p},
#     \qquad \gamma_{\min} \le \gamma \le \gamma_{\max}.
#
# The *shape* of :math:`N(\gamma,r)` is the same everywhere: the spectral
# index :math:`p = 3` and the Lorentz-factor support :math:`[1, 10^8]` are
# held fixed across all shells.  Only the amplitude :math:`K(r)` varies,
# determined by the equipartition condition
#
# .. math::
#
#     u_e = \varepsilon_e\,u_B = \varepsilon_e\,\frac{B(r)^2}{8\pi},
#
# so that :math:`K(r) \propto B(r)^2` and inner shells (stronger field) host
# a proportionally denser electron population.
# :meth:`~trilobite.radiation.synchrotron.electron_distributions.ElectronDistribution.normalize_from_magnetic_field`
# handles this calculation for an array of :math:`B` values directly.

p = 3
gamma_min, gamma_max = 1.0, 1e8
epsilon_e, epsilon_B = 0.1, 0.1

GAMMA, _ = np.meshgrid(gammas, radii)

N_distributions = {}
for alpha_B in alpha_B_values:
    B = B_profiles[alpha_B]
    K = PowerLaw.normalize_from_magnetic_field(
        B,
        epsilon_B=epsilon_B,
        epsilon_E=epsilon_e,
        p=p,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
    ).cgs.value
    N_distributions[alpha_B] = K[:, np.newaxis] * GAMMA ** (-p)

# %%
# Per-Ray Decomposition
# ----------------------
# :meth:`~InhomogeneousSphereSynchrotronEngine.compute_ray_specific_intensity`
# returns the area-weighted flux density contribution
# :math:`dF_{\nu,j} = (2\pi b_j\,\Delta b_j / D_A^2)\,I_\nu(b_j)` for each
# impact parameter :math:`b_j`.  Here we examine the fiducial case
# :math:`\alpha_B = 1`.
#
# Unlike the cylinder engine, the sphere needs no explicit slab depth: the
# radiative-transfer path through each shell is determined entirely from the
# radial grid geometry.

freq = np.geomspace(1e7, 1e14, 400) * u.Hz
distance = 60 * u.Mpc
alpha_B_fid = 1.0

ray_flux = engine.compute_ray_specific_intensity(
    freq,
    radii,
    B_profiles[alpha_B_fid],
    N_distributions[alpha_B_fid],
    gamma=gammas,
    luminosity_distance=distance,
).to("mJy")

total_fid = ray_flux.sum(axis=-1)

set_plot_style()

fig, ax = plt.subplots(figsize=(9, 5.5))

freq_ghz = freq.to_value(u.GHz)
cmap = plt.cm.plasma
norm = LogNorm(vmin=r_min, vmax=r_max)

for j in range(n_r):
    ax.loglog(
        freq_ghz,
        ray_flux[:, j].to_value("mJy"),
        color=cmap(norm(radii[j].to_value(u.cm))),
        lw=0.8,
        alpha=0.6,
    )

ax.loglog(freq_ghz, total_fid.to_value("mJy"), color="k", lw=2.5, label=r"Total $F_\nu$")

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=ax, label=r"$b\ [\mathrm{cm}]$")

ax.set_xlabel(r"$\nu\ [\mathrm{GHz}]$")
ax.set_ylabel(r"$F_\nu\ [\mathrm{mJy}]$")
ax.set_title(rf"Ray Decomposition — Inhomogeneous Sphere ($\alpha_B = {alpha_B_fid}$)")
ax.legend()
ax.grid(True, which="both", ls="--", alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Rays at small impact parameters (purple) probe only the bright inner shells
# and peak at higher frequencies because :math:`\nu_c \propto B\gamma^2` with
# :math:`B` large near the centre.  Rays at large impact parameters (yellow)
# pass through the weaker outer shells and peak at lower frequencies.
# The superposition broadens and flattens the total spectrum relative to any
# single-shell prediction.

# %%
# SED Comparison: Varying :math:`\alpha_B`
# -----------------------------------------
# We now compute total SEDs for all four slope indices.  For each case the
# same central field :math:`B_0` and equipartition fractions are used, so the
# normalizations are comparable.  The only difference is how rapidly the field
# (and with it the electron density) falls off with radius.

set_plot_style()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax_sed = axes[0]
ax_ratio = axes[1]

seds = {}
for alpha_B, color in zip(alpha_B_values, colors):
    F_nu = engine.compute_flux_density(
        freq,
        radii,
        B_profiles[alpha_B],
        N_distributions[alpha_B],
        gamma=gammas,
        luminosity_distance=distance,
    ).to("mJy")
    seds[alpha_B] = F_nu
    ax_sed.loglog(
        freq_ghz,
        F_nu.to_value("mJy"),
        color=color,
        lw=2,
        label=rf"$\alpha_B = {alpha_B}$",
    )

ax_sed.set_xlabel(r"$\nu\ [\mathrm{GHz}]$")
ax_sed.set_ylabel(r"$F_\nu\ [\mathrm{mJy}]$")
ax_sed.set_title("Total SED — Inhomogeneous Sphere")
ax_sed.legend()
ax_sed.grid(True, which="both", ls="--", alpha=0.3)

# Normalise each SED to its peak so spectral shapes can be compared directly.
for alpha_B, color in zip(alpha_B_values, colors):
    F = seds[alpha_B].to_value("mJy")
    ax_ratio.semilogx(
        freq_ghz,
        F / np.nanmax(F),
        color=color,
        lw=2,
        label=rf"$\alpha_B = {alpha_B}$",
    )

ax_ratio.set_xlabel(r"$\nu\ [\mathrm{GHz}]$")
ax_ratio.set_ylabel(r"$F_\nu\,/\,F_{\nu,\mathrm{peak}}$")
ax_ratio.set_title("Peak-Normalised Spectra")
ax_ratio.legend()
ax_ratio.grid(True, which="both", ls="--", alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Steeper field gradients (larger :math:`\alpha_B`) concentrate both the
# magnetic energy and the electron density near the inner boundary.  The
# resulting SED is dominated by these compact, high-:math:`B` shells, which
# push the SSA peak to higher frequencies and increase the peak flux density.
# Conversely, a shallow gradient (small :math:`\alpha_B`) distributes emission
# more evenly across the sphere, producing a lower but broader SED that more
# closely resembles a uniform single-zone model.
#
# This sensitivity to :math:`\alpha_B` means that multi-frequency radio
# observations of an expanding blast wave can, in principle, constrain the
# radial structure of the post-shock magnetic field independently of the total
# field energy — a diagnostic inaccessible to homogeneous one-zone models.
