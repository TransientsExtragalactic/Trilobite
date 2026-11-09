r"""
Synchrotron SEDs from an Inhomogeneous Cylinder
================================================

.. currentmodule:: trilobite.radiation.synchrotron.SEDs.numerical.inhomogeneous

Most synchrotron models (including most Trilobite models) treat the emitting
region as a single homogeneous zone.  Real astrophysical sources —
expanding blast waves in supernovae and GRBs, structured jets, and
tidally-disrupted stellar debris — have genuine radial structure: the
magnetic field strength, electron density, and SSA opacity all vary with
distance from the center.  These inhomogeneities can significantly shape
the observed SED.

Introducing inhomogeneities comes at the cost of increased complexity and
computational expense, and parameter inference for these physical scenarios
is often highly degenerate.  Nonetheless, it is instructive to explore the
SEDs produced by simple inhomogeneous models to build intuition for how
they differ from their homogeneous counterparts.  These models may also be
used to develop theoretical predictions for transients that do not conform
to the traditional single-zone paradigm.

One such model is :class:`InhomogeneousCylinderSynchrotronEngine`, which
decomposes the source into a sequence of concentric cylindrical annuli
observed edge-on, each with its own plasma properties.  The contribution
of each ring to the observed flux is weighted by its projected annular
area:

.. math::

    F_\nu \approx \sum_j \frac{2\pi r_j\,\Delta r_j}{D_A^2}\,I_\nu(r_j).

In this example we:

1. Construct a power-law magnetic field profile :math:`B(r) \propto r^{-1}`.
2. Build a per-ring equipartition power-law electron distribution.
3. Compute per-ring synchrotron SEDs and decompose the total spectrum into
   annular contributions coloured by radius.
"""
# sphinx_gallery_thumbnail_number = 2

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from matplotlib.colors import LogNorm

from trilobite.radiation.synchrotron import InhomogeneousCylinderSynchrotronEngine
from trilobite.radiation.synchrotron.electron_distributions import PowerLaw
from trilobite.utils.plot_utils import set_plot_style

# %%
# Engine Setup
# ------------
# To get started we instantiate :class:`InhomogeneousCylinderSynchrotronEngine`.
# Like all numerical synchrotron engines in Trilobite, it must pre-tabulate
# the synchrotron kernel before use (see :ref:`synch_numerical_sed_theory`).
# We load the pitch-angle-averaged kernel here, which is appropriate when
# the electron pitch-angle distribution is isotropic.

engine = InhomogeneousCylinderSynchrotronEngine()
engine.load_avg_first_kernel()

# %%
# Radial and Lorentz-Factor Grids
# --------------------------------
# We model a cylinder extending from :math:`r_{\min} = 10^{16}\ \mathrm{cm}`
# to :math:`r_{\max} = 10^{17}\ \mathrm{cm}` on 50 logarithmically-spaced
# annuli.  The Lorentz factor grid spans six decades.

r_min, r_max = 1e16, 1e17  # cm
N_radii, N_gamma = 50, 100

radii = np.geomspace(r_min, r_max, N_radii) * u.cm
gammas = np.geomspace(1, 1e8, N_gamma)

# %%
# Magnetic Field Profile
# ----------------------
# We adopt the power-law field profile
#
# .. math::
#
#     B(r) = B_0 \left(\frac{r}{R_B}\right)^{-\alpha_B},
#
# with :math:`\alpha_B = 1`, typical of a toroidal field frozen into a
# uniformly expanding shell.  Inner rings carry a stronger field and emit
# at higher characteristic frequencies :math:`\nu_c \propto B\gamma^2`.

B0 = 1.0 * u.G
R_B = 1e16 * u.cm
alpha_B = 1

B = B0 * (radii / R_B) ** (-alpha_B)

set_plot_style()

fig, ax = plt.subplots(figsize=(7, 4))

ax.loglog(radii.to_value(u.cm), B.to_value(u.G), color="C0", lw=2)
ax.set_xlabel(r"$r\ [\mathrm{cm}]$")
ax.set_ylabel(r"$B\ [\mathrm{G}]$")
ax.set_title(r"Magnetic Field Profile $B(r) \propto r^{-1}$")
ax.grid(True, which="both", ls="--", alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Electron Distribution
# ---------------------
# Each ring is assigned an equipartition power-law electron distribution.
# :meth:`~trilobite.radiation.synchrotron.electron_distributions.PowerLaw.normalize_from_magnetic_field`
# returns the normalisation :math:`K(r)` such that
#
# .. math::
#
#     N(\gamma, r) = K(r)\,\gamma^{-p},
#     \qquad \gamma_{\min} \le \gamma \le \gamma_{\max},
#
# where :math:`K(r)` is fixed by the equipartition condition
# :math:`\epsilon_e / \epsilon_B = u_e / u_B` with
# :math:`u_B = B^2 / 8\pi`.  Because :math:`u_B \propto B^2 \propto r^{-2}`,
# inner rings host a denser electron population.

p = 3
epsilon_e = 0.1
epsilon_B = 0.01

GAMMA, _ = np.meshgrid(gammas, radii)

K = PowerLaw.normalize_from_magnetic_field(
    B,
    epsilon_B=epsilon_B,
    epsilon_E=epsilon_e,
    p=p,
    gamma_min=1,
    gamma_max=1e8,
).cgs.value

N = K[:, np.newaxis] * GAMMA ** (-p)

# %%
# Per-Ring Flux Contributions
# ----------------------------
# :meth:`InhomogeneousCylinderSynchrotronEngine.compute_ring_specific_intensity`
# returns the area-weighted flux density :math:`dF_{\nu,j}` for each ring,
# with shape ``(n_freq, n_radii)``.  Summing over the last axis recovers the
# total :math:`F_\nu`.  We hold the line-of-sight slab depth fixed at
# :math:`\ell = 10^{16}\ \mathrm{cm}` for all rings.

freq = np.geomspace(1e-3, 1e5, 500) * u.GHz
distance = 60 * u.Mpc
slab_depth = 1e16 * u.cm

ring_flux = engine.compute_ring_specific_intensity(
    freq,
    radii,
    slab_depth,
    B,
    N,
    gamma=gammas,
    luminosity_distance=distance,
).to("Jy")

total_flux = ring_flux.sum(axis=-1)

# %%
# Spectral Decomposition
# ----------------------
# Each curve below is the flux contribution :math:`dF_{\nu,j}` from one
# annular ring, coloured by its radius using a logarithmic colormap.
# The black curve is the composite total
# :math:`F_\nu = \sum_j dF_{\nu,j}`.
#
# Because :math:`\nu_c \propto B \propto r^{-1}`, inner rings (purple) peak
# at higher frequencies while outer rings (yellow) peak at lower frequencies.
# Their superposition produces a broader, flatter composite spectrum than
# any single homogeneous zone could reproduce.

set_plot_style()

fig, ax = plt.subplots(figsize=(9, 5.5))

freq_ghz = freq.to_value(u.GHz)
cmap = plt.cm.viridis
norm = LogNorm(vmin=r_min, vmax=r_max)

for i in range(N_radii):
    ax.loglog(
        freq_ghz,
        ring_flux[:, i].to_value("Jy"),
        color=cmap(norm(radii[i].to_value(u.cm))),
        lw=0.9,
        alpha=0.7,
    )

ax.loglog(freq_ghz, total_flux.to_value("Jy"), color="k", lw=2.5, label=r"Total $F_\nu$")

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig.colorbar(sm, ax=ax, label=r"$r\ [\mathrm{cm}]$")

ax.set_xlabel(r"$\nu\ [\mathrm{GHz}]$")
ax.set_ylabel(r"$F_\nu\ [\mathrm{Jy}]$")
ax.set_title("Inhomogeneous Cylinder Synchrotron SED")
ax.legend()
ax.grid(True, which="both", ls="--", alpha=0.3)
ax.set_ylim([1e-6, 3e-2])
ax.set_xlim([1e-2, 1e2])

plt.tight_layout()
plt.show()

# %%
# The composite SED is noticeably broader and flatter than a single-zone
# prediction.  Each ring contributes a spectral hump shifted to the
# frequency set by its local :math:`B(r)`, so observations spanning a
# narrow frequency window may mistake this multi-ring superposition for a
# simple broken power-law spectrum.
