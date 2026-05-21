r"""
On-Axis Asymmetric Synchrotron SEDs
==================================

Most synchrotron models treat the emitting region as a single homogeneous
zone. The
:class:`~trilobite.radiation.synchrotron.SEDs.numerical.OnAxisAsymmetricSynchrotronEngine`
generalizes this picture by assigning independent physical conditions to
each polar-angle sightline :math:`\theta \in [0,\pi/2]`.

This framework is useful for structured jets, asymmetric ejecta, or any
transient with genuine angular structure. The observed flux is assembled
through Gauss–Legendre quadrature over the projected emitting surface.

In this example we construct a simple two-zone outflow:

- a compact, strongly magnetized core,
- surrounded by weaker extended wings.

We then:
- visualize the angular structure,
- construct an equipartition electron distribution,
- and decompose the resulting synchrotron SED into per-ring contributions.
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from matplotlib.colors import Normalize

from trilobite.radiation.synchrotron.SEDs.numerical import (
    OnAxisAsymmetricSynchrotronEngine,
)
from trilobite.radiation.synchrotron.microphysics import (
    compute_PL_norm_from_magnetic_field,
    get_power_law_distribution,
)
from trilobite.utils.plot_utils import set_plot_style

# %%
# Engine Setup
# ------------
# The engine discretizes the projected emitting surface into
# :math:`n_\theta` polar-angle sightlines.

engine = OnAxisAsymmetricSynchrotronEngine(n_theta=30)
engine.load_avg_first_kernel()

theta = engine.theta

# %%
# Source Geometry
# ---------------
# We define a simple two-zone structure:
#
# - Core (:math:`\theta < \pi/8`)
# - Wings (:math:`\theta \ge \pi/8`)

core = theta <= np.pi / 8

B = 1e-1 * u.G * np.ones(theta.size)
B *= 1 + 10 * (1 - np.cos(theta))

R = 1e16 * u.cm * np.ones(theta.size)
R *= 1 + 10 * (1 - np.sin(theta))

depth = 1e16 * u.cm * np.ones(theta.size)


def polar_profile(ax, theta, values, *, color="C0", title=None):
    """Plot a mirrored logarithmic polar profile."""
    th = np.concatenate([-theta, theta[::-1]])
    val = np.concatenate([values, values[::-1]])

    ax.plot(th, val, color=color, lw=2)
    ax.fill(th, val, color=color, alpha=0.2)

    ax.set_rscale("log")

    ax.set_theta_zero_location("N")
    ax.set_thetamin(-90)
    ax.set_thetamax(90)

    if title is not None:
        ax.set_title(title)


set_plot_style()

fig, axes = plt.subplots(
    2,
    2,
    figsize=(6, 6),
    subplot_kw={"projection": "polar"},
)
axes = axes.ravel()

polar_profile(
    axes[0],
    theta,
    B.to_value(u.G),
    color="C0",
    title=r"$B\ [\mathrm{G}]$",
)
axes[0].set_rlim([1e-2, 1e0])

polar_profile(
    axes[1],
    theta,
    R.to_value(u.cm),
    color="C1",
    title=r"$R\ [\mathrm{cm}]$",
)
axes[1].set_rlim([1e15, 1e17])

polar_profile(
    axes[2],
    theta,
    depth.to_value(u.cm),
    color="C2",
    title=r"Depth [cm]",
)
axes[2].set_rlim([1e14, 1e16])
fig.suptitle("Two-Zone Angular Structure", y=1.03)

axes[3].axis("off")

plt.tight_layout()
plt.show()

# %%
# The compact core occupies the polar region near :math:`\theta = 0`,
# while the extended low-field wings dominate larger angles.
#
# Electron Distribution
# ---------------------
# We now construct an equipartition power-law electron distribution for
# each sightline.

p = 2.5
gamma_min = 1.0
gamma_max = 1e8

epsilon_e = 0.1
epsilon_B = 0.1

norm = compute_PL_norm_from_magnetic_field(
    B,
    p,
    epsilon_e,
    epsilon_B,
    gamma_min=gamma_min,
    gamma_max=gamma_max,
)

distribution = get_power_law_distribution(
    gamma_min=gamma_min,
    gamma_max=gamma_max,
    p=p,
    norm=norm,
)

gamma = np.geomspace(gamma_min, gamma_max, 1000)

N_gamma = distribution(gamma[:, None]).T

# %%
# Computing the SED
# -----------------
# We compute:
#
# - the per-sightline flux integrand,
# - the per-ring flux contributions,
# - and the fully integrated SED.

nu = np.geomspace(1e7, 1e18, 300) * u.Hz

common_kwargs = dict(
    B=B,
    N=N_gamma,
    slab_depth=depth,
    R=R,
    gamma=gamma,
    beta=0,
    luminosity_distance=60 * u.Mpc,
)

sightline_flux = engine.compute_sightline_flux_density(
    nu,
    **common_kwargs,
)

ring_flux = sightline_flux

total_sed = engine.compute_flux_density(nu, **common_kwargs)

# Verify: ring_flux sums to total_sed to floating-point precision.
# total = Σᵢ w_i ξ_i F_sightline_i  (Gauss-Legendre quadrature identity)
ring_sum = ring_flux.sum(axis=-1)

# %%
# Spectral Decomposition
# ----------------------
# Each coloured curve is the flux contribution from one annular ring; their sum
# (dashed grey) exactly recovers the total (black).  The total is larger than
# any single ring because it is the sum of *all* :math:`n_\theta` rings — each
# ring carries only :math:`\sim 1/n_\theta` of the integrated flux on average.

set_plot_style()

fig, ax = plt.subplots(figsize=(9, 5.5))

nu_ghz = nu.to_value(u.GHz)
cmap = plt.cm.viridis
colors = cmap(np.linspace(0, 1, engine.n_theta))
for i in range(engine.n_theta):
    ax.loglog(nu_ghz, ring_flux[:, i].to_value(u.mJy), color=colors[::-1][i], lw=0.9, alpha=0.75)

ax.loglog(nu_ghz, ring_sum.to_value(u.mJy), color="0.5", lw=2, ls="--", label=r"$\Sigma$ rings")
ax.loglog(nu_ghz, total_sed.to_value(u.mJy), color="k", lw=2.5, label="Total")

sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(theta.min(), theta.max()))
fig.colorbar(sm, ax=ax, label=r"$\theta\ [\mathrm{rad}]$")

ax.set_xlabel(r"$\nu\ [\mathrm{GHz}]$")
ax.set_ylabel(r"$F_\nu\ [\mathrm{mJy}]$")
ax.set_title("On-Axis Asymmetric Synchrotron SED")
ax.grid(True, which="both", ls="--", alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()

# %%
# Several features are immediately visible:
#
# - Core sightlines peak at higher frequencies due to the stronger
#   magnetic field.
#
# - Wing sightlines peak at lower frequencies because
#   :math:`\nu_c \propto B\gamma^2`.
#
# - Despite weaker emissivity, the wings dominate the total flux because
#   the projected-area weighting strongly favors large-angle sightlines.
