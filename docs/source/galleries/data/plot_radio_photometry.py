"""
Radio Photometry Container
==========================

This example walks through the core workflow for radio photometry in
Trilobite: constructing a
:class:`~trilobite.data.photometry.RadioPhotometryContainer`, inspecting
its contents, grouping observations into temporal epochs, and plotting
individual SEDs.
"""

# %%
# Building a Container
# --------------------
#
# The :class:`~trilobite.data.photometry.RadioPhotometryContainer` accepts
# an :class:`astropy.table.Table` whose columns follow the standardised
# schema (``time``, ``freq``, ``flux_density``, ``flux_density_error``,
# ``flux_upper_limit``).  Here we generate a synthetic dataset that mimics
# a multi-frequency radio campaign on a supernova — three VLA bands observed
# over ~400 days, with a handful of upper limits late in the campaign.

import numpy as np
from astropy import units as u
from astropy.table import Table

from trilobite.data import RadioPhotometryContainer

rng = np.random.default_rng(42)

# Three representative VLA frequencies.
freqs_ghz = [5.0, 8.5, 15.0]
n_per_band = 18

rows = {"time": [], "freq": [], "flux_density": [], "flux_density_error": [], "flux_upper_limit": []}

for nu in freqs_ghz:
    t = np.sort(rng.uniform(5, 400, n_per_band))
    # Simple power-law decay, dimmer at higher frequency.
    f = 3e-3 * (t / 20.0) ** -0.8 * (nu / 5.0) ** -0.7 * rng.lognormal(0, 0.12, n_per_band)
    err = f * rng.uniform(0.08, 0.18, n_per_band)

    # Mark the last two epochs of the highest-frequency band as upper limits.
    ul = np.full(n_per_band, np.nan)
    if nu == 15.0:
        f[-2:] = np.nan
        err[-2:] = np.nan
        ul[-2:] = rng.uniform(3e-4, 6e-4, 2)

    rows["time"].extend(t)
    rows["freq"].extend([nu] * n_per_band)
    rows["flux_density"].extend(f)
    rows["flux_density_error"].extend(err)
    rows["flux_upper_limit"].extend(ul)

table = Table(
    {
        "time": np.array(rows["time"]) * u.day,
        "freq": np.array(rows["freq"]) * u.GHz,
        "flux_density": np.array(rows["flux_density"]) * u.Jy,
        "flux_density_error": np.array(rows["flux_density_error"]) * u.Jy,
        "flux_upper_limit": np.array(rows["flux_upper_limit"]) * u.Jy,
    }
)

container = RadioPhotometryContainer(table)

print(f"Observations : {container.n_obs}")
print(f"Detections   : {container.n_detections}")
print(f"Upper limits : {container.n_non_detections}")
print(f"Freq range   : {container.freq.min():.1f} – {container.freq.max():.1f}")
print(f"Time range   : {container.time.min():.1f} – {container.time.max():.1f}")

# %%
# Plotting the Full Dataset
# -------------------------
#
# :meth:`~trilobite.data.photometry.RadioPhotometryContainer.plot_photometry`
# produces a light-curve view coloured by frequency, with upper limits
# rendered as downward arrows.

import matplotlib.pyplot as plt

from trilobite.utils.plot_utils import set_plot_style

set_plot_style()

fig, ax = plt.subplots(figsize=(9, 5))
container.plot_photometry(fig=fig, axes=ax, show_upper_limits=True)
ax.set_title("Synthetic radio photometry — all bands")
plt.tight_layout()
plt.show()

# %%
# Epoching
# --------
#
# Observations taken within a short window of each other can be grouped
# into epochs using
# :meth:`~trilobite.data.photometry.RadioPhotometryContainer.set_epochs_from_time_gaps`.
# Here a gap of 10 days separates distinct observing campaigns.

container.set_epochs_from_time_gaps(max_gap=10.0 * u.day)
print(f"Epochs identified: {container.n_epochs}")

# %%
# Plotting Individual Epochs
# --------------------------
#
# Once epochs are defined,
# :meth:`~trilobite.data.photometry.RadioPhotometryContainer.plot_epoch`
# plots the broadband SED at a single epoch.  Here we compare the first
# and last detected epochs side-by-side.

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, epoch_id, label in zip(axes, [0, container.n_epochs - 2], ["Early", "Late"]):
    container.plot_epoch(
        epoch_id=epoch_id,
        fig=fig,
        axes=ax,
        show_upper_limits=True,
        color=colors[epoch_id % len(colors)],
        label=f"Epoch {epoch_id}",
    )
    ax.set_title(f"{label} epoch (id={epoch_id})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()

fig.suptitle("Per-epoch radio SEDs")
plt.tight_layout()
plt.show()
