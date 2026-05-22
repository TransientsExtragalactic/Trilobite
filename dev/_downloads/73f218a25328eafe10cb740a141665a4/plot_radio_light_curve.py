"""
Radio Light Curve Container
============================

A :class:`~trilobite.data.light_curve.RadioLightCurveContainer` represents
a time series of flux density measurements taken at a **single fixed
observing frequency**.  This is the natural representation for a single-band
VLA or ATCA monitoring campaign, and the direct input to single-frequency
light-curve models.
"""

# %%
# Building the Container
# -----------------------
#
# Unlike :class:`~trilobite.data.photometry.RadioPhotometryContainer`, the
# light-curve container stores the frequency as **metadata** rather than a
# per-row column, reflecting the assumption that all measurements share the
# same band.
#
# The table schema requires ``time``, ``flux_density``,
# ``flux_density_error``, and ``flux_upper_limit``.

import numpy as np
from astropy import units as u
from astropy.table import Table

from trilobite.data import RadioLightCurveContainer

rng = np.random.default_rng(7)

n = 25
time = np.sort(rng.uniform(5, 500, n)) * u.day

# Power-law rise then decline, typical of a radio supernova.
t_peak = 80.0
flux_val = (
    5e-3
    * np.where(
        time.value < t_peak,
        (time.value / t_peak) ** 1.2,
        (time.value / t_peak) ** -0.9,
    )
    * rng.lognormal(0, 0.1, n)
)
err_val = flux_val * rng.uniform(0.07, 0.20, n)

# The last three observations are upper limits.
ul_val = np.full(n, np.nan)
ul_val[-3:] = rng.uniform(1e-3, 2e-3, 3)
flux_val[-3:] = np.nan
err_val[-3:] = np.nan

table = Table(
    {
        "time": time,
        "flux_density": flux_val * u.Jy,
        "flux_density_error": err_val * u.Jy,
        "flux_upper_limit": ul_val * u.Jy,
    }
)

lc = RadioLightCurveContainer(table, frequency=8.5 * u.GHz)

print(f"Frequency    : {lc.frequency}")
print(f"Observations : {lc.n_obs}")
print(f"Detections   : {lc.n_detections}")
print(f"Upper limits : {lc.n_non_detections}")

# %%
# Plotting
# ---------
#
# The container gives direct access to ``time``, ``flux_density``, and the
# detection mask, making it straightforward to build a publication-quality
# light curve.

import matplotlib.pyplot as plt

from trilobite.utils.plot_utils import set_plot_style

set_plot_style()

det = lc.detection_mask
ul = lc.non_detection_mask

fig, ax = plt.subplots(figsize=(8, 4))
ax.errorbar(
    lc.time[det].to_value(u.day),
    lc.flux_density[det].to_value(u.mJy),
    yerr=lc.flux_density_error[det].to_value(u.mJy),
    fmt="o",
    color="steelblue",
    label="Detection",
)
ax.errorbar(
    lc.time[ul].to_value(u.day),
    lc.flux_upper_limit[ul].to_value(u.mJy),
    fmt="v",
    color="gray",
    alpha=0.7,
    label="Upper limit",
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Time since explosion (days)")
ax.set_ylabel(r"$F_\nu$ (mJy)")
ax.set_title(f"Radio light curve at {lc.frequency}")
ax.legend()
plt.tight_layout()
plt.show()
