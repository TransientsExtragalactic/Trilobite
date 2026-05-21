"""
Optical Light Curve Container
==============================

This example demonstrates how to construct an
:class:`~triceratops.data.light_curve.OpticalLightCurveContainer` from
tabular data, access its dual flux/magnitude representations, separate
detections from upper limits, and plot the resulting light curve.
"""

# %%
# Setup
# -----
#
# We begin by constructing a synthetic g-band optical light curve that
# contains both detections and upper limits.

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.table import Table

from triceratops.data import OpticalLightCurveContainer

rng = np.random.default_rng(42)
n = 20
time = np.sort(rng.uniform(5, 300, n)) * u.day

# Detections: realistic faint transient in g-band
flux_det = rng.lognormal(-63, 0.3, n) * u.Unit("erg/(s cm2 Hz)")
err_det = flux_det * rng.uniform(0.05, 0.15, n)
ul = np.full(n, np.nan) * u.Unit("erg/(s cm2 Hz)")

# Mark the last 4 rows as upper limits
n_upper = 4
flux_det[n - n_upper :] = np.nan * u.Unit("erg/(s cm2 Hz)")
err_det[n - n_upper :] = np.nan * u.Unit("erg/(s cm2 Hz)")
ul[n - n_upper :] = rng.uniform(1e-30, 3e-30, n_upper) * u.Unit("erg/(s cm2 Hz)")

table = Table(
    {
        "time": time,
        "flux_density": flux_det,
        "flux_density_error": err_det,
        "flux_upper_limit": ul,
    }
)

lc = OpticalLightCurveContainer(table, band="g")

print(f"Band:           {lc.band}")
print(f"Observations:   {lc.n_obs}")
print(f"Detections:     {lc.n_detections}")
print(f"Upper limits:   {lc.n_non_detections}")

# %%
# Dual Flux / Magnitude Representation
# --------------------------------------
#
# Regardless of how the data were supplied, both flux and AB magnitude
# representations are always available. Here we supplied flux columns, but
# magnitude properties are computed on-the-fly.

det = lc.detection_mask
print("\nFlux density (detections, CGS):")
print(lc.flux[det][:3])

print("\nAB magnitude (detections, computed from flux):")
print(lc.mag[det][:3])

# %%
# Plotting
# ---------
#
# Plot the detections as error bars and the upper limits as downward arrows.

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle(f"Synthetic g-band Optical Light Curve  (band='{lc.band}')")

# --- Left panel: flux density ---
ax = axes[0]
ax.errorbar(
    lc.time[det].to(u.day).value,
    lc.flux[det].to(u.Unit("erg/(s cm2 Hz)")).value,
    yerr=lc.flux_error[det].to(u.Unit("erg/(s cm2 Hz)")).value,
    fmt="o",
    color="steelblue",
    label="Detection",
)

ax.errorbar(
    lc.time.to(u.day).value,
    lc.flux_upper_limit.to(u.Unit("erg/(s cm2 Hz)")).value,
    fmt="v",
    color="gray",
    label="Upper limit",
)
ax.set_yscale("log")
ax.set_xlabel("Time (days)")
ax.set_ylabel(r"$F_\nu$ (erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$)")
ax.legend()

# --- Right panel: AB magnitude ---
ax = axes[1]
ax.errorbar(
    lc.time[det].to(u.day).value,
    lc.mag[det],
    yerr=lc.mag_error[det],
    fmt="o",
    color="steelblue",
    label="Detection",
)
ax.errorbar(
    lc.time.to(u.day).value,
    lc.mag_upper_limit,
    fmt="^",
    color="gray",
    label="Upper limit (mag)",
)
ax.invert_yaxis()
ax.set_xlabel("Time (days)")
ax.set_ylabel(r"$m_\mathrm{AB}$ (mag)")
ax.legend()

plt.tight_layout()
plt.show()
