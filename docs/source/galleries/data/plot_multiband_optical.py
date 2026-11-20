"""
Multi-Band Optical Light Curves
================================

When working with multi-band optical photometry it is often useful to
isolate individual filter light curves for per-band modeling or
visualisation.
:meth:`~trilobite.data.optical_photometry.OpticalPhotometryContainer.extract_lightcurve`
selects all observations with a given ``band_name`` and returns an
:class:`~trilobite.data.light_curve.OpticalLightCurveContainer` — the
natural input to single-band optical models.
"""

# %%
# Multi-Band Optical Dataset
# ---------------------------
#
# We construct a four-band (g, r, i, z) dataset representing a fading
# transient observed over ~300 days, similar to a Type Ia supernova at late
# times.  The magnitude supplied here; the container will derive fluxes on
# demand.

import numpy as np
from astropy import units as u
from astropy.table import Table

from trilobite.data import OpticalPhotometryContainer

rng = np.random.default_rng(55)

bands = ["g", "r", "i", "z"]
# Band-dependent absolute magnitude at peak and decline rate.
peak_mag = {"g": 17.5, "r": 17.2, "i": 17.0, "z": 16.9}
decline = {"g": 0.018, "r": 0.014, "i": 0.012, "z": 0.010}  # mag/day
n_per_band = 18

rows = {k: [] for k in ["time", "band_name", "mag_ab", "mag_ab_error", "mag_ab_upper_limit"]}

for band in bands:
    t = np.sort(rng.uniform(2, 280, n_per_band))
    m = peak_mag[band] + decline[band] * t + rng.normal(0, 0.05, n_per_band)
    merr = rng.uniform(0.03, 0.12, n_per_band)
    m_ul = np.full(n_per_band, np.nan)
    # Late-time upper limits (source below detection threshold).
    m[-3:] = np.nan
    merr[-3:] = np.nan
    m_ul[-3:] = peak_mag[band] + decline[band] * t[-3:] + 0.3  # faint limit

    rows["time"].extend(t)
    rows["band_name"].extend([band] * n_per_band)
    rows["mag_ab"].extend(m)
    rows["mag_ab_error"].extend(merr)
    rows["mag_ab_upper_limit"].extend(m_ul)

table = Table(
    {
        "time": np.array(rows["time"]) * u.day,
        "band_name": np.array(rows["band_name"]),
        "mag_ab": np.array(rows["mag_ab"]),
        "mag_ab_error": np.array(rows["mag_ab_error"]),
        "mag_ab_upper_limit": np.array(rows["mag_ab_upper_limit"]),
    }
)

phot = OpticalPhotometryContainer(table)
print(f"Observations : {phot.n_obs}")
print(f"Detections   : {phot.n_detections}")

# %%
# Extracting Single-Band Light Curves
# -------------------------------------
#
# A single call per band isolates all observations in that filter and wraps
# them in an :class:`~trilobite.data.light_curve.OpticalLightCurveContainer`.
# The ``band_name`` column is dropped from the table and stored as metadata
# on the returned container.

light_curves = {band: phot.extract_lightcurve(band) for band in bands}

for band, lc in light_curves.items():
    print(f"  {band}: {lc.n_detections} detections, {lc.n_non_detections} ULs, band metadata='{lc.band}'")

# %%
# Per-Band Plot
# --------------
#
# We plot each extracted light curve in AB magnitudes.  Because the
# container supplies both magnitude and flux representations, switching to a
# flux-density plot requires only changing which property is accessed.

import matplotlib.pyplot as plt

from trilobite.utils.plot_utils import set_plot_style

set_plot_style()

band_colors = {"g": "#1f77b4", "r": "#d62728", "i": "#8c564b", "z": "#7f7f7f"}

fig, ax = plt.subplots(figsize=(9, 5))

for band, lc in light_curves.items():
    col = band_colors[band]
    det = lc.detection_mask
    ul = lc.non_detection_mask

    ax.errorbar(
        lc.time[det].to_value(u.day),
        lc.mag[det],
        yerr=lc.mag_error[det],
        fmt="o",
        color=col,
        label=band,
    )
    if np.any(ul):
        ax.errorbar(
            lc.time[ul].to_value(u.day),
            lc.mag_upper_limit[ul],
            fmt="^",
            color=col,
            alpha=0.5,
        )

ax.invert_yaxis()
ax.set_xlabel("Time (days)")
ax.set_ylabel(r"$m_\mathrm{AB}$ (mag)")
ax.set_title("Multi-band optical light curves extracted from photometry container")
ax.legend(title="Band")
plt.tight_layout()
plt.show()
