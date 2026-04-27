r"""
Loading Filters from the SVO Filter Profile Service
====================================================

The `Spanish Virtual Observatory (SVO) Filter Profile Service
<https://svo2.cab.inta-csic.es/svo/theory/fps/>`_ catalogues thousands of
photometric bandpasses from virtually every major telescope and instrument.
Triceratops can fetch any of them via ``astroquery.svo_fps`` and wrap them
directly into a :class:`~triceratops.utils.phot_utils.PhotometryFilter` or a
:class:`~triceratops.utils.phot_utils.FilterBundle`.

.. note::

   This example requires an internet connection on first run. Results are
   cached locally by ``astroquery`` for one week, so subsequent runs are
   offline.

   Install the required optional dependencies with::

       pip install triceratops[optical]

Relevant API references
-----------------------
- :func:`triceratops.utils.phot_utils.list_svo_filters`
- :func:`triceratops.utils.phot_utils.load_filter_from_svo`
- :func:`triceratops.utils.phot_utils.load_filters_from_svo`
- :class:`triceratops.utils.phot_utils.FilterBundle`
"""

# %%
# Imports
# -------
import matplotlib.pyplot as plt
import numpy as np

from triceratops.utils.phot_utils import (
    FilterBundle,
    load_filter_from_svo,
    list_svo_filters,
)
from triceratops.utils.plot_utils import set_plot_style

set_plot_style()

# %%
# Browsing Available Filters
# --------------------------
#
# :func:`~triceratops.utils.phot_utils.list_svo_filters` returns a metadata
# table for every filter registered for a given facility. Passing
# ``instrument=`` restricts the listing to a single detector/camera — the
# filtering is applied client-side on the ``filterID`` column, so the function
# always works regardless of SVO server capabilities.
#
# Here we browse the HST/WFC3 UVIS1 channel (63 broadband and narrowband
# filters) and inspect the five bluest entries.

hst_uvis1 = list_svo_filters("HST", instrument="WFC3_UVIS1")
print(f"HST/WFC3_UVIS1: {len(hst_uvis1)} filters in the SVO catalogue.")
print(hst_uvis1["filterID", "WavelengthEff", "FWHM"][:5])

# %%
# Loading a Single Filter
# -----------------------
#
# :func:`~triceratops.utils.phot_utils.load_filter_from_svo` takes any SVO
# filter ID in ``"facility/instrument.band"`` format and returns a
# :class:`~triceratops.utils.phot_utils.PhotometryFilter` with the
# transmission curve already normalised into integration weights.

f555w = load_filter_from_svo("HST/WFC3_UVIS1.F555W")
print(f555w)
print(f"  effective wavelength : {f555w.effective_wavelength * 1e8:.1f} Å")
print(f"  equivalent width     : {f555w.filter_width_lambda * 1e8:.1f} Å")

# %%
# Building a FilterBundle from a Selection of Filters
# ---------------------------------------------------
#
# We load five classic WFC3/UVIS1 broadband filters spanning UV to near-IR
# and assemble them into a :class:`~triceratops.utils.phot_utils.FilterBundle`.
# The bundle merges all individual grids into one common frequency grid and
# precomputes the weight matrix, so all five filters can be applied to any SED
# in a single matrix multiplication — the hot-loop method for MCMC.

BROADBAND_IDS = [
    "HST/WFC3_UVIS1.F275W",
    "HST/WFC3_UVIS1.F336W",
    "HST/WFC3_UVIS1.F438W",
    "HST/WFC3_UVIS1.F555W",
    "HST/WFC3_UVIS1.F814W",
]

# Use the band name (e.g. "F555W") as the bundle key for readability
filters = {fid.split(".")[-1]: load_filter_from_svo(fid) for fid in BROADBAND_IDS}
bundle = FilterBundle(filters)

print(bundle)
for name, filt in bundle.filters.items():
    print(f"  {name:6s}  lam_eff = {filt.effective_wavelength * 1e8:.0f} Å")

# %%
# Visualising the Passbands
# -------------------------

colors = plt.cm.plasma(np.linspace(0.05, 0.90, len(filters)))

fig, ax = plt.subplots(figsize=(10, 4))
for color, (name, filt) in zip(colors, filters.items()):
    ax.plot(filt.wavelength * 1e8, filt.transmission, color=color, lw=2, label=name)
    ax.fill_between(filt.wavelength * 1e8, 0, filt.transmission, color=color, alpha=0.15)

ax.set_xlabel("Wavelength [Å]")
ax.set_ylabel("Transmission")
ax.set_title("HST/WFC3 UVIS1 broadband passbands (from the SVO catalogue)")
ax.set_ylim(bottom=0.0)
ax.legend(ncol=5, fontsize=9)
plt.tight_layout()
plt.show()

# %%
# Any filter loaded from the SVO is a standard
# :class:`~triceratops.utils.phot_utils.PhotometryFilter`, so all the usual
# methods — :meth:`~triceratops.utils.phot_utils.PhotometryFilter.convolve_nu`,
# :meth:`~triceratops.utils.phot_utils.FilterBundle.apply`,
# serialisation to HDF5/JSON — work without modification.
#
# To load every filter for a facility at once, use
# :func:`~triceratops.utils.phot_utils.load_filters_from_svo`:
#
# .. code-block:: python
#
#     from triceratops.utils.phot_utils import load_filters_from_svo, FilterBundle
#     all_hst = load_filters_from_svo("HST", instrument="WFC3_UVIS1")
#     bundle = FilterBundle(all_hst)
#
# To explore other facilities, browse
# https://svo2.cab.inta-csic.es/svo/theory/fps/ and replace ``"HST"`` /
# ``"WFC3_UVIS1"`` with any facility/instrument pair listed there.
