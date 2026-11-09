"""
From Container to InferenceData
================================

Every data container in Trilobite exposes a
:meth:`to_inference_data` method that converts validated observational
data into an :class:`~trilobite.data.core.InferenceData` object — the
single format accepted by all likelihood functions.

This example walks through that conversion for a
:class:`~trilobite.data.photometry.RadioPhotometryContainer` using the
:class:`~trilobite.models.supernovae.chevalier_shock.ChevalierShockModel`.
The model's declared variables (``frequency``, ``time``) and output
(``flux_density``) tell the container which columns to map and which unit
coercions to apply.
"""

# %%
# Step 1 — Build a Synthetic Container
# --------------------------------------
#
# In a real workflow you would load from a FITS file.  Here we construct
# synthetic multi-frequency radio photometry so the example is
# self-contained.

import numpy as np
from astropy import units as u
from astropy.table import Table

from trilobite.data import RadioPhotometryContainer

rng = np.random.default_rng(0)
freqs_ghz = [5.0, 8.5, 15.0]
n_per_band = 12

rows = {k: [] for k in ["time", "freq", "flux_density", "flux_density_error", "flux_upper_limit"]}

for nu in freqs_ghz:
    t = np.sort(rng.uniform(5, 400, n_per_band))
    f = 2e-3 * (t / 30.0) ** -0.7 * (nu / 5.0) ** -0.65 * rng.lognormal(0, 0.12, n_per_band)
    err = f * rng.uniform(0.07, 0.18, n_per_band)
    ul = np.full(n_per_band, np.nan)
    # Two upper limits at the end of each band.
    f[-2:] = np.nan
    err[-2:] = np.nan
    ul[-2:] = rng.uniform(2e-4, 5e-4, 2)
    rows["time"].extend(t)
    rows["freq"].extend([nu] * n_per_band)
    rows["flux_density"].extend(f)
    rows["flux_density_error"].extend(err)
    rows["flux_upper_limit"].extend(ul)

container = RadioPhotometryContainer(
    Table(
        {
            "time": np.array(rows["time"]) * u.day,
            "freq": np.array(rows["freq"]) * u.GHz,
            "flux_density": np.array(rows["flux_density"]) * u.Jy,
            "flux_density_error": np.array(rows["flux_density_error"]) * u.Jy,
            "flux_upper_limit": np.array(rows["flux_upper_limit"]) * u.Jy,
        }
    )
)

print(f"Observations : {container.n_obs}")
print(f"Detections   : {container.n_detections}")
print(f"Upper limits : {container.n_non_detections}")

# %%
# Step 2 — Instantiate the Model
# --------------------------------
#
# The model declares the independent variables (``frequency`` in Hz,
# ``time`` in seconds) and the observable output (``flux_density`` in Jy).
# The container maps its columns to those variables automatically:
# ``"frequency"`` is resolved to the ``"freq"`` column, and ``"time"`` maps
# directly.

from trilobite.models.supernovae.chevalier_shock import ChevalierShockModel

model = ChevalierShockModel()
print("\nModel variables :", model.variable_names)
print("Model outputs   :", model.output_names)

# %%
# Step 3 — Convert to InferenceData
# ----------------------------------
#
# A single call performs all unit coercions, column mapping, and upper-limit
# handling.  The ``infer_errors`` flag derives a 3-sigma error floor for
# non-detections, making the result compatible with both standard and
# censored likelihoods.

inference_data = container.to_inference_data(model, infer_errors=True, detection_threshold=3.0)
print(inference_data.describe())

# %%
# The :class:`~trilobite.data.core.InferenceData` object exposes independent
# variables through ``x`` (a dict of plain NumPy arrays in base model units)
# and observables through ``observables`` (a dict of
# :class:`~trilobite.data.core.Observable` instances).

obs = inference_data.observables["flux_density"]
print(f"\nObservable 'flux_density':")
print(f"  shape        : {obs.shape}")
print(f"  has errors   : {obs.has_error}")
print(f"  has censoring: {obs.has_censoring}")

# %%
# Step 4 — Quick Diagnostic Plot
# --------------------------------
#
# Always inspect your ``InferenceData`` before running inference.  A quick
# plot confirms that unit conversions landed sensibly.

import matplotlib.pyplot as plt

from trilobite.utils.plot_utils import set_plot_style

set_plot_style()

time_s = inference_data.x["time"]
time_day = time_s / 86400.0
is_det = np.isfinite(obs.value)

fig, ax = plt.subplots(figsize=(8, 4))
ax.errorbar(
    time_day[is_det],
    obs.value[is_det],
    yerr=obs.error[is_det],
    fmt="o",
    color="steelblue",
    label="Detection",
)
ax.errorbar(
    time_day[~is_det],
    obs.upper[~is_det],
    fmt="v",
    color="gray",
    alpha=0.7,
    label="Upper limit",
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Time (days)")
ax.set_ylabel(r"$F_\nu$ (Jy)")
ax.set_title("InferenceData — flux density vs. time")
ax.legend()
plt.tight_layout()
plt.show()

# %%
# The resulting :class:`~trilobite.data.core.InferenceData` object can now
# be passed directly to a likelihood:
#
# .. code-block:: python
#
#     from trilobite.inference.likelihood import GaussianCensoredLikelihood
#     from trilobite.inference.problem import InferenceProblem
#
#     likelihood = GaussianCensoredLikelihood(model=model, data=inference_data)
#     problem    = InferenceProblem(likelihood)
#
# See the :ref:`inference_gallery` for a complete worked example.
