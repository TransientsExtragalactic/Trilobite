"""
====================================
Plot Radio Photometry Epochs
====================================

This example will demonstrate how to load radio photometry data and separate it into epochs for
plotting using the Triceratops library. We'll utilize the :class:`data.photometry.RadioPhotometryContainer`
to manage and visualize the data.
"""

# %%
# Setup
# -----
# First, we need to import the necessary libraries and set up the plotting style.
import matplotlib.pyplot as plt
from astropy import units as u

from triceratops.data.photometry import RadioPhotometryContainer
from triceratops.utils.plot_utils import set_plot_style

# %%
# We'll use a set of synthetically generated radio photometry data for this example. The data is
# stored in the ``_data/example_photometry.fits`` file within the documentation source directory.
#
# This dataset is almost in the perfect format for loading into a
# :class:`data.photometry.RadioPhotometryContainer`, except that it doesn't have an explicit ``time`` column, so
# we will need to select one. We also need to give a reference time for the observations, which we will set to the JD
# of the first observation.

# Resolve the path to the dataset. In this case, we seek out the data in the
# documentation.
data_file = "../../../../../docs/source/_data/example_photometry.fits"

# Load the dataset as a RadioPhotometryContainer. We specify that the ``time`` column
# should be taken from the ``time_midobs`` column in the dataset.
photometry_data = RadioPhotometryContainer.from_file(
    data_file,
    column_map={"time_midobs": "time"},
    time_starts=2458368.45645 * u.day,  # Reference time (JD of first observation)
)
print(f"Loaded photometry data with {len(photometry_data)} entries.")

# Render the table into the documentation.
photometry_data.table.pprint()

# %%
# Now that we have loaded the data, we can plot it. The :class:`~data.photometry.RadioPhotometryContainer`
# includes a built-in plotting method that makes this straightforward:
# :meth:`~data.photometry.RadioPhotometryContainer.plot_photometry`.

# Set the plot style.
set_plot_style()

# Create a figure and axes for the plot.
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the photometry data.
photometry_data.plot_photometry(fig=fig, axes=ax, show_upper_limits=True)

# Customize the plot.
fig.show()

# %%
# As you may notice, the data points (even sorted by color) are not all that easy to pick apart. Let's instead
# plot the empirical SEDs at different epochs to better visualize the evolution of the radio emission over time.
# To do this, we'll need to use the built-in epoching functionality of the
# :class:`~data.photometry.RadioPhotometryContainer`.
#
# Let's first look at the various time ranges covered by the data:

# Create a histogram of observation times.
fig, ax = plt.subplots(figsize=(10, 4))
_, edges, _ = ax.hist(photometry_data.time.to_value(u.day), bins=60, color="skyblue", edgecolor="black")
ax.set_xlabel("Time since explosion (days)")
ax.set_ylabel("Number of observations")
ax.set_title("Histogram of Observation Times")
fig.show()

# %%
# If you already have clearly defined epochs in mind, you can specify them directly in your table using the
# ``epoch_id`` column, or you can use the built-in method
# :meth:`~data.photometry.RadioPhotometryContainer.set_epochs_from_indices`; however, we do not have that information here.
#
# Instead, we can use the :meth:`~data.photometry.RadioPhotometryContainer.set_epochs_from_time_gaps` method to
# automatically define epochs based on gaps in the observation times. We'll specify a gap threshold of 2 days to
# separate the epochs.

# Set epochs based on time gaps of 2 days.
photometry_data.set_epochs_from_time_gaps(max_gap=2.0 * u.day)

# Create another set of histograms, this time colored by epoch.
fig, ax = plt.subplots(figsize=(10, 4))
num_epochs = photometry_data.n_epochs
colors = plt.get_cmap("rainbow")

for epoch_id in range(num_epochs):
    epoch_mask = photometry_data.get_epoch_mask(epoch_id)
    ax.hist(
        photometry_data.time[epoch_mask].to_value(u.day),
        bins=edges,
        alpha=0.6,
        label=f"Epoch {epoch_id + 1}",
        color=colors(epoch_id / num_epochs),
        edgecolor="black",
    )

ax.set_xlabel("Time since explosion (days)")
ax.set_ylabel("Number of observations")
ax.set_title("Histogram of Observation Times by Epoch")
ax.legend()
fig.show()

# %%
# Now that we have defined each of the epochs, we can plot the SEDs for each epoch using the epoch masks.

fig, axes = plt.subplots(figsize=(10, 6))

for epoch_id in range(num_epochs):
    epoch_mask = photometry_data.get_epoch_mask(epoch_id)

    freq, flux, err = (
        photometry_data.freq[epoch_mask].to_value(u.Hz),
        photometry_data.flux_density[epoch_mask].to_value(u.mJy),
        photometry_data.flux_density_error[epoch_mask].to_value(u.mJy),
    )

    # Sort by frequency so that things are in order.
    sorted_indices = freq.argsort()
    freq = freq[sorted_indices]
    flux = flux[sorted_indices]
    err = err[sorted_indices]

    # Plot the SED for this epoch.
    # sphinx_gallery_thumbnail_number = -1
    axes.errorbar(freq, flux, yerr=err, fmt="o-", color=colors(epoch_id / num_epochs))

axes.set_xscale("log")
axes.set_yscale("log")
axes.set_xlabel("Frequency (Hz)")
axes.set_ylabel("Flux Density (mJy)")
axes.set_title("Radio SEDs at Different Epochs")
fig.show()
