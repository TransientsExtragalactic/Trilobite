"""
====================================
Plot Single Epochs
====================================

In this simple cookbook, we'll load in some data and
break it into epochs and then plot a subset of those
epochs using the :meth:`~data.photometry.RadioPhotometryContainer.plot_epoch`
method of :class:`data.photometry.RadioPhotometryContainer`.
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
# If you already have clearly defined epochs in mind, you can specify them directly in your table using the
# ``epoch_id`` column, or you can use the built-in method
# :meth:`~data.photometry.RadioPhotometryContainer.set_epochs_from_indices`; however, we do not have that information here.
#
# Instead, we can use the :meth:`~data.photometry.RadioPhotometryContainer.set_epochs_from_time_gaps` method to
# automatically define epochs based on gaps in the observation times. We'll specify a gap threshold of 2 days to
# separate the epochs.

# Set epochs based on time gaps of 2 days.
photometry_data.set_epochs_from_time_gaps(max_gap=3.0 * u.day)

# %%
# Now that we have defined each of the epochs, we can plot the SEDs for each epoch using the epoch masks.
set_plot_style()
fig, axes = plt.subplots(figsize=(10, 6))

# Plot specific epochs by their IDs.
epoch_ids_to_plot = [0, 2, 5]  # Plot epochs with IDs 0
colors = ["darkblue", "darkgreen", "darkred"]

for k, epoch_id in enumerate(epoch_ids_to_plot):
    photometry_data.plot_epoch(
        epoch_id=epoch_id,
        fig=fig,
        axes=axes,
        show_upper_limits=True,
        color=colors[k],
        detection_style={"marker": "o", "markersize": 5, "mfc": "w", "mec": colors[k], "ls": "--", "linewidth": 1.5},
        label=f"Epoch {epoch_id}",
    )

axes.set_xlabel("Frequency (GHz)")
axes.set_ylabel("Flux Density (mJy)")
axes.legend()
axes.set_title("Radio Photometry for Specific Epochs")
axes.set_xscale("log")
axes.set_yscale("log")

plt.show()
