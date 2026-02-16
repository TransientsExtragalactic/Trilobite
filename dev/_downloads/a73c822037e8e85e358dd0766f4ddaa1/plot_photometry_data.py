"""
====================================
Plot Radio Photometry Data
====================================

This example demonstrates how to load and plot radio photometry data using the
Triceratops library. We'll be using the :class:`data.photometry.RadioPhotometryContainer`
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
# Now that we have loaded the data, we can plot it. The :class:`data.photometry.RadioPhotometryContainer`
# includes a built-in plotting method that makes this straightforward:
# :meth:`data.photometry.RadioPhotometryContainer.plot_photometry`.

# Set the plot style.
set_plot_style()

# Create a figure and axes for the plot.
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the photometry data.
photometry_data.plot_photometry(fig=fig, axes=ax, show_upper_limits=True)

# Customize the plot.
fig.show()
