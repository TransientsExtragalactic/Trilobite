.. _data_gallery:

*****************************
Working With Radio Photometry
*****************************

These examples demonstrate how to work with **observational radio data** in Triceratops. They cover
the full data ingestion workflow:

- loading a FITS photometry table,
- inspecting its contents,
- grouping observations into temporal epochs,
- and plotting individual epochs or the full light curve.

The :mod:`~triceratops.data.photometry` subpackage provides container classes for radio photometry that
carry unit-aware arrays (frequency, flux density, flux uncertainty, upper-limit flags) and
integrate directly with the inference pipeline.

.. rubric:: What you'll find here

- Loading :class:`~triceratops.data.photometry.RadioPhotometryContainer` objects from FITS files
- Inspecting and printing photometry tables
- Defining temporal epochs from observation time-gaps
- Plotting light curves and individual SED epochs

.. rubric:: API reference

- :mod:`~triceratops.data.photometry`
- :class:`~triceratops.data.photometry.RadioPhotometryContainer`,
- :class:`~triceratops.data.photometry.RadioPhotometryEpochContainer`
