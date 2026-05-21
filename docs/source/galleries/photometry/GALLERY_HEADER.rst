.. _photometry_gallery:

*******************************
Optical Filters and Photometry
*******************************

These examples demonstrate the optical photometry system in Trilobite. They cover
constructing and visualising photometric bandpass filters, assembling them into a
:class:`~trilobite.utils.phot_utils.FilterBundle` optimised for MCMC hot loops, and
converting between flux density and standard magnitude systems (AB, ST).

These tools are the bridge between a physical model that predicts a spectral energy
distribution and the broadband multi-filter light curves delivered by surveys such as ZTF,
LSST/Rubin, DECam, and space-based observatories like HST and Kepler.

.. rubric:: What you'll find here

- Creating, inspecting, and visualising photometric filters
- Assembling filters into a bundle and exploiting matrix-multiply convolution during MCMC
- AB and ST magnitude conversions and filter-convolved synthetic light curves
- Loading filters from the `SVO Filter Profile Service <https://svo2.cab.inta-csic.es/svo/theory/fps/>`_

.. rubric:: API reference

:class:`~trilobite.utils.phot_utils.PhotometryFilter` —
:class:`~trilobite.utils.phot_utils.FilterBundle` —
:func:`~trilobite.utils.phot_utils.flux_to_ab_mag` —
:func:`~trilobite.utils.phot_utils.filter_to_ab_mag` —
:func:`~trilobite.utils.phot_utils.load_filter_from_speclite` —
:func:`~trilobite.utils.phot_utils.load_filter_from_svo` —
:func:`~trilobite.utils.phot_utils.list_svo_filters` —
:func:`~trilobite.utils.phot_utils.load_filters_from_svo`
