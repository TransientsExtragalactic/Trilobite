.. _data_gallery:

*******************************
Working With Observational Data
*******************************

These examples demonstrate how to load, inspect, and visualize observational
data in Trilobite using the container classes in :mod:`trilobite.data`. They
cover the full range of supported data types — radio photometry, radio light
curves, optical photometry, and optical light curves — as well as the bridge
from validated containers to the inference pipeline.

.. rubric:: What you'll find here

- Constructing :class:`~trilobite.data.photometry.RadioPhotometryContainer`
  objects, grouping observations into epochs, and plotting SEDs.
- Working with :class:`~trilobite.data.light_curve.RadioLightCurveContainer`
  for single-frequency time-series data.
- Building multi-band :class:`~trilobite.data.optical_photometry.OpticalPhotometryContainer`
  objects and extracting per-band light curves.
- Using :class:`~trilobite.data.light_curve.OpticalLightCurveContainer` for
  flux/magnitude dual representation.
- Converting any container to :class:`~trilobite.data.core.InferenceData` and
  understanding its structure before wiring into a likelihood.
- Extracting per-band light curves from multi-frequency photometry with
  :meth:`~trilobite.data.photometry.RadioPhotometryContainer.extract_lightcurve`
  and :meth:`~trilobite.data.optical_photometry.OpticalPhotometryContainer.extract_lightcurve`.

.. rubric:: API reference

- :mod:`trilobite.data`
- :class:`~trilobite.data.photometry.RadioPhotometryContainer`
- :class:`~trilobite.data.light_curve.RadioLightCurveContainer`
- :class:`~trilobite.data.optical_photometry.OpticalPhotometryContainer`
- :class:`~trilobite.data.light_curve.OpticalLightCurveContainer`
- :class:`~trilobite.data.core.InferenceData`
