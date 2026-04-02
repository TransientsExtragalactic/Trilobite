.. _modeling_gallery:

*****************************
Forward Modeling
*****************************

These examples show how to evaluate Triceratops' high-level physical models directly — without
running inference — to generate synthetic light curves and SEDs, visualize model parameter
sensitivity, and compare model predictions to observations.

All models inherit from :class:`~triceratops.models.core.Model` and share a common interface:
declare :attr:`PARAMETERS` (fixed or free) and call :meth:`~triceratops.models.core.Model.__call__`
to evaluate. Examples range from Chevalier self-similar shock models for radio supernovae through
phenomenological light-curve fitters and TDE radio afterglow models.

.. rubric:: What you'll find here

- Evaluating a Chevalier shock-synchrotron model with radiative cooling
- Modeling free-free absorption (FFA) fadeout in the evolving SED of a dense-CSM supernova
- Fitting a phenomenological light curve with a FRED-like model
- Generating TDE radio afterglow predictions
- Modeling a Type IIn supernova embedded in dense circumstellar material

.. rubric:: API reference

:mod:`triceratops.models` — :class:`~triceratops.models.core.Model`,
:class:`~triceratops.models.core.ModelParameter`
