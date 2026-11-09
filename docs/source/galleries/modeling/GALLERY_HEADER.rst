.. _modeling_gallery:

*****************************
Forward Modeling
*****************************

These examples evaluate Trilobite's physical models directly, without running inference, and compare the
predictions with published observations of real transients. Each one is anchored to a specific source, with
shock, wind and microphysics parameters taken from the literature, so the output can be checked against
measured values.

The radio supernova examples follow the same pipeline: a shock model gives the radius and speed, the
strong-shock jump conditions give the magnetic field, the cooling engine gives the cooling break, and the
numerical synchrotron engine produces the SED. Absorption by the unshocked wind is added where it matters.

.. rubric:: What you'll find here

- A radio supernova in a red-supergiant wind (SN 1993J): shock speed, magnetic field and light curves
- Free-free absorption in an extremely dense wind (SN 2010jl): when each radio band turns on
- Fitting radio light curves with a phenomenological function to read off peak times and slopes
- A TDE radio afterglow from a freely expanding outflow

.. rubric:: API reference

:mod:`trilobite.models` — :class:`~trilobite.models.core.base.Model`,
:class:`~trilobite.models.core.parameters.ModelParameter`
