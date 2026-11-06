Spectral Inversions
===================

These examples test the closed-form SSA inversions in
:mod:`trilobite.radiation.synchrotron.spectral_inversion`. Each one follows the same round trip:
build a synchrotron SED with the numerical engine from known :math:`B` and :math:`R`,
extract the break frequency and flux with a smoothly broken power-law fit, and invert those two numbers.
Because the SED is generated independently of the inversion, the recovered values measure the
accuracy of the inversion itself.

**Theory:** :ref:`synchrotron_spectral_inversion`

**API:** :ref:`synchrotron_core`
