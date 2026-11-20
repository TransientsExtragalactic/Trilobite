Closures and Equipartition
===========================

These examples turn an observed radio SED into the physical properties of the emitting region: radius
:math:`R`, magnetic field :math:`B` and energy content. They start from the peak flux and frequency
(:math:`F_\mathrm{pk}`, :math:`\nu_\mathrm{pk}`), apply an inverse closure, and show how the answer depends on the
assumed energy partition and on measurement uncertainty. A separate example estimates the bolometric synchrotron
luminosity of a shell.

**Theory:** :ref:`sed_forward_closure`, :ref:`sed_inverse_closure`, :ref:`synch_equipartition_theory`

**API:** :ref:`synchrotron_core`, :ref:`synchrotron_seds`

.. note::

    The analytic one-zone closures are being phased out in favor of the numerical engines. Where these examples
    invert an SED, the SED is generated with the numerical engine first, so the inversion is tested against an
    independent calculation. See :ref:`sphx_glr_auto_examples_synchrotron_e_inversions` for the closed-form
    inversions.
