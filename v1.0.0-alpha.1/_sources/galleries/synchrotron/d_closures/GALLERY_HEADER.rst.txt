Closures and Equipartition
===========================

These examples demonstrate the *closure relations* that bridge the gap between observed SED
parameters (:math:`F_\mathrm{pk}`, :math:`\nu_\mathrm{pk}`) and the physical properties of
the emitting region (radius :math:`R`, magnetic field :math:`B`, electron density :math:`n_e`).

The **forward closure** maps physical parameters onto observables; the **inverse closure** does
the reverse. Together they enable a complete equipartition analysis: given a measured SED,
recover the source size, magnetic field, and energy content — and propagate measurement
uncertainties through to physical posteriors via MCMC.

**Theory:** :ref:`sed_forward_closure`, :ref:`sed_inverse_closure`, :ref:`synch_equipartition_theory`

**API:** :ref:`synchrotron_microphysics`, :ref:`synchrotron_seds`
