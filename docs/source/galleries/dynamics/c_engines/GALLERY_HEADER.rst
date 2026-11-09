Numerical Shock Engines
========================

These examples demonstrate Trilobite's thin-shell shock engines — the
numerical integrators that evolve blast-wave radius, velocity, and swept-up
mass through an arbitrary circumstellar medium (CSM) density profile.

Four engines are covered:

- :class:`~trilobite.dynamics.shocks.numerical.PressureDrivenThinShellShockEngine`
  — the standard non-relativistic pressure-driven thin shell
- :class:`~trilobite.dynamics.shocks.numerical.RelPressureDrivenThinShellShockEngine`
  — the fully relativistic variant; shows where :math:`\Gamma\beta \gtrsim 1`
  corrections become essential
- CSM environment comparison (thin shell) — same ejecta launched into ISM,
  wind, truncated wind, and shell density profiles using the
  :class:`~trilobite.dynamics.shocks.numerical.PressureDrivenThinShellShockEngine`
- CSM environment comparison (mechanical) — same comparison using the
  :class:`~trilobite.dynamics.shocks.numerical.MechanicalShockEngine` to show
  how CSM structure separately imprints on the forward and reverse shock

Use these examples when you need to evolve a shock through a realistic,
non-power-law density profile.

**API:** :ref:`numerical_shocks_overview`, :ref:`profiles_user_guide`
