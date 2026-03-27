#cython: language_level=3, boundscheck=False
r"""
[C-LEVEL] Source-term convention for one-zone accretion disk models (Cython).

This module defines what a *source term* is within the one-zone framework.
Concrete implementations live in :mod:`triceratops.dynamics.accretion.one_zone.physics`.

A source function has the :c:type:`source_func` signature (defined in
``closure.pxd``)::

    int source_func(
        const DiskState*      state,    # current disk state
        const DiskDerived*    derived,  # geometric / kinematic quantities
        const DiskParameters* params,   # fixed model parameters
        const ClosureResult*  closure,  # thermodynamic closure at this step
        DiskStep*             step,     # MODIFIED IN-PLACE: add/subtract dM_dt, dJ_dt
    ) nogil

Semantics
---------

* Called **after** the base viscous derivative and **before** the writer
  at each integration step.
* May add or subtract from ``step.dM_dt`` and ``step.dJ_dt``.
* May in principle also modify ``step.dt`` (e.g., to restrict the time step
  when a source is very large).
* Must **not** modify thermodynamic closure fields
  (``ClosureResult.log_T_c``, etc.) — those are set by the closure function.
* Return **0** for success.  Any non-zero value causes the integrator to
  return ``-30`` (SOURCE_FAIL).

Registering a source term
--------------------------

Set ``closure._source_fn = <your_function>`` in the ``__cinit__`` of a
concrete :class:`~.closure.OneZoneClosure` subclass.
``_source_fn = NULL`` (the default) means no source term.

Available implementations
--------------------------

* :func:`~triceratops.dynamics.accretion.one_zone.physics._fallback.fallback_source_func` —
  power-law debris-stream supply.

See Also
--------

:mod:`triceratops.dynamics.accretion.one_zone.closure` :
    ``source_func`` typedef and ``OneZoneClosure`` base class.
:mod:`triceratops.dynamics.accretion.one_zone.physics._fallback` :
    Fallback mass supply implementation.
"""
