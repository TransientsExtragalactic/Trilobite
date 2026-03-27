r"""
C-level physics building blocks for one-zone disk closures.

These Cython modules provide the low-level functions that closure
implementations ``cimport`` and compose.  They are not intended to be called
directly from Python.

Modules
-------

_eos
    Equation-of-state primitives: isothermal sound speed for a pure ideal gas
    (:func:`~_eos.compute_ideal_gas_cs`) and for a gas + radiation pressure
    mixture (:func:`~_eos.compute_gas_rad_cs`).
_viscous
    Metzger et al. (2008) boundary-corrected alpha-disk viscous derivative
    (:func:`~_viscous.viscous_derivative_func`).  Computes ``dM/dt``,
    ``dJ/dt``, and the adaptive timestep ``dt``.
_fallback
    Power-law debris-stream source term
    (:func:`~_fallback.fallback_source_func`).  Models TDE fallback as
    :math:`\\dot{M}_{\\rm fb} \\propto t^{-\\beta}` and deposits the
    corresponding angular momentum at the disk outer radius.
"""
