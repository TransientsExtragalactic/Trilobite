#cython: language_level=3, boundscheck=False
r"""
[C-LEVEL] Output writers for one-zone accretion disk closures.

The one-zone integrator calls a :c:type:`writer_func` once per accepted step.
The writer receives all current disk state, and must write exactly
``n_result_fields`` values into the pre-allocated result array.  The array is
laid out **field-major** (C order, shape ``(N_RESULT_FIELDS, n_steps)``), so
field ``f`` at step ``i`` lives at offset ``f * n_steps + i``.

:func:`standard_writer_func` is the canonical implementation of that contract.
Because every :class:`~.closure.OneZoneClosure` fills the same
:c:type:`ClosureResult` struct, this writer is agnostic to EOS or opacity
prescription — it reads pre-computed closure fields and expands them to
physical values.

The default layout is as follows:

=====  ========================  ===========
Index  Quantity                  Units
=====  ========================  ===========
0      step index                —
1      t                         s
2      M_D                       g
3      J_D                       g cm² s⁻¹
4      R_D                       cm
5      Σ                         g cm⁻²
6      Ω                         s⁻¹
7      T_eff                     K
8      T_c                       K
9      τ                         —
10     c_s                       cm s⁻¹
11     ν                         cm² s⁻¹
12     q_visc                    erg cm⁻² s⁻¹
13     dM/dt                     g s⁻¹
14     dJ/dt                     g cm² s⁻²
15     dt                        s
16     t_visc                    s
17     H                         cm
18     H/R                       —
19     ρ                         g cm⁻³
=====  ========================  ===========

Extending the writer
--------------------

Use :func:`standard_writer_func` unchanged for any closure that fills the
standard :c:type:`ClosureResult` struct — different EOS or opacity choices do
not require a custom writer.

Write a custom writer when you need to:

* **Add output fields** (e.g. extra diagnostics not in ``ClosureResult``).
  Define a new function with the same ``writer_func`` signature
  (see :mod:`.closure`) and a matching ``N_RESULT_FIELDS`` constant, then
  assign both in your closure's ``__cinit__``::

      def __cinit__(self):
          self._writer_fn      = my_extended_writer_func   # cdef function
          self.n_result_fields = MY_N_RESULT_FIELDS         # cdef int

* **Change the field layout** (e.g. drop unused columns, reorder for a
  specific downstream format).  Same pattern as above — a new writer function
  + a matching field count.

The integrator allocates ``result_array`` as ``(n_result_fields, n_steps)``
*before* the first step using ``self.n_result_fields``, so the count must be
set in ``__cinit__``.

See Also
--------
:mod:`trilobite.dynamics.accretion.one_zone.closure` :
    ``ClosureResult`` struct and ``writer_func`` typedef.
:mod:`trilobite.dynamics.accretion.one_zone.integrator` :
    Hot loop that calls ``_writer_fn`` at each step.
"""
from libc.math cimport exp, log

from .closure cimport (
    ClosureResult, DiskDerived, DiskParameters, DiskState, DiskStep,
)


DEF _N_RESULT_FIELDS = 20

# Number of output fields written by standard_writer_func.
# Cimported by model files so they can set self.n_result_fields in __cinit__.
# If you add fields to standard_writer_func, increment this constant and
# add the corresponding row(s) to the field-layout table in the module docstring.
cdef int N_RESULT_FIELDS = _N_RESULT_FIELDS

# Python-visible alias for NumPy array pre-allocation in Python code.
N_RESULT_FIELDS_PY = _N_RESULT_FIELDS

cdef int standard_writer_func(
    const int step_index,
    const DiskState* state,
    const DiskDerived* derived,
    const DiskParameters* params,
    const ClosureResult* closure,
    const DiskStep* step,
    double* result_array,
    int n_steps,
) nogil:
    """Write one time step of output into the result array.

    Writes exactly :data:`N_RESULT_FIELDS` values into ``result_array`` at
    column ``step_index``.  The array is field-major: field ``f`` at step ``i``
    lives at offset ``f * n_steps + i``.  No other memory is touched.

    Parameters
    ----------
    step_index
        Column index (0-based) of the current step.
    state
        Current disk state (``t``, ``M``, ``J``).
    derived
        Geometry and kinematics derived from state (``R``, ``Sigma``, ``Omega``).
    params
        Fixed model parameters (``MBH``, ``R_in``, ``alpha``, ``mu``, …).
    closure
        Thermodynamic closure result for this step (log-scale fields from
        ``ClosureResult``).
    step
        ODE derivatives for this step (``dM_dt``, ``dJ_dt``, ``dt``).
    result_array
        Pointer to ``result[0, 0]`` of the pre-allocated
        ``(N_RESULT_FIELDS, n_steps)`` C-contiguous double array.
    n_steps
        Total number of columns allocated (= ``max_steps`` passed to the
        integrator).  Used as the row stride.

    Returns
    -------
    int
        Always 0 (success).  A non-zero return would signal a write error to
        the integrator, but the standard writer never fails.
    """
    cdef double H = exp(closure.log_cs - log(derived.Omega))
    cdef double rho = exp(closure.log_rho)

    result_array[ 0 * n_steps + step_index] = step_index
    result_array[ 1 * n_steps + step_index] = state.t
    result_array[ 2 * n_steps + step_index] = state.M
    result_array[ 3 * n_steps + step_index] = state.J
    result_array[ 4 * n_steps + step_index] = derived.R
    result_array[ 5 * n_steps + step_index] = derived.Sigma
    result_array[ 6 * n_steps + step_index] = derived.Omega
    result_array[ 7 * n_steps + step_index] = exp(closure.log_T_eff)
    result_array[ 8 * n_steps + step_index] = exp(closure.log_T_c)
    result_array[ 9 * n_steps + step_index] = exp(closure.log_tau)
    result_array[10 * n_steps + step_index] = exp(closure.log_cs)
    result_array[11 * n_steps + step_index] = exp(closure.log_nu)
    result_array[12 * n_steps + step_index] = exp(closure.log_q_visc)
    result_array[13 * n_steps + step_index] = step.dM_dt
    result_array[14 * n_steps + step_index] = step.dJ_dt
    result_array[15 * n_steps + step_index] = step.dt
    result_array[16 * n_steps + step_index] = closure.t_visc
    result_array[17 * n_steps + step_index] = H
    result_array[18 * n_steps + step_index] = H / derived.R
    result_array[19 * n_steps + step_index] = rho
    return 0
