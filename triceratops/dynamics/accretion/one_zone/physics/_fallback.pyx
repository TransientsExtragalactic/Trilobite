#cython: language_level=3, boundscheck=False
r"""
Fallback debris-stream source term (Cython).

Implements a power-law mass supply rate representing the fallback of tidally
disrupted debris onto the accretion disk, as in tidal disruption event (TDE)
models.

Parameter access
----------------
Fallback parameters are read from ``params.fallback`` (a ``FallbackParams*``),
which is set by the closure's ``_pack_params`` method before the hot loop.
The pointer is ``NULL`` when fallback is disabled; this function is only
registered as a source term when fallback is active, so it will always receive
a valid pointer.

The inflow rate is

.. math::

    \dot{M}_{\rm fb}(t) = M_{\rm fb,0}
        \left(\frac{t}{t_{\rm fb}}\right)^{-\beta_{\rm fb}}.

The deposited angular momentum assumes fallback material circularises at
the disk outer radius :math:`R_D`:

.. math::

    \dot{J}_{\rm fb} = \dot{M}_{\rm fb}\,\sqrt{G M_{\rm BH} R_D}.

See Also
--------
:mod:`triceratops.dynamics.accretion.one_zone._sources` :
    Source-term convention (signature, semantics, return codes).
"""
from libc.math cimport exp, log, pow

from ..closure cimport (
    LOG_G_CGS,
    ClosureResult, DiskDerived, DiskParameters, DiskState, DiskStep,
)


cdef int fallback_source_func(
    const DiskState* state,
    const DiskDerived* derived,
    const DiskParameters* params,
    const ClosureResult* closure,
    DiskStep* step,
) nogil:
    r"""Add a power-law fallback mass supply to the current time step.

    This source function (and all fallback implementations) expect that the
    4 runtime parameters immediately following the defaults are as follows:

        - ``M_fb_0``: fallback rate at the reference time (g s⁻¹)
        - ``R_C``: fallback circularization radius (cm)
        - ``t_fb``: fallback reference time (s)
        - ``beta_fb``: fallback power-law index (dimensionless)

    If this contract is not maintained, a new source function should be defined to
    correct the behavior.

    Parameters
    ----------
    state
        Current disk state; ``state.t`` is the current time in seconds.
    derived
        Derived quantities; ``derived.R`` is the disk outer radius (cm).
    params
        Model parameters.  ``params.fallback`` must be a valid
        :c:type:`FallbackParams*` (set by the closure's ``_pack_params``).
    closure
        Thermodynamic closure result (not used here).
    step
        Time-step struct modified in-place.

    Returns
    -------
    int
        Always 0 (SUCCESS).
    """
    cdef double M_fb_0  = params.fallback.M_fb_0
    cdef double R_C     = params.fallback.R_c
    cdef double t_fb    = params.fallback.t_fb
    cdef double beta_fb = params.fallback.beta_fb
    cdef double mdot_fb = M_fb_0 * pow(state.t / t_fb, -beta_fb)

    step.dM_dt += mdot_fb
    step.dJ_dt += mdot_fb * exp(0.5 * (LOG_G_CGS + log(params.MBH) + log(R_C)))
    return 0
