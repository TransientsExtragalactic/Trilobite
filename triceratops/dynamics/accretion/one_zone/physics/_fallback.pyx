#cython: language_level=3, boundscheck=False
r"""
Fallback debris-stream source term (Cython).

Implements a power-law mass supply rate representing the fallback of tidally
disrupted debris onto the accretion disk, as in tidal disruption event (TDE)
models.

Extra-parameter layout (``params.extra``, last 3 indices)
----------------------------------------------------------
The fallback parameters always occupy the **last three** slots of
``params.extra``, regardless of how many preceding extra parameters exist
(e.g. the advective entropy gradient ``xi``):

=======  =========  ========================================
Index    Symbol     Description
=======  =========  ========================================
n_extra-3  M_fb_0   Fallback rate at ``t_fb`` (g s⁻¹)
n_extra-2  t_fb     Reference (peak) time (s)
n_extra-1  beta_fb  Power-law index (dimensionless; typically 5/3)
=======  =========  ========================================

This convention accommodates both non-advective closures (where
``extra = [M_fb_0, t_fb, beta_fb]``, n_extra=3, reads ``[0,1,2]``) and
advective closures (where ``extra = [xi, M_fb_0, t_fb, beta_fb]``,
n_extra=4, reads ``[1,2,3]``).

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

    Reads the fallback parameters from the **last three** elements of
    ``params.extra`` (indices ``n_extra-3``, ``n_extra-2``, ``n_extra-1``).
    This works for both non-advective closures (n_extra=3) and advective
    closures (n_extra=4, where ``extra[0]`` = ``xi`` is reserved).

    Parameters
    ----------
    state
        Current disk state; ``state.t`` is the current time in seconds.
    derived
        Derived quantities; ``derived.R`` is the disk outer radius (cm).
    params
        Model parameters.  ``params.extra`` must point to at least 3 elements
        with ``[M_fb_0 (g s⁻¹), t_fb (s), beta_fb]`` in the last three slots.
    closure
        Thermodynamic closure result (not used here).
    step
        Time-step struct modified in-place.

    Returns
    -------
    int
        Always 0 (SUCCESS).
    """
    cdef int n = params.n_extra
    cdef double M_fb_0  = params.extra[n - 3]
    cdef double t_fb    = params.extra[n - 2]
    cdef double beta_fb = params.extra[n - 1]
    cdef double mdot_fb = M_fb_0 * pow(state.t / t_fb, -beta_fb)

    step.dM_dt += mdot_fb
    step.dJ_dt += mdot_fb * exp(0.5 * (LOG_G_CGS + log(params.MBH) + log(derived.R)))
    return 0
