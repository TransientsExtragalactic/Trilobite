#cython: language_level=3, boundscheck=False
r"""
Alpha-disk viscous derivative for one-zone accretion disk models (Cython).

Implements the Metzger et al. (2008) boundary-corrected viscous evolution
equations for a thin :math:`\alpha`-disk.  This prescription is independent
of the EOS and opacity used — it requires only the viscous timescale
:math:`t_{\rm visc}` from the closure result and the inner-boundary radius.

Adding a new viscous prescription
-----------------------------------
Add a new ``cdef int`` function with the :c:type:`derivative_func` signature
(defined in ``closure.pxd``) and declare it in ``_viscous.pxd``.  Concrete
closure files in ``models/`` can then ``cimport`` the new derivative.

See Also
--------
:mod:`triceratops.dynamics.accretion.one_zone.closure` :
    ``derivative_func`` typedef and struct definitions.
"""
from libc.math cimport exp, log, sqrt, fmin, fabs
from libc.stdio cimport printf

from ..closure cimport (
    DISK_F0, LOG_G_CGS,
    ClosureResult, DiskDerived, DiskParameters, DiskState, DiskStep,
)


cdef int viscous_derivative_func(
    const DiskState* state,
    const DiskDerived* derived,
    const DiskParameters* params,
    const ClosureResult* closure,
    DiskStep* out,
) nogil:
    r"""Compute the explicit-Euler time step via the Metzger+08 alpha-disk prescription.

    The inner-boundary correction factor

    .. math::

        f_{\rm corr} = \frac{F_0}{1 - \sqrt{R_{\rm in} / R_D}}

    accounts for the zero-torque inner boundary condition.  The resulting
    mass and angular-momentum drain rates are

    .. math::

        \dot{M} = -\frac{M_D\,f_{\rm corr}}{t_{\rm visc}}, \qquad
        \dot{J} = \dot{M}\,\sqrt{G M_{\rm BH} R_{\rm in}}.

    The time step is set to a fraction of the viscous timescale controlled by
    ``params.epsilon``: :math:`\Delta t = \epsilon\,t_{\rm visc}`.

    Parameters
    ----------
    state
        Current disk state.
    derived
        Disk geometry and kinematics.
    params
        Fixed model parameters.
    closure
        Closure result containing ``t_visc``.
    out
        Output struct; ``dM_dt``, ``dJ_dt``, ``dt`` are set.

    Returns
    -------
    int
        Always 0 (SUCCESS).
    """
    cdef double f_corr = DISK_F0 / (1.0 - sqrt(params.R_in / derived.R))

    out.dM_dt += -exp(log(state.M) - log(closure.t_visc) + log(f_corr))
    out.dJ_dt += out.dM_dt * exp(0.5 * (LOG_G_CGS + log(params.MBH) + log(params.R_in)))

    # Adaptive timestep: dt = epsilon * min(dt_M, dt_J, dt_visc), where each
    # partial constraint ensures the state variable changes by at most epsilon
    # fractionally per step and that the viscous timescale is not exceeded.
    cdef double dt_J = fabs(state.J / out.dJ_dt)
    cdef double dt_M = fabs(state.M / out.dM_dt)
    cdef double dt_visc = closure.t_visc

    out.dt = params.epsilon * fmin(fmin(dt_J, dt_M), dt_visc)
    return 0
