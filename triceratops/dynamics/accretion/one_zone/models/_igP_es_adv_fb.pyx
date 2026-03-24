#cython: language_level=3, boundscheck=False
r"""
``igP_es_adv_fb`` closure — ideal-gas / full pressure / ES + advection + fallback supply (Cython).

Wires :func:`~._igP_es_adv.igP_es_adv_closure_func` and
:func:`~._igP_es_adv.igP_es_adv_writer_func` together with a power-law
fallback mass supply source term.

Naming convention: ``igP`` = ideal-gas / full (gas + radiation) pressure EOS,
``es`` = electron-scattering opacity, ``adv`` = advective cooling,
``fb`` = fallback mass supply.

Extra-parameter layout (``params.extra``)
-----------------------------------------
=======  =========  =====================================================
Index    Symbol     Description
=======  =========  =====================================================
0        xi         Entropy gradient parameter (dimensionless, > 0)
1        M_fb_0     Fallback rate at ``t_fb`` (g s⁻¹)
2        t_fb       Reference (peak) time (s)
3        beta_fb    Power-law index (dimensionless; typically 5/3)
=======  =========  =====================================================

``xi`` occupies ``extra[0]`` because the advective closure function reads
its entropy gradient parameter from that slot.  The fallback parameters
follow at ``extra[1..3]``.

See Also
--------
:class:`~triceratops.dynamics.accretion.one_zone.core.igP_es_adv_fbDisk` :
    Python-level model class that uses this closure.
:class:`~._igP_es_adv.igP_es_advClosure` :
    Base advective closure (no fallback supply).
"""
from libc.math cimport exp, log, pow

from ..closure cimport (
    LOG_G_CGS,
    OneZoneClosure,
    ClosureResult, DiskDerived, DiskParameters, DiskState, DiskStep,
)
from ..physics._viscous cimport viscous_derivative_func
from ._igP_es_adv cimport (
    ADV_N_RESULT_FIELDS,
    igP_es_adv_closure_func,
    igP_es_adv_writer_func,
)


cdef int igP_es_adv_fallback_source_func(
    const DiskState* state,
    const DiskDerived* derived,
    const DiskParameters* params,
    const ClosureResult* closure,
    DiskStep* step,
) nogil:
    r"""Add a power-law fallback mass supply to the current time step.

    Reads from ``params.extra[1..3]`` (``extra[0]`` = xi is reserved for the
    advective closure):

    =========  =========  ======================
    Index      Symbol     Description
    =========  =========  ======================
    1          M_fb_0     Fallback rate (g s⁻¹)
    2          t_fb       Reference time (s)
    3          beta_fb    Power-law index
    =========  =========  ======================

    The inflow is

    .. math::

        \dot{M}_{\rm fb}(t) = M_{\rm fb,0}
            \left(\frac{t}{t_{\rm fb}}\right)^{-\beta_{\rm fb}},

    circularising at :math:`R_D` to deposit angular momentum
    :math:`\dot{J}_{\rm fb} = \dot{M}_{\rm fb}\sqrt{G M_{\rm BH} R_D}`.

    Returns
    -------
    int
        Always 0 (SUCCESS).
    """
    cdef double M_fb_0  = params.extra[1]
    cdef double t_fb    = params.extra[2]
    cdef double beta_fb = params.extra[3]
    cdef double mdot_fb = M_fb_0 * pow(state.t / t_fb, -beta_fb)

    step.dM_dt += mdot_fb
    step.dJ_dt += mdot_fb * exp(0.5 * (LOG_G_CGS + log(params.MBH) + log(derived.R)))
    return 0


cdef class igP_es_adv_fbClosure(OneZoneClosure):
    """
    ``igP_es_adv_fb`` closure — ``igP_es_adv`` + power-law fallback mass supply.

    Assembles :func:`~._igP_es_adv.igP_es_adv_closure_func`,
    :func:`~..physics._viscous.viscous_derivative_func`,
    :func:`~._igP_es_adv.igP_es_adv_writer_func`, and
    :func:`igP_es_adv_fallback_source_func` into a runnable
    :class:`~..closure.OneZoneClosure`.

    Extra-parameter layout (``params.extra``):
        [0] xi        — entropy gradient parameter (dimensionless, > 0)
        [1] M_fb_0    — fallback rate at t_fb (g s⁻¹)
        [2] t_fb      — reference time (s)
        [3] beta_fb   — power-law index (dimensionless)
    """

    def __cinit__(self):
        self._closure_fn     = igP_es_adv_closure_func
        self._derivative_fn  = viscous_derivative_func
        self._writer_fn      = igP_es_adv_writer_func
        self._source_fn      = igP_es_adv_fallback_source_func
        self.n_result_fields = ADV_N_RESULT_FIELDS
