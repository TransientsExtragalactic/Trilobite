#cython: language_level=3, boundscheck=False
r"""
``igP_es_adv`` closure — ideal-gas / full pressure / electron scattering + advection (Cython).

Defines the closure function and :class:`OneZoneClosure` subclass for a
one-zone accretion disk governed by combined gas and radiation pressure with
electron-scattering opacity, including an advective cooling term.

The standard ``igP_es`` closure assumes all viscous heating is radiated locally
(``q_visc = q_rad``).  This closure extends the energy balance to

.. math::

    q_{\rm visc} = q_{\rm rad} + q_{\rm adv},

where the advective term carries a fraction of the heating inward with the
accretion flow.  Normalising by :math:`q_{\rm visc}` and solving for
:math:`T_c` yields the dimensionless residual

.. math::

    f(\log T_c) = 1 - A\,c_s^{-2}\,T_c^4 - B\,c_s^2 = 0,

where:

.. math::

    A &= \frac{64}{27}\,\frac{\sigma_{\rm SB}}{\kappa_{\rm es}\,\alpha\,\Omega\,\Sigma^2},\\
    B &= \frac{4}{9\pi}\,\xi\,F_0\,\alpha\,\frac{M_D}{R_D^4\,\Omega^2\,\Sigma}.

The parameter :math:`\xi` (``xi``, ``params.extra[0]``) is the entropy
gradient parameter controlling the strength of advective cooling.  Setting
:math:`\xi \to 0` recovers the non-advective ``igP_es`` limit.

Naming convention: ``igP`` = ideal-gas / full (gas + radiation) pressure EOS,
``es`` = electron-scattering opacity, ``adv`` = advective cooling.

Physics building blocks used
-----------------------------
* EOS: :func:`~..physics._eos.compute_gas_rad_cs`
* Viscous derivative: :func:`~..physics._viscous.viscous_derivative_func`
* Output writer: :func:`igP_es_adv_writer_func` (custom 21-field writer)

Extra parameters
----------------
``params.extra[0]``: ``xi`` — entropy gradient parameter (dimensionless, > 0).

See Also
--------
:class:`~triceratops.dynamics.accretion.one_zone.core.igP_es_advDisk` :
    Python-level model class that uses this closure.
:class:`~._igP_es.igP_esClosure` :
    Non-advective sibling closure.
"""
from libc.math cimport exp, log, pi, sqrt

from triceratops.math_utils._bracket_root_finder cimport find_root

from ..closure cimport (
    LOG_DISK_F0, LOG_K_B_CGS, LOG_KAPPA_ES, LOG_M_P_CGS, LOG_SIGMA_SB_CGS,
    OneZoneClosure,
    ClosureResult, DiskDerived, DiskParameters, DiskState, DiskStep,
)
from ..physics._eos cimport compute_gas_rad_cs
from ..physics._viscous cimport viscous_derivative_func


# ======================================================== #
#  Root-finding context                                    #
# ======================================================== #

cdef struct igP_es_advRootData:
    double log_A  # log((64/27) sigma_SB / (kappa_es alpha Omega Sigma^2))
    double log_B  # log((4/9pi) xi F0 alpha M / (R^3 Omega Sigma))
    double mu     # mean molecular weight (dimensionless)
    double Sigma  # surface density (g cm^-2)
    double Omega  # Keplerian angular velocity (s^-1)


cdef inline double logsumexp(double a, double b) nogil:
    """Numerically stable log(exp(a) + exp(b))."""
    if a > b:
        return a + log(1.0 + exp(b - a))
    else:
        return b + log(1.0 + exp(a - b))


cdef int igP_es_adv_residual(
    double log_T,
    void* user_data,
    double* f_out,
) nogil:
    r"""Residual of the normalised energy balance (see module docstring).

    .. math::

        f(\log T_c) = 1
            - A\,c_s^{-2}(T_c)\,T_c^4
            - B\,c_s^2(T_c)
        = 0

    Evaluating in log-space avoids catastrophic cancellation when either
    term dominates.

    Returns
    -------
    int
        Always 0 (SUCCESS); non-zero would signal a NaN/Inf in the residual.
    """
    cdef igP_es_advRootData* d = <igP_es_advRootData*>user_data
    cdef double c_s   = compute_gas_rad_cs(exp(log_T), d.mu, d.Sigma, d.Omega)
    cdef double log_cs = log(c_s)
    cdef double term1 = d.log_A - 2.0 * log_cs + 4.0 * log_T  # log(q_rad / q_visc)
    cdef double term2 = d.log_B + 2.0 * log_cs                 # log(q_adv / q_visc)
    f_out[0] = 1.0 - exp(term1) - exp(term2)
    return 0


# ======================================================== #
#  Closure function                                        #
# ======================================================== #

cdef int igP_es_adv_closure_func(
    const DiskState* state,
    const DiskDerived* derived,
    const DiskParameters* params,
    const ClosureResult* prev,
    ClosureResult* out,
) nogil:
    r"""Compute thermodynamic quantities for the ``igP_es_adv`` closure.

    Solves the energy balance :math:`f(\log T_c) = 0` (see module docstring)
    using bracket expansion + Brent's method.  The previous step's temperature
    is used as the warm-start guess; the gas-only analytic temperature is the
    cold-start fallback.

    Parameters
    ----------
    state
        Current disk state (M, J, t).
    derived
        Geometry/kinematics (R, Sigma, Omega).
    params
        Fixed model parameters.  ``params.extra[0]`` must be ``xi``.
    prev
        Previous-step closure result (warm-start source for ``log_T_c``).
    out
        Closure result to populate.

    Returns
    -------
    int
        0 on success; non-zero propagates the :func:`find_root` error code.
    """
    cdef double log_A, log_B, log_T_guess, log_T, c_s
    cdef igP_es_advRootData root_data
    cdef int status
    cdef double xi = params.extra[0]

    log_A = (
        log(64.0 / 27.0)
        + LOG_SIGMA_SB_CGS
        - LOG_KAPPA_ES
        - log(params.alpha)
        - log(derived.Omega)
        - 2.0 * log(derived.Sigma)
    )
    log_B = (
        log(4.0 / (9.0 * pi))
        + log(xi)
        + LOG_DISK_F0
        + log(params.alpha)
        + log(state.M)
        - 4.0 * log(derived.R)
        - 2.0 * log(derived.Omega)
        - log(derived.Sigma)
    )

    if prev.log_T_c != 0.0:
        log_T_guess = prev.log_T_c
    else:
        # Gas-only analytic temperature as a cold-start seed.
        log_T_guess = (1.0 / 3.0) * (
              log(27.0 / 64.0)
            + LOG_K_B_CGS
            + log(params.alpha)
            + LOG_KAPPA_ES
            - LOG_M_P_CGS
            - log(params.mu)
            - LOG_SIGMA_SB_CGS
            + log(derived.Omega)
            + 2.0 * log(derived.Sigma)
        )

    root_data.log_A = log_A
    root_data.log_B = log_B
    root_data.mu    = params.mu
    root_data.Sigma = derived.Sigma
    root_data.Omega = derived.Omega

    status = find_root(
        igP_es_adv_residual, &root_data,
        log_T_guess,
        0.5,             # step: initial half-width of bracket in log-T
        2.0,             # grow_factor
        60,              # max_expand
        log(1.0e1), log(1.0e14),  # domain: 10 K – 1e14 K
        1.0e-30,                  # tolerance
        1.0e-10, 1.0e-12,         # tol_x, tol_f
        100,                      # maxiter
        &log_T,
    )
    if status != 0:
        return status

    out.log_T_c = log_T

    c_s = compute_gas_rad_cs(exp(log_T), params.mu, derived.Sigma, derived.Omega)
    out.log_cs = log(c_s)

    out.log_tau = log(0.5) + LOG_KAPPA_ES + log(derived.Sigma)

    out.log_T_eff = log_T + 0.25 * (log(8.0 / 3.0) - LOG_KAPPA_ES - log(derived.Sigma))

    out.log_nu = log(params.alpha) + 2.0 * out.log_cs - log(derived.Omega)

    out.log_q_visc = (
        log(9.0 / 8.0) + out.log_nu + log(derived.Sigma) + 2.0 * log(derived.Omega)
    )

    out.t_visc = exp(2.0 * log(derived.R) - out.log_nu)
    return 0


# ======================================================== #
#  Writer                                                  #
# ======================================================== #

# 21-field layout: inserts q_adv at row 13, shifting dM_dt…rho by one row.
# Differs from the standard 20-field writer; closures using this writer must
# set n_result_fields = ADV_N_RESULT_FIELDS in __cinit__.
DEF _ADV_N_RESULT_FIELDS = 21
cdef int ADV_N_RESULT_FIELDS = _ADV_N_RESULT_FIELDS
ADV_N_RESULT_FIELDS_PY = _ADV_N_RESULT_FIELDS


cdef int igP_es_adv_writer_func(
    const int step_index,
    const DiskState* state,
    const DiskDerived* derived,
    const DiskParameters* params,
    const ClosureResult* closure,
    const DiskStep* step,
    double* result_array,
    int n_steps,
) nogil:
    r"""Write one time step of output (21 fields) into ``result_array``.

    Extends the standard 20-field layout by inserting ``q_adv`` at row 13.
    ``q_adv`` is derived from the solved temperature via:

    .. math::

        q_{\rm adv} = q_{\rm visc}\,B\,c_s^2,
        \quad
        B = \frac{4}{9\pi}\,\xi\,F_0\,\alpha\,
            \frac{M_D}{R_D^4\,\Omega^2\,\Sigma}.

    Field layout (``ADV_N_RESULT_FIELDS = 21``)
    -------------------------------------------
    =====  ============  =================
    Index  Quantity      Units
    =====  ============  =================
    0      step index    —
    1      t             s
    2      M_D           g
    3      J_D           g cm² s⁻¹
    4      R_D           cm
    5      Σ             g cm⁻²
    6      Ω             s⁻¹
    7      T_eff         K
    8      T_c           K
    9      τ             —
    10     c_s           cm s⁻¹
    11     ν             cm² s⁻¹
    12     q_visc        erg cm⁻² s⁻¹
    13     q_adv         erg cm⁻² s⁻¹  ← new
    14     dM/dt         g s⁻¹
    15     dJ/dt         g cm² s⁻²
    16     dt            s
    17     t_visc        s
    18     H             cm
    19     H/R           —
    20     ρ             g cm⁻³
    =====  ============  =================
    """
    # Recompute log_B from params to derive q_adv = q_visc * B * c_s^2.
    cdef double xi = params.extra[0]
    cdef double log_B = (
        log(4.0 / (9.0 * pi))
        + log(xi)
        + LOG_DISK_F0
        + log(params.alpha)
        + log(state.M)
        - 4.0 * log(derived.R)
        - 2.0 * log(derived.Omega)
        - log(derived.Sigma)
    )
    cdef double log_q_adv = closure.log_q_visc + log_B + 2.0 * closure.log_cs
    cdef double H   = exp(closure.log_cs - log(derived.Omega))
    cdef double rho = derived.Sigma / (sqrt(2.0 * pi) * H)

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
    result_array[13 * n_steps + step_index] = exp(log_q_adv)
    result_array[14 * n_steps + step_index] = step.dM_dt
    result_array[15 * n_steps + step_index] = step.dJ_dt
    result_array[16 * n_steps + step_index] = step.dt
    result_array[17 * n_steps + step_index] = closure.t_visc
    result_array[18 * n_steps + step_index] = H
    result_array[19 * n_steps + step_index] = H / derived.R
    result_array[20 * n_steps + step_index] = rho
    return 0


# ======================================================== #
#  Closure class                                           #
# ======================================================== #

cdef class igP_es_advClosure(OneZoneClosure):
    """
    ``igP_es_adv`` closure — ideal-gas/full pressure / ES + advective cooling.

    Assembles :func:`igP_es_adv_closure_func`,
    :func:`~..physics._viscous.viscous_derivative_func`, and
    :func:`igP_es_adv_writer_func` into a runnable
    :class:`~..closure.OneZoneClosure`.

    Requires ``params.extra[0]`` = ``xi`` (entropy gradient parameter, > 0).
    Set ``n_result_fields = 21`` (one more than the standard writer).
    """

    def __cinit__(self):
        self._closure_fn     = igP_es_adv_closure_func
        self._derivative_fn  = viscous_derivative_func
        self._writer_fn      = igP_es_adv_writer_func
        self.n_result_fields = ADV_N_RESULT_FIELDS
