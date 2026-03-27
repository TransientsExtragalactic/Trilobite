#cython: language_level=3, boundscheck=False
r"""
``igP_adv`` closure — ideal-gas / full pressure with advective cooling and runtime opacity (Cython).

Defines the closure function and :class:`OneZoneClosure` subclass for a
one-zone accretion disk governed by combined gas and radiation pressure,
including an advective cooling term.

The energy balance is

.. math::

    q_{\rm visc} = q_{\rm rad} + q_{\rm adv},

Normalising by :math:`q_{\rm visc}` and solving for :math:`T_c` yields the
dimensionless residual

.. math::

    f(\log T_c) = 1 - A\,c_s^{-2}\,T_c^4 - B\,c_s^2 = 0,

where:

.. math::

    A &= \frac{128}{27}\,\frac{\sigma_{\rm SB}}{\kappa\,\alpha\,\Omega\,\Sigma^2},\\
    B &= \frac{4}{9\pi}\,\xi\,F_0\,\alpha\,\frac{M_D}{R_D^4\,\Omega^2\,\Sigma}.

The opacity :math:`\kappa` is supplied at runtime via ``params.kappa_func``
and is re-evaluated at each temperature trial inside the residual.

The parameter :math:`\xi` (``params.extra[0]``) is the entropy gradient
parameter.  Setting :math:`\xi \to 0` recovers the non-advective ``igP``
limit.

Naming convention: ``igP`` = ideal-gas / full (gas + radiation) pressure EOS,
``adv`` = advective cooling.

Physics building blocks used
-----------------------------
* EOS: :func:`~..physics._eos.compute_gas_rad_cs`
* Viscous derivative: :func:`~..physics._viscous.viscous_derivative_func`
* Output writer: :func:`igP_adv_writer_func` (custom 21-field writer)

Extra parameters
----------------
``params.extra[0]``: ``xi`` — entropy gradient parameter (dimensionless, > 0).

See Also
--------
:class:`~triceratops.dynamics.accretion.one_zone.core.AdvectiveDisk` :
    Python-level model class that uses this closure.
:class:`~._igP.igPClosure` :
    Non-advective sibling closure.
"""
from libc.math cimport exp, log, pi, sqrt
from cpython.ref cimport PyObject
from triceratops.radiation.opacity.opacity_base cimport C_GreyOpacityBase

from triceratops.math_utils._bracket_root_finder cimport find_root

from ..closure cimport (
    LOG_DISK_F0, LOG_K_B_CGS, LOG_KAPPA_ES, LOG_M_P_CGS, LOG_SIGMA_SB_CGS,
    OneZoneClosure,
    ClosureResult, DiskDerived, DiskParameters, DiskState, DiskStep,
)
from ..physics._eos cimport compute_gas_rad_cs
from ..physics._viscous cimport viscous_derivative_func
from ..physics._fallback cimport fallback_source_func


# ======================================================== #
#  Root-finding context                                    #
# ======================================================== #

cdef struct igP_advRootData:
    # log_A_base = log(128/27) + LOG_SIGMA_SB - log(alpha) - log(Omega) - 2*log(Sigma)
    # Full log_A = log_A_base - log_kappa (kappa in denominator, re-evaluated per trial T).
    double log_A_base
    double log_B      # log((4/9pi) xi F0 alpha M / (R^4 Omega^2 Sigma))
    double mu         # mean molecular weight (dimensionless)
    double Sigma      # surface density (g cm^-2)
    double Omega      # Keplerian angular velocity (s^-1)
    void*  opacity    # raw PyObject* to a C_GreyOpacityBase instance


cdef int igP_adv_residual(
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

    where :math:`A` is re-evaluated at the current :math:`(T_c, \rho)`.
    """
    cdef igP_advRootData* d = <igP_advRootData*>user_data
    cdef double c_s    = compute_gas_rad_cs(exp(log_T), d.mu, d.Sigma, d.Omega)
    cdef double log_cs = log(c_s)
    cdef double log_rho = log(d.Sigma) + log(d.Omega) - log_cs - 0.5 * log(2.0 * pi)
    cdef double log_kappa = (<C_GreyOpacityBase><PyObject*>d.opacity)._log_opacity(log_T, log_rho)
    cdef double log_A_full = d.log_A_base - log_kappa
    cdef double term1 = log_A_full - 2.0 * log_cs + 4.0 * log_T  # log(q_rad / q_visc)
    cdef double term2 = d.log_B + 2.0 * log_cs                   # log(q_adv / q_visc)
    f_out[0] = 1.0 - exp(term1) - exp(term2)
    return 0


# ======================================================== #
#  Closure function                                        #
# ======================================================== #

cdef int igP_adv_closure_func(
    const DiskState* state,
    const DiskDerived* derived,
    const DiskParameters* params,
    const ClosureResult* prev,
    ClosureResult* out,
) nogil:
    r"""Compute thermodynamic quantities for the ``igP_adv`` closure.

    Solves :func:`igP_adv_residual` using bracket expansion + Brent's method.
    The previous step's temperature is used as the warm-start guess.

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
    cdef double log_A_base, log_B, log_T_guess, log_T, log_rho, log_kappa, c_s, xi
    cdef igP_advRootData root_data
    cdef int status
    xi = params.extra[0]

    log_A_base = (
        log(32.0 / 27.0)
        + LOG_SIGMA_SB_CGS
        - log(params.alpha)
        - log(derived.Omega)
        - 2.0 * log(derived.Sigma)
    )
    log_B = (
        log(4.0 / 3.0)
        + log(xi)
        - 2.0 * log(derived.R)
        - 2.0 * log(derived.Omega)
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

    root_data.log_A_base = log_A_base
    root_data.log_B      = log_B
    root_data.mu         = params.mu
    root_data.Sigma      = derived.Sigma
    root_data.Omega      = derived.Omega
    root_data.opacity    = params.opacity

    status = find_root(
        igP_adv_residual, &root_data,
        log_T_guess,
        0.05,             # step: initial half-width of bracket in log-T
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

    log_rho = log(derived.Sigma) + log(derived.Omega) - out.log_cs - 0.5 * log(2.0 * pi)
    out.log_rho = log_rho

    log_kappa = (<C_GreyOpacityBase><PyObject*>params.opacity)._log_opacity(log_T, log_rho)

    out.log_tau = log(0.5) + log_kappa + log(derived.Sigma)

    out.log_T_eff = log_T + 0.25 * (log(8.0 / 3.0) - log_kappa - log(derived.Sigma))

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
DEF _ADV_N_RESULT_FIELDS = 21
cdef int ADV_N_RESULT_FIELDS = _ADV_N_RESULT_FIELDS
ADV_N_RESULT_FIELDS_PY = _ADV_N_RESULT_FIELDS


cdef int igP_adv_writer_func(
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

cdef class igPAdvClosure(OneZoneClosure):
    """
    ``igP_adv`` closure — full pressure (gas + radiation) + advective cooling.

    Assembles :func:`igP_adv_closure_func`,
    :func:`~..physics._viscous.viscous_derivative_func`, and
    :func:`igP_adv_writer_func` into a runnable
    :class:`~..closure.OneZoneClosure`.

    Requires ``params.extra[0]`` = ``xi`` (entropy gradient parameter, > 0).
    Set ``n_result_fields = 21`` (one more than the standard writer).

    Defaults to electron-scattering opacity.  Set ``closure.opacity`` to
    use a different opacity law.

    Parameters
    ----------
    with_fallback : bool, optional
        If ``True``, install :func:`~..physics._fallback.fallback_source_func`
        as the source term.  Default ``False``.
    """

    def __cinit__(self, bint with_fallback=False):
        from triceratops.radiation.opacity.models.core import ElectronScatteringOpacity
        self._closure_fn     = igP_adv_closure_func
        self._derivative_fn  = viscous_derivative_func
        self._writer_fn      = igP_adv_writer_func
        self.n_result_fields = ADV_N_RESULT_FIELDS
        # Default to ES opacity.
        self.opacity = ElectronScatteringOpacity()

        if with_fallback:
            self._source_fn = fallback_source_func
        else:
            self._source_fn = NULL
