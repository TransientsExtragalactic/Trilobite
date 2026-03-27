#cython: language_level=3, boundscheck=False
r"""
``igP`` closure — ideal-gas / full pressure with runtime-configurable opacity (Cython).

Defines the closure function and :class:`OneZoneClosure` subclass for a
one-zone accretion disk governed by combined gas and radiation pressure.
The midplane temperature is found iteratively via bracket expansion + Brent's
method at each step.

The energy balance solved is:

.. math::

    f(\log T_c) = \log Q + 2\log c_s(T_c) - 4\log T_c = 0,

where

.. math::

    Q = \frac{27}{64}\,\frac{\alpha\,\kappa\,\Omega\,\Sigma^2}{\sigma_{\rm SB}},

and :math:`c_s(T_c)` is the combined gas+radiation sound speed from
:func:`~..physics._eos.compute_gas_rad_cs`.  The opacity :math:`\kappa` is
supplied at runtime via ``params.kappa_func``.

Naming convention: ``igP`` = ideal-gas / full (gas + radiation) pressure EOS.
The opacity is no longer encoded in the class name — it is a runtime parameter
set via :meth:`~..closure.OneZoneClosure.set_opacity`.

Physics building blocks used
-----------------------------
* EOS: :func:`~..physics._eos.compute_gas_rad_cs`
* Viscous derivative: :func:`~..physics._viscous.viscous_derivative_func`
* Output writer: :func:`~.._writer.standard_writer_func`

See Also
--------
:class:`~triceratops.dynamics.accretion.one_zone.core.FullPressureDisk` :
    Python-level model class that uses this closure.
:class:`~._gP.gPClosure` :
    Simpler sibling with analytic temperature solve (gas pressure only).
:class:`~._igP_adv.igPAdvClosure` :
    Extended sibling with advective cooling.
"""
from libc.math cimport exp, log, pi, sqrt
from cpython.ref cimport PyObject
from triceratops.radiation.opacity.opacity_base cimport C_GreyOpacityBase

from triceratops.math_utils._bracket_root_finder cimport find_root

from ..closure cimport (
    LOG_K_B_CGS, LOG_KAPPA_ES, LOG_M_P_CGS, LOG_SIGMA_SB_CGS,
    OneZoneClosure,
    ClosureResult, DiskDerived, DiskParameters, DiskState,
)
from .._writer cimport N_RESULT_FIELDS, standard_writer_func
from ..physics._eos cimport compute_gas_rad_cs
from ..physics._viscous cimport viscous_derivative_func
from ..physics._fallback cimport fallback_source_func


# ======================================================== #
#  Root-finding context                                    #
# ======================================================== #

cdef struct igPRootData:
    double log_Q_coef  # log((27/64) alpha Omega Sigma^2 / sigma_SB)  (without kappa)
    double mu          # mean molecular weight (dimensionless)
    double Sigma       # surface density (g cm^-2)
    double Omega       # Keplerian angular velocity (s^-1)
    void*  opacity     # raw PyObject* to a C_GreyOpacityBase instance


cdef int igP_residual(
    double log_T,
    void* user_data,
    double* f_out,
) nogil:
    r"""Log-space residual of the :math:`q^+ = q^-` energy balance.

    .. math::

        f(\log T_c) = \log Q_{\rm coef} + \log\kappa(T_c, \rho) + 2\log c_s(T_c) - 4\log T_c = 0
    """
    cdef igPRootData* d = <igPRootData*>user_data
    cdef double c_s    = compute_gas_rad_cs(exp(log_T), d.mu, d.Sigma, d.Omega)
    cdef double log_cs = log(c_s)
    # log_rho = log_Sigma + log_Omega - log_cs - 0.5*log(2*pi)
    cdef double log_rho   = log(d.Sigma) + log(d.Omega) - log_cs - 0.5 * log(2.0 * pi)
    cdef double log_kappa = (<C_GreyOpacityBase><PyObject*>d.opacity)._log_opacity(log_T, log_rho)
    f_out[0] = d.log_Q_coef + log_kappa + 2.0 * log_cs - 4.0 * log_T
    return 0


# ======================================================== #
#  Closure function                                        #
# ======================================================== #

cdef int igP_closure_func(
    const DiskState* state,
    const DiskDerived* derived,
    const DiskParameters* params,
    const ClosureResult* prev,
    ClosureResult* out,
) nogil:
    r"""Compute thermodynamic quantities for the full-pressure closure.

    The midplane temperature :math:`T_c` is found by solving
    :func:`igP_residual` using bracket expansion + Brent's method.
    The previous step's temperature is used as the warm-start guess;
    the gas-only analytic temperature is the cold-start fallback.

    Returns
    -------
    int
        0 on SUCCESS; non-zero propagates the :func:`find_root` error code.
    """
    cdef double log_Q_coef, log_T_guess, log_T, log_rho, log_kappa, c_s
    cdef igPRootData root_data
    cdef int status

    log_Q_coef = (
        log(27.0 / 128.0)
        + log(params.alpha)
        + 2.0 * log(derived.Sigma)
        + log(derived.Omega)
        - LOG_SIGMA_SB_CGS
    )

    if prev.log_T_c != 0.0:
        log_T_guess = prev.log_T_c
    else:
        # Gas-only analytic temperature as a cold-start seed (uses ES as proxy).
        log_T_guess = (1.0 / 3.0) * (
              log(27.0 / 128.0)
            + LOG_K_B_CGS
            + log(params.alpha)
            + LOG_KAPPA_ES
            - LOG_M_P_CGS
            - log(params.mu)
            - LOG_SIGMA_SB_CGS
            + log(derived.Omega)
            + 2.0 * log(derived.Sigma)
        )

    root_data.log_Q_coef = log_Q_coef
    root_data.mu         = params.mu
    root_data.Sigma      = derived.Sigma
    root_data.Omega      = derived.Omega
    root_data.opacity    = params.opacity

    status = find_root(
        igP_residual, &root_data,
        log_T_guess,
        0.01,   # step: initial half-width of bracket in log-T
        2.0,    # grow_factor
        2,      # max_expand
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
#  Closure class                                           #
# ======================================================== #

cdef class igPClosure(OneZoneClosure):
    """
    ``igP`` closure — full pressure (gas + radiation) with runtime opacity.

    Assembles :func:`igP_closure_func`,
    :func:`~..physics._viscous.viscous_derivative_func`, and
    :func:`~.._writer.standard_writer_func` into a runnable
    :class:`~..closure.OneZoneClosure`.

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
        self._closure_fn     = igP_closure_func
        self._derivative_fn  = viscous_derivative_func
        self._writer_fn      = standard_writer_func
        self.n_result_fields = N_RESULT_FIELDS
        # Default to ES opacity.
        self.opacity = ElectronScatteringOpacity()
        if with_fallback:
            self._source_fn = fallback_source_func
        else:
            self._source_fn = NULL
