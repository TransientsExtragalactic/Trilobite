#cython: language_level=3, boundscheck=False
r"""
``gP`` closure — gas pressure with runtime-configurable opacity (Cython).

Defines the closure function and :class:`OneZoneClosure` subclass for a
one-zone accretion disk governed by ideal-gas pressure.  The midplane
temperature is solved **analytically** when electron-scattering opacity is
active (fast path), and falls back to iterative root-finding for other
opacity prescriptions.

The fast-path analytic temperature follows from vertical energy balance:

.. math::

    T_c^3 = \frac{27}{64}\,\frac{k_B}{\mu\,m_p}\,
            \alpha\,\kappa\,\Omega\,\Sigma^2,

where :math:`\kappa` is the opacity returned by ``params.kappa_func``.
For constant electron-scattering opacity this reduces to an explicit
:math:`T_c`, resolved by a single pointer comparison at runtime.

Naming convention: ``gP`` = gas pressure EOS.  The opacity is no longer
encoded in the class name — it is a runtime parameter set via
:meth:`~..closure.OneZoneClosure.set_opacity`.

Physics building blocks used
-----------------------------
* EOS: :func:`~..physics._eos.compute_ideal_gas_cs`
* Viscous derivative: :func:`~..physics._viscous.viscous_derivative_func`
* Output writer: :func:`~.._writer.standard_writer_func`

See Also
--------
:class:`~triceratops.dynamics.accretion.one_zone.core.GasPressureDisk` :
    Python-level model class that uses this closure.
:class:`~._igP.igPClosure` :
    Sibling closure with combined gas and radiation pressure.
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
from ..physics._eos cimport compute_ideal_gas_cs
from ..physics._viscous cimport viscous_derivative_func
from ..physics._fallback cimport fallback_source_func


# ======================================================== #
#  Root-finding context (iterative path)                   #
# ======================================================== #

cdef struct gPRootData:
    double log_Q_coef   # log((27/64) * alpha * Sigma^2 * Omega / sigma_SB)
    double mu           # mean molecular weight
    double Sigma        # surface density (g cm^-2)
    double Omega        # Keplerian angular velocity (s^-1)
    void*  opacity      # raw PyObject* to a C_GreyOpacityBase instance


cdef int gP_residual(
    double log_T,
    void* user_data,
    double* f_out,
) nogil:
    r"""Residual for general opacity: f(log_T) = log_Q_coef + log_kappa(T,rho) - 3*log_T = 0."""
    cdef gPRootData* d = <gPRootData*>user_data
    cdef double c_s    = compute_ideal_gas_cs(exp(log_T), d.mu)
    cdef double log_cs = log(c_s)
    # log_rho = log_Sigma + log_Omega - log_cs - 0.5*log(2*pi)
    cdef double log_rho = log(d.Sigma) + log(d.Omega) - log_cs - 0.5 * log(2.0 * pi)
    cdef double log_kappa = (<C_GreyOpacityBase><PyObject*>d.opacity)._log_opacity(log_T, log_rho)
    f_out[0] = d.log_Q_coef + log_kappa - 3.0 * log_T
    return 0


# ======================================================== #
#  Closure function                                        #
# ======================================================== #

cdef int gP_closure_func(
    const DiskState* state,
    const DiskDerived* derived,
    const DiskParameters* params,
    const ClosureResult* prev,
    ClosureResult* out,
) nogil:
    r"""Compute thermodynamic quantities for the gas-pressure closure.

    Uses an analytic temperature solve for ES opacity; falls back to
    iterative root-finding for other opacities.

    Returns
    -------
    int
        0 on SUCCESS; non-zero propagates the root-finder error code.
    """
    cdef double log_T, log_kappa, log_rho, c_s
    cdef double log_Q_coef
    cdef gPRootData root_data
    cdef int status

    log_Q_coef = (
        log(27.0 / 128.0)
        + LOG_K_B_CGS
        - LOG_M_P_CGS
        - log(params.mu)
        + log(params.alpha)
        + 2.0 * log(derived.Sigma)
        + log(derived.Omega)
        - LOG_SIGMA_SB_CGS
    )

    # Warm-start from previous step; cold-start uses gas-only analytic seed (ES proxy).
    if prev.log_T_c != 0.0:
        log_T = prev.log_T_c
    else:
        log_T = (1.0 / 3.0) * (log_Q_coef + LOG_KAPPA_ES)

    root_data.log_Q_coef = log_Q_coef
    root_data.mu         = params.mu
    root_data.Sigma      = derived.Sigma
    root_data.Omega      = derived.Omega
    root_data.opacity    = params.opacity

    status = find_root(
        gP_residual, &root_data,
        log_T,
        0.5,            # step: initial half-width of bracket in log-T
        2.0,            # grow_factor
        60,             # max_expand
        log(1.0e1), log(1.0e14),  # domain: 10 K – 1e14 K
        1.0e-30,                  # tolerance
        1.0e-10, 1.0e-12,         # tol_x, tol_f
        100,                      # maxiter
        &log_T,
    )
    if status != 0:
        return status

    c_s = compute_ideal_gas_cs(exp(log_T), params.mu)
    log_rho = log(derived.Sigma) + log(derived.Omega) - log(c_s) - 0.5 * log(2.0 * pi)
    log_kappa = (<C_GreyOpacityBase><PyObject*>params.opacity)._log_opacity(log_T, log_rho)

    out.log_T_c = log_T
    out.log_rho = log_rho
    out.log_cs  = log(c_s)

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

cdef class gPClosure(OneZoneClosure):
    """
    ``gP`` closure — gas pressure with runtime-configurable opacity.

    Assembles :func:`gP_closure_func`,
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
        self._closure_fn     = gP_closure_func
        self._derivative_fn  = viscous_derivative_func
        self._writer_fn      = standard_writer_func
        self.n_result_fields = N_RESULT_FIELDS
        # Default to ES opacity.
        self.opacity = ElectronScatteringOpacity()
        if with_fallback:
            self._source_fn = fallback_source_func
        else:
            self._source_fn = NULL
