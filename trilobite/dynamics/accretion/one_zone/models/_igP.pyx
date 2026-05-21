#cython: language_level=3, boundscheck=False
r"""
``igP`` / ``FullPressureClosure`` — full-pressure closure with runtime opacity (Cython).

Defines the closure function and :class:`FullPressureClosure` for one-zone
accretion disk models.  :class:`FullPressureClosure` is the single concrete
class that covers both gas-pressure-only and combined gas+radiation pressure
physics:

* ``gas_pressure_only=True``  → uses :func:`gP_closure_func` (analytic-seed
  iterative solve using the ideal-gas EOS).
* ``gas_pressure_only=False`` → uses :func:`igP_closure_func` (iterative solve
  using the combined gas+radiation EOS).

In both modes the same derivative, writer, and (optional) fallback source term
are used, so a single class replaces what were formerly two separate classes
(``gPClosure`` and ``igPClosure``).

The energy balance solved by :func:`igP_closure_func` is:

.. math::

    f(\log T_c) = \log Q + 2\log c_s(T_c) - 4\log T_c = 0,

where

.. math::

    Q = \frac{27}{64}\,\frac{\alpha\,\kappa\,\Omega\,\Sigma^2}{\sigma_{\rm SB}},

and :math:`c_s(T_c)` is the combined gas+radiation sound speed from
:func:`~..physics._eos.compute_gas_rad_cs`.

Physics building blocks used
-----------------------------
* EOS: :func:`~..physics._eos.compute_ideal_gas_cs` (gP path),
  :func:`~..physics._eos.compute_gas_rad_cs` (igP path)
* Viscous derivative: :func:`~..physics._viscous.viscous_derivative_func`
* Output writer: :func:`~.._writer.standard_writer_func`

See Also
--------
:class:`~trilobite.dynamics.accretion.one_zone.core.GasPressureDisk` :
    Python-level model that selects ``gas_pressure_only=True``.
:class:`~trilobite.dynamics.accretion.one_zone.core.FullPressureDisk` :
    Python-level model that selects ``gas_pressure_only=False``.
:class:`~._igP_adv.AdvectiveClosure` :
    Extended closure with advective cooling.
"""
from libc.math cimport exp, log, pi, sqrt
from cpython.ref cimport PyObject
from trilobite.radiation.opacity.opacity_base cimport C_GreyOpacityBase

from trilobite.math_utils._bracket_root_finder cimport find_root

from ..closure cimport (
    LOG_K_B_CGS, LOG_KAPPA_ES, LOG_M_P_CGS, LOG_SIGMA_SB_CGS,
    FallbackParams,
    OneZoneClosure,
    ClosureResult, DiskDerived, DiskParameters, DiskState,
)
from .._writer cimport N_RESULT_FIELDS, standard_writer_func
from ..physics._eos cimport compute_gas_rad_cs
from ..physics._viscous cimport viscous_derivative_func
from ..physics._fallback cimport fallback_source_func
from ..physics._param_wrappers cimport CFallback
from ._gP cimport gP_closure_func


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
        log(27.0 / 32.0)
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
              log(27.0 / 32.0)
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
        0.05,   # step: initial half-width of bracket in log-T
        2.0,    # grow_factor
        64,      # max_expand
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
#  FullPressureClosure                                     #
# ======================================================== #

cdef class FullPressureClosure(OneZoneClosure):
    """
    Merged gas-pressure / full-pressure closure with optional fallback.

    This single class replaces the former ``gPClosure`` and ``igPClosure``
    pair.  The temperature-solve path is selected at construction time via
    ``gas_pressure_only`` — there is no branch inside the hot loop.

    Parameters
    ----------
    gas_pressure_only : bool, optional
        If ``True``, use the gas-pressure-only analytic-seed iterative solve
        (:func:`~._gP.gP_closure_func`).  If ``False`` (default), use the
        combined gas+radiation iterative solve (:func:`igP_closure_func`).
    with_fallback : bool, optional
        If ``True``, install
        :func:`~..physics._fallback.fallback_source_func` as the source term.
        The runtime parameters ``M_fb_0``, ``R_c``, ``t_fb``, and ``beta_fb``
        must then be provided to :meth:`bind_runtime_parameters`.
        Default ``False``.
    mu : double, optional
        Mean molecular weight of the disk gas (dimensionless).  Default ``0.6``.

    Notes
    -----
    Call :meth:`bind_runtime_parameters` with the processed runtime parameter
    dict (output of
    :meth:`~..base.OneZoneAccretionDiskBase.process_runtime_parameters`)
    **before** passing this closure to
    :func:`~..integrator.run_one_zone_model`.  The integrator calls
    :meth:`_pack_params` once (before the hot loop) to fill a
    ``DiskParameters`` struct from the values stored by
    :meth:`bind_runtime_parameters`.
    """

    def __cinit__(
        self,
        bint gas_pressure_only=False,
        bint with_fallback=False,
        double mu=0.6,
    ):
        from trilobite.radiation.opacity import ElectronScatteringOpacity

        if gas_pressure_only:
            self._closure_fn = gP_closure_func
        else:
            self._closure_fn = igP_closure_func
        self._derivative_fn  = viscous_derivative_func
        self._writer_fn      = standard_writer_func
        self.n_result_fields = N_RESULT_FIELDS
        self._mu             = mu
        self._has_fallback   = with_fallback
        self._fallback_ptr   = NULL

        # Default to electron-scattering opacity; caller overrides via .opacity setter.
        self.opacity = ElectronScatteringOpacity()

        if with_fallback:
            self._source_fn = fallback_source_func
        else:
            self._source_fn = NULL

    def bind_runtime_parameters(self, dict run_params):
        """Populate closure fields from the processed runtime parameter dict.

        Must be called once per solve, before passing this closure to
        :func:`~..integrator.run_one_zone_model`.  After this call,
        :meth:`_pack_params` can be called (GIL-free) to assemble the
        ``DiskParameters`` struct for the integrator.

        Parameters
        ----------
        run_params : dict
            Processed parameter dict as returned by
            :meth:`~..base.OneZoneAccretionDiskBase.process_runtime_parameters`.
            Expected keys: ``"log_M_BH"``, ``"log_R_in"``, ``"alpha"`` and,
            when ``with_fallback=True``, also ``"log_M_fb_0"``, ``"log_R_c"``,
            ``"log_t_fb"``, ``"beta_fb"``.
        """
        self._M_BH  = exp(run_params["log_M_BH"])
        self._R_in  = exp(run_params["log_R_in"])
        self._alpha = run_params["alpha"]

        if self._has_fallback:
            self._fallback_obj = CFallback(
                exp(run_params["log_M_fb_0"]),
                exp(run_params["log_R_c"]),
                exp(run_params["log_t_fb"]),
                run_params["beta_fb"],
            )
            self._fallback_ptr = self._fallback_obj.ptr()
        else:
            self._fallback_ptr = NULL

    cdef void _pack_params(self, DiskParameters* p) nogil:
        """Fill *p* from closure-owned fields.  Called GIL-free by the integrator."""
        p.MBH       = self._M_BH
        p.R_in      = self._R_in
        p.alpha     = self._alpha
        p.mu        = self._mu
        p.fallback  = self._fallback_ptr
        p.advection = NULL
        p.opacity   = self._opacity_ptr


# ======================================================== #
# Backward-compatible aliases                              #
# ======================================================== #
# igPClosure was the old name before the gP/igP merger.
igPClosure = FullPressureClosure
