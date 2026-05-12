"""
Numerical thin-shell shock engine for astrophysical transients.

This module provides the :class:`NumericalThinShellShockEngine` (non-relativistic) and
:class:`RelativisticNumericalThinShellShockEngine` (relativistic), general-purpose numerical
shock engines that integrate the thin-shell equations of motion for arbitrary ejecta and
circumstellar medium (CSM) density profiles.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u
from scipy.integrate import solve_ivp

from triceratops.dynamics.shocks.core.shock_engine import ShockEngine
from triceratops.radiation.constants import c_cgs

if TYPE_CHECKING:
    from triceratops._typing import (
        _ArrayLike,
        _UnitBearingArrayLike,
        _UnitBearingScalarLike,
    )


class NumericalThinShellShockEngine(ShockEngine):
    r"""
    Base class for numerically integrated thin-shell shock models with arbitrary ejecta and CSM profiles.

    This :class:`~triceratops.dynamics.shocks.shock_engine.ShockEngine` subclass implements a general thin-shell
    shock model
    dependent on arbitrary ejecta and circumstellar medium (CSM) density profiles. The model assumes a thin-shell
    shock model and utilizes conservation of momentum in the form

    .. math::

        M_{\rm sh} \frac{\partial}{\partial t}\left(v_{\rm sh}\right) = 4\pi R_{\rm sh}^2 (P_2-P_3)

    where :math:`M_{\rm sh}(t)` is the mass of the shocked shell, :math:`v_{\rm sh}(t)` is the shock velocity,
    :math:`R_{\rm sh}(t)` is the shock radius, and :math:`P_{\rm shocked,\;CSM}` and :math:`P_{\rm shocked,\;ej}`
    are the pressures just behind the forward and reverse shocks, respectively.

    .. hint::

        For a detailed description of the theory behind this engine, see :ref:`numeric_thin_shell_shocks`.

    Notes
    -----
    This engine is suitable for scenarios where the supernova ejecta is **expanding homologously** into a
    general CSM density profile :math:`\rho_{\rm CSM}(r)`. Because the ejecta is homologously expanding, it must
    have a general density profile of the form

    .. math::

        \rho(r,t) = t^{-3} G\left(\frac{r}{t}\right),

    where :math:`G(v)` is an arbitrary function of velocity. It can be shown that, under these assumptions,
    conservation of momentum requires that the following set of differential equations be followed:

    .. math::

        \begin{aligned}
        \frac{dR_{\rm sh}}{dt} &= v_{\rm sh}\\
        \frac{dv_{\rm sh}}{dt} &= \frac{-4\pi R_{\rm sh}^2}{ M_{\rm sh}}\left(1-\frac{1}{\chi}\right)
        \left(\rho_{\rm csm} v_{\rm sh}^2 - t^{-3} G[v_{\rm ej}] \Delta^2\right)\\
        \frac{dM_{\rm sh}}{dt} &= 4\pi R_{\rm sh}^2 \left\{\rho_{\rm csm} v_{\rm sh} + t^{-3} G[v_{\rm ej}]
        \Delta\right\}
        \end{aligned}

    where :math:`\Delta = v_{\rm ej} - v_{\rm sh}` is the velocity difference between the ejecta at the shock
    radius and the shock velocity, :math:`\rho_{\rm csm} = \rho_{\rm CSM}(R)` is the CSM density at the shock radius,
    and :math:`M` is the mass of the shocked shell.

    Equivalently, using :math:`\tau = \log t` and making the transformation that :math:`R_{\rm sh} = \xi t`,
    :math:`\Delta = \xi - v_{\rm sh}`, the system can be rewritten as

    .. math::

        \boxed{
        \begin{aligned}
            \frac{d\xi}{d\tau} &= -\Delta,\\[4pt]
            \frac{d\Delta}{d\tau} &=
            -\Delta
            + \frac{4\pi \xi^2}{M_{\rm sh}} \left(1-\frac{1}{\chi}\right)
            \left(t^3\rho_{\rm csm}(\xi t)\,(\xi-\Delta)^2
                  - G(\xi)\,\Delta^2\right),\\[6pt]
            \frac{dM_{\rm sh}}{d\tau} &=
            4\pi \xi^2
            \left[t^3\rho_{\rm csm}(\xi t)\,(\xi-\Delta)
                  + G(\xi)\,\Delta\right].
        \end{aligned}
        }

    """

    # =========================================== #
    # Initialization                              #
    # =========================================== #
    def __init__(self, **kwargs):
        """
        Initialize the :class:`NumericalThinShellShockEngine`.

        Parameters
        ----------
        kwargs:
            Additional keyword arguments passed to the base
            :class:`~triceratops.dynamics.shocks.shock_engine.ShockEngine` class.
        """
        super().__init__(**kwargs)

    # ============================================================= #
    # Supplementary Numerical Methods                               #
    # ============================================================= #
    @staticmethod
    def generate_evaluation_kernel(rho_csm: Callable, G_ej: Callable, gamma: float = 5 / 3):
        r"""
        Generate the evaluation kernel for the ODE.

        This method generates a ``callable`` function which acts as the RHS of the relevant set of
        ODE's for the thin-shell shock model. The generated function is suitable for use with
        :func:`scipy.integrate.solve_ivp`.

        In this base class, the inputs ``rho_csm`` and ``G_ej`` are arbitrary functions which return the CSM density
        and ejecta density profile function respectively in CGS units. Subclasses may provide more assistance in
        generating these functions correctly.

        Parameters
        ----------
        rho_csm: callable
            The function :math:`\rho_{\rm CSM}(r)` which returns the CSM density at radius ``r`` in CGS units.
            This should be a function which takes as input a float or array-like of radii in ``cm`` and returns
            the corresponding CSM density in ``g/cm^3``.
        G_ej: callable
            The function :math:`G(v)` which returns the ejecta density profile function at velocity ``v`` in CGS units.
            This should be a function which takes as input a float or array-like of velocities in ``cm/s`` and returns
            the corresponding ejecta density profile function in ``g * s^3 / cm^3``. The true density is

            .. math::

                \rho_{\rm ej}(r,t) = t^{-3} G\left(\frac{r}{t}\right).
        gamma: float, optional
            The adiabatic index of the shocked gas. Default is ``5/3``.

        Returns
        -------
        callable
            A function which takes as input the independent variable ``tau = log(t)`` and the state vector
            ``y = [xi, Delta, M]``, and returns the derivatives ``dy/dtau`` as a numpy array.
        """
        # Use gamma and the strong-shock RH conditions to compute chi, the compression
        # factor.
        chi = (gamma + 1) / (gamma - 1)

        # With the compression factor defined, we can now define the evaluation kernel:
        def _evaluation_kernel(tau, y):
            # Expand the y-vector into the components xi, Delta, M.
            xi, delta, m = y
            t = np.exp(tau)

            # Using the functions rho_csm and G_ej, we can compute the two necessary
            # CGS density state quantities.
            _rho_csm = rho_csm(xi * t)
            _G_ej = G_ej(xi)

            # Compute the derivatives.
            _dxi_dtau = -delta
            _ddelta_dtau = -delta + (4 * xi**2 * np.pi / m) * (1 - (1 / chi)) * (
                t**3 * _rho_csm * (xi - delta) ** 2 - _G_ej * delta**2
            )
            _dm_dtau = 4.0 * np.pi * xi**2 * (t**3 * _rho_csm * (xi - delta) + _G_ej * delta)

            return np.array([_dxi_dtau, _ddelta_dtau, _dm_dtau])

        return _evaluation_kernel

    # ============================================================ #
    # Core Numerical Methods                                       #
    # ============================================================ #
    def compute_shock_properties(
        self,
        time: "_UnitBearingArrayLike",
        rho_csm: Callable[["_ArrayLike"], "_ArrayLike"] = None,
        G_ej: Callable[["_ArrayLike"], "_ArrayLike"] = None,
        R_0: "_UnitBearingScalarLike" = 1e11 * u.cm,
        v_0: "_UnitBearingScalarLike" = 1e7 * u.cm / u.s,
        M_0: "_UnitBearingScalarLike" = 1e28 * u.g,
        t_0: "_UnitBearingScalarLike" = 1.0 * u.s,
        gamma: float = 5 / 3,
        **kwargs,
    ):
        r"""
        Compute the properties of the shock at a given time.

        This function computes the solution to the thin-shell shock equations at the specified time(s) using
        the provided CSM density profile function and ejecta density profile function.

        Parameters
        ----------
        time: ~astropy.units.Quantity or float or numpy.ndarray
            The time(s) at which to evaluate the shock properties. If units are provided,
            they will be taken into account. Otherwise, CGS units (seconds) are assumed.
            If ``time`` is provided as an array of shape ``(N,)``, the results will all have
            corresponding shapes ``(N,)``.
        rho_csm: callable
            The function :math:`\rho_{\rm CSM}(r)` which returns the CSM density at radius ``r`` in CGS units.
            This should be a function which takes as input a float or array-like of radii in ``cm`` and returns
            the corresponding CSM density in ``g/cm^3``.
        G_ej: callable
            The function :math:`G(v)` which returns the ejecta density profile function at velocity ``v`` in CGS units.
            This should be a function which takes as input a float or array-like of velocities in ``cm/s`` and returns
            the corresponding ejecta density profile function in ``g * s^3 / cm^3``. The true density is

            .. math::

                \rho_{\rm ej}(r,t) = t^{-3} G\left(\frac{r}{t}\right).

        R_0: ~astropy.units.Quantity or float
            The initial shock radius at time ``t_0``. If units are provided, they will be taken into account.
            Otherwise, CGS units (cm) are assumed.
        v_0: ~astropy.units.Quantity or float
            The initial shock velocity at time ``t_0``. If units are provided, they will be taken into account.
            Otherwise, CGS units (cm/s) are assumed.
        M_0: ~astropy.units.Quantity or float
            The initial shocked mass at time ``t_0``. If units are provided, they will be taken into account.
            Otherwise, CGS units (g) are assumed.
        t_0: ~astropy.units.Quantity or float
            The initial time at which the shock properties are defined. If units are provided,
            they will be taken into account. Otherwise, CGS units (seconds) are assumed.
        gamma: float
            The adiabatic index of the shocked gas. Default is ``5/3``.
        kwargs:
            Additional keyword arguments to pass to the ODE solver.

        Returns
        -------
        dict of str, ~astropy.units.Quantity
            A dictionary containing the computed shock properties:

            - ``'radius'``: The shock radius at the given time(s) with units of cm.
            - ``'velocity'``: The shock velocity at the given time(s) with units of cm/s.
            - ``'mass'``: The shocked mass at the given time(s) with units of g.
        """
        # Ensure that the time array has been converted to CGS units (seconds).
        if isinstance(time, u.Quantity):
            time = time.to(u.s).value
        if isinstance(R_0, u.Quantity):
            R_0 = R_0.to(u.cm).value
        if isinstance(v_0, u.Quantity):
            v_0 = v_0.to(u.cm / u.s).value
        if isinstance(M_0, u.Quantity):
            M_0 = M_0.to(u.g).value
        if isinstance(t_0, u.Quantity):
            t_0 = t_0.to(u.s).value

        # Pass off to the low-level CGS computation method.
        shock_properties_cgs = self._compute_shock_properties_cgs(
            time=time,
            rho_csm=rho_csm,
            G_ej=G_ej,
            R_0=R_0,
            v_0=v_0,
            M_0=M_0,
            t_0=t_0,
        )

        # Attach units to the outputs.
        shock_properties = {
            "radius": shock_properties_cgs["radius"] * u.cm,
            "velocity": shock_properties_cgs["velocity"] * (u.cm / u.s),
            "mass": shock_properties_cgs["mass"] * u.g,
        }

        return shock_properties

    # noinspection PyUnresolvedReferences
    def _compute_shock_properties_cgs(
        self,
        time: "_ArrayLike",
        rho_csm: Callable[["_ArrayLike"], "_ArrayLike"] = None,
        G_ej: Callable[["_ArrayLike"], "_ArrayLike"] = None,
        R_0: float = 1e8,
        v_0: float = 1e9,
        M_0: float = 1e28,
        t_0: float = 1.0,
        gamma: float = 5 / 3,
        **kwargs,
    ):
        r"""
        Compute the properties of the shock at a given time in CGS units.

        This function computes the solution to the thin-shell shock equations at the specified time(s) using
        the provided CSM density profile function and ejecta density profile function.

        Parameters
        ----------
        time: array-like
            The time(s) at which to evaluate the shock properties in seconds. This can be a
            scalar or an array of times. The results will match the shape of the input time array.
        rho_csm: callable
            The function :math:`\rho_{\rm CSM}(r)` which returns the CSM density at radius ``r`` in CGS units.
            This should be a function which takes as input a float or array-like of radii in ``cm`` and returns
            the corresponding CSM density in ``g/cm^3``.
        G_ej: callable
            The function :math:`G(v)` which returns the ejecta density profile function at velocity ``v`` in CGS units.
            This should be a function which takes as input a float or array-like of velocities in ``cm/s`` and returns
            the corresponding ejecta density profile function in ``g * s^3 / cm^3``. The true density is

            .. math::

                \rho_{\rm ej}(r,t) = t^{-3} Gleft(\frac{r}{t}\right).

        R_0: float
            The initial shock radius at time ``t_0`` in cm.
        v_0: float
            The initial shock velocity at time ``t_0`` in cm/s.
        M_0: float
            The initial shocked mass at time ``t_0`` in g.
        t_0:
            The initial time at which the shock properties are defined in seconds.
        gamma: float
            The adiabatic index of the shocked gas. Default is ``5/3``.
        **kwargs:
            Additional keyword arguments to pass to the ODE solver.

        Returns
        -------
        dict: of str, array-like
            A dictionary containing the computed shock properties:

            - 'radius': The shock radius at the given time(s) in cm.
            - 'velocity': The shock velocity at the given time(s) in cm/s.
            - 'mass': The shocked mass at the given time(s) in g.
        """
        # Quick check to ensure that the user has provided the necessary functions.
        if rho_csm is None:
            raise ValueError("A CSM density profile function `rho_csm` must be provided.")
        if G_ej is None:
            raise ValueError("An ejecta density profile function `G_ej` must be provided.")

        # --- Parameter Management and Coercion --- #
        # Coerce the parameters into the initial conditions for the ODE solver.
        xi_0 = R_0 / t_0
        delta_0 = xi_0 - v_0
        m_0 = M_0

        # --- Mange the Kernel and ODE Solver --- #
        # Generate the evaluation kernel for the ODE solver.
        evaluation_kernel = self.generate_evaluation_kernel(
            rho_csm=rho_csm,
            G_ej=G_ej,
            gamma=gamma,
        )

        # Set up the ODE solver.
        t_bound = (np.log(t_0), np.log(np.amax(time)))
        y_0 = np.array([xi_0, delta_0, m_0])

        # Perform the integration using solve_ivp.
        sol = solve_ivp(
            fun=evaluation_kernel,
            t_span=t_bound,
            y0=y_0,
            t_eval=np.log(time),
            rtol=kwargs.get("rtol", 1e-10),
            method=kwargs.get("method", "Radau"),  # Implicit method for stiff problems
            **kwargs,
        )

        # --- Extract Data and Check Validity --- #
        if sol.status < 0:
            raise RuntimeError(f"ODE solver failed to integrate the thin-shell shock equations: \n{sol.message}")

        # Extract the shock radius and velocity from the solution.
        xi_sol = sol.y[0]
        delta_sol = sol.y[1]
        m_sol = sol.y[2]

        # Convert integration space variables into physical shock properties.
        shock_radius = xi_sol * np.exp(sol.t)
        shock_velocity = xi_sol - delta_sol
        shock_mass = m_sol

        return {
            "radius": shock_radius,
            "velocity": shock_velocity,
            "mass": shock_mass,
        }


class RelativisticNumericalThinShellShockEngine(ShockEngine):
    r"""
    Numerically integrated relativistic thin-shell shock engine with arbitrary upstream profiles.

    This :class:`~triceratops.dynamics.shocks.shock_engine.ShockEngine` subclass integrates the
    relativistic thin-shell equations of motion derived from the covariant conservation laws

    .. math::

        \nabla_\mu T^{\mu\nu} = 0,
        \qquad
        \nabla_\mu (\rho u^\mu) = 0.

    The shell state is described by its lab-frame energy :math:`E_{\rm sh}`, radial momentum
    :math:`p_{\rm sh}`, and baryonic rest mass :math:`M_s`. The shell velocity follows from the
    four-momentum,

    .. math::

        \beta_{\rm sh} = \frac{p_{\rm sh}\,c}{E_{\rm sh}},
        \qquad
        v_{\rm sh} = \frac{p_{\rm sh}\,c^2}{E_{\rm sh}}.

    The equations of motion are

    .. math::

        \begin{aligned}
        \frac{dR_{\rm sh}}{dt} &= v_{\rm sh},\\[4pt]
        \frac{dE_{\rm sh}}{dt} &=
            4\pi R^2\!\left[
                w_{\rm ej}\gamma_{\rm ej}^2(\beta_{\rm ej}-\beta_{\rm sh})
                - w_{\rm csm}\gamma_{\rm csm}^2(\beta_{\rm csm}-\beta_{\rm sh})
                + (P_{\rm ej}-P_{\rm csm})\beta_{\rm sh}
            \right],\\[4pt]
        \frac{dp_{\rm sh}}{dt} &=
            4\pi R^2\!\left[
                w_{\rm ej}\gamma_{\rm ej}^2\beta_{\rm ej}(\beta_{\rm ej}-\beta_{\rm sh})
                - w_{\rm csm}\gamma_{\rm csm}^2\beta_{\rm csm}(\beta_{\rm csm}-\beta_{\rm sh})
                + (P_{\rm ej}-P_{\rm csm})
            \right],\\[4pt]
        \frac{dM_s}{dt} &=
            4\pi R^2\!\left[
                \rho_{\rm ej}\gamma_{\rm ej}(\beta_{\rm ej}-\beta_{\rm sh})
                - \rho_{\rm csm}\gamma_{\rm csm}(\beta_{\rm csm}-\beta_{\rm sh})
            \right],
        \end{aligned}

    where the upstream thermodynamic quantities are computed from the prescribed fluid profiles
    via the perfect-fluid EOS :math:`P = (\hat\gamma - 1)U_{\rm int}` and
    :math:`w = \rho c^2 + \hat\gamma\,U_{\rm int}`.

    These equations are integrated in log-time :math:`\tau = \ln t` for numerical stability across
    large dynamic ranges.

    .. hint::

        For the full derivation see :ref:`relativistic_numeric_thin_shell_shocks`.

    Notes
    -----
    The six upstream callables ``rho_ej``, ``v_ej``, ``U_int_ej``, ``rho_csm``, ``v_csm``, and
    ``U_int_csm`` must each accept ``(r, t)`` in CGS units and return the corresponding fluid
    quantity in CGS.  No cold-upstream or homologous-expansion approximation is imposed.
    """

    def __init__(self, **kwargs):
        """
        Initialize the :class:`RelativisticNumericalThinShellShockEngine`.

        Parameters
        ----------
        kwargs
            Additional keyword arguments passed to the base
            :class:`~triceratops.dynamics.shocks.shock_engine.ShockEngine` class.
        """
        super().__init__(**kwargs)

    # ============================================================= #
    # Supplementary Numerical Methods                               #
    # ============================================================= #
    @staticmethod
    def generate_evaluation_kernel(
        rho_ej: Callable,
        v_ej: Callable,
        U_int_ej: Callable,
        rho_csm: Callable,
        v_csm: Callable,
        U_int_csm: Callable,
        gamma_hat_ej: float = 5 / 3,
        gamma_hat_csm: float = 5 / 3,
    ):
        r"""
        Generate the RHS kernel for the relativistic thin-shell ODE.

        Returns a callable suitable for :func:`scipy.integrate.solve_ivp` that evaluates
        :math:`d\mathbf{y}/d\tau` where :math:`\tau = \ln t` and
        :math:`\mathbf{y} = (R_{\rm sh},\, E_{\rm sh},\, p_{\rm sh},\, M_s)`.

        Parameters
        ----------
        rho_ej : callable
            :math:`\rho_{\rm ej}(r,\,t)` — ejecta rest-mass density in :math:`\mathrm{g\,cm^{-3}}`.
        v_ej : callable
            :math:`v_{\rm ej}(r,\,t)` — ejecta lab-frame velocity in :math:`\mathrm{cm\,s^{-1}}`.
        U_int_ej : callable
            :math:`U_{\rm int,\,ej}(r,\,t)` — ejecta proper internal energy density in
            :math:`\mathrm{erg\,cm^{-3}}`.
        rho_csm : callable
            :math:`\rho_{\rm csm}(r,\,t)` — CSM rest-mass density in :math:`\mathrm{g\,cm^{-3}}`.
        v_csm : callable
            :math:`v_{\rm csm}(r,\,t)` — CSM lab-frame velocity in :math:`\mathrm{cm\,s^{-1}}`.
        U_int_csm : callable
            :math:`U_{\rm int,\,csm}(r,\,t)` — CSM proper internal energy density in
            :math:`\mathrm{erg\,cm^{-3}}`.
        gamma_hat_ej : float, optional
            Adiabatic index :math:`\hat\gamma` for the upstream ejecta EOS. Default ``5/3``.
        gamma_hat_csm : float, optional
            Adiabatic index :math:`\hat\gamma` for the upstream CSM EOS. Default ``5/3``.

        Returns
        -------
        callable
            Function ``f(tau, y)`` returning ``dy/dtau`` as a length-4 numpy array with
            elements ``[dR/dtau, dE/dtau, dp/dtau, dM/dtau]``.
        """
        # Work with rescaled variables Ê = E/c² and p̂ = p/c (both in grams) so that
        # all four state components share the same dimensional scale and scipy's
        # numerical Jacobian does not overflow when E reaches ~10^50 erg.
        # In these variables: β = p̂/Ê, v_sh = β·c, and the enthalpy density
        # rescales to ŵ = w/c² = ρ + γ̂ U_int/c² (g cm⁻³).
        c2 = c_cgs**2

        def _kernel(tau, y):
            # y = [R (cm), Ê = E/c² (g), p̂ = p/c (g), M (g)]
            R, E_hat, p_hat, M = y
            t = np.exp(tau)

            # Shell kinematics — β and v from rescaled four-momentum.
            beta_s = p_hat / E_hat
            v_sh = beta_s * c_cgs

            # Upstream ejecta state at (R, t).
            _rho_ej = rho_ej(R, t)
            _v_ej = v_ej(R, t)
            _U_ej = U_int_ej(R, t)
            _beta_ej = _v_ej / c_cgs
            _gamma_ej = 1.0 / np.sqrt(1.0 - _beta_ej**2)
            # Rescaled enthalpy and pressure (÷ c²) to stay in g cm⁻³.
            _w_hat_ej = _rho_ej + gamma_hat_ej * _U_ej / c2
            _P_hat_ej = (gamma_hat_ej - 1.0) * _U_ej / c2

            # Upstream CSM state at (R, t).
            _rho_csm = rho_csm(R, t)
            _v_csm = v_csm(R, t)
            _U_csm = U_int_csm(R, t)
            _beta_csm = _v_csm / c_cgs
            _gamma_csm = 1.0 / np.sqrt(1.0 - _beta_csm**2)
            _w_hat_csm = _rho_csm + gamma_hat_csm * _U_csm / c2
            _P_hat_csm = (gamma_hat_csm - 1.0) * _U_csm / c2

            # Velocity differences.
            d_beta_ej = _beta_ej - beta_s
            d_beta_csm = _beta_csm - beta_s

            # t · 4π R² · c prefactor (converts dX/dt → dX/dτ; the extra c
            # restores the factor that the theory doc absorbs into c = 1 natural units).
            area_factor = 4.0 * np.pi * R**2 * t * c_cgs

            # dÊ/dτ  (energy ÷ c²).
            dE_hat_dtau = area_factor * (
                _w_hat_ej * _gamma_ej**2 * d_beta_ej
                - _w_hat_csm * _gamma_csm**2 * d_beta_csm
                + (_P_hat_ej - _P_hat_csm) * beta_s
            )

            # dp̂/dτ  (momentum ÷ c).
            dp_hat_dtau = area_factor * (
                _w_hat_ej * _gamma_ej**2 * _beta_ej * d_beta_ej
                - _w_hat_csm * _gamma_csm**2 * _beta_csm * d_beta_csm
                + (_P_hat_ej - _P_hat_csm)
            )

            # Baryonic rest-mass.
            dM_dtau = area_factor * (_rho_ej * _gamma_ej * d_beta_ej - _rho_csm * _gamma_csm * d_beta_csm)

            # Radius.
            dR_dtau = t * v_sh

            return np.array([dR_dtau, dE_hat_dtau, dp_hat_dtau, dM_dtau])

        return _kernel

    # ============================================================ #
    # Core Numerical Methods                                       #
    # ============================================================ #
    def compute_shock_properties(
        self,
        time: "_UnitBearingArrayLike",
        rho_ej: Callable[["_ArrayLike", "_ArrayLike"], "_ArrayLike"] = None,
        v_ej: Callable[["_ArrayLike", "_ArrayLike"], "_ArrayLike"] = None,
        U_int_ej: Callable[["_ArrayLike", "_ArrayLike"], "_ArrayLike"] = None,
        rho_csm: Callable[["_ArrayLike", "_ArrayLike"], "_ArrayLike"] = None,
        v_csm: Callable[["_ArrayLike", "_ArrayLike"], "_ArrayLike"] = None,
        U_int_csm: Callable[["_ArrayLike", "_ArrayLike"], "_ArrayLike"] = None,
        R_0: "_UnitBearingScalarLike" = 1e11 * u.cm,
        E_0: "_UnitBearingScalarLike" = 1e48 * u.erg,
        p_0: "_UnitBearingScalarLike" = 1e38 * u.g * u.cm / u.s,
        M_0: "_UnitBearingScalarLike" = 1e28 * u.g,
        t_0: "_UnitBearingScalarLike" = 1.0 * u.s,
        gamma_hat_ej: float = 5 / 3,
        gamma_hat_csm: float = 5 / 3,
        **kwargs,
    ):
        r"""
        Compute relativistic thin-shell shock properties at the requested times.

        Parameters
        ----------
        time : ~astropy.units.Quantity or array-like
            Times at which to evaluate the shock. Bare floats are interpreted as seconds.
        rho_ej : callable
            :math:`\rho_{\rm ej}(r,\,t)` in CGS (:math:`\mathrm{g\,cm^{-3}}`).
        v_ej : callable
            :math:`v_{\rm ej}(r,\,t)` in CGS (:math:`\mathrm{cm\,s^{-1}}`).
        U_int_ej : callable
            :math:`U_{\rm int,\,ej}(r,\,t)` in CGS (:math:`\mathrm{erg\,cm^{-3}}`).
        rho_csm : callable
            :math:`\rho_{\rm csm}(r,\,t)` in CGS (:math:`\mathrm{g\,cm^{-3}}`).
        v_csm : callable
            :math:`v_{\rm csm}(r,\,t)` in CGS (:math:`\mathrm{cm\,s^{-1}}`).
        U_int_csm : callable
            :math:`U_{\rm int,\,csm}(r,\,t)` in CGS (:math:`\mathrm{erg\,cm^{-3}}`).
        R_0 : ~astropy.units.Quantity or float
            Initial shock radius. Bare float interpreted as cm.
        E_0 : ~astropy.units.Quantity or float
            Initial shell lab-frame energy. Bare float interpreted as erg.
        p_0 : ~astropy.units.Quantity or float
            Initial shell radial momentum. Bare float interpreted as g cm s⁻¹.
        M_0 : ~astropy.units.Quantity or float
            Initial shell baryonic rest mass. Bare float interpreted as g.
        t_0 : ~astropy.units.Quantity or float
            Reference time at which the initial conditions are defined. Bare float
            interpreted as seconds.
        gamma_hat_ej : float, optional
            Adiabatic index for the upstream ejecta EOS. Default ``5/3``.
        gamma_hat_csm : float, optional
            Adiabatic index for the upstream CSM EOS. Default ``5/3``.
        **kwargs
            Forwarded to :func:`scipy.integrate.solve_ivp`.

        Returns
        -------
        dict of str, ~astropy.units.Quantity
            Keys: ``'radius'`` (cm), ``'velocity'`` (cm s⁻¹), ``'mass'`` (g),
            ``'energy'`` (erg), ``'momentum'`` (g cm s⁻¹), ``'beta'`` (dimensionless),
            ``'lorentz_factor'`` (dimensionless).
        """
        if isinstance(time, u.Quantity):
            time = time.to(u.s).value
        if isinstance(R_0, u.Quantity):
            R_0 = R_0.to(u.cm).value
        if isinstance(E_0, u.Quantity):
            E_0 = E_0.to(u.erg).value
        if isinstance(p_0, u.Quantity):
            p_0 = p_0.to(u.g * u.cm / u.s).value
        if isinstance(M_0, u.Quantity):
            M_0 = M_0.to(u.g).value
        if isinstance(t_0, u.Quantity):
            t_0 = t_0.to(u.s).value

        result = self._compute_shock_properties_cgs(
            time=time,
            rho_ej=rho_ej,
            v_ej=v_ej,
            U_int_ej=U_int_ej,
            rho_csm=rho_csm,
            v_csm=v_csm,
            U_int_csm=U_int_csm,
            R_0=R_0,
            E_0=E_0,
            p_0=p_0,
            M_0=M_0,
            t_0=t_0,
            gamma_hat_ej=gamma_hat_ej,
            gamma_hat_csm=gamma_hat_csm,
            **kwargs,
        )

        return {
            "radius": result["radius"] * u.cm,
            "velocity": result["velocity"] * (u.cm / u.s),
            "mass": result["mass"] * u.g,
            "energy": result["energy"] * u.erg,
            "momentum": result["momentum"] * (u.g * u.cm / u.s),
            "beta": result["beta"] * u.dimensionless_unscaled,
            "lorentz_factor": result["lorentz_factor"] * u.dimensionless_unscaled,
        }

    def _compute_shock_properties_cgs(
        self,
        time: "_ArrayLike",
        rho_ej: Callable = None,
        v_ej: Callable = None,
        U_int_ej: Callable = None,
        rho_csm: Callable = None,
        v_csm: Callable = None,
        U_int_csm: Callable = None,
        R_0: float = 1e11,
        E_0: float = 1e48,
        p_0: float = 1e38,
        M_0: float = 1e28,
        t_0: float = 1.0,
        gamma_hat_ej: float = 5 / 3,
        gamma_hat_csm: float = 5 / 3,
        **kwargs,
    ):
        r"""
        Compute relativistic thin-shell shock properties in CGS units.

        Parameters
        ----------
        time : array-like
            Times in seconds.
        rho_ej : callable
            :math:`\rho_{\rm ej}(r,\,t)` in :math:`\mathrm{g\,cm^{-3}}`.
        v_ej : callable
            :math:`v_{\rm ej}(r,\,t)` in :math:`\mathrm{cm\,s^{-1}}`.
        U_int_ej : callable
            :math:`U_{\rm int,\,ej}(r,\,t)` in :math:`\mathrm{erg\,cm^{-3}}`.
        rho_csm : callable
            :math:`\rho_{\rm csm}(r,\,t)` in :math:`\mathrm{g\,cm^{-3}}`.
        v_csm : callable
            :math:`v_{\rm csm}(r,\,t)` in :math:`\mathrm{cm\,s^{-1}}`.
        U_int_csm : callable
            :math:`U_{\rm int,\,csm}(r,\,t)` in :math:`\mathrm{erg\,cm^{-3}}`.
        R_0 : float
            Initial shock radius in cm.
        E_0 : float
            Initial shell lab-frame energy in erg.
        p_0 : float
            Initial shell radial momentum in g cm s⁻¹.
        M_0 : float
            Initial shell baryonic rest mass in g.
        t_0 : float
            Reference time in seconds.
        gamma_hat_ej : float, optional
            Adiabatic index for the upstream ejecta EOS. Default ``5/3``.
        gamma_hat_csm : float, optional
            Adiabatic index for the upstream CSM EOS. Default ``5/3``.
        **kwargs
            Forwarded to :func:`scipy.integrate.solve_ivp`.

        Returns
        -------
        dict of str, array-like
            Keys: ``'radius'`` (cm), ``'velocity'`` (cm s⁻¹), ``'mass'`` (g),
            ``'energy'`` (erg), ``'momentum'`` (g cm s⁻¹), ``'beta'``, ``'lorentz_factor'``.
        """
        for name, fn in [
            ("rho_ej", rho_ej),
            ("v_ej", v_ej),
            ("U_int_ej", U_int_ej),
            ("rho_csm", rho_csm),
            ("v_csm", v_csm),
            ("U_int_csm", U_int_csm),
        ]:
            if fn is None:
                raise ValueError(f"Upstream callable `{name}` must be provided.")

        kernel = self.generate_evaluation_kernel(
            rho_ej=rho_ej,
            v_ej=v_ej,
            U_int_ej=U_int_ej,
            rho_csm=rho_csm,
            v_csm=v_csm,
            U_int_csm=U_int_csm,
            gamma_hat_ej=gamma_hat_ej,
            gamma_hat_csm=gamma_hat_csm,
        )

        # Rescale E and p into the kernel's internal units (÷ c²/÷ c) so that
        # all state components are in grams — see generate_evaluation_kernel.
        y0 = np.array([R_0, E_0 / c_cgs**2, p_0 / c_cgs, M_0])
        t_span = (np.log(t_0), np.log(np.amax(time)))

        sol = solve_ivp(
            fun=kernel,
            t_span=t_span,
            y0=y0,
            t_eval=np.log(time),
            rtol=kwargs.pop("rtol", 1e-10),
            method=kwargs.pop("method", "Radau"),
            **kwargs,
        )

        if sol.status < 0:
            raise RuntimeError(f"ODE solver failed to integrate the relativistic thin-shell equations:\n{sol.message}")

        R_sol = sol.y[0]
        E_hat_sol = sol.y[1]  # Ê = E/c²  (g)
        p_hat_sol = sol.y[2]  # p̂ = p/c   (g)
        M_sol = sol.y[3]

        # β from rescaled variables; convert back to physical E and p.
        beta_sol = p_hat_sol / E_hat_sol
        gamma_sol = 1.0 / np.sqrt(1.0 - beta_sol**2)
        v_sol = beta_sol * c_cgs
        E_sol = E_hat_sol * c_cgs**2
        p_sol = p_hat_sol * c_cgs

        return {
            "radius": R_sol,
            "velocity": v_sol,
            "mass": M_sol,
            "energy": E_sol,
            "momentum": p_sol,
            "beta": beta_sol,
            "lorentz_factor": gamma_sol,
        }
