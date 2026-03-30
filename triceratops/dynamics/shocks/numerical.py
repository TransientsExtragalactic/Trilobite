"""
Numerical thin-shell shock engine for astrophysical transients.

This module provides the :class:`NumericalThinShellShockEngine`, a general-purpose numerical
shock engine that integrates the thin-shell equations of motion for arbitrary ejecta and
circumstellar medium (CSM) density profiles.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u
from scipy.integrate import solve_ivp

from .shock_engine import ShockEngine

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
        print(chi, (1 - (1 / chi)))

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

                \rho_{\rm ej}(r,t) = t^{-3} G\\left(\frac{r}{t}\right).

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
