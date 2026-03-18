"""
Ejecta property computation for supernova models.

Function in this module compute various properties of the supernova ejecta and ambient CSM in
different physical scenarios of interest. They can then be baked into the models of :mod:`models`.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u
from scipy.integrate import solve_ivp

from ..shock_engine import ShockEngine
from .profiles import _normalize_inner_BPL_ejecta_cgs

# Handle type aliases and static type checking
if TYPE_CHECKING:
    from triceratops._typing import (
        _ArrayLike,
        _UnitBearingArrayLike,
        _UnitBearingScalarLike,
    )


# ====================================================== #
# Shock Engine Classes                                   #
# ====================================================== #
# These are the core encapsulations for the various sorts of shock dynamics that
# we've implemented. They may rely on additional utility functions below.
class ChevalierSelfSimilarShockEngine(ShockEngine):
    r"""
    Implementation of the "classical" Chevalier 1982 self-similar supernova shock model.

    This :class:`~dynamics.shock_engine.ShockEngine` subclass implements the self-similar shock solutions
    described in :footcite:t:`chevalierSelfsimilarSolutionsInteraction1982` for the interaction between
    supernova ejecta and a surrounding circumstellar medium (CSM). The model assumes power-law density profiles
    for both the ejecta and the CSM, leading to a self-similar evolution of the shock structure over time.

    The engine calculates the position and velocity of the shock-interface as functions of time, based on
    the ejecta and CSM density profiles. It can be used to model the dynamical evolution of supernova remnants
    and their interaction with the surrounding medium.

    .. note::

        A derivation of this model can be found on the :ref:`supernova_shocks_theory` guide. Much of the
        relevant detail omitted here can be found there.

    Notes
    -----

    .. important::

        This model is based off of the classical self-similar solutions for supernova ejecta interacting
        with a circumstellar medium as described in
        :footcite:t:`chevalierSelfsimilarSolutionsInteraction1982`, and a discussion of their relevance
        for synchrotron emission from supernovae may be found in
        :footcite:t:`ChevalierXRayRadioEmission1982`. The implementation here follows these sources quite closely
        with some moderate improvements to the formalism.

    **Model Assumptions:**

    This model assumes the following:

    - The supernova ejecta is expanding homologously such that :math:`r = vt`.
    - As required by homologous expansion, the ejecta density profile must be a function of the velocity of the
      form

      .. math::

            \rho(r,t) = t^{-3} f\left(\frac{r}{t}\right),

      for some, generally unknown, function :math:`f(v)`.

      In this model, the supernova ejecta density profile follows a broken power-law in velocity space such that

      .. math::

            \rho_{\rm ej}(r,t) = K_{\rm ej} t^{-3} \begin{cases}
                v^{-\delta}, & v < v_t \\
                v_t^{n-\delta} v^{-n}, & v \geq v_t,
            \end{cases}

      where :math:`v_t` is the transition velocity between the inner and outer ejecta profiles, and :math:`K` is
      the normalization constant.

    - The circumstellar medium (CSM) surrounding the supernova progenitor follows a power-law density profile of the
      form

      .. math::

            \rho_{\rm CSM}(r) = K_{\rm CSM} r^{-s},

      where :math:`\rho_0` is the normalization constant and :math:`s` is the CSM density power-law index.

    - The interaction between the ejecta and the CSM produces a forward and reverse shock structure that evolves
      self-similarly over time and maintains the shock within a region suitable for thin-shell approximation.

    **Model Features**

    Under the previously described assumptions, the position of the discontinuity surface between the forward and
    reverse shocks evolves self-similarly as :math:`R(t)` such that

    .. math::

        R(t) = \left(\frac{\zeta K_{\rm CSM}}{K_{\rm ej}}\right)^{\frac{1}{s-n}} t^{\frac{3-n}{s-n}},

    where :math:`A` is a dimensionless constant that depends on the ejecta and CSM density power-law indices
    :math:`n` and :math:`s`, respectively. From conservation of momentum, it can be derived that

    .. math::

        \zeta = \frac{3\lambda^2 +4\frac{ (\lambda -1)\lambda}{3-s} }
                    { 3(1-\lambda)^2-4\frac{ (\lambda  -1)\lambda}{n-3}}.

    is a factor generally of order unity.

    **Connection to Energetics**:

    The ejecta normalization constant :math:`K_{\rm ej}` and transition velocity :math:`v_t` can be computed
    from the total ejecta kinetic energy :math:`E_{\rm ej}` and ejecta mass :math:`M_{\rm ej}` instead of requiring
    them as direct inputs.

    See Also
    --------
    ChevalierSelfSimilarWindShockEngine
        Specialized version of :class:`ChevalierSelfSimilarShockEngine` for steady-wind CSM.
    NumericalThinShellShockEngine
        A numerical implementation of thin-shell shock dynamics for arbitrary ejecta and CSM profiles.

    References
    ----------
    .. footbibliography::

    """

    # =========================================== #
    # Initialization                              #
    # =========================================== #
    def __init__(self, **kwargs):
        """
        Instantiate the :class:`ChevalierSelfSimilarShockEngine`.

        Parameters
        ----------
        kwargs:
            Additional parameters for initializing the shock engine. Currently, no additional
            parameters are required.
        """
        super().__init__(**kwargs)

    # ============================================================= #
    # Supplementary Numerical Methods                               #
    # ============================================================= #
    @staticmethod
    def compute_scale_parameter(n: float, s: float):
        r"""
        Compute the radius scale parameter for the Chevalier self-similar shock solution.

        This function computes the dimensionless :math:`\zeta` parameter that appears in the
        radius normalization of the Chevalier self-similar shock solution. This parameter
        depends on the ejecta density power-law index :math:`n` and the CSM density power-law
        index :math:`s`.

        Parameters
        ----------
        n: float
            The outer ejecta density profile power-law index.
        s: float
            The CSM density profile power-law index.

        Returns
        -------
        float
            The computed scale parameter :math:`\zeta`.

        Notes
        -----
        The equation for :math:`\zeta` is

        .. math::

            \zeta = \frac{3\lambda^2 +4\frac{ (\lambda -1)\lambda}{3-s} }
                        { 3(1-\lambda)^2-4\frac{ (\lambda  -1)\lambda}{n-3}},

        where :math:`\lambda = \frac{3-n}{s-n}`. A derivation of this parameter can be found in
        :ref:`supernova_shocks_theory`.
        """
        # Construct lambda from n and s.
        _lambda = (3 - n) / (s - n)

        # Compute the A, B, C, and D terms
        A = 4 * _lambda * (_lambda - 1) / (3 - s)
        B = 4 * _lambda * (_lambda - 1) / (n - 3)
        C = 3 * _lambda**2
        D = 3 * (1 - _lambda) ** 2

        return (C + A) / (D - B)

    # ============================================================ #
    # Core Shock Engine Methods                                    #
    # ============================================================ #
    def compute_shock_properties(
        self,
        time: "_UnitBearingArrayLike",
        E_ej: "_UnitBearingScalarLike" = 1e51 * u.erg,
        M_ej: "_UnitBearingScalarLike" = 10 * u.Msun,
        K_csm: "_UnitBearingScalarLike" = None,
        n: float = 10.0,
        s: float = 2.0,
        delta: float = 0.0,
    ) -> dict[str, u.Quantity]:
        r"""
        Compute the shock properties at a given time.

        This method calculates the shock radius and velocity as a function of time since the
        explosion. See the class documentation (:class:`ChevalierSelfSimilarShockEngine`) for details
        on the relevant theory and assumptions.

        Parameters
        ----------
        time: ~astropy.units.Quantity or float or numpy.ndarray
            The time(s) at which to evaluate the shock properties. If units are provided,
            they will be taken into account. Otherwise, CGS units (seconds) are assumed.
            If ``time`` is provided as an array of shape ``(N,)``, the results will all have
            corresponding shapes ``(N,)``.
        E_ej: ~astropy.units.Quantity or float
            The total energy in the ejecta from the explosion. If units are provided,
            they will be taken into account. Otherwise, CGS units (erg) are assumed.
        M_ej: ~astropy.units.Quantity or float
            The total mass in the ejecta from the explosion. If units are provided,
            they will be taken into account. Otherwise, CGS units (grams) are assumed.
        K_csm: ~astropy.units.Quantity or float, optional
            The scaling (:math:`K_{\rm CSM}`) for the CSM density profile of the form
            :math:`\rho_{\rm CSM}(r) = K_{\rm CSM} r^{-s}`. If units are provided, they will be
            taken into account. Otherwise, CGS units (``g * cm^{(s-3)}``) are assumed. If not provided,
            a default scaling based on a wind-like CSM with :math:`\dot{M} \sim 10^{-5} M_{\odot}/yr`
            and :math:`v_w \sim 1000 km/s` is used at a radius of :math:`r = 10^{16} cm`.

            .. note::

                For science scenarios, ``K_csm`` should always be provided explicitly to ensure
                physical accuracy. The default is only a placeholder.
        n: float, optional
            The outer ejecta density profile power-law index. Default is ``10.0``. Must be steeper than
            5 for convergence.
        s: float, optional
            The CSM density profile power-law index. Default is ``2.0``.
        delta: float, optional
            The inner ejecta density profile power-law index. Default is ``0.0``. Must be less than
            3 for convergence. In general, it is suitable to use ``0.0``.

        Returns
        -------
        dict of str, ~astropy.units.Quantity
            A dictionary containing the computed shock properties:

            - ``'radius'``: The shock radius at the given time(s) with units of cm.
            - ``'velocity'``: The shock velocity at the given time(s) with units of cm/s.

        """
        # Validate inputs and determine a scaling for K_csm if one
        # is not provided. To do this, we set a standard density based on a wind-like
        # density profile. THIS IS PURELY A MEANS FOR PICKING A DEFAULT, IT SHOULD
        # NOT BE USED IN SCIENCE RUNS.
        if K_csm is None:
            # Assume a generic wind-like CSM with M_dot ~ 1e-5 Msun/yr and v_w ~ 1000 km/s scaled
            # at r = 1e16 cm.
            K_csm = ((1e16 * u.cm) ** (s - 2)) * (1e-5 * u.Msun / u.yr) / (4.0 * np.pi * (1000 * u.km / u.s))

        # Scale everything down to CGS for internal computation.
        if isinstance(E_ej, u.Quantity):
            E_ej = E_ej.to(u.erg).value
        if isinstance(M_ej, u.Quantity):
            M_ej = M_ej.to(u.g).value
        if isinstance(time, u.Quantity):
            time = time.to(u.s).value
        if isinstance(K_csm, u.Quantity):
            K_csm = K_csm.to(u.g * u.cm ** (s - 3)).value

        # Perform checks on ``n``, ``s``, and ``delta`` to ensure convergence.
        if delta >= 3:
            raise ValueError("The inner ejecta density profile index `delta` must be less than 3 for convergence.")
        if n <= 5:
            raise ValueError("The outer ejecta density profile index `n` must be greater than 5 for convergence.")

        # Call the internal CGS computation method.
        shock_properties_cgs = self._compute_shock_properties_cgs(
            time=time, E_ej=E_ej, M_ej=M_ej, K_csm=K_csm, n=n, s=s, delta=delta
        )

        # Attach units to the outputs.
        shock_properties = {
            "radius": shock_properties_cgs["radius"] * u.cm,
            "velocity": shock_properties_cgs["velocity"] * (u.cm / u.s),
        }

        return shock_properties

    def _compute_shock_properties_cgs(
        self,
        time: "_ArrayLike",
        E_ej: float = 1e51,
        M_ej: float = 1e34,
        K_csm: float = 5e11,
        n: float = 10.0,
        s: float = 2.0,
        delta: float = 0.0,
    ):
        r"""
        Compute the shock properties at a given time in CGS units.

        This method computes the shock radius and velocity at a given time based on the
        Chevalier self-similar solution for supernova ejecta interacting with a circumstellar
        medium (CSM). The ejecta and CSM are assumed to follow power-law density profiles.

        Parameters
        ----------
        time: array-like
            The time(s) at which to evaluate the shock properties in seconds. This can be a scalar or
            an array of times. The results will match the shape of the input time array.
        E_ej: float
            The total kinetic energy of the ejecta in erg. Default is ``1e51``.
        M_ej: float
            The total mass of the ejecta in grams. Default is ``1e34``.
        K_csm: float
            The normalization constant of the CSM density profile in CGS units ``g * cm^{(s-3}}``.
        n: float
            The outer ejecta density profile power-law index. Default is ``10.0``.
        s: float
            The CSM density profile power-law index. Default is ``2.0``.
        delta:
            The inner ejecta density profile power-law index. Default is ``0.0``.

        Returns
        -------
        dict of str, array-like
            A dictionary containing the computed shock properties:

            - 'radius': The shock radius at the given time(s) in cm.
            - 'velocity': The shock velocity at the given time(s) in cm/s.

        """
        # Using the ``_compute_v_t_and_K_from_energetics_cgs`` static method to get v_t and K. We can
        # discard v_t, but K is necessary.
        v_t, K = _normalize_inner_BPL_ejecta_cgs(
            E_ej=E_ej,
            M_ej=M_ej,
            n=n,
            delta=delta,
        )

        # Correct K since it needs to be multiplied by v_t^(n - delta) for the outer ejecta profile.
        K_EJ = K * v_t ** (n - delta)

        # Compute relevant factors: We of course need ``_lambda`` (the scaling in time of the radius),
        # ``_gamma`` appears in the radius scale factor, as does the ``_lambda_gamma_constant``.
        _lambda = (3 - n) / (s - n)
        SCALE_CONSTANT = self.compute_scale_parameter(n=n, s=s)
        R_0 = (SCALE_CONSTANT * K_csm / K_EJ) ** (1 / (s - n))

        # Compute the shock radius and velocity at the given time(s).
        shock_radius = R_0 * time**_lambda
        shock_velocity = _lambda * shock_radius / time

        return {
            "radius": shock_radius,
            "velocity": shock_velocity,
        }

    # =========================================== #
    # UTILITIES                                   #
    # =========================================== #
    @staticmethod
    def normalize_csm_density(
        rho_0: "_UnitBearingScalarLike",
        r_0: "_UnitBearingScalarLike",
        s: float,
    ) -> "_UnitBearingScalarLike":
        r"""
        Compute the CSM density normalization constant from a reference density.

        This function computes the normalization constant :math:`K_{\rm CSM}` for a power-law
        circumstellar medium (CSM) density profile of the form

        .. math::

            \rho_{\rm CSM}(r) = K_{\rm CSM} r^{-s} = \rho_0 \left(\frac{r}{r_0}\right)^{-s},

        given a reference density :math:`\rho_0` at a reference radius :math:`r_0`.

        Parameters
        ----------
        rho_0: astropy.units.Quantity or float
            The reference density of the CSM at radius :math:`r_0`. If units are provided,
            they will be taken into account. Otherwise, CGS units (``grams/cm^3``) are assumed.
        r_0: astropy.units.Quantity or float
            The reference radius at which the density is specified. If units are provided,
            they will be taken into account. Otherwise, CGS units (``cm``) are assumed.
        s: float
            The CSM density profile power-law index.

        Returns
        -------
        K_csm: astropy.units.Quantity
            The computed normalization constant :math:`K_{\rm CSM}` with units of
            ``grams * cm^(s-3)``.

        """
        # Convert inputs to CGS for internal computation.
        if isinstance(rho_0, u.Quantity):
            rho_0 = rho_0.to(u.g / u.cm**3).value
        if isinstance(r_0, u.Quantity):
            r_0 = r_0.to(u.cm).value

        # Compute K_csm in CGS.
        K_csm_cgs = rho_0 * r_0**s

        # Attach units to K_csm.
        K_csm_units = u.g * u.cm ** (s - 3)
        K_csm = K_csm_cgs * K_csm_units

        return K_csm

    @staticmethod
    def normalize_outer_ejecta_density(
        rho_0: "_UnitBearingScalarLike",
        v_0: "_UnitBearingScalarLike",
        t_0: "_UnitBearingScalarLike",
        n: float,
    ) -> "_UnitBearingScalarLike":
        r"""
        Compute the ejecta density normalization constant from a reference density.

        This function computes the normalization constant :math:`K_{\rm ej}` for a power-law
        outer ejecta density profile of the form

        .. math::

            \rho_{\rm ej}(r,t) = K_{\rm ej} t^{-3} \left(\frac{r}{t}\right)^{-n} =
            \rho_0 \left(\frac{r/t}{v_0}\right)^{-n} left(\frac{t}{t_0}\right)^{-3},

        given a reference density :math:`\rho_0` at a reference velocity :math:`v_0` and time :math:`t_0`.

        Parameters
        ----------
        rho_0: astropy.units.Quantity or float
            The reference density of the ejecta at velocity :math:`v_0` and time :math:`t_0`. If units are provided,
            they will be taken into account. Otherwise, CGS units (``grams/cm^3``) are assumed.
        v_0: astropy.units.Quantity or float
            The reference velocity at which the density is specified. If units are provided,
            they will be taken into account. Otherwise, CGS units (``cm/s``) are assumed.
        t_0: astropy.units.Quantity or float
            The reference time at which the density is specified. If units are provided,
            they will be taken into account. Otherwise, CGS units (``s``) are assumed.
        n: float
            The outer ejecta density profile power-law index.

        Returns
        -------
        K_ej: astropy.units.Quantity
            The computed normalization constant :math:`K_{\rm ej}` with units of
            ``grams * cm^(n-3) * s^(3-n)``.

        """
        # Convert inputs to CGS for internal computation.
        if isinstance(rho_0, u.Quantity):
            rho_0 = rho_0.to(u.g / u.cm**3).value
        if isinstance(v_0, u.Quantity):
            v_0 = v_0.to(u.cm / u.s).value
        if isinstance(t_0, u.Quantity):
            t_0 = t_0.to(u.s).value

        # Compute K_ej in CGS.
        K_ej_cgs = rho_0 * v_0**n * t_0**3

        # Attach units to K_ej.
        K_ej_units = u.g * u.cm ** (n - 3) * u.s ** (3 - n)
        K_ej = K_ej_cgs * K_ej_units

        return K_ej


class ChevalierSelfSimilarWindShockEngine(ChevalierSelfSimilarShockEngine):
    r"""
    Specialized version of :class:`ChevalierSelfSimilarShockEngine` for steady-wind CSM.

    In the case of a steady wind CSM with injection rate :math:`\dot{M}` and characteristic velocity
    :math:`v_{\rm wind}`, the corresponding CSM density profile is

    .. math::

        \rho_{\rm CSM}(r) = \frac{\\dot{M}}{4\pi r^2 v_{\rm wind}}.

    This corresponds to a conventional :class:`ChevalierSelfSimilarShockEngine` with CSM density power-law index
    :math:`s = 2` and normalization constant

    .. math::

        K_{\rm CSM} = \frac{\dot{M}}{4\pi v_{\rm wind}}.

    .. note::

        A derivation of this model can be found on the :ref:`supernova_shocks_theory` guide. Much of the
        relevant detail omitted here can be found there.

    """

    # =========================================== #
    # Initialization                              #
    # =========================================== #
    def __init__(self, **kwargs):
        """
        Instantiate the :class:`ChevalierSelfSimilarWindShockEngine`.

        Parameters
        ----------
        kwargs:
            Additional parameters for initializing the shock engine. Currently, no additional
            parameters are required.
        """
        super().__init__(**kwargs)

    def compute_shock_properties(
        self,
        time: "_UnitBearingArrayLike",
        E_ej: "_UnitBearingScalarLike" = 1e51 * u.erg,
        M_ej: "_UnitBearingScalarLike" = 10 * u.Msun,
        M_dot: "_UnitBearingScalarLike" = 1e-5 * u.Msun / u.yr,
        v_wind: "_UnitBearingScalarLike" = 1000 * u.km / u.s,
        n: float = 10.0,
        delta: float = 0.0,
    ) -> dict[str, u.Quantity]:
        """
        Compute the shock properties at a given time.

        This method calculates the shock radius and velocity as a function of time since the
        explosion. See the class documentation (:class:`ChevalierSelfSimilarShockEngine`) for details
        on the relevant theory and assumptions.

        Parameters
        ----------
        time: ~astropy.units.Quantity or float or numpy.ndarray
            The time(s) at which to evaluate the shock properties. If units are provided,
            they will be taken into account. Otherwise, CGS units (seconds) are assumed.
            If ``time`` is provided as an array of shape ``(N,)``, the results will all have
            corresponding shapes ``(N,)``.
        E_ej: ~astropy.units.Quantity or float
            The total energy in the ejecta from the explosion. If units are provided,
            they will be taken into account. Otherwise, CGS units (erg) are assumed.
        M_ej: ~astropy.units.Quantity or float
            The total mass in the ejecta from the explosion. If units are provided,
            they will be taken into account. Otherwise, CGS units (grams) are assumed.
        M_dot: ~astropy.units.Quantity or float
            The mass-loss rate of the progenitor star's wind. If units are provided,
            they will be taken into account. Otherwise, CGS units (grams/second) are assumed.
            By default, this is set to ``1e-5 Msun/yr``.
        v_wind: ~astropy.units.Quantity or float
            The velocity of the progenitor star's wind. If units are provided,
            they will be taken into account. Otherwise, CGS units (cm/second) are assumed. By
            default, this is set to ``1000 km/s``.
        n: float, optional
            The outer ejecta density profile power-law index. Default is ``10.0``. Must be steeper than
            5 for convergence.
        delta: float, optional
            The inner ejecta density profile power-law index. Default is ``0.0``. Must be less than
            3 for convergence. In general, it is suitable to use ``0.0``.

        Returns
        -------
        dict of str, ~astropy.units.Quantity
            A dictionary containing the computed shock properties:

            - 'radius': The shock radius at the given time(s) with units of cm.
            - 'velocity': The shock velocity at the given time(s) with units of cm/s.

        """
        # Scale everything down to CGS for internal computation.
        if isinstance(E_ej, u.Quantity):
            E_ej = E_ej.to(u.erg).value
        if isinstance(M_ej, u.Quantity):
            M_ej = M_ej.to(u.g).value
        if isinstance(time, u.Quantity):
            time = time.to(u.s).value
        if isinstance(M_dot, u.Quantity):
            M_dot = M_dot.to(u.g / u.s).value
        if isinstance(v_wind, u.Quantity):
            v_wind = v_wind.to(u.cm / u.s).value

        # Perform checks on ``n``, ``s``, and ``delta`` to ensure convergence.
        if delta >= 3:
            raise ValueError("The inner ejecta density profile index `delta` must be less than 3 for convergence.")
        if n <= 5:
            raise ValueError("The outer ejecta density profile index `n` must be greater than 5 for convergence.")

        # Call the internal CGS computation method.
        shock_properties_cgs = self._compute_shock_properties_cgs(
            time=time, E_ej=E_ej, M_ej=M_ej, M_dot=M_dot, v_wind=v_wind, n=n, delta=delta
        )

        # Attach units to the outputs.
        shock_properties = {
            "radius": shock_properties_cgs["radius"] * u.cm,
            "velocity": shock_properties_cgs["velocity"] * (u.cm / u.s),
        }

        return shock_properties

    def _compute_shock_properties_cgs(
        self,
        time: "_ArrayLike",
        E_ej: float = 1e51,
        M_ej: float = 1e34,
        M_dot: float = 6.3e20,
        v_wind: float = 1e8,
        n: float = 10.0,
        delta: float = 0.0,
    ):
        """
        Compute the shock properties at a given time in CGS units.

        This method computes the shock radius and velocity at a given time based on the
        Chevalier self-similar solution for supernova ejecta interacting with a circumstellar
        medium (CSM). The ejecta and CSM are assumed to follow power-law density profiles.

        Parameters
        ----------
        time: array-like
            The time(s) at which to evaluate the shock properties in seconds. This can be a scalar or
            an array of times. The results will match the shape of the input time array.
        E_ej: float
            The total kinetic energy of the ejecta in erg. Default is ``1e51``.
        M_ej: float
            The total mass of the ejecta in grams. Default is ``1e34``.
        M_dot: float
            The mass-loss rate of the progenitor star's wind in grams/second. Default is
            ``1e-5 Msun/yr`` in CGS units.
        v_wind: float
            The velocity of the progenitor star's wind in cm/second. Default is
            ``1000 km/s`` in CGS units.
        n: float
            The outer ejecta density profile power-law index. Default is ``10.0``.
        delta:
            The inner ejecta density profile power-law index. Default is ``0.0``.

        Returns
        -------
        dict of str, array-like
            A dictionary containing the computed shock properties:

            - 'radius': The shock radius at the given time(s) in cm.
            - 'velocity': The shock velocity at the given time(s) in cm/s.

        """
        # Compute the correct K_CSM for the wind-like CSM
        K_csm = M_dot / (4.0 * np.pi * v_wind)

        # Now just pass off to the super-class
        return super()._compute_shock_properties_cgs(
            time=time,
            E_ej=E_ej,
            M_ej=M_ej,
            K_csm=K_csm,
            n=n,
            s=2.0,
            delta=delta,
        )


class NumericalThinShellShockEngine(ShockEngine):
    r"""
    Base class for numerically integrated thin-shell shock models with arbitrary ejecta and CSM profiles.

    This :class:`~dynamics.shock_engine.ShockEngine` subclass implements a general thin-shell shock model
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
            Additional keyword arguments passed to the base :class:`~dynamics.shock_engine.ShockEngine` class.
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
