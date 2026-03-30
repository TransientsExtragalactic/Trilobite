"""
Ejecta property computation for supernova models.

Function in this module compute various properties of the supernova ejecta and ambient CSM in
different physical scenarios of interest. They can then be baked into the models of :mod:`triceratops.models`.
"""

from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u

from ..shocks.shock_engine import ShockEngine
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

    This :class:`~triceratops.dynamics.shock_engine.ShockEngine` subclass implements the self-similar shock solutions
    described in :footcite:t:`chevalierSelfsimilarSolutionsInteraction1982` for the interaction between
    supernova ejecta and a surrounding circumstellar medium (CSM). The model assumes power-law density profiles
    for both the ejecta and the CSM, leading to a self-similar evolution of the shock structure over time.

    The engine calculates the position and velocity of the shock-interface as functions of time, based on
    the ejecta and CSM density profiles. It can be used to model the dynamical evolution of supernova remnants
    and their interaction with the surrounding medium.

    .. note::

        A derivation of this model can be found on the :ref:`chevalier_theory` guide. Much of the
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
        :ref:`chevalier_theory`.
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

        A derivation of this model can be found on the :ref:`chevalier_theory` guide. Much of the
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
