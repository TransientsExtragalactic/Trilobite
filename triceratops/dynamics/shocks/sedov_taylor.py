r"""
Sedov--Taylor blast-wave shock dynamics.

This module provides the normalization function :func:`sedov_taylor_beta` and the
:class:`SedovTaylorShockEngine`, which implements the self-similar Sedov--Taylor
solution for a point explosion of energy :math:`E` in a uniform ambient medium of
density :math:`\rho_0`. The shock radius and velocity evolve as

.. math::

    R_s(t) = \beta(\gamma)\left(\frac{E\,t^2}{\rho_0}\right)^{1/5}, \qquad
    v_s(t) = \frac{2}{5}\frac{R_s(t)}{t},

where :math:`\beta(\gamma)` is evaluated by numerical integration of the
self-similar Sedov profiles.

All internal calculations are performed in CGS units. Public methods accept
:class:`astropy.units.Quantity` inputs and return unit-bearing quantities.
"""

import warnings
from typing import NamedTuple, Union

import numpy as np
from astropy import units as u
from scipy.integrate import IntegrationWarning, quad

from triceratops._typing import _ArrayLike, _UnitBearingArrayLike, _UnitBearingScalarLike
from triceratops.dynamics.shocks.core.rankine_hugoniot import StrongColdShockConditions
from triceratops.dynamics.shocks.core.shock_engine import ShockEngine
from triceratops.utils.misc_utils import ensure_in_units


def sedov_taylor_beta(gamma, *, eps=1e-10, epsabs=1e-12, epsrel=1e-12, limit=500):
    r"""
    Compute the Sedov--Taylor shock-radius normalization coefficient :math:`\beta(\gamma)`.

    For a point explosion of energy :math:`E` in a uniform ambient medium of density
    :math:`\rho_0`, the shock radius evolves as

    .. math::

        R_s(t) = \beta(\gamma) \left(\frac{E\,t^2}{\rho_0}\right)^{1/5}.

    The coefficient :math:`\beta` is fixed by requiring that the total kinetic plus
    thermal energy of the self-similar solution equals :math:`E`.

    Parameters
    ----------
    gamma : float
        Adiabatic index of the gas. Common values are :math:`5/3` (monatomic ideal gas)
        and :math:`7/5 = 1.4` (diatomic ideal gas). Must satisfy :math:`\gamma > 1`
        and :math:`\gamma \neq 2`; the explicit similarity formulae are singular at
        :math:`\gamma = 2`.
    eps : float, optional
        Fractional offset from the lower endpoint of the similarity interval.
        The integration range is

        .. math::

            V \in \left[\frac{1}{\gamma} + \varepsilon\,\Delta V,\;
                        \frac{2}{\gamma+1}\right], \qquad
            \Delta V = \frac{2}{\gamma+1} - \frac{1}{\gamma}.

        The lower endpoint corresponds to the origin, where some explicit
        similarity expressions are numerically singular.
    epsabs, epsrel : float, optional
        Absolute and relative tolerances passed to :func:`scipy.integrate.quad`.
    limit : int, optional
        Maximum number of subintervals used by :func:`scipy.integrate.quad`.

    Returns
    -------
    beta : float
        Normalization coefficient in
        :math:`R_s(t) = \beta\,(E\,t^2/\rho_0)^{1/5}`.

    Notes
    -----
    The standard Sedov similarity variable is :math:`\xi = r / R_s(t)`. In the
    parametric representation the physical fields are

    .. math::

        u(r,t) = \frac{2r}{5t}\,V(\xi), \qquad
        \rho(r,t) = \rho_0\,G(\xi), \qquad
        c(r,t)^2 = \left(\frac{2r}{5t}\right)^2 Z(\xi),

    with :math:`c^2 = \gamma p/\rho`. Integrating the energy density
    :math:`e = \tfrac{1}{2}\rho u^2 + p/(\gamma-1)` over the blast gives

    .. math::

        E = \frac{\rho_0 R_s^5}{t^2}\,C_E(\gamma), \qquad
        C_E = \frac{16\pi}{25}
              \int_0^1 G(\xi)
              \left[\frac{V(\xi)^2}{2} + \frac{Z(\xi)}{\gamma(\gamma-1)}\right]
              \xi^4\,\mathrm{d}\xi.

    Because :math:`R_s^5/t^2 = \beta^5 E/\rho_0`, the normalization condition reduces to
    :math:`1 = \beta^5 C_E`, hence :math:`\beta = C_E^{-1/5}`.

    The explicit Sedov solution expresses :math:`\xi`, :math:`G`, and :math:`Z` as
    functions of :math:`V`, so the integral is evaluated via
    :math:`\int F(\xi)\,\mathrm{d}\xi = \int F(V)\,(\mathrm{d}\xi/\mathrm{d}V)\,\mathrm{d}V`
    over :math:`V_0 = 1/\gamma` (origin) to :math:`V_s = 2/(\gamma+1)` (shock).

    Some references write :math:`R_s(t) = (E\,t^2/\alpha\rho_0)^{1/5}`, in which
    case :math:`\alpha = \beta^{-5} = C_E`. For :math:`\gamma = 5/3`:
    :math:`\beta \approx 1.1517`, :math:`\beta^5 \approx 2.026`,
    :math:`\alpha \approx 0.4936`.

    Examples
    --------
    Evaluate the normalization coefficient for common adiabatic indices:

    .. code-block:: python

        beta_monoatomic = sedov_taylor_beta(5.0 / 3.0)
        beta_diatomic = sedov_taylor_beta(1.4)

        print(beta_monoatomic)
        print(beta_diatomic)

    which gives approximately:

    .. code-block:: text

        1.1517...
        1.033...
    """
    g = float(gamma)

    if not np.isfinite(g):
        raise ValueError("gamma must be finite.")
    if g <= 1.0:
        raise ValueError("gamma must be greater than 1 for the Sedov solution.")
    if np.isclose(g, 2.0):
        raise ValueError("This explicit filled-solution formula is singular at gamma = 2.")

    V0 = 1.0 / g
    Vs = 2.0 / (g + 1.0)

    if not (Vs > V0):
        raise ValueError("Expected Vs = 2/(gamma+1) > V0 = 1/gamma; check input gamma.")

    dV = Vs - V0

    # Exponents from the explicit Sedov similarity solution.
    nu1 = -(13.0 * g**2 - 7.0 * g + 12.0) / ((3.0 * g - 1.0) * (2.0 * g + 1.0))
    nu2 = 5.0 * (g - 1.0) / (2.0 * g + 1.0)
    nu3 = 3.0 / (2.0 * g + 1.0)
    nu4 = -nu1 / (2.0 - g)
    nu5 = -2.0 / (2.0 - g)

    def _profiles(V):
        A = 0.5 * (g + 1.0) * V
        B = ((g + 1.0) / (7.0 - g)) * (5.0 - (3.0 * g - 1.0) * V)
        C = ((g + 1.0) / (g - 1.0)) * (g * V - 1.0)
        D = ((g + 1.0) / (g - 1.0)) * (1.0 - V)
        xi = (A**-2.0 * B**nu1 * C**nu2) ** 0.2
        G = ((g + 1.0) / (g - 1.0)) * C**nu3 * B**nu4 * D**nu5
        Z = g * (g - 1.0) * (1.0 - V) * V**2 / (2.0 * (g * V - 1.0))
        return xi, G, Z

    def _dxi_dV(V):
        xi, _, _ = _profiles(V)
        dlogxi = (1.0 / 5.0) * (
            -2.0 / V - nu1 * (3.0 * g - 1.0) / (5.0 - (3.0 * g - 1.0) * V) + nu2 * g / (g * V - 1.0)
        )
        return xi * dlogxi

    def _integrand(V):
        xi, G, Z = _profiles(V)
        return G * (0.5 * V**2 + Z / (g * (g - 1.0))) * xi**4 * _dxi_dV(V)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        integral, _ = quad(
            _integrand,
            V0 + eps * dV,
            Vs,
            epsabs=epsabs,
            epsrel=epsrel,
            limit=limit,
        )

    C_E = (16.0 * np.pi / 25.0) * integral

    if not np.isfinite(C_E) or C_E <= 0.0:
        raise RuntimeError(
            f"Computed invalid Sedov energy constant C_E={C_E}. Try increasing eps or relaxing quadrature tolerances."
        )

    return C_E ** (-1.0 / 5.0)


# ============================================================ #
# State Classes                                                #
# ============================================================ #
class SedovTaylorShockState(NamedTuple):
    r"""
    Time-dependent state returned by :class:`SedovTaylorShockEngine`.

    The low-level CGS interface returns plain :class:`numpy.ndarray` fields in
    CGS units. The public unit-aware interface returns the same structure with
    fields converted to :class:`astropy.units.Quantity`.
    """

    radius: Union[np.ndarray, u.Quantity]
    r"""
    Shock radius :math:`R_s(t)` in cm.
    """

    velocity: Union[np.ndarray, u.Quantity]
    r"""
    Shock velocity :math:`v_s(t) = \dot{R}_s(t)` in cm/s.
    """

    post_shock_density: Union[np.ndarray, u.Quantity]
    r"""
    Immediate post-shock density :math:`\rho_s` in :math:`\mathrm{g\,cm^{-3}}`,
    derived from the strong-shock Rankine--Hugoniot relation
    :math:`\rho_s = (\gamma+1)/(\gamma-1)\,\rho_0`.
    """

    post_shock_pressure: Union[np.ndarray, u.Quantity]
    r"""
    Immediate post-shock pressure :math:`p_s` in :math:`\mathrm{dyn\,cm^{-2}}`,
    derived from the strong-shock Rankine--Hugoniot relation
    :math:`p_s = 2\rho_0 v_s^2/(\gamma+1)`.
    """

    post_shock_temperature: Union[np.ndarray, u.Quantity]
    r"""
    Immediate post-shock temperature :math:`T_s` in K, derived from the ideal
    gas relation :math:`T_s = p_s \mu m_p / (\rho_s k_B)`.
    """

    thermal_energy_density: Union[np.ndarray, u.Quantity]
    r"""
    Post-shock thermal energy density
    :math:`e_{\rm th} = p_s / (\gamma - 1)` in :math:`\mathrm{erg\,cm^{-3}}`.
    """


# ============================================================ #
# Sedov--Taylor Shock Engine                                   #
# ============================================================ #
class SedovTaylorShockEngine(ShockEngine):
    r"""
    Sedov--Taylor blast-wave shock engine for a point explosion in a uniform medium.

    This :class:`~triceratops.dynamics.shocks.core.shock_engine.ShockEngine` subclass
    implements the self-similar Sedov--Taylor solution for a point explosion of energy
    :math:`E` expanding into a uniform ambient medium of density :math:`\rho_0`. The
    shock radius and velocity evolve as

    .. math::

        R_s(t) = \beta(\gamma)\left(\frac{E\,t^2}{\rho_0}\right)^{1/5}, \qquad
        v_s(t) = \frac{2}{5}\frac{R_s(t)}{t},

    where :math:`\beta(\gamma)` is computed once at instantiation by
    :func:`sedov_taylor_beta`.

    In addition to the shock kinematics, the engine returns immediate post-shock
    thermodynamic quantities derived from the strong-shock Rankine--Hugoniot conditions:

    .. math::

        \rho_s = \frac{\gamma+1}{\gamma-1}\,\rho_0, \qquad
        p_s    = \frac{2}{\gamma+1}\,\rho_0\,v_s^2, \qquad
        e_{\rm th} = \frac{p_s}{\gamma - 1}.

    The post-shock temperature follows from the ideal-gas law,

    .. math::

        T_s = \frac{p_s\,\mu\,m_p}{\rho_s\,k_B},

    where :math:`\mu` is the mean molecular weight (in units of the proton mass
    :math:`m_p`) and :math:`k_B` is Boltzmann's constant.

    Parameters
    ----------
    gamma : float, optional
        Adiabatic index of the ambient gas. Default is :math:`5/3` (monatomic ideal
        gas). Must satisfy :math:`\gamma > 1` and :math:`\gamma \neq 2`.
    mu : float, optional
        Mean molecular weight in units of the proton mass :math:`m_p`. Default is
        ``0.5``, appropriate for a fully ionized hydrogen plasma.
    **integration_kwargs
        Keyword arguments forwarded to :func:`sedov_taylor_beta` for the numerical
        integration that evaluates :math:`\beta(\gamma)`. Available keys:

        - ``eps`` *(float)* — fractional offset from the singular lower endpoint of
          the similarity interval. Default ``1e-10``.
        - ``epsabs`` *(float)* — absolute quadrature tolerance. Default ``1e-12``.
        - ``epsrel`` *(float)* — relative quadrature tolerance. Default ``1e-12``.
        - ``limit`` *(int)* — maximum number of quadrature subintervals. Default ``500``.

    See Also
    --------
    sedov_taylor_beta
        Normalization coefficient used by this engine.
    ChevalierSelfSimilarShockEngine
        Analogous engine for power-law ejecta interacting with a power-law CSM.
    """

    _STATE_CLASS = SedovTaylorShockState

    def __init__(self, gamma: float = 5.0 / 3.0, mu: float = 0.5, **integration_kwargs):
        """
        Instantiate the :class:`SedovTaylorShockEngine`.

        Parameters
        ----------
        gamma : float, optional
            Adiabatic index of the gas. Default is :math:`5/3`.
        mu : float, optional
            Mean molecular weight in units of the proton mass. Default is ``0.5``.
        **integration_kwargs
            Forwarded to :func:`sedov_taylor_beta`; see class docstring for available keys.
        """
        super().__init__()
        self._gamma = float(gamma)
        self._mu = float(mu)
        self._beta = sedov_taylor_beta(self._gamma, **integration_kwargs)

    # ============================================================ #
    # Core Shock Engine Methods                                    #
    # ============================================================ #
    def compute_shock_properties(
        self,
        time: "_UnitBearingArrayLike",
        E: "_UnitBearingScalarLike" = 1e51 * u.erg,
        rho_0: "_UnitBearingScalarLike" = 1.67e-24 * u.g / u.cm**3,
    ) -> dict[str, u.Quantity]:
        r"""
        Compute the shock properties at one or more times.

        Parameters
        ----------
        time : ~astropy.units.Quantity or float or numpy.ndarray
            Time(s) since the explosion. If a :class:`~astropy.units.Quantity`, units
            are converted automatically; otherwise seconds (CGS) are assumed.
        E : ~astropy.units.Quantity or float, optional
            Total explosion energy. If a :class:`~astropy.units.Quantity`, units are
            converted automatically; otherwise erg (CGS) are assumed.
            Default is :math:`10^{51}` erg.
        rho_0 : ~astropy.units.Quantity or float, optional
            Uniform ambient density. If a :class:`~astropy.units.Quantity`, units are
            converted automatically; otherwise :math:`\mathrm{g\,cm^{-3}}` (CGS) are
            assumed. Default is :math:`1.67 \times 10^{-24}\ \mathrm{g\,cm^{-3}}`
            (approximately one hydrogen atom per :math:`\mathrm{cm^3}`).

        Returns
        -------
        dict of str, ~astropy.units.Quantity
            A dictionary with the following keys:

            - ``'radius'``: shock radius :math:`R_s(t)` in :math:`\mathrm{cm}`.
            - ``'velocity'``: shock velocity :math:`v_s(t)` in :math:`\mathrm{cm\,s^{-1}}`.
            - ``'post_shock_density'``: immediate post-shock density :math:`\rho_s` in
              :math:`\mathrm{g\,cm^{-3}}`.
            - ``'post_shock_pressure'``: immediate post-shock pressure :math:`p_s` in
              :math:`\mathrm{dyn\,cm^{-2}}`.
            - ``'post_shock_temperature'``: immediate post-shock temperature :math:`T_s`
              in :math:`\mathrm{K}`.
            - ``'thermal_energy_density'``: post-shock thermal energy density
              :math:`e_{\rm th} = p_s/(\gamma-1)` in :math:`\mathrm{erg\,cm^{-3}}`.
        """
        t_cgs = ensure_in_units(time, u.s)
        E_cgs = ensure_in_units(E, u.erg)
        rho_0_cgs = ensure_in_units(rho_0, u.g / u.cm**3)

        props = self._compute_shock_properties_cgs(t_cgs, E_cgs, rho_0_cgs)

        return SedovTaylorShockState(
            radius=props.radius * u.cm,
            velocity=props.velocity * (u.cm / u.s),
            post_shock_density=props.post_shock_density * (u.g / u.cm**3),
            post_shock_pressure=props.post_shock_pressure * (u.dyne / u.cm**2),
            post_shock_temperature=props.post_shock_temperature * u.K,
            thermal_energy_density=props.thermal_energy_density * (u.erg / u.cm**3),
        )

    def _compute_shock_properties_cgs(
        self,
        time: "_ArrayLike",
        E: float = 1e51,
        rho_0: float = 1.67e-24,
    ) -> dict[str, "_ArrayLike"]:
        r"""
        Compute the shock properties in CGS units.

        Parameters
        ----------
        time : float or array-like
            Time(s) since the explosion in seconds.
        E : float, optional
            Total explosion energy in erg. Default is ``1e51``.
        rho_0 : float, optional
            Uniform ambient density in :math:`\mathrm{g\,cm^{-3}}`. Default is ``1.67e-24``.

        Returns
        -------
        dict of str, float or array-like
            Keys ``'radius'``, ``'velocity'``, ``'post_shock_density'``,
            ``'post_shock_pressure'``, ``'post_shock_temperature'``, and
            ``'thermal_energy_density'`` in CGS units.
        """
        R = self._beta * (E * time**2 / rho_0) ** 0.2
        v = 0.4 * R / time

        rh = StrongColdShockConditions._solve(v, rho_0, gamma=self._gamma, mu=self._mu)

        return SedovTaylorShockState(
            radius=R,
            velocity=v,
            post_shock_density=rh["post_shock_density"],
            post_shock_pressure=rh["post_shock_pressure"],
            post_shock_temperature=rh["post_shock_temperature"],
            thermal_energy_density=rh["post_shock_thermal_energy_density"],
        )

    # =========================================== #
    # DUNDER METHODS                              #
    # =========================================== #
    def __str__(self):
        return f"SedovTaylorShockEngine(gamma={self._gamma:.4f}, mu={self._mu:.4f})"

    def __repr__(self):
        return f"<SedovTaylorShockEngine(gamma={self._gamma:.4f}, mu={self._mu:.4f})>"
