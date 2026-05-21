r"""
Blandford--McKee ultra-relativistic blastwave shock dynamics.

This module implements the analytic Blandford--McKee (BM) self-similar solution
:footcite:t:`1976PhFl...19.1130B` for an ultra-relativistic blastwave expanding
into a stratified external medium with power-law density profile

.. math::

    \rho_{\rm ext}(r) = K r^{-k}.

For an adiabatic, energy-conserving blastwave the shock Lorentz factor satisfies

.. math::

    \Gamma^2 R^{3-k} = \frac{(17-4k)\,E}{8\pi K c^2},

where :math:`E` is the (isotropic-equivalent) explosion energy and :math:`R \approx ct`
is the shock radius in the ultra-relativistic limit.

The BM self-similar profiles at the shock front (:math:`\chi = 1`) reduce to the
standard relativistic Rankine--Hugoniot conditions, which are evaluated by
:class:`~triceratops.dynamics.shocks.core.relativistic_jump_conditions.UltraRelativisticColdShockConditions`.

The primary classes are:

- :class:`BlandfordMcKeeShockEngine`: general power-law external medium.
- :class:`BlandfordMcKeeWindShockEngine`: convenience specialisation for a steady
  stellar wind (:math:`k = 2`) parameterised by mass-loss rate and wind velocity.

Notes
-----
The BM solution is valid only in the ultra-relativistic limit :math:`\Gamma \gg 1`.
Both classes emit a :class:`RuntimeWarning` when the Lorentz factor drops below a
user-configurable threshold (default :math:`\Gamma = 2`). The solution is undefined
for :math:`k \geq 3` (the swept-up mass diverges), and a :class:`ValueError` is
raised in that case.

All internal calculations are performed in CGS units. Public methods accept
:class:`astropy.units.Quantity` inputs and return unit-bearing quantities.

References
----------
.. footbibliography::
"""

import warnings
from typing import NamedTuple, Union

import astropy.constants as const
import numpy as np
from astropy import units as u

from triceratops._typing import _ArrayLike, _UnitBearingArrayLike, _UnitBearingScalarLike
from triceratops.dynamics.shocks.core.relativistic_jump_conditions import _maxwell_juttner_temperature_cgs
from triceratops.dynamics.shocks.core.shock_engine import ShockEngine

# --- CGS constants for the low-level backend ---
c_cgs = const.c.cgs.value  # cm s⁻¹
m_p_cgs = const.m_p.cgs.value  # g


# ============================================================ #
# State Class                                                   #
# ============================================================ #


class BlandfordMcKeeShockState(NamedTuple):
    r"""Time-dependent state returned by Blandford--McKee shock engines.

    Fields that carry physical dimensions are returned as
    :class:`~astropy.units.Quantity` by the public interface and as plain
    :class:`~numpy.ndarray` by the CGS backend.

    Notes
    -----
    Post-shock quantities are evaluated by
    :class:`~triceratops.dynamics.shocks.core.relativistic_jump_conditions.UltraRelativisticColdShockConditions`
    at the shock surface (:math:`\chi = 1`), which corresponds to a cold,
    ultra-relativistic upstream medium.

    Proper (comoving-frame) thermodynamic quantities are Lorentz scalars.
    Lab-frame quantities are defined in the rest frame of the unshocked external
    medium.
    """

    radius: Union[np.ndarray, u.Quantity]
    r"""Shock radius :math:`R(t) \approx ct` in cm."""

    lorentz_factor: Union[np.ndarray, u.Quantity]
    r"""Shock Lorentz factor :math:`\Gamma(t)` [dimensionless]."""

    beta: Union[np.ndarray, u.Quantity]
    r"""Shock velocity in units of :math:`c`, :math:`\beta = v/c` [dimensionless]."""

    velocity: Union[np.ndarray, u.Quantity]
    r"""Shock velocity :math:`v = \beta c` in cm/s."""

    fluid_lorentz_factor: Union[np.ndarray, u.Quantity]
    r"""
    Lorentz factor of the immediately post-shock fluid,
    :math:`\gamma_2 \approx \Gamma / \sqrt{2}` [dimensionless].
    """

    post_shock_pressure: Union[np.ndarray, u.Quantity]
    r"""
    Proper (comoving-frame) post-shock pressure
    :math:`p_2 \simeq \tfrac{2}{3}\rho_{\rm ext}c^2\Gamma^2`
    in :math:`\mathrm{dyne\,cm^{-2}}`.
    """

    post_shock_temperature: Union[np.ndarray, u.Quantity]
    r"""
    Proper (comoving-frame) post-shock temperature :math:`T_2` in K,
    from the Maxwell--Jüttner internal-energy inversion.
    """

    post_shock_comoving_density: Union[np.ndarray, u.Quantity]
    r"""
    Proper (comoving-frame) post-shock rest-mass density :math:`\rho_2`
    in :math:`\mathrm{g\,cm^{-3}}`.
    """

    post_shock_lab_density: Union[np.ndarray, u.Quantity]
    r"""
    Lab-frame post-shock rest-mass density
    :math:`\gamma_2\rho_2`
    in :math:`\mathrm{g\,cm^{-3}}`.
    """

    thermal_energy_density_comoving: Union[np.ndarray, u.Quantity]
    r"""
    Proper (comoving-frame) post-shock thermal (internal) energy density
    :math:`U_{\rm int,2} = p_2 / (\hat{\gamma} - 1)`
    in :math:`\mathrm{erg\,cm^{-3}}`.
    """

    thermal_energy_density_lab: Union[np.ndarray, u.Quantity]
    r"""
    Lab-frame post-shock energy density
    :math:`T^{00} = (e_2 + p_2)\gamma_2^2 - p_2`
    in :math:`\mathrm{erg\,cm^{-3}}`.
    """


# ============================================================ #
# Main Engine                                                   #
# ============================================================ #


class BlandfordMcKeeShockEngine(ShockEngine):
    r"""
    Blandford--McKee ultra-relativistic blastwave shock engine.

    Implements the analytic self-similar solution of
    :footcite:t:`1976PhFl...19.1130B` for a relativistic blastwave of isotropic-
    equivalent energy :math:`E` propagating into a stratified external medium

    .. math::

        \rho_{\rm ext}(r) = K r^{-k},

    where :math:`k < 3`.

    .. note::

        A full derivation of this model is given on the :ref:`blandford_mckee_theory`
        reference page.

    **Physics summary:**

    Energy conservation for the adiabatic BM solution gives

    .. math::

        \Gamma^2(R)
        =
        \frac{(17 - 4k)\,E}{8\pi K c^2 R^{3-k}},

    where the analytic normalization coefficient

    .. math::

        C_E(k) = \frac{8\pi}{17 - 4k}

    follows from integrating the self-similar energy density
    :math:`T^{00} \approx 4p\gamma^2` over the shocked shell using the closed-form
    profiles

    .. math::

        g(\chi) = \chi^{-1},
        \qquad
        f(\chi) = \chi^{(4k-17)/[3(4-k)]},
        \qquad
        h(\chi) = \chi^{(2k-7)/(4-k)},

    with similarity variable
    :math:`\chi = 1 + 2(m+1)\Gamma^2(1 - r/R)`, :math:`m = 3 - k`.

    In the ultra-relativistic limit the shock radius is :math:`R \approx ct`, so

    .. math::

        \Gamma^2(t)
        \approx
        \frac{(17 - 4k)\,E}{8\pi K c^{5-k} t^{3-k}}.

    Post-shock thermodynamic quantities are evaluated at :math:`\chi = 1` by
    :class:`~triceratops.dynamics.shocks.core.relativistic_jump_conditions.UltraRelativisticColdShockConditions`.

    **Validity:**

    The BM solution requires :math:`\Gamma \gg 1` and :math:`k < 3`. A
    :class:`RuntimeWarning` is emitted when :math:`\Gamma` falls below
    ``lorentz_warn_threshold`` (default 2). A :class:`ValueError` is raised for
    :math:`k \geq 3`.

    Parameters
    ----------
    mu : float, optional
        Mean molecular weight in units of the proton mass used for the post-shock
        temperature calculation. Default is ``0.61`` (fully ionized, solar-composition
        plasma).
    gamma_hat : float, optional
        Adiabatic index of the shocked gas. Default is :math:`4/3` (ultra-relativistic
        EOS).
    lorentz_warn_threshold : float, optional
        Lorentz factor below which a :class:`RuntimeWarning` is emitted. Default
        is ``2.0``.

    See Also
    --------
    BlandfordMcKeeWindShockEngine :
        Specialisation for a steady stellar wind (:math:`k = 2`).

    References
    ----------
    .. footbibliography::
    """

    _STATE_CLASS = BlandfordMcKeeShockState

    # =========================================== #
    # Initialization                              #
    # =========================================== #

    def __init__(
        self,
        mu: float = 0.61,
        gamma_hat: float = 4.0 / 3.0,
        lorentz_warn_threshold: float = 2.0,
        **kwargs,
    ):
        r"""
        Instantiate the :class:`BlandfordMcKeeShockEngine`.

        Parameters
        ----------
        mu : float, optional
            Mean molecular weight in units of the proton mass. Default is ``0.61``.
        gamma_hat : float, optional
            Adiabatic index of the shocked gas. Default is :math:`4/3`.
        lorentz_warn_threshold : float, optional
            Emit a :class:`RuntimeWarning` when :math:`\Gamma` falls below this
            value. Default is ``2.0``.
        **kwargs
            Forwarded to :class:`~triceratops.dynamics.shocks.core.shock_engine.ShockEngine`.
        """
        super().__init__(**kwargs)
        self._mu = float(mu)
        self._gamma_hat = float(gamma_hat)
        self._lorentz_warn_threshold = float(lorentz_warn_threshold)

    # =========================================== #
    # Utilities                                   #
    # =========================================== #

    @staticmethod
    def normalize_csm_density(
        rho_0: "_UnitBearingScalarLike",
        r_0: "_UnitBearingScalarLike",
        k: float,
    ) -> u.Quantity:
        r"""
        Compute the CSM normalization constant from a reference density.

        Returns :math:`K` such that

        .. math::

            \rho_{\rm ext}(r) = K r^{-k} = \rho_0 \left(\frac{r}{r_0}\right)^{-k}.

        Parameters
        ----------
        rho_0 : ~astropy.units.Quantity or float
            Reference density at :math:`r_0`. Bare float assumed
            :math:`\mathrm{g\,cm^{-3}}`.
        r_0 : ~astropy.units.Quantity or float
            Reference radius. Bare float assumed cm.
        k : float
            CSM power-law index.

        Returns
        -------
        K : ~astropy.units.Quantity
            Normalization constant with units
            :math:`\mathrm{g\,cm^{k-3}}`.
        """
        if isinstance(rho_0, u.Quantity):
            rho_0 = rho_0.to(u.g / u.cm**3).value
        if isinstance(r_0, u.Quantity):
            r_0 = r_0.to(u.cm).value
        return (rho_0 * r_0**k) * (u.g * u.cm ** (k - 3))

    # =========================================== #
    # Public shock-property interface             #
    # =========================================== #

    def compute_shock_properties(
        self,
        time: "_UnitBearingArrayLike",
        E: "_UnitBearingScalarLike" = 1e51 * u.erg,
        K_csm: "_UnitBearingScalarLike" = None,
        k: float = 0.0,
    ) -> BlandfordMcKeeShockState:
        r"""
        Compute the BM shock state at one or more times.

        Parameters
        ----------
        time : ~astropy.units.Quantity or array-like
            Time(s) since the explosion. Bare float assumed to be in seconds.
        E : ~astropy.units.Quantity or float, optional
            Isotropic-equivalent explosion energy. Bare float assumed to be in
            erg. Default is :math:`10^{51}` erg.
        K_csm : ~astropy.units.Quantity or float, optional
            External-medium normalization constant :math:`K` such that
            :math:`\rho_{\rm ext}(r) = K r^{-k}`. Bare float assumed to have
            CGS units :math:`\mathrm{g\,cm^{k-3}}`. If ``None``, a placeholder
            value corresponding to :math:`n = 1\,\mathrm{cm^{-3}}` at
            :math:`r = 1\,\mathrm{cm}` is used; this is only suitable for
            exploratory calculations.
        k : float, optional
            Power-law index of the external density profile. Must satisfy
            :math:`k < 3`. Default is ``0.0`` (uniform ISM).

        Returns
        -------
        BlandfordMcKeeShockState
            Shock kinematics and post-shock thermodynamics at all requested
            times, with :class:`~astropy.units.Quantity` units attached to
            every field.

        Raises
        ------
        ValueError
            If :math:`k \geq 3`.

        Warns
        -----
        RuntimeWarning
            If any :math:`\Gamma(t)` falls below ``lorentz_warn_threshold``.
        """
        if k >= 3.0:
            raise ValueError(f"The CSM power-law index k must be less than 3 for the BM solution; got k = {k}.")

        if K_csm is None:
            K_csm = m_p_cgs * u.g * u.cm ** (k - 3)

        if isinstance(time, u.Quantity):
            time = time.to(u.s).value
        if isinstance(E, u.Quantity):
            E = E.to(u.erg).value
        if isinstance(K_csm, u.Quantity):
            K_csm = K_csm.to(u.g * u.cm ** (k - 3)).value

        cgs = self._compute_shock_properties_cgs(time=time, E=E, K_csm=K_csm, k=k)

        return BlandfordMcKeeShockState(
            radius=cgs.radius * u.cm,
            lorentz_factor=cgs.lorentz_factor,
            beta=cgs.beta,
            velocity=cgs.velocity * (u.cm / u.s),
            fluid_lorentz_factor=cgs.fluid_lorentz_factor,
            post_shock_pressure=cgs.post_shock_pressure * (u.dyne / u.cm**2),
            post_shock_temperature=cgs.post_shock_temperature * u.K,
            post_shock_comoving_density=cgs.post_shock_comoving_density * (u.g / u.cm**3),
            post_shock_lab_density=cgs.post_shock_lab_density * (u.g / u.cm**3),
            thermal_energy_density_comoving=cgs.thermal_energy_density_comoving * (u.erg / u.cm**3),
            thermal_energy_density_lab=cgs.thermal_energy_density_lab * (u.erg / u.cm**3),
        )

    # =========================================== #
    # CGS backend                                 #
    # =========================================== #

    def _compute_shock_properties_cgs(
        self,
        time: "_ArrayLike",
        E: float = 1e51,
        K_csm: float = m_p_cgs,
        k: float = 0.0,
    ) -> BlandfordMcKeeShockState:
        r"""
        Compute the BM shock state in CGS units.

        Parameters
        ----------
        time : array-like
            Time(s) in seconds.
        E : float
            Isotropic-equivalent explosion energy in erg.
        K_csm : float
            External-medium normalization in :math:`\mathrm{g\,cm^{k-3}}`.
        k : float
            External density power-law index. Must satisfy :math:`k < 3`.

        Returns
        -------
        BlandfordMcKeeShockState
            All fields are plain :class:`~numpy.ndarray` in CGS units.
            Dimensionless fields (:attr:`~BlandfordMcKeeShockState.lorentz_factor`,
            :attr:`~BlandfordMcKeeShockState.beta`,
            :attr:`~BlandfordMcKeeShockState.fluid_lorentz_factor`) are bare arrays.
        """
        time = np.asarray(time, dtype=float)

        # Shock radius: ultra-relativistic approximation R ≈ ct
        R = c_cgs * time

        # Shock Lorentz factor from BM energy normalization:
        #   E = [8pi/(17-4k)] * K * c^2 * Gamma^2 * R^{3-k}
        Gamma2 = (17.0 - 4.0 * k) * E / (8.0 * np.pi * K_csm * c_cgs**2 * R ** (3.0 - k))
        Gamma = np.sqrt(Gamma2)

        if np.any(Gamma < self._lorentz_warn_threshold):
            warnings.warn(
                f"The BM shock Lorentz factor Γ has fallen below {self._lorentz_warn_threshold} at one or more "
                "requested times. The ultra-relativistic approximation is breaking down; "
                "consider switching to a trans-relativistic or Newtonian solver.",
                RuntimeWarning,
                stacklevel=3,
            )

        beta = np.sqrt(1.0 - 1.0 / Gamma2)
        velocity = beta * c_cgs

        # Upstream (external medium) proper rest-mass density at the shock front
        rho_ext = K_csm * R ** (-k)

        # ------------------------------------------------------------------ #
        # Post-shock quantities from BM profiles at chi=1 (f=g=h=1).         #
        # All computed directly from Gamma^2 to avoid the catastrophic        #
        # cancellation in gamma_1 = 1/sqrt(1-beta_1^2) that the RH solver    #
        # would incur when beta_1 = beta_shock -> 1.                          #
        # ------------------------------------------------------------------ #

        # Lab-frame downstream fluid Lorentz factor: gamma_2_lab^2 = Gamma^2/2
        gamma2_lab = np.sqrt(Gamma2 / 2.0)

        # Proper post-shock pressure: p_2 = (2/3) rho_ext c^2 Gamma^2
        p2 = (2.0 / 3.0) * rho_ext * c_cgs**2 * Gamma2

        # Proper downstream rest-mass density from baryon-flux conservation.
        # In the Gamma >> 1 limit, the compression ratio is
        #   R_comp = Gamma * sqrt(gamma_hat*(2-gamma_hat)) / (gamma_hat-1),
        # which for gamma_hat=4/3 gives 2*sqrt(2)*Gamma.
        # Using gamma_hat*(2-gamma_hat) = 1 - (gamma_hat-1)^2 avoids
        # recomputing sqrt(1-beta_2^2) from beta_2 = gamma_hat-1.
        beta2_sf = self._gamma_hat - 1.0
        one_minus_beta2_sq = self._gamma_hat * (2.0 - self._gamma_hat)  # = 1 - beta2_sf^2
        R_comp = Gamma * np.sqrt(one_minus_beta2_sq) / beta2_sf
        rho2_proper = R_comp * rho_ext

        # Proper thermal (internal) energy density: U_int = p/(gamma_hat-1)
        U_int2 = p2 / (self._gamma_hat - 1.0)

        # Proper total energy density: e = rho*c^2 + U_int
        e2_total = rho2_proper * c_cgs**2 + U_int2

        # Post-shock temperature from the Maxwell-Jüttner inversion
        T2 = _maxwell_juttner_temperature_cgs(U_int2, rho2_proper, self._mu)

        # Lab-frame quantities via Lorentz transformation
        rho2_lab = gamma2_lab * rho2_proper
        e2_lab = (e2_total + p2) * gamma2_lab**2 - p2  # T^{00} = (e+p)γ² − p

        return BlandfordMcKeeShockState(
            radius=R,
            lorentz_factor=Gamma,
            beta=beta,
            velocity=velocity,
            fluid_lorentz_factor=gamma2_lab,
            post_shock_pressure=p2,
            post_shock_temperature=T2,
            post_shock_comoving_density=rho2_proper,
            post_shock_lab_density=rho2_lab,
            thermal_energy_density_comoving=U_int2,
            thermal_energy_density_lab=e2_lab,
        )


# ============================================================ #
# Wind Specialisation                                           #
# ============================================================ #


class BlandfordMcKeeWindShockEngine(BlandfordMcKeeShockEngine):
    r"""
    Blandford--McKee engine specialised for a steady stellar-wind external medium.

    For a wind with mass-loss rate :math:`\dot{M}` and terminal velocity
    :math:`v_w`, the density profile is

    .. math::

        \rho_{\rm ext}(r)
        =
        \frac{\dot{M}}{4\pi v_w r^2}
        =
        K r^{-2},
        \qquad
        K = \frac{\dot{M}}{4\pi v_w}.

    This corresponds to :class:`BlandfordMcKeeShockEngine` with :math:`k = 2`
    and normalization :math:`K = \dot{M}/(4\pi v_w)`. The BM energy relation
    becomes

    .. math::

        \Gamma^2 R
        =
        \frac{9E}{8\pi K c^2}.

    Parameters
    ----------
    mu : float, optional
        Mean molecular weight. Default is ``0.61``.
    gamma_hat : float, optional
        Adiabatic index. Default is :math:`4/3`.
    lorentz_warn_threshold : float, optional
        Lorentz factor threshold for the ultra-relativistic warning. Default ``2.0``.

    See Also
    --------
    BlandfordMcKeeShockEngine : General power-law external medium.
    """

    # =========================================== #
    # Public shock-property interface             #
    # =========================================== #

    def compute_shock_properties(
        self,
        time: "_UnitBearingArrayLike",
        E: "_UnitBearingScalarLike" = 1e51 * u.erg,
        M_dot: "_UnitBearingScalarLike" = 1e-5 * u.Msun / u.yr,
        v_wind: "_UnitBearingScalarLike" = 1000.0 * u.km / u.s,
    ) -> BlandfordMcKeeShockState:
        r"""
        Compute the BM shock state for a stellar-wind external medium.

        Parameters
        ----------
        time : ~astropy.units.Quantity or array-like
            Time(s) since the explosion. Bare float assumed in seconds.
        E : ~astropy.units.Quantity or float, optional
            Isotropic-equivalent explosion energy. Bare float assumed in erg.
            Default is :math:`10^{51}` erg.
        M_dot : ~astropy.units.Quantity or float, optional
            Stellar wind mass-loss rate. Bare float assumed in
            :math:`\mathrm{g\,s^{-1}}`. Default is
            :math:`10^{-5}\,M_\odot\,\mathrm{yr^{-1}}`.
        v_wind : ~astropy.units.Quantity or float, optional
            Stellar wind terminal velocity. Bare float assumed in cm/s.
            Default is :math:`1000\,\mathrm{km\,s^{-1}}`.

        Returns
        -------
        BlandfordMcKeeShockState
            Shock kinematics and post-shock thermodynamics with
            :class:`~astropy.units.Quantity` units.

        Warns
        -----
        RuntimeWarning
            If any :math:`\Gamma(t)` falls below ``lorentz_warn_threshold``.
        """
        if isinstance(time, u.Quantity):
            time = time.to(u.s).value
        if isinstance(E, u.Quantity):
            E = E.to(u.erg).value
        if isinstance(M_dot, u.Quantity):
            M_dot = M_dot.to(u.g / u.s).value
        if isinstance(v_wind, u.Quantity):
            v_wind = v_wind.to(u.cm / u.s).value

        cgs = self._compute_shock_properties_cgs(time=time, E=E, M_dot=M_dot, v_wind=v_wind)

        return BlandfordMcKeeShockState(
            radius=cgs.radius * u.cm,
            lorentz_factor=cgs.lorentz_factor,
            beta=cgs.beta,
            velocity=cgs.velocity * (u.cm / u.s),
            fluid_lorentz_factor=cgs.fluid_lorentz_factor,
            post_shock_pressure=cgs.post_shock_pressure * (u.dyne / u.cm**2),
            post_shock_temperature=cgs.post_shock_temperature * u.K,
            post_shock_comoving_density=cgs.post_shock_comoving_density * (u.g / u.cm**3),
            post_shock_lab_density=cgs.post_shock_lab_density * (u.g / u.cm**3),
            thermal_energy_density_comoving=cgs.thermal_energy_density_comoving * (u.erg / u.cm**3),
            thermal_energy_density_lab=cgs.thermal_energy_density_lab * (u.erg / u.cm**3),
        )

    # =========================================== #
    # CGS backend                                 #
    # =========================================== #

    def _compute_shock_properties_cgs(
        self,
        time: "_ArrayLike",
        E: float = 1e51,
        M_dot: float = 6.3e20,
        v_wind: float = 1e8,
    ) -> BlandfordMcKeeShockState:
        """CGS backend: convert wind parameters to K_csm then delegate to the parent solver."""
        K_csm = M_dot / (4.0 * np.pi * v_wind)
        return super()._compute_shock_properties_cgs(time=time, E=E, K_csm=K_csm, k=2.0)
