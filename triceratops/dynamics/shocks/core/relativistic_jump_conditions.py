"""
Relativistic Rankine–Hugoniot jump conditions for high-velocity astrophysical shocks.

This module provides solvers for relativistic shock jump conditions, covering the
full range from mildly relativistic to ultra-relativistic flows. All solvers accept
lab-frame velocities and proper comoving-frame thermodynamic quantities as input.

Classes
-------
RelativisticJumpConditions
    Abstract base class for all relativistic jump-condition solvers.
UltraRelativisticShockConditions
    Analytic solver for the ultra-relativistic strong-shock limit (Γ₁ ≫ 1).
UltraRelativisticColdShockConditions
    Analytic solver for the UR strong-shock limit into a cold upstream medium.
RelativisticShockConditions
    General numerical solver for arbitrary upstream states.
RelativisticColdShockConditions
    Numerical solver for cold upstream media (P₁ ≈ 0).

References
----------
See :ref:`relativistic_jump_conditions_theory` for the full theoretical derivation.
"""

from abc import ABC, abstractmethod
from collections import namedtuple
from typing import TYPE_CHECKING

import astropy.constants as const
import astropy.units as u
import numpy as np
from scipy.optimize import brentq

from triceratops.utils.misc_utils import ensure_in_units

if TYPE_CHECKING:
    from triceratops._typing import _UnitBearingArrayLike

# --- CGS constants for low-level API --- #
c_cgs = const.c.cgs.value  # Speed of light (cm s⁻¹)
k_B_cgs = const.k_B.cgs.value  # Boltzmann constant (erg K⁻¹)
m_p_cgs = const.m_p.cgs.value  # Proton mass (g)


# ================================================================== #
# Private Helper Functions                                             #
# ================================================================== #
def _lorentz_factor(beta):
    """Lorentz factor Γ = 1/√(1−β²)."""
    return 1.0 / np.sqrt(1.0 - beta**2)


def _shock_frame_beta(beta_shock, beta_upstream):
    """Upstream fluid velocity in the shock frame via relativistic velocity addition.

    Parameters
    ----------
    beta_shock : float
        Shock velocity in the lab frame, β_sh = v_sh / c.
    beta_upstream : float
        Upstream fluid bulk velocity in the lab frame, β_up = v_up / c.

    Returns
    -------
    float
        Upstream fluid velocity in the shock rest frame, β₁.
    """
    return (beta_shock - beta_upstream) / (1.0 - beta_shock * beta_upstream)


def _maxwell_juttner_temperature_cgs(U_int, rho, mu):
    r"""Post-shock temperature from the Maxwell–Jüttner internal energy relation.

    Inverts the Gammie–McKinney approximation :footcite:p:`1998ApJ...498..313G`

    .. math::

        U_{\rm int} \approx \rho c^2 \Theta \frac{6 + 15\Theta}{4 + 5\Theta},

    where :math:`\Theta = k_B T / (\mu m_p c^2)`, to obtain

    .. math::

        \Theta = \frac{5\xi - 6 + \sqrt{(6 - 5\xi)^2 + 240\xi}}{30},

    with :math:`\xi = U_{\rm int} / (\rho c^2)`.

    Notes
    -----
    The coefficient in the discriminant is 240, not 120 as printed in the
    ``relativistic_jump_conditions_theory`` reference page (typographical error there).
    The value 240 = 4 · 15 · 4 follows directly from the quadratic formula applied to
    ``15Θ² + (6 − 5ξ)Θ − 4ξ = 0`` and is confirmed by substituting Θ = 1
    (ξ = 7/3) and recovering the identity.

    Parameters
    ----------
    U_int : float
        Proper internal energy density (erg cm⁻³).
    rho : float
        Proper rest-mass density (g cm⁻³).
    mu : float
        Mean molecular weight.

    Returns
    -------
    float
        Post-shock temperature (K).
    """
    xi = U_int / (rho * c_cgs**2)
    discriminant = (6.0 - 5.0 * xi) ** 2 + 240.0 * xi
    theta = (5.0 * xi - 6.0 + np.sqrt(discriminant)) / 30.0
    return theta * mu * m_p_cgs * c_cgs**2 / k_B_cgs


def _relativistic_implicit_eq(beta2, beta1, gamma1, omega1, pi1, gamma_hat):
    r"""Residual of the relativistic jump condition.

    Evaluates the left-hand side of :eq:`eq:relativistic_jump_condition_final`
    from :ref:`relativistic_jump_conditions_theory`:

    .. math::

        \left(\frac{\hat{\gamma}-1}{\hat{\gamma}}\right)
        \frac{\Gamma_1\beta_1}{\Gamma_2\beta_2}
        \left[\Omega_1\frac{\Gamma_1}{\Gamma_2} - 1\right]
        - \Omega_1\Gamma_1^2\beta_1^2\left(1 - \frac{\beta_2}{\beta_1}\right)
        - \Pi_1 = 0.

    The physical root lies in the open interval (0, β₁).

    Parameters
    ----------
    beta2 : float
        Trial downstream shock-frame velocity β₂.
    beta1, gamma1 : float
        Upstream shock-frame velocity β₁ and Lorentz factor Γ₁.
    omega1, pi1 : float
        Dimensionless upstream enthalpy Ω₁ = w₁/(ρ₁c²) and pressure Π₁ = P₁/(ρ₁c²).
    gamma_hat : float
        Adiabatic index γ̂.

    Returns
    -------
    float
        Residual; zero at the physical downstream velocity.
    """
    gamma2 = _lorentz_factor(beta2)
    term1 = ((gamma_hat - 1.0) / gamma_hat) * (gamma1 * beta1) / (gamma2 * beta2) * (omega1 * gamma1 / gamma2 - 1.0)
    term2 = omega1 * gamma1**2 * beta1**2 * (1.0 - beta2 / beta1)
    return term1 - term2 - pi1


def _reconstruct_state_cgs(beta1, beta2, rho1, omega1, gamma_hat, mu):
    r"""Reconstruct the full post-shock state from β₁, β₂, and the upstream state.

    Given the downstream shock-frame velocity β₂ (from root-finding or analytic
    formula), applies the continuity and energy-momentum conditions to recover all
    post-shock thermodynamic quantities.

    Parameters
    ----------
    beta1 : float
        Upstream fluid velocity in the shock frame, β₁.
    beta2 : float
        Downstream fluid velocity in the shock frame, β₂.
    rho1 : float
        Upstream proper rest-mass density (g cm⁻³).
    omega1 : float
        Upstream dimensionless enthalpy Ω₁ = w₁ / (ρ₁c²).
    gamma_hat : float
        Adiabatic index γ̂.
    mu : float
        Mean molecular weight.

    Returns
    -------
    dict
        Post-shock quantities in CGS, keyed by :attr:`RelativisticJumpConditions.OUTPUT_FIELDS` names.
    """
    gamma1 = _lorentz_factor(beta1)
    gamma2 = _lorentz_factor(beta2)

    # Compression ratio (proper rest-mass density ratio)
    R = (gamma1 * beta1) / (gamma2 * beta2)
    rho2 = R * rho1

    # Enthalpy density reconstruction (energy-flux continuity)
    w1 = rho1 * c_cgs**2 * omega1
    w2 = w1 * gamma1**2 * beta1 / (gamma2**2 * beta2)

    # EOS: P₂ = [(γ̂−1)/γ̂](w₂ − ρ₂c²)
    P2 = ((gamma_hat - 1.0) / gamma_hat) * (w2 - rho2 * c_cgs**2)

    # Proper internal and total energy densities
    U_int2 = P2 / (gamma_hat - 1.0)
    e2 = rho2 * c_cgs**2 + U_int2

    # Proper number density and temperature
    n2 = rho2 / (mu * m_p_cgs)
    T2 = _maxwell_juttner_temperature_cgs(U_int2, rho2, mu)

    return {
        "compression_ratio": R,
        "post_shock_beta": beta2,
        "post_shock_lorentz_factor": gamma2,
        "post_shock_density": rho2,
        "post_shock_number_density": n2,
        "post_shock_pressure": P2,
        "post_shock_temperature": T2,
        "post_shock_energy_density": e2,
    }


def _solve_strong_cold_shock_beta(beta1, gamma_hat):
    beta1 = float(beta1)
    gamma_hat = float(gamma_hat)

    if not (0.0 < beta1 < 1.0):
        raise ValueError("`beta1` must satisfy 0 < beta1 < 1.")
    if not (1.0 < gamma_hat <= 2.0):
        raise ValueError("`gamma_hat` should satisfy 1 < gamma_hat <= 2.")

    # Weak-shock / NR limit. Avoid the singular residual near beta2 = 0.
    if beta1 < 1e-6:
        chi = (gamma_hat + 1.0) / (gamma_hat - 1.0)
        return beta1 / chi

    gamma1 = _lorentz_factor(beta1)

    def residual(beta2):
        return _relativistic_implicit_eq(
            beta2=beta2,
            beta1=beta1,
            gamma1=gamma1,
            omega1=1.0,
            pi1=0.0,
            gamma_hat=gamma_hat,
        )

    # The physical downstream speed satisfies 0 < beta2 < beta1.
    # Use the NR strong-shock estimate as a safer lower-scale guide.
    chi = (gamma_hat + 1.0) / (gamma_hat - 1.0)
    beta2_guess = beta1 / chi

    lo = max(np.nextafter(0.0, 1.0), 0.01 * beta2_guess)
    hi = np.nextafter(beta1, 0.0)

    # The general residual is not always nicely signed at lo, so scan inside
    # the physical interval and find a sign-changing sub-bracket.
    grid = np.linspace(lo, hi, 64)
    vals = np.array([residual(x) for x in grid])

    idx = np.where(np.signbit(vals[:-1]) != np.signbit(vals[1:]))[0]
    if idx.size == 0:
        raise ValueError(
            "Could not bracket physical shock root for "
            f"beta1={beta1}, gamma_hat={gamma_hat}. "
            f"residual min/max = {np.nanmin(vals)}, {np.nanmax(vals)}"
        )

    i = idx[0]
    return brentq(
        residual,
        grid[i],
        grid[i + 1],
        xtol=1e-14,
        rtol=1e-14,
        maxiter=100,
    )


# ================================================================== #
# Abstract Base Class                                                #
# ================================================================== #


class RelativisticJumpConditions(ABC):
    """Abstract base class for relativistic Rankine–Hugoniot jump-condition solvers.

    Mirrors the interface of
    :class:`~triceratops.dynamics.shocks.core.rankine_hugoniot.JumpConditions`
    but defines a separate hierarchy for relativistic flows.  All subclasses
    accept lab-frame velocities and return results as proper comoving-frame
    thermodynamic quantities attached to :class:`~astropy.units.Quantity` objects.

    All output fields are defined by :attr:`OUTPUT_FIELDS` and shared across every
    concrete subclass.  :meth:`__init_subclass__` auto-generates a named-tuple type
    ``{ClassName}Result`` for each concrete subclass.

    Subclasses must implement :meth:`_solve` (CGS backend) and :meth:`solve`
    (public unit-aware wrapper).
    """

    OUTPUT_FIELDS: tuple = (
        ("compression_ratio", None),
        ("post_shock_beta", None),
        ("post_shock_lorentz_factor", None),
        ("post_shock_density", u.g / u.cm**3),
        ("post_shock_number_density", u.cm**-3),
        ("post_shock_pressure", u.dyn / u.cm**2),
        ("post_shock_temperature", u.K),
        ("post_shock_energy_density", u.erg / u.cm**3),
    )
    """tuple: Post-shock output fields as ``(name, unit)`` pairs.

    ``None`` units are returned as bare floats (dimensionless quantities).
    The field order determines the named-tuple returned by :meth:`solve`.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.OUTPUT_FIELDS:
            cls._result_type = namedtuple(
                f"{cls.__name__}Result",
                [name for name, _ in cls.OUTPUT_FIELDS],
            )

    @classmethod
    def _make_result(cls, result_dict: dict):
        """Attach :attr:`OUTPUT_FIELDS` units to *result_dict* and return the named tuple."""
        # noinspection PyArgumentList
        return cls._result_type(
            **{
                name: (result_dict[name] * unit if unit is not None else result_dict[name])
                for name, unit in cls.OUTPUT_FIELDS
            }
        )

    @classmethod
    @abstractmethod
    def _solve(cls, *args, **kwargs) -> dict:
        """CGS backend.  Returns a plain ``dict`` keyed by :attr:`OUTPUT_FIELDS` names."""
        ...

    @classmethod
    @abstractmethod
    def solve(cls, *args, **kwargs):
        """Compute jump-condition quantities and return them as a named tuple.

        Subclasses define the explicit input signature.  Physical quantities are
        returned as :class:`~astropy.units.Quantity` objects; dimensionless fields
        (β, Γ, compression ratio) are bare floats.
        """
        ...


# ================================================================== #
# General Numerical Solver                                             #
# ================================================================== #


class RelativisticShockConditions(RelativisticJumpConditions):
    """General relativistic jump conditions solved numerically.

    Finds the downstream shock-frame velocity β₂ by bracketed root-finding
    (:footcite:t:`brent2013algorithms`) applied to the implicit condition
    (Eq. :eq:`eq:relativistic_jump_condition_final` of
    :ref:`relativistic_jump_conditions_theory`), then reconstructs the full
    post-shock state analytically.

    Lab-frame shock and upstream velocities are the primary velocity inputs;
    thermodynamic inputs (ρ, P, T) are proper comoving-frame quantities.
    Either *upstream_pressure* or *upstream_temperature* is required to
    determine the upstream dimensionless enthalpy Ω₁.

    See Also
    --------
    RelativisticColdShockConditions : Specialisation for P₁ ≈ 0.
    UltraRelativisticShockConditions : Analytic solver for Γ₁ ≫ 1.
    """

    @classmethod
    def _solve(
        cls,
        shock_velocity,
        upstream_density,
        upstream_velocity=0.0,
        upstream_pressure=None,
        upstream_temperature=None,
        mu=0.61,
        gamma_hat=4.0 / 3.0,
    ) -> dict:
        """CGS backend — all inputs are plain floats in CGS.

        Parameters
        ----------
        shock_velocity : float
            Shock speed in the lab frame (cm s⁻¹).
        upstream_density : float
            Proper rest-mass density of the upstream medium (g cm⁻³).
        upstream_velocity : float, optional
            Lab-frame bulk velocity of the upstream fluid (cm s⁻¹). Default 0.
        upstream_pressure : float or None, optional
            Proper upstream pressure (dyn cm⁻²).
        upstream_temperature : float or None, optional
            Proper upstream temperature (K). Used only when *upstream_pressure* is ``None``.
        mu : float, optional
            Mean molecular weight. Default 0.61.
        gamma_hat : float, optional
            Adiabatic index γ̂. Default 4/3.

        Returns
        -------
        dict
            Post-shock quantities in CGS, keyed by :attr:`OUTPUT_FIELDS` names.

        Raises
        ------
        ValueError
            If neither *upstream_pressure* nor *upstream_temperature* is supplied.
        """
        if upstream_pressure is None and upstream_temperature is None:
            raise ValueError(
                "RelativisticShockConditions requires upstream_pressure or "
                "upstream_temperature to determine the upstream enthalpy Ω₁. "
                "For a cold upstream (P₁ = 0), use RelativisticColdShockConditions."
            )

        # Resolve upstream pressure from temperature if needed
        if upstream_pressure is None:
            upstream_pressure = upstream_density * k_B_cgs * upstream_temperature / (mu * m_p_cgs)

        # Lab-frame β values
        beta_sh = shock_velocity / c_cgs
        beta_up = upstream_velocity / c_cgs

        # Upstream velocity in the shock rest frame
        beta1 = _shock_frame_beta(beta_sh, beta_up)
        gamma1 = _lorentz_factor(beta1)

        # Dimensionless upstream state
        pi1 = upstream_pressure / (upstream_density * c_cgs**2)
        omega1 = 1.0 + (gamma_hat / (gamma_hat - 1.0)) * pi1

        # Bracketed root-find for the physical downstream velocity
        _EPS = 1e-10
        beta2 = brentq(
            _relativistic_implicit_eq,
            _EPS,
            beta1 - _EPS,
            args=(beta1, gamma1, omega1, pi1, gamma_hat),
        )

        return _reconstruct_state_cgs(beta1, beta2, upstream_density, omega1, gamma_hat, mu)

    @classmethod
    def solve(
        cls,
        shock_velocity: "_UnitBearingArrayLike",
        upstream_density: "_UnitBearingArrayLike",
        upstream_velocity: "_UnitBearingArrayLike" = 0.0 * u.cm / u.s,
        upstream_pressure=None,
        upstream_temperature=None,
        mu: float = 0.61,
        gamma_hat: float = 4.0 / 3.0,
    ):
        """Solve the relativistic jump conditions for a general upstream state.

        Parameters
        ----------
        shock_velocity : ~astropy.units.Quantity or float
            Shock speed in the lab frame. Bare floats are interpreted as cm s⁻¹.
        upstream_density : ~astropy.units.Quantity or float
            Proper rest-mass density of the upstream medium. Bare floats are
            interpreted as g cm⁻³.
        upstream_velocity : ~astropy.units.Quantity or float, optional
            Lab-frame bulk velocity of the upstream fluid. Bare floats are
            interpreted as cm s⁻¹. Default 0.
        upstream_pressure : ~astropy.units.Quantity or float or None, optional
            Proper upstream pressure. Bare floats are interpreted as dyn cm⁻².
            Takes precedence over *upstream_temperature* if both are given.
        upstream_temperature : ~astropy.units.Quantity or float or None, optional
            Proper upstream temperature (K). Used only when *upstream_pressure*
            is ``None``.
        mu : float, optional
            Mean molecular weight. Default 0.61.
        gamma_hat : float, optional
            Adiabatic index γ̂. Default 4/3.

        Returns
        -------
        RelativisticShockConditionsResult
            Named tuple with post-shock fields in physical units as defined by
            :attr:`~RelativisticJumpConditions.OUTPUT_FIELDS`.

        Raises
        ------
        ValueError
            If neither *upstream_pressure* nor *upstream_temperature* is supplied.
        """
        v_sh = ensure_in_units(shock_velocity, u.cm / u.s)
        rho1 = ensure_in_units(upstream_density, u.g / u.cm**3)
        v_up = ensure_in_units(upstream_velocity, u.cm / u.s)
        P1 = ensure_in_units(upstream_pressure, u.dyn / u.cm**2) if upstream_pressure is not None else None
        T1 = ensure_in_units(upstream_temperature, u.K) if upstream_temperature is not None else None
        return cls._make_result(
            cls._solve(v_sh, rho1, v_up, upstream_pressure=P1, upstream_temperature=T1, mu=mu, gamma_hat=gamma_hat)
        )


# ================================================================== #
# Numerical Solver — Cold Upstream                                     #
# ================================================================== #


class RelativisticColdShockConditions(RelativisticShockConditions):
    """Relativistic jump conditions for a cold upstream medium (P₁ ≈ 0).

    A specialisation of :class:`RelativisticShockConditions` that fixes Π₁ = 0
    and Ω₁ = 1.  Upstream pressure and temperature are not required as inputs.
    This is the appropriate choice for most astrophysical blast-wave problems
    (supernovae, TDE jets) where the ambient medium is cold relative to the shock.

    See Also
    --------
    RelativisticShockConditions : General hot-upstream solver.
    UltraRelativisticColdShockConditions : Analytic cold-upstream solver for Γ₁ ≫ 1.
    """

    @classmethod
    def _solve(
        cls,
        shock_velocity,
        upstream_density,
        upstream_velocity=0.0,
        mu=0.61,
        gamma_hat=4.0 / 3.0,
    ) -> dict:
        return super()._solve(
            shock_velocity,
            upstream_density,
            upstream_velocity=upstream_velocity,
            upstream_pressure=0.0,
            mu=mu,
            gamma_hat=gamma_hat,
        )

    @classmethod
    def solve(
        cls,
        shock_velocity: "_UnitBearingArrayLike",
        upstream_density: "_UnitBearingArrayLike",
        upstream_velocity: "_UnitBearingArrayLike" = 0.0 * u.cm / u.s,
        mu: float = 0.61,
        gamma_hat: float = 4.0 / 3.0,
    ):
        """Solve the relativistic jump conditions for a cold upstream medium.

        Parameters
        ----------
        shock_velocity : ~astropy.units.Quantity or float
            Shock speed in the lab frame. Bare floats are interpreted as cm s⁻¹.
        upstream_density : ~astropy.units.Quantity or float
            Proper rest-mass density of the upstream medium. Bare floats are
            interpreted as g cm⁻³.
        upstream_velocity : ~astropy.units.Quantity or float, optional
            Lab-frame bulk velocity of the upstream fluid. Bare floats are
            interpreted as cm s⁻¹. Default 0.
        mu : float, optional
            Mean molecular weight. Default 0.61.
        gamma_hat : float, optional
            Adiabatic index γ̂. Default 4/3.

        Returns
        -------
        RelativisticColdShockConditionsResult
            Named tuple with post-shock fields in physical units.
        """
        v_sh = ensure_in_units(shock_velocity, u.cm / u.s)
        rho1 = ensure_in_units(upstream_density, u.g / u.cm**3)
        v_up = ensure_in_units(upstream_velocity, u.cm / u.s)
        return cls._make_result(cls._solve(v_sh, rho1, v_up, mu=mu, gamma_hat=gamma_hat))


# ================================================================== #
# Analytic Ultra-Relativistic Solver                                   #
# ================================================================== #


class UltraRelativisticShockConditions(RelativisticJumpConditions):
    r"""Analytic jump-condition solver for the ultra-relativistic strong-shock limit.

    In the limit :math:`\Gamma_1 \gg 1` the downstream shock-frame velocity
    satisfies

    .. math::

        \beta_2 = \hat{\gamma} - 1,

    independently of the upstream pressure (Π₁ drops out at leading order in
    :math:`\Gamma_1`).  For :math:`\hat{\gamma} = 4/3` this gives the classical
    GRB result :math:`\beta_2 = 1/3` :footcite:p:`2007nras.book.....P`.

    The full post-shock state is then reconstructed analytically from the upstream
    state.  Despite the analytic β₂, the shock velocity is still required because
    the compression ratio and downstream thermodynamic quantities depend on Γ₁.

    Notes
    -----
    Although Π₁ does not enter β₂ in the UR limit, the upstream enthalpy Ω₁ still
    affects :math:`w_2` and therefore :math:`P_2`, :math:`U_{\rm int,2}`, and
    :math:`T_2`.  Pass *upstream_pressure* or *upstream_temperature* to account for
    a warm upstream medium; leave both ``None`` to assume P₁ = 0 (equivalent to
    :class:`UltraRelativisticColdShockConditions`).

    See Also
    --------
    UltraRelativisticColdShockConditions : Simplified cold-upstream variant.
    RelativisticShockConditions : General numerical solver.
    """

    @classmethod
    def _ur_beta2(cls, gamma_hat: float) -> float:
        r"""Analytic downstream velocity in the UR strong-shock limit.

        Returns :math:`\beta_2 = \hat{\gamma} - 1`, derived by solving
        :math:`\beta_2(1-\beta_2) = [(\hat{\gamma}-1)/\hat{\gamma}](1-\beta_2^2)`.
        """
        return gamma_hat - 1.0

    @classmethod
    def _solve(
        cls,
        shock_velocity,
        upstream_density,
        upstream_velocity=0.0,
        upstream_pressure=None,
        upstream_temperature=None,
        mu=0.61,
        gamma_hat=4.0 / 3.0,
    ) -> dict:
        """CGS backend — all inputs are plain floats in CGS.

        Parameters
        ----------
        shock_velocity : float
            Shock speed in the lab frame (cm s⁻¹).
        upstream_density : float
            Proper rest-mass density of the upstream medium (g cm⁻³).
        upstream_velocity : float, optional
            Lab-frame bulk velocity of the upstream fluid (cm s⁻¹). Default 0.
        upstream_pressure : float or None, optional
            Proper upstream pressure (dyn cm⁻²). Affects state reconstruction
            via Ω₁ but does not alter β₂. Default None (→ P₁ = 0).
        upstream_temperature : float or None, optional
            Proper upstream temperature (K). Used to compute P₁ when
            *upstream_pressure* is ``None``. Default None.
        mu : float, optional
            Mean molecular weight. Default 0.61.
        gamma_hat : float, optional
            Adiabatic index γ̂. Default 4/3.

        Returns
        -------
        dict
            Post-shock quantities in CGS.
        """
        # Resolve upstream pressure; default to cold (P₁ = 0) if neither supplied
        if upstream_pressure is None and upstream_temperature is not None:
            upstream_pressure = upstream_density * k_B_cgs * upstream_temperature / (mu * m_p_cgs)
        P1 = upstream_pressure if upstream_pressure is not None else 0.0

        beta_sh = shock_velocity / c_cgs
        beta_up = upstream_velocity / c_cgs
        beta1 = _shock_frame_beta(beta_sh, beta_up)

        pi1 = P1 / (upstream_density * c_cgs**2)
        omega1 = 1.0 + (gamma_hat / (gamma_hat - 1.0)) * pi1

        # Analytic downstream velocity (UR limit)
        beta2 = cls._ur_beta2(gamma_hat)

        return _reconstruct_state_cgs(beta1, beta2, upstream_density, omega1, gamma_hat, mu)

    @classmethod
    def solve(
        cls,
        shock_velocity: "_UnitBearingArrayLike",
        upstream_density: "_UnitBearingArrayLike",
        upstream_velocity: "_UnitBearingArrayLike" = 0.0 * u.cm / u.s,
        upstream_pressure=None,
        upstream_temperature=None,
        mu: float = 0.61,
        gamma_hat: float = 4.0 / 3.0,
    ):
        """Solve the ultra-relativistic jump conditions analytically.

        Parameters
        ----------
        shock_velocity : ~astropy.units.Quantity or float
            Shock speed in the lab frame. Bare floats are interpreted as cm s⁻¹.
        upstream_density : ~astropy.units.Quantity or float
            Proper rest-mass density of the upstream medium. Bare floats are
            interpreted as g cm⁻³.
        upstream_velocity : ~astropy.units.Quantity or float, optional
            Lab-frame bulk velocity of the upstream fluid. Bare floats are
            interpreted as cm s⁻¹. Default 0.
        upstream_pressure : ~astropy.units.Quantity or float or None, optional
            Proper upstream pressure. Bare floats are interpreted as dyn cm⁻².
            Affects state reconstruction but not β₂. Default None (cold upstream).
        upstream_temperature : ~astropy.units.Quantity or float or None, optional
            Proper upstream temperature (K). Used only when *upstream_pressure*
            is ``None``. Default None.
        mu : float, optional
            Mean molecular weight. Default 0.61.
        gamma_hat : float, optional
            Adiabatic index γ̂. Default 4/3.

        Returns
        -------
        UltraRelativisticShockConditionsResult
            Named tuple with post-shock fields in physical units.
        """
        v_sh = ensure_in_units(shock_velocity, u.cm / u.s)
        rho1 = ensure_in_units(upstream_density, u.g / u.cm**3)
        v_up = ensure_in_units(upstream_velocity, u.cm / u.s)
        P1 = ensure_in_units(upstream_pressure, u.dyn / u.cm**2) if upstream_pressure is not None else None
        T1 = ensure_in_units(upstream_temperature, u.K) if upstream_temperature is not None else None
        return cls._make_result(
            cls._solve(v_sh, rho1, v_up, upstream_pressure=P1, upstream_temperature=T1, mu=mu, gamma_hat=gamma_hat)
        )


# ================================================================== #
# Analytic Ultra-Relativistic Solver — Cold Upstream                   #
# ================================================================== #


class UltraRelativisticColdShockConditions(UltraRelativisticShockConditions):
    r"""Analytic jump conditions for the UR strong-shock limit into a cold medium.

    A specialisation of :class:`UltraRelativisticShockConditions` that fixes
    P₁ = 0 (Π₁ = 0, Ω₁ = 1).  Upstream pressure and temperature are not required
    as inputs.  This is the canonical limit used in GRB afterglow modeling, where
    :math:`\hat{\gamma} = 4/3` gives the classic result :math:`\beta_2 = 1/3`.

    See Also
    --------
    UltraRelativisticShockConditions : Generalisation that accepts a warm upstream.
    RelativisticColdShockConditions : Numerical cold-upstream solver (arbitrary Γ₁).
    """

    @classmethod
    def _solve(
        cls,
        shock_velocity,
        upstream_density,
        upstream_velocity=0.0,
        mu=0.61,
        gamma_hat=4.0 / 3.0,
    ) -> dict:
        return super()._solve(
            shock_velocity,
            upstream_density,
            upstream_velocity=upstream_velocity,
            upstream_pressure=0.0,
            mu=mu,
            gamma_hat=gamma_hat,
        )

    @classmethod
    def solve(
        cls,
        shock_velocity: "_UnitBearingArrayLike",
        upstream_density: "_UnitBearingArrayLike",
        upstream_velocity: "_UnitBearingArrayLike" = 0.0 * u.cm / u.s,
        mu: float = 0.61,
        gamma_hat: float = 4.0 / 3.0,
    ):
        """Solve the cold-upstream ultra-relativistic jump conditions analytically.

        Parameters
        ----------
        shock_velocity : ~astropy.units.Quantity or float
            Shock speed in the lab frame. Bare floats are interpreted as cm s⁻¹.
        upstream_density : ~astropy.units.Quantity or float
            Proper rest-mass density of the upstream medium. Bare floats are
            interpreted as g cm⁻³.
        upstream_velocity : ~astropy.units.Quantity or float, optional
            Lab-frame bulk velocity of the upstream fluid. Bare floats are
            interpreted as cm s⁻¹. Default 0.
        mu : float, optional
            Mean molecular weight. Default 0.61.
        gamma_hat : float, optional
            Adiabatic index γ̂. Default 4/3.

        Returns
        -------
        UltraRelativisticColdShockConditionsResult
            Named tuple with post-shock fields in physical units.
        """
        v_sh = ensure_in_units(shock_velocity, u.cm / u.s)
        rho1 = ensure_in_units(upstream_density, u.g / u.cm**3)
        v_up = ensure_in_units(upstream_velocity, u.cm / u.s)
        return cls._make_result(cls._solve(v_sh, rho1, v_up, mu=mu, gamma_hat=gamma_hat))
