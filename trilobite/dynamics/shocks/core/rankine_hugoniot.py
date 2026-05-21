"""
Various utilities for working with the Rankine-Hugoniot conditions in fluid dynamics.

This module contains a few (advanced) RH scenarios which are necessary in various parts of the
dynamics module. The intent here is not to exhaustively cover all possible RH conditions, but rather to
provide a few key implementations that are useful for astrophysical shock dynamics calculations and sufficiently
complex to be worth isolating into their own module.
"""

from abc import ABC, abstractmethod
from collections import namedtuple
from typing import TYPE_CHECKING

import astropy.constants as const
import astropy.units as u
import numpy as np

from trilobite.utils.misc_utils import ensure_in_units

if TYPE_CHECKING:
    from trilobite._typing import _ArrayLike, _UnitBearingArrayLike

# --- Handle Necessary CGS constants for Low-Level API --- #
k_B_cgs = const.k_B.cgs.value  # Boltzmann constant in CGS
m_p_cgs = const.m_p.cgs.value  # Proton mass in CGS


# ========================================= #
# Base Class                                #
# ========================================= #
class JumpConditions(ABC):
    r"""Abstract base class for classical Rankine–Hugoniot jump-condition solvers.

    This class defines the common interface for all non-relativistic shock solvers
    in Trilobite. Subclasses implement specific physical regimes (e.g. strong vs.
    finite-Mach shocks, cold vs. hot upstream media) while sharing a unified API.

    The solver operates in two layers:

    - **Public interface** (:meth:`solve`): accepts either bare floats (interpreted
      as CGS) or :class:`~astropy.units.Quantity` objects and returns results with
      units attached.
    - **Private backend** (:meth:`_solve`): operates purely on CGS floats/arrays
      and returns a plain ``dict`` for performance-critical workflows.

    Output structure is controlled by two class-level attributes:

    - :attr:`OUTPUT_FIELDS`: defines post-shock quantities
    - :attr:`UPSTREAM_OUTPUT_FIELDS`: defines pre-shock quantities returned when
      ``upstream=True``

    For each subclass, a named-tuple type is automatically generated at class
    creation time (e.g. ``StrongShockConditionsResult``), ensuring consistent,
    structured outputs without manual boilerplate.

    Subclasses must implement:

    - :meth:`_solve` — CGS backend returning a ``dict``
    - :meth:`solve` — unit-aware wrapper with explicit input signature
    """

    # -------- Class Configuration ---------- #
    OUTPUT_FIELDS: tuple = ()
    """tuple: Post-shock output fields as ``(name, unit)`` pairs.

    Each entry is a tuple of the form ``(name, unit)``, where *name* is the string
    key for the output quantity and *unit* is an :class:`~astropy.units.Unit` to attach
    to the result. Use ``None`` for dimensionless quantities.
    The order determines the field order of the named tuple returned by :meth:`solve`.
    """

    UPSTREAM_OUTPUT_FIELDS: tuple = ()
    """tuple: Pre-shock output fields as ``(name, unit)`` pairs.

    Same format as :attr:`OUTPUT_FIELDS`. Defines the named tuple returned by
    :meth:`solve` when called with ``upstream=True``.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.OUTPUT_FIELDS:
            cls._result_type = namedtuple(
                f"{cls.__name__}Result",
                [name for name, _ in cls.OUTPUT_FIELDS],
            )
        if cls.UPSTREAM_OUTPUT_FIELDS:
            cls._upstream_result_type = namedtuple(
                f"{cls.__name__}UpstreamResult",
                [name for name, _ in cls.UPSTREAM_OUTPUT_FIELDS],
            )

    # -------- Utility Functions ---------- #
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
    def _make_upstream_result(cls, result_dict: dict):
        """Attach :attr:`UPSTREAM_OUTPUT_FIELDS` units to *result_dict* and return the named tuple."""
        # noinspection PyArgumentList
        return cls._upstream_result_type(
            **{
                name: (result_dict[name] * unit if unit is not None else result_dict[name])
                for name, unit in cls.UPSTREAM_OUTPUT_FIELDS
            }
        )

    # -------- Private API ---------- #
    @classmethod
    @abstractmethod
    def _solve(cls, *args, **kwargs) -> dict:
        """CGS backend. Returns a plain ``dict`` keyed by :attr:`OUTPUT_FIELDS` names."""
        ...

    # -------- Public API ---------- #
    @classmethod
    @abstractmethod
    def solve(cls, *args, **kwargs):
        """Compute all jump-condition quantities and return them as a named tuple.

        Subclasses define the explicit input signature. The return type is the named tuple
        auto-generated from :attr:`OUTPUT_FIELDS`, with physical quantities as
        :class:`~astropy.units.Quantity` and dimensionless fields as bare floats.
        """
        ...


# ========================================= #
# Strong Shocks                             #
# ========================================= #
class StrongShockConditions(JumpConditions):
    r"""Strong-shock Rankine–Hugoniot conditions with a finite upstream thermal state.

    This class implements the classical Rankine–Hugoniot jump conditions in the
    strong-shock limit (:math:`\mathcal{M}_1 \gg 1`), while retaining dependence
    on the upstream thermodynamic state (pressure or temperature).

    Unlike the cold-upstream limit, the downstream pressure, temperature, and
    energy density depend explicitly on the upstream internal energy. As a result,
    either the upstream pressure or upstream temperature must be provided.

    In this regime, the compression ratio becomes constant:

    .. math::

        R \equiv \frac{\rho_2}{\rho_1} = \frac{\gamma + 1}{\gamma - 1},

    but the full momentum and energy jump conditions remain sensitive to the
    upstream state.

    See Also
    --------
    StrongColdShockConditions : Strong shocks with negligible upstream pressure.
    WeakShockConditions : General finite-Mach-number shock solver.
    """

    OUTPUT_FIELDS = (
        ("compression_ratio", None),
        ("post_shock_density", u.g / u.cm**3),
        ("post_shock_number_density", u.cm**-3),
        ("post_shock_velocity", u.cm / u.s),
        ("post_shock_pressure", u.dyn / u.cm**2),
        ("post_shock_temperature", u.K),
        ("post_shock_energy_density", u.erg / u.cm**3),
    )

    UPSTREAM_OUTPUT_FIELDS = (
        ("compression_ratio", None),
        ("pre_shock_density", u.g / u.cm**3),
        ("pre_shock_number_density", u.cm**-3),
        ("pre_shock_velocity", u.cm / u.s),
        ("pre_shock_pressure", u.dyn / u.cm**2),
        ("pre_shock_temperature", u.K),
        ("pre_shock_energy_density", u.erg / u.cm**3),
    )

    # ---------------------------------------------- #
    # Private CGS helpers                            #
    # ---------------------------------------------- #
    @classmethod
    def _compression_ratio(cls, gamma: float = 5 / 3) -> float:
        r"""Compression ratio for a strong shock: :math:`(\gamma+1)/(\gamma-1)`.

        Parameters
        ----------
        gamma : float, optional
            Adiabatic index. Default 5/3 (ideal monatomic gas).

        Returns
        -------
        float
        """
        return (gamma + 1) / (gamma - 1)

    # --- Density --- #
    @classmethod
    def _post_shock_density(cls, upstream_density: "_ArrayLike", gamma: float = 5 / 3):
        r"""
        Compute the downstream (post-shock) mass density for a strong shock.

        In the strong-shock limit (:math:`M_1 \gg 1`), the Rankine–Hugoniot
        relations give a constant compression ratio

        .. math::

            R \equiv \frac{\rho_2}{\rho_1} = \frac{\gamma + 1}{\gamma - 1},

        so that the downstream density is

        .. math::

            \rho_2 = R\,\rho_1.

        Parameters
        ----------
        upstream_density : float or ~numpy.ndarray
            Upstream (pre-shock) mass density :math:`\rho_1` in g/cm\ :sup:`3`.
        gamma : float, optional
            Adiabatic index :math:`\gamma`. Default is ``5/3`` (ideal monatomic gas).

        Returns
        -------
        float or ~numpy.ndarray
            Downstream (post-shock) mass density :math:`\rho_2` in g/cm\ :sup:`3`.

        Notes
        -----
        This result is independent of the Mach number in the strong-shock limit.
        """
        R = cls._compression_ratio(gamma=gamma)
        return R * upstream_density

    @classmethod
    def _post_shock_number_density(cls, upstream_density: "_ArrayLike", gamma: float = 5 / 3, mu: float = 0.61):
        r"""
        Compute the downstream (post-shock) number density for a strong shock.

        The number density is obtained by converting the post-shock mass density
        using a fixed mean molecular weight:

        .. math::

            n_2 = \frac{\rho_2}{\mu m_p},

        where :math:`\mu` is the mean molecular weight and :math:`m_p` is the proton mass.

        Parameters
        ----------
        upstream_density : float or ~numpy.ndarray
            Upstream (pre-shock) mass density :math:`\rho_1` in g/cm\ :sup:`3`.
        gamma : float, optional
            Adiabatic index. Default is ``5/3``.
        mu : float, optional
            Mean molecular weight in units of the proton mass. Default is ``0.61``
            (typical for fully ionized plasma with near-solar composition).

        Returns
        -------
        float or ~numpy.ndarray
            Downstream number density :math:`n_2` in cm\ :sup:`-3`.

        Notes
        -----
        This assumes a single-fluid ideal gas with fixed composition.
        """
        rho2 = cls._post_shock_density(upstream_density, gamma=gamma)
        return rho2 / (mu * m_p_cgs)

    @classmethod
    def _pre_shock_density(cls, downstream_density: "_ArrayLike", gamma: float = 5 / 3):
        r"""
        Compute the upstream (pre-shock) mass density from the downstream density.

        In the strong-shock limit, the density jump is fixed:

        .. math::

            \rho_1 = \frac{\rho_2}{R}, \quad
            R = \frac{\gamma + 1}{\gamma - 1}.

        Parameters
        ----------
        downstream_density : float or ~numpy.ndarray
            Downstream (post-shock) mass density :math:`\rho_2` in g/cm\ :sup:`3`.
        gamma : float, optional
            Adiabatic index. Default is ``5/3``.

        Returns
        -------
        float or ~numpy.ndarray
            Upstream (pre-shock) mass density :math:`\rho_1` in g/cm\ :sup:`3`.

        Notes
        -----
        This is simply the inverse of :meth:`_post_shock_density`.
        """
        R = cls._compression_ratio(gamma=gamma)
        return downstream_density / R

    @classmethod
    def _pre_shock_number_density(cls, downstream_density: "_ArrayLike", gamma: float = 5 / 3, mu: float = 0.61):
        r"""
        Compute the upstream (pre-shock) number density from the downstream density.

        The upstream number density is obtained from the inverted density jump:

        .. math::

            n_1 = \frac{\rho_1}{\mu m_p}, \quad
            \rho_1 = \frac{\rho_2}{R}.

        Parameters
        ----------
        downstream_density : float or ~numpy.ndarray
            Downstream (post-shock) mass density :math:`\rho_2` in g/cm\ :sup:`3`.
        gamma : float, optional
            Adiabatic index. Default is ``5/3``.
        mu : float, optional
            Mean molecular weight in units of the proton mass. Default is ``0.61``.

        Returns
        -------
        float or ~numpy.ndarray
            Upstream number density :math:`n_1` in cm\ :sup:`-3`.

        Notes
        -----
        Assumes constant composition and a single-fluid ideal gas.
        """
        rho1 = cls._pre_shock_density(downstream_density, gamma=gamma)
        return rho1 / (mu * m_p_cgs)

    # --- Velocity --- #
    @classmethod
    def _post_shock_velocity(
        cls, shock_velocity: "_ArrayLike", upstream_velocity: "_ArrayLike" = 0.0, gamma: float = 5 / 3
    ):
        r"""Compute the downstream (post-shock) bulk velocity in the lab frame.

        The velocity is obtained by transforming from the shock frame using
        the Rankine–Hugoniot condition for mass conservation:

        .. math::

            u_2 = \frac{u_1}{R},

        where :math:`u_1 = v_{\rm sh} - v_1` is the upstream velocity in the
        shock frame. Transforming back to the lab frame gives

        .. math::

            v_2 = v_{\rm sh} - \frac{u_1}{R}
                = \left(1 - \frac{1}{R}\right) v_{\rm sh}
                  + \frac{1}{R} v_1.

        Parameters
        ----------
        shock_velocity : float or ~numpy.ndarray
            Shock velocity :math:`v_{\rm sh}` in cm/s (lab frame).
        upstream_velocity : float or ~numpy.ndarray, optional
            Upstream bulk velocity :math:`v_1` in cm/s (lab frame). Default is 0.
        gamma : float, optional
            Adiabatic index. Default is ``5/3``.

        Returns
        -------
        float or ~numpy.ndarray
            Downstream bulk velocity :math:`v_2` in cm/s (lab frame).

        Notes
        -----
        Valid in the strong-shock limit with an ideal gas equation of state.
        """
        shock_velocity = np.asarray(shock_velocity)
        upstream_velocity = np.asarray(upstream_velocity)
        R = cls._compression_ratio(gamma=gamma)
        res = (1 / R) * upstream_velocity + (1 - 1 / R) * shock_velocity
        return res if res.ndim > 0 else res.item()

    @classmethod
    def _pre_shock_velocity(cls, shock_velocity: "_ArrayLike", downstream_velocity: "_ArrayLike", gamma: float = 5 / 3):
        r"""
        Compute the upstream (pre-shock) bulk velocity from downstream conditions.

        This is the inverse of :meth:`_post_shock_velocity`. Solving

        .. math::

            v_2 = \left(1 - \frac{1}{R}\right) v_{\rm sh}
                  + \frac{1}{R} v_1

        for :math:`v_1` gives

        .. math::

            v_1 = R\,v_2 - (R - 1)\,v_{\rm sh}.

        Parameters
        ----------
        shock_velocity : float or ~numpy.ndarray
            Shock velocity :math:`v_{\rm sh}` in cm/s (lab frame).
        downstream_velocity : float or ~numpy.ndarray
            Downstream bulk velocity :math:`v_2` in cm/s (lab frame).
        gamma : float, optional
            Adiabatic index. Default is ``5/3``.

        Returns
        -------
        float or ~numpy.ndarray
            Upstream bulk velocity :math:`v_1` in cm/s (lab frame).

        Notes
        -----
        This transformation assumes a strong shock and ideal gas behavior.
        """
        # Cast to arrays.
        shock_velocity = np.asarray(shock_velocity)
        downstream_velocity = np.asarray(downstream_velocity)

        # Compute the compression ratio.
        R = cls._compression_ratio(gamma=gamma)

        # Determine the result and return.
        res = R * downstream_velocity - (R - 1) * shock_velocity
        return res if res.ndim > 0 else res.item()

    # --- Pressure --- #
    @classmethod
    def _post_shock_pressure(
        cls,
        shock_velocity: "_ArrayLike",
        upstream_density: "_ArrayLike",
        upstream_velocity: "_ArrayLike" = 0.0,
        upstream_pressure: "_ArrayLike" = None,
        upstream_temperature: "_ArrayLike" = None,
        mu: float = 0.61,
        gamma: float = 5 / 3,
    ):
        r"""
        Compute the downstream (post-shock) pressure for a strong shock.

        This method evaluates the Rankine–Hugoniot momentum conservation condition:

        .. math::

            P_2 = P_1 + \rho_1 u_1^2 \left(1 - \frac{1}{R}\right),

        where:

        - :math:`u_1 = v_{\rm sh} - v_1` is the upstream velocity in the shock frame,
        - :math:`R = (\gamma+1)/(\gamma-1)` is the strong-shock compression ratio.

        The upstream pressure :math:`P_1` may be provided directly or derived from
        the upstream temperature using the ideal gas relation:

        .. math::

            P_1 = \frac{\rho_1 k_B T_1}{\mu m_p}.

        Parameters
        ----------
        shock_velocity : float or ~numpy.ndarray
            Shock velocity :math:`v_{\rm sh}` in cm/s (lab frame).
        upstream_density : float or ~numpy.ndarray
            Upstream mass density :math:`\rho_1` in g/cm\ :sup:`3`.
        upstream_velocity : float or ~numpy.ndarray, optional
            Upstream bulk velocity :math:`v_1` in cm/s (lab frame). Default is 0.
        upstream_pressure : float or ~numpy.ndarray, optional
            Upstream pressure :math:`P_1` in dyne/cm\ :sup:`2`.
        upstream_temperature : float or ~numpy.ndarray, optional
            Upstream temperature :math:`T_1` in K. Used to compute pressure if
            ``upstream_pressure`` is not provided.
        mu : float, optional
            Mean molecular weight in units of the proton mass.
        gamma : float, optional
            Adiabatic index.

        Returns
        -------
        float or ~numpy.ndarray
            Downstream pressure :math:`P_2` in dyne/cm\ :sup:`2`.

        Raises
        ------
        ValueError
            If neither or both of ``upstream_pressure`` and ``upstream_temperature`` are provided.

        Notes
        -----
        This expression is valid in the strong-shock limit, where the compression
        ratio is independent of Mach number but the upstream thermodynamic state
        still contributes through :math:`P_1`.
        """
        # Validate thermodynamic inputs
        if (upstream_pressure is None) and (upstream_temperature is None):
            raise ValueError("Must provide either upstream_pressure or upstream_temperature.")
        elif (upstream_pressure is not None) and (upstream_temperature is not None):
            raise ValueError("Provide only one of upstream_pressure or upstream_temperature.")

        # Coerce base quantities
        shock_velocity = np.asarray(shock_velocity)
        upstream_density = np.asarray(upstream_density)
        upstream_velocity = np.asarray(upstream_velocity)

        # Compute upstream pressure if needed
        if upstream_pressure is None:
            upstream_temperature = np.asarray(upstream_temperature)
            upstream_pressure = upstream_density * k_B_cgs * upstream_temperature / (mu * m_p_cgs)
        else:
            upstream_pressure = np.asarray(upstream_pressure)

        # Shock-frame velocity
        u1 = shock_velocity - upstream_velocity

        # Compression ratio
        R = cls._compression_ratio(gamma=gamma)

        # Momentum jump condition
        downstream_pressure = upstream_pressure + upstream_density * u1**2 * (1 - 1 / R)

        return downstream_pressure if downstream_pressure.ndim > 0 else downstream_pressure.item()

    @classmethod
    def _pre_shock_pressure(
        cls,
        shock_velocity: "_ArrayLike",
        downstream_density: "_ArrayLike",
        downstream_velocity: "_ArrayLike",
        downstream_pressure: "_ArrayLike" = None,
        downstream_temperature: "_ArrayLike" = None,
        mu: float = 0.61,
        gamma: float = 5 / 3,
    ):
        r"""
        Recover the upstream (pre-shock) pressure from downstream conditions.

        This method inverts the Rankine–Hugoniot momentum conservation relation:

        .. math::

            P_2 = P_1 + \rho_1 u_1^2 \left(1 - \frac{1}{R}\right),

        solving for :math:`P_1`:

        .. math::

            P_1 = P_2 - \rho_1 u_1^2 \left(1 - \frac{1}{R}\right).

        Here:

        - :math:`u_1 = v_{\rm sh} - v_1` is the upstream velocity in the shock frame,
        - :math:`\rho_1 = \rho_2 / R`,
        - :math:`R = (\gamma+1)/(\gamma-1)`.

        Parameters
        ----------
        shock_velocity : float or ~numpy.ndarray
            Shock velocity :math:`v_{\rm sh}` in cm/s (lab frame).
        downstream_density : float or ~numpy.ndarray
            Downstream mass density :math:`\rho_2` in g/cm\ :sup:`3`.
        downstream_velocity : float or ~numpy.ndarray
            Downstream bulk velocity :math:`v_2` in cm/s (lab frame).
        downstream_pressure : float or ~numpy.ndarray, optional
            Downstream pressure :math:`P_2` in dyne/cm\ :sup:`2`.
        downstream_temperature : float or ~numpy.ndarray, optional
            Downstream temperature :math:`T_2` in K.
        mu : float, optional
            Mean molecular weight.
        gamma : float, optional
            Adiabatic index.

        Returns
        -------
        float or ~numpy.ndarray
            Upstream pressure :math:`P_1` in dyne/cm\ :sup:`2`.

        Raises
        ------
        ValueError
            If neither or both of ``downstream_pressure`` and ``downstream_temperature`` are provided.

        Notes
        -----
        This inversion assumes a strong shock and ideal gas behavior.
        """
        # Validate thermodynamic inputs
        if (downstream_pressure is None) and (downstream_temperature is None):
            raise ValueError("Must provide either downstream_pressure or downstream_temperature.")
        elif (downstream_pressure is not None) and (downstream_temperature is not None):
            raise ValueError("Provide only one of downstream_pressure or downstream_temperature.")

        # Coerce base quantities
        shock_velocity = np.asarray(shock_velocity)
        rho2 = np.asarray(downstream_density)
        v2 = np.asarray(downstream_velocity)

        # Compute downstream pressure if needed
        if downstream_pressure is None:
            downstream_temperature = np.asarray(downstream_temperature)
            P2 = rho2 * k_B_cgs * downstream_temperature / (mu * m_p_cgs)
        else:
            P2 = np.asarray(downstream_pressure)

        # Compression ratio
        R = cls._compression_ratio(gamma=gamma)

        # Upstream density
        rho1 = rho2 / R

        # Recover upstream velocity
        v1 = cls._pre_shock_velocity(shock_velocity, v2, gamma=gamma)

        # Shock-frame upstream velocity
        u1 = shock_velocity - v1

        # Correct inversion (SUBTRACT, not add)
        P1 = P2 - rho1 * u1**2 * (1 - 1 / R)

        return P1 if P1.ndim > 0 else P1.item()

    # --- Temperature --- #
    @classmethod
    def _post_shock_temperature(
        cls,
        shock_velocity: "_ArrayLike",
        upstream_density: "_ArrayLike",
        upstream_velocity: "_ArrayLike" = 0.0,
        upstream_pressure: "_ArrayLike" = None,
        upstream_temperature: "_ArrayLike" = None,
        mu: float = 0.61,
        gamma: float = 5 / 3,
    ):
        r"""
        Compute the downstream (post-shock) temperature for a strong shock.

        The temperature is obtained from the ideal gas relation:

        .. math::

            T_2 = \frac{\mu m_p}{k_B} \frac{P_2}{\rho_2},

        where the downstream pressure :math:`P_2` is computed from the
        Rankine–Hugoniot momentum condition:

        .. math::

            P_2 = P_1 + \rho_1 u_1^2 \left(1 - \frac{1}{R}\right),

        and the downstream density is

        .. math::

            \rho_2 = R \rho_1, \quad R = \frac{\gamma+1}{\gamma-1}.

        Parameters
        ----------
        shock_velocity : float or ~numpy.ndarray
            Shock velocity :math:`v_{\rm sh}` in cm/s (lab frame).
        upstream_density : float or ~numpy.ndarray
            Upstream mass density :math:`\rho_1` in g/cm\ :sup:`3`.
        upstream_velocity : float or ~numpy.ndarray, optional
            Upstream velocity :math:`v_1` in cm/s. Default is 0.
        upstream_pressure : float or ~numpy.ndarray, optional
            Upstream pressure :math:`P_1` in dyne/cm\ :sup:`2`.
        upstream_temperature : float or ~numpy.ndarray, optional
            Upstream temperature :math:`T_1` in K.
        mu : float, optional
            Mean molecular weight.
        gamma : float, optional
            Adiabatic index.

        Returns
        -------
        float or ~numpy.ndarray
            Downstream temperature :math:`T_2` in K.

        Notes
        -----
        Either ``upstream_pressure`` or ``upstream_temperature`` must be provided.
        """
        P2 = np.asarray(
            cls._post_shock_pressure(
                shock_velocity,
                upstream_density,
                upstream_velocity,
                upstream_pressure=upstream_pressure,
                upstream_temperature=upstream_temperature,
                mu=mu,
                gamma=gamma,
            )
        )

        rho2 = np.asarray(cls._post_shock_density(upstream_density, gamma=gamma))

        T2 = (P2 / rho2) * (mu * m_p_cgs / k_B_cgs)

        return T2 if T2.ndim > 0 else T2.item()

    @classmethod
    def _pre_shock_temperature(
        cls,
        shock_velocity: "_ArrayLike",
        downstream_density: "_ArrayLike",
        downstream_velocity: "_ArrayLike",
        downstream_pressure: "_ArrayLike" = None,
        downstream_temperature: "_ArrayLike" = None,
        mu: float = 0.61,
        gamma: float = 5 / 3,
    ):
        r"""
        Recover the upstream (pre-shock) temperature from downstream conditions.

        The temperature is obtained from the ideal gas relation:

        .. math::

            T_1 = \frac{\mu m_p}{k_B} \frac{P_1}{\rho_1},

        where the upstream pressure :math:`P_1` and density :math:`\rho_1`
        are computed using the corresponding inversion methods.

        The downstream thermodynamic state may be specified via either
        pressure or temperature.

        Parameters
        ----------
        shock_velocity : float or ~numpy.ndarray
            Shock velocity :math:`v_{\rm sh}` in cm/s.
        downstream_density : float or ~numpy.ndarray
            Downstream mass density :math:`\rho_2`.
        downstream_velocity : float or ~numpy.ndarray
            Downstream velocity :math:`v_2`.
        downstream_pressure : float or ~numpy.ndarray, optional
            Downstream pressure :math:`P_2`.
        downstream_temperature : float or ~numpy.ndarray, optional
            Downstream temperature :math:`T_2`.
        mu : float, optional
            Mean molecular weight.
        gamma : float, optional
            Adiabatic index.

        Returns
        -------
        float or ~numpy.ndarray
            Upstream temperature :math:`T_1` in K.

        Notes
        -----
        This method delegates all Rankine–Hugoniot inversion logic to the
        pressure and density solvers, ensuring consistency across the API.
        """
        # Coerce base inputs
        shock_velocity = np.asarray(shock_velocity)
        rho2 = np.asarray(downstream_density)
        v2 = np.asarray(downstream_velocity)

        # Delegate RH inversion to pressure solver
        P1 = cls._pre_shock_pressure(
            shock_velocity=shock_velocity,
            downstream_density=rho2,
            downstream_velocity=v2,
            downstream_pressure=downstream_pressure,
            downstream_temperature=downstream_temperature,
            mu=mu,
            gamma=gamma,
        )

        # Delegate density inversion
        rho1 = cls._pre_shock_density(rho2, gamma=gamma)

        # Ideal gas relation
        T1 = (P1 / rho1) * (mu * m_p_cgs / k_B_cgs)

        return T1 if np.ndim(T1) > 0 else T1.item()

    # --- Energy Density --- #
    @classmethod
    def _post_shock_energy_density(
        cls,
        shock_velocity: "_ArrayLike",
        upstream_density: "_ArrayLike",
        upstream_velocity: "_ArrayLike" = 0.0,
        upstream_pressure: "_ArrayLike" = None,
        upstream_temperature: "_ArrayLike" = None,
        mu: float = 0.61,
        gamma: float = 5 / 3,
    ):
        r"""
        Compute the downstream (post-shock) internal energy density.

        For an ideal gas,

        .. math::

            e_2 = \frac{P_2}{\gamma - 1},

        where :math:`P_2` is computed by :meth:`_post_shock_pressure`. The upstream
        thermodynamic state may be supplied using either ``upstream_pressure`` or
        ``upstream_temperature``.

        Parameters
        ----------
        shock_velocity : float or ~numpy.ndarray
            Shock velocity :math:`v_{\rm sh}` in cm/s.
        upstream_density : float or ~numpy.ndarray
            Upstream mass density :math:`\rho_1` in g/cm\ :sup:`3`.
        upstream_velocity : float or ~numpy.ndarray, optional
            Upstream velocity :math:`v_1` in cm/s. Default is 0.
        upstream_pressure : float or ~numpy.ndarray, optional
            Upstream pressure :math:`P_1` in dyne/cm\ :sup:`2`.
        upstream_temperature : float or ~numpy.ndarray, optional
            Upstream temperature :math:`T_1` in K.
        mu : float, optional
            Mean molecular weight.
        gamma : float, optional
            Adiabatic index.

        Returns
        -------
        float or ~numpy.ndarray
            Downstream internal energy density :math:`e_2` in erg/cm\ :sup:`3`.
        """
        P2 = np.asarray(
            cls._post_shock_pressure(
                shock_velocity=shock_velocity,
                upstream_density=upstream_density,
                upstream_velocity=upstream_velocity,
                upstream_pressure=upstream_pressure,
                upstream_temperature=upstream_temperature,
                mu=mu,
                gamma=gamma,
            )
        )

        e2 = P2 / (gamma - 1)
        return e2 if e2.ndim > 0 else e2.item()

    @classmethod
    def _pre_shock_energy_density(
        cls,
        shock_velocity: "_ArrayLike",
        downstream_density: "_ArrayLike",
        downstream_velocity: "_ArrayLike",
        downstream_pressure: "_ArrayLike" = None,
        downstream_temperature: "_ArrayLike" = None,
        mu: float = 0.61,
        gamma: float = 5 / 3,
    ):
        r"""
        Recover the upstream (pre-shock) internal energy density.

        For an ideal gas,

        .. math::

            e_1 = \frac{P_1}{\gamma - 1},

        where :math:`P_1` is computed by :meth:`_pre_shock_pressure`. The downstream
        thermodynamic state may be supplied using either ``downstream_pressure`` or
        ``downstream_temperature``.

        Parameters
        ----------
        shock_velocity : float or ~numpy.ndarray
            Shock velocity :math:`v_{\rm sh}` in cm/s.
        downstream_density : float or ~numpy.ndarray
            Downstream mass density :math:`\rho_2` in g/cm\ :sup:`3`.
        downstream_velocity : float or ~numpy.ndarray
            Downstream velocity :math:`v_2` in cm/s.
        downstream_pressure : float or ~numpy.ndarray, optional
            Downstream pressure :math:`P_2` in dyne/cm\ :sup:`2`.
        downstream_temperature : float or ~numpy.ndarray, optional
            Downstream temperature :math:`T_2` in K.
        mu : float, optional
            Mean molecular weight.
        gamma : float, optional
            Adiabatic index.

        Returns
        -------
        float or ~numpy.ndarray
            Upstream internal energy density :math:`e_1` in erg/cm\ :sup:`3`.
        """
        P1 = np.asarray(
            cls._pre_shock_pressure(
                shock_velocity=shock_velocity,
                downstream_density=downstream_density,
                downstream_velocity=downstream_velocity,
                downstream_pressure=downstream_pressure,
                downstream_temperature=downstream_temperature,
                mu=mu,
                gamma=gamma,
            )
        )

        e1 = P1 / (gamma - 1)
        return e1 if e1.ndim > 0 else e1.item()

    # ---------------------------------------------- #
    # Public API — individual quantities             #
    # ---------------------------------------------- #
    @classmethod
    def compute_compression_ratio(cls, gamma: float = 5 / 3) -> float:
        r"""Compute the compression ratio for a strong shock.

        In the strong-shock limit (:math:`M_1 \gg 1`), the Rankine–Hugoniot
        relations give a constant density jump:

        .. math::

            R \equiv \frac{\rho_2}{\rho_1}
            = \frac{\gamma + 1}{\gamma - 1}.

        Parameters
        ----------
        gamma : float, optional
            Adiabatic index :math:`\gamma`. Default is ``5/3``.

        Returns
        -------
        float
            Compression ratio :math:`R = \rho_2 / \rho_1`.

        Notes
        -----
        This result is independent of shock velocity and upstream conditions
        in the strong-shock limit.
        """
        return cls._compression_ratio(gamma=gamma)

    # --- Density --- #
    @classmethod
    def compute_post_shock_density(cls, upstream_density: "_UnitBearingArrayLike", gamma: float = 5 / 3) -> u.Quantity:
        r"""Compute the downstream (post-shock) mass density.

        For a strong shock, the downstream density is related to the upstream
        density by the compression ratio:

        .. math::

            \rho_2 = R \, \rho_1,
            \quad R = \frac{\gamma + 1}{\gamma - 1}.

        Parameters
        ----------
        upstream_density : float or ~astropy.units.Quantity
            Upstream mass density :math:`\rho_1`. Bare float assumed
            :math:`{\rm g/cm^3}`.
        gamma : float, optional
            Adiabatic index.

        Returns
        -------
        ~astropy.units.Quantity
            Downstream density :math:`\rho_2` in :math:`{\rm g/cm^3}`.

        Notes
        -----
        Independent of Mach number in the strong-shock limit.
        """
        upstream_density = ensure_in_units(upstream_density, u.g / u.cm**3)
        return cls._post_shock_density(upstream_density, gamma=gamma) * u.g / u.cm**3

    @classmethod
    def compute_post_shock_number_density(
        cls, upstream_density: "_UnitBearingArrayLike", gamma: float = 5 / 3, mu: float = 0.61
    ) -> u.Quantity:
        r"""Compute the downstream (post-shock) number density.

        The number density is obtained from the post-shock mass density via

        .. math::

            n_2 = \frac{\rho_2}{\mu m_p}.

        Parameters
        ----------
        upstream_density : float or ~astropy.units.Quantity
            Upstream mass density :math:`\rho_1`. Bare float assumed
            :math:`{\rm g/cm^3}`.
        gamma : float, optional
            Adiabatic index.
        mu : float, optional
            Mean molecular weight in units of the proton mass.

        Returns
        -------
        ~astropy.units.Quantity
            Downstream number density :math:`n_2` in :math:`{\rm cm^{-3}}`.

        Notes
        -----
        Assumes constant composition and a single-fluid ideal gas.
        """
        rho1 = ensure_in_units(upstream_density, u.g / u.cm**3)
        return cls._post_shock_number_density(rho1, gamma=gamma, mu=mu) * u.cm**-3

    @classmethod
    def compute_pre_shock_density(cls, downstream_density: "_UnitBearingArrayLike", gamma: float = 5 / 3) -> u.Quantity:
        r"""Recover the upstream (pre-shock) mass density.

        Inverting the strong-shock density jump:

        .. math::

            \rho_1 = \frac{\rho_2}{R}, \quad
            R = \frac{\gamma + 1}{\gamma - 1}.

        Parameters
        ----------
        downstream_density : float or ~astropy.units.Quantity
            Downstream mass density :math:`\rho_2`. Bare float assumed
            :math:`{\rm g/cm^3}`.
        gamma : float, optional
            Adiabatic index.

        Returns
        -------
        ~astropy.units.Quantity
            Upstream density :math:`\rho_1` in :math:`{\rm g/cm^3}`.

        Notes
        -----
        This is the inverse of :meth:`compute_post_shock_density`.
        """
        rho2 = ensure_in_units(downstream_density, u.g / u.cm**3)
        return cls._pre_shock_density(rho2, gamma=gamma) * u.g / u.cm**3

    @classmethod
    def compute_pre_shock_number_density(
        cls, downstream_density: "_UnitBearingArrayLike", gamma: float = 5 / 3, mu: float = 0.61
    ) -> u.Quantity:
        r"""Recover the upstream (pre-shock) number density from a known downstream density.

        The upstream number density is obtained by inverting the strong-shock density
        jump and converting from mass density via the mean molecular weight:

        .. math::

            \rho_1 = \frac{\rho_2}{R}, \quad
            n_1 = \frac{\rho_1}{\mu m_p},

        where :math:`R = (\gamma+1)/(\gamma-1)` is the compression ratio.

        Parameters
        ----------
        downstream_density : float or ~astropy.units.Quantity
            Downstream (post-shock) mass density :math:`\rho_2`. Bare float assumed
            :math:`{\rm g/cm^3}`.
        gamma : float, optional
            Adiabatic index. Default is ``5/3``.
        mu : float, optional
            Mean molecular weight in units of the proton mass. Default is ``0.61``.

        Returns
        -------
        ~astropy.units.Quantity
            Upstream number density :math:`n_1` in :math:`{\rm cm^{-3}}`.

        Notes
        -----
        Assumes a single-fluid ideal gas with fixed composition.
        """
        downstream_density = ensure_in_units(downstream_density, u.g / u.cm**3)
        return cls._pre_shock_number_density(downstream_density, gamma=gamma, mu=mu) * u.cm**-3

    # --- Velocity --- #
    @classmethod
    def compute_post_shock_velocity(
        cls,
        shock_velocity: "_UnitBearingArrayLike",
        upstream_velocity: "_UnitBearingArrayLike" = 0.0 * u.cm / u.s,
        gamma: float = 5 / 3,
    ) -> u.Quantity:
        r"""Compute the downstream (post-shock) bulk velocity in the lab frame.

        The velocity is obtained by transforming from the shock frame using
        mass conservation:

        .. math::

            v_2 = \left(1 - \frac{1}{R}\right) v_{\rm sh}
                  + \frac{1}{R} v_1,

        where :math:`R` is the compression ratio.

        Parameters
        ----------
        shock_velocity : float or ~astropy.units.Quantity
            Shock velocity :math:`v_{\rm sh}`. Bare float assumed cm/s.
        upstream_velocity : float or ~astropy.units.Quantity, optional
            Upstream velocity :math:`v_1`. Bare float assumed cm/s.
        gamma : float, optional
            Adiabatic index.

        Returns
        -------
        ~astropy.units.Quantity
            Downstream velocity :math:`v_2` in cm/s.

        Notes
        -----
        Valid in the strong-shock limit.
        """
        v_sh = ensure_in_units(shock_velocity, u.cm / u.s)
        v_1 = ensure_in_units(upstream_velocity, u.cm / u.s)
        return cls._post_shock_velocity(v_sh, v_1, gamma=gamma) * u.cm / u.s

    @classmethod
    def compute_pre_shock_velocity(
        cls, shock_velocity: "_UnitBearingArrayLike", downstream_velocity: "_UnitBearingArrayLike", gamma: float = 5 / 3
    ) -> u.Quantity:
        r"""Recover the upstream (pre-shock) bulk velocity.

        Inverting the downstream velocity relation:

        .. math::

            v_1 = R\,v_2 - (R - 1)\,v_{\rm sh}.

        Parameters
        ----------
        shock_velocity : float or ~astropy.units.Quantity
            Shock velocity :math:`v_{\rm sh}`. Bare float assumed cm/s.
        downstream_velocity : float or ~astropy.units.Quantity
            Downstream velocity :math:`v_2`. Bare float assumed cm/s.
        gamma : float, optional
            Adiabatic index.

        Returns
        -------
        ~astropy.units.Quantity
            Upstream velocity :math:`v_1` in cm/s.

        Notes
        -----
        This is the inverse of :meth:`compute_post_shock_velocity`.
        """
        v_sh = ensure_in_units(shock_velocity, u.cm / u.s)
        v_2 = ensure_in_units(downstream_velocity, u.cm / u.s)
        return cls._pre_shock_velocity(v_sh, v_2, gamma=gamma) * u.cm / u.s

    # --- Pressure --- #
    @classmethod
    def compute_post_shock_pressure(
        cls,
        shock_velocity: "_UnitBearingArrayLike",
        upstream_density: "_UnitBearingArrayLike",
        upstream_velocity: "_UnitBearingArrayLike" = 0.0 * u.cm / u.s,
        upstream_pressure: "_UnitBearingArrayLike" = None,
        upstream_temperature: "_UnitBearingArrayLike" = None,
        mu: float = 0.61,
        gamma: float = 5 / 3,
    ) -> u.Quantity:
        r"""Compute the downstream (post-shock) gas pressure.

        Evaluates the Rankine–Hugoniot momentum conservation condition:

        .. math::

            P_2 = P_1 + \rho_1 u_1^2 \left(1 - \frac{1}{R}\right),

        where :math:`u_1 = v_{\rm sh} - v_1` is the upstream comoving velocity and
        :math:`R = (\gamma+1)/(\gamma-1)`.

        Parameters
        ----------
        shock_velocity : float or ~astropy.units.Quantity
            Shock velocity :math:`v_{\rm sh}`. Bare float assumed cm/s.
        upstream_density : float or ~astropy.units.Quantity
            Upstream mass density :math:`\rho_1`. Bare float assumed g/cm\ :sup:`3`.
        upstream_velocity : float or ~astropy.units.Quantity, optional
            Upstream bulk velocity :math:`v_1`. Bare float assumed cm/s. Default 0.
        upstream_pressure : float or ~astropy.units.Quantity, optional
            Upstream pressure :math:`P_1` in dyne/cm\ :sup:`2`. Mutually exclusive with
            ``upstream_temperature``.
        upstream_temperature : float or ~astropy.units.Quantity, optional
            Upstream temperature :math:`T_1` in K. Used to derive :math:`P_1` if
            ``upstream_pressure`` is not provided. Mutually exclusive with
            ``upstream_pressure``.
        mu : float, optional
            Mean molecular weight in units of the proton mass. Default 0.61.
        gamma : float, optional
            Adiabatic index. Default 5/3.

        Returns
        -------
        ~astropy.units.Quantity
            Downstream pressure :math:`P_2` in dyne/cm\ :sup:`2`.

        Raises
        ------
        ValueError
            If neither or both of ``upstream_pressure`` and ``upstream_temperature`` are given.
        """
        v_sh = ensure_in_units(shock_velocity, u.cm / u.s)
        rho1 = ensure_in_units(upstream_density, u.g / u.cm**3)
        v_1 = ensure_in_units(upstream_velocity, u.cm / u.s)
        P1 = ensure_in_units(upstream_pressure, u.dyn / u.cm**2) if upstream_pressure is not None else None
        T1 = ensure_in_units(upstream_temperature, u.K) if upstream_temperature is not None else None
        return (
            cls._post_shock_pressure(v_sh, rho1, v_1, upstream_pressure=P1, upstream_temperature=T1, mu=mu, gamma=gamma)
            * u.dyn
            / u.cm**2
        )

    @classmethod
    def compute_pre_shock_pressure(
        cls,
        shock_velocity: "_UnitBearingArrayLike",
        downstream_density: "_UnitBearingArrayLike",
        downstream_velocity: "_UnitBearingArrayLike" = 0.0 * u.cm / u.s,
        downstream_pressure: "_UnitBearingArrayLike" = None,
        downstream_temperature: "_UnitBearingArrayLike" = None,
        mu: float = 0.61,
        gamma: float = 5 / 3,
    ) -> u.Quantity:
        r"""Recover the upstream (pre-shock) gas pressure from downstream conditions.

        Inverts the Rankine–Hugoniot momentum conservation relation:

        .. math::

            P_1 = P_2 - \rho_1 u_1^2 \left(1 - \frac{1}{R}\right),

        where :math:`\rho_1 = \rho_2 / R` and :math:`u_1 = v_{\rm sh} - v_1`.

        Parameters
        ----------
        shock_velocity : float or ~astropy.units.Quantity
            Shock velocity :math:`v_{\rm sh}`. Bare float assumed cm/s.
        downstream_density : float or ~astropy.units.Quantity
            Downstream mass density :math:`\rho_2`. Bare float assumed g/cm\ :sup:`3`.
        downstream_velocity : float or ~astropy.units.Quantity
            Downstream bulk velocity :math:`v_2`. Bare float assumed cm/s.
        downstream_pressure : float or ~astropy.units.Quantity, optional
            Downstream pressure :math:`P_2`. Bare float assumed dyne/cm\ :sup:`2`.
            Mutually exclusive with ``downstream_temperature``.
        downstream_temperature : float or ~astropy.units.Quantity, optional
            Downstream temperature :math:`T_2` in K. Used to derive :math:`P_2` if
            ``downstream_pressure`` is not provided. Mutually exclusive with
            ``downstream_pressure``.
        mu : float, optional
            Mean molecular weight in units of the proton mass. Default 0.61.
        gamma : float, optional
            Adiabatic index. Default 5/3.

        Returns
        -------
        ~astropy.units.Quantity
            Upstream pressure :math:`P_1` in dyne/cm\ :sup:`2`.

        Raises
        ------
        ValueError
            If neither or both of ``downstream_pressure`` and ``downstream_temperature`` are given.
        """
        v_sh = ensure_in_units(shock_velocity, u.cm / u.s)
        rho2 = ensure_in_units(downstream_density, u.g / u.cm**3)
        v_2 = ensure_in_units(downstream_velocity, u.cm / u.s)
        P2 = ensure_in_units(downstream_pressure, u.dyn / u.cm**2) if downstream_pressure is not None else None
        T2 = ensure_in_units(downstream_temperature, u.K) if downstream_temperature is not None else None
        return (
            cls._pre_shock_pressure(
                v_sh, rho2, v_2, downstream_pressure=P2, downstream_temperature=T2, mu=mu, gamma=gamma
            )
            * u.dyn
            / u.cm**2
        )

    # --- Temperature --- #
    @classmethod
    def compute_post_shock_temperature(
        cls,
        shock_velocity: "_UnitBearingArrayLike",
        upstream_density: "_UnitBearingArrayLike",
        upstream_velocity: "_UnitBearingArrayLike" = 0.0 * u.cm / u.s,
        upstream_pressure: "_UnitBearingArrayLike" = None,
        upstream_temperature: "_UnitBearingArrayLike" = None,
        mu: float = 0.61,
        gamma: float = 5 / 3,
    ) -> u.Quantity:
        r"""Compute the downstream (post-shock) temperature.

        The temperature follows from the ideal gas relation:

        .. math::

            T_2 = \frac{\mu m_p}{k_B} \frac{P_2}{\rho_2},

        where :math:`P_2` is evaluated via the Rankine–Hugoniot momentum condition.

        Parameters
        ----------
        shock_velocity : float or ~astropy.units.Quantity
            Shock velocity :math:`v_{\rm sh}`. Bare float assumed cm/s.
        upstream_density : float or ~astropy.units.Quantity
            Upstream mass density :math:`\rho_1`. Bare float assumed g/cm\ :sup:`3`.
        upstream_velocity : float or ~astropy.units.Quantity, optional
            Upstream bulk velocity :math:`v_1`. Bare float assumed cm/s. Default 0.
        upstream_pressure : float or ~astropy.units.Quantity, optional
            Upstream pressure :math:`P_1` in dyne/cm\ :sup:`2`. Mutually exclusive with
            ``upstream_temperature``.
        upstream_temperature : float or ~astropy.units.Quantity, optional
            Upstream temperature :math:`T_1` in K. Mutually exclusive with
            ``upstream_pressure``.
        mu : float, optional
            Mean molecular weight in units of the proton mass. Default 0.61.
        gamma : float, optional
            Adiabatic index. Default 5/3.

        Returns
        -------
        ~astropy.units.Quantity
            Downstream temperature :math:`T_2` in K.

        Raises
        ------
        ValueError
            If neither or both of ``upstream_pressure`` and ``upstream_temperature`` are given.
        """
        v_sh = ensure_in_units(shock_velocity, u.cm / u.s)
        rho1 = ensure_in_units(upstream_density, u.g / u.cm**3)
        v_1 = ensure_in_units(upstream_velocity, u.cm / u.s)
        P1 = ensure_in_units(upstream_pressure, u.dyn / u.cm**2) if upstream_pressure is not None else None
        T1 = ensure_in_units(upstream_temperature, u.K) if upstream_temperature is not None else None
        return (
            cls._post_shock_temperature(
                v_sh, rho1, v_1, upstream_pressure=P1, upstream_temperature=T1, mu=mu, gamma=gamma
            )
            * u.K
        )

    @classmethod
    def compute_pre_shock_temperature(
        cls,
        shock_velocity: "_UnitBearingArrayLike",
        downstream_density: "_UnitBearingArrayLike",
        downstream_velocity: "_UnitBearingArrayLike" = 0.0 * u.cm / u.s,
        downstream_pressure: "_UnitBearingArrayLike" = None,
        downstream_temperature: "_UnitBearingArrayLike" = None,
        mu: float = 0.61,
        gamma: float = 5 / 3,
    ) -> u.Quantity:
        r"""Recover the upstream (pre-shock) temperature from downstream conditions.

        The upstream temperature follows from the ideal gas relation:

        .. math::

            T_1 = \frac{\mu m_p}{k_B} \frac{P_1}{\rho_1},

        where :math:`P_1` is recovered by inverting the Rankine–Hugoniot momentum condition.

        Parameters
        ----------
        shock_velocity : float or ~astropy.units.Quantity
            Shock velocity :math:`v_{\rm sh}`. Bare float assumed cm/s.
        downstream_density : float or ~astropy.units.Quantity
            Downstream mass density :math:`\rho_2`. Bare float assumed g/cm\ :sup:`3`.
        downstream_velocity : float or ~astropy.units.Quantity
            Downstream bulk velocity :math:`v_2`. Bare float assumed cm/s.
        downstream_pressure : float or ~astropy.units.Quantity, optional
            Downstream pressure :math:`P_2`. Bare float assumed dyne/cm\ :sup:`2`.
            Mutually exclusive with ``downstream_temperature``.
        downstream_temperature : float or ~astropy.units.Quantity, optional
            Downstream temperature :math:`T_2` in K. Mutually exclusive with
            ``downstream_pressure``.
        mu : float, optional
            Mean molecular weight in units of the proton mass. Default 0.61.
        gamma : float, optional
            Adiabatic index. Default 5/3.

        Returns
        -------
        ~astropy.units.Quantity
            Upstream temperature :math:`T_1` in K.

        Raises
        ------
        ValueError
            If neither or both of ``downstream_pressure`` and ``downstream_temperature`` are given.
        """
        v_sh = ensure_in_units(shock_velocity, u.cm / u.s)
        rho2 = ensure_in_units(downstream_density, u.g / u.cm**3)
        v_2 = ensure_in_units(downstream_velocity, u.cm / u.s)
        P2 = ensure_in_units(downstream_pressure, u.dyn / u.cm**2) if downstream_pressure is not None else None
        T2 = ensure_in_units(downstream_temperature, u.K) if downstream_temperature is not None else None
        return (
            cls._pre_shock_temperature(
                v_sh, rho2, v_2, downstream_pressure=P2, downstream_temperature=T2, mu=mu, gamma=gamma
            )
            * u.K
        )

    # --- Energy Density --- #
    @classmethod
    def compute_post_shock_energy_density(
        cls,
        shock_velocity: "_UnitBearingArrayLike",
        upstream_density: "_UnitBearingArrayLike",
        upstream_velocity: "_UnitBearingArrayLike" = 0.0 * u.cm / u.s,
        upstream_pressure: "_UnitBearingArrayLike" = None,
        upstream_temperature: "_UnitBearingArrayLike" = None,
        mu: float = 0.61,
        gamma: float = 5 / 3,
    ) -> u.Quantity:
        r"""Compute the downstream (post-shock) internal energy density.

        The thermal energy density is related to the post-shock pressure by

        .. math::

            e_2 = \frac{P_2}{\gamma - 1}.

        Parameters
        ----------
        shock_velocity : float or ~astropy.units.Quantity
            Shock velocity :math:`v_{\rm sh}`. Bare float assumed cm/s.
        upstream_density : float or ~astropy.units.Quantity
            Upstream mass density :math:`\rho_1`. Bare float assumed g/cm\ :sup:`3`.
        upstream_velocity : float or ~astropy.units.Quantity, optional
            Upstream bulk velocity :math:`v_1`. Bare float assumed cm/s. Default 0.
        upstream_pressure : float or ~astropy.units.Quantity, optional
            Upstream pressure :math:`P_1`. Bare float assumed dyne/cm\ :sup:`2`.
            Mutually exclusive with ``upstream_temperature``.
        upstream_temperature : float or ~astropy.units.Quantity, optional
            Upstream temperature :math:`T_1` in K. Mutually exclusive with
            ``upstream_pressure``.
        mu : float, optional
            Mean molecular weight in units of the proton mass. Default 0.61.
        gamma : float, optional
            Adiabatic index. Default 5/3.

        Returns
        -------
        ~astropy.units.Quantity
            Downstream internal (thermal) energy density :math:`e_2` in erg/cm\ :sup:`3`.

        Raises
        ------
        ValueError
            If neither or both of ``upstream_pressure`` and ``upstream_temperature`` are given.
        """
        v_sh = ensure_in_units(shock_velocity, u.cm / u.s)
        rho1 = ensure_in_units(upstream_density, u.g / u.cm**3)
        v_1 = ensure_in_units(upstream_velocity, u.cm / u.s)
        P1 = ensure_in_units(upstream_pressure, u.dyn / u.cm**2) if upstream_pressure is not None else None
        T1 = ensure_in_units(upstream_temperature, u.K) if upstream_temperature is not None else None
        return (
            cls._post_shock_energy_density(
                v_sh, rho1, v_1, upstream_pressure=P1, upstream_temperature=T1, mu=mu, gamma=gamma
            )
            * u.erg
            / u.cm**3
        )

    @classmethod
    def compute_pre_shock_energy_density(
        cls,
        shock_velocity: "_UnitBearingArrayLike",
        downstream_density: "_UnitBearingArrayLike",
        downstream_velocity: "_UnitBearingArrayLike" = 0.0 * u.cm / u.s,
        downstream_pressure: "_UnitBearingArrayLike" = None,
        downstream_temperature: "_UnitBearingArrayLike" = None,
        mu: float = 0.61,
        gamma: float = 5 / 3,
    ) -> u.Quantity:
        r"""Recover the upstream (pre-shock) internal energy density.

        .. math::

            e_1 = \frac{P_1}{\gamma - 1},

        where :math:`P_1` is recovered by inverting the Rankine–Hugoniot momentum condition.

        Parameters
        ----------
        shock_velocity : float or ~astropy.units.Quantity
            Shock velocity :math:`v_{\rm sh}`. Bare float assumed cm/s.
        downstream_density : float or ~astropy.units.Quantity
            Downstream mass density :math:`\rho_2`. Bare float assumed g/cm\ :sup:`3`.
        downstream_velocity : float or ~astropy.units.Quantity
            Downstream bulk velocity :math:`v_2`. Bare float assumed cm/s.
        downstream_pressure : float or ~astropy.units.Quantity, optional
            Downstream pressure :math:`P_2`. Bare float assumed dyne/cm\ :sup:`2`.
            Mutually exclusive with ``downstream_temperature``.
        downstream_temperature : float or ~astropy.units.Quantity, optional
            Downstream temperature :math:`T_2` in K. Mutually exclusive with
            ``downstream_pressure``.
        mu : float, optional
            Mean molecular weight in units of the proton mass. Default 0.61.
        gamma : float, optional
            Adiabatic index. Default 5/3.

        Returns
        -------
        ~astropy.units.Quantity
            Upstream internal (thermal) energy density :math:`e_1` in erg/cm\ :sup:`3`.

        Raises
        ------
        ValueError
            If neither or both of ``downstream_pressure`` and ``downstream_temperature`` are given.
        """
        v_sh = ensure_in_units(shock_velocity, u.cm / u.s)
        rho2 = ensure_in_units(downstream_density, u.g / u.cm**3)
        v_2 = ensure_in_units(downstream_velocity, u.cm / u.s)
        P2 = ensure_in_units(downstream_pressure, u.dyn / u.cm**2) if downstream_pressure is not None else None
        T2 = ensure_in_units(downstream_temperature, u.K) if downstream_temperature is not None else None
        return (
            cls._pre_shock_energy_density(
                v_sh, rho2, v_2, downstream_pressure=P2, downstream_temperature=T2, mu=mu, gamma=gamma
            )
            * u.erg
            / u.cm**3
        )

    # ---------------------------------------------- #
    # Aggregate backend + public aggregate           #
    # ---------------------------------------------- #
    @classmethod
    def _solve(
        cls,
        shock_velocity,
        flow_density,
        flow_velocity=0.0,
        flow_pressure=None,
        flow_temperature=None,
        mu: float = 0.61,
        gamma: float = 5 / 3,
        upstream: bool = False,
    ) -> dict:
        r"""CGS aggregate backend.

        Parameters
        ----------
        shock_velocity : float or ~numpy.ndarray
            Shock front velocity in cm/s.
        flow_density : float or ~numpy.ndarray
            Upstream density (``upstream=False``) or downstream density
            (``upstream=True``) in g/cm\ :sup:`3`.
        flow_velocity : float or ~numpy.ndarray, optional
            Upstream or downstream bulk velocity in cm/s. Default 0.
        flow_pressure : float or ~numpy.ndarray, optional
            Upstream or downstream pressure in dyne/cm\ :sup:`2`.
            Mutually exclusive with ``flow_temperature``.
        flow_temperature : float or ~numpy.ndarray, optional
            Upstream or downstream temperature in K.
            Mutually exclusive with ``flow_pressure``.
        mu : float, optional
            Mean molecular weight. Default 0.61.
        gamma : float, optional
            Adiabatic index. Default 5/3.
        upstream : bool, optional
            If ``False`` (default), treat inputs as upstream conditions and return
            post-shock quantities. If ``True``, treat inputs as downstream conditions
            and return pre-shock quantities.

        Returns
        -------
        dict
            Plain CGS values keyed by the relevant ``OUTPUT_FIELDS`` or
            ``UPSTREAM_OUTPUT_FIELDS`` names.
        """
        if not upstream:
            return {
                "compression_ratio": cls._compression_ratio(gamma),
                "post_shock_density": cls._post_shock_density(flow_density, gamma),
                "post_shock_number_density": cls._post_shock_number_density(flow_density, gamma, mu),
                "post_shock_velocity": cls._post_shock_velocity(shock_velocity, flow_velocity, gamma),
                "post_shock_pressure": cls._post_shock_pressure(
                    shock_velocity,
                    flow_density,
                    flow_velocity,
                    upstream_pressure=flow_pressure,
                    upstream_temperature=flow_temperature,
                    mu=mu,
                    gamma=gamma,
                ),
                "post_shock_temperature": cls._post_shock_temperature(
                    shock_velocity,
                    flow_density,
                    flow_velocity,
                    upstream_pressure=flow_pressure,
                    upstream_temperature=flow_temperature,
                    mu=mu,
                    gamma=gamma,
                ),
                "post_shock_energy_density": cls._post_shock_energy_density(
                    shock_velocity,
                    flow_density,
                    flow_velocity,
                    upstream_pressure=flow_pressure,
                    upstream_temperature=flow_temperature,
                    mu=mu,
                    gamma=gamma,
                ),
            }
        else:
            return {
                "compression_ratio": cls._compression_ratio(gamma),
                "pre_shock_density": cls._pre_shock_density(flow_density, gamma),
                "pre_shock_number_density": cls._pre_shock_number_density(flow_density, gamma, mu),
                "pre_shock_velocity": cls._pre_shock_velocity(shock_velocity, flow_velocity, gamma),
                "pre_shock_pressure": cls._pre_shock_pressure(
                    shock_velocity,
                    flow_density,
                    flow_velocity,
                    downstream_pressure=flow_pressure,
                    downstream_temperature=flow_temperature,
                    mu=mu,
                    gamma=gamma,
                ),
                "pre_shock_temperature": cls._pre_shock_temperature(
                    shock_velocity,
                    flow_density,
                    flow_velocity,
                    downstream_pressure=flow_pressure,
                    downstream_temperature=flow_temperature,
                    mu=mu,
                    gamma=gamma,
                ),
                "pre_shock_energy_density": cls._pre_shock_energy_density(
                    shock_velocity,
                    flow_density,
                    flow_velocity,
                    downstream_pressure=flow_pressure,
                    downstream_temperature=flow_temperature,
                    mu=mu,
                    gamma=gamma,
                ),
            }

    @classmethod
    def solve(
        cls,
        shock_velocity: "_UnitBearingArrayLike",
        flow_density: "_UnitBearingArrayLike",
        flow_velocity: "_UnitBearingArrayLike" = 0.0 * u.cm / u.s,
        flow_pressure: "_UnitBearingArrayLike" = None,
        flow_temperature: "_UnitBearingArrayLike" = None,
        mu: float = 0.61,
        gamma: float = 5 / 3,
        upstream: bool = False,
    ):
        r"""Compute all jump-condition quantities in one call.

        Parameters
        ----------
        shock_velocity : float or ~astropy.units.Quantity
            Shock front velocity. Bare float assumed cm/s.
        flow_density : float or ~astropy.units.Quantity
            Upstream density when ``upstream=False``; downstream density when
            ``upstream=True``. Bare float assumed g/cm\ :sup:`3`.
        flow_velocity : float or ~astropy.units.Quantity, optional
            Upstream or downstream bulk velocity. Bare float assumed cm/s. Default 0.
        flow_pressure : float or ~astropy.units.Quantity, optional
            Upstream pressure (``upstream=False``) or downstream pressure
            (``upstream=True``). Bare float assumed dyne/cm\ :sup:`2`.
            Mutually exclusive with ``flow_temperature``.
        flow_temperature : float or ~astropy.units.Quantity, optional
            Upstream or downstream temperature in K.
            Mutually exclusive with ``flow_pressure``.
        mu : float, optional
            Mean molecular weight. Default 0.61.
        gamma : float, optional
            Adiabatic index. Default 5/3.
        upstream : bool, optional
            If ``False`` (default), return a :class:`StrongShockConditionsResult`
            with post-shock quantities. If ``True``, return a
            :class:`StrongShockConditionsUpstreamResult` with pre-shock quantities.

        Returns
        -------
        StrongShockConditionsResult or StrongShockConditionsUpstreamResult
            Named tuple with fields from :attr:`OUTPUT_FIELDS` (``upstream=False``)
            or :attr:`UPSTREAM_OUTPUT_FIELDS` (``upstream=True``).
        """
        v_sh = ensure_in_units(shock_velocity, u.cm / u.s)
        rho = ensure_in_units(flow_density, u.g / u.cm**3)
        v = ensure_in_units(flow_velocity, u.cm / u.s)
        P = ensure_in_units(flow_pressure, u.dyn / u.cm**2) if flow_pressure is not None else None
        T = ensure_in_units(flow_temperature, u.K) if flow_temperature is not None else None
        result = cls._solve(v_sh, rho, v, P, T, mu, gamma, upstream=upstream)
        return cls._make_upstream_result(result) if upstream else cls._make_result(result)


class StrongColdShockConditions(StrongShockConditions):
    r"""Strong-shock Rankine–Hugoniot conditions in the cold-upstream limit.

    This class specializes :class:`StrongShockConditions` to the case where the
    upstream medium is dynamically cold, i.e. the upstream pressure and internal
    energy are negligible (:math:`P_1 \approx 0`, :math:`T_1 \approx 0`).

    In this limit, the jump conditions simplify considerably:

    - The compression ratio is constant:

      .. math::

          R = \frac{\gamma + 1}{\gamma - 1}

    - The downstream pressure is set entirely by the upstream kinetic energy:

      .. math::

          P_2 = \rho_1 u_1^2 \left(1 - \frac{1}{R}\right)

    - The downstream temperature depends only on the shock-frame velocity and
      is independent of upstream density.

    Because the upstream thermal state is neglected, no upstream pressure or
    temperature needs to be provided.

    Notes
    -----
    - This is the **simplest and most commonly used strong-shock model** for
      astrophysical blast waves propagating into cold media (e.g. ISM, CSM).
    - Thermodynamic inversion is not defined in the cold limit, so the upstream
      solution returns only kinematic quantities.
    - Compared to :class:`StrongShockConditions`, this class removes all
      dependence on upstream pressure and temperature.

    See Also
    --------
    StrongShockConditions : Strong shocks with a finite upstream thermal state.
    WeakShockConditions : General finite-Mach-number shock solver.
    """

    OUTPUT_FIELDS = (
        ("compression_ratio", None),
        ("post_shock_density", u.g / u.cm**3),
        ("post_shock_number_density", u.cm**-3),
        ("post_shock_velocity", u.cm / u.s),
        ("post_shock_pressure", u.dyn / u.cm**2),
        ("post_shock_temperature", u.K),
        ("post_shock_thermal_energy_density", u.erg / u.cm**3),
    )

    UPSTREAM_OUTPUT_FIELDS = (
        ("compression_ratio", None),
        ("pre_shock_density", u.g / u.cm**3),
        ("pre_shock_number_density", u.cm**-3),
        ("pre_shock_velocity", u.cm / u.s),
    )

    # ---------------------------------------------- #
    # Private CGS helpers (cold-limit overrides)     #
    # ---------------------------------------------- #

    # --- Pressure --- #
    @classmethod
    def _post_shock_pressure(
        cls,
        shock_velocity: "_ArrayLike",
        upstream_density: "_ArrayLike",
        upstream_velocity: "_ArrayLike" = 0.0,
        gamma: float = 5 / 3,
    ):
        r"""Post-shock pressure in the cold-upstream limit: :math:`P_2 = \rho_1 u_1^2 (1 - 1/R)`.

        Parameters
        ----------
        shock_velocity : float or ~numpy.ndarray
            Shock velocity :math:`v_{\rm sh}` in cm/s (lab frame).
        upstream_density : float or ~numpy.ndarray
            Upstream mass density :math:`\rho_1` in g/cm\ :sup:`3`.
        upstream_velocity : float or ~numpy.ndarray, optional
            Upstream bulk velocity :math:`v_1` in cm/s. Default 0.
        gamma : float, optional
            Adiabatic index. Default 5/3.

        Returns
        -------
        float or ~numpy.ndarray
            Post-shock pressure :math:`P_2` in dyne/cm\ :sup:`2`.
        """
        shock_velocity = np.asarray(shock_velocity)
        upstream_density = np.asarray(upstream_density)
        upstream_velocity = np.asarray(upstream_velocity)
        R = cls._compression_ratio(gamma=gamma)
        u1 = shock_velocity - upstream_velocity
        res = upstream_density * u1**2 * (1 - 1 / R)
        return res if res.ndim > 0 else res.item()

    # --- Temperature --- #
    @classmethod
    def _post_shock_temperature(
        cls, shock_velocity: "_ArrayLike", upstream_velocity: "_ArrayLike" = 0.0, mu: float = 0.61, gamma: float = 5 / 3
    ):
        r"""Post-shock temperature in the cold-upstream limit.

        In the cold limit, :math:`T_2` is independent of upstream density:

        .. math::

            T_2 = \frac{\mu m_p}{k_B} \frac{R - 1}{R^2}\, u_1^2,

        where :math:`u_1 = v_{\rm sh} - v_1`.

        Parameters
        ----------
        shock_velocity : float or ~numpy.ndarray
            Shock velocity :math:`v_{\rm sh}` in cm/s.
        upstream_velocity : float or ~numpy.ndarray, optional
            Upstream bulk velocity :math:`v_1` in cm/s. Default 0.
        mu : float, optional
            Mean molecular weight. Default 0.61.
        gamma : float, optional
            Adiabatic index. Default 5/3.

        Returns
        -------
        float or ~numpy.ndarray
            Post-shock temperature :math:`T_2` in K.
        """
        shock_velocity = np.asarray(shock_velocity)
        upstream_velocity = np.asarray(upstream_velocity)
        R = cls._compression_ratio(gamma=gamma)
        u1 = shock_velocity - upstream_velocity
        res = (mu * m_p_cgs / k_B_cgs) * (R - 1) / R**2 * u1**2
        return res if res.ndim > 0 else res.item()

    # --- Energy Density --- #
    @classmethod
    def _post_shock_thermal_energy_density(
        cls,
        shock_velocity: "_ArrayLike",
        upstream_density: "_ArrayLike",
        upstream_velocity: "_ArrayLike" = 0.0,
        gamma: float = 5 / 3,
    ):
        r"""Post-shock thermal energy density: :math:`e_2 = P_2 / (\gamma - 1)`.

        Parameters
        ----------
        shock_velocity : float or ~numpy.ndarray
            Shock velocity :math:`v_{\rm sh}` in cm/s.
        upstream_density : float or ~numpy.ndarray
            Upstream mass density :math:`\rho_1` in g/cm\ :sup:`3`.
        upstream_velocity : float or ~numpy.ndarray, optional
            Upstream bulk velocity :math:`v_1` in cm/s. Default 0.
        gamma : float, optional
            Adiabatic index. Default 5/3.

        Returns
        -------
        float or ~numpy.ndarray
            Post-shock thermal energy density :math:`e_2` in erg/cm\ :sup:`3`.
        """
        P2 = np.asarray(cls._post_shock_pressure(shock_velocity, upstream_density, upstream_velocity, gamma))
        e2 = P2 / (gamma - 1)
        return e2 if e2.ndim > 0 else e2.item()

    # ---------------------------------------------- #
    # Public overrides (cold-specific signatures)    #
    # ---------------------------------------------- #

    @classmethod
    def compute_post_shock_pressure(
        cls,
        shock_velocity: "_UnitBearingArrayLike",
        upstream_density: "_UnitBearingArrayLike",
        upstream_velocity: "_UnitBearingArrayLike" = 0.0 * u.cm / u.s,
        gamma: float = 5 / 3,
    ) -> u.Quantity:
        r"""Post-shock gas pressure in the cold-upstream limit.

        Parameters
        ----------
        shock_velocity : float or ~astropy.units.Quantity
            Shock velocity :math:`v_{\rm sh}`. Bare float assumed cm/s.
        upstream_density : float or ~astropy.units.Quantity
            Upstream mass density :math:`\rho_1`. Bare float assumed g/cm\ :sup:`3`.
        upstream_velocity : float or ~astropy.units.Quantity, optional
            Upstream bulk velocity :math:`v_1`. Bare float assumed cm/s. Default 0.
        gamma : float, optional
            Adiabatic index. Default 5/3.

        Returns
        -------
        ~astropy.units.Quantity
            Post-shock pressure :math:`P_2` in dyne/cm\ :sup:`2`.
        """
        v_sh = ensure_in_units(shock_velocity, u.cm / u.s)
        rho1 = ensure_in_units(upstream_density, u.g / u.cm**3)
        v_1 = ensure_in_units(upstream_velocity, u.cm / u.s)
        return cls._post_shock_pressure(v_sh, rho1, v_1, gamma) * u.dyn / u.cm**2

    @classmethod
    def compute_post_shock_temperature(
        cls,
        shock_velocity: "_UnitBearingArrayLike",
        upstream_velocity: "_UnitBearingArrayLike" = 0.0 * u.cm / u.s,
        mu: float = 0.61,
        gamma: float = 5 / 3,
    ) -> u.Quantity:
        r"""Post-shock temperature in the cold-upstream limit.

        Does not require upstream density — see :meth:`_post_shock_temperature`.

        Parameters
        ----------
        shock_velocity : float or ~astropy.units.Quantity
            Shock velocity :math:`v_{\rm sh}`. Bare float assumed cm/s.
        upstream_velocity : float or ~astropy.units.Quantity, optional
            Upstream bulk velocity :math:`v_1`. Bare float assumed cm/s. Default 0.
        mu : float, optional
            Mean molecular weight. Default 0.61.
        gamma : float, optional
            Adiabatic index. Default 5/3.

        Returns
        -------
        ~astropy.units.Quantity
            Post-shock temperature :math:`T_2` in K.
        """
        v_sh = ensure_in_units(shock_velocity, u.cm / u.s)
        v_1 = ensure_in_units(upstream_velocity, u.cm / u.s)
        return cls._post_shock_temperature(v_sh, v_1, mu, gamma) * u.K

    @classmethod
    def compute_post_shock_thermal_energy_density(
        cls,
        shock_velocity: "_UnitBearingArrayLike",
        upstream_density: "_UnitBearingArrayLike",
        upstream_velocity: "_UnitBearingArrayLike" = 0.0 * u.cm / u.s,
        gamma: float = 5 / 3,
    ) -> u.Quantity:
        r"""Post-shock thermal energy density in the cold-upstream limit.

        Parameters
        ----------
        shock_velocity : float or ~astropy.units.Quantity
            Shock velocity :math:`v_{\rm sh}`. Bare float assumed cm/s.
        upstream_density : float or ~astropy.units.Quantity
            Upstream mass density :math:`\rho_1`. Bare float assumed g/cm\ :sup:`3`.
        upstream_velocity : float or ~astropy.units.Quantity, optional
            Upstream bulk velocity :math:`v_1`. Bare float assumed cm/s. Default 0.
        gamma : float, optional
            Adiabatic index. Default 5/3.

        Returns
        -------
        ~astropy.units.Quantity
            Post-shock thermal energy density :math:`e_2` in erg/cm\ :sup:`3`.
        """
        v_sh = ensure_in_units(shock_velocity, u.cm / u.s)
        rho1 = ensure_in_units(upstream_density, u.g / u.cm**3)
        v_1 = ensure_in_units(upstream_velocity, u.cm / u.s)
        return cls._post_shock_thermal_energy_density(v_sh, rho1, v_1, gamma) * u.erg / u.cm**3

    # ---------------------------------------------- #
    # Public aggregate                               #
    # ---------------------------------------------- #
    @classmethod
    def _solve(
        cls,
        shock_velocity,
        flow_density,
        flow_velocity=0.0,
        gamma: float = 5 / 3,
        mu: float = 0.61,
        upstream: bool = False,
    ) -> dict:
        r"""CGS aggregate backend.

        Parameters
        ----------
        shock_velocity : float or ~numpy.ndarray
            Shock front velocity in cm/s.
        flow_density : float or ~numpy.ndarray
            Upstream density (``upstream=False``) or downstream density
            (``upstream=True``) in g/cm\ :sup:`3`.
        flow_velocity : float or ~numpy.ndarray, optional
            Upstream or downstream bulk velocity in cm/s. Default 0.
        gamma : float, optional
            Adiabatic index. Default 5/3.
        mu : float, optional
            Mean molecular weight. Default 0.61.
        upstream : bool, optional
            If ``False`` (default), return post-shock quantities from upstream inputs.
            If ``True``, return pre-shock quantities from downstream inputs.

        Returns
        -------
        dict
            Plain CGS values.
        """
        if not upstream:
            return {
                "compression_ratio": cls._compression_ratio(gamma),
                "post_shock_density": cls._post_shock_density(flow_density, gamma),
                "post_shock_number_density": cls._post_shock_number_density(flow_density, gamma, mu),
                "post_shock_velocity": cls._post_shock_velocity(shock_velocity, flow_velocity, gamma),
                "post_shock_pressure": cls._post_shock_pressure(shock_velocity, flow_density, flow_velocity, gamma),
                "post_shock_temperature": cls._post_shock_temperature(shock_velocity, flow_velocity, mu, gamma),
                "post_shock_thermal_energy_density": cls._post_shock_thermal_energy_density(
                    shock_velocity, flow_density, flow_velocity, gamma
                ),
            }
        else:
            return {
                "compression_ratio": cls._compression_ratio(gamma),
                "pre_shock_density": cls._pre_shock_density(flow_density, gamma),
                "pre_shock_number_density": cls._pre_shock_number_density(flow_density, gamma, mu),
                "pre_shock_velocity": cls._pre_shock_velocity(shock_velocity, flow_velocity, gamma),
            }

    @classmethod
    def solve(
        cls,
        shock_velocity: "_UnitBearingArrayLike",
        flow_density: "_UnitBearingArrayLike",
        flow_velocity: "_UnitBearingArrayLike" = 0.0 * u.cm / u.s,
        gamma: float = 5 / 3,
        mu: float = 0.61,
        upstream: bool = False,
    ):
        r"""Compute all jump-condition quantities in one call.

        Parameters
        ----------
        shock_velocity : float or ~astropy.units.Quantity
            Shock front velocity. Bare float assumed cm/s.
        flow_density : float or ~astropy.units.Quantity
            Upstream density when ``upstream=False``; downstream density when
            ``upstream=True``. Bare float assumed g/cm\ :sup:`3`.
        flow_velocity : float or ~astropy.units.Quantity, optional
            Upstream or downstream bulk velocity. Bare float assumed cm/s. Default 0.
        gamma : float, optional
            Adiabatic index. Default 5/3.
        mu : float, optional
            Mean molecular weight. Default 0.61.
        upstream : bool, optional
            If ``False`` (default), return a :class:`StrongColdShockConditionsResult`
            with post-shock quantities. If ``True``, return a
            :class:`StrongColdShockConditionsUpstreamResult` with pre-shock
            kinematic quantities (density, number density, velocity only).

        Returns
        -------
        StrongColdShockConditionsResult or StrongColdShockConditionsUpstreamResult
        """
        v_sh = ensure_in_units(shock_velocity, u.cm / u.s)
        rho = ensure_in_units(flow_density, u.g / u.cm**3)
        v = ensure_in_units(flow_velocity, u.cm / u.s)
        result = cls._solve(v_sh, rho, v, gamma, mu, upstream=upstream)
        return cls._make_upstream_result(result) if upstream else cls._make_result(result)


class WeakShockConditions(JumpConditions):
    r"""Rankine-Hugoniot jump conditions for a general normal shock at finite Mach number.

    Unlike :class:`StrongShockConditions`, the compression ratio here depends on the
    upstream Mach number :math:`\mathcal{M}_1`:

    .. math::

        R = \frac{(\gamma+1)\,\mathcal{M}_1^2}{(\gamma-1)\,\mathcal{M}_1^2 + 2},

    which approaches :math:`(\gamma+1)/(\gamma-1)` only in the limit
    :math:`\mathcal{M}_1 \to \infty`.

    The Mach number is not passed explicitly to thermodynamic methods
    (pressure, temperature, energy density); instead, those methods compute it
    self-consistently from the shock-frame velocity and the provided thermodynamic
    state via :meth:`_resolve_mach_number`. Kinematic methods (density, velocity)
    take the Mach number explicitly because they carry no thermodynamic information.

    All ``compute_*`` public methods accept bare floats (assumed CGS) or
    :class:`~astropy.units.Quantity` objects and return :class:`~astropy.units.Quantity`.
    """

    OUTPUT_FIELDS = (
        ("compression_ratio", None),
        ("post_shock_density", u.g / u.cm**3),
        ("post_shock_number_density", u.cm**-3),
        ("post_shock_velocity", u.cm / u.s),
        ("post_shock_pressure", u.dyn / u.cm**2),
        ("post_shock_temperature", u.K),
        ("post_shock_energy_density", u.erg / u.cm**3),
    )

    UPSTREAM_OUTPUT_FIELDS = (
        ("compression_ratio", None),
        ("pre_shock_density", u.g / u.cm**3),
        ("pre_shock_number_density", u.cm**-3),
        ("pre_shock_velocity", u.cm / u.s),
        ("pre_shock_pressure", u.dyn / u.cm**2),
        ("pre_shock_temperature", u.K),
        ("pre_shock_energy_density", u.erg / u.cm**3),
    )

    # ---------------------------------------------- #
    # Private CGS helpers                            #
    # ---------------------------------------------- #

    # --- Mach number --- #
    @classmethod
    def _post_shock_mach_from_upstream(cls, upstream_mach_number: "_ArrayLike", gamma: float = 5 / 3) -> "_ArrayLike":
        r"""Compute the downstream Mach number from the upstream Mach number.

        For a normal shock, the upstream and downstream Mach numbers are related by

        .. math::

            \mathcal{M}_2^2 =
            \frac{(\gamma - 1)\,\mathcal{M}_1^2 + 2}
                 {2\gamma\,\mathcal{M}_1^2 - (\gamma - 1)}.

        Parameters
        ----------
        upstream_mach_number : float or ~numpy.ndarray
            Upstream Mach number :math:`\mathcal{M}_1`.
        gamma : float, optional
            Adiabatic index :math:`\gamma`. Default is ``5/3``.

        Returns
        -------
        float or ~numpy.ndarray
            Downstream Mach number :math:`\mathcal{M}_2`.

        Notes
        -----
        This relation assumes a steady, one-dimensional normal shock in an ideal gas.
        The downstream flow is always subsonic (:math:`\mathcal{M}_2 < 1`) for
        :math:`\mathcal{M}_1 > 1`.
        """
        # Construct the square of the downstream Mach number.
        M1sq = np.asarray(upstream_mach_number) ** 2

        # Apply the Rankine–Hugoniot relation to compute M2^2.
        M2 = np.sqrt(((gamma - 1) * M1sq + 2) / (2 * gamma * M1sq - (gamma - 1)))

        # Return.
        return M2 if M2.ndim > 0 else M2.item()

    @classmethod
    def _pre_shock_mach_from_downstream(
        cls, downstream_mach_number: "_ArrayLike", gamma: float = 5 / 3
    ) -> "_ArrayLike":
        r"""Compute the upstream Mach number from the downstream Mach number.

        This is the inverse of :meth:`_post_shock_mach_from_upstream`, given by

        .. math::

            \mathcal{M}_1^2 =
            \frac{(\gamma - 1)\,\mathcal{M}_2^2 + 2}
                 {2\gamma\,\mathcal{M}_2^2 - (\gamma - 1)}.

        Parameters
        ----------
        downstream_mach_number : float or ~numpy.ndarray
            Downstream Mach number :math:`\mathcal{M}_2`.
        gamma : float, optional
            Adiabatic index :math:`\gamma`. Default is ``5/3``.

        Returns
        -------
        float or ~numpy.ndarray
            Upstream Mach number :math:`\mathcal{M}_1`.

        Notes
        -----
        This relation is valid for a normal shock in an ideal gas.
        """
        # Compute the square of the upstream Mach number from the downstream Mach number.
        M2sq = np.asarray(downstream_mach_number) ** 2

        # Construct the square of the upstream Mach number using the inverse Rankine–Hugoniot relation.
        M1 = np.sqrt(((gamma - 1) * M2sq + 2) / (2 * gamma * M2sq - (gamma - 1)))
        return M1 if M1.ndim > 0 else M1.item()

    @classmethod
    def _resolve_sound_speed(
        cls,
        temperature: "_ArrayLike" = None,
        density: "_ArrayLike" = None,
        sound_speed: "_ArrayLike" = None,
        pressure: "_ArrayLike" = None,
        mu: float = 0.61,
        gamma: float = 5 / 3,
    ):
        r"""Resolve the sound speed for an ideal gas from available inputs.

        The sound speed is defined as

        .. math::

            c_s^2 = \gamma \frac{P}{\rho}
            \quad \text{or} \quad
            c_s^2 = \gamma \frac{k_B T}{\mu m_p},

        depending on which thermodynamic variables are provided.

        Parameters
        ----------
        temperature : float or ~numpy.ndarray, optional
            Gas temperature :math:`T` in K.
        density : float or ~numpy.ndarray, optional
            Mass density :math:`\rho` in g/cm\ :sup:`3`.
        sound_speed : float or ~numpy.ndarray, optional
            Directly specified sound speed :math:`c_s` in cm/s.
        pressure : float or ~numpy.ndarray, optional
            Gas pressure :math:`P` in dyne/cm\ :sup:`2`.
        mu : float, optional
            Mean molecular weight. Default 0.61.
        gamma : float, optional
            Adiabatic index. Default 5/3.

        Returns
        -------
        float or ~numpy.ndarray
            Sound speed :math:`c_s` in cm/s.

        Raises
        ------
        ValueError
            If insufficient or inconsistent inputs are provided.

        Notes
        -----
        Priority order:

        1. If ``sound_speed`` is provided, it is returned directly.
        2. If ``temperature`` is provided, compute via the ideal gas relation.
        3. If both ``pressure`` and ``density`` are provided, compute via
           :math:`c_s^2 = \gamma P / \rho`.

        All other combinations are considered invalid.
        """
        # --- Direct case --- #
        if sound_speed is not None:
            cs = np.asarray(sound_speed)
            return cs if cs.ndim > 0 else cs.item()

        # --- Temperature-based --- #
        if temperature is not None:
            T = np.asarray(temperature)
            cs = np.sqrt(gamma * k_B_cgs * T / (mu * m_p_cgs))
            return cs if cs.ndim > 0 else cs.item()

        # --- Pressure + density --- #
        if (pressure is not None) and (density is not None):
            P = np.asarray(pressure)
            rho = np.asarray(density)
            cs = np.sqrt(gamma * P / rho)
            return cs if cs.ndim > 0 else cs.item()

        # --- Invalid combinations --- #
        if (pressure is not None) or (density is not None):
            raise ValueError("Must provide both pressure and density to compute sound speed.")

        raise ValueError(
            "Insufficient information to compute sound speed. "
            "Provide sound_speed, or temperature, or both pressure and density."
        )

    @classmethod
    def _resolve_mach_number(
        cls,
        shock_velocity: "_ArrayLike",
        flow_velocity: "_ArrayLike" = 0.0,
        temperature: "_ArrayLike" = None,
        sound_speed: "_ArrayLike" = None,
        density: "_ArrayLike" = None,
        pressure: "_ArrayLike" = None,
        mu: float = 0.61,
        gamma: float = 5 / 3,
    ):
        r"""Resolve the Mach number from kinematic and thermodynamic inputs.

        The Mach number is defined as

        .. math::

            \mathcal{M} = \frac{u}{c_s},

        where

        .. math::

            u = v_{\rm sh} - v

        is the flow velocity in the shock frame and :math:`c_s` is the sound speed.

        The sound speed is resolved using :meth:`_resolve_sound_speed`.

        Parameters
        ----------
        shock_velocity : float or ~numpy.ndarray
            Shock velocity :math:`v_{\rm sh}` in cm/s.
        flow_velocity : float or ~numpy.ndarray
            Fluid velocity :math:`v` in cm/s (lab frame).
        temperature : float or ~numpy.ndarray, optional
            Gas temperature :math:`T` in K.
        sound_speed : float or ~numpy.ndarray, optional
            Sound speed :math:`c_s` in cm/s.
        density : float or ~numpy.ndarray, optional
            Mass density :math:`\rho` in g/cm\ :sup:`3`.
        pressure : float or ~numpy.ndarray, optional
            Gas pressure :math:`P` in dyne/cm\ :sup:`2`.
        mu : float, optional
            Mean molecular weight. Default 0.61.
        gamma : float, optional
            Adiabatic index. Default 5/3.

        Returns
        -------
        float or ~numpy.ndarray
            Mach number :math:`\mathcal{M}`.

        Raises
        ------
        ValueError
            If insufficient information is provided to compute the sound speed.

        Notes
        -----
        The Mach number is always computed in the shock frame. If the flow
        is at rest (:math:`v = 0`), this reduces to

        .. math::

            \mathcal{M} = \frac{v_{\rm sh}}{c_s}.
        """
        # --- Coerce inputs --- #
        shock_velocity = np.asarray(shock_velocity)
        flow_velocity = np.asarray(flow_velocity)

        # --- Compute shock-frame velocity --- #
        _u = shock_velocity - flow_velocity

        # --- Resolve sound speed --- #
        cs = cls._resolve_sound_speed(
            temperature=temperature,
            density=density,
            sound_speed=sound_speed,
            pressure=pressure,
            mu=mu,
            gamma=gamma,
        )

        cs = np.asarray(cs)

        # --- Compute Mach number --- #
        M = _u / cs

        return M if M.ndim > 0 else M.item()

    # --- Compression Ratios --- #
    @classmethod
    def _compression_ratio(cls, upstream_mach_number: "_ArrayLike", gamma: float = 5 / 3) -> "_ArrayLike":
        r"""Compute the compression ratio from the upstream Mach number.

        The compression ratio is defined as

        .. math::

            R \equiv \frac{\rho_2}{\rho_1}
            = \frac{(\gamma + 1)\,\mathcal{M}_1^2}
                   {(\gamma - 1)\,\mathcal{M}_1^2 + 2}.

        Parameters
        ----------
        upstream_mach_number : float or ~numpy.ndarray
            Upstream Mach number :math:`\mathcal{M}_1`.
        gamma : float, optional
            Adiabatic index :math:`\gamma`. Default is ``5/3``.

        Returns
        -------
        float or ~numpy.ndarray
            Compression ratio :math:`R = \rho_2 / \rho_1`.

        Notes
        -----
        In the strong-shock limit (:math:`\mathcal{M}_1 \to \infty`), this reduces to

        .. math::

            R \to \frac{\gamma + 1}{\gamma - 1}.
        """
        M1sq = np.asarray(upstream_mach_number) ** 2
        R = (gamma + 1) * M1sq / ((gamma - 1) * M1sq + 2)
        return R if R.ndim > 0 else R.item()

    @classmethod
    def _compression_ratio_from_downstream(
        cls, downstream_mach_number: "_ArrayLike", gamma: float = 5 / 3
    ) -> "_ArrayLike":
        r"""Compute the compression ratio from the downstream Mach number.

        Derived by substituting :math:`\mathcal{M}_1^2 = [(\gamma-1)\mathcal{M}_2^2 + 2]
        / [2\gamma\mathcal{M}_2^2 - (\gamma-1)]` into the upstream form:

        .. math::

            R = \frac{(\gamma - 1)\,\mathcal{M}_2^2 + 2}
                     {(\gamma + 1)\,\mathcal{M}_2^2}.

        Limits: :math:`R \to 1` as :math:`\mathcal{M}_2 \to 1` (no shock);
        :math:`R \to (\gamma+1)/(\gamma-1)` as
        :math:`\mathcal{M}_2 \to \sqrt{(\gamma-1)/(2\gamma)}` (strong-shock limit).

        Parameters
        ----------
        downstream_mach_number : float or ~numpy.ndarray
            Downstream Mach number :math:`\mathcal{M}_2 \in (0,\,1]`.
        gamma : float, optional
            Adiabatic index :math:`\gamma`. Default is ``5/3``.

        Returns
        -------
        float or ~numpy.ndarray
            Compression ratio :math:`R = \rho_2 / \rho_1`.
        """
        M2sq = np.asarray(downstream_mach_number) ** 2
        R = ((gamma - 1) * M2sq + 2) / ((gamma + 1) * M2sq)
        return R if R.ndim > 0 else R.item()

    # --- Density --- #
    @classmethod
    def _post_shock_density(
        cls, upstream_density: "_ArrayLike", upstream_mach_number: "_ArrayLike", gamma: float = 5 / 3
    ):
        r"""Downstream (post-shock) mass density for a general normal shock.

        The density jump depends on the upstream Mach number:

        .. math::

            \rho_2 = R(\mathcal{M}_1)\,\rho_1,
            \quad R = \frac{(\gamma+1)\,\mathcal{M}_1^2}{(\gamma-1)\,\mathcal{M}_1^2 + 2}.

        Parameters
        ----------
        upstream_density : float or ~numpy.ndarray
            Upstream mass density :math:`\rho_1` in g/cm\ :sup:`3`.
        upstream_mach_number : float or ~numpy.ndarray
            Upstream Mach number :math:`\mathcal{M}_1`.
        gamma : float, optional
            Adiabatic index. Default 5/3.

        Returns
        -------
        float or ~numpy.ndarray
            Downstream mass density :math:`\rho_2` in g/cm\ :sup:`3`.
        """
        R = cls._compression_ratio(upstream_mach_number, gamma)
        return R * upstream_density

    @classmethod
    def _post_shock_number_density(
        cls, upstream_density: "_ArrayLike", upstream_mach_number: "_ArrayLike", gamma: float = 5 / 3, mu: float = 0.61
    ):
        r"""Downstream (post-shock) number density: :math:`n_2 = \rho_2 / (\mu m_p)`.

        Parameters
        ----------
        upstream_density : float or ~numpy.ndarray
            Upstream mass density :math:`\rho_1` in g/cm\ :sup:`3`.
        upstream_mach_number : float or ~numpy.ndarray
            Upstream Mach number :math:`\mathcal{M}_1`.
        gamma : float, optional
            Adiabatic index. Default 5/3.
        mu : float, optional
            Mean molecular weight. Default 0.61.

        Returns
        -------
        float or ~numpy.ndarray
            Downstream number density :math:`n_2` in cm\ :sup:`-3`.
        """
        rho2 = cls._post_shock_density(upstream_density, upstream_mach_number, gamma)
        return rho2 / (mu * m_p_cgs)

    @classmethod
    def _pre_shock_density(
        cls, downstream_density: "_ArrayLike", downstream_mach_number: "_ArrayLike", gamma: float = 5 / 3
    ):
        r"""Upstream (pre-shock) mass density from downstream conditions.

        .. math::

            \rho_1 = \frac{\rho_2}{R(\mathcal{M}_2)}

        Parameters
        ----------
        downstream_density : float or ~numpy.ndarray
            Downstream mass density :math:`\rho_2` in g/cm\ :sup:`3`.
        downstream_mach_number : float or ~numpy.ndarray
            Downstream Mach number :math:`\mathcal{M}_2`.
        gamma : float, optional
            Adiabatic index. Default 5/3.

        Returns
        -------
        float or ~numpy.ndarray
            Upstream mass density :math:`\rho_1` in g/cm\ :sup:`3`.
        """
        R = cls._compression_ratio_from_downstream(downstream_mach_number, gamma)
        return downstream_density / R

    @classmethod
    def _pre_shock_number_density(
        cls,
        downstream_density: "_ArrayLike",
        downstream_mach_number: "_ArrayLike",
        gamma: float = 5 / 3,
        mu: float = 0.61,
    ):
        r"""Upstream (pre-shock) number density from downstream conditions.

        Parameters
        ----------
        downstream_density : float or ~numpy.ndarray
            Downstream mass density :math:`\rho_2` in g/cm\ :sup:`3`.
        downstream_mach_number : float or ~numpy.ndarray
            Downstream Mach number :math:`\mathcal{M}_2`.
        gamma : float, optional
            Adiabatic index. Default 5/3.
        mu : float, optional
            Mean molecular weight. Default 0.61.

        Returns
        -------
        float or ~numpy.ndarray
            Upstream number density :math:`n_1` in cm\ :sup:`-3`.
        """
        rho1 = cls._pre_shock_density(downstream_density, downstream_mach_number, gamma)
        return rho1 / (mu * m_p_cgs)

    # --- Velocity --- #
    @classmethod
    def _post_shock_velocity(
        cls,
        shock_velocity: "_ArrayLike",
        upstream_mach_number: "_ArrayLike",
        upstream_velocity: "_ArrayLike" = 0.0,
        gamma: float = 5 / 3,
    ):
        r"""Downstream bulk velocity in the lab frame.

        .. math::

            v_2 = \left(1 - \frac{1}{R}\right) v_{\rm sh} + \frac{1}{R}\,v_1,
            \quad R = R(\mathcal{M}_1).

        Parameters
        ----------
        shock_velocity : float or ~numpy.ndarray
            Shock velocity :math:`v_{\rm sh}` in cm/s (lab frame).
        upstream_mach_number : float or ~numpy.ndarray
            Upstream Mach number :math:`\mathcal{M}_1`.
        upstream_velocity : float or ~numpy.ndarray, optional
            Upstream bulk velocity :math:`v_1` in cm/s. Default 0.
        gamma : float, optional
            Adiabatic index. Default 5/3.

        Returns
        -------
        float or ~numpy.ndarray
            Downstream bulk velocity :math:`v_2` in cm/s (lab frame).
        """
        # Cast the shock velocity and upstream velocity to arrays for consistent broadcasting.
        shock_velocity = np.asarray(shock_velocity)
        upstream_velocity = np.asarray(upstream_velocity)
        upstream_mach_number = np.asarray(upstream_mach_number)

        # Set the compression ratio.
        R = cls._compression_ratio(upstream_mach_number, gamma)

        # Compute the downstream velocity using the Rankine–Hugoniot relation.
        res = (1 / R) * upstream_velocity + (1 - 1 / R) * shock_velocity
        return res if res.ndim > 0 else res.item()

    @classmethod
    def _pre_shock_velocity(
        cls,
        shock_velocity: "_ArrayLike",
        downstream_velocity: "_ArrayLike",
        downstream_mach_number: "_ArrayLike",
        gamma: float = 5 / 3,
    ):
        r"""Upstream bulk velocity from downstream conditions.

        Inverts the post-shock velocity relation using :math:`R(\mathcal{M}_2)`:

        .. math::

            v_1 = R\,v_2 - (R - 1)\,v_{\rm sh}.

        Parameters
        ----------
        shock_velocity : float or ~numpy.ndarray
            Shock velocity :math:`v_{\rm sh}` in cm/s (lab frame).
        downstream_velocity : float or ~numpy.ndarray
            Downstream bulk velocity :math:`v_2` in cm/s (lab frame).
        downstream_mach_number : float or ~numpy.ndarray
            Downstream Mach number :math:`\mathcal{M}_2`.
        gamma : float, optional
            Adiabatic index. Default 5/3.

        Returns
        -------
        float or ~numpy.ndarray
            Upstream bulk velocity :math:`v_1` in cm/s (lab frame).
        """
        # Cast the shock velocity and downstream velocity to arrays for consistent broadcasting.
        shock_velocity = np.asarray(shock_velocity)
        downstream_velocity = np.asarray(downstream_velocity)
        downstream_mach_number = np.asarray(downstream_mach_number)

        # Compute the compression ratio from the downstream Mach number.
        R = cls._compression_ratio_from_downstream(downstream_mach_number, gamma)
        res = R * downstream_velocity - (R - 1) * shock_velocity

        return res if res.ndim > 0 else res.item()

    # --- Pressure --- #
    @classmethod
    def _post_shock_pressure(
        cls,
        shock_velocity: "_ArrayLike",
        upstream_density: "_ArrayLike",
        upstream_velocity: "_ArrayLike" = 0.0,
        upstream_pressure: "_ArrayLike" = None,
        upstream_temperature: "_ArrayLike" = None,
        mu: float = 0.61,
        gamma: float = 5 / 3,
    ):
        r"""Downstream pressure from the Rankine–Hugoniot momentum condition.

        .. math::

            P_2 = P_1 + \rho_1\,u_1^2\!\left(1 - \frac{1}{R}\right),
            \quad
            u_1 = v_{\rm sh} - v_1,
            \quad
            R = R(\mathcal{M}_1).

        The upstream Mach number is computed self-consistently from the
        thermodynamic state and flow velocity.

        Parameters
        ----------
        shock_velocity : float or ~numpy.ndarray
            Shock velocity :math:`v_{\rm sh}` in cm/s (lab frame).
        upstream_density : float or ~numpy.ndarray
            Upstream mass density :math:`\rho_1` in g/cm\ :sup:`3`.
        upstream_velocity : float or ~numpy.ndarray, optional
            Upstream bulk velocity :math:`v_1` in cm/s. Default 0.
        upstream_pressure : float or ~numpy.ndarray, optional
            Upstream pressure :math:`P_1` in dyne/cm\ :sup:`2`.
            Mutually exclusive with ``upstream_temperature``.
        upstream_temperature : float or ~numpy.ndarray, optional
            Upstream temperature :math:`T_1` in K.
            Mutually exclusive with ``upstream_pressure``.
        mu : float, optional
            Mean molecular weight. Default 0.61.
        gamma : float, optional
            Adiabatic index. Default 5/3.

        Returns
        -------
        float or ~numpy.ndarray
            Downstream pressure :math:`P_2` in dyne/cm\ :sup:`2`.

        Raises
        ------
        ValueError
            If neither or both of ``upstream_pressure`` and
            ``upstream_temperature`` are provided.
        """
        # --- Validate thermodynamic inputs --- #
        if (upstream_pressure is None) and (upstream_temperature is None):
            raise ValueError("Must provide either upstream_pressure or upstream_temperature.")
        if (upstream_pressure is not None) and (upstream_temperature is not None):
            raise ValueError("Provide only one of upstream_pressure or upstream_temperature.")

        # --- Coerce inputs --- #
        shock_velocity = np.asarray(shock_velocity)
        rho1 = np.asarray(upstream_density)
        v1 = np.asarray(upstream_velocity)

        # --- Resolve upstream pressure --- #
        if upstream_pressure is None:
            T1 = np.asarray(upstream_temperature)
            P1 = rho1 * k_B_cgs * T1 / (mu * m_p_cgs)
        else:
            P1 = np.asarray(upstream_pressure)

        # --- Shock-frame velocity --- #
        u1 = shock_velocity - v1

        # Extract the mach number and the compression ratio.
        M1 = cls._resolve_mach_number(
            shock_velocity=shock_velocity,
            flow_velocity=v1,
            temperature=upstream_temperature,
            pressure=P1,
            density=rho1,
            mu=mu,
            gamma=gamma,
        )
        R = cls._compression_ratio(M1, gamma)

        # Apply the RH conditions.
        P2 = P1 + rho1 * u1**2 * (1 - 1 / R)

        return P2 if P2.ndim > 0 else P2.item()

    @classmethod
    def _pre_shock_pressure(
        cls,
        shock_velocity: "_ArrayLike",
        downstream_density: "_ArrayLike",
        downstream_velocity: "_ArrayLike",
        downstream_pressure: "_ArrayLike" = None,
        downstream_temperature: "_ArrayLike" = None,
        mu: float = 0.61,
        gamma: float = 5 / 3,
    ):
        r"""Upstream pressure recovered from downstream conditions.

        Inverts the Rankine–Hugoniot momentum condition:

        .. math::

            P_1 = P_2 - \rho_1\,u_1^2\!\left(1 - \frac{1}{R}\right),
            \quad
            \rho_1 = \frac{\rho_2}{R},
            \quad
            u_1 = v_{\rm sh} - v_1.

        The compression ratio is computed from the downstream Mach number,
        which is resolved self-consistently from the thermodynamic state.

        Parameters
        ----------
        shock_velocity : float or ~numpy.ndarray
            Shock velocity :math:`v_{\rm sh}` in cm/s (lab frame).
        downstream_density : float or ~numpy.ndarray
            Downstream mass density :math:`\rho_2` in g/cm\ :sup:`3`.
        downstream_velocity : float or ~numpy.ndarray
            Downstream bulk velocity :math:`v_2` in cm/s (lab frame).
        downstream_pressure : float or ~numpy.ndarray, optional
            Downstream pressure :math:`P_2` in dyne/cm\ :sup:`2`.
            Mutually exclusive with ``downstream_temperature``.
        downstream_temperature : float or ~numpy.ndarray, optional
            Downstream temperature :math:`T_2` in K.
            Mutually exclusive with ``downstream_pressure``.
        mu : float, optional
            Mean molecular weight. Default 0.61.
        gamma : float, optional
            Adiabatic index. Default 5/3.

        Returns
        -------
        float or ~numpy.ndarray
            Upstream pressure :math:`P_1` in dyne/cm\ :sup:`2`.

        Raises
        ------
        ValueError
            If neither or both of ``downstream_pressure`` and
            ``downstream_temperature`` are provided.
        """
        if (downstream_pressure is None) and (downstream_temperature is None):
            raise ValueError("Must provide either downstream_pressure or downstream_temperature.")
        if (downstream_pressure is not None) and (downstream_temperature is not None):
            raise ValueError("Provide only one of downstream_pressure or downstream_temperature.")

        shock_velocity = np.asarray(shock_velocity)
        rho2 = np.asarray(downstream_density)
        v2 = np.asarray(downstream_velocity)

        if downstream_pressure is None:
            T2 = np.asarray(downstream_temperature)
            P2 = rho2 * k_B_cgs * T2 / (mu * m_p_cgs)
        else:
            P2 = np.asarray(downstream_pressure)

        # Resolve downstream Mach number self-consistently
        M2 = cls._resolve_mach_number(
            shock_velocity=shock_velocity,
            flow_velocity=v2,
            temperature=downstream_temperature,
            pressure=P2,
            density=rho2,
            mu=mu,
            gamma=gamma,
        )

        # Compression ratio
        R = cls._compression_ratio_from_downstream(M2, gamma)
        rho1 = rho2 / R

        # Upstream velocity
        v1 = cls._pre_shock_velocity(shock_velocity, v2, M2, gamma=gamma)

        # Shock-frame velocity
        u1 = shock_velocity - v1
        P1 = P2 - rho1 * u1**2 * (1 - 1 / R)

        return P1 if P1.ndim > 0 else P1.item()

    # --- Temperature --- #
    @classmethod
    def _post_shock_temperature(
        cls,
        shock_velocity: "_ArrayLike",
        upstream_density: "_ArrayLike",
        upstream_velocity: "_ArrayLike" = 0.0,
        upstream_pressure: "_ArrayLike" = None,
        upstream_temperature: "_ArrayLike" = None,
        mu: float = 0.61,
        gamma: float = 5 / 3,
    ):
        r"""Downstream temperature from the ideal gas relation.

        .. math::

            T_2 = \frac{\mu m_p}{k_B} \frac{P_2}{\rho_2}

        Parameters
        ----------
        shock_velocity : float or ~numpy.ndarray
        upstream_density : float or ~numpy.ndarray
        upstream_velocity : float or ~numpy.ndarray, optional
        upstream_pressure : float or ~numpy.ndarray, optional
        upstream_temperature : float or ~numpy.ndarray, optional
        mu : float, optional
        gamma : float, optional

        Returns
        -------
        float or ~numpy.ndarray
        """
        shock_velocity = np.asarray(shock_velocity)
        upstream_density = np.asarray(upstream_density)
        upstream_velocity = np.asarray(upstream_velocity)
        M1 = cls._resolve_mach_number(
            shock_velocity=shock_velocity,
            flow_velocity=upstream_velocity,
            temperature=upstream_temperature,
            pressure=upstream_pressure,
            density=upstream_density,
            mu=mu,
            gamma=gamma,
        )
        P2 = np.asarray(
            cls._post_shock_pressure(
                shock_velocity,
                upstream_density,
                upstream_velocity,
                upstream_pressure=upstream_pressure,
                upstream_temperature=upstream_temperature,
                mu=mu,
                gamma=gamma,
            )
        )
        rho2 = np.asarray(cls._post_shock_density(upstream_density, M1, gamma))
        T2 = (P2 / rho2) * (mu * m_p_cgs / k_B_cgs)
        return T2 if T2.ndim > 0 else T2.item()

    @classmethod
    def _pre_shock_temperature(
        cls,
        shock_velocity: "_ArrayLike",
        downstream_density: "_ArrayLike",
        downstream_velocity: "_ArrayLike",
        downstream_pressure: "_ArrayLike" = None,
        downstream_temperature: "_ArrayLike" = None,
        mu: float = 0.61,
        gamma: float = 5 / 3,
    ):
        r"""Upstream temperature: :math:`T_1 = \mu m_p P_1 / (k_B \rho_1)`.

        Parameters
        ----------
        shock_velocity : float or ~numpy.ndarray
            Shock velocity in cm/s.
        downstream_density : float or ~numpy.ndarray
            Downstream mass density :math:`\rho_2` in g/cm\ :sup:`3`.
        downstream_velocity : float or ~numpy.ndarray
            Downstream bulk velocity in cm/s.
        downstream_pressure : float or ~numpy.ndarray, optional
            Downstream pressure in dyne/cm\ :sup:`2`. Mutually exclusive with
            ``downstream_temperature``.
        downstream_temperature : float or ~numpy.ndarray, optional
            Downstream temperature in K. Mutually exclusive with ``downstream_pressure``.
        mu : float, optional
            Mean molecular weight. Default 0.61.
        gamma : float, optional
            Adiabatic index. Default 5/3.

        Returns
        -------
        float or ~numpy.ndarray
            Upstream temperature :math:`T_1` in K.
        """
        rho2 = np.asarray(downstream_density)
        v2 = np.asarray(downstream_velocity)
        # Resolve P2 so _resolve_mach_number can use it
        if downstream_pressure is None and downstream_temperature is None:
            raise ValueError("Must provide either downstream_pressure or downstream_temperature.")
        if downstream_pressure is None:
            P2_cgs = rho2 * k_B_cgs * np.asarray(downstream_temperature) / (mu * m_p_cgs)
        else:
            P2_cgs = np.asarray(downstream_pressure)
        M2 = cls._resolve_mach_number(
            np.asarray(shock_velocity),
            v2,
            pressure=P2_cgs,
            density=rho2,
            mu=mu,
            gamma=gamma,
        )
        P1 = cls._pre_shock_pressure(
            shock_velocity,
            rho2,
            v2,
            downstream_pressure=downstream_pressure,
            downstream_temperature=downstream_temperature,
            mu=mu,
            gamma=gamma,
        )
        rho1 = cls._pre_shock_density(rho2, M2, gamma)
        T1 = (P1 / rho1) * (mu * m_p_cgs / k_B_cgs)
        return T1 if np.ndim(T1) > 0 else T1.item()

    # --- Energy Density --- #
    @classmethod
    def _post_shock_energy_density(
        cls,
        shock_velocity: "_ArrayLike",
        upstream_density: "_ArrayLike",
        upstream_velocity: "_ArrayLike" = 0.0,
        upstream_pressure: "_ArrayLike" = None,
        upstream_temperature: "_ArrayLike" = None,
        mu: float = 0.61,
        gamma: float = 5 / 3,
    ):
        r"""Downstream internal energy density :math:`e_2 = P_2 / (\gamma - 1)`.

        Parameters
        ----------
        shock_velocity : float or ~numpy.ndarray
            Shock velocity in cm/s.
        upstream_density : float or ~numpy.ndarray
            Upstream mass density :math:`\rho_1` in g/cm\ :sup:`3`.
        upstream_velocity : float or ~numpy.ndarray, optional
            Upstream bulk velocity in cm/s. Default 0.
        upstream_pressure : float or ~numpy.ndarray, optional
            Upstream pressure in dyne/cm\ :sup:`2`. Mutually exclusive with
            ``upstream_temperature``.
        upstream_temperature : float or ~numpy.ndarray, optional
            Upstream temperature in K. Mutually exclusive with ``upstream_pressure``.
        mu : float, optional
            Mean molecular weight. Default 0.61.
        gamma : float, optional
            Adiabatic index. Default 5/3.

        Returns
        -------
        float or ~numpy.ndarray
            Downstream internal energy density :math:`e_2` in erg/cm\ :sup:`3`.
        """
        P2 = np.asarray(
            cls._post_shock_pressure(
                shock_velocity,
                upstream_density,
                upstream_velocity,
                upstream_pressure=upstream_pressure,
                upstream_temperature=upstream_temperature,
                mu=mu,
                gamma=gamma,
            )
        )
        e2 = P2 / (gamma - 1)
        return e2 if e2.ndim > 0 else e2.item()

    @classmethod
    def _pre_shock_energy_density(
        cls,
        shock_velocity: "_ArrayLike",
        downstream_density: "_ArrayLike",
        downstream_velocity: "_ArrayLike",
        downstream_pressure: "_ArrayLike" = None,
        downstream_temperature: "_ArrayLike" = None,
        mu: float = 0.61,
        gamma: float = 5 / 3,
    ):
        r"""Upstream internal energy density :math:`e_1 = P_1 / (\gamma - 1)`.

        Parameters
        ----------
        shock_velocity : float or ~numpy.ndarray
            Shock velocity in cm/s.
        downstream_density : float or ~numpy.ndarray
            Downstream mass density :math:`\rho_2` in g/cm\ :sup:`3`.
        downstream_velocity : float or ~numpy.ndarray
            Downstream bulk velocity in cm/s.
        downstream_pressure : float or ~numpy.ndarray, optional
            Downstream pressure in dyne/cm\ :sup:`2`. Mutually exclusive with
            ``downstream_temperature``.
        downstream_temperature : float or ~numpy.ndarray, optional
            Downstream temperature in K. Mutually exclusive with ``downstream_pressure``.
        mu : float, optional
            Mean molecular weight. Default 0.61.
        gamma : float, optional
            Adiabatic index. Default 5/3.

        Returns
        -------
        float or ~numpy.ndarray
            Upstream internal energy density :math:`e_1` in erg/cm\ :sup:`3`.
        """
        P1 = np.asarray(
            cls._pre_shock_pressure(
                shock_velocity,
                downstream_density,
                downstream_velocity,
                downstream_pressure=downstream_pressure,
                downstream_temperature=downstream_temperature,
                mu=mu,
                gamma=gamma,
            )
        )
        e1 = P1 / (gamma - 1)
        return e1 if e1.ndim > 0 else e1.item()

    # ---------------------------------------------- #
    # Public API — kinematic (explicit Mach)         #
    # ---------------------------------------------- #
    @classmethod
    def compute_compression_ratio(cls, upstream_mach_number: float, gamma: float = 5 / 3) -> float:
        r"""Compression ratio :math:`R(\mathcal{M}_1) = (\gamma+1)\mathcal{M}_1^2 / [(\gamma-1)\mathcal{M}_1^2 + 2]`.

        Parameters
        ----------
        upstream_mach_number : float or ~numpy.ndarray
            Upstream Mach number :math:`\mathcal{M}_1`.
        gamma : float, optional
            Adiabatic index. Default 5/3.

        Returns
        -------
        float or ~numpy.ndarray
        """
        return cls._compression_ratio(np.asarray(upstream_mach_number), gamma)

    @classmethod
    def compute_post_shock_mach_number(cls, upstream_mach_number: float, gamma: float = 5 / 3) -> float:
        r"""Downstream Mach number from the upstream Mach number.

        Parameters
        ----------
        upstream_mach_number : float or ~numpy.ndarray
            Upstream Mach number :math:`\mathcal{M}_1 > 1`.
        gamma : float, optional
            Adiabatic index. Default 5/3.

        Returns
        -------
        float or ~numpy.ndarray
            Downstream Mach number :math:`\mathcal{M}_2 \in (0, 1)`.
        """
        return cls._post_shock_mach_from_upstream(np.asarray(upstream_mach_number), gamma)

    @classmethod
    def compute_pre_shock_mach_number(cls, downstream_mach_number: float, gamma: float = 5 / 3) -> float:
        r"""Upstream Mach number from the downstream Mach number.

        Parameters
        ----------
        downstream_mach_number : float or ~numpy.ndarray
            Downstream Mach number :math:`\mathcal{M}_2 \in (0, 1)`.
        gamma : float, optional
            Adiabatic index. Default 5/3.

        Returns
        -------
        float or ~numpy.ndarray
            Upstream Mach number :math:`\mathcal{M}_1 > 1`.
        """
        return cls._pre_shock_mach_from_downstream(np.asarray(downstream_mach_number), gamma)

    @classmethod
    def compute_post_shock_density(
        cls, upstream_density: "_UnitBearingArrayLike", upstream_mach_number: float, gamma: float = 5 / 3
    ) -> u.Quantity:
        r"""Post-shock mass density :math:`\rho_2 = R(\mathcal{M}_1)\,\rho_1`.

        Parameters
        ----------
        upstream_density : float or ~astropy.units.Quantity
            Upstream mass density :math:`\rho_1`. Bare float assumed g/cm\ :sup:`3`.
        upstream_mach_number : float or ~numpy.ndarray
            Upstream Mach number :math:`\mathcal{M}_1`.
        gamma : float, optional
            Adiabatic index. Default 5/3.

        Returns
        -------
        ~astropy.units.Quantity
            Post-shock density :math:`\rho_2` in g/cm\ :sup:`3`.
        """
        rho1 = ensure_in_units(upstream_density, u.g / u.cm**3)
        return cls._post_shock_density(rho1, np.asarray(upstream_mach_number), gamma) * u.g / u.cm**3

    @classmethod
    def compute_post_shock_number_density(
        cls,
        upstream_density: "_UnitBearingArrayLike",
        upstream_mach_number: float,
        gamma: float = 5 / 3,
        mu: float = 0.61,
    ) -> u.Quantity:
        r"""Post-shock number density :math:`n_2 = \rho_2 / (\mu m_p)`.

        Parameters
        ----------
        upstream_density : float or ~astropy.units.Quantity
            Upstream mass density. Bare float assumed g/cm\ :sup:`3`.
        upstream_mach_number : float or ~numpy.ndarray
            Upstream Mach number :math:`\mathcal{M}_1`.
        gamma : float, optional
            Adiabatic index. Default 5/3.
        mu : float, optional
            Mean molecular weight. Default 0.61.

        Returns
        -------
        ~astropy.units.Quantity
            Post-shock number density in cm\ :sup:`-3`.
        """
        rho1 = ensure_in_units(upstream_density, u.g / u.cm**3)
        return cls._post_shock_number_density(rho1, np.asarray(upstream_mach_number), gamma, mu) * u.cm**-3

    @classmethod
    def compute_pre_shock_density(
        cls, downstream_density: "_UnitBearingArrayLike", downstream_mach_number: float, gamma: float = 5 / 3
    ) -> u.Quantity:
        r"""Pre-shock mass density :math:`\rho_1 = \rho_2 / R(\mathcal{M}_2)`.

        Parameters
        ----------
        downstream_density : float or ~astropy.units.Quantity
            Downstream mass density :math:`\rho_2`. Bare float assumed g/cm\ :sup:`3`.
        downstream_mach_number : float or ~numpy.ndarray
            Downstream Mach number :math:`\mathcal{M}_2`.
        gamma : float, optional
            Adiabatic index. Default 5/3.

        Returns
        -------
        ~astropy.units.Quantity
            Pre-shock density :math:`\rho_1` in g/cm\ :sup:`3`.
        """
        rho2 = ensure_in_units(downstream_density, u.g / u.cm**3)
        return cls._pre_shock_density(rho2, np.asarray(downstream_mach_number), gamma) * u.g / u.cm**3

    @classmethod
    def compute_pre_shock_number_density(
        cls,
        downstream_density: "_UnitBearingArrayLike",
        downstream_mach_number: float,
        gamma: float = 5 / 3,
        mu: float = 0.61,
    ) -> u.Quantity:
        r"""Pre-shock number density :math:`n_1 = \rho_1 / (\mu m_p)`.

        Parameters
        ----------
        downstream_density : float or ~astropy.units.Quantity
            Downstream mass density. Bare float assumed g/cm\ :sup:`3`.
        downstream_mach_number : float or ~numpy.ndarray
            Downstream Mach number :math:`\mathcal{M}_2`.
        gamma : float, optional
            Adiabatic index. Default 5/3.
        mu : float, optional
            Mean molecular weight. Default 0.61.

        Returns
        -------
        ~astropy.units.Quantity
            Pre-shock number density in cm\ :sup:`-3`.
        """
        rho2 = ensure_in_units(downstream_density, u.g / u.cm**3)
        return cls._pre_shock_number_density(rho2, np.asarray(downstream_mach_number), gamma, mu) * u.cm**-3

    @classmethod
    def compute_post_shock_velocity(
        cls,
        shock_velocity: "_UnitBearingArrayLike",
        upstream_mach_number: float,
        upstream_velocity: "_UnitBearingArrayLike" = 0.0 * u.cm / u.s,
        gamma: float = 5 / 3,
    ) -> u.Quantity:
        r"""Post-shock bulk velocity in the lab frame.

        Parameters
        ----------
        shock_velocity : float or ~astropy.units.Quantity
            Shock velocity :math:`v_{\rm sh}`. Bare float assumed cm/s.
        upstream_mach_number : float or ~numpy.ndarray
            Upstream Mach number :math:`\mathcal{M}_1`.
        upstream_velocity : float or ~astropy.units.Quantity, optional
            Upstream bulk velocity :math:`v_1`. Bare float assumed cm/s. Default 0.
        gamma : float, optional
            Adiabatic index. Default 5/3.

        Returns
        -------
        ~astropy.units.Quantity
            Post-shock bulk velocity :math:`v_2` in cm/s.
        """
        v_sh = ensure_in_units(shock_velocity, u.cm / u.s)
        v_1 = ensure_in_units(upstream_velocity, u.cm / u.s)
        return cls._post_shock_velocity(v_sh, np.asarray(upstream_mach_number), v_1, gamma) * u.cm / u.s

    @classmethod
    def compute_pre_shock_velocity(
        cls,
        shock_velocity: "_UnitBearingArrayLike",
        downstream_velocity: "_UnitBearingArrayLike",
        downstream_mach_number: float,
        gamma: float = 5 / 3,
    ) -> u.Quantity:
        r"""Pre-shock bulk velocity from downstream conditions.

        Parameters
        ----------
        shock_velocity : float or ~astropy.units.Quantity
            Shock velocity :math:`v_{\rm sh}`. Bare float assumed cm/s.
        downstream_velocity : float or ~astropy.units.Quantity
            Downstream bulk velocity :math:`v_2`. Bare float assumed cm/s.
        downstream_mach_number : float or ~numpy.ndarray
            Downstream Mach number :math:`\mathcal{M}_2`.
        gamma : float, optional
            Adiabatic index. Default 5/3.

        Returns
        -------
        ~astropy.units.Quantity
            Pre-shock bulk velocity :math:`v_1` in cm/s.
        """
        v_sh = ensure_in_units(shock_velocity, u.cm / u.s)
        v_2 = ensure_in_units(downstream_velocity, u.cm / u.s)
        return cls._pre_shock_velocity(v_sh, v_2, np.asarray(downstream_mach_number), gamma) * u.cm / u.s

    # ---------------------------------------------- #
    # Public API — thermodynamic (Mach implicit)     #
    # ---------------------------------------------- #

    @classmethod
    def compute_post_shock_pressure(
        cls,
        shock_velocity: "_UnitBearingArrayLike",
        upstream_density: "_UnitBearingArrayLike",
        upstream_velocity: "_UnitBearingArrayLike" = 0.0 * u.cm / u.s,
        upstream_pressure: "_UnitBearingArrayLike" = None,
        upstream_temperature: "_UnitBearingArrayLike" = None,
        mu: float = 0.61,
        gamma: float = 5 / 3,
    ) -> u.Quantity:
        r"""Post-shock gas pressure.

        The upstream Mach number is computed self-consistently from the provided
        thermodynamic state.

        Parameters
        ----------
        shock_velocity : float or ~astropy.units.Quantity
            Shock velocity :math:`v_{\rm sh}`. Bare float assumed cm/s.
        upstream_density : float or ~astropy.units.Quantity
            Upstream mass density :math:`\rho_1`. Bare float assumed g/cm\ :sup:`3`.
        upstream_velocity : float or ~astropy.units.Quantity, optional
            Upstream bulk velocity :math:`v_1`. Bare float assumed cm/s. Default 0.
        upstream_pressure : float or ~astropy.units.Quantity, optional
            Upstream pressure :math:`P_1` in dyne/cm\ :sup:`2`. Mutually exclusive
            with ``upstream_temperature``.
        upstream_temperature : float or ~astropy.units.Quantity, optional
            Upstream temperature :math:`T_1` in K. Mutually exclusive with
            ``upstream_pressure``.
        mu : float, optional
            Mean molecular weight. Default 0.61.
        gamma : float, optional
            Adiabatic index. Default 5/3.

        Returns
        -------
        ~astropy.units.Quantity
            Post-shock pressure :math:`P_2` in dyne/cm\ :sup:`2`.

        Raises
        ------
        ValueError
            If neither or both of ``upstream_pressure`` and ``upstream_temperature`` are given.
        """
        v_sh = ensure_in_units(shock_velocity, u.cm / u.s)
        rho1 = ensure_in_units(upstream_density, u.g / u.cm**3)
        v_1 = ensure_in_units(upstream_velocity, u.cm / u.s)
        P1 = ensure_in_units(upstream_pressure, u.dyn / u.cm**2) if upstream_pressure is not None else None
        T1 = ensure_in_units(upstream_temperature, u.K) if upstream_temperature is not None else None
        return (
            cls._post_shock_pressure(v_sh, rho1, v_1, upstream_pressure=P1, upstream_temperature=T1, mu=mu, gamma=gamma)
            * u.dyn
            / u.cm**2
        )

    @classmethod
    def compute_pre_shock_pressure(
        cls,
        shock_velocity: "_UnitBearingArrayLike",
        downstream_density: "_UnitBearingArrayLike",
        downstream_velocity: "_UnitBearingArrayLike",
        downstream_pressure: "_UnitBearingArrayLike" = None,
        downstream_temperature: "_UnitBearingArrayLike" = None,
        mu: float = 0.61,
        gamma: float = 5 / 3,
    ) -> u.Quantity:
        r"""Pre-shock gas pressure recovered from downstream conditions.

        The downstream Mach number is computed self-consistently from the provided
        thermodynamic state.

        Parameters
        ----------
        shock_velocity : float or ~astropy.units.Quantity
            Shock velocity :math:`v_{\rm sh}`. Bare float assumed cm/s.
        downstream_density : float or ~astropy.units.Quantity
            Downstream mass density :math:`\rho_2`. Bare float assumed g/cm\ :sup:`3`.
        downstream_velocity : float or ~astropy.units.Quantity
            Downstream bulk velocity :math:`v_2`. Bare float assumed cm/s.
        downstream_pressure : float or ~astropy.units.Quantity, optional
            Downstream pressure :math:`P_2`. Bare float assumed dyne/cm\ :sup:`2`.
            Mutually exclusive with ``downstream_temperature``.
        downstream_temperature : float or ~astropy.units.Quantity, optional
            Downstream temperature :math:`T_2` in K. Mutually exclusive with
            ``downstream_pressure``.
        mu : float, optional
            Mean molecular weight. Default 0.61.
        gamma : float, optional
            Adiabatic index. Default 5/3.

        Returns
        -------
        ~astropy.units.Quantity
            Pre-shock pressure :math:`P_1` in dyne/cm\ :sup:`2`.
        """
        v_sh = ensure_in_units(shock_velocity, u.cm / u.s)
        rho2 = ensure_in_units(downstream_density, u.g / u.cm**3)
        v_2 = ensure_in_units(downstream_velocity, u.cm / u.s)
        P2 = ensure_in_units(downstream_pressure, u.dyn / u.cm**2) if downstream_pressure is not None else None
        T2 = ensure_in_units(downstream_temperature, u.K) if downstream_temperature is not None else None
        return (
            cls._pre_shock_pressure(
                v_sh, rho2, v_2, downstream_pressure=P2, downstream_temperature=T2, mu=mu, gamma=gamma
            )
            * u.dyn
            / u.cm**2
        )

    @classmethod
    def compute_post_shock_temperature(
        cls,
        shock_velocity: "_UnitBearingArrayLike",
        upstream_density: "_UnitBearingArrayLike",
        upstream_velocity: "_UnitBearingArrayLike" = 0.0 * u.cm / u.s,
        upstream_pressure: "_UnitBearingArrayLike" = None,
        upstream_temperature: "_UnitBearingArrayLike" = None,
        mu: float = 0.61,
        gamma: float = 5 / 3,
    ) -> u.Quantity:
        r"""Post-shock temperature from the ideal gas relation.

        Parameters
        ----------
        shock_velocity : float or ~astropy.units.Quantity
            Shock velocity :math:`v_{\rm sh}`. Bare float assumed cm/s.
        upstream_density : float or ~astropy.units.Quantity
            Upstream mass density :math:`\rho_1`. Bare float assumed g/cm\ :sup:`3`.
        upstream_velocity : float or ~astropy.units.Quantity, optional
            Upstream bulk velocity. Bare float assumed cm/s. Default 0.
        upstream_pressure : float or ~astropy.units.Quantity, optional
            Upstream pressure in dyne/cm\ :sup:`2`. Mutually exclusive with
            ``upstream_temperature``.
        upstream_temperature : float or ~astropy.units.Quantity, optional
            Upstream temperature in K. Mutually exclusive with ``upstream_pressure``.
        mu : float, optional
            Mean molecular weight. Default 0.61.
        gamma : float, optional
            Adiabatic index. Default 5/3.

        Returns
        -------
        ~astropy.units.Quantity
            Post-shock temperature :math:`T_2` in K.
        """
        v_sh = ensure_in_units(shock_velocity, u.cm / u.s)
        rho1 = ensure_in_units(upstream_density, u.g / u.cm**3)
        v_1 = ensure_in_units(upstream_velocity, u.cm / u.s)
        P1 = ensure_in_units(upstream_pressure, u.dyn / u.cm**2) if upstream_pressure is not None else None
        T1 = ensure_in_units(upstream_temperature, u.K) if upstream_temperature is not None else None
        return (
            cls._post_shock_temperature(
                v_sh, rho1, v_1, upstream_pressure=P1, upstream_temperature=T1, mu=mu, gamma=gamma
            )
            * u.K
        )

    @classmethod
    def compute_pre_shock_temperature(
        cls,
        shock_velocity: "_UnitBearingArrayLike",
        downstream_density: "_UnitBearingArrayLike",
        downstream_velocity: "_UnitBearingArrayLike",
        downstream_pressure: "_UnitBearingArrayLike" = None,
        downstream_temperature: "_UnitBearingArrayLike" = None,
        mu: float = 0.61,
        gamma: float = 5 / 3,
    ) -> u.Quantity:
        r"""Pre-shock temperature from downstream conditions.

        Parameters
        ----------
        shock_velocity : float or ~astropy.units.Quantity
            Shock velocity :math:`v_{\rm sh}`. Bare float assumed cm/s.
        downstream_density : float or ~astropy.units.Quantity
            Downstream mass density :math:`\rho_2`. Bare float assumed g/cm\ :sup:`3`.
        downstream_velocity : float or ~astropy.units.Quantity
            Downstream bulk velocity. Bare float assumed cm/s.
        downstream_pressure : float or ~astropy.units.Quantity, optional
            Downstream pressure in dyne/cm\ :sup:`2`. Mutually exclusive with
            ``downstream_temperature``.
        downstream_temperature : float or ~astropy.units.Quantity, optional
            Downstream temperature in K. Mutually exclusive with
            ``downstream_pressure``.
        mu : float, optional
            Mean molecular weight. Default 0.61.
        gamma : float, optional
            Adiabatic index. Default 5/3.

        Returns
        -------
        ~astropy.units.Quantity
            Pre-shock temperature :math:`T_1` in K.
        """
        v_sh = ensure_in_units(shock_velocity, u.cm / u.s)
        rho2 = ensure_in_units(downstream_density, u.g / u.cm**3)
        v_2 = ensure_in_units(downstream_velocity, u.cm / u.s)
        P2 = ensure_in_units(downstream_pressure, u.dyn / u.cm**2) if downstream_pressure is not None else None
        T2 = ensure_in_units(downstream_temperature, u.K) if downstream_temperature is not None else None
        return (
            cls._pre_shock_temperature(
                v_sh, rho2, v_2, downstream_pressure=P2, downstream_temperature=T2, mu=mu, gamma=gamma
            )
            * u.K
        )

    @classmethod
    def compute_post_shock_energy_density(
        cls,
        shock_velocity: "_UnitBearingArrayLike",
        upstream_density: "_UnitBearingArrayLike",
        upstream_velocity: "_UnitBearingArrayLike" = 0.0 * u.cm / u.s,
        upstream_pressure: "_UnitBearingArrayLike" = None,
        upstream_temperature: "_UnitBearingArrayLike" = None,
        mu: float = 0.61,
        gamma: float = 5 / 3,
    ) -> u.Quantity:
        r"""Post-shock internal energy density :math:`e_2 = P_2 / (\gamma - 1)`.

        Parameters
        ----------
        shock_velocity : float or ~astropy.units.Quantity
            Shock velocity. Bare float assumed cm/s.
        upstream_density : float or ~astropy.units.Quantity
            Upstream mass density. Bare float assumed g/cm\ :sup:`3`.
        upstream_velocity : float or ~astropy.units.Quantity, optional
            Upstream bulk velocity. Bare float assumed cm/s. Default 0.
        upstream_pressure : float or ~astropy.units.Quantity, optional
            Upstream pressure in dyne/cm\ :sup:`2`. Mutually exclusive with
            ``upstream_temperature``.
        upstream_temperature : float or ~astropy.units.Quantity, optional
            Upstream temperature in K. Mutually exclusive with ``upstream_pressure``.
        mu : float, optional
            Mean molecular weight. Default 0.61.
        gamma : float, optional
            Adiabatic index. Default 5/3.

        Returns
        -------
        ~astropy.units.Quantity
            Post-shock internal energy density :math:`e_2` in erg/cm\ :sup:`3`.
        """
        v_sh = ensure_in_units(shock_velocity, u.cm / u.s)
        rho1 = ensure_in_units(upstream_density, u.g / u.cm**3)
        v_1 = ensure_in_units(upstream_velocity, u.cm / u.s)
        P1 = ensure_in_units(upstream_pressure, u.dyn / u.cm**2) if upstream_pressure is not None else None
        T1 = ensure_in_units(upstream_temperature, u.K) if upstream_temperature is not None else None
        return (
            cls._post_shock_energy_density(
                v_sh, rho1, v_1, upstream_pressure=P1, upstream_temperature=T1, mu=mu, gamma=gamma
            )
            * u.erg
            / u.cm**3
        )

    @classmethod
    def compute_pre_shock_energy_density(
        cls,
        shock_velocity: "_UnitBearingArrayLike",
        downstream_density: "_UnitBearingArrayLike",
        downstream_velocity: "_UnitBearingArrayLike",
        downstream_pressure: "_UnitBearingArrayLike" = None,
        downstream_temperature: "_UnitBearingArrayLike" = None,
        mu: float = 0.61,
        gamma: float = 5 / 3,
    ) -> u.Quantity:
        r"""Pre-shock internal energy density :math:`e_1 = P_1 / (\gamma - 1)`.

        Parameters
        ----------
        shock_velocity : float or ~astropy.units.Quantity
            Shock velocity. Bare float assumed cm/s.
        downstream_density : float or ~astropy.units.Quantity
            Downstream mass density. Bare float assumed g/cm\ :sup:`3`.
        downstream_velocity : float or ~astropy.units.Quantity
            Downstream bulk velocity. Bare float assumed cm/s.
        downstream_pressure : float or ~astropy.units.Quantity, optional
            Downstream pressure in dyne/cm\ :sup:`2`. Mutually exclusive with
            ``downstream_temperature``.
        downstream_temperature : float or ~astropy.units.Quantity, optional
            Downstream temperature in K. Mutually exclusive with
            ``downstream_pressure``.
        mu : float, optional
            Mean molecular weight. Default 0.61.
        gamma : float, optional
            Adiabatic index. Default 5/3.

        Returns
        -------
        ~astropy.units.Quantity
            Pre-shock internal energy density :math:`e_1` in erg/cm\ :sup:`3`.
        """
        v_sh = ensure_in_units(shock_velocity, u.cm / u.s)
        rho2 = ensure_in_units(downstream_density, u.g / u.cm**3)
        v_2 = ensure_in_units(downstream_velocity, u.cm / u.s)
        P2 = ensure_in_units(downstream_pressure, u.dyn / u.cm**2) if downstream_pressure is not None else None
        T2 = ensure_in_units(downstream_temperature, u.K) if downstream_temperature is not None else None
        return (
            cls._pre_shock_energy_density(
                v_sh, rho2, v_2, downstream_pressure=P2, downstream_temperature=T2, mu=mu, gamma=gamma
            )
            * u.erg
            / u.cm**3
        )

    # ---------------------------------------------- #
    # Aggregate backend + public aggregate           #
    # ---------------------------------------------- #
    @classmethod
    def _solve(
        cls,
        shock_velocity,
        flow_density,
        flow_velocity=0.0,
        flow_pressure=None,
        flow_temperature=None,
        mu: float = 0.61,
        gamma: float = 5 / 3,
        upstream: bool = False,
    ) -> dict:
        r"""CGS aggregate backend.

        The Mach number on the relevant side is computed from the shock-frame velocity
        and the thermodynamic state using :meth:`_resolve_mach_number`. It is then
        used for density and velocity computations; pressure, temperature, and energy
        density methods compute the Mach number independently for self-consistency.

        Parameters
        ----------
        shock_velocity : float or ~numpy.ndarray
            Shock front velocity in cm/s.
        flow_density : float or ~numpy.ndarray
            Upstream density (``upstream=False``) or downstream density
            (``upstream=True``) in g/cm\ :sup:`3`.
        flow_velocity : float or ~numpy.ndarray, optional
            Upstream or downstream bulk velocity in cm/s. Default 0.
        flow_pressure : float or ~numpy.ndarray, optional
            Upstream or downstream pressure in dyne/cm\ :sup:`2`.
            Mutually exclusive with ``flow_temperature``.
        flow_temperature : float or ~numpy.ndarray, optional
            Upstream or downstream temperature in K.
            Mutually exclusive with ``flow_pressure``.
        mu : float, optional
            Mean molecular weight. Default 0.61.
        gamma : float, optional
            Adiabatic index. Default 5/3.
        upstream : bool, optional
            If ``False`` (default), inputs are upstream conditions and post-shock
            quantities are returned. If ``True``, inputs are downstream conditions
            and pre-shock quantities are returned.

        Returns
        -------
        dict
            Plain CGS values.
        """
        flow_mach = cls._resolve_mach_number(
            shock_velocity,
            flow_velocity,
            temperature=flow_temperature,
            pressure=flow_pressure,
            density=flow_density,
            mu=mu,
            gamma=gamma,
        )
        if not upstream:
            M1 = flow_mach
            return {
                "compression_ratio": cls._compression_ratio(M1, gamma),
                "post_shock_density": cls._post_shock_density(flow_density, M1, gamma),
                "post_shock_number_density": cls._post_shock_number_density(flow_density, M1, gamma, mu),
                "post_shock_velocity": cls._post_shock_velocity(shock_velocity, M1, flow_velocity, gamma),
                "post_shock_pressure": cls._post_shock_pressure(
                    shock_velocity,
                    flow_density,
                    flow_velocity,
                    upstream_pressure=flow_pressure,
                    upstream_temperature=flow_temperature,
                    mu=mu,
                    gamma=gamma,
                ),
                "post_shock_temperature": cls._post_shock_temperature(
                    shock_velocity,
                    flow_density,
                    flow_velocity,
                    upstream_pressure=flow_pressure,
                    upstream_temperature=flow_temperature,
                    mu=mu,
                    gamma=gamma,
                ),
                "post_shock_energy_density": cls._post_shock_energy_density(
                    shock_velocity,
                    flow_density,
                    flow_velocity,
                    upstream_pressure=flow_pressure,
                    upstream_temperature=flow_temperature,
                    mu=mu,
                    gamma=gamma,
                ),
            }
        else:
            M2 = flow_mach
            return {
                "compression_ratio": cls._compression_ratio_from_downstream(M2, gamma),
                "pre_shock_density": cls._pre_shock_density(flow_density, M2, gamma),
                "pre_shock_number_density": cls._pre_shock_number_density(flow_density, M2, gamma, mu),
                "pre_shock_velocity": cls._pre_shock_velocity(shock_velocity, flow_velocity, M2, gamma),
                "pre_shock_pressure": cls._pre_shock_pressure(
                    shock_velocity,
                    flow_density,
                    flow_velocity,
                    downstream_pressure=flow_pressure,
                    downstream_temperature=flow_temperature,
                    mu=mu,
                    gamma=gamma,
                ),
                "pre_shock_temperature": cls._pre_shock_temperature(
                    shock_velocity,
                    flow_density,
                    flow_velocity,
                    downstream_pressure=flow_pressure,
                    downstream_temperature=flow_temperature,
                    mu=mu,
                    gamma=gamma,
                ),
                "pre_shock_energy_density": cls._pre_shock_energy_density(
                    shock_velocity,
                    flow_density,
                    flow_velocity,
                    downstream_pressure=flow_pressure,
                    downstream_temperature=flow_temperature,
                    mu=mu,
                    gamma=gamma,
                ),
            }

    @classmethod
    def solve(
        cls,
        shock_velocity: "_UnitBearingArrayLike",
        flow_density: "_UnitBearingArrayLike",
        flow_velocity: "_UnitBearingArrayLike" = 0.0 * u.cm / u.s,
        flow_pressure: "_UnitBearingArrayLike" = None,
        flow_temperature: "_UnitBearingArrayLike" = None,
        mu: float = 0.61,
        gamma: float = 5 / 3,
        upstream: bool = False,
    ):
        r"""Compute all jump-condition quantities in one call.

        The Mach number on the flow side is derived from the shock-frame velocity
        and the thermodynamic state. At least one of ``flow_pressure`` or
        ``flow_temperature`` must be provided.

        Parameters
        ----------
        shock_velocity : float or ~astropy.units.Quantity
            Shock front velocity. Bare float assumed cm/s.
        flow_density : float or ~astropy.units.Quantity
            Upstream density (``upstream=False``) or downstream density
            (``upstream=True``). Bare float assumed g/cm\ :sup:`3`.
        flow_velocity : float or ~astropy.units.Quantity, optional
            Upstream or downstream bulk velocity. Bare float assumed cm/s. Default 0.
        flow_pressure : float or ~astropy.units.Quantity, optional
            Upstream or downstream pressure. Bare float assumed dyne/cm\ :sup:`2`.
            Mutually exclusive with ``flow_temperature``.
        flow_temperature : float or ~astropy.units.Quantity, optional
            Upstream or downstream temperature in K.
            Mutually exclusive with ``flow_pressure``.
        mu : float, optional
            Mean molecular weight. Default 0.61.
        gamma : float, optional
            Adiabatic index. Default 5/3.
        upstream : bool, optional
            If ``False`` (default), return a :class:`WeakShockConditionsResult`
            with post-shock quantities. If ``True``, return a
            :class:`WeakShockConditionsUpstreamResult` with pre-shock quantities.

        Returns
        -------
        WeakShockConditionsResult or WeakShockConditionsUpstreamResult
        """
        v_sh = ensure_in_units(shock_velocity, u.cm / u.s)
        rho = ensure_in_units(flow_density, u.g / u.cm**3)
        v = ensure_in_units(flow_velocity, u.cm / u.s)
        P = ensure_in_units(flow_pressure, u.dyn / u.cm**2) if flow_pressure is not None else None
        T = ensure_in_units(flow_temperature, u.K) if flow_temperature is not None else None
        result = cls._solve(v_sh, rho, v, P, T, mu, gamma, upstream=upstream)
        return cls._make_upstream_result(result) if upstream else cls._make_result(result)


class WeakColdShockConditions(WeakShockConditions):
    r"""Rankine-Hugoniot conditions for a general normal shock into a *cold* upstream medium.

    This is the weak-shock analogue of :class:`StrongColdShockConditions`. "Cold upstream"
    means the upstream thermal pressure is negligible (:math:`P_1 \approx 0`,
    :math:`T_1 \approx 0`), so all post-shock thermodynamic quantities depend only on
    :math:`v_{\rm sh}`, :math:`\rho_1`, the upstream Mach number :math:`\mathcal{M}_1`,
    and the adiabatic index :math:`\gamma`. No upstream pressure or temperature is required.

    Because the cold-limit Mach number cannot be inferred from thermodynamics, it must be
    supplied explicitly. The post-shock temperature is density-independent in the cold limit:

    .. math::

        T_2 = \frac{\mu m_p}{k_B}\,\frac{R-1}{R^2}\,u_1^2,
        \quad R = R(\mathcal{M}_1).

    Pre-shock pressure, temperature, and energy density are not defined in this limit
    and are not part of the public API.
    """

    OUTPUT_FIELDS = (
        ("compression_ratio", None),
        ("post_shock_density", u.g / u.cm**3),
        ("post_shock_number_density", u.cm**-3),
        ("post_shock_velocity", u.cm / u.s),
        ("post_shock_pressure", u.dyn / u.cm**2),
        ("post_shock_temperature", u.K),
        ("post_shock_thermal_energy_density", u.erg / u.cm**3),
    )

    UPSTREAM_OUTPUT_FIELDS = (
        ("compression_ratio", None),
        ("pre_shock_density", u.g / u.cm**3),
        ("pre_shock_number_density", u.cm**-3),
        ("pre_shock_velocity", u.cm / u.s),
    )

    # ---------------------------------------------- #
    # Private CGS helpers (cold-limit overrides)     #
    # ---------------------------------------------- #

    @classmethod
    def _post_shock_pressure(
        cls,
        shock_velocity: "_ArrayLike",
        upstream_density: "_ArrayLike",
        upstream_mach_number: "_ArrayLike",
        upstream_velocity: "_ArrayLike" = 0.0,
        gamma: float = 5 / 3,
    ):
        r"""Post-shock pressure in the cold-upstream limit: :math:`P_2 = \rho_1 u_1^2 (1 - 1/R(\mathcal{M}_1))`.

        Parameters
        ----------
        shock_velocity : float or ~numpy.ndarray
            Shock velocity in cm/s.
        upstream_density : float or ~numpy.ndarray
            Upstream mass density :math:`\rho_1` in g/cm\ :sup:`3`.
        upstream_mach_number : float or ~numpy.ndarray
            Upstream Mach number :math:`\mathcal{M}_1`.
        upstream_velocity : float or ~numpy.ndarray, optional
            Upstream bulk velocity in cm/s. Default 0.
        gamma : float, optional
            Adiabatic index. Default 5/3.

        Returns
        -------
        float or ~numpy.ndarray
            Post-shock pressure :math:`P_2` in dyne/cm\ :sup:`2`.
        """
        shock_velocity = np.asarray(shock_velocity)
        upstream_density = np.asarray(upstream_density)
        upstream_velocity = np.asarray(upstream_velocity)
        R = cls._compression_ratio(upstream_mach_number, gamma)
        u1 = shock_velocity - upstream_velocity
        res = upstream_density * u1**2 * (1 - 1 / R)
        return res if res.ndim > 0 else res.item()

    @classmethod
    def _post_shock_temperature(
        cls,
        shock_velocity: "_ArrayLike",
        upstream_mach_number: "_ArrayLike",
        upstream_velocity: "_ArrayLike" = 0.0,
        mu: float = 0.61,
        gamma: float = 5 / 3,
    ):
        r"""Post-shock temperature in the cold-upstream limit.

        Density-independent in the cold limit:

        .. math::

            T_2 = \frac{\mu m_p}{k_B} \frac{R(\mathcal{M}_1) - 1}{R(\mathcal{M}_1)^2}\, u_1^2.

        Parameters
        ----------
        shock_velocity : float or ~numpy.ndarray
            Shock velocity in cm/s.
        upstream_mach_number : float or ~numpy.ndarray
            Upstream Mach number :math:`\mathcal{M}_1`.
        upstream_velocity : float or ~numpy.ndarray, optional
            Upstream bulk velocity in cm/s. Default 0.
        mu : float, optional
            Mean molecular weight. Default 0.61.
        gamma : float, optional
            Adiabatic index. Default 5/3.

        Returns
        -------
        float or ~numpy.ndarray
            Post-shock temperature :math:`T_2` in K.
        """
        shock_velocity = np.asarray(shock_velocity)
        upstream_velocity = np.asarray(upstream_velocity)
        R = cls._compression_ratio(upstream_mach_number, gamma)
        u1 = shock_velocity - upstream_velocity
        res = (mu * m_p_cgs / k_B_cgs) * (R - 1) / R**2 * u1**2
        return res if res.ndim > 0 else res.item()

    @classmethod
    def _post_shock_thermal_energy_density(
        cls,
        shock_velocity: "_ArrayLike",
        upstream_density: "_ArrayLike",
        upstream_mach_number: "_ArrayLike",
        upstream_velocity: "_ArrayLike" = 0.0,
        gamma: float = 5 / 3,
    ):
        r"""Post-shock thermal energy density: :math:`e_2 = P_2 / (\gamma - 1)`.

        Parameters
        ----------
        shock_velocity : float or ~numpy.ndarray
            Shock velocity in cm/s.
        upstream_density : float or ~numpy.ndarray
            Upstream mass density :math:`\rho_1` in g/cm\ :sup:`3`.
        upstream_mach_number : float or ~numpy.ndarray
            Upstream Mach number :math:`\mathcal{M}_1`.
        upstream_velocity : float or ~numpy.ndarray, optional
            Upstream bulk velocity in cm/s. Default 0.
        gamma : float, optional
            Adiabatic index. Default 5/3.

        Returns
        -------
        float or ~numpy.ndarray
            Post-shock thermal energy density :math:`e_2` in erg/cm\ :sup:`3`.
        """
        P2 = np.asarray(
            cls._post_shock_pressure(shock_velocity, upstream_density, upstream_mach_number, upstream_velocity, gamma)
        )
        e2 = P2 / (gamma - 1)
        return e2 if e2.ndim > 0 else e2.item()

    # ---------------------------------------------- #
    # Aggregate backend                              #
    # ---------------------------------------------- #

    @classmethod
    def _solve(
        cls,
        shock_velocity,
        flow_density,
        flow_mach_number,
        flow_velocity=0.0,
        gamma: float = 5 / 3,
        mu: float = 0.61,
        upstream: bool = False,
    ) -> dict:
        r"""CGS aggregate backend.

        Parameters
        ----------
        shock_velocity : float or ~numpy.ndarray
            Shock front velocity in cm/s.
        flow_density : float or ~numpy.ndarray
            Upstream density (``upstream=False``) or downstream density
            (``upstream=True``) in g/cm\ :sup:`3`.
        flow_mach_number : float or ~numpy.ndarray
            Upstream Mach number :math:`\mathcal{M}_1` when ``upstream=False``;
            downstream Mach number :math:`\mathcal{M}_2` when ``upstream=True``.
        flow_velocity : float or ~numpy.ndarray, optional
            Upstream or downstream bulk velocity in cm/s. Default 0.
        gamma : float, optional
            Adiabatic index. Default 5/3.
        mu : float, optional
            Mean molecular weight. Default 0.61.
        upstream : bool, optional
            If ``False`` (default), return post-shock quantities from upstream inputs.
            If ``True``, return pre-shock kinematic quantities from downstream inputs.

        Returns
        -------
        dict
            Plain CGS values.
        """
        if not upstream:
            M1 = flow_mach_number
            return {
                "compression_ratio": cls._compression_ratio(M1, gamma),
                "post_shock_density": cls._post_shock_density(flow_density, M1, gamma),
                "post_shock_number_density": cls._post_shock_number_density(flow_density, M1, gamma, mu),
                "post_shock_velocity": cls._post_shock_velocity(shock_velocity, M1, flow_velocity, gamma),
                "post_shock_pressure": cls._post_shock_pressure(shock_velocity, flow_density, M1, flow_velocity, gamma),
                "post_shock_temperature": cls._post_shock_temperature(shock_velocity, M1, flow_velocity, mu, gamma),
                "post_shock_thermal_energy_density": cls._post_shock_thermal_energy_density(
                    shock_velocity, flow_density, M1, flow_velocity, gamma
                ),
            }
        else:
            M2 = flow_mach_number
            return {
                "compression_ratio": cls._compression_ratio_from_downstream(M2, gamma),
                "pre_shock_density": cls._pre_shock_density(flow_density, M2, gamma),
                "pre_shock_number_density": cls._pre_shock_number_density(flow_density, M2, gamma, mu),
                "pre_shock_velocity": cls._pre_shock_velocity(shock_velocity, flow_velocity, M2, gamma),
            }

    # ---------------------------------------------- #
    # Public overrides (cold-specific signatures)    #
    # ---------------------------------------------- #

    @classmethod
    def compute_post_shock_pressure(
        cls,
        shock_velocity: "_UnitBearingArrayLike",
        upstream_density: "_UnitBearingArrayLike",
        upstream_mach_number: float,
        upstream_velocity: "_UnitBearingArrayLike" = 0.0 * u.cm / u.s,
        gamma: float = 5 / 3,
    ) -> u.Quantity:
        r"""Post-shock gas pressure in the cold-upstream limit.

        Parameters
        ----------
        shock_velocity : float or ~astropy.units.Quantity
            Shock velocity :math:`v_{\rm sh}`. Bare float assumed cm/s.
        upstream_density : float or ~astropy.units.Quantity
            Upstream mass density :math:`\rho_1`. Bare float assumed g/cm\ :sup:`3`.
        upstream_mach_number : float or ~numpy.ndarray
            Upstream Mach number :math:`\mathcal{M}_1`.
        upstream_velocity : float or ~astropy.units.Quantity, optional
            Upstream bulk velocity :math:`v_1`. Bare float assumed cm/s. Default 0.
        gamma : float, optional
            Adiabatic index. Default 5/3.

        Returns
        -------
        ~astropy.units.Quantity
            Post-shock pressure :math:`P_2` in dyne/cm\ :sup:`2`.
        """
        v_sh = ensure_in_units(shock_velocity, u.cm / u.s)
        rho1 = ensure_in_units(upstream_density, u.g / u.cm**3)
        v_1 = ensure_in_units(upstream_velocity, u.cm / u.s)
        return cls._post_shock_pressure(v_sh, rho1, np.asarray(upstream_mach_number), v_1, gamma) * u.dyn / u.cm**2

    @classmethod
    def compute_post_shock_temperature(
        cls,
        shock_velocity: "_UnitBearingArrayLike",
        upstream_mach_number: float,
        upstream_velocity: "_UnitBearingArrayLike" = 0.0 * u.cm / u.s,
        mu: float = 0.61,
        gamma: float = 5 / 3,
    ) -> u.Quantity:
        r"""Post-shock temperature in the cold-upstream limit.

        Does not require upstream density — see :meth:`_post_shock_temperature`.

        Parameters
        ----------
        shock_velocity : float or ~astropy.units.Quantity
            Shock velocity :math:`v_{\rm sh}`. Bare float assumed cm/s.
        upstream_mach_number : float or ~numpy.ndarray
            Upstream Mach number :math:`\mathcal{M}_1`.
        upstream_velocity : float or ~astropy.units.Quantity, optional
            Upstream bulk velocity :math:`v_1`. Bare float assumed cm/s. Default 0.
        mu : float, optional
            Mean molecular weight. Default 0.61.
        gamma : float, optional
            Adiabatic index. Default 5/3.

        Returns
        -------
        ~astropy.units.Quantity
            Post-shock temperature :math:`T_2` in K.
        """
        v_sh = ensure_in_units(shock_velocity, u.cm / u.s)
        v_1 = ensure_in_units(upstream_velocity, u.cm / u.s)
        return cls._post_shock_temperature(v_sh, np.asarray(upstream_mach_number), v_1, mu, gamma) * u.K

    @classmethod
    def compute_post_shock_thermal_energy_density(
        cls,
        shock_velocity: "_UnitBearingArrayLike",
        upstream_density: "_UnitBearingArrayLike",
        upstream_mach_number: float,
        upstream_velocity: "_UnitBearingArrayLike" = 0.0 * u.cm / u.s,
        gamma: float = 5 / 3,
    ) -> u.Quantity:
        r"""Post-shock thermal energy density in the cold-upstream limit.

        Parameters
        ----------
        shock_velocity : float or ~astropy.units.Quantity
            Shock velocity :math:`v_{\rm sh}`. Bare float assumed cm/s.
        upstream_density : float or ~astropy.units.Quantity
            Upstream mass density :math:`\rho_1`. Bare float assumed g/cm\ :sup:`3`.
        upstream_mach_number : float or ~numpy.ndarray
            Upstream Mach number :math:`\mathcal{M}_1`.
        upstream_velocity : float or ~astropy.units.Quantity, optional
            Upstream bulk velocity :math:`v_1`. Bare float assumed cm/s. Default 0.
        gamma : float, optional
            Adiabatic index. Default 5/3.

        Returns
        -------
        ~astropy.units.Quantity
            Post-shock thermal energy density :math:`e_2` in erg/cm\ :sup:`3`.
        """
        v_sh = ensure_in_units(shock_velocity, u.cm / u.s)
        rho1 = ensure_in_units(upstream_density, u.g / u.cm**3)
        v_1 = ensure_in_units(upstream_velocity, u.cm / u.s)
        return (
            cls._post_shock_thermal_energy_density(v_sh, rho1, np.asarray(upstream_mach_number), v_1, gamma)
            * u.erg
            / u.cm**3
        )

    # ---------------------------------------------- #
    # Public aggregate                               #
    # ---------------------------------------------- #

    @classmethod
    def solve(
        cls,
        shock_velocity: "_UnitBearingArrayLike",
        flow_density: "_UnitBearingArrayLike",
        flow_mach_number: float,
        flow_velocity: "_UnitBearingArrayLike" = 0.0 * u.cm / u.s,
        gamma: float = 5 / 3,
        mu: float = 0.61,
        upstream: bool = False,
    ):
        r"""Compute all jump-condition quantities in one call.

        Parameters
        ----------
        shock_velocity : float or ~astropy.units.Quantity
            Shock front velocity. Bare float assumed cm/s.
        flow_density : float or ~astropy.units.Quantity
            Upstream density (``upstream=False``) or downstream density
            (``upstream=True``). Bare float assumed g/cm\ :sup:`3`.
        flow_mach_number : float or ~numpy.ndarray
            Upstream Mach number :math:`\mathcal{M}_1` when ``upstream=False``;
            downstream Mach number :math:`\mathcal{M}_2` when ``upstream=True``.
        flow_velocity : float or ~astropy.units.Quantity, optional
            Upstream or downstream bulk velocity. Bare float assumed cm/s. Default 0.
        gamma : float, optional
            Adiabatic index. Default 5/3.
        mu : float, optional
            Mean molecular weight. Default 0.61.
        upstream : bool, optional
            If ``False`` (default), return a :class:`WeakColdShockConditionsResult`
            with post-shock quantities. If ``True``, return a
            :class:`WeakColdShockConditionsUpstreamResult` with pre-shock kinematic
            quantities (density, number density, velocity only).

        Returns
        -------
        WeakColdShockConditionsResult or WeakColdShockConditionsUpstreamResult
        """
        v_sh = ensure_in_units(shock_velocity, u.cm / u.s)
        rho = ensure_in_units(flow_density, u.g / u.cm**3)
        v = ensure_in_units(flow_velocity, u.cm / u.s)
        result = cls._solve(v_sh, rho, np.asarray(flow_mach_number), v, gamma, mu, upstream=upstream)
        return cls._make_upstream_result(result) if upstream else cls._make_result(result)
