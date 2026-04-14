r"""
Thin-disk analytical accretion disk models.

This module implements the standard Shakura-Sunyaev (1973) steady-state alpha-disk
model, providing both a low-level CGS API and a unit-aware public API following the
two-level convention used throughout Triceratops.

The central class is :class:`AlphaDisk`, which evaluates the canonical SS73 scalings
for surface density, midplane temperature, scale height, midplane density, optical
depth, kinematic viscosity, and radial drift velocity at an arbitrary radius (or radius
array).  In addition, the class inherits methods from :class:`ThinDiskBase` for
computing the multi-colour blackbody SED by integrating the Planck function over the
disk annuli, and the analytic bolometric luminosity.

Two-Level API
-------------
- ``_compute_cgs(radius_cm, M_BH_g, mdot_gs, R_in_cm)`` — pure-NumPy CGS evaluation.
- ``compute(radius, M_BH, mdot, R_in)`` — strips :class:`~astropy.units.Quantity` units,
  delegates to ``_compute_cgs``, and returns a ``dict`` of unit-bearing results.

See Also
--------
:mod:`triceratops.dynamics.accretion.utils` : Low-level disk utility functions (T_eff profile, SED integrals, etc.)
:mod:`triceratops.dynamics.accretion.one_zone` : Time-dependent one-zone disk models.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union

import numpy as np
from astropy import constants as const
from astropy import units as u

from triceratops.dynamics.accretion.utils import (
    _log_disk_bolometric_luminosity,
    _log_disk_effective_temperature,
    _log_disk_flux_density,
    _log_disk_spectral_luminosity,
)
from triceratops.utils.misc_utils import ensure_in_units

if TYPE_CHECKING:
    from triceratops._typing import _ArrayLike, _UnitBearingArrayLike

__all__ = ["ThinDiskBase", "AlphaDisk"]

# ---------------------------------------------------------------------------
# CGS normalisation constants (SS73 dimensionless ratios)
# ---------------------------------------------------------------------------
#: Solar mass in grams  (M₁ ≡ M / M_sun)
_M_SUN_CGS: float = const.M_sun.cgs.value
#: Normalisation accretion rate: 10¹⁶ g s⁻¹  (defines Ṁ₁₆)
_MDOT_NORM_CGS: float = 1.0e16
#: Normalisation radius: 10¹⁰ cm  (defines R₁₀)
_R_NORM_CGS: float = 1.0e10


# =============================================================================
# Abstract base class
# =============================================================================
class ThinDiskBase(ABC):
    """
    Abstract base class for steady-state thin-disk accretion models.

    A :class:`ThinDiskBase` encapsulates the physical scalings needed to evaluate
    the local disk structure (temperature, surface density, scale height, etc.) at
    an arbitrary radius given the global disk parameters.  Subclasses implement the
    concrete scaling relations in :meth:`_compute_cgs`.

    The class also provides shared, physics-agnostic methods for:

    - :meth:`compute_effective_temperature` — T_eff(r) from the first-principles
      viscous dissipation formula (independent of the specific scaling relations).
    - :meth:`compute_sed` — multi-colour blackbody SED by numerically integrating
      the Planck function over the disk annuli.
    - :meth:`compute_bolometric_luminosity` — analytic L_bol = G M Ṁ / (2 R_in).

    Design Notes
    ------------
    The base class is **stateless** with respect to the disk physical parameters:
    ``M_BH``, ``mdot``, and ``R_in`` are passed at call time, not stored at
    construction.  Subclass ``__init__`` methods may cache model-level constants
    (e.g. the viscosity parameter α) but must not cache physical state.
    """

    # =========================================================================
    # Initialisation
    # =========================================================================
    @abstractmethod
    def __init__(self, **kwargs):
        """
        Instantiate the disk engine.

        Parameters
        ----------
        kwargs
            Model-level configuration (e.g. viscosity parameter ``alpha``).
            Physical disk state must **not** be stored here.
        """
        pass

    # =========================================================================
    # Core structural computation
    # =========================================================================
    def compute(
        self,
        radius: "_UnitBearingArrayLike",
        M_BH: "_UnitBearingArrayLike",
        mdot: "_UnitBearingArrayLike",
        R_in: "_UnitBearingArrayLike",
    ) -> "dict[str, Union[u.Quantity, _ArrayLike]]":
        r"""
        Evaluate the disk structure at the given radius (or radii).

        This is the high-level, unit-aware wrapper around :meth:`_compute_cgs`.
        If the inputs carry :class:`~astropy.units.Quantity` units they are
        converted internally; otherwise CGS units are assumed.

        Parameters
        ----------
        radius : array-like or `~astropy.units.Quantity`
            Radial coordinate(s) [cm or Quantity].
        M_BH : float or `~astropy.units.Quantity`
            Black-hole (central object) mass [g or Quantity].
        mdot : float or `~astropy.units.Quantity`
            Mass accretion rate :math:`\dot{M}` [g s⁻¹ or Quantity].
        R_in : float or `~astropy.units.Quantity`
            Inner truncation radius (e.g. ISCO) [cm or Quantity].

        Returns
        -------
        dict
            Keys and units:

            - ``"Sigma"``  : surface density [g cm⁻²]
            - ``"T_c"``    : midplane temperature [K]
            - ``"H"``      : pressure scale height [cm]
            - ``"rho"``    : midplane mass density [g cm⁻³]
            - ``"tau"``    : vertical optical depth [dimensionless]
            - ``"nu"``     : kinematic viscosity [cm² s⁻¹]
            - ``"v_R"``    : radial drift velocity [cm s⁻¹]
        """
        radius_cm = np.asarray(ensure_in_units(radius, "cm"), dtype=float)
        M_BH_g = float(ensure_in_units(M_BH, "g"))
        mdot_gs = float(ensure_in_units(mdot, "g/s"))
        R_in_cm = float(ensure_in_units(R_in, "cm"))

        raw = self._compute_cgs(radius_cm, M_BH_g, mdot_gs, R_in_cm)

        return {
            "Sigma": raw["Sigma"] * u.g / u.cm**2,
            "T_c": raw["T_c"] * u.K,
            "H": raw["H"] * u.cm,
            "rho": raw["rho"] * u.g / u.cm**3,
            "tau": raw["tau"],
            "nu": raw["nu"] * u.cm**2 / u.s,
            "v_R": raw["v_R"] * u.cm / u.s,
        }

    @abstractmethod
    def _compute_cgs(
        self,
        radius_cm: np.ndarray,
        M_BH_g: float,
        mdot_gs: float,
        R_in_cm: float,
    ) -> "dict[str, np.ndarray]":
        """
        Low-level CGS evaluation of the disk structure.

        All inputs and outputs are plain floats / NumPy arrays in CGS.
        Override this method in subclasses to implement the specific
        scaling relations.

        Parameters
        ----------
        radius_cm : ndarray
            Radial coordinate(s) in cm.
        M_BH_g : float
            Black-hole mass in grams.
        mdot_gs : float
            Accretion rate in g s⁻¹.
        R_in_cm : float
            Inner truncation radius in cm.

        Returns
        -------
        dict
            Keys: ``"Sigma"``, ``"T_c"``, ``"H"``, ``"rho"``, ``"tau"``,
            ``"nu"``, ``"v_R"`` — all CGS, no units attached.
        """
        pass

    # =========================================================================
    # Effective temperature profile
    # =========================================================================
    def compute_effective_temperature(
        self,
        radius: "_UnitBearingArrayLike",
        M_BH: "_UnitBearingArrayLike",
        mdot: "_UnitBearingArrayLike",
        R_in: "_UnitBearingArrayLike",
    ) -> u.Quantity:
        r"""
        Effective (surface) temperature profile of the steady-state disk.

        Computed from the first-principles viscous dissipation formula,
        independent of the specific scaling-relation prescription:

        .. math::

            T_{\rm eff}(r) = \left(\frac{3\,G\,M_{\rm BH}\,\dot{M}}
                                   {8\pi\,\sigma_{\rm SB}\,r^3}
                             \left[1-\sqrt{\frac{R_{\rm in}}{r}}\right]
                             \right)^{1/4}

        This differs from the midplane temperature :math:`T_c` returned by
        :meth:`compute`; the two are related by
        :math:`T_c = \bigl(\tfrac{3\tau}{4}\bigr)^{1/4} T_{\rm eff}`.

        Parameters
        ----------
        radius : array-like or `~astropy.units.Quantity`
            Radial coordinate(s).
        M_BH : float or `~astropy.units.Quantity`
            Black-hole mass.
        mdot : float or `~astropy.units.Quantity`
            Mass accretion rate.
        R_in : float or `~astropy.units.Quantity`
            Inner truncation radius.

        Returns
        -------
        `~astropy.units.Quantity`
            Effective temperature :math:`T_{\rm eff}(r)` in K.
        """
        radius_val = ensure_in_units(radius, "cm")
        scalar_radius = np.ndim(radius_val) == 0
        log_r = np.log(np.atleast_1d(np.asarray(radius_val, dtype=float)))
        log_M_BH = float(np.log(ensure_in_units(M_BH, "g")))
        log_M_dot = float(np.log(ensure_in_units(mdot, "g/s")))
        log_R_in = float(np.log(ensure_in_units(R_in, "cm")))

        log_T = np.asarray(_log_disk_effective_temperature(log_r, log_M_BH, log_M_dot, log_R_in))
        result = np.exp(log_T)
        if scalar_radius:
            result = result.item()
        return result * u.K

    # =========================================================================
    # SED computation
    # =========================================================================
    def compute_sed(
        self,
        nu: "_UnitBearingArrayLike",
        M_BH: "_UnitBearingArrayLike",
        mdot: "_UnitBearingArrayLike",
        R_in: "_UnitBearingArrayLike",
        R_out: "_UnitBearingArrayLike",
        D_L: "Union[_UnitBearingArrayLike, None]" = None,
        cos_theta: float = 1.0,
        N_r: int = 500,
    ) -> "dict[str, u.Quantity]":
        r"""
        Multi-colour blackbody spectral energy distribution of the disk.

        Integrates the Planck function weighted by the local effective temperature
        :math:`T_{\rm eff}(r)` over all annuli from :math:`R_{\rm in}` to
        :math:`R_{\rm out}`:

        .. math::

            L_\nu = 4\pi^2
                    \int_{R_{\rm in}}^{R_{\rm out}}
                    B_\nu\!\left[T_{\rm eff}(r)\right]\,r\,\mathrm{d}r

        If a luminosity distance ``D_L`` is supplied the observed flux density is
        also returned:

        .. math::

            F_\nu = \frac{4\pi\cos\theta}{D_L^2}
                    \int_{R_{\rm in}}^{R_{\rm out}}
                    B_\nu\!\left[T_{\rm eff}(r)\right]\,r\,\mathrm{d}r

        The radial integral is evaluated on a log-spaced grid using the
        trapezoidal rule (see :func:`.utils._disk_planck_ring_integral`).

        Parameters
        ----------
        nu : array-like or `~astropy.units.Quantity`
            Frequency grid [Hz or Quantity].
        M_BH : float or `~astropy.units.Quantity`
            Black-hole mass.
        mdot : float or `~astropy.units.Quantity`
            Mass accretion rate.
        R_in : float or `~astropy.units.Quantity`
            Inner truncation radius.
        R_out : float or `~astropy.units.Quantity`
            Outer disk radius.
        D_L : float or `~astropy.units.Quantity`, optional
            Luminosity distance.  If provided, ``"F_nu"`` is included in the
            returned dict.
        cos_theta : float, optional
            Cosine of the inclination angle (1 = face-on).
        N_r : int, optional
            Number of radial quadrature points (default 500).

        Returns
        -------
        dict
            - ``"L_nu"`` : spectral luminosity [erg s⁻¹ Hz⁻¹] (always present)
            - ``"F_nu"`` : flux density [erg s⁻¹ Hz⁻¹ cm⁻²] (only if *D_L* given)
        """
        nu_val = ensure_in_units(nu, "Hz")
        scalar_nu = np.ndim(nu_val) == 0
        log_nu = np.log(np.atleast_1d(np.asarray(nu_val, dtype=float)))

        log_M_BH = float(np.log(ensure_in_units(M_BH, "g")))
        log_M_dot = float(np.log(ensure_in_units(mdot, "g/s")))
        log_R_in = float(np.log(ensure_in_units(R_in, "cm")))
        log_R_out = float(np.log(ensure_in_units(R_out, "cm")))

        log_L = _log_disk_spectral_luminosity(log_nu, log_M_BH, log_M_dot, log_R_in, log_R_out, N_r)
        L_nu_arr = np.exp(log_L)
        if scalar_nu:
            L_nu_arr = L_nu_arr.item()

        out: dict[str, u.Quantity] = {"L_nu": L_nu_arr * u.erg / u.s / u.Hz}

        if D_L is not None:
            log_DL = float(np.log(ensure_in_units(D_L, "cm")))
            log_F = _log_disk_flux_density(log_nu, log_M_BH, log_M_dot, log_R_in, log_R_out, log_DL, cos_theta, N_r)
            F_nu_arr = np.exp(log_F)
            if scalar_nu:
                F_nu_arr = F_nu_arr.item()
            out["F_nu"] = F_nu_arr * u.erg / u.s / u.Hz / u.cm**2

        return out

    # =========================================================================
    # Bolometric luminosity
    # =========================================================================
    def compute_bolometric_luminosity(
        self,
        M_BH: "_UnitBearingArrayLike",
        mdot: "_UnitBearingArrayLike",
        R_in: "_UnitBearingArrayLike",
    ) -> u.Quantity:
        r"""
        Analytic bolometric luminosity of the steady-state thin disk.

        Integrating the viscous dissipation profile :math:`Q^+(r)` over both
        disk faces with a zero-torque inner boundary and
        :math:`R_{\rm out} \to \infty` gives the exact result:

        .. math::

            L_{\rm bol} = \frac{G\,M_{\rm BH}\,\dot{M}}{2\,R_{\rm in}}

        This is equivalent to half the gravitational energy released as material
        falls from infinity to :math:`R_{\rm in}`.

        Parameters
        ----------
        M_BH : float or `~astropy.units.Quantity`
            Black-hole mass.
        mdot : float or `~astropy.units.Quantity`
            Mass accretion rate :math:`\dot{M}`.
        R_in : float or `~astropy.units.Quantity`
            Inner truncation radius.

        Returns
        -------
        `~astropy.units.Quantity`
            Bolometric luminosity :math:`L_{\rm bol}` [erg s⁻¹].
        """
        log_M_BH = float(np.log(ensure_in_units(M_BH, "g")))
        log_M_dot = float(np.log(ensure_in_units(mdot, "g/s")))
        log_R_in = float(np.log(ensure_in_units(R_in, "cm")))

        log_L = _log_disk_bolometric_luminosity(log_M_BH, log_M_dot, log_R_in)
        return float(np.exp(log_L)) * u.erg / u.s

    # =========================================================================
    # Dunder methods
    # =========================================================================
    def __call__(
        self,
        radius: "_UnitBearingArrayLike",
        M_BH: "_UnitBearingArrayLike",
        mdot: "_UnitBearingArrayLike",
        R_in: "_UnitBearingArrayLike",
    ) -> "dict[str, Union[u.Quantity, _ArrayLike]]":
        """
        Evaluate the disk structure (alias for :meth:`compute`).

        See :meth:`compute` for full parameter and return documentation.
        """
        return self.compute(radius, M_BH, mdot, R_in)


# =============================================================================
# Sunyaev-Shakura alpha disk
# =============================================================================
class AlphaDisk(ThinDiskBase):
    r"""
    Sunyaev-Shakura alpha-disk model (SS73 zone-B scalings).

    This model assumes a geometrically thin, optically thick accretion disk in
    which viscosity is parameterised by the dimensionless Shakura-Sunyaev
    :math:`\alpha` parameter.  The disk structure is determined by the balance
    of viscous heating and radiative cooling, yielding the following analytical
    scalings at radius :math:`R`:

    .. math::

        \begin{aligned}
        \Sigma &\;=\; 5.2\;
        \alpha^{-4/5}\,
        \dot{M}_{16}^{7/10}\,
        M_1^{1/4}\,
        R_{10}^{-3/4}\,
        f^{14/5}
        \quad [\mathrm{g\,cm^{-2}}],
        \\[4pt]
        T_c &\;=\; 1.4\times10^{4}\;
        \alpha^{-1/5}\,
        \dot{M}_{16}^{3/10}\,
        M_1^{1/4}\,
        R_{10}^{-3/4}\,
        f^{6/5}
        \quad [\mathrm{K}],
        \\[4pt]
        H &\;=\; 1.7\times10^{8}\;
        \alpha^{-1/10}\,
        \dot{M}_{16}^{3/20}\,
        M_1^{-3/8}\,
        R_{10}^{9/8}\,
        f^{3/5}
        \quad [\mathrm{cm}],
        \\[4pt]
        \rho &\;=\; 3.1\times10^{-8}\;
        \alpha^{-7/10}\,
        \dot{M}_{16}^{11/20}\,
        M_1^{5/8}\,
        R_{10}^{-15/8}\,
        f^{11/5}
        \quad [\mathrm{g\,cm^{-3}}],
        \\[4pt]
        \tau &\;=\; 190\;
        \alpha^{-4/5}\,
        \dot{M}_{16}^{1/5}\,
        f^{4/5},
        \\[4pt]
        \nu &\;=\; 1.8\times10^{14}\;
        \alpha^{4/5}\,
        \dot{M}_{16}^{3/10}\,
        M_1^{-1/4}\,
        R_{10}^{3/4}\,
        f^{6/5}
        \quad [\mathrm{cm^2\,s^{-1}}],
        \\[4pt]
        v_R &\;=\; 2.7\times10^{4}\;
        \alpha^{4/5}\,
        \dot{M}_{16}^{3/10}\,
        M_1^{-1/4}\,
        R_{10}^{-1/4}\,
        f^{-14/5}
        \quad [\mathrm{cm\,s^{-1}}],
        \end{aligned}

    where :math:`\dot{M}_{16} \equiv \dot{M}/10^{16}\ \mathrm{g\,s^{-1}}`,
    :math:`M_1 \equiv M_{\rm BH}/M_\odot`,
    :math:`R_{10} \equiv R/10^{10}\ \mathrm{cm}`, and

    .. math::

        f^4 \;\equiv\; 1 - \sqrt{\frac{R_{\rm in}}{R}}.

    Parameters
    ----------
    alpha : float
        Shakura-Sunyaev dimensionless viscosity parameter
        (:math:`0 < \alpha \lesssim 1`).

    Examples
    --------
    Evaluate the disk structure at three radii for a :math:`10\,M_\odot` black hole:

    .. code-block:: python

        import astropy.units as u
        from astropy import constants as const
        from triceratops.dynamics.accretion import (
            AlphaDisk,
        )

        disk = AlphaDisk(alpha=0.1)
        R = [1e9, 1e10, 1e11] * u.cm
        result = disk.compute(
            R,
            10 * const.M_sun,
            1e16 * u.g / u.s,
            3e6 * u.cm,
        )
        print(
            result["T_c"]
        )  # midplane temperature at each radius

    Compute the bolometric luminosity and a broadband SED:

    .. code-block:: python

        import numpy as np

        L_bol = disk.compute_bolometric_luminosity(
            10 * const.M_sun, 1e16 * u.g / u.s, 3e6 * u.cm
        )

        nu = np.geomspace(1e13, 1e17, 300) * u.Hz
        sed = disk.compute_sed(
            nu,
            10 * const.M_sun,
            1e16 * u.g / u.s,
            R_in=3e6 * u.cm,
            R_out=1e12 * u.cm,
            D_L=100 * u.Mpc,
        )
        print(sed["F_nu"])

    References
    ----------
    Shakura, N. I., & Sunyaev, R. A. 1973, A&A, 24, 337.
    """

    def __init__(self, alpha: float):
        """
        Instantiate an AlphaDisk with the given viscosity parameter.

        Parameters
        ----------
        alpha : float
            Shakura-Sunyaev viscosity parameter.
        """
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must satisfy 0 < alpha <= 1; got {alpha}.")
        self.alpha = float(alpha)

    # -------------------------------------------------------------------------
    # Low-level CGS implementation
    # -------------------------------------------------------------------------
    def _compute_cgs(
        self,
        radius_cm: np.ndarray,
        M_BH_g: float,
        mdot_gs: float,
        R_in_cm: float,
    ) -> "dict[str, np.ndarray]":
        """
        Evaluate the Shakura-Sunyaev scalings in CGS.

        Parameters
        ----------
        radius_cm : ndarray
            Radial coordinate(s) in cm.  Must satisfy ``radius_cm >= R_in_cm``.
        M_BH_g : float
            Black-hole mass in grams.
        mdot_gs : float
            Accretion rate in g s⁻¹.
        R_in_cm : float
            Inner truncation radius in cm.

        Returns
        -------
        dict
            Keys: ``"Sigma"``, ``"T_c"``, ``"H"``, ``"rho"``, ``"tau"``,
            ``"nu"``, ``"v_R"`` (all CGS, plain NumPy values).
        """
        # --- Dimensionless SS ratios ---
        Mdot_16 = mdot_gs / _MDOT_NORM_CGS
        M1 = M_BH_g / _M_SUN_CGS
        R10 = radius_cm / _R_NORM_CGS

        # Inner-boundary correction factor: f⁴ ≡ 1 − √(R_in/R)
        with np.errstate(invalid="ignore"):
            xi = 1.0 - np.sqrt(R_in_cm / radius_cm)
        f = np.maximum(xi, 0.0) ** 0.25  # f = max(f⁴, 0)^{1/4}

        alpha = self.alpha

        # Precompute shared scalar powers to avoid redundant exponentiation.
        a_neg45 = alpha ** (-4.0 / 5.0)
        a_pos45 = alpha ** (4.0 / 5.0)

        m_310 = Mdot_16 ** (3.0 / 10.0)
        m_710 = Mdot_16 ** (7.0 / 10.0)

        M1_14 = M1 ** (1.0 / 4.0)
        M1_neg14 = M1 ** (-1.0 / 4.0)

        R10_neg34 = R10 ** (-3.0 / 4.0)

        f_65 = f ** (6.0 / 5.0)
        f_145 = f ** (14.0 / 5.0)

        # --- SS73 scalings (exact coefficients from the class docstring) ---
        Sigma = 5.2 * a_neg45 * m_710 * M1_14 * R10_neg34 * f_145
        T_c = 1.4e4 * alpha ** (-1.0 / 5.0) * m_310 * M1_14 * R10_neg34 * f_65
        H = (
            1.7e8
            * alpha ** (-1.0 / 10.0)
            * Mdot_16 ** (3.0 / 20.0)
            * M1 ** (-3.0 / 8.0)
            * R10 ** (9.0 / 8.0)
            * f ** (3.0 / 5.0)
        )
        rho = (
            3.1e-8
            * alpha ** (-7.0 / 10.0)
            * Mdot_16 ** (11.0 / 20.0)
            * M1 ** (5.0 / 8.0)
            * R10 ** (-15.0 / 8.0)
            * f ** (11.0 / 5.0)
        )
        tau = 190.0 * a_neg45 * Mdot_16 ** (1.0 / 5.0) * f ** (4.0 / 5.0)
        nu = 1.8e14 * a_pos45 * m_310 * M1_neg14 * R10 ** (3.0 / 4.0) * f_65
        v_R = 2.7e4 * a_pos45 * m_310 * M1_neg14 * R10 ** (-1.0 / 4.0) * f ** (-14.0 / 5.0)

        return {
            "Sigma": Sigma,
            "T_c": T_c,
            "H": H,
            "rho": rho,
            "tau": tau,
            "nu": nu,
            "v_R": v_R,
        }

    # -------------------------------------------------------------------------
    # Dunder methods
    # -------------------------------------------------------------------------
    def __str__(self) -> str:
        return f"AlphaDisk(alpha={self.alpha})"

    def __repr__(self) -> str:
        return f"<AlphaDisk(alpha={self.alpha})>"
