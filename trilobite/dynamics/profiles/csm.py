r"""
Circumstellar-medium density profiles for dynamical calculations.

This module defines the CSM density-profile interface used by the dynamics
package, together with a collection of common analytic circumstellar-medium
profiles. These profiles are intended primarily as upstream source functions for
shock engines, but are kept independent of any particular shock model so they
can also be reused by other dynamical calculations.

The implemented profiles include steady winds, uniform media, single and broken
power laws, top-hat shells, Gaussian shells, wind-plus-floor profiles, and sharp
or smoothly truncated winds.

.. rubric:: Conventions

All profiles use the two-argument source-function signature

.. math::

    \rho = \rho(r, t),

even when the profile is stationary. This keeps CSM profiles interchangeable
with time-dependent density fields in shock-engine source terms. Stationary
profiles simply ignore the time argument internally.

Bare numerical inputs are interpreted as CGS values. Unit-bearing inputs are
converted once in the validation layer before optimized numerical evaluation.


.. rubric:: Module Design

Profiles are implemented as **stateless, classmethod-based evaluators**. Physical
parameters are supplied at evaluation time rather than stored on an instance.
This is useful for inference and parameter scans, where the same profile form is
evaluated many times with different parameter values.

Each profile separates the user-facing and solver-facing paths:

``eval``
    Safe, unit-aware evaluation. Radius, time, and profile parameters may be
    supplied as :class:`~astropy.units.Quantity` objects. Inputs are converted to
    CGS, validated, and the returned density is given in
    :math:`\mathrm{g\,cm^{-3}}`.

``feval``
    Fast unit-free evaluation. Inputs are assumed to already be processed,
    validated, and in CGS units. This path is intended for ODE right-hand sides
    and other hot loops.

``as_callable``
    Freeze a parameter set into a unit-aware callable of ``(r, t)``.

``as_optimized_callable``
    Freeze a parameter set into a unit-free CGS callable of ``(r, t)`` suitable
    for repeated evaluation inside numerical solvers.
"""

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
from astropy import units as u
from scipy.integrate import quad
from scipy.special import expit

from trilobite.utils.misc_utils import ensure_in_units

from .core import _DynamicalProfile

if TYPE_CHECKING:
    from trilobite._typing import _UnitBearingArrayLike, _UnitBearingScalarLike


# =========================================================================== #
# CSM Density Profile ABCs                                                    #
# =========================================================================== #
class CSMDensityProfile(_DynamicalProfile):
    r"""
    Abstract base class for CSM density profiles.

    A CSM density profile represents an upstream circumstellar-medium density
    field,

    .. math::

        \rho_{\rm CSM} = \rho_{\rm CSM}(r, t;\,\boldsymbol{\theta}),

    where ``r`` is radius, ``t`` is time, and :math:`\boldsymbol{\theta}`
    contains the physical parameters defining the profile.

    The public evaluation method :meth:`eval` accepts unit-bearing arguments and
    returns an :class:`~astropy.units.Quantity` in :math:`\mathrm{g\,cm^{-3}}`.
    The fast evaluation method :meth:`feval` and the optimized callable from
    :meth:`as_optimized_callable` bypass unit handling for use inside ODE kernels.

    Subclasses must implement:

    ``_validate_and_process_parameters``
        Validate profile parameters and convert them to unit-free CGS values.

    ``_opt_eval``
        Evaluate the density using already-processed unit-free inputs.

    Notes
    -----
    This class assumes the two-argument signature ``(r, t)`` for compatibility
    with shock-engine source terms. Stationary profiles (see
    :class:`StationaryCSMDensityProfile`) may ignore ``t`` internally.

    See Also
    --------
    StationaryCSMDensityProfile : ABC for time-independent CSM profiles.
    WindCSMProfile : Steady wind-like CSM.
    UniformCSMProfile : Uniform CSM density.
    """

    OUTPUT_UNITS = u.g / u.cm**3

    @classmethod
    @abstractmethod
    def _validate_and_process_parameters(cls, **parameters: Any) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _opt_eval(cls, r: np.ndarray, t: np.ndarray, **parameters: Any) -> np.ndarray:
        raise NotImplementedError


class StationaryCSMDensityProfile(CSMDensityProfile):
    r"""
    Abstract base class for stationary (time-independent) CSM density profiles.

    A stationary CSM profile satisfies

    .. math::

        \rho_{\rm CSM}(r, t) = \rho_{\rm CSM}(r).

    The time argument is **optional** at every call site:

    - ``eval(r, **params)`` — no time argument.
    - ``eval(r, t, **params)`` — time provided positionally (ignored internally).
    - ``eval(r, t=t, **params)`` — time provided as a keyword (ignored internally).

    This allows stationary profiles to be used interchangeably with
    time-dependent profiles in shock engines without requiring a dummy time
    value at the call site.
    """

    @classmethod
    def _validate_and_process_arguments(
        cls,
        r: "_UnitBearingArrayLike",
        _t: "_UnitBearingArrayLike" = None,
    ) -> tuple[np.ndarray]:
        r_cgs = np.asarray(ensure_in_units(r, u.cm))
        return (r_cgs,)

    @classmethod
    @abstractmethod
    def _validate_and_process_parameters(cls, **parameters: Any) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _opt_eval(cls, r: np.ndarray, _t=None, **parameters: Any) -> np.ndarray:
        raise NotImplementedError


# =========================================================================== #
# Concrete CSM Profiles                                                       #
# =========================================================================== #


class WindCSMProfile(StationaryCSMDensityProfile):
    r"""
    Steady spherically symmetric wind CSM density profile.

    A steady, freely expanding wind with constant mass-loss rate and terminal
    velocity produces the density field

    .. math::

        \rho_{\mathrm{CSM}}(r) = A\,r^{-2},

    where the wind-density normalization is

    .. math::

        A = \frac{\dot{M}}{4\pi v_w}.

    The normalization :math:`A` carries units of :math:`\mathrm{g\,cm^{-1}}`.
    In transient and afterglow applications it is conventional to express this
    via the dimensionless wind-density parameter :math:`A_*`,

    .. math::

        A = 5\times10^{11}\,A_*\;\mathrm{g\,cm^{-1}},

    so that :math:`A_* = 1` corresponds to
    :math:`\dot{M} = 10^{-5}\,M_\odot\,\mathrm{yr}^{-1}` and
    :math:`v_w = 1000\,\mathrm{km\,s^{-1}}`.

    .. rubric:: Physical relevance

    The :math:`\rho \propto r^{-2}` wind profile is the canonical CSM
    environment for core-collapse supernovae from stars with vigorous mass
    loss — Type IIn supernovae, Wolf-Rayet progenitors, and red supergiants.
    It enters the Chevalier (1982) self-similar shock solution as the
    :math:`s = 2` case and is the standard upstream density for radio and
    X-ray afterglow modelling.  Typical values range from
    :math:`A_* \sim 0.01\text{--}1` for WR stars to
    :math:`A_* \sim 1\text{--}100` for dense RSG winds (Type IIn events).

    .. rubric:: Parameters

    Exactly one parameterisation route must be supplied.  Route 3 requires
    ``mass_loss_rate`` and ``wind_velocity`` together; supplying either alone
    raises a :class:`ValueError`.

    .. list-table::
       :header-rows: 1
       :widths: 22 6 20 52

       * - Parameter
         - Opt.
         - Type
         - Description
       * - ``A``
         - ✓
         - float or :class:`~astropy.units.Quantity`
         - Wind-density normalization in :math:`\mathrm{g\,cm^{-1}}` (Route 1).
           Bare values interpreted as :math:`\mathrm{g\,cm^{-1}}`. Must be
           positive.
       * - ``A_star``
         - ☐
         - float
         - Dimensionless wind-density parameter (Route 2),
           :math:`A = 5\times10^{11}A_*\,\mathrm{g\,cm^{-1}}`. Must be
           positive. Converted to ``A`` before numerical evaluation.
       * - ``mass_loss_rate``
         - ☐
         - float or :class:`~astropy.units.Quantity`
         - Progenitor mass-loss rate (Route 3). Bare values interpreted as
           :math:`\mathrm{g\,s^{-1}}`. Must be positive. Converted to ``A``
           before numerical evaluation.
       * - ``wind_velocity``
         - ☐
         - float or :class:`~astropy.units.Quantity`
         - Wind terminal velocity (Route 3). Bare values interpreted as
           :math:`\mathrm{cm\,s^{-1}}`. Must be positive. Converted to ``A``
           before numerical evaluation.

    .. rubric:: Example

    .. plot::

       import numpy as np
       import astropy.units as u
       import matplotlib.pyplot as plt
       from trilobite.dynamics.profiles.csm import WindCSMProfile
       from trilobite.utils.plot_utils import set_plot_style

       set_plot_style()
       r = np.geomspace(1e14, 1e18, 300) * u.cm
       fig, ax = plt.subplots()
       for A_star, ls in [(0.1, '--'), (1.0, '-'), (10.0, ':')]:
           rho = WindCSMProfile.eval(r, A_star=A_star)
           ax.loglog(r.to(u.cm).value, rho.to(u.g / u.cm**3).value,
                     ls=ls, label=fr'$A_* = {A_star}$')
       ax.set_xlabel(r'$r\ [\mathrm{cm}]$')
       ax.set_ylabel(r'$\rho\ [\mathrm{g\,cm^{-3}}]$')
       ax.set_title('Wind CSM Profile')
       ax.legend()
       plt.tight_layout()

    See Also
    --------
    normalize
        Compute :math:`A` from a mass-loss rate and wind velocity.
    normalize_A_star
        Compute :math:`A_*` from a mass-loss rate and wind velocity.
    normalize_from_A_star
        Compute :math:`A` from a supplied :math:`A_*`.
    """

    A_STAR_NORMALIZATION = 5.0e11 * u.g / u.cm

    @classmethod
    def normalize(
        cls,
        mass_loss_rate: "_UnitBearingScalarLike",
        wind_velocity: "_UnitBearingScalarLike",
    ) -> u.Quantity:
        r"""
        Compute the wind-density normalization :math:`A`.

        For a steady, spherically symmetric wind,

        .. math::

            A = \frac{\dot M}{4\pi v_w}.

        Parameters
        ----------
        mass_loss_rate : float or ~astropy.units.Quantity
            Progenitor mass-loss rate. Unit-bearing inputs are converted to
            :math:`\mathrm{g\,s^{-1}}`; bare values are interpreted as
            :math:`\mathrm{g\,s^{-1}}`.
        wind_velocity : float or ~astropy.units.Quantity
            Wind velocity. Unit-bearing inputs are converted to
            :math:`\mathrm{cm\,s^{-1}}`; bare values are interpreted as
            :math:`\mathrm{cm\,s^{-1}}`.

        Returns
        -------
        A : ~astropy.units.Quantity
            Wind-density normalization in :math:`\mathrm{g\,cm^{-1}}`.

        Raises
        ------
        ValueError
            If ``wind_velocity`` is non-positive.
        """
        mdot = ensure_in_units(mass_loss_rate, u.g / u.s)
        v_w = ensure_in_units(wind_velocity, u.cm / u.s)

        if np.any(v_w <= 0):
            raise ValueError("`wind_velocity` must be positive.")

        return mdot / (4.0 * np.pi * v_w) * (u.g / u.cm)

    @classmethod
    def normalize_A_star(
        cls,
        mass_loss_rate: "_UnitBearingScalarLike",
        wind_velocity: "_UnitBearingScalarLike",
    ) -> float:
        r"""
        Compute the dimensionless wind-density parameter :math:`A_*`.

        The parameter :math:`A_*` is defined by

        .. math::

            A = 5\times10^{11} A_*\ {\rm g\,cm^{-1}}.

        Equivalently,

        .. math::

            A_*
            =
            \left(\frac{\dot M}{10^{-5}\ M_\odot\,{\rm yr^{-1}}}\right)
            \left(\frac{v_w}{1000\ {\rm km\,s^{-1}}}\right)^{-1}.

        Parameters
        ----------
        mass_loss_rate : float or ~astropy.units.Quantity
            Progenitor mass-loss rate. Unit-bearing inputs are converted to
            :math:`\mathrm{g\,s^{-1}}`; bare values are interpreted as
            :math:`\mathrm{g\,s^{-1}}`.
        wind_velocity : float or ~astropy.units.Quantity
            Wind velocity. Unit-bearing inputs are converted to
            :math:`\mathrm{cm\,s^{-1}}`; bare values are interpreted as
            :math:`\mathrm{cm\,s^{-1}}`.

        Returns
        -------
        A_star : float
            Dimensionless wind-density parameter.
        """
        A = cls.normalize(
            mass_loss_rate=mass_loss_rate,
            wind_velocity=wind_velocity,
        )

        return float((A / cls.A_STAR_NORMALIZATION).decompose())

    @classmethod
    def normalize_from_A_star(
        cls,
        A_star: float,
    ) -> u.Quantity:
        r"""
        Compute :math:`A` from the dimensionless wind-density parameter.

        Parameters
        ----------
        A_star : float
            Dimensionless wind-density parameter, defined by
            :math:`A = 5\times10^{11}A_*\ {\rm g\,cm^{-1}}`.

        Returns
        -------
        A : ~astropy.units.Quantity
            Wind-density normalization in :math:`\mathrm{g\,cm^{-1}}`.

        Raises
        ------
        ValueError
            If ``A_star`` is non-positive.
        """
        A_star = float(A_star)

        if A_star <= 0:
            raise ValueError("`A_star` must be positive.")

        return A_star * cls.A_STAR_NORMALIZATION

    @classmethod
    def _validate_and_process_parameters(cls, *, A=None, A_star=None, mass_loss_rate=None, wind_velocity=None, **_):
        has_A = A is not None
        has_A_star = A_star is not None
        has_physical = mass_loss_rate is not None or wind_velocity is not None

        n_routes = sum([has_A, has_A_star, has_physical])
        if n_routes == 0:
            raise ValueError(
                "One parameterization route must be supplied: `A` (Route 1), "
                "`A_star` (Route 2), or both `mass_loss_rate` and `wind_velocity` (Route 3)."
            )
        if n_routes > 1:
            raise ValueError(
                "Provide only one parameterization route: `A`, `A_star`, or (`mass_loss_rate` + `wind_velocity`)."
            )

        if has_physical:
            if mass_loss_rate is None or wind_velocity is None:
                raise ValueError("`mass_loss_rate` and `wind_velocity` must be supplied together (Route 3).")
            A = cls.normalize(mass_loss_rate=mass_loss_rate, wind_velocity=wind_velocity)
        elif has_A_star:
            A = cls.normalize_from_A_star(A_star)

        A_cgs = float(ensure_in_units(A, u.g / u.cm))

        if A_cgs <= 0:
            raise ValueError("Wind normalization `A` must be positive.")

        return {"A": A_cgs}

    @classmethod
    def _opt_eval(cls, r, _t=None, *, A, **_):
        return A * r**-2

    # ------------------------------------ #
    # Utility Methods                      #
    # ------------------------------------ #
    @classmethod
    def compute_mass_loss_rate(
        cls,
        A: "_UnitBearingScalarLike",
        wind_velocity: "_UnitBearingScalarLike",
    ) -> u.Quantity:
        r"""
        Compute the progenitor mass-loss rate from the wind-density normalization.

        Parameters
        ----------
        A : float or ~astropy.units.Quantity
            Wind-density normalization in :math:`\mathrm{g\,cm^{-1}}`. Unit-bearing
            inputs are converted to :math:`\mathrm{g\,cm^{-1}}`; bare values are
            interpreted as :math:`\mathrm{g\,cm^{-1}}`.
        wind_velocity : float or ~astropy.units.Quantity
            Wind velocity. Unit-bearing inputs are converted to
            :math:`\mathrm{cm\,s^{-1}}`; bare values are interpreted as
            :math:`\mathrm{cm\,s^{-1}}`.

        Returns
        -------
        mass_loss_rate : ~astropy.units.Quantity
            Progenitor mass-loss rate in :math:`\mathrm{g\,s^{-1}}`.
        """
        A = ensure_in_units(A, u.g / u.cm)
        v_w = ensure_in_units(wind_velocity, u.cm / u.s)

        if np.any(v_w <= 0):
            raise ValueError("`wind_velocity` must be positive.")

        return 4.0 * np.pi * A * v_w * (u.g / u.s)

    @classmethod
    def enclosed_mass(
        cls,
        r: "_UnitBearingArrayLike",
        *,
        A=None,
        A_star=None,
        mass_loss_rate=None,
        wind_velocity=None,
    ) -> u.Quantity:
        r"""
        Compute the total CSM mass enclosed within radius ``r``.

        For a wind density :math:`\rho = A r^{-2}`, the spherically enclosed
        mass is

        .. math::

            M(<r)
            = 4\pi \int_0^r A\,{r'}^{-2}\,{r'}^2\,\mathrm{d}r'
            = 4\pi A\,r.

        Parameters
        ----------
        r : float, array-like, or ~astropy.units.Quantity
            Outer bounding radius. Bare values interpreted as cm.
        A : float or ~astropy.units.Quantity, optional
            Wind normalization in :math:`\mathrm{g\,cm^{-1}}` (Route 1).
        A_star : float, optional
            Dimensionless wind-density parameter (Route 2).
        mass_loss_rate : float or ~astropy.units.Quantity, optional
            Mass-loss rate (Route 3). Must be supplied with ``wind_velocity``.
        wind_velocity : float or ~astropy.units.Quantity, optional
            Wind velocity (Route 3). Must be supplied with ``mass_loss_rate``.

        Returns
        -------
        mass : ~astropy.units.Quantity
            Enclosed CSM mass in grams.
        """
        params = cls._validate_and_process_parameters(
            A=A,
            A_star=A_star,
            mass_loss_rate=mass_loss_rate,
            wind_velocity=wind_velocity,
        )
        r_cgs = np.asarray(ensure_in_units(r, u.cm))
        return 4.0 * np.pi * params["A"] * r_cgs * u.g

    @classmethod
    def shell_mass(
        cls,
        r_inner: "_UnitBearingScalarLike",
        r_outer: "_UnitBearingScalarLike",
        *,
        A=None,
        A_star=None,
        mass_loss_rate=None,
        wind_velocity=None,
    ) -> u.Quantity:
        r"""
        Compute the total CSM mass in a radial shell.

        For a wind density :math:`\rho = A r^{-2}`, the shell mass between
        :math:`r_{\rm in}` and :math:`r_{\rm out}` is

        .. math::

            M(r_{\rm in}, r_{\rm out})
            = 4\pi A\,(r_{\rm out} - r_{\rm in}).

        Parameters
        ----------
        r_inner : float or ~astropy.units.Quantity
            Inner shell boundary. Bare values interpreted as cm.
        r_outer : float or ~astropy.units.Quantity
            Outer shell boundary. Bare values interpreted as cm. Must exceed
            ``r_inner``.
        A : float or ~astropy.units.Quantity, optional
            Wind normalization in :math:`\mathrm{g\,cm^{-1}}` (Route 1).
        A_star : float, optional
            Dimensionless wind-density parameter (Route 2).
        mass_loss_rate : float or ~astropy.units.Quantity, optional
            Mass-loss rate (Route 3). Must be supplied with ``wind_velocity``.
        wind_velocity : float or ~astropy.units.Quantity, optional
            Wind velocity (Route 3). Must be supplied with ``mass_loss_rate``.

        Returns
        -------
        mass : ~astropy.units.Quantity
            Shell mass in grams.

        Raises
        ------
        ValueError
            If ``r_outer`` does not exceed ``r_inner``.
        """
        params = cls._validate_and_process_parameters(
            A=A,
            A_star=A_star,
            mass_loss_rate=mass_loss_rate,
            wind_velocity=wind_velocity,
        )
        r_in = float(ensure_in_units(r_inner, u.cm))
        r_out = float(ensure_in_units(r_outer, u.cm))
        if r_out <= r_in:
            raise ValueError("`r_outer` must exceed `r_inner`.")
        return 4.0 * np.pi * params["A"] * (r_out - r_in) * u.g

    @classmethod
    def column_density(
        cls,
        r: "_UnitBearingArrayLike",
        *,
        A=None,
        A_star=None,
        mass_loss_rate=None,
        wind_velocity=None,
    ) -> u.Quantity:
        r"""
        Compute the radial column density from ``r`` to infinity.

        For a wind density :math:`\rho = A r^{-2}`, the semi-infinite column is

        .. math::

            \Sigma(r)
            = \int_r^\infty A\,{r'}^{-2}\,\mathrm{d}r'
            = \frac{A}{r}.

        Parameters
        ----------
        r : float, array-like, or ~astropy.units.Quantity
            Lower integration limit. Bare values interpreted as cm.
        A : float or ~astropy.units.Quantity, optional
            Wind normalization in :math:`\mathrm{g\,cm^{-1}}` (Route 1).
        A_star : float, optional
            Dimensionless wind-density parameter (Route 2).
        mass_loss_rate : float or ~astropy.units.Quantity, optional
            Mass-loss rate (Route 3). Must be supplied with ``wind_velocity``.
        wind_velocity : float or ~astropy.units.Quantity, optional
            Wind velocity (Route 3). Must be supplied with ``mass_loss_rate``.

        Returns
        -------
        column : ~astropy.units.Quantity
            Column density in :math:`\mathrm{g\,cm^{-2}}`.
        """
        params = cls._validate_and_process_parameters(
            A=A,
            A_star=A_star,
            mass_loss_rate=mass_loss_rate,
            wind_velocity=wind_velocity,
        )
        r_cgs = np.asarray(ensure_in_units(r, u.cm))
        return params["A"] / r_cgs * (u.g / u.cm**2)


class UniformCSMProfile(StationaryCSMDensityProfile):
    r"""
    Uniform CSM density profile.

    The density is constant everywhere,

    .. math::

        \rho_{\mathrm{CSM}}(r) = \rho_0.

    .. rubric:: Physical relevance

    A uniform (constant-density) CSM describes a pre-existing, well-mixed
    interstellar medium or the interior of an old stellar wind bubble.  It
    enters the Chevalier self-similar solution as the :math:`s = 0` case and
    is the standard upstream medium assumed in GRB afterglow modelling (the
    ISM scenario of Sari et al. 1998).  Typical ISM densities range from
    :math:`n \sim 10^{-3}` to :math:`1\ \mathrm{cm}^{-3}`, or
    :math:`\rho_0 \sim 10^{-27}` to :math:`10^{-24}\ \mathrm{g\,cm^{-3}}`.

    .. rubric:: Parameters

    .. list-table::
       :header-rows: 1
       :widths: 22 6 20 52

       * - Parameter
         - Opt.
         - Type
         - Description
       * - ``rho_0``
         - ✓
         - float or :class:`~astropy.units.Quantity`
         - Uniform CSM density in :math:`\mathrm{g\,cm^{-3}}`. Bare values
           interpreted as :math:`\mathrm{g\,cm^{-3}}`. Must be non-negative.

    .. rubric:: Example

    .. plot::

       import numpy as np
       import astropy.units as u
       import matplotlib.pyplot as plt
       from trilobite.dynamics.profiles.csm import UniformCSMProfile
       from trilobite.utils.plot_utils import set_plot_style

       set_plot_style()
       r = np.geomspace(1e14, 1e18, 300) * u.cm
       fig, ax = plt.subplots()
       for rho0, ls in [(1e-24, '--'), (1e-23, '-'), (1e-22, ':')]:
           rho = UniformCSMProfile.eval(r, rho_0=rho0)
           ax.loglog(r.value, rho.value, ls=ls,
                     label=fr'$\rho_0 = 10^{{{int(np.log10(rho0))}}}\ \mathrm{{g\,cm^{{-3}}}}$')
       ax.set_xlabel(r'$r\ [\mathrm{cm}]$')
       ax.set_ylabel(r'$\rho\ [\mathrm{g\,cm^{-3}}]$')
       ax.set_title('Uniform CSM Profile')
       ax.legend()
       plt.tight_layout()
    """

    @classmethod
    def _validate_and_process_parameters(cls, *, rho_0, **_):
        rho = float(ensure_in_units(rho_0, u.g / u.cm**3))
        if rho < 0:
            raise ValueError("`rho_0` must be non-negative.")
        return {"rho_0": rho}

    @classmethod
    def _opt_eval(cls, r, _t=None, *, rho_0, **_):
        r_arr = np.asarray(r)
        if r_arr.ndim == 0:
            return rho_0
        return np.full_like(r_arr, rho_0, dtype=float)


class PowerLawCSMProfile(StationaryCSMDensityProfile):
    r"""
    Power-law CSM density profile.

    The density follows a single power law anchored at a reference radius,

    .. math::

        \rho_{\mathrm{CSM}}(r)
        = \rho_{\rm ref}
          \left(\frac{r}{r_{\rm ref}}\right)^{-s}.

    Special cases recover the wind (:math:`s = 2`) and uniform (:math:`s = 0`)
    profiles.

    .. rubric:: Physical relevance

    The power-law parameterisation generalises the standard wind and ISM CSM
    models and is the appropriate choice when the radial density gradient is
    constrained observationally but does not necessarily equal :math:`s = 2`.
    It is used in parametric shock-dynamics studies and in fitting the
    broadband spectral evolution of Type Ib/c and IIb supernovae, where
    :math:`s \approx 1.5`–:math:`2.5` is inferred from radio light curves.

    .. rubric:: Parameters

    .. list-table::
       :header-rows: 1
       :widths: 22 6 20 52

       * - Parameter
         - Opt.
         - Type
         - Description
       * - ``rho_ref``
         - ✓
         - float or :class:`~astropy.units.Quantity`
         - Density at the reference radius in :math:`\mathrm{g\,cm^{-3}}`.
           Bare values interpreted as :math:`\mathrm{g\,cm^{-3}}`. Must be
           non-negative.
       * - ``r_ref``
         - ✓
         - float or :class:`~astropy.units.Quantity`
         - Reference radius. Bare values interpreted as cm. Must be positive.
       * - ``slope``
         - ✓
         - float
         - Power-law slope :math:`s`.  Positive values give outward-declining
           density.

    .. rubric:: Example

    .. plot::

       import numpy as np
       import astropy.units as u
       import matplotlib.pyplot as plt
       from trilobite.dynamics.profiles.csm import PowerLawCSMProfile
       from trilobite.utils.plot_utils import set_plot_style

       set_plot_style()
       r = np.geomspace(1e14, 1e18, 300) * u.cm
       r_ref = 1e16 * u.cm
       rho_ref = 1e-18 * u.g / u.cm**3
       fig, ax = plt.subplots()
       for s, ls in [(0.0, ':'), (1.0, '--'), (2.0, '-'), (3.0, '-.')]:
           rho = PowerLawCSMProfile.eval(r, rho_ref=rho_ref, r_ref=r_ref, slope=s)
           ax.loglog(r.value, rho.value, ls=ls, label=fr'$s={s}$')
       ax.set_xlabel(r'$r\ [\mathrm{cm}]$')
       ax.set_ylabel(r'$\rho\ [\mathrm{g\,cm^{-3}}]$')
       ax.set_title('Power-Law CSM Profile')
       ax.legend()
       plt.tight_layout()
    """

    @classmethod
    def _validate_and_process_parameters(cls, *, rho_ref, r_ref, slope, **_):
        rho = float(ensure_in_units(rho_ref, u.g / u.cm**3))
        r = float(ensure_in_units(r_ref, u.cm))
        s = float(slope)

        if rho < 0:
            raise ValueError("`rho_ref` must be non-negative.")
        if r <= 0:
            raise ValueError("`r_ref` must be positive.")

        return {"rho_ref": rho, "r_ref": r, "slope": s}

    @classmethod
    def _opt_eval(cls, r, _t=None, *, rho_ref, r_ref, slope, **_):
        return rho_ref * (r / r_ref) ** (-slope)


class ShellCSMProfile(StationaryCSMDensityProfile):
    r"""
    Top-hat shell CSM density profile.

    The density is elevated inside a radial shell and falls to a background
    floor outside it,

    .. math::

        \rho_{\mathrm{CSM}}(r)
        =
        \begin{cases}
            \rho_{\rm shell}, & r_{\rm in} \le r \le r_{\rm out}, \\
            \rho_{\rm floor}, & \text{otherwise}.
        \end{cases}

    .. rubric:: Physical relevance

    A top-hat shell is the simplest model for CSM produced by a brief,
    intense mass-loss episode — an LBV giant eruption, a common-envelope
    ejection, or pulsational pair-instability mass shedding.  It is
    appropriate when the mass-loss duration is short compared with the time
    between ejection and shock interaction, so that the radial shell thickness
    is well-defined.  Type IIn supernovae such as SN 2006gy and SN 2009ip
    show clear signatures of interaction with pre-formed dense shells.

    .. rubric:: Parameters

    .. list-table::
       :header-rows: 1
       :widths: 22 6 20 52

       * - Parameter
         - Opt.
         - Type
         - Description
       * - ``r_inner``
         - ✓
         - float or :class:`~astropy.units.Quantity`
         - Inner shell boundary. Bare values interpreted as cm. Must be
           non-negative.
       * - ``r_outer``
         - ✓
         - float or :class:`~astropy.units.Quantity`
         - Outer shell boundary. Bare values interpreted as cm. Must exceed
           ``r_inner``.
       * - ``shell_density``
         - ✓
         - float or :class:`~astropy.units.Quantity`
         - Density inside the shell in :math:`\mathrm{g\,cm^{-3}}`. Must be
           non-negative.
       * - ``density_floor``
         - ✓
         - float or :class:`~astropy.units.Quantity`
         - Density outside the shell in :math:`\mathrm{g\,cm^{-3}}`. Bare
           values interpreted as :math:`\mathrm{g\,cm^{-3}}`. Defaults to 0.

    .. rubric:: Example

    .. plot::

       import numpy as np
       import astropy.units as u
       import matplotlib.pyplot as plt
       from trilobite.dynamics.profiles.csm import ShellCSMProfile
       from trilobite.utils.plot_utils import set_plot_style

       set_plot_style()
       r = np.geomspace(1e14, 1e18, 600) * u.cm
       rho = ShellCSMProfile.eval(
           r,
           r_inner=3e15 * u.cm,
           r_outer=1e16 * u.cm,
           shell_density=1e-18 * u.g / u.cm**3,
           density_floor=1e-23 * u.g / u.cm**3,
       )
       fig, ax = plt.subplots()
       ax.loglog(r.value, rho.value)
       ax.set_xlabel(r'$r\ [\mathrm{cm}]$')
       ax.set_ylabel(r'$\rho\ [\mathrm{g\,cm^{-3}}]$')
       ax.set_title('Top-Hat Shell CSM Profile')
       plt.tight_layout()
    """

    @classmethod
    def _validate_and_process_parameters(cls, *, r_inner, r_outer, shell_density, density_floor=0.0, **_):
        r_in = float(ensure_in_units(r_inner, u.cm))
        r_out = float(ensure_in_units(r_outer, u.cm))
        rho_s = float(ensure_in_units(shell_density, u.g / u.cm**3))
        rho_f = float(ensure_in_units(density_floor, u.g / u.cm**3))

        if r_in < 0:
            raise ValueError("`r_inner` must be non-negative.")
        if r_out <= r_in:
            raise ValueError("`r_outer` must be greater than `r_inner`.")
        if rho_s < 0:
            raise ValueError("`shell_density` must be non-negative.")
        if rho_f < 0:
            raise ValueError("`density_floor` must be non-negative.")

        return {
            "r_inner": r_in,
            "r_outer": r_out,
            "shell_density": rho_s,
            "density_floor": rho_f,
        }

    @classmethod
    def _opt_eval(cls, r, _t=None, *, r_inner, r_outer, shell_density, density_floor, **_):
        inside = (r >= r_inner) & (r <= r_outer)
        return np.where(inside, shell_density, density_floor)


class GaussianShellCSMProfile(StationaryCSMDensityProfile):
    r"""
    Smooth Gaussian-shell CSM density profile.

    The profile superimposes a Gaussian density excess on a uniform background,

    .. math::

        \rho_{\mathrm{CSM}}(r)
        =
        \rho_{\rm bg}
        +
        \rho_{\rm shell}
        \exp\!\left[
            -\frac{1}{2}
            \left(\frac{r - R_{\rm shell}}{\sigma}\right)^2
        \right].

    The total density at the shell peak is
    :math:`\rho_{\rm bg} + \rho_{\rm shell}`.

    .. rubric:: Physical relevance

    The Gaussian shell is a smooth alternative to the top-hat
    :class:`ShellCSMProfile` and is appropriate when the ejection event has
    a finite duration or when the shell has been broadened by internal
    pressure gradients.  It avoids the density discontinuities that can cause
    numerical noise in shock solvers, and is well-suited to fitting
    broadband light curves of interacting supernovae where the shock
    interaction timescale smears out sharp shell features.

    .. rubric:: Parameters

    .. list-table::
       :header-rows: 1
       :widths: 22 6 20 52

       * - Parameter
         - Opt.
         - Type
         - Description
       * - ``background_density``
         - ✓
         - float or :class:`~astropy.units.Quantity`
         - Uniform background CSM density in :math:`\mathrm{g\,cm^{-3}}`.
           Must be non-negative.
       * - ``shell_density``
         - ✓
         - float or :class:`~astropy.units.Quantity`
         - Peak density excess above the background in
           :math:`\mathrm{g\,cm^{-3}}`. Must be non-negative.
       * - ``shell_radius``
         - ✓
         - float or :class:`~astropy.units.Quantity`
         - Shell centre radius :math:`R_{\rm shell}`. Bare values interpreted
           as cm. Must be non-negative.
       * - ``shell_width``
         - ✓
         - float or :class:`~astropy.units.Quantity`
         - Gaussian width :math:`\sigma`. Bare values interpreted as cm.
           Must be positive.

    .. rubric:: Example

    .. plot::

       import numpy as np
       import astropy.units as u
       import matplotlib.pyplot as plt
       from trilobite.dynamics.profiles.csm import GaussianShellCSMProfile
       from trilobite.utils.plot_utils import set_plot_style

       set_plot_style()
       r = np.geomspace(1e14, 1e18, 500) * u.cm
       rho = GaussianShellCSMProfile.eval(
           r,
           background_density=1e-23 * u.g / u.cm**3,
           shell_density=5e-19 * u.g / u.cm**3,
           shell_radius=1e16 * u.cm,
           shell_width=2e15 * u.cm,
       )
       fig, ax = plt.subplots()
       ax.loglog(r.value, rho.value)
       ax.set_xlabel(r'$r\ [\mathrm{cm}]$')
       ax.set_ylabel(r'$\rho\ [\mathrm{g\,cm^{-3}}]$')
       ax.set_title('Gaussian Shell CSM Profile')
       plt.tight_layout()
    """

    @classmethod
    def _validate_and_process_parameters(cls, *, background_density, shell_density, shell_radius, shell_width, **_):
        rho_bg = float(ensure_in_units(background_density, u.g / u.cm**3))
        rho_s = float(ensure_in_units(shell_density, u.g / u.cm**3))
        r_s = float(ensure_in_units(shell_radius, u.cm))
        sigma = float(ensure_in_units(shell_width, u.cm))

        if rho_bg < 0:
            raise ValueError("`background_density` must be non-negative.")
        if rho_s < 0:
            raise ValueError("`shell_density` must be non-negative.")
        if r_s < 0:
            raise ValueError("`shell_radius` must be non-negative.")
        if sigma <= 0:
            raise ValueError("`shell_width` must be positive.")

        return {
            "background_density": rho_bg,
            "shell_density": rho_s,
            "shell_radius": r_s,
            "shell_width": sigma,
        }

    @classmethod
    def _opt_eval(cls, r, _t=None, *, background_density, shell_density, shell_radius, shell_width, **_):
        x = (r - shell_radius) / shell_width
        return background_density + shell_density * np.exp(-0.5 * x**2)


class BrokenPowerLawCSMProfile(StationaryCSMDensityProfile):
    r"""
    Continuous broken-power-law CSM density profile.

    The profile transitions sharply between two power-law slopes at a break
    radius while remaining continuous at the break,

    .. math::

        \rho_{\mathrm{CSM}}(r)
        =
        \rho_b
        \begin{cases}
            (r/r_b)^{-s_{\rm in}}, & r < r_b, \\
            (r/r_b)^{-s_{\rm out}}, & r \ge r_b.
        \end{cases}

    The density is continuous at :math:`r_b` by construction
    (:math:`\rho(r_b) = \rho_b`), but the slope is discontinuous.

    .. rubric:: Physical relevance

    The broken power law describes the transition from an inner wind-like
    region (:math:`s_{\rm in} = 2`) to an outer region with a different slope,
    such as the interior of a wind-blown bubble (:math:`s_{\rm out} \approx 0`)
    or a pre-existing slower wind (:math:`s_{\rm out} \approx 1`).  It models
    the CSM produced by a progenitor that underwent a change in mass-loss rate
    or wind velocity at some epoch corresponding to :math:`r_b`, and is used
    to fit radio and X-ray light curves of Type Ib/c supernovae that show
    evidence for a CSM density transition.

    .. rubric:: Parameters

    .. list-table::
       :header-rows: 1
       :widths: 22 6 20 52

       * - Parameter
         - Opt.
         - Type
         - Description
       * - ``density_break``
         - ✓
         - float or :class:`~astropy.units.Quantity`
         - Density at the break radius in :math:`\mathrm{g\,cm^{-3}}`. Must
           be non-negative.
       * - ``radius_break``
         - ✓
         - float or :class:`~astropy.units.Quantity`
         - Break radius. Bare values interpreted as cm. Must be positive.
       * - ``slope_inner``
         - ✓
         - float
         - Inner power-law slope :math:`s_{\rm in}` (applied for
           :math:`r < r_b`).
       * - ``slope_outer``
         - ✓
         - float
         - Outer power-law slope :math:`s_{\rm out}` (applied for
           :math:`r \ge r_b`).

    .. rubric:: Example

    .. plot::

       import numpy as np
       import astropy.units as u
       import matplotlib.pyplot as plt
       from trilobite.dynamics.profiles.csm import BrokenPowerLawCSMProfile
       from trilobite.utils.plot_utils import set_plot_style

       set_plot_style()
       r = np.geomspace(1e14, 1e18, 400) * u.cm
       rho = BrokenPowerLawCSMProfile.eval(
           r,
           density_break=1e-18 * u.g / u.cm**3,
           radius_break=1e16 * u.cm,
           slope_inner=2.0,
           slope_outer=0.0,
       )
       fig, ax = plt.subplots()
       ax.loglog(r.value, rho.value)
       ax.axvline(1e16, ls='--', color='gray', label=r'$r_b$')
       ax.set_xlabel(r'$r\ [\mathrm{cm}]$')
       ax.set_ylabel(r'$\rho\ [\mathrm{g\,cm^{-3}}]$')
       ax.set_title('Broken Power-Law CSM Profile')
       ax.legend()
       plt.tight_layout()
    """

    @classmethod
    def _validate_and_process_parameters(cls, *, density_break, radius_break, slope_inner, slope_outer, **_):
        rho_b = float(ensure_in_units(density_break, u.g / u.cm**3))
        r_b = float(ensure_in_units(radius_break, u.cm))

        if rho_b < 0:
            raise ValueError("`density_break` must be non-negative.")
        if r_b <= 0:
            raise ValueError("`radius_break` must be positive.")

        return {
            "density_break": rho_b,
            "radius_break": r_b,
            "slope_inner": float(slope_inner),
            "slope_outer": float(slope_outer),
        }

    @classmethod
    def _opt_eval(cls, r, _t=None, *, density_break, radius_break, slope_inner, slope_outer, **_):
        x = r / radius_break
        return np.where(
            r < radius_break,
            density_break * x ** (-slope_inner),
            density_break * x ** (-slope_outer),
        )


class TruncatedWindCSMProfile(StationaryCSMDensityProfile):
    r"""
    Sharply outer-truncated wind CSM density profile.

    The profile follows a steady wind inside a maximum radius and transitions
    discontinuously to a constant density floor outside,

    .. math::

        \rho_{\mathrm{CSM}}(r)
        =
        \begin{cases}
            A\,r^{-2},       & r \le r_{\max}, \\
            \rho_{\rm floor}, & r > r_{\max}.
        \end{cases}

    .. rubric:: Physical relevance

    The truncated wind describes a progenitor whose mass loss ceased or
    slowed sharply at some time before the explosion, leaving a wind region
    extending out to :math:`r_{\max} = v_w\,t_{\rm stop}` and a lower ambient
    density beyond.  It is appropriate for Wolf-Rayet stars in which the
    stellar wind ceases in the final years before core collapse, and for
    binary interaction scenarios in which the onset of mass transfer truncates
    the pre-existing wind.  The sharp truncation avoids the additional free
    parameter of a transition scale, at the cost of a density discontinuity
    that may require special handling in shock solvers.

    .. rubric:: Parameters

    .. list-table::
       :header-rows: 1
       :widths: 22 6 20 52

       * - Parameter
         - Opt.
         - Type
         - Description
       * - ``A``
         - ✓
         - float or :class:`~astropy.units.Quantity`
         - Wind-density normalization in :math:`\mathrm{g\,cm^{-1}}`. Bare
           values interpreted as :math:`\mathrm{g\,cm^{-1}}`. Must be
           positive. Use :meth:`normalize` to compute from mass-loss rate and
           wind speed.
       * - ``r_max``
         - ✓
         - float or :class:`~astropy.units.Quantity`
         - Outer wind radius. Bare values interpreted as cm. Must be positive.
       * - ``density_floor``
         - ✓
         - float or :class:`~astropy.units.Quantity`
         - Density returned for :math:`r > r_{\max}` in
           :math:`\mathrm{g\,cm^{-3}}`. Defaults to 0.

    .. rubric:: Example

    .. plot::

       import numpy as np
       import astropy.units as u
       import matplotlib.pyplot as plt
       from trilobite.dynamics.profiles.csm import TruncatedWindCSMProfile
       from trilobite.utils.plot_utils import set_plot_style

       set_plot_style()
       r = np.geomspace(1e14, 1e18, 500) * u.cm
       r_max = 3e16 * u.cm
       rho = TruncatedWindCSMProfile.eval(
           r,
           A=5e11 * u.g / u.cm,
           r_max=r_max,
           density_floor=1e-23 * u.g / u.cm**3,
       )
       fig, ax = plt.subplots()
       ax.loglog(r.value, rho.value)
       ax.axvline(r_max.value, ls='--', color='gray', label=r'$r_{\max}$')
       ax.set_xlabel(r'$r\ [\mathrm{cm}]$')
       ax.set_ylabel(r'$\rho\ [\mathrm{g\,cm^{-3}}]$')
       ax.set_title('Truncated Wind CSM Profile')
       ax.legend()
       plt.tight_layout()

    See Also
    --------
    TruncatedWindCSMProfile.normalize : Compute ``A`` from mass-loss rate and wind speed.
    """

    @classmethod
    def normalize(
        cls,
        mass_loss_rate: "_UnitBearingScalarLike",
        wind_velocity: "_UnitBearingScalarLike",
    ) -> u.Quantity:
        r"""
        Compute the wind-density normalization :math:`A = \dot{M}/(4\pi v_w)`.

        Parameters
        ----------
        mass_loss_rate : float or ~astropy.units.Quantity
            Progenitor mass-loss rate. Bare values interpreted as g/s.
        wind_velocity : float or ~astropy.units.Quantity
            Wind velocity. Bare values interpreted as cm/s.

        Returns
        -------
        A : ~astropy.units.Quantity
            Wind-density normalization in :math:`\mathrm{g\,cm^{-1}}`.
        """
        return WindCSMProfile.normalize(mass_loss_rate, wind_velocity)

    @classmethod
    def _validate_and_process_parameters(
        cls, *, A=None, A_star=None, mass_loss_rate=None, wind_velocity=None, r_max, density_floor=0.0, **_
    ):
        has_A = A is not None
        has_A_star = A_star is not None
        has_physical = mass_loss_rate is not None or wind_velocity is not None
        n_routes = sum([has_A, has_A_star, has_physical])
        if n_routes == 0:
            raise ValueError(
                "One parameterization route must be supplied: `A` (Route 1), "
                "`A_star` (Route 2), or both `mass_loss_rate` and `wind_velocity` (Route 3)."
            )
        if n_routes > 1:
            raise ValueError(
                "Provide only one parameterization route: `A`, `A_star`, or (`mass_loss_rate` + `wind_velocity`)."
            )
        if has_physical:
            if mass_loss_rate is None or wind_velocity is None:
                raise ValueError("`mass_loss_rate` and `wind_velocity` must be supplied together (Route 3).")
            A = cls.normalize(mass_loss_rate=mass_loss_rate, wind_velocity=wind_velocity)
        elif has_A_star:
            A = WindCSMProfile.normalize_from_A_star(A_star)

        A_cgs = float(ensure_in_units(A, u.g / u.cm))
        r_max_cgs = float(ensure_in_units(r_max, u.cm))
        rho_f = float(ensure_in_units(density_floor, u.g / u.cm**3))

        if A_cgs <= 0:
            raise ValueError("`A` must be positive.")
        if r_max_cgs <= 0:
            raise ValueError("`r_max` must be positive.")
        if rho_f < 0:
            raise ValueError("`density_floor` must be non-negative.")

        return {"A": A_cgs, "r_max": r_max_cgs, "density_floor": rho_f}

    @classmethod
    def _opt_eval(cls, r, _t=None, *, A, r_max, density_floor, **_):
        rho_wind = A * r**-2
        return np.where(r <= r_max, rho_wind, density_floor)


class WindWithFloorCSMProfile(StationaryCSMDensityProfile):
    r"""
    Wind CSM density profile with a constant background floor.

    The wind density and a constant ambient density are summed,

    .. math::

        \rho_{\mathrm{CSM}}(r) = A\,r^{-2} + \rho_{\rm floor}.

    At large radii, where :math:`A r^{-2} \ll \rho_{\rm floor}`, the profile
    approaches the floor value.  The transition radius is
    :math:`r_t = \sqrt{A / \rho_{\rm floor}}`.

    .. rubric:: Physical relevance

    A wind superimposed on an ambient ISM is physically appropriate at all
    radii: the wind density eventually falls below the ambient level far from
    the progenitor.  This profile avoids the singularity at large :math:`r`
    inherent in a pure wind model and is more realistic than the
    :class:`TruncatedWindCSMProfile` when the ambient medium is non-negligible
    within the dynamical range of interest.  It is also useful when fitting
    multi-epoch radio observations of core-collapse supernovae whose shock
    velocities span a large range of radii.

    .. rubric:: Parameters

    .. list-table::
       :header-rows: 1
       :widths: 22 6 20 52

       * - Parameter
         - Opt.
         - Type
         - Description
       * - ``A``
         - ✓
         - float or :class:`~astropy.units.Quantity`
         - Wind-density normalization in :math:`\mathrm{g\,cm^{-1}}`. Bare
           values interpreted as :math:`\mathrm{g\,cm^{-1}}`. Must be
           positive. Use :meth:`normalize` to compute from mass-loss rate and
           wind speed.
       * - ``density_floor``
         - ✓
         - float or :class:`~astropy.units.Quantity`
         - Constant background density in :math:`\mathrm{g\,cm^{-3}}`. Must
           be non-negative.

    .. rubric:: Example

    .. plot::

       import numpy as np
       import astropy.units as u
       import matplotlib.pyplot as plt
       from trilobite.dynamics.profiles.csm import WindWithFloorCSMProfile
       from trilobite.utils.plot_utils import set_plot_style

       set_plot_style()
       r = np.geomspace(1e14, 1e18, 400) * u.cm
       rho = WindWithFloorCSMProfile.eval(
           r,
           A=5e11 * u.g / u.cm,
           density_floor=1e-23 * u.g / u.cm**3,
       )
       fig, ax = plt.subplots()
       ax.loglog(r.value, rho.value)
       ax.set_xlabel(r'$r\ [\mathrm{cm}]$')
       ax.set_ylabel(r'$\rho\ [\mathrm{g\,cm^{-3}}]$')
       ax.set_title('Wind + Floor CSM Profile')
       plt.tight_layout()

    See Also
    --------
    WindWithFloorCSMProfile.normalize : Compute ``A`` from mass-loss rate and wind speed.
    """

    @classmethod
    def normalize(
        cls,
        mass_loss_rate: "_UnitBearingScalarLike",
        wind_velocity: "_UnitBearingScalarLike",
    ) -> u.Quantity:
        r"""
        Compute the wind-density normalization :math:`A = \dot{M}/(4\pi v_w)`.

        Parameters
        ----------
        mass_loss_rate : float or ~astropy.units.Quantity
            Progenitor mass-loss rate. Bare values interpreted as g/s.
        wind_velocity : float or ~astropy.units.Quantity
            Wind velocity. Bare values interpreted as cm/s.

        Returns
        -------
        A : ~astropy.units.Quantity
            Wind-density normalization in :math:`\mathrm{g\,cm^{-1}}`.
        """
        return WindCSMProfile.normalize(mass_loss_rate, wind_velocity)

    @classmethod
    def _validate_and_process_parameters(
        cls, *, A=None, A_star=None, mass_loss_rate=None, wind_velocity=None, density_floor, **_
    ):
        has_A = A is not None
        has_A_star = A_star is not None
        has_physical = mass_loss_rate is not None or wind_velocity is not None
        n_routes = sum([has_A, has_A_star, has_physical])
        if n_routes == 0:
            raise ValueError(
                "One parameterization route must be supplied: `A` (Route 1), "
                "`A_star` (Route 2), or both `mass_loss_rate` and `wind_velocity` (Route 3)."
            )
        if n_routes > 1:
            raise ValueError(
                "Provide only one parameterization route: `A`, `A_star`, or (`mass_loss_rate` + `wind_velocity`)."
            )
        if has_physical:
            if mass_loss_rate is None or wind_velocity is None:
                raise ValueError("`mass_loss_rate` and `wind_velocity` must be supplied together (Route 3).")
            A = cls.normalize(mass_loss_rate=mass_loss_rate, wind_velocity=wind_velocity)
        elif has_A_star:
            A = WindCSMProfile.normalize_from_A_star(A_star)

        A_cgs = float(ensure_in_units(A, u.g / u.cm))
        rho_f = float(ensure_in_units(density_floor, u.g / u.cm**3))

        if A_cgs <= 0:
            raise ValueError("`A` must be positive.")
        if rho_f < 0:
            raise ValueError("`density_floor` must be non-negative.")

        return {"A": A_cgs, "density_floor": rho_f}

    @classmethod
    def _opt_eval(cls, r, _t=None, *, A, density_floor, **_):
        return A * r**-2 + density_floor


class SmoothTruncatedWindCSMProfile(StationaryCSMDensityProfile):
    r"""
    Smoothly outer-truncated wind CSM density profile.

    A hyperbolic-tangent envelope transitions the wind density to an ambient
    floor,

    .. math::

        \rho_{\mathrm{CSM}}(r)
        =
        \rho_{\rm floor}
        +
        \left(A r^{-2} - \rho_{\rm floor}\right)
        \frac{1}{2}
        \left[
            1 - \tanh\!\left(\frac{r - r_{\max}}{\Delta r}\right)
        \right].

    At :math:`r = r_{\max}` the density is exactly
    :math:`(A r_{\max}^{-2} + \rho_{\rm floor})/2`; the transition is
    complete over a scale :math:`\Delta r`.

    .. rubric:: Physical relevance

    The smooth truncation is the preferred alternative to
    :class:`TruncatedWindCSMProfile` when density discontinuities cause
    numerical noise in shock solvers.  The tanh profile captures the physical
    reality that the wind does not stop instantaneously, and allows the
    solver to maintain strict conservation across the transition.  The free
    parameter :math:`\Delta r` encodes the radial range over which the
    mass-loss rate declined; choosing
    :math:`\Delta r \ll r_{\max}` recovers the sharp truncation.

    .. rubric:: Parameters

    .. list-table::
       :header-rows: 1
       :widths: 22 6 20 52

       * - Parameter
         - Opt.
         - Type
         - Description
       * - ``A``
         - ✓
         - float or :class:`~astropy.units.Quantity`
         - Wind-density normalization in :math:`\mathrm{g\,cm^{-1}}`. Bare
           values interpreted as :math:`\mathrm{g\,cm^{-1}}`. Must be
           positive. Use :meth:`normalize` to compute from mass-loss rate and
           wind speed.
       * - ``r_max``
         - ✓
         - float or :class:`~astropy.units.Quantity`
         - Characteristic outer wind radius. Bare values interpreted as cm.
           Must be positive.
       * - ``transition_width``
         - ✓
         - float or :class:`~astropy.units.Quantity`
         - Width :math:`\Delta r` of the smooth transition. Bare values
           interpreted as cm. Must be positive.
       * - ``density_floor``
         - ✓
         - float or :class:`~astropy.units.Quantity`
         - Asymptotic density well outside the wind region in
           :math:`\mathrm{g\,cm^{-3}}`. Defaults to 0.

    .. rubric:: Example

    .. plot::

       import numpy as np
       import astropy.units as u
       import matplotlib.pyplot as plt
       from trilobite.dynamics.profiles.csm import (
           SmoothTruncatedWindCSMProfile, TruncatedWindCSMProfile,
       )
       from trilobite.utils.plot_utils import set_plot_style

       set_plot_style()
       r = np.geomspace(1e14, 1e18, 500) * u.cm
       A = 5e11 * u.g / u.cm
       r_max = 3e16 * u.cm
       rho_floor = 1e-23 * u.g / u.cm**3

       rho_sharp = TruncatedWindCSMProfile.eval(
           r, A=A, r_max=r_max, density_floor=rho_floor,
       )
       rho_smooth = SmoothTruncatedWindCSMProfile.eval(
           r, A=A, r_max=r_max, transition_width=5e15 * u.cm, density_floor=rho_floor,
       )
       fig, ax = plt.subplots()
       ax.loglog(r.value, rho_sharp.value, ls='--', label='Sharp truncation')
       ax.loglog(r.value, rho_smooth.value, label='Smooth truncation')
       ax.axvline(r_max.value, ls=':', color='gray', label=r'$r_{\max}$')
       ax.set_xlabel(r'$r\ [\mathrm{cm}]$')
       ax.set_ylabel(r'$\rho\ [\mathrm{g\,cm^{-3}}]$')
       ax.set_title('Smooth vs. Sharp Truncated Wind')
       ax.legend()
       plt.tight_layout()

    See Also
    --------
    SmoothTruncatedWindCSMProfile.normalize : Compute ``A`` from mass-loss rate and wind speed.
    TruncatedWindCSMProfile : Sharply truncated variant of this profile.
    """

    @classmethod
    def normalize(
        cls,
        mass_loss_rate: "_UnitBearingScalarLike",
        wind_velocity: "_UnitBearingScalarLike",
    ) -> u.Quantity:
        r"""
        Compute the wind-density normalization :math:`A = \dot{M}/(4\pi v_w)`.

        Parameters
        ----------
        mass_loss_rate : float or ~astropy.units.Quantity
            Progenitor mass-loss rate. Bare values interpreted as g/s.
        wind_velocity : float or ~astropy.units.Quantity
            Wind velocity. Bare values interpreted as cm/s.

        Returns
        -------
        A : ~astropy.units.Quantity
            Wind-density normalization in :math:`\mathrm{g\,cm^{-1}}`.
        """
        return WindCSMProfile.normalize(mass_loss_rate, wind_velocity)

    @classmethod
    def _validate_and_process_parameters(
        cls,
        *,
        A=None,
        A_star=None,
        mass_loss_rate=None,
        wind_velocity=None,
        r_max,
        transition_width,
        density_floor=0.0,
        **_,
    ):
        has_A = A is not None
        has_A_star = A_star is not None
        has_physical = mass_loss_rate is not None or wind_velocity is not None
        n_routes = sum([has_A, has_A_star, has_physical])
        if n_routes == 0:
            raise ValueError(
                "One parameterization route must be supplied: `A` (Route 1), "
                "`A_star` (Route 2), or both `mass_loss_rate` and `wind_velocity` (Route 3)."
            )
        if n_routes > 1:
            raise ValueError(
                "Provide only one parameterization route: `A`, `A_star`, or (`mass_loss_rate` + `wind_velocity`)."
            )
        if has_physical:
            if mass_loss_rate is None or wind_velocity is None:
                raise ValueError("`mass_loss_rate` and `wind_velocity` must be supplied together (Route 3).")
            A = cls.normalize(mass_loss_rate=mass_loss_rate, wind_velocity=wind_velocity)
        elif has_A_star:
            A = WindCSMProfile.normalize_from_A_star(A_star)

        A_cgs = float(ensure_in_units(A, u.g / u.cm))
        r_max_cgs = float(ensure_in_units(r_max, u.cm))
        dr_cgs = float(ensure_in_units(transition_width, u.cm))
        rho_f = float(ensure_in_units(density_floor, u.g / u.cm**3))

        if A_cgs <= 0:
            raise ValueError("`A` must be positive.")
        if r_max_cgs <= 0:
            raise ValueError("`r_max` must be positive.")
        if dr_cgs <= 0:
            raise ValueError("`transition_width` must be positive.")
        if rho_f < 0:
            raise ValueError("`density_floor` must be non-negative.")

        return {
            "A": A_cgs,
            "r_max": r_max_cgs,
            "transition_width": dr_cgs,
            "density_floor": rho_f,
        }

    @classmethod
    def _opt_eval(cls, r, _t=None, *, A, r_max, transition_width, density_floor, **_):
        rho_wind = A * r**-2
        switch = 0.5 * (1.0 - np.tanh((r - r_max) / transition_width))
        return density_floor + (rho_wind - density_floor) * switch


class ExponentialCSMProfile(StationaryCSMDensityProfile):
    r"""
    Exponential CSM density profile.

    The density falls off exponentially outward from a reference radius on top
    of a constant background,

    .. math::

        \rho_{\mathrm{CSM}}(r)
        =
        \rho_{\rm floor}
        +
        \rho_0
        \exp\!\left(-\frac{r - r_0}{H}\right).

    For :math:`r < r_0` the profile rises above :math:`\rho_0 + \rho_{\rm floor}`;
    for :math:`r \gg r_0 + H` it asymptotes to :math:`\rho_{\rm floor}`.

    .. rubric:: Physical relevance

    An exponential density profile is characteristic of the extended
    atmosphere of a cool giant star, the outer shell of an AGB wind, or a
    thick accretion disk atmosphere.  It is also used as a phenomenological
    model for the narrow-velocity CSM inferred from absorption features in
    Type IIn spectra (e.g. H:math:`\alpha` P Cygni profiles), where the
    density scale height :math:`H` encodes the wind acceleration zone.
    When :math:`r_0 = 0`, the profile is a pure falling exponential from the
    origin.

    .. rubric:: Parameters

    .. list-table::
       :header-rows: 1
       :widths: 22 6 20 52

       * - Parameter
         - Opt.
         - Type
         - Description
       * - ``rho_0``
         - ✓
         - float or :class:`~astropy.units.Quantity`
         - Density amplitude above the floor at :math:`r = r_0` in
           :math:`\mathrm{g\,cm^{-3}}`. Must be non-negative.
       * - ``r_0``
         - ✓
         - float or :class:`~astropy.units.Quantity`
         - Reference (anchor) radius. Bare values interpreted as cm. Must
           be non-negative.
       * - ``scale_height``
         - ✓
         - float or :class:`~astropy.units.Quantity`
         - Exponential scale height :math:`H`. Bare values interpreted as cm.
           Must be positive.
       * - ``density_floor``
         - ✓
         - float or :class:`~astropy.units.Quantity`
         - Asymptotic background density in :math:`\mathrm{g\,cm^{-3}}`.
           Defaults to 0.

    .. rubric:: Example

    .. plot::

       import numpy as np
       import astropy.units as u
       import matplotlib.pyplot as plt
       from trilobite.dynamics.profiles.csm import ExponentialCSMProfile
       from trilobite.utils.plot_utils import set_plot_style

       set_plot_style()
       r = np.linspace(0, 3e16, 500) * u.cm
       fig, ax = plt.subplots()
       for H, ls in [(3e15, '--'), (6e15, '-'), (1.2e16, ':')]:
           rho = ExponentialCSMProfile.eval(
               r,
               rho_0=1e-18 * u.g / u.cm**3,
               r_0=0.0 * u.cm,
               scale_height=H * u.cm,
               density_floor=1e-23 * u.g / u.cm**3,
           )
           ax.semilogy(r.value, rho.value, ls=ls,
                       label=fr'$H = {H:.0e}\ \mathrm{{cm}}$')
       ax.set_xlabel(r'$r\ [\mathrm{cm}]$')
       ax.set_ylabel(r'$\rho\ [\mathrm{g\,cm^{-3}}]$')
       ax.set_title('Exponential CSM Profile')
       ax.legend()
       plt.tight_layout()
    """

    @classmethod
    def _validate_and_process_parameters(cls, *, rho_0, r_0, scale_height, density_floor=0.0, **_):
        rho0 = float(ensure_in_units(rho_0, u.g / u.cm**3))
        r0 = float(ensure_in_units(r_0, u.cm))
        H = float(ensure_in_units(scale_height, u.cm))
        rho_f = float(ensure_in_units(density_floor, u.g / u.cm**3))

        if rho0 < 0:
            raise ValueError("`rho_0` must be non-negative.")
        if r0 < 0:
            raise ValueError("`r_0` must be non-negative.")
        if H <= 0:
            raise ValueError("`scale_height` must be positive.")
        if rho_f < 0:
            raise ValueError("`density_floor` must be non-negative.")

        return {"rho_0": rho0, "r_0": r0, "scale_height": H, "density_floor": rho_f}

    @classmethod
    def _opt_eval(cls, r, _t=None, *, rho_0, r_0, scale_height, density_floor, **_):
        return density_floor + rho_0 * np.exp(-(r - r_0) / scale_height)


class SmoothBPLCSMProfile(StationaryCSMDensityProfile):
    r"""
    Smooth broken-power-law CSM density profile.

    The transition between two power-law slopes is made smooth by a softplus
    envelope,

    .. math::

        \rho_{\mathrm{CSM}}(r)
        =
        \rho_b
        \left(\frac{r}{r_b}\right)^{-s_{\rm in}}
        \left[
            \frac{1 + (r/r_b)^{1/\Delta}}{2}
        \right]^{-(s_{\rm out} - s_{\rm in})\Delta},

    where :math:`\Delta > 0` is a dimensionless smoothness parameter.  The
    density at the break radius is exactly :math:`\rho(r_b) = \rho_b`
    regardless of :math:`\Delta`.

    **Asymptotic behaviour.** For :math:`r \ll r_b`:

    .. math::

        \rho \approx \rho_b\,2^{(s_{\rm out}-s_{\rm in})\Delta}
                     (r/r_b)^{-s_{\rm in}},

    and for :math:`r \gg r_b`:

    .. math::

        \rho \approx \rho_b\,2^{(s_{\rm out}-s_{\rm in})\Delta}
                     (r/r_b)^{-s_{\rm out}}.

    Both asymptotes recover the correct power-law slopes; the prefactor
    :math:`2^{(s_{\rm out}-s_{\rm in})\Delta}` approaches unity as
    :math:`\Delta \to 0` (sharp break).

    **Transition sharpness.** Smaller :math:`\Delta` gives a sharper
    transition (approaching :class:`BrokenPowerLawCSMProfile` as
    :math:`\Delta \to 0`); larger :math:`\Delta` gives a more gradual
    transition spanning a larger radial range.

    .. rubric:: Physical relevance

    The smooth broken power law is the appropriate choice when the CSM density
    transition is known to be gradual, either from physical reasoning (e.g.\ a
    wind velocity change that occurs over a dynamical timescale) or from
    observations (e.g.\ a gradual spectral steepening in the radio light
    curve).  Avoiding the sharp break of :class:`BrokenPowerLawCSMProfile`
    prevents numerical artefacts in ODE integrators that track the shock
    position.

    .. rubric:: Parameters

    .. list-table::
       :header-rows: 1
       :widths: 22 6 20 52

       * - Parameter
         - Opt.
         - Type
         - Description
       * - ``density_break``
         - ✓
         - float or :class:`~astropy.units.Quantity`
         - Density at the break radius in :math:`\mathrm{g\,cm^{-3}}`. Must
           be non-negative.
       * - ``radius_break``
         - ✓
         - float or :class:`~astropy.units.Quantity`
         - Break radius. Bare values interpreted as cm. Must be positive.
       * - ``slope_inner``
         - ✓
         - float
         - Inner power-law slope :math:`s_{\rm in}`.
       * - ``slope_outer``
         - ✓
         - float
         - Outer power-law slope :math:`s_{\rm out}`.
       * - ``smoothness``
         - ✓
         - float
         - Dimensionless transition-width parameter :math:`\Delta`. Must be
           positive.  Use :math:`\Delta \lesssim 0.1` for a near-sharp break
           and :math:`\Delta \gtrsim 1` for a very gradual transition.

    .. rubric:: Example

    .. plot::

       import numpy as np
       import astropy.units as u
       import matplotlib.pyplot as plt
       from trilobite.dynamics.profiles.csm import (
           SmoothBPLCSMProfile, BrokenPowerLawCSMProfile,
       )
       from trilobite.utils.plot_utils import set_plot_style

       set_plot_style()
       r = np.geomspace(1e14, 1e18, 400) * u.cm
       kw = dict(
           density_break=1e-18 * u.g / u.cm**3,
           radius_break=1e16 * u.cm,
           slope_inner=2.0,
           slope_outer=0.0,
       )
       rho_sharp = BrokenPowerLawCSMProfile.eval(r, **kw)
       fig, ax = plt.subplots()
       ax.loglog(r.value, rho_sharp.value, 'k--', label='Sharp BPL')
       for delta, ls in [(0.1, ':'), (0.5, '-.'), (2.0, '-')]:
           rho = SmoothBPLCSMProfile.eval(r, smoothness=delta, **kw)
           ax.loglog(r.value, rho.value, ls=ls, label=fr'$\Delta={delta}$')
       ax.axvline(1e16, ls=':', color='gray', alpha=0.5, label=r'$r_b$')
       ax.set_xlabel(r'$r\ [\mathrm{cm}]$')
       ax.set_ylabel(r'$\rho\ [\mathrm{g\,cm^{-3}}]$')
       ax.set_title('Smooth Broken Power-Law CSM Profile')
       ax.legend()
       plt.tight_layout()

    See Also
    --------
    BrokenPowerLawCSMProfile : Sharp broken-power-law variant.
    """

    @classmethod
    def _validate_and_process_parameters(
        cls, *, density_break, radius_break, slope_inner, slope_outer, smoothness, **_
    ):
        rho_b = float(ensure_in_units(density_break, u.g / u.cm**3))
        r_b = float(ensure_in_units(radius_break, u.cm))
        Delta = float(smoothness)

        if rho_b < 0:
            raise ValueError("`density_break` must be non-negative.")
        if r_b <= 0:
            raise ValueError("`radius_break` must be positive.")
        if Delta <= 0:
            raise ValueError("`smoothness` must be positive.")

        return {
            "density_break": rho_b,
            "radius_break": r_b,
            "slope_inner": float(slope_inner),
            "slope_outer": float(slope_outer),
            "smoothness": Delta,
        }

    @classmethod
    def _opt_eval(cls, r, _t=None, *, density_break, radius_break, slope_inner, slope_outer, smoothness, **_):
        x = r / radius_break
        # Evaluate log((1 + x^{1/Delta})/2) via the numerically stable form
        # log(1 + exp(z)) = logaddexp(0, z), where z = log(x)/Delta.
        z = np.log(np.maximum(x, 1e-300)) / smoothness
        log_half_factor = np.logaddexp(0.0, z) - np.log(2.0)
        exponent = -(slope_outer - slope_inner) * smoothness
        return density_break * x ** (-slope_inner) * np.exp(exponent * log_half_factor)


class CoredPowerLawCSMProfile(StationaryCSMDensityProfile):
    r"""
    Cored power-law CSM density profile.

    The profile is constant in an inner core of radius :math:`r_c` and
    transitions to a power law at larger radii,

    .. math::

        \rho_{\mathrm{CSM}}(r)
        =
        \rho_0
        \left[1 + \left(\frac{r}{r_c}\right)^2\right]^{-s/2}.

    At :math:`r \ll r_c` the density approaches the central value
    :math:`\rho_0`; at :math:`r \gg r_c` it approaches
    :math:`\rho_0 (r/r_c)^{-s}`.  The density at the core radius is
    :math:`\rho_0 / 2^{s/2}`.

    .. rubric:: Physical relevance

    A cored power law is appropriate whenever the density profile has a
    flat, roughly uniform inner region surrounded by a steeper fall-off.
    This morphology arises in the interiors of wind-blown bubbles, in
    the cool shells swept up by fast winds around compact stars, and in
    the stratified atmospheres of AGB stars.  It is also used as a prior
    in Bayesian CSM inference when the shock is observed at radii spanning
    both the flat core and the power-law envelope.

    .. rubric:: Parameters

    .. list-table::
       :header-rows: 1
       :widths: 22 6 20 52

       * - Parameter
         - Opt.
         - Type
         - Description
       * - ``rho_0``
         - ✓
         - float or :class:`~astropy.units.Quantity`
         - Central density in :math:`\mathrm{g\,cm^{-3}}`. Must be
           non-negative.
       * - ``r_core``
         - ✓
         - float or :class:`~astropy.units.Quantity`
         - Core radius. Bare values interpreted as cm. Must be positive.
       * - ``slope``
         - ✓
         - float
         - Outer power-law slope :math:`s` (the density falls as
           :math:`r^{-s}` for :math:`r \gg r_c`).  Must be non-negative.

    .. rubric:: Example

    .. plot::

       import numpy as np
       import astropy.units as u
       import matplotlib.pyplot as plt
       from trilobite.dynamics.profiles.csm import CoredPowerLawCSMProfile
       from trilobite.utils.plot_utils import set_plot_style

       set_plot_style()
       r = np.geomspace(1e13, 1e18, 400) * u.cm
       r_core = 1e15 * u.cm
       rho_0 = 1e-18 * u.g / u.cm**3
       fig, ax = plt.subplots()
       for s, ls in [(1.0, '--'), (2.0, '-'), (3.0, ':')]:
           rho = CoredPowerLawCSMProfile.eval(r, rho_0=rho_0, r_core=r_core, slope=s)
           ax.loglog(r.value, rho.value, ls=ls, label=fr'$s={s}$')
       ax.axvline(r_core.value, ls=':', color='gray', alpha=0.5, label=r'$r_c$')
       ax.set_xlabel(r'$r\ [\mathrm{cm}]$')
       ax.set_ylabel(r'$\rho\ [\mathrm{g\,cm^{-3}}]$')
       ax.set_title('Cored Power-Law CSM Profile')
       ax.legend()
       plt.tight_layout()
    """

    @classmethod
    def _validate_and_process_parameters(cls, *, rho_0, r_core, slope, **_):
        rho0 = float(ensure_in_units(rho_0, u.g / u.cm**3))
        rc = float(ensure_in_units(r_core, u.cm))
        s = float(slope)

        if rho0 < 0:
            raise ValueError("`rho_0` must be non-negative.")
        if rc <= 0:
            raise ValueError("`r_core` must be positive.")
        if s < 0:
            raise ValueError("`slope` must be non-negative.")

        return {"rho_0": rho0, "r_core": rc, "slope": s}

    @classmethod
    def _opt_eval(cls, r, _t=None, *, rho_0, r_core, slope, **_):
        return rho_0 * (1.0 + (r / r_core) ** 2) ** (-0.5 * slope)


class FiniteWindCSMProfile(StationaryCSMDensityProfile):
    r"""
    Wind CSM profile confined between inner and outer radii.

    The steady-wind density :math:`A r^{-2}` is returned only within a
    finite radial window; outside the window the density equals a constant
    floor,

    .. math::

        \rho_{\mathrm{CSM}}(r)
        =
        \begin{cases}
            A\,r^{-2},        & r_{\min} \le r \le r_{\max}, \\
            \rho_{\rm floor}, & \text{otherwise}.
        \end{cases}

    .. rubric:: Physical relevance

    The finite-wind profile models a mass-loss episode that was active
    only during a bounded interval of pre-explosion time.  If the wind
    velocity was :math:`v_w` and the wind was active from :math:`t_{\rm start}`
    to :math:`t_{\rm end}` before the explosion, then
    :math:`r_{\min} = v_w\,t_{\rm start}` and
    :math:`r_{\max} = v_w\,t_{\rm end}`.  The convenience method
    :meth:`normalize_from_timing` converts wind velocity and active-time
    limits to radial boundaries.  This parameterisation is appropriate for
    binary-interaction scenarios (e.g.\ common-envelope ejection followed by
    resumption of normal winds) and for pulsational pair-instability events
    that produce a sequence of separated wind episodes.

    .. rubric:: Parameters

    .. list-table::
       :header-rows: 1
       :widths: 22 6 20 52

       * - Parameter
         - Opt.
         - Type
         - Description
       * - ``A``
         - ✓
         - float or :class:`~astropy.units.Quantity`
         - Wind-density normalization in :math:`\mathrm{g\,cm^{-1}}`. Bare
           values interpreted as :math:`\mathrm{g\,cm^{-1}}`. Must be
           positive. Use :meth:`normalize` to compute from mass-loss rate and
           wind speed.
       * - ``r_min``
         - ✓
         - float or :class:`~astropy.units.Quantity`
         - Inner wind radius. Bare values interpreted as cm. Must be
           non-negative.
       * - ``r_max``
         - ✓
         - float or :class:`~astropy.units.Quantity`
         - Outer wind radius. Bare values interpreted as cm. Must exceed
           ``r_min``.
       * - ``density_floor``
         - ✓
         - float or :class:`~astropy.units.Quantity`
         - Density returned outside the wind window in
           :math:`\mathrm{g\,cm^{-3}}`. Defaults to 0.

    .. rubric:: Example

    .. plot::

       import numpy as np
       import astropy.units as u
       import matplotlib.pyplot as plt
       from trilobite.dynamics.profiles.csm import FiniteWindCSMProfile
       from trilobite.utils.plot_utils import set_plot_style

       set_plot_style()
       r = np.geomspace(1e13, 1e18, 600) * u.cm
       rho = FiniteWindCSMProfile.eval(
           r,
           A=5e11 * u.g / u.cm,
           r_min=1e15 * u.cm,
           r_max=3e16 * u.cm,
           density_floor=1e-23 * u.g / u.cm**3,
       )
       fig, ax = plt.subplots()
       ax.loglog(r.value, rho.value)
       ax.axvspan(1e15, 3e16, alpha=0.1, color='C0', label='Wind region')
       ax.set_xlabel(r'$r\ [\mathrm{cm}]$')
       ax.set_ylabel(r'$\rho\ [\mathrm{g\,cm^{-3}}]$')
       ax.set_title('Finite Wind CSM Profile')
       ax.legend()
       plt.tight_layout()

    See Also
    --------
    FiniteWindCSMProfile.normalize : Compute ``A`` from mass-loss rate and wind speed.
    FiniteWindCSMProfile.normalize_from_timing : Compute radial limits from timing.
    """

    @classmethod
    def normalize(
        cls,
        mass_loss_rate: "_UnitBearingScalarLike",
        wind_velocity: "_UnitBearingScalarLike",
    ) -> u.Quantity:
        r"""
        Compute the wind-density normalization :math:`A = \dot{M}/(4\pi v_w)`.

        Parameters
        ----------
        mass_loss_rate : float or ~astropy.units.Quantity
            Progenitor mass-loss rate. Bare values interpreted as g/s.
        wind_velocity : float or ~astropy.units.Quantity
            Wind velocity. Bare values interpreted as cm/s.

        Returns
        -------
        A : ~astropy.units.Quantity
            Wind-density normalization in :math:`\mathrm{g\,cm^{-1}}`.
        """
        return WindCSMProfile.normalize(mass_loss_rate, wind_velocity)

    @classmethod
    def normalize_from_timing(
        cls,
        wind_velocity: "_UnitBearingScalarLike",
        t_start: "_UnitBearingScalarLike",
        t_end: "_UnitBearingScalarLike",
    ) -> tuple:
        r"""
        Compute radial wind boundaries from wind velocity and active-time limits.

        For a wind of velocity :math:`v_w` active from :math:`t_{\rm start}` to
        :math:`t_{\rm end}` before the explosion, the corresponding radial limits
        are

        .. math::

            r_{\min} = v_w\,t_{\rm start},
            \qquad
            r_{\max} = v_w\,t_{\rm end}.

        Parameters
        ----------
        wind_velocity : float or ~astropy.units.Quantity
            Wind terminal velocity. Bare values interpreted as cm/s.
        t_start : float or ~astropy.units.Quantity
            Time before explosion at which the wind *started*. Bare values
            interpreted as seconds.
        t_end : float or ~astropy.units.Quantity
            Time before explosion at which the wind *ended*.  Must exceed
            ``t_start``. Bare values interpreted as seconds.

        Returns
        -------
        r_min : ~astropy.units.Quantity
            Inner wind radius in cm.
        r_max : ~astropy.units.Quantity
            Outer wind radius in cm.

        Raises
        ------
        ValueError
            If ``t_end`` does not exceed ``t_start``.
        """
        v = float(ensure_in_units(wind_velocity, u.cm / u.s))
        t0 = float(ensure_in_units(t_start, u.s))
        t1 = float(ensure_in_units(t_end, u.s))

        if t1 <= t0:
            raise ValueError("`t_end` must exceed `t_start`.")
        if t0 < 0 or t1 < 0:
            raise ValueError("Time limits must be non-negative.")

        return v * t0 * u.cm, v * t1 * u.cm

    @classmethod
    def _validate_and_process_parameters(cls, *, A, r_min, r_max, density_floor=0.0, **_):
        A_cgs = float(ensure_in_units(A, u.g / u.cm))
        r_min_cgs = float(ensure_in_units(r_min, u.cm))
        r_max_cgs = float(ensure_in_units(r_max, u.cm))
        rho_f = float(ensure_in_units(density_floor, u.g / u.cm**3))

        if A_cgs <= 0:
            raise ValueError("`A` must be positive.")
        if r_min_cgs < 0:
            raise ValueError("`r_min` must be non-negative.")
        if r_max_cgs <= r_min_cgs:
            raise ValueError("`r_max` must exceed `r_min`.")
        if rho_f < 0:
            raise ValueError("`density_floor` must be non-negative.")

        return {"A": A_cgs, "r_min": r_min_cgs, "r_max": r_max_cgs, "density_floor": rho_f}

    @classmethod
    def _opt_eval(cls, r, _t=None, *, A, r_min, r_max, density_floor, **_):
        inside = (r >= r_min) & (r <= r_max)
        return np.where(inside, A * r**-2, density_floor)


class WindBubbleCSMProfile(StationaryCSMDensityProfile):
    r"""
    Stellar wind-bubble CSM density profile.

    The profile describes the four-zone structure produced by the interaction
    of a fast stellar wind with the surrounding ISM:

    .. math::

        \rho_{\mathrm{CSM}}(r)
        =
        \begin{cases}
            A\,r^{-2},            & r \le R_{\rm ts},                              \\
            \rho_{\rm bub},       & R_{\rm ts} < r \le R_{\rm sh,\,in},            \\
            \rho_{\rm sh},        & R_{\rm sh,\,in} < r \le R_{\rm sh,\,out},      \\
            \rho_{\rm ISM},       & r > R_{\rm sh,\,out},
        \end{cases}

    where :math:`R_{\rm ts}` is the wind termination shock radius,
    :math:`\rho_{\rm bub}` is the hot shocked-wind (bubble) density,
    :math:`R_{\rm sh,\,in}` and :math:`R_{\rm sh,\,out}` are the inner and
    outer boundaries of the dense swept-up shell, and :math:`\rho_{\rm ISM}`
    is the undisturbed ambient ISM density.

    .. rubric:: Physical relevance

    The wind-bubble structure is the physically motivated CSM environment for
    massive stars that spend significant time as Wolf-Rayet stars before core
    collapse.  A fast WR wind creates a low-density hot bubble and sweeps a
    dense shell of ISM material.  When the supernova shock exits the free-wind
    zone and interacts with the bubble and shell, the radio, X-ray, and optical
    light curves show distinctive signatures.  The four-zone parameterisation
    provides a self-consistent upstream density for self-similar shock models
    applied to Type Ib/c supernovae and long-duration GRBs in massive-star
    environments.

    .. rubric:: Parameters

    The radii must satisfy
    :math:`R_{\rm ts} < R_{\rm sh,\,in} < R_{\rm sh,\,out}`.

    .. list-table::
       :header-rows: 1
       :widths: 28 6 20 46

       * - Parameter
         - Opt.
         - Type
         - Description
       * - ``A``
         - ✓
         - float or :class:`~astropy.units.Quantity`
         - Wind-density normalization in :math:`\mathrm{g\,cm^{-1}}`. Bare
           values interpreted as :math:`\mathrm{g\,cm^{-1}}`. Must be
           positive. Use :meth:`normalize` to compute from mass-loss rate and
           wind speed.
       * - ``r_wind_termination``
         - ✓
         - float or :class:`~astropy.units.Quantity`
         - Wind termination shock radius :math:`R_{\rm ts}`. Bare values
           interpreted as cm. Must be positive.
       * - ``rho_bubble``
         - ✓
         - float or :class:`~astropy.units.Quantity`
         - Hot bubble density :math:`\rho_{\rm bub}` in
           :math:`\mathrm{g\,cm^{-3}}`. Must be non-negative.
       * - ``r_shell_inner``
         - ✓
         - float or :class:`~astropy.units.Quantity`
         - Inner shell radius :math:`R_{\rm sh,\,in}`. Bare values
           interpreted as cm. Must exceed ``r_wind_termination``.
       * - ``r_shell_outer``
         - ✓
         - float or :class:`~astropy.units.Quantity`
         - Outer shell radius :math:`R_{\rm sh,\,out}`. Bare values
           interpreted as cm. Must exceed ``r_shell_inner``.
       * - ``rho_shell``
         - ✓
         - float or :class:`~astropy.units.Quantity`
         - Dense shell density :math:`\rho_{\rm sh}` in
           :math:`\mathrm{g\,cm^{-3}}`. Must be non-negative.
       * - ``rho_ism``
         - ✓
         - float or :class:`~astropy.units.Quantity`
         - Undisturbed ISM density :math:`\rho_{\rm ISM}` in
           :math:`\mathrm{g\,cm^{-3}}`. Defaults to 0.

    .. rubric:: Example

    .. plot::

       import numpy as np
       import astropy.units as u
       import matplotlib.pyplot as plt
       from trilobite.dynamics.profiles.csm import WindBubbleCSMProfile
       from trilobite.utils.plot_utils import set_plot_style

       set_plot_style()
       r = np.geomspace(1e14, 1e19, 800) * u.cm
       rho = WindBubbleCSMProfile.eval(
           r,
           A=5e11 * u.g / u.cm,
           r_wind_termination=3e16 * u.cm,
           rho_bubble=1e-25 * u.g / u.cm**3,
           r_shell_inner=3e17 * u.cm,
           r_shell_outer=4e17 * u.cm,
           rho_shell=1e-21 * u.g / u.cm**3,
           rho_ism=1e-24 * u.g / u.cm**3,
       )
       fig, ax = plt.subplots()
       ax.loglog(r.value, rho.value)
       ax.set_xlabel(r'$r\ [\mathrm{cm}]$')
       ax.set_ylabel(r'$\rho\ [\mathrm{g\,cm^{-3}}]$')
       ax.set_title('Wind Bubble CSM Profile')
       for x, lbl in [(3e16, r'$R_\mathrm{ts}$'), (3e17, r'$R_\mathrm{sh,in}$'),
                      (4e17, r'$R_\mathrm{sh,out}$')]:
           ax.axvline(x, ls='--', color='gray', alpha=0.6)
           ax.text(x * 1.05, 1e-19, lbl, rotation=90, fontsize=8, va='top')
       plt.tight_layout()

    See Also
    --------
    WindBubbleCSMProfile.normalize : Compute ``A`` from mass-loss rate and wind speed.
    """

    @classmethod
    def normalize(
        cls,
        mass_loss_rate: "_UnitBearingScalarLike",
        wind_velocity: "_UnitBearingScalarLike",
    ) -> u.Quantity:
        r"""
        Compute the wind-density normalization :math:`A = \dot{M}/(4\pi v_w)`.

        Parameters
        ----------
        mass_loss_rate : float or ~astropy.units.Quantity
            Progenitor mass-loss rate. Bare values interpreted as g/s.
        wind_velocity : float or ~astropy.units.Quantity
            Wind velocity. Bare values interpreted as cm/s.

        Returns
        -------
        A : ~astropy.units.Quantity
            Wind-density normalization in :math:`\mathrm{g\,cm^{-1}}`.
        """
        return WindCSMProfile.normalize(mass_loss_rate, wind_velocity)

    @classmethod
    def _validate_and_process_parameters(
        cls,
        *,
        A,
        r_wind_termination,
        rho_bubble,
        r_shell_inner,
        r_shell_outer,
        rho_shell,
        rho_ism=0.0,
        **_,
    ):
        A_cgs = float(ensure_in_units(A, u.g / u.cm))
        r_ts = float(ensure_in_units(r_wind_termination, u.cm))
        rho_bub = float(ensure_in_units(rho_bubble, u.g / u.cm**3))
        r_sh_in = float(ensure_in_units(r_shell_inner, u.cm))
        r_sh_out = float(ensure_in_units(r_shell_outer, u.cm))
        rho_sh = float(ensure_in_units(rho_shell, u.g / u.cm**3))
        rho_ism_cgs = float(ensure_in_units(rho_ism, u.g / u.cm**3))

        if A_cgs <= 0:
            raise ValueError("`A` must be positive.")
        if r_ts <= 0:
            raise ValueError("`r_wind_termination` must be positive.")
        if r_sh_in <= r_ts:
            raise ValueError("`r_shell_inner` must exceed `r_wind_termination`.")
        if r_sh_out <= r_sh_in:
            raise ValueError("`r_shell_outer` must exceed `r_shell_inner`.")
        if rho_bub < 0:
            raise ValueError("`rho_bubble` must be non-negative.")
        if rho_sh < 0:
            raise ValueError("`rho_shell` must be non-negative.")
        if rho_ism_cgs < 0:
            raise ValueError("`rho_ism` must be non-negative.")

        return {
            "A": A_cgs,
            "r_wind_termination": r_ts,
            "rho_bubble": rho_bub,
            "r_shell_inner": r_sh_in,
            "r_shell_outer": r_sh_out,
            "rho_shell": rho_sh,
            "rho_ism": rho_ism_cgs,
        }

    @classmethod
    def _opt_eval(
        cls,
        r,
        _t=None,
        *,
        A,
        r_wind_termination,
        rho_bubble,
        r_shell_inner,
        r_shell_outer,
        rho_shell,
        rho_ism,
        **_,
    ):
        result = np.where(r > r_shell_outer, rho_ism, rho_shell)
        result = np.where((r > r_shell_inner) & (r <= r_shell_outer), rho_shell, result)
        result = np.where((r > r_wind_termination) & (r <= r_shell_inner), rho_bubble, result)
        result = np.where(r <= r_wind_termination, A * r**-2, result)
        return result


class LogNormalCSMProfile(StationaryCSMDensityProfile):
    r"""
    Log-normal CSM density profile.

    The density peaks at a characteristic radius and has Gaussian tails in
    log-radius,

    .. math::

        \rho_{\mathrm{CSM}}(r)
        =
        \rho_0
        \exp\!\left[
            -\frac{(\ln r - \ln r_0)^2}{2\sigma^2}
        \right]
        =
        \rho_0
        \exp\!\left[
            -\frac{1}{2}
            \left(\frac{\ln(r/r_0)}{\sigma}\right)^2
        \right],

    where :math:`r_0` is the radius of peak density and :math:`\sigma` is the
    width in log-radius (dimensionless).

    .. rubric:: Physical relevance

    A log-normal density profile naturally arises from an eruptive mass-loss
    event that deposits material over a narrow range of velocities, producing
    a peaked radial density distribution.  It is an appropriate model for the
    CSM produced by LBV giant eruptions (e.g.\ Eta Carinae analogues), where
    the expelled shell has a well-defined expansion velocity and a finite
    radial extent set by the duration of the eruption.  Compared with the
    top-hat :class:`ShellCSMProfile`, the log-normal form avoids sharp
    density edges and provides a smooth, differentiable profile suitable for
    ODE-based shock integrators.  The parameter :math:`\sigma` encodes the
    fractional velocity spread of the ejected material.

    .. rubric:: Parameters

    .. list-table::
       :header-rows: 1
       :widths: 22 6 20 52

       * - Parameter
         - Opt.
         - Type
         - Description
       * - ``rho_0``
         - ✓
         - float or :class:`~astropy.units.Quantity`
         - Peak density at :math:`r = r_0` in :math:`\mathrm{g\,cm^{-3}}`.
           Must be non-negative.
       * - ``r_peak``
         - ✓
         - float or :class:`~astropy.units.Quantity`
         - Radius of peak density :math:`r_0`. Bare values interpreted as cm.
           Must be positive.
       * - ``sigma``
         - ✓
         - float
         - Dimensionless log-radius width :math:`\sigma`. Must be positive.
           Larger values produce a broader profile.

    .. rubric:: Example

    .. plot::

       import numpy as np
       import astropy.units as u
       import matplotlib.pyplot as plt
       from trilobite.dynamics.profiles.csm import LogNormalCSMProfile
       from trilobite.utils.plot_utils import set_plot_style

       set_plot_style()
       r = np.geomspace(1e14, 1e18, 500) * u.cm
       r_peak = 1e16 * u.cm
       rho_0 = 1e-18 * u.g / u.cm**3
       fig, ax = plt.subplots()
       for sigma, ls in [(0.3, '--'), (0.6, '-'), (1.2, ':')]:
           rho = LogNormalCSMProfile.eval(r, rho_0=rho_0, r_peak=r_peak, sigma=sigma)
           ax.loglog(r.value, rho.value, ls=ls, label=fr'$\sigma={sigma}$')
       ax.axvline(r_peak.value, ls=':', color='gray', alpha=0.5, label=r'$r_0$')
       ax.set_xlabel(r'$r\ [\mathrm{cm}]$')
       ax.set_ylabel(r'$\rho\ [\mathrm{g\,cm^{-3}}]$')
       ax.set_title('Log-Normal CSM Profile')
       ax.legend()
       plt.tight_layout()
    """

    @classmethod
    def _validate_and_process_parameters(cls, *, rho_0, r_peak, sigma, **_):
        rho0 = float(ensure_in_units(rho_0, u.g / u.cm**3))
        r0 = float(ensure_in_units(r_peak, u.cm))
        sig = float(sigma)

        if rho0 < 0:
            raise ValueError("`rho_0` must be non-negative.")
        if r0 <= 0:
            raise ValueError("`r_peak` must be positive.")
        if sig <= 0:
            raise ValueError("`sigma` must be positive.")

        return {"rho_0": rho0, "r_peak": r0, "sigma": sig}

    @classmethod
    def _opt_eval(cls, r, _t=None, *, rho_0, r_peak, sigma, **_):
        x = np.log(np.maximum(r, 1e-300) / r_peak) / sigma
        return rho_0 * np.exp(-0.5 * x**2)


class TwoWindCSMProfile(StationaryCSMDensityProfile):
    r"""
    Smooth two-wind CSM density profile.

    This profile represents a stationary wind-like CSM whose density remains
    proportional to :math:`r^{-2}`, but whose wind normalization transitions
    smoothly from an inner wind value :math:`A_1` to an outer wind value
    :math:`A_2`:

    .. math::

        \rho_{\mathrm{CSM}}(r)
        =
        \frac{A(r)}{r^2}.

    The transition is controlled by a hyperbolic-tangent switch in logarithmic
    radius,

    .. math::

        S(r)
        =
        \frac{1}{2}
        \left[
            1
            +
            \tanh\left(
                \frac{\ln(r / r_{\rm transition})}{\Delta}
            \right)
        \right],

    where :math:`r_{\rm transition}` is the characteristic transition radius
    and :math:`\Delta` is the dimensionless transition width in
    :math:`\ln r`.

    The wind normalization is interpolated logarithmically,

    .. math::

        A(r)
        =
        A_1^{1-S(r)} A_2^{S(r)}.

    This multiplicative interpolation preserves positivity and treats upward
    and downward normalization changes symmetrically.

    .. rubric:: Physical relevance

    This profile is useful for modelling a progenitor wind whose mass-loss
    normalization changed before explosion while retaining an approximately
    steady-wind radial structure. Since material at larger radii was ejected
    earlier, the transition from :math:`A_1` to :math:`A_2` can represent a
    change in :math:`\dot{M}/v_w` at the lookback time corresponding to
    :math:`r_{\rm transition}`.

    .. rubric:: Wind parameterization

    The inner and outer winds may each be supplied using one of three equivalent
    parameterization routes:

    1. Direct wind normalizations ``A_1`` and ``A_2``.
    2. Dimensionless wind parameters ``A_star_1`` and ``A_star_2``.
    3. Physical wind parameters
       ``mass_loss_rate_1``, ``wind_velocity_1``,
       ``mass_loss_rate_2``, and ``wind_velocity_2``.

    The same route must be used for both winds in a single call. Mixing routes
    raises a :class:`ValueError`.

    .. rubric:: Parameters

    .. list-table::
       :header-rows: 1
       :widths: 26 6 22 46

       * - Parameter
         - Opt.
         - Type
         - Description
       * - ``A_1``
         - ☐
         - float or :class:`~astropy.units.Quantity`
         - Inner wind-density normalization in
           :math:`\mathrm{g\,cm^{-1}}`. Bare values interpreted as
           :math:`\mathrm{g\,cm^{-1}}`. Must be positive.
       * - ``A_2``
         - ☐
         - float or :class:`~astropy.units.Quantity`
         - Outer wind-density normalization in
           :math:`\mathrm{g\,cm^{-1}}`. Bare values interpreted as
           :math:`\mathrm{g\,cm^{-1}}`. Must be positive.
       * - ``A_star_1``
         - ☐
         - float
         - Inner dimensionless wind-density parameter,
           :math:`A_1 = 5\times10^{11} A_{*,1}\,\mathrm{g\,cm^{-1}}`.
       * - ``A_star_2``
         - ☐
         - float
         - Outer dimensionless wind-density parameter,
           :math:`A_2 = 5\times10^{11} A_{*,2}\,\mathrm{g\,cm^{-1}}`.
       * - ``mass_loss_rate_1``
         - ☐
         - float or :class:`~astropy.units.Quantity`
         - Inner wind mass-loss rate. Bare values interpreted as
           :math:`\mathrm{g\,s^{-1}}`.
       * - ``wind_velocity_1``
         - ☐
         - float or :class:`~astropy.units.Quantity`
         - Inner wind velocity. Bare values interpreted as
           :math:`\mathrm{cm\,s^{-1}}`.
       * - ``mass_loss_rate_2``
         - ☐
         - float or :class:`~astropy.units.Quantity`
         - Outer wind mass-loss rate. Bare values interpreted as
           :math:`\mathrm{g\,s^{-1}}`.
       * - ``wind_velocity_2``
         - ☐
         - float or :class:`~astropy.units.Quantity`
         - Outer wind velocity. Bare values interpreted as
           :math:`\mathrm{cm\,s^{-1}}`.
       * - ``r_transition``
         - ✓
         - float or :class:`~astropy.units.Quantity`
         - Characteristic transition radius. Bare values interpreted as cm.
           Must be positive.
       * - ``transition_width``
         - ✓
         - float
         - Dimensionless transition width in :math:`\ln r`. Must be
           positive.

    .. rubric:: Example

    .. plot::

       import numpy as np
       import astropy.units as u
       import matplotlib.pyplot as plt
       from trilobite.dynamics.profiles.csm import TwoWindCSMProfile
       from trilobite.utils.plot_utils import set_plot_style

       set_plot_style()

       r = np.geomspace(1e14, 1e18, 500) * u.cm

       rho = TwoWindCSMProfile.eval(
           r,
           A_star_1=1.0,
           A_star_2=100.0,
           r_transition=1e16 * u.cm,
           transition_width=0.25,
       )

       fig, ax = plt.subplots()
       ax.loglog(r.value, rho.value)
       ax.axvline(1e16, ls='--', color='gray', label=r'$r_{\rm transition}$')
       ax.set_xlabel(r'$r\ [\mathrm{cm}]$')
       ax.set_ylabel(r'$\rho\ [\mathrm{g\,cm^{-3}}]$')
       ax.set_title('Smooth Two-Wind CSM Profile')
       ax.legend()
       plt.tight_layout()

    Notes
    -----
    The density is asymptotically wind-like on both sides of the transition:

    .. math::

        \rho(r \ll r_{\rm transition}) \approx A_1 r^{-2},

    and

    .. math::

        \rho(r \gg r_{\rm transition}) \approx A_2 r^{-2}.

    Inside the transition region, the effective local slope differs from
    :math:`-2` because :math:`A(r)` varies with radius.

    See Also
    --------
    WindCSMProfile : Single steady-wind CSM profile.
    SmoothTruncatedWindCSMProfile : Smooth transition from a wind to a floor.
    SmoothBPLCSMProfile : Smooth transition between two power-law slopes.
    """

    @classmethod
    def normalize(
        cls,
        mass_loss_rate: "_UnitBearingScalarLike",
        wind_velocity: "_UnitBearingScalarLike",
    ) -> u.Quantity:
        r"""
        Compute the wind-density normalization :math:`A = \dot{M}/(4\pi v_w)`.

        Parameters
        ----------
        mass_loss_rate : float or ~astropy.units.Quantity
            Progenitor mass-loss rate. Bare values interpreted as g/s.
        wind_velocity : float or ~astropy.units.Quantity
            Wind velocity. Bare values interpreted as cm/s.

        Returns
        -------
        A : ~astropy.units.Quantity
            Wind-density normalization in :math:`\mathrm{g\,cm^{-1}}`.
        """
        return WindCSMProfile.normalize(mass_loss_rate, wind_velocity)

    @classmethod
    def normalize_A_star(
        cls,
        mass_loss_rate: "_UnitBearingScalarLike",
        wind_velocity: "_UnitBearingScalarLike",
    ) -> float:
        r"""Compute the dimensionless wind-density parameter :math:`A_*`."""
        return WindCSMProfile.normalize_A_star(mass_loss_rate, wind_velocity)

    @classmethod
    def normalize_from_A_star(
        cls,
        A_star: float,
    ) -> u.Quantity:
        r"""Compute :math:`A` from the dimensionless wind-density parameter."""
        return WindCSMProfile.normalize_from_A_star(A_star)

    @classmethod
    def _resolve_wind_normalizations(
        cls,
        *,
        A_1=None,
        A_2=None,
        A_star_1=None,
        A_star_2=None,
        mass_loss_rate_1=None,
        wind_velocity_1=None,
        mass_loss_rate_2=None,
        wind_velocity_2=None,
    ) -> tuple[float, float]:
        r"""Resolve the two wind parameterizations into CGS ``A_1`` and ``A_2``."""
        has_A = A_1 is not None or A_2 is not None
        has_A_star = A_star_1 is not None or A_star_2 is not None
        has_physical = (
            mass_loss_rate_1 is not None
            or wind_velocity_1 is not None
            or mass_loss_rate_2 is not None
            or wind_velocity_2 is not None
        )

        n_routes = sum([has_A, has_A_star, has_physical])

        if n_routes == 0:
            raise ValueError(
                "One wind parameterization route must be supplied: "
                "(`A_1`, `A_2`), (`A_star_1`, `A_star_2`), or "
                "(`mass_loss_rate_1`, `wind_velocity_1`, "
                "`mass_loss_rate_2`, `wind_velocity_2`)."
            )

        if n_routes > 1:
            raise ValueError(
                "Provide only one wind parameterization route: "
                "(`A_1`, `A_2`), (`A_star_1`, `A_star_2`), or "
                "the physical mass-loss/velocity route."
            )

        if has_A:
            if A_1 is None or A_2 is None:
                raise ValueError("`A_1` and `A_2` must be supplied together.")

        elif has_A_star:
            if A_star_1 is None or A_star_2 is None:
                raise ValueError("`A_star_1` and `A_star_2` must be supplied together.")

            A_1 = cls.normalize_from_A_star(A_star_1)
            A_2 = cls.normalize_from_A_star(A_star_2)

        else:
            missing = []
            if mass_loss_rate_1 is None:
                missing.append("mass_loss_rate_1")
            if wind_velocity_1 is None:
                missing.append("wind_velocity_1")
            if mass_loss_rate_2 is None:
                missing.append("mass_loss_rate_2")
            if wind_velocity_2 is None:
                missing.append("wind_velocity_2")

            if missing:
                missing_str = "`, `".join(missing)
                raise ValueError(
                    "The physical wind route requires all four parameters: "
                    "`mass_loss_rate_1`, `wind_velocity_1`, "
                    "`mass_loss_rate_2`, and `wind_velocity_2`. "
                    f"Missing: `{missing_str}`."
                )

            A_1 = cls.normalize(
                mass_loss_rate=mass_loss_rate_1,
                wind_velocity=wind_velocity_1,
            )
            A_2 = cls.normalize(
                mass_loss_rate=mass_loss_rate_2,
                wind_velocity=wind_velocity_2,
            )

        A_1_cgs = float(ensure_in_units(A_1, u.g / u.cm))
        A_2_cgs = float(ensure_in_units(A_2, u.g / u.cm))

        if A_1_cgs <= 0:
            raise ValueError("`A_1` must be positive.")
        if A_2_cgs <= 0:
            raise ValueError("`A_2` must be positive.")

        return A_1_cgs, A_2_cgs

    @classmethod
    def _validate_and_process_parameters(
        cls,
        *,
        A_1=None,
        A_2=None,
        A_star_1=None,
        A_star_2=None,
        mass_loss_rate_1=None,
        wind_velocity_1=None,
        mass_loss_rate_2=None,
        wind_velocity_2=None,
        r_transition,
        transition_width,
        **_,
    ):
        A_1_cgs, A_2_cgs = cls._resolve_wind_normalizations(
            A_1=A_1,
            A_2=A_2,
            A_star_1=A_star_1,
            A_star_2=A_star_2,
            mass_loss_rate_1=mass_loss_rate_1,
            wind_velocity_1=wind_velocity_1,
            mass_loss_rate_2=mass_loss_rate_2,
            wind_velocity_2=wind_velocity_2,
        )

        r_transition_cgs = float(ensure_in_units(r_transition, u.cm))
        width = float(transition_width)

        if r_transition_cgs <= 0:
            raise ValueError("`r_transition` must be positive.")
        if width <= 0:
            raise ValueError("`transition_width` must be positive.")

        return {
            "A_1": A_1_cgs,
            "A_2": A_2_cgs,
            "r_transition": r_transition_cgs,
            "transition_width": width,
        }

    @classmethod
    def _opt_eval(
        cls,
        r,
        _t=None,
        *,
        A_1,
        A_2,
        r_transition,
        transition_width,
        **_,
    ):
        # The transition is performed in log-radius so that the width is
        # dimensionless and behaves naturally for wind profiles spanning many
        # orders of magnitude in radius.
        x = np.log(np.maximum(r, 1e-300) / r_transition) / transition_width
        switch = 0.5 * (1.0 + np.tanh(x))

        # Interpolate multiplicatively between the two wind normalizations.
        # This preserves positivity and treats upward / downward changes in
        # the mass-loading symmetrically.
        log_A = (1.0 - switch) * np.log(A_1) + switch * np.log(A_2)
        A = np.exp(log_A)

        return A * r**-2


class AxisymmetricCSMDensityProfile(StationaryCSMDensityProfile):
    r"""
    Axisymmetric CSM density profile with a dense, disk-confined interior component.

    This profile describes a stationary, axisymmetric circumstellar medium consisting of a
    spherical exterior component plus an interior component that is concentrated toward the
    equatorial plane and fades out beyond a transition radius,

    .. math::

        \rho(r, \theta)
        =
        \rho_{\rm ext}(r)
        +
        \rho_{\rm int}(r)\,S(r)\,G(\theta),

    with power-law radial dependence for each component,

    .. math::

        \rho_{\rm int}(r) = \rho_{\rm int}\left(\frac{r}{r_{\rm ref}}\right)^{-\alpha_{\rm int}},
        \qquad
        \rho_{\rm ext}(r) = \rho_{\rm ext}\left(\frac{r}{r_{\rm ref}}\right)^{-\alpha_{\rm ext}},

    where :math:`\rho_{\rm int}` and :math:`\rho_{\rm ext}` are the component densities at the
    reference radius :math:`r_{\rm ref}`. Parameterizing by the density at a reference radius,
    rather than by a power-law amplitude, means that the same value corresponds to the same physical
    density regardless of the slope.

    The radial transition

    .. math::

        S(r) = \frac{1}{1 + (r / r_{\rm trans})^{1/\delta_{\rm trans}}}

    switches the interior component off beyond :math:`r_{\rm trans}`, with a sharpness set by
    :math:`\delta_{\rm trans}` (smaller is sharper). The angular shape is

    .. math::

        G(\theta)
        =
        \frac{f_{\rm pole} + (1 - f_{\rm pole})\,e^{-|\cot\theta| / b}}
             {f_{\rm pole} + (1 - f_{\rm pole})\,I(b)},
        \qquad
        I(b) = \int_0^\infty \frac{e^{-x/b}}{(1 + x^2)^{3/2}}\,dx,

    where :math:`\theta` is the polar angle measured from the symmetry axis, :math:`b` is the
    ``aspect`` (larger is a thicker disk), and :math:`f_{\rm pole}` is the polar-to-equatorial
    density contrast floor of the interior component. The normalization makes the
    **solid-angle average of** :math:`G` **equal to unity**, so :math:`\rho_{\rm int}` is the
    solid-angle-averaged interior density at :math:`r_{\rm ref}` (for :math:`r \ll r_{\rm trans}`).
    The structure follows :footcite:t:`Klencki_2026`.

    .. rubric:: Sightlines

    Angle is a parameter, not an argument, so that the profile keeps the two-argument ``(r, t)``
    source-function signature of the shock engines. A one-dimensional radial profile along a single
    sightline is obtained by freezing the angle:

    .. code-block:: python

        rho_equator = AxisymmetricCSMDensityProfile.as_optimized_callable(
            theta=np.pi / 2, **params
        )
        rho_pole = AxisymmetricCSMDensityProfile.as_optimized_callable(
            theta=0.0, **params
        )

    .. rubric:: Sector weights

    The half-maximum boundary of the disk, where :math:`G` is halfway between its polar floor
    and its equatorial peak, lies at latitude :math:`\arctan(b\ln 2)` for any
    :math:`f_{\rm pole}`. The disk and the polar cones therefore subtend the solid-angle fractions
    returned by :meth:`equatorial_solid_angle_fraction` and :meth:`polar_solid_angle_fraction`,
    which sum to unity.

    .. rubric:: Parameters

    .. list-table::
       :header-rows: 1
       :widths: 26 6 22 46

       * - Parameter
         - Opt.
         - Type
         - Description
       * - ``rho_int``
         - ☐
         - float or :class:`~astropy.units.Quantity`
         - Solid-angle-averaged interior density at ``r_ref``. Bare values interpreted as
           :math:`\mathrm{g\,cm^{-3}}`. Must be non-negative.
       * - ``rho_ext``
         - ☐
         - float or :class:`~astropy.units.Quantity`
         - Exterior (spherical) density at ``r_ref``. Bare values interpreted as
           :math:`\mathrm{g\,cm^{-3}}`. Must be non-negative.
       * - ``r_ref``
         - ☐
         - float or :class:`~astropy.units.Quantity`
         - Reference radius shared by both components. Bare values interpreted as cm.
           Must be positive.
       * - ``alpha_int``
         - ☐
         - float
         - Power-law slope of the interior component.
       * - ``alpha_ext``
         - ☐
         - float
         - Power-law slope of the exterior component.
       * - ``r_trans``
         - ☐
         - float or :class:`~astropy.units.Quantity`
         - Radius at which the interior component has faded to half. Bare values interpreted
           as cm. Must be positive.
       * - ``delta_trans``
         - ☐
         - float
         - Dimensionless sharpness of the transition. Must be positive.
       * - ``f_pole``
         - ☐
         - float
         - Polar density floor of the interior component relative to its equatorial peak
           (before normalization). Must lie in :math:`[0, 1]`.
       * - ``aspect``
         - ☐
         - float
         - Angular thickness :math:`b` of the interior component. Must be positive.
       * - ``theta``
         - ☐
         - float or :class:`~astropy.units.Quantity`
         - Polar angle from the symmetry axis. Bare values interpreted as radians.
           May be an array that broadcasts against ``r``.

    .. rubric:: Example

    .. plot::

       import numpy as np
       import astropy.units as u
       import matplotlib.pyplot as plt
       from trilobite.dynamics.profiles.csm import AxisymmetricCSMDensityProfile
       from trilobite.utils.plot_utils import set_plot_style

       set_plot_style()

       params = dict(
           rho_int=2e-16, rho_ext=1e-18, r_ref=1e15, alpha_int=1.5, alpha_ext=2.0,
           r_trans=5e15, delta_trans=0.3, f_pole=0.2, aspect=2.0,
       )
       r = np.geomspace(1e14, 1e18, 500) * u.cm

       fig, ax = plt.subplots()
       for theta, label in [(np.pi / 2, "equator"), (0.0, "pole")]:
           rho = AxisymmetricCSMDensityProfile.eval(r, theta=theta, **params)
           ax.loglog(r.value, rho.value, label=label)
       ax.set_xlabel(r'$r\ [\mathrm{cm}]$')
       ax.set_ylabel(r'$\rho\ [\mathrm{g\,cm^{-3}}]$')
       ax.legend()
       plt.tight_layout()

    Notes
    -----
    - The density is symmetric under :math:`\theta \to \pi - \theta` and :math:`\theta \to -\theta`.
    - The angular normalization :math:`I(b)` is evaluated by numerical quadrature once per call to the
      validation layer; freeze parameters with ``as_optimized_callable`` to avoid repeating it.
    - With ``rho_int = 0``, or ``f_pole = 1``, the profile reduces to a spherical sum of power laws.

    References
    ----------
    .. footbibliography::

    See Also
    --------
    PowerLawCSMProfile : Spherical single power law.
    BrokenPowerLawCSMProfile : Spherical broken power law.
    """

    # ------------------------------------------------------------------ #
    # Angular Structure                                                  #
    # ------------------------------------------------------------------ #
    @classmethod
    def _validate_shape_parameters(cls, f_pole: float, aspect: float) -> tuple[float, float]:
        f_pole, aspect = float(f_pole), float(aspect)
        if not 0.0 <= f_pole <= 1.0:
            raise ValueError("`f_pole` must lie in [0, 1].")
        if not (np.isfinite(aspect) and aspect > 0):
            raise ValueError("`aspect` must be positive and finite.")
        return f_pole, aspect

    @classmethod
    def _unnormalized_angular_shape(cls, theta, f_pole: float, aspect: float) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        # |cot(theta)| = |cos| / |sin|, which is +inf on the axis; exp(-inf) = 0 there.
        with np.errstate(divide="ignore", invalid="ignore"):
            abs_cot = np.abs(np.cos(theta)) / np.abs(np.sin(theta))
        return f_pole + (1.0 - f_pole) * np.exp(-abs_cot / aspect)

    @classmethod
    def angular_norm(cls, f_pole: float, aspect: float) -> float:
        r"""
        Solid-angle average of the unnormalized angular shape.

        Evaluates :math:`f_{\rm pole} + (1 - f_{\rm pole})\,I(b)` with
        :math:`I(b) = \int_0^\infty e^{-x/b}(1+x^2)^{-3/2}\,dx`, the average over the sphere of
        :math:`e^{-|\cot\theta|/b}`.

        Parameters
        ----------
        f_pole : float
            Polar density floor, in :math:`[0, 1]`.
        aspect : float
            Angular thickness :math:`b`. Must be positive.

        Returns
        -------
        float
            The normalization such that :meth:`angular_shape` has unit solid-angle average.
        """
        f_pole, aspect = cls._validate_shape_parameters(f_pole, aspect)
        integral, _ = quad(lambda x: np.exp(-x / aspect) * (1.0 + x * x) ** -1.5, 0.0, np.inf, epsabs=0.0, epsrel=1e-12)
        return f_pole + (1.0 - f_pole) * integral

    @classmethod
    def angular_shape(cls, theta, f_pole: float, aspect: float) -> np.ndarray:
        r"""
        Evaluate the normalized angular shape :math:`G(\theta)`.

        The solid-angle average of :math:`G` is unity.

        Parameters
        ----------
        theta : float, array-like, or ~astropy.units.Quantity
            Polar angle from the symmetry axis. Bare values interpreted as radians.
        f_pole : float
            Polar density floor, in :math:`[0, 1]`.
        aspect : float
            Angular thickness :math:`b`. Must be positive.

        Returns
        -------
        ~numpy.ndarray
            :math:`G(\theta)`, dimensionless.
        """
        theta = np.asarray(ensure_in_units(theta, u.rad), dtype=float)
        f_pole, aspect = cls._validate_shape_parameters(f_pole, aspect)
        return cls._unnormalized_angular_shape(theta, f_pole, aspect) / cls.angular_norm(f_pole, aspect)

    @classmethod
    def disk_half_opening_angle(cls, aspect: float) -> u.Quantity:
        r"""
        Latitude of the half-maximum boundary of the disk, :math:`\arctan(b\ln 2)`.

        Parameters
        ----------
        aspect : float
            Angular thickness :math:`b`. Must be positive.

        Returns
        -------
        ~astropy.units.Quantity
            Half-opening angle of the disk, measured from the equatorial plane, in radians.
        """
        _, aspect = cls._validate_shape_parameters(0.0, aspect)
        return np.arctan(aspect * np.log(2.0)) * u.rad

    @classmethod
    def equatorial_solid_angle_fraction(cls, aspect: float) -> float:
        r"""
        Fraction of the sphere inside the disk, :math:`\sin[\arctan(b\ln 2)]`.

        Parameters
        ----------
        aspect : float
            Angular thickness :math:`b`. Must be positive.

        Returns
        -------
        float
            The solid-angle fraction of the band within the disk half-opening angle of the equator.
        """
        return float(np.sin(cls.disk_half_opening_angle(aspect).to_value(u.rad)))

    @classmethod
    def polar_solid_angle_fraction(cls, aspect: float) -> float:
        r"""
        Fraction of the sphere inside the two polar cones, ``1 - equatorial_solid_angle_fraction``.

        Parameters
        ----------
        aspect : float
            Angular thickness :math:`b`. Must be positive.

        Returns
        -------
        float
            The solid-angle fraction outside the disk.
        """
        return 1.0 - cls.equatorial_solid_angle_fraction(aspect)

    # ------------------------------------------------------------------ #
    # Profile Interface                                                  #
    # ------------------------------------------------------------------ #
    @classmethod
    def _validate_and_process_parameters(
        cls,
        *,
        rho_int,
        rho_ext,
        r_ref,
        alpha_int,
        alpha_ext,
        r_trans,
        delta_trans,
        f_pole,
        aspect,
        theta,
        **_,
    ):
        rho_int = float(ensure_in_units(rho_int, u.g / u.cm**3))
        rho_ext = float(ensure_in_units(rho_ext, u.g / u.cm**3))
        r_ref = float(ensure_in_units(r_ref, u.cm))
        r_trans = float(ensure_in_units(r_trans, u.cm))
        alpha_int, alpha_ext, delta_trans = float(alpha_int), float(alpha_ext), float(delta_trans)
        theta = np.asarray(ensure_in_units(theta, u.rad), dtype=float)

        if rho_int < 0:
            raise ValueError("`rho_int` must be non-negative.")
        if rho_ext < 0:
            raise ValueError("`rho_ext` must be non-negative.")
        if r_ref <= 0:
            raise ValueError("`r_ref` must be positive.")
        if r_trans <= 0:
            raise ValueError("`r_trans` must be positive.")
        if not delta_trans > 0:
            raise ValueError("`delta_trans` must be positive.")
        if not np.all(np.isfinite(theta)):
            raise ValueError("`theta` must be finite.")
        f_pole, aspect = cls._validate_shape_parameters(f_pole, aspect)

        return {
            "rho_int": rho_int,
            "rho_ext": rho_ext,
            "r_ref": r_ref,
            "alpha_int": alpha_int,
            "alpha_ext": alpha_ext,
            "r_trans": r_trans,
            "delta_trans": delta_trans,
            "f_pole": f_pole,
            "aspect": aspect,
            "theta": theta,
            "angular_norm": cls.angular_norm(f_pole, aspect),
        }

    @classmethod
    def _opt_eval(
        cls,
        r,
        _t=None,
        *,
        rho_int,
        rho_ext,
        r_ref,
        alpha_int,
        alpha_ext,
        r_trans,
        delta_trans,
        f_pole,
        aspect,
        theta,
        angular_norm,
        **_,
    ):
        x = r / r_ref
        # S(r) = 1 / (1 + (r / r_trans)^(1/delta)), written with expit to avoid overflow.
        transition = expit(-np.log(r / r_trans) / delta_trans)
        shape = cls._unnormalized_angular_shape(theta, f_pole, aspect) / angular_norm
        return rho_ext * x ** (-alpha_ext) + rho_int * x ** (-alpha_int) * transition * shape
