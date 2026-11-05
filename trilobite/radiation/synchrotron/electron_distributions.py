"""
Relativistic electron distributions for synchrotron radiation.

Provides encapsulated classes for many common electron distributions relevant to
synchrotron radiation (power-law, thermal, etc.). These classes provide methods for
computing various statistical properties of the distributions, normalizing the distributions, and
computing basic synchrotron quantities.

See Also
--------
:ref:`synch_theory_populations` for the corresponding documentation on synchrotron theory.
:ref:`synchrotron_theory` for a general introduction to synchrotron emission.
:ref:`synchrotron_electron_distributions` for documentation on working with electron distributions in Trilobite.

"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Optional, Union

import numpy as np
from astropy import units as u
from scipy.integrate import quad
from scipy.special import kve

from trilobite.radiation.constants import c_cgs, electron_rest_energy_cgs, sigma_T_cgs
from trilobite.utils.misc_utils import ensure_in_units

if TYPE_CHECKING:
    from trilobite._typing import _ArrayLike, _UnitBearingArrayLike


# =========================================================================== #
# Utility Functions                                                           #
# =========================================================================== #
# These are generic functions which do not correspond to any particular distribution function,
# but are nonetheless useful for the work of the module.
def _opt_equipartition_magnetic_field(u_therm: "_ArrayLike", epsilon_B: "_ArrayLike") -> "_ArrayLike":
    return np.sqrt(8.0 * np.pi * epsilon_B * u_therm)


def _opt_equipartition_electron_energy(u_therm: "_ArrayLike", epsilon_e: "_ArrayLike") -> "_ArrayLike":
    return epsilon_e * u_therm


def equipartition_magnetic_field(u_therm: "_UnitBearingArrayLike", epsilon_B: "_ArrayLike") -> u.Quantity:
    r"""
    Compute the magnetic field implied by a thermal energy density.

    This function assumes that a fixed fraction of the thermal/internal energy
    density is carried by magnetic fields,

    .. math::

        u_B = \epsilon_B u_{\rm therm},

    where :math:`\epsilon_B` is the magnetic energy fraction. In Gaussian CGS
    units, the magnetic energy density is

    .. math::

        u_B = \frac{B^2}{8\pi}.

    Solving for :math:`B` gives

    .. math::

        B = \left(8\pi \epsilon_B u_{\rm therm}\right)^{1/2}.

    Bare numerical values for ``u_therm`` are interpreted as CGS energy
    densities, :math:`\mathrm{erg\,cm^{-3}}`. The returned magnetic field is
    always expressed in Gauss.

    Parameters
    ----------
    u_therm : float, ~numpy.ndarray, or ~astropy.units.Quantity
        Thermal or internal energy density of the emitting/shocked region.
        If no units are attached, values are interpreted as
        :math:`\mathrm{erg\,cm^{-3}}`.
    epsilon_B : float or ~numpy.ndarray
        Fraction of the thermal energy density placed in magnetic fields.
        Physically, this should usually satisfy
        :math:`0 \leq \epsilon_B \leq 1`, though this function does not
        enforce that bound.

    Returns
    -------
    B : ~astropy.units.Quantity
        Equipartition magnetic field strength in Gauss.

    Notes
    -----
    This calculation uses the Gaussian-CGS magnetic energy density convention,
    :math:`u_B = B^2/(8\pi)`. It is therefore appropriate for code paths where
    magnetic fields are represented in Gauss and energy densities in
    :math:`\mathrm{erg\,cm^{-3}}`.

    Examples
    --------
    For a thermal energy density of :math:`10^5\,\mathrm{erg\,cm^{-3}}` and a magnetic energy
    fraction of :math:`\epsilon_B = 0.01`, the equipartition magnetic field is

    >>> equipartition_magnetic_field(
    ...     1e3 * u.erg / u.cm**3, 0.01
    ... )
    <Quantity 15.85330919 G>

    For a higher :math:`\epsilon_B`,

    >>> equipartition_magnetic_field(
    ...     1e3 * u.erg / u.cm**3, 0.1
    ... )
    <Quantity 50.13256549 G>

    """
    u_cgs = ensure_in_units(u_therm, u.erg / u.cm**3)
    return np.sqrt(8.0 * np.pi * epsilon_B * u_cgs) * u.G


def equipartition_electron_energy(u_therm: "_UnitBearingArrayLike", epsilon_e: "_ArrayLike") -> u.Quantity:
    r"""
    Compute the electron energy density implied by a thermal energy density.

    This function assumes that a fixed fraction of the thermal/internal energy
    density is placed into non-thermal electrons,

    .. math::

        u_e = \epsilon_e u_{\rm therm},

    where :math:`\epsilon_e` is the electron energy fraction.

    Bare numerical values for ``u_therm`` are interpreted as CGS energy
    densities, :math:`\mathrm{erg\,cm^{-3}}`. The returned electron energy
    density is always expressed in :math:`\mathrm{erg\,cm^{-3}}`.

    Parameters
    ----------
    u_therm : float, ~numpy.ndarray, or ~astropy.units.Quantity
        Thermal or internal energy density of the emitting/shocked region.
        If no units are attached, values are interpreted as
        :math:`\mathrm{erg\,cm^{-3}}`.
    epsilon_e : float or ~numpy.ndarray
        Fraction of the thermal energy density placed in electrons.
        Physically, this should usually satisfy
        :math:`0 \leq \epsilon_e \leq 1`, though this function does not
        enforce that bound.

    Returns
    -------
    u_e : ~astropy.units.Quantity
        Electron energy density in :math:`\mathrm{erg\,cm^{-3}}`.

    Notes
    -----
    This is a simple equipartition-style closure for the electron energy
    reservoir. It does not specify the shape, normalization, or cutoffs of the
    electron distribution function; it only computes the n_total electron energy
    density available to that distribution.

    Examples
    --------
    For a thermal energy density of :math:`10^3\,\mathrm{erg\,cm^{-3}}` and an
    electron energy fraction of :math:`\epsilon_e = 0.1`,

    >>> equipartition_electron_energy(
    ...     1e3 * u.erg / u.cm**3, 0.1
    ... )
    <Quantity 100. erg / cm3>

    Bare values are interpreted as :math:`\mathrm{erg\,cm^{-3}}`:

    >>> equipartition_electron_energy(1e3, 0.01)
    <Quantity 10. erg / cm3>
    """
    u_cgs = ensure_in_units(u_therm, u.erg / u.cm**3)
    return epsilon_e * u_cgs * (u.erg / u.cm**3)


# =========================================================================== #
# Distribution Base Class                                                     #
# =========================================================================== #
# This is the base class for all electron distributions. It defines the interface that all distributions must implement,
# and provides some common methods that can be used by all distributions.


def _f8(*values):
    """Convert each value to a ``float64`` array (0-d for scalars); used by the validated public wrappers."""
    return tuple(np.asarray(v, dtype="f8") for v in values)


class ElectronDistribution(ABC):
    r"""
    Abstract base class for relativistic electron energy distributions.

    Subclasses must implement :meth:`pdf` and :meth:`support`.  All other
    methods have default implementations built on :meth:`moment`, which
    itself falls back to numerical quadrature over :meth:`pdf`; subclasses
    should override :meth:`moment` with an analytic expression to make all
    dependent methods analytic.

    The interface is **stateless**: call class methods directly, passing
    distribution parameters as keyword arguments each time.

    Distribution parameters may be scalars or arrays. Public methods validate their inputs and then
    dispatch to private ``_opt_*`` backends, which skip validation and broadcast over array parameters
    using standard NumPy rules. The exceptions are quantities defined by a numerical integral
    (the Maxwell-Jüttner CDF and truncated moments, and moments above order two), which are
    evaluated one temperature at a time.

    See Also
    --------
    PowerLaw : Truncated power-law distribution.
    BrokenPowerLaw : Truncated broken power-law distribution.
    MaxwellJuettner : Relativistic thermal (Maxwell-Jüttner) distribution.
    MaxwellJuettnerPowerLaw : Mixed thermal plus power-law distribution.
    MaxwellJuettnerBrokenPowerLaw : Mixed thermal plus broken-power-law distribution.

    Examples
    --------
    Evaluate the power-law shape at a single Lorentz factor:

    >>> PowerLaw.pdf(10.0, p=2.0)
    0.01

    Compute the zeroth and second raw moments analytically:

    >>> PowerLaw.moment(0, p=3.0)
    0.5
    >>> PowerLaw.moment(2, p=4.0)
    1.0

    Freeze parameters with :meth:`as_callable` to obtain a single-argument
    shape function:

    >>> f = PowerLaw.as_callable(norm=1.0, p=2.0)
    >>> f(100.0)
    0.0001
    """

    # ---------------------------------------------- #
    # Parameter processing and backends              #
    # ---------------------------------------------- #
    # Every concrete distribution follows the two-level pattern used throughout trilobite:
    #
    #   * public methods (``pdf``, ``moment``, ``cdf``, ``normalize_from_*``, ...) validate their inputs with
    #     ``_validate_and_process_parameters``, which also converts them to ``float64`` arrays, and then
    #   * dispatch to private ``_opt_*`` backends, which do no validation or coercion and broadcast over array
    #     parameters using standard NumPy rules.
    #
    # The private ``_normalize_*`` and ``_bol_emiss_*`` methods are built on the backends and so are likewise
    # unguarded: callers must supply parameters that have already been validated.
    @classmethod
    def _validate_and_process_parameters(cls, **params) -> dict:
        r"""
        Validate distribution parameters and return them as ``float64`` arrays.

        The default implementation returns ``params`` unchanged; concrete distributions override it.
        """
        return params

    @classmethod
    def _opt_moment(cls, order: float, **params):
        r"""Unvalidated raw moment; falls back to :meth:`moment`, and analytic distributions override it."""
        return cls.moment(order, **params)

    # ---------------------------------------------- #
    # Utility Methods                                #
    # ---------------------------------------------- #
    @classmethod
    def as_callable(cls, norm: float = 1.0, **params):
        r"""
        Return a callable representation of the distribution.

        This method freezes the distribution amplitude and shape parameters,
        returning a single-argument function that evaluates the distribution at
        the supplied Lorentz factor.

        Parameters
        ----------
        norm : float, optional
            Distribution amplitude :math:`N_0`. Default is ``1.0``.
        **params
            Distribution-specific shape parameters.

        Returns
        -------
        callable
            Function with signature ``pdf(gamma)`` that evaluates
            :meth:`pdf` with the stored ``norm`` and distribution parameters.

        Examples
        --------
        >>> pdf = PowerLaw.as_callable(
        ...     norm=1.0, p=3.0, gamma_min=1.0
        ... )
        >>> pdf(10.0)
        0.001
        """
        norm = ensure_in_units(norm, u.cm**-3)

        def _pdf(gamma):
            return cls.pdf(gamma, norm=norm, **params)

        return _pdf

    # ---------------------------------------------- #
    # Statistical Methods (abstract)                 #
    # ---------------------------------------------- #
    @classmethod
    @abstractmethod
    def pdf(cls, gamma: "_ArrayLike", norm: float = 1.0, **params) -> np.ndarray:
        r"""
        Evaluate the electron Lorentz-factor distribution.

        This method returns the differential number density distribution
        :math:`N(\gamma)`, defined so that

        .. math::

            dN = N(\gamma)\,d\gamma

        is the number of electrons with Lorentz factors in the interval
        :math:`[\gamma, \gamma + d\gamma]`.

        The returned distribution is normalized by the multiplicative factor
        ``norm``. Subclasses define the shape of :math:`N(\gamma)` through their
        distribution-specific parameters, such as a power-law index or minimum and
        maximum Lorentz factors.

        Parameters
        ----------
        gamma : float or ~numpy.ndarray
            Electron Lorentz factor or array of Lorentz factors at which to
            evaluate the distribution.
        norm : float, optional
            Multiplicative normalization applied to the distribution. For example,
            if the subclass implements a shape function :math:`f(\gamma)`, this
            method should return :math:`N(\gamma) = \mathrm{norm}\,f(\gamma)`.
            Default is ``1.0``.
        **params
            Distribution-specific shape parameters. Common examples include
            ``p``, ``gamma_min``, ``gamma_max``, or cutoff parameters, depending on
            the concrete subclass.

        Returns
        -------
        N_gamma : ~numpy.ndarray
            Differential electron distribution evaluated at ``gamma``. Values
            outside the distribution support should generally be zero.

        Notes
        -----
        The ``norm`` parameter is part of the interface contract so that callers
        can optionally obtain the amplitude-scaled distribution in one call.  The
        built-in concrete subclasses return the shape function :math:`f(\gamma)`
        (i.e. they behave as if ``norm=1`` always); the amplitude :math:`N_0` is
        applied by the caller.  Subclasses that do need to apply ``norm`` inline
        should multiply the shape by ``norm`` before returning.
        """

    @classmethod
    @abstractmethod
    def support(cls, **params) -> tuple:
        r"""
        Return the non-zero Lorentz-factor support of the distribution.

        The support defines the interval over which :meth:`pdf` may be non-zero,

        .. math::

            \gamma_{\min} \leq \gamma \leq \gamma_{\max}.

        Subclasses should compute these bounds from their distribution-specific
        parameters. For distributions with a formally unbounded upper tail,
        ``gamma_max`` may be returned as ``numpy.inf``.

        Parameters
        ----------
        **params
            Distribution-specific parameters needed to determine the lower and
            upper Lorentz-factor bounds. Common examples include ``gamma_min``,
            ``gamma_max``, cutoff scales, or thermal parameters.

        Returns
        -------
        gamma_min, gamma_max : tuple
            Lower and upper bounds of the region where :meth:`pdf` is non-zero.
            The values should be returned as ``(gamma_min, gamma_max)``.

        Notes
        -----
        The support is used by normalization, moment, sampling, and validation
        routines. Implementations should therefore return bounds that are consistent
        with the behavior of :meth:`pdf`.
        """

    # ---------------------------------------------- #
    # Statistical Methods                            #
    # ---------------------------------------------- #
    @classmethod
    def n_total(cls, norm: float = 1.0, **params) -> float:
        r"""
        Compute the total number of electrons in the distribution.

        .. math::

            N_{\rm tot} = N_0\, M_0,

        where :math:`M_0 = \int_{\gamma_{\min}}^{\gamma_{\max}} f(\gamma)\,d\gamma`
        is the zeroth moment of the shape function and :math:`N_0 = \mathrm{norm}`.

        Parameters
        ----------
        norm : float, optional
            Distribution amplitude :math:`N_0`.  Default is ``1.0``.
        **params
            Distribution-specific shape parameters.

        Returns
        -------
        N_tot : float
            Total number of electrons.
        """
        return norm * cls.moment(0, **params)

    @classmethod
    def n_eff(cls, norm: float = 1.0, **params) -> float:
        r"""
        Compute the synchrotron-effective electron count.

        This method returns the :math:`\gamma^2`-weighted number of electrons,

        .. math::

            N_{\rm eff} = N_0\, M_2,

        where

        .. math::

            M_2 =
            \int_{\gamma_{\min}}^{\gamma_{\max}}
            \gamma^2 f(\gamma)\,d\gamma

        is the second raw moment of the shape function and
        :math:`N_0 = \mathrm{norm}` is the distribution amplitude.

        The :math:`\gamma^2` weighting is useful because the bolometric
        synchrotron power radiated by a single ultra-relativistic electron
        scales as

        .. math::

            P_{\rm syn} \propto \gamma^2 U_B,

        for magnetic energy density :math:`U_B = B^2/(8\pi)`. Therefore, the
        total bolometric synchrotron emissivity of a distribution is
        proportional to

        .. math::

            \int \gamma^2 N(\gamma)\,d\gamma
            =
            N_0
            \int \gamma^2 f(\gamma)\,d\gamma.

        This quantity is not the literal total number of electrons. Instead, it
        is an effective radiating electron count: the number of electrons the
        distribution would have if each electron contributed with unit
        :math:`\gamma^2` weight. It is mainly intended for synchrotron power,
        emissivity, and normalization calculations.

        Parameters
        ----------
        norm : float, optional
            Distribution amplitude :math:`N_0`. Default is ``1.0``.
        **params
            Distribution-specific shape parameters.

        Returns
        -------
        N_eff : float
            Synchrotron-effective electron count,
            :math:`N_{\rm eff} = \int \gamma^2 N(\gamma)\,d\gamma`.
        """
        return norm * cls.moment(2, **params)

    @classmethod
    def n_between(
        cls,
        a: float,
        b: float,
        norm: float = 1.0,
        **params,
    ) -> float:
        r"""
        Compute the number of electrons between two Lorentz factors.

        This method computes

        .. math::

            N(a \leq \gamma \leq b)
            =
            N_0
            \int_a^b f(\gamma)\,d\gamma,

        where :math:`f(\gamma)` is the unit-normalization shape function and
        :math:`N_0 = \mathrm{norm}` is the distribution amplitude. The bounds
        are clipped to the support of the distribution.

        Parameters
        ----------
        a : float
            Lower Lorentz-factor bound.
        b : float
            Upper Lorentz-factor bound.
        norm : float, optional
            Distribution amplitude :math:`N_0`. Default is ``1.0``.
        **params
            Distribution-specific shape parameters.

        Returns
        -------
        N_between : float
            Number of electrons with Lorentz factors in the interval
            ``[a, b]``.

        Notes
        -----
        If ``[a, b]`` does not overlap the distribution support, this method
        returns ``0.0``.
        """
        return norm * cls.moment_between(0, a, b, **params)

    @classmethod
    def n_eff_between(
        cls,
        a: float,
        b: float,
        norm: float = 1.0,
        **params,
    ) -> float:
        r"""
        Compute the synchrotron-effective electron count between two Lorentz factors.

        This method computes the :math:`\gamma^2`-weighted electron count,

        .. math::

            N_{\rm eff}(a \leq \gamma \leq b)
            =
            N_0
            \int_a^b \gamma^2 f(\gamma)\,d\gamma,

        where :math:`f(\gamma)` is the unit-normalization shape function and
        :math:`N_0 = \mathrm{norm}` is the distribution amplitude. The
        :math:`\gamma^2` weighting matches the scaling of the bolometric
        synchrotron power of a single relativistic electron.

        Parameters
        ----------
        a : float
            Lower Lorentz-factor bound.
        b : float
            Upper Lorentz-factor bound.
        norm : float, optional
            Distribution amplitude :math:`N_0`. Default is ``1.0``.
        **params
            Distribution-specific shape parameters.

        Returns
        -------
        N_eff_between : float
            Synchrotron-effective electron count in the interval ``[a, b]``.

        Notes
        -----
        If ``[a, b]`` does not overlap the distribution support, this method
        returns ``0.0``.
        """
        return norm * cls.moment_between(2, a, b, **params)

    @classmethod
    def cdf(
        cls,
        gamma: "_ArrayLike",
        norm: float = 1.0,
        **params,
    ) -> np.ndarray:
        r"""
        Evaluate the cumulative distribution function.

        .. math::

            F(\gamma)
            =
            N_0 \int_{\gamma_{\min}}^\gamma f(\gamma')\,d\gamma',

        where :math:`f(\gamma)` is the shape function and :math:`N_0 = \mathrm{norm}`.
        Values below the lower support bound return zero; values above the upper
        support bound return :meth:`n_total`.

        Parameters
        ----------
        gamma : float or ~numpy.ndarray
            Electron Lorentz factor or array of Lorentz factors at which to
            evaluate the cumulative distribution.
        norm : float, optional
            Distribution amplitude :math:`N_0`.  Default is ``1.0``.
        **params
            Distribution-specific shape parameters.

        Returns
        -------
        F_gamma : ~numpy.ndarray
            Cumulative integral of the electron distribution evaluated at ``gamma``.

        Notes
        -----
        Implemented by numerical quadrature over :meth:`pdf` with ``norm=1``; the
        amplitude is applied at the end.  Subclasses may override this method
        with an analytic expression.
        """
        gamma_min, gamma_max = cls.support(**params)
        gamma_array = np.asarray(gamma)
        scalar_input = gamma_array.ndim == 0
        gamma_flat = np.atleast_1d(gamma_array).astype(float)
        result = np.empty_like(gamma_flat, dtype=float)

        for index, gamma_value in enumerate(gamma_flat):
            if gamma_value <= gamma_min:
                result[index] = 0.0
            elif gamma_value >= gamma_max:
                result[index] = cls.moment(0, **params)
            else:
                result[index], _ = quad(
                    lambda g: cls.pdf(g, norm=1.0, **params), gamma_min, gamma_value, epsrel=1e-8, epsabs=0
                )

        if scalar_input:
            return norm * result[0]

        return norm * result.reshape(gamma_array.shape)

    @classmethod
    def sf(
        cls,
        gamma: "_ArrayLike",
        norm: float = 1.0,
        **params,
    ) -> np.ndarray:
        r"""
        Evaluate the survival function.

        The survival function is the complementary cumulative integral,

        .. math::

            S(\gamma) = N_0 \int_\gamma^{\gamma_{\max}} f(\gamma')\,d\gamma'
            = N_{\rm tot} - F(\gamma).

        Parameters
        ----------
        gamma : float or ~numpy.ndarray
            Electron Lorentz factor or array of Lorentz factors at which to
            evaluate the survival function.
        norm : float, optional
            Distribution amplitude :math:`N_0`.  Default is ``1.0``.
        **params
            Distribution-specific shape parameters.

        Returns
        -------
        S_gamma : ~numpy.ndarray
            Survival integral of the electron distribution evaluated at ``gamma``.
        """
        total = cls.n_total(norm=norm, **params)
        return total - cls.cdf(gamma, norm=norm, **params)

    @classmethod
    def logpdf(
        cls,
        gamma: "_ArrayLike",
        norm: float = 1.0,
        **params,
    ) -> np.ndarray:
        r"""
        Evaluate the natural logarithm of the differential distribution.

        Parameters
        ----------
        gamma : float or ~numpy.ndarray
            Electron Lorentz factor or array of Lorentz factors.
        norm : float, optional
            Distribution amplitude :math:`N_0`.  Default is ``1.0``.
        **params
            Distribution-specific shape parameters.

        Returns
        -------
        log_N_gamma : ~numpy.ndarray
            Natural logarithm of :math:`N_0 f(\gamma)` evaluated at ``gamma``.
        """
        return np.log(norm) + np.log(cls.pdf(gamma, norm=1.0, **params))

    @classmethod
    def logcdf(
        cls,
        gamma: "_ArrayLike",
        norm: float = 1.0,
        **params,
    ) -> np.ndarray:
        r"""
        Evaluate the natural logarithm of the cumulative distribution function.

        Parameters
        ----------
        gamma : float or ~numpy.ndarray
            Electron Lorentz factor or array of Lorentz factors.
        norm : float, optional
            Distribution amplitude :math:`N_0`.  Default is ``1.0``.
        **params
            Distribution-specific shape parameters.

        Returns
        -------
        log_F_gamma : ~numpy.ndarray
            Natural logarithm of :meth:`cdf` evaluated at ``gamma``.
        """
        return np.log(cls.cdf(gamma, norm=norm, **params))

    @classmethod
    def logsf(
        cls,
        gamma: "_ArrayLike",
        norm: float = 1.0,
        **params,
    ) -> np.ndarray:
        r"""
        Evaluate the natural logarithm of the survival function.

        Parameters
        ----------
        gamma : float or ~numpy.ndarray
            Electron Lorentz factor or array of Lorentz factors.
        norm : float, optional
            Distribution amplitude :math:`N_0`.  Default is ``1.0``.
        **params
            Distribution-specific shape parameters.

        Returns
        -------
        log_S_gamma : ~numpy.ndarray
            Natural logarithm of :meth:`sf` evaluated at ``gamma``.
        """
        return np.log(cls.sf(gamma, norm=norm, **params))

    @classmethod
    def moment(
        cls,
        order: float,
        **params,
    ) -> float:
        r"""
        Compute the :math:`k`-th raw moment of the shape function.

        .. math::

            M_k = \int_{\gamma_{\min}}^{\gamma_{\max}} \gamma^k\, f(\gamma)\,d\gamma,

        where :math:`f(\gamma)` is the shape function returned by :meth:`pdf`
        with ``norm=1``.  The distribution amplitude :math:`N_0` is not
        included; callers that need a physical quantity multiply by it
        externally (see :meth:`n_total`, :meth:`n_eff`).

        All integration-based methods (:meth:`n_total`, :meth:`n_eff`,
        :meth:`mean`, :meth:`var`, :meth:`std`) delegate to this method.
        Subclasses should override it with an analytic expression where one
        is available.

        Parameters
        ----------
        order : float
            Moment order :math:`k`.  ``order=0`` gives the norm of the shape
            function, ``order=1`` the first raw moment, ``order=2`` the second.
        **params
            Distribution-specific shape parameters.

        Returns
        -------
        M_k : float
            Raw :math:`k`-th moment of the shape function.

        Notes
        -----
        The default implementation uses numerical quadrature over :meth:`pdf`
        with ``norm=1``.  Subclasses should override this method with an
        analytic expression when one is available.
        """
        gamma_min, gamma_max = cls.support(**params)
        result, _ = quad(
            lambda gamma: gamma**order * cls.pdf(gamma, norm=1.0, **params),
            gamma_min,
            gamma_max,
        )
        return result

    @classmethod
    def moment_between(
        cls,
        order: float,
        a: float,
        b: float,
        **params,
    ) -> float:
        r"""
        Compute a truncated raw moment of the shape function.

        This method computes

        .. math::

            M_k(a,b)
            =
            \int_a^b \gamma^k f(\gamma)\,d\gamma,

        where :math:`f(\gamma)` is the shape function returned by :meth:`pdf`
        with ``norm=1``. The integration limits are clipped to the support of
        the distribution.

        Parameters
        ----------
        order : float
            Moment order :math:`k`.
        a : float
            Lower Lorentz-factor bound.
        b : float
            Upper Lorentz-factor bound.
        **params
            Distribution-specific shape parameters.

        Returns
        -------
        M_k_between : float
            Truncated raw moment of the shape function over the interval
            ``[a, b]``.

        Notes
        -----
        If the requested interval does not overlap the distribution support,
        this method returns ``0.0``.
        """
        gamma_min, gamma_max = cls.support(**params)

        lower = max(float(a), gamma_min)
        upper = min(float(b), gamma_max)

        if upper <= lower:
            return 0.0

        result, _ = quad(
            lambda gamma: gamma**order * cls.pdf(gamma, norm=1.0, **params),
            lower,
            upper,
        )

        return result

    @classmethod
    def mean(cls, **params) -> float:
        r"""
        Compute the mean electron Lorentz factor.

        .. math::

            \langle \gamma \rangle = \frac{M_1}{M_0},

        where :math:`M_k` is the :math:`k`-th moment of the shape function.
        This quantity is independent of the distribution amplitude.

        Parameters
        ----------
        **params
            Distribution-specific shape parameters.

        Returns
        -------
        mean_gamma : float
            Mean electron Lorentz factor.
        """
        m0 = cls.moment(0, **params)
        if m0 == 0.0:
            return np.nan
        return cls.moment(1, **params) / m0

    @classmethod
    def var(cls, **params) -> float:
        r"""
        Compute the variance of the electron Lorentz factor.

        .. math::

            \operatorname{Var}(\gamma) = \frac{M_2}{M_0} - \left(\frac{M_1}{M_0}\right)^2.

        This quantity is independent of the distribution amplitude.

        Parameters
        ----------
        **params
            Distribution-specific shape parameters.

        Returns
        -------
        var_gamma : float
            Variance of the electron Lorentz factor.
        """
        m0 = cls.moment(0, **params)
        if m0 == 0.0:
            return np.nan
        m1 = cls.moment(1, **params)
        m2 = cls.moment(2, **params)
        return m2 / m0 - (m1 / m0) ** 2

    @classmethod
    def std(cls, **params) -> float:
        r"""
        Compute the standard deviation of the electron Lorentz factor.

        .. math::

            \sigma_\gamma = \sqrt{\operatorname{Var}(\gamma)}.

        This quantity is independent of the distribution amplitude.

        Parameters
        ----------
        **params
            Distribution-specific shape parameters.

        Returns
        -------
        std_gamma : float
            Standard deviation of the electron Lorentz factor.
        """
        return np.sqrt(cls.var(**params))

    @classmethod
    def mean_energy(cls, **params) -> float:
        r"""
        Compute the mean total relativistic electron energy.

        The mean total energy is

        .. math::

            \langle E \rangle
            =
            m_e c^2 \langle \gamma \rangle
            =
            m_e c^2
            \frac{M_1}{M_0},

        where :math:`M_k` is the :math:`k`-th raw moment of the electron
        Lorentz-factor shape function.

        This includes the electron rest-mass energy and follows the standard
        ultra-relativistic synchrotron convention in which electron energies
        are written as :math:`E = \gamma m_e c^2`.

        Parameters
        ----------
        **params
            Distribution-specific shape parameters.

        Returns
        -------
        mean_energy : float
            Mean total relativistic electron energy in erg.
        """
        return electron_rest_energy_cgs * cls.mean(**params)

    @classmethod
    def mean_kinetic_energy(cls, **params) -> float:
        r"""
        Compute the mean kinetic electron energy.

        The mean kinetic energy is

        .. math::

            \langle E_{\rm kin} \rangle
            =
            m_e c^2 \langle \gamma - 1 \rangle
            =
            m_e c^2
            \left(
                \frac{M_1}{M_0} - 1
            \right),

        where :math:`M_k` is the :math:`k`-th raw moment of the electron
        Lorentz-factor shape function.

        This excludes the electron rest-mass contribution and is useful when
        comparing to an explicitly kinetic/internal electron energy budget.

        Parameters
        ----------
        **params
            Distribution-specific shape parameters.

        Returns
        -------
        mean_kinetic_energy : float
            Mean kinetic electron energy in erg.
        """
        return electron_rest_energy_cgs * (cls.mean(**params) - 1.0)

    # ------------------------------------------------------------------ #
    # Normalization (Private)                                            #
    # ------------------------------------------------------------------ #
    # These methods all concern the normalization of the distribution in various scenarios
    # and from various physical parameters. We provide the typical public / private split here
    # because many of these methods have unit-dependencies.
    #
    # All methods here assume CGS units.

    @classmethod
    def _normalize_from_n_total(cls, n_total: float, **params) -> float:
        return n_total / cls._opt_moment(0.0, **params)

    @classmethod
    def _normalize_from_n_eff(cls, n_eff: float, **params) -> float:
        return n_eff / cls._opt_moment(2.0, **params)

    @classmethod
    def _normalize_from_magnetic_field(cls, B: float, epsilon_B: float, epsilon_E: float, **params) -> float:
        u_B = B**2 / (8.0 * np.pi)
        return (epsilon_E / epsilon_B) * u_B / (electron_rest_energy_cgs * cls._opt_moment(1.0, **params))

    @classmethod
    def _normalize_from_energy_density(cls, u_therm: float, epsilon_E: float, **params) -> float:
        return epsilon_E * u_therm / (electron_rest_energy_cgs * cls._opt_moment(1.0, **params))

    # ------------------------------------------------------------------ #
    # Normalization (Public)                                             #
    # ------------------------------------------------------------------ #
    @classmethod
    def normalize_from_n_total(cls, n_total, **params) -> u.Quantity:
        r"""
        Compute the distribution's normalization from a known total electron number density.

        Parameters
        ----------
        n_total : float, array-like, or ~astropy.units.Quantity
            Total electron number density in :math:`\\mathrm{cm^{-3}}`.
        **params
            Distribution-specific parameters.

        Returns
        -------
        ~astropy.units.Quantity
            Distribution amplitude :math:`N_0` in :math:`\\mathrm{cm^{-3}}`.
        """
        n_cgs = ensure_in_units(n_total, u.cm**-3)
        params = cls._validate_and_process_parameters(**params)
        return cls._normalize_from_n_total(n_cgs, **params) * u.cm**-3

    @classmethod
    def normalize_from_n_eff(cls, n_eff, **params) -> u.Quantity:
        r"""
        Compute the normalization from a known effective radiating electron number density.

        Parameters
        ----------
        n_eff : float, array-like, or ~astropy.units.Quantity
            Effective electron number density in :math:`\\mathrm{cm^{-3}}`.
        **params
            Distribution-specific parameters.

        Returns
        -------
        ~astropy.units.Quantity
            Distribution amplitude :math:`N_0` in :math:`\\mathrm{cm^{-3}}`.
        """
        n_cgs = ensure_in_units(n_eff, u.cm**-3)
        params = cls._validate_and_process_parameters(**params)
        return cls._normalize_from_n_eff(n_cgs, **params) * u.cm**-3

    @classmethod
    def normalize_from_magnetic_field(cls, B, epsilon_B: float, epsilon_E: float, **params) -> u.Quantity:
        r"""
        Compute N₀ via equipartition with a magnetic field.

        Solves

        .. math::

            N_0\, m_e c^2\, M^{(1)} = \\frac{\\varepsilon_E}{\\varepsilon_B}
            \\frac{B^2}{8\pi}

        for :math:`N_0`, where :math:`M^{(1)}` is the first moment of
        :meth:`pdf`.

        Parameters
        ----------
        B : float, array-like, or ~astropy.units.Quantity
            Magnetic field strength.  Bare values are interpreted as Gauss.
        epsilon_B : float or array-like
            Fraction of thermal energy in the magnetic field.
        epsilon_E : float or array-like
            Fraction of thermal energy in relativistic electrons.
        **params
            Distribution-specific parameters.

        Returns
        -------
        ~astropy.units.Quantity
            Power-law normalization :math:`N_0` in :math:`\\mathrm{cm^{-3}}`.
        """
        B_cgs = ensure_in_units(B, u.G)
        params = cls._validate_and_process_parameters(**params)
        return cls._normalize_from_magnetic_field(B_cgs, epsilon_B, epsilon_E, **params) * u.cm**-3

    @classmethod
    def normalize_from_energy_density(cls, u_therm, epsilon_E: float, **params) -> u.Quantity:
        r"""
        Compute N₀ from a fraction ε_E of the thermal energy density.

        Solves

        .. math::

            N_0\, m_e c^2\, M^{(1)} = \\varepsilon_E\, u_{\\rm int}

        for :math:`N_0`.

        Parameters
        ----------
        u_therm : float, array-like, or ~astropy.units.Quantity
            Thermal energy density.  Bare values are interpreted as
            :math:`\\mathrm{erg\,cm^{-3}}`.
        epsilon_E : float or array-like
            Fraction of thermal energy in relativistic electrons.
        **params
            Distribution-specific parameters.

        Returns
        -------
        ~astropy.units.Quantity
            Distribution amplitude :math:`N_0` in :math:`\\mathrm{cm^{-3}}`.
        """
        u_cgs = ensure_in_units(u_therm, u.erg / u.cm**3)
        params = cls._validate_and_process_parameters(**params)
        return cls._normalize_from_energy_density(u_cgs, epsilon_E, **params) * u.cm**-3

    # ------------------------------------------------------------------ #
    # Power-law integral helper                                          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _integral_power_law(
        exponent: "_ArrayLike",
        lower: "_ArrayLike",
        upper: "_ArrayLike",
    ) -> Union[float, np.ndarray]:
        r"""
        Evaluate a definite power-law integral, broadcasting over array inputs.

        This is an unvalidated backend: inputs must already be floats or ``float64`` arrays.

        Computes

        .. math::

            I(q; a, b)
            =
            \int_a^b \gamma^{q-1}\,d\gamma
            =
            \begin{cases}
            \dfrac{b^q - a^q}{q}, & q \neq 0, \\\\
            \ln(b/a), & q = 0.
            \end{cases}

        Parameters
        ----------
        exponent : float
            Integral exponent :math:`q`.
        lower : float
            Lower integration bound.
        upper : float
            Upper integration bound.

        Returns
        -------
        integral : float or ~numpy.ndarray
            Broadcast shape of the inputs. Elements with ``upper <= lower`` are ``0.0``.
        """
        if np.ndim(exponent) == 0:
            # Common case (scalar index): pick the form once in Python and skip the masking machinery.
            if abs(exponent) <= 1e-14:
                integral = np.log(upper / lower)
            else:
                integral = (upper**exponent - lower**exponent) / exponent
            return np.where(upper <= lower, 0.0, integral)[()]

        is_log = np.abs(exponent) <= 1e-14  # == isclose(exponent, 0, rtol=0, atol=1e-14), without its overhead
        safe_exponent = np.where(is_log, 1.0, exponent)  # avoid 0/0 in the unused branch
        power_form = (upper**safe_exponent - lower**safe_exponent) / safe_exponent
        log_form = np.log(upper / lower)
        return np.where(upper <= lower, 0.0, np.where(is_log, log_form, power_form))[()]

    # ------------------------------------------------------------------ #
    # Bolometric emissivity (Private)                                    #
    # ------------------------------------------------------------------ #

    @classmethod
    def _bol_emiss_from_magnetic_field(cls, B: float, epsilon_B: float, epsilon_E: float, **params) -> float:
        norm = cls._normalize_from_magnetic_field(B, epsilon_B, epsilon_E, **params)
        U_B = B**2 / (8.0 * np.pi)
        return (4.0 / 3.0) * sigma_T_cgs * c_cgs * U_B * norm * cls._opt_moment(2.0, **params)

    @classmethod
    def _bol_emiss_from_energy_density(cls, u_therm: float, epsilon_B: float, epsilon_E: float, **params) -> float:
        B = np.sqrt(8.0 * np.pi * epsilon_B * u_therm)
        return cls._bol_emiss_from_magnetic_field(B, epsilon_B, epsilon_E, **params)

    # ------------------------------------------------------------------ #
    # Bolometric emissivity (Public)                                     #
    # ------------------------------------------------------------------ #
    @classmethod
    def bol_emiss_from_magnetic_field(cls, B, epsilon_B: float, epsilon_E: float, **params) -> u.Quantity:
        r"""
        Bolometric synchrotron emissivity from a magnetic field via equipartition.

        Derives :math:`N_0` from :math:`B`, :math:`\varepsilon_B`, and
        :math:`\varepsilon_E` using :meth:`normalize_from_magnetic_field`, then
        computes

        .. math::

            j = \frac{4}{3}\,\sigma_T c\,\frac{B^2}{8\pi}\,N_0\,M^{(2)},

        where :math:`M^{(2)} = \int \gamma^2 f(\gamma)\,d\gamma` is the second
        moment of the shape function (see :meth:`n_eff`).

        Parameters
        ----------
        B : float, array-like, or ~astropy.units.Quantity
            Magnetic field strength.  Bare values are interpreted as Gauss.
        epsilon_B : float or array-like
            Fraction of thermal energy in the magnetic field.
        epsilon_E : float or array-like
            Fraction of thermal energy in relativistic electrons.
        **params
            Distribution-specific parameters.

        Returns
        -------
        ~astropy.units.Quantity
            Bolometric synchrotron emissivity in
            :math:`\mathrm{erg\,s^{-1}\,cm^{-3}}`.
        """
        B_cgs = ensure_in_units(B, u.G)
        params = cls._validate_and_process_parameters(**params)
        return cls._bol_emiss_from_magnetic_field(B_cgs, epsilon_B, epsilon_E, **params) * (u.erg / u.s / u.cm**3)

    @classmethod
    def bol_emiss_from_energy_density(cls, u_therm, epsilon_B: float, epsilon_E: float, **params) -> u.Quantity:
        r"""
        Bolometric synchrotron emissivity from thermal energy density via equipartition.

        Derives both :math:`B` and :math:`N_0` from ``u_therm``, ``epsilon_B``,
        and ``epsilon_E`` via the equipartition closures, then computes the
        emissivity as in :meth:`bol_emiss_from_magnetic_field`.

        Parameters
        ----------
        u_therm : float, array-like, or ~astropy.units.Quantity
            Thermal energy density.  Bare values are interpreted as
            :math:`\mathrm{erg\,cm^{-3}}`.
        epsilon_B : float or array-like
            Fraction of thermal energy in the magnetic field.
        epsilon_E : float or array-like
            Fraction of thermal energy in relativistic electrons.
        **params
            Distribution-specific parameters.

        Returns
        -------
        ~astropy.units.Quantity
            Bolometric synchrotron emissivity in
            :math:`\mathrm{erg\,s^{-1}\,cm^{-3}}`.
        """
        u_cgs = ensure_in_units(u_therm, u.erg / u.cm**3)
        params = cls._validate_and_process_parameters(**params)
        return cls._bol_emiss_from_energy_density(u_cgs, epsilon_B, epsilon_E, **params) * (u.erg / u.s / u.cm**3)


# =========================================================================== #
# Distribution Implementations                                                #
# =========================================================================== #
class PowerLaw(ElectronDistribution):
    r"""
    Truncated power-law distribution of electron Lorentz factors.

    The shape function is

    .. math::

        f(\gamma) = \gamma^{-p},

    over the interval

    .. math::

        \gamma_{\min} \leq \gamma \leq \gamma_{\max},

    and is zero outside this interval. The full distribution is

    .. math::

        N(\gamma) = N_0 f(\gamma),

    where :math:`N_0` is supplied through the ``norm`` argument.

    .. rubric:: Distribution parameters

    .. list-table::
        :header-rows: 1
        :widths: 20 20 60

        * - Parameter
          - Type
          - Description
        * - ``p``
          - float
          - Power-law index of the distribution. Must be positive.
        * - ``gamma_min``
          - float, optional
          - Minimum Lorentz factor. Must satisfy ``gamma_min >= 1``.
            Default is ``1.0``.
        * - ``gamma_max``
          - float, optional
          - Maximum Lorentz factor. Must satisfy
            ``gamma_max > gamma_min``. Default is ``numpy.inf``.

    Notes
    -----
    Raw moments are computed analytically as

    .. math::

        M_k =
        \int_{\gamma_{\min}}^{\gamma_{\max}}
        \gamma^k f(\gamma)\,d\gamma
        =
        \int_{\gamma_{\min}}^{\gamma_{\max}}
        \gamma^{k-p}\,d\gamma.

    Therefore,

    .. math::

        M_k =
        \begin{cases}
        \dfrac{
            \gamma_{\max}^{k+1-p}
            -
            \gamma_{\min}^{k+1-p}
        }{
            k+1-p
        },
        & k + 1 - p \neq 0, \\\\
        \ln\left(\dfrac{\gamma_{\max}}{\gamma_{\min}}\right),
        & k + 1 - p = 0.
        \end{cases}

    This implementation treats ``p``, ``gamma_min``, ``gamma_max``, and
    ``norm`` as scalar parameters. The Lorentz factor ``gamma`` may be scalar
    or array-like.

    See Also
    --------
    BrokenPowerLaw : Power-law distribution with a spectral break.
    MaxwellJuettner : Relativistic thermal (Maxwell-Jüttner) distribution.
    MaxwellJuettnerPowerLaw : Mixed thermal plus power-law distribution.

    Examples
    --------
    Evaluate the shape function at a single Lorentz factor:

    >>> PowerLaw.pdf(10.0, p=2.0)
    0.01

    Evaluate at an array of Lorentz factors with a finite upper cutoff:

    >>> import numpy as np
    >>> PowerLaw.pdf(np.array([1.0, 10.0, 1e3]), p=2.0, gamma_max=100.0)
    array([1.  , 0.01, 0.  ])

    Compute the zeroth and second raw moments analytically (both have
    closed forms for the truncated power-law):

    >>> PowerLaw.moment(0, p=3.0)
    0.5
    >>> PowerLaw.moment(2, p=4.0)
    1.0

    Compute the mean Lorentz factor :math:`\langle\gamma\rangle = M_1/M_0`:

    >>> PowerLaw.mean(p=4.0)
    1.5

    Normalize to a known total electron number density of
    :math:`10^{-3}\,\mathrm{cm^{-3}}`:

    >>> import astropy.units as u
    >>> PowerLaw.normalize_from_n_total(1e-3 * u.cm**-3, p=3.0)
    <Quantity 0.002 1 / cm3>
    """

    # ------------------------------------------------------------------ #
    # Helper Methods                                                     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_parameters(
        p: float,
        gamma_min: float = 1.0,
        gamma_max: float = np.inf,
    ) -> None:
        r"""
        Validate power-law parameters.

        Parameters
        ----------
        p : float
            Power-law index.
        gamma_min : float, optional
            Minimum Lorentz factor. Default is ``1.0``.
        gamma_max : float, optional
            Maximum Lorentz factor. Default is ``numpy.inf``.

        Raises
        ------
        ValueError
            If the power-law index is non-finite or non-positive, if
            ``gamma_min < 1``, or if ``gamma_max <= gamma_min``.
        """
        p, gamma_min, gamma_max = _f8(p, gamma_min, gamma_max)

        if not np.all(np.isfinite(p)):
            raise ValueError("Power-law index `p` must be finite.")

        if np.any(p <= 0.0):
            raise ValueError("Power-law index `p` must be positive.")

        if np.any(gamma_min < 1.0):
            raise ValueError("`gamma_min` must be greater than or equal to 1.")

        if np.any(gamma_max <= gamma_min):
            raise ValueError("`gamma_max` must be greater than `gamma_min`.")

    @classmethod
    def _validate_and_process_parameters(cls, *, p, gamma_min=1.0, gamma_max=np.inf, **_) -> dict:
        r"""Validate the power-law parameters and return them as ``float64`` arrays."""
        cls._validate_parameters(p, gamma_min=gamma_min, gamma_max=gamma_max)
        p, gamma_min, gamma_max = _f8(p, gamma_min, gamma_max)
        return {"p": p, "gamma_min": gamma_min, "gamma_max": gamma_max}

    # ------------------------------------------------------------------ #
    # Unvalidated backends                                               #
    # ------------------------------------------------------------------ #
    # No validation or coercion; floats or float64 arrays, broadcast with standard NumPy rules.
    @classmethod
    def _opt_pdf(cls, gamma, norm, p, gamma_min, gamma_max, **_):
        inside_support = (gamma >= gamma_min) & (gamma <= gamma_max)
        return np.where(inside_support, norm * gamma ** (-p), 0.0)[()]

    @classmethod
    def _opt_moment(cls, order, p, gamma_min, gamma_max, **_):
        return cls._integral_power_law(order + 1.0 - p, gamma_min, gamma_max)

    @classmethod
    def _opt_moment_between(cls, order, a, b, p, gamma_min, gamma_max, **_):
        lower = np.maximum(a, gamma_min)
        upper = np.minimum(b, gamma_max)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):  # masked-out empty intervals
            return cls._integral_power_law(order + 1.0 - p, lower, upper)

    @classmethod
    def _opt_cdf(cls, gamma, norm, p, gamma_min, gamma_max, **_):
        return norm * cls._opt_moment_between(0.0, gamma_min, gamma, p=p, gamma_min=gamma_min, gamma_max=gamma_max)

    # ------------------------------------------------------------------ #
    # Distribution Interface                                             #
    # ------------------------------------------------------------------ #
    @classmethod
    def pdf(
        cls,
        gamma: "_ArrayLike",
        norm: float = 1.0,
        *,
        p: float,
        gamma_min: float = 1.0,
        gamma_max: float = np.inf,
        **_,
    ) -> np.ndarray:
        r"""
        Evaluate the power-law distribution.

        This method returns

        .. math::

            N(\gamma)
            =
            N_0 \gamma^{-p},

        for :math:`\gamma_{\min} \leq \gamma \leq \gamma_{\max}`, and zero
        outside this interval.

        Parameters
        ----------
        gamma : float or array-like
            Lorentz factor or Lorentz factors at which to evaluate the
            distribution.
        norm : float, optional
            Distribution amplitude :math:`N_0`. Default is ``1.0``.
        p : float
            Power-law index.
        gamma_min : float, optional
            Minimum Lorentz factor. Default is ``1.0``.
        gamma_max : float, optional
            Maximum Lorentz factor. Default is ``numpy.inf``.
        **_
            Ignored keyword arguments, accepted for API compatibility.

        Returns
        -------
        N_gamma : float or ~numpy.ndarray
            Distribution evaluated at ``gamma``.
        """
        params = cls._validate_and_process_parameters(p=p, gamma_min=gamma_min, gamma_max=gamma_max)
        gamma, norm = _f8(gamma, norm)
        return cls._opt_pdf(gamma, norm, **params)

    @classmethod
    def support(
        cls,
        *,
        gamma_min: float = 1.0,
        gamma_max: float = np.inf,
        **_,
    ) -> tuple[float, float]:
        r"""
        Return the non-zero support of the power-law distribution.

        Parameters
        ----------
        gamma_min : float, optional
            Minimum Lorentz factor. Default is ``1.0``.
        gamma_max : float, optional
            Maximum Lorentz factor. Default is ``numpy.inf``.
        **_
            Ignored keyword arguments, accepted for API compatibility.

        Returns
        -------
        gamma_min, gamma_max : tuple of float
            Lower and upper Lorentz-factor bounds.
        """
        gamma_min, gamma_max = _f8(gamma_min, gamma_max)

        if np.any(gamma_min < 1.0):
            raise ValueError("`gamma_min` must be greater than or equal to 1.")

        if np.any(gamma_max <= gamma_min):
            raise ValueError("`gamma_max` must be greater than `gamma_min`.")

        return gamma_min[()], gamma_max[()]

    @classmethod
    def moment(
        cls,
        order: float,
        *,
        p: float,
        gamma_min: float = 1.0,
        gamma_max: float = np.inf,
        **_,
    ) -> float:
        r"""
        Compute an analytic raw moment of the power-law shape function.

        This method computes

        .. math::

            M_k =
            \int_{\gamma_{\min}}^{\gamma_{\max}}
            \gamma^k \gamma^{-p}\,d\gamma.

        Parameters
        ----------
        order : float
            Moment order :math:`k`.
        p : float
            Power-law index.
        gamma_min : float, optional
            Minimum Lorentz factor. Default is ``1.0``.
        gamma_max : float, optional
            Maximum Lorentz factor. Default is ``numpy.inf``.
        **_
            Ignored keyword arguments, accepted for API compatibility.

        Returns
        -------
        M_k : float
            Raw moment of the shape function.
        """
        params = cls._validate_and_process_parameters(p=p, gamma_min=gamma_min, gamma_max=gamma_max)
        return cls._opt_moment(float(order), **params)

    @classmethod
    def moment_between(
        cls,
        order: float,
        a: float,
        b: float,
        *,
        p: float,
        gamma_min: float = 1.0,
        gamma_max: float = np.inf,
        **_,
    ) -> float:
        r"""
        Compute an analytic truncated raw moment of the shape function.

        This method computes

        .. math::

            M_k(a,b)
            =
            \int_a^b
            \gamma^k \gamma^{-p}\,d\gamma,

        after clipping the requested interval to the distribution support.

        Parameters
        ----------
        order : float
            Moment order :math:`k`.
        a : float
            Requested lower Lorentz-factor bound.
        b : float
            Requested upper Lorentz-factor bound.
        p : float
            Power-law index.
        gamma_min : float, optional
            Minimum Lorentz factor. Default is ``1.0``.
        gamma_max : float, optional
            Maximum Lorentz factor. Default is ``numpy.inf``.
        **_
            Ignored keyword arguments, accepted for API compatibility.

        Returns
        -------
        M_k_between : float
            Truncated raw moment over the clipped interval. Returns ``0.0`` if
            the requested interval does not overlap the support.
        """
        params = cls._validate_and_process_parameters(p=p, gamma_min=gamma_min, gamma_max=gamma_max)
        a, b = _f8(a, b)
        return cls._opt_moment_between(float(order), a, b, **params)

    @classmethod
    def cdf(
        cls,
        gamma: "_ArrayLike",
        norm: float = 1.0,
        *,
        p: float,
        gamma_min: float = 1.0,
        gamma_max: float = np.inf,
        **_,
    ) -> np.ndarray:
        r"""
        Evaluate the cumulative power-law distribution analytically.

        This method computes

        .. math::

            F(\gamma)
            =
            N_0
            \int_{\gamma_{\min}}^\gamma
            \gamma'^{-p}\,d\gamma',

        with values below :math:`\gamma_{\min}` clipped to zero and values
        above :math:`\gamma_{\max}` clipped to the total integral.

        Parameters
        ----------
        gamma : float or array-like
            Lorentz factor or Lorentz factors at which to evaluate the
            cumulative distribution.
        norm : float, optional
            Distribution amplitude :math:`N_0`. Default is ``1.0``.
        p : float
            Power-law index.
        gamma_min : float, optional
            Minimum Lorentz factor. Default is ``1.0``.
        gamma_max : float, optional
            Maximum Lorentz factor. Default is ``numpy.inf``.
        **_
            Ignored keyword arguments, accepted for API compatibility.

        Returns
        -------
        F_gamma : float or ~numpy.ndarray
            Cumulative distribution evaluated at ``gamma``.
        """
        params = cls._validate_and_process_parameters(p=p, gamma_min=gamma_min, gamma_max=gamma_max)
        gamma, norm = _f8(gamma, norm)
        return cls._opt_cdf(gamma, norm, **params)


class BrokenPowerLaw(ElectronDistribution):
    r"""
    Truncated broken power-law distribution of electron Lorentz factors.

    The shape function is

    .. math::

        f(\gamma)
        =
        \begin{cases}
        \left(\gamma / \gamma_c\right)^{-p_1},
        & \gamma_{\min} \leq \gamma < \gamma_c, \\\\
        \left(\gamma / \gamma_c\right)^{-p_2},
        & \gamma_c \leq \gamma \leq \gamma_{\max},
        \end{cases}

    and is zero outside the interval
    :math:`\gamma_{\min} \leq \gamma \leq \gamma_{\max}`. The full
    distribution is

    .. math::

        N(\gamma) = N_0 f(\gamma),

    where :math:`N_0` is supplied through the ``norm`` argument.

    The distribution is continuous at :math:`\gamma_c`, since both branches
    evaluate to :math:`N_0` at the break.

    .. rubric:: Distribution parameters

    .. list-table::
        :header-rows: 1
        :widths: 20 20 60

        * - Parameter
          - Type
          - Description
        * - ``p1``
          - float
          - Power-law index below the break. Must be positive.
        * - ``p2``
          - float
          - Power-law index above the break. Must be positive.
        * - ``gamma_c``
          - float
          - Break Lorentz factor. Must satisfy
            ``gamma_min < gamma_c < gamma_max``.
        * - ``gamma_min``
          - float, optional
          - Minimum Lorentz factor. Must satisfy ``gamma_min >= 1``.
            Default is ``1.0``.
        * - ``gamma_max``
          - float, optional
          - Maximum Lorentz factor. Must satisfy
            ``gamma_max > gamma_min``. Default is ``numpy.inf``.

    Notes
    -----
    Moments are computed analytically by splitting the integral at the break:

    .. math::

        M_k
        =
        \int_{\gamma_{\min}}^{\gamma_c}
        \gamma^k
        \left(\gamma / \gamma_c\right)^{-p_1}
        d\gamma
        +
        \int_{\gamma_c}^{\gamma_{\max}}
        \gamma^k
        \left(\gamma / \gamma_c\right)^{-p_2}
        d\gamma.

    This implementation treats ``p1``, ``p2``, ``gamma_c``, ``gamma_min``,
    ``gamma_max``, and ``norm`` as scalar parameters. The Lorentz factor
    ``gamma`` may be scalar or array-like.

    See Also
    --------
    PowerLaw : Single power-law distribution without a spectral break.
    MaxwellJuettner : Relativistic thermal (Maxwell-Jüttner) distribution.
    MaxwellJuettnerBrokenPowerLaw : Mixed thermal plus broken-power-law distribution.

    Examples
    --------
    Evaluate below and above the spectral break at :math:`\gamma_c = 10`:

    >>> BrokenPowerLaw.pdf(5.0, p1=2.0, p2=3.0, gamma_c=10.0)
    4.0
    >>> BrokenPowerLaw.pdf(20.0, p1=2.0, p2=3.0, gamma_c=10.0)
    0.125

    The distribution is continuous at :math:`\gamma_c` (both branches
    evaluate to :math:`N_0` at the break):

    >>> BrokenPowerLaw.pdf(10.0, p1=2.0, p2=3.0, gamma_c=10.0)
    1.0

    Evaluate on an array spanning the break:

    >>> import numpy as np
    >>> BrokenPowerLaw.pdf(np.array([5.0, 10.0, 20.0]), p1=2.0, p2=3.0, gamma_c=10.0)
    array([4.   , 1.   , 0.125])

    Normalize to a total electron number density of
    :math:`10^{-3}\,\mathrm{cm^{-3}}`:

    >>> import astropy.units as u
    >>> BrokenPowerLaw.normalize_from_n_total(
    ...     1e-3 * u.cm**-3, p1=2.0, p2=4.0, gamma_c=10.0
    ... )  # doctest: +ELLIPSIS
    <Quantity ...e-05 1 / cm3>
    """

    # ------------------------------------------------------------------ #
    # Helper Methods                                                     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_parameters(
        p1: float,
        p2: float,
        gamma_c: float,
        gamma_min: float = 1.0,
        gamma_max: float = np.inf,
    ) -> None:
        r"""
        Validate broken-power-law parameters.

        All arguments may be scalars or arrays; array inputs are broadcast against one another and the
        checks must hold for every element.

        Parameters
        ----------
        p1 : float
            Power-law index below the break.
        p2 : float
            Power-law index above the break.
        gamma_c : float
            Break Lorentz factor.
        gamma_min : float, optional
            Minimum Lorentz factor. Default is ``1.0``.
        gamma_max : float, optional
            Maximum Lorentz factor. Default is ``numpy.inf``.

        Raises
        ------
        ValueError
            If any parameter is outside the supported physical range.
        """
        p1 = np.asarray(p1, dtype="f8")
        p2 = np.asarray(p2, dtype="f8")
        gamma_c = np.asarray(gamma_c, dtype="f8")
        gamma_min = np.asarray(gamma_min, dtype="f8")
        gamma_max = np.asarray(gamma_max, dtype="f8")

        if not np.all(np.isfinite(p1)):
            raise ValueError("Lower power-law index `p1` must be finite.")

        if not np.all(np.isfinite(p2)):
            raise ValueError("Upper power-law index `p2` must be finite.")

        if np.any(p1 <= 0.0):
            raise ValueError("Lower power-law index `p1` must be positive.")

        if np.any(p2 <= 0.0):
            raise ValueError("Upper power-law index `p2` must be positive.")

        if np.any(gamma_min < 1.0):
            raise ValueError("`gamma_min` must be greater than or equal to 1.")

        if np.any(gamma_max <= gamma_min):
            raise ValueError("`gamma_max` must be greater than `gamma_min`.")

        if np.any(gamma_c <= gamma_min):
            raise ValueError("`gamma_c` must be greater than `gamma_min`.")

        if np.any(gamma_c >= gamma_max):
            raise ValueError("`gamma_c` must be less than `gamma_max`.")

    @classmethod
    def _validate_and_process_parameters(cls, *, p1, p2, gamma_c, gamma_min=1.0, gamma_max=np.inf, **_) -> dict:
        r"""Validate the broken power-law parameters and return them as ``float64`` arrays."""
        cls._validate_parameters(p1, p2, gamma_c, gamma_min=gamma_min, gamma_max=gamma_max)
        p1, p2, gamma_c, gamma_min, gamma_max = _f8(p1, p2, gamma_c, gamma_min, gamma_max)
        return {"p1": p1, "p2": p2, "gamma_c": gamma_c, "gamma_min": gamma_min, "gamma_max": gamma_max}

    @classmethod
    def _branch_moment(
        cls,
        order: float,
        p: float,
        gamma_c: float,
        lower: float,
        upper: float,
    ) -> float:
        r"""
        Compute the moment contribution from one broken-power-law branch.

        Array-valued ``p``, ``gamma_c``, ``lower`` and ``upper`` are broadcast against one another. This is an
        unvalidated backend; see :meth:`moment` for the validated public interface.

        The branch shape is

        .. math::

            f_i(\gamma) = \left(\gamma / \gamma_c\right)^{-p_i}
            =
            \gamma_c^{p_i}\gamma^{-p_i}.

        Therefore the branch contribution to the raw moment is

        .. math::

            \int_a^b \gamma^k f_i(\gamma)\,d\gamma
            =
            \gamma_c^{p_i}
            \int_a^b \gamma^{k-p_i}\,d\gamma.
        """
        exponent = order + 1.0 - p
        return gamma_c**p * cls._integral_power_law(exponent, lower, upper)

    # ------------------------------------------------------------------ #
    # Distribution Interface                                             #
    # ------------------------------------------------------------------ #
    @classmethod
    def pdf(
        cls,
        gamma: "_ArrayLike",
        norm: float = 1.0,
        *,
        p1: float,
        p2: float,
        gamma_c: float,
        gamma_min: float = 1.0,
        gamma_max: float = np.inf,
        **_,
    ) -> np.ndarray:
        r"""
        Evaluate the broken power-law distribution.

        This method returns

        .. math::

            N(\gamma)
            =
            N_0
            \begin{cases}
            \left(\gamma / \gamma_c\right)^{-p_1},
            & \gamma_{\min} \leq \gamma < \gamma_c, \\\\
            \left(\gamma / \gamma_c\right)^{-p_2},
            & \gamma_c \leq \gamma \leq \gamma_{\max},
            \end{cases}

        and zero outside the support.

        Parameters
        ----------
        gamma : float or array-like
            Lorentz factor or Lorentz factors at which to evaluate the
            distribution.
        norm : float or array-like, optional
            Distribution amplitude :math:`N_0`. Default is ``1.0``.
        p1 : float or array-like
            Power-law index below the break.
        p2 : float or array-like
            Power-law index above the break.
        gamma_c : float or array-like
            Break Lorentz factor.
        gamma_min : float or array-like, optional
            Minimum Lorentz factor. Default is ``1.0``.
        gamma_max : float or array-like, optional
            Maximum Lorentz factor. Default is ``numpy.inf``.
        **_
            Ignored keyword arguments, accepted for API compatibility.

        Returns
        -------
        N_gamma : float or ~numpy.ndarray
            Distribution evaluated at ``gamma``. Array-valued arguments follow standard NumPy broadcasting, so
            for example ``gamma[np.newaxis, :]`` with ``gamma_c[:, np.newaxis]`` gives a ``(len(gamma_c), len(gamma))``
            result.
        """
        params = cls._validate_and_process_parameters(
            p1=p1, p2=p2, gamma_c=gamma_c, gamma_min=gamma_min, gamma_max=gamma_max
        )
        gamma, norm = _f8(gamma, norm)
        return cls._opt_pdf(gamma, norm, **params)

    @classmethod
    def support(
        cls,
        *,
        gamma_min: float = 1.0,
        gamma_max: float = np.inf,
        **_,
    ) -> tuple[float, float]:
        r"""
        Return the non-zero support of the broken power-law distribution.

        Parameters
        ----------
        gamma_min : float, optional
            Minimum Lorentz factor. Default is ``1.0``.
        gamma_max : float, optional
            Maximum Lorentz factor. Default is ``numpy.inf``.
        **_
            Ignored keyword arguments, accepted for API compatibility.

        Returns
        -------
        gamma_min, gamma_max : tuple of float
            Lower and upper Lorentz-factor bounds.
        """
        gamma_min, gamma_max = _f8(gamma_min, gamma_max)

        if np.any(gamma_min < 1.0):
            raise ValueError("`gamma_min` must be greater than or equal to 1.")

        if np.any(gamma_max <= gamma_min):
            raise ValueError("`gamma_max` must be greater than `gamma_min`.")

        return gamma_min[()], gamma_max[()]

    @classmethod
    def moment(
        cls,
        order: float,
        *,
        p1: float,
        p2: float,
        gamma_c: float,
        gamma_min: float = 1.0,
        gamma_max: float = np.inf,
        **_,
    ) -> float:
        r"""
        Compute an analytic raw moment of the broken power-law shape function.

        This method computes

        .. math::

            M_k
            =
            \int_{\gamma_{\min}}^{\gamma_{\max}}
            \gamma^k f(\gamma)\,d\gamma,

        by splitting the integral at :math:`\gamma_c`.

        Parameters
        ----------
        order : float
            Moment order :math:`k`.
        p1 : float or array-like
            Power-law index below the break.
        p2 : float or array-like
            Power-law index above the break.
        gamma_c : float or array-like
            Break Lorentz factor.
        gamma_min : float or array-like, optional
            Minimum Lorentz factor. Default is ``1.0``.
        gamma_max : float or array-like, optional
            Maximum Lorentz factor. Default is ``numpy.inf``.
        **_
            Ignored keyword arguments, accepted for API compatibility.

        Returns
        -------
        M_k : float or ~numpy.ndarray
            Raw moment of the shape function. Array-valued parameters are broadcast against one
            another and the result has the broadcast shape.
        """
        params = cls._validate_and_process_parameters(
            p1=p1, p2=p2, gamma_c=gamma_c, gamma_min=gamma_min, gamma_max=gamma_max
        )
        return cls._opt_moment(float(order), **params)

    # ------------------------------------------------------------------ #
    # Unvalidated backends                                               #
    # ------------------------------------------------------------------ #
    # No validation or coercion: parameters must satisfy ``1 <= gamma_min < gamma_c < gamma_max`` with positive
    # indices. All arguments may be floats or ``float64`` arrays and are broadcast with standard NumPy rules.
    @classmethod
    def _opt_pdf(cls, gamma, norm, p1, p2, gamma_c, gamma_min, gamma_max, **_):
        # To evaluate K breaks on a grid of G Lorentz factors, pass ``gamma[None, :]``, ``norm[:, None]`` and
        # ``gamma_c[:, None]`` to obtain a (K, G) result.
        below = (gamma >= gamma_min) & (gamma < gamma_c)
        above = (gamma >= gamma_c) & (gamma <= gamma_max)
        gamma_safe = np.maximum(gamma, gamma_min)

        return np.where(
            below,
            norm * (gamma_safe / gamma_c) ** (-p1),
            np.where(above, norm * (gamma_safe / gamma_c) ** (-p2), 0.0),
        )[()]

    @classmethod
    def _opt_moment(cls, order, p1, p2, gamma_c, gamma_min, gamma_max, **_):
        return cls._branch_moment(order, p1, gamma_c, gamma_min, gamma_c) + cls._branch_moment(
            order, p2, gamma_c, gamma_c, gamma_max
        )

    @classmethod
    def _opt_moment_between(cls, order, a, b, p1, p2, gamma_c, gamma_min, gamma_max, **_):
        # Empty intervals (upper <= lower) contribute exactly zero from both branches, so no early return is needed.
        lower = np.maximum(a, gamma_min)
        upper = np.minimum(b, gamma_max)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):  # masked-out empty intervals
            lower_branch = cls._branch_moment(order, p1, gamma_c, lower, np.minimum(upper, gamma_c))
            upper_branch = cls._branch_moment(order, p2, gamma_c, np.maximum(lower, gamma_c), upper)

        return lower_branch + upper_branch

    @classmethod
    def _opt_cdf(cls, gamma, norm, p1, p2, gamma_c, gamma_min, gamma_max, **_):
        return norm * cls._opt_moment_between(
            0.0, gamma_min, gamma, p1=p1, p2=p2, gamma_c=gamma_c, gamma_min=gamma_min, gamma_max=gamma_max
        )

    @classmethod
    def moment_between(
        cls,
        order: float,
        a: float,
        b: float,
        *,
        p1: float,
        p2: float,
        gamma_c: float,
        gamma_min: float = 1.0,
        gamma_max: float = np.inf,
        **_,
    ) -> float:
        r"""
        Compute an analytic truncated raw moment of the broken power-law shape.

        This method computes

        .. math::

            M_k(a,b)
            =
            \int_a^b
            \gamma^k f(\gamma)\,d\gamma,

        after clipping the requested interval to the distribution support and
        splitting the remaining interval at :math:`\gamma_c`.

        Parameters
        ----------
        order : float
            Moment order :math:`k`.
        a : float
            Requested lower Lorentz-factor bound.
        b : float
            Requested upper Lorentz-factor bound.
        p1 : float
            Power-law index below the break.
        p2 : float
            Power-law index above the break.
        gamma_c : float
            Break Lorentz factor.
        gamma_min : float, optional
            Minimum Lorentz factor. Default is ``1.0``.
        gamma_max : float, optional
            Maximum Lorentz factor. Default is ``numpy.inf``.
        **_
            Ignored keyword arguments, accepted for API compatibility.

        Returns
        -------
        M_k_between : float
            Truncated raw moment over the clipped interval. Returns ``0.0`` if
            the requested interval does not overlap the support.
        """
        params = cls._validate_and_process_parameters(
            p1=p1, p2=p2, gamma_c=gamma_c, gamma_min=gamma_min, gamma_max=gamma_max
        )
        a, b = _f8(a, b)
        return cls._opt_moment_between(float(order), a, b, **params)

    @classmethod
    def cdf(
        cls,
        gamma: "_ArrayLike",
        norm: float = 1.0,
        *,
        p1: float,
        p2: float,
        gamma_c: float,
        gamma_min: float = 1.0,
        gamma_max: float = np.inf,
        **_,
    ) -> np.ndarray:
        r"""
        Evaluate the cumulative broken power-law distribution analytically.

        This method computes

        .. math::

            F(\gamma)
            =
            N_0
            \int_{\gamma_{\min}}^\gamma
            f(\gamma')\,d\gamma',

        with values below :math:`\gamma_{\min}` clipped to zero and values
        above :math:`\gamma_{\max}` clipped to the total integral.

        Parameters
        ----------
        gamma : float or array-like
            Lorentz factor or Lorentz factors at which to evaluate the
            cumulative distribution.
        norm : float, optional
            Distribution amplitude :math:`N_0`. Default is ``1.0``.
        p1 : float
            Power-law index below the break.
        p2 : float
            Power-law index above the break.
        gamma_c : float
            Break Lorentz factor.
        gamma_min : float, optional
            Minimum Lorentz factor. Default is ``1.0``.
        gamma_max : float, optional
            Maximum Lorentz factor. Default is ``numpy.inf``.
        **_
            Ignored keyword arguments, accepted for API compatibility.

        Returns
        -------
        F_gamma : float or ~numpy.ndarray
            Cumulative distribution evaluated at ``gamma``.
        """
        params = cls._validate_and_process_parameters(
            p1=p1, p2=p2, gamma_c=gamma_c, gamma_min=gamma_min, gamma_max=gamma_max
        )
        gamma, norm = _f8(gamma, norm)
        return cls._opt_cdf(gamma, norm, **params)


class MaxwellJuettner(ElectronDistribution):
    r"""
    Relativistic thermal electron distribution (Maxwell-Jüttner).

    The Maxwell-Jüttner distribution is the relativistic generalisation of
    the Maxwell-Boltzmann distribution.  Its shape function (``norm=1``) is

    .. math::

        f(\gamma)
        =
        \frac{\gamma^2\,\beta}{\Theta\,K_2(1/\Theta)}
        \exp\!\left(-\frac{\gamma}{\Theta}\right),
        \qquad \gamma \geq 1,

    where

    * :math:`\beta = \sqrt{1 - \gamma^{-2}}` is the electron speed in units of
      :math:`c`,
    * :math:`\Theta = k_B T / (m_e c^2)` is the dimensionless temperature,
    * :math:`K_2` is the modified Bessel function of the second kind of
      order 2.

    The denominator :math:`\Theta\,K_2(1/\Theta)` is the canonical
    partition-function factor that ensures
    :math:`\int_1^\infty f(\gamma)\,d\gamma = 1`.  The full distribution is

    .. math::

        N(\gamma) = N_0\,f(\gamma),

    where :math:`N_0` is supplied via the ``norm`` argument.  Because
    :math:`f` is already probability-normalised, :math:`N_0` equals the
    total electron number density :math:`n_e` directly.

    .. rubric:: Distribution parameters

    .. list-table::
        :header-rows: 1
        :widths: 20 20 60

        * - Parameter
          - Type
          - Description
        * - ``Theta``
          - float
          - Dimensionless electron temperature
            :math:`\Theta = k_B T / (m_e c^2)`.  Must be positive.
            Values :math:`\Theta \ll 1` are mildly relativistic;
            :math:`\Theta \gtrsim 1` are ultra-relativistic.

    Notes
    -----
    **Raw moments.** Because the distribution is probability-normalised, the
    zeroth moment is exactly :math:`M_0 = 1`. Exact closed forms for the
    first and second raw moments follow from Bessel-function recurrences:

    .. math::

        M_1 &= \frac{K_1(1/\Theta)}{K_2(1/\Theta)} + 3\Theta, \\
        M_2 &= 1 + 12\Theta^2
               + \frac{3\Theta\,K_1(1/\Theta)}{K_2(1/\Theta)}.

    Higher moments are not implemented analytically and fall back to
    numerical quadrature.

    **Approximate mean.** :footcite:p:`1998ApJ...498..313G` provide the
    compact approximation

    .. math::

        \langle\gamma\rangle
        \approx
        \frac{15\Theta^2 + 11\Theta + 4}{4 + 5\Theta},

    accurate to :math:`\lesssim 1\%` for all :math:`\Theta > 0`; see
    :meth:`approx_first_moment`.

    **Normalization.** The amplitude :math:`N_0` is equal to the total
    number density :math:`n_e` because :math:`M_0 = 1`.

    See Also
    --------
    PowerLaw : Non-thermal power-law distribution.
    BrokenPowerLaw : Non-thermal broken power-law distribution.
    MaxwellJuettnerPowerLaw : Mixed thermal plus power-law distribution.
    MaxwellJuettnerBrokenPowerLaw : Mixed thermal plus broken-power-law distribution.

    Examples
    --------
    Evaluate the distribution at :math:`\gamma = 2` for
    :math:`\Theta = 1`:

    >>> float(MaxwellJuettner.pdf(2.0, Theta=1.0))  # doctest: +ELLIPSIS
    0.28...

    The zeroth moment is always 1 (the shape function is
    probability-normalised):

    >>> MaxwellJuettner.moment(0, Theta=1.0)
    1.0

    Exact first moment via Bessel functions:

    >>> float(MaxwellJuettner.moment(1, Theta=1.0))  # doctest: +ELLIPSIS
    3.3...

    Approximate mean Lorentz factor at :math:`\Theta = 1`:

    >>> MaxwellJuettner.approx_first_moment(Theta=1.0)
    3.3333333333333335

    Normalize to a total electron number density of
    :math:`10^{-3}\,\mathrm{cm^{-3}}` (returns :math:`N_0 = n_e` directly):

    >>> import astropy.units as u
    >>> MaxwellJuettner.normalize_from_n_total(1e-3 * u.cm**-3, Theta=1.0)
    <Quantity 0.001 1 / cm3>
    """

    # ------------------------------------------------------------------ #
    # Helper Methods                                                     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_parameters(Theta: float) -> None:
        r"""
        Validate Maxwell-Jüttner parameters.

        Parameters
        ----------
        Theta : float
            Dimensionless electron temperature.

        Raises
        ------
        ValueError
            If ``Theta`` is non-finite or non-positive.
        """
        Theta = _f8(Theta)[0]

        if not np.all(np.isfinite(Theta)):
            raise ValueError("Dimensionless temperature `Theta` must be finite.")

        if np.any(Theta <= 0.0):
            raise ValueError("Dimensionless temperature `Theta` must be positive.")

    @classmethod
    def _validate_and_process_parameters(cls, *, Theta=1.0, **_) -> dict:
        r"""Validate the Maxwell-Juttner parameters and return them as ``float64`` arrays."""
        cls._validate_parameters(Theta)
        return {"Theta": _f8(Theta)[0]}

    # ------------------------------------------------------------------ #
    # Unvalidated backends                                               #
    # ------------------------------------------------------------------ #
    # No validation or coercion; floats or float64 arrays, broadcast with standard NumPy rules.
    @classmethod
    def _opt_pdf(cls, gamma, norm, Theta, **_):
        inside_support = gamma >= 1.0

        # Clamp to 1.0 so sqrt(1 - gamma^{-2}) is defined for out-of-support elements;
        # np.where selects 0.0 for those positions anyway.
        gamma_safe = np.maximum(gamma, 1.0)
        beta = np.sqrt(1.0 - gamma_safe**-2)

        # Use exponentially scaled K_2 for numerical stability:
        #
        #   exp(-gamma/Theta) / K_2(1/Theta) = exp((1 - gamma)/Theta) / kve(2, 1/Theta).
        normalization = Theta * kve(2, 1.0 / Theta)

        return np.where(
            inside_support,
            norm * gamma_safe**2 * beta * np.exp((1.0 - gamma_safe) / Theta) / normalization,
            0.0,
        )[()]

    @classmethod
    def _opt_moment(cls, order, Theta, **_):
        # Closed forms for orders 0-2; higher orders use quadrature, looped over any array of temperatures.
        if order == 0:
            return np.ones_like(Theta)[()]

        z = 1.0 / Theta
        k1_over_k2 = kve(1, z) / kve(2, z)

        if order == 1:
            return k1_over_k2 + 3.0 * Theta

        if order == 2:
            return 1.0 + 12.0 * Theta**2 + 3.0 * Theta * k1_over_k2

        def _quad(theta):
            return quad(lambda g: g**order * cls._opt_pdf(g, 1.0, theta), 1.0, np.inf)[0]

        return np.vectorize(_quad, otypes=["f8"])(Theta)[()]

    @classmethod
    def _opt_approx_first_moment(cls, Theta, **_):
        return (15.0 * Theta**2 + 11.0 * Theta + 4.0) / (5.0 * Theta + 4.0)

    @classmethod
    def _opt_approx_first_kinetic_moment(cls, Theta, **_):
        return Theta * (6.0 + 15.0 * Theta) / (4.0 + 5.0 * Theta)

    # ---------------------------------------------- #
    # Statistical Methods                            #
    # ---------------------------------------------- #
    @classmethod
    def pdf(
        cls,
        gamma: "_ArrayLike",
        norm: float = 1.0,
        *,
        Theta: float = 1.0,
        **_,
    ) -> np.ndarray:
        r"""
        Evaluate the Maxwell-Jüttner distribution.

        This method returns

        .. math::

            N(\gamma)
            =
            N_0
            \frac{
                \gamma^2 \beta
            }{
                \Theta K_2(1/\Theta)
            }
            \exp(-\gamma/\Theta),

        where

        .. math::

            \beta = \sqrt{1 - \gamma^{-2}},

        and the support is fixed to :math:`1 \leq \gamma < \infty`.

        Parameters
        ----------
        gamma : float or array-like
            Electron Lorentz factor or Lorentz factors at which to evaluate
            the distribution.
        norm : float, optional
            Distribution amplitude :math:`N_0`. Default is ``1.0``.
        Theta : float, optional
            Dimensionless electron temperature,
            :math:`\Theta = k_B T/(m_e c^2)`. Default is ``1.0``.
        **_
            Ignored keyword arguments, accepted for API compatibility.

        Returns
        -------
        N_gamma : float or ~numpy.ndarray
            Maxwell-Jüttner distribution evaluated at ``gamma``.
        """
        params = cls._validate_and_process_parameters(Theta=Theta)
        gamma, norm = _f8(gamma, norm)
        return cls._opt_pdf(gamma, norm, **params)

    @classmethod
    def support(cls, **_) -> tuple:
        r"""
        Return the non-zero support of the Maxwell-Jüttner distribution.

        The Maxwell-Jüttner distribution is non-zero for all Lorentz factors
        :math:`\gamma \geq 1`, so the support is :math:`[1, \infty)`.

        Parameters
        ----------
        **_
            Ignored keyword arguments, accepted for API compatibility.

        Returns
        -------
        gamma_min, gamma_max : tuple of float
            Lower and upper support bounds ``(1, numpy.inf)``.

        Examples
        --------
        >>> MaxwellJuettner.support(Theta=1.0)
        (1, inf)
        """
        return 1, np.inf

    @classmethod
    def moment(cls, order: float, *, Theta: float = 1.0, **_) -> float:
        r"""
        Evaluate raw moments of the Maxwell-Jüttner distribution.

        The distribution is

        .. math::

            f(\gamma)
            =
            \frac{\gamma^2 \beta}{\Theta K_2(1/\Theta)}
            \exp(-\gamma/\Theta),

        with :math:`\beta = \sqrt{1-\gamma^{-2}}`.

        The general raw moment may be written as

        .. math::

            \langle \gamma^n \rangle
            =
            \frac{(-1)^n}{Z(z)}
            \frac{d^n Z}{dz^n},

        where

        .. math::

            Z(z) = \frac{K_2(z)}{z},
            \qquad
            z = \frac{1}{\Theta}.

        Exact closed forms are provided for ``order = 0``, ``order = 1``, and
        ``order = 2``.  All other non-negative orders fall back to numerical
        quadrature over :meth:`pdf`.

        Parameters
        ----------
        order : float
            Moment order. Closed-form results are returned for ``0``, ``1``,
            and ``2``; all other non-negative values use numerical quadrature.
        Theta : float, optional
            Dimensionless electron temperature. Default is ``1.0``.
        **_
            Ignored keyword arguments, accepted for API compatibility.

        Returns
        -------
        moment_order : float
            Raw moment :math:`\langle \gamma^n \rangle`.
        """
        if order < 0:
            raise ValueError("Moment order `order` must be non-negative.")

        params = cls._validate_and_process_parameters(Theta=Theta)
        return cls._opt_moment(order, **params)

    @classmethod
    def approx_first_moment(cls, Theta: float = 1.0, **_) -> float:
        r"""
        Compute the approximate first-moment of the MJ distribution.

        Formally, the first moment of the MJ distribution is given by

        .. math::

            M^{(1)} =  \int_{1}^{\infty} \frac{\gamma^3 \beta}{\Theta K_2(1/\Theta)}
            \exp\!\left(-\frac{\gamma}{\Theta}\right), d\gamma.

        This integral is\ :footcite:p:`1998ApJ...498..313G, 2000ApJ...541..234O, chandrasekhar1957introduction`

        .. math::

            M^{(1)} = \left[\frac{3K_3(1/\Theta) + K_1(1/\Theta)}{4K_2(1/\Theta)}\right],

        which, in turn, is well approximated by\ :footcite:p:`1998ApJ...498..313G`

        .. math::

            M^{(1)} \approx \frac{15 \Theta^2 + 11 \Theta + 4}{4 + 5\Theta}.

        .. note::

            Relative to the literature, we here maintain the standard convention and consider
            the total (not just kinetic) energy of the electrons, i.e., we include the rest mass
            energy in the definition of the Lorentz factor.

        Parameters
        ----------
        Theta : float, optional
            Dimensionless electron temperature,
            :math:`\Theta = k_B T/(m_e c^2)`. Default is ``1.0``.
        **_
            Ignored keyword arguments, accepted for API compatibility.

        Returns
        -------
        first_moment : float
            Approximate first raw moment :math:`\langle\gamma\rangle`.
        """
        return cls._opt_approx_first_moment(**cls._validate_and_process_parameters(Theta=Theta))

    @classmethod
    def approx_first_kinetic_moment(cls, Theta: float = 1.0, **_) -> float:
        r"""
        Compute the approximate first kinetic moment of the MJ distribution.

        Formally, the first kinetic moment of the MJ distribution is given by

        .. math::

            M_{\rm kin}^{(1)}
            =
            \int_{1}^{\infty}
            (\gamma - 1)
            \frac{
                \gamma^2 \beta
            }{
                \Theta K_2(1/\Theta)
            }
            \exp\!\left(-\frac{\gamma}{\Theta}\right)
            \,d\gamma.

        Equivalently,

        .. math::

            M_{\rm kin}^{(1)}
            =
            M^{(1)} - 1,

        where :math:`M^{(1)} = \langle\gamma\rangle` is the total first
        Lorentz-factor moment.

        This kinetic moment is\ :footcite:p:`1998ApJ...498..313G, 2000ApJ...541..234O, chandrasekhar1957introduction`

        .. math::

            M_{\rm kin}^{(1)}
            =
            \left[
            \frac{
                3K_3(1/\Theta) + K_1(1/\Theta)
            }{
                4K_2(1/\Theta)
            }
            - 1
            \right],

        which, in turn, is well approximated by\ :footcite:p:`1998ApJ...498..313G`

        .. math::

            M_{\rm kin}^{(1)}
            \approx
            \Theta
            \left(
            \frac{
                6 + 15\Theta
            }{
                4 + 5\Theta
            }
            \right).

        .. note::

            This is the kinetic/internal electron-energy moment. It excludes
            the rest-mass contribution included in :math:`M^{(1)}`.

        Parameters
        ----------
        Theta : float, optional
            Dimensionless electron temperature,
            :math:`\Theta = k_B T/(m_e c^2)`. Default is ``1.0``.
        **_
            Ignored keyword arguments, accepted for API compatibility.

        Returns
        -------
        first_kinetic_moment : float
            Approximate first kinetic moment
            :math:`\langle\gamma - 1\rangle`.
        """
        return cls._opt_approx_first_kinetic_moment(**cls._validate_and_process_parameters(Theta=Theta))

    @classmethod
    def cdf(
        cls,
        gamma: "_ArrayLike",
        norm: float = 1.0,
        *,
        Theta: float = 1.0,
        **_,
    ) -> np.ndarray:
        r"""
        Evaluate the Maxwell-Jüttner cumulative distribution function.

        .. math::

            F(\gamma)
            =
            N_0 \int_1^\gamma f(\gamma')\,d\gamma'

        This override uses a vectorized trapezoidal integration on a log-spaced
        grid followed by linear interpolation, avoiding the per-element
        :func:`~scipy.integrate.quad` loop of the base-class implementation.

        Parameters
        ----------
        gamma : float or array-like
            Lorentz factor or Lorentz factors at which to evaluate the CDF.
        norm : float, optional
            Distribution amplitude :math:`N_0`. Default is ``1.0``.
        Theta : float, optional
            Dimensionless electron temperature. Default is ``1.0``.
        **_
            Ignored keyword arguments, accepted for API compatibility.

        Returns
        -------
        F_gamma : float or ~numpy.ndarray
            Cumulative distribution evaluated at ``gamma``.
        """
        Theta = cls._validate_and_process_parameters(Theta=Theta)["Theta"]
        if Theta.ndim != 0:
            raise ValueError("`MaxwellJuettner.cdf` shares one integration grid and requires a scalar `Theta`.")

        gamma_array = np.asarray(gamma, dtype="f8")
        scalar_input = gamma_array.ndim == 0
        gamma_flat = np.atleast_1d(gamma_array).ravel()

        Theta = float(Theta)
        norm = float(norm)

        # Practical upper limit: MJ PDF decays exponentially above ~20 Theta
        g_upper = max(float(gamma_flat.max()), 1.0) * 1.01
        g_upper = max(g_upper, 1.0 + 20.0 * Theta)

        # Log-spaced grid is dense near gamma=1 (where the PDF rises steeply)
        # and sparser at high gamma (where the PDF is negligible)
        g_grid = np.exp(np.linspace(0.0, np.log(g_upper), 10_000))
        pdf_grid = cls._opt_pdf(g_grid, 1.0, Theta)

        # Vectorized cumulative trapezoidal rule; no additional imports needed
        dg = np.diff(g_grid)
        cdf_increments = 0.5 * (pdf_grid[:-1] + pdf_grid[1:]) * dg
        cdf_grid = np.concatenate([[0.0], np.cumsum(cdf_increments)])

        # Interpolate to requested values; clamp below gamma_min=1 to 0
        result = np.interp(gamma_flat, g_grid, cdf_grid, left=0.0, right=cdf_grid[-1])
        result = norm * np.where(gamma_flat <= 1.0, 0.0, result)

        if scalar_input:
            return float(result[0])
        return result.reshape(gamma_array.shape)

    # ------------------------------------------------------------------ #
    # Normalization (Private)                                            #
    # ------------------------------------------------------------------ #
    # These methods all concern the normalization of the distribution in various scenarios
    # and from various physical parameters. We provide the typical public / private split here
    # because many of these methods have unit-dependencies.
    #
    # All methods here assume CGS units.

    @classmethod
    def _normalize_from_n_total(cls, n_total: float, **params) -> float:
        return n_total

    @classmethod
    def _normalize_from_magnetic_field(cls, B: float, epsilon_B: float, epsilon_E: float, Theta=1.0, **_) -> float:
        u_B = B**2 / (8.0 * np.pi)
        return (epsilon_E / epsilon_B) * u_B / (electron_rest_energy_cgs * cls._opt_approx_first_moment(Theta))

    @classmethod
    def _normalize_from_energy_density(cls, u_therm: float, epsilon_E: float, Theta=1.0, **_) -> float:
        return epsilon_E * u_therm / (electron_rest_energy_cgs * cls._opt_approx_first_moment(Theta))


# =========================================================================== #
# Mixed Distributions                                                         #
# =========================================================================== #
class MixedThermalNonThermal(ElectronDistribution):
    r"""
    Base class for mixed thermal and non-thermal electron distributions.

    This class represents a mixture of a thermal Maxwell-Jüttner component and
    a non-thermal component,

    .. math::

        f_{\rm mix}(\gamma)
        =
        \delta f_{\rm th}(\gamma)
        +
        (1-\delta) f_{\rm nth}(\gamma),

    where :math:`0 \leq \delta \leq 1`.

    The thermal component is taken to be the Maxwell-Jüttner distribution, and
    the non-thermal component is supplied by subclasses through
    ``NON_THERMAL_DISTRIBUTION``.

    Notes
    -----
    The non-thermal classes in this module generally define shape functions
    rather than probability-normalized distributions. Therefore, this class
    internally normalizes the non-thermal shape by its zeroth moment,

    .. math::

        f_{\rm nth}(\gamma)
        =
        \frac{
            f_{\rm shape,nth}(\gamma)
        }{
            M_{\rm nth}^{(0)}
        }.

    With this convention,

    .. math::

        \int f_{\rm mix}(\gamma)\,d\gamma = 1,

    and ``delta`` is a true number-fraction mixture weight.

    This is distinct from an energy-fraction parameter. If ``delta`` is instead
    intended to represent the fraction of electron energy in the thermal
    population, the component normalizations should be computed separately from
    their energy budgets rather than treated as a direct PDF mixture weight.

    See Also
    --------
    MaxwellJuettnerPowerLaw : Concrete mixture with a power-law non-thermal component.
    MaxwellJuettnerBrokenPowerLaw : Concrete mixture with a broken-power-law non-thermal component.
    MaxwellJuettner : Thermal Maxwell-Jüttner component used in all subclasses.
    """

    THERMAL_DISTRIBUTION: ClassVar[type[ElectronDistribution]] = MaxwellJuettner
    NON_THERMAL_DISTRIBUTION: ClassVar[Optional[type[ElectronDistribution]]] = None

    # ------------------------------------------------------------------ #
    # Helper Methods                                                     #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_mixture_parameter(delta: float) -> None:
        r"""
        Validate the thermal mixture fraction.

        Parameters
        ----------
        delta : float
            Fractional weight of the thermal Maxwell-Jüttner component.

        Raises
        ------
        ValueError
            If ``delta`` is non-finite or outside the interval ``[0, 1]``.
        """
        delta = _f8(delta)[0]

        if not np.all(np.isfinite(delta)):
            raise ValueError("Mixture parameter `delta` must be finite.")

        if np.any(delta < 0.0) or np.any(delta > 1.0):
            raise ValueError("Mixture parameter `delta` must satisfy 0 <= delta <= 1.")

    @classmethod
    def _validate_non_thermal_distribution(cls) -> None:
        r"""Validate that the subclass defines a non-thermal distribution."""
        if cls.NON_THERMAL_DISTRIBUTION is None:
            raise NotImplementedError(
                "`MixedThermalNonThermal` must be subclassed with a concrete `NON_THERMAL_DISTRIBUTION`."
            )

    @classmethod
    def _non_thermal_m0(cls, **params) -> float:
        r"""
        Return the zeroth moment of the non-thermal shape function.

        This is used to normalize the non-thermal component before forming the
        mixture.
        """
        return cls.NON_THERMAL_DISTRIBUTION._opt_moment(0.0, **params)

    @classmethod
    def _validate_and_process_parameters(cls, *, delta, **params) -> dict:
        r"""
        Validate the mixture and component parameters and return them, merged, as ``float64`` arrays.

        Also checks that the non-thermal component is normalizable.
        """
        cls._validate_mixture_parameter(delta)
        cls._validate_non_thermal_distribution()

        thermal = cls.THERMAL_DISTRIBUTION._validate_and_process_parameters(**params)
        non_thermal = cls.NON_THERMAL_DISTRIBUTION._validate_and_process_parameters(**params)

        m0 = cls._non_thermal_m0(**non_thermal)
        if not np.all(np.isfinite(m0)):
            raise ValueError(
                "The zeroth moment of the non-thermal component is not finite. "
                "Use parameters that give a normalizable non-thermal shape, "
                "for example a finite `gamma_max` or a sufficiently steep index."
            )
        if np.any(m0 <= 0.0):
            raise ValueError("The zeroth moment of the non-thermal component must be positive.")

        return {**thermal, **non_thermal, "delta": _f8(delta)[0]}

    # ------------------------------------------------------------------ #
    # Unvalidated backends                                               #
    # ------------------------------------------------------------------ #
    # No validation or coercion; ``params`` is the merged dictionary from `_validate_and_process_parameters`.
    @classmethod
    def _opt_pdf(cls, gamma, norm, delta, **params):
        thermal_pdf = cls.THERMAL_DISTRIBUTION._opt_pdf(gamma, 1.0, **params)
        non_thermal_pdf = cls.NON_THERMAL_DISTRIBUTION._opt_pdf(gamma, 1.0, **params) / cls._non_thermal_m0(**params)
        return norm * (delta * thermal_pdf + (1.0 - delta) * non_thermal_pdf)

    @classmethod
    def _opt_moment(cls, order, delta, **params):
        if order == 0:
            return 1.0

        thermal_moment = cls.THERMAL_DISTRIBUTION._opt_moment(order, **params)
        non_thermal_moment = cls.NON_THERMAL_DISTRIBUTION._opt_moment(order, **params) / cls._non_thermal_m0(**params)
        return delta * thermal_moment + (1.0 - delta) * non_thermal_moment

    @classmethod
    def _opt_moment_between(cls, order, a, b, delta, **params):
        # The Maxwell-Juttner truncated moment is a scalar numerical integral (base-class quadrature).
        thermal_moment = cls.THERMAL_DISTRIBUTION.moment_between(order, a, b, **params)
        non_thermal_moment = cls.NON_THERMAL_DISTRIBUTION._opt_moment_between(order, a, b, **params)
        return delta * thermal_moment + (1.0 - delta) * non_thermal_moment / cls._non_thermal_m0(**params)

    # ------------------------------------------------------------------ #
    # Distribution Interface                                             #
    # ------------------------------------------------------------------ #
    @classmethod
    def pdf(
        cls,
        gamma: "_ArrayLike",
        norm: float = 1.0,
        *,
        delta: float,
        **params,
    ) -> np.ndarray:
        r"""
        Evaluate the mixed thermal/non-thermal electron distribution.

        This method returns

        .. math::

            N(\gamma)
            =
            N_0
            \left[
            \delta f_{\rm MJ}(\gamma)
            +
            (1-\delta) f_{\rm nth}(\gamma)
            \right],

        where :math:`f_{\rm MJ}` is the normalized Maxwell-Jüttner
        distribution and :math:`f_{\rm nth}` is the normalized non-thermal
        component.

        Parameters
        ----------
        gamma : float or array-like
            Lorentz factor or Lorentz factors at which to evaluate the
            distribution.
        norm : float, optional
            Distribution amplitude :math:`N_0`. Default is ``1.0``.
        delta : float
            Mixture weight of the thermal Maxwell-Jüttner component. Must
            satisfy ``0 <= delta <= 1``.
        **params
            Parameters forwarded to the thermal and non-thermal component
            distributions. For example, ``Theta`` is used by the
            Maxwell-Jüttner component, while ``p``, ``gamma_min``, and
            ``gamma_max`` may be used by the power-law component.

        Returns
        -------
        N_gamma : float or ~numpy.ndarray
            Mixed electron distribution evaluated at ``gamma``.
        """
        params = cls._validate_and_process_parameters(delta=delta, **params)
        gamma, norm = _f8(gamma, norm)
        return cls._opt_pdf(gamma, norm, **params)

    @classmethod
    def support(cls, **params) -> tuple[float, float]:
        r"""
        Return the non-zero support of the mixed distribution.

        The support is the union of the thermal and non-thermal supports.

        Parameters
        ----------
        **params
            Parameters forwarded to the component distributions.

        Returns
        -------
        gamma_min, gamma_max : tuple of float
            Lower and upper support bounds of the mixed distribution.
        """
        cls._validate_non_thermal_distribution()

        thermal_min, thermal_max = cls.THERMAL_DISTRIBUTION.support(**params)
        non_thermal_min, non_thermal_max = cls.NON_THERMAL_DISTRIBUTION.support(
            **params,
        )

        return min(thermal_min, non_thermal_min), max(
            thermal_max,
            non_thermal_max,
        )

    @classmethod
    def moment(
        cls,
        order: float,
        *,
        delta: float,
        **params,
    ) -> float:
        r"""
        Compute a raw moment of the mixed distribution.

        This method computes

        .. math::

            M_k^{\rm mix}
            =
            \delta M_k^{\rm MJ}
            +
            (1-\delta) M_k^{\rm nth,norm},

        where

        .. math::

            M_k^{\rm nth,norm}
            =
            \frac{
                M_k^{\rm nth,shape}
            }{
                M_0^{\rm nth,shape}
            }.

        Parameters
        ----------
        order : float
            Moment order :math:`k`.
        delta : float
            Mixture weight of the thermal Maxwell-Jüttner component.
        **params
            Parameters forwarded to the component distributions.

        Returns
        -------
        M_k : float
            Raw moment of the normalized mixture shape.
        """
        params = cls._validate_and_process_parameters(delta=delta, **params)
        return cls._opt_moment(float(order), **params)

    @classmethod
    def moment_between(
        cls,
        order: float,
        a: float,
        b: float,
        *,
        delta: float,
        **params,
    ) -> float:
        r"""
        Compute a truncated raw moment of the mixed distribution.

        This method computes

        .. math::

            M_k^{\rm mix}(a,b)
            =
            \delta M_k^{\rm MJ}(a,b)
            +
            (1-\delta) M_k^{\rm nth,norm}(a,b),

        where the non-thermal component is normalized by its full zeroth
        moment.

        Parameters
        ----------
        order : int
            Moment order :math:`k`.
        a : float
            Requested lower Lorentz-factor bound.
        b : float
            Requested upper Lorentz-factor bound.
        delta : float
            Mixture weight of the thermal Maxwell-Jüttner component.
        **params
            Parameters forwarded to the component distributions.

        Returns
        -------
        M_k_between : float
            Truncated raw moment over the requested interval.
        """
        params = cls._validate_and_process_parameters(delta=delta, **params)
        a, b = _f8(a, b)
        return cls._opt_moment_between(float(order), a, b, **params)


class MaxwellJuettnerPowerLaw(MixedThermalNonThermal):
    r"""
    Mixed Maxwell-Jüttner plus power-law electron distribution.

    The normalized mixture shape is

    .. math::

        f_{\rm mix}(\gamma)
        =
        \delta f_{\rm MJ}(\gamma;\Theta)
        +
        (1-\delta)
        \frac{
            f_{\rm PL}(\gamma;p,\gamma_{\min},\gamma_{\max})
        }{
            M_{\rm PL}^{(0)}
        },

    where :math:`f_{\rm MJ}` is the Maxwell-Jüttner shape (already
    probability-normalised, :math:`\int f_{\rm MJ}\,d\gamma = 1`) and
    :math:`M_{\rm PL}^{(0)} = \int f_{\rm PL}\,d\gamma` is the zeroth moment
    of the power-law shape that re-normalises the non-thermal component to
    unit integral.  With this convention :math:`\delta` is a true
    number-fraction weight:

    .. math::

        \int f_{\rm mix}(\gamma)\,d\gamma = 1.

    The full distribution is

    .. math::

        N(\gamma) = N_0\,f_{\rm mix}(\gamma).

    .. rubric:: Distribution parameters

    .. list-table::
        :header-rows: 1
        :widths: 20 20 60

        * - Parameter
          - Type
          - Description
        * - ``delta``
          - float
          - Fractional weight of the thermal Maxwell-Jüttner component.
            Must satisfy :math:`0 \leq \delta \leq 1`.  Setting
            ``delta=1`` gives a pure Maxwell-Jüttner distribution;
            ``delta=0`` gives a pure (re-normalised) power-law.
        * - ``Theta``
          - float, optional
          - Dimensionless temperature of the thermal component,
            :math:`\Theta = k_B T / (m_e c^2)`.  Default is ``1.0``.
        * - ``p``
          - float
          - Power-law index of the non-thermal component. Must be positive.
        * - ``gamma_min``
          - float, optional
          - Minimum Lorentz factor of the power-law component.  Default
            is ``1.0``.
        * - ``gamma_max``
          - float, optional
          - Maximum Lorentz factor of the power-law component.  Default
            is ``numpy.inf``.

    See Also
    --------
    MixedThermalNonThermal : Base class defining the mixture interface.
    MaxwellJuettner : Thermal component.
    PowerLaw : Non-thermal component.
    MaxwellJuettnerBrokenPowerLaw : Mixture with a broken-power-law non-thermal component.

    Examples
    --------
    Evaluate the mixed distribution at :math:`\gamma = 2` with equal
    thermal and non-thermal weights (:math:`\delta = 0.5`):

    >>> float(
    ...     MaxwellJuettnerPowerLaw.pdf(
    ...         2.0, delta=0.5, Theta=1.0, p=3.0
    ...     )
    ... )  # doctest: +ELLIPSIS
    0.2...

    A pure thermal distribution (``delta=1``) agrees exactly with
    :class:`MaxwellJuettner`:

    >>> import numpy as np
    >>> gamma = np.array([1.5, 2.0, 5.0])
    >>> np.allclose(
    ...     MaxwellJuettnerPowerLaw.pdf(
    ...         gamma, delta=1.0, Theta=1.0, p=3.0
    ...     ),
    ...     MaxwellJuettner.pdf(gamma, Theta=1.0),
    ... )
    True

    Because :math:`\int f_{\rm mix}\,d\gamma = 1`, the amplitude
    :math:`N_0` equals the total electron number density directly:

    >>> import astropy.units as u
    >>> MaxwellJuettnerPowerLaw.normalize_from_n_total(
    ...     1e-3 * u.cm**-3, delta=0.5, Theta=1.0, p=3.0
    ... )
    <Quantity 0.001 1 / cm3>
    """

    NON_THERMAL_DISTRIBUTION: ClassVar[type[ElectronDistribution]] = PowerLaw


class MaxwellJuettnerBrokenPowerLaw(MixedThermalNonThermal):
    r"""
    Mixed Maxwell-Jüttner plus broken-power-law electron distribution.

    The normalized mixture shape is

    .. math::

        f_{\rm mix}(\gamma)
        =
        \delta f_{\rm MJ}(\gamma;\Theta)
        +
        (1-\delta)
        \frac{
            f_{\rm BPL}(\gamma;p_1,p_2,\gamma_c,\gamma_{\min},\gamma_{\max})
        }{
            M_{\rm BPL}^{(0)}
        },

    where :math:`f_{\rm MJ}` is the Maxwell-Jüttner shape (already
    probability-normalised, :math:`\int f_{\rm MJ}\,d\gamma = 1`) and
    :math:`M_{\rm BPL}^{(0)} = \int f_{\rm BPL}\,d\gamma` is the zeroth
    moment of the broken-power-law shape that re-normalises the non-thermal
    component to unit integral.  With this convention :math:`\delta` is a
    true number-fraction weight:

    .. math::

        \int f_{\rm mix}(\gamma)\,d\gamma = 1.

    The full distribution is

    .. math::

        N(\gamma) = N_0\,f_{\rm mix}(\gamma).

    .. rubric:: Distribution parameters

    .. list-table::
        :header-rows: 1
        :widths: 20 20 60

        * - Parameter
          - Type
          - Description
        * - ``delta``
          - float
          - Fractional weight of the thermal Maxwell-Jüttner component.
            Must satisfy :math:`0 \leq \delta \leq 1`.  Setting
            ``delta=1`` gives a pure Maxwell-Jüttner distribution;
            ``delta=0`` gives a pure (re-normalised) broken-power-law.
        * - ``Theta``
          - float, optional
          - Dimensionless temperature of the thermal component,
            :math:`\Theta = k_B T / (m_e c^2)`.  Default is ``1.0``.
        * - ``p1``
          - float
          - Broken-power-law index below the spectral break :math:`\gamma_c`.
            Must be positive.
        * - ``p2``
          - float
          - Broken-power-law index above the spectral break :math:`\gamma_c`.
            Must be positive.
        * - ``gamma_c``
          - float
          - Break Lorentz factor.  Must satisfy
            :math:`\gamma_{\min} < \gamma_c < \gamma_{\max}`.
        * - ``gamma_min``
          - float, optional
          - Minimum Lorentz factor of the broken-power-law component.
            Default is ``1.0``.
        * - ``gamma_max``
          - float, optional
          - Maximum Lorentz factor of the broken-power-law component.
            Default is ``numpy.inf``.

    See Also
    --------
    MixedThermalNonThermal : Base class defining the mixture interface.
    MaxwellJuettner : Thermal component.
    BrokenPowerLaw : Non-thermal component.
    MaxwellJuettnerPowerLaw : Mixture with a single power-law non-thermal component.

    Examples
    --------
    Evaluate the mixed distribution at :math:`\gamma = 5` with equal
    thermal and non-thermal weights (:math:`\delta = 0.5`):

    >>> float(
    ...     MaxwellJuettnerBrokenPowerLaw.pdf(
    ...         5.0,
    ...         delta=0.5,
    ...         Theta=1.0,
    ...         p1=2.0,
    ...         p2=4.0,
    ...         gamma_c=10.0,
    ...     )
    ... )  # doctest: +ELLIPSIS
    0.07...

    A pure thermal distribution (``delta=1``) agrees exactly with
    :class:`MaxwellJuettner`:

    >>> import numpy as np
    >>> gamma = np.array([1.5, 2.0, 5.0])
    >>> np.allclose(
    ...     MaxwellJuettnerBrokenPowerLaw.pdf(
    ...         gamma,
    ...         delta=1.0,
    ...         Theta=1.0,
    ...         p1=2.0,
    ...         p2=4.0,
    ...         gamma_c=10.0,
    ...     ),
    ...     MaxwellJuettner.pdf(gamma, Theta=1.0),
    ... )
    True

    Because :math:`\int f_{\rm mix}\,d\gamma = 1`, the amplitude
    :math:`N_0` equals the total electron number density directly:

    >>> import astropy.units as u
    >>> MaxwellJuettnerBrokenPowerLaw.normalize_from_n_total(
    ...     1e-3 * u.cm**-3,
    ...     delta=0.5,
    ...     Theta=1.0,
    ...     p1=2.0,
    ...     p2=4.0,
    ...     gamma_c=10.0,
    ... )
    <Quantity 0.001 1 / cm3>
    """

    NON_THERMAL_DISTRIBUTION: ClassVar[type[ElectronDistribution]] = BrokenPowerLaw
