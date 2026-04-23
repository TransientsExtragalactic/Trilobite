"""Abstract base class for Triceratops opacity laws.

This module provides the core base class for generic opacity laws in Triceratops:
:class:`~triceratops.radiation.opacity.base.OpacityLaw`.  This defines the public unit-bearing
interface and the private log-space interface that subclasses override.

C-backed subclasses delegate the log-space methods to a Cython companion
object that implements the same interface at C speed.
"""

from typing import TYPE_CHECKING, ClassVar, Optional, Union

import numpy as np
from astropy import units as u

from triceratops.utils.misc_utils import ensure_in_units

if TYPE_CHECKING:
    from triceratops._typing import _UnitBearingArrayLike


# ================================================================= #
# Base class for generic opacity laws (frequency-dependent or mean) #
# ================================================================= #


class OpacityLaw:
    r"""Base class for opacity laws :math:`\kappa(\nu, \rho, T)`.

    This class defines the interface for computing frequency-dependent
    opacities as a function of frequency, density, and temperature.  It
    supports both pure-Python implementations and high-performance C-backed
    implementations with a unified API.

    .. important::

        This is an abstract base class and should not be instantiated directly.  Use
        one of the concrete subclasses provided in the library, or create a custom
        subclass that implements the required log-space methods.

    Notes
    -----
    An opacity law describes how radiation interacts with matter through
    absorption and scattering.  In general, the opacity depends on:

    - Frequency :math:`\nu`
    - Mass density :math:`\rho`
    - Temperature :math:`T`

    This base class provides:

    - A **public, unit-aware interface** operating on physical quantities.
    - A **private, log-space interface** for numerical stability and performance.
    - Optional support for **C/Cython-backed implementations**.
    - A consistent interface for computing **log-derivatives** of opacity.

    .. rubric:: Subclass Responsibilities

    Subclasses must implement the opacity law in **log-space** by overriding:

    - :meth:`_log_opacity`
    - :meth:`_dlogkappa_dlogrho`
    - :meth:`_dlogkappa_dlogT`
    - :meth:`_dlogkappa_dlognu`

    All private methods use the argument order :math:`(\ln\nu,\,\ln T,\,\ln\rho)`.
    The public methods handle unit conversion, log-transformation, and
    exponentiation automatically and should **not be overridden**.

    .. rubric:: C-backed Implementations

    Subclasses may delegate computations to a compiled backend by setting
    ``IS_C_BACKED = True`` and overriding :meth:`_initialize_C_object`.
    The private methods then automatically dispatch to the corresponding
    methods on the Cython object (see :mod:`triceratops.radiation.opacity.opacity_base`).

    See Also
    --------
    ~triceratops.radiation.opacity.grey_opacity.base.GreyOpacityLaw : Subclass for frequency-averaged
     (grey) opacity laws.
    ~triceratops.radiation.opacity.opacity_base.C_OpacityBase : C-level companion class.
    """

    IS_C_BACKED: bool = False
    """bool: Whether this opacity law is backed by a Cython extension."""

    mean_type: ClassVar[Optional[str]] = None
    """str or None: Frequency-averaging convention. ``"rosseland"`` or ``"planck"``
    for grey subclasses; ``None`` for frequency-dependent laws."""

    # --------------------------------------------------------------------- #
    # Initialization and Cython backend support                             #
    # --------------------------------------------------------------------- #

    def __init__(self, **parameters) -> None:
        """Initialize the opacity law.

        Parameters
        ----------
        **parameters
            Model-specific parameters forwarded to the C backend when
            :attr:`IS_C_BACKED` is ``True``.
        """
        self._c_object = self._initialize_C_object(**parameters) if self.IS_C_BACKED else None

    def _initialize_C_object(self, **parameters):
        """Construct the Cython backend object.

        Must be overridden by subclasses that set :attr:`IS_C_BACKED = True`.
        The returned object must expose ``log_opacity``, ``dlogkappa_dlogrho``,
        ``dlogkappa_dlogT``, and ``dlogkappa_dlognu`` methods matching the
        log-space interface in :mod:`triceratops.radiation.opacity.opacity_base`.

        Parameters
        ----------
        **parameters
            Model-specific parameters required to construct the backend.

        Returns
        -------
        object
            A Cython object implementing the log-space interface.

        Raises
        ------
        NotImplementedError
            If called on a subclass that declares ``IS_C_BACKED = True`` but
            does not override this method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} sets IS_C_BACKED=True but does not override _initialize_C_object()."
        )

    # ================================================================ #
    # Low-level log-space interface — (log_nu, log_T, log_rho)         #
    # ================================================================ #

    def _log_opacity(
        self,
        log_nu: Union[float, np.ndarray],
        log_T: Union[float, np.ndarray],
        log_rho: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        r"""Return :math:`\ln\kappa` (:math:`\mathrm{cm^2\,g^{-1}}`).

        Parameters
        ----------
        log_nu : float or ndarray
            :math:`\ln\nu` [Hz].
        log_T : float or ndarray
            :math:`\ln T` [K].
        log_rho : float or ndarray
            :math:`\ln\rho` [:math:`\mathrm{g\,cm^{-3}}`].

        Returns
        -------
        float or ndarray
            :math:`\ln\kappa` [:math:`\mathrm{cm^2\,g^{-1}}`].
        """
        if self.IS_C_BACKED:
            return self._c_object.log_opacity(log_nu, log_T, log_rho)
        raise NotImplementedError(f"{type(self).__name__} must implement _log_opacity().")

    def _dlogkappa_dlogrho(
        self,
        log_nu: Union[float, np.ndarray],
        log_T: Union[float, np.ndarray],
        log_rho: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        r"""Return :math:`\partial\ln\kappa/\partial\ln\rho` (dimensionless)."""
        if self.IS_C_BACKED:
            return self._c_object.dlogkappa_dlogrho(log_nu, log_T, log_rho)
        raise NotImplementedError(f"{type(self).__name__} must implement _dlogkappa_dlogrho().")

    def _dlogkappa_dlogT(
        self,
        log_nu: Union[float, np.ndarray],
        log_T: Union[float, np.ndarray],
        log_rho: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        r"""Return :math:`\partial\ln\kappa/\partial\ln T` (dimensionless)."""
        if self.IS_C_BACKED:
            return self._c_object.dlogkappa_dlogT(log_nu, log_T, log_rho)
        raise NotImplementedError(f"{type(self).__name__} must implement _dlogkappa_dlogT().")

    def _dlogkappa_dlognu(
        self,
        log_nu: Union[float, np.ndarray],
        log_T: Union[float, np.ndarray],
        log_rho: Union[float, np.ndarray],
    ) -> Union[float, np.ndarray]:
        r"""Return :math:`\partial\ln\kappa/\partial\ln\nu` (dimensionless)."""
        if self.IS_C_BACKED:
            return self._c_object.dlogkappa_dlognu(log_nu, log_T, log_rho)
        raise NotImplementedError(f"{type(self).__name__} must implement _dlogkappa_dlognu().")

    # ================================================================ #
    # Public unit-bearing interface — (nu, rho, T)                     #
    # ================================================================ #

    def opacity(
        self,
        nu: "_UnitBearingArrayLike",
        rho: "_UnitBearingArrayLike",
        T: "_UnitBearingArrayLike",
    ) -> u.Quantity:
        r"""Return opacity :math:`\kappa(\nu, \rho, T)`.

        Parameters
        ----------
        nu : `~astropy.units.Quantity`
            Frequency, convertible to Hz.
        rho : `~astropy.units.Quantity`
            Mass density, convertible to :math:`\mathrm{g\,cm^{-3}}`.
        T : `~astropy.units.Quantity`
            Temperature, convertible to K.

        Returns
        -------
        `~astropy.units.Quantity`
            Opacity :math:`\kappa` in :math:`\mathrm{cm^2\,g^{-1}}`.

        Notes
        -----
        Inputs are broadcast following NumPy rules.  Internally, this method
        evaluates :math:`\ln\kappa` and exponentiates the result.
        Subclasses should override :meth:`_log_opacity`, not this method.

        Examples
        --------
        .. code-block:: python

            import astropy.units as u
            from triceratops.radiation.opacity import (
                get_opacity,
            )

            op = get_opacity("kramers_ff")
            nu = 1e14 * u.Hz
            kappa = op.opacity(
                nu, 1e-5 * u.g / u.cm**3, 1e7 * u.K
            )
        """
        log_nu = np.log(ensure_in_units(nu, u.Hz))
        log_T = np.log(ensure_in_units(T, u.K))
        log_rho = np.log(ensure_in_units(rho, u.g / u.cm**3))
        return np.exp(self._log_opacity(log_nu, log_T, log_rho)) * u.cm**2 / u.g

    def dlogkappa_dlogrho(
        self,
        nu: "_UnitBearingArrayLike",
        rho: "_UnitBearingArrayLike",
        T: "_UnitBearingArrayLike",
    ) -> Union[float, np.ndarray]:
        r"""Return :math:`\partial\ln\kappa/\partial\ln\rho` (dimensionless).

        Parameters
        ----------
        nu : `~astropy.units.Quantity`
            Frequency, convertible to Hz.
        rho : `~astropy.units.Quantity`
            Mass density, convertible to :math:`\mathrm{g\,cm^{-3}}`.
        T : `~astropy.units.Quantity`
            Temperature, convertible to K.

        Returns
        -------
        float or ndarray
            :math:`\partial\ln\kappa/\partial\ln\rho` (dimensionless).

        Notes
        -----
        For a power-law opacity :math:`\kappa \propto \rho^a`, this returns
        the exponent *a*.

        Examples
        --------
        .. code-block:: python

            import astropy.units as u
            from triceratops.radiation.opacity import (
                get_opacity,
            )

            op = get_opacity("kramers_ff")
            nu = 1e14 * u.Hz
            dlnk_dlnrho = op.dlogkappa_dlogrho(
                nu, 1e-5 * u.g / u.cm**3, 1e7 * u.K
            )
            # Returns 1.0 for Kramers (kappa ~ rho^1)
        """
        log_nu = np.log(ensure_in_units(nu, u.Hz))
        log_T = np.log(ensure_in_units(T, u.K))
        log_rho = np.log(ensure_in_units(rho, u.g / u.cm**3))
        return self._dlogkappa_dlogrho(log_nu, log_T, log_rho)

    def dlogkappa_dlogT(
        self,
        nu: "_UnitBearingArrayLike",
        rho: "_UnitBearingArrayLike",
        T: "_UnitBearingArrayLike",
    ) -> Union[float, np.ndarray]:
        r"""Return :math:`\partial\ln\kappa/\partial\ln T` (dimensionless).

        Parameters
        ----------
        nu : `~astropy.units.Quantity`
            Frequency, convertible to Hz.
        rho : `~astropy.units.Quantity`
            Mass density, convertible to :math:`\mathrm{g\,cm^{-3}}`.
        T : `~astropy.units.Quantity`
            Temperature, convertible to K.

        Returns
        -------
        float or ndarray
            :math:`\partial\ln\kappa/\partial\ln T` (dimensionless).

        Notes
        -----
        For a power-law opacity :math:`\kappa \propto T^b`, this returns
        the exponent *b*.

        Examples
        --------
        .. code-block:: python

            import astropy.units as u
            from triceratops.radiation.opacity import (
                get_opacity,
            )

            op = get_opacity("kramers_ff")
            nu = 1e14 * u.Hz
            dlnk_dlnT = op.dlogkappa_dlogT(
                nu, 1e-5 * u.g / u.cm**3, 1e7 * u.K
            )
            # Returns -3.5 for Kramers (kappa ~ T^-3.5)
        """
        log_nu = np.log(ensure_in_units(nu, u.Hz))
        log_T = np.log(ensure_in_units(T, u.K))
        log_rho = np.log(ensure_in_units(rho, u.g / u.cm**3))
        return self._dlogkappa_dlogT(log_nu, log_T, log_rho)

    @classmethod
    def load_default(cls, **kwargs) -> "OpacityLaw":
        r"""Construct an instance with default settings.

        The base implementation calls ``cls(**kwargs)`` directly, which is
        appropriate for all analytic opacity laws whose parameters are passed
        to their constructor.  Subclasses that require external data (e.g. a
        table file) override this method to perform the necessary I/O before
        constructing the instance.

        Parameters
        ----------
        **kwargs
            Forwarded verbatim to the constructor (or the overriding loader).

        Returns
        -------
        OpacityLaw
            A fully initialised opacity instance.

        Examples
        --------
        .. code-block:: python

            from triceratops.radiation.opacity.grey_opacity.rosseland import (
                KramersESOpacity,
            )

            op = KramersESOpacity.load_default(kappa_es=0.20)
        """
        return cls(**kwargs)

    def dlogkappa_dlognu(
        self,
        nu: "_UnitBearingArrayLike",
        rho: "_UnitBearingArrayLike",
        T: "_UnitBearingArrayLike",
    ) -> Union[float, np.ndarray]:
        r"""Return :math:`\partial\ln\kappa/\partial\ln\nu` (dimensionless).

        Parameters
        ----------
        nu : `~astropy.units.Quantity`
            Frequency, convertible to Hz.
        rho : `~astropy.units.Quantity`
            Mass density, convertible to :math:`\mathrm{g\,cm^{-3}}`.
        T : `~astropy.units.Quantity`
            Temperature, convertible to K.

        Returns
        -------
        float or ndarray
            :math:`\partial\ln\kappa/\partial\ln\nu` (dimensionless).

        Notes
        -----
        Encodes the spectral dependence of the opacity.  For a power-law
        :math:`\kappa \propto \nu^c`, this returns the exponent *c*.
        Grey (frequency-averaged) laws return 0 everywhere.

        Examples
        --------
        .. code-block:: python

            import astropy.units as u
            from triceratops.radiation.opacity import (
                get_opacity,
            )

            op = get_opacity("kramers_ff")
            nu = 1e14 * u.Hz
            dlnk_dlnnu = op.dlogkappa_dlognu(
                nu, 1e-5 * u.g / u.cm**3, 1e7 * u.K
            )
            # Returns 0.0 — Kramers is a grey opacity
        """
        log_nu = np.log(ensure_in_units(nu, u.Hz))
        log_T = np.log(ensure_in_units(T, u.K))
        log_rho = np.log(ensure_in_units(rho, u.g / u.cm**3))
        return self._dlogkappa_dlognu(log_nu, log_T, log_rho)
