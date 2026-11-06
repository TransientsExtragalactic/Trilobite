"""
Velocity profile classes.

This module provides an abstract base class and concrete implementations for
upstream velocity profiles used by numerical shock engines. Three standard
profiles are provided:

- :class:`StaticVelocityProfile` — zero velocity field (stationary medium).
- :class:`HomologousVelocityProfile` — homologous flow :math:`u(r,t) = r/t`.
- :class:`ConstantVelocityProfile` — spatially uniform constant velocity.
"""

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from astropy import units as u

from trilobite.utils.misc_utils import ensure_in_units

from .core import _DynamicalProfile

if TYPE_CHECKING:
    pass


# =========================================================================== #
# Velocity Profile ABC                                                        #
# =========================================================================== #
class VelocityProfile(_DynamicalProfile):
    r"""
    Abstract base class for upstream velocity profiles.

    A velocity profile represents the bulk velocity field

    .. math::

        u = u(r, t;\,\boldsymbol{\theta}),

    in :math:`\mathrm{cm\,s^{-1}}`. Subclasses must implement
    :meth:`_validate_and_process_parameters` and :meth:`_opt_eval`.

    See Also
    --------
    StaticVelocityProfile : Zero velocity (stationary medium).
    HomologousVelocityProfile : Homologous flow :math:`u = r/t`.
    ConstantVelocityProfile : Spatially uniform constant velocity.
    """

    OUTPUT_UNITS: ClassVar = u.cm / u.s


# =========================================================================== #
# Concrete Velocity Profiles                                                  #
# =========================================================================== #
class StaticVelocityProfile(VelocityProfile):
    r"""
    Zero velocity field.

    Returns :math:`u(r, t) = 0` everywhere. Suitable for a stationary
    circumstellar medium.

    No parameters are required.

    Examples
    --------
    >>> import astropy.units as u
    >>> StaticVelocityProfile.eval(
    ...     1e16 * u.cm, 10 * u.day
    ... )
    <Quantity 0. cm / s>
    """

    @classmethod
    def _validate_and_process_parameters(cls, **_) -> dict[str, Any]:
        return {}

    @classmethod
    def _opt_eval(cls, r: np.ndarray, t: np.ndarray, **_) -> Any:
        if np.asarray(r).ndim == 0:
            return 0.0
        return np.zeros_like(np.asarray(r), dtype=float)


class HomologousVelocityProfile(VelocityProfile):
    r"""
    Homologous velocity field.

    .. math::

        u(r, t) = \frac{r}{t},

    the velocity field of freely expanding (homologous) ejecta.

    No parameters are required.

    Examples
    --------
    >>> import astropy.units as u
    >>> HomologousVelocityProfile.eval(
    ...     1e16 * u.cm, 10 * u.day
    ... )
    <Quantity ...  cm / s>
    """

    @classmethod
    def _validate_and_process_parameters(cls, **_) -> dict[str, Any]:
        return {}

    @classmethod
    def _opt_eval(cls, r: np.ndarray, t: np.ndarray, **_) -> np.ndarray:
        return np.asarray(r, dtype=float) / np.asarray(t, dtype=float)


class ConstantVelocityProfile(VelocityProfile):
    r"""
    Spatially uniform constant velocity field.

    .. math::

        u(r, t) = v_0,

    a constant velocity independent of radius and time.

    Parameters
    ----------
    v_0 : float or ~astropy.units.Quantity
        Constant velocity. Bare values are interpreted as cm/s.

    Examples
    --------
    >>> import astropy.units as u
    >>> ConstantVelocityProfile.eval(
    ...     1e16 * u.cm, 10 * u.day, v_0=100 * u.km / u.s
    ... )
    <Quantity 10000000. cm / s>
    """

    @classmethod
    def _validate_and_process_parameters(cls, *, v_0, **_) -> dict[str, Any]:
        v = float(ensure_in_units(v_0, u.cm / u.s))
        return {"v_0": v}

    @classmethod
    def _opt_eval(cls, r: np.ndarray, t: np.ndarray, *, v_0: float, **_) -> Any:
        r_arr = np.asarray(r)
        if r_arr.ndim == 0:
            return v_0
        return np.full_like(r_arr, v_0, dtype=float)
