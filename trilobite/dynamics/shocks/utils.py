r"""
Upstream source-function utilities for numerical shock engines.

This module provides :func:`make_homologous_stationary_sources`, a convenience
factory that assembles the four upstream callables
``(rho_1, u_1, rho_4, u_4)`` required by numerical shock engines for the
standard case of homologously expanding ejecta running into a stationary CSM.

For profile construction use the classes in
:mod:`trilobite.dynamics.profiles` directly::

    from trilobite.dynamics.profiles import (
        BrokenPowerLawEjectaProfile,
        WindCSMProfile,
    )

    K, v_t = BrokenPowerLawEjectaProfile.normalize(
        E_ej, M_ej, n=10, delta=0
    )
    rho_ej = BrokenPowerLawEjectaProfile.as_optimized_callable(
        K=K, v_t=v_t, n=10, delta=0
    )
    rho_csm = WindCSMProfile.as_optimized_callable(
        mass_loss_rate=M_dot, wind_velocity=v_wind
    )
    rho_1, u_1, rho_4, u_4 = (
        make_homologous_stationary_sources(
            rho_ej, rho_csm
        )
    )
"""

import inspect
from collections.abc import Callable

import numpy as np

from trilobite.dynamics.profiles.core import _DynamicalProfile


def make_homologous_stationary_sources(
    rho_ej,
    rho_csm,
    ej_params: dict = None,
    csm_params: dict = None,
) -> "tuple[Callable, Callable, Callable, Callable]":
    r"""
    Build the four upstream source callables for homologous ejecta in stationary CSM.

    This convenience factory constructs ``(rho_1, u_1, rho_4, u_4)`` — the
    four two-argument callables expected by numerical shock engines — for the
    standard case of freely expanding homologous ejecta running into a
    stationary circumstellar medium:

    .. math::

        \rho_1(r,\,t) = \rho_{\rm ej}(r,\,t),
        \qquad
        u_1(r,\,t) = \frac{r}{t},
        \qquad
        \rho_4(r,\,t) = \rho_{\rm CSM}(r,\,t),
        \qquad
        u_4(r,\,t) = 0.

    Each source argument may be supplied as either a pre-built unit-free CGS
    callable **or** a :class:`~trilobite.dynamics.profiles.core._DynamicalProfile`
    subclass.  When a profile class is given the corresponding params dict is
    forwarded to :meth:`~trilobite.dynamics.profiles.core._DynamicalProfile.as_optimized_callable`
    to construct the unit-free callable.

    Parameters
    ----------
    rho_ej : callable or type
        Upstream ejecta density.  Either a two-argument unit-free callable
        ``rho_ej(r, t)`` returning density in :math:`\mathrm{g\,cm^{-3}}`, or
        a :class:`~trilobite.dynamics.profiles.ejecta.EjectaDensityProfile` subclass.
        When a class is supplied ``ej_params`` must provide the profile
        parameters.
    rho_csm : callable or type
        Upstream CSM density.  Either a two-argument unit-free callable
        ``rho_csm(r, t)`` returning density in :math:`\mathrm{g\,cm^{-3}}`, or
        a :class:`~trilobite.dynamics.profiles.csm.CSMDensityProfile` subclass.
        When a class is supplied ``csm_params`` must provide the profile
        parameters.
    ej_params : dict, optional
        Keyword arguments forwarded to ``rho_ej.as_optimized_callable()`` when
        ``rho_ej`` is a profile class.  Ignored when ``rho_ej`` is already a
        callable.
    csm_params : dict, optional
        Keyword arguments forwarded to ``rho_csm.as_optimized_callable()`` when
        ``rho_csm`` is a profile class.  Ignored when ``rho_csm`` is already a
        callable.

    Returns
    -------
    rho_1 : callable
        Upstream ejecta density ``rho_1(r, t)`` in :math:`\mathrm{g\,cm^{-3}}`.
    u_1 : callable
        Upstream ejecta velocity ``u_1(r, t) = r/t`` in cm/s (homologous flow).
    rho_4 : callable
        Upstream CSM density ``rho_4(r, t)`` in :math:`\mathrm{g\,cm^{-3}}`.
    u_4 : callable
        Upstream CSM velocity ``u_4(r, t) = 0`` in cm/s (stationary medium).

    See Also
    --------
    trilobite.dynamics.profiles.ejecta.EjectaDensityProfile :
        Base class for ejecta density profiles.
    trilobite.dynamics.profiles.csm.CSMDensityProfile :
        Base class for CSM density profiles.
    """
    if inspect.isclass(rho_ej) and issubclass(rho_ej, _DynamicalProfile):
        rho_1 = rho_ej.as_optimized_callable(**(ej_params or {}))
    else:
        rho_1 = rho_ej

    if inspect.isclass(rho_csm) and issubclass(rho_csm, _DynamicalProfile):
        rho_4 = rho_csm.as_optimized_callable(**(csm_params or {}))
    else:
        rho_4 = rho_csm

    def u_1(r, t):
        return np.asarray(r, dtype=float) / np.asarray(t, dtype=float)

    def u_4(r, _t=None):
        r_arr = np.asarray(r)
        return 0.0 if r_arr.ndim == 0 else np.zeros_like(r_arr, dtype=float)

    return rho_1, u_1, rho_4, u_4
