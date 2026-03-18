r"""
Concrete one-zone disk model: gas pressure with electron-scattering opacity.

This module provides the canonical :class:`GasPressureElectronScatteringDisk`
implementation of :class:`~triceratops.dynamics.accretion.one_zone.base.OneZoneAccretionDiskBase`.
It is the simplest physically self-consistent one-zone closure and serves as
the reference implementation and the starting point for more elaborate models.

See Also
--------
:mod:`triceratops.dynamics.accretion.one_zone.base` :
    Abstract base class and result container.
:mod:`triceratops.dynamics.accretion.utils` :
    Low-level disk utility functions used by the closure pipeline.
:ref:`one_zone_disk`, :ref:`one_zone_disk_theory`
"""

import numpy as np

from triceratops.dynamics.accretion.one_zone.base import OneZoneAccretionDiskBase
from triceratops.dynamics.accretion.utils import (
    _compute_log_f,
    _log_alpha_viscosity,
    _log_central_temperature,
    _log_midplane_density,
    _log_omega_K,
    _log_optical_depth,
    _log_scale_height,
    _log_T_eff_from_visc_dissipation_rate,
    _log_viscous_timescale,
)
from triceratops.physics_utils import IdealGasEOS
from triceratops.physics_utils.constants import _log_G_cgs, _log_kappa_es_cgs

# ================================================================== #
# GasPressureElectronScatteringDisk                                  #
# ================================================================== #


class GasPressureElectronScatteringDisk(OneZoneAccretionDiskBase):
    r"""
    One-zone disk model dominated by gas pressure with electron-scattering opacity.

    This is the simplest physically self-consistent closure within the one-zone
    framework and serves as the canonical reference implementation.  It follows
    the formulation of :footcite:t:`metzgerTimeDependentModelsAccretion2008`
    and assumes:

    - **Opacity**: pure electron scattering,
      :math:`\kappa_{\rm es} = 0.34\,\text{cm}^2\,\text{g}^{-1}`.
    - **Pressure**: ideal gas,
      :math:`P = \rho k_B T_c / (\mu m_p)`.
    - **Viscosity**: Shakura-Sunyaev :math:`\alpha`-prescription,
      :math:`\nu = \alpha c_s^2 / \Omega`.
    - **Cooling**: optically thick blackbody radiation,
      :math:`T_c = (3\tau/4)^{1/4} T_{\rm eff}`.

    **One-zone geometry**

    The disk is reduced to a single spatial zone by introducing two
    dimensionless radial-averaging constants :math:`A` and :math:`B` that
    relate the area-averaged surface density and the angular-momentum-weighted
    radius to the total disk mass :math:`M_D` and angular momentum :math:`J_D`:

    .. math::

        \Sigma &= \frac{M_D}{\pi A\, R_D^2}, \\[4pt]
        J_D    &= \frac{B}{A}\, M_D \sqrt{G M_{\rm BH} R_D}.

    The ratio :math:`\xi \equiv B/A` allows the disk outer radius
    :math:`R_D` to be recovered from the state vector alone:

    .. math::

        R_D = \frac{J_D^2}{\xi^2\, M_D^2\, G M_{\rm BH}}.

    **Thermodynamic closure**

    Because :math:`\nu` depends on :math:`c_s`, which in turn depends on
    :math:`T_c`, and :math:`T_c` is set by the viscous heating rate, the
    system has an apparent circular dependency.  This is resolved by a
    **warm-start** scheme: the viscous heating rate at step :math:`n` is
    computed using the viscous timescale from step :math:`n-1`:

    .. math::

        Q^+_{(n)}
        = \frac{9}{8\pi A F}\,\frac{G M_{\rm BH}\,\dot{M}_{(n-1)}}{R_D^3}
          \,f(R_D),

    where :math:`\dot{M}_{(n-1)} = F M_D / t_{\rm visc,(n-1)}` and
    :math:`f(R) = 1 - \sqrt{R_{\rm in}/R}` is the zero-torque inner-boundary
    correction.  From :math:`Q^+` and :math:`\tau = \Sigma\kappa_{\rm es}`,
    the central temperature follows:

    .. math::

        T_c = \left(\frac{3\tau}{4}\right)^{1/4} T_{\rm eff},
        \quad
        T_{\rm eff} = \left(\frac{Q^+}{\sigma_{\rm SB}}\right)^{1/4}.

    **Angular-momentum evolution**

    Accreted matter leaves the disk through the inner edge carrying the
    Keplerian specific angular momentum
    :math:`\ell_{\rm in} = \sqrt{G M_{\rm BH} R_{\rm in}}`.
    The angular-momentum sink is therefore:

    .. math::

        \dot{J}_{\rm visc}
        = -f_{\rm disk}\,\frac{M_D}{t_{\rm visc}}
          \,\sqrt{G M_{\rm BH} R_{\rm in}}.

    Because :math:`\ell_{\rm in} \ll \ell_D = J_D / M_D` for
    :math:`R_{\rm in} \ll R_D`, angular momentum drains more slowly than mass
    and the disk outer radius :math:`R_D` grows over time as material
    accretes inward — the standard viscous spreading behaviour.

    Parameters
    ----------
    mu : float, optional
        Mean molecular weight of the disk gas (dimensionless).
        Default ``0.6``, appropriate for a solar-composition fully-ionised
        plasma (:math:`\approx 0.62` for solar abundances).
    A : float, optional
        Radial-profile correction constant for surface-density averaging.
        Default ``1.62``
        (:footcite:t:`metzgerTimeDependentModelsAccretion2008`).
    B : float, optional
        Radial-profile correction constant for angular-momentum averaging.
        Default ``1.33``.
    F : float, optional
        Geometric factor relating the mass-loss rate to
        :math:`M_D / t_{\rm visc}`.  Default ``1.6``.

    Examples
    --------
    Construct a disk model, solve for 5 viscous timescales, and inspect the
    central temperature evolution:

    .. code-block:: python

        import numpy as np
        from astropy import constants as const
        from astropy import units as u
        from triceratops.dynamics.accretion.one_zone import (
            GasPressureElectronScatteringDisk,
        )

        disk = GasPressureElectronScatteringDisk(mu=0.62)

        result = disk.solve(
            initial_conditions={
                "M_D_0": 0.1 * const.M_sun,
                "J_D_0": 1.5e49 * u.g * u.cm**2 / u.s,
            },
            runtime_parameters={
                "M_BH": 3.0 * const.M_sun,
                "R_in": 3.0e6 * u.cm,
                "alpha": 0.1,
                "t_visc_0": 1.0e8 * u.s,
            },
            t_span=(1.0e6, 5.0e9),
            t_eval=np.geomspace(1.0e6, 5.0e9, 200),
        )

        data = result.reconstruct()
        print(data["T_c"].to("K"))
        print(data["R_D"].to("cm"))

    See Also
    --------
    OneZoneAccretionDiskBase :
        Abstract base class describing the full closure pipeline and the
        declarative parameter system.
    :func:`~triceratops.dynamics.accretion.utils.alpha_viscosity` :
        Public interface to the Shakura-Sunyaev viscosity formula used here.
    :ref:`one_zone_disk`, :ref:`one_zone_disk_theory`

    References
    ----------
    .. footbibliography::
    """

    # ==================================================== #
    # Parameter Declarations                               #
    # ==================================================== #

    CONTEXT_PARAMETERS: dict[str, dict] = {
        "mu": {"description": "Mean molecular weight", "base_units": None, "default": 0.6},
        "A": {"description": "Surface density correction", "base_units": None, "default": 1.62},
        "B": {"description": "Angular momentum correction", "base_units": None, "default": 1.33},
        "F": {"description": "Geometric mass-loss factor", "base_units": None, "default": 1.6},
    }

    RUNTIME_PARAMETERS: dict[str, dict] = {
        "M_BH": {
            "description": "Black hole mass",
            "base_units": "g",
            "default": None,
            "log_transform": True,
        },
        "R_in": {
            "description": "Inner truncation radius",
            "base_units": "cm",
            "default": None,
            "log_transform": True,
        },
        "alpha": {
            "description": "Shakura-Sunyaev alpha",
            "base_units": None,
            "default": None,
            "log_transform": False,
        },
        "t_visc_0": {
            "description": "Initial viscous timescale",
            "base_units": "s",
            "default": None,
            "log_transform": False,
        },
    }

    INITIAL_CONDITIONS: dict[str, dict] = {
        "M_D_0": {
            "description": "Initial disk mass",
            "base_units": "g",
            "default": None,
            "log_transform": True,
        },
        "J_D_0": {
            "description": "Initial disk angular momentum",
            "base_units": "g*cm**2/s",
            "default": None,
            "log_transform": True,
        },
    }

    RESULT_FIELDS: dict[str, dict] = {
        "R_D": {"description": "Disk outer radius", "units": "cm"},
        "Sigma": {"description": "Surface density", "units": "g/cm**2"},
        "Omega": {"description": "Angular velocity", "units": "rad/s"},
        "T_eff": {"description": "Effective temperature", "units": "K"},
        "T_c": {"description": "Central temperature", "units": "K"},
        "tau": {"description": "Optical depth", "units": None},
        "c_s": {"description": "Sound speed", "units": "cm/s"},
        "nu": {"description": "Kinematic viscosity", "units": "cm**2/s"},
        "t_visc": {"description": "Viscous timescale", "units": "s"},
        "Q_visc": {"description": "Viscous dissipation rate", "units": "erg cm-2 s-1"},
        "mdot": {"description": "Accretion rate", "units": "g/s"},
        "H": {"description": "Scale height", "units": "cm"},
        "rho_mid": {"description": "Midplane density", "units": "g/cm**3"},
    }

    # ==================================================== #
    # Initialization                                       #
    # ==================================================== #

    def __init__(
        self,
        mu: float = 0.6,
        A: float = 1.62,
        B: float = 1.33,
        F: float = 1.6,
    ):
        # Pass off to the base class to handle context parameter storage and validation
        super().__init__(
            mu=mu,
            A=A,
            B=B,
            F=F,
        )

        # In addition to passing everything off to the context parameters, we'll
        # also generate the equation of state engine.
        self.eos = IdealGasEOS(mu=self._context_parameters["mu"])
        self._A = self._context_parameters["A"]
        self._B = self._context_parameters["B"]
        self._F = self._context_parameters["F"]

    # ==================================================== #
    # Serialization                                        #
    # ==================================================== #

    def to_spec_dict(self) -> dict:
        r"""
        Return a JSON-serialisable spec dict for this model instance.

        The dict encodes the ``mu``, ``A``, ``B``, and ``F`` context
        parameters alongside a ``"target"`` string so that
        :meth:`from_spec_dict` (called by
        :meth:`~OneZoneAccretionResult.from_hdf5`) can reconstruct an
        identical model instance.

        Returns
        -------
        dict
            JSON-serialisable specification dict with keys ``"target"``,
            ``"mu"``, ``"A"``, ``"B"``, and ``"F"``.

        Examples
        --------
        .. code-block:: python

            spec = disk.to_spec_dict()
            import json

            print(json.dumps(spec, indent=2))

        See Also
        --------
        from_spec_dict :
            Class method that reconstructs the model from this dict.
        """
        return {
            "target": "triceratops.dynamics.accretion.one_zone.core:GasPressureElectronScatteringDisk",
            "mu": self._context_parameters["mu"],
            "A": self._A,
            "B": self._B,
            "F": self._F,
        }

    @classmethod
    def from_spec_dict(cls, data: dict) -> "GasPressureElectronScatteringDisk":
        r"""
        Construct a :class:`GasPressureElectronScatteringDisk` from a spec dict.

        Filters the ``"target"`` key (which identifies the class but is not a
        constructor argument) and passes the remaining entries directly to
        ``__init__``.

        Parameters
        ----------
        data : dict
            Spec dict produced by :meth:`to_spec_dict`.  May optionally
            contain a ``"target"`` key, which is silently ignored.

        Returns
        -------
        GasPressureElectronScatteringDisk
            Freshly constructed model instance with parameters matching
            those in *data*.

        Examples
        --------
        .. code-block:: python

            spec = disk.to_spec_dict()
            loaded = GasPressureElectronScatteringDisk.from_spec_dict(
                spec
            )
            assert np.isclose(loaded.eos.mu, disk.eos.mu)

        See Also
        --------
        to_spec_dict :
            Instance method that produces the dict this class method reads.
        """
        kwargs = {k: v for k, v in data.items() if k != "target"}
        return cls(**kwargs)

    # ==================================================== #
    # Closure Pipeline                                     #
    # ==================================================== #

    def _compute_disk_closure(self, t, state, run_params: dict) -> dict:
        r"""
        Compute disk radius, surface density, and angular velocity from the ODE state.

        Inverts the one-zone scaling relations to recover :math:`R_D`,
        :math:`\Sigma`, and :math:`\Omega_K` from the current state vector
        ``[log(M_D), log(J_D)]``.

        **Disk radius**

        The ratio :math:`\xi = B/A` and the angular-momentum definition
        :math:`J_D = \xi M_D \sqrt{G M_{\rm BH} R_D}` give:

        .. math::

            R_D = \frac{J_D^2}{\xi^2\,M_D^2\,G M_{\rm BH}},

        or in log-space:

        .. math::

            \ln R_D = 2\bigl(\ln J_D - \ln\xi - \ln M_D\bigr)
                      - \ln G - \ln M_{\rm BH}.

        **Surface density**

        .. math::

            \Sigma = \frac{M_D}{\pi A R_D^2}
            \implies
            \ln\Sigma = \ln M_D - 2\ln R_D - \ln\pi - \ln A.

        **Angular velocity**

        .. math::

            \Omega_K = \sqrt{\frac{G M_{\rm BH}}{R_D^3}}.

        Parameters
        ----------
        t : float
            Current integration time [s].  Not used directly but required
            by the closure interface.
        state : np.ndarray, shape (2,)
            ``[log(M_D / \text{g}),\; log(J_D / (\text{g\,cm}^2\,\text{s}^{-1}))]``.
        run_params : dict
            Must contain ``"log_M_BH"`` (log of black hole mass in grams).

        Returns
        -------
        dict
            Keys: ``"log_r"``, ``"log_sigma"``, ``"log_Omega"``.
        """
        log_M_BH = run_params["log_M_BH"]
        A = self._A
        B = self._B

        log_m, log_j = state

        log_xi = np.log(B) - np.log(A)
        log_r = 2.0 * (log_j - log_xi - log_m) - _log_G_cgs - log_M_BH

        log_sigma = log_m - 2.0 * log_r - np.log(np.pi) - np.log(A)
        log_omega = _log_omega_K(log_M_BH, log_r)

        return {
            "log_r": log_r,
            "log_sigma": log_sigma,
            "log_Omega": log_omega,
        }

    def _compute_thermodynamic_closure(self, t, state, params: dict, run_params: dict) -> None:
        r"""
        Compute the central temperature and sound speed.

        Uses the viscous heating rate :math:`Q^+` — computed from the cached
        viscous timescale of the *previous* step — and the optically thick
        cooling relation to solve for :math:`T_c` analytically.  This
        warm-start approach breaks the circular dependency between
        :math:`T_c`, :math:`c_s`, :math:`\nu`, and :math:`t_{\rm visc}`.

        **Viscous heating rate**

        The one-zone heating rate (integrated over the disk face) is:

        .. math::

            Q^+
            = \frac{9}{8\pi A F}\,\frac{G M_{\rm BH}\,\dot{M}_{\rm prev}}{R_D^3}
              \,f(R_D),

        where :math:`\dot{M}_{\rm prev} = F M_D / t_{\rm visc}^{(n-1)}` and
        :math:`f(R) = 1 - \sqrt{R_{\rm in}/R}` is the inner-boundary
        correction factor.

        **Effective and central temperatures**

        From :math:`Q^+` and the electron-scattering optical depth
        :math:`\tau = \Sigma\kappa_{\rm es}`:

        .. math::

            T_{\rm eff} &= \left(\frac{Q^+}{\sigma_{\rm SB}}\right)^{1/4}, \\[4pt]
            T_c         &= \left(\frac{3\tau}{4}\right)^{1/4} T_{\rm eff}.

        **Sound speed**

        The isothermal sound speed is computed from :math:`T_c` via the
        equation of state (see :class:`~triceratops.physics_utils.eos.IdealGasEOS`):

        .. math::

            c_s^2 = \frac{k_B T_c}{\mu m_p}.

        Parameters
        ----------
        t : float
            Current integration time [s].  Not used directly.
        state : np.ndarray, shape (2,)
            ``[log(M_D), log(J_D)]``.
        params : dict
            Modified **in-place**.  Must already contain ``"log_r"`` and
            ``"log_sigma"`` from :meth:`_compute_disk_closure`.  On exit,
            adds ``"log_cs"``, ``"log_T_c"``, ``"log_tau"``,
            ``"log_q_visc"``, ``"log_T_eff"``, and ``"f_disk"``.
        run_params : dict
            Must contain ``"log_M_BH"`` and ``"log_R_in"``.

        Raises
        ------
        ValueError
            If ``"t_visc"`` is absent from :attr:`_runtime_cache`.  Seed the
            cache with ``self._runtime_cache = {"t_visc": t_visc_0}`` before
            the first closure call, or use :meth:`~OneZoneAccretionDiskBase.solve`
            which handles this automatically.
        """
        A = self._A
        F = self._F
        log_M_BH = run_params["log_M_BH"]
        log_R_in = run_params["log_R_in"]

        t_visc = self._runtime_cache.get("t_visc", None)
        if t_visc is None:
            raise ValueError(
                "t_visc not found in _runtime_cache.  "
                "Seed self._runtime_cache = {'t_visc': t_visc_0} before calling "
                "the closure pipeline, or pass 't_visc_0' in runtime_parameters."
            )
        log_t_visc = np.log(t_visc)

        log_R = params["log_r"]
        log_m, _ = state

        log_M_dot_prev = log_m - log_t_visc + np.log(F)

        log_q_visc = (
            np.log(9.0 / (8.0 * np.pi * A * F))
            + _log_G_cgs
            + log_M_BH
            + log_M_dot_prev
            - 3.0 * log_R
            + _compute_log_f(log_R, log_R_in)
        )

        log_T_eff = _log_T_eff_from_visc_dissipation_rate(log_q_visc)

        log_sigma = params["log_sigma"]
        log_tau = _log_optical_depth(log_sigma, _log_kappa_es_cgs)

        log_T_c = _log_central_temperature(log_T_eff, log_tau)

        log_cs = self.eos._compute_log_sound_speed(log_T_c)

        params["log_cs"] = log_cs
        params["log_T_c"] = log_T_c
        params["log_tau"] = log_tau
        params["log_q_visc"] = log_q_visc
        params["log_T_eff"] = log_T_eff
        params["f_disk"] = F

    def _compute_viscous_closure(self, t, state, params: dict, run_params: dict) -> None:
        r"""
        Compute the kinematic viscosity and viscous timescale.

        Applies the Shakura-Sunyaev :math:`\alpha`-prescription to obtain the
        kinematic viscosity :math:`\nu`, then derives the viscous (spreading)
        timescale :math:`t_{\rm visc}`:

        .. math::

            \nu = \alpha \frac{c_s^2}{\Omega},
            \qquad
            t_{\rm visc} = \frac{R_D^2}{\nu}.

        The updated :math:`t_{\rm visc}` is stored in
        :attr:`~OneZoneAccretionDiskBase._runtime_cache` by
        :meth:`~OneZoneAccretionDiskBase._IVP_kernel` after this method returns,
        where it seeds the thermodynamic closure at the next solver step.

        Parameters
        ----------
        t : float
            Current integration time [s].  Not used directly.
        state : np.ndarray, shape (2,)
            ``[log(M_D), log(J_D)]``.  Not used directly.
        params : dict
            Modified **in-place**.  Must already contain ``"log_cs"``,
            ``"log_Omega"``, and ``"log_r"`` from the preceding closure
            steps.  On exit, adds ``"log_nu"`` and ``"t_visc"``.
        run_params : dict
            Must contain ``"alpha"`` — the Shakura-Sunyaev viscosity
            parameter.
        """
        alpha = run_params["alpha"]
        log_cs = params["log_cs"]
        log_Omega = params["log_Omega"]

        log_nu = _log_alpha_viscosity(2.0 * log_cs, log_Omega, alpha)
        t_visc = np.exp(_log_viscous_timescale(params["log_r"], log_nu))

        params["log_nu"] = log_nu
        params["t_visc"] = t_visc

    def _compute_angular_momentum_sources(self, t, state, cycle_params: dict, runtime_params: dict) -> float:
        r"""
        Viscous angular-momentum sink at the inner disk edge.

        Accreted matter leaves the disk carrying the Keplerian specific
        angular momentum at the inner truncation radius:

        .. math::

            \ell_{\rm in} = \sqrt{G M_{\rm BH} R_{\rm in}}.

        The resulting angular-momentum drain rate is:

        .. math::

            \dot{J}_{\rm visc}
            = -f_{\rm disk}\,\frac{M_D}{t_{\rm visc}}
              \,\sqrt{G M_{\rm BH} R_{\rm in}}.

        Because :math:`\ell_{\rm in} \ll J_D / M_D` when
        :math:`R_{\rm in} \ll R_D`, angular momentum drains more slowly
        than mass, causing the disk outer radius to increase over time
        as material accretes inward — the standard viscous spreading
        behaviour of an :math:`\alpha`-disk.

        Parameters
        ----------
        t : float
            Current integration time [s].  Not used directly.
        state : np.ndarray, shape (2,)
            ``[log(M_D), log(J_D)]``.
        cycle_params : dict
            Must contain ``"f_disk"`` and ``"t_visc"``.
        runtime_params : dict
            Must contain ``"log_M_BH"`` and ``"log_R_in"``.

        Returns
        -------
        float
            :math:`\dot{J}_{\rm visc}` [:math:`\text{g cm}^2 \text{s}^{-2}`].
            Always negative (angular momentum is removed, not added).
        """
        m = np.exp(state[0])
        f = cycle_params["f_disk"]
        t_visc = cycle_params["t_visc"]
        l_in = np.exp(0.5 * (_log_G_cgs + runtime_params["log_M_BH"] + runtime_params["log_R_in"]))
        return -f * m / t_visc * l_in

    def _extract_result_fields(self, params: dict, state_i: np.ndarray) -> dict:
        r"""
        Extract raw float values for every key in :attr:`RESULT_FIELDS`.

        Translates the fully populated closure parameter dict into the
        physical output quantities declared in :attr:`RESULT_FIELDS`.
        Three derived quantities — scale height :math:`H`, midplane density
        :math:`\rho_{\rm mid}`, and accretion rate :math:`\dot{M}` — are
        computed here from the closure outputs rather than in the forward-
        pass closures, since they are only needed during reconstruction and
        not during the ODE integration.

        **Scale height**

        .. math::

            H = \frac{c_s}{\Omega}

        **Midplane density**

        .. math::

            \rho_{\rm mid} = \frac{\Sigma}{2H}

        **Accretion rate**

        .. math::

            \dot{M} = f_{\rm disk}\,\frac{M_D}{t_{\rm visc}}

        Parameters
        ----------
        params : dict
            Fully populated closure parameter dict for this timestep (output
            of all three closure steps applied by
            :meth:`~OneZoneAccretionDiskBase.reconstruct_state`).  Must
            contain ``"log_r"``, ``"log_sigma"``, ``"log_Omega"``,
            ``"log_T_eff"``, ``"log_T_c"``, ``"log_tau"``, ``"log_cs"``,
            ``"log_nu"``, ``"t_visc"``, ``"log_q_visc"``, and ``"f_disk"``.
        state_i : np.ndarray, shape (2,)
            ``[log(M_D), log(J_D)]`` at this timestep.

        Returns
        -------
        dict
            Raw float values keyed exactly by :attr:`RESULT_FIELDS`:
            ``"R_D"``, ``"Sigma"``, ``"Omega"``, ``"T_eff"``, ``"T_c"``,
            ``"tau"``, ``"c_s"``, ``"nu"``, ``"t_visc"``, ``"Q_visc"``,
            ``"mdot"``, ``"H"``, ``"rho_mid"``.
        """
        log_m = state_i[0]
        log_H = _log_scale_height(params["log_cs"], params["log_Omega"])
        log_rho_mid = _log_midplane_density(params["log_sigma"], log_H)

        return {
            "R_D": np.exp(params["log_r"]),
            "Sigma": np.exp(params["log_sigma"]),
            "Omega": np.exp(params["log_Omega"]),
            "T_eff": np.exp(params["log_T_eff"]),
            "T_c": np.exp(params["log_T_c"]),
            "tau": np.exp(params["log_tau"]),
            "c_s": np.exp(params["log_cs"]),
            "nu": np.exp(params["log_nu"]),
            "t_visc": params["t_visc"],
            "Q_visc": np.exp(params["log_q_visc"]),
            "mdot": params["f_disk"] * np.exp(log_m) / params["t_visc"],
            "H": np.exp(log_H),
            "rho_mid": np.exp(log_rho_mid),
        }
