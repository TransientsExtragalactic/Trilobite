.. _synchrotron_cooling:
=====================================
Radiative Cooling Engines
=====================================

.. hint::

    See :ref:`synchrotron_theory` for a detailed discussion of radiative cooling
    processes and their role in shaping synchrotron spectra.

The :mod:`radiation.synchrotron.cooling` module provides the **low-level and
high-level tools required to compute radiative cooling of relativistic
electrons**. Whereas the :mod:`radiation.synchrotron.core` module focuses on
*emission*, this module focuses on *energy loss* and the evolution of electron
Lorentz factors.

The cooling API is intentionally **simple, explicit, and composable**:

- no assumptions about electron distributions,
- no implicit microphysical closures,
- no hidden state inside cooling engines.

Instead, all physical parameters (magnetic field strength, radiation field
properties, timescales) are supplied explicitly at call time, making these tools
well-suited for inference, parameter studies, and time-dependent modeling.

Overview of the API
-------------------

Radiative cooling is exposed through **cooling engines**, each corresponding to
a distinct physical process. Each engine implements a common interface defined
by :class:`~radiation.synchrotron.cooling.SynchrotronCoolingEngine`.

Every cooling engine provides methods to compute:

- the **cooling rate** :math:`dE/dt`,
- the **cooling time** :math:`t_{\rm cool}`,
- the **cooling Lorentz factor** :math:`\gamma_c`,
- a **characteristic synchrotron frequency** associated with cooled electrons.

Two concrete engines are currently implemented:

- :class:`SynchrotronRadiativeCoolingEngine`
- :class:`InverseComptonCoolingEngine`

Both engines expose **unit-aware public APIs** and **optimized CGS low-level
implementations**.

Cooling Times and Cooling Lorentz Factors
-----------------------------------------

For any radiative process, the cooling time is defined as

.. math::

    t_{\rm cool}(\gamma)
    =
    \frac{\gamma m_e c^2}{|dE/dt|}.

Given a characteristic timescale :math:`t` (e.g. source age or dynamical time),
the **cooling Lorentz factor** :math:`\gamma_c` is defined implicitly by

.. math::

    t_{\rm cool}(\gamma_c) = t.

Electrons with :math:`\gamma > \gamma_c` cool efficiently over time :math:`t`,
while those with :math:`\gamma < \gamma_c` do not. Cooling Lorentz factors play a
central role in determining spectral breaks in synchrotron emission.

Synchrotron Radiative Cooling
-----------------------------

Synchrotron cooling describes energy losses due to synchrotron radiation in a
magnetic field. In the isotropic, Thomson-regime limit, the energy loss rate is

.. math::

    \left(\frac{dE}{dt}\right)_{\rm syn}
    =
    \frac{4}{3}\,\sigma_T c\,\gamma^2\,u_B,
    \qquad
    u_B = \frac{B^2}{8\pi}.

This process is implemented by
:class:`~radiation.synchrotron.cooling.SynchrotronRadiativeCoolingEngine`.

.. currentmodule:: radiation.synchrotron.cooling

.. tab-set::

    .. tab-item:: High-Level API

        The high-level API performs unit validation and returns
        :class:`astropy.units.Quantity` objects.

        .. code-block:: python

            from triceratops.radiation.synchrotron.cooling import (
                SynchrotronRadiativeCoolingEngine
            )
            import astropy.units as u

            engine = SynchrotronRadiativeCoolingEngine()

            B = 1.0 * u.G
            gamma = 1e4
            t_dyn = 1e6 * u.s

            # Cooling rate
            dEdt = engine.compute_cooling_rate(B=B, gamma=gamma)
            print(dEdt)

            # Cooling time at gamma
            t_cool = engine.compute_cooling_time(B=B, gamma=gamma)
            print(t_cool)

            # Cooling Lorentz factor
            gamma_c = engine.compute_cooling_gamma(B=B, t=t_dyn)
            print(gamma_c)

            # Synchrotron frequency of cooled electrons
            nu_c = engine.compute_characteristic_frequency(B=B, gamma=gamma_c)
            print(nu_c)

    .. tab-item:: Low-Level API

        The low-level API operates purely in CGS units and performs no unit
        checking. It is intended for internal use and performance-critical
        workflows.

        .. code-block:: python

            from triceratops.radiation.synchrotron.cooling import (
                _opt_compute_synchrotron_cooling_rate,
                _opt_compute_synchrotron_cooling_time,
                _opt_compute_synchrotron_cooling_gamma,
            )

            B_cgs = 1.0        # Gauss
            gamma = 1e4
            t_dyn = 1e6       # seconds

            dEdt = _opt_compute_synchrotron_cooling_rate(B_cgs, gamma)
            t_cool = _opt_compute_synchrotron_cooling_time(B_cgs, gamma)
            gamma_c = _opt_compute_synchrotron_cooling_gamma(B_cgs, t_dyn)

            print(dEdt, t_cool, gamma_c)

Inverse Compton Cooling
-----------------------

Inverse Compton (IC) cooling describes energy losses due to scattering of ambient
photons by relativistic electrons. In the isotropic, Thomson-regime limit, the
energy loss rate is

.. math::

    \left(\frac{dE}{dt}\right)_{\rm IC}
    =
    \frac{4}{3}\,\sigma_T c\,\gamma^2\,u_{\rm rad},

with radiation energy density

.. math::

    u_{\rm rad}
    =
    \frac{L_{\rm bol}}{4\pi R^2 c}.

This process is implemented by
:class:`~radiation.synchrotron.cooling.InverseComptonCoolingEngine`.

.. tab-set::

    .. tab-item:: High-Level API

        .. code-block:: python

            from triceratops.radiation.synchrotron.cooling import (
                InverseComptonCoolingEngine
            )
            import astropy.units as u

            engine = InverseComptonCoolingEngine()

            L_bol = 1e42 * u.erg / u.s
            R = 1e16 * u.cm
            gamma = 1e4
            t_dyn = 1e6 * u.s

            dEdt = engine.compute_cooling_rate(
                L_bol=L_bol,
                R=R,
                gamma=gamma,
            )
            print(dEdt)

            t_cool = engine.compute_cooling_time(
                L_bol=L_bol,
                R=R,
                gamma=gamma,
            )
            print(t_cool)

            gamma_c = engine.compute_cooling_gamma(
                L_bol=L_bol,
                R=R,
                t=t_dyn,
            )
            print(gamma_c)

            # Synchrotron frequency corresponding to gamma_c
            nu_c = engine.compute_characteristic_frequency(
                B=1.0 * u.G,
                gamma=gamma_c,
            )
            print(nu_c)

    .. tab-item:: Low-Level API

        .. code-block:: python

            from triceratops.radiation.synchrotron.cooling import (
                _opt_compute_IC_cooling_rate,
                _opt_compute_IC_cooling_time,
                _opt_compute_IC_cooling_gamma,
            )

            L_bol = 1e42   # erg/s
            R = 1e16       # cm
            gamma = 1e4
            t_dyn = 1e6   # s

            dEdt = _opt_compute_IC_cooling_rate(L_bol, R, gamma)
            t_cool = _opt_compute_IC_cooling_time(L_bol, R, gamma)
            gamma_c = _opt_compute_IC_cooling_gamma(L_bol, R, t_dyn)

            print(dEdt, t_cool, gamma_c)

Combining Cooling Channels
--------------------------

Triceratops does **not** implicitly combine cooling processes. When multiple
cooling channels are relevant, users are expected to explicitly sum cooling
rates:

.. math::

    \left(\frac{dE}{dt}\right)_{\rm total}
    =
    \left(\frac{dE}{dt}\right)_{\rm syn}
    +
    \left(\frac{dE}{dt}\right)_{\rm IC}.

This explicit design avoids ambiguity and provides full control in inference
and modeling pipelines.

Higher-level modeling layers may build composite cooling models on top of these
engines, but the primitives provided here form the **canonical foundation for
radiative cooling calculations in Triceratops**.
