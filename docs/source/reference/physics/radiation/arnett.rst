.. _arnett:

================
Arnett Engines
================

.. hint::

    This page covers both the physics and the usage of the Arnett engines in
    :mod:`trilobite.radiation.arnett`. If you only want to compute a light curve, skip to
    :ref:`arnett_usage`. To add a new power source, see :ref:`arnett_custom_engines`.

The bolometric light curves of many optically thick transients are set not by how fast their energy source
releases energy, but by how fast the ejecta can let it out. In a supernova, the radioactive decay of
:math:`^{56}\mathrm{Ni}` deposits energy deep inside an expanding envelope, and the photons then random-walk
outward on a timescale of weeks. The same is true for a magnetar embedded in the ejecta of a superluminous
supernova (SLSN). The simplest model that captures this behavior is the one-zone *Arnett model*, which follows
the internal energy of a homologously expanding, radiation-dominated sphere that is heated from within and
loses energy by diffusion.

The :mod:`trilobite.radiation.arnett` module implements this model as a family of **engines**. Every engine
solves the same diffusion integral on a compiled quadrature kernel; engines differ only in *what heats the
ejecta* and *how efficiently that heat is trapped*. The engines are intended for stripped-envelope supernovae
(SESNe, powered by :math:`^{56}\mathrm{Ni}`), SLSNe (powered by magnetar spin-down, possibly with
:math:`^{56}\mathrm{Ni}`), and any other transient for which a gray, one-zone description of the diffusion
is adequate.

.. note::

    The Arnett model is a bolometric model. It predicts the total radiated luminosity :math:`L(t)`, not a
    spectrum or a photometric band. It also describes the *diffusion phase* of the light curve, which is where
    the photosphere is well approximated by the ejecta surface. It is not appropriate for the nebular phase
    (where the ejecta are no longer optically thick), or for events in which the luminosity is dominated by
    shock cooling or circumstellar interaction.

.. contents::
    :local:
    :depth: 2

----

Theoretical Background
----------------------

The One-Zone Model
^^^^^^^^^^^^^^^^^^

The model follows :footcite:t:`1982ApJ...253..785A`. We consider ejecta of mass :math:`M_\mathrm{ej}` and kinetic energy
:math:`E_\mathrm{ej}` and make the following assumptions:

1. **One zone.** The ejecta are treated as a single uniform-density sphere with a single radiation-dominated
   internal energy :math:`E_\mathrm{th}(t)`, so that the pressure is :math:`p = E_\mathrm{th}/3V`.
2. **Homologous expansion.** The radius evolves as :math:`R(t) = R_0 + v_\mathrm{sc}\,t`, where the scale velocity
   is fixed by the kinetic energy of a uniform sphere,

   .. math::

       E_\mathrm{ej} = \frac{3}{10}\,M_\mathrm{ej}\,v_\mathrm{sc}^2
       \qquad\Longrightarrow\qquad
       v_\mathrm{sc} = \sqrt{\frac{10\,E_\mathrm{ej}}{3\,M_\mathrm{ej}}}.

   The ejecta are not assumed to decelerate or accelerate: :math:`E_\mathrm{ej}` is the *final* kinetic energy,
   and includes any energy added by the central source.
3. **Gray diffusion.** The luminosity is the internal energy divided by the diffusion time,

   .. math::

       L(t) = \frac{E_\mathrm{th}(t)}{t_\mathrm{diff}(t)}, \qquad
       t_\mathrm{diff}(t) = \frac{\kappa\,M_\mathrm{ej}}{\beta\,c\,R(t)},

   where :math:`\kappa` is a constant optical opacity and :math:`\beta` is a dimensionless constant of order
   :math:`13.8` that absorbs the geometry and density profile of the ejecta. The quantity
   :math:`t_\mathrm{diff}` decreases as the ejecta expand, because the photons have less material to cross.
4. **A deposited power** :math:`Q(t)` heats the ejecta. This is the power injected by the source,
   :math:`P(t)`, times the fraction :math:`D(t)` of it that is thermalized.

The first law of thermodynamics for a radiation-dominated gas with :math:`p = E_\mathrm{th}/3V` and
:math:`V \propto R^3` reads

.. math::

    \frac{d}{dt}\left(E_\mathrm{th}\,R\right) = R\left(Q - L\right).

Defining :math:`x = E_\mathrm{th} R` and using :math:`L = E_\mathrm{th}/t_\mathrm{diff}`, this becomes

.. math::

    \frac{dx}{dt} = R\,Q - \frac{x}{t_\mathrm{diff}}.

.. _arnett_clocks:

The Diffusion Clock
^^^^^^^^^^^^^^^^^^^

Because :math:`R = v_\mathrm{sc}(t + a)` with :math:`a = R_0/v_\mathrm{sc}`, the inverse diffusion time is

.. math::

    \frac{1}{t_\mathrm{diff}(t)} = \frac{\beta\,c\,v_\mathrm{sc}\,(t + a)}{\kappa\,M_\mathrm{ej}}
    \equiv \frac{2\,(t + a)}{t_d^2},
    \qquad
    t_d = \sqrt{\frac{2\,\kappa\,M_\mathrm{ej}}{\beta\,c\,v_\mathrm{sc}}}.

The constant :math:`t_d` is the **Arnett diffusion time**; it is the time at which the homologous clock defined
below reaches unity. We then define the **diffusion clock**

.. math::

    u(t) = \int_0^t \frac{dt'}{t_\mathrm{diff}(t')} = \frac{t^2 + 2at}{t_d^2},

which counts the number of diffusion times that have elapsed. Since :math:`L = x\,(du/dt)/R` and
:math:`(du/dt)/R = 2/(v_\mathrm{sc} t_d^2)` is a constant, :math:`L` is proportional to :math:`x`, and the
equation for :math:`x` turns into a relaxation equation for the luminosity,

.. math::

    \frac{dL}{du} = Q(t) - L, \qquad L(0) = L_\mathrm{init}.

That is, **the luminosity relaxes toward the deposited power on the diffusion clock**. The initial luminosity is
nonzero only if the ejecta start with stored heat: for an initial thermal energy :math:`E_\mathrm{th,0}`,
:math:`L_\mathrm{init} = E_\mathrm{th,0}\,(2a/t_d^2)`.

Two clocks are implemented:

.. list-table::
    :header-rows: 1
    :widths: 20 30 50

    * - Clock
      - :math:`u(t)`
      - Meaning
    * - ``"homologous"``
      - :math:`(t/t_d)^2`
      - Ejecta that start from a point (:math:`R_0 = 0`) with no stored heat. This is the classic Arnett
        model and is the default.
    * - ``"finite_radius"``
      - :math:`(t^2 + 2at)/t_d^2`
      - Ejecta that start from a finite radius :math:`R_0` with initial thermal energy :math:`E_\mathrm{th,0}`,
        as in :footcite:t:`Chatzopoulos2012`. Relevant for progenitors with extended envelopes.

The Formal Solution
^^^^^^^^^^^^^^^^^^^

The relaxation equation is linear, and is solved with the integrating factor :math:`e^{u}`:

.. math::

    \frac{d}{du}\left(L\,e^{u}\right) = Q\,e^{u}
    \qquad\Longrightarrow\qquad
    L(t) = L_\mathrm{init}\,e^{-u(t)} + \int_0^t Q(t')\,\frac{du}{dt'}\,e^{-[u(t) - u(t')]}\,dt'.

For the homologous clock :math:`du/dt = 2t/t_d^2`, which recovers the familiar form of the Arnett integral,

.. math::

    L(t) = e^{-(t/t_d)^2}\int_0^t \frac{2t'}{t_d^2}\,Q(t')\,e^{(t'/t_d)^2}\,dt'.

The integral has an intuitive interpretation: energy deposited at time :math:`t'` is released into the light
curve with the weight :math:`(du/dt')\,e^{-[u(t) - u(t')]}`, a kernel that is peaked at the present time and falls
off over about one diffusion time. The luminosity therefore *lags* the heating, and the lag is
:math:`\sim t_d`.

Incomplete Trapping
^^^^^^^^^^^^^^^^^^^

The deposited power is :math:`Q = P\,D`. For a source whose energy is carried by hard photons or particles (the
gamma rays and positrons from :math:`^{56}\mathrm{Co}`, or the high-energy radiation of a magnetar wind nebula),
only a fraction of the injected power is thermalized. For a uniform sphere the radial optical depth to a
mass-absorption opacity :math:`\kappa_h` is

.. math::

    \tau_h(t) = \frac{3\,\kappa_h\,M_\mathrm{ej}}{4\pi R^2}
    = \frac{A_h}{t^2},
    \qquad
    A_h = \frac{9\,\kappa_h\,M_\mathrm{ej}^2}{40\pi\,E_\mathrm{ej}},

where the last form uses :math:`R = v_\mathrm{sc} t`. The Arnett engines use the one-zone deposition fraction

.. math::

    D(t) = 1 - e^{-A_h/t^2},

which tends to unity at early times and to :math:`A_h/t^2` at late times. This is a coarse treatment:
:footcite:t:`Sharon2020` show that the gamma-ray deposition in realistic core-collapse ejecta differs in detail
from this form, and :footcite:t:`Milne1999` show that positron escape depends on the magnetic field and
ionization state of the ejecta. The opacities :math:`\kappa_h` should therefore be thought of as effective
parameters of the model.

.. _arnett_trapping_mode:

Where the Trapping is Applied
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are two ways to apply the trapping factor, which is the same function :math:`1 - e^{-A_h/t^2}` in both. They are
selected by the option ``trapping_mode`` of the engines and of the optical models.

- ``"deposition"`` (the default). The factor is the fraction of the power that is thermalized *when it is deposited*,
  and is part of the heating :math:`Q = P\,D` inside the integral above. Energy that leaks out of the ejecta before it
  has diffused is gone for good.
- ``"emergence"``. The factor is applied to the diffused luminosity at the time of observation,

  .. math::

      L(t) = \left(1 - e^{-A_h/t^2}\right) L_\mathrm{diffused}(t),

  where :math:`L_\mathrm{diffused}` is the luminosity of the same source with no leakage. This is the form used by the
  MOSFiT magnetar model (Eq. 5 of :footcite:t:`Nicholl_2017`), and is what to use to reproduce it.

The two agree when the source is fully trapped, :math:`A_h/t^2 \gg 1`, and differ when it leaks while the energy is still
diffusing out. Each channel has its own factor in both modes, and the two modes give the same result for a channel that
does not leak.

The Emergence Factor
^^^^^^^^^^^^^^^^^^^^

Finally, the engines allow for a multiplicative **emergence factor** :math:`\mathcal{E}(t)` on the diffused
luminosity, which is intended to model corrections to the observed luminosity that are not part of the diffusion
itself. Altogether, the quantity computed by the engines is

.. math::

    L(t) = \mathcal{E}(t)\left[L_\mathrm{init}\,e^{-u(t)} + \int_0^t P(t')\,D(t')\,\frac{du}{dt'}\,
    e^{-[u(t) - u(t')]}\,dt'\right].

By default :math:`\mathcal{E} = 1`.

----

Working with Arnett Models
--------------------------

Every engine in :mod:`trilobite.radiation.arnett` is, in effect, a thin wrapper around a single integrator. The
integrator evaluates the formal solution above at each requested time using a fixed quadrature rule, and is
compiled with ``numba`` so that it can be evaluated over thousands of parameter realizations in parallel (as is
needed for inference). Everything else, i.e. the clock, the injected power, the deposition fraction, and the
emergence factor, is a component that is plugged into the integrator. An engine is the particular combination
of those components that describes a physical scenario.

Because the diffusion equation is linear in the deposited power, an engine with several power sources is solved
one source (a :class:`~trilobite.radiation.arnett.Channel`) at a time, and the luminosities are summed. Each
source keeps its own trapping.

Existing Engines
^^^^^^^^^^^^^^^^

.. list-table::
    :header-rows: 1
    :widths: 30 40 30

    * - Engine
      - Power source
      - Source parameters
    * - :class:`~trilobite.radiation.arnett.NickelCobaltArnettEngine`
      - Radioactive decay of :math:`^{56}\mathrm{Ni} \to {}^{56}\mathrm{Co} \to {}^{56}\mathrm{Fe}`, split into a
        gamma-ray channel and a positron channel. Typical of SESNe.
      - ``M_Ni``, ``kappa_gamma``, ``kappa_positron``
    * - :class:`~trilobite.radiation.arnett.MagnetarArnettEngine`
      - Spin-down of a magnetar with a free braking index :math:`n`
        (:footcite:p:`Kasen2010,Omand2024`). Typical of SLSNe.
      - ``P_spin``, ``B_dipole``, ``braking_index``, ``kappa_magnetar``, ``I_ns``, ``R_ns``
    * - :class:`~trilobite.radiation.arnett.MagnetarNickelArnettEngine`
      - Both of the above, evolving on a common clock with independent trapping.
      - The union of the two sets above

All engines also take the *common* parameters ``M_ej``, ``E_ej``, ``kappa``, ``R_0``, and ``E_th_0``. The last
two are zero unless the engine is built with ``clock="finite_radius"``. The full set of arguments, their units, and
their defaults are tabulated in the ``Notes`` section of each class.

.. _arnett_usage:

Instantiation and Usage
^^^^^^^^^^^^^^^^^^^^^^^

An engine is constructed once, with the *structural* choices of the model, and is then evaluated many times with
different *physical* parameters. The structural choices are the options of
:class:`~trilobite.radiation.arnett.ArnettEngine`:

.. list-table::
    :header-rows: 1
    :widths: 20 20 60

    * - Option
      - Default
      - Meaning
    * - ``gamma_trapping``
      - ``True``
      - Include incomplete trapping of the hard-photon channel(s). If ``False``, all of the injected power
        is thermalized.
    * - ``positron_trapping``
      - ``False``
      - (Nickel engines only.) Include incomplete trapping of the positron channel. By default positrons
        deposit locally and completely.
    * - ``trapping_mode``
      - ``"deposition"``
      - Where the incomplete trapping is applied, to the power as it is deposited (``"deposition"``) or to the
        diffused luminosity at the time of observation (``"emergence"``, as in MOSFiT). See
        :ref:`arnett_trapping_mode`.
    * - ``clock``
      - ``"homologous"``
      - ``"homologous"`` or ``"finite_radius"``; see :ref:`arnett_clocks`.
    * - ``beta``
      - ``13.8``
      - Arnett's constant :math:`\beta`.
    * - ``quadrature``
      - engine-specific
      - A :class:`~trilobite.radiation.arnett.QuadratureRule`; see :ref:`arnett_quadrature`.
    * - ``window_depth``
      - ``40``
      - Number of diffusion-clock units of past injection retained by the integrator.

The light curve is then computed with :meth:`~trilobite.radiation.arnett.ArnettEngine.compute_luminosity` (or,
equivalently, by calling the engine). Parameters may be :class:`~astropy.units.Quantity` objects, which are
converted, or bare floats, which are interpreted in CGS.

.. code-block:: python

    import astropy.units as u
    import numpy as np

    from trilobite.radiation.arnett import NickelCobaltArnettEngine

    engine = NickelCobaltArnettEngine()
    time = np.linspace(5, 120, 100) * u.day

    L = engine.compute_luminosity(
        time,
        M_ej=2 * u.Msun,
        E_ej=1e51 * u.erg,
        kappa=0.1 * u.cm**2 / u.g,
        M_Ni=0.3 * u.Msun,
    )  # shape (100,), erg / s

The Arnett diffusion time for a given set of ejecta can be computed on its own:

.. code-block:: python

    t_d = engine.compute_diffusion_time(2 * u.Msun, 1e51 * u.erg, 0.1 * u.cm**2 / u.g)
    print(t_d.to(u.day))  # ~16.8 d

**Parameter sweeps.** Every physical parameter may be a scalar or a one-dimensional array. All array-valued
parameters must have the same length :math:`N`, and scalars are broadcast against them. The result then has shape
``(n_times, N)``. This is the intended way to evaluate many parameter sets (e.g. the walkers of an MCMC
sampler) in a single call; the realizations are computed in parallel.

.. code-block:: python

    M_ej = np.array([1.0, 2.0, 3.0]) * u.Msun

    L = engine(
        time,
        M_ej=M_ej,  # array: 3 realizations
        E_ej=1e51 * u.erg,  # scalars broadcast
        kappa=0.1 * u.cm**2 / u.g,
        M_Ni=0.1 * u.Msun,
    )  # shape (100, 3)

**Combining sources.** Engines with more than one power source are used in the same way; the parameters of
the sources are simply combined.

.. code-block:: python

    from trilobite.radiation.arnett import MagnetarNickelArnettEngine, QuadratureRule

    engine = MagnetarNickelArnettEngine(
        positron_trapping=True,
        quadrature=QuadratureRule.tanh_sinh(96),
    )

    L = engine(
        time,
        M_ej=5 * u.Msun,
        E_ej=1e51 * u.erg,
        kappa=0.1 * u.cm**2 / u.g,
        M_Ni=0.1 * u.Msun,  # nickel parameters
        P_spin=2 * u.ms,  # magnetar parameters
        B_dipole=3e14 * u.G,
    )

For a finite starting radius and stored heat, build the engine with ``clock="finite_radius"``, which
enables the ``R_0`` and ``E_th_0`` parameters:

.. code-block:: python

    engine = NickelCobaltArnettEngine(clock="finite_radius")
    L = engine(
        time,
        M_ej=2 * u.Msun,
        E_ej=1e51 * u.erg,
        kappa=0.1 * u.cm**2 / u.g,
        M_Ni=0.3 * u.Msun,
        R_0=1e13 * u.cm,
        E_th_0=1e49 * u.erg,
    )

.. important::

    Passing a nonzero ``R_0`` or ``E_th_0`` to an engine with the default ``"homologous"`` clock raises a
    ``ValueError`` rather than being silently ignored. Passing a parameter that the engine does not use at
    all (for example ``M_Ni`` to a :class:`~trilobite.radiation.arnett.MagnetarArnettEngine`) raises a
    ``TypeError`` listing the accepted parameters.

.. note::

    Parameters that depend on a switch are accepted but ignored when the switch is off. For example,
    ``kappa_gamma`` has no effect on an engine built with ``gamma_trapping=False``.

Optical Photometry Models
^^^^^^^^^^^^^^^^^^^^^^^^^

The engines predict a bolometric luminosity. To compare with photometry, use the optical models in
:mod:`trilobite.models.supernovae.arnett`, which radiate the engine luminosity from a photosphere and convolve the
spectrum with a :class:`~trilobite.utils.phot_utils.FilterBundle`. There is one model for each engine, and each comes
in two versions, which differ in the spectrum of the photosphere: a blackbody, or a blackbody with a UV suppression (the
``Cutoff`` models; see :ref:`blackbody_cutoff`). They are used and fit like any other
:class:`~trilobite.models.core.optical.OpticalModel`:

.. list-table::
    :header-rows: 1
    :widths: 34 33 33

    * - Engine
      - Blackbody model
      - UV-suppressed model
    * - :class:`~trilobite.radiation.arnett.NickelCobaltArnettEngine`
      - :class:`~trilobite.models.supernovae.arnett.NickelCobaltArnettModel`
      - :class:`~trilobite.models.supernovae.arnett.NickelCobaltArnettCutoffModel`
    * - :class:`~trilobite.radiation.arnett.MagnetarArnettEngine`
      - :class:`~trilobite.models.supernovae.arnett.MagnetarArnettModel`
      - :class:`~trilobite.models.supernovae.arnett.MagnetarArnettCutoffModel`
    * - :class:`~trilobite.radiation.arnett.MagnetarNickelArnettEngine`
      - :class:`~trilobite.models.supernovae.arnett.MagnetarNickelArnettModel`
      - :class:`~trilobite.models.supernovae.arnett.MagnetarNickelArnettCutoffModel`

.. code-block:: python

    from trilobite.models.supernovae import NickelCobaltArnettModel

    model = NickelCobaltArnettModel(bundle, redshift=0.02)  # bundle: a FilterBundle
    flux = model.forward_model(
        {"band_index": ["V", "R", "V"], "time": [10, 10, 30] * u.day},
        {"M_ej": 2 * u.Msun, "E_ej": 1e51 * u.erg, "M_Ni": 0.3 * u.Msun},
    )

The photosphere of all of them is that of MOSFiT (:footcite:p:`Nicholl_2017`): the ejecta expand at constant velocity,
the temperature follows from the Stefan-Boltzmann law, and the photosphere recedes into the ejecta once the temperature
reaches a floor, ``T_floor``. The UV-suppressed spectrum is also theirs. The magnetar model that reproduces MOSFiT's
``slsn`` model is
:class:`~trilobite.models.supernovae.arnett.MagnetarArnettCutoffModel` with ``trapping_mode="emergence"``, as in
:ref:`sphx_glr_auto_examples_inference_plot_slsn_magnetar_swift.py`.

The ``time`` variable is measured on the time axis of the data, and the parameter ``t_exp`` is the explosion epoch on that
axis. If the explosion epoch is known, measure the time from it (with the ``time_start`` of the data containers, as for the
radio models) and leave ``t_exp`` at its default of zero, as above. If it is not, fit ``t_exp`` along with the physical
parameters.

----

The Structure of the Arnett Engine
----------------------------------

It is useful to know how an engine is assembled, both to understand its cost and accuracy and to write a new
engine. The pieces are:

.. code-block:: text

    ArnettEngine
    |-- DiffusionClock        u(t), du/dt, t(u)           (homologous or finite-radius)
    |-- Channel(s)            one per power source
    |   |-- injection         P(t, params)  [erg/s]       compiled, user-supplied
    |   `-- opacity           (optional)  -> D(t) = 1 - exp(-A/t^2)
    |-- emergence function    E(t, params)  (optional)    compiled
    `-- QuadratureRule        nodes and weights on [0, 1]

On construction, the engine collects its channels and, for each, asks :func:`~trilobite.radiation.arnett.build_solver`
for a compiled integrator specialized to that clock, injection function, deposition function, and emergence
function. These solvers are compiled on first use and cached for the lifetime of the process, so that engines
with the same physics share a compilation. When the engine is called, it

1. converts the inputs to CGS and computes :math:`t_d` (and the clock parameters),
2. computes the trapping parameter :math:`A_h` for each trapped channel,
3. broadcasts all parameters to a common length and packs them into one row per realization, and
4. runs each channel's solver and sums the results.

For each requested time and realization, the solver integrates the formal solution only over the window
:math:`[t_\mathrm{min}, t]`, where :math:`t_\mathrm{min} = t(\max[u(t) - S, 0])` and
:math:`S` is the ``window_depth``. Injection more than :math:`S` diffusion times in the past is exponentially
suppressed and is dropped; for the default :math:`S = 40` this is a fractional error of
:math:`e^{-40} \approx 4\times10^{-18}`. The window keeps the quadrature nodes where the kernel is
non-negligible, which matters at late times when :math:`u \gg 1`.

.. note::

    The solver itself is available directly as :func:`~trilobite.radiation.arnett.build_solver` for cases where
    the engine interface is too restrictive, e.g. when you need a custom deposition function. It operates on
    plain CGS arrays and performs no unit handling or validation; see its docstring for the call signature.

Components
^^^^^^^^^^

Injection Functions
~~~~~~~~~~~~~~~~~~~

An **injection function** gives the power :math:`P(t)` released by the source in :math:`\mathrm{erg\,s^{-1}}`.
It must be a function decorated with ``numba.njit`` with the signature

.. code-block:: python

    @njit
    def injection(t, params):
        # t: time since explosion, in s (float)
        # params: 1D float64 array, the injection parameters of one realization
        return ...  # erg / s

so that it can be called from inside the compiled integrator. Parameters are passed positionally in an array
whose length is the ``n_injection`` of the :class:`~trilobite.radiation.arnett.Channel`; the order is
whatever the engine's ``_source_parameters`` method returns. The implementations used by the built-in engines are
good examples: the nickel channels evaluate the exact Bateman solution for the :math:`^{56}\mathrm{Ni}`
chain, and the magnetar channel evaluates

.. math::

    P(t) = L_0\left(1 + \frac{t}{t_\mathrm{sd}}\right)^{-\frac{n+1}{n-1}}

with :math:`L_0`, :math:`t_\mathrm{sd}`, and the exponent computed from the spin period, field, and braking index.

Deposition Functions
~~~~~~~~~~~~~~~~~~~~

A **deposition function** :math:`D(t)` has the same signature as an injection function but returns a
dimensionless fraction in :math:`[0, 1]`. Within an engine, a deposition function is not specified directly.
Instead, each :class:`~trilobite.radiation.arnett.Channel` carries an optional ``opacity`` name:

- ``opacity=None``: the channel is fully thermalized, :math:`D = 1`.
- ``opacity="name"``: the channel is trapped as :math:`D = 1 - e^{-A/t^2}`, where :math:`A` is built from the
  opacity that the engine reports under that name (see :ref:`arnett_custom_engines`).

Arbitrary deposition functions may be used by calling :func:`~trilobite.radiation.arnett.build_solver` directly
with the ``deposition_function`` keyword.

Emergence Function
~~~~~~~~~~~~~~~~~~

The **emergence function** :math:`\mathcal{E}(t)` multiplies the diffused luminosity (including the
:math:`L_\mathrm{init}` term) at the requested time. It is supplied by overriding
:meth:`~trilobite.radiation.arnett.ArnettEngine._get_emergence_function` to return a ``numba.njit`` function
``E(t, params)``, and defaults to no correction. The engines currently pass the emergence function an empty
parameter row, so it can depend on time alone.

.. _arnett_quadrature:

Quadrature
~~~~~~~~~~

The integral is evaluated with a :class:`~trilobite.radiation.arnett.QuadratureRule`, an immutable triple of
nodes :math:`x_i`, complementary nodes :math:`1 - x_i`, and weights :math:`w_i` on :math:`[0, 1]`, which is mapped
onto :math:`[t_\mathrm{min}, t]`. The complementary nodes are stored separately because the diffusion kernel
depends on the distance to the *upper* end point, and computing :math:`1 - x_i` in floating point loses digits
where the kernel is largest.

Three constructors are provided, and the best choice depends on where the injected power has structure:

.. list-table::
    :header-rows: 1
    :widths: 35 65

    * - Constructor
      - Use
    * - :meth:`~trilobite.radiation.arnett.QuadratureRule.gauss_legendre`
      - Smooth integrands with interior structure. The default of the nickel engine (96 nodes).
    * - :meth:`~trilobite.radiation.arnett.QuadratureRule.gauss_legendre_power`
      - Packs nodes toward :math:`t_\mathrm{min}` (via :math:`x = y^p`). The default for the magnetar engines
        (128 nodes, :math:`p = 3`), where spin-down is often fast compared to the diffusion time.
    * - :meth:`~trilobite.radiation.arnett.QuadratureRule.tanh_sinh`
      - Double-exponential rule that is very strong at both end points and sparse in the interior.

.. code-block:: python

    from trilobite.radiation.arnett import MagnetarArnettEngine, QuadratureRule

    # A magnetar that spins down within a day needs nodes near the explosion.
    engine = MagnetarArnettEngine(quadrature=QuadratureRule.gauss_legendre_power(192, power=4))

When in doubt, check convergence by comparing the light curve at two node counts.

.. _arnett_custom_engines:

Implementing Custom Engines
^^^^^^^^^^^^^^^^^^^^^^^^^^^

To add a new power source, subclass :class:`~trilobite.radiation.arnett.ArnettEngine` and supply its physics.
The subclass must provide

1. ``_PARAMETER_UNITS``: a dictionary mapping every accepted parameter name to its unit. Extend the one on the
   base class, which holds the common parameters.
2. :meth:`~trilobite.radiation.arnett.ArnettEngine._get_channels`: a list of
   :class:`~trilobite.radiation.arnett.Channel` objects, one per source. Names must be unique.
3. :meth:`~trilobite.radiation.arnett.ArnettEngine._source_parameters`: a method that receives the engine's
   source parameters as CGS keyword arguments (everything in ``_PARAMETER_UNITS`` other than the common
   parameters) and returns a pair ``(injection, opacities)``. ``injection`` maps each channel name to the tuple
   of that channel's injection parameters (scalars or 1D arrays), and ``opacities`` maps each opacity name used
   by a channel to its value in :math:`\mathrm{cm^2\,g^{-1}}`.

and may override

- :meth:`~trilobite.radiation.arnett.ArnettEngine._default_quadrature`, to choose a rule suited to the source, and
- :meth:`~trilobite.radiation.arnett.ArnettEngine._get_emergence_function`.

As an example, here is an engine powered by a fallback-accretion-like power law,
:math:`P(t) = L_0\,(1 + t/t_0)^{-5/3}`, which is fully thermalized:

.. code-block:: python

    import astropy.units as u
    from numba import njit

    from trilobite.radiation.arnett import ArnettEngine, Channel, QuadratureRule


    @njit
    def _fallback_injection(t, params):
        # params = [L_0, t_0]
        return params[0] * (1.0 + t / params[1]) ** (-5.0 / 3.0)


    class FallbackArnettEngine(ArnettEngine):
        _PARAMETER_UNITS = {
            **ArnettEngine._PARAMETER_UNITS,
            "L_0": u.erg / u.s,
            "t_0": u.s,
        }

        @classmethod
        def _default_quadrature(cls):
            return QuadratureRule.gauss_legendre_power(128, power=3)

        def _get_channels(self):
            return [Channel("fallback", _fallback_injection, 2)]

        def _source_parameters(self, L_0, t_0):
            return {"fallback": (L_0, t_0)}, {}


    engine = FallbackArnettEngine()
    L = engine(
        time,
        M_ej=5 * u.Msun,
        E_ej=1e51 * u.erg,
        kappa=0.1 * u.cm**2 / u.g,
        L_0=1e44 * u.erg / u.s,
        t_0=1e5 * u.s,
    )

To make a channel leak, give it an opacity name, e.g. ``Channel("fallback", _fallback_injection, 2, "kappa_hard")``,
add ``"kappa_hard": u.cm**2 / u.g`` to ``_PARAMETER_UNITS``, and return ``{"kappa_hard": kappa_hard}`` as the
second element of ``_source_parameters``. See the source of
:class:`~trilobite.radiation.arnett.NickelCobaltArnettEngine` for a worked example with two channels and
user-controlled switches.

.. note::

    The injection function must be decorated with ``numba.njit`` and the resulting dispatcher passed to
    :class:`~trilobite.radiation.arnett.Channel` directly. A plain Python function raises a ``TypeError`` when the
    engine is built.

----

API Reference
-------------

.. autosummary::
    :nosignatures:

    ~trilobite.radiation.arnett.ArnettEngine
    ~trilobite.radiation.arnett.NickelCobaltArnettEngine
    ~trilobite.radiation.arnett.MagnetarArnettEngine
    ~trilobite.radiation.arnett.MagnetarNickelArnettEngine
    ~trilobite.radiation.arnett.Channel
    ~trilobite.radiation.arnett.QuadratureRule
    ~trilobite.radiation.arnett.DiffusionClock
    ~trilobite.radiation.arnett.build_solver

The full module documentation is at :mod:`trilobite.radiation.arnett`.

----

.. footbibliography::
