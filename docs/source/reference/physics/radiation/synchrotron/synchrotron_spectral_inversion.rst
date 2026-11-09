.. _synchrotron_spectral_inversion:

=====================================
Spectral Inversion
=====================================

.. seealso::

    :ref:`synch_sed_theory` for the SED formalism and notation used throughout this page, and
    :ref:`synchrotron_seds` for the SED classes and the closures they use.

The :mod:`trilobite.radiation.synchrotron.spectral_inversion` module provides **pen-and-paper inversions of
synchrotron spectra**. Given a small number of observed spectral quantities, typically the epoch :math:`t`, a break
frequency :math:`\nu_{\rm brk}` and the flux :math:`F_{\rm brk}` at that break, each inversion returns the physical
parameters of a one-zone source, such as its magnetic field strength :math:`B` and emitting radius :math:`R`, without
building an SED model or running a fit.

These functions are intended as an **easy analysis tool which sits outside of the SED generators** in
:mod:`trilobite.radiation.synchrotron.SEDs`. The SED classes derive a spectrum from physical parameters, and may be
inverted through their own closures with the spectral regime detected for you. The inversions here instead take the
opposite view:

- they are **closed-form** (or, in one case, a short and robust numerical solve),
- they **assume the regime** they are named after rather than detecting it, and
- they return **diagnostics** so that the assumed regime can be checked after the fact.

The module is organized into *families* of inversions, each tied to a particular derivation in the literature and
sharing a common set of conventions (see below). New families can be added without changing the others.

.. currentmodule:: trilobite.radiation.synchrotron.spectral_inversion

.. contents::
   :local:
   :depth: 2

----

Overview
--------

Every inversion is a public, unit-aware function which accepts :class:`~astropy.units.Quantity` objects (or bare
floats) and returns a dictionary of results. The simplest use takes the best-fit break of a broken power law and
returns the magnetic field and radius:

.. code-block:: python

    import astropy.units as u
    from trilobite.radiation.synchrotron.spectral_inversion import invert_ssa_Diggins26

    out = invert_ssa_Diggins26(
        nu_brk=5 * u.GHz,
        F_brk=8 * u.mJy,
        distance=35 * u.Mpc,
        p=3.0,
        epsilon_E=0.1,
        epsilon_B=0.1,
    )
    print(out["B"], out["R"])

The remainder of this section describes the conventions shared by every inversion and how to choose between them.

Conventions
^^^^^^^^^^^

**Break frequency and flux.** :math:`\nu_{\rm brk}` is the **best-fit break frequency** of the spectrum. For a
self-absorbed (SSA) break it is identified with the SSA frequency :math:`\nu_a`, which is accurate to a fraction of a
percent across the relevant parameter space. :math:`F_{\rm brk}` is the **asymptotic intercept flux**: the flux
density at :math:`\nu_{\rm brk}` of the optically thick asymptote of the fit, :math:`F_{\rm asym}`.

.. important::

    The user must supply the correct asymptotic flux. If you fit a traditional (sharp) broken power law, the fitted
    normalization at the break is exactly this quantity. If you fit a smoothed broken power law, the flux at the
    break differs from the asymptotic intercept by a factor that depends on the smoothing, and the asymptotic flux
    must be extracted from the fit before it is passed to these functions.

**Local, non-relativistic sources.** The inversions assume a source in the local universe, so they take a single
``distance`` argument and do **not** apply a cosmological redshift correction. They also assume a non-relativistic
outflow, so no Doppler or time-dilation correction is made. Unless you are working with relativistic sources such as
gamma-ray bursts, this is perfectly serviceable. For relativistic sources these inversions should not be used.

**Pitch angle.** Two treatments are supported, matching the rest of Trilobite. By default a fixed pitch angle
``alpha`` is used (:math:`\pi/2` unless specified). Setting ``pitch_average=True`` instead selects the isotropic
coefficients :math:`c_5` and :math:`c_6`, and, in the cooled inversions, the angle-averaged synchrotron loss rate.

**Units.** Inputs may be :class:`~astropy.units.Quantity` objects or bare floats. Bare floats are interpreted in CGS
(Hz, :math:`\mathrm{erg\,cm^{-2}\,s^{-1}\,Hz^{-1}}`, cm, s and :math:`\mathrm{erg\,s^{-1}}`), with the exception of
:func:`invert_ssa_DeMarchi22`, which keeps the units of its original implementation (GHz, Jy and Mpc). Because the
electron normalization uses the first moment of the distribution, ``p > 2`` is required unless ``gamma_max`` is
finite.

**Broadcasting.** Every observable (``nu_brk``, ``F_brk``, ``distance``, ``t``, ``L``) and every microphysical
parameter (``p``, ``epsilon_E``, ``epsilon_B``, ``f_A``, ``f_V``, ``alpha``, ``gamma_min``, ``gamma_max``) may be a
scalar or an array, including a :class:`~astropy.units.Quantity` array. They are broadcast against one another
following the usual numpy rules, and every array in a returned dictionary has the broadcast shape. A column of break
frequencies and a row of electron indices therefore give a grid of solutions, and one luminosity per observation is
simply an array alongside ``nu_brk`` and ``F_brk``. This makes the inversions well suited to propagating posterior
draws.

**Return values.** Each function returns a dictionary. All inversions return the magnetic field ``"B"`` and radius
``"R"``; the cooled inversions additionally return the cooling Lorentz factor and a set of regime diagnostics (see
:ref:`spectral_inversion_regime`).

Picking an Inversion
^^^^^^^^^^^^^^^^^^^^

The right inversion depends on what you believe about the electrons at the break, and on what else you know about the
source.

.. list-table::
    :header-rows: 1
    :widths: 38 34 28

    * - If the electrons at the break are...
      - Use...
      - In addition to the break, you need...
    * - **uncooled** (the cooling break lies above the SSA break)
      - :func:`invert_ssa_Diggins26`
      - nothing
    * - **cooled by synchrotron losses**, with the cooling time set by the source age
      - :func:`invert_ssa_synchrotron_cooled_Diggins26`
      - the epoch ``t``
    * - **cooled by inverse-Compton losses** on a known seed radiation field
      - :func:`invert_ssa_ic_cooled_Diggins26`
      - ``t`` and the luminosity ``L``
    * - **cooled by both** synchrotron and inverse-Compton losses
      - :func:`invert_ssa_combined_cooled_Diggins26`
      - ``t`` and ``L``
    * - **not known** to be cooled or uncooled
      - :func:`invert_ssa_with_cooling_Diggins26`
      - ``t`` and ``L`` (``L=0`` if there is no seed field)
    * - uncooled, with a fixed pitch angle, for comparison with the literature
      - :func:`invert_ssa_DeMarchi22`
      - nothing

If you can confidently say which regime applies, use the inversion named after it. If you cannot, use
:func:`invert_ssa_with_cooling_Diggins26`, which evaluates the cooled and uncooled solutions and keeps the one which
is self-consistent.

----

The Diggins et al. (2026) Inversions
-------------------------------------

This family inverts an SSA break for a one-zone source with a power-law electron distribution whose normalization is
set by equipartition with the magnetic field. The functions are labeled ``_Diggins26`` and are derived in the appendix
of Diggins et al. (2026). They differ in how the electrons at the break are treated:

- :func:`invert_ssa_Diggins26` assumes **uncooled** electrons,
- :func:`invert_ssa_synchrotron_cooled_Diggins26`, :func:`invert_ssa_ic_cooled_Diggins26` and
  :func:`invert_ssa_combined_cooled_Diggins26` assume **cooled** electrons, with the cooling Lorentz factor set
  self-consistently by the source age, and
- :func:`invert_ssa_with_cooling_Diggins26` **chooses between** the cooled and uncooled solutions.

All of them share the same underlying forward model, summarized below.

.. dropdown:: Forward model and reduced notation
    :icon: chevron-down

    For an electron distribution :math:`N(\gamma) = N_0\gamma^{-q}`, the SSA frequency and the optically thick flux
    of a homogeneous sphere of radius :math:`R` at distance :math:`D` are

    .. math::

        \nu_a = 2c_1\,(m_ec^2)^{\frac{2(q-1)}{q+4}}\left[c_6(q)\,f_V^{1/3}R\right]^{\frac{2}{q+4}}
                N_0^{\frac{2}{q+4}}\,(B\sin\alpha)^{\frac{q+2}{q+4}},

    .. math::

        F_{\rm asym} = \frac{c_5(q)}{c_6(q)}\,\frac{\pi f_A R^2}{D^2}\,(B\sin\alpha)^{-1/2}
                       \left(\frac{\nu_a}{2c_1}\right)^{5/2}.

    The electron normalization is fixed by equipartition with the *uncooled* energy budget,

    .. math::

        N_0 = A_N B^2, \qquad
        A_N = \frac{1}{8\pi m_ec^2}\,\frac{\epsilon_E}{\epsilon_B}\,\frac{1}{M_\gamma^{(1)}},

    where :math:`M_\gamma^{(1)}` is the first moment of the Lorentz-factor distribution for the index :math:`p`.
    Collecting the dependence on :math:`B` and :math:`R`,

    .. math::

        \nu_a = A_\nu\,A_N^{\frac{2}{q+4}}\,R^{\frac{2}{q+4}}B^{\frac{q+6}{q+4}},
        \qquad
        F_{\rm asym} = A_F\,A_N^{\frac{5}{q+4}}\,D^{-2}\,(BR)^{\frac{2q+13}{q+4}},

    with the reduced constants

    .. math::

        A_\nu(q) = 2c_1\,(m_ec^2)^{\frac{2(q-1)}{q+4}}\left[c_6(q)\,f_V^{1/3}\right]^{\frac{2}{q+4}}
                   \sin^{\frac{q+2}{q+4}}\alpha,
        \qquad
        A_F(q) = \frac{c_5(q)}{c_6(q)}\,\pi f_A\,\sin^{-1/2}\alpha\left(\frac{A_\nu(q)}{2c_1}\right)^{5/2}.

Uncooled Electrons
^^^^^^^^^^^^^^^^^^

:func:`invert_ssa_Diggins26` is the default choice when the electrons at the break have not had time to cool. With no
cooling, the flux fixes the product :math:`BR` and the break frequency then fixes :math:`B`, so the solution is a
single closed-form expression. It needs nothing beyond the break, the distance and the microphysics.

.. code-block:: python

    out = invert_ssa_Diggins26(nu_brk, F_brk, distance, p=3.0, epsilon_E=0.1, epsilon_B=0.1)
    out["B"], out["R"]

.. dropdown:: Derivation
    :icon: chevron-down

    For uncooled electrons :math:`q = p`, and the forward model above is inverted directly,

    .. math::

        B = \frac{\nu_{\rm brk}}{A_\nu}\,A_F^{\frac{2}{2p+13}}\,A_N^{-\frac{4}{2p+13}}\,
            \left(F_{\rm brk}D^2\right)^{-\frac{2}{2p+13}},
        \qquad
        R = \frac{A_\nu}{\nu_{\rm brk}}\,A_F^{-\frac{p+6}{2p+13}}\,A_N^{-\frac{1}{2p+13}}\,
            \left(F_{\rm brk}D^2\right)^{\frac{p+6}{2p+13}}.

Cooled Electrons
^^^^^^^^^^^^^^^^

If the electrons at the break have cooled, the normalization of the emitting segment is reduced, and the inferred
:math:`B` and :math:`R` change. Three inversions are provided, which differ only in what sets the cooling Lorentz
factor :math:`\gamma_c`:

- :func:`invert_ssa_synchrotron_cooled_Diggins26`: synchrotron losses, :math:`\gamma_c = A_c/(B^2 t)`,
- :func:`invert_ssa_ic_cooled_Diggins26`: inverse-Compton losses on the radiation field of a central source of
  luminosity :math:`L`, :math:`\gamma_c = A_{\rm IC}R^2/(Lt)`, and
- :func:`invert_ssa_combined_cooled_Diggins26`: both, with the loss rates adding.

All three require the epoch ``t``, which is treated as the effective cooling time of the source. The IC-cooled and
combined inversions additionally require the seed luminosity ``L`` at that epoch. They are valid when the cooling
break lies well below the SSA break, :math:`\nu_c \ll \nu_a`, in the slow-cooling regime :math:`\gamma_c > \gamma_m`;
whether this holds for your solution is reported through the diagnostics described in
:ref:`spectral_inversion_regime`.

.. code-block:: python

    out = invert_ssa_synchrotron_cooled_Diggins26(
        nu_brk, F_brk, distance, t, p=3.0, epsilon_E=0.1, epsilon_B=0.1
    )
    out["B"], out["R"], out["gamma_c"], out["nu_c_over_nu_a"]

.. important::

    For isotropic pitch angles (``pitch_average=True``) the synchrotron cooling constant is
    :math:`9\pi m_e c/\sigma_T`, from the angle-averaged loss rate :math:`\langle\sin^2\alpha\rangle = 2/3`. This is
    **not** the :math:`\sin\alpha\to1` limit of the fixed-angle constant, :math:`6\pi m_e c/(\sigma_T\sin^2\alpha)`.
    It assumes that pitch-angle scattering is fast compared with cooling.

.. dropdown:: Cooled forward model
    :icon: chevron-down

    In the slow-cooling regime :math:`\gamma_m < \gamma_c`, continuous injection and radiative losses give

    .. math::

        N(\gamma) = N_0
        \begin{cases}
        \gamma^{-p}, & \gamma_m \le \gamma < \gamma_c,\\
        \gamma_c\,\gamma^{-(p+1)}, & \gamma_c \le \gamma \le \gamma_{\rm max}.
        \end{cases}

    For :math:`\nu_c \ll \nu_a \ll \nu_{\rm max}`, the emission and absorption near :math:`\nu_a` come from the
    cooled segment, so the forward model above applies with

    .. math::

        p \to p+1, \qquad N_0 \to N_0\gamma_c .

    The normalization :math:`N_0 = A_NB^2` is still set by the uncooled energy budget, which overestimates the
    electron energy by a factor :math:`1/(1-\delta_c)`, with
    :math:`\delta_c = (\gamma_c/\gamma_m)^{2-p}/(p-1)` for :math:`\gamma_{\rm max}\to\infty` and :math:`p>2`. This is
    returned as ``delta_c``. We write :math:`A_\nu^{(c)} = A_\nu(p+1)` and :math:`A_F^{(c)} = A_F(p+1)` below.

.. dropdown:: Derivation: synchrotron cooling
    :icon: chevron-down

    With :math:`\gamma_c = A_c/(B^2 t)`, the factor :math:`B^2` in :math:`N_0` cancels the :math:`B^{-2}` in
    :math:`\gamma_c`, so :math:`N_0\gamma_c = A_NA_c/t \equiv \Lambda` is independent of :math:`B` and :math:`R`.
    The system is then a pair of power laws,

    .. math::

        B = \left(\frac{\nu_{\rm brk}}{A_\nu^{(c)}}\right)^{\frac{2p+15}{2p+7}}
            \left(\frac{F_{\rm brk}D^2}{A_F^{(c)}}\right)^{-\frac{2}{2p+7}}\Lambda^{-\frac{4}{2p+7}},
        \qquad
        R = \left(\frac{F_{\rm brk}D^2}{A_F^{(c)}}\right)^{\frac{p+3}{2p+7}}
            \left(\frac{\nu_{\rm brk}}{A_\nu^{(c)}}\right)^{-\frac{2p+5}{2p+7}}\Lambda^{-\frac{1}{2p+7}},

    with the synchrotron cooling constant

    .. math::

        A_c =
        \begin{cases}
        \dfrac{6\pi m_e c}{\sigma_T\sin^2\alpha} & \text{fixed pitch angle},\\[2ex]
        \dfrac{9\pi m_e c}{\sigma_T} & \text{pitch averaged}.
        \end{cases}

.. dropdown:: Derivation: inverse-Compton cooling
    :icon: chevron-down

    In the Thomson regime, an electron in the radiation field of a central source of luminosity :math:`L` has
    :math:`\gamma_c = A_{\rm IC}R^2/(Lt)` with :math:`A_{\rm IC} = 3\pi m_ec^2/\sigma_T`. This loss rate is
    independent of the pitch angle. Now :math:`\gamma_c\propto R^2`, so there is no cancellation against the
    normalization,

    .. math::

        N_0\gamma_c = \Lambda_{\rm IC}\,(BR)^2, \qquad \Lambda_{\rm IC} = \frac{A_NA_{\rm IC}}{L\,t},

    but the system remains log-linear and can be solved in closed form,

    .. math::

        B = \left(\frac{\nu_{\rm brk}}{A_\nu^{(c)}}\right)^{\frac{2p+25}{2p+17}}
            \left(\frac{F_{\rm brk}D^2}{A_F^{(c)}}\right)^{-\frac{6}{2p+17}}\Lambda_{\rm IC}^{-\frac{4}{2p+17}},
        \qquad
        R = \left(\frac{F_{\rm brk}D^2}{A_F^{(c)}}\right)^{\frac{p+7}{2p+17}}
            \left(\frac{\nu_{\rm brk}}{A_\nu^{(c)}}\right)^{-\frac{2p+15}{2p+17}}\Lambda_{\rm IC}^{-\frac{1}{2p+17}}.

.. dropdown:: Derivation: combined cooling
    :icon: chevron-down

    With both loss channels the rates add,

    .. math::

        \gamma_c = \frac{A_c}{B^2t}\,\frac{1}{1+Y},
        \qquad
        Y \equiv \frac{A_cL}{A_{\rm IC}R^2B^2},

    and :math:`\gamma_c` is a sum of two power laws, so there is no closed form. The optically thick asymptote does
    not involve :math:`\gamma_c`, and gives :math:`R(B)\propto B^{1/4}`. Substituting this and
    :math:`\gamma_c(B, R(B))` into the SSA condition leaves a single equation :math:`g(\ln B) = 0`, whose slope

    .. math::

        \frac{dg}{d\ln B} = \frac{2p+17}{4} - \frac{5}{2}\,\frac{1}{1+Y}
        \in \left[\frac{2p+7}{4},\ \frac{2p+17}{4}\right]

    is bounded and strictly positive, so the root is unique. Since :math:`Y\propto B^{-5/2}` along :math:`R(B)`,
    :math:`g` is also concave, and Newton iterations started below the root approach it monotonically.

    Adding a second loss channel can only lower :math:`\gamma_c` at fixed :math:`(B, R)`, so each single-channel
    closed form is a lower bound on :math:`B`. Starting from the larger of the two, the minimum slope gives an upper
    bound, and the root is bracketed analytically. The bracket guards a Newton iteration which converges in a few
    steps and is vectorized over inputs. A luminosity of zero removes the IC channel and recovers the
    synchrotron-cooled result.

.. _spectral_inversion_regime:

Checking the Assumed Regime
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The cooled inversions assume :math:`\nu_c \ll \nu_a` and :math:`\gamma_c > \gamma_m`, where the characteristic
frequency of the cooling electrons is :math:`\nu_c = c_{1,\gamma}B\sin\alpha\,\gamma_c^2`. Because the regime is
assumed rather than detected, each cooled inversion returns diagnostics for checking it:

- ``nu_c_over_nu_a``: the ratio :math:`\nu_c/\nu_a`, which should be much smaller than one,
- ``consistent``: whether :math:`\nu_c < \nu_a`,
- ``accurate``: whether :math:`\nu_c < 0.1\,\nu_a`, for which the cooled asymptotic forms are good to better than
  0.1%,
- ``slow_cooling``: whether :math:`\gamma_c > \gamma_{\min}`,
- ``delta_c``: the fractional energy error of the uncooled normalization (NaN unless :math:`p>2` and
  ``gamma_max`` is infinite), and
- for the IC-cooled and combined inversions, the Compton parameter ``Y`` and ``ic_dominated`` (:math:`Y>1`).

A warning is also raised if :math:`\nu_c \ge \nu_a` for any input, since the results are then not self-consistent.

Choosing Between Cooled and Uncooled
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Often you do not know in advance whether the electrons at the break are cooled.
:func:`invert_ssa_with_cooling_Diggins26` evaluates both solutions and keeps the one which is consistent with its own
assumption. The luminosity ``L`` is treated as data, with one value per observation, alongside ``t``, ``nu_brk`` and
``F_brk``; use ``L=0`` if there is no seed radiation field.

.. code-block:: python

    out = invert_ssa_with_cooling_Diggins26(
        nu_brk, F_brk, distance, t, L=1e42 * u.erg / u.s, p=3.0, epsilon_E=0.1, epsilon_B=0.1
    )
    out["branch"], out["B"], out["R"]

For each observation, the function proceeds as follows:

1. The **uncooled** solution is evaluated, and the cooling break it implies is computed with *both* loss channels.
   It is accepted if :math:`\nu_c > \nu_a`.
2. The **cooled** (combined) solution is evaluated. It is accepted if :math:`\nu_c < \nu_a` and
   :math:`\gamma_c > \gamma_{\min}`.
3. If exactly one is accepted, that branch is returned and labeled ``"uncooled"`` or ``"cooled"``.

Observations for which neither or both are accepted cannot be assigned to a branch, and are labeled:

- ``"fast_cooling"``: the cooled ordering holds but :math:`\gamma_c \le \gamma_{\min}`, a regime that the
  slow-cooling distribution does not describe, or
- ``"ambiguous"``: both or neither branch is consistent, typically because :math:`\nu_c\approx\nu_a`, where the
  asymptotic forms fail.

These observations return NaN for :math:`B`, :math:`R` and :math:`\gamma_c`. The full solutions of both branches are
returned under ``"cooled"`` and ``"uncooled"`` for inspection.

----

The De Marchi et al. (2022) Inversion
-------------------------------------

:func:`invert_ssa_DeMarchi22` implements the analytic SSA inversion of :footcite:t:`demarchiRadioAnalysisSN2004C2022`
for uncooled electrons. It uses the same ``nu_brk`` and ``F_brk`` as the other inversions and is useful for comparing
directly with results in the literature.

.. warning::

    The De Marchi inversion does **not** pitch-angle average. It uses the fixed pitch angle ``theta`` throughout. To
    use the isotropic pitch-angle coefficients, use :func:`invert_ssa_Diggins26` with ``pitch_average=True``.

Its interface follows the original implementation more closely than the rest of the module: bare floats are in GHz,
Jy and Mpc, the filling factor is called ``f``, and the singular case :math:`p = 2` is not supported.

.. code-block:: python

    out = invert_ssa_DeMarchi22(5 * u.GHz, 8 * u.mJy, 35 * u.Mpc, p=3.0, f=0.5)
    out["B"], out["R"]

----

API Reference
-------------

.. autosummary::
    :nosignatures:

    invert_ssa_Diggins26
    invert_ssa_synchrotron_cooled_Diggins26
    invert_ssa_ic_cooled_Diggins26
    invert_ssa_combined_cooled_Diggins26
    invert_ssa_with_cooling_Diggins26
    invert_ssa_DeMarchi22

.. footbibliography::
