.. _synchrotron_seds:

=========================================
Synchrotron Spectral Energy Distributions
=========================================

Synchrotron radiation is a core emission mechanism in Triceratops and underpins
the modeling of a wide range of transient astrophysical sources, including
gamma-ray bursts, supernovae, tidal disruption events, and fast radio bursts.

The **synchrotron spectral energy distribution (SED)** depends on the ordering of
several characteristic frequencies—most notably the injection frequency
:math:`\nu_m`, cooling frequency :math:`\nu_c`, self-absorption frequency
:math:`\nu_a`, and maximum synchrotron frequency :math:`\nu_{\max}`.
The theoretical basis for these spectra is described in detail in
:ref:`synchrotron_theory`, with a comprehensive catalog of spectral regimes and
derivations provided in :ref:`synch_sed_theory`.

This document describes **how Triceratops implements synchrotron SEDs**, how the
various abstractions fit together, and how to use them effectively in practice.

.. important::

    This guide assumes familiarity with the physics of synchrotron emission.
    Users are strongly encouraged to read :ref:`synchrotron_theory` and
    :ref:`synch_sed_theory` before working with the SED API.

.. contents::
    :local:
    :depth: 2

Overview
--------

At its core, Triceratops treats synchrotron SEDs as **log-space compositions of
scale-free spectral segments**. Each SED is built by:

1. Identifying the **global spectral regime** from the ordering of break frequencies,
2. Selecting the corresponding **asymptotic spectral slopes**,
3. Connecting those slopes using **smoothed broken power laws (SBPLs)**,
4. Applying a physically motivated **normalization**.

This approach, referred to internally as **log-space SED surgery**, ensures that
all spectra are:

- Numerically stable across many decades in frequency,
- Continuous and differentiable,
- Faithful to analytic synchrotron asymptotes,
- Easy to extend to new physical regimes.

To support both flexibility and usability, Triceratops exposes **two complementary
interfaces** to synchrotron SEDs:

- A **high-level, object-oriented interface** for most users,
- A **low-level functional interface** for advanced use cases and custom modeling.

All SEDs reside in :mod:`radiation.synchrotron.SEDs`, which is
also accessible as the ``SEDs`` submodule of :mod:`radiation.synchrotron`.
The top-level public classes are re-exported from
:mod:`radiation.synchrotron` for convenience.

----

.. _sed_class_hierarchy:

Class Hierarchy
---------------

The synchrotron SED system is organized around a two-level abstract class hierarchy
that separates **interface** from **regime dispatch logic**:

.. code-block:: text

    SynchrotronSED  (ABC)
    │   The minimal interface every SED must satisfy.
    │   Methods: sed(), _log_opt_sed(), from_physics_to_params(),
    │            from_params_to_physics(), _opt_from_physics_to_params(),
    │            _opt_from_params_to_physics()
    │
    ├── PowerLaw_SynchrotronSED
    │       Optically thin, uncooled power-law SED (single regime).
    │
    ├── SSA_SED_PowerLaw
    │       Phenomenological SSA broken power-law SED (De Marchi closure).
    │
    └── MultiSpectrumSynchrotronSED  (ABC)
            Adds automatic regime selection and dispatch.
            Methods: determine_sed_regime(), _compute_sed_regime(),
                     _log_opt_sed_from_regime()
            Attribute: SPECTRUM_FUNCTIONS
            │
            ├── PowerLaw_Cooling_SynchrotronSED
            │       Optically thin SED with radiative cooling (2 regimes).
            │
            ├── PowerLaw_SSA_SynchrotronSED
            │       Non-cooling SED with SSA (2 regimes).
            │
            └── PowerLaw_Cooling_SSA_SynchrotronSED
                    Full SED: cooling + SSA (up to 8 regimes).

The key design principle is that **SED objects are stateless**: no physical parameters
are stored at instantiation. All parameters are supplied at call time, making objects
safe to reuse across parameter sweeps or inference loops.

----

.. _sed_high_level:

The High-Level Interface
------------------------

Most users should interact with synchrotron spectra through the **SED classes**
defined in :mod:`radiation.synchrotron.SEDs`. These classes provide a
**structured, declarative interface** for evaluating physically consistent
synchrotron spectral energy distributions without requiring users to manually
assemble broken power laws or reason about regime selection.

Each SED class represents a **specific physical model** (e.g. cooling only,
SSA only, cooling + SSA) and is responsible for orchestrating all steps required
to compute a spectrum from a minimal set of inputs.

.. rubric:: Public SED Classes

.. currentmodule:: radiation.synchrotron.SEDs.one_zone

.. autosummary::
    :toctree: ../../../../_as_gen
    :nosignatures:

    SynchrotronSED
    MultiSpectrumSynchrotronSED
    PowerLaw_SynchrotronSED
    PowerLaw_Cooling_SynchrotronSED
    PowerLaw_SSA_SynchrotronSED
    PowerLaw_Cooling_SSA_SynchrotronSED
    SSA_SED_PowerLaw


Instantiating SED Objects
^^^^^^^^^^^^^^^^^^^^^^^^^^

SED objects in Triceratops are intentionally lightweight. In most cases,
instantiating an SED class requires **no physical parameters at all**:

.. code-block:: python

    from triceratops.radiation.synchrotron import PowerLaw_Cooling_SynchrotronSED

    sed = PowerLaw_Cooling_SynchrotronSED()

This reflects an important design principle: SED instances do **not** store
model parameters. Break frequencies, peak fluxes, and other physical quantities
are supplied at *evaluation time*, not at initialization.

This makes SED objects safe to reuse in parameter sweeps, inference loops, and
population modeling without worrying about hidden state.

In a small number of cases, an SED class may accept configuration flags at
initialization (for example, selecting between alternative numerical kernels or
enabling optional corrections). When present, such flags control *how* the SED
is evaluated, not *what physical system* is being modeled.


.. _sed_evaluation:

SED Evaluation
^^^^^^^^^^^^^^

.. currentmodule:: radiation.synchrotron.SEDs.one_zone

The primary interface for evaluating a synchrotron spectrum is the
:meth:`SynchrotronSED.sed` method. Given an array of frequencies and the required physical or
phenomenological parameters, this method returns the corresponding flux
density:

.. code-block:: python

    import numpy as np
    from astropy import units as u
    from triceratops.radiation.synchrotron.SEDs import PowerLaw_Cooling_SynchrotronSED

    sed = PowerLaw_Cooling_SynchrotronSED()
    nu = np.logspace(9, 18, 500) * u.Hz

    Fnu = sed.sed(
        nu,
        F_norm=1e-26 * u.erg / (u.s * u.cm**2 * u.Hz),
        nu_m=1e12 * u.Hz,
        nu_c=1e15 * u.Hz,
        nu_max=1e19 * u.Hz,
        p=2.5,
    )

Internally, the SED class performs the following steps:

1. Validate and coerce input units,
2. Convert inputs to logarithmic CGS form,
3. Determine the global spectral regime,
4. Compute any derived break frequencies (e.g. :math:`\nu_a`),
5. Dispatch to a regime-specific optimized kernel,
6. Apply the overall normalization.

For convenience, all SED classes also support **direct call syntax**, making the
following equivalent:

.. code-block:: python

    Fnu = sed(nu, F_norm=..., nu_m=..., nu_c=..., nu_max=..., p=2.5)

This allows SED objects to be used interchangeably with ordinary callable
functions in modeling pipelines.

A typical synchrotron spectrum produced by one of these classes looks like:

.. plot::
   :include-source: True

    import numpy as np
    from astropy import units as u
    from triceratops.radiation.synchrotron.SEDs import PowerLaw_Cooling_SynchrotronSED

    sed = PowerLaw_Cooling_SynchrotronSED()
    nu = np.logspace(9, 18, 500) * u.Hz

    Fnu = sed.sed(
        nu,
        F_norm=1e-26 * u.erg / (u.s * u.cm ** 2 * u.Hz),
        nu_m=1e12 * u.Hz,
        nu_c=1e15 * u.Hz,
        nu_max=1e19 * u.Hz,
        p=2.5,
    )

    import matplotlib.pyplot as plt

    plt.loglog(nu, Fnu)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel(r"$F_\nu$ [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
    plt.tight_layout()
    plt.show()


.. _sed_normalization:

SED Normalization
^^^^^^^^^^^^^^^^^

All synchrotron SEDs in Triceratops are normalized using a **normalization flux
density** :math:`F_{\nu,\mathrm{norm}}`, defined as the flux density of the
**dominant optically thin emitting electron population** at a reference
frequency.

.. important::

    :math:`F_{\nu,\mathrm{norm}}` is **not necessarily the peak of the observed
    spectrum**. In the presence of synchrotron self-absorption, the actual peak
    flux (at :math:`\nu_a`) will differ from :math:`F_{\nu,\mathrm{norm}}`. The
    normalization is always defined in the optically thin limit, regardless of
    absorption.

    This convention ensures that the normalization parameter has a unique,
    physically transparent meaning across *all* SED types and all spectral
    regimes. When you obtain ``F_norm`` from :meth:`from_physics_to_params`, it
    is already expressed in this convention.

The **peak flux** :math:`F_{\nu,\mathrm{peak}}` (i.e. the maximum of the
observed SED) is computed internally from :math:`F_{\nu,\mathrm{norm}}` by the
normalization routines. Methods that return ``F_peak`` in their output
dictionaries provide this translated value.

The mapping between the two depends on the spectral regime:

- **Optically thin, uncooled** (:class:`PowerLaw_SynchrotronSED`):
  :math:`F_\mathrm{peak} = F_\mathrm{norm}` — the spectrum peaks at
  :math:`\nu_m`.
- **Cooling SEDs** (:class:`PowerLaw_Cooling_SynchrotronSED`): the peak shifts
  to the lower of :math:`\nu_m` and :math:`\nu_c` depending on regime.
- **SSA SEDs**: the peak is set by the SSA turnover :math:`\nu_a`, which is
  derived internally. :math:`F_\mathrm{norm} \neq F_\mathrm{peak}` in general.

For a detailed derivation of these mappings see :ref:`synch_sed_theory`.


.. _sed_regime_introspection:

Regime Determination
^^^^^^^^^^^^^^^^^^^^

For :class:`MultiSpectrumSynchrotronSED` subclasses, the **spectral regime** is
determined once, globally, before evaluating the SED across the frequency grid.
Users can query the selected regime explicitly via the public
:meth:`~MultiSpectrumSynchrotronSED.determine_sed_regime` method:

.. code-block:: python

    from triceratops.radiation.synchrotron import PowerLaw_Cooling_SSA_SynchrotronSED
    import astropy.units as u
    import numpy as np

    sed = PowerLaw_Cooling_SSA_SynchrotronSED()

    regime = sed.determine_sed_regime(
        F_norm=1e-26 * u.erg / (u.s * u.cm**2 * u.Hz),
        nu_m=1e12 * u.Hz,
        nu_c=1e15 * u.Hz,
        nu_max=1e19 * u.Hz,
        omega=4 * np.pi,
        gamma_m=300,
        p=2.5,
    )
    print(regime)  # integer regime index, see synch_sed_theory for definitions

This is useful for diagnosing which branch of the SED is being evaluated,
particularly when debugging or verifying physical parameter ranges.

The returned regime identifier is an integer corresponding to the spectral
ordering defined in :ref:`synch_sed_theory`. The mapping from regime index to
frequency ordering is documented in the class-level ``SPECTRUM_FUNCTIONS``
attribute.


----

.. _sed_available_classes:

Available SED Classes
---------------------

Triceratops provides five synchrotron SED classes, each corresponding to a
different combination of physical effects. All classes share the same
conceptual interface and normalization strategy; they differ only in which
physical processes are included internally.

For a detailed derivation of the spectral slopes, break orderings, and regime
definitions referenced below, see :ref:`synch_sed_theory`.


Power-Law SED (No Cooling, No SSA)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`PowerLaw_SynchrotronSED` implements the canonical optically thin,
uncooled synchrotron SED for a power-law electron population. It contains a
single physical spectral break at the injection frequency :math:`\nu_m` and an
optional high-frequency exponential cutoff at :math:`\nu_{\max}`.

The spectral structure is:

- :math:`F_\nu \propto \nu^{1/3}` for :math:`\nu < \nu_m`,
- :math:`F_\nu \propto \nu^{-(p-1)/2}` for :math:`\nu_m \leq \nu < \nu_{\max}`,
- Exponential suppression above :math:`\nu_{\max}`.

**Required ``sed()`` parameters:**
``F_norm``, ``nu_m``

**Optional ``sed()`` parameters:**
``nu_max`` (default: no cutoff), ``p`` (default: 2.5), ``s`` (default: −0.5)

.. dropdown:: Example

    .. plot::
        :include-source: True

        import numpy as np
        from astropy import units as u
        import matplotlib.pyplot as plt
        from triceratops.radiation.synchrotron import PowerLaw_SynchrotronSED

        sed = PowerLaw_SynchrotronSED()
        nu = np.logspace(8, 18, 500) * u.Hz

        Fnu = sed.sed(
            nu,
            F_norm=1e-26 * u.erg / (u.s * u.cm**2 * u.Hz),
            nu_m=1e12 * u.Hz,
            nu_max=1e18 * u.Hz,
            p=2.4,
        )

        plt.loglog(nu, Fnu)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel(r"$F_\nu$")
        plt.title("PowerLaw_SynchrotronSED")
        plt.tight_layout()
        plt.show()


Power-Law + Cooling SED
^^^^^^^^^^^^^^^^^^^^^^^^

:class:`PowerLaw_Cooling_SynchrotronSED` adds radiative cooling to the optically
thin spectrum. Depending on the ordering of :math:`\nu_m`, :math:`\nu_c`, and
:math:`\nu_{\max}`, the spectrum automatically transitions between:

- **Fast cooling** (:math:`\nu_c < \nu_m`): The spectrum follows :math:`\nu^{1/3}
  \to \nu^{-1/2} \to \nu^{-p/2}`.
- **Slow cooling** (:math:`\nu_m < \nu_c < \nu_{\max}`): :math:`\nu^{1/3} \to
  \nu^{-(p-1)/2} \to \nu^{-p/2}`.
- **Effectively non-cooling** (:math:`\nu_c > \nu_{\max}`): Equivalent to the
  uncooled power-law case with an exponential cutoff at :math:`\nu_{\max}`.

The correct regime is selected internally based solely on the supplied break
frequencies.

**Required ``sed()`` parameters:**
``F_norm``, ``nu_m``, ``nu_c``

**Optional ``sed()`` parameters:**
``nu_max`` (default: no cutoff), ``p``, ``s``, ``gamma_m``

.. dropdown:: Example

    .. plot::
        :include-source: True

        from triceratops.radiation.synchrotron import PowerLaw_Cooling_SynchrotronSED
        import numpy as np
        import matplotlib.pyplot as plt
        from astropy import units as u

        sed = PowerLaw_Cooling_SynchrotronSED()

        nu = np.logspace(8, 20, 500) * u.Hz

        Fnu = sed.sed(
            nu,
            F_norm=1e-26 * u.erg / (u.s * u.cm**2 * u.Hz),
            nu_m=1e12 * u.Hz,
            nu_c=1e15 * u.Hz,
            nu_max=1e19 * u.Hz,
            p=2.5,
        )

        plt.loglog(nu, Fnu)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel(r"$F_\nu$")
        plt.title("PowerLaw_Cooling_SynchrotronSED (slow cooling)")
        plt.tight_layout()
        plt.show()

For the explicit spectral slopes corresponding to each cooling regime, see
:ref:`synch_sed_theory`.


Power-Law + SSA SED
^^^^^^^^^^^^^^^^^^^^

.. note::

    See :ref:`synch_sed_theory` for a detailed discussion of the underlying SSA theory, and
    :ref:`stratified_absorption` for treatment of stratified absorption corrections.

:class:`PowerLaw_SSA_SynchrotronSED` includes synchrotron self-absorption but
neglects radiative cooling. This is appropriate for compact or dense emission
regions where the low-frequency spectrum is optically thick, but electrons do not
cool appreciably.

Unlike simple textbook treatments, Triceratops computes the self-absorption
frequency :math:`\nu_a` **self-consistently** from the supplied normalization and
microphysical parameters, rather than requiring it as a user input.

The spectrum is classified into two regimes:

- **SSA-1** (:math:`\nu_a < \nu_m`): The self-absorption break falls below the injection break.
- **SSA-2** (:math:`\nu_m < \nu_a`): The source is optically thick at and above the injection break.

**Required ``sed()`` parameters:**
``F_norm``, ``nu_m``, ``omega``, ``gamma_m``

**Optional ``sed()`` parameters:**
``nu_max``, ``p``, ``s``

.. dropdown:: Example

    .. plot::
        :include-source: True

        from triceratops.radiation.synchrotron import PowerLaw_SSA_SynchrotronSED
        import numpy as np
        import matplotlib.pyplot as plt
        from astropy import units as u

        sed = PowerLaw_SSA_SynchrotronSED()

        nu = np.logspace(7, 18, 500) * u.Hz

        Fnu = sed.sed(
            nu,
            F_norm=1e-26 * u.erg / (u.s * u.cm**2 * u.Hz),
            nu_m=1e11 * u.Hz,
            omega=4 * np.pi,
            gamma_m=300,
            p=2.5,
        )

        plt.loglog(nu, Fnu)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel(r"$F_\nu$")
        plt.title("PowerLaw_SSA_SynchrotronSED")
        plt.tight_layout()
        plt.show()

For the analytic scalings used to compute :math:`\nu_a`, see
:ref:`synch_sed_theory`.


Power-Law + Cooling + SSA SED
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`PowerLaw_Cooling_SSA_SynchrotronSED` is the most general synchrotron
model implemented in Triceratops. It includes:

- Radiative cooling (fast, slow, and non-cooling regimes),
- Synchrotron self-absorption with all allowed :math:`\nu_a` orderings,
- Stratified SSA corrections where applicable,
- A high-frequency exponential cutoff.

This class automatically handles all allowed orderings of
:math:`\nu_m`, :math:`\nu_c`, :math:`\nu_a`, and :math:`\nu_{\max}`, selecting
the appropriate spectral shape internally (up to 8 distinct regimes).

It is the recommended choice when modeling broadband synchrotron emission from
transient shocks, GRB afterglows, radio supernovae, or similar systems.

**Required ``sed()`` parameters:**
``F_norm``, ``nu_m``, ``nu_c``, ``omega``, ``gamma_m``

**Optional ``sed()`` parameters:**
``nu_max``, ``p``, ``s``

.. dropdown:: Example

    .. plot::
        :include-source: True

        from triceratops.radiation.synchrotron import PowerLaw_Cooling_SSA_SynchrotronSED
        import numpy as np
        import matplotlib.pyplot as plt
        from astropy import units as u

        sed = PowerLaw_Cooling_SSA_SynchrotronSED()

        nu = np.logspace(7, 20, 600) * u.Hz

        Fnu = sed.sed(
            nu,
            F_norm=1e-26 * u.erg / (u.s * u.cm**2 * u.Hz),
            nu_m=1e12 * u.Hz,
            nu_c=1e15 * u.Hz,
            nu_max=1e19 * u.Hz,
            omega=4 * np.pi,
            gamma_m=300,
            p=2.5,
        )

        plt.loglog(nu, Fnu)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel(r"$F_\nu$")
        plt.title("PowerLaw_Cooling_SSA_SynchrotronSED")
        plt.tight_layout()
        plt.show()

Because of the large number of possible spectral orderings, users are strongly
encouraged to consult :ref:`synch_sed_theory` when interpreting the resulting
spectral shapes.


Phenomenological SSA SED (De Marchi Closure)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`SSA_SED_PowerLaw` implements a **simpler, two-segment phenomenological
SSA SED** where the break frequency :math:`\nu_{\rm brk}` is supplied directly
by the user rather than derived internally. This class is appropriate when one
wishes to fit the SSA turnover directly without invoking any microphysical
closure assumptions.

The spectral structure is:

- :math:`F_\nu \propto \nu^{5/2}` for :math:`\nu < \nu_{\rm brk}` (optically thick),
- :math:`F_\nu \propto \nu^{-(p-1)/2}` for :math:`\nu > \nu_{\rm brk}` (optically thin),

connected with a smoothed broken power law.

Closure relations following the formalism of
:footcite:t:`demarchiRadioAnalysisSN2004C2022` are implemented in
:func:`~radiation.synchrotron.SEDs.one_zone_closure.invert_powerlaw_ssa_sed_demarchi`,
mapping :math:`\nu_{\rm brk}` and :math:`F_{\nu,\rm brk}` to source radius and
magnetic field strength.

**Required ``sed()`` parameters:**
``nu_brk``, ``F_nu_brk``, ``p``, ``s``

.. dropdown:: Example

    .. code-block:: python

        import numpy as np
        from astropy import units as u
        from triceratops.radiation.synchrotron.SEDs import SSA_SED_PowerLaw

        sed = SSA_SED_PowerLaw()
        nu = np.logspace(7, 15, 500) * u.Hz

        Fnu = sed.sed(
            nu,
            nu_brk=1e10 * u.Hz,
            F_nu_brk=1e-25 * u.erg / (u.s * u.cm**2 * u.Hz),
            p=2.5,
            s=0.5,
        )


----

.. _sed_closure_relations:

Closure Relations
-----------------

The closure relation system maps between the **phenomenological SED parameters**
(break frequencies, normalization fluxes) and the **underlying physical
parameters** (magnetic field strength, emitting radius, electron Lorentz
factors, energy fractions). These mappings are model-dependent and rely on
analytic closure relations from synchrotron theory, typically supplemented by
equipartition-style assumptions.

Triceratops implements closures at two levels:

1. **Object methods** on the SED classes (``from_physics_to_params`` and
   ``from_params_to_physics``), which provide unit-aware, user-facing interfaces.
2. **Standalone functions** in :mod:`~radiation.synchrotron.SEDs.one_zone_closure`,
   which expose the same logic for direct use without SED class instantiation.

.. _sed_forward_closure:

Forward Closure: Physics → SED Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The **forward closure** converts a physical description of the emission region
into phenomenological SED parameters. This is the direction used in inference
workflows, where physical parameters are sampled and translated into observable
quantities.

.. code-block:: python

    import numpy as np
    from astropy import units as u
    from triceratops.radiation.synchrotron import PowerLaw_SynchrotronSED

    sed = PowerLaw_SynchrotronSED()

    # Physical description of the emitting region
    params = sed.from_physics_to_params(
        B=0.1 * u.G,
        R=1e16 * u.cm,
        gamma_min=100.0,
        gamma_max=1e6,
        p=2.3,
        epsilon_E=0.1,
        epsilon_B=0.1,
        redshift=0.01,
        pitch_average=True,
    )
    # params contains: F_norm, F_peak, nu_m, nu_max (all with units)

    # Evaluate the SED
    nu = np.logspace(8, 18, 300) * u.Hz
    Fnu = sed.sed(nu, **params, p=2.3)

Internally, this calls :meth:`~SynchrotronSED._opt_from_physics_to_params`, which
works entirely in log-space CGS for numerical stability. The public method
handles unit coercion and cosmological distance resolution (via
:func:`~utils.cosmology.resolve_cosmological_distances`).

.. _sed_inverse_closure:

Inverse Closure: SED Parameters → Physics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The **inverse closure** infers physical source parameters (radius :math:`R`,
magnetic field :math:`B`) from observed SED peak quantities. This is the
direction used for physical interpretation after fitting phenomenological
parameters.

.. code-block:: python

    from triceratops.radiation.synchrotron import PowerLaw_SynchrotronSED
    import astropy.units as u

    sed = PowerLaw_SynchrotronSED()

    physics = sed.from_params_to_physics(
        F_peak=1e-26 * u.erg / (u.s * u.cm**2 * u.Hz),
        nu_peak=1e12 * u.Hz,
        p=2.5,
        epsilon_E=0.1,
        epsilon_B=0.1,
        f_V=0.5,
        redshift=0.01,
    )
    # physics contains: R (cm), B (G)

.. warning::

    The inverse closure provides a **default physical interpretation**, not a
    unique truth. It is only meaningful when the assumptions of the underlying
    SED model are approximately satisfied (single zone, power-law electrons, no
    cooling at the peak, no SSA at the peak). See the :class:`PowerLaw_SynchrotronSED`
    docstring for a detailed list of caveats.

.. _sed_standalone_closures:

Standalone Closure Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All closure logic is also available as standalone functions in
:mod:`~radiation.synchrotron.SEDs.one_zone_closure`. These are
useful when the SED class interface is not needed, or when using specialized
closures (e.g. the implicit-cooling or De Marchi SSA inversions) that are not
exposed through any SED class method.

.. currentmodule:: radiation.synchrotron.SEDs.one_zone_closure

.. rubric:: Public Inversion Functions

.. autosummary::
    :toctree: ../../../../_as_gen
    :nosignatures:

    invert_powerlaw_sed
    invert_powerlaw_cooling_sed
    invert_powerlaw_ssa_sed
    invert_powerlaw_cooling_ssa_sed
    invert_powerlaw_implicit_cooling_sed
    invert_powerlaw_implicit_cooling_ssa_sed
    invert_powerlaw_ssa_sed_demarchi

.. rubric:: Private (Low-Level) Inversion Functions

The following private functions expose the same inversions without unit handling
or cosmological distance resolution. They operate on logarithmic CGS inputs and
are intended for inference routines where unit overhead must be minimized.

.. autosummary::
    :toctree: ../../../../_as_gen
    :nosignatures:

    _invert_powerlaw_sed
    _invert_powerlaw_cooling_sed
    _invert_powerlaw_ssa_sed
    _invert_powerlaw_cooling_ssa_sed
    _invert_powerlaw_implicit_cooling_sed
    _invert_powerlaw_implicit_cooling_ssa_sed
    _invert_powerlaw_ssa_sed_demarchi


----

.. _sed_low_level:

The Low-Level Interface
------------------------

While most users will interact with synchrotron spectra through the high-level
SED classes, Triceratops also exposes a comprehensive **low-level interface**
that provides direct access to the numerical and conceptual building blocks from
which those SEDs are constructed.

This interface is intentionally more granular and assumes familiarity with both
synchrotron theory and the internal structure of the SED system. It is primarily
intended for advanced use cases, including:

- Defining **custom synchrotron SEDs** not covered by the built-in classes,
- Implementing or testing **new physical regimes** or closure relations,
- Inspecting and validating the **numerical components** underlying the
  high-level API,
- Reproducing analytic results from the literature at the level of individual
  spectral segments,
- Debugging or benchmarking normalization and regime-selection logic.

The low-level interface reflects the internal architecture of the synchrotron
SED system: complex spectra are assembled from small, scale-free components
whose behavior is well-defined in logarithmic space. These components are then
combined, ordered, and normalized by the high-level SED classes.

Conceptually, the low-level interface is organized into four categories:

1. **Shape functions**, which define smooth, scale-free spectral transitions,
2. **SED functions**, which encode complete synchrotron spectra for specific
   physical regimes,
3. **SSA utilities**, which compute :math:`\nu_a` and select the appropriate
   SSA regime,
4. **Normalization functions**, which convert scale-free shapes to physical flux densities.

Together, these form the substrate from which all high-level synchrotron SEDs
in Triceratops are built.


.. _low_level_shape_functions:

Shape Functions
^^^^^^^^^^^^^^^

Shape functions define the **mathematical form of spectral transitions**
independently of any specific physical scenario. They operate entirely in
logarithmic space and are designed to remain numerically stable across many
decades in frequency.

These functions do *not* encode synchrotron physics themselves. Instead, they
provide reusable, scale-free primitives that are composed into physical SEDs
elsewhere in the codebase.

.. currentmodule:: radiation.synchrotron.SEDs._one_zone_functions

.. rubric:: Shape Function API

.. autosummary::
    :toctree: ../../../../_as_gen
    :nosignatures:

    log_smoothed_SFBPL
    log_exp_cutoff_sed
    smoothed_BPL

In brief:

- :func:`log_smoothed_SFBPL` implements the core **scale-free smoothed broken
  power law (SFBPL)** factor. Multiple breaks are composed by adding
  SFBPL factors in log-space. The smoothing parameter :math:`s` controls the
  sharpness of the transition.
- :func:`log_exp_cutoff_sed` applies a smooth exponential truncation at high
  frequencies and is used to model finite maximum electron energies (the
  :math:`\nu_{\max}` cutoff).
- :func:`smoothed_BPL` is the un-logged version of a simple smoothed broken
  power law, normalized at the break frequency, used by :class:`SSA_SED_PowerLaw`.

These functions return **unnormalized spectral shapes** and are not intended to
be used directly in most workflows.

.. rubric:: Log-Space SED Composition Example

To illustrate log-space SED surgery, the following constructs a two-break SED
manually using two SFBPL factors:

.. code-block:: python

    import numpy as np
    from triceratops.radiation.synchrotron.SEDs._one_zone_functions import (
        log_smoothed_SFBPL,
        log_exp_cutoff_sed,
    )

    # Frequency grid (in log-space)
    log_nu = np.linspace(np.log(1e9), np.log(1e20), 500)

    # Break frequencies in log-space
    log_nu_m = np.log(1e12)
    log_nu_c = np.log(1e15)
    log_nu_max = np.log(1e19)
    p = 2.5

    # Start with a baseline power law: F ~ nu^(1/3)
    log_sed = (1.0 / 3.0) * log_nu

    # Add break at nu_m: slope transitions from 1/3 to -(p-1)/2
    log_sed += log_smoothed_SFBPL(log_nu - log_nu_m, 1.0 / 3.0, -(p - 1) / 2.0, -0.5)

    # Add break at nu_c: slope transitions from -(p-1)/2 to -p/2
    log_sed += log_smoothed_SFBPL(log_nu - log_nu_c, -(p - 1) / 2.0, -p / 2.0, -0.5)

    # Apply high-frequency exponential cutoff
    log_sed += log_exp_cutoff_sed(log_nu - log_nu_max)

    # Apply normalization
    log_F_norm = np.log(1e-26)
    # (Normalization offset to anchor at nu_m would need to be computed separately)


.. _low_level_sed_functions:

SED Functions
^^^^^^^^^^^^^

SED functions encode **complete synchrotron spectral shapes** corresponding to
specific physical scenarios and frequency orderings. Each function implements
a single, well-defined spectrum as derived in :ref:`synch_sed_theory`.

For example, a power-law electron population with radiative cooling and
synchrotron self-absorption can produce multiple distinct spectral shapes
depending on the ordering of :math:`\nu_m`, :math:`\nu_c`, :math:`\nu_a`, and
:math:`\nu_{\max}`. Each of these possibilities is implemented as a separate
low-level SED function.

To keep this manageable and explicit, Triceratops adopts a standardized naming
convention for low-level SED functions:

.. code-block:: text

    _log_<electron_pop>_<sed_type>_sed_<physics tags>_<spectrum number>

In practice, this expands to names such as:

.. code-block:: text

    _log_powerlaw_sbpl_sed_cool_2
    _log_powerlaw_sbpl_sed_ssa_1
    _log_powerlaw_sbpl_sed_ssa_cool_7

where:

- ``<electron_pop>`` indicates the assumed electron distribution
  (currently always ``powerlaw``),
- ``<sed_type>`` specifies the mathematical representation of the spectrum:
  ``sbpl`` for a smoothed broken power law,
- ``<physics tags>`` indicate which physical processes are included, in
  **order-independent form** (e.g. ``cool``, ``ssa``, or ``ssa_cool``),
- ``<spectrum number>`` labels the specific frequency ordering as defined in
  :ref:`synch_sed_theory`.

Each SED function:

- Operates entirely in **logarithmic space**,
- Assumes all inputs are already validated and dimensionless,
- Returns the **logarithm of the scale-free spectral shape**,
- Applies *no normalization*.

These functions are selected and composed automatically by the high-level SED
classes and are rarely called directly by users.

.. dropdown:: Available Low-Level SED Functions

   The following low-level SED functions are implemented in
   :mod:`radiation.synchrotron.SEDs._one_zone_functions`. Each corresponds
   to a unique spectral regime defined in :ref:`synch_sed_theory`.

   .. rubric:: Power Law (No Cooling, No SSA)

   .. autosummary::
      :toctree: ../../../../_as_gen
      :nosignatures:

        _log_powerlaw_sbpl_sed

   .. rubric:: Power Law + Cooling

   .. autosummary::
      :toctree: ../../../../_as_gen
      :nosignatures:

      _log_powerlaw_sbpl_sed_cool_1
      _log_powerlaw_sbpl_sed_cool_2

   .. rubric:: Power Law + SSA

   .. autosummary::
      :toctree: ../../../../_as_gen
      :nosignatures:

      _log_powerlaw_sbpl_sed_ssa_1
      _log_powerlaw_sbpl_sed_ssa_2

   .. rubric:: Power Law + Cooling + SSA

   .. autosummary::
      :toctree: ../../../../_as_gen
      :nosignatures:

      _log_powerlaw_sbpl_sed_ssa_cool_3
      _log_powerlaw_sbpl_sed_ssa_cool_4
      _log_powerlaw_sbpl_sed_ssa_cool_5
      _log_powerlaw_sbpl_sed_ssa_cool_6
      _log_powerlaw_sbpl_sed_ssa_cool_7
      _log_powerlaw_sbpl_sed_ssa_cool_8


.. _low_level_ssa_utilities:

SSA Utilities
^^^^^^^^^^^^^

SSA utilities compute the **self-absorption frequency** :math:`\nu_a` and
select the physically consistent SSA regime for a given set of input
parameters. These are called internally by :class:`PowerLaw_SSA_SynchrotronSED`
and :class:`PowerLaw_Cooling_SSA_SynchrotronSED` before dispatching to the
appropriate SED kernel.

.. currentmodule:: radiation.synchrotron.SEDs._one_zone_ssa

.. rubric:: SSA Utility API

.. autosummary::
    :toctree: ../../../../_as_gen
    :nosignatures:

    compute_ssa_frequencies_without_cooling
    compute_ssa_frequencies_with_cooling
    select_ssa_sed_regime_from_candidates_without_cooling
    select_ssa_sed_regime_from_candidates_with_cooling


----

.. _sed_extending:

Extending the SED System
------------------------

Adding a new synchrotron SED model is straightforward. The system is designed so
that new regimes or physical processes can be incorporated without modifying
existing code.

Step 1: Implement any new low-level SED functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the new SED requires spectral shapes not already provided, add them to
:mod:`radiation.synchrotron.SEDs._one_zone_functions` following the
established naming convention. Each function should:

- Accept ``log_nu`` (natural log of frequency in Hz) and regime-specific break
  frequencies in the same log form,
- Return the **natural logarithm of the scale-free spectral shape** (no normalization),
- Be fast and branch-free.

Step 2: Subclass the appropriate base class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a **single-regime SED** (like :class:`PowerLaw_SynchrotronSED`), subclass
:class:`SynchrotronSED` and implement:

- ``_log_opt_sed(self, log_nu, **params)`` — the log-space kernel,
- ``sed(self, nu, **params)`` — the unit-aware public interface.

For a **multi-regime SED** (like :class:`PowerLaw_Cooling_SSA_SynchrotronSED`),
subclass :class:`MultiSpectrumSynchrotronSED` and implement:

- ``_compute_sed_regime(self, **params) -> (regime, derived)`` — regime logic,
- ``determine_sed_regime(self, **params) -> regime`` — unit-aware public wrapper,
- ``_log_opt_sed_from_regime(self, log_nu, regime, **params)`` — regime dispatch kernel,
- ``SPECTRUM_FUNCTIONS`` — class attribute mapping regime identifiers to functions.

The inherited :meth:`MultiSpectrumSynchrotronSED._log_opt_sed` orchestrates
regime determination and dispatch automatically; subclasses should not override it.

Step 3: Implement closure relations (optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If physical inversion is needed, implement:

- ``_opt_from_physics_to_params(self, log_B, log_R, ...)`` — log-space forward closure,
- ``from_physics_to_params(self, B, R, ...)`` — unit-aware public wrapper,
- ``_opt_from_params_to_physics(self, log_F_peak, ...)`` — log-space inversion,
- ``from_params_to_physics(self, F_peak, ...)`` — unit-aware public wrapper.

Step 4: Register and export
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add the new class to ``__all__`` in
:mod:`radiation.synchrotron.SEDs.one_zone` and to the parent
:mod:`radiation.synchrotron.SEDs.__init__` re-exports.

.. rubric:: Minimal SED Skeleton

The following illustrates the minimal structure for a custom single-regime SED:

.. code-block:: python

    import numpy as np
    import astropy.units as u
    from triceratops.radiation.synchrotron.SEDs.one_zone import SynchrotronSED
    from triceratops.utils.misc_utils import ensure_in_units


    class MySynchrotronSED(SynchrotronSED):
        """Custom synchrotron SED."""

        def _log_opt_sed(self, log_nu, log_F_norm, log_nu_0, alpha):
            # All inputs are dimensionless CGS log-values.
            # Spectral shape: F ~ (nu/nu_0)^alpha
            return log_F_norm + alpha * (log_nu - log_nu_0)

        def sed(self, nu, F_norm, nu_0, alpha):
            nu = ensure_in_units(nu, "Hz")
            F_norm = ensure_in_units(F_norm, "erg cm^-2 s^-1 Hz^-1")
            nu_0 = ensure_in_units(nu_0, "Hz")

            log_F = self._log_opt_sed(
                np.log(nu), np.log(F_norm), np.log(nu_0), alpha
            )
            return np.exp(log_F) * u.erg / (u.cm**2 * u.s * u.Hz)


----

Relationship to the High-Level Interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The low-level interface should be viewed as the **implementation layer** of the
synchrotron SED system. High-level SED classes orchestrate these components to
provide a stable, physically meaningful, and easy-to-use user-facing API.

Most users will never need to interact with the low-level functions directly.
However, they are fully documented and exposed to support transparency,
extensibility, and rigorous validation of synchrotron spectral models.

References
----------

.. footbibliography::
