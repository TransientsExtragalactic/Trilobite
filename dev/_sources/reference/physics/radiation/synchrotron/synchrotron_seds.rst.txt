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

The High-Level Interface
-----------------------

Most users should interact with synchrotron spectra through the **SED classes**
defined in :mod:`radiation.synchrotron.SEDs`. These classes provide a
**structured, declarative interface** for evaluating physically consistent
synchrotron spectral energy distributions without requiring users to manually
assemble broken power laws or reason about regime selection.

Each SED class represents a **specific physical model** (e.g. cooling only,
SSA only, cooling + SSA) and is responsible for orchestrating all steps required
to compute a spectrum from a minimal set of inputs.


The Synchrotron SED Class
^^^^^^^^^^^^^^^^^^^^^^^^^^

All of the synchrotron SEDs in Triceratops inherit from the base
:class:`~radiation.synchrotron.SEDs.SynchrotronSED` class. This class
defines the **minimal, uniform interface** that all synchrotron SED
implementations must satisfy, independent of the physical processes included
(e.g. cooling, self-absorption).

The purpose of this base class is *not* to encode synchrotron physics directly,
but to provide a consistent and predictable framework for evaluating spectra,
handling units, and (optionally) linking phenomenological parameters to physical
models.

In principle, a synchrotron SED could be represented as a standalone function
mapping frequency to flux density. In practice, however, even the simplest
realistic synchrotron spectrum involves more than a single formula. The shape
of the spectrum depends on the relative ordering of multiple characteristic
frequencies, may include optional physical effects such as radiative cooling or
synchrotron self-absorption, and must remain numerically stable across many
orders of magnitude in frequency.

Encapsulating SEDs in classes allows Triceratops to manage this complexity in a
clean and transparent way. Each class is responsible for determining the
appropriate spectral regime, computing any internally derived quantities
(such as the self-absorption frequency :math:`\nu_a`), selecting the correct
scale-free spectral shape, and applying the appropriate normalization.

From the user’s perspective, this design has an important consequence: you
specify *what physics you want included*, not *how to assemble the spectrum by
hand*. The details of regime logic, break ordering, and normalization are handled
internally and consistently.

Instantiating SED Objects
##########################

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

SED Calculation
###############

.. currentmodule:: radiation.synchrotron.SEDs

The primary interface for evaluating a synchrotron spectrum is the
:meth:`SynchrotronSED.sed` method. Given an array of frequencies and the required physical or
phenomenological parameters, this method returns the corresponding flux
density:

.. code-block:: python

    import numpy as np
    from astropy import units as u
    from triceratops.radiation.synchrotron.SEDs import PowerLaw_SSA_SynchrotronSED
    sed = PowerLaw_SSA_SynchrotronSED()

    nu = np.logspace(9, 18, 500) * u.Hz

    Fnu = sed.sed(
        nu,
        nu_m=1e12 * u.Hz,
        F_peak=1e-26 * u.erg / (u.s * u.cm**2 * u.Hz),
        p=2.5,
    )

Internally, the SED class performs the following steps:

1. Validate and coerce input units,
2. Convert inputs to logarithmic CGS form,
3. Determine the global spectral regime,
4. Compute any derived break frequencies,
5. Dispatch to a regime-specific optimized kernel,
6. Apply the overall normalization.

For convenience, most SED classes also support direct call syntax, making the
following equivalent:

.. code-block:: python

    Fnu = sed(nu, nu_m=..., nu_c=..., F_peak=...)

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
        nu_m=1e12 * u.Hz,
        nu_c=1e15 * u.Hz,
        F_peak=1e-26 * u.erg / (u.s * u.cm ** 2 * u.Hz),
        p=2.5,
    )

    import matplotlib.pyplot as plt

    plt.loglog(nu, Fnu)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel(r"$F_\nu$")
    plt.tight_layout()
    plt.show()


SED Normalization
#################

All synchrotron SEDs in Triceratops are normalized using the **peak flux density**
:math:`F_{\rm peak}`. This choice is made consistently across *all* physical
scenarios—whether or not the spectrum includes radiative cooling, synchrotron
self-absorption, or additional structure—so that phenomenological modeling
always uses the same normalization convention.

Normalizing at the spectral peak has two key advantages. First, it provides a
stable and intuitive anchor for fitting and inference, since the peak flux is
often the best-constrained observable in broadband data. Second, it cleanly
separates the **shape** of the spectrum (which depends on frequency ordering and
included physics) from its **overall amplitude**, which is treated
phenomenologically by default.

Each SED class is responsible for applying the correct normalization internally.
The location of the peak frequency and the spectral slopes surrounding it depend
on the physical processes included (e.g. cooling, SSA) and on the ordering of
characteristic break frequencies. These details are handled entirely inside the
class; the user only supplies the value of :math:`F_{\rm peak}` at evaluation
time. Peak flux values are never stored on the SED object itself.

In some cases, an SED class also supports *physical normalization* and parameter
inversion through the optional methods
:meth:`SynchrotronSED.from_params_to_physics` and :meth:`SynchrotronSED.from_physics_to_params`. These methods
attempt to relate phenomenological SED parameters—such as
:math:`F_{\rm peak}`, :math:`\nu_m`, or :math:`\nu_c`—to underlying physical
quantities like magnetic field strength, emitting radius, or energy density.

Such mappings necessarily rely on analytic approximations and closure relations,
most commonly equipartition assumptions (see :ref:`synchrotron_theory`). As a
result, they should be viewed as *model-dependent conveniences* rather than
fundamental identities.

Not all SED classes support physical normalization or parameter inversion.
Users should consult the documentation for each specific SED class to determine
whether these capabilities are available and what assumptions they encode.

Available SED Classes
^^^^^^^^^^^^^^^^^^^^^

Triceratops provides four high-level synchrotron SED classes, each corresponding
to a different combination of physical effects. All classes share the same
conceptual interface and normalization strategy; they differ only in which
physical processes are included internally.

For a detailed derivation of the spectral slopes, break orderings, and regime
definitions referenced below, see :ref:`synch_sed_theory`.

Power Law + Cooling SED
#######################

The **Power Law + Cooling SED** includes radiative cooling but neglects
synchrotron self-absorption. It is appropriate for optically thin sources where
electron cooling modifies the high-energy spectrum.

Depending on the ordering of :math:`\nu_m`, :math:`\nu_c`, and
:math:`\nu_{\max}`, the spectrum automatically transitions between:

- Fast-cooling,
- Slow-cooling,
- Effectively non-cooling regimes.

The correct regime is selected internally based solely on the supplied break
frequencies.

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
            nu_m=1e12 * u.Hz,
            nu_c=1e15 * u.Hz,
            nu_max=1e19 * u.Hz,
            F_peak=1e-26 * u.erg / (u.s * u.cm**2 * u.Hz),
            p=2.5,
            s=-0.05,
        )

        plt.loglog(nu, Fnu)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel(r"$F_\nu$")
        plt.show()

For the explicit spectral slopes corresponding to each cooling regime, see
:ref:`synch_sed_theory`.


Power Law + SSA SED
####################

The **Power Law + SSA SED** includes synchrotron self-absorption but neglects
radiative cooling. This is appropriate for compact or dense emission regions
where the low-frequency spectrum is optically thick, but electrons do not cool
appreciably.

Unlike simple textbook treatments, Triceratops computes the self-absorption
frequency :math:`\nu_a` **self-consistently** from the supplied normalization and
microphysical parameters, rather than requiring it as a user input.

The spectrum automatically transitions between optically thick and thin regimes
based on the internally derived :math:`\nu_a`.

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
            nu_m=1e11 * u.Hz,
            F_peak=1e-26 * u.erg / (u.s * u.cm**2 * u.Hz),
            omega=4 * np.pi,
            gamma_m=300,
            p=2.5,
        )

        plt.loglog(nu, Fnu)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel(r"$F_\nu$")
        plt.show()

For the analytic scalings used to compute :math:`\nu_a`, see
:ref:`synch_sed_theory`.


Power Law + Cooling + SSA SED
##############################

The **Power Law + Cooling + SSA SED** is the most general synchrotron model
implemented in Triceratops. It includes:

- Radiative cooling,
- Synchrotron self-absorption,
- Stratified SSA corrections where applicable,
- A high-frequency exponential cutoff.

This class automatically handles all allowed orderings of
:math:`\nu_m`, :math:`\nu_c`, :math:`\nu_a`, and :math:`\nu_{\max}`, selecting the
appropriate spectral shape internally.

It is the recommended choice when modeling broadband synchrotron emission from
transient shocks, GRB afterglows, radio supernovae, or similar systems.

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
            nu_m=1e12 * u.Hz,
            nu_c=1e15 * u.Hz,
            nu_max=1e19 * u.Hz,
            F_peak=1e-26 * u.erg / (u.s * u.cm**2 * u.Hz),
            omega=4 * np.pi,
            gamma_m=300,
            p=2.5,
            s=-0.05,
        )

        plt.loglog(nu, Fnu)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel(r"$F_\nu$")
        plt.show()

Because of the large number of possible spectral orderings, users are strongly
encouraged to consult :ref:`synch_sed_theory` when interpreting the resulting
spectral shapes.

----

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
3. **Normalization utilities**, which connect scale-free spectra to physical
   flux densities,
4. **Regime-determination logic**, which selects the globally consistent spectral
   configuration.

Together, these form the substrate from which all high-level synchrotron SEDs
in Triceratops are built.

Shape Functions
^^^^^^^^^^^^^^^

Shape functions define the **mathematical form of spectral transitions**
independently of any specific physical scenario. They operate entirely in
logarithmic space and are designed to remain numerically stable across many
decades in frequency.

These functions do *not* encode synchrotron physics themselves. Instead, they
provide reusable, scale-free primitives that are composed into physical SEDs
elsewhere in the codebase.

The most important shape functions are:

.. currentmodule:: radiation.synchrotron.SEDs

.. rubric:: Shape Function API
.. autosummary::
    :toctree: ../../../../_as_gen
    :nosignatures:

    log_smoothed_SFBPL
    log_exp_cutoff_sed

In brief:

- ``log_smoothed_SFBPL`` implements a scale-free smoothed broken power law,
  allowing multiple spectral breaks to be connected smoothly while preserving
  asymptotic power-law behavior.
- ``log_exp_cutoff_sed`` applies a smooth exponential truncation at high
  frequencies and is used to model finite maximum electron energies.

These functions return **unnormalized spectral shapes in log-space** and are
not intended to be used directly in most workflows.

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

    _log_powerlaw_sbpl_sed_cool_4
    _log_powerlaw_sbpl_sed_ssa_2
    _log_powerlaw_sbpl_sed_cool_ssa_7

where:

- ``<electron_pop>`` indicates the assumed electron distribution
  (currently always ``powerlaw``),
- ``<sed_type>`` specifies the mathematical representation of the spectrum:
  ``bpl`` for a sharp broken power law or ``sbpl`` for a smoothed broken power law,
- ``<physics tags>`` indicate which physical processes are included, in
  **order-independent form** (e.g. ``cool``, ``ssa``, or ``cool_ssa``),
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
   :mod:`radiation.synchrotron.SEDs`. Each corresponds to a unique spectral
   regime defined in :ref:`synch_sed_theory`.

   .. rubric:: Power Law (No Cooling, No SSA)

   .. autosummary::
      :toctree: ../../../../_as_gen
      :nosignatures:

        _log_powerlaw_sbpl_sed

   .. rubric:: Power Law + Cooling

   .. autosummary::
      :toctree: ../../../../_as_gen
      :nosignatures:

      _log_powerlaw_sbpl_sed_cool_2
      _log_powerlaw_sbpl_sed_cool_1

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




Relationship to the High-Level Interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The low-level interface should be viewed as the **implementation layer** of the
synchrotron SED system. High-level SED classes orchestrate these components to
provide a stable, physically meaningful, and easy-to-use user-facing API.

Most users will never need to interact with the low-level functions directly.
However, they are fully documented and exposed to support transparency,
extensibility, and rigorous validation of synchrotron spectral models.
