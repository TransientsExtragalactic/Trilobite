.. _synchrotron_seds:

=========================================
Synchrotron Spectral Energy Distributions
=========================================

Triceratops models synchrotron SEDs as log-space compositions of scale-free spectral segments,
enabling numerically stable broadband spectra for astrophysical transients (GRBs, supernovae,
TDEs). Each spectrum is parameterized by the ordering of characteristic break frequencies—the
injection frequency :math:`\nu_m`, cooling frequency :math:`\nu_c`, self-absorption frequency
:math:`\nu_a`, and maximum frequency :math:`\nu_{\max}`. For background on the underlying physics
see :ref:`synchrotron_theory`; for derivations of individual spectral regimes see
:ref:`synch_sed_theory`.

.. contents::
   :local:
   :depth: 2

----

.. _sed_quick_start:

Quick Start
-----------

The following end-to-end example instantiates the most general SED class, converts physical
source parameters to phenomenological SED parameters, and plots the resulting spectrum:

.. plot::
   :include-source: True

    import numpy as np
    from astropy import units as u
    import matplotlib.pyplot as plt
    from triceratops.radiation.synchrotron import PowerLaw_Cooling_SSA_SynchrotronSED

    sed = PowerLaw_Cooling_SSA_SynchrotronSED()
    nu = np.logspace(8, 18, 500) * u.Hz

    # Convert physical parameters to SED parameters
    D_L = 100 * u.Mpc
    parameters = dict(
        B=0.5 * u.G,
        R=1e16 * u.cm,
        gamma_min=100.0,
        gamma_c=1e4,
        gamma_max=1e7,
        p=2.5,
        f_V=1.0,
        f_A=1.0,
        epsilon_E=0.1,
        epsilon_B=0.1,
        luminosity_distance=D_L,
        pitch_average=True,
    )

    norm = sed.from_physics_to_params(**parameters)

    # Evaluate and plot
    Fnu = sed.sed(nu, nu_m=norm['nu_m'],
                  nu_c=norm['nu_c'],
                  F_norm=norm['F_norm'],
                  nu_max=norm['nu_max'],
                  p=parameters['p'],
                  omega=norm['omega'],
                  gamma_m=parameters['gamma_min'])
    plt.loglog(nu, Fnu)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel(r"$F_\nu$ [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
    plt.tight_layout()
    plt.show()

----

.. _sed_choosing:

Choosing a SED Model
--------------------

Triceratops provides four physically motivated SED classes and one phenomenological option.
Choose based on which physical processes are important for your source:

.. list-table::
   :widths: 40 10 10 40
   :header-rows: 1

   * - Model
     - Cooling
     - SSA
     - Use when…
   * - :class:`~triceratops.radiation.synchrotron.SEDs.one_zone.PowerLaw_SynchrotronSED`
     - ✗
     - ✗
     - Simple power-law spectra; no breaks beyond :math:`\nu_m`
   * - :class:`~triceratops.radiation.synchrotron.SEDs.one_zone.PowerLaw_Cooling_SynchrotronSED`
     - ✓
     - ✗
     - Optically thin emission with a fast- or slow-cooling break
   * - :class:`~triceratops.radiation.synchrotron.SEDs.one_zone.PowerLaw_SSA_SynchrotronSED`
     - ✗
     - ✓
     - Compact or dense sources with an SSA turnover; electrons do not cool appreciably
   * - :class:`~triceratops.radiation.synchrotron.SEDs.one_zone.PowerLaw_Cooling_SSA_SynchrotronSED`
     - ✓
     - ✓
     - Full broadband modeling (GRBs, SNe, TDEs); up to 8 spectral regimes

:class:`~triceratops.radiation.synchrotron.SEDs.one_zone.SSA_SED_PowerLaw` is a phenomenological
alternative where the break frequency is supplied directly by the user rather than derived from
microphysics, following the closure of :footcite:t:`demarchiRadioAnalysisSN2004C2022`. Use it
when you prefer to fit the SSA turnover without invoking any closure assumptions.

----

.. _sed_evaluation:

Evaluating a Spectrum
---------------------

.. _sed_instantiation:

Instantiation
^^^^^^^^^^^^^

SED objects in Triceratops are intentionally lightweight. Instantiating an SED class requires
**no physical parameters**:

.. code-block:: python

   from triceratops.radiation.synchrotron import PowerLaw_Cooling_SynchrotronSED

   sed = PowerLaw_Cooling_SynchrotronSED()

SED instances do **not** store model parameters. Break frequencies, peak fluxes, and other
physical quantities are supplied at *evaluation time*, not at initialization. This makes SED
objects safe to reuse in parameter sweeps, inference loops, and population modeling without
hidden state.

.. _sed_calling:

Calling ``sed()``
^^^^^^^^^^^^^^^^^

The primary interface for evaluating a synchrotron spectrum is :meth:`~triceratops.radiation.synchrotron.SEDs.one_zone.SynchrotronSED.sed`.
Supply a frequency array and the phenomenological parameters:

.. code-block:: python

   import numpy as np
   from astropy import units as u

   nu = np.logspace(9, 18, 500) * u.Hz

   Fnu = sed.sed(
       nu,
       F_norm=1e-26 * u.erg / (u.s * u.cm**2 * u.Hz),
       nu_m=1e12 * u.Hz,
       nu_c=1e15 * u.Hz,
       nu_max=1e19 * u.Hz,
       p=2.5,
   )

All SED classes also support **direct call syntax**:

.. code-block:: python

   Fnu = sed(nu, F_norm=..., nu_m=..., nu_c=..., nu_max=..., p=2.5)

The following example shows a slow-cooling spectrum with annotated break frequencies:

.. plot::
   :include-source: True

   import numpy as np
   from astropy import units as u
   import matplotlib.pyplot as plt
   from triceratops.radiation.synchrotron import PowerLaw_Cooling_SynchrotronSED

   sed = PowerLaw_Cooling_SynchrotronSED()
   nu = np.logspace(9, 18, 500) * u.Hz

   nu_m = 1e12 * u.Hz
   nu_c = 1e15 * u.Hz
   nu_max = 1e19 * u.Hz

   Fnu = sed.sed(
       nu,
       F_norm=1e-26 * u.erg / (u.s * u.cm**2 * u.Hz),
       nu_m=nu_m,
       nu_c=nu_c,
       nu_max=nu_max,
       p=2.5,
   )

   plt.loglog(nu, Fnu)
   plt.axvline(nu_m.value, color="C1", ls="--", label=r"$\nu_m$")
   plt.axvline(nu_c.value, color="C2", ls="--", label=r"$\nu_c$")
   plt.axvline(nu_max.value, color="C3", ls="--", label=r"$\nu_{\max}$")
   plt.xlabel("Frequency [Hz]")
   plt.ylabel(r"$F_\nu$ [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
   plt.legend()
   plt.tight_layout()
   plt.show()

.. note::

   :class:`~triceratops.radiation.synchrotron.SEDs.one_zone.PowerLaw_Cooling_SynchrotronSED`,
   :class:`~triceratops.radiation.synchrotron.SEDs.one_zone.PowerLaw_SSA_SynchrotronSED`, and
   :class:`~triceratops.radiation.synchrotron.SEDs.one_zone.PowerLaw_Cooling_SSA_SynchrotronSED` use
   keyword-only arguments after ``nu``; supplying parameters positionally will raise a
   ``TypeError``.

----

.. _sed_forward_closure:

Normalizing from Physical Parameters
-------------------------------------

If you have physical source parameters (magnetic field :math:`B`, radius :math:`R`,
energy partition fractions :math:`\varepsilon_E`, :math:`\varepsilon_B`, …) rather than
phenomenological ones, use :meth:`~triceratops.radiation.synchrotron.SEDs.one_zone.SynchrotronSED.from_physics_to_params`
to compute all required SED inputs in one step.

.. code-block:: python

   import numpy as np
   from astropy import units as u
   from triceratops.radiation.synchrotron import PowerLaw_Cooling_SSA_SynchrotronSED

   sed = PowerLaw_Cooling_SSA_SynchrotronSED()

   params = sed.from_physics_to_params(
       B=0.5 * u.G,
       R=1e16 * u.cm,
       gamma_min=100.0,
       gamma_c=1e4,
       gamma_max=1e7,
       p=2.5,
       f_V=1.0,
       f_A=1.0,
       epsilon_E=0.1,
       epsilon_B=0.1,
       redshift=0.01,
       pitch_average=True,
   )
   # params contains: F_norm, nu_m, nu_c, nu_a, nu_max, nu_peak, F_peak, regime (all with units)

Pass the result directly to :meth:`~triceratops.radiation.synchrotron.SEDs.one_zone.SynchrotronSED.sed`:

.. code-block:: python

   nu = np.logspace(8, 18, 500) * u.Hz
   Fnu = sed.sed(nu, **params, p=2.5)

.. important::

   :math:`F_{\nu,\mathrm{norm}}` is **not necessarily the peak of the observed spectrum**.
   In the presence of synchrotron self-absorption the actual peak flux (at :math:`\nu_a`) will
   differ from :math:`F_{\nu,\mathrm{norm}}`. The normalization is always defined in the
   optically thin limit, regardless of absorption.

   This convention ensures that the normalization parameter has a unique, physically transparent
   meaning across *all* SED types and spectral regimes. The **peak flux**
   :math:`F_{\nu,\mathrm{peak}}` (the maximum of the observed SED) is computed internally from
   :math:`F_{\nu,\mathrm{norm}}` and returned as ``F_peak`` in the output dictionary. For a
   detailed derivation of these mappings see :ref:`single_zone_sed_normalization`.

.. rubric:: Distance Options

The method accepts several equivalent ways to specify the cosmological distance:

.. code-block:: python

   # Redshift (luminosity distance resolved from default cosmology)
   params = sed.from_physics_to_params(..., redshift=0.01)

   # Explicit luminosity distance
   params = sed.from_physics_to_params(..., luminosity_distance=150 * u.Mpc)

   # Explicit angular diameter distance
   params = sed.from_physics_to_params(..., angular_diameter_distance=148 * u.Mpc)

   # Custom astropy cosmology
   from astropy.cosmology import Planck18
   params = sed.from_physics_to_params(..., redshift=0.01, cosmology=Planck18)

----

.. _sed_inverse_closure:

Inverting the SED
-----------------

Given observed peak flux and peak frequency, you can recover physical source parameters
(radius :math:`R`, magnetic field :math:`B`) using
:meth:`~triceratops.radiation.synchrotron.SEDs.one_zone.SynchrotronSED.from_params_to_physics`:

.. code-block:: python

   from triceratops.radiation.synchrotron import PowerLaw_Cooling_SSA_SynchrotronSED
   import astropy.units as u

   sed = PowerLaw_Cooling_SSA_SynchrotronSED()

   physics = sed.from_params_to_physics(
       regime=params["regime"],
       F_peak=params["F_peak"],
       nu_peak=params["nu_peak"],
       gamma_min=100.0,
       gamma_c=1e4,
       p=2.5,
       epsilon_E=0.1,
       epsilon_B=0.1,
       f_V=1.0,
       f_A=1.0,
       redshift=0.01,
       pitch_average=True,
   )
   print(f"R = {physics['R']:.2e},  B = {physics['B']:.2e}")

.. note::

   The cooling and SSA classes require an explicit ``regime`` argument because multiple distinct
   spectral orderings can produce identical peak quantities. When chaining
   ``from_physics_to_params`` → ``from_params_to_physics``, pass ``params["regime"]`` directly
   to guarantee a round-trip consistent inversion.

.. warning::

   The inverse closure provides a **default physical interpretation**, not a unique truth. The theory
    documentation (:ref:`synch_sed_theory`) details the assumptions and limitations of the closure relations
    implemented in Triceratops. It is sometimes the case that a user needs a specific inversion which is not
    implemented currently.

.. _sed_standalone_closures:

Standalone Closure Functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All closure logic is also available as standalone functions in
:mod:`~radiation.synchrotron.SEDs.one_zone_closure`. These are useful when the SED class
interface is not needed, or for specialized closures (e.g. the implicit-cooling or De Marchi SSA
inversions) that are not exposed through any SED class method.

For **relativistic sources in the coasting phase**, the
:func:`~triceratops.radiation.synchrotron.SEDs.one_zone_closure.invert_barniol_duran_coasting`
function implements the :footcite:t:`duran2013radius` equipartition inversion, which
simultaneously solves for the radius :math:`R`, bulk Lorentz factor :math:`\Gamma`,
equipartition energy :math:`E`, and the derived microphysical quantities
:math:`\gamma_e`, :math:`N_e`, and :math:`B`.  See :ref:`barnoil_duran` for the
theoretical background.

.. currentmodule:: triceratops.radiation.synchrotron.SEDs.one_zone_closure

.. rubric:: Public Inversion Functions

.. autosummary::
   :nosignatures:

   invert_powerlaw_sed
   invert_powerlaw_cooling_sed
   invert_powerlaw_ssa_sed
   invert_powerlaw_cooling_ssa_sed
   invert_powerlaw_implicit_cooling_sed
   invert_powerlaw_implicit_cooling_ssa_sed
   invert_powerlaw_ssa_sed_demarchi
   invert_barniol_duran_coasting

.. rubric:: Private (Low-Level) Inversion Functions

The following private functions expose the same inversions without unit handling or cosmological
distance resolution. They operate on logarithmic CGS inputs and are intended for inference
routines where unit overhead must be minimized.

.. autosummary::
   :nosignatures:

   _invert_powerlaw_sed
   _invert_powerlaw_cooling_sed
   _invert_powerlaw_ssa_sed
   _invert_powerlaw_cooling_ssa_sed
   _invert_powerlaw_implicit_cooling_sed
   _invert_powerlaw_implicit_cooling_ssa_sed
   _invert_powerlaw_ssa_sed_demarchi
   _invert_barniol_duran_coasting

----

.. _sed_regime_introspection:

Inspecting the Spectral Regime
-------------------------------

For :class:`~triceratops.radiation.synchrotron.SEDs.one_zone.MultiSpectrumSynchrotronSED` subclasses, you
can query which spectral regime was selected for a given set of parameters via
:meth:`~triceratops.radiation.synchrotron.SEDs.one_zone.MultiSpectrumSynchrotronSED.determine_sed_regime`:

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
   print(regime)  # e.g. "Spectrum3"

The returned identifier is a string of the form ``Spectrum1`` … ``Spectrum8``, corresponding to
the spectral orderings catalogued in :ref:`synch_sed_theory`. The mapping from regime name to
frequency ordering is also recorded in the class-level ``SPECTRUM_FUNCTIONS`` attribute.

This is useful for diagnosing which branch of the SED is being evaluated, particularly when
debugging or verifying physical parameter ranges.

----

.. _sed_api_reference:

API Reference
-------------

.. currentmodule:: triceratops.radiation.synchrotron.SEDs.one_zone

.. autosummary::
   :nosignatures:

   SynchrotronSED
   MultiSpectrumSynchrotronSED
   PowerLaw_SynchrotronSED
   PowerLaw_Cooling_SynchrotronSED
   PowerLaw_SSA_SynchrotronSED
   PowerLaw_Cooling_SSA_SynchrotronSED
   SSA_SED_PowerLaw

----

.. _sed_developer_notes:

Implementation and Developer Notes
------------------------------------

.. note::

   This section is for developers extending or debugging the SED system.

.. _sed_class_hierarchy:

Class Hierarchy
^^^^^^^^^^^^^^^

The synchrotron SED system is organized around a two-level abstract class hierarchy that
separates **interface** from **regime dispatch logic**:

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

The key design principle is that **SED objects are stateless**: no physical parameters are stored
at instantiation. All parameters are supplied at call time, making objects safe to reuse across
parameter sweeps or inference loops.

.. _sed_low_level:

Low-Level Interface
^^^^^^^^^^^^^^^^^^^^

While most users will interact with synchrotron spectra through the high-level SED classes,
Triceratops also exposes a **low-level interface** that provides direct access to the building
blocks from which those SEDs are constructed. It is primarily intended for:

- Defining custom synchrotron SEDs not covered by the built-in classes,
- Implementing or testing new physical regimes or closure relations,
- Inspecting and validating the numerical components underlying the high-level API,
- Debugging or benchmarking normalization and regime-selection logic.

The low-level interface is organized into three categories: **shape functions**,
**SED functions**, and **SSA utilities**.

.. _low_level_shape_functions:

Shape Functions
"""""""""""""""

Shape functions define the mathematical form of spectral transitions independently of any
specific physical scenario. They operate entirely in logarithmic space.

.. currentmodule:: triceratops.radiation.synchrotron.SEDs._one_zone_functions

.. rubric:: Shape Function API

.. autosummary::
   :nosignatures:

   log_smoothed_SFBPL
   log_exp_cutoff_sed
   smoothed_BPL

- :func:`log_smoothed_SFBPL` — the core scale-free smoothed broken power-law (SFBPL) factor.
  Multiple breaks are composed by adding SFBPL factors in log-space.
- :func:`log_exp_cutoff_sed` — smooth exponential truncation at high frequencies
  (the :math:`\nu_{\max}` cutoff).
- :func:`smoothed_BPL` — un-logged smoothed broken power law, used by
  :class:`~triceratops.radiation.synchrotron.SEDs.one_zone.SSA_SED_PowerLaw`.

.. rubric:: Log-Space SED Composition Example

.. code-block:: python

   import numpy as np
   from triceratops.radiation.synchrotron.SEDs._one_zone_functions import (
       log_smoothed_SFBPL,
       log_exp_cutoff_sed,
   )

   log_nu = np.linspace(np.log(1e9), np.log(1e20), 500)
   log_nu_m = np.log(1e12)
   log_nu_c = np.log(1e15)
   log_nu_max = np.log(1e19)
   p = 2.5

   # Baseline power law: F ~ nu^(1/3)
   log_sed = (1.0 / 3.0) * log_nu

   # Break at nu_m: slope 1/3 → -(p-1)/2
   log_sed += log_smoothed_SFBPL(log_nu - log_nu_m, 1.0 / 3.0, -(p - 1) / 2.0, -0.5)

   # Break at nu_c: slope -(p-1)/2 → -p/2
   log_sed += log_smoothed_SFBPL(log_nu - log_nu_c, -(p - 1) / 2.0, -p / 2.0, -0.5)

   # High-frequency exponential cutoff
   log_sed += log_exp_cutoff_sed(log_nu - log_nu_max)

.. _low_level_sed_functions:

SED Functions
"""""""""""""

SED functions encode complete synchrotron spectral shapes for specific physical scenarios and
frequency orderings, as derived in :ref:`synch_sed_theory`. Each function operates entirely in
logarithmic space, assumes all inputs are already validated and dimensionless, and returns the
logarithm of the scale-free spectral shape with *no normalization*.

Naming convention:

.. code-block:: text

   _log_<electron_pop>_<sed_type>_sed_<physics tags>_<spectrum number>

For example: ``_log_powerlaw_sbpl_sed_ssa_cool_7``

.. dropdown:: Available Low-Level SED Functions

   The following low-level SED functions are implemented in
   :mod:`triceratops.radiation.synchrotron.SEDs._one_zone_functions`. Each corresponds
   to a unique spectral regime defined in :ref:`synch_sed_theory`.

   .. rubric:: Power Law (No Cooling, No SSA)

   .. autosummary::
      :nosignatures:

      _log_powerlaw_sbpl_sed

   .. rubric:: Power Law + Cooling

   .. autosummary::
      :nosignatures:

      _log_powerlaw_sbpl_sed_cool_1
      _log_powerlaw_sbpl_sed_cool_2

   .. rubric:: Power Law + SSA

   .. autosummary::
      :nosignatures:

      _log_powerlaw_sbpl_sed_ssa_1
      _log_powerlaw_sbpl_sed_ssa_2

   .. rubric:: Power Law + Cooling + SSA

   .. autosummary::
      :nosignatures:

      _log_powerlaw_sbpl_sed_ssa_cool_3
      _log_powerlaw_sbpl_sed_ssa_cool_4
      _log_powerlaw_sbpl_sed_ssa_cool_5
      _log_powerlaw_sbpl_sed_ssa_cool_6
      _log_powerlaw_sbpl_sed_ssa_cool_7
      _log_powerlaw_sbpl_sed_ssa_cool_8

.. _low_level_ssa_utilities:

SSA Utilities
"""""""""""""

SSA utilities compute the self-absorption frequency :math:`\nu_a` and select the physically
consistent SSA regime for a given set of input parameters.

.. currentmodule:: triceratops.radiation.synchrotron.SEDs._one_zone_ssa

.. rubric:: SSA Utility API

.. autosummary::
   :nosignatures:

   compute_ssa_frequencies_without_cooling
   compute_ssa_frequencies_with_cooling
   select_ssa_sed_regime_from_candidates_without_cooling
   select_ssa_sed_regime_from_candidates_with_cooling

.. _sed_extending:

Extending the SED System
^^^^^^^^^^^^^^^^^^^^^^^^^

Adding a new synchrotron SED model is straightforward. The system is designed so that new
regimes or physical processes can be incorporated without modifying existing code.

**Step 1: Implement any new low-level SED functions**

If the new SED requires spectral shapes not already provided, add them to
:mod:`triceratops.radiation.synchrotron.SEDs._one_zone_functions` following the established naming
convention. Each function should:

- Accept ``log_nu`` (natural log of frequency in Hz) and regime-specific break frequencies in
  the same log form,
- Return the **natural logarithm of the scale-free spectral shape** (no normalization),
- Be fast and branch-free.

**Step 2: Subclass the appropriate base class**

For a **single-regime SED** (like :class:`~triceratops.radiation.synchrotron.SEDs.one_zone.PowerLaw_SynchrotronSED`),
subclass :class:`~triceratops.radiation.synchrotron.SEDs.one_zone.SynchrotronSED` and implement:

- ``_log_opt_sed(self, log_nu, **params)`` — the log-space kernel,
- ``sed(self, nu, **params)`` — the unit-aware public interface.

For a **multi-regime SED** (like
:class:`~triceratops.radiation.synchrotron.SEDs.one_zone.PowerLaw_Cooling_SSA_SynchrotronSED`), subclass
:class:`~triceratops.radiation.synchrotron.SEDs.one_zone.MultiSpectrumSynchrotronSED` and implement:

- ``_compute_sed_regime(self, **params) -> (regime, derived)`` — regime logic,
- ``determine_sed_regime(self, **params) -> regime`` — unit-aware public wrapper,
- ``_log_opt_sed_from_regime(self, log_nu, regime, **params)`` — regime dispatch kernel,
- ``SPECTRUM_FUNCTIONS`` — class attribute mapping regime identifiers to functions.

The inherited ``_log_opt_sed`` orchestrates regime determination and dispatch automatically;
subclasses should not override it.

**Step 3: Implement closure relations (optional)**

If physical inversion is needed, implement:

- ``_opt_from_physics_to_params(self, log_B, log_R, ...)`` — log-space forward closure,
- ``from_physics_to_params(self, B, R, ...)`` — unit-aware public wrapper,
- ``_opt_from_params_to_physics(self, log_F_peak, ...)`` — log-space inversion,
- ``from_params_to_physics(self, F_peak, ...)`` — unit-aware public wrapper.

**Step 4: Register and export**

Add the new class to ``__all__`` in :mod:`triceratops.radiation.synchrotron.SEDs.one_zone` and to the
parent :mod:`triceratops.radiation.synchrotron.SEDs.__init__` re-exports.

.. rubric:: Minimal SED Skeleton

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

The low-level interface should be viewed as the **implementation layer** of the synchrotron SED
system. High-level SED classes orchestrate these components to provide a stable, physically
meaningful, and easy-to-use user-facing API. Most users will never need to interact with the
low-level functions directly. However, they are fully documented and exposed to support
transparency, extensibility, and rigorous validation of synchrotron spectral models.

----

References
----------

.. footbibliography::
