.. _synchrotron_microphysics:

=========================================
Synchrotron Microphysics
=========================================

As discussed in :ref:`synchrotron_theory`, it is, in most cases, not possible to uniquely determine the microphysical
processes which govern the synchrotron emission from first principles. Instead, we rely on a set of phenomenological
parameters / closures which encapsulate our ignorance of the detailed physics at play. These microphysical parameters are
critical to the modeling of synchrotron emission, as they directly influence the resulting spectra and light curves.

In this document, we'll describe the available microphysical closures implemented in Triceratops, how to use them,
and provide references for further reading on the subject. The :mod:`triceratops.radiation.synchrotron.microphysics` module is
responsible for two core aspects of synchrotron microphysics:

1. Operations regarding electron distributions, including computing moments of the distributions
   and normalizing them.
2. Implementing microphysical closures, such as equipartition closure, to relate macrophysical quantities
   (e.g. shock energy densities) to microphysical ones (e.g. electron distribution normalizations and magnetic field strengths).

.. contents::
    :local:
    :depth: 2

Electron Distributions
----------------------

.. hint::

    See :ref:`synch_theory_populations` for the corresponding documentation on synchrotron theory.

While it is possible to compute synchrotron emission from an arbitrary population of electrons, in practice, one
often selects a specific distribution function for their electron population. Triceratops provides explicit support
for three electron distributions: **power-law**, **broken power-law**, and **Maxwell-Jüttner** (thermal). Mixed
thermal-plus-nonthermal (Maxwell-Jüttner + power-law) populations are also supported.

Power-Law Electron Distributions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The most commonly used electron distribution in astrophysical synchrotron modeling is the power-law distribution.
In this framework, the number density of electrons per unit Lorentz factor :math:`\gamma` is given by:

.. math::

    \frac{dN}{d\gamma} = N_0 \gamma^{-p},
    \qquad
    \gamma_{\min} \le \gamma \le \gamma_{\max},

where:

- :math:`p` is the power-law index,
- :math:`\gamma_{\min}` and :math:`\gamma_{\max}` define the support of the distribution,
- :math:`N_0` is the normalization constant.

This choice is motivated by both theoretical arguments for diffusive shock acceleration and its
empirical success in modeling non-thermal emission from a wide range of astrophysical sources.
A detailed discussion of the physical origin of this distribution can be found in
:ref:`synchrotron_theory`.

.. note::

    (See :ref:`synchrotron_theory` for detailed discussion) For the sake of clarity, Triceratops
    always defaults to using the Lorentz factor :math:`\gamma` as the independent variable in
    electron distributions, not the energy. Nonetheless, support for both is included and conversion functions
    are available.

.. hint::

    From the perspective of **model building**, the important point is that essentially all synchrotron
    observables depend on *moments* of this distribution rather than on its detailed shape. Thus, there are
    a number of useful functions in the :mod:`triceratops.radiation.synchrotron` module for computing these moments
    directly without needing to manipulate the distribution itself.


.. rubric:: API Reference

*current module*: :mod:`triceratops.radiation.synchrotron.microphysics`

.. tab-set::

    .. tab-item:: High-Level API

        The following high-level helper functions are provided for working with power-law electron
        distributions:

        .. currentmodule:: triceratops.radiation.synchrotron.microphysics
        .. autosummary::
           :nosignatures:

           compute_electron_gamma_PL_moment
           compute_electron_energy_PL_moment
           compute_mean_gamma_PL
           compute_mean_energy_PL
           compute_PL_total_number_density
           compute_PL_effective_number_density
           swap_electron_PL_normalization

    .. tab-item:: Low-Level API


        Mirroring the high-level functions, the following low-level functions are provided for
        working directly with the power-law electron distribution:

        - ``_opt_compute_PL_moment``, which computes the generic
          moment of the power-law distribution:

          .. math::

                I = \int_{x_0}^{x_1} x^{(n-p)} dx,

          where :math:`x` is the variable of integration (e.g. :math:`\gamma` or :math:`E`), and
          :math:`n` is the order of the moment.
        - ``_opt_compute_PL_n_total``, which computes the total number of electrons in the power-law
          distribution:

          .. math::

              N_{\rm total} = \int_{\gamma_{\min}}^{\gamma_{\max}} \frac{dN}{d\gamma} d\gamma = N_0 M^{(1)}_\gamma.

        - ``_opt_compute_PL_n_eff``, which computes the effective number of radiating electrons in the
          distribution (see :ref:`synchrotron_theory` for details):

          .. math::

                N_{\rm eff} = N_0 M^{(2)}_\gamma. = N_{\rm total} \frac{M^{(2)}_\gamma}{M^{(1)}_\gamma}.

        - ``_opt_compute_convert_PL_norm_energy_to_gamma``, which converts the normalization of a power-law
          electron energy distribution to that of a power-law electron Lorentz factor distribution.
        - ``_opt_compute_convert_PL_norm_gamma_to_energy``, which converts the normalization of a power-law
          electron Lorentz factor distribution to that of a power-law electron energy distribution.

.. rubric:: Examples

For the most part, these functions are intended for internal use by higher-level modeling functions. However,
they can also be used directly when building custom models. For example, to compute the mean Lorentz factor
of a power-law electron distribution with :math:`p = 2.5`, :math:`\gamma_{\min} = 10^2`, and
:math:`\gamma_{\max} = 10^6`, one could use the following code:

.. code-block:: python

    from triceratops.radiation.synchrotron.microphysics import compute_mean_gamma_PL

    p = 2.5
    gamma_min = 1e2
    gamma_max = 1e6

    mean_gamma = compute_mean_gamma_PL(p, gamma_min, gamma_max)
    print(f"Mean Lorentz factor: {mean_gamma:.2f}")

You can also switch between normalizations of power-law distributions in energy and Lorentz factor. For example,
to convert a power-law electron energy distribution normalization of :math:`N_0 = 1e10 \, \rm cm^{-3} \, erg^{-1}`
with :math:`p = 2.5`, :math:`E_{\min} = 10^2 \, m_e c^2`, and :math:`E_{\max} = 10^6 \, m_e c^2` to the
corresponding Lorentz factor normalization, one could use the following code:

.. code-block:: python

    from triceratops.radiation.synchrotron.microphysics import (
        swap_electron_PL_normalization
    )

    p = 2.5
    N0_energy = 1e10  # in units of cm^-3 erg^-1

    N0_gamma = swap_electron_PL_normalization(
        N0_energy, p=p, mode='energy')
    print(f"Power-law normalization in gamma: {N0_gamma:.2e} cm^-3")

Broken Power-Law Electron Distributions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In many astrophysical environments, a single power-law electron distribution is insufficient to
accurately describe the non-thermal particle population. In such cases, a **broken power-law (BPL)**
distribution provides a more flexible and physically motivated model.

In this framework, the number density of electrons per unit Lorentz factor :math:`\gamma` is given by:

.. math::

    \frac{dN}{d\gamma} =
    \begin{cases}
        N_0 \left(\dfrac{\gamma}{\gamma_b}\right)^{-a_1},
        & \gamma_{\min} \le \gamma < \gamma_b, \\[1em]
        N_0 \left(\dfrac{\gamma}{\gamma_b}\right)^{-a_2},
        & \gamma_b \le \gamma \le \gamma_{\max},
    \end{cases}

where:

- :math:`a_1` and :math:`a_2` are the power-law indices below and above the break,
- :math:`\gamma_b` is the break Lorentz factor,
- :math:`\gamma_{\min}` and :math:`\gamma_{\max}` define the support of the distribution,
- :math:`N_0` is the normalization constant, defined at :math:`\gamma = \gamma_b`.

Broken power-law distributions arise naturally in scenarios where particle acceleration,
cooling, or escape processes modify the electron spectrum, such as radiative cooling breaks
in synchrotron-emitting plasmas or multi-stage acceleration mechanisms.

As with the single power-law case, essentially all synchrotron observables depend on **moments**
of the distribution rather than its detailed shape. Triceratops therefore provides a comprehensive
set of utilities for computing these moments directly.

.. note::

    As with power-law distributions, Triceratops consistently treats the Lorentz factor
    :math:`\gamma` as the independent variable for broken power-law distributions. Support
    for energy-space representations is provided internally, and conversion utilities
    are available where appropriate.

.. hint::

    From a modeling perspective, broken power-law distributions should be viewed as a
    *minimal extension* of the single-slope case. All high-level synchrotron quantities
    (e.g., emissivities, characteristic frequencies, and cooling rates) can be expressed
    in terms of a small set of distribution moments.


.. rubric:: API Reference

*current module*: :mod:`triceratops.radiation.synchrotron.microphysics`

.. tab-set::

    .. tab-item:: High-Level API

        The following high-level helper functions are provided for working with broken
        power-law electron distributions:

        .. currentmodule:: triceratops.radiation.synchrotron.microphysics
        .. autosummary::
           :nosignatures:

           compute_electron_gamma_BPL_moment
           compute_electron_energy_BPL_moment
           compute_mean_gamma_BPL
           compute_mean_energy_BPL
           compute_BPL_total_number_density
           compute_BPL_effective_number_density
           swap_electron_BPL_normalization

    .. tab-item:: Low-Level API

        Mirroring the high-level interface, the following low-level functions are provided
        for direct manipulation of broken power-law electron distributions:

        - ``_opt_compute_BPL_moment``, which computes the generic moment of a broken
          power-law distribution:

          .. math::

              I = \int_{\gamma_{\min}}^{\gamma_{\max}} \gamma^n \, N(\gamma)\, d\gamma.

        - ``_opt_compute_BPL_n_total``, which computes the total number density of electrons:

          .. math::

              N_{\rm total} = \int_{\gamma_{\min}}^{\gamma_{\max}} N(\gamma)\, d\gamma.

        - ``_opt_compute_BPL_n_eff``, which computes the effective number of radiating electrons
          contributing to synchrotron emission:

          .. math::

              N_{\rm eff} = \int_{\gamma_{\min}}^{\gamma_{\max}} N(\gamma)\, \gamma^2 \, d\gamma.

        - ``_opt_convert_BPL_norm_energy_to_gamma``, which converts the normalization of a
          broken power-law electron energy distribution to its Lorentz factor representation.
        - ``_opt_convert_BPL_norm_gamma_to_energy``, which performs the inverse conversion.

.. rubric:: Examples

As with power-law distributions, broken power-law utilities are primarily intended for use
by higher-level modeling routines. However, they can also be used directly for custom analyses.

For example, to compute the mean Lorentz factor of a broken power-law electron distribution
with :math:`a_1 = 2.0`, :math:`a_2 = 3.5`, :math:`\gamma_b = 10^4`,
:math:`\gamma_{\min} = 10^2`, and :math:`\gamma_{\max} = 10^7`, one could write:

.. code-block:: python

    from triceratops.radiation.synchrotron.microphysics import compute_mean_gamma_BPL

    a1 = 2.0
    a2 = 3.5
    gamma_b = 1e4
    gamma_min = 1e2
    gamma_max = 1e7

    mean_gamma = compute_mean_gamma_BPL(
        a1, a2, gamma_b,
        gamma_min=gamma_min,
        gamma_max=gamma_max,
    )

    print(f"Mean Lorentz factor: {mean_gamma:.2e}")

Similarly, to compute the effective number density of radiating electrons in a broken
power-law distribution with normalization :math:`N_0 = 10^5 \, \rm cm^{-3}`, one could use:

.. code-block:: python

    from triceratops.radiation.synchrotron.microphysics import (
        compute_BPL_effective_number_density
    )

    N0 = 1e5  # cm^-3

    n_eff = compute_BPL_effective_number_density(
        N0,
        a1=2.0,
        a2=3.5,
        gamma_b=1e4,
        gamma_min=1e2,
        gamma_max=1e7,
    )

    print(f"Effective radiating density: {n_eff:.2e}")

Maxwell-Jüttner Distributions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A **Maxwell-Jüttner distribution** is the relativistic thermal equilibrium distribution.
In the ultra-relativistic limit it takes the form

.. math::

    N(\gamma) = \frac{N_{\rm therm}}{2\Theta^3}\,\gamma^2\,e^{-\gamma/\Theta},

where:

- :math:`\Theta = kT / (m_e c^2)` is the dimensionless electron temperature,
- :math:`N_{\rm therm}` is the normalization (total number density of thermal electrons).

This distribution is appropriate for modeling thermalized electron populations, such as
shock-heated plasmas or dense radiative environments where electrons equilibrate to a
well-defined temperature.

.. note::

    As with the power-law and broken power-law cases, Triceratops consistently treats the
    Lorentz factor :math:`\gamma` as the independent variable. The Maxwell-Jüttner
    distribution is fully specified by :math:`\Theta` and its normalization
    :math:`N_{\rm therm}`.

.. hint::

    From a modeling perspective, the Maxwell-Jüttner distribution is fundamentally different
    from power-law distributions in that its shape is fixed by the temperature. As a result,
    synchrotron observables depend only on the normalization and temperature, rather than on
    higher-order moments of a free functional form.

The mean energy density of the Maxwell-Jüttner population is approximated by
:footcite:p:`1998ApJ...498..313G`

.. math::

    U_e \approx N_{\rm therm}\,m_e c^2\,\Theta\,\frac{6 + 15\Theta}{4 + 5\Theta},

which smoothly interpolates between the non-relativistic and ultra-relativistic limits.

Mixed Thermal + Non-thermal Populations
"""""""""""""""""""""""""""""""""""""""""

Many astrophysical sources are best described by a **mixed** population containing both
a thermalized component and a non-thermal power-law tail. Triceratops parameterizes this
split with a single parameter :math:`\delta \in [0,1]`:

.. math::

    \varepsilon_{E,\rm therm} = \delta\,\varepsilon_E,
    \qquad
    \varepsilon_{E,\rm PL} = (1 - \delta)\,\varepsilon_E.

Each component's normalization is then computed independently via equipartition.
Setting :math:`\delta = 1` recovers a pure thermal population; :math:`\delta = 0`
recovers the standard power-law-only case.
.. rubric:: API Reference

*current module*: :mod:`triceratops.radiation.synchrotron.microphysics`

.. tab-set::

    .. tab-item:: High-Level API

        The following high-level helper functions are provided for working with
        Maxwell-Jüttner and mixed thermal + non-thermal electron populations:

        .. autosummary::
           :nosignatures:

           compute_electron_gamma_MJD_moment
           compute_mean_gamma_MJD
           compute_mean_energy_MJD
           compute_MJD_total_number_density
           compute_MJD_effective_number_density
           compute_MJD_norm_from_magnetic_field
           compute_MJD_norm_from_thermal_energy_density
           compute_MJD_and_PL_norm_from_magnetic_field
           compute_MJD_and_PL_norm_from_thermal_energy_density
           compute_bol_emissivity_MJD
           compute_bol_emissivity_MJD_from_thermal_energy_density
           get_maxwell_juttner_distribution

    .. tab-item:: Low-Level API

        Mirroring the high-level interface, the following low-level functions are provided
        for Maxwell-Jüttner distributions and mixed populations:

        - ``_opt_compute_MJD_moment``, which computes the closed-form Maxwell-Jüttner
          moment shape factor :math:`S^{(\ell)}(\Theta)`.

        - ``_opt_compute_MJD_n_eff``, which computes the effective radiating number
          density :math:`n_{\rm eff} = \int \gamma^2 N(\gamma)\,d\gamma`.

        - ``_opt_normalize_MJD_from_magnetic_field``, which computes the normalization
          :math:`N_{\rm therm}` given a magnetic field strength and equipartition parameters.

        - ``_opt_normalize_MJD_from_thermal_energy_density``, which computes
          :math:`N_{\rm therm}` directly from a thermal energy density.

        - ``_opt_normalize_MJD_and_PL_from_magnetic_field``, which computes both
          :math:`N_{\rm therm}` and the power-law normalization for mixed populations.

        - ``_opt_normalize_MJD_and_PL_from_thermal_energy_density``, which performs the
          same calculation using the thermal energy density as input.

        - ``_opt_compute_bol_emiss_MJD_from_magnetic_field``, which computes the
          bolometric synchrotron emissivity for a Maxwell-Jüttner population from
          an explicit magnetic field and thermal normalization.

        - ``_opt_compute_bol_emiss_MJD_from_thermal_energy_density_full``, which computes
          the bolometric synchrotron emissivity for a Maxwell-Jüttner population using
          the full equipartition closure.

.. rubric:: Examples

As with the power-law and broken power-law cases, these functions are primarily intended
for use by higher-level modeling routines, but can also be used directly.

For example, to compute the normalization of a Maxwell-Jüttner distribution from a magnetic
field strength:

.. code-block:: python

    import astropy.units as u
    from triceratops.radiation.synchrotron.microphysics import (
        compute_MJD_norm_from_magnetic_field,
    )

    B = 0.5 * u.G
    Theta = 10.0
    epsilon_B = 0.01
    epsilon_E = 0.1

    N_therm = compute_MJD_norm_from_magnetic_field(
        B=B,
        Theta=Theta,
        epsilon_B=epsilon_B,
        epsilon_E=epsilon_E,
    )

    print(f"Thermal number density: {N_therm:.3e}")

Similarly, to compute the normalization of a mixed thermal + power-law population:

.. code-block:: python

    import astropy.units as u
    from triceratops.radiation.synchrotron.microphysics import (
        compute_MJD_and_PL_norm_from_thermal_energy_density,
    )

    u_therm = 1e-2 * u.erg / u.cm**3
    Theta = 10.0
    p = 3.0
    delta = 0.3
    epsilon_E = 0.1

    N_therm, N0_pl = compute_MJD_and_PL_norm_from_thermal_energy_density(
        u_therm=u_therm,
        Theta=Theta,
        p=p,
        delta=delta,
        epsilon_E=epsilon_E,
        gamma_min=1.0,
        gamma_max=1e8,
    )

    print(f"Thermal density: {N_therm:.3e}")
    print(f"PL normalization: {N0_pl:.3e}")

----

Equipartition Closure
----------------------

.. hint::

    See :ref:`synch_equipartition_theory` for the corresponding documentation on synchrotron theory.

The most common closure mechanism used in the literature (and the only primarily used in Triceratops) is the
**equipartition closure**. In this framework, we assume that a fixed fraction of the thermal energy density
behind the shock is partitioned into relativistic electrons and magnetic fields. Specifically, we define two
dimensionless parameters:

- :math:`\epsilon_e`, the fraction of thermal energy in relativistic electrons,
- :math:`\epsilon_B`, the fraction of thermal energy in magnetic fields.

As a result, we can (given knowledge of the shock dynamics) compute the normalization of the electron
distribution and the magnetic field strength directly from the macrophysical quantities. This approach
has been widely used in the literature to model synchrotron emission from a variety of astrophysical
sources (e.g. :footcite:t:`Margutti2019COW`,
:footcite:t:`demarchiRadioAnalysisSN2004C2022`, :footcite:t:`wuDelayedRadioEmission2025`, etc.)

In Triceratops, the helper functions for performing equipartition closure are provided in the
:mod:`triceratops.radiation.synchrotron.microphysics` module. Before highlighting some of these functions, we'll provide
the API in the tab-set below:

.. tab-set::

    .. tab-item:: High-Level API

        The following high-level helper functions are provided for performing equipartition closure:

        .. currentmodule:: triceratops.radiation.synchrotron.microphysics
        .. autosummary::
           :nosignatures:

           compute_equipartition_magnetic_field
           compute_PL_norm_from_magnetic_field
           compute_PL_norm_from_thermal_energy_density
           compute_BPL_norm_from_magnetic_field
           compute_BPL_norm_from_thermal_energy_density
           compute_MJD_norm_from_magnetic_field
           compute_MJD_norm_from_thermal_energy_density
           compute_MJD_and_PL_norm_from_magnetic_field
           compute_MJD_and_PL_norm_from_thermal_energy_density
           compute_bol_emissivity
           compute_bol_emissivity_from_thermal_energy_density
           compute_bol_emissivity_BPL
           compute_bol_emissivity_BPL_from_thermal_energy_density


    .. tab-item:: Low-Level API

        Mirroring the high-level functions, the following low-level functions are provided for
        performing equipartition closure:

        - ``_opt_normalize_PL_from_magnetic_field``, which computes the normalization of a power-law
          electron distribution given a magnetic field strength and the equipartition parameter
          :math:`\epsilon_e`.
        - ``_opt_normalize_PL_from_thermal_energy_density``, which computes the normalization of a
          power-law electron distribution given a thermal energy density and the equipartition parameter
          :math:`\epsilon_e`.
        - ``_opt_normalize_energy_PL_from_magnetic_field``, which computes the normalization of a power-law
          electron energy distribution given a magnetic field strength and the equipartition parameter
          :math:`\epsilon_e`.
        - ``_opt_normalize_energy_PL_from_thermal_energy_density``, which computes the normalization of a
          power-law electron energy distribution given a thermal energy density and the equipartition parameter
          :math:`\epsilon_e`.
        - ``_opt_compute_equipartition_magnetic_field``, which computes the magnetic field strength given a
          thermal energy density and the equipartition parameter :math:`\epsilon_B`.
        - ``_opt_compute_bol_emiss_from_magnetic_field``, which computes the bolometric synchrotron emissivity
          given a magnetic field strength and a power-law electron distribution normalization.
        - ``_opt_compute_bol_emiss_from_thermal_energy_density``, which computes the bolometric synchrotron
          emissivity given a thermal energy density and the equipartition parameters :math:`\epsilon_e` and
          :math:`\epsilon_B`.
        - ``_opt_compute_bol_emiss_from_thermal_energy_density_full``, which computes the bolometric synchrotron
          emissivity given a thermal energy density and the equipartition parameters :math:`\epsilon_e` and
          :math:`\epsilon_B`, **without requiring knowledge of the PL normalization**.

        There are corresponding BPL functions for each of these as well.

The most important of these functions is :func:`compute_PL_norm_from_thermal_energy_density`, which computes
the normalization of a power-law
electron distribution given a thermal energy density and the equipartition parameter :math:`\epsilon_e`. This
is generally the correct way to convert macrophysical quantities into microphysical ones when building
synchrotron models.

For use in models, users should instead use the low-level CGS version of the function
(``_opt_normalize_PL_from_thermal_energy_density``), which is optimized for performance.

.. rubric:: Examples

For the most part, these functions are intended for internal use by higher-level modeling functions. However,
they can also be used directly when building custom models. For example, let's compute the equipartition magnetic
field strength from a thermal energy density of :math:`u_{\rm th} = 10^{-2} \, \rm erg \, cm^{-3}`
and an equipartition parameter of :math:`\epsilon_B = 0.01`. We can do this using the following code:

.. code-block:: python

    import astropy.units as u
    from triceratops.radiation.synchrotron.microphysics import (
        compute_equipartition_magnetic_field
    )

    u_therm = 1e-2 * u.erg / u.cm**3
    epsilon_B = 0.01

    B = compute_equipartition_magnetic_field(
        u_therm=u_therm,
        epsilon_B=epsilon_B,
    )

    print(f"Magnetic field strength: {B:.3e}")

    >>> 5.013e-02 G

If we want instead to compute the bolometric luminosity assuming :math:`\epsilon_e = \epsilon_B = 0.01`, and :math:`p=2.5`,
we can use the following code:

.. code-block:: python

    import astropy.units as u
    from triceratops.radiation.synchrotron.microphysics import (
        compute_bol_emissivity_from_thermal_energy_density
    )

    u_therm = 1e-2 * u.erg / u.cm ** 3
    epsilon_e = 0.01
    epsilon_B = 0.01
    p = 3

    j_bol = compute_bol_emissivity_from_thermal_energy_density(
        u_therm=u_therm,
        epsilon_E=epsilon_e,
        epsilon_B=epsilon_B,
        p=p,
        gamma_min=1,
        gamma_max=1e8
    )

    print(f"Bolometric emissivity: {j_bol:.3e}")
    >>> 5.983e-15 erg / (s cm3)
