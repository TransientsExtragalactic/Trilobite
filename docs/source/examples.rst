.. _gallery_examples:

###########################
Triceratops Example Gallery
###########################

The example gallery contains runnable, self-contained scripts that demonstrate the key
capabilities of Triceratops. Examples are organized by topic so you can jump directly to what
you need — from loading data and running inference, to exploring synchrotron physics and
accretion disk dynamics.

Each script is fully executable, produces annotated output, and can be downloaded as a Jupyter
notebook. Browse a section below to get started.

----

Observational Workflow
======================

These galleries cover the data-to-results pipeline: loading and inspecting photometry,
then fitting physical models to it with Bayesian inference.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Data
      :link: auto_examples/data/index
      :link-type: doc

      **Loading and handling observational data.**

      Load radio photometry from FITS files, inspect tables, group observations into
      temporal epochs, and plot light curves using the :mod:`~triceratops.data` subpackage.

   .. grid-item-card:: Optical Filters and Photometry
      :link: auto_examples/photometry/index
      :link-type: doc

      **Filter construction, batched convolution, and magnitude systems.**

      Build photometric filters, assemble them into a
      :class:`~triceratops.utils.phot_utils.FilterBundle` for MCMC-optimised matrix-multiply
      convolution, and convert between flux density and AB/ST magnitudes.

   .. grid-item-card:: Inference
      :link: auto_examples/inference/index
      :link-type: doc

      **Bayesian parameter estimation.**

      End-to-end MCMC workflows — single- and multi-epoch SED fitting, upper-limit handling,
      shock-parameter recovery, and posterior propagation through closure relations.
      Built on the :mod:`~triceratops.inference` subpackage.

----

Physical Models
===============

These galleries demonstrate the high-level model objects in Triceratops — from self-similar
shock models for radio supernovae to TDE afterglows and accretion disk dynamics.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Forward Modeling
      :link: auto_examples/modeling/index
      :link-type: doc

      **Using Triceratops physical models.**

      Evaluate models directly to generate synthetic light curves and SEDs. Covers
      Chevalier shock models, FFA fadeout, phenomenological light curves, TDE afterglows,
      and Type IIn supernovae in dense CSM.

   .. grid-item-card:: Shock Dynamics
      :link: auto_examples/dynamics/index
      :link-type: doc

      **Shock propagation and the numerical engine.**

      Low-level examples from :mod:`~triceratops.dynamics`: Rankine–Hugoniot jump conditions,
      blast-wave evolution in power-law density profiles, and the time-stepping shock engine.

   .. grid-item-card:: Accretion Disk Models
      :link: auto_examples/accretion/index
      :link-type: doc

      **One-zone accretion disk dynamics.**

      Time-dependent disk evolution — thermal S-curves, limit cycles, fallback disks, and
      advective solutions — using the :mod:`~triceratops.dynamics.accretion.one_zone` Cython
      integrator.

----

Radiative Physics
=================

These galleries cover the low-level radiative physics machinery in Triceratops. The synchrotron
gallery is organized into four progressive sub-sections.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Synchrotron Emission
      :link: auto_examples/synchrotron/index
      :link-type: doc

      **Synchrotron radiation from fundamentals to closures.**

      Four sub-sections: kernel functions and the ν–γ–B relation; SED model hierarchy and
      spectral regimes; electron cooling (synchrotron and IC); forward/inverse closure
      relations and equipartition analysis.

   .. grid-item-card:: Opacity Laws
      :link: auto_examples/opacity/index
      :link-type: doc

      **Kramers, electron scattering, OPAL table opacity.**

      Demonstrations of the :mod:`~triceratops.radiation.opacity` module — the canonical
      :math:`\kappa_R(T,\rho)` plot for solar composition from the bundled OPAL table,
      showing the electron-scattering plateau, Kramers rise, and iron-group peak.

   .. grid-item-card:: Free-Free Emission
      :link: auto_examples/free_free/index
      :link-type: doc

      **Thermal bremsstrahlung from ionized gas.**

      Free-free emissivity and flux density in the X-ray regime, optical depth
      profiles for wind and shell CSM geometries, and FFA attenuation of
      synchrotron SEDs using the :mod:`~triceratops.radiation.free_free` module.

.. toctree::
   :hidden:

   auto_examples/data/index
   auto_examples/photometry/index
   auto_examples/inference/index
   auto_examples/modeling/index
   auto_examples/dynamics/index
   auto_examples/synchrotron/index
   auto_examples/accretion/index
   auto_examples/opacity/index
   auto_examples/free_free/index
