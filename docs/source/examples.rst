.. raw:: html

   <div class="page-header">

.. image:: /images/logo.svg
   :class: page-header-logo
   :alt: Trilobite logo

.. raw:: html

   <div class="page-header-text">

.. _gallery_examples:

===================
Example Gallery
===================

.. raw:: html

   <p class="page-header-tagline">Learn by running real workflows.</p>
   <p class="page-header-lede">
      Runnable, self-contained examples covering the full Trilobite workflow, from raw observations to physical parameters. Follow the path below, or search for a specific example or science case.
   </p>
   </div>
   </div>

.. raw:: html

   <div id="gallery-search"></div>

.. rst-class:: trilobite-stage

1. Observe
----------

Get data into Trilobite and connect model SEDs to broadband photometry, before fitting anything.

.. grid:: 1 1 2 2
   :gutter: 2
   :class-container: trilobite-topic-grid

   .. grid-item-card:: Observational Data
      :link: auto_examples/data/index
      :link-type: doc
      :class-card: sd-shadow-sm

      Radio and optical photometry and light-curve containers from :mod:`trilobite.data`, per-band
      extraction, and conversion to :class:`~trilobite.data.core.InferenceData`.

      +++
      7 examples

   .. grid-item-card:: Optical Filters and Photometry
      :link: auto_examples/photometry/index
      :link-type: doc
      :class-card: sd-shadow-sm

      Build filters, assemble a :class:`~trilobite.utils.phot_utils.FilterBundle` for fast convolution in MCMC
      loops, load SVO profiles, and convert between flux density and AB/ST magnitudes.

      +++
      4 examples

.. rst-class:: trilobite-stage

2. Understand the physics
-------------------------

The radiative, shock, and accretion building blocks that every Trilobite model is assembled from.

.. grid:: 1 1 2 3
   :gutter: 2
   :class-container: trilobite-topic-grid

   .. grid-item-card:: Synchrotron Emission
      :link: auto_examples/synchrotron/index
      :link-type: doc
      :class-card: sd-shadow-sm

      Electron distributions, SED regimes with cooling and self-absorption, numerical engines for
      inhomogeneous and off-axis sources, and closure relations and equipartition.

      +++
      15 examples

   .. grid-item-card:: Free-Free Emission
      :link: auto_examples/free_free/index
      :link-type: doc
      :class-card: sd-shadow-sm

      Gaunt factors, thermal bremsstrahlung flux, and free-free absorption of synchrotron SEDs.

      +++
      3 examples

   .. grid-item-card:: Opacity Laws
      :link: auto_examples/opacity/index
      :link-type: doc
      :class-card: sd-shadow-sm

      Analytic grey laws and the tabulated OPAL and TOPS Rosseland means from
      :mod:`trilobite.radiation.opacity`.

      +++
      2 examples

   .. grid-item-card:: Shock Dynamics
      :link: auto_examples/dynamics/index
      :link-type: doc
      :class-card: sd-shadow-sm

      Rankine-Hugoniot jump conditions, self-similar solutions (Chevalier, Sedov-Taylor, Blandford-McKee), and
      the numerical thin-shell and mechanical shock engines in :mod:`trilobite.dynamics`.

      +++
      15 examples

   .. grid-item-card:: Accretion Disks
      :link: auto_examples/accretion/index
      :link-type: doc
      :class-card: sd-shadow-sm

      Steady-state Shakura-Sunyaev disks and time-dependent one-zone evolution: equations of state, thermal
      S-curves, fallback accretion, and TDE disk observables.

      +++
      8 examples

.. rst-class:: trilobite-stage

3. Build a model
----------------

Assemble components into a model that predicts observables.

.. grid:: 1
   :gutter: 2
   :class-container: trilobite-topic-grid

   .. grid-item-card:: Forward Modeling
      :link: auto_examples/modeling/index
      :link-type: doc
      :class-card: sd-shadow-sm

      Evaluate high-level :class:`~trilobite.models.core.base.Model` objects directly: Chevalier shock models,
      free-free fadeout, TDE radio afterglows, Type IIn supernovae in dense CSM, and phenomenological light
      curves.

      +++
      6 examples

.. rst-class:: trilobite-stage

4. Fit it to data
-----------------

Priors, likelihoods, MCMC, and posteriors.

.. grid:: 1
   :gutter: 2
   :class-container: trilobite-topic-grid

   .. grid-item-card:: Inference and Parameter Estimation
      :link: auto_examples/inference/index
      :link-type: doc
      :class-card: sd-shadow-sm

      End-to-end workflows built on :mod:`trilobite.inference`: single- and multi-epoch SED fitting, upper
      limits and censored data, nuisance parameters, shock-parameter recovery, and posterior propagation
      through closure relations.

      +++
      8 examples

.. toctree::
   :hidden:

   auto_examples/data/index
   auto_examples/photometry/index
   auto_examples/synchrotron/index
   auto_examples/free_free/index
   auto_examples/opacity/index
   auto_examples/dynamics/index
   auto_examples/accretion/index
   auto_examples/modeling/index
   auto_examples/inference/index
