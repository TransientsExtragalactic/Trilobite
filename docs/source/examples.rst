.. _gallery_examples:

.. raw:: html

.. raw:: html

   <style>
     .bd-article h1 {
         font-size: 2.6rem;
         color: #1f7a8c;
         margin-bottom: 0.5rem;
     }
   </style>

   <div style="
    display: flex;
    align-items: center;
    gap: 2.5rem;
    padding: 2.5rem 0 2.4rem 0;   /* increase bottom padding */
    border-bottom: 1px solid #e5e7eb;
    margin-bottom: 1.8rem;        /* ← THIS is key */
    ">

       <img src="_images/logo.png" style="width: 25%; height: auto;">

       <div style="max-width: 620px;">

================================
Example Gallery
================================

.. raw:: html

           <p style="
               margin: 0.4rem 0 0 0;
               font-size: 1rem;
               color: #666;
           ">
               Practical examples to get started with Trilobite.
           </p>

           <p style="
               margin: 1rem 0 0 0;
               font-size: 1.1rem;
               line-height: 1.6;
               color: #3a4a52;
           ">
               Explore <strong>runnable, self-contained examples</strong> —
               from <strong>loading observational data</strong> and
               <strong>running inference</strong> to modeling
               <strong>synchrotron emission</strong> and
               <strong>accretion disk dynamics</strong>.
           </p>

       </div>
   </div>

Working with Data
=================

Load and inspect your observations before fitting anything.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Observational Data
      :link: auto_examples/data/index
      :link-type: doc

      **Load, group, and visualize photometric observations.**

      Load radio photometry from FITS files, group observations into temporal epochs,
      and plot light curves using the :mod:`~trilobite.data` subpackage.

   .. grid-item-card:: Optical Filters and Photometry
      :link: auto_examples/photometry/index
      :link-type: doc

      **Build and convolve optical filters for broadband photometry.**

      Construct filters, assemble a :class:`~trilobite.utils.phot_utils.FilterBundle`
      for MCMC-optimised matrix-multiply convolution, and convert between flux density
      and AB/ST magnitudes.

----

Understanding the Physics
==========================

Learn the radiative and dynamical building blocks that every Trilobite model is built from.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Synchrotron Emission
      :link: auto_examples/synchrotron/index
      :link-type: doc

      **Understand the synchrotron emission that drives radio transients.**

      Four sub-sections: kernel functions and the :math:`\nu`--:math:`\gamma`--:math:`B`
      relation; SED spectral regimes; electron cooling (synchrotron and IC); closure
      relations and equipartition analysis.

   .. grid-item-card:: Free-Free Emission
      :link: auto_examples/free_free/index
      :link-type: doc

      **Compute free-free emission and absorption in ionized gas.**

      Thermal bremsstrahlung emissivity, flux density, optical depth profiles for wind
      and shell CSM geometries, and FFA attenuation of synchrotron SEDs using
      :mod:`~trilobite.radiation.free_free`.

   .. grid-item-card:: Opacity Laws
      :link: auto_examples/opacity/index
      :link-type: doc

      **Evaluate grey and frequency-dependent opacity laws.**

      Kramers, electron scattering, and OPAL/TOPS Rosseland mean opacity from the
      bundled tables using :mod:`~trilobite.radiation.opacity`.

   .. grid-item-card:: Shock Dynamics
      :link: auto_examples/dynamics/index
      :link-type: doc

      **Step through shock propagation at the integrator level.**

      The numerical shock engine, blast-wave evolution in power-law density profiles,
      and the time-stepping integrator from :mod:`~trilobite.dynamics`.

----

Building Forward Models
=======================

Assemble physical components into models that generate synthetic observables.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Physical Models
      :link: auto_examples/modeling/index
      :link-type: doc

      **Generate synthetic light curves and SEDs with high-level model objects.**

      Chevalier shock models, FFA fadeout, TDE afterglows, Type IIn supernovae in
      dense CSM, and phenomenological light curves.

   .. grid-item-card:: Accretion Disk Models
      :link: auto_examples/accretion/index
      :link-type: doc

      **Evolve accretion disks through their thermodynamic phases.**

      Thermal S-curves, limit cycles, fallback disks, and advective solutions using
      the :mod:`~trilobite.dynamics.accretion.one_zone` Cython integrator.

----

Running Inference
=================

Fit models to data and recover physical parameters from your observations.

.. grid:: 1
   :gutter: 3

   .. grid-item-card:: Inference and Parameter Estimation
      :link: auto_examples/inference/index
      :link-type: doc

      **Fit a physical model to data and recover posterior distributions over its parameters.**

      End-to-end MCMC workflows — single- and multi-epoch SED fitting, upper-limit
      handling, shock-parameter recovery, and posterior propagation through closure
      relations. Built on the :mod:`~trilobite.inference` subpackage.

.. toctree::
   :hidden:

   auto_examples/data/index
   auto_examples/photometry/index
   auto_examples/synchrotron/index
   auto_examples/free_free/index
   auto_examples/opacity/index
   auto_examples/dynamics/index
   auto_examples/modeling/index
   auto_examples/accretion/index
   auto_examples/inference/index
