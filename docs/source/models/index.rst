.. raw:: html

   <div class="page-header">

.. image:: /images/logo.svg
   :class: page-header-logo
   :alt: Trilobite logo

.. raw:: html

   <div class="page-header-text">

.. _model_directory:

==================
Models
==================

.. raw:: html

   <p class="page-header-tagline">A searchable catalog of every forward model in Trilobite.</p>
   <p class="page-header-lede">
      Browse GRB, supernova, TDE and LFBOT models, each labeled with its development status.
   </p>
   </div>
   </div>

Trilobite provides an ever-expanding collection of forward models for astrophysical inference,
phenomenological curve fitting, and physically motivated transient modeling. Models range from lightweight
mathematical forms to physically grounded shock and emission models, and all share the unified
:class:`~trilobite.models.core.base.Model` interface: consistent parameter handling, unit support, and
compatibility with likelihood and inference tools.

.. toctree::
   :hidden:

   overview

.. model-search::

.. grid:: 1 1 2 2
   :gutter: 2

   .. grid-item-card:: New to the framework?
      :link: models_overview
      :link-type: ref
      :class-card: sd-shadow-sm trilobite-callout

      Start with the :ref:`models_overview`: design philosophy, ``ModelParameter`` and ``ModelVariable``,
      unit handling, inference integration, and how to write your own model.

   .. grid-item-card:: Can't find your model?
      :link: extending_model_library
      :link-type: ref
      :class-card: sd-shadow-sm trilobite-callout

      Most models are designed to be subclassed. Phenomenological models adapt with minimal effort, and
      physical models expose their core components so assumptions can be modified cleanly.

.. rst-class:: trilobite-model-browse

Development status
------------------

Every model carries a development stage, shown as a badge in the tables below and at the top of its API page. The
stage is declared on the model class with the ``STATUS`` attribute (see
:class:`~trilobite.models.core.status.ModelStatus`), so it can never drift from the code. Instantiating an
experimental or deprecated model emits a warning.

.. model-status-legend::

.. rst-class:: trilobite-model-browse

Browse by category
------------------

.. grid:: 2 3 3 3
   :gutter: 2
   :class-container: trilobite-jump-grid

   .. grid-item-card:: Generic
      :link: generic_models
      :link-type: ref
      :text-align: center
      :class-card: sd-shadow-sm

      Curve fits, pulses, evolving SEDs

      +++
      :model-count:`Generic`

   .. grid-item-card:: SEDs
      :link: sed_models
      :link-type: ref
      :text-align: center
      :class-card: sd-shadow-sm

      Physical multi-wavelength spectra

      +++
      :model-count:`SEDs`

   .. grid-item-card:: Supernovae
      :link: supernova_models
      :link-type: ref
      :text-align: center
      :class-card: sd-shadow-sm

      Shock and CSM interaction

      +++
      :model-count:`Supernovae`

   .. grid-item-card:: LFBOTs
      :link: lfbot_models
      :link-type: ref
      :text-align: center
      :class-card: sd-shadow-sm

      Two-component radio emission

      +++
      :model-count:`LFBOTs`

   .. grid-item-card:: GRBs
      :link: grb_models
      :link-type: ref
      :text-align: center
      :class-card: sd-shadow-sm

      Prompt and afterglow emission

      +++
      :model-count:`GRBs`

   .. grid-item-card:: Optical
      :link: optical_models
      :link-type: ref
      :text-align: center
      :class-card: sd-shadow-sm

      Filter-convolved photometry

      +++
      :model-count:`Optical`

.. _generic_models:

.. rst-class:: trilobite-model-browse

Generic Models
--------------

Flexible, modular models that are not tied to a specific physical scenario. They are ideal for rapid
prototyping, exploratory analysis, and building custom likelihoods.

.. dropdown:: Curve-fitting models
   :open:
   :class-container: trilobite-dropdown

   .. model-table:: trilobite.models.generic.curves

.. dropdown:: Light curve models
   :class-container: trilobite-dropdown

   .. model-table:: trilobite.models.generic.light_curve

.. dropdown:: Evolving SED models
   :class-container: trilobite-dropdown

   .. model-table:: trilobite.models.generic.evolving_seds

.. _sed_models:

.. rst-class:: trilobite-model-browse

SED Models
----------

Full-physics spectral energy distributions for a range of emission mechanisms and physical assumptions,
suited to multi-wavelength fitting and inference.

.. dropdown:: Synchrotron
   :open:
   :class-container: trilobite-dropdown

   .. model-table:: trilobite.models.SEDs.synchrotron

.. _supernova_models:

.. rst-class:: trilobite-model-browse

Supernova Models
----------------

Physically motivated shock dynamics, radiation processes, and circumstellar interaction physics.

.. dropdown:: Self-similar shock models
   :open:
   :class-container: trilobite-dropdown

   .. model-table:: trilobite.models.supernovae

.. _lfbot_models:

.. rst-class:: trilobite-model-browse

LFBOT Models
------------

Circumstellar interaction of luminous fast blue optical transients (LFBOTs). The two-component model follows an
equatorial and a polar sightline through an axisymmetric CSM, each with its own thin-shell shock, synchrotron
emission and free-free attenuation. It provides both a paired ``(time, frequency)`` path for inference and a
``tabulate`` path for analysis and plotting.

.. dropdown:: Two-component synchrotron model
   :open:
   :class-container: trilobite-dropdown

   .. model-table:: trilobite.models.LFBOTs

.. _grb_models:

.. rst-class:: trilobite-model-browse

GRB Models
----------

Spectral and temporal properties of gamma-ray burst (GRB) emission, from empirical spectral characterisation to
physically motivated jet and afterglow models.

.. dropdown:: Prompt-emission spectral models
   :open:
   :class-container: trilobite-dropdown

   .. model-table:: trilobite.models.GRBs

.. _optical_models:

.. rst-class:: trilobite-model-browse

Optical Models
--------------

Filter-convolved flux densities for multi-band photometric observations. Optical models are built on the
:class:`~trilobite.models.core.optical.OpticalModel` and
:class:`~trilobite.models.core.optical.OpticalEpochModel` abstract base classes, which handle
:class:`~trilobite.utils.phot_utils.FilterBundle` convolution, time deduplication, and band-index resolution
automatically. See :ref:`the optical model pattern <optical_model_pattern>` for a guide to implementing custom optical models.

.. dropdown:: Blackbody models
   :open:
   :class-container: trilobite-dropdown

   .. model-table:: trilobite.models.SEDs.blackbody_optical

.. dropdown:: TDE light-curve models
   :class-container: trilobite-dropdown

   .. model-table:: trilobite.models.TDEs

.. dropdown:: Abstract base classes
   :class-container: trilobite-dropdown

   .. currentmodule:: trilobite.models.core.optical

   .. autosummary::
      :nosignatures:

      OpticalModel
      OpticalEpochModel

.. _extending_model_library:

.. rst-class:: trilobite-model-browse

Extending the Model Library
---------------------------

Every model subclasses :class:`~trilobite.models.core.base.Model`, declaring its ``PARAMETERS`` and ``VARIABLES``
and implementing the forward model. The :ref:`models_overview` walks through a complete worked example
(:ref:`forward_modeling`, :ref:`model_parameters`) and covers specialized patterns such as
:ref:`the optical model pattern <optical_model_pattern>`.
