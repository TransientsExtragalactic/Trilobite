.. _model_directory:

=========================================
Trilobite Model Directory
=========================================

Trilobite provides an ever-expanding collection of forward models for astrophysical inference,
phenomenological curve fitting, and physically motivated transient modeling. Models range from lightweight
mathematical forms to physically grounded shock and emission models, and all share the unified
:class:`~trilobite.models.core.base.Model` interface: consistent parameter handling, unit support, and
compatibility with likelihood and inference tools.

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
      17 models

   .. grid-item-card:: SEDs
      :link: sed_models
      :link-type: ref
      :text-align: center
      :class-card: sd-shadow-sm

      Physical multi-wavelength spectra

      +++
      5 models

   .. grid-item-card:: Supernovae
      :link: supernova_models
      :link-type: ref
      :text-align: center
      :class-card: sd-shadow-sm

      Shock and CSM interaction

      +++
      1 model

   .. grid-item-card:: LFBOTs
      :link: lfbot_models
      :link-type: ref
      :text-align: center
      :class-card: sd-shadow-sm

      Two-component radio emission

      +++
      1 model

   .. grid-item-card:: GRBs
      :link: grb_models
      :link-type: ref
      :text-align: center
      :class-card: sd-shadow-sm

      Prompt and afterglow emission

      +++
      1 model

   .. grid-item-card:: Optical
      :link: optical_models
      :link-type: ref
      :text-align: center
      :class-card: sd-shadow-sm

      Filter-convolved photometry

      +++
      4 models

.. _generic_models:

Generic Models
--------------

Flexible, modular models that are not tied to a specific physical scenario. They are ideal for rapid
prototyping, exploratory analysis, and building custom likelihoods.

.. dropdown:: Curve-fitting models :bdg-primary:`4`
   :name: curve_fitting_models
   :open:
   :class-container: trilobite-dropdown

   Flexible mathematical forms for fitting data.

   .. currentmodule:: trilobite.models.generic.curves

   .. autosummary::
      :nosignatures:

      BrokenPowerLaw
      SmoothedBrokenPowerLaw
      TripleBrokenPowerLaw
      SmoothedTripleBrokenPowerLaw

.. dropdown:: Light curve models :bdg-primary:`10`
   :name: light_curve_models
   :class-container: trilobite-dropdown

   Phenomenological models for time-domain transient analysis and pulse-shape modeling.

   .. currentmodule:: trilobite.models.generic.light_curve

   .. autosummary::
      :nosignatures:

      FRED
      GeneralizedFRED
      GaussianPulse
      LogNormalPulse
      BrokenPowerLawTime
      SmoothedBrokenPowerLawTime
      ExponentialRisePowerLawDecay
      NorrisPulse
      WeibullPulse
      LogisticPulse

.. dropdown:: Evolving SED models :bdg-primary:`3`
   :name: evolving_models
   :class-container: trilobite-dropdown

   Phenomenological SED models in which the spectral parameters evolve in time.

   .. currentmodule:: trilobite.models.generic.evolving_seds

   .. autosummary::
      :nosignatures:

      ~evolving_sbpl.PL_Evolving_SBPL_Model
      ~evolving_sbpl.BPL_Evolving_SBPL_Model
      ~evolving_sbpl.TripleBPL_Evolving_SBPL_Model

.. _sed_models:

SED Models
----------

Full-physics spectral energy distributions for a range of emission mechanisms and physical assumptions,
suited to multi-wavelength fitting and inference.

.. dropdown:: Synchrotron :bdg-primary:`5`
   :open:
   :class-container: trilobite-dropdown

   .. currentmodule:: trilobite.models.SEDs.synchrotron

   .. autosummary::
      :nosignatures:

      SSA_Cooling_SynchrotronSEDModel
      SSA_SynchrotronSEDModel
      Cooling_SynchrotronSEDModel
      SynchrotronSEDModel
      Synchrotron_SSA_SBPL_Model

.. _supernova_models:

Supernova Models
----------------

Physically motivated shock dynamics, radiation processes, and circumstellar interaction physics.

.. dropdown:: Self-similar shock models :bdg-primary:`1`
   :open:
   :class-container: trilobite-dropdown

   .. currentmodule:: trilobite.models.supernovae

   .. autosummary::
      :nosignatures:

      chevalier_shock.ChevalierShockModel

.. _lfbot_models:

LFBOT Models
------------

Circumstellar interaction of luminous fast blue optical transients (LFBOTs). The two-component model follows an
equatorial and a polar sightline through an axisymmetric CSM, each with its own thin-shell shock, synchrotron
emission and free-free attenuation. It provides both a paired ``(time, frequency)`` path for inference and a
``tabulate`` path for analysis and plotting.

.. dropdown:: Two-component synchrotron model :bdg-primary:`1`
   :open:
   :class-container: trilobite-dropdown

   .. currentmodule:: trilobite.models.LFBOTs.diggins26

   .. autosummary::
      :nosignatures:

      Diggins26_TwoComponentModel

.. _grb_models:

GRB Models
----------

Spectral and temporal properties of gamma-ray burst (GRB) emission, from empirical spectral characterisation to
physically motivated jet and afterglow models.

.. dropdown:: Prompt-emission spectral models :bdg-primary:`1`
   :open:
   :class-container: trilobite-dropdown

   .. currentmodule:: trilobite.models.GRBs

   .. autosummary::
      :nosignatures:

      band.BandFunctionModel

.. _optical_models:

Optical Models
--------------

Filter-convolved flux densities for multi-band photometric observations. Optical models are built on the
:class:`~trilobite.models.core.optical.OpticalModel` and
:class:`~trilobite.models.core.optical.OpticalEpochModel` abstract base classes, which handle
:class:`~trilobite.utils.phot_utils.FilterBundle` convolution, time deduplication, and band-index resolution
automatically. See :ref:`the optical model pattern <optical_model_pattern>` for a guide to implementing custom optical models.

.. dropdown:: Blackbody models :bdg-primary:`2`
   :open:
   :class-container: trilobite-dropdown

   .. currentmodule:: trilobite.models.SEDs.blackbody_optical

   .. autosummary::
      :nosignatures:

      BlackbodyOpticalModel
      BlackbodyOpticalEpochModel

.. dropdown:: Abstract base classes :bdg-secondary:`2`
   :class-container: trilobite-dropdown

   .. currentmodule:: trilobite.models.core.optical

   .. autosummary::
      :nosignatures:

      OpticalModel
      OpticalEpochModel

.. _extending_model_library:

Extending the Model Library
---------------------------

Every model subclasses :class:`~trilobite.models.core.base.Model`, declaring its ``PARAMETERS`` and ``VARIABLES``
and implementing the forward model. The :ref:`models_overview` walks through a complete worked example
(:ref:`forward_modeling`, :ref:`model_parameters`) and covers specialized patterns such as
:ref:`the optical model pattern <optical_model_pattern>`.
