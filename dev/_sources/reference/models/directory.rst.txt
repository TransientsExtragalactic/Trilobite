.. _model_directory:

=========================================
Triceratops Model Directory
=========================================

Triceratops provides an ever-expanding collection of forward models
designed for astrophysical inference, phenomenological curve fitting,
and physically motivated transient modeling.

The goal of the model library is twofold:

- Provide robust, reusable forward models for inference workflows
- Offer modular building blocks that can be extended, subclassed, or composed

Models span a range of complexity, from lightweight mathematical
curve-fitting forms to physically grounded shock and emission models.
All models share a unified interface through the :class:`~triceratops.models.core.base.Model`
base class, ensuring consistent parameter handling, unit support,
and compatibility with likelihood and inference tools.

If you are new to the framework, we recommend first reading the
:ref:`models_overview` documentation, which explains:

- The design philosophy of the modeling framework
- The role of ``ModelParameter`` and ``ModelVariable``
- How unit handling and coercion are implemented
- How models integrate with inference engines
- Best practices for implementing new models

Below is the catalog of existing models, organized by category. Each model includes a link to
its documentation and source code.

.. hint::

    Even if we do not yet provide the exact model you need,
    many models are designed to be easily subclassed or extended.
    Most phenomenological models can be adapted with minimal effort.
    Physically motivated models are built to expose their core components
    so that researchers can modify assumptions cleanly.

---------------------------------------------------------------------

Quick Navigation
----------------

.. card-carousel:: 3

   .. card:: 📈 Generic Models
        :link: generic_models
        :link-type: ref
        :class-card: sd-shadow-sm

        Flexible, modular models for curve fitting and exploratory analysis.

   .. card:: Physical SEDs
        :link: sed_models
        :link-type: ref
        :class-card: sd-shadow-sm

        Flexible spectral energy distribution models for multi-wavelength fitting.

   .. card:: 💥 Supernova Models
      :link: supernova_models
      :link-type: ref
      :class-card: sd-shadow-sm

      Physically motivated explosion and shock interaction models.

   .. card:: 🌟 GRB Models
      :link: grb_models
      :link-type: ref
      :class-card: sd-shadow-sm

      Empirical and physical models for gamma-ray burst emission.

---------------------------------------------------------------------

.. _generic_models:

Generic Models
----------------

Generic models are designed to be flexible, modular, and easily subclassed. They are ideal for rapid prototyping,
exploratory data analysis, and building custom likelihood functions. These models are not tied
to specific physical scenarios but provide mathematical forms that can be adapted to a wide range of applications.

.. _curve_fitting_models:

Curve-Fitting Models
^^^^^^^^^^^^^^^^^^^^

These models provide flexible mathematical forms for fitting data.
They are especially useful during exploratory analysis and for
building likelihood functions.

.. rubric:: Generic Broken Power Laws

.. currentmodule:: triceratops.models.generic.curves

.. autosummary::
   :nosignatures:

   BrokenPowerLaw
   SmoothedBrokenPowerLaw
   TripleBrokenPowerLaw
   SmoothedTripleBrokenPowerLaw


.. _light_curve_models:

Light Curve Models
^^^^^^^^^^^^^^^^^^

These phenomenological models are designed for time-domain transient
analysis and pulse-shape modeling.

.. rubric:: Generic Light Curve Models

.. currentmodule:: triceratops.models.generic.light_curve

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

.. _evolving_models:

Evolving Phenomenological Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Models for phenomenological modeling (not tied to specific physical scenarios) that include time-evolution of parameters.

.. rubric:: Generic SED Evolution

.. currentmodule:: triceratops.models.generic.evolving_seds

.. autosummary::
   :nosignatures:

   ~evolving_sbpl.PL_Evolving_SBPL_Model
   ~evolving_sbpl.BPL_Evolving_SBPL_Model
   ~evolving_sbpl.TripleBPL_Evolving_SBPL_Model

---------------------------------------------------------------------

.. _sed_models:

SED Models
-----------

SED models are designed to provide full-physics spectral energy distribution modeling for various different
scenarios, emission mechanisms, and physical assumptions. These models are ideal for multi-wavelength fitting and inference.

.. rubric:: Synchrotron

.. currentmodule:: triceratops.models.SEDs.synchrotron

.. autosummary::
   :nosignatures:

    SSA_Cooling_SynchrotronSEDModel
    SSA_SynchrotronSEDModel
    Cooling_SynchrotronSEDModel
    SynchrotronSEDModel
    Synchrotron_SSA_SBPL_Model

---------------------------------------------------------------------

.. _supernova_models:

Supernova Models
----------------

These models incorporate physically motivated shock dynamics,
radiation processes, and circumstellar interaction physics.

.. rubric:: Self-Similar Shock Models

.. currentmodule:: triceratops.models.supernovae

.. autosummary::
   :nosignatures:

   chevalier_shock.ChevalierShockModel

---------------------------------------------------------------------

.. _grb_models:

GRB Models
----------

These models are designed to describe the spectral and temporal
properties of gamma-ray burst (GRB) emission. They range from
empirical spectral characterisation to physically motivated
jet and afterglow models.

.. rubric:: Prompt-Emission Spectral Models

.. currentmodule:: triceratops.models.GRBs

.. autosummary::
   :nosignatures:

   band.BandFunctionModel

---------------------------------------------------------------------

Extending the Model Library
----------------------------
