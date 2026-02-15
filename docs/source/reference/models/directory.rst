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
All models share a unified interface through the :class:`~models.core.base.Model`
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

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: 📈 Curve-Fitting Models
      :link: curve_fitting_models
      :link-type: ref
      :class-card: sd-shadow-sm

      Generic broken power laws and mathematical scaling
      relations for rapid exploratory modeling.

   .. grid-item-card:: 🌊 Light Curve Models
      :link: light_curve_models
      :link-type: ref
      :class-card: sd-shadow-sm

      Phenomenological transient pulse and time-domain models.

   .. grid-item-card:: 💥 Supernova Models
      :link: supernova_models
      :link-type: ref
      :class-card: sd-shadow-sm

      Physically motivated explosion and shock interaction models.

---------------------------------------------------------------------

.. _curve_fitting_models:

Curve-Fitting Models
--------------------

These models provide flexible mathematical forms for fitting data.
They are especially useful during exploratory analysis and for
building likelihood functions.

.. rubric:: Generic Broken Power Laws

.. currentmodule:: models.generic.bpl

.. autosummary::
   :nosignatures:

   BrokenPowerLaw
   SmoothedBrokenPowerLaw
   TripleBrokenPowerLaw
   SmoothedTripleBrokenPowerLaw


---------------------------------------------------------------------

.. _light_curve_models:

Light Curve Models
------------------

These phenomenological models are designed for time-domain transient
analysis and pulse-shape modeling.

.. rubric:: Generic Light Curve Models

.. currentmodule:: models.generic.light_curve

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


---------------------------------------------------------------------

.. _supernova_models:

Supernova Models
----------------

These models incorporate physically motivated shock dynamics,
radiation processes, and circumstellar interaction physics.

.. rubric:: Self-Similar Shock Models

.. currentmodule:: models.supernovae

.. autosummary::
   :nosignatures:

   chevalier_shock.ChevalierShockModel

---------------------------------------------------------------------

Extending the Model Library
----------------------------



---------------------------------------------------------------------
