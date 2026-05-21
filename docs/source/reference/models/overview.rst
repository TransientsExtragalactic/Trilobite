.. _models_overview:

====================================
Trilobite Models
====================================

At the heart of every Trilobite workflow is a **model**. At their core, models are a relatively simple concept:
They are a mapping :math:`\mathcal{M}` from a set of parameters :math:`\boldsymbol{\Theta}` and a set of variables
:math:`{\bf x}` to a set of observables :math:`{\bf y}`:

.. math::

    \mathcal{M}(\boldsymbol{\Theta}, {\bf x}) = {\bf y}.

That's really all there is to it! The exact nature of the mapping is entirely dependent on the model itself, as are
the variables, the parameters, and the observables.

The intention of Trilobite is to provide an **extensive backend library of physics and mathematics modules** so that,
when it comes time to write your model, you can generate the model with just a few lines of code to patch together
the different pieces of physics that you need. Once your model is completed, you can use it to generate forward modeled
synthetic data or to perform inference on real data. This is the principle of **rapid-deployment modeling**.

In this guide, we'll discuss the steps and the nuances to consider when building your own Trilobite models.

.. contents::
    :local:
    :depth: 2

----

The Model Class
----------------

(*source module*: :class:`trilobite.models.core.base.Model`)

In order to explain the model building process, we first need to understand the :class:`trilobite.models.core.base.Model` class,
which is **the building block for all Trilobite models**. This class provides a basic template for building a model, and
its suggested that you actually look at the source code of the model class while building your own so you can understand
what the different wrapper methods are doing.

.. note::

    For **any model** you develop in the Trilobite framework, you should do the following basic steps:

    1. **Develop the forward model**: This is the core of your model, where you determine the physics you
       want to represent. (See :ref:`forward_modeling` for more details on this step.)
    2. **Define the model's parameters and variables**: This is where you specify the parameters and variables that
       your model will use, along with their base units and other metadata. (See :ref:`model_parameters` for more details on this step.)

    This is really all there is to it! In a few cases, specialized modeling patterns have been developed for
    specific types of models (e.g. optical photometry models), but in general, the above two steps are all you need to do to build a new model.

To get started, you can create a new class that inherits from :class:`trilobite.models.core.base.Model` and
implement the necessary methods and attributes.

.. dropdown:: Example

    .. code-block:: python

        from trilobite.models.core.base import Model

        class MyModel(Model):
            # Step 1: Define the model's parameters and variables as class attributes.
            PARAMETERS = (
                # ... define your parameters here ...
            )
            VARIABLES = (
                # ... define your variables here ...
            )

            # Step 2: Implement the forward model method to compute the outputs based on the inputs.
            def _forward_model(self, variables, parameters):
                # ... implement your model's physics here ...
                return outputs


.. _forward_modeling:
The Forward Modelling Directive
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The most important method in the :class:`trilobite.models.core.base.Model` class is the
:meth:`trilobite.models.core.base.Model._forward_model` method:

.. code-block:: python

    @abstractmethod
    def _forward_model(
        self,
        variables: _ModelVariablesInputRaw,
        parameters: _ModelParametersInputRaw,
    ) -> _ModelOutputRaw:
        """
        Compute the model's outputs based on the provided variables and parameters.

        Parameters
        ----------
        variables: dict of str, array-like
            The model's input variables. Each variable must be provided as a key-value pair in the
            dictionary, where the key is the variable name and the value is the variable's value. Each element
            must either be a float or an array-like object. Standard numpy broadcasting rules are applied
            throughout.
        parameters: dict of str, array-like
            The model's parameters. Each parameter must be provided as a key-value pair in the
            dictionary, where the key is the parameter name and the value is the parameter's value. Each element
            must either be a float or an array-like object. Standard numpy broadcasting rules are applied
            throughout. All parameters must be provided; there are no default values at this level.

        Returns
        -------
        outputs: dict of str, array-like
            The model's outputs. Each output is provided as a key-value pair in the dictionary, where the key
            is the output name and the value is the computed output value. Each element will either be a float
            or an array-like object, depending on the inputs. Standard numpy broadcasting rules are applied throughout.
        """
        pass

This is the real workhorse of the model class: it takes in the model's variables and parameters (as :class:`dict`) and
performs whatever work the model need to generate its outputs. It then bundles those outputs into another :class:`dict`
and returns them.

When building your own model, you will need to implement this method to define how your model computes its outputs
based on the provided variables and parameters.

There are a few pieces of advice to be aware of when implementing this method:

1. **Use Numpy Broadcasting**: The inputs to this method can be either scalar values or array-like objects.
   Make sure to leverage numpy's broadcasting capabilities to handle both cases seamlessly. This will allow your
   model to work with a variety of input shapes and sizes.

   .. hint::

        The convention throughout Trilobite is that **numpy broadcasting rules are implicitly assumed** everywhere
        unless otherwise specified. As an example, consider a function

        .. code-block:: python

            def example_function(a: np.ndarray, b: np.ndarray) -> np.ndarray:
                return a + b

        The developer should **NOT** attempt to coerce the shapes of ``a`` and ``b`` to be the same shape.
        Instead, they should
        simply rely on numpy's broadcasting rules to handle the shapes appropriately.
        This means that if ``a`` is of shape
        ``(N, M)`` and ``b`` is of shape ``(M,)``, numpy will automatically
        broadcast ``b`` to shape ``(N, M)`` during the addition operation.

2. **Performance is Critical**: Since this method will be called frequently during model evaluations,
   it's essential to optimize its performance. Avoid unnecessary computations and leverage efficient
   numerical libraries whenever possible.

   To ensure that this method is performant, consider the following tips:

   - **Vectorization**: Whenever possible, use vectorized operations instead of loops. Numpy and similar libraries
     are optimized for vectorized computations, which can significantly speed up your model evaluations.
   - **No Units**: Units should **never** be coerced within the computational layer. The public-facing method
     :meth:`trilobite.models.core.base.Model.forward_model` handles all unit conversions before and after calling
     :meth:`trilobite.models.core.base.Model._forward_model`. This ensures that the computational layer remains as efficient
     as possible. The convention for base units is discussed in :ref:`model_parameters`.
   - **Low-Level Building Blocks**: Whenever possible, leverage low-level building blocks from the Trilobite
     library. These building blocks are designed to be efficient and can help you avoid reinventing the wheel.
   - **Cythonize if Necessary**: If you find that certain parts of your model are still too slow, consider implementing
     those parts in Cython or using other performance optimization techniques.
3. **Clear Documentation**: Document the expected inputs and outputs of your model clearly. This will help
   other developers (and your future self) understand how to use your model effectively.

.. _model_parameters:

Model Parameters and Variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At the top of every :class:`trilobite.models.core.base.Model` class are two important class attributes: the
:attr:`trilobite.models.core.base.Model.PARAMETERS` and :attr:`trilobite.models.core.base.Model.VARIABLES`.

.. code-block:: python

    # =============================================== #
    # Parameter and Variable Declarations             #
    # =============================================== #
    # Each model must declare its parameters and variables as class-level attributes. These
    # must each be instances of `ModelParameter` and `ModelVariable`, respectively.
    PARAMETERS: tuple[ModelParameter, ...] = ()
    """tuple of :class:`ModelParameter`: The model's parameters.

    Each element of :attr:`PARAMETERS` is a :class:`ModelParameter` instance that defines a single
    parameter of the model. These parameters are used to configure the model and control its behavior. Each
    parameter contains information about the base units, default value, and valid range for that parameter.
    """
    VARIABLES: tuple[ModelVariable, ...] = ()
    """tuple of :class:`ModelVariable`: The model's variables.

    Each element of :attr:`VARIABLES` is a :class:`ModelVariable` instance that defines a single
    variable of the model. These variables represent the inputs to the model that can vary during
    evaluation. Each variable contains information about the base units, name, etc. of the variable. Notably,
    variables differ from the parameters (:attr:`PARAMETERS`) in that variables are do **NOT** have default
    values, do **NOT** have validity ranges, and are expected to be provided at model evaluation time.
    """

These two attributes are used to define the parameters and variables that your model will use. Each parameter
and variable is represented by an instance of the :class:`trilobite.models.core.parameters.ModelParameter` and
:class:`trilobite.models.core.parameters.ModelVariable` classes, respectively.

When building your own model, you'll need to populate these attributes with the appropriate parameters and
variables for your model. This will help ensure that your model is well-defined and can be used effectively
within the Trilobite framework.

For example,

.. dropdown:: Example Parameter and Variable Declarations

    .. code-block:: python

        # =============================================== #
        # Parameter and Variable Declarations             #
        # =============================================== #
        # Each model must declare its parameters and variables as class-level attributes. These
        # must each be instances of `ModelParameter` and `ModelVariable`, respectively.
        PARAMETERS: tuple["ModelParameter", ...] = (
            ModelParameter(
                "norm",
                1.0,
                description="Normalization constant of the SED.",
                base_units="Jy",
                bounds=(0.0, None),
                latex=r"F_{\rm nu,0}",
            ),
            ModelParameter(
                "nu_break",
                5.0,
                description="Break frequency of the SED.",
                base_units="GHz",
                bounds=(0.0, None),
                latex=r"\nu_{\rm break}",
            ),
            ModelParameter(
                "p",
                3.0,
                description="Power-law index for the electron energy distribution.",
                base_units="",
                bounds=(0, None),
                latex=r"p",
            ),
            ModelParameter(
                "s",
                -0.5,
                description="The smoothing parameter of the broken power-law.",
                base_units="",
                bounds=(None, 0),
                latex=r"s",
            ),
        )
        """tuple of :class:`ModelParameter`: The model's parameters.

        Each element of :attr:`PARAMETERS` is a :class:`ModelParameter` instance that defines a single
        parameter of the model. These parameters are used to configure the model and control its behavior. Each
        parameter contains information about the base units, default value, and valid range for that parameter.
        """
        VARIABLES: tuple["ModelVariable", ...] = (
            ModelVariable(
                "frequency", description="Observing frequency at which to evaluate the SED.", base_units="GHz", latex=r"\nu"
            ),
        )
        """tuple of :class:`ModelVariable`: The model's variables.

        Each element of :attr:`VARIABLES` is a :class:`ModelVariable` instance that defines a single
        variable of the model. These variables represent the inputs to the model that can vary during
        evaluation. Each variable contains information about the base units, name, etc. of the variable. Notably,
        variables differ from the parameters (:attr:`PARAMETERS`) in that variables are do **NOT** have default
        values, do **NOT** have validity ranges, and are expected to be provided at model evaluation time.
        """

The ModelParameter and ModelVariable Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each element of the :attr:`~trilobite.models.core.base.Model.PARAMETERS` and :attr:`~trilobite.models.core.base.Model.VARIABLES` must be
an instance of the :class:`~trilobite.models.core.parameters.ModelParameter` and :class:`~trilobite.models.core.parameters.ModelVariable` classes, respectively.

By means of a brief introduction to each of these classes, we can understand how to use them effectively when defining our model's parameters and variables.
The core idea with these classes is that they provide a simple structure for tracking a parameters / variables name,
its base units, and other metadata that is useful for documentation and validation.

From the :class:`~trilobite.models.core.parameters.ModelParameter` documentation:

.. code-block:: python

    def __init__(
        self,
        name: str,
        default: Union[float, u.Quantity],
        *,
        base_units: Union[str, u.Unit] = u.dimensionless_unscaled,
        description: str = "",
        latex: str = "",
        bounds: BoundType = (None, None),
    ):
        r"""
        Initialize a :class:`ModelParameter` instance.

        Parameters
        ----------
        name : str
            Name of the parameter. This must match the keyword expected by the model's
            forward function.
        default : float or astropy.units.Quantity
            Default value of the parameter. If a Quantity is provided, it will be coerced
            into ``base_units``.
        base_units : str or astropy.units.Unit, optional
            Base units assumed by the model's internal (low-level) physics implementation.
            Defaults to dimensionless.
        description : str, optional
            Human-readable description of the parameter.
        latex : str, optional
            LaTeX representation of the parameter symbol (e.g. ``r"$\epsilon_B$"``).
        bounds : tuple or callable, optional
            Either:

            - A ``(min, max)`` tuple of floats in base units, or
            - A callable ``f(value) -> bool`` that returns whether the value is valid.

        """

Thus, a :class:`~trilobite.models.core.parameters.ModelParameter` instance requires a name, a default value, and optionally accepts base units, a description,
a LaTeX representation, and bounds for valid values. Likewise, the :class:`~trilobite.models.core.parameters.ModelVariable` class has a similar structure:

.. code-block:: python

    def __init__(
        self,
        name: str,
        *,
        base_units: Union[str, u.Unit],
        description: str = "",
        latex: str = "",
    ):
        self.name: str = name
        """str: Name of the variable."""

        self.description: str = description
        """str: Human-readable description of the variable."""

        self.latex: str = latex
        """str: LaTeX representation of the variable symbol."""

        self.base_units: u.Unit = self._parse_base_units(base_units)
        """astropy.units.Unit: Base units for this variable."""

Parameter Bounds
~~~~~~~~~~~~~~~~

One confusing aspect of the :class:`~trilobite.models.core.parameters.ModelParameter` class is the
``bounds`` argument. This argument can either be a tuple of ``(min, max)`` values or a callable function that
takes in a value and returns whether that value is valid. For example, a parameter which must be positive
could be specified as

.. code-block:: python

    param = ModelParameter(
        "positive_param",
        1.0,
        base_units="Jy",
        bounds=(0.0, None),
    )

Likewise, a parameter which must be any value except for 2.0 would require a callable function:

.. code-block:: python

    def is_valid(value: float) -> bool:
        return value != 2.0

    param = ModelParameter(
        "not_two_param",
        1.0,
        base_units="Jy",
        bounds=is_valid,
    )

.. important::

    Bounds checking is not performed automatically by ``_forward_model``: this is for performance reasons. Instead,
    bounds checking is performed at the public-facing method :meth:`trilobite.models.core.base.Model.forward_model`, which
    handles all unit conversions and validation before and after calling :meth:`trilobite.models.core.base.Model._forward_model`.
    This ensures that the computational layer remains as efficient as possible.

    In inference workflows, it is instead the likelihood object which evaluates the bounds of the parameters during
    sampling.

    DO NOT implement bounds checking in your ``_forward_model`` method!

Units
~~~~~

Every parameter and variable in Trilobite has associated ``base_units``. These base units are used to ensure that
the model's computations are performed in a consistent unit system. When building your model, it's important to
choose appropriate base units for each parameter and variable. The choice of base units can impact the performance and
accuracy of your model, so it's worth taking the time to consider this carefully.

When implementing the :meth:`trilobite.models.core.base.Model._forward_model` method, you can assume that all inputs are provided
in their base units. Likewise, the outputs you return from this method should also be in the appropriate base units.

The public facing API method :meth:`trilobite.models.core.base.Model.forward_model` will handle all unit conversions before and after
calling :meth:`trilobite.models.core.base.Model._forward_model`. This ensures that the computational layer remains as efficient as possible.

Unitless variables should have ``""`` as their base units.

----

A Simple Example
-----------------

Let's look at the simple mode :class:`~trilobite.models.SEDs.synchrotron.Synchrotron_SSA_SBPL_Model`,
which models a synchrotron self-absorbed broken power-law spectral energy distribution (SED).

.. hint::

    You can read more about this model and other synchrotron SED models in :ref:`synchrotron_overview`.

Looking at the source code, the variables and parameters are

.. code-block:: python

    # =============================================== #
    # Parameter and Variable Declarations             #
    # =============================================== #
    # Each model must declare its parameters and variables as class-level attributes. These
    # must each be instances of `ModelParameter` and `ModelVariable`, respectively.
    PARAMETERS: tuple["ModelParameter", ...] = (
        ModelParameter(
            "norm",
            1.0,
            description="Normalization constant of the SED.",
            base_units="Jy",
            bounds=(0.0, None),
            latex=r"F_{\rm nu,0}",
        ),
        ModelParameter(
            "nu_break",
            5.0,
            description="Break frequency of the SED.",
            base_units="GHz",
            bounds=(0.0, None),
            latex=r"\nu_{\rm break}",
        ),
        ModelParameter(
            "p",
            3.0,
            description="Power-law index for the electron energy distribution.",
            base_units="",
            bounds=(0, None),
            latex=r"p",
        ),
        ModelParameter(
            "s",
            -0.5,
            description="The smoothing parameter of the broken power-law.",
            base_units="",
            bounds=(None, 0),
            latex=r"s",
        ),
    )
    """tuple of :class:`ModelParameter`: The model's parameters.

    Each element of :attr:`PARAMETERS` is a :class:`ModelParameter` instance that defines a single
    parameter of the model. These parameters are used to configure the model and control its behavior. Each
    parameter contains information about the base units, default value, and valid range for that parameter.
    """
    VARIABLES: tuple["ModelVariable", ...] = (
        ModelVariable(
            "frequency", description="Observing frequency at which to evaluate the SED.", base_units="GHz", latex=r"\nu"
        ),
    )
    """tuple of :class:`ModelVariable`: The model's variables.

    Each element of :attr:`VARIABLES` is a :class:`ModelVariable` instance that defines a single
    variable of the model. These variables represent the inputs to the model that can vary during
    evaluation. Each variable contains information about the base units, name, etc. of the variable. Notably,
    variables differ from the parameters (:attr:`PARAMETERS`) in that variables are do **NOT** have default
    values, do **NOT** have validity ranges, and are expected to be provided at model evaluation time.
    """

As such, we can immediately see that this model has four parameters: ``norm``, ``nu_break``, ``p``, and ``s``,
and one variable: ``frequency``. The base units for each parameter and variable are also clearly defined. The forward
model then implements the mapping:

.. math::

    \mathcal{M}: \nu \rightarrow F_\nu,\;{\rm s.t.}\; F_\nu = F_{\nu,0}
    \left[\left(\frac{\nu}{\nu_{\rm break}}\right)^{\alpha_1/s} +
    \left(\frac{\nu}{\nu_{\rm break}}\right)^{\alpha_2/s}\right]^{s},

We can see this in the ``_forward_model`` method:

.. code-block:: python

    # =============================================== #
    # Model Evaluation Method                         #
    # =============================================== #
    def _forward_model(
        self,
        variables: _ModelVariablesInputRaw,
        parameters: _ModelParametersInputRaw,
    ) -> _ModelOutputRaw:
        # Construct the indices from the value of p.
        p = parameters["p"]
        alpha_2 = 5 / 3
        alpha_1 = -(p - 1) / 2

        result = smoothed_BPL(
            variables["frequency"], parameters["norm"], parameters["nu_break"], alpha_1, alpha_2, parameters["s"]
        )
        return {"flux_density": result}

That's all there really is to it! Before implementing your own models, its worth spending some time looking through
the existing models in the Trilobite library to see how they are structured and implemented. Likewise, familiarizing
yourself with the existing physics toolchains will help you leverage the existing building blocks effectively.

.. _optical_model_pattern:

----

Specialized Model Patterns
---------------------------

While the :class:`~trilobite.models.core.base.Model` class provides a fully
general interface for defining forward models, many common modeling tasks
follow recurring structural patterns. To support these use cases,
Trilobite provides a set of **specialized model base classes** that extend
the core model interface with additional functionality.

These patterns are designed to:

- Encapsulate common workflows (e.g., filter convolution for photometric data),
- Reduce boilerplate in model implementations,
- Provide clearer and more constrained interfaces for specific problem domains.

In practice, users are encouraged to subclass one of these specialized models
whenever their modeling task aligns with the corresponding pattern, rather than
re-implementing the same logic within a generic
:class:`~trilobite.models.core.base.Model`.

The following sections describe the available specialized model patterns and
their intended use cases.

Optical Photometry Models
^^^^^^^^^^^^^^^^^^^^^^^^^^

A very common scenario in transient modeling is the need to predict **broadband
photometric observations**—that is, fluxes measured through a set of filters at
discrete epochs.

.. hint::

    From the user’s perspective, an optical photometry model behaves as follows:

    - The model is constructed with a collection of filters (a
      :class:`~trilobite.utils.phot_utils.FilterBundle`).
    - At evaluation time, the user provides:

      - ``band_index`` — specifying which filters to evaluate
      - ``time`` (for time-dependent models)

    - The model returns fluxes corresponding to those filters and epochs.

For example:

.. code-block:: python

    model = MyOpticalModel(bundle)

    flux = model.forward_model(
        {"band_index": ["V", "R", "V"], "time": [1e5, 1e5, 2e5]},
        params,
    )

In this interface, the user never explicitly performs filter convolution—the
model takes care of this internally.

Conceptually, this process is implemented by constructing an underlying
spectral energy distribution (SED), :math:`F_\nu(\nu, t)`, and convolving it
with the filter transmission curves:

.. math::

    F_{\rm band}(t) = \int F_\nu(\nu, t)\, T_{\rm band}(\nu)\, d\nu

Trilobite provides two abstract base classes that implement this workflow:

- :class:`~trilobite.models.core.optical.OpticalModel` — for **time-evolving**
  multi-band observations.
  Variables: ``band_index`` and ``time``.

- :class:`~trilobite.models.core.optical.OpticalEpochModel` — for **single-epoch**
  multi-band SEDs.
  Variable: ``band_index`` only.

These classes manage:

- Filter convolution
- Band selection via ``band_index``
- Output assembly and broadcasting

As a result, model implementations can focus entirely on computing the
underlying SED.

The ``_compute_sed`` Contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To define a custom optical model, subclasses implement ``_compute_sed`` rather
than :meth:`~trilobite.models.core.base.Model._forward_model`.

This method evaluates the SED on a shared frequency grid, which is then passed
to the base class for convolution and output construction.

.. code-block:: python

    # For OpticalModel (time-evolving):
    def _compute_sed(
        self,
        nu_grid: np.ndarray,    # shape (N_nu,), Hz
        t_unique: np.ndarray,   # shape (N_t,), seconds — deduplicated
        parameters: dict,
    ) -> np.ndarray:            # shape (N_t, N_nu), erg/s/cm²/Hz

    # For OpticalEpochModel (single epoch):
    def _compute_sed(
        self,
        nu_grid: np.ndarray,    # shape (N_nu,), Hz
        parameters: dict,
    ) -> np.ndarray:            # shape (N_nu,), erg/s/cm²/Hz

The ``nu_grid`` argument is always ``self.bundle.frequency_grid``, constructed
from the union of all filter transmission curves. The returned SED must be in
linear-space :math:`F_\nu` with units of
:math:`\mathrm{erg\,s^{-1}\,cm^{-2}\,Hz^{-1}}`.

Once computed, the base class applies
:meth:`~trilobite.utils.phot_utils.FilterBundle.apply` to perform the filter
convolution and assembles the requested outputs.

.. important::

    For :class:`~trilobite.models.core.optical.OpticalModel`, the ``t_unique``
    array contains only the **unique** times present in the input—not the full
    input array.

    This ensures that expensive computations are performed only once per epoch,
    even if multiple filters are evaluated at that time.

The ``band_index`` Variable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both classes define ``band_index`` as a variable identifying which filters to
evaluate. It may be provided either as integer indices into the
:class:`~trilobite.utils.phot_utils.FilterBundle`, or as filter names:

.. code-block:: python

    model.forward_model(
        {"band_index": [0, 1, 0], "time": [1e5, 1e5, 2e5]},
        params,
    )

    model.forward_model(
        {"band_index": ["V", "R", "V"], "time": [1e5, 1e5, 2e5]},
        params,
    )

String inputs are automatically resolved during
:meth:`~trilobite.models.core.base.Model.coerce_model_variables`.

FilterBundle Ownership and Mutation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`~trilobite.utils.phot_utils.FilterBundle` is provided at construction
time and stored as ``self.bundle``. The associated convolution weights are
precomputed.

Filters may be modified via:

.. code-block:: python

    model.add_filter("B", b_filter)
    model.remove_filter("V")

.. warning::

    Modifying the filter bundle rebuilds the convolution weights and must be
    done **before** any inference run. Changing filters mid-inference will
    invalidate previously computed results.


.. dropdown:: Full Example

    The following skeleton illustrates the minimal implementation of an
    :class:`~trilobite.models.core.optical.OpticalModel`:

    .. code-block:: python

        import numpy as np
        from trilobite.models.core.optical import OpticalModel
        from trilobite.models.core.parameters import ModelParameter

        class MyOpticalModel(OpticalModel):
            PARAMETERS = (
                ModelParameter("T", 1e4, base_units="K", bounds=(0, None)),
            )

            DESCRIPTION = "My custom optical photometry model."

            def __init__(self, bundle):
                super().__init__(bundle)

            def _compute_sed(self, nu_grid, t_unique, parameters):
                # nu_grid: (N_nu,) Hz
                # t_unique: (N_t,) s
                T = parameters["T"]

                # ... compute SED ...
                return f_nu  # shape (N_t, N_nu)
