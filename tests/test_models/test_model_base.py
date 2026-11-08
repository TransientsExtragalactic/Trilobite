"""
Base tests for the Trilobite Model abstraction.

This module verifies:

- Proper subclass validation
- Forward model behavior
- Default parameter handling
- Bounds enforcement
- Metadata properties
- Tupled output ordering
"""

from typing import NamedTuple

import pytest
from astropy import units as u

from trilobite.models.core import Model, ModelParameter, ModelStatus, ModelVariable


# ============================================================
# Reusable Output Structure
# ============================================================
class SimpleOutputs(NamedTuple):
    y: str


import matplotlib.pyplot as plt

# ============================================================
# Reusable Test Structure
# ============================================================
import numpy as np
import pytest


class BaseModelTest:
    """
    Reusable pytest base class for testing Model subclasses.

    Subclasses must define:

    - MODEL: the model class
    - VARIABLES: dict of input variables
    - PARAMETERS: dict of input parameters

    Optional plotting configuration:

    - LOG_X: bool
    - LOG_Y: bool
    """

    MODEL = None
    VARIABLES = None
    PARAMETERS = None

    LOG_X = False
    LOG_Y = False

    # ---------------------------------- #
    # Sanity Checks                      #
    # ---------------------------------- #

    def test_model_initializes(self):
        """Model can be instantiated."""
        model = self.MODEL()
        assert model is not None

    def test_model_evaluates(self):
        """Model produces outputs with correct structure."""
        model = self.MODEL()

        outputs = model(self.VARIABLES, self.PARAMETERS)

        assert isinstance(outputs, self.MODEL.OUTPUTS)

        for value in outputs:
            assert value is not None

    # ---------------------------------- #
    # Optional Diagnostic Plotting       #
    # ---------------------------------- #
    def test_model_diagnostic_plot(
        self,
        diagnostic_plots,
        diagnostic_plots_dir,
    ):
        """
        Generate diagnostic plot if enabled via CLI.
        """

        if not diagnostic_plots:
            pytest.skip("Diagnostic plots disabled.")

        model = self.MODEL()
        outputs = model(self.VARIABLES, self.PARAMETERS)

        x_name = model.variable_names[0]
        x = self.VARIABLES[x_name]

        # Strip units for plotting if necessary
        if hasattr(x, "value"):
            x = x.value

        for name in outputs._fields:
            y = getattr(outputs, name)

            if hasattr(y, "value"):
                y_plot = y.value
            else:
                y_plot = y

            plt.figure()

            # --- Flexible log scaling ---
            if self.LOG_X and self.LOG_Y:
                plt.loglog(x, y_plot)
            elif self.LOG_X:
                plt.semilogx(x, y_plot)
            elif self.LOG_Y:
                plt.semilogy(x, y_plot)
            else:
                plt.plot(x, y_plot)

            plt.xlabel(x_name)
            plt.ylabel(name)
            plt.title(f"{self.MODEL.__name__} – {name}")
            plt.tight_layout()

            fname = f"{self.MODEL.__name__}_{name}.png"
            plt.savefig(diagnostic_plots_dir / fname)
            plt.close()

    # ---------------------------------- #
    # Serialization Test                 #
    # ---------------------------------- #
    def test_model_spec_roundtrip(self):
        """
        Model → ModelSpec → Model should preserve behavior.

        This ensures:
        - Constructor arguments were properly registered
        - The model is importable via its module path
        - All constructor kwargs are JSON-serializable
        - Forward evaluation is identical after reconstruction
        """

        # Instantiate model
        model = self.MODEL()

        # Convert to ModelSpec
        spec = model.to_model_spec()

        # Ensure spec target matches model class
        expected_target = f"{self.MODEL.__module__}:{self.MODEL.__name__}"
        assert spec.target == expected_target

        # Reconstruct model
        reconstructed = self.MODEL.from_model_spec(spec)

        # Ensure type correctness
        assert isinstance(reconstructed, self.MODEL)


# ============================================================
# Minimal Valid Concrete Model
# ============================================================
class SimpleModel(Model):
    """
    Minimal valid concrete model for base-class testing.
    """

    PARAMETERS = (
        ModelParameter(
            name="a",
            default=1.0,
            base_units=u.dimensionless_unscaled,
        ),
    )

    VARIABLES = (
        ModelVariable(
            name="x",
            base_units=u.dimensionless_unscaled,
        ),
    )

    OUTPUTS = SimpleOutputs
    UNITS = SimpleOutputs(y=u.dimensionless_unscaled)

    def __init__(self):
        self._register_init()

    def _forward_model(self, variables, parameters):
        return self.OUTPUTS(y=parameters["a"] * variables["x"])


# ============================================================
# Subclass Validation Tests
# ============================================================


def test_model_requires_variables():
    """Model subclasses must declare at least one VARIABLE."""

    with pytest.raises(ValueError):

        class _BadModel(Model):
            PARAMETERS = ()
            VARIABLES = ()

            class Outputs(NamedTuple):
                y: str

            OUTPUTS = Outputs
            UNITS = Outputs(y=u.dimensionless_unscaled)

            def _forward_model(self, variables, parameters):
                return self.OUTPUTS(y=0.0)


def test_outputs_units_field_mismatch():
    """UNITS must structurally match OUTPUTS."""

    with pytest.raises(TypeError):

        class _BadModel(Model):
            PARAMETERS = (ModelParameter("a", 1.0, base_units=u.dimensionless_unscaled),)

            VARIABLES = (ModelVariable("x", base_units=u.dimensionless_unscaled),)

            class Outputs(NamedTuple):
                y: str

            OUTPUTS = Outputs

            # Invalid: wrong field structure
            class BadUnits(NamedTuple):
                y: u.Unit
                z: u.Unit

            UNITS = BadUnits(y=u.dimensionless_unscaled, z=u.dimensionless_unscaled)

            def _forward_model(self, variables, parameters):
                return self.OUTPUTS(y=parameters["a"] * variables["x"])


# ============================================================
# Forward Model Behavior
# ============================================================


def test_forward_model_default_parameter():
    """Missing parameters should be filled with defaults."""

    model = SimpleModel()
    result = model({"x": 2.0}, {})

    assert result.y.value == 2.0
    assert result.y.unit == u.dimensionless_unscaled


def test_forward_model_parameter_override():
    """Provided parameters should override defaults."""

    model = SimpleModel()
    result = model({"x": 2.0}, {"a": 3.0})

    assert result.y.value == 6.0


def test_forward_model_missing_variable_raises():
    """Missing required variable should raise ValueError."""

    model = SimpleModel()

    with pytest.raises(ValueError):
        model({}, {})


# ============================================================
# Bounds Checking
# ============================================================


def test_parameter_bounds_violation():
    """Parameter bounds should be enforced."""

    class BoundedModel(SimpleModel):
        PARAMETERS = (
            ModelParameter(
                name="a",
                default=1.0,
                base_units=u.dimensionless_unscaled,
                bounds=(0.0, 10.0),
            ),
        )

    model = BoundedModel()

    with pytest.raises(ValueError):
        model({"x": 1.0}, {"a": -1.0})


# ============================================================
# Metadata Properties
# ============================================================


def test_metadata_properties():
    """parameter_names, variable_names, and output_names should match definitions."""

    model = SimpleModel()

    assert model.parameter_names == ("a",)
    assert model.variable_names == ("x",)
    assert model.output_names == ("y",)


# ============================================================
# Tupled Output Order
# ============================================================
def test_forward_model_tupled():
    """_forward_model_tupled should respect OUTPUT ordering."""

    model = SimpleModel()

    result = model._forward_model_tupled(
        {"x": 2.0},
        {"a": 3.0},
    )

    assert isinstance(result, tuple)
    assert result[0] == 6.0


# ============================================================
# Development-status tests
# ============================================================
def _walk_models(cls):
    for sub in cls.__subclasses__():
        yield sub
        yield from _walk_models(sub)


def _make_model(**attrs):
    """Build a throwaway concrete Model subclass with the given class attributes."""
    return type("_StatusModel", (SimpleModel,), attrs)


def test_status_default_is_unvalidated():
    """A model that declares no status is UNVALIDATED."""
    assert Model.STATUS is ModelStatus.UNVALIDATED
    assert _make_model().STATUS is ModelStatus.UNVALIDATED


def test_status_is_not_inherited():
    """Subclassing a validated model resets the status (and note) to the safe default."""
    parent = _make_model(STATUS=ModelStatus.VALIDATED, STATUS_NOTE="checked against X")
    child = type("_Child", (parent,), {})
    assert parent.STATUS is ModelStatus.VALIDATED
    assert child.STATUS is ModelStatus.UNVALIDATED
    assert child.STATUS_NOTE == ""


def test_status_string_coercion_and_validation():
    """Strings are accepted case-insensitively; unknown ones are rejected at class definition."""
    assert _make_model(STATUS="Validated").STATUS is ModelStatus.VALIDATED
    with pytest.raises(ValueError):
        _make_model(STATUS="finished")


@pytest.mark.parametrize("status", [ModelStatus.EXPERIMENTAL, ModelStatus.DEPRECATED])
def test_status_warns_on_instantiation(status):
    """Experimental and deprecated models warn when instantiated, quoting the note."""
    cls = _make_model(STATUS=status, STATUS_NOTE="use Other instead")
    with pytest.warns(UserWarning, match="use Other instead"):
        cls()


@pytest.mark.parametrize("status", [ModelStatus.VALIDATED, ModelStatus.UNVALIDATED])
def test_status_silent_otherwise(status):
    """Validated and unvalidated models do not warn."""
    import warnings

    cls = _make_model(STATUS=status)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cls()


def test_all_library_models_declare_status():
    """Every concrete in-tree model must declare STATUS explicitly rather than rely on the default."""
    import inspect

    import trilobite.models  # noqa: F401  (registers every model subclass)

    undeclared = sorted(
        f"{c.__module__}.{c.__name__}"
        for c in set(_walk_models(Model))
        if c.__module__.startswith("trilobite.") and not inspect.isabstract(c) and not c._status_declared
    )
    assert not undeclared, f"Models missing an explicit STATUS: {undeclared}"
