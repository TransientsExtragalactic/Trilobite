"""
Utilities for working with Likelihood classes.

These helpers provide deterministic reconstruction of likelihood
objects using fully-qualified import paths of the form:

"""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from triceratops.data.core import InferenceData
    from triceratops.inference.likelihood.base import Likelihood
    from triceratops.models.core.base import Model


# ---------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------
def get_likelihood_target(likelihood: "Likelihood") -> str:
    """
    Return fully-qualified import path for a likelihood instance.

    Parameters
    ----------
    likelihood : Likelihood
        Likelihood instance.

    Returns
    -------
    str
        Import path in the form "module:ClassName".
    """
    cls = likelihood.__class__
    return f"{cls.__module__}:{cls.__name__}"


def build_likelihood(
    target: str,
    model: "Model",
    data: "InferenceData",
) -> "Likelihood":
    """
    Construct a Likelihood from a fully-qualified import path.

    Parameters
    ----------
    target : str
        Import path in the form "module:ClassName".
    model : Model
        Forward model instance.
    data : InferenceData
        Validated inference dataset.

    Returns
    -------
    Likelihood
        Instantiated likelihood object.

    Raises
    ------
    ValueError
        If the target string is malformed.
    ImportError
        If the module or class cannot be imported.
    TypeError
        If the imported object is not a Likelihood subclass.
    """
    from triceratops.inference.likelihood.base import Likelihood

    if ":" not in target:
        raise ValueError(f"Invalid likelihood target '{target}'. Expected format 'module:ClassName'.")

    module_name, class_name = target.split(":", 1)

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(f"Could not import module '{module_name}'.") from exc

    try:
        cls = getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(f"Class '{class_name}' not found in module '{module_name}'.") from exc

    if not issubclass(cls, Likelihood):
        raise TypeError(f"Imported class '{class_name}' is not a subclass of Likelihood.")

    return cls(model=model, data=data)
