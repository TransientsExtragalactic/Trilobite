"""
Utilities for working with Likelihood classes.

These helpers provide deterministic reconstruction of likelihood
objects using fully-qualified import paths of the form:
"""

import importlib
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from trilobite.data.core import InferenceData
    from trilobite.inference.likelihood.base import Likelihood
    from trilobite.models.core.base import Model


# ---------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------
def get_likelihood_target(likelihood: "Likelihood") -> str:
    """
    Return fully-qualified import path for a likelihood instance.

    Parameters
    ----------
    likelihood : ~trilobite.inference.likelihood.base.Likelihood
        Likelihood instance.

    Returns
    -------
    str
        Import path in the form "module:ClassName".
    """
    cls = likelihood.__class__
    return f"{cls.__module__}:{cls.__name__}"


def _to_json_safe(value, name: str):
    """Convert numpy scalars / arrays (recursively) to plain Python objects; reject anything else non-JSON."""
    import numpy as np

    # numpy scalars must be checked first: ``np.float64`` is a ``float`` subclass and would otherwise pass through.
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v, name) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v, name) for k, v in value.items()}
    raise TypeError(
        f"Likelihood option '{name}' of type {type(value).__name__} is not JSON-serializable, so the "
        "InferenceProblem cannot be serialized."
    )


def get_likelihood_config(likelihood: "Likelihood") -> dict:
    """
    Return the JSON-serializable construction options of a likelihood instance.

    Parameters
    ----------
    likelihood : ~trilobite.inference.likelihood.base.Likelihood
        Likelihood instance.

    Returns
    -------
    dict
        The keyword arguments the likelihood was constructed with (excluding ``model`` and ``data``),
        e.g. ``{"sys_dex": 0.1}``.

    Raises
    ------
    TypeError
        If an option cannot be represented in JSON.
    """
    return {name: _to_json_safe(value, name) for name, value in likelihood.config.items()}


def build_likelihood(
    target: str,
    model: "Model",
    data: "InferenceData",
    config: Optional[dict] = None,
) -> "Likelihood":
    """
    Construct a Likelihood from a fully-qualified import path.

    Parameters
    ----------
    target : str
        Import path in the form "module:ClassName".
    model : ~trilobite.models.core.base.Model
        Forward model instance.
    data : InferenceData
        Validated inference dataset.
    config : dict, optional
        Keyword arguments forwarded to the likelihood constructor (see :func:`get_likelihood_config`).

    Returns
    -------
    ~trilobite.inference.likelihood.base.Likelihood
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
    from trilobite.inference.likelihood.base import Likelihood

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

    return cls(model=model, data=data, **(config or {}))
