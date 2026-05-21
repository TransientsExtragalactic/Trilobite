"""Serialization protocol objects for Trilobite models."""

import importlib
import json
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, ClassVar, Union

# ============================================================= #
# Package versioning and Type checking
# ============================================================= #=
try:
    __pkg_version__ = version("trilobite")
except PackageNotFoundError:
    __pkg_version__ = "unknown"


@dataclass(frozen=True)
class ModelSpec:
    """
    Serializable specification for constructing a Model.

    A :class:`ModelSpec` describes a model purely by:

    - Its fully-qualified import path
    - The keyword arguments required for initialization

    This object is:

    - JSON serializable
    - HDF5-storable
    - MPI-safe
    - Deterministic
    - Reproducible

    Notes
    -----
    The ``target`` must be of the form::

        "package.module:ClassName"

    Example
    -------

    .. code-block:: python

        spec = ModelSpec(
            target="trilobite.models.sed:PL_Evolving_SSA_SED_Model",
            kwargs={"p": 2.5, "epsilon_B": 0.1},
        )

        model = spec.build()
    """

    FORMAT: ClassVar[str] = "ModelSpec"
    VERSION: ClassVar[str] = __pkg_version__

    target: str
    """Fully-qualified import path in the form ``'module:ClassName'``."""

    kwargs: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments passed to the model constructor."""

    def build(self) -> Any:
        """
        Instantiate the model described by this specification.

        Returns
        -------
        Any
            Instantiated model object.

        Raises
        ------
        ValueError
            If the target string is malformed.
        ImportError
            If the module or class cannot be imported.
        """
        if ":" not in self.target:
            raise ValueError(f"Invalid target '{self.target}'. Expected format 'module:ClassName'.")

        module_name, class_name = self.target.split(":", 1)

        module = importlib.import_module(module_name)

        try:
            cls = getattr(module, class_name)
        except AttributeError as exc:
            raise ImportError(f"Class '{class_name}' not found in module '{module_name}'.") from exc

        return cls(**self.kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Convert this ModelSpec to a JSON-safe dictionary."""
        return {
            "format": self.FORMAT,
            "version": self.VERSION,
            "target": self.target,
            "kwargs": self.kwargs,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelSpec":
        """Construct a ModelSpec from a dictionary."""
        if not isinstance(data, dict):
            raise TypeError("Input must be a dictionary.")

        if data.get("format") != cls.FORMAT:
            raise ValueError(f"Dictionary does not represent {cls.FORMAT}.")

        return cls(
            target=data["target"],
            kwargs=data.get("kwargs", {}),
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialize this ModelSpec to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_string: str) -> "ModelSpec":
        """Deserialize a ModelSpec from a JSON string."""
        data = json.loads(json_string)
        return cls.from_dict(data)

    def to_string(self) -> str:
        """
        Return a compact single-line string representation.

        Suitable for storing in HDF5 attributes.
        """
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_string(cls, string: str) -> "ModelSpec":
        """Deserialize a ModelSpec from a compact string representation."""
        data = json.loads(string)
        return cls.from_dict(data)

    def to_file(self, filename: Union[str, Path], overwrite: bool = False):
        """
        Write this ModelSpec to a JSON file.

        Parameters
        ----------
        filename : str or Path
            Destination file path.
        overwrite : bool, optional
            If False and the file exists, raise FileExistsError.
        """
        path = Path(filename)

        if path.exists() and not overwrite:
            raise FileExistsError(f"File '{path}' already exists.")

        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def from_file(cls, filename: Union[str, Path]) -> "ModelSpec":
        """Load a ModelSpec from a JSON file."""
        path = Path(filename)

        if not path.exists():
            raise FileNotFoundError(f"File '{path}' not found.")

        with open(path) as f:
            return cls.from_json(f.read())
