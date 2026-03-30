"""IO utilities for Triceratops."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from astropy import units as u
from astropy.cosmology import FLRW, FlatLambdaCDM
from ruamel.yaml.comments import CommentedMap

if TYPE_CHECKING:
    from ruamel.yaml import YAML
    from ruamel.yaml.constructor import Constructor
    from ruamel.yaml.nodes import Node
    from ruamel.yaml.representer import Representer


# ----------------------------------------------- #
# YAML Reader/Writer tools                        #
# ----------------------------------------------- #
class _YAMLHandler(ABC):
    """
    Abstract base class for defining custom YAML representers and constructors.

    Subclasses must define:
        - `__tag__`: a YAML tag string (e.g., "!unyt_quantity")
        - `__type__`: the Python type to associate with this handler
        - `to_yaml()`: a static method to serialize the Python object to a YAML node
        - `from_yaml()`: a static method to deserialize a YAML node to a Python object

    This class provides a consistent interface for registering type-specific
    (de)serialization logic with a `ruamel.yaml.YAML` instance.

    Example
    -------
    class MyTypeHandler(_YAMLHandler):
        __tag__ = "!my_type"
        __type__ = MyType

        @staticmethod
        def to_yaml(representer, obj):
            return representer.represent_mapping(MyTypeHandler.__tag__, {
                "x": obj.x,
                "y": obj.y
            })

        @staticmethod
        def from_yaml(loader, node):
            data = CommentedMap()
            loader.construct_mapping(node, maptyp=data,  deep=True)
            return MyType(data["x"], data["y"])

    yaml = YAML()
    MyTypeHandler.register(yaml)
    """

    __tag__: str = None
    __type__: type = None

    @staticmethod
    @abstractmethod
    def to_yaml(representer: "Representer", obj: Any) -> "Node":
        """
        Convert a Python object to a YAML node.

        Parameters
        ----------
        representer : ``ruamel.yaml.representer.Representer``
            The YAML representer to use for creating the node.
        obj : Any
            The Python object to serialize.

        Returns
        -------
        ``ruamel.yaml.nodes.Node``
            A YAML node representing the serialized object.
        """
        pass

    @staticmethod
    @abstractmethod
    def from_yaml(loader: "Constructor", node: "Node") -> Any:
        """
        Convert a YAML node to a Python object.

        Parameters
        ----------
        loader : ``ruamel.yaml.constructor.Constructor``
            The YAML loader to use for constructing the Python object.
        node : ``ruamel.yaml.nodes.Node``
            The YAML node to deserialize.

        Returns
        -------
        Any
            The deserialized Python object.
        """
        pass

    @classmethod
    def register(cls, yaml: "YAML") -> None:
        """
        Register this handler's representer and constructor with a YAML instance.

        Parameters
        ----------
        yaml : ``ruamel.yaml.YAML``
            The YAML instance to register the handler with.
        """
        if cls.__tag__ is None or cls.__type__ is None:
            raise ValueError(f"{cls.__name__} must define both __tag__ and __type__.")

        if isinstance(cls.__type__, (list, tuple, set)):
            for t in cls.__type__:
                yaml.representer.add_representer(t, cls.to_yaml)
                yaml.constructor.add_constructor(cls.__tag__, cls.from_yaml)
        else:
            yaml.representer.add_representer(cls.__type__, cls.to_yaml)
            yaml.constructor.add_constructor(cls.__tag__, cls.from_yaml)


class AstropyQuantityHandler(_YAMLHandler):
    """Astropy Quantity handler for YAML serialization/deserialization."""

    __tag__ = "!astropy_quantity"
    __type__ = u.Quantity

    @staticmethod
    def to_yaml(representer, obj: u.Quantity):
        value = obj.value
        if hasattr(value, "tolist"):
            value = value.tolist()

        return representer.represent_mapping(
            AstropyQuantityHandler.__tag__,
            {
                "value": value,
                "unit": obj.unit.to_string(),
            },
        )

    @staticmethod
    def from_yaml(loader, node):
        data = CommentedMap()
        loader.construct_mapping(node, maptyp=data, deep=True)

        value = data["value"]
        if isinstance(value, (list, tuple)):
            value = np.asarray(value)

        return value * u.Unit(data["unit"])


class AstropyCosmologyHandler(_YAMLHandler):
    """
    YAML handler for Astropy built-in cosmologies.

    Stores only the cosmology name (e.g., "Planck18") and
    reconstructs using Astropy's cosmology registry.
    """

    __tag__ = "!astropy_cosmology"
    __type__ = [FlatLambdaCDM, FLRW]

    @staticmethod
    def to_yaml(representer, obj: FLRW):
        # Require that cosmology has a name
        if obj.name is None:
            raise ValueError("Only named cosmologies can be serialized. Custom cosmologies are not supported.")

        return representer.represent_mapping(
            AstropyCosmologyHandler.__tag__,
            {"name": obj.name},
        )

    @staticmethod
    def from_yaml(loader, node):
        from astropy import cosmology

        data = CommentedMap()
        loader.construct_mapping(node, maptyp=data, deep=True)

        name = data["name"]

        try:
            return getattr(cosmology, name)
        except AttributeError as exp:
            raise ValueError(f"Unknown built-in cosmology: {name}") from exp


class NumpyArrayHandler(_YAMLHandler):
    """NumPy array handler for YAML serialization/deserialization."""

    __tag__ = "!ndarray"
    __type__ = np.ndarray

    @staticmethod
    def to_yaml(representer, obj: np.ndarray):
        # We store both dtype and shape to ensure safe reconstruction
        return representer.represent_mapping(
            NumpyArrayHandler.__tag__, {"dtype": str(obj.dtype), "shape": obj.shape, "data": obj.tolist()}
        )

    @staticmethod
    def from_yaml(loader, node):
        data = CommentedMap()
        loader.construct_mapping(node, maptyp=data, deep=True)

        # Validate required fields
        if not all(k in data for k in ("dtype", "shape", "data")):
            raise ValueError(f"Invalid ndarray YAML mapping: {data}")

        arr = np.array(data["data"], dtype=np.dtype(data["dtype"]))

        # Optionally enforce shape
        if tuple(arr.shape) != tuple(data["shape"]):
            try:
                arr = arr.reshape(data["shape"])
            except Exception as e:
                raise ValueError(f"Shape mismatch when reconstructing ndarray: {e}") from e

        return arr


class PathHandler(_YAMLHandler):
    """Path handler for YAML serialization/deserialization."""

    __tag__ = "!path"
    __type__ = Path

    @staticmethod
    def to_yaml(representer, obj: Path):
        data = {"absolute": obj.is_absolute(), "parts": list(obj.parts)}
        if obj.drive:
            data["drive"] = obj.drive
        return representer.represent_mapping(PathHandler.__tag__, data)

    @staticmethod
    def from_yaml(loader, node):
        data = CommentedMap()
        loader.construct_mapping(node, maptyp=data, deep=True)
        parts = data.get("parts", [])
        path = Path(*parts)
        if data.get("absolute", False) and not path.is_absolute():
            path = path.resolve()
        return path


def get_unit_compatible_yaml() -> "YAML":
    """
    Get a YAML instance configured for unyt compatibility.

    This function creates a `ruamel.yaml.YAML` instance and registers
    custom representers and constructors for unyt types.

    Returns
    -------
    ``ruamel.yaml.YAML``
        A YAML instance with unyt support.
    """
    from ruamel.yaml import YAML

    yaml = YAML(typ="rt")

    AstropyQuantityHandler.register(yaml)
    PathHandler.register(yaml)
    NumpyArrayHandler.register(yaml)
    AstropyCosmologyHandler.register(yaml)

    return yaml


unit_compatible_yaml = get_unit_compatible_yaml()
