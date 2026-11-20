"""
Registry linking verification tests to the models they verify.

Every verification test (or test class) is decorated with :func:`verifies`, which names the model(s) it verifies. The
decorator marks the test with ``pytest.mark.verification`` and records a :class:`VerificationCase`, so that cheap
always-on tests can check the bookkeeping without running any verification: that every verification test is registered
and that every model declared ``ModelStatus.VERIFIED`` has at least one verification test. Whether a test is *suitable*
evidence is not checked here; that is a judgement made when the change is reviewed.
"""

import importlib
import inspect
import pkgutil
import sys
from dataclasses import dataclass
from typing import Iterable

import pytest

from trilobite.models.core.base import Model
from trilobite.models.core.status import ModelStatus


@dataclass(frozen=True)
class VerificationCase:
    """One registered verification test (or test class)."""

    models: tuple
    test: str

    @property
    def model_names(self) -> tuple:
        """tuple of str: The names of the verified models."""
        return tuple(m.__name__ for m in self.models)


_CASES: list = []


def verifies(*models):
    """
    Declare which model(s) a verification test verifies.

    Parameters
    ----------
    *models : type
        The :class:`~trilobite.models.core.base.Model` subclass(es) the test verifies. At least one is required.

    Returns
    -------
    callable
        A decorator for a test function or class. It marks the test with ``pytest.mark.verification`` and registers
        the case.
    """
    if not models:
        raise TypeError("verifies() needs at least one model class.")
    for model in models:
        if not (inspect.isclass(model) and issubclass(model, Model)):
            raise TypeError(f"verifies() takes Model subclasses, got {model!r}.")

    def decorate(obj):
        case = VerificationCase(tuple(models), f"{obj.__module__}.{obj.__qualname__}")
        _CASES.append(case)
        obj.__verifies__ = case
        return pytest.mark.verification(obj)

    return decorate


def all_cases() -> list:
    """list of VerificationCase: Every case registered so far."""
    return list(_CASES)


def cases_for(model: type) -> list:
    """
    Return the cases registered for a model.

    Parameters
    ----------
    model : type
        A :class:`~trilobite.models.core.base.Model` subclass. Subclasses do not inherit their parents' cases, just as
        they do not inherit ``STATUS``.

    Returns
    -------
    list of VerificationCase
        The registered cases that verify ``model``.
    """
    return [case for case in _CASES if model in case.models]


def _test_modules() -> list:
    package = importlib.import_module(__package__)
    return [
        f"{__package__}.{info.name}" for info in pkgutil.iter_modules(package.__path__) if info.name.startswith("test_")
    ]


def load_all() -> list:
    """list of VerificationCase: Import every verification test module so that all cases are registered."""
    for name in _test_modules():
        importlib.import_module(name)
    return all_cases()


def unregistered_tests() -> list:
    """
    Find verification tests that do not carry a :func:`verifies` declaration.

    Returns
    -------
    list of str
        Dotted names of test functions and classes in the verification package without a registered case.
    """
    load_all()
    missing = []
    for name in _test_modules():
        module = sys.modules[name]
        for attr, obj in vars(module).items():
            if getattr(obj, "__module__", None) != module.__name__:
                continue
            is_test = (inspect.isfunction(obj) and attr.startswith("test")) or (
                inspect.isclass(obj) and attr.startswith("Test")
            )
            if is_test and not hasattr(obj, "__verifies__"):
                missing.append(f"{module.__name__}.{attr}")
    return missing


def all_models() -> list:
    """
    Collect the concrete models defined in the Trilobite package.

    Returns
    -------
    list of type
        Every non-abstract, non-private :class:`~trilobite.models.core.base.Model` subclass defined under
        ``trilobite``, sorted by name.
    """
    import trilobite.models as package

    for info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        importlib.import_module(info.name)
    found, stack = set(), list(Model.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls in found:
            continue
        found.add(cls)
        stack.extend(cls.__subclasses__())
    return sorted(
        (
            cls
            for cls in found
            if cls.__module__.startswith("trilobite.")
            and not inspect.isabstract(cls)
            and not cls.__name__.startswith("_")
        ),
        key=lambda cls: cls.__qualname__,
    )


def verification_problems(models: Iterable) -> list:
    """
    Check that every model declared ``VERIFIED`` has a verification test.

    Parameters
    ----------
    models : iterable of type
        The models to check.

    Returns
    -------
    list of str
        One message per ``VERIFIED`` model that has no registered verification test.
    """
    return [
        f"{cls.__name__}: declared VERIFIED but no verification test is registered for it"
        for cls in models
        if cls.STATUS is ModelStatus.VERIFIED and not cases_for(cls)
    ]
