"""
Development-stage markers for Trilobite models.

Every :class:`~trilobite.models.core.base.Model` carries a ``STATUS`` class attribute, a member of
:class:`ModelStatus`, which records how far along its development and verification the model is. The status is
surfaced in the documentation (model directory badges and API pages) and, for the stages that warrant caution,
at runtime.
"""

import warnings
from enum import Enum

__all__ = ["ModelStatus"]


class ModelStatus(Enum):
    """
    Development stage of a model.

    The status is declared on a model with the ``STATUS`` class attribute:

    .. code-block:: python

        class MyModel(Model):
            STATUS = ModelStatus.VALIDATED
            STATUS_NOTE = "Reproduces the closed-form solution of Chevalier (1982) to 1e-6."

    Each member's value is the short lower-case key used in the documentation search index. Members also expose
    a human-readable :attr:`label` and :attr:`description`.

    The members, in order of increasing caution:

    - ``VALIDATED``: verified against analytic limits, independent codes, or published results.
    - ``UNVALIDATED``: implemented and tested, but not yet independently verified. This is the default for any
      model that does not declare a status.
    - ``EXPERIMENTAL``: under active development; the physics, parameters or API may change without notice.
      Instantiating an experimental model emits a :class:`UserWarning`.
    - ``DEPRECATED``: scheduled for removal; prefer the replacement named in ``STATUS_NOTE``. Instantiating a
      deprecated model emits a :class:`UserWarning`.
    """

    VALIDATED = "validated"
    UNVALIDATED = "unvalidated"
    EXPERIMENTAL = "experimental"
    DEPRECATED = "deprecated"

    @property
    def label(self) -> str:
        """str: Display name, e.g. ``"Validated"``."""
        return self.value.capitalize()

    @property
    def description(self) -> str:
        """str: One-sentence explanation of the stage."""
        return _DESCRIPTIONS[self]

    @property
    def badge_color(self) -> str:
        """str: ``sphinx-design`` badge colour role used when rendering the status in the documentation."""
        return _BADGE_COLORS[self]

    @property
    def warns(self) -> bool:
        """bool: Whether instantiating a model at this stage emits a warning."""
        return self in (ModelStatus.EXPERIMENTAL, ModelStatus.DEPRECATED)

    @classmethod
    def coerce(cls, value) -> "ModelStatus":
        """
        Convert a member or its string value (case-insensitive) to a :class:`ModelStatus`.

        Parameters
        ----------
        value : ModelStatus or str
            The status, or its string key (e.g. ``"validated"``).

        Returns
        -------
        ModelStatus
            The corresponding :class:`ModelStatus` member.

        Raises
        ------
        ValueError
            If ``value`` is not a recognised status.
        """
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).lower())
        except ValueError:
            valid = ", ".join(repr(m.value) for m in cls)
            raise ValueError(f"Invalid model status {value!r}. Valid statuses are: {valid}.") from None


_DESCRIPTIONS = {
    ModelStatus.VALIDATED: "Verified against analytic limits, independent codes, or published results.",
    ModelStatus.UNVALIDATED: "Implemented and tested, but not yet independently verified.",
    ModelStatus.EXPERIMENTAL: "Under active development; physics, parameters or API may change without notice.",
    ModelStatus.DEPRECATED: "Scheduled for removal; prefer the suggested replacement.",
}

_BADGE_COLORS = {
    ModelStatus.VALIDATED: "success",
    ModelStatus.UNVALIDATED: "secondary",
    ModelStatus.EXPERIMENTAL: "warning",
    ModelStatus.DEPRECATED: "danger",
}


def _warn_if_needed(cls) -> None:
    """Emit the instantiation warning for ``cls`` if its status calls for one."""
    status = cls.STATUS
    if not status.warns:
        return
    note = f" {cls.STATUS_NOTE}" if cls.STATUS_NOTE else ""
    warnings.warn(
        f"{cls.__name__} is {status.value}: {status.description}{note}",
        UserWarning,
        stacklevel=3,
    )
