"""
Development-stage markers for Trilobite models.

Every :class:`~trilobite.models.core.base.Model` carries a ``STATUS`` class attribute, a member of
:class:`ModelStatus`, which records how far along its development the model is and whether it has been verified.
The status is surfaced in the documentation (model directory badges and API pages) and, for the stages that warrant
caution, at runtime.
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
            STATUS = ModelStatus.VERIFIED
            STATUS_NOTE = "Reproduces the light curves of Author et al. (2016)."

    Each member's value is the short lower-case key used in the documentation search index. Members also expose
    a human-readable :attr:`label` and :attr:`description`.

    The members, in order of increasing caution:

    - ``VERIFIED``: backed by a verification test (see ``tests/verification``) that compares the model with
      published results or another independent reference. A model may only be marked ``VERIFIED`` if such a test
      exists; whether the test is suitable is judged when the change is reviewed.
    - ``NOVEL``: new physics with no published counterpart to check against, so it cannot be verified against the
      literature yet.
    - ``UNVERIFIED``: implemented and tested, but not yet verified. This is the default for any model that does not
      declare a status.
    - ``EXPERIMENTAL``: under active development; the physics, parameters or API may change without notice.
      Instantiating an experimental model emits a :class:`UserWarning`.
    - ``DEPRECATED``: scheduled for removal; prefer the replacement named in ``STATUS_NOTE``. Instantiating a
      deprecated model emits a :class:`UserWarning`.
    """

    VERIFIED = "verified"
    NOVEL = "novel"
    UNVERIFIED = "unverified"
    EXPERIMENTAL = "experimental"
    DEPRECATED = "deprecated"

    @property
    def label(self) -> str:
        """str: Display name, e.g. ``"Verified"``."""
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
            The status, or its string key (e.g. ``"verified"``).

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
    ModelStatus.VERIFIED: "Backed by a verification test against published results or an independent reference.",
    ModelStatus.NOVEL: "New physics with no published counterpart to verify against.",
    ModelStatus.UNVERIFIED: "Implemented and tested, but not yet verified.",
    ModelStatus.EXPERIMENTAL: "Under active development; physics, parameters or API may change without notice.",
    ModelStatus.DEPRECATED: "Scheduled for removal; prefer the suggested replacement.",
}

_BADGE_COLORS = {
    ModelStatus.VERIFIED: "success",
    ModelStatus.NOVEL: "info",
    ModelStatus.UNVERIFIED: "secondary",
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
