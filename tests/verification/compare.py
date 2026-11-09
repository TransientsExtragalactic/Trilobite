"""
Comparison helpers for verification tests.

A test receives a :class:`Verifier` through the ``verify`` fixture. Its comparison methods assert like ordinary
assertions, but also attach the outcome (metric, value, tolerance, pass or fail, the models verified and their
``ModelStatus``) to the test report, from which ``tests/conftest.py`` builds the end-of-run summary and the
``verification_report.json`` artifact. A failing comparison is recorded before it raises.
"""

import json

import numpy as np

#: Deviation metrics accepted by :meth:`Verifier.agrees`.
#:
#: ``rel``: ``max |actual / expected - 1|``. ``abs``: ``max |actual - expected|``. ``dex``:
#: ``max |log10(actual / expected)|``.
METRICS = ("rel", "abs", "dex")


def deviation(actual, expected, metric: str = "rel") -> float:
    """
    Largest deviation between two arrays under a metric.

    Parameters
    ----------
    actual, expected : array-like
        The values to compare. They are broadcast against each other.
    metric : {"rel", "abs", "dex"}, optional
        The deviation metric; see :data:`METRICS`.

    Returns
    -------
    float
        The largest deviation, or ``inf`` if any compared value is not finite.
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}.")
    a = np.asarray(actual, dtype=float)
    e = np.asarray(expected, dtype=float)
    with np.errstate(all="ignore"):
        if metric == "rel":
            dev = np.abs(a / e - 1.0)
        elif metric == "abs":
            dev = np.abs(a - e)
        else:
            dev = np.abs(np.log10(a / e))
    dev = np.where(np.isfinite(a) & np.isfinite(e) & np.isfinite(dev), dev, np.inf)
    return float(np.max(dev))


def _number(value):
    """JSON-safe number: finite floats stay numbers, everything else becomes a string such as ``"inf"``."""
    value = float(value)
    return value if np.isfinite(value) else str(value)


class Verifier:
    """
    Assertion helper that records every comparison on the test report.

    Parameters
    ----------
    node : pytest.Item
        The running test item; recorded outcomes are attached to its ``user_properties``.
    """

    def __init__(self, node):
        self._node = node
        function = getattr(node, "function", None)
        owner = getattr(node, "cls", None)
        self.case = getattr(function, "__verifies__", None) or getattr(owner, "__verifies__", None)

    def _record(self, **entry) -> None:
        entry["test"] = self._node.nodeid
        if self.case is not None:
            entry["models"] = [{"name": m.__name__, "status": m.STATUS.value} for m in self.case.models]
        self._node.user_properties.append(("verification", json.dumps(entry)))

    def agrees(self, actual, expected, *, tol: float, metric: str = "rel", label: str = "") -> float:
        """
        Assert that two values agree to within a tolerance.

        Parameters
        ----------
        actual, expected : array-like
            The model result and the reference.
        tol : float
            The largest acceptable deviation under ``metric``.
        metric : {"rel", "abs", "dex"}, optional
            The deviation metric; see :data:`METRICS`.
        label : str, optional
            What is being compared, for the report and the failure message.

        Returns
        -------
        float
            The measured deviation.
        """
        dev = deviation(actual, expected, metric)
        passed = bool(dev <= tol)
        self._record(label=label, metric=metric, value=_number(dev), tolerance=_number(tol), passed=passed)
        assert passed, f"{label or 'comparison'}: {metric} deviation {dev:.4g} exceeds the tolerance {tol:.4g}"
        return dev

    def within(self, value, low: float, high: float, *, label: str = "") -> float:
        """
        Assert that a value lies inside a closed interval.

        Parameters
        ----------
        value : float
            The quantity under test.
        low, high : float
            The interval, for example a published value with its uncertainty.
        label : str, optional
            What is being compared, for the report and the failure message.

        Returns
        -------
        float
            ``value`` as a float.
        """
        value = float(value)
        passed = bool(low <= value <= high)
        self._record(
            label=label, metric="within", value=_number(value), tolerance=[_number(low), _number(high)], passed=passed
        )
        assert passed, f"{label or 'value'} = {value:.6g} lies outside [{low:.6g}, {high:.6g}]"
        return value


def parse_records(user_properties) -> list:
    """list of dict: The verification entries recorded in a test report's ``user_properties``."""
    return [json.loads(value) for name, value in user_properties if name == "verification"]
