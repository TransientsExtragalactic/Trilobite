"""
Test configuration for the Trilobite project.

This module sets up fixtures and configurations for running tests
using the pytest framework for the Trilobite project. Please read the documentation and
associated comments for more details on how to use and extend these configurations.
"""

from pathlib import Path

import pytest


# ---------------------------------- #
# CLI Options                        #
# ---------------------------------- #
# These are CLI options which enrich the behavior
# and control of pytest for this suite of tests.
def pytest_addoption(parser):
    """
    Add the following CLI options to pytest.

    - ``--diagnostic_plots``: If set, enables the generation of diagnostic plots during test runs. By
       default, these are disabled to speed up testing. The plots will be saved in the ``./diagnostic_plots/``
       directory unless ``--diagnostic_plots_dir`` is specified.
    - ``--diagnostic_plots_dir``: Specifies the directory where diagnostic plots will be saved if
       ``--diagnostic_plots`` is enabled. Defaults to ``./diagnostic_plots/``.
    - ``--run-verification``: Also runs the model verification tests in ``tests/verification``, which are
       deselected by default. Selecting the marker with ``-m verification`` has the same effect.
    - ``--verification-report``: Where a verification run writes its JSON report. Defaults to
       ``verification_report.json``.
    """
    parser.addoption(
        "--diagnostic_plots",
        action="store_true",
        default=False,
        help="Enable generation of diagnostic plots during test runs.",
    )
    parser.addoption(
        "--diagnostic_plots_dir",
        action="store",
        default="./diagnostic_plots/",
        help="Directory to save diagnostic plots if enabled.",
    )
    parser.addoption(
        "--run-verification",
        action="store_true",
        default=False,
        help="Also run the model verification tests in tests/verification (deselected by default).",
    )
    parser.addoption(
        "--verification-report",
        action="store",
        default="verification_report.json",
        help="Where to write the JSON report of a verification run.",
    )


# -------------------------------------------------- #
# Model verification                                 #
# -------------------------------------------------- #
# Verification tests (tests/verification) compare the models with published results or independent references. They are
# deselected unless asked for, either with ``--run-verification`` or by selecting the marker with ``-m verification``.
# Every comparison a verification test makes is recorded on its report (see ``verification.compare``); the records
# are gathered here into an end-of-run summary and a JSON report.
_VERIFICATION_DIR = Path(__file__).resolve().parent / "verification"
_VERIFICATION_RECORDS = []


def _verification_enabled(config):
    """Whether the verification tests were asked for, by flag or by selecting the marker."""
    markexpr = config.getoption("markexpr", default="") or ""
    return bool(config.getoption("--run-verification") or "verification" in markexpr)


def pytest_report_header(config):
    """State in the run header whether verification is enabled."""
    if _verification_enabled(config):
        return "model verification: enabled"
    return "model verification: deselected (enable with --run-verification or -m verification)"


def pytest_collection_modifyitems(config, items):
    """Mark everything under tests/verification and deselect it unless verification was asked for."""
    enabled = _verification_enabled(config)
    selected, deselected = [], []
    for item in items:
        if _VERIFICATION_DIR in Path(str(item.fspath)).resolve().parents:
            item.add_marker(pytest.mark.verification)
        if item.get_closest_marker("verification") and not enabled:
            deselected.append(item)
        else:
            selected.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


def pytest_runtest_logreport(report):
    """Gather the comparisons a verification test attached to its report."""
    if report.when == "call":
        for name, value in report.user_properties:
            if name == "verification":
                import json

                _VERIFICATION_RECORDS.append(json.loads(value))


def pytest_terminal_summary(terminalreporter):
    """Print every recorded verification comparison at the end of the run."""
    if not _VERIFICATION_RECORDS:
        return
    terminalreporter.section("model verification")
    for entry in _VERIFICATION_RECORDS:
        models = ", ".join(f"{m['name']} [{m['status']}]" for m in entry.get("models", [])) or "unregistered"
        tolerance = entry["tolerance"]
        outcome = "PASS" if entry["passed"] else "FAIL"
        what = entry["label"] or entry["test"]
        terminalreporter.write_line(
            f"{outcome}  {models}: {what}  {entry['metric']} = {entry['value']}  (tolerance {tolerance})"
        )


def pytest_sessionfinish(session):
    """Write the JSON verification report (on the controlling process only)."""
    if not _VERIFICATION_RECORDS or hasattr(session.config, "workerinput"):
        return
    import json
    import platform
    import time

    path = Path(session.config.getoption("--verification-report"))
    report = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "n_comparisons": len(_VERIFICATION_RECORDS),
        "n_failed": sum(not entry["passed"] for entry in _VERIFICATION_RECORDS),
        "results": _VERIFICATION_RECORDS,
    }
    path.write_text(json.dumps(report, indent=2) + "\n")


# -------------------------------------------------- #
# Add Global Fixtures                                #
# -------------------------------------------------- #
# Here we define the global fixtures that will be used across multiple test modules.
# These fixtures can be overridden or extended in individual test modules as needed.
@pytest.fixture(scope="session")
def timestamp():
    """Fixture to provide a timestamp for the test session."""
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d_%H%M%S")


@pytest.fixture(scope="session")
def diagnostic_plots(request):
    """Fixture to determine if diagnostic plots should be generated during tests."""
    # Check if we've been asked to perform the tests with diagnostic plots. If we
    # aren't, we don't have any work to do here.
    _diagnostic_plots_required = request.config.getoption("--diagnostic_plots")
    return _diagnostic_plots_required


@pytest.fixture(scope="session")
def diagnostic_plots_dir(request, timestamp):
    """Fixture to provide the directory for saving diagnostic plots."""
    # Check if we've been asked to perform the tests with diagnostic plots. If we
    # aren't, we don't have any work to do here.
    _diagnostic_plots_required = request.config.getoption("--diagnostic_plots")
    if not _diagnostic_plots_required:
        return None

    # Get the directory from the CLI option
    _dir = Path(request.config.getoption("--diagnostic_plots_dir"))

    # Append the timestamp to ensure uniqueness
    _dir = _dir / f"run_{timestamp}"

    # Create the directory if it doesn't exist
    _dir.mkdir(parents=True, exist_ok=True)

    request.config.pluginmanager.get_plugin("terminalreporter").write_line(
        f"[pytest] Diagnostic plots directory: {_dir}"
    )

    return _dir
