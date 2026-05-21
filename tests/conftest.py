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
    Adds the following CLI options to pytest:

    - ``--diagnostic_plots``: If set, enables the generation of diagnostic plots during test runs. By
       default, these are disabled to speed up testing. The plots will be saved in the ``./diagnostic_plots/``
       directory unless ``--diagnostic_plots_dir`` is specified.
    - ``--diagnostic_plots_dir``: Specifies the directory where diagnostic plots will be saved if
       ``--diagnostic_plots`` is enabled. Defaults to ``./diagnostic_plots/``.
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


# -------------------------------------------------- #
# Add Global Fixtures                                #
# -------------------------------------------------- #
# Here we define the global fixtures that will be used across multiple test modules.
# These fixtures can be overridden or extended in individual test modules as needed.
@pytest.fixture(scope="session")
def timestamp():
    """
    Fixture to provide a timestamp for the test session.
    """
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d_%H%M%S")


@pytest.fixture(scope="session")
def diagnostic_plots(request):
    """
    Fixture to determine if diagnostic plots should be generated during tests.
    """
    # Check if we've been asked to perform the tests with diagnostic plots. If we
    # aren't, we don't have any work to do here.
    _diagnostic_plots_required = request.config.getoption("--diagnostic_plots")
    return _diagnostic_plots_required


@pytest.fixture(scope="session")
def diagnostic_plots_dir(request, timestamp):
    """
    Fixture to provide the directory for saving diagnostic plots.
    """
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
