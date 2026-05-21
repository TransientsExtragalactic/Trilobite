"""Logging utilities for the trilobite ecosystem.

This module defines and configures the global logger (:attr:`trilobite_logger`),
which serves as the main entry point for logging messages across the trilobite
codebase. Logging behavior—such as log level, formatting, and enable/disable
switches—is controlled by the central configuration (:attr:`~trilobite.utils.config.trilobite_config`) under
the ``logging.main`` namespace.

"""

import logging
from typing import TypeVar

from .config import trilobite_config

Instance = TypeVar("Instance")

# ======================== #
# Configure the global log #
# ======================== #
trilobite_logger = logging.getLogger("trilobite")
""": logging.Logger: The main logger for the trilobite package."""
trilobite_logger.setLevel(
    getattr(logging, trilobite_config["system.logging.main.level"])
)  # Allow DEBUG, handlers filter final output
trilobite_logger.propagate = False  # Avoid duplicate logs to root logger

# Don't permit double handler adding.
if not trilobite_logger.hasHandlers():
    # Console handler with minimal formatting
    console_handler = logging.StreamHandler()
    console_fmt = trilobite_config["system.logging.main.format"]
    console_handler.setFormatter(logging.Formatter(console_fmt))
    trilobite_logger.addHandler(console_handler)
