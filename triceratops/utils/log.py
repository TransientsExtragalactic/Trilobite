"""Logging utilities for the triceratops ecosystem.

This module defines and configures the global logger (:attr:`triceratops_logger`),
which serves as the main entry point for logging messages across the triceratops
codebase. Logging behavior—such as log level, formatting, and enable/disable
switches—is controlled by the central configuration (:attr:`~triceratops.utils.config.triceratops_config`) under
the ``logging.main`` namespace.

"""

import logging
from typing import TypeVar

from .config import triceratops_config

Instance = TypeVar("Instance")

# ======================== #
# Configure the global log #
# ======================== #
triceratops_logger = logging.getLogger("triceratops")
""": logging.Logger: The main logger for the triceratops package."""
triceratops_logger.setLevel(
    getattr(logging, triceratops_config["system.logging.main.level"])
)  # Allow DEBUG, handlers filter final output
triceratops_logger.propagate = False  # Avoid duplicate logs to root logger

# Don't permit double handler adding.
if not triceratops_logger.hasHandlers():
    # Console handler with minimal formatting
    console_handler = logging.StreamHandler()
    console_fmt = triceratops_config["system.logging.main.format"]
    console_handler.setFormatter(logging.Formatter(console_fmt))
    triceratops_logger.addHandler(console_handler)
