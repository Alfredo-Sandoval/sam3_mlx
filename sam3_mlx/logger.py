"""Logging helpers ported from the official SAM3 surface."""

from __future__ import annotations

import logging
import os


LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


class ColoredFormatter(logging.Formatter):
    """A command-line formatter with different colors for each level."""

    def __init__(self):
        super().__init__()
        reset = "\033[0m"
        colors = {
            logging.DEBUG: f"{reset}\033[36m",
            logging.INFO: f"{reset}\033[32m",
            logging.WARNING: f"{reset}\033[33m",
            logging.ERROR: f"{reset}\033[31m",
            logging.CRITICAL: f"{reset}\033[35m",
        }
        fmt_str = (
            "{color}%(levelname)s %(asctime)s %(process)d "
            "%(filename)s:%(lineno)4d:{reset} %(message)s"
        )
        self.formatters = {
            level: logging.Formatter(fmt_str.format(color=color, reset=reset))
            for level, color in colors.items()
        }
        self.default_formatter = self.formatters[logging.INFO]

    def format(self, record):
        formatter = self.formatters.get(record.levelno, self.default_formatter)
        return formatter.format(record)


def get_logger(name, level=logging.INFO):
    """Return a configured command-line logger."""
    if "LOG_LEVEL" in os.environ:
        level_name = os.environ["LOG_LEVEL"].upper()
        if level_name not in LOG_LEVELS:
            raise ValueError(
                f"Invalid LOG_LEVEL: {level_name}, must be one of {list(LOG_LEVELS)}"
            )
        level = LOG_LEVELS[level_name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if not any(
        getattr(handler, "_sam3_mlx_colored", False) for handler in logger.handlers
    ):
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(ColoredFormatter())
        handler._sam3_mlx_colored = True
        logger.addHandler(handler)
    return logger
