"""Logging configuration.

A single place to set up logging so that scripts, the CLI and notebooks all
produce the same format. Libraries should never configure logging on import;
only entry points call :func:`setup_logging`.
"""

from __future__ import annotations

import logging
import sys

__all__ = ["setup_logging"]

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_NOISY_LIBRARIES = ("urllib3", "matplotlib", "lightgbm", "numexpr")


def setup_logging(level: int | str = logging.INFO, quiet_libraries: bool = True) -> None:
    """Configure root logging for an entry point.

    Args:
        level: Logging level for the application's own loggers, either a
            ``logging`` constant or its name.
        quiet_libraries: If ``True``, raise third-party loggers to WARNING so
            that their debug output does not drown the application's.
    """
    logging.basicConfig(
        level=level,
        format=_FORMAT,
        datefmt=_DATE_FORMAT,
        stream=sys.stderr,
        force=True,
    )
    if quiet_libraries:
        for name in _NOISY_LIBRARIES:
            logging.getLogger(name).setLevel(logging.WARNING)
