"""Configuration loading.

Configuration is passed explicitly as a :class:`Config` object rather than read
from module-level globals, so that two experiments with different settings can
run in the same process without interfering with each other.

Resolution order for the config file:

1. An explicit path passed to :func:`load_config`.
2. The ``CQ_CONFIG`` environment variable.
3. ``config.yaml`` in the current working directory, or any parent directory.
4. The packaged defaults shipped with the library.

Secrets are never read from the config file. API credentials come from the
``CQ_API_KEY`` and ``CQ_API_SECRET`` environment variables only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = ["DEFAULT_CONFIG_PATH", "Config", "load_config"]

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "default_config.yaml"
"""Location of the defaults bundled inside the installed package."""

_CONFIG_FILENAME = "config.yaml"


@dataclass(frozen=True)
class Config:
    """An immutable view over a parsed configuration file.

    Attributes:
        raw: The parsed YAML document.
        path: Absolute path the configuration was loaded from.
    """

    raw: dict[str, Any]
    path: Path

    def __getitem__(self, key: str) -> Any:
        """Return a top-level section.

        Args:
            key: Top-level key, e.g. ``"risk"``.

        Returns:
            The value stored under ``key``.

        Raises:
            KeyError: If the key is absent.
        """
        return self.raw[key]

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Look up a nested value using dotted notation.

        Example:
            >>> config.get("risk.target_annual_vol", 0.2)  # doctest: +SKIP
            0.2

        Args:
            dotted_key: Path through the document, e.g. ``"risk.max_drawdown"``.
            default: Value returned when any part of the path is missing.

        Returns:
            The resolved value, or ``default``.
        """
        node: Any = self.raw
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    @property
    def data_root(self) -> Path:
        """Absolute path to the data directory, created if it does not exist.

        Relative paths in the config are resolved against the config file's own
        location, so moving the file moves its data with it.
        """
        root = Path(self.get("data.root", "./data")).expanduser()
        if not root.is_absolute():
            root = self.path.parent / root
        root.mkdir(parents=True, exist_ok=True)
        return root


def _discover_config(start: Path | None = None) -> Path | None:
    """Search the current directory and its parents for a config file.

    Args:
        start: Directory to start from. Defaults to the working directory.

    Returns:
        The first matching path, or ``None`` if there is none.
    """
    current = (start or Path.cwd()).resolve()
    for directory in (current, *current.parents):
        candidate = directory / _CONFIG_FILENAME
        if candidate.is_file():
            return candidate
    return None


def load_config(path: str | Path | None = None) -> Config:
    """Load a configuration file.

    Args:
        path: Explicit path to a YAML file. When ``None``, the resolution order
            described in the module docstring applies.

    Returns:
        The parsed configuration.

    Raises:
        FileNotFoundError: If an explicit path was given but does not exist.
    """
    if path is not None:
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"config file not found: {resolved}")
    elif env_path := os.environ.get("CQ_CONFIG"):
        resolved = Path(env_path).expanduser().resolve()
    else:
        resolved = _discover_config() or DEFAULT_CONFIG_PATH

    with open(resolved, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return Config(raw=raw, path=resolved)
