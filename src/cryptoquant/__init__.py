"""cryptoquant: a local, zero-cost research and trading stack for crypto.

The package is organised in three layers:

* :mod:`cryptoquant.data` - acquiring and storing market data.
* :mod:`cryptoquant.research` - features, labels and, above all, validation.
* :mod:`cryptoquant.trading` - backtesting, risk sizing and live execution.

:mod:`cryptoquant.pipeline` wires them together into an end-to-end workflow.

Warning:
    Nothing in this package is financial advice, and no backtest result implies
    a live result. See the README for the validation gates that any strategy
    should clear before it is trusted with capital.
"""

from importlib import metadata

try:
    __version__ = metadata.version("cryptoquant")
except metadata.PackageNotFoundError:  # running from a source checkout
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
