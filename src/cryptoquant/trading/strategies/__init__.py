"""Signal generators.

Import strategies from here rather than from their defining modules, so that
internal reorganisation does not break user code.
"""

from .adaptive_multifactor import AdaptiveMultiFactor
from .base import Strategy
from .ensemble import Ensemble
from .meta_ml import MetaLabelML
from .moving_average import MovingAverageCrossover
from .trend_carry import TrendCarry

__all__ = [
    "AdaptiveMultiFactor",
    "Ensemble",
    "MetaLabelML",
    "MovingAverageCrossover",
    "Strategy",
    "TrendCarry",
]
