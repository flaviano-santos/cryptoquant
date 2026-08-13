"""Research primitives: features, labels and validation.

:mod:`~cryptoquant.research.validation` is the important one. Features and
labels determine whether a model *can* learn something; validation determines
whether you should believe it did.
"""

from . import features, labeling, validation

__all__ = ["features", "labeling", "validation"]
