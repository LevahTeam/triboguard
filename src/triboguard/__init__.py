"""TriboGuard: is there enough evidence here to name a mechanism?

A treatment that halves a cell population may have killed half the cells or
stopped half of them dividing. Those are different biological claims with
different follow-up experiments, and a single count of surviving cells cannot
tell them apart. This package decides when the distinction is supported by the
data, refuses to name a mechanism when it is not, and works out the cheapest
measurement that would settle it.
"""

from triboguard import abstention, boundary, design, inference, kinetics

__all__ = ["abstention", "boundary", "design", "inference", "kinetics"]
