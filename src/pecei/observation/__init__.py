"""Observation layer: a derived, leak-free view of the world for one observer.

Strictly separated from ground truth: an Observation contains only cells within
the observer's conical FOV (range + half-angle + line-of-sight occlusion) plus
the observer's own body, as copied snapshots — never live World/Grid references.
"""
from .observation import (
    DEFAULT_HALF_ANGLE,
    DEFAULT_RANGE,
    CellView,
    Observation,
    observe,
)

__all__ = [
    "DEFAULT_HALF_ANGLE",
    "DEFAULT_RANGE",
    "CellView",
    "Observation",
    "observe",
]
