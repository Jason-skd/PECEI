"""Engine layer: round scheduling. Imports world (mutable state) only.

One ``apply`` = one round (resolve an action, then tick the environment). The
DSL interpreter (M4) maps ``act`` onto ``RoundEngine.apply``.
"""
from .round_engine import BrittleFailure, RoundEngine

__all__ = ["BrittleFailure", "RoundEngine"]
