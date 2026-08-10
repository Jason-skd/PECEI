"""Memory evolution package: three-stage evolving memory for the agent."""

from pecei.memory.evolution import (
    BufferItem,
    MemoryEvolution,
    DEFAULT_ALPHA,
    DEFAULT_BUFFER_CAPACITY,
    DEFAULT_DEFRAG_THRESHOLD,
)

__all__ = [
    "BufferItem",
    "MemoryEvolution",
    "DEFAULT_ALPHA",
    "DEFAULT_BUFFER_CAPACITY",
    "DEFAULT_DEFRAG_THRESHOLD",
]
