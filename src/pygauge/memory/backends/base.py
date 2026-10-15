# base.py
from typing import Protocol

from pygauge.memory.records import MemorySnapshot

class MemoryBackend(Protocol):
    @property
    def name(self) -> str:
        ...

    def snapshot(self) -> MemorySnapshot:
        """Return a snapshot of the memory."""
        ...
