# base.py
from typing import Protocol

class MemoryBackend(Protocol):
    @property
    def name(self) -> str:
        ...

    def snapshot(self) -> int:
        """Return the currently observed memory usage in bytes."""
        ...
