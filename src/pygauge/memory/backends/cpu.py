# cpu.py
import os
import psutil

from .base import MemoryBackend
from pygauge.memory.records import MemorySnapshot

class CPUMemoryBackend(MemoryBackend):
    def __init__(self, process: psutil.Process | None = None):
        self._process = (process if process is not None else psutil.Process(os.getpid()))

    @property
    def name(self) -> str:
        return "cpu"

    def snapshot(self) -> MemorySnapshot:
        rss = self._process.memory_info().rss

        return MemorySnapshot(allocated_bytes=rss)