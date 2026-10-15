# cpu.py
import os
import psutil

from .base import MemoryBackend

class CPUMemoryBackend(MemoryBackend):
    def __init__(self):
        self._process = psutil.Process(os.getpid())

    @property
    def name(self) -> str:
        return "cpu"

    def snapshot(self) -> int:
        return self._process.memory_info().rss