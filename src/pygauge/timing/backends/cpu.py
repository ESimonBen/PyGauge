# cpu.py
import time
from typing import Callable

from .base import TimingBackend

class CPUTimingBackend(TimingBackend):
    def __init__(self, clock: Callable[[], int] = time.perf_counter_ns):
        self._clock = clock

    @property
    def name(self) -> str:
        return "cpu"

    def start(self) -> int:
        return self._clock()

    def stop(self, start_token: int) -> tuple[int, int]:
        return start_token, self._clock()

    def synchronize(self) -> None:
        pass

    def elapsed_ns(self, completed_token: tuple[int, int]) -> int:
        start_ns, end_ns = completed_token
        elapsed_ns = end_ns - start_ns

        if elapsed_ns < 0:
            raise ValueError("Negative CPU elapsed time")

        return elapsed_ns

    def reset(self) -> None:
        pass