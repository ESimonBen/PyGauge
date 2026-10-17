# records.py
from dataclasses import dataclass

from pygauge.memory import MemorySnapshot

@dataclass(frozen=True)
class PerformanceRecord:
    invocation_id: int
    module_name: str
    module_type: str
    elapsed_ns: int | None
    timing_backend: str | None
    memory_start: MemorySnapshot | None = None
    memory_end: MemorySnapshot | None = None
    memory_backend: str | None = None

    @property
    def elapsed_ms(self) -> float | None:
        if self.elapsed_ns is None:
            return None

        return self.elapsed_ns / 1_000_000

    @property
    def memory_delta_bytes(self) -> int:
        if self.memory_start is None or self.memory_end is None:
            return None

        return self.memory_end.allocated_bytes - self.memory_start.allocated_bytes

    @property
    def reserved_delta_bytes(self) -> int:
        if self.memory_start is None or self.memory_end is None or self.memory_start.reserved_bytes is None or self.memory_end.reserved_bytes is None:
            return None

        return self.memory_end.reserved_bytes - self.memory_start.reserved_bytes

@dataclass(frozen=True)
class ModuleSummary:
    module_name: str
    module_type: str
    invocation_count: int
    timing_backend: str
    memory_backend: str | None

    total_elapsed_ns: int
    mean_elapsed_ns: float
    median_elapsed_ns: int
    std_elapsed_ns:  float
    min_elapsed_ns: int
    max_elapsed_ns: int
    p95_elapsed_ns: int

    memory_sample_count: int
    mean_memory_delta_bytes: float | None
    min_memory_delta_bytes: int | None
    max_memory_delta_bytes: int | None
    reserved_sample_count: int
    mean_reserved_delta_bytes: float | None

    @property
    def total_elapsed_ms(self):
        return self.total_elapsed_ns / 1_000_000

    @property
    def mean_elapsed_ms(self):
        return self.mean_elapsed_ns / 1_000_000