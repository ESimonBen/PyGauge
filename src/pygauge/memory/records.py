# records.py
from dataclasses import dataclass

@dataclass(frozen=True)
class MemorySnapshot:
    allocated_bytes: int
    reserved_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.allocated_bytes < 0:
            raise ValueError("Allocated memory cannot be negative")

        if self.reserved_bytes is not None and self.reserved_bytes < 0:
            raise ValueError("Reserved bytes cannot be negative")

@dataclass(frozen=True)
class MemoryRecord:
    invocation_id: int
    module_name: str
    module_type: str
    start: MemorySnapshot
    end: MemorySnapshot
    backend: str

    @property
    def delta_bytes(self) -> int:
        return self.end.allocated_bytes - self.start.allocated_bytes

    @property
    def delta_mib(self) -> float:
        return self.delta_bytes / (1024 ** 2)

    @property
    def reserved_delta_bytes(self) -> int | None:
        if self.end.reserved_bytes is None or self.start.reserved_bytes is None:
            return None

        return self.end.reserved_bytes - self.start.reserved_bytes

    @property
    def start_bytes(self) -> int:
        return self.start.allocated_bytes

    @property
    def end_bytes(self) -> int:
        return self.end.allocated_bytes