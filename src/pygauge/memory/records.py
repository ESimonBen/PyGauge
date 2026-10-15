# records.py
from dataclasses import dataclass

@dataclass(frozen=True)
class MemoryRecord:
    invocation_id: int
    module_name: str
    module_type: str
    start_bytes: int
    end_bytes: int
    backend: str

    @property
    def delta_bytes(self) -> int:
        return self.end_bytes - self.start_bytes

    @property
    def delta_mib(self) -> float:
        return self.delta_bytes / (1024 ** 2)
