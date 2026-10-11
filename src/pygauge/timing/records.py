# records.py
from dataclasses import dataclass

@dataclass(frozen=True)
class TimingRecord:
    invocation_id: int
    module_name: str
    module_type: str
    elapsed_ns: int
    backend: str

    @property
    def elapsed_ms(self) -> float:
        return self.elapsed_ns / 1_000_000