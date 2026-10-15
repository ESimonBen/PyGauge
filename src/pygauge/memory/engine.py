# engine.py
from dataclasses import dataclass

from .records import MemoryRecord
from pygauge.events import ModuleEvent
from .backends import MemoryBackend, CPUMemoryBackend

@dataclass
class ActiveMemory:
    event: ModuleEvent
    start_bytes: int

class MemoryEngine:
    def __init__(self, backend: MemoryBackend | None = None):
        self._backend = backend or CPUMemoryBackend()
        self._active: dict[int, ActiveMemory] = {}
        self._records: list[MemoryRecord] = []

    def start(self, event: ModuleEvent) -> None:
        invocation_id = event.invocation_id

        if invocation_id in self._active:
            raise ValueError(f"Invocation {invocation_id} is already active")

        if any (r.invocation_id == invocation_id for r in self._records):
            raise ValueError(f"Invocation {invocation_id} is was already recorded")

        start_bytes = self._backend.snapshot()

        self._active[invocation_id] = ActiveMemory(event=event, start_bytes=start_bytes)

    def stop(self, event: ModuleEvent) -> MemoryRecord:
        invocation_id = event.invocation_id

        if invocation_id not in self._active:
            raise ValueError(f"Invocation {invocation_id} was never started")

        active = self._active[invocation_id]

        if active.event.module is not event.module:
            raise ValueError("Module error mismatch")

        if active.event.name != event.name or active.event.module_type != event.module_type:
            raise ValueError("Module metadata mismatch")

        end_bytes = self._backend.snapshot()

        record = MemoryRecord(
            invocation_id=invocation_id,
            module_name=event.name,
            module_type=event.module_type,
            start_bytes=active.start_bytes,
            end_bytes=end_bytes,
            backend=self._backend.name
        )

        del self._active[invocation_id]
        self._records.append(record)

        return record

    def abort(self, invocation_id: int) -> None:
        if invocation_id not in self._active:
            raise ValueError(f"Invocation {invocation_id} was never started")

        del self._active[invocation_id]

    def get_records(self) -> list[MemoryRecord]:
        return self._records.copy()

    def reset(self) -> None:
        self._active.clear()
        self._records.clear()