# engine.py
from dataclasses import dataclass

from pygauge.events import ModuleEvent
from .records import MemoryRecord, MemorySnapshot
from .backends import MemoryBackend, CPUMemoryBackend

@dataclass
class ActiveMemory:
    event: ModuleEvent
    start: MemorySnapshot

    @property
    def start_bytes(self) -> int:
        return self.start.allocated_bytes

class MemoryEngine:
    def __init__(self, backend: MemoryBackend | None = None):
        self._backend = (backend if backend is not None else CPUMemoryBackend())
        self._active: dict[int, ActiveMemory] = {}
        self._records: list[MemoryRecord] = []
        self._used_ids: set[int] = set()

    def start(self, event: ModuleEvent) -> None:
        invocation_id = event.invocation_id

        if invocation_id in self._active:
            raise ValueError(f"Invocation {invocation_id} is already active")

        if invocation_id in self._used_ids:
            raise ValueError(f"Invocation {invocation_id} is was already recorded")

        start_snapshot = self._backend.snapshot()

        self._active[invocation_id] = ActiveMemory(event=event, start=start_snapshot)

        self._used_ids.add(invocation_id)

    def stop(self, event: ModuleEvent) -> MemoryRecord:
        invocation_id = event.invocation_id

        if invocation_id not in self._active:
            raise ValueError(f"Invocation {invocation_id} was never started")

        active = self._active[invocation_id]

        if active.event.module is not event.module:
            raise ValueError("Module error mismatch")

        if active.event.name != event.name or active.event.module_type != event.module_type:
            raise ValueError("Module metadata mismatch")

        # Preserve active state if the backend fails.
        end_snapshot = self._backend.snapshot()

        record = MemoryRecord(
            invocation_id=invocation_id,
            module_name=event.name,
            module_type=event.module_type,
            start=active.start,
            end=end_snapshot,
            backend=self._backend.name
        )

        self._records.append(record)
        del self._active[invocation_id]

        return record

    def abort(self, invocation_id: int) -> None:
        if invocation_id not in self._active:
            raise ValueError(f"Invocation {invocation_id} was never started")

        del self._active[invocation_id]

    def get_records(self) -> list[MemoryRecord]:
        return self._records.copy()

    def reset(self) -> None:
        if self._active:
            raise RuntimeError("Cannot reset while measurements are active")

        self._active.clear()
        self._records.clear()
        self._used_ids.clear()

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def record_count(self) -> int:
        return len(self._records)