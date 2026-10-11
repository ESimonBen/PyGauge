# engine.py
from typing import Any
from dataclasses import dataclass

from pygauge.events import ModuleEvent
from pygauge.timing.records import TimingRecord
from pygauge.timing.backends.base import TimingBackend
from pygauge.timing.backends.cpu import CPUTimingBackend

@dataclass
class ActiveTiming:
    event: ModuleEvent
    start_token: Any

@dataclass
class PendingTiming:
    event: ModuleEvent
    completed_token: Any

class TimingEngine:
    def __init__(self, backend: TimingBackend | None = None):
        self._backend = backend or CPUTimingBackend()

        self._active: dict[int, ActiveTiming] = {}
        self._pending: list[PendingTiming] = []
        self._records: list[TimingRecord] = []

    def start(self, event: ModuleEvent) -> None:
        invocation_id = event.invocation_id

        if invocation_id in self._active:
            raise ValueError(f"Invocation {invocation_id} is already active")

        if any(p.event.invocation_id == invocation_id for p in self._pending) or any(r.invocation_id == invocation_id for r in self._records):
            raise ValueError(f"Invocation {invocation_id} was already recorded")

        token = self._backend.start()

        self._active[invocation_id] = ActiveTiming(
            event=event,
            start_token=token
        )

    def stop(self, event: ModuleEvent) -> None:
        invocation_id = event.invocation_id

        if invocation_id not in self._active:
            raise ValueError(f"Invocation {invocation_id} was never started")

        active = self._active[invocation_id]

        if active.event.module is not event.module:
            raise ValueError("Module identity mismatch")

        if active.event.name != event.name or active.event.module_type != event.module_type:
            raise ValueError("Module metadata mismatch")

        completed_token = self._backend.stop(active.start_token)

        del self._active[invocation_id]

        self._pending.append(
            PendingTiming(
                event=event,
                completed_token=completed_token
            )
        )

    def abort(self, invocation_id: int) -> None:
        if invocation_id not in self._active:
            raise ValueError(f"Invocation {invocation_id} is not active")

        del self._active[invocation_id]

    def finalize(self) -> None:
        if not self._pending:
            return

        self._backend.synchronize()

        # Resolve before commiting records. If resolution fails, pending measurements remain available.
        resolved = []

        for pending in self._pending:
            elapsed_ns = self._backend.elapsed_ns(pending.completed_token)

            record = TimingRecord(
                invocation_id=pending.event.invocation_id,
                module_name=pending.event.name,
                module_type=pending.event.module_type,
                elapsed_ns=elapsed_ns,
                backend=self._backend.name
            )

            resolved.append(record)

        self._records.extend(resolved)
        self._pending.clear()

    def get_records(self) -> list[TimingRecord]:
        return self._records.copy()

    def reset(self) -> None:
        self._active.clear()
        self._pending.clear()
        self._records.clear()
        self._backend.reset()