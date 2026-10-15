# test_memory_integration.py
import pytest
import torch
from torch import nn

from pygauge.enums import HookMode
from pygauge.hooks import HookManager
from pygauge.events import ModuleEvent
from pygauge.memory.engine import MemoryEngine
from pygauge.memory.records import MemorySnapshot


class FakeMemoryBackend:
    def __init__(self, snapshots):
        self._snapshots = iter(snapshots)

    @property
    def name(self) -> str:
        return "fake"

    def snapshot(self) -> MemorySnapshot:
        value = next(self._snapshots)

        if isinstance(value, Exception):
            raise value

        if isinstance(value, MemorySnapshot):
            return value

        return MemorySnapshot(allocated_bytes=value)


def make_engine(snapshots):
    return MemoryEngine(FakeMemoryBackend(snapshots))


def test_shared_module_invocations():
    shared = nn.Linear(4, 4)

    model = nn.Sequential(
        shared,
        nn.ReLU(),
        shared
    )

    engine = MemoryEngine()

    with HookManager(
        on_start=engine.start,
        on_end=engine.stop
    ) as manager:
        manager.attach(model)
        model(torch.randn(2, 4))

    records = engine.get_records()

    linear_records = [
        record
        for record in records
        if record.module_type == "Linear"
    ]

    assert len(records) == 3
    assert len(linear_records) == 2

    assert (
        linear_records[0].invocation_id
        != linear_records[1].invocation_id
    )

    assert engine.active_count == 0


def test_nested_modules_all_mode():
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2)
    )

    engine = MemoryEngine()

    with HookManager(
        mode=HookMode.ALL,
        on_start=engine.start,
        on_end=engine.stop
    ) as manager:
        manager.attach(model)
        model(torch.randn(2, 4))

    records = engine.get_records()

    assert len(records) == 4

    assert {r.module_name for r in records} == {
        "", "0", "1", "2"
    }

    assert len({
        r.invocation_id for r in records
    }) == 4

    assert engine.active_count == 0

    # Parent/child deltas overlap in time.
    # Do not assert that their sum equals total memory usage.


def test_multiple_forward_passes():
    model = nn.Linear(4, 4)
    engine = MemoryEngine()

    with HookManager(
        on_start=engine.start,
        on_end=engine.stop
    ) as manager:
        manager.attach(model)

        x = torch.randn(2, 4)

        for _ in range(10):
            model(x)

    records = engine.get_records()

    assert len(records) == 10

    assert len({
        r.invocation_id for r in records
    }) == 10

    assert engine.active_count == 0


def test_forward_failure_does_not_leave_active_memory():
    class FailingLayer(nn.Module):
        def forward(self, x):
            raise RuntimeError("Forward failed")

    model = FailingLayer()
    engine = MemoryEngine()

    with HookManager(
        on_start=engine.start,
        on_end=engine.stop
    ) as manager:
        manager.attach(model)

        with pytest.raises(
            RuntimeError,
            match="Forward failed"
        ):
            model(torch.randn(2, 4))

    assert engine.active_count == 0

    # HookManager uses always_call=True.
    # A record may exist even when forward fails.
    records = engine.get_records()

    assert len(records) == 1


def test_stop_snapshot_failure_preserves_state():
    module = nn.Linear(4, 4)

    event = ModuleEvent(
        module=module,
        name="linear",
        module_type="Linear",
        invocation_id=1
    )

    engine = make_engine([
        1000,
        RuntimeError("Snapshot failed"),
        2000
    ])

    engine.start(event)

    with pytest.raises(
        RuntimeError,
        match="Snapshot failed"
    ):
        engine.stop(event)

    assert engine.active_count == 1
    assert engine.record_count == 0

    # Retry after a transient backend failure.
    record = engine.stop(event)

    assert record.delta_bytes == 1000
    assert engine.active_count == 0
    assert engine.record_count == 1


def test_start_snapshot_failure_does_not_register_id():
    module = nn.Linear(4, 4)

    event = ModuleEvent(
        module=module,
        name="linear",
        module_type="Linear",
        invocation_id=1
    )

    engine = make_engine([
        RuntimeError("Snapshot failed"),
        1000,
        2000
    ])

    with pytest.raises(
        RuntimeError,
        match="Snapshot failed"
    ):
        engine.start(event)

    assert engine.active_count == 0
    assert engine.record_count == 0

    # Same ID should be usable after failed start.
    engine.start(event)
    record = engine.stop(event)

    assert record.delta_bytes == 1000


def test_long_profiling_session():
    module = nn.Linear(4, 4)

    invocation_count = 5000

    snapshots = [
        1000
        for _ in range(invocation_count * 2)
    ]

    engine = make_engine(snapshots)

    for invocation_id in range(invocation_count):
        event = ModuleEvent(
            module=module,
            name="linear",
            module_type="Linear",
            invocation_id=invocation_id
        )

        engine.start(event)
        engine.stop(event)

    records = engine.get_records()

    assert len(records) == invocation_count
    assert engine.active_count == 0

    assert len({
        r.invocation_id for r in records
    }) == invocation_count

    assert all(
        r.delta_bytes == 0
        for r in records
    )
