# test_memory.py
import pytest
import torch
from torch import nn

from pygauge.hooks import HookManager
from pygauge.events import ModuleEvent
from pygauge.memory import MemoryEngine, MemorySnapshot

class FakeMemoryBackend:
    def __init__(self, snapshots):
        self._snapshots = iter(snapshots)

    @property
    def name(self):
        return "fake"

    def snapshot(self):
        value = next(self._snapshots)

        if isinstance(value, Exception):
            raise value

        if isinstance(value, MemorySnapshot):
            return value

        return MemorySnapshot(
            allocated_bytes=value
        )


@pytest.fixture
def module():
    return nn.Linear(4, 4)


@pytest.fixture
def event(module):
    return ModuleEvent(
        module=module,
        name="linear",
        module_type="Linear",
        invocation_id=1
    )


def make_engine(snapshots):
    return MemoryEngine(FakeMemoryBackend(snapshots))


def test_start_records_memory(event):
    engine = make_engine([1000])

    engine.start(event)

    active = engine._active[event.invocation_id]

    assert active.start_bytes == 1000
    assert active.event is event


def test_stop_calculates_positive_delta(event):
    engine = make_engine([1000, 5000])

    engine.start(event)
    record = engine.stop(event)

    assert record.start_bytes == 1000
    assert record.end_bytes == 5000
    assert record.delta_bytes == 4000
    assert record.backend == "fake"


def test_stop_calculates_negative_delta(event):
    engine = make_engine([5000, 1000])

    engine.start(event)
    record = engine.stop(event)

    assert record.delta_bytes == -4000


def test_zero_memory_delta(event):
    engine = make_engine([1000, 1000])

    engine.start(event)
    record = engine.stop(event)

    assert record.delta_bytes == 0


def test_duplicate_start_raises(event):
    engine = make_engine([1000])

    engine.start(event)

    with pytest.raises(ValueError, match="already active"):
        engine.start(event)


def test_stop_without_start_raises(event):
    engine = make_engine([100])

    with pytest.raises(ValueError, match="never started"):
        engine.stop(event)


def test_abort_active_measurement(event):
    engine = make_engine([1000])

    engine.start(event)
    engine.abort(event.invocation_id)

    assert engine._active == {}
    assert engine.get_records() == []


def test_reset_clears_state(event):
    engine = make_engine([1000, 2000])

    engine.start(event)
    engine.stop(event)

    engine.reset()

    assert engine._active == {}
    assert engine.get_records() == []


def test_get_records_returns_copy(event):
    engine = make_engine([1000, 2000])

    engine.start(event)
    engine.stop(event)

    records = engine.get_records()
    records.clear()

    assert len(engine.get_records()) == 1

def test_nested_invocations():
    parent = nn.Sequential(nn.Linear(4, 4))
    child = parent[0]

    parent_event = ModuleEvent(
        parent, "", "Sequential", 1
    )

    child_event = ModuleEvent(
        child, "0", "Linear", 2
    )

    engine = make_engine([
        1000,  # Parent starts
        1200,  # Child starts
        1500,  # Child ends
        1600   # Parent ends
    ])

    engine.start(parent_event)
    engine.start(child_event)

    engine.stop(child_event)
    engine.stop(parent_event)

    records = {
        r.invocation_id: r
        for r in engine.get_records()
    }

    assert records[1].delta_bytes == 600
    assert records[2].delta_bytes == 300


def test_memory_engine_hook_integration():
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2)
    )

    engine = MemoryEngine()

    with HookManager(on_start=engine.start, on_end=engine.stop) as manager:
        manager.attach(model)
        model(torch.randn(2, 4))

    records = engine.get_records()

    assert len(records) == 3
    assert [r.module_name for r in records] == [
        "0", "1", "2"
    ]
    assert len({r.invocation_id for r in records}) == 3
    assert all(r.start_bytes >= 0 for r in records)
    assert all(r.end_bytes >= 0 for r in records)

def test_reserved_memory_delta(event):
    engine = make_engine([
        MemorySnapshot(
            allocated_bytes=1000,
            reserved_bytes=2000
        ),
        MemorySnapshot(
            allocated_bytes=1500,
            reserved_bytes=4000
        )
    ])

    engine.start(event)
    record = engine.stop(event)

    assert record.delta_bytes == 500
    assert record.reserved_delta_bytes == 2000


def test_cpu_reserved_memory_is_unavailable(event):
    engine = make_engine([1000, 1500])

    engine.start(event)
    record = engine.stop(event)

    assert record.reserved_delta_bytes is None


@pytest.mark.parametrize(
    "allocated,reserved",
    [
        (-1, None),
        (100, -1),
    ]
)
def test_invalid_snapshot_rejected(allocated, reserved):
    with pytest.raises(ValueError):
        MemorySnapshot(allocated_bytes=allocated, reserved_bytes=reserved)


def test_start_snapshot_failure_preserves_state(event):
    engine = make_engine([
        RuntimeError("Snapshot failed")
    ])

    with pytest.raises(
        RuntimeError,
        match="Snapshot failed"
    ):
        engine.start(event)

    assert engine.active_count == 0
    assert engine.record_count == 0
    assert event.invocation_id not in engine._used_ids


def test_stop_snapshot_failure_preserves_active(event):
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

    # Retry the measurement.
    record = engine.stop(event)

    assert record.delta_bytes == 1000
    assert engine.active_count == 0
    assert engine.record_count == 1


def test_reset_rejected_while_active(event):
    engine = make_engine([1000])

    engine.start(event)

    with pytest.raises(RuntimeError, match="Cannot reset"):
        engine.reset()

    assert engine.active_count == 1


def test_duplicate_completed_invocation(event):
    engine = make_engine([1000, 2000])

    engine.start(event)
    engine.stop(event)

    with pytest.raises(
        ValueError,
        match="already recorded"
    ):
        engine.start(event)


def test_aborted_invocation_cannot_be_reused(event):
    engine = make_engine([1000])

    engine.start(event)
    engine.abort(event.invocation_id)

    with pytest.raises(
        ValueError,
        match="already recorded"
    ):
        engine.start(event)


def test_reset_allows_invocation_id_reuse(event):
    engine = make_engine([
        1000, 2000,
        3000, 4000
    ])

    engine.start(event)
    engine.stop(event)

    engine.reset()

    engine.start(event)
    record = engine.stop(event)

    assert record.delta_bytes == 1000
    assert engine.record_count == 1


def test_wrong_module_preserves_active(event):
    engine = make_engine([1000, 2000])

    engine.start(event)

    wrong_event = ModuleEvent(
        module=nn.ReLU(),
        name=event.name,
        module_type=event.module_type,
        invocation_id=event.invocation_id
    )

    with pytest.raises(ValueError, match="Module error mismatch"):
        engine.stop(wrong_event)

    assert engine.active_count == 1

    engine.stop(event)

    assert engine.record_count == 1
