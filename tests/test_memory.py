# test_memory.py
import pytest
import torch
from torch import nn

from pygauge.events import ModuleEvent
from pygauge.hooks import HookManager
from pygauge.memory.engine import MemoryEngine


class FakeMemoryBackend:
    def __init__(self, snapshots):
        self._snapshots = iter(snapshots)

    @property
    def name(self):
        return "fake"

    def snapshot(self):
        return next(self._snapshots)


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
    engine = make_engine([])

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

