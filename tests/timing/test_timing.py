# test_timing.py

import torch
import pytest
from torch import nn

from pygauge.enums import HookMode
from pygauge.hooks import HookManager
from pygauge.events import ModuleEvent
from pygauge.timing import TimingEngine
from pygauge.timing.backends import CPUTimingBackend


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

def make_engine(timestamps):
    clock = iter(timestamps)

    return TimingEngine(
        CPUTimingBackend(
            clock=lambda: next(clock)
        )
    )

def test_start_records_timestamp(event):
    engine = make_engine([1000])

    engine.start(event)

    active = engine._active[event.invocation_id]

    assert active.start_token == 1000
    assert active.event is event

def test_stop_calculates_elapsed_time(event):
    engine = make_engine([1000, 6000])

    engine.start(event)
    engine.stop(event)
    engine.finalize()

    record = engine.get_records()[0]

    assert record.elapsed_ns == 5000
    assert record.elapsed_ms == pytest.approx(0.005)
    assert record.backend == "cpu"

def test_stop_creates_record(event):
    engine = make_engine([100, 300])

    engine.start(event)
    engine.stop(event)

    assert engine.get_records() == []

    engine.finalize()
    records = engine.get_records()

    assert len(records) == 1

    record = records[0]

    assert record.invocation_id == 1
    assert record.module_name == "linear"
    assert record.module_type == "Linear"
    assert record.elapsed_ns == 200

def test_duplicate_start_raises(event):
    engine = make_engine([100])

    engine.start(event)

    with pytest.raises(ValueError, match="already active"):
        engine.start(event)


def test_stop_without_start_raises(event):
    engine = make_engine([100])

    with pytest.raises(ValueError, match="never started"):
        engine.stop(event)


def test_multiple_invocations(module):
    engine = make_engine([100, 200, 300, 400])

    event1 = ModuleEvent(
        module=module,
        name="linear",
        module_type="Linear",
        invocation_id=1
    )

    event2 = ModuleEvent(
        module=module,
        name="linear",
        module_type="Linear",
        invocation_id=2
    )

    engine.start(event1)
    engine.stop(event1)

    engine.start(event2)
    engine.stop(event2)

    engine.finalize()

    records = engine.get_records()

    assert len(records) == 2
    assert records[0].invocation_id == 1
    assert records[1].invocation_id == 2
    assert records[0].elapsed_ns == 100
    assert records[1].elapsed_ns == 100


def test_reset_clears_state(event):
    engine = make_engine([100, 200])

    engine.start(event)
    engine.stop(event)

    engine.reset()

    assert engine.get_records() == []
    assert engine._active == {}
    assert engine._pending == []

    engine.finalize()

    assert engine.get_records() == []

def test_timing_engine_hook_integration():
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2)
    )

    engine = TimingEngine()

    with HookManager(
        on_start=engine.start,
        on_end=engine.stop
    ) as manager:
        manager.attach(model)
        model(torch.randn(2, 4))

    engine.finalize()
    records = engine.get_records()

    assert len(records) == 3
    assert [r.module_name for r in records] == [
        "0", "1", "2"
    ]
    assert len({r.invocation_id for r in records}) == 3
    assert all(r.elapsed_ns >= 0 for r in records)

def test_all_mode_timing_integration():
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2)
    )

    engine = TimingEngine()

    with HookManager(
        mode=HookMode.ALL,
        on_start=engine.start,
        on_end=engine.stop
    ) as manager:
        manager.attach(model)
        model(torch.randn(2, 4))

    engine.finalize()
    records = engine.get_records()

    assert len(records) == 4
    assert {r.module_name for r in records} == {
        "", "0", "1", "2"
    }
    assert engine._active == {}

def test_shared_module_timing():
    shared = nn.Linear(4, 4)
    model = nn.Sequential(
        shared,
        nn.ReLU(),
        shared
    )

    engine = TimingEngine()

    with HookManager(
        on_start=engine.start,
        on_end=engine.stop
    ) as manager:
        manager.attach(model)
        model(torch.randn(2, 4))

    engine.finalize()
    records = engine.get_records()

    linear_records = [
        r for r in records
        if r.module_type == "Linear"
    ]

    assert len(linear_records) == 2
    assert (
        linear_records[0].invocation_id
        != linear_records[1].invocation_id
    )

def test_multiple_forward_passes():
    model = nn.Linear(4, 4)
    engine = TimingEngine()

    with HookManager(
        on_start=engine.start,
        on_end=engine.stop
    ) as manager:
        manager.attach(model)

        x = torch.randn(2, 4)

        model(x)
        model(x)

    engine.finalize()
    records = engine.get_records()

    assert len(records) == 2
    assert records[0].invocation_id != records[1].invocation_id

def test_stop_with_wrong_module(event):
    engine = make_engine([100, 200])
    engine.start(event)

    wrong_event = ModuleEvent(
        module=nn.ReLU(),
        name=event.name,
        module_type=event.module_type,
        invocation_id=event.invocation_id
    )

    with pytest.raises(ValueError, match="Module identity mismatch"):
        engine.stop(wrong_event)

def test_mismatch_preserves_active_timing(event):
    engine = make_engine([100, 200])
    engine.start(event)

    wrong_event = ModuleEvent(
        module=nn.ReLU(),
        name=event.name,
        module_type=event.module_type,
        invocation_id=event.invocation_id
    )

    with pytest.raises(ValueError):
        engine.stop(wrong_event)

    assert event.invocation_id in engine._active

    engine.stop(event)
    engine.finalize()

    assert engine.get_records()[0].elapsed_ns == 100


def test_abort_active_measurement(event):
    engine = make_engine([100])

    engine.start(event)
    engine.abort(event.invocation_id)

    assert engine._active == {}
    assert engine.get_records() == []


def test_abort_unknown_invocation():
    engine = make_engine([])

    with pytest.raises(ValueError, match="not active"):
        engine.abort(999)


def test_nested_invocations():
    parent = nn.Sequential(nn.Linear(4, 4))
    child = parent[0]

    parent_event = ModuleEvent(parent, "", "Sequential", 1)
    child_event = ModuleEvent(child, "0", "Linear", 2)

    engine = make_engine([100, 200, 500, 800])

    engine.start(parent_event)
    engine.start(child_event)
    engine.stop(child_event)
    engine.stop(parent_event)

    engine.finalize()

    records = {
        r.invocation_id: r
        for r in engine.get_records()
    }

    assert records[1].elapsed_ns == 700
    assert records[2].elapsed_ns == 300


def test_negative_elapsed_time(event):
    engine = make_engine([500, 100])

    engine.start(event)
    engine.stop(event)

    with pytest.raises(ValueError, match="Negative CPU elapsed time"):
        engine.finalize()

    assert engine.get_records() == []
    assert len(engine._pending) == 1


def test_get_records_returns_copy(event):
    engine = make_engine([100, 200])

    engine.start(event)
    engine.stop(event)
    engine.finalize()

    records = engine.get_records()
    records.clear()

    assert len(engine.get_records()) == 1


def test_reset_clears_all_state(event):
    engine = make_engine([100, 200])

    engine.start(event)
    engine.stop(event)

    assert len(engine._pending) == 1

    engine.reset()

    assert engine._active == {}
    assert engine._pending == []
    assert engine.get_records() == []


def test_finalize_is_idempotent(event):
    engine = make_engine([100, 200])

    engine.start(event)
    engine.stop(event)

    engine.finalize()
    engine.finalize()

    assert len(engine.get_records()) == 1


def test_duplicate_completed_invocation_rejected(event):
    engine = make_engine([100, 200])

    engine.start(event)
    engine.stop(event)

    with pytest.raises(ValueError, match="already recorded"):
        engine.start(event)

def test_forward_failure_does_not_leave_active_timing():
    class FailingLayer(nn.Module):
        def forward(self, x):
            raise RuntimeError("Forward failed")

    model = FailingLayer()
    engine = TimingEngine()

    with HookManager(
        on_start=engine.start,
        on_end=engine.stop
    ) as manager:
        manager.attach(model)

        with pytest.raises(RuntimeError, match="Forward failed"):
            model(torch.randn(2, 4))

    assert engine._active == {}

    engine.finalize()

    # The hook fired, but this record does not
    # indicate whether forward succeeded.
    assert len(engine.get_records()) == 1