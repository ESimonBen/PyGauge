# test_hooks.py
import pytest
import torch
from torch import nn

from pygauge.enums import HookMode
from pygauge.hooks import HookManager

@pytest.fixture
def model():
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2)
    )

@pytest.fixture
def manager():
    return HookManager()

def test_attach_registers_hooks(model, manager):
    manager.attach(model)

    # Three leaf modules, two hooks per module
    assert len(manager._handles) == 6
    assert manager.is_attached() is True

    manager.detach()
    assert manager.is_attached() is False

def test_callbacks_receive_events(model):
    events = []
    manager = HookManager(
        on_start = lambda e: events.append(("START", e)),
        on_end = lambda e: events.append(("END", e))
    )

    manager.attach(model)
    model(torch.randn(2, 4))
    manager.detach()

    assert len(events) == 6
    assert events[0][0] == "START"
    assert events[0][1].name == "0"
    assert events[0][1].module_type == "Linear"
    assert events[1][0] == "END"

def test_unique_invocation_ids():
    events = []
    manager = HookManager(on_start=lambda e: events.append(e))
    model = nn.Linear(4, 4)
    manager.attach(model)

    x = torch.randn(2, 4)
    model(x)
    model(x)
    manager.detach()

    assert len(events) == 2
    assert events[0].invocation_id != events[1].invocation_id
    assert events[0].name == events[1].name

def test_leaf_mode(model, manager):
    manager.mode = HookMode.LEAF
    manager.attach(model)
    # Sequential has 3 leaf children (Linear, ReLU, Linear), container itself skipped
    assert len(manager._handles) == 6
    manager.detach()

def test_all_mode(model):
    manager = HookManager(mode=HookMode.ALL)
    manager.attach(model)

    assert len(manager._handles) == 8
    manager.detach()

def test_filtered_mode():
    model = nn.Sequential(
        nn.Linear(4, 4),
        nn.ReLU(),
        nn.Linear(4, 4)
    )

    manager = HookManager(
        mode=HookMode.FILTERED,
        module_filter=lambda name, mod: isinstance(mod, nn.ReLU)
    )

    manager.attach(model)
    assert len(manager._handles) == 2  # Only ReLU matched
    manager.detach()

def test_shared_module():
    events = []
    shared = nn.Linear(4, 4)
    model = nn.Sequential(shared, nn.ReLU(), shared)
    manager = HookManager(on_start=lambda e: events.append(e))
    manager.attach(model)
    model(torch.randn(2, 4))
    manager.detach()

    linear_events = [e for e in events if e.module is shared]
    assert len(linear_events) == 2
    assert linear_events[0].invocation_id != linear_events[1].invocation_id

def test_detach_idempotent(model, manager):
    manager.attach(model)
    manager.detach()
    manager.detach()  # Should not raise errors
    assert len(manager._handles) == 0

def test_context_manager_cleanup(model):
    with HookManager() as manager:
        manager.attach(model)
        assert manager.is_attached() is True
    assert manager.is_attached() is False

class FailingLayer(nn.Module):
    def forward(self, x):
        raise RuntimeError("Forward failed")

def test_forward_exception():
    starts = []
    ends = []
    model = FailingLayer()
    manager = HookManager(
        on_start=starts.append,
        on_end=ends.append
    )
    manager.attach(model)

    with pytest.raises(RuntimeError, match="Forward failed"):
        model(torch.randn(2, 4))

    manager.detach()
    assert len(starts) == 1
    assert len(ends) == 1
    assert starts[0].invocation_id == ends[0].invocation_id

def test_hooks_execute(model, capsys):
    manager = HookManager(
        on_start=lambda e: print(f"[START] {e.name}: {e.module_type}"),
        on_end=lambda e: print(f"[END] {e.name}: {e.module_type}")
    )

    manager.attach(model)

    x = torch.randn(2, 4)

    model(x)

    output = capsys.readouterr().out

    assert "[START] 0: Linear" in output
    assert "[END] 0: Linear" in output
    assert "[START] 1: ReLU" in output
    assert "[END] 1: ReLU" in output
    assert "[START] 2: Linear" in output
    assert "[END] 2: Linear" in output

    manager.detach()

def test_detach_removes_hooks(model, manager, capsys):
    manager.attach(model)
    manager.detach()

    assert len(manager._handles) == 0

    x = torch.randn(2, 4)

    model(x)

    output = capsys.readouterr().out

    assert output == ""

    manager.detach()

def test_reattach_does_not_duplicate_hooks(model, capsys):
    manager = HookManager(
        on_start=lambda e: print(f"[START] {e.name}: {e.module_type}"),
        on_end=lambda e: print(f"[END] {e.name}: {e.module_type}")
    )

    manager.attach(model)
    manager.attach(model)

    x = torch.randn(2, 4)

    model(x)

    output = capsys.readouterr().out

    assert output.count("[START]") == 3
    assert output.count("[END]") == 3

    manager.detach()

def test_multiple_forward_passes(model, capsys):
    manager = HookManager(
        on_start=lambda e: print(f"[START] {e.name}: {e.module_type}"),
        on_end=lambda e: print(f"[END] {e.name}: {e.module_type}")
    )

    manager.attach(model)

    x = torch.randn(2, 4)

    model(x)
    model(x)

    output = capsys.readouterr().out

    assert output.count("[START]") == 6
    assert output.count("[END]") == 6

    manager.detach()

def test_event_order(model):
    events = []

    manager = HookManager(
        on_start=lambda e: events.append(("START", e)),
        on_end=lambda e: events.append(("END", e))
    )

    manager.attach(model)
    model(torch.randn(2, 4))

    assert len(events) == 6

    for i in range(0, len(events), 2):
        start_type, start_event = events[i]
        end_type, end_event = events[i + 1]

        assert start_type == "START"
        assert end_type == "END"

        assert start_event.invocation_id == end_event.invocation_id
        assert start_event.module is end_event.module

def test_attach_failure_cleanup(model, manager, monkeypatch):
    failing_module = model[1]

    def failing_registration(*args, **kwargs):
        raise RuntimeError("Registration failed")

    monkeypatch.setattr(
        failing_module,
        "register_forward_hook",
        failing_registration
    )

    with pytest.raises(RuntimeError, match="Registration failed"):
        manager.attach(model)

    assert manager.is_attached() is False
    assert len(manager._handles) == 0
    assert len(manager._module_names) == 0
    assert len(manager._active_stacks) == 0

def test_callback_exception():
    starts = []
    ends = []

    def failing_start(event):
        starts.append(event)
        raise RuntimeError("Start callback failed")

    model = nn.Linear(4, 4)

    manager = HookManager(
        on_start=failing_start,
        on_end=ends.append
    )

    with manager:
        manager.attach(model)

        with pytest.raises(
            RuntimeError,
            match="Start callback failed"
        ):
            model(torch.randn(2, 4))

        assert manager._active_stacks[model] == []

    assert len(starts) == 1
    assert len(ends) == 0

def test_filtered_mode_requires_filter():
    with pytest.raises(ValueError, match="FILTERED mode requires a module_filter"):
        HookManager(mode=HookMode.FILTERED)

def test_all_mode_nested_event_order(model):
    events = []

    manager = HookManager(
        mode=HookMode.ALL,
        on_start=lambda e: events.append(("START", e.name)),
        on_end=lambda e: events.append(("END", e.name))
    )

    with manager:
        manager.attach(model)
        model(torch.randn(2, 4))

    assert events == [
        ("START", ""),
        ("START", "0"),
        ("END", "0"),
        ("START", "1"),
        ("END", "1"),
        ("START", "2"),
        ("END", "2"),
        ("END", "")
    ]
