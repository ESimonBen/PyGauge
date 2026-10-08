# test_hooks.py
import pytest
import torch
from torch import nn

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
    assert len(manager._handles) == 6;

    manager.detach()

def test_hooks_execute(model, manager, capsys):
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

def test_reattach_does_not_duplicate_hooks(model, manager, capsys):
    manager.attach(model)
    manager.attach(model)

    x = torch.randn(2, 4)

    model(x)

    output = capsys.readouterr().out

    assert output.count("[START]") == 3
    assert output.count("[END]") == 3

    manager.detach()

def test_multiple_forward_passes(model, manager, capsys):
    manager.attach(model)

    x = torch.randn(2, 4)

    model(x)
    model(x)

    output = capsys.readouterr().out

    assert output.count("[START]") == 6
    assert output.count("[END]") == 6

    manager.detach()