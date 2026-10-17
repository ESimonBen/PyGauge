# test_cuda_timing.py

import pytest
import torch
from torch import nn

from pygauge import HookManager, HookMode
from pygauge.timing import TimingEngine, CUDATimingBackend


CUDA_AVAILABLE = torch.cuda.is_available()

pytestmark = pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="CUDA is unavailable"
)


def test_cuda_backend_initialization():
    backend = CUDATimingBackend()

    assert backend.name == "cuda"


def test_cuda_event_timing():
    backend = CUDATimingBackend()

    x = torch.randn(512, 512, device="cuda")
    y = torch.randn(512, 512, device="cuda")

    # Warm up the operation.
    for _ in range(5):
        torch.mm(x, y)

    torch.cuda.synchronize()

    start = backend.start()
    torch.mm(x, y)
    completed = backend.stop(start)

    backend.synchronize()

    elapsed_ns = backend.elapsed_ns(completed)

    assert elapsed_ns >= 0


def test_cuda_multiple_invocations():
    model = nn.Linear(64, 64).cuda()
    engine = TimingEngine(CUDATimingBackend())

    with HookManager(
        on_start=engine.start,
        on_end=engine.stop
    ) as manager:
        manager.attach(model)

        x = torch.randn(32, 64, device="cuda")

        model(x)
        model(x)

    engine.finalize()
    records = engine.get_records()

    assert len(records) == 2
    assert all(r.backend == "cuda" for r in records)
    assert all(r.elapsed_ns >= 0 for r in records)


def test_cuda_nested_events():
    model = nn.Sequential(
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 64)
    ).cuda()

    engine = TimingEngine(CUDATimingBackend())

    with HookManager(
        mode=HookMode.ALL,
        on_start=engine.start,
        on_end=engine.stop
    ) as manager:
        manager.attach(model)
        model(torch.randn(32, 64, device="cuda"))

    engine.finalize()
    records = engine.get_records()

    assert len(records) == 4
    assert all(r.elapsed_ns >= 0 for r in records)


def test_cuda_reset():
    model = nn.Linear(16, 16).cuda()
    engine = TimingEngine(CUDATimingBackend())

    with HookManager(
        on_start=engine.start,
        on_end=engine.stop
    ) as manager:
        manager.attach(model)
        model(torch.randn(8, 16, device="cuda"))

    engine.finalize()
    engine.reset()

    assert engine.get_records() == []
    assert engine._active == {}
    assert engine._pending == []
