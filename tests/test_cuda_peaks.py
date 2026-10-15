# test_cuda_peaks.py
import torch
import pytest
from torch import nn

from pygauge.hooks import HookManager
from pygauge.memory.engine import MemoryEngine
from pygauge.memory.peaks import CUDAPeakCollector
from pygauge.memory.backends.cuda import CUDAMemoryBackend


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable"
)


def test_stop_before_start_raises():
    collector = CUDAPeakCollector("cuda:0")

    with pytest.raises(
        RuntimeError,
        match="not started"
    ):
        collector.stop()


def test_start_twice_raises():
    collector = CUDAPeakCollector("cuda:0")

    collector.start()

    try:
        with pytest.raises(
            RuntimeError,
            match="already active"
        ):
            collector.start()
    finally:
        collector.stop()


def test_peak_collection_returns_valid_record():
    collector = CUDAPeakCollector("cuda:0")

    collector.start()

    tensor = torch.randn(
        (1024, 1024),
        device="cuda:0"
    )

    output = tensor * 2

    record = collector.stop()

    assert record.device == "cuda:0"
    assert record.peak_allocated_bytes >= 0
    assert record.peak_reserved_bytes >= 0

    assert (
        record.peak_reserved_bytes
        >= record.peak_allocated_bytes
    )

    # Keep allocations alive through collector.stop().
    assert output.is_cuda


def test_peak_collector_can_be_reused():
    collector = CUDAPeakCollector("cuda:0")

    collector.start()
    first = collector.stop()

    collector.start()
    second = collector.stop()

    assert first.peak_allocated_bytes >= 0
    assert second.peak_allocated_bytes >= 0


def test_peak_collection_observes_allocation():
    collector = CUDAPeakCollector("cuda:0")

    collector.start()

    baseline = torch.cuda.memory_allocated("cuda:0")

    tensor = torch.empty(
        (2048, 2048),
        device="cuda:0"
    )

    record = collector.stop()

    assert record.peak_allocated_bytes > baseline

    del tensor

def test_cuda_memory_engine_hook_integration():
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 64)
    ).cuda()

    backend = CUDAMemoryBackend("cuda:0")
    engine = MemoryEngine(backend)

    x = torch.randn(
        (32, 128),
        device="cuda:0"
    )

    with HookManager(
        on_start=engine.start,
        on_end=engine.stop
    ) as manager:
        manager.attach(model)
        output = model(x)

    records = engine.get_records()

    assert output.is_cuda
    assert len(records) == 3

    assert [r.module_name for r in records] == [
        "0", "1", "2"
    ]

    assert all(r.backend == "cuda" for r in records)
    assert all(r.start_bytes >= 0 for r in records)
    assert all(r.end_bytes >= 0 for r in records)
    assert engine.active_count == 0

