# test_cuda_memory.py
import gc
import torch
import pytest

from pygauge.memory.backends.cuda import CUDAMemoryBackend


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable"
)


def test_cuda_backend_initialization():
    backend = CUDAMemoryBackend("cuda:0")

    assert backend.name == "cuda"
    assert backend.device == torch.device("cuda:0")


def test_cuda_backend_rejects_cpu_device():
    with pytest.raises(
        ValueError,
        match="CUDA backend requires"
    ):
        CUDAMemoryBackend("cpu")


def test_cuda_backend_snapshot():
    backend = CUDAMemoryBackend("cuda:0")

    snapshot = backend.snapshot()

    assert isinstance(snapshot.allocated_bytes, int)
    assert isinstance(snapshot.reserved_bytes, int)

    assert snapshot.allocated_bytes >= 0
    assert snapshot.reserved_bytes >= 0

    assert (
        snapshot.reserved_bytes
        >= snapshot.allocated_bytes
    )


def test_cuda_tensor_allocation_increases_memory():
    backend = CUDAMemoryBackend("cuda:0")

    before = backend.snapshot()

    tensor = torch.empty(
        (2048, 2048),
        device="cuda:0",
        dtype=torch.float32
    )

    # Keep tensor alive during measurement.
    after = backend.snapshot()

    assert (
        after.allocated_bytes
        > before.allocated_bytes
    )

    assert (
        after.reserved_bytes
        >= after.allocated_bytes
    )

    del tensor
    gc.collect()

    torch.cuda.synchronize("cuda:0")


def test_cuda_backend_with_synchronization():
    backend = CUDAMemoryBackend(
        "cuda:0",
        synchronize=True
    )

    snapshot = backend.snapshot()

    assert snapshot.allocated_bytes >= 0
    assert snapshot.reserved_bytes is not None