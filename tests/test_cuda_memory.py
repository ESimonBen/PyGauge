# test_cuda_memory.py
import torch
import pytest

from pygauge.memory.backends.cuda import CUDAMemoryBackend


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable"
)
def test_cuda_backend_snapshot():
    backend = CUDAMemoryBackend()

    memory = backend.snapshot()

    assert isinstance(memory, int)
    assert memory >= 0
