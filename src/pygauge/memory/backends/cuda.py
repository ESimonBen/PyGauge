# cuda.py
import torch

from .base import MemoryBackend
from pygauge.memory.records import MemorySnapshot

class CUDAMemoryBackend(MemoryBackend):
    def __init__(self, device: str = "cuda:0", synchronize: bool = False):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")

        self._device = torch.device(device)

        if self._device.type != "cuda":
            raise ValueError("CUDA backend requires CUDA device")

        if self._device.index is None:
            raise ValueError("CUDA device must have an explicit index")

        torch.cuda.get_device_properties(self._device)

        self._synchronize = synchronize

    @property
    def name(self) -> str:
        return "cuda"

    @property
    def device(self) -> torch.device:
        return self._device

    def snapshot(self) -> MemorySnapshot:
        if self._synchronize:
            torch.cuda.synchronize(self._device)

        allocated = torch.cuda.memory_allocated(self._device)
        reserved = torch.cuda.memory_reserved(self._device)

        return MemorySnapshot(allocated_bytes=allocated, reserved_bytes=reserved)