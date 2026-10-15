# cuda.py
import torch

from .base import MemoryBackend

class CUDAMemoryBackend(MemoryBackend):
    def __init__(self, device: str = "cuda:0"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")

        self._device = torch.device(device)

        if self._device.type != "cuda":
            raise ValueError("CUDA backend requires CUDA device")

        torch.cuda.get_device_properties(self._device)

    @property
    def name(self) -> str:
        return "cuda"

    def snapshot(self) -> int:
        return torch.cuda.memory_allocated(self._device)