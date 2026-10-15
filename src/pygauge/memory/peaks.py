# peaks.py
import torch
from dataclasses import dataclass

@dataclass(frozen=True)
class CUDAPeakRecord:
    device: str
    peak_allocated_bytes: int
    peak_reserved_bytes: int

class CUDAPeakCollector:
    def __init__(self, device: str = "cuda:0"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")

        self._device = torch.device(device)

        if self._device.type != "cuda" or self._device.index is None:
            raise ValueError("An explicit CUDA device required")

        torch.cuda.get_device_properties(self._device)

        self._active = False

    def start(self) -> None:
        if self._active:
            raise RuntimeError("Peak collection is already active")

        torch.cuda.synchronize(self._device)

        torch.cuda.reset_peak_memory_stats(self._device)

        self._active = True

    def stop(self) -> CUDAPeakRecord:
        if not self._active:
            raise RuntimeError("Peak collection was not started")

        torch.cuda.synchronize(self._device)

        record = CUDAPeakRecord(
            device=str(self._device),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(self._device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(self._device)
        )

        self._active = False

        return record

