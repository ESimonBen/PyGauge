# cuda.py
import torch
from dataclasses import dataclass

from .base import TimingBackend

@dataclass(frozen=True)
class CUDAEventPair:
    start: torch.cuda.Event
    end: torch.cuda.Event

class CUDATimingBackend(TimingBackend):
    def __init__(self, device: str = "cuda:0"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")

        self._device = torch.device(device)

        if self._device.type != "cuda":
            raise RuntimeError("CUDA backend requires CUDA device")

        # Validate the device index
        torch.cuda.get_device_properties(self._device)

    @property
    def name(self) -> str:
        return "cuda"

    def start(self) -> torch.cuda.Event:
        event = torch.cuda.Event(enable_timing=True)

        with torch.cuda.device(self._device):
            event.record()

        return event

    def stop(self, start_token: torch.cuda.Event) -> CUDAEventPair:
        end_token = torch.cuda.Event(enable_timing=True)

        with torch.cuda.device(self._device):
            end_token.record()

        return CUDAEventPair(start=start_token, end=end_token)

    def synchronize(self) -> None:
        torch.cuda.synchronize(self._device)

    def elapsed_ns(self, completed_token: CUDAEventPair) -> int:
        elapsed_ms = completed_token.start.elapsed_time(completed_token.end)

        if elapsed_ms < 0:
            raise ValueError("Negative CUDA elapsed time")

        return round(elapsed_ms * 1_000_000)

    def reset(self) -> None:
        pass