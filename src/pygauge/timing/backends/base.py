# base.py
from typing import Any, Protocol

class TimingBackend(Protocol):
    @property
    def name(self) -> str:
        ...

    def start(self) -> Any:
        """Begin a measurement and return its start position."""
        ...

    def stop(self, start_token: Any) -> Any:
        """Finish a measurement and return a completed token."""
        ...

    def synchronize(self) -> None:
        """Wait for pending measurements if necessary."""
        ...

    def elapsed_ns(self, completed_token: Any) -> int:
        """Resolve a completed measurement to nanoseconds."""
        ...

    def reset(self) -> None:
        """Clear backend-owned resources."""
        ...