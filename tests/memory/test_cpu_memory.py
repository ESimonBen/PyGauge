# test_cpu_memory.py
from psutil import Process
from types import SimpleNamespace

from pygauge.memory.backends import CPUMemoryBackend


class FakeProcess(Process):
    def memory_info(self):
        return SimpleNamespace(rss=123_456)


def test_cpu_backend_snapshot():
    backend = CPUMemoryBackend(process=FakeProcess())

    snapshot = backend.snapshot()

    assert snapshot.allocated_bytes == 123_456
    assert snapshot.reserved_bytes is None