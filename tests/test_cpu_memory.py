# test_cpu_memory.py
from pygauge.memory.backends.cpu import CPUMemoryBackend


def test_cpu_backend_snapshot():
    backend = CPUMemoryBackend()

    memory = backend.snapshot()

    assert isinstance(memory, int)
    assert memory > 0