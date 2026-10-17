# memory_overhead.py
import time
import argparse
import statistics

import torch
from torch import nn

from pygauge.hooks import HookManager
from pygauge.memory import MemoryEngine
from pygauge.memory.backends import CPUMemoryBackend, CUDAMemoryBackend


def build_model(device: str) -> nn.Module:
    return nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 128)
    ).to(device).eval()


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def benchmark(
    model: nn.Module,
    x: torch.Tensor,
    device: str,
    iterations: int = 200,
    warmup: int = 30
) -> float:
    for _ in range(warmup):
        model(x)

    synchronize(device)

    samples = []

    for _ in range(iterations):
        synchronize(device)
        start_ns = time.perf_counter_ns()

        model(x)

        synchronize(device)
        elapsed_ns = time.perf_counter_ns() - start_ns

        samples.append(elapsed_ns)

    return statistics.median(samples) / 1_000_000


def run_benchmark(device: str) -> None:
    model = build_model(device)

    x = torch.randn(
        (64, 512),
        device=device
    )

    backend = (
        CUDAMemoryBackend(device)
        if device.startswith("cuda")
        else CPUMemoryBackend()
    )

    engine = MemoryEngine(backend)

    with torch.inference_mode():
        baseline = benchmark(model, x, device)

        with HookManager() as manager:
            manager.attach(model)
            hooks_only = benchmark(model, x, device)

        with HookManager(
            on_start=engine.start,
            on_end=engine.stop
        ) as manager:
            manager.attach(model)

            # Reset before each measured forward to
            # avoid accumulating records across samples.
            # The benchmark() helper below is not aware
            # of the engine, so benchmark a fixed session.
            memory_enabled = benchmark(
                model, x, device
            )

    def overhead(value, reference):
        return (
            (value - reference) / reference * 100
            if reference > 0
            else 0.0
        )

    print(f"\nDevice: {device}")
    print(f"Baseline:       {baseline:.4f} ms")
    print(f"Hooks only:     {hooks_only:.4f} ms")
    print(f"Memory enabled: {memory_enabled:.4f} ms")

    print(
        "Hook overhead: "
        f"{overhead(hooks_only, baseline):.2f}%"
    )

    print(
        "Memory overhead: "
        f"{overhead(memory_enabled, baseline):.2f}%"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda:0"],
        default="cpu"
    )

    args = parser.parse_args()

    if (
        args.device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError("CUDA is unavailable")

    run_benchmark(args.device)
