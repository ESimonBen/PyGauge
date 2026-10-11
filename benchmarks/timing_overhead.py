# timing_overhead.py
import time
import torch
import statistics

from torch import nn

from pygauge import HookManager
from pygauge.timing import TimingEngine

def benchmark(fn, iterations=100, warmup=20):
    for _ in range(warmup):
        fn()

    durations = []

    for _ in range(iterations):
        start_ns = time.perf_counter_ns()
        fn()
        durations.append(time.perf_counter_ns() - start_ns)

    return statistics.median(durations) / 1_000_000

def main():
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 128)
    ).eval()

    x = torch.randn(32, 128)

    # Avoid gradient tracking
    with torch.inference_mode():
        baseline = benchmark(lambda: model(x))

        with HookManager() as manager:
            manager.attach(model)
            hooks_only = benchmark(lambda: model(x))

        engine = TimingEngine()

        with HookManager(on_start=engine.start, on_end=engine.stop) as manager:
            manager.attach(model)

            def profiled():
                engine.reset()
                model(x)
                engine.finalize()

            profiled_ms = benchmark(profiled)

    print(f"Baseline:     {baseline:.4f} ms")
    print(f"Hooks only:   {hooks_only:.4f} ms")
    print(f"With timing:  {profiled_ms:.4f} ms")

if __name__ == "__main__":
    main()