# test_aggregation_integration.py
import torch
from torch import nn

from pygauge import ModuleEvent
from pygauge import HookManager

from pygauge.timing import TimingEngine
from pygauge.memory import MemoryEngine

from pygauge.aggregation import PerformanceAggregator


def test_end_to_end_aggregation():
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )

    timing_engine = TimingEngine()
    memory_engine = MemoryEngine()

    def on_start(event: ModuleEvent) -> None:
        timing_engine.start(event)
        memory_engine.start(event)

    def on_end(event: ModuleEvent) -> None:
        memory_engine.stop(event)
        timing_engine.stop(event)

    with HookManager(
        on_start=on_start,
        on_end=on_end,
    ) as manager:
        manager.attach(model)
        model(torch.randn(2, 4))

    timing_engine.finalize()

    aggregator = PerformanceAggregator()

    records = aggregator.aggregate(
        timing_engine.get_records(),
        memory_engine.get_records(),
    )

    summaries = aggregator.summarize(records)

    assert len(records) == 3
    assert len(summaries) == 3

    assert [r.module_name for r in records] == [
        "0", "1", "2"
    ]

    assert all(r.elapsed_ns >= 0 for r in records)

    assert all(
        r.memory_delta_bytes is not None
        for r in records
    )

    assert all(
        s.invocation_count == 1
        for s in summaries
    )
