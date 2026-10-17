# __init__.py
from typing import Sequence
from collections import defaultdict
from statistics import mean, median, pstdev

from .enums import JoinPolicy
from .statistics import percentile
from pygauge.timing import TimingRecord
from pygauge.memory import MemoryRecord
from .records import PerformanceRecord, ModuleSummary

class PerformanceAggregator:
    def aggregate(self, timing_records: Sequence[TimingRecord], memory_records: Sequence[MemoryRecord] = (), policy: JoinPolicy = JoinPolicy.TIMING_LEFT) -> list[PerformanceRecord]:
        timing_by_id = self._index_records(timing_records, "time")
        memory_by_id = self._index_records(memory_records, "memory")

        timing_ids = set(timing_by_id)
        memory_ids = set(memory_by_id)

        unmatched_timing = timing_ids - memory_ids
        unmatched_memory = memory_ids - timing_ids

        if unmatched_memory:
            raise ValueError(f"Memory records without matching timing records: {sorted(unmatched_memory)}")

        if policy == JoinPolicy.STRICT and unmatched_timing:
            raise ValueError(f"Timing records without matching memory records: {sorted(unmatched_timing)}")

        results = []

        for timing in timing_records:
            memory = memory_by_id.get(timing.invocation_id)

            if memory is not None:
                if timing.module_name != memory.module_name or timing.module_type != memory.module_type:
                    raise ValueError(f"Module metadata mismatch for invocation {timing.invocation_id}")

            results.append(
                PerformanceRecord(
                    invocation_id=timing.invocation_id,
                    module_name=timing.module_name,
                    module_type=timing.module_type,
                    elapsed_ns=timing.elapsed_ns,
                    timing_backend=timing.backend,
                    memory_start=memory.start if memory is not None else None,
                    memory_end=memory.end if memory is not None else None,
                    memory_backend=memory.backend if memory is not None else None
                )
            )

        return results

    def summarize(self, records: Sequence[PerformanceRecord]) -> list[ModuleSummary]:
        groups = defaultdict(list)

        for record in records:
            key = (record.module_name, record.module_type, record.timing_backend, record.memory_backend)
            groups[key].append(record)

        summaries = []

        for (name, module_type, timing_backend, memory_backend), group in groups.items():
            times = [r.elapsed_ns for r in group]
            deltas = [r.memory_delta_bytes for r in group if r.memory_delta_bytes is not None]
            reserved_deltas = [r.reserved_delta_bytes for r in group if r.reserved_delta_bytes is not None]

            summaries.append(
                ModuleSummary(
                    module_name=name,
                    module_type=module_type,
                    timing_backend=timing_backend,
                    memory_backend=memory_backend,
                    invocation_count=len(group),
                    total_elapsed_ns=sum(times),
                    mean_elapsed_ns=mean(times),
                    median_elapsed_ns=median(times),
                    std_elapsed_ns=pstdev(times),
                    min_elapsed_ns=min(times),
                    max_elapsed_ns=max(times),
                    p95_elapsed_ns=percentile(times, 95),
                    memory_sample_count=len(deltas),
                    mean_memory_delta_bytes=(
                        mean(deltas) if deltas else None
                    ),
                    min_memory_delta_bytes=(
                        min(deltas) if deltas else None
                    ),
                    max_memory_delta_bytes=(
                        max(deltas) if deltas else None
                    ),
                    reserved_sample_count=len(reserved_deltas),
                    mean_reserved_delta_bytes=(
                        mean(reserved_deltas)
                        if reserved_deltas else None
                    )
                )
            )

        return summaries

    @staticmethod
    def _index_records(records, kind):
        indexed = {}

        for record in records:
            if record.invocation_id in indexed:
                raise ValueError(f"Duplicate {kind} invocation ID: {record.invocation_id}")

            indexed[record.invocation_id] = record

        return indexed