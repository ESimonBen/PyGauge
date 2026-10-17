# sorting.py
from enum import Enum, auto
from collections.abc import Sequence

from .records import ModuleSummary

class SummarySort(Enum):
    TOTAL_TIME = auto()
    MEAN_TIME = auto()
    INVOCATIONS = auto()
    MEMORY_DELTA = auto()
    NAME = auto()

def sort_summaries(summaries: Sequence[ModuleSummary], sort_by: SummarySort = SummarySort.TOTAL_TIME, descending: bool =True) -> list[ModuleSummary]:
    def metric(summary: ModuleSummary):
        if sort_by == SummarySort.TOTAL_TIME:
            return summary.total_elapsed_ns

        if sort_by == SummarySort.MEAN_TIME:
            return summary.mean_elapsed_ns

        if sort_by == SummarySort.INVOCATIONS:
            return summary.invocation_count

        if sort_by == SummarySort.MEMORY_DELTA:
            return summary.mean_memory_delta_bytes if summary.mean_memory_delta_bytes is not None else float("-inf")

        if sort_by == SummarySort.NAME:
            return summary.module_type

        raise ValueError(f"Unsupported sort key: {sort_by}")

    return sorted(summaries, key=metric, reverse=descending)