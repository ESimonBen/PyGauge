# test_aggregation
import pytest

from pygauge.aggregation import (
    PerformanceAggregator,
    JoinPolicy,
    SummarySort,
    percentile,
    sort_summaries,
)
from pygauge.timing import TimingRecord
from pygauge.memory import MemoryRecord, MemorySnapshot


# ============================================================
# Fixtures and helpers
# ============================================================

@pytest.fixture
def aggregator():
    return PerformanceAggregator()


def make_timing(
    invocation_id,
    name="linear",
    module_type="Linear",
    elapsed_ns=1000,
    backend="cpu",
):
    return TimingRecord(
        invocation_id=invocation_id,
        module_name=name,
        module_type=module_type,
        elapsed_ns=elapsed_ns,
        backend=backend,
    )


def make_memory(
    invocation_id,
    name="linear",
    module_type="Linear",
    start=1000,
    end=2000,
    reserved_start=None,
    reserved_end=None,
    backend="cpu",
):
    return MemoryRecord(
        invocation_id=invocation_id,
        module_name=name,
        module_type=module_type,
        start=MemorySnapshot(
            allocated_bytes=start,
            reserved_bytes=reserved_start,
        ),
        end=MemorySnapshot(
            allocated_bytes=end,
            reserved_bytes=reserved_end,
        ),
        backend=backend,
    )


# ============================================================
# Record aggregation and joining
# ============================================================

class TestRecordAggregation:

    def test_matching_records(self, aggregator):
        records = aggregator.aggregate(
            [make_timing(1, elapsed_ns=5000)],
            [make_memory(1, start=1000, end=3000)],
        )

        assert len(records) == 1

        record = records[0]

        assert record.invocation_id == 1
        assert record.elapsed_ns == 5000
        assert record.memory_delta_bytes == 2000
        assert record.timing_backend == "cpu"
        assert record.memory_backend == "cpu"

    def test_without_memory(self, aggregator):
        records = aggregator.aggregate([make_timing(1)])

        assert len(records) == 1
        assert records[0].memory_delta_bytes is None
        assert records[0].memory_backend is None

    def test_preserves_timing_order(self, aggregator):
        timing = [
            make_timing(3),
            make_timing(1),
            make_timing(2),
        ]

        memory = [
            make_memory(2),
            make_memory(3),
            make_memory(1),
        ]

        records = aggregator.aggregate(timing, memory)

        assert [r.invocation_id for r in records] == [3, 1, 2]

    def test_negative_memory_delta(self, aggregator):
        records = aggregator.aggregate(
            [make_timing(1)],
            [make_memory(1, start=5000, end=1000)],
        )

        assert records[0].memory_delta_bytes == -4000

    def test_cuda_reserved_memory_preserved(self, aggregator):
        records = aggregator.aggregate(
            [make_timing(1, backend="cuda")],
            [
                make_memory(
                    1,
                    start=1000,
                    end=1500,
                    reserved_start=2000,
                    reserved_end=4000,
                    backend="cuda",
                )
            ],
        )

        record = records[0]

        assert record.memory_delta_bytes == 500
        assert record.reserved_delta_bytes == 2000


# ============================================================
# Join policies and validation
# ============================================================

class TestJoinPolicies:

    def test_strict_join_requires_memory(self, aggregator):
        with pytest.raises(
            ValueError,
            match="Timing records without matching memory",
        ):
            aggregator.aggregate(
                [make_timing(1)],
                [],
                policy=JoinPolicy.STRICT,
            )

    def test_left_join_allows_missing_memory(self, aggregator):
        records = aggregator.aggregate(
            [make_timing(1)],
            [],
            policy=JoinPolicy.TIMING_LEFT,
        )

        assert len(records) == 1
        assert records[0].memory_delta_bytes is None

    def test_unmatched_memory_rejected(self, aggregator):
        with pytest.raises(
            ValueError,
            match="Memory records without matching timing",
        ):
            aggregator.aggregate(
                [make_timing(1)],
                [make_memory(2)],
            )

    def test_duplicate_timing_id_raises(self, aggregator):
        with pytest.raises(
            ValueError,
            match="Duplicate time invocation ID: 1",
        ):
            aggregator.aggregate([
                make_timing(1),
                make_timing(1),
            ])

    def test_duplicate_memory_id_raises(self, aggregator):
        with pytest.raises(
            ValueError,
            match="Duplicate memory invocation ID: 1",
        ):
            aggregator.aggregate(
                [make_timing(1)],
                [
                    make_memory(1),
                    make_memory(1),
                ],
            )

    def test_module_metadata_mismatch_raises(self, aggregator):
        with pytest.raises(
            ValueError,
            match="Module metadata mismatch",
        ):
            aggregator.aggregate(
                [
                    make_timing(
                        1,
                        name="linear",
                        module_type="Linear",
                    )
                ],
                [
                    make_memory(
                        1,
                        name="relu",
                        module_type="ReLU",
                    )
                ],
            )


# ============================================================
# Module summaries
# ============================================================

class TestModuleSummaries:

    def test_repeated_invocations(self, aggregator):
        timing = [
            make_timing(1, elapsed_ns=1000),
            make_timing(2, elapsed_ns=3000),
            make_timing(3, elapsed_ns=5000),
        ]

        memory = [
            make_memory(1, start=1000, end=2000),
            make_memory(2, start=2000, end=4000),
            make_memory(3, start=4000, end=3000),
        ]

        records = aggregator.aggregate(timing, memory)
        summaries = aggregator.summarize(records)

        assert len(summaries) == 1

        summary = summaries[0]

        assert summary.module_name == "linear"
        assert summary.invocation_count == 3

        assert summary.total_elapsed_ns == 9000
        assert summary.mean_elapsed_ns == 3000
        assert summary.min_elapsed_ns == 1000
        assert summary.max_elapsed_ns == 5000

        assert summary.memory_sample_count == 3
        assert summary.mean_memory_delta_bytes == pytest.approx(
            2000 / 3
        )
        assert summary.min_memory_delta_bytes == -1000
        assert summary.max_memory_delta_bytes == 2000

    def test_multiple_modules(self, aggregator):
        records = aggregator.aggregate([
            make_timing(1, name="linear", module_type="Linear"),
            make_timing(2, name="relu", module_type="ReLU"),
            make_timing(3, name="linear", module_type="Linear"),
        ])

        summaries = aggregator.summarize(records)

        assert len(summaries) == 2

        by_name = {
            summary.module_name: summary
            for summary in summaries
        }

        assert by_name["linear"].invocation_count == 2
        assert by_name["relu"].invocation_count == 1

    def test_without_memory(self, aggregator):
        records = aggregator.aggregate([make_timing(1)])
        summary = aggregator.summarize(records)[0]

        assert summary.memory_sample_count == 0
        assert summary.mean_memory_delta_bytes is None
        assert summary.min_memory_delta_bytes is None
        assert summary.max_memory_delta_bytes is None

    def test_empty_records(self, aggregator):
        assert aggregator.summarize([]) == []
        assert aggregator.aggregate([]) == []

    def test_advanced_statistics(self, aggregator):
        records = aggregator.aggregate([
            make_timing(1, elapsed_ns=1000),
            make_timing(2, elapsed_ns=2000),
            make_timing(3, elapsed_ns=3000),
            make_timing(4, elapsed_ns=4000),
        ])

        summary = aggregator.summarize(records)[0]

        assert summary.total_elapsed_ns == 10000
        assert summary.mean_elapsed_ns == 2500
        assert summary.median_elapsed_ns == 2500
        assert summary.min_elapsed_ns == 1000
        assert summary.max_elapsed_ns == 4000
        assert summary.p95_elapsed_ns == pytest.approx(3850)
        assert summary.std_elapsed_ns == pytest.approx(
            1118.034, rel=1e-5
        )

    def test_backend_groups_remain_separate(self, aggregator):
        records = aggregator.aggregate([
            make_timing(1, backend="cpu"),
            make_timing(2, backend="cuda"),
        ])

        summaries = aggregator.summarize(records)

        assert len(summaries) == 2
        assert {s.timing_backend for s in summaries} == {
            "cpu", "cuda"
        }


# ============================================================
# Statistical utilities
# ============================================================

class TestStatistics:

    def test_percentile_interpolation(self):
        assert percentile([10, 20, 30, 40], 50) == 25
        assert percentile([10, 20, 30, 40], 95) == pytest.approx(
            38.5
        )

    def test_percentile_single_sample(self):
        assert percentile([42], 95) == 42

    def test_percentile_empty_raises(self):
        with pytest.raises(ValueError):
            percentile([], 95)


# ============================================================
# Sorting
# ============================================================

class TestSorting:

    def test_sort_by_total_time(self, aggregator):
        records = aggregator.aggregate([
            make_timing(
                1,
                name="slow",
                module_type="Linear",
                elapsed_ns=5000,
            ),
            make_timing(
                2,
                name="fast",
                module_type="ReLU",
                elapsed_ns=1000,
            ),
        ])

        summaries = aggregator.summarize(records)

        sorted_results = sort_summaries(
            summaries,
            SummarySort.TOTAL_TIME,
        )

        assert [s.module_name for s in sorted_results] == [
            "slow", "fast"
        ]
