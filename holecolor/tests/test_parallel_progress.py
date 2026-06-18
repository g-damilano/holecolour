from __future__ import annotations

import time

from holecolor.config.schema import ParallelConfig
from holecolor.core.parallel import parallel_map


def _sleepy_identity(task: tuple[int, float]) -> int:
    value, delay_s = task
    time.sleep(float(delay_s))
    return int(value)


def test_parallel_map_reports_completed_futures_before_ordered_results_block() -> None:
    cfg = ParallelConfig(
        enabled=True,
        backend="thread",
        max_workers=3,
        min_parallel_tasks=1,
        show_progress=False,
    )
    events: list[tuple[int, int, float]] = []
    started = time.perf_counter()

    out = parallel_map(
        _sleepy_identity,
        [(0, 0.35), (1, 0.01), (2, 0.01)],
        cfg,
        progress_callback=lambda current, total: events.append((current, total, time.perf_counter() - started)),
    )

    assert out == [0, 1, 2]
    assert [event[0] for event in events] == [1, 2, 3]
    assert all(event[1] == 3 for event in events)
    assert events[0][2] < 0.20
