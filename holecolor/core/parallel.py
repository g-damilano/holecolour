from __future__ import annotations

import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, is_dataclass, replace
from typing import Any, Callable, Iterable, Sequence, TypeVar

import cv2
from tqdm.auto import tqdm

from holecolor.config.schema import ParallelConfig

T = TypeVar('T')
R = TypeVar('R')


def _worker_initializer(opencv_threads: int) -> None:
    threads = int(opencv_threads)
    if threads > 0:
        try:
            cv2.setNumThreads(threads)
        except Exception:
            pass
        # best-effort BLAS/OpenMP caps to limit oversubscription in worker pools
        os.environ.setdefault('OMP_NUM_THREADS', str(threads))
        os.environ.setdefault('OPENBLAS_NUM_THREADS', str(threads))
        os.environ.setdefault('MKL_NUM_THREADS', str(threads))
        os.environ.setdefault('NUMEXPR_NUM_THREADS', str(threads))


def _coerce_cfg(cfg: ParallelConfig | dict[str, Any] | None) -> ParallelConfig:
    if cfg is None:
        return ParallelConfig()
    if isinstance(cfg, ParallelConfig):
        return cfg
    return ParallelConfig(**cfg)


def resolve_parallel_backend(cfg: ParallelConfig | dict[str, Any] | None, n_tasks: int) -> tuple[str, int]:
    cfg2 = _coerce_cfg(cfg)
    if not cfg2.enabled or cfg2.backend == 'none' or int(n_tasks) < max(2, int(cfg2.min_parallel_tasks)):
        return 'none', 1
    cpu = max(1, (os.cpu_count() or 1))
    max_workers = int(cfg2.max_workers) if int(cfg2.max_workers) > 0 else max(1, cpu - 1)
    max_workers = max(1, min(max_workers, int(n_tasks)))
    if max_workers <= 1:
        return 'none', 1
    if cfg2.backend == 'auto':
        return 'process', max_workers
    return str(cfg2.backend), max_workers


def prefer_thread_for_image_tasks(cfg: ParallelConfig | dict[str, Any] | None) -> ParallelConfig:
    """Avoid process-pool serialization for large frame arrays unless requested."""
    cfg2 = _coerce_cfg(cfg)
    if cfg2.backend == 'auto':
        return replace(cfg2, backend='thread')
    return cfg2


def iter_with_progress(iterable: Iterable[T], total: int | None = None, cfg: ParallelConfig | dict[str, Any] | None = None, desc: str | None = None) -> Iterable[T]:
    cfg2 = _coerce_cfg(cfg)
    disable = (not bool(cfg2.show_progress)) or (not sys.stderr.isatty())
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        leave=cfg2.progress_leave,
        disable=disable,
        mininterval=max(0.02, float(cfg2.progress_mininterval_s)),
    )


def parallel_map(
    fn: Callable[[T], R],
    items: Sequence[T],
    cfg: ParallelConfig | dict[str, Any] | None = None,
    desc: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[R]:
    cfg2 = _coerce_cfg(cfg)
    backend, max_workers = resolve_parallel_backend(cfg2, len(items))
    if backend == 'none':
        out: list[R] = []
        total = len(items)
        for i, item in enumerate(iter_with_progress(items, total=total, cfg=cfg2, desc=desc), start=1):
            out.append(fn(item))
            if progress_callback is not None:
                progress_callback(i, total)
        return out

    executor_cls = ProcessPoolExecutor if backend == 'process' else ThreadPoolExecutor
    init_kwargs: dict[str, Any] = {}
    if backend == 'process':
        init_kwargs = {
            'initializer': _worker_initializer,
            'initargs': (int(cfg2.opencv_threads_per_worker),),
            'mp_context': mp.get_context('spawn'),
        }
    elif backend == 'thread' and int(cfg2.opencv_threads_per_worker) > 0:
        _worker_initializer(int(cfg2.opencv_threads_per_worker))

    with executor_cls(max_workers=max_workers, **init_kwargs) as ex:
        total = len(items)
        futures = {ex.submit(fn, item): idx for idx, item in enumerate(items)}
        out: list[Any] = [None] * total
        for i, future in enumerate(iter_with_progress(as_completed(futures), total=total, cfg=cfg2, desc=desc), start=1):
            out[futures[future]] = future.result()
            if progress_callback is not None:
                progress_callback(i, total)
        return out


def cfg_to_jsonable(cfg: ParallelConfig | dict[str, Any] | None) -> dict[str, Any]:
    cfg2 = _coerce_cfg(cfg)
    if is_dataclass(cfg2):
        return asdict(cfg2)
    return dict(cfg2)
