from __future__ import annotations

import json
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from tqdm.auto import tqdm


def _now_local_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _fmt_seconds(value: float | None) -> str:
    if value is None or not isinstance(value, (int, float)):
        return '--:--'
    secs = max(0, int(round(float(value))))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f'{h:d}:{m:02d}:{s:02d}'
    return f'{m:02d}:{s:02d}'


def _console(msg: str) -> None:
    ts = _now_local_iso()
    tqdm.write(f'[holecolor {ts}] {msg}')


def format_status_line(payload: dict[str, Any]) -> str:
    event = str(payload.get('event') or 'status')
    stage = payload.get('stage') or 'idle'
    current = payload.get('current')
    total = payload.get('total')
    frac = payload.get('overall_fraction')
    frac_txt = '--.-%'
    if isinstance(frac, (int, float)):
        frac_txt = f'{100.0 * float(frac):5.1f}%'
    prog = ''
    if current is not None and total is not None:
        prog = f' | step {current}/{total}'
    elif current is not None:
        prog = f' | step {current}'
    message = payload.get('message')
    msg = f' | {message}' if message else ''
    return (
        f'[{event}] stage={stage} | overall={frac_txt}{prog} | '
        f'elapsed={payload.get("elapsed_hms", "--:--")} | '
        f'eta={payload.get("eta_hms", "--:--")} | '
        f'overall_eta={payload.get("overall_eta_hms", "--:--")}{msg}'
    )


class RunStatusTracker:
    def __init__(
        self,
        out_dir: Path,
        enabled: bool = True,
        heartbeat_interval_s: float = 1.0,
        progress_mininterval_s: float = 0.2,
    ) -> None:
        self.logs_dir = Path(out_dir) / 'logs'
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.logs_dir / 'run_status.jsonl'
        self.current_path = self.logs_dir / 'current_status.json'
        self.timings_path = self.logs_dir / 'stage_timings.json'
        self.summary_path = self.logs_dir / 'progress_summary.txt'
        self.stage_plan_path = self.logs_dir / 'stage_plan.json'
        self.stage: str | None = None
        self.stage_started_at: float | None = None
        self.stage_started_utc: str | None = None
        self.stage_started_local: str | None = None
        self.stage_total: int | None = None
        self.stage_current: int | None = None
        self.stage_message: str | None = None
        self.stage_timings: list[dict[str, Any]] = []
        self.enabled = bool(enabled) and sys.stderr.isatty()
        self.heartbeat_interval_s = max(0.1, float(heartbeat_interval_s))
        self.progress_mininterval_s = max(0.02, float(progress_mininterval_s))
        self._heartbeat_stop: threading.Event | None = None
        self._heartbeat_thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._pulse = 0
        self._event_seq = 0
        self._stage_bar: tqdm | None = None
        self.stage_index: int | None = None
        self.stages_total: int | None = None
        self.run_started_at = time.perf_counter()
        self.last_payload: dict[str, Any] | None = None
        self.event('run_initialized', stage=None, current=None, total=None, message='Run initialized')

    def _compute_overall_eta(self, overall_fraction: float | None) -> tuple[float | None, str]:
        run_elapsed = float(time.perf_counter() - self.run_started_at)
        if overall_fraction is None or overall_fraction <= 0.0 or overall_fraction >= 1.0:
            return None, _fmt_seconds(None)
        rate = overall_fraction / max(run_elapsed, 1e-9)
        if rate <= 0.0:
            return None, _fmt_seconds(None)
        overall_eta_s = (1.0 - overall_fraction) / rate
        return overall_eta_s, _fmt_seconds(overall_eta_s)

    def _make_payload(
        self,
        event: str,
        stage: str | None,
        current: int | None = None,
        total: int | None = None,
        message: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        self._event_seq += 1
        now_utc = datetime.now(timezone.utc).isoformat()
        now_local = _now_local_iso()
        run_elapsed_s = float(time.perf_counter() - self.run_started_at)
        elapsed_s = None
        eta_s = None
        stage_fraction = None
        overall_fraction = None
        if stage is not None and self.stage_started_at is not None:
            elapsed_s = float(time.perf_counter() - self.stage_started_at)
        if total and current is not None:
            stage_fraction = float(current) / float(total) if total else None
            if elapsed_s is not None and current > 0 and total > current:
                rate = float(current) / max(float(elapsed_s), 1e-9)
                if rate > 0:
                    eta_s = float(total - current) / rate
        if event == 'run_completed':
            overall_fraction = 1.0
        elif self.stage_index is not None and self.stages_total:
            base = max(self.stage_index - 1, 0)
            frac = 0.0 if stage_fraction is None else float(stage_fraction)
            overall_fraction = float(base + frac) / float(self.stages_total)
        overall_eta_s, overall_eta_hms = self._compute_overall_eta(overall_fraction)
        payload = {
            'timestamp_utc': now_utc,
            'timestamp_local': now_local,
            'event': str(event),
            'stage': stage,
            'current': None if current is None else int(current),
            'total': None if total is None else int(total),
            'message': message,
            'elapsed_s': elapsed_s,
            'elapsed_hms': _fmt_seconds(elapsed_s),
            'eta_s': eta_s,
            'eta_hms': _fmt_seconds(eta_s),
            'stage_fraction': stage_fraction,
            'overall_fraction': overall_fraction,
            'overall_eta_s': overall_eta_s,
            'overall_eta_hms': overall_eta_hms,
            'run_elapsed_s': run_elapsed_s,
            'run_elapsed_hms': _fmt_seconds(run_elapsed_s),
            'stage_index': self.stage_index,
            'stages_total': self.stages_total,
            'stage_started_utc': self.stage_started_utc,
            'stage_started_local': self.stage_started_local,
            'event_seq': int(self._event_seq),
            'pulse': int(self._pulse),
        }
        if extra:
            payload.update(_to_jsonable(extra))
        return payload

    def _write_summary_text(self, payload: dict[str, Any]) -> None:
        lines = [
            'holecolor run status',
            f"timestamp_local: {payload.get('timestamp_local')}",
            f"event: {payload.get('event')}",
            f"stage: {payload.get('stage') or 'idle'}",
            f"stage_index: {payload.get('stage_index')} / {payload.get('stages_total')}",
            f"stage_progress: {payload.get('current')} / {payload.get('total')}",
            f"stage_elapsed: {payload.get('elapsed_hms')}",
            f"stage_eta: {payload.get('eta_hms')}",
            f"overall_fraction: {payload.get('overall_fraction')}",
            f"overall_eta: {payload.get('overall_eta_hms')}",
            f"run_elapsed: {payload.get('run_elapsed_hms')}",
            f"message: {payload.get('message')}",
            '',
            'stage timings:',
        ]
        if self.stage_timings:
            for row in self.stage_timings:
                lines.append(
                    f"- {row.get('stage')}: {row.get('started_local')} -> {row.get('completed_local')} "
                    f"({_fmt_seconds(row.get('elapsed_s'))})"
                )
        else:
            lines.append('- (none completed yet)')
        lines += ['', 'status_line:', format_status_line(payload), '']
        self.summary_path.write_text('\n'.join(lines), encoding='utf-8')

    def event(self, event: str, stage: str | None, current: int | None = None, total: int | None = None, message: str | None = None, **extra: Any) -> None:
        with self._lock:
            payload = self._make_payload(event, stage, current=current, total=total, message=message, **extra)
            line = json.dumps(payload, ensure_ascii=False)
            with self.jsonl_path.open('a', encoding='utf-8') as f:
                f.write(line + '\n')
            self.current_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
            self._write_summary_text(payload)
            self.last_payload = payload

    def _stage_postfix(self, current: int | None, total: int | None) -> str:
        elapsed_s = None if self.stage_started_at is None else float(time.perf_counter() - self.stage_started_at)
        parts = [f'elapsed={_fmt_seconds(elapsed_s)}']
        overall_frac = None
        if self.stage_index is not None and self.stages_total:
            frac = 0.0
            if total and current is not None and total > 0:
                frac = float(current) / float(total)
            overall_frac = float(max(self.stage_index - 1, 0) + frac) / float(self.stages_total)
            _, overall_eta_hms = self._compute_overall_eta(overall_frac)
            parts.append(f'run_eta={overall_eta_hms}')
        if total and current is not None and current > 0:
            rate = float(current) / max(float(elapsed_s or 0.0), 1e-9)
            eta_s = float(total - current) / rate if rate > 0 and total >= current else None
            parts.append(f'eta={_fmt_seconds(eta_s)}')
            parts.append(f'{current}/{total}')
        else:
            spinner = '|/-\\'
            parts.append(spinner[self._pulse % len(spinner)])
        if overall_frac is not None:
            parts.append(f'overall={100.0 * overall_frac:4.1f}%')
        return ' | '.join(parts)

    def _refresh_stage_bar(self, current: int | None, total: int | None) -> None:
        if self._stage_bar is None:
            return
        if total is not None and self._stage_bar.total != int(total):
            self._stage_bar.total = int(total)
        target = None
        if current is not None:
            target = int(current)
        elif total is None:
            target = self._stage_bar.n
        if target is not None and target > self._stage_bar.n:
            self._stage_bar.update(target - self._stage_bar.n)
        self._pulse += 1
        self._stage_bar.set_postfix_str(self._stage_postfix(current, total), refresh=False)

    def _heartbeat_loop(self) -> None:
        while self._heartbeat_stop is not None and not self._heartbeat_stop.wait(self.heartbeat_interval_s):
            with self._lock:
                self._pulse += 1
                self.event(
                    'stage_heartbeat',
                    stage=self.stage,
                    current=self.stage_current,
                    total=self.stage_total,
                    message=self.stage_message,
                )
                self._refresh_stage_bar(self.stage_current, self.stage_total)

    def _start_heartbeat(self) -> None:
        self._stop_heartbeat()
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True, name='holecolor-status-heartbeat')
        self._heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        if self._heartbeat_stop is not None:
            self._heartbeat_stop.set()
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=max(0.1, self.heartbeat_interval_s * 2.0))
        self._heartbeat_stop = None
        self._heartbeat_thread = None

    def start(self, stage: str, total: int | None = None, message: str | None = None, **extra: Any) -> None:
        with self._lock:
            self.stage = str(stage)
            self.stage_started_at = time.perf_counter()
            self.stage_started_utc = datetime.now(timezone.utc).isoformat()
            self.stage_started_local = _now_local_iso()
            self.stage_total = None if total is None else int(total)
            self.stage_current = 0 if total is not None else None
            self.stage_message = message
            if self.enabled:
                stage_total = 1 if total is None else max(1, int(total))
                self._stage_bar = tqdm(
                    total=stage_total,
                    desc=f'Stage: {self.stage}',
                    leave=False,
                    disable=not self.enabled,
                    dynamic_ncols=True,
                    position=1,
                    mininterval=self.progress_mininterval_s,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                )
            if self.enabled:
                _console('stage started: ' + self.stage + (f' :: {message}' if message else ''))
            self.event('stage_started', stage=self.stage, current=self.stage_current, total=self.stage_total, message=message, **extra)
            self._start_heartbeat()

    def progress(self, current: int | None = None, total: int | None = None, message: str | None = None, **extra: Any) -> None:
        with self._lock:
            if current is not None:
                self.stage_current = int(current)
            if total is not None:
                self.stage_total = int(total)
            if message is not None:
                self.stage_message = message
            self.event('stage_progress', stage=self.stage, current=self.stage_current, total=self.stage_total, message=self.stage_message, **extra)
            self._refresh_stage_bar(self.stage_current, self.stage_total)

    def complete(self, message: str | None = None, **extra: Any) -> None:
        self._stop_heartbeat()
        with self._lock:
            elapsed_s = None if self.stage_started_at is None else float(time.perf_counter() - self.stage_started_at)
            if self._stage_bar is not None:
                if self.stage_total is not None and self._stage_bar.n < self._stage_bar.total:
                    self._stage_bar.update(self._stage_bar.total - self._stage_bar.n)
                self._stage_bar.close()
                self._stage_bar = None
            completed_utc = datetime.now(timezone.utc).isoformat()
            completed_local = _now_local_iso()
            self.stage_timings.append({
                'stage': self.stage,
                'started_utc': self.stage_started_utc,
                'started_local': self.stage_started_local,
                'completed_utc': completed_utc,
                'completed_local': completed_local,
                'elapsed_s': elapsed_s,
                'message': message or self.stage_message,
            })
            self.timings_path.write_text(json.dumps(self.stage_timings, indent=2), encoding='utf-8')
            self.event('stage_completed', stage=self.stage, current=self.stage_total if self.stage_total is not None else self.stage_current, total=self.stage_total, message=message or self.stage_message, **extra)
            if self.enabled:
                _console(f'stage completed: {self.stage} | elapsed={_fmt_seconds(elapsed_s)}')
            self.stage = None
            self.stage_started_at = None
            self.stage_started_utc = None
            self.stage_started_local = None
            self.stage_total = None
            self.stage_current = None
            self.stage_message = None

    def fail(self, message: str, **extra: Any) -> None:
        self._stop_heartbeat()
        with self._lock:
            if self._stage_bar is not None:
                self._stage_bar.close()
                self._stage_bar = None
            self.event('stage_failed', stage=self.stage, current=self.stage_current, total=self.stage_total, message=message, **extra)
            if self.enabled:
                _console(f'stage failed: {self.stage} :: {message}')
            self.stage = None
            self.stage_started_at = None
            self.stage_started_utc = None
            self.stage_started_local = None
            self.stage_total = None
            self.stage_current = None
            self.stage_message = None

    def finish_run(self, message: str = 'Run finished', **extra: Any) -> None:
        self._stop_heartbeat()
        with self._lock:
            self.event('run_completed', stage=self.stage, current=self.stage_current, total=self.stage_total, message=message, **extra)


class PipelineProgress:
    def __init__(
        self,
        out_dir: Path,
        stages: list[str],
        enabled: bool = True,
        heartbeat_interval_s: float = 1.0,
        progress_mininterval_s: float = 0.2,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.stages = list(stages)
        self.enabled = bool(enabled) and sys.stderr.isatty()
        self.progress_mininterval_s = max(0.02, float(progress_mininterval_s))
        self.tracker = RunStatusTracker(
            self.out_dir,
            enabled=self.enabled,
            heartbeat_interval_s=heartbeat_interval_s,
            progress_mininterval_s=self.progress_mininterval_s,
        )
        self.tracker.stage_plan_path.write_text(json.dumps({'stages': self.stages}, indent=2), encoding='utf-8')
        self.index = 0
        self.master = tqdm(
            total=len(self.stages),
            desc='Pipeline stages',
            leave=True,
            disable=not self.enabled,
            dynamic_ncols=True,
            position=0,
            mininterval=self.progress_mininterval_s,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
        )

    @contextmanager
    def stage(self, stage_name: str, total: int | None = None, message: str | None = None, **extra: Any) -> Iterator[RunStatusTracker]:
        self.index += 1
        self.tracker.stage_index = self.index
        self.tracker.stages_total = len(self.stages)
        self.tracker.start(stage_name, total=total, message=message, **extra)
        try:
            yield self.tracker
        except Exception as exc:
            self.tracker.fail(str(exc))
            if self.enabled:
                self.master.close()
            raise
        else:
            self.tracker.complete(message=message)
            if self.enabled:
                self.master.update(1)
                self.master.set_postfix_str(stage_name, refresh=False)

    def close(self) -> None:
        self.tracker.finish_run(message='Pipeline finished')
        if self.enabled:
            self.master.close()
