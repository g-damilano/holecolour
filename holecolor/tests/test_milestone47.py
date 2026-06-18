from holecolor.core.status import RunStatusTracker, format_status_line


def test_milestone47_format_status_line_includes_stage_and_eta():
    line = format_status_line({
        "event": "stage_heartbeat",
        "stage": "Geometry propagation",
        "current": 3,
        "total": 10,
        "elapsed_hms": "00:12",
        "eta_hms": "00:20",
        "overall_eta_hms": "01:10",
        "overall_fraction": 0.4,
        "message": "Propagating hole geometry",
    })
    assert "Geometry propagation" in line
    assert "3/10" in line
    assert "overall= 40.0%" in line


def test_milestone47_console_heartbeat_updates_tracker(tmp_path):
    tracker = RunStatusTracker(tmp_path, enabled=False, heartbeat_interval_s=0.01)
    tracker.console_heartbeat_interval_s = 0.0
    tracker.start("Per-frame analysis", total=5, message="Extracting per-frame measurements")
    tracker.progress(1, 5, message="Extracting per-frame measurements")
    assert tracker.last_payload is not None
    assert tracker.last_payload.get("event") == "stage_progress"
    tracker.complete()
