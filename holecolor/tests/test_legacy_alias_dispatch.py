from pathlib import Path
import holecolor.pipeline as pipeline


def test_legacy_alias_to_milestone16_dispatch(monkeypatch, tmp_path: Path) -> None:
    sentinel = {"ok": 1}
    called = {}
    def fake(frames, out_dir, cfg=None):
        called["out_dir"] = out_dir
        called["cfg"] = cfg
        return sentinel
    monkeypatch.setattr(pipeline, "run_milestone16", fake)
    out = pipeline.run_milestone3([], tmp_path, None)
    assert out is sentinel
    assert called["out_dir"] == tmp_path


def test_legacy_alias_to_milestone18_dispatch(monkeypatch, tmp_path: Path) -> None:
    sentinel = {"ok": 2}
    called = {}
    def fake(frames, out_dir, cfg=None, hole_grid_bundle=None):
        called["out_dir"] = out_dir
        called["cfg"] = cfg
        called["hole_grid_bundle"] = hole_grid_bundle
        return sentinel
    monkeypatch.setattr(pipeline, "run_milestone18", fake)
    bundle = object()
    out = pipeline.run_milestone24([], tmp_path, None, hole_grid_bundle=bundle)
    assert out is sentinel
    assert called["out_dir"] == tmp_path
    assert called["hole_grid_bundle"] is bundle
