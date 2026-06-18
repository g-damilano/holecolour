from __future__ import annotations

from dataclasses import asdict

from pathlib import Path
import pandas as pd
import numpy as np

from holecolor.audit.frame_audit import audit_sequence
from holecolor.config.schema import PipelineConfig
from holecolor.core.paths import ensure_dir
from holecolor.descriptors.color_spaces import rgb_to_hsv
from holecolor.photometry.selector import run_photometry_selection
from holecolor.qc.reports import write_json


def run_color_only_analysis(frames, out_dir: Path, cfg: PipelineConfig) -> dict:
    out_dir = ensure_dir(out_dir)
    audit_dir = ensure_dir(out_dir / 'audit')
    photometry_dir = ensure_dir(out_dir / 'photometry')
    color_dir = ensure_dir(out_dir / 'color_only')
    records = audit_sequence(frames, cfg.audit, progress_cfg=cfg.parallel)
    pd.DataFrame([asdict(r) for r in records]).to_csv(audit_dir / 'frame_qc.csv', index=False)
    winner, scores, corrected = run_photometry_selection(frames, cfg.photometry, progress_cfg=cfg.parallel)
    pd.DataFrame([asdict(s) for s in scores]).to_csv(photometry_dir / 'candidate_scores.csv', index=False)
    rows = []
    for fr in corrected:
        img = fr.image.astype(np.float32) / 255.0
        hsv = rgb_to_hsv(fr.image)
        rows.append({
            'frame_id': fr.frame_id,
            'time_s': fr.time_s,
            'mean_r': float(img[...,0].mean()),
            'mean_g': float(img[...,1].mean()),
            'mean_b': float(img[...,2].mean()),
            'mean_h': float(hsv[...,0].mean()),
            'mean_s': float(hsv[...,1].mean()),
            'mean_v': float(hsv[...,2].mean()),
            'std_r': float(img[...,0].std()),
            'std_g': float(img[...,1].std()),
            'std_b': float(img[...,2].std()),
        })
    df = pd.DataFrame(rows)
    df.to_csv(color_dir / 'frame_color_timeseries.csv', index=False)
    summary = {
        'n_frames': int(len(frames)),
        'winner_correction': winner,
        'accepted_frames': int(sum(r.accepted for r in records)),
        'outputs': ['audit/frame_qc.csv', 'photometry/candidate_scores.csv', 'color_only/frame_color_timeseries.csv'],
    }
    write_json(color_dir / 'summary.json', summary)
    return summary
