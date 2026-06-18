from __future__ import annotations
import cv2, json, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage as ndi


def norm01(x, lo=1, hi=99):
    a, b = np.percentile(x, [lo, hi])
    if b <= a:
        b = a + 1
    return np.clip((x - a) / (b - a), 0, 1)


def extract_points(score, mask, min_score=0.7, min_dist=18, peak_size=15):
    mx = ndi.maximum_filter(score, size=peak_size)
    cand = (score == mx) & mask & (score >= min_score)
    ys, xs = np.where(cand)
    vals = score[ys, xs]
    order = np.argsort(vals)[::-1]
    selected = []
    for idx in order:
        x, y, v = int(xs[idx]), int(ys[idx]), float(vals[idx])
        if all((x - sx) ** 2 + (y - sy) ** 2 >= min_dist ** 2 for sx, sy, _ in selected):
            selected.append((x, y, v))
    return selected


def detect_holes(arr, wafer, consensus):
    """Detect holes using temporal color variance as primary cue.
    
    Steps:
    1. Identify low-variance pixels (holes are color-stable across frames)
    2. Constrain to scaffold region with morphological margin
    3. Extract candidate centers from consensus map within low-variance zone
    4. Enforce regular grid structure via neighbor spacing consistency
    5. Return hole centers with unified radius
    """
    gray = arr.mean(axis=3)
    color_std = arr.std(axis=0).mean(axis=2)

    if wafer.sum() < 0.05 * wafer.size:
        wafer = np.ones_like(wafer, dtype=bool)

    lowvar_thr = np.percentile(color_std[wafer], 45)
    low_var_mask = (color_std <= lowvar_thr).astype(np.uint8)

    low_var_mask = low_var_mask & wafer.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    low_var_mask = cv2.morphologyEx(low_var_mask, cv2.MORPH_CLOSE, kernel)
    low_var_mask = cv2.dilate(low_var_mask, kernel)

    candidate_points = extract_points(consensus, low_var_mask.astype(bool), min_score=0.38, min_dist=12, peak_size=11)
    if not candidate_points:
        return [], low_var_mask

    centers = np.array([(x, y) for x, y, _ in candidate_points], dtype=np.float32)

    labels, n = ndi.label(low_var_mask)
    areas = [int((labels == i).sum()) for i in range(1, n + 1)]
    if areas:
        r_med = float(np.median(np.sqrt(np.array(areas) / np.pi)))
    else:
        r_med = 7.0

    if len(centers) > 1:
        from scipy.spatial import cKDTree
        kd = cKDTree(centers)
        k = min(len(centers), 9)
        dists, _ = kd.query(centers, k=k)
        spacing = float(np.median(dists[:, 1]))
        selected = []
        for i in range(len(centers)):
            neigh = dists[i, 1:]
            consistent = np.sum((neigh >= 0.65 * spacing) & (neigh <= 1.35 * spacing))
            if consistent >= 3 or len(centers) < 8:
                selected.append(i)
        centers = centers[selected]

    holes = [(float(x), float(y), max(3.0, min(40.0, r_med))) for x, y in centers]
    return holes, low_var_mask


def run_probe(video_path: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    arr = np.stack(frames)
    H, W = arr.shape[1:3]

    gray = arr.mean(axis=3)
    mean_gray = gray.mean(axis=0)
    med_gray = np.median(gray, axis=0)
    std_gray = gray.std(axis=0)
    mad_gray = np.median(np.abs(gray - med_gray[None, :, :]), axis=0)
    ptp_gray = gray.max(axis=0) - gray.min(axis=0)

    # Wafer/support from dark support in temporal mean
    blur = cv2.GaussianBlur(mean_gray.astype(np.uint8), (0, 0), 9)
    mask = (blur < 100).astype(np.uint8)
    labels, n = ndi.label(mask)
    areas = [(labels == i).sum() for i in range(1, n + 1)]
    lab = 1 + int(np.argmax(areas))
    wafer_comp = (labels == lab).astype(np.uint8)
    contours, _ = cv2.findContours(wafer_comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    (xc, yc), r = cv2.minEnclosingCircle(max(contours, key=cv2.contourArea))
    Y, X = np.indices((H, W))
    wafer = (X - xc) ** 2 + (Y - yc) ** 2 <= (0.98 * r) ** 2

    inv_std = norm01(1.0 / (std_gray + 1e-6))
    inv_mad = norm01(1.0 / (mad_gray + 1e-6))
    blob_log = norm01(-ndi.gaussian_laplace(cv2.GaussianBlur(mean_gray.astype(np.float32), (0, 0), 1.0), sigma=6))
    center_ring = norm01(cv2.GaussianBlur(mean_gray.astype(np.float32), (0, 0), 3) - cv2.GaussianBlur(mean_gray.astype(np.float32), (0, 0), 10))
    consensus = 0.40 * inv_std + 0.20 * inv_mad + 0.20 * blob_log + 0.20 * center_ring

    candidates, low_var_mask = detect_holes(arr, wafer, consensus)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, title, img in zip(axes, ["inv_std", "blob_log", "consensus"], [inv_std, blob_log, consensus]):
        ax.imshow(img, cmap="viridis")
        ax.add_patch(plt.Circle((xc, yc), r, fill=False, color="white", linewidth=2))
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "probe_maps.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(arr[0])
    ax.add_patch(plt.Circle((xc, yc), r, fill=False, color="white", linewidth=2))
    for x, y, _ in candidates:
        ax.plot(x, y, "o", ms=5, mec="yellow", mfc="none", mew=1.5)
    ax.set_title(f"hole detections ({len(candidates)})")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "consensus_overlay.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    (out / "summary.json").write_text(json.dumps({
        "n_frames": int(arr.shape[0]),
        "wafer_circle": {"cx": float(xc), "cy": float(yc), "r": float(r)},
        "candidate_count": len(candidates),
    }, indent=2))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--out", default="probe_out")
    args = ap.parse_args()
    run_probe(args.video, args.out)
