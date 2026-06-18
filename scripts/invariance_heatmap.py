import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_sampled_frames(video_path: str, max_frames: int = 300, grayscale: bool = True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)

    if frame_count <= 0:
        raise RuntimeError("Video reports zero frames.")

    n_samples = min(max_frames, frame_count)
    sample_idxs = np.linspace(0, frame_count - 1, n_samples, dtype=int)

    frames = []
    first_rgb = None

    for idx in sample_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if first_rgb is None:
            first_rgb = rgb.copy()

        if grayscale:
            arr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            arr = rgb.astype(np.float32)

        frames.append(arr)

    cap.release()

    if not frames:
        raise RuntimeError("No frames could be read from the video.")

    stack = np.stack(frames, axis=0)

    metadata = {
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "fps": fps,
        "frames_used": len(frames),
    }

    return stack, first_rgb, metadata


def compute_invariance_map(stack: np.ndarray, robust_percentile: float = 95.0):
    """
    stack shape:
      grayscale -> (T, H, W)
      RGB       -> (T, H, W, 3)
    """
    temporal_std = stack.std(axis=0)

    # If RGB, reduce channel std to one scalar per pixel
    if temporal_std.ndim == 3:
        temporal_std_scalar = np.linalg.norm(temporal_std, axis=-1)
    else:
        temporal_std_scalar = temporal_std

    scale = np.percentile(temporal_std_scalar, robust_percentile)
    if scale <= 1e-12:
        scale = max(float(temporal_std_scalar.max()), 1.0)

    invariance = 1.0 - np.clip(temporal_std_scalar / scale, 0.0, 1.0)
    return invariance, temporal_std_scalar


def save_heatmap(invariance: np.ndarray, out_path: Path, title: str = "Pixel Signal Invariance Heatmap"):
    plt.figure(figsize=(8, 6))
    im = plt.imshow(invariance)
    plt.colorbar(im, label="Invariance score (higher = more stable)")
    plt.title(title)
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_overlay(base_rgb: np.ndarray, invariance: np.ndarray, out_path: Path):
    plt.figure(figsize=(8, 6))
    plt.imshow(base_rgb)
    plt.imshow(invariance, alpha=0.45)
    plt.title("Invariance Overlay")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compute pixel signal invariance heatmap from a video.")
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("--outdir", default="invariance_output", help="Output directory")
    parser.add_argument("--max-frames", type=int, default=300, help="Maximum number of sampled frames")
    parser.add_argument("--rgb", action="store_true", help="Use RGB instead of grayscale")
    parser.add_argument("--pctl", type=float, default=95.0, help="Robust percentile for normalization")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    stack, first_rgb, metadata = load_sampled_frames(
        args.video,
        max_frames=args.max_frames,
        grayscale=not args.rgb
    )

    invariance, temporal_std = compute_invariance_map(stack, robust_percentile=args.pctl)

    np.save(outdir / "invariance.npy", invariance)
    np.save(outdir / "temporal_std.npy", temporal_std)

    save_heatmap(invariance, outdir / "invariance_heatmap.png")
    save_overlay(first_rgb, invariance, outdir / "invariance_overlay.png")

    print("Done.")
    print(f"Video: {args.video}")
    print(f"Frames used: {metadata['frames_used']} / {metadata['frame_count']}")
    print(f"Resolution: {metadata['width']} x {metadata['height']}")
    print(f"FPS: {metadata['fps']:.3f}")
    print(f"Outputs saved in: {outdir.resolve()}")


if __name__ == "__main__":
    main()