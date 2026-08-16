"""Amplitude-overlay video export."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from matplotlib import colormaps

from .reader import load_and_resize


def export_amplitude_video(
    frames: list[Path],
    motion_amp: np.ndarray,          # [R, C, T]
    output_path: Path,
    fps: float = 14.0,
    colormap: str = "hot",
    alpha: float = 0.6,
    vmax: float = 0.0,               # 0 = auto from global max
    scale_width: int = 0,
    mb_size: int = 16,
    progress_callback=None,
) -> None:
    """Write MP4 with per-frame amplitude heatmap overlaid on original frames.

    Each video frame is:
        composite = (1 - alpha) * grayscale_original  +  alpha * heatmap
    """
    cmap = colormaps[colormap]
    R, C, T = motion_amp.shape
    n_frames = min(T, len(frames))

    # Global amplitude ceiling for consistent colour scale across frames
    global_max = float(motion_amp.max()) if vmax <= 0 else vmax
    if global_max == 0:
        global_max = 1.0

    # Resolve output frame size from the first source image
    ref = load_and_resize(frames[0], scale_width=scale_width, mb_size=mb_size)
    h, w = ref.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise IOError(f"cv2.VideoWriter could not open: {output_path}")

    try:
        for t in range(n_frames):
            # ---- original frame (grayscale → BGR) ----
            img = load_and_resize(frames[t], scale_width=scale_width, mb_size=mb_size)
            img_u8 = np.clip(img / (img.max() + 1e-9) * 255, 0, 255).astype(np.uint8)
            gray_bgr = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)

            # ---- amplitude heatmap ----
            amp_t = motion_amp[:, :, t]
            amp_norm = np.clip(amp_t / global_max, 0.0, 1.0)
            # resize block grid → full frame resolution
            amp_full = cv2.resize(amp_norm.astype(np.float32), (w, h),
                                  interpolation=cv2.INTER_LINEAR)
            # matplotlib colormap → RGBA float → BGR uint8
            heat_rgba = cmap(amp_full)                        # (H, W, 4) float 0-1
            heat_bgr  = (heat_rgba[:, :, [2, 1, 0]] * 255).astype(np.uint8)

            # ---- composite ----
            composite = cv2.addWeighted(gray_bgr, 1.0 - alpha,
                                        heat_bgr, alpha, 0)
            writer.write(composite)

            if progress_callback:
                progress_callback(int((t + 1) / n_frames * 100))
    finally:
        writer.release()
