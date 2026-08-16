"""Block-matching motion estimation — port of motionEstES_PLmods.m.

Core algorithm: exhaustive search (MAD cost) per macroblock.
Frame-level parallelism via joblib; inner loop JIT-compiled with numba.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
from numba import njit, prange
from joblib import Parallel, delayed
import cv2

from ..io.reader import load_and_resize


# ---------------------------------------------------------------------------
# Numba inner loop — exhaustive block match (MAD cost)
# ---------------------------------------------------------------------------

@njit(parallel=True, cache=True)
def _exhaustive_block_match(
    img_ref: np.ndarray,
    img_search: np.ndarray,
    mb_size: int,
    search_range: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return motion_vect[R,C,2] (Y,X) and amplitude[R,C] for one frame pair."""
    h, w = img_ref.shape
    rows = h // mb_size
    cols = w // mb_size

    motion_vect = np.zeros((rows, cols, 2), dtype=np.float64)
    motion_amp = np.zeros((rows, cols), dtype=np.float64)

    for r in prange(rows):
        for c in range(cols):
            r0 = r * mb_size
            c0 = c * mb_size

            best_mad = np.inf
            best_dy = 0
            best_dx = 0

            for dy in range(-search_range, search_range + 1):
                for dx in range(-search_range, search_range + 1):
                    r1 = r0 + dy
                    c1 = c0 + dx
                    if r1 < 0 or r1 + mb_size > h or c1 < 0 or c1 + mb_size > w:
                        continue
                    mad = 0.0
                    for i in range(mb_size):
                        for j in range(mb_size):
                            mad += abs(img_ref[r0 + i, c0 + j] - img_search[r1 + i, c1 + j])
                    mad /= mb_size * mb_size
                    if mad < best_mad:
                        best_mad = mad
                        best_dy = dy
                        best_dx = dx

            motion_vect[r, c, 0] = best_dy
            motion_vect[r, c, 1] = best_dx
            motion_amp[r, c] = np.sqrt(best_dy ** 2 + best_dx ** 2)

    return motion_vect, motion_amp


# ---------------------------------------------------------------------------
# Edge helpers (port of local Canny logic in GetMotionVectors.m)
# ---------------------------------------------------------------------------

def _compute_local_edges(
    img: np.ndarray,
    mb_size: int,
    global_edges: np.ndarray | None,
    min_brightness: float,
) -> np.ndarray:
    """Per-frame Canny edge mask scaled to macroblock grid."""
    if global_edges is not None:
        edges = global_edges.astype(np.uint8)
    else:
        inv = (img.max() - img).astype(np.uint8)
        edges = cv2.Canny(inv, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    if min_brightness > 0:
        edges[img < min_brightness] = 0

    # Scale to macroblock grid: block is included only if all pixels are active
    h, w = edges.shape
    rows, cols = h // mb_size, w // mb_size
    scaled = edges.copy()
    for r in range(rows):
        for c in range(cols):
            block = edges[r * mb_size:(r + 1) * mb_size, c * mb_size:(c + 1) * mb_size]
            val = 1 if block.all() else 0
            scaled[r * mb_size:(r + 1) * mb_size, c * mb_size:(c + 1) * mb_size] = val

    return scaled.astype(np.float64)


# ---------------------------------------------------------------------------
# Per-frame worker (called by joblib)
# ---------------------------------------------------------------------------

def _process_frame_pair(
    i: int,
    frames: list[Path],
    delay: int,
    scale_width: int,
    mb_size: int,
    search_range: int,
    global_edges: np.ndarray | None,
    min_brightness: float,
    correct_intensity: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img_init = load_and_resize(frames[i], scale_width, mb_size)
    img_next = load_and_resize(frames[i + delay], scale_width, mb_size)

    local_edges = _compute_local_edges(img_init, mb_size, global_edges, min_brightness)

    img_masked = img_init * local_edges

    if correct_intensity and img_masked.max() > 0:
        img_masked = img_masked / img_masked.max() * 255.0

    vect, amp = _exhaustive_block_match(
        np.ascontiguousarray(img_masked),
        np.ascontiguousarray(img_next),
        mb_size,
        search_range,
    )
    return vect, amp, local_edges


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_motion_vectors(
    frames: list[Path],
    ref_image: np.ndarray,
    mb_size: int = 16,
    search_range: int = 7,
    num_frames: int | None = None,
    delay: int = 2,
    scale_width: int = 0,
    correct_intensity: bool = False,
    global_edges: np.ndarray | None = None,
    min_brightness: float = 0.0,
    n_jobs: int = -1,
    progress_callback=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parallel frame-level motion estimation.

    Returns
    -------
    raw_motion_vect  : [R, C, 2, T]  — Y/X displacement per block per frame
    raw_motion_amp   : [R, C, T]     — displacement magnitude
    box_center       : [R, C, 2]     — pixel (y, x) center of each block
    edge_detection   : [H, W, T]     — per-frame edge masks
    """
    if num_frames is None:
        num_frames = len(frames)

    n = num_frames - delay

    if min_brightness > 0:
        # Compute global brightness threshold as % of max across first pass
        max_int = max(load_and_resize(frames[i], scale_width, mb_size).max() for i in range(n))
        min_brightness = max_int / 100.0 * min_brightness

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_process_frame_pair)(
            i, frames, delay, scale_width, mb_size, search_range,
            global_edges, min_brightness, correct_intensity,
        )
        for i in range(n)
    )

    if progress_callback:
        progress_callback(100)

    vects = np.stack([r[0] for r in results], axis=-1)   # [R, C, 2, T]
    amps = np.stack([r[1] for r in results], axis=-1)    # [R, C, T]
    edges = np.stack([r[2] for r in results], axis=-1)   # [H, W, T]

    # Box centers (pixel coords of block center)
    rows, cols = vects.shape[:2]
    box_center = np.zeros((rows, cols, 2), dtype=np.float64)
    for r in range(rows):
        for c in range(cols):
            box_center[r, c, 0] = (r + 0.5) * mb_size  # y
            box_center[r, c, 1] = (c + 0.5) * mb_size  # x

    return vects, amps, box_center, edges


def rotate_motion_vect(
    motion_vect: np.ndarray,
    angle_deg: float,
) -> np.ndarray:
    """Rotate Y/X motion vectors by angle_deg (mirrors RotateMotionVect.m)."""
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    # motion_vect: [R, C, 2, T]
    rotated = np.einsum("ij,...j->...i", rot, motion_vect.transpose(0, 1, 3, 2))
    return rotated.transpose(0, 1, 3, 2)
