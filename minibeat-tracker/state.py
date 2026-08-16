"""Shared session state — replaces MATLAB base workspace assignin/evalin pattern."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np


@dataclass
class SessionState:
    # --- Step 1: folder / image ---
    folder: Path | None = None
    frames: list[Path] = field(default_factory=list)
    ref_image: np.ndarray | None = None
    frames_number: int = 0
    filename: str = ""

    # --- Step 2: motion vectors ---
    raw_motion_vect: np.ndarray | None = None   # [R, C, 2, T]  Y/X per block per frame
    raw_motion_amp: np.ndarray | None = None    # [R, C, T]
    motion_vect: np.ndarray | None = None       # working copy (after cleaning)
    motion_amp: np.ndarray | None = None
    rot_motion_vect: np.ndarray | None = None   # after axis rotation
    box_center: np.ndarray | None = None        # [R, C, 2]  pixel coords of block centers
    edge_detection: np.ndarray | None = None    # [H, W, T]
    center_of_beating: np.ndarray | None = None

    # --- Step 3: contraction time series ---
    raw_results: np.ndarray | None = None       # [N, 9]
    raw_results_legend: list[str] = field(default_factory=list)
    raw_mask: np.ndarray | None = None
    contraction_maps: dict | None = None
    peak_times: np.ndarray | None = None
    peak_heights: np.ndarray | None = None

    # --- Step 4: evaluation ---
    mean_x: np.ndarray | None = None
    mean_y: np.ndarray | None = None
    mean_absolute: np.ndarray | None = None
    areas: float | None = None
    fractional_x: float | None = None
    fractional_y: float | None = None

    # --- Parameters (mirrors MATLAB defaults) ---
    framerate: float = 14.0
    mb_size: int = 16
    search_range: int = 7
    delay: int = 2
    num_frames: int | None = None
    scale_width: int = 0
    correct_intensity: bool = False
    pixel_size_um: float = 0.0
    min_brightness: float = 0.0
    clean_vect_thresh: float = 2.0
    clean_fft_thresh: float = 4.0
    clean_var_thresh: float = 0.8
    cust_max: float = 0.0

    def reset_vectors(self) -> None:
        self.motion_vect = self.raw_motion_vect.copy() if self.raw_motion_vect is not None else None
        self.motion_amp = self.raw_motion_amp.copy() if self.raw_motion_amp is not None else None
        self.rot_motion_vect = None

    def has_vectors(self) -> bool:
        return self.motion_vect is not None

    def has_contraction_data(self) -> bool:
        return self.raw_results is not None
