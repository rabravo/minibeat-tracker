"""
mb_pipeline.py
--------------
Headless pipeline for MiniBeat Tracker — runs the full motion analysis
workflow without a GUI, designed for use on HPC clusters via mb_server.py.

Pipeline steps mirror the four-panel napari GUI:
  1. Load TIF frames from directory
  2. Motion estimation  (numba block-match + joblib parallelism)
  3. Contraction data + peak detection
  4. Peak analysis + CSV + amplitude video export
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional


BASE_DIR = Path(__file__).resolve().parent

LogFn = Callable[[str], None]


# ---------------------------------------------------------------------------
# Bootstrap: register minibeat-tracker as importable minibeat_tracker
# ---------------------------------------------------------------------------

def _bootstrap_mbt() -> None:
    """Make minibeat-tracker/ importable under the name minibeat_tracker."""
    if "minibeat_tracker" in sys.modules:
        return

    pkg_root = BASE_DIR.parent / "minibeat-tracker"
    if not pkg_root.is_dir():
        raise RuntimeError(
            f"minibeat-tracker package not found at {pkg_root}. "
            "Make sure minibeat-hpc-tracker/ and minibeat-tracker/ sit "
            "side-by-side inside cardio-tracker/."
        )

    # Register all packages as stubs (shallowest first) WITHOUT executing
    # __init__.py. This lets Python's regular import machinery handle lazy
    # module loading and relative imports correctly, avoiding cross-dependency
    # ordering issues. Skip 'widgets' — it imports Qt/napari.
    inits = sorted(pkg_root.rglob("__init__.py"), key=lambda p: len(p.parts))
    for init in inits:
        rel_parts = init.parent.relative_to(pkg_root.parent).parts
        if "widgets" in rel_parts:
            continue
        mod_name = ".".join(part.replace("-", "_") for part in rel_parts)
        if mod_name in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(
            mod_name,
            str(init),
            submodule_search_locations=[str(init.parent)],
        )
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = mod_name
        mod.__path__ = [str(init.parent)]
        sys.modules[mod_name] = mod
        # Don't exec_module here — Python resolves submodule imports lazily
        # once parent packages are in sys.modules with the correct __path__.


_bootstrap_mbt()

from minibeat_tracker.io.reader import scan_tif_folder, load_and_resize   # noqa: E402
from minibeat_tracker.core.motion import get_motion_vectors                # noqa: E402
from minibeat_tracker.core.cleaning import (                               # noqa: E402
    clean_neighbor, clean_fft, clean_variation,
)
from minibeat_tracker.core.analysis import (                               # noqa: E402
    get_contraction_data, detect_peaks, analyze_peaks, do_autocorr,
)
from minibeat_tracker.io.export import (                                   # noqa: E402
    export_beating_data, export_peaks, export_ana_peaks,
)
from minibeat_tracker.io.video import export_amplitude_video               # noqa: E402


# ---------------------------------------------------------------------------
# Job parameters
# ---------------------------------------------------------------------------

@dataclass
class JobParams:
    src_dir: str = ""
    tgt_dir: str = ""

    # Step 2: motion estimation
    framerate: float = 14.0
    mb_size: int = 16
    search_range: int = 7
    delay: int = 2
    num_frames: Optional[int] = None
    scale_width: int = 0
    correct_intensity: bool = False
    min_brightness: float = 0.0

    # Step 3: contraction analysis
    pixel_size_um: float = 0.0
    min_peak_height: float = 0.0
    min_peak_sep_sec: float = 0.0
    skip_first_peak: bool = True

    # Cleaning thresholds
    clean_vect_thresh: float = 2.0
    clean_fft_thresh: float = 4.0
    clean_var_thresh: float = 0.8

    # Video export
    export_video: bool = True
    video_colormap: str = "hot"
    video_alpha: float = 0.6

    n_jobs: int = -1


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    """Run the full MiniBeat headless pipeline for one job."""

    def __init__(self, params: JobParams, log: Optional[LogFn] = None):
        self.params = params
        self.log = log or (lambda msg: print(msg, flush=True))

    # ------------------------------------------------------------------
    def preflight(self) -> list[str]:
        issues: list[str] = []
        src = Path(self.params.src_dir)
        if not self.params.src_dir or not src.is_dir():
            issues.append(f"Input directory not found: {self.params.src_dir!r}")
        if not self.params.tgt_dir:
            issues.append("Output directory is empty")
        else:
            Path(self.params.tgt_dir).mkdir(parents=True, exist_ok=True)
        return issues

    # ------------------------------------------------------------------
    def run(self) -> int:
        problems = self.preflight()
        if problems:
            for p in problems:
                self.log(f"Preflight: {p}")
            return 2

        t0 = datetime.now()
        self.log(f"=== MiniBeat pipeline start @ {t0.isoformat(timespec='seconds')} ===")

        p = self.params
        src = Path(p.src_dir)
        tgt = Path(p.tgt_dir)
        stem = src.name

        # ---- Step 1: load frames ------------------------------------------------
        self.log("Step 1: Loading frames...")
        frames = scan_tif_folder(src)
        if not frames:
            self.log(f"No TIF files found in {src}")
            return 1
        self.log(f"  {len(frames)} frames found")

        ref = load_and_resize(frames[0], p.scale_width, p.mb_size)
        self.log(f"  Reference frame: {frames[0].name} → {ref.shape[1]}×{ref.shape[0]} px")

        # ---- Step 2: motion estimation ------------------------------------------
        self.log("Step 2: Motion estimation (block matching)...")
        raw_vect, raw_amp, box_center, edge_det = get_motion_vectors(
            frames=frames,
            ref_image=ref,
            mb_size=p.mb_size,
            search_range=p.search_range,
            num_frames=p.num_frames,
            delay=p.delay,
            scale_width=p.scale_width,
            correct_intensity=p.correct_intensity,
            min_brightness=p.min_brightness,
            n_jobs=p.n_jobs,
        )
        R, C, _, T = raw_vect.shape
        self.log(f"  Vectors: {R}×{C} blocks × {T} frames  (mb_size={p.mb_size})")

        self.log("  Cleaning: neighbor filter...")
        vect, amp = clean_neighbor(raw_vect, raw_amp, threshold=p.clean_vect_thresh)
        self.log("  Cleaning: FFT filter...")
        vect, amp = clean_fft(vect, p.framerate, freq_thresh=p.clean_fft_thresh)
        self.log("  Cleaning: variation filter...")
        vect, amp = clean_variation(vect, amp, cov_threshold=p.clean_var_thresh)

        # ---- Step 3: contraction data -------------------------------------------
        self.log("Step 3: Contraction analysis...")
        raw_results, _legend = get_contraction_data(
            vect, amp,
            delay=p.delay,
            framerate=p.framerate,
            pixel_size=p.pixel_size_um,
        )
        self.log(f"  Time series: {raw_results.shape[0]} time points")

        peak_times, peak_heights = detect_peaks(
            raw_results,
            framerate=p.framerate,
            min_peak_height=p.min_peak_height,
            min_peak_sep_sec=p.min_peak_sep_sec,
        )
        self.log(f"  Detected {len(peak_times)} peaks")

        beat_rate_ac, corr = do_autocorr(raw_results[:, 1], p.framerate)
        self.log(f"  Autocorr beat rate: {beat_rate_ac:.1f} bpm  (corr={corr:.3f})")

        # ---- Step 4: export -----------------------------------------------------
        self.log("Step 4: Exporting results...")
        height_unit = "µm/sec" if p.pixel_size_um > 0 else "px/sec"

        export_beating_data(tgt / f"{stem}_BeatingData.csv", raw_results, height_unit)
        self.log(f"  Saved {stem}_BeatingData.csv")

        export_peaks(tgt / f"{stem}_RawPeaks.csv", peak_times, peak_heights, height_unit)
        self.log(f"  Saved {stem}_RawPeaks.csv")

        if len(peak_times) >= 4:
            ana = analyze_peaks(peak_times, peak_heights, skip_first=p.skip_first_peak)
            export_ana_peaks(
                tgt / f"{stem}_AnaPeaks.csv",
                tgt / f"{stem}_AnaPeaksMean.csv",
                ana["contract_heights"],
                ana["relax_heights"],
                ana["peaks_int_time_diff"],
                ana["peaks_time_diff"],
                height_unit,
            )
            self.log(f"  Saved {stem}_AnaPeaks.csv, {stem}_AnaPeaksMean.csv")
            self.log(
                f"  Beat rate: {ana['mean_beat_rate']:.1f} ± {ana['std_beat_rate']:.1f} bpm  "
                f"({len(ana['beat_rate_bpm'])} cycles)"
            )
        else:
            self.log(
                f"  Skipping cycle analysis — need ≥ 4 peaks, got {len(peak_times)}"
            )

        if p.export_video:
            self.log("  Exporting amplitude overlay video...")
            export_amplitude_video(
                frames=frames,
                motion_amp=amp,
                output_path=tgt / f"{stem}_amplitude.mp4",
                fps=p.framerate,
                colormap=p.video_colormap,
                alpha=p.video_alpha,
                scale_width=p.scale_width,
                mb_size=p.mb_size,
            )
            self.log(f"  Saved {stem}_amplitude.mp4")

        # Save run params alongside outputs
        params_out = tgt / "mb_params.json"
        params_out.write_text(json.dumps(asdict(p), indent=2, default=str))
        self.log("  Saved mb_params.json")

        elapsed = datetime.now() - t0
        self.log(f"=== MiniBeat pipeline DONE in {elapsed} ===")
        return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Run MiniBeat headlessly.")
    ap.add_argument("--src", required=True, help="Input directory of TIF frames")
    ap.add_argument("--tgt", required=True, help="Output directory")
    ap.add_argument("--framerate", type=float, default=14.0)
    ap.add_argument("--mb-size", type=int, default=16)
    ap.add_argument("--search-range", type=int, default=7)
    ap.add_argument("--delay", type=int, default=2)
    ap.add_argument("--pixel-size-um", type=float, default=0.0)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args(argv)

    params = JobParams(
        src_dir=os.path.abspath(args.src),
        tgt_dir=os.path.abspath(args.tgt),
        framerate=args.framerate,
        mb_size=args.mb_size,
        search_range=args.search_range,
        delay=args.delay,
        pixel_size_um=args.pixel_size_um,
        export_video=not args.no_video,
        n_jobs=args.n_jobs,
    )
    return Pipeline(params).run()


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
