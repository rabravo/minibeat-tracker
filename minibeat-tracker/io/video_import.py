"""MP4 import — Path A (ffmpeg direct) and Path B (ab_video_mp42tif.sh)."""
from __future__ import annotations

import atexit
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Path A — direct ffmpeg extraction
# ---------------------------------------------------------------------------

def extract_frames_ffmpeg(
    mp4_path: Path,
    fps: float,
    progress_callback=None,
) -> Path:
    """Extract grayscale TIF frames to a managed temp directory.

    The temp directory is registered for automatic deletion on interpreter exit.
    Returns the directory containing the extracted frame_NNNN.tif files.
    """
    out_dir = Path(tempfile.mkdtemp(prefix="cardio_frames_"))
    atexit.register(shutil.rmtree, str(out_dir), True)

    # Two-pass: first get duration for progress, then extract
    duration = _probe_duration(mp4_path)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(mp4_path),
        "-vf", f"fps={fps}",
        "-pix_fmt", "gray",
        "-progress", "pipe:1",
        str(out_dir / "frame_%04d.tif"),
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    for line in proc.stdout:
        line = line.strip()
        if line.startswith("out_time_ms=") and duration and progress_callback:
            try:
                ms = int(line.split("=")[1])
                pct = min(99, int(ms / 1000 / duration * 100))
                progress_callback(pct)
            except ValueError:
                pass

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")

    if progress_callback:
        progress_callback(100)

    return out_dir


def _probe_duration(mp4_path: Path) -> float | None:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(mp4_path)],
            capture_output=True, text=True, timeout=10,
        )
        return float(r.stdout.strip())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Path B — ab_video_mp42tif.sh
# ---------------------------------------------------------------------------

def extract_via_script(
    mp4_path: Path,
    fps: float,
    progress_callback=None,
) -> Path:
    """Run ab_video_mp42tif.sh --gray, select individual frames, return output dir.

    Requires ab_video_mp42tif.sh to be on PATH.
    """
    script = shutil.which("ab_video_mp42tif.sh")
    if script is None:
        raise FileNotFoundError(
            "ab_video_mp42tif.sh not found in PATH. "
            "Ensure /usr/local/bin is in your PATH."
        )

    if progress_callback:
        progress_callback(5)

    result = subprocess.run(
        [script, "--gray", str(mp4_path), str(fps)],
        input="f\n",          # select individual frames (option f)
        text=True,
        capture_output=True,
        cwd=mp4_path.parent,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"ab_video_mp42tif.sh failed:\n{result.stderr.strip()}"
        )

    if progress_callback:
        progress_callback(100)

    # Mirror the script's output dir naming: sed 's/[^a-zA-Z0-9_-]/_/g' on stem
    safe_stem = re.sub(r"[^a-zA-Z0-9_-]", "_", mp4_path.stem)
    out_dir = mp4_path.parent / f"{safe_stem}_tif"

    if not out_dir.is_dir():
        raise FileNotFoundError(
            f"Expected output directory not found: {out_dir}\n"
            f"Script stdout: {result.stdout.strip()}"
        )

    return out_dir
