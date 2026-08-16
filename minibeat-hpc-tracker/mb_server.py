"""
mb_server.py
------------
Flask web front-end for MiniBeat Tracker HPC mode.

Designed for headless Linux clusters reached over SSH port-forwarding.
The pipeline itself is driven through mb_pipeline.py (no Qt, no napari).

Security: binds to 127.0.0.1 by default. The SSH tunnel is the only path
in. There is no auth on the HTTP layer.

Usage
-----
Start on the cluster node:

    ./run_server.sh

Open an SSH tunnel from your laptop:

    ssh -N -L 8766:localhost:8766 user@cluster.example.edu

Open http://localhost:8766/ in your browser.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import threading
import time
import uuid
import zipfile
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Deque, Dict, List, Optional

try:
    from zoneinfo import ZoneInfo
    _DISPLAY_TZ = ZoneInfo("America/New_York")
except ImportError:
    _DISPLAY_TZ = timezone(timedelta(hours=-5), "EST")

try:
    from flask import (Flask, abort, jsonify, render_template, request,
                       send_file, send_from_directory, Response)
    from werkzeug.utils import secure_filename
except ImportError:
    sys.stderr.write(
        "Flask is not installed.\n"
        "  conda activate minibeat-hpc\n"
        "  # or: pip install flask werkzeug\n"
    )
    raise

from mb_pipeline import JobParams, Pipeline

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8766
DEFAULT_DATA_ROOT = BASE_DIR / "WebJobs"

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}
MAX_LOG_LINES = 10000
MAX_UPLOAD_MB = 8192       # 8 GB ceiling — large videos welcome


# ---------------------------------------------------------------------------
# In-memory job registry
# ---------------------------------------------------------------------------

class Job:
    STATE_PENDING = "pending"
    STATE_RUNNING = "running"
    STATE_DONE    = "done"
    STATE_ERROR   = "error"

    def __init__(self, job_id: str, root: Path):
        self.id   = job_id
        self.root = root
        self.input_dir  = root / "input"
        self.output_dir = root / "output"
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.params: Optional[JobParams] = None
        self.state   = self.STATE_PENDING
        self.created = datetime.now(tz=_DISPLAY_TZ)
        self.started:  Optional[datetime] = None
        self.finished: Optional[datetime] = None
        self.rc: Optional[int] = None

        self._log:  Deque[str]          = deque(maxlen=MAX_LOG_LINES)
        self._lock  = threading.Lock()
        self._cv    = threading.Condition(self._lock)
        self._thread: Optional[threading.Thread] = None

    def append_log(self, line: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        with self._cv:
            self._log.append(f"[{ts}] {line}")
            self._cv.notify_all()

    def log_since(self, offset: int) -> List[str]:
        with self._lock:
            total = len(self._log)
            if offset >= total:
                return []
            return list(self._log)[offset:]

    def log_total(self) -> int:
        with self._lock:
            return len(self._log)

    def start(self, params: JobParams) -> None:
        if self.state == self.STATE_RUNNING:
            raise RuntimeError("Job is already running")
        self.params = params
        self.state  = self.STATE_RUNNING
        self.started = datetime.now(tz=_DISPLAY_TZ)

        def _runner():
            try:
                pipe = Pipeline(params, log=self.append_log)
                self.rc = pipe.run()
                self.state = self.STATE_DONE if self.rc == 0 else self.STATE_ERROR
            except Exception as ex:
                self.append_log(f"FATAL: {ex}")
                self.state = self.STATE_ERROR
                self.rc = 99
            finally:
                self.finished = datetime.now(tz=_DISPLAY_TZ)
                with self._cv:
                    self._cv.notify_all()

        t = threading.Thread(target=_runner, daemon=True)
        self._thread = t
        t.start()

    def is_running(self) -> bool:
        return self.state == self.STATE_RUNNING

    def snapshot(self) -> Dict:
        elapsed: Optional[str] = None
        if self.started and self.finished:
            secs = int((self.finished - self.started).total_seconds())
            h, rem = divmod(secs, 3600)
            m, s   = divmod(rem, 60)
            elapsed = f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
        return {
            "id":       self.id,
            "state":    self.state,
            "rc":       self.rc,
            "created":  self.created.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "started":  self.started.isoformat(timespec="seconds")  if self.started  else None,
            "finished": self.finished.isoformat(timespec="seconds") if self.finished else None,
            "elapsed":  elapsed,
            "cpu_count": os.cpu_count() or 1,
            "input_dir":  str(self.input_dir),
            "output_dir": str(self.output_dir),
            "log_total":  self.log_total(),
            "params":     asdict(self.params) if self.params else None,
        }


class JobRegistry:
    def __init__(self, data_root: Path):
        self.data_root = data_root
        self.data_root.mkdir(parents=True, exist_ok=True)
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._reload_existing()

    def _reload_existing(self) -> None:
        for child in self.data_root.iterdir():
            if not child.is_dir():
                continue
            try:
                job = Job(child.name, child)
                job.state = Job.STATE_DONE
                # Restore params if mb_params.json exists
                params_file = child / "output" / "mb_params.json"
                if params_file.exists():
                    data = json.loads(params_file.read_text())
                    job.params = JobParams(**{
                        k: v for k, v in data.items()
                        if k in JobParams.__dataclass_fields__
                    })
                self._jobs[child.name] = job
            except Exception:
                pass

    def create(self) -> Job:
        with self._lock:
            jid = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
            job = Job(jid, self.data_root / jid)
            self._jobs[jid] = job
            return job

    def get(self, jid: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(jid)

    def list(self) -> List[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _allowed(name: str) -> bool:
    return Path(name).suffix.lower() in ALLOWED_EXTENSIONS


def _safe_relative(base: Path, target: Path) -> Optional[Path]:
    try:
        return target.resolve().relative_to(base.resolve())
    except (ValueError, OSError):
        return None


def _params_from_form(form) -> JobParams:
    def _bool(v: str) -> bool:
        return str(v).lower() in ("1", "true", "yes", "on")
    def _int_or_none(v: str) -> Optional[int]:
        v = v.strip()
        return int(v) if v and v != "0" else None

    return JobParams(
        framerate        = float(form.get("framerate",        14.0)),
        mb_size          = int(form.get("mb_size",            16)),
        search_range     = int(form.get("search_range",        7)),
        delay            = int(form.get("delay",               2)),
        num_frames       = _int_or_none(form.get("num_frames", "")),
        scale_width      = int(form.get("scale_width",         0)),
        correct_intensity= _bool(form.get("correct_intensity", "no")),
        min_brightness   = float(form.get("min_brightness",    0.0)),
        pixel_size_um    = float(form.get("pixel_size_um",     0.0)),
        min_peak_height  = float(form.get("min_peak_height",   0.0)),
        min_peak_sep_sec = float(form.get("min_peak_sep_sec",  0.0)),
        skip_first_peak  = _bool(form.get("skip_first_peak",  "yes")),
        clean_vect_thresh= float(form.get("clean_vect_thresh", 2.0)),
        clean_fft_thresh = float(form.get("clean_fft_thresh",  4.0)),
        clean_var_thresh = float(form.get("clean_var_thresh",  0.8)),
        export_video     = _bool(form.get("export_video",     "yes")),
        video_colormap   = form.get("video_colormap",         "hot"),
        video_alpha      = float(form.get("video_alpha",       0.6)),
        n_jobs           = int(form.get("n_jobs",             -1)),
    )


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(data_root: Path) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(BASE_DIR / "web_templates"),
        static_folder=str(BASE_DIR / "web_static"),
    )
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
    registry = JobRegistry(data_root)

    @app.route("/")
    def index():
        return render_template(
            "index.html",
            jobs=[j.snapshot() for j in registry.list()],
        )

    @app.route("/jobs", methods=["POST"])
    def create_job():
        f = request.files.get("video")
        if not f or not f.filename:
            return jsonify({"error": "No video file uploaded"}), 400

        name = secure_filename(f.filename)
        if not name or not _allowed(name):
            return jsonify({
                "error": f"Unsupported file type (allowed: {sorted(ALLOWED_EXTENSIONS)})",
            }), 400

        job = registry.create()
        f.save(str(job.input_dir / name))
        job.append_log(f"Uploaded: {name}")

        params = _params_from_form(request.form)
        params.src_dir       = str(job.input_dir)
        params.tgt_dir       = str(job.output_dir)
        params.video_filename = name

        try:
            job.start(params)
        except RuntimeError as ex:
            return jsonify({"error": str(ex)}), 409

        return jsonify({"job_id": job.id, "video": name}), 202

    @app.route("/jobs/<jid>")
    def job_detail(jid):
        job = registry.get(jid)
        if not job:
            abort(404)
        return render_template("job.html", job=job.snapshot())

    @app.route("/jobs/<jid>/status")
    def job_status(jid):
        job = registry.get(jid)
        if not job:
            abort(404)
        return jsonify(job.snapshot())

    @app.route("/jobs/<jid>/log")
    def job_log(jid):
        job = registry.get(jid)
        if not job:
            abort(404)
        offset = int(request.args.get("offset", 0))
        lines  = job.log_since(offset)
        return jsonify({
            "offset":  offset + len(lines),
            "lines":   lines,
            "running": job.is_running(),
            "state":   job.state,
        })

    @app.route("/jobs/<jid>/stream")
    def job_stream(jid):
        job = registry.get(jid)
        if not job:
            abort(404)

        def gen():
            offset = 0
            last_keepalive = time.time()
            while True:
                lines = job.log_since(offset)
                if lines:
                    offset += len(lines)
                    for ln in lines:
                        yield f"data: {ln}\n\n"
                if not job.is_running() and offset >= job.log_total():
                    yield f"event: done\ndata: {job.state}\n\n"
                    return
                if time.time() - last_keepalive > 15:
                    yield ": keepalive\n\n"
                    last_keepalive = time.time()
                time.sleep(0.4)

        return Response(gen(), mimetype="text/event-stream")

    @app.route("/jobs/<jid>/files")
    def job_files(jid):
        job = registry.get(jid)
        if not job:
            abort(404)
        out: List[Dict] = []
        for path in job.output_dir.rglob("*"):
            if path.is_file():
                rel = path.relative_to(job.output_dir)
                out.append({
                    "path": str(rel).replace(os.sep, "/"),
                    "size": path.stat().st_size,
                })
        out.sort(key=lambda x: x["path"])
        return jsonify({"files": out})

    @app.route("/jobs/<jid>/download/<path:relpath>")
    def job_download(jid, relpath):
        job = registry.get(jid)
        if not job:
            abort(404)
        target = (job.output_dir / relpath).resolve()
        if _safe_relative(job.output_dir, target) is None or not target.is_file():
            abort(404)
        return send_from_directory(job.output_dir, relpath, as_attachment=True)

    @app.route("/jobs/<jid>/zip")
    def job_zip(jid):
        job = registry.get(jid)
        if not job:
            abort(404)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in job.output_dir.rglob("*"):
                if path.is_file():
                    zf.write(path, arcname=path.relative_to(job.output_dir))
        buf.seek(0)
        return send_file(
            buf, mimetype="application/zip", as_attachment=True,
            download_name=f"minibeat-{job.id}.zip",
        )

    @app.route("/jobs/<jid>", methods=["DELETE"])
    def job_delete(jid):
        job = registry.get(jid)
        if not job:
            abort(404)
        if job.is_running():
            return jsonify({"error": "Cannot delete a running job"}), 409
        try:
            shutil.rmtree(job.root, ignore_errors=True)
        finally:
            registry._jobs.pop(jid, None)
        return ("", 204)

    @app.route("/health")
    def health():
        return jsonify({
            "status": "ok",
            "base_dir":  str(BASE_DIR),
            "data_root": str(data_root),
            "cpu_count": os.cpu_count(),
        })

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

BANNER = r"""
============================================================
 MiniBeat HPC web controller
============================================================
 Listening on:   http://{host}:{port}
 Job data root:  {data_root}

 To reach this from your laptop, open an SSH tunnel:

     ssh -N -L {port}:localhost:{port} {user}@<this-host>

 then open:

     http://localhost:{port}/

 The server binds to 127.0.0.1 only — there is no
 authentication on the HTTP port. The SSH tunnel is the
 only way in.
============================================================
"""


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="MiniBeat HPC web controller (browser UI over SSH tunnel).")
    parser.add_argument("--host",      default=DEFAULT_HOST)
    parser.add_argument("--port",      type=int, default=DEFAULT_PORT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--debug",     action="store_true")
    args = parser.parse_args(argv)

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        sys.stderr.write(
            f"WARNING: binding to {args.host} exposes the port outside "
            "the SSH tunnel. There is no auth on this server.\n"
        )

    app = create_app(args.data_root)
    print(BANNER.format(
        host=args.host, port=args.port,
        data_root=args.data_root,
        user=os.environ.get("USER", "you"),
    ))
    app.run(host=args.host, port=args.port, debug=args.debug,
            threaded=True, use_reloader=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
