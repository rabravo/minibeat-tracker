"""Step 1 — Get Folder dock widget."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QLineEdit, QSpinBox, QDoubleSpinBox, QGroupBox,
    QFormLayout, QCheckBox, QButtonGroup, QRadioButton, QProgressBar,
)
from qtpy.QtCore import Qt, QThread, Signal

from ..state import SessionState
from ..io.reader import scan_tif_folder, load_and_resize, load_bioformats_meta
from ..io.video_import import extract_frames_ffmpeg, extract_via_script


# ---------------------------------------------------------------------------
# Background worker for video extraction
# ---------------------------------------------------------------------------

class _VideoImportWorker(QThread):
    progress = Signal(int)
    finished = Signal(str)   # output folder path
    error    = Signal(str)

    def __init__(self, mp4_path: Path, fps: float, use_script: bool):
        super().__init__()
        self._mp4  = mp4_path
        self._fps  = fps
        self._use_script = use_script

    def run(self):
        try:
            if self._use_script:
                out = extract_via_script(
                    self._mp4, self._fps,
                    progress_callback=lambda v: self.progress.emit(v),
                )
            else:
                out = extract_frames_ffmpeg(
                    self._mp4, self._fps,
                    progress_callback=lambda v: self.progress.emit(v),
                )
            self.finished.emit(str(out))
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

class FolderPanel(QWidget):
    """Napari dock widget — Step 1: load TIF folder or import MP4."""

    def __init__(self, viewer, state: SessionState, parent=None):
        super().__init__(parent)
        self._viewer = viewer
        self._state  = state
        self._worker: _VideoImportWorker | None = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setAlignment(Qt.AlignTop)

        # --- TIF folder ---
        folder_grp = QGroupBox("Image Folder")
        flay = QHBoxLayout(folder_grp)
        self._folder_edit = QLineEdit()
        self._folder_edit.setPlaceholderText("Path to TIF folder…")
        self._folder_edit.setReadOnly(True)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._on_browse_folder)
        flay.addWidget(self._folder_edit)
        flay.addWidget(browse_btn)
        root.addWidget(folder_grp)

        load_btn = QPushButton("Load Folder")
        load_btn.clicked.connect(self._on_load_folder)
        root.addWidget(load_btn)

        # --- Video import ---
        vid_grp = QGroupBox("Import MP4 Video")
        vlay = QVBoxLayout(vid_grp)

        vid_file_row = QHBoxLayout()
        self._vid_edit = QLineEdit()
        self._vid_edit.setPlaceholderText("Path to .mp4…")
        self._vid_edit.setReadOnly(True)
        vid_browse = QPushButton("Browse…")
        vid_browse.clicked.connect(self._on_browse_video)
        vid_file_row.addWidget(self._vid_edit)
        vid_file_row.addWidget(vid_browse)
        vlay.addLayout(vid_file_row)

        # Extraction FPS (independent of acquisition FPS)
        fps_row = QFormLayout()
        self._vid_fps = QDoubleSpinBox()
        self._vid_fps.setRange(0.1, 10000.0)
        self._vid_fps.setValue(14.0)
        self._vid_fps.setSuffix(" fps")
        fps_row.addRow("Extract at:", self._vid_fps)
        vlay.addLayout(fps_row)

        # Path A / Path B radio
        self._path_a = QRadioButton("Path A — ffmpeg direct (temp dir, no files kept)")
        self._path_b = QRadioButton("Path B — ab_video_mp42tif.sh (keeps TIFs on disk)")
        self._path_a.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self._path_a)
        grp.addButton(self._path_b)
        vlay.addWidget(self._path_a)
        vlay.addWidget(self._path_b)

        self._vid_import_btn = QPushButton("Import Video")
        self._vid_import_btn.clicked.connect(self._on_import_video)
        vlay.addWidget(self._vid_import_btn)

        self._vid_progress = QProgressBar()
        self._vid_progress.setRange(0, 100)
        self._vid_progress.setVisible(False)
        vlay.addWidget(self._vid_progress)

        root.addWidget(vid_grp)

        # --- Acquisition parameters ---
        acq_grp = QGroupBox("Acquisition Parameters")
        acq_form = QFormLayout(acq_grp)

        self._fps_spin = QDoubleSpinBox()
        self._fps_spin.setRange(0.1, 10000.0)
        self._fps_spin.setValue(14.0)
        self._fps_spin.setSuffix(" fps")
        acq_form.addRow("Frame rate:", self._fps_spin)

        self._delay_spin = QSpinBox()
        self._delay_spin.setRange(1, 100)
        self._delay_spin.setValue(2)
        acq_form.addRow("Frame delay:", self._delay_spin)

        self._num_frames_spin = QSpinBox()
        self._num_frames_spin.setRange(0, 100000)
        self._num_frames_spin.setSpecialValueText("All")
        acq_form.addRow("Limit frames:", self._num_frames_spin)

        self._pixel_size_spin = QDoubleSpinBox()
        self._pixel_size_spin.setRange(0.0, 10.0)
        self._pixel_size_spin.setDecimals(4)
        self._pixel_size_spin.setSingleStep(0.001)
        self._pixel_size_spin.setSuffix(" µm/px")
        self._pixel_size_spin.setSpecialValueText("Unknown")
        acq_form.addRow("Pixel size:", self._pixel_size_spin)

        root.addWidget(acq_grp)

        # --- Block matching ---
        bm_grp = QGroupBox("Block Matching")
        bm_form = QFormLayout(bm_grp)

        self._mb_spin = QSpinBox()
        self._mb_spin.setRange(4, 128)
        self._mb_spin.setValue(16)
        self._mb_spin.setSingleStep(4)
        bm_form.addRow("Block size (px):", self._mb_spin)

        self._search_spin = QSpinBox()
        self._search_spin.setRange(1, 64)
        self._search_spin.setValue(7)
        bm_form.addRow("Search range (px):", self._search_spin)

        self._scale_spin = QSpinBox()
        self._scale_spin.setRange(0, 4096)
        self._scale_spin.setSpecialValueText("None")
        bm_form.addRow("Resize width (px):", self._scale_spin)

        self._intensity_chk = QCheckBox("Correct intensity")
        bm_form.addRow(self._intensity_chk)

        root.addWidget(bm_grp)

        self._status_lbl = QLabel("")
        self._status_lbl.setWordWrap(True)
        root.addWidget(self._status_lbl)

    # ------------------------------------------------------------------
    # Slots — TIF folder
    # ------------------------------------------------------------------

    def _on_browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select TIF folder")
        if folder:
            self._folder_edit.setText(folder)

    def _on_load_folder(self):
        folder = self._folder_edit.text().strip()
        if not folder:
            self._status_lbl.setText("No folder selected.")
            return
        path = Path(folder)
        if not path.is_dir():
            self._status_lbl.setText("Folder not found.")
            return
        files = scan_tif_folder(path)
        if not files:
            self._status_lbl.setText("No TIF/TIFF files found.")
            return
        self._populate_from_folder(path, files)

    # ------------------------------------------------------------------
    # Slots — video import
    # ------------------------------------------------------------------

    def _on_browse_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select MP4 video", "", "Video files (*.mp4 *.MP4 *.mov *.avi)"
        )
        if path:
            self._vid_edit.setText(path)
            # Pre-fill extraction FPS from acquisition FPS
            self._vid_fps.setValue(self._fps_spin.value())

    def _on_import_video(self):
        mp4 = self._vid_edit.text().strip()
        if not mp4:
            self._status_lbl.setText("No video selected.")
            return
        mp4_path = Path(mp4)
        if not mp4_path.is_file():
            self._status_lbl.setText("Video file not found.")
            return

        self._vid_import_btn.setEnabled(False)
        self._vid_progress.setValue(0)
        self._vid_progress.setVisible(True)
        use_script = self._path_b.isChecked()
        mode = "ab_video_mp42tif.sh" if use_script else "ffmpeg"
        self._status_lbl.setText(f"Extracting frames via {mode}…")

        self._worker = _VideoImportWorker(
            mp4_path, self._vid_fps.value(), use_script
        )
        self._worker.progress.connect(self._vid_progress.setValue)
        self._worker.finished.connect(self._on_import_done)
        self._worker.error.connect(self._on_import_error)
        self._worker.start()

    def _on_import_done(self, out_dir: str):
        self._vid_progress.setVisible(False)
        self._vid_import_btn.setEnabled(True)

        path = Path(out_dir)
        files = scan_tif_folder(path)
        if not files:
            self._status_lbl.setText(f"No TIF files found in {out_dir}")
            return

        # Reflect the output dir in the folder field so the user can see it
        self._folder_edit.setText(str(path))
        self._populate_from_folder(path, files)

    def _on_import_error(self, msg: str):
        self._vid_progress.setVisible(False)
        self._vid_import_btn.setEnabled(True)
        self._status_lbl.setText(f"Import failed:\n{msg}")

    # ------------------------------------------------------------------
    # Shared state population
    # ------------------------------------------------------------------

    def _populate_from_folder(self, path: Path, files: list):
        meta = {}
        try:
            meta = load_bioformats_meta(files[0])
        except Exception:
            pass

        if meta.get("framerate"):
            self._fps_spin.setValue(float(meta["framerate"]))
        if meta.get("pixel_size_um"):
            self._pixel_size_spin.setValue(float(meta["pixel_size_um"]))

        mb    = self._mb_spin.value()
        scale = self._scale_spin.value()
        ref   = load_and_resize(files[0], scale_width=scale, mb_size=mb)

        s = self._state
        s.folder          = path
        s.frames          = files
        s.ref_image       = ref
        s.framerate       = self._fps_spin.value()
        s.delay           = self._delay_spin.value()
        s.pixel_size_um   = self._pixel_size_spin.value()
        s.mb_size         = mb
        s.search_range    = self._search_spin.value()
        s.scale_width     = scale
        s.correct_intensity = self._intensity_chk.isChecked()
        nf = self._num_frames_spin.value()
        s.num_frames      = nf if nf > 0 else len(files)

        try:
            self._viewer.layers["Reference"].data = ref
        except KeyError:
            self._viewer.add_image(ref, name="Reference", colormap="gray")

        self._status_lbl.setText(
            f"Loaded {len(files)} frames from {path.name}\n"
            f"Reference: {ref.shape[1]}×{ref.shape[0]} px"
        )
