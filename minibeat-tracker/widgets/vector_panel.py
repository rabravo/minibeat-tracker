"""Step 2 — Produce Vectors dock widget."""
from __future__ import annotations

import threading
import numpy as np
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox, QPushButton,
    QLabel, QProgressBar, QSpinBox, QDoubleSpinBox, QCheckBox,
    QComboBox,
)
from qtpy.QtCore import Qt, QThread, Signal

from ..state import SessionState
from ..core.motion import get_motion_vectors
from ..core.cleaning import clean_neighbor, clean_fft, clean_variation


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class _MotionWorker(QThread):
    progress = Signal(int)
    finished = Signal(object, object, object, object)  # vect, amp, centers, edges
    error = Signal(str)

    def __init__(self, state: SessionState, n_jobs: int):
        super().__init__()
        self._s = state
        self._n_jobs = n_jobs

    def run(self):
        try:
            vect, amp, centers, edges = get_motion_vectors(
                self._s.frames,
                self._s.ref_image,
                mb_size=self._s.mb_size,
                search_range=self._s.search_range,
                num_frames=self._s.num_frames,
                delay=self._s.delay,
                scale_width=self._s.scale_width,
                correct_intensity=self._s.correct_intensity,
                n_jobs=self._n_jobs,
                progress_callback=lambda v: self.progress.emit(v),
            )
            self.finished.emit(vect, amp, centers, edges)
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class VectorPanel(QWidget):
    """Napari dock widget — mirrors MotionGUI Step 2 (Produce Vectors)."""

    def __init__(self, viewer, state: SessionState, parent=None):
        super().__init__(parent)
        self._viewer = viewer
        self._state = state
        self._worker: _MotionWorker | None = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setAlignment(Qt.AlignTop)

        # --- Cleaning options ---
        clean_grp = QGroupBox("Post-processing / Cleaning")
        cform = QFormLayout(clean_grp)

        self._clean_neighbor_chk = QCheckBox("Neighbour deviation filter")
        self._clean_neighbor_chk.setChecked(True)
        cform.addRow(self._clean_neighbor_chk)

        self._neighbor_thresh = QDoubleSpinBox()
        self._neighbor_thresh.setRange(0.5, 10.0)
        self._neighbor_thresh.setValue(2.0)
        cform.addRow("  Threshold (×std):", self._neighbor_thresh)

        self._clean_fft_chk = QCheckBox("FFT temporal filter")
        cform.addRow(self._clean_fft_chk)

        self._fft_thresh = QDoubleSpinBox()
        self._fft_thresh.setRange(0.5, 20.0)
        self._fft_thresh.setValue(4.0)
        cform.addRow("  Freq cutoff (× fund.):", self._fft_thresh)

        self._clean_var_chk = QCheckBox("Variation filter (CoV)")
        cform.addRow(self._clean_var_chk)

        self._cov_thresh = QDoubleSpinBox()
        self._cov_thresh.setRange(0.1, 5.0)
        self._cov_thresh.setValue(0.8)
        cform.addRow("  CoV threshold:", self._cov_thresh)

        root.addWidget(clean_grp)

        # --- Parallelism ---
        par_grp = QGroupBox("Computation")
        pform = QFormLayout(par_grp)

        self._njobs_spin = QSpinBox()
        self._njobs_spin.setRange(-1, 64)
        self._njobs_spin.setValue(-1)
        self._njobs_spin.setSpecialValueText("All CPUs")
        pform.addRow("n_jobs:", self._njobs_spin)

        root.addWidget(par_grp)

        # --- Rotation ---
        rot_grp = QGroupBox("Rotation")
        rform = QFormLayout(rot_grp)
        self._rotation_spin = QDoubleSpinBox()
        self._rotation_spin.setRange(-360.0, 360.0)
        self._rotation_spin.setValue(0.0)
        self._rotation_spin.setSuffix(" °")
        rform.addRow("Rotate vectors:", self._rotation_spin)
        root.addWidget(rot_grp)

        # --- Run button ---
        self._run_btn = QPushButton("Produce Vectors")
        self._run_btn.setDefault(True)
        self._run_btn.clicked.connect(self._on_run)
        root.addWidget(self._run_btn)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setVisible(False)
        root.addWidget(self._progress)

        self._status_lbl = QLabel("")
        self._status_lbl.setWordWrap(True)
        root.addWidget(self._status_lbl)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_run(self):
        if not self._state.frames:
            self._status_lbl.setText("Load a folder first (Step 1).")
            return

        self._run_btn.setEnabled(False)
        self._progress.setValue(0)
        self._progress.setVisible(True)
        self._status_lbl.setText("Computing motion vectors…")

        self._worker = _MotionWorker(self._state, self._njobs_spin.value())
        self._worker.progress.connect(self._progress.setValue)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_finished(self, vect, amp, centers, edges):
        self._progress.setVisible(False)
        self._run_btn.setEnabled(True)

        # Apply cleaning if requested
        if self._clean_neighbor_chk.isChecked():
            vect, amp = clean_neighbor(vect, amp, threshold=self._neighbor_thresh.value())
        if self._clean_fft_chk.isChecked():
            vect, amp = clean_fft(vect, self._state.framerate, freq_thresh=self._fft_thresh.value())
        if self._clean_var_chk.isChecked():
            vect, amp = clean_variation(vect, amp, cov_threshold=self._cov_thresh.value())

        # Apply rotation
        angle = self._rotation_spin.value()
        if angle != 0.0:
            from ..core.motion import rotate_motion_vect
            vect = rotate_motion_vect(vect, angle)
            amp = np.linalg.norm(vect, axis=2)

        self._state.raw_motion_vect = vect
        self._state.raw_motion_amp = amp
        self._state.motion_vect = vect
        self._state.motion_amp = amp
        self._state.box_center = centers
        self._state.edge_detection = edges

        # Visualise mean amplitude in napari — scale so each block covers mb_size pixels
        mean_amp = amp.mean(axis=2)
        mb = self._state.mb_size
        block_scale = (mb, mb)
        try:
            layer = self._viewer.layers["Mean Amplitude"]
            layer.data  = mean_amp
            layer.scale = block_scale
        except KeyError:
            self._viewer.add_image(mean_amp, name="Mean Amplitude",
                                   colormap="inferno", opacity=0.7,
                                   scale=block_scale)

        R, C, T = amp.shape
        self._status_lbl.setText(
            f"Done — grid {R}×{C}, {T} frames processed."
        )

    def _on_error(self, msg: str):
        self._progress.setVisible(False)
        self._run_btn.setEnabled(True)
        self._status_lbl.setText(f"Error: {msg}")
