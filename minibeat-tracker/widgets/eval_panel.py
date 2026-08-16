"""Step 4 — Evaluate Data dock widget."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox, QPushButton,
    QLabel, QFileDialog, QCheckBox, QDoubleSpinBox, QComboBox,
    QProgressBar,
)
from qtpy.QtCore import Qt, QThread, Signal

from ..state import SessionState
from ..core.analysis import analyze_peaks
from ..io.export import export_beating_data, export_peaks, export_ana_peaks
from ..io.video import export_amplitude_video


# ---------------------------------------------------------------------------
# Background worker for video export
# ---------------------------------------------------------------------------

class _VideoWorker(QThread):
    progress = Signal(int)
    finished = Signal(str)   # path written
    error    = Signal(str)

    def __init__(self, state: SessionState, path: Path,
                 fps: float, colormap: str, alpha: float, vmax: float):
        super().__init__()
        self._state   = state
        self._path    = path
        self._fps     = fps
        self._colormap = colormap
        self._alpha   = alpha
        self._vmax    = vmax

    def run(self):
        try:
            export_amplitude_video(
                frames       = self._state.frames,
                motion_amp   = self._state.motion_amp,
                output_path  = self._path,
                fps          = self._fps,
                colormap     = self._colormap,
                alpha        = self._alpha,
                vmax         = self._vmax,
                scale_width  = self._state.scale_width,
                mb_size      = self._state.mb_size,
                progress_callback = lambda v: self.progress.emit(v),
            )
            self.finished.emit(str(self._path))
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

class EvalPanel(QWidget):
    """Napari dock widget — mirrors MotionGUI Step 4 (Evaluate Data)."""

    def __init__(self, viewer, state: SessionState, plot_panel=None, parent=None):
        super().__init__(parent)
        self._viewer     = viewer
        self._state      = state
        self._plot_panel = plot_panel
        self._ana: dict | None = None
        self._out_dir: Path | None = None
        self._worker: _VideoWorker | None = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setAlignment(Qt.AlignTop)

        # --- Peak analysis ---
        opt_grp = QGroupBox("Peak Analysis")
        oform = QFormLayout(opt_grp)
        self._skip_first_chk = QCheckBox("Skip first peak")
        self._skip_first_chk.setChecked(True)
        oform.addRow(self._skip_first_chk)
        root.addWidget(opt_grp)

        analyse_btn = QPushButton("Analyse Peaks")
        analyse_btn.clicked.connect(self._on_analyse)
        root.addWidget(analyse_btn)

        self._results_lbl = QLabel("")
        self._results_lbl.setWordWrap(True)
        root.addWidget(self._results_lbl)

        # --- CSV export ---
        exp_grp = QGroupBox("Export CSV")
        elay = QVBoxLayout(exp_grp)

        self._export_dir_lbl = QLabel("Output dir: same as image folder")
        elay.addWidget(self._export_dir_lbl)

        choose_dir_btn = QPushButton("Choose output directory…")
        choose_dir_btn.clicked.connect(self._on_choose_dir)
        elay.addWidget(choose_dir_btn)

        self._export_beating_chk = QCheckBox("Beating data (_BeatingData.csv)")
        self._export_beating_chk.setChecked(True)
        elay.addWidget(self._export_beating_chk)

        self._export_peaks_chk = QCheckBox("Raw peaks (_RawPeaks.csv)")
        self._export_peaks_chk.setChecked(True)
        elay.addWidget(self._export_peaks_chk)

        self._export_ana_chk = QCheckBox("Analysed peaks (_AnaPeaks.csv)")
        self._export_ana_chk.setChecked(True)
        elay.addWidget(self._export_ana_chk)

        export_csv_btn = QPushButton("Export CSV")
        export_csv_btn.clicked.connect(self._on_export_csv)
        elay.addWidget(export_csv_btn)

        root.addWidget(exp_grp)

        # --- Video export ---
        vid_grp = QGroupBox("Export Amplitude Video")
        vform = QFormLayout(vid_grp)

        self._vid_fps = QDoubleSpinBox()
        self._vid_fps.setRange(1.0, 120.0)
        self._vid_fps.setValue(14.0)
        self._vid_fps.setSuffix(" fps")
        vform.addRow("Output FPS:", self._vid_fps)

        self._vid_alpha = QDoubleSpinBox()
        self._vid_alpha.setRange(0.1, 1.0)
        self._vid_alpha.setSingleStep(0.05)
        self._vid_alpha.setValue(0.6)
        vform.addRow("Heatmap opacity:", self._vid_alpha)

        self._vid_vmax = QDoubleSpinBox()
        self._vid_vmax.setRange(0.0, 1e6)
        self._vid_vmax.setValue(0.0)
        self._vid_vmax.setSpecialValueText("Auto")
        vform.addRow("Amplitude max:", self._vid_vmax)

        self._vid_cmap = QComboBox()
        for cm in ["hot", "inferno", "magma", "plasma", "jet", "turbo"]:
            self._vid_cmap.addItem(cm)
        vform.addRow("Colormap:", self._vid_cmap)

        export_vid_btn = QPushButton("Export Video…")
        export_vid_btn.clicked.connect(self._on_export_video)
        vform.addRow(export_vid_btn)

        self._vid_progress = QProgressBar()
        self._vid_progress.setRange(0, 100)
        self._vid_progress.setVisible(False)
        vform.addRow(self._vid_progress)

        root.addWidget(vid_grp)

        self._status_lbl = QLabel("")
        self._status_lbl.setWordWrap(True)
        root.addWidget(self._status_lbl)

    # ------------------------------------------------------------------
    # Slots — peak analysis
    # ------------------------------------------------------------------

    def _on_analyse(self):
        s = self._state
        if not s.has_contraction_data():
            self._results_lbl.setText("Run Step 3 first.")
            return

        self._ana = analyze_peaks(
            s.peak_times,
            s.peak_heights,
            skip_first=self._skip_first_chk.isChecked(),
        )

        unit = "µm/s" if s.pixel_size_um > 0 else "px/s"
        txt = (
            f"Beat rate:       {self._ana['mean_beat_rate']:.1f} ± {self._ana['std_beat_rate']:.1f} BPM\n"
            f"Contraction vel: {self._ana['mean_contract']:.3f} ± {self._ana['std_contract']:.3f} {unit}\n"
            f"Relaxation vel:  {self._ana['mean_relax']:.3f} ± {self._ana['std_relax']:.3f} {unit}\n"
            f"Time int:        {self._ana['mean_time_int']:.3f} ± {self._ana['std_time_int']:.3f} s\n"
            f"Cycles detected: {len(self._ana['contract_heights'])}"
        )
        self._results_lbl.setText(txt)

        if self._plot_panel is not None:
            self._plot_panel.refresh_cycles(self._state, self._ana)

    # ------------------------------------------------------------------
    # Slots — CSV export
    # ------------------------------------------------------------------

    def _on_choose_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select output directory")
        if d:
            self._out_dir = Path(d)
            self._export_dir_lbl.setText(f"Output dir: {d}")

    def _on_export_csv(self):
        s = self._state
        if not s.has_contraction_data():
            self._status_lbl.setText("Run Steps 3 & 4 first.")
            return

        out  = self._out_dir or (s.folder if s.folder else Path("."))
        stem = s.folder.name if s.folder else "cardio"
        unit = "µm/s" if s.pixel_size_um > 0 else "px/s"
        saved: list[str] = []

        if self._export_beating_chk.isChecked():
            p = out / f"{stem}_BeatingData.csv"
            export_beating_data(p, s.raw_results, unit)
            saved.append(p.name)

        if self._export_peaks_chk.isChecked() and s.peak_times is not None:
            p = out / f"{stem}_RawPeaks.csv"
            export_peaks(p, s.peak_times, s.peak_heights, unit)
            saved.append(p.name)

        if self._export_ana_chk.isChecked() and self._ana is not None:
            p1 = out / f"{stem}_AnaPeaks.csv"
            p2 = out / f"{stem}_AnaPeaksMean.csv"
            export_ana_peaks(
                p1, p2,
                self._ana["contract_heights"],
                self._ana["relax_heights"],
                self._ana["peaks_int_time_diff"],
                self._ana["peaks_time_diff"],
                unit,
            )
            saved += [p1.name, p2.name]

        self._status_lbl.setText(
            "Saved:\n" + "\n".join(saved) if saved
            else "Nothing exported — check checkboxes."
        )

    # ------------------------------------------------------------------
    # Slots — video export
    # ------------------------------------------------------------------

    def _on_export_video(self):
        s = self._state
        if s.motion_amp is None:
            self._status_lbl.setText("Produce vectors first (Step 2).")
            return
        if not s.frames:
            self._status_lbl.setText("Load a folder first (Step 1).")
            return

        stem = s.folder.name if s.folder else "cardio"
        default_path = str((self._out_dir or s.folder or Path(".")) /
                           f"{stem}_amplitude.mp4")
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save amplitude video", default_path,
            "MP4 video (*.mp4)"
        )
        if not out_path:
            return

        self._vid_progress.setValue(0)
        self._vid_progress.setVisible(True)
        self._status_lbl.setText("Rendering video…")

        self._worker = _VideoWorker(
            state    = s,
            path     = Path(out_path),
            fps      = self._vid_fps.value(),
            colormap = self._vid_cmap.currentText(),
            alpha    = self._vid_alpha.value(),
            vmax     = self._vid_vmax.value(),
        )
        self._worker.progress.connect(self._vid_progress.setValue)
        self._worker.finished.connect(self._on_video_done)
        self._worker.error.connect(self._on_video_error)
        self._worker.start()

    def _on_video_done(self, path: str):
        self._vid_progress.setVisible(False)
        self._status_lbl.setText(f"Video saved:\n{path}")

    def _on_video_error(self, msg: str):
        self._vid_progress.setVisible(False)
        self._status_lbl.setText(f"Video export failed:\n{msg}")
