"""Step 3 — Get Contraction Data dock widget."""
from __future__ import annotations

import numpy as np
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox, QPushButton,
    QLabel, QDoubleSpinBox, QCheckBox,
)
from qtpy.QtCore import Qt

from ..state import SessionState
from ..core.analysis import get_contraction_data, calc_mean_contraction, do_autocorr, detect_peaks


class ContractionPanel(QWidget):
    """Napari dock widget — mirrors MotionGUI Step 3 (Get Contraction Data)."""

    def __init__(self, viewer, state: SessionState, plot_panel=None, parent=None):
        super().__init__(parent)
        self._viewer = viewer
        self._state = state
        self._plot_panel = plot_panel
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setAlignment(Qt.AlignTop)

        # --- Contraction parameters ---
        param_grp = QGroupBox("Contraction Parameters")
        pform = QFormLayout(param_grp)

        self._min_bright_spin = QDoubleSpinBox()
        self._min_bright_spin.setRange(0.0, 100.0)
        self._min_bright_spin.setValue(0.0)
        self._min_bright_spin.setSuffix(" % max")
        pform.addRow("Min brightness:", self._min_bright_spin)

        self._peak_min_h_spin = QDoubleSpinBox()
        self._peak_min_h_spin.setRange(0.0, 1e6)
        self._peak_min_h_spin.setValue(0.0)
        self._peak_min_h_spin.setSpecialValueText("Auto")
        pform.addRow("Min peak height:", self._peak_min_h_spin)

        self._peak_min_sep_spin = QDoubleSpinBox()
        self._peak_min_sep_spin.setRange(0.0, 60.0)
        self._peak_min_sep_spin.setValue(0.0)
        self._peak_min_sep_spin.setSuffix(" sec")
        self._peak_min_sep_spin.setSpecialValueText("Auto")
        pform.addRow("Min peak separation:", self._peak_min_sep_spin)

        root.addWidget(param_grp)

        # --- Autocorr beat rate ---
        auto_grp = QGroupBox("Beat Rate Estimation")
        aform = QFormLayout(auto_grp)

        self._min_bpm_spin = QDoubleSpinBox()
        self._min_bpm_spin.setRange(1.0, 300.0)
        self._min_bpm_spin.setValue(10.0)
        aform.addRow("Min BPM:", self._min_bpm_spin)

        self._max_bpm_spin = QDoubleSpinBox()
        self._max_bpm_spin.setRange(1.0, 600.0)
        self._max_bpm_spin.setValue(300.0)
        aform.addRow("Max BPM:", self._max_bpm_spin)

        root.addWidget(auto_grp)

        # --- Run ---
        run_btn = QPushButton("Get Contraction Data")
        run_btn.setDefault(True)
        run_btn.clicked.connect(self._on_run)
        root.addWidget(run_btn)

        self._status_lbl = QLabel("")
        self._status_lbl.setWordWrap(True)
        root.addWidget(self._status_lbl)

    def _on_run(self):
        s = self._state
        if not s.has_vectors():
            self._status_lbl.setText("Produce vectors first (Step 2).")
            return

        raw_results, legend = get_contraction_data(
            s.motion_vect,
            s.motion_amp,
            delay=s.delay,
            framerate=s.framerate,
            pixel_size=s.pixel_size_um,
            mask=s.raw_mask,
        )
        s.raw_results = raw_results

        # Spatial maps
        contraction_maps = calc_mean_contraction(s.motion_amp, s.motion_vect)
        s.contraction_maps = contraction_maps

        # Autocorr beat rate
        signal = raw_results[:, 1]
        beat_rate, corr = do_autocorr(
            signal,
            s.framerate,
            min_beat_bpm=self._min_bpm_spin.value(),
            max_beat_bpm=self._max_bpm_spin.value(),
        )

        # Peak detection
        peak_times, peak_heights = detect_peaks(
            raw_results,
            s.framerate,
            min_peak_height=self._peak_min_h_spin.value(),
            min_peak_sep_sec=self._peak_min_sep_spin.value(),
        )
        s.peak_times = peak_times
        s.peak_heights = peak_heights

        # Show mean-amplitude spatial map in napari — scale to original image pixels
        mean_abs = contraction_maps["mean_absolute"]
        mb = s.mb_size
        block_scale = (mb, mb)
        try:
            layer = self._viewer.layers["Contraction Map"]
            layer.data  = mean_abs
            layer.scale = block_scale
        except KeyError:
            self._viewer.add_image(
                mean_abs, name="Contraction Map", colormap="magma", opacity=0.7,
                scale=block_scale
            )

        if self._plot_panel is not None:
            self._plot_panel.refresh_signal(s)

        unit = "µm/s" if s.pixel_size_um > 0 else "px/s"
        self._status_lbl.setText(
            f"Contraction data ready.\n"
            f"AutoCorr beat rate: {beat_rate:.1f} BPM (r={corr:.3f})\n"
            f"Peaks detected: {len(peak_times)}\n"
            f"Active area: {contraction_maps['areas'] * 100:.1f}%\n"
            f"Mean contraction: {raw_results[:, 1].mean():.3f} {unit}"
        )
