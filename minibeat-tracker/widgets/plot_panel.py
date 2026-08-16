"""Plot panel — MATLAB-style dark plots for signal and cycle analysis."""
from __future__ import annotations

import numpy as np
from qtpy.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QLabel, QSizePolicy
from qtpy.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

from ..state import SessionState

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
BG        = "#000000"
AX_BG     = "#000000"
LINE      = "#4DA6FF"
CONTRACT  = "#DD3333"
RELAX     = "#4466FF"
TEXT      = "#FFFFFF"
GRID      = "#333333"
MK_SIZE_S = 28


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dark_fig(nrows: int, ncols: int, figsize: tuple) -> tuple[Figure, list]:
    # constrained_layout avoids the tight_layout crash inside napari dock widgets
    fig = Figure(figsize=figsize, facecolor=BG, constrained_layout=True)
    axes = [fig.add_subplot(nrows, ncols, i + 1) for i in range(nrows * ncols)]
    for ax in axes:
        _dark_ax(ax)
    return fig, axes


def _dark_ax(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    ax.set_facecolor(AX_BG)
    for spine in ax.spines.values():
        spine.set_color(TEXT)
    ax.tick_params(colors=TEXT, labelsize=7)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    ax.grid(True, color=GRID, linewidth=0.4, linestyle="-")
    if title:
        ax.set_title(title, fontsize=9, fontweight="bold", color=TEXT)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)


def _tab_page(fig: Figure, parent: QWidget) -> tuple[QWidget, FigureCanvasQTAgg]:
    """Wrap figure + toolbar in a plain QWidget so napari doesn't see the Figure."""
    canvas = FigureCanvasQTAgg(fig)
    canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    toolbar = NavigationToolbar2QT(canvas, parent)

    # Outer container hides the Figure from napari's dock-widget introspection
    outer = QWidget()
    outer.setObjectName("mpl_container")
    lay = QVBoxLayout(outer)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(0)
    lay.addWidget(toolbar)
    lay.addWidget(canvas)
    return outer, canvas


# ---------------------------------------------------------------------------
# Plot panel
# ---------------------------------------------------------------------------

class PlotPanel(QWidget):
    """Signal, Cycles, and Contraction Map tabs."""

    def __init__(self, viewer, state: SessionState, parent=None):
        super().__init__(parent)
        self._viewer = viewer
        self._state  = state
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(2, 2, 2, 2)
        root.setSpacing(2)

        tabs = QTabWidget()
        root.addWidget(tabs)

        # Tab 1 — Signal
        self._sig_fig, _axes = _dark_fig(1, 1, (6, 3))
        self._sig_ax = _axes[0]
        _dark_ax(self._sig_ax, title="Abs Velocity",
                 xlabel="Time (sec)", ylabel="Beating velocity / (pixel/sec)")
        sig_w, self._sig_canvas = _tab_page(self._sig_fig, self)
        tabs.addTab(sig_w, "Signal")

        # Tab 2 — Cycles
        self._cyc_fig, _axes2 = _dark_fig(1, 2, (7, 3.5))
        self._cyc_ax_h, self._cyc_ax_t = _axes2
        _dark_ax(self._cyc_ax_h, title="Peak Height",
                 xlabel="Peak #", ylabel="Contraction / (pixel/sec)")
        _dark_ax(self._cyc_ax_t, title="Peak Time Diff",
                 xlabel="Peak #", ylabel="Time (sec)")
        cyc_w, self._cyc_canvas = _tab_page(self._cyc_fig, self)
        tabs.addTab(cyc_w, "Cycles")

        # Tab 3 — Contraction Map
        self._map_fig, _axes3 = _dark_fig(1, 1, (5, 4))
        self._map_ax = _axes3[0]
        _dark_ax(self._map_ax, title="Mean Contraction Amplitude",
                 xlabel="Block col", ylabel="Block row")
        map_w, self._map_canvas = _tab_page(self._map_fig, self)
        tabs.addTab(map_w, "Contraction Map")

        self._tabs = tabs

        self._status = QLabel("Run Steps 3 & 4 to populate plots.")
        self._status.setAlignment(Qt.AlignCenter)
        self._status.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        root.addWidget(self._status)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh_signal(self, state: SessionState):
        if state.raw_results is None:
            return
        unit = "µm/s" if state.pixel_size_um > 0 else "pixel/sec"
        t, amp = state.raw_results[:, 0], state.raw_results[:, 1]

        ax = self._sig_ax
        ax.cla()
        _dark_ax(ax, title="Abs Velocity",
                 xlabel="Time (sec)", ylabel=f"Beating velocity / ({unit})")
        ax.plot(t, amp, color=LINE, linewidth=0.9)

        if state.peak_times is not None and len(state.peak_times):
            pt, ph = state.peak_times, state.peak_heights
            ax.scatter(pt[0::2], ph[0::2], color=CONTRACT, marker="^",
                       s=MK_SIZE_S, zorder=5, label="Contraction")
            ax.scatter(pt[1::2], ph[1::2], color=RELAX, marker="o",
                       s=MK_SIZE_S, zorder=5, label="Relaxation")
            ax.legend(fontsize=7, facecolor="#111111",
                      edgecolor="#444444", labelcolor=TEXT)

        self._sig_canvas.draw_idle()
        self._refresh_map(state)
        self._tabs.setCurrentIndex(0)
        self._status.setText(
            f"Signal: {len(t)} frames  ·  "
            f"Peaks: {len(state.peak_times) if state.peak_times is not None else 0}"
        )

    def refresh_cycles(self, state: SessionState, ana: dict):
        unit = "µm/s" if state.pixel_size_um > 0 else "pixel/sec"
        n    = len(ana["contract_heights"])

        if "contract_times" in ana and state.raw_results is not None:
            self._redraw_signal(state, ana, unit)

        # Peak Height
        ax = self._cyc_ax_h
        ax.cla()
        _dark_ax(ax, title="Peak Height",
                 xlabel="Peak #", ylabel=f"Contraction / ({unit})")
        x = np.arange(n)
        ax.scatter(x, ana["contract_heights"], color=CONTRACT, marker="^",
                   s=MK_SIZE_S, label="Contraction")
        ax.scatter(x, ana["relax_heights"], color=RELAX, marker="o",
                   s=MK_SIZE_S, label="Relaxation")
        ax.legend(fontsize=7, facecolor="#111111",
                  edgecolor="#444444", labelcolor=TEXT)

        # Peak Time Diff
        ax = self._cyc_ax_t
        ax.cla()
        _dark_ax(ax, title="Peak Time Diff",
                 xlabel="Peak #", ylabel="Time (sec)")
        ctd, rtc = ana["peaks_int_time_diff"], ana["relax_to_contract_diff"]
        ax.scatter(np.arange(len(ctd)), ctd, color=CONTRACT, marker="^",
                   s=MK_SIZE_S, label="Contract→Relax")
        ax.scatter(np.arange(len(rtc)), rtc, color=RELAX, marker="o",
                   s=MK_SIZE_S, label="Relax→Contract")
        ax.legend(fontsize=7, facecolor="#111111",
                  edgecolor="#444444", labelcolor=TEXT)

        self._cyc_canvas.draw_idle()
        self._tabs.setCurrentIndex(1)
        self._status.setText(
            f"Cycles: {n}  ·  "
            f"Beat rate: {ana['mean_beat_rate']:.1f} ± {ana['std_beat_rate']:.1f} BPM"
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _redraw_signal(self, state, ana, unit):
        t, amp = state.raw_results[:, 0], state.raw_results[:, 1]
        ax = self._sig_ax
        ax.cla()
        _dark_ax(ax, title="Abs Velocity",
                 xlabel="Time (sec)", ylabel=f"Beating velocity / ({unit})")
        ax.plot(t, amp, color=LINE, linewidth=0.9)
        ax.scatter(ana["contract_times"], ana["contract_heights"],
                   color=CONTRACT, marker="^", s=MK_SIZE_S, zorder=5,
                   label="Contraction")
        ax.scatter(ana["relax_times"], ana["relax_heights"],
                   color=RELAX, marker="o", s=MK_SIZE_S, zorder=5,
                   label="Relaxation")
        ax.legend(fontsize=7, facecolor="#111111",
                  edgecolor="#444444", labelcolor=TEXT)
        self._sig_canvas.draw_idle()

    def _refresh_map(self, state: SessionState):
        if state.contraction_maps is None:
            return
        ax = self._map_ax
        ax.cla()
        _dark_ax(ax, title="Mean Contraction Amplitude",
                 xlabel="Block col", ylabel="Block row")
        im = ax.imshow(state.contraction_maps["mean_absolute"],
                       cmap="hot", aspect="auto", origin="upper")
        cb = self._map_fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cb.ax.yaxis.set_tick_params(color=TEXT, labelcolor=TEXT)
        cb.outline.set_edgecolor(TEXT)
        self._map_canvas.draw_idle()
