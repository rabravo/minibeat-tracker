"""Launch napari viewer with MiniBeat dock widgets."""
from __future__ import annotations

import napari
from qtpy.QtWidgets import QTabWidget, QApplication, QAction
from qtpy.QtCore import Qt
from joblib.externals.loky import get_reusable_executor

from .state import SessionState
from .widgets import FolderPanel, VectorPanel, ContractionPanel, EvalPanel, PlotPanel


def _on_float(dock, floating: bool) -> None:
    """Promote a floated dock to a real peer window so it can go behind napari."""
    if floating:
        dock.setWindowFlags(
            Qt.Window |
            Qt.WindowTitleHint |
            Qt.WindowMinMaxButtonsHint |
            Qt.WindowCloseButtonHint
        )
        dock.show()   # flag change hides the widget — must re-show


def run():
    viewer = napari.Viewer(title="MiniBeat")
    state  = SessionState()

    # --- Fit window to 90% of available screen ---
    screen = QApplication.primaryScreen().availableGeometry()
    viewer.window._qt_window.resize(
        int(screen.width()  * 0.9),
        int(screen.height() * 0.9),
    )

    # --- Widgets ---
    plots = PlotPanel(viewer, state)
    step1 = FolderPanel(viewer, state)
    step2 = VectorPanel(viewer, state)
    step3 = ContractionPanel(viewer, state, plot_panel=plots)
    step4 = EvalPanel(viewer, state, plot_panel=plots)

    # --- Group the four step panels into one tabbed dock ---
    steps_tabs = QTabWidget()
    steps_tabs.addTab(step1, "1 · Folder")
    steps_tabs.addTab(step2, "2 · Vectors")
    steps_tabs.addTab(step3, "3 · Contraction")
    steps_tabs.addTab(step4, "4 · Evaluate")

    _d1 = viewer.window.add_dock_widget(steps_tabs, name="MiniBeat", area="right")
    _d2 = viewer.window.add_dock_widget(plots,      name="Plots",         area="right")

    # When a dock is floated, Qt defaults to Qt.Tool which stays on top of the
    # parent window and can never go behind it. Switching to Qt.Window makes the
    # floated panel a true peer window that can be freely raised or lowered.
    for dock in (_d1, _d2):
        dock.topLevelChanged.connect(lambda floating, d=dock: _on_float(d, floating))

    # --- Panels menu: lets users reopen closed panels ---
    menubar = viewer.window._qt_window.menuBar()
    panels_menu = menubar.addMenu("Panels")

    def _make_toggle(dock, label: str):
        act = QAction(label, viewer.window._qt_window, checkable=True, checked=True)
        act.triggered.connect(lambda chk: dock.show() if chk else dock.hide())
        dock.visibilityChanged.connect(act.setChecked)
        panels_menu.addAction(act)

    _make_toggle(_d1, "MiniBeat")
    _make_toggle(_d2, "Plots")

    napari.run()

    # Clean up joblib worker pool to avoid semaphore leak warnings on exit
    try:
        get_reusable_executor().shutdown(wait=False)
    except Exception:
        pass
