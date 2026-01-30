#!/usr/bin/env python3
import math
import sys
from pathlib import Path

import numpy as np

try:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import (
        QApplication,
        QAbstractItemView,
        QFileDialog,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QComboBox,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )
    from PyQt5.QtGui import QIcon
    QT_LIB = 5
except ImportError:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QApplication,
        QAbstractItemView,
        QFileDialog,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QComboBox,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QPushButton,
        QVBoxLayout,
        QWidget,
    )
    from PyQt6.QtGui import QIcon
    QT_LIB = 6

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from graphplot.simmatrix import SimMatrixPlot

DEFAULT_MATRIX_KEY = "matrix_pop_avg"
DEFAULT_COLORMAP = "plasma"
COLORMAPS = [
    "plasma",
    "viridis",
    "magma",
    "inferno",
    "cividis",
    "turbo",
    "jet",
]

if QT_LIB == 6:
    Qt.Horizontal = Qt.Orientation.Horizontal
    Qt.Vertical = Qt.Orientation.Vertical


def _user_role():
    return getattr(Qt, "UserRole", getattr(Qt.ItemDataRole, "UserRole"))


USER_ROLE = _user_role()


class ConnectomeViewer(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.files = []
        self.titles = {}
        self.setWindowTitle("Connectome Viewer")
        self._set_window_icon()
        self.setAcceptDrops(True)
        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget(self)
        main_layout = QHBoxLayout(central)

        controls_layout = QVBoxLayout()

        header = QLabel("Connectome matrices (.npz)")
        controls_layout.addWidget(header)

        self.add_button = QPushButton("Add Files")
        self.add_button.clicked.connect(self._open_files)
        controls_layout.addWidget(self.add_button)

        self.remove_button = QPushButton("Remove Selected")
        self.remove_button.clicked.connect(self._remove_selected)
        controls_layout.addWidget(self.remove_button)

        self.clear_button = QPushButton("Clear List")
        self.clear_button.clicked.connect(self._clear_files)
        controls_layout.addWidget(self.clear_button)

        self.export_button = QPushButton("Export Grid")
        self.export_button.clicked.connect(self._export_grid)
        controls_layout.addWidget(self.export_button)

        key_label = QLabel("Matrix key:")
        controls_layout.addWidget(key_label)

        self.key_edit = QLineEdit(DEFAULT_MATRIX_KEY)
        self.key_edit.setPlaceholderText(DEFAULT_MATRIX_KEY)
        self.key_edit.editingFinished.connect(self._plot_selected)
        self.key_edit.returnPressed.connect(self._plot_selected)
        controls_layout.addWidget(self.key_edit)

        title_label = QLabel("Plot title:")
        controls_layout.addWidget(title_label)

        self.title_edit = QLineEdit("")
        self.title_edit.setPlaceholderText("Defaults to file name")
        self.title_edit.editingFinished.connect(self._plot_selected)
        self.title_edit.returnPressed.connect(self._plot_selected)
        controls_layout.addWidget(self.title_edit)

        cmap_label = QLabel("Colormap:")
        controls_layout.addWidget(cmap_label)

        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(COLORMAPS)
        self.cmap_combo.setCurrentText(DEFAULT_COLORMAP)
        self.cmap_combo.currentIndexChanged.connect(self._plot_selected)
        controls_layout.addWidget(self.cmap_combo)

        self.file_list = QListWidget()
        if QT_LIB == 6:
            self.file_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        else:
            self.file_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.file_list.currentItemChanged.connect(self._on_selection_changed)
        controls_layout.addWidget(self.file_list)

        hint = QLabel("Drag & drop .npz files here.")
        hint.setWordWrap(True)
        controls_layout.addWidget(hint)

        controls_layout.addStretch(1)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        main_layout.addLayout(controls_layout, 0)
        main_layout.addWidget(self.canvas, 1)

        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready.")

    def _set_window_icon(self) -> None:
        icon_path = Path(__file__).with_name("icons") / "conviewer.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

    def _open_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select .npz files",
            "",
            "NumPy archives (*.npz);;All files (*)",
        )
        self._add_files(paths)

    def _add_files(self, paths) -> None:
        added_any = False
        for raw_path in paths:
            if not raw_path:
                continue
            path = Path(raw_path)
            if path.suffix.lower() != ".npz" or not path.exists():
                continue
            if path in self.files:
                continue
            self.files.append(path)
            item = QListWidgetItem(path.name)
            item.setToolTip(str(path))
            item.setData(USER_ROLE, str(path))
            self.file_list.addItem(item)
            added_any = True

        if added_any and self.file_list.currentItem() is None:
            self.file_list.setCurrentRow(self.file_list.count() - 1)

        if not added_any:
            self.statusBar().showMessage("No valid .npz files added.")

    def _remove_selected(self) -> None:
        item = self.file_list.currentItem()
        if item is None:
            return
        path = Path(item.data(USER_ROLE))
        if path in self.files:
            self.files.remove(path)
        row = self.file_list.row(item)
        self.file_list.takeItem(row)
        if self.file_list.count() == 0:
            self._clear_plot()

    def _clear_files(self) -> None:
        self.files.clear()
        self.file_list.clear()
        self._clear_plot()
        self.statusBar().showMessage("File list cleared.")

    def _current_matrix_key(self) -> str:
        key = self.key_edit.text().strip()
        if not key:
            key = DEFAULT_MATRIX_KEY
            self.key_edit.setText(key)
        return key

    def _export_grid(self) -> None:
        if not self.files:
            self.statusBar().showMessage("No files to export.")
            return
        save_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export connectome grid",
            "",
            "PDF (*.pdf);;SVG (*.svg);;PNG (*.png)",
        )
        if not save_path:
            return

        output_path = Path(save_path)
        if output_path.suffix.lower() not in {".pdf", ".svg", ".png"}:
            if "PDF" in selected_filter:
                output_path = output_path.with_suffix(".pdf")
            elif "SVG" in selected_filter:
                output_path = output_path.with_suffix(".svg")
            else:
                output_path = output_path.with_suffix(".png")

        key = self._current_matrix_key()
        colormap = self.cmap_combo.currentText() or DEFAULT_COLORMAP
        matrices = []
        titles = []
        skipped = []
        for path in self.files:
            try:
                with np.load(path) as npz:
                    if key not in npz:
                        raise KeyError(f"Key '{key}' not found")
                    matrix = npz[key]
            except Exception as exc:
                skipped.append(f"{path.name} ({exc})")
                continue
            matrices.append(matrix)
            titles.append(self.titles.get(path, path.name))

        if not matrices:
            self.statusBar().showMessage("No matrices exported (missing keys or load errors).")
            return

        cols = min(4, len(matrices))
        rows = int(math.ceil(len(matrices) / cols))
        export_figure = Figure(figsize=(4 * cols, 4 * rows))
        axes = export_figure.subplots(rows, cols)
        if isinstance(axes, np.ndarray):
            flat_axes = axes.flatten()
        else:
            flat_axes = [axes]

        for idx, (matrix, title) in enumerate(zip(matrices, titles)):
            ax = flat_axes[idx]
            SimMatrixPlot.plot_simmatrix(matrix, ax=ax, titles=title, colormap=colormap)

        for ax in flat_axes[len(matrices):]:
            ax.axis("off")

        export_figure.tight_layout()
        export_figure.savefig(str(output_path))

        if skipped:
            self.statusBar().showMessage(
                f"Exported {len(matrices)} matrices to {output_path.name}. "
                f"Skipped {len(skipped)}."
            )
        else:
            self.statusBar().showMessage(f"Exported {len(matrices)} matrices to {output_path.name}.")

    def _current_title(self, path: Path) -> str:
        title = self.title_edit.text().strip()
        if not title:
            title = path.name
            self.title_edit.setText(title)
        self.titles[path] = title
        return title

    def _on_selection_changed(self, current, _previous) -> None:
        if current is None:
            self._clear_plot()
            return
        path = Path(current.data(USER_ROLE))
        title = self.titles.get(path, path.name)
        self.title_edit.setText(title)
        self._plot_selected()

    def _plot_selected(self, *_args) -> None:
        item = self.file_list.currentItem()
        if item is None:
            self._clear_plot()
            return
        path = Path(item.data(USER_ROLE))
        current_title = self._current_title(path)

        key = self._current_matrix_key()
        colormap = self.cmap_combo.currentText() or DEFAULT_COLORMAP
        try:
            with np.load(path) as npz:
                if key not in npz:
                    available = ", ".join(npz.files)
                    raise KeyError(f"Key '{key}' not found. Available: {available}")
                matrix = npz[key]
        except Exception as exc:
            self._clear_plot()
            self.statusBar().showMessage(f"Failed to load {path.name}: {exc}")
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        SimMatrixPlot.plot_simmatrix(matrix, ax=ax, titles=current_title, colormap=colormap)
        self.canvas.draw_idle()
        self.statusBar().showMessage(f"Plotted {path.name} ({key}, {colormap}).")

    def _clear_plot(self) -> None:
        self.figure.clear()
        self.canvas.draw_idle()

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile() and url.toLocalFile().lower().endswith(".npz"):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event) -> None:
        if not event.mimeData().hasUrls():
            event.ignore()
            return
        paths = []
        for url in event.mimeData().urls():
            if url.isLocalFile():
                paths.append(url.toLocalFile())
        self._add_files(paths)
        event.acceptProposedAction()


def main() -> int:
    app = QApplication(sys.argv)
    window = ConnectomeViewer()
    window.resize(1200, 800)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
