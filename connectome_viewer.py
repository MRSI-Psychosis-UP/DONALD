#!/usr/bin/env python3
import math
import sys
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import (
        QApplication,
        QAbstractItemView,
        QDialog,
        QFileDialog,
        QCheckBox,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QComboBox,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QPushButton,
        QSplitter,
        QSpinBox,
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
        QDialog,
        QFileDialog,
        QCheckBox,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QComboBox,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QPushButton,
        QSplitter,
        QSpinBox,
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
import matplotlib.transforms as mtransforms

from graphplot.simmatrix import SimMatrixPlot
from graphplot.colorbar import ColorBar

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
COLORBAR_BARS = [
    "blueblackred",
    "redblackblue",
    "mask_alpha",
    "bluewhitered",
    "redwhiteblue",
    "wg",
    "bbo",
    "bo",
    "obb",
    "bwo",
    "bwg",
    "bwTo",
    "owb",
    "spectrum_fsl",
    "random9",
]

PARCEL_LABEL_KEYS = ("parcel_labels_group", "parcel_labels_group.npy")
PARCEL_NAME_KEYS = ("parcel_names_group", "parcel_names_group.npy")
CMAPS_DIR = Path(__file__).with_name("graphplot") / "cmaps"

if QT_LIB == 6:
    Qt.Horizontal = Qt.Orientation.Horizontal
    Qt.Vertical = Qt.Orientation.Vertical
    Qt.Checked = Qt.CheckState.Checked
    Qt.Unchecked = Qt.CheckState.Unchecked


def _user_role():
    return getattr(Qt, "UserRole", getattr(Qt.ItemDataRole, "UserRole"))


USER_ROLE = _user_role()


def _is_valid_matrix_shape(shape) -> bool:
    if len(shape) == 2:
        return shape[0] == shape[1]
    if len(shape) == 3:
        a, b, c = shape
        return (a == b != c) or (a == c != b) or (b == c != a)
    return False


def _average_to_square(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim == 2:
        return matrix
    if matrix.ndim != 3:
        return matrix
    a, b, c = matrix.shape
    if a == b != c:
        return matrix.mean(axis=2)
    if a == c != b:
        return matrix.mean(axis=1)
    if b == c != a:
        return matrix.mean(axis=0)
    return matrix


def _load_matrix_from_npz(path: Path, key: str, average: bool = True) -> np.ndarray:
    with np.load(path) as npz:
        if key not in npz:
            raise KeyError(f"Key '{key}' not found")
        matrix = npz[key]
    if average and matrix.ndim == 3:
        matrix = _average_to_square(matrix)
    return matrix


def _load_parcel_metadata(path: Path):
    labels = None
    names = None
    with np.load(path) as npz:
        for key in PARCEL_LABEL_KEYS:
            if key in npz:
                labels = npz[key]
                break
        for key in PARCEL_NAME_KEYS:
            if key in npz:
                names = npz[key]
                break
    if labels is None:
        labels_path = path.with_name("parcel_labels_group.npy")
        if labels_path.exists():
            labels = np.load(labels_path)
    if names is None:
        names_path = path.with_name("parcel_names_group.npy")
        if names_path.exists():
            names = np.load(names_path)
    return labels, names


def _to_string_list(values):
    if values is None:
        return None
    try:
        flat = values.tolist()
    except Exception:
        flat = values
    if isinstance(flat, list):
        return [str(v) for v in flat]
    return [str(flat)]


def _load_covars_info(path: Path):
    try:
        with np.load(path, allow_pickle=True) as npz:
            if "covars" not in npz:
                return None
            covars = npz["covars"]
    except Exception:
        return None
    df = None
    if pd is not None:
        try:
            df = pd.DataFrame.from_records(covars)
        except Exception:
            df = None
    columns = list(covars.dtype.names) if getattr(covars.dtype, "names", None) else []
    return {"data": covars, "df": df, "columns": columns}


def _load_group_value(path: Path, index: int):
    try:
        with np.load(path, allow_pickle=True) as npz:
            if "group" not in npz:
                return None
            group_data = npz["group"]
    except Exception:
        return None
    if np.isscalar(group_data):
        return str(group_data)
    group_arr = np.asarray(group_data)
    if group_arr.ndim == 0:
        return str(group_arr.item())
    if index < 0 or index >= len(group_arr):
        return None
    return str(group_arr[index])


def _covars_columns(info):
    if info is None:
        return []
    df = info.get("df")
    if df is not None:
        return list(df.columns)
    return info.get("columns", [])


def _covars_series(info, name):
    if info is None:
        return None
    df = info.get("df")
    if df is not None:
        return df[name].to_numpy()
    data = info.get("data")
    if data is None or data.dtype.names is None:
        return None
    return data[name]


def _decode_strings(values):
    return np.array(
        [v.decode() if isinstance(v, (bytes, bytearray)) else str(v) for v in values],
        dtype=object,
    )


def _series_is_numeric(values) -> bool:
    try:
        np.asarray(values, dtype=float)
        return True
    except Exception:
        return False


def _discover_custom_cmaps():
    if not CMAPS_DIR.exists():
        return []
    return sorted({path.stem for path in CMAPS_DIR.glob("*.cmap")})


def _apply_rotation(ax, matrix: np.ndarray, degrees: float) -> None:
    if not ax.images:
        return
    im = ax.images[-1]
    height, width = matrix.shape[:2]
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    transform = mtransforms.Affine2D().rotate_deg_around(cx, cy, degrees)
    im.set_transform(transform + ax.transData)

    corners = np.array(
        [
            [-0.5, -0.5],
            [width - 0.5, -0.5],
            [width - 0.5, height - 0.5],
            [-0.5, height - 0.5],
        ]
    )
    theta = np.deg2rad(degrees)
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )
    rotated = (corners - np.array([cx, cy])) @ rotation.T + np.array([cx, cy])
    xmin, ymin = rotated.min(axis=0)
    xmax, ymax = rotated.max(axis=0)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)


def _remove_axes_border(ax) -> None:
    ax.set_frame_on(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _stack_axis(shape):
    if len(shape) != 3:
        return None
    a, b, c = shape
    if a == b != c:
        return 2
    if a == c != b:
        return 1
    if b == c != a:
        return 0
    return None


def _select_stack_slice(matrix: np.ndarray, axis: int, index: int) -> np.ndarray:
    if axis == 0:
        return matrix[index, :, :]
    if axis == 1:
        return matrix[:, index, :]
    return matrix[:, :, index]


def _strip_prefix(value: str, prefix: str) -> str:
    if value.startswith(prefix):
        return value[len(prefix) :]
    return value


def _parse_participant_id(value: str):
    text = str(value)
    if "-" in text:
        parts = text.split("-")
        if parts[0] in {"sub", "subject", "participant"}:
            return None, parts[-1]
        return parts[0], parts[-1]
    if "_" in text:
        parts = text.split("_")
        if parts[0] in {"sub", "subject", "participant"}:
            return None, parts[-1]
        return parts[0], parts[-1]
    return None, text


def _get_valid_keys(path: Path):
    keys = []
    try:
        with np.load(path) as npz:
            for key in npz.files:
                if key == "covars" or key in PARCEL_LABEL_KEYS or key in PARCEL_NAME_KEYS:
                    continue
                try:
                    shape = npz[key].shape
                except Exception:
                    continue
                if _is_valid_matrix_shape(shape):
                    keys.append(key)
    except Exception:
        return []
    return keys


def _format_value(value) -> str:
    try:
        return f"{float(value):.2f}"
    except Exception:
        return str(value)


class HistogramDialog(QDialog):
    def __init__(self, entries, entry_ids, titles, parent=None) -> None:
        super().__init__(parent)
        self.entries = entries
        self.entry_ids = list(entry_ids)
        self.titles = dict(titles)
        self._cache = {}
        self.setWindowTitle("Histogram Viewer")
        self._build_ui()
        self._update_plot()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)

        self.list_widget = QListWidget()
        for entry_id in self.entry_ids:
            entry = self.entries.get(entry_id)
            if entry is None:
                continue
            label = self.titles.get(entry_id, entry.get("label", "Matrix"))
            item = QListWidgetItem(label)
            item.setData(USER_ROLE, entry_id)
            item.setCheckState(Qt.Checked)
            self.list_widget.addItem(item)
        self.list_widget.itemChanged.connect(self._update_plot)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        layout.addWidget(self.list_widget, 0)
        layout.addWidget(self.canvas, 1)

    def _matrix_for_entry(self, entry):
        if entry.get("kind") == "derived":
            return entry.get("matrix"), entry.get("selected_key")
        key = entry.get("selected_key")
        if not key:
            valid_keys = _get_valid_keys(entry["path"])
            if not valid_keys:
                raise KeyError("No valid matrix key")
            key = valid_keys[0]
            entry["selected_key"] = key
        matrix = _load_matrix_from_npz(entry["path"], key, average=True)
        return matrix, key

    def _load_upper_triangle(self, entry_id: str):
        if entry_id in self._cache:
            return self._cache[entry_id]
        entry = self.entries.get(entry_id)
        if entry is None:
            raise KeyError("Entry not found")
        matrix, _ = self._matrix_for_entry(entry)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            values = matrix.ravel()
        else:
            idx = np.triu_indices_from(matrix, k=1)
            values = matrix[idx]
        self._cache[entry_id] = values
        return values

    def _update_plot(self) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        plotted = False
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() != Qt.Checked:
                continue
            entry_id = item.data(USER_ROLE)
            try:
                values = self._load_upper_triangle(entry_id)
            except Exception:
                continue
            label = self.titles.get(entry_id, item.text())
            ax.hist(values, bins=50, alpha=0.6, histtype="step", label=label)
            plotted = True
        if plotted:
            ax.set_title("Upper Triangle Histogram")
            ax.set_xlabel("Value")
            ax.set_ylabel("Count")
            ax.legend(fontsize=8)
        else:
            ax.set_title("No matrices selected")
            ax.set_xticks([])
            ax.set_yticks([])
        self.canvas.draw_idle()


class ConnectomeViewer(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._entries = {}
        self._derived_counter = 0
        self.titles = {}
        self._covars_cache = {}
        self._current_matrix = None
        self._current_parcel_labels = None
        self._current_parcel_names = None
        self._current_axes = None
        self._colorbar = ColorBar()
        self._custom_cmaps = set()
        self.setWindowTitle("Connectome Viewer")
        self._set_window_icon()
        self.setAcceptDrops(True)
        self._hist_dialog = None
        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget(self)
        main_layout = QHBoxLayout(central)
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setHandleWidth(10)
        self.main_splitter.setChildrenCollapsible(False)
        main_layout.addWidget(self.main_splitter)

        left_panel = QWidget()
        controls_layout = QVBoxLayout(left_panel)

        header = QLabel("Connectome matrices (.npz)")
        controls_layout.addWidget(header)

        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)
        self.export_button = QPushButton("Export Grid")
        self.export_button.clicked.connect(self._export_grid)
        export_layout.addWidget(self.export_button)

        export_cols_label = QLabel("Export columns:")
        export_layout.addWidget(export_cols_label)

        self.export_cols_spin = QSpinBox()
        self.export_cols_spin.setRange(1, 12)
        self.export_cols_spin.setValue(4)
        export_layout.addWidget(self.export_cols_spin)

        self.export_rotate_check = QCheckBox("Rotate 45 deg CW")
        export_layout.addWidget(self.export_rotate_check)

        selector_group = QGroupBox("Selector")
        selector_layout = QVBoxLayout(selector_group)
        key_label = QLabel("Matrix key:")
        selector_layout.addWidget(key_label)

        self.key_combo = QComboBox()
        self.key_combo.currentIndexChanged.connect(self._on_key_changed)
        selector_layout.addWidget(self.key_combo)

        sample_label = QLabel("Sample index:")
        selector_layout.addWidget(sample_label)

        sample_layout = QHBoxLayout()
        self.sample_spin = QSpinBox()
        self.sample_spin.setRange(-1, 0)
        self.sample_spin.setSpecialValueText("Average")
        self.sample_spin.setEnabled(False)
        self.sample_spin.valueChanged.connect(self._on_sample_changed)
        sample_layout.addWidget(self.sample_spin)
        self.sample_add_button = QPushButton("Add")
        self.sample_add_button.clicked.connect(self._add_sample_entry)
        self.sample_add_button.setEnabled(False)
        sample_layout.addWidget(self.sample_add_button)
        selector_layout.addLayout(sample_layout)

        covar_label = QLabel("Covariate:")
        selector_layout.addWidget(covar_label)

        self.covar_combo = QComboBox()
        selector_layout.addWidget(self.covar_combo)

        covar_value_label = QLabel("Covar value:")
        selector_layout.addWidget(covar_value_label)

        self.covar_value_edit = QLineEdit("")
        self.covar_value_edit.setPlaceholderText("Numeric value")
        selector_layout.addWidget(self.covar_value_edit)

        self.average_button = QPushButton("Average")
        self.average_button.clicked.connect(self._average_selected)
        selector_layout.addWidget(self.average_button)

        cmap_group = QGroupBox("Color map")
        cmap_layout = QVBoxLayout(cmap_group)
        cmap_label = QLabel("Colormap:")
        cmap_layout.addWidget(cmap_label)

        self.cmap_combo = QComboBox()
        self._reload_colormaps()
        self.cmap_combo.currentIndexChanged.connect(self._plot_selected)
        cmap_layout.addWidget(self.cmap_combo)

        list_group = QGroupBox("Matrices")
        list_layout = QVBoxLayout(list_group)
        self.add_button = QPushButton("Add Files")
        self.add_button.clicked.connect(self._open_files)
        list_layout.addWidget(self.add_button)

        self.remove_button = QPushButton("Remove Selected")
        self.remove_button.clicked.connect(self._remove_selected)
        list_layout.addWidget(self.remove_button)

        self.clear_button = QPushButton("Clear List")
        self.clear_button.clicked.connect(self._clear_files)
        list_layout.addWidget(self.clear_button)

        self.hist_button = QPushButton("View Histogram")
        self.hist_button.clicked.connect(self._open_histogram)
        list_layout.addWidget(self.hist_button)

        title_label = QLabel("Plot title:")
        list_layout.addWidget(title_label)

        self.title_edit = QLineEdit("")
        self.title_edit.setPlaceholderText("Defaults to file name")
        self.title_edit.editingFinished.connect(self._on_title_edited)
        self.title_edit.returnPressed.connect(self._on_title_edited)
        list_layout.addWidget(self.title_edit)

        self.file_list = QListWidget()
        if QT_LIB == 6:
            self.file_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        else:
            self.file_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.file_list.currentItemChanged.connect(self._on_selection_changed)
        list_layout.addWidget(self.file_list)

        move_layout = QHBoxLayout()
        self.move_up_button = QPushButton("Move Up")
        self.move_up_button.clicked.connect(lambda: self._move_selected(-1))
        move_layout.addWidget(self.move_up_button)
        self.move_down_button = QPushButton("Move Down")
        self.move_down_button.clicked.connect(lambda: self._move_selected(1))
        move_layout.addWidget(self.move_down_button)
        list_layout.addLayout(move_layout)

        hint = QLabel("Drag & drop .npz files here.")
        hint.setWordWrap(True)
        list_layout.addWidget(hint)

        gradients_group = QGroupBox("Gradients")
        gradients_layout = QVBoxLayout(gradients_group)
        gradients_row = QHBoxLayout()
        self.gradients_show_button = QPushButton("Show")
        self.gradients_show_button.clicked.connect(self._show_gradients)
        gradients_row.addWidget(self.gradients_show_button)
        self.gradients_spin = QSpinBox()
        self.gradients_spin.setRange(1, 10)
        self.gradients_spin.setValue(4)
        gradients_row.addWidget(self.gradients_spin)
        gradients_layout.addLayout(gradients_row)

        group_style = (
            "QGroupBox {"
            "font-weight: bold;"
            "font-size: 14pt;"
            "border: 3px solid #666666;"
            "border-radius: 8px;"
            "margin-top: 14px;"
            "padding-top: 8px;"
            "}"
            "QGroupBox::title {"
            "subcontrol-origin: margin;"
            "left: 12px;"
            "padding: 0 6px;"
            "}"
        )
        for group in (list_group, selector_group, cmap_group, export_group, gradients_group):
            group.setStyleSheet(group_style)

        controls_layout.addWidget(list_group)
        controls_layout.addWidget(selector_group)
        controls_layout.addWidget(cmap_group)
        controls_layout.addWidget(export_group)

        controls_layout.addStretch(1)

        center_panel = QWidget()
        right_layout = QVBoxLayout(center_panel)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect("motion_notify_event", self._on_hover)
        right_layout.addWidget(self.canvas, 1)

        self.hover_label = QLabel("")
        self.hover_label.setWordWrap(True)
        right_layout.addWidget(self.hover_label, 0)

        right_panel = QWidget()
        right_panel_layout = QVBoxLayout(right_panel)
        right_panel_layout.addWidget(gradients_group)
        right_panel_layout.addStretch(1)

        self.main_splitter.addWidget(left_panel)
        self.main_splitter.addWidget(center_panel)
        self.main_splitter.addWidget(right_panel)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)
        self.main_splitter.setSizes([420, 1000, 280])

        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready.")

    def _set_window_icon(self) -> None:
        icon_path = Path(__file__).with_name("icons") / "conviewer.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

    def _file_entry_id(self, path: Path) -> str:
        return f"file::{path}"

    def _new_derived_id(self) -> str:
        self._derived_counter += 1
        return f"derived::{self._derived_counter}"

    def _entry_ids(self):
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            entry_id = item.data(USER_ROLE)
            if entry_id:
                yield entry_id

    def _current_entry_id(self):
        item = self.file_list.currentItem()
        if item is None:
            return None
        return item.data(USER_ROLE)

    def _current_entry(self):
        entry_id = self._current_entry_id()
        if entry_id is None:
            return None
        return self._entries.get(entry_id)

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
            entry_id = self._file_entry_id(path)
            if entry_id in self._entries:
                continue
            entry = {
                "id": entry_id,
                "kind": "file",
                "path": path,
                "selected_key": None,
                "sample_index": None,
                "auto_title": True,
                "label": path.name,
            }
            self._entries[entry_id] = entry
            item = QListWidgetItem(path.name)
            item.setToolTip(str(path))
            item.setData(USER_ROLE, entry_id)
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
        entry_id = item.data(USER_ROLE)
        if entry_id in self._entries:
            self._entries.pop(entry_id, None)
        self.titles.pop(entry_id, None)
        row = self.file_list.row(item)
        self.file_list.takeItem(row)
        if self.file_list.count() == 0:
            self._clear_plot()

    def _clear_files(self) -> None:
        self._entries.clear()
        self.titles.clear()
        self.file_list.clear()
        self._clear_plot()
        self.statusBar().showMessage("File list cleared.")

    def _refresh_key_options(self, entry) -> None:
        self.key_combo.blockSignals(True)
        valid_keys = _get_valid_keys(entry["path"])
        self.key_combo.clear()
        self.key_combo.addItems(valid_keys)
        selected_key = entry.get("selected_key")
        if selected_key and selected_key in valid_keys:
            self.key_combo.setCurrentText(selected_key)
        elif valid_keys:
            entry["selected_key"] = valid_keys[0]
            self.key_combo.setCurrentIndex(0)
        else:
            entry["selected_key"] = None
        self.key_combo.setEnabled(bool(valid_keys))
        self.key_combo.blockSignals(False)

    def _refresh_covars_options(self, source_path: Path, entry) -> None:
        info = self._covars_cache.get(source_path)
        if info is None:
            info = _load_covars_info(source_path)
            self._covars_cache[source_path] = info
        self.covar_combo.blockSignals(True)
        self.covar_combo.clear()
        columns = _covars_columns(info)
        for col in columns:
            self.covar_combo.addItem(col)
        covar_name = entry.get("covar_name")
        if covar_name and self.covar_combo.findText(covar_name) >= 0:
            self.covar_combo.setCurrentText(covar_name)
        self.covar_combo.blockSignals(False)
        enabled = self.covar_combo.count() > 0
        self.covar_combo.setEnabled(enabled)
        self.average_button.setEnabled(enabled)

    def _default_title_for_entry(self, entry) -> str:
        base_label = entry.get("label", "Matrix")
        if entry.get("kind") != "file":
            return base_label
        sample_index = entry.get("sample_index")
        if sample_index is None or sample_index < 0:
            return base_label
        source_path = entry.get("path")
        if source_path is None:
            return base_label
        info = self._covars_cache.get(source_path)
        if info is None:
            info = _load_covars_info(source_path)
            self._covars_cache[source_path] = info
        if info is None:
            return base_label
        participant = _covars_series(info, "participant_id")
        session = _covars_series(info, "session_id")
        if participant is None or session is None:
            return base_label
        if sample_index >= len(participant) or sample_index >= len(session):
            return base_label
        participant_value = str(participant[sample_index])
        session_value = str(session[sample_index])
        group_value = _load_group_value(source_path, sample_index)
        group, sub = _parse_participant_id(participant_value)
        if group_value:
            group = group_value
        sub = _strip_prefix(sub, "sub-")
        ses = _strip_prefix(session_value, "ses-")
        if group:
            return f"{group}-sub-{sub}_ses-{ses}"
        return f"sub-{sub}_ses-{ses}"

    def _apply_title_for_entry(self, entry, force: bool = False) -> str:
        entry_id = entry["id"]
        default_title = self._default_title_for_entry(entry)
        if force or entry.get("auto_title", True):
            self.titles[entry_id] = default_title
            self.title_edit.setText(default_title)
            return default_title
        current = self.title_edit.text().strip() or self.titles.get(entry_id, default_title)
        self.titles[entry_id] = current
        return current

    def _on_title_edited(self) -> None:
        entry = self._current_entry()
        if entry is None:
            return
        entry_id = entry["id"]
        default_title = self._default_title_for_entry(entry)
        text = self.title_edit.text().strip()
        if text and text != default_title:
            entry["auto_title"] = False
            self.titles[entry_id] = text
        else:
            entry["auto_title"] = True
            self.titles[entry_id] = default_title
            self.title_edit.setText(default_title)
        self._plot_selected()

    def _on_sample_changed(self, value: int) -> None:
        entry = self._current_entry()
        if entry is None or entry.get("kind") != "file":
            return
        entry["sample_index"] = value
        if entry.get("auto_title", True):
            self._apply_title_for_entry(entry, force=True)
        self._plot_selected()

    def _update_sample_controls(self, entry, axis, stack_len) -> None:
        self.sample_spin.blockSignals(True)
        if axis is None or stack_len is None:
            self.sample_spin.setEnabled(False)
            self.sample_spin.setRange(-1, 0)
            self.sample_spin.setSpecialValueText("Average")
            self.sample_spin.setValue(-1)
            self.sample_add_button.setEnabled(False)
            if entry is not None:
                entry["sample_index"] = None
                entry["stack_axis"] = None
                entry["stack_len"] = None
        else:
            self.sample_spin.setEnabled(True)
            self.sample_spin.setRange(-1, max(stack_len - 1, 0))
            self.sample_spin.setSpecialValueText("Average")
            entry.setdefault("sample_index", -1)
            if entry["sample_index"] is None or entry["sample_index"] >= stack_len:
                entry["sample_index"] = -1
            entry["stack_axis"] = axis
            entry["stack_len"] = stack_len
            self.sample_spin.setValue(entry["sample_index"])
            self.sample_add_button.setEnabled(True)
        self.sample_spin.blockSignals(False)

    def _reload_colormaps(self) -> None:
        custom = _discover_custom_cmaps()
        self._custom_cmaps = set(custom)
        current = self.cmap_combo.currentText() if hasattr(self, "cmap_combo") else ""
        names = []
        seen = set()
        for name in COLORMAPS + COLORBAR_BARS + custom:
            if name not in seen:
                names.append(name)
                seen.add(name)
        self.cmap_combo.blockSignals(True)
        self.cmap_combo.clear()
        self.cmap_combo.addItems(names)
        if current and current in names:
            self.cmap_combo.setCurrentText(current)
        elif DEFAULT_COLORMAP in names:
            self.cmap_combo.setCurrentText(DEFAULT_COLORMAP)
        self.cmap_combo.blockSignals(False)

    def _selected_colormap_name(self) -> str:
        name = self.cmap_combo.currentText().strip()
        return name or DEFAULT_COLORMAP

    def _selected_colormap(self):
        name = self._selected_colormap_name()
        if name in self._custom_cmaps:
            try:
                return self._colorbar.load_fsl_cmap(name)
            except Exception:
                return name
        return name

    def _move_selected(self, direction: int) -> None:
        row = self.file_list.currentRow()
        if row < 0:
            return
        new_row = row + direction
        if new_row < 0 or new_row >= self.file_list.count():
            return
        item = self.file_list.takeItem(row)
        self.file_list.insertItem(new_row, item)
        self.file_list.setCurrentRow(new_row)

    def _on_key_changed(self) -> None:
        entry = self._current_entry()
        if entry and entry.get("kind") == "file":
            key = self.key_combo.currentText().strip()
            entry["selected_key"] = key or None
            entry["sample_index"] = None
        self._plot_selected()

    def _open_histogram(self) -> None:
        if self.file_list.count() == 0:
            self.statusBar().showMessage("No matrices to plot.")
            return
        dialog = HistogramDialog(self._entries, list(self._entry_ids()), self.titles, parent=self)
        dialog.show()
        self._hist_dialog = dialog

    def _ensure_entry_key(self, entry):
        if entry.get("kind") != "file":
            return entry.get("selected_key")
        key = entry.get("selected_key")
        if key:
            return key
        valid_keys = _get_valid_keys(entry["path"])
        if not valid_keys:
            return None
        entry["selected_key"] = valid_keys[0]
        return entry["selected_key"]

    def _matrix_for_entry(self, entry):
        if entry.get("kind") == "derived":
            return entry.get("matrix"), entry.get("selected_key")
        key = self._ensure_entry_key(entry)
        if not key:
            raise KeyError("No valid matrix key selected")
        raw = _load_matrix_from_npz(entry["path"], key, average=False)
        if raw.ndim == 3:
            axis = _stack_axis(raw.shape)
            if axis is None:
                matrix = _average_to_square(raw)
            else:
                entry.setdefault("sample_index", -1)
                entry["stack_axis"] = axis
                entry["stack_len"] = raw.shape[axis]
                sample_index = entry.get("sample_index", -1)
                if sample_index is None or sample_index < 0:
                    matrix = _average_to_square(raw)
                else:
                    matrix = _select_stack_slice(raw, axis, sample_index)
        else:
            matrix = raw
        return matrix, key

    def _export_grid(self) -> None:
        if self.file_list.count() == 0:
            self.statusBar().showMessage("No matrices to export.")
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

        colormap = self._selected_colormap()
        matrices = []
        titles = []
        skipped = []
        for entry_id in self._entry_ids():
            entry = self._entries.get(entry_id)
            if entry is None:
                continue
            try:
                matrix, _ = self._matrix_for_entry(entry)
            except Exception as exc:
                label = entry.get("label", entry_id)
                skipped.append(f"{label} ({exc})")
                continue
            matrices.append(matrix)
            titles.append(self.titles.get(entry_id, entry.get("label", "Matrix")))

        if not matrices:
            self.statusBar().showMessage("No matrices exported (missing keys or load errors).")
            return

        cols = min(self.export_cols_spin.value(), len(matrices))
        rows = int(math.ceil(len(matrices) / cols))
        export_figure = Figure(figsize=(4 * cols, 4 * rows))
        axes = export_figure.subplots(rows, cols)
        if isinstance(axes, np.ndarray):
            flat_axes = axes.flatten()
        else:
            flat_axes = [axes]

        rotate = self.export_rotate_check.isChecked()
        for idx, (matrix, title) in enumerate(zip(matrices, titles)):
            ax = flat_axes[idx]
            SimMatrixPlot.plot_simmatrix(matrix, ax=ax, titles=title, colormap=colormap)
            _remove_axes_border(ax)
            if rotate:
                _apply_rotation(ax, matrix, -45.0)

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

    def _on_selection_changed(self, current, _previous) -> None:
        if current is None:
            self._clear_plot()
            return
        entry_id = current.data(USER_ROLE)
        entry = self._entries.get(entry_id)
        if entry is None:
            self._clear_plot()
            return

        if entry.get("kind") == "file":
            self.key_combo.setEnabled(True)
            self._refresh_key_options(entry)
        else:
            self.key_combo.blockSignals(True)
            self.key_combo.clear()
            selected_key = entry.get("selected_key")
            if selected_key:
                self.key_combo.addItem(selected_key)
                self.key_combo.setCurrentText(selected_key)
            self.key_combo.setEnabled(False)
            self.key_combo.blockSignals(False)

        source_path = entry.get("source_path", entry.get("path"))
        if source_path:
            self._refresh_covars_options(source_path, entry)

        if entry.get("auto_title", True):
            self._apply_title_for_entry(entry, force=True)
        else:
            self.title_edit.setText(self.titles.get(entry_id, entry.get("label", "Matrix")))
        self._plot_selected()

    def _plot_selected(self, *_args) -> None:
        entry_id = self._current_entry_id()
        if entry_id is None:
            self._clear_plot()
            return
        entry = self._entries.get(entry_id)
        if entry is None:
            self._clear_plot()
            return

        if entry.get("kind") == "derived":
            matrix = entry.get("matrix")
            key = entry.get("selected_key")
            source_path = entry.get("source_path")
            self._update_sample_controls(entry, None, None)
        else:
            key = self._ensure_entry_key(entry)
            if not key:
                self._clear_plot()
                self.statusBar().showMessage("No valid matrix key selected.")
                return
            source_path = entry.get("path")
            try:
                raw = _load_matrix_from_npz(entry["path"], key, average=False)
            except Exception as exc:
                self._clear_plot()
                self.statusBar().showMessage(f"Failed to load {entry.get('label', 'file')}: {exc}")
                return
            if raw.ndim == 3:
                axis = _stack_axis(raw.shape)
                if axis is None:
                    self._update_sample_controls(entry, None, None)
                    matrix = _average_to_square(raw)
                else:
                    self._update_sample_controls(entry, axis, raw.shape[axis])
                    sample_index = entry.get("sample_index", -1)
                    if sample_index is None or sample_index < 0:
                        matrix = _average_to_square(raw)
                    else:
                        matrix = _select_stack_slice(raw, axis, sample_index)
            else:
                self._update_sample_controls(entry, None, None)
                matrix = raw

        current_title = self._apply_title_for_entry(entry)

        labels, names = (None, None)
        if source_path:
            labels, names = _load_parcel_metadata(source_path)
        labels_list = _to_string_list(labels)
        names_list = _to_string_list(names)
        if labels_list and len(labels_list) != matrix.shape[0]:
            labels_list = None
        if names_list and len(names_list) != matrix.shape[0]:
            names_list = None

        self._current_matrix = matrix
        self._current_parcel_labels = labels_list
        self._current_parcel_names = names_list

        colormap = self._selected_colormap()
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        SimMatrixPlot.plot_simmatrix(matrix, ax=ax, titles=current_title, colormap=colormap)
        _remove_axes_border(ax)
        self._current_axes = ax
        self.canvas.draw_idle()
        key_text = f", {key}" if key else ""
        self.statusBar().showMessage(f"Plotted {entry.get('label', 'matrix')}{key_text}.")

    def _average_selected(self) -> None:
        entry = self._current_entry()
        if entry is None:
            self.statusBar().showMessage("No matrix selected.")
            return
        source_path = entry.get("path") if entry.get("kind") == "file" else entry.get("source_path")
        if source_path is None:
            self.statusBar().showMessage("No source file for averaging.")
            return
        key = entry.get("selected_key") or self._ensure_entry_key(entry)
        if not key:
            self.statusBar().showMessage("No valid matrix key selected.")
            return
        info = self._covars_cache.get(source_path)
        if info is None:
            info = _load_covars_info(source_path)
            self._covars_cache[source_path] = info
        if info is None:
            self.statusBar().showMessage("Covars not found in file.")
            return
        covar_name = self.covar_combo.currentText().strip()
        if not covar_name:
            self.statusBar().showMessage("Select a covariate.")
            return
        value_text = self.covar_value_edit.text().strip()
        if not value_text:
            self.statusBar().showMessage("Enter a covariate value.")
            return
        value_float = None
        try:
            value_float = float(value_text)
        except ValueError:
            value_float = None
        series = _covars_series(info, covar_name)
        if series is None:
            self.statusBar().showMessage("Covar column not available.")
            return
        series_arr = np.asarray(series)
        if _series_is_numeric(series_arr):
            if value_float is None:
                self.statusBar().showMessage("Covar value must be numeric.")
                return
            series_float = np.asarray(series_arr, dtype=float)
            mask = np.isclose(series_float, value_float)
        else:
            series_str = _decode_strings(series_arr)
            mask = series_str == value_text
        indices = np.where(mask)[0]
        if indices.size == 0:
            self.statusBar().showMessage("No covar matches the selected value.")
            return

        try:
            with np.load(source_path) as npz:
                if key not in npz:
                    raise KeyError(f"Key '{key}' not found")
                matrix = npz[key]
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to load matrix: {exc}")
            return

        if matrix.ndim != 3:
            self.statusBar().showMessage("Selected matrix is not multi-dimensional.")
            return

        shape = matrix.shape
        axis = None
        if shape[0] == shape[1] != shape[2]:
            axis = 2
        elif shape[0] == shape[2] != shape[1]:
            axis = 1
        elif shape[1] == shape[2] != shape[0]:
            axis = 0
        else:
            self.statusBar().showMessage("Matrix is not a stack of square matrices.")
            return

        covars_len = len(series_arr)
        if shape[axis] != covars_len:
            self.statusBar().showMessage("Covars length does not match matrix stack size.")
            return

        if axis == 0:
            subset = matrix[indices, :, :]
            averaged = subset.mean(axis=0)
        elif axis == 1:
            subset = matrix[:, indices, :]
            averaged = subset.mean(axis=1)
        else:
            subset = matrix[:, :, indices]
            averaged = subset.mean(axis=2)

        derived_id = self._new_derived_id()
        label = f"avg {covar_name}={value_text} ({Path(source_path).name})"
        covar_value = value_float if value_float is not None else value_text
        self._entries[derived_id] = {
            "id": derived_id,
            "kind": "derived",
            "matrix": averaged,
            "source_path": source_path,
            "selected_key": key,
            "covar_name": covar_name,
            "covar_value": covar_value,
            "sample_index": None,
            "auto_title": True,
            "label": label,
        }
        self.titles[derived_id] = label
        item = QListWidgetItem(label)
        item.setData(USER_ROLE, derived_id)
        self.file_list.addItem(item)
        self.file_list.setCurrentItem(item)
        self.statusBar().showMessage(f"Added averaged matrix ({covar_name}={value_text}).")

    def _add_sample_entry(self) -> None:
        entry = self._current_entry()
        if entry is None or entry.get("kind") != "file":
            self.statusBar().showMessage("Select a file-based matrix first.")
            return
        key = self._ensure_entry_key(entry)
        if not key:
            self.statusBar().showMessage("No valid matrix key selected.")
            return
        source_path = entry.get("path")
        if source_path is None:
            self.statusBar().showMessage("No source file available.")
            return
        try:
            raw = _load_matrix_from_npz(source_path, key, average=False)
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to load matrix: {exc}")
            return
        if raw.ndim != 3:
            self.statusBar().showMessage("Selected matrix is not multi-dimensional.")
            return
        axis = _stack_axis(raw.shape)
        if axis is None:
            self.statusBar().showMessage("Matrix is not a stack of square matrices.")
            return
        sample_index = entry.get("sample_index", -1)
        if sample_index is None or sample_index < 0:
            matrix = _average_to_square(raw)
            label = f"avg {key} ({Path(source_path).name})"
        else:
            if sample_index >= raw.shape[axis]:
                self.statusBar().showMessage("Sample index out of range.")
                return
            matrix = _select_stack_slice(raw, axis, sample_index)
            label = self._default_title_for_entry(entry)

        derived_id = self._new_derived_id()
        self._entries[derived_id] = {
            "id": derived_id,
            "kind": "derived",
            "matrix": matrix,
            "source_path": source_path,
            "selected_key": key,
            "sample_index": sample_index,
            "auto_title": True,
            "label": label,
        }
        self.titles[derived_id] = label
        item = QListWidgetItem(label)
        item.setData(USER_ROLE, derived_id)
        self.file_list.addItem(item)
        self.file_list.setCurrentItem(item)
        self.statusBar().showMessage("Added sample matrix to list.")

    def _show_gradients(self) -> None:
        if self._current_matrix is None:
            self.statusBar().showMessage("No matrix selected for gradients.")
            return
        conn_matrix = np.asarray(self._current_matrix)
        if conn_matrix.ndim != 2 or conn_matrix.shape[0] != conn_matrix.shape[1]:
            self.statusBar().showMessage("Gradients require a square matrix.")
            return
        if conn_matrix.shape[0] != 400:
            self.statusBar().showMessage("Gradients require a 400x400 matrix (Schaefer scale=400).")
            return
        try:
            from brainspace.datasets import load_parcellation, load_conte69
            from brainspace.gradient import GradientMaps
            from brainspace.utils.parcellation import map_to_labels
            from brainspace.plotting import plot_hemispheres
        except Exception as exc:
            self.statusBar().showMessage(f"brainspace not available: {exc}")
            return

        labeling = load_parcellation("schaefer", scale=400, join=True)
        surf_lh, surf_rh = load_conte69()

        gm = GradientMaps(n_components=10, random_state=0)
        gm.fit(conn_matrix)

        n_grad = self.gradients_spin.value()
        mask = labeling != 0
        grad = [None] * n_grad
        for i in range(n_grad):
            grad[i] = map_to_labels(gm.gradients_[:, i], labeling, mask=mask, fill=np.nan)

        label_text_list = [f"Grad{i}" for i in range(n_grad)]
        plot_hemispheres(
            surf_lh,
            surf_rh,
            array_name=grad,
            size=(1200, 400),
            cmap="viridis_r",
            color_bar=True,
            label_text=label_text_list,
            zoom=1.55,
        )

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, figsize=(5, 4))
        ax.scatter(range(gm.lambdas_.size), gm.lambdas_)
        ax.set_xlabel("Component Nb")
        ax.set_ylabel("Eigenvalue")
        plt.show()

    def _clear_plot(self) -> None:
        self.figure.clear()
        self._current_matrix = None
        self._current_parcel_labels = None
        self._current_parcel_names = None
        self._current_axes = None
        self.hover_label.setText("")
        self.canvas.draw_idle()

    def _on_hover(self, event) -> None:
        if (
            self._current_matrix is None
            or self._current_axes is None
            or event.inaxes != self._current_axes
            or event.xdata is None
            or event.ydata is None
        ):
            self.hover_label.setText("")
            return
        row = int(event.ydata)
        col = int(event.xdata)
        if (
            row < 0
            or col < 0
            or row >= self._current_matrix.shape[0]
            or col >= self._current_matrix.shape[1]
        ):
            self.hover_label.setText("")
            return
        row_label = str(row)
        col_label = str(col)
        if self._current_parcel_labels and row < len(self._current_parcel_labels):
            row_label = self._current_parcel_labels[row]
        if self._current_parcel_labels and col < len(self._current_parcel_labels):
            col_label = self._current_parcel_labels[col]
        row_name = ""
        col_name = ""
        if self._current_parcel_names and row < len(self._current_parcel_names):
            row_name = self._current_parcel_names[row]
        if self._current_parcel_names and col < len(self._current_parcel_names):
            col_name = self._current_parcel_names[col]
        value_text = _format_value(self._current_matrix[row, col])
        parts = [f"Row {row}: {row_label}", f"Col {col}: {col_label}", f"Value: {value_text}"]
        if row_name or col_name:
            parts.append(f"Names: {row_name} | {col_name}")
        self.hover_label.setText("  ".join(parts))

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
