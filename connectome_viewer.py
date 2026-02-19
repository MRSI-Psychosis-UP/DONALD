#!/usr/bin/env python3
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from PyQt6.QtCore import Qt, QSize
    from PyQt6.QtWidgets import (
        QApplication,
        QAbstractItemView,
        QDialog,
        QGridLayout,
        QFileDialog,
        QCheckBox,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QComboBox,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QProgressBar,
        QPushButton,
        QSplashScreen,
        QSplitter,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )
    from PyQt6.QtGui import QAction, QIcon, QFontMetrics, QPixmap, QColor
    QT_LIB = 6
except ImportError:
    from PyQt5.QtCore import Qt, QSize
    from PyQt5.QtWidgets import (
        QAction,
        QApplication,
        QAbstractItemView,
        QDialog,
        QGridLayout,
        QFileDialog,
        QCheckBox,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QComboBox,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QProgressBar,
        QPushButton,
        QSplashScreen,
        QSplitter,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )
    from PyQt5.QtGui import QIcon, QFontMetrics, QPixmap, QColor
    QT_LIB = 5

try:
    if QT_LIB == 6:
        from PyQt6.QtGui import QPainter
        from PyQt6.QtSvg import QSvgRenderer
    else:
        from PyQt5.QtGui import QPainter
        from PyQt5.QtSvg import QSvgRenderer
except Exception:
    QPainter = None
    QSvgRenderer = None

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except ImportError:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.transforms as mtransforms

from mrsitoolbox.graphplot.simmatrix import SimMatrixPlot
from mrsitoolbox.graphplot.colorbar import ColorBar
from mrsitoolbox.graphplot import colorbar as colorbar_module
from mrsitoolbox.connectomics.nettools import NetTools

nettools = NetTools()
# Ensure Qt can locate platform plugins when installed via pip wheels.
if QT_LIB == 6:
    try:
        import PyQt6

        qt_plugins = Path(PyQt6.__file__).resolve().parent / "Qt6" / "plugins"
        os.environ.setdefault("QT_PLUGIN_PATH", str(qt_plugins))
        os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(qt_plugins / "platforms"))
    except Exception:
        pass
else:
    try:
        import PyQt5

        qt5_base = Path(PyQt5.__file__).resolve().parent
        qt_plugins = qt5_base / "Qt5" / "plugins"
        if not qt_plugins.exists():
            qt_plugins = qt5_base / "Qt" / "plugins"
        os.environ.setdefault("QT_PLUGIN_PATH", str(qt_plugins))
        os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(qt_plugins / "platforms"))
    except Exception:
        pass

DEFAULT_COLORMAP = "plasma"
DEFAULT_GRADIENT_COLORMAP = "spectrum_fsl"


def _first_env_value(*names: str) -> str:
    for name in names:
        value = str(os.getenv(name, "")).strip()
        if value:
            return value
    return ""


_MATLAB_FROM_ENV = _first_env_value(
    "MRSI_MATLAB_CMD",
    "MATLAB_CMD",
    "MATLAB_EXECUTABLE",
)
_MATLAB_FROM_PATH = shutil.which("matlab")
_MATLAB_DEFAULT_CANDIDATE = _MATLAB_FROM_ENV or _MATLAB_FROM_PATH or ""
if _MATLAB_DEFAULT_CANDIDATE:
    _matlab_path = Path(_MATLAB_DEFAULT_CANDIDATE).expanduser()
    DEFAULT_MATLAB_CMD = str(_matlab_path.resolve()) if _matlab_path.exists() else str(_matlab_path)
else:
    DEFAULT_MATLAB_CMD = ""

_NBS_FROM_ENV = _first_env_value(
    "MRSI_NBS_PATH",
    "MATLAB_NBS_PATH",
    "NBS_PATH",
)
if _NBS_FROM_ENV:
    _nbs_path = Path(_NBS_FROM_ENV).expanduser()
    DEFAULT_MATLAB_NBS_PATH = str(_nbs_path.resolve()) if _nbs_path.exists() else str(_nbs_path)
else:
    DEFAULT_MATLAB_NBS_PATH = ""
_CONFIG_HOME = Path(os.getenv("XDG_CONFIG_HOME", str(Path.home() / ".config"))).expanduser()
CONFIG_PATH = _CONFIG_HOME / "mrsi_viewer" / "connectome_viewer.json"
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
CMAPS_DIR = Path(colorbar_module.__file__).with_name("cmaps")
ROOTDIR = Path(__file__).resolve().parent
DEFAULT_PARCELLATION_DIR = ROOTDIR / "data"
APP_ICON_CANDIDATES = (
    ROOTDIR / "icons" / "conviewer.png",
    ROOTDIR / "icons" / "conviewer_2.png",
)

if QT_LIB == 6:
    Qt.Horizontal = Qt.Orientation.Horizontal
    Qt.Vertical = Qt.Orientation.Vertical
    Qt.Checked = Qt.CheckState.Checked
    Qt.Unchecked = Qt.CheckState.Unchecked
    Qt.ElideRight = Qt.TextElideMode.ElideRight


def _user_role():
    return getattr(Qt, "UserRole", getattr(Qt.ItemDataRole, "UserRole"))


USER_ROLE = _user_role()


def _dialog_accepted_code():
    accepted = getattr(QDialog, "Accepted", None)
    if accepted is not None:
        return accepted
    dialog_code = getattr(QDialog, "DialogCode", None)
    if dialog_code is not None and hasattr(dialog_code, "Accepted"):
        return dialog_code.Accepted
    return 1


def _load_app_icon() -> QIcon:
    for icon_path in APP_ICON_CANDIDATES:
        if not icon_path.exists():
            continue
        icon = QIcon(str(icon_path))
        if not icon.isNull():
            return icon
    return QIcon()


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
    with np.load(path, allow_pickle=True) as npz:
        if key not in npz:
            raise KeyError(f"Key '{key}' not found")
        matrix = npz[key]
    if average and matrix.ndim == 3:
        matrix = _average_to_square(matrix)
    return matrix


def _load_parcel_metadata(path: Path):
    labels = None
    names = None
    with np.load(path, allow_pickle=True) as npz:
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
            labels = np.load(labels_path, allow_pickle=True)
    if names is None:
        names_path = path.with_name("parcel_names_group.npy")
        if names_path.exists():
            names = np.load(names_path, allow_pickle=True)
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
        with np.load(path, allow_pickle=True) as npz:
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


def _coerce_label_indices(labels, expected_len: int):
    if labels is None:
        return None
    values = np.asarray(labels).reshape(-1)
    if values.size != expected_len:
        return None
    out = []
    for raw in values:
        value = raw.item() if isinstance(raw, np.generic) else raw
        if isinstance(value, (bytes, bytearray)):
            value = value.decode(errors="ignore")
        try:
            number = float(value)
        except Exception:
            return None
        if not np.isfinite(number):
            return None
        rounded = int(round(number))
        if not np.isclose(number, rounded):
            return None
        out.append(rounded)
    return out


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


class PreferencesDialog(QDialog):
    def __init__(
        self,
        *,
        theme_name: str,
        matrix_cmap: str,
        gradient_cmap: str,
        matlab_cmd: str,
        matlab_nbs_path: str,
        colormap_names,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.resize(720, 340)
        self._colormap_names = list(colormap_names or [])
        self._build_ui(
            theme_name=theme_name,
            matrix_cmap=matrix_cmap,
            gradient_cmap=gradient_cmap,
            matlab_cmd=matlab_cmd,
            matlab_nbs_path=matlab_nbs_path,
        )

    def _build_ui(
        self,
        *,
        theme_name: str,
        matrix_cmap: str,
        gradient_cmap: str,
        matlab_cmd: str,
        matlab_nbs_path: str,
    ) -> None:
        layout = QVBoxLayout(self)
        form = QGridLayout()
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(10)

        row = 0
        form.addWidget(QLabel("Theme"), row, 0)
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light", "Teya", "Donald"])
        if self.theme_combo.findText(theme_name) >= 0:
            self.theme_combo.setCurrentText(theme_name)
        else:
            self.theme_combo.setCurrentText("Dark")
        form.addWidget(self.theme_combo, row, 1, 1, 3)

        row += 1
        form.addWidget(QLabel("Default matrix color map"), row, 0)
        self.matrix_cmap_combo = QComboBox()
        self.matrix_cmap_combo.addItems(self._colormap_names)
        if self.matrix_cmap_combo.findText(matrix_cmap) >= 0:
            self.matrix_cmap_combo.setCurrentText(matrix_cmap)
        elif self.matrix_cmap_combo.count() > 0:
            self.matrix_cmap_combo.setCurrentIndex(0)
        form.addWidget(self.matrix_cmap_combo, row, 1, 1, 3)

        row += 1
        form.addWidget(QLabel("Default gradients color map"), row, 0)
        self.gradients_cmap_combo = QComboBox()
        self.gradients_cmap_combo.addItems(self._colormap_names)
        if self.gradients_cmap_combo.findText(gradient_cmap) >= 0:
            self.gradients_cmap_combo.setCurrentText(gradient_cmap)
        elif self.gradients_cmap_combo.count() > 0:
            self.gradients_cmap_combo.setCurrentIndex(0)
        form.addWidget(self.gradients_cmap_combo, row, 1, 1, 3)

        row += 1
        form.addWidget(QLabel("MATLAB executable"), row, 0)
        self.matlab_cmd_edit = QLineEdit(str(matlab_cmd or DEFAULT_MATLAB_CMD))
        form.addWidget(self.matlab_cmd_edit, row, 1, 1, 2)
        matlab_browse_button = QPushButton("Browse")
        matlab_browse_button.clicked.connect(self._browse_matlab_cmd)
        form.addWidget(matlab_browse_button, row, 3)

        row += 1
        form.addWidget(QLabel("NBS path"), row, 0)
        self.nbs_path_edit = QLineEdit(str(matlab_nbs_path or DEFAULT_MATLAB_NBS_PATH))
        form.addWidget(self.nbs_path_edit, row, 1, 1, 2)
        nbs_browse_button = QPushButton("Browse")
        nbs_browse_button.clicked.connect(self._browse_nbs_path)
        form.addWidget(nbs_browse_button, row, 3)

        layout.addLayout(form)
        layout.addStretch(1)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_row.addWidget(self.cancel_button)
        self.save_button = QPushButton("Save Configuration")
        self.save_button.clicked.connect(self.accept)
        button_row.addWidget(self.save_button)
        layout.addLayout(button_row)

    def _browse_matlab_cmd(self) -> None:
        current = self.matlab_cmd_edit.text().strip()
        start_dir = str(Path(current).parent) if current else str(Path.home())
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select MATLAB executable",
            start_dir,
            "All files (*)",
        )
        if selected:
            self.matlab_cmd_edit.setText(str(Path(selected).resolve()))

    def _browse_nbs_path(self) -> None:
        current = self.nbs_path_edit.text().strip()
        start_dir = str(Path(current).expanduser()) if current else str(Path.home())
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select NBS directory",
            start_dir,
        )
        if selected:
            self.nbs_path_edit.setText(str(Path(selected).resolve()))

    def values(self):
        return {
            "theme": self.theme_combo.currentText().strip(),
            "matrix_colormap": self.matrix_cmap_combo.currentText().strip(),
            "gradient_colormap": self.gradients_cmap_combo.currentText().strip(),
            "matlab_cmd": self.matlab_cmd_edit.text().strip(),
            "matlab_nbs_path": self.nbs_path_edit.text().strip(),
        }


class ConnectomeViewer(QMainWindow):
    _global_font_adjusted = False

    def __init__(self) -> None:
        super().__init__()
        self._increase_global_font_size()
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
        self._preferences = self._load_preferences()
        self._last_gradients = None
        self._active_parcellation_path = None
        self._active_parcellation_img = None
        self._active_parcellation_data = None
        self._surface_dialog = None
        self._nbs_dialog = None
        self._selector_dialog = None
        self._harmonize_dialog = None
        self._left_panel_saved_width = 320
        self._right_panel_saved_width = 240
        self._plot_title_full = ""
        self._plot_title_tooltip = ""
        self._theme_name = self._preferences["theme"]
        self._default_matrix_colormap = self._preferences["matrix_colormap"]
        self._default_gradient_colormap = self._preferences["gradient_colormap"]
        self._matlab_cmd_default = self._preferences["matlab_cmd"]
        self._matlab_nbs_path_default = self._preferences["matlab_nbs_path"]
        self._zoom_level = int(self._preferences.get("zoom_level", 0))
        self._base_app_font_point_size = None
        app = QApplication.instance()
        if app is not None:
            app_font = app.font()
            if app_font.pointSize() > 0:
                self._base_app_font_point_size = app_font.pointSize()
        self._compact_ui = False
        self._styled_groups = []
        self.setWindowTitle("Donald")
        self._set_window_icon()
        self.setAcceptDrops(True)
        self._hist_dialog = None
        self._build_ui()
        self._build_menu_bar()
        self._apply_zoom_level(self._zoom_level)

    @classmethod
    def _increase_global_font_size(cls) -> None:
        if cls._global_font_adjusted:
            return
        app = QApplication.instance()
        if app is None:
            return
        screen = app.primaryScreen()
        if screen is not None:
            geom = screen.availableGeometry()
            dpi = screen.logicalDotsPerInch()
            if geom.width() <= 1600 or geom.height() <= 900 or dpi >= 120.0:
                cls._global_font_adjusted = True
                return
        font = app.font()
        if font.pointSize() > 0:
            font.setPointSize(font.pointSize() + 1)
        elif font.pixelSize() > 0:
            font.setPixelSize(font.pixelSize() + 1)
        else:
            return
        app.setFont(font)
        cls._global_font_adjusted = True

    @staticmethod
    def _normalize_theme_name(theme_name: str) -> str:
        theme = str(theme_name or "Dark").strip().title()
        if theme not in {"Light", "Dark", "Teya", "Donald"}:
            theme = "Dark"
        return theme

    def _default_preferences(self):
        return {
            "theme": "Dark",
            "matrix_colormap": DEFAULT_COLORMAP,
            "gradient_colormap": DEFAULT_GRADIENT_COLORMAP,
            "matlab_cmd": DEFAULT_MATLAB_CMD,
            "matlab_nbs_path": DEFAULT_MATLAB_NBS_PATH,
            "zoom_level": 0,
        }

    def _available_colormap_names(self):
        custom = _discover_custom_cmaps()
        self._custom_cmaps = set(custom)
        names = []
        seen = set()
        for name in COLORMAPS + COLORBAR_BARS + custom:
            if name and name not in seen:
                names.append(name)
                seen.add(name)
        return names

    @staticmethod
    def _normalize_executable_path(value: str) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        path = Path(text).expanduser()
        if path.is_file():
            try:
                return str(path.resolve())
            except Exception:
                return str(path)
        resolved = shutil.which(text)
        if resolved:
            try:
                return str(Path(resolved).resolve())
            except Exception:
                return resolved
        return text

    @staticmethod
    def _normalize_directory_path(value: str) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        path = Path(text).expanduser()
        if path.is_dir():
            try:
                return str(path.resolve())
            except Exception:
                return str(path)
        return str(path)

    def _sanitize_preferences(self, raw_values):
        prefs = dict(self._default_preferences())
        if isinstance(raw_values, dict):
            prefs.update(raw_values)
        names = self._available_colormap_names()
        fallback_matrix = (
            DEFAULT_COLORMAP if DEFAULT_COLORMAP in names else (names[0] if names else DEFAULT_COLORMAP)
        )
        fallback_gradient = (
            DEFAULT_GRADIENT_COLORMAP
            if DEFAULT_GRADIENT_COLORMAP in names
            else (names[0] if names else DEFAULT_GRADIENT_COLORMAP)
        )
        prefs["theme"] = self._normalize_theme_name(prefs.get("theme"))
        matrix_name = str(prefs.get("matrix_colormap", fallback_matrix) or "").strip()
        gradient_name = str(prefs.get("gradient_colormap", fallback_gradient) or "").strip()
        prefs["matrix_colormap"] = matrix_name if matrix_name in names else fallback_matrix
        prefs["gradient_colormap"] = gradient_name if gradient_name in names else fallback_gradient
        matlab_cmd = str(prefs.get("matlab_cmd", DEFAULT_MATLAB_CMD) or "").strip()
        matlab_cmd = self._normalize_executable_path(matlab_cmd)
        default_matlab_cmd = self._normalize_executable_path(DEFAULT_MATLAB_CMD)
        prefs["matlab_cmd"] = matlab_cmd or default_matlab_cmd or ""
        matlab_nbs_path = str(prefs.get("matlab_nbs_path", DEFAULT_MATLAB_NBS_PATH) or "").strip()
        matlab_nbs_path = self._normalize_directory_path(matlab_nbs_path)
        default_nbs_path = self._normalize_directory_path(DEFAULT_MATLAB_NBS_PATH)
        prefs["matlab_nbs_path"] = matlab_nbs_path or default_nbs_path or ""
        try:
            zoom_level = int(prefs.get("zoom_level", 0))
        except Exception:
            zoom_level = 0
        prefs["zoom_level"] = max(-6, min(12, zoom_level))
        return prefs

    def _validate_nbs_preferences(self) -> bool:
        matlab_cmd = self._normalize_executable_path(self._matlab_cmd_default)
        nbs_path = self._normalize_directory_path(self._matlab_nbs_path_default)

        matlab_ok = False
        if matlab_cmd:
            matlab_path = Path(matlab_cmd)
            if matlab_path.is_file():
                matlab_ok = True
            elif shutil.which(matlab_cmd):
                matlab_ok = True

        nbs_ok = bool(nbs_path) and Path(nbs_path).is_dir()

        if matlab_ok and nbs_ok:
            self._matlab_cmd_default = matlab_cmd
            self._matlab_nbs_path_default = nbs_path
            return True

        message = (
            "NBS is not configured.\n"
            "Go to Settings > Preferences and select valid MATLAB executable and NBS paths."
        )
        try:
            QMessageBox.warning(self, "NBS Configuration Required", message)
        except Exception:
            pass
        self.statusBar().showMessage(
            "NBS blocked: configure MATLAB and NBS paths in Settings > Preferences."
        )
        return False

    def _load_preferences(self):
        if not CONFIG_PATH.exists():
            return self._sanitize_preferences({})
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            raw = {}
        return self._sanitize_preferences(raw)

    def _save_preferences(self) -> bool:
        payload = {
            "theme": self._theme_name,
            "matrix_colormap": self._default_matrix_colormap,
            "gradient_colormap": self._default_gradient_colormap,
            "matlab_cmd": self._matlab_cmd_default,
            "matlab_nbs_path": self._matlab_nbs_path_default,
            "zoom_level": int(self._zoom_level),
        }
        try:
            CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            return True
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to save preferences: {exc}")
            return False

    def _build_menu_bar(self) -> None:
        bar = self.menuBar()
        bar.clear()

        file_menu = bar.addMenu("File")
        self.add_npz_action = QAction("Add NPZ", self)
        self.add_npz_action.setShortcut("Ctrl+O")
        self.add_npz_action.triggered.connect(self._open_files)
        file_menu.addAction(self.add_npz_action)

        file_menu.addSeparator()
        close_action = QAction("Exit", self)
        close_action.setShortcut("Ctrl+Q")
        close_action.triggered.connect(self.close)
        file_menu.addAction(close_action)

        view_menu = bar.addMenu("View")
        zoom_in_action = QAction("Zoom In", self)
        zoom_in_action.setShortcut("Ctrl+=")
        zoom_in_action.triggered.connect(self._zoom_in)
        view_menu.addAction(zoom_in_action)

        zoom_out_action = QAction("Zoom Out", self)
        zoom_out_action.setShortcut("Ctrl+-")
        zoom_out_action.triggered.connect(self._zoom_out)
        view_menu.addAction(zoom_out_action)

        zoom_reset_action = QAction("Reset Zoom", self)
        zoom_reset_action.setShortcut("Ctrl+0")
        zoom_reset_action.triggered.connect(self._zoom_reset)
        view_menu.addAction(zoom_reset_action)

        analysis_menu = bar.addMenu("Analysis")
        self.compute_gradients_action = QAction("Compute Gradients", self)
        self.compute_gradients_action.triggered.connect(self._compute_gradients)
        analysis_menu.addAction(self.compute_gradients_action)

        self.nbs_prepare_action = QAction("NBS Prepare", self)
        self.nbs_prepare_action.triggered.connect(self._open_nbs_prepare_dialog)
        analysis_menu.addAction(self.nbs_prepare_action)

        self.harmonize_prepare_action = QAction("Harmonize Prepare", self)
        self.harmonize_prepare_action.triggered.connect(self._open_harmonize_prepare_dialog)
        analysis_menu.addAction(self.harmonize_prepare_action)

        settings_menu = bar.addMenu("Settings")
        preferences_action = QAction("Preferences", self)
        preferences_action.triggered.connect(self._open_preferences_dialog)
        settings_menu.addAction(preferences_action)

        self._update_nbs_prepare_button()

    def _apply_zoom_level(self, zoom_level: int, show_status: bool = False) -> None:
        try:
            level = int(zoom_level)
        except Exception:
            level = 0
        level = max(-6, min(12, level))
        self._zoom_level = level
        if self._base_app_font_point_size is None:
            return
        app = QApplication.instance()
        if app is None:
            return
        target_pt = max(7, self._base_app_font_point_size + level)
        font = app.font()
        if font.pointSize() <= 0:
            return
        font.setPointSize(target_pt)
        app.setFont(font)
        if show_status:
            self.statusBar().showMessage(f"UI zoom: {target_pt} pt")

    def _zoom_in(self) -> None:
        self._apply_zoom_level(self._zoom_level + 1, show_status=True)

    def _zoom_out(self) -> None:
        self._apply_zoom_level(self._zoom_level - 1, show_status=True)

    def _zoom_reset(self) -> None:
        self._apply_zoom_level(0, show_status=True)

    def _open_preferences_dialog(self) -> None:
        names = self._available_colormap_names()
        dialog = PreferencesDialog(
            theme_name=self._theme_name,
            matrix_cmap=self._default_matrix_colormap,
            gradient_cmap=self._default_gradient_colormap,
            matlab_cmd=self._matlab_cmd_default,
            matlab_nbs_path=self._matlab_nbs_path_default,
            colormap_names=names,
            parent=self,
        )
        if dialog.exec() != _dialog_accepted_code():
            return
        new_values = dialog.values()
        new_values["zoom_level"] = self._zoom_level
        prefs = self._sanitize_preferences(new_values)
        self._theme_name = prefs["theme"]
        self._default_matrix_colormap = prefs["matrix_colormap"]
        self._default_gradient_colormap = prefs["gradient_colormap"]
        self._matlab_cmd_default = prefs["matlab_cmd"]
        self._matlab_nbs_path_default = prefs["matlab_nbs_path"]
        self._preferences = prefs

        if getattr(self, "_nbs_dialog", None) is not None:
            try:
                self._nbs_dialog._matlab_cmd_default = self._matlab_cmd_default
                self._nbs_dialog._matlab_nbs_path_default = self._matlab_nbs_path_default
            except Exception:
                pass

        self._apply_theme(self._theme_name)
        self._reload_colormaps()
        if hasattr(self, "cmap_combo") and self.cmap_combo.findText(self._default_matrix_colormap) >= 0:
            self.cmap_combo.setCurrentText(self._default_matrix_colormap)
        if hasattr(self, "gradients_cmap_combo") and self.gradients_cmap_combo.findText(self._default_gradient_colormap) >= 0:
            self.gradients_cmap_combo.setCurrentText(self._default_gradient_colormap)

        if self._save_preferences():
            self.statusBar().showMessage(f"Preferences saved to {CONFIG_PATH}.")

    def _build_ui(self) -> None:
        screen = QApplication.primaryScreen()
        screen_geom = screen.availableGeometry() if screen is not None else None
        screen_dpi = screen.logicalDotsPerInch() if screen is not None else 96.0
        compact_ui = False
        if screen_geom is not None:
            compact_ui = screen_geom.width() <= 1600 or screen_geom.height() <= 900
        compact_ui = compact_ui or screen_dpi >= 120.0
        self._compact_ui = compact_ui

        central = QWidget(self)
        main_layout = QHBoxLayout(central)
        if compact_ui:
            main_layout.setContentsMargins(6, 6, 6, 6)
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setHandleWidth(8 if compact_ui else 10)
        self.main_splitter.setChildrenCollapsible(True)
        main_layout.addWidget(self.main_splitter)

        left_panel = QWidget()
        controls_layout = QVBoxLayout(left_panel)
        controls_layout.setSpacing(8 if compact_ui else 10)

        header = QLabel("Connectome matrices (.npz)")
        controls_layout.addWidget(header)

        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)
        self.export_button = QPushButton("Export Grid")
        self.export_button.clicked.connect(self._export_grid)
        export_layout.addWidget(self.export_button)

        self.write_matrix_button = QPushButton("Write to File")
        self.write_matrix_button.clicked.connect(self._write_selected_matrix_to_file)
        self.write_matrix_button.setEnabled(False)
        export_layout.addWidget(self.write_matrix_button)

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
        self.covar_value_edit = QLineEdit("")
        self.covar_value_edit.setPlaceholderText("Numeric value")
        # Hidden legacy selector widgets kept for compatibility with existing
        # internals; aggregation now runs via popup dialog.
        covar_label.hide()
        self.covar_combo.hide()
        covar_value_label.hide()
        self.covar_value_edit.hide()

        self.selector_prepare_button = QPushButton("Prepare")
        self.selector_prepare_button.clicked.connect(self._open_selector_prepare_dialog)
        self.selector_prepare_button.setEnabled(False)
        selector_layout.addWidget(self.selector_prepare_button)

        self.cmap_combo = QComboBox()
        self.cmap_combo.currentIndexChanged.connect(self._plot_selected)

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
        self.gradients_compute_button = QPushButton("Compute")
        self.gradients_compute_button.clicked.connect(self._compute_gradients)
        gradients_row.addWidget(self.gradients_compute_button)
        gradients_layout.addLayout(gradients_row)

        gradients_label = QLabel("N components:")
        gradients_layout.addWidget(gradients_label)

        gradients_spin_row = QHBoxLayout()
        self.gradients_spin = QSpinBox()
        self.gradients_spin.setRange(1, 10)
        self.gradients_spin.setValue(4)
        gradients_spin_row.addWidget(self.gradients_spin)
        gradients_layout.addLayout(gradients_spin_row)

        self.select_parcellation_button = QPushButton("Set Parcellation")
        self.select_parcellation_button.clicked.connect(self._select_parcellation_template)
        gradients_layout.addWidget(self.select_parcellation_button)

        self.parcellation_label = QLabel("Parcellation: none")
        self.parcellation_label.setWordWrap(True)
        gradients_layout.addWidget(self.parcellation_label)

        gradients_cmap_label = QLabel("3D colorbar:")
        gradients_layout.addWidget(gradients_cmap_label)
        self.gradients_cmap_combo = QComboBox()
        gradients_layout.addWidget(self.gradients_cmap_combo)
        self._reload_colormaps()

        self.gradients_progress = QProgressBar()
        self.gradients_progress.setRange(0, 1)
        self.gradients_progress.setValue(0)
        self.gradients_progress.setFormat("Idle")
        gradients_layout.addWidget(self.gradients_progress)

        gradients_actions_row = QHBoxLayout()
        self.gradients_save_button = QPushButton("Save")
        self.gradients_save_button.clicked.connect(self._save_gradients_projection)
        self.gradients_save_button.setEnabled(False)
        gradients_actions_row.addWidget(self.gradients_save_button)
        self.gradients_render_button = QPushButton("Render 3D")
        self.gradients_render_button.clicked.connect(self._render_gradients_3d)
        self.gradients_render_button.setEnabled(False)
        gradients_actions_row.addWidget(self.gradients_render_button)
        gradients_layout.addLayout(gradients_actions_row)

        nbs_group = QGroupBox("NBS")
        nbs_layout = QVBoxLayout(nbs_group)
        self.nbs_prepare_button = QPushButton("Prepare")
        self.nbs_prepare_button.clicked.connect(self._open_nbs_prepare_dialog)
        self.nbs_prepare_button.setEnabled(False)
        nbs_layout.addWidget(self.nbs_prepare_button)

        harmonize_group = QGroupBox("Harmonize")
        harmonize_layout = QVBoxLayout(harmonize_group)
        self.harmonize_prepare_button = QPushButton("Prepare")
        self.harmonize_prepare_button.clicked.connect(self._open_harmonize_prepare_dialog)
        self.harmonize_prepare_button.setEnabled(False)
        harmonize_layout.addWidget(self.harmonize_prepare_button)

        self._styled_groups = [
            list_group,
            selector_group,
            export_group,
            gradients_group,
            nbs_group,
            harmonize_group,
        ]

        controls_layout.addWidget(list_group)
        controls_layout.addWidget(selector_group)
        controls_layout.addWidget(export_group)

        controls_layout.addStretch(1)

        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setSpacing(6 if compact_ui else 8)

        plot_toolbar = QHBoxLayout()
        plot_toolbar.setSpacing(6 if compact_ui else 8)
        self.left_sidebar_button = QPushButton("◀")
        self.left_sidebar_button.setFixedWidth(30)
        self.left_sidebar_button.clicked.connect(self._toggle_left_sidebar)
        plot_toolbar.addWidget(self.left_sidebar_button)

        self.plot_title_label = QLabel("")
        title_font = self.plot_title_label.font()
        if title_font.pointSize() > 0:
            title_font.setPointSize(max(title_font.pointSize() - 1, 9))
        self.plot_title_label.setFont(title_font)
        self.plot_title_label.setToolTip("")
        plot_toolbar.addWidget(self.plot_title_label, 1)

        plot_toolbar.addWidget(QLabel("Color map:"))
        self.cmap_combo.setMinimumWidth(150 if compact_ui else 190)
        plot_toolbar.addWidget(self.cmap_combo)

        plot_toolbar.addWidget(QLabel("Theme:"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light", "Teya", "Donald"])
        self.theme_combo.setCurrentText(self._theme_name)
        self.theme_combo.currentTextChanged.connect(self._on_theme_changed)
        self.theme_combo.setMinimumWidth(90 if compact_ui else 110)
        plot_toolbar.addWidget(self.theme_combo)

        plot_toolbar.addWidget(QLabel("Display scaling:"))
        self.display_auto_check = QCheckBox("Auto")
        self.display_auto_check.setChecked(True)
        self.display_auto_check.stateChanged.connect(self._on_display_scaling_changed)
        plot_toolbar.addWidget(self.display_auto_check)

        self.display_min_edit = QLineEdit("")
        self.display_min_edit.setPlaceholderText("Min")
        self.display_min_edit.setFixedWidth(78 if compact_ui else 90)
        self.display_min_edit.setEnabled(False)
        self.display_min_edit.editingFinished.connect(self._on_display_scaling_changed)
        plot_toolbar.addWidget(self.display_min_edit)

        self.display_max_edit = QLineEdit("")
        self.display_max_edit.setPlaceholderText("Max")
        self.display_max_edit.setFixedWidth(78 if compact_ui else 90)
        self.display_max_edit.setEnabled(False)
        self.display_max_edit.editingFinished.connect(self._on_display_scaling_changed)
        plot_toolbar.addWidget(self.display_max_edit)

        self.right_sidebar_button = QPushButton("▶")
        self.right_sidebar_button.setFixedWidth(30)
        self.right_sidebar_button.clicked.connect(self._toggle_right_sidebar)
        plot_toolbar.addWidget(self.right_sidebar_button)
        center_layout.addLayout(plot_toolbar)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.mpl_connect("motion_notify_event", self._on_hover)
        center_layout.addWidget(self.canvas, 1)

        self.hover_label = QLabel("")
        self.hover_label.setWordWrap(True)
        center_layout.addWidget(self.hover_label, 0)

        right_panel = QWidget()
        right_panel_layout = QVBoxLayout(right_panel)
        right_panel_layout.addWidget(gradients_group)
        right_panel_layout.addWidget(nbs_group)
        right_panel_layout.addWidget(harmonize_group)
        right_panel_layout.addStretch(1)
        left_panel.setMinimumWidth(0)
        right_panel.setMinimumWidth(0)

        self.main_splitter.addWidget(left_panel)
        self.main_splitter.addWidget(center_panel)
        self.main_splitter.addWidget(right_panel)
        self.main_splitter.splitterMoved.connect(self._on_splitter_moved)
        self.main_splitter.setCollapsible(0, True)
        self.main_splitter.setCollapsible(1, False)
        self.main_splitter.setCollapsible(2, True)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)
        if screen_geom is not None:
            total_w = max(screen_geom.width(), 1000)
            left_w = int(total_w * (0.31 if compact_ui else 0.26))
            right_w = int(total_w * (0.15 if compact_ui else 0.18))
            center_w = max(total_w - left_w - right_w, 480)
            self.main_splitter.setSizes([left_w, center_w, right_w])
        else:
            self.main_splitter.setSizes([380, 900, 240])
        current_sizes = self.main_splitter.sizes()
        if len(current_sizes) == 3:
            if current_sizes[0] > 0:
                self._left_panel_saved_width = current_sizes[0]
            if current_sizes[2] > 0:
                self._right_panel_saved_width = current_sizes[2]

        self.setCentralWidget(central)
        self._apply_theme(self._theme_name)
        self._update_parcellation_label()
        self._refresh_sidebar_toggle_buttons()
        self._set_plot_title("")
        self._update_nbs_prepare_button()
        self.statusBar().showMessage("Ready.")

    def _set_window_icon(self) -> None:
        icon = _load_app_icon()
        if icon.isNull():
            return
        self.setWindowIcon(icon)
        app = QApplication.instance()
        if app is not None:
            app.setWindowIcon(icon)

    def _on_theme_changed(self, theme_name: str) -> None:
        self._apply_theme(theme_name)

    def _theme_icon_color(self) -> str:
        if self._theme_name == "Dark":
            return "#e8edf7"
        if self._theme_name == "Teya":
            return "#06b6b0"
        if self._theme_name == "Donald":
            return "#ffffff"
        return "#1f2937"

    def _base_theme_stylesheet(self) -> str:
        if self._theme_name == "Dark":
            return (
                "QMainWindow, QWidget { background-color: #1f2430; color: #e5e7eb; }"
                "QLabel { color: #e5e7eb; }"
                "QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QListWidget, QTableWidget, QProgressBar {"
                "background: #2a3140; color: #e5e7eb; border: 1px solid #556070; border-radius: 5px; padding: 4px;"
                "}"
                "QPushButton {"
                "background: #2d3646; color: #e5e7eb; border: 1px solid #5f6d82; border-radius: 6px; padding: 5px 10px;"
                "}"
                "QPushButton:hover { background: #374256; }"
                "QPushButton:pressed { background: #2b3444; }"
                "QPushButton:disabled { color: #8e98a8; background: #252c38; border-color: #464f5e; }"
                "QListWidget::item:selected, QTableWidget::item:selected { background: #3b82f6; color: #ffffff; }"
                "QComboBox QAbstractItemView { background: #2a3140; color: #e5e7eb; selection-background-color: #3b82f6; }"
                "QStatusBar { background: #242a35; color: #e5e7eb; border-top: 1px solid #3d4556; }"
                "QToolTip { background-color: #101722; color: #e5e7eb; border: 1px solid #4b5563; }"
            )
        if self._theme_name == "Teya":
            return (
                "QMainWindow, QWidget { background-color: #ffd0e5; color: #0b7f7a; }"
                "QLabel { color: #0b7f7a; }"
                "QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QListWidget, QTableWidget, QProgressBar {"
                "background: #ffe6f1; color: #0b7f7a; border: 1px solid #1db8b2; border-radius: 5px; padding: 4px;"
                "}"
                "QPushButton {"
                "background: #ffc0dc; color: #0b7f7a; border: 1px solid #1db8b2; border-radius: 6px; padding: 5px 10px;"
                "}"
                "QPushButton:hover { background: #ffb1d5; }"
                "QPushButton:pressed { background: #ffa3cd; }"
                "QPushButton:disabled { color: #68a9a6; background: #ffd9ea; border-color: #87cfcb; }"
                "QListWidget::item:selected, QTableWidget::item:selected { background: #2ecfc9; color: #073f3c; }"
                "QComboBox QAbstractItemView { background: #ffe6f1; color: #0b7f7a; selection-background-color: #2ecfc9; }"
                "QStatusBar { background: #ffbddb; color: #0b7f7a; border-top: 1px solid #1db8b2; }"
                "QToolTip { background-color: #ffeef6; color: #0b7f7a; border: 1px solid #1db8b2; }"
            )
        if self._theme_name == "Donald":
            return (
                "QMainWindow, QWidget { background-color: #d97706; color: #ffffff; }"
                "QLabel { color: #ffffff; }"
                "QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QListWidget, QTableWidget, QProgressBar {"
                "background: #c96a04; color: #ffffff; border: 1px solid #f3a451; border-radius: 5px; padding: 4px;"
                "}"
                "QPushButton {"
                "background: #b85f00; color: #ffffff; border: 1px solid #f3a451; border-radius: 6px; padding: 5px 10px;"
                "}"
                "QPushButton:hover { background: #c76b06; }"
                "QPushButton:pressed { background: #a85400; }"
                "QPushButton:disabled { color: #ffe3be; background: #d58933; border-color: #f0b97c; }"
                "QListWidget::item:selected, QTableWidget::item:selected { background: #2563eb; color: #ffffff; }"
                "QComboBox QAbstractItemView { background: #c96a04; color: #ffffff; selection-background-color: #2563eb; }"
                "QStatusBar { background: #b85f00; color: #ffffff; border-top: 1px solid #f3a451; }"
                "QToolTip { background-color: #f08c19; color: #ffffff; border: 1px solid #ffd19e; }"
            )
        return (
            "QMainWindow, QWidget { background-color: #f3f5f8; color: #1f2937; }"
            "QLabel { color: #1f2937; }"
            "QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QListWidget, QTableWidget, QProgressBar {"
            "background: #ffffff; color: #111827; border: 1px solid #c9d0da; border-radius: 5px; padding: 4px;"
            "}"
            "QPushButton {"
            "background: #ffffff; color: #1f2937; border: 1px solid #b7c0cc; border-radius: 6px; padding: 5px 10px;"
            "}"
            "QPushButton:hover { background: #edf2f7; }"
            "QPushButton:pressed { background: #e6ebf2; }"
            "QPushButton:disabled { color: #9099a5; background: #edf1f5; border-color: #d2d9e2; }"
            "QListWidget::item:selected, QTableWidget::item:selected { background: #2563eb; color: #ffffff; }"
            "QComboBox QAbstractItemView { background: #ffffff; color: #111827; selection-background-color: #2563eb; }"
            "QStatusBar { background: #e8edf3; color: #1f2937; border-top: 1px solid #cfd7e2; }"
            "QToolTip { background-color: #ffffff; color: #1f2937; border: 1px solid #c9d0da; }"
        )

    def _groupbox_stylesheet(self) -> str:
        font_pt = "10pt" if self._compact_ui else "11pt"
        margin_top = "8px" if self._compact_ui else "10px"
        title_left = "8px" if self._compact_ui else "10px"
        padding_top = "4px" if self._compact_ui else "6px"
        if self._theme_name == "Dark":
            border_color = "#4f5a6b"
            bg_color = "#242b36"
            text_color = "#e5e7eb"
        elif self._theme_name == "Teya":
            border_color = "#1db8b2"
            bg_color = "#ffe0ef"
            text_color = "#0b7f7a"
        elif self._theme_name == "Donald":
            border_color = "#f3a451"
            bg_color = "#c96a04"
            text_color = "#ffffff"
        else:
            border_color = "#c9ced6"
            bg_color = "#fcfcfc"
            text_color = "#1f2937"
        return (
            "QGroupBox {"
            "font-weight: 600;"
            f"font-size: {font_pt};"
            f"color: {text_color};"
            f"border: 1px solid {border_color};"
            "border-radius: 6px;"
            f"margin-top: {margin_top};"
            f"padding-top: {padding_top};"
            f"background: {bg_color};"
            "}"
            "QGroupBox::title {"
            "subcontrol-origin: margin;"
            f"left: {title_left};"
            "padding: 0 4px;"
            "}"
        )

    def _apply_groupbox_style(self) -> None:
        style = self._groupbox_stylesheet()
        for group in self._styled_groups:
            if group is not None:
                group.setStyleSheet(style)

    def _apply_theme(self, theme_name: str) -> None:
        theme = (theme_name or "Dark").strip().title()
        if theme not in {"Light", "Dark", "Teya", "Donald"}:
            theme = "Dark"
        self._theme_name = theme
        if hasattr(self, "theme_combo") and self.theme_combo.currentText() != theme:
            self.theme_combo.blockSignals(True)
            self.theme_combo.setCurrentText(theme)
            self.theme_combo.blockSignals(False)

        self.setStyleSheet(self._base_theme_stylesheet())
        self._apply_groupbox_style()
        if hasattr(self, "plot_title_label"):
            if theme == "Dark":
                plot_color = "#e5e7eb"
            elif theme == "Teya":
                plot_color = "#0b7f7a"
            elif theme == "Donald":
                plot_color = "#ffffff"
            else:
                plot_color = "#2f3640"
            self.plot_title_label.setStyleSheet(f"color: {plot_color};")
        self._apply_button_icons(
            compact_ui=self._compact_ui,
            color=self._theme_icon_color(),
        )
        if getattr(self, "_nbs_dialog", None) is not None and hasattr(self._nbs_dialog, "set_theme"):
            try:
                self._nbs_dialog.set_theme(theme)
            except Exception:
                pass
        if getattr(self, "_selector_dialog", None) is not None and hasattr(self._selector_dialog, "set_theme"):
            try:
                self._selector_dialog.set_theme(theme)
            except Exception:
                pass
        if getattr(self, "_harmonize_dialog", None) is not None and hasattr(self._harmonize_dialog, "set_theme"):
            try:
                self._harmonize_dialog.set_theme(theme)
            except Exception:
                pass
        if getattr(self, "_surface_dialog", None) is not None and hasattr(
            self._surface_dialog, "set_theme"
        ):
            try:
                self._surface_dialog.set_theme(theme)
            except Exception:
                pass

    def _svg_icon(self, filename: str, color: str = None) -> QIcon:
        icon_path = Path(__file__).with_name("icons") / "svg" / filename
        if not icon_path.exists():
            return QIcon()

        svg_text = None
        if color:
            try:
                svg_text = icon_path.read_text(encoding="utf-8")
                svg_text = svg_text.replace("currentColor", color).replace("currentcolor", color)
            except Exception:
                svg_text = None

        if svg_text:
            try:
                pixmap = QPixmap()
                if pixmap.loadFromData(svg_text.encode("utf-8"), "SVG"):
                    return QIcon(pixmap)
            except Exception:
                pass
            if QSvgRenderer is not None and QPainter is not None:
                try:
                    renderer = QSvgRenderer(svg_text.encode("utf-8"))
                    if renderer.isValid():
                        size = renderer.defaultSize()
                        if not size.isValid():
                            size = QSize(24, 24)
                        pixmap = QPixmap(size)
                        transparent = Qt.GlobalColor.transparent if QT_LIB == 6 else Qt.transparent
                        pixmap.fill(transparent)
                        painter = QPainter(pixmap)
                        renderer.render(painter)
                        painter.end()
                        return QIcon(pixmap)
                except Exception:
                    pass

        icon = QIcon(str(icon_path))
        if not icon.isNull():
            return icon

        if QSvgRenderer is not None and QPainter is not None:
            try:
                renderer = QSvgRenderer(str(icon_path))
                if renderer.isValid():
                    size = renderer.defaultSize()
                    if not size.isValid():
                        size = QSize(24, 24)
                    pixmap = QPixmap(size)
                    transparent = Qt.GlobalColor.transparent if QT_LIB == 6 else Qt.transparent
                    pixmap.fill(transparent)
                    painter = QPainter(pixmap)
                    renderer.render(painter)
                    painter.end()
                    return QIcon(pixmap)
            except Exception:
                pass
        return QIcon()

    def _apply_button_icons(self, compact_ui: bool = False, color: str = None) -> None:
        icon_size = QSize(16, 16) if compact_ui else QSize(18, 18)
        mapping = [
            (getattr(self, "add_button", None), "folder_plus.svg"),
            (getattr(self, "remove_button", None), "trash.svg"),
            (getattr(self, "clear_button", None), "broom_clear.svg"),
            (getattr(self, "hist_button", None), "histogram.svg"),
            (getattr(self, "export_button", None), "export_grid.svg"),
            (getattr(self, "write_matrix_button", None), "save_disk.svg"),
            (getattr(self, "move_up_button", None), "arrow_up.svg"),
            (getattr(self, "move_down_button", None), "arrow_down.svg"),
            (getattr(self, "gradients_compute_button", None), "play_circle_compute.svg"),
            (getattr(self, "gradients_save_button", None), "save_disk.svg"),
            (getattr(self, "gradients_render_button", None), "cube_3d.svg"),
            (getattr(self, "nbs_prepare_button", None), "wrench_prepare.svg"),
            (getattr(self, "harmonize_prepare_button", None), "wrench_prepare.svg"),
            (getattr(self, "select_parcellation_button", None), "settings_sliders.svg"),
            (getattr(self, "selector_prepare_button", None), "filter_threshold.svg"),
            (getattr(self, "sample_add_button", None), "folder_plus.svg"),
        ]
        for button, icon_name in mapping:
            if button is None:
                continue
            icon = self._svg_icon(icon_name, color=color)
            if icon.isNull():
                continue
            button.setIcon(icon)
            button.setIconSize(icon_size)

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

    def _current_source_path(self):
        entry = self._current_entry()
        if entry is None:
            return None
        source_path = entry.get("source_path", entry.get("path"))
        if not source_path:
            return None
        return Path(source_path)

    def _default_dialog_dir(self) -> Path:
        source_path = self._current_source_path()
        if source_path is not None:
            return source_path.parent
        return Path.cwd()

    def _default_parcellation_dir(self) -> Path:
        if DEFAULT_PARCELLATION_DIR.exists():
            return DEFAULT_PARCELLATION_DIR
        return ROOTDIR

    def _update_parcellation_label(self) -> None:
        if self._active_parcellation_path is None:
            self.parcellation_label.setText("Parcellation: none")
        else:
            self.parcellation_label.setText(f"Parcellation: {self._active_parcellation_path.name}")

    def _select_parcellation_template(self) -> bool:
        template_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select 3D parcellation template",
            str(self._default_parcellation_dir()),
            "NIfTI files (*.nii *.nii.gz);;All files (*)",
        )
        if not template_path:
            return False
        return self._set_active_parcellation(Path(template_path))

    def _set_active_parcellation(self, template_path: Path) -> bool:
        try:
            import nibabel as nib
        except Exception as exc:
            self.statusBar().showMessage(f"nibabel not available: {exc}")
            return False
        try:
            template_img = nib.load(str(template_path))
            template_data = np.asarray(template_img.get_fdata(), dtype=int)
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to load template: {exc}")
            return False
        if template_data.ndim != 3:
            self.statusBar().showMessage("Template must be a 3D NIfTI image.")
            return False
        self._active_parcellation_path = Path(template_path)
        self._active_parcellation_img = template_img
        self._active_parcellation_data = template_data
        self._update_parcellation_label()
        return True

    def _reset_gradients_output(self) -> None:
        self._last_gradients = None
        if hasattr(self, "gradients_save_button"):
            self.gradients_save_button.setEnabled(False)
        if hasattr(self, "gradients_render_button"):
            self.gradients_render_button.setEnabled(False)
        if hasattr(self, "gradients_progress"):
            self.gradients_progress.setRange(0, 1)
            self.gradients_progress.setValue(0)
            self.gradients_progress.setFormat("Idle")

    def _current_nbs_source(self):
        entry = self._current_entry()
        if entry is None or entry.get("kind") != "file":
            return None
        source_path = entry.get("path")
        if source_path is None:
            return None
        source_path = Path(source_path)
        if not source_path.exists():
            return None
        key = self._ensure_entry_key(entry)
        if not key:
            return None

        stack_axis = entry.get("stack_axis")
        stack_len = entry.get("stack_len")
        if stack_axis is not None and stack_len is not None and int(stack_len) > 1:
            return {"path": source_path, "key": key, "stack_len": int(stack_len)}

        try:
            raw = _load_matrix_from_npz(source_path, key, average=False)
        except Exception:
            return None
        if raw.ndim != 3:
            return None
        axis = _stack_axis(raw.shape)
        if axis is None or int(raw.shape[axis]) <= 1:
            return None
        return {"path": source_path, "key": key, "stack_len": int(raw.shape[axis])}

    def _update_nbs_prepare_button(self) -> None:
        enabled = self._current_nbs_source() is not None
        if not hasattr(self, "nbs_prepare_button"):
            if hasattr(self, "nbs_prepare_action"):
                self.nbs_prepare_action.setEnabled(enabled)
            self._update_selector_prepare_button()
            self._update_harmonize_prepare_button()
            return
        self.nbs_prepare_button.setEnabled(enabled)
        if hasattr(self, "nbs_prepare_action"):
            self.nbs_prepare_action.setEnabled(enabled)
        self._update_selector_prepare_button()
        self._update_harmonize_prepare_button()

    def _current_selector_source(self):
        return self._current_nbs_source()

    def _current_harmonize_source(self):
        return self._current_nbs_source()

    def _update_selector_prepare_button(self) -> None:
        if not hasattr(self, "selector_prepare_button"):
            return
        source = self._current_selector_source()
        enabled = False
        if source is not None:
            source_path = source["path"]
            info = self._covars_cache.get(source_path)
            if info is None:
                info = _load_covars_info(source_path)
                self._covars_cache[source_path] = info
            enabled = bool(_covars_columns(info))
        self.selector_prepare_button.setEnabled(enabled)
        self._update_harmonize_prepare_button()
        self._update_write_to_file_button()

    def _update_harmonize_prepare_button(self) -> None:
        if not hasattr(self, "harmonize_prepare_button"):
            if hasattr(self, "harmonize_prepare_action"):
                self.harmonize_prepare_action.setEnabled(False)
            return
        source = self._current_harmonize_source()
        enabled = False
        if source is not None:
            source_path = source["path"]
            info = self._covars_cache.get(source_path)
            if info is None:
                info = _load_covars_info(source_path)
                self._covars_cache[source_path] = info
            enabled = bool(_covars_columns(info))
        self.harmonize_prepare_button.setEnabled(enabled)
        if hasattr(self, "harmonize_prepare_action"):
            self.harmonize_prepare_action.setEnabled(enabled)

    def _update_write_to_file_button(self) -> None:
        if not hasattr(self, "write_matrix_button"):
            return
        entry = self._current_entry()
        self.write_matrix_button.setEnabled(entry is not None)

    def _open_nbs_prepare_dialog(self) -> None:
        source = self._current_nbs_source()
        if source is None:
            self.statusBar().showMessage(
                "NBS Prepare requires a file-based matrix stack (multiple matrices)."
            )
            return
        if not self._validate_nbs_preferences():
            return

        source_path = source["path"]
        covars_info = self._covars_cache.get(source_path)
        if covars_info is None:
            covars_info = _load_covars_info(source_path)
            self._covars_cache[source_path] = covars_info
        if covars_info is None:
            self.statusBar().showMessage("Covars not found in selected file.")
            return

        covars_len = 0
        df = covars_info.get("df")
        if df is not None:
            covars_len = len(df)
        else:
            data = covars_info.get("data")
            if data is not None:
                covars_len = len(data)
        if covars_len and covars_len != source["stack_len"]:
            self.statusBar().showMessage(
                "Covars length does not match matrix stack size."
            )
            return

        try:
            from window.nbs_prepare import NBSPrepareDialog
        except Exception:
            try:
                from mrsi_viewer.window.nbs_prepare import NBSPrepareDialog
            except Exception as exc:
                self.statusBar().showMessage(f"Failed to open NBS window: {exc}")
                return

        self._nbs_dialog = NBSPrepareDialog(
            covars_info=covars_info,
            source_path=source_path,
            matrix_key=source["key"],
            matlab_cmd_default=self._matlab_cmd_default,
            matlab_nbs_path_default=self._matlab_nbs_path_default,
            theme_name=self._theme_name,
            parent=self,
        )
        self._nbs_dialog.show()
        self.statusBar().showMessage(
            f"Opened NBS Prepare ({source_path.name}, key={source['key']})."
        )

    def _open_selector_prepare_dialog(self) -> None:
        source = self._current_selector_source()
        if source is None:
            self.statusBar().showMessage(
                "Selector Prepare requires a file-based matrix stack (multiple matrices)."
            )
            return

        source_path = source["path"]
        covars_info = self._covars_cache.get(source_path)
        if covars_info is None:
            covars_info = _load_covars_info(source_path)
            self._covars_cache[source_path] = covars_info
        if covars_info is None:
            self.statusBar().showMessage("Covars not found in selected file.")
            return

        covars_len = 0
        df = covars_info.get("df")
        if df is not None:
            covars_len = len(df)
        else:
            data = covars_info.get("data")
            if data is not None:
                covars_len = len(data)
        if covars_len and covars_len != source["stack_len"]:
            self.statusBar().showMessage("Covars length does not match matrix stack size.")
            return

        try:
            from window.selector_prepare import SelectorPrepareDialog
        except Exception:
            try:
                from mrsi_viewer.window.selector_prepare import SelectorPrepareDialog
            except Exception as exc:
                self.statusBar().showMessage(f"Failed to open Selector window: {exc}")
                return

        self._selector_dialog = SelectorPrepareDialog(
            covars_info=covars_info,
            source_path=source_path,
            matrix_key=source["key"],
            theme_name=self._theme_name,
            export_callback=self._import_selector_aggregate,
            parent=self,
        )
        self._selector_dialog.show()
        self.statusBar().showMessage(
            f"Opened Selector Prepare ({source_path.name}, key={source['key']})."
        )

    def _open_harmonize_prepare_dialog(self) -> None:
        source = self._current_harmonize_source()
        if source is None:
            self.statusBar().showMessage(
                "Harmonize Prepare requires a file-based matrix stack (multiple matrices)."
            )
            return

        source_path = source["path"]
        covars_info = self._covars_cache.get(source_path)
        if covars_info is None:
            covars_info = _load_covars_info(source_path)
            self._covars_cache[source_path] = covars_info
        if covars_info is None:
            self.statusBar().showMessage("Covars not found in selected file.")
            return

        covars_len = 0
        df = covars_info.get("df")
        if df is not None:
            covars_len = len(df)
        else:
            data = covars_info.get("data")
            if data is not None:
                covars_len = len(data)
        if covars_len and covars_len != source["stack_len"]:
            self.statusBar().showMessage("Covars length does not match matrix stack size.")
            return

        try:
            from window.harmonize_prepare import HarmonizePrepareDialog
        except Exception:
            try:
                from mrsi_viewer.window.harmonize_prepare import HarmonizePrepareDialog
            except Exception as exc:
                self.statusBar().showMessage(f"Failed to open Harmonize window: {exc}")
                return

        self._harmonize_dialog = HarmonizePrepareDialog(
            covars_info=covars_info,
            source_path=source_path,
            matrix_key=source["key"],
            theme_name=self._theme_name,
            export_callback=self._import_harmonized_result,
            parent=self,
        )
        self._harmonize_dialog.show()
        self.statusBar().showMessage(
            f"Opened Harmonize Prepare ({source_path.name}, key={source['key']})."
        )

    def _import_harmonized_result(self, payload) -> bool:
        output_raw = str(payload.get("output_path") or "").strip()
        if not output_raw:
            self.statusBar().showMessage("Harmonize export payload missing output path.")
            return False
        output_path = Path(output_raw)
        if not output_path.is_file():
            self.statusBar().showMessage(f"Harmonized file not found: {output_path}")
            return False

        self._add_files([str(output_path)])
        target_id = self._file_entry_id(output_path)
        selected = False
        for row in range(self.file_list.count()):
            item = self.file_list.item(row)
            if item is None:
                continue
            if item.data(USER_ROLE) == target_id:
                self.file_list.setCurrentItem(item)
                selected = True
                break
        if not selected:
            self.statusBar().showMessage(
                f"Harmonized file saved to {output_path.name} (already in workspace)."
            )
        else:
            self.statusBar().showMessage(
                f"Imported harmonized matrix stack: {output_path.name}."
            )
        return True

    def _import_selector_aggregate(self, payload) -> bool:
        try:
            matrix = np.asarray(payload.get("matrix"), dtype=float)
        except Exception as exc:
            self.statusBar().showMessage(f"Invalid aggregated matrix payload: {exc}")
            return False
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            self.statusBar().showMessage("Aggregated matrix must be square.")
            return False

        source_path_raw = str(payload.get("source_path", "")).strip()
        source_path = Path(source_path_raw) if source_path_raw else None
        source_name = source_path.name if source_path is not None else "matrix"
        matrix_key = str(payload.get("matrix_key") or "matrix")
        method = str(payload.get("method") or "mean").strip().lower()
        method_label = "zfisher" if method == "zfisher" else "avg"
        selected_rows = list(payload.get("selected_rows") or [])
        n_selected = len(selected_rows)
        n_total = int(payload.get("n_total_rows") or n_selected)
        filter_covar = str(payload.get("filter_covar") or "").strip()
        filter_values = [str(v) for v in (payload.get("filter_values") or []) if str(v) != ""]

        if filter_covar and filter_values:
            filter_text = f"{filter_covar}={','.join(filter_values)}"
        else:
            filter_text = f"n={n_selected}/{n_total}"
        label = f"{method_label} {matrix_key} [{filter_text}] ({source_name})"

        derived_id = self._new_derived_id()
        self._entries[derived_id] = {
            "id": derived_id,
            "kind": "derived",
            "matrix": matrix,
            "source_path": source_path,
            "selected_key": matrix_key,
            "sample_index": None,
            "auto_title": True,
            "label": label,
            "aggregation_method": method,
            "selected_rows": selected_rows,
            "covar_name": filter_covar or None,
            "covar_value": ",".join(filter_values) if filter_values else None,
        }
        self.titles[derived_id] = label
        item = QListWidgetItem(label)
        item.setData(USER_ROLE, derived_id)
        self.file_list.addItem(item)
        self.file_list.setCurrentItem(item)
        self.statusBar().showMessage(f"Imported aggregated matrix ({method_label}).")
        return True

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
        else:
            self._update_nbs_prepare_button()

    def _clear_files(self) -> None:
        self._entries.clear()
        self.titles.clear()
        self.file_list.clear()
        self._clear_plot()
        self._update_nbs_prepare_button()
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
        columns = _covars_columns(info)
        # Keep legacy fields populated for compatibility with existing metadata.
        self.covar_combo.blockSignals(True)
        self.covar_combo.clear()
        for col in columns:
            self.covar_combo.addItem(col)
        covar_name = entry.get("covar_name")
        if covar_name and self.covar_combo.findText(covar_name) >= 0:
            self.covar_combo.setCurrentText(covar_name)
        self.covar_combo.blockSignals(False)
        enabled = self.covar_combo.count() > 0
        self.covar_combo.setEnabled(enabled)
        self._update_selector_prepare_button()

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
        names = self._available_colormap_names()
        current = self.cmap_combo.currentText() if hasattr(self, "cmap_combo") else ""
        current_3d = (
            self.gradients_cmap_combo.currentText()
            if hasattr(self, "gradients_cmap_combo")
            else ""
        )
        self.cmap_combo.blockSignals(True)
        self.cmap_combo.clear()
        self.cmap_combo.addItems(names)
        if current and current in names:
            self.cmap_combo.setCurrentText(current)
        elif self._default_matrix_colormap in names:
            self.cmap_combo.setCurrentText(self._default_matrix_colormap)
        elif DEFAULT_COLORMAP in names:
            self.cmap_combo.setCurrentText(DEFAULT_COLORMAP)
        elif names:
            self.cmap_combo.setCurrentIndex(0)
        self.cmap_combo.blockSignals(False)

        if hasattr(self, "gradients_cmap_combo"):
            self.gradients_cmap_combo.blockSignals(True)
            self.gradients_cmap_combo.clear()
            self.gradients_cmap_combo.addItems(names)
            if current_3d and current_3d in names:
                self.gradients_cmap_combo.setCurrentText(current_3d)
            elif self._default_gradient_colormap in names:
                self.gradients_cmap_combo.setCurrentText(self._default_gradient_colormap)
            elif "spectrum_fsl" in names:
                self.gradients_cmap_combo.setCurrentText("spectrum_fsl")
            elif names:
                self.gradients_cmap_combo.setCurrentIndex(0)
            self.gradients_cmap_combo.blockSignals(False)

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

    def _set_plot_title(self, full_title: str, tooltip_text: str = "") -> None:
        self._plot_title_full = str(full_title or "")
        self._plot_title_tooltip = str(tooltip_text or full_title or "")
        self._update_plot_title_label()

    def _update_plot_title_label(self) -> None:
        if not hasattr(self, "plot_title_label"):
            return
        full_title = self._plot_title_full
        self.plot_title_label.setToolTip(self._plot_title_tooltip)
        if not full_title:
            self.plot_title_label.setText("")
            return
        metrics = QFontMetrics(self.plot_title_label.font())
        avail = max(self.plot_title_label.width() - 4, 60)
        elided = metrics.elidedText(full_title, Qt.ElideRight, avail)
        self.plot_title_label.setText(elided)

    def _current_display_limits(self):
        if not hasattr(self, "display_auto_check") or self.display_auto_check.isChecked():
            return None, None, None
        vmin = None
        vmax = None
        min_text = self.display_min_edit.text().strip() if hasattr(self, "display_min_edit") else ""
        max_text = self.display_max_edit.text().strip() if hasattr(self, "display_max_edit") else ""
        if min_text:
            try:
                vmin = float(min_text)
            except ValueError:
                return None, None, "Display min must be numeric."
        if max_text:
            try:
                vmax = float(max_text)
            except ValueError:
                return None, None, "Display max must be numeric."
        if vmin is not None and vmax is not None and vmin >= vmax:
            return None, None, "Display min must be smaller than max."
        return vmin, vmax, None

    def _on_display_scaling_changed(self, *_args) -> None:
        auto_scale = self.display_auto_check.isChecked()
        self.display_min_edit.setEnabled(not auto_scale)
        self.display_max_edit.setEnabled(not auto_scale)
        if self._current_entry_id() is not None:
            self._plot_selected()

    def _on_splitter_moved(self, *_args) -> None:
        sizes = self.main_splitter.sizes()
        if len(sizes) == 3:
            if sizes[0] > 0:
                self._left_panel_saved_width = sizes[0]
            if sizes[2] > 0:
                self._right_panel_saved_width = sizes[2]
        self._refresh_sidebar_toggle_buttons()

    def _refresh_sidebar_toggle_buttons(self) -> None:
        if not hasattr(self, "left_sidebar_button") or not hasattr(self, "right_sidebar_button"):
            return
        sizes = self.main_splitter.sizes()
        if len(sizes) != 3:
            return
        left_visible = sizes[0] > 0
        right_visible = sizes[2] > 0
        self.left_sidebar_button.setText("◀" if left_visible else "▶")
        self.left_sidebar_button.setToolTip(
            "Collapse data sidebar" if left_visible else "Expand data sidebar"
        )
        self.right_sidebar_button.setText("▶" if right_visible else "◀")
        self.right_sidebar_button.setToolTip(
            "Collapse analyses sidebar" if right_visible else "Expand analyses sidebar"
        )

    def _toggle_left_sidebar(self) -> None:
        sizes = self.main_splitter.sizes()
        if len(sizes) != 3:
            return
        left, center, right = sizes
        total = max(left + center + right, 1)
        if left > 0:
            self._left_panel_saved_width = left
            left = 0
        else:
            desired = max(self._left_panel_saved_width, 220)
            max_left = max(total - right - 320, 120)
            left = min(desired, max_left)
        center = max(total - left - right, 260)
        self.main_splitter.setSizes([left, center, right])
        self._refresh_sidebar_toggle_buttons()

    def _toggle_right_sidebar(self) -> None:
        sizes = self.main_splitter.sizes()
        if len(sizes) != 3:
            return
        left, center, right = sizes
        total = max(left + center + right, 1)
        if right > 0:
            self._right_panel_saved_width = right
            right = 0
        else:
            desired = max(self._right_panel_saved_width, 200)
            max_right = max(total - left - 320, 120)
            right = min(desired, max_right)
        center = max(total - left - right, 260)
        self.main_splitter.setSizes([left, center, right])
        self._refresh_sidebar_toggle_buttons()

    def _selected_surface_colormap_name(self) -> str:
        if hasattr(self, "gradients_cmap_combo"):
            name = self.gradients_cmap_combo.currentText().strip()
            if name:
                return name
        return "spectrum_fsl"

    def _selected_surface_colormap(self):
        name = self._selected_surface_colormap_name()
        if name == "spectrum_fsl":
            try:
                return self._colorbar.load_fsl_cmap(name)
            except Exception:
                try:
                    return self._colorbar.bars(name)
                except Exception:
                    return "viridis"
        if name in COLORBAR_BARS:
            try:
                return self._colorbar.bars(name)
            except Exception:
                return "viridis"
        if name in self._custom_cmaps:
            try:
                return self._colorbar.load_fsl_cmap(name)
            except Exception:
                try:
                    return self._colorbar.bars(name)
                except Exception:
                    return "viridis"
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

    @staticmethod
    def _safe_name_fragment(text: str) -> str:
        token = str(text or "").strip()
        if not token:
            return "matrix"
        cleaned = []
        for ch in token:
            if ch.isalnum() or ch in {"-", "_"}:
                cleaned.append(ch)
            else:
                cleaned.append("_")
        out = "".join(cleaned).strip("_")
        while "__" in out:
            out = out.replace("__", "_")
        return out or "matrix"

    def _collect_export_metadata(self, entry, selected_key):
        metadata = {}
        labels = None
        names = None

        source_path_raw = entry.get("source_path", entry.get("path"))
        source_path = Path(source_path_raw) if source_path_raw else None
        if source_path is not None and source_path.exists():
            try:
                with np.load(source_path, allow_pickle=True) as npz:
                    for key in PARCEL_LABEL_KEYS:
                        if key in npz:
                            labels = np.asarray(npz[key])
                            break
                    for key in PARCEL_NAME_KEYS:
                        if key in npz:
                            names = np.asarray(npz[key])
                            break
                    for key in ("group", "modality", "metabolites"):
                        if key in npz:
                            metadata[key] = np.asarray(npz[key])
            except Exception:
                pass

            if labels is None or names is None:
                try:
                    extra_labels, extra_names = _load_parcel_metadata(source_path)
                    if labels is None and extra_labels is not None:
                        labels = np.asarray(extra_labels)
                    if names is None and extra_names is not None:
                        names = np.asarray(extra_names)
                except Exception:
                    pass

        if labels is None and self._current_parcel_labels:
            labels = np.asarray(self._current_parcel_labels)
        if names is None and self._current_parcel_names:
            names = np.asarray(self._current_parcel_names)

        if labels is not None:
            metadata["parcel_labels_group"] = labels
        if names is not None:
            metadata["parcel_names_group"] = names
        if source_path is not None:
            metadata["source_file"] = np.asarray(str(source_path))
        if selected_key:
            metadata["source_key"] = np.asarray(str(selected_key))
        sample_index = entry.get("sample_index")
        if sample_index is not None:
            try:
                metadata["sample_index"] = np.asarray(int(sample_index))
            except Exception:
                pass
        return metadata

    def _write_selected_matrix_to_file(self) -> None:
        entry = self._current_entry()
        if entry is None:
            self.statusBar().showMessage("No matrix selected.")
            return

        try:
            matrix, selected_key = self._matrix_for_entry(entry)
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to resolve selected matrix: {exc}")
            return

        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            self.statusBar().showMessage("Selected matrix is not a square 2D matrix.")
            return

        source_path_raw = entry.get("source_path", entry.get("path"))
        source_path = Path(source_path_raw) if source_path_raw else None
        source_stem = source_path.stem if source_path is not None else self._safe_name_fragment(entry.get("label", "matrix"))
        key_part = self._safe_name_fragment(selected_key or "matrix")
        default_name = f"{self._safe_name_fragment(source_stem)}_{key_part}_matrix_pop_avg.npz"
        start_dir = self._default_dialog_dir()

        save_path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Write selected matrix to NPZ",
            str(start_dir / default_name),
            "NumPy archive (*.npz);;All files (*)",
        )
        if not save_path:
            return
        output_path = Path(save_path)
        if output_path.suffix.lower() != ".npz":
            output_path = output_path.with_suffix(".npz")

        payload = {"matrix_pop_avg": matrix}
        payload.update(self._collect_export_metadata(entry, selected_key))
        try:
            np.savez(output_path, **payload)
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to write NPZ: {exc}")
            return
        self.statusBar().showMessage(f"Wrote selected matrix to {output_path.name}.")

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
        vmin, vmax, scaling_error = self._current_display_limits()
        if scaling_error:
            self.statusBar().showMessage(scaling_error)
            return
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
            SimMatrixPlot.plot_simmatrix(
                matrix,
                ax=ax,
                titles=title,
                colormap=colormap,
                vmin=vmin,
                vmax=vmax,
            )
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
        self._update_write_to_file_button()

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

        vmin, vmax, scaling_error = self._current_display_limits()
        if scaling_error:
            self.statusBar().showMessage(scaling_error)
            return

        colormap = self._selected_colormap()
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        SimMatrixPlot.plot_simmatrix(
            matrix,
            ax=ax,
            titles=None,
            colormap=colormap,
            vmin=vmin,
            vmax=vmax,
        )
        _remove_axes_border(ax)
        self._current_axes = ax
        title_tooltip = str(source_path) if source_path is not None else current_title
        self._set_plot_title(current_title, tooltip_text=title_tooltip)
        self._reset_gradients_output()
        self.canvas.draw_idle()
        self._update_nbs_prepare_button()
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
            with np.load(source_path, allow_pickle=True) as npz:
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

    def _compute_gradients(self) -> None:
        self._reset_gradients_output()
        if self._current_matrix is None:
            self.statusBar().showMessage("No matrix selected for gradients.")
            return
        conn_matrix = np.asarray(self._current_matrix, dtype=float)
        if conn_matrix.ndim != 2 or conn_matrix.shape[0] != conn_matrix.shape[1]:
            self.statusBar().showMessage("Gradients require a square matrix.")
            return

        source_dir = self._default_dialog_dir()
        if self._active_parcellation_data is None:
            if not self._select_parcellation_template():
                self.statusBar().showMessage("Gradient compute canceled (no template selected).")
                return

        try:
            import nibabel as nib
        except Exception as exc:
            self.statusBar().showMessage(f"nibabel not available: {exc}")
            return

        template_img = self._active_parcellation_img
        template_data = self._active_parcellation_data
        if template_img is None or template_data is None:
            self.statusBar().showMessage("No active parcellation template.")
            return

        source_path = self._current_source_path()
        if source_path is None or not source_path.exists():
            self.statusBar().showMessage("Projection requires a source .npz with parcel labels.")
            return
        parcel_labels, _ = _load_parcel_metadata(source_path)
        label_indices = _coerce_label_indices(parcel_labels, conn_matrix.shape[0])
        if label_indices is None:
            self.statusBar().showMessage(
                "parcel_labels_group missing/invalid or does not match matrix nodes."
            )
            return
        template_labels = set(np.asarray(template_data, dtype=int).reshape(-1).tolist())
        template_labels.discard(0)
        if not template_labels:
            self.statusBar().showMessage("Template has no non-zero labels.")
            return
        keep_indices = [idx for idx, label in enumerate(label_indices) if label in template_labels]
        if not keep_indices:
            self.statusBar().showMessage(
                "No overlap between matrix parcel_labels_group and active parcellation labels."
            )
            return
        projection_labels = [label_indices[idx] for idx in keep_indices]

        n_grad = self.gradients_spin.value()
        self.gradients_progress.setRange(0, n_grad)
        self.gradients_progress.setValue(0)
        self.gradients_progress.setFormat(f"0/{n_grad} components")
        self.gradients_compute_button.setEnabled(False)
        self.gradients_spin.setEnabled(False)
        self.select_parcellation_button.setEnabled(False)
        QApplication.processEvents()

        gradients = np.zeros((conn_matrix.shape[0], n_grad), dtype=float)
        projected_maps = []
        try:
            for comp_idx in range(1, n_grad + 1):
                try:
                    component = nettools.dimreduce_matrix(
                        conn_matrix,
                        method="diffusion",
                        scale_factor=1.0,
                        output_dim=comp_idx,
                    )
                except Exception as exc:
                    self.statusBar().showMessage(f"Failed to compute component {comp_idx}: {exc}")
                    return
                component = np.asarray(component, dtype=float).reshape(-1)
                if component.size != conn_matrix.shape[0]:
                    self.statusBar().showMessage(
                        f"Component {comp_idx} size mismatch ({component.size} vs {conn_matrix.shape[0]})."
                    )
                    return
                gradients[:, comp_idx - 1] = component
                component_restricted = component[keep_indices]
                try:
                    projected = nettools.project_to_3dspace(
                        component_restricted,
                        template_data,
                        projection_labels,
                    )
                except Exception as exc:
                    self.statusBar().showMessage(f"Failed to project component {comp_idx}: {exc}")
                    return
                projected_maps.append(np.asarray(projected, dtype=np.float32))
                self.gradients_progress.setValue(comp_idx)
                self.gradients_progress.setFormat(f"{comp_idx}/{n_grad} components")
                QApplication.processEvents()
        finally:
            self.gradients_compute_button.setEnabled(True)
            self.gradients_spin.setEnabled(True)
            self.select_parcellation_button.setEnabled(True)

        if n_grad == 1:
            output_data = projected_maps[0]
        else:
            output_data = np.stack(projected_maps, axis=-1)

        source_stem = source_path.stem if source_path is not None else "matrix"
        default_name = f"{source_stem}_diffusion_components-{n_grad}.nii.gz"

        self._last_gradients = {
            "gradients": gradients,
            "n_grad": n_grad,
            "n_nodes": conn_matrix.shape[0],
            "projected_data": np.asarray(output_data, dtype=np.float32),
            "affine": np.asarray(template_img.affine, dtype=float),
            "header": template_img.header.copy(),
            "source_name": source_path.name if source_path is not None else "matrix",
            "source_dir": str(source_dir),
            "output_name": default_name,
        }
        self.gradients_save_button.setEnabled(True)
        self.gradients_render_button.setEnabled(True)
        self.gradients_progress.setValue(n_grad)
        self.gradients_progress.setFormat(f"{n_grad}/{n_grad} components (done)")
        self.statusBar().showMessage(
            f"Computed {n_grad} projected component(s). Click Save or Render 3D."
        )

    def _save_gradients_projection(self) -> None:
        if not self._last_gradients:
            self.statusBar().showMessage("No projected gradients to save. Click Compute first.")
            return
        projected_data = self._last_gradients.get("projected_data")
        if projected_data is None:
            self.statusBar().showMessage("No projected data available to save.")
            return

        try:
            import nibabel as nib
        except Exception as exc:
            self.statusBar().showMessage(f"nibabel not available: {exc}")
            return

        base_dir = Path(self._last_gradients.get("source_dir", str(self._default_dialog_dir())))
        default_name = self._last_gradients.get("output_name", "diffusion_components.nii.gz")
        save_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save gradient projection",
            str(base_dir / default_name),
            "NIfTI GZip (*.nii.gz);;NIfTI (*.nii);;All files (*)",
        )
        if not save_path:
            return

        output_path = Path(save_path)
        lower_name = output_path.name.lower()
        if not (lower_name.endswith(".nii") or lower_name.endswith(".nii.gz")):
            if "NIfTI (*.nii)" in selected_filter:
                output_path = output_path.with_suffix(".nii")
            else:
                output_path = output_path.with_suffix(".nii.gz")

        affine = self._last_gradients.get("affine")
        header = self._last_gradients.get("header")
        if affine is None:
            affine = np.eye(4)
        try:
            if header is not None:
                out_img = nib.Nifti1Image(projected_data, affine, header.copy())
            else:
                out_img = nib.Nifti1Image(projected_data, affine)
            nib.save(out_img, str(output_path))
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to save NIfTI: {exc}")
            return
        self.statusBar().showMessage(f"Saved projection to {output_path.name}.")

    def _render_gradients_3d(self) -> None:
        if not self._last_gradients:
            self.statusBar().showMessage("No gradients computed yet.")
            return
        projected_data = self._last_gradients.get("projected_data")
        if projected_data is None:
            self.statusBar().showMessage("No projected 3D data available. Compute gradients again.")
            return
        try:
            from window.plot_msmode import MSModeSurfaceDialog
        except Exception:
            try:
                from mrsi_viewer.window.plot_msmode import MSModeSurfaceDialog
            except Exception as exc:
                self.statusBar().showMessage(f"Nilearn surface viewer unavailable: {exc}")
                return
        try:
            n_grad = int(self._last_gradients.get("n_grad", 1))
            source_name = self._last_gradients.get("source_name", "matrix")
            cmap_name = self._selected_surface_colormap_name()
            title = f"Diffusion components ({n_grad}) - {source_name}"
            self._surface_dialog = MSModeSurfaceDialog.from_array(
                projected_data,
                affine=self._last_gradients.get("affine"),
                title=title,
                cmap=self._selected_surface_colormap(),
                cmap_name=cmap_name,
                theme_name=self._theme_name,
                parent=self,
            )
            screen = QApplication.primaryScreen()
            if screen is not None:
                geom = screen.availableGeometry()
                width = max(int(geom.width() * 0.88), 900)
                height = min(max(300 + 300 * max(n_grad, 1), 420), int(geom.height() * 0.9))
            else:
                width = 1500
                height = 300 + 300 * max(n_grad, 1)
            self._surface_dialog.resize(width, height)
            self._surface_dialog.show()
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to render 3D surfaces: {exc}")
            return
        self.statusBar().showMessage("Opened Nilearn 3D surface viewer.")

    def _clear_plot(self) -> None:
        self.figure.clear()
        self._current_matrix = None
        self._current_parcel_labels = None
        self._current_parcel_names = None
        self._current_axes = None
        self._set_plot_title("")
        self._reset_gradients_output()
        self._update_nbs_prepare_button()
        self._update_write_to_file_button()
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

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_plot_title_label()

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
    app_icon = _load_app_icon()
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)
    splash_candidates = [
        ROOTDIR / "assets" / "splash.png",
        ROOTDIR / "icons" / "conviewer_2.png",
        ROOTDIR / "icons" / "conviewer.png",
    ]
    splash_pix = QPixmap()
    for candidate in splash_candidates:
        if candidate.exists():
            splash_pix = QPixmap(str(candidate))
            if not splash_pix.isNull():
                break

    splash = None
    if not splash_pix.isNull():
        if hasattr(Qt, "WindowType"):
            splash_hint = Qt.WindowType.WindowStaysOnTopHint
        else:
            splash_hint = Qt.WindowStaysOnTopHint
        if hasattr(Qt, "AlignmentFlag"):
            splash_align = Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter
        else:
            splash_align = Qt.AlignBottom | Qt.AlignHCenter
        splash = QSplashScreen(splash_pix, splash_hint)
        splash.show()
        app.processEvents()
        splash.showMessage("Loading resources...", splash_align, QColor("white"))
        app.processEvents()
        splash.showMessage("Building UI...", splash_align, QColor("white"))
        app.processEvents()
        splash_deadline = time.monotonic() + 3.0
        while time.monotonic() < splash_deadline:
            app.processEvents()
            time.sleep(0.02)

    window = ConnectomeViewer()
    if splash is not None:
        splash.showMessage("Starting Donald...", splash_align, QColor("white"))
        app.processEvents()
    window.showMaximized()
    if splash is not None:
        splash.finish(window)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
