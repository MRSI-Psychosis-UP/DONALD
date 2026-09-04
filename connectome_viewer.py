#!/usr/bin/env python3
import atexit
import faulthandler
import json
import math
import os
import re
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from PyQt6.QtCore import Qt, QSize, QTimer, qInstallMessageHandler
    from PyQt6.QtWidgets import (
        QApplication,
        QAbstractItemView,
        QDialog,
        QFrame,
        QGridLayout,
        QFileDialog,
        QCheckBox,
        QGroupBox,
        QHeaderView,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPlainTextEdit,
        QComboBox,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QProgressBar,
        QPushButton,
        QSplashScreen,
        QSizePolicy,
        QSplitter,
        QStackedWidget,
        QSpinBox,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
    from PyQt6.QtGui import QAction, QIcon, QFontMetrics, QPixmap, QColor
    QT_LIB = 6
except ImportError:
    from PyQt5.QtCore import Qt, QSize, QTimer, qInstallMessageHandler
    from PyQt5.QtWidgets import (
        QAction,
        QApplication,
        QAbstractItemView,
        QDialog,
        QFrame,
        QGridLayout,
        QFileDialog,
        QCheckBox,
        QGroupBox,
        QHeaderView,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPlainTextEdit,
        QComboBox,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QProgressBar,
        QPushButton,
        QSplashScreen,
        QSizePolicy,
        QSplitter,
        QStackedWidget,
        QSpinBox,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
    from PyQt5.QtGui import QIcon, QFontMetrics, QPixmap, QColor
    QT_LIB = 5

_DIAGNOSTIC_LOG_HANDLE = None
_QT_MESSAGE_HANDLER = None


def _diagnostic_log_path() -> Path:
    raw = str(os.getenv("MRSI_VIEWER_DIAGNOSTICS_LOG", "")).strip()
    if raw:
        return Path(raw).expanduser()
    return Path("/tmp/mrsi_viewer_diagnostics.log")


def _write_diagnostic_line(text: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {text}"
    try:
        sys.__stderr__.write(line + "\n")
        sys.__stderr__.flush()
    except Exception:
        pass
    global _DIAGNOSTIC_LOG_HANDLE
    if _DIAGNOSTIC_LOG_HANDLE is not None:
        try:
            _DIAGNOSTIC_LOG_HANDLE.write(line + "\n")
            _DIAGNOSTIC_LOG_HANDLE.flush()
        except Exception:
            pass


def _install_runtime_diagnostics() -> Path:
    global _DIAGNOSTIC_LOG_HANDLE, _QT_MESSAGE_HANDLER

    log_path = _diagnostic_log_path()
    os.environ.setdefault("MRSI_VIEWER_DIAGNOSTICS_LOG", str(log_path))
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        _DIAGNOSTIC_LOG_HANDLE = open(log_path, "a", encoding="utf-8", buffering=1)
    except Exception:
        _DIAGNOSTIC_LOG_HANDLE = None

    _write_diagnostic_line(
        f"=== Launch pid={os.getpid()} argv={sys.argv!r} qt_lib={QT_LIB} python={sys.version.split()[0]} ==="
    )

    try:
        faulthandler.enable(file=_DIAGNOSTIC_LOG_HANDLE or sys.__stderr__, all_threads=True)
        _write_diagnostic_line("faulthandler enabled")
    except Exception as exc:
        _write_diagnostic_line(f"failed to enable faulthandler: {exc}")

    default_excepthook = sys.excepthook

    def _diagnostic_excepthook(exc_type, exc_value, exc_tb):
        _write_diagnostic_line(
            "Unhandled Python exception:\n" + "".join(traceback.format_exception(exc_type, exc_value, exc_tb)).rstrip()
        )
        try:
            default_excepthook(exc_type, exc_value, exc_tb)
        except Exception:
            pass

    sys.excepthook = _diagnostic_excepthook

    try:
        def _qt_message_handler(msg_type, context, message):
            file_name = getattr(context, "file", "") if context is not None else ""
            line_no = getattr(context, "line", 0) if context is not None else 0
            function_name = getattr(context, "function", "") if context is not None else ""
            mode_name = getattr(msg_type, "name", str(msg_type))
            _write_diagnostic_line(
                f"[QT {mode_name}] {message} ({file_name}:{line_no} {function_name})"
            )

        _QT_MESSAGE_HANDLER = _qt_message_handler
        qInstallMessageHandler(_QT_MESSAGE_HANDLER)
        _write_diagnostic_line("Qt message handler installed")
    except Exception as exc:
        _write_diagnostic_line(f"failed to install Qt message handler: {exc}")

    def _on_exit():
        _write_diagnostic_line("Process exiting")

    atexit.register(_on_exit)
    _write_diagnostic_line(f"diagnostic log path: {log_path}")
    return log_path

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
from matplotlib.colors import Normalize
import matplotlib.transforms as mtransforms
try:
    from matplotlib.widgets import RectangleSelector
except Exception:
    RectangleSelector = None

try:
    from services.combine import (
        align_matrices_by_intersection as _combine_align_matrices,
        apply_matrix_operation as _combine_apply_op,
        combine_operation_label as _combine_op_label,
        combine_operation_symbol as _combine_op_symbol,
        compute_correlation_stats as _combine_corr_stats,
        correlation_vectors as _combine_corr_vectors,
        format_p_value as _format_combine_p_value,
        parcellation_centroids as _combine_parcellation_centroids,
        pairwise_distance_matrix as _combine_pairwise_distance_matrix,
        residualize_matrix as _combine_residualize_matrix,
        spatially_residualize_matrix as _combine_spatially_residualize_matrix,
    )
    from services.data_access import MatrixDataAccess as _MatrixDataAccess
    from services.entry_helpers import (
        safe_name_fragment as _safe_name_fragment_helper,
    )
    from services.prepare_dialog_controller import PrepareDialogController as _PrepareDialogController
    from services.gradient_dialog_controller import GradientDialogController as _GradientDialogController
    from services.workspace_file_controller import WorkspaceFileController as _WorkspaceFileController
    from services.plot_prep import (
        normalize_matrix_labels as _normalize_plot_matrix_labels,
        resolve_entry_plot as _resolve_entry_plot,
    )
    from services.workspace_matrix_controller import WorkspaceMatrixController as _WorkspaceMatrixController
    from services.workspace import WorkspaceStore as _WorkspaceStore
except Exception:
    from mrsi_viewer.services.combine import (
        align_matrices_by_intersection as _combine_align_matrices,
        apply_matrix_operation as _combine_apply_op,
        combine_operation_label as _combine_op_label,
        combine_operation_symbol as _combine_op_symbol,
        compute_correlation_stats as _combine_corr_stats,
        correlation_vectors as _combine_corr_vectors,
        format_p_value as _format_combine_p_value,
        parcellation_centroids as _combine_parcellation_centroids,
        pairwise_distance_matrix as _combine_pairwise_distance_matrix,
        residualize_matrix as _combine_residualize_matrix,
        spatially_residualize_matrix as _combine_spatially_residualize_matrix,
    )
    from mrsi_viewer.services.data_access import MatrixDataAccess as _MatrixDataAccess
    from mrsi_viewer.services.entry_helpers import (
        safe_name_fragment as _safe_name_fragment_helper,
    )
    from mrsi_viewer.services.prepare_dialog_controller import PrepareDialogController as _PrepareDialogController
    from mrsi_viewer.services.gradient_dialog_controller import GradientDialogController as _GradientDialogController
    from mrsi_viewer.services.workspace_file_controller import WorkspaceFileController as _WorkspaceFileController
    from mrsi_viewer.services.plot_prep import (
        normalize_matrix_labels as _normalize_plot_matrix_labels,
        resolve_entry_plot as _resolve_entry_plot,
    )
    from mrsi_viewer.services.workspace_matrix_controller import WorkspaceMatrixController as _WorkspaceMatrixController
    from mrsi_viewer.services.workspace import WorkspaceStore as _WorkspaceStore

try:
    from mrsitoolbox.graphplot.simmatrix import SimMatrixPlot
    from mrsitoolbox.graphplot.colorbar import ColorBar
    from mrsitoolbox.graphplot import colorbar as colorbar_module
    from mrsitoolbox.connectomics.nettools import NetTools
except ImportError as exc:
    raise ImportError(
        "Donald requires the current mrsitoolbox pip package. "
        "Install or upgrade it with: pip install --upgrade mrsitoolbox"
    ) from exc

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

PARCEL_LABEL_KEYS = (
    "parcel_labels_group",
    "parcel_label_group",
    "parcel_labels",
    "parcel_label",
    "labels_indices",
    "parcel_labels_group.npy",
    "parcel_label_group.npy",
    "parcel_labels.npy",
    "parcel_label.npy",
    "labels_indices.npy",
)
PARCEL_NAME_KEYS = (
    "parcel_names_group",
    "parcel_name_group",
    "parcel_names",
    "parcel_name",
    "label_names",
    "labels",
    "parcel_names_group.npy",
    "parcel_name_group.npy",
    "parcel_names.npy",
    "parcel_name.npy",
    "label_names.npy",
    "labels.npy",
)
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


def _is_enabled_flag():
    return getattr(Qt, "ItemIsEnabled", getattr(Qt.ItemFlag, "ItemIsEnabled"))


def _is_user_checkable_flag():
    return getattr(Qt, "ItemIsUserCheckable", getattr(Qt.ItemFlag, "ItemIsUserCheckable"))


def _is_selectable_flag():
    return getattr(Qt, "ItemIsSelectable", getattr(Qt.ItemFlag, "ItemIsSelectable"))


def _is_editable_flag():
    return getattr(Qt, "ItemIsEditable", getattr(Qt.ItemFlag, "ItemIsEditable"))


def _qheader_resize_mode(name: str):
    resize_mode = getattr(QHeaderView, "ResizeMode", None)
    if resize_mode is not None and hasattr(resize_mode, name):
        return getattr(resize_mode, name)
    return getattr(QHeaderView, name)


def _set_header_resize_mode(header, section: int, mode_name: str) -> None:
    header.setSectionResizeMode(section, _qheader_resize_mode(mode_name))


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


def _load_parcel_metadata(path: Path, label_key: str = "", name_key: str = ""):
    label_key = str(label_key or "").strip()
    name_key = str(name_key or "").strip()
    labels = None
    names = None
    with np.load(path, allow_pickle=True) as npz:
        if label_key:
            if label_key in npz:
                labels = npz[label_key]
        else:
            for key in PARCEL_LABEL_KEYS:
                if key in npz:
                    labels = npz[key]
                    break
        if name_key:
            if name_key in npz:
                names = npz[name_key]
        else:
            for key in PARCEL_NAME_KEYS:
                if key in npz:
                    names = npz[key]
                    break
    if labels is None:
        label_sidecar_names = (label_key,) if label_key else PARCEL_LABEL_KEYS
        for sidecar_name in label_sidecar_names:
            labels_path = path.with_name(sidecar_name)
            if labels_path.exists():
                labels = np.load(labels_path, allow_pickle=True)
                break
    if names is None:
        name_sidecar_names = (name_key,) if name_key else PARCEL_NAME_KEYS
        for sidecar_name in name_sidecar_names:
            names_path = path.with_name(sidecar_name)
            if names_path.exists():
                names = np.load(names_path, allow_pickle=True)
                break
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


def _covars_to_rows(info):
    if info is None:
        return [], []

    df = info.get("df")
    if df is not None:
        columns = [str(col) for col in df.columns]
        rows = []
        for row in df.to_dict(orient="records"):
            rows.append([_display_text(row.get(col)) for col in columns])
        return columns, rows

    data = info.get("data")
    if data is None:
        return [], []
    arr = np.asarray(data)

    if getattr(arr.dtype, "names", None):
        columns = [str(col) for col in arr.dtype.names]
        rows = []
        for record in arr:
            rows.append([_display_text(record[col]) for col in columns])
        return columns, rows

    if arr.ndim == 2:
        columns = [f"col_{idx}" for idx in range(arr.shape[1])]
        rows = []
        for row in arr:
            rows.append([_display_text(value) for value in row])
        return columns, rows

    return [], []


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


def _normalize_subject_token(value: str) -> str:
    token = str(value).strip()
    if token.lower().startswith("sub-"):
        token = token[4:]
    return token


def _normalize_session_token(value: str) -> str:
    token = str(value).strip()
    if token.lower().startswith("ses-"):
        token = token[4:]
    upper = token.upper()
    if upper.startswith("T") and len(upper) > 1:
        return f"V{upper[1:]}"
    if token.isdigit():
        return f"V{token}"
    return token


def _display_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return value.decode("latin-1", errors="ignore")
    try:
        if isinstance(value, np.generic):
            return str(value.item())
    except Exception:
        pass
    return str(value)


def _column_is_numeric(values):
    has_value = False
    for value in values:
        text = _display_text(value).strip()
        if text == "":
            continue
        has_value = True
        try:
            float(text)
        except Exception:
            return False
    return has_value


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


def _flatten_display_vector(values):
    try:
        array = np.asarray(values)
    except Exception:
        return None
    if array.ndim == 0:
        array = array.reshape(1)
    elif array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    elif array.ndim != 1:
        return None
    return [_display_text(value) for value in array.tolist()]


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


class LabelInfoDialog(QDialog):
    def __init__(self, source_path: Path, parent=None) -> None:
        super().__init__(parent)
        self._source_path = Path(source_path)
        self._vector_options = self._load_vector_options()
        self.setWindowTitle(f"Label Info - {self._source_path.name}")
        self.resize(860, 620)
        self._build_ui()
        self._populate_key_selectors()
        self._update_table()

    def _load_vector_options(self):
        options = {}
        try:
            with np.load(self._source_path, allow_pickle=True) as npz:
                for key in npz.files:
                    values = _flatten_display_vector(npz[key])
                    if values is not None:
                        options[str(key)] = values
        except Exception:
            return {}
        return options

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        info_label = QLabel(
            "Inspect parcel label indices and names from the selected NPZ. "
            "If the standard keys are missing, choose alternative 1D arrays from the dropdowns."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        form = QGridLayout()
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(10)

        form.addWidget(QLabel("Label indices key"), 0, 0)
        self.indices_combo = QComboBox()
        self.indices_combo.currentIndexChanged.connect(self._update_table)
        form.addWidget(self.indices_combo, 0, 1)

        form.addWidget(QLabel("Label names key"), 0, 2)
        self.names_combo = QComboBox()
        self.names_combo.currentIndexChanged.connect(self._update_table)
        form.addWidget(self.names_combo, 0, 3)

        layout.addLayout(form)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Row", "Label Index", "Label Name"])
        self.table.setAlternatingRowColors(True)
        if QT_LIB == 6:
            self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        else:
            self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        if header is not None:
            header.setStretchLastSection(True)
            _set_header_resize_mode(header, 0, "ResizeToContents")
            _set_header_resize_mode(header, 1, "ResizeToContents")
            _set_header_resize_mode(header, 2, "Stretch")
        layout.addWidget(self.table, 1)

        close_row = QHBoxLayout()
        close_row.addStretch(1)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        close_row.addWidget(close_button)
        layout.addLayout(close_row)

    def _preferred_key(self, choices, preferred_keys):
        for key in preferred_keys:
            if key in choices:
                return key
        return ""

    def _populate_key_selectors(self) -> None:
        options = [""] + sorted(self._vector_options.keys())
        self.indices_combo.blockSignals(True)
        self.names_combo.blockSignals(True)
        self.indices_combo.clear()
        self.names_combo.clear()
        self.indices_combo.addItem("(None)", "")
        self.names_combo.addItem("(None)", "")
        for key in sorted(self._vector_options.keys()):
            self.indices_combo.addItem(key, key)
            self.names_combo.addItem(key, key)

        preferred_indices = self._preferred_key(options, PARCEL_LABEL_KEYS)
        preferred_names = self._preferred_key(options, PARCEL_NAME_KEYS)
        if preferred_indices:
            self.indices_combo.setCurrentText(preferred_indices)
        elif self.indices_combo.count() > 1:
            self.indices_combo.setCurrentIndex(1)
        if preferred_names:
            self.names_combo.setCurrentText(preferred_names)
        elif self.names_combo.count() > 1:
            fallback_index = 2 if self.names_combo.count() > 2 else 1
            self.names_combo.setCurrentIndex(fallback_index)
        self.indices_combo.blockSignals(False)
        self.names_combo.blockSignals(False)

    def _selected_vector(self, combo: QComboBox):
        key = combo.currentData()
        if not key:
            return None
        return list(self._vector_options.get(str(key), []))

    def _update_table(self) -> None:
        indices = self._selected_vector(self.indices_combo) or []
        names = self._selected_vector(self.names_combo) or []
        row_count = max(len(indices), len(names))
        self.table.setRowCount(row_count)
        for row in range(row_count):
            values = [
                str(row),
                indices[row] if row < len(indices) else "",
                names[row] if row < len(names) else "",
            ]
            for col, text in enumerate(values):
                item = QTableWidgetItem(str(text))
                item.setFlags(item.flags() & ~_is_editable_flag())
                self.table.setItem(row, col, item)

        missing = []
        if self._preferred_key(self._vector_options, PARCEL_LABEL_KEYS) == "":
            missing.append("parcel label indices")
        if self._preferred_key(self._vector_options, PARCEL_NAME_KEYS) == "":
            missing.append("parcel label names")
        if missing:
            missing_text = ", ".join(missing)
            self.status_label.setText(
                f"Standard keys not found for {missing_text}. Choose alternative arrays from the dropdowns."
            )
        else:
            self.status_label.setText("Loaded standard parcel label keys from the NPZ file.")


class ParticipantsInfoDialog(QDialog):
    def __init__(self, source_path: Path, covars_info, parent=None) -> None:
        super().__init__(parent)
        self._source_path = Path(source_path)
        self._columns, self._rows = _covars_to_rows(covars_info)
        self.setWindowTitle(f"Participants - {self._source_path.name}")
        self.resize(1080, 760)
        self._build_ui()
        self._populate_table()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        summary = QLabel(
            f"{self._source_path.name} | rows: {len(self._rows)} | covariates: {len(self._columns)}"
        )
        summary.setWordWrap(False)
        layout.addWidget(summary)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        if QT_LIB == 6:
            self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        else:
            self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table, 1)

        close_row = QHBoxLayout()
        close_row.addStretch(1)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        close_row.addWidget(close_button)
        layout.addLayout(close_row)

    def _populate_table(self) -> None:
        self.table.clear()
        self.table.setColumnCount(len(self._columns))
        self.table.setHorizontalHeaderLabels(self._columns)
        self.table.setRowCount(len(self._rows))

        editable_flag = _is_editable_flag()
        for row_idx, row_values in enumerate(self._rows):
            for col_idx, value in enumerate(row_values):
                item = QTableWidgetItem(_display_text(value))
                item.setFlags(item.flags() & ~editable_flag)
                self.table.setItem(row_idx, col_idx, item)

        header = self.table.horizontalHeader()
        if header is not None and self._columns:
            for col_idx in range(max(len(self._columns) - 1, 0)):
                _set_header_resize_mode(header, col_idx, "ResizeToContents")
            _set_header_resize_mode(header, len(self._columns) - 1, "Stretch")

        if not self._columns:
            self.status_label.setText("No covariate columns were found in this NPZ file.")
        else:
            self.status_label.setText("Loaded participant covariates from NPZ `covars`.")


class PreferencesDialog(QDialog):
    def __init__(
        self,
        *,
        theme_name: str,
        matrix_cmap: str,
        gradient_cmap: str,
        matlab_cmd: str,
        matlab_nbs_path: str,
        results_dir: str,
        bids_dir: str,
        atlas_dir: str,
        colormap_names,
        dialog_title: str = "Preferences",
        require_results_dir: bool = True,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._require_results_dir = bool(require_results_dir)
        self.setWindowTitle(dialog_title)
        self.resize(860, 520)
        self._colormap_names = list(colormap_names or [])
        self._build_ui(
            theme_name=theme_name,
            matrix_cmap=matrix_cmap,
            gradient_cmap=gradient_cmap,
            matlab_cmd=matlab_cmd,
            matlab_nbs_path=matlab_nbs_path,
            results_dir=results_dir,
            bids_dir=bids_dir,
            atlas_dir=atlas_dir,
        )

    def _build_ui(
        self,
        *,
        theme_name: str,
        matrix_cmap: str,
        gradient_cmap: str,
        matlab_cmd: str,
        matlab_nbs_path: str,
        results_dir: str,
        bids_dir: str,
        atlas_dir: str,
    ) -> None:
        layout = QVBoxLayout(self)
        info_label = QLabel(
            "Configure application defaults. Results folder is required and will be used as the default output root."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
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

        row += 1
        form.addWidget(QLabel("Results folder"), row, 0)
        self.results_dir_edit = QLineEdit(str(results_dir or ""))
        self.results_dir_edit.setPlaceholderText("Required output folder")
        form.addWidget(self.results_dir_edit, row, 1, 1, 2)
        results_browse_button = QPushButton("Browse")
        results_browse_button.clicked.connect(self._browse_results_dir)
        form.addWidget(results_browse_button, row, 3)

        row += 1
        form.addWidget(QLabel("BIDS folder"), row, 0)
        self.bids_dir_edit = QLineEdit(str(bids_dir or ""))
        self.bids_dir_edit.setPlaceholderText("Optional default folder when opening matrices")
        form.addWidget(self.bids_dir_edit, row, 1, 1, 2)
        bids_browse_button = QPushButton("Browse")
        bids_browse_button.clicked.connect(self._browse_bids_dir)
        form.addWidget(bids_browse_button, row, 3)

        row += 1
        form.addWidget(QLabel("Atlas folder"), row, 0)
        self.atlas_dir_edit = QLineEdit(str(atlas_dir or ""))
        self.atlas_dir_edit.setPlaceholderText("Optional default folder for parcellation / atlas files")
        form.addWidget(self.atlas_dir_edit, row, 1, 1, 2)
        atlas_browse_button = QPushButton("Browse")
        atlas_browse_button.clicked.connect(self._browse_atlas_dir)
        form.addWidget(atlas_browse_button, row, 3)

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

    @staticmethod
    def _browse_directory(parent, title: str, current: str) -> str:
        start_dir = str(Path(current).expanduser()) if current else str(Path.home())
        return QFileDialog.getExistingDirectory(parent, title, start_dir)

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

    def _browse_results_dir(self) -> None:
        selected = self._browse_directory(
            self,
            "Select results folder",
            self.results_dir_edit.text().strip(),
        )
        if selected:
            self.results_dir_edit.setText(str(Path(selected).resolve()))

    def _browse_bids_dir(self) -> None:
        selected = self._browse_directory(
            self,
            "Select BIDS folder",
            self.bids_dir_edit.text().strip(),
        )
        if selected:
            self.bids_dir_edit.setText(str(Path(selected).resolve()))

    def _browse_atlas_dir(self) -> None:
        selected = self._browse_directory(
            self,
            "Select atlas folder",
            self.atlas_dir_edit.text().strip(),
        )
        if selected:
            self.atlas_dir_edit.setText(str(Path(selected).resolve()))

    def accept(self) -> None:
        results_dir = self.results_dir_edit.text().strip()
        if self._require_results_dir and not results_dir:
            QMessageBox.warning(self, "Results Folder Required", "Select a results folder before continuing.")
            return
        if results_dir:
            try:
                results_path = Path(results_dir).expanduser()
                results_path.mkdir(parents=True, exist_ok=True)
                self.results_dir_edit.setText(str(results_path.resolve()))
            except Exception as exc:
                QMessageBox.warning(self, "Invalid Results Folder", f"Failed to create results folder:\n{exc}")
                return
        super().accept()

    def values(self):
        return {
            "theme": self.theme_combo.currentText().strip(),
            "matrix_colormap": self.matrix_cmap_combo.currentText().strip(),
            "gradient_colormap": self.gradients_cmap_combo.currentText().strip(),
            "matlab_cmd": self.matlab_cmd_edit.text().strip(),
            "matlab_nbs_path": self.nbs_path_edit.text().strip(),
            "results_dir": self.results_dir_edit.text().strip(),
            "bids_dir": self.bids_dir_edit.text().strip(),
            "atlas_dir": self.atlas_dir_edit.text().strip(),
        }


class ExportGridDialog(QDialog):
    _FILTERS = "PDF (*.pdf);;SVG (*.svg);;PNG (*.png)"

    def __init__(
        self,
        *,
        default_path: str,
        default_columns: int,
        rotate: bool,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._selected_filter = "PDF (*.pdf)"
        self.setWindowTitle("Export Grid")
        self.resize(720, 220)
        self._build_ui(
            default_path=default_path,
            default_columns=default_columns,
            rotate=rotate,
        )

    def _build_ui(self, *, default_path: str, default_columns: int, rotate: bool) -> None:
        layout = QVBoxLayout(self)

        info_label = QLabel(
            "Choose the output file, grid layout, and optional 45 degree rotation for the exported workspace matrices."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        form = QGridLayout()
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(10)

        form.addWidget(QLabel("Output path"), 0, 0)
        self.output_path_edit = QLineEdit(str(default_path or ""))
        self.output_path_edit.setPlaceholderText("Choose PDF, SVG, or PNG output")
        form.addWidget(self.output_path_edit, 0, 1, 1, 2)
        self.output_browse_button = QPushButton("Browse")
        self.output_browse_button.clicked.connect(self._browse_output_path)
        form.addWidget(self.output_browse_button, 0, 3)

        form.addWidget(QLabel("Columns"), 1, 0)
        self.columns_spin = QSpinBox()
        self.columns_spin.setRange(1, 12)
        self.columns_spin.setValue(max(1, min(int(default_columns), 12)))
        form.addWidget(self.columns_spin, 1, 1)

        self.rotate_check = QCheckBox("Rotate 45 deg")
        self.rotate_check.setObjectName("greenSquareIndicator")
        self.rotate_check.setChecked(bool(rotate))
        form.addWidget(self.rotate_check, 1, 2, 1, 2)

        layout.addLayout(form)
        layout.addStretch(1)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_row.addWidget(self.cancel_button)
        self.export_button = QPushButton("Export")
        self.export_button.clicked.connect(self.accept)
        button_row.addWidget(self.export_button)
        layout.addLayout(button_row)

    def _browse_output_path(self) -> None:
        current = self.output_path_edit.text().strip()
        if current:
            start_path = Path(current).expanduser()
        else:
            start_path = Path.home() / "connectome_grid.pdf"
        selected, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Choose export path",
            str(start_path),
            self._FILTERS,
        )
        if not selected:
            return
        self.output_path_edit.setText(str(Path(selected).expanduser()))
        if selected_filter:
            self._selected_filter = selected_filter

    def set_theme(self, theme_name: str) -> None:
        theme = str(theme_name or "Dark").strip().title()
        if theme == "Dark":
            unchecked_bg = "#1f2430"
            unchecked_border = "#94a3b8"
        elif theme == "Teya":
            unchecked_bg = "#ffe6f1"
            unchecked_border = "#0b7f7a"
        elif theme == "Donald":
            unchecked_bg = "#c96a04"
            unchecked_border = "#ffd19e"
        else:
            unchecked_bg = "#ffffff"
            unchecked_border = "#94a3b8"
        self.rotate_check.setStyleSheet(
            "QCheckBox#greenSquareIndicator::indicator {"
            "width: 16px;"
            "height: 16px;"
            "border-radius: 2px;"
            "}"
            f"QCheckBox#greenSquareIndicator::indicator:unchecked {{ background: {unchecked_bg}; border: 2px solid {unchecked_border}; }}"
            "QCheckBox#greenSquareIndicator::indicator:checked { background: #16a34a; border: 2px solid #15803d; }"
        )

    def accept(self) -> None:
        output_raw = self.output_path_edit.text().strip()
        if not output_raw:
            QMessageBox.warning(self, "Output Path Required", "Choose an export path before continuing.")
            return
        try:
            output_path = Path(output_raw).expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.output_path_edit.setText(str(output_path))
        except Exception as exc:
            QMessageBox.warning(self, "Invalid Output Path", f"Failed to prepare output folder:\n{exc}")
            return
        super().accept()

    def values(self):
        return {
            "output_path": self.output_path_edit.text().strip(),
            "columns": int(self.columns_spin.value()),
            "rotate": bool(self.rotate_check.isChecked()),
            "selected_filter": self._selected_filter,
        }


class ConnectomeViewer(QMainWindow):
    _global_font_adjusted = False

    def __init__(self) -> None:
        super().__init__()
        self._increase_global_font_size()
        self._workspace = _WorkspaceStore()
        self._entries = self._workspace.entries
        self.titles = self._workspace.titles
        self._data_access = _MatrixDataAccess(
            load_covars_info=_load_covars_info,
            get_valid_keys=_get_valid_keys,
            load_parcel_metadata=_load_parcel_metadata,
            load_group_value=_load_group_value,
            load_matrix_from_npz=_load_matrix_from_npz,
            stack_axis=_stack_axis,
            average_to_square=_average_to_square,
            select_stack_slice=_select_stack_slice,
            to_string_list=_to_string_list,
        )
        self._covars_cache = self._data_access.covars_cache
        self._valid_keys_cache = self._data_access.valid_keys_cache
        self._parcel_metadata_cache = self._data_access.parcel_metadata_cache
        self._group_values_cache = self._data_access.group_values_cache
        self._current_matrix = None
        self._current_parcel_labels = None
        self._current_parcel_names = None
        self._current_axes = None
        self._zoom_selector = None
        self._matrix_full_xlim = None
        self._matrix_full_ylim = None
        self._colorbar = ColorBar()
        self._custom_cmaps = set()
        self._preferences = self._load_preferences()
        self._last_gradients = None
        self._active_parcellation_path = None
        self._active_parcellation_img = None
        self._active_parcellation_data = None
        self._gradients_busy = False
        self._gradients_progress_state = {"minimum": 0, "maximum": 1, "value": 0, "text": "Idle"}
        self._surface_dialog = None
        self._gradient_scatter_dialog = None
        self._gradient_classification_dialog = None
        self._gradients_dialog = None
        self._nbs_dialog = None
        self._selector_dialog = None
        self._harmonize_dialog = None
        self._combine_dialog = None
        self._combine_pending_result = None
        self._batch_import_dialog = None
        self._stack_prepare_dialog = None
        self._label_info_dialog = None
        self._participants_info_dialog = None
        self._left_panel_saved_width = 320
        self._right_panel_saved_width = 240
        self._plot_title_full = ""
        self._plot_title_tooltip = ""
        self._export_grid_columns = 4
        self._export_grid_rotate = False
        self._export_grid_output_path = ""
        self._export_grid_selected_filter = "PDF (*.pdf)"
        self._theme_name = self._preferences["theme"]
        self._default_matrix_colormap = self._preferences["matrix_colormap"]
        self._default_gradient_colormap = self._preferences["gradient_colormap"]
        self._workspace_matrix_controller = _WorkspaceMatrixController(
            self,
            fallback_colormap=DEFAULT_COLORMAP,
            covars_columns=_covars_columns,
        )
        self._workspace_file_controller = _WorkspaceFileController(self)
        self._workspace_file_controller.bind_viewer_methods()
        self._prepare_dialog_controller = _PrepareDialogController(
            self,
            covars_columns=_covars_columns,
            load_matrix_from_npz=_load_matrix_from_npz,
            stack_axis=_stack_axis,
        )
        self._gradient_colormap_name = self._default_gradient_colormap
        self._gradient_selected_entry_id = None
        self._gradient_precomputed_bundle = None
        self._gradient_precomputed_selected_row = None
        self._gradient_use_precomputed_bundle = False
        self._gradient_component_count = 4
        self._gradient_parcel_label_key = ""
        self._gradient_parcel_name_key = ""
        self._gradient_hemisphere_mode = "separate"
        self._gradient_surface_mesh = "fsaverage4"
        self._gradient_surface_render_count = 1
        self._gradient_surface_procrustes = False
        self._gradient_classification_surface_mesh = self._gradient_surface_mesh
        self._gradient_classification_hemisphere_mode = "separate"
        self._gradient_scatter_rotation = "Default"
        self._gradient_scatter_triangular_rgb = False
        self._gradient_preload_matching_paths = False
        self._gradient_classification_fit_mode = "triangle"
        self._gradient_triangular_color_order = "RBG"
        self._gradient_rgb_scalar_mode = "barycentric"
        self._gradient_rgb_mask_strictness = 1.0
        self._gradient_classification_colormap_name = self._default_gradient_colormap
        self._gradient_classification_component = "1"
        self._gradient_classification_x_axis = "gradient2"
        self._gradient_classification_y_axis = "gradient1"
        self._gradient_classification_z_axis = "none"
        self._gradient_classification_ref_entry_id = None
        self._gradient_classification_ignore_lh_parcel = ""
        self._gradient_classification_ignore_rh_parcel = ""
        self._gradient_classification_adjacency_path = ""
        self._gradient_classification_adjacency_cache = None
        self._gradient_network_component = "all"
        self._gradient_component_rotations = ["Default"] * 10
        self._gradient_dialog_controller = _GradientDialogController(
            self,
            parcel_label_keys=PARCEL_LABEL_KEYS,
            parcel_name_keys=PARCEL_NAME_KEYS,
            to_string_list=_to_string_list,
            display_text=_display_text,
            load_covars_info=_load_covars_info,
            covars_to_rows=_covars_to_rows,
            normalize_subject_token=_normalize_subject_token,
            normalize_session_token=_normalize_session_token,
            flatten_display_vector=_flatten_display_vector,
            coerce_label_indices=_coerce_label_indices,
        )
        self._gradient_dialog_controller.bind_viewer_methods()
        self._matlab_cmd_default = self._preferences["matlab_cmd"]
        self._matlab_nbs_path_default = self._preferences["matlab_nbs_path"]
        self._results_dir_default = self._preferences["results_dir"]
        self._bids_dir_default = self._preferences["bids_dir"]
        self._atlas_dir_default = self._preferences["atlas_dir"]
        self._zoom_level = int(self._preferences.get("zoom_level", 0))
        self._base_app_font_point_size = None
        self._hover_vline = None
        self._hover_hline = None
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
            "results_dir": "",
            "bids_dir": _first_env_value("BIDSDATAPATH"),
            "atlas_dir": "",
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
        prefs["results_dir"] = self._normalize_directory_path(str(prefs.get("results_dir", "") or "").strip())
        prefs["bids_dir"] = self._normalize_directory_path(str(prefs.get("bids_dir", "") or "").strip())
        prefs["atlas_dir"] = self._normalize_directory_path(str(prefs.get("atlas_dir", "") or "").strip())
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
            "results_dir": self._results_dir_default,
            "bids_dir": self._bids_dir_default,
            "atlas_dir": self._atlas_dir_default,
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
        self.compute_gradients_action = QAction("Gradients", self)
        self.compute_gradients_action.triggered.connect(self._open_gradients_dialog)
        analysis_menu.addAction(self.compute_gradients_action)

        self.nbs_prepare_action = QAction("NBS Prepare", self)
        self.nbs_prepare_action.triggered.connect(self._open_nbs_prepare_dialog)
        analysis_menu.addAction(self.nbs_prepare_action)

        self.connectivity_compute_action = QAction("Compute Connectivity", self)
        self.connectivity_compute_action.triggered.connect(self._open_connectivity_compute_folder)
        analysis_menu.addAction(self.connectivity_compute_action)

        self.harmonize_prepare_action = QAction("Harmonize Prepare", self)
        self.harmonize_prepare_action.triggered.connect(self._open_harmonize_prepare_dialog)
        analysis_menu.addAction(self.harmonize_prepare_action)

        settings_menu = bar.addMenu("Settings")
        preferences_action = QAction("Preferences", self)
        preferences_action.triggered.connect(self._open_preferences_dialog)
        settings_menu.addAction(preferences_action)

        self._update_nbs_prepare_button()
        self._update_gradients_button()

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

    def _open_preferences_dialog(self, _checked=False, *, initial_setup: bool = False) -> bool:
        names = self._available_colormap_names()
        dialog = PreferencesDialog(
            theme_name=self._theme_name,
            matrix_cmap=self._default_matrix_colormap,
            gradient_cmap=self._default_gradient_colormap,
            matlab_cmd=self._matlab_cmd_default,
            matlab_nbs_path=self._matlab_nbs_path_default,
            results_dir=self._results_dir_default,
            bids_dir=self._bids_dir_default,
            atlas_dir=self._atlas_dir_default,
            colormap_names=names,
            dialog_title="Initial Configuration" if initial_setup else "Preferences",
            require_results_dir=True,
            parent=self,
        )
        if dialog.exec() != _dialog_accepted_code():
            return False
        new_values = dialog.values()
        new_values["zoom_level"] = self._zoom_level
        prefs = self._sanitize_preferences(new_values)
        self._theme_name = prefs["theme"]
        self._default_matrix_colormap = prefs["matrix_colormap"]
        self._default_gradient_colormap = prefs["gradient_colormap"]
        self._gradient_colormap_name = self._default_gradient_colormap
        self._matlab_cmd_default = prefs["matlab_cmd"]
        self._matlab_nbs_path_default = prefs["matlab_nbs_path"]
        self._results_dir_default = prefs["results_dir"]
        self._bids_dir_default = prefs["bids_dir"]
        self._atlas_dir_default = prefs["atlas_dir"]
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
        self._sync_gradients_dialog_state()

        if self._save_preferences():
            self.statusBar().showMessage(f"Preferences saved to {CONFIG_PATH}.")
        return True

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
        self.cmap_combo.currentIndexChanged.connect(self._on_colormap_changed)

        list_group = QGroupBox("Matrices")
        list_layout = QVBoxLayout(list_group)
        self.add_button = QPushButton("Add Files")
        self.add_button.clicked.connect(self._open_files)
        list_layout.addWidget(self.add_button)

        self.add_batch_button = QPushButton("Add Batch")
        self.add_batch_button.clicked.connect(self._open_batch_folder)
        list_layout.addWidget(self.add_batch_button)

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
        self.remove_button = QPushButton("Remove Selected")
        self.remove_button.clicked.connect(self._remove_selected)
        move_layout.addWidget(self.remove_button)
        list_layout.addLayout(move_layout)

        hint = QLabel("Drag & drop .npz files here.")
        hint.setWordWrap(True)
        list_layout.addWidget(hint)

        info_group = QGroupBox("Info")
        info_layout = QVBoxLayout(info_group)
        self.view_labels_button = QPushButton("View Labels")
        self.view_labels_button.clicked.connect(self._open_label_info_dialog)
        self.view_labels_button.setEnabled(False)
        info_layout.addWidget(self.view_labels_button)
        self.view_participants_button = QPushButton("View Participants")
        self.view_participants_button.clicked.connect(self._open_participants_info_dialog)
        self.view_participants_button.setEnabled(False)
        info_layout.addWidget(self.view_participants_button)

        # Connectivity tile is added to the analysis grid below.
        try:
            from window.compute_connectivity import ConnectivityDropButton
            from window.icon_tile import SquareIconTile
        except Exception:
            from mrsi_viewer.window.compute_connectivity import ConnectivityDropButton
            from mrsi_viewer.window.icon_tile import SquareIconTile
        self.connectivity_compute_button = ConnectivityDropButton("Connectivity")
        self.connectivity_compute_button.clicked.connect(self._open_connectivity_compute_folder)
        self.connectivity_compute_button.folder_dropped.connect(self._open_compute_connectivity_dialog)
        # Deliberately always enabled: this computes matrices from profiles, so
        # unlike the analysis buttons it does not need matrices loaded first.
        self.connectivity_compute_button.setEnabled(True)
        self._configure_right_panel_icon_button(self.connectivity_compute_button, "Connectivity")

        # Gradients tile is added to the analysis grid below.
        self.gradients_open_button = SquareIconTile("Gradients")
        self.gradients_open_button.clicked.connect(self._open_gradients_dialog)
        self.gradients_open_button.setEnabled(False)
        self._configure_right_panel_icon_button(self.gradients_open_button, "Gradients")

        # NBS tile is added to the analysis grid below.
        self.nbs_prepare_button = SquareIconTile("Prepare")
        self.nbs_prepare_button.clicked.connect(self._open_nbs_prepare_dialog)
        self.nbs_prepare_button.setEnabled(False)
        self._configure_right_panel_icon_button(self.nbs_prepare_button, "NBS")

        # Harmonize tile is added to the analysis grid below.
        self.harmonize_prepare_button = SquareIconTile("Prepare")
        self.harmonize_prepare_button.clicked.connect(self._open_harmonize_prepare_dialog)
        self.harmonize_prepare_button.setEnabled(False)
        self._configure_right_panel_icon_button(self.harmonize_prepare_button, "Harmonize")

        # Combine tile is added to the analysis grid below.
        self.combine_open_button = SquareIconTile("Combine")
        self.combine_open_button.clicked.connect(self._open_combine_dialog)
        self.combine_open_button.setEnabled(False)
        self._configure_right_panel_icon_button(self.combine_open_button, "Combine")

        self._reload_colormaps()

        self._styled_groups = [
            list_group,
            selector_group,
            info_group,
            export_group,
            # The analysis tiles no longer live in group boxes; they are
            # styled by the #rightPanelIconButton rule instead.
        ]

        controls_layout.addWidget(list_group)
        controls_layout.addWidget(selector_group)
        controls_layout.addWidget(info_group)
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

        plot_toolbar.addWidget(QLabel("Scale:"))
        self.display_scale_combo = QComboBox()
        self.display_scale_combo.addItems(["Linear", "Log"])
        self.display_scale_combo.setCurrentText("Linear")
        self.display_scale_combo.currentTextChanged.connect(self._on_display_scaling_changed)
        self.display_scale_combo.setFixedWidth(78 if compact_ui else 92)
        plot_toolbar.addWidget(self.display_scale_combo)

        self.zoom_region_check = QCheckBox("Zoom region")
        self.zoom_region_check.setToolTip("Drag on the matrix to zoom into a region.")
        self.zoom_region_check.stateChanged.connect(self._on_zoom_region_toggled)
        plot_toolbar.addWidget(self.zoom_region_check)

        self.zoom_reset_button = QPushButton("Reset view")
        self.zoom_reset_button.setEnabled(False)
        self.zoom_reset_button.clicked.connect(self._reset_matrix_zoom)
        plot_toolbar.addWidget(self.zoom_reset_button)

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
        # One grid of square icon tiles instead of a column of full-width
        # buttons in individual group boxes, following EEG-UP's compute
        # toolbar. The per-function group titles went with it: they duplicated
        # the tooltip and forced every tile to span the whole panel width,
        # which is what made the icons read as small and stranded.
        analysis_tiles = QWidget()
        analysis_grid = QGridLayout(analysis_tiles)
        analysis_grid.setContentsMargins(4, 4, 4, 4)
        analysis_grid.setSpacing(6)
        tiles = [
            (self.connectivity_compute_button, "Connectivity"),
            (self.gradients_open_button, "Gradients"),
            (self.nbs_prepare_button, "NBS"),
            (self.harmonize_prepare_button, "Harmonize"),
            (self.combine_open_button, "Combine"),
        ]
        columns = 2
        for index, (tile_button, tile_label) in enumerate(tiles):
            # Already configured as a tile where each button is created; only
            # the placement is decided here.
            del tile_label
            analysis_grid.addWidget(tile_button, index // columns, index % columns)
        for column in range(columns):
            analysis_grid.setColumnStretch(column, 1)
        right_panel_layout.addWidget(analysis_tiles)
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

    def _right_panel_tile_stylesheet(self) -> str:
        """Card look for the right-panel icon tiles.

        Appended to whichever theme string is active rather than repeated in
        all four, so a tile tweak is one edit instead of four.
        """
        if self._theme_name == "Dark":
            bg, border, hover = "#2a3140", "#4b5668", "#60a5fa"
        elif self._theme_name == "Teya":
            bg, border, hover = "#ffffff", "#bfe9e7", "#0b7f7a"
        elif self._theme_name == "Donald":
            bg, border, hover = "#2d2a26", "#6b5c4a", "#ffd19e"
        else:
            bg, border, hover = "#ffffff", "#d8dee9", "#2563eb"
        return (
            "QPushButton#rightPanelIconButton {"
            f"background: {bg}; border: 1px solid {border}; border-radius: 10px; padding: 6px;"
            "}"
            f"QPushButton#rightPanelIconButton:hover {{ border: 2px solid {hover}; }}"
            f"QPushButton#rightPanelIconButton:pressed {{ background: {border}; }}"
            "QPushButton#rightPanelIconButton:disabled { border-style: dashed; }"
        )

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
                "QPushButton#workflowStepButton { text-align: left; padding-left: 10px; }"
                "QPushButton#workflowStepButton:checked { background: #2563eb; border: 2px solid #60a5fa; color: #ffffff; font-weight: 600; }"
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
                "QPushButton#workflowStepButton { text-align: left; padding-left: 10px; }"
                "QPushButton#workflowStepButton:checked { background: #2ecfc9; border: 2px solid #0b7f7a; color: #073f3c; font-weight: 700; }"
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
                "QPushButton#workflowStepButton { text-align: left; padding-left: 10px; }"
                "QPushButton#workflowStepButton:checked { background: #b85f00; border: 2px solid #ffd19e; color: #ffffff; font-weight: 700; }"
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
            "QPushButton#workflowStepButton { text-align: left; padding-left: 10px; }"
            "QPushButton#workflowStepButton:checked { background: #2563eb; border: 2px solid #1d4ed8; color: #ffffff; font-weight: 600; }"
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

        self.setStyleSheet(self._base_theme_stylesheet() + self._right_panel_tile_stylesheet())
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
        if getattr(self, "_gradients_dialog", None) is not None and hasattr(self._gradients_dialog, "set_theme"):
            try:
                self._gradients_dialog.set_theme(theme)
            except Exception:
                pass
        if getattr(self, "_harmonize_dialog", None) is not None and hasattr(self._harmonize_dialog, "set_theme"):
            try:
                self._harmonize_dialog.set_theme(theme)
            except Exception:
                pass
        if getattr(self, "_combine_dialog", None) is not None and hasattr(self._combine_dialog, "set_theme"):
            try:
                self._combine_dialog.set_theme(theme)
            except Exception:
                pass
        if getattr(self, "_stack_prepare_dialog", None) is not None and hasattr(
            self._stack_prepare_dialog, "set_theme"
        ):
            try:
                self._stack_prepare_dialog.set_theme(theme)
            except Exception:
                pass
        if getattr(self, "_batch_import_dialog", None) is not None:
            try:
                if hasattr(self._batch_import_dialog, "set_theme"):
                    self._batch_import_dialog.set_theme(theme)
                else:
                    self._batch_import_dialog.setStyleSheet(self.styleSheet())
            except Exception:
                pass
        if getattr(self, "_surface_dialog", None) is not None and hasattr(
            self._surface_dialog, "set_theme"
        ):
            try:
                self._surface_dialog.set_theme(theme)
            except Exception:
                pass
        if getattr(self, "_gradient_scatter_dialog", None) is not None and hasattr(
            self._gradient_scatter_dialog, "set_theme"
        ):
            try:
                self._gradient_scatter_dialog.set_theme(theme)
            except Exception:
                pass
        if getattr(self, "_gradient_classification_dialog", None) is not None and hasattr(
            self._gradient_classification_dialog, "set_theme"
        ):
            try:
                self._gradient_classification_dialog.set_theme(theme)
            except Exception:
                pass
        if getattr(self, "_label_info_dialog", None) is not None:
            try:
                self._label_info_dialog.setStyleSheet(self.styleSheet())
            except Exception:
                pass
        if getattr(self, "_participants_info_dialog", None) is not None:
            try:
                self._participants_info_dialog.setStyleSheet(self.styleSheet())
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
        try:
            from window.icon_tile import SquareIconTile
        except Exception:
            from mrsi_viewer.window.icon_tile import SquareIconTile
        icon_size = QSize(16, 16) if compact_ui else QSize(18, 18)
        right_panel_icon_size = (
            QSize(self._TILE_ICON_SIZE_COMPACT, self._TILE_ICON_SIZE_COMPACT)
            if compact_ui
            else QSize(self._TILE_ICON_SIZE, self._TILE_ICON_SIZE)
        )
        mapping = [
            (getattr(self, "add_button", None), "folder_plus.svg"),
            (getattr(self, "add_batch_button", None), "folder_plus.svg"),
            (getattr(self, "remove_button", None), "trash.svg"),
            (getattr(self, "clear_button", None), "broom_clear.svg"),
            (getattr(self, "hist_button", None), "histogram.svg"),
            (getattr(self, "view_labels_button", None), "info.svg"),
            (getattr(self, "view_participants_button", None), "info.svg"),
            (getattr(self, "export_button", None), "export_grid.svg"),
            (getattr(self, "write_matrix_button", None), "save_disk.svg"),
            (getattr(self, "move_up_button", None), "arrow_up.svg"),
            (getattr(self, "move_down_button", None), "arrow_down.svg"),
            (getattr(self, "connectivity_compute_button", None), "connectivity.svg"),
            (getattr(self, "gradients_open_button", None), "gradient.png"),
            (getattr(self, "nbs_prepare_button", None), "nbs.svg"),
            (getattr(self, "harmonize_prepare_button", None), "harmonize.svg"),
            (getattr(self, "combine_open_button", None), "combine.svg"),
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
            if isinstance(button, SquareIconTile):
                continue  # the tile rescales its own icon from its width
            if button.objectName() == "rightPanelIconButton":
                button.setIconSize(right_panel_icon_size)
            else:
                button.setIconSize(icon_size)

    #: Square icon tiles, following EEG-UP's compute toolbar
    #: (ui/pages/workflow_page.py). The icon carries the meaning, so the tile
    #: is sized from the icon rather than from text metrics, and the label
    #: survives only as a tooltip.
    _TILE_ICON_SIZE = 72
    _TILE_ICON_SIZE_COMPACT = 56
    _TILE_MIN_SIZE = 104
    _TILE_MIN_SIZE_COMPACT = 84

    def _configure_right_panel_icon_button(self, button, label: str) -> None:
        try:
            from window.icon_tile import SquareIconTile
        except Exception:
            from mrsi_viewer.window.icon_tile import SquareIconTile
        button.setText("")
        button.setObjectName("rightPanelIconButton")
        if not str(button.toolTip() or "").strip():
            button.setToolTip(label)
        button.setStatusTip(label)
        try:
            button.setAccessibleName(label)
        except Exception:
            pass
        # Size is the tile's own business: SquareIconTile keeps itself square
        # and derives its icon size from its width, so the tiles track the side
        # panel as it is dragged. Only the floor is set from here.
        side = self._TILE_MIN_SIZE_COMPACT if self._compact_ui else self._TILE_MIN_SIZE
        if isinstance(button, SquareIconTile):
            button.MIN_SIDE = side
            button.setMinimumSize(QSize(side, side))
        else:
            button.setFixedSize(QSize(side, side))

    def _file_entry_id(self, path: Path) -> str:
        return self._workspace.file_entry_id(path)

    def _new_derived_id(self) -> str:
        return self._workspace.new_derived_id()

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

    def _add_workspace_list_item(self, entry, *, tooltip=None, select: bool = True):
        if not isinstance(entry, dict):
            return None
        item = QListWidgetItem(str(entry.get("label") or entry.get("id") or ""))
        item.setData(USER_ROLE, entry.get("id"))
        if tooltip is not None:
            item.setToolTip(str(tooltip))
        self.file_list.addItem(item)
        if select:
            self.file_list.setCurrentItem(item)
        return item

    def _select_workspace_entry(self, entry_id) -> bool:
        for row in range(self.file_list.count()):
            item = self.file_list.item(row)
            if item is None:
                continue
            if item.data(USER_ROLE) == entry_id:
                self.file_list.setCurrentItem(item)
                return True
        return False

    def _current_source_path(self):
        entry = self._current_entry()
        if entry is None:
            return None
        source_path = entry.get("source_path", entry.get("path"))
        if not source_path:
            return None
        return Path(source_path)

    def ensure_initial_configuration(self) -> bool:
        if str(self._results_dir_default or "").strip():
            return True
        return self._open_preferences_dialog(initial_setup=True)

    def _default_open_dir(self) -> Path:
        bids_dir = str(self._bids_dir_default or "").strip()
        if bids_dir:
            return Path(bids_dir).expanduser()
        source_path = self._current_source_path()
        if source_path is not None:
            return source_path.parent
        return Path.cwd()

    def _default_results_dir(self) -> Path:
        results_dir = str(self._results_dir_default or "").strip()
        if results_dir:
            return Path(results_dir).expanduser()
        source_path = self._current_source_path()
        if source_path is not None:
            return source_path.parent
        return Path.cwd()

    def _default_dialog_dir(self) -> Path:
        return self._default_open_dir()

    def _default_parcellation_dir(self) -> Path:
        atlas_dir = str(self._atlas_dir_default or "").strip()
        if atlas_dir:
            return Path(atlas_dir).expanduser()
        if DEFAULT_PARCELLATION_DIR.exists():
            return DEFAULT_PARCELLATION_DIR
        return ROOTDIR

    def _update_view_labels_button(self) -> None:
        if not hasattr(self, "view_labels_button"):
            return
        source_path = self._current_source_path()
        self.view_labels_button.setEnabled(source_path is not None and source_path.exists())
        self._update_view_participants_button()

    def _update_view_participants_button(self) -> None:
        if not hasattr(self, "view_participants_button"):
            return
        source_path = self._current_source_path()
        enabled = False
        if source_path is not None and source_path.exists():
            info = self._covars_info_cached(source_path)
            enabled = bool(_covars_columns(info))
        self.view_participants_button.setEnabled(enabled)

    def _open_label_info_dialog(self) -> None:
        source_path = self._current_source_path()
        if source_path is None or not source_path.exists():
            self.statusBar().showMessage("Select a matrix backed by an NPZ file to inspect labels.")
            return
        try:
            dialog = LabelInfoDialog(source_path, parent=self)
            dialog.setStyleSheet(self.styleSheet())
            dialog.finished.connect(lambda *_args: setattr(self, "_label_info_dialog", None))
            self._label_info_dialog = dialog
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()
            self.statusBar().showMessage(f"Opened label info for {source_path.name}.")
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to open label info: {exc}")

    def _open_participants_info_dialog(self) -> None:
        source_path = self._current_source_path()
        if source_path is None or not source_path.exists():
            self.statusBar().showMessage("Select a matrix backed by an NPZ file to inspect participants.")
            return
        covars_info = self._covars_info_cached(source_path)
        if covars_info is None:
            self.statusBar().showMessage("Covars not found in selected file.")
            return
        columns, rows = _covars_to_rows(covars_info)
        if not columns:
            self.statusBar().showMessage("Selected file has no tabular covars to display.")
            return
        try:
            dialog = ParticipantsInfoDialog(source_path, covars_info, parent=self)
            dialog.setStyleSheet(self.styleSheet())
            dialog.finished.connect(lambda *_args: setattr(self, "_participants_info_dialog", None))
            self._participants_info_dialog = dialog
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()
            self.statusBar().showMessage(
                f"Opened participants view for {source_path.name} ({len(rows)} rows)."
            )
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to open participants view: {exc}")

    def _update_parcellation_label(self) -> None:
        if getattr(self, "_gradients_dialog", None) is not None:
            try:
                self._gradients_dialog.set_parcellation_path(self._active_parcellation_path)
            except Exception:
                pass

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
        self._reset_gradients_output()
        return True




    @staticmethod

    @staticmethod

    @staticmethod

    @staticmethod

    @staticmethod

    @staticmethod

    @staticmethod

    @staticmethod

    @staticmethod

    @staticmethod

    @staticmethod






















    @staticmethod

    @staticmethod



























    @staticmethod





    @staticmethod

    @staticmethod

    @staticmethod

    @staticmethod


    @staticmethod





    @staticmethod

















    def _current_nbs_source(self):
        return self._prepare_dialog_controller.current_nbs_source()

    def _update_nbs_prepare_button(self) -> None:
        self._prepare_dialog_controller.update_nbs_prepare_button()

    def _current_selector_source(self):
        return self._prepare_dialog_controller.current_selector_source()

    def _current_harmonize_source(self):
        return self._prepare_dialog_controller.current_harmonize_source()

    def _update_selector_prepare_button(self) -> None:
        self._prepare_dialog_controller.update_selector_prepare_button()

    def _update_harmonize_prepare_button(self) -> None:
        self._prepare_dialog_controller.update_harmonize_prepare_button()

    def _update_write_to_file_button(self) -> None:
        self._prepare_dialog_controller.update_write_to_file_button()

    def _open_nbs_prepare_dialog(self) -> None:
        self._prepare_dialog_controller.open_nbs_prepare_dialog()

    def _current_gradient_rgb_payload(self):
        dialog = getattr(self, "_gradient_scatter_dialog", None)
        if dialog is None or not hasattr(dialog, "triangular_rgb_model_payload"):
            return None
        try:
            return dialog.triangular_rgb_model_payload()
        except Exception:
            return None

    def _open_selector_prepare_dialog(self) -> None:
        self._prepare_dialog_controller.open_selector_prepare_dialog()

    def _open_harmonize_prepare_dialog(self) -> None:
        self._prepare_dialog_controller.open_harmonize_prepare_dialog()

    def _open_combine_dialog(self) -> None:
        if not self._available_combine_matrix_entries():
            self.statusBar().showMessage("Add at least one workspace matrix before opening Combine.")
            return

        if getattr(self, "_combine_dialog", None) is None:
            try:
                from window.combine_prepare import CombinePrepareDialog
            except Exception:
                try:
                    from mrsi_viewer.window.combine_prepare import CombinePrepareDialog
                except Exception as exc:
                    self.statusBar().showMessage(f"Failed to open Combine window: {exc}")
                    return

            self._combine_dialog = CombinePrepareDialog(
                theme_name=self._theme_name,
                process_callback=self._process_combine_operation,
                confirm_callback=self._confirm_combine_result,
                parent=self,
            )

        self._sync_combine_dialog_state()
        self._combine_dialog.show()
        try:
            self._combine_dialog.raise_()
            self._combine_dialog.activateWindow()
        except Exception:
            pass
        self.statusBar().showMessage("Opened Combine window.")

    @staticmethod
    def _combine_operation_label(operation: str) -> str:
        return _combine_op_label(operation)

    @staticmethod
    def _combine_operation_symbol(operation: str) -> str:
        return _combine_op_symbol(operation)

    def _format_p_value(self, value) -> str:
        return _format_combine_p_value(value)

    def _align_combine_matrices_by_intersection(self, first_entry, first_matrix, second_entry, second_matrix):
        first_labels_raw, first_names = self._entry_parcel_metadata(first_entry, expected_len=first_matrix.shape[0])
        second_labels_raw, second_names = self._entry_parcel_metadata(second_entry, expected_len=second_matrix.shape[0])
        alignment = _combine_align_matrices(
            first_matrix,
            second_matrix,
            first_labels_raw=first_labels_raw,
            second_labels_raw=second_labels_raw,
            first_names=first_names,
            second_names=second_names,
            coerce_label_indices=_coerce_label_indices,
        )
        return (
            alignment.first_matrix,
            alignment.second_matrix,
            alignment.parcel_labels_group,
            alignment.parcel_names_group,
            alignment.summary_text,
            alignment.used_label_intersection,
        )

    def _process_combine_operation(self, first_entry_id, second_entry_id, operation) -> None:
        dialog = getattr(self, "_combine_dialog", None)
        self._combine_pending_result = None
        if dialog is not None:
            dialog.set_can_confirm(False)
        operation_name = str(operation or "").strip().lower()
        first_entry = self._entries.get(first_entry_id)
        second_entry = self._entries.get(second_entry_id) if second_entry_id is not None else None
        if first_entry is None or (operation_name != "spatial_correction" and second_entry is None):
            message = "Select two valid workspace matrices."
            if operation_name == "spatial_correction":
                message = "Select a valid workspace matrix."
            if dialog is not None:
                dialog.set_status(message)
            self.statusBar().showMessage(message)
            return
        if operation_name == "spatial_correction":
            self._process_combine_spatial_correction(first_entry_id)
            return

        first_label = self._gradient_matrix_label_for_entry(first_entry)
        second_label = self._gradient_matrix_label_for_entry(second_entry)
        operation_label = self._combine_operation_label(operation)

        try:
            first_matrix, _ = self._matrix_for_entry(first_entry)
            second_matrix, _ = self._matrix_for_entry(second_entry)
            first_matrix = np.asarray(first_matrix, dtype=float)
            second_matrix = np.asarray(second_matrix, dtype=float)
            (
                first_matrix,
                second_matrix,
                aligned_labels,
                aligned_names,
                alignment_summary,
                used_label_intersection,
            ) = self._align_combine_matrices_by_intersection(
                first_entry,
                first_matrix,
                second_entry,
                second_matrix,
            )
        except Exception as exc:
            message = f"Failed to load matrices for {operation_label}: {exc}"
            if dialog is not None:
                dialog.set_status(message)
            self.statusBar().showMessage(message)
            return

        try:
            operation_name = str(operation or "").strip().lower()
            if operation_name == "correlation":
                self._show_combine_correlation(
                    first_matrix,
                    second_matrix,
                    first_label=first_label,
                    second_label=second_label,
                    alignment_summary=alignment_summary,
                )
                return

            if operation_name == "intersect" and not used_label_intersection:
                raise ValueError(
                    "Intersect requires square matrices with valid parcel_labels_group in both inputs."
                )

            correction_summary = ""
            if operation_name == "correct":
                correction_stats = self._combine_residualize_matrix(first_matrix, second_matrix)
                result_matrix = np.asarray(correction_stats.residual_matrix, dtype=float)
                correction_summary = (
                    f"regressed Matrix A on Matrix B using {correction_stats.mode_text}; "
                    f"n={correction_stats.first_values.size}; "
                    f"beta={float(correction_stats.slope):.4g}; "
                    f"intercept={float(correction_stats.intercept):.4g}"
                )
            else:
                result_matrix = self._combine_apply_matrix_operation(first_matrix, second_matrix, operation)
            finite_values = np.asarray(result_matrix, dtype=float)
            finite_values = finite_values[np.isfinite(finite_values)]
            if finite_values.size:
                min_text = f"{float(np.min(finite_values)):.4g}"
                max_text = f"{float(np.max(finite_values)):.4g}"
            else:
                min_text = "n/a"
                max_text = "n/a"
            if operation_name == "intersect":
                result_label = f"intersect({first_label}, {second_label})"
            elif operation_name == "correct":
                result_label = f"correct({first_label}, {second_label})"
            else:
                result_label = f"{first_label} {self._combine_operation_symbol(operation)} {second_label}"
            result_labels = aligned_labels
            result_names = aligned_names
            if result_matrix.ndim == 2 and result_matrix.shape[0] == result_matrix.shape[1]:
                if result_labels is None:
                    result_labels, result_names = self._entry_parcel_metadata(
                        first_entry,
                        expected_len=result_matrix.shape[0],
                    )
            result_summary = (
                f"{operation_label} result | shape={tuple(result_matrix.shape)} | "
                f"{alignment_summary} | "
                f"{correction_summary + ' | ' if correction_summary else ''}"
                f"min={min_text} | max={max_text}"
            )
            source_path = first_entry.get("source_path", first_entry.get("path"))
            self._combine_pending_result = {
                "matrix": np.asarray(result_matrix, dtype=float),
                "label": result_label,
                "summary_text": result_summary,
                "source_path": source_path,
                "operation": str(operation or ""),
                "first_entry_id": first_entry_id,
                "second_entry_id": second_entry_id,
                "parcel_labels_group": np.asarray(result_labels) if result_labels is not None else None,
                "parcel_names_group": np.asarray(result_names, dtype=object) if result_names is not None else None,
            }

            if dialog is not None:
                dialog.show_matrix_result(
                    result_matrix,
                    title=result_label,
                    summary_text=result_summary,
                )
                dialog.set_status("Processed matrix operation. Click Add to Workspace to confirm.")
            self.statusBar().showMessage(
                f"{operation_label} complete. Preview ready; confirm to add it to the workspace."
            )
        except Exception as exc:
            message = f"{operation_label} failed: {exc}"
            if dialog is not None:
                dialog.set_status(message)
            self.statusBar().showMessage(message)

    def _process_combine_spatial_correction(self, first_entry_id) -> None:
        dialog = getattr(self, "_combine_dialog", None)
        self._combine_pending_result = None
        if dialog is not None and hasattr(dialog, "clear_progress"):
            dialog.clear_progress()
        first_entry = self._entries.get(first_entry_id)
        if first_entry is None:
            message = "Select a valid workspace matrix."
            if dialog is not None:
                dialog.set_status(message)
                dialog.set_can_confirm(False)
            self.statusBar().showMessage(message)
            return

        operation = "spatial_correction"
        operation_label = self._combine_operation_label(operation)
        first_label = self._gradient_matrix_label_for_entry(first_entry)
        try:
            selected_key, source_payload, selected_matrix = self._combine_source_npz_payload(first_entry)
            selected_matrix = np.asarray(selected_matrix)
            selected_node_count = self._combine_matrix_node_count(selected_matrix)
            if selected_node_count is None:
                raise ValueError("Spatial correction requires a square matrix or a 3D stack of square matrices.")
            result_labels, result_names = self._entry_parcel_metadata(
                first_entry,
                expected_len=selected_node_count,
            )
            template_img, template_data, template_path = self._combine_spatial_parcellation_img_and_data()
            result_labels = self._combine_spatial_correction_labels(
                result_labels,
                template_data,
                selected_node_count,
            )
            parcel_coordinates, spatial_labels = _combine_parcellation_centroids(
                template_data,
                result_labels,
                affine=np.asarray(template_img.affine, dtype=float),
            )
            distance_matrix = _combine_pairwise_distance_matrix(parcel_coordinates)
            compatible_keys = self._combine_spatial_compatible_matrix_keys(source_payload, selected_node_count)
            if selected_key not in compatible_keys:
                compatible_keys.insert(0, selected_key)
            total_matrices = sum(
                self._combine_matrix_slice_count(np.asarray(source_payload[key]))
                for key in compatible_keys
            )
            if total_matrices <= 0:
                raise ValueError("No matrix slices were available for spatial correction.")
            corrected_payload = {
                key: np.array(value, copy=True)
                for key, value in source_payload.items()
            }
            progress_value = 0
            beta_values = []
            intercept_values = []
            first_value_count = 0

            def _advance_progress():
                nonlocal progress_value
                progress_value += 1
                if dialog is not None and hasattr(dialog, "set_progress"):
                    dialog.set_progress(
                        0,
                        total_matrices,
                        progress_value,
                        f"{progress_value}/{total_matrices} matrices",
                    )
                try:
                    QApplication.processEvents()
                except Exception:
                    pass

            for key in compatible_keys:
                corrected_array, key_stats = self._combine_spatial_correct_matrix_array(
                    source_payload[key],
                    distance_matrix,
                    progress_callback=_advance_progress,
                )
                corrected_payload[key] = corrected_array
                beta_values.extend(key_stats.get("slopes", []))
                intercept_values.extend(key_stats.get("intercepts", []))
                first_value_count += int(key_stats.get("n_values", 0))

            result_matrix = np.asarray(corrected_payload[selected_key], dtype=float)
        except Exception as exc:
            message = f"{operation_label} failed: {exc}"
            if dialog is not None:
                dialog.set_status(message)
                dialog.set_can_confirm(False)
                if hasattr(dialog, "clear_progress"):
                    dialog.clear_progress()
            self.statusBar().showMessage(message)
            return

        display_matrix = _average_to_square(result_matrix) if result_matrix.ndim == 3 else result_matrix
        finite_values = np.asarray(display_matrix, dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        min_text = f"{float(np.min(finite_values)):.4g}" if finite_values.size else "n/a"
        max_text = f"{float(np.max(finite_values)):.4g}" if finite_values.size else "n/a"
        result_label = f"spatial_correct({first_label})"
        template_name = Path(template_path).name if template_path is not None else "active parcellation"
        beta_text = (
            f"{float(np.mean(beta_values)):.4g}"
            if beta_values
            else "n/a"
        )
        intercept_text = (
            f"{float(np.mean(intercept_values)):.4g}"
            if intercept_values
            else "n/a"
        )
        result_summary = (
            f"{operation_label} result | shape={tuple(result_matrix.shape)} | "
            f"parcellation={template_name} | "
            f"corrected {len(compatible_keys)} matrix key(s), {total_matrices} matrix slice(s); "
            f"regressed each matrix on parcel centroid distance using upper triangle excluding diagonal; "
            f"n={first_value_count}; "
            f"mean beta={beta_text}; "
            f"mean intercept={intercept_text} | "
            f"min={min_text} | max={max_text}"
        )
        source_path = first_entry.get("source_path", first_entry.get("path"))
        self._combine_pending_result = {
            "matrix": result_matrix,
            "label": result_label,
            "summary_text": result_summary,
            "source_path": source_path,
            "operation": operation,
            "first_entry_id": first_entry_id,
            "second_entry_id": None,
            "selected_key": selected_key,
            "sample_index": first_entry.get("sample_index", -1) if result_matrix.ndim == 3 else None,
            "stack_axis": _stack_axis(result_matrix.shape) if result_matrix.ndim == 3 else None,
            "stack_len": (
                result_matrix.shape[_stack_axis(result_matrix.shape)]
                if result_matrix.ndim == 3 and _stack_axis(result_matrix.shape) is not None
                else None
            ),
            "parcel_labels_group": np.asarray(spatial_labels),
            "parcel_names_group": np.asarray(result_names, dtype=object) if result_names is not None else None,
            "spatial_correction_parcellation": str(template_path) if template_path is not None else "",
            "spatial_correction_keys": np.asarray(compatible_keys, dtype=object),
            "npz_payload": corrected_payload,
        }

        if dialog is not None:
            dialog.show_matrix_result(
                display_matrix,
                title=result_label,
                summary_text=result_summary,
            )
            dialog.set_status("Processed spatial correction. Click Add to Workspace to confirm.")
        self.statusBar().showMessage(
            f"{operation_label} complete. Preview ready; confirm to add it to the workspace."
        )

    def _combine_source_npz_payload(self, entry):
        if entry.get("kind") == "file":
            selected_key = self._ensure_entry_key(entry)
            if not selected_key:
                raise KeyError("No valid matrix key selected.")
            source_path = Path(entry["path"])
            payload = {}
            with np.load(source_path, allow_pickle=True) as npz:
                if selected_key not in npz:
                    raise KeyError(f"Key '{selected_key}' not found.")
                for key in npz.files:
                    payload[key] = np.array(npz[key], copy=True)
            return selected_key, payload, payload[selected_key]

        selected_key = str(entry.get("selected_key") or "").strip() or "matrix"
        matrix = np.asarray(entry.get("matrix"))
        payload = {}
        stored_payload = entry.get("npz_payload")
        if isinstance(stored_payload, dict):
            for key, value in stored_payload.items():
                payload[str(key)] = np.array(value, copy=True)
        if selected_key not in payload:
            payload[selected_key] = np.array(matrix, copy=True)
        return selected_key, payload, payload[selected_key]

    @staticmethod
    def _combine_matrix_node_count(matrix_array):
        array = np.asarray(matrix_array)
        if array.ndim == 2 and array.shape[0] == array.shape[1]:
            return int(array.shape[0])
        if array.ndim == 3:
            axis = _stack_axis(array.shape)
            if axis is None:
                return None
            dims = [int(array.shape[idx]) for idx in range(3) if idx != axis]
            if len(dims) == 2 and dims[0] == dims[1]:
                return int(dims[0])
        return None

    @staticmethod
    def _combine_matrix_slice_count(matrix_array):
        array = np.asarray(matrix_array)
        if array.ndim == 2 and array.shape[0] == array.shape[1]:
            return 1
        if array.ndim == 3:
            axis = _stack_axis(array.shape)
            if axis is not None:
                return int(array.shape[axis])
        return 0

    def _combine_spatial_compatible_matrix_keys(self, payload, node_count):
        compatible = []
        for key, value in dict(payload or {}).items():
            try:
                if self._combine_matrix_node_count(np.asarray(value)) == int(node_count):
                    compatible.append(str(key))
            except Exception:
                continue
        return compatible

    @staticmethod
    def _combine_assign_stack_slice(output, axis, index, matrix):
        if axis == 0:
            output[index, :, :] = matrix
        elif axis == 1:
            output[:, index, :] = matrix
        else:
            output[:, :, index] = matrix

    def _combine_spatial_correct_matrix_array(self, matrix_array, distance_matrix, *, progress_callback=None):
        array = np.asarray(matrix_array, dtype=float)
        stats = {"slopes": [], "intercepts": [], "n_values": 0}
        if array.ndim == 2:
            residual_stats = self._combine_residualize_matrix(array, distance_matrix)
            stats["slopes"].append(float(residual_stats.slope))
            stats["intercepts"].append(float(residual_stats.intercept))
            stats["n_values"] += int(residual_stats.first_values.size)
            if progress_callback is not None:
                progress_callback()
            return np.asarray(residual_stats.residual_matrix, dtype=float), stats

        axis = _stack_axis(array.shape)
        if axis is None:
            raise ValueError(f"Cannot identify matrix stack axis for shape {array.shape}.")
        corrected = np.empty(array.shape, dtype=float)
        for idx in range(int(array.shape[axis])):
            matrix = _select_stack_slice(array, axis, idx)
            residual_stats = self._combine_residualize_matrix(matrix, distance_matrix)
            self._combine_assign_stack_slice(corrected, axis, idx, residual_stats.residual_matrix)
            stats["slopes"].append(float(residual_stats.slope))
            stats["intercepts"].append(float(residual_stats.intercept))
            stats["n_values"] += int(residual_stats.first_values.size)
            if progress_callback is not None:
                progress_callback()
        return corrected, stats

    def _combine_spatial_parcellation_img_and_data(self):
        template_img = getattr(self, "_active_parcellation_img", None)
        template_data = getattr(self, "_active_parcellation_data", None)
        template_path = getattr(self, "_active_parcellation_path", None)
        if template_img is not None and template_data is not None:
            return template_img, np.asarray(template_data, dtype=int), template_path

        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select parcellation NIfTI for spatial correction",
            str(self._default_parcellation_dir()),
            "NIfTI files (*.nii *.nii.gz);;All files (*)",
        )
        if not selected:
            raise RuntimeError("No parcellation NIfTI selected.")
        try:
            import nibabel as nib
        except Exception as exc:
            raise RuntimeError(f"nibabel not available: {exc}") from exc
        try:
            template_img = nib.load(str(selected))
            template_data = np.asarray(template_img.get_fdata(), dtype=int)
        except Exception as exc:
            raise RuntimeError(f"Failed to load parcellation NIfTI: {exc}") from exc
        if template_data.ndim != 3:
            raise RuntimeError("Parcellation NIfTI must be a 3D image.")

        self._active_parcellation_path = Path(selected)
        self._active_parcellation_img = template_img
        self._active_parcellation_data = template_data
        self._update_parcellation_label()
        return template_img, template_data, self._active_parcellation_path

    @staticmethod
    def _combine_spatial_correction_labels(parcel_labels, template_data, matrix_size):
        matrix_size = int(matrix_size)
        if parcel_labels is not None:
            labels = np.asarray(parcel_labels).reshape(-1)
            if labels.size == matrix_size:
                try:
                    labels = np.asarray([int(label) for label in labels.tolist()], dtype=int)
                except Exception as exc:
                    raise ValueError("Matrix parcel_labels_group must be integer-valued.") from exc
                if len(set(labels.tolist())) != labels.size:
                    raise ValueError("Matrix parcel_labels_group must be unique.")
                return labels

        image_labels = np.unique(np.asarray(template_data, dtype=int))
        image_labels = image_labels[image_labels != 0]
        image_labels = np.asarray(sorted(int(label) for label in image_labels.tolist()), dtype=int)
        if image_labels.size == matrix_size:
            return image_labels
        raise ValueError(
            "Spatial correction requires matrix parcel_labels_group matching the matrix size. "
            f"No valid labels were found, and the parcellation has {image_labels.size} non-zero labels "
            f"for a {matrix_size}x{matrix_size} matrix."
        )

    def _confirm_combine_result(self) -> None:
        dialog = getattr(self, "_combine_dialog", None)
        payload = self._combine_pending_result
        if not isinstance(payload, dict):
            message = "No processed combine result is waiting for confirmation."
            if dialog is not None:
                dialog.set_status(message)
                dialog.set_can_confirm(False)
            self.statusBar().showMessage(message)
            return

        _derived_id, entry = self._workspace.add_derived_entry(
            np.asarray(payload["matrix"], dtype=float),
            label=str(payload.get("label") or "combine_result"),
            source_path=payload.get("source_path"),
            selected_key=payload.get("selected_key"),
            sample_index=payload.get("sample_index"),
            extra_fields={
                "combine_operation": str(payload.get("operation") or ""),
                "combine_source_a": payload.get("first_entry_id"),
                "combine_source_b": payload.get("second_entry_id"),
                "spatial_correction_parcellation": payload.get("spatial_correction_parcellation"),
                "spatial_correction_keys": payload.get("spatial_correction_keys"),
                "npz_payload": payload.get("npz_payload"),
                "stack_axis": payload.get("stack_axis"),
                "stack_len": payload.get("stack_len"),
                "parcel_labels_group": (
                    np.asarray(payload["parcel_labels_group"])
                    if payload.get("parcel_labels_group") is not None
                    else None
                ),
                "parcel_names_group": (
                    np.asarray(payload["parcel_names_group"], dtype=object)
                    if payload.get("parcel_names_group") is not None
                    else None
                ),
            },
        )
        self._add_workspace_list_item(entry, tooltip=entry["label"], select=True)
        self._sync_combine_dialog_state()
        self._combine_pending_result = None

        if dialog is not None:
            dialog.set_can_confirm(False)
            dialog.set_status("Added the previewed result to the workspace.")
        self.statusBar().showMessage("Added confirmed combine result to the workspace.")

    def _combine_apply_matrix_operation(self, first_matrix, second_matrix, operation):
        return _combine_apply_op(first_matrix, second_matrix, operation)

    def _combine_vectors_for_correlation(self, first_matrix, second_matrix):
        return _combine_corr_vectors(first_matrix, second_matrix)

    def _combine_residualize_matrix(self, first_matrix, second_matrix):
        return _combine_residualize_matrix(first_matrix, second_matrix)

    def _combine_spatially_residualize_matrix(self, matrix, parcellation_data, parcel_labels, affine=None):
        return _combine_spatially_residualize_matrix(
            matrix,
            parcellation_data,
            parcel_labels,
            affine=affine,
        )

    def _show_combine_correlation(
        self,
        first_matrix,
        second_matrix,
        *,
        first_label: str,
        second_label: str,
        alignment_summary: str = "",
    ) -> None:
        dialog = getattr(self, "_combine_dialog", None)
        first_values, second_values, mode_text = self._combine_vectors_for_correlation(first_matrix, second_matrix)
        stats = _combine_corr_stats(first_values, second_values, mode_text)

        title = f"Correlation: {first_label} vs {second_label}"
        summary_parts = []
        if alignment_summary:
            summary_parts.append(str(alignment_summary))
        summary_parts.append(f"Pearson correlation on {stats.mode_text}")
        summary_parts.append(f"n={stats.first_values.size}")
        summary_parts.append(f"r={float(stats.r_value):.4f}")
        summary_parts.append(f"p={self._format_p_value(stats.p_value)}")
        summary = " | ".join(summary_parts)

        if dialog is not None:
            dialog.show_correlation_result(
                stats.first_values,
                stats.second_values,
                title=title,
                summary_text=summary,
                r_value=stats.r_value,
                p_value=stats.p_value,
                slope=stats.slope,
                intercept=stats.intercept,
            )
            dialog.set_status("Processed correlation and displayed the regression plot.")
        self.statusBar().showMessage(
            f"Correlation complete: r={float(stats.r_value):.4f}, p={self._format_p_value(stats.p_value)}."
        )

    def _get_valid_keys_cached(self, path: Path):
        return list(self._data_access.get_valid_keys(path))

    def _invalidate_path_caches(self, path: Path) -> None:
        self._data_access.invalidate_path(path)

    def _load_parcel_metadata_cached(self, path: Path, label_key: str = "", name_key: str = ""):
        return self._data_access.load_parcel_metadata_cached(
            path,
            label_key=label_key,
            name_key=name_key,
        )

    def _load_group_value_cached(self, path: Path, index: int):
        return self._data_access.load_group_value_cached(path, index)

    def _covars_info_cached(self, path: Path):
        return self._data_access.covars_info(path)

    def _apply_key_options_state(self, state) -> None:
        self._workspace_matrix_controller.apply_key_options_state(state)

    def _refresh_key_options(self, entry) -> None:
        self._workspace_matrix_controller.refresh_key_options(entry)

    def _refresh_covars_options(self, source_path: Path, entry) -> None:
        self._workspace_matrix_controller.refresh_covars_options(source_path, entry)

    def _default_title_for_entry(self, entry) -> str:
        return self._workspace_matrix_controller.default_title_for_entry(entry)

    def _apply_title_for_entry(self, entry, force: bool = False) -> str:
        return self._workspace_matrix_controller.apply_title_for_entry(entry, force=force)

    def _on_title_edited(self) -> None:
        self._workspace_matrix_controller.on_title_edited()

    def _on_sample_changed(self, value: int) -> None:
        self._workspace_matrix_controller.on_sample_changed(value)

    def _update_sample_controls(self, entry, axis, stack_len) -> None:
        self._workspace_matrix_controller.update_sample_controls(entry, axis, stack_len)

    def _reload_colormaps(self) -> None:
        names = self._available_colormap_names()
        current = self.cmap_combo.currentText() if hasattr(self, "cmap_combo") else ""
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
        current_3d = str(self._gradient_colormap_name or "").strip()
        if current_3d not in names:
            if self._default_gradient_colormap in names:
                current_3d = self._default_gradient_colormap
            elif "spectrum_fsl" in names:
                current_3d = "spectrum_fsl"
            elif names:
                current_3d = names[0]
            else:
                current_3d = "spectrum_fsl"
            self._gradient_colormap_name = current_3d
        self._sync_gradients_dialog_state()

    def _load_display_controls_for_entry(self, entry) -> None:
        self._workspace_matrix_controller.load_display_controls_for_entry(entry)

    def _store_display_controls_for_entry(self, entry) -> None:
        return self._workspace_matrix_controller.store_display_controls_for_entry(entry)

    def _on_colormap_changed(self, *_args) -> None:
        self._workspace_matrix_controller.on_colormap_changed(*_args)

    def _selected_colormap_name(self, entry=None) -> str:
        return self._workspace_matrix_controller.selected_colormap_name(entry)

    def _selected_colormap(self, entry=None):
        return self._workspace_matrix_controller.selected_colormap(entry)

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

    def _current_display_limits(self, entry=None):
        return self._workspace_matrix_controller.current_display_limits(entry)

    def _current_display_scale(self, entry=None) -> str:
        return self._workspace_matrix_controller.current_display_scale(entry)

    def _log_scale_error(self, matrix, vmin, vmax):
        return self._workspace_matrix_controller.log_scale_error(matrix, vmin, vmax)

    def _on_display_scaling_changed(self, *_args) -> None:
        self._workspace_matrix_controller.on_display_scaling_changed(*_args)

    def _on_zoom_region_toggled(self, *_args) -> None:
        if not hasattr(self, "zoom_region_check"):
            return
        if self.zoom_region_check.isChecked():
            self._enable_zoom_selector()
            self.statusBar().showMessage("Drag on the matrix to zoom into a region.")
        else:
            self._reset_zoom_selector()

    def _update_zoom_selector(self) -> None:
        self._reset_zoom_selector()
        if hasattr(self, "zoom_region_check") and self.zoom_region_check.isChecked():
            self._enable_zoom_selector()

    def _enable_zoom_selector(self) -> None:
        if RectangleSelector is None:
            self.statusBar().showMessage("Region zoom is unavailable (matplotlib widgets missing).")
            if hasattr(self, "zoom_region_check"):
                self.zoom_region_check.blockSignals(True)
                self.zoom_region_check.setChecked(False)
                self.zoom_region_check.blockSignals(False)
            return
        if self._current_axes is None:
            return
        if self._zoom_selector is not None:
            try:
                self._zoom_selector.set_active(True)
            except Exception:
                pass
            return
        self._zoom_selector = RectangleSelector(
            self._current_axes,
            self._on_zoom_region_selected,
            useblit=False,
            button=[1],
            minspanx=1,
            minspany=1,
        )
        try:
            self._zoom_selector.set_active(True)
        except Exception:
            pass

    def _reset_zoom_selector(self) -> None:
        if self._zoom_selector is None:
            return
        try:
            self._zoom_selector.set_active(False)
        except Exception:
            pass
        self._zoom_selector = None

    def _on_zoom_region_selected(self, eclick, erelease) -> None:
        if self._current_axes is None or self._current_matrix is None:
            return
        if (
            eclick.xdata is None
            or eclick.ydata is None
            or erelease.xdata is None
            or erelease.ydata is None
        ):
            return
        x0, x1 = eclick.xdata, erelease.xdata
        y0, y1 = eclick.ydata, erelease.ydata
        if abs(x1 - x0) < 1 or abs(y1 - y0) < 1:
            return
        nrows, ncols = self._current_matrix.shape[:2]
        x0 = max(-0.5, min(ncols - 0.5, x0))
        x1 = max(-0.5, min(ncols - 0.5, x1))
        y0 = max(-0.5, min(nrows - 0.5, y0))
        y1 = max(-0.5, min(nrows - 0.5, y1))
        xmin, xmax = sorted([x0, x1])
        ymin, ymax = sorted([y0, y1])
        ax = self._current_axes
        if ax.xaxis_inverted():
            ax.set_xlim(xmax, xmin)
        else:
            ax.set_xlim(xmin, xmax)
        if ax.yaxis_inverted():
            ax.set_ylim(ymax, ymin)
        else:
            ax.set_ylim(ymin, ymax)
        if hasattr(self, "zoom_reset_button"):
            self.zoom_reset_button.setEnabled(True)
        self.canvas.draw_idle()

    def _reset_matrix_zoom(self) -> None:
        if self._current_axes is None:
            return
        if self._matrix_full_xlim is not None and self._matrix_full_ylim is not None:
            self._current_axes.set_xlim(self._matrix_full_xlim)
            self._current_axes.set_ylim(self._matrix_full_ylim)
        else:
            self._current_axes.autoscale()
        if hasattr(self, "zoom_reset_button"):
            self.zoom_reset_button.setEnabled(False)
        self.canvas.draw_idle()

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
        dialog = getattr(self, "_gradients_dialog", None)
        if dialog is not None:
            try:
                name = dialog.selected_colormap()
            except Exception:
                name = ""
            if name:
                self._gradient_colormap_name = name
                return name
        name = str(self._gradient_colormap_name or "").strip()
        if name:
            return name
        return "spectrum_fsl"

    def _selected_surface_colormap(self, name: str = None):
        if name is None:
            name = self._selected_surface_colormap_name()
        else:
            name = str(name or "").strip() or self._selected_surface_colormap_name()
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
        self._sync_combine_dialog_state()

    def _open_histogram(self) -> None:
        if self.file_list.count() == 0:
            self.statusBar().showMessage("No matrices to plot.")
            return
        dialog = HistogramDialog(self._entries, list(self._entry_ids()), self.titles, parent=self)
        dialog.show()
        self._hist_dialog = dialog

    def _ensure_entry_key(self, entry):
        return self._data_access.ensure_entry_key(entry)

    def _matrix_for_entry(self, entry):
        return self._data_access.matrix_for_entry(entry)

    @staticmethod
    def _safe_name_fragment(text: str) -> str:
        return _safe_name_fragment_helper(text)

    def _write_selected_matrix_to_file(self) -> None:
        self._workspace_matrix_controller.write_selected_matrix_to_file(
            file_dialog_class=QFileDialog,
        )

    def _export_grid(self) -> None:
        self._workspace_matrix_controller.export_grid(
            export_grid_dialog_class=ExportGridDialog,
            dialog_accepted_code=_dialog_accepted_code,
            figure_class=Figure,
            sim_matrix_plot=SimMatrixPlot,
            remove_axes_border=_remove_axes_border,
            apply_rotation=_apply_rotation,
        )

    def _on_selection_changed(self, current, _previous) -> None:
        self._workspace_matrix_controller.on_selection_changed(current, _previous)

    def _plot_selected(self, *_args) -> None:
        entry_id = self._current_entry_id()
        if entry_id is None:
            self._clear_plot()
            return
        entry = self._entries.get(entry_id)
        if entry is None:
            self._clear_plot()
            return
        self._store_display_controls_for_entry(entry)

        try:
            plot_resolution = _resolve_entry_plot(
                entry,
                ensure_entry_key=self._ensure_entry_key,
                load_matrix_from_npz=_load_matrix_from_npz,
                stack_axis=_stack_axis,
                average_to_square=_average_to_square,
                select_stack_slice=_select_stack_slice,
            )
        except KeyError:
            self._clear_plot()
            self.statusBar().showMessage("No valid matrix key selected.")
            return
        except Exception as exc:
            self._clear_plot()
            self.statusBar().showMessage(f"Failed to load {entry.get('label', 'file')}: {exc}")
            return

        matrix = plot_resolution.matrix
        key = plot_resolution.key
        source_path = plot_resolution.source_path
        self._update_sample_controls(entry, plot_resolution.stack_axis, plot_resolution.stack_len)

        current_title = self._apply_title_for_entry(entry)

        labels, names = self._entry_parcel_metadata(entry, expected_len=matrix.shape[0])
        labels_list, names_list = _normalize_plot_matrix_labels(
            matrix,
            labels,
            names,
            to_string_list=_to_string_list,
        )

        self._current_matrix = matrix
        self._current_parcel_labels = labels_list
        self._current_parcel_names = names_list

        vmin, vmax, scaling_error = self._current_display_limits(entry)
        if scaling_error:
            self.statusBar().showMessage(scaling_error)
            return

        colormap = self._selected_colormap(entry)
        zscale = self._current_display_scale(entry)
        if zscale == "log":
            log_error = self._log_scale_error(matrix, vmin, vmax)
            if log_error:
                self.statusBar().showMessage(log_error)
                return
        self.figure.clear()
        self._reset_zoom_selector()
        ax = self.figure.add_subplot(111)
        SimMatrixPlot.plot_simmatrix(
            matrix,
            ax=ax,
            titles=current_title,
            colormap=colormap,
            vmin=vmin,
            vmax=vmax,
            zscale=zscale,
        )
        self._hover_vline = ax.axvline(
            0.0,
            color="#22c55e",
            linewidth=0.8,
            alpha=0.85,
            visible=False,
            zorder=20,
        )
        self._hover_hline = ax.axhline(
            0.0,
            color="#22c55e",
            linewidth=0.8,
            alpha=0.85,
            visible=False,
            zorder=20,
        )
        _remove_axes_border(ax)
        self._current_axes = ax
        self._matrix_full_xlim = ax.get_xlim()
        self._matrix_full_ylim = ax.get_ylim()
        if hasattr(self, "zoom_reset_button"):
            self.zoom_reset_button.setEnabled(False)
        self._update_zoom_selector()
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
        info = self._covars_info_cached(source_path)
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

        label = f"avg {covar_name}={value_text} ({Path(source_path).name})"
        covar_value = value_float if value_float is not None else value_text
        _derived_id, entry = self._workspace.add_derived_entry(
            averaged,
            label=label,
            source_path=source_path,
            selected_key=key,
            sample_index=None,
            extra_fields={
                "covar_name": covar_name,
                "covar_value": covar_value,
            },
        )
        self._add_workspace_list_item(entry, select=True)
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

        _derived_id, entry = self._workspace.add_derived_entry(
            matrix,
            label=label,
            source_path=source_path,
            selected_key=key,
            sample_index=sample_index,
        )
        self._add_workspace_list_item(entry, select=True)
        self.statusBar().showMessage("Added sample matrix to list.")



    @staticmethod


    @staticmethod






    @staticmethod


    @staticmethod




    @staticmethod

    @staticmethod



    def _clear_plot(self) -> None:
        self.figure.clear()
        self._current_matrix = None
        self._current_parcel_labels = None
        self._current_parcel_names = None
        self._current_axes = None
        self._reset_zoom_selector()
        self._matrix_full_xlim = None
        self._matrix_full_ylim = None
        self._hover_vline = None
        self._hover_hline = None
        if hasattr(self, "zoom_reset_button"):
            self.zoom_reset_button.setEnabled(False)
        self._set_plot_title("")
        self._reset_gradients_output()
        self._update_nbs_prepare_button()
        self._update_write_to_file_button()
        self._update_view_labels_button()
        self.hover_label.setText("")
        self.canvas.draw_idle()

    def _hide_hover_crosshair(self) -> bool:
        changed = False
        for line in (self._hover_vline, self._hover_hline):
            if line is not None and line.get_visible():
                line.set_visible(False)
                changed = True
        return changed

    def _update_hover_crosshair(self, x_value: float, y_value: float) -> bool:
        changed = False
        if self._hover_vline is not None:
            self._hover_vline.set_xdata([x_value, x_value])
            if not self._hover_vline.get_visible():
                self._hover_vline.set_visible(True)
            changed = True
        if self._hover_hline is not None:
            self._hover_hline.set_ydata([y_value, y_value])
            if not self._hover_hline.get_visible():
                self._hover_hline.set_visible(True)
            changed = True
        return changed

    def _on_hover(self, event) -> None:
        if (
            self._current_matrix is None
            or self._current_axes is None
            or event.inaxes != self._current_axes
            or event.xdata is None
            or event.ydata is None
        ):
            if self._hide_hover_crosshair():
                self.canvas.draw_idle()
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
            if self._hide_hover_crosshair():
                self.canvas.draw_idle()
            self.hover_label.setText("")
            return
        self._update_hover_crosshair(event.xdata, event.ydata)
        self.canvas.draw_idle()
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
                if not url.isLocalFile():
                    continue
                local_path = Path(url.toLocalFile())
                if local_path.is_dir() or local_path.suffix.lower() == ".npz":
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event) -> None:
        if not event.mimeData().hasUrls():
            event.ignore()
            return
        file_paths = []
        folder_paths = []
        for url in event.mimeData().urls():
            if not url.isLocalFile():
                continue
            local_path = Path(url.toLocalFile())
            if local_path.is_dir():
                folder_paths.append(local_path)
            else:
                file_paths.append(str(local_path))

        if file_paths:
            self._add_files(file_paths)
        for folder_path in folder_paths:
            self._open_batch_import_dialog(folder_path)
        if not file_paths and not folder_paths:
            event.ignore()
            return
        event.acceptProposedAction()


def main() -> int:
    diag_path = _install_runtime_diagnostics()
    app = QApplication(sys.argv)
    _write_diagnostic_line("QApplication created")
    try:
        app.aboutToQuit.connect(lambda: _write_diagnostic_line("QApplication.aboutToQuit emitted"))
    except Exception:
        pass
    if sys.platform.startswith("linux"):
        app.setApplicationName("donald")
        if hasattr(app, "setDesktopFileName"):
            app.setDesktopFileName("donald")
    elif sys.platform == "darwin":
        app.setApplicationName("Donald")
        if hasattr(app, "setApplicationDisplayName"):
            app.setApplicationDisplayName("Donald")
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
    _write_diagnostic_line(f"DONALD constructed: {window!r}")
    try:
        window.destroyed.connect(lambda *_args: _write_diagnostic_line("DONALD destroyed"))
    except Exception:
        pass
    if not str(window._results_dir_default or "").strip() and splash is not None:
        splash.close()
        splash = None
    if not window.ensure_initial_configuration():
        _write_diagnostic_line("Initial configuration canceled")
        if splash is not None:
            splash.close()
        window.close()
        return 0
    if splash is not None:
        splash.showMessage("Starting Donald...", splash_align, QColor("white"))
        app.processEvents()
    window.showMaximized()
    _write_diagnostic_line(f"Viewer shown; diagnostics logging to {diag_path}")
    if splash is not None:
        splash.finish(window)
    exit_code = app.exec()
    _write_diagnostic_line(f"QApplication exited with code {exit_code}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
