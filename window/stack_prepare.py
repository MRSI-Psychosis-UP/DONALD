#!/usr/bin/env python3
"""Prepare and stack selected connectivity NPZ files into a population stack."""

from pathlib import Path
import contextlib
import importlib.resources as importlib_resources
import importlib.util
import os
import re
import sys
import traceback

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from PyQt6.QtCore import QObject, QThread, QTimer, Qt, pyqtSignal
    from PyQt6.QtWidgets import (
        QAbstractItemView,
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QSizePolicy,
        QSpinBox,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
    QT_LIB = 6
except Exception:
    from PyQt5.QtCore import QObject, QThread, QTimer, Qt, pyqtSignal
    from PyQt5.QtWidgets import (
        QAbstractItemView,
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QSizePolicy,
        QSpinBox,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
    QT_LIB = 5


try:
    from window.shared.theme import (
        switch_checkbox_theme_styles as _switch_checkbox_theme_styles,
        workflow_dialog_stylesheet as _workflow_dialog_stylesheet,
    )
except Exception:
    from mrsi_viewer.window.shared.theme import (
        switch_checkbox_theme_styles as _switch_checkbox_theme_styles,
        workflow_dialog_stylesheet as _workflow_dialog_stylesheet,
    )


if QT_LIB == 6:
    Qt.Checked = Qt.CheckState.Checked
    Qt.Unchecked = Qt.CheckState.Unchecked


NUMERIC_RANGE_RE = re.compile(
    r"^\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*-\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*$"
)
ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def _stack_diagnostic_log(text) -> None:
    message = f"[STACK] {text}"
    try:
        sys.__stderr__.write(message + "\n")
        sys.__stderr__.flush()
    except Exception:
        pass
    log_path_raw = str(os.getenv("MRSI_VIEWER_DIAGNOSTICS_LOG", "")).strip()
    if not log_path_raw:
        return
    try:
        with open(log_path_raw, "a", encoding="utf-8") as handle:
            handle.write(message + "\n")
    except Exception:
        pass


def _ensure_pandas():
    if pd is None:
        raise RuntimeError("pandas is required for covariate TSV loading.")


def _is_enabled_flag():
    return getattr(Qt, "ItemIsEnabled", getattr(Qt.ItemFlag, "ItemIsEnabled"))


def _is_selectable_flag():
    return getattr(Qt, "ItemIsSelectable", getattr(Qt.ItemFlag, "ItemIsSelectable"))


def _is_user_checkable_flag():
    return getattr(Qt, "ItemIsUserCheckable", getattr(Qt.ItemFlag, "ItemIsUserCheckable"))


def _is_editable_flag():
    return getattr(Qt, "ItemIsEditable", getattr(Qt.ItemFlag, "ItemIsEditable"))


def _copy_drop_action():
    return getattr(Qt, "CopyAction", getattr(Qt.DropAction, "CopyAction"))


def _size_policy_expanding():
    policy = getattr(QSizePolicy, "Expanding", None)
    if policy is not None:
        return policy
    return QSizePolicy.Policy.Expanding


def _load_construct_matrix_pop_main():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "construct_matrix_pop.py"
    spec = importlib.util.spec_from_file_location("construct_matrix_pop_gui", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load construct_matrix_pop module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "main"):
        raise ImportError("construct_matrix_pop.py does not define main")
    return module.main


class _GuiLogStream:
    def __init__(self, callback):
        self._callback = callback
        self._buffer = ""

    def write(self, text):
        if text is None:
            return 0
        chunk = str(text)
        if chunk == "":
            return 0
        self._buffer += chunk
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._callback(line)
        return len(chunk)

    def flush(self):
        if self._buffer:
            self._callback(self._buffer)
            self._buffer = ""


def _decode_scalar(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode()
        except Exception:
            return str(value)
    return value


def _display_text(value):
    value = _decode_scalar(value)
    if value is None:
        return ""
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


def _switch_checkbox_style(track_off: str, track_on: str, border_color: str) -> str:
    return (
        "QCheckBox#switchCheckBox { spacing: 10px; } "
        "QCheckBox#switchCheckBox::indicator { width: 42px; height: 22px; border-radius: 11px; } "
        f"QCheckBox#switchCheckBox::indicator:unchecked {{ background-color: {track_off}; border: 1px solid {border_color}; }} "
        f"QCheckBox#switchCheckBox::indicator:checked {{ background-color: {track_on}; border: 1px solid {track_on}; }}"
    )


def _max_parcel_label_from_npz(npz_path) -> int:
    try:
        with np.load(str(npz_path), allow_pickle=True) as archive:
            labels = None
            for key in ("labels_indices", "parcel_labels"):
                if key in archive:
                    labels = np.asarray(archive[key]).reshape(-1)
                    break
            if labels is not None and labels.size > 0:
                values = []
                for value in labels:
                    text = _display_text(value).strip()
                    if text == "":
                        continue
                    try:
                        values.append(int(float(text)))
                    except Exception:
                        continue
                if values:
                    return max(0, max(values))
            for key in ("connectivity", "connectome_density", "simmatrix_sp"):
                if key not in archive:
                    continue
                matrix = np.asarray(archive[key])
                if matrix.ndim >= 2 and matrix.shape[-1] > 0:
                    return max(0, int(matrix.shape[-1] - 1))
    except Exception as exc:
        _stack_diagnostic_log(f"Failed to inspect parcel labels in {npz_path}: {exc}")
    return 0


def _covars_to_rows(covars_df):
    if covars_df is None:
        return [], []
    columns = [str(col) for col in covars_df.columns]
    records = covars_df.to_dict(orient="records")
    rows = []
    for record in records:
        rows.append({col: _decode_scalar(record.get(col)) for col in columns})
    return columns, rows


def _slugify(text):
    token = str(text or "").strip()
    if not token:
        return "stack"
    chars = []
    for ch in token:
        if ch.isalnum() or ch in {"-", "_"}:
            chars.append(ch)
        else:
            chars.append("_")
    out = "".join(chars).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "stack"


def _infer_group_from_path(path: Path) -> str:
    parts = list(path.parts)
    if "derivatives" in parts:
        idx = parts.index("derivatives")
        if idx > 0:
            return parts[idx - 1]
    return "group"


def _connectivity_modalities_from_path(path: Path):
    found = []
    seen = set()
    texts = (Path(path).name.lower(), str(path).lower())
    for text in texts:
        for match in re.findall(r"connectivity[_-]([a-z0-9]+)", text):
            modality = str(match).strip().lower()
            if not modality or modality in seen:
                continue
            seen.add(modality)
            found.append(modality)
    if found:
        return found

    lowered = str(path).lower().replace("\\", "/")
    for modality in ("mrsi", "func", "dwi"):
        if f"/{modality}/" in lowered and modality not in seen:
            found.append(modality)
            seen.add(modality)
    return found


def _infer_modality_from_path(path: Path) -> str:
    modalities = _connectivity_modalities_from_path(path)
    if modalities:
        return modalities[0]
    return "connectivity"


def _extract_metadata(filename):
    base_filename = os.path.basename(str(filename))
    results = {}
    patterns = {
        "sub": r"sub-([^_]+)",
        "ses": r"ses-([^_]+)",
        "run": r"run-([^_]+)",
        "acq": r"acq-([^_]+)",
        "space": r"space-([^_]+)",
        "parcscheme": r"atlas-(?:chimera)?([^_]+)",
        "scale": r"scale(\d+)",
        "grow": r"grow(\d+)mm",
        "npert": r"npert[-_](\d+)",
        "filt": r"filt([^_]+)",
        "met": r"met-([^_]+)",
        "res": r"res-([^_]+)",
    }
    for key, pat in patterns.items():
        match = re.search(pat, base_filename)
        results[key] = match.group(1) if match else None
    for key in ("scale", "grow", "npert"):
        if results[key] is not None:
            results[key] = int(results[key])
    return results


def _group_root_from_path(path: Path) -> Path:
    current = Path(path).resolve()
    for parent in [current.parent] + list(current.parents):
        if parent.name == "derivatives":
            return parent.parent
    return current.parent


def _atlas_tag_from_path(path: Path) -> str:
    meta = _extract_metadata(str(path))
    base_name = Path(path).name
    atlas_match = re.search(r"atlas-([^_]+)", base_name)
    atlas = atlas_match.group(1).strip() if atlas_match else ""
    scale = meta.get("scale")
    if atlas and scale is not None and f"scale{scale}" not in atlas:
        return f"{atlas}_scale{scale}"
    if atlas:
        return atlas
    return "unknown"


def _modalities_from_paths(paths):
    found = []
    seen = set()
    for path in paths or []:
        for modality in _connectivity_modalities_from_path(Path(path)):
            if not modality or modality in seen:
                continue
            seen.add(modality)
            found.append(modality)
    if not found:
        return []
    preferred = {"mrsi": 0, "func": 1, "dwi": 2}
    return sorted(found, key=lambda item: (preferred.get(item, 99), item))


def _default_parcellation_dir() -> Path:
    viewer_atlas = _REPO_ROOT / "mrsi_viewer" / "data" / "atlas"
    if viewer_atlas.exists():
        return viewer_atlas
    try:
        toolbox_atlas = importlib_resources.files("mrsitoolbox").joinpath("data", "atlas")
        if toolbox_atlas.is_dir():
            return Path(str(toolbox_atlas))
    except Exception:
        pass
    return Path.home()


def _output_path_for_modality(base_output_path, modality: str) -> str:
    base_path = Path(str(base_output_path)).expanduser()
    modality = str(modality or "").strip().lower() or "connectivity"
    parent = base_path.parent
    if parent.name.lower() in {"mrsi", "func", "dwi", "multimodal"} and parent.parent.name.lower() == "connectivity":
        parent = parent.parent / "multimodal"
    elif parent.name.lower() == "connectivity":
        parent = parent / "multimodal"

    name = re.sub(
        r"connectivity[_-][a-z0-9]+",
        f"connectivity_{modality}",
        base_path.name,
        flags=re.IGNORECASE,
    )
    if name == base_path.name:
        name = base_path.name.replace("connectivity", f"connectivity_{modality}")
    return str(parent / name)


def _load_retained_labels_from_output(output_path):
    path = Path(str(output_path))
    if not path.is_file():
        raise FileNotFoundError(f"Expected stack output was not created: {path}")
    with np.load(path, allow_pickle=True) as npz_file:
        for key in ("parcel_labels_group", "parcel_labels"):
            if key in npz_file:
                return np.asarray(npz_file[key]).reshape(-1).tolist()
    raise KeyError(f"No parcel label vector was found in {path}")


class TsvDropLabel(QLabel):
    file_dropped = pyqtSignal(str)

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setAcceptDrops(True)

    @staticmethod
    def _tsv_path_from_event(event):
        mime = event.mimeData()
        if mime is None or not mime.hasUrls():
            return None
        for url in mime.urls():
            if not url.isLocalFile():
                continue
            candidate = url.toLocalFile()
            if str(candidate).lower().endswith(".tsv"):
                return candidate
        return None

    def dragEnterEvent(self, event):
        if self._tsv_path_from_event(event):
            try:
                event.setDropAction(_copy_drop_action())
            except Exception:
                pass
            event.accept()
            return
        event.ignore()

    def dropEvent(self, event):
        selected = self._tsv_path_from_event(event)
        if selected:
            queued_path = str(Path(selected))
            try:
                event.setDropAction(_copy_drop_action())
            except Exception:
                pass
            event.accept()
            # Defer heavy TSV loading until the DnD event has fully unwound.
            QTimer.singleShot(0, lambda path=queued_path: self.file_dropped.emit(path))
            return
        event.ignore()


class FileDropLineEdit(QLineEdit):
    file_dropped = pyqtSignal(str)

    def __init__(self, accepted_suffixes=None, parent=None):
        super().__init__(parent)
        self._accepted_suffixes = tuple(str(item).lower() for item in (accepted_suffixes or ()))
        self.setAcceptDrops(True)

    def _path_from_event(self, event):
        mime = event.mimeData()
        if mime is None or not mime.hasUrls():
            return None
        for url in mime.urls():
            if not url.isLocalFile():
                continue
            candidate = url.toLocalFile()
            lowered = str(candidate).lower()
            if any(lowered.endswith(suffix) for suffix in self._accepted_suffixes):
                return candidate
        return None

    def dragEnterEvent(self, event):
        selected = self._path_from_event(event)
        if selected:
            event.acceptProposedAction()
            return
        event.ignore()

    def dropEvent(self, event):
        selected = self._path_from_event(event)
        if selected:
            self.setText(str(selected))
            self.file_dropped.emit(str(selected))
            event.acceptProposedAction()
            return
        event.ignore()


class CovarsSelectionWidget(QWidget):
    configuration_changed = pyqtSignal()

    def __init__(self, selected_paths=None, parent=None):
        super().__init__(parent)
        _ensure_pandas()
        self._selected_paths = [Path(path) for path in (selected_paths or [])]
        self._covars_df = None
        self._covars_path = None
        self._covars_error = ""
        self._columns = []
        self._rows = []
        self._filtered_indices = []
        self._excluded_indices = set()
        self._table_refreshing = False
        # Keep DnD handling on the dedicated input widget only.
        self.setAcceptDrops(False)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)

        intro_label = QLabel("Drop a covariates TSV here, then filter and exclude rows before processing.")
        intro_label.setWordWrap(True)
        root.addWidget(intro_label)

        top = QHBoxLayout()
        self.tsv_drop_label = FileDropLineEdit(accepted_suffixes=(".tsv",))
        self.tsv_drop_label.setPlaceholderText("Drop a .tsv file here or use Browse.")
        self.tsv_drop_label.setStyleSheet("padding: 8px; border: 1px dashed #6b7280;")
        self.tsv_drop_label.file_dropped.connect(lambda path: self._load_covars_file(Path(path)))
        top.addWidget(self.tsv_drop_label, 1)
        self.browse_covars_button = QPushButton("Browse TSV")
        self.browse_covars_button.clicked.connect(self._browse_covars_file)
        top.addWidget(self.browse_covars_button)
        self.clear_covars_button = QPushButton("Clear TSV")
        self.clear_covars_button.clicked.connect(self._clear_covars_file)
        top.addWidget(self.clear_covars_button)
        root.addLayout(top)

        self.covars_path_label = QLabel("No covars file loaded.")
        self.covars_path_label.setWordWrap(True)
        root.addWidget(self.covars_path_label)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Covariate"))
        self.filter_covar_combo = QComboBox()
        filter_row.addWidget(self.filter_covar_combo)
        filter_row.addWidget(QLabel("Value"))
        self.filter_value_edit = QLineEdit("")
        self.filter_value_edit.setPlaceholderText("e.g. control, 0,1 or 32-46")
        self.filter_value_edit.returnPressed.connect(self._apply_filter)
        filter_row.addWidget(self.filter_value_edit, 1)
        self.filter_button = QPushButton("Filter")
        self.filter_button.clicked.connect(self._apply_filter)
        filter_row.addWidget(self.filter_button)
        self.filter_reset_button = QPushButton("Reset")
        self.filter_reset_button.clicked.connect(self._reset_filter)
        filter_row.addWidget(self.filter_reset_button)
        root.addLayout(filter_row)

        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(False)
        self.table.setMinimumHeight(420)
        expanding = _size_policy_expanding()
        self.table.setSizePolicy(expanding, expanding)
        self.table.itemChanged.connect(self._on_table_item_changed)
        if QT_LIB == 6:
            self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        else:
            self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        root.addWidget(self.table, 1)

        self.showing_rows_label = QLabel("No covars filter applied.")
        root.addWidget(self.showing_rows_label)

        self.status_label = QLabel("Ready.")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

    def set_selected_paths(self, selected_paths):
        self._selected_paths = [Path(path) for path in (selected_paths or [])]

    def covars_columns(self):
        return list(self._columns)

    def covars_error(self):
        return str(self._covars_error or "")

    def has_covars(self):
        return self._covars_df is not None

    def default_batch_col(self) -> str:
        preferred = {"scanner", "site", "batch"}
        for name in self._columns:
            if str(name).strip().lower() in preferred:
                return str(name)
        return str(self._columns[0]) if self._columns else ""

    def covariate_type(self, covar_name: str) -> str:
        values = [row.get(covar_name) for row in self._rows]
        return "continuous" if _column_is_numeric(values) else "categorical"

    def is_id_like(self, covar_name: str) -> bool:
        return str(covar_name).strip().lower() in {
            "participant_id",
            "subject_id",
            "session_id",
            "id",
            "sub",
            "ses",
        }

    def current_selection_suffix(self):
        covar = self.filter_covar_combo.currentText().strip()
        values = self._parse_filter_values(self.filter_value_edit.text())
        if not covar or not values:
            return "all", "all"
        return _slugify(covar), _slugify("-".join(values))

    def selected_covars_df(self):
        if self._covars_df is None:
            return None
        selected = self.selected_row_indices()
        if not selected:
            return self._covars_df.iloc[0:0].copy()
        return self._covars_df.iloc[selected].reset_index(drop=True)

    def _set_status(self, text):
        self.status_label.setText(str(text))

    def _browse_covars_file(self):
        start_dir = str(self._selected_paths[0].parent) if self._selected_paths else str(Path.home())
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select covariates TSV",
            start_dir,
            "TSV files (*.tsv);;All files (*)",
        )
        if selected:
            self._load_covars_file(Path(selected))

    def _clear_covars_file(self):
        self._covars_df = None
        self._covars_path = None
        self._covars_error = ""
        self._columns = []
        self._rows = []
        self._filtered_indices = []
        self._excluded_indices = set()
        self.covars_path_label.setText("No covars file loaded.")
        self.tsv_drop_label.clear()
        self.tsv_drop_label.setPlaceholderText("Drop a .tsv file here or use Browse.")
        self.filter_covar_combo.clear()
        self.table.clear()
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        self.showing_rows_label.setText("No covars filter applied.")
        self._set_status("Covars file cleared. All selected NPZ files will be used.")
        self.configuration_changed.emit()

    def _load_covars_file(self, path: Path):
        _stack_diagnostic_log(f"CovarsSelectionWidget loading TSV: {path}")
        _ensure_pandas()
        if path.suffix.lower() != ".tsv":
            QMessageBox.warning(self, "Invalid Covars File", "Select a .tsv covariates file.")
            return
        df = pd.read_csv(path, sep="\t")
        self._covars_df = df
        self._covars_path = Path(path)
        self._columns, self._rows = _covars_to_rows(df)
        self._filtered_indices = list(range(len(self._rows)))
        self._excluded_indices = set()
        self._covars_error = ""
        col_map = {str(col).lower(): col for col in df.columns}
        sub_col = (
            col_map.get("participant_id")
            or col_map.get("subject_id")
            or col_map.get("subject")
            or col_map.get("sub")
            or col_map.get("id")
        )
        ses_col = col_map.get("session_id") or col_map.get("session") or col_map.get("ses")
        if not sub_col or not ses_col:
            self._covars_error = (
                "Covars file is missing participant_id/subject_id and/or session_id/session columns."
            )
        self.filter_covar_combo.clear()
        self.filter_covar_combo.addItems(self._columns)
        self._refresh_table()
        self.tsv_drop_label.setText(str(path))
        self.covars_path_label.setText(f"Loaded covars: {path}")
        if self._covars_error:
            self._set_status(self._covars_error)
        else:
            self._set_status(
                f"Loaded covars TSV with {len(self._rows)} rows. Filter rows before processing if needed."
            )
        _stack_diagnostic_log(
            f"CovarsSelectionWidget loaded TSV rows={len(self._rows)} cols={len(self._columns)} error={self._covars_error!r}"
        )
        _stack_diagnostic_log("CovarsSelectionWidget emitting configuration_changed")
        self.configuration_changed.emit()
        _stack_diagnostic_log("CovarsSelectionWidget emitted configuration_changed")

    def dragEnterEvent(self, event):
        event.ignore()

    def dropEvent(self, event):
        event.ignore()

    @staticmethod
    def _parse_filter_values(text):
        values = [token.strip() for token in str(text).split(",")]
        return [token for token in values if token != ""]

    @classmethod
    def _parse_numeric_filter_targets(cls, tokens):
        exact_targets = []
        range_targets = []
        for token in tokens:
            text = str(token).strip()
            if text == "":
                continue
            range_match = NUMERIC_RANGE_RE.match(text)
            if range_match:
                start = float(range_match.group(1))
                stop = float(range_match.group(2))
                low, high = (start, stop) if start <= stop else (stop, start)
                range_targets.append((low, high))
                continue
            exact_targets.append(float(text))
        return exact_targets, range_targets

    @staticmethod
    def _matches_numeric(value, exact_targets, range_targets):
        try:
            val = float(_display_text(value).strip())
        except Exception:
            return False
        if any(np.isclose(val, target) for target in exact_targets):
            return True
        for low, high in range_targets:
            if (val > low or np.isclose(val, low)) and (val < high or np.isclose(val, high)):
                return True
        return False

    @staticmethod
    def _matches_any(source_value, targets):
        text = _display_text(source_value).strip()
        if text == "":
            return False
        return text in targets

    def _apply_filter(self):
        if not self._columns:
            self._set_status("No covariates available to filter.")
            return
        covar_name = self.filter_covar_combo.currentText().strip()
        target_text = self.filter_value_edit.text().strip()
        if not covar_name:
            self._set_status("Select a covariate to filter.")
            return
        if target_text == "":
            self._set_status("Enter a covariate value to filter.")
            return
        target_values = self._parse_filter_values(target_text)
        if not target_values:
            self._set_status("Enter at least one filter value.")
            return

        values = [row.get(covar_name) for row in self._rows]
        numeric = _column_is_numeric(values)
        matched = []
        if numeric:
            try:
                exact_targets, range_targets = self._parse_numeric_filter_targets(target_values)
            except Exception:
                self._set_status("Numeric filters support values like 0,1 or ranges like 32-46.")
                return
            for row_idx, value in enumerate(values):
                if self._matches_numeric(value, exact_targets, range_targets):
                    matched.append(row_idx)
        else:
            for row_idx, value in enumerate(values):
                if self._matches_any(value, target_values):
                    matched.append(row_idx)
        self._filtered_indices = matched
        self._refresh_table()
        self._set_status(
            f"Filter applied: {covar_name} in [{', '.join(target_values)}]. "
            f"Showing {len(self._filtered_indices)}/{len(self._rows)} rows."
        )
        self.configuration_changed.emit()

    def _reset_filter(self):
        self._filtered_indices = list(range(len(self._rows)))
        self._refresh_table()
        self._set_status(f"Filter reset. Showing {len(self._filtered_indices)} rows.")
        self.configuration_changed.emit()

    def _refresh_table(self):
        self._table_refreshing = True
        self.table.blockSignals(True)
        self.table.setColumnCount(len(self._columns) + 1)
        self.table.setHorizontalHeaderLabels(["Exclude"] + list(self._columns))
        self.table.setRowCount(len(self._filtered_indices))
        enabled_flag = _is_enabled_flag()
        selectable_flag = _is_selectable_flag()
        checkable_flag = _is_user_checkable_flag()
        editable_flag = _is_editable_flag()
        for table_row, source_idx in enumerate(self._filtered_indices):
            row_data = self._rows[source_idx]
            include_item = QTableWidgetItem("")
            include_item.setFlags(enabled_flag | selectable_flag | checkable_flag)
            include_item.setCheckState(
                Qt.Checked if source_idx in self._excluded_indices else Qt.Unchecked
            )
            self.table.setItem(table_row, 0, include_item)
            for col_idx, covar_name in enumerate(self._columns, start=1):
                item = QTableWidgetItem(_display_text(row_data.get(covar_name)))
                item.setFlags(item.flags() & ~editable_flag)
                self.table.setItem(table_row, col_idx, item)
        if self.table.columnCount() > 0:
            header = self.table.horizontalHeader()
            if QT_LIB == 6:
                header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
                for col_idx in range(1, self.table.columnCount()):
                    header.setSectionResizeMode(col_idx, QHeaderView.ResizeMode.Stretch)
            else:
                header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
                for col_idx in range(1, self.table.columnCount()):
                    header.setSectionResizeMode(col_idx, QHeaderView.Stretch)
        self.table.blockSignals(False)
        self._table_refreshing = False
        self._update_showing_rows_label()

    def _on_table_item_changed(self, item):
        if item is None or self._table_refreshing or item.column() != 0:
            return
        row = item.row()
        if row < 0 or row >= len(self._filtered_indices):
            return
        source_idx = self._filtered_indices[row]
        if item.checkState() == Qt.Checked:
            self._excluded_indices.add(source_idx)
        else:
            self._excluded_indices.discard(source_idx)
        self._update_showing_rows_label()
        self.configuration_changed.emit()

    def _update_showing_rows_label(self):
        self.showing_rows_label.setText(
            f"Showing {len(self._filtered_indices)}/{len(self._rows)} rows | "
            f"Included: {len(self.selected_row_indices())}"
        )

    def selected_row_indices(self):
        return [idx for idx in self._filtered_indices if idx not in self._excluded_indices]

    def prepare_for_close(self):
        _stack_diagnostic_log(
            f"CovarsSelectionWidget prepare_for_close rows={len(self._rows)} cols={len(self._columns)}"
        )
        try:
            self.setAcceptDrops(False)
        except Exception:
            pass
        if hasattr(self, "tsv_drop_label") and self.tsv_drop_label is not None:
            try:
                self.tsv_drop_label.setAcceptDrops(False)
            except Exception:
                pass
        # Avoid mutating the populated QTableWidget during dialog teardown.
        # Qt will destroy child widgets/items when the parent dialog closes.


class _StackWorker(QObject):
    log_message = pyqtSignal(str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, process_kwargs):
        super().__init__()
        self._process_kwargs = dict(process_kwargs or {})

    def _run_construct(self, construct_main, process_kwargs):
        log_stream = _GuiLogStream(self.log_message.emit)
        with contextlib.redirect_stdout(log_stream), contextlib.redirect_stderr(log_stream):
            summary = construct_main(**process_kwargs)
        log_stream.flush()
        return summary

    def _run_single(self, construct_main, process_kwargs):
        summary = self._run_construct(construct_main, process_kwargs)
        return {
            "message": str(summary or "").strip(),
            "primary_index": 0,
            "runs": [
                {
                    "summary": summary,
                    "process_kwargs": dict(process_kwargs),
                }
            ],
        }

    def _run_multimodal(self, construct_main, process_kwargs, multimodal_config):
        ordered_modalities = [
            str(item).strip().lower()
            for item in multimodal_config.get("ordered_modalities", [])
            if str(item).strip()
        ]
        align_cross_modality = bool(multimodal_config.get("align_cross_modality"))
        effective_paths = [str(path) for path in process_kwargs.get("con_path_list", []) if str(path).strip()]
        if len(ordered_modalities) <= 1:
            return self._run_single(construct_main, process_kwargs)

        base_output_path = str(process_kwargs.get("matrix_outpath") or "").strip()
        retained_labels = []
        runs = []
        processed_modalities = []
        self.log_message.emit(
            "Multimodal stack enabled. Processing order: " + ", ".join(ordered_modalities)
        )

        total_modalities = len(ordered_modalities)
        for idx, modality in enumerate(ordered_modalities, start=1):
            modality_paths = [
                str(path)
                for path in effective_paths
                if _infer_modality_from_path(Path(path)) == modality
            ]
            if not modality_paths:
                self.log_message.emit(
                    f"[{idx}/{total_modalities}] Skipping {modality}: no NPZ files remain after filtering."
                )
                continue

            run_kwargs = dict(process_kwargs)
            run_kwargs["con_path_list"] = modality_paths
            run_kwargs["modality"] = modality
            run_kwargs["matrix_outpath"] = _output_path_for_modality(base_output_path, modality)
            run_kwargs["parcel_labels_retain"] = list(retained_labels) if align_cross_modality else []

            self.log_message.emit(
                f"[{idx}/{total_modalities}] Running {modality} on {len(modality_paths)} NPZ file(s)."
            )
            summary = self._run_construct(construct_main, run_kwargs)
            runs.append(
                {
                    "summary": summary,
                    "process_kwargs": dict(run_kwargs),
                }
            )
            processed_modalities.append(modality)

            output_path = Path(str(run_kwargs.get("matrix_outpath") or ""))
            if not output_path.is_file():
                raise RuntimeError(
                    str(summary or f"{modality} processing finished without creating {output_path}.")
                )
            if idx == 1:
                retained_labels = _load_retained_labels_from_output(output_path)
                if align_cross_modality:
                    self.log_message.emit(
                        f"Reference modality {modality} produced {len(retained_labels)} retained parcel labels."
                    )

        if not runs:
            raise RuntimeError("No modality-specific NPZ files were available for multimodal processing.")

        message = (
            "Saved multimodal stacked connectivity: "
            + ", ".join(
                f"{run['process_kwargs'].get('modality')} -> {run['process_kwargs'].get('matrix_outpath')}"
                for run in runs
            )
        )
        return {
            "message": message,
            "primary_index": 0,
            "runs": runs,
            "modalities": processed_modalities,
        }

    def run(self):
        try:
            construct_main = _load_construct_matrix_pop_main()
            process_kwargs = dict(self._process_kwargs)
            multimodal_config = process_kwargs.pop("_multimodal_config", None)
            if multimodal_config and multimodal_config.get("enabled"):
                summary = self._run_multimodal(construct_main, process_kwargs, multimodal_config)
            else:
                summary = self._run_single(construct_main, process_kwargs)
        except Exception as exc:
            self.log_message.emit(traceback.format_exc())
            self.failed.emit(str(exc))
            return
        self.completed.emit(summary)


class StackPrepareDialog(QWidget):
    configuration_changed = pyqtSignal()
    process_state_changed = pyqtSignal()
    log_message_emitted = pyqtSignal(str)

    def __init__(
        self,
        selected_paths,
        theme_name="Dark",
        export_callback=None,
        close_callback=None,
        selected_paths_provider=None,
        validation_error_provider=None,
        covars_df_provider=None,
        selection_suffix_provider=None,
        multimodal_config_provider=None,
        include_covars_widget=True,
        show_embedded_terminal=True,
        show_import_button=True,
        show_process_button=True,
        default_results_dir="",
        default_bids_dir="",
        default_atlas_dir="",
        parent=None,
    ):
        super().__init__(parent)
        _ensure_pandas()
        self._selected_paths = [Path(path) for path in (selected_paths or [])]
        self._theme_name = "Dark"
        self._export_callback = export_callback
        self._close_callback = close_callback
        self._selected_paths_provider = selected_paths_provider
        self._validation_error_provider = validation_error_provider
        self._covars_df_provider = covars_df_provider
        self._selection_suffix_provider = selection_suffix_provider
        self._multimodal_config_provider = multimodal_config_provider
        self._include_covars_widget = bool(include_covars_widget)
        self._show_embedded_terminal = bool(show_embedded_terminal)
        self._show_import_button = bool(show_import_button)
        self._show_process_button = bool(show_process_button)
        self._default_results_dir = str(default_results_dir or "").strip()
        self._default_bids_dir = str(default_bids_dir or "").strip()
        self._default_atlas_dir = str(default_atlas_dir or "").strip()
        self._covars_widget = None
        self._matrix_key = None
        self._output_path_auto = True
        self._processing = False
        self._worker_thread = None
        self._worker = None
        self._last_process_kwargs = None
        self._last_output_path = None
        self._last_subjects_tsv_path = None
        self._last_gradients_path = None
        self._last_voxel_count_path = None
        self._last_external_validation_error = ""
        self._last_exit_code = None
        self._include_parcel_label_max = 0
        self._detected_atlas = _atlas_tag_from_path(self._selected_paths[0]) if self._selected_paths else "unknown"
        self._detected_modality = (
            _infer_modality_from_path(self._selected_paths[0]) if self._selected_paths else "connectivity"
        )
        self._detected_modalities = _modalities_from_paths(self._selected_paths)
        self.setWindowTitle("Stack Connectivity")
        self.resize(1080, 760)
        # Keep drag/drop ownership inside the dedicated covariate widget.
        self.setAcceptDrops(False)
        self._build_ui()
        _stack_diagnostic_log(
            f"StackPrepareDialog initialized include_covars_widget={self._include_covars_widget} "
            f"selected_paths={len(self._selected_paths)}"
        )
        try:
            self.destroyed.connect(lambda *_args: _stack_diagnostic_log("StackPrepareDialog destroyed"))
        except Exception:
            pass
        self._refresh_selection_summary()
        self._set_default_output_path(force=True)
        self.set_theme(theme_name)
        self._apply_terminal_style()
        self._update_process_enabled()

    def _build_ui(self):
        root = QVBoxLayout(self)

        summary_group = QGroupBox("Selection")
        summary_layout = QVBoxLayout(summary_group)
        self.selection_label = QLabel("")
        self.selection_label.setWordWrap(True)
        summary_layout.addWidget(self.selection_label)
        root.addWidget(summary_group)

        options_group = QGroupBox("Processing")
        options_layout = QGridLayout(options_group)
        options_layout.addWidget(QLabel("Ignore parcels"), 0, 0)
        self.ignore_parc_edit = QLineEdit("")
        self.ignore_parc_edit.setPlaceholderText("Comma-separated substrings, e.g. wm-, ventricle")
        options_layout.addWidget(self.ignore_parc_edit, 0, 1, 1, 3)

        options_layout.addWidget(QLabel("Qmask (optional)"), 1, 0)
        self.qmask_path_edit = FileDropLineEdit(accepted_suffixes=(".nii", ".nii.gz"))
        self.qmask_path_edit.setPlaceholderText("Optional qmask .nii or .nii.gz path")
        self.qmask_path_edit.textChanged.connect(self._sync_processing_options)
        self.qmask_path_edit.textChanged.connect(self._update_process_enabled)
        options_layout.addWidget(self.qmask_path_edit, 1, 1, 1, 2)
        self.qmask_browse_button = QPushButton("Browse")
        self.qmask_browse_button.clicked.connect(self._browse_qmask_file)
        options_layout.addWidget(self.qmask_browse_button, 1, 3)

        self.parcellation_label = QLabel("Parcellation image (required)")
        options_layout.addWidget(self.parcellation_label, 2, 0)
        self.parcellation_path_edit = FileDropLineEdit(accepted_suffixes=(".nii", ".nii.gz"))
        self.parcellation_path_edit.setPlaceholderText(
            "Required NIfTI parcellation image (.nii or .nii.gz)"
        )
        self.parcellation_path_edit.textChanged.connect(self._sync_processing_options)
        self.parcellation_path_edit.textChanged.connect(self._update_process_enabled)
        options_layout.addWidget(self.parcellation_path_edit, 2, 1, 1, 2)
        self.parcellation_browse_button = QPushButton("Browse")
        self.parcellation_browse_button.clicked.connect(self._browse_parcellation_file)
        options_layout.addWidget(self.parcellation_browse_button, 2, 3)

        self.mrsi_cov_label = QLabel("MRSI coverage")
        options_layout.addWidget(self.mrsi_cov_label, 3, 0)
        self.mrsi_cov_spin = QDoubleSpinBox()
        self.mrsi_cov_spin.setRange(0.0, 1.0)
        self.mrsi_cov_spin.setSingleStep(0.01)
        self.mrsi_cov_spin.setDecimals(2)
        self.mrsi_cov_spin.setValue(0.67)
        options_layout.addWidget(self.mrsi_cov_spin, 3, 1)

        self.compute_gradients_check = QCheckBox("Compute gradients")
        self.compute_gradients_check.setObjectName("switchCheckBox")
        self.compute_gradients_check.setChecked(True)
        self.compute_gradients_check.toggled.connect(self._sync_processing_options)
        self.compute_gradients_check.toggled.connect(self._update_process_enabled)
        options_layout.addWidget(self.compute_gradients_check, 4, 0, 1, 4)

        self.include_parcels_row = QWidget()
        include_parcels_layout = QHBoxLayout(self.include_parcels_row)
        include_parcels_layout.setContentsMargins(0, 0, 0, 0)
        include_parcels_layout.setSpacing(8)
        self.include_parcels_check = QCheckBox("Include parcels")
        self.include_parcels_check.setObjectName("switchCheckBox")
        self.include_parcels_check.setChecked(False)
        self.include_parcels_check.toggled.connect(self._sync_include_parcel_controls)
        include_parcels_layout.addWidget(self.include_parcels_check)
        self.include_parcels_start_label = QLabel("Start label")
        include_parcels_layout.addWidget(self.include_parcels_start_label)
        self.include_parcels_start_spin = QSpinBox()
        self.include_parcels_start_spin.setRange(0, 0)
        self.include_parcels_start_spin.setValue(1)
        self.include_parcels_start_spin.setKeyboardTracking(False)
        self.include_parcels_start_spin.valueChanged.connect(self._on_include_parcel_start_changed)
        include_parcels_layout.addWidget(self.include_parcels_start_spin)
        self.include_parcels_end_label = QLabel("End label")
        include_parcels_layout.addWidget(self.include_parcels_end_label)
        self.include_parcels_end_spin = QSpinBox()
        self.include_parcels_end_spin.setRange(0, 0)
        self.include_parcels_end_spin.setValue(0)
        self.include_parcels_end_spin.setKeyboardTracking(False)
        self.include_parcels_end_spin.valueChanged.connect(self._on_include_parcel_end_changed)
        include_parcels_layout.addWidget(self.include_parcels_end_spin)
        include_parcels_layout.addStretch(1)
        options_layout.addWidget(self.include_parcels_row, 5, 0, 1, 4)
        root.addWidget(options_group)

        self._refresh_include_parcel_defaults(force=True)
        self._sync_include_parcel_controls()
        self._sync_processing_options()

        if self._include_covars_widget:
            covars_group = QGroupBox("Covariates TSV (Optional)")
            covars_layout = QVBoxLayout(covars_group)
            self._covars_widget = CovarsSelectionWidget(selected_paths=self._selected_paths, parent=covars_group)
            self._covars_widget.configuration_changed.connect(self._handle_covars_selection_changed)
            try:
                self._covars_widget.destroyed.connect(
                    lambda *_args: _stack_diagnostic_log("CovarsSelectionWidget destroyed")
                )
            except Exception:
                pass
            covars_layout.addWidget(self._covars_widget, 1)
            root.addWidget(covars_group, 1)

        output_group = QGroupBox("Output")
        output_layout = QGridLayout(output_group)
        output_layout.addWidget(QLabel("Output NPZ (BIDS)"), 0, 0)
        self.output_path_edit = QLineEdit("")
        self.output_path_edit.setReadOnly(True)
        self.output_path_edit.setToolTip(
            "Stack output is written next to the source connectivity matrices under "
            "derivatives/group/connectivity/<modality>."
        )
        output_layout.addWidget(self.output_path_edit, 0, 1)
        self.output_browse_button = QPushButton("Browse")
        self.output_browse_button.setEnabled(False)
        self.output_browse_button.setToolTip(
            "Stack output location is fixed to the source BIDS derivatives/group/connectivity folder."
        )
        self.output_browse_button.clicked.connect(self._browse_output_path)
        output_layout.addWidget(self.output_browse_button, 0, 2)
        root.addWidget(output_group)

        self.log_group = QGroupBox("Integrated Terminal")
        log_layout = QVBoxLayout(self.log_group)
        self.log_output = QPlainTextEdit()
        self.log_output.setObjectName("stackTerminal")
        self.log_output.setReadOnly(True)
        self.log_output.document().setMaximumBlockCount(4000)
        if QT_LIB == 6:
            self.log_output.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        else:
            self.log_output.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.log_output.setPlaceholderText("Stack progress will appear here.")
        self.log_output.setMinimumHeight(180)
        log_layout.addWidget(self.log_output, 1)
        root.addWidget(self.log_group, 1)
        self.log_group.setVisible(self._show_embedded_terminal)

        self.status_label = QLabel("Ready.")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        actions = QHBoxLayout()
        actions.addStretch(1)
        self.import_button = QPushButton("Import to workspace")
        self.import_button.setEnabled(False)
        self.import_button.clicked.connect(self._import_last_output)
        self.import_button.setVisible(self._show_import_button)
        actions.addWidget(self.import_button)
        self.close_button = QPushButton("Close")
        if callable(self._close_callback):
            self.close_button.setText("Back")
        self.close_button.clicked.connect(self._handle_close_requested)
        actions.addWidget(self.close_button)
        self.process_button = QPushButton("Process")
        self.process_button.setEnabled(False)
        self.process_button.clicked.connect(self._process)
        self.process_button.setVisible(self._show_process_button)
        actions.addWidget(self.process_button)
        root.addLayout(actions)

    def set_selected_paths(self, selected_paths):
        self._selected_paths = [Path(path) for path in (selected_paths or [])]
        if self._covars_widget is not None:
            self._covars_widget.set_selected_paths(self._selected_paths)
        self._detected_atlas = _atlas_tag_from_path(self._selected_paths[0]) if self._selected_paths else "unknown"
        self._detected_modality = (
            _infer_modality_from_path(self._selected_paths[0]) if self._selected_paths else "connectivity"
        )
        self._detected_modalities = _modalities_from_paths(self._selected_paths)
        self._refresh_selection_summary()
        self._refresh_include_parcel_defaults(force=True)
        self._sync_processing_options()
        if self._output_path_auto:
            self._set_default_output_path(force=True)
        self._update_process_enabled()
        self.configuration_changed.emit()

    def _handle_covars_selection_changed(self):
        if self._output_path_auto:
            self._set_default_output_path(force=True)
        self._update_process_enabled()
        self.configuration_changed.emit()

    def _refresh_selection_summary(self):
        first_path = self._selected_paths[0] if self._selected_paths else None
        modalities = ", ".join(self._detected_modalities) if self._detected_modalities else "unknown"
        self.selection_label.setText(
            f"Selected NPZ files: {len(self._selected_paths)}"
            + (f"\nFirst file: {first_path}" if first_path else "")
            + (f"\nAtlas: {self._detected_atlas}" if self._selected_paths else "")
            + (f"\nPrimary modality: {self._detected_modality}" if self._selected_paths else "")
            + (f"\nDetected modalities: {modalities}" if self._selected_paths else "")
            + (
                f"\nMatrix key: {self._matrix_key}"
                if self._selected_paths and self._matrix_key
                else ("\nMatrix key: Auto (modality default)" if self._selected_paths else "")
            )
        )

    def set_matrix_key(self, matrix_key):
        normalized = str(matrix_key or "").strip() or None
        if normalized == self._matrix_key:
            return
        self._matrix_key = normalized
        self._refresh_selection_summary()
        self._update_process_enabled()
        self.configuration_changed.emit()

    def _handle_close_requested(self):
        _stack_diagnostic_log(
            f"_handle_close_requested processing={self._process_running()} close_callback={callable(self._close_callback)}"
        )
        if self._process_running():
            self._set_status("Wait for processing to finish before closing.")
            return
        if callable(self._close_callback):
            self._cleanup_worker_thread()
            self._close_callback()
            return
        self._prepare_child_widgets_for_close()
        self._cleanup_worker_thread()
        self.close()

    def selected_covars_df(self):
        covars = self._selected_covars_df()
        if covars is None:
            return None
        return covars.copy()

    def detected_modalities(self):
        return list(self._detected_modalities)

    def effective_selected_paths(self):
        if callable(self._selected_paths_provider):
            try:
                provided = self._selected_paths_provider()
            except Exception:
                provided = None
            if provided is not None:
                return [Path(path) for path in provided if str(path).strip()]
        return list(self._selected_paths)

    def refresh_process_state(self):
        if self._output_path_auto:
            self._set_default_output_path(force=True)
        self._update_process_enabled()

    def can_process(self):
        return bool(getattr(self, "process_button", None) and self.process_button.isEnabled())

    def can_import_last_output(self):
        return bool(getattr(self, "import_button", None) and self.import_button.isEnabled())

    def is_processing(self):
        return self._process_running()

    def trigger_process(self):
        self._process()

    def trigger_import_last_output(self):
        self._import_last_output()

    def _external_validation_error(self):
        if not callable(self._validation_error_provider):
            return ""
        try:
            message = self._validation_error_provider()
        except Exception:
            return ""
        return str(message or "").strip()

    def _sync_processing_options(self):
        has_mrsi = "mrsi" in {str(item).strip().lower() for item in (self._detected_modalities or [])}
        if not has_mrsi:
            has_mrsi = str(self._detected_modality or "").strip().lower() == "mrsi"
        self.qmask_path_edit.setEnabled(has_mrsi)
        self.qmask_browse_button.setEnabled(has_mrsi)
        show_mrsi_cov = has_mrsi and bool(self.qmask_path_edit.text().strip())
        self.mrsi_cov_label.setVisible(show_mrsi_cov)
        self.mrsi_cov_spin.setVisible(show_mrsi_cov)
        self.mrsi_cov_spin.setEnabled(show_mrsi_cov)
        self._sync_include_parcel_controls()
        self._update_parcellation_requirement_ui()

    def _parcellation_required(self):
        return True

    def _update_parcellation_requirement_ui(self):
        self.parcellation_label.setText("Parcellation image (required)")
        self.parcellation_path_edit.setPlaceholderText(
            "Required NIfTI parcellation image (.nii or .nii.gz)"
        )

    def _set_include_parcel_spin_bounds(self, start_value=None, end_value=None):
        max_label = max(0, int(self._include_parcel_label_max))
        if start_value is None:
            start_value = self.include_parcels_start_spin.value()
        if end_value is None:
            end_value = self.include_parcels_end_spin.value()
        start_value = max(0, min(int(start_value), max_label))
        end_value = max(start_value, min(int(end_value), max_label))

        start_blocked = self.include_parcels_start_spin.blockSignals(True)
        end_blocked = self.include_parcels_end_spin.blockSignals(True)
        self.include_parcels_start_spin.setRange(0, max_label)
        self.include_parcels_end_spin.setRange(0, max_label)
        self.include_parcels_start_spin.setValue(start_value)
        self.include_parcels_end_spin.setValue(end_value)
        self.include_parcels_start_spin.setMaximum(end_value)
        self.include_parcels_end_spin.setMinimum(start_value)
        self.include_parcels_start_spin.blockSignals(start_blocked)
        self.include_parcels_end_spin.blockSignals(end_blocked)

    def _refresh_include_parcel_defaults(self, force=False):
        if self._selected_paths:
            self._include_parcel_label_max = _max_parcel_label_from_npz(self._selected_paths[0])
        else:
            self._include_parcel_label_max = 0
        if force:
            self._set_include_parcel_spin_bounds(
                start_value=1,
                end_value=self._include_parcel_label_max,
            )
        else:
            self._set_include_parcel_spin_bounds()

    def _sync_include_parcel_controls(self):
        enabled = bool(self.include_parcels_check.isChecked())
        for widget in (
            self.include_parcels_start_label,
            self.include_parcels_start_spin,
            self.include_parcels_end_label,
            self.include_parcels_end_spin,
        ):
            widget.setEnabled(enabled)

    def _on_include_parcel_start_changed(self, value):
        self._set_include_parcel_spin_bounds(start_value=value)

    def _on_include_parcel_end_changed(self, value):
        self._set_include_parcel_spin_bounds(end_value=value)

    def _browse_qmask_file(self):
        start_dir = str(self._selected_paths[0].parent) if self._selected_paths else (
            self._default_bids_dir or str(Path.home())
        )
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select qmask image",
            start_dir,
            "NIfTI files (*.nii *.nii.gz);;All files (*)",
        )
        if selected:
            self.qmask_path_edit.setText(selected)
        self._update_process_enabled()

    def _browse_parcellation_file(self):
        start_dir = self._default_atlas_dir or str(_default_parcellation_dir())
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select parcellation image",
            start_dir,
            "NIfTI files (*.nii *.nii.gz);;All files (*)",
        )
        if selected:
            self.parcellation_path_edit.setText(selected)
        self._update_process_enabled()

    def _current_selection_suffix(self):
        if callable(self._selection_suffix_provider):
            try:
                suffix = self._selection_suffix_provider()
            except Exception:
                suffix = None
            if isinstance(suffix, (tuple, list)) and len(suffix) == 2:
                return str(suffix[0] or "all"), str(suffix[1] or "all")
        if self._covars_widget is not None:
            return self._covars_widget.current_selection_suffix()
        return "all", "all"

    def _current_multimodal_config(self):
        if not callable(self._multimodal_config_provider):
            return None
        try:
            config = self._multimodal_config_provider()
        except Exception:
            return None
        if not isinstance(config, dict) or not config.get("enabled"):
            return None
        ordered_modalities = [
            str(item).strip().lower()
            for item in config.get("ordered_modalities", [])
            if str(item).strip()
        ]
        if len(ordered_modalities) <= 1:
            return None
        return {
            "enabled": True,
            "ordered_modalities": ordered_modalities,
        }

    def _resolved_stack_output_path(self) -> Path | None:
        if not self._selected_paths:
            return None
        first = self._selected_paths[0]
        group_root = _group_root_from_path(first)
        group_name = _slugify(_infer_group_from_path(first))
        atlas_tag = _slugify(_atlas_tag_from_path(first))
        multimodal_config = self._current_multimodal_config()
        if multimodal_config and multimodal_config.get("ordered_modalities"):
            modality = _slugify(multimodal_config["ordered_modalities"][0])
            output_dir = group_root / "derivatives" / "group" / "connectivity" / "multimodal"
        else:
            modality = _slugify(_infer_modality_from_path(first))
            output_dir = group_root / "derivatives" / "group" / "connectivity" / modality
        filter_var, filter_values = self._current_selection_suffix()
        sel_fragment = ""
        if not (filter_var == "all" and filter_values == "all"):
            sel_fragment = f"_sel-{filter_var}_{filter_values}"
        name = (
            f"{group_name}_atlas-{atlas_tag}{sel_fragment}"
            f"_desc-group_connectivity_{modality}.npz"
        )
        return output_dir / name

    def _set_default_output_path(self, force=False):
        if not self._selected_paths:
            return
        if not force and not self._output_path_auto:
            return
        resolved = self._resolved_stack_output_path()
        if resolved is None:
            return
        self.output_path_edit.setText(str(resolved))
        self._output_path_auto = True

    def _on_output_path_edited(self, _text):
        self._output_path_auto = False
        self._update_process_enabled()

    def _browse_output_path(self):
        resolved = self._resolved_stack_output_path()
        if resolved is not None:
            self.output_path_edit.setText(str(resolved))
        self._set_status(
            "Stack output path is fixed to derivatives/group/connectivity/<modality> in the source BIDS dataset."
        )
        self._update_process_enabled()

    def _selected_covars_df(self):
        if callable(self._covars_df_provider):
            try:
                covars = self._covars_df_provider()
            except Exception:
                covars = None
            if covars is None:
                return None
            return covars.copy() if hasattr(covars, "copy") else covars
        if self._covars_widget is not None:
            return self._covars_widget.selected_covars_df()
        return None

    def _prepare_child_widgets_for_close(self):
        _stack_diagnostic_log("_prepare_child_widgets_for_close called")
        if self._covars_widget is not None and hasattr(self._covars_widget, "prepare_for_close"):
            try:
                self._covars_widget.prepare_for_close()
            except Exception:
                pass

    def _parse_ignore_parcels(self):
        values = [token.strip() for token in self.ignore_parc_edit.text().split(",")]
        return [token for token in values if token]

    def _update_process_enabled(self):
        effective_paths = self.effective_selected_paths()
        required_ok = bool(self.output_path_edit.text().strip()) and bool(effective_paths)
        covars_df = self._selected_covars_df()
        covars_error = self._covars_widget.covars_error() if self._covars_widget is not None else ""
        covars_ok = not covars_error
        if covars_df is not None:
            covars_ok = covars_ok and len(covars_df) > 0
        parcellation_path = self.parcellation_path_edit.text().strip()
        parcellation_ok = True
        if self._parcellation_required():
            parcellation_ok = bool(parcellation_path) and Path(parcellation_path).is_file()
        elif parcellation_path:
            parcellation_ok = Path(parcellation_path).is_file()
        validation_error = self._external_validation_error()
        can_process = (
            required_ok
            and covars_ok
            and parcellation_ok
            and not validation_error
            and not self._process_running()
        )
        self.process_button.setEnabled(can_process)
        if validation_error and not self._process_running():
            self._set_status(validation_error)
        elif not parcellation_ok and not self._process_running():
            self._set_status("Select a valid parcellation image before processing.")
        elif (
            not validation_error
            and self._last_external_validation_error
            and self.status_label.text().strip() == self._last_external_validation_error
            and can_process
        ):
            self._set_status("Ready to process.")
        self._last_external_validation_error = validation_error
        can_import = (
            not self._process_running()
            and self._export_callback is not None
            and bool(self._last_output_path)
            and Path(str(self._last_output_path)).is_file()
        )
        self.import_button.setEnabled(can_import)
        self.process_state_changed.emit()

    def _set_status(self, text):
        self.status_label.setText(str(text))

    def _process_running(self):
        if bool(self._processing):
            return True
        thread = self._worker_thread
        if thread is None:
            return False
        try:
            return bool(thread.isRunning())
        except Exception as exc:
            _stack_diagnostic_log(f"_process_running thread check failed: {exc}")
            return False

    def _append_log(self, text):
        cleaned = ANSI_ESCAPE_RE.sub("", str(text or ""))
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
        cleaned = cleaned.rstrip("\n")
        if not cleaned:
            return
        for line in cleaned.split("\n"):
            if line.strip() == "":
                continue
            self.log_output.appendPlainText(line)
            self.log_message_emitted.emit(line)
        scroll = self.log_output.verticalScrollBar()
        if scroll is not None:
            scroll.setValue(scroll.maximum())

    def _apply_terminal_style(self):
        self.log_output.setStyleSheet(
            "QPlainTextEdit#stackTerminal {"
            " background-color: #000000;"
            " color: #b8f7c6;"
            " border: 1px solid #404040;"
            " border-radius: 4px;"
            " selection-background-color: #2d6cdf;"
            " font-family: 'DejaVu Sans Mono', 'Courier New', monospace;"
            " font-size: 10.5pt;"
            "}"
        )

    def _clear_last_result(self):
        self._last_output_path = None
        self._last_subjects_tsv_path = None
        self._last_gradients_path = None
        self._last_voxel_count_path = None

    def _build_process_kwargs(self):
        covars_df = self._selected_covars_df()
        if covars_df is not None and covars_df.empty:
            raise ValueError("No covariate rows remain after filtering.")
        validation_error = self._external_validation_error()
        if validation_error:
            raise ValueError(validation_error)
        effective_paths = self.effective_selected_paths()
        if not effective_paths:
            raise ValueError("No NPZ files remain after applying the current filters.")
        output_path = self._resolved_stack_output_path()
        if output_path is None:
            raise ValueError("Unable to resolve a BIDS output path for the stack.")
        self.output_path_edit.setText(str(output_path))

        return {
            "con_path_list": [str(path) for path in effective_paths],
            "covar_df": covars_df,
            "modality": self._detected_modality,
            "matrix_key": self._matrix_key,
            "ignore_parc_list": self._parse_ignore_parcels(),
            "qmask_path": self.qmask_path_edit.text().strip() or None,
            "mrsi_cov": float(self.mrsi_cov_spin.value()),
            "comp_gradients": bool(self.compute_gradients_check.isChecked()),
            "include_parcels": bool(self.include_parcels_check.isChecked()),
            "include_parcels_start": int(self.include_parcels_start_spin.value()),
            "include_parcels_end": int(self.include_parcels_end_spin.value()),
            "matrix_outpath": str(output_path),
            "parcellation_img_path": self.parcellation_path_edit.text().strip() or None,
            "parcel_labels_retain": [],
            "_multimodal_config": self._current_multimodal_config(),
        }

    def _append_processor_log(self, text):
        self._append_log(text)
        app = QApplication.instance()
        if app is not None:
            app.processEvents()

    def _import_last_output(self):
        if not self._last_output_path:
            self._set_status("No completed stack output is available to import.")
            return
        if self._export_callback is None:
            self._set_status("No workspace import callback is configured.")
            return
        try:
            ok = bool(self._export_callback({"output_path": self._last_output_path}))
        except Exception as exc:
            self._set_status(f"Workspace import failed: {exc}")
            return
        if ok:
            self._set_status(f"Imported {Path(self._last_output_path).name} into the workspace.")
        else:
            self._set_status("The stacked file was written, but the workspace import was rejected.")
        self._update_process_enabled()

    def _process(self):
        if self._process_running():
            self._set_status("A stack process is already running.")
            return
        if not self.process_button.isEnabled():
            self._set_status("Complete the required selections before processing.")
            return

        try:
            process_kwargs = self._build_process_kwargs()
        except Exception as exc:
            self._set_status(f"Failed to prepare processing inputs: {exc}")
            return

        self._clear_last_result()
        self.log_output.clear()
        self._append_log("Starting construct_matrix_pop.main(...).")
        self._processing = True
        self._last_process_kwargs = dict(process_kwargs)
        self._set_status("Processing started. Progress is shown in the integrated terminal.")
        self._update_process_enabled()

        self._worker_thread = QThread(self)
        self._worker = _StackWorker(process_kwargs)
        self._worker.moveToThread(self._worker_thread)
        self._worker.log_message.connect(self._append_processor_log)
        self._worker.completed.connect(self._on_worker_completed)
        self._worker.failed.connect(self._on_worker_failed)
        self._worker.completed.connect(self._worker_thread.quit)
        self._worker.failed.connect(self._worker_thread.quit)
        self._worker_thread.started.connect(self._worker.run)
        self._worker_thread.finished.connect(self._on_worker_finished)
        self._worker_thread.finished.connect(self._worker.deleteLater)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)
        self._worker_thread.start()

    def _handle_process_success(self, summary_payload):
        if isinstance(summary_payload, dict) and isinstance(summary_payload.get("runs"), list):
            runs = [run for run in summary_payload.get("runs", []) if isinstance(run, dict)]
            message = str(summary_payload.get("message") or "").strip()
            primary_index = int(summary_payload.get("primary_index", 0) or 0)
        else:
            runs = [{"summary": summary_payload, "process_kwargs": dict(self._last_process_kwargs or {})}]
            message = str(summary_payload).strip() if summary_payload is not None else ""
            primary_index = 0

        if not runs:
            self._set_status(message or "Processing finished, but no output file was created.")
            return

        primary_index = max(0, min(primary_index, len(runs) - 1))
        primary_run = runs[primary_index]
        primary_kwargs = dict(primary_run.get("process_kwargs") or {})
        matrix_outpath = str(primary_kwargs.get("matrix_outpath") or "").strip()
        if matrix_outpath and Path(matrix_outpath).is_file():
            self._last_output_path = matrix_outpath
            subjects_path = matrix_outpath.replace(".npz", "_subjects.tsv")
            if Path(subjects_path).is_file():
                self._last_subjects_tsv_path = subjects_path
            if bool(primary_kwargs.get("comp_gradients")):
                gradients_path = matrix_outpath.replace("connectivity", "gradients")
                if Path(gradients_path).is_file():
                    self._last_gradients_path = gradients_path
            if str(primary_kwargs.get("modality") or "").strip().lower() == "mrsi":
                voxel_count_path = (
                    str(Path(matrix_outpath).parent / Path(matrix_outpath).name)
                    .replace(f"connectivity_{primary_kwargs.get('modality')}", "voxelcount_per_parcel")
                    .replace(".npz", ".csv")
                )
                if Path(voxel_count_path).is_file():
                    self._last_voxel_count_path = voxel_count_path
            if message:
                self._append_log(message)
                self._set_status(f"{message} Use 'Import to workspace' to load the output.")
            else:
                self._set_status(
                    f"Saved {Path(matrix_outpath).name}. Use 'Import to workspace' to load the output."
                )
        else:
            self._set_status(message or "Processing finished, but no output file was created.")

    def _on_worker_completed(self, summary):
        self._last_exit_code = 0
        self._handle_process_success(summary)
        self._append_log("Process finished with exit code 0")

    def _on_worker_failed(self, message):
        self._last_exit_code = 1
        self._set_status(f"Processing failed: {message}")
        self._append_log("Process finished with exit code 1")

    def _cleanup_worker_thread(self):
        thread = self._worker_thread
        worker = self._worker
        running = False
        if thread is not None:
            try:
                running = bool(thread.isRunning())
            except Exception as exc:
                _stack_diagnostic_log(f"_cleanup_worker_thread isRunning check failed: {exc}")
        _stack_diagnostic_log(
            f"_cleanup_worker_thread thread_exists={thread is not None} running={running}"
        )

        # Drop references first to avoid re-entrant cleanup touching stale wrappers.
        self._worker = None
        self._worker_thread = None

        if running and thread is not None:
            try:
                if hasattr(thread, "requestInterruption"):
                    thread.requestInterruption()
            except Exception:
                pass
            try:
                thread.quit()
            except Exception:
                pass

        # Worker QObject is already wired to deleteLater on thread finish.
        # Avoid invoking deleteLater here on possibly stale wrappers.
        _stack_diagnostic_log("_cleanup_worker_thread complete")

    def _on_worker_finished(self):
        self._processing = False
        self._cleanup_worker_thread()
        self._update_process_enabled()

    def prepare_for_close(self) -> bool:
        _stack_diagnostic_log(f"prepare_for_close called processing={self._process_running()}")
        if self._process_running():
            return False
        self._prepare_child_widgets_for_close()
        self._cleanup_worker_thread()
        return True

    def closeEvent(self, event):
        _stack_diagnostic_log(f"closeEvent entered processing={self._process_running()}")
        if self._process_running():
            event.ignore()
            self._set_status("Wait for processing to finish before closing.")
            _stack_diagnostic_log("closeEvent ignored because processing is active")
            return
        self._prepare_child_widgets_for_close()
        self._cleanup_worker_thread()
        _stack_diagnostic_log("closeEvent delegating to QWidget.closeEvent")
        super().closeEvent(event)

    def set_theme(self, theme_name):
        theme, style = _workflow_dialog_stylesheet(
            theme_name,
            control_selector="QPushButton, QComboBox, QLineEdit, QTableWidget, QPlainTextEdit",
        )
        self._theme_name = theme
        style += _switch_checkbox_theme_styles(theme, builder=_switch_checkbox_style)
        self.setStyleSheet(style)
        self._apply_terminal_style()
