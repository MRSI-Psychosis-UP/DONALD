#!/usr/bin/env python3
"""NBS preparation dialog for covariate role selection and row filtering."""

import csv
import importlib.util
import json
import os
import re
import shlex
import shutil
import sys
from pathlib import Path

import numpy as np

try:
    from window.shared.controls import make_toggle_button as _make_toggle_button
    from window.shared.covariate_table import (
        CovariateFilterPage as _CovariateFilterPage,
        filter_row_indices as _filter_row_indices,
        parse_filter_values as _parse_filter_values,
    )
    from window.shared.covars import (
        column_is_numeric as _column_is_numeric,
        covars_to_rows as _covars_to_rows,
        display_text as _display_text,
    )
    from window.shared.matrix import stack_axis as _stack_axis
    from window.shared.qt_flags import (
        install_qt_compat_aliases,
        is_editable_flag as _is_editable_flag,
        is_enabled_flag as _is_enabled_flag,
        is_selectable_flag as _is_selectable_flag,
        is_user_checkable_flag as _is_user_checkable_flag,
    )
    from window.shared.workflow import WorkflowShell as _WorkflowShell
    from window.shared.theme import workflow_dialog_stylesheet as _workflow_dialog_stylesheet
except Exception:
    from mrsi_viewer.window.shared.controls import make_toggle_button as _make_toggle_button
    from mrsi_viewer.window.shared.covariate_table import (
        CovariateFilterPage as _CovariateFilterPage,
        filter_row_indices as _filter_row_indices,
        parse_filter_values as _parse_filter_values,
    )
    from mrsi_viewer.window.shared.covars import (
        column_is_numeric as _column_is_numeric,
        covars_to_rows as _covars_to_rows,
        display_text as _display_text,
    )
    from mrsi_viewer.window.shared.matrix import stack_axis as _stack_axis
    from mrsi_viewer.window.shared.qt_flags import (
        install_qt_compat_aliases,
        is_editable_flag as _is_editable_flag,
        is_enabled_flag as _is_enabled_flag,
        is_selectable_flag as _is_selectable_flag,
        is_user_checkable_flag as _is_user_checkable_flag,
    )
    from mrsi_viewer.window.shared.workflow import WorkflowShell as _WorkflowShell
    from mrsi_viewer.window.shared.theme import workflow_dialog_stylesheet as _workflow_dialog_stylesheet

try:
    from PyQt6.QtCore import QProcess, QProcessEnvironment, QTimer, Qt
    from PyQt6.QtGui import QPixmap
    from PyQt6.QtWidgets import (
        QAbstractItemView,
        QCheckBox,
        QComboBox,
        QDialog,
        QDoubleSpinBox,
        QFileDialog,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHeaderView,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSplitter,
        QSpinBox,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
    QT_LIB = 6
except Exception:
    from PyQt5.QtCore import QProcess, QProcessEnvironment, QTimer, Qt
    from PyQt5.QtGui import QPixmap
    from PyQt5.QtWidgets import (
        QAbstractItemView,
        QCheckBox,
        QComboBox,
        QDialog,
        QDoubleSpinBox,
        QFileDialog,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHeaderView,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSplitter,
        QSpinBox,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
    QT_LIB = 5


install_qt_compat_aliases(Qt, QT_LIB)


def _numeric_sortable(values):
    out = []
    for value in values:
        try:
            out.append(float(value))
        except Exception:
            return None
    return out


def _qprocess_not_running():
    if hasattr(QProcess, "NotRunning"):
        return QProcess.NotRunning
    process_state = getattr(QProcess, "ProcessState", None)
    if process_state is not None and hasattr(process_state, "NotRunning"):
        return process_state.NotRunning
    return 0


def _slugify_fragment(value):
    text = str(value)
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", text)
    slug = slug.strip("-_")
    return slug or "value"


def _describe_contrast_tail(contrast_text: str) -> str:
    """Summarize the tail a contrast string implies, for display only.

    Mirrors run_nbs.py's own contrast -> tail convention (sign of the numeric
    entries; 'b' means a two-tailed pos/neg pair with Bonferroni doubling) without
    importing that CLI module into the GUI process.
    """
    text = (contrast_text or "").strip()
    if not text:
        return "n/a (contrast required)"
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    parts = [p for p in re.split(r"[,\s]+", text) if p]
    if any(p.lower() == "b" for p in parts):
        return "both (two-tailed, Bonferroni-doubled)"
    try:
        values = [float(p) for p in parts]
    except ValueError:
        return "unknown"
    has_pos = any(v > 0 for v in values)
    has_neg = any(v < 0 for v in values)
    if has_pos and has_neg:
        return "both (mixed-sign contrast)"
    if has_pos:
        return "right (one-tailed)"
    if has_neg:
        return "left (one-tailed)"
    return "unknown"


def _connectivity_file_metadata(path: Path) -> dict:
    base_name = Path(path).name
    results = {}
    patterns = {
        "atlas": r"atlas-([^_]+)",
        "scale": r"scale(\d+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, base_name)
        results[key] = match.group(1) if match else None
    if results["scale"] is not None:
        try:
            results["scale"] = int(results["scale"])
        except Exception:
            results["scale"] = None
    return results


def _connectivity_atlas_string(path: Path) -> str:
    meta = _connectivity_file_metadata(Path(path))
    atlas = str(meta.get("atlas") or "").strip()
    scale = meta.get("scale")
    if atlas and scale is not None and f"scale{scale}" not in atlas:
        return f"{atlas}_scale{scale}"
    if atlas:
        return atlas
    return "unknown"


class _ImagePreviewDialog(QDialog):
    def __init__(self, image_path, title, parent=None):
        super().__init__(parent)
        self._image_path = Path(image_path).expanduser()
        self._pixmap = QPixmap(str(self._image_path))
        self.setWindowTitle(title)
        self.resize(980, 760)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        self.info_label = QLabel(str(self._image_path))
        root.addWidget(self.info_label)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter if hasattr(Qt, "AlignmentFlag") else Qt.AlignCenter)
        self.image_label.setMinimumSize(480, 320)

        scroller = QScrollArea()
        scroller.setWidgetResizable(True)
        scroller.setWidget(self.image_label)
        root.addWidget(scroller, 1)

        close_row = QHBoxLayout()
        close_row.addStretch(1)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        close_row.addWidget(close_button)
        root.addLayout(close_row)

        self._apply_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_pixmap()

    def _apply_pixmap(self):
        if self._pixmap.isNull():
            self.image_label.setText(f"Failed to load image: {self._image_path}")
            self.image_label.setPixmap(QPixmap())
            return
        target_size = self.image_label.size()
        if not target_size.isValid() or target_size.width() <= 1 or target_size.height() <= 1:
            self.image_label.setPixmap(self._pixmap)
            return
        transform_mode = Qt.TransformationMode.SmoothTransformation if hasattr(Qt, "TransformationMode") else Qt.SmoothTransformation
        aspect_mode = Qt.AspectRatioMode.KeepAspectRatio if hasattr(Qt, "AspectRatioMode") else Qt.KeepAspectRatio
        scaled = self._pixmap.scaled(target_size, aspect_mode, transform_mode)
        self.image_label.setPixmap(scaled)



class NBSPrepareDialog(QDialog):
    """Dialog to prepare NBS covariates and select filtered row subsets."""

    ROLE_OPTIONS = ("Ignore", "Confound", "Regressor")
    REGRESSOR_TYPE_MODEL_OPTIONS = ("Auto", "Categorical", "Continuous")
    MODALITY_OPTIONS = ("fmri", "mrsi", "dwi", "anat", "morph", "pet", "other")
    TEST_OPTIONS = ("t-test", "Ftest")
    STEP_TITLES = ("Data", "Model", "NBS Settings", "Run", "Results")
    RUN_STEP_INDEX = 3
    RESULTS_STEP_INDEX = 4
    NUMERIC_RANGE_RE = re.compile(
        r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*-\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*$"
    )

    def __init__(
        self,
        covars_info,
        source_path,
        matrix_key,
        matlab_cmd_default="",
        matlab_nbs_path_default="",
        output_dir_default="",
        atlas_dir_default="",
        bids_dir_default="",
        workspace_reference_options=None,
        theme_name="Dark",
        gradient_rgb_payload_callback=None,
        parent=None,
    ):
        super().__init__(parent)
        self._source_path = Path(source_path)
        self._matrix_key = str(matrix_key)
        self._matlab_cmd_default = str(matlab_cmd_default or "").strip()
        self._matlab_nbs_path_default = str(matlab_nbs_path_default or "").strip()
        self._output_dir_default = str(output_dir_default or "").strip()
        self._atlas_dir_default = str(atlas_dir_default or "").strip()
        self._bids_dir_default = str(bids_dir_default or "").strip()
        self._theme_name = "Dark"
        self._source_group = None
        self._source_modality = None
        self._source_atlas_string = _connectivity_atlas_string(self._source_path)
        self._output_dir_auto_value = ""
        self._last_result_npz_path = None
        self._last_nbs_summary = None
        self._last_result_was_staged = False
        self._export_process = None
        self._export_output_tail = []
        self._covariate_balance_process = None
        self._covariate_balance_output_tail = []
        self._last_covariate_balance_dir = None
        self._last_covariate_balance_plot_path = None
        self._covariate_balance_preview_dialog = None
        self._columns, self._rows = _covars_to_rows(covars_info)
        self._filtered_indices = list(range(len(self._rows)))
        self._excluded_indices = set()
        self._covar_checks = {}
        self._confound_buttons = {}
        self._regressor_buttons = {}
        self._last_run_payload = None
        self._run_process = None
        self._run_output_tail = []
        self._results_process = None
        self._results_output_tail = []
        self._results_output_dir_auto_value = ""
        self._last_results_output_dir = None
        self._workspace_reference_options = self._normalize_workspace_reference_options(workspace_reference_options)
        self._gradient_rgb_payload_callback = (
            gradient_rgb_payload_callback if callable(gradient_rgb_payload_callback) else None
        )
        self._reference_workspace_combo_syncing = False
        self._reference_matrix_key_options = []
        self._reference_matrix_key_combo_syncing = False
        self._run_cancel_requested = False
        self._current_step = 0
        self._model_updating = False
        self._data_table_refreshing = False
        self._log_expanded = False
        self._log_auto_expanded = False
        self.terminal_output = None
        self._load_source_metadata()

        self.setWindowTitle("NBS Prepare")
        self.resize(1200, 800)
        self.set_theme(theme_name)
        self._build_ui()
        self._refresh_table()
        self._update_test_options()
        self._update_run_state()

    def _default_results_root(self) -> Path:
        if self._output_dir_default:
            return Path(self._output_dir_default).expanduser()
        if self._source_path is not None:
            return self._source_path.parent
        return Path.cwd()

    def _default_output_dir_for_context(self, modality_text=None) -> str:
        group = _slugify_fragment(self._source_group or "group")
        modality = str(modality_text or "").strip().lower()
        if not modality:
            modality = str(self._source_modality or "mrsi").strip().lower() or "mrsi"
        modality = _slugify_fragment(modality)
        atlas_string = _slugify_fragment(self._source_atlas_string or "unknown")
        return str(self._default_results_root() / "nbs" / group / modality / atlas_string)

    def _results_threshold_token(self, result_path: Path | None = None) -> str | None:
        candidate_path = None
        if result_path is not None:
            candidate_path = Path(result_path).expanduser()
        else:
            raw_path = str(self._last_result_npz_path or "").strip()
            if raw_path:
                candidate_path = Path(raw_path).expanduser()
        if candidate_path is not None:
            match = re.search(r"_th-([^_]+)", candidate_path.name)
            if match:
                token = match.group(1).strip()
                if token:
                    return token
            try:
                with np.load(candidate_path, allow_pickle=True) as npz:
                    if "t_thresh" in npz:
                        value = np.asarray(npz["t_thresh"]).reshape(-1)[0]
                        return f"{float(value):g}"
            except Exception:
                pass
        payload = dict(self._last_run_payload or {})
        if "t_thresh" in payload:
            try:
                return f"{float(payload['t_thresh']):g}"
            except Exception:
                pass
        return None

    def _default_results_output_dir(self) -> str:
        result_path_text = str(self._last_result_npz_path or "").strip()
        if result_path_text:
            result_path = Path(result_path_text).expanduser()
            base_dir = result_path.parent
        else:
            result_path = None
            base_dir = self._default_results_root()
        threshold_token = self._results_threshold_token(result_path)
        folder_name = "nbs_control_structural_context"
        if threshold_token:
            folder_name = f"{folder_name}_th-{threshold_token}"
        output_dir = base_dir / folder_name
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return str(output_dir)

    def _default_reference_matrix_dir(self) -> str:
        candidates = [
            str(self._bids_dir_default or "").strip(),
            str(self._atlas_dir_default or "").strip(),
            str(self._default_results_root()),
            str(self._source_path.parent),
        ]
        for candidate in candidates:
            if not candidate:
                continue
            try:
                path = Path(candidate).expanduser()
            except Exception:
                continue
            if path.is_file():
                return str(path.parent)
            if path.exists():
                return str(path)
        return str(Path.cwd())

    def _normalize_workspace_reference_options(self, options) -> list[dict[str, str]]:
        normalized = []
        seen_paths = set()
        for option in options or []:
            if not isinstance(option, dict):
                continue
            path_text = str(option.get("path") or "").strip()
            if not path_text:
                continue
            try:
                path_obj = Path(path_text).expanduser()
            except Exception:
                continue
            if not path_obj.is_file():
                continue
            try:
                path_key = str(path_obj.resolve())
            except Exception:
                path_key = str(path_obj)
            if path_key in seen_paths:
                continue
            label = str(option.get("label") or path_obj.name).strip() or path_obj.name
            normalized.append({"label": label, "path": str(path_obj)})
            seen_paths.add(path_key)
        normalized.sort(key=lambda item: item["label"].lower())
        return normalized

    def set_workspace_reference_options(self, options) -> None:
        current_path = self.reference_matrix_edit.text().strip() if hasattr(self, "reference_matrix_edit") else ""
        self._workspace_reference_options = self._normalize_workspace_reference_options(options)
        if not hasattr(self, "reference_workspace_combo"):
            return
        self._reference_workspace_combo_syncing = True
        self.reference_workspace_combo.blockSignals(True)
        self.reference_workspace_combo.clear()
        if self._workspace_reference_options:
            self.reference_workspace_combo.addItem("Manual / browse...", "")
            for option in self._workspace_reference_options:
                self.reference_workspace_combo.addItem(str(option["label"]), str(option["path"]))
            self.reference_workspace_combo.setEnabled(not self._results_is_running())
        else:
            self.reference_workspace_combo.addItem("No workspace NPZ files available", "")
            self.reference_workspace_combo.setEnabled(False)
        self.reference_workspace_combo.blockSignals(False)
        self._reference_workspace_combo_syncing = False
        self._sync_reference_workspace_combo(current_path)

    def _sync_reference_workspace_combo(self, reference_path: str | None = None) -> None:
        if not hasattr(self, "reference_workspace_combo"):
            return
        if self._reference_workspace_combo_syncing:
            return
        reference_text = str(reference_path or "").strip()
        try:
            resolved_reference = str(Path(reference_text).expanduser().resolve()) if reference_text else ""
        except Exception:
            resolved_reference = reference_text
        target_index = 0
        for idx, option in enumerate(self._workspace_reference_options, start=1):
            try:
                option_key = str(Path(option["path"]).expanduser().resolve())
            except Exception:
                option_key = str(option["path"])
            if resolved_reference and option_key == resolved_reference:
                target_index = idx
                break
        self._reference_workspace_combo_syncing = True
        self.reference_workspace_combo.setCurrentIndex(target_index)
        self._reference_workspace_combo_syncing = False

    def _on_reference_matrix_path_changed(self, text) -> None:
        self._sync_reference_workspace_combo(text)
        current_key = self._selected_reference_matrix_key()
        self._populate_reference_matrix_key_combo(text, preferred_key=current_key)
        self._update_run_state()

    def _on_reference_workspace_changed(self, _index) -> None:
        if self._reference_workspace_combo_syncing or not hasattr(self, "reference_workspace_combo"):
            return
        selected_path = str(self.reference_workspace_combo.currentData() or "").strip()
        if not selected_path:
            return
        self.reference_matrix_edit.setText(selected_path)

    def _reference_matrix_key_label(self, key: str, shape: tuple[int, ...]) -> str:
        shape_text = " x ".join(str(int(dim)) for dim in shape) if shape else "unknown"
        return f"{key} ({shape_text})"

    def _load_reference_matrix_key_options(self, reference_path: str | None):
        reference_text = str(reference_path or "").strip()
        if not reference_text:
            return [], None
        try:
            path = Path(reference_text).expanduser()
        except Exception:
            return [], None
        if not path.is_file():
            return [], None
        try:
            with np.load(path, allow_pickle=True) as data:
                options = []
                for key in data.files:
                    try:
                        arr = np.asarray(data[key])
                    except Exception:
                        continue
                    shape = tuple(int(dim) for dim in getattr(arr, "shape", ()))
                    priority = None
                    if arr.ndim == 3 and len(shape) == 3 and shape[1] == shape[2]:
                        if str(key) == "matrix_subj_list":
                            priority = 0
                        elif shape[0] == 1:
                            priority = 2
                    elif arr.ndim == 2 and len(shape) == 2 and shape[0] == shape[1]:
                        priority = 1
                    if priority is None:
                        continue
                    options.append(
                        {
                            "key": str(key),
                            "label": self._reference_matrix_key_label(str(key), shape),
                            "priority": int(priority),
                        }
                    )
        except Exception as exc:
            return [], str(exc)
        options.sort(key=lambda item: (int(item["priority"]), str(item["key"]).lower()))
        return [{"key": item["key"], "label": item["label"]} for item in options], None

    def _selected_reference_matrix_key(self) -> str:
        if not hasattr(self, "reference_matrix_key_combo"):
            return ""
        return str(self.reference_matrix_key_combo.currentData() or "").strip()

    def _populate_reference_matrix_key_combo(self, reference_path: str | None, preferred_key: str | None = None) -> None:
        if not hasattr(self, "reference_matrix_key_combo"):
            return
        options, error_text = self._load_reference_matrix_key_options(reference_path)
        self._reference_matrix_key_options = list(options)
        self._reference_matrix_key_combo_syncing = True
        self.reference_matrix_key_combo.blockSignals(True)
        self.reference_matrix_key_combo.clear()
        enabled = False
        if options:
            for option in options:
                self.reference_matrix_key_combo.addItem(str(option["label"]), str(option["key"]))
            preferred = str(preferred_key or "").strip()
            if not preferred:
                preferred = "matrix_subj_list" if any(opt["key"] == "matrix_subj_list" for opt in options) else str(options[0]["key"])
            target_index = 0
            for idx, option in enumerate(options):
                if str(option["key"]) == preferred:
                    target_index = idx
                    break
            self.reference_matrix_key_combo.setCurrentIndex(target_index)
            enabled = not self._results_is_running()
        else:
            reference_text = str(reference_path or "").strip()
            if error_text:
                placeholder = f"Could not read NPZ keys: {error_text}"
            elif reference_text:
                placeholder = "No square matrix keys found in NPZ"
            else:
                placeholder = "Select reference matrix first"
            self.reference_matrix_key_combo.addItem(placeholder, "")
            self.reference_matrix_key_combo.setCurrentIndex(0)
        self.reference_matrix_key_combo.setEnabled(enabled)
        self.reference_matrix_key_combo.blockSignals(False)
        self._reference_matrix_key_combo_syncing = False

    def _on_reference_matrix_key_changed(self, _index) -> None:
        if self._reference_matrix_key_combo_syncing:
            return
        self._update_run_state()

    def _is_results_output_dir_auto_managed(self) -> bool:
        if not hasattr(self, "results_output_dir_edit"):
            return False
        current = self.results_output_dir_edit.text().strip()
        auto_value = str(self._results_output_dir_auto_value or "").strip()
        return (not current) or (auto_value and current == auto_value)

    def _refresh_default_results_output_dir(self, force=False) -> None:
        if not hasattr(self, "results_output_dir_edit"):
            return
        if not force and not self._is_results_output_dir_auto_managed():
            return
        default_output = self._default_results_output_dir()
        self.results_output_dir_edit.setText(default_output)
        self._results_output_dir_auto_value = default_output

    def _is_output_dir_auto_managed(self) -> bool:
        if not hasattr(self, "output_dir_edit"):
            return False
        current = self.output_dir_edit.text().strip()
        auto_value = str(self._output_dir_auto_value or "").strip()
        return (not current) or (auto_value and current == auto_value)

    def _refresh_default_output_dir(self, *_args, force=False) -> None:
        if not hasattr(self, "output_dir_edit"):
            return
        if not force and not self._is_output_dir_auto_managed():
            return
        modality = self.modality_combo.currentText().strip().lower() if hasattr(self, "modality_combo") else ""
        default_output = self._default_output_dir_for_context(modality)
        self.output_dir_edit.setText(default_output)
        self._output_dir_auto_value = default_output

    def _build_ui(self):
        root_layout = QVBoxLayout(self)

        self.workflow_shell = _WorkflowShell(self.STEP_TITLES, on_step_selected=self._go_to_step)
        self._step_buttons = self.workflow_shell.step_buttons
        self.step_stack = self.workflow_shell.step_stack
        self.workflow_shell.add_step(self._build_step_data())
        self.workflow_shell.add_step(self._build_step_model())
        self.workflow_shell.add_step(self._build_step_nbs_settings())
        self.workflow_shell.add_step(self._build_step_run())
        self.workflow_shell.add_step(self._build_step_results())

        self.log_toggle_button = QPushButton("Show log ▾")
        self.log_toggle_button.clicked.connect(self._toggle_log_drawer)
        self.log_toggle_button.setMaximumWidth(140)
        self.workflow_shell.right_layout.addWidget(self.log_toggle_button, 0)

        self.log_drawer = QFrame()
        log_layout = QVBoxLayout(self.log_drawer)
        log_layout.setContentsMargins(0, 0, 0, 0)
        self.terminal_output = QPlainTextEdit()
        self.terminal_output.setObjectName("nbsTerminal")
        self.terminal_output.setReadOnly(True)
        self.terminal_output.document().setMaximumBlockCount(4000)
        if QT_LIB == 6:
            self.terminal_output.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        else:
            self.terminal_output.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.terminal_output.setMinimumHeight(180)
        log_layout.addWidget(self.terminal_output)
        self.workflow_shell.right_layout.addWidget(self.log_drawer, 0)
        self._apply_terminal_style()
        self._set_log_drawer_expanded(False)

        root_layout.addWidget(self.workflow_shell, 1)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(False)
        root_layout.addWidget(self.status_label)

        actions = QHBoxLayout()
        actions.addStretch(1)
        self.back_button = QPushButton("Back")
        self.back_button.clicked.connect(self._go_prev_step)
        actions.addWidget(self.back_button)
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self._go_next_step)
        actions.addWidget(self.next_button)
        self.run_button = QPushButton("Run NBS")
        self.run_button.setMinimumWidth(130)
        self.run_button.clicked.connect(self._run_configuration)
        actions.addWidget(self.run_button)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        actions.addWidget(close_button)
        root_layout.addLayout(actions)

        if len(self._step_buttons) > self.RESULTS_STEP_INDEX:
            self._step_buttons[self.RESULTS_STEP_INDEX].setEnabled(False)
        self._go_to_step(0)
        self._update_dataset_summary()

    def _build_step_data(self):
        self.data_page = _CovariateFilterPage(self._columns, exclude_mode="toggle_button")
        self.dataset_summary_label = self.data_page.dataset_summary_label
        self.filter_covar_combo = self.data_page.filter_covar_combo
        self.filter_value_edit = self.data_page.filter_value_edit
        self.filter_button = self.data_page.filter_button
        self.filter_reset_button = self.data_page.filter_reset_button
        self.table = self.data_page.table
        self.showing_rows_label = self.data_page.showing_rows_label
        self.data_page.set_dataset_summary(
            self._source_path.name,
            self._matrix_key,
            len(self._rows),
            len(self._columns),
        )
        self.data_page.set_filter_callbacks(self._apply_filter, self._reset_filter)
        self.table.setSortingEnabled(False)
        return self.data_page

    def _build_step_model(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        self.model_table = QTableWidget()
        self.model_table.setColumnCount(5)
        self.model_table.setHorizontalHeaderLabels(["Include", "Name", "Type", "Confound", "Regressor"])
        self.model_table.setRowCount(len(self._columns))
        if QT_LIB == 6:
            self.model_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            self.model_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            self.model_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
            self.model_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
            self.model_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        else:
            self.model_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
            self.model_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
            self.model_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
            self.model_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
            self.model_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.model_table.setAlternatingRowColors(True)
        self.model_table.setSelectionBehavior(QAbstractItemView.SelectRows if QT_LIB == 5 else QAbstractItemView.SelectionBehavior.SelectRows)
        self.model_table.setSelectionMode(QAbstractItemView.SingleSelection if QT_LIB == 5 else QAbstractItemView.SelectionMode.SingleSelection)

        self._covar_checks = {}
        self._confound_buttons = {}
        self._regressor_buttons = {}
        self._model_updating = True
        for row_idx, covar_name in enumerate(self._columns):
            include_button = _make_toggle_button()
            include_button.clicked.connect(
                lambda _checked=False, name=covar_name: self._on_include_model_toggled(name)
            )
            self.model_table.setCellWidget(row_idx, 0, include_button)

            name_item = QTableWidgetItem(covar_name)
            name_item.setFlags(name_item.flags() & ~_is_editable_flag())
            self.model_table.setItem(row_idx, 1, name_item)

            type_item = QTableWidgetItem(self._covariate_type_label(covar_name))
            type_item.setFlags(type_item.flags() & ~_is_editable_flag())
            self.model_table.setItem(row_idx, 2, type_item)

            confound_button = _make_toggle_button()
            confound_button.clicked.connect(
                lambda _checked=False, name=covar_name: self._on_role_button_clicked(name, "Confound")
            )
            self.model_table.setCellWidget(row_idx, 3, confound_button)

            regressor_button = _make_toggle_button()
            regressor_button.clicked.connect(
                lambda _checked=False, name=covar_name: self._on_role_button_clicked(name, "Regressor")
            )
            self.model_table.setCellWidget(row_idx, 4, regressor_button)

            self._covar_checks[covar_name] = include_button
            self._confound_buttons[covar_name] = confound_button
            self._regressor_buttons[covar_name] = regressor_button
            self._sync_model_row_buttons(covar_name, emit_update=False)
        self._model_updating = False
        layout.addWidget(self.model_table, 1)

        model_controls = QHBoxLayout()
        model_controls.addWidget(QLabel("Regressor type"))
        self.model_regressor_type_combo = QComboBox()
        self.model_regressor_type_combo.addItems(self.REGRESSOR_TYPE_MODEL_OPTIONS)
        self.model_regressor_type_combo.setCurrentText("Auto")
        self.model_regressor_type_combo.currentTextChanged.connect(self._update_run_summary)
        self.model_regressor_type_combo.currentTextChanged.connect(self._update_run_state)
        model_controls.addWidget(self.model_regressor_type_combo)
        self.covariate_balance_button = QPushButton("Covariate Balance")
        self.covariate_balance_button.clicked.connect(self._run_covariate_balance_analysis)
        model_controls.addWidget(self.covariate_balance_button)
        model_controls.addStretch(1)
        layout.addLayout(model_controls)

        custom_contrast_frame = QFrame()
        custom_layout = QHBoxLayout(custom_contrast_frame)
        custom_layout.setContentsMargins(0, 0, 0, 0)
        custom_layout.addWidget(QLabel("Custom contrast"))
        self.contrast_edit = QLineEdit("")
        self.contrast_edit.setPlaceholderText("e.g. 0 0 1 1 1")
        self.contrast_edit.setToolTip(
            "Tail is determined by the sign of the regressor's entry, not a separate setting:\n"
            "  ...1   -> right-tailed, one run, p-value uncorrected\n"
            "  ...-1  -> left-tailed, one run, p-value uncorrected\n"
            "  ...b   -> two-tailed: runs +1 and -1 and Bonferroni-doubles each p-value (capped at 1)\n"
            "Run the right- and left-tailed contrasts separately if you want to apply your own "
            "correction for multiple comparisons instead of the built-in two-tailed doubling."
        )
        custom_layout.addWidget(self.contrast_edit, 1)
        layout.addWidget(custom_contrast_frame)
        self.contrast_edit.textChanged.connect(self._update_run_summary)
        return page

    def _build_step_nbs_settings(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        core_group = QGroupBox("Core settings")
        core_grid = QGridLayout(core_group)
        core_grid.addWidget(QLabel("Test type"), 0, 0)
        self.test_combo = QComboBox()
        self.test_combo.addItems(self.TEST_OPTIONS)
        self.test_combo.currentTextChanged.connect(self._on_test_type_changed)
        core_grid.addWidget(self.test_combo, 0, 1)

        core_grid.addWidget(QLabel("Primary threshold (T)"), 0, 2)
        self.t_thresh_spin = QDoubleSpinBox()
        self.t_thresh_spin.setRange(0.1, 100.0)
        self.t_thresh_spin.setDecimals(3)
        self.t_thresh_spin.setSingleStep(0.1)
        self.t_thresh_spin.setValue(3.5)
        core_grid.addWidget(self.t_thresh_spin, 0, 3)

        core_grid.addWidget(QLabel("Permutations"), 1, 0)
        self.nperm_spin = QSpinBox()
        self.nperm_spin.setRange(100, 500000)
        self.nperm_spin.setSingleStep(500)
        self.nperm_spin.setValue(5000)
        core_grid.addWidget(self.nperm_spin, 1, 1)

        max_threads = max(1, int(os.cpu_count() or 1))
        core_grid.addWidget(QLabel(f"Threads (max {max_threads})"), 2, 0)
        self.nthreads_spin = QSpinBox()
        self.nthreads_spin.setRange(1, max_threads)
        self.nthreads_spin.setValue(min(28, max_threads))
        self.nthreads_spin.setToolTip(f"Available CPU threads: {max_threads}")
        core_grid.addWidget(self.nthreads_spin, 2, 1)

        core_grid.addWidget(QLabel("Modality"), 2, 2)
        self.modality_combo = QComboBox()
        self.modality_combo.addItems(list(self.MODALITY_OPTIONS))
        source_modality = (self._source_modality or "").strip().lower()
        if source_modality:
            if self.modality_combo.findText(source_modality) < 0:
                self.modality_combo.addItem(source_modality)
            self.modality_combo.setCurrentText(source_modality)
        else:
            self.modality_combo.setCurrentText("mrsi")
        self.modality_combo.currentTextChanged.connect(self._refresh_default_output_dir)
        self.modality_combo.currentTextChanged.connect(self._update_run_summary)
        core_grid.addWidget(self.modality_combo, 2, 3)

        core_grid.addWidget(QLabel("Engine"), 3, 0)
        self.engine_combo = QComboBox()
        self.engine_combo.addItem("Python (MATLAB-compatible)", "python")
        self.engine_combo.addItem("MATLAB (reference, requires MATLAB)", "matlab")
        self.engine_combo.setCurrentIndex(0)
        self.engine_combo.currentIndexChanged.connect(self._on_engine_changed)
        self.engine_combo.currentIndexChanged.connect(self._update_run_summary)
        core_grid.addWidget(self.engine_combo, 3, 1)

        core_grid.addWidget(QLabel("Component size"), 3, 2)
        self.component_size_combo = QComboBox()
        self.component_size_combo.addItem("Extent", "extent")
        self.component_size_combo.addItem("Intensity", "intensity")
        self.component_size_combo.setCurrentIndex(0)
        self.component_size_combo.currentIndexChanged.connect(self._update_run_summary)
        core_grid.addWidget(self.component_size_combo, 3, 3)

        core_grid.addWidget(QLabel("Seed"), 4, 0)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2_147_483_647)
        self.seed_spin.setSpecialValueText("random")
        self.seed_spin.setValue(0)
        self.seed_spin.setToolTip(
            "Seed for the permutation RNG (python engine). Leave at 'random' for a fresh "
            "null distribution on every run; set a value to make permutations reproducible."
        )
        self.seed_spin.valueChanged.connect(self._update_run_summary)
        core_grid.addWidget(self.seed_spin, 4, 1)

        self.export_matlab_check = QCheckBox("Also export MATLAB NBS1.2 inputs")
        self.export_matlab_check.setChecked(False)
        self.export_matlab_check.setToolTip(
            "Write matrices/, designMatrix.txt, COG.txt and nodeLabels.txt under matlab_nbs/ "
            "so the same analysis can be reproduced or cross-checked in MATLAB NBS1.2. Not "
            "needed to run the python engine."
        )
        core_grid.addWidget(self.export_matlab_check, 4, 2, 1, 2)

        self.matlab_persistent_check = QCheckBox(
            "Persistent MATLAB session (reuse workers between runs)"
        )
        self.matlab_persistent_check.setChecked(True)
        core_grid.addWidget(self.matlab_persistent_check, 5, 0, 1, 4)

        self.matlab_no_precompute_check = QCheckBox(
            "Skip MATLAB precompute (lower memory, often slower)"
        )
        self.matlab_no_precompute_check.setChecked(False)
        core_grid.addWidget(self.matlab_no_precompute_check, 6, 0, 1, 4)
        self._on_engine_changed()

        layout.addWidget(core_group)
        layout.addStretch(1)
        return page

    def _build_step_run(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        summary_group = QGroupBox("Run summary")
        summary_layout = QVBoxLayout(summary_group)
        self.run_summary_label = QLabel("")
        self.run_summary_label.setWordWrap(True)
        summary_layout.addWidget(self.run_summary_label)
        layout.addWidget(summary_group)

        output_group = QGroupBox("Output controls")
        output_grid = QGridLayout(output_group)
        output_grid.addWidget(QLabel("Parcellation"), 0, 0)
        self.parcellation_path_edit = QLineEdit("")
        self.parcellation_path_edit.setPlaceholderText("Optional override (.nii/.nii.gz)")
        self.parcellation_path_edit.setReadOnly(True)
        output_grid.addWidget(self.parcellation_path_edit, 0, 1, 1, 2)
        self.parcellation_browse_button = QPushButton("Browse")
        self.parcellation_browse_button.clicked.connect(self._browse_parcellation_path)
        output_grid.addWidget(self.parcellation_browse_button, 0, 3)

        output_grid.addWidget(QLabel("Output folder"), 1, 0)
        initial_output_dir = self._default_output_dir_for_context(
            self.modality_combo.currentText().strip().lower()
            if hasattr(self, "modality_combo")
            else ""
        )
        self._output_dir_auto_value = initial_output_dir
        self.output_dir_edit = QLineEdit(initial_output_dir)
        output_grid.addWidget(self.output_dir_edit, 1, 1, 1, 2)
        self.output_dir_button = QPushButton("Browse")
        self.output_dir_button.clicked.connect(self._browse_output_dir)
        output_grid.addWidget(self.output_dir_button, 1, 3)

        output_grid.addWidget(QLabel("Naming prefix"), 2, 0)
        self.output_prefix_edit = QLineEdit("")
        self.output_prefix_edit.setPlaceholderText("Optional prefix for copied result .npz")
        output_grid.addWidget(self.output_prefix_edit, 2, 1, 1, 3)

        self.display_results_check = QCheckBox("Display results")
        self.display_results_check.setChecked(True)
        output_grid.addWidget(self.display_results_check, 3, 0, 1, 2)
        self.export_on_finish_check = QCheckBox("Export results on finish")
        self.export_on_finish_check.setChecked(False)
        output_grid.addWidget(self.export_on_finish_check, 3, 2, 1, 2)
        self.display_collapse_check = QCheckBox("Collapse parcels")
        self.display_collapse_check.setChecked(False)
        output_grid.addWidget(self.display_collapse_check, 4, 0, 1, 2)
        layout.addWidget(output_group)

        run_group = QGroupBox("Run controls")
        run_layout = QVBoxLayout(run_group)
        self.run_progress_label = QLabel("Idle")
        run_layout.addWidget(self.run_progress_label)
        self.run_progress = QProgressBar()
        self.run_progress.setRange(0, 1)
        self.run_progress.setValue(0)
        run_layout.addWidget(self.run_progress)
        run_buttons = QHBoxLayout()
        self.save_button = QPushButton("Save Setup")
        self.save_button.clicked.connect(self._save_configuration)
        run_buttons.addWidget(self.save_button)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._cancel_run)
        run_buttons.addWidget(self.stop_button)
        self.export_button = QPushButton("Export Results")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._export_results)
        run_buttons.addWidget(self.export_button)
        run_buttons.addStretch(1)
        run_layout.addLayout(run_buttons)
        layout.addWidget(run_group)
        layout.addStretch(1)
        return page

    def _build_step_results(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        info_group = QGroupBox("Structural context results")
        info_grid = QGridLayout(info_group)

        info_grid.addWidget(QLabel("NBS result bundle"), 0, 0)
        self.results_input_nbs_edit = QLineEdit("")
        self.results_input_nbs_edit.setReadOnly(True)
        self.results_input_nbs_edit.setPlaceholderText("Available after a successful NBS run")
        info_grid.addWidget(self.results_input_nbs_edit, 0, 1, 1, 3)

        info_grid.addWidget(QLabel("Workspace reference"), 1, 0)
        self.reference_workspace_combo = QComboBox()
        self.reference_workspace_combo.currentIndexChanged.connect(self._on_reference_workspace_changed)
        self.set_workspace_reference_options(self._workspace_reference_options)
        info_grid.addWidget(self.reference_workspace_combo, 1, 1, 1, 3)

        info_grid.addWidget(QLabel("Reference matrix"), 2, 0)
        self.reference_matrix_edit = QLineEdit("")
        self.reference_matrix_edit.setPlaceholderText("Select structural reference NPZ")
        self.reference_matrix_edit.textChanged.connect(self._on_reference_matrix_path_changed)
        info_grid.addWidget(self.reference_matrix_edit, 2, 1, 1, 2)
        self.reference_matrix_button = QPushButton("Browse")
        self.reference_matrix_button.clicked.connect(self._browse_reference_matrix)
        info_grid.addWidget(self.reference_matrix_button, 2, 3)

        info_grid.addWidget(QLabel("Reference matrix key"), 3, 0)
        self.reference_matrix_key_combo = QComboBox()
        self.reference_matrix_key_combo.currentIndexChanged.connect(self._on_reference_matrix_key_changed)
        info_grid.addWidget(self.reference_matrix_key_combo, 3, 1, 1, 3)
        self._populate_reference_matrix_key_combo(self.reference_matrix_edit.text())

        info_grid.addWidget(QLabel("Output folder"), 4, 0)
        initial_results_output = self._default_results_output_dir()
        self._results_output_dir_auto_value = initial_results_output
        self.results_output_dir_edit = QLineEdit(initial_results_output)
        self.results_output_dir_edit.textChanged.connect(self._update_run_state)
        info_grid.addWidget(self.results_output_dir_edit, 4, 1, 1, 2)
        self.results_output_dir_button = QPushButton("Browse")
        self.results_output_dir_button.clicked.connect(self._browse_results_output_dir)
        info_grid.addWidget(self.results_output_dir_button, 4, 3)

        self.results_status_label = QLabel("Run NBS first to enable this step.")
        self.results_status_label.setWordWrap(True)
        info_grid.addWidget(self.results_status_label, 5, 0, 1, 4)

        self.results_run_button = QPushButton("Generate Structural Context")
        self.results_run_button.clicked.connect(self._run_results_analysis)
        self.results_run_button.setEnabled(False)
        info_grid.addWidget(self.results_run_button, 6, 0, 1, 4)

        self._sync_reference_workspace_combo(self.reference_matrix_edit.text())
        layout.addWidget(info_group)
        layout.addStretch(1)
        return page

    def _results_script_path(self):
        return Path(__file__).resolve().parents[1] / "scripts" / "plot_nbs_control_structural_context.py"

    def _results_is_running(self):
        return (
            self._results_process is not None
            and self._results_process.state() != _qprocess_not_running()
        )

    def _results_ready(self):
        result_path = str(self._last_result_npz_path or "").strip()
        return bool(result_path) and Path(result_path).is_file()

    def _refresh_results_step(self, force_defaults=False):
        result_path = str(self._last_result_npz_path or "").strip()
        if hasattr(self, "results_input_nbs_edit"):
            self.results_input_nbs_edit.setText(result_path)
        if hasattr(self, "results_status_label"):
            if result_path and Path(result_path).is_file():
                self.results_status_label.setText(
                    "Use the generated NBS components_all bundle as input, select a structural reference NPZ, then choose which matrix key to load from that file."
                )
            else:
                self.results_status_label.setText("Run NBS first to enable this step.")
        self._refresh_default_results_output_dir(force=force_defaults)

    def _set_results_busy(self, busy):
        if hasattr(self, "results_run_button"):
            self.results_run_button.setText("Running..." if busy else "Generate Structural Context")
        if hasattr(self, "reference_matrix_button"):
            self.reference_matrix_button.setEnabled(not busy)
        if hasattr(self, "reference_workspace_combo"):
            self.reference_workspace_combo.setEnabled((not busy) and bool(self._workspace_reference_options))
        if hasattr(self, "results_output_dir_button"):
            self.results_output_dir_button.setEnabled(not busy)
        if hasattr(self, "reference_matrix_edit"):
            self.reference_matrix_edit.setEnabled(not busy)
        if hasattr(self, "reference_matrix_key_combo"):
            self.reference_matrix_key_combo.setEnabled((not busy) and bool(self._reference_matrix_key_options))
        if hasattr(self, "results_output_dir_edit"):
            self.results_output_dir_edit.setEnabled(not busy)
        self._update_run_state()

    def _browse_reference_matrix(self):
        current = self.reference_matrix_edit.text().strip() if hasattr(self, "reference_matrix_edit") else ""
        if current:
            start_dir = str(Path(current).expanduser().parent)
        else:
            start_dir = self._default_reference_matrix_dir()
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select reference matrix",
            start_dir,
            "NPZ (*.npz);;All files (*)",
        )
        if selected:
            self.reference_matrix_edit.setText(selected)

    def _browse_results_output_dir(self):
        start_dir = self.results_output_dir_edit.text().strip() if hasattr(self, "results_output_dir_edit") else ""
        start_dir = start_dir or self._default_results_output_dir()
        selected = QFileDialog.getExistingDirectory(self, "Select results output folder", start_dir)
        if selected:
            self.results_output_dir_edit.setText(selected)
            self._results_output_dir_auto_value = ""

    def _go_to_step(self, step_index):
        step = max(0, min(int(step_index), len(self.STEP_TITLES) - 1))
        if step > 0 and step < len(self._step_buttons) and not self._step_buttons[step].isEnabled():
            step = self._current_step if 0 <= self._current_step < len(self.STEP_TITLES) else 0
        step = self.workflow_shell.set_current_step(step)
        self._current_step = step
        if step == self.RUN_STEP_INDEX:
            self._update_run_summary()
            if not self._log_expanded:
                self._set_log_drawer_expanded(True)
                self._log_auto_expanded = True
        elif step == self.RESULTS_STEP_INDEX:
            self._refresh_results_step(force_defaults=False)
        elif self._log_auto_expanded:
            self._set_log_drawer_expanded(False)
            self._log_auto_expanded = False
        self._update_step_navigation()

    def _go_next_step(self):
        self._go_to_step(self._current_step + 1)

    def _go_prev_step(self):
        self._go_to_step(self._current_step - 1)

    def _update_step_navigation(self):
        max_step = len(self.STEP_TITLES) - 1
        next_enabled = (
            self._current_step < max_step
            and (self._current_step + 1) < len(self._step_buttons)
            and self._step_buttons[self._current_step + 1].isEnabled()
        )
        self.back_button.setEnabled(self._current_step > 0)
        self.next_button.setVisible(self._current_step < max_step)
        self.next_button.setEnabled(next_enabled and not (self._process_is_running() or self._export_is_running() or self._covariate_balance_is_running() or self._results_is_running()))
        self.run_button.setVisible(self._current_step == self.RUN_STEP_INDEX)
        self.run_button.setText("Run NBS")
        self._update_run_state()

    def _set_log_drawer_expanded(self, expanded):
        self._log_expanded = bool(expanded)
        self.log_drawer.setVisible(self._log_expanded)
        self.log_toggle_button.setText("Hide log ▴" if self._log_expanded else "Show log ▾")

    def _toggle_log_drawer(self):
        self._log_auto_expanded = False
        self._set_log_drawer_expanded(not self._log_expanded)

    def _update_dataset_summary(self):
        if not hasattr(self, "dataset_summary_label"):
            return
        summary = (
            f"{self._source_path.name} | key: {self._matrix_key} | "
            f"rows: {len(self._rows)} | covariates: {len(self._columns)}"
        )
        self.dataset_summary_label.setText(summary)

    def _covariate_type_label(self, covar_name):
        lower = str(covar_name).strip().lower()
        if lower in {"participant_id", "session_id", "subject_id", "id", "sub", "ses"}:
            return "ID"
        values = [row.get(covar_name) for row in self._rows]
        if _column_is_numeric(values):
            return "numeric"
        return "categorical"

    def _on_exclude_row_toggled(self, source_idx, checked):
        if self._data_table_refreshing:
            return
        source_idx = int(source_idx)
        if checked:
            self._excluded_indices.add(source_idx)
        else:
            self._excluded_indices.discard(source_idx)
        self._update_showing_rows_label()
        self._update_test_options()
        self._update_run_state()
        self._update_run_summary()

    def _on_model_table_item_changed(self, item):
        _ = item
        return

    def _on_test_type_changed(self, *_args):
        self._update_run_summary()

    def _browse_output_dir(self):
        start_dir = self.output_dir_edit.text().strip() or self._default_output_dir_for_context()
        selected = QFileDialog.getExistingDirectory(self, "Select output folder", start_dir)
        if selected:
            self.output_dir_edit.setText(selected)
            self._output_dir_auto_value = ""

    def _selected_engine(self):
        if not hasattr(self, "engine_combo"):
            return "matlab"
        data = self.engine_combo.currentData()
        if not data:
            return "matlab"
        return str(data).strip().lower()

    def _on_engine_changed(self, *_args):
        is_matlab = self._selected_engine() == "matlab"
        if hasattr(self, "matlab_persistent_check"):
            self.matlab_persistent_check.setEnabled(is_matlab)
        if hasattr(self, "matlab_no_precompute_check"):
            self.matlab_no_precompute_check.setEnabled(is_matlab)
        if hasattr(self, "export_matlab_check"):
            if is_matlab:
                # MATLAB engine always needs its own export; reflect that rather than
                # letting the user uncheck something the run requires.
                self.export_matlab_check.setChecked(True)
            self.export_matlab_check.setEnabled(not is_matlab)
        if hasattr(self, "seed_spin"):
            self.seed_spin.setEnabled(not is_matlab)

    def _selected_regressor_type(self):
        mode = self.model_regressor_type_combo.currentText().strip().lower()
        if mode == "categorical":
            return "categorical"
        if mode == "continuous":
            return "continuous"
        regressor = self._regressor_name()
        if not regressor:
            return "categorical"
        values = [_display_text(self._rows[idx].get(regressor)).strip() for idx in self.selected_row_indices()]
        values = [v for v in values if v != ""]
        if not values:
            return "categorical"
        if not _column_is_numeric(values):
            return "categorical"
        unique_values = {str(v) for v in values}
        return "continuous" if len(unique_values) > 8 else "categorical"

    def _selected_export_regressor_type(self):
        return self._selected_regressor_type()

    def _update_run_summary(self):
        if not hasattr(self, "run_summary_label"):
            return
        selected = self.selected_covariates()
        regressor = selected.get("regressor") or "none"
        confounds = ", ".join(selected.get("nuisance") or []) or "none"
        n_included = len(self.selected_row_indices())
        threshold = (
            f"{float(self.t_thresh_spin.value()):g}"
            if hasattr(self, "t_thresh_spin")
            else "NA"
        )
        nperm = (
            f"{int(self.nperm_spin.value())}"
            if hasattr(self, "nperm_spin")
            else "NA"
        )
        engine = self._selected_engine()
        engine_label = "MATLAB (reference)" if engine == "matlab" else "Python (MATLAB-compatible)"
        test_label = self.test_combo.currentText().strip() if hasattr(self, "test_combo") else "t-test"
        size_label = (
            self.component_size_combo.currentText().strip()
            if hasattr(self, "component_size_combo")
            else "Extent"
        )
        tail_label = _describe_contrast_tail(
            self.contrast_edit.text() if hasattr(self, "contrast_edit") else ""
        )
        seed_label = (
            "random"
            if not hasattr(self, "seed_spin") or self.seed_spin.value() == self.seed_spin.minimum()
            else str(int(self.seed_spin.value()))
        )
        self.run_summary_label.setText(
            f"Engine: {engine_label}\n"
            f"Regressor: {regressor}\n"
            f"Confounds: {confounds}\n"
            f"Test: {test_label}\n"
            f"Component size: {size_label}\n"
            f"Tail (from contrast): {tail_label}\n"
            f"N included: {n_included}\n"
            f"Permutations: {nperm}\n"
            f"Seed: {seed_label}\n"
            f"Threshold: {threshold}"
        )

    def _update_showing_rows_label(self):
        if not hasattr(self, "showing_rows_label"):
            return
        included = len(self.selected_row_indices())
        self.showing_rows_label.setText(
            f"Showing {len(self._filtered_indices)}/{len(self._rows)} rows | Included: {included}"
        )

    def _set_status(self, text):
        self.status_label.setText(str(text))

    def _apply_terminal_style(self):
        if self.terminal_output is None:
            return
        self.terminal_output.setStyleSheet(
            "QPlainTextEdit#nbsTerminal {"
            " background-color: #000000;"
            " color: #b8f7c6;"
            " border: 1px solid #404040;"
            " border-radius: 4px;"
            " selection-background-color: #2d6cdf;"
            " font-family: 'DejaVu Sans Mono', 'Courier New', monospace;"
            " font-size: 10.5pt;"
            "}"
        )

    def _append_terminal_line(self, text):
        if self.terminal_output is None:
            return
        if text is None:
            return
        lines = str(text).splitlines() or [str(text)]
        for line in lines:
            if line.strip() == "":
                continue
            self.terminal_output.appendPlainText(line)
        scroll = self.terminal_output.verticalScrollBar()
        if scroll is not None:
            scroll.setValue(scroll.maximum())

    def _load_source_metadata(self):
        self._source_group = None
        self._source_modality = None
        try:
            with np.load(self._source_path, allow_pickle=True) as npz:
                if "group" in npz:
                    group = _display_text(npz["group"]).strip()
                    self._source_group = group or None
                if "modality" in npz:
                    modality = _display_text(npz["modality"]).strip().lower()
                    self._source_modality = modality or None
        except Exception:
            self._source_group = None
            self._source_modality = None

    def _browse_parcellation_path(self):
        start_dir = self._atlas_dir_default or (
            str(self._source_path.parent) if self._source_path is not None else str(Path.cwd())
        )
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select parcellation NIfTI",
            start_dir,
            "NIfTI (*.nii *.nii.gz);;All files (*)",
        )
        if not selected:
            return
        self.parcellation_path_edit.setText(selected)

    def set_theme(self, theme_name):
        toggle_style = (
            "QPushButton#tableExcludeButton { min-width: 24px; max-width: 24px; min-height: 24px; max-height: 24px; border-radius: 12px; } "
            "QPushButton#tableExcludeButton:hover { border: 1px solid #64748b; background: rgba(148, 163, 184, 0.12); } "
            "QPushButton#tableExcludeButton:checked { background: #dc2626; color: transparent; border: 1px solid #fca5a5; } "
            "QPushButton#tableExcludeButton:disabled { background: transparent; color: transparent; border: 1px solid #cbd5e1; } "
            "QPushButton#tableExcludeButton:checked:disabled { background: #fca5a5; color: transparent; border: 1px solid #fca5a5; } "
        )
        _theme, style = _workflow_dialog_stylesheet(
            theme_name,
            control_selector="QPushButton, QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QTableWidget",
            extra_styles=toggle_style,
        )
        self.setStyleSheet(style)
        self._apply_terminal_style()

    def _process_is_running(self):
        return (
            self._run_process is not None
            and self._run_process.state() != _qprocess_not_running()
        )

    def _export_is_running(self):
        return (
            self._export_process is not None
            and self._export_process.state() != _qprocess_not_running()
        )

    def _covariate_balance_is_running(self):
        return (
            self._covariate_balance_process is not None
            and self._covariate_balance_process.state() != _qprocess_not_running()
        )

    def _set_run_busy(self, busy):
        self.run_button.setEnabled(not busy)
        if hasattr(self, "save_button"):
            self.save_button.setEnabled(not busy)
        if hasattr(self, "stop_button"):
            self.stop_button.setEnabled(busy)
        if hasattr(self, "back_button"):
            self.back_button.setEnabled(not busy and self._current_step > 0)
        if hasattr(self, "next_button"):
            self.next_button.setEnabled(not busy and self._current_step < (len(self.STEP_TITLES) - 1))
        self.run_button.setText("Running..." if busy else "Run")
        if hasattr(self, "run_progress"):
            if busy:
                self.run_progress.setRange(0, 0)
                self.run_progress_label.setText("Running...")
            else:
                self.run_progress.setRange(0, 1)
                self.run_progress.setValue(1 if self._last_result_npz_path else 0)
                if self.run_progress_label.text().strip() == "Running...":
                    self.run_progress_label.setText("Idle")
        if busy:
            self.export_button.setEnabled(False)
        elif self._last_result_npz_path and Path(self._last_result_npz_path).is_file():
            self.export_button.setEnabled(True)
        self._update_run_state()

    def _set_export_busy(self, busy):
        self.export_button.setText("Exporting..." if busy else "Export Results")
        if busy:
            self.export_button.setEnabled(False)
        elif not self._process_is_running() and self._last_result_npz_path and Path(self._last_result_npz_path).is_file():
            self.export_button.setEnabled(True)
        self._update_run_state()

    @staticmethod
    def _format_nbs_summary(summary):
        if not isinstance(summary, dict):
            return ""
        n_sig = int(summary.get("n_significant", 0))
        global_p = summary.get("global_pvalue", None)
        sig_pvals = summary.get("significant_pvalues", []) or []
        sig_indices = summary.get("significant_indices", []) or []
        if global_p is None:
            global_text = "NA"
        else:
            try:
                global_text = f"{float(global_p):.6g}"
            except Exception:
                global_text = str(global_p)
        if sig_pvals:
            pairs_text = ""
            if len(sig_indices) == len(sig_pvals):
                try:
                    pairs = [
                        f"C{int(idx)}={float(p):.6g}"
                        for idx, p in zip(sig_indices, sig_pvals)
                    ]
                    pairs_text = ", ".join(pairs)
                except Exception:
                    pairs_text = ""
            try:
                pvals_text = ", ".join(f"{float(p):.6g}" for p in sig_pvals)
            except Exception:
                pvals_text = ", ".join(str(p) for p in sig_pvals)
            if pairs_text:
                pvals_text = pairs_text
        else:
            pvals_text = "none"
        return (
            f"Significant components: {n_sig} | "
            f"p-values: {pvals_text} | "
            f"global p-value: {global_text}"
        )

    def _regressor_name(self):
        return self.selected_covariates().get("regressor")

    def _regressor_classes(self):
        regressor = self._regressor_name()
        if not regressor:
            return []
        classes = []
        for idx in self.selected_row_indices():
            value = _display_text(self._rows[idx].get(regressor)).strip()
            if value != "":
                classes.append(value)
        unique_classes = list(set(classes))
        numeric = _numeric_sortable(unique_classes)
        if numeric is not None:
            unique_classes.sort(key=lambda v: float(v))
        else:
            unique_classes.sort()
        return unique_classes

    def _update_test_options(self):
        if not hasattr(self, "test_combo"):
            return
        current = self.test_combo.currentText().strip()
        classes = self._regressor_classes()
        self.test_combo.blockSignals(True)
        self.test_combo.clear()
        self.test_combo.addItem("t-test")
        if len(classes) > 2:
            self.test_combo.addItem("Ftest")
        if current and self.test_combo.findText(current) >= 0:
            self.test_combo.setCurrentText(current)
        self.test_combo.blockSignals(False)
        self._on_test_type_changed()

    def _update_run_state(self):
        classes = self._regressor_classes()
        any_busy = (
            self._process_is_running()
            or self._export_is_running()
            or self._covariate_balance_is_running()
            or self._results_is_running()
        )
        can_run = len(classes) >= 2 and not any_busy
        if hasattr(self, "run_button"):
            self.run_button.setEnabled(can_run and self._current_step == self.RUN_STEP_INDEX)
        result_path = str(self._last_result_npz_path or "").strip()
        results_enabled = bool(result_path) and Path(result_path).is_file()
        if len(getattr(self, "_step_buttons", [])) > self.RESULTS_STEP_INDEX:
            self._step_buttons[self.RESULTS_STEP_INDEX].setEnabled(results_enabled and not any_busy)
        if hasattr(self, "results_run_button"):
            reference_path = self.reference_matrix_edit.text().strip() if hasattr(self, "reference_matrix_edit") else ""
            reference_key = self._selected_reference_matrix_key()
            output_dir = self.results_output_dir_edit.text().strip() if hasattr(self, "results_output_dir_edit") else ""
            self.results_run_button.setEnabled(
                results_enabled
                and bool(reference_path)
                and bool(reference_key)
                and bool(output_dir)
                and (not any_busy)
            )
        next_enabled = (
            self._current_step < (len(self.STEP_TITLES) - 1)
            and (self._current_step + 1) < len(getattr(self, "_step_buttons", []))
            and self._step_buttons[self._current_step + 1].isEnabled()
        )
        if hasattr(self, "back_button"):
            self.back_button.setEnabled((not any_busy) and self._current_step > 0)
        if hasattr(self, "next_button"):
            self.next_button.setVisible(self._current_step < (len(self.STEP_TITLES) - 1))
            self.next_button.setEnabled((not any_busy) and next_enabled)
        if hasattr(self, "covariate_balance_button"):
            selected = self.selected_covariates()
            regressor_type = self._selected_regressor_type()
            can_balance = (
                bool(selected.get("regressor"))
                and len(selected.get("nuisance") or []) >= 1
                and len(self.selected_row_indices()) >= 2
                and regressor_type != "continuous"
                and (not any_busy)
            )
            self.covariate_balance_button.setEnabled(can_balance)

    def _on_covar_toggled(self, covar_name, checked):
        _ = covar_name
        _ = checked
        self._update_test_options()
        self._update_run_state()
        self._update_run_summary()

    def _on_include_model_toggled(self, covar_name):
        self._sync_model_row_buttons(covar_name, emit_update=True)

    def _on_role_button_clicked(self, covar_name, role_value):
        if self._model_updating:
            return
        include_button = self._covar_checks.get(covar_name)
        confound_button = self._confound_buttons.get(covar_name)
        regressor_button = self._regressor_buttons.get(covar_name)
        if include_button is None or confound_button is None or regressor_button is None:
            return

        self._model_updating = True
        include_button.setChecked(True)
        if role_value == "Regressor":
            confound_button.setChecked(False)
            regressor_button.setChecked(True)
            for other_name, other_button in self._regressor_buttons.items():
                if other_name == covar_name:
                    continue
                other_button.setChecked(False)
            self._set_status(f"Regressor set to: {covar_name}")
        elif role_value == "Confound":
            confound_button.setChecked(True)
            regressor_button.setChecked(False)
            self._set_status("Confounds updated.")
        else:
            confound_button.setChecked(False)
            regressor_button.setChecked(False)
            self._set_status("Roles updated.")
        self._model_updating = False

        self._sync_model_row_buttons(covar_name, emit_update=False)

        self._update_test_options()
        self._update_run_state()
        self._update_run_summary()

    def _sync_model_row_buttons(self, covar_name, emit_update=True):
        include_button = self._covar_checks.get(covar_name)
        confound_button = self._confound_buttons.get(covar_name)
        regressor_button = self._regressor_buttons.get(covar_name)
        if include_button is None or confound_button is None or regressor_button is None:
            return

        include_checked = bool(include_button.isChecked())
        confound_button.setEnabled(include_checked)
        regressor_button.setEnabled(include_checked)
        if not include_checked:
            confound_button.setChecked(False)
            regressor_button.setChecked(False)
        elif confound_button.isChecked():
            regressor_button.setChecked(False)
        elif regressor_button.isChecked():
            confound_button.setChecked(False)

        if emit_update:
            self._update_test_options()
            self._update_run_state()
            self._update_run_summary()

    def _match_value(self, source_value, target_text, numeric):
        text = _display_text(source_value).strip()
        if numeric:
            try:
                return np.isclose(float(text), float(target_text))
            except Exception:
                return False
        return text == target_text

    @classmethod
    def _is_range_token(cls, token):
        return bool(cls.NUMERIC_RANGE_RE.match(str(token).strip()))

    def _apply_filter(self):
        if not self._columns:
            self._set_status("No covariates available to filter.")
            return

        matched, target_values, error = _filter_row_indices(
            self._rows,
            self.filter_covar_combo.currentText(),
            self.filter_value_edit.text(),
            self.NUMERIC_RANGE_RE,
        )
        if error:
            self._set_status(error)
            return
        self._filtered_indices = matched
        self._refresh_table()
        self._update_test_options()
        self._update_run_state()
        self._update_run_summary()
        self._set_status(
            f"Filter applied: {self.filter_covar_combo.currentText().strip()} in [{', '.join(target_values)}]. "
            f"Showing {len(self._filtered_indices)}/{len(self._rows)} rows."
        )

    def _reset_filter(self):
        self._filtered_indices = list(range(len(self._rows)))
        self._refresh_table()
        self._update_test_options()
        self._update_run_state()
        self._update_run_summary()
        self._set_status(f"Filter reset. Showing {len(self._filtered_indices)} rows.")

    def _refresh_table(self):
        self._data_table_refreshing = True
        self.data_page.populate_rows(
            self._filtered_indices,
            self._rows,
            self._excluded_indices,
            on_exclude_toggled=self._on_exclude_row_toggled,
            resize_to_contents=True,
        )
        self._data_table_refreshing = False
        self._update_showing_rows_label()
        if not self._columns:
            self._set_status("No covariate columns available in this file.")
        elif len(self._filtered_indices) == 0:
            self._set_status("No rows match the current filter.")
        elif not self.status_label.text():
            self._set_status(
                f"Showing {len(self._filtered_indices)}/{len(self._rows)} rows."
            )

    def _subject_session_pairs(self):
        lower_map = {name.lower(): name for name in self._columns}
        participant_col = lower_map.get("participant_id")
        session_col = lower_map.get("session_id")
        if participant_col is None or session_col is None:
            return []
        pairs = []
        for idx in self.selected_row_indices():
            row = self._rows[idx]
            pairs.append(
                {
                    "participant_id": _display_text(row.get(participant_col)),
                    "session_id": _display_text(row.get(session_col)),
                }
            )
        return pairs

    def current_configuration(self):
        selected = self.selected_covariates()
        return {
            "source_file": str(self._source_path),
            "matrix_key": self._matrix_key,
            "selected_rows": self.selected_row_indices(),
            "selected_pairs": self._subject_session_pairs(),
            "filter_covar": self.filter_covar_combo.currentText().strip(),
            "filter_values": _parse_filter_values(self.filter_value_edit.text()),
            "covariates": selected,
            "regressor_classes": self._regressor_classes(),
            "test": self.test_combo.currentText().strip(),
            "engine": self._selected_engine(),
            "component_size": (
                str(self.component_size_combo.currentData() or "extent").strip().lower()
                if hasattr(self, "component_size_combo")
                else "extent"
            ),
            "nthreads": int(self.nthreads_spin.value()),
            "nperm": int(self.nperm_spin.value()),
            "t_thresh": float(self.t_thresh_spin.value()),
            "matlab_cmd": self._matlab_cmd_default,
            "matlab_nbs_path": self._matlab_nbs_path_default,
            "parcellation_path": self.parcellation_path_edit.text().strip(),
            "contrast": self.contrast_edit.text().strip(),
            "contrast_mode": "custom",
            "model_regressor_type": self.model_regressor_type_combo.currentText().strip().lower(),
            "modality": self.modality_combo.currentText().strip().lower(),
            "matlab_persistent": bool(self.matlab_persistent_check.isChecked()),
            "matlab_no_precompute": bool(self.matlab_no_precompute_check.isChecked()),
            "seed": (
                None
                if not hasattr(self, "seed_spin") or self.seed_spin.value() == self.seed_spin.minimum()
                else int(self.seed_spin.value())
            ),
            "export_matlab": (
                bool(self.export_matlab_check.isChecked())
                if hasattr(self, "export_matlab_check")
                else False
            ),
            "display_no_show": (not bool(self.display_results_check.isChecked())),
            "display_collapse_parcels": bool(self.display_collapse_check.isChecked()),
            "display_regressor_type": self._selected_export_regressor_type(),
            "export_on_finish": bool(self.export_on_finish_check.isChecked()),
            "output_folder": self.output_dir_edit.text().strip(),
            "output_prefix": self.output_prefix_edit.text().strip(),
        }

    def _extract_subject_session(self, covars_subset):
        field_map = {name.lower(): name for name in (covars_subset.dtype.names or [])}
        sub_field = field_map.get("participant_id")
        ses_field = field_map.get("session_id")
        if sub_field is None or ses_field is None:
            raise ValueError("Covars must contain participant_id and session_id.")
        subject_ids = np.asarray(
            [_display_text(v) for v in covars_subset[sub_field]],
            dtype=object,
        )
        session_ids = np.asarray(
            [_display_text(v) for v in covars_subset[ses_field]],
            dtype=object,
        )
        return subject_ids, session_ids

    def _to_subject_stack(self, matrix_3d, axis, indices):
        if axis == 0:
            return matrix_3d[indices, :, :]
        if axis == 1:
            return matrix_3d[:, indices, :].transpose(1, 0, 2)
        return matrix_3d[:, :, indices].transpose(2, 0, 1)

    def _extract_parcel_metadata(self, npz):
        labels = None
        names = None
        for key in ("parcel_labels_group", "parcel_labels_group.npy"):
            if key in npz:
                labels = np.asarray(npz[key])
                break
        for key in ("parcel_names_group", "parcel_names_group.npy"):
            if key in npz:
                names = np.asarray(npz[key])
                break
        if labels is None or names is None:
            raise ValueError("Missing parcel_labels_group or parcel_names_group in source file.")
        return labels, names

    def _build_filtered_conn_subset(self):
        indices = np.asarray(self.selected_row_indices(), dtype=int)
        if indices.size == 0:
            raise ValueError("No rows selected for NBS run.")

        with np.load(self._source_path, allow_pickle=True) as npz:
            if self._matrix_key not in npz:
                raise ValueError(f"Matrix key '{self._matrix_key}' not found in source file.")
            raw_matrix = np.asarray(npz[self._matrix_key], dtype=float)
            if raw_matrix.ndim != 3:
                raise ValueError("NBS run requires a 3D matrix stack.")
            axis = _stack_axis(raw_matrix.shape)
            if axis is None:
                raise ValueError("Selected matrix is not a stack of square matrices.")
            stack_len = raw_matrix.shape[axis]
            if int(indices.max()) >= stack_len:
                raise ValueError("Filtered row index exceeds matrix stack size.")

            if "covars" not in npz:
                raise ValueError("Source file is missing covars.")
            covars_raw = np.asarray(npz["covars"])
            if covars_raw.shape[0] != stack_len:
                raise ValueError("Covars length does not match matrix stack size.")
            covars_subset = covars_raw[indices]
            subject_ids, session_ids = self._extract_subject_session(covars_subset)

            matrix_subj_list = self._to_subject_stack(raw_matrix, axis, indices)
            matrix_pop_avg = np.asarray(matrix_subj_list.mean(axis=0), dtype=float)
            labels, names = self._extract_parcel_metadata(npz)

            metabolites = np.asarray(npz["metabolites"]) if "metabolites" in npz else np.array([])
            if "metab_profiles_subj_list" in npz:
                metab_profiles = np.asarray(npz["metab_profiles_subj_list"])
                if metab_profiles.shape[0] == stack_len:
                    metab_profiles = metab_profiles[indices]
                else:
                    metab_profiles = np.zeros((indices.size, 1), dtype=float)
            else:
                metab_profiles = np.zeros((indices.size, 1), dtype=float)

            group = _display_text(npz["group"]) if "group" in npz else "group"
            modality_from_npz = _display_text(npz["modality"]).strip().lower() if "modality" in npz else ""
            selected_modality = self.modality_combo.currentText().strip().lower()
            modality = selected_modality or modality_from_npz or "mrsi"

        reg_name = self._regressor_name() or "regressor"
        # Path.stem only strips ".npz", so re-subsetting a source that is itself an
        # already-built "..._nbs_input.npz" (e.g. re-importing a prior result's
        # connectivity_path for verification) would otherwise double the suffix into
        # "..._nbs_input_key-..._nbs_input.npz".
        source_stem = self._source_path.stem
        if source_stem.endswith("_nbs_input"):
            source_stem = source_stem[: -len("_nbs_input")]
        subset_name = (
            f"{source_stem}_key-{_slugify_fragment(self._matrix_key)}"
            f"_reg-{_slugify_fragment(reg_name)}_n-{indices.size}_nbs_input.npz"
        )
        subset_path = self._source_path.with_name(subset_name)
        np.savez(
            subset_path,
            matrix_subj_list=np.asarray(matrix_subj_list, dtype=float),
            matrix_pop_avg=np.asarray(matrix_pop_avg, dtype=float),
            subject_id_list=subject_ids.astype(str),
            session_id_list=session_ids.astype(str),
            metabolites=metabolites,
            metab_profiles_subj_list=np.asarray(metab_profiles),
            parcel_labels_group=np.asarray(labels),
            parcel_names_group=np.asarray(names),
            covars=covars_subset,
            group=np.asarray(group),
            modality=np.asarray(modality),
            source_file=np.asarray(str(self._source_path)),
            source_key=np.asarray(self._matrix_key),
        )
        return subset_path

    def _build_run_command(self, conn_subset_path):
        regressor = self._regressor_name()
        if not regressor:
            raise ValueError("Select one Regressor before running.")

        selected = self.selected_covariates()
        nuisance_terms = selected.get("nuisance") or []
        test_choice = self.test_combo.currentText().strip()
        matlab_test = "F" if test_choice == "Ftest" else "t"
        engine = self._selected_engine()
        component_size = (
            str(self.component_size_combo.currentData() or "extent").strip().lower()
            if hasattr(self, "component_size_combo")
            else "extent"
        )

        run_script = Path(__file__).resolve().with_name("run_nbs.py")
        if not run_script.exists():
            raise FileNotFoundError(f"run_nbs.py not found at {run_script}")

        command = [
            str(run_script),
            "--engine",
            engine,
            "--input",
            str(conn_subset_path),
            "--nthreads",
            str(int(self.nthreads_spin.value())),
            "--t_thresh",
            f"{float(self.t_thresh_spin.value()):g}",
            "--nperm",
            str(int(self.nperm_spin.value())),
            "--matlab-test",
            matlab_test,
            "--matlab-size",
            component_size,
            "--regress",
            str(regressor),
            "--diag",
            "group",
        ]
        if engine == "python":
            command += ["--python-impl", "matlab_compat"]
            if hasattr(self, "seed_spin") and self.seed_spin.value() != self.seed_spin.minimum():
                command += ["--seed", str(int(self.seed_spin.value()))]
        if hasattr(self, "export_matlab_check") and self.export_matlab_check.isChecked():
            command.append("--export-matlab")
        if nuisance_terms:
            command += ["--nuisance", ",".join(str(x) for x in nuisance_terms)]

        regressor_type = self._selected_regressor_type()
        if regressor_type in {"categorical", "continuous"}:
            command += ["--regressor-type", regressor_type]

        filter_covar = self.filter_covar_combo.currentText().strip()
        filter_values = _parse_filter_values(self.filter_value_edit.text())
        if filter_covar and filter_values:
            select_values = [value for value in filter_values if not self._is_range_token(value)]
            for value in select_values:
                command += ["--select", f"{filter_covar},{value}"]

        contrast_text = self.contrast_edit.text().strip()
        if not contrast_text:
            raise ValueError("Custom contrast is required.")
        command += ["--contrast", contrast_text]

        if engine == "matlab":
            matlab_cmd = str(self._matlab_cmd_default or "").strip()
            matlab_ok = False
            if matlab_cmd:
                matlab_path = Path(matlab_cmd)
                if matlab_path.is_file() or shutil.which(matlab_cmd):
                    matlab_ok = True
            matlab_nbs_path = str(self._matlab_nbs_path_default or "").strip()
            nbs_ok = bool(matlab_nbs_path) and Path(matlab_nbs_path).is_dir()
            if not matlab_ok or not nbs_ok:
                raise ValueError(
                    "MATLAB engine requires valid MATLAB executable and NBS path in Preferences."
                )
            command += ["--matlab-cmd", matlab_cmd]
            command += ["--matlab-nbs-path", matlab_nbs_path]

        parcellation_path = self.parcellation_path_edit.text().strip()
        if parcellation_path:
            command += ["--parcellation-path", parcellation_path]
        modality = self.modality_combo.currentText().strip().lower()
        if modality:
            command += ["--modality", modality]
        command.append("--stage-no-significant")
        if engine == "matlab":
            if self.matlab_persistent_check.isChecked():
                command.append("--matlab-persistent")
            if self.matlab_no_precompute_check.isChecked():
                command.append("--matlab-no-precompute")
        return command

    def _build_process_environment(self):
        env = QProcessEnvironment.systemEnvironment()
        viewer_root = str(Path(__file__).resolve().parents[1])
        devanalyse = env.value("DEVANALYSEPATH", "")
        if not devanalyse:
            preferred_root = ""
            if self._output_dir_default:
                try:
                    preferred_root = str(Path(self._output_dir_default).expanduser().resolve().parent)
                except Exception:
                    preferred_root = str(Path(self._output_dir_default).expanduser().parent)
            devanalyse_root = preferred_root or viewer_root
            env.insert("DEVANALYSEPATH", devanalyse_root)
            print(f"[NBS] DEVANALYSEPATH not set; using {devanalyse_root}", flush=True)
            self._append_terminal_line(f"[NBS] DEVANALYSEPATH not set; using {devanalyse_root}")
        bids_data = env.value("BIDSDATAPATH", "")
        if not bids_data or bids_data == ".":
            fallback_bids = self._bids_dir_default or str(Path(viewer_root) / "data" / "BIDS")
            env.insert("BIDSDATAPATH", fallback_bids)
            print(f"[NBS] BIDSDATAPATH not set; using {fallback_bids}", flush=True)
            self._append_terminal_line(f"[NBS] BIDSDATAPATH not set; using {fallback_bids}")
        pythonpath_current = env.value("PYTHONPATH", "")
        pythonpath_parts = [p for p in pythonpath_current.split(os.pathsep) if p]
        for candidate in (viewer_root,):
            if candidate and candidate not in pythonpath_parts and Path(candidate).exists():
                pythonpath_parts.insert(0, candidate)
        if pythonpath_parts:
            env.insert("PYTHONPATH", os.pathsep.join(pythonpath_parts))
        return env

    def _covariate_balance_module(self):
        return "mrsitoolbox.scripts.covariate_balance_love_plot"

    def _covariate_balance_output_dir(self, regressor_name):
        base_dir_text = self.output_dir_edit.text().strip() if hasattr(self, "output_dir_edit") else ""
        if base_dir_text:
            base_dir = Path(base_dir_text).expanduser()
        else:
            base_dir = Path(self._default_output_dir_for_context()).expanduser()
        return (
            base_dir
            / "covariate_balance"
            / f"reg-{_slugify_fragment(regressor_name)}_n-{len(self.selected_row_indices())}"
        )

    def _build_covariate_balance_input(self):
        selected = self.selected_covariates()
        regressor_name = str(selected.get("regressor") or "").strip()
        if not regressor_name:
            raise ValueError("Select exactly one covariate as Regressor before running balance diagnostics.")

        nuisance_terms = [
            str(name).strip()
            for name in list(selected.get("nuisance") or [])
            if str(name).strip() and str(name).strip() != regressor_name
        ]
        if not nuisance_terms:
            raise ValueError("Select at least one nuisance covariate before running balance diagnostics.")
        if self._selected_regressor_type() == "continuous":
            raise ValueError("Covariate balance Love plots require a categorical regressor.")

        selected_indices = self.selected_row_indices()
        if len(selected_indices) < 2:
            raise ValueError("Select at least two rows before running balance diagnostics.")

        column_order = []
        lower_map = {str(name).strip().lower(): str(name) for name in self._columns}
        for identity_name in ("participant_id", "session_id"):
            resolved = lower_map.get(identity_name)
            if resolved is not None and resolved not in column_order:
                column_order.append(resolved)
        if regressor_name not in column_order:
            column_order.append(regressor_name)
        for covariate in nuisance_terms:
            if covariate not in column_order:
                column_order.append(covariate)

        output_dir = self._covariate_balance_output_dir(regressor_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        input_path = output_dir / (
            f"{self._source_path.stem}_key-{_slugify_fragment(self._matrix_key)}"
            f"_reg-{_slugify_fragment(regressor_name)}_n-{len(selected_indices)}_covariates.tsv"
        )
        with open(input_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=column_order, delimiter="\t")
            writer.writeheader()
            for idx in selected_indices:
                row = self._rows[int(idx)]
                payload = {column: _display_text(row.get(column)) for column in column_order}
                writer.writerow(payload)
        return input_path, output_dir, regressor_name, nuisance_terms

    def _run_covariate_balance_analysis(self):
        if self._process_is_running() or self._results_is_running():
            self._set_status("Wait for active processing to finish before launching covariate balance diagnostics.")
            return
        if self._export_is_running():
            self._set_status("Wait for export to finish before launching covariate balance diagnostics.")
            return
        if self._covariate_balance_is_running():
            self._set_status("Covariate balance diagnostics are already running.")
            return

        module_name = self._covariate_balance_module()
        try:
            module_spec = importlib.util.find_spec(module_name)
        except (ImportError, AttributeError, ValueError):
            module_spec = None
        if module_spec is None:
            self._set_status(
                "Covariate balance diagnostics require a newer mrsitoolbox pip package. "
                "Install or upgrade it with: pip install --upgrade mrsitoolbox"
            )
            return
        try:
            input_path, output_dir, regressor_name, nuisance_terms = self._build_covariate_balance_input()
        except Exception as exc:
            self._set_status(str(exc))
            return

        python_exe = sys.executable or "python3"
        command = [
            "-m",
            module_name,
            "--input",
            str(input_path),
            "--regressor",
            str(regressor_name),
            "--covars",
            ",".join(str(name) for name in nuisance_terms),
            "--outdir",
            str(output_dir),
        ]
        self._covariate_balance_output_tail = []
        self._last_covariate_balance_dir = str(output_dir)
        self._last_covariate_balance_plot_path = str(output_dir / "love_plot_smd.png")
        self._covariate_balance_process = QProcess(self)
        self._covariate_balance_process.readyReadStandardOutput.connect(self._on_covariate_balance_output)
        self._covariate_balance_process.readyReadStandardError.connect(self._on_covariate_balance_output)
        self._covariate_balance_process.finished.connect(self._on_covariate_balance_finished)
        self._covariate_balance_process.setWorkingDirectory(str(Path(__file__).resolve().parents[1]))
        self._covariate_balance_process.setProcessEnvironment(self._build_process_environment())
        self._append_terminal_line(f"[NBS-COVARS] Launch command: {shlex.join([python_exe] + command)}")
        self._set_status("Launching covariate balance diagnostics...")
        if hasattr(self, "covariate_balance_button"):
            self.covariate_balance_button.setText("Running...")
        self._covariate_balance_process.start(python_exe, command)
        self._update_run_state()
        if not self._covariate_balance_process.waitForStarted(3000):
            self._set_status("Failed to start covariate balance diagnostics process.")
            self._append_terminal_line(
                "[NBS-COVARS] Failed to start covariate balance diagnostics process."
            )
            self._covariate_balance_process = None
            if hasattr(self, "covariate_balance_button"):
                self.covariate_balance_button.setText("Covariate Balance")
            self._update_run_state()
            return

    def _on_covariate_balance_output(self):
        if self._covariate_balance_process is None:
            return
        for read_fn in (
            self._covariate_balance_process.readAllStandardOutput,
            self._covariate_balance_process.readAllStandardError,
        ):
            raw = bytes(read_fn()).decode("utf-8", errors="ignore")
            if not raw.strip():
                continue
            for line in raw.splitlines():
                text = line.strip()
                if text:
                    self._covariate_balance_output_tail.append(text)
                    self._append_terminal_line(f"[NBS-COVARS] {text}")
        if self._covariate_balance_output_tail:
            self._covariate_balance_output_tail = self._covariate_balance_output_tail[-12:]
            self._set_status(self._covariate_balance_output_tail[-1])

    def _show_covariate_balance_plot(self, plot_path):
        plot_path = Path(plot_path).expanduser()
        if not plot_path.is_file():
            self._set_status(f"Covariate balance Love plot not found: {plot_path}")
            return
        dialog = _ImagePreviewDialog(plot_path, "Covariate Balance Love Plot", parent=self)
        _theme, style = _workflow_dialog_stylesheet(self._theme_name)
        dialog.setStyleSheet(style)
        self._covariate_balance_preview_dialog = dialog
        dialog.finished.connect(lambda *_args: setattr(self, "_covariate_balance_preview_dialog", None))
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _on_covariate_balance_finished(self, exit_code, _exit_status):
        plot_path = Path(str(self._last_covariate_balance_plot_path or "")).expanduser()
        self._append_terminal_line(f"[NBS-COVARS] Process finished with exit code {exit_code}")
        if exit_code == 0:
            if plot_path.is_file():
                self._set_status(f"Covariate balance Love plot written to {plot_path}")
                self._append_terminal_line(f"[NBS-COVARS] Love plot saved to: {plot_path}")
                self._show_covariate_balance_plot(plot_path)
            else:
                self._set_status("Covariate balance diagnostics completed, but no Love plot was found.")
        else:
            tail = self._covariate_balance_output_tail[-1] if self._covariate_balance_output_tail else "See terminal output."
            self._set_status(f"Covariate balance diagnostics failed (exit {exit_code}). {tail}")
        self._covariate_balance_process = None
        if hasattr(self, "covariate_balance_button"):
            self.covariate_balance_button.setText("Covariate Balance")
        self._update_run_state()

    def _on_process_output(self):
        if self._run_process is None:
            return
        for read_fn in (
            self._run_process.readAllStandardOutput,
            self._run_process.readAllStandardError,
        ):
            raw = bytes(read_fn()).decode("utf-8", errors="ignore")
            if not raw.strip():
                continue
            for line in raw.splitlines():
                text = line.strip()
                if text:
                    self._run_output_tail.append(text)
                    print(f"[NBS] {text}", flush=True)
                    self._append_terminal_line(f"[NBS] {text}")
                    result_marker = "[NBS_RESULT]"
                    summary_marker = "[NBS_SUMMARY]"
                    if result_marker in text:
                        result_path = text.split(result_marker, 1)[1].strip()
                        if result_path:
                            self._last_result_npz_path = result_path
                    if summary_marker in text:
                        raw_summary = text.split(summary_marker, 1)[1].strip()
                        if raw_summary:
                            try:
                                self._last_nbs_summary = json.loads(raw_summary)
                                self._last_result_was_staged = bool(
                                    dict(self._last_nbs_summary or {}).get("result_staged")
                                )
                            except Exception:
                                self._last_nbs_summary = None
                                self._last_result_was_staged = False
        if self._run_output_tail:
            self._run_output_tail = self._run_output_tail[-8:]
            self._set_status(self._run_output_tail[-1])
            if hasattr(self, "run_progress_label"):
                self.run_progress_label.setText(self._run_output_tail[-1])

    def _on_process_finished(self, exit_code, _exit_status):
        self._set_run_busy(False)
        self._update_run_state()
        print(f"[NBS] Process finished with exit code {exit_code}", flush=True)
        self._append_terminal_line(f"[NBS] Process finished with exit code {exit_code}")
        if self._run_cancel_requested:
            self.export_button.setEnabled(False)
            if hasattr(self, "run_progress_label"):
                self.run_progress_label.setText("Cancelled")
            if hasattr(self, "run_progress"):
                self.run_progress.setRange(0, 1)
                self.run_progress.setValue(0)
            self._set_status("NBS run cancelled by user.")
        elif exit_code == 0:
            message = "NBS run completed successfully."
            if self._last_run_payload and self._last_run_payload.get("conn_subset_path"):
                subset_name = Path(self._last_run_payload["conn_subset_path"]).name
                message = f"NBS run completed successfully. Input: {subset_name}"
            staged_empty_result = (
                bool(self._last_result_was_staged)
                and int(dict(self._last_nbs_summary or {}).get("n_significant", 0) or 0) == 0
            )
            if self._last_result_npz_path and Path(self._last_result_npz_path).is_file():
                if staged_empty_result:
                    keep_result = self._confirm_save_no_significant_result(self._last_result_npz_path)
                    if keep_result:
                        original_staged_path = self._last_result_npz_path
                        copied_path = self._copy_result_to_output_folder(self._last_result_npz_path)
                        if copied_path:
                            self._last_result_npz_path = copied_path
                        self._cleanup_staged_result(original_staged_path, keep_if_same=self._last_result_npz_path)
                        self.export_button.setEnabled(True)
                        message = f"{message} No significant components were found; staged bundle kept after confirmation. Ready to export results."
                    else:
                        self._cleanup_staged_result(self._last_result_npz_path)
                        self._last_result_npz_path = None
                        self.export_button.setEnabled(False)
                        message = f"{message} No significant components were found; staged bundle discarded."
                else:
                    copied_path = self._copy_result_to_output_folder(self._last_result_npz_path)
                    if copied_path:
                        self._last_result_npz_path = copied_path
                    self.export_button.setEnabled(True)
                    message = f"{message} Ready to export results."
            else:
                self.export_button.setEnabled(False)
                if staged_empty_result:
                    message = f"{message} No significant components were found and no staged bundle was available."
                else:
                    message = f"{message} No bundled result path found in output."
            summary_line = self._format_nbs_summary(self._last_nbs_summary)
            if summary_line:
                message = f"{message} {summary_line}"
            if hasattr(self, "run_progress_label"):
                self.run_progress_label.setText("Completed")
            if hasattr(self, "run_progress"):
                self.run_progress.setRange(0, 1)
                self.run_progress.setValue(1)
            self._set_status(message)
            self._refresh_results_step(force_defaults=True)
            self._update_run_state()
            if self.export_on_finish_check.isChecked() and self.export_button.isEnabled():
                self._export_results()
        else:
            tail = self._run_output_tail[-1] if self._run_output_tail else "See terminal output."
            self._set_status(f"NBS run failed (exit {exit_code}). {tail}")
            self.export_button.setEnabled(False)
            if hasattr(self, "run_progress_label"):
                self.run_progress_label.setText("Failed")
            if hasattr(self, "run_progress"):
                self.run_progress.setRange(0, 1)
                self.run_progress.setValue(0)
        self._run_cancel_requested = False
        self._run_process = None
        self._update_run_state()

    def _force_kill_run_if_needed(self, process):
        if process is None:
            return
        if self._run_process is not process:
            return
        if process.state() == _qprocess_not_running():
            return
        print("[NBS] Process did not terminate in time. Sending kill signal.", flush=True)
        self._append_terminal_line("[NBS] Process did not terminate in time. Sending kill signal.")
        process.kill()

    def _cancel_run(self):
        if not self._process_is_running():
            self._set_status("No NBS run is currently active.")
            return
        self._run_cancel_requested = True
        self._set_status("Stopping NBS run...")
        if hasattr(self, "run_progress_label"):
            self.run_progress_label.setText("Stopping...")
        if hasattr(self, "stop_button"):
            self.stop_button.setEnabled(False)
        print("[NBS] Stop requested by user. Sending terminate signal.", flush=True)
        self._append_terminal_line("[NBS] Stop requested by user. Sending terminate signal.")
        process = self._run_process
        process.terminate()
        QTimer.singleShot(3000, lambda p=process: self._force_kill_run_if_needed(p))

    def _save_configuration(self):
        config = self.current_configuration()
        default_name = f"{self._source_path.stem}_{self._matrix_key}_nbs_prepare.json"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save NBS preparation",
            str(self._source_path.with_name(default_name)),
            "JSON (*.json);;All files (*)",
        )
        if not path:
            return
        output_path = Path(path)
        if output_path.suffix.lower() != ".json":
            output_path = output_path.with_suffix(".json")
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
        except Exception as exc:
            self._set_status(f"Failed to save configuration: {exc}")
            return
        self._set_status(f"Saved configuration to {output_path.name}")

    def _run_configuration(self):
        regressor = self._regressor_name()
        if not regressor:
            self._set_status("Select exactly one covariate as Regressor before running.")
            return
        classes = self._regressor_classes()
        if len(classes) < 2:
            self._set_status("Regressor must contain at least 2 classes in the current selection.")
            return
        selected_test = self.test_combo.currentText().strip()
        if selected_test == "Ftest" and len(classes) <= 2:
            self._set_status("Ftest is only available when regressor has more than 2 classes.")
            return
        if self._process_is_running():
            self._set_status("NBS run already in progress.")
            return
        if not self.contrast_edit.text().strip():
            self._set_status("Provide a custom contrast before running.")
            return
        try:
            conn_subset_path = self._build_filtered_conn_subset()
            command = self._build_run_command(conn_subset_path)
        except Exception as exc:
            self._set_status(f"Failed to prepare run command: {exc}")
            return

        self._last_run_payload = self.current_configuration()
        self._last_run_payload["conn_subset_path"] = str(conn_subset_path)
        self._last_run_payload["command"] = ["python3"] + command

        self._run_output_tail = []
        self._run_cancel_requested = False
        self._last_nbs_summary = None
        self._last_result_npz_path = None
        self._last_result_was_staged = False
        self.export_button.setEnabled(False)
        self._go_to_step(self.RUN_STEP_INDEX)
        self._run_process = QProcess(self)
        self._run_process.readyReadStandardOutput.connect(self._on_process_output)
        self._run_process.readyReadStandardError.connect(self._on_process_output)
        self._run_process.finished.connect(self._on_process_finished)
        self._run_process.setWorkingDirectory(str(Path(__file__).resolve().parents[1]))
        self._run_process.setProcessEnvironment(self._build_process_environment())

        python_exe = sys.executable or "python3"
        full_cmd = [python_exe] + command
        print("[NBS] Launch command:", shlex.join(full_cmd), flush=True)
        self._append_terminal_line(f"[NBS] Launch command: {shlex.join(full_cmd)}")
        self._set_run_busy(True)
        if hasattr(self, "run_progress_label"):
            self.run_progress_label.setText("Running...")
        self._set_status(
            f"Launching NBS ({selected_test}) with {len(self.selected_row_indices())} rows..."
        )
        self._run_process.start(python_exe, command)
        if not self._run_process.waitForStarted(3000):
            self._set_run_busy(False)
            self._set_status(
                "Failed to start run_nbs.py process. Check Python executable/path."
            )
            self._append_terminal_line(
                "[NBS] Failed to start run_nbs.py process. Check Python executable/path."
            )
            if hasattr(self, "run_progress_label"):
                self.run_progress_label.setText("Failed to start")
            self._run_process = None
            return

    def _confirm_save_no_significant_result(self, staged_result_path):
        result_path = Path(str(staged_result_path)).expanduser()
        summary = dict(self._last_nbs_summary or {})
        pvalue = summary.get("global_pvalue", None)
        pvalue_text = f"{float(pvalue):.6g}" if pvalue is not None else "NA"
        reply = QMessageBox.question(
            self,
            "No Significant NBS Result",
            (
                "No significant NBS components were detected.\n\n"
                f"Global p-value: {pvalue_text}\n"
                f"Staged bundle: {result_path}\n\n"
                "Do you want to save the staged result bundle anyway?"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return reply == QMessageBox.StandardButton.Yes

    def _cleanup_staged_result(self, result_path, keep_if_same=None):
        try:
            src = Path(str(result_path)).expanduser().resolve()
        except Exception:
            return
        if keep_if_same is not None:
            try:
                if src == Path(str(keep_if_same)).expanduser().resolve():
                    return
            except Exception:
                pass
        try:
            if src.is_file():
                src.unlink()
                self._append_terminal_line(f"[NBS] Deleted staged result bundle: {src}")
        except Exception as exc:
            self._append_terminal_line(f"[NBS] Failed to delete staged result bundle {src}: {exc}")
            return
        try:
            parent = src.parent
            if parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
        except Exception:
            pass

    def _copy_result_to_output_folder(self, result_path):
        try:
            src = Path(str(result_path)).expanduser().resolve()
        except Exception:
            return result_path
        target_dir_text = self.output_dir_edit.text().strip() if hasattr(self, "output_dir_edit") else ""
        if not target_dir_text:
            return str(src)
        target_dir = Path(target_dir_text).expanduser()
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self._append_terminal_line(f"[NBS] Could not create output folder {target_dir}: {exc}")
            return str(src)
        prefix = self.output_prefix_edit.text().strip() if hasattr(self, "output_prefix_edit") else ""
        dst_name = f"{prefix}_{src.name}" if prefix else src.name
        dst = target_dir / dst_name
        try:
            if src.resolve() == dst.resolve():
                return str(src)
        except Exception:
            pass
        try:
            shutil.copy2(str(src), str(dst))
            self._append_terminal_line(f"[NBS] Copied result bundle to: {dst}")
            return str(dst)
        except Exception as exc:
            self._append_terminal_line(f"[NBS] Failed to copy result bundle to {dst}: {exc}")
            return str(src)

    def _on_export_output(self):
        if self._export_process is None:
            return
        for read_fn in (
            self._export_process.readAllStandardOutput,
            self._export_process.readAllStandardError,
        ):
            raw = bytes(read_fn()).decode("utf-8", errors="ignore")
            if not raw.strip():
                continue
            for line in raw.splitlines():
                text = line.strip()
                if text:
                    self._export_output_tail.append(text)
                    print(f"[NBS-EXPORT] {text}", flush=True)
                    self._append_terminal_line(f"[NBS-EXPORT] {text}")
        if self._export_output_tail:
            self._export_output_tail = self._export_output_tail[-8:]
            self._set_status(self._export_output_tail[-1])
            if hasattr(self, "run_progress_label"):
                self.run_progress_label.setText(self._export_output_tail[-1])

    def _on_export_finished(self, exit_code, _exit_status):
        self._set_export_busy(False)
        print(f"[NBS-EXPORT] Process finished with exit code {exit_code}", flush=True)
        self._append_terminal_line(f"[NBS-EXPORT] Process finished with exit code {exit_code}")
        if exit_code == 0:
            cleanup_msg = self._cleanup_matlab_export_dir()
            if cleanup_msg:
                self._set_status(f"NBS result export completed. {cleanup_msg}")
            else:
                self._set_status("NBS result export completed.")
        else:
            tail = self._export_output_tail[-1] if self._export_output_tail else "See terminal output."
            self._set_status(f"NBS result export failed (exit {exit_code}). {tail}")
        self._export_process = None

    def _cleanup_matlab_export_dir(self):
        result_path = Path((self._last_result_npz_path or "").strip())
        if not result_path.exists():
            return ""
        connectome_dir = result_path.parent
        base_dir = None
        if connectome_dir.name == "connectome_plots":
            base_dir = connectome_dir.parent
        else:
            for parent in result_path.parents:
                if parent.name == "connectome_plots":
                    base_dir = parent.parent
                    break
        if base_dir is None:
            return ""
        matlab_dir = base_dir / "matlab_nbs"
        if not matlab_dir.exists():
            return ""
        try:
            shutil.rmtree(matlab_dir)
            print(f"[NBS-EXPORT] Deleted MATLAB export folder: {matlab_dir}", flush=True)
            self._append_terminal_line(f"[NBS-EXPORT] Deleted MATLAB export folder: {matlab_dir}")
            return f"Deleted {matlab_dir.name}."
        except Exception as exc:
            print(f"[NBS-EXPORT] Failed to delete {matlab_dir}: {exc}", flush=True)
            self._append_terminal_line(f"[NBS-EXPORT] Failed to delete {matlab_dir}: {exc}")
            return f"Could not delete {matlab_dir.name}: {exc}"

    def _active_gradient_rgb_payload(self):
        payload = None
        if self._gradient_rgb_payload_callback is not None:
            try:
                payload = self._gradient_rgb_payload_callback()
            except Exception as exc:
                self._append_terminal_line(f"[NBS-EXPORT] Gradient RGB payload unavailable: {exc}")
                payload = None
        if payload is None:
            parent = self.parent()
            if parent is not None and hasattr(parent, "_current_gradient_rgb_payload"):
                try:
                    payload = parent._current_gradient_rgb_payload()
                except Exception:
                    payload = None
        if not isinstance(payload, dict) or not bool(payload.get("enabled", False)):
            return None
        model = payload.get("model")
        if not isinstance(model, dict):
            return None
        return dict(payload)

    def _write_gradient_rgb_payload_file(self, result_path):
        payload = self._active_gradient_rgb_payload()
        if not payload:
            return None
        try:
            result = Path(str(result_path)).expanduser()
            payload_path = result.parent / f"{result.stem}_gradient_rgb_payload.json"
            payload["nbs_result_path"] = str(result)
            with open(payload_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            self._append_terminal_line(
                f"[NBS-EXPORT] Using active triangular RGB Gradient model: {payload_path}"
            )
            return str(payload_path)
        except Exception as exc:
            self._append_terminal_line(f"[NBS-EXPORT] Failed to write Gradient RGB payload: {exc}")
            return None

    def _export_results(self):
        if self._process_is_running() or self._results_is_running():
            self._set_status("Wait for active processing to finish before exporting.")
            return
        result_path = (self._last_result_npz_path or "").strip()
        if not result_path or not Path(result_path).is_file():
            self._set_status("No NBS result file available to export.")
            return
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_nbs_group.py"
        if not script_path.exists():
            self._set_status(f"Export script not found: {script_path}")
            return
        if self._export_process is not None and self._export_process.state() != _qprocess_not_running():
            self._set_status("Export already running.")
            return

        python_exe = sys.executable or "python3"
        command = [str(script_path), "--result", result_path]
        gradient_payload_path = self._write_gradient_rgb_payload_file(result_path)
        if gradient_payload_path:
            command += ["--gradient-rgb-payload", gradient_payload_path]
        if not self.display_results_check.isChecked():
            command.append("--no-show")
        if self.display_collapse_check.isChecked():
            command.append("--collapse-parcels")
        regressor_name = (self._regressor_name() or "").strip()
        if regressor_name:
            regressor_spec = regressor_name
            filter_covar = self.filter_covar_combo.currentText().strip().lower()
            filter_values = _parse_filter_values(self.filter_value_edit.text())
            if filter_values and filter_covar == regressor_name.lower():
                select_values = [value for value in filter_values if not self._is_range_token(value)]
                if select_values:
                    regressor_spec = ",".join([regressor_name] + select_values)
            command += ["--regressor", regressor_spec]
            regressor_type = self._selected_export_regressor_type()
            if regressor_type in {"categorical", "continuous"}:
                command += ["--regressor-type", regressor_type]
        self._export_output_tail = []
        self._export_process = QProcess(self)
        self._export_process.readyReadStandardOutput.connect(self._on_export_output)
        self._export_process.readyReadStandardError.connect(self._on_export_output)
        self._export_process.finished.connect(self._on_export_finished)
        self._export_process.setWorkingDirectory(str(Path(__file__).resolve().parents[1]))
        export_env = self._build_process_environment()
        if not export_env.contains("NUMBA_DISABLE_JIT"):
            export_env.insert("NUMBA_DISABLE_JIT", "1")
        self._export_process.setProcessEnvironment(export_env)
        print("[NBS-EXPORT] Launch command:", shlex.join([python_exe] + command), flush=True)
        self._append_terminal_line(f"[NBS-EXPORT] Launch command: {shlex.join([python_exe] + command)}")
        self._set_export_busy(True)
        self._set_status("Launching NBS result export...")
        self._export_process.start(python_exe, command)
        if not self._export_process.waitForStarted(3000):
            self._set_export_busy(False)
            self._set_status(
                "Failed to start analyze_nbs_group.py process. Check Python executable/path."
            )
            self._append_terminal_line(
                "[NBS-EXPORT] Failed to start analyze_nbs_group.py process. Check Python executable/path."
            )
            self._export_process = None

    def _on_results_output(self):
        if self._results_process is None:
            return
        for read_fn in (
            self._results_process.readAllStandardOutput,
            self._results_process.readAllStandardError,
        ):
            raw = bytes(read_fn()).decode("utf-8", errors="ignore")
            if not raw.strip():
                continue
            for line in raw.splitlines():
                text = line.strip()
                if text:
                    self._results_output_tail.append(text)
                    self._append_terminal_line(f"[NBS-RESULTS] {text}")
        if self._results_output_tail:
            self._results_output_tail = self._results_output_tail[-12:]
            self._set_status(self._results_output_tail[-1])

    def _on_results_finished(self, exit_code, _exit_status):
        self._set_results_busy(False)
        self._append_terminal_line(f"[NBS-RESULTS] Process finished with exit code {exit_code}")
        if exit_code == 0:
            output_dir = str(self._last_results_output_dir or "").strip()
            if output_dir:
                self._set_status(f"Structural context results completed. Output: {output_dir}")
            else:
                self._set_status("Structural context results completed.")
        else:
            tail = self._results_output_tail[-1] if self._results_output_tail else "See terminal output."
            self._set_status(f"Structural context results failed (exit {exit_code}). {tail}")
        self._results_process = None
        self._update_run_state()

    def _run_results_analysis(self):
        if self._process_is_running() or self._export_is_running() or self._covariate_balance_is_running():
            self._set_status("Wait for the current NBS task to finish before generating results.")
            return
        if self._results_is_running():
            self._set_status("Results generation is already running.")
            return
        result_path = str(self._last_result_npz_path or "").strip()
        if not result_path or not Path(result_path).is_file():
            self._set_status("No NBS result bundle is available yet.")
            return
        reference_path = self.reference_matrix_edit.text().strip() if hasattr(self, "reference_matrix_edit") else ""
        if not reference_path or not Path(reference_path).expanduser().is_file():
            self._set_status("Select a valid reference matrix NPZ.")
            return
        reference_key = self._selected_reference_matrix_key()
        if not reference_key:
            self._set_status("Select a valid matrix key from the reference NPZ.")
            return
        output_dir_text = self.results_output_dir_edit.text().strip() if hasattr(self, "results_output_dir_edit") else ""
        if not output_dir_text:
            self._set_status("Select an output folder for the results.")
            return
        output_dir = Path(output_dir_text).expanduser()
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self._set_status(f"Could not create results output folder: {exc}")
            return
        script_path = self._results_script_path()
        if not script_path.exists():
            self._set_status(f"Results script not found: {script_path}")
            return

        python_exe = sys.executable or "python3"
        command = [
            str(script_path),
            "-i",
            result_path,
            "--input_struct",
            str(Path(reference_path).expanduser()),
            "--input_struct_key",
            reference_key,
            "--output_dir",
            str(output_dir),
        ]
        self._results_output_tail = []
        self._last_results_output_dir = str(output_dir)
        self._results_process = QProcess(self)
        self._results_process.readyReadStandardOutput.connect(self._on_results_output)
        self._results_process.readyReadStandardError.connect(self._on_results_output)
        self._results_process.finished.connect(self._on_results_finished)
        self._results_process.setWorkingDirectory(str(Path(__file__).resolve().parents[1]))
        self._results_process.setProcessEnvironment(self._build_process_environment())
        self._append_terminal_line(f"[NBS-RESULTS] Launch command: {shlex.join([python_exe] + command)}")
        self._set_results_busy(True)
        self._set_status("Launching structural context results...")
        self._results_process.start(python_exe, command)
        if not self._results_process.waitForStarted(3000):
            self._set_results_busy(False)
            self._set_status("Failed to start structural context results process.")
            self._append_terminal_line("[NBS-RESULTS] Failed to start structural context results process.")
            self._results_process = None
            self._update_run_state()
            return

    def selected_covariates(self):
        selected_columns = []
        nuisance = []
        regressor = None
        for covar_name, include_button in self._covar_checks.items():
            if include_button is None or not include_button.isChecked():
                continue
            is_confound = bool(self._confound_buttons.get(covar_name).isChecked()) if self._confound_buttons.get(covar_name) is not None else False
            is_regressor = bool(self._regressor_buttons.get(covar_name).isChecked()) if self._regressor_buttons.get(covar_name) is not None else False
            if not is_confound and not is_regressor:
                continue
            selected_columns.append(covar_name)
            if is_confound:
                nuisance.append(covar_name)
            elif is_regressor:
                regressor = covar_name
        return {
            "selected_columns": selected_columns,
            "nuisance": nuisance,
            "regressor": regressor,
        }

    def selected_row_indices(self):
        return [idx for idx in self._filtered_indices if idx not in self._excluded_indices]


__all__ = ["NBSPrepareDialog"]
