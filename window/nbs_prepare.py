#!/usr/bin/env python3
"""NBS preparation dialog for covariate role selection and row filtering."""

import json
import os
import re
import shlex
import shutil
import sys
from pathlib import Path

import numpy as np

try:
    from PyQt6.QtCore import QProcess, QProcessEnvironment, QTimer, Qt
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
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QSplitter,
        QStackedWidget,
        QSpinBox,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
    QT_LIB = 6
except Exception:
    from PyQt5.QtCore import QProcess, QProcessEnvironment, QTimer, Qt
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
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QSplitter,
        QStackedWidget,
        QSpinBox,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
    QT_LIB = 5


if QT_LIB == 6:
    Qt.Checked = Qt.CheckState.Checked


def _is_editable_flag():
    return getattr(Qt, "ItemIsEditable", getattr(Qt.ItemFlag, "ItemIsEditable"))


def _is_enabled_flag():
    return getattr(Qt, "ItemIsEnabled", getattr(Qt.ItemFlag, "ItemIsEnabled"))


def _is_selectable_flag():
    return getattr(Qt, "ItemIsSelectable", getattr(Qt.ItemFlag, "ItemIsSelectable"))


def _is_user_checkable_flag():
    return getattr(Qt, "ItemIsUserCheckable", getattr(Qt.ItemFlag, "ItemIsUserCheckable"))


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


def _covars_to_rows(covars_info):
    if covars_info is None:
        return [], []

    df = covars_info.get("df")
    if df is not None:
        columns = [str(col) for col in df.columns]
        records = df.to_dict(orient="records")
        rows = []
        for record in records:
            rows.append({col: _decode_scalar(record.get(col)) for col in columns})
        return columns, rows

    data = covars_info.get("data")
    if data is None:
        return [], []

    arr = np.asarray(data)
    if getattr(arr.dtype, "names", None):
        columns = [str(col) for col in arr.dtype.names]
        rows = []
        for rec in arr:
            rows.append({col: _decode_scalar(rec[col]) for col in columns})
        return columns, rows

    if arr.ndim == 2:
        columns = [f"col_{i}" for i in range(arr.shape[1])]
        rows = []
        for row in arr:
            rows.append({columns[i]: _decode_scalar(row[i]) for i in range(arr.shape[1])})
        return columns, rows

    return [], []


def _column_is_numeric(values):
    has_value = False
    for value in values:
        value = _decode_scalar(value)
        text = _display_text(value).strip()
        if text == "":
            continue
        has_value = True
        try:
            float(text)
        except Exception:
            return False
    return has_value


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


def _slugify_fragment(value):
    text = str(value)
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", text)
    slug = slug.strip("-_")
    return slug or "value"


class NBSPrepareDialog(QDialog):
    """Dialog to prepare NBS covariates and select filtered row subsets."""

    ROLE_OPTIONS = ("Ignore", "Confound", "Regressor")
    REGRESSOR_TYPE_MODEL_OPTIONS = ("Auto", "Categorical", "Continuous")
    MODALITY_OPTIONS = ("fmri", "mrsi", "dwi", "anat", "morph", "pet", "other")
    TEST_OPTIONS = ("t-test", "Ftest")
    STEP_TITLES = ("Data", "Model", "NBS Settings", "Run")
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
        theme_name="Dark",
        parent=None,
    ):
        super().__init__(parent)
        self._source_path = Path(source_path)
        self._matrix_key = str(matrix_key)
        self._matlab_cmd_default = str(matlab_cmd_default or "").strip()
        self._matlab_nbs_path_default = str(matlab_nbs_path_default or "").strip()
        self._theme_name = "Dark"
        self._source_group = None
        self._source_modality = None
        self._last_result_npz_path = None
        self._last_nbs_summary = None
        self._export_process = None
        self._export_output_tail = []
        self._columns, self._rows = _covars_to_rows(covars_info)
        self._filtered_indices = list(range(len(self._rows)))
        self._excluded_indices = set()
        self._covar_checks = {}
        self._role_combos = {}
        self._last_run_payload = None
        self._run_process = None
        self._run_output_tail = []
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

    def _build_ui(self):
        root_layout = QVBoxLayout(self)

        content_row = QHBoxLayout()

        stepper_frame = QFrame()
        stepper_layout = QVBoxLayout(stepper_frame)
        stepper_layout.setContentsMargins(6, 6, 6, 6)
        stepper_layout.setSpacing(8)
        stepper_title = QLabel("Workflow")
        stepper_layout.addWidget(stepper_title)
        self._step_buttons = []
        for idx, title in enumerate(self.STEP_TITLES):
            button = QPushButton(f"{idx + 1}. {title}")
            button.setObjectName("workflowStepButton")
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, i=idx: self._go_to_step(i))
            button.setMinimumHeight(36)
            stepper_layout.addWidget(button)
            self._step_buttons.append(button)
        stepper_layout.addStretch(1)
        content_row.addWidget(stepper_frame, 0)

        right_panel = QVBoxLayout()
        self.step_stack = QStackedWidget()
        self.step_stack.addWidget(self._build_step_data())
        self.step_stack.addWidget(self._build_step_model())
        self.step_stack.addWidget(self._build_step_nbs_settings())
        self.step_stack.addWidget(self._build_step_run())
        right_panel.addWidget(self.step_stack, 1)

        self.log_toggle_button = QPushButton("Show log ▾")
        self.log_toggle_button.clicked.connect(self._toggle_log_drawer)
        self.log_toggle_button.setMaximumWidth(140)
        right_panel.addWidget(self.log_toggle_button, 0)

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
        right_panel.addWidget(self.log_drawer, 0)
        self._apply_terminal_style()
        self._set_log_drawer_expanded(False)

        content_row.addLayout(right_panel, 1)
        root_layout.addLayout(content_row, 1)

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
        self.run_button = QPushButton("Run")
        self.run_button.setMinimumWidth(130)
        self.run_button.clicked.connect(self._run_configuration)
        actions.addWidget(self.run_button)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        actions.addWidget(close_button)
        root_layout.addLayout(actions)

        self._go_to_step(0)
        self._update_dataset_summary()

    def _build_step_data(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        self.dataset_summary_label = QLabel("")
        self.dataset_summary_label.setWordWrap(False)
        layout.addWidget(self.dataset_summary_label)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Covariate"))
        self.filter_covar_combo = QComboBox()
        self.filter_covar_combo.addItems(self._columns)
        filter_row.addWidget(self.filter_covar_combo)
        filter_row.addWidget(QLabel("Value"))
        self.filter_value_edit = QLineEdit("")
        self.filter_value_edit.setPlaceholderText("e.g. 0,1 or 32-46")
        filter_row.addWidget(self.filter_value_edit, 1)
        self.filter_button = QPushButton("Filter")
        self.filter_button.clicked.connect(self._apply_filter)
        filter_row.addWidget(self.filter_button)
        self.filter_reset_button = QPushButton("Reset")
        self.filter_reset_button.clicked.connect(self._reset_filter)
        filter_row.addWidget(self.filter_reset_button)
        layout.addLayout(filter_row)

        self.table = QTableWidget()
        self.table.setColumnCount(len(self._columns) + 1)
        self.table.setHorizontalHeaderLabels(["Exclude"] + list(self._columns))
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(False)
        if QT_LIB == 6:
            self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
            header = self.table.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            for col_idx in range(1, len(self._columns) + 1):
                header.setSectionResizeMode(col_idx, QHeaderView.ResizeMode.Stretch)
        else:
            self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
            header = self.table.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
            for col_idx in range(1, len(self._columns) + 1):
                header.setSectionResizeMode(col_idx, QHeaderView.Stretch)
        self.table.itemChanged.connect(self._on_data_table_item_changed)
        layout.addWidget(self.table, 1)

        bottom = QHBoxLayout()
        self.showing_rows_label = QLabel("")
        bottom.addWidget(self.showing_rows_label)
        bottom.addStretch(1)
        layout.addLayout(bottom)
        return page

    def _build_step_model(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        self.model_table = QTableWidget()
        self.model_table.setColumnCount(4)
        self.model_table.setHorizontalHeaderLabels(["Include", "Name", "Type", "Role"])
        self.model_table.setRowCount(len(self._columns))
        if QT_LIB == 6:
            self.model_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            self.model_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            self.model_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
            self.model_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        else:
            self.model_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
            self.model_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
            self.model_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
            self.model_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.model_table.setAlternatingRowColors(True)
        self.model_table.setSelectionBehavior(QAbstractItemView.SelectRows if QT_LIB == 5 else QAbstractItemView.SelectionBehavior.SelectRows)
        self.model_table.setSelectionMode(QAbstractItemView.SingleSelection if QT_LIB == 5 else QAbstractItemView.SelectionMode.SingleSelection)

        self._covar_checks = {}
        self._role_combos = {}
        self._model_updating = True
        for row_idx, covar_name in enumerate(self._columns):
            include_item = QTableWidgetItem("")
            include_item.setFlags(_is_enabled_flag() | _is_user_checkable_flag())
            include_item.setCheckState(Qt.Unchecked)
            self.model_table.setItem(row_idx, 0, include_item)

            name_item = QTableWidgetItem(covar_name)
            name_item.setFlags(name_item.flags() & ~_is_editable_flag())
            self.model_table.setItem(row_idx, 1, name_item)

            type_item = QTableWidgetItem(self._covariate_type_label(covar_name))
            type_item.setFlags(type_item.flags() & ~_is_editable_flag())
            self.model_table.setItem(row_idx, 2, type_item)

            role_combo = QComboBox()
            role_combo.addItems(self.ROLE_OPTIONS)
            role_combo.setCurrentText("Ignore")
            role_combo.currentTextChanged.connect(
                lambda value, name=covar_name: self._on_role_changed(name, value)
            )
            self.model_table.setCellWidget(row_idx, 3, role_combo)

            self._covar_checks[covar_name] = include_item
            self._role_combos[covar_name] = role_combo
        self._model_updating = False
        self.model_table.itemChanged.connect(self._on_model_table_item_changed)
        layout.addWidget(self.model_table, 1)

        model_controls = QHBoxLayout()
        model_controls.addWidget(QLabel("Regressor type"))
        self.model_regressor_type_combo = QComboBox()
        self.model_regressor_type_combo.addItems(self.REGRESSOR_TYPE_MODEL_OPTIONS)
        self.model_regressor_type_combo.setCurrentText("Auto")
        self.model_regressor_type_combo.currentTextChanged.connect(self._update_run_summary)
        model_controls.addWidget(self.model_regressor_type_combo)
        model_controls.addStretch(1)
        layout.addLayout(model_controls)

        custom_contrast_frame = QFrame()
        custom_layout = QHBoxLayout(custom_contrast_frame)
        custom_layout.setContentsMargins(0, 0, 0, 0)
        custom_layout.addWidget(QLabel("Custom contrast"))
        self.contrast_edit = QLineEdit("")
        self.contrast_edit.setPlaceholderText("e.g. 0 0 1 1 1")
        custom_layout.addWidget(self.contrast_edit, 1)
        layout.addWidget(custom_contrast_frame)
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
        core_grid.addWidget(self.modality_combo, 2, 3)

        self.matlab_persistent_check = QCheckBox(
            "Persistent MATLAB session (reuse workers between runs)"
        )
        self.matlab_persistent_check.setChecked(True)
        core_grid.addWidget(self.matlab_persistent_check, 3, 0, 1, 4)

        self.matlab_no_precompute_check = QCheckBox(
            "Skip MATLAB precompute (lower memory, often slower)"
        )
        self.matlab_no_precompute_check.setChecked(False)
        core_grid.addWidget(self.matlab_no_precompute_check, 4, 0, 1, 4)

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
        self.output_dir_edit = QLineEdit("")
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

    def _go_to_step(self, step_index):
        step = max(0, min(step_index, len(self.STEP_TITLES) - 1))
        self._current_step = step
        self.step_stack.setCurrentIndex(step)
        for idx, button in enumerate(self._step_buttons):
            is_current = idx == step
            button.setChecked(is_current)
            prefix = "▶ " if is_current else ""
            button.setText(f"{prefix}{idx + 1}. {self.STEP_TITLES[idx]}")
        if step == 3:
            self._update_run_summary()
            if not self._log_expanded:
                self._set_log_drawer_expanded(True)
                self._log_auto_expanded = True
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
        self.back_button.setEnabled(self._current_step > 0)
        self.next_button.setVisible(self._current_step < max_step)
        self.run_button.setVisible(self._current_step == max_step)
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

    def _on_data_table_item_changed(self, item):
        if item is None or self._data_table_refreshing:
            return
        if item.column() != 0:
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
        self._update_test_options()
        self._update_run_state()
        self._update_run_summary()

    def _on_model_table_item_changed(self, item):
        if item is None or self._model_updating:
            return
        if item.column() != 0:
            return
        row = item.row()
        if row < 0 or row >= len(self._columns):
            return
        covar_name = self._columns[row]
        role_combo = self._role_combos.get(covar_name)
        if role_combo is None:
            return
        checked = item.checkState() == Qt.Checked
        if not checked and role_combo.currentText() != "Ignore":
            self._model_updating = True
            role_combo.setCurrentText("Ignore")
            self._model_updating = False
        self._update_test_options()
        self._update_run_state()
        self._update_run_summary()

    def _on_test_type_changed(self, *_args):
        self._update_run_summary()

    def _browse_output_dir(self):
        start_dir = self.output_dir_edit.text().strip() or str(self._source_path.parent)
        selected = QFileDialog.getExistingDirectory(self, "Select output folder", start_dir)
        if selected:
            self.output_dir_edit.setText(selected)

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
        self.run_summary_label.setText(
            f"Regressor: {regressor}\n"
            f"Confounds: {confounds}\n"
            f"N included: {n_included}\n"
            f"Permutations: {nperm}\n"
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
        start_dir = str(self._source_path.parent) if self._source_path is not None else str(Path.cwd())
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
        theme = str(theme_name or "Dark").strip().title()
        if theme not in {"Light", "Dark", "Teya", "Donald"}:
            theme = "Dark"
        self._theme_name = theme
        if theme == "Dark":
            style = (
                "QWidget { background: #1f2430; color: #e5e7eb; font-size: 11pt; } "
                "QPushButton, QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QTableWidget { "
                "background: #2a3140; color: #e5e7eb; border: 1px solid #556070; border-radius: 5px; } "
                "QPushButton { min-height: 30px; padding: 4px 10px; } "
                "QPushButton:hover { background: #344054; } "
                "QPushButton#workflowStepButton { text-align: left; padding-left: 10px; } "
                "QPushButton#workflowStepButton:checked { background: #2563eb; border: 2px solid #60a5fa; color: #ffffff; font-weight: 600; } "
                "QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { min-height: 30px; padding: 2px 4px; } "
                "QHeaderView::section { background: #2d3646; color: #e5e7eb; border: 1px solid #556070; } "
                "QTableWidget::item:selected { background: #3b82f6; color: #ffffff; }"
            )
        elif theme == "Teya":
            style = (
                "QWidget { background: #ffd0e5; color: #0b7f7a; font-size: 11pt; } "
                "QPushButton, QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QTableWidget { "
                "background: #ffe6f1; color: #0b7f7a; border: 1px solid #1db8b2; border-radius: 5px; } "
                "QPushButton { min-height: 30px; padding: 4px 10px; } "
                "QPushButton:hover { background: #ffd9ea; } "
                "QPushButton#workflowStepButton { text-align: left; padding-left: 10px; } "
                "QPushButton#workflowStepButton:checked { background: #2ecfc9; border: 2px solid #0b7f7a; color: #073f3c; font-weight: 700; } "
                "QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { min-height: 30px; padding: 2px 4px; } "
                "QHeaderView::section { background: #ffc4df; color: #0b7f7a; border: 1px solid #1db8b2; } "
                "QTableWidget::item:selected { background: #2ecfc9; color: #073f3c; }"
            )
        elif theme == "Donald":
            style = (
                "QWidget { background: #d97706; color: #ffffff; font-size: 11pt; } "
                "QPushButton, QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QTableWidget { "
                "background: #c96a04; color: #ffffff; border: 1px solid #f3a451; border-radius: 5px; } "
                "QPushButton { min-height: 30px; padding: 4px 10px; } "
                "QPushButton:hover { background: #c76b06; } "
                "QPushButton#workflowStepButton { text-align: left; padding-left: 10px; } "
                "QPushButton#workflowStepButton:checked { background: #b85f00; border: 2px solid #ffd19e; color: #ffffff; font-weight: 700; } "
                "QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { min-height: 30px; padding: 2px 4px; } "
                "QHeaderView::section { background: #c96a04; color: #ffffff; border: 1px solid #f3a451; } "
                "QTableWidget::item:selected { background: #2563eb; color: #ffffff; }"
            )
        else:
            style = (
                "QWidget { background: #f4f6f9; color: #1f2937; font-size: 11pt; } "
                "QPushButton, QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QTableWidget { "
                "background: #ffffff; color: #1f2937; border: 1px solid #c9d0da; border-radius: 5px; } "
                "QPushButton { min-height: 30px; padding: 4px 10px; } "
                "QPushButton:hover { background: #edf2f7; } "
                "QPushButton#workflowStepButton { text-align: left; padding-left: 10px; } "
                "QPushButton#workflowStepButton:checked { background: #2563eb; border: 2px solid #1d4ed8; color: #ffffff; font-weight: 600; } "
                "QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { min-height: 30px; padding: 2px 4px; } "
                "QHeaderView::section { background: #eef2f7; color: #1f2937; border: 1px solid #c9d0da; } "
                "QTableWidget::item:selected { background: #2563eb; color: #ffffff; }"
            )
        self.setStyleSheet(style)
        self._apply_terminal_style()

    def _process_is_running(self):
        return (
            self._run_process is not None
            and self._run_process.state() != _qprocess_not_running()
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

    def _set_export_busy(self, busy):
        self.export_button.setText("Exporting..." if busy else "Export Results")
        if busy:
            self.export_button.setEnabled(False)
        elif not self._process_is_running() and self._last_result_npz_path and Path(self._last_result_npz_path).is_file():
            self.export_button.setEnabled(True)

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
        can_run = len(classes) >= 2 and not self._process_is_running()
        if hasattr(self, "run_button"):
            self.run_button.setEnabled(can_run)
        if hasattr(self, "next_button"):
            self.next_button.setEnabled(
                (not self._process_is_running())
                and (self._current_step < (len(self.STEP_TITLES) - 1))
            )

    @staticmethod
    def _parse_filter_values(text):
        values = [token.strip() for token in str(text).split(",")]
        return [token for token in values if token != ""]

    @classmethod
    def _is_range_token(cls, token):
        return bool(cls.NUMERIC_RANGE_RE.match(str(token).strip()))

    @classmethod
    def _parse_numeric_filter_targets(cls, tokens):
        exact_targets = []
        range_targets = []
        for token in tokens:
            text = str(token).strip()
            if text == "":
                continue
            range_match = cls.NUMERIC_RANGE_RE.match(text)
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
    def _matches_any(source_value, targets, numeric):
        text = _display_text(source_value).strip()
        if text == "":
            return False
        if numeric:
            try:
                value = float(text)
            except Exception:
                return False
            return any(np.isclose(value, target) for target in targets)
        return text in targets

    def _on_covar_toggled(self, covar_name, checked):
        _ = covar_name
        _ = checked
        self._update_test_options()
        self._update_run_state()
        self._update_run_summary()

    def _on_role_changed(self, covar_name, role_value):
        if self._model_updating:
            return
        include_item = self._covar_checks.get(covar_name)
        role_combo = self._role_combos.get(covar_name)
        if include_item is None or role_combo is None:
            return

        self._model_updating = True
        if role_value == "Regressor":
            for other_name, other_combo in self._role_combos.items():
                if other_name == covar_name:
                    continue
                if other_combo.currentText() == "Regressor":
                    other_combo.blockSignals(True)
                    other_combo.setCurrentText("Ignore")
                    other_combo.blockSignals(False)
                    other_item = self._covar_checks.get(other_name)
                    if other_item is not None:
                        other_item.setCheckState(Qt.Unchecked)
            include_item.setCheckState(Qt.Checked)
            self._set_status(f"Regressor set to: {covar_name}")
        elif role_value == "Confound":
            include_item.setCheckState(Qt.Checked)
            self._set_status("Confounds updated.")
        else:
            include_item.setCheckState(Qt.Unchecked)
            self._set_status("Roles updated.")
        self._model_updating = False

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
        if numeric:
            try:
                exact_targets, range_targets = self._parse_numeric_filter_targets(target_values)
            except Exception:
                self._set_status(
                    "Selected covariate is numeric. Use values like 0,1 or ranges like 32-46."
                )
                return
            if not exact_targets and not range_targets:
                self._set_status("Enter at least one numeric value or numeric range.")
                return
        else:
            filter_targets = target_values
        matched = []
        for row_idx, value in enumerate(values):
            if numeric:
                if self._matches_numeric(value, exact_targets, range_targets):
                    matched.append(row_idx)
            elif self._matches_any(value, filter_targets, numeric):
                matched.append(row_idx)
        self._filtered_indices = matched
        self._refresh_table()
        self._update_test_options()
        self._update_run_state()
        self._update_run_summary()
        self._set_status(
            f"Filter applied: {covar_name} in [{', '.join(target_values)}]. "
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
        self.table.blockSignals(True)
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
        self.table.resizeColumnsToContents()
        self.table.resizeRowsToContents()
        self.table.blockSignals(False)
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
            "filter_values": self._parse_filter_values(self.filter_value_edit.text()),
            "covariates": selected,
            "regressor_classes": self._regressor_classes(),
            "test": self.test_combo.currentText().strip(),
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
        subset_name = (
            f"{self._source_path.stem}_key-{_slugify_fragment(self._matrix_key)}"
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

        run_script = Path(__file__).resolve().with_name("run_nbs.py")
        if not run_script.exists():
            raise FileNotFoundError(f"run_nbs.py not found at {run_script}")

        command = [
            str(run_script),
            "--engine",
            "matlab",
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
            "--regress",
            str(regressor),
            "--diag",
            "group",
        ]
        if nuisance_terms:
            command += ["--nuisance", ",".join(str(x) for x in nuisance_terms)]

        regressor_type = self._selected_regressor_type()
        if regressor_type in {"categorical", "continuous"}:
            command += ["--regressor-type", regressor_type]

        filter_covar = self.filter_covar_combo.currentText().strip()
        filter_values = self._parse_filter_values(self.filter_value_edit.text())
        if filter_covar and filter_values:
            select_values = [value for value in filter_values if not self._is_range_token(value)]
            for value in select_values:
                command += ["--select", f"{filter_covar},{value}"]

        contrast_text = self.contrast_edit.text().strip()
        if not contrast_text:
            raise ValueError("Custom contrast is required.")
        command += ["--contrast", contrast_text]

        matlab_cmd = self._matlab_cmd_default
        if matlab_cmd:
            command += ["--matlab-cmd", matlab_cmd]
        matlab_nbs_path = self._matlab_nbs_path_default
        if matlab_nbs_path:
            command += ["--matlab-nbs-path", matlab_nbs_path]
        parcellation_path = self.parcellation_path_edit.text().strip()
        if parcellation_path:
            command += ["--parcellation-path", parcellation_path]
        modality = self.modality_combo.currentText().strip().lower()
        if modality:
            command += ["--modality", modality]
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
            env.insert("DEVANALYSEPATH", viewer_root)
            print(f"[NBS] DEVANALYSEPATH not set; using {viewer_root}", flush=True)
            self._append_terminal_line(f"[NBS] DEVANALYSEPATH not set; using {viewer_root}")
        bids_data = env.value("BIDSDATAPATH", "")
        if not bids_data or bids_data == ".":
            fallback_bids = str(Path(viewer_root) / "data" / "BIDS")
            env.insert("BIDSDATAPATH", fallback_bids)
            print(f"[NBS] BIDSDATAPATH not set; using {fallback_bids}", flush=True)
            self._append_terminal_line(f"[NBS] BIDSDATAPATH not set; using {fallback_bids}")
        pythonpath_current = env.value("PYTHONPATH", "")
        pythonpath_parts = [p for p in pythonpath_current.split(os.pathsep) if p]
        mrsitoolbox_root = str(Path(viewer_root).parent / "mrsitoolbox")
        for candidate in (viewer_root, mrsitoolbox_root):
            if candidate and candidate not in pythonpath_parts and Path(candidate).exists():
                pythonpath_parts.insert(0, candidate)
        if pythonpath_parts:
            env.insert("PYTHONPATH", os.pathsep.join(pythonpath_parts))
        return env

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
                            except Exception:
                                self._last_nbs_summary = None
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
            if self._last_result_npz_path and Path(self._last_result_npz_path).is_file():
                copied_path = self._copy_result_to_output_folder(self._last_result_npz_path)
                if copied_path:
                    self._last_result_npz_path = copied_path
                self.export_button.setEnabled(True)
                message = f"{message} Ready to export results."
            else:
                self.export_button.setEnabled(False)
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
        self.export_button.setEnabled(False)
        self._go_to_step(3)
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

    def _export_results(self):
        if self._process_is_running():
            self._set_status("Wait for NBS run to finish before exporting.")
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
        if not self.display_results_check.isChecked():
            command.append("--no-show")
        if self.display_collapse_check.isChecked():
            command.append("--collapse-parcels")
        regressor_name = (self._regressor_name() or "").strip()
        if regressor_name:
            regressor_spec = regressor_name
            filter_covar = self.filter_covar_combo.currentText().strip().lower()
            filter_values = self._parse_filter_values(self.filter_value_edit.text())
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

    def selected_covariates(self):
        selected_columns = []
        nuisance = []
        regressor = None
        for covar_name, include_item in self._covar_checks.items():
            if include_item.checkState() != Qt.Checked:
                continue
            role = self._role_combos[covar_name].currentText()
            if role == "Ignore":
                continue
            selected_columns.append(covar_name)
            if role == "Confound":
                nuisance.append(covar_name)
            elif role == "Regressor":
                regressor = covar_name
        return {
            "selected_columns": selected_columns,
            "nuisance": nuisance,
            "regressor": regressor,
        }

    def selected_row_indices(self):
        return [idx for idx in self._filtered_indices if idx not in self._excluded_indices]


__all__ = ["NBSPrepareDialog"]
