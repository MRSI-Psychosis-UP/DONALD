#!/usr/bin/env python3
"""Prepare dialog for neuroCombat harmonization from matrix stacks."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QAbstractItemView,
        QComboBox,
        QDialog,
        QFileDialog,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPushButton,
        QStackedWidget,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
    QT_LIB = 6
except Exception:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import (
        QAbstractItemView,
        QComboBox,
        QDialog,
        QFileDialog,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPushButton,
        QStackedWidget,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
    QT_LIB = 5

from matplotlib.figure import Figure

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except Exception:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


if QT_LIB == 6:
    Qt.Checked = Qt.CheckState.Checked


def _is_enabled_flag():
    return getattr(Qt, "ItemIsEnabled", getattr(Qt.ItemFlag, "ItemIsEnabled"))


def _is_selectable_flag():
    return getattr(Qt, "ItemIsSelectable", getattr(Qt.ItemFlag, "ItemIsSelectable"))


def _is_user_checkable_flag():
    return getattr(Qt, "ItemIsUserCheckable", getattr(Qt.ItemFlag, "ItemIsUserCheckable"))


def _is_editable_flag():
    return getattr(Qt, "ItemIsEditable", getattr(Qt.ItemFlag, "ItemIsEditable"))


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
        text = _display_text(value).strip()
        if text == "":
            continue
        has_value = True
        try:
            float(text)
        except Exception:
            return False
    return has_value


def _load_harmonize_module():
    try:
        from scripts import harmonize_connectivity as mod

        return mod
    except Exception:
        pass

    try:
        from mrsi_viewer.scripts import harmonize_connectivity as mod

        return mod
    except Exception:
        pass

    module_path = Path(__file__).resolve().parents[1] / "scripts" / "harmonize_connectivity.py"
    spec = importlib.util.spec_from_file_location("harmonize_connectivity", str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load harmonize_connectivity.py from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class HarmonizePrepareDialog(QDialog):
    """Popup dialog to filter rows and run neuroCombat harmonization."""

    STEP_TITLES = ("Data", "Model", "Run")
    NUMERIC_RANGE_RE = re.compile(
        r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*-\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*$"
    )
    TYPE_OPTIONS = ("categorical", "continuous")

    def __init__(
        self,
        covars_info,
        source_path,
        matrix_key,
        theme_name="Dark",
        export_callback=None,
        parent=None,
    ):
        super().__init__(parent)
        self._source_path = Path(source_path)
        self._matrix_key = str(matrix_key)
        self._columns, self._rows = _covars_to_rows(covars_info)
        self._filtered_indices = list(range(len(self._rows)))
        self._excluded_indices = set()
        self._data_table_refreshing = False
        self._current_step = 0
        self._running = False
        self._export_callback = export_callback
        self._last_run = None
        self._last_output_path = None
        self._output_name_auto = True

        self._harm_mod = _load_harmonize_module()

        self.setWindowTitle("Harmonize Prepare")
        self.resize(1220, 850)
        self._build_ui()
        self.set_theme(theme_name)
        self._refresh_data_table()
        self._init_model_table()
        self._go_to_step(0)

    def _build_ui(self):
        root_layout = QVBoxLayout(self)

        content_row = QHBoxLayout()
        stepper_frame = QFrame()
        stepper_layout = QVBoxLayout(stepper_frame)
        stepper_layout.setContentsMargins(6, 6, 6, 6)
        stepper_layout.setSpacing(8)
        stepper_layout.addWidget(QLabel("Workflow"))

        self._step_buttons = []
        for idx, title in enumerate(self.STEP_TITLES):
            button = QPushButton(f"{idx + 1}. {title}")
            button.setObjectName("workflowStepButton")
            button.setCheckable(True)
            button.setMinimumHeight(36)
            button.clicked.connect(lambda _checked=False, i=idx: self._go_to_step(i))
            stepper_layout.addWidget(button)
            self._step_buttons.append(button)
        stepper_layout.addStretch(1)
        content_row.addWidget(stepper_frame, 0)

        right_layout = QVBoxLayout()
        self.step_stack = QStackedWidget()
        self.step_stack.addWidget(self._build_step_data())
        self.step_stack.addWidget(self._build_step_model())
        self.step_stack.addWidget(self._build_step_run())
        right_layout.addWidget(self.step_stack, 1)

        content_row.addLayout(right_layout, 1)
        root_layout.addLayout(content_row, 1)

        self.status_label = QLabel("")
        root_layout.addWidget(self.status_label)

        actions = QHBoxLayout()
        actions.addStretch(1)
        self.back_button = QPushButton("Back")
        self.back_button.clicked.connect(self._go_prev_step)
        actions.addWidget(self.back_button)
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self._go_next_step)
        actions.addWidget(self.next_button)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        actions.addWidget(self.close_button)
        root_layout.addLayout(actions)

    def _build_step_data(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        self.dataset_summary_label = QLabel(
            f"{self._source_path.name} | key: {self._matrix_key} | "
            f"rows: {len(self._rows)} | covariates: {len(self._columns)}"
        )
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

        self.data_table = QTableWidget()
        self.data_table.setColumnCount(len(self._columns) + 1)
        self.data_table.setHorizontalHeaderLabels(["Exclude"] + list(self._columns))
        self.data_table.setAlternatingRowColors(True)
        if QT_LIB == 6:
            self.data_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            self.data_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
            header = self.data_table.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            for col_idx in range(1, len(self._columns) + 1):
                header.setSectionResizeMode(col_idx, QHeaderView.ResizeMode.Stretch)
        else:
            self.data_table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.data_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
            header = self.data_table.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
            for col_idx in range(1, len(self._columns) + 1):
                header.setSectionResizeMode(col_idx, QHeaderView.Stretch)
        self.data_table.itemChanged.connect(self._on_data_table_item_changed)
        layout.addWidget(self.data_table, 1)

        self.showing_rows_label = QLabel("")
        layout.addWidget(self.showing_rows_label)
        return page

    def _build_step_model(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        model_group = QGroupBox("Batch and confounds")
        model_layout = QVBoxLayout(model_group)

        batch_row = QHBoxLayout()
        batch_row.addWidget(QLabel("Batch column"))
        self.batch_combo = QComboBox()
        self.batch_combo.addItems(self._columns)
        self.batch_combo.currentTextChanged.connect(self._on_batch_changed)
        batch_row.addWidget(self.batch_combo, 1)
        model_layout.addLayout(batch_row)

        self.confound_table = QTableWidget()
        self.confound_table.setColumnCount(3)
        self.confound_table.setHorizontalHeaderLabels(["Include", "Covariate", "Type"])
        self.confound_table.setRowCount(len(self._columns))
        self.confound_table.setAlternatingRowColors(True)
        if QT_LIB == 6:
            header = self.confound_table.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
            header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
            self.confound_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            self.confound_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        else:
            header = self.confound_table.horizontalHeader()
            header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
            header.setSectionResizeMode(1, QHeaderView.Stretch)
            header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
            self.confound_table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.confound_table.setSelectionMode(QAbstractItemView.SingleSelection)
        model_layout.addWidget(self.confound_table, 1)

        self.model_hint_label = QLabel(
            "Select nuisance/confound covariates and define each type for neuroCombat."
        )
        self.model_hint_label.setWordWrap(True)
        model_layout.addWidget(self.model_hint_label)

        layout.addWidget(model_group, 1)
        return page

    def _build_step_run(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        output_group = QGroupBox("Output")
        out_grid = QGridLayout(output_group)
        out_grid.addWidget(QLabel("Output folder"), 0, 0)
        self.output_dir_edit = QLineEdit(str(self._source_path.parent))
        out_grid.addWidget(self.output_dir_edit, 0, 1)
        self.output_dir_button = QPushButton("Browse")
        self.output_dir_button.clicked.connect(self._browse_output_dir)
        out_grid.addWidget(self.output_dir_button, 0, 2)

        out_grid.addWidget(QLabel("Output file"), 1, 0)
        self.output_name_edit = QLineEdit("")
        self.output_name_edit.setPlaceholderText("harmonized_connmat.npz")
        self.output_name_edit.textEdited.connect(self._on_output_name_edited)
        out_grid.addWidget(self.output_name_edit, 1, 1, 1, 2)

        out_grid.addWidget(QLabel("Full path"), 2, 0)
        self.output_path_label = QLabel("")
        self.output_path_label.setWordWrap(True)
        out_grid.addWidget(self.output_path_label, 2, 1, 1, 2)
        layout.addWidget(output_group)

        run_group = QGroupBox("Run")
        run_layout = QVBoxLayout(run_group)
        run_buttons = QHBoxLayout()
        self.run_button = QPushButton("Run neuroCombat")
        self.run_button.clicked.connect(self._run_harmonization)
        run_buttons.addWidget(self.run_button)
        self.export_button = QPushButton("Export To Workspace")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._export_to_workspace)
        run_buttons.addWidget(self.export_button)
        run_buttons.addStretch(1)
        run_layout.addLayout(run_buttons)

        self.run_summary_label = QLabel("No harmonization computed yet.")
        self.run_summary_label.setWordWrap(True)
        run_layout.addWidget(self.run_summary_label)

        self.result_figure = Figure(figsize=(10, 6))
        self.result_canvas = FigureCanvas(self.result_figure)
        run_layout.addWidget(self.result_canvas, 1)
        layout.addWidget(run_group, 1)

        self._refresh_output_path_preview()
        return page

    def _go_to_step(self, step_index):
        step = max(0, min(int(step_index), len(self.STEP_TITLES) - 1))
        self._current_step = step
        self.step_stack.setCurrentIndex(step)
        for idx, button in enumerate(self._step_buttons):
            is_current = idx == step
            button.setChecked(is_current)
            prefix = "▶ " if is_current else ""
            button.setText(f"{prefix}{idx + 1}. {self.STEP_TITLES[idx]}")
        self.back_button.setEnabled(step > 0 and not self._running)
        self.next_button.setVisible(step < (len(self.STEP_TITLES) - 1))
        self.next_button.setEnabled((step < (len(self.STEP_TITLES) - 1)) and (not self._running))

    def _go_next_step(self):
        self._go_to_step(self._current_step + 1)

    def _go_prev_step(self):
        self._go_to_step(self._current_step - 1)

    def _set_status(self, text):
        self.status_label.setText(str(text))

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
                self._set_status("Selected covariate is numeric. Use values like 0,1 or ranges like 32-46.")
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
        self._refresh_data_table()
        self._set_status(
            f"Filter applied: {covar_name} in [{', '.join(target_values)}]. "
            f"Showing {len(self._filtered_indices)}/{len(self._rows)} rows."
        )

    def _reset_filter(self):
        self._filtered_indices = list(range(len(self._rows)))
        self._refresh_data_table()
        self._set_status(f"Filter reset. Showing {len(self._filtered_indices)} rows.")

    def _refresh_data_table(self):
        self._data_table_refreshing = True
        self.data_table.blockSignals(True)
        self.data_table.setRowCount(len(self._filtered_indices))
        enabled_flag = _is_enabled_flag()
        selectable_flag = _is_selectable_flag()
        checkable_flag = _is_user_checkable_flag()
        editable_flag = _is_editable_flag()
        for table_row, source_idx in enumerate(self._filtered_indices):
            row_data = self._rows[source_idx]
            include_item = QTableWidgetItem("")
            include_item.setFlags(enabled_flag | selectable_flag | checkable_flag)
            include_item.setCheckState(Qt.Checked if source_idx in self._excluded_indices else Qt.Unchecked)
            self.data_table.setItem(table_row, 0, include_item)
            for col_idx, covar_name in enumerate(self._columns, start=1):
                item = QTableWidgetItem(_display_text(row_data.get(covar_name)))
                item.setFlags(item.flags() & ~editable_flag)
                self.data_table.setItem(table_row, col_idx, item)
        self.data_table.blockSignals(False)
        self._data_table_refreshing = False
        self._update_showing_rows_label()

    def _update_showing_rows_label(self):
        self.showing_rows_label.setText(
            f"Showing {len(self._filtered_indices)}/{len(self._rows)} rows | "
            f"Included: {len(self.selected_row_indices())}"
        )

    def selected_row_indices(self):
        return [idx for idx in self._filtered_indices if idx not in self._excluded_indices]

    def _default_batch_col(self) -> str:
        preferred = {"scanner", "site", "batch"}
        for name in self._columns:
            if name.strip().lower() in preferred:
                return name
        return self._columns[0] if self._columns else ""

    def _covariate_type(self, covar_name: str) -> str:
        values = [row.get(covar_name) for row in self._rows]
        return "continuous" if _column_is_numeric(values) else "categorical"

    def _is_id_like(self, covar_name: str) -> bool:
        return covar_name.strip().lower() in {
            "participant_id",
            "subject_id",
            "session_id",
            "id",
            "sub",
            "ses",
        }

    def _init_model_table(self):
        default_batch = self._default_batch_col()
        if default_batch and self.batch_combo.findText(default_batch) >= 0:
            self.batch_combo.setCurrentText(default_batch)

        for row_idx, covar_name in enumerate(self._columns):
            include_item = QTableWidgetItem("")
            flags = _is_enabled_flag() | _is_user_checkable_flag() | _is_selectable_flag()
            include_item.setFlags(flags)
            include_item.setCheckState(Qt.Unchecked)
            self.confound_table.setItem(row_idx, 0, include_item)

            name_item = QTableWidgetItem(covar_name)
            name_item.setFlags(name_item.flags() & ~_is_editable_flag())
            self.confound_table.setItem(row_idx, 1, name_item)

            type_combo = QComboBox()
            for option in self.TYPE_OPTIONS:
                type_combo.addItem(option)
            type_combo.setCurrentText(self._covariate_type(covar_name))
            type_combo.currentTextChanged.connect(
                lambda _value, self=self: self._apply_default_output_name(force=False)
            )
            self.confound_table.setCellWidget(row_idx, 2, type_combo)

            if self._is_id_like(covar_name):
                include_item.setCheckState(Qt.Unchecked)

        self.confound_table.itemChanged.connect(
            lambda _item, self=self: self._apply_default_output_name(force=False)
        )
        self._apply_default_output_name(force=True)

    def _selected_confounds(self):
        categorical = []
        continuous = []
        for row_idx, covar_name in enumerate(self._columns):
            include_item = self.confound_table.item(row_idx, 0)
            if include_item is None or include_item.checkState() != Qt.Checked:
                continue
            combo = self.confound_table.cellWidget(row_idx, 2)
            covar_type = combo.currentText().strip().lower() if combo is not None else "categorical"
            if covar_type == "continuous":
                continuous.append(covar_name)
            else:
                categorical.append(covar_name)
        return categorical, continuous

    def _on_batch_changed(self, _value):
        self._apply_default_output_name(force=False)

    def _on_output_name_edited(self, _text):
        self._output_name_auto = False
        self._refresh_output_path_preview()

    def _apply_default_output_name(self, force=False):
        if (not force) and (not self._output_name_auto):
            self._refresh_output_path_preview()
            return
        batch_col = self.batch_combo.currentText().strip() if hasattr(self, "batch_combo") else "batch"
        categorical, continuous = self._selected_confounds() if hasattr(self, "confound_table") else ([], [])
        try:
            default_path = self._harm_mod.build_default_output_path(
                source_path=self._source_path,
                matrix_key=self._matrix_key,
                batch_col=batch_col or "batch",
                categorical_cols=categorical,
                continuous_cols=continuous,
                output_dir=self.output_dir_edit.text().strip() if hasattr(self, "output_dir_edit") else self._source_path.parent,
            )
            self.output_name_edit.setText(default_path.name)
            self._output_name_auto = True
        except Exception:
            if force:
                self.output_name_edit.setText("harmonized_connmat.npz")
        self._refresh_output_path_preview()

    def _browse_output_dir(self):
        start_dir = self.output_dir_edit.text().strip() or str(self._source_path.parent)
        selected = QFileDialog.getExistingDirectory(self, "Select output folder", start_dir)
        if selected:
            self.output_dir_edit.setText(selected)
            self._apply_default_output_name(force=False)

    def _output_path(self) -> Path:
        out_dir = self.output_dir_edit.text().strip() or str(self._source_path.parent)
        out_name = self.output_name_edit.text().strip() or "harmonized_connmat.npz"
        output_path = Path(out_dir).expanduser() / out_name
        if output_path.suffix.lower() != ".npz":
            output_path = output_path.with_suffix(".npz")
        return output_path

    def _refresh_output_path_preview(self):
        if not hasattr(self, "output_path_label"):
            return
        try:
            output = self._output_path()
            self.output_path_label.setText(str(output))
        except Exception:
            self.output_path_label.setText("Invalid output path")

    def _set_busy(self, busy: bool):
        self._running = bool(busy)
        self.run_button.setEnabled(not busy)
        self.export_button.setEnabled((not busy) and (self._last_output_path is not None))
        self.back_button.setEnabled((not busy) and self._current_step > 0)
        self.next_button.setEnabled((not busy) and self._current_step < (len(self.STEP_TITLES) - 1))
        self.close_button.setEnabled(not busy)

    def _run_harmonization(self):
        if self._running:
            return

        selected_rows = self.selected_row_indices()
        if len(selected_rows) < 2:
            self._set_status("Select at least two rows before harmonization.")
            return

        batch_col = self.batch_combo.currentText().strip()
        if not batch_col:
            self._set_status("Select a batch column.")
            return

        categorical, continuous = self._selected_confounds()
        categorical = [col for col in categorical if col != batch_col]
        continuous = [col for col in continuous if col != batch_col]

        output_path = self._output_path()

        self._set_busy(True)
        self._set_status("Running neuroCombat harmonization...")
        self.run_summary_label.setText("Running...")

        try:
            run = self._harm_mod.run_harmonization(
                source_path=self._source_path,
                matrix_key=self._matrix_key,
                batch_col=batch_col,
                categorical_cols=categorical,
                continuous_cols=continuous,
                selected_indices=selected_rows,
                output_path=output_path,
                show_plots=False,
            )
        except Exception as exc:
            self._set_busy(False)
            self._set_status(f"Harmonization failed: {exc}")
            self.run_summary_label.setText("Run failed.")
            return

        self._last_run = run
        self._last_output_path = Path(run["output_path"])

        summary = run["result"]["summary"]
        self.run_summary_label.setText(
            f"Saved: {self._last_output_path}\n"
            f"N subjects: {summary.get('n_subjects')} | parcels: {summary.get('n_parcels')}\n"
            f"Batch: {summary.get('batch_col')} | counts: {summary.get('batch_counts')}\n"
            f"Confounds categorical: {', '.join(summary.get('categorical_cols') or []) or 'none'}\n"
            f"Confounds continuous: {', '.join(summary.get('continuous_cols') or []) or 'none'}\n"
            f"Mean original={summary.get('mean_original'):.4g}, harmonized={summary.get('mean_harmonized'):.4g}"
        )

        try:
            self._harm_mod.create_harmonization_plots(
                original_stack=run["prepared"]["matrix_stack"],
                harmonized_stack=run["result"]["harmonized_stack"],
                batch_values=run["result"]["batch_values"],
                batch_col=run["result"]["batch_col"],
                figure=self.result_figure,
            )
            self.result_canvas.draw_idle()
        except Exception as exc:
            self._set_status(f"Harmonization complete, but plotting failed: {exc}")

        self._set_busy(False)
        self._refresh_output_path_preview()
        self._set_status(f"Harmonization complete: {self._last_output_path.name}")

        if self._export_callback is not None:
            if self._ask_yes_no(
                "Export to workspace",
                "Harmonization finished. Export the result to the workspace now?",
            ):
                self._export_to_workspace()

    def _ask_yes_no(self, title: str, text: str) -> bool:
        if QT_LIB == 6:
            buttons = QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            choice = QMessageBox.question(
                self,
                title,
                text,
                buttons,
                QMessageBox.StandardButton.Yes,
            )
            return choice == QMessageBox.StandardButton.Yes
        choice = QMessageBox.question(
            self,
            title,
            text,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        return choice == QMessageBox.Yes

    def _export_to_workspace(self):
        if self._last_output_path is None:
            self._set_status("Run harmonization first.")
            return
        if self._export_callback is None:
            self._set_status("No workspace export callback configured.")
            return
        payload = {
            "output_path": str(self._last_output_path),
            "source_path": str(self._source_path),
            "matrix_key": self._matrix_key,
            "summary": self._last_run["result"].get("summary", {}) if self._last_run else {},
        }
        try:
            ok = bool(self._export_callback(payload))
        except Exception as exc:
            self._set_status(f"Workspace export failed: {exc}")
            return
        if ok:
            self._set_status("Exported harmonized result to workspace.")
        else:
            self._set_status("Workspace export was not applied.")

    def set_theme(self, theme_name):
        theme = str(theme_name or "Dark").strip().title()
        if theme not in {"Light", "Dark", "Teya", "Donald"}:
            theme = "Dark"
        if theme == "Dark":
            style = (
                "QWidget { background: #1f2430; color: #e5e7eb; font-size: 11pt; } "
                "QPushButton, QComboBox, QLineEdit, QTableWidget { "
                "background: #2a3140; color: #e5e7eb; border: 1px solid #556070; border-radius: 5px; } "
                "QPushButton { min-height: 30px; padding: 4px 10px; } "
                "QPushButton:hover { background: #344054; } "
                "QPushButton#workflowStepButton { text-align: left; padding-left: 10px; } "
                "QPushButton#workflowStepButton:checked { background: #2563eb; border: 2px solid #60a5fa; color: #ffffff; font-weight: 600; } "
                "QLineEdit, QComboBox { min-height: 30px; padding: 2px 4px; } "
                "QHeaderView::section { background: #2d3646; color: #e5e7eb; border: 1px solid #556070; } "
                "QTableWidget::item:selected { background: #3b82f6; color: #ffffff; }"
            )
        elif theme == "Teya":
            style = (
                "QWidget { background: #ffd0e5; color: #0b7f7a; font-size: 11pt; } "
                "QPushButton, QComboBox, QLineEdit, QTableWidget { "
                "background: #ffe6f1; color: #0b7f7a; border: 1px solid #1db8b2; border-radius: 5px; } "
                "QPushButton { min-height: 30px; padding: 4px 10px; } "
                "QPushButton:hover { background: #ffd9ea; } "
                "QPushButton#workflowStepButton { text-align: left; padding-left: 10px; } "
                "QPushButton#workflowStepButton:checked { background: #2ecfc9; border: 2px solid #0b7f7a; color: #073f3c; font-weight: 700; } "
                "QLineEdit, QComboBox { min-height: 30px; padding: 2px 4px; } "
                "QHeaderView::section { background: #ffc4df; color: #0b7f7a; border: 1px solid #1db8b2; } "
                "QTableWidget::item:selected { background: #2ecfc9; color: #073f3c; }"
            )
        elif theme == "Donald":
            style = (
                "QWidget { background: #d97706; color: #ffffff; font-size: 11pt; } "
                "QPushButton, QComboBox, QLineEdit, QTableWidget { "
                "background: #c96a04; color: #ffffff; border: 1px solid #f3a451; border-radius: 5px; } "
                "QPushButton { min-height: 30px; padding: 4px 10px; } "
                "QPushButton:hover { background: #c76b06; } "
                "QPushButton#workflowStepButton { text-align: left; padding-left: 10px; } "
                "QPushButton#workflowStepButton:checked { background: #b85f00; border: 2px solid #ffd19e; color: #ffffff; font-weight: 700; } "
                "QLineEdit, QComboBox { min-height: 30px; padding: 2px 4px; } "
                "QHeaderView::section { background: #c96a04; color: #ffffff; border: 1px solid #f3a451; } "
                "QTableWidget::item:selected { background: #2563eb; color: #ffffff; }"
            )
        else:
            style = (
                "QWidget { background: #f4f6f9; color: #1f2937; font-size: 11pt; } "
                "QPushButton, QComboBox, QLineEdit, QTableWidget { "
                "background: #ffffff; color: #1f2937; border: 1px solid #c9d0da; border-radius: 5px; } "
                "QPushButton { min-height: 30px; padding: 4px 10px; } "
                "QPushButton:hover { background: #edf2f7; } "
                "QPushButton#workflowStepButton { text-align: left; padding-left: 10px; } "
                "QPushButton#workflowStepButton:checked { background: #2563eb; border: 2px solid #1d4ed8; color: #ffffff; font-weight: 600; } "
                "QLineEdit, QComboBox { min-height: 30px; padding: 2px 4px; } "
                "QHeaderView::section { background: #eef2f7; color: #1f2937; border: 1px solid #c9d0da; } "
                "QTableWidget::item:selected { background: #2563eb; color: #ffffff; }"
            )
        self.setStyleSheet(style)


__all__ = ["HarmonizePrepareDialog"]
