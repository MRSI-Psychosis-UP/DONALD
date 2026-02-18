#!/usr/bin/env python3
"""Selector preparation dialog for filtered matrix aggregation."""

from pathlib import Path
import re

import numpy as np

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QAbstractItemView,
        QComboBox,
        QDialog,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QStackedWidget,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
        QHeaderView,
    )
    QT_LIB = 6
except Exception:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import (
        QAbstractItemView,
        QComboBox,
        QDialog,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QStackedWidget,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
        QHeaderView,
    )
    QT_LIB = 5


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


class SelectorPrepareDialog(QDialog):
    """Popup dialog to filter subjects and aggregate matrix stacks."""

    STEP_TITLES = ("Data", "Aggregate", "Export")
    NUMERIC_RANGE_RE = re.compile(
        r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*-\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*$"
    )

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
        self._aggregated_matrix = None
        self._aggregated_method = None
        self._export_callback = export_callback

        self.setWindowTitle("Selector Prepare")
        self.resize(1080, 760)
        self._build_ui()
        self.set_theme(theme_name)
        self._refresh_table()
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
        self.step_stack.addWidget(self._build_step_aggregate())
        self.step_stack.addWidget(self._build_step_export())
        right_layout.addWidget(self.step_stack, 1)
        content_row.addLayout(right_layout, 1)
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
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        actions.addWidget(self.close_button)
        root_layout.addLayout(actions)

    def _build_step_data(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        self.dataset_summary_label = QLabel("")
        self.dataset_summary_label.setWordWrap(False)
        self.dataset_summary_label.setText(
            f"{self._source_path.name} | key: {self._matrix_key} | "
            f"rows: {len(self._rows)} | covariates: {len(self._columns)}"
        )
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
        self.table.itemChanged.connect(self._on_table_item_changed)
        layout.addWidget(self.table, 1)

        self.showing_rows_label = QLabel("")
        layout.addWidget(self.showing_rows_label)
        return page

    def _build_step_aggregate(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        group = QGroupBox("Aggregator")
        grid = QGridLayout(group)
        grid.addWidget(QLabel("Method"), 0, 0)
        self.aggregate_method_combo = QComboBox()
        self.aggregate_method_combo.addItem("Mean (average)", "mean")
        self.aggregate_method_combo.addItem("Fisher Z (arctanh mean tanh)", "zfisher")
        grid.addWidget(self.aggregate_method_combo, 0, 1)
        self.aggregate_button = QPushButton("Aggregate")
        self.aggregate_button.clicked.connect(self._aggregate_selected)
        grid.addWidget(self.aggregate_button, 0, 2)
        layout.addWidget(group)

        self.aggregate_status_label = QLabel("No aggregation computed yet.")
        self.aggregate_status_label.setWordWrap(True)
        layout.addWidget(self.aggregate_status_label)
        layout.addStretch(1)
        return page

    def _build_step_export(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        summary_group = QGroupBox("Export Summary")
        summary_layout = QVBoxLayout(summary_group)
        self.export_summary_label = QLabel("Run aggregation first.")
        self.export_summary_label.setWordWrap(True)
        summary_layout.addWidget(self.export_summary_label)
        layout.addWidget(summary_group)

        export_row = QHBoxLayout()
        self.export_button = QPushButton("Export To Workspace")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._export_to_workspace)
        export_row.addWidget(self.export_button)
        export_row.addStretch(1)
        layout.addLayout(export_row)
        layout.addStretch(1)
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
        self.back_button.setEnabled(step > 0)
        self.next_button.setVisible(step < (len(self.STEP_TITLES) - 1))

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

    def _on_table_item_changed(self, item):
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
        self._refresh_table()
        self._set_status(
            f"Filter applied: {covar_name} in [{', '.join(target_values)}]. "
            f"Showing {len(self._filtered_indices)}/{len(self._rows)} rows."
        )

    def _reset_filter(self):
        self._filtered_indices = list(range(len(self._rows)))
        self._refresh_table()
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
        self.table.blockSignals(False)
        self._data_table_refreshing = False
        self._update_showing_rows_label()

    def _update_showing_rows_label(self):
        self.showing_rows_label.setText(
            f"Showing {len(self._filtered_indices)}/{len(self._rows)} rows | "
            f"Included: {len(self.selected_row_indices())}"
        )

    def selected_row_indices(self):
        return [idx for idx in self._filtered_indices if idx not in self._excluded_indices]

    def _load_subject_stack(self):
        with np.load(self._source_path, allow_pickle=True) as npz:
            if self._matrix_key not in npz:
                raise ValueError(f"Matrix key '{self._matrix_key}' not found in source file.")
            raw_matrix = np.asarray(npz[self._matrix_key], dtype=float)
        if raw_matrix.ndim != 3:
            raise ValueError("Aggregation requires a 3D matrix stack.")
        axis = _stack_axis(raw_matrix.shape)
        if axis is None:
            raise ValueError("Selected matrix is not a stack of square matrices.")
        if raw_matrix.shape[axis] != len(self._rows):
            raise ValueError("Covars length does not match matrix stack size.")
        selected = np.asarray(self.selected_row_indices(), dtype=int)
        if selected.size == 0:
            raise ValueError("No rows selected for aggregation.")

        if axis == 0:
            stack = raw_matrix[selected, :, :]
        elif axis == 1:
            stack = raw_matrix[:, selected, :].transpose(1, 0, 2)
        else:
            stack = raw_matrix[:, :, selected].transpose(2, 0, 1)
        return np.asarray(stack, dtype=float), selected

    def _aggregate_selected(self):
        try:
            stack, selected = self._load_subject_stack()
        except Exception as exc:
            self._set_status(f"Aggregation failed: {exc}")
            return

        method = str(self.aggregate_method_combo.currentData() or "mean")
        if method == "zfisher":
            clipped = np.clip(stack, -0.999999, 0.999999)
            with np.errstate(invalid="ignore"):
                z_stack = np.arctanh(clipped)
            z_mean = np.nanmean(z_stack, axis=0)
            aggregated = np.tanh(z_mean)
            method_label = "Fisher Z"
        else:
            aggregated = np.nanmean(stack, axis=0)
            method_label = "Mean"

        aggregated = np.asarray(aggregated, dtype=float)
        if aggregated.ndim != 2 or aggregated.shape[0] != aggregated.shape[1]:
            self._set_status("Aggregation failed: output is not a square matrix.")
            return
        aggregated = np.nan_to_num(aggregated, nan=0.0, posinf=0.0, neginf=0.0)

        self._aggregated_matrix = aggregated
        self._aggregated_method = method
        self.aggregate_status_label.setText(
            f"Computed {method_label} aggregation with {selected.size} selected rows."
        )
        self.export_summary_label.setText(
            f"Source: {self._source_path.name}\n"
            f"Key: {self._matrix_key}\n"
            f"Method: {method_label}\n"
            f"Selected rows: {selected.size}/{len(self._rows)}\n"
            f"Output shape: {aggregated.shape[0]} x {aggregated.shape[1]}"
        )
        self.export_button.setEnabled(True)
        self._set_status("Aggregation complete. Go to Export to import into workspace.")
        self._go_to_step(2)

    def _export_to_workspace(self):
        if self._aggregated_matrix is None:
            self._set_status("Run aggregation first.")
            return
        payload = {
            "matrix": np.asarray(self._aggregated_matrix, dtype=float),
            "method": self._aggregated_method or "mean",
            "source_path": str(self._source_path),
            "matrix_key": self._matrix_key,
            "selected_rows": self.selected_row_indices(),
            "n_total_rows": len(self._rows),
            "filter_covar": self.filter_covar_combo.currentText().strip(),
            "filter_values": self._parse_filter_values(self.filter_value_edit.text()),
        }
        if self._export_callback is None:
            self._set_status("No export callback defined.")
            return
        try:
            ok = bool(self._export_callback(payload))
        except Exception as exc:
            self._set_status(f"Export failed: {exc}")
            return
        if ok:
            self._set_status("Exported aggregated matrix to workspace.")
            self.accept()
        else:
            self._set_status("Export was not applied.")

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


__all__ = ["SelectorPrepareDialog"]
