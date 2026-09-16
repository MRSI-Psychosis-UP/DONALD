#!/usr/bin/env python3
"""Shared filterable covariate table widgets and helpers."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

try:
    from window.shared.controls import make_toggle_button
    from window.shared.covars import column_is_numeric, display_text
    from window.shared.qt_flags import (
        install_qt_compat_aliases,
        is_editable_flag,
        is_enabled_flag,
        is_selectable_flag,
        is_user_checkable_flag,
    )
except Exception:
    from mrsi_viewer.window.shared.controls import make_toggle_button
    from mrsi_viewer.window.shared.covars import column_is_numeric, display_text
    from mrsi_viewer.window.shared.qt_flags import (
        install_qt_compat_aliases,
        is_editable_flag,
        is_enabled_flag,
        is_selectable_flag,
        is_user_checkable_flag,
    )

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QAbstractItemView,
        QComboBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QPushButton,
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
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
    QT_LIB = 5


install_qt_compat_aliases(Qt, QT_LIB)


def format_dataset_summary(source_name: str, matrix_key: str, row_count: int, column_count: int) -> str:
    return (
        f"{source_name} | key: {matrix_key} | "
        f"rows: {int(row_count)} | covariates: {int(column_count)}"
    )


def parse_filter_values(text) -> list[str]:
    values = [token.strip() for token in str(text).split(",")]
    return [token for token in values if token != ""]


def parse_numeric_filter_targets(tokens: Iterable[str], numeric_range_re) -> tuple[list[float], list[tuple[float, float]]]:
    exact_targets = []
    range_targets = []
    for token in tokens:
        text = str(token).strip()
        if text == "":
            continue
        range_match = numeric_range_re.match(text)
        if range_match:
            start = float(range_match.group(1))
            stop = float(range_match.group(2))
            low, high = (start, stop) if start <= stop else (stop, start)
            range_targets.append((low, high))
            continue
        exact_targets.append(float(text))
    return exact_targets, range_targets


def matches_numeric(value, exact_targets: Sequence[float], range_targets: Sequence[tuple[float, float]]) -> bool:
    try:
        numeric_value = float(display_text(value).strip())
    except Exception:
        return False
    if any(np.isclose(numeric_value, target) for target in exact_targets):
        return True
    for low, high in range_targets:
        if (numeric_value > low or np.isclose(numeric_value, low)) and (
            numeric_value < high or np.isclose(numeric_value, high)
        ):
            return True
    return False


def matches_any(source_value, targets: Sequence, numeric: bool) -> bool:
    text = display_text(source_value).strip()
    if text == "":
        return False
    if numeric:
        try:
            value = float(text)
        except Exception:
            return False
        return any(np.isclose(value, target) for target in targets)
    return text in targets


def filter_row_indices(rows, covar_name: str, target_text: str, numeric_range_re):
    covar_name = str(covar_name or "").strip()
    target_text = str(target_text or "").strip()
    if not covar_name:
        return None, None, "Select a covariate to filter."
    if target_text == "":
        return None, None, "Enter a covariate value to filter."

    target_values = parse_filter_values(target_text)
    if not target_values:
        return None, None, "Enter at least one filter value."

    values = [row.get(covar_name) for row in rows]
    numeric = column_is_numeric(values)
    if numeric:
        try:
            exact_targets, range_targets = parse_numeric_filter_targets(target_values, numeric_range_re)
        except Exception:
            return None, None, "Selected covariate is numeric. Use values like 0,1 or ranges like 32-46."
        if not exact_targets and not range_targets:
            return None, None, "Enter at least one numeric value or numeric range."
    else:
        exact_targets = target_values
        range_targets = []

    matched = []
    for row_idx, value in enumerate(values):
        if numeric:
            if matches_numeric(value, exact_targets, range_targets):
                matched.append(row_idx)
        elif matches_any(value, exact_targets, numeric):
            matched.append(row_idx)
    return matched, target_values, None


class CovariateFilterPage(QWidget):
    """Common covariate filter page used by the prepare dialogs."""

    def __init__(
        self,
        columns,
        *,
        exclude_mode: str = "toggle_button",
        exclude_object_name: str = "tableExcludeButton",
        filter_field_label: str = "Covariate",
        filter_value_label: str = "Value",
        filter_placeholder: str = "e.g. 0,1 or 32-46",
        parent=None,
    ):
        super().__init__(parent)
        self._columns = [str(column) for column in columns]
        self._exclude_mode = str(exclude_mode or "toggle_button").strip().lower()
        self._exclude_object_name = str(exclude_object_name or "tableExcludeButton")

        layout = QVBoxLayout(self)

        self.dataset_summary_label = QLabel("")
        self.dataset_summary_label.setWordWrap(False)
        layout.addWidget(self.dataset_summary_label)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel(str(filter_field_label or "Covariate")))
        self.filter_covar_combo = QComboBox()
        self.filter_covar_combo.addItems(self._columns)
        filter_row.addWidget(self.filter_covar_combo)
        filter_row.addWidget(QLabel(str(filter_value_label or "Value")))
        self.filter_value_edit = QLineEdit("")
        self.filter_value_edit.setPlaceholderText(str(filter_placeholder or "e.g. 0,1 or 32-46"))
        filter_row.addWidget(self.filter_value_edit, 1)
        self.filter_button = QPushButton("Filter")
        filter_row.addWidget(self.filter_button)
        self.filter_reset_button = QPushButton("Reset")
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
        layout.addWidget(self.table, 1)

        self.showing_rows_label = QLabel("")
        layout.addWidget(self.showing_rows_label)

    def set_dataset_summary(self, source_name: str, matrix_key: str, row_count: int, column_count: int) -> None:
        self.dataset_summary_label.setText(
            format_dataset_summary(source_name, matrix_key, row_count, column_count)
        )

    def set_filter_callbacks(self, apply_callback, reset_callback) -> None:
        self.filter_button.clicked.connect(apply_callback)
        self.filter_reset_button.clicked.connect(reset_callback)

    def populate_rows(
        self,
        filtered_indices,
        rows,
        excluded_indices,
        *,
        on_exclude_toggled=None,
        resize_to_contents: bool = False,
    ) -> None:
        editable_flag = is_editable_flag(Qt)
        enabled_flag = is_enabled_flag(Qt)
        selectable_flag = is_selectable_flag(Qt)
        checkable_flag = is_user_checkable_flag(Qt)

        self.table.blockSignals(True)
        self.table.setRowCount(len(filtered_indices))
        for table_row, source_idx in enumerate(filtered_indices):
            row_data = rows[source_idx]
            if self._exclude_mode == "checkbox_item":
                exclude_item = QTableWidgetItem("")
                exclude_item.setFlags(enabled_flag | selectable_flag | checkable_flag)
                exclude_item.setCheckState(
                    Qt.Checked if source_idx in excluded_indices else Qt.Unchecked
                )
                self.table.setItem(table_row, 0, exclude_item)
            else:
                exclude_button = make_toggle_button(
                    checked=(source_idx in excluded_indices),
                    object_name=self._exclude_object_name,
                )
                if on_exclude_toggled is not None:
                    exclude_button.clicked.connect(
                        lambda checked=False, idx=source_idx: on_exclude_toggled(idx, checked)
                    )
                self.table.setCellWidget(table_row, 0, exclude_button)

            for col_idx, covar_name in enumerate(self._columns, start=1):
                item = QTableWidgetItem(display_text(row_data.get(covar_name)))
                item.setFlags(item.flags() & ~editable_flag)
                self.table.setItem(table_row, col_idx, item)
        if resize_to_contents:
            self.table.resizeColumnsToContents()
            self.table.resizeRowsToContents()
        self.table.blockSignals(False)

    def update_showing_rows(self, filtered_count: int, total_count: int, included_count: int) -> None:
        self.showing_rows_label.setText(
            f"Showing {int(filtered_count)}/{int(total_count)} rows | Included: {int(included_count)}"
        )
