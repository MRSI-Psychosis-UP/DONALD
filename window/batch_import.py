#!/usr/bin/env python3
"""Batch connectivity import dialog."""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from PyQt6.QtCore import Qt, QTimer
    from PyQt6.QtWidgets import (
        QAbstractItemView,
        QCheckBox,
        QComboBox,
        QDialog,
        QFrame,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QSplitter,
        QStackedWidget,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
    from PyQt6.QtGui import QColor
    QT_LIB = 6
except ImportError:
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtWidgets import (
        QAbstractItemView,
        QCheckBox,
        QComboBox,
        QDialog,
        QFrame,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QSplitter,
        QStackedWidget,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
    from PyQt5.QtGui import QColor
    QT_LIB = 5


if QT_LIB == 6:
    Qt.Checked = Qt.CheckState.Checked
    Qt.Unchecked = Qt.CheckState.Unchecked
    Qt.Vertical = Qt.Orientation.Vertical


def _write_diagnostic_line(text: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        sys.__stderr__.write(f"[{stamp}] {text}\n")
        sys.__stderr__.flush()
    except Exception:
        pass


def _user_role():
    return getattr(Qt, "UserRole", getattr(Qt.ItemDataRole, "UserRole"))


USER_ROLE = _user_role()


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
        results["scale"] = int(results["scale"])
    return results


def _connectivity_atlas_tag(path: Path) -> str:
    meta = _connectivity_file_metadata(Path(path))
    atlas = str(meta.get("atlas") or "").strip()
    scale = meta.get("scale")
    if atlas and scale is not None and f"scale{scale}" not in atlas:
        return f"{atlas}_scale{scale}"
    if atlas:
        return atlas
    return "unknown"


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
            seen.add(modality)
            found.append(modality)
    return found


def _infer_modality_from_path(path: Path) -> str:
    modalities = _connectivity_modalities_from_path(path)
    if modalities:
        return modalities[0]
    return "connectivity"


def _preferred_stack_matrix_key(modality: str) -> str | None:
    modality_token = str(modality or "").strip().lower()
    if modality_token == "dwi":
        return "connectome_density"
    if modality_token == "func":
        return "connectivity"
    if modality_token == "mrsi":
        return "simmatrix_sp"
    return None


def _square_matrix_candidate_shape(shape) -> bool:
    dims = tuple(int(dim) for dim in tuple(shape))
    if len(dims) == 2:
        return dims[0] == dims[1]
    if len(dims) == 3:
        if dims[1] == dims[2]:
            return True
        if dims[0] == dims[1]:
            return True
    return False


def _candidate_matrix_keys_for_path(path: Path) -> dict[str, tuple[int, ...]]:
    candidates = {}
    try:
        with np.load(str(path), allow_pickle=True) as archive:
            for key in archive.files:
                try:
                    array = np.asarray(archive[key])
                except Exception:
                    continue
                if not _square_matrix_candidate_shape(array.shape):
                    continue
                candidates[str(key)] = tuple(int(dim) for dim in array.shape)
    except Exception as exc:
        _write_diagnostic_line(f"Failed to inspect matrix candidates in {path}: {exc}")
    return candidates


def _shared_matrix_key_candidates(paths) -> list[dict[str, object]]:
    selected_paths = [Path(path) for path in (paths or []) if str(path).strip()]
    if not selected_paths:
        return []
    first_path = selected_paths[0]
    candidates = _candidate_matrix_keys_for_path(first_path)
    items = []
    for key in sorted(candidates.keys()):
        shape = tuple(candidates[key])
        items.append(
            {
                "key": str(key),
                "shapes": [shape],
                "shape_text": "x".join(str(dim) for dim in shape),
                "source_path": str(first_path),
            }
        )
    return items


def _is_enabled_flag():
    return getattr(Qt, "ItemIsEnabled", getattr(Qt.ItemFlag, "ItemIsEnabled"))


def _is_user_checkable_flag():
    return getattr(Qt, "ItemIsUserCheckable", getattr(Qt.ItemFlag, "ItemIsUserCheckable"))


def _align_right_flag():
    return getattr(Qt, "AlignRight", getattr(Qt.AlignmentFlag, "AlignRight"))


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
    text = str(value or "").strip()
    if text.startswith("sub-"):
        return text[4:]
    return text


def _normalize_session_token(value: str) -> str:
    text = str(value or "").strip()
    if text.startswith("ses-"):
        return text[4:]
    return text


class BatchMatrixImportDialog(QDialog):
    _MODALITY_OPTIONS = (
        ("All", ""),
        ("dwi", "connectivity_dwi"),
        ("mrsi", "connectivity_mrsi"),
        ("func", "connectivity_func"),
    )
    _STEP_TITLES = ("Select Matrices", "Selection", "Trim", "Setup", "Multimodal", "Run")
    _HARMONIZE_TYPE_OPTIONS = ("categorical", "continuous")

    def __init__(self, folder_path: Path, candidate_paths, export_callback=None, parent=None) -> None:
        super().__init__(parent)
        self._folder_path = Path(folder_path)
        self._candidate_paths = [Path(path) for path in candidate_paths]
        self._export_callback = export_callback
        self._atlas_options = self._detected_atlas_options()
        self._requested_action = "add"
        self._selection_widget = None
        self._trim_widget = None
        self._stack_prepare_widget = None
        self._stack_matrix_key_candidates = []
        self._current_step = 0
        self._multimodal_excluded_pairs = set()
        self._multimodal_table_refreshing = False
        self._multimodal_duplicate_entries = []
        self._workflow_log_expanded = False
        self._optional_steps_refresh_pending = False
        self.setWindowTitle("Add Batch")
        self.resize(1200, 800)
        self._build_ui()
        self._populate_files()
        self._apply_filters()
        self.set_theme(getattr(parent, "_theme_name", "Dark"))
        self._go_to_step(0)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self.workflow_splitter = QSplitter(Qt.Vertical)

        content_widget = QWidget()
        content_row = QHBoxLayout(content_widget)
        content_row.setContentsMargins(0, 0, 0, 0)

        stepper_frame = QFrame()
        stepper_layout = QVBoxLayout(stepper_frame)
        stepper_layout.setContentsMargins(6, 6, 6, 6)
        stepper_layout.setSpacing(8)
        stepper_title = QLabel("Workflow")
        stepper_layout.addWidget(stepper_title)
        self._step_buttons = []
        for idx, title in enumerate(self._STEP_TITLES):
            button = QPushButton(f"{idx + 1}. {title}")
            button.setObjectName("workflowStepButton")
            button.setCheckable(True)
            button.setMinimumHeight(36)
            button.clicked.connect(lambda _checked=False, i=idx: self._go_to_step(i))
            stepper_layout.addWidget(button)
            self._step_buttons.append(button)
        for button in self._step_buttons[1:]:
            button.setEnabled(False)
        stepper_layout.addStretch(1)
        content_row.addWidget(stepper_frame, 0)

        self.step_stack = QStackedWidget()
        content_row.addWidget(self.step_stack, 1)
        self.workflow_splitter.addWidget(content_widget)

        terminal_group = QGroupBox("Integrated Terminal")
        terminal_layout = QVBoxLayout(terminal_group)
        self.workflow_terminal = QPlainTextEdit()
        self.workflow_terminal.setReadOnly(True)
        self.workflow_terminal.setPlaceholderText("Stack progress will appear here.")
        self.workflow_terminal.setMinimumHeight(150)
        if QT_LIB == 6:
            self.workflow_terminal.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        else:
            self.workflow_terminal.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.workflow_terminal.setStyleSheet(
            "QPlainTextEdit {"
            " background-color: #000000;"
            " color: #b8f7c6;"
            " border: 1px solid #404040;"
            " border-radius: 4px;"
            " selection-background-color: #2d6cdf;"
            " font-family: 'DejaVu Sans Mono', 'Courier New', monospace;"
            " font-size: 10.5pt;"
            "}"
        )
        terminal_layout.addWidget(self.workflow_terminal, 1)
        self.workflow_splitter.addWidget(terminal_group)
        self.workflow_splitter.setStretchFactor(0, 4)
        self.workflow_splitter.setStretchFactor(1, 1)
        self.workflow_splitter.setSizes([620, 180])
        layout.addWidget(self.workflow_splitter, 1)

        self.workflow_log_toggle_button = QPushButton("Show log ▾")
        self.workflow_log_toggle_button.clicked.connect(self._toggle_workflow_log_drawer)
        self.workflow_log_toggle_button.setMaximumWidth(140)
        layout.addWidget(self.workflow_log_toggle_button, 0, _align_right_flag())
        self.workflow_log_drawer = terminal_group
        self._set_workflow_log_drawer_expanded(False)

        selection_page = QWidget()
        selection_layout = QVBoxLayout(selection_page)

        self.folder_label = QLabel(f"Folder: {self._folder_path}")
        self.folder_label.setWordWrap(True)
        selection_layout.addWidget(self.folder_label)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filter"))
        self.filter_value_edit = QLineEdit("")
        self.filter_value_edit.setPlaceholderText("Substring match, e.g. scale3 or sub-001")
        self.filter_value_edit.textChanged.connect(self._apply_filters)
        self.filter_value_edit.returnPressed.connect(self._apply_filters)
        filter_row.addWidget(self.filter_value_edit, 1)

        filter_row.addWidget(QLabel("Modality"))
        self.modality_combo = QComboBox()
        for label, token in self._MODALITY_OPTIONS:
            self.modality_combo.addItem(label, token)
        self.modality_combo.currentIndexChanged.connect(self._apply_filters)
        filter_row.addWidget(self.modality_combo)

        self.filter_button = QPushButton("Filter")
        self.filter_button.clicked.connect(self._apply_filters)
        filter_row.addWidget(self.filter_button)

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self._reset_filters)
        filter_row.addWidget(self.reset_button)
        selection_layout.addLayout(filter_row)

        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        selection_layout.addWidget(self.summary_label)

        self.file_list = QListWidget()
        if QT_LIB == 6:
            self.file_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        else:
            self.file_list.setSelectionMode(QAbstractItemView.NoSelection)
        self.file_list.itemChanged.connect(self._on_item_changed)
        selection_layout.addWidget(self.file_list, 1)

        select_row = QHBoxLayout()
        self.select_visible_button = QPushButton("Select Visible")
        self.select_visible_button.clicked.connect(lambda: self._set_visible_checked(True))
        select_row.addWidget(self.select_visible_button)
        self.clear_visible_button = QPushButton("Clear Visible")
        self.clear_visible_button.clicked.connect(lambda: self._set_visible_checked(False))
        select_row.addWidget(self.clear_visible_button)
        select_row.addStretch(1)
        selection_layout.addLayout(select_row)

        actions = QHBoxLayout()
        actions.addStretch(1)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        actions.addWidget(self.cancel_button)
        self.stack_button = QPushButton("Stack")
        self.stack_button.clicked.connect(self._open_stack_prepare)
        actions.addWidget(self.stack_button)
        self.add_selected_button = QPushButton("Add Selected")
        self.add_selected_button.clicked.connect(self._accept_if_selection)
        actions.addWidget(self.add_selected_button)
        selection_layout.addLayout(actions)

        self.selection_page = QWidget()
        self.selection_tab_layout = QVBoxLayout(self.selection_page)
        self.matrix_key_group = QGroupBox("Matrix Entry")
        matrix_key_layout = QVBoxLayout(self.matrix_key_group)
        matrix_key_row = QHBoxLayout()
        matrix_key_row.addWidget(QLabel("Matrix key"))
        self.matrix_key_combo = QComboBox()
        self.matrix_key_combo.currentIndexChanged.connect(self._on_stack_matrix_key_changed)
        matrix_key_row.addWidget(self.matrix_key_combo, 1)
        matrix_key_layout.addLayout(matrix_key_row)
        self.matrix_key_summary_label = QLabel(
            "Square matrix entries will be detected across the selected NPZ files."
        )
        self.matrix_key_summary_label.setWordWrap(True)
        matrix_key_layout.addWidget(self.matrix_key_summary_label)
        self.matrix_key_group.setVisible(False)
        self.selection_tab_layout.addWidget(self.matrix_key_group)
        self.selection_placeholder = QLabel(
            "Step 2 appears here after you select matrices in Step 1 and click 'Stack'."
        )
        self.selection_placeholder.setWordWrap(True)
        self.selection_tab_layout.addWidget(self.selection_placeholder)
        self.selection_tab_layout.addStretch(1)

        self.trim_page = QWidget()
        self.trim_tab_layout = QVBoxLayout(self.trim_page)
        self.trim_placeholder = QLabel(
            "Step 3 appears here after you open the Selection step."
        )
        self.trim_placeholder.setWordWrap(True)
        self.trim_tab_layout.addWidget(self.trim_placeholder)
        self.trim_tab_layout.addStretch(1)

        self.setup_page = QWidget()
        self.setup_tab_layout = QVBoxLayout(self.setup_page)
        self.setup_placeholder = QLabel(
            "Step 4 appears here after you open the Selection step."
        )
        self.setup_placeholder.setWordWrap(True)
        self.setup_tab_layout.addWidget(self.setup_placeholder)
        self.setup_tab_layout.addStretch(1)

        multimodal_page = QWidget()
        multimodal_layout = QVBoxLayout(multimodal_page)
        self.multimodal_enable_check = QCheckBox("Enable multimodal stack")
        self.multimodal_enable_check.toggled.connect(self._update_multimodal_controls)
        multimodal_layout.addWidget(self.multimodal_enable_check)
        self.multimodal_align_check = QCheckBox("Align cross-modality matrices")
        self.multimodal_align_check.setChecked(False)
        multimodal_layout.addWidget(self.multimodal_align_check)
        reference_row = QHBoxLayout()
        reference_row.addWidget(QLabel("Reference modality"))
        self.reference_modality_combo = QComboBox()
        self.reference_modality_combo.currentIndexChanged.connect(self._refresh_multimodal_table)
        self.reference_modality_combo.currentIndexChanged.connect(self._refresh_run_table)
        reference_row.addWidget(self.reference_modality_combo, 1)
        multimodal_layout.addLayout(reference_row)
        self.multimodal_summary_label = QLabel("")
        self.multimodal_summary_label.setWordWrap(True)
        multimodal_layout.addWidget(self.multimodal_summary_label)
        self.multimodal_table = QTableWidget()
        self.multimodal_table.setAlternatingRowColors(True)
        self.multimodal_table.itemChanged.connect(self._on_multimodal_table_item_changed)
        if QT_LIB == 6:
            self.multimodal_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            self.multimodal_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        else:
            self.multimodal_table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.multimodal_table.setSelectionMode(QAbstractItemView.SingleSelection)
        multimodal_layout.addWidget(self.multimodal_table, 1)
        self.multimodal_optional_label = QLabel(
            "Optional step. Continue to Run without configuring multimodal stacking."
        )
        self.multimodal_optional_label.setWordWrap(True)
        multimodal_layout.addWidget(self.multimodal_optional_label)

        run_page = QWidget()
        run_layout = QVBoxLayout(run_page)
        self.run_info_label = QLabel(
            "Final step. Review the subject/session pairs that will be stacked, then run processing."
        )
        self.run_info_label.setWordWrap(True)
        run_layout.addWidget(self.run_info_label)
        self.run_summary_label = QLabel(
            "The stack preview will appear here after Selection, Trim, and Setup are ready."
        )
        self.run_summary_label.setWordWrap(True)
        run_layout.addWidget(self.run_summary_label)
        self.run_table = QTableWidget()
        self.run_table.setAlternatingRowColors(True)
        if QT_LIB == 6:
            self.run_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            self.run_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        else:
            self.run_table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.run_table.setSelectionMode(QAbstractItemView.SingleSelection)
        run_layout.addWidget(self.run_table, 1)
        run_actions = QHBoxLayout()
        run_actions.addStretch(1)
        self.run_import_button = QPushButton("Import to workspace")
        self.run_import_button.setEnabled(False)
        self.run_import_button.clicked.connect(self._trigger_workflow_import)
        run_actions.addWidget(self.run_import_button)
        self.run_close_button = QPushButton("Close")
        self.run_close_button.clicked.connect(self._request_close)
        run_actions.addWidget(self.run_close_button)
        self.run_process_button = QPushButton("Process")
        self.run_process_button.setEnabled(False)
        self.run_process_button.clicked.connect(self._trigger_workflow_process)
        run_actions.addWidget(self.run_process_button)
        run_layout.addLayout(run_actions)

        self.step_stack.addWidget(selection_page)
        self.step_stack.addWidget(self.selection_page)
        self.step_stack.addWidget(self.trim_page)
        self.step_stack.addWidget(self.setup_page)
        self.step_stack.addWidget(multimodal_page)
        self.step_stack.addWidget(run_page)

    def _populate_files(self) -> None:
        self.file_list.blockSignals(True)
        self.file_list.clear()
        flags = _is_enabled_flag() | _is_user_checkable_flag()
        for path in self._candidate_paths:
            item = QListWidgetItem(self._relative_label(path))
            item.setToolTip(str(path))
            item.setData(USER_ROLE, str(path))
            item.setFlags(item.flags() | flags)
            item.setCheckState(Qt.Unchecked)
            self.file_list.addItem(item)
        self.file_list.blockSignals(False)

    def _relative_label(self, path: Path) -> str:
        try:
            return path.relative_to(self._folder_path).as_posix()
        except Exception:
            return path.name

    def _detected_atlas_options(self):
        atlases = {_connectivity_atlas_tag(path) for path in self._candidate_paths}
        return sorted(atlases, key=lambda value: (str(value).lower() == "unknown", str(value).lower()))

    def _filter_tokens(self):
        values = [token.strip().lower() for token in self.filter_value_edit.text().split(",")]
        return [token for token in values if token]

    def _matches_current_filters(self, path: Path) -> bool:
        name = self._relative_label(path).lower()
        modality_token = str(self.modality_combo.currentData() or "").strip().lower()
        if modality_token and modality_token not in name:
            return False
        for token in self._filter_tokens():
            if token not in name:
                return False
        return True

    def _checked_count(self) -> int:
        total = 0
        for row in range(self.file_list.count()):
            item = self.file_list.item(row)
            if item is not None and item.checkState() == Qt.Checked:
                total += 1
        return total

    def _on_item_changed(self, _item) -> None:
        self._apply_filters()
        if self._selection_widget is not None:
            self._selection_widget.set_selected_paths(self.selected_paths())
        if self._trim_widget is not None:
            self._trim_widget.set_selected_paths(self.selected_paths())
        if self._stack_prepare_widget is not None:
            self._stack_prepare_widget.set_selected_paths(self.selected_paths())
            self._refresh_optional_steps()

    def _apply_filters(self) -> None:
        visible_count = 0
        for row in range(self.file_list.count()):
            item = self.file_list.item(row)
            if item is None:
                continue
            raw_path = item.data(USER_ROLE)
            path = Path(str(raw_path))
            is_visible = self._matches_current_filters(path)
            item.setHidden(not is_visible)
            if is_visible:
                visible_count += 1
        total = self.file_list.count()
        selected = self._checked_count()
        self.summary_label.setText(
            f"Showing {visible_count} of {total} connectivity matrices. Selected: {selected}."
        )

    def _reset_filters(self) -> None:
        self.filter_value_edit.clear()
        self.modality_combo.setCurrentIndex(0)
        self._apply_filters()

    def _set_visible_checked(self, checked: bool) -> None:
        target_state = Qt.Checked if checked else Qt.Unchecked
        self.file_list.blockSignals(True)
        for row in range(self.file_list.count()):
            item = self.file_list.item(row)
            if item is None or item.isHidden():
                continue
            item.setCheckState(target_state)
        self.file_list.blockSignals(False)
        self._apply_filters()

    def selected_paths(self):
        paths = []
        for row in range(self.file_list.count()):
            item = self.file_list.item(row)
            if item is None or item.checkState() != Qt.Checked:
                continue
            raw_path = item.data(USER_ROLE)
            if raw_path:
                paths.append(str(raw_path))
        return paths

    def requested_action(self) -> str:
        return str(self._requested_action or "add")

    def _stack_processing(self) -> bool:
        if self._stack_prepare_widget is None:
            return False
        if hasattr(self._stack_prepare_widget, "is_processing"):
            try:
                return bool(self._stack_prepare_widget.is_processing())
            except Exception:
                return False
        return False

    def _teardown_embedded_workflow_widgets(self) -> None:
        if self._stack_prepare_widget is not None:
            _write_diagnostic_line("Tearing down embedded stack_prepare_widget")
            if hasattr(self._stack_prepare_widget, "prepare_for_close"):
                try:
                    self._stack_prepare_widget.prepare_for_close()
                except Exception:
                    pass
            try:
                self._stack_prepare_widget.hide()
            except Exception:
                pass
            self._stack_prepare_widget = None
        if self._trim_widget is not None:
            _write_diagnostic_line("Tearing down embedded trim_widget")
            try:
                self._trim_widget.hide()
            except Exception:
                pass
            self._trim_widget = None
        if self._selection_widget is not None:
            _write_diagnostic_line("Tearing down embedded selection_widget")
            if hasattr(self._selection_widget, "prepare_for_close"):
                try:
                    self._selection_widget.prepare_for_close()
                except Exception:
                    pass
            try:
                self._selection_widget.hide()
            except Exception:
                pass
            self._selection_widget = None

    def _request_close(self) -> None:
        if self._stack_processing():
            QMessageBox.information(
                self,
                "Processing Active",
                "Wait for the stack process to finish before closing this window.",
            )
            return
        self.reject()

    def reject(self) -> None:
        if self._stack_processing():
            QMessageBox.information(
                self,
                "Processing Active",
                "Wait for the stack process to finish before closing this window.",
            )
            return
        super().reject()

    def accept(self) -> None:
        super().accept()

    def _append_workflow_terminal(self, text) -> None:
        if not hasattr(self, "workflow_terminal") or self.workflow_terminal is None:
            return
        cleaned = str(text or "").rstrip("\n")
        if not cleaned:
            return
        self.workflow_terminal.appendPlainText(cleaned)
        scroll = self.workflow_terminal.verticalScrollBar()
        if scroll is not None:
            scroll.setValue(scroll.maximum())

    def _set_workflow_log_drawer_expanded(self, expanded: bool) -> None:
        self._workflow_log_expanded = bool(expanded)
        if hasattr(self, "workflow_log_drawer") and self.workflow_log_drawer is not None:
            self.workflow_log_drawer.setVisible(self._workflow_log_expanded)
        if hasattr(self, "workflow_log_toggle_button") and self.workflow_log_toggle_button is not None:
            self.workflow_log_toggle_button.setText(
                "Hide log ▴" if self._workflow_log_expanded else "Show log ▾"
            )
        if self._workflow_log_expanded and hasattr(self, "workflow_splitter"):
            self.workflow_splitter.setSizes([620, 180])

    def _toggle_workflow_log_drawer(self) -> None:
        self._set_workflow_log_drawer_expanded(not self._workflow_log_expanded)

    def _go_to_step(self, step_index: int) -> None:
        step = max(0, min(int(step_index), len(self._STEP_TITLES) - 1))
        if step > 0 and not self._step_buttons[step].isEnabled():
            step = 0
        self._current_step = step
        self.step_stack.setCurrentIndex(step)
        for idx, button in enumerate(self._step_buttons):
            is_current = idx == step
            button.setChecked(is_current)
            prefix = "▶ " if is_current else ""
            button.setText(f"{prefix}{idx + 1}. {self._STEP_TITLES[idx]}")
        if step == self._run_step_index():
            self._refresh_run_table()
        self._sync_workflow_process_button()

    def _run_step_index(self) -> int:
        return len(self._STEP_TITLES) - 1

    def _run_preview_visible(self) -> bool:
        if not hasattr(self, "step_stack") or self.step_stack is None:
            return False
        try:
            return int(self.step_stack.currentIndex()) == self._run_step_index()
        except Exception:
            return False

    def _group_from_path(self, path: Path) -> str:
        parts = list(Path(path).parts)
        if "derivatives" in parts:
            idx = parts.index("derivatives")
            if idx > 0:
                return str(parts[idx - 1])
        return str(self._folder_path.name)

    def _path_subject_session(self, path: Path):
        text = Path(path).name
        sub_match = re.search(r"sub-([^_]+)", text)
        ses_match = re.search(r"ses-([^_]+)", text)
        sub = _normalize_subject_token(sub_match.group(1) if sub_match else "")
        ses = _normalize_session_token(ses_match.group(1) if ses_match else "")
        return self._group_from_path(path), sub, ses

    def _path_subject_session_pair(self, path: Path):
        _group, sub, ses = self._path_subject_session(path)
        return sub, ses

    def _covar_row_subject_session(self, covar_row, default_group: str, col_map=None):
        if col_map is None:
            covars_df = self._filtered_covars_df()
            if covars_df is None:
                return default_group, "", ""
            col_map = {str(col).lower(): col for col in covars_df.columns}
        group_col = col_map.get("group")
        sub_col = (
            col_map.get("participant_id")
            or col_map.get("subject_id")
            or col_map.get("subject")
            or col_map.get("sub")
            or col_map.get("id")
        )
        ses_col = col_map.get("session_id") or col_map.get("session") or col_map.get("ses")
        participant_value = covar_row.get(sub_col, "") if sub_col is not None else ""
        session_value = covar_row.get(ses_col, "") if ses_col is not None else ""
        covar_group = str(covar_row.get(group_col, "")).strip() if group_col is not None else ""
        parsed_group, parsed_sub = _parse_participant_id(str(participant_value))
        group = covar_group or parsed_group or default_group
        sub = _normalize_subject_token(str(parsed_sub or participant_value).strip())
        ses = _normalize_session_token(str(session_value).strip())
        return group, sub, ses

    def _allowed_pair_keys_from_covars(self):
        covars_df = self._filtered_covars_df()
        if covars_df is None:
            return None
        if len(covars_df) == 0:
            return set()
        default_group = self._group_from_path(Path(self.selected_paths()[0])) if self.selected_paths() else ""
        col_map = {str(col).lower(): col for col in covars_df.columns}
        keys = set()
        for _, covar_row in covars_df.iterrows():
            _group, sub, ses = self._covar_row_subject_session(covar_row, default_group, col_map=col_map)
            keys.add((sub, ses))
        return keys

    def _pair_matches_path(self, path: Path, sub: str, ses: str) -> bool:
        pair = self._path_subject_session_pair(path)
        return pair == (_normalize_subject_token(sub), _normalize_session_token(ses))

    def _paths_for_subject_session(self, paths, sub: str, ses: str):
        return [Path(path) for path in paths if self._pair_matches_path(Path(path), sub, ses)]

    def _modality_counts_for_paths(self, paths, modalities):
        counts = {modality: 0 for modality in modalities}
        for path in paths:
            modality = _infer_modality_from_path(Path(path))
            if modality in counts:
                counts[modality] += 1
        return counts

    def _modality_paths_for_paths(self, paths, modalities):
        matches = {modality: [] for modality in modalities}
        for path in paths:
            resolved = str(Path(path))
            modality = _infer_modality_from_path(Path(path))
            if modality in matches:
                matches[modality].append(resolved)
        return matches

    def effective_selected_paths(self):
        selected = [Path(path) for path in self.selected_paths()]
        allowed_pairs = self._allowed_pair_keys_from_covars()
        out = []
        for path in selected:
            pair_key = self._path_subject_session_pair(path)
            if allowed_pairs is not None and pair_key not in allowed_pairs:
                continue
            if pair_key in self._multimodal_excluded_pairs:
                continue
            out.append(str(path))
        return out

    def _filtered_covars_df(self):
        if self._selection_widget is not None and hasattr(self._selection_widget, "selected_covars_df"):
            try:
                return self._selection_widget.selected_covars_df()
            except Exception:
                return None
        if self._stack_prepare_widget is None:
            return None
        if not hasattr(self._stack_prepare_widget, "selected_covars_df"):
            return None
        try:
            return self._stack_prepare_widget.selected_covars_df()
        except Exception:
            return None

    def _selection_suffix(self):
        if self._selection_widget is not None and hasattr(self._selection_widget, "current_selection_suffix"):
            try:
                return self._selection_widget.current_selection_suffix()
            except Exception:
                return ("all", "all")
        return ("all", "all")

    def _selected_trim_parcels(self):
        if self._trim_widget is not None and hasattr(self._trim_widget, "selected_trim_parcels"):
            try:
                return self._trim_widget.selected_trim_parcels()
            except Exception:
                pass
        return {"names": [], "labels": [], "key": ""}

    def _harmonization_columns(self):
        if self._selection_widget is not None and hasattr(self._selection_widget, "covars_columns"):
            try:
                return [str(col) for col in self._selection_widget.covars_columns()]
            except Exception:
                return []
        covars_df = self._filtered_covars_df()
        if covars_df is None:
            return []
        return [str(col) for col in covars_df.columns]

    def _harm_default_batch_col(self) -> str:
        if self._selection_widget is not None and hasattr(self._selection_widget, "default_batch_col"):
            try:
                return str(self._selection_widget.default_batch_col() or "")
            except Exception:
                return ""
        columns = self._harmonization_columns()
        preferred = {"scanner", "site", "batch"}
        for name in columns:
            if name.strip().lower() in preferred:
                return name
        return columns[0] if columns else ""

    def _harm_covariate_type(self, covar_name: str) -> str:
        if self._selection_widget is not None and hasattr(self._selection_widget, "covariate_type"):
            try:
                return str(self._selection_widget.covariate_type(covar_name) or "categorical")
            except Exception:
                return "categorical"
        covars_df = self._filtered_covars_df()
        if covars_df is None or covar_name not in covars_df.columns:
            return "categorical"
        values = covars_df[covar_name].tolist()
        return "continuous" if _column_is_numeric(values) else "categorical"

    def _harm_is_id_like(self, covar_name: str) -> bool:
        if self._selection_widget is not None and hasattr(self._selection_widget, "is_id_like"):
            try:
                return bool(self._selection_widget.is_id_like(covar_name))
            except Exception:
                return False
        return str(covar_name).strip().lower() in {
            "participant_id",
            "subject_id",
            "session_id",
            "id",
            "sub",
            "ses",
        }

    def _selected_modalities(self):
        if self._stack_prepare_widget is not None and hasattr(self._stack_prepare_widget, "detected_modalities"):
            try:
                modalities = list(self._stack_prepare_widget.detected_modalities())
            except Exception:
                modalities = []
        else:
            modalities = []
        if not modalities:
            modalities = sorted({_infer_modality_from_path(Path(path)) for path in self.selected_paths() if str(path).strip()})
        return [item for item in modalities if item]

    def _selected_stack_matrix_key(self):
        if not hasattr(self, "matrix_key_combo") or self.matrix_key_combo is None:
            return None
        data = self.matrix_key_combo.currentData()
        text = self.matrix_key_combo.currentText().strip()
        if data is not None:
            return str(data).strip() or None
        if text.lower().startswith("auto "):
            return None
        return text or None

    def _refresh_stack_matrix_key_options(self):
        if not hasattr(self, "matrix_key_combo") or self.matrix_key_combo is None:
            return

        selected_paths = [Path(path) for path in self.selected_paths() if str(path).strip()]
        preferred = _preferred_stack_matrix_key(self._selected_modalities()[0] if self._selected_modalities() else "")
        previous_key = self._selected_stack_matrix_key()
        candidates = _shared_matrix_key_candidates(selected_paths)
        self._stack_matrix_key_candidates = candidates

        self.matrix_key_combo.blockSignals(True)
        self.matrix_key_combo.clear()
        self.matrix_key_combo.addItem("Auto (modality default)", None)
        for entry in candidates:
            label = str(entry["key"])
            shape_text = str(entry.get("shape_text") or "").strip()
            if shape_text:
                label = f"{label} ({shape_text})"
            self.matrix_key_combo.addItem(label, str(entry["key"]))

        selected_index = 0
        available_keys = [str(entry["key"]) for entry in candidates]
        if previous_key and previous_key in available_keys:
            selected_index = available_keys.index(previous_key) + 1
        elif preferred and preferred in available_keys:
            selected_index = available_keys.index(preferred) + 1
        elif available_keys:
            selected_index = 1
        self.matrix_key_combo.setCurrentIndex(selected_index)
        self.matrix_key_combo.blockSignals(False)

        detected_count = len(candidates)
        if not selected_paths:
            self.matrix_key_summary_label.setText("No NPZ files are currently selected for stacking.")
        elif detected_count == 0:
            self.matrix_key_summary_label.setText(
                "No square matrix entries were detected in the first selected NPZ file. "
                "The stack will fall back to the modality default key."
            )
        else:
            detected_text = ", ".join(
                f"{entry['key']} ({entry['shape_text']})" if entry.get("shape_text") else str(entry["key"])
                for entry in candidates
            )
            self.matrix_key_summary_label.setText(
                f"Detected {detected_count} square matrix entr"
                f"{'y' if detected_count == 1 else 'ies'} in the first selected NPZ file "
                f"({selected_paths[0]}): {detected_text}"
            )
        self.matrix_key_group.setVisible(self._selection_widget is not None)

        if self._trim_widget is not None and hasattr(self._trim_widget, "set_matrix_key"):
            self._trim_widget.set_matrix_key(self._selected_stack_matrix_key())
        if self._stack_prepare_widget is not None and hasattr(self._stack_prepare_widget, "set_matrix_key"):
            self._stack_prepare_widget.set_matrix_key(self._selected_stack_matrix_key())

    def _on_stack_matrix_key_changed(self, _index):
        if self._trim_widget is not None and hasattr(self._trim_widget, "set_matrix_key"):
            self._trim_widget.set_matrix_key(self._selected_stack_matrix_key())
        if self._stack_prepare_widget is not None and hasattr(self._stack_prepare_widget, "set_matrix_key"):
            self._stack_prepare_widget.set_matrix_key(self._selected_stack_matrix_key())
        self._sync_workflow_process_button()

    def _ordered_modalities(self):
        modalities = self._selected_modalities()
        reference = self.reference_modality_combo.currentText().strip()
        if reference and reference in modalities:
            return [reference] + [item for item in modalities if item != reference]
        return modalities

    def _multimodal_process_config(self):
        if not self.multimodal_enable_check.isChecked():
            return None
        effective_paths = [Path(path) for path in self.effective_selected_paths()]
        ordered_modalities = []
        for modality in self._ordered_modalities():
            if any(_infer_modality_from_path(path) == modality for path in effective_paths):
                ordered_modalities.append(modality)
        if len(ordered_modalities) <= 1:
            return None
        return {
            "enabled": True,
            "ordered_modalities": ordered_modalities,
            "align_cross_modality": bool(self.multimodal_align_check.isChecked()),
        }

    def _build_pair_table_rows(self, paths, modalities=None, include_empty_covar_rows=False):
        selected_paths = [Path(path) for path in (paths or []) if str(path).strip()]
        ordered_modalities = [str(item).strip() for item in (modalities or self._ordered_modalities()) if str(item).strip()]
        if not ordered_modalities:
            ordered_modalities = sorted(
                {
                    _infer_modality_from_path(path)
                    for path in selected_paths
                    if str(_infer_modality_from_path(path)).strip()
                }
            )

        rows = []
        extra_columns = []
        covars_df = self._filtered_covars_df()
        if covars_df is not None and len(covars_df) > 0:
            covar_columns = [str(col) for col in covars_df.columns]
            col_map = {str(col).lower(): col for col in covars_df.columns}
            group_col = col_map.get("group")
            sub_col = (
                col_map.get("participant_id")
                or col_map.get("subject_id")
                or col_map.get("subject")
                or col_map.get("sub")
                or col_map.get("id")
            )
            ses_col = col_map.get("session_id") or col_map.get("session") or col_map.get("ses")
            hidden_cols = {group_col, sub_col, ses_col}
            extra_columns = [col for col in covar_columns if col not in hidden_cols and col is not None]
            default_group = self._group_from_path(selected_paths[0]) if selected_paths else ""
            for _, covar_row in covars_df.iterrows():
                group, sub, ses = self._covar_row_subject_session(covar_row, default_group, col_map=col_map)
                key = (sub, ses)
                matched_paths = self._paths_for_subject_session(selected_paths, sub, ses)
                if not matched_paths and not include_empty_covar_rows:
                    continue
                counts = self._modality_counts_for_paths(matched_paths, ordered_modalities)
                matched_by_modality = self._modality_paths_for_paths(matched_paths, ordered_modalities)
                row = {
                    "group": group,
                    "sub": sub,
                    "ses": ses,
                    "_key": key,
                    "_counts": counts,
                    "_paths_by_modality": matched_by_modality,
                    "_match_count": len(matched_paths),
                }
                for modality in ordered_modalities:
                    row[modality] = counts.get(modality, 0)
                for column in extra_columns:
                    row[column] = str(covar_row[column])
                rows.append(row)
        else:
            pair_rows = {}
            for path in selected_paths:
                group, sub, ses = self._path_subject_session(path)
                key = (sub, ses)
                if key not in pair_rows:
                    pair_rows[key] = {
                        "group": group,
                        "sub": sub,
                        "ses": ses,
                        "_key": key,
                    }
            for key in sorted(pair_rows.keys()):
                row = dict(pair_rows[key])
                matched_paths = self._paths_for_subject_session(selected_paths, row["sub"], row["ses"])
                counts = self._modality_counts_for_paths(matched_paths, ordered_modalities)
                matched_by_modality = self._modality_paths_for_paths(matched_paths, ordered_modalities)
                row["_counts"] = counts
                row["_paths_by_modality"] = matched_by_modality
                row["_match_count"] = len(matched_paths)
                for modality in ordered_modalities:
                    row[modality] = counts.get(modality, 0)
                rows.append(row)
        return rows, ordered_modalities, extra_columns

    def _populate_pair_table(self, table, rows, modalities, extra_columns, include_exclude_column=False):
        table.blockSignals(True)
        table.clear()
        headers = ["group", "sub", "ses"] + list(modalities) + list(extra_columns)
        if include_exclude_column:
            headers = ["Exclude"] + headers
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(rows))

        duplicate_entries = []
        editable_flag = getattr(Qt, "ItemIsEditable", getattr(Qt.ItemFlag, "ItemIsEditable"))
        align_center = getattr(Qt, "AlignCenter", getattr(Qt.AlignmentFlag, "AlignCenter"))
        data_offset = 0
        if include_exclude_column:
            data_offset = 1

        for row_idx, row_data in enumerate(rows):
            pair_key = row_data.get("_key")
            if include_exclude_column:
                exclude_item = QTableWidgetItem("")
                exclude_item.setData(USER_ROLE, pair_key)
                exclude_item.setFlags((exclude_item.flags() | _is_user_checkable_flag()) & ~editable_flag)
                exclude_item.setCheckState(
                    Qt.Checked if pair_key in self._multimodal_excluded_pairs else Qt.Unchecked
                )
                table.setItem(row_idx, 0, exclude_item)

            for col_idx, header in enumerate(headers[data_offset:], start=data_offset):
                if header in modalities:
                    count = int(row_data.get(header, 0) or 0)
                    modality_paths = list(row_data.get("_paths_by_modality", {}).get(header, []))
                    if count <= 0:
                        item = QTableWidgetItem("✗")
                        item.setToolTip("Missing")
                        item.setForeground(QColor("#dc2626"))
                    elif count == 1:
                        item = QTableWidgetItem("✓")
                        item.setToolTip(modality_paths[0] if modality_paths else "Present")
                        item.setForeground(QColor("#16a34a"))
                    else:
                        item = QTableWidgetItem(f"o {count}")
                        item.setToolTip(
                            "\n".join(modality_paths)
                            if modality_paths
                            else f"{count} matching NPZ files detected"
                        )
                        item.setForeground(QColor("#ca8a04"))
                        duplicate_entries.append(
                            {
                                "pair": pair_key,
                                "modality": header,
                                "count": count,
                            }
                        )
                    item.setTextAlignment(int(align_center))
                else:
                    item = QTableWidgetItem(str(row_data.get(header, "")))
                item.setFlags(item.flags() & ~editable_flag)
                table.setItem(row_idx, col_idx, item)

        header = table.horizontalHeader()
        fixed_headers = {"group", "sub", "ses", *modalities}
        if include_exclude_column:
            fixed_headers.add("Exclude")
        if QT_LIB == 6:
            for col_idx, header_name in enumerate(headers):
                mode = (
                    QHeaderView.ResizeMode.Stretch
                    if header_name not in fixed_headers
                    else QHeaderView.ResizeMode.ResizeToContents
                )
                header.setSectionResizeMode(col_idx, mode)
        else:
            for col_idx, header_name in enumerate(headers):
                mode = QHeaderView.Stretch if header_name not in fixed_headers else QHeaderView.ResizeToContents
                header.setSectionResizeMode(col_idx, mode)
        table.blockSignals(False)
        return duplicate_entries

    def _refresh_run_table(self):
        if not hasattr(self, "run_table") or self.run_table is None:
            return

        ready = self._stack_prepare_widget is not None and self._selection_widget is not None
        if not ready:
            self.run_table.clear()
            self.run_table.setRowCount(0)
            self.run_table.setColumnCount(0)
            self.run_summary_label.setText(
                "The stack preview will appear here after Selection, Trim, and Setup are ready."
            )
            return

        effective_paths = [Path(path) for path in self.effective_selected_paths()]
        rows, modalities, extra_columns = self._build_pair_table_rows(
            effective_paths,
            modalities=self._ordered_modalities(),
            include_empty_covar_rows=False,
        )
        duplicate_entries = self._populate_pair_table(
            self.run_table,
            rows,
            modalities,
            extra_columns,
            include_exclude_column=False,
        )
        duplicate_pair_count = len({tuple(entry["pair"]) for entry in duplicate_entries})
        validation_error = self.multimodal_validation_error()
        if not effective_paths:
            self.run_summary_label.setText(
                "No NPZ files remain after the current covariate and multimodal filters."
            )
            return
        if validation_error:
            self.run_summary_label.setText(
                f"{len(rows)} subject/session pairs will be stacked from {len(effective_paths)} NPZ file(s). "
                f"Processing is blocked: {validation_error}"
            )
            return
        if duplicate_pair_count:
            self.run_summary_label.setText(
                f"{len(rows)} subject/session pairs will be stacked from {len(effective_paths)} NPZ file(s). "
                f"Duplicate NPZ files remain for {duplicate_pair_count} sub/ses pair(s)."
            )
            return
        self.run_summary_label.setText(
            f"{len(rows)} subject/session pairs will be stacked from {len(effective_paths)} NPZ file(s)."
        )

    def _update_multimodal_controls(self):
        modalities = self._selected_modalities()
        has_multiple = len(modalities) > 1
        enabled = has_multiple and self.multimodal_enable_check.isChecked()
        self.reference_modality_combo.setEnabled(enabled)
        self.multimodal_align_check.setEnabled(enabled)
        self.multimodal_table.setVisible(enabled)
        if not has_multiple:
            self.multimodal_enable_check.setChecked(False)
            self.multimodal_enable_check.setEnabled(False)
            self.multimodal_align_check.setChecked(False)
            self.multimodal_align_check.setEnabled(False)
            self.multimodal_summary_label.setText(
                "Multimodal stacking becomes available when more than one modality is detected in the selected NPZ files."
            )
        else:
            self.multimodal_enable_check.setEnabled(True)
            self.multimodal_summary_label.setText(
                "Optional step. Enable multimodal stacking to review subject/session coverage across modalities."
            )
        self.reference_modality_combo.setVisible(has_multiple)
        self._refresh_multimodal_table()
        if self._stack_prepare_widget is not None and hasattr(self._stack_prepare_widget, "refresh_process_state"):
            self._stack_prepare_widget.refresh_process_state()
        if self._run_preview_visible():
            self._refresh_run_table()

    def _refresh_optional_steps(self):
        modalities = self._selected_modalities()
        current_reference = self.reference_modality_combo.currentText().strip()
        self.reference_modality_combo.blockSignals(True)
        self.reference_modality_combo.clear()
        self.reference_modality_combo.addItems(modalities)
        if current_reference and current_reference in modalities:
            self.reference_modality_combo.setCurrentText(current_reference)
        self.reference_modality_combo.blockSignals(False)
        ready = self._stack_prepare_widget is not None and self._selection_widget is not None
        self._refresh_stack_matrix_key_options()
        for idx in range(1, len(self._STEP_TITLES)):
            self._step_buttons[idx].setEnabled(ready)
        self._update_multimodal_controls()
        if self._run_preview_visible():
            self._refresh_run_table()
        self._sync_workflow_process_button()

    def _schedule_refresh_optional_steps(self):
        if self._optional_steps_refresh_pending:
            return
        self._optional_steps_refresh_pending = True

        def _run_refresh():
            self._optional_steps_refresh_pending = False
            self._refresh_optional_steps()

        try:
            QTimer.singleShot(0, _run_refresh)
        except Exception:
            self._optional_steps_refresh_pending = False
            self._refresh_optional_steps()

    def _refresh_multimodal_table(self):
        self._multimodal_table_refreshing = True
        self._multimodal_duplicate_entries = []
        if not self.multimodal_enable_check.isChecked():
            self.multimodal_table.setRowCount(0)
            self.multimodal_table.setColumnCount(0)
            self._multimodal_table_refreshing = False
            return

        selected_paths = [Path(path) for path in self.selected_paths()]
        modalities = self._ordered_modalities()
        if len(modalities) <= 1:
            self.multimodal_table.setRowCount(0)
            self.multimodal_table.setColumnCount(0)
            self._multimodal_table_refreshing = False
            return

        rows, modalities, extra_columns = self._build_pair_table_rows(
            selected_paths,
            modalities=modalities,
            include_empty_covar_rows=True,
        )
        self._multimodal_duplicate_entries = self._populate_pair_table(
            self.multimodal_table,
            rows,
            modalities,
            extra_columns,
            include_exclude_column=True,
        )
        excluded_count = len([row for row in rows if row.get("_key") in self._multimodal_excluded_pairs])
        duplicate_pair_count = len({tuple(entry["pair"]) for entry in self._multimodal_duplicate_entries})
        if duplicate_pair_count:
            self.multimodal_summary_label.setText(
                "Optional step. "
                f"{len(rows)} subject/session pairs shown. Excluded: {excluded_count}. "
                f"Warning: duplicate NPZ files detected for {duplicate_pair_count} sub/ses pair(s). "
                "Processing is blocked until duplicates are removed or excluded."
            )
        else:
            self.multimodal_summary_label.setText(
                f"Optional step. {len(rows)} subject/session pairs shown. Excluded: {excluded_count}."
            )
        self._multimodal_table_refreshing = False

    def _on_multimodal_table_item_changed(self, item):
        if item is None or self._multimodal_table_refreshing or item.column() != 0:
            return
        pair_key = item.data(USER_ROLE)
        if not pair_key:
            return
        if item.checkState() == Qt.Checked:
            self._multimodal_excluded_pairs.add(tuple(pair_key))
        else:
            self._multimodal_excluded_pairs.discard(tuple(pair_key))
        self._refresh_multimodal_table()
        if self._run_preview_visible():
            self._refresh_run_table()
        if self._stack_prepare_widget is not None and hasattr(self._stack_prepare_widget, "refresh_process_state"):
            self._stack_prepare_widget.refresh_process_state()

    def multimodal_validation_error(self):
        if not self.multimodal_enable_check.isChecked():
            return ""
        duplicate_pair_count = len({tuple(entry["pair"]) for entry in self._multimodal_duplicate_entries})
        if duplicate_pair_count:
            return (
                f"Multiple NPZ files were found for {duplicate_pair_count} sub/ses pair(s) in the Multimodal step. "
                "Remove or exclude the duplicates before processing."
            )
        return ""

    def _sync_workflow_process_button(self):
        can_process = False
        can_import = False
        processing = self._stack_processing()
        if self._stack_prepare_widget is not None:
            if hasattr(self._stack_prepare_widget, "can_process"):
                try:
                    can_process = bool(self._stack_prepare_widget.can_process())
                except Exception:
                    can_process = False
            elif hasattr(self._stack_prepare_widget, "process_button"):
                try:
                    can_process = bool(self._stack_prepare_widget.process_button.isEnabled())
                except Exception:
                    can_process = False
            if hasattr(self._stack_prepare_widget, "can_import_last_output"):
                try:
                    can_import = bool(self._stack_prepare_widget.can_import_last_output())
                except Exception:
                    can_import = False
            elif hasattr(self._stack_prepare_widget, "import_button"):
                try:
                    can_import = bool(self._stack_prepare_widget.import_button.isEnabled())
                except Exception:
                    can_import = False
        if hasattr(self, "run_import_button") and self.run_import_button is not None:
            self.run_import_button.setEnabled(can_import)
        if hasattr(self, "run_process_button") and self.run_process_button is not None:
            self.run_process_button.setEnabled(can_process)
        if hasattr(self, "run_close_button") and self.run_close_button is not None:
            self.run_close_button.setEnabled(not processing)

    def _trigger_workflow_process(self):
        if self._stack_prepare_widget is None:
            return
        if hasattr(self._stack_prepare_widget, "trigger_process"):
            self._stack_prepare_widget.trigger_process()
        else:
            self._stack_prepare_widget._process()
        self._sync_workflow_process_button()

    def _trigger_workflow_import(self):
        if self._stack_prepare_widget is None:
            return
        if hasattr(self._stack_prepare_widget, "trigger_import_last_output"):
            self._stack_prepare_widget.trigger_import_last_output()
        else:
            self._stack_prepare_widget._import_last_output()
        self._sync_workflow_process_button()

    def set_theme(self, theme_name: str) -> None:
        theme = str(theme_name or "Dark").strip().title()
        if theme not in {"Light", "Dark", "Teya", "Donald"}:
            theme = "Dark"
        if theme == "Dark":
            style = (
                "QWidget { background: #1f2430; color: #e5e7eb; font-size: 11pt; } "
                "QPushButton, QComboBox, QLineEdit, QListWidget, QTableWidget { "
                "background: #2a3140; color: #e5e7eb; border: 1px solid #556070; border-radius: 5px; } "
                "QPushButton { min-height: 30px; padding: 4px 10px; } "
                "QPushButton:hover { background: #344054; } "
                "QPushButton#workflowStepButton { text-align: left; padding-left: 10px; } "
                "QPushButton#workflowStepButton:checked { background: #2563eb; border: 2px solid #60a5fa; color: #ffffff; font-weight: 600; } "
                "QLineEdit, QComboBox { min-height: 30px; padding: 2px 4px; } "
                "QListWidget::item:selected, QTableWidget::item:selected { background: #3b82f6; color: #ffffff; }"
            )
        elif theme == "Teya":
            style = (
                "QWidget { background: #ffd0e5; color: #0b7f7a; font-size: 11pt; } "
                "QPushButton, QComboBox, QLineEdit, QListWidget, QTableWidget { "
                "background: #ffe6f1; color: #0b7f7a; border: 1px solid #1db8b2; border-radius: 5px; } "
                "QPushButton { min-height: 30px; padding: 4px 10px; } "
                "QPushButton:hover { background: #ffd9ea; } "
                "QPushButton#workflowStepButton { text-align: left; padding-left: 10px; } "
                "QPushButton#workflowStepButton:checked { background: #2ecfc9; border: 2px solid #0b7f7a; color: #073f3c; font-weight: 700; } "
                "QLineEdit, QComboBox { min-height: 30px; padding: 2px 4px; } "
                "QListWidget::item:selected, QTableWidget::item:selected { background: #2ecfc9; color: #073f3c; }"
            )
        elif theme == "Donald":
            style = (
                "QWidget { background: #d97706; color: #ffffff; font-size: 11pt; } "
                "QPushButton, QComboBox, QLineEdit, QListWidget, QTableWidget { "
                "background: #c96a04; color: #ffffff; border: 1px solid #f3a451; border-radius: 5px; } "
                "QPushButton { min-height: 30px; padding: 4px 10px; } "
                "QPushButton:hover { background: #c76b06; } "
                "QPushButton#workflowStepButton { text-align: left; padding-left: 10px; } "
                "QPushButton#workflowStepButton:checked { background: #b85f00; border: 2px solid #ffd19e; color: #ffffff; font-weight: 700; } "
                "QLineEdit, QComboBox { min-height: 30px; padding: 2px 4px; } "
                "QListWidget::item:selected, QTableWidget::item:selected { background: #2563eb; color: #ffffff; }"
            )
        else:
            style = (
                "QWidget { background: #f4f6f9; color: #1f2937; font-size: 11pt; } "
                "QPushButton, QComboBox, QLineEdit, QListWidget, QTableWidget { "
                "background: #ffffff; color: #1f2937; border: 1px solid #c9d0da; border-radius: 5px; } "
                "QPushButton { min-height: 30px; padding: 4px 10px; } "
                "QPushButton:hover { background: #edf2f7; } "
                "QPushButton#workflowStepButton { text-align: left; padding-left: 10px; } "
                "QPushButton#workflowStepButton:checked { background: #2563eb; border: 2px solid #1d4ed8; color: #ffffff; font-weight: 600; } "
                "QLineEdit, QComboBox { min-height: 30px; padding: 2px 4px; } "
                "QListWidget::item:selected, QTableWidget::item:selected { background: #2563eb; color: #ffffff; }"
            )
        self.setStyleSheet(style)

    def _accept_if_selection(self) -> None:
        if not self.selected_paths():
            QMessageBox.information(self, "No Files Selected", "Select at least one matrix to add.")
            return
        self._requested_action = "add"
        self.accept()

    def _open_stack_prepare(self) -> None:
        selected = self.selected_paths()
        if not selected:
            QMessageBox.information(self, "No Files Selected", "Select at least one matrix to stack.")
            return
        selected_atlases = sorted({_connectivity_atlas_tag(Path(path)) for path in selected})
        if len(selected_atlases) > 1:
            QMessageBox.warning(
                self,
                "Mixed Atlases",
                "Stack requires a single atlas selection.\n\n"
                f"Detected atlases: {', '.join(selected_atlases)}\n\n"
                "Use the text filter or a narrower modality selection before stacking.",
            )
            return
        try:
            from window.stack_prepare import CovarsSelectionWidget, ParcelTrimWidget, StackPrepareDialog
        except Exception:
            try:
                from mrsi_viewer.window.stack_prepare import CovarsSelectionWidget, ParcelTrimWidget, StackPrepareDialog
            except Exception as exc:
                QMessageBox.warning(self, "Stack Unavailable", f"Failed to open stack step: {exc}")
                return

        if self._selection_widget is None:
            self._selection_widget = CovarsSelectionWidget(
                selected_paths=selected,
                parent=self.selection_page,
            )
            _write_diagnostic_line("Created embedded selection_widget with parent=selection_page")
            try:
                self._selection_widget.destroyed.connect(
                    lambda *_args: _write_diagnostic_line("embedded selection_widget destroyed")
                )
            except Exception:
                pass
            self._selection_widget.configuration_changed.connect(self._schedule_refresh_optional_steps)
            self.selection_tab_layout.removeWidget(self.selection_placeholder)
            self.selection_placeholder.hide()
            self.selection_tab_layout.addWidget(self._selection_widget, 1)
        else:
            self._selection_widget.set_selected_paths(selected)
        self._refresh_stack_matrix_key_options()

        if self._trim_widget is None:
            self._trim_widget = ParcelTrimWidget(
                selected_paths=selected,
                matrix_key=self._selected_stack_matrix_key(),
                parent=self.trim_page,
            )
            _write_diagnostic_line("Created embedded trim_widget with parent=trim_page")
            try:
                self._trim_widget.destroyed.connect(
                    lambda *_args: _write_diagnostic_line("embedded trim_widget destroyed")
                )
            except Exception:
                pass
            if hasattr(self._trim_widget, "configuration_changed"):
                self._trim_widget.configuration_changed.connect(self._schedule_refresh_optional_steps)
            self.trim_tab_layout.removeWidget(self.trim_placeholder)
            self.trim_placeholder.hide()
            self.trim_tab_layout.insertWidget(0, self._trim_widget, 1)
        else:
            self._trim_widget.set_selected_paths(selected)
            if hasattr(self._trim_widget, "set_matrix_key"):
                self._trim_widget.set_matrix_key(self._selected_stack_matrix_key())
            self._trim_widget.show()

        if self._stack_prepare_widget is None:
            if hasattr(self, "workflow_terminal") and self.workflow_terminal is not None:
                self.workflow_terminal.clear()
            self._stack_prepare_widget = StackPrepareDialog(
                selected_paths=selected,
                theme_name=getattr(self.parent(), "_theme_name", "Dark"),
                export_callback=self._export_callback,
                close_callback=lambda: self._go_to_step(1),
                selected_paths_provider=self.effective_selected_paths,
                validation_error_provider=self.multimodal_validation_error,
                covars_df_provider=self._filtered_covars_df,
                selection_suffix_provider=self._selection_suffix,
                multimodal_config_provider=self._multimodal_process_config,
                include_covars_widget=False,
                include_trim_widget=False,
                trim_parcels_provider=self._selected_trim_parcels,
                show_embedded_terminal=False,
                show_import_button=False,
                show_process_button=False,
                default_results_dir=getattr(self.parent(), "_results_dir_default", ""),
                default_bids_dir=getattr(self.parent(), "_bids_dir_default", ""),
                default_atlas_dir=getattr(self.parent(), "_atlas_dir_default", ""),
                parent=self.setup_page,
            )
            _write_diagnostic_line("Created embedded stack_prepare_widget with parent=setup_page")
            try:
                self._stack_prepare_widget.destroyed.connect(
                    lambda *_args: _write_diagnostic_line("embedded stack_prepare_widget destroyed")
                )
            except Exception:
                pass
            if hasattr(self._stack_prepare_widget, "configuration_changed"):
                self._stack_prepare_widget.configuration_changed.connect(self._schedule_refresh_optional_steps)
            if hasattr(self._stack_prepare_widget, "process_state_changed"):
                self._stack_prepare_widget.process_state_changed.connect(self._sync_workflow_process_button)
            if hasattr(self._stack_prepare_widget, "log_message_emitted"):
                self._stack_prepare_widget.log_message_emitted.connect(self._append_workflow_terminal)
            self.setup_tab_layout.removeWidget(self.setup_placeholder)
            self.setup_placeholder.hide()
            self.setup_tab_layout.insertWidget(0, self._stack_prepare_widget, 1)
        else:
            self._stack_prepare_widget.set_selected_paths(selected)
            self._stack_prepare_widget.show()
        if hasattr(self._stack_prepare_widget, "set_matrix_key"):
            self._stack_prepare_widget.set_matrix_key(self._selected_stack_matrix_key())
        if hasattr(self._stack_prepare_widget, "refresh_process_state"):
            self._stack_prepare_widget.refresh_process_state()
        self._refresh_optional_steps()
        self._go_to_step(1)
