#!/usr/bin/env python3
"""NBS preparation dialog for covariate role selection and row filtering."""

import json
import re
import sys
from pathlib import Path

import numpy as np

try:
    from PyQt6.QtCore import QProcess, Qt
    from PyQt6.QtWidgets import (
        QAbstractItemView,
        QCheckBox,
        QComboBox,
        QDialog,
        QDoubleSpinBox,
        QFileDialog,
        QGridLayout,
        QGroupBox,
        QHeaderView,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QScrollArea,
        QSpinBox,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
    QT_LIB = 6
except Exception:
    from PyQt5.QtCore import QProcess, Qt
    from PyQt5.QtWidgets import (
        QAbstractItemView,
        QCheckBox,
        QComboBox,
        QDialog,
        QDoubleSpinBox,
        QFileDialog,
        QGridLayout,
        QGroupBox,
        QHeaderView,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPushButton,
        QScrollArea,
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

    ROLE_OPTIONS = ("None", "Nuisance", "Regressor")

    def __init__(self, covars_info, source_path, matrix_key, parent=None):
        super().__init__(parent)
        self._source_path = Path(source_path)
        self._matrix_key = str(matrix_key)
        self._columns, self._rows = _covars_to_rows(covars_info)
        self._filtered_indices = list(range(len(self._rows)))
        self._covar_checks = {}
        self._role_combos = {}
        self._last_run_payload = None
        self._run_process = None
        self._run_output_tail = []

        self.setWindowTitle("NBS Prepare")
        self.resize(1200, 800)
        self.setStyleSheet(
            "QWidget { font-size: 11pt; } "
            "QPushButton { min-height: 30px; } "
            "QComboBox { min-height: 30px; } "
            "QLineEdit { min-height: 30px; }"
        )
        self._build_ui()
        self._refresh_table()
        self._update_test_options()
        self._update_run_state()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        header = QLabel(
            f"File: {self._source_path.name} | Key: {self._matrix_key} | "
            f"Rows: {len(self._rows)} | Covariates: {len(self._columns)}"
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        roles_group = QGroupBox("Covariate Roles")
        roles_layout = QVBoxLayout(roles_group)
        roles_help = QLabel(
            "Tick covariates to include, then set each role. "
            "Only one covariate can be set as Regressor."
        )
        roles_help.setWordWrap(True)
        roles_layout.addWidget(roles_help)

        role_scroll = QScrollArea()
        role_scroll.setWidgetResizable(True)
        role_container = QWidget()
        role_grid = QGridLayout(role_container)
        role_grid.addWidget(QLabel("Use Covariate"), 0, 0)
        role_grid.addWidget(QLabel("Role"), 0, 1)

        for row_idx, covar_name in enumerate(self._columns, start=1):
            check = QCheckBox(covar_name)
            check.toggled.connect(
                lambda checked, name=covar_name: self._on_covar_toggled(name, checked)
            )
            role = QComboBox()
            role.addItems(self.ROLE_OPTIONS)
            role.setCurrentText("None")
            role.setEnabled(False)
            role.currentTextChanged.connect(
                lambda value, name=covar_name: self._on_role_changed(name, value)
            )
            role_grid.addWidget(check, row_idx, 0)
            role_grid.addWidget(role, row_idx, 1)
            self._covar_checks[covar_name] = check
            self._role_combos[covar_name] = role

        role_grid.setColumnStretch(0, 1)
        role_scroll.setWidget(role_container)
        roles_layout.addWidget(role_scroll)
        layout.addWidget(roles_group, 1)

        run_group = QGroupBox("Run Settings")
        run_grid = QGridLayout(run_group)
        run_grid.addWidget(QLabel("Threads"), 0, 0)
        self.nthreads_spin = QSpinBox()
        self.nthreads_spin.setRange(1, 256)
        self.nthreads_spin.setValue(28)
        run_grid.addWidget(self.nthreads_spin, 0, 1)

        run_grid.addWidget(QLabel("Permutations"), 0, 2)
        self.nperm_spin = QSpinBox()
        self.nperm_spin.setRange(100, 500000)
        self.nperm_spin.setSingleStep(500)
        self.nperm_spin.setValue(5000)
        run_grid.addWidget(self.nperm_spin, 0, 3)

        run_grid.addWidget(QLabel("T threshold"), 0, 4)
        self.t_thresh_spin = QDoubleSpinBox()
        self.t_thresh_spin.setRange(0.1, 100.0)
        self.t_thresh_spin.setDecimals(3)
        self.t_thresh_spin.setSingleStep(0.1)
        self.t_thresh_spin.setValue(3.5)
        run_grid.addWidget(self.t_thresh_spin, 0, 5)

        run_grid.addWidget(QLabel("MATLAB cmd"), 1, 0)
        self.matlab_cmd_edit = QLineEdit("matlab")
        run_grid.addWidget(self.matlab_cmd_edit, 1, 1, 1, 2)

        run_grid.addWidget(QLabel("NBS path"), 1, 3)
        self.matlab_nbs_path_edit = QLineEdit("/home/flucchetti/Connectome/Dev/NBS1.2")
        run_grid.addWidget(self.matlab_nbs_path_edit, 1, 4, 1, 2)
        layout.addWidget(run_group)

        filter_group = QGroupBox("Filter Rows")
        filter_layout = QHBoxLayout(filter_group)
        filter_layout.addWidget(QLabel("Covariate:"))
        self.filter_covar_combo = QComboBox()
        self.filter_covar_combo.addItems(self._columns)
        filter_layout.addWidget(self.filter_covar_combo)
        filter_layout.addWidget(QLabel("Value:"))
        self.filter_value_edit = QLineEdit("")
        self.filter_value_edit.setPlaceholderText("e.g. 0,1")
        filter_layout.addWidget(self.filter_value_edit, 1)
        self.filter_button = QPushButton("Filter")
        self.filter_button.clicked.connect(self._apply_filter)
        filter_layout.addWidget(self.filter_button)
        self.filter_reset_button = QPushButton("Reset")
        self.filter_reset_button.clicked.connect(self._reset_filter)
        filter_layout.addWidget(self.filter_reset_button)
        layout.addWidget(filter_group)

        self.table = QTableWidget()
        self.table.setColumnCount(len(self._columns))
        self.table.setHorizontalHeaderLabels(self._columns)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(False)
        if QT_LIB == 6:
            self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
            self.table.horizontalHeader().setSectionResizeMode(
                QHeaderView.ResizeMode.Stretch
            )
        else:
            self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table, 3)

        footer = QHBoxLayout()
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        footer.addWidget(self.status_label, 1)
        layout.addLayout(footer)

        actions = QHBoxLayout()
        actions.addWidget(QLabel("Test:"))
        self.test_combo = QComboBox()
        self.test_combo.setMinimumWidth(160)
        actions.addWidget(self.test_combo)
        actions.addStretch(1)
        self.save_button = QPushButton("Save")
        self.save_button.setMinimumHeight(44)
        self.save_button.setMinimumWidth(140)
        self.save_button.clicked.connect(self._save_configuration)
        actions.addWidget(self.save_button)
        self.run_button = QPushButton("Run")
        self.run_button.setMinimumHeight(44)
        self.run_button.setMinimumWidth(140)
        self.run_button.clicked.connect(self._run_configuration)
        actions.addWidget(self.run_button)
        close_button = QPushButton("Close")
        close_button.setMinimumHeight(44)
        close_button.clicked.connect(self.close)
        actions.addWidget(close_button)
        layout.addLayout(actions)

    def _set_status(self, text):
        self.status_label.setText(str(text))

    def _process_is_running(self):
        return (
            self._run_process is not None
            and self._run_process.state() != _qprocess_not_running()
        )

    def _set_run_busy(self, busy):
        self.run_button.setEnabled(not busy)
        self.save_button.setEnabled(not busy)
        self.run_button.setText("Running..." if busy else "Run")

    def _regressor_name(self):
        return self.selected_covariates().get("regressor")

    def _regressor_classes(self):
        regressor = self._regressor_name()
        if not regressor:
            return []
        classes = []
        for idx in self._filtered_indices:
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

    def _update_run_state(self):
        classes = self._regressor_classes()
        self.run_button.setEnabled(len(classes) >= 2 and not self._process_is_running())

    @staticmethod
    def _parse_filter_values(text):
        values = [token.strip() for token in str(text).split(",")]
        return [token for token in values if token != ""]

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
        role_combo = self._role_combos[covar_name]
        role_combo.setEnabled(bool(checked))
        if not checked:
            role_combo.blockSignals(True)
            role_combo.setCurrentText("None")
            role_combo.blockSignals(False)
        selected = self.selected_covariates()["selected_columns"]
        if selected:
            self._set_status("Selected: " + ", ".join(selected))
        else:
            self._set_status("Selected: none")
        self._update_test_options()
        self._update_run_state()

    def _on_role_changed(self, covar_name, role_value):
        if role_value != "Regressor":
            self._set_status(
                "Roles updated."
                if self.selected_covariates()["selected_columns"]
                else "Selected: none"
            )
            self._update_test_options()
            self._update_run_state()
            return

        for other_name, other_combo in self._role_combos.items():
            if other_name == covar_name:
                continue
            if not self._covar_checks[other_name].isChecked():
                continue
            if other_combo.currentText() == "Regressor":
                current = self._role_combos[covar_name]
                current.blockSignals(True)
                current.setCurrentText("None")
                current.blockSignals(False)
                self._set_status("Only one covariate can be set as Regressor.")
                self._update_test_options()
                self._update_run_state()
                return
        self._set_status(f"Regressor set to: {covar_name}")
        self._update_test_options()
        self._update_run_state()

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
                filter_targets = [float(token) for token in target_values]
            except Exception:
                self._set_status(
                    "Selected covariate is numeric. Use comma-separated numeric values."
                )
                return
        else:
            filter_targets = target_values
        matched = []
        for row_idx, value in enumerate(values):
            if self._matches_any(value, filter_targets, numeric):
                matched.append(row_idx)
        self._filtered_indices = matched
        self._refresh_table()
        self._update_test_options()
        self._update_run_state()
        self._set_status(
            f"Filter applied: {covar_name} in [{', '.join(target_values)}]. "
            f"Showing {len(self._filtered_indices)}/{len(self._rows)} rows."
        )

    def _reset_filter(self):
        self._filtered_indices = list(range(len(self._rows)))
        self._refresh_table()
        self._update_test_options()
        self._update_run_state()
        self._set_status(f"Filter reset. Showing {len(self._filtered_indices)} rows.")

    def _refresh_table(self):
        self.table.setRowCount(len(self._filtered_indices))
        editable_flag = _is_editable_flag()
        for table_row, source_idx in enumerate(self._filtered_indices):
            row_data = self._rows[source_idx]
            for col_idx, covar_name in enumerate(self._columns):
                item = QTableWidgetItem(_display_text(row_data.get(covar_name)))
                item.setFlags(item.flags() & ~editable_flag)
                self.table.setItem(table_row, col_idx, item)
        self.table.resizeColumnsToContents()
        self.table.resizeRowsToContents()
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
        for idx in self._filtered_indices:
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
            "matlab_cmd": self.matlab_cmd_edit.text().strip(),
            "matlab_nbs_path": self.matlab_nbs_path_edit.text().strip(),
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
        indices = np.asarray(self._filtered_indices, dtype=int)
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
            modality = _display_text(npz["modality"]) if "modality" in npz else "modality"

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

        filter_covar = self.filter_covar_combo.currentText().strip()
        filter_values = self._parse_filter_values(self.filter_value_edit.text())
        if filter_covar and filter_values:
            for value in filter_values:
                command += ["--select", f"{filter_covar},{value}"]

        if matlab_test == "t":
            command += ["--contrast", "b"]

        matlab_cmd = self.matlab_cmd_edit.text().strip()
        if matlab_cmd:
            command += ["--matlab-cmd", matlab_cmd]
        matlab_nbs_path = self.matlab_nbs_path_edit.text().strip()
        if matlab_nbs_path:
            command += ["--matlab-nbs-path", matlab_nbs_path]
        return command

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
        if self._run_output_tail:
            self._run_output_tail = self._run_output_tail[-8:]
            self._set_status(self._run_output_tail[-1])

    def _on_process_finished(self, exit_code, _exit_status):
        self._set_run_busy(False)
        self._update_run_state()
        if exit_code == 0:
            message = "NBS run completed successfully."
            if self._last_run_payload and self._last_run_payload.get("conn_subset_path"):
                subset_name = Path(self._last_run_payload["conn_subset_path"]).name
                message = f"NBS run completed successfully. Input: {subset_name}"
            self._set_status(message)
        else:
            tail = self._run_output_tail[-1] if self._run_output_tail else "See terminal output."
            self._set_status(f"NBS run failed (exit {exit_code}). {tail}")
        self._run_process = None

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
        self._run_process = QProcess(self)
        self._run_process.readyReadStandardOutput.connect(self._on_process_output)
        self._run_process.readyReadStandardError.connect(self._on_process_output)
        self._run_process.finished.connect(self._on_process_finished)
        self._run_process.setWorkingDirectory(str(Path(__file__).resolve().parents[1]))

        python_exe = sys.executable or "python3"
        self._set_run_busy(True)
        self._set_status(
            f"Launching NBS ({selected_test}) with {len(self._filtered_indices)} rows..."
        )
        self._run_process.start(python_exe, command)
        if not self._run_process.waitForStarted(3000):
            self._set_run_busy(False)
            self._set_status(
                "Failed to start run_nbs.py process. Check Python executable/path."
            )
            self._run_process = None
            return

    def selected_covariates(self):
        selected_columns = []
        nuisance = []
        regressor = None
        for covar_name, button in self._covar_checks.items():
            if not button.isChecked():
                continue
            selected_columns.append(covar_name)
            role = self._role_combos[covar_name].currentText()
            if role == "Nuisance":
                nuisance.append(covar_name)
            elif role == "Regressor":
                regressor = covar_name
        return {
            "selected_columns": selected_columns,
            "nuisance": nuisance,
            "regressor": regressor,
        }

    def selected_row_indices(self):
        return list(self._filtered_indices)


__all__ = ["NBSPrepareDialog"]
