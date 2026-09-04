#!/usr/bin/env python3
"""Gradient scatter and classification dialogs."""

import json
import os
import re
import shlex
from itertools import combinations
from pathlib import Path
from warnings import warn

import nibabel as nib
import networkx as nx
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Circle

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
except Exception:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from nilearn import surface

try:
    from PyQt6.QtCore import QProcess, Qt
    from PyQt6.QtWidgets import (
        QCheckBox,
        QColorDialog,
        QComboBox,
        QDialog,
        QDoubleSpinBox,
        QFormLayout,
        QFileDialog,
        QFrame,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSlider,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )
except Exception:
    from PyQt5.QtCore import QProcess, Qt
    from PyQt5.QtWidgets import (
        QCheckBox,
        QColorDialog,
        QComboBox,
        QDialog,
        QDoubleSpinBox,
        QFormLayout,
        QFileDialog,
        QFrame,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSlider,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )

try:
    from window.shared.theme import dialog_theme_stylesheet as _shared_dialog_theme_stylesheet
except Exception:
    from mrsi_viewer.window.shared.theme import dialog_theme_stylesheet as _shared_dialog_theme_stylesheet

try:
    from .gradient_free_energy import GradientFreeEnergyDialog
    from .gradient_surface import GradientSurfaceDialog
except Exception:
    from gradient_free_energy import GradientFreeEnergyDialog
    from gradient_surface import GradientSurfaceDialog


def _load_nettools_class():
    from mrsitoolbox.connectomics.nettools import NetTools

    return NetTools


NetTools = _load_nettools_class()
nettools = NetTools()


FREE_ENERGY_PATHS_DEFAULT_DIR = Path("/home/ecelereau/Connectome/Dev/mrsitoolbox/results")
FREE_ENERGY_PATHS_SUFFIX = "desc-free_energy_paths.npz"


def _mrsitoolbox_root():
    viewer_root = Path(__file__).resolve().parents[1]
    return viewer_root.parent / "mrsitoolbox"


def _dialog_theme_stylesheet(theme_name="Dark"):
    return _shared_dialog_theme_stylesheet(theme_name)


METABOLITE_COLOR_SPECS = (
    ("glx", "Glx", "#d62728"),
    ("cho", "Cho", "#2ca02c"),
    ("naa", "NAA", "#1f77b4"),
    ("ins", "Ins", "#f2c94c"),
    ("crpcr", "CrPCr", "#000000"),
    ("water", "Water", "#7a7a7a"),
)


class RostrocaudalStatsRunDialog(QDialog):
    """Prompt for the null-test command after writing a free-energy path NPZ."""

    def __init__(
        self,
        *,
        output_path,
        subject_id,
        session_id,
        input_path="",
        pathsfolder=None,
        theme_name="Dark",
        parent=None,
    ):
        super().__init__(parent)
        self._output_path = Path(output_path).expanduser()
        self._subject_id = str(subject_id or "").strip()
        self._session_id = str(session_id or "").strip()
        self._mrsitoolbox_root = _mrsitoolbox_root()
        self.setWindowTitle("Run Rostrocaudal Null Test")

        input_default = str(input_path or "").strip()
        folder_default = str(pathsfolder or self._output_path.parent)

        self.input_path_edit = QLineEdit(input_default)
        self.input_path_edit.setToolTip("Gradient NPZ passed to test_rostrocaudal_gradient.py with -i.")
        self.input_browse_button = QPushButton("Browse")
        self.input_browse_button.clicked.connect(self._browse_input_path)

        self.pathsfolder_edit = QLineEdit(folder_default)
        self.pathsfolder_edit.setToolTip("Folder containing the saved path NPZ and receiving the stats NPZ.")
        self.pathsfolder_browse_button = QPushButton("Browse")
        self.pathsfolder_browse_button.clicked.connect(self._browse_pathsfolder)

        self.null_models_combo = QComboBox()
        self.null_models_combo.addItem("Moran", "moran")
        self.null_models_combo.addItem("Burt", "burt")
        self.null_models_combo.addItem("Moran + Burt", "moran,burt")

        self.n_perm_spin = QSpinBox()
        self.n_perm_spin.setRange(1, 10000000)
        self.n_perm_spin.setValue(1000)
        self.n_perm_spin.setSingleStep(100)

        self.n_proc_spin = QSpinBox()
        self.n_proc_spin.setRange(1, max(1, int(os.cpu_count() or 1)))
        self.n_proc_spin.setValue(1)

        self.null_n_proc_spin = QSpinBox()
        max_workers = max(1, int(os.cpu_count() or 1))
        self.null_n_proc_spin.setRange(1, max_workers)
        self.null_n_proc_spin.setValue(min(32, max_workers))

        self.overwrite_check = QCheckBox("Overwrite existing stats")
        self.overwrite_check.setChecked(True)

        self.endpoint_neighbor_check = QCheckBox("Also run endpoint-neighbor nulls")
        self.endpoint_neighbor_check.setChecked(True)

        self.endpoint_neighbor_groups_edit = QLineEdit("lh,rh")
        self.endpoint_neighbor_groups_edit.setToolTip("Groups passed to endpoint_neighbor_free_energy_nulls.py with --groups.")

        self.endpoint_neighbor_n_proc_spin = QSpinBox()
        self.endpoint_neighbor_n_proc_spin.setRange(1, max_workers)
        self.endpoint_neighbor_n_proc_spin.setValue(min(32, max_workers))

        self.endpoint_neighbor_shortest_only_check = QCheckBox("Shortest only")
        self.endpoint_neighbor_shortest_only_check.setChecked(False)
        self.endpoint_neighbor_shortest_only_check.setToolTip("Fastest endpoint-neighbor null mode; skips simple-path enumeration.")

        self.command_preview = QLineEdit()
        self.command_preview.setReadOnly(True)
        self.command_preview.setToolTip("Command(s) that will be launched from the mrsitoolbox folder.")

        run_button = QPushButton("Run")
        run_button.clicked.connect(self.accept)
        skip_button = QPushButton("Skip")
        skip_button.clicked.connect(self.reject)

        form = QFormLayout()
        subject_label = QLabel(f"sub-{self._subject_id} / ses-{self._session_id}")
        subject_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            if hasattr(Qt, "TextInteractionFlag")
            else Qt.TextSelectableByMouse
        )
        form.addRow("Subject/session", subject_label)
        form.addRow("Saved path", QLabel(self._output_path.name))

        input_row = QHBoxLayout()
        input_row.addWidget(self.input_path_edit, 1)
        input_row.addWidget(self.input_browse_button, 0)
        form.addRow("Gradient NPZ", input_row)

        folder_row = QHBoxLayout()
        folder_row.addWidget(self.pathsfolder_edit, 1)
        folder_row.addWidget(self.pathsfolder_browse_button, 0)
        form.addRow("Paths folder", folder_row)

        form.addRow("Null models", self.null_models_combo)
        form.addRow("Permutations", self.n_perm_spin)
        form.addRow("Subject workers", self.n_proc_spin)
        form.addRow("Null workers", self.null_n_proc_spin)
        form.addRow("", self.overwrite_check)
        form.addRow("", self.endpoint_neighbor_check)
        endpoint_row = QHBoxLayout()
        endpoint_row.addWidget(QLabel("Groups"))
        endpoint_row.addWidget(self.endpoint_neighbor_groups_edit, 1)
        endpoint_row.addWidget(QLabel("Workers"))
        endpoint_row.addWidget(self.endpoint_neighbor_n_proc_spin, 0)
        endpoint_row.addWidget(self.endpoint_neighbor_shortest_only_check, 0)
        form.addRow("Endpoint nulls", endpoint_row)
        form.addRow("Command", self.command_preview)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(skip_button)
        buttons.addWidget(run_button)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(buttons)

        for widget in (
            self.input_path_edit,
            self.pathsfolder_edit,
        ):
            widget.textChanged.connect(self._update_command_preview)
        self.null_models_combo.currentIndexChanged.connect(lambda _index: self._update_command_preview())
        self.n_perm_spin.valueChanged.connect(lambda _value: self._update_command_preview())
        self.n_proc_spin.valueChanged.connect(lambda _value: self._update_command_preview())
        self.null_n_proc_spin.valueChanged.connect(lambda _value: self._update_command_preview())
        self.overwrite_check.toggled.connect(lambda _checked: self._update_command_preview())
        self.endpoint_neighbor_check.toggled.connect(lambda _checked: self._update_command_preview())
        self.endpoint_neighbor_groups_edit.textChanged.connect(lambda _text: self._update_command_preview())
        self.endpoint_neighbor_n_proc_spin.valueChanged.connect(lambda _value: self._update_command_preview())
        self.endpoint_neighbor_shortest_only_check.toggled.connect(lambda _checked: self._update_command_preview())

        self.setMinimumWidth(760)
        _theme, style = _dialog_theme_stylesheet(theme_name)
        self.setStyleSheet(style)
        self._update_command_preview()

    def _browse_input_path(self):
        start = self.input_path_edit.text().strip() or str(self._mrsitoolbox_root)
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Choose gradient NPZ",
            start,
            "NumPy archive (*.npz);;All files (*)",
        )
        if path:
            self.input_path_edit.setText(path)

    def _browse_pathsfolder(self):
        start = self.pathsfolder_edit.text().strip() or str(self._output_path.parent)
        path = QFileDialog.getExistingDirectory(self, "Choose paths folder", start)
        if path:
            self.pathsfolder_edit.setText(path)

    def command_argv(self):
        script_path = self._mrsitoolbox_root / "experiments" / "gradient_analysis" / "test_rostrocaudal_gradient.py"
        argv = [
            "python3",
            str(script_path),
            "-i",
            self.input_path_edit.text().strip(),
            "--pathsfolder",
            self.pathsfolder_edit.text().strip(),
            "--path-file",
            str(self._output_path),
            "--null-models",
            str(self.null_models_combo.currentData() or "moran"),
            "--n-perm",
            str(int(self.n_perm_spin.value())),
            "--n-proc",
            str(int(self.n_proc_spin.value())),
            "--null-n-proc",
            str(int(self.null_n_proc_spin.value())),
            "--subject-id",
            self._subject_id,
            "--session-id",
            self._session_id,
        ]
        if self.overwrite_check.isChecked():
            argv.append("--overwrite")
        if self.endpoint_neighbor_shortest_only_check.isChecked():
            argv.append("--shortest-only")
        return argv

    def command_text(self):
        commands = [shlex.join(self.command_argv())]
        if self.endpoint_neighbor_check.isChecked():
            commands.append(shlex.join(self.endpoint_neighbor_command_argv()))
        return " && ".join(commands)

    def endpoint_neighbor_command_argv(self):
        script_path = self._mrsitoolbox_root / "experiments" / "gradient_analysis" / "endpoint_neighbor_free_energy_nulls.py"
        argv = [
            "python3",
            str(script_path),
            "-i",
            str(self._output_path),
            "--groups",
            self.endpoint_neighbor_groups_edit.text().strip() or "lh,rh",
            "--nproc",
            str(int(self.endpoint_neighbor_n_proc_spin.value())),
        ]
        if self.overwrite_check.isChecked():
            argv.append("--overwrite")
        return argv

    def values(self):
        argvs = [self.command_argv()]
        if self.endpoint_neighbor_check.isChecked():
            argvs.append(self.endpoint_neighbor_command_argv())
        return {
            "argv": self.command_argv(),
            "argvs": argvs,
            "cwd": self._mrsitoolbox_root,
        }

    def _update_command_preview(self):
        self.command_preview.setText(self.command_text())

    def accept(self):
        input_path = Path(self.input_path_edit.text().strip()).expanduser()
        if not input_path.is_file():
            warn(f"Rostrocaudal null test not launched: gradient NPZ does not exist: {input_path}")
            return
        pathsfolder = Path(self.pathsfolder_edit.text().strip()).expanduser()
        if not pathsfolder.is_dir():
            warn(f"Rostrocaudal null test not launched: paths folder does not exist: {pathsfolder}")
            return
        if not self._output_path.is_file():
            warn(f"Rostrocaudal null test not launched: saved path NPZ does not exist: {self._output_path}")
            return
        if self.endpoint_neighbor_check.isChecked() and not self.endpoint_neighbor_groups_edit.text().strip():
            warn("Endpoint-neighbor null test not launched: groups cannot be empty.")
            return
        super().accept()


class RostrocaudalStatsProgressDialog(QDialog):
    """Modeless progress dialog for the rostrocaudal null-test process."""

    def __init__(
        self,
        *,
        argv,
        argvs=None,
        cwd,
        subject_id,
        session_id,
        theme_name="Dark",
        parent=None,
    ):
        super().__init__(parent)
        self._commands = [list(command or []) for command in list(argvs or []) if command]
        if not self._commands:
            self._commands = [list(argv or [])]
        self._argv = list(self._commands[0] if self._commands else [])
        self._cwd = Path(cwd).expanduser()
        self._subject_id = str(subject_id or "").strip()
        self._session_id = str(session_id or "").strip()
        self._process = None
        self._line_buffer = ""
        self._running = False
        self._command_index = -1
        self._models = self._models_from_argv(self._argv)
        self._n_perm = self._n_perm_from_argv(self._argv)
        self._model_offsets = {model: idx * self._n_perm for idx, model in enumerate(self._models)}
        self._total_progress = max(1, self._n_perm * max(1, len(self._models)))

        self.setWindowTitle("Rostrocaudal Null Test Progress")
        self.status_label = QLabel(f"Preparing sub-{self._subject_id}_ses-{self._session_id}")
        self.status_label.setWordWrap(True)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, self._total_progress)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.output_text = QPlainTextEdit()
        self.output_text.setReadOnly(True)
        try:
            self.output_text.document().setMaximumBlockCount(500)
        except Exception:
            pass
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._on_cancel_clicked)

        layout = QVBoxLayout(self)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.output_text, 1)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)

        self.setMinimumSize(780, 420)
        _theme, style = _dialog_theme_stylesheet(theme_name)
        self.setStyleSheet(style)

    @staticmethod
    def _models_from_argv(argv):
        values = []
        try:
            idx = list(argv).index("--null-models")
            raw = str(list(argv)[idx + 1])
        except Exception:
            raw = "moran"
        for token in raw.split(","):
            model = token.strip().lower()
            if model in {"moran", "burt"} and model not in values:
                values.append(model)
        return values or ["moran"]

    @staticmethod
    def _n_perm_from_argv(argv):
        try:
            idx = list(argv).index("--n-perm")
            value = int(str(list(argv)[idx + 1]))
        except Exception:
            value = 1000
        return max(1, int(value))

    @staticmethod
    def _merged_channel_mode():
        return (
            QProcess.ProcessChannelMode.MergedChannels
            if hasattr(QProcess, "ProcessChannelMode")
            else QProcess.MergedChannels
        )

    def start(self):
        if not self._commands or len(self._commands[0]) < 1:
            self.status_label.setText("No command to run.")
            return
        self._command_index = -1
        self._start_next_command()

    def _is_rostrocaudal_command(self, argv):
        return any("test_rostrocaudal_gradient.py" in str(part) for part in list(argv or []))

    def _prepare_progress_for_command(self):
        if self._is_rostrocaudal_command(self._argv):
            self._models = self._models_from_argv(self._argv)
            self._n_perm = self._n_perm_from_argv(self._argv)
            self._model_offsets = {model: idx * self._n_perm for idx, model in enumerate(self._models)}
            self._total_progress = max(1, self._n_perm * max(1, len(self._models)))
            self.progress_bar.setRange(0, self._total_progress)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("%p%")
        else:
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setFormat("Running")

    def _start_next_command(self):
        self._command_index += 1
        if self._command_index >= len(self._commands):
            self._running = False
            if self.progress_bar.minimum() == 0 and self.progress_bar.maximum() == 0:
                self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(self.progress_bar.maximum())
            self.progress_bar.setFormat("%p%")
            self.status_label.setText(
                f"Finished sub-{self._subject_id}_ses-{self._session_id}"
            )
            self.cancel_button.setText("Close")
            return

        self._argv = list(self._commands[self._command_index])
        if len(self._argv) < 1:
            self._start_next_command()
            return
        self._line_buffer = ""
        self._prepare_progress_for_command()
        self._process = QProcess(self)
        self._process.setWorkingDirectory(str(self._cwd))
        self._process.setProcessChannelMode(self._merged_channel_mode())
        self._process.readyReadStandardOutput.connect(self._on_ready_read)
        self._process.finished.connect(self._on_finished)
        self._process.errorOccurred.connect(self._on_error)
        self._running = True
        command_count = len(self._commands)
        prefix = f" [{self._command_index + 1}/{command_count}]" if command_count > 1 else ""
        self.status_label.setText(f"Running{prefix} sub-{self._subject_id}_ses-{self._session_id}")
        self._append_output("$ " + shlex.join(self._argv))
        self._process.start(str(self._argv[0]), [str(value) for value in self._argv[1:]])

    def _append_output(self, text):
        if not text:
            return
        self.output_text.appendPlainText(str(text).rstrip())

    def _on_ready_read(self):
        if self._process is None:
            return
        data = bytes(self._process.readAllStandardOutput())
        if not data:
            return
        text = data.decode("utf-8", errors="replace")
        self._append_output(text)
        self._parse_progress_text(text)

    def _parse_progress_text(self, text):
        self._line_buffer += str(text)
        lines = self._line_buffer.splitlines()
        if self._line_buffer and not self._line_buffer.endswith(("\n", "\r")):
            self._line_buffer = lines.pop() if lines else self._line_buffer
        else:
            self._line_buffer = ""

        for line in lines:
            self._parse_progress_line(line)

    def _parse_progress_line(self, line):
        text = str(line or "").strip()
        if not text:
            return

        progress_match = re.search(
            r"\|\s*(moran|burt)\s*\|.*?null progress\s+(\d+)\s*/\s*(\d+)",
            text,
            flags=re.IGNORECASE,
        )
        if progress_match is not None:
            model = progress_match.group(1).lower()
            current = int(progress_match.group(2))
            total = max(1, int(progress_match.group(3)))
            if total != self._n_perm:
                self._n_perm = total
                self._model_offsets = {name: idx * self._n_perm for idx, name in enumerate(self._models)}
                self._total_progress = max(1, self._n_perm * max(1, len(self._models)))
                self.progress_bar.setRange(0, self._total_progress)
            value = self._model_offsets.get(model, 0) + min(current, total)
            self.progress_bar.setValue(max(0, min(int(value), int(self._total_progress))))
            self.status_label.setText(
                f"sub-{self._subject_id}_ses-{self._session_id} | {model} null {current}/{total}"
            )
            return

        step_match = re.search(r"\|\s*(moran|burt)\s*\|\s*step\s+([^|]+)", text, flags=re.IGNORECASE)
        if step_match is not None:
            self.status_label.setText(
                f"sub-{self._subject_id}_ses-{self._session_id} | {step_match.group(1).lower()} | step {step_match.group(2).strip()}"
            )
            return

        if "Completed sub-" in text:
            self.progress_bar.setValue(self.progress_bar.maximum())
            self.status_label.setText(text)

    def _on_finished(self, exit_code=0, _exit_status=None):
        if self._line_buffer:
            self._parse_progress_line(self._line_buffer)
            self._line_buffer = ""
        self._running = False
        exit_code = int(exit_code)
        if exit_code == 0:
            self._process = None
            if self._command_index + 1 < len(self._commands):
                self._start_next_command()
                return
            if self.progress_bar.minimum() == 0 and self.progress_bar.maximum() == 0:
                self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(self.progress_bar.maximum())
            self.progress_bar.setFormat("%p%")
            self.status_label.setText(f"Finished sub-{self._subject_id}_ses-{self._session_id}")
        else:
            self.status_label.setText(
                f"Null test command exited with code {exit_code}"
            )
        self.cancel_button.setText("Close")

    def _on_error(self, error):
        self.status_label.setText(f"Null test process error: {error}")

    def _on_cancel_clicked(self):
        if self._running and self._process is not None:
            self.status_label.setText("Stopping null test...")
            self._process.terminate()
            return
        self.close()

    def closeEvent(self, event):
        if self._running:
            event.ignore()
            self.status_label.setText("Process is still running. Use Cancel to stop it.")
            return
        super().closeEvent(event)


class CollapsibleSection(QWidget):
    """Small collapsible container used by the scatter sidebar."""

    def __init__(self, title, *, collapsed=False, parent=None):
        super().__init__(parent)
        self._title = str(title or "")
        self.toggle_button = QPushButton()
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(not bool(collapsed))
        self.toggle_button.clicked.connect(self._sync_state)
        self.content = QWidget(self)
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(10, 6, 4, 8)
        self.content_layout.setSpacing(6)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.content)
        self._sync_state()

    def _sync_state(self):
        expanded = bool(self.toggle_button.isChecked())
        self.toggle_button.setText(("- " if expanded else "+ ") + self._title)
        self.content.setVisible(expanded)

    def addLayout(self, layout):
        self.content_layout.addLayout(layout)

    def addWidget(self, widget):
        self.content_layout.addWidget(widget)

    def addStretch(self, stretch=1):
        self.content_layout.addStretch(stretch)


class MetabolitePlotSettingsDialog(QDialog):
    """Modeless settings dialog for metabolite profile figures."""

    def __init__(self, owner, *, theme_name="Dark", parent=None):
        super().__init__(parent)
        self._owner = owner
        self.setWindowTitle("Metabolite Plot Settings")
        self._build_ui()
        self.set_theme(theme_name)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.x_axis_label_edit = QLineEdit("Gradient 1")
        form.addRow("X axis name", self.x_axis_label_edit)

        self.y_axis_label_edit = QLineEdit("Gradient 2")
        form.addRow("Y axis name", self.y_axis_label_edit)

        self.x_axis_fontsize_spin = QSpinBox()
        self.x_axis_fontsize_spin.setRange(6, 48)
        form.addRow("X label size", self.x_axis_fontsize_spin)

        self.y_axis_fontsize_spin = QSpinBox()
        self.y_axis_fontsize_spin.setRange(6, 48)
        form.addRow("Y label size", self.y_axis_fontsize_spin)

        self.tick_fontsize_spin = QSpinBox()
        self.tick_fontsize_spin.setRange(6, 36)
        form.addRow("Tick size", self.tick_fontsize_spin)

        self.line_width_spin = QDoubleSpinBox()
        self.line_width_spin.setDecimals(2)
        self.line_width_spin.setRange(0.2, 12.0)
        self.line_width_spin.setSingleStep(0.1)
        form.addRow("Line thickness", self.line_width_spin)

        self.confidence_interval_spin = QDoubleSpinBox()
        self.confidence_interval_spin.setDecimals(1)
        self.confidence_interval_spin.setRange(1.0, 100.0)
        self.confidence_interval_spin.setSingleStep(1.0)
        self.confidence_interval_spin.setSuffix("%")
        form.addRow("Interval", self.confidence_interval_spin)

        self.boxplot_bars_check = QCheckBox("Boxplot bars")
        self.boxplot_bars_check.setToolTip("Draw whisker and interquartile bars at each plotted median point.")
        form.addRow("", self.boxplot_bars_check)

        self.color_edits = {}
        for key, label, _default_color in METABOLITE_COLOR_SPECS:
            row = QHBoxLayout()
            edit = QLineEdit("")
            edit.setPlaceholderText("#RRGGBB")
            choose_button = QPushButton("Choose")
            choose_button.clicked.connect(lambda _checked=False, color_key=key: self._choose_color(color_key))
            row.addWidget(edit, 1)
            row.addWidget(choose_button, 0)
            form.addRow(f"{label} color", row)
            self.color_edits[key] = edit

        layout.addLayout(form)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self._apply_changes)
        buttons.addWidget(self.apply_button)
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self._save_changes)
        buttons.addWidget(self.save_button)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        buttons.addWidget(self.close_button)
        layout.addLayout(buttons)

    def set_theme(self, theme_name="Dark"):
        theme, style = _dialog_theme_stylesheet(theme_name)
        self._theme_name = theme
        self.setStyleSheet(style)

    def sync_from_settings(self, settings):
        current = dict(settings or {})
        self.x_axis_label_edit.setText(str(current.get("x_axis_label", "Gradient 1") or "Gradient 1"))
        self.y_axis_label_edit.setText(str(current.get("y_axis_label", "Gradient 2") or "Gradient 2"))
        self.x_axis_fontsize_spin.setValue(int(current.get("x_axis_fontsize", 11)))
        self.y_axis_fontsize_spin.setValue(int(current.get("y_axis_fontsize", 11)))
        self.tick_fontsize_spin.setValue(int(current.get("tick_fontsize", 9)))
        self.line_width_spin.setValue(float(current.get("line_width", 1.6)))
        self.confidence_interval_spin.setValue(float(current.get("confidence_interval", 95.0)))
        self.boxplot_bars_check.setChecked(bool(current.get("boxplot_bars", False)))
        colors = dict(current.get("colors", {}) or {})
        defaults = {key: color for key, _label, color in METABOLITE_COLOR_SPECS}
        for key, edit in self.color_edits.items():
            edit.setText(str(colors.get(key, defaults.get(key, "")) or ""))

    def current_settings(self):
        return {
            "x_axis_label": self.x_axis_label_edit.text().strip() or "Gradient 1",
            "y_axis_label": self.y_axis_label_edit.text().strip() or "Gradient 2",
            "x_axis_fontsize": int(self.x_axis_fontsize_spin.value()),
            "y_axis_fontsize": int(self.y_axis_fontsize_spin.value()),
            "tick_fontsize": int(self.tick_fontsize_spin.value()),
            "line_width": float(self.line_width_spin.value()),
            "confidence_interval": float(self.confidence_interval_spin.value()),
            "boxplot_bars": bool(self.boxplot_bars_check.isChecked()),
            "colors": {
                key: edit.text().strip()
                for key, edit in self.color_edits.items()
                if edit.text().strip()
            },
        }

    def _choose_color(self, key):
        edit = self.color_edits.get(key)
        initial = edit.text().strip() if edit is not None else ""
        color = QColorDialog.getColor(parent=self)
        if color.isValid() and edit is not None:
            edit.setText(str(color.name()))

    def _apply_changes(self):
        if self._owner is not None:
            self._owner._apply_metabolite_plot_settings(self.current_settings())

    def _save_changes(self):
        self._apply_changes()
        self.close()


class ScatterAppearanceDialog(QDialog):
    """Modeless settings dialog for scatter plot decoration."""

    def __init__(self, owner, *, theme_name="Dark", parent=None):
        super().__init__(parent)
        self._owner = owner
        self._subplot_title_edits = {}
        self.setWindowTitle("Scatter Settings")
        self._build_ui()
        self.set_theme(theme_name)

    def _build_ui(self):
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.figure_title_edit = QLineEdit("")
        form.addRow("Figure title", self.figure_title_edit)

        self.axis_label_fontsize_spin = QSpinBox()
        self.axis_label_fontsize_spin.setRange(6, 48)
        form.addRow("Axis label size", self.axis_label_fontsize_spin)

        self.tick_label_fontsize_spin = QSpinBox()
        self.tick_label_fontsize_spin.setRange(6, 48)
        form.addRow("Tick label size", self.tick_label_fontsize_spin)
        layout.addLayout(form)

        self.subplot_titles_container = QWidget()
        self.subplot_titles_form = QFormLayout(self.subplot_titles_container)
        layout.addWidget(self.subplot_titles_container)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self._apply_changes)
        buttons.addWidget(self.apply_button)
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self._save_changes)
        buttons.addWidget(self.save_button)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        buttons.addWidget(self.close_button)
        layout.addLayout(buttons)

    def set_theme(self, theme_name="Dark"):
        theme, style = _dialog_theme_stylesheet(theme_name)
        self._theme_name = theme
        self.setStyleSheet(style)

    def sync_from_settings(self, settings, subplot_specs):
        current = dict(settings or {})
        subplot_titles = dict(current.get("subplot_titles", {}) or {})

        self.figure_title_edit.setText(str(current.get("figure_title", "") or ""))
        self.axis_label_fontsize_spin.setValue(int(current.get("axis_label_fontsize", 12)))
        self.tick_label_fontsize_spin.setValue(int(current.get("tick_label_fontsize", 11)))

        while self.subplot_titles_form.rowCount() > 0:
            self.subplot_titles_form.removeRow(0)
        self._subplot_title_edits = {}

        for spec in list(subplot_specs or []):
            key = str(spec.get("name", "all")).strip().lower()
            label = str(spec.get("title", "") or key.upper())
            edit = QLineEdit(str(subplot_titles.get(key, "") or ""))
            self.subplot_titles_form.addRow(f"{label} subplot title", edit)
            self._subplot_title_edits[key] = edit

        self.subplot_titles_container.setVisible(bool(self._subplot_title_edits))

    def current_settings(self):
        return {
            "figure_title": self.figure_title_edit.text().strip(),
            "axis_label_fontsize": int(self.axis_label_fontsize_spin.value()),
            "tick_label_fontsize": int(self.tick_label_fontsize_spin.value()),
            "subplot_titles": {
                str(key): edit.text().strip()
                for key, edit in self._subplot_title_edits.items()
            },
        }

    def _apply_changes(self):
        if self._owner is not None:
            self._owner._apply_scatter_appearance_settings(self.current_settings())

    def _save_changes(self):
        self._apply_changes()
        self.close()


class GradientScatterDialog(QDialog):
    """Interactive scatter viewer for arbitrary gradient or spatial axes."""

    def __init__(
        self,
        x_values,
        y_values,
        *,
        z_values=None,
        color_values=None,
        point_labels=None,
        point_ids=None,
        title="Gradient Scatter",
        x_label="X axis",
        y_label="Y axis",
        z_label="Z axis",
        color_label="Gradient 1",
        gradient1_values=None,
        rgb_x_values=None,
        rgb_y_values=None,
        rgb_z_values=None,
        path_metric_coords=None,
        parent=None,
        cmap=None,
        cmap_name="spectrum_fsl",
        theme_name="Dark",
        hemisphere_mode="both",
        rotation_preset="Default",
        use_triangular_rgb=False,
        rgb_fit_mode="triangle",
        triangular_color_order="RBG",
        rgb_scalar_mode="barycentric",
        edge_pairs=None,
        edge_color="#111827",
        edge_alpha=0.16,
        edge_linewidth=0.45,
        point_group_codes=None,
        metabolite_profiles=None,
        metabolite_names=None,
        metabolite_subject_labels=None,
        show_proximity_circles=False,
        initial_proximity_slider_value=0,
        auto_preload_matching_paths=False,
        project_paths_callback=None,
        export_metadata=None,
    ):
        super().__init__(parent)
        self._x = np.asarray(x_values, dtype=float).reshape(-1)
        self._y = np.asarray(y_values, dtype=float).reshape(-1)
        if self._x.shape != self._y.shape:
            raise ValueError("Gradient scatter axes must have matching lengths.")
        z_data = None if z_values is None else np.asarray(z_values, dtype=float).reshape(-1)
        if z_data is not None and z_data.shape != self._x.shape:
            raise ValueError("Gradient scatter Z axis must match the axis lengths.")
        self._is_3d = z_data is not None
        color_data = self._y if color_values is None else np.asarray(color_values, dtype=float).reshape(-1)
        if color_data.shape != self._x.shape:
            raise ValueError("Gradient scatter color data must match the axis lengths.")
        if gradient1_values is None:
            gradient1_data = np.asarray(color_data, dtype=float).reshape(-1)
        else:
            gradient1_data = np.asarray(gradient1_values, dtype=float).reshape(-1)
            if gradient1_data.shape != self._x.shape:
                raise ValueError("Gradient scatter Gradient 1 data must match the axis lengths.")
        rgb_x_data = None
        rgb_y_data = None
        rgb_z_data = None
        if rgb_x_values is not None or rgb_y_values is not None or rgb_z_values is not None:
            if rgb_x_values is None or rgb_y_values is None:
                raise ValueError("Both RGB X and RGB Y coordinate arrays are required.")
            if self._is_3d and rgb_z_values is None:
                raise ValueError("RGB Z coordinate array is required for 3D RGB coloring.")
            if not self._is_3d and rgb_z_values is not None:
                raise ValueError("RGB Z coordinate array requires a 3D scatter Z axis.")
            rgb_x_data = np.asarray(rgb_x_values, dtype=float).reshape(-1)
            rgb_y_data = np.asarray(rgb_y_values, dtype=float).reshape(-1)
            if rgb_x_data.shape != self._x.shape or rgb_y_data.shape != self._x.shape:
                raise ValueError("Gradient scatter RGB coordinate arrays must match the axis lengths.")
            if self._is_3d:
                rgb_z_data = np.asarray(rgb_z_values, dtype=float).reshape(-1)
                if rgb_z_data.shape != self._x.shape:
                    raise ValueError("Gradient scatter RGB Z coordinate array must match the axis lengths.")
        if point_labels is None:
            label_data = np.asarray([f"Point {idx + 1}" for idx in range(self._x.size)], dtype=object)
        else:
            label_data = np.asarray(point_labels, dtype=object).reshape(-1)
            if label_data.shape != self._x.shape:
                raise ValueError("Gradient scatter point labels must match the axis lengths.")
        if point_ids is None:
            point_id_data = np.arange(1, self._x.size + 1, dtype=int)
        else:
            point_id_data = np.asarray(point_ids, dtype=object).reshape(-1)
            if point_id_data.shape != self._x.shape:
                raise ValueError("Gradient scatter point ids must match the axis lengths.")
        if point_group_codes is None:
            group_data = np.full(self._x.shape, -1, dtype=int)
        else:
            group_data = np.asarray(point_group_codes, dtype=int).reshape(-1)
            if group_data.shape != self._x.shape:
                raise ValueError("Gradient scatter point hemisphere codes must match the axis lengths.")
        finite_mask = np.isfinite(self._x) & np.isfinite(self._y) & np.isfinite(color_data) & np.isfinite(gradient1_data)
        if self._is_3d:
            finite_mask &= np.isfinite(z_data)
        if bool(use_triangular_rgb) and rgb_x_data is not None and rgb_y_data is not None:
            finite_mask &= np.isfinite(rgb_x_data) & np.isfinite(rgb_y_data)
            if self._is_3d:
                finite_mask &= np.isfinite(rgb_z_data)
        metric_data = None
        if path_metric_coords is not None:
            metric_data = np.asarray(path_metric_coords, dtype=float)
            if metric_data.ndim == 1:
                metric_data = metric_data[:, np.newaxis]
            if metric_data.ndim != 2 or metric_data.shape[0] != self._x.shape[0]:
                raise ValueError("Gradient scatter path metric coordinates must match the axis lengths.")
            finite_mask &= np.all(np.isfinite(metric_data), axis=1)
        if not np.any(finite_mask):
            raise ValueError("Gradient scatter requires finite data points.")
        self._x = self._x[finite_mask]
        self._y = self._y[finite_mask]
        self._z = np.asarray(z_data[finite_mask], dtype=float) if self._is_3d else None
        self._color = color_data[finite_mask]
        self._gradient1 = gradient1_data[finite_mask]
        self._rgb_x = np.asarray(rgb_x_data[finite_mask], dtype=float) if rgb_x_data is not None else None
        self._rgb_y = np.asarray(rgb_y_data[finite_mask], dtype=float) if rgb_y_data is not None else None
        self._rgb_z = np.asarray(rgb_z_data[finite_mask], dtype=float) if rgb_z_data is not None else None
        self._path_metric_coords = (
            np.asarray(metric_data[finite_mask, :], dtype=float)
            if metric_data is not None
            else None
        )
        self._point_labels = np.asarray([str(value) for value in label_data[finite_mask].tolist()], dtype=object)
        self._point_ids = np.asarray([str(value) for value in point_id_data[finite_mask].tolist()], dtype=object)
        self._point_group_codes = np.asarray(group_data[finite_mask], dtype=int)
        metabolite_profile_data = None
        if metabolite_profiles is not None:
            try:
                raw_profiles = np.asarray(metabolite_profiles, dtype=float)
                if raw_profiles.ndim == 2 and raw_profiles.shape[0] == finite_mask.shape[0]:
                    metabolite_profile_data = np.asarray(raw_profiles[finite_mask, :], dtype=float)
                elif raw_profiles.ndim == 3 and raw_profiles.shape[1] == finite_mask.shape[0]:
                    metabolite_profile_data = np.asarray(raw_profiles[:, finite_mask, :], dtype=float)
            except Exception:
                metabolite_profile_data = None
        metabolite_name_data = []
        if metabolite_names is not None:
            try:
                metabolite_name_data = [
                    str(value).strip()
                    for value in np.asarray(metabolite_names, dtype=object).reshape(-1).tolist()
                    if str(value).strip()
                ]
            except Exception:
                metabolite_name_data = []
        if metabolite_profile_data is not None:
            n_met = int(metabolite_profile_data.shape[-1])
            if len(metabolite_name_data) != n_met:
                metabolite_name_data = [f"Metabolite {idx + 1}" for idx in range(n_met)]
        else:
            metabolite_name_data = []
        metabolite_subject_label_data = []
        if metabolite_profile_data is not None and metabolite_profile_data.ndim == 3:
            n_subjects = int(metabolite_profile_data.shape[0])
            if metabolite_subject_labels is not None:
                try:
                    metabolite_subject_label_data = [
                        str(value).strip()
                        for value in np.asarray(metabolite_subject_labels, dtype=object).reshape(-1).tolist()
                    ]
                except Exception:
                    metabolite_subject_label_data = []
            if len(metabolite_subject_label_data) != n_subjects:
                metabolite_subject_label_data = [f"Subject {idx + 1}" for idx in range(n_subjects)]
        self._metabolite_profiles = metabolite_profile_data
        self._metabolite_names = metabolite_name_data
        self._metabolite_subject_labels = metabolite_subject_label_data
        self._title = str(title or "Gradient Scatter")
        self._x_label = str(x_label or "X axis")
        self._y_label = str(y_label or "Y axis")
        self._z_label = str(z_label or "Z axis")
        self._color_label = str(color_label or "Gradient 1")
        self._cmap_name = str(cmap_name or "spectrum_fsl")
        self._cmap = cmap if cmap is not None else GradientSurfaceDialog._default_cmap(self._cmap_name)
        self._theme_name = "Dark"
        self._hemisphere_mode = self._normalize_scatter_hemisphere_mode(hemisphere_mode)
        self._rotation_preset = self._normalize_rotation_preset(rotation_preset)
        self._use_triangular_rgb = bool(use_triangular_rgb)
        self._rgb_fit_mode = self._normalize_rgb_fit_mode(rgb_fit_mode)
        self._triangular_color_order = self._normalize_triangular_color_order(triangular_color_order)
        self._rgb_scalar_mode = self._normalize_rgb_scalar_mode(rgb_scalar_mode)
        self._edge_pairs = self._normalize_edge_pairs(edge_pairs, self._x.size)
        self._edge_color = str(edge_color or "#111827")
        try:
            self._edge_alpha = float(np.clip(float(edge_alpha), 0.0, 1.0))
        except Exception:
            self._edge_alpha = 0.16
        try:
            self._edge_linewidth = max(0.0, float(edge_linewidth))
        except Exception:
            self._edge_linewidth = 0.45
        self._path_width_scaling_mode = "exp"
        self._path_width_scaling_strength = 2.0
        self._project_paths_callback = project_paths_callback if callable(project_paths_callback) else None
        self._export_metadata = dict(export_metadata or {}) if isinstance(export_metadata, dict) else {}
        self._default_fixed_anchor_indices = self._derive_default_fixed_anchor_indices()
        self._manual_anchor_overrides = {}
        self._manual_subc_anchor_overrides = {}
        self._loaded_fixed_endpoint_file = ""
        self._loaded_fixed_endpoint_source = ""
        self._free_energy_paths_load_dir = self._default_free_energy_paths_dir()
        self._endpoint_selection_mode = "adaptive"
        self._manual_endpoint_target = None
        self._project_paths_payload = None
        self._selected_ctx_path_indices = {"lh": 0, "rh": 0, "all": 0}
        self._fibrenet_layout = "diffusion"
        self._show_proximity_circles = bool(show_proximity_circles)
        self._show_adjacency_edges = True
        self._show_all_ordered_paths = False
        self._use_edge_bundling = False
        self._edge_bundling_note = ""
        self._normalize_free_energy_by_segments = True
        display_x, display_y = self._rotate_points(self._x, self._y, self._rotation_preset)
        if self._is_3d:
            self._display_coords = np.column_stack((display_x, display_y, self._z))
        else:
            self._display_coords = np.column_stack((display_x, display_y))
        self._path_channel_order = self._default_path_channel_order()
        self._proximity_max_radius = self._compute_max_radius(self._display_coords)
        self._proximity_slider_steps = 1000
        self._initial_proximity_slider_value = self._normalize_proximity_slider_value(
            initial_proximity_slider_value,
            self._proximity_slider_steps,
        )
        self._auto_preload_matching_paths = bool(auto_preload_matching_paths)
        self._proximity_radius = self._slider_to_radius(self._initial_proximity_slider_value)
        self._edge_distances = self._compute_edge_distances(self._display_coords, self._edge_pairs)
        self._path_metric_edge_distances = self._compute_edge_distances(
            self._path_metric_coords if self._path_metric_coords is not None else self._display_coords,
            self._edge_pairs,
        )
        self._fixed_xlim, self._fixed_ylim, self._fixed_zlim = self._compute_fixed_axes(self._display_coords)
        self._point_artist = None
        self._point_artist_entries = []
        self._hover_cid = None
        self._click_cid = None
        self._free_energy_dialog = None
        self._appearance_dialog = None
        self._metabolite_settings_dialog = None
        self._scatter_appearance_settings = self._default_scatter_appearance_settings()
        self._metabolite_plot_settings = self._default_metabolite_plot_settings()
        self._last_rgb_model = None
        self._last_rgb_model_x = None
        self._last_rgb_model_y = None
        self._last_rgb_point_colors = None
        self.setWindowTitle(self._title)

        self.figure = Figure(figsize=(7.4, 6.4), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.info_label = QLabel(self._info_text())
        self.info_label.setWordWrap(True)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.interface_mode_combo = QComboBox()
        self.interface_mode_combo.addItem("Basic", "basic")
        self.interface_mode_combo.addItem("Advanced", "advanced")
        self.interface_mode_combo.setToolTip("Basic mode shows the common controls. Advanced mode reveals tuning parameters.")
        self.interface_mode_combo.currentIndexChanged.connect(lambda _index: self._sync_interface_mode())
        self.interface_preset_combo = QComboBox()
        self.interface_preset_combo.addItem("Default view", "default")
        self.interface_preset_combo.addItem("Publication figure", "publication")
        self.interface_preset_combo.addItem("Path exploration", "path_exploration")
        self.interface_preset_combo.addItem("Triangle RGB", "triangle_rgb")
        self.interface_preset_combo.addItem("Debug mode", "debug")
        self.interface_preset_combo.setToolTip("Apply a small preset for common plotting workflows.")
        self.interface_preset_combo.currentIndexChanged.connect(self._on_interface_preset_changed)
        self.settings_button = QPushButton("Settings")
        self.settings_button.clicked.connect(self._open_appearance_dialog)
        self.save_button = QPushButton("Save Figure")
        self.save_button.clicked.connect(self._save_figure)
        self.metabolite_bins_label = QLabel("Number of metabolite bins")
        self.metabolite_bins_spin = QSpinBox()
        self.metabolite_bins_spin.setRange(2, 200)
        self.metabolite_bins_spin.setValue(12)
        self.metabolite_bins_spin.setToolTip("Number of triangular gradient bins for metabolite profile averaging.")
        self.metabolite_axis_combo = QComboBox()
        self.metabolite_axis_combo.addItem("Triangle RGB coloring", "triangular_rgb")
        self.metabolite_axis_combo.addItem("Parcels by lobe", "parcel_lobe")
        self.metabolite_axis_combo.setToolTip("Choose the x-axis used by metabolite profile plots.")
        self.metabolite_axis_combo.currentIndexChanged.connect(lambda _index: self._sync_proximity_controls())
        self.metabolite_subject_combo = QComboBox()
        self.metabolite_subject_combo.addItem("Average subjects", -1)
        for subject_idx, subject_label in enumerate(self._metabolite_subject_labels):
            self.metabolite_subject_combo.addItem(str(subject_label), int(subject_idx))
        self.metabolite_subject_combo.setToolTip("Choose whether metabolite profiles are averaged across subjects or shown for one subject.")
        self.metabolite_zscore_check = QCheckBox("Standardize values")
        self.metabolite_zscore_check.setChecked(True)
        self.metabolite_zscore_check.setToolTip(
            "Z-score each metabolite profile before plotting. Disable to plot raw metabolite signals."
        )
        self.metabolite_zscore_check.toggled.connect(lambda _checked: self._sync_proximity_controls())
        self.metabolite_correction_combo = QComboBox()
        self.metabolite_correction_combo.addItem("NC", "none")
        self.metabolite_correction_combo.addItem("Cr", "cr")
        self.metabolite_correction_combo.addItem("Metabolite summary", "sum_m")
        self.metabolite_correction_combo.setToolTip(
            "Metabolite correction after z-transform: none, subtract CrPCr, or subtract the mean of non-water metabolite channels."
        )
        self.metabolite_show_water_check = QCheckBox("Show water signal")
        self.metabolite_show_water_check.setChecked(True)
        self.metabolite_show_water_check.setToolTip("Include the water signal curve when a water_signal profile is available.")
        self.metabolite_percentile_band_check = QCheckBox("Show interval band")
        self.metabolite_percentile_band_check.setChecked(True)
        self.metabolite_percentile_band_check.setToolTip("Show the configured percentile interval around the median curve.")
        self.metabolite_settings_button = QPushButton("Settings")
        self.metabolite_settings_button.clicked.connect(self._open_metabolite_settings_dialog)
        self.metabolite_profiles_button = QPushButton("Plot metabolite profiles")
        self.metabolite_profiles_button.clicked.connect(self._on_plot_metabolite_profiles_clicked)

        proximity_controls = QHBoxLayout()
        self.proximity_check = QCheckBox("Proximity circles")
        self.proximity_check.setChecked(bool(self._show_proximity_circles))
        self.proximity_check.toggled.connect(self._on_proximity_toggled)
        slider_orientation = Qt.Orientation.Horizontal if hasattr(Qt, "Orientation") else Qt.Horizontal
        self.proximity_slider = QSlider(slider_orientation)
        self.proximity_slider.setRange(0, self._proximity_slider_steps)
        self.proximity_slider.setValue(int(self._initial_proximity_slider_value))
        self.proximity_slider.valueChanged.connect(self._on_proximity_slider_changed)
        proximity_controls.addWidget(self.proximity_slider, 1)
        self.proximity_value_label = QLabel(self._proximity_label_text())
        proximity_controls.addWidget(self.proximity_value_label, 0)
        self.edge_width_label = QLabel("Edge width")
        self.edge_width_spin = QDoubleSpinBox()
        self.edge_width_spin.setDecimals(2)
        self.edge_width_spin.setRange(0.05, 10.0)
        self.edge_width_spin.setSingleStep(0.05)
        self.edge_width_spin.setValue(float(self._edge_linewidth))
        self.edge_width_spin.setToolTip("Line width used for graph edges in the scatter plot.")
        self.edge_width_spin.valueChanged.connect(self._on_edge_width_changed)
        self.path_width_mode_label = QLabel("Width map")
        self.path_width_mode_combo = QComboBox()
        self.path_width_mode_combo.addItem("Exp", "exp")
        self.path_width_mode_combo.addItem("Linear", "linear")
        self.path_width_mode_combo.addItem("Log", "log")
        self.path_width_mode_combo.setToolTip("Maps path energy to line width when free-energy scaling is available.")
        self.path_width_mode_combo.currentIndexChanged.connect(self._on_path_width_mode_changed)
        self.path_width_scale_label = QLabel("Scale")
        self.path_width_scale_spin = QDoubleSpinBox()
        self.path_width_scale_spin.setDecimals(2)
        self.path_width_scale_spin.setRange(0.05, 1e12)
        self.path_width_scale_spin.setSingleStep(0.5)
        self.path_width_scale_spin.setValue(float(self._path_width_scaling_strength))
        self.path_width_scale_spin.setToolTip("Strength of the energy-to-width mapping.")
        self.path_width_scale_spin.valueChanged.connect(self._on_path_width_scale_changed)

        self.show_adjacency_check = QCheckBox("Show neighboring parcels")
        self.show_adjacency_check.setChecked(True)
        self.show_adjacency_check.setToolTip("Show edges from the underlying parcel adjacency graph.")
        self.show_adjacency_check.toggled.connect(self._on_show_adjacency_toggled)
        self.all_paths_check = QCheckBox("All ordered paths")
        self.all_paths_check.toggled.connect(self._on_all_paths_toggled)
        self.edge_bundling_check = QCheckBox("Bundle edges")
        self.edge_bundling_check.toggled.connect(self._on_edge_bundling_toggled)
        self.free_energy_norm_segments_check = QCheckBox("Normalize by segments")
        self.free_energy_norm_segments_check.setToolTip(
            "Divide each path energy by its number of node-to-node segments before computing free energy."
        )
        self.free_energy_norm_segments_check.setChecked(True)
        self.free_energy_norm_segments_check.toggled.connect(self._on_free_energy_norm_segments_toggled)
        self.free_energy_lambda_label = QLabel("Lambda")
        self.free_energy_lambda_spin = QDoubleSpinBox()
        self.free_energy_lambda_spin.setDecimals(3)
        self.free_energy_lambda_spin.setRange(0.001, 1000.0)
        self.free_energy_lambda_spin.setSingleStep(0.1)
        self.free_energy_lambda_spin.setValue(1.0)
        self.free_energy_lambda_spin.setToolTip("Temperature parameter for free energy: larger values make high-energy paths contribute less.")
        self.free_energy_lambda_spin.valueChanged.connect(self._on_free_energy_lambda_changed)
        self.generate_paths_button = QPushButton("Generate paths")
        self.generate_paths_button.clicked.connect(self._on_generate_paths_clicked)
        self.compute_free_energy_button = QPushButton("Compute free energy")
        self.compute_free_energy_button.setToolTip("Compute the free-energy summary for the generated path families.")
        self.compute_free_energy_button.setEnabled(False)
        self.compute_free_energy_button.clicked.connect(self._on_compute_free_energy_clicked)
        self.write_free_energy_button = QPushButton("Write free energy")
        self.write_free_energy_button.setEnabled(False)
        self.write_free_energy_button.clicked.connect(self._on_write_free_energy_clicked)
        self.project_paths_button = QPushButton("Project to 3D brain")
        self.project_paths_button.setEnabled(False)
        self.project_paths_button.clicked.connect(self._on_project_paths_clicked)
        self.fibrenet_paths_button = QPushButton("Project to FibreNet")
        self.fibrenet_paths_button.setEnabled(False)
        self.fibrenet_paths_button.clicked.connect(self._on_fibrenet_paths_clicked)
        self.fibrenet_layout_label = QLabel("FibreNet layout")
        self.fibrenet_layout_combo = QComboBox()
        self.fibrenet_layout_combo.addItem("Diffusion", "diffusion")
        self.fibrenet_layout_combo.addItem("Spiral", "spiral")
        self.fibrenet_layout_combo.setToolTip(
            "Choose whether FibreNet places nodes by GM diffusion components or along the selected path sequence."
        )
        self.fibrenet_layout_combo.currentIndexChanged.connect(self._on_fibrenet_layout_changed)
        self.screenshot_paths_button = QPushButton("Save 3D screenshot")
        self.screenshot_paths_button.setEnabled(False)
        self.screenshot_paths_button.clicked.connect(self._on_screenshot_paths_clicked)
        self.export_paths_button = QPushButton("Export paths")
        self.export_paths_button.setEnabled(False)
        self.export_paths_button.clicked.connect(self._on_export_paths_clicked)

        self.left_path_combo = QComboBox()
        self.left_path_combo.currentIndexChanged.connect(
            lambda _index: self._on_ctx_path_selection_changed("lh")
        )
        self.right_path_combo = QComboBox()
        self.right_path_combo.currentIndexChanged.connect(
            lambda _index: self._on_ctx_path_selection_changed("rh")
        )

        self.path_order_first_combo = QComboBox()
        self.path_order_first_combo.currentIndexChanged.connect(
            lambda _index: self._on_path_order_combo_changed(0)
        )
        self.path_order_second_combo = QComboBox()
        self.path_order_second_combo.currentIndexChanged.connect(
            lambda _index: self._on_path_order_combo_changed(1)
        )
        self.path_order_third_combo = QComboBox()
        self.path_order_third_combo.currentIndexChanged.connect(
            lambda _index: self._on_path_order_combo_changed(2)
        )

        self.endpoint_mode_combo = QComboBox()
        self.endpoint_mode_combo.addItem("Adaptive", "adaptive")
        self.endpoint_mode_combo.addItem("Average gradients", "average")
        self.endpoint_mode_combo.addItem("Manual click", "manual")
        self.endpoint_mode_combo.currentIndexChanged.connect(self._on_endpoint_mode_changed)
        self.manual_endpoint_target_combo = QComboBox()
        self.manual_endpoint_target_combo.currentIndexChanged.connect(
            self._on_manual_endpoint_target_changed
        )
        self.load_free_energy_endpoints_button = QPushButton("Load endpoints")
        self.load_free_energy_endpoints_button.setToolTip(
            "Load endpoint anchors from a desc-free_energy_paths.npz archive."
        )
        self.load_free_energy_endpoints_button.clicked.connect(self._on_load_free_energy_endpoints_clicked)
        self.clear_manual_endpoints_button = QPushButton("Clear manual")
        self.clear_manual_endpoints_button.clicked.connect(self._on_clear_manual_endpoints_clicked)

        sidebar = QWidget(self)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(8, 8, 8, 8)
        sidebar_layout.setSpacing(8)

        view_section = CollapsibleSection("View")
        view_form = QFormLayout()
        view_form.addRow("Preset", self.interface_preset_combo)
        view_form.addRow("Mode", self.interface_mode_combo)
        view_section.addLayout(view_form)
        view_section.addWidget(self.show_adjacency_check)
        view_section.addWidget(self.proximity_check)
        view_section.addLayout(proximity_controls)
        advanced_view_form = QFormLayout()
        advanced_view_form.addRow(self.edge_width_label, self.edge_width_spin)
        advanced_view_form.addRow(self.path_width_mode_label, self.path_width_mode_combo)
        advanced_view_form.addRow(self.path_width_scale_label, self.path_width_scale_spin)
        view_section.addLayout(advanced_view_form)

        metabolite_section = CollapsibleSection("Metabolites / Coloring")
        metabolite_form = QFormLayout()
        metabolite_form.addRow("X axis", self.metabolite_axis_combo)
        metabolite_form.addRow(self.metabolite_bins_label, self.metabolite_bins_spin)
        metabolite_form.addRow("Subject", self.metabolite_subject_combo)
        metabolite_form.addRow("Correction", self.metabolite_correction_combo)
        metabolite_section.addLayout(metabolite_form)
        metabolite_section.addWidget(self.metabolite_zscore_check)
        metabolite_section.addWidget(self.metabolite_show_water_check)
        metabolite_section.addWidget(self.metabolite_percentile_band_check)
        metabolite_section.addWidget(self.metabolite_settings_button)
        metabolite_section.addWidget(self.metabolite_profiles_button)

        path_section = CollapsibleSection("Path Generation")
        rgb_form = QFormLayout()
        rgb_form.addRow("RGB first", self.path_order_first_combo)
        rgb_form.addRow("RGB second", self.path_order_second_combo)
        rgb_form.addRow("RGB third", self.path_order_third_combo)
        path_section.addLayout(rgb_form)
        endpoint_form = QFormLayout()
        endpoint_form.addRow("Endpoints", self.endpoint_mode_combo)
        endpoint_form.addRow("Click target", self.manual_endpoint_target_combo)
        path_section.addLayout(endpoint_form)
        path_section.addWidget(self.load_free_energy_endpoints_button)
        path_section.addWidget(self.clear_manual_endpoints_button)
        path_section.addWidget(self.generate_paths_button)
        path_section.addWidget(self.compute_free_energy_button)
        free_energy_form = QFormLayout()
        free_energy_form.addRow(self.free_energy_lambda_label, self.free_energy_lambda_spin)
        path_section.addLayout(free_energy_form)
        path_section.addWidget(self.free_energy_norm_segments_check)

        visualization_section = CollapsibleSection("Visualization")
        selected_path_form = QFormLayout()
        selected_path_form.addRow("LH path", self.left_path_combo)
        selected_path_form.addRow("RH path", self.right_path_combo)
        visualization_section.addLayout(selected_path_form)
        visualization_section.addWidget(self.project_paths_button)
        visualization_section.addWidget(self.fibrenet_paths_button)
        fibrenet_form = QFormLayout()
        fibrenet_form.addRow(self.fibrenet_layout_label, self.fibrenet_layout_combo)
        visualization_section.addLayout(fibrenet_form)
        visualization_section.addWidget(self.all_paths_check)
        visualization_section.addWidget(self.edge_bundling_check)

        export_section = CollapsibleSection("Export")
        export_section.addWidget(self.save_button)
        export_section.addWidget(self.screenshot_paths_button)
        export_section.addWidget(self.export_paths_button)
        export_section.addWidget(self.write_free_energy_button)
        export_section.addWidget(self.settings_button)

        sidebar_layout.addWidget(self.info_label)
        sidebar_layout.addWidget(view_section)
        sidebar_layout.addWidget(self._sidebar_separator())
        sidebar_layout.addWidget(metabolite_section)
        sidebar_layout.addWidget(self._sidebar_separator())
        sidebar_layout.addWidget(path_section)
        sidebar_layout.addWidget(self._sidebar_separator())
        sidebar_layout.addWidget(visualization_section)
        sidebar_layout.addWidget(self._sidebar_separator())
        sidebar_layout.addWidget(export_section)
        sidebar_layout.addStretch(1)
        sidebar_scroll = QScrollArea(self)
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setWidget(sidebar)
        sidebar_scroll.setMinimumWidth(280)
        sidebar_scroll.setMaximumWidth(380)

        self.toolbar.setMaximumHeight(34)
        plot_layout = QVBoxLayout()
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(2)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas, 1)
        plot_layout.addWidget(self.status_label, 0)

        layout = QHBoxLayout(self)
        layout.addWidget(sidebar_scroll, 0)
        layout.addLayout(plot_layout, 1)
        self._advanced_widgets = [
            self.edge_width_label,
            self.edge_width_spin,
            self.path_width_mode_label,
            self.path_width_mode_combo,
            self.path_width_scale_label,
            self.path_width_scale_spin,
            self.free_energy_lambda_label,
            self.free_energy_lambda_spin,
            self.free_energy_norm_segments_check,
            self.fibrenet_layout_label,
            self.fibrenet_layout_combo,
            self.all_paths_check,
            self.edge_bundling_check,
        ]
        self.set_theme(theme_name)
        self._populate_path_order_combos()
        self._populate_path_selection_combos()
        self._populate_manual_endpoint_targets()
        self._sync_proximity_controls()
        self._sync_interface_mode()
        self._ensure_hover_callback()

        self._render()
        if self._auto_preload_matching_paths:
            self._auto_preload_matching_free_energy_paths()

    @staticmethod
    def _sidebar_separator():
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine if hasattr(QFrame, "Shape") else QFrame.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken if hasattr(QFrame, "Shadow") else QFrame.Sunken)
        return separator

    def _sync_interface_mode(self):
        advanced = True
        try:
            advanced = str(self.interface_mode_combo.currentData() or "basic") == "advanced"
        except Exception:
            advanced = False
        for widget in list(getattr(self, "_advanced_widgets", [])):
            try:
                widget.setVisible(bool(advanced))
            except Exception:
                pass
        if hasattr(self, "status_label"):
            self.status_label.setText(self._status_text())

    def _on_interface_preset_changed(self, _index):
        try:
            preset = str(self.interface_preset_combo.currentData() or "default")
        except Exception:
            preset = "default"
        advanced_mode = preset in {"path_exploration", "debug"}
        try:
            self.interface_mode_combo.blockSignals(True)
            self.interface_mode_combo.setCurrentIndex(1 if advanced_mode else 0)
        finally:
            self.interface_mode_combo.blockSignals(False)
        if preset == "publication":
            self.show_adjacency_check.setChecked(False)
            self.proximity_check.setChecked(False)
            self.all_paths_check.setChecked(False)
            self.edge_bundling_check.setChecked(False)
        elif preset == "path_exploration":
            self.show_adjacency_check.setChecked(True)
            self.proximity_check.setChecked(bool(self._proximity_max_radius > 0.0))
            self.all_paths_check.setChecked(True)
        elif preset == "triangle_rgb":
            index = self.metabolite_axis_combo.findData("triangular_rgb")
            if index >= 0:
                self.metabolite_axis_combo.setCurrentIndex(index)
            self.metabolite_zscore_check.setChecked(True)
        elif preset == "debug":
            self.show_adjacency_check.setChecked(True)
            self.proximity_check.setChecked(True)
            self.all_paths_check.setChecked(True)
            self.edge_bundling_check.setChecked(False)
        else:
            self.show_adjacency_check.setChecked(True)
            self.proximity_check.setChecked(bool(self._show_proximity_circles))
            self.all_paths_check.setChecked(False)
            self.edge_bundling_check.setChecked(False)
        self._sync_proximity_controls()
        self._render()

    def _status_text(self):
        parts = []
        if isinstance(self._project_paths_payload, dict):
            path_text = self._path_count_summary_text()
            if path_text:
                parts.append(path_text.strip(" |"))
            else:
                parts.append("No generated paths")
        else:
            parts.append("No paths loaded")
        try:
            parts.append(f"lambda={float(self.free_energy_lambda_spin.value()):.3g}")
        except Exception:
            pass
        parts.append(f"mode={self.interface_mode_combo.currentText()}")
        loaded_endpoint_file = str(getattr(self, "_loaded_fixed_endpoint_file", "") or "").strip()
        if loaded_endpoint_file:
            parts.append(f"endpoints={Path(loaded_endpoint_file).name}")
        anchor_text = self._anchor_status_text()
        if anchor_text:
            parts.append(anchor_text)
        if bool(getattr(self, "_normalize_free_energy_by_segments", False)):
            parts.append("energy/segment")
        free_energies = []
        if isinstance(self._project_paths_payload, dict):
            free_energy_payload = self._project_paths_payload.get("free_energy_payload")
            if isinstance(free_energy_payload, dict):
                for group in list(free_energy_payload.get("groups", [])):
                    for family in list(dict(group).get("families", [])):
                        try:
                            value = float(dict(family).get("free_energy"))
                        except Exception:
                            continue
                        if np.isfinite(value):
                            free_energies.append(value)
        if free_energies:
            parts.append(f"mean F={float(np.mean(np.asarray(free_energies, dtype=float))):.4g}")
        return " | ".join(parts)

    def _anchor_status_text(self):
        if not self._use_triangular_rgb or self._is_3d:
            return ""
        signature = self._current_endpoint_anchor_signature()
        if not signature:
            return ""
        parts = []
        for group_name in sorted(signature):
            anchors = dict(signature.get(group_name, {}) or {})
            if not anchors:
                continue
            label = self._group_display_name(group_name) if group_name != "all" else "All"
            values = []
            for channel in ("R", "G", "B", "SUBC"):
                if channel in anchors:
                    values.append(f"{channel}:{int(anchors[channel]) + 1}")
            if values:
                parts.append(f"{label} {'/'.join(values)}")
        return "anchors=" + "; ".join(parts) if parts else ""

    def set_theme(self, theme_name="Dark"):
        theme, style = _dialog_theme_stylesheet(theme_name)
        self._theme_name = theme
        self.setStyleSheet(style)
        if self._appearance_dialog is not None:
            self._appearance_dialog.set_theme(theme)
        if self._metabolite_settings_dialog is not None:
            self._metabolite_settings_dialog.set_theme(theme)

    @staticmethod
    def _default_metabolite_plot_settings():
        return {
            "x_axis_label": "Gradient 1",
            "y_axis_label": "Gradient 2",
            "x_axis_fontsize": 11,
            "y_axis_fontsize": 11,
            "tick_fontsize": 9,
            "line_width": 1.6,
            "confidence_interval": 95.0,
            "boxplot_bars": False,
            "colors": {
                key: color for key, _label, color in METABOLITE_COLOR_SPECS
            },
        }

    @classmethod
    def _sanitize_metabolite_plot_settings(cls, settings):
        defaults = cls._default_metabolite_plot_settings()
        raw = dict(settings or {})
        sanitized = {
            "x_axis_label": str(raw.get("x_axis_label", defaults["x_axis_label"]) or defaults["x_axis_label"]).strip(),
            "y_axis_label": str(raw.get("y_axis_label", defaults["y_axis_label"]) or defaults["y_axis_label"]).strip(),
        }
        for key in ("x_axis_fontsize", "y_axis_fontsize", "tick_fontsize"):
            try:
                value = int(raw.get(key, defaults[key]))
            except Exception:
                value = int(defaults[key])
            sanitized[key] = max(6, min(48, value))
        try:
            line_width = float(raw.get("line_width", defaults["line_width"]))
        except Exception:
            line_width = float(defaults["line_width"])
        sanitized["line_width"] = max(0.2, min(12.0, line_width))
        try:
            confidence_interval = float(raw.get("confidence_interval", defaults["confidence_interval"]))
        except Exception:
            confidence_interval = float(defaults["confidence_interval"])
        sanitized["confidence_interval"] = max(1.0, min(100.0, confidence_interval))
        sanitized["boxplot_bars"] = bool(raw.get("boxplot_bars", defaults["boxplot_bars"]))
        default_colors = dict(defaults.get("colors", {}) or {})
        raw_colors = dict(raw.get("colors", {}) or {})
        sanitized["colors"] = {}
        for key, _label, default_color in METABOLITE_COLOR_SPECS:
            color = str(raw_colors.get(key, default_colors.get(key, default_color)) or default_color).strip()
            sanitized["colors"][key] = color if color else str(default_color)
        return sanitized

    def _current_metabolite_plot_settings(self):
        current = self._sanitize_metabolite_plot_settings(getattr(self, "_metabolite_plot_settings", None))
        self._metabolite_plot_settings = current
        return current

    def _apply_metabolite_plot_settings(self, settings):
        self._metabolite_plot_settings = self._sanitize_metabolite_plot_settings(settings)

    def _open_metabolite_settings_dialog(self):
        if self._metabolite_settings_dialog is None:
            self._metabolite_settings_dialog = MetabolitePlotSettingsDialog(
                self,
                theme_name=self._theme_name,
                parent=self,
            )
        self._metabolite_settings_dialog.set_theme(self._theme_name)
        self._metabolite_settings_dialog.sync_from_settings(self._current_metabolite_plot_settings())
        self._metabolite_settings_dialog.show()
        try:
            self._metabolite_settings_dialog.raise_()
            self._metabolite_settings_dialog.activateWindow()
        except Exception:
            pass

    def _default_scatter_appearance_settings(self):
        subplot_specs = list(self._display_group_specs())
        subplot_titles = {}
        for spec in subplot_specs:
            key = str(spec.get("name", "all")).strip().lower()
            if len(subplot_specs) <= 1:
                subplot_titles[key] = ""
            else:
                subplot_titles[key] = str(spec.get("title", "") or "").strip()
        return {
            "figure_title": str(self._title or "").strip(),
            "axis_label_fontsize": 12,
            "tick_label_fontsize": 11,
            "subplot_titles": subplot_titles,
        }

    @staticmethod
    def _sanitize_scatter_appearance_settings(raw_settings, defaults):
        defaults = dict(defaults or {})
        raw = dict(raw_settings or {})
        subplot_defaults = dict(defaults.get("subplot_titles", {}) or {})
        subplot_raw = dict(raw.get("subplot_titles", {}) or {})
        try:
            axis_label_fontsize = int(raw.get("axis_label_fontsize", defaults.get("axis_label_fontsize", 12)))
        except Exception:
            axis_label_fontsize = int(defaults.get("axis_label_fontsize", 12))
        try:
            tick_label_fontsize = int(raw.get("tick_label_fontsize", defaults.get("tick_label_fontsize", 11)))
        except Exception:
            tick_label_fontsize = int(defaults.get("tick_label_fontsize", 11))
        return {
            "figure_title": str(raw.get("figure_title", defaults.get("figure_title", "")) or "").strip(),
            "axis_label_fontsize": max(6, min(48, axis_label_fontsize)),
            "tick_label_fontsize": max(6, min(48, tick_label_fontsize)),
            "subplot_titles": {
                str(key): str(subplot_raw.get(key, subplot_defaults.get(key, "")) or "").strip()
                for key in subplot_defaults.keys()
            },
        }

    def _current_scatter_appearance_settings(self):
        defaults = self._default_scatter_appearance_settings()
        current = self._sanitize_scatter_appearance_settings(
            getattr(self, "_scatter_appearance_settings", None),
            defaults,
        )
        self._scatter_appearance_settings = current
        return current

    def _scatter_subplot_title(self, subplot_spec, subplot_count: int) -> str:
        settings = self._current_scatter_appearance_settings()
        titles = dict(settings.get("subplot_titles", {}) or {})
        key = str(subplot_spec.get("name", "all")).strip().lower()
        if key in titles:
            return str(titles.get(key, "") or "").strip()
        if subplot_count <= 1:
            return ""
        return str(subplot_spec.get("title", "") or "").strip()

    def _apply_scatter_appearance_settings(self, settings) -> None:
        defaults = self._default_scatter_appearance_settings()
        self._scatter_appearance_settings = self._sanitize_scatter_appearance_settings(settings, defaults)
        current_title = str(self._scatter_appearance_settings.get("figure_title", "") or "").strip()
        self.setWindowTitle(current_title or str(self._title or "Gradient Scatter"))
        self._render()

    def _open_appearance_dialog(self) -> None:
        subplot_specs = list(self._display_group_specs())
        if self._appearance_dialog is None:
            self._appearance_dialog = ScatterAppearanceDialog(
                self,
                theme_name=self._theme_name,
                parent=self,
            )
        self._appearance_dialog.set_theme(self._theme_name)
        self._appearance_dialog.sync_from_settings(self._current_scatter_appearance_settings(), subplot_specs)
        self._appearance_dialog.show()
        try:
            self._appearance_dialog.raise_()
            self._appearance_dialog.activateWindow()
        except Exception:
            pass

    def _info_text(self):
        mode = (
            f"{'3D Pyramid' if self._is_3d else self._rgb_fit_mode.title()} {self._triangular_color_order}"
            if self._use_triangular_rgb
            else f"Cmap: {self._cmap_name}"
        )
        edge_text = ""
        if self._edge_pairs.size:
            edge_text = f" | Adjacency: {self._edge_pairs.shape[0]}"
            edge_text += f" | Visible: {self._active_edge_count()}"
            if self._show_all_ordered_paths and self._use_triangular_rgb:
                edge_text += " | All paths"
            if self._use_edge_bundling:
                edge_text += " | Bundled"
                if self._edge_bundling_note:
                    edge_text += f" | {self._edge_bundling_note}"
        endpoint_text = f" | Path: {self._path_channel_order}" if self._use_triangular_rgb else ""
        if self._is_3d:
            endpoint_text = " | 3D"
        path_text = self._path_count_summary_text()
        hemisphere_text = f" | Hemi: {self._hemisphere_mode.upper()}"
        endpoint_mode_text = (
            f" | Endpoints: {self._endpoint_mode_display_text()}"
            if self._use_triangular_rgb and not self._is_3d
            else ""
        )
        manual_target_text = ""
        if self._use_triangular_rgb and not self._is_3d and self._endpoint_selection_mode == "manual":
            current_target = self.manual_endpoint_target_combo.currentText().strip()
            if current_target:
                manual_target_text = f" | Click: {current_target}"
        return (
            f"Points: {self._x.size}{edge_text}{path_text}{endpoint_text}{endpoint_mode_text}{manual_target_text}"
            f"{hemisphere_text} | Rotation: {self._rotation_preset} | {mode}"
        )

    @staticmethod
    def _normalize_rotation_preset(value):
        text = str(value or "Default").strip()
        valid = {"Default", "+90", "-90", "180"}
        if text not in valid:
            text = "Default"
        return text

    @staticmethod
    def _normalize_scatter_hemisphere_mode(value):
        text = str(value or "both").strip().lower()
        if text not in {"both", "lh", "rh", "separate"}:
            text = "both"
        return text

    @staticmethod
    def _rotate_points(x_values, y_values, preset):
        if preset == "+90":
            return -y_values, x_values
        if preset == "-90":
            return y_values, -x_values
        if preset == "180":
            return -x_values, -y_values
        return x_values, y_values

    def _rotated_rgb_points(self):
        if self._rgb_x is None or self._rgb_y is None:
            return self._rotate_points(self._x, self._y, self._rotation_preset)
        return self._rotate_points(self._rgb_x, self._rgb_y, self._rotation_preset)

    def _rotated_rgb_points_3d(self):
        if not self._is_3d:
            rgb_x, rgb_y = self._rotated_rgb_points()
            return rgb_x, rgb_y, None
        if self._rgb_x is None or self._rgb_y is None or self._rgb_z is None:
            rgb_x, rgb_y = self._rotate_points(self._x, self._y, self._rotation_preset)
            return rgb_x, rgb_y, np.asarray(self._z, dtype=float)
        rgb_x, rgb_y = self._rotate_points(self._rgb_x, self._rgb_y, self._rotation_preset)
        return rgb_x, rgb_y, np.asarray(self._rgb_z, dtype=float)

    @staticmethod
    def _json_ready_rgb_model(model):
        payload = {}
        for key, value in dict(model or {}).items():
            if isinstance(value, np.ndarray):
                payload[key] = np.asarray(value).tolist()
            elif isinstance(value, (list, tuple)):
                payload[key] = np.asarray(value).tolist()
            elif isinstance(value, np.generic):
                payload[key] = value.item()
            else:
                payload[key] = value
        return payload

    def triangular_rgb_model_payload(self):
        if not self._use_triangular_rgb or self._is_3d:
            return None
        rgb_x_plot, rgb_y_plot = self._rotated_rgb_points()
        model = self._last_rgb_model
        if model is None:
            try:
                model = self._rgb_model(
                    rgb_x_plot,
                    rgb_y_plot,
                    self._triangular_color_order,
                    fit_mode=self._rgb_fit_mode,
                )
            except Exception:
                return None
        return {
            "version": 1,
            "enabled": True,
            "fit_mode": str(self._rgb_fit_mode or "triangle"),
            "color_order": str(self._triangular_color_order or "RBG"),
            "scalar_mode": str(self._rgb_scalar_mode or "barycentric"),
            "rotation_preset": str(self._rotation_preset or "Default"),
            "x_label": str(self._x_label or "Gradient 2"),
            "y_label": str(self._y_label or "Gradient 1"),
            "model": self._json_ready_rgb_model(model),
            "point_ids": [str(value) for value in self._point_ids.tolist()],
            "point_labels": [str(value) for value in self._point_labels.tolist()],
            "rgb_x_values": np.asarray(rgb_x_plot, dtype=float).tolist(),
            "rgb_y_values": np.asarray(rgb_y_plot, dtype=float).tolist(),
            "group_codes": np.asarray(self._point_group_codes, dtype=int).tolist(),
        }

    @staticmethod
    def _rgb_scalar_from_model(x_values, y_values, model, scalar_mode="barycentric"):
        x_valid = np.asarray(x_values, dtype=float).reshape(-1)
        y_valid = np.asarray(y_values, dtype=float).reshape(-1)
        scalar_values = np.full(x_valid.shape, np.nan, dtype=float)
        finite_mask = np.isfinite(x_valid) & np.isfinite(y_valid)
        if not np.any(finite_mask):
            return scalar_values

        fit_mode = GradientScatterDialog._normalize_rgb_fit_mode(model.get("fit_mode", "triangle"))
        scalar_map = {"R": 0.0, "B": 0.5, "G": 1.0}
        if fit_mode != "square":
            try:
                barycentric_scalar = np.asarray(
                    nettools.triangular_rgb_scalar_from_model(
                        x_valid,
                        y_valid,
                        model,
                        channel_scalar_map=scalar_map,
                    ),
                    dtype=float,
                )
                mode = GradientScatterDialog._normalize_rgb_scalar_mode(scalar_mode)
                if mode != "principal_curve":
                    return barycentric_scalar
                if hasattr(nettools, "principal_curve_scalar"):
                    principal_scalar = np.asarray(
                        nettools.principal_curve_scalar(x_valid, y_valid),
                        dtype=float,
                    )
                else:
                    principal_scalar = GradientScatterDialog._principal_curve_scalar(x_valid, y_valid)
                comparable = np.isfinite(principal_scalar) & np.isfinite(barycentric_scalar)
                if int(np.sum(comparable)) >= 3:
                    try:
                        corr = float(np.corrcoef(principal_scalar[comparable], barycentric_scalar[comparable])[0, 1])
                    except Exception:
                        corr = np.nan
                    if np.isfinite(corr) and corr < 0.0:
                        principal_scalar = 1.0 - principal_scalar
                return principal_scalar
            except Exception:
                return scalar_values

        order = [str(channel).strip().upper() for channel in str(model.get("order", "RBG"))]
        vertex_scalars = np.asarray(
            [scalar_map.get(channel, float(idx)) for idx, channel in enumerate(order[:3])],
            dtype=float,
        )
        try:
            anchor_points = np.asarray(model["anchor_points"], dtype=float)[:3, :]
        except Exception:
            return scalar_values
        if anchor_points.ndim != 2 or anchor_points.shape[0] != vertex_scalars.size:
            return scalar_values
        points = np.column_stack((x_valid[finite_mask], y_valid[finite_mask]))
        deltas = points[:, np.newaxis, :] - anchor_points[np.newaxis, :, :]
        distances = np.sqrt(np.sum(np.square(deltas), axis=2))
        weights = 1.0 / np.maximum(distances, 1e-9)
        close_mask = distances <= 1e-9
        if np.any(close_mask):
            for row_idx in np.flatnonzero(np.any(close_mask, axis=1)).tolist():
                weights[row_idx, :] = close_mask[row_idx, :].astype(float)
        weight_sum = weights.sum(axis=1, keepdims=True)
        weight_sum[weight_sum <= 0] = 1.0
        scalar_values[finite_mask] = (weights / weight_sum) @ vertex_scalars
        return scalar_values

    @staticmethod
    def _principal_curve_scalar(x_values, y_values, *, n_curve_points=512):
        x_valid = np.asarray(x_values, dtype=float).reshape(-1)
        y_valid = np.asarray(y_values, dtype=float).reshape(-1)
        scalar_values = np.full(x_valid.shape, np.nan, dtype=float)
        finite_mask = np.isfinite(x_valid) & np.isfinite(y_valid)
        if int(np.sum(finite_mask)) < 2:
            return scalar_values
        points = np.column_stack((x_valid[finite_mask], y_valid[finite_mask]))
        centered = points - np.nanmean(points, axis=0, keepdims=True)
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            direction = np.asarray(vh[0], dtype=float)
        except Exception:
            direction = np.array((1.0, 0.0), dtype=float)
        scores = centered @ direction
        score_min = float(np.nanmin(scores))
        score_max = float(np.nanmax(scores))
        if not np.isfinite(score_min) or not np.isfinite(score_max) or score_max <= score_min:
            scalar_values[finite_mask] = 0.0
            return scalar_values
        t = (scores - score_min) / (score_max - score_min)
        order = np.argsort(t)
        t_sorted = np.asarray(t[order], dtype=float)
        points_sorted = np.asarray(points[order, :], dtype=float)
        unique_t, inverse = np.unique(t_sorted, return_inverse=True)
        if unique_t.size < 2:
            scalar_values[finite_mask] = t
            return scalar_values
        unique_points = np.zeros((unique_t.size, 2), dtype=float)
        counts = np.zeros(unique_t.size, dtype=float)
        for idx, group_idx in enumerate(inverse.tolist()):
            unique_points[group_idx, :] += points_sorted[idx, :]
            counts[group_idx] += 1.0
        counts[counts <= 0.0] = 1.0
        unique_points /= counts[:, np.newaxis]
        n_curve_points = max(32, min(4096, int(n_curve_points)))
        curve_t = np.linspace(float(unique_t[0]), float(unique_t[-1]), n_curve_points)
        curve_points = None
        if unique_t.size >= 4:
            try:
                from scipy.interpolate import UnivariateSpline

                centered_curve = unique_points - np.nanmean(unique_points, axis=0, keepdims=True)
                variance = float(np.nanmean(np.sum(np.square(centered_curve), axis=1)))
                smoothing_value = max(0.0, 0.0025 * unique_t.size * variance)
                spline_x = UnivariateSpline(unique_t, unique_points[:, 0], s=smoothing_value)
                spline_y = UnivariateSpline(unique_t, unique_points[:, 1], s=smoothing_value)
                curve_points = np.column_stack((spline_x(curve_t), spline_y(curve_t)))
            except Exception:
                curve_points = None
        if curve_points is None:
            curve_points = np.column_stack(
                (
                    np.interp(curve_t, unique_t, unique_points[:, 0]),
                    np.interp(curve_t, unique_t, unique_points[:, 1]),
                )
            )
        segment_lengths = np.sqrt(np.sum(np.square(np.diff(curve_points, axis=0)), axis=1))
        arc_length = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        total_length = float(arc_length[-1])
        if not np.isfinite(total_length) or total_length <= 1e-12:
            scalar_values[finite_mask] = t
            return scalar_values
        distances = np.sqrt(np.sum(np.square(points[:, np.newaxis, :] - curve_points[np.newaxis, :, :]), axis=2))
        nearest = np.argmin(distances, axis=1)
        scalar_values[finite_mask] = arc_length[nearest] / total_length
        return scalar_values

    @staticmethod
    def _rotate_axis_labels(x_label, y_label, preset):
        if preset == "+90":
            return f"-{y_label}", x_label
        if preset == "-90":
            return y_label, f"-{x_label}"
        if preset == "180":
            return f"-{x_label}", f"-{y_label}"
        return x_label, y_label

    @staticmethod
    def _compute_display_range(values):
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return -1.0, 1.0
        vmin = float(np.percentile(finite, 2))
        vmax = float(np.percentile(finite, 98))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
            vmin = float(np.nanmin(finite))
            vmax = float(np.nanmax(finite))
        if np.isclose(vmin, vmax):
            vmin -= 1.0
            vmax += 1.0
        return vmin, vmax

    @staticmethod
    def _normalize_triangular_color_order(value):
        text = str(value or "RBG").strip().upper()
        valid = {"RGB", "RBG", "GRB", "GBR", "BRG", "BGR"}
        if text not in valid:
            text = "RBG"
        return text

    @staticmethod
    def _normalize_rgb_fit_mode(value):
        text = str(value or "triangle").strip().lower()
        if text not in {"triangle", "square"}:
            text = "triangle"
        return text

    @staticmethod
    def _normalize_rgb_scalar_mode(value):
        text = str(value or "barycentric").strip().lower().replace("-", "_").replace(" ", "_")
        mapping = {
            "bary": "barycentric",
            "barycentric": "barycentric",
            "principal": "principal_curve",
            "principal_curve": "principal_curve",
            "principalcurve": "principal_curve",
        }
        return mapping.get(text, "barycentric")

    @staticmethod
    def _normalize_path_channel(value):
        text = str(value or "").strip().upper()
        return text if text in {"R", "G", "B"} else "R"

    @classmethod
    def _coerce_path_channel_order(cls, values, fallback="RBG"):
        requested = []
        if isinstance(values, (str, bytes)):
            requested = [char for char in str(values).strip().upper()]
        else:
            requested = [str(value or "").strip().upper() for value in list(values or [])]
        normalized = []
        for value in requested:
            if value in {"R", "G", "B"} and value not in normalized:
                normalized.append(value)
        for value in str(fallback or "RBG").strip().upper():
            if value in {"R", "G", "B"} and value not in normalized:
                normalized.append(value)
        for value in ("R", "G", "B"):
            if value not in normalized:
                normalized.append(value)
        return "".join(normalized[:3])

    def _default_path_channel_order(self):
        fallback = self._coerce_path_channel_order(self._triangular_color_order)
        if self._display_coords.ndim != 2 or self._display_coords.shape[0] < 3:
            return fallback
        try:
            rgb_model = self._rgb_model(
                self._display_coords[:, 0],
                self._display_coords[:, 1],
                self._triangular_color_order,
                fit_mode=self._rgb_fit_mode,
            )
            anchors = self._rgb_anchor_indices(
                self._display_coords[:, 0],
                self._display_coords[:, 1],
                rgb_model,
            )
        except Exception:
            return fallback
        if not {"R", "G", "B"}.issubset(set(anchors.keys())):
            return fallback
        ranked = []
        fallback_order = [str(channel) for channel in fallback]
        for channel in ("R", "G", "B"):
            anchor_index = int(anchors[channel])
            if anchor_index < 0 or anchor_index >= self._gradient1.shape[0]:
                return fallback
            try:
                fallback_rank = fallback_order.index(channel)
            except Exception:
                fallback_rank = 99
            ranked.append((float(self._gradient1[anchor_index]), -fallback_rank, channel))
        ranked.sort(reverse=True)
        order = "".join(channel for _grad1, _rank, channel in ranked)
        return self._coerce_path_channel_order(order, fallback=fallback)

    @staticmethod
    def _normalize_path_width_scaling_mode(value):
        text = str(value or "exp").strip().lower()
        if text not in {"exp", "linear", "log"}:
            text = "exp"
        return text

    @staticmethod
    def _normalize_edge_pairs(edge_pairs, n_points):
        if edge_pairs is None:
            return np.zeros((0, 2), dtype=int)
        pairs = np.asarray(edge_pairs, dtype=int)
        if pairs.size == 0:
            return np.zeros((0, 2), dtype=int)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("Scatter edge pairs must be an Nx2 array.")
        valid = (
            (pairs[:, 0] >= 0)
            & (pairs[:, 1] >= 0)
            & (pairs[:, 0] < int(n_points))
            & (pairs[:, 1] < int(n_points))
            & (pairs[:, 0] != pairs[:, 1])
        )
        pairs = pairs[valid]
        if pairs.size == 0:
            return np.zeros((0, 2), dtype=int)
        pairs = np.sort(pairs, axis=1)
        return np.unique(pairs, axis=0)

    @staticmethod
    def _compute_max_radius(coords):
        points = np.asarray(coords, dtype=float)
        if points.ndim != 2 or points.shape[0] < 2:
            return 0.0
        deltas = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        distances = np.sqrt(np.sum(np.square(deltas), axis=2))
        return float(np.nanmax(distances))

    @staticmethod
    def _compute_edge_distances(coords, edge_pairs):
        points = np.asarray(coords, dtype=float)
        pairs = np.asarray(edge_pairs, dtype=int)
        if points.ndim != 2 or pairs.ndim != 2 or pairs.shape[0] == 0:
            return np.zeros(0, dtype=float)
        deltas = points[pairs[:, 0], :] - points[pairs[:, 1], :]
        return np.sqrt(np.sum(np.square(deltas), axis=1))

    def _slider_to_radius(self, slider_value):
        try:
            slider = int(slider_value)
        except Exception:
            slider = 0
        slider = max(0, min(self._proximity_slider_steps, slider))
        if self._proximity_slider_steps <= 0 or self._proximity_max_radius <= 0.0:
            return 0.0
        return float(self._proximity_max_radius * (slider / float(self._proximity_slider_steps)))

    @staticmethod
    def _normalize_proximity_slider_value(value, max_steps):
        try:
            slider = int(value)
        except Exception:
            slider = 0
        return max(0, min(int(max_steps), slider))

    def _proximity_label_text(self):
        return f"r = {self._proximity_radius:.4f} / {self._proximity_max_radius:.4f}"

    @staticmethod
    def _compute_fixed_axes(coords):
        points = np.asarray(coords, dtype=float)
        if points.ndim != 2 or points.shape[0] == 0:
            return (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)
        x_values = points[:, 0]
        y_values = points[:, 1]
        x_min = float(np.nanmin(x_values))
        x_max = float(np.nanmax(x_values))
        y_min = float(np.nanmin(y_values))
        y_max = float(np.nanmax(y_values))
        x_span = x_max - x_min
        y_span = y_max - y_min
        if not np.isfinite(x_span) or np.isclose(x_span, 0.0):
            x_pad = max(abs(x_min), abs(x_max), 1.0) * 0.12
        else:
            x_pad = x_span * 0.08
        if not np.isfinite(y_span) or np.isclose(y_span, 0.0):
            y_pad = max(abs(y_min), abs(y_max), 1.0) * 0.12
        else:
            y_pad = y_span * 0.08
        z_lim = (-1.0, 1.0)
        if points.shape[1] >= 3:
            z_values = points[:, 2]
            z_min = float(np.nanmin(z_values))
            z_max = float(np.nanmax(z_values))
            z_span = z_max - z_min
            if not np.isfinite(z_span) or np.isclose(z_span, 0.0):
                z_pad = max(abs(z_min), abs(z_max), 1.0) * 0.12
            else:
                z_pad = z_span * 0.08
            z_lim = (z_min - z_pad, z_max + z_pad)
        return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad), z_lim

    def _sync_proximity_controls(self):
        if self._is_3d:
            for widget in (
                self.proximity_check,
                self.proximity_slider,
                self.edge_width_spin,
                self.path_width_mode_combo,
                self.path_width_scale_spin,
                self.show_adjacency_check,
                self.all_paths_check,
                self.edge_bundling_check,
                self.free_energy_norm_segments_check,
                self.generate_paths_button,
                self.path_order_first_combo,
                self.path_order_second_combo,
                self.path_order_third_combo,
                self.endpoint_mode_combo,
                self.manual_endpoint_target_combo,
                self.load_free_energy_endpoints_button,
                self.clear_manual_endpoints_button,
                self.free_energy_lambda_spin,
                self.project_paths_button,
                self.fibrenet_paths_button,
                self.fibrenet_layout_combo,
                self.screenshot_paths_button,
                self.export_paths_button,
                self.compute_free_energy_button,
                self.write_free_energy_button,
                self.left_path_combo,
                self.right_path_combo,
            ):
                widget.setEnabled(False)
            self.metabolite_bins_label.setEnabled(False)
            self.metabolite_bins_spin.setEnabled(False)
            self.metabolite_axis_combo.setEnabled(False)
            self.metabolite_subject_combo.setEnabled(False)
            self.metabolite_zscore_check.setEnabled(False)
            self.metabolite_correction_combo.setEnabled(False)
            self.metabolite_show_water_check.setEnabled(False)
            self.metabolite_percentile_band_check.setEnabled(False)
            self.metabolite_settings_button.setEnabled(False)
            self.metabolite_profiles_button.setEnabled(False)
            self.proximity_value_label.setText("3D mode")
            if hasattr(self, "status_label"):
                self.status_label.setText(self._status_text())
            self._sync_interface_mode()
            return
        enabled = self._display_coords.shape[0] > 0 and self._proximity_max_radius > 0.0
        self.proximity_check.setEnabled(enabled)
        self.proximity_slider.setEnabled(enabled)
        self.edge_width_spin.setEnabled(self._edge_pairs.shape[0] > 0)
        self.path_width_mode_combo.setEnabled(self._use_triangular_rgb)
        self.path_width_scale_spin.setEnabled(self._use_triangular_rgb)
        self.show_adjacency_check.setEnabled(self._edge_pairs.shape[0] > 0)
        self.all_paths_check.setEnabled(self._use_triangular_rgb and self._edge_pairs.shape[0] > 0)
        self.edge_bundling_check.setEnabled(self._edge_pairs.shape[0] > 0 or self._use_triangular_rgb)
        self.free_energy_norm_segments_check.setEnabled(self._use_triangular_rgb and self._edge_pairs.shape[0] > 0)
        self.generate_paths_button.setEnabled(self._use_triangular_rgb and self._edge_pairs.shape[0] > 0)
        self.path_order_first_combo.setEnabled(self._use_triangular_rgb)
        self.path_order_second_combo.setEnabled(self._use_triangular_rgb)
        self.path_order_third_combo.setEnabled(self._use_triangular_rgb)
        metabolite_enabled = (
            self._use_triangular_rgb
            and self._metabolite_profiles is not None
            and (
                (
                    self._metabolite_profiles.ndim == 2
                    and self._metabolite_profiles.shape[0] == self._x.shape[0]
                    and self._metabolite_profiles.shape[1] > 0
                )
                or (
                    self._metabolite_profiles.ndim == 3
                    and self._metabolite_profiles.shape[1] == self._x.shape[0]
                    and self._metabolite_profiles.shape[2] > 0
                )
            )
        )
        metabolite_axis_mode = str(self.metabolite_axis_combo.currentData() or "triangular_rgb")
        self.metabolite_bins_label.setEnabled(metabolite_enabled and metabolite_axis_mode != "parcel_lobe")
        self.metabolite_bins_spin.setEnabled(metabolite_enabled and metabolite_axis_mode != "parcel_lobe")
        self.metabolite_axis_combo.setEnabled(metabolite_enabled)
        self.metabolite_subject_combo.setEnabled(
            metabolite_enabled
            and self._metabolite_profiles is not None
            and self._metabolite_profiles.ndim == 3
            and self._metabolite_profiles.shape[0] > 1
        )
        self.metabolite_zscore_check.setEnabled(metabolite_enabled)
        self.metabolite_correction_combo.setEnabled(
            metabolite_enabled and bool(self.metabolite_zscore_check.isChecked())
        )
        self.metabolite_show_water_check.setEnabled(metabolite_enabled)
        self.metabolite_percentile_band_check.setEnabled(metabolite_enabled)
        self.metabolite_settings_button.setEnabled(metabolite_enabled)
        self.metabolite_profiles_button.setEnabled(metabolite_enabled)
        self.endpoint_mode_combo.setEnabled(self._use_triangular_rgb)
        self.manual_endpoint_target_combo.setEnabled(self._use_triangular_rgb and self._endpoint_selection_mode == "manual")
        self.load_free_energy_endpoints_button.setEnabled(self._use_triangular_rgb)
        self.clear_manual_endpoints_button.setEnabled(self._use_triangular_rgb and self._endpoint_selection_mode == "manual")
        self.free_energy_lambda_spin.setEnabled(self._use_triangular_rgb)
        projectable = False
        exportable = False
        if isinstance(self._project_paths_payload, dict):
            for group_payload in list(self._project_paths_payload.get("group_paths", [])):
                if len(self._selected_ctx_path_nodes(group_payload)) >= 2:
                    if self._project_paths_callback is not None:
                        projectable = True
                if len(list(group_payload.get("subc_optimal_path", []))) >= 2:
                    if self._project_paths_callback is not None:
                        projectable = True
                if len(list(group_payload.get("all_full_paths", []))) > 0:
                    exportable = True
                if len(list(group_payload.get("subc_paths", []))) > 0:
                    exportable = True
            if (
                self._project_paths_callback is not None
                and not projectable
                and len(self._project_paths_payload.get("optimal_full_path", [])) >= 2
            ):
                projectable = True
            if not exportable and len(list(self._project_paths_payload.get("all_full_paths", []))) > 0:
                exportable = True
        self.project_paths_button.setEnabled(
            projectable
        )
        self.fibrenet_paths_button.setEnabled(projectable)
        self.fibrenet_layout_combo.setEnabled(projectable)
        self.screenshot_paths_button.setEnabled(projectable)
        self.export_paths_button.setEnabled(exportable)
        self.compute_free_energy_button.setEnabled(exportable)
        free_energy_ready = False
        if isinstance(self._project_paths_payload, dict):
            free_energy_payload = self._project_paths_payload.get("free_energy_payload")
            free_energy_ready = isinstance(free_energy_payload, dict) and bool(list(free_energy_payload.get("groups", [])))
        self.write_free_energy_button.setEnabled(free_energy_ready)
        self._sync_path_selection_controls_enabled()
        self.proximity_value_label.setText(self._proximity_label_text())
        if hasattr(self, "status_label"):
            self.status_label.setText(self._status_text())
        self._sync_interface_mode()

    def _path_count_summary_text(self, group_name=None):
        if not self._use_triangular_rgb or not isinstance(self._project_paths_payload, dict):
            return ""
        group_payloads = list(self._project_paths_payload.get("group_paths", []))
        if not group_payloads:
            return ""
        parts = []
        target_group = str(group_name or "").strip().lower()
        for group_payload in group_payloads:
            group_name = str(group_payload.get("group", "all")).strip().lower()
            if target_group and group_name != target_group:
                continue
            ctx_count = int(group_payload.get("ctx_path_count", group_payload.get("full_path_count", len(list(group_payload.get("all_full_paths", []))) or 0)))
            subc_count = int(group_payload.get("subc_path_count", len(list(group_payload.get("subc_paths", []))) or 0))
            if group_name == "lh":
                parts.append(f"LH ctx/subc: {ctx_count}/{subc_count}")
            elif group_name == "rh":
                parts.append(f"RH ctx/subc: {ctx_count}/{subc_count}")
            else:
                parts.append(f"Paths ctx/subc: {ctx_count}/{subc_count}")
        return " | " + " | ".join(parts)

    def _ensure_hover_callback(self):
        if self._is_3d:
            return
        if self._hover_cid is None:
            self._hover_cid = self.canvas.mpl_connect("motion_notify_event", self._on_hover)
        if self._click_cid is None:
            self._click_cid = self.canvas.mpl_connect("button_press_event", self._on_click)

    def _hide_hover_annotation(self):
        changed = False
        for entry in list(self._point_artist_entries):
            annotation = entry.get("annotation")
            if annotation is not None and annotation.get_visible():
                annotation.set_visible(False)
                changed = True
        if changed:
            self.canvas.draw_idle()

    def _path_membership_text(self, node_index):
        if not isinstance(self._project_paths_payload, dict):
            return ""
        labels = []
        node_index = int(node_index)
        for group_payload in list(self._project_paths_payload.get("group_paths", [])):
            group_name = str(group_payload.get("group", "all")).strip().upper()
            anchors = {str(key): int(value) for key, value in dict(group_payload.get("anchors", {})).items()}
            for channel, anchor_index in anchors.items():
                if int(anchor_index) == node_index:
                    labels.append(f"{group_name} {channel} endpoint")
            if int(group_payload.get("subc_anchor", -1)) == node_index:
                labels.append(f"{group_name} SUBC endpoint")
            if node_index in {int(value) for value in self._selected_ctx_path_nodes(group_payload)}:
                labels.append(f"{group_name} selected CTX path")
            if node_index in {int(value) for value in list(group_payload.get("subc_optimal_path", []))}:
                labels.append(f"{group_name} selected SUBC path")
        return ", ".join(dict.fromkeys(labels))

    @staticmethod
    def _format_metabolite_value(value):
        try:
            value = float(value)
        except Exception:
            return ""
        if not np.isfinite(value):
            return ""
        return f"{value:.4g}"

    def _metabolite_hover_lines_for_index(self, global_index, *, values_per_line=3):
        if self._metabolite_profiles is None:
            return []
        try:
            profiles = np.asarray(self._metabolite_profiles, dtype=float)
            if profiles.ndim == 3:
                if global_index < 0 or global_index >= profiles.shape[1]:
                    return []
                values = np.nanmean(profiles[:, global_index, :], axis=0)
                prefix = "Metabolites mean"
            elif profiles.ndim == 2:
                if global_index < 0 or global_index >= profiles.shape[0]:
                    return []
                values = profiles[global_index, :]
                prefix = "Metabolites"
            else:
                return []
            values = np.asarray(values, dtype=float).reshape(-1)
        except Exception:
            return []

        names = list(self._metabolite_names or [])
        if len(names) != values.shape[0]:
            names = [f"M{idx + 1}" for idx in range(values.shape[0])]

        parts = []
        for name, value in zip(names, values):
            formatted = self._format_metabolite_value(value)
            if formatted:
                parts.append(f"{str(name).strip()}={formatted}")
        if not parts:
            return []

        chunk_size = max(1, int(values_per_line))
        lines = []
        for idx in range(0, len(parts), chunk_size):
            label = prefix if idx == 0 else " " * len(prefix)
            lines.append(f"{label}: " + ", ".join(parts[idx : idx + chunk_size]))
        return lines
    def _hover_label_for_index(self, global_index):
        label = (
            str(self._point_labels[global_index])
            if 0 <= global_index < self._point_labels.shape[0]
            else f"Point {global_index + 1}"
        )
        lines = [label]
        if 0 <= global_index < self._x.shape[0]:
            lines.append(f"x={float(self._x[global_index]):.4g}, y={float(self._y[global_index]):.4g}")
            if 0 <= global_index < self._gradient1.shape[0]:
                lines.append(f"gradient1={float(self._gradient1[global_index]):.4g}")
        membership = self._path_membership_text(global_index)
        if membership:
            lines.append(membership)
        lines.extend(self._metabolite_hover_lines_for_index(global_index))
        return "\n".join(lines)

    def _hover_status_text_for_index(self, global_index):
        label = (
            str(self._point_labels[global_index])
            if 0 <= global_index < self._point_labels.shape[0]
            else f"Point {global_index + 1}"
        )
        lines = self._metabolite_hover_lines_for_index(global_index, values_per_line=12)
        if not lines:
            return label
        return f"{label} | " + " ".join(line.strip() for line in lines)

    def _on_hover(self, event):
        if self._point_artist is None or not self._point_artist_entries:
            self._hide_hover_annotation()
            return
        if event.inaxes is None:
            self._hide_hover_annotation()
            return
        for entry in list(self._point_artist_entries):
            annotation = entry.get("annotation")
            if annotation is None:
                continue
            if event.inaxes != entry.get("axes"):
                if annotation.get_visible():
                    annotation.set_visible(False)
                continue
            artist = entry.get("artist")
            contains, details = artist.contains(event)
            if not contains:
                if annotation.get_visible():
                    annotation.set_visible(False)
                continue
            indices = np.asarray(details.get("ind", []), dtype=int).reshape(-1)
            if indices.size == 0:
                if annotation.get_visible():
                    annotation.set_visible(False)
                continue
            local_index = int(indices[0])
            offsets = np.asarray(artist.get_offsets(), dtype=float)
            index_map = np.asarray(entry.get("indices", []), dtype=int).reshape(-1)
            if (
                local_index < 0
                or local_index >= offsets.shape[0]
                or local_index >= index_map.shape[0]
            ):
                if annotation.get_visible():
                    annotation.set_visible(False)
                continue
            global_index = int(index_map[local_index])
            x_coord, y_coord = offsets[local_index]
            annotation.xy = (float(x_coord), float(y_coord))
            annotation.set_text(self._hover_label_for_index(global_index))
            annotation.set_visible(True)
            if hasattr(self, "status_label"):
                self.status_label.setText(self._hover_status_text_for_index(global_index))
            self.canvas.draw_idle()
            return
        self._hide_hover_annotation()

    def _on_click(self, event):
        if (
            self._endpoint_selection_mode != "manual"
            or event.inaxes is None
            or getattr(event, "button", None) not in {1, None}
        ):
            return
        target_group, target_channel = self._current_manual_endpoint_target()
        if not target_group or target_channel not in {"R", "G", "B", "SUBC"}:
            return
        candidate_indices = set(self._candidate_indices_for_group(target_group).tolist())
        for entry in list(self._point_artist_entries):
            if event.inaxes != entry.get("axes"):
                continue
            entry_group = str(entry.get("group", "all")).strip().lower()
            if entry_group not in {"", "all"} and target_group not in {"all", entry_group}:
                continue
            artist = entry.get("artist")
            contains, details = artist.contains(event)
            if not contains:
                continue
            indices = np.asarray(details.get("ind", []), dtype=int).reshape(-1)
            if indices.size == 0:
                continue
            local_index = int(indices[0])
            index_map = np.asarray(entry.get("indices", []), dtype=int).reshape(-1)
            if local_index < 0 or local_index >= index_map.shape[0]:
                continue
            global_index = int(index_map[local_index])
            if global_index not in candidate_indices:
                continue
            self._assign_manual_anchor(target_group, target_channel, global_index)
            self._advance_manual_endpoint_target()
            self._invalidate_generated_paths()
            self._sync_proximity_controls()
            self._render(preserve_view=True)
            return

    def _visible_edge_pairs(self):
        if self._edge_pairs.shape[0] == 0:
            return np.zeros((0, 2), dtype=int)
        if self._proximity_radius <= 0.0:
            return np.zeros((0, 2), dtype=int)
        visible = self._edge_distances <= (2.0 * self._proximity_radius + 1e-12)
        if not np.any(visible):
            return np.zeros((0, 2), dtype=int)
        return np.asarray(self._edge_pairs[visible], dtype=int)

    def _visible_edge_distances(self):
        if self._edge_pairs.shape[0] == 0:
            return np.zeros(0, dtype=float)
        if self._proximity_radius <= 0.0:
            return np.zeros(0, dtype=float)
        visible = self._edge_distances <= (2.0 * self._proximity_radius + 1e-12)
        if not np.any(visible):
            return np.zeros(0, dtype=float)
        return np.asarray(self._path_metric_edge_distances[visible], dtype=float)

    def _active_edge_count(self):
        return int(self._visible_edge_pairs().shape[0])

    @staticmethod
    def _rgb_anchor_indices(x_plot, y_plot, rgb_model, candidate_indices=None):
        points = np.column_stack((np.asarray(x_plot, dtype=float), np.asarray(y_plot, dtype=float)))
        if points.shape[0] == 0:
            return {}
        if candidate_indices is None:
            candidate_indices = np.arange(points.shape[0], dtype=int)
        else:
            candidate_indices = np.asarray(candidate_indices, dtype=int).reshape(-1)
        if candidate_indices.size < 3:
            return {}
        anchor_points = np.asarray(rgb_model.get("anchor_points"), dtype=float)
        order = [str(channel) for channel in rgb_model.get("order", "RBG")]
        anchors = {}
        used = set()
        for anchor_point, channel in zip(anchor_points, order):
            distances = np.sqrt(np.sum(np.square(points[candidate_indices, :] - anchor_point[np.newaxis, :]), axis=1))
            ranking = np.argsort(distances)
            chosen = int(candidate_indices[int(ranking[0])])
            for rank_idx in ranking.tolist():
                idx = int(candidate_indices[int(rank_idx)])
                if idx not in used:
                    chosen = int(idx)
                    break
            used.add(chosen)
            anchors[channel] = chosen
        return anchors

    def _resolved_rgb_anchor_indices(self, x_plot, y_plot, rgb_model, candidate_indices=None, group_name=None):
        candidate_indices = (
            np.arange(np.asarray(x_plot, dtype=float).shape[0], dtype=int)
            if candidate_indices is None
            else np.asarray(candidate_indices, dtype=int).reshape(-1)
        )
        resolved = dict(self._effective_anchor_indices_for_group(group_name))
        if {"R", "G", "B"}.issubset(set(resolved.keys())):
            candidate_set = {int(value) for value in candidate_indices.tolist()}
            anchor_values = [int(resolved[channel]) for channel in ("R", "G", "B")]
            if (
                all(int(value) in candidate_set for value in anchor_values)
                and len(set(anchor_values)) == 3
            ):
                return {channel: int(resolved[channel]) for channel in ("R", "G", "B")}
        return self._rgb_anchor_indices(
            x_plot,
            y_plot,
            rgb_model,
            candidate_indices=candidate_indices,
        )

    @staticmethod
    def _triangular_anchor_indices(x_plot, y_plot, triangle_model, candidate_indices=None):
        return GradientScatterDialog._rgb_anchor_indices(
            x_plot,
            y_plot,
            triangle_model,
            candidate_indices=candidate_indices,
        )

    def _path_group_specs(self):
        codes = np.asarray(self._point_group_codes, dtype=int).reshape(-1)
        if codes.shape[0] != self._x.shape[0]:
            return [{"name": "all", "eligible_mask": np.ones(self._x.shape, dtype=bool)}]
        has_lh = bool(np.any(codes == 0))
        has_rh = bool(np.any(codes == 1))
        if self._hemisphere_mode == "lh":
            return [{"name": "lh", "eligible_mask": np.asarray((codes == 0) | (codes == 2), dtype=bool)}]
        if self._hemisphere_mode == "rh":
            return [{"name": "rh", "eligible_mask": np.asarray((codes == 1) | (codes == 2), dtype=bool)}]
        if has_lh and has_rh:
            return [
                {"name": "lh", "eligible_mask": np.asarray((codes == 0) | (codes == 2), dtype=bool)},
                {"name": "rh", "eligible_mask": np.asarray((codes == 1) | (codes == 2), dtype=bool)},
            ]
        if has_lh:
            return [{"name": "lh", "eligible_mask": np.asarray((codes == 0) | (codes == 2), dtype=bool)}]
        if has_rh:
            return [{"name": "rh", "eligible_mask": np.asarray((codes == 1) | (codes == 2), dtype=bool)}]
        return [{"name": "all", "eligible_mask": np.ones(codes.shape, dtype=bool)}]

    def _subc_target_names(self, group_name):
        text = str(group_name or "all").strip().lower()
        if text == "lh":
            return ("thal-lh-ventrolateral",)
        if text == "rh":
            return ("thal-rh-ventrolateral",)
        return ("thal-lh-ventrolateral", "thal-rh-ventrolateral")

    @staticmethod
    def _normalize_label_token(text):
        normalized = str(text or "").strip().lower()
        for old, new in (
            ("_", "-"),
            (" ", "-"),
            (".", "-"),
            ("/", "-"),
        ):
            normalized = normalized.replace(old, new)
        while "--" in normalized:
            normalized = normalized.replace("--", "-")
        return normalized

    def _populate_path_order_combos(self):
        combo_specs = (
            (self.path_order_first_combo, self._path_channel_order[0] if len(self._path_channel_order) >= 1 else "R"),
            (self.path_order_second_combo, self._path_channel_order[1] if len(self._path_channel_order) >= 2 else "G"),
            (self.path_order_third_combo, self._path_channel_order[2] if len(self._path_channel_order) >= 3 else "B"),
        )
        for combo, current_value in combo_specs:
            combo.blockSignals(True)
            combo.clear()
            for label in ("R", "G", "B"):
                combo.addItem(label, label)
            index = combo.findData(current_value)
            combo.setCurrentIndex(index if index >= 0 else 0)
            combo.blockSignals(False)

    def _group_display_name(self, group_name):
        name = str(group_name or "all").strip().lower()
        if name == "lh":
            return "LH"
        if name == "rh":
            return "RH"
        return "All"

    def _path_selection_combo_for_group(self, group_name):
        name = str(group_name or "").strip().lower()
        if name == "lh":
            return getattr(self, "left_path_combo", None)
        if name == "rh":
            return getattr(self, "right_path_combo", None)
        return None

    def _group_payload_for_name(self, group_name, payload=None):
        payload = self._project_paths_payload if payload is None else payload
        if not isinstance(payload, dict):
            return None
        target = str(group_name or "all").strip().lower()
        for group_payload in list(payload.get("group_paths", [])):
            group_dict = dict(group_payload or {})
            if str(group_dict.get("group", "all")).strip().lower() == target:
                return group_payload
        return None

    def _current_endpoint_anchor_signature(self):
        if not self._use_triangular_rgb or self._is_3d:
            return {}
        signature = {}
        for group_spec in self._path_group_specs():
            group_name = str(group_spec.get("name", "all")).strip().lower()
            anchors = {}
            for channel, node_index in dict(self._effective_anchor_indices_for_group(group_name)).items():
                channel_text = str(channel or "").strip().upper()
                if channel_text not in {"R", "G", "B"}:
                    continue
                try:
                    anchors[channel_text] = int(node_index)
                except Exception:
                    continue
            subc_index = self._find_subc_anchor_index(self._candidate_indices_for_group(group_name), group_name)
            if subc_index is not None:
                anchors["SUBC"] = int(subc_index)
            if anchors:
                signature[group_name] = anchors
        return signature

    @staticmethod
    def _payload_endpoint_anchor_signature(payload):
        if not isinstance(payload, dict):
            return {}
        signature = {}
        for group_payload in list(payload.get("group_paths", [])):
            group = dict(group_payload or {})
            group_name = str(group.get("group", "all")).strip().lower()
            anchors = {}
            for channel, node_index in dict(group.get("anchors", {}) or {}).items():
                channel_text = str(channel or "").strip().upper()
                if channel_text not in {"R", "G", "B"}:
                    continue
                try:
                    anchors[channel_text] = int(node_index)
                except Exception:
                    continue
            if group.get("subc_anchor") is not None:
                try:
                    anchors["SUBC"] = int(group.get("subc_anchor"))
                except Exception:
                    pass
            if anchors:
                signature[group_name] = anchors
        return signature

    def _generated_paths_match_current_endpoints(self):
        if not isinstance(self._project_paths_payload, dict):
            return False
        payload_order = self._coerce_path_channel_order(
            self._project_paths_payload.get("channel_order", ""),
            fallback=self._triangular_color_order,
        )
        current_order = self._coerce_path_channel_order(
            self._path_channel_order,
            fallback=self._triangular_color_order,
        )
        if payload_order != current_order:
            return False
        return (
            self._payload_endpoint_anchor_signature(self._project_paths_payload)
            == self._current_endpoint_anchor_signature()
        )

    @staticmethod
    def _path_energy_list(group_payload, key):
        values = np.asarray(dict(group_payload or {}).get(key, []), dtype=float).reshape(-1)
        return [float(value) if np.isfinite(value) else float("nan") for value in values.tolist()]

    def _selected_ctx_path_nodes(self, group_payload):
        group = dict(group_payload or {})
        selected = [int(node) for node in list(group.get("selected_ctx_path", []))]
        if len(selected) >= 2:
            return selected
        paths = [
            [int(node) for node in list(path_nodes or [])]
            for path_nodes in list(group.get("all_full_paths", []))
        ]
        try:
            index = int(group.get("selected_ctx_path_index", 0))
        except Exception:
            index = 0
        if 0 <= index < len(paths) and len(paths[index]) >= 2:
            return paths[index]
        fallback = [int(node) for node in list(group.get("optimal_full_path", []))]
        return fallback if len(fallback) >= 2 else []

    def _selected_ctx_path_energy(self, group_payload):
        group = dict(group_payload or {})
        try:
            value = float(group.get("selected_ctx_path_energy"))
            if np.isfinite(value):
                return value
        except Exception:
            pass
        try:
            index = int(group.get("selected_ctx_path_index", 0))
        except Exception:
            index = 0
        energies = self._path_energy_list(group, "ctx_path_energies")
        if 0 <= index < len(energies) and np.isfinite(energies[index]):
            return float(energies[index])
        try:
            value = float(group.get("ctx_optimal_path_energy"))
            if np.isfinite(value):
                return value
        except Exception:
            pass
        return None

    def _apply_selected_ctx_path_to_group(self, group_payload, selected_index=None):
        if not isinstance(group_payload, dict):
            return
        group_name = str(group_payload.get("group", "all")).strip().lower()
        paths = [
            [int(node) for node in list(path_nodes or [])]
            for path_nodes in list(group_payload.get("all_full_paths", []))
        ]
        if not paths:
            group_payload["selected_ctx_path_index"] = 0
            group_payload["selected_ctx_path"] = []
            group_payload["selected_ctx_path_energy"] = None
            return
        if selected_index is None:
            selected_index = self._selected_ctx_path_indices.get(group_name, 0)
        try:
            selected_index = int(selected_index)
        except Exception:
            selected_index = 0
        selected_index = int(np.clip(selected_index, 0, len(paths) - 1))
        self._selected_ctx_path_indices[group_name] = selected_index
        energies = self._path_energy_list(group_payload, "ctx_path_energies")
        selected_energy = energies[selected_index] if selected_index < len(energies) else float("nan")
        group_payload["selected_ctx_path_index"] = selected_index
        group_payload["selected_ctx_path"] = list(paths[selected_index])
        group_payload["selected_ctx_path_energy"] = (
            float(selected_energy) if np.isfinite(selected_energy) else None
        )

    def _path_selection_label(self, rank, path_nodes, energy):
        nodes = [int(node) for node in list(path_nodes or [])]
        prefix = f"{int(rank)}"
        if int(rank) == 0:
            prefix += " best"
        if energy is not None and np.isfinite(float(energy)):
            energy_text = f"E={float(energy):.4f}"
        else:
            energy_text = "E=n/a"
        return f"{prefix} | {energy_text} | nodes={len(nodes)}"

    def _populate_path_selection_combos(self):
        for group_name in ("lh", "rh"):
            combo = self._path_selection_combo_for_group(group_name)
            if combo is None:
                continue
            combo.blockSignals(True)
            combo.clear()
            group_payload = self._group_payload_for_name(group_name)
            paths = []
            energies = []
            selected_index = 0
            if isinstance(group_payload, dict):
                paths = [
                    [int(node) for node in list(path_nodes or [])]
                    for path_nodes in list(group_payload.get("all_full_paths", []))
                ]
                energies = self._path_energy_list(group_payload, "ctx_path_energies")
                self._apply_selected_ctx_path_to_group(group_payload)
                selected_index = int(group_payload.get("selected_ctx_path_index", 0))
            if not paths:
                combo.addItem(f"No {self._group_display_name(group_name)} paths", None)
                combo.setCurrentIndex(0)
                combo.blockSignals(False)
                continue
            for rank, path_nodes in enumerate(paths):
                energy = energies[rank] if rank < len(energies) else None
                combo.addItem(self._path_selection_label(rank, path_nodes, energy), int(rank))
            combo.setCurrentIndex(int(np.clip(selected_index, 0, combo.count() - 1)))
            combo.blockSignals(False)
        self._sync_path_selection_controls_enabled()

    def _sync_path_selection_controls_enabled(self):
        if not hasattr(self, "left_path_combo") or not hasattr(self, "right_path_combo"):
            return
        for group_name in ("lh", "rh"):
            combo = self._path_selection_combo_for_group(group_name)
            group_payload = self._group_payload_for_name(group_name)
            enabled = (
                combo is not None
                and self._use_triangular_rgb
                and isinstance(group_payload, dict)
                and len(list(group_payload.get("all_full_paths", []))) > 0
            )
            if combo is not None:
                combo.setEnabled(bool(enabled))

    def _on_ctx_path_selection_changed(self, group_name):
        combo = self._path_selection_combo_for_group(group_name)
        if combo is None or not isinstance(self._project_paths_payload, dict):
            return
        selected_index = combo.currentData()
        if selected_index is None:
            return
        group_payload = self._group_payload_for_name(group_name)
        if not isinstance(group_payload, dict):
            return
        self._apply_selected_ctx_path_to_group(group_payload, selected_index)
        if self._show_all_ordered_paths:
            self._show_all_ordered_paths = False
            if hasattr(self, "all_paths_check"):
                self.all_paths_check.blockSignals(True)
                self.all_paths_check.setChecked(False)
                self.all_paths_check.blockSignals(False)
            self._project_paths_payload["show_all_ordered_paths"] = False
        self._sync_proximity_controls()
        self._render(preserve_view=True)

    def _display_group_specs(self):
        path_groups = list(self._path_group_specs())
        if self._hemisphere_mode == "separate" and len(path_groups) > 1:
            specs = []
            for group_spec in path_groups:
                name = str(group_spec.get("name", "all")).strip().lower()
                eligible_mask = np.asarray(group_spec.get("eligible_mask"), dtype=bool).reshape(-1)
                if eligible_mask.shape[0] != self._x.shape[0] or not np.any(eligible_mask):
                    continue
                specs.append(
                    {
                        "name": name,
                        "title": self._group_display_name(name),
                        "indices": np.flatnonzero(eligible_mask),
                    }
                )
            return specs or [{"name": "all", "title": self._group_display_name("all"), "indices": np.arange(self._x.shape[0], dtype=int)}]
        return [{"name": "all", "title": "", "indices": np.arange(self._x.shape[0], dtype=int)}]

    def _candidate_indices_for_group(self, group_name):
        target = str(group_name or "all").strip().lower()
        for group_spec in self._path_group_specs():
            if str(group_spec.get("name", "all")).strip().lower() != target:
                continue
            eligible_mask = np.asarray(group_spec.get("eligible_mask"), dtype=bool).reshape(-1)
            if eligible_mask.shape[0] != self._x.shape[0]:
                return np.arange(self._x.shape[0], dtype=int)
            return np.flatnonzero(eligible_mask)
        return np.arange(self._x.shape[0], dtype=int)

    def _anchor_option_label(self, node_index):
        idx = int(node_index)
        if idx < 0 or idx >= self._point_labels.shape[0]:
            return "Unknown"
        return f"{str(self._point_labels[idx])} [{str(self._point_ids[idx])}]"

    def _average_gradient_pair_for_anchor_defaults(self):
        gradients_avg = np.asarray(
            dict(self._export_metadata or {}).get("gradients_avg", np.empty((0, 0), dtype=float)),
            dtype=float,
        )
        if gradients_avg.ndim != 2:
            return None
        n_points = int(self._point_ids.shape[0])
        if gradients_avg.shape[1] == n_points:
            canonical = np.asarray(gradients_avg, dtype=float)
        elif gradients_avg.shape[0] == n_points:
            canonical = np.asarray(gradients_avg.T, dtype=float)
        else:
            return None
        if canonical.shape[0] < 2:
            return None
        return np.asarray(canonical[:2, :].T, dtype=float)

    def _derive_default_fixed_anchor_indices(self):
        gradient_pair = self._average_gradient_pair_for_anchor_defaults()
        if gradient_pair is None or gradient_pair.ndim != 2 or gradient_pair.shape[1] < 2:
            return {}
        gradient1 = np.asarray(gradient_pair[:, 0], dtype=float)
        gradient2 = np.asarray(gradient_pair[:, 1], dtype=float)
        scatter_coords = np.column_stack((gradient2, gradient1))
        finite_mask = np.all(np.isfinite(scatter_coords), axis=1)
        if not np.any(finite_mask):
            return {}
        fixed = {}
        for group_spec in self._path_group_specs():
            group_name = str(group_spec.get("name", "all")).strip().lower()
            eligible_mask = np.asarray(group_spec.get("eligible_mask"), dtype=bool).reshape(-1)
            if eligible_mask.shape[0] != self._x.shape[0]:
                continue
            candidate_indices = np.flatnonzero(eligible_mask & finite_mask)
            if candidate_indices.size < 3:
                continue
            triangle_model = self._rgb_model(
                scatter_coords[candidate_indices, 0],
                scatter_coords[candidate_indices, 1],
                self._triangular_color_order,
                fit_mode=self._rgb_fit_mode,
            )
            anchors = self._rgb_anchor_indices(
                scatter_coords[:, 0],
                scatter_coords[:, 1],
                triangle_model,
                candidate_indices=candidate_indices,
            )
            if {"R", "G", "B"}.issubset(set(anchors.keys())):
                fixed[group_name] = {str(channel): int(index) for channel, index in anchors.items()}
        return fixed

    def _auto_anchor_indices_for_group(self, group_name):
        candidate_indices = self._candidate_indices_for_group(group_name)
        if candidate_indices.size < 3 or self._display_coords.shape[0] < 3:
            return {}
        rgb_x_plot, rgb_y_plot = self._rotated_rgb_points()
        rgb_coords = np.column_stack((np.asarray(rgb_x_plot, dtype=float), np.asarray(rgb_y_plot, dtype=float)))
        coords = np.asarray(rgb_coords[candidate_indices, :], dtype=float)
        finite_mask = np.all(np.isfinite(coords), axis=1)
        if np.count_nonzero(finite_mask) < 3:
            return {}
        triangle_model = self._rgb_model(
            coords[finite_mask, 0],
            coords[finite_mask, 1],
            self._triangular_color_order,
            fit_mode=self._rgb_fit_mode,
        )
        anchors = self._rgb_anchor_indices(
            rgb_coords[:, 0],
            rgb_coords[:, 1],
            triangle_model,
            candidate_indices=candidate_indices,
        )
        return {str(channel): int(index) for channel, index in anchors.items()}

    def _effective_anchor_indices_for_group(self, group_name):
        name = str(group_name or "all").strip().lower()
        auto_base = dict(self._auto_anchor_indices_for_group(name))
        avg_base = dict(self._default_fixed_anchor_indices.get(name, {}))
        mode = str(self._endpoint_selection_mode or "adaptive").strip().lower()
        if mode == "average":
            base = dict(avg_base if {"R", "G", "B"}.issubset(set(avg_base.keys())) else auto_base)
        elif mode == "manual":
            base = dict(auto_base if {"R", "G", "B"}.issubset(set(auto_base.keys())) else avg_base)
        else:
            base = dict(auto_base)
        overrides = dict(self._manual_anchor_overrides.get(name, {}))
        candidate_indices = np.asarray(self._candidate_indices_for_group(name), dtype=int).reshape(-1)
        candidate_set = {int(value) for value in candidate_indices.tolist()}

        for channel, value in list(overrides.items()):
            if value is None:
                continue
            try:
                node_index = int(value)
            except Exception:
                continue
            if candidate_set and node_index not in candidate_set:
                continue
            base[str(channel)] = node_index

        manual_channels = {
            str(channel).strip().upper()
            for channel, value in list(overrides.items())
            if value is not None
        }
        resolved = {}
        used = set()

        for channel in ("R", "G", "B"):
            if channel not in manual_channels:
                continue
            try:
                idx = int(base.get(channel))
            except Exception:
                continue
            if candidate_set and idx not in candidate_set:
                continue
            if idx in used:
                continue
            resolved[channel] = idx
            used.add(idx)

        for channel in ("R", "G", "B"):
            if channel in resolved:
                continue
            candidate_pool = []
            for source in (base, auto_base, avg_base):
                value = source.get(channel)
                try:
                    idx = int(value)
                except Exception:
                    continue
                candidate_pool.append(idx)
            candidate_pool.extend(candidate_indices.tolist())
            chosen = None
            for idx in candidate_pool:
                idx = int(idx)
                if candidate_set and idx not in candidate_set:
                    continue
                if idx in used:
                    continue
                chosen = idx
                break
            if chosen is not None:
                resolved[channel] = int(chosen)
                used.add(int(chosen))

        return {str(channel): int(index) for channel, index in resolved.items()}

    def _average_endpoints_available(self):
        for anchors in list(dict(self._default_fixed_anchor_indices or {}).values()):
            if {"R", "G", "B"}.issubset(set(dict(anchors or {}).keys())):
                return True
        return False

    def _endpoint_mode_display_text(self):
        mode = str(self._endpoint_selection_mode or "adaptive").strip().lower()
        if mode == "manual":
            if str(getattr(self, "_loaded_fixed_endpoint_file", "") or "").strip():
                return "manual file"
            return "manual"
        if mode == "average":
            return "gradients_avg" if self._average_endpoints_available() else "adaptive (no avg)"
        return "adaptive"

    def _manual_endpoint_target_specs(self):
        specs = []
        for group_spec in self._path_group_specs():
            group_name = str(group_spec.get("name", "all")).strip().lower()
            for channel in ("R", "G", "B", "SUBC"):
                if group_name == "all":
                    label = channel
                else:
                    label = f"{self._group_display_name(group_name)} {channel}"
                specs.append((label, f"{group_name}:{channel}"))
        return specs

    def _populate_manual_endpoint_targets(self):
        current = self._manual_endpoint_target
        self.manual_endpoint_target_combo.blockSignals(True)
        self.manual_endpoint_target_combo.clear()
        for label, value in self._manual_endpoint_target_specs():
            self.manual_endpoint_target_combo.addItem(label, value)
        selected_index = -1
        if current is not None:
            selected_index = self.manual_endpoint_target_combo.findData(current)
        if selected_index < 0 and self.manual_endpoint_target_combo.count() > 0:
            selected_index = 0
        if selected_index >= 0:
            self.manual_endpoint_target_combo.setCurrentIndex(selected_index)
            self._manual_endpoint_target = self.manual_endpoint_target_combo.currentData()
        else:
            self._manual_endpoint_target = None
        self.manual_endpoint_target_combo.blockSignals(False)

    def _current_manual_endpoint_target(self):
        value = self.manual_endpoint_target_combo.currentData()
        if value is None:
            return None, None
        text = str(value).strip()
        if ":" not in text:
            return None, None
        group_name, channel = text.split(":", 1)
        channel_text = str(channel or "").strip().upper()
        channel = "SUBC" if channel_text in {"SUBC", "S", "THAL", "THALAMUS"} else self._normalize_path_channel(channel)
        return str(group_name or "all").strip().lower(), channel

    def _assign_manual_anchor(self, group_name, channel, node_index):
        group_key = str(group_name or "all").strip().lower()
        channel_text = str(channel or "").strip().upper()
        try:
            node_value = int(node_index)
        except Exception:
            return
        if channel_text in {"SUBC", "S", "THAL", "THALAMUS"}:
            self._manual_subc_anchor_overrides[group_key] = node_value
            self._loaded_fixed_endpoint_file = ""
            self._loaded_fixed_endpoint_source = ""
            return
        channel_key = self._normalize_path_channel(channel)
        overrides = dict(self._manual_anchor_overrides.get(group_key, {}))
        overrides[channel_key] = node_value
        for other_channel, other_value in list(overrides.items()):
            if other_channel != channel_key and int(other_value) == node_value:
                overrides.pop(other_channel, None)
        self._manual_anchor_overrides[group_key] = overrides
        self._loaded_fixed_endpoint_file = ""
        self._loaded_fixed_endpoint_source = ""

    def _advance_manual_endpoint_target(self):
        count = int(self.manual_endpoint_target_combo.count())
        if count <= 1:
            return
        current_index = int(self.manual_endpoint_target_combo.currentIndex())
        self.manual_endpoint_target_combo.setCurrentIndex((current_index + 1) % count)

    def _on_endpoint_mode_changed(self, _index):
        value = self.endpoint_mode_combo.currentData()
        mode = str(value or "adaptive").strip().lower()
        if mode not in {"adaptive", "average", "manual"}:
            mode = "adaptive"
        self._endpoint_selection_mode = mode
        self._invalidate_generated_paths()
        self._sync_proximity_controls()
        self._render()

    def _on_manual_endpoint_target_changed(self, _index):
        self._manual_endpoint_target = self.manual_endpoint_target_combo.currentData()
        self._render()

    def _on_clear_manual_endpoints_clicked(self):
        self._manual_anchor_overrides = {}
        self._manual_subc_anchor_overrides = {}
        self._loaded_fixed_endpoint_file = ""
        self._loaded_fixed_endpoint_source = ""
        self._invalidate_generated_paths()
        self._sync_proximity_controls()
        self._render()

    def _default_free_energy_paths_dir(self):
        metadata = dict(getattr(self, "_export_metadata", {}) or {})
        for key in ("free_energy_paths_dir", "results_dir"):
            value = str(metadata.get(key, "") or "").strip()
            if value:
                path = Path(value).expanduser()
                if path.is_dir():
                    return path
        if FREE_ENERGY_PATHS_DEFAULT_DIR.is_dir():
            return FREE_ENERGY_PATHS_DEFAULT_DIR
        return Path.cwd()

    @staticmethod
    def _normalize_matching_entity(value, prefix):
        text = str(value or "").strip().lower()
        text = re.sub(rf"^{re.escape(str(prefix).lower())}[-_]?", "", text)
        return re.sub(r"[^a-z0-9]+", "", text)

    @classmethod
    def _bids_entities_from_text(cls, value):
        text = str(value or "")
        subject_match = re.search(r"(?:^|[/_\s])sub-([^/_\s]+)", text, flags=re.IGNORECASE)
        session_match = re.search(r"(?:^|[/_\s])ses-([^/_\s]+)", text, flags=re.IGNORECASE)
        subject = (
            cls._normalize_matching_entity(subject_match.group(1), "sub")
            if subject_match is not None
            else ""
        )
        session = (
            cls._normalize_matching_entity(session_match.group(1), "ses")
            if session_match is not None
            else ""
        )
        return subject, session

    @staticmethod
    def _canonical_parcellation_token(value):
        text = str(value or "").strip()
        if not text:
            return ""
        name = Path(text).name.lower()
        for suffix in (".nii.gz", ".nii", ".npz", ".npy"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        name = re.sub(r"scale", "", name, flags=re.IGNORECASE)
        name = re.sub(r"(?:atlas|parcellation)", "", name, flags=re.IGNORECASE)
        return re.sub(r"[^a-z0-9]+", "", name)

    @classmethod
    def _parcellation_match_tokens(cls, *values):
        tokens = set()
        for value in values:
            text = str(value or "").strip()
            if not text:
                continue
            for candidate in (text, Path(text).name, Path(text).parent.name):
                token = cls._canonical_parcellation_token(candidate)
                if token:
                    tokens.add(token)
        return tokens

    @classmethod
    def _npz_first_scalar_text(cls, npz, *keys):
        for key in keys:
            if key in npz.files:
                return cls._npz_scalar_text(npz[key]).strip()
        return ""

    @staticmethod
    def _set_overlap_fraction(current_values, candidate_values):
        current = set(current_values or ())
        candidate = set(candidate_values or ())
        if not current or not candidate:
            return 0.0
        return float(len(current.intersection(candidate)) / len(current))

    def _saved_path_match_context(self):
        metadata = dict(getattr(self, "_export_metadata", {}) or {})
        subject = self._normalize_matching_entity(metadata.get("subject_id", ""), "sub")
        session = self._normalize_matching_entity(metadata.get("session_id", ""), "ses")
        if not subject or not session:
            for value in (metadata.get("source_path", ""), metadata.get("source_name", "")):
                parsed_subject, parsed_session = self._bids_entities_from_text(value)
                subject = subject or parsed_subject
                session = session or parsed_session

        parc_path = str(
            metadata.get("parc_path", metadata.get("template_path", "")) or ""
        ).strip()
        parcellation_tokens = self._parcellation_match_tokens(parc_path)
        current_labels = {
            self._normalize_label_token(value)
            for value in np.asarray(self._point_ids, dtype=object).reshape(-1).tolist()
            if str(value).strip()
        }
        current_names = {
            self._normalize_label_token(value)
            for value in np.asarray(self._point_labels, dtype=object).reshape(-1).tolist()
            if str(value).strip()
        }
        return {
            "subject": subject,
            "session": session,
            "parcellation_tokens": parcellation_tokens,
            "labels": current_labels,
            "names": current_names,
        }

    def _find_best_matching_free_energy_path(self):
        context = self._saved_path_match_context()
        subject = str(context.get("subject", ""))
        session = str(context.get("session", ""))
        if not subject or not session:
            return None, "current gradient selection has no complete subject/session metadata"

        search_root = Path(
            getattr(self, "_free_energy_paths_load_dir", None)
            or self._default_free_energy_paths_dir()
        ).expanduser()
        if not search_root.is_dir():
            return None, f"saved-path folder does not exist: {search_root}"
        try:
            candidates = sorted(search_root.rglob(f"*{FREE_ENERGY_PATHS_SUFFIX}"))
        except Exception as exc:
            return None, f"could not search {search_root}: {exc}"

        ranked = []
        current_tokens = set(context.get("parcellation_tokens", set()) or set())
        current_labels = set(context.get("labels", set()) or set())
        current_names = set(context.get("names", set()) or set())
        for candidate_path in candidates:
            filename_subject, filename_session = self._bids_entities_from_text(candidate_path.name)
            if filename_subject and filename_subject != subject:
                continue
            if filename_session and filename_session != session:
                continue
            try:
                with np.load(str(candidate_path), allow_pickle=True) as npz:
                    candidate_subject = self._normalize_matching_entity(
                        self._npz_first_scalar_text(npz, "subject_id", "participant_id"),
                        "sub",
                    ) or filename_subject
                    candidate_session = self._normalize_matching_entity(
                        self._npz_first_scalar_text(npz, "session_id", "session"),
                        "ses",
                    ) or filename_session
                    if candidate_subject != subject or candidate_session != session:
                        continue
                    candidate_scheme = self._npz_first_scalar_text(
                        npz,
                        "parc_scheme",
                        "parcellation_scheme",
                    )
                    candidate_parc_path = self._npz_first_scalar_text(
                        npz,
                        "parc_path",
                        "template_path",
                    )
                    candidate_tokens = self._parcellation_match_tokens(
                        candidate_scheme,
                        candidate_parc_path,
                        candidate_path.parent.name,
                    )
                    candidate_labels = {
                        self._normalize_label_token(value)
                        for value in np.asarray(
                            npz["parcel_labels"] if "parcel_labels" in npz.files else [],
                            dtype=object,
                        ).reshape(-1).tolist()
                        if str(value).strip()
                    }
                    candidate_names = {
                        self._normalize_label_token(value)
                        for value in np.asarray(
                            npz["parcel_names"] if "parcel_names" in npz.files else [],
                            dtype=object,
                        ).reshape(-1).tolist()
                        if str(value).strip()
                    }
            except Exception:
                continue

            token_match = bool(current_tokens.intersection(candidate_tokens))
            name_overlap = self._set_overlap_fraction(current_names, candidate_names)
            label_overlap = self._set_overlap_fraction(current_labels, candidate_labels)
            if current_tokens:
                if not token_match:
                    continue
            elif name_overlap < 0.75 and label_overlap < 0.98:
                continue
            try:
                modified_time = float(candidate_path.stat().st_mtime)
            except Exception:
                modified_time = 0.0
            rank = (
                int(token_match),
                float(name_overlap),
                float(label_overlap),
                modified_time,
                str(candidate_path),
            )
            ranked.append((rank, candidate_path))

        if not ranked:
            return None, (
                f"no saved path matched sub-{subject}, ses-{session}, "
                "and the current parcellation"
            )
        ranked.sort(key=lambda item: item[0], reverse=True)
        best_path = Path(ranked[0][1])
        return best_path, f"selected the best of {len(ranked)} matching saved path files"

    def _auto_preload_matching_free_energy_paths(self):
        if self._is_3d or not self._use_triangular_rgb:
            self.status_label.setText(
                "Automatic saved-path preload applies only to a 2D triangular-RGB scatter."
            )
            return
        input_path, match_note = self._find_best_matching_free_energy_path()
        if input_path is None:
            self.status_label.setText(f"Automatic saved-path preload skipped: {match_note}.")
            return
        try:
            matched_count, unresolved, generated_paths = self._apply_free_energy_endpoints_path(
                input_path,
                source="auto_match",
            )
        except Exception as exc:
            warn(f"Failed to auto-load free-energy paths from `{input_path}`: {exc}")
            self.status_label.setText(f"Automatic saved-path preload failed: {exc}")
            return
        unresolved_text = f"; unresolved {len(unresolved)}" if unresolved else ""
        path_text = "generated paths" if generated_paths else "endpoints only; adjacency unavailable"
        self.status_label.setText(
            f"Auto-loaded {matched_count} endpoint anchors and {path_text} from "
            f"{input_path.name}{unresolved_text}; {match_note}."
        )

    def _load_saved_adjacency_edges(self, free_energy_path):
        with np.load(str(free_energy_path), allow_pickle=True) as npz:
            adjacency_text = self._npz_first_scalar_text(npz, "adjacency_path")
        if not adjacency_text:
            return None
        adjacency_path = Path(adjacency_text).expanduser()
        if not adjacency_path.is_file():
            return None

        with np.load(str(adjacency_path), allow_pickle=True) as npz:
            if "adjacency_mat" not in npz.files:
                return None
            adjacency = np.asarray(npz["adjacency_mat"], dtype=float)
            adjacency_labels = np.asarray(
                npz["parcel_labels"] if "parcel_labels" in npz.files else [],
                dtype=object,
            ).reshape(-1)
            adjacency_names = np.asarray(
                npz["parcel_names"] if "parcel_names" in npz.files else [],
                dtype=object,
            ).reshape(-1)
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            return None

        def _mapped_positions(current_values, saved_values):
            saved = np.asarray(saved_values, dtype=object).reshape(-1)
            if saved.shape[0] != adjacency.shape[0]:
                return [], []
            lookup = {}
            for saved_index, value in enumerate(saved.tolist()):
                token = self._normalize_label_token(value)
                if token and token not in lookup:
                    lookup[token] = int(saved_index)
            positions = []
            indices = []
            for point_index, value in enumerate(
                np.asarray(current_values, dtype=object).reshape(-1).tolist()
            ):
                saved_index = lookup.get(self._normalize_label_token(value))
                if saved_index is None:
                    continue
                positions.append(int(point_index))
                indices.append(int(saved_index))
            return positions, indices

        mapped_positions, mapped_indices = _mapped_positions(
            self._point_ids,
            adjacency_labels,
        )
        name_positions, name_indices = _mapped_positions(
            self._point_labels,
            adjacency_names,
        )
        if len(name_indices) > len(mapped_indices):
            mapped_positions, mapped_indices = name_positions, name_indices
        if len(mapped_indices) < 2:
            return None

        mapped_positions = np.asarray(mapped_positions, dtype=int)
        mapped_indices = np.asarray(mapped_indices, dtype=int)
        edge_matrix = np.asarray(adjacency[np.ix_(mapped_indices, mapped_indices)], dtype=float)
        edge_matrix = np.nan_to_num(edge_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        edge_matrix = np.maximum(np.abs(edge_matrix), np.abs(edge_matrix.T))
        upper_i, upper_j = np.triu_indices(edge_matrix.shape[0], k=1)
        keep = edge_matrix[upper_i, upper_j] > 0.0
        if not np.any(keep):
            return None
        self._edge_pairs = self._normalize_edge_pairs(
            np.column_stack(
                (
                    mapped_positions[upper_i[keep]],
                    mapped_positions[upper_j[keep]],
                )
            ),
            self._x.size,
        )
        self._edge_distances = self._compute_edge_distances(
            self._display_coords,
            self._edge_pairs,
        )
        self._path_metric_edge_distances = self._compute_edge_distances(
            self._path_metric_coords
            if self._path_metric_coords is not None
            else self._display_coords,
            self._edge_pairs,
        )
        self._export_metadata["adjacency_path"] = str(adjacency_path)
        return adjacency_path

    @staticmethod
    def _npz_scalar_text(value):
        try:
            array = np.asarray(value, dtype=object)
            if array.shape == ():
                return str(array.item())
            if array.size == 1:
                return str(array.reshape(-1)[0])
        except Exception:
            pass
        return str(value)

    @staticmethod
    def _npz_python_object(value):
        try:
            array = np.asarray(value, dtype=object)
            if array.shape == ():
                return array.item()
            if array.size == 1:
                item = array.reshape(-1)[0]
                if isinstance(item, (dict, list, tuple)):
                    return item
            return array.tolist()
        except Exception:
            return value

    @staticmethod
    def _normalize_loaded_endpoint_group(value):
        text = str(value or "all").strip().lower()
        mapping = {
            "left": "lh",
            "l": "lh",
            "lh": "lh",
            "right": "rh",
            "r": "rh",
            "rh": "rh",
            "both": "all",
            "all": "all",
        }
        return mapping.get(text, text or "all")

    @staticmethod
    def _endpoint_record_text(record, *keys):
        if not isinstance(record, dict):
            return ""
        for key in keys:
            value = record.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    def _endpoint_record_matches_index(self, record, node_index):
        if not isinstance(record, dict):
            return False
        idx = int(node_index)
        if idx < 0 or idx >= self._point_ids.shape[0]:
            return False
        saved_label = self._endpoint_record_text(record, "node_label", "parcel_label", "label", "id")
        saved_name = self._endpoint_record_text(record, "node_name", "parcel_name", "name")
        if not saved_label and not saved_name:
            return True
        current_label = str(self._point_ids[idx]).strip()
        current_name = str(self._point_labels[idx]).strip()
        if saved_label and (
            saved_label == current_label
            or self._normalize_label_token(saved_label) == self._normalize_label_token(current_label)
        ):
            return True
        if saved_name and (
            saved_name == current_name
            or self._normalize_label_token(saved_name) == self._normalize_label_token(current_name)
        ):
            return True
        return False

    def _match_loaded_endpoint_node(self, record, group_name):
        if not isinstance(record, dict):
            return None
        candidates = np.asarray(self._candidate_indices_for_group(group_name), dtype=int).reshape(-1)
        candidate_set = {int(value) for value in candidates.tolist()}
        for key in ("node_index", "scatter_index", "index"):
            if key not in record:
                continue
            try:
                idx = int(record.get(key))
            except Exception:
                continue
            if idx in candidate_set and self._endpoint_record_matches_index(record, idx):
                return int(idx)
        for idx in candidates.tolist():
            if self._endpoint_record_matches_index(record, int(idx)):
                return int(idx)
        return None

    def _endpoint_groups_from_free_energy_npz(self, npz):
        endpoint_groups = {}
        global_path_order = ""

        def _merge_group(group_name, group_info):
            if not isinstance(group_info, dict):
                return
            group_key = self._normalize_loaded_endpoint_group(group_name)
            anchors = dict(group_info.get("anchors", {}) or {})
            subc_record = None
            for subc_key in ("subc_anchor", "subc_endpoint"):
                candidate = group_info.get(subc_key)
                if isinstance(candidate, dict):
                    subc_record = dict(candidate)
                    break
                if candidate is not None:
                    try:
                        candidate_array = np.asarray(candidate).reshape(-1)
                        if candidate_array.size == 1:
                            subc_record = {"node_index": int(candidate_array[0])}
                            break
                    except Exception:
                        pass
            if not anchors and subc_record is None:
                return
            existing = dict(endpoint_groups.get(group_key, {}))
            existing["anchors"] = dict(existing.get("anchors", {}))
            for channel, record in anchors.items():
                channel_text = str(channel or "").strip().upper()
                if channel_text not in {"R", "G", "B"} or not isinstance(record, dict):
                    continue
                existing["anchors"][channel_text] = dict(record)
            if subc_record is not None:
                existing["subc_anchor"] = dict(subc_record)
            path_order = str(group_info.get("path_order", group_info.get("channel_order", "")) or "").strip()
            if path_order:
                existing["path_order"] = path_order
            endpoint_groups[group_key] = existing

        if "path_order_override" in npz.files:
            global_path_order = self._npz_scalar_text(npz["path_order_override"]).strip()

        fixed_payload = None
        if "fixed_endpoints_json" in npz.files:
            try:
                fixed_payload = json.loads(self._npz_scalar_text(npz["fixed_endpoints_json"]))
            except Exception:
                fixed_payload = None
        if isinstance(fixed_payload, dict):
            for group_name, group_info in fixed_payload.items():
                if isinstance(group_info, dict):
                    _merge_group(group_name, group_info)

        summary_payload = None
        if "summary_json" in npz.files:
            try:
                summary_payload = json.loads(self._npz_scalar_text(npz["summary_json"]))
            except Exception:
                summary_payload = None
        if isinstance(summary_payload, dict):
            if not global_path_order:
                global_path_order = str(summary_payload.get("path_order_override", "") or "").strip()
            fixed_from_summary = summary_payload.get("fixed_endpoints")
            if isinstance(fixed_from_summary, dict):
                for group_name, group_info in fixed_from_summary.items():
                    if isinstance(group_info, dict):
                        _merge_group(group_name, group_info)
            for group_info in list(summary_payload.get("groups", [])):
                if isinstance(group_info, dict):
                    _merge_group(group_info.get("group", "all"), group_info)

        if "groups" in npz.files:
            groups_payload = self._npz_python_object(npz["groups"])
            for group_info in list(groups_payload if isinstance(groups_payload, (list, tuple)) else [groups_payload]):
                if isinstance(group_info, dict):
                    _merge_group(group_info.get("group", "all"), group_info)

        return endpoint_groups, global_path_order

    def _load_free_energy_endpoint_overrides(self, path):
        input_path = Path(path).expanduser()
        if not input_path.is_file():
            raise ValueError(f"Endpoint file does not exist: {input_path}")
        with np.load(str(input_path), allow_pickle=True) as npz:
            endpoint_groups, global_path_order = self._endpoint_groups_from_free_energy_npz(npz)
        if not endpoint_groups:
            raise ValueError("No fixed endpoint anchors were found in this NPZ.")

        current_groups = {
            self._normalize_loaded_endpoint_group(spec.get("name", "all"))
            for spec in self._path_group_specs()
        }
        overrides = {}
        subc_overrides = {}
        unresolved = []
        matched_count = 0
        for loaded_group, group_info in endpoint_groups.items():
            group_key = loaded_group
            if group_key not in current_groups:
                if group_key == "all" and len(current_groups) == 1:
                    group_key = next(iter(current_groups))
                else:
                    unresolved.append(f"{loaded_group}: group unavailable")
                    continue
            group_overrides = dict(overrides.get(group_key, {}))
            for channel, record in dict(group_info.get("anchors", {}) or {}).items():
                channel_text = str(channel or "").strip().upper()
                if channel_text not in {"R", "G", "B"} or not isinstance(record, dict):
                    continue
                node_index = self._match_loaded_endpoint_node(record, group_key)
                if node_index is None:
                    saved_label = self._endpoint_record_text(record, "node_label", "label")
                    saved_name = self._endpoint_record_text(record, "node_name", "name")
                    unresolved.append(f"{loaded_group} {channel_text}: {saved_name or saved_label or 'unmatched'}")
                    continue
                group_overrides[channel_text] = int(node_index)
                matched_count += 1
            if group_overrides:
                overrides[group_key] = group_overrides
            subc_record = group_info.get("subc_anchor")
            if isinstance(subc_record, dict):
                node_index = self._match_loaded_endpoint_node(subc_record, group_key)
                if node_index is None:
                    saved_label = self._endpoint_record_text(subc_record, "node_label", "label")
                    saved_name = self._endpoint_record_text(subc_record, "node_name", "name")
                    unresolved.append(f"{loaded_group} SUBC: {saved_name or saved_label or 'unmatched'}")
                else:
                    subc_overrides[group_key] = int(node_index)
                    matched_count += 1

        if not overrides and not subc_overrides:
            raise ValueError("None of the saved endpoints matched the current scatter points.")

        path_order = ""
        for group_info in endpoint_groups.values():
            path_order = str(dict(group_info or {}).get("path_order", "") or "").strip()
            if path_order:
                break
        if not path_order:
            path_order = str(global_path_order or "").strip()
        return (
            overrides,
            subc_overrides,
            self._coerce_path_channel_order(path_order, fallback=self._path_channel_order),
            matched_count,
            unresolved,
        )

    def _on_load_free_energy_endpoints_clicked(self):
        start_dir = Path(getattr(self, "_free_energy_paths_load_dir", None) or self._default_free_energy_paths_dir())
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Load free-energy endpoints",
            str(start_dir),
            f"Free-energy paths (*{FREE_ENERGY_PATHS_SUFFIX});;NumPy archive (*.npz);;All files (*)",
        )
        if not path:
            return
        input_path = Path(path).expanduser()
        try:
            matched_count, unresolved, generated_paths = self._apply_free_energy_endpoints_path(
                input_path,
                source="file",
            )
        except Exception as exc:
            warn(f"Failed to load free-energy endpoints from `{input_path}`: {exc}")
            if hasattr(self, "status_label"):
                self.status_label.setText(f"Endpoint load failed: {exc}")
            return

        unresolved_text = f"; unresolved {len(unresolved)}" if unresolved else ""
        path_text = " and generated paths" if generated_paths else ""
        if hasattr(self, "status_label"):
            self.status_label.setText(
                f"Loaded {matched_count} endpoint anchors{path_text} from "
                f"{input_path.name}{unresolved_text}"
            )

    def _apply_free_energy_endpoints_path(self, input_path, *, source):
        input_path = Path(input_path).expanduser()
        self._free_energy_paths_load_dir = input_path.parent
        overrides, subc_overrides, path_order, matched_count, unresolved = (
            self._load_free_energy_endpoint_overrides(input_path)
        )
        self._manual_anchor_overrides = overrides
        self._manual_subc_anchor_overrides = subc_overrides
        self._loaded_fixed_endpoint_file = str(input_path)
        self._loaded_fixed_endpoint_source = str(source or "file")
        self._endpoint_selection_mode = "manual"
        manual_index = self.endpoint_mode_combo.findData("manual")
        if manual_index >= 0:
            self.endpoint_mode_combo.blockSignals(True)
            self.endpoint_mode_combo.setCurrentIndex(manual_index)
            self.endpoint_mode_combo.blockSignals(False)
        self._path_channel_order = path_order
        self._populate_path_order_combos()
        self._selected_ctx_path_indices = {"lh": 0, "rh": 0, "all": 0}
        self._invalidate_generated_paths()
        if self._edge_pairs.shape[0] == 0:
            try:
                self._load_saved_adjacency_edges(input_path)
            except Exception as exc:
                warn(f"Failed to load adjacency referenced by `{input_path}`: {exc}")
        if self._edge_pairs.shape[0] > 0:
            self._on_generate_paths_clicked()
        else:
            self._sync_proximity_controls()
            self._render()
        generated_paths = isinstance(self._project_paths_payload, dict)
        if unresolved:
            warn(
                "Some free-energy endpoints were not matched: "
                + "; ".join(str(value) for value in unresolved[:12])
            )
        return matched_count, unresolved, generated_paths

    def _find_subc_anchor_index(self, candidate_indices, group_name):
        indices = np.asarray(candidate_indices, dtype=int).reshape(-1)
        if indices.size == 0:
            return None
        candidate_set = {int(value) for value in indices.tolist()}
        group_key = str(group_name or "all").strip().lower()
        overrides = dict(getattr(self, "_manual_subc_anchor_overrides", {}) or {})
        for key in (group_key, "all"):
            if key not in overrides:
                continue
            try:
                node_index = int(overrides.get(key))
            except Exception:
                continue
            if node_index in candidate_set:
                return int(node_index)
        targets = tuple(self._normalize_label_token(target) for target in self._subc_target_names(group_name))
        for idx in indices.tolist():
            if idx < 0 or idx >= self._point_labels.shape[0]:
                continue
            label_text = self._normalize_label_token(self._point_labels[idx])
            if any(target in label_text for target in targets):
                return int(idx)
        return None

    @staticmethod
    def _binary_adjacency_dict(node_count, edge_pairs, forbidden_nodes=None):
        n_nodes = max(0, int(node_count))
        blocked = {int(node) for node in (forbidden_nodes or set())}
        adjacency = {
            int(node): set()
            for node in range(n_nodes)
            if int(node) not in blocked
        }
        pairs = np.asarray(edge_pairs, dtype=int)
        if pairs.ndim != 2 or pairs.shape[0] == 0:
            return {node: [] for node in sorted(adjacency)}
        for node_u, node_v in pairs:
            u = int(node_u)
            v = int(node_v)
            if u < 0 or v < 0 or u >= n_nodes or v >= n_nodes or u == v:
                continue
            if u in blocked or v in blocked:
                continue
            if u not in adjacency:
                adjacency[u] = set()
            if v not in adjacency:
                adjacency[v] = set()
            adjacency[u].add(v)
            adjacency[v].add(u)
        return {node: sorted(neighbors) for node, neighbors in sorted(adjacency.items())}

    @staticmethod
    def _binary_graph(node_count, edge_pairs, forbidden_nodes=None):
        adjacency = GradientScatterDialog._binary_adjacency_dict(
            node_count,
            edge_pairs,
            forbidden_nodes=forbidden_nodes,
        )
        return nx.from_dict_of_lists(adjacency)

    @staticmethod
    def _shortest_path(node_count, edge_pairs, edge_weights, start_index, end_index, forbidden_nodes=None):
        """Return the unweighted hop-count shortest path on the binary adjacency."""
        if node_count <= 0:
            return None
        start = int(start_index)
        end = int(end_index)
        if start < 0 or end < 0 or start >= node_count or end >= node_count:
            return None
        if start == end:
            return [start]

        blocked = {int(node) for node in (forbidden_nodes or set())}
        blocked.discard(start)
        blocked.discard(end)
        graph = GradientScatterDialog._binary_graph(
            node_count,
            edge_pairs,
            forbidden_nodes=blocked,
        )
        if start not in graph or end not in graph:
            return None
        try:
            return [int(node) for node in nx.shortest_path(graph, source=start, target=end)]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    @staticmethod
    def _rgb_basis_color(channel):
        mapping = {
            "R": np.asarray((1.0, 0.0, 0.0), dtype=float),
            "G": np.asarray((0.0, 1.0, 0.0), dtype=float),
            "B": np.asarray((0.0, 0.0, 1.0), dtype=float),
        }
        return np.asarray(mapping.get(str(channel).strip().upper(), (0.5, 0.5, 0.5)), dtype=float)

    @staticmethod
    def _pair_channel_color(first, second):
        pair = frozenset((str(first).strip().upper(), str(second).strip().upper()))
        mapping = {
            frozenset(("R", "B")): np.asarray((0.58, 0.28, 0.82), dtype=float),  # violet
            frozenset(("R", "G")): np.asarray((1.00, 0.55, 0.00), dtype=float),  # orange
            frozenset(("G", "B")): np.asarray((0.10, 0.76, 0.72), dtype=float),  # turquoise
        }
        if pair in mapping:
            return np.asarray(mapping[pair], dtype=float)
        return np.clip(
            0.5 * (
                GradientScatterDialog._rgb_basis_color(first)
                + GradientScatterDialog._rgb_basis_color(second)
            ),
            0.0,
            1.0,
        )

    def _ctx_segment_records_for_full_path(self, path_nodes, anchors, channel_order):
        nodes = [int(node) for node in list(path_nodes or [])]
        anchor_map = {str(key): int(value) for key, value in dict(anchors or {}).items()}
        order = [str(channel) for channel in list(channel_order or []) if str(channel) in anchor_map]
        if len(nodes) < 2 or len(order) < 2:
            return []

        def _segment_record(first, second, segment_nodes):
            color = self._pair_channel_color(first, second)
            return {
                "first": str(first),
                "second": str(second),
                "nodes": [int(node) for node in list(segment_nodes or [])],
                "color": [float(value) for value in color.tolist()],
            }

        if len(order) == 2:
            return [_segment_record(order[0], order[1], nodes)]

        middle_anchor = int(anchor_map[order[1]])
        try:
            split_index = next(
                idx
                for idx, node in enumerate(nodes)
                if int(node) == middle_anchor and 0 < idx < len(nodes) - 1
            )
        except StopIteration:
            return [_segment_record(order[0], order[-1], nodes)]

        return [
            _segment_record(order[0], order[1], nodes[: split_index + 1]),
            _segment_record(order[1], order[2], nodes[split_index:]),
        ]

    @staticmethod
    def _energy_scaling_range(payload, family_type):
        if not isinstance(payload, dict):
            return None
        scaling = dict(payload.get("energy_width_scaling", {})).get(str(family_type), None)
        if not isinstance(scaling, dict):
            return None
        try:
            emin = float(scaling.get("min"))
            emax = float(scaling.get("max"))
        except Exception:
            return None
        if not np.isfinite(emin) or not np.isfinite(emax):
            return None
        return {"min": emin, "max": emax}

    @staticmethod
    def _path_display_width(
        base_width,
        energy=None,
        scaling=None,
        *,
        mode="scatter",
        scaling_mode="exp",
        scaling_strength=2.0,
    ):
        try:
            base = max(0.05, float(base_width))
        except Exception:
            base = 0.45
        if mode == "brain":
            default_width = max(1.2, base * 6.0)
            min_scale = 4.5
            max_scale = 9.0
        else:
            default_width = max(0.7, base * 2.0)
            min_scale = 1.2
            max_scale = 4.0
        if energy is None or scaling is None:
            return default_width
        try:
            energy_value = float(energy)
            emin = float(scaling.get("min"))
            emax = float(scaling.get("max"))
        except Exception:
            return default_width
        if not np.isfinite(energy_value) or not np.isfinite(emin) or not np.isfinite(emax):
            return default_width
        if np.isclose(emax, emin):
            norm = 0.5
        else:
            norm = float(np.clip((energy_value - emin) / (emax - emin), 0.0, 1.0))
        mode_name = GradientScatterDialog._normalize_path_width_scaling_mode(scaling_mode)
        try:
            scale_value = max(0.05, float(scaling_strength))
        except Exception:
            scale_value = 2.0
        if mode_name == "linear":
            mapped = norm
        elif mode_name == "log":
            mapped = float(np.log1p(scale_value * norm) / np.log1p(scale_value))
        else:
            denominator = float(np.expm1(scale_value))
            if np.isclose(denominator, 0.0):
                mapped = norm
            else:
                mapped = float(np.expm1(scale_value * norm) / denominator)
        return max(default_width * 0.7, base * (min_scale + (max_scale - min_scale) * mapped))

    @staticmethod
    def _enumerate_simple_paths(
        node_count,
        edge_pairs,
        edge_weights,
        start_index,
        end_index,
        *,
        forbidden_nodes=None,
        max_paths=96,
        max_depth=24,
    ):
        """Enumerate capped simple paths on the binary adjacency; energies rank paths later."""
        if node_count <= 0:
            return []
        start = int(start_index)
        end = int(end_index)
        if start < 0 or end < 0 or start >= node_count or end >= node_count:
            return []
        if start == end:
            return [[start]]

        blocked = {int(node) for node in (forbidden_nodes or set())}
        blocked.discard(start)
        blocked.discard(end)
        graph = GradientScatterDialog._binary_graph(
            node_count,
            edge_pairs,
            forbidden_nodes=blocked,
        )
        max_paths = max(1, int(max_paths))
        max_depth = max(2, int(max_depth))
        if start not in graph or end not in graph:
            return []
        try:
            results = []
            seen = set()
            shortest = nx.shortest_path(graph, source=start, target=end)
            if len(shortest) <= max_depth:
                shortest_nodes = tuple(int(node) for node in shortest)
                results.append(list(shortest_nodes))
                seen.add(shortest_nodes)
            paths = nx.all_simple_paths(
                graph,
                source=start,
                target=end,
                cutoff=max_depth - 1,
            )
            for path in paths:
                nodes = tuple(int(node) for node in path)
                if nodes in seen:
                    continue
                results.append(list(nodes))
                seen.add(nodes)
                if len(results) >= max_paths:
                    break
            return results[:max_paths]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    @staticmethod
    def _ordered_anchor_paths(node_count, edge_pairs, edge_weights, channel_order, anchors, forbidden_nodes=None):
        order = [str(channel) for channel in channel_order if str(channel) in anchors]
        if len(order) < 3:
            return []

        visited_nodes = set(int(node) for node in (forbidden_nodes or set()))
        segments = []
        for idx in range(len(order) - 1):
            first = order[idx]
            second = order[idx + 1]
            start = int(anchors[first])
            end = int(anchors[second])
            forbidden = set(visited_nodes)
            forbidden.discard(start)
            forbidden.discard(end)
            for future_channel in order[idx + 2 :]:
                future_anchor = anchors.get(future_channel)
                if future_anchor is not None and int(future_anchor) not in {start, end}:
                    forbidden.add(int(future_anchor))
            path = GradientScatterDialog._shortest_path(
                node_count,
                edge_pairs,
                edge_weights,
                start,
                end,
                forbidden_nodes=forbidden,
            )
            if path is None or len(path) < 2:
                return []
            segments.append((first, second, path))
            visited_nodes.update(int(node) for node in path)
        return segments

    @staticmethod
    def _ordered_anchor_pair_paths(node_count, edge_pairs, edge_weights, channel_order, anchors, forbidden_nodes=None):
        order = [str(channel) for channel in channel_order if str(channel) in anchors]
        if len(order) < 2:
            return []

        pair_paths = []
        for idx in range(len(order) - 1):
            first = order[idx]
            second = order[idx + 1]
            start = int(anchors[first])
            end = int(anchors[second])
            forbidden = {int(node) for node in (forbidden_nodes or set())}
            for channel in order:
                anchor = anchors.get(channel)
                if anchor is None:
                    continue
                anchor = int(anchor)
                if channel not in {first, second}:
                    forbidden.add(anchor)
            shortest = GradientScatterDialog._shortest_path(
                node_count,
                edge_pairs,
                edge_weights,
                start,
                end,
                forbidden_nodes=forbidden,
            )
            if shortest is None or len(shortest) < 2:
                pair_paths.append((first, second, []))
                continue
            max_depth = min(max(2, len(shortest) + 4), max(2, int(node_count)))
            all_paths = GradientScatterDialog._enumerate_simple_paths(
                node_count,
                edge_pairs,
                edge_weights,
                start,
                end,
                forbidden_nodes=forbidden,
                max_paths=96,
                max_depth=min(max_depth, 24),
            )
            pair_paths.append((first, second, all_paths))
        return pair_paths

    @staticmethod
    def _path_segments(x_plot, y_plot, node_path):
        if node_path is None or len(node_path) < 2:
            return np.zeros((0, 2, 2), dtype=float)
        coords = np.column_stack((np.asarray(x_plot, dtype=float), np.asarray(y_plot, dtype=float)))
        segments = []
        for idx in range(len(node_path) - 1):
            a = int(node_path[idx])
            b = int(node_path[idx + 1])
            segments.append(np.asarray((coords[a], coords[b]), dtype=float))
        if not segments:
            return np.zeros((0, 2, 2), dtype=float)
        return np.asarray(segments, dtype=float)

    @staticmethod
    def _path_metric_length(coords, node_path):
        nodes = np.asarray([int(node) for node in list(node_path or [])], dtype=int)
        points = np.asarray(coords, dtype=float)
        if nodes.size < 2 or points.ndim != 2 or points.shape[1] < 1:
            return None
        if np.any((nodes < 0) | (nodes >= points.shape[0])):
            return None
        path_points = np.asarray(points[nodes, :], dtype=float)
        if not np.all(np.isfinite(path_points)):
            return None
        steps = np.diff(path_points, axis=0)
        if steps.shape[0] == 0:
            return None
        lengths = np.linalg.norm(steps, axis=1)
        if not np.all(np.isfinite(lengths)):
            return None
        return float(np.sum(lengths))

    @staticmethod
    def _sort_paths_by_energy(paths, energies, metric_coords=None):
        path_list = [
            [int(node) for node in list(path_nodes or [])]
            for path_nodes in list(paths or [])
            if len(list(path_nodes or [])) >= 2
        ]
        energy_values = np.asarray(energies, dtype=float).reshape(-1)
        records = []
        for idx, path_nodes in enumerate(path_list):
            energy = float(energy_values[idx]) if idx < energy_values.size and np.isfinite(energy_values[idx]) else float("nan")
            path_length = None
            if metric_coords is not None:
                path_length = GradientScatterDialog._path_metric_length(metric_coords, path_nodes)
            finite_energy = bool(np.isfinite(energy))
            finite_length = path_length is not None and np.isfinite(path_length)
            records.append(
                (
                    0 if finite_energy else 1,
                    float(energy) if finite_energy else float("inf"),
                    float(path_length) if finite_length else float("inf"),
                    len(path_nodes),
                    idx,
                    path_nodes,
                    float(energy) if finite_energy else float("nan"),
                )
            )
        records.sort(key=lambda item: item[:5])
        return [list(record[5]) for record in records], [float(record[6]) for record in records]

    @staticmethod
    def _path_edge_pairs(node_path):
        nodes = np.asarray([int(node) for node in list(node_path or [])], dtype=int)
        if nodes.size < 2:
            return np.zeros((0, 2), dtype=int)
        return np.column_stack((nodes[:-1], nodes[1:])).astype(int, copy=False)

    @staticmethod
    def _load_edge_bundling_utils():
        from mrsitoolbox.graphplot.edge_bundling import bundle_edges_2d, polylines_to_segments

        return bundle_edges_2d, polylines_to_segments

    def _bundled_segments_from_pairs(self, edge_pairs):
        pairs = np.asarray(edge_pairs, dtype=int)
        if pairs.ndim != 2 or pairs.shape[0] == 0:
            return np.zeros((0, 2, 2), dtype=float)
        try:
            bundle_edges_2d, polylines_to_segments = self._load_edge_bundling_utils()
            bundled = bundle_edges_2d(self._display_coords, pairs, method="hammer")
            segments = polylines_to_segments(bundled.polylines)
            if segments.size:
                self._edge_bundling_note = ""
                return np.asarray(segments, dtype=float)
            self._edge_bundling_note = "straight fallback"
        except Exception as exc:
            self._edge_bundling_note = "bundle unavailable"
            warn(f"Edge bundling fallback to straight segments: {exc}")
        points = np.asarray(self._display_coords, dtype=float)
        return np.stack((points[pairs[:, 0], :], points[pairs[:, 1], :]), axis=1)

    def _build_triangular_anchor_paths_payload(
        self,
        x_plot,
        y_plot,
        triangle_model,
        path_channel_order,
        point_colors,
        visible_edge_pairs,
        visible_edge_distances,
        anchor_x_plot=None,
        anchor_y_plot=None,
    ):
        if visible_edge_pairs.shape[0] == 0:
            return None
        scatter_coords = np.column_stack((np.asarray(x_plot, dtype=float), np.asarray(y_plot, dtype=float)))
        anchor_x_plot = np.asarray(x_plot if anchor_x_plot is None else anchor_x_plot, dtype=float)
        anchor_y_plot = np.asarray(y_plot if anchor_y_plot is None else anchor_y_plot, dtype=float)
        metric_coords = (
            np.asarray(self._path_metric_coords, dtype=float)
            if self._path_metric_coords is not None
            else np.asarray(scatter_coords, dtype=float)
        )
        project_group_paths = []
        valid_optimal_paths = []

        for group_spec in self._path_group_specs():
            eligible_mask = np.asarray(group_spec.get("eligible_mask"), dtype=bool).reshape(-1)
            if eligible_mask.shape[0] != self._x.shape[0]:
                continue
            candidate_indices = np.flatnonzero(eligible_mask)
            anchors = self._resolved_rgb_anchor_indices(
                anchor_x_plot,
                anchor_y_plot,
                triangle_model,
                candidate_indices=candidate_indices,
                group_name=group_spec.get("name", "all"),
            )
            if not {"R", "G", "B"}.issubset(set(anchors.keys())):
                continue
            forbidden_nodes = set(np.flatnonzero(~eligible_mask).tolist())
            group_pair_paths = []
            path_order = [str(channel) for channel in list(str(path_channel_order or self._triangular_color_order))]
            pair_paths = self._ordered_anchor_pair_paths(
                self._x.size,
                visible_edge_pairs,
                visible_edge_distances,
                path_order,
                anchors,
                forbidden_nodes=forbidden_nodes,
            )
            valid_pair_records = []
            for first, second, paths in pair_paths:
                color = tuple(self._pair_channel_color(first, second).tolist())
                valid_paths = []
                for path in paths:
                    record = self._path_record(first, second, path, color)
                    record["group"] = str(group_spec.get("name", "all"))
                    group_pair_paths.append(record)
                    valid_paths.append([int(node) for node in list(path)])
                valid_pair_records.append((first, second, valid_paths))

            order_channels = [str(channel) for channel in path_order]
            subc_paths = []
            subc_optimal_path = []
            subc_color = None
            subc_anchor_index = self._find_subc_anchor_index(
                candidate_indices,
                group_spec.get("name", "all"),
            )
            if (
                subc_anchor_index is not None
                and len(order_channels) >= 2
                and order_channels[1] in anchors
            ):
                start_channel = str(order_channels[1])
                start_index = int(anchors[start_channel])
                start_color = (
                    np.asarray(point_colors[start_index], dtype=float)
                    if 0 <= start_index < np.asarray(point_colors, dtype=float).shape[0]
                    else np.asarray((0.0, 0.0, 0.0), dtype=float)
                )
                target_color = (
                    np.asarray(point_colors[subc_anchor_index], dtype=float)
                    if 0 <= subc_anchor_index < np.asarray(point_colors, dtype=float).shape[0]
                    else np.asarray((0.0, 0.0, 0.0), dtype=float)
                )
                subc_color = np.clip(0.5 * (start_color + target_color), 0.0, 1.0)
                subc_forbidden_nodes = set(forbidden_nodes)
                first_ctx_anchor = anchors.get(order_channels[0])
                if first_ctx_anchor is not None and int(first_ctx_anchor) not in {start_index, int(subc_anchor_index)}:
                    subc_forbidden_nodes.add(int(first_ctx_anchor))
                shortest_subc = self._shortest_path(
                    self._x.size,
                    visible_edge_pairs,
                    visible_edge_distances,
                    start_index,
                    int(subc_anchor_index),
                    forbidden_nodes=subc_forbidden_nodes,
                )
                if shortest_subc is not None and len(shortest_subc) >= 2:
                    max_depth = min(max(2, len(shortest_subc) + 4), max(2, int(self._x.size)))
                    all_subc_paths = self._enumerate_simple_paths(
                        self._x.size,
                        visible_edge_pairs,
                        visible_edge_distances,
                        start_index,
                        int(subc_anchor_index),
                        forbidden_nodes=subc_forbidden_nodes,
                        max_paths=96,
                        max_depth=min(max_depth, 24),
                    )
                    for path in all_subc_paths:
                        subc_paths.append([int(node) for node in list(path)])
                    if not subc_paths:
                        subc_paths = [[int(node) for node in list(shortest_subc)]]
                    if subc_paths:
                        subc_optimal_path = list(subc_paths[0])

            ordered_segments = self._ordered_anchor_paths(
                self._x.size,
                visible_edge_pairs,
                visible_edge_distances,
                path_order,
                anchors,
                forbidden_nodes=forbidden_nodes,
            )
            valid_ordered_segments = []
            for first, second, path in ordered_segments:
                segments = self._path_segments(x_plot, y_plot, path)
                if segments.shape[0] == 0:
                    continue
                valid_ordered_segments.append((first, second, [int(node) for node in list(path)]))

            all_full_paths = self._combine_ordered_path_records(valid_pair_records, max_full_paths=256)
            ctx_path_energies = []
            for path_nodes in list(all_full_paths):
                energy = self._ctx_full_path_energy(
                    scatter_coords,
                    path_nodes,
                    anchors,
                    path_order,
                )
                ctx_path_energies.append(float(energy) if energy is not None and np.isfinite(energy) else float("nan"))
            all_full_paths, ctx_path_energies = self._sort_paths_by_energy(
                all_full_paths,
                ctx_path_energies,
                metric_coords=metric_coords,
            )
            optimal_full_path = []
            optimal_full_energy = None
            if all_full_paths:
                optimal_full_path = [int(node) for node in list(all_full_paths[0])]
                if ctx_path_energies and np.isfinite(ctx_path_energies[0]):
                    optimal_full_energy = float(ctx_path_energies[0])
            if not optimal_full_path:
                fallback_path = self._combine_ordered_segments(valid_ordered_segments)
                if len(fallback_path) >= 2:
                    optimal_full_path = [int(node) for node in list(fallback_path)]
                    fallback_energy = self._ctx_full_path_energy(
                        scatter_coords,
                        optimal_full_path,
                        anchors,
                        path_order,
                    )
                    if fallback_energy is not None and np.isfinite(fallback_energy):
                        optimal_full_energy = float(fallback_energy)

            optimal_segment_records = self._ctx_segment_records_for_full_path(
                optimal_full_path,
                anchors,
                path_order,
            )

            optimal_subc_energy = None
            subc_path_energies = []
            if subc_paths:
                start_channel = str(order_channels[1]) if len(order_channels) >= 2 else ""
                start_index = int(anchors[start_channel]) if start_channel in anchors else None
                if start_index is not None and subc_anchor_index is not None:
                    subc_ref_unit = self._reference_unit_vector(start_index, int(subc_anchor_index))
                    if subc_ref_unit is not None:
                        subc_path_energies = []
                        for path_nodes in list(subc_paths):
                            energy = self._path_directionality_energy(
                                scatter_coords,
                                path_nodes,
                                subc_ref_unit,
                            )
                            subc_path_energies.append(float(energy) if energy is not None and np.isfinite(energy) else float("nan"))
                        subc_paths, subc_path_energies = self._sort_paths_by_energy(
                            subc_paths,
                            subc_path_energies,
                            metric_coords=metric_coords,
                        )
                        if subc_paths:
                            subc_optimal_path = [int(node) for node in list(subc_paths[0])]
                            if subc_path_energies and np.isfinite(subc_path_energies[0]):
                                optimal_subc_energy = float(subc_path_energies[0])

            group_payload = {
                "group": str(group_spec.get("name", "all")),
                "anchors": {str(key): int(value) for key, value in anchors.items()},
                "optimal_segments": [
                    self._path_record(
                        str(record.get("first", "")),
                        str(record.get("second", "")),
                        record.get("nodes", []),
                        tuple(np.asarray(record.get("color", (0.0, 0.0, 0.0)), dtype=float).reshape(3).tolist()),
                    )
                    for record in list(optimal_segment_records)
                ],
                "optimal_full_path": [int(node) for node in optimal_full_path] if len(optimal_full_path) >= 2 else [],
                "all_full_paths": all_full_paths,
                "ctx_path_energies": [float(value) if np.isfinite(value) else float("nan") for value in ctx_path_energies],
                "full_path_count": 0,
                "ctx_path_count": 0,
                "all_pair_paths": group_pair_paths,
                "subc_anchor": int(subc_anchor_index) if subc_anchor_index is not None else None,
                "subc_paths": [list(path) for path in subc_paths],
                "subc_path_energies": [float(value) if np.isfinite(value) else float("nan") for value in subc_path_energies],
                "subc_optimal_path": [int(node) for node in subc_optimal_path] if len(subc_optimal_path) >= 2 else [],
                "ctx_optimal_path_energy": float(optimal_full_energy) if optimal_full_energy is not None else None,
                "subc_optimal_path_energy": float(optimal_subc_energy) if optimal_subc_energy is not None else None,
                "subc_color": [float(value) for value in np.asarray(subc_color, dtype=float).tolist()] if subc_color is not None else [],
                "subc_path_count": int(len(subc_paths)),
            }
            self._apply_selected_ctx_path_to_group(group_payload)
            group_payload["full_path_count"] = int(len(group_payload["all_full_paths"]))
            group_payload["ctx_path_count"] = int(group_payload["full_path_count"])
            if group_payload["optimal_full_path"]:
                valid_optimal_paths.append(group_payload["optimal_full_path"])
            project_group_paths.append(group_payload)

        if not project_group_paths:
            return None
        return {
            "channel_order": str(path_channel_order or self._triangular_color_order),
            "color_order": str(triangle_model.get("order", self._triangular_color_order)),
            "fit_mode": str(triangle_model.get("fit_mode", self._rgb_fit_mode)),
            "group_paths": project_group_paths,
            "optimal_full_path": list(valid_optimal_paths[0]) if valid_optimal_paths else [],
            "show_all_ordered_paths": bool(self._show_all_ordered_paths),
            "rotation_preset": str(self._rotation_preset),
            "radius": float(self._proximity_radius),
        }

    def _draw_triangular_anchor_paths(self, ax, x_plot, y_plot, point_colors, path_payload, group_name=None):
        if not isinstance(path_payload, dict):
            return
        target_group = str(group_name or "").strip().lower()
        group_payloads = []
        for group_payload in list(path_payload.get("group_paths", [])):
            payload = dict(group_payload or {})
            payload_group = str(payload.get("group", "all")).strip().lower()
            if target_group and payload_group != target_group:
                continue
            group_payloads.append(payload)
        if not group_payloads:
            return

        width_mode = self._normalize_path_width_scaling_mode(
            path_payload.get("width_scaling_mode", self._path_width_scaling_mode)
        )
        try:
            width_strength = max(
                0.05,
                float(path_payload.get("width_scaling_strength", self._path_width_scaling_strength)),
            )
        except Exception:
            width_strength = float(self._path_width_scaling_strength)
        all_segments = []
        all_colors = []
        all_widths = []
        highlighted_segments = []
        highlighted_colors = []
        highlighted_widths = []
        anchor_positions = []
        anchor_colors = []
        anchor_labels = []
        bundled_all_groups = {}
        bundled_highlight_groups = {}
        channel_order = [str(channel) for channel in str(path_payload.get("channel_order", self._triangular_color_order or ""))]
        ctx_scaling = self._energy_scaling_range(path_payload, "ctx")
        subc_scaling = self._energy_scaling_range(path_payload, "subc")

        for group_payload in group_payloads:
            anchors = dict(group_payload.get("anchors", {}))
            ctx_all_paths = [
                [int(node) for node in list(path_nodes or [])]
                for path_nodes in list(group_payload.get("all_full_paths", []))
            ]
            ctx_all_energies = [
                float(value) if np.isfinite(value) else float("nan")
                for value in np.asarray(group_payload.get("ctx_path_energies", []), dtype=float).reshape(-1).tolist()
            ]
            if self._show_all_ordered_paths:
                for idx, path_nodes in enumerate(ctx_all_paths):
                    energy = ctx_all_energies[idx] if idx < len(ctx_all_energies) else None
                    path_width = self._path_display_width(
                        self._edge_linewidth,
                        energy,
                        ctx_scaling,
                        mode="scatter",
                        scaling_mode=width_mode,
                        scaling_strength=width_strength,
                    )
                    for record in self._ctx_segment_records_for_full_path(path_nodes, anchors, channel_order):
                        color = tuple(np.asarray(record.get("color", (0.0, 0.0, 0.0)), dtype=float).reshape(3).tolist())
                        node_pairs = self._path_edge_pairs(record.get("nodes", []))
                        if self._use_edge_bundling and node_pairs.size:
                            bundle_key = (
                                tuple(np.round(np.asarray(color, dtype=float), 6).tolist()),
                                round(float(path_width), 2),
                            )
                            bundled_all_groups.setdefault(bundle_key, []).append(node_pairs)
                        else:
                            segments = self._path_segments(x_plot, y_plot, record.get("nodes", []))
                            for segment in segments:
                                all_segments.append(segment)
                                all_colors.append(color)
                                all_widths.append(path_width)
                subc_color = np.asarray(group_payload.get("subc_color", (0.0, 0.0, 0.0)), dtype=float).reshape(-1)
                if subc_color.shape != (3,):
                    subc_color = np.asarray((0.0, 0.0, 0.0), dtype=float)
                subc_energies = [
                    float(value) if np.isfinite(value) else float("nan")
                    for value in np.asarray(group_payload.get("subc_path_energies", []), dtype=float).reshape(-1).tolist()
                ]
                for idx, path_nodes in enumerate(list(group_payload.get("subc_paths", []))):
                    nodes = [int(node) for node in list(path_nodes or [])]
                    path_width = self._path_display_width(
                        self._edge_linewidth,
                        subc_energies[idx] if idx < len(subc_energies) else None,
                        subc_scaling,
                        mode="scatter",
                        scaling_mode=width_mode,
                        scaling_strength=width_strength,
                    )
                    subc_color_tuple = tuple(subc_color.tolist())
                    node_pairs = self._path_edge_pairs(nodes)
                    if self._use_edge_bundling and node_pairs.size:
                        bundle_key = (
                            tuple(np.round(np.asarray(subc_color_tuple, dtype=float), 6).tolist()),
                            round(float(path_width), 2),
                        )
                        bundled_all_groups.setdefault(bundle_key, []).append(node_pairs)
                    else:
                        segments = self._path_segments(x_plot, y_plot, nodes)
                        for segment in segments:
                            all_segments.append(segment)
                            all_colors.append(subc_color_tuple)
                            all_widths.append(path_width)

            optimal_full_path = self._selected_ctx_path_nodes(group_payload)
            optimal_ctx_width = self._path_display_width(
                self._edge_linewidth,
                self._selected_ctx_path_energy(group_payload),
                ctx_scaling,
                mode="scatter",
                scaling_mode=width_mode,
                scaling_strength=width_strength,
            )
            for record in self._ctx_segment_records_for_full_path(optimal_full_path, anchors, channel_order):
                color = tuple(np.asarray(record.get("color", (0.0, 0.0, 0.0)), dtype=float).reshape(3).tolist())
                node_pairs = self._path_edge_pairs(record.get("nodes", []))
                if self._use_edge_bundling and node_pairs.size:
                    bundle_key = (
                        tuple(np.round(np.asarray(color, dtype=float), 6).tolist()),
                        round(float(optimal_ctx_width), 2),
                    )
                    bundled_highlight_groups.setdefault(bundle_key, []).append(node_pairs)
                else:
                    segments = self._path_segments(x_plot, y_plot, record.get("nodes", []))
                    for segment in segments:
                        highlighted_segments.append(segment)
                        highlighted_colors.append(color)
                        highlighted_widths.append(optimal_ctx_width)

            subc_color = np.asarray(group_payload.get("subc_color", (0.0, 0.0, 0.0)), dtype=float).reshape(-1)
            if subc_color.shape != (3,):
                subc_color = np.asarray((0.0, 0.0, 0.0), dtype=float)
            subc_segments = self._path_segments(x_plot, y_plot, list(group_payload.get("subc_optimal_path", [])))
            subc_optimal_width = self._path_display_width(
                self._edge_linewidth,
                group_payload.get("subc_optimal_path_energy"),
                subc_scaling,
                mode="scatter",
                scaling_mode=width_mode,
                scaling_strength=width_strength,
            )
            if self._use_edge_bundling:
                node_pairs = self._path_edge_pairs(list(group_payload.get("subc_optimal_path", [])))
                if node_pairs.size:
                    bundle_key = (
                        tuple(np.round(np.asarray(subc_color, dtype=float), 6).tolist()),
                        round(float(subc_optimal_width), 2),
                    )
                    bundled_highlight_groups.setdefault(bundle_key, []).append(node_pairs)
            else:
                for segment in subc_segments:
                    highlighted_segments.append(segment)
                    highlighted_colors.append(tuple(subc_color.tolist()))
                    highlighted_widths.append(subc_optimal_width)

            for channel in ("R", "G", "B"):
                index = dict(group_payload.get("anchors", {})).get(channel)
                if index is None:
                    continue
                index = int(index)
                if index < 0 or index >= len(x_plot):
                    continue
                anchor_positions.append((float(x_plot[index]), float(y_plot[index])))
                anchor_labels.append(channel)
                if channel == "R":
                    anchor_colors.append("#ef4444")
                elif channel == "G":
                    anchor_colors.append("#22c55e")
                else:
                    anchor_colors.append("#3b82f6")

            subc_anchor_index = group_payload.get("subc_anchor")
            if subc_anchor_index is not None:
                subc_anchor_index = int(subc_anchor_index)
                if 0 <= subc_anchor_index < len(x_plot):
                    anchor_positions.append((float(x_plot[subc_anchor_index]), float(y_plot[subc_anchor_index])))
                    anchor_labels.append("SUBC")
                    if (
                        0 <= subc_anchor_index < np.asarray(point_colors, dtype=float).shape[0]
                        and np.all(np.isfinite(np.asarray(point_colors[subc_anchor_index], dtype=float)))
                    ):
                        anchor_colors.append(tuple(np.asarray(point_colors[subc_anchor_index], dtype=float).tolist()))
                    else:
                        anchor_colors.append("#111827")

        if self._use_edge_bundling and bundled_all_groups:
            for (color_key, width_key), group_pairs in bundled_all_groups.items():
                merged_pairs = np.vstack(group_pairs) if group_pairs else np.zeros((0, 2), dtype=int)
                bundled_segments = self._bundled_segments_from_pairs(merged_pairs)
                for segment in bundled_segments:
                    all_segments.append(segment)
                    all_colors.append(tuple(color_key))
                    all_widths.append(float(width_key))
        if self._use_edge_bundling and bundled_highlight_groups:
            for (color_key, width_key), group_pairs in bundled_highlight_groups.items():
                merged_pairs = np.vstack(group_pairs) if group_pairs else np.zeros((0, 2), dtype=int)
                bundled_segments = self._bundled_segments_from_pairs(merged_pairs)
                for segment in bundled_segments:
                    highlighted_segments.append(segment)
                    highlighted_colors.append(tuple(color_key))
                    highlighted_widths.append(float(width_key))

        if all_segments:
            ax.add_collection(
                LineCollection(
                    np.asarray(all_segments, dtype=float),
                    colors=all_colors,
                    linewidths=all_widths,
                    alpha=0.35,
                    zorder=3,
                )
            )
        if highlighted_segments:
            ax.add_collection(
                LineCollection(
                    np.asarray(highlighted_segments, dtype=float),
                    colors=highlighted_colors,
                    linewidths=highlighted_widths,
                    alpha=0.95,
                    zorder=4,
                )
            )
        if anchor_positions:
            anchor_positions = np.asarray(anchor_positions, dtype=float)
            ax.scatter(
                anchor_positions[:, 0],
                anchor_positions[:, 1],
                s=92,
                c=anchor_colors,
                edgecolors="#111827",
                linewidths=1.1,
                zorder=5,
            )
            for (x_coord, y_coord), label in zip(anchor_positions.tolist(), anchor_labels):
                ax.annotate(
                    str(label),
                    xy=(float(x_coord), float(y_coord)),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    fontweight="bold",
                    color="#111827",
                    zorder=6,
                )

    def _draw_active_anchor_markers(self, ax, x_plot, y_plot, group_name=None):
        if not self._use_triangular_rgb:
            return
        target_groups = []
        if group_name:
            target_groups = [str(group_name).strip().lower()]
        else:
            target_groups = [str(spec.get("name", "all")).strip().lower() for spec in self._path_group_specs()]
        positions = []
        colors = []
        labels = []
        sizes = []
        linewidths = []
        current_target_group = None
        current_target_channel = None
        if self._endpoint_selection_mode == "manual":
            current_target_group, current_target_channel = self._current_manual_endpoint_target()
        for current_group in target_groups:
            anchors = dict(self._effective_anchor_indices_for_group(current_group))
            for channel in ("R", "G", "B"):
                if channel not in anchors:
                    continue
                idx = int(anchors[channel])
                if idx < 0 or idx >= len(x_plot):
                    continue
                positions.append((float(x_plot[idx]), float(y_plot[idx])))
                colors.append(tuple(self._rgb_basis_color(channel).tolist()))
                labels.append(channel)
                is_current_target = (
                    str(current_group).strip().lower() == str(current_target_group or "").strip().lower()
                    and str(channel).strip().upper() == str(current_target_channel or "").strip().upper()
                )
                sizes.append(135 if is_current_target else 90)
                linewidths.append(2.2 if is_current_target else 1.1)
            subc_index = self._find_subc_anchor_index(self._candidate_indices_for_group(current_group), current_group)
            if subc_index is not None:
                idx = int(subc_index)
                if 0 <= idx < len(x_plot):
                    positions.append((float(x_plot[idx]), float(y_plot[idx])))
                    colors.append("#111827")
                    labels.append("SUBC")
                    is_current_target = (
                        str(current_group).strip().lower() == str(current_target_group or "").strip().lower()
                        and str(current_target_channel or "").strip().upper() == "SUBC"
                    )
                    sizes.append(135 if is_current_target else 90)
                    linewidths.append(2.2 if is_current_target else 1.1)
        if not positions:
            return
        coords = np.asarray(positions, dtype=float)
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            s=sizes,
            c=colors,
            edgecolors="#111827",
            linewidths=linewidths,
            zorder=5,
        )
        for (x_coord, y_coord), label in zip(coords.tolist(), labels):
            ax.annotate(
                str(label),
                xy=(float(x_coord), float(y_coord)),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
                color="#111827",
                zorder=6,
            )

    @staticmethod
    def _edge_subset_for_indices(edge_pairs, edge_distances, allowed_indices):
        pairs = np.asarray(edge_pairs, dtype=int)
        distances = np.asarray(edge_distances, dtype=float).reshape(-1)
        allowed = {int(value) for value in np.asarray(allowed_indices, dtype=int).reshape(-1).tolist()}
        if pairs.ndim != 2 or pairs.shape[0] == 0 or not allowed:
            return np.zeros((0, 2), dtype=int), np.zeros(0, dtype=float)
        keep_mask = np.asarray(
            [(int(pair[0]) in allowed and int(pair[1]) in allowed) for pair in pairs.tolist()],
            dtype=bool,
        )
        if not np.any(keep_mask):
            return np.zeros((0, 2), dtype=int), np.zeros(0, dtype=float)
        subset_pairs = np.asarray(pairs[keep_mask], dtype=int)
        subset_distances = np.asarray(distances[keep_mask], dtype=float) if distances.shape[0] == pairs.shape[0] else np.zeros(subset_pairs.shape[0], dtype=float)
        return subset_pairs, subset_distances

    def _draw_proximity_overlay(self, ax, x_plot, y_plot):
        if not self._show_proximity_circles or self._proximity_radius <= 0.0:
            return
        for x_coord, y_coord in zip(x_plot, y_plot):
            ax.add_patch(
                Circle(
                    (float(x_coord), float(y_coord)),
                    radius=float(self._proximity_radius),
                    facecolor="#9ca3af",
                    edgecolor="#6b7280",
                    linewidth=0.45,
                    alpha=0.11,
                    zorder=0,
                )
            )

    def _on_proximity_toggled(self, checked):
        self._show_proximity_circles = bool(checked)
        self._sync_proximity_controls()
        self._render()

    def _on_proximity_slider_changed(self, value):
        self._proximity_radius = self._slider_to_radius(value)
        self._invalidate_generated_paths()
        self._sync_proximity_controls()
        self._render()

    def _on_edge_width_changed(self, value):
        try:
            self._edge_linewidth = max(0.05, float(value))
        except Exception:
            self._edge_linewidth = 0.45
        if isinstance(self._project_paths_payload, dict):
            self._project_paths_payload["edge_linewidth"] = float(self._edge_linewidth)
        self._render()

    def _on_path_width_mode_changed(self, _index):
        self._path_width_scaling_mode = self._normalize_path_width_scaling_mode(
            self.path_width_mode_combo.currentData()
        )
        if isinstance(self._project_paths_payload, dict):
            self._project_paths_payload["width_scaling_mode"] = self._path_width_scaling_mode
        self._render()

    def _on_path_width_scale_changed(self, value):
        try:
            self._path_width_scaling_strength = max(0.05, float(value))
        except Exception:
            self._path_width_scaling_strength = 2.0
        if isinstance(self._project_paths_payload, dict):
            self._project_paths_payload["width_scaling_strength"] = float(self._path_width_scaling_strength)
        self._render()

    def _on_all_paths_toggled(self, checked):
        self._show_all_ordered_paths = bool(checked)
        if isinstance(self._project_paths_payload, dict):
            self._project_paths_payload["show_all_ordered_paths"] = bool(self._show_all_ordered_paths)
        self._render()

    def _on_show_adjacency_toggled(self, checked):
        self._show_adjacency_edges = bool(checked)
        self._render()

    def _on_edge_bundling_toggled(self, checked):
        self._use_edge_bundling = bool(checked)
        self._render()

    def _on_free_energy_norm_segments_toggled(self, checked):
        self._normalize_free_energy_by_segments = bool(checked)
        self._clear_free_energy_payload()
        self._sync_proximity_controls()
        self._render()

    def _on_path_order_combo_changed(self, index):
        requested = [
            self.path_order_first_combo.currentData(),
            self.path_order_second_combo.currentData(),
            self.path_order_third_combo.currentData(),
        ]
        current = self._path_channel_order or self._triangular_color_order
        self._path_channel_order = self._coerce_path_channel_order(requested, fallback=current)
        self._populate_path_order_combos()
        self._invalidate_generated_paths()
        self._sync_proximity_controls()
        self._render()

    def _on_plot_metabolite_profiles_clicked(self):
        if not self._use_triangular_rgb or self._is_3d:
            return
        profiles = self._metabolite_profiles
        if profiles is None:
            return
        profiles = np.asarray(profiles, dtype=float)
        profiles_are_subjectwise = profiles.ndim == 3
        if profiles_are_subjectwise:
            if profiles.shape[1] != self._x.shape[0] or profiles.shape[2] == 0:
                return
        elif profiles.ndim == 2:
            if profiles.shape[0] != self._x.shape[0] or profiles.shape[1] == 0:
                return
        else:
            return
        selected_subject_index = -1
        selected_subject_label = "Average subjects"
        if profiles_are_subjectwise:
            try:
                selected_subject_index = int(self.metabolite_subject_combo.currentData())
            except Exception:
                selected_subject_index = -1
            if 0 <= selected_subject_index < profiles.shape[0]:
                profiles = np.asarray(profiles[[selected_subject_index], :, :], dtype=float)
                if 0 <= selected_subject_index < len(self._metabolite_subject_labels):
                    selected_subject_label = str(self._metabolite_subject_labels[selected_subject_index] or "").strip()
                if not selected_subject_label:
                    selected_subject_label = f"Subject {selected_subject_index + 1}"
        subject_title_suffix = f" | {selected_subject_label}" if profiles_are_subjectwise else ""
        rgb_x_plot, rgb_y_plot = self._rotated_rgb_points()
        model = self._last_rgb_model
        if model is None:
            try:
                model = self._rgb_model(
                    rgb_x_plot,
                    rgb_y_plot,
                    self._triangular_color_order,
                    fit_mode=self._rgb_fit_mode,
                )
            except Exception:
                return
        scalar = self._rgb_scalar_from_model(
            rgb_x_plot,
            rgb_y_plot,
            model,
            scalar_mode=self._rgb_scalar_mode,
        )
        scalar_mode_label = (
            "principal-curve coordinate"
            if self._rgb_scalar_mode == "principal_curve"
            else "triangular RGB embedding"
        )
        if profiles_are_subjectwise:
            profile_finite = np.any(np.isfinite(profiles), axis=(0, 2))
        else:
            profile_finite = np.any(np.isfinite(profiles), axis=1)
        finite = np.isfinite(scalar) & profile_finite
        group_codes = np.asarray(self._point_group_codes, dtype=int).reshape(-1)
        if group_codes.shape[0] != finite.shape[0]:
            group_codes = np.full(finite.shape, -1, dtype=int)
        panel_specs = []
        if self._hemisphere_mode == "lh":
            panel_specs = [("LH", finite & np.isin(group_codes, (0, 2)))]
        elif self._hemisphere_mode == "rh":
            panel_specs = [("RH", finite & np.isin(group_codes, (1, 2)))]
        else:
            lh_mask = finite & np.isin(group_codes, (0, 2))
            rh_mask = finite & np.isin(group_codes, (1, 2))
            if int(np.sum(lh_mask)) >= 2:
                panel_specs.append(("LH", lh_mask))
            if int(np.sum(rh_mask)) >= 2:
                panel_specs.append(("RH", rh_mask))
        if not panel_specs:
            panel_specs = [("All", finite)]
        panel_specs = [(title, mask) for title, mask in panel_specs if int(np.sum(mask)) >= 2]
        if not panel_specs:
            return

        names = list(self._metabolite_names or [])
        n_metabolites = int(profiles.shape[-1])
        if len(names) != n_metabolites:
            names = [f"Metabolite {idx + 1}" for idx in range(n_metabolites)]
        normalized_names = [str(name).strip().lower().replace(" ", "") for name in names]
        zscore_enabled = bool(self.metabolite_zscore_check.isChecked())
        correction_mode = "none"
        if zscore_enabled:
            try:
                correction_mode = str(self.metabolite_correction_combo.currentData() or "none")
            except Exception:
                correction_mode = "none"
        cr_correction_enabled = correction_mode == "cr"
        sum_m_correction_enabled = correction_mode == "sum_m"
        water_name_tokens = {"water", "h2o", "watersignal", "water_signal"}
        non_water_metabolite_indices = [
            idx for idx, name in enumerate(normalized_names) if name not in water_name_tokens
        ]
        baseline_idx = None
        if cr_correction_enabled:
            try:
                baseline_idx = normalized_names.index("crpcr")
            except ValueError:
                return
            keep_metabolite_indices = [idx for idx in range(len(names)) if idx != baseline_idx]
        else:
            keep_metabolite_indices = list(range(len(names)))
        show_water_signal = bool(self.metabolite_show_water_check.isChecked())
        if not show_water_signal:
            keep_metabolite_indices = [
                idx for idx in keep_metabolite_indices if normalized_names[idx] not in water_name_tokens
            ]
        if not keep_metabolite_indices:
            return
        plot_names = [names[idx] for idx in keep_metabolite_indices]
        correction_target_positions = [
            pos
            for pos, original_idx in enumerate(keep_metabolite_indices)
            if normalized_names[original_idx] not in water_name_tokens
        ]
        metabolite_axis_mode = str(self.metabolite_axis_combo.currentData() or "triangular_rgb")
        metabolite_plot_settings = self._current_metabolite_plot_settings()
        metabolite_x_label = str(metabolite_plot_settings.get("x_axis_label", "Gradient 1") or "Gradient 1")
        metabolite_y_label = str(metabolite_plot_settings.get("y_axis_label", "Gradient 2") or "Gradient 2")
        metabolite_x_fontsize = int(metabolite_plot_settings.get("x_axis_fontsize", 11))
        metabolite_y_fontsize = int(metabolite_plot_settings.get("y_axis_fontsize", 11))
        metabolite_tick_fontsize = int(metabolite_plot_settings.get("tick_fontsize", 9))
        metabolite_linewidth = float(metabolite_plot_settings.get("line_width", 1.6))
        metabolite_confidence_interval = float(metabolite_plot_settings.get("confidence_interval", 95.0))
        metabolite_confidence_interval = max(1.0, min(100.0, metabolite_confidence_interval))
        metabolite_ci_lower = (100.0 - metabolite_confidence_interval) / 2.0
        metabolite_ci_upper = 100.0 - metabolite_ci_lower
        metabolite_colors = dict(metabolite_plot_settings.get("colors", {}) or {})
        show_percentile_band = bool(self.metabolite_percentile_band_check.isChecked())
        show_boxplot_bars = bool(metabolite_plot_settings.get("boxplot_bars", False))

        def _median_interval(values, axis=0):
            arr = np.asarray(values, dtype=float)
            if arr.size == 0:
                shape = list(arr.shape)
                if shape:
                    shape.pop(axis)
                return (
                    np.full(shape, np.nan, dtype=float),
                    np.full(shape, np.nan, dtype=float),
                    np.full(shape, np.nan, dtype=float),
                    np.full(shape, np.nan, dtype=float),
                    np.full(shape, np.nan, dtype=float),
                )
            return (
                np.nanmedian(arr, axis=axis),
                np.nanpercentile(arr, metabolite_ci_lower, axis=axis),
                np.nanpercentile(arr, metabolite_ci_upper, axis=axis),
                np.nanpercentile(arr, 25.0, axis=axis),
                np.nanpercentile(arr, 75.0, axis=axis),
            )

        def _draw_boxplot_bars(ax, x_values, low_values, q1_values, q3_values, high_values, color):
            if not show_boxplot_bars:
                return
            x_arr = np.asarray(x_values, dtype=float)
            low_arr = np.asarray(low_values, dtype=float)
            q1_arr = np.asarray(q1_values, dtype=float)
            q3_arr = np.asarray(q3_values, dtype=float)
            high_arr = np.asarray(high_values, dtype=float)
            valid = (
                np.isfinite(x_arr)
                & np.isfinite(low_arr)
                & np.isfinite(q1_arr)
                & np.isfinite(q3_arr)
                & np.isfinite(high_arr)
            )
            if not np.any(valid):
                return
            ax.vlines(
                x_arr[valid],
                low_arr[valid],
                high_arr[valid],
                color=color,
                alpha=0.35,
                linewidth=max(0.7, float(metabolite_linewidth) * 0.65),
                zorder=2,
            )
            ax.vlines(
                x_arr[valid],
                q1_arr[valid],
                q3_arr[valid],
                color=color,
                alpha=0.78,
                linewidth=max(1.8, float(metabolite_linewidth) * 1.8),
                zorder=3,
            )

        def _apply_zscore_correction(z_all_metabolites):
            selected_profiles = np.asarray(z_all_metabolites, dtype=float)[..., keep_metabolite_indices].copy()
            if cr_correction_enabled and baseline_idx is not None and correction_target_positions:
                selected_profiles[..., correction_target_positions] = (
                    selected_profiles[..., correction_target_positions]
                    - np.asarray(z_all_metabolites, dtype=float)[..., [baseline_idx]]
                )
            elif sum_m_correction_enabled and non_water_metabolite_indices and correction_target_positions:
                metabolite_mean = np.nanmean(
                    np.asarray(z_all_metabolites, dtype=float)[..., non_water_metabolite_indices],
                    axis=-1,
                    keepdims=True,
                )
                selected_profiles[..., correction_target_positions] = (
                    selected_profiles[..., correction_target_positions] - metabolite_mean
                )
            return selected_profiles

        def _zscore_ylabel(raw_label):
            if cr_correction_enabled:
                return "Median z-scored signal after CrPCr subtraction"
            if sum_m_correction_enabled:
                return "Median z-scored signal after metabolite-mean subtraction"
            return raw_label

        def _zscore_title_prefix(raw_title, cr_title, mean_title):
            if cr_correction_enabled:
                return cr_title
            if sum_m_correction_enabled:
                return mean_title
            return raw_title

        def _metabolite_colors():
            cmap = self._cmap if self._cmap is not None else GradientSurfaceDialog._default_cmap(self._cmap_name)
            if not callable(cmap):
                try:
                    import matplotlib.cm as mpl_cm

                    cmap = mpl_cm.get_cmap(str(cmap))
                except Exception:
                    cmap = GradientSurfaceDialog._default_cmap("viridis")
            if not callable(cmap):
                import matplotlib.cm as mpl_cm

                cmap = mpl_cm.get_cmap("viridis")
            metabolite_palette = {
                "glx": metabolite_colors.get("glx", "#d62728"),
                "glugln": metabolite_colors.get("glx", "#d62728"),
                "glu+gln": metabolite_colors.get("glx", "#d62728"),
                "cho": metabolite_colors.get("cho", "#2ca02c"),
                "gpcpch": metabolite_colors.get("cho", "#2ca02c"),
                "gpc+pch": metabolite_colors.get("cho", "#2ca02c"),
                "naa": metabolite_colors.get("naa", "#1f77b4"),
                "naanaag": metabolite_colors.get("naa", "#1f77b4"),
                "naa+naag": metabolite_colors.get("naa", "#1f77b4"),
                "ins": metabolite_colors.get("ins", "#f2c94c"),
                "crpcr": metabolite_colors.get("crpcr", "#000000"),
                "water": metabolite_colors.get("water", "#7a7a7a"),
                "watersignal": metabolite_colors.get("water", "#7a7a7a"),
                "water_signal": metabolite_colors.get("water", "#7a7a7a"),
            }
            fallback_colors = cmap(np.linspace(0.05, 0.95, len(plot_names)))
            colors = []
            for met_idx, name in enumerate(plot_names):
                normalized_name = str(name).strip().lower().replace(" ", "")
                colors.append(metabolite_palette.get(normalized_name, fallback_colors[met_idx]))
            return colors

        def _parcel_lobe(parcel_name):
            text = str(parcel_name or "").strip().lower().replace("_", "-")
            if any(token in text for token in ("frontal", "precentral", "paracentral")):
                return "frontal"
            if any(token in text for token in ("temporal", "fusiform", "entorhinal", "parahippocampal", "bankssts")):
                return "temporal"
            if any(token in text for token in ("parietal", "postcentral", "precuneus", "supramarginal")):
                return "parietal"
            if any(token in text for token in ("occipital", "lingual", "cuneus", "pericalcarine")):
                return "occipital"
            if "cing" in text:
                return "cingulate"
            if "insula" in text or "insular" in text:
                return "insula"
            return "subc"

        if metabolite_axis_mode == "parcel_lobe":
            color_values = _metabolite_colors()
            lobe_specs = [
                ("occipital", "Occipital"),
                ("parietal", "Parietal"),
                ("temporal", "Temporal"),
                ("frontal", "Frontal"),
                ("cingulate", "Cingulate"),
                ("insula", "Insula"),
                ("subc", "Subc/other"),
            ]
            parcel_lobes = np.asarray(
                [_parcel_lobe(label) for label in np.asarray(self._point_labels, dtype=object).tolist()],
                dtype=object,
            )

            def _processed_panel_profiles(mask):
                panel_indices = np.flatnonzero(np.asarray(mask, dtype=bool))
                if panel_indices.size == 0:
                    return panel_indices, np.zeros((0, len(plot_names)), dtype=float)
                if profiles_are_subjectwise:
                    local_profiles = np.asarray(profiles[:, panel_indices, :], dtype=float)
                    if zscore_enabled:
                        mean = np.nanmean(local_profiles, axis=1, keepdims=True)
                        std = np.nanstd(local_profiles, axis=1, keepdims=True)
                        std[~np.isfinite(std) | (std <= 1e-12)] = 1.0
                        local_profiles = (local_profiles - mean) / std
                        local_profiles = _apply_zscore_correction(local_profiles)
                    else:
                        local_profiles = local_profiles[:, :, keep_metabolite_indices]
                    return panel_indices, local_profiles
                local_profiles = np.asarray(profiles[panel_indices, :], dtype=float)
                if zscore_enabled:
                    mean = np.nanmean(local_profiles, axis=0, keepdims=True)
                    std = np.nanstd(local_profiles, axis=0, keepdims=True)
                    std[~np.isfinite(std) | (std <= 1e-12)] = 1.0
                    local_profiles = (local_profiles - mean) / std
                    local_profiles = _apply_zscore_correction(local_profiles)
                else:
                    local_profiles = local_profiles[:, keep_metabolite_indices]
                return panel_indices, local_profiles

            panel_profile_payloads = [
                (panel_title, *_processed_panel_profiles(mask))
                for panel_title, mask in panel_specs
            ]
            n_plot_rows = 1 if zscore_enabled else len(plot_names)
            fig_width = 8.8 if len(panel_specs) == 1 else 12.6
            fig_height = 4.2 if zscore_enabled else max(3.2, 1.9 * len(plot_names) + 1.0)
            fig = Figure(figsize=(fig_width, fig_height), constrained_layout=True)
            canvas = FigureCanvas(fig)
            toolbar = NavigationToolbar(canvas, self)
            axes_grid = fig.subplots(n_plot_rows, len(panel_specs), squeeze=False)
            lobe_keys = [key for key, _title in lobe_specs]
            lobe_titles = [title for _key, title in lobe_specs]
            x_values = np.arange(len(lobe_specs), dtype=float)
            for row_idx in range(n_plot_rows):
                for col_idx, (panel_title, panel_indices, panel_profiles) in enumerate(panel_profile_payloads):
                    ax = axes_grid[row_idx, col_idx]
                    if zscore_enabled:
                        any_values = False
                        for met_idx, metabolite_name in enumerate(plot_names):
                            lobe_values = np.full(len(lobe_specs), np.nan, dtype=float)
                            for lobe_idx, lobe_key in enumerate(lobe_keys):
                                lobe_local = np.flatnonzero(parcel_lobes[panel_indices] == lobe_key)
                                if lobe_local.size == 0:
                                    continue
                                values = (
                                    panel_profiles[:, lobe_local, met_idx].reshape(-1)
                                    if np.asarray(panel_profiles).ndim == 3
                                    else panel_profiles[lobe_local, met_idx]
                                )
                                lobe_values[lobe_idx] = np.nanmedian(values)
                            valid_values = np.isfinite(lobe_values)
                            if not np.any(valid_values):
                                continue
                            any_values = True
                            if show_percentile_band or show_boxplot_bars:
                                lobe_low = np.full(len(lobe_specs), np.nan, dtype=float)
                                lobe_high = np.full(len(lobe_specs), np.nan, dtype=float)
                                lobe_q1 = np.full(len(lobe_specs), np.nan, dtype=float)
                                lobe_q3 = np.full(len(lobe_specs), np.nan, dtype=float)
                                for lobe_idx, lobe_key in enumerate(lobe_keys):
                                    lobe_local = np.flatnonzero(parcel_lobes[panel_indices] == lobe_key)
                                    if lobe_local.size == 0:
                                        continue
                                    values = (
                                        panel_profiles[:, lobe_local, met_idx].reshape(-1)
                                        if np.asarray(panel_profiles).ndim == 3
                                        else panel_profiles[lobe_local, met_idx]
                                    )
                                    _median, low, high, q1, q3 = _median_interval(values, axis=0)
                                    lobe_low[lobe_idx] = low
                                    lobe_high[lobe_idx] = high
                                    lobe_q1[lobe_idx] = q1
                                    lobe_q3[lobe_idx] = q3
                                band_mask = valid_values & np.isfinite(lobe_low) & np.isfinite(lobe_high)
                                if show_percentile_band and np.any(band_mask):
                                    ax.fill_between(
                                        x_values[band_mask],
                                        lobe_low[band_mask],
                                        lobe_high[band_mask],
                                        color=color_values[met_idx],
                                        alpha=0.16,
                                        linewidth=0.0,
                                    )
                                _draw_boxplot_bars(
                                    ax,
                                    x_values,
                                    lobe_low,
                                    lobe_q1,
                                    lobe_q3,
                                    lobe_high,
                                    color_values[met_idx],
                                )
                            ax.plot(
                                x_values[valid_values],
                                lobe_values[valid_values],
                                marker="o",
                                markersize=4.0,
                                linewidth=metabolite_linewidth,
                                color=color_values[met_idx],
                                label=str(metabolite_name),
                            )
                        if not any_values:
                            ax.text(0.5, 0.5, "No finite values", transform=ax.transAxes, ha="center", va="center")
                    else:
                        lobe_values = np.full(len(lobe_specs), np.nan, dtype=float)
                        for lobe_idx, lobe_key in enumerate(lobe_keys):
                            lobe_local = np.flatnonzero(parcel_lobes[panel_indices] == lobe_key)
                            if lobe_local.size == 0:
                                continue
                            values = (
                                panel_profiles[:, lobe_local, row_idx].reshape(-1)
                                if np.asarray(panel_profiles).ndim == 3
                                else panel_profiles[lobe_local, row_idx]
                            )
                            lobe_values[lobe_idx] = np.nanmedian(values)
                        valid_values = np.isfinite(lobe_values)
                        if np.any(valid_values):
                            if show_percentile_band or show_boxplot_bars:
                                lobe_low = np.full(len(lobe_specs), np.nan, dtype=float)
                                lobe_high = np.full(len(lobe_specs), np.nan, dtype=float)
                                lobe_q1 = np.full(len(lobe_specs), np.nan, dtype=float)
                                lobe_q3 = np.full(len(lobe_specs), np.nan, dtype=float)
                                for lobe_idx, lobe_key in enumerate(lobe_keys):
                                    lobe_local = np.flatnonzero(parcel_lobes[panel_indices] == lobe_key)
                                    if lobe_local.size == 0:
                                        continue
                                    values = (
                                        panel_profiles[:, lobe_local, row_idx].reshape(-1)
                                        if np.asarray(panel_profiles).ndim == 3
                                        else panel_profiles[lobe_local, row_idx]
                                    )
                                    _median, low, high, q1, q3 = _median_interval(values, axis=0)
                                    lobe_low[lobe_idx] = low
                                    lobe_high[lobe_idx] = high
                                    lobe_q1[lobe_idx] = q1
                                    lobe_q3[lobe_idx] = q3
                                band_mask = valid_values & np.isfinite(lobe_low) & np.isfinite(lobe_high)
                                if show_percentile_band and np.any(band_mask):
                                    ax.fill_between(
                                        x_values[band_mask],
                                        lobe_low[band_mask],
                                        lobe_high[band_mask],
                                        color=color_values[row_idx],
                                        alpha=0.16,
                                        linewidth=0.0,
                                    )
                                _draw_boxplot_bars(
                                    ax,
                                    x_values,
                                    lobe_low,
                                    lobe_q1,
                                    lobe_q3,
                                    lobe_high,
                                    color_values[row_idx],
                                )
                            ax.plot(
                                x_values[valid_values],
                                lobe_values[valid_values],
                                marker="o",
                                markersize=4.0,
                                linewidth=metabolite_linewidth,
                                color=color_values[row_idx],
                            )
                        else:
                            ax.text(0.5, 0.5, "No finite values", transform=ax.transAxes, ha="center", va="center")
                    ax.set_xlim(-0.5, float(len(lobe_specs)) - 0.5)
                    ax.set_xticks(x_values)
                    ax.set_xticklabels(lobe_titles, rotation=35, ha="right", fontsize=metabolite_tick_fontsize)
                    if row_idx == 0:
                        ax.set_title(f"{panel_title} | parcels by lobe")
                    if col_idx == 0 and not zscore_enabled:
                        ax.set_ylabel(str(plot_names[row_idx]), fontsize=metabolite_y_fontsize)
                    ax.tick_params(axis="both", labelsize=metabolite_tick_fontsize)
                    if row_idx == n_plot_rows - 1:
                        ax.set_xlabel(metabolite_x_label, fontsize=metabolite_x_fontsize)
                    ax.grid(True, axis="y", alpha=0.22)
            ylabel = _zscore_ylabel("Median z-scored metabolite signal") if zscore_enabled else "Median raw metabolite signal"
            try:
                fig.supylabel(metabolite_y_label or ylabel, fontsize=metabolite_y_fontsize)
            except Exception:
                pass
            for legend_ax in axes_grid.reshape(-1):
                handles, labels = legend_ax.get_legend_handles_labels()
                if handles:
                    legend_ax.legend(handles, labels, loc="best", fontsize=8, frameon=False)
                    break
            title_prefix = (
                _zscore_title_prefix(
                    "Parcel median z-scored metabolite profiles",
                    "Parcel median metabolite profiles minus CrPCr",
                    "Parcel median metabolite profiles minus metabolite mean",
                )
                if zscore_enabled
                else "Parcel raw median metabolite profiles"
            )
            fig.suptitle(f"{title_prefix} grouped by lobe - {self._title}{subject_title_suffix}")

            dialog = QDialog(self)
            dialog.setWindowTitle("Metabolite profiles by parcel lobe")
            layout = QVBoxLayout(dialog)
            layout.addWidget(toolbar)
            layout.addWidget(canvas, 1)
            dialog.resize(1180, 920)
            dialog._figure = fig
            dialog._canvas = canvas
            dialog._toolbar = toolbar
            if not hasattr(self, "_metabolite_profile_dialogs"):
                self._metabolite_profile_dialogs = []
            self._metabolite_profile_dialogs.append(dialog)
            canvas.draw_idle()
            dialog.show()
            return

        try:
            n_bins = int(self.metabolite_bins_spin.value())
        except Exception:
            n_bins = 12
        n_bins = max(2, min(200, n_bins))
        panel_mask = np.zeros(finite.shape, dtype=bool)
        for _panel_title, mask in panel_specs:
            panel_mask |= np.asarray(mask, dtype=bool)
        scalar_for_bins = np.asarray(scalar[panel_mask], dtype=float)
        scalar_for_bins = scalar_for_bins[np.isfinite(scalar_for_bins)]
        if scalar_for_bins.size < 2:
            return
        scalar_min = float(np.nanmin(scalar_for_bins))
        scalar_max = float(np.nanmax(scalar_for_bins))
        if not np.isfinite(scalar_min) or not np.isfinite(scalar_max) or scalar_max <= scalar_min:
            return
        bin_edges = np.linspace(scalar_min, scalar_max, n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        try:
            point_colors = np.asarray(self._rgb_colors_from_model(rgb_x_plot, rgb_y_plot, model), dtype=float)
        except Exception:
            point_colors = np.full((scalar.shape[0], 3), 0.65, dtype=float)
        color_finite = (
            np.isfinite(scalar)
            & np.all(np.isfinite(point_colors), axis=1)
        )
        if np.sum(color_finite) >= 2:
            color_order = np.argsort(np.asarray(scalar[color_finite], dtype=float))
            color_scalar = np.asarray(scalar[color_finite], dtype=float)[color_order]
            color_rgb = np.clip(np.asarray(point_colors[color_finite, :], dtype=float)[color_order, :], 0.0, 1.0)
            unique_scalar, unique_indices = np.unique(color_scalar, return_index=True)
            color_scalar = color_scalar[unique_indices]
            color_rgb = color_rgb[unique_indices, :]
        else:
            color_scalar = np.asarray([scalar_min, scalar_max], dtype=float)
            color_rgb = np.asarray([[0.65, 0.65, 0.65], [0.65, 0.65, 0.65]], dtype=float)
        strip_x = np.linspace(scalar_min, scalar_max, 512)
        if color_scalar.size >= 2 and color_scalar[-1] > color_scalar[0]:
            strip_rgb = np.column_stack(
                [
                    np.interp(strip_x, color_scalar, color_rgb[:, channel])
                    for channel in range(3)
                ]
            )
        else:
            strip_rgb = np.repeat(color_rgb[:1, :], strip_x.size, axis=0)
        strip_image = np.clip(strip_rgb[np.newaxis, :, :], 0.0, 1.0)
        if not zscore_enabled:
            cmap = self._cmap if self._cmap is not None else GradientSurfaceDialog._default_cmap(self._cmap_name)
            if not callable(cmap):
                try:
                    import matplotlib.cm as mpl_cm

                    cmap = mpl_cm.get_cmap(str(cmap))
                except Exception:
                    cmap = GradientSurfaceDialog._default_cmap("viridis")
            if not callable(cmap):
                import matplotlib.cm as mpl_cm

                cmap = mpl_cm.get_cmap("viridis")
            metabolite_palette = {
                "glx": metabolite_colors.get("glx", "#d62728"),
                "glugln": metabolite_colors.get("glx", "#d62728"),
                "glu+gln": metabolite_colors.get("glx", "#d62728"),
                "cho": metabolite_colors.get("cho", "#2ca02c"),
                "gpcpch": metabolite_colors.get("cho", "#2ca02c"),
                "gpc+pch": metabolite_colors.get("cho", "#2ca02c"),
                "naa": metabolite_colors.get("naa", "#1f77b4"),
                "naanaag": metabolite_colors.get("naa", "#1f77b4"),
                "naa+naag": metabolite_colors.get("naa", "#1f77b4"),
                "ins": metabolite_colors.get("ins", "#f2c94c"),
                "crpcr": metabolite_colors.get("crpcr", "#000000"),
                "water": metabolite_colors.get("water", "#7a7a7a"),
                "watersignal": metabolite_colors.get("water", "#7a7a7a"),
                "water_signal": metabolite_colors.get("water", "#7a7a7a"),
            }
            fallback_colors = cmap(np.linspace(0.05, 0.95, len(plot_names)))
            color_values = []
            for met_idx, name in enumerate(plot_names):
                normalized_name = str(name).strip().lower().replace(" ", "")
                color_values.append(metabolite_palette.get(normalized_name, fallback_colors[met_idx]))

            fig_width = 8.8 if len(panel_specs) == 1 else 12.6
            fig_height = max(3.2, 1.9 * len(plot_names) + 1.0)
            fig = Figure(figsize=(fig_width, fig_height), constrained_layout=True)
            canvas = FigureCanvas(fig)
            toolbar = NavigationToolbar(canvas, self)
            axes_grid = fig.subplots(len(plot_names), len(panel_specs), squeeze=False)

            def _binned_raw_profile(mask, metabolite_index):
                local_scalar = np.asarray(scalar[mask], dtype=float)
                local_valid = np.isfinite(local_scalar)
                if profiles_are_subjectwise:
                    local_profiles = np.asarray(profiles[:, mask, metabolite_index], dtype=float)
                    local_valid &= np.any(np.isfinite(local_profiles), axis=0)
                else:
                    local_profiles = np.asarray(profiles[mask, metabolite_index], dtype=float)
                    local_valid &= np.isfinite(local_profiles)
                if int(np.sum(local_valid)) < 2:
                    empty = np.full(n_bins, np.nan, dtype=float)
                    return empty, empty.copy(), empty.copy(), empty.copy(), empty.copy(), np.zeros(n_bins, dtype=int)
                local_scalar = local_scalar[local_valid]
                if profiles_are_subjectwise:
                    local_profiles = local_profiles[:, local_valid]
                else:
                    local_profiles = local_profiles[local_valid]
                bin_indices = np.digitize(local_scalar, bin_edges, right=False) - 1
                bin_indices = np.clip(bin_indices, 0, n_bins - 1)
                binned = np.full(n_bins, np.nan, dtype=float)
                low = np.full(n_bins, np.nan, dtype=float)
                high = np.full(n_bins, np.nan, dtype=float)
                q1 = np.full(n_bins, np.nan, dtype=float)
                q3 = np.full(n_bins, np.nan, dtype=float)
                bin_counts = np.zeros(n_bins, dtype=int)
                for bin_idx in range(n_bins):
                    in_bin = bin_indices == bin_idx
                    bin_counts[bin_idx] = int(np.sum(in_bin))
                    if bin_counts[bin_idx] <= 0:
                        continue
                    if profiles_are_subjectwise:
                        values = np.asarray(local_profiles[:, in_bin], dtype=float).reshape(-1)
                    else:
                        values = np.asarray(local_profiles[in_bin], dtype=float).reshape(-1)
                    binned[bin_idx], low[bin_idx], high[bin_idx], q1[bin_idx], q3[bin_idx] = _median_interval(values, axis=0)
                return binned, low, high, q1, q3, bin_counts

            for row_idx, (metabolite_index, metabolite_name) in enumerate(
                zip(keep_metabolite_indices, plot_names)
            ):
                for col_idx, (panel_title, mask) in enumerate(panel_specs):
                    ax = axes_grid[row_idx, col_idx]
                    binned_profile, binned_low, binned_high, binned_q1, binned_q3, _bin_counts = _binned_raw_profile(mask, metabolite_index)
                    valid_bins = np.isfinite(binned_profile)
                    if np.any(valid_bins):
                        band_mask = valid_bins & np.isfinite(binned_low) & np.isfinite(binned_high)
                        if show_percentile_band and np.any(band_mask):
                            ax.fill_between(
                                bin_centers[band_mask],
                                binned_low[band_mask],
                                binned_high[band_mask],
                                color=color_values[row_idx],
                                alpha=0.16,
                                linewidth=0.0,
                            )
                        _draw_boxplot_bars(
                            ax,
                            bin_centers,
                            binned_low,
                            binned_q1,
                            binned_q3,
                            binned_high,
                            color_values[row_idx],
                        )
                        ax.plot(
                            bin_centers[valid_bins],
                            binned_profile[valid_bins],
                            marker="o",
                            markersize=4.0,
                            linewidth=metabolite_linewidth,
                            color=color_values[row_idx],
                        )
                    else:
                        ax.text(0.5, 0.5, "No finite values", transform=ax.transAxes, ha="center", va="center")
                    if row_idx == 0:
                        ax.set_title(f"{panel_title} | n={int(np.sum(mask))} | bins={n_bins}")
                    if col_idx == 0:
                        ax.set_ylabel(str(metabolite_name), fontsize=metabolite_y_fontsize)
                    if row_idx == len(plot_names) - 1:
                        ax.set_xlabel(metabolite_x_label, fontsize=metabolite_x_fontsize)
                    ax.set_xlim(scalar_min, scalar_max)
                    ax.tick_params(axis="both", labelsize=metabolite_tick_fontsize)
                    ax.grid(True, alpha=0.22)

            try:
                fig.supylabel(metabolite_y_label, fontsize=metabolite_y_fontsize)
            except Exception:
                pass
            fig.suptitle(f"Raw metabolite median profiles along {scalar_mode_label} - {self._title}{subject_title_suffix}")

            dialog = QDialog(self)
            dialog.setWindowTitle(f"Raw metabolite profiles along {scalar_mode_label}")
            layout = QVBoxLayout(dialog)
            layout.addWidget(toolbar)
            layout.addWidget(canvas, 1)
            dialog.resize(980, max(560, int(170 * len(plot_names) + 160)))
            dialog._figure = fig
            dialog._canvas = canvas
            dialog._toolbar = toolbar
            if not hasattr(self, "_metabolite_profile_dialogs"):
                self._metabolite_profile_dialogs = []
            self._metabolite_profile_dialogs.append(dialog)
            canvas.draw_idle()
            dialog.show()
            return
        msmode_coords = None
        if self._path_metric_coords is not None:
            try:
                candidate_coords = np.asarray(self._path_metric_coords, dtype=float)
                if candidate_coords.ndim == 2 and candidate_coords.shape[0] == scalar.shape[0]:
                    msmode_coords = candidate_coords
            except Exception:
                msmode_coords = None
        if msmode_coords is not None and msmode_coords.shape[1] >= 1:
            msmode1_values = np.asarray(msmode_coords[:, 0], dtype=float)
        else:
            msmode1_values = np.asarray(self._gradient1, dtype=float)
        if msmode_coords is not None and msmode_coords.shape[1] >= 2:
            msmode2_values = np.asarray(msmode_coords[:, 1], dtype=float)
        else:
            msmode2_values = np.asarray(self._x, dtype=float)

        def _axis_bins(axis_values):
            axis_arr = np.asarray(axis_values, dtype=float)
            axis_for_bins = axis_arr[panel_mask]
            axis_for_bins = axis_for_bins[np.isfinite(axis_for_bins)]
            if axis_for_bins.size < 2:
                return None, None, None
            axis_min = float(np.nanmin(axis_for_bins))
            axis_max = float(np.nanmax(axis_for_bins))
            if not np.isfinite(axis_min) or not np.isfinite(axis_max) or axis_max <= axis_min:
                return None, None, None
            edges = np.linspace(axis_min, axis_max, n_bins + 1)
            centers = 0.5 * (edges[:-1] + edges[1:])
            return edges, centers, (axis_min, axis_max)

        msmode1_edges, msmode1_centers, msmode1_limits = _axis_bins(msmode1_values)
        msmode2_edges, msmode2_centers, msmode2_limits = _axis_bins(msmode2_values)

        fig_width = 8.8 if len(panel_specs) == 1 else 12.6
        fig = Figure(figsize=(fig_width, 8.4), constrained_layout=True)
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, self)
        cmap = self._cmap if self._cmap is not None else GradientSurfaceDialog._default_cmap(self._cmap_name)
        if not callable(cmap):
            try:
                import matplotlib.cm as mpl_cm

                cmap = mpl_cm.get_cmap(str(cmap))
            except Exception:
                cmap = GradientSurfaceDialog._default_cmap("viridis")
        if not callable(cmap):
            import matplotlib.cm as mpl_cm

            cmap = mpl_cm.get_cmap("viridis")
        metabolite_palette = {
            "glx": metabolite_colors.get("glx", "#d62728"),
            "glugln": metabolite_colors.get("glx", "#d62728"),
            "glu+gln": metabolite_colors.get("glx", "#d62728"),
            "cho": metabolite_colors.get("cho", "#2ca02c"),
            "gpcpch": metabolite_colors.get("cho", "#2ca02c"),
            "gpc+pch": metabolite_colors.get("cho", "#2ca02c"),
            "naa": metabolite_colors.get("naa", "#1f77b4"),
            "naanaag": metabolite_colors.get("naa", "#1f77b4"),
            "naa+naag": metabolite_colors.get("naa", "#1f77b4"),
            "ins": metabolite_colors.get("ins", "#f2c94c"),
            "crpcr": metabolite_colors.get("crpcr", "#000000"),
            "water": metabolite_colors.get("water", "#7a7a7a"),
            "watersignal": metabolite_colors.get("water", "#7a7a7a"),
            "water_signal": metabolite_colors.get("water", "#7a7a7a"),
        }
        fallback_colors = cmap(np.linspace(0.05, 0.95, len(plot_names)))
        color_values = []
        for met_idx, name in enumerate(plot_names):
            normalized_name = str(name).strip().lower().replace(" ", "")
            color_values.append(metabolite_palette.get(normalized_name, fallback_colors[met_idx]))
        axes_grid = fig.subplots(
            4,
            len(panel_specs),
            squeeze=False,
            gridspec_kw={"height_ratios": [1.0, 0.075, 1.0, 1.0]},
        )
        tri_axes = axes_grid[0, :]
        strip_axes = axes_grid[1, :]
        msmode1_axes = axes_grid[2, :]
        msmode2_axes = axes_grid[3, :]

        def _panel_profiles(mask):
            if profiles_are_subjectwise:
                local_profiles = np.asarray(profiles[:, mask, :], dtype=float)
                mean = np.nanmean(local_profiles, axis=1, keepdims=True)
                std = np.nanstd(local_profiles, axis=1, keepdims=True)
                std[~np.isfinite(std) | (std <= 1e-12)] = 1.0
                z_profiles = (local_profiles - mean) / std
                subjectwise_profiles = _apply_zscore_correction(z_profiles)
                return subjectwise_profiles, None
            else:
                local_profiles = np.asarray(profiles[mask, :], dtype=float)
                mean = np.nanmean(local_profiles, axis=0, keepdims=True)
                std = np.nanstd(local_profiles, axis=0, keepdims=True)
                std[~np.isfinite(std) | (std <= 1e-12)] = 1.0
                z_all_metabolites = (local_profiles - mean) / std
                z_profiles = _apply_zscore_correction(z_all_metabolites)
                return None, z_profiles

        def _plot_binned_axis(
            ax,
            axis_values,
            mask,
            subjectwise_profiles,
            z_profiles,
            edges,
            centers,
            *,
            xlabel="",
            title="",
        ):
            local_axis = np.asarray(axis_values[mask], dtype=float)
            local_valid = np.isfinite(local_axis)
            if subjectwise_profiles is not None:
                local_valid &= np.any(np.isfinite(subjectwise_profiles), axis=(0, 2))
            else:
                local_valid &= np.any(np.isfinite(z_profiles), axis=1)
            if edges is None or centers is None or int(np.sum(local_valid)) < 2:
                ax.text(0.5, 0.5, "No finite values", transform=ax.transAxes, ha="center", va="center")
                if title:
                    ax.set_title(title)
                ax.set_xlabel(xlabel or metabolite_x_label, fontsize=metabolite_x_fontsize)
                ax.tick_params(axis="both", labelsize=metabolite_tick_fontsize)
                ax.grid(True, alpha=0.22)
                return
            local_axis = local_axis[local_valid]
            if subjectwise_profiles is not None:
                local_subjectwise = subjectwise_profiles[:, local_valid, :]
                local_z = None
            else:
                local_subjectwise = None
                local_z = z_profiles[local_valid, :]
            bin_indices = np.digitize(local_axis, edges, right=False) - 1
            bin_indices = np.clip(bin_indices, 0, n_bins - 1)
            binned_profiles = np.full((n_bins, len(plot_names)), np.nan, dtype=float)
            binned_low = np.full((n_bins, len(plot_names)), np.nan, dtype=float)
            binned_high = np.full((n_bins, len(plot_names)), np.nan, dtype=float)
            binned_q1 = np.full((n_bins, len(plot_names)), np.nan, dtype=float)
            binned_q3 = np.full((n_bins, len(plot_names)), np.nan, dtype=float)
            bin_counts = np.zeros(n_bins, dtype=int)
            for bin_idx in range(n_bins):
                in_bin = bin_indices == bin_idx
                bin_counts[bin_idx] = int(np.sum(in_bin))
                if bin_counts[bin_idx] > 0:
                    if local_subjectwise is not None:
                        values = np.asarray(local_subjectwise[:, in_bin, :], dtype=float).reshape(-1, len(plot_names))
                    else:
                        values = np.asarray(local_z[in_bin, :], dtype=float)
                    (
                        binned_profiles[bin_idx, :],
                        binned_low[bin_idx, :],
                        binned_high[bin_idx, :],
                        binned_q1[bin_idx, :],
                        binned_q3[bin_idx, :],
                    ) = _median_interval(values, axis=0)
            for met_idx, name in enumerate(plot_names):
                valid_bins = np.isfinite(binned_profiles[:, met_idx])
                band_mask = valid_bins & np.isfinite(binned_low[:, met_idx]) & np.isfinite(binned_high[:, met_idx])
                if show_percentile_band and np.any(band_mask):
                    ax.fill_between(
                        centers[band_mask],
                        binned_low[band_mask, met_idx],
                        binned_high[band_mask, met_idx],
                        color=color_values[met_idx],
                        alpha=0.16,
                        linewidth=0.0,
                    )
                _draw_boxplot_bars(
                    ax,
                    centers,
                    binned_low[:, met_idx],
                    binned_q1[:, met_idx],
                    binned_q3[:, met_idx],
                    binned_high[:, met_idx],
                    color_values[met_idx],
                )
                ax.plot(
                    centers[valid_bins],
                    binned_profiles[valid_bins, met_idx],
                    marker="o",
                    markersize=4.0,
                    linewidth=metabolite_linewidth,
                    color=color_values[met_idx],
                    label=str(name),
                )
            ax.axhline(0.0, color="0.35", linewidth=0.8, alpha=0.6)
            if title:
                ax.set_title(title)
            ax.set_xlabel(xlabel or metabolite_x_label, fontsize=metabolite_x_fontsize)
            ax.tick_params(axis="both", labelsize=metabolite_tick_fontsize)
            ax.grid(True, alpha=0.22)

        for col_idx, (panel_title, mask) in enumerate(panel_specs):
            subjectwise_profiles, z_profiles = _panel_profiles(mask)
            tri_ax = tri_axes[col_idx]
            strip_ax = strip_axes[col_idx]
            msmode1_ax = msmode1_axes[col_idx]
            msmode2_ax = msmode2_axes[col_idx]
            _plot_binned_axis(
                tri_ax,
                scalar,
                mask,
                subjectwise_profiles,
                z_profiles,
                bin_edges,
                bin_centers,
                title=f"{panel_title} | n={int(np.sum(mask))} | bins={n_bins}",
            )
            strip_ax.imshow(
                strip_image,
                aspect="auto",
                extent=(scalar_min, scalar_max, 0.0, 1.0),
                origin="lower",
            )
            strip_ax.set_yticks([])
            strip_ax.set_xlim(scalar_min, scalar_max)
            strip_ax.set_xlabel(metabolite_x_label, fontsize=metabolite_x_fontsize)
            strip_ax.tick_params(axis="both", labelsize=metabolite_tick_fontsize)
            for spine in strip_ax.spines.values():
                spine.set_visible(False)
            _plot_binned_axis(
                msmode1_ax,
                msmode1_values,
                mask,
                subjectwise_profiles,
                z_profiles,
                msmode1_edges,
                msmode1_centers,
                xlabel=metabolite_x_label,
            )
            if msmode1_limits is not None:
                msmode1_ax.set_xlim(*msmode1_limits)
            _plot_binned_axis(
                msmode2_ax,
                msmode2_values,
                mask,
                subjectwise_profiles,
                z_profiles,
                msmode2_edges,
                msmode2_centers,
                xlabel=metabolite_x_label,
            )
            if msmode2_limits is not None:
                msmode2_ax.set_xlim(*msmode2_limits)
        ylabel = _zscore_ylabel("Median z-scored metabolite signal")
        title_prefix = _zscore_title_prefix(
            "Binned median z-scored metabolite profiles",
            "Binned median metabolite profiles minus CrPCr",
            "Binned median metabolite profiles minus metabolite mean",
        )
        tri_axes[0].set_ylabel(metabolite_y_label or ylabel, fontsize=metabolite_y_fontsize)
        msmode1_axes[0].set_ylabel(metabolite_y_label or ylabel, fontsize=metabolite_y_fontsize)
        msmode2_axes[0].set_ylabel(metabolite_y_label or ylabel, fontsize=metabolite_y_fontsize)
        tri_axes[-1].legend(loc="best", fontsize=8, frameon=False)
        fig.suptitle(f"{title_prefix} along {scalar_mode_label} and MSMode axes - {self._title}{subject_title_suffix}")

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Metabolite profiles along {scalar_mode_label}")
        layout = QVBoxLayout(dialog)
        layout.addWidget(toolbar)
        layout.addWidget(canvas, 1)
        dialog.resize(980, 860)
        dialog._figure = fig
        dialog._canvas = canvas
        dialog._toolbar = toolbar
        if not hasattr(self, "_metabolite_profile_dialogs"):
            self._metabolite_profile_dialogs = []
        self._metabolite_profile_dialogs.append(dialog)
        canvas.draw_idle()
        dialog.show()

    def _on_project_paths_clicked(self):
        if self._project_paths_callback is None or not isinstance(self._project_paths_payload, dict):
            return
        payload = self._selected_only_project_paths_payload()
        has_projectable_path = False
        for group_payload in list(payload.get("group_paths", [])):
            if len(self._selected_ctx_path_nodes(group_payload)) >= 2:
                has_projectable_path = True
                break
            if len(list(group_payload.get("subc_optimal_path", []))) >= 2:
                has_projectable_path = True
                break
        if not has_projectable_path and len(list(payload.get("optimal_full_path", []))) < 2:
            return
        self._project_paths_callback(payload)

    def _on_screenshot_paths_clicked(self):
        if self._project_paths_callback is None or not isinstance(self._project_paths_payload, dict):
            return
        payload = self._selected_only_project_paths_payload()
        payload["glassbrain_screenshot"] = True
        self._project_paths_callback(payload)

    def _selected_only_project_paths_payload(self):
        payload = dict(self._project_paths_payload or {})
        payload["show_all_ordered_paths"] = False
        selected_paths = []
        first_selected_ctx_path = None

        def _finite_energy_or_nan(value):
            try:
                energy = float(value)
            except Exception:
                return float("nan")
            return energy if np.isfinite(energy) else float("nan")

        for group_payload in list(payload.get("group_paths", [])):
            if not isinstance(group_payload, dict):
                continue
            group_copy = dict(group_payload)
            selected_path = self._selected_ctx_path_nodes(group_copy)
            subc_path = [int(node) for node in list(group_copy.get("subc_optimal_path", []))]
            if len(selected_path) >= 2:
                selected_path = [int(node) for node in selected_path]
                selected_energy = self._selected_ctx_path_energy(group_copy)
                group_copy["selected_ctx_path"] = list(selected_path)
                group_copy["optimal_full_path"] = list(selected_path)
                group_copy["all_full_paths"] = [list(selected_path)]
                group_copy["ctx_path_energies"] = [_finite_energy_or_nan(selected_energy)]
                group_copy["selected_ctx_path_energy"] = selected_energy
                group_copy["full_path_count"] = 1
                group_copy["ctx_path_count"] = 1
                if first_selected_ctx_path is None:
                    first_selected_ctx_path = list(selected_path)
            if len(subc_path) >= 2:
                subc_energy = group_copy.get("subc_optimal_path_energy")
                group_copy["subc_paths"] = [list(subc_path)]
                group_copy["subc_path_energies"] = [_finite_energy_or_nan(subc_energy)]
                group_copy["subc_path_count"] = 1
            else:
                group_copy["subc_paths"] = []
                group_copy["subc_path_energies"] = []
                group_copy["subc_path_count"] = 0
            if len(selected_path) >= 2 or len(subc_path) >= 2:
                selected_paths.append(group_copy)
        payload["group_paths"] = selected_paths
        payload["optimal_full_path"] = (
            list(first_selected_ctx_path)
            if first_selected_ctx_path is not None
            else list(payload.get("optimal_full_path", []))
        )
        return payload

    @staticmethod
    def _normalize_fibrenet_layout(value):
        text = str(value or "diffusion").strip().lower()
        if text in {"spiral", "path_spiral", "sequence", "sequence_spiral"}:
            return "spiral"
        return "diffusion"

    def _selected_fibrenet_layout(self):
        combo = getattr(self, "fibrenet_layout_combo", None)
        if combo is not None:
            try:
                self._fibrenet_layout = self._normalize_fibrenet_layout(combo.currentData())
            except Exception:
                pass
        self._fibrenet_layout = self._normalize_fibrenet_layout(
            getattr(self, "_fibrenet_layout", "diffusion")
        )
        return self._fibrenet_layout

    def _on_fibrenet_layout_changed(self, _index):
        layout = self._selected_fibrenet_layout()
        if isinstance(self._project_paths_payload, dict):
            self._project_paths_payload["fibrenet_layout"] = layout

    def _on_fibrenet_paths_clicked(self):
        if self._project_paths_callback is None or not isinstance(self._project_paths_payload, dict):
            return
        payload = dict(self._project_paths_payload)
        payload["fibrenet_projection"] = True
        payload["fibrenet_layout"] = self._selected_fibrenet_layout()
        self._project_paths_callback(payload)

    def _path_export_node(self, node_index):
        idx = int(node_index)
        node_id = str(self._point_ids[idx]) if 0 <= idx < self._point_ids.shape[0] else str(idx)
        node_name = str(self._point_labels[idx]) if 0 <= idx < self._point_labels.shape[0] else f"Point {idx + 1}"
        if 0 <= idx < self._display_coords.shape[0]:
            x_coord = float(self._display_coords[idx, 0])
            y_coord = float(self._display_coords[idx, 1])
        else:
            x_coord = float("nan")
            y_coord = float("nan")
        return {
            "scatter_index": idx,
            "node_label": node_id,
            "node_name": node_name,
            "x_coord": x_coord,
            "y_coord": y_coord,
        }

    def _invalidate_generated_paths(self):
        self._project_paths_payload = None
        self._selected_ctx_path_indices = {"lh": 0, "rh": 0, "all": 0}
        if hasattr(self, "left_path_combo") and hasattr(self, "right_path_combo"):
            self._populate_path_selection_combos()

    def _clear_free_energy_payload(self):
        if not isinstance(self._project_paths_payload, dict):
            return
        self._project_paths_payload.pop("free_energy_payload", None)
        self._project_paths_payload.pop("energy_width_scaling", None)

    def _on_free_energy_lambda_changed(self, _value):
        if not isinstance(self._project_paths_payload, dict):
            return
        self._clear_free_energy_payload()
        self._sync_proximity_controls()
        self._render()

    def _on_generate_paths_clicked(self):
        if self._is_3d or not self._use_triangular_rgb:
            return
        self._selected_ctx_path_indices = {"lh": 0, "rh": 0, "all": 0}
        x_plot, y_plot = self._rotate_points(self._x, self._y, self._rotation_preset)
        rgb_x_plot, rgb_y_plot = self._rotated_rgb_points()
        triangle_model = self._rgb_model(
            rgb_x_plot,
            rgb_y_plot,
            self._triangular_color_order,
            fit_mode=self._rgb_fit_mode,
        )
        point_colors = self._rgb_colors_from_model(rgb_x_plot, rgb_y_plot, triangle_model)
        path_edge_pairs = np.asarray(self._edge_pairs, dtype=int)
        path_edge_distances = np.asarray(self._path_metric_edge_distances, dtype=float)
        payload = self._build_triangular_anchor_paths_payload(
            x_plot,
            y_plot,
            triangle_model,
            self._path_channel_order,
            point_colors,
            path_edge_pairs,
            path_edge_distances,
            anchor_x_plot=rgb_x_plot,
            anchor_y_plot=rgb_y_plot,
        )
        if isinstance(payload, dict):
            payload["point_colors"] = np.asarray(point_colors, dtype=float).tolist()
            payload["show_all_ordered_paths"] = bool(self._show_all_ordered_paths)
            payload["edge_linewidth"] = float(self._edge_linewidth)
            payload["width_scaling_mode"] = self._path_width_scaling_mode
            payload["width_scaling_strength"] = float(self._path_width_scaling_strength)
            payload["edge_pairs"] = np.asarray(self._edge_pairs, dtype=int).tolist()
            payload["node_count"] = int(self._x.size)
            payload["fibrenet_layout"] = self._selected_fibrenet_layout()
            payload["endpoint_anchor_signature"] = self._payload_endpoint_anchor_signature(payload)
        self._project_paths_payload = payload
        self._populate_path_selection_combos()
        self._sync_proximity_controls()
        self._render(preserve_view=True)

    @staticmethod
    def _normalize_free_energy_lambda(value):
        try:
            lam = float(value)
        except Exception:
            lam = 1.0
        return max(1e-6, lam)

    @staticmethod
    def _path_directionality_energy(
        coords,
        path_nodes,
        reference_unit_vector,
    ):
        nodes = np.asarray([int(node) for node in list(path_nodes or [])], dtype=int)
        points = np.asarray(coords, dtype=float)
        ref = np.asarray(reference_unit_vector, dtype=float).reshape(-1)
        if nodes.size < 2 or points.ndim != 2 or points.shape[1] != 2 or ref.shape != (2,):
            return None
        if np.any((nodes < 0) | (nodes >= points.shape[0])):
            return None
        if not np.all(np.isfinite(points[nodes, :])) or not np.all(np.isfinite(ref)):
            return None
        ref_norm = float(np.linalg.norm(ref))
        if ref_norm <= 0.0:
            return None
        ref_unit = ref / ref_norm
        steps = np.diff(points[nodes, :], axis=0)
        if steps.shape[0] == 0:
            return None
        step_norms = np.linalg.norm(steps, axis=1)
        valid = np.isfinite(step_norms) & (step_norms > 1e-12)
        if not np.any(valid):
            return None
        valid_steps = steps[valid, :]
        valid_step_norms = step_norms[valid]
        alignment = valid_steps @ ref_unit
        direction_penalties = valid_step_norms - alignment
        return float(np.sum(direction_penalties))

    @staticmethod
    def _stable_free_energy(energies, lam):
        values = np.asarray(energies, dtype=float).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return float("nan")
        lam = GradientScatterDialog._normalize_free_energy_lambda(lam)
        scaled = -lam * values
        max_scaled = float(np.max(scaled))
        return float(-(1.0 / lam) * (max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))))

    @staticmethod
    def _path_segment_count(path_nodes) -> int:
        nodes = [int(node) for node in list(path_nodes or [])]
        return max(len(nodes) - 1, 0)

    @staticmethod
    def _normalize_path_energy_by_segments(energy, path_nodes):
        try:
            value = float(energy)
        except Exception:
            return None
        if not np.isfinite(value):
            return None
        n_segments = GradientScatterDialog._path_segment_count(path_nodes)
        if n_segments <= 0:
            return None
        return float(value) / float(n_segments)

    def _reference_unit_vector(self, start_index, end_index):
        coords = np.asarray(self._display_coords, dtype=float)
        start = int(start_index)
        end = int(end_index)
        if (
            coords.ndim != 2
            or coords.shape[1] != 2
            or start < 0
            or end < 0
            or start >= coords.shape[0]
            or end >= coords.shape[0]
        ):
            return None
        vector = np.asarray(coords[end, :] - coords[start, :], dtype=float)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 0.0:
            return None
        return vector / norm

    def _ctx_full_path_energy(
        self,
        coords,
        full_path_nodes,
        anchors,
        channel_order,
    ):
        order = [str(channel) for channel in list(channel_order or []) if str(channel) in anchors]
        if len(order) < 2:
            return None
        if len(order) == 2:
            start_anchor = int(anchors[order[0]])
            end_anchor = int(anchors[order[1]])
            ref_unit = self._reference_unit_vector(start_anchor, end_anchor)
            if ref_unit is None:
                return None
            return self._path_directionality_energy(
                coords,
                full_path_nodes,
                ref_unit,
            )

        nodes = [int(node) for node in list(full_path_nodes or [])]
        if len(nodes) < 2:
            return None
        start_anchor = int(anchors[order[0]])
        middle_anchor = int(anchors[order[1]])
        end_anchor = int(anchors[order[2]])
        if nodes[0] != start_anchor or nodes[-1] != end_anchor:
            return None
        try:
            split_index = next(
                idx
                for idx, node in enumerate(nodes)
                if int(node) == middle_anchor and idx > 0 and idx < len(nodes) - 1
            )
        except StopIteration:
            return None

        first_segment = nodes[: split_index + 1]
        second_segment = nodes[split_index:]
        ref_first = self._reference_unit_vector(start_anchor, middle_anchor)
        ref_second = self._reference_unit_vector(middle_anchor, end_anchor)
        if ref_first is None or ref_second is None:
            return None
        energy_first = self._path_directionality_energy(
            coords,
            first_segment,
            ref_first,
        )
        energy_second = self._path_directionality_energy(
            coords,
            second_segment,
            ref_second,
        )
        if energy_first is None or energy_second is None:
            return None
        return float(energy_first + energy_second)

    def _compute_free_energy_payload(self, lam):
        if not isinstance(self._project_paths_payload, dict):
            return None
        lam = self._normalize_free_energy_lambda(lam)
        normalize_by_segments = bool(self._normalize_free_energy_by_segments)
        channel_order = str(self._project_paths_payload.get("channel_order", self._triangular_color_order or "")).strip()
        if len(channel_order) < 2:
            return None
        coords = np.asarray(self._display_coords, dtype=float)
        groups = []
        for group_payload in list(self._project_paths_payload.get("group_paths", [])):
            group = dict(group_payload or {})
            anchors = {str(key): int(value) for key, value in dict(group.get("anchors", {})).items()}
            families = []
            order = [str(channel) for channel in channel_order if str(channel) in anchors]
            if len(order) >= 2:
                ctx_records = []
                for full_path in list(group.get("all_full_paths", [])):
                    energy = self._ctx_full_path_energy(coords, full_path, anchors, order)
                    if energy is not None and np.isfinite(energy):
                        path_nodes = [int(node) for node in list(full_path or [])]
                        raw_energy = float(energy)
                        segment_count = self._path_segment_count(path_nodes)
                        if normalize_by_segments:
                            normalized_energy = self._normalize_path_energy_by_segments(raw_energy, path_nodes)
                            if normalized_energy is None:
                                continue
                            energy = normalized_energy
                        ctx_records.append(
                            {
                                "nodes": path_nodes,
                                "energy": float(energy),
                                "raw_energy": float(raw_energy),
                                "segment_count": int(segment_count),
                                "energy_normalized_by_segments": bool(normalize_by_segments),
                            }
                        )
                if ctx_records:
                    ctx_energies = [float(record["energy"]) for record in ctx_records]
                    ctx_colors = []
                    for record in list(group.get("optimal_segments", [])):
                        color = np.asarray(dict(record).get("color", (0.2, 0.2, 0.2)), dtype=float).reshape(-1)
                        if color.shape == (3,):
                            ctx_colors.append(color)
                    if ctx_colors:
                        ctx_color = np.mean(np.asarray(ctx_colors, dtype=float), axis=0)
                    else:
                        ctx_color = np.asarray((0.2, 0.2, 0.2), dtype=float)
                    reference_vectors = []
                    segment_labels = []
                    for idx in range(len(order) - 1):
                        ref_unit = self._reference_unit_vector(anchors[order[idx]], anchors[order[idx + 1]])
                        if ref_unit is not None:
                            reference_vectors.append([float(value) for value in np.asarray(ref_unit, dtype=float).tolist()])
                        segment_labels.append(f"{order[idx]}{order[idx + 1]}")
                    families.append(
                        {
                            "label": "CTX",
                            "family_type": "ctx",
                            "segment_labels": segment_labels,
                            "reference_vectors": reference_vectors,
                            "energies": [float(value) for value in ctx_energies],
                            "path_energies": ctx_records,
                            "free_energy": self._stable_free_energy(ctx_energies, lam),
                            "energy_normalized_by_segments": bool(normalize_by_segments),
                            "color": [float(value) for value in np.asarray(ctx_color, dtype=float).tolist()],
                            "n_paths": int(len(ctx_energies)),
                        }
                    )

            if len(channel_order) >= 2 and channel_order[1] in anchors and group.get("subc_anchor") is not None:
                start_index = int(anchors[channel_order[1]])
                subc_index = int(group.get("subc_anchor"))
                ref_unit = self._reference_unit_vector(start_index, subc_index)
                if ref_unit is not None:
                    subc_records = []
                    for path_nodes in list(group.get("subc_paths", [])):
                        energy = self._path_directionality_energy(
                            coords,
                            path_nodes,
                            ref_unit,
                        )
                        if energy is not None and np.isfinite(energy):
                            nodes = [int(node) for node in list(path_nodes or [])]
                            raw_energy = float(energy)
                            segment_count = self._path_segment_count(nodes)
                            if normalize_by_segments:
                                normalized_energy = self._normalize_path_energy_by_segments(raw_energy, nodes)
                                if normalized_energy is None:
                                    continue
                                energy = normalized_energy
                            subc_records.append(
                                {
                                    "nodes": nodes,
                                    "energy": float(energy),
                                    "raw_energy": float(raw_energy),
                                    "segment_count": int(segment_count),
                                    "energy_normalized_by_segments": bool(normalize_by_segments),
                                }
                            )
                    if subc_records:
                        energies = [float(record["energy"]) for record in subc_records]
                        subc_color = np.asarray(group.get("subc_color", (0.1, 0.1, 0.1)), dtype=float).reshape(-1)
                        if subc_color.shape != (3,):
                            subc_color = np.asarray((0.1, 0.1, 0.1), dtype=float)
                        subc_name = ""
                        if 0 <= subc_index < self._point_labels.shape[0]:
                            subc_name = str(self._point_labels[subc_index]).strip()
                        families.append(
                            {
                                "label": "SUBC",
                                "family_type": "subc",
                                "segment_labels": [f"{channel_order[1]}->{subc_name or 'thal'}"],
                                "reference_vector": [float(value) for value in ref_unit.tolist()],
                                "energies": [float(value) for value in energies],
                                "path_energies": subc_records,
                                "free_energy": self._stable_free_energy(energies, lam),
                                "energy_normalized_by_segments": bool(normalize_by_segments),
                                "color": [float(value) for value in subc_color.tolist()],
                                "n_paths": int(len(energies)),
                            }
                        )

            if families:
                groups.append(
                    {
                        "group": str(group.get("group", "all")),
                        "families": families,
                    }
                )

        if not groups:
            return None
        return {
            "title": self._title,
            "lambda": float(lam),
            "rotation": str(self._rotation_preset),
            "x_axis_label": self._rotate_axis_labels(self._x_label, self._y_label, self._rotation_preset)[0],
            "y_axis_label": self._rotate_axis_labels(self._x_label, self._y_label, self._rotation_preset)[1],
            "normalize_energy_by_segments": bool(normalize_by_segments),
            "endpoint_anchor_signature": self._payload_endpoint_anchor_signature(self._project_paths_payload),
            "groups": groups,
        }

    def _apply_free_energy_scaling(self, free_energy_payload):
        if not isinstance(self._project_paths_payload, dict) or not isinstance(free_energy_payload, dict):
            return

        family_energy_ranges = {}
        for family_type in ("ctx", "subc"):
            energies = []
            for group in list(free_energy_payload.get("groups", [])):
                for family in list(dict(group).get("families", [])):
                    if str(dict(family).get("family_type", "")).strip().lower() != family_type:
                        continue
                    for record in list(dict(family).get("path_energies", [])):
                        try:
                            value = float(dict(record).get("energy"))
                        except Exception:
                            continue
                        if np.isfinite(value):
                            energies.append(value)
            if energies:
                energy_values = np.asarray(energies, dtype=float)
                family_energy_ranges[family_type] = {
                    "min": float(np.min(energy_values)),
                    "max": float(np.max(energy_values)),
                }

        group_payloads = {}
        for group in list(self._project_paths_payload.get("group_paths", [])):
            if isinstance(group, dict):
                group_payloads[str(group.get("group", "all")).strip().lower()] = group
        metric_coords = (
            np.asarray(self._path_metric_coords, dtype=float)
            if self._path_metric_coords is not None
            else np.asarray(self._display_coords, dtype=float)
        )
        for free_group in list(free_energy_payload.get("groups", [])):
            free_group_dict = dict(free_group or {})
            group_name = str(free_group_dict.get("group", "all")).strip().lower()
            target_group = group_payloads.get(group_name)
            if target_group is None:
                continue
            ctx_lookup = {}
            subc_lookup = {}
            for family in list(free_group_dict.get("families", [])):
                family_dict = dict(family or {})
                family_type = str(family_dict.get("family_type", "")).strip().lower()
                for record in list(family_dict.get("path_energies", [])):
                    record_dict = dict(record or {})
                    path_key = tuple(int(node) for node in list(record_dict.get("nodes", [])))
                    try:
                        energy = float(record_dict.get("energy"))
                    except Exception:
                        continue
                    if family_type == "ctx":
                        ctx_lookup[path_key] = energy
                    elif family_type == "subc":
                        subc_lookup[path_key] = energy
            target_group["ctx_path_energies"] = [
                float(ctx_lookup.get(tuple(int(node) for node in list(path or [])), float("nan")))
                for path in list(target_group.get("all_full_paths", []))
            ]
            target_group["subc_path_energies"] = [
                float(subc_lookup.get(tuple(int(node) for node in list(path or [])), float("nan")))
                for path in list(target_group.get("subc_paths", []))
            ]
            optimal_ctx_key = tuple(int(node) for node in list(target_group.get("optimal_full_path", [])))
            optimal_subc_key = tuple(int(node) for node in list(target_group.get("subc_optimal_path", [])))
            target_group["ctx_optimal_path_energy"] = (
                float(ctx_lookup[optimal_ctx_key]) if optimal_ctx_key in ctx_lookup else None
            )
            target_group["subc_optimal_path_energy"] = (
                float(subc_lookup[optimal_subc_key]) if optimal_subc_key in subc_lookup else None
            )
            target_group["all_full_paths"], target_group["ctx_path_energies"] = self._sort_paths_by_energy(
                target_group.get("all_full_paths", []),
                target_group.get("ctx_path_energies", []),
                metric_coords=metric_coords,
            )
            if target_group["all_full_paths"]:
                target_group["optimal_full_path"] = list(target_group["all_full_paths"][0])
                first_energy = target_group["ctx_path_energies"][0] if target_group["ctx_path_energies"] else float("nan")
                target_group["ctx_optimal_path_energy"] = (
                    float(first_energy) if np.isfinite(first_energy) else None
                )
            target_group["subc_paths"], target_group["subc_path_energies"] = self._sort_paths_by_energy(
                target_group.get("subc_paths", []),
                target_group.get("subc_path_energies", []),
                metric_coords=metric_coords,
            )
            if target_group["subc_paths"]:
                target_group["subc_optimal_path"] = list(target_group["subc_paths"][0])
                first_energy = target_group["subc_path_energies"][0] if target_group["subc_path_energies"] else float("nan")
                target_group["subc_optimal_path_energy"] = (
                    float(first_energy) if np.isfinite(first_energy) else None
                )
            self._selected_ctx_path_indices[group_name] = 0
            self._apply_selected_ctx_path_to_group(target_group, 0)

        self._project_paths_payload["energy_width_scaling"] = family_energy_ranges
        self._project_paths_payload["free_energy_payload"] = dict(free_energy_payload)
        self._project_paths_payload["edge_linewidth"] = float(self._edge_linewidth)
        self._project_paths_payload["width_scaling_mode"] = self._path_width_scaling_mode
        self._project_paths_payload["width_scaling_strength"] = float(self._path_width_scaling_strength)

    def _on_compute_free_energy_clicked(self):
        if not isinstance(self._project_paths_payload, dict) or not self._generated_paths_match_current_endpoints():
            self._on_generate_paths_clicked()
        if not isinstance(self._project_paths_payload, dict):
            return
        payload = self._compute_free_energy_payload(self.free_energy_lambda_spin.value())
        if not isinstance(payload, dict):
            return
        self._apply_free_energy_scaling(payload)
        self._populate_path_selection_combos()
        self._sync_proximity_controls()
        self._render(preserve_view=True)
        dialog = GradientFreeEnergyDialog(
            payload,
            parent=self,
            theme_name=self._theme_name,
        )
        self._free_energy_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    @staticmethod
    def _safe_json_default(value):
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.ndarray,)):
            return value.tolist()
        try:
            if value is not None and np.isnan(value):
                return None
        except Exception:
            pass
        return str(value)

    @staticmethod
    def _safe_name_fragment(text):
        token = str(text or "").strip()
        if not token:
            return "gradient"
        cleaned = []
        for ch in token:
            if ch.isalnum() or ch in {"-", "_"}:
                cleaned.append(ch)
            else:
                cleaned.append("_")
        out = "".join(cleaned).strip("_")
        while "__" in out:
            out = out.replace("__", "_")
        return out or "gradient"

    @staticmethod
    def _derive_parc_scheme(path_text):
        text = str(path_text or "").strip()
        if not text:
            return ""
        path = Path(text)
        name = path.name
        stem = name[:-7] if name.endswith(".nii.gz") else path.stem
        base = stem
        scale = ""
        if "-" in stem:
            left, right = stem.rsplit("-", 1)
            if right.isdigit():
                base = left
                scale = f"_scale{right}"
        filtered = []
        for ch in base:
            if ch.isalnum():
                filtered.append(ch)
        return "".join(filtered) + scale

    def _numeric_point_ids(self):
        values = []
        for value in self._point_ids.tolist():
            try:
                values.append(int(str(value).strip()))
            except Exception:
                return np.asarray(self._point_ids, dtype=object)
        return np.asarray(values, dtype=int)

    def _gradient_component_arrays_for_export(self):
        metadata = dict(self._export_metadata or {})
        gradient1 = np.asarray(metadata.get("gradient1_values", self._gradient1), dtype=float).reshape(-1)
        if gradient1.shape[0] != self._x.shape[0]:
            gradient1 = np.asarray(self._gradient1, dtype=float).reshape(-1)

        gradient2_raw = metadata.get("gradient2_values", None)
        if gradient2_raw is not None:
            gradient2 = np.asarray(gradient2_raw, dtype=float).reshape(-1)
        else:
            gradient2 = np.full(self._x.shape, np.nan, dtype=float)
            x_text = str(self._x_label or "").strip().lower()
            y_text = str(self._y_label or "").strip().lower()
            if x_text == "gradient 2":
                gradient2 = np.asarray(self._x, dtype=float).reshape(-1)
            elif y_text == "gradient 2":
                gradient2 = np.asarray(self._y, dtype=float).reshape(-1)
        if gradient2.shape[0] != self._x.shape[0]:
            gradient2 = np.full(self._x.shape, np.nan, dtype=float)
        return gradient1, gradient2

    def _batch_export_node_payload(self, node_index, gradient1_values, gradient2_values):
        idx = int(node_index)
        node_label = str(self._point_ids[idx]) if 0 <= idx < self._point_ids.shape[0] else str(idx)
        node_name = str(self._point_labels[idx]) if 0 <= idx < self._point_labels.shape[0] else f"Point {idx + 1}"
        gradient1_coord = float(gradient1_values[idx]) if 0 <= idx < gradient1_values.shape[0] else float("nan")
        gradient2_coord = float(gradient2_values[idx]) if 0 <= idx < gradient2_values.shape[0] else float("nan")
        scatter_x = float(self._display_coords[idx, 0]) if 0 <= idx < self._display_coords.shape[0] else float("nan")
        scatter_y = float(self._display_coords[idx, 1]) if 0 <= idx < self._display_coords.shape[0] else float("nan")
        return {
            "node_index": idx,
            "node_label": node_label,
            "node_name": node_name,
            "gradient1_coord": gradient1_coord,
            "gradient2_coord": gradient2_coord,
            "scatter_x": scatter_x,
            "scatter_y": scatter_y,
        }

    def _batch_export_path_record(self, path_nodes, energy, gradient1_values, gradient2_values, **extra):
        nodes = [int(node) for node in list(path_nodes or [])]
        record = {
            "nodes": nodes,
            "node_labels": [str(self._point_ids[idx]) for idx in nodes],
            "node_names": [str(self._point_labels[idx]) for idx in nodes],
            "gradient1_coords": [float(gradient1_values[idx]) for idx in nodes],
            "gradient2_coords": [float(gradient2_values[idx]) for idx in nodes],
            "scatter_coords": [
                [float(self._display_coords[idx, 0]), float(self._display_coords[idx, 1])]
                for idx in nodes
            ],
            "energy": float(energy) if energy is not None and np.isfinite(float(energy)) else float("nan"),
        }
        record.update(extra)
        return record

    def _free_energy_export_payload(self):
        if not isinstance(self._project_paths_payload, dict):
            return None
        free_energy_payload = self._project_paths_payload.get("free_energy_payload")
        if not isinstance(free_energy_payload, dict):
            return None

        metadata = dict(self._export_metadata or {})
        gradient1_values, gradient2_values = self._gradient_component_arrays_for_export()
        channel_order = str(self._project_paths_payload.get("channel_order", self._triangular_color_order or "")).strip()
        color_order = str(self._project_paths_payload.get("color_order", self._triangular_color_order or "")).strip()
        normalize_by_segments = bool(free_energy_payload.get("normalize_energy_by_segments", False))
        path_group_lookup = {
            str(group.get("group", "all")).strip().lower(): dict(group)
            for group in list(self._project_paths_payload.get("group_paths", []))
            if isinstance(group, dict)
        }
        group_exports = []
        fixed_endpoints = {}

        for free_group in list(free_energy_payload.get("groups", [])):
            free_group_dict = dict(free_group or {})
            group_name = str(free_group_dict.get("group", "all")).strip().lower()
            path_group = path_group_lookup.get(group_name)
            if path_group is None:
                continue
            anchors = {str(key): int(value) for key, value in dict(path_group.get("anchors", {})).items()}
            group_record = {
                "group": group_name,
                "color_order": str(color_order),
                "path_order": str(channel_order),
                "anchors": {
                    channel: self._batch_export_node_payload(index, gradient1_values, gradient2_values)
                    for channel, index in anchors.items()
                },
                "subc_anchor": (
                    self._batch_export_node_payload(int(path_group.get("subc_anchor")), gradient1_values, gradient2_values)
                    if path_group.get("subc_anchor") is not None
                    else None
                ),
                "ctx_segment_labels": [
                    f"{channel_order[idx]}{channel_order[idx + 1]}"
                    for idx in range(max(0, len(channel_order) - 1))
                ],
                "ctx_reference_vectors": [],
                "subc_reference_vector": None,
                "ctx_paths": [],
                "ctx_optimal_path": None,
                "ctx_path_count": 0,
                "ctx_free_energy": float("nan"),
                "subc_paths": [],
                "subc_optimal_path": None,
                "subc_path_count": 0,
                "subc_free_energy": float("nan"),
                "normalize_energy_by_segments": bool(normalize_by_segments),
            }
            fixed_endpoints[group_name] = {
                "path_order": str(channel_order),
                "anchors": {
                    channel: {
                        "node_index": int(node.get("node_index", -1)),
                        "node_label": str(node.get("node_label", "")),
                        "node_name": str(node.get("node_name", "")),
                    }
                    for channel, node in group_record["anchors"].items()
                },
                "subc_anchor": (
                    {
                        "node_index": int(group_record["subc_anchor"].get("node_index", -1)),
                        "node_label": str(group_record["subc_anchor"].get("node_label", "")),
                        "node_name": str(group_record["subc_anchor"].get("node_name", "")),
                    }
                    if isinstance(group_record.get("subc_anchor"), dict)
                    else None
                ),
            }

            ctx_lookup = {}
            subc_lookup = {}
            for family in list(free_group_dict.get("families", [])):
                family_dict = dict(family or {})
                family_type = str(family_dict.get("family_type", "")).strip().lower()
                if family_type == "ctx":
                    group_record["ctx_reference_vectors"] = [
                        [float(value) for value in np.asarray(vector, dtype=float).reshape(-1).tolist()]
                        for vector in list(family_dict.get("reference_vectors", []))
                    ]
                    group_record["ctx_free_energy"] = float(family_dict.get("free_energy", float("nan")))
                    for record in list(family_dict.get("path_energies", [])):
                        record_dict = dict(record or {})
                        nodes = [int(node) for node in list(record_dict.get("nodes", []))]
                        if len(nodes) < 2:
                            continue
                        energy = record_dict.get("energy")
                        ctx_lookup[tuple(nodes)] = float(energy) if energy is not None else float("nan")
                        path_extra = {
                            "raw_energy": float(record_dict.get("raw_energy", energy))
                            if record_dict.get("raw_energy", energy) is not None
                            else float("nan"),
                            "segment_count": int(record_dict.get("segment_count", self._path_segment_count(nodes))),
                            "energy_normalized_by_segments": bool(
                                record_dict.get("energy_normalized_by_segments", normalize_by_segments)
                            ),
                        }
                        group_record["ctx_paths"].append(
                            self._batch_export_path_record(
                                nodes,
                                energy,
                                gradient1_values,
                                gradient2_values,
                                family="CTX",
                                path_label=str(channel_order),
                                segment_labels=list(group_record["ctx_segment_labels"]),
                                **path_extra,
                            )
                        )
                    group_record["ctx_path_count"] = int(family_dict.get("n_paths", len(group_record["ctx_paths"])))
                elif family_type == "subc":
                    reference = family_dict.get("reference_vector")
                    if reference is not None:
                        group_record["subc_reference_vector"] = [
                            float(value) for value in np.asarray(reference, dtype=float).reshape(-1).tolist()
                        ]
                    group_record["subc_free_energy"] = float(family_dict.get("free_energy", float("nan")))
                    for record in list(family_dict.get("path_energies", [])):
                        record_dict = dict(record or {})
                        nodes = [int(node) for node in list(record_dict.get("nodes", []))]
                        if len(nodes) < 2:
                            continue
                        energy = record_dict.get("energy")
                        subc_lookup[tuple(nodes)] = float(energy) if energy is not None else float("nan")
                        path_extra = {
                            "raw_energy": float(record_dict.get("raw_energy", energy))
                            if record_dict.get("raw_energy", energy) is not None
                            else float("nan"),
                            "segment_count": int(record_dict.get("segment_count", self._path_segment_count(nodes))),
                            "energy_normalized_by_segments": bool(
                                record_dict.get("energy_normalized_by_segments", normalize_by_segments)
                            ),
                        }
                        group_record["subc_paths"].append(
                            self._batch_export_path_record(
                                nodes,
                                energy,
                                gradient1_values,
                                gradient2_values,
                                family="SUBC",
                                path_label=f"{channel_order[1]}->thal" if len(channel_order) >= 2 else "->thal",
                                segment_labels=[f"{channel_order[1]}->thal"] if len(channel_order) >= 2 else ["->thal"],
                                **path_extra,
                            )
                        )
                    group_record["subc_path_count"] = int(family_dict.get("n_paths", len(group_record["subc_paths"])))

            optimal_ctx_nodes = [int(node) for node in list(path_group.get("optimal_full_path", []))]
            if len(optimal_ctx_nodes) >= 2:
                group_record["ctx_optimal_path"] = self._batch_export_path_record(
                    optimal_ctx_nodes,
                    ctx_lookup.get(tuple(optimal_ctx_nodes), path_group.get("ctx_optimal_path_energy")),
                    gradient1_values,
                    gradient2_values,
                    family="CTX",
                    path_label=str(channel_order),
                    segment_labels=list(group_record["ctx_segment_labels"]),
                    segment_count=self._path_segment_count(optimal_ctx_nodes),
                    energy_normalized_by_segments=bool(normalize_by_segments),
                )

            optimal_subc_nodes = [int(node) for node in list(path_group.get("subc_optimal_path", []))]
            if len(optimal_subc_nodes) >= 2:
                group_record["subc_optimal_path"] = self._batch_export_path_record(
                    optimal_subc_nodes,
                    subc_lookup.get(tuple(optimal_subc_nodes), path_group.get("subc_optimal_path_energy")),
                    gradient1_values,
                    gradient2_values,
                    family="SUBC",
                    path_label=f"{channel_order[1]}->thal" if len(channel_order) >= 2 else "->thal",
                    segment_labels=[f"{channel_order[1]}->thal"] if len(channel_order) >= 2 else ["->thal"],
                    segment_count=self._path_segment_count(optimal_subc_nodes),
                    energy_normalized_by_segments=bool(normalize_by_segments),
                )

            group_exports.append(group_record)

        if not group_exports:
            return None

        subject_id = str(metadata.get("subject_id", "") or "").strip()
        session_id = str(metadata.get("session_id", "") or "").strip()
        group_name = str(metadata.get("group", "") or "").strip()
        modality = str(metadata.get("modality", "") or "").strip()
        input_path = str(metadata.get("input_path", metadata.get("source_path", "")) or "").strip()
        group_gradients_path = str(metadata.get("group_gradients_path", input_path) or "").strip()
        parc_path = str(metadata.get("parc_path", metadata.get("template_path", "")) or "").strip()
        parc_scheme = str(metadata.get("parc_scheme", "") or "").strip() or self._derive_parc_scheme(parc_path)
        adjacency_path = str(metadata.get("adjacency_path", "") or "").strip()
        covars_row = dict(metadata.get("covars_row", {}) or {})
        gradients_pair = np.asarray(
            metadata.get(
                "gradients_pair",
                np.column_stack((gradient1_values, gradient2_values)),
            ),
            dtype=float,
        )
        gradients_avg = np.asarray(metadata.get("gradients_avg", np.empty((0, 0), dtype=float)), dtype=float)

        def _group_metric(group_key, metric_key, default_value):
            normalized = str(group_key).strip().lower()
            for group in group_exports:
                if str(group.get("group", "")).strip().lower() == normalized:
                    return group.get(metric_key, default_value)
            if normalized in {"lh", "rh"}:
                for group in group_exports:
                    if str(group.get("group", "")).strip().lower() == "all":
                        return group.get(metric_key, default_value)
            return default_value

        ctx_path_count_lh = int(_group_metric("lh", "ctx_path_count", 0))
        ctx_path_count_rh = int(_group_metric("rh", "ctx_path_count", 0))
        subc_path_count_lh = int(_group_metric("lh", "subc_path_count", 0))
        subc_path_count_rh = int(_group_metric("rh", "subc_path_count", 0))
        ctx_free_energy_lh = float(_group_metric("lh", "ctx_free_energy", float("nan")))
        ctx_free_energy_rh = float(_group_metric("rh", "ctx_free_energy", float("nan")))
        subc_free_energy_lh = float(_group_metric("lh", "subc_free_energy", float("nan")))
        subc_free_energy_rh = float(_group_metric("rh", "subc_free_energy", float("nan")))
        fixed_endpoint_file = str(getattr(self, "_loaded_fixed_endpoint_file", "") or "").strip()
        fixed_endpoint_source = (
            str(getattr(self, "_loaded_fixed_endpoint_source", "") or "").strip()
            if fixed_endpoint_file
            else f"gui_{str(self._endpoint_selection_mode or 'adaptive')}"
        )

        summary = {
            "subject_id": subject_id,
            "session_id": session_id,
            "group": group_name,
            "modality": modality,
            "input_path": input_path,
            "group_gradients_path": group_gradients_path,
            "parc_scheme": parc_scheme,
            "parc_path": parc_path,
            "adjacency_path": adjacency_path,
            "lambda_value": float(free_energy_payload.get("lambda", self.free_energy_lambda_spin.value())),
            "color_order": str(color_order),
            "path_order_override": str(channel_order),
            "axes": {
                "x": str(free_energy_payload.get("x_axis_label", self._x_label)),
                "y": str(free_energy_payload.get("y_axis_label", self._y_label)),
            },
            "normalize_energy_by_segments": bool(normalize_by_segments),
            "endpoint_selection_mode": str(self._endpoint_selection_mode or "adaptive"),
            "ctx_path_count_lh": ctx_path_count_lh,
            "ctx_path_count_rh": ctx_path_count_rh,
            "subc_path_count_lh": subc_path_count_lh,
            "subc_path_count_rh": subc_path_count_rh,
            "ctx_free_energy_lh": ctx_free_energy_lh,
            "ctx_free_energy_rh": ctx_free_energy_rh,
            "subc_free_energy_lh": subc_free_energy_lh,
            "subc_free_energy_rh": subc_free_energy_rh,
            "fixed_endpoint_source": fixed_endpoint_source,
            "fixed_endpoint_file": fixed_endpoint_file,
            "fixed_endpoints": fixed_endpoints,
            "groups": group_exports,
        }
        summary_json = json.dumps(summary, default=self._safe_json_default, indent=2)

        return {
            "subject_id": subject_id,
            "session_id": session_id,
            "group": group_name,
            "modality": modality,
            "input_path": input_path,
            "group_gradients_path": group_gradients_path,
            "parc_scheme": parc_scheme,
            "parc_path": parc_path,
            "adjacency_path": adjacency_path,
            "lambda_value": float(free_energy_payload.get("lambda", self.free_energy_lambda_spin.value())),
            "color_order": str(color_order),
            "path_order_override": str(channel_order),
            "x_axis_label": str(free_energy_payload.get("x_axis_label", self._x_label)),
            "y_axis_label": str(free_energy_payload.get("y_axis_label", self._y_label)),
            "normalize_energy_by_segments": bool(normalize_by_segments),
            "endpoint_selection_mode": str(self._endpoint_selection_mode or "adaptive"),
            "ctx_path_count_lh": ctx_path_count_lh,
            "ctx_path_count_rh": ctx_path_count_rh,
            "subc_path_count_lh": subc_path_count_lh,
            "subc_path_count_rh": subc_path_count_rh,
            "ctx_free_energy_lh": ctx_free_energy_lh,
            "ctx_free_energy_rh": ctx_free_energy_rh,
            "subc_free_energy_lh": subc_free_energy_lh,
            "subc_free_energy_rh": subc_free_energy_rh,
            "fixed_endpoint_source": fixed_endpoint_source,
            "fixed_endpoint_file": fixed_endpoint_file,
            "parcel_labels": self._numeric_point_ids(),
            "parcel_names": np.asarray(self._point_labels, dtype=object),
            "hemisphere_codes": np.asarray(self._point_group_codes, dtype=int),
            "gradients_pair": np.asarray(gradients_pair, dtype=float),
            "gradients_avg": np.asarray(gradients_avg, dtype=float),
            "covars_row": dict(covars_row),
            "groups": group_exports,
            "summary_json": summary_json,
        }

    def _free_energy_write_start_dir(self):
        loaded_path_text = str(
            getattr(self, "_loaded_fixed_endpoint_file", "") or ""
        ).strip()
        if loaded_path_text:
            loaded_path = Path(loaded_path_text).expanduser()
            if loaded_path.is_file():
                return loaded_path.parent
        return Path(
            str(dict(self._export_metadata or {}).get("source_dir", Path.cwd()))
        ).expanduser()

    def _on_write_free_energy_clicked(self):
        export_payload = self._free_energy_export_payload()
        if export_payload is None:
            return
        subject_id = str(export_payload.get("subject_id", "") or "").strip()
        session_id = str(export_payload.get("session_id", "") or "").strip()
        order_tag = str(export_payload.get("path_order_override", self._path_channel_order or "RGB") or "RGB").strip()
        source_dir = self._free_energy_write_start_dir()
        if subject_id and session_id:
            default_name = f"sub-{subject_id}_ses-{session_id}_order-{order_tag}_desc-free_energy_paths.npz"
        else:
            source_name = str(dict(self._export_metadata or {}).get("source_name", self._title))
            default_name = f"{self._safe_name_fragment(source_name)}_order-{order_tag}_desc-free_energy_paths.npz"
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Write free energy to NPZ",
            str(source_dir / default_name),
            "NumPy archive (*.npz);;All files (*)",
        )
        if not path:
            return
        output_path = Path(path)
        if output_path.suffix.lower() != ".npz":
            output_path = output_path.with_suffix(".npz")

        covars_row = dict(export_payload.get("covars_row", {}) or {})
        try:
            np.savez_compressed(
                str(output_path),
                subject_id=np.asarray(str(export_payload.get("subject_id", ""))),
                session_id=np.asarray(str(export_payload.get("session_id", ""))),
                group=np.asarray(str(export_payload.get("group", ""))),
                modality=np.asarray(str(export_payload.get("modality", ""))),
                input_path=np.asarray(str(export_payload.get("input_path", ""))),
                group_gradients_path=np.asarray(str(export_payload.get("group_gradients_path", ""))),
                parc_scheme=np.asarray(str(export_payload.get("parc_scheme", ""))),
                parc_path=np.asarray(str(export_payload.get("parc_path", ""))),
                adjacency_path=np.asarray(str(export_payload.get("adjacency_path", ""))),
                lambda_value=np.asarray(float(export_payload.get("lambda_value", 1.0))),
                color_order=np.asarray(str(export_payload.get("color_order", ""))),
                path_order_override=np.asarray(str(export_payload.get("path_order_override", ""))),
                x_axis_label=np.asarray(str(export_payload.get("x_axis_label", ""))),
                y_axis_label=np.asarray(str(export_payload.get("y_axis_label", ""))),
                normalize_energy_by_segments=np.asarray(bool(export_payload.get("normalize_energy_by_segments", False))),
                norm_segments=np.asarray(bool(export_payload.get("normalize_energy_by_segments", False))),
                endpoint_selection_mode=np.asarray(str(export_payload.get("endpoint_selection_mode", "adaptive"))),
                ctx_path_count_lh=np.asarray(int(export_payload.get("ctx_path_count_lh", 0)), dtype=int),
                ctx_path_count_rh=np.asarray(int(export_payload.get("ctx_path_count_rh", 0)), dtype=int),
                subc_path_count_lh=np.asarray(int(export_payload.get("subc_path_count_lh", 0)), dtype=int),
                subc_path_count_rh=np.asarray(int(export_payload.get("subc_path_count_rh", 0)), dtype=int),
                ctx_free_energy_lh=np.asarray(float(export_payload.get("ctx_free_energy_lh", float("nan"))), dtype=float),
                ctx_free_energy_rh=np.asarray(float(export_payload.get("ctx_free_energy_rh", float("nan"))), dtype=float),
                subc_free_energy_lh=np.asarray(float(export_payload.get("subc_free_energy_lh", float("nan"))), dtype=float),
                subc_free_energy_rh=np.asarray(float(export_payload.get("subc_free_energy_rh", float("nan"))), dtype=float),
                parcel_labels=np.asarray(export_payload.get("parcel_labels"), dtype=object),
                parcel_names=np.asarray(export_payload.get("parcel_names"), dtype=object),
                hemisphere_codes=np.asarray(export_payload.get("hemisphere_codes"), dtype=int),
                gradients_pair=np.asarray(export_payload.get("gradients_pair"), dtype=float),
                gradients_avg=np.asarray(export_payload.get("gradients_avg"), dtype=float),
                covars_row=np.asarray([covars_row], dtype=object),
                covars_row_json=np.asarray(json.dumps(covars_row, default=self._safe_json_default, indent=2)),
                fixed_endpoint_source=np.asarray(str(export_payload.get("fixed_endpoint_source", ""))),
                fixed_endpoint_file=np.asarray(str(export_payload.get("fixed_endpoint_file", ""))),
                fixed_endpoints_json=np.asarray(
                    json.dumps(
                        {
                            str(group.get("group", "all")): {
                                "path_order": str(group.get("path_order", "")),
                                "anchors": {
                                    str(channel): {
                                        "node_index": int(node.get("node_index", -1)),
                                        "node_label": str(node.get("node_label", "")),
                                        "node_name": str(node.get("node_name", "")),
                                    }
                                    for channel, node in dict(group.get("anchors", {})).items()
                                },
                                "subc_anchor": (
                                    {
                                        "node_index": int(group.get("subc_anchor", {}).get("node_index", -1)),
                                        "node_label": str(group.get("subc_anchor", {}).get("node_label", "")),
                                        "node_name": str(group.get("subc_anchor", {}).get("node_name", "")),
                                    }
                                    if isinstance(group.get("subc_anchor"), dict)
                                    else None
                                ),
                            }
                            for group in list(export_payload.get("groups", []))
                        },
                        default=self._safe_json_default,
                        indent=2,
                    )
                ),
                groups=np.asarray(export_payload.get("groups", []), dtype=object),
                summary_json=np.asarray(str(export_payload.get("summary_json", ""))),
            )
        except Exception as exc:
            warn(f"Failed to write free energy NPZ `{output_path}`: {exc}")
            return
        warn(f"Wrote free energy NPZ to {output_path}")
        self._maybe_run_rostrocaudal_stats_after_write(output_path, export_payload)

    def _rostrocaudal_input_path_default(self, export_payload):
        metadata = dict(getattr(self, "_export_metadata", {}) or {})
        payload = dict(export_payload or {})
        for key in ("group_gradients_path", "input_path", "source_path"):
            value = str(payload.get(key, metadata.get(key, "")) or "").strip()
            if value:
                return value
        return ""

    def _maybe_run_rostrocaudal_stats_after_write(self, output_path, export_payload):
        subject_id = str(dict(export_payload or {}).get("subject_id", "") or "").strip()
        session_id = str(dict(export_payload or {}).get("session_id", "") or "").strip()
        if not subject_id or not session_id:
            warn("Rostrocaudal null test not offered: saved NPZ has no subject/session metadata.")
            return

        dialog = RostrocaudalStatsRunDialog(
            output_path=Path(output_path).expanduser(),
            subject_id=subject_id,
            session_id=session_id,
            input_path=self._rostrocaudal_input_path_default(export_payload),
            pathsfolder=Path(output_path).expanduser().parent,
            theme_name=getattr(self, "_theme_name", "Dark"),
            parent=self,
        )
        if dialog.exec() != 1:
            return

        values = dialog.values()
        argv = list(values.get("argv", []))
        argvs = [list(command or []) for command in list(values.get("argvs", [])) if command]
        cwd = Path(values.get("cwd", _mrsitoolbox_root())).expanduser()
        progress_dialog = RostrocaudalStatsProgressDialog(
            argv=argv,
            argvs=argvs,
            cwd=cwd,
            subject_id=subject_id,
            session_id=session_id,
            theme_name=getattr(self, "_theme_name", "Dark"),
            parent=self,
        )
        self._rostrocaudal_stats_progress_dialog = progress_dialog
        progress_dialog.show()
        progress_dialog.start()

        self.status_label.setText(
            f"Started null tests for sub-{subject_id}_ses-{session_id}"
        )

    def _export_paths_payload(self):
        if not isinstance(self._project_paths_payload, dict):
            return None
        groups = []
        ctx_path_label = str(self._project_paths_payload.get("channel_order", self._triangular_color_order or "")).strip()
        ctx_segment_labels = [
            f"{ctx_path_label[idx]}{ctx_path_label[idx + 1]}"
            for idx in range(max(0, len(ctx_path_label) - 1))
        ]
        subc_from_endpoint = ctx_path_label[1] if len(ctx_path_label) >= 2 else ""
        for group_payload in list(self._project_paths_payload.get("group_paths", [])):
            ctx_full_paths = []
            for path in list(group_payload.get("all_full_paths", [])):
                nodes = [int(node) for node in list(path or [])]
                if len(nodes) < 2:
                    continue
                ctx_full_paths.append([self._path_export_node(node) for node in nodes])
            subc_full_paths = []
            for path in list(group_payload.get("subc_paths", [])):
                nodes = [int(node) for node in list(path or [])]
                if len(nodes) < 2:
                    continue
                subc_full_paths.append([self._path_export_node(node) for node in nodes])
            ctx_optimal_nodes = [
                self._path_export_node(node)
                for node in list(group_payload.get("optimal_full_path", []))
            ]
            subc_optimal_nodes = [
                self._path_export_node(node)
                for node in list(group_payload.get("subc_optimal_path", []))
            ]
            ctx_endpoints = {}
            for channel, node_index in dict(group_payload.get("anchors", {})).items():
                try:
                    ctx_endpoints[str(channel)] = self._path_export_node(int(node_index))
                except Exception:
                    continue
            ordered_ctx_endpoints = []
            for channel in [str(channel) for channel in str(self._triangular_color_order or "RGB")]:
                if channel in ctx_endpoints:
                    ordered_ctx_endpoints.append(ctx_endpoints[channel])
            subc_endpoint = None
            if group_payload.get("subc_anchor") is not None:
                try:
                    subc_endpoint = self._path_export_node(int(group_payload.get("subc_anchor")))
                except Exception:
                    subc_endpoint = None
            subc_target_name = ""
            subc_target_label = ""
            if isinstance(subc_endpoint, dict):
                subc_target_name = str(subc_endpoint.get("node_name", "")).strip()
                subc_target_label = str(subc_endpoint.get("node_label", "")).strip()
            subc_path_label = (
                f"{subc_from_endpoint}->{subc_target_name or subc_target_label}"
                if subc_from_endpoint and (subc_target_name or subc_target_label)
                else str(subc_target_name or subc_target_label or subc_from_endpoint)
            )

            def _export_segment_record(record):
                rec = dict(record or {})
                first = str(rec.get("first", "")).strip()
                second = str(rec.get("second", "")).strip()
                nodes = [self._path_export_node(node) for node in list(rec.get("nodes", []))]
                return {
                    "pair_label": f"{first}{second}" if first or second else "",
                    "from_endpoint": first,
                    "to_endpoint": second,
                    "nodes": nodes,
                }

            ctx_optimal_segments_detail = [
                _export_segment_record(record)
                for record in list(group_payload.get("optimal_segments", []))
            ]
            ctx_segment_paths_detail = [
                _export_segment_record(record)
                for record in list(group_payload.get("all_pair_paths", []))
            ]
            ctx_optimal_path_detail = {
                "path_label": ctx_path_label,
                "segment_labels": list(ctx_segment_labels),
                "nodes": ctx_optimal_nodes,
            }
            ctx_paths_detail = [
                {
                    "path_label": ctx_path_label,
                    "segment_labels": list(ctx_segment_labels),
                    "nodes": path_nodes,
                }
                for path_nodes in ctx_full_paths
            ]
            subc_optimal_path_detail = {
                "path_label": subc_path_label,
                "from_endpoint": subc_from_endpoint,
                "to_endpoint_label": subc_target_label,
                "to_endpoint_name": subc_target_name,
                "nodes": subc_optimal_nodes,
            }
            subc_paths_detail = [
                {
                    "path_label": subc_path_label,
                    "from_endpoint": subc_from_endpoint,
                    "to_endpoint_label": subc_target_label,
                    "to_endpoint_name": subc_target_name,
                    "nodes": path_nodes,
                }
                for path_nodes in subc_full_paths
            ]
            groups.append(
                {
                    "group": str(group_payload.get("group", "all")),
                    "ctx_path_count": int(group_payload.get("ctx_path_count", len(ctx_full_paths))),
                    "subc_path_count": int(group_payload.get("subc_path_count", len(subc_full_paths))),
                    "ctx_path_label": ctx_path_label,
                    "ctx_segment_labels": list(ctx_segment_labels),
                    "ctx_endpoints": ctx_endpoints,
                    "ctx_ordered_endpoints": ordered_ctx_endpoints,
                    "subc_endpoint": subc_endpoint,
                    "subc_path_label": subc_path_label,
                    "subc_from_endpoint": subc_from_endpoint,
                    "ctx_optimal_path": ctx_optimal_nodes,
                    "ctx_paths": ctx_full_paths,
                    "ctx_optimal_path_detail": ctx_optimal_path_detail,
                    "ctx_paths_detail": ctx_paths_detail,
                    "ctx_optimal_segments": ctx_optimal_segments_detail,
                    "ctx_segment_paths": ctx_segment_paths_detail,
                    "subc_optimal_path": subc_optimal_nodes,
                    "subc_paths": subc_full_paths,
                    "subc_optimal_path_detail": subc_optimal_path_detail,
                    "subc_paths_detail": subc_paths_detail,
                }
            )
        return {
            "title": self._title,
            "x_axis_label": self._rotate_axis_labels(
                self._x_label,
                self._y_label,
                self._rotation_preset,
            )[0],
            "y_axis_label": self._rotate_axis_labels(
                self._x_label,
                self._y_label,
                self._rotation_preset,
            )[1],
            "rotation": self._rotation_preset,
            "radius": float(self._proximity_radius),
            "fit_mode": self._rgb_fit_mode,
            "color_order": self._triangular_color_order,
            "groups": groups,
        }

    def _on_export_paths_clicked(self):
        export_payload = self._export_paths_payload()
        if export_payload is None:
            return
        has_paths = any(
            int(group.get("ctx_path_count", 0)) > 0 or int(group.get("subc_path_count", 0)) > 0
            for group in export_payload["groups"]
        )
        if not has_paths:
            return
        default_name = "gradient_paths.json"
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export classification paths",
            str(Path.cwd() / default_name),
            "JSON (*.json);;Text (*.txt)",
        )
        if not path:
            return
        output_path = Path(path)
        if output_path.suffix.lower() not in {".json", ".txt"}:
            if "Text" in selected_filter:
                output_path = output_path.with_suffix(".txt")
            else:
                output_path = output_path.with_suffix(".json")
        if output_path.suffix.lower() == ".txt":
            lines = [
                f"Title: {export_payload['title']}",
                f"X axis: {export_payload['x_axis_label']}",
                f"Y axis: {export_payload['y_axis_label']}",
                f"Rotation: {export_payload['rotation']}",
                f"Radius: {export_payload['radius']:.4f}",
                f"Fit mode: {export_payload['fit_mode']}",
                f"Color order: {export_payload['color_order']}",
                "",
            ]
            for group in export_payload["groups"]:
                lines.append(
                    f"[{str(group['group']).upper()}] ctx={int(group['ctx_path_count'])} subc={int(group['subc_path_count'])}"
                )
                if group["ctx_optimal_path"]:
                    lines.append("CTX optimal:")
                    lines.extend(
                        f"  {node['node_label']} | {node['node_name']} | x={node['x_coord']:.6f} | y={node['y_coord']:.6f}"
                        for node in group["ctx_optimal_path"]
                    )
                for idx, path_nodes in enumerate(group["ctx_paths"], start=1):
                    lines.append(f"CTX path {idx}:")
                    lines.extend(
                        f"  {node['node_label']} | {node['node_name']} | x={node['x_coord']:.6f} | y={node['y_coord']:.6f}"
                        for node in path_nodes
                    )
                if group["subc_optimal_path"]:
                    lines.append("SUBC optimal:")
                    lines.extend(
                        f"  {node['node_label']} | {node['node_name']} | x={node['x_coord']:.6f} | y={node['y_coord']:.6f}"
                        for node in group["subc_optimal_path"]
                    )
                for idx, path_nodes in enumerate(group["subc_paths"], start=1):
                    lines.append(f"SUBC path {idx}:")
                    lines.extend(
                        f"  {node['node_label']} | {node['node_name']} | x={node['x_coord']:.6f} | y={node['y_coord']:.6f}"
                        for node in path_nodes
                    )
                lines.append("")
            output_path.write_text("\n".join(lines), encoding="utf-8")
        else:
            output_path.write_text(json.dumps(export_payload, indent=2), encoding="utf-8")

    @staticmethod
    def _combine_ordered_segments(segments):
        combined = []
        for _first, _second, path in list(segments or []):
            nodes = [int(node) for node in list(path or [])]
            if len(nodes) < 2:
                return []
            if not combined:
                combined.extend(nodes)
            elif combined[-1] == nodes[0]:
                combined.extend(nodes[1:])
            else:
                combined.extend(nodes)
        return combined

    @staticmethod
    def _path_record(first, second, path_nodes, color):
        return {
            "first": str(first),
            "second": str(second),
            "nodes": [int(node) for node in list(path_nodes or [])],
            "color": [float(value) for value in np.asarray(color, dtype=float).reshape(3).tolist()],
        }

    @staticmethod
    def _combine_ordered_path_records(segment_records, max_full_paths=256):
        records = list(segment_records or [])
        if not records:
            return []
        max_full_paths = max(1, int(max_full_paths))
        combined = []

        def extend(segment_index, current_path):
            if len(combined) >= max_full_paths:
                return
            if segment_index >= len(records):
                if len(current_path) >= 2:
                    combined.append(list(current_path))
                return
            _first, _second, candidate_paths = records[segment_index]
            for path_nodes in list(candidate_paths or []):
                nodes = [int(node) for node in list(path_nodes or [])]
                if len(nodes) < 2:
                    continue
                if not current_path:
                    extend(segment_index + 1, nodes)
                else:
                    if current_path[-1] != nodes[0]:
                        continue
                    used = set(int(node) for node in current_path[:-1])
                    if any(int(node) in used for node in nodes[1:]):
                        continue
                    extend(segment_index + 1, list(current_path) + nodes[1:])
                if len(combined) >= max_full_paths:
                    break

        extend(0, [])
        return combined

    @staticmethod
    def _fallback_triangle_vertices(x_values, y_values):
        x_values = np.asarray(x_values, dtype=float).reshape(-1)
        y_values = np.asarray(y_values, dtype=float).reshape(-1)
        finite_mask = np.isfinite(x_values) & np.isfinite(y_values)
        if not np.any(finite_mask):
            return np.array([[0.5, 1.0], [0.0, 0.0], [1.0, 0.0]], dtype=float)
        x_valid = x_values[finite_mask]
        y_valid = y_values[finite_mask]
        x_min, x_max, y_min, y_max = GradientScatterDialog._triangular_rgb_bounds(x_valid, y_valid)
        return np.array(
            [
                [0.5 * (x_min + x_max), y_max],
                [x_min, y_min],
                [x_max, y_min],
            ],
            dtype=float,
        )

    @staticmethod
    def _fit_square_outline(x_values, y_values):
        x_values = np.asarray(x_values, dtype=float).reshape(-1)
        y_values = np.asarray(y_values, dtype=float).reshape(-1)
        finite_mask = np.isfinite(x_values) & np.isfinite(y_values)
        if not np.any(finite_mask):
            center_x = 0.0
            center_y = 0.0
            half = 1.0
        else:
            x_valid = x_values[finite_mask]
            y_valid = y_values[finite_mask]
            x_min, x_max, y_min, y_max = GradientScatterDialog._triangular_rgb_bounds(x_valid, y_valid)
            center_x = 0.5 * (x_min + x_max)
            center_y = 0.5 * (y_min + y_max)
            half = max(0.5 * (x_max - x_min), 0.5 * (y_max - y_min), 1e-6)
        left = center_x - half
        right = center_x + half
        bottom = center_y - half
        top = center_y + half
        outline = np.asarray(
            (
                (left, top),
                (right, top),
                (right, bottom),
                (left, bottom),
            ),
            dtype=float,
        )
        anchor_points = np.asarray(
            (
                (center_x, top),
                (left, center_y),
                (right, center_y),
            ),
            dtype=float,
        )
        return outline, anchor_points

    @staticmethod
    def _triangular_rgb_bounds(x_values, y_values):
        x_valid = np.asarray(x_values, dtype=float).reshape(-1)
        y_valid = np.asarray(y_values, dtype=float).reshape(-1)
        x_min, x_max = np.nanmin(x_valid), np.nanmax(x_valid)
        y_min, y_max = np.nanmin(y_valid), np.nanmax(y_valid)
        return float(x_min), float(x_max), float(y_min), float(y_max)

    @staticmethod
    def _triangle_area2(a, b, c):
        return abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    @staticmethod
    def _fit_triangle_vertices(x_values, y_values):
        return np.asarray(nettools.fit_triangle_vertices(x_values, y_values), dtype=float)

    @staticmethod
    def _fallback_triangle_vertices_3d(x_values, y_values, z_values):
        points = np.column_stack(
            (
                np.asarray(x_values, dtype=float).reshape(-1),
                np.asarray(y_values, dtype=float).reshape(-1),
                np.asarray(z_values, dtype=float).reshape(-1),
            )
        )
        finite_mask = np.isfinite(points).all(axis=1)
        if not np.any(finite_mask):
            return np.array(
                [[0.0, 0.0, 1.0], [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0]],
                dtype=float,
            )
        valid = points[finite_mask]
        mins = np.nanmin(valid, axis=0)
        maxs = np.nanmax(valid, axis=0)
        center = 0.5 * (mins + maxs)
        vertices = np.asarray(
            (
                (center[0], center[1], maxs[2]),
                (mins[0], mins[1], mins[2]),
                (maxs[0], mins[1], mins[2]),
            ),
            dtype=float,
        )
        if GradientScatterDialog._triangle_area3d(vertices[0], vertices[1], vertices[2]) <= 1e-8:
            scale = max(float(np.nanmax(maxs - mins)), 1.0)
            vertices = np.asarray(
                (
                    center + np.array((0.0, 0.0, scale), dtype=float),
                    center + np.array((-scale, -scale, -scale), dtype=float),
                    center + np.array((scale, -scale, -scale), dtype=float),
                ),
                dtype=float,
            )
        return vertices

    @staticmethod
    def _triangle_area3d(a, b, c):
        return float(np.linalg.norm(np.cross(np.asarray(b, dtype=float) - a, np.asarray(c, dtype=float) - a)))

    @staticmethod
    def _tetra_volume6(a, b, c, d):
        return float(
            abs(
                np.dot(
                    np.asarray(b, dtype=float) - a,
                    np.cross(np.asarray(c, dtype=float) - a, np.asarray(d, dtype=float) - a),
                )
            )
        )

    @staticmethod
    def _fallback_pyramid_vertices_3d(x_values, y_values, z_values):
        points = np.column_stack(
            (
                np.asarray(x_values, dtype=float).reshape(-1),
                np.asarray(y_values, dtype=float).reshape(-1),
                np.asarray(z_values, dtype=float).reshape(-1),
            )
        )
        finite_mask = np.isfinite(points).all(axis=1)
        if not np.any(finite_mask):
            return np.asarray(
                (
                    (0.0, 0.0, 1.0),
                    (-1.0, -1.0, -1.0),
                    (1.0, -1.0, -1.0),
                    (0.0, 1.0, -1.0),
                ),
                dtype=float,
            )
        valid = points[finite_mask]
        mins = np.nanmin(valid, axis=0)
        maxs = np.nanmax(valid, axis=0)
        center = 0.5 * (mins + maxs)
        span = np.maximum(maxs - mins, 1e-6)
        scale = max(float(np.nanmax(span)), 1.0)
        vertices = np.asarray(
            (
                (center[0], center[1], maxs[2] + 0.08 * scale),
                (mins[0], mins[1], mins[2]),
                (maxs[0], mins[1], mins[2]),
                (center[0], maxs[1], mins[2]),
            ),
            dtype=float,
        )
        if GradientScatterDialog._tetra_volume6(vertices[0], vertices[1], vertices[2], vertices[3]) <= 1e-8:
            vertices = np.asarray(
                (
                    center + np.array((0.0, 0.0, scale), dtype=float),
                    center + np.array((-scale, -scale, -scale), dtype=float),
                    center + np.array((scale, -scale, -scale), dtype=float),
                    center + np.array((0.0, scale, -scale), dtype=float),
                ),
                dtype=float,
            )
        return vertices

    @staticmethod
    def _order_triangle_vertices_3d(vertices):
        vertices = np.asarray(vertices, dtype=float)
        apex_index = int(np.argmax(vertices[:, 2]))
        apex = vertices[apex_index]
        base = np.delete(vertices, apex_index, axis=0)
        base = base[np.argsort(base[:, 0])]
        return np.asarray((apex, base[0], base[1]), dtype=float)

    @staticmethod
    def _order_pyramid_vertices_3d(vertices):
        vertices = np.asarray(vertices, dtype=float)
        apex_index = int(np.argmax(vertices[:, 2]))
        apex = vertices[apex_index]
        base = np.delete(vertices, apex_index, axis=0)
        base = base[np.lexsort((base[:, 1], base[:, 0]))]
        return np.asarray((apex, base[0], base[1], base[2]), dtype=float)

    @staticmethod
    def _limit_hull_candidates_3d(candidate_points, max_count=56):
        points = np.asarray(candidate_points, dtype=float)
        if points.shape[0] <= max_count:
            return points
        extrema = []
        for axis in range(3):
            extrema.append(points[np.argmin(points[:, axis])])
            extrema.append(points[np.argmax(points[:, axis])])
        extrema = np.unique(np.asarray(extrema, dtype=float), axis=0)
        remaining = points
        if extrema.size:
            keep = np.ones(points.shape[0], dtype=bool)
            for vertex in extrema:
                keep &= ~np.all(np.isclose(points, vertex[np.newaxis, :]), axis=1)
            remaining = points[keep]
        sample_count = max(0, int(max_count) - int(extrema.shape[0]))
        if remaining.shape[0] > sample_count and sample_count > 0:
            indices = np.linspace(0, remaining.shape[0] - 1, sample_count, dtype=int)
            remaining = remaining[indices]
        elif sample_count <= 0:
            remaining = np.empty((0, 3), dtype=float)
        return np.unique(np.vstack((extrema, remaining)), axis=0)

    @staticmethod
    def _fit_triangle_vertices_3d(x_values, y_values, z_values):
        points = np.column_stack(
            (
                np.asarray(x_values, dtype=float).reshape(-1),
                np.asarray(y_values, dtype=float).reshape(-1),
                np.asarray(z_values, dtype=float).reshape(-1),
            )
        )
        points = points[np.isfinite(points).all(axis=1)]
        if points.shape[0] < 3:
            return GradientScatterDialog._fallback_triangle_vertices_3d(x_values, y_values, z_values)
        unique_points = np.unique(points, axis=0)
        if unique_points.shape[0] < 3:
            return GradientScatterDialog._fallback_triangle_vertices_3d(x_values, y_values, z_values)

        candidate_points = unique_points
        try:
            from scipy.spatial import ConvexHull

            hull = ConvexHull(unique_points)
            candidate_points = unique_points[np.unique(hull.vertices)]
        except Exception:
            pass

        if candidate_points.shape[0] > 96:
            step = max(1, int(np.ceil(candidate_points.shape[0] / 96.0)))
            reduced = candidate_points[::step]
            extrema = []
            for axis in range(3):
                extrema.append(candidate_points[np.argmin(candidate_points[:, axis])])
                extrema.append(candidate_points[np.argmax(candidate_points[:, axis])])
            candidate_points = np.unique(np.vstack((reduced, np.asarray(extrema, dtype=float))), axis=0)

        best_vertices = None
        best_area = -1.0
        for idx_a, idx_b, idx_c in combinations(range(candidate_points.shape[0]), 3):
            a = candidate_points[idx_a]
            b = candidate_points[idx_b]
            c = candidate_points[idx_c]
            area = GradientScatterDialog._triangle_area3d(a, b, c)
            if area > best_area:
                best_area = area
                best_vertices = np.asarray((a, b, c), dtype=float)
        if best_vertices is None or best_area <= 1e-8:
            return GradientScatterDialog._fallback_triangle_vertices_3d(x_values, y_values, z_values)
        return GradientScatterDialog._order_triangle_vertices_3d(best_vertices)

    @staticmethod
    def _fit_pyramid_vertices_3d(x_values, y_values, z_values):
        points = np.column_stack(
            (
                np.asarray(x_values, dtype=float).reshape(-1),
                np.asarray(y_values, dtype=float).reshape(-1),
                np.asarray(z_values, dtype=float).reshape(-1),
            )
        )
        points = points[np.isfinite(points).all(axis=1)]
        if points.shape[0] < 4:
            return GradientScatterDialog._fallback_pyramid_vertices_3d(x_values, y_values, z_values)
        unique_points = np.unique(points, axis=0)
        if unique_points.shape[0] < 4:
            return GradientScatterDialog._fallback_pyramid_vertices_3d(x_values, y_values, z_values)

        candidate_points = unique_points
        try:
            from scipy.spatial import ConvexHull

            hull = ConvexHull(unique_points)
            candidate_points = unique_points[np.unique(hull.vertices)]
        except Exception:
            pass
        candidate_points = GradientScatterDialog._limit_hull_candidates_3d(candidate_points, max_count=56)

        if candidate_points.shape[0] < 4:
            return GradientScatterDialog._fallback_pyramid_vertices_3d(x_values, y_values, z_values)
        best_vertices = None
        best_volume = -1.0
        for idx_a, idx_b, idx_c, idx_d in combinations(range(candidate_points.shape[0]), 4):
            a = candidate_points[idx_a]
            b = candidate_points[idx_b]
            c = candidate_points[idx_c]
            d = candidate_points[idx_d]
            volume = GradientScatterDialog._tetra_volume6(a, b, c, d)
            if volume > best_volume:
                best_volume = volume
                best_vertices = np.asarray((a, b, c, d), dtype=float)
        if best_vertices is None or best_volume <= 1e-8:
            return GradientScatterDialog._fallback_pyramid_vertices_3d(x_values, y_values, z_values)
        return GradientScatterDialog._order_pyramid_vertices_3d(best_vertices)

    @staticmethod
    def _rgb_model(x_values, y_values, color_order="RBG", fit_mode="triangle"):
        order = GradientScatterDialog._normalize_triangular_color_order(color_order)
        mode = GradientScatterDialog._normalize_rgb_fit_mode(fit_mode)
        if mode == "square":
            outline, anchor_points = GradientScatterDialog._fit_square_outline(x_values, y_values)
            vertices = np.asarray(outline, dtype=float)
            rgb_basis = {
                "R": np.array((1.0, 0.0, 0.0), dtype=float),
                "G": np.array((0.0, 1.0, 0.0), dtype=float),
                "B": np.array((0.0, 0.0, 1.0), dtype=float),
            }
            vertex_colors = np.asarray([rgb_basis[channel] for channel in order], dtype=float)
            return {
                "vertices": np.asarray(vertices, dtype=float),
                "anchor_points": np.asarray(anchor_points, dtype=float),
                "vertex_colors": vertex_colors,
                "order": order,
                "fit_mode": mode,
            }
        triangle_model = nettools.triangular_rgb_model(x_values, y_values, color_order=order)
        return {
            "vertices": np.asarray(triangle_model["vertices"], dtype=float),
            "anchor_points": np.asarray(triangle_model["anchor_points"], dtype=float),
            "vertex_colors": np.asarray(triangle_model["vertex_colors"], dtype=float),
            "order": str(triangle_model.get("order", order)),
            "fit_mode": mode,
        }

    @staticmethod
    def _rgb_model_3d(x_values, y_values, z_values, color_order="RBG"):
        order = GradientScatterDialog._normalize_triangular_color_order(color_order)
        rgb_basis = {
            "R": np.array((1.0, 0.0, 0.0), dtype=float),
            "G": np.array((0.0, 1.0, 0.0), dtype=float),
            "B": np.array((0.0, 0.0, 1.0), dtype=float),
        }
        vertices = GradientScatterDialog._fit_pyramid_vertices_3d(x_values, y_values, z_values)
        apex_color = np.array((1.0, 1.0, 1.0), dtype=float)
        vertex_colors = np.asarray(
            [apex_color] + [rgb_basis[channel] for channel in order],
            dtype=float,
        )
        return {
            "vertices": np.asarray(vertices, dtype=float),
            "anchor_points": np.asarray(vertices, dtype=float),
            "vertex_colors": vertex_colors,
            "order": f"W{order}",
            "fit_mode": "pyramid3d",
        }

    @staticmethod
    def _normalize_rgb_chroma(values):
        colors = np.asarray(values, dtype=float)
        if colors.ndim != 2 or colors.shape[1] != 3:
            return np.clip(colors, 0.0, 1.0)
        colors = np.clip(colors, 0.0, 1.0)
        scale = np.max(colors, axis=1, keepdims=True)
        scale[scale <= 1e-9] = 1.0
        return np.clip(colors / scale, 0.0, 1.0)

    @staticmethod
    def _rgb_colors_from_model(x_values, y_values, model):
        x_valid = np.asarray(x_values, dtype=float).reshape(-1)
        y_valid = np.asarray(y_values, dtype=float).reshape(-1)
        colors = np.full((x_valid.shape[0], 3), 0.65, dtype=float)
        finite_mask = np.isfinite(x_valid) & np.isfinite(y_valid)
        if not np.any(finite_mask):
            return colors

        fit_mode = GradientScatterDialog._normalize_rgb_fit_mode(model.get("fit_mode", "triangle"))
        if fit_mode != "square":
            return np.asarray(nettools.triangular_rgb_colors_from_model(x_values, y_values, model), dtype=float)
        vertex_colors = np.asarray(model["vertex_colors"], dtype=float)
        if fit_mode == "square":
            anchor_points = np.asarray(model["anchor_points"], dtype=float)
            points = np.column_stack((x_valid[finite_mask], y_valid[finite_mask]))
            deltas = points[:, np.newaxis, :] - anchor_points[np.newaxis, :, :]
            distances = np.sqrt(np.sum(np.square(deltas), axis=2))
            weights = 1.0 / np.maximum(distances, 1e-9)
            close_mask = distances <= 1e-9
            if np.any(close_mask):
                for row_idx in np.flatnonzero(np.any(close_mask, axis=1)).tolist():
                    weights[row_idx, :] = close_mask[row_idx, :].astype(float)
            weight_sum = weights.sum(axis=1, keepdims=True)
            weight_sum[weight_sum <= 0] = 1.0
            weights /= weight_sum
            colors[finite_mask] = GradientScatterDialog._normalize_rgb_chroma(weights @ vertex_colors)
            return np.clip(colors, 0.0, 1.0)

        return np.clip(colors, 0.0, 1.0)

    @staticmethod
    def _rgb_colors_from_model_3d(x_values, y_values, z_values, model):
        x_valid = np.asarray(x_values, dtype=float).reshape(-1)
        y_valid = np.asarray(y_values, dtype=float).reshape(-1)
        z_valid = np.asarray(z_values, dtype=float).reshape(-1)
        colors = np.full((x_valid.shape[0], 3), 0.65, dtype=float)
        finite_mask = np.isfinite(x_valid) & np.isfinite(y_valid) & np.isfinite(z_valid)
        if not np.any(finite_mask):
            return colors

        vertices = np.asarray(model["vertices"], dtype=float)
        if vertices.shape[0] >= 4:
            v0, v1, v2, v3 = vertices[:4]
            transform = np.column_stack((v1 - v0, v2 - v0, v3 - v0))
            try:
                if abs(float(np.linalg.det(transform))) <= 1e-12:
                    return colors
                points = np.column_stack((x_valid[finite_mask], y_valid[finite_mask], z_valid[finite_mask]))
                coeffs = np.linalg.solve(transform, (points - v0[np.newaxis, :]).T).T
            except Exception:
                return colors
            w0 = 1.0 - np.sum(coeffs, axis=1)
            weights = np.column_stack((w0, coeffs))
            weights = np.clip(weights, 0.0, 1.0)
            weight_sum = weights.sum(axis=1, keepdims=True)
            weight_sum[weight_sum <= 0] = 1.0
            weights /= weight_sum
            vertex_colors = np.asarray(model["vertex_colors"], dtype=float)[:4, :]
            colors[finite_mask] = GradientScatterDialog._normalize_rgb_chroma(weights @ vertex_colors)
            return np.clip(colors, 0.0, 1.0)

        v0, v1, v2 = vertices[:3]
        e1 = v1 - v0
        e2 = v2 - v0
        d00 = float(np.dot(e1, e1))
        d01 = float(np.dot(e1, e2))
        d11 = float(np.dot(e2, e2))
        denom = d00 * d11 - d01 * d01
        if np.isclose(denom, 0.0):
            return colors

        points = np.column_stack((x_valid[finite_mask], y_valid[finite_mask], z_valid[finite_mask]))
        deltas = points - v0[np.newaxis, :]
        d20 = deltas @ e1
        d21 = deltas @ e2
        w1 = (d11 * d20 - d01 * d21) / denom
        w2 = (d00 * d21 - d01 * d20) / denom
        w0 = 1.0 - w1 - w2
        weights = np.column_stack((w0, w1, w2))
        weights = np.clip(weights, 0.0, 1.0)
        weight_sum = weights.sum(axis=1, keepdims=True)
        weight_sum[weight_sum <= 0] = 1.0
        weights /= weight_sum
        vertex_colors = np.asarray(model["vertex_colors"], dtype=float)
        colors[finite_mask] = GradientScatterDialog._normalize_rgb_chroma(weights @ vertex_colors)
        return np.clip(colors, 0.0, 1.0)

    @staticmethod
    def _triangular_rgb_model(x_values, y_values, color_order="RBG"):
        return GradientScatterDialog._rgb_model(
            x_values,
            y_values,
            color_order=color_order,
            fit_mode="triangle",
        )

    @staticmethod
    def _triangular_rgb_colors_from_model(x_values, y_values, model):
        return GradientScatterDialog._rgb_colors_from_model(x_values, y_values, model)

    @staticmethod
    def _valid_view_limits(limits):
        try:
            values = np.asarray(limits, dtype=float).reshape(2)
        except Exception:
            return None
        if not np.all(np.isfinite(values)) or np.isclose(values[0], values[1]):
            return None
        return (float(values[0]), float(values[1]))

    def _capture_scatter_view_limits(self):
        limits_by_group = {}
        for entry in list(getattr(self, "_point_artist_entries", [])):
            ax = entry.get("axes") if isinstance(entry, dict) else None
            if ax is None:
                continue
            group_name = str(entry.get("group", "all") or "all").strip().lower()
            xlim = self._valid_view_limits(ax.get_xlim())
            ylim = self._valid_view_limits(ax.get_ylim())
            if xlim is None or ylim is None:
                continue
            limits_by_group[group_name] = {"xlim": xlim, "ylim": ylim}
        return limits_by_group

    def _restore_scatter_view_limits(self, ax, group_name, limits_by_group):
        if not limits_by_group:
            return
        group_key = str(group_name or "all").strip().lower()
        limits = limits_by_group.get(group_key)
        if limits is None and group_key != "all":
            limits = limits_by_group.get("all")
        if not isinstance(limits, dict):
            return
        xlim = self._valid_view_limits(limits.get("xlim"))
        ylim = self._valid_view_limits(limits.get("ylim"))
        if xlim is None or ylim is None:
            return
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    def _render(self, *, preserve_view=False):
        preserved_view_limits = self._capture_scatter_view_limits() if bool(preserve_view) else {}
        self.figure.clear()
        self._point_artist = None
        self._point_artist_entries = []
        appearance = self._current_scatter_appearance_settings()
        figure_title = str(appearance.get("figure_title", "") or "").strip()
        axis_label_fontsize = int(appearance.get("axis_label_fontsize", 12))
        tick_label_fontsize = int(appearance.get("tick_label_fontsize", 11))
        x_plot, y_plot = self._rotate_points(self._x, self._y, self._rotation_preset)
        z_plot = np.asarray(self._z, dtype=float) if self._is_3d else None
        x_label, y_label = self._rotate_axis_labels(
            self._x_label,
            self._y_label,
            self._rotation_preset,
        )
        z_label = self._z_label
        visible_edge_pairs = np.zeros((0, 2), dtype=int) if self._is_3d else self._visible_edge_pairs()
        visible_edge_distances = np.zeros(0, dtype=float) if self._is_3d else self._visible_edge_distances()
        subplot_specs = self._display_group_specs()
        if len(subplot_specs) <= 1:
            axes = [self.figure.add_subplot(111, projection="3d" if self._is_3d else None)]
        else:
            grid = self.figure.add_gridspec(1, len(subplot_specs), wspace=0.16)
            axes = [
                self.figure.add_subplot(grid[0, idx], projection="3d" if self._is_3d else None)
                for idx in range(len(subplot_specs))
            ]
        if figure_title:
            self.figure.suptitle(figure_title, fontsize=13)
        shared_scatter = None

        global_triangle_model = None
        point_colors = None
        self._last_rgb_model = None
        self._last_rgb_model_x = None
        self._last_rgb_model_y = None
        self._last_rgb_point_colors = None
        if self._use_triangular_rgb:
            if self._is_3d:
                rgb_x_plot, rgb_y_plot, rgb_z_plot = self._rotated_rgb_points_3d()
                global_triangle_model = self._rgb_model_3d(
                    rgb_x_plot,
                    rgb_y_plot,
                    rgb_z_plot,
                    self._triangular_color_order,
                )
                point_colors = self._rgb_colors_from_model_3d(
                    rgb_x_plot,
                    rgb_y_plot,
                    rgb_z_plot,
                    global_triangle_model,
                )
            else:
                rgb_x_plot, rgb_y_plot = self._rotated_rgb_points()
                global_triangle_model = self._rgb_model(
                    rgb_x_plot,
                    rgb_y_plot,
                    self._triangular_color_order,
                    fit_mode=self._rgb_fit_mode,
                )
                point_colors = self._rgb_colors_from_model(rgb_x_plot, rgb_y_plot, global_triangle_model)
                self._last_rgb_model = global_triangle_model
                self._last_rgb_model_x = np.asarray(rgb_x_plot, dtype=float)
                self._last_rgb_model_y = np.asarray(rgb_y_plot, dtype=float)
                self._last_rgb_point_colors = np.asarray(point_colors, dtype=float)
            if isinstance(self._project_paths_payload, dict):
                self._project_paths_payload["point_colors"] = np.asarray(point_colors, dtype=float).tolist()
                self._project_paths_payload["show_all_ordered_paths"] = bool(self._show_all_ordered_paths)
                self._project_paths_payload["edge_linewidth"] = float(self._edge_linewidth)
                self._project_paths_payload["width_scaling_mode"] = self._path_width_scaling_mode
                self._project_paths_payload["width_scaling_strength"] = float(self._path_width_scaling_strength)
                self._project_paths_payload["edge_pairs"] = np.asarray(self._edge_pairs, dtype=int).tolist()
                self._project_paths_payload["node_count"] = int(self._x.size)
                self._project_paths_payload["fibrenet_layout"] = self._selected_fibrenet_layout()
        else:
            vmin, vmax = self._compute_display_range(self._color)

        for ax, subplot_spec in zip(axes, subplot_specs):
            local_indices = np.asarray(subplot_spec.get("indices", np.arange(self._x.shape[0], dtype=int)), dtype=int).reshape(-1)
            if local_indices.size == 0:
                ax.set_axis_off()
                continue

            local_pairs, local_distances = self._edge_subset_for_indices(
                visible_edge_pairs,
                visible_edge_distances,
                local_indices,
            )
            if not self._is_3d:
                self._draw_proximity_overlay(ax, x_plot[local_indices], y_plot[local_indices])
            if not self._is_3d and self._show_adjacency_edges and local_pairs.size:
                if self._use_edge_bundling:
                    segments = self._bundled_segments_from_pairs(local_pairs)
                else:
                    segments = np.stack(
                        (
                            np.column_stack((x_plot[local_pairs[:, 0]], y_plot[local_pairs[:, 0]])),
                            np.column_stack((x_plot[local_pairs[:, 1]], y_plot[local_pairs[:, 1]])),
                        ),
                        axis=1,
                    )
                ax.add_collection(
                    LineCollection(
                        segments,
                        colors=self._edge_color,
                        linewidths=self._edge_linewidth,
                        alpha=self._edge_alpha,
                        zorder=1,
                    )
                )

            if self._use_triangular_rgb:
                if not self._is_3d and isinstance(self._project_paths_payload, dict):
                    self._draw_triangular_anchor_paths(
                        ax,
                        x_plot,
                        y_plot,
                        point_colors,
                        self._project_paths_payload,
                        group_name=subplot_spec.get("name") if len(subplot_specs) > 1 else None,
                    )
                elif not self._is_3d:
                    self._draw_active_anchor_markers(
                        ax,
                        x_plot,
                        y_plot,
                        group_name=subplot_spec.get("name") if len(subplot_specs) > 1 else None,
                    )
                if self._is_3d:
                    scatter = ax.scatter(
                        x_plot[local_indices],
                        y_plot[local_indices],
                        z_plot[local_indices],
                        c=np.asarray(point_colors[local_indices], dtype=float),
                        s=42,
                        alpha=0.92,
                        linewidths=0.2,
                        edgecolors="#111827",
                        depthshade=False,
                        zorder=2,
                    )
                else:
                    scatter = ax.scatter(
                        x_plot[local_indices],
                        y_plot[local_indices],
                        c=np.asarray(point_colors[local_indices], dtype=float),
                        s=38,
                        alpha=0.92,
                        linewidths=0.2,
                        edgecolors="#111827",
                        zorder=2,
                    )
                if self._is_3d and local_indices.size >= 3 and self._rgb_x is None and self._rgb_y is None and self._rgb_z is None:
                    local_model = self._rgb_model_3d(
                        x_plot[local_indices],
                        y_plot[local_indices],
                        z_plot[local_indices],
                        self._triangular_color_order,
                    )
                    vertices = np.asarray(local_model["vertices"], dtype=float)
                    if vertices.shape[0] >= 4:
                        base = vertices[[1, 2, 3, 1], :]
                        ax.plot(
                            base[:, 0],
                            base[:, 1],
                            base[:, 2],
                            linestyle="--",
                            linewidth=1.1,
                            color="#111827",
                            alpha=0.6,
                            zorder=3,
                        )
                        for base_idx in (1, 2, 3):
                            edge = vertices[[0, base_idx], :]
                            ax.plot(
                                edge[:, 0],
                                edge[:, 1],
                                edge[:, 2],
                                linestyle="--",
                                linewidth=1.1,
                                color="#111827",
                                alpha=0.6,
                                zorder=3,
                            )
                    else:
                        outline = np.vstack((vertices, vertices[0]))
                        ax.plot(
                            outline[:, 0],
                            outline[:, 1],
                            outline[:, 2],
                            linestyle="--",
                            linewidth=1.1,
                            color="#111827",
                            alpha=0.6,
                            zorder=3,
                        )
                elif local_indices.size >= 3 and self._rgb_x is None and self._rgb_y is None:
                    local_model = self._rgb_model(
                        x_plot[local_indices],
                        y_plot[local_indices],
                        self._triangular_color_order,
                        fit_mode=self._rgb_fit_mode,
                    )
                    vertices = np.asarray(local_model["vertices"], dtype=float)
                    outline = np.vstack((vertices, vertices[0]))
                    ax.plot(
                        outline[:, 0],
                        outline[:, 1],
                        linestyle="--",
                        linewidth=1.1,
                        color="#111827",
                        alpha=0.6,
                        zorder=3,
                    )
            else:
                if self._is_3d:
                    scatter = ax.scatter(
                        x_plot[local_indices],
                        y_plot[local_indices],
                        z_plot[local_indices],
                        c=np.asarray(self._color[local_indices], dtype=float),
                        cmap=self._cmap,
                        norm=Normalize(vmin=vmin, vmax=vmax),
                        s=42,
                        alpha=0.9,
                        linewidths=0.2,
                        edgecolors="#111827",
                        depthshade=False,
                        zorder=2,
                    )
                else:
                    scatter = ax.scatter(
                        x_plot[local_indices],
                        y_plot[local_indices],
                        c=np.asarray(self._color[local_indices], dtype=float),
                        cmap=self._cmap,
                        norm=Normalize(vmin=vmin, vmax=vmax),
                        s=38,
                        alpha=0.9,
                        linewidths=0.2,
                        edgecolors="#111827",
                        zorder=2,
                    )
                if shared_scatter is None:
                    shared_scatter = scatter

            if self._point_artist is None:
                self._point_artist = scatter
            if not self._is_3d:
                annotation = ax.annotate(
                    "",
                    xy=(0.0, 0.0),
                    xytext=(10, 10),
                    textcoords="offset points",
                    ha="left",
                    va="bottom",
                    bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.88, "edgecolor": "#6b7280"},
                    arrowprops={"arrowstyle": "->", "color": "#6b7280", "lw": 0.8},
                )
                annotation.set_visible(False)
                try:
                    annotation.set_in_layout(False)
                except Exception:
                    pass
                self._point_artist_entries.append(
                    {
                        "axes": ax,
                        "artist": scatter,
                        "indices": np.asarray(local_indices, dtype=int),
                        "group": str(subplot_spec.get("name", "all")).strip().lower(),
                        "annotation": annotation,
                    }
                )

            title_text = self._scatter_subplot_title(subplot_spec, len(subplot_specs))
            if title_text:
                ax.set_title(title_text, fontsize=13 if len(subplot_specs) == 1 else 11)
            ax.set_xlabel(x_label, fontsize=axis_label_fontsize)
            ax.set_ylabel(y_label, fontsize=axis_label_fontsize)
            if self._is_3d:
                ax.set_zlabel(z_label, fontsize=axis_label_fontsize)
            ax.tick_params(axis="both", labelsize=tick_label_fontsize)
            if self._is_3d:
                ax.tick_params(axis="z", labelsize=tick_label_fontsize)
            ax.grid(True, alpha=0.25)
            ax.set_xlim(*self._fixed_xlim)
            ax.set_ylim(*self._fixed_ylim)
            if self._is_3d:
                ax.set_zlim(*self._fixed_zlim)
            ax.set_autoscale_on(False)
            try:
                if self._is_3d:
                    spans = np.asarray(
                        (
                            self._fixed_xlim[1] - self._fixed_xlim[0],
                            self._fixed_ylim[1] - self._fixed_ylim[0],
                            self._fixed_zlim[1] - self._fixed_zlim[0],
                        ),
                        dtype=float,
                    )
                    ax.set_box_aspect(tuple(np.maximum(spans, 1e-6).tolist()))
                else:
                    ax.set_box_aspect(1.0)
            except Exception:
                ax.set_aspect("auto")
            if not self._is_3d:
                self._restore_scatter_view_limits(
                    ax,
                    subplot_spec.get("name") if len(subplot_specs) > 1 else "all",
                    preserved_view_limits,
                )
            if self._use_triangular_rgb and not self._is_3d:
                if self._show_proximity_circles:
                    ax.text(
                        0.02,
                        0.84,
                        f"Radius: {self._proximity_radius:.4f}",
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=9,
                        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
                    )

        if not self._use_triangular_rgb and shared_scatter is not None:
            cbar = self.figure.colorbar(shared_scatter, ax=axes)
            cbar.set_label(self._color_label, fontsize=axis_label_fontsize)
            cbar.ax.tick_params(labelsize=tick_label_fontsize)
        self._sync_proximity_controls()
        self.info_label.setText(self._info_text())
        if hasattr(self, "status_label"):
            self.status_label.setText(self._status_text())
        self.canvas.draw_idle()

    def _save_figure(self):
        default_name = "gradient_scatter.png"
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save gradient scatter figure",
            str(Path.cwd() / default_name),
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)",
        )
        if not path:
            return
        output_path = Path(path)
        if output_path.suffix.lower() not in {".png", ".pdf", ".svg"}:
            if "PDF" in selected_filter:
                output_path = output_path.with_suffix(".pdf")
            elif "SVG" in selected_filter:
                output_path = output_path.with_suffix(".svg")
            else:
                output_path = output_path.with_suffix(".png")
        self.figure.savefig(str(output_path), dpi=200, bbox_inches="tight")


class GradientClassificationDialog(QDialog):
    """RGB-classified fsaverage viewer using the selected scatter axes."""

    def __init__(
        self,
        x_volume_img,
        y_volume_img,
        x_values,
        y_values,
        *,
        z_volume_img=None,
        z_values=None,
        support_mask_img=None,
        coverage_mask_img=None,
        coverage_threshold_multiplier=1.0,
        rgb_x_volume_img=None,
        rgb_y_volume_img=None,
        rgb_z_volume_img=None,
        rgb_x_values=None,
        rgb_y_values=None,
        rgb_z_values=None,
        title="Gradient Classification",
        x_label="Gradient 2",
        y_label="Gradient 1",
        z_label="",
        parent=None,
        theme_name="Dark",
        hemisphere_mode="both",
        fsaverage_mesh="fsaverage4",
        rotation_preset="Default",
        rgb_fit_mode="triangle",
        triangular_color_order="RBG",
    ):
        super().__init__(parent)
        self._x_img = nib.as_closest_canonical(x_volume_img)
        self._y_img = nib.as_closest_canonical(y_volume_img)
        self._z_img = nib.as_closest_canonical(z_volume_img) if z_volume_img is not None else None
        self._rgb_x_img = nib.as_closest_canonical(rgb_x_volume_img) if rgb_x_volume_img is not None else self._x_img
        self._rgb_y_img = nib.as_closest_canonical(rgb_y_volume_img) if rgb_y_volume_img is not None else self._y_img
        self._rgb_z_img = (
            nib.as_closest_canonical(rgb_z_volume_img)
            if rgb_z_volume_img is not None
            else self._z_img
        )
        self._support_img = None if support_mask_img is None else nib.as_closest_canonical(support_mask_img)
        self._coverage_mask_img = None if coverage_mask_img is None else nib.as_closest_canonical(coverage_mask_img)
        try:
            self._coverage_threshold_multiplier = max(0.0, float(coverage_threshold_multiplier))
        except Exception:
            self._coverage_threshold_multiplier = 1.0
        self._x_data = np.asarray(self._x_img.get_fdata(), dtype=float)
        self._y_data = np.asarray(self._y_img.get_fdata(), dtype=float)
        self._z_data = np.asarray(self._z_img.get_fdata(), dtype=float) if self._z_img is not None else None
        self._rgb_x_data = np.asarray(self._rgb_x_img.get_fdata(), dtype=float)
        self._rgb_y_data = np.asarray(self._rgb_y_img.get_fdata(), dtype=float)
        self._rgb_z_data = (
            np.asarray(self._rgb_z_img.get_fdata(), dtype=float)
            if self._rgb_z_img is not None
            else None
        )
        self._is_3d_embedding = self._z_data is not None
        if self._x_data.ndim != 3 or self._y_data.ndim != 3:
            raise ValueError("Classification requires 3D projected volumes for the selected axes.")
        if self._x_data.shape != self._y_data.shape:
            raise ValueError("Classification axis volumes must have matching shapes.")
        if self._is_3d_embedding and self._z_data.shape != self._x_data.shape:
            raise ValueError("Classification Z axis volume must match the axis volume shape.")
        if self._rgb_x_data.shape != self._x_data.shape or self._rgb_y_data.shape != self._x_data.shape:
            raise ValueError("Classification RGB source volumes must match the axis volume shape.")
        if self._is_3d_embedding and (
            self._rgb_z_data is None or self._rgb_z_data.shape != self._x_data.shape
        ):
            raise ValueError("Classification RGB Z source volume must match the axis volume shape.")
        if self._support_img is not None:
            support_data = np.asarray(self._support_img.get_fdata(), dtype=float)
            if support_data.shape != self._x_data.shape:
                raise ValueError("Classification support mask must match the axis volume shape.")
        if self._coverage_mask_img is not None:
            coverage_data = np.asarray(self._coverage_mask_img.get_fdata(), dtype=float)
            if coverage_data.shape != self._x_data.shape:
                raise ValueError("Classification coverage mask must match the axis volume shape.")

        self._x_values = np.asarray(x_values, dtype=float).reshape(-1)
        self._y_values = np.asarray(y_values, dtype=float).reshape(-1)
        if self._x_values.shape != self._y_values.shape:
            raise ValueError("Classification axis arrays must have matching lengths.")
        self._z_values = None if z_values is None else np.asarray(z_values, dtype=float).reshape(-1)
        if self._is_3d_embedding:
            if self._z_values is None or self._z_values.shape != self._x_values.shape:
                raise ValueError("Classification Z axis array must match the axis array lengths.")
        if rgb_x_values is None and rgb_y_values is None and rgb_z_values is None:
            self._rgb_x_values = np.asarray(self._x_values, dtype=float)
            self._rgb_y_values = np.asarray(self._y_values, dtype=float)
            self._rgb_z_values = np.asarray(self._z_values, dtype=float) if self._is_3d_embedding else None
        elif rgb_x_values is not None and rgb_y_values is not None and (rgb_z_values is not None or not self._is_3d_embedding):
            self._rgb_x_values = np.asarray(rgb_x_values, dtype=float).reshape(-1)
            self._rgb_y_values = np.asarray(rgb_y_values, dtype=float).reshape(-1)
            if self._rgb_x_values.shape != self._x_values.shape or self._rgb_y_values.shape != self._x_values.shape:
                raise ValueError("Classification RGB source arrays must match the axis array lengths.")
            self._rgb_z_values = None
            if self._is_3d_embedding:
                self._rgb_z_values = np.asarray(rgb_z_values, dtype=float).reshape(-1)
                if self._rgb_z_values.shape != self._x_values.shape:
                    raise ValueError("Classification RGB Z source array must match the axis array lengths.")
        else:
            raise ValueError("Classification RGB source arrays must be provided together.")
        finite_mask = (
            np.isfinite(self._x_values)
            & np.isfinite(self._y_values)
            & np.isfinite(self._rgb_x_values)
            & np.isfinite(self._rgb_y_values)
        )
        if self._is_3d_embedding:
            finite_mask &= np.isfinite(self._z_values) & np.isfinite(self._rgb_z_values)
        if not np.any(finite_mask):
            raise ValueError("Classification requires finite axis values.")
        self._x_values = self._x_values[finite_mask]
        self._y_values = self._y_values[finite_mask]
        if self._is_3d_embedding:
            self._z_values = self._z_values[finite_mask]
        self._rgb_x_values = self._rgb_x_values[finite_mask]
        self._rgb_y_values = self._rgb_y_values[finite_mask]
        if self._is_3d_embedding:
            self._rgb_z_values = self._rgb_z_values[finite_mask]
        scatter_x, scatter_y = GradientScatterDialog._rotate_points(
            self._rgb_x_values,
            self._rgb_y_values,
            GradientScatterDialog._normalize_rotation_preset(rotation_preset),
        )
        self._rgb_fit_mode = GradientScatterDialog._normalize_rgb_fit_mode(rgb_fit_mode)
        self._triangular_color_order = GradientScatterDialog._normalize_triangular_color_order(
            triangular_color_order
        )
        if self._is_3d_embedding:
            self._triangular_model = GradientScatterDialog._rgb_model_3d(
                scatter_x,
                scatter_y,
                self._rgb_z_values,
                self._triangular_color_order,
            )
        else:
            self._triangular_model = GradientScatterDialog._rgb_model(
                scatter_x,
                scatter_y,
                self._triangular_color_order,
                fit_mode=self._rgb_fit_mode,
            )

        self._title = str(title or "Gradient Classification")
        self._x_label = str(x_label or "Gradient 2")
        self._y_label = str(y_label or "Gradient 1")
        self._z_label = str(z_label or "")
        self._theme_name = "Dark"
        self._hemisphere_mode = GradientSurfaceDialog._normalize_hemisphere_mode(hemisphere_mode)
        self._fsaverage_mesh = GradientSurfaceDialog._normalize_fsaverage_mesh(fsaverage_mesh)
        self._rotation_preset = GradientScatterDialog._normalize_rotation_preset(rotation_preset)
        self.setWindowTitle(self._title)

        if self._hemisphere_mode == "both":
            fig_width, fig_height = 13.8, 9.4
        else:
            fig_width, fig_height = 10.6, 5.1
        self.figure = Figure(figsize=(15.5, 8.6), constrained_layout=True)
        self.figure.set_size_inches(fig_width, fig_height, forward=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        axis_text = (
            f"{self._x_label} / {self._y_label} / {self._z_label}"
            if self._is_3d_embedding
            else f"{self._x_label} / {self._y_label}"
        )
        self.info_label = QLabel(
            f"Axes: {axis_text} | Mesh: {self._fsaverage_mesh} | "
            f"Hemisphere: {self._hemisphere_mode.upper()} | Rotation: {self._rotation_preset} | "
            f"{'3D Pyramid' if self._is_3d_embedding else self._rgb_fit_mode.title()} {self._triangular_color_order}"
        )
        self.save_button = QPushButton("Save Figure")
        self.save_button.clicked.connect(self._save_figure)

        controls = QHBoxLayout()
        controls.addWidget(self.info_label, 1)
        controls.addWidget(self.save_button, 0)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)
        self.set_theme(theme_name)

        self._render()

    @classmethod
    def from_array(
        cls,
        x_volume_data,
        y_volume_data,
        *,
        z_volume_data=None,
        affine=None,
        x_values,
        y_values,
        z_values=None,
        support_mask_data=None,
        coverage_mask_data=None,
        coverage_threshold_multiplier=1.0,
        rgb_x_volume_data=None,
        rgb_y_volume_data=None,
        rgb_z_volume_data=None,
        rgb_x_values=None,
        rgb_y_values=None,
        rgb_z_values=None,
        title="Gradient Classification",
        x_label="Gradient 2",
        y_label="Gradient 1",
        z_label="",
        parent=None,
        theme_name="Dark",
        hemisphere_mode="both",
        fsaverage_mesh="fsaverage4",
        rotation_preset="Default",
        rgb_fit_mode="triangle",
        triangular_color_order="RBG",
    ):
        x_arr = np.asarray(x_volume_data, dtype=float)
        y_arr = np.asarray(y_volume_data, dtype=float)
        z_arr = None if z_volume_data is None else np.asarray(z_volume_data, dtype=float)
        if x_arr.ndim != 3 or y_arr.ndim != 3:
            raise ValueError(
                f"Expected 3D arrays for classification axes. Got shapes {x_arr.shape} and {y_arr.shape}."
            )
        if x_arr.shape != y_arr.shape:
            raise ValueError("Classification axis arrays must have matching shapes.")
        if z_arr is not None and (z_arr.ndim != 3 or z_arr.shape != x_arr.shape):
            raise ValueError("Classification Z axis array must match the axis volume shape.")
        if affine is None:
            affine = np.eye(4)
        x_img = nib.Nifti1Image(x_arr, affine)
        y_img = nib.Nifti1Image(y_arr, affine)
        z_img = nib.Nifti1Image(z_arr, affine) if z_arr is not None else None
        support_img = None
        if support_mask_data is not None:
            support_arr = np.asarray(support_mask_data, dtype=float)
            if support_arr.shape != x_arr.shape:
                raise ValueError("Classification support mask must match the axis volume shape.")
            support_img = nib.Nifti1Image(support_arr, affine)
        coverage_img = None
        if coverage_mask_data is not None:
            coverage_arr = np.asarray(coverage_mask_data, dtype=float)
            if coverage_arr.shape != x_arr.shape:
                raise ValueError("Classification coverage mask must match the axis volume shape.")
            coverage_img = nib.Nifti1Image(coverage_arr, affine)
        rgb_x_img = None
        rgb_y_img = None
        rgb_z_img = None
        if rgb_x_volume_data is not None or rgb_y_volume_data is not None or rgb_z_volume_data is not None:
            if rgb_x_volume_data is None or rgb_y_volume_data is None:
                raise ValueError("Both classification RGB source volumes are required.")
            if z_arr is not None and rgb_z_volume_data is None:
                raise ValueError("Classification RGB Z source volume is required for 3D classification.")
            rgb_x_arr = np.asarray(rgb_x_volume_data, dtype=float)
            rgb_y_arr = np.asarray(rgb_y_volume_data, dtype=float)
            if rgb_x_arr.shape != x_arr.shape or rgb_y_arr.shape != x_arr.shape:
                raise ValueError("Classification RGB source volumes must match the axis volume shape.")
            rgb_x_img = nib.Nifti1Image(rgb_x_arr, affine)
            rgb_y_img = nib.Nifti1Image(rgb_y_arr, affine)
            if z_arr is not None:
                rgb_z_arr = np.asarray(rgb_z_volume_data, dtype=float)
                if rgb_z_arr.shape != x_arr.shape:
                    raise ValueError("Classification RGB Z source volume must match the axis volume shape.")
                rgb_z_img = nib.Nifti1Image(rgb_z_arr, affine)
        return cls(
            x_img,
            y_img,
            x_values,
            y_values,
            z_volume_img=z_img,
            z_values=z_values,
            support_mask_img=support_img,
            coverage_mask_img=coverage_img,
            coverage_threshold_multiplier=coverage_threshold_multiplier,
            rgb_x_volume_img=rgb_x_img,
            rgb_y_volume_img=rgb_y_img,
            rgb_z_volume_img=rgb_z_img,
            rgb_x_values=rgb_x_values,
            rgb_y_values=rgb_y_values,
            rgb_z_values=rgb_z_values,
            title=title,
            x_label=x_label,
            y_label=y_label,
            z_label=z_label,
            parent=parent,
            theme_name=theme_name,
            hemisphere_mode=hemisphere_mode,
            fsaverage_mesh=fsaverage_mesh,
            rotation_preset=rotation_preset,
            rgb_fit_mode=rgb_fit_mode,
            triangular_color_order=triangular_color_order,
        )

    def set_theme(self, theme_name="Dark"):
        theme, style = _dialog_theme_stylesheet(theme_name)
        self._theme_name = theme
        self.setStyleSheet(style)

    def _surface_views_layout(self):
        assets = GradientSurfaceDialog._get_surface_assets(self._fsaverage_mesh)
        if self._hemisphere_mode == "lh":
            return [
                [
                    ("left", "medial", assets["mesh_left"], assets["sulc_left"], "LH Medial"),
                ],
                [
                    ("left", "lateral", assets["mesh_left"], assets["sulc_left"], "LH Lateral"),
                ],
            ]
        if self._hemisphere_mode == "rh":
            return [
                [
                    ("right", "medial", assets["mesh_right"], assets["sulc_right"], "RH Medial"),
                ],
                [
                    ("right", "lateral", assets["mesh_right"], assets["sulc_right"], "RH Lateral"),
                ],
            ]
        return [
            [
                ("left", "medial", assets["mesh_left"], assets["sulc_left"], "LH Medial"),
                ("right", "medial", assets["mesh_right"], assets["sulc_right"], "RH Medial"),
            ],
            [
                ("left", "lateral", assets["mesh_left"], assets["sulc_left"], "LH Lateral"),
                ("right", "lateral", assets["mesh_right"], assets["sulc_right"], "RH Lateral"),
            ],
        ]

    @staticmethod
    def _mesh_arrays(mesh):
        if hasattr(mesh, "coordinates") and hasattr(mesh, "faces"):
            return np.asarray(mesh.coordinates, dtype=float), np.asarray(mesh.faces, dtype=int)
        if isinstance(mesh, (tuple, list)) and len(mesh) >= 2:
            return np.asarray(mesh[0], dtype=float), np.asarray(mesh[1], dtype=int)
        raise ValueError("Unsupported surface mesh format.")

    @staticmethod
    def _view_angles(hemi, view):
        mapping = {
            ("left", "lateral"): (0.0, 180.0),
            ("left", "medial"): (0.0, 0.0),
            ("right", "lateral"): (0.0, 0.0),
            ("right", "medial"): (0.0, 180.0),
        }
        return mapping.get((hemi, view), (0.0, 180.0))

    @staticmethod
    def _background_face_gray(bg_map, faces):
        n_faces = int(np.asarray(faces).shape[0])
        if bg_map is None:
            return np.full((n_faces, 3), 0.72, dtype=float)
        bg = np.asarray(bg_map, dtype=float).reshape(-1)
        if bg.size < np.max(faces) + 1:
            return np.full((n_faces, 3), 0.72, dtype=float)
        bg_finite = bg[np.isfinite(bg)]
        if bg_finite.size == 0:
            return np.full((n_faces, 3), 0.72, dtype=float)
        bg_min = float(np.nanmin(bg_finite))
        bg_max = float(np.nanmax(bg_finite))
        if np.isclose(bg_min, bg_max):
            bg_norm = np.full(bg.shape, 0.72, dtype=float)
        else:
            bg_norm = 0.55 + 0.35 * ((bg - bg_min) / (bg_max - bg_min))
        face_gray = np.asarray(bg_norm[faces].mean(axis=1), dtype=float)
        return np.repeat(face_gray[:, np.newaxis], 3, axis=1)

    def _plot_rgb_surface(self, ax, mesh, vertex_colors, vertex_alpha, bg_map, *, hemi, view, title):
        coords, faces = self._mesh_arrays(mesh)
        bg_face_rgb = self._background_face_gray(bg_map, faces)

        base = ax.plot_trisurf(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            triangles=faces,
            linewidth=0.0,
            antialiased=False,
            shade=False,
        )
        base_rgba = np.concatenate((bg_face_rgb, np.ones((bg_face_rgb.shape[0], 1), dtype=float)), axis=1)
        base.set_facecolors(base_rgba)
        base.set_edgecolors("none")

        overlay = ax.plot_trisurf(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            triangles=faces,
            linewidth=0.0,
            antialiased=False,
            shade=False,
        )
        face_rgb = np.asarray(vertex_colors[faces].mean(axis=1), dtype=float)
        face_rgb = np.clip(0.92 * face_rgb + 0.08 * bg_face_rgb, 0.0, 1.0)
        face_alpha = np.asarray(vertex_alpha[faces].mean(axis=1), dtype=float)
        face_alpha = np.clip(face_alpha, 0.0, 1.0)
        overlay_rgba = np.concatenate((face_rgb, face_alpha[:, np.newaxis]), axis=1)
        overlay.set_facecolors(overlay_rgba)
        overlay.set_edgecolors("none")
        elev, azim = self._view_angles(hemi, view)
        ax.view_init(elev=elev, azim=azim)
        try:
            ax.set_proj_type("ortho")
        except Exception:
            pass
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        span = np.maximum(maxs - mins, 1e-6)
        pad = 0.04 * span
        ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
        ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])
        ax.set_zlim(mins[2] - pad[2], maxs[2] + pad[2])
        try:
            ax.set_box_aspect(tuple(span.tolist()))
        except Exception:
            pass
        ax.set_axis_off()
        ax.set_title(title, fontsize=10, pad=4)

    def _surface_vertex_colors(self, x_img, y_img, mesh):
        x_texture = surface.vol_to_surf(
            x_img,
            mesh,
            radius=0.5,
            interpolation="linear",
        )
        y_texture = surface.vol_to_surf(
            y_img,
            mesh,
            radius=0.5,
            interpolation="linear",
        )
        rgb_x_texture = surface.vol_to_surf(
            self._rgb_x_img,
            mesh,
            radius=0.5,
            interpolation="linear",
        )
        rgb_y_texture = surface.vol_to_surf(
            self._rgb_y_img,
            mesh,
            radius=0.5,
            interpolation="linear",
        )
        z_texture = None
        rgb_z_texture = None
        if self._is_3d_embedding:
            z_texture = surface.vol_to_surf(
                self._z_img,
                mesh,
                radius=0.5,
                interpolation="linear",
            )
            rgb_z_texture = surface.vol_to_surf(
                self._rgb_z_img,
                mesh,
                radius=0.5,
                interpolation="linear",
            )
        if self._support_img is not None:
            support_texture = surface.vol_to_surf(
                self._support_img,
                mesh,
                radius=0.5,
                interpolation="nearest_most_frequent",
            )
            vertex_alpha = np.clip(np.asarray(support_texture, dtype=float), 0.0, 1.0)
        else:
            vertex_alpha = np.asarray(np.isfinite(x_texture) & np.isfinite(y_texture), dtype=float)
            if self._is_3d_embedding:
                vertex_alpha = vertex_alpha * np.asarray(np.isfinite(z_texture), dtype=float)
        vertex_alpha = vertex_alpha * np.asarray(
            np.isfinite(x_texture)
            & np.isfinite(y_texture)
            & np.isfinite(rgb_x_texture)
            & np.isfinite(rgb_y_texture),
            dtype=float,
        )
        if self._is_3d_embedding:
            vertex_alpha = vertex_alpha * np.asarray(
                np.isfinite(z_texture) & np.isfinite(rgb_z_texture),
                dtype=float,
            )
        if self._coverage_mask_img is not None:
            coverage_texture = surface.vol_to_surf(
                self._coverage_mask_img,
                mesh,
                radius=0.5,
                interpolation="linear",
            )
            _vmin, _vmax, threshold = GradientSurfaceDialog._compute_display_range(
                np.asarray(self._coverage_mask_img.get_fdata(), dtype=float)
            )
            coverage_mask = np.isfinite(coverage_texture)
            if threshold is not None:
                threshold = float(threshold) * float(self._coverage_threshold_multiplier)
                coverage_mask &= np.abs(np.asarray(coverage_texture, dtype=float)) >= threshold
            vertex_alpha = vertex_alpha * np.asarray(coverage_mask, dtype=float)
        rgb_x_rot, rgb_y_rot = GradientScatterDialog._rotate_points(
            np.asarray(rgb_x_texture, dtype=float),
            np.asarray(rgb_y_texture, dtype=float),
            self._rotation_preset,
        )
        if self._is_3d_embedding:
            vertex_colors = GradientScatterDialog._rgb_colors_from_model_3d(
                rgb_x_rot,
                rgb_y_rot,
                np.asarray(rgb_z_texture, dtype=float),
                self._triangular_model,
            )
        else:
            vertex_colors = GradientScatterDialog._rgb_colors_from_model(
                rgb_x_rot,
                rgb_y_rot,
                self._triangular_model,
            )
        return vertex_colors, vertex_alpha

    def _render(self):
        self.figure.clear()
        view_rows = self._surface_views_layout()
        n_rows = len(view_rows)
        n_cols = max(len(row) for row in view_rows)
        gs = self.figure.add_gridspec(n_rows, n_cols, wspace=0.02, hspace=0.08)

        for row_idx, row_views in enumerate(view_rows):
            for col_idx, (hemi, view, mesh, bg_map, title) in enumerate(row_views):
                ax = self.figure.add_subplot(gs[row_idx, col_idx], projection="3d")
                vertex_colors, vertex_alpha = self._surface_vertex_colors(self._x_img, self._y_img, mesh)
                self._plot_rgb_surface(
                    ax,
                    mesh,
                    vertex_colors,
                    vertex_alpha,
                    bg_map,
                    hemi=hemi,
                    view=view,
                    title=title,
                )

        self.figure.suptitle(self._title, fontsize=15)
        self.canvas.draw_idle()

    def _save_figure(self):
        default_name = "gradient_classification.png"
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save gradient classification figure",
            str(Path.cwd() / default_name),
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)",
        )
        if not path:
            return
        output_path = Path(path)
        if output_path.suffix.lower() not in {".png", ".pdf", ".svg"}:
            if "PDF" in selected_filter:
                output_path = output_path.with_suffix(".pdf")
            elif "SVG" in selected_filter:
                output_path = output_path.with_suffix(".svg")
            else:
                output_path = output_path.with_suffix(".png")
        self.figure.savefig(str(output_path), dpi=200, bbox_inches="tight")


MSModeSurfaceDialog = GradientSurfaceDialog


__all__ = [
    "GradientSurfaceDialog",
    "GradientScatterDialog",
    "GradientClassificationDialog",
    "MSModeSurfaceDialog",
]
