"""Compute metabolic connectivity matrices from MRSIPrep metabolic-profile files.

Opened when a dropped folder contains ``*_desc-metabolicprofiles_mrsi.npz``
files, mirroring the batch-matrix import dialog: the same filter-and-check list
for choosing what to include, followed by the computation parameters and a
progress bar per selected profile.

All numerics live in :mod:`services.connectivity_compute`; this module is
presentation and threading only.
"""

from __future__ import annotations

import os
import traceback
from pathlib import Path

try:
    from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal
    from PyQt6.QtWidgets import (
        QAbstractItemView,
        QCheckBox,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )
except Exception:  # PyQt5 fallback, matching the rest of the app
    from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal
    from PyQt5.QtWidgets import (
        QAbstractItemView,
        QCheckBox,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QFormLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )

try:
    from services.connectivity_compute import (
        CORRELATION_MODES,
        PROFILE_MARKER,
        MODE_AUGMENTED,
        MODE_PER_PERTURBATION,
        ComputeParams,
        compute_profile,
        output_path_for,
        read_profile_info,
    )
except Exception:  # installed-package layout
    from mrsi_viewer.services.connectivity_compute import (  # type: ignore
        CORRELATION_MODES,
        PROFILE_MARKER,
        MODE_AUGMENTED,
        MODE_PER_PERTURBATION,
        ComputeParams,
        compute_profile,
        output_path_for,
        read_profile_info,
    )


try:
    from window.icon_tile import SquareIconTile
except Exception:
    from mrsi_viewer.window.icon_tile import SquareIconTile


class ConnectivityDropButton(SquareIconTile):
    """Entry point for the compute-connectivity workflow.

    Exists as its own button because a folder produced by MRSIPrep usually
    holds *both* the computed matrices and the profiles they came from, so a
    plain drop on the main window is ambiguous and resolves to batch import.
    Dropping here states the intent, and the button filters the folder down to
    metabolic-profile files itself.
    """

    folder_dropped = pyqtSignal(str)

    def __init__(self, text="Connectivity", parent=None):
        super().__init__(text, parent)
        self.setAcceptDrops(True)
        self.setToolTip(
            "Compute metabolic connectivity from MRSIPrep profiles.\n"
            "Click to choose a folder, or drop one here.\n"
            "Only *_desc-metabolicprofiles_mrsi.npz files are considered."
        )

    def _dropped_folder(self, event):
        if not event.mimeData().hasUrls():
            return None
        for url in event.mimeData().urls():
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile())
            if path.is_dir():
                return path
            # A profile file dropped directly: use its folder.
            if path.suffix.lower() == ".npz" and PROFILE_MARKER in path.name.lower():
                return path.parent
        return None

    def dragEnterEvent(self, event):
        if self._dropped_folder(event) is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        folder = self._dropped_folder(event)
        if folder is None:
            event.ignore()
            return
        event.acceptProposedAction()
        self.folder_dropped.emit(str(folder))


def _checked_state():
    return Qt.CheckState.Checked if hasattr(Qt, "CheckState") else Qt.Checked


def _unchecked_state():
    return Qt.CheckState.Unchecked if hasattr(Qt, "CheckState") else Qt.Unchecked


def _user_role():
    return Qt.ItemDataRole.UserRole if hasattr(Qt, "ItemDataRole") else Qt.UserRole


def _item_flags():
    if hasattr(Qt, "ItemFlag"):
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable
    return Qt.ItemIsEnabled | Qt.ItemIsUserCheckable


class _ComputeWorker(QObject):
    """Runs one profile. One worker per selected file, so each owns a bar."""

    progress = pyqtSignal(int, int, int)  # index, done, total
    finished = pyqtSignal(int, object, str)  # index, result-or-None, error text

    def __init__(self, index: int, path: Path, params: ComputeParams):
        super().__init__()
        self._index = index
        self._path = Path(path)
        self._params = params

    def run(self) -> None:
        try:
            result = compute_profile(
                self._path,
                self._params,
                progress=lambda done, total: self.progress.emit(self._index, done, total),
            )
            self.finished.emit(self._index, result, "")
        except Exception as exc:
            traceback.print_exc()
            self.finished.emit(self._index, None, str(exc))


class _ProfileRow(QWidget):
    """One progress bar plus status for a single profile."""

    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.name_label = QLabel(label)
        self.name_label.setMinimumWidth(320)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.status_label = QLabel("queued")
        self.status_label.setMinimumWidth(120)
        layout.addWidget(self.name_label, 3)
        layout.addWidget(self.bar, 4)
        layout.addWidget(self.status_label, 1)

    def set_progress(self, done: int, total: int) -> None:
        self.bar.setRange(0, max(int(total), 1))
        self.bar.setValue(int(done))
        self.status_label.setText(f"{done}/{total}")

    def set_done(self, text: str, ok: bool = True) -> None:
        if ok:
            self.bar.setValue(self.bar.maximum())
        self.status_label.setText(text)


class ComputeConnectivityDialog(QDialog):
    """Select metabolic profiles, set parameters, compute one matrix each."""

    def __init__(self, folder_path: Path, candidate_paths, parent=None) -> None:
        super().__init__(parent)
        self._folder_path = Path(folder_path)
        self._infos = [read_profile_info(path) for path in candidate_paths]
        self._rows = {}
        self._threads = []
        self._pending = []
        self._active = 0
        self._results = []
        self._errors = []
        self._running = False
        self.setWindowTitle("Compute Connectivity")
        self.resize(1000, 700)
        self._build_ui()
        self._populate()
        self._apply_filters()
        self.set_theme(getattr(parent, "_theme_name", "Dark"))

    # ---------------------------------------------------------------- UI ----
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        header = QLabel(
            f"Metabolic profiles found under <b>{self._folder_path.name}</b>. "
            "Each selected profile produces one connectivity matrix, written beside it."
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filter:"))
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("comma-separated substrings, e.g. sub-01, atlas-synthseg")
        self.filter_edit.textChanged.connect(self._apply_filters)
        filter_row.addWidget(self.filter_edit, 1)
        self.atlas_combo = QComboBox()
        self.atlas_combo.addItem("All atlases", "")
        for atlas in sorted({info.atlas for info in self._infos if info.atlas}):
            self.atlas_combo.addItem(atlas, atlas)
        self.atlas_combo.currentIndexChanged.connect(self._apply_filters)
        filter_row.addWidget(self.atlas_combo)
        select_all = QPushButton("Select shown")
        select_all.clicked.connect(lambda: self._set_visible_checked(True))
        clear_all = QPushButton("Clear shown")
        clear_all.clicked.connect(lambda: self._set_visible_checked(False))
        filter_row.addWidget(select_all)
        filter_row.addWidget(clear_all)
        layout.addLayout(filter_row)

        self.file_list = QListWidget()
        self.file_list.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection
            if hasattr(QAbstractItemView, "SelectionMode")
            else QAbstractItemView.NoSelection
        )
        self.file_list.itemChanged.connect(lambda _item: self._apply_filters())
        layout.addWidget(self.file_list, 3)

        self.summary_label = QLabel("")
        layout.addWidget(self.summary_label)

        params_box = QGroupBox("Computation")
        form = QFormLayout(params_box)

        self.correlation_combo = QComboBox()
        for mode in CORRELATION_MODES:
            self.correlation_combo.addItem(mode, mode)
        form.addRow("Correlation", self.correlation_combo)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Correlate the npert-augmented profile (one pass)", MODE_AUGMENTED)
        self.mode_combo.addItem("Correlate each perturbation, then average (mean + std)", MODE_PER_PERTURBATION)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        form.addRow("Mode", self.mode_combo)

        self.nproc_spin = QSpinBox()
        self.nproc_spin.setRange(1, max(1, (os.cpu_count() or 4)))
        self.nproc_spin.setValue(min(4, max(1, (os.cpu_count() or 4))))
        form.addRow("Profiles in parallel", self.nproc_spin)

        self.overwrite_check = QCheckBox("Overwrite existing matrices")
        form.addRow("", self.overwrite_check)

        self.mode_hint = QLabel("")
        self.mode_hint.setWordWrap(True)
        form.addRow("", self.mode_hint)
        layout.addWidget(params_box)

        self.progress_box = QGroupBox("Progress")
        progress_outer = QVBoxLayout(self.progress_box)
        self.progress_area = QScrollArea()
        self.progress_area.setWidgetResizable(True)
        self.progress_host = QWidget()
        self.progress_layout = QVBoxLayout(self.progress_host)
        self.progress_layout.addStretch(1)
        self.progress_area.setWidget(self.progress_host)
        progress_outer.addWidget(self.progress_area)
        self.progress_box.setVisible(False)
        layout.addWidget(self.progress_box, 2)

        self.buttons = QDialogButtonBox()
        self.run_button = self.buttons.addButton(
            "Compute",
            QDialogButtonBox.ButtonRole.AcceptRole if hasattr(QDialogButtonBox, "ButtonRole") else QDialogButtonBox.AcceptRole,
        )
        self.close_button = self.buttons.addButton(
            QDialogButtonBox.StandardButton.Close if hasattr(QDialogButtonBox, "StandardButton") else QDialogButtonBox.Close
        )
        self.run_button.clicked.connect(self._start)
        self.close_button.clicked.connect(self.reject)
        layout.addWidget(self.buttons)

        self._on_mode_changed()

    def set_theme(self, _name) -> None:
        """Themes are applied app-wide by the parent; hook kept for parity."""

    # ----------------------------------------------------------- listing ----
    def _populate(self) -> None:
        self.file_list.blockSignals(True)
        self.file_list.clear()
        for info in self._infos:
            bits = [info.label(self._folder_path)]
            if info.ok:
                detail = f"{info.n_parcels} parcels"
                if info.metabolites:
                    detail += f", {len(info.metabolites)} metabolites"
                if info.npert:
                    detail += f", npert={info.npert}"
                bits.append(f"  [{detail}]")
            else:
                bits.append(f"  [unreadable: {info.error}]")
            item = QListWidgetItem("".join(bits))
            item.setToolTip(str(info.path))
            item.setData(_user_role(), str(info.path))
            if info.ok:
                item.setFlags(item.flags() | _item_flags())
                item.setCheckState(_checked_state())
            else:
                item.setFlags(_item_flags() & ~_item_flags())
            self.file_list.addItem(item)
        self.file_list.blockSignals(False)

    def _filter_tokens(self):
        return [t.strip().lower() for t in self.filter_edit.text().split(",") if t.strip()]

    def _matches(self, info) -> bool:
        atlas = str(self.atlas_combo.currentData() or "")
        if atlas and info.atlas != atlas:
            return False
        name = info.label(self._folder_path).lower()
        return all(token in name for token in self._filter_tokens())

    def _apply_filters(self) -> None:
        visible = 0
        for row in range(self.file_list.count()):
            item = self.file_list.item(row)
            if item is None:
                continue
            info = self._infos[row]
            shown = self._matches(info)
            item.setHidden(not shown)
            visible += 1 if shown else 0
        selected = len(self.selected_infos())
        unreadable = sum(1 for info in self._infos if not info.ok)
        text = f"Showing {visible} of {len(self._infos)} profiles. Selected: {selected}."
        if unreadable:
            text += f"  ({unreadable} unreadable, not selectable)"
        self.summary_label.setText(text)
        self.run_button.setEnabled(selected > 0 and not self._running)

    def _set_visible_checked(self, checked: bool) -> None:
        self.file_list.blockSignals(True)
        for row in range(self.file_list.count()):
            item = self.file_list.item(row)
            if item is None or item.isHidden() or not self._infos[row].ok:
                continue
            item.setCheckState(_checked_state() if checked else _unchecked_state())
        self.file_list.blockSignals(False)
        self._apply_filters()

    def selected_infos(self):
        chosen = []
        for row in range(self.file_list.count()):
            item = self.file_list.item(row)
            if item is None or not self._infos[row].ok:
                continue
            if item.checkState() == _checked_state():
                chosen.append(self._infos[row])
        return chosen

    def _on_mode_changed(self) -> None:
        mode = self.mode_combo.currentData()
        if mode == MODE_PER_PERTURBATION:
            without = [info for info in self.selected_infos() if not info.supports_perturbation]
            hint = (
                "One matrix per perturbation, averaged; the per-edge standard deviation is saved "
                "alongside as <tt>matrix_std</tt>."
            )
            if without:
                hint += f"  <b>{len(without)} selected profile(s) have no perturbations and will be skipped.</b>"
            self.mode_hint.setText(hint)
        else:
            self.mode_hint.setText(
                "One correlation pass over the full npert-augmented feature vector. Fastest, and "
                "what MRSIPrep itself writes."
            )

    # ------------------------------------------------------------- run ----
    def _params(self) -> ComputeParams:
        return ComputeParams(
            correlation=str(self.correlation_combo.currentData()),
            mode=str(self.mode_combo.currentData()),
            n_parallel=int(self.nproc_spin.value()),
            overwrite=bool(self.overwrite_check.isChecked()),
        )

    def _start(self) -> None:
        if self._running:
            return
        params = self._params()
        selected = self.selected_infos()
        if params.mode == MODE_PER_PERTURBATION:
            selected = [info for info in selected if info.supports_perturbation]
        if not selected:
            self.summary_label.setText("Nothing to compute: no selected profile supports the chosen mode.")
            return

        self._running = True
        self._results = []
        self._errors = []
        self._rows = {}
        self._threads = []
        self.run_button.setEnabled(False)
        self.progress_box.setVisible(True)

        while self.progress_layout.count() > 1:
            item = self.progress_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        for index, info in enumerate(selected):
            row = _ProfileRow(info.label(self._folder_path))
            self.progress_layout.insertWidget(self.progress_layout.count() - 1, row)
            self._rows[index] = row

        # One bar per profile is only honest if the work is actually concurrent,
        # so the queue is drained up to n_parallel at a time rather than all at
        # once -- otherwise every bar would appear to start and stall together.
        self._pending = list(enumerate(selected))
        self._active = 0
        self._pump(params)

    def _pump(self, params: ComputeParams) -> None:
        while self._pending and self._active < params.n_parallel:
            index, info = self._pending.pop(0)
            self._launch(index, info, params)
        if not self._pending and self._active == 0:
            self._finish()

    def _launch(self, index: int, info, params: ComputeParams) -> None:
        thread = QThread(self)
        worker = _ComputeWorker(index, info.path, params)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_progress)
        worker.finished.connect(lambda i, r, e, p=params: self._on_finished(i, r, e, p))
        worker.finished.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        self._threads.append((thread, worker))
        self._active += 1
        row = self._rows.get(index)
        if row is not None:
            row.status_label.setText("running")
        thread.start()

    def _on_progress(self, index: int, done: int, total: int) -> None:
        row = self._rows.get(index)
        if row is not None:
            row.set_progress(done, total)

    def _on_finished(self, index: int, result, error: str, params: ComputeParams) -> None:
        row = self._rows.get(index)
        if error:
            self._errors.append(error)
            if row is not None:
                row.set_done("failed", ok=False)
        elif result is not None and result.get("skipped"):
            if row is not None:
                row.set_done("exists, skipped")
        else:
            self._results.append(result)
            if row is not None:
                row.set_done("done")
        self._active -= 1
        self._pump(params)

    def _finish(self) -> None:
        self._running = False
        self.run_button.setEnabled(True)
        parts = [f"{len(self._results)} matrices written"]
        if self._errors:
            parts.append(f"{len(self._errors)} failed")
        self.summary_label.setText(". ".join(parts) + ".")
        if self._results:
            self.accept()

    def output_paths(self):
        """Matrices written by the last run, for the caller to load."""
        return [Path(result["output"]) for result in self._results if result.get("output")]

    def reject(self) -> None:
        for thread, _worker in self._threads:
            if thread.isRunning():
                thread.quit()
                thread.wait(2000)
        super().reject()
