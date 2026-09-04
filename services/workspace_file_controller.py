#!/usr/bin/env python3
"""Controller for workspace file import, batch scan, and stack import flows."""

from __future__ import annotations

import os
from pathlib import Path
from types import MethodType

import numpy as np

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QDialog, QFileDialog, QMessageBox
except ImportError:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QDialog, QFileDialog, QMessageBox


def _dialog_accepted_code():
    accepted = getattr(QDialog, "Accepted", None)
    if accepted is not None:
        return accepted
    dialog_code = getattr(QDialog, "DialogCode", None)
    if dialog_code is not None and hasattr(dialog_code, "Accepted"):
        return dialog_code.Accepted
    return 1


def _user_role():
    return getattr(Qt, "UserRole", getattr(Qt.ItemDataRole, "UserRole"))


USER_ROLE = _user_role()


class WorkspaceFileController:
    """Owns viewer file import/remove/clear orchestration."""

    BOUND_METHOD_NAMES = (
        '_import_harmonized_result',
        '_import_stacked_result',
        '_open_stack_prepare_dialog',
        '_import_selector_aggregate',
        '_open_files',
        '_batch_connectivity_paths',
        '_open_batch_import_dialog',
        '_metabolic_profile_paths',
        '_open_compute_connectivity_dialog',
        '_open_connectivity_compute_folder',
        '_open_batch_folder',
        '_add_files',
        '_remove_selected',
        '_clear_files',
    )

    def __init__(self, viewer) -> None:
        object.__setattr__(self, '_viewer', viewer)

    def __getattr__(self, name):
        return getattr(self._viewer, name)

    def __setattr__(self, name, value):
        if name == '_viewer':
            object.__setattr__(self, name, value)
            return
        setattr(self._viewer, name, value)

    def bind_viewer_methods(self) -> None:
        owner = type(self)
        for name in self.BOUND_METHOD_NAMES:
            descriptor = owner.__dict__.get(name)
            if isinstance(descriptor, staticmethod):
                setattr(self._viewer, name, descriptor.__func__)
                continue
            if descriptor is None:
                setattr(self._viewer, name, getattr(self, name))
                continue
            setattr(self._viewer, name, MethodType(descriptor, self._viewer))

    def _import_harmonized_result(self, payload) -> bool:
        output_raw = str(payload.get("output_path") or "").strip()
        if not output_raw:
            self.statusBar().showMessage("Harmonize export payload missing output path.")
            return False
        output_path = Path(output_raw)
        if not output_path.is_file():
            self.statusBar().showMessage(f"Harmonized file not found: {output_path}")
            return False

        self._invalidate_path_caches(output_path)
        self._add_files([str(output_path)])
        target_id = self._file_entry_id(output_path)
        selected = self._select_workspace_entry(target_id)
        if not selected:
            self.statusBar().showMessage(
                f"Harmonized file saved to {output_path.name} (already in workspace)."
            )
        else:
            self.statusBar().showMessage(
                f"Imported harmonized matrix stack: {output_path.name}."
            )
        return True

    def _import_stacked_result(self, payload) -> bool:
        output_raw = str(payload.get("output_path") or "").strip()
        if not output_raw:
            self.statusBar().showMessage("Stack export payload missing output path.")
            return False
        output_path = Path(output_raw)
        if not output_path.is_file():
            self.statusBar().showMessage(f"Stacked file not found: {output_path}")
            return False

        self._invalidate_path_caches(output_path)
        self._add_files([str(output_path)])
        target_id = self._file_entry_id(output_path)
        if self._select_workspace_entry(target_id):
            self.statusBar().showMessage(f"Imported stacked matrix file: {output_path.name}.")
            return True
        self.statusBar().showMessage(f"Stacked matrix saved to {output_path.name}.")
        return True

    def _open_stack_prepare_dialog(self, selected_paths) -> None:
        paths = [str(path) for path in (selected_paths or []) if str(path).strip()]
        if not paths:
            self.statusBar().showMessage("Stack requires at least one selected matrix file.")
            return

        try:
            from window.stack_prepare import StackPrepareDialog
        except Exception:
            try:
                from mrsi_viewer.window.stack_prepare import StackPrepareDialog
            except Exception as exc:
                self.statusBar().showMessage(f"Failed to open Stack window: {exc}")
                return

        self._stack_prepare_dialog = StackPrepareDialog(
            selected_paths=paths,
            theme_name=self._theme_name,
            export_callback=self._import_stacked_result,
            default_results_dir=self._results_dir_default,
            default_bids_dir=self._bids_dir_default,
            default_atlas_dir=self._atlas_dir_default,
            parent=self,
        )
        self._stack_prepare_dialog.show()
        self.statusBar().showMessage(f"Opened Stack Prepare ({len(paths)} selected files).")

    def _import_selector_aggregate(self, payload) -> bool:
        try:
            matrix = np.asarray(payload.get("matrix"), dtype=float)
        except Exception as exc:
            self.statusBar().showMessage(f"Invalid aggregated matrix payload: {exc}")
            return False
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            self.statusBar().showMessage("Aggregated matrix must be square.")
            return False

        source_path_raw = str(payload.get("source_path", "")).strip()
        source_path = Path(source_path_raw) if source_path_raw else None
        source_name = source_path.name if source_path is not None else "matrix"
        matrix_key = str(payload.get("matrix_key") or "matrix")
        method = str(payload.get("method") or "mean").strip().lower()
        method_label = "zfisher" if method == "zfisher" else "avg"
        selected_rows = list(payload.get("selected_rows") or [])
        n_selected = len(selected_rows)
        n_total = int(payload.get("n_total_rows") or n_selected)
        filter_covar = str(payload.get("filter_covar") or "").strip()
        filter_values = [str(value) for value in (payload.get("filter_values") or []) if str(value) != ""]

        if filter_covar and filter_values:
            filter_text = f"{filter_covar}={','.join(filter_values)}"
        else:
            filter_text = f"n={n_selected}/{n_total}"
        label = f"{method_label} {matrix_key} [{filter_text}] ({source_name})"

        _derived_id, entry = self._workspace.add_derived_entry(
            matrix,
            label=label,
            source_path=source_path,
            selected_key=matrix_key,
            sample_index=None,
            extra_fields={
                "aggregation_method": method,
                "selected_rows": selected_rows,
                "covar_name": filter_covar or None,
                "covar_value": ",".join(filter_values) if filter_values else None,
            },
        )
        self._add_workspace_list_item(entry, select=True)
        self.statusBar().showMessage(f"Imported aggregated matrix ({method_label}).")
        return True

    def _open_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select .npz files",
            str(self._default_open_dir()),
            "NumPy archives (*.npz);;All files (*)",
        )
        self._add_files(paths)

    def _batch_connectivity_paths(self, folder_path: Path):
        folder_path = Path(folder_path)
        candidates = []
        try:
            for root, dirs, files in os.walk(folder_path, followlinks=False):
                dirs.sort(key=str.lower)
                files.sort(key=str.lower)
                root_path = Path(root)
                for filename in files:
                    name = filename.lower()
                    if not name.endswith(".npz"):
                        continue
                    if "connectivity" not in name:
                        continue
                    candidates.append(root_path / filename)
        except Exception:
            return candidates
        return sorted(candidates, key=lambda path: str(path.relative_to(folder_path)).lower())

    def _metabolic_profile_paths(self, folder_path: Path):
        """MRSIPrep metabolic-profile npz files under a dropped folder.

        Kept separate from _batch_connectivity_paths because the two feed
        different dialogs: profiles are an input to be computed from, matrices
        are ready to import.
        """
        try:
            from services.connectivity_compute import find_profile_files
        except Exception:
            from mrsi_viewer.services.connectivity_compute import find_profile_files
        try:
            return find_profile_files(Path(folder_path))
        except Exception:
            return []

    def _open_connectivity_compute_folder(self) -> None:
        """Folder chooser behind the Connectivity button and menu entry."""
        selected_dir = QFileDialog.getExistingDirectory(
            self,
            "Select folder with MRSIPrep metabolic profiles",
            str(self._default_open_dir()),
        )
        if not selected_dir:
            return
        self._open_compute_connectivity_dialog(Path(selected_dir))

    def _open_compute_connectivity_dialog(self, folder_path: Path, candidate_paths=None):
        folder_path = Path(folder_path)
        candidate_paths = list(candidate_paths or self._metabolic_profile_paths(folder_path))
        if not candidate_paths:
            # Reached directly from the Connectivity button, so say what was
            # looked for rather than failing silently the way the drop-through
            # path can afford to.
            self.statusBar().showMessage(
                f"No metabolic profiles found in {folder_path.name}."
            )
            QMessageBox.information(
                self,
                "No Metabolic Profiles",
                (
                    "No files matching *_desc-metabolicprofiles_mrsi.npz were found in:\n"
                    f"{folder_path}\n\n"
                    "Subfolders were scanned recursively. These are written by MRSIPrep "
                    "alongside the connectivity matrices; a folder holding only matrices "
                    "has nothing to compute from."
                ),
            )
            return []
        try:
            from window.compute_connectivity import ComputeConnectivityDialog
        except Exception:
            from mrsi_viewer.window.compute_connectivity import ComputeConnectivityDialog

        dialog = ComputeConnectivityDialog(folder_path, candidate_paths, parent=self)
        if dialog.exec() != _dialog_accepted_code():
            return []
        written = dialog.output_paths()
        if not written:
            return []
        added = self._add_files([str(path) for path in written])
        self.statusBar().showMessage(
            f"Computed {len(written)} connectivity matrices from {folder_path.name}; added {len(added)}."
        )
        return added

    def _open_batch_import_dialog(self, folder_path: Path):
        folder_path = Path(folder_path)
        candidate_paths = self._batch_connectivity_paths(folder_path)
        if not candidate_paths:
            # No ready-made matrices, but the folder may hold the profiles they
            # are computed from -- offer to compute rather than dead-ending.
            profile_paths = self._metabolic_profile_paths(folder_path)
            if profile_paths:
                return self._open_compute_connectivity_dialog(folder_path, profile_paths)
            self.statusBar().showMessage(
                f"No .npz files containing 'connectivity' found in {folder_path.name} or its subfolders."
            )
            QMessageBox.information(
                self,
                "No Connectivity Matrices",
                (
                    f"No files ending in .npz and containing 'connectivity' were found in:\n"
                    f"{folder_path}\n\n"
                    "Subfolders were scanned recursively."
                ),
            )
            return []

        try:
            from window.batch_import import BatchMatrixImportDialog
        except Exception:
            from mrsi_viewer.window.batch_import import BatchMatrixImportDialog

        dialog = BatchMatrixImportDialog(
            folder_path,
            candidate_paths,
            export_callback=self._import_stacked_result,
            parent=self,
        )
        self._batch_import_dialog = dialog
        if hasattr(dialog, "set_theme"):
            dialog.set_theme(self._theme_name)
        try:
            if dialog.exec() != _dialog_accepted_code():
                return []

            selected_paths = dialog.selected_paths()
            added_paths = self._add_files(selected_paths)
            if not added_paths:
                self.statusBar().showMessage("No new batch files were added.")
                return []
            self.statusBar().showMessage(
                f"Added {len(added_paths)} matrix files from {folder_path.name}."
            )
            return added_paths
        finally:
            if self._batch_import_dialog is dialog:
                self._batch_import_dialog = None

    def _open_batch_folder(self) -> None:
        selected_dir = QFileDialog.getExistingDirectory(
            self,
            "Select folder with connectivity matrices",
            str(self._default_open_dir()),
        )
        if not selected_dir:
            return

        self._open_batch_import_dialog(Path(selected_dir))

    def _add_files(self, paths):
        added_paths = []
        added_any = False
        loaded_precomputed_bundle = None
        skipped_messages = []
        for raw_path in paths:
            if not raw_path:
                continue
            path = Path(raw_path)
            if path.suffix.lower() != ".npz":
                skipped_messages.append(f"{path.name or path}: not an .npz file")
                continue
            try:
                path_is_file = path.is_file()
            except OSError as exc:
                skipped_messages.append(f"{path.name or path}: cannot access file ({exc})")
                continue
            if not path_is_file:
                skipped_messages.append(f"{path.name or path}: file not found")
                continue
            self._invalidate_path_caches(path)
            try:
                precomputed_bundle = self._load_precomputed_gradient_bundle(path)
            except Exception as exc:
                skipped_messages.append(f"{path.name}: failed to load gradients ({exc})")
                continue
            if precomputed_bundle is not None:
                loaded_precomputed_bundle = precomputed_bundle
                added_paths.append(path)
                continue
            entry_id = self._file_entry_id(path)
            if entry_id in self._entries:
                added_paths.append(path)
                continue
            _entry_id, entry = self._workspace.add_file_entry(path, label=path.name)
            self._add_workspace_list_item(entry, tooltip=path, select=False)
            added_paths.append(path)
            added_any = True

        if added_any and self.file_list.currentItem() is None:
            self.file_list.setCurrentRow(self.file_list.count() - 1)

        if loaded_precomputed_bundle is not None:
            self._activate_precomputed_gradient_bundle(loaded_precomputed_bundle)

        if not added_paths and loaded_precomputed_bundle is None:
            if skipped_messages:
                self.statusBar().showMessage("; ".join(skipped_messages[-2:]))
            else:
                self.statusBar().showMessage("No valid .npz files added.")
        self._update_gradients_button()
        self._sync_combine_dialog_state()
        return added_paths

    def _remove_selected(self) -> None:
        item = self.file_list.currentItem()
        if item is None:
            return
        entry_id = item.data(USER_ROLE)
        self._workspace.remove_entry(entry_id)
        if getattr(self, "_gradient_classification_ref_entry_id", None) == entry_id:
            self._gradient_classification_ref_entry_id = None
        row = self.file_list.row(item)
        self.file_list.takeItem(row)
        if self.file_list.count() == 0:
            self._clear_plot()
        else:
            self._update_nbs_prepare_button()
            self._update_gradients_button()
        self._sync_combine_dialog_state()

    def _clear_files(self) -> None:
        self._workspace.clear()
        self._data_access.clear_caches()
        self._gradient_precomputed_bundle = None
        self._gradient_precomputed_selected_row = None
        self._gradient_use_precomputed_bundle = False
        self._gradient_classification_ref_entry_id = None
        self.file_list.clear()
        self._clear_plot()
        self._update_nbs_prepare_button()
        self._sync_combine_dialog_state()
        self.statusBar().showMessage("File list cleared.")
