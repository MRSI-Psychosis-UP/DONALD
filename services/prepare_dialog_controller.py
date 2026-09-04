#!/usr/bin/env python3
"""Controller for launcher flows shared by prepare dialogs."""

from __future__ import annotations

from pathlib import Path


class PrepareDialogController:
    """Owns repeated prepare-dialog source resolution and validation logic."""

    def __init__(self, viewer, *, covars_columns, load_matrix_from_npz, stack_axis) -> None:
        self._viewer = viewer
        self._covars_columns = covars_columns
        self._load_matrix_from_npz = load_matrix_from_npz
        self._stack_axis = stack_axis

    def _covars_length(self, covars_info) -> int:
        if covars_info is None:
            return 0
        df = covars_info.get("df")
        if df is not None:
            return len(df)
        data = covars_info.get("data")
        if data is not None:
            return len(data)
        return 0

    def current_stack_source(self):
        viewer = self._viewer
        entry = viewer._current_entry()
        if entry is None or entry.get("kind") != "file":
            return None
        source_path = entry.get("path")
        if source_path is None:
            return None
        source_path = Path(source_path)
        if not source_path.exists():
            return None
        key = viewer._ensure_entry_key(entry)
        if not key:
            return None

        stack_axis = entry.get("stack_axis")
        stack_len = entry.get("stack_len")
        if stack_axis is not None and stack_len is not None and int(stack_len) > 1:
            return {"path": source_path, "key": key, "stack_len": int(stack_len)}

        try:
            raw = self._load_matrix_from_npz(source_path, key, average=False)
        except Exception:
            return None
        if raw.ndim != 3:
            return None
        axis = self._stack_axis(raw.shape)
        if axis is None or int(raw.shape[axis]) <= 1:
            return None
        return {"path": source_path, "key": key, "stack_len": int(raw.shape[axis])}

    def current_nbs_source(self):
        return self.current_stack_source()

    def current_selector_source(self):
        return self.current_stack_source()

    def current_harmonize_source(self):
        return self.current_stack_source()

    def _workspace_reference_options(self):
        viewer = self._viewer
        options = []
        seen_paths = set()
        for entry_id in viewer._entry_ids():
            entry = viewer._entries.get(entry_id)
            if not isinstance(entry, dict) or entry.get("kind") != "file":
                continue
            path_raw = entry.get("path")
            if not path_raw:
                continue
            path = Path(path_raw)
            if not path.is_file() or path.suffix.lower() != ".npz":
                continue
            try:
                normalized = str(path.resolve())
            except Exception:
                normalized = str(path)
            if normalized in seen_paths:
                continue
            label = str(entry.get("label") or path.name).strip() or path.name
            key = str(entry.get("selected_key") or "").strip()
            if key and key not in label:
                label = f"{label} [{key}]"
            options.append({"id": str(entry_id), "label": label, "path": str(path)})
            seen_paths.add(normalized)
        options.sort(key=lambda item: item["label"].lower())
        return options

    def _validated_source_context(self, source, *, empty_message: str):
        viewer = self._viewer
        if source is None:
            viewer.statusBar().showMessage(empty_message)
            return None

        source_path = source["path"]
        covars_info = viewer._covars_info_cached(source_path)
        if covars_info is None:
            viewer.statusBar().showMessage("Covars not found in selected file.")
            return None

        covars_len = self._covars_length(covars_info)
        if covars_len and covars_len != source["stack_len"]:
            viewer.statusBar().showMessage("Covars length does not match matrix stack size.")
            return None

        return {
            "source": source,
            "covars_info": covars_info,
        }

    def update_nbs_prepare_button(self) -> None:
        viewer = self._viewer
        enabled = self.current_nbs_source() is not None
        dialog = getattr(viewer, "_nbs_dialog", None)
        if dialog is not None and hasattr(dialog, "set_workspace_reference_options"):
            try:
                dialog.set_workspace_reference_options(self._workspace_reference_options())
            except Exception:
                pass
        if not hasattr(viewer, "nbs_prepare_button"):
            if hasattr(viewer, "nbs_prepare_action"):
                viewer.nbs_prepare_action.setEnabled(enabled)
            viewer._update_gradients_button()
            viewer._update_selector_prepare_button()
            viewer._update_harmonize_prepare_button()
            viewer._update_combine_button()
            return
        viewer.nbs_prepare_button.setEnabled(enabled)
        if hasattr(viewer, "nbs_prepare_action"):
            viewer.nbs_prepare_action.setEnabled(enabled)
        viewer._update_gradients_button()
        viewer._update_selector_prepare_button()
        viewer._update_harmonize_prepare_button()
        viewer._update_combine_button()

    def update_selector_prepare_button(self) -> None:
        viewer = self._viewer
        if not hasattr(viewer, "selector_prepare_button"):
            return
        source = self.current_selector_source()
        enabled = False
        if source is not None:
            info = viewer._covars_info_cached(source["path"])
            enabled = bool(self._covars_columns(info))
        viewer.selector_prepare_button.setEnabled(enabled)
        viewer._update_harmonize_prepare_button()
        viewer._update_write_to_file_button()

    def update_harmonize_prepare_button(self) -> None:
        viewer = self._viewer
        if not hasattr(viewer, "harmonize_prepare_button"):
            if hasattr(viewer, "harmonize_prepare_action"):
                viewer.harmonize_prepare_action.setEnabled(False)
            return
        source = self.current_harmonize_source()
        enabled = False
        if source is not None:
            info = viewer._covars_info_cached(source["path"])
            enabled = bool(self._covars_columns(info))
        viewer.harmonize_prepare_button.setEnabled(enabled)
        if hasattr(viewer, "harmonize_prepare_action"):
            viewer.harmonize_prepare_action.setEnabled(enabled)

    def update_write_to_file_button(self) -> None:
        viewer = self._viewer
        if not hasattr(viewer, "write_matrix_button"):
            return
        viewer.write_matrix_button.setEnabled(viewer._current_entry() is not None)

    def open_nbs_prepare_dialog(self) -> None:
        viewer = self._viewer
        context = self._validated_source_context(
            self.current_nbs_source(),
            empty_message="NBS Prepare requires a file-based matrix stack (multiple matrices).",
        )
        if context is None:
            return

        try:
            from window.nbs_prepare import NBSPrepareDialog
        except Exception:
            try:
                from mrsi_viewer.window.nbs_prepare import NBSPrepareDialog
            except Exception as exc:
                viewer.statusBar().showMessage(f"Failed to open NBS window: {exc}")
                return

        source = context["source"]
        source_path = source["path"]
        viewer._nbs_dialog = NBSPrepareDialog(
            covars_info=context["covars_info"],
            source_path=source_path,
            matrix_key=source["key"],
            matlab_cmd_default=viewer._matlab_cmd_default,
            matlab_nbs_path_default=viewer._matlab_nbs_path_default,
            output_dir_default=viewer._results_dir_default,
            atlas_dir_default=viewer._atlas_dir_default,
            bids_dir_default=viewer._bids_dir_default,
            workspace_reference_options=self._workspace_reference_options(),
            gradient_rgb_payload_callback=getattr(viewer, "_current_gradient_rgb_payload", None),
            theme_name=viewer._theme_name,
            parent=viewer,
        )
        viewer._nbs_dialog.show()
        viewer.statusBar().showMessage(
            f"Opened NBS Prepare ({source_path.name}, key={source['key']})."
        )

    def open_selector_prepare_dialog(self) -> None:
        viewer = self._viewer
        context = self._validated_source_context(
            self.current_selector_source(),
            empty_message="Selector Prepare requires a file-based matrix stack (multiple matrices).",
        )
        if context is None:
            return

        try:
            from window.selector_prepare import SelectorPrepareDialog
        except Exception:
            try:
                from mrsi_viewer.window.selector_prepare import SelectorPrepareDialog
            except Exception as exc:
                viewer.statusBar().showMessage(f"Failed to open Selector window: {exc}")
                return

        source = context["source"]
        source_path = source["path"]
        viewer._selector_dialog = SelectorPrepareDialog(
            covars_info=context["covars_info"],
            source_path=source_path,
            matrix_key=source["key"],
            theme_name=viewer._theme_name,
            export_callback=viewer._import_selector_aggregate,
            parent=viewer,
        )
        viewer._selector_dialog.show()
        viewer.statusBar().showMessage(
            f"Opened Selector Prepare ({source_path.name}, key={source['key']})."
        )

    def open_harmonize_prepare_dialog(self) -> None:
        viewer = self._viewer
        context = self._validated_source_context(
            self.current_harmonize_source(),
            empty_message="Harmonize Prepare requires a file-based matrix stack (multiple matrices).",
        )
        if context is None:
            return

        try:
            from window.harmonize_prepare import HarmonizePrepareDialog
        except Exception:
            try:
                from mrsi_viewer.window.harmonize_prepare import HarmonizePrepareDialog
            except Exception as exc:
                viewer.statusBar().showMessage(f"Failed to open Harmonize window: {exc}")
                return

        source = context["source"]
        source_path = source["path"]
        viewer._harmonize_dialog = HarmonizePrepareDialog(
            covars_info=context["covars_info"],
            source_path=source_path,
            matrix_key=source["key"],
            output_dir_default=viewer._results_dir_default,
            theme_name=viewer._theme_name,
            export_callback=viewer._import_harmonized_result,
            parent=viewer,
        )
        viewer._harmonize_dialog.show()
        viewer.statusBar().showMessage(
            f"Opened Harmonize Prepare ({source_path.name}, key={source['key']})."
        )
