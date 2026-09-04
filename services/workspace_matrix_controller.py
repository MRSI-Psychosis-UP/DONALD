#!/usr/bin/env python3
"""Controller for workspace matrix selection, display state, and export flows."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .display_state import (
    build_display_controls_state,
    ensure_entry_display_settings,
    log_scale_error,
    normalize_display_scale,
    parse_display_limits,
    selected_colormap_name,
    store_display_controls,
    update_sample_control_state,
)
from .entry_helpers import (
    collect_export_metadata,
    default_entry_title,
    normalize_covar_selection,
)
from .selection_export import (
    build_key_options_state,
    build_selection_change_state,
    collect_grid_export_items,
    default_matrix_export_name,
    normalize_grid_export_output_path,
)


class WorkspaceMatrixController:
    """Owns viewer orchestration around workspace matrix selection and export."""

    def __init__(self, viewer, *, fallback_colormap: str, covars_columns) -> None:
        self._viewer = viewer
        self._fallback_colormap = str(fallback_colormap or "")
        self._covars_columns = covars_columns

    def apply_key_options_state(self, state) -> None:
        viewer = self._viewer
        viewer.key_combo.blockSignals(True)
        viewer.key_combo.clear()
        for key in state.items:
            viewer.key_combo.addItem(key)
        if state.selected_key and viewer.key_combo.findText(state.selected_key) >= 0:
            viewer.key_combo.setCurrentText(state.selected_key)
        viewer.key_combo.setEnabled(state.enabled)
        viewer.key_combo.blockSignals(False)

    def refresh_key_options(self, entry) -> None:
        valid_keys = viewer._get_valid_keys_cached(entry["path"]) if (viewer := self._viewer) and entry is not None else []
        self.apply_key_options_state(
            build_key_options_state(entry, valid_keys=valid_keys),
        )

    def refresh_covars_options(self, source_path: Path, entry) -> None:
        viewer = self._viewer
        info = viewer._covars_info_cached(source_path)
        columns, selected_name = normalize_covar_selection(
            self._covars_columns(info),
            entry.get("covar_name"),
        )
        viewer.covar_combo.blockSignals(True)
        viewer.covar_combo.clear()
        for col in columns:
            viewer.covar_combo.addItem(col)
        if selected_name and viewer.covar_combo.findText(selected_name) >= 0:
            viewer.covar_combo.setCurrentText(selected_name)
        viewer.covar_combo.blockSignals(False)
        enabled = viewer.covar_combo.count() > 0
        viewer.covar_combo.setEnabled(enabled)
        viewer._update_selector_prepare_button()

    def default_title_for_entry(self, entry) -> str:
        viewer = self._viewer
        source_path = entry.get("path")
        sample_index = entry.get("sample_index")
        info = viewer._covars_info_cached(source_path) if source_path is not None else None
        group_value = (
            viewer._load_group_value_cached(source_path, sample_index)
            if source_path is not None and sample_index is not None and sample_index >= 0
            else None
        )
        return default_entry_title(
            entry,
            covars_info=info,
            group_value=group_value,
        )

    def apply_title_for_entry(self, entry, force: bool = False) -> str:
        viewer = self._viewer
        entry_id = entry["id"]
        default_title_text = self.default_title_for_entry(entry)
        if force or entry.get("auto_title", True):
            viewer.titles[entry_id] = default_title_text
            viewer.title_edit.setText(default_title_text)
            return default_title_text
        current = viewer.title_edit.text().strip() or viewer.titles.get(entry_id, default_title_text)
        viewer.titles[entry_id] = current
        return current

    def on_title_edited(self) -> None:
        viewer = self._viewer
        entry = viewer._current_entry()
        if entry is None:
            return
        entry_id = entry["id"]
        default_title_text = self.default_title_for_entry(entry)
        text = viewer.title_edit.text().strip()
        if text and text != default_title_text:
            entry["auto_title"] = False
            viewer.titles[entry_id] = text
        else:
            entry["auto_title"] = True
            viewer.titles[entry_id] = default_title_text
            viewer.title_edit.setText(default_title_text)
        viewer._plot_selected()

    def on_sample_changed(self, value: int) -> None:
        viewer = self._viewer
        entry = viewer._current_entry()
        if entry is None or entry.get("kind") not in {"file", "derived"}:
            return
        entry["sample_index"] = value
        if entry.get("kind") == "file" and entry.get("auto_title", True):
            self.apply_title_for_entry(entry, force=True)
        viewer._plot_selected()

    def update_sample_controls(self, entry, axis, stack_len) -> None:
        viewer = self._viewer
        state = update_sample_control_state(entry, axis=axis, stack_len=stack_len)
        viewer.sample_spin.blockSignals(True)
        viewer.sample_spin.setEnabled(state.enabled)
        viewer.sample_spin.setRange(state.minimum, state.maximum)
        viewer.sample_spin.setSpecialValueText(state.special_value_text)
        viewer.sample_spin.setValue(state.value)
        viewer.sample_add_button.setEnabled(state.add_enabled)
        viewer.sample_spin.blockSignals(False)

    def ensure_entry_display_settings(self, entry):
        viewer = self._viewer
        return ensure_entry_display_settings(
            entry,
            default_matrix_colormap=viewer._default_matrix_colormap,
            available_colormap_names=viewer._available_colormap_names(),
            fallback_colormap=self._fallback_colormap,
        )

    def load_display_controls_for_entry(self, entry) -> None:
        viewer = self._viewer
        state = build_display_controls_state(
            entry,
            default_matrix_colormap=viewer._default_matrix_colormap,
            available_colormap_names=viewer._available_colormap_names(),
            fallback_colormap=self._fallback_colormap,
        )
        if state is None:
            return

        viewer.cmap_combo.blockSignals(True)
        if state.colormap_name and viewer.cmap_combo.findText(state.colormap_name) >= 0:
            viewer.cmap_combo.setCurrentText(state.colormap_name)
        elif viewer.cmap_combo.count() > 0:
            viewer.cmap_combo.setCurrentIndex(0)
        viewer.cmap_combo.blockSignals(False)

        viewer.display_auto_check.blockSignals(True)
        viewer.display_auto_check.setChecked(state.auto_scale)
        viewer.display_auto_check.blockSignals(False)

        viewer.display_min_edit.blockSignals(True)
        viewer.display_min_edit.setText(state.min_text)
        viewer.display_min_edit.blockSignals(False)

        viewer.display_max_edit.blockSignals(True)
        viewer.display_max_edit.setText(state.max_text)
        viewer.display_max_edit.blockSignals(False)

        viewer.display_scale_combo.blockSignals(True)
        viewer.display_scale_combo.setCurrentText(state.scale_label)
        viewer.display_scale_combo.blockSignals(False)

        viewer.display_min_edit.setEnabled(state.min_enabled)
        viewer.display_max_edit.setEnabled(state.max_enabled)

    def store_display_controls_for_entry(self, entry):
        viewer = self._viewer
        if entry is None:
            return None
        return store_display_controls(
            entry,
            colormap_name=viewer.cmap_combo.currentText().strip(),
            auto_scale=bool(viewer.display_auto_check.isChecked()),
            min_text=viewer.display_min_edit.text().strip(),
            max_text=viewer.display_max_edit.text().strip(),
            scale_choice=viewer.display_scale_combo.currentText().strip(),
            default_matrix_colormap=viewer._default_matrix_colormap,
            available_colormap_names=viewer._available_colormap_names(),
            fallback_colormap=self._fallback_colormap,
        )

    def on_colormap_changed(self, *_args) -> None:
        viewer = self._viewer
        entry = viewer._current_entry()
        if entry is None:
            return
        self.store_display_controls_for_entry(entry)
        viewer._plot_selected()

    def selected_colormap_name(self, entry=None) -> str:
        viewer = self._viewer
        current_name = viewer.cmap_combo.currentText().strip() if hasattr(viewer, "cmap_combo") else ""
        return selected_colormap_name(
            entry=entry,
            current_name=current_name,
            default_matrix_colormap=viewer._default_matrix_colormap,
            available_colormap_names=viewer._available_colormap_names(),
            fallback_colormap=self._fallback_colormap,
        )

    def selected_colormap(self, entry=None):
        viewer = self._viewer
        name = self.selected_colormap_name(entry)
        if name in viewer._custom_cmaps:
            try:
                return viewer._colorbar.load_fsl_cmap(name)
            except Exception:
                return name
        return name

    def current_display_limits(self, entry=None):
        viewer = self._viewer
        if entry is not None:
            self.ensure_entry_display_settings(entry)
            auto_scale = bool(entry.get("display_auto", True))
            min_text = str(entry.get("display_min_text", "") or "").strip()
            max_text = str(entry.get("display_max_text", "") or "").strip()
        else:
            if not hasattr(viewer, "display_auto_check"):
                return None, None, None
            auto_scale = bool(viewer.display_auto_check.isChecked())
            min_text = viewer.display_min_edit.text().strip() if hasattr(viewer, "display_min_edit") else ""
            max_text = viewer.display_max_edit.text().strip() if hasattr(viewer, "display_max_edit") else ""
        return parse_display_limits(
            auto_scale=auto_scale,
            min_text=min_text,
            max_text=max_text,
        )

    def current_display_scale(self, entry=None) -> str:
        viewer = self._viewer
        if entry is not None:
            self.ensure_entry_display_settings(entry)
            choice = str(entry.get("display_scale", "linear") or "").strip().lower()
        else:
            if not hasattr(viewer, "display_scale_combo"):
                return "linear"
            choice = viewer.display_scale_combo.currentText().strip().lower()
        return normalize_display_scale(choice)

    @staticmethod
    def log_scale_error(matrix, vmin, vmax):
        return log_scale_error(matrix, vmin, vmax)

    def on_display_scaling_changed(self, *_args) -> None:
        viewer = self._viewer
        entry = viewer._current_entry()
        if entry is not None:
            state = self.store_display_controls_for_entry(entry)
            if state is not None:
                viewer.display_min_edit.setEnabled(state.min_enabled)
                viewer.display_max_edit.setEnabled(state.max_enabled)
            viewer._plot_selected()
            return
        auto_scale = viewer.display_auto_check.isChecked()
        viewer.display_min_edit.setEnabled(not auto_scale)
        viewer.display_max_edit.setEnabled(not auto_scale)

    def _collect_export_metadata(self, entry, selected_key):
        viewer = self._viewer
        labels, names = viewer._entry_parcel_metadata(entry)
        return collect_export_metadata(
            entry,
            selected_key,
            labels=labels,
            names=names,
            current_parcel_labels=viewer._current_parcel_labels,
            current_parcel_names=viewer._current_parcel_names,
        )

    def write_selected_matrix_to_file(self, *, file_dialog_class) -> None:
        viewer = self._viewer
        entry = viewer._current_entry()
        if entry is None:
            viewer.statusBar().showMessage("No matrix selected.")
            return

        full_payload = entry.get("npz_payload")
        if isinstance(full_payload, dict) and full_payload:
            selected_key = str(entry.get("selected_key") or "matrix").strip() or "matrix"
            default_name = default_matrix_export_name(entry, selected_key=selected_key)
            start_dir = viewer._default_results_dir()
            save_path, _selected_filter = file_dialog_class.getSaveFileName(
                viewer,
                "Write corrected NPZ",
                str(start_dir / default_name),
                "NumPy archive (*.npz);;All files (*)",
            )
            if not save_path:
                return
            output_path = Path(save_path)
            if output_path.suffix.lower() != ".npz":
                output_path = output_path.with_suffix(".npz")
            try:
                np.savez(output_path, **{str(key): value for key, value in full_payload.items()})
            except Exception as exc:
                viewer.statusBar().showMessage(f"Failed to write NPZ: {exc}")
                return
            viewer.statusBar().showMessage(f"Wrote corrected NPZ to {output_path.name}.")
            return

        try:
            matrix, selected_key = viewer._matrix_for_entry(entry)
        except Exception as exc:
            viewer.statusBar().showMessage(f"Failed to resolve selected matrix: {exc}")
            return

        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            viewer.statusBar().showMessage("Selected matrix is not a square 2D matrix.")
            return

        default_name = default_matrix_export_name(entry, selected_key=selected_key)
        start_dir = viewer._default_results_dir()

        save_path, _selected_filter = file_dialog_class.getSaveFileName(
            viewer,
            "Write selected matrix to NPZ",
            str(start_dir / default_name),
            "NumPy archive (*.npz);;All files (*)",
        )
        if not save_path:
            return
        output_path = Path(save_path)
        if output_path.suffix.lower() != ".npz":
            output_path = output_path.with_suffix(".npz")

        payload = {"matrix_pop_avg": matrix}
        payload.update(self._collect_export_metadata(entry, selected_key))
        try:
            np.savez(output_path, **payload)
        except Exception as exc:
            viewer.statusBar().showMessage(f"Failed to write NPZ: {exc}")
            return
        viewer.statusBar().showMessage(f"Wrote selected matrix to {output_path.name}.")

    def export_grid(
        self,
        *,
        export_grid_dialog_class,
        dialog_accepted_code,
        figure_class,
        sim_matrix_plot,
        remove_axes_border,
        apply_rotation,
    ) -> None:
        viewer = self._viewer
        if viewer.file_list.count() == 0:
            viewer.statusBar().showMessage("No matrices to export.")
            return
        current_entry = viewer._current_entry()
        if current_entry is not None:
            self.store_display_controls_for_entry(current_entry)

        default_output_path = viewer._export_grid_output_path or str(
            viewer._default_results_dir() / "connectome_grid.pdf"
        )
        dialog = export_grid_dialog_class(
            default_path=default_output_path,
            default_columns=viewer._export_grid_columns,
            rotate=viewer._export_grid_rotate,
            parent=viewer,
        )
        dialog.setStyleSheet(viewer.styleSheet())
        dialog.set_theme(viewer._theme_name)
        if dialog.exec() != dialog_accepted_code():
            return

        values = dialog.values()
        viewer._export_grid_output_path = values["output_path"]
        viewer._export_grid_columns = int(values["columns"])
        viewer._export_grid_rotate = bool(values["rotate"])
        viewer._export_grid_selected_filter = str(values.get("selected_filter") or "PDF (*.pdf)")

        output_path = normalize_grid_export_output_path(
            values["output_path"],
            selected_filter=viewer._export_grid_selected_filter,
        )
        viewer._export_grid_output_path = str(output_path)

        export_bundle = collect_grid_export_items(
            list(viewer._entry_ids()),
            viewer._entries,
            viewer.titles,
            matrix_for_entry=viewer._matrix_for_entry,
            display_limits_for_entry=self.current_display_limits,
            display_scale_for_entry=self.current_display_scale,
            log_scale_error=self.log_scale_error,
            selected_colormap_for_entry=self.selected_colormap,
        )
        if export_bundle.error:
            viewer.statusBar().showMessage(export_bundle.error)
            return
        plot_items = list(export_bundle.items)
        skipped = list(export_bundle.skipped)

        if not plot_items:
            viewer.statusBar().showMessage("No matrices exported (missing keys or load errors).")
            return

        cols = min(viewer._export_grid_columns, len(plot_items))
        rows = int(math.ceil(len(plot_items) / cols))
        export_figure = figure_class(figsize=(4 * cols, 4 * rows))
        export_figure.patch.set_facecolor("white")
        axes = export_figure.subplots(rows, cols)
        if isinstance(axes, np.ndarray):
            flat_axes = axes.flatten()
        else:
            flat_axes = [axes]
        for ax in flat_axes:
            ax.set_facecolor("white")

        rotate = viewer._export_grid_rotate
        for idx, plot_item in enumerate(plot_items):
            matrix = plot_item.matrix
            ax = flat_axes[idx]
            sim_matrix_plot.plot_simmatrix(
                matrix,
                ax=ax,
                titles=plot_item.title,
                colormap=plot_item.colormap,
                vmin=plot_item.vmin,
                vmax=plot_item.vmax,
                zscale=plot_item.zscale,
            )
            remove_axes_border(ax)
            if rotate:
                apply_rotation(ax, matrix, 45.0)

        for ax in flat_axes[len(plot_items):]:
            ax.axis("off")

        export_figure.tight_layout()
        output_format = (output_path.suffix or ".pdf").lstrip(".").lower() or "pdf"
        export_figure.savefig(
            str(output_path),
            format=output_format,
            bbox_inches="tight",
            dpi=600,
            facecolor="white",
            edgecolor="none",
            transparent=False,
        )

        if skipped:
            viewer.statusBar().showMessage(
                f"Exported {len(plot_items)} matrices to {output_path.name}. "
                f"Skipped {len(skipped)}."
            )
        else:
            viewer.statusBar().showMessage(f"Exported {len(plot_items)} matrices to {output_path.name}.")

    def on_selection_changed(self, current, _previous) -> None:
        viewer = self._viewer
        if current is None:
            viewer._clear_plot()
            return
        entry_id = current.data(USER_ROLE)
        entry = viewer._entries.get(entry_id)
        if entry is None:
            viewer._clear_plot()
            return

        selection_state = build_selection_change_state(
            entry,
            valid_keys=viewer._get_valid_keys_cached(entry["path"]) if entry.get("kind") == "file" else None,
            stored_title=viewer.titles.get(entry_id, entry.get("label", "Matrix")),
            default_title=self.default_title_for_entry(entry),
        )
        self.apply_key_options_state(selection_state.key_options)

        if selection_state.source_path:
            self.refresh_covars_options(selection_state.source_path, entry)
        self.load_display_controls_for_entry(entry)

        if selection_state.auto_title:
            viewer.titles[entry_id] = selection_state.title_text
        viewer.title_edit.setText(selection_state.title_text)
        viewer._plot_selected()
        viewer._update_write_to_file_button()
        viewer._update_view_labels_button()
        viewer._sync_combine_dialog_state()


USER_ROLE = 0x0100
