#!/usr/bin/env python3
"""Controller for gradient dialog state, precomputed bundles, and classification dialog wiring."""

from __future__ import annotations

import math
import re
from pathlib import Path
from types import MethodType

import numpy as np
from matplotlib.colors import Normalize

try:
    from PyQt6.QtWidgets import QApplication, QFileDialog
except Exception:
    from PyQt5.QtWidgets import QApplication, QFileDialog

from mrsitoolbox.connectomics.nettools import NetTools

from .data_access import MatrixDataAccess as _MatrixDataAccess

nettools = NetTools()

# Edit these values to tune the 3D glass-brain path edges without changing GUI code.
GLASSBRAIN_EDGE_COLOR = (0.0, 0.0, 0.0)
GLASSBRAIN_EDGE_WIDTH_MULTIPLIER = 1.8
GLASSBRAIN_EDGE_MIN_WIDTH = 5.0
GLASSBRAIN_EDGE_RADIUS_SCALE = 0.12
GLASSBRAIN_SUBC_DASH_LENGTH = 7.0
GLASSBRAIN_SUBC_DASH_GAP = 4.0

PARCEL_LABEL_KEYS = ()
PARCEL_NAME_KEYS = ()
_to_string_list = None
_display_text = None
_load_covars_info = None
_covars_to_rows = None
_normalize_subject_token = None
_normalize_session_token = None
_flatten_display_vector = None
_coerce_label_indices = None


class GradientDialogController:
    """Owns gradient/classification dialog state and precomputed bundle orchestration."""

    BOUND_METHOD_NAMES = (
        '_reset_gradients_output',
        '_set_gradients_progress',
        '_current_gradient_component_count',
        '_normalize_gradient_hemisphere_mode',
        '_normalize_gradient_surface_mesh',
        '_normalize_gradient_surface_render_count',
        '_normalize_gradient_surface_procrustes',
        '_normalize_gradient_classification_component',
        '_normalize_gradient_classification_axis',
        '_normalize_gradient_scatter_rotation',
        '_normalize_gradient_triangular_color_order',
        '_normalize_gradient_rgb_scalar_mode',
        '_normalize_gradient_rgb_mask_strictness',
        '_normalize_gradient_classification_fit_mode',
        '_normalize_gradient_rotation_preset',
        '_normalize_gradient_network_component',
        '_ensure_gradient_rotation_count',
        '_available_gradient_network_component_count',
        '_gradient_surface_procrustes_available',
        '_selected_gradient_hemisphere_mode',
        '_selected_gradient_surface_mesh',
        '_selected_gradient_surface_render_count',
        '_selected_gradient_surface_procrustes',
        '_selected_gradient_classification_surface_mesh',
        '_selected_gradient_classification_hemisphere_mode',
        '_selected_gradient_scatter_rotation',
        '_selected_gradient_triangular_rgb',
        '_selected_gradient_preload_matching_paths',
        '_selected_gradient_classification_fit_mode',
        '_selected_gradient_triangular_color_order',
        '_selected_gradient_rgb_scalar_mode',
        '_selected_gradient_rgb_mask_strictness',
        '_selected_gradient_classification_colormap',
        '_selected_gradient_classification_component',
        '_selected_gradient_classification_x_axis',
        '_selected_gradient_classification_y_axis',
        '_selected_gradient_classification_z_axis',
        '_selected_gradient_classification_ref_entry_id',
        '_selected_gradient_classification_ignore_lh_parcel',
        '_selected_gradient_classification_ignore_rh_parcel',
        '_gradient_classification_ignore_parcel_options',
        '_is_gradient_classification_axis_available',
        '_gradient_classification_axis_label',
        '_classification_uses_reference_axis',
        '_classification_reference_component_count',
        '_classification_axis_component_count',
        '_infer_projection_hemisphere_from_name',
        '_gradient_projection_hemisphere_codes',
        '_gradient_projection_hemisphere_mask',
        '_can_classify_gradients',
        '_current_gradient_rotation_presets',
        '_selected_gradient_network_component',
        '_has_square_current_matrix',
        '_update_gradients_button',
        '_set_gradient_source_mode',
        '_on_gradient_component_changed',
        '_on_gradient_colormap_changed',
        '_on_gradient_hemisphere_changed',
        '_on_gradient_surface_mesh_changed',
        '_on_gradient_surface_render_count_changed',
        '_on_gradient_surface_procrustes_changed',
        '_on_gradient_classification_surface_mesh_changed',
        '_on_gradient_classification_hemisphere_changed',
        '_on_gradient_scatter_rotation_changed',
        '_on_gradient_triangular_rgb_changed',
        '_on_gradient_preload_matching_paths_changed',
        '_on_gradient_classification_fit_mode_changed',
        '_on_gradient_triangular_color_order_changed',
        '_on_gradient_rgb_scalar_mode_changed',
        '_on_gradient_rgb_mask_strictness_changed',
        '_on_gradient_classification_colormap_changed',
        '_on_gradient_classification_component_changed',
        '_on_gradient_classification_x_axis_changed',
        '_on_gradient_classification_y_axis_changed',
        '_on_gradient_classification_z_axis_changed',
        '_on_gradient_classification_ref_entry_changed',
        '_on_gradient_classification_ignore_lh_changed',
        '_on_gradient_classification_ignore_rh_changed',
        '_load_gradient_classification_adjacency_npz',
        '_gradient_classification_adjacency_data',
        '_set_gradient_classification_adjacency',
        '_select_gradient_classification_adjacency',
        '_clear_gradient_classification_adjacency',
        '_npz_optional_scalar_text',
        '_npz_optional_display_vector',
        '_npz_optional_array',
        '_npz_subject_session_vectors',
        '_metabolite_names_from_array',
        '_reduce_metabolite_profiles',
        '_reduce_water_signal_profile',
        '_append_water_signal_profile',
        '_load_metabolite_profiles_from_npz',
        '_selected_gradient_metabolite_profiles_path',
        '_metabolite_subject_labels_from_npz',
        '_load_metabolite_profiles_for_gradient_context',
        '_canonicalize_precomputed_gradients',
        '_canonicalize_average_gradients',
        '_path_exists_safely',
        '_resolve_existing_path',
        '_portable_parcellation_path_candidates',
        '_resolve_portable_parcellation_path',
        '_load_precomputed_gradient_bundle',
        '_gradient_precomputed_row_pair',
        '_gradient_precomputed_selection_text',
        '_activate_precomputed_gradient_bundle',
        '_confirm_precomputed_gradient_row',
        '_classification_scatter_edge_pairs',
        '_gradient_entry_source_path',
        '_entry_parcel_metadata',
        '_preferred_parcel_key',
        '_parcel_key_option_label',
        '_available_gradient_parcel_key_options',
        '_selected_gradient_parcel_label_key',
        '_selected_gradient_parcel_name_key',
        '_gradient_matrix_label_for_entry',
        '_available_gradient_matrix_entries',
        '_available_gradient_reference_matrix_entries',
        '_available_combine_matrix_entries',
        '_sync_combine_dialog_state',
        '_update_combine_button',
        '_selected_gradient_entry_id',
        '_selected_gradient_entry',
        '_selected_gradient_classification_ref_entry',
        '_gradient_matrix_for_entry',
        '_has_square_matrix_entry',
        '_selected_gradient_matrix_label',
        '_on_gradient_matrix_entry_changed',
        '_on_gradient_parcel_label_key_changed',
        '_on_gradient_parcel_name_key_changed',
        '_on_gradient_network_component_changed',
        '_on_gradient_rotation_changed',
        '_sync_gradients_dialog_state',
        '_open_gradients_dialog',
        '_compute_gradients',
        '_save_gradients_projection',
        '_compute_spectral_coords_and_order',
        '_gradient_template_img_and_data',
        '_projected_gradient_component_count',
        '_ensure_projected_gradient_data',
        '_surface_render_gradient_matrix',
        '_project_gradient_matrix_to_volume',
        '_gradient_spatial_embedding',
        '_classification_spatial_embedding',
        '_classification_spatial_indices',
        '_classification_reference_gradient_payload',
        '_classification_axis_payload',
        '_rescale_classification_axis_to_range',
        '_render_gradients_3d',
        '_classify_gradients_fsaverage',
        '_project_classification_paths_to_fibrenet',
        '_project_classification_paths_to_brain',
        '_gradient_rotation_angles',
        '_actor_bounds_center',
        '_apply_gradient_component_rotation',
        '_render_gradients_network',
    )

    def __init__(
        self,
        viewer,
        *,
        parcel_label_keys,
        parcel_name_keys,
        to_string_list,
        display_text,
        load_covars_info,
        covars_to_rows,
        normalize_subject_token,
        normalize_session_token,
        flatten_display_vector,
        coerce_label_indices,
    ) -> None:
        object.__setattr__(self, '_viewer', viewer)
        global PARCEL_LABEL_KEYS, PARCEL_NAME_KEYS
        global _to_string_list, _display_text, _load_covars_info, _covars_to_rows
        global _normalize_subject_token, _normalize_session_token, _flatten_display_vector, _coerce_label_indices
        PARCEL_LABEL_KEYS = tuple(parcel_label_keys or ())
        PARCEL_NAME_KEYS = tuple(parcel_name_keys or ())
        _to_string_list = to_string_list
        _display_text = display_text
        _load_covars_info = load_covars_info
        _covars_to_rows = covars_to_rows
        _normalize_subject_token = normalize_subject_token
        _normalize_session_token = normalize_session_token
        _flatten_display_vector = flatten_display_vector
        _coerce_label_indices = coerce_label_indices

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
            if isinstance(descriptor, classmethod):
                setattr(self._viewer, name, getattr(owner, name))
                continue
            if descriptor is None:
                setattr(self._viewer, name, getattr(self, name))
                continue
            setattr(self._viewer, name, MethodType(descriptor, self._viewer))

    def _reset_gradients_output(self) -> None:
        self._last_gradients = None
        self._set_gradients_progress(0, 1, 0, "Idle")
        self._sync_gradients_dialog_state()

    def _set_gradients_progress(self, minimum: int, maximum: int, value: int, text: str) -> None:
        self._gradients_progress_state = {
            "minimum": int(minimum),
            "maximum": int(maximum),
            "value": int(value),
            "text": str(text or ""),
        }
        if getattr(self, "_gradients_dialog", None) is not None:
            try:
                self._gradients_dialog.set_progress(int(minimum), int(maximum), int(value), str(text or ""))
            except Exception:
                pass

    def _current_gradient_component_count(self) -> int:
        dialog = getattr(self, "_gradients_dialog", None)
        if dialog is not None:
            try:
                self._gradient_component_count = int(dialog.component_count())
            except Exception:
                pass
        try:
            value = int(self._gradient_component_count)
        except Exception:
            value = 4
        return max(1, min(10, value))

    @staticmethod
    def _normalize_gradient_hemisphere_mode(value: str) -> str:
        text = str(value or "both").strip().lower()
        if text not in {"both", "lh", "rh", "separate"}:
            text = "both"
        return text

    @staticmethod
    def _normalize_gradient_surface_mesh(value: str) -> str:
        text = str(value or "fsaverage4").strip().lower()
        valid = {"fsaverage3", "fsaverage4", "fsaverage5", "fsaverage6", "fsaverage7"}
        if text not in valid:
            text = "fsaverage4"
        return text

    @staticmethod
    def _normalize_gradient_surface_render_count(value, max_components=None) -> int:
        try:
            count = int(value)
        except Exception:
            count = 1
        if count < 1:
            count = 1
        if max_components is not None:
            count = min(count, max(1, int(max_components)))
        return count

    @staticmethod
    def _normalize_gradient_surface_procrustes(enabled) -> bool:
        return bool(enabled)

    @staticmethod
    def _normalize_gradient_classification_component(value, max_components=None) -> str:
        text = str(value or "1").strip().lower()
        if text.startswith("c"):
            text = text[1:]
        try:
            index = int(text)
        except Exception:
            index = 1
        if index < 1:
            index = 1
        if max_components is not None:
            index = min(index, max(1, int(max_components)))
        return str(index)

    @staticmethod
    def _normalize_gradient_classification_axis(value: str, default: str = "gradient1") -> str:
        fallback = str(default or "gradient1").strip().lower()
        valid_axes = {
            "none",
            "gradient1",
            "gradient2",
            "gradient3",
            "gradient_ref1",
            "gradient_ref2",
            "gradient_ref3",
            "spatial",
        }
        if fallback not in valid_axes:
            fallback = "gradient1"
        text = str(value or "").strip().lower()
        compact = re.sub(r"[\s_\-]+", "", text)
        mapping = {
            "none": "none",
            "2d": "none",
            "no z": "none",
            "no z axis": "none",
            "gradient1": "gradient1",
            "gradient 1": "gradient1",
            "g1": "gradient1",
            "c1": "gradient1",
            "1": "gradient1",
            "gradient2": "gradient2",
            "gradient 2": "gradient2",
            "g2": "gradient2",
            "c2": "gradient2",
            "2": "gradient2",
            "gradient3": "gradient3",
            "gradient 3": "gradient3",
            "g3": "gradient3",
            "c3": "gradient3",
            "3": "gradient3",
            "gradient_ref1": "gradient_ref1",
            "gradient ref 1": "gradient_ref1",
            "gradient reference 1": "gradient_ref1",
            "reference gradient 1": "gradient_ref1",
            "ref gradient 1": "gradient_ref1",
            "gref1": "gradient_ref1",
            "ref1": "gradient_ref1",
            "r1": "gradient_ref1",
            "gradient_ref2": "gradient_ref2",
            "gradient ref 2": "gradient_ref2",
            "gradient reference 2": "gradient_ref2",
            "reference gradient 2": "gradient_ref2",
            "ref gradient 2": "gradient_ref2",
            "gref2": "gradient_ref2",
            "ref2": "gradient_ref2",
            "r2": "gradient_ref2",
            "gradient_ref3": "gradient_ref3",
            "gradient ref 3": "gradient_ref3",
            "gradient reference 3": "gradient_ref3",
            "reference gradient 3": "gradient_ref3",
            "ref gradient 3": "gradient_ref3",
            "gref3": "gradient_ref3",
            "ref3": "gradient_ref3",
            "r3": "gradient_ref3",
            "spatial": "spatial",
            "space": "spatial",
        }
        compact_mapping = {
            "noz": "none",
            "nozaxis": "none",
            "gradientref1": "gradient_ref1",
            "gradientreference1": "gradient_ref1",
            "referencegradient1": "gradient_ref1",
            "refgradient1": "gradient_ref1",
            "gradientref2": "gradient_ref2",
            "gradientreference2": "gradient_ref2",
            "referencegradient2": "gradient_ref2",
            "refgradient2": "gradient_ref2",
            "gradientref3": "gradient_ref3",
            "gradientreference3": "gradient_ref3",
            "referencegradient3": "gradient_ref3",
            "refgradient3": "gradient_ref3",
        }
        normalized = mapping.get(text, compact_mapping.get(compact, mapping.get(compact, fallback)))
        if normalized not in valid_axes:
            normalized = fallback
        return normalized

    @staticmethod
    def _normalize_gradient_scatter_rotation(value: str) -> str:
        text = str(value or "Default").strip()
        valid = {"Default", "+90", "-90", "180"}
        if text not in valid:
            text = "Default"
        return text

    @staticmethod
    def _normalize_gradient_triangular_color_order(value: str) -> str:
        text = str(value or "RBG").strip().upper()
        valid = {"RGB", "RBG", "GRB", "GBR", "BRG", "BGR"}
        if text not in valid:
            text = "RBG"
        return text

    @staticmethod
    def _normalize_gradient_rgb_scalar_mode(value: str) -> str:
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
    def _normalize_gradient_rgb_mask_strictness(value) -> float:
        try:
            return max(0.0, min(5.0, float(value)))
        except Exception:
            return 1.0

    @staticmethod
    def _normalize_gradient_classification_fit_mode(value: str) -> str:
        text = str(value or "triangle").strip().lower()
        if text not in {"triangle", "square"}:
            text = "triangle"
        return text

    @staticmethod
    def _normalize_gradient_rotation_preset(value: str) -> str:
        text = str(value or "Default").strip()
        valid = {"Default", "X +90", "X -90", "Y +90", "Y -90", "Y 180", "Z +90", "Z -90"}
        if text not in valid:
            text = "Default"
        return text

    @staticmethod
    def _normalize_gradient_network_component(value, max_components=None) -> str:
        text = str(value or "all").strip().lower()
        if text in {"", "all"}:
            return "all"
        if text.startswith("c"):
            text = text[1:]
        try:
            index = int(text)
        except Exception:
            return "all"
        if index < 1:
            return "all"
        if max_components is not None and index > int(max_components):
            return "all"
        return str(index)

    def _ensure_gradient_rotation_count(self, count: int) -> None:
        target = max(1, min(10, int(count)))
        current = list(self._gradient_component_rotations or [])
        while len(current) < target:
            current.append("Default")
        self._gradient_component_rotations = current[:10]

    def _available_gradient_network_component_count(self) -> int:
        results = self._last_gradients or {}
        try:
            if results:
                value = int(results.get("n_grad", 0))
                if value > 0:
                    return max(1, min(10, value))
        except Exception:
            pass
        bundle = self._gradient_precomputed_bundle
        if isinstance(bundle, dict):
            try:
                value = int(bundle.get("component_count_total", 0))
            except Exception:
                value = 0
            if value > 0:
                return max(1, min(10, value))
        return self._current_gradient_component_count()

    def _gradient_surface_procrustes_available(self) -> bool:
        bundle = self._gradient_precomputed_bundle
        if not isinstance(bundle, dict):
            return False
        avg = bundle.get("gradients_avg")
        if avg is None:
            return False
        try:
            avg_array = np.asarray(avg, dtype=float)
        except Exception:
            return False
        return avg_array.ndim == 2 and avg_array.shape[0] > 0 and avg_array.shape[1] > 0

    def _selected_gradient_hemisphere_mode(self) -> str:
        dialog = getattr(self, "_gradients_dialog", None)
        if dialog is not None:
            try:
                self._gradient_hemisphere_mode = self._normalize_gradient_hemisphere_mode(
                    dialog.selected_hemisphere()
                )
            except Exception:
                pass
        return self._normalize_gradient_hemisphere_mode(self._gradient_hemisphere_mode)

    def _selected_gradient_surface_mesh(self) -> str:
        dialog = getattr(self, "_gradients_dialog", None)
        if dialog is not None:
            try:
                self._gradient_surface_mesh = self._normalize_gradient_surface_mesh(
                    dialog.selected_surface_mesh()
                )
            except Exception:
                pass
        self._gradient_surface_mesh = self._normalize_gradient_surface_mesh(self._gradient_surface_mesh)
        return self._gradient_surface_mesh

    def _selected_gradient_surface_render_count(self) -> int:
        dialog = getattr(self, "_gradients_dialog", None)
        available = self._available_gradient_network_component_count()
        if dialog is not None and hasattr(dialog, "selected_surface_render_component_count"):
            try:
                self._gradient_surface_render_count = self._normalize_gradient_surface_render_count(
                    dialog.selected_surface_render_component_count(),
                    max_components=available,
                )
            except Exception:
                pass
        self._gradient_surface_render_count = self._normalize_gradient_surface_render_count(
            self._gradient_surface_render_count,
            max_components=available,
        )
        return self._gradient_surface_render_count

    def _selected_gradient_surface_procrustes(self) -> bool:
        dialog = getattr(self, "_gradients_dialog", None)
        if dialog is not None and hasattr(dialog, "use_surface_procrustes"):
            try:
                self._gradient_surface_procrustes = self._normalize_gradient_surface_procrustes(
                    dialog.use_surface_procrustes()
                )
            except Exception:
                pass
        if not self._gradient_surface_procrustes_available():
            self._gradient_surface_procrustes = False
        return bool(self._gradient_surface_procrustes)

    def _selected_gradient_classification_surface_mesh(self) -> str:
        dialog = getattr(self, "_gradients_dialog", None)
        if dialog is not None:
            try:
                self._gradient_classification_surface_mesh = self._normalize_gradient_surface_mesh(
                    dialog.selected_classification_surface_mesh()
                )
            except Exception:
                pass
        self._gradient_classification_surface_mesh = self._normalize_gradient_surface_mesh(
            self._gradient_classification_surface_mesh
        )
        return self._gradient_classification_surface_mesh

    def _selected_gradient_classification_hemisphere_mode(self) -> str:
        dialog = getattr(self, "_gradients_dialog", None)
        if dialog is not None:
            try:
                self._gradient_classification_hemisphere_mode = self._normalize_gradient_hemisphere_mode(
                    dialog.selected_classification_hemisphere()
                )
            except Exception:
                pass
        self._gradient_classification_hemisphere_mode = self._normalize_gradient_hemisphere_mode(
            self._gradient_classification_hemisphere_mode
        )
        return self._gradient_classification_hemisphere_mode

    def _selected_gradient_scatter_rotation(self) -> str:
        dialog = getattr(self, "_gradients_dialog", None)
        if dialog is not None:
            try:
                self._gradient_scatter_rotation = self._normalize_gradient_scatter_rotation(
                    dialog.selected_scatter_rotation()
                )
            except Exception:
                pass
        self._gradient_scatter_rotation = self._normalize_gradient_scatter_rotation(
            self._gradient_scatter_rotation
        )
        return self._gradient_scatter_rotation

    def _selected_gradient_triangular_rgb(self) -> bool:
        dialog = getattr(self, "_gradients_dialog", None)
        if dialog is not None:
            try:
                self._gradient_scatter_triangular_rgb = bool(dialog.use_triangular_rgb())
            except Exception:
                pass
        self._gradient_scatter_triangular_rgb = bool(self._gradient_scatter_triangular_rgb)
        return self._gradient_scatter_triangular_rgb

    def _selected_gradient_preload_matching_paths(self) -> bool:
        dialog = getattr(self, "_gradients_dialog", None)
        if dialog is not None:
            try:
                self._gradient_preload_matching_paths = bool(
                    dialog.preload_matching_paths()
                )
            except Exception:
                pass
        self._gradient_preload_matching_paths = bool(
            getattr(self, "_gradient_preload_matching_paths", False)
        )
        return self._gradient_preload_matching_paths

    def _selected_gradient_classification_fit_mode(self) -> str:
        dialog = getattr(self, "_gradients_dialog", None)
        if dialog is not None:
            try:
                self._gradient_classification_fit_mode = self._normalize_gradient_classification_fit_mode(
                    dialog.selected_classification_fit_mode()
                )
            except Exception:
                pass
        self._gradient_classification_fit_mode = self._normalize_gradient_classification_fit_mode(
            self._gradient_classification_fit_mode
        )
        return self._gradient_classification_fit_mode

    def _selected_gradient_triangular_color_order(self) -> str:
        dialog = getattr(self, "_gradients_dialog", None)
        if dialog is not None:
            try:
                self._gradient_triangular_color_order = self._normalize_gradient_triangular_color_order(
                    dialog.selected_triangular_color_order()
                )
            except Exception:
                pass
        self._gradient_triangular_color_order = self._normalize_gradient_triangular_color_order(
            self._gradient_triangular_color_order
        )
        return self._gradient_triangular_color_order

    def _selected_gradient_rgb_scalar_mode(self) -> str:
        dialog = getattr(self, "_gradients_dialog", None)
        if dialog is not None and hasattr(dialog, "selected_rgb_scalar_mode"):
            try:
                self._gradient_rgb_scalar_mode = self._normalize_gradient_rgb_scalar_mode(
                    dialog.selected_rgb_scalar_mode()
                )
            except Exception:
                pass
        self._gradient_rgb_scalar_mode = self._normalize_gradient_rgb_scalar_mode(
            getattr(self, "_gradient_rgb_scalar_mode", "barycentric")
        )
        return self._gradient_rgb_scalar_mode

    def _selected_gradient_rgb_mask_strictness(self) -> float:
        dialog = getattr(self, "_gradients_dialog", None)
        if dialog is not None:
            try:
                self._gradient_rgb_mask_strictness = self._normalize_gradient_rgb_mask_strictness(
                    dialog.selected_rgb_mask_strictness()
                )
            except Exception:
                pass
        self._gradient_rgb_mask_strictness = self._normalize_gradient_rgb_mask_strictness(
            getattr(self, "_gradient_rgb_mask_strictness", 1.0)
        )
        return self._gradient_rgb_mask_strictness

    def _selected_gradient_classification_colormap(self) -> str:
        dialog = getattr(self, "_gradients_dialog", None)
        names = self._available_colormap_names()
        if dialog is not None:
            try:
                value = str(dialog.selected_classification_colormap() or "").strip()
                if value:
                    self._gradient_classification_colormap_name = value
            except Exception:
                pass
        current = str(self._gradient_classification_colormap_name or "").strip()
        if current not in names:
            current = self._gradient_colormap_name if self._gradient_colormap_name in names else (names[0] if names else "spectrum_fsl")
            self._gradient_classification_colormap_name = current
        return current

    def _selected_gradient_classification_component(self) -> str:
        dialog = getattr(self, "_gradients_dialog", None)
        max_components = self._available_gradient_network_component_count()
        if dialog is not None:
            try:
                self._gradient_classification_component = self._normalize_gradient_classification_component(
                    dialog.selected_classification_component(),
                    max_components=max_components,
                )
            except Exception:
                pass
        self._gradient_classification_component = self._normalize_gradient_classification_component(
            self._gradient_classification_component,
            max_components=max_components,
        )
        return self._gradient_classification_component

    def _selected_gradient_classification_x_axis(self) -> str:
        dialog = getattr(self, "_gradients_dialog", None)
        if dialog is not None:
            try:
                self._gradient_classification_x_axis = self._normalize_gradient_classification_axis(
                    dialog.selected_classification_x_axis(),
                    default="gradient2",
                )
            except Exception:
                pass
        self._gradient_classification_x_axis = self._normalize_gradient_classification_axis(
            self._gradient_classification_x_axis,
            default="gradient2",
        )
        return self._gradient_classification_x_axis

    def _selected_gradient_classification_y_axis(self) -> str:
        dialog = getattr(self, "_gradients_dialog", None)
        if dialog is not None:
            try:
                self._gradient_classification_y_axis = self._normalize_gradient_classification_axis(
                    dialog.selected_classification_y_axis(),
                    default="gradient1",
                )
            except Exception:
                pass
        self._gradient_classification_y_axis = self._normalize_gradient_classification_axis(
            self._gradient_classification_y_axis,
            default="gradient1",
        )
        return self._gradient_classification_y_axis

    def _selected_gradient_classification_z_axis(self) -> str:
        dialog = getattr(self, "_gradients_dialog", None)
        if dialog is not None and hasattr(dialog, "selected_classification_z_axis"):
            try:
                self._gradient_classification_z_axis = self._normalize_gradient_classification_axis(
                    dialog.selected_classification_z_axis(),
                    default="none",
                )
            except Exception:
                pass
        self._gradient_classification_z_axis = self._normalize_gradient_classification_axis(
            getattr(self, "_gradient_classification_z_axis", "none"),
            default="none",
        )
        return self._gradient_classification_z_axis

    def _selected_gradient_classification_ref_entry_id(self):
        dialog = getattr(self, "_gradients_dialog", None)
        if dialog is not None and hasattr(dialog, "selected_classification_ref_matrix_entry_id"):
            try:
                candidate = dialog.selected_classification_ref_matrix_entry_id()
                if candidate in self._entries:
                    self._gradient_classification_ref_entry_id = candidate
            except Exception:
                pass

        stored = getattr(self, "_gradient_classification_ref_entry_id", None)
        if stored in self._entries:
            current_gradient_entry_id = self._selected_gradient_entry_id()
            if stored != current_gradient_entry_id or self._gradient_use_precomputed_bundle:
                return stored

        for option in self._available_gradient_reference_matrix_entries():
            entry_id = option.get("id")
            if entry_id in self._entries:
                self._gradient_classification_ref_entry_id = entry_id
                return entry_id
        self._gradient_classification_ref_entry_id = None
        return None

    def _selected_gradient_classification_ignore_lh_parcel(self) -> str:
        dialog = getattr(self, "_gradients_dialog", None)
        if dialog is not None and hasattr(dialog, "selected_classification_ignore_lh_parcel"):
            try:
                self._gradient_classification_ignore_lh_parcel = str(
                    dialog.selected_classification_ignore_lh_parcel() or ""
                ).strip()
            except Exception:
                pass
        self._gradient_classification_ignore_lh_parcel = str(
            self._gradient_classification_ignore_lh_parcel or ""
        ).strip()
        return self._gradient_classification_ignore_lh_parcel

    def _selected_gradient_classification_ignore_rh_parcel(self) -> str:
        dialog = getattr(self, "_gradients_dialog", None)
        if dialog is not None and hasattr(dialog, "selected_classification_ignore_rh_parcel"):
            try:
                self._gradient_classification_ignore_rh_parcel = str(
                    dialog.selected_classification_ignore_rh_parcel() or ""
                ).strip()
            except Exception:
                pass
        self._gradient_classification_ignore_rh_parcel = str(
            self._gradient_classification_ignore_rh_parcel or ""
        ).strip()
        return self._gradient_classification_ignore_rh_parcel

    def _gradient_classification_ignore_parcel_options(self):
        results = self._last_gradients or {}
        projection_labels_raw = results.get("projection_labels", None)
        if projection_labels_raw is None:
            return {"lh": [], "rh": []}
        try:
            projection_labels = np.asarray(projection_labels_raw, dtype=int).reshape(-1)
        except Exception:
            return {"lh": [], "rh": []}
        parcel_names = _to_string_list(results.get("parcel_names"))
        if projection_labels.size == 0:
            return {"lh": [], "rh": []}
        if not parcel_names or len(parcel_names) != projection_labels.size:
            parcel_names = [f"Parcel {int(label)}" for label in projection_labels.tolist()]
        try:
            hemisphere_codes = np.asarray(self._gradient_projection_hemisphere_codes(), dtype=int).reshape(-1)
        except Exception:
            hemisphere_codes = np.full(projection_labels.shape, -1, dtype=int)
        if hemisphere_codes.shape != projection_labels.shape:
            hemisphere_codes = np.full(projection_labels.shape, -1, dtype=int)

        lh_names = []
        rh_names = []
        seen_lh = set()
        seen_rh = set()
        for name, code in zip(parcel_names, hemisphere_codes.tolist()):
            text = str(name or "").strip()
            if not text:
                continue
            if int(code) in {0, 2} and text not in seen_lh:
                seen_lh.add(text)
                lh_names.append(text)
            if int(code) in {1, 2} and text not in seen_rh:
                seen_rh.add(text)
                rh_names.append(text)
        return {"lh": lh_names, "rh": rh_names}

    def _is_gradient_classification_axis_available(self, axis_key: str, results=None) -> bool:
        current = self._last_gradients if results is None else results
        if not current:
            return False
        axis = self._normalize_gradient_classification_axis(axis_key)
        try:
            n_grad = int(current.get("n_grad", 0))
        except Exception:
            n_grad = 0
        if axis == "gradient1":
            return n_grad >= 1
        if axis == "gradient2":
            return n_grad >= 2
        if axis == "gradient3":
            return n_grad >= 3
        if axis == "gradient_ref1":
            return self._selected_gradient_classification_ref_entry_id() in self._entries
        if axis == "gradient_ref2":
            return self._selected_gradient_classification_ref_entry_id() in self._entries
        if axis == "gradient_ref3":
            return self._selected_gradient_classification_ref_entry_id() in self._entries
        if axis == "none":
            return True
        if axis == "spatial":
            try:
                projection_labels = np.asarray(current.get("projection_labels"), dtype=int).reshape(-1)
            except Exception:
                projection_labels = np.zeros(0, dtype=int)
            return projection_labels.size > 0 and (
                bool(str(current.get("template_path") or "").strip()) or self._active_parcellation_img is not None
            )
        return False

    @staticmethod
    def _gradient_classification_axis_label(axis_key: str, axis_role: str = "x") -> str:
        axis = GradientDialogController._normalize_gradient_classification_axis(axis_key)
        if axis == "gradient1":
            return "Gradient 1"
        if axis == "gradient2":
            return "Gradient 2"
        if axis == "gradient3":
            return "Gradient 3"
        if axis == "gradient_ref1":
            return "Gradient Ref 1"
        if axis == "gradient_ref2":
            return "Gradient Ref 2"
        if axis == "gradient_ref3":
            return "Gradient Ref 3"
        if axis == "none":
            return ""
        return "Spatial 1" if str(axis_role).strip().lower() == "x" else "Spatial 2"

    @staticmethod
    def _classification_uses_reference_axis(*axes) -> bool:
        axes = {
            GradientDialogController._normalize_gradient_classification_axis(axis, default="none")
            for axis in axes
        }
        return bool(axes & {"gradient_ref1", "gradient_ref2", "gradient_ref3"})

    @staticmethod
    def _classification_reference_component_count(*axes) -> int:
        count = 0
        for raw_axis in axes:
            axis = GradientDialogController._normalize_gradient_classification_axis(raw_axis, default="none")
            if axis == "gradient_ref1":
                count = max(count, 1)
            elif axis == "gradient_ref2":
                count = max(count, 2)
            elif axis == "gradient_ref3":
                count = max(count, 3)
        return count

    @staticmethod
    def _classification_axis_component_count(axis_key: str) -> int:
        axis = GradientDialogController._normalize_gradient_classification_axis(axis_key, default="none")
        if axis in {"gradient3", "gradient_ref3"}:
            return 3
        if axis in {"gradient2", "gradient_ref2"}:
            return 2
        if axis in {"gradient1", "gradient_ref1"}:
            return 1
        return 0

    @staticmethod
    def _infer_projection_hemisphere_from_name(name):
        text = str(name or "").strip().lower()
        if not text:
            return None
        if any(token in text for token in ("brainstem", "brain-stem", "midbrain")):
            return "midline"
        tokens = [token for token in re.split(r"[^a-z0-9]+", text) if token]
        if tokens:
            first = tokens[0]
            if first in {"lh", "left", "l"}:
                return "lh"
            if first in {"rh", "right", "r"}:
                return "rh"
            if "lh" in tokens or "left" in tokens:
                return "lh"
            if "rh" in tokens or "right" in tokens:
                return "rh"
        if text.startswith(("lh_", "lh-", "left_", "left-", "ctx-lh", "hemi-l")):
            return "lh"
        if text.startswith(("rh_", "rh-", "right_", "right-", "ctx-rh", "hemi-r")):
            return "rh"
        return None

    def _gradient_projection_hemisphere_codes(self):
        results = self._last_gradients or {}
        if not results:
            raise RuntimeError("No gradients are available.")
        projection_labels = np.asarray(results.get("projection_labels"), dtype=int).reshape(-1)
        if projection_labels.size == 0:
            raise RuntimeError("No projected parcel labels are available.")

        cached = results.get("hemisphere_codes")
        if isinstance(cached, dict):
            cached_labels = np.asarray(cached.get("projection_labels"), dtype=int).reshape(-1)
            cached_codes = np.asarray(cached.get("codes"), dtype=int).reshape(-1)
            if (
                cached_labels.shape == projection_labels.shape
                and np.array_equal(cached_labels, projection_labels)
                and cached_codes.shape == projection_labels.shape
            ):
                return cached_codes

        codes = np.full(projection_labels.shape, -1, dtype=int)
        parcel_names = _to_string_list(results.get("parcel_names"))
        if parcel_names and len(parcel_names) == projection_labels.size:
            for idx, parcel_name in enumerate(parcel_names):
                hemisphere = self._infer_projection_hemisphere_from_name(parcel_name)
                if hemisphere == "lh":
                    codes[idx] = 0
                elif hemisphere == "rh":
                    codes[idx] = 1
                elif hemisphere == "midline":
                    codes[idx] = 2

        unresolved = codes < 0
        if np.any(unresolved):
            template_img, template_data = self._gradient_template_img_and_data()
            centroid_codes = None
            try:
                centroids_world = np.asarray(
                    nettools.compute_centroids(
                        template_img,
                        labels=np.asarray(projection_labels, dtype=int),
                        world=True,
                    ),
                    dtype=float,
                )
                if centroids_world.shape[0] == projection_labels.size:
                    centroid_x = np.asarray(centroids_world[:, 0], dtype=float)
                    finite_x = centroid_x[np.isfinite(centroid_x)]
                    if finite_x.size and np.nanmin(finite_x) < 0.0 < np.nanmax(finite_x):
                        centroid_codes = np.where(centroid_x < 0.0, 0, 1).astype(int, copy=False)
            except Exception:
                centroid_codes = None

            if centroid_codes is None:
                centroids_vox = np.asarray(
                    nettools.compute_centroids(
                        template_img,
                        labels=np.asarray(projection_labels, dtype=int),
                        world=False,
                    ),
                    dtype=float,
                )
                if centroids_vox.shape[0] != projection_labels.size:
                    raise RuntimeError("Parcel centroid count does not match the projected labels.")
                midline_x = float(np.asarray(template_data, dtype=int).shape[0] * 0.5)
                centroid_codes = np.where(centroids_vox[:, 0] < midline_x, 0, 1).astype(int, copy=False)

            codes[unresolved] = centroid_codes[unresolved]

        results["hemisphere_codes"] = {
            "projection_labels": np.asarray(projection_labels, dtype=int),
            "codes": np.asarray(codes, dtype=int),
        }
        return np.asarray(codes, dtype=int)

    def _gradient_projection_hemisphere_mask(self, hemisphere_mode: str, projection_labels=None):
        mode = self._normalize_gradient_hemisphere_mode(hemisphere_mode)
        labels = np.asarray(
            self._last_gradients.get("projection_labels")
            if projection_labels is None
            else projection_labels,
            dtype=int,
        ).reshape(-1)
        if labels.size == 0 or mode in {"both", "separate"}:
            return np.ones(labels.shape, dtype=bool)
        codes = self._gradient_projection_hemisphere_codes()
        if codes.shape != labels.shape:
            raise RuntimeError("Hemisphere membership is out of sync with the projected labels.")
        if mode == "lh":
            return np.asarray((codes == 0) | (codes == 2), dtype=bool)
        return np.asarray((codes == 1) | (codes == 2), dtype=bool)

    def _can_classify_gradients(self) -> bool:
        results = self._last_gradients or {}
        if not results:
            return False
        try:
            n_grad = int(results.get("n_grad", 0))
        except Exception:
            n_grad = 0
        if n_grad < 1:
            return False
        x_axis = self._selected_gradient_classification_x_axis()
        y_axis = self._selected_gradient_classification_y_axis()
        z_axis = self._selected_gradient_classification_z_axis()
        return (
            self._is_gradient_classification_axis_available(x_axis, results)
            and self._is_gradient_classification_axis_available(y_axis, results)
            and self._is_gradient_classification_axis_available(z_axis, results)
        )

    def _current_gradient_rotation_presets(self):
        dialog = getattr(self, "_gradients_dialog", None)
        count = self._current_gradient_component_count()
        if dialog is not None:
            try:
                presets = [
                    self._normalize_gradient_rotation_preset(value)
                    for value in dialog.rotation_presets()[:count]
                ]
                while len(presets) < count:
                    presets.append("Default")
                for idx, value in enumerate(presets):
                    if idx < len(self._gradient_component_rotations):
                        self._gradient_component_rotations[idx] = value
                    else:
                        self._gradient_component_rotations.append(value)
            except Exception:
                pass
        self._ensure_gradient_rotation_count(count)
        return [
            self._normalize_gradient_rotation_preset(value)
            for value in self._gradient_component_rotations[:count]
        ]

    def _selected_gradient_network_component(self) -> str:
        dialog = getattr(self, "_gradients_dialog", None)
        max_components = self._available_gradient_network_component_count()
        if dialog is not None:
            try:
                self._gradient_network_component = self._normalize_gradient_network_component(
                    dialog.selected_network_component(),
                    max_components=max_components,
                )
            except Exception:
                pass
        self._gradient_network_component = self._normalize_gradient_network_component(
            self._gradient_network_component,
            max_components=max_components,
        )
        return self._gradient_network_component

    def _has_square_current_matrix(self) -> bool:
        if self._current_matrix is None:
            return False
        matrix = np.asarray(self._current_matrix)
        return matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]

    def _update_gradients_button(self) -> None:
        enabled = bool(self._available_gradient_matrix_entries()) or isinstance(
            self._gradient_precomputed_bundle,
            dict,
        )
        if hasattr(self, "gradients_open_button"):
            self.gradients_open_button.setEnabled(enabled)
        if hasattr(self, "compute_gradients_action"):
            self.compute_gradients_action.setEnabled(enabled)
        self._sync_gradients_dialog_state()

    def _set_gradient_source_mode(self, use_precomputed: bool, *, reset_results: bool = False) -> None:
        next_mode = bool(use_precomputed) and isinstance(self._gradient_precomputed_bundle, dict)
        if next_mode == bool(self._gradient_use_precomputed_bundle):
            if reset_results:
                self._reset_gradients_output()
            return
        self._gradient_use_precomputed_bundle = next_mode
        if reset_results:
            self._reset_gradients_output()
        else:
            self._sync_gradients_dialog_state()

    def _on_gradient_component_changed(self, value: int) -> None:
        try:
            self._gradient_component_count = max(1, min(10, int(value)))
        except Exception:
            self._gradient_component_count = 4
        self._ensure_gradient_rotation_count(self._gradient_component_count)
        self._gradient_network_component = self._normalize_gradient_network_component(
            self._gradient_network_component,
            max_components=self._available_gradient_network_component_count(),
        )
        self._gradient_classification_component = self._normalize_gradient_classification_component(
            self._gradient_classification_component,
            max_components=self._available_gradient_network_component_count(),
        )
        self._sync_gradients_dialog_state()

    def _on_gradient_colormap_changed(self, value: str) -> None:
        text = str(value or "").strip()
        if text:
            self._gradient_colormap_name = text

    def _on_gradient_hemisphere_changed(self, value: str) -> None:
        self._gradient_hemisphere_mode = self._normalize_gradient_hemisphere_mode(value)

    def _on_gradient_surface_mesh_changed(self, value: str) -> None:
        self._gradient_surface_mesh = self._normalize_gradient_surface_mesh(value)

    def _on_gradient_surface_render_count_changed(self, value: int) -> None:
        self._gradient_surface_render_count = self._normalize_gradient_surface_render_count(
            value,
            max_components=self._available_gradient_network_component_count(),
        )

    def _on_gradient_surface_procrustes_changed(self, enabled: bool) -> None:
        self._gradient_surface_procrustes = self._normalize_gradient_surface_procrustes(enabled)

    def _on_gradient_classification_surface_mesh_changed(self, value: str) -> None:
        self._gradient_classification_surface_mesh = self._normalize_gradient_surface_mesh(value)

    def _on_gradient_classification_hemisphere_changed(self, value: str) -> None:
        self._gradient_classification_hemisphere_mode = self._normalize_gradient_hemisphere_mode(value)

    def _on_gradient_scatter_rotation_changed(self, value: str) -> None:
        self._gradient_scatter_rotation = self._normalize_gradient_scatter_rotation(value)

    def _on_gradient_triangular_rgb_changed(self, enabled: bool) -> None:
        self._gradient_scatter_triangular_rgb = bool(enabled)
        self._sync_gradients_dialog_state()

    def _on_gradient_preload_matching_paths_changed(self, enabled: bool) -> None:
        self._gradient_preload_matching_paths = bool(enabled)

    def _on_gradient_classification_fit_mode_changed(self, value: str) -> None:
        self._gradient_classification_fit_mode = self._normalize_gradient_classification_fit_mode(value)

    def _on_gradient_triangular_color_order_changed(self, value: str) -> None:
        self._gradient_triangular_color_order = self._normalize_gradient_triangular_color_order(value)

    def _on_gradient_rgb_scalar_mode_changed(self, value: str) -> None:
        self._gradient_rgb_scalar_mode = self._normalize_gradient_rgb_scalar_mode(value)

    def _on_gradient_rgb_mask_strictness_changed(self, value) -> None:
        self._gradient_rgb_mask_strictness = self._normalize_gradient_rgb_mask_strictness(value)

    def _on_gradient_classification_colormap_changed(self, value: str) -> None:
        text = str(value or "").strip()
        if text:
            self._gradient_classification_colormap_name = text

    def _on_gradient_classification_component_changed(self, value: str) -> None:
        self._gradient_classification_component = self._normalize_gradient_classification_component(
            value,
            max_components=self._available_gradient_network_component_count(),
        )

    def _on_gradient_classification_x_axis_changed(self, value: str) -> None:
        self._gradient_classification_x_axis = self._normalize_gradient_classification_axis(
            value,
            default="gradient2",
        )
        self._sync_gradients_dialog_state()

    def _on_gradient_classification_y_axis_changed(self, value: str) -> None:
        self._gradient_classification_y_axis = self._normalize_gradient_classification_axis(
            value,
            default="gradient1",
        )
        self._sync_gradients_dialog_state()

    def _on_gradient_classification_z_axis_changed(self, value: str) -> None:
        self._gradient_classification_z_axis = self._normalize_gradient_classification_axis(
            value,
            default="none",
        )
        self._sync_gradients_dialog_state()

    def _on_gradient_classification_ref_entry_changed(self, entry_id) -> None:
        self._gradient_classification_ref_entry_id = entry_id if entry_id in self._entries else None
        self._sync_gradients_dialog_state()

    def _on_gradient_classification_ignore_lh_changed(self, value: str) -> None:
        self._gradient_classification_ignore_lh_parcel = str(value or "").strip()

    def _on_gradient_classification_ignore_rh_changed(self, value: str) -> None:
        self._gradient_classification_ignore_rh_parcel = str(value or "").strip()

    @staticmethod
    def _load_gradient_classification_adjacency_npz(path: Path):
        with np.load(path, allow_pickle=True) as npz:
            if "adjacency_mat" not in npz:
                raise KeyError("Key 'adjacency_mat' was not found in the selected NPZ.")
            adjacency = np.asarray(npz["adjacency_mat"], dtype=float)
            parcel_labels = None
            for key in ("parcel_labels",) + PARCEL_LABEL_KEYS:
                if key in npz:
                    parcel_labels = npz[key]
                    break
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("adjacency_mat must be a square 2D matrix.")
        label_indices = _coerce_label_indices(parcel_labels, adjacency.shape[0])
        if label_indices is None:
            raise ValueError(
                "parcel_labels is missing, invalid, or does not match adjacency_mat size."
            )
        labels_array = np.asarray(label_indices, dtype=int)
        if np.unique(labels_array).size != labels_array.size:
            raise ValueError("parcel_labels in the adjacency NPZ must be unique.")
        return {
            "path": str(path),
            "adjacency": adjacency,
            "parcel_labels": labels_array,
        }

    def _gradient_classification_adjacency_data(self):
        path_text = str(self._gradient_classification_adjacency_path or "").strip()
        if not path_text:
            return None
        cached = self._gradient_classification_adjacency_cache
        if isinstance(cached, dict) and str(cached.get("path") or "") == path_text:
            return cached
        path = Path(path_text)
        if not path.exists():
            raise RuntimeError(f"Adjacency file not found: {path.name}")
        data = self._load_gradient_classification_adjacency_npz(path)
        self._gradient_classification_adjacency_cache = data
        return data

    def _set_gradient_classification_adjacency(self, path: Path) -> bool:
        try:
            data = self._load_gradient_classification_adjacency_npz(path)
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to load classification adjacency: {exc}")
            return False
        self._gradient_classification_adjacency_path = str(path)
        self._gradient_classification_adjacency_cache = data
        self._sync_gradients_dialog_state()
        self.statusBar().showMessage(f"Loaded classification adjacency from {path.name}.")
        return True

    def _select_gradient_classification_adjacency(self) -> None:
        start_dir = self._default_results_dir()
        existing = str(self._gradient_classification_adjacency_path or "").strip()
        if existing:
            existing_path = Path(existing)
            if existing_path.exists():
                start_dir = existing_path.parent
        else:
            source_path = self._current_source_path()
            if source_path is not None and source_path.exists():
                start_dir = source_path.parent
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select classification adjacency NPZ",
            str(start_dir),
            "NumPy archive (*.npz);;All files (*)",
        )
        if not selected:
            return
        self._set_gradient_classification_adjacency(Path(selected))

    def _clear_gradient_classification_adjacency(self, *, show_status: bool = True) -> None:
        self._gradient_classification_adjacency_path = ""
        self._gradient_classification_adjacency_cache = None
        self._sync_gradients_dialog_state()
        if show_status:
            self.statusBar().showMessage("Removed classification adjacency.")

    @staticmethod
    def _npz_optional_scalar_text(npz, *keys):
        for key in keys:
            if key not in npz:
                continue
            try:
                values = np.asarray(npz[key])
            except Exception:
                continue
            if values.ndim == 0:
                return _display_text(values.item()).strip()
            if values.size == 1:
                return _display_text(values.reshape(-1)[0]).strip()
        return ""

    @staticmethod
    def _npz_optional_display_vector(npz, *keys):
        for key in keys:
            if key not in npz:
                continue
            try:
                values = _flatten_display_vector(npz[key])
            except Exception:
                values = None
            if values is not None:
                return [str(value) for value in values]
        return None

    @staticmethod
    def _npz_optional_array(npz, *keys):
        for key in keys:
            if key not in npz:
                continue
            try:
                return np.asarray(npz[key])
            except Exception:
                continue
        return None

    @classmethod
    def _npz_subject_session_vectors(cls, npz, *, path=None):
        subjects = cls._npz_optional_display_vector(
            npz,
            "subject_id_list",
            "participant_id_list",
            "subject_ids",
            "participant_ids",
        )
        sessions = cls._npz_optional_display_vector(
            npz,
            "session_id_list",
            "session_ids",
            "sessions",
        )
        if subjects:
            return list(subjects or []), list(sessions or [])

        covars = None
        if "covars" in npz:
            try:
                covars = np.asarray(npz["covars"])
            except Exception:
                covars = None
        if covars is not None and getattr(covars.dtype, "names", None):
            field_lookup = {str(name).strip().lower(): str(name) for name in covars.dtype.names}
            subject_field = (
                field_lookup.get("participant_id")
                or field_lookup.get("subject_id")
                or field_lookup.get("subject")
                or field_lookup.get("sub")
                or field_lookup.get("id")
            )
            session_field = field_lookup.get("session_id") or field_lookup.get("session") or field_lookup.get("ses")
            if subject_field:
                subjects = [_display_text(value).strip() for value in covars[subject_field].reshape(-1).tolist()]
                sessions = (
                    [_display_text(value).strip() for value in covars[session_field].reshape(-1).tolist()]
                    if session_field
                    else []
                )
                return subjects, sessions

        if path is not None and _load_covars_info is not None and _covars_to_rows is not None:
            try:
                covars_info = _load_covars_info(Path(path))
                covars_columns, covars_rows = _covars_to_rows(covars_info)
            except Exception:
                covars_columns, covars_rows = [], []
            column_lookup = {str(column).strip().lower(): idx for idx, column in enumerate(covars_columns or [])}
            subject_idx = None
            for key in ("participant_id", "subject_id", "subject", "sub", "id"):
                if key in column_lookup:
                    subject_idx = column_lookup[key]
                    break
            session_idx = None
            for key in ("session_id", "session", "ses"):
                if key in column_lookup:
                    session_idx = column_lookup[key]
                    break
            if subject_idx is not None and covars_rows:
                subjects = []
                sessions = []
                for row in covars_rows:
                    if isinstance(row, dict):
                        row_lookup = {str(key).strip().lower(): value for key, value in row.items()}
                        subjects.append(
                            _display_text(
                                row_lookup.get("participant_id")
                                or row_lookup.get("subject_id")
                                or row_lookup.get("subject")
                                or row_lookup.get("sub")
                                or row_lookup.get("id")
                                or ""
                            ).strip()
                        )
                        sessions.append(
                            _display_text(
                                row_lookup.get("session_id")
                                or row_lookup.get("session")
                                or row_lookup.get("ses")
                                or ""
                            ).strip()
                        )
                    else:
                        row_values = list(row)
                        subjects.append(_display_text(row_values[subject_idx]).strip() if subject_idx < len(row_values) else "")
                        sessions.append(_display_text(row_values[session_idx]).strip() if session_idx is not None and session_idx < len(row_values) else "")
                return subjects, sessions
        return [], []

    @staticmethod
    def _metabolite_names_from_array(values):
        if values is None:
            return []
        try:
            names = [str(value).strip() for value in np.asarray(values).reshape(-1).tolist()]
        except Exception:
            return []
        return [name for name in names if name and name.lower() not in {"nan", "none"}]

    @classmethod
    def _reduce_metabolite_profiles(
        cls,
        profiles,
        metabolites,
        *,
        n_nodes,
        keep_indices=None,
        row_index=None,
        subject_count=None,
        preserve_subjects=False,
    ):
        names = cls._metabolite_names_from_array(metabolites)
        if not names:
            return None, []
        try:
            arr = np.asarray(profiles, dtype=float)
        except Exception:
            return None, []
        if arr.ndim < 2 or arr.size == 0:
            return None, []

        try:
            n_nodes = int(n_nodes)
        except Exception:
            n_nodes = 0
        if n_nodes <= 0:
            return None, []

        try:
            subject_count = int(subject_count) if subject_count is not None else 0
        except Exception:
            subject_count = 0

        subject_axis = None
        if subject_count > 0:
            subject_candidates = [
                axis
                for axis, size in enumerate(arr.shape)
                if int(size) == subject_count and int(size) != n_nodes
            ]
            if not subject_candidates:
                subject_candidates = [
                    axis
                    for axis, size in enumerate(arr.shape)
                    if int(size) == subject_count
                ]
            if subject_candidates:
                subject_axis = int(subject_candidates[0])

        if row_index is not None and subject_axis is not None:
            try:
                idx = int(row_index)
            except Exception:
                idx = -1
            if 0 <= idx < arr.shape[subject_axis]:
                arr = np.take(arr, idx, axis=subject_axis)
                subject_axis = None

        n_met = int(len(names))
        node_axes = [
            axis
            for axis, size in enumerate(arr.shape)
            if int(size) == n_nodes and axis != subject_axis
        ]
        if not node_axes:
            return None, []
        if keep_indices is not None:
            node_axis = int(node_axes[0])
        else:
            preferred = [axis for axis in node_axes if axis != 0 or subject_axis == 0]
            node_axis = int(preferred[0] if preferred else node_axes[0])

        met_axes = [
            axis
            for axis, size in enumerate(arr.shape)
            if int(size) == n_met and axis not in {node_axis, subject_axis}
        ]
        if not met_axes:
            return None, []
        met_axis = int(met_axes[-1])

        if keep_indices is not None:
            try:
                keep = np.asarray(keep_indices, dtype=int).reshape(-1)
            except Exception:
                keep = np.zeros(0, dtype=int)
            if keep.size:
                keep = keep[(keep >= 0) & (keep < arr.shape[node_axis])]
                if keep.size == 0:
                    return None, []
                arr = np.take(arr, keep, axis=node_axis)

        original_axes = list(range(arr.ndim))
        if bool(preserve_subjects) and subject_axis is not None:
            ordered_axes = [
                subject_axis,
                node_axis,
                met_axis,
            ] + [
                axis
                for axis in original_axes
                if axis not in {subject_axis, node_axis, met_axis}
            ]
            arr = np.transpose(arr, ordered_axes)
            if arr.ndim > 3:
                profile = np.nanmean(arr, axis=tuple(range(3, arr.ndim)))
            else:
                profile = arr
        else:
            moved_axes = [node_axis] + [axis for axis in original_axes if axis != node_axis]
            arr = np.moveaxis(arr, node_axis, 0)
            met_axis = moved_axes.index(met_axis)
            arr = np.moveaxis(arr, met_axis, -1)
            if arr.ndim > 2:
                profile = np.nanmean(arr, axis=tuple(range(1, arr.ndim - 1)))
            else:
                profile = arr
        profile = np.asarray(profile, dtype=float)
        if profile.ndim not in {2, 3} or profile.shape[0] == 0:
            return None, []
        met_size = profile.shape[-1]
        if met_size != n_met:
            count = min(met_size, n_met)
            if count <= 0:
                return None, []
            profile = profile[..., :count]
            names = names[:count]
        return profile, names

    @classmethod
    def _reduce_water_signal_profile(
        cls,
        water_signal,
        *,
        n_nodes,
        keep_indices=None,
        row_index=None,
        subject_count=None,
        preserve_subjects=False,
    ):
        if water_signal is None:
            return None
        try:
            arr = np.asarray(water_signal, dtype=float)
        except Exception:
            return None
        if arr.ndim == 0 or arr.size == 0:
            return None
        if arr.ndim == 1:
            arr = arr[:, np.newaxis]
        elif arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
        profile, names = cls._reduce_metabolite_profiles(
            arr,
            ["water"],
            n_nodes=n_nodes,
            keep_indices=keep_indices,
            row_index=row_index,
            subject_count=subject_count,
            preserve_subjects=preserve_subjects,
        )
        if profile is None or not names:
            return None
        return np.asarray(profile, dtype=float)

    @staticmethod
    def _append_water_signal_profile(profile, names, water_profile):
        if profile is None or water_profile is None:
            return profile, names
        normalized = {str(name).strip().lower() for name in list(names or [])}
        if "water" in normalized:
            return profile, names
        try:
            profile_arr = np.asarray(profile, dtype=float)
            water_arr = np.asarray(water_profile, dtype=float)
        except Exception:
            return profile, names
        if profile_arr.ndim not in {2, 3} or water_arr.ndim not in {2, 3}:
            return profile, names
        if water_arr.shape[-1] != 1:
            return profile, names
        if profile_arr.shape[:-1] != water_arr.shape[:-1]:
            return profile, names
        return np.concatenate([profile_arr, water_arr], axis=-1), list(names or []) + ["water"]

    @classmethod
    def _load_metabolite_profiles_from_npz(
        cls,
        path: Path,
        *,
        n_nodes,
        keep_indices=None,
        row_index=None,
        preserve_subjects=False,
    ):
        try:
            with np.load(Path(path), allow_pickle=True) as npz:
                metabolites = cls._npz_optional_array(npz, "metabolites", "metabolites.npy")
                profiles = cls._npz_optional_array(
                    npz,
                    "metab_profiles_subj_list",
                    "metab_profiles_subj_list.npy",
                )
                water_signal = cls._npz_optional_array(
                    npz,
                    "water_signal_subj_list",
                    "water_signal_subj_list.npy",
                    "water_signal",
                    "water_signal.npy",
                )
                subject_values, _session_values = cls._npz_subject_session_vectors(npz, path=path)
                subject_count = len(subject_values or [])
                if subject_count <= 0:
                    matrix_stack = cls._npz_optional_array(npz, "matrix_subj_list", "matrix_subj_list.npy")
                    if matrix_stack is not None and np.asarray(matrix_stack).ndim >= 3:
                        subject_count = int(np.asarray(matrix_stack).shape[0])
        except Exception:
            return None, []
        if metabolites is None or profiles is None:
            return None, []
        profile, names = cls._reduce_metabolite_profiles(
            profiles,
            metabolites,
            n_nodes=n_nodes,
            keep_indices=keep_indices,
            row_index=row_index,
            subject_count=subject_count,
            preserve_subjects=preserve_subjects,
        )
        water_profile = cls._reduce_water_signal_profile(
            water_signal,
            n_nodes=n_nodes,
            keep_indices=keep_indices,
            row_index=row_index,
            subject_count=subject_count,
            preserve_subjects=preserve_subjects,
        )
        return cls._append_water_signal_profile(profile, names, water_profile)

    def _selected_gradient_metabolite_profiles_path(self):
        dialog = getattr(self, "_gradients_dialog", None)
        if dialog is None or not hasattr(dialog, "metabolite_profiles_path"):
            return None
        try:
            text = str(dialog.metabolite_profiles_path() or "").strip()
        except Exception:
            text = ""
        return Path(text).expanduser() if text else None

    @classmethod
    def _metabolite_subject_labels_from_npz(cls, path: Path):
        try:
            with np.load(Path(path), allow_pickle=True) as npz:
                subjects, sessions = cls._npz_subject_session_vectors(npz, path=path)
        except Exception:
            return []
        subjects = list(subjects or [])
        sessions = list(sessions or [])
        if not subjects:
            return []
        labels = []
        for idx, subject in enumerate(subjects):
            subject_text = str(subject).strip()
            session_text = str(sessions[idx]).strip() if idx < len(sessions) else ""
            labels.append(f"{subject_text} {session_text}".strip() or f"Subject {idx + 1}")
        return labels

    @classmethod
    def _load_metabolite_profiles_for_gradient_context(
        cls,
        path: Path,
        *,
        target_labels,
        target_names=None,
        row_index=None,
        preserve_subjects=True,
    ):
        try:
            with np.load(Path(path), allow_pickle=True) as npz:
                metabolites = cls._npz_optional_array(npz, "metabolites", "metabolites.npy")
                profiles = cls._npz_optional_array(
                    npz,
                    "metab_profiles_subj_list",
                    "metab_profiles_subj_list.npy",
                    "metab_profiles",
                    "metab_profiles.npy",
                    "node_signals_subj_list",
                    "node_signals_subj_list.npy",
                    "parcel_concentrations",
                    "parcel_concentrations.npy",
                )
                water_signal = cls._npz_optional_array(
                    npz,
                    "water_signal_subj_list",
                    "water_signal_subj_list.npy",
                    "water_signal",
                    "water_signal.npy",
                )
                source_labels = cls._npz_optional_array(
                    npz,
                    *PARCEL_LABEL_KEYS,
                    "parcel_labels",
                    "parcel_labels.npy",
                    "labels_indices",
                    "labels_indices.npy",
                )
                source_names = cls._npz_optional_display_vector(
                    npz,
                    *PARCEL_NAME_KEYS,
                    "parcel_names",
                    "parcel_names.npy",
                    "labels",
                    "labels.npy",
                )
                subject_values, _session_values = cls._npz_subject_session_vectors(npz, path=path)
                subject_count = len(subject_values or [])
                if subject_count <= 0:
                    matrix_stack = cls._npz_optional_array(npz, "matrix_subj_list", "matrix_subj_list.npy")
                    if matrix_stack is not None and np.asarray(matrix_stack).ndim >= 3:
                        subject_count = int(np.asarray(matrix_stack).shape[0])
        except Exception:
            return None, []

        names = cls._metabolite_names_from_array(metabolites)
        if profiles is None or not names:
            return None, []

        target_labels_arr = np.asarray(target_labels, dtype=int).reshape(-1)
        if target_labels_arr.size == 0:
            return None, []
        target_name_list = []
        if target_names is not None:
            try:
                target_name_list = [
                    str(value).strip()
                    for value in np.asarray(target_names, dtype=object).reshape(-1).tolist()
                ]
            except Exception:
                target_name_list = []

        keep_indices = None
        source_node_count = None
        if source_labels is not None:
            try:
                src_labels = np.asarray(source_labels, dtype=int).reshape(-1)
            except Exception:
                src_labels = np.zeros(0, dtype=int)
            if src_labels.size:
                label_to_index = {}
                for idx, label in enumerate(src_labels.tolist()):
                    label_to_index.setdefault(int(label), int(idx))
                mapped = [label_to_index.get(int(label), -1) for label in target_labels_arr.tolist()]
                if all(idx >= 0 for idx in mapped):
                    keep_indices = np.asarray(mapped, dtype=int)
                    source_node_count = int(src_labels.size)

        if keep_indices is None and source_names is not None and target_name_list:
            source_name_list = [str(value).strip() for value in list(source_names)]
            if source_name_list:
                name_to_index = {}
                for idx, name in enumerate(source_name_list):
                    normalized = str(name).strip().lower()
                    if normalized:
                        name_to_index.setdefault(normalized, int(idx))
                mapped = [
                    name_to_index.get(str(name).strip().lower(), -1)
                    for name in target_name_list
                ]
                if len(mapped) == target_labels_arr.size and all(idx >= 0 for idx in mapped):
                    keep_indices = np.asarray(mapped, dtype=int)
                    source_node_count = int(len(source_name_list))

        if source_node_count is None:
            source_node_count = int(target_labels_arr.size)
            keep_indices = None

        profile, names = cls._reduce_metabolite_profiles(
            profiles,
            metabolites,
            n_nodes=source_node_count,
            keep_indices=keep_indices,
            row_index=row_index,
            subject_count=subject_count,
            preserve_subjects=preserve_subjects,
        )
        if profile is None:
            return None, []
        profile = np.asarray(profile, dtype=float)
        water_profile = cls._reduce_water_signal_profile(
            water_signal,
            n_nodes=source_node_count,
            keep_indices=keep_indices,
            row_index=row_index,
            subject_count=subject_count,
            preserve_subjects=preserve_subjects,
        )
        profile, names = cls._append_water_signal_profile(profile, names, water_profile)
        profile = np.asarray(profile, dtype=float)
        node_axis = 1 if profile.ndim == 3 else 0
        if profile.shape[node_axis] != target_labels_arr.size:
            return None, []
        return profile, names

    @staticmethod
    def _canonicalize_precomputed_gradients(raw_gradients, *, row_count=None, label_count=None):
        gradients = np.asarray(raw_gradients, dtype=float)
        if gradients.ndim != 3:
            raise ValueError("Precomputed gradients must be a 3D array.")

        sample_axis = 0
        if row_count is not None:
            try:
                target_rows = int(row_count)
            except Exception:
                target_rows = 0
            if target_rows > 0:
                candidates = [idx for idx, size in enumerate(gradients.shape) if int(size) == target_rows]
                if candidates:
                    sample_axis = 0 if 0 in candidates else candidates[0]
        gradients = np.moveaxis(gradients, sample_axis, 0)

        if label_count is not None:
            try:
                target_labels = int(label_count)
            except Exception:
                target_labels = 0
            if target_labels > 0:
                if gradients.shape[2] == target_labels:
                    pass
                elif gradients.shape[1] == target_labels:
                    gradients = np.swapaxes(gradients, 1, 2)
                else:
                    raise ValueError(
                        f"Could not align gradients shape {tuple(gradients.shape)} to {target_labels} parcel labels."
                    )
        elif gradients.shape[1] > gradients.shape[2]:
            gradients = np.swapaxes(gradients, 1, 2)

        return np.asarray(gradients, dtype=float)

    @staticmethod
    def _canonicalize_average_gradients(raw_gradients_avg, *, label_count=None):
        gradients_avg = np.asarray(raw_gradients_avg, dtype=float)
        if gradients_avg.ndim != 2:
            raise ValueError("gradients_avg must be a 2D array.")
        if label_count is not None:
            try:
                target_labels = int(label_count)
            except Exception:
                target_labels = 0
            if target_labels > 0:
                if gradients_avg.shape[1] == target_labels:
                    return np.asarray(gradients_avg, dtype=float)
                if gradients_avg.shape[0] == target_labels:
                    return np.asarray(gradients_avg.T, dtype=float)
                raise ValueError(
                    f"Could not align gradients_avg shape {tuple(gradients_avg.shape)} to {target_labels} parcel labels."
                )
        if gradients_avg.shape[1] >= gradients_avg.shape[0]:
            return np.asarray(gradients_avg, dtype=float)
        return np.asarray(gradients_avg.T, dtype=float)

    @staticmethod
    def _path_exists_safely(path_value) -> bool:
        try:
            return Path(path_value).expanduser().exists()
        except OSError:
            return False

    @staticmethod
    def _resolve_existing_path(path_value):
        path = Path(path_value).expanduser()
        try:
            return path.resolve()
        except OSError:
            return path

    def _portable_parcellation_path_candidates(self, path_value):
        path = Path(path_value).expanduser()
        yielded = set()

        def _yield(candidate):
            candidate = Path(candidate).expanduser()
            key = str(candidate)
            if key in yielded:
                return
            yielded.add(key)
            yield candidate

        yield from _yield(path)

        parts = path.parts
        if "mrsi_viewer" in parts:
            idx = parts.index("mrsi_viewer")
            suffix_parts = parts[idx + 1 :]
            repo_root = Path(__file__).resolve().parents[1]
            candidate = repo_root.joinpath(*suffix_parts) if suffix_parts else repo_root
            yield from _yield(candidate)

        if len(parts) > 3 and parts[0] == "/" and parts[1] == "home":
            candidate = Path.home().joinpath(*parts[3:])
            yield from _yield(candidate)

    def _resolve_portable_parcellation_path(self, path_value):
        path = Path(path_value).expanduser()
        for candidate in self._portable_parcellation_path_candidates(path):
            if self._path_exists_safely(candidate):
                return self._resolve_existing_path(candidate)
        return path

    def _load_precomputed_gradient_bundle(self, path: Path):
        path = Path(path)
        try:
            with np.load(path, allow_pickle=True) as npz:
                if "gradients" not in npz:
                    return None

                raw_gradients = np.asarray(npz["gradients"], dtype=float)

                parcel_labels_raw = None
                parcel_names_raw = None
                for key in PARCEL_LABEL_KEYS:
                    if key in npz:
                        parcel_labels_raw = np.asarray(npz[key]).reshape(-1)
                        break
                for key in PARCEL_NAME_KEYS:
                    if key in npz:
                        parcel_names_raw = np.asarray(npz[key]).reshape(-1)
                        break

                label_count = None
                parcel_labels = None
                if parcel_labels_raw is not None:
                    coerced = _coerce_label_indices(parcel_labels_raw, parcel_labels_raw.size)
                    if coerced is None:
                        raise ValueError("parcel_labels_group is present but invalid.")
                    parcel_labels = np.asarray(coerced, dtype=int)
                    label_count = parcel_labels.size

                gradients_avg = None
                if "gradients_avg" in npz:
                    try:
                        gradients_avg = self._canonicalize_average_gradients(
                            npz["gradients_avg"],
                            label_count=label_count,
                        )
                    except Exception:
                        gradients_avg = None

                subject_values = self._npz_optional_display_vector(
                    npz,
                    "subject_id_list",
                    "participant_id_list",
                    "subject_ids",
                    "participant_ids",
                )
                session_values = self._npz_optional_display_vector(
                    npz,
                    "session_id_list",
                    "session_ids",
                )

                candidate_row_count = 0
                if subject_values:
                    candidate_row_count = len(subject_values)
                elif session_values:
                    candidate_row_count = len(session_values)

                covars_info = _load_covars_info(path)
                covars_columns, covars_rows = _covars_to_rows(covars_info)
                if covars_rows:
                    candidate_row_count = len(covars_rows)

                canonical_gradients = self._canonicalize_precomputed_gradients(
                    raw_gradients,
                    row_count=candidate_row_count or None,
                    label_count=label_count,
                )

                n_rows = int(canonical_gradients.shape[0])
                n_components = int(canonical_gradients.shape[1])
                n_labels = int(canonical_gradients.shape[2])

                if parcel_labels is None:
                    parcel_labels = np.arange(1, n_labels + 1, dtype=int)
                elif parcel_labels.size != n_labels:
                    raise ValueError(
                        f"parcel_labels_group has {parcel_labels.size} labels but gradients expect {n_labels} parcels."
                    )
                if np.unique(parcel_labels).size != parcel_labels.size:
                    raise ValueError("parcel_labels_group must be unique for precomputed gradients.")

                parcel_names = _to_string_list(parcel_names_raw) if parcel_names_raw is not None else None
                if parcel_names is None or len(parcel_names) != n_labels:
                    parcel_names = [f"Parcel {int(label)}" for label in parcel_labels.tolist()]

                row_dicts = []
                for row in list(covars_rows or []):
                    if isinstance(row, dict):
                        row_dicts.append(dict(row))
                    else:
                        row_dicts.append(
                            {
                                str(column): _display_text(value)
                                for column, value in zip(list(covars_columns or []), list(row))
                            }
                        )
                if row_dicts and len(row_dicts) != n_rows:
                    row_dicts = []
                    covars_columns = []
                if not row_dicts:
                    row_dicts = [{} for _ in range(n_rows)]
                    covars_columns = []

                dialog_rows = []
                for row_index, row_data in enumerate(row_dicts):
                    payload_row = {"__row_index__": int(row_index)}
                    payload_row.update(
                        {str(key): _display_text(value) for key, value in dict(row_data).items()}
                    )
                    dialog_rows.append(payload_row)

                def _merge_vector_column(column_name: str, values, *, overwrite: bool = False) -> None:
                    if values is None or len(values) != n_rows:
                        return
                    if column_name not in covars_columns:
                        covars_columns.append(column_name)
                    for row_idx, raw_value in enumerate(values):
                        text = _display_text(raw_value).strip()
                        if overwrite or not str(row_dicts[row_idx].get(column_name, "")).strip():
                            row_dicts[row_idx][column_name] = text

                _merge_vector_column("participant_id", subject_values)
                _merge_vector_column("session_id", session_values)
                _merge_vector_column(
                    "group",
                    self._npz_optional_display_vector(npz, "group"),
                )
                _merge_vector_column(
                    "modality",
                    self._npz_optional_display_vector(npz, "modality"),
                )
                _merge_vector_column(
                    "metabolites",
                    self._npz_optional_display_vector(npz, "metabolites"),
                )

                metabolites = self._npz_optional_array(npz, "metabolites", "metabolites.npy")
                metab_profiles = self._npz_optional_array(
                    npz,
                    "metab_profiles_subj_list",
                    "metab_profiles_subj_list.npy",
                )
                water_signal = self._npz_optional_array(
                    npz,
                    "water_signal_subj_list",
                    "water_signal_subj_list.npy",
                    "water_signal",
                    "water_signal.npy",
                )

                parcellation_path = None
                parcellation_text = self._npz_optional_scalar_text(npz, "parc_path", "parc_path.npy")
                if parcellation_text:
                    candidate = Path(parcellation_text).expanduser()
                    if not candidate.is_absolute():
                        candidate = (path.parent / candidate).resolve()
                    parcellation_path = self._resolve_portable_parcellation_path(candidate)
        except Exception:
            raise

        summary = f"{path.name} | rows: {n_rows} | components: {n_components} | parcels: {n_labels}"
        return {
            "path": path,
            "label": path.name,
            "summary": summary,
            "gradients": canonical_gradients,
            "component_count_total": n_components,
            "n_rows": n_rows,
            "n_labels": n_labels,
            "gradients_avg": None if gradients_avg is None else np.asarray(gradients_avg, dtype=float),
            "parcel_labels": np.asarray(parcel_labels, dtype=int),
            "parcel_names": list(parcel_names),
            "covars_columns": [str(column) for column in covars_columns],
            "covars_rows": row_dicts,
            "dialog_covars_rows": dialog_rows,
            "parcellation_path": parcellation_path,
            "parcellation_path_original": str(parcellation_text or ""),
            "metabolites": None if metabolites is None else np.asarray(metabolites),
            "metab_profiles_subj_list": None if metab_profiles is None else np.asarray(metab_profiles),
            "water_signal": None if water_signal is None else np.asarray(water_signal),
        }

    @staticmethod
    def _gradient_precomputed_row_pair(bundle, row_index: int):
        if not isinstance(bundle, dict):
            return "", ""
        rows = list(bundle.get("covars_rows") or [])
        if row_index < 0 or row_index >= len(rows):
            return "", ""
        row = rows[row_index]
        lower_map = {str(key).lower(): key for key in row.keys()}
        participant_key = (
            lower_map.get("participant_id")
            or lower_map.get("subject_id")
            or lower_map.get("participant")
            or lower_map.get("subject")
        )
        session_key = lower_map.get("session_id") or lower_map.get("session") or lower_map.get("ses")
        participant = _display_text(row.get(participant_key, "")).strip() if participant_key else ""
        session = _display_text(row.get(session_key, "")).strip() if session_key else ""
        return participant, session

    def _gradient_precomputed_selection_text(self) -> str:
        bundle = self._gradient_precomputed_bundle
        if not isinstance(bundle, dict):
            return "Selected pair: none"
        row_index = self._gradient_precomputed_selected_row
        if row_index is None:
            return "Selected pair: none"
        participant, session = self._gradient_precomputed_row_pair(bundle, int(row_index))
        parts = [f"row {int(row_index)}"]
        if participant:
            parts.append(participant)
        if session:
            parts.append(session)
        return "Selected pair: " + " | ".join(parts)

    def _activate_precomputed_gradient_bundle(self, bundle) -> None:
        if not isinstance(bundle, dict):
            return
        self._gradient_precomputed_bundle = bundle
        self._gradient_precomputed_selected_row = None
        self._gradient_use_precomputed_bundle = True
        try:
            self._gradient_component_count = max(
                1,
                min(10, int(bundle.get("component_count_total", 2))),
            )
        except Exception:
            self._gradient_component_count = 2
        self._gradient_surface_render_count = 1
        self._gradient_surface_procrustes = False

        parcellation_path = bundle.get("parcellation_path")
        did_reset = False
        parcellation_exists = isinstance(parcellation_path, Path) and self._path_exists_safely(parcellation_path)
        if parcellation_exists:
            active_path = Path(self._active_parcellation_path) if self._active_parcellation_path is not None else None
            if active_path != parcellation_path:
                did_reset = bool(self._set_active_parcellation(parcellation_path))
        if not did_reset:
            self._reset_gradients_output()

        self._update_gradients_button()
        self._open_gradients_dialog(prefer_precomputed=True)
        if getattr(self, "_gradients_dialog", None) is not None and hasattr(self._gradients_dialog, "focus_precomputed_tab"):
            try:
                self._gradients_dialog.focus_precomputed_tab()
            except Exception:
                pass

        status = f"Loaded precomputed gradients from {bundle.get('label', 'bundle')}. Select a participant/session row."
        original_path_text = str(bundle.get("parcellation_path_original") or "").strip()
        if did_reset and original_path_text and str(parcellation_path) != original_path_text:
            status += " Adapted bundle parcellation path to this checkout."
        if isinstance(parcellation_path, Path) and not parcellation_exists:
            status += " Bundle parcellation path was not found; set it manually before confirming."
        self.statusBar().showMessage(status)

    def _confirm_precomputed_gradient_row(self, row_index: int) -> None:
        bundle = self._gradient_precomputed_bundle
        if not isinstance(bundle, dict):
            self.statusBar().showMessage("No precomputed gradients bundle is loaded.")
            return

        try:
            row_index = int(row_index)
        except Exception:
            self.statusBar().showMessage("Invalid precomputed gradient row.")
            return

        gradients_stack = np.asarray(bundle.get("gradients"), dtype=float)
        if gradients_stack.ndim != 3:
            self.statusBar().showMessage("Precomputed gradients array is invalid.")
            return
        if row_index < 0 or row_index >= gradients_stack.shape[0]:
            self.statusBar().showMessage("Selected participant/session row is out of range.")
            return

        parcel_labels = np.asarray(bundle.get("parcel_labels"), dtype=int).reshape(-1)
        parcel_names = list(bundle.get("parcel_names") or [])
        if len(parcel_names) != parcel_labels.size:
            parcel_names = [f"Parcel {int(label)}" for label in parcel_labels.tolist()]
        selected_components = np.asarray(gradients_stack[row_index], dtype=float)
        if selected_components.ndim != 2:
            self.statusBar().showMessage("Selected precomputed gradients row is not 2D.")
            return
        if selected_components.shape[1] == parcel_labels.size:
            pass
        elif selected_components.shape[0] == parcel_labels.size:
            selected_components = np.swapaxes(selected_components, 0, 1)
        else:
            self.statusBar().showMessage(
                "Selected precomputed gradients row does not align with the parcel label count."
            )
            return

        total_components = int(selected_components.shape[0])
        n_grad = max(1, min(10, total_components))
        components = np.asarray(selected_components[:n_grad, :], dtype=float).T
        if components.shape[0] != parcel_labels.size:
            self.statusBar().showMessage("Projected node count does not match the parcel label count.")
            return
        participant, session = self._gradient_precomputed_row_pair(bundle, row_index)
        display_bits = [bundle.get("label", "precomputed gradients")]
        if participant:
            display_bits.append(participant)
        if session:
            display_bits.append(session)
        source_name = " | ".join(display_bits)

        stem_bits = [bundle.get("path", Path("gradients")).stem]
        if participant:
            stem_bits.append(self._safe_name_fragment(_normalize_subject_token(participant)))
        if session:
            stem_bits.append(self._safe_name_fragment(_normalize_session_token(session)))
        default_name = "_".join(bit for bit in stem_bits if bit) + f"_diffusion_components-{n_grad}.nii.gz"

        template_path_text = ""
        bundle_parcellation = bundle.get("parcellation_path")
        if isinstance(bundle_parcellation, Path):
            template_path_text = str(bundle_parcellation)
        elif self._active_parcellation_path is not None:
            template_path_text = str(self._active_parcellation_path)

        custom_metabolite_path = self._selected_gradient_metabolite_profiles_path()
        if custom_metabolite_path is not None and custom_metabolite_path.exists():
            metabolite_profiles, metabolite_names = self._load_metabolite_profiles_for_gradient_context(
                custom_metabolite_path,
                target_labels=np.asarray(parcel_labels, dtype=int),
                target_names=parcel_names,
                row_index=row_index,
                preserve_subjects=False,
            )
            if metabolite_profiles is None:
                self.statusBar().showMessage(
                    f"Custom metabolite profiles did not align to the selected gradient row: {custom_metabolite_path}"
                )
        else:
            metabolite_profiles, metabolite_names = self._reduce_metabolite_profiles(
                bundle.get("metab_profiles_subj_list"),
                bundle.get("metabolites"),
                n_nodes=parcel_labels.size,
                row_index=row_index,
                subject_count=bundle.get("n_rows"),
            )
            water_profile = self._reduce_water_signal_profile(
                bundle.get("water_signal"),
                n_nodes=parcel_labels.size,
                row_index=row_index,
                subject_count=bundle.get("n_rows"),
            )
            metabolite_profiles, metabolite_names = self._append_water_signal_profile(
                metabolite_profiles,
                metabolite_names,
                water_profile,
            )

        self._gradient_component_count = n_grad
        self._gradient_surface_render_count = 1
        self._gradient_precomputed_selected_row = row_index
        self._last_gradients = {
            "gradients": np.asarray(components, dtype=float),
            "n_grad": n_grad,
            "n_nodes": parcel_labels.size,
            "projected_data": None,
            "affine": None,
            "header": None,
            "source_name": source_name,
            "source_dir": str(bundle["path"].parent),
            "output_name": default_name,
            "keep_indices": np.arange(parcel_labels.size, dtype=int),
            "projection_labels": np.asarray(parcel_labels, dtype=int),
            "support_mask": None,
            "template_path": template_path_text,
            "parcel_names": parcel_names,
            "matrix_entry_id": None,
            "matrix_label": bundle.get("label", ""),
            "precomputed_source_path": str(bundle["path"]),
            "precomputed_row_index": int(row_index),
            "metabolite_profiles": metabolite_profiles,
            "metabolite_names": metabolite_names,
        }
        self._set_gradients_progress(0, n_grad, n_grad, self._gradient_precomputed_selection_text())
        self._sync_gradients_dialog_state()
        self.statusBar().showMessage(
            f"Loaded precomputed gradients for row {row_index}. No diffusion embedding or fsaverage projection was run; projection will happen only when needed."
        )

    def _classification_scatter_edge_pairs(self, projection_labels, finite_mask):
        adjacency_data = self._gradient_classification_adjacency_data()
        if not adjacency_data:
            return np.zeros((0, 2), dtype=int), None

        labels = np.asarray(projection_labels, dtype=int).reshape(-1)
        mask = np.asarray(finite_mask, dtype=bool).reshape(-1)
        if labels.shape != mask.shape:
            raise RuntimeError("Classification labels do not align with the scatter mask.")
        labels = labels[mask]
        if labels.size < 2:
            return np.zeros((0, 2), dtype=int), None

        adjacency = np.asarray(adjacency_data["adjacency"], dtype=float)
        adjacency_labels = np.asarray(adjacency_data["parcel_labels"], dtype=int).reshape(-1)
        label_to_index = {int(label): idx for idx, label in enumerate(adjacency_labels.tolist())}

        mapped_positions = []
        mapped_indices = []
        missing_count = 0
        for scatter_index, label in enumerate(labels.tolist()):
            adjacency_index = label_to_index.get(int(label))
            if adjacency_index is None:
                missing_count += 1
                continue
            mapped_positions.append(int(scatter_index))
            mapped_indices.append(int(adjacency_index))

        if len(mapped_indices) < 2:
            note = (
                "Adjacency file does not overlap with the current classification labels."
                if missing_count
                else "Adjacency file does not contain enough nodes for edge rendering."
            )
            return np.zeros((0, 2), dtype=int), note

        mapped_positions = np.asarray(mapped_positions, dtype=int)
        mapped_indices = np.asarray(mapped_indices, dtype=int)
        edge_matrix = np.asarray(adjacency[np.ix_(mapped_indices, mapped_indices)], dtype=float)
        edge_matrix = np.nan_to_num(edge_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        edge_matrix = np.maximum(np.abs(edge_matrix), np.abs(edge_matrix.T))
        upper_i, upper_j = np.triu_indices(edge_matrix.shape[0], k=1)
        keep = edge_matrix[upper_i, upper_j] > 0.0
        if not np.any(keep):
            note = "Adjacency file loaded, but no non-zero edges matched the displayed nodes."
            return np.zeros((0, 2), dtype=int), note

        edge_pairs = np.column_stack(
            (
                mapped_positions[upper_i[keep]],
                mapped_positions[upper_j[keep]],
            )
        ).astype(int, copy=False)
        note = f"Adjacency edges: {edge_pairs.shape[0]}"
        if missing_count:
            note += f" ({missing_count} labels unmatched)"
        return edge_pairs, note

    @staticmethod
    def _gradient_entry_source_path(entry):
        return _MatrixDataAccess.entry_source_path(entry)

    def _entry_parcel_metadata(self, entry, expected_len=None, label_key: str = "", name_key: str = ""):
        try:
            return self._data_access.entry_parcel_metadata(
                entry,
                expected_len=expected_len,
                label_key=label_key,
                name_key=name_key,
            )
        except TypeError:
            return self._data_access.entry_parcel_metadata(entry, expected_len=expected_len)

    @staticmethod
    def _preferred_parcel_key(options, preferred_keys):
        keys = [str(option.get("key", "")).strip() for option in list(options or []) if isinstance(option, dict)]
        keys = [key for key in keys if key]
        if not keys:
            return ""
        key_set = set(keys)
        for preferred in preferred_keys:
            text = str(preferred or "").strip()
            if text in key_set:
                return text
        lower_to_key = {key.lower(): key for key in keys}
        for preferred in preferred_keys:
            text = str(preferred or "").strip().lower()
            if text in lower_to_key:
                return lower_to_key[text]
        return ""

    @staticmethod
    def _parcel_key_option_label(option) -> str:
        if not isinstance(option, dict):
            return str(option or "").strip()
        key = str(option.get("key", "")).strip()
        if not key:
            return ""
        shape = option.get("shape")
        dtype = str(option.get("dtype", "")).strip()
        if shape:
            shape_text = "x".join(str(dim) for dim in tuple(shape))
            suffix = f" [{shape_text}"
            if dtype:
                suffix += f", {dtype}"
            suffix += "]"
            return f"{key}{suffix}"
        return key

    def _available_gradient_parcel_key_options(self):
        entry = self._selected_gradient_entry()
        source_path = self._gradient_entry_source_path(entry)
        if source_path is None or not source_path.exists():
            return {
                "labels": [],
                "names": [],
                "auto_label_key": "",
                "auto_name_key": "",
            }
        expected_len = None
        try:
            matrix = np.asarray(self._gradient_matrix_for_entry(entry))
            if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
                expected_len = int(matrix.shape[0])
        except Exception:
            expected_len = None
        vector_options = self._data_access.npz_vector_keys(source_path, expected_len=expected_len)
        formatted_options = []
        for option in vector_options:
            key = str(option.get("key", "")).strip()
            label = self._parcel_key_option_label(option)
            if key and label:
                item = dict(option)
                item["label"] = label
                formatted_options.append(item)
        return {
            "labels": formatted_options,
            "names": formatted_options,
            "auto_label_key": self._preferred_parcel_key(formatted_options, PARCEL_LABEL_KEYS),
            "auto_name_key": self._preferred_parcel_key(formatted_options, PARCEL_NAME_KEYS),
        }

    def _selected_gradient_parcel_label_key(self) -> str:
        dialog = getattr(self, "_gradients_dialog", None)
        if dialog is not None and hasattr(dialog, "selected_parcel_label_key"):
            try:
                self._gradient_parcel_label_key = str(dialog.selected_parcel_label_key() or "").strip()
            except Exception:
                pass
        self._gradient_parcel_label_key = str(
            getattr(self, "_gradient_parcel_label_key", "") or ""
        ).strip()
        return self._gradient_parcel_label_key

    def _selected_gradient_parcel_name_key(self) -> str:
        dialog = getattr(self, "_gradients_dialog", None)
        if dialog is not None and hasattr(dialog, "selected_parcel_name_key"):
            try:
                self._gradient_parcel_name_key = str(dialog.selected_parcel_name_key() or "").strip()
            except Exception:
                pass
        self._gradient_parcel_name_key = str(
            getattr(self, "_gradient_parcel_name_key", "") or ""
        ).strip()
        return self._gradient_parcel_name_key

    def _gradient_matrix_label_for_entry(self, entry) -> str:
        if entry is None:
            return ""
        label = str(entry.get("label") or "").strip()
        key = str(entry.get("selected_key") or "").strip()
        source_path = self._gradient_entry_source_path(entry)
        if not label and source_path is not None:
            label = source_path.name
        if key and key not in label:
            label = f"{label} [{key}]" if label else key
        return label or "matrix"

    def _available_gradient_matrix_entries(self):
        options = []
        for entry_id in self._entry_ids():
            entry = self._entries.get(entry_id)
            if entry is None:
                continue
            options.append(
                {
                    "id": entry_id,
                    "label": self._gradient_matrix_label_for_entry(entry),
                }
            )
        return options

    def _available_gradient_reference_matrix_entries(self):
        current_gradient_entry_id = self._selected_gradient_entry_id()
        options = []
        for option in self._available_gradient_matrix_entries():
            entry_id = option.get("id")
            if (
                entry_id == current_gradient_entry_id
                and not bool(getattr(self, "_gradient_use_precomputed_bundle", False))
            ):
                continue
            options.append(option)
        return options

    def _available_combine_matrix_entries(self):
        return self._available_gradient_matrix_entries()

    def _sync_combine_dialog_state(self) -> None:
        dialog = getattr(self, "_combine_dialog", None)
        if dialog is None:
            return
        dialog.set_matrix_options(self._available_combine_matrix_entries())

    def _update_combine_button(self) -> None:
        enabled = bool(self._available_combine_matrix_entries())
        if hasattr(self, "combine_open_button"):
            self.combine_open_button.setEnabled(enabled)
        self._sync_combine_dialog_state()

    def _selected_gradient_entry_id(self):
        stored = self._gradient_selected_entry_id
        if stored in self._entries:
            return stored
        current = self._current_entry_id()
        if current in self._entries:
            return current
        for option in self._available_gradient_matrix_entries():
            entry_id = option.get("id")
            if entry_id in self._entries:
                return entry_id
        return None

    def _selected_gradient_entry(self):
        entry_id = self._selected_gradient_entry_id()
        if entry_id is None:
            return None
        return self._entries.get(entry_id)

    def _selected_gradient_classification_ref_entry(self):
        entry_id = self._selected_gradient_classification_ref_entry_id()
        if entry_id is None:
            return None
        return self._entries.get(entry_id)

    def _gradient_matrix_for_entry(self, entry):
        if entry is None:
            return None
        if entry is self._current_entry() and self._current_matrix is not None:
            return np.asarray(self._current_matrix)
        matrix, _selected_key = self._matrix_for_entry(entry)
        return np.asarray(matrix)

    def _has_square_matrix_entry(self, entry) -> bool:
        if entry is None:
            return False
        try:
            matrix = self._gradient_matrix_for_entry(entry)
        except Exception:
            return False
        return matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]

    def _selected_gradient_matrix_label(self) -> str:
        return self._gradient_matrix_label_for_entry(self._selected_gradient_entry())

    def _on_gradient_matrix_entry_changed(self, entry_id) -> None:
        normalized_id = entry_id if entry_id in self._entries else None
        if normalized_id == self._gradient_selected_entry_id:
            return
        self._gradient_selected_entry_id = normalized_id
        self._reset_gradients_output()

    def _on_gradient_parcel_label_key_changed(self, value: str) -> None:
        text = str(value or "").strip()
        if text == str(getattr(self, "_gradient_parcel_label_key", "") or "").strip():
            return
        self._gradient_parcel_label_key = text
        self._reset_gradients_output()

    def _on_gradient_parcel_name_key_changed(self, value: str) -> None:
        text = str(value or "").strip()
        if text == str(getattr(self, "_gradient_parcel_name_key", "") or "").strip():
            return
        self._gradient_parcel_name_key = text
        self._reset_gradients_output()

    def _on_gradient_network_component_changed(self, value: str) -> None:
        self._gradient_network_component = self._normalize_gradient_network_component(
            value,
            max_components=self._available_gradient_network_component_count(),
        )

    def _on_gradient_rotation_changed(self, index: int, value: str) -> None:
        try:
            idx = int(index)
        except Exception:
            return
        if idx < 0 or idx >= 10:
            return
        self._ensure_gradient_rotation_count(idx + 1)
        self._gradient_component_rotations[idx] = self._normalize_gradient_rotation_preset(value)

    def _sync_gradients_dialog_state(self) -> None:
        dialog = getattr(self, "_gradients_dialog", None)
        if dialog is None:
            return
        matrix_options = self._available_gradient_matrix_entries()
        selected_entry_id = self._selected_gradient_entry_id()
        if self._gradient_selected_entry_id not in self._entries:
            self._gradient_selected_entry_id = selected_entry_id
        selected_entry = self._selected_gradient_entry()
        names = self._available_colormap_names()
        current_cmap = str(self._gradient_colormap_name or "").strip()
        if current_cmap not in names:
            current_cmap = (
                self._default_gradient_colormap
                if self._default_gradient_colormap in names
                else (names[0] if names else "spectrum_fsl")
            )
            self._gradient_colormap_name = current_cmap
        precomputed_bundle = (
            self._gradient_precomputed_bundle
            if bool(self._gradient_use_precomputed_bundle) and isinstance(self._gradient_precomputed_bundle, dict)
            else None
        )
        if precomputed_bundle is not None:
            component_count = max(
                1,
                min(
                    10,
                    int(precomputed_bundle.get("component_count_total", self._current_gradient_component_count())),
                ),
            )
            self._gradient_component_count = component_count
            matrix_source = f"Precomputed gradients: {precomputed_bundle.get('label', 'bundle')}"
        else:
            component_count = self._current_gradient_component_count()
            matrix_source = self._gradient_matrix_label_for_entry(selected_entry)

        dialog.set_matrix_options(matrix_options, selected_entry_id=selected_entry_id)
        parcel_key_options = self._available_gradient_parcel_key_options()
        if hasattr(dialog, "set_parcel_key_options"):
            dialog.set_parcel_key_options(
                parcel_key_options.get("labels", []),
                parcel_key_options.get("names", []),
                selected_label_key=self._selected_gradient_parcel_label_key(),
                selected_name_key=self._selected_gradient_parcel_name_key(),
                auto_label_key=parcel_key_options.get("auto_label_key", ""),
                auto_name_key=parcel_key_options.get("auto_name_key", ""),
            )
        dialog.set_classification_reference_matrix_options(
            self._available_gradient_reference_matrix_entries(),
            selected_entry_id=self._selected_gradient_classification_ref_entry_id(),
        )
        dialog.set_matrix_source(matrix_source)
        dialog.set_component_count(component_count)
        dialog.set_surface_render_component_limit(self._available_gradient_network_component_count())
        dialog.set_surface_render_component_count(self._selected_gradient_surface_render_count())
        dialog.set_surface_procrustes_available(self._gradient_surface_procrustes_available())
        dialog.set_surface_procrustes_enabled(self._selected_gradient_surface_procrustes())
        dialog.set_colormap_names(names, current_colormap=current_cmap)
        dialog.set_parcellation_path(self._active_parcellation_path)
        dialog.set_hemisphere_mode(self._selected_gradient_hemisphere_mode())
        dialog.set_surface_mesh(self._selected_gradient_surface_mesh())
        dialog.set_classification_surface_mesh(self._selected_gradient_classification_surface_mesh())
        dialog.set_classification_hemisphere_mode(self._selected_gradient_classification_hemisphere_mode())
        dialog.set_scatter_rotation(self._selected_gradient_scatter_rotation())
        dialog.set_triangular_rgb(self._selected_gradient_triangular_rgb())
        dialog.set_preload_matching_paths(
            self._selected_gradient_preload_matching_paths()
        )
        dialog.set_classification_fit_mode(self._selected_gradient_classification_fit_mode())
        dialog.set_triangular_color_order(self._selected_gradient_triangular_color_order())
        if hasattr(dialog, "set_rgb_scalar_mode"):
            dialog.set_rgb_scalar_mode(self._selected_gradient_rgb_scalar_mode())
        dialog.set_rgb_mask_strictness(self._selected_gradient_rgb_mask_strictness())
        dialog.set_classification_colormap(self._selected_gradient_classification_colormap())
        dialog.set_classification_component_options(
            self._available_gradient_network_component_count(),
            selected_component=self._selected_gradient_classification_component(),
        )
        dialog.set_classification_axes(
            self._selected_gradient_classification_x_axis(),
            self._selected_gradient_classification_y_axis(),
            self._selected_gradient_classification_z_axis(),
        )
        ignore_options = self._gradient_classification_ignore_parcel_options()
        dialog.set_classification_ignore_parcel_options(
            ignore_options.get("lh", []),
            ignore_options.get("rh", []),
            selected_lh=self._selected_gradient_classification_ignore_lh_parcel(),
            selected_rh=self._selected_gradient_classification_ignore_rh_parcel(),
        )
        dialog.set_classification_adjacency_path(self._gradient_classification_adjacency_path)
        dialog.set_network_component_options(
            self._available_gradient_network_component_count(),
            selected_component=self._selected_gradient_network_component(),
        )
        dialog.set_rotation_presets(self._current_gradient_rotation_presets())
        progress = self._gradients_progress_state
        dialog.set_progress(
            progress["minimum"],
            progress["maximum"],
            progress["value"],
            progress["text"],
        )
        dialog.set_precomputed_mode(precomputed_bundle is not None)
        if precomputed_bundle is not None:
            dialog.set_precomputed_rows(
                precomputed_bundle.get("covars_columns") or [],
                precomputed_bundle.get("dialog_covars_rows") or [],
                selected_row=self._gradient_precomputed_selected_row,
                summary_text=precomputed_bundle.get("summary", ""),
                selection_text=self._gradient_precomputed_selection_text(),
            )
        else:
            dialog.set_precomputed_rows([], [], selected_row=None, summary_text="", selection_text="")
        dialog.set_can_compute(precomputed_bundle is None and self._has_square_matrix_entry(selected_entry))
        dialog.set_busy(self._gradients_busy)
        dialog.set_has_results(bool(self._last_gradients))
        dialog.set_can_classify(self._can_classify_gradients())

    def _open_gradients_dialog(self, *_args, prefer_precomputed=None) -> None:
        has_matrix_entries = bool(self._available_gradient_matrix_entries())
        has_precomputed_bundle = isinstance(self._gradient_precomputed_bundle, dict)
        if bool(prefer_precomputed) and has_precomputed_bundle:
            self._set_gradient_source_mode(True, reset_results=False)
        elif has_matrix_entries:
            self._set_gradient_source_mode(False, reset_results=False)
        elif has_precomputed_bundle:
            self._set_gradient_source_mode(True, reset_results=False)
        else:
            self._set_gradient_source_mode(False, reset_results=False)
        if getattr(self, "_gradients_dialog", None) is None:
            try:
                from window.gradients_prepare import GradientsPrepareDialog
            except Exception:
                try:
                    from mrsi_viewer.window.gradients_prepare import GradientsPrepareDialog
                except Exception as exc:
                    self.statusBar().showMessage(f"Failed to open Gradients window: {exc}")
                    return

            self._gradients_dialog = GradientsPrepareDialog(
                theme_name=self._theme_name,
                component_count=self._gradient_component_count,
                colormap_names=self._available_colormap_names(),
                current_colormap=self._gradient_colormap_name,
                parcellation_path=self._active_parcellation_path,
                open_parcellation_callback=self._select_parcellation_template,
                compute_callback=self._compute_gradients,
                save_callback=self._save_gradients_projection,
                render_3d_callback=self._render_gradients_3d,
                classify_callback=self._classify_gradients_fsaverage,
                render_network_callback=self._render_gradients_network,
                matrix_changed_callback=self._on_gradient_matrix_entry_changed,
                parcel_label_key_changed_callback=self._on_gradient_parcel_label_key_changed,
                parcel_name_key_changed_callback=self._on_gradient_parcel_name_key_changed,
                component_changed_callback=self._on_gradient_component_changed,
                colormap_changed_callback=self._on_gradient_colormap_changed,
                hemisphere_changed_callback=self._on_gradient_hemisphere_changed,
                surface_mesh_changed_callback=self._on_gradient_surface_mesh_changed,
                surface_render_count_changed_callback=self._on_gradient_surface_render_count_changed,
                surface_procrustes_changed_callback=self._on_gradient_surface_procrustes_changed,
                classification_surface_mesh_changed_callback=self._on_gradient_classification_surface_mesh_changed,
                classification_hemisphere_changed_callback=self._on_gradient_classification_hemisphere_changed,
                scatter_rotation_changed_callback=self._on_gradient_scatter_rotation_changed,
                triangular_rgb_changed_callback=self._on_gradient_triangular_rgb_changed,
                preload_matching_paths_changed_callback=(
                    self._on_gradient_preload_matching_paths_changed
                ),
                classification_fit_mode_changed_callback=self._on_gradient_classification_fit_mode_changed,
                triangular_color_order_changed_callback=self._on_gradient_triangular_color_order_changed,
                rgb_scalar_mode_changed_callback=self._on_gradient_rgb_scalar_mode_changed,
                rgb_mask_strictness_changed_callback=self._on_gradient_rgb_mask_strictness_changed,
                classification_colormap_changed_callback=self._on_gradient_classification_colormap_changed,
                classification_component_changed_callback=self._on_gradient_classification_component_changed,
                classification_x_axis_changed_callback=self._on_gradient_classification_x_axis_changed,
                classification_y_axis_changed_callback=self._on_gradient_classification_y_axis_changed,
                classification_z_axis_changed_callback=self._on_gradient_classification_z_axis_changed,
                classification_ref_matrix_changed_callback=self._on_gradient_classification_ref_entry_changed,
                classification_ignore_lh_changed_callback=self._on_gradient_classification_ignore_lh_changed,
                classification_ignore_rh_changed_callback=self._on_gradient_classification_ignore_rh_changed,
                open_classification_adjacency_callback=self._select_gradient_classification_adjacency,
                remove_classification_adjacency_callback=self._clear_gradient_classification_adjacency,
                precomputed_row_confirm_callback=self._confirm_precomputed_gradient_row,
                network_component_changed_callback=self._on_gradient_network_component_changed,
                rotation_changed_callback=self._on_gradient_rotation_changed,
                parent=self,
            )

        self._sync_gradients_dialog_state()
        self._gradients_dialog.show()
        try:
            self._gradients_dialog.raise_()
            self._gradients_dialog.activateWindow()
        except Exception:
            pass
        self.statusBar().showMessage("Opened Gradients window.")


    def _compute_gradients(self) -> None:
        self._reset_gradients_output()
        entry = self._selected_gradient_entry()
        if entry is None:
            self.statusBar().showMessage("No workspace matrix selected for gradients.")
            return
        matrix_label = self._gradient_matrix_label_for_entry(entry)
        try:
            conn_matrix = np.asarray(self._gradient_matrix_for_entry(entry), dtype=float)
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to load {matrix_label} for gradients: {exc}")
            return
        if conn_matrix.ndim != 2 or conn_matrix.shape[0] != conn_matrix.shape[1]:
            self.statusBar().showMessage(f"Gradients require a square matrix. {matrix_label} is not square.")
            return

        source_dir = self._default_results_dir()
        if self._active_parcellation_data is None:
            if not self._select_parcellation_template():
                self.statusBar().showMessage("Gradient compute canceled (no template selected).")
                return

        template_img = self._active_parcellation_img
        template_data = self._active_parcellation_data
        if template_img is None or template_data is None:
            self.statusBar().showMessage("No active parcellation template.")
            return

        label_key = self._selected_gradient_parcel_label_key()
        name_key = self._selected_gradient_parcel_name_key()
        parcel_labels, parcel_names = self._entry_parcel_metadata(
            entry,
            expected_len=conn_matrix.shape[0],
            label_key=label_key,
            name_key=name_key,
        )
        source_path = self._gradient_entry_source_path(entry)
        if (source_path is None or not source_path.exists()) and parcel_labels is None:
            self.statusBar().showMessage("Projection requires parcel labels for the selected matrix.")
            return
        label_indices = _coerce_label_indices(parcel_labels, conn_matrix.shape[0])
        if label_indices is None:
            if label_key:
                self.statusBar().showMessage(
                    f"Selected parcel label key '{label_key}' is missing/invalid or does not match matrix nodes."
                )
            else:
                self.statusBar().showMessage(
                    "Parcel labels missing/invalid or do not match matrix nodes. Select the label key in Gradients."
                )
            return
        template_labels = set(np.asarray(template_data, dtype=int).reshape(-1).tolist())
        template_labels.discard(0)
        if not template_labels:
            self.statusBar().showMessage("Template has no non-zero labels.")
            return
        keep_indices = [idx for idx, label in enumerate(label_indices) if label in template_labels]
        if not keep_indices:
            self.statusBar().showMessage(
                "No overlap between matrix parcel_labels_group and active parcellation labels."
            )
            return
        projection_labels = [label_indices[idx] for idx in keep_indices]

        kept_names = None
        if parcel_names and len(parcel_names) == conn_matrix.shape[0]:
            kept_names = [parcel_names[idx] for idx in keep_indices]

        n_grad = self._current_gradient_component_count()
        self._gradients_busy = True
        self._set_gradients_progress(0, n_grad, 0, f"0/{n_grad} components")
        self._sync_gradients_dialog_state()
        QApplication.processEvents()

        gradients = np.zeros((len(keep_indices), n_grad), dtype=float)
        try:
            for comp_idx in range(1, n_grad + 1):
                try:
                    component = nettools.dimreduce_matrix(
                        conn_matrix,
                        method="diffusion",
                        scale_factor=1.0,
                        output_dim=comp_idx,
                    )
                except Exception as exc:
                    self.statusBar().showMessage(f"Failed to compute component {comp_idx}: {exc}")
                    return
                component = np.asarray(component, dtype=float).reshape(-1)
                if component.size != conn_matrix.shape[0]:
                    self.statusBar().showMessage(
                        f"Component {comp_idx} size mismatch ({component.size} vs {conn_matrix.shape[0]})."
                    )
                    return
                gradients[:, comp_idx - 1] = np.asarray(component[keep_indices], dtype=float)
                self._set_gradients_progress(0, n_grad, comp_idx, f"{comp_idx}/{n_grad} components")
                QApplication.processEvents()
        finally:
            self._gradients_busy = False
            self._sync_gradients_dialog_state()

        support_mask = np.asarray(
            np.isin(template_data, np.asarray(projection_labels, dtype=int)),
            dtype=np.float32,
        )
        source_stem = source_path.stem if source_path is not None else "matrix"
        default_name = f"{source_stem}_diffusion_components-{n_grad}.nii.gz"
        metabolite_profiles = None
        metabolite_names = []
        custom_metabolite_path = self._selected_gradient_metabolite_profiles_path()
        if custom_metabolite_path is not None:
            if custom_metabolite_path.exists():
                metabolite_profiles, metabolite_names = self._load_metabolite_profiles_for_gradient_context(
                    custom_metabolite_path,
                    target_labels=np.asarray(projection_labels, dtype=int),
                    target_names=kept_names,
                    preserve_subjects=True,
                )
                if metabolite_profiles is None:
                    self.statusBar().showMessage(
                        f"Custom metabolite profiles did not align to the gradient parcels: {custom_metabolite_path}"
                    )
            else:
                self.statusBar().showMessage(f"Custom metabolite profile file does not exist: {custom_metabolite_path}")
        elif source_path is not None and source_path.exists():
            metabolite_profiles, metabolite_names = self._load_metabolite_profiles_from_npz(
                source_path,
                n_nodes=conn_matrix.shape[0],
                keep_indices=keep_indices,
                preserve_subjects=True,
            )

        self._last_gradients = {
            "gradients": np.asarray(gradients, dtype=float),
            "n_grad": n_grad,
            "n_nodes": conn_matrix.shape[0],
            "projected_data": None,
            "affine": np.asarray(template_img.affine, dtype=float),
            "header": template_img.header.copy(),
            "source_name": matrix_label or (source_path.name if source_path is not None else "matrix"),
            "source_dir": str(source_dir),
            "output_name": default_name,
            "keep_indices": np.asarray(keep_indices, dtype=int),
            "projection_labels": np.asarray(projection_labels, dtype=int),
            "support_mask": support_mask,
            "template_path": str(self._active_parcellation_path) if self._active_parcellation_path else "",
            "parcel_names": kept_names,
            "matrix_entry_id": self._selected_gradient_entry_id(),
            "matrix_label": matrix_label,
            "metabolite_profiles": metabolite_profiles,
            "metabolite_names": metabolite_names,
        }
        self._set_gradients_progress(0, n_grad, n_grad, f"{n_grad}/{n_grad} components (done)")
        self._sync_gradients_dialog_state()
        self.statusBar().showMessage(
            f"Computed {n_grad} diffusion component(s). Projection will run only when you save, render, classify, or open the network view."
        )

    def _save_gradients_projection(self) -> None:
        if not self._last_gradients:
            self.statusBar().showMessage("No projected gradients to save. Click Compute first.")
            return
        try:
            projected_data = self._ensure_projected_gradient_data(
                int(self._last_gradients.get("n_grad", 1))
            )
        except Exception as exc:
            self.statusBar().showMessage(f"No projected data available to save: {exc}")
            return

        try:
            import nibabel as nib
        except Exception as exc:
            self.statusBar().showMessage(f"nibabel not available: {exc}")
            return

        base_dir = Path(self._last_gradients.get("source_dir", str(self._default_results_dir())))
        default_name = self._last_gradients.get("output_name", "diffusion_components.nii.gz")
        save_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save gradient projection",
            str(base_dir / default_name),
            "NIfTI GZip (*.nii.gz);;NIfTI (*.nii);;All files (*)",
        )
        if not save_path:
            return

        output_path = Path(save_path)
        lower_name = output_path.name.lower()
        if not (lower_name.endswith(".nii") or lower_name.endswith(".nii.gz")):
            if "NIfTI (*.nii)" in selected_filter:
                output_path = output_path.with_suffix(".nii")
            else:
                output_path = output_path.with_suffix(".nii.gz")

        affine = self._last_gradients.get("affine")
        header = self._last_gradients.get("header")
        if affine is None:
            affine = np.eye(4)
        try:
            if header is not None:
                out_img = nib.Nifti1Image(projected_data, affine, header.copy())
            else:
                out_img = nib.Nifti1Image(projected_data, affine)
            nib.save(out_img, str(output_path))
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to save NIfTI: {exc}")
            return
        self.statusBar().showMessage(f"Saved projection to {output_path.name}.")

    @staticmethod
    def _compute_spectral_coords_and_order(matrix: np.ndarray):
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Projection matrix must be square.")

        n_nodes = matrix.shape[0]
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        matrix = 0.5 * (matrix + matrix.T)
        np.fill_diagonal(matrix, 0.0)

        if n_nodes == 0:
            return np.zeros((0, 2), dtype=float), np.zeros(0, dtype=int)
        if n_nodes == 1:
            return np.array([[1.0, 0.0]], dtype=float), np.array([0], dtype=int)
        if n_nodes == 2:
            return np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=float), np.array([0, 1], dtype=int)

        degree = np.sum(matrix, axis=1)
        laplacian = np.diag(degree) - matrix
        _eigvals, eigvecs = np.linalg.eigh(laplacian)
        if eigvecs.shape[1] < 3:
            raise ValueError("Could not compute the second and third Laplacian eigenvectors.")

        coords = np.asarray(eigvecs[:, 1:3], dtype=float)
        coords -= np.mean(coords, axis=0, keepdims=True)
        scale = np.max(np.linalg.norm(coords, axis=1))
        if scale > 0.0:
            coords /= scale

        angles = np.arctan2(coords[:, 1], coords[:, 0])
        order = np.argsort(angles)
        return coords, order

    def _gradient_template_img_and_data(self):
        results = self._last_gradients or {}
        template_img = self._active_parcellation_img
        template_data = self._active_parcellation_data
        template_path_raw = str(results.get("template_path") or "").strip()
        if template_path_raw:
            template_path = Path(template_path_raw)
            active_path = Path(self._active_parcellation_path) if self._active_parcellation_path is not None else None
            if template_img is None or active_path != template_path:
                try:
                    import nibabel as nib

                    template_img = nib.load(str(template_path))
                    template_data = np.asarray(template_img.get_fdata(), dtype=int)
                except Exception as exc:
                    raise RuntimeError(f"Failed to load parcellation template: {exc}") from exc
        if template_img is None:
            raise RuntimeError("No parcellation template available.")
        if template_data is None:
            try:
                template_data = np.asarray(template_img.get_fdata(), dtype=int)
            except Exception as exc:
                raise RuntimeError(f"Failed to read parcellation template data: {exc}") from exc
        template_data = np.asarray(template_data, dtype=int)
        if template_data.ndim != 3:
            raise RuntimeError("Parcellation template must be a 3D image.")
        return template_img, template_data

    @staticmethod
    def _projected_gradient_component_count(projected_data) -> int:
        if projected_data is None:
            return 0
        array = np.asarray(projected_data)
        if array.ndim == 3:
            return 1
        if array.ndim == 4:
            return int(array.shape[3])
        return 0

    def _ensure_projected_gradient_data(self, required_components: int):
        results = self._last_gradients or {}
        if not results:
            raise RuntimeError("No gradients are available.")

        gradients = np.asarray(results.get("gradients"), dtype=float)
        if gradients.ndim != 2 or gradients.shape[1] < 1:
            raise RuntimeError("No gradient components are available.")

        projection_labels = np.asarray(results.get("projection_labels"), dtype=int).reshape(-1)
        if projection_labels.size != gradients.shape[0]:
            raise RuntimeError("Gradient projection labels are out of sync with the node data.")

        target_count = self._normalize_gradient_surface_render_count(
            required_components,
            max_components=gradients.shape[1],
        )
        projected_data = results.get("projected_data")
        current_count = self._projected_gradient_component_count(projected_data)
        if current_count >= target_count:
            return projected_data

        template_img, template_data = self._gradient_template_img_and_data()
        results["affine"] = np.asarray(template_img.affine, dtype=float)
        try:
            results["header"] = template_img.header.copy()
        except Exception:
            results["header"] = None
        if results.get("support_mask") is None:
            results["support_mask"] = np.asarray(
                np.isin(template_data, projection_labels),
                dtype=np.float32,
            )
        projected_maps = []
        if projected_data is not None and current_count > 0:
            projected_array = np.asarray(projected_data, dtype=np.float32)
            if projected_array.ndim == 3:
                projected_maps.append(projected_array)
            elif projected_array.ndim == 4:
                projected_maps.extend(
                    [np.asarray(projected_array[..., idx], dtype=np.float32) for idx in range(projected_array.shape[3])]
                )

        for comp_idx in range(current_count, target_count):
            try:
                projected = nettools.project_to_3dspace(
                    np.asarray(gradients[:, comp_idx], dtype=float),
                    template_data,
                    projection_labels,
                )
            except Exception as exc:
                raise RuntimeError(f"Failed to project Gradient {comp_idx + 1}: {exc}") from exc
            projected_maps.append(np.asarray(projected, dtype=np.float32))

        if not projected_maps:
            results["projected_data"] = None
        elif len(projected_maps) == 1:
            results["projected_data"] = projected_maps[0]
        else:
            results["projected_data"] = np.stack(projected_maps, axis=-1)
        return results.get("projected_data")

    def _surface_render_gradient_matrix(self, required_components: int, *, use_procrustes: bool = False):
        results = self._last_gradients or {}
        if not results:
            raise RuntimeError("No gradients are available.")

        gradients = np.asarray(results.get("gradients"), dtype=float)
        if gradients.ndim != 2 or gradients.shape[1] < 1:
            raise RuntimeError("No gradient components are available.")
        target_count = self._normalize_gradient_surface_render_count(
            required_components,
            max_components=gradients.shape[1],
        )
        components = np.asarray(gradients[:, :target_count], dtype=float)
        if not use_procrustes:
            return components

        bundle = self._gradient_precomputed_bundle
        if not isinstance(bundle, dict):
            return components
        gradients_avg = bundle.get("gradients_avg")
        if gradients_avg is None:
            return components
        gradients_avg = np.asarray(gradients_avg, dtype=float)
        if gradients_avg.ndim != 2:
            return components
        if gradients_avg.shape[1] != components.shape[0]:
            return components

        try:
            from brainspace.gradient.alignment import procrustes
        except Exception as exc:
            raise RuntimeError(f"brainspace procrustes is unavailable: {exc}") from exc

        aligned = np.asarray(components, dtype=float).copy()
        max_ref_components = min(aligned.shape[1], gradients_avg.shape[0])
        for comp_idx in range(max_ref_components):
            source = np.asarray(aligned[:, comp_idx], dtype=float).reshape(-1, 1)
            target = np.asarray(gradients_avg[comp_idx], dtype=float).reshape(-1, 1)
            try:
                # Align the selected subject/session component onto the average reference.
                aligned_component = procrustes(source, target, center=False, scale=False)
            except Exception as exc:
                raise RuntimeError(f"Failed Procrustes alignment for Gradient {comp_idx + 1}: {exc}") from exc
            aligned[:, comp_idx] = np.asarray(aligned_component, dtype=float).reshape(-1)
        return aligned

    def _project_gradient_matrix_to_volume(self, gradient_matrix: np.ndarray):
        results = self._last_gradients or {}
        if not results:
            raise RuntimeError("No gradients are available.")
        components = np.asarray(gradient_matrix, dtype=float)
        if components.ndim != 2 or components.shape[1] < 1:
            raise RuntimeError("Gradient matrix must be 2D with at least one component.")

        projection_labels = np.asarray(results.get("projection_labels"), dtype=int).reshape(-1)
        if projection_labels.size != components.shape[0]:
            raise RuntimeError("Gradient projection labels are out of sync with the node data.")

        template_img, template_data = self._gradient_template_img_and_data()
        results["affine"] = np.asarray(template_img.affine, dtype=float)
        try:
            results["header"] = template_img.header.copy()
        except Exception:
            results["header"] = None
        if results.get("support_mask") is None:
            results["support_mask"] = np.asarray(
                np.isin(template_data, projection_labels),
                dtype=np.float32,
            )

        projected_maps = []
        for comp_idx in range(components.shape[1]):
            try:
                projected = nettools.project_to_3dspace(
                    np.asarray(components[:, comp_idx], dtype=float),
                    template_data,
                    projection_labels,
                )
            except Exception as exc:
                raise RuntimeError(f"Failed to project Gradient {comp_idx + 1}: {exc}") from exc
            projected_maps.append(np.asarray(projected, dtype=np.float32))

        if len(projected_maps) == 1:
            return projected_maps[0]
        return np.stack(projected_maps, axis=-1)

    def _gradient_spatial_embedding(self):
        results = self._last_gradients or {}
        if not results:
            raise RuntimeError("No gradients available for spatial classification.")

        try:
            projection_labels = np.asarray(results.get("projection_labels"), dtype=int).reshape(-1)
        except Exception as exc:
            raise RuntimeError(f"Invalid projection labels for spatial classification: {exc}") from exc
        if projection_labels.size == 0:
            raise RuntimeError("No projection labels available for spatial classification.")

        cached = results.get("spatial_embedding")
        if isinstance(cached, dict):
            cached_labels = np.asarray(cached.get("projection_labels", []), dtype=int).reshape(-1)
            coords = np.asarray(cached.get("coords"), dtype=float)
            if (
                cached_labels.shape == projection_labels.shape
                and np.array_equal(cached_labels, projection_labels)
                and coords.shape == (projection_labels.size, 2)
            ):
                return cached

        try:
            from scipy.spatial.distance import cdist
        except Exception as exc:
            raise RuntimeError(f"scipy distance tools are unavailable: {exc}") from exc

        template_img, template_data = self._gradient_template_img_and_data()
        try:
            # Spatial distances should be measured in affine/world space, not voxel index space.
            centroids_world = np.asarray(
                nettools.compute_centroids(
                    template_img,
                    labels=np.asarray(projection_labels, dtype=int),
                    world=True,
                ),
                dtype=float,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to compute world-space parcel centroids: {exc}") from exc
        if centroids_world.shape[0] != projection_labels.shape[0]:
            raise RuntimeError("Spatial centroid count does not match the projected parcels.")

        distance_matrix = np.asarray(cdist(centroids_world, centroids_world, metric="euclidean"), dtype=float)
        positive_distances = distance_matrix[distance_matrix > 0]
        if positive_distances.size == 0:
            raise RuntimeError("Spatial centroid distances are degenerate.")
        sigma = float(np.median(positive_distances))
        if not np.isfinite(sigma) or sigma <= 0.0:
            sigma = float(np.mean(positive_distances))
        if not np.isfinite(sigma) or sigma <= 0.0:
            sigma = 1.0
        # The spectral helper expects an affinity-like matrix. Raw distances collapse badly here.
        affinity_matrix = np.exp(-np.square(distance_matrix) / (2.0 * sigma * sigma))
        np.fill_diagonal(affinity_matrix, 0.0)
        coords, order = self._compute_spectral_coords_and_order(affinity_matrix)

        try:
            projected_x = np.asarray(
                nettools.project_to_3dspace(coords[:, 0], template_data, projection_labels),
                dtype=np.float32,
            )
            projected_y = np.asarray(
                nettools.project_to_3dspace(coords[:, 1], template_data, projection_labels),
                dtype=np.float32,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to project spatial spectral coordinates: {exc}") from exc

        cached = {
            "projection_labels": np.asarray(projection_labels, dtype=int),
            "centroids_world": centroids_world,
            "distance_matrix": distance_matrix,
            "affinity_matrix": affinity_matrix,
            "coords": np.asarray(coords, dtype=float),
            "order": np.asarray(order, dtype=int),
            "projected_x": projected_x,
            "projected_y": projected_y,
        }
        results["spatial_embedding"] = cached
        return cached

    def _classification_spatial_embedding(self, axis_gradients, *, align_to_gradients: bool = True):
        spatial = dict(self._gradient_spatial_embedding() or {})
        if not align_to_gradients:
            return spatial

        gradient_coords = np.asarray(axis_gradients, dtype=float)
        spatial_coords = np.asarray(spatial.get("coords"), dtype=float)
        projection_labels = np.asarray(spatial.get("projection_labels", []), dtype=int).reshape(-1)
        if (
            gradient_coords.ndim != 2
            or spatial_coords.ndim != 2
            or gradient_coords.shape[0] != spatial_coords.shape[0]
            or spatial_coords.shape[1] != 2
            or gradient_coords.shape[1] < 2
        ):
            return spatial

        results = self._last_gradients or {}
        cached = results.get("spatial_embedding_aligned")
        if isinstance(cached, dict):
            cached_labels = np.asarray(cached.get("projection_labels", []), dtype=int).reshape(-1)
            cached_coords = np.asarray(cached.get("coords"), dtype=float)
            if (
                cached_labels.shape == projection_labels.shape
                and np.array_equal(cached_labels, projection_labels)
                and cached_coords.shape == spatial_coords.shape
            ):
                return cached

        try:
            from brainspace.gradient.alignment import procrustes
        except Exception:
            return spatial

        target_coords = np.column_stack(
            (
                np.asarray(gradient_coords[:, 1], dtype=float),
                np.asarray(gradient_coords[:, 0], dtype=float),
            )
        )
        finite_mask = np.all(np.isfinite(spatial_coords), axis=1) & np.all(np.isfinite(target_coords), axis=1)
        if int(np.sum(finite_mask)) < 3:
            return spatial

        aligned_coords = np.asarray(spatial_coords, dtype=float).copy()
        try:
            aligned_coords[finite_mask, :] = np.asarray(
                procrustes(
                    np.asarray(spatial_coords[finite_mask, :], dtype=float),
                    np.asarray(target_coords[finite_mask, :], dtype=float),
                    center=True,
                    scale=False,
                ),
                dtype=float,
            )
        except Exception:
            return spatial

        try:
            _template_img, template_data = self._gradient_template_img_and_data()
            projected_x = np.asarray(
                nettools.project_to_3dspace(aligned_coords[:, 0], template_data, projection_labels),
                dtype=np.float32,
            )
            projected_y = np.asarray(
                nettools.project_to_3dspace(aligned_coords[:, 1], template_data, projection_labels),
                dtype=np.float32,
            )
        except Exception:
            return spatial

        aligned = {
            **spatial,
            "coords": np.asarray(aligned_coords, dtype=float),
            "projected_x": projected_x,
            "projected_y": projected_y,
            "aligned_to_gradients": True,
        }
        results["spatial_embedding_aligned"] = aligned
        return aligned

    @staticmethod
    def _classification_spatial_indices(x_axis: str, y_axis: str):
        x_norm = GradientDialogController._normalize_gradient_classification_axis(x_axis, default="gradient1")
        y_norm = GradientDialogController._normalize_gradient_classification_axis(y_axis, default="gradient1")
        if x_norm == "spatial" and y_norm == "spatial":
            return 0, 1, "Spatial 1", "Spatial 2"
        if x_norm == "spatial":
            x_index = 1 if y_norm in {"gradient2", "gradient3", "gradient_ref2", "gradient_ref3"} else 0
            return x_index, 0, "Spatial", "Spatial"
        if y_norm == "spatial":
            y_index = 1 if x_norm in {"gradient2", "gradient3", "gradient_ref2", "gradient_ref3"} else 0
            return 0, y_index, "Spatial", "Spatial"
        return 0, 0, "Spatial", "Spatial"

    def _classification_reference_gradient_payload(self, required_components: int):
        try:
            required_components = int(required_components)
        except Exception:
            required_components = 1
        required_components = max(1, min(3, required_components))

        entry = self._selected_gradient_classification_ref_entry()
        if entry is None:
            raise RuntimeError("Select a reference matrix for Gradient Ref classification axes.")

        ref_label = self._gradient_matrix_label_for_entry(entry)
        try:
            ref_matrix = np.asarray(self._gradient_matrix_for_entry(entry), dtype=float)
        except Exception as exc:
            raise RuntimeError(f"Failed to load reference matrix {ref_label}: {exc}") from exc
        if ref_matrix.ndim != 2 or ref_matrix.shape[0] != ref_matrix.shape[1]:
            raise RuntimeError(f"Reference gradients require a square matrix. {ref_label} is not square.")

        results = self._last_gradients or {}
        projection_labels = np.asarray(results.get("projection_labels"), dtype=int).reshape(-1)
        if projection_labels.size == 0:
            raise RuntimeError("Current gradient projection labels are unavailable.")

        ref_labels_raw, _ref_names = self._entry_parcel_metadata(entry, expected_len=ref_matrix.shape[0])
        ref_label_indices = _coerce_label_indices(ref_labels_raw, ref_matrix.shape[0])
        if ref_label_indices is not None:
            ref_label_indices = np.asarray(ref_label_indices, dtype=int).reshape(-1)
            if np.unique(ref_label_indices).size != ref_label_indices.size:
                raise RuntimeError("Reference matrix parcel_labels_group must be unique.")
            label_to_index = {
                int(label): int(idx)
                for idx, label in enumerate(ref_label_indices.tolist())
            }
            missing = [int(label) for label in projection_labels.tolist() if int(label) not in label_to_index]
            if missing:
                preview = ", ".join(str(label) for label in missing[:8])
                if len(missing) > 8:
                    preview += ", ..."
                raise RuntimeError(
                    f"Reference matrix {ref_label} is missing projection label(s): {preview}"
                )
            ref_keep_indices = np.asarray(
                [label_to_index[int(label)] for label in projection_labels.tolist()],
                dtype=int,
            )
        else:
            keep_indices = np.asarray(results.get("keep_indices", []), dtype=int).reshape(-1)
            if ref_matrix.shape[0] == projection_labels.size:
                ref_keep_indices = np.arange(projection_labels.size, dtype=int)
            elif (
                keep_indices.size == projection_labels.size
                and keep_indices.size > 0
                and int(np.max(keep_indices)) < ref_matrix.shape[0]
            ):
                ref_keep_indices = np.asarray(keep_indices, dtype=int)
            else:
                raise RuntimeError(
                    "Reference matrix needs parcel_labels_group matching the active projection labels, "
                    "or the same row order/size as the current matrix."
                )

        ref_gradients = np.zeros((projection_labels.size, required_components), dtype=float)
        for comp_idx in range(1, required_components + 1):
            try:
                component = nettools.dimreduce_matrix(
                    ref_matrix,
                    method="diffusion",
                    scale_factor=1.0,
                    output_dim=comp_idx,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to compute Gradient Ref {comp_idx} from {ref_label}: {exc}"
                ) from exc
            component = np.asarray(component, dtype=float).reshape(-1)
            if component.size != ref_matrix.shape[0]:
                raise RuntimeError(
                    f"Gradient Ref {comp_idx} size mismatch ({component.size} vs {ref_matrix.shape[0]})."
                )
            ref_gradients[:, comp_idx - 1] = np.asarray(component[ref_keep_indices], dtype=float)

        projected_data = self._project_gradient_matrix_to_volume(ref_gradients)
        return {
            "entry_id": entry.get("id"),
            "label": ref_label,
            "gradients": np.asarray(ref_gradients, dtype=float),
            "projected_data": projected_data,
        }

    def _classification_axis_payload(
        self,
        axis_key: str,
        gradients,
        projected_data,
        *,
        spatial_index: int = 0,
        spatial_label: str = "Spatial",
        spatial_embedding_override=None,
        reference_payload=None,
    ):
        axis = self._normalize_gradient_classification_axis(axis_key, default="gradient1")
        if axis == "gradient1":
            if gradients.ndim != 2 or gradients.shape[1] < 1:
                raise RuntimeError("Gradient 1 is not available for classification.")
            if projected_data.ndim == 4:
                volume = np.asarray(projected_data[..., 0], dtype=float)
            else:
                volume = np.asarray(projected_data, dtype=float)
            return np.asarray(gradients[:, 0], dtype=float), volume, "Gradient 1"
        if axis == "gradient2":
            if gradients.ndim != 2 or gradients.shape[1] < 2:
                raise RuntimeError("Gradient 2 is not available for classification.")
            if projected_data.ndim != 4 or projected_data.shape[3] < 2:
                raise RuntimeError("Projected Gradient 2 volume is not available.")
            return np.asarray(gradients[:, 1], dtype=float), np.asarray(projected_data[..., 1], dtype=float), "Gradient 2"
        if axis == "gradient3":
            if gradients.ndim != 2 or gradients.shape[1] < 3:
                raise RuntimeError("Gradient 3 is not available for classification.")
            if projected_data.ndim != 4 or projected_data.shape[3] < 3:
                raise RuntimeError("Projected Gradient 3 volume is not available.")
            return np.asarray(gradients[:, 2], dtype=float), np.asarray(projected_data[..., 2], dtype=float), "Gradient 3"

        if axis in {"gradient_ref1", "gradient_ref2", "gradient_ref3"}:
            if not isinstance(reference_payload, dict):
                raise RuntimeError("Reference gradients are not available.")
            ref_gradients = np.asarray(reference_payload.get("gradients"), dtype=float)
            ref_projected = np.asarray(reference_payload.get("projected_data"), dtype=float)
            comp_idx = {"gradient_ref1": 0, "gradient_ref2": 1, "gradient_ref3": 2}.get(axis, 0)
            if ref_gradients.ndim != 2 or ref_gradients.shape[1] <= comp_idx:
                raise RuntimeError(f"Gradient Ref {comp_idx + 1} is not available for classification.")
            if comp_idx == 0 and ref_projected.ndim == 3:
                volume = np.asarray(ref_projected, dtype=float)
            elif ref_projected.ndim == 4 and ref_projected.shape[3] > comp_idx:
                volume = np.asarray(ref_projected[..., comp_idx], dtype=float)
            else:
                raise RuntimeError(f"Projected Gradient Ref {comp_idx + 1} volume is not available.")
            return (
                np.asarray(ref_gradients[:, comp_idx], dtype=float),
                volume,
                f"Gradient Ref {comp_idx + 1}",
            )

        spatial = (
            dict(spatial_embedding_override)
            if isinstance(spatial_embedding_override, dict)
            else self._gradient_spatial_embedding()
        )
        coord_index = max(0, min(int(spatial_index), 1))
        volume_key = "projected_x" if coord_index == 0 else "projected_y"
        return (
            np.asarray(spatial["coords"][:, coord_index], dtype=float),
            np.asarray(spatial[volume_key], dtype=float),
            str(spatial_label or ("Spatial 1" if coord_index == 0 else "Spatial 2")),
        )

    @staticmethod
    def _rescale_classification_axis_to_range(values, volume, target_values):
        axis_values = np.asarray(values, dtype=float)
        axis_volume = np.asarray(volume, dtype=float)
        target = np.asarray(target_values, dtype=float)

        finite_axis = axis_values[np.isfinite(axis_values)]
        finite_target = target[np.isfinite(target)]
        if finite_axis.size == 0 or finite_target.size == 0:
            return axis_values, axis_volume

        src_min = float(np.min(finite_axis))
        src_max = float(np.max(finite_axis))
        tgt_min = float(np.min(finite_target))
        tgt_max = float(np.max(finite_target))
        if not all(np.isfinite(value) for value in (src_min, src_max, tgt_min, tgt_max)):
            return axis_values, axis_volume

        scaled_values = np.array(axis_values, copy=True, dtype=float)
        scaled_volume = np.array(axis_volume, copy=True, dtype=float)
        finite_values_mask = np.isfinite(scaled_values)
        finite_volume_mask = np.isfinite(scaled_volume)

        if abs(src_max - src_min) <= 1e-12:
            midpoint = 0.5 * (tgt_min + tgt_max)
            scaled_values[finite_values_mask] = midpoint
            scaled_volume[finite_volume_mask] = midpoint
            return scaled_values, scaled_volume

        scale = (tgt_max - tgt_min) / (src_max - src_min)
        scaled_values[finite_values_mask] = (scaled_values[finite_values_mask] - src_min) * scale + tgt_min
        scaled_volume[finite_volume_mask] = (scaled_volume[finite_volume_mask] - src_min) * scale + tgt_min
        return scaled_values, scaled_volume

    def _render_gradients_3d(self) -> None:
        if not self._last_gradients:
            self.statusBar().showMessage("No gradients computed yet.")
            return
        try:
            from window.plot_gradient import GradientSurfaceDialog
        except Exception:
            try:
                from mrsi_viewer.window.plot_gradient import GradientSurfaceDialog
            except Exception as exc:
                self.statusBar().showMessage(f"Gradient surface viewer unavailable: {exc}")
                return
        try:
            available_n_grad = int(self._last_gradients.get("n_grad", 1))
            render_count = self._selected_gradient_surface_render_count()
            render_count = max(1, min(render_count, available_n_grad))
            use_procrustes = self._selected_gradient_surface_procrustes()
            if use_procrustes:
                component_matrix = self._surface_render_gradient_matrix(
                    render_count,
                    use_procrustes=True,
                )
                projected_data = self._project_gradient_matrix_to_volume(component_matrix)
            else:
                projected_data = self._ensure_projected_gradient_data(render_count)
            if projected_data is None:
                raise RuntimeError("No projected fsaverage data are available.")
            render_data = np.asarray(projected_data, dtype=float)
            if render_count == 1 and render_data.ndim == 4:
                render_data = np.asarray(render_data[..., 0], dtype=float)
            elif render_data.ndim == 4 and render_data.shape[3] > render_count:
                render_data = render_data[..., :render_count]
            source_name = self._last_gradients.get("source_name", "matrix")
            cmap_name = self._selected_surface_colormap_name()
            cmap = self._selected_surface_colormap()
            hemisphere_mode = self._selected_gradient_hemisphere_mode()
            surface_mesh = self._selected_gradient_surface_mesh()
            title = (
                f"Gradient 1 - {source_name}"
                if render_count == 1
                else f"First {render_count} Gradients - {source_name}"
            )
            if use_procrustes:
                title += " | Procrustes to gradients_avg"
            self._surface_dialog = GradientSurfaceDialog.from_array(
                render_data,
                affine=self._last_gradients.get("affine"),
                title=title,
                cmap=cmap,
                cmap_name=cmap_name,
                theme_name=self._theme_name,
                hemisphere_mode=hemisphere_mode,
                fsaverage_mesh=surface_mesh,
                parent=self,
            )
            screen = QApplication.primaryScreen()
            if screen is not None:
                geom = screen.availableGeometry()
                width = max(int(geom.width() * 0.88), 900)
                height = min(max(300 + 300 * max(render_count, 1), 420), int(geom.height() * 0.9))
            else:
                width = 1500
                height = 300 + 300 * max(render_count, 1)
            self._surface_dialog.resize(width, height)
            self._surface_dialog.show()
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to render Gradient fsaverage surfaces: {exc}")
            return
        self.statusBar().showMessage("Opened Gradient fsaverage viewer.")

    def _classify_gradients_fsaverage(self) -> None:
        if not self._last_gradients:
            self.statusBar().showMessage("No gradients computed yet.")
            return

        gradients = np.asarray(self._last_gradients.get("gradients"), dtype=float)
        projection_labels = np.asarray(self._last_gradients.get("projection_labels"), dtype=int).reshape(-1)
        if gradients.ndim != 2 or gradients.shape[1] < 1:
            self.statusBar().showMessage("No gradient components are available for classification.")
            return
        if projection_labels.size == 0 or projection_labels.size != gradients.shape[0]:
            self.statusBar().showMessage("Projected gradient nodes are out of sync. Compute gradients again.")
            return
        if not self._can_classify_gradients():
            self.statusBar().showMessage("The selected classification axes are not available for the current gradients.")
            return

        try:
            from window.plot_gradient import GradientClassificationDialog, GradientScatterDialog
        except Exception:
            try:
                from mrsi_viewer.window.plot_gradient import GradientClassificationDialog, GradientScatterDialog
            except Exception as exc:
                self.statusBar().showMessage(f"Gradient classification viewer unavailable: {exc}")
                return

        try:
            source_name = self._last_gradients.get("source_name", "matrix")
            hemisphere_mode = self._selected_gradient_classification_hemisphere_mode()
            surface_hemisphere_mode = "both" if hemisphere_mode == "separate" else hemisphere_mode
            surface_mesh = self._selected_gradient_classification_surface_mesh()
            scatter_rotation = self._selected_gradient_scatter_rotation()
            use_triangular_rgb = self._selected_gradient_triangular_rgb()
            preload_matching_paths = self._selected_gradient_preload_matching_paths()
            classification_fit_mode = self._selected_gradient_classification_fit_mode()
            triangular_color_order = self._selected_gradient_triangular_color_order()
            rgb_scalar_mode = self._selected_gradient_rgb_scalar_mode()
            rgb_mask_strictness = self._selected_gradient_rgb_mask_strictness()
            x_axis = self._selected_gradient_classification_x_axis()
            y_axis = self._selected_gradient_classification_y_axis()
            z_axis = self._selected_gradient_classification_z_axis()
            is_3d_embedding = z_axis != "none"
            effective_use_triangular_rgb = bool(use_triangular_rgb) or is_3d_embedding
            x_spatial_index, y_spatial_index, x_spatial_label, y_spatial_label = self._classification_spatial_indices(
                x_axis,
                y_axis,
            )
            class_component = int(
                self._normalize_gradient_classification_component(
                    self._selected_gradient_classification_component(),
                    max_components=gradients.shape[1],
                )
            ) - 1
            class_component = max(0, min(class_component, gradients.shape[1] - 1))
            use_own_rgb_source = bool(effective_use_triangular_rgb) and self._classification_uses_reference_axis(
                x_axis,
                y_axis,
                z_axis,
            )
            required_rgb_components = 3 if is_3d_embedding else 2
            if use_own_rgb_source and gradients.shape[1] < required_rgb_components:
                raise RuntimeError(
                    f"Gradient Ref RGB coloring requires current Gradient 1 through Gradient {required_rgb_components}."
                )
            required_projection_count = 1
            required_projection_count = max(
                required_projection_count,
                self._classification_axis_component_count(x_axis),
                self._classification_axis_component_count(y_axis),
                self._classification_axis_component_count(z_axis),
            )
            if use_own_rgb_source:
                required_projection_count = max(required_projection_count, required_rgb_components)
            required_projection_count = max(required_projection_count, class_component + 1)
            projected_data = self._ensure_projected_gradient_data(required_projection_count)
            if projected_data is None:
                raise RuntimeError("No projected fsaverage data are available for classification.")
            classification_cmap_name = self._selected_gradient_classification_colormap()
            classification_cmap = self._selected_surface_colormap(classification_cmap_name)
            axis_gradients = np.asarray(gradients, dtype=float)
            rgb_x_values = None
            rgb_y_values = None
            rgb_z_values = None
            rgb_x_volume = None
            rgb_y_volume = None
            rgb_z_volume = None
            if use_own_rgb_source:
                rgb_x_values = np.asarray(axis_gradients[:, 1], dtype=float)
                rgb_y_values = np.asarray(axis_gradients[:, 0], dtype=float)
                if is_3d_embedding:
                    rgb_z_values = np.asarray(axis_gradients[:, 2], dtype=float)
                if projected_data.ndim != 4 or projected_data.shape[3] < required_rgb_components:
                    raise RuntimeError(
                        f"Projected current Gradient 1 through Gradient {required_rgb_components} volumes are required for RGB coloring."
                    )
                rgb_y_volume = np.asarray(projected_data[..., 0], dtype=float)
                rgb_x_volume = np.asarray(projected_data[..., 1], dtype=float)
                if is_3d_embedding:
                    rgb_z_volume = np.asarray(projected_data[..., 2], dtype=float)
            reference_payload = None
            reference_component_count = self._classification_reference_component_count(x_axis, y_axis, z_axis)
            if reference_component_count > 0:
                reference_payload = self._classification_reference_gradient_payload(reference_component_count)
            spatial_embedding_override = None
            if x_axis == "spatial" or y_axis == "spatial":
                spatial_alignment_gradients = axis_gradients
                if reference_payload is not None and (
                    x_axis in {"gradient_ref1", "gradient_ref2", "gradient_ref3"}
                    or y_axis in {"gradient_ref1", "gradient_ref2", "gradient_ref3"}
                ):
                    spatial_alignment_gradients = np.asarray(reference_payload.get("gradients"), dtype=float)
                spatial_embedding_override = self._classification_spatial_embedding(
                    spatial_alignment_gradients,
                    align_to_gradients=True,
                )
            x_values, x_volume, x_label = self._classification_axis_payload(
                x_axis,
                axis_gradients,
                projected_data,
                spatial_index=x_spatial_index,
                spatial_label=x_spatial_label,
                spatial_embedding_override=spatial_embedding_override,
                reference_payload=reference_payload,
            )
            y_values, y_volume, y_label = self._classification_axis_payload(
                y_axis,
                axis_gradients,
                projected_data,
                spatial_index=y_spatial_index,
                spatial_label=y_spatial_label,
                spatial_embedding_override=spatial_embedding_override,
                reference_payload=reference_payload,
            )
            z_values = None
            z_volume = None
            z_label = ""
            if is_3d_embedding:
                z_values, z_volume, z_label = self._classification_axis_payload(
                    z_axis,
                    axis_gradients,
                    projected_data,
                    reference_payload=reference_payload,
                )
            gradient_like_axes = {"gradient1", "gradient2", "gradient3", "gradient_ref1", "gradient_ref2", "gradient_ref3"}
            if x_axis == "spatial" and y_axis in gradient_like_axes:
                x_values, x_volume = self._rescale_classification_axis_to_range(
                    x_values,
                    x_volume,
                    y_values,
                )
            if y_axis == "spatial" and x_axis in gradient_like_axes:
                y_values, y_volume = self._rescale_classification_axis_to_range(
                    y_values,
                    y_volume,
                    x_values,
                )
            class_component_values = np.asarray(axis_gradients[:, class_component], dtype=float)
            finite_mask = np.isfinite(x_values) & np.isfinite(y_values) & np.isfinite(class_component_values)
            if is_3d_embedding:
                finite_mask &= np.isfinite(z_values)
            if use_own_rgb_source:
                finite_mask &= np.isfinite(rgb_x_values) & np.isfinite(rgb_y_values)
                if is_3d_embedding:
                    finite_mask &= np.isfinite(rgb_z_values)
            if not np.any(finite_mask):
                axis_text = f"{z_label} vs {y_label} vs {x_label}" if is_3d_embedding else f"{y_label} vs {x_label}"
                self.statusBar().showMessage(f"No finite values available for {axis_text} classification.")
                return
            projection_labels = np.asarray(self._last_gradients.get("projection_labels"), dtype=int).reshape(-1)
            if projection_labels.shape[0] != x_values.shape[0]:
                self.statusBar().showMessage(
                    "Classification labels are out of sync with the projected nodes. Compute gradients again."
                )
                return
            finite_mask &= self._gradient_projection_hemisphere_mask(
                hemisphere_mode,
                projection_labels,
            )
            if not np.any(finite_mask):
                self.statusBar().showMessage(
                    f"No {hemisphere_mode.upper()} parcels are available for {y_label} vs {x_label} classification."
                )
                return
            parcel_names = _to_string_list(self._last_gradients.get("parcel_names"))
            point_labels = []
            for idx, label in enumerate(projection_labels.tolist()):
                label_text = ""
                if parcel_names and idx < len(parcel_names):
                    label_text = str(parcel_names[idx] or "").strip()
                if not label_text:
                    label_text = f"Parcel {int(label)}"
                point_labels.append(label_text)
            point_labels = np.asarray(point_labels, dtype=object)
            hemisphere_codes_full = np.asarray(self._gradient_projection_hemisphere_codes(), dtype=int).reshape(-1)
            if hemisphere_codes_full.shape != projection_labels.shape:
                raise RuntimeError("Hemisphere membership is out of sync with the projected labels.")

            ignore_lh_name = self._selected_gradient_classification_ignore_lh_parcel()
            ignore_rh_name = self._selected_gradient_classification_ignore_rh_parcel()
            if ignore_lh_name or ignore_rh_name:
                ignore_mask = np.zeros(projection_labels.shape, dtype=bool)
                normalized_labels = np.asarray([str(text or "").strip().lower() for text in point_labels.tolist()], dtype=object)
                if ignore_lh_name:
                    target = str(ignore_lh_name).strip().lower()
                    ignore_mask |= ((hemisphere_codes_full == 0) | (hemisphere_codes_full == 2)) & (normalized_labels == target)
                if ignore_rh_name:
                    target = str(ignore_rh_name).strip().lower()
                    ignore_mask |= ((hemisphere_codes_full == 1) | (hemisphere_codes_full == 2)) & (normalized_labels == target)
                finite_mask &= ~ignore_mask
            else:
                ignore_mask = np.zeros(projection_labels.shape, dtype=bool)
            if not np.any(finite_mask):
                self.statusBar().showMessage(
                    f"No parcels remain for {y_label} vs {x_label} after applying the ignore-parcel selection."
                )
                return
            metabolite_profiles = self._last_gradients.get("metabolite_profiles")
            metabolite_names = list(self._last_gradients.get("metabolite_names") or [])
            metabolite_subject_labels = list(self._last_gradients.get("metabolite_subject_labels") or [])
            custom_metabolite_path = self._selected_gradient_metabolite_profiles_path()
            if custom_metabolite_path is not None:
                if custom_metabolite_path.exists():
                    metabolite_subject_labels = self._metabolite_subject_labels_from_npz(custom_metabolite_path)
                    row_index_for_profiles = self._last_gradients.get("precomputed_row_index", None)
                    metabolite_profiles, metabolite_names = self._load_metabolite_profiles_for_gradient_context(
                        custom_metabolite_path,
                        target_labels=projection_labels,
                        target_names=point_labels,
                        row_index=row_index_for_profiles,
                        preserve_subjects=row_index_for_profiles is None,
                    )
                    if metabolite_profiles is None:
                        metabolite_names = []
                        self.statusBar().showMessage(
                            f"Custom metabolite profiles did not align to this gradient context: {custom_metabolite_path}"
                        )
                else:
                    metabolite_profiles = None
                    metabolite_names = []
                    metabolite_subject_labels = []
                    self.statusBar().showMessage(f"Custom metabolite profile file does not exist: {custom_metabolite_path}")
            elif not metabolite_subject_labels:
                entry_id = self._last_gradients.get("matrix_entry_id")
                entry = self._entries.get(entry_id) if entry_id in self._entries else None
                profile_source_path = self._gradient_entry_source_path(entry)
                if profile_source_path is not None and profile_source_path.exists():
                    metabolite_subject_labels = self._metabolite_subject_labels_from_npz(profile_source_path)
            if metabolite_profiles is not None:
                try:
                    metabolite_profiles = np.asarray(metabolite_profiles, dtype=float)
                    valid_metabolite_profiles = (
                        metabolite_profiles.ndim == 2
                        and metabolite_profiles.shape[0] == finite_mask.shape[0]
                        and metabolite_profiles.shape[1] > 0
                    ) or (
                        metabolite_profiles.ndim == 3
                        and metabolite_profiles.shape[1] == finite_mask.shape[0]
                        and metabolite_profiles.shape[2] > 0
                    )
                    if not valid_metabolite_profiles:
                        metabolite_profiles = None
                        metabolite_names = []
                        metabolite_subject_labels = []
                    elif metabolite_profiles.ndim == 3 and len(metabolite_subject_labels) != metabolite_profiles.shape[0]:
                        metabolite_subject_labels = [
                            f"Subject {idx + 1}" for idx in range(int(metabolite_profiles.shape[0]))
                        ]
                except Exception:
                    metabolite_profiles = None
                    metabolite_names = []
                    metabolite_subject_labels = []
            scatter_title = (
                f"{z_label} vs {y_label} vs {x_label} - {source_name}"
                if is_3d_embedding
                else f"{y_label} vs {x_label} - {source_name}"
            )
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to prepare classification axes: {exc}")
            return

        edge_pairs = np.zeros((0, 2), dtype=int)
        adjacency_note = None
        if not is_3d_embedding:
            try:
                edge_pairs, adjacency_note = self._classification_scatter_edge_pairs(
                    projection_labels,
                    finite_mask,
                )
            except Exception as exc:
                adjacency_note = f"Adjacency skipped: {exc}"

        scatter_error = None
        surface_error = None
        try:
            scatter_projection_labels = np.asarray(projection_labels[finite_mask], dtype=int)
            scatter_point_labels = np.asarray(point_labels[finite_mask], dtype=object)
            scatter_hemisphere_codes = np.asarray(hemisphere_codes_full[finite_mask], dtype=int)
            scatter_export_metadata = {
                "source_name": source_name,
                "source_dir": str(self._last_gradients.get("source_dir", self._default_results_dir())),
                "input_path": str(self._last_gradients.get("precomputed_source_path", "") or ""),
                "group_gradients_path": str(self._last_gradients.get("precomputed_source_path", "") or ""),
                "parc_path": str(self._last_gradients.get("template_path", "") or ""),
                "template_path": str(self._last_gradients.get("template_path", "") or ""),
                "adjacency_path": str(self._gradient_classification_adjacency_path or ""),
                "gradient1_values": np.asarray(axis_gradients[:, 0], dtype=float)[finite_mask],
                "gradient2_values": (
                    np.asarray(axis_gradients[:, 1], dtype=float)[finite_mask]
                    if axis_gradients.ndim == 2 and axis_gradients.shape[1] >= 2
                    else np.full(scatter_projection_labels.shape, np.nan, dtype=float)
                ),
                "gradient3_values": (
                    np.asarray(axis_gradients[:, 2], dtype=float)[finite_mask]
                    if axis_gradients.ndim == 2 and axis_gradients.shape[1] >= 3
                    else np.full(scatter_projection_labels.shape, np.nan, dtype=float)
                ),
            }
            gradients_pair_export = np.full((scatter_projection_labels.size, 2), np.nan, dtype=float)
            if axis_gradients.ndim == 2 and axis_gradients.shape[1] >= 1:
                gradients_pair_export[:, 0] = np.asarray(axis_gradients[:, 0], dtype=float)[finite_mask]
            if axis_gradients.ndim == 2 and axis_gradients.shape[1] >= 2:
                gradients_pair_export[:, 1] = np.asarray(axis_gradients[:, 1], dtype=float)[finite_mask]
            scatter_export_metadata["gradients_pair"] = gradients_pair_export

            gradients_avg_export = np.empty((0, 0), dtype=float)
            bundle = self._gradient_precomputed_bundle
            if isinstance(bundle, dict):
                gradients_avg_raw = bundle.get("gradients_avg")
                if gradients_avg_raw is not None:
                    gradients_avg_array = np.asarray(gradients_avg_raw, dtype=float)
                    if gradients_avg_array.ndim == 2 and gradients_avg_array.shape[1] == finite_mask.shape[0]:
                        gradients_avg_export = np.asarray(gradients_avg_array[:, finite_mask], dtype=float)
            scatter_export_metadata["gradients_avg"] = gradients_avg_export

            covars_row = {}
            participant = ""
            session = ""
            group_value = ""
            modality_value = ""
            precomputed_row_index = self._last_gradients.get("precomputed_row_index", None)
            if isinstance(bundle, dict) and precomputed_row_index is not None:
                try:
                    row_index = int(precomputed_row_index)
                    participant, session = self._gradient_precomputed_row_pair(bundle, row_index)
                    bundle_rows = list(bundle.get("covars_rows") or [])
                    if 0 <= row_index < len(bundle_rows) and isinstance(bundle_rows[row_index], dict):
                        covars_row = dict(bundle_rows[row_index])
                except Exception:
                    covars_row = {}
                group_value = str(covars_row.get("group", "") or "").strip()
                modality_value = str(covars_row.get("modality", "") or "").strip()
            else:
                entry_id = self._last_gradients.get("matrix_entry_id")
                entry = self._entries.get(entry_id) if entry_id in self._entries else None
                source_path = self._gradient_entry_source_path(entry)
                if source_path is not None and source_path.exists():
                    try:
                        with np.load(source_path, allow_pickle=True) as npz:
                            group_value = self._npz_optional_scalar_text(npz, "group")
                            modality_value = self._npz_optional_scalar_text(npz, "modality")
                            participant = self._npz_optional_scalar_text(
                                npz,
                                "subject_id",
                                "participant_id",
                            )
                            session = self._npz_optional_scalar_text(
                                npz,
                                "session_id",
                                "session",
                            )
                    except Exception:
                        pass
                    scatter_export_metadata["source_path"] = str(source_path)
            scatter_export_metadata["subject_id"] = participant
            scatter_export_metadata["session_id"] = session
            scatter_export_metadata["group"] = group_value
            scatter_export_metadata["modality"] = modality_value
            scatter_export_metadata["covars_row"] = covars_row

            def _project_paths_callback(payload):
                self._project_classification_paths_to_brain(
                    payload,
                    scatter_projection_labels,
                    scatter_point_labels,
                    hemisphere_mode=hemisphere_mode,
                    source_name=source_name,
                    x_label=x_label,
                    y_label=y_label,
                )

            self._gradient_scatter_dialog = GradientScatterDialog(
                x_values[finite_mask],
                y_values[finite_mask],
                z_values=z_values[finite_mask] if is_3d_embedding else None,
                color_values=class_component_values[finite_mask],
                gradient1_values=np.asarray(axis_gradients[:, 0], dtype=float)[finite_mask],
                rgb_x_values=rgb_x_values[finite_mask] if use_own_rgb_source else None,
                rgb_y_values=rgb_y_values[finite_mask] if use_own_rgb_source else None,
                rgb_z_values=rgb_z_values[finite_mask] if use_own_rgb_source and is_3d_embedding else None,
                path_metric_coords=np.asarray(axis_gradients, dtype=float)[finite_mask],
                point_labels=point_labels[finite_mask],
                point_ids=scatter_projection_labels,
                metabolite_profiles=(
                    None
                    if metabolite_profiles is None
                    else (
                        np.asarray(metabolite_profiles, dtype=float)[finite_mask]
                        if np.asarray(metabolite_profiles, dtype=float).ndim == 2
                        else np.asarray(metabolite_profiles, dtype=float)[:, finite_mask, :]
                    )
                ),
                metabolite_names=metabolite_names,
                metabolite_subject_labels=metabolite_subject_labels,
                title=scatter_title,
                x_label=x_label,
                y_label=y_label,
                z_label=z_label,
                color_label=f"Gradient {class_component + 1}",
                cmap=classification_cmap,
                cmap_name=classification_cmap_name,
                theme_name=self._theme_name,
                rotation_preset=scatter_rotation,
                use_triangular_rgb=effective_use_triangular_rgb,
                rgb_fit_mode=classification_fit_mode,
                triangular_color_order=triangular_color_order,
                rgb_scalar_mode=rgb_scalar_mode,
                edge_pairs=None if is_3d_embedding else edge_pairs,
                point_group_codes=scatter_hemisphere_codes,
                hemisphere_mode=hemisphere_mode,
                show_proximity_circles=False,
                initial_proximity_slider_value=1000,
                auto_preload_matching_paths=preload_matching_paths,
                project_paths_callback=None if is_3d_embedding else _project_paths_callback,
                export_metadata=scatter_export_metadata,
                parent=self,
            )
            self._gradient_scatter_dialog.resize(960 if is_3d_embedding else 860, 780 if is_3d_embedding else 760)
            self._gradient_scatter_dialog.show()
        except Exception as exc:
            scatter_error = str(exc)

        try:
            support_mask_data = self._last_gradients.get("support_mask")
            if np.any(ignore_mask):
                ignored_labels = np.asarray(projection_labels[ignore_mask], dtype=int)
                if ignored_labels.size > 0:
                    _template_img, template_data = self._gradient_template_img_and_data()
                    ignored_voxels = np.isin(np.asarray(template_data, dtype=int), ignored_labels)
                    x_volume = np.asarray(x_volume, dtype=float).copy()
                    y_volume = np.asarray(y_volume, dtype=float).copy()
                    x_volume[ignored_voxels] = np.nan
                    y_volume[ignored_voxels] = np.nan
                    if is_3d_embedding:
                        z_volume = np.asarray(z_volume, dtype=float).copy()
                        z_volume[ignored_voxels] = np.nan
                    if support_mask_data is not None:
                        support_mask_data = np.asarray(support_mask_data, dtype=float).copy()
                        support_mask_data[ignored_voxels] = 0.0

            if projected_data.ndim == 4:
                classification_projection = np.asarray(projected_data[..., class_component], dtype=float)
            else:
                classification_projection = np.asarray(projected_data, dtype=float)
            if np.any(ignore_mask):
                ignored_labels = np.asarray(projection_labels[ignore_mask], dtype=int)
                if ignored_labels.size > 0:
                    _template_img, template_data = self._gradient_template_img_and_data()
                    ignored_voxels = np.isin(np.asarray(template_data, dtype=int), ignored_labels)
                    classification_projection = np.asarray(classification_projection, dtype=float).copy()
                    classification_projection[ignored_voxels] = np.nan

            if effective_use_triangular_rgb:
                self._gradient_classification_dialog = GradientClassificationDialog.from_array(
                    x_volume,
                    y_volume,
                    z_volume_data=z_volume if is_3d_embedding else None,
                    affine=self._last_gradients.get("affine"),
                    x_values=x_values[finite_mask],
                    y_values=y_values[finite_mask],
                    z_values=z_values[finite_mask] if is_3d_embedding else None,
                    support_mask_data=support_mask_data,
                    coverage_mask_data=classification_projection,
                    coverage_threshold_multiplier=rgb_mask_strictness,
                    rgb_x_volume_data=rgb_x_volume if use_own_rgb_source else None,
                    rgb_y_volume_data=rgb_y_volume if use_own_rgb_source else None,
                    rgb_z_volume_data=rgb_z_volume if use_own_rgb_source and is_3d_embedding else None,
                    rgb_x_values=rgb_x_values[finite_mask] if use_own_rgb_source else None,
                    rgb_y_values=rgb_y_values[finite_mask] if use_own_rgb_source else None,
                    rgb_z_values=rgb_z_values[finite_mask] if use_own_rgb_source and is_3d_embedding else None,
                    title=(
                        f"{z_label} vs {y_label} vs {x_label} Classification - {source_name}"
                        if is_3d_embedding
                        else f"{y_label} vs {x_label} Classification - {source_name}"
                    ),
                    x_label=x_label,
                    y_label=y_label,
                    z_label=z_label,
                    theme_name=self._theme_name,
                    hemisphere_mode=surface_hemisphere_mode,
                    fsaverage_mesh=surface_mesh,
                    rotation_preset=scatter_rotation,
                    rgb_fit_mode=classification_fit_mode,
                    triangular_color_order=triangular_color_order,
                    parent=self,
                )
            else:
                self._gradient_classification_dialog = GradientSurfaceDialog.from_array(
                    classification_projection,
                    affine=self._last_gradients.get("affine"),
                    title=f"Gradient {class_component + 1} Classification - {source_name}",
                    cmap=classification_cmap,
                    cmap_name=classification_cmap_name,
                    theme_name=self._theme_name,
                    hemisphere_mode=surface_hemisphere_mode,
                    fsaverage_mesh=surface_mesh,
                    parent=self,
                )
            screen = QApplication.primaryScreen()
            if screen is not None:
                geom = screen.availableGeometry()
                width = max(int(geom.width() * 0.9), 1200)
                height = min(max(int(geom.height() * 0.82), 760), int(geom.height() * 0.95))
            else:
                width = 1500
                height = 900
            self._gradient_classification_dialog.resize(width, height)
            self._gradient_classification_dialog.show()
        except Exception as exc:
            surface_error = str(exc)

        adjacency_suffix = ""
        if adjacency_note:
            note_text = str(adjacency_note).strip()
            if note_text:
                if note_text[-1] not in ".!?":
                    note_text += "."
                adjacency_suffix = f" {note_text}"

        if scatter_error and surface_error:
            self.statusBar().showMessage(
                f"Failed to open classification scatter and fsaverage windows: {scatter_error}; {surface_error}{adjacency_suffix}"
            )
            return
        if scatter_error:
            self.statusBar().showMessage(
                f"Opened classified fsaverage viewer, but the scatter window failed: {scatter_error}{adjacency_suffix}"
            )
            return
        if surface_error:
            self.statusBar().showMessage(
                f"Opened classification scatter, but the fsaverage window failed: {surface_error}{adjacency_suffix}"
            )
            return
        if effective_use_triangular_rgb:
            self.statusBar().showMessage(
                f"Opened classification scatter and fsaverage viewers.{adjacency_suffix}"
            )
        else:
            self.statusBar().showMessage(
                f"Opened classification scatter and fsaverage viewers using Gradient {class_component + 1}.{adjacency_suffix}"
            )

    def _project_classification_paths_to_fibrenet(
        self,
        project_payload,
        group_payloads,
        scatter_projection_labels,
        scatter_point_labels,
        point_colors,
        *,
        source_name="matrix",
        x_label="Gradient 2",
        y_label="Gradient 1",
    ) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        from mrsitoolbox.connectomics.netfibre import NetFibre

        scatter_projection_labels = np.asarray(scatter_projection_labels, dtype=int).reshape(-1)
        scatter_point_labels = np.asarray(scatter_point_labels, dtype=object).reshape(-1)
        point_colors = np.asarray(point_colors, dtype=float)
        if point_colors.ndim != 2 or point_colors.shape[0] != scatter_projection_labels.size or point_colors.shape[1] < 3:
            point_colors = np.full((scatter_projection_labels.size, 3), 0.45, dtype=float)
        else:
            point_colors = np.asarray(point_colors[:, :3], dtype=float)
        point_colors = np.clip(np.nan_to_num(point_colors, nan=0.45, posinf=1.0, neginf=0.0), 0.0, 1.0)
        layout_name = str(project_payload.get("fibrenet_layout", "diffusion") or "diffusion").strip().lower()
        if layout_name in {"spiral", "path_spiral", "sequence", "sequence_spiral"}:
            layout_name = "spiral"
            layout_label = "Spiral"
        else:
            layout_name = "diffusion"
            layout_label = "Diffusion"
        edge_pairs = np.asarray(project_payload.get("edge_pairs", []), dtype=int)
        if (
            layout_name == "diffusion"
            and (edge_pairs.ndim != 2 or edge_pairs.shape[1] != 2 or edge_pairs.shape[0] == 0)
        ):
            self.statusBar().showMessage("No GM adjacency edge pairs are available for FibreNet projection.")
            return
        try:
            node_count = int(project_payload.get("node_count", scatter_projection_labels.size))
        except Exception:
            node_count = int(scatter_projection_labels.size)
        node_count = max(node_count, int(scatter_projection_labels.size))

        def _group_title(group_name):
            text = str(group_name or "all").strip().lower()
            if text == "lh":
                return "LH"
            if text == "rh":
                return "RH"
            return "All"

        def _coerce_path(path_value):
            nodes = []
            for node in list(path_value or []):
                try:
                    nodes.append(int(node))
                except Exception:
                    continue
            return nodes

        def _node_label(node_index):
            label = ""
            if 0 <= int(node_index) < scatter_point_labels.shape[0]:
                label = str(scatter_point_labels[int(node_index)]).strip()
            if label and label.lower() not in {"nan", "none"}:
                return label
            if 0 <= int(node_index) < scatter_projection_labels.shape[0]:
                return f"Parcel {int(scatter_projection_labels[int(node_index)])}"
            return f"Node {int(node_index)}"

        def _add_rgb_triangle_legend(fig, payload, *, x_caption, y_caption):
            color_order = str(payload.get("color_order", "RGB") or "RGB").strip().upper()
            if len(color_order) < 3:
                color_order = "RGB"
            color_basis = {
                "R": np.asarray((1.0, 0.0, 0.0), dtype=float),
                "G": np.asarray((0.0, 1.0, 0.0), dtype=float),
                "B": np.asarray((0.0, 0.0, 1.0), dtype=float),
            }
            vertices = np.asarray(
                [
                    (0.5, np.sqrt(3.0) / 2.0),
                    (0.0, 0.0),
                    (1.0, 0.0),
                ],
                dtype=float,
            )
            vertex_colors = np.asarray([color_basis.get(channel, color_basis["R"]) for channel in color_order[:3]], dtype=float)
            width = 160
            height = 140
            yy, xx = np.mgrid[0:1:complex(height), 0:1:complex(width)]
            points = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
            tri = vertices
            denom = (
                (tri[1, 1] - tri[2, 1]) * (tri[0, 0] - tri[2, 0])
                + (tri[2, 0] - tri[1, 0]) * (tri[0, 1] - tri[2, 1])
            )
            if abs(float(denom)) <= 1e-12:
                return
            w0 = (
                (tri[1, 1] - tri[2, 1]) * (points[:, 0] - tri[2, 0])
                + (tri[2, 0] - tri[1, 0]) * (points[:, 1] - tri[2, 1])
            ) / denom
            w1 = (
                (tri[2, 1] - tri[0, 1]) * (points[:, 0] - tri[2, 0])
                + (tri[0, 0] - tri[2, 0]) * (points[:, 1] - tri[2, 1])
            ) / denom
            w2 = 1.0 - w0 - w1
            weights = np.column_stack((w0, w1, w2))
            inside = np.all(weights >= -1e-6, axis=1)
            image = np.ones((points.shape[0], 4), dtype=float)
            image[:, :3] = 1.0
            image[:, 3] = 0.0
            if np.any(inside):
                rgb = np.clip(weights[inside] @ vertex_colors, 0.0, 1.0)
                image[inside, :3] = rgb
                image[inside, 3] = 1.0
            image = image.reshape(height, width, 4)
            legend_ax = fig.add_axes([0.915, 0.14, 0.07, 0.20], facecolor="white")
            legend_ax.imshow(image, origin="lower", extent=(0.0, 1.0, 0.0, 1.0), interpolation="bilinear")
            outline = Polygon(vertices, closed=True, facecolor="none", edgecolor="black", linewidth=1.0)
            legend_ax.add_patch(outline)
            text_offsets = np.asarray(((0.0, 0.05), (-0.05, -0.06), (0.05, -0.06)), dtype=float)
            for idx, channel in enumerate(color_order[:3]):
                vx, vy = vertices[idx]
                ox, oy = text_offsets[idx]
                legend_ax.text(
                    float(vx + ox),
                    float(vy + oy),
                    str(channel),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="black",
                    weight="bold",
                )
            legend_ax.text(0.5, -0.16, str(x_caption), ha="center", va="top", fontsize=8, color="black")
            legend_ax.text(-0.10, 0.5, str(y_caption), ha="right", va="center", fontsize=8, color="black", rotation=90)
            legend_ax.set_xlim(-0.12, 1.12)
            legend_ax.set_ylim(-0.18, 1.02)
            legend_ax.set_xticks([])
            legend_ax.set_yticks([])
            legend_ax.set_frame_on(False)

        def _selected_path_record(group_payload):
            group = dict(group_payload or {})
            nodes = _coerce_path(group.get("selected_ctx_path", []))
            path_kind = "path"
            subc_nodes = _coerce_path(group.get("subc_optimal_path", []))
            if len(nodes) < 2:
                nodes = _coerce_path(group.get("optimal_full_path", []))
            if len(nodes) < 2:
                nodes = list(subc_nodes)
                subc_nodes = []
                path_kind = "subcortical path"
            if len(nodes) < 2:
                return None
            if any(node < 0 or node >= scatter_projection_labels.size for node in nodes):
                return None
            if any(node < 0 or node >= scatter_projection_labels.size for node in subc_nodes):
                subc_nodes = []
            try:
                rank = int(group.get("selected_ctx_path_index", 0))
            except Exception:
                rank = 0
            energy = None
            try:
                value = float(group.get("selected_ctx_path_energy"))
                if np.isfinite(value):
                    energy = value
            except Exception:
                pass
            if energy is None:
                try:
                    value = float(group.get("ctx_optimal_path_energy"))
                    if np.isfinite(value):
                        energy = value
                except Exception:
                    pass
            title = f"{_group_title(group.get('group', 'all'))} {path_kind} {rank}"
            if energy is not None:
                title += f" | E={energy:.4f}"
            return {
                "group": str(group.get("group", "all")),
                "nodes": nodes,
                "subc_nodes": subc_nodes if len(subc_nodes) >= 2 else [],
                "subc_color": group.get("subc_color", []),
                "title": title,
            }

        records = []
        for group_payload in list(group_payloads or []):
            record = _selected_path_record(group_payload)
            if record is not None:
                records.append(record)
        if not records:
            record = _selected_path_record(project_payload)
            if record is not None:
                records.append(record)
        if not records:
            self.statusBar().showMessage("No selected path is available for FibreNet projection.")
            return

        panel_count = len(records)
        max_nodes = max(len(record["nodes"]) for record in records)
        fig_width = max(8.5, min(18.0, 7.2 * panel_count))
        fig_height = max(8.0, min(13.0, 7.0 + 0.08 * max_nodes))
        fig = plt.figure(figsize=(fig_width, fig_height), facecolor="white")
        fig.suptitle(
            f"FibreNet {layout_label} path projection - {source_name} ({y_label} vs {x_label})",
            color="black",
            fontsize=13,
        )

        for panel_idx, record in enumerate(records, start=1):
            nodes = [int(node) for node in record["nodes"]]
            node_names = [_node_label(node) for node in nodes]
            node_colors = [tuple(point_colors[node, :3].tolist()) for node in nodes]
            branch_paths = []
            branch_names = []
            branch_colors = []
            subc_nodes = [int(node) for node in list(record.get("subc_nodes", []))]
            if layout_name == "spiral" and len(subc_nodes) >= 2:
                branch_paths.append(subc_nodes)
                branch_names.append([_node_label(node) for node in subc_nodes])
                branch_colors.append([tuple(point_colors[node, :3].tolist()) for node in subc_nodes])
            fontsize_names = 7 if len(nodes) > 30 else 8 if len(nodes) > 18 else 9
            ax = fig.add_subplot(1, panel_count, panel_idx)
            NetFibre.plot_fibrenet_path(
                nodes,
                edge_pairs=edge_pairs,
                n_nodes=node_count,
                node_names=node_names,
                node_colors=node_colors,
                ax=ax,
                title=str(record["title"]),
                components=(2, 3),
                layout=layout_name,
                edge_color="black",
                edge_width=2.4,
                node_size=260,
                font_size=fontsize_names,
                branch_paths=branch_paths,
                branch_node_names=branch_names,
                branch_node_colors=branch_colors,
                branch_edge_color="black",
                branch_edge_width=1.7,
                branch_edge_linestyle="--",
                branch_node_size=150,
                endpoint_node_size=450 if layout_name == "spiral" else None,
            )

        _add_rgb_triangle_legend(fig, project_payload, x_caption=x_label, y_caption=y_label)
        fig.subplots_adjust(left=0.03, right=0.90, top=0.88, bottom=0.04, wspace=0.32)
        manager = getattr(fig.canvas, "manager", None)
        if manager is not None and hasattr(manager, "set_window_title"):
            manager.set_window_title(f"FibreNet paths - {source_name}")
        fig.canvas.draw_idle()
        backend_name = str(plt.get_backend()).strip().lower()
        if backend_name not in {"agg", "pdf", "svg", "ps", "template"}:
            fig.show()
            plt.show(block=False)
        self._gradient_path_fibrenet_figures = getattr(self, "_gradient_path_fibrenet_figures", [])
        self._gradient_path_fibrenet_figures.append(fig)
        groups = ", ".join(_group_title(record.get("group")) for record in records)
        self.statusBar().showMessage(
            f"Opened FibreNet {layout_label.lower()} path projection for {groups} on {source_name}."
        )

    def _project_classification_paths_to_brain(
        self,
        payload,
        scatter_projection_labels,
        scatter_point_labels,
        *,
        hemisphere_mode="both",
        source_name="matrix",
        x_label="Gradient 2",
        y_label="Gradient 1",
    ) -> None:
        project_payload = dict(payload or {})
        save_three_view_screenshot = bool(project_payload.get("glassbrain_screenshot", False))
        group_payloads = []
        for group_payload in list(project_payload.get("group_paths", [])):
            group_dict = dict(group_payload or {})
            selected_path = list(
                group_dict.get("selected_ctx_path", group_dict.get("optimal_full_path", []))
            )
            subc_path = list(group_dict.get("subc_optimal_path", []))
            if len(selected_path) >= 2 or len(subc_path) >= 2:
                group_payloads.append(group_dict)
        optimal_full_path = [int(node) for node in list(project_payload.get("optimal_full_path", []))]
        if (
            not group_payloads
            and len(optimal_full_path) < 2
            and not any(len(list(dict(group).get("subc_optimal_path", []))) >= 2 for group in list(project_payload.get("group_paths", [])))
        ):
            self.statusBar().showMessage("No complete ordered path is available at the current slider radius.")
            return

        scatter_projection_labels = np.asarray(scatter_projection_labels, dtype=int).reshape(-1)
        scatter_point_labels = np.asarray(scatter_point_labels, dtype=object).reshape(-1)
        if scatter_projection_labels.size == 0:
            self.statusBar().showMessage("No classification parcels are available for 3D path projection.")
            return

        if np.any((np.asarray(optimal_full_path, dtype=int) < 0) | (np.asarray(optimal_full_path, dtype=int) >= scatter_projection_labels.size)):
            self.statusBar().showMessage("The selected scatter path is out of range for the current parcels.")
            return

        point_colors = np.asarray(project_payload.get("point_colors", []), dtype=float)
        if point_colors.ndim != 2 or point_colors.shape[0] != scatter_projection_labels.size or point_colors.shape[1] < 3:
            point_colors = np.full((scatter_projection_labels.size, 3), 0.35, dtype=float)
        else:
            point_colors = np.asarray(point_colors[:, :3], dtype=float)
        point_colors = np.clip(np.nan_to_num(point_colors, nan=0.35, posinf=1.0, neginf=0.0), 0.0, 1.0)

        if bool(project_payload.get("fibrenet_projection", False)):
            try:
                self._project_classification_paths_to_fibrenet(
                    project_payload,
                    group_payloads,
                    scatter_projection_labels,
                    scatter_point_labels,
                    point_colors,
                    source_name=source_name,
                    x_label=x_label,
                    y_label=y_label,
                )
            except Exception as exc:
                self.statusBar().showMessage(f"Failed to project the selected path set to FibreNet: {exc}")
            return

        try:
            template_img, _template_data = self._gradient_template_img_and_data()
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to load the parcellation template for 3D projection: {exc}")
            return

        from mrsitoolbox.graphplot import glassbrain

        try:
            centroids_mni = np.asarray(
                nettools.compute_centroids(
                    template_img,
                    labels=np.asarray(scatter_projection_labels, dtype=int),
                    world=True,
                ),
                dtype=float,
            )
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to compute parcel centroids for 3D projection: {exc}")
            return

        if centroids_mni.shape != (scatter_projection_labels.size, 3):
            self.statusBar().showMessage("Centroid count mismatch for 3D path projection.")
            return

        def _path_coords(path_nodes):
            node_indices = np.asarray([int(node) for node in list(path_nodes or [])], dtype=int)
            if node_indices.size < 2:
                return None
            if np.any((node_indices < 0) | (node_indices >= centroids_mni.shape[0])):
                return None
            coords = np.asarray(centroids_mni[node_indices, :], dtype=float)
            if coords.ndim != 2 or coords.shape[1] != 3:
                return None
            if not np.all(np.isfinite(coords)):
                return None
            return coords

        def _dashed_path_segments(path_coords, dash_length=GLASSBRAIN_SUBC_DASH_LENGTH, gap_length=GLASSBRAIN_SUBC_DASH_GAP):
            coords = np.asarray(path_coords, dtype=float)
            if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] < 2:
                return []
            deltas = np.diff(coords, axis=0)
            segment_lengths = np.linalg.norm(deltas, axis=1)
            valid_lengths = np.isfinite(segment_lengths) & (segment_lengths > 1e-9)
            if not np.any(valid_lengths):
                return []
            cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
            total_length = float(cumulative[-1])
            if not np.isfinite(total_length) or total_length <= 1e-9:
                return []
            dash = max(1e-6, float(dash_length))
            gap = max(0.0, float(gap_length))
            period = dash + gap if gap > 0.0 else dash

            def _point_at(distance):
                value = float(np.clip(distance, 0.0, total_length))
                if value <= 0.0:
                    return np.asarray(coords[0], dtype=float)
                if value >= total_length:
                    return np.asarray(coords[-1], dtype=float)
                seg_idx = int(np.searchsorted(cumulative, value, side="right") - 1)
                seg_idx = int(np.clip(seg_idx, 0, segment_lengths.shape[0] - 1))
                length = float(segment_lengths[seg_idx])
                if length <= 1e-9:
                    return np.asarray(coords[seg_idx], dtype=float)
                frac = (value - float(cumulative[seg_idx])) / length
                return np.asarray(coords[seg_idx] + frac * (coords[seg_idx + 1] - coords[seg_idx]), dtype=float)

            segments = []
            start = 0.0
            while start < total_length - 1e-9:
                stop = min(start + dash, total_length)
                points = [_point_at(start)]
                inner_indices = np.where((cumulative > start + 1e-9) & (cumulative < stop - 1e-9))[0]
                for idx in inner_indices.tolist():
                    points.append(np.asarray(coords[int(idx)], dtype=float))
                points.append(_point_at(stop))
                segment = np.asarray(points, dtype=float)
                if segment.shape[0] >= 2 and np.linalg.norm(segment[-1] - segment[0]) > 1e-9:
                    segments.append(segment)
                start += period
            return segments

        def _rgb_basis_color(channel):
            mapping = {
                "R": np.asarray((1.0, 0.0, 0.0), dtype=float),
                "G": np.asarray((0.0, 1.0, 0.0), dtype=float),
                "B": np.asarray((0.0, 0.0, 1.0), dtype=float),
            }
            return np.asarray(mapping.get(str(channel).strip().upper(), (0.5, 0.5, 0.5)), dtype=float)

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
                0.5 * (_rgb_basis_color(first) + _rgb_basis_color(second)),
                0.0,
                1.0,
            )

        channel_order = [
            str(channel)
            for channel in str(project_payload.get("channel_order", "")).strip()
        ]
        energy_scaling = dict(project_payload.get("energy_width_scaling", {}))
        try:
            base_edge_width = max(0.05, float(project_payload.get("edge_linewidth", 0.45)))
        except Exception:
            base_edge_width = 0.45
        width_mode = str(project_payload.get("width_scaling_mode", "exp")).strip().lower()
        if width_mode not in {"exp", "linear", "log"}:
            width_mode = "exp"
        try:
            width_strength = max(0.05, float(project_payload.get("width_scaling_strength", 2.0)))
        except Exception:
            width_strength = 2.0

        def _path_width_from_energy(energy, family_type):
            scaling_value = energy_scaling.get(str(family_type), {})
            scaling = dict(scaling_value) if isinstance(scaling_value, dict) else {}
            default_width = max(1.2, base_edge_width * 6.0)
            if not scaling:
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
            if width_mode == "linear":
                mapped = norm
            elif width_mode == "log":
                mapped = float(np.log1p(width_strength * norm) / np.log1p(width_strength))
            else:
                denominator = float(np.expm1(width_strength))
                if np.isclose(denominator, 0.0):
                    mapped = norm
                else:
                    mapped = float(np.expm1(width_strength * norm) / denominator)
            return max(default_width * 0.7, base_edge_width * (4.5 + 4.5 * mapped))

        def _lookup_path_energy(group_payload, family_key, path_nodes, fallback_key=None):
            nodes_key = tuple(int(node) for node in list(path_nodes or []))
            if len(nodes_key) < 2:
                return None
            candidate_paths = list(group_payload.get(family_key, []))
            candidate_energies = np.asarray(
                group_payload.get(
                    "ctx_path_energies" if family_key == "all_full_paths" else "subc_path_energies",
                    [],
                ),
                dtype=float,
            ).reshape(-1)
            for idx, candidate in enumerate(candidate_paths):
                candidate_key = tuple(int(node) for node in list(candidate or []))
                if candidate_key == nodes_key and idx < candidate_energies.size and np.isfinite(candidate_energies[idx]):
                    return float(candidate_energies[idx])
            if fallback_key is not None:
                try:
                    fallback = group_payload.get(fallback_key)
                    if fallback is not None and np.isfinite(float(fallback)):
                        return float(fallback)
                except Exception:
                    return None
            return None

        def _ctx_segment_records(group_payload, full_path_nodes):
            nodes = [int(node) for node in list(full_path_nodes or [])]
            anchors = {str(key): int(value) for key, value in dict(group_payload.get("anchors", {})).items()}
            order = [str(channel) for channel in channel_order if str(channel) in anchors]
            if len(nodes) < 2 or len(order) < 2:
                return []

            def _record(first, second, segment_nodes):
                color = _pair_channel_color(first, second)
                return {
                    "nodes": [int(node) for node in list(segment_nodes or [])],
                    "color": color,
                }

            if len(order) == 2:
                return [_record(order[0], order[1], nodes)]

            middle_anchor = int(anchors[order[1]])
            try:
                split_index = next(
                    idx
                    for idx, node in enumerate(nodes)
                    if int(node) == middle_anchor and 0 < idx < len(nodes) - 1
                )
            except StopIteration:
                return [_record(order[0], order[-1], nodes)]
            return [
                _record(order[0], order[1], nodes[: split_index + 1]),
                _record(order[1], order[2], nodes[split_index:]),
            ]

        draw_paths = []
        draw_colors = []
        draw_widths = []
        if not group_payloads and len(optimal_full_path) >= 2:
            group_payloads = [project_payload]

        anchor_channels = ("R", "G", "B")
        anchor_centroids = []
        anchor_node_colors = []
        anchor_node_labels = []
        seen_anchor_indices = set()
        union_nodes = set()
        endpoint_nodes = set()
        intermediate_nodes = set()
        show_all_paths = bool(project_payload.get("show_all_ordered_paths"))
        for group_payload in group_payloads:
            seen_paths = set()
            ctx_paths = (
                list(group_payload.get("all_full_paths", []))
                if show_all_paths
                else [group_payload.get("selected_ctx_path", group_payload.get("optimal_full_path", []))]
            )
            for path_nodes in ctx_paths:
                nodes = tuple(int(node) for node in list(path_nodes or []))
                if len(nodes) < 2:
                    continue
                width = _lookup_path_energy(
                    group_payload,
                    "all_full_paths",
                    nodes,
                    fallback_key="ctx_optimal_path_energy",
                )
                path_width = _path_width_from_energy(width, "ctx")
                for record in _ctx_segment_records(group_payload, nodes):
                    segment_nodes = tuple(int(node) for node in list(record.get("nodes", [])))
                    if len(segment_nodes) < 2:
                        continue
                    coords = _path_coords(segment_nodes)
                    if coords is None:
                        continue
                    seen_paths.add(segment_nodes)
                    draw_paths.append(coords)
                    draw_colors.append(np.asarray(record.get("color", (0.0, 0.0, 0.0)), dtype=float).reshape(3))
                    draw_widths.append(path_width)
                union_nodes.update(int(node) for node in nodes)
                endpoint_nodes.update((int(nodes[0]), int(nodes[-1])))
                intermediate_nodes.update(int(node) for node in nodes[1:-1])

            subc_color = np.asarray(group_payload.get("subc_color", (0.0, 0.0, 0.0)), dtype=float).reshape(-1)
            if subc_color.shape != (3,):
                subc_color = np.asarray((0.0, 0.0, 0.0), dtype=float)
            subc_paths = list(group_payload.get("subc_paths", [])) if show_all_paths else [group_payload.get("subc_optimal_path", [])]
            for path_nodes in subc_paths:
                nodes = tuple(int(node) for node in list(path_nodes or []))
                if len(nodes) < 2 or nodes in seen_paths:
                    continue
                coords = _path_coords(nodes)
                if coords is None:
                    continue
                seen_paths.add(nodes)
                subc_width = _path_width_from_energy(
                    _lookup_path_energy(
                        group_payload,
                        "subc_paths",
                        nodes,
                        fallback_key="subc_optimal_path_energy",
                    ),
                    "subc",
                )
                dashed_segments = _dashed_path_segments(coords)
                if not dashed_segments:
                    dashed_segments = [coords]
                for dashed_coords in dashed_segments:
                    draw_paths.append(dashed_coords)
                    draw_colors.append(np.asarray(subc_color, dtype=float))
                    draw_widths.append(subc_width)
                union_nodes.update(int(node) for node in nodes)
                endpoint_nodes.update((int(nodes[0]), int(nodes[-1])))
                intermediate_nodes.update(int(node) for node in nodes[1:-1])

            anchors = dict(group_payload.get("anchors", {}))
            for channel in anchor_channels:
                anchor_idx = anchors.get(channel)
                if anchor_idx is None:
                    continue
                anchor_idx = int(anchor_idx)
                if anchor_idx in seen_anchor_indices or not (0 <= anchor_idx < centroids_mni.shape[0]):
                    continue
                seen_anchor_indices.add(anchor_idx)
                anchor_centroids.append(centroids_mni[anchor_idx])
                if anchor_idx < point_colors.shape[0]:
                    anchor_node_colors.append(point_colors[anchor_idx])
                else:
                    anchor_node_colors.append((0.0, 0.0, 0.0))
                label_text = str(scatter_point_labels[anchor_idx]).strip() if anchor_idx < scatter_point_labels.shape[0] else ""
                anchor_node_labels.append(label_text or f"Parcel {int(scatter_projection_labels[anchor_idx])}")
            subc_anchor = group_payload.get("subc_anchor")
            if subc_anchor is not None:
                subc_anchor = int(subc_anchor)
                if subc_anchor not in seen_anchor_indices and 0 <= subc_anchor < centroids_mni.shape[0]:
                    seen_anchor_indices.add(subc_anchor)
                    anchor_centroids.append(centroids_mni[subc_anchor])
                    if subc_anchor < point_colors.shape[0]:
                        anchor_node_colors.append(point_colors[subc_anchor])
                    else:
                        anchor_node_colors.append((0.0, 0.0, 0.0))
                    label_text = str(scatter_point_labels[subc_anchor]).strip() if subc_anchor < scatter_point_labels.shape[0] else ""
                    anchor_node_labels.append(label_text or f"Parcel {int(scatter_projection_labels[subc_anchor])}")

        if not draw_paths:
            self.statusBar().showMessage("The selected ordered path cannot be projected to the 3D brain.")
            return

        path_node_indices = np.asarray(sorted(int(node) for node in union_nodes), dtype=int)
        if path_node_indices.size:
            path_node_indices = path_node_indices[
                (path_node_indices >= 0) & (path_node_indices < centroids_mni.shape[0])
            ]
        try:
            node_colors = np.empty((centroids_mni.shape[0], 4), dtype=float)
            node_colors[:, :3] = np.asarray((0.6, 0.6, 0.6), dtype=float)
            node_colors[:, 3] = 0.18
            node_sizes = np.full(centroids_mni.shape[0], 1.45, dtype=float)
            if path_node_indices.size:
                node_colors[path_node_indices, :3] = np.asarray(point_colors[path_node_indices, :], dtype=float)
                node_colors[path_node_indices, 3] = 0.95
            intermediate_indices = np.asarray(
                sorted(int(node) for node in (intermediate_nodes - endpoint_nodes)),
                dtype=int,
            )
            intermediate_indices = intermediate_indices[
                (intermediate_indices >= 0) & (intermediate_indices < centroids_mni.shape[0])
            ]
            endpoint_indices = np.asarray(sorted(int(node) for node in endpoint_nodes), dtype=int)
            endpoint_indices = endpoint_indices[
                (endpoint_indices >= 0) & (endpoint_indices < centroids_mni.shape[0])
            ]
            if intermediate_indices.size:
                node_sizes[intermediate_indices] = 2.35
            if endpoint_indices.size:
                node_sizes[endpoint_indices] = 3.05
            normalized_hemisphere_mode = self._normalize_gradient_hemisphere_mode(hemisphere_mode)
            brain_hemi = normalized_hemisphere_mode if normalized_hemisphere_mode in {"lh", "rh"} else "both"
            viewer = glassbrain.GlassBrainPlotter(
                brain_opacity=0.12,
                background="white",
                window_size=(1200, 900),
                path_radius_scale=GLASSBRAIN_EDGE_RADIUS_SCALE,
            )
            title = f"Gradient paths - {source_name}"
            glassbrain_path_widths = np.asarray(draw_widths, dtype=float)
            if glassbrain_path_widths.size:
                glassbrain_path_widths = np.maximum(
                    glassbrain_path_widths * float(GLASSBRAIN_EDGE_WIDTH_MULTIPLIER),
                    float(GLASSBRAIN_EDGE_MIN_WIDTH),
                )
            glassbrain_path_colors = np.tile(
                np.asarray(GLASSBRAIN_EDGE_COLOR, dtype=float).reshape(1, 3),
                (len(draw_paths), 1),
            )
            render_kwargs = {
                "node_coords": centroids_mni,
                "paths": draw_paths,
                "path_colors": glassbrain_path_colors,
                "path_widths": glassbrain_path_widths,
                "node_colors": node_colors,
                "node_sizes": node_sizes,
                "anchor_coords": np.asarray(anchor_centroids, dtype=float) if anchor_centroids else None,
                "anchor_colors": np.asarray(anchor_node_colors, dtype=float) if anchor_node_colors else None,
                "anchor_labels": anchor_node_labels,
                "anchor_size": 3.45,
                "hemisphere_mode": brain_hemi,
                "title": title,
            }
            if save_three_view_screenshot:
                safe_source = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(source_name or "gradient_paths")).strip("_")
                default_name = f"{safe_source or 'gradient_paths'}_glassbrain_three_views.png"
                output_path, _selected_filter = QFileDialog.getSaveFileName(
                    self,
                    "Save glass-brain three-view screenshot",
                    default_name,
                    "PNG Image (*.png);;PDF Document (*.pdf);;SVG Vector (*.svg)",
                )
                if not output_path:
                    self.statusBar().showMessage("Glass-brain screenshot cancelled.")
                    return
                output_path = str(output_path)
                if Path(output_path).suffix.lower() not in {".png", ".pdf", ".svg"}:
                    output_path = str(Path(output_path).with_suffix(".png"))
                screenshot_kwargs = dict(render_kwargs)
                screenshot_kwargs["title"] = None
                screenshot_kwargs["anchor_labels"] = []
                viewer.save_three_view_panel(
                    output_path,
                    views=("dorsal", "coronal", "lateral"),
                    show_titles=False,
                    **screenshot_kwargs,
                )
                self.statusBar().showMessage(f"Saved glass-brain three-view screenshot to {output_path}.")
                return
            viewer.show_paths(
                **render_kwargs,
            )
            self._gradient_path_glassbrain = viewer
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to project the selected path set to the PyVista glass brain: {exc}")
            return

        path_label = " -> ".join(list(project_payload.get("channel_order", "RBG")))
        self.statusBar().showMessage(
            f"Opened PyVista glass-brain path projection for {path_label} on {source_name} ({y_label} vs {x_label})."
        )

    @staticmethod
    def _gradient_rotation_angles(preset: str):
        mapping = {
            "Default": (0.0, 0.0, 0.0),
            "X +90": (90.0, 0.0, 0.0),
            "X -90": (-90.0, 0.0, 0.0),
            "Y +90": (0.0, 90.0, 0.0),
            "Y -90": (0.0, -90.0, 0.0),
            "Y 180": (0.0, 180.0, 0.0),
            "Z +90": (0.0, 0.0, 90.0),
            "Z -90": (0.0, 0.0, -90.0),
        }
        return mapping.get(str(preset or "Default"), (0.0, 0.0, 0.0))

    @staticmethod
    def _actor_bounds_center(actor_obj):
        bounds = actor_obj.GetBounds() if actor_obj is not None else None
        if not bounds or len(bounds) != 6:
            return None
        if not np.all(np.isfinite(bounds)):
            return None
        return (
            float((bounds[0] + bounds[1]) * 0.5),
            float((bounds[2] + bounds[3]) * 0.5),
            float((bounds[4] + bounds[5]) * 0.5),
        )

    def _apply_gradient_component_rotation(self, actors, preset: str, fallback_center) -> None:
        rx, ry, rz = self._gradient_rotation_angles(preset)
        if np.isclose(rx, 0.0) and np.isclose(ry, 0.0) and np.isclose(rz, 0.0):
            return
        origin = None
        for actor_obj in actors:
            origin = self._actor_bounds_center(actor_obj)
            if origin is not None:
                break
        if origin is None:
            center = np.asarray(fallback_center, dtype=float).reshape(3)
            origin = (float(center[0]), float(center[1]), float(center[2]))
        for actor_obj in actors:
            if actor_obj is None:
                continue
            actor_obj.SetOrigin(*origin)
            if not np.isclose(rx, 0.0):
                actor_obj.RotateX(rx)
            if not np.isclose(ry, 0.0):
                actor_obj.RotateY(ry)
            if not np.isclose(rz, 0.0):
                actor_obj.RotateZ(rz)

    def _render_gradients_network(self) -> None:
        if not self._last_gradients:
            self.statusBar().showMessage("No gradients computed yet.")
            return

        gradients = np.asarray(self._last_gradients.get("gradients"), dtype=float)
        projection_labels = np.asarray(self._last_gradients.get("projection_labels"), dtype=int).reshape(-1)
        if gradients.ndim != 2 or projection_labels.size == 0:
            self.statusBar().showMessage("Gradient node data unavailable. Compute gradients again.")
            return
        if projection_labels.shape[0] != gradients.shape[0]:
            self.statusBar().showMessage("Gradient node mapping is invalid. Compute gradients again.")
            return

        template_path_raw = str(self._last_gradients.get("template_path") or "").strip()
        template_img = self._active_parcellation_img
        if template_path_raw:
            template_path = Path(template_path_raw)
            if self._active_parcellation_path is None or Path(self._active_parcellation_path) != template_path:
                try:
                    import nibabel as nib

                    template_img = nib.load(str(template_path))
                except Exception as exc:
                    self.statusBar().showMessage(f"Failed to load parcellation for network render: {exc}")
                    return
        if template_img is None:
            self.statusBar().showMessage("No parcellation available for network render.")
            return

        try:
            from dipy.viz import window, actor
            from mrsitoolbox.graphplot.netplot import NetPlot
        except Exception as exc:
            self.statusBar().showMessage(f"Fury network viewer unavailable: {exc}")
            return

        try:
            centroids_mni = np.asarray(
                nettools.compute_centroids(
                    template_img,
                    labels=projection_labels.astype(int),
                    world=False,
                ),
                dtype=float,
            )
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to compute parcel centroids: {exc}")
            return

        if centroids_mni.shape[0] != projection_labels.shape[0]:
            self.statusBar().showMessage("Centroid count mismatch for network render.")
            return

        cmap = self._selected_surface_colormap()
        if not callable(cmap):
            try:
                import matplotlib.cm as mpl_cm

                cmap = mpl_cm.get_cmap(str(cmap))
            except Exception as exc:
                self.statusBar().showMessage(f"Invalid colormap for network render: {exc}")
                return
        n_grad = int(self._last_gradients.get("n_grad", gradients.shape[1] if gradients.ndim == 2 else 1))
        source_name = str(self._last_gradients.get("source_name", "matrix"))
        hemisphere_mode = self._selected_gradient_hemisphere_mode()
        rotation_presets = self._current_gradient_rotation_presets()
        selected_component = self._normalize_gradient_network_component(
            self._selected_gradient_network_component(),
            max_components=n_grad,
        )
        if selected_component == "all":
            component_indices = list(range(n_grad))
        else:
            component_indices = [int(selected_component) - 1]
        opened_components = []
        netplot = NetPlot(window)
        netplot.scene.background((1, 1, 1))
        template_shape = tuple(int(v) for v in np.asarray(netplot.mni_template.shape[:3], dtype=int))
        n_panels = max(1, len(component_indices))
        n_cols = max(1, min(3, n_panels))
        n_rows = max(1, int(math.ceil(n_panels / n_cols)))
        x_step = float(template_shape[0] + 36)
        y_step = float(template_shape[1] + 48)
        title_z = float(template_shape[2] * 0.5)
        hemisphere_mask = self._gradient_projection_hemisphere_mask(
            hemisphere_mode,
            projection_labels,
        )
        brain_hemi = None if hemisphere_mode in {"both", "separate"} else hemisphere_mode

        for panel_idx, comp_idx in enumerate(component_indices):
            component_values = np.asarray(gradients[:, comp_idx], dtype=float)
            finite_mask = (
                np.isfinite(component_values)
                & np.all(np.isfinite(centroids_mni), axis=1)
                & hemisphere_mask
            )
            if not np.any(finite_mask):
                continue

            component_centroids = centroids_mni[finite_mask]
            component_values = component_values[finite_mask]
            abs_values = np.abs(component_values)
            if abs_values.size == 0:
                continue

            vmax_abs = float(np.nanmax(abs_values))
            if not np.isfinite(vmax_abs) or np.isclose(vmax_abs, 0.0):
                radii = np.full(component_values.shape[0], 3.0, dtype=float)
            else:
                radii = 2.5 + 2.5 * (abs_values / vmax_abs)

            finite_values = component_values[np.isfinite(component_values)]
            if finite_values.size == 0:
                continue
            vmin = float(np.nanmin(finite_values))
            vmax = float(np.nanmax(finite_values))
            if np.isclose(vmin, vmax):
                vmin -= 1.0
                vmax += 1.0

            colors = np.asarray(
                cmap(Normalize(vmin=vmin, vmax=vmax)(component_values))[:, :3],
                dtype=float,
            )
            row = int(panel_idx // n_cols)
            col = int(panel_idx % n_cols)
            offset = np.array((col * x_step, row * y_step, 0.0), dtype=float)
            fallback_center = (
                float(offset[0] + template_shape[0] * 0.5),
                float(offset[1] + template_shape[1] * 0.5),
                float(offset[2] + template_shape[2] * 0.5),
            )
            try:
                brain_actor = netplot.add_brain(
                    netplot.mni_template,
                    hemisphere=brain_hemi,
                    opacity=0.16,
                    offset=offset,
                )
                node_actor = netplot.add_nodes(
                    component_centroids,
                    node_radius=radii,
                    node_color=colors,
                    node_labels=None,
                    offset=offset,
                    return_actor=True,
                )
                self._apply_gradient_component_rotation(
                    (brain_actor, node_actor),
                    rotation_presets[comp_idx] if comp_idx < len(rotation_presets) else "Default",
                    fallback_center=fallback_center,
                )
                title_actor = actor.text_3d(
                    f"C{comp_idx + 1} ({rotation_presets[comp_idx] if comp_idx < len(rotation_presets) else 'Default'})",
                    position=(
                        float(offset[0] + template_shape[0] * 0.42),
                        float(offset[1] + template_shape[1] + 12.0),
                        title_z,
                    ),
                    color=(0.0, 0.0, 0.0),
                    font_size=10,
                    justification="center",
                )
                netplot.scene.add(title_actor)
            except Exception as exc:
                self.statusBar().showMessage(
                    f"Failed to build network component {comp_idx + 1}: {exc}"
                )
                return
            opened_components.append(comp_idx + 1)

        if not opened_components:
            self.statusBar().showMessage("No finite gradient nodes available for network render.")
            return
        netplot.scene.ResetCamera()
        netplot.scene.ResetCameraClippingRange()
        win_width = int(max(900, min(1800, n_cols * 420)))
        win_height = int(max(700, min(1400, n_rows * 420)))
        component_text = ", ".join(str(idx) for idx in opened_components)
        self.statusBar().showMessage(
            f"Opening Fury network viewer for components {component_text}."
        )
        QApplication.processEvents()
        try:
            window.show(
                netplot.scene,
                title=f"Gradients Network - {source_name}",
                size=(win_width, win_height),
            )
        except Exception as exc:
            self.statusBar().showMessage(f"Failed to render network viewer: {exc}")
            return
        self.statusBar().showMessage("Closed Fury network viewer.")

__all__ = ['GradientDialogController']
