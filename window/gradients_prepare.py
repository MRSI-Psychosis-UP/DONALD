#!/usr/bin/env python3
"""Popup dialog for gradients projection and network rendering."""

from __future__ import annotations

from pathlib import Path

try:
    from PyQt6.QtWidgets import (
        QAbstractItemView,
        QCheckBox,
        QComboBox,
        QDialog,
        QDoubleSpinBox,
        QGridLayout,
        QGroupBox,
        QHeaderView,
        QHBoxLayout,
        QLabel,
        QFileDialog,
        QLineEdit,
        QProgressBar,
        QPushButton,
        QSpinBox,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
except Exception:
    from PyQt5.QtWidgets import (
        QAbstractItemView,
        QCheckBox,
        QComboBox,
        QDialog,
        QDoubleSpinBox,
        QGridLayout,
        QGroupBox,
        QHeaderView,
        QHBoxLayout,
        QLabel,
        QFileDialog,
        QLineEdit,
        QProgressBar,
        QPushButton,
        QSpinBox,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )


try:
    from window.shared.theme import panel_dialog_stylesheet as _panel_dialog_stylesheet
except Exception:
    from mrsi_viewer.window.shared.theme import panel_dialog_stylesheet as _panel_dialog_stylesheet


class GradientsPrepareDialog(QDialog):
    """Small modeless window for the gradients workflow."""

    def __init__(
        self,
        *,
        theme_name="Dark",
        component_count=4,
        colormap_names=None,
        current_colormap="",
        parcellation_path=None,
        open_parcellation_callback=None,
        compute_callback=None,
        save_callback=None,
        render_3d_callback=None,
        classify_callback=None,
        render_network_callback=None,
        matrix_changed_callback=None,
        parcel_label_key_changed_callback=None,
        parcel_name_key_changed_callback=None,
        component_changed_callback=None,
        colormap_changed_callback=None,
        hemisphere_changed_callback=None,
        surface_mesh_changed_callback=None,
        surface_render_count_changed_callback=None,
        surface_procrustes_changed_callback=None,
        scatter_rotation_changed_callback=None,
        triangular_rgb_changed_callback=None,
        preload_matching_paths_changed_callback=None,
        classification_fit_mode_changed_callback=None,
        triangular_color_order_changed_callback=None,
        rgb_scalar_mode_changed_callback=None,
        rgb_mask_strictness_changed_callback=None,
        classification_surface_mesh_changed_callback=None,
        classification_hemisphere_changed_callback=None,
        classification_colormap_changed_callback=None,
        classification_component_changed_callback=None,
        classification_x_axis_changed_callback=None,
        classification_y_axis_changed_callback=None,
        classification_z_axis_changed_callback=None,
        classification_ref_matrix_changed_callback=None,
        classification_ignore_lh_changed_callback=None,
        classification_ignore_rh_changed_callback=None,
        open_classification_adjacency_callback=None,
        remove_classification_adjacency_callback=None,
        precomputed_row_confirm_callback=None,
        network_component_changed_callback=None,
        rotation_changed_callback=None,
        parent=None,
    ):
        super().__init__(parent)
        self._open_parcellation_callback = open_parcellation_callback
        self._compute_callback = compute_callback
        self._save_callback = save_callback
        self._render_3d_callback = render_3d_callback
        self._classify_callback = classify_callback
        self._render_network_callback = render_network_callback
        self._matrix_changed_callback = matrix_changed_callback
        self._parcel_label_key_changed_callback = parcel_label_key_changed_callback
        self._parcel_name_key_changed_callback = parcel_name_key_changed_callback
        self._component_changed_callback = component_changed_callback
        self._colormap_changed_callback = colormap_changed_callback
        self._hemisphere_changed_callback = hemisphere_changed_callback
        self._surface_mesh_changed_callback = surface_mesh_changed_callback
        self._surface_render_count_changed_callback = surface_render_count_changed_callback
        self._surface_procrustes_changed_callback = surface_procrustes_changed_callback
        self._scatter_rotation_changed_callback = scatter_rotation_changed_callback
        self._triangular_rgb_changed_callback = triangular_rgb_changed_callback
        self._preload_matching_paths_changed_callback = preload_matching_paths_changed_callback
        self._classification_fit_mode_changed_callback = classification_fit_mode_changed_callback
        self._triangular_color_order_changed_callback = triangular_color_order_changed_callback
        self._rgb_scalar_mode_changed_callback = rgb_scalar_mode_changed_callback
        self._rgb_mask_strictness_changed_callback = rgb_mask_strictness_changed_callback
        self._classification_surface_mesh_changed_callback = classification_surface_mesh_changed_callback
        self._classification_hemisphere_changed_callback = classification_hemisphere_changed_callback
        self._classification_colormap_changed_callback = classification_colormap_changed_callback
        self._classification_component_changed_callback = classification_component_changed_callback
        self._classification_x_axis_changed_callback = classification_x_axis_changed_callback
        self._classification_y_axis_changed_callback = classification_y_axis_changed_callback
        self._classification_z_axis_changed_callback = classification_z_axis_changed_callback
        self._classification_ref_matrix_changed_callback = classification_ref_matrix_changed_callback
        self._classification_ignore_lh_changed_callback = classification_ignore_lh_changed_callback
        self._classification_ignore_rh_changed_callback = classification_ignore_rh_changed_callback
        self._open_classification_adjacency_callback = open_classification_adjacency_callback
        self._remove_classification_adjacency_callback = remove_classification_adjacency_callback
        self._precomputed_row_confirm_callback = precomputed_row_confirm_callback
        self._network_component_changed_callback = network_component_changed_callback
        self._rotation_changed_callback = rotation_changed_callback
        self._busy = False
        self._can_compute = False
        self._has_results = False
        self._can_classify = False
        self._has_classification_adjacency = False
        self._precomputed_mode = False
        self._precomputed_row_indices = []
        self._precomputed_columns_ref = None
        self._precomputed_rows_ref = None
        self._surface_procrustes_available = False

        self.setWindowTitle("Gradients")
        self.resize(580, 560)
        self._build_ui()
        self.set_matrix_source(None)
        self.set_parcel_key_options([], [])
        self.set_component_count(component_count)
        self.set_colormap_names(colormap_names or [], current_colormap=current_colormap)
        self.set_parcellation_path(parcellation_path)
        self.set_hemisphere_mode("separate")
        self.set_surface_mesh("fsaverage4")
        self.set_surface_render_component_limit(component_count)
        self.set_surface_render_component_count(1)
        self.set_surface_procrustes_enabled(False)
        self.set_surface_procrustes_available(False)
        self.set_scatter_rotation("Default")
        self.set_triangular_rgb(False)
        self.set_preload_matching_paths(False)
        self.set_classification_fit_mode("triangle")
        self.set_triangular_color_order("RBG")
        self.set_rgb_scalar_mode("barycentric")
        self.set_rgb_mask_strictness(1.0)
        self.set_classification_surface_mesh("fsaverage4")
        self.set_classification_hemisphere_mode("separate")
        self.set_classification_component_options(component_count, selected_component="1")
        self.set_classification_axes("gradient2", "gradient1")
        self.set_classification_ignore_parcel_options([], [], selected_lh="", selected_rh="")
        self.set_classification_adjacency_path(None)
        self.set_precomputed_mode(False)
        self.set_precomputed_rows([], [], selected_row=None, summary_text="", selection_text="")
        self.set_rotation_presets([])
        self.set_progress(0, 1, 0, "Idle")
        self.set_theme(theme_name)
        self._refresh_action_state()

    def _build_ui(self):
        root_layout = QVBoxLayout(self)

        self.matrix_source_label = QLabel("Matrix: none")
        self.matrix_source_label.setWordWrap(True)
        root_layout.addWidget(self.matrix_source_label, 0)

        self.tabs = QTabWidget()
        root_layout.addWidget(self.tabs, 1)

        options_tab = QWidget()
        options_layout = QGridLayout(options_tab)
        row = 0
        options_layout.addWidget(QLabel("Workspace matrix"), row, 0)
        self.matrix_combo = QComboBox()
        self.matrix_combo.currentIndexChanged.connect(self._on_matrix_changed)
        options_layout.addWidget(self.matrix_combo, row, 1)
        row += 1

        options_layout.addWidget(QLabel("N components"), row, 0)
        self.components_spin = QSpinBox()
        self.components_spin.setRange(1, 10)
        self.components_spin.setValue(4)
        self.components_spin.valueChanged.connect(self._on_component_changed)
        options_layout.addWidget(self.components_spin, row, 1)
        row += 1

        options_layout.addWidget(QLabel("Parcellation"), row, 0)
        self.parcellation_button = QPushButton("Set Parcellation")
        self.parcellation_button.clicked.connect(self._trigger_open_parcellation)
        options_layout.addWidget(self.parcellation_button, row, 1)
        row += 1

        self.parcellation_label = QLabel("Parcellation: none")
        self.parcellation_label.setWordWrap(True)
        options_layout.addWidget(self.parcellation_label, row, 0, 1, 2)
        row += 1

        options_layout.addWidget(QLabel("Parcel labels key"), row, 0)
        self.parcel_label_key_combo = QComboBox()
        self.parcel_label_key_combo.setToolTip(
            "Vector key containing integer parcel labels in the selected matrix NPZ."
        )
        self.parcel_label_key_combo.currentIndexChanged.connect(
            self._on_parcel_label_key_changed
        )
        options_layout.addWidget(self.parcel_label_key_combo, row, 1)
        row += 1

        options_layout.addWidget(QLabel("Parcel names key"), row, 0)
        self.parcel_name_key_combo = QComboBox()
        self.parcel_name_key_combo.setToolTip(
            "Vector key containing parcel display names in the selected matrix NPZ."
        )
        self.parcel_name_key_combo.currentIndexChanged.connect(
            self._on_parcel_name_key_changed
        )
        options_layout.addWidget(self.parcel_name_key_combo, row, 1)
        row += 1

        options_layout.addWidget(QLabel("Metab profiles"), row, 0)
        metab_profile_actions = QHBoxLayout()
        self.metabolite_profiles_path_edit = QLineEdit()
        self.metabolite_profiles_path_edit.setPlaceholderText("Embedded in selected matrix")
        metab_profile_actions.addWidget(self.metabolite_profiles_path_edit, 1)
        self.metabolite_profiles_browse_button = QPushButton("Browse")
        self.metabolite_profiles_browse_button.clicked.connect(self._choose_metabolite_profiles_path)
        metab_profile_actions.addWidget(self.metabolite_profiles_browse_button, 0)
        self.metabolite_profiles_clear_button = QPushButton("Clear")
        self.metabolite_profiles_clear_button.clicked.connect(self._clear_metabolite_profiles_path)
        metab_profile_actions.addWidget(self.metabolite_profiles_clear_button, 0)
        options_layout.addLayout(metab_profile_actions, row, 1)
        row += 1

        compute_actions = QHBoxLayout()
        self.compute_button = QPushButton("Process")
        self.compute_button.clicked.connect(self._trigger_compute)
        compute_actions.addWidget(self.compute_button)
        self.save_button = QPushButton("Write to File")
        self.save_button.clicked.connect(self._trigger_save)
        compute_actions.addWidget(self.save_button)
        options_layout.addLayout(compute_actions, row, 0, 1, 2)
        options_layout.setRowStretch(row + 1, 1)
        self.tabs.addTab(options_tab, "Compute")

        self.precomputed_tab = QWidget()
        precomputed_layout = QVBoxLayout(self.precomputed_tab)
        self.precomputed_summary_label = QLabel("No precomputed gradients bundle loaded.")
        self.precomputed_summary_label.setWordWrap(True)
        precomputed_layout.addWidget(self.precomputed_summary_label, 0)

        self.precomputed_selection_label = QLabel("Selected pair: none")
        self.precomputed_selection_label.setWordWrap(True)
        precomputed_layout.addWidget(self.precomputed_selection_label, 0)

        self.precomputed_table = QTableWidget()
        self.precomputed_table.setAlternatingRowColors(True)
        self.precomputed_table.verticalHeader().setVisible(False)
        try:
            self.precomputed_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            self.precomputed_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
            self.precomputed_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        except Exception:
            self.precomputed_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            self.precomputed_table.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.precomputed_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.precomputed_table.itemSelectionChanged.connect(self._on_precomputed_table_selection_changed)
        self.precomputed_table.itemDoubleClicked.connect(self._on_precomputed_table_double_clicked)
        precomputed_layout.addWidget(self.precomputed_table, 1)

        precomputed_actions = QHBoxLayout()
        precomputed_actions.addStretch(1)
        self.precomputed_confirm_button = QPushButton("Use selected row")
        self.precomputed_confirm_button.clicked.connect(self._trigger_precomputed_row_confirm)
        precomputed_actions.addWidget(self.precomputed_confirm_button, 0)
        precomputed_layout.addLayout(precomputed_actions, 0)

        display_tab = QWidget()
        display_layout = QGridLayout(display_tab)
        row = 0
        display_layout.addWidget(QLabel("3D colorbar"), row, 0)
        self.colorbar_combo = QComboBox()
        self.colorbar_combo.currentTextChanged.connect(self._on_colormap_changed)
        display_layout.addWidget(self.colorbar_combo, row, 1)
        row += 1

        display_layout.addWidget(QLabel("Hemisphere"), row, 0)
        self.hemisphere_combo = QComboBox()
        self.hemisphere_combo.addItems(["Both", "LH", "RH"])
        self.hemisphere_combo.currentTextChanged.connect(self._on_hemisphere_changed)
        display_layout.addWidget(self.hemisphere_combo, row, 1)
        row += 1

        display_layout.addWidget(QLabel("fsaverage scale"), row, 0)
        self.surface_mesh_combo = QComboBox()
        self.surface_mesh_combo.addItems(
            ["fsaverage3", "fsaverage4", "fsaverage5", "fsaverage6", "fsaverage7"]
        )
        self.surface_mesh_combo.currentTextChanged.connect(self._on_surface_mesh_changed)
        display_layout.addWidget(self.surface_mesh_combo, row, 1)
        row += 1

        display_layout.addWidget(QLabel("Gradients to render"), row, 0)
        self.surface_render_count_spin = QSpinBox()
        self.surface_render_count_spin.setRange(1, 10)
        self.surface_render_count_spin.setValue(1)
        self.surface_render_count_spin.valueChanged.connect(self._on_surface_render_count_changed)
        display_layout.addWidget(self.surface_render_count_spin, row, 1)
        row += 1

        self.surface_procrustes_check = QCheckBox("Align to gradients_avg (Procrustes)")
        self.surface_procrustes_check.toggled.connect(self._on_surface_procrustes_changed)
        display_layout.addWidget(self.surface_procrustes_check, row, 0, 1, 2)
        row += 1

        self.render_3d_button = QPushButton("Render")
        self.render_3d_button.clicked.connect(self._trigger_render_3d)
        display_layout.addWidget(self.render_3d_button, row, 0, 1, 2)
        row += 1
        display_layout.setRowStretch(row, 1)

        self.tabs.addTab(display_tab, "fsaverage")

        classification_tab = QWidget()
        classification_layout = QGridLayout(classification_tab)
        row = 0
        classification_layout.addWidget(QLabel("Color scheme"), row, 0)
        self.classification_colorbar_combo = QComboBox()
        self.classification_colorbar_combo.currentTextChanged.connect(
            self._on_classification_colormap_changed
        )
        classification_layout.addWidget(self.classification_colorbar_combo, row, 1)
        row += 1

        classification_layout.addWidget(QLabel("Surface component"), row, 0)
        self.classification_component_combo = QComboBox()
        self.classification_component_combo.currentIndexChanged.connect(
            self._on_classification_component_changed
        )
        classification_layout.addWidget(self.classification_component_combo, row, 1)
        row += 1

        classification_layout.addWidget(QLabel("X axis"), row, 0)
        self.classification_x_axis_combo = QComboBox()
        self.classification_x_axis_combo.addItem("Gradient 1", "gradient1")
        self.classification_x_axis_combo.addItem("Gradient 2", "gradient2")
        self.classification_x_axis_combo.addItem("Gradient 3", "gradient3")
        self.classification_x_axis_combo.addItem("Gradient Ref 1", "gradient_ref1")
        self.classification_x_axis_combo.addItem("Gradient Ref 2", "gradient_ref2")
        self.classification_x_axis_combo.addItem("Gradient Ref 3", "gradient_ref3")
        self.classification_x_axis_combo.addItem("Spatial", "spatial")
        self.classification_x_axis_combo.currentIndexChanged.connect(
            self._on_classification_x_axis_changed
        )
        classification_layout.addWidget(self.classification_x_axis_combo, row, 1)
        row += 1

        classification_layout.addWidget(QLabel("Y axis"), row, 0)
        self.classification_y_axis_combo = QComboBox()
        self.classification_y_axis_combo.addItem("Gradient 1", "gradient1")
        self.classification_y_axis_combo.addItem("Gradient 2", "gradient2")
        self.classification_y_axis_combo.addItem("Gradient 3", "gradient3")
        self.classification_y_axis_combo.addItem("Gradient Ref 1", "gradient_ref1")
        self.classification_y_axis_combo.addItem("Gradient Ref 2", "gradient_ref2")
        self.classification_y_axis_combo.addItem("Gradient Ref 3", "gradient_ref3")
        self.classification_y_axis_combo.addItem("Spatial", "spatial")
        self.classification_y_axis_combo.currentIndexChanged.connect(
            self._on_classification_y_axis_changed
        )
        classification_layout.addWidget(self.classification_y_axis_combo, row, 1)
        row += 1

        classification_layout.addWidget(QLabel("Z axis"), row, 0)
        self.classification_z_axis_combo = QComboBox()
        self.classification_z_axis_combo.addItem("None (2D)", "none")
        self.classification_z_axis_combo.addItem("Gradient 1", "gradient1")
        self.classification_z_axis_combo.addItem("Gradient 2", "gradient2")
        self.classification_z_axis_combo.addItem("Gradient 3", "gradient3")
        self.classification_z_axis_combo.addItem("Gradient Ref 1", "gradient_ref1")
        self.classification_z_axis_combo.addItem("Gradient Ref 2", "gradient_ref2")
        self.classification_z_axis_combo.addItem("Gradient Ref 3", "gradient_ref3")
        self.classification_z_axis_combo.currentIndexChanged.connect(
            self._on_classification_z_axis_changed
        )
        classification_layout.addWidget(self.classification_z_axis_combo, row, 1)
        row += 1

        self.classification_ref_matrix_label = QLabel("Reference matrix")
        classification_layout.addWidget(self.classification_ref_matrix_label, row, 0)
        self.classification_ref_matrix_combo = QComboBox()
        self.classification_ref_matrix_combo.currentIndexChanged.connect(
            self._on_classification_ref_matrix_changed
        )
        classification_layout.addWidget(self.classification_ref_matrix_combo, row, 1)
        row += 1

        classification_layout.addWidget(QLabel("Ignore parcel (LH)"), row, 0)
        self.classification_ignore_lh_combo = QComboBox()
        self.classification_ignore_lh_combo.currentIndexChanged.connect(
            self._on_classification_ignore_lh_changed
        )
        classification_layout.addWidget(self.classification_ignore_lh_combo, row, 1)
        row += 1

        classification_layout.addWidget(QLabel("Ignore parcel (RH)"), row, 0)
        self.classification_ignore_rh_combo = QComboBox()
        self.classification_ignore_rh_combo.currentIndexChanged.connect(
            self._on_classification_ignore_rh_changed
        )
        classification_layout.addWidget(self.classification_ignore_rh_combo, row, 1)
        row += 1

        classification_layout.addWidget(QLabel("fsaverage scale"), row, 0)
        self.classification_surface_mesh_combo = QComboBox()
        self.classification_surface_mesh_combo.addItems(
            ["fsaverage3", "fsaverage4", "fsaverage5", "fsaverage6", "fsaverage7"]
        )
        self.classification_surface_mesh_combo.currentTextChanged.connect(
            self._on_classification_surface_mesh_changed
        )
        classification_layout.addWidget(self.classification_surface_mesh_combo, row, 1)
        row += 1

        classification_layout.addWidget(QLabel("Hemisphere"), row, 0)
        self.classification_hemisphere_combo = QComboBox()
        self.classification_hemisphere_combo.addItems(["Both", "LH", "RH", "Separate"])
        self.classification_hemisphere_combo.currentTextChanged.connect(
            self._on_classification_hemisphere_changed
        )
        classification_layout.addWidget(self.classification_hemisphere_combo, row, 1)
        row += 1

        classification_layout.addWidget(QLabel("Scatter rotation"), row, 0)
        self.scatter_rotation_combo = QComboBox()
        self.scatter_rotation_combo.addItems(["Default", "+90", "-90", "180"])
        self.scatter_rotation_combo.currentTextChanged.connect(self._on_scatter_rotation_changed)
        classification_layout.addWidget(self.scatter_rotation_combo, row, 1)
        row += 1

        self.triangular_rgb_check = QCheckBox("Triangular RGB coloring")
        self.triangular_rgb_check.setObjectName("triangularRgbCheck")
        self.triangular_rgb_check.toggled.connect(self._on_triangular_rgb_changed)
        classification_layout.addWidget(self.triangular_rgb_check, row, 0, 1, 2)
        row += 1

        self.preload_matching_paths_check = QCheckBox("Preload matching saved path")
        self.preload_matching_paths_check.setObjectName("preloadMatchingPathsCheck")
        self.preload_matching_paths_check.setToolTip(
            "When View opens a 2D RGB scatter, find the newest desc-free_energy_paths.npz "
            "matching its subject, session, and parcellation, then load its endpoints and generate paths. "
            "If needed, the adjacency referenced by that archive is loaded as well."
        )
        self.preload_matching_paths_check.toggled.connect(
            self._on_preload_matching_paths_changed
        )
        classification_layout.addWidget(self.preload_matching_paths_check, row, 0, 1, 2)
        row += 1

        classification_layout.addWidget(QLabel("Fit mode"), row, 0)
        self.classification_fit_mode_combo = QComboBox()
        self.classification_fit_mode_combo.addItem("Triangular", "triangle")
        self.classification_fit_mode_combo.addItem("Square", "square")
        self.classification_fit_mode_combo.currentIndexChanged.connect(
            self._on_classification_fit_mode_changed
        )
        classification_layout.addWidget(self.classification_fit_mode_combo, row, 1)
        row += 1

        classification_layout.addWidget(QLabel("RGB order (triangle/base)"), row, 0)
        self.triangular_color_order_combo = QComboBox()
        self.triangular_color_order_combo.addItems(["RBG", "RGB", "BRG", "BGR", "GRB", "GBR"])
        self.triangular_color_order_combo.currentTextChanged.connect(
            self._on_triangular_color_order_changed
        )
        classification_layout.addWidget(self.triangular_color_order_combo, row, 1)
        row += 1

        classification_layout.addWidget(QLabel("RGB scalar"), row, 0)
        self.rgb_scalar_mode_combo = QComboBox()
        self.rgb_scalar_mode_combo.addItem("Barycentric", "barycentric")
        self.rgb_scalar_mode_combo.addItem("Principal curve", "principal_curve")
        self.rgb_scalar_mode_combo.setToolTip(
            "Choose how triangular RGB colors are collapsed to a one-dimensional coordinate for ordering and binning."
        )
        self.rgb_scalar_mode_combo.currentIndexChanged.connect(self._on_rgb_scalar_mode_changed)
        classification_layout.addWidget(self.rgb_scalar_mode_combo, row, 1)
        row += 1

        classification_layout.addWidget(QLabel("RGB mask strictness"), row, 0)
        self.rgb_mask_strictness_spin = QDoubleSpinBox()
        self.rgb_mask_strictness_spin.setRange(0.0, 5.0)
        self.rgb_mask_strictness_spin.setDecimals(2)
        self.rgb_mask_strictness_spin.setSingleStep(0.10)
        self.rgb_mask_strictness_spin.setValue(1.0)
        self.rgb_mask_strictness_spin.valueChanged.connect(self._on_rgb_mask_strictness_changed)
        classification_layout.addWidget(self.rgb_mask_strictness_spin, row, 1)
        row += 1

        classification_layout.addWidget(QLabel("Adjacency"), row, 0)
        adjacency_actions = QHBoxLayout()
        self.classification_adjacency_button = QPushButton("Upload NPZ")
        self.classification_adjacency_button.clicked.connect(
            self._trigger_open_classification_adjacency
        )
        adjacency_actions.addWidget(self.classification_adjacency_button, 1)
        self.classification_adjacency_remove_button = QPushButton("Remove")
        self.classification_adjacency_remove_button.clicked.connect(
            self._trigger_remove_classification_adjacency
        )
        adjacency_actions.addWidget(self.classification_adjacency_remove_button, 0)
        classification_layout.addLayout(adjacency_actions, row, 1)
        row += 1

        self.classification_adjacency_label = QLabel("Adjacency: none")
        self.classification_adjacency_label.setWordWrap(True)
        classification_layout.addWidget(self.classification_adjacency_label, row, 0, 1, 2)
        row += 1

        self.classify_button = QPushButton("View Scatter + Surface")
        self.classify_button.clicked.connect(self._trigger_classify)
        classification_layout.addWidget(self.classify_button, row, 0, 1, 2)
        classification_layout.setRowStretch(row + 1, 1)
        self.tabs.addTab(classification_tab, "Scatter + Surface")

        network_tab = QWidget()
        rotation_layout = QGridLayout(network_tab)
        row = 0
        rotation_layout.addWidget(QLabel("Display component"), row, 0)
        self.network_component_combo = QComboBox()
        self.network_component_combo.currentIndexChanged.connect(self._on_network_component_changed)
        rotation_layout.addWidget(self.network_component_combo, row, 1, 1, 3)
        row += 1

        rotation_layout.addWidget(QLabel("Per-component rotation"), row, 0, 1, 4)
        self._rotation_labels = []
        self._rotation_combos = []
        rotation_options = [
            "Default",
            "X +90",
            "X -90",
            "Y +90",
            "Y -90",
            "Y 180",
            "Z +90",
            "Z -90",
        ]
        for idx in range(10):
            grid_row = row + 1 + (idx // 2)
            grid_col = (idx % 2) * 2
            label = QLabel(f"C{idx + 1}")
            combo = QComboBox()
            combo.addItems(rotation_options)
            combo.currentTextChanged.connect(
                lambda value, comp_idx=idx: self._on_rotation_changed(comp_idx, value)
            )
            rotation_layout.addWidget(label, grid_row, grid_col)
            rotation_layout.addWidget(combo, grid_row, grid_col + 1)
            self._rotation_labels.append(label)
            self._rotation_combos.append(combo)

        network_button_row = row + 6
        self.render_network_button = QPushButton("Render")
        self.render_network_button.clicked.connect(self._trigger_render_network)
        rotation_layout.addWidget(self.render_network_button, network_button_row, 0, 1, 4)
        rotation_layout.setRowStretch(network_button_row + 1, 1)
        self.tabs.addTab(network_tab, "Network")

        progress_row = QHBoxLayout()
        progress_row.addWidget(QLabel("Progress"))
        self.progress_bar = QProgressBar()
        progress_row.addWidget(self.progress_bar, 1)
        root_layout.addLayout(progress_row)

    def _on_component_changed(self, value):
        if self._component_changed_callback is not None:
            try:
                self._component_changed_callback(int(value))
            except Exception:
                pass

    def _on_matrix_changed(self, _index):
        if self._matrix_changed_callback is not None:
            try:
                self._matrix_changed_callback(self.selected_matrix_entry_id())
            except Exception:
                pass

    def _on_parcel_label_key_changed(self, _index):
        if self._parcel_label_key_changed_callback is not None:
            try:
                self._parcel_label_key_changed_callback(self.selected_parcel_label_key())
            except Exception:
                pass

    def _on_parcel_name_key_changed(self, _index):
        if self._parcel_name_key_changed_callback is not None:
            try:
                self._parcel_name_key_changed_callback(self.selected_parcel_name_key())
            except Exception:
                pass

    def _on_colormap_changed(self, value):
        if self._colormap_changed_callback is not None:
            try:
                self._colormap_changed_callback(str(value))
            except Exception:
                pass

    def _on_hemisphere_changed(self, value):
        if self._hemisphere_changed_callback is not None:
            try:
                self._hemisphere_changed_callback(str(value))
            except Exception:
                pass

    def _on_surface_mesh_changed(self, value):
        if self._surface_mesh_changed_callback is not None:
            try:
                self._surface_mesh_changed_callback(str(value))
            except Exception:
                pass

    def _on_surface_render_count_changed(self, value):
        if self._surface_render_count_changed_callback is not None:
            try:
                self._surface_render_count_changed_callback(int(value))
            except Exception:
                pass

    def _on_surface_procrustes_changed(self, checked):
        if self._surface_procrustes_changed_callback is not None:
            try:
                self._surface_procrustes_changed_callback(bool(checked))
            except Exception:
                pass

    def _on_scatter_rotation_changed(self, value):
        if self._scatter_rotation_changed_callback is not None:
            try:
                self._scatter_rotation_changed_callback(str(value))
            except Exception:
                pass

    def _on_triangular_rgb_changed(self, checked):
        if self._triangular_rgb_changed_callback is not None:
            try:
                self._triangular_rgb_changed_callback(bool(checked))
            except Exception:
                pass
        self._refresh_action_state()

    def _on_preload_matching_paths_changed(self, checked):
        if self._preload_matching_paths_changed_callback is not None:
            try:
                self._preload_matching_paths_changed_callback(bool(checked))
            except Exception:
                pass

    def _on_classification_fit_mode_changed(self, _index):
        if self._classification_fit_mode_changed_callback is not None:
            try:
                self._classification_fit_mode_changed_callback(self.selected_classification_fit_mode())
            except Exception:
                pass

    def _on_triangular_color_order_changed(self, value):
        if self._triangular_color_order_changed_callback is not None:
            try:
                self._triangular_color_order_changed_callback(str(value))
            except Exception:
                pass

    def _on_rgb_scalar_mode_changed(self, _index):
        if self._rgb_scalar_mode_changed_callback is not None:
            try:
                self._rgb_scalar_mode_changed_callback(self.selected_rgb_scalar_mode())
            except Exception:
                pass

    def _on_rgb_mask_strictness_changed(self, value):
        if self._rgb_mask_strictness_changed_callback is not None:
            try:
                self._rgb_mask_strictness_changed_callback(float(value))
            except Exception:
                pass

    def _on_classification_surface_mesh_changed(self, value):
        if self._classification_surface_mesh_changed_callback is not None:
            try:
                self._classification_surface_mesh_changed_callback(str(value))
            except Exception:
                pass

    def _on_classification_hemisphere_changed(self, value):
        if self._classification_hemisphere_changed_callback is not None:
            try:
                self._classification_hemisphere_changed_callback(str(value))
            except Exception:
                pass

    def _on_classification_colormap_changed(self, value):
        if self._classification_colormap_changed_callback is not None:
            try:
                self._classification_colormap_changed_callback(str(value))
            except Exception:
                pass

    def _on_classification_component_changed(self, _index):
        if self._classification_component_changed_callback is not None:
            try:
                self._classification_component_changed_callback(
                    self.selected_classification_component()
                )
            except Exception:
                pass

    def _on_classification_x_axis_changed(self, _index):
        self._refresh_classification_reference_visibility()
        if self._classification_x_axis_changed_callback is not None:
            try:
                self._classification_x_axis_changed_callback(self.selected_classification_x_axis())
            except Exception:
                pass

    def _on_classification_y_axis_changed(self, _index):
        self._refresh_classification_reference_visibility()
        if self._classification_y_axis_changed_callback is not None:
            try:
                self._classification_y_axis_changed_callback(self.selected_classification_y_axis())
            except Exception:
                pass

    def _on_classification_z_axis_changed(self, _index):
        self._refresh_classification_reference_visibility()
        if self._classification_z_axis_changed_callback is not None:
            try:
                self._classification_z_axis_changed_callback(self.selected_classification_z_axis())
            except Exception:
                pass

    def _on_classification_ref_matrix_changed(self, _index):
        if self._classification_ref_matrix_changed_callback is not None:
            try:
                self._classification_ref_matrix_changed_callback(
                    self.selected_classification_ref_matrix_entry_id()
                )
            except Exception:
                pass

    def _on_classification_ignore_lh_changed(self, _index):
        if self._classification_ignore_lh_changed_callback is not None:
            try:
                self._classification_ignore_lh_changed_callback(
                    self.selected_classification_ignore_lh_parcel()
                )
            except Exception:
                pass

    def _on_classification_ignore_rh_changed(self, _index):
        if self._classification_ignore_rh_changed_callback is not None:
            try:
                self._classification_ignore_rh_changed_callback(
                    self.selected_classification_ignore_rh_parcel()
                )
            except Exception:
                pass

    def _on_rotation_changed(self, index, value):
        if self._rotation_changed_callback is not None:
            try:
                self._rotation_changed_callback(int(index), str(value))
            except Exception:
                pass

    def _on_network_component_changed(self, _index):
        if self._network_component_changed_callback is not None:
            try:
                self._network_component_changed_callback(self.selected_network_component())
            except Exception:
                pass

    def _trigger_open_parcellation(self):
        if self._open_parcellation_callback is not None:
            self._open_parcellation_callback()

    def _trigger_compute(self):
        if self._compute_callback is not None:
            self._compute_callback()

    def _trigger_save(self):
        if self._save_callback is not None:
            self._save_callback()

    def _trigger_render_3d(self):
        if self._render_3d_callback is not None:
            self._render_3d_callback()

    def _trigger_classify(self):
        if self._classify_callback is not None:
            self._classify_callback()

    def _trigger_open_classification_adjacency(self):
        if self._open_classification_adjacency_callback is not None:
            self._open_classification_adjacency_callback()

    def _trigger_remove_classification_adjacency(self):
        if self._remove_classification_adjacency_callback is not None:
            self._remove_classification_adjacency_callback()

    def _trigger_precomputed_row_confirm(self):
        if self._precomputed_row_confirm_callback is None:
            return
        row_index = self.selected_precomputed_row_index()
        if row_index is None:
            return
        try:
            self._precomputed_row_confirm_callback(int(row_index))
        except Exception:
            pass

    def _trigger_render_network(self):
        if self._render_network_callback is not None:
            self._render_network_callback()

    def _choose_metabolite_profiles_path(self):
        current = self.metabolite_profiles_path()
        start_dir = str(Path(current).parent) if current else ""
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Select metabolite profiles NPZ",
            start_dir,
            "NPZ files (*.npz);;All files (*)",
        )
        if path:
            self.set_metabolite_profiles_path(path)

    def _clear_metabolite_profiles_path(self):
        self.set_metabolite_profiles_path("")

    def metabolite_profiles_path(self) -> str:
        return str(self.metabolite_profiles_path_edit.text() or "").strip()

    def set_metabolite_profiles_path(self, path) -> None:
        self.metabolite_profiles_path_edit.setText(str(path or "").strip())

    def component_count(self) -> int:
        return int(self.components_spin.value())

    def selected_matrix_entry_id(self):
        return self.matrix_combo.currentData()

    def selected_parcel_label_key(self) -> str:
        value = self.parcel_label_key_combo.currentData()
        return str(value or "").strip()

    def selected_parcel_name_key(self) -> str:
        value = self.parcel_name_key_combo.currentData()
        return str(value or "").strip()

    def selected_colormap(self) -> str:
        return self.colorbar_combo.currentText().strip()

    def selected_hemisphere(self) -> str:
        return self.hemisphere_combo.currentText().strip()

    def selected_surface_mesh(self) -> str:
        return self.surface_mesh_combo.currentText().strip()

    def selected_surface_render_component_count(self) -> int:
        return int(self.surface_render_count_spin.value())

    def use_surface_procrustes(self) -> bool:
        return bool(self.surface_procrustes_check.isChecked())

    def selected_scatter_rotation(self) -> str:
        return self.scatter_rotation_combo.currentText().strip()

    def use_triangular_rgb(self) -> bool:
        return bool(self.triangular_rgb_check.isChecked())

    def preload_matching_paths(self) -> bool:
        return bool(self.preload_matching_paths_check.isChecked())

    def selected_classification_fit_mode(self) -> str:
        value = self.classification_fit_mode_combo.currentData()
        if value is None:
            return "triangle"
        text = str(value).strip().lower()
        return text if text in {"triangle", "square"} else "triangle"

    def selected_classification_colormap(self) -> str:
        return self.classification_colorbar_combo.currentText().strip()

    def selected_triangular_color_order(self) -> str:
        return self.triangular_color_order_combo.currentText().strip().upper()

    def selected_rgb_scalar_mode(self) -> str:
        value = self.rgb_scalar_mode_combo.currentData()
        text = str(value or "barycentric").strip().lower()
        return text if text in {"barycentric", "principal_curve"} else "barycentric"

    def selected_rgb_mask_strictness(self) -> float:
        try:
            return float(self.rgb_mask_strictness_spin.value())
        except Exception:
            return 1.0

    def selected_classification_surface_mesh(self) -> str:
        return self.classification_surface_mesh_combo.currentText().strip()

    def selected_classification_hemisphere(self) -> str:
        return self.classification_hemisphere_combo.currentText().strip()

    def selected_classification_component(self) -> str:
        value = self.classification_component_combo.currentData()
        if value is None:
            return "1"
        return str(value).strip() or "1"

    def selected_classification_x_axis(self) -> str:
        value = self.classification_x_axis_combo.currentData()
        if value is None:
            return "gradient2"
        return str(value).strip() or "gradient2"

    def selected_classification_y_axis(self) -> str:
        value = self.classification_y_axis_combo.currentData()
        if value is None:
            return "gradient1"
        return str(value).strip() or "gradient1"

    def selected_classification_z_axis(self) -> str:
        value = self.classification_z_axis_combo.currentData()
        if value is None:
            return "none"
        return str(value).strip() or "none"

    def selected_classification_ref_matrix_entry_id(self):
        return self.classification_ref_matrix_combo.currentData()

    def selected_classification_ignore_lh_parcel(self) -> str:
        value = self.classification_ignore_lh_combo.currentData()
        return str(value or "").strip()

    def selected_classification_ignore_rh_parcel(self) -> str:
        value = self.classification_ignore_rh_combo.currentData()
        return str(value or "").strip()

    def selected_network_component(self) -> str:
        value = self.network_component_combo.currentData()
        if value is None:
            return "all"
        return str(value).strip() or "all"

    def selected_precomputed_row_index(self):
        selected_items = self.precomputed_table.selectedItems()
        if not selected_items:
            return None
        row = selected_items[0].row()
        if row < 0 or row >= len(self._precomputed_row_indices):
            return None
        return self._precomputed_row_indices[row]

    def rotation_presets(self):
        return [combo.currentText().strip() for combo in self._rotation_combos]

    def set_component_count(self, value: int) -> None:
        try:
            value = int(value)
        except Exception:
            value = 4
        value = max(1, min(10, value))
        self.components_spin.blockSignals(True)
        self.components_spin.setValue(value)
        self.components_spin.blockSignals(False)
        self._refresh_rotation_visibility(value)

    def set_colormap_names(self, names, current_colormap="") -> None:
        current = str(current_colormap or "").strip()
        self.colorbar_combo.blockSignals(True)
        self.colorbar_combo.clear()
        self.classification_colorbar_combo.blockSignals(True)
        self.classification_colorbar_combo.clear()
        for name in names:
            text = str(name).strip()
            if text:
                self.colorbar_combo.addItem(text)
                self.classification_colorbar_combo.addItem(text)
        if current and self.colorbar_combo.findText(current) >= 0:
            self.colorbar_combo.setCurrentText(current)
        elif self.colorbar_combo.count() > 0:
            self.colorbar_combo.setCurrentIndex(0)
        if current and self.classification_colorbar_combo.findText(current) >= 0:
            self.classification_colorbar_combo.setCurrentText(current)
        elif self.classification_colorbar_combo.count() > 0:
            self.classification_colorbar_combo.setCurrentIndex(0)
        self.colorbar_combo.blockSignals(False)
        self.classification_colorbar_combo.blockSignals(False)

    def set_parcellation_path(self, path) -> None:
        if path is None:
            self.parcellation_label.setText("Parcellation: none")
            return
        try:
            name = Path(path).name
        except Exception:
            name = str(path)
        self.parcellation_label.setText(f"Parcellation: {name}")

    def set_matrix_source(self, text) -> None:
        label = str(text or "").strip()
        self.matrix_source_label.setText(f"Matrix: {label or 'none'}")

    def set_matrix_options(self, options, selected_entry_id=None) -> None:
        current_data = self.matrix_combo.currentData()
        target_id = selected_entry_id if selected_entry_id is not None else current_data
        self.matrix_combo.blockSignals(True)
        self.matrix_combo.clear()
        selected_index = -1
        for idx, item in enumerate(list(options or [])):
            if isinstance(item, dict):
                label = str(item.get("label", "")).strip()
                entry_id = item.get("id")
            else:
                try:
                    label, entry_id = item
                except Exception:
                    continue
                label = str(label).strip()
            if not label:
                label = str(entry_id or "matrix")
            self.matrix_combo.addItem(label, entry_id)
            if target_id is not None and entry_id == target_id:
                selected_index = idx
        if selected_index < 0 and self.matrix_combo.count() > 0:
            selected_index = 0
        if selected_index >= 0:
            self.matrix_combo.setCurrentIndex(selected_index)
        self.matrix_combo.blockSignals(False)

    def set_parcel_key_options(
        self,
        label_options,
        name_options,
        *,
        selected_label_key="",
        selected_name_key="",
        auto_label_key="",
        auto_name_key="",
    ) -> None:
        self._set_parcel_key_combo(
            self.parcel_label_key_combo,
            label_options,
            selected_label_key,
            auto_label_key=auto_label_key,
        )
        self._set_parcel_key_combo(
            self.parcel_name_key_combo,
            name_options,
            selected_name_key,
            auto_label_key=auto_name_key,
        )

    @staticmethod
    def _set_parcel_key_combo(combo, options, selected_key, *, auto_label_key="") -> None:
        selected = str(selected_key or "").strip()
        auto_key = str(auto_label_key or "").strip()
        auto_label = f"Auto ({auto_key})" if auto_key else "Auto detect"
        seen = set()

        combo.blockSignals(True)
        combo.clear()
        combo.addItem(auto_label, "")
        selected_index = 0
        for item in list(options or []):
            if isinstance(item, dict):
                key = str(item.get("key", "")).strip()
                label = str(item.get("label", "")).strip() or key
            else:
                key = str(item or "").strip()
                label = key
            if not key or key in seen:
                continue
            seen.add(key)
            combo.addItem(label, key)
            if key == selected:
                selected_index = combo.count() - 1
        if selected and selected not in seen:
            combo.addItem(f"{selected} (missing)", selected)
            selected_index = combo.count() - 1
        combo.setCurrentIndex(selected_index)
        combo.blockSignals(False)

    def set_classification_reference_matrix_options(self, options, selected_entry_id=None) -> None:
        current_data = self.classification_ref_matrix_combo.currentData()
        target_id = selected_entry_id if selected_entry_id is not None else current_data
        self.classification_ref_matrix_combo.blockSignals(True)
        self.classification_ref_matrix_combo.clear()
        selected_index = -1
        for idx, item in enumerate(list(options or [])):
            if isinstance(item, dict):
                label = str(item.get("label", "")).strip()
                entry_id = item.get("id")
            else:
                try:
                    label, entry_id = item
                except Exception:
                    continue
                label = str(label).strip()
            if not label:
                label = str(entry_id or "matrix")
            self.classification_ref_matrix_combo.addItem(label, entry_id)
            if target_id is not None and entry_id == target_id:
                selected_index = idx
        if selected_index < 0 and self.classification_ref_matrix_combo.count() > 0:
            selected_index = 0
        if selected_index >= 0:
            self.classification_ref_matrix_combo.setCurrentIndex(selected_index)
        elif self.classification_ref_matrix_combo.count() == 0:
            self.classification_ref_matrix_combo.addItem("No workspace matrices available", None)
        self.classification_ref_matrix_combo.blockSignals(False)
        self._refresh_classification_reference_visibility()

    def set_progress(self, minimum: int, maximum: int, value: int, text: str) -> None:
        self.progress_bar.setRange(int(minimum), int(maximum))
        self.progress_bar.setValue(int(value))
        self.progress_bar.setFormat(str(text or ""))

    def set_precomputed_mode(self, enabled: bool) -> None:
        self._precomputed_mode = bool(enabled)
        tab_index = self.tabs.indexOf(self.precomputed_tab)
        if self._precomputed_mode and tab_index < 0:
            self.tabs.insertTab(1, self.precomputed_tab, "Covars")
        elif not self._precomputed_mode and tab_index >= 0:
            self.tabs.removeTab(tab_index)
        self._refresh_action_state()

    def focus_precomputed_tab(self) -> None:
        tab_index = self.tabs.indexOf(self.precomputed_tab)
        if tab_index >= 0:
            self.tabs.setCurrentIndex(tab_index)

    def set_precomputed_rows(
        self,
        columns,
        rows,
        *,
        selected_row=None,
        summary_text="",
        selection_text="",
    ) -> None:
        self.precomputed_summary_label.setText(str(summary_text or "No precomputed gradients bundle loaded."))
        self.precomputed_selection_label.setText(str(selection_text or "Selected pair: none"))

        same_payload = rows is self._precomputed_rows_ref and columns is self._precomputed_columns_ref

        if same_payload:
            self.precomputed_table.blockSignals(True)
            self.precomputed_table.clearSelection()
            if selected_row is not None:
                try:
                    selected_row = int(selected_row)
                except Exception:
                    selected_row = None
                if selected_row is not None:
                    for table_row, source_row in enumerate(self._precomputed_row_indices):
                        if int(source_row) == selected_row:
                            self.precomputed_table.selectRow(table_row)
                            break
            elif self.precomputed_table.rowCount() == 1:
                self.precomputed_table.selectRow(0)
            self.precomputed_table.blockSignals(False)
            self._refresh_action_state()
            return

        row_dicts = [dict(row) for row in list(rows or [])]
        self._precomputed_row_indices = [int(row.get("__row_index__", idx)) for idx, row in enumerate(row_dicts)]
        visible_columns = [str(col) for col in list(columns or []) if str(col) != "__row_index__"]
        self._precomputed_columns_ref = columns
        self._precomputed_rows_ref = rows
        self.precomputed_table.blockSignals(True)
        self.precomputed_table.clear()
        self.precomputed_table.setColumnCount(len(visible_columns) + 1)
        self.precomputed_table.setHorizontalHeaderLabels(["Index"] + visible_columns)
        self.precomputed_table.setRowCount(len(row_dicts))

        for table_row, row_data in enumerate(row_dicts):
            index_item = QTableWidgetItem(str(self._precomputed_row_indices[table_row]))
            self.precomputed_table.setItem(table_row, 0, index_item)
            for col_idx, column_name in enumerate(visible_columns, start=1):
                item = QTableWidgetItem(str(row_data.get(column_name, "")))
                self.precomputed_table.setItem(table_row, col_idx, item)

        header = self.precomputed_table.horizontalHeader()
        if header is not None and self.precomputed_table.columnCount() > 0:
            try:
                header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
                for col_idx in range(1, self.precomputed_table.columnCount() - 1):
                    header.setSectionResizeMode(col_idx, QHeaderView.ResizeMode.ResizeToContents)
                header.setSectionResizeMode(
                    self.precomputed_table.columnCount() - 1,
                    QHeaderView.ResizeMode.Stretch,
                )
            except Exception:
                header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
                for col_idx in range(1, self.precomputed_table.columnCount() - 1):
                    header.setSectionResizeMode(col_idx, QHeaderView.ResizeToContents)
                header.setSectionResizeMode(
                    self.precomputed_table.columnCount() - 1,
                    QHeaderView.Stretch,
                )

        if selected_row is not None:
            target_row = -1
            try:
                selected_row = int(selected_row)
            except Exception:
                selected_row = None
            if selected_row is not None:
                for table_row, source_row in enumerate(self._precomputed_row_indices):
                    if int(source_row) == selected_row:
                        target_row = table_row
                        break
            if target_row >= 0:
                self.precomputed_table.selectRow(target_row)
        elif len(row_dicts) == 1:
            self.precomputed_table.selectRow(0)

        self.precomputed_table.blockSignals(False)
        self._refresh_action_state()

    def set_hemisphere_mode(self, value: str) -> None:
        text = str(value or "separate").strip().upper()
        if text not in {"BOTH", "LH", "RH", "SEPARATE"}:
            text = "SEPARATE"
        if text == "BOTH":
            display = "Both"
        elif text == "SEPARATE":
            display = "Separate"
        else:
            display = text
        self.hemisphere_combo.blockSignals(True)
        if self.hemisphere_combo.findText(display) >= 0:
            self.hemisphere_combo.setCurrentText(display)
        self.hemisphere_combo.blockSignals(False)

    def set_surface_mesh(self, value: str) -> None:
        text = str(value or "fsaverage4").strip()
        if self.surface_mesh_combo.findText(text) < 0:
            text = "fsaverage4"
        self.surface_mesh_combo.blockSignals(True)
        self.surface_mesh_combo.setCurrentText(text)
        self.surface_mesh_combo.blockSignals(False)

    def set_surface_render_component_limit(self, value: int) -> None:
        try:
            count = int(value)
        except Exception:
            count = 1
        count = max(1, min(10, count))
        self.surface_render_count_spin.blockSignals(True)
        self.surface_render_count_spin.setRange(1, count)
        current = self.surface_render_count_spin.value()
        self.surface_render_count_spin.setValue(max(1, min(current, count)))
        self.surface_render_count_spin.blockSignals(False)

    def set_surface_render_component_count(self, value: int) -> None:
        try:
            count = int(value)
        except Exception:
            count = 1
        count = max(1, min(count, self.surface_render_count_spin.maximum()))
        self.surface_render_count_spin.blockSignals(True)
        self.surface_render_count_spin.setValue(count)
        self.surface_render_count_spin.blockSignals(False)

    def set_surface_procrustes_enabled(self, enabled: bool) -> None:
        self.surface_procrustes_check.blockSignals(True)
        self.surface_procrustes_check.setChecked(bool(enabled))
        self.surface_procrustes_check.blockSignals(False)

    def set_surface_procrustes_available(self, available: bool) -> None:
        self._surface_procrustes_available = bool(available)
        if not self._surface_procrustes_available:
            self.set_surface_procrustes_enabled(False)
        self._refresh_action_state()

    def set_scatter_rotation(self, value: str) -> None:
        text = str(value or "Default").strip()
        if self.scatter_rotation_combo.findText(text) < 0:
            text = "Default"
        self.scatter_rotation_combo.blockSignals(True)
        self.scatter_rotation_combo.setCurrentText(text)
        self.scatter_rotation_combo.blockSignals(False)

    def set_triangular_rgb(self, enabled: bool) -> None:
        self.triangular_rgb_check.blockSignals(True)
        self.triangular_rgb_check.setChecked(bool(enabled))
        self.triangular_rgb_check.blockSignals(False)
        self._refresh_action_state()

    def set_preload_matching_paths(self, enabled: bool) -> None:
        self.preload_matching_paths_check.blockSignals(True)
        self.preload_matching_paths_check.setChecked(bool(enabled))
        self.preload_matching_paths_check.blockSignals(False)

    def set_classification_fit_mode(self, value: str) -> None:
        text = str(value or "triangle").strip().lower()
        if text not in {"triangle", "square"}:
            text = "triangle"
        index = self.classification_fit_mode_combo.findData(text)
        if index < 0:
            index = self.classification_fit_mode_combo.findData("triangle")
        if index < 0:
            index = 0
        self.classification_fit_mode_combo.blockSignals(True)
        self.classification_fit_mode_combo.setCurrentIndex(index)
        self.classification_fit_mode_combo.blockSignals(False)

    def set_triangular_color_order(self, value: str) -> None:
        text = str(value or "RBG").strip().upper()
        if self.triangular_color_order_combo.findText(text) < 0:
            text = "RBG"
        self.triangular_color_order_combo.blockSignals(True)
        self.triangular_color_order_combo.setCurrentText(text)
        self.triangular_color_order_combo.blockSignals(False)

    def set_rgb_scalar_mode(self, value: str) -> None:
        text = str(value or "barycentric").strip().lower()
        if text not in {"barycentric", "principal_curve"}:
            text = "barycentric"
        index = self.rgb_scalar_mode_combo.findData(text)
        if index < 0:
            index = self.rgb_scalar_mode_combo.findData("barycentric")
        if index < 0:
            index = 0
        self.rgb_scalar_mode_combo.blockSignals(True)
        self.rgb_scalar_mode_combo.setCurrentIndex(index)
        self.rgb_scalar_mode_combo.blockSignals(False)

    def set_rgb_mask_strictness(self, value: float) -> None:
        try:
            number = max(0.0, min(5.0, float(value)))
        except Exception:
            number = 1.0
        self.rgb_mask_strictness_spin.blockSignals(True)
        self.rgb_mask_strictness_spin.setValue(number)
        self.rgb_mask_strictness_spin.blockSignals(False)

    def set_classification_surface_mesh(self, value: str) -> None:
        text = str(value or "fsaverage4").strip()
        if self.classification_surface_mesh_combo.findText(text) < 0:
            text = "fsaverage4"
        self.classification_surface_mesh_combo.blockSignals(True)
        self.classification_surface_mesh_combo.setCurrentText(text)
        self.classification_surface_mesh_combo.blockSignals(False)

    def set_classification_hemisphere_mode(self, value: str) -> None:
        text = str(value or "separate").strip().upper()
        if text not in {"BOTH", "LH", "RH", "SEPARATE"}:
            text = "SEPARATE"
        if text == "BOTH":
            display = "Both"
        elif text == "SEPARATE":
            display = "Separate"
        else:
            display = text
        self.classification_hemisphere_combo.blockSignals(True)
        if self.classification_hemisphere_combo.findText(display) >= 0:
            self.classification_hemisphere_combo.setCurrentText(display)
        self.classification_hemisphere_combo.blockSignals(False)

    def set_classification_adjacency_path(self, path) -> None:
        if path is None or not str(path).strip():
            self._has_classification_adjacency = False
            self.classification_adjacency_label.setText("Adjacency: none")
            return
        try:
            name = Path(path).name
        except Exception:
            name = str(path)
        self._has_classification_adjacency = True
        self.classification_adjacency_label.setText(f"Adjacency: {name}")

    def set_classification_colormap(self, value: str) -> None:
        text = str(value or "").strip()
        self.classification_colorbar_combo.blockSignals(True)
        if text and self.classification_colorbar_combo.findText(text) >= 0:
            self.classification_colorbar_combo.setCurrentText(text)
        elif self.classification_colorbar_combo.count() > 0:
            self.classification_colorbar_combo.setCurrentIndex(0)
        self.classification_colorbar_combo.blockSignals(False)

    def set_classification_component_options(self, component_count: int, selected_component="1") -> None:
        count = max(1, min(10, int(component_count)))
        selected = str(selected_component or "1").strip()
        self.classification_component_combo.blockSignals(True)
        self.classification_component_combo.clear()
        selected_index = 0
        for idx in range(count):
            value = str(idx + 1)
            self.classification_component_combo.addItem(f"C{idx + 1}", value)
            if value == selected:
                selected_index = idx
        self.classification_component_combo.setCurrentIndex(selected_index)
        self.classification_component_combo.blockSignals(False)

    def set_classification_axes(self, x_axis="gradient2", y_axis="gradient1", z_axis="none") -> None:
        self._set_classification_axis_combo(self.classification_x_axis_combo, x_axis, "gradient2")
        self._set_classification_axis_combo(self.classification_y_axis_combo, y_axis, "gradient1")
        self._set_classification_axis_combo(self.classification_z_axis_combo, z_axis, "none")
        self._refresh_classification_reference_visibility()

    def set_classification_ignore_parcel_options(self, lh_names, rh_names, *, selected_lh="", selected_rh="") -> None:
        self._set_ignore_parcel_combo(self.classification_ignore_lh_combo, lh_names, selected_lh)
        self._set_ignore_parcel_combo(self.classification_ignore_rh_combo, rh_names, selected_rh)

    @staticmethod
    def _set_classification_axis_combo(combo, value, fallback) -> None:
        selected = str(value or fallback).strip().lower()
        if not selected:
            selected = fallback
        index = combo.findData(selected)
        if index < 0:
            index = combo.findData(fallback)
        if index < 0:
            index = 0
        combo.blockSignals(True)
        combo.setCurrentIndex(index)
        combo.blockSignals(False)

    @staticmethod
    def _set_ignore_parcel_combo(combo, names, selected_name) -> None:
        selected = str(selected_name or "").strip()
        unique_names = []
        seen = set()
        for name in list(names or []):
            text = str(name or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            unique_names.append(text)
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("None", "")
        selected_index = 0
        for idx, text in enumerate(unique_names, start=1):
            combo.addItem(text, text)
            if text == selected:
                selected_index = idx
        combo.setCurrentIndex(selected_index)
        combo.blockSignals(False)

    def set_rotation_presets(self, presets) -> None:
        values = [str(value or "Default").strip() or "Default" for value in list(presets or [])]
        while len(values) < len(self._rotation_combos):
            values.append("Default")
        for combo, value in zip(self._rotation_combos, values):
            combo.blockSignals(True)
            if combo.findText(value) >= 0:
                combo.setCurrentText(value)
            elif combo.count() > 0:
                combo.setCurrentIndex(0)
            combo.blockSignals(False)
        self._refresh_rotation_visibility(self.component_count())

    def set_network_component_options(self, component_count: int, selected_component="all") -> None:
        count = max(1, min(10, int(component_count)))
        selected = str(selected_component or "all").strip().lower()
        ordinals = [
            "First",
            "Second",
            "Third",
            "Fourth",
            "Fifth",
            "Sixth",
            "Seventh",
            "Eighth",
            "Ninth",
            "Tenth",
        ]
        self.network_component_combo.blockSignals(True)
        self.network_component_combo.clear()
        self.network_component_combo.addItem("All", "all")
        selected_index = 0
        for idx in range(count):
            value = str(idx + 1)
            self.network_component_combo.addItem(f"{ordinals[idx]} (C{idx + 1})", value)
            if value == selected:
                selected_index = idx + 1
        self.network_component_combo.setCurrentIndex(selected_index)
        self.network_component_combo.blockSignals(False)

    def _refresh_rotation_visibility(self, component_count: int) -> None:
        count = max(1, min(10, int(component_count)))
        for idx, (label, combo) in enumerate(zip(self._rotation_labels, self._rotation_combos)):
            visible = idx < count
            label.setVisible(visible)
            combo.setVisible(visible)

    def _refresh_classification_reference_visibility(self) -> None:
        x_axis = str(self.classification_x_axis_combo.currentData() or "").strip().lower()
        y_axis = str(self.classification_y_axis_combo.currentData() or "").strip().lower()
        z_axis = str(self.classification_z_axis_combo.currentData() or "").strip().lower()
        visible = x_axis in {"gradient_ref1", "gradient_ref2", "gradient_ref3"} or y_axis in {
            "gradient_ref1",
            "gradient_ref2",
            "gradient_ref3",
        } or z_axis in {
            "gradient_ref1",
            "gradient_ref2",
            "gradient_ref3",
        }
        self.classification_ref_matrix_label.setVisible(visible)
        self.classification_ref_matrix_combo.setVisible(visible)

    def set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)
        self._refresh_action_state()

    def set_can_compute(self, enabled: bool) -> None:
        self._can_compute = bool(enabled)
        self._refresh_action_state()

    def set_has_results(self, enabled: bool) -> None:
        self._has_results = bool(enabled)
        self._refresh_action_state()

    def set_can_classify(self, enabled: bool) -> None:
        self._can_classify = bool(enabled)
        self._refresh_action_state()

    def _refresh_action_state(self) -> None:
        can_interact = not self._busy
        self.compute_button.setEnabled(self._can_compute and can_interact and not self._precomputed_mode)
        self.matrix_combo.setEnabled(can_interact)
        self.parcel_label_key_combo.setEnabled(can_interact and not self._precomputed_mode)
        self.parcel_name_key_combo.setEnabled(can_interact and not self._precomputed_mode)
        self.components_spin.setEnabled(can_interact and not self._precomputed_mode)
        self.parcellation_button.setEnabled(can_interact)
        self.hemisphere_combo.setEnabled(can_interact)
        self.surface_mesh_combo.setEnabled(can_interact)
        self.surface_render_count_spin.setEnabled(can_interact)
        self.surface_procrustes_check.setEnabled(can_interact and self._surface_procrustes_available)
        self.scatter_rotation_combo.setEnabled(can_interact)
        self.triangular_rgb_check.setEnabled(can_interact)
        self.preload_matching_paths_check.setEnabled(
            can_interact and self.triangular_rgb_check.isChecked()
        )
        self.classification_fit_mode_combo.setEnabled(can_interact)
        self.triangular_color_order_combo.setEnabled(can_interact)
        self.rgb_scalar_mode_combo.setEnabled(can_interact)
        self.rgb_mask_strictness_spin.setEnabled(can_interact)
        self.classification_surface_mesh_combo.setEnabled(can_interact)
        self.classification_hemisphere_combo.setEnabled(can_interact)
        self.classification_colorbar_combo.setEnabled(can_interact)
        self.classification_component_combo.setEnabled(can_interact)
        self.classification_x_axis_combo.setEnabled(can_interact)
        self.classification_y_axis_combo.setEnabled(can_interact)
        self.classification_z_axis_combo.setEnabled(can_interact)
        self.classification_ref_matrix_combo.setEnabled(
            can_interact
            and self.classification_ref_matrix_combo.count() > 0
            and self.classification_ref_matrix_combo.currentData() is not None
        )
        self.classification_ignore_lh_combo.setEnabled(can_interact)
        self.classification_ignore_rh_combo.setEnabled(can_interact)
        self.classification_adjacency_button.setEnabled(can_interact)
        self.classification_adjacency_remove_button.setEnabled(
            can_interact and self._has_classification_adjacency
        )
        self.precomputed_table.setEnabled(can_interact and self._precomputed_mode)
        self.precomputed_confirm_button.setEnabled(
            can_interact
            and self._precomputed_mode
            and self.selected_precomputed_row_index() is not None
        )
        self.network_component_combo.setEnabled(can_interact)
        for combo in self._rotation_combos:
            combo.setEnabled(can_interact)
        self.save_button.setEnabled(self._has_results and can_interact)
        self.render_3d_button.setEnabled(self._has_results and can_interact)
        self.classify_button.setEnabled(
            self._has_results
            and self._can_classify
            and can_interact
        )
        self.render_network_button.setEnabled(self._has_results and can_interact)

    def _on_precomputed_table_selection_changed(self):
        self._refresh_action_state()

    def _on_precomputed_table_double_clicked(self, _item):
        self._trigger_precomputed_row_confirm()

    def set_theme(self, theme_name="Dark"):
        self._theme_name, style = _panel_dialog_stylesheet(
            theme_name,
            control_selector="QPushButton, QComboBox, QSpinBox, QTableWidget",
            include_groupbox=True,
            include_tabs=True,
            include_progress=True,
            checkbox_indicator_selector="QCheckBox#triangularRgbCheck",
        )
        self.setStyleSheet(style)
