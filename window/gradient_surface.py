#!/usr/bin/env python3
"""Gradient fsaverage surface rendering dialog."""

from pathlib import Path

import nibabel as nib
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
except Exception:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from nilearn import datasets, plotting, surface

try:
    from PyQt6.QtWidgets import QFileDialog, QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout
except Exception:
    from PyQt5.QtWidgets import QFileDialog, QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

try:
    from window.shared.theme import dialog_theme_stylesheet as _shared_dialog_theme_stylesheet
except Exception:
    from mrsi_viewer.window.shared.theme import dialog_theme_stylesheet as _shared_dialog_theme_stylesheet


def _dialog_theme_stylesheet(theme_name="Dark"):
    return _shared_dialog_theme_stylesheet(theme_name)


class GradientSurfaceDialog(QDialog):
    """Interactive Nilearn surface viewer for 3D/4D gradient maps."""

    _surface_assets_cache = None

    def __init__(
        self,
        volume_img,
        title="Gradient Components",
        parent=None,
        cmap=None,
        cmap_name="spectrum_fsl",
        theme_name="Dark",
        hemisphere_mode="both",
        fsaverage_mesh="fsaverage4",
        surface_style="smooth",
        parcellation_img=None,
        support_mask_img=None,
        parcel_boundary_color="black",
        parcel_boundary_linewidth=0.45,
    ):
        super().__init__(parent)
        self._img = nib.as_closest_canonical(volume_img)
        self._data = np.asarray(self._img.get_fdata(), dtype=float)
        if self._data.ndim == 3:
            self._data = self._data[..., np.newaxis]
        if self._data.ndim != 4:
            raise ValueError(f"Expected 3D or 4D volume data. Got shape {self._data.shape}.")

        self._n_components = self._data.shape[3]
        self._title = title
        self._cmap_name = str(cmap_name or "spectrum_fsl")
        self._cmap = cmap if cmap is not None else self._default_cmap(self._cmap_name)
        self._theme_name = "Dark"
        self._hemisphere_mode = self._normalize_hemisphere_mode(hemisphere_mode)
        self._fsaverage_mesh = self._normalize_fsaverage_mesh(fsaverage_mesh)
        self._surface_style = self._normalize_surface_style(surface_style)
        self._parcellation_img = self._coerce_parcellation_img(parcellation_img)
        self._parcellation_data = self._parcellation_data_array(self._parcellation_img)
        self._support_img = self._coerce_parcellation_img(support_mask_img)
        self._parcel_boundary_color = str(parcel_boundary_color or "black")
        try:
            self._parcel_boundary_linewidth = max(0.05, float(parcel_boundary_linewidth))
        except Exception:
            self._parcel_boundary_linewidth = 0.45
        self.setWindowTitle(title)

        self.figure = Figure(
            figsize=(20, max(3.2 * self._n_components, 4.0)),
            constrained_layout=True,
        )
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.info_label = QLabel(
            "Views per component: "
            + self._hemisphere_label_text()
            + f" | Mesh: {self._fsaverage_mesh}"
            + f" | Style: {self._surface_style_label()}"
            + f" | Cmap: {self._cmap_name}"
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

    def set_theme(self, theme_name="Dark"):
        theme, style = _dialog_theme_stylesheet(theme_name)
        self._theme_name = theme
        self.setStyleSheet(style)

    @staticmethod
    def _default_cmap(cmap_name):
        try:
            try:
                from mrsitoolbox.graphplot.colorbar import ColorBar
            except ImportError as exc:
                raise ImportError(
                    "Gradient surface rendering requires the current mrsitoolbox pip package. "
                    "Install or upgrade it with: pip install --upgrade mrsitoolbox"
                ) from exc

            colorbar = ColorBar()
            if cmap_name == "spectrum_fsl":
                try:
                    return colorbar.load_fsl_cmap(cmap_name)
                except Exception:
                    return colorbar.bars(cmap_name)
            try:
                return colorbar.load_fsl_cmap(cmap_name)
            except Exception:
                return colorbar.bars(cmap_name)
        except Exception:
            return "viridis"

    @classmethod
    def from_array(
        cls,
        volume_data,
        affine=None,
        title="Gradient Components",
        parent=None,
        cmap=None,
        cmap_name="spectrum_fsl",
        theme_name="Dark",
        hemisphere_mode="both",
        fsaverage_mesh="fsaverage4",
        surface_style="smooth",
        parcellation_img=None,
        support_mask_img=None,
        parcel_boundary_color="black",
        parcel_boundary_linewidth=0.45,
    ):
        """Create dialog from ndarray + affine."""
        arr = np.asarray(volume_data, dtype=float)
        if arr.ndim not in (3, 4):
            raise ValueError(f"Expected 3D or 4D array. Got shape {arr.shape}.")
        if affine is None:
            affine = np.eye(4)
        img = nib.Nifti1Image(arr, affine)
        return cls(
            img,
            title=title,
            parent=parent,
            cmap=cmap,
            cmap_name=cmap_name,
            theme_name=theme_name,
            hemisphere_mode=hemisphere_mode,
            fsaverage_mesh=fsaverage_mesh,
            surface_style=surface_style,
            parcellation_img=parcellation_img,
            support_mask_img=support_mask_img,
            parcel_boundary_color=parcel_boundary_color,
            parcel_boundary_linewidth=parcel_boundary_linewidth,
        )

    @staticmethod
    def _normalize_hemisphere_mode(value):
        text = str(value or "both").strip().lower()
        if text not in {"both", "lh", "rh"}:
            text = "both"
        return text

    @staticmethod
    def _normalize_fsaverage_mesh(value):
        text = str(value or "fsaverage4").strip().lower()
        valid = {"fsaverage3", "fsaverage4", "fsaverage5", "fsaverage6", "fsaverage7"}
        if text not in valid:
            text = "fsaverage4"
        return text

    @staticmethod
    def _normalize_surface_style(value):
        text = str(value or "smooth").strip().lower().replace("-", "_").replace(" ", "_")
        if text in {"parcel", "parcels", "parcel_boundaries", "parcel_boundary", "parcel_outline", "parcel_outlines"}:
            return "parcel_outlines"
        return "smooth"

    def _surface_style_label(self):
        if self._surface_style == "parcel_outlines":
            return "Parcel outlines"
        return "Smooth"

    @staticmethod
    def _coerce_parcellation_img(parcellation_img):
        if parcellation_img is None:
            return None
        try:
            if isinstance(parcellation_img, (str, Path)):
                parcellation_img = nib.load(str(parcellation_img))
            return nib.as_closest_canonical(parcellation_img)
        except Exception:
            return None

    @staticmethod
    def _parcellation_data_array(parcellation_img):
        if parcellation_img is None:
            return None
        try:
            data = np.asarray(parcellation_img.get_fdata(), dtype=float)
        except Exception:
            return None
        if data.ndim != 3:
            return None
        return data

    def _hemisphere_label_text(self):
        if self._hemisphere_mode == "lh":
            return "LH lateral | LH medial"
        if self._hemisphere_mode == "rh":
            return "RH medial | RH lateral"
        return "LH lateral | LH medial | RH medial | RH lateral"

    @classmethod
    def _get_surface_assets(cls, fsaverage_mesh="fsaverage4"):
        """Load and cache fsaverage meshes/background maps once."""
        mesh_name = cls._normalize_fsaverage_mesh(fsaverage_mesh)
        if cls._surface_assets_cache is None:
            cls._surface_assets_cache = {}
        if mesh_name in cls._surface_assets_cache:
            return cls._surface_assets_cache[mesh_name]

        try:
            fsaverage = datasets.fetch_surf_fsaverage(mesh=mesh_name)
        except Exception:
            try:
                fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage4")
                mesh_name = "fsaverage4"
            except Exception:
                fsaverage = datasets.fetch_surf_fsaverage()
                mesh_name = "fsaverage4"

        mesh_left = surface.load_surf_mesh(fsaverage.pial_left)
        mesh_right = surface.load_surf_mesh(fsaverage.pial_right)
        try:
            white_left = surface.load_surf_mesh(fsaverage.white_left)
            white_right = surface.load_surf_mesh(fsaverage.white_right)
        except Exception:
            white_left = mesh_left
            white_right = mesh_right
        sulc_left = surface.load_surf_data(fsaverage.sulc_left)
        sulc_right = surface.load_surf_data(fsaverage.sulc_right)

        cls._surface_assets_cache[mesh_name] = {
            "mesh_left": mesh_left,
            "mesh_right": mesh_right,
            "white_left": white_left,
            "white_right": white_right,
            "sulc_left": sulc_left,
            "sulc_right": sulc_right,
        }
        return cls._surface_assets_cache[mesh_name]

    @staticmethod
    def _compute_display_range(data_4d):
        finite = data_4d[np.isfinite(data_4d)]
        if finite.size == 0:
            return -1.0, 1.0, None
        nonzero = finite[finite != 0]
        scale_data = nonzero if nonzero.size else finite
        vmin = float(np.percentile(scale_data, 2))
        vmax = float(np.percentile(scale_data, 98))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
            vmin = float(np.nanmin(scale_data))
            vmax = float(np.nanmax(scale_data))
        if np.isclose(vmin, vmax):
            vmin -= 1.0
            vmax += 1.0
        threshold = None
        if nonzero.size:
            thr = float(np.percentile(np.abs(nonzero), 2))
            if np.isfinite(thr) and thr > 0:
                threshold = thr
        return vmin, vmax, threshold

    @staticmethod
    def _mesh_arrays(mesh):
        if hasattr(mesh, "coordinates") and hasattr(mesh, "faces"):
            return np.asarray(mesh.coordinates, dtype=float), np.asarray(mesh.faces, dtype=int)
        if isinstance(mesh, (tuple, list)) and len(mesh) >= 2:
            return np.asarray(mesh[0], dtype=float), np.asarray(mesh[1], dtype=int)
        raise ValueError("Unsupported fsaverage surface mesh format.")

    @staticmethod
    def _vol_to_surf_robust(img, mesh, *, radius, interpolation, inner_mesh=None):
        methods = (interpolation, "nearest_most_frequent", "linear")
        seen_methods = set()
        for method in methods:
            if method in seen_methods:
                continue
            seen_methods.add(method)
            if inner_mesh is not None:
                try:
                    return surface.vol_to_surf(
                        img,
                        mesh,
                        inner_mesh=inner_mesh,
                        interpolation=method,
                    )
                except Exception:
                    pass
            try:
                return surface.vol_to_surf(
                    img,
                    mesh,
                    radius=radius,
                    interpolation=method,
                )
            except Exception:
                continue
        return surface.vol_to_surf(img, mesh, radius=radius)

    @classmethod
    def _nearest_source_indices(cls, source_coords, target_coords):
        source = np.asarray(source_coords, dtype=float)
        target = np.asarray(target_coords, dtype=float)
        if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
            return np.zeros(target.shape[0], dtype=int)
        if source.shape[0] == 0 or target.shape[0] == 0:
            return np.zeros(target.shape[0], dtype=int)
        try:
            from scipy.spatial import cKDTree

            _dist, indices = cKDTree(source).query(target, k=1)
            return np.asarray(indices, dtype=int)
        except Exception:
            max_pairs = 2_000_000
            chunk = max(1, int(max_pairs // max(source.shape[0], 1)))
            nearest = np.zeros(target.shape[0], dtype=int)
            for start in range(0, target.shape[0], chunk):
                stop = min(target.shape[0], start + chunk)
                diff = target[start:stop, np.newaxis, :] - source[np.newaxis, :, :]
                dist_sq = np.einsum("ijk,ijk->ij", diff, diff)
                nearest[start:stop] = np.argmin(dist_sq, axis=1)
            return nearest

    @classmethod
    def _filled_label_texture(cls, mesh, label_texture):
        if label_texture is None:
            return None
        coords, _faces = cls._mesh_arrays(mesh)
        labels = np.asarray(label_texture, dtype=float).reshape(-1)
        if labels.shape[0] != coords.shape[0]:
            return labels
        finite_labels = np.isfinite(labels)
        label_values = np.zeros(labels.shape, dtype=np.int64)
        label_values[finite_labels] = np.rint(labels[finite_labels]).astype(np.int64, copy=False)
        source_mask = finite_labels & (label_values > 0)
        target_mask = ~source_mask
        if not np.any(source_mask) or not np.any(target_mask):
            return labels
        filled = np.asarray(label_values, dtype=float)
        nearest = cls._nearest_source_indices(coords[source_mask], coords[target_mask])
        filled[target_mask] = label_values[source_mask][nearest]
        return filled

    @classmethod
    def _filled_surface_texture(cls, mesh, texture, *, label_texture=None, fill_mask=None):
        coords, _faces = cls._mesh_arrays(mesh)
        values = np.asarray(texture, dtype=float).reshape(-1)
        if values.shape[0] != coords.shape[0]:
            return values

        if fill_mask is None:
            eligible = np.ones(values.shape, dtype=bool)
        else:
            eligible = np.asarray(fill_mask, dtype=bool).reshape(-1)
            if eligible.shape != values.shape:
                eligible = np.ones(values.shape, dtype=bool)

        filled = np.asarray(values, dtype=float).copy()
        finite_values = np.isfinite(filled)
        if label_texture is not None:
            labels = np.asarray(label_texture, dtype=float).reshape(-1)
            if labels.shape == values.shape:
                finite_labels = np.isfinite(labels)
                label_values = np.zeros(labels.shape, dtype=np.int64)
                label_values[finite_labels] = np.rint(labels[finite_labels]).astype(np.int64, copy=False)
                valid_labels = finite_labels & (label_values > 0)
                for label in np.unique(label_values[eligible & valid_labels]):
                    same_label = label_values == int(label)
                    source_mask = eligible & same_label & finite_values
                    target_mask = eligible & same_label & ~np.isfinite(filled)
                    if not np.any(source_mask) or not np.any(target_mask):
                        continue
                    nearest = cls._nearest_source_indices(coords[source_mask], coords[target_mask])
                    filled[target_mask] = filled[source_mask][nearest]

        source_mask = eligible & np.isfinite(filled)
        target_mask = eligible & ~np.isfinite(filled)
        if np.any(source_mask) and np.any(target_mask):
            nearest = cls._nearest_source_indices(coords[source_mask], coords[target_mask])
            filled[target_mask] = filled[source_mask][nearest]
        return filled

    @classmethod
    def _boundary_segments_from_labels(cls, mesh, vertex_labels):
        coords, faces = cls._mesh_arrays(mesh)
        labels = np.asarray(vertex_labels, dtype=float).reshape(-1)
        if coords.ndim != 2 or coords.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
            return np.zeros((0, 2, 3), dtype=float)
        if labels.shape[0] != coords.shape[0]:
            return np.zeros((0, 2, 3), dtype=float)

        finite_labels = np.isfinite(labels)
        label_values = np.zeros(labels.shape, dtype=np.int64)
        label_values[finite_labels] = np.rint(labels[finite_labels]).astype(np.int64, copy=False)
        valid = finite_labels & (label_values > 0)
        edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])).astype(np.int64, copy=False)
        in_bounds = np.all((edges >= 0) & (edges < coords.shape[0]), axis=1)
        edges = edges[in_bounds]
        if edges.size == 0:
            return np.zeros((0, 2, 3), dtype=float)

        keep = valid[edges[:, 0]] & valid[edges[:, 1]] & (label_values[edges[:, 0]] != label_values[edges[:, 1]])
        edges = edges[keep]
        if edges.size == 0:
            return np.zeros((0, 2, 3), dtype=float)
        edges = np.unique(np.sort(edges, axis=1), axis=0)
        return np.asarray(coords[edges], dtype=float)

    def _parcel_value_texture_from_labels(self, comp_img, label_texture):
        if self._parcellation_data is None or label_texture is None:
            return None
        try:
            comp_data = np.asarray(comp_img.get_fdata(), dtype=float)
        except Exception:
            return None
        if comp_data.shape != self._parcellation_data.shape:
            return None

        surface_labels = np.asarray(label_texture, dtype=float).reshape(-1)
        finite_surface = np.isfinite(surface_labels)
        surface_label_values = np.zeros(surface_labels.shape, dtype=np.int64)
        surface_label_values[finite_surface] = np.rint(surface_labels[finite_surface]).astype(np.int64, copy=False)
        valid_surface = finite_surface & (surface_label_values > 0)
        texture = np.full(surface_labels.shape, np.nan, dtype=float)
        if not np.any(valid_surface):
            return texture

        finite_parcels = np.isfinite(self._parcellation_data)
        parcel_labels = np.zeros(self._parcellation_data.shape, dtype=np.int64)
        parcel_labels[finite_parcels] = np.rint(
            self._parcellation_data[finite_parcels]
        ).astype(np.int64, copy=False)
        for label in np.unique(surface_label_values[valid_surface]):
            voxel_values = comp_data[parcel_labels == int(label)]
            voxel_values = voxel_values[np.isfinite(voxel_values)]
            if voxel_values.size == 0:
                continue
            texture[surface_label_values == int(label)] = float(np.nanmedian(voxel_values))
        return texture

    @staticmethod
    def _support_mask_from_texture(support_texture, label_texture=None):
        if support_texture is not None:
            support = np.asarray(support_texture, dtype=float).reshape(-1)
            return np.isfinite(support) & (support > 0.0)
        if label_texture is not None:
            labels = np.asarray(label_texture, dtype=float).reshape(-1)
            return np.isfinite(labels) & (np.rint(labels) > 0)
        return None

    def _surface_texture(self, comp_img, mesh, inner_mesh=None, label_texture=None, support_texture=None):
        if self._surface_style == "parcel_outlines":
            parcel_texture = self._parcel_value_texture_from_labels(comp_img, label_texture)
            if parcel_texture is not None:
                fill_mask = self._support_mask_from_texture(support_texture, label_texture)
                return self._filled_surface_texture(
                    mesh,
                    parcel_texture,
                    label_texture=label_texture,
                    fill_mask=fill_mask,
                )
        interpolation = "linear" if self._surface_style == "smooth" else "nearest_most_frequent"
        texture = self._vol_to_surf_robust(
            comp_img,
            mesh,
            radius=3.0,
            interpolation=interpolation,
            inner_mesh=inner_mesh,
        )
        fill_mask = self._support_mask_from_texture(support_texture, label_texture)
        if fill_mask is not None:
            texture = np.asarray(texture, dtype=float).copy()
            if texture.shape == fill_mask.shape:
                texture[~fill_mask] = np.nan
        return self._filled_surface_texture(
            mesh,
            texture,
            label_texture=label_texture,
            fill_mask=fill_mask,
        )

    def _surface_label_texture(self, mesh, inner_mesh=None):
        if self._surface_style != "parcel_outlines" or self._parcellation_img is None:
            return None
        texture = self._vol_to_surf_robust(
            self._parcellation_img,
            mesh,
            radius=3.0,
            interpolation="nearest_most_frequent",
            inner_mesh=inner_mesh,
        )
        return self._filled_label_texture(mesh, texture)

    def _surface_support_texture(self, mesh, inner_mesh=None):
        if self._support_img is None:
            return None
        texture = self._vol_to_surf_robust(
            self._support_img,
            mesh,
            radius=3.0,
            interpolation="nearest_most_frequent",
            inner_mesh=inner_mesh,
        )
        return self._filled_label_texture(mesh, texture)

    def _add_parcel_boundaries(self, ax, mesh, label_texture):
        if label_texture is None:
            return
        segments = self._boundary_segments_from_labels(mesh, label_texture)
        if segments.size == 0:
            return
        collection = Line3DCollection(
            segments,
            colors=self._parcel_boundary_color,
            linewidths=self._parcel_boundary_linewidth,
            alpha=0.95,
        )
        ax.add_collection3d(collection)

    def _render(self):
        self.figure.clear()
        assets = self._get_surface_assets(self._fsaverage_mesh)

        vmin, vmax, _threshold = self._compute_display_range(self._data)
        row_norm = Normalize(vmin=vmin, vmax=vmax)

        if self._hemisphere_mode == "lh":
            views = (
                ("left", "lateral", assets["mesh_left"], assets["sulc_left"], "LH Lateral"),
                ("left", "medial", assets["mesh_left"], assets["sulc_left"], "LH Medial"),
            )
        elif self._hemisphere_mode == "rh":
            views = (
                ("right", "medial", assets["mesh_right"], assets["sulc_right"], "RH Medial"),
                ("right", "lateral", assets["mesh_right"], assets["sulc_right"], "RH Lateral"),
            )
        else:
            views = (
                ("left", "lateral", assets["mesh_left"], assets["sulc_left"], "LH Lateral"),
                ("left", "medial", assets["mesh_left"], assets["sulc_left"], "LH Medial"),
                ("right", "medial", assets["mesh_right"], assets["sulc_right"], "RH Medial"),
                ("right", "lateral", assets["mesh_right"], assets["sulc_right"], "RH Lateral"),
            )
        n_view_cols = len(views)
        gs = self.figure.add_gridspec(
            self._n_components,
            n_view_cols + 1,
            width_ratios=[1.0] * n_view_cols + [0.05],
            wspace=0.02,
            hspace=0.08,
        )

        row_axes_all = []
        label_textures = {
            "left": self._surface_label_texture(assets["mesh_left"], assets.get("white_left")),
            "right": self._surface_label_texture(assets["mesh_right"], assets.get("white_right")),
        }
        support_textures = {
            "left": self._surface_support_texture(assets["mesh_left"], assets.get("white_left")),
            "right": self._surface_support_texture(assets["mesh_right"], assets.get("white_right")),
        }
        for comp_idx in range(self._n_components):
            comp_img = nib.Nifti1Image(self._data[..., comp_idx], self._img.affine)
            texture_left = self._surface_texture(
                comp_img,
                assets["mesh_left"],
                assets.get("white_left"),
                label_textures.get("left"),
                support_textures.get("left"),
            )
            texture_right = self._surface_texture(
                comp_img,
                assets["mesh_right"],
                assets.get("white_right"),
                label_textures.get("right"),
                support_textures.get("right"),
            )
            textures = {"left": texture_left, "right": texture_right}
            row_axes = []

            for view_idx, (hemi, view, mesh, bg_map, view_label) in enumerate(views):
                ax = self.figure.add_subplot(gs[comp_idx, view_idx], projection="3d")
                plotting.plot_surf_stat_map(
                    mesh,
                    textures[hemi],
                    hemi=hemi,
                    view=view,
                    bg_map=bg_map,
                    cmap=self._cmap,
                    colorbar=False,
                    vmin=vmin,
                    vmax=vmax,
                    threshold=None,
                    axes=ax,
                )
                self._add_parcel_boundaries(ax, mesh, label_textures.get(hemi))
                ax.set_title(view_label, fontsize=9, pad=2)
                row_axes.append(ax)
            row_axes_all.append(row_axes)

            cax = self.figure.add_subplot(gs[comp_idx, n_view_cols])
            mappable = ScalarMappable(norm=row_norm, cmap=self._cmap)
            mappable.set_array([])
            cbar = self.figure.colorbar(mappable, cax=cax)
            cbar.ax.tick_params(labelsize=8, length=2)
            cbar.set_label("Gradient", fontsize=10)

        prev_row_bottom = 0.99
        for comp_idx, row_axes in enumerate(row_axes_all):
            left_pos = row_axes[0].get_position()
            right_pos = row_axes[-1].get_position()
            center_x = (left_pos.x0 + right_pos.x1) / 2.0
            row_top = max(ax.get_position().y1 for ax in row_axes)
            row_bottom = min(ax.get_position().y0 for ax in row_axes)
            gap_above = max(0.0, prev_row_bottom - row_top)
            y_text = row_top + (0.28 * gap_above if gap_above > 0 else 0.006)
            y_text = min(y_text, 0.972)
            self.figure.text(
                center_x,
                y_text,
                f"Gradient {comp_idx + 1}",
                ha="center",
                va="bottom",
                fontsize=16,
            )
            prev_row_bottom = row_bottom

        self.canvas.draw_idle()

    def _save_figure(self):
        default_name = f"gradient_components_{self._n_components}.png"
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save gradient surface figure",
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


__all__ = ["GradientSurfaceDialog"]
