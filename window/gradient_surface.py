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

from nilearn import datasets, image, surface

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
    _default_sampling_radius = 6.0
    _isoline_level_count = 9
    _isoline_smoothing_iterations = 32
    _isoline_smoothing_relaxation = 0.55
    _isoline_display_scale = 1.012
    _isoline_zero_linewidth = 1.45
    _isoline_near_zero_linewidth = 0.8
    _isoline_linewidth_graduation = 1.35
    _isoline_linewidth_min = 0.2
    _isoline_linewidth_max = 3.0
    _parcel_outline_smoothing_iterations = 2
    _parcel_outline_smoothing_max = 32
    _parcel_outline_subdivision_max = 3

    def __init__(
        self,
        volume_img,
        title="Gradient Components",
        parent=None,
        cmap=None,
        cmap_name="spectrum_fsl",
        color_scale_min=None,
        color_scale_max=None,
        theme_name="Dark",
        hemisphere_mode="both",
        fsaverage_mesh="fsaverage4",
        surface_style="smooth",
        parcellation_img=None,
        support_mask_img=None,
        mask_img=None,
        sampling_radius=None,
        contour_count=None,
        contour_smoothing=None,
        contour_min_linewidth=None,
        contour_width_by_value=False,
        parcel_outline_smoothing=None,
        show_parcel_outlines=True,
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
        self._color_scale_min, self._color_scale_max = self._normalize_color_scale_limits(
            color_scale_min,
            color_scale_max,
        )
        self._theme_name = "Dark"
        self._hemisphere_mode = self._normalize_hemisphere_mode(hemisphere_mode)
        self._fsaverage_mesh = self._normalize_fsaverage_mesh(fsaverage_mesh)
        self._surface_style = self._normalize_surface_style(surface_style)
        self._parcellation_img = self._coerce_parcellation_img(parcellation_img)
        self._parcellation_data = self._parcellation_data_array(self._parcellation_img)
        self._support_img = self._coerce_parcellation_img(support_mask_img)
        self._mask_img = self._mask_to_gradient_grid(mask_img)
        self._sampling_radius = self._normalize_sampling_radius(sampling_radius)
        self._contour_count = self._normalize_contour_count(contour_count)
        self._contour_smoothing_iterations = self._normalize_contour_smoothing(contour_smoothing)
        self._contour_min_linewidth = self._normalize_contour_min_linewidth(contour_min_linewidth)
        self._contour_width_by_value = bool(contour_width_by_value)
        self._parcel_outline_smoothing = self._normalize_parcel_outline_smoothing(
            parcel_outline_smoothing
        )
        self._show_parcel_outlines = bool(show_parcel_outlines)
        self._parcel_boundary_color = str(parcel_boundary_color or "black")
        try:
            self._parcel_boundary_linewidth = max(0.05, float(parcel_boundary_linewidth))
        except Exception:
            self._parcel_boundary_linewidth = 0.45
        self._parcel_boundary_segments_cache = {}
        self.setWindowTitle(title)

        self.figure = Figure(
            figsize=(20, max(3.2 * self._n_components, 4.0)),
            constrained_layout=True,
        )
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = None

        self.info_label = QLabel(
            "Views per component: "
            + self._hemisphere_label_text()
            + f" | Mesh: {self._fsaverage_mesh}"
            + f" | Style: {self._surface_style_label()}"
            + f" | Radius: {self._sampling_radius:g} mm"
            + (
                f" | Contours: {self._contour_count}"
                + f" | Smooth: {self._contour_smoothing_iterations}"
                + f" | Min width: {self._contour_min_linewidth:g}"
                + (" | Width by value" if self._contour_width_by_value else "")
                if self._surface_style == "contours"
                else ""
            )
            + (
                (
                    f" | Outline smooth: {self._parcel_outline_smoothing}"
                    if self._show_parcel_outlines
                    else " | Outlines: off"
                )
                if self._surface_style == "parcel_outlines"
                else ""
            )
            + (" | Mask: on" if self._mask_img is not None else "")
            + (
                f" | Scale: {self._color_scale_min:g}..{self._color_scale_max:g}"
                if self._has_color_scale_override()
                else " | Scale: auto"
            )
            + f" | Cmap: {self._cmap_name}"
        )
        self.save_button = QPushButton("Save Figure")
        self.save_button.clicked.connect(self._save_figure)

        controls = QHBoxLayout()
        controls.addWidget(self.info_label, 1)
        controls.addWidget(self.save_button, 0)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
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
        color_scale_min=None,
        color_scale_max=None,
        theme_name="Dark",
        hemisphere_mode="both",
        fsaverage_mesh="fsaverage4",
        surface_style="smooth",
        parcellation_img=None,
        support_mask_img=None,
        mask_img=None,
        sampling_radius=None,
        contour_count=None,
        contour_smoothing=None,
        contour_min_linewidth=None,
        contour_width_by_value=False,
        parcel_outline_smoothing=None,
        show_parcel_outlines=True,
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
            color_scale_min=color_scale_min,
            color_scale_max=color_scale_max,
            theme_name=theme_name,
            hemisphere_mode=hemisphere_mode,
            fsaverage_mesh=fsaverage_mesh,
            surface_style=surface_style,
            parcellation_img=parcellation_img,
            support_mask_img=support_mask_img,
            mask_img=mask_img,
            sampling_radius=sampling_radius,
            contour_count=contour_count,
            contour_smoothing=contour_smoothing,
            contour_min_linewidth=contour_min_linewidth,
            contour_width_by_value=contour_width_by_value,
            parcel_outline_smoothing=parcel_outline_smoothing,
            show_parcel_outlines=show_parcel_outlines,
            parcel_boundary_color=parcel_boundary_color,
            parcel_boundary_linewidth=parcel_boundary_linewidth,
        )

    @classmethod
    def render_static(
        cls,
        volume_img,
        output_path,
        *,
        affine=None,
        cmap=None,
        cmap_name="spectrum_fsl",
        color_scale_min=None,
        color_scale_max=None,
        hemisphere_mode="both",
        fsaverage_mesh="fsaverage4",
        surface_style="smooth",
        parcellation_img=None,
        support_mask_img=None,
        mask_img=None,
        sampling_radius=None,
        contour_count=None,
        contour_smoothing=None,
        contour_min_linewidth=None,
        contour_width_by_value=False,
        parcel_outline_smoothing=None,
        show_parcel_outlines=True,
        parcel_boundary_color="black",
        parcel_boundary_linewidth=0.45,
        title="Gradient Components",
    ):
        """Render fsaverage surfaces directly to a file without creating a dialog."""
        if not hasattr(volume_img, "get_fdata"):
            if affine is None:
                affine = np.eye(4)
            volume_img = nib.Nifti1Image(np.asarray(volume_img, dtype=float), affine)
        renderer = cls.__new__(cls)
        renderer._img = nib.as_closest_canonical(volume_img)
        renderer._data = np.asarray(renderer._img.get_fdata(), dtype=float)
        if renderer._data.ndim == 3:
            renderer._data = renderer._data[..., np.newaxis]
        if renderer._data.ndim != 4:
            raise ValueError(f"Expected 3D or 4D volume data. Got shape {renderer._data.shape}.")
        renderer._n_components = renderer._data.shape[3]
        renderer._title = str(title or "Gradient Components")
        renderer._cmap_name = str(cmap_name or "spectrum_fsl")
        renderer._cmap = cmap if cmap is not None else renderer._default_cmap(renderer._cmap_name)
        renderer._color_scale_min, renderer._color_scale_max = renderer._normalize_color_scale_limits(
            color_scale_min,
            color_scale_max,
        )
        renderer._hemisphere_mode = renderer._normalize_hemisphere_mode(hemisphere_mode)
        renderer._fsaverage_mesh = renderer._normalize_fsaverage_mesh(fsaverage_mesh)
        renderer._surface_style = renderer._normalize_surface_style(surface_style)
        renderer._parcellation_img = renderer._coerce_parcellation_img(parcellation_img)
        renderer._parcellation_data = renderer._parcellation_data_array(renderer._parcellation_img)
        renderer._support_img = renderer._coerce_parcellation_img(support_mask_img)
        renderer._mask_img = renderer._mask_to_gradient_grid(mask_img)
        renderer._sampling_radius = renderer._normalize_sampling_radius(sampling_radius)
        renderer._contour_count = renderer._normalize_contour_count(contour_count)
        renderer._contour_smoothing_iterations = renderer._normalize_contour_smoothing(contour_smoothing)
        renderer._contour_min_linewidth = renderer._normalize_contour_min_linewidth(contour_min_linewidth)
        renderer._contour_width_by_value = bool(contour_width_by_value)
        renderer._parcel_outline_smoothing = renderer._normalize_parcel_outline_smoothing(
            parcel_outline_smoothing
        )
        renderer._show_parcel_outlines = bool(show_parcel_outlines)
        renderer._parcel_boundary_color = str(parcel_boundary_color or "black")
        try:
            renderer._parcel_boundary_linewidth = max(0.05, float(parcel_boundary_linewidth))
        except Exception:
            renderer._parcel_boundary_linewidth = 0.45
        renderer._parcel_boundary_segments_cache = {}

        assets = cls._get_surface_assets(renderer._fsaverage_mesh)
        vmin, vmax, _threshold = renderer._compute_display_range(
            renderer._data,
            renderer._color_scale_min,
            renderer._color_scale_max,
        )
        row_norm = Normalize(vmin=vmin, vmax=vmax)
        if renderer._hemisphere_mode == "lh":
            views = (
                ("left", "lateral", assets["mesh_left"], assets["sulc_left"], "LH Lateral"),
                ("left", "medial", assets["mesh_left"], assets["sulc_left"], "LH Medial"),
            )
        elif renderer._hemisphere_mode == "rh":
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

        figure = Figure(
            figsize=(20, max(3.2 * renderer._n_components, 4.0)),
            constrained_layout=True,
        )
        n_view_cols = len(views)
        grid = figure.add_gridspec(
            renderer._n_components,
            n_view_cols + 1,
            width_ratios=[1.0] * n_view_cols + [0.05],
            wspace=0.02,
            hspace=0.08,
        )
        label_textures = {
            "left": renderer._surface_label_texture(assets["mesh_left"], assets.get("white_left")),
            "right": renderer._surface_label_texture(assets["mesh_right"], assets.get("white_right")),
        }
        support_textures = {
            "left": renderer._surface_support_texture(assets["mesh_left"], assets.get("white_left")),
            "right": renderer._surface_support_texture(assets["mesh_right"], assets.get("white_right")),
        }
        mask_textures = {
            "left": renderer._surface_mask_texture(assets["mesh_left"], assets.get("white_left")),
            "right": renderer._surface_mask_texture(assets["mesh_right"], assets.get("white_right")),
        }
        row_axes_all = []
        for comp_idx in range(renderer._n_components):
            comp_img = nib.Nifti1Image(renderer._data[..., comp_idx], renderer._img.affine)
            textures = {
                "left": renderer._surface_texture(
                    comp_img,
                    assets["mesh_left"],
                    assets.get("white_left"),
                    label_textures.get("left"),
                    support_textures.get("left"),
                    mask_textures.get("left"),
                ),
                "right": renderer._surface_texture(
                    comp_img,
                    assets["mesh_right"],
                    assets.get("white_right"),
                    label_textures.get("right"),
                    support_textures.get("right"),
                    mask_textures.get("right"),
                ),
            }
            display_norm = row_norm
            contour_levels = np.asarray([], dtype=float)
            if renderer._surface_style == "contours":
                for hemi_name, mesh_key in (("left", "mesh_left"), ("right", "mesh_right")):
                    _coords, faces = renderer._mesh_arrays(assets[mesh_key])
                    textures[hemi_name] = renderer._smooth_surface_texture(
                        textures[hemi_name],
                        faces,
                        iterations=renderer._contour_smoothing_iterations,
                    )
                contour_levels = renderer._shared_isoline_levels(
                    textures.values(),
                    level_count=renderer._contour_count,
                )
                if contour_levels.size and not renderer._has_color_scale_override():
                    bound = float(np.max(np.abs(contour_levels)))
                    if np.isfinite(bound) and bound > np.finfo(float).eps:
                        display_norm = Normalize(vmin=-bound, vmax=bound)
            row_axes = []
            for view_idx, (hemi, view, mesh, bg_map, view_label) in enumerate(views):
                ax = figure.add_subplot(grid[comp_idx, view_idx], projection="3d")
                display_mesh = renderer._display_mesh_for_style(renderer._surface_style, hemi, mesh, assets)
                renderer._plot_scalar_surface(
                    ax,
                    display_mesh,
                    textures[hemi],
                    bg_map,
                    display_norm,
                    hemi=hemi,
                    view=view,
                    title=view_label,
                    show_values=renderer._surface_style != "contours",
                )
                if renderer._surface_style == "contours":
                    renderer._add_surface_contours(
                        ax,
                        display_mesh,
                        textures[hemi],
                        display_norm,
                        contour_levels,
                        hemi=hemi,
                        view=view,
                    )
                if renderer._surface_style == "parcel_outlines" and renderer._show_parcel_outlines:
                    renderer._add_parcel_boundaries(
                        ax,
                        display_mesh,
                        label_textures.get(hemi),
                        hemi=hemi,
                        view=view,
                    )
                row_axes.append(ax)
            row_axes_all.append(row_axes)
            cax = figure.add_subplot(grid[comp_idx, n_view_cols])
            mappable = ScalarMappable(norm=display_norm, cmap=renderer._cmap)
            mappable.set_array([])
            colorbar = figure.colorbar(mappable, cax=cax)
            colorbar.ax.tick_params(labelsize=8, length=2)
            colorbar.set_label("Gradient", fontsize=10)

        previous_bottom = 0.99
        for comp_idx, row_axes in enumerate(row_axes_all):
            left = row_axes[0].get_position()
            right = row_axes[-1].get_position()
            center_x = (left.x0 + right.x1) / 2.0
            row_top = max(axis.get_position().y1 for axis in row_axes)
            row_bottom = min(axis.get_position().y0 for axis in row_axes)
            gap_above = max(0.0, previous_bottom - row_top)
            y_text = min(row_top + (0.28 * gap_above if gap_above > 0 else 0.006), 0.972)
            figure.text(center_x, y_text, f"Gradient {comp_idx + 1}", ha="center", va="bottom", fontsize=16)
            previous_bottom = row_bottom

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(str(output_path), dpi=200, bbox_inches="tight")
        return output_path

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
        if text in {"contour", "contours", "contour_lines", "isoline", "isolines", "iso_contour", "iso_contours", "isocontour", "isocontours"}:
            return "contours"
        return "smooth"

    @classmethod
    def _normalize_sampling_radius(cls, value):
        if value is None:
            return float(cls._default_sampling_radius)
        try:
            radius = float(value)
        except Exception:
            return float(cls._default_sampling_radius)
        if not np.isfinite(radius):
            return float(cls._default_sampling_radius)
        return float(min(max(radius, 0.5), 12.0))

    @classmethod
    def _normalize_contour_count(cls, value):
        if value is None:
            return int(cls._isoline_level_count)
        try:
            count = int(value)
        except Exception:
            return int(cls._isoline_level_count)
        return int(min(max(count, 1), 31))

    @classmethod
    def _normalize_contour_smoothing(cls, value):
        if value is None:
            return int(cls._isoline_smoothing_iterations)
        try:
            count = int(value)
        except Exception:
            return int(cls._isoline_smoothing_iterations)
        return int(min(max(count, 0), 128))

    @classmethod
    def _normalize_contour_min_linewidth(cls, value):
        if value is None:
            return float(cls._isoline_linewidth_min)
        try:
            linewidth = float(value)
        except Exception:
            return float(cls._isoline_linewidth_min)
        if not np.isfinite(linewidth):
            return float(cls._isoline_linewidth_min)
        return float(min(max(linewidth, 0.05), cls._isoline_linewidth_max))

    @classmethod
    def _normalize_parcel_outline_smoothing(cls, value):
        if value is None:
            return int(cls._parcel_outline_smoothing_iterations)
        try:
            count = int(value)
        except Exception:
            return int(cls._parcel_outline_smoothing_iterations)
        return int(min(max(count, 0), cls._parcel_outline_smoothing_max))

    @staticmethod
    def _normalize_color_scale_bound(value):
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        try:
            bound = float(value)
        except Exception:
            return None
        if not np.isfinite(bound):
            return None
        return bound

    @classmethod
    def _normalize_color_scale_limits(cls, vmin, vmax):
        lower = cls._normalize_color_scale_bound(vmin)
        upper = cls._normalize_color_scale_bound(vmax)
        if lower is not None and upper is not None and lower > upper:
            lower, upper = upper, lower
        return lower, upper

    def _has_color_scale_override(self) -> bool:
        return self._color_scale_min is not None or self._color_scale_max is not None

    def _surface_style_label(self):
        if self._surface_style == "parcel_outlines":
            return "Parcel outlines"
        if self._surface_style == "contours":
            return "Iso-contours"
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
    def _coerce_mask_img(mask_img):
        if mask_img is None:
            return None
        if isinstance(mask_img, (str, Path)) and not str(mask_img).strip():
            return None
        try:
            if isinstance(mask_img, (str, Path)):
                mask_img = nib.load(str(mask_img))
            return nib.as_closest_canonical(mask_img)
        except Exception as exc:
            raise ValueError(f"Could not load mask image: {exc}") from exc

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

    @staticmethod
    def _mask_data_array(mask_img):
        if mask_img is None:
            return None
        try:
            data = np.asarray(mask_img.get_fdata(), dtype=float)
        except Exception as exc:
            raise ValueError(f"Could not read mask image data: {exc}") from exc
        if data.ndim == 4:
            data = data[..., 0]
        if data.ndim != 3:
            raise ValueError(f"Mask image must be 3D or 4D. Got shape {data.shape}.")
        return data

    def _mask_to_gradient_grid(self, mask_img):
        mask_img = self._coerce_mask_img(mask_img)
        if mask_img is None:
            return None
        mask_data = self._mask_data_array(mask_img)
        mask_img = nib.Nifti1Image(mask_data, mask_img.affine)
        target_shape = tuple(self._data.shape[:3])
        same_shape = tuple(mask_data.shape) == target_shape
        same_affine = np.allclose(mask_img.affine, self._img.affine, atol=1e-4)
        if same_shape and same_affine:
            return mask_img
        reference = nib.Nifti1Image(np.zeros(target_shape, dtype=np.float32), self._img.affine)
        return image.resample_to_img(mask_img, reference, interpolation="nearest")

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
            infl_left = surface.load_surf_mesh(fsaverage.infl_left)
            infl_right = surface.load_surf_mesh(fsaverage.infl_right)
        except Exception:
            infl_left = mesh_left
            infl_right = mesh_right
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
            "infl_left": infl_left,
            "infl_right": infl_right,
            "white_left": white_left,
            "white_right": white_right,
            "sulc_left": sulc_left,
            "sulc_right": sulc_right,
        }
        return cls._surface_assets_cache[mesh_name]

    @classmethod
    def _compute_display_range(cls, data_4d, color_scale_min=None, color_scale_max=None):
        manual_min, manual_max = cls._normalize_color_scale_limits(color_scale_min, color_scale_max)
        finite = data_4d[np.isfinite(data_4d)]
        if finite.size == 0:
            vmin, vmax = -1.0, 1.0
            if manual_min is not None:
                vmin = manual_min
            if manual_max is not None:
                vmax = manual_max
            if np.isclose(vmin, vmax):
                vmin -= 1.0
                vmax += 1.0
            return float(vmin), float(vmax), None
        nonzero = finite[finite != 0]
        scale_data = nonzero if nonzero.size else finite
        vmin = float(np.percentile(scale_data, 2))
        vmax = float(np.percentile(scale_data, 98))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or np.isclose(vmin, vmax):
            vmin = float(np.nanmin(scale_data))
            vmax = float(np.nanmax(scale_data))
        if manual_min is not None:
            vmin = manual_min
        if manual_max is not None:
            vmax = manual_max
        if vmin > vmax:
            vmin, vmax = vmax, vmin
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
        faces = np.asarray(faces, dtype=int)
        n_faces = int(faces.shape[0])
        if bg_map is None:
            return np.full((n_faces, 3), 0.72, dtype=float)
        bg = np.asarray(bg_map, dtype=float).reshape(-1)
        if bg.size == 0 or faces.size == 0 or bg.size <= int(np.max(faces)):
            return np.full((n_faces, 3), 0.72, dtype=float)
        finite = bg[np.isfinite(bg)]
        if finite.size == 0:
            return np.full((n_faces, 3), 0.72, dtype=float)
        bg_min = float(np.nanmin(finite))
        bg_max = float(np.nanmax(finite))
        if np.isclose(bg_min, bg_max):
            normalized = np.full(bg.shape, 0.72, dtype=float)
        else:
            normalized = 0.55 + 0.35 * ((bg - bg_min) / (bg_max - bg_min))
        face_gray = np.asarray(normalized[faces].mean(axis=1), dtype=float)
        return np.repeat(face_gray[:, np.newaxis], 3, axis=1)

    def _plot_scalar_surface(self, ax, mesh, vertex_values, bg_map, norm, *, hemi, view, title, show_values=True):
        coords, faces = self._mesh_arrays(mesh)
        values = np.asarray(vertex_values, dtype=float).reshape(-1)
        if values.shape[0] != coords.shape[0]:
            values = np.full(coords.shape[0], np.nan, dtype=float)

        if not show_values:
            ax.computed_zorder = False
        bg_face_rgb = self._background_face_gray(bg_map, faces)
        if not show_values:
            bg_face_rgb = np.clip(0.55 * bg_face_rgb + 0.45, 0.0, 1.0)
        surface_plot = ax.plot_trisurf(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            triangles=faces,
            linewidth=0.0,
            antialiased=False,
            shade=False,
        )
        try:
            surface_plot.set_rasterized(True)
        except Exception:
            pass
        face_values = np.asarray(values[faces], dtype=float)
        finite_face = np.any(np.isfinite(face_values), axis=1)
        face_scalar = np.full(faces.shape[0], np.nan, dtype=float)
        if np.any(finite_face):
            face_scalar[finite_face] = np.nanmean(face_values[finite_face], axis=1)

        face_rgba = np.ones((faces.shape[0], 4), dtype=float)
        face_rgba[:, :3] = bg_face_rgb
        finite_scalar = np.isfinite(face_scalar)
        if show_values and np.any(finite_scalar):
            mappable = ScalarMappable(norm=norm, cmap=self._cmap)
            mapped = np.asarray(mappable.to_rgba(face_scalar[finite_scalar]), dtype=float)
            face_rgba[finite_scalar, :3] = np.clip(
                0.94 * mapped[:, :3] + 0.06 * bg_face_rgb[finite_scalar],
                0.0,
                1.0,
            )
            face_rgba[finite_scalar, 3] = 1.0
        surface_plot.set_facecolors(face_rgba)
        surface_plot.set_edgecolors("none")

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
        ax.set_title(title, fontsize=9, pad=2)

    @staticmethod
    def _isoline_segments(coordinates, faces, values, level):
        coordinates = np.asarray(coordinates, dtype=float)
        faces = np.asarray(faces, dtype=int)
        values = np.asarray(values, dtype=float).reshape(-1)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
            return np.empty((0, 2, 3), dtype=float)
        if values.shape[0] != coordinates.shape[0]:
            return np.empty((0, 2, 3), dtype=float)
        in_bounds = np.all((faces >= 0) & (faces < coordinates.shape[0]), axis=1)
        faces = faces[in_bounds]
        if faces.size == 0:
            return np.empty((0, 2, 3), dtype=float)

        triangle_values = values[faces]
        triangle_coordinates = coordinates[faces]
        finite_faces = np.all(np.isfinite(triangle_values), axis=1)
        edge_pairs = ((0, 1), (1, 2), (2, 0))
        crossings = np.zeros((faces.shape[0], len(edge_pairs)), dtype=bool)
        crossing_points = np.zeros((faces.shape[0], len(edge_pairs), 3), dtype=float)
        for edge_index, (first, second) in enumerate(edge_pairs):
            first_value = triangle_values[:, first]
            second_value = triangle_values[:, second]
            delta = second_value - first_value
            crosses = finite_faces & (
                ((first_value <= level) & (second_value > level))
                | ((second_value <= level) & (first_value > level))
            )
            crosses &= np.abs(delta) > np.finfo(float).eps
            crossings[:, edge_index] = crosses
            fraction = np.zeros(delta.shape, dtype=float)
            fraction[crosses] = (level - first_value[crosses]) / delta[crosses]
            crossing_points[:, edge_index, :] = (
                triangle_coordinates[:, first, :]
                + fraction[:, np.newaxis]
                * (triangle_coordinates[:, second, :] - triangle_coordinates[:, first, :])
            )

        crossing_count = crossings.sum(axis=1)
        segment_parts = []
        for first_edge, second_edge in ((0, 1), (0, 2), (1, 2)):
            mask = (
                (crossing_count == 2)
                & crossings[:, first_edge]
                & crossings[:, second_edge]
            )
            if np.any(mask):
                segment_parts.append(
                    np.stack(
                        (
                            crossing_points[mask, first_edge, :],
                            crossing_points[mask, second_edge, :],
                        ),
                        axis=1,
                    )
                )
        if not segment_parts:
            return np.empty((0, 2, 3), dtype=float)
        return np.concatenate(segment_parts, axis=0)

    @classmethod
    def _surface_contour_segments(cls, mesh, vertex_values, levels):
        coords, faces = cls._mesh_arrays(mesh)
        values = np.asarray(vertex_values, dtype=float).reshape(-1)
        contours = []
        for level in np.asarray(levels, dtype=float).reshape(-1):
            if not np.isfinite(level):
                continue
            segments = cls._isoline_segments(coords, faces, values, float(level))
            if segments.size:
                contours.append((float(level), segments))
        return contours

    @classmethod
    def _shared_isoline_levels(cls, textures, level_count=None):
        finite_values = [
            np.asarray(texture, dtype=float)[np.isfinite(texture)]
            for texture in textures
            if np.any(np.isfinite(texture))
        ]
        if not finite_values:
            return np.asarray([], dtype=float)
        pooled = np.concatenate(finite_values)
        bound = float(np.nanpercentile(np.abs(pooled), 98.0))
        if not np.isfinite(bound) or bound <= np.finfo(float).eps:
            bound = float(np.nanmax(np.abs(pooled)))
        if not np.isfinite(bound) or bound <= np.finfo(float).eps:
            return np.asarray([], dtype=float)
        count = cls._normalize_contour_count(level_count)
        return np.linspace(-bound, bound, count + 2, dtype=float)[1:-1]

    def _isoline_linewidth(self, level, levels, graduation=None):
        levels = np.asarray(levels, dtype=float)
        graduation = self._isoline_linewidth_graduation if graduation is None else float(graduation)
        min_linewidth = getattr(self, "_contour_min_linewidth", self._isoline_linewidth_min)
        level_spacing = float(np.median(np.diff(levels))) if levels.size > 1 else 1.0
        is_zero = abs(float(level)) <= 0.25 * abs(level_spacing)
        if not bool(getattr(self, "_contour_width_by_value", False)):
            return float(np.clip(max(self._isoline_near_zero_linewidth, min_linewidth), min_linewidth, self._isoline_linewidth_max)), is_zero
        if is_zero:
            return float(max(self._isoline_zero_linewidth, min_linewidth)), True
        nonzero = np.abs(levels[np.abs(levels) > 0.25 * abs(level_spacing)])
        nonzero_magnitudes = np.unique(np.round(nonzero, 12))
        if nonzero_magnitudes.size == 0:
            return float(max(self._isoline_near_zero_linewidth, min_linewidth)), False
        magnitude_rank = int(np.argmin(np.abs(nonzero_magnitudes - abs(float(level)))))
        linewidth = self._isoline_near_zero_linewidth * (
            graduation**magnitude_rank if level > 0.0 else graduation ** (-magnitude_rank)
        )
        return float(np.clip(linewidth, min_linewidth, self._isoline_linewidth_max)), False

    @staticmethod
    def _visible_isoline_segments(segments, surface_coordinates, *, hemisphere, view):
        if segments.size == 0:
            return segments
        if view == "lateral":
            axis = 0
            direction = -1.0 if hemisphere == "left" else 1.0
        elif view == "medial":
            axis = 0
            direction = 1.0 if hemisphere == "left" else -1.0
        else:
            return segments
        surface_depth = direction * np.asarray(surface_coordinates, dtype=float)[:, axis]
        segment_depth = direction * np.asarray(segments, dtype=float).mean(axis=1)[:, axis]
        return segments[segment_depth >= float(np.median(surface_depth))]

    @classmethod
    def _mesh_vertex_adjacency(cls, vertex_count, faces):
        try:
            from scipy import sparse
        except Exception:
            return None
        try:
            vertex_count = int(vertex_count)
        except Exception:
            return None
        faces = np.asarray(faces, dtype=int)
        if vertex_count <= 0 or faces.ndim != 2 or faces.shape[1] != 3 or faces.size == 0:
            return None
        edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
        rows = np.concatenate((edges[:, 0], edges[:, 1]))
        columns = np.concatenate((edges[:, 1], edges[:, 0]))
        valid_edges = (
            (rows >= 0)
            & (rows < vertex_count)
            & (columns >= 0)
            & (columns < vertex_count)
        )
        rows = rows[valid_edges]
        columns = columns[valid_edges]
        if rows.size == 0:
            return None
        adjacency = sparse.coo_matrix(
            (np.ones(rows.size, dtype=float), (rows, columns)),
            shape=(vertex_count, vertex_count),
        ).tocsr()
        adjacency.data[:] = 1.0
        adjacency.eliminate_zeros()
        return adjacency

    @classmethod
    def _smooth_surface_texture(cls, values, faces, *, iterations=None, relaxation=None):
        values = np.asarray(values, dtype=float).reshape(-1)
        faces = np.asarray(faces, dtype=int)
        if values.size == 0 or faces.size == 0:
            return values.copy()
        iterations = cls._isoline_smoothing_iterations if iterations is None else int(iterations)
        relaxation = cls._isoline_smoothing_relaxation if relaxation is None else float(relaxation)
        adjacency = cls._mesh_vertex_adjacency(values.size, faces)
        if adjacency is None:
            return values.copy()
        finite = np.isfinite(values)
        smoothed = values.copy()
        finite_float = finite.astype(float)
        neighbor_count = np.asarray(adjacency @ finite_float, dtype=float).reshape(-1)
        valid_neighbors = finite & (neighbor_count > 0.0)
        relaxation = float(np.clip(relaxation, 0.0, 1.0))
        for _ in range(max(0, int(iterations))):
            weighted_sum = np.asarray(
                adjacency @ np.where(finite, smoothed, 0.0),
                dtype=float,
            ).reshape(-1)
            neighbor_mean = np.zeros(smoothed.shape, dtype=float)
            neighbor_mean[valid_neighbors] = weighted_sum[valid_neighbors] / neighbor_count[valid_neighbors]
            smoothed[valid_neighbors] = (
                (1.0 - relaxation) * smoothed[valid_neighbors]
                + relaxation * neighbor_mean[valid_neighbors]
            )
        return smoothed

    @classmethod
    def _display_mesh_for_style(cls, style, hemi, mesh, assets):
        if style in {"contours", "parcel_outlines"}:
            inflated = assets.get("infl_left" if hemi == "left" else "infl_right")
            if inflated is not None:
                return inflated
        return mesh

    def _add_surface_contours(self, ax, mesh, vertex_values, norm, levels, *, hemi, view):
        levels = np.asarray(levels, dtype=float).reshape(-1)
        if levels.size == 0:
            return
        coords, faces = self._mesh_arrays(mesh)
        center = np.nanmean(coords, axis=0)
        mappable = ScalarMappable(norm=norm, cmap=self._cmap)
        for level in levels:
            segments = self._isoline_segments(coords, faces, vertex_values, float(level))
            segments = self._visible_isoline_segments(segments, coords, hemisphere=hemi, view=view)
            if segments.size == 0:
                continue
            segments = center + self._isoline_display_scale * (segments - center)
            linewidth, is_zero = self._isoline_linewidth(float(level), levels)
            collection = Line3DCollection(
                segments,
                colors=[mappable.to_rgba(float(level))],
                linewidths=linewidth,
                linestyles=":" if is_zero else ("--" if level < 0.0 else "-"),
                alpha=0.9 if is_zero else float(np.clip(0.72 + 0.08 * linewidth, 0.72, 0.94)),
                zorder=10,
            )
            collection.set_zorder(10)
            ax.add_collection3d(collection)

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
    def _mode_filtered_label_texture(cls, mesh, label_texture, *, iterations=2):
        coords, faces = cls._mesh_arrays(mesh)
        labels = np.asarray(label_texture, dtype=float).reshape(-1)
        if labels.shape[0] != coords.shape[0]:
            return labels
        adjacency = cls._mesh_vertex_adjacency(coords.shape[0], faces)
        if adjacency is None:
            return labels

        finite_labels = np.isfinite(labels)
        cleaned = np.zeros(labels.shape, dtype=np.int64)
        cleaned[finite_labels] = np.rint(labels[finite_labels]).astype(np.int64, copy=False)
        valid = finite_labels & (cleaned > 0)
        if not np.any(valid):
            return labels

        indptr = adjacency.indptr
        indices = adjacency.indices
        for _ in range(max(0, int(iterations))):
            updated = cleaned.copy()
            for vertex in np.flatnonzero(valid):
                neighbors = indices[indptr[vertex] : indptr[vertex + 1]]
                neighbor_labels = cleaned[neighbors]
                neighbor_labels = neighbor_labels[neighbor_labels > 0]
                if neighbor_labels.size < 2:
                    continue
                values, counts = np.unique(neighbor_labels, return_counts=True)
                best_idx = int(np.argmax(counts))
                best_label = int(values[best_idx])
                best_count = int(counts[best_idx])
                current_count = int(np.sum(neighbor_labels == cleaned[vertex]))
                if best_label != int(cleaned[vertex]) and best_count >= max(2, current_count + 2):
                    updated[vertex] = best_label
            cleaned = updated
        output = np.asarray(labels, dtype=float).copy()
        output[valid] = cleaned[valid].astype(float)
        return output

    @classmethod
    def _remove_tiny_label_components(cls, mesh, label_texture, *, min_vertices=None):
        coords, faces = cls._mesh_arrays(mesh)
        labels = np.asarray(label_texture, dtype=float).reshape(-1)
        if labels.shape[0] != coords.shape[0]:
            return labels
        adjacency = cls._mesh_vertex_adjacency(coords.shape[0], faces)
        if adjacency is None:
            return labels

        finite_labels = np.isfinite(labels)
        cleaned = np.zeros(labels.shape, dtype=np.int64)
        cleaned[finite_labels] = np.rint(labels[finite_labels]).astype(np.int64, copy=False)
        valid = finite_labels & (cleaned > 0)
        if not np.any(valid):
            return labels
        if min_vertices is None:
            min_vertices = max(4, min(12, coords.shape[0] // 2000))
        min_vertices = max(1, int(min_vertices))
        if min_vertices <= 1:
            return labels

        indptr = adjacency.indptr
        indices = adjacency.indices
        for label in np.unique(cleaned[valid]):
            label = int(label)
            label_vertices = np.flatnonzero(cleaned == label)
            if label_vertices.size <= min_vertices:
                continue
            remaining = set(int(vertex) for vertex in label_vertices.tolist())
            while remaining:
                start = remaining.pop()
                component = [start]
                stack = [start]
                while stack:
                    vertex = stack.pop()
                    neighbors = indices[indptr[vertex] : indptr[vertex + 1]]
                    same_neighbors = [
                        int(neighbor)
                        for neighbor in neighbors
                        if int(neighbor) in remaining and int(cleaned[int(neighbor)]) == label
                    ]
                    for neighbor in same_neighbors:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
                        component.append(neighbor)
                if len(component) >= min_vertices:
                    continue
                component_mask = np.zeros(cleaned.shape, dtype=bool)
                component_mask[np.asarray(component, dtype=int)] = True
                neighbor_labels = []
                for vertex in component:
                    neighbors = indices[indptr[vertex] : indptr[vertex + 1]]
                    outside = neighbors[~component_mask[neighbors]]
                    outside_labels = cleaned[outside]
                    neighbor_labels.extend(outside_labels[outside_labels > 0].tolist())
                if not neighbor_labels:
                    continue
                values, counts = np.unique(np.asarray(neighbor_labels, dtype=np.int64), return_counts=True)
                cleaned[np.asarray(component, dtype=int)] = int(values[int(np.argmax(counts))])

        output = np.asarray(labels, dtype=float).copy()
        output[valid] = cleaned[valid].astype(float)
        return output

    @classmethod
    def _clean_label_texture_for_boundaries(cls, mesh, label_texture):
        if label_texture is None:
            return None
        cleaned = cls._filled_label_texture(mesh, label_texture)
        cleaned = cls._mode_filtered_label_texture(mesh, cleaned, iterations=2)
        cleaned = cls._remove_tiny_label_components(mesh, cleaned)
        cleaned = cls._mode_filtered_label_texture(mesh, cleaned, iterations=1)
        return cleaned

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
    def _boundary_edges_from_labels(cls, mesh, vertex_labels, *, return_label_pairs=False):
        coords, faces = cls._mesh_arrays(mesh)
        labels = np.asarray(vertex_labels, dtype=float).reshape(-1)
        if coords.ndim != 2 or coords.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
            empty_edges = np.zeros((0, 2), dtype=np.int64)
            if return_label_pairs:
                return empty_edges, np.zeros((0, 2), dtype=np.int64)
            return empty_edges
        if labels.shape[0] != coords.shape[0]:
            empty_edges = np.zeros((0, 2), dtype=np.int64)
            if return_label_pairs:
                return empty_edges, np.zeros((0, 2), dtype=np.int64)
            return empty_edges

        finite_labels = np.isfinite(labels)
        label_values = np.zeros(labels.shape, dtype=np.int64)
        label_values[finite_labels] = np.rint(labels[finite_labels]).astype(np.int64, copy=False)
        valid = finite_labels & (label_values > 0)
        edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])).astype(np.int64, copy=False)
        in_bounds = np.all((edges >= 0) & (edges < coords.shape[0]), axis=1)
        edges = edges[in_bounds]
        if edges.size == 0:
            empty_edges = np.zeros((0, 2), dtype=np.int64)
            if return_label_pairs:
                return empty_edges, np.zeros((0, 2), dtype=np.int64)
            return empty_edges

        keep = valid[edges[:, 0]] & valid[edges[:, 1]] & (label_values[edges[:, 0]] != label_values[edges[:, 1]])
        edges = edges[keep]
        if edges.size == 0:
            empty_edges = np.zeros((0, 2), dtype=np.int64)
            if return_label_pairs:
                return empty_edges, np.zeros((0, 2), dtype=np.int64)
            return empty_edges
        label_pairs = np.sort(label_values[edges], axis=1)
        edges = np.sort(edges, axis=1)
        unique = np.unique(np.column_stack((edges, label_pairs)), axis=0)
        edges = unique[:, :2].astype(np.int64, copy=False)
        label_pairs = unique[:, 2:].astype(np.int64, copy=False)
        if return_label_pairs:
            return edges, label_pairs
        return edges

    @classmethod
    def _boundary_segments_from_labels(cls, mesh, vertex_labels):
        coords, _faces = cls._mesh_arrays(mesh)
        edges = cls._boundary_edges_from_labels(mesh, vertex_labels)
        if edges.size == 0:
            return np.zeros((0, 2, 3), dtype=float)
        return np.asarray(coords[edges], dtype=float)

    @staticmethod
    def _edge_key(first, second):
        first = int(first)
        second = int(second)
        return (first, second) if first <= second else (second, first)

    @classmethod
    def _boundary_polylines_from_edges(cls, coords, edges):
        coords = np.asarray(coords, dtype=float)
        edges = np.asarray(edges, dtype=np.int64)
        if edges.ndim != 2 or edges.shape[1] != 2 or edges.size == 0:
            return []

        adjacency = {}
        unused = set()
        for first, second in edges:
            first = int(first)
            second = int(second)
            if first == second:
                continue
            key = cls._edge_key(first, second)
            if key in unused:
                continue
            unused.add(key)
            adjacency.setdefault(first, set()).add(second)
            adjacency.setdefault(second, set()).add(first)

        if not unused:
            return []

        def consume_path(start, next_vertex):
            line = [int(start), int(next_vertex)]
            unused.discard(cls._edge_key(start, next_vertex))
            previous = int(start)
            current = int(next_vertex)
            while current != int(start):
                neighbors = adjacency.get(current, set())
                if len(neighbors) != 2:
                    break
                candidates = [
                    int(neighbor)
                    for neighbor in sorted(neighbors)
                    if int(neighbor) != previous
                    and cls._edge_key(current, int(neighbor)) in unused
                ]
                if not candidates:
                    break
                next_candidate = candidates[0]
                unused.discard(cls._edge_key(current, next_candidate))
                line.append(next_candidate)
                previous, current = current, next_candidate
            return line

        polylines = []
        starts = sorted(vertex for vertex, neighbors in adjacency.items() if len(neighbors) != 2)
        for start in starts:
            for neighbor in sorted(adjacency.get(start, set())):
                if cls._edge_key(start, neighbor) not in unused:
                    continue
                vertex_line = consume_path(start, neighbor)
                if len(vertex_line) >= 2:
                    polylines.append(np.asarray(coords[vertex_line], dtype=float))

        while unused:
            first, second = next(iter(unused))
            vertex_line = consume_path(first, second)
            if len(vertex_line) >= 2:
                polylines.append(np.asarray(coords[vertex_line], dtype=float))

        return polylines

    @classmethod
    def _smooth_boundary_polyline(cls, points, iterations):
        smoothed = np.asarray(points, dtype=float)
        if smoothed.ndim != 2 or smoothed.shape[0] < 3 or smoothed.shape[1] != 3:
            return smoothed
        iterations = cls._normalize_parcel_outline_smoothing(iterations)
        subdivision_steps = min(iterations, cls._parcel_outline_subdivision_max)
        relaxation_steps = max(0, iterations - subdivision_steps)
        for _ in range(subdivision_steps):
            if smoothed.shape[0] < 3:
                break
            closed = smoothed.shape[0] > 3 and np.allclose(smoothed[0], smoothed[-1])
            if closed:
                base = smoothed[:-1]
                if base.shape[0] < 3:
                    break
                next_points = np.roll(base, -1, axis=0)
                q_points = 0.75 * base + 0.25 * next_points
                r_points = 0.25 * base + 0.75 * next_points
                new_points = np.empty((base.shape[0] * 2, 3), dtype=float)
                new_points[0::2] = q_points
                new_points[1::2] = r_points
                smoothed = np.vstack((new_points, new_points[:1]))
            else:
                first = smoothed[:-1]
                second = smoothed[1:]
                q_points = 0.75 * first + 0.25 * second
                r_points = 0.25 * first + 0.75 * second
                middle = np.empty((q_points.shape[0] * 2, 3), dtype=float)
                middle[0::2] = q_points
                middle[1::2] = r_points
                smoothed = np.vstack((smoothed[:1], middle, smoothed[-1:]))
        for _ in range(relaxation_steps):
            if smoothed.shape[0] < 3:
                break
            closed = smoothed.shape[0] > 3 and np.allclose(smoothed[0], smoothed[-1])
            if closed:
                base = smoothed[:-1]
                if base.shape[0] < 3:
                    break
                prev_points = np.roll(base, 1, axis=0)
                next_points = np.roll(base, -1, axis=0)
                base = 0.5 * base + 0.25 * (prev_points + next_points)
                smoothed = np.vstack((base, base[:1]))
            else:
                interior = smoothed[1:-1]
                if interior.shape[0] == 0:
                    break
                smoothed[1:-1] = 0.5 * interior + 0.25 * (smoothed[:-2] + smoothed[2:])
        return smoothed

    @staticmethod
    def _segments_from_polylines(polylines):
        segment_parts = []
        for polyline in polylines:
            points = np.asarray(polyline, dtype=float)
            if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 3:
                continue
            segment_parts.append(np.stack((points[:-1], points[1:]), axis=1))
        if not segment_parts:
            return np.zeros((0, 2, 3), dtype=float)
        return np.concatenate(segment_parts, axis=0)

    @classmethod
    def _smoothed_boundary_segments_from_labels(cls, mesh, vertex_labels, *, iterations):
        iterations = cls._normalize_parcel_outline_smoothing(iterations)
        if iterations <= 0:
            return cls._boundary_segments_from_labels(mesh, vertex_labels)
        coords, _faces = cls._mesh_arrays(mesh)
        edges, label_pairs = cls._boundary_edges_from_labels(
            mesh,
            vertex_labels,
            return_label_pairs=True,
        )
        if edges.size == 0:
            return np.zeros((0, 2, 3), dtype=float)

        polylines = []
        for label_pair in np.unique(label_pairs, axis=0):
            pair_mask = np.all(label_pairs == label_pair, axis=1)
            for polyline in cls._boundary_polylines_from_edges(coords, edges[pair_mask]):
                polylines.append(cls._smooth_boundary_polyline(polyline, iterations))
        return cls._segments_from_polylines(polylines)

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
        label_mask = None
        if label_texture is not None:
            labels = np.asarray(label_texture, dtype=float).reshape(-1)
            label_mask = np.isfinite(labels) & (np.rint(labels) > 0)
        if support_texture is not None:
            support = np.asarray(support_texture, dtype=float).reshape(-1)
            support_mask = np.isfinite(support) & (support > 0.0)
            if label_mask is not None and label_mask.shape == support_mask.shape:
                return support_mask | label_mask
            return support_mask
        return label_mask

    @staticmethod
    def _mask_bool_from_texture(mask_texture):
        if mask_texture is None:
            return None
        mask = np.asarray(mask_texture, dtype=float).reshape(-1)
        return np.isfinite(mask) & (mask > 0.0)

    @classmethod
    def _combine_fill_masks(cls, fill_mask, mask_texture):
        mask_bool = cls._mask_bool_from_texture(mask_texture)
        if mask_bool is None:
            return fill_mask
        if fill_mask is None:
            return mask_bool
        fill = np.asarray(fill_mask, dtype=bool).reshape(-1)
        if fill.shape != mask_bool.shape:
            return mask_bool
        return fill & mask_bool

    def _surface_texture(
        self,
        comp_img,
        mesh,
        inner_mesh=None,
        label_texture=None,
        support_texture=None,
        mask_texture=None,
    ):
        if self._surface_style == "parcel_outlines":
            parcel_texture = self._parcel_value_texture_from_labels(comp_img, label_texture)
            if parcel_texture is not None:
                fill_mask = self._support_mask_from_texture(support_texture, label_texture)
                fill_mask = self._combine_fill_masks(fill_mask, mask_texture)
                if fill_mask is not None and parcel_texture.shape == fill_mask.shape:
                    parcel_texture = np.asarray(parcel_texture, dtype=float).copy()
                    parcel_texture[~fill_mask] = np.nan
                return self._filled_surface_texture(
                    mesh,
                    parcel_texture,
                    label_texture=label_texture,
                    fill_mask=fill_mask,
                )
        interpolation = "linear" if self._surface_style in {"smooth", "contours"} else "nearest_most_frequent"
        texture = self._vol_to_surf_robust(
            comp_img,
            mesh,
            radius=self._sampling_radius,
            interpolation=interpolation,
            inner_mesh=inner_mesh,
        )
        fill_mask = None if self._surface_style in {"smooth", "contours"} else self._support_mask_from_texture(
            support_texture,
            label_texture,
        )
        fill_mask = self._combine_fill_masks(fill_mask, mask_texture)
        if fill_mask is not None:
            texture = np.asarray(texture, dtype=float).copy()
            if texture.shape == fill_mask.shape:
                texture[~fill_mask] = np.nan
        texture = self._filled_surface_texture(
            mesh,
            texture,
            label_texture=label_texture,
            fill_mask=fill_mask,
        )
        if fill_mask is not None and texture.shape == fill_mask.shape:
            texture = np.asarray(texture, dtype=float).copy()
            texture[~fill_mask] = np.nan
        return texture

    def _surface_label_texture(self, mesh, inner_mesh=None):
        if self._parcellation_img is None:
            return None
        texture = self._vol_to_surf_robust(
            self._parcellation_img,
            mesh,
            radius=self._sampling_radius,
            interpolation="nearest_most_frequent",
            inner_mesh=inner_mesh,
        )
        texture = self._filled_label_texture(mesh, texture)
        if self._surface_style == "parcel_outlines":
            texture = self._clean_label_texture_for_boundaries(mesh, texture)
        return texture

    def _surface_support_texture(self, mesh, inner_mesh=None):
        if self._support_img is None:
            return None
        texture = self._vol_to_surf_robust(
            self._support_img,
            mesh,
            radius=self._sampling_radius,
            interpolation="nearest_most_frequent",
            inner_mesh=inner_mesh,
        )
        return self._filled_label_texture(mesh, texture)

    def _surface_mask_texture(self, mesh, inner_mesh=None):
        if self._mask_img is None:
            return None
        return self._vol_to_surf_robust(
            self._mask_img,
            mesh,
            radius=self._sampling_radius,
            interpolation="nearest_most_frequent",
            inner_mesh=inner_mesh,
        )

    def _parcel_boundary_segments_for_texture(self, mesh, label_texture):
        smoothing = self._normalize_parcel_outline_smoothing(
            getattr(self, "_parcel_outline_smoothing", 0)
        )
        cache = getattr(self, "_parcel_boundary_segments_cache", None)
        if cache is None:
            cache = {}
            self._parcel_boundary_segments_cache = cache
        cache_key = (id(mesh), id(label_texture), smoothing)
        if cache_key in cache:
            return cache[cache_key]
        if smoothing > 0:
            segments = self._smoothed_boundary_segments_from_labels(
                mesh,
                label_texture,
                iterations=smoothing,
            )
        else:
            segments = self._boundary_segments_from_labels(mesh, label_texture)
        cache[cache_key] = segments
        return segments

    def _add_parcel_boundaries(self, ax, mesh, label_texture, *, hemi=None, view=None):
        if label_texture is None:
            return
        segments = self._parcel_boundary_segments_for_texture(mesh, label_texture)
        if segments.size == 0:
            return
        coords, _faces = self._mesh_arrays(mesh)
        if hemi is not None and view is not None:
            segments = self._visible_isoline_segments(
                segments,
                coords,
                hemisphere=hemi,
                view=view,
            )
            if segments.size == 0:
                return
        center = np.nanmean(coords, axis=0)
        segments = center + 1.006 * (segments - center)
        collection = Line3DCollection(
            segments,
            colors=self._parcel_boundary_color,
            linewidths=self._parcel_boundary_linewidth,
            alpha=0.95,
            zorder=10,
        )
        collection.set_zorder(10)
        ax.add_collection3d(collection)

    def _render(self):
        self.figure.clear()
        self._parcel_boundary_segments_cache = {}
        assets = self._get_surface_assets(self._fsaverage_mesh)

        vmin, vmax, _threshold = self._compute_display_range(
            self._data,
            self._color_scale_min,
            self._color_scale_max,
        )
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
        mask_textures = {
            "left": self._surface_mask_texture(assets["mesh_left"], assets.get("white_left")),
            "right": self._surface_mask_texture(assets["mesh_right"], assets.get("white_right")),
        }
        for comp_idx in range(self._n_components):
            comp_img = nib.Nifti1Image(self._data[..., comp_idx], self._img.affine)
            texture_left = self._surface_texture(
                comp_img,
                assets["mesh_left"],
                assets.get("white_left"),
                label_textures.get("left"),
                support_textures.get("left"),
                mask_textures.get("left"),
            )
            texture_right = self._surface_texture(
                comp_img,
                assets["mesh_right"],
                assets.get("white_right"),
                label_textures.get("right"),
                support_textures.get("right"),
                mask_textures.get("right"),
            )
            textures = {"left": texture_left, "right": texture_right}
            display_norm = row_norm
            contour_levels = np.asarray([], dtype=float)
            if self._surface_style == "contours":
                for hemi_name, mesh_key in (("left", "mesh_left"), ("right", "mesh_right")):
                    _coords, faces = self._mesh_arrays(assets[mesh_key])
                    textures[hemi_name] = self._smooth_surface_texture(
                        textures[hemi_name],
                        faces,
                        iterations=self._contour_smoothing_iterations,
                    )
                contour_levels = self._shared_isoline_levels(
                    textures.values(),
                    level_count=self._contour_count,
                )
                if contour_levels.size and not self._has_color_scale_override():
                    bound = float(np.max(np.abs(contour_levels)))
                    if np.isfinite(bound) and bound > np.finfo(float).eps:
                        display_norm = Normalize(vmin=-bound, vmax=bound)
            row_axes = []

            for view_idx, (hemi, view, mesh, bg_map, view_label) in enumerate(views):
                ax = self.figure.add_subplot(gs[comp_idx, view_idx], projection="3d")
                display_mesh = self._display_mesh_for_style(self._surface_style, hemi, mesh, assets)
                self._plot_scalar_surface(
                    ax,
                    display_mesh,
                    textures[hemi],
                    bg_map,
                    display_norm,
                    hemi=hemi,
                    view=view,
                    title=view_label,
                    show_values=self._surface_style != "contours",
                )
                if self._surface_style == "contours":
                    self._add_surface_contours(
                        ax,
                        display_mesh,
                        textures[hemi],
                        display_norm,
                        contour_levels,
                        hemi=hemi,
                        view=view,
                    )
                if self._surface_style == "parcel_outlines" and self._show_parcel_outlines:
                    self._add_parcel_boundaries(
                        ax,
                        display_mesh,
                        label_textures.get(hemi),
                        hemi=hemi,
                        view=view,
                    )
                row_axes.append(ax)
            row_axes_all.append(row_axes)

            cax = self.figure.add_subplot(gs[comp_idx, n_view_cols])
            mappable = ScalarMappable(norm=display_norm, cmap=self._cmap)
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
