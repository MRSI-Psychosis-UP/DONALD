#!/usr/bin/env python3
"""Nilearn surface rendering dialog for MSM/gradient volume maps.

This module is designed for GUI integration (e.g., connectome_viewer).
It renders each component on one row with four views:
left-lateral, left-medial, right-medial, right-lateral,
and a per-component colorbar on the right.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
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
    from PyQt6.QtWidgets import (
        QDialog,
        QFileDialog,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QVBoxLayout,
    )
except Exception:
    from PyQt5.QtWidgets import (
        QDialog,
        QFileDialog,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QVBoxLayout,
    )


class MSModeSurfaceDialog(QDialog):
    """Interactive Nilearn surface viewer for 3D/4D component maps."""
    _surface_assets_cache = None

    def __init__(
        self,
        volume_img,
        title="Surface Components",
        parent=None,
        cmap=None,
        cmap_name="spectrum_fsl",
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
        self.setWindowTitle(title)

        self.figure = Figure(
            figsize=(20, max(3.2 * self._n_components, 4.0)),
            constrained_layout=True,
        )
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.info_label = QLabel(
            "Views per component: LH lateral | LH medial | RH medial | RH lateral"
            f" | Cmap: {self._cmap_name}"
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

        self._render()

    @staticmethod
    def _default_cmap(cmap_name):
        try:
            try:
                from graphplot.colorbar import ColorBar
            except Exception:
                from mrsitoolbox.graphplot.colorbar import ColorBar

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
        title="Surface Components",
        parent=None,
        cmap=None,
        cmap_name="spectrum_fsl",
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
        )

    @classmethod
    def _get_surface_assets(cls):
        """Load and cache fsaverage meshes/background maps once."""
        if cls._surface_assets_cache is not None:
            return cls._surface_assets_cache

        try:
            fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage4")
        except Exception:
            fsaverage = datasets.fetch_surf_fsaverage()

        mesh_left = surface.load_surf_mesh(fsaverage.pial_left)
        mesh_right = surface.load_surf_mesh(fsaverage.pial_right)
        sulc_left = surface.load_surf_data(fsaverage.sulc_left)
        sulc_right = surface.load_surf_data(fsaverage.sulc_right)

        cls._surface_assets_cache = {
            "mesh_left": mesh_left,
            "mesh_right": mesh_right,
            "sulc_left": sulc_left,
            "sulc_right": sulc_right,
        }
        return cls._surface_assets_cache

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

    def _render(self):
        self.figure.clear()
        assets = self._get_surface_assets()

        vmin, vmax, threshold = self._compute_display_range(self._data)
        row_norm = Normalize(vmin=vmin, vmax=vmax)

        views = (
            ("left", "lateral", assets["mesh_left"], assets["sulc_left"], "LH Lateral"),
            ("left", "medial", assets["mesh_left"], assets["sulc_left"], "LH Medial"),
            ("right", "medial", assets["mesh_right"], assets["sulc_right"], "RH Medial"),
            ("right", "lateral", assets["mesh_right"], assets["sulc_right"], "RH Lateral"),
        )
        gs = self.figure.add_gridspec(
            self._n_components,
            5,
            width_ratios=[1.0, 1.0, 1.0, 1.0, 0.05],
            wspace=0.02,
            hspace=0.08,
        )

        row_axes_all = []
        for comp_idx in range(self._n_components):
            comp_img = nib.Nifti1Image(self._data[..., comp_idx], self._img.affine)
            texture_left = surface.vol_to_surf(
                comp_img,
                assets["mesh_left"],
                radius=0.5,
                interpolation="linear",
            )
            texture_right = surface.vol_to_surf(
                comp_img,
                assets["mesh_right"],
                radius=0.5,
                interpolation="linear",
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
                    threshold=threshold,
                    axes=ax,
                )
                ax.set_title(view_label, fontsize=9, pad=2)
                row_axes.append(ax)
            row_axes_all.append(row_axes)

            cax = self.figure.add_subplot(gs[comp_idx, 4])
            mappable = ScalarMappable(norm=row_norm, cmap=self._cmap)
            mappable.set_array([])
            cbar = self.figure.colorbar(mappable, cax=cax)
            cbar.ax.tick_params(labelsize=8, length=2)
            cbar.set_label("Gradient", fontsize=10)

        # Center component titles per row at a safe height in the gap above each row.
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
                f"MS Mode {comp_idx + 1}",
                ha="center",
                va="bottom",
                fontsize=16,
            )
            prev_row_bottom = row_bottom

        self.canvas.draw_idle()

    def _save_figure(self):
        default_name = f"surface_components_{self._n_components}.png"
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save surface figure",
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


__all__ = ["MSModeSurfaceDialog"]
