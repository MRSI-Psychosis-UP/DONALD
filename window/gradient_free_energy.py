#!/usr/bin/env python3
"""Free-energy histogram dialog for gradient path analysis."""

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
except Exception:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

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


class GradientFreeEnergyDialog(QDialog):
    """Histogram viewer for path directionality energies and free energies."""

    def __init__(self, payload, *, parent=None, theme_name="Dark"):
        super().__init__(parent)
        self._payload = dict(payload or {})
        self._theme_name = "Dark"
        self.setWindowTitle(f"Free Energy - {str(self._payload.get('title', 'Gradients'))}")

        self.figure = Figure(figsize=(11.2, 6.8), constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        lambda_value = float(self._payload.get("lambda", 1.0))
        axis_text = f"{str(self._payload.get('y_axis_label', 'Y'))} vs {str(self._payload.get('x_axis_label', 'X'))}"
        include_line_proximity = bool(self._payload.get("use_line_proximity_energy", True))
        normalize_by_segments = bool(self._payload.get("normalize_energy_by_segments", False))
        if include_line_proximity:
            energy_text = "Energy = sum((|step| - step dot ref_unit) + |step| * d_line / |ref|)"
        else:
            energy_text = "Energy = sum(|step| - step dot ref_unit)"
        if normalize_by_segments:
            energy_text += " / segments"
        self.info_label = QLabel(
            f"lambda = {lambda_value:.3f} | {energy_text} | {axis_text}"
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

    def _render(self):
        self.figure.clear()
        groups = [dict(group) for group in list(self._payload.get("groups", []))]
        if not groups:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "No free-energy paths available.", ha="center", va="center")
            ax.axis("off")
            self.canvas.draw_idle()
            return

        n_rows = len(groups)
        n_cols = max(1, max(len(list(dict(group).get("families", []))) for group in groups))
        axes = np.asarray(self.figure.subplots(n_rows, n_cols, squeeze=False), dtype=object)
        title_text = str(self._payload.get("title", "Gradients"))
        rotation_text = str(self._payload.get("rotation", "Default"))
        norm_text = " | Energy/segments" if bool(self._payload.get("normalize_energy_by_segments", False)) else ""
        self.figure.suptitle(f"Free Energy Histogram - {title_text} | Rotation: {rotation_text}{norm_text}", fontsize=12)

        for row_idx, group in enumerate(groups):
            families = [dict(family) for family in list(group.get("families", []))]
            for col_idx in range(n_cols):
                ax = axes[row_idx, col_idx]
                if col_idx >= len(families):
                    ax.axis("off")
                    continue
                family = families[col_idx]
                energies = np.sort(np.asarray(family.get("energies", []), dtype=float).reshape(-1))
                if energies.size == 0:
                    ax.text(0.5, 0.5, "No paths", ha="center", va="center")
                    ax.axis("off")
                    continue
                color = np.asarray(family.get("color", (0.2, 0.2, 0.2)), dtype=float).reshape(-1)
                if color.shape != (3,):
                    color = np.asarray((0.2, 0.2, 0.2), dtype=float)
                free_energy = float(family.get("free_energy", float("nan")))
                mean_energy = float(np.mean(energies)) if energies.size else float("nan")
                n_bins = max(6, min(20, int(np.ceil(np.sqrt(energies.size)))))
                ax.hist(
                    energies,
                    bins=n_bins,
                    color=tuple(color.tolist()),
                    alpha=0.7,
                    edgecolor="#111827",
                    linewidth=0.8,
                )
                if np.isfinite(mean_energy):
                    ax.axvline(
                        mean_energy,
                        color=tuple(color.tolist()),
                        linestyle="--",
                        linewidth=1.3,
                        alpha=0.95,
                    )
                ax.set_title(
                    f"{str(group.get('group', 'all')).upper()} | {str(family.get('label', 'path'))}\n"
                    f"F={free_energy:.4f} | n={int(family.get('n_paths', energies.size))}",
                    fontsize=10,
                )
                ax.set_xlabel("Energy / segments" if bool(self._payload.get("normalize_energy_by_segments", False)) else "Energy")
                ax.set_ylabel("Count")
                ax.grid(True, alpha=0.25)

        self.canvas.draw_idle()

    def _save_figure(self):
        default_name = "gradient_free_energy.png"
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save free-energy figure",
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


__all__ = ["GradientFreeEnergyDialog"]
