#!/usr/bin/env python3
"""Dialog to combine workspace matrices and preview the result."""

from __future__ import annotations

import numpy as np

try:
    from PyQt6.QtWidgets import (
        QComboBox,
        QDialog,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QProgressBar,
        QPushButton,
        QVBoxLayout,
    )
except Exception:
    from PyQt5.QtWidgets import (
        QComboBox,
        QDialog,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QProgressBar,
        QPushButton,
        QVBoxLayout,
    )

from matplotlib.figure import Figure

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
except Exception:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


try:
    from window.shared.theme import panel_dialog_stylesheet as _panel_dialog_stylesheet
except Exception:
    from mrsi_viewer.window.shared.theme import panel_dialog_stylesheet as _panel_dialog_stylesheet


class CombinePrepareDialog(QDialog):
    """Modeless window to combine two workspace matrices."""

    OPERATION_OPTIONS = (
        ("Addition", "add"),
        ("Subtraction", "subtract"),
        ("Intersect", "intersect"),
        ("Correct", "correct"),
        ("Spatial Correction", "spatial_correction"),
        ("Correlation", "correlation"),
        ("Elementwise Product", "elementwise_product"),
        ("Matrix Multiplication", "matmul"),
    )

    def __init__(self, *, theme_name="Dark", process_callback=None, confirm_callback=None, parent=None):
        super().__init__(parent)
        self._process_callback = process_callback
        self._confirm_callback = confirm_callback
        self._plot_colors = {}
        self._result_payload = {"kind": "empty", "message": "No result yet."}

        self.setWindowTitle("Combine Matrices")
        self.resize(920, 760)
        self._build_ui()
        self.set_matrix_options([])
        self.set_theme(theme_name)
        self.set_status("Select two workspace matrices and choose an operation.")
        self.clear_result()

    def _build_ui(self):
        root_layout = QVBoxLayout(self)

        description = QLabel(
            "Select two matrices from the workspace, choose an operation, and preview the result here."
        )
        description.setWordWrap(True)
        root_layout.addWidget(description, 0)

        selection_group = QGroupBox("Selection")
        selection_layout = QGridLayout(selection_group)
        selection_layout.addWidget(QLabel("Matrix A"), 0, 0)
        selection_layout.addWidget(QLabel("Matrix B"), 0, 1)

        self.matrix_a_combo = QComboBox()
        self.matrix_a_combo.currentIndexChanged.connect(self._on_inputs_changed)
        selection_layout.addWidget(self.matrix_a_combo, 1, 0)

        self.matrix_b_combo = QComboBox()
        self.matrix_b_combo.currentIndexChanged.connect(self._on_inputs_changed)
        selection_layout.addWidget(self.matrix_b_combo, 1, 1)
        root_layout.addWidget(selection_group, 0)

        action_row = QHBoxLayout()
        action_row.addWidget(QLabel("Operation"), 0)
        self.operation_combo = QComboBox()
        for label, value in self.OPERATION_OPTIONS:
            self.operation_combo.addItem(label, value)
        self.operation_combo.currentIndexChanged.connect(self._on_inputs_changed)
        action_row.addWidget(self.operation_combo, 1)

        self.process_button = QPushButton("Process")
        self.process_button.clicked.connect(self._trigger_process)
        action_row.addWidget(self.process_button, 0)

        self.confirm_button = QPushButton("Add to Workspace")
        self.confirm_button.clicked.connect(self._trigger_confirm)
        self.confirm_button.setEnabled(False)
        action_row.addWidget(self.confirm_button, 0)
        root_layout.addLayout(action_row, 0)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        root_layout.addWidget(self.status_label, 0)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("")
        self.progress_bar.setVisible(False)
        root_layout.addWidget(self.progress_bar, 0)

        result_group = QGroupBox("Result")
        result_layout = QVBoxLayout(result_group)
        self.result_summary_label = QLabel("No result yet.")
        self.result_summary_label.setWordWrap(True)
        result_layout.addWidget(self.result_summary_label, 0)

        self.result_figure = Figure()
        self.result_canvas = FigureCanvas(self.result_figure)
        result_layout.addWidget(self.result_canvas, 1)
        root_layout.addWidget(result_group, 1)

    def selected_matrix_a_id(self):
        return self.matrix_a_combo.currentData()

    def selected_matrix_b_id(self):
        return self.matrix_b_combo.currentData()

    def selected_operation(self):
        return self.operation_combo.currentData() or "add"

    def set_matrix_options(self, options):
        current_a = self.selected_matrix_a_id()
        current_b = self.selected_matrix_b_id()

        self.matrix_a_combo.blockSignals(True)
        self.matrix_b_combo.blockSignals(True)
        self.matrix_a_combo.clear()
        self.matrix_b_combo.clear()

        normalized = list(options or [])
        for option in normalized:
            label = str(option.get("label") or option.get("id") or "matrix")
            value = option.get("id")
            self.matrix_a_combo.addItem(label, value)
            self.matrix_b_combo.addItem(label, value)

        if current_a is not None:
            index = self.matrix_a_combo.findData(current_a)
            if index >= 0:
                self.matrix_a_combo.setCurrentIndex(index)
        if self.matrix_a_combo.currentIndex() < 0 and self.matrix_a_combo.count() > 0:
            self.matrix_a_combo.setCurrentIndex(0)

        if current_b is not None:
            index = self.matrix_b_combo.findData(current_b)
            if index >= 0:
                self.matrix_b_combo.setCurrentIndex(index)
        if self.matrix_b_combo.currentIndex() < 0:
            if self.matrix_b_combo.count() > 1:
                self.matrix_b_combo.setCurrentIndex(1)
            elif self.matrix_b_combo.count() > 0:
                self.matrix_b_combo.setCurrentIndex(0)

        self.matrix_a_combo.blockSignals(False)
        self.matrix_b_combo.blockSignals(False)
        self._refresh_process_button()
        self.set_can_confirm(False)

    def set_status(self, text: str):
        self.status_label.setText(str(text or ""))

    def set_progress(self, minimum: int, maximum: int, value: int, text: str = ""):
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(int(minimum), int(maximum))
        self.progress_bar.setValue(int(value))
        self.progress_bar.setFormat(str(text or ""))

    def clear_progress(self):
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("")
        self.progress_bar.setVisible(False)

    def set_can_confirm(self, enabled: bool):
        self.confirm_button.setEnabled(bool(enabled) and self._confirm_callback is not None)

    def clear_result(self, message: str = "No result yet."):
        self._result_payload = {"kind": "empty", "message": str(message or "No result yet.")}
        self.set_can_confirm(False)
        self.clear_progress()
        self._render_result_payload()

    def show_matrix_result(self, matrix, *, title: str = "", summary_text: str = ""):
        self._result_payload = {
            "kind": "matrix",
            "matrix": np.asarray(matrix, dtype=float),
            "title": str(title or "Matrix result"),
            "summary_text": str(summary_text or ""),
        }
        self.set_can_confirm(True)
        self._render_result_payload()

    def show_correlation_result(
        self,
        x_values,
        y_values,
        *,
        title: str = "",
        summary_text: str = "",
        r_value=None,
        p_value=None,
        slope=None,
        intercept=None,
    ):
        self._result_payload = {
            "kind": "correlation",
            "x_values": np.asarray(x_values, dtype=float).reshape(-1),
            "y_values": np.asarray(y_values, dtype=float).reshape(-1),
            "title": str(title or "Correlation"),
            "summary_text": str(summary_text or ""),
            "r_value": r_value,
            "p_value": p_value,
            "slope": slope,
            "intercept": intercept,
        }
        self.set_can_confirm(False)
        self._render_result_payload()

    def _refresh_process_button(self):
        operation = self.selected_operation()
        needs_matrix_b = operation != "spatial_correction"
        self.matrix_b_combo.setEnabled(bool(needs_matrix_b))
        enabled = (
            self._process_callback is not None
            and self.matrix_a_combo.count() > 0
            and self.selected_matrix_a_id() is not None
            and (
                not needs_matrix_b
                or (
                    self.matrix_b_combo.count() > 0
                    and self.selected_matrix_b_id() is not None
                )
            )
        )
        self.process_button.setEnabled(enabled)

    def _on_inputs_changed(self, *_args):
        self._refresh_process_button()
        self.set_can_confirm(False)

    def _trigger_process(self):
        if self._process_callback is None:
            return
        operation = self.selected_operation()
        self._process_callback(
            self.selected_matrix_a_id(),
            None if operation == "spatial_correction" else self.selected_matrix_b_id(),
            operation,
        )

    def _trigger_confirm(self):
        if self._confirm_callback is None:
            return
        self._confirm_callback()

    def _theme_plot_colors(self, theme: str):
        if theme == "Dark":
            return {
                "figure": "#1f2430",
                "axes": "#2a3140",
                "text": "#e5e7eb",
                "grid": "#556070",
                "accent": "#60a5fa",
                "scatter": "#93c5fd",
                "line": "#f97316",
            }
        if theme == "Teya":
            return {
                "figure": "#ffd0e5",
                "axes": "#ffe6f1",
                "text": "#0b7f7a",
                "grid": "#1db8b2",
                "accent": "#2ecfc9",
                "scatter": "#0b7f7a",
                "line": "#db2777",
            }
        if theme == "Donald":
            return {
                "figure": "#d97706",
                "axes": "#c96a04",
                "text": "#ffffff",
                "grid": "#f3a451",
                "accent": "#ffd19e",
                "scatter": "#ffffff",
                "line": "#2563eb",
            }
        return {
            "figure": "#f4f6f9",
            "axes": "#ffffff",
            "text": "#1f2937",
            "grid": "#c9d0da",
            "accent": "#2563eb",
            "scatter": "#1d4ed8",
            "line": "#dc2626",
        }

    def _apply_axes_style(self, ax):
        colors = self._plot_colors or self._theme_plot_colors("Dark")
        ax.set_facecolor(colors["axes"])
        ax.tick_params(colors=colors["text"])
        ax.xaxis.label.set_color(colors["text"])
        ax.yaxis.label.set_color(colors["text"])
        ax.title.set_color(colors["text"])
        for spine in ax.spines.values():
            spine.set_color(colors["grid"])

    def _render_result_payload(self):
        payload = dict(self._result_payload or {})
        kind = payload.get("kind", "empty")
        colors = self._plot_colors or self._theme_plot_colors("Dark")
        self.result_figure.clear()
        self.result_figure.patch.set_facecolor(colors["figure"])

        if kind == "matrix":
            matrix = np.asarray(payload.get("matrix"), dtype=float)
            ax = self.result_figure.add_subplot(111)
            self._apply_axes_style(ax)
            image = ax.imshow(
                matrix,
                cmap="viridis",
                aspect="equal" if matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1] else "auto",
                interpolation="nearest",
            )
            ax.set_title(str(payload.get("title") or "Matrix result"))
            colorbar = self.result_figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            colorbar.ax.tick_params(colors=colors["text"])
            colorbar.outline.set_edgecolor(colors["grid"])
            self.result_summary_label.setText(str(payload.get("summary_text") or ""))
        elif kind == "correlation":
            x_values = np.asarray(payload.get("x_values"), dtype=float).reshape(-1)
            y_values = np.asarray(payload.get("y_values"), dtype=float).reshape(-1)
            ax = self.result_figure.add_subplot(111)
            self._apply_axes_style(ax)
            ax.grid(True, color=colors["grid"], alpha=0.35, linewidth=0.8)

            if x_values.size > 0:
                point_count = min(x_values.size, 20000)
                if point_count < x_values.size:
                    indices = np.linspace(0, x_values.size - 1, point_count, dtype=int)
                    x_plot = x_values[indices]
                    y_plot = y_values[indices]
                else:
                    x_plot = x_values
                    y_plot = y_values
                ax.scatter(
                    x_plot,
                    y_plot,
                    s=14,
                    alpha=0.55,
                    color=colors["scatter"],
                    edgecolors="none",
                )

            slope = payload.get("slope")
            intercept = payload.get("intercept")
            if x_values.size > 1 and slope is not None and intercept is not None:
                x_line = np.array([np.nanmin(x_values), np.nanmax(x_values)], dtype=float)
                y_line = slope * x_line + intercept
                ax.plot(x_line, y_line, color=colors["line"], linewidth=2.0)

            ax.set_xlabel("Matrix A values")
            ax.set_ylabel("Matrix B values")
            ax.set_title(str(payload.get("title") or "Correlation"))

            annotation_lines = []
            if payload.get("r_value") is not None:
                annotation_lines.append(f"r = {float(payload['r_value']):.4f}")
            if payload.get("p_value") is not None:
                p_value = float(payload["p_value"])
                annotation_lines.append(f"p = {p_value:.4e}" if np.isfinite(p_value) else "p = n/a")
            annotation_lines.append(f"n = {int(x_values.size)}")
            ax.text(
                0.03,
                0.97,
                "\n".join(annotation_lines),
                transform=ax.transAxes,
                va="top",
                ha="left",
                color=colors["text"],
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "facecolor": colors["axes"],
                    "edgecolor": colors["grid"],
                    "alpha": 0.92,
                },
            )
            self.result_summary_label.setText(str(payload.get("summary_text") or ""))
        else:
            ax = self.result_figure.add_subplot(111)
            ax.set_facecolor(colors["axes"])
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                str(payload.get("message") or "No result yet."),
                transform=ax.transAxes,
                ha="center",
                va="center",
                color=colors["text"],
                fontsize=11,
            )
            self.result_summary_label.setText(str(payload.get("message") or "No result yet."))

        self.result_figure.tight_layout()
        self.result_canvas.draw_idle()

    def set_theme(self, theme_name="Dark"):
        theme, style = _panel_dialog_stylesheet(
            theme_name,
            control_selector="QPushButton, QComboBox",
            include_groupbox=True,
        )
        self._plot_colors = self._theme_plot_colors(theme)
        self.setStyleSheet(style)
        self._render_result_payload()


__all__ = ["CombinePrepareDialog"]
