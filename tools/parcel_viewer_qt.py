import sys
import os
from typing import Optional, Dict, Tuple

import numpy as np
import nibabel as nib

from PyQt6 import QtWidgets, QtCore
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap, to_rgba


def load_tsv_lut(tsv_path: str) -> Tuple[Dict[int, str], Dict[int, Tuple[float, float, float, float]]]:
    """Load a TSV LUT mapping index->name and index->color (hex).

    Expects columns with headers: index, name, color.
    Returns:
        (names_map, colors_map)
    """
    names: Dict[int, str] = {}
    colors: Dict[int, Tuple[float, float, float, float]] = {}
    if not os.path.isfile(tsv_path):
        return names, colors

    with open(tsv_path, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split('\t')
        # Create index for columns
        try:
            idx_index = header.index('index')
        except ValueError:
            # Try alternative header names
            idx_index = header.index('label') if 'label' in header else 0
        try:
            idx_name = header.index('name')
        except ValueError:
            idx_name = None
        try:
            idx_color = header.index('color')
        except ValueError:
            idx_color = None

        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip('\n').split('\t')
            try:
                label_val = int(float(parts[idx_index]))
            except Exception:
                continue
            if idx_name is not None and idx_name < len(parts):
                names[label_val] = parts[idx_name]
            if idx_color is not None and idx_color < len(parts):
                try:
                    colors[label_val] = to_rgba(parts[idx_color])
                except ValueError:
                    pass
    return names, colors


def build_listed_colormap(label_colors: Dict[int, Tuple[float, float, float, float]],
                          max_size_hint: int = 4096) -> Optional[ListedColormap]:
    """Build a ListedColormap from a label->rgba map. Returns None if too sparse/large."""
    if not label_colors:
        return None
    max_idx = max(label_colors.keys())
    if max_idx + 1 > max_size_hint:
        return None
    # Initialize colormap; 0 reserved for background (transparent black)
    table = np.zeros((max_idx + 1, 4), dtype=float)
    table[0] = (0, 0, 0, 0)
    for k, rgba in label_colors.items():
        if 0 <= k <= max_idx:
            table[k] = rgba
    return ListedColormap(table)


class SliceCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(6, 6))
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_aspect('equal')
        self.fig.tight_layout(pad=0.1)
        self.image_artist = None
        self.crosshair = None

    def show_slice(self, slice2d: np.ndarray, cmap=None, vmin=None, vmax=None):
        self.ax.clear()
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.image_artist = self.ax.imshow(
            slice2d, origin='lower', interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax
        )
        self.draw_idle()

    def draw_crosshair(self, x: int, y: int, color: str = 'white'):
        if self.ax is None:
            return
        self.ax.axhline(y, color=color, linewidth=0.6, alpha=0.6)
        self.ax.axvline(x, color=color, linewidth=0.6, alpha=0.6)
        self.draw_idle()


class ParcelViewer(QtWidgets.QMainWindow):
    def __init__(self, image_path: Optional[str] = None, lut_path: Optional[str] = None):
        super().__init__()
        self.setWindowTitle('Parcel Viewer (PyQt6)')
        self.resize(900, 800)

        # Data
        self.data: Optional[np.ndarray] = None
        self.affine = None
        self.header = None
        self.plane = 'Axial (Z)'
        self.slice_index = 0

        # LUT
        self.label_names: Dict[int, str] = {}
        self.label_colors: Dict[int, Tuple[float, float, float, float]] = {}
        self.cmap: Optional[ListedColormap] = None

        # UI
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        controls = QtWidgets.QHBoxLayout()
        layout.addLayout(controls)

        self.open_btn = QtWidgets.QPushButton('Open NIfTI...')
        self.open_btn.clicked.connect(self.open_nifti_dialog)
        controls.addWidget(self.open_btn)

        self.open_lut_btn = QtWidgets.QPushButton('Open LUT TSV...')
        self.open_lut_btn.clicked.connect(self.open_lut_dialog)
        controls.addWidget(self.open_lut_btn)

        controls.addSpacing(20)

        self.plane_combo = QtWidgets.QComboBox()
        self.plane_combo.addItems(['Axial (Z)', 'Coronal (Y)', 'Sagittal (X)'])
        self.plane_combo.currentTextChanged.connect(self.on_plane_changed)
        controls.addWidget(QtWidgets.QLabel('Plane:'))
        controls.addWidget(self.plane_combo)

        controls.addSpacing(10)

        self.slice_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(0)
        self.slice_slider.setSingleStep(1)
        self.slice_slider.valueChanged.connect(self.on_slice_changed)
        layout.addWidget(self.slice_slider)

        self.canvas = SliceCanvas()
        layout.addWidget(self.canvas, stretch=1)

        self.status = self.statusBar()

        # Mouse click event
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)

        # Load initial files if provided
        if lut_path:
            self.load_lut(lut_path)
        if image_path:
            self.load_image(image_path)
        elif not image_path:
            # Try to prompt user at startup
            QtCore.QTimer.singleShot(0, self.open_nifti_dialog)

    # --- Loading ---
    def open_nifti_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open NIfTI Parcellation', '', 'NIfTI files (*.nii *.nii.gz)'
        )
        if path:
            self.load_image(path)
            # Try auto LUT detection in same dir with same stem
            base = os.path.splitext(os.path.basename(path))[0]
            if base.endswith('.nii'):
                base = base[:-4]
            for cand in [f"{base}.tsv", f"{base}_labels.tsv", f"{base}-labels.tsv"]:
                lut_guess = os.path.join(os.path.dirname(path), cand)
                if os.path.isfile(lut_guess):
                    self.load_lut(lut_guess)
                    break

    def open_lut_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open LUT TSV', '', 'TSV files (*.tsv)'
        )
        if path:
            self.load_lut(path)

    def load_image(self, path: str):
        try:
            img = nib.load(path)
            data = img.get_fdata()
            # Cast to int if it looks like a label image
            if np.allclose(data, np.round(data)):
                data = data.astype(np.int32)
            else:
                data = data.astype(np.float32)
            self.data = data
            self.affine = img.affine
            self.header = img.header
            self.setWindowTitle(f'Parcel Viewer - {os.path.basename(path)}')
            self.update_slider_range()
            self.update_view()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Error', f'Failed to load image:\n{e}')

    def load_lut(self, path: str):
        try:
            names, colors = load_tsv_lut(path)
            self.label_names = names
            self.label_colors = colors
            self.cmap = build_listed_colormap(colors)
            # If no colormap could be built, keep None and use default categorical
            self.status.showMessage(f'Loaded LUT: {os.path.basename(path)} ({len(names)} labels)', 5000)
            self.update_view()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, 'LUT Error', f'Failed to load LUT TSV:\n{e}')

    # --- UI updates ---
    def on_plane_changed(self, text: str):
        self.plane = text
        self.update_slider_range()
        self.update_view()

    def on_slice_changed(self, value: int):
        self.slice_index = value
        self.update_view()

    def update_slider_range(self):
        if self.data is None:
            self.slice_slider.setMaximum(0)
            self.slice_slider.setValue(0)
            return
        nx, ny, nz = self.data.shape
        if self.plane.startswith('Axial'):
            max_idx = nz - 1
        elif self.plane.startswith('Coronal'):
            max_idx = ny - 1
        else:  # Sagittal
            max_idx = nx - 1
        self.slice_slider.setMaximum(max_idx)
        if self.slice_index > max_idx:
            self.slice_index = max_idx
        self.slice_slider.setValue(self.slice_index)

    def update_view(self):
        if self.data is None:
            return

        # Select slice based on plane
        if self.plane.startswith('Axial'):
            sl = self.data[:, :, self.slice_index]
        elif self.plane.startswith('Coronal'):
            sl = self.data[:, self.slice_index, :]
        else:  # Sagittal
            sl = self.data[self.slice_index, :, :]

        # Choose colormap
        cmap = None
        vmin = None
        vmax = None
        if np.issubdtype(sl.dtype, np.integer):
            # Label image
            if self.cmap is not None:
                cmap = self.cmap
                vmin = 0
                vmax = self.cmap.N - 1
            else:
                # Use a categorical palette
                from matplotlib import cm
                cmap = cm.get_cmap('tab20', 20)
        self.canvas.show_slice(sl, cmap=cmap, vmin=vmin, vmax=vmax)

    # --- Mouse handling ---
    def on_canvas_click(self, event):
        if event.inaxes != self.canvas.ax or self.data is None:
            return
        if event.xdata is None or event.ydata is None:
            return

        nx, ny, nz = self.data.shape

        # Determine 2D slice dimensions (rows=M, cols=N) for current plane
        if self.plane.startswith('Axial'):
            M, N = nx, ny
            # Map display (i=row, j=col) -> volume (x=i, y=j, z=fixed)
            i = int(np.clip(round(event.ydata), 0, M - 1))
            j = int(np.clip(round(event.xdata), 0, N - 1))
            x, y, z = i, j, self.slice_index
        elif self.plane.startswith('Coronal'):
            M, N = nx, nz
            i = int(np.clip(round(event.ydata), 0, M - 1))
            j = int(np.clip(round(event.xdata), 0, N - 1))
            x, y, z = i, self.slice_index, j
        else:  # Sagittal
            M, N = ny, nz
            i = int(np.clip(round(event.ydata), 0, M - 1))
            j = int(np.clip(round(event.xdata), 0, N - 1))
            x, y, z = self.slice_index, i, j

        # Bounds check (paranoid)
        if not (0 <= x < nx and 0 <= y < ny and 0 <= z < nz):
            return

        val = self.data[x, y, z]
        name = self.label_names.get(int(val), None)
        msg = f'Click voxel=({x}, {y}, {z}) label={int(val)}'
        if name:
            msg += f' "{name}"'

        print(msg, flush=True)
        self.status.showMessage(msg, 5000)
        # Draw crosshair at integer pixel position
        self.canvas.draw_crosshair(j, i)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Simple PyQt parcellation viewer with click-to-label.')
    parser.add_argument('--image', '-i', type=str, default=None, help='Path to parcellation NIfTI (.nii or .nii.gz).')
    parser.add_argument('--lut', '-l', type=str, default=None, help='Optional TSV LUT with columns index,name,color.')
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    viewer = ParcelViewer(image_path=args.image, lut_path=args.lut)
    viewer.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
