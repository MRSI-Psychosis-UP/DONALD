import unittest

import numpy as np

from window.plot_msmode import GradientScatterDialog


class PlotMsModeRgbTests(unittest.TestCase):
    def test_red_green_edge_midpoint_maps_to_bright_yellow(self):
        model = {
            "vertices": np.asarray(((0.0, 1.0), (-1.0, -1.0), (1.0, -1.0)), dtype=float),
            "anchor_points": np.asarray(((0.0, 1.0), (-1.0, -1.0), (1.0, -1.0)), dtype=float),
            "vertex_colors": np.asarray(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)), dtype=float),
            "order": "RBG",
            "fit_mode": "triangle",
        }

        colors = GradientScatterDialog._rgb_colors_from_model(
            np.asarray((0.5,), dtype=float),
            np.asarray((0.0,), dtype=float),
            model,
        )

        np.testing.assert_allclose(colors[0], np.asarray((1.0, 1.0, 0.0), dtype=float), atol=1e-6)

    def test_full_path_combination_honors_requested_cap(self):
        segment_records = [
            ("R", "G", [[0, 1], [0, 2, 1]]),
            ("G", "B", [[1, 3], [1, 4, 3]]),
        ]

        paths = GradientScatterDialog._combine_ordered_path_records(
            segment_records,
            max_full_paths=3,
        )

        self.assertEqual(len(paths), 3)


if __name__ == "__main__":
    unittest.main()
