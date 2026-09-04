import ast
import inspect
import os
from pathlib import Path
import tempfile
import textwrap
import unittest
from unittest import mock

import numpy as np

import services.gradient_dialog_controller as gradient_module
from services.gradient_dialog_controller import GradientDialogController
from window.gradients_prepare import GradientsPrepareDialog
from window.plot_msmode import GradientScatterDialog


class _StatusBarStub:
    def __init__(self):
        self.messages = []

    def showMessage(self, message):
        self.messages.append(str(message))


class _HeaderStub(dict):
    def copy(self):
        return _HeaderStub(self)


class _TemplateImageStub:
    def __init__(self):
        self.affine = np.eye(4, dtype=float)
        self.header = _HeaderStub()


class _DataAccessStub:
    def entry_parcel_metadata(self, _entry, expected_len=None):
        labels = np.asarray([1, 2, 3], dtype=int)
        if expected_len is not None and int(expected_len) != labels.size:
            raise AssertionError("Unexpected expected_len")
        return labels, ["Parcel 1", "Parcel 2", "Parcel 3"]


class _ViewerStub:
    def __init__(self):
        self._status_bar = _StatusBarStub()
        self._entries = {
            "entry-1": {
                "id": "entry-1",
                "label": "matrix_a",
                "path": Path("/tmp/matrix_a.npz"),
            }
        }
        self._matrix = np.asarray(
            [
                [0.0, 1.0, 2.0],
                [1.0, 0.0, 3.0],
                [2.0, 3.0, 0.0],
            ],
            dtype=float,
        )
        self._data_access = _DataAccessStub()
        self._gradient_selected_entry_id = "entry-1"
        self._gradient_component_count = 2
        self._gradient_precomputed_bundle = None
        self._gradient_precomputed_selected_row = None
        self._gradient_use_precomputed_bundle = False
        self._active_parcellation_path = Path("/tmp/template.nii.gz")
        self._active_parcellation_img = _TemplateImageStub()
        self._active_parcellation_data = np.asarray(
            [
                [[1, 0], [0, 2]],
                [[0, 3], [0, 0]],
            ],
            dtype=int,
        )
        self._gradients_busy = False
        self._gradients_progress_state = {"minimum": 0, "maximum": 1, "value": 0, "text": "Idle"}
        self._gradients_dialog = None
        self._last_gradients = None
        self._theme_name = "Dark"

    def statusBar(self):
        return self._status_bar

    def _default_results_dir(self):
        return Path("/tmp")

    def _current_entry(self):
        return None

    def _matrix_for_entry(self, _entry):
        return np.asarray(self._matrix, dtype=float), None


class _NetToolsStub:
    def __init__(self):
        self.project_calls = 0

    def dimreduce_matrix(self, matrix, *, output_dim, **_kwargs):
        return np.arange(matrix.shape[0], dtype=float) + float(output_dim)

    def project_to_3dspace(self, *_args, **_kwargs):
        self.project_calls += 1
        raise AssertionError("Gradient compute should not project to 3D space.")


class _QApplicationStub:
    @staticmethod
    def processEvents():
        return None

    @staticmethod
    def primaryScreen():
        return None


def _make_controller(viewer):
    return GradientDialogController(
        viewer,
        parcel_label_keys=("parcel_labels_group",),
        parcel_name_keys=("parcel_names_group",),
        to_string_list=lambda values: [str(value) for value in np.asarray(values).reshape(-1).tolist()],
        display_text=lambda value: str(value),
        load_covars_info=lambda _path: None,
        covars_to_rows=lambda _info: ([], []),
        normalize_subject_token=lambda value: str(value),
        normalize_session_token=lambda value: str(value),
        flatten_display_vector=lambda values: [str(value) for value in np.asarray(values).reshape(-1).tolist()],
        coerce_label_indices=lambda labels, expected_len: (
            np.asarray(labels, dtype=int).reshape(-1).tolist()
            if np.asarray(labels).size == int(expected_len)
            else None
        ),
    )


class GradientDialogControllerTests(unittest.TestCase):
    def test_compute_gradients_defers_projection_until_render(self):
        viewer = _ViewerStub()
        controller = _make_controller(viewer)
        nettools_stub = _NetToolsStub()

        with mock.patch.object(gradient_module, "nettools", nettools_stub), mock.patch.object(
            gradient_module, "QApplication", _QApplicationStub
        ):
            controller._compute_gradients()

        results = viewer._last_gradients
        self.assertIsNotNone(results)
        self.assertIsNone(results["projected_data"])
        self.assertEqual(results["gradients"].shape, (3, 2))
        self.assertEqual(nettools_stub.project_calls, 0)
        self.assertIn("Projection will run only when", viewer._status_bar.messages[-1])

    def test_bind_viewer_methods_uses_viewer_for_instance_method_self(self):
        viewer = _ViewerStub()
        controller = _make_controller(viewer)

        controller.bind_viewer_methods()

        self.assertIs(viewer._open_gradients_dialog.__self__, viewer)
        self.assertEqual(viewer._normalize_gradient_surface_mesh("bad-mesh"), "fsaverage4")

    def test_classification_scatter_call_sets_default_proximity_keywords(self):
        source = textwrap.dedent(inspect.getsource(GradientDialogController._classify_gradients_fsaverage))
        tree = ast.parse(source)
        scatter_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "GradientScatterDialog"
        ]
        self.assertEqual(len(scatter_calls), 1)
        keywords = {keyword.arg: keyword.value for keyword in scatter_calls[0].keywords if keyword.arg}
        self.assertIn("show_proximity_circles", keywords)
        self.assertIn("initial_proximity_slider_value", keywords)
        self.assertIn("use_line_proximity_energy", keywords)
        self.assertIn("path_metric_coords", keywords)
        self.assertIs(keywords["show_proximity_circles"].value, False)
        self.assertEqual(keywords["initial_proximity_slider_value"].value, 1000)
        self.assertIs(keywords["use_line_proximity_energy"].value, False)

    def test_classification_scatter_call_passes_matching_path_preload_choice(self):
        source = textwrap.dedent(inspect.getsource(GradientDialogController._classify_gradients_fsaverage))
        tree = ast.parse(source)
        scatter_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "GradientScatterDialog"
        ]
        self.assertEqual(len(scatter_calls), 1)
        keywords = {keyword.arg: keyword.value for keyword in scatter_calls[0].keywords if keyword.arg}
        self.assertIn("auto_preload_matching_paths", keywords)
        self.assertIsInstance(keywords["auto_preload_matching_paths"], ast.Name)

    def test_matching_path_preload_control_is_off_by_default(self):
        init_source = inspect.getsource(GradientsPrepareDialog.__init__)
        ui_source = inspect.getsource(GradientsPrepareDialog._build_ui)
        setter_source = inspect.getsource(GradientsPrepareDialog.set_triangular_rgb)

        self.assertIn("self.set_preload_matching_paths(False)", init_source)
        self.assertIn('QCheckBox("Preload matching saved path")', ui_source)
        self.assertIn("self._refresh_action_state()", setter_source)

    def test_project_paths_callback_receives_selected_path_only(self):
        dialog = GradientScatterDialog.__new__(GradientScatterDialog)
        captured = []
        dialog._project_paths_callback = captured.append
        dialog._project_paths_payload = {
            "show_all_ordered_paths": True,
            "optimal_full_path": [0, 1],
            "group_paths": [
                {
                    "group": "lh",
                    "all_full_paths": [[0, 1], [0, 2, 1]],
                    "ctx_path_energies": [0.1, 0.9],
                    "selected_ctx_path_index": 1,
                    "selected_ctx_path": [0, 2, 1],
                    "selected_ctx_path_energy": 0.9,
                    "optimal_full_path": [0, 1],
                    "ctx_optimal_path_energy": 0.1,
                    "subc_optimal_path": [],
                }
            ],
        }

        dialog._on_project_paths_clicked()

        self.assertEqual(len(captured), 1)
        payload = captured[0]
        self.assertFalse(payload["show_all_ordered_paths"])
        self.assertEqual(payload["optimal_full_path"], [0, 2, 1])
        self.assertEqual(payload["group_paths"][0]["selected_ctx_path"], [0, 2, 1])
        self.assertEqual(payload["group_paths"][0]["optimal_full_path"], [0, 2, 1])
        self.assertEqual(payload["group_paths"][0]["all_full_paths"], [[0, 2, 1]])
        self.assertEqual(payload["group_paths"][0]["ctx_path_energies"], [0.9])
        self.assertTrue(dialog._project_paths_payload["show_all_ordered_paths"])
        self.assertEqual(
            dialog._project_paths_payload["group_paths"][0]["all_full_paths"],
            [[0, 1], [0, 2, 1]],
        )

    def test_path_combo_selection_disables_all_path_overlay(self):
        class _ComboStub:
            def currentData(self):
                return 1

        class _CheckStub:
            def __init__(self):
                self.checked = True

            def blockSignals(self, _blocked):
                return None

            def setChecked(self, checked):
                self.checked = bool(checked)

        dialog = GradientScatterDialog.__new__(GradientScatterDialog)
        group_payload = {
            "group": "lh",
            "all_full_paths": [[0, 1], [0, 2, 1]],
            "ctx_path_energies": [0.1, 0.9],
            "selected_ctx_path_index": 0,
            "selected_ctx_path": [0, 1],
            "selected_ctx_path_energy": 0.1,
            "optimal_full_path": [0, 1],
            "ctx_optimal_path_energy": 0.1,
        }
        dialog._project_paths_payload = {
            "show_all_ordered_paths": True,
            "group_paths": [group_payload],
        }
        dialog._selected_ctx_path_indices = {"lh": 0, "rh": 0, "all": 0}
        dialog._show_all_ordered_paths = True
        dialog.all_paths_check = _CheckStub()
        dialog._path_selection_combo_for_group = lambda _group_name: _ComboStub()
        dialog._group_payload_for_name = lambda _group_name: group_payload
        dialog._sync_proximity_controls = lambda: None
        render_calls = []
        dialog._render = lambda **kwargs: render_calls.append(dict(kwargs))

        dialog._on_ctx_path_selection_changed("lh")

        self.assertFalse(dialog._show_all_ordered_paths)
        self.assertFalse(dialog.all_paths_check.checked)
        self.assertFalse(dialog._project_paths_payload["show_all_ordered_paths"])
        self.assertEqual(group_payload["selected_ctx_path_index"], 1)
        self.assertEqual(group_payload["selected_ctx_path"], [0, 2, 1])
        self.assertEqual(render_calls, [{"preserve_view": True}])

    def test_sync_gradients_dialog_state_reuses_precomputed_dialog_rows(self):
        source = textwrap.dedent(inspect.getsource(GradientDialogController._sync_gradients_dialog_state))
        self.assertIn("dialog_covars_rows", source)


class GradientSavedPathMatchingTests(unittest.TestCase):
    @staticmethod
    def _matching_dialog(search_root):
        dialog = GradientScatterDialog.__new__(GradientScatterDialog)
        dialog._export_metadata = {
            "subject_id": "sub-CHUVA054",
            "session_id": "ses-V1",
            "parc_path": "/tmp/chimera-LFMIHIFIS-3/chimera-LFMIHIFIS-3.nii.gz",
        }
        dialog._point_ids = np.asarray([1, 2], dtype=object)
        dialog._point_labels = np.asarray(["ctx-lh-a", "ctx-rh-a"], dtype=object)
        dialog._free_energy_paths_load_dir = Path(search_root)
        return dialog

    @staticmethod
    def _write_path_file(path, *, session="V1", scheme="chimeraLFMIHIFIS_scale3"):
        np.savez_compressed(
            path,
            subject_id=np.asarray("CHUVA054"),
            session_id=np.asarray(session),
            parc_scheme=np.asarray(scheme),
            parc_path=np.asarray(f"/tmp/{scheme}.nii.gz"),
            parcel_labels=np.asarray([1, 2]),
            parcel_names=np.asarray(["ctx-lh-a", "ctx-rh-a"]),
        )

    def test_parcellation_tokens_normalize_scale_and_filename_punctuation(self):
        hyphenated = GradientScatterDialog._canonical_parcellation_token(
            "chimera-LFMIHIFIS-3.nii.gz"
        )
        compact = GradientScatterDialog._canonical_parcellation_token(
            "chimeraLFMIHIFIS_scale3"
        )

        self.assertEqual(hyphenated, compact)

    def test_match_prefers_newest_exact_subject_session_and_parcellation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            older = root / "sub-CHUVA054_ses-V1_run-old_desc-free_energy_paths.npz"
            newest = root / "sub-CHUVA054_ses-V1_run-new_desc-free_energy_paths.npz"
            wrong_session = root / "sub-CHUVA054_ses-V2_desc-free_energy_paths.npz"
            wrong_scheme = root / "sub-CHUVA054_ses-V1_scheme-SF_desc-free_energy_paths.npz"
            self._write_path_file(older)
            self._write_path_file(newest)
            self._write_path_file(wrong_session, session="V2")
            self._write_path_file(wrong_scheme, scheme="chimeraSFMIHIFIS_scale200")
            os.utime(older, (100.0, 100.0))
            os.utime(newest, (200.0, 200.0))
            os.utime(wrong_session, (300.0, 300.0))
            os.utime(wrong_scheme, (400.0, 400.0))
            dialog = self._matching_dialog(root)

            matched_path, note = dialog._find_best_matching_free_energy_path()

            self.assertEqual(matched_path, newest)
            self.assertIn("2 matching", note)

    def test_match_requires_subject_and_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dialog = self._matching_dialog(tmpdir)
            dialog._export_metadata = {"subject_id": "CHUVA054"}

            matched_path, note = dialog._find_best_matching_free_energy_path()

            self.assertIsNone(matched_path)
            self.assertIn("no complete subject/session", note)

    def test_saved_adjacency_is_label_aligned_for_path_generation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            adjacency_path = root / "adjacency.npz"
            np.savez_compressed(
                adjacency_path,
                adjacency_mat=np.asarray(
                    [
                        [0.0, 1.0, 0.0],
                        [1.0, 0.0, 1.0],
                        [0.0, 1.0, 0.0],
                    ]
                ),
                parcel_labels=np.asarray([30, 10, 20]),
                parcel_names=np.asarray(["c", "a", "b"]),
            )
            free_energy_path = root / "sub-X_ses-Y_desc-free_energy_paths.npz"
            np.savez_compressed(
                free_energy_path,
                adjacency_path=np.asarray(str(adjacency_path)),
            )
            dialog = GradientScatterDialog.__new__(GradientScatterDialog)
            dialog._point_ids = np.asarray([10, 20, 30], dtype=object)
            dialog._point_labels = np.asarray(["a", "b", "c"], dtype=object)
            dialog._x = np.asarray([0.0, 1.0, 2.0])
            dialog._display_coords = np.column_stack((dialog._x, np.zeros(3)))
            dialog._path_metric_coords = np.asarray(dialog._display_coords, dtype=float)
            dialog._export_metadata = {}

            loaded_path = dialog._load_saved_adjacency_edges(free_energy_path)

            self.assertEqual(loaded_path, adjacency_path)
            self.assertEqual(dialog._edge_pairs.tolist(), [[0, 1], [0, 2]])
            self.assertEqual(dialog._export_metadata["adjacency_path"], str(adjacency_path))

    def test_write_directory_uses_loaded_free_energy_archive_parent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            loaded_dir = Path(tmpdir) / "loaded"
            fallback_dir = Path(tmpdir) / "fallback"
            loaded_dir.mkdir()
            fallback_dir.mkdir()
            loaded_path = loaded_dir / "sub-X_ses-Y_desc-free_energy_paths.npz"
            loaded_path.touch()
            dialog = GradientScatterDialog.__new__(GradientScatterDialog)
            dialog._loaded_fixed_endpoint_file = str(loaded_path)
            dialog._export_metadata = {"source_dir": str(fallback_dir)}

            start_dir = dialog._free_energy_write_start_dir()

            self.assertEqual(start_dir, loaded_dir)


if __name__ == "__main__":
    unittest.main()
