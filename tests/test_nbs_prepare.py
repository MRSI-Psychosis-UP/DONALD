from pathlib import Path
import tempfile
import unittest

import numpy as np

import window.nbs_prepare as nbs_prepare
from window.nbs_prepare import (
    NBSPrepareDialog,
    _compact_inline_text,
    _find_covariate_balance_script,
)


class _Index:
    def __init__(self, row):
        self._row = row

    def row(self):
        return self._row


class _SelectionModel:
    def __init__(self, rows):
        self._rows = list(rows)

    def selectedRows(self):
        return [_Index(row) for row in self._rows]


class _Table:
    def __init__(self, selected_rows=(), selected_indexes=()):
        self._selection_model = _SelectionModel(selected_rows)
        self._selected_indexes = list(selected_indexes)

    def selectionModel(self):
        return self._selection_model

    def selectedIndexes(self):
        return [_Index(row) for row in self._selected_indexes]


class _LineEdit:
    def __init__(self, text):
        self._text = text

    def text(self):
        return self._text


class _Combo:
    def __init__(self, text):
        self._text = text

    def currentText(self):
        return self._text


class _Button:
    def __init__(self):
        self.enabled = None
        self.tooltip = ""

    def setEnabled(self, enabled):
        self.enabled = bool(enabled)

    def setToolTip(self, text):
        self.tooltip = str(text)


class _FakeMessageBox:
    class StandardButton:
        Close = 1
        Cancel = 2

    response = StandardButton.Close
    calls = []

    @classmethod
    def question(cls, parent, title, message, buttons, default_button):
        cls.calls.append(
            {
                "parent": parent,
                "title": title,
                "message": message,
                "buttons": buttons,
                "default_button": default_button,
            }
        )
        return cls.response


class NBSPrepareDialogTests(unittest.TestCase):
    def test_compact_inline_text_preserves_bounded_labels(self):
        text = "prefix " + ("x" * 300) + " suffix"
        compact = _compact_inline_text(text, max_chars=80)

        self.assertLessEqual(len(compact), 80)
        self.assertTrue(compact.startswith("prefix"))
        self.assertTrue(compact.endswith("suffix"))
        self.assertIn(" ... ", compact)

    def test_highlighted_data_rows_map_to_source_indices_and_skip_excluded(self):
        dialog = NBSPrepareDialog.__new__(NBSPrepareDialog)
        dialog.table = _Table(selected_rows=[0, 2, 3])
        dialog._filtered_indices = [4, 5, 6, 7]
        dialog._excluded_indices = {6}

        self.assertEqual(dialog._highlighted_data_row_indices(), [4, 7])

    def test_covariate_balance_rows_fall_back_to_included_rows_without_highlight(self):
        dialog = NBSPrepareDialog.__new__(NBSPrepareDialog)
        dialog.table = _Table()
        dialog._filtered_indices = [4, 5, 6, 7]
        dialog._excluded_indices = {5}

        self.assertEqual(dialog._covariate_balance_row_indices(), [4, 6, 7])

    def test_covariate_balance_rows_do_not_fallback_when_highlighted_rows_are_excluded(self):
        dialog = NBSPrepareDialog.__new__(NBSPrepareDialog)
        dialog.table = _Table(selected_rows=[1])
        dialog._filtered_indices = [4, 5, 6, 7]
        dialog._excluded_indices = {5}

        self.assertEqual(dialog._covariate_balance_row_indices(), [])

    def test_covariate_balance_input_uses_highlighted_table_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            dialog = NBSPrepareDialog.__new__(NBSPrepareDialog)
            dialog.table = _Table(selected_rows=[1, 3])
            dialog._source_path = tmp_root / "source_stack.npz"
            dialog._matrix_key = "matrix_subj_list"
            dialog._columns = ["participant_id", "session_id", "group", "age"]
            dialog._rows = [
                {"participant_id": "sub-001", "session_id": "ses-01", "group": "0", "age": "20"},
                {"participant_id": "sub-002", "session_id": "ses-01", "group": "1", "age": "30"},
                {"participant_id": "sub-003", "session_id": "ses-01", "group": "0", "age": "40"},
                {"participant_id": "sub-004", "session_id": "ses-01", "group": "1", "age": "50"},
            ]
            dialog._filtered_indices = [0, 1, 2, 3]
            dialog._excluded_indices = set()
            dialog.output_dir_edit = _LineEdit(str(tmp_root / "out"))
            dialog.selected_covariates = lambda: {"regressor": "group", "nuisance": ["age"]}
            dialog._selected_regressor_type = lambda: "categorical"

            input_path, _outdir, _regressor, _nuisance, using_highlighted, count = (
                dialog._build_covariate_balance_input()
            )

            self.assertTrue(using_highlighted)
            self.assertEqual(count, 2)
            written_lines = input_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(written_lines[0], "participant_id\tsession_id\tgroup\tage")
            self.assertEqual(written_lines[1:], ["sub-002\tses-01\t1\t30", "sub-004\tses-01\t1\t50"])

    def test_covariate_balance_script_discovery_checks_toolbox_root_env(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            viewer_root = tmp_root / "mrsi_viewer"
            toolbox_root = tmp_root / "mrsitoolbox"
            script_path = toolbox_root / "scripts" / "covariate_balance_love_plot.py"
            viewer_root.mkdir()
            script_path.parent.mkdir(parents=True)
            script_path.write_text("print('ok')\n", encoding="utf-8")

            found = _find_covariate_balance_script(
                viewer_root,
                environ={"MRSITOOLBOX_ROOT": str(toolbox_root)},
                cwd=viewer_root,
            )

        self.assertEqual(found, script_path)

    def test_network_filter_does_not_limit_selected_parcels_without_exclusion(self):
        dialog = NBSPrepareDialog.__new__(NBSPrepareDialog)
        dialog._parcel_rows = [
            {"Matrix index": "0", "Parcel label": "10", "Parcel name": "ctx-a"},
            {"Matrix index": "1", "Parcel label": "11", "Parcel name": "ctx-b"},
            {"Matrix index": "2", "Parcel label": "12", "Parcel name": "ctx-c"},
        ]
        dialog._filtered_parcel_indices = [1]
        dialog._excluded_parcel_indices = {2}

        self.assertEqual(dialog.selected_parcel_indices(), [0, 1])

    def test_network_filter_matches_text_substrings_case_insensitively(self):
        dialog = NBSPrepareDialog.__new__(NBSPrepareDialog)
        dialog._parcel_rows = [
            {"Matrix index": "0", "Parcel label": "10", "Parcel name": "ctx-rh-cerebellum"},
            {"Matrix index": "1", "Parcel label": "11", "Parcel name": "ctx-lh-frontal"},
            {"Matrix index": "2", "Parcel label": "12", "Parcel name": "subc-thalamus"},
        ]

        matched, target_values, error = dialog._filter_network_row_indices("Parcel name", "CER")

        self.assertIsNone(error)
        self.assertEqual(target_values, ["CER"])
        self.assertEqual(matched, [0])

    def test_network_filter_comma_separated_text_terms_match_as_or(self):
        dialog = NBSPrepareDialog.__new__(NBSPrepareDialog)
        dialog._parcel_rows = [
            {"Matrix index": "0", "Parcel label": "10", "Parcel name": "cerebellum-right"},
            {"Matrix index": "1", "Parcel label": "11", "Parcel name": "ctx-lh-frontal"},
            {"Matrix index": "2", "Parcel label": "12", "Parcel name": "subc-thalamus"},
        ]

        matched, target_values, error = dialog._filter_network_row_indices("Parcel name", "cer,ctx")

        self.assertIsNone(error)
        self.assertEqual(target_values, ["cer", "ctx"])
        self.assertEqual(matched, [0, 1])

    def test_network_filter_keeps_numeric_ranges_for_labels(self):
        dialog = NBSPrepareDialog.__new__(NBSPrepareDialog)
        dialog._parcel_rows = [
            {"Matrix index": "0", "Parcel label": "10", "Parcel name": "a"},
            {"Matrix index": "1", "Parcel label": "11", "Parcel name": "b"},
            {"Matrix index": "2", "Parcel label": "20", "Parcel name": "c"},
        ]

        matched, _target_values, error = dialog._filter_network_row_indices("Parcel label", "10-11")

        self.assertIsNone(error)
        self.assertEqual(matched, [0, 1])

    def test_covariate_balance_button_remains_clickable_when_setup_needs_attention(self):
        dialog = type("DialogStub", (), {})()
        dialog._update_run_state = NBSPrepareDialog._update_run_state.__get__(dialog)
        dialog._covariate_balance_status_text = NBSPrepareDialog._covariate_balance_status_text.__get__(dialog)
        dialog._regressor_classes = lambda: ["0", "1"]
        dialog._process_is_running = lambda: False
        dialog._export_is_running = lambda: False
        dialog._covariate_balance_is_running = lambda: False
        dialog._results_is_running = lambda: False
        dialog.selected_parcel_indices = lambda: [0, 1]
        dialog._last_result_npz_path = None
        dialog.RESULTS_STEP_INDEX = NBSPrepareDialog.RESULTS_STEP_INDEX
        dialog.RUN_STEP_INDEX = NBSPrepareDialog.RUN_STEP_INDEX
        dialog.STEP_TITLES = NBSPrepareDialog.STEP_TITLES
        dialog._current_step = NBSPrepareDialog.RUN_STEP_INDEX
        dialog._step_buttons = []
        dialog.covariate_balance_button = _Button()
        dialog.selected_covariates = lambda: {"regressor": "Diag", "nuisance": ["Age"]}
        dialog._selected_regressor_type = lambda: "categorical"
        dialog._highlighted_data_table_rows = lambda: [0]
        dialog._highlighted_data_row_indices = lambda: [0]
        dialog.selected_row_indices = lambda: [0, 1, 2]

        dialog._update_run_state()

        self.assertTrue(dialog.covariate_balance_button.enabled)
        self.assertIn("Select at least two highlighted Data table rows", dialog.covariate_balance_button.tooltip)

    def test_confirm_close_uses_cancel_default_and_accepts_close(self):
        dialog = NBSPrepareDialog.__new__(NBSPrepareDialog)
        dialog._process_is_running = lambda: False
        dialog._export_is_running = lambda: False
        dialog._covariate_balance_is_running = lambda: False
        dialog._results_is_running = lambda: False

        original_message_box = nbs_prepare.QMessageBox
        try:
            _FakeMessageBox.calls = []
            _FakeMessageBox.response = _FakeMessageBox.StandardButton.Close
            nbs_prepare.QMessageBox = _FakeMessageBox

            self.assertTrue(dialog._confirm_close_requested())

            self.assertEqual(_FakeMessageBox.calls[0]["default_button"], _FakeMessageBox.StandardButton.Cancel)
            self.assertIn("Close the NBS window?", _FakeMessageBox.calls[0]["message"])
        finally:
            nbs_prepare.QMessageBox = original_message_box

    def test_confirm_close_cancel_blocks_running_task_close(self):
        dialog = NBSPrepareDialog.__new__(NBSPrepareDialog)
        dialog._process_is_running = lambda: True
        dialog._export_is_running = lambda: False
        dialog._covariate_balance_is_running = lambda: False
        dialog._results_is_running = lambda: False

        original_message_box = nbs_prepare.QMessageBox
        try:
            _FakeMessageBox.calls = []
            _FakeMessageBox.response = _FakeMessageBox.StandardButton.Cancel
            nbs_prepare.QMessageBox = _FakeMessageBox

            self.assertFalse(dialog._confirm_close_requested())

            self.assertIn("NBS run is still running", _FakeMessageBox.calls[0]["message"])
        finally:
            nbs_prepare.QMessageBox = original_message_box

    def test_component_pvalue_rows_preserve_two_tail_summary_components(self):
        rows = NBSPrepareDialog._component_pvalue_rows(
            summary={
                "components": [
                    {
                        "component_idx": 0,
                        "tail": "right",
                        "tail_component_idx": 0,
                        "pvalue": 0.08,
                        "pvalue_uncorrected": 0.04,
                        "significant": False,
                    },
                    {
                        "component_idx": 1,
                        "tail": "left",
                        "tail_component_idx": 0,
                        "pvalue": 0.02,
                        "pvalue_uncorrected": 0.01,
                        "significant": True,
                    },
                ],
            }
        )

        self.assertEqual(
            rows,
            [
                {
                    "component_idx": 0,
                    "tail": "right",
                    "tail_component_idx": 0,
                    "pvalue": 0.08,
                    "pvalue_uncorrected": 0.04,
                    "significant": False,
                },
                {
                    "component_idx": 1,
                    "tail": "left",
                    "tail_component_idx": 0,
                    "pvalue": 0.02,
                    "pvalue_uncorrected": 0.01,
                    "significant": True,
                },
            ],
        )

    def test_component_pvalue_rows_fall_back_to_result_npz_tail_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = Path(tmpdir) / "components_all.npz"
            np.savez(
                result_path,
                comp_pvals=np.asarray([0.12, 0.44]),
                comp_pvals_uncorrected=np.asarray([0.06, 0.22]),
                comp_tails=np.asarray(["right", "left"], dtype=object),
                comp_tail_indices=np.asarray([0, 1]),
                sig_indices=np.asarray([0]),
            )

            rows = NBSPrepareDialog._component_pvalue_rows(result_path=result_path)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["tail"], "right")
        self.assertEqual(rows[1]["tail"], "left")
        self.assertEqual(rows[0]["pvalue"], 0.12)
        self.assertEqual(rows[0]["pvalue_uncorrected"], 0.06)
        self.assertTrue(rows[0]["significant"])
        self.assertFalse(rows[1]["significant"])

    def test_filtered_conn_subset_removes_excluded_parcel_rows_and_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            source_path = tmp_root / "source_stack.npz"
            matrix = np.arange(3 * 4 * 4, dtype=float).reshape(3, 4, 4)
            profiles = np.arange(3 * 4 * 2, dtype=float).reshape(3, 4, 2)
            covars = np.asarray(
                [
                    ("sub-001", "V1", 0),
                    ("sub-002", "V1", 1),
                    ("sub-003", "V1", 0),
                ],
                dtype=[("participant_id", "U12"), ("session_id", "U4"), ("Diag", "i4")],
            )
            np.savez(
                source_path,
                matrix_subj_list=matrix,
                covars=covars,
                parcel_labels_group=np.asarray([10, 11, 12, 13]),
                parcel_names_group=np.asarray(["ctx-a", "ctx-b", "ctx-c", "ctx-d"], dtype=object),
                metabolites=np.asarray(["CrPCr", "GluGln"], dtype=object),
                metab_profiles_subj_list=profiles,
                group=np.asarray("test-group"),
                modality=np.asarray("mrsi"),
            )

            dialog = NBSPrepareDialog.__new__(NBSPrepareDialog)
            dialog._source_path = source_path
            dialog._matrix_key = "matrix_subj_list"
            dialog._filtered_indices = [0, 2]
            dialog._excluded_indices = set()
            dialog._excluded_parcel_indices = {1, 3}
            dialog._parcel_rows = [
                {"Matrix index": "0", "Parcel label": "10", "Parcel name": "ctx-a"},
                {"Matrix index": "1", "Parcel label": "11", "Parcel name": "ctx-b"},
                {"Matrix index": "2", "Parcel label": "12", "Parcel name": "ctx-c"},
                {"Matrix index": "3", "Parcel label": "13", "Parcel name": "ctx-d"},
            ]
            dialog.modality_combo = _Combo("mrsi")
            dialog._regressor_name = lambda: "Diag"

            subset_path = dialog._build_filtered_conn_subset()

            self.assertEqual(subset_path.parent, tmp_root / "derivatives" / "nbs_work" / "mrsi")
            self.assertTrue(subset_path.name.endswith("_key-matrix_subj_list_reg-Diag_n-2_p-2_nbs_input.npz"))

            with np.load(subset_path, allow_pickle=True) as out:
                expected_matrix = matrix[[0, 2]][:, [0, 2]][:, :, [0, 2]]
                np.testing.assert_array_equal(out["matrix_subj_list"], expected_matrix)
                np.testing.assert_array_equal(out["matrix_pop_avg"], expected_matrix.mean(axis=0))
                np.testing.assert_array_equal(out["metab_profiles_subj_list"], profiles[[0, 2]][:, [0, 2], :])
                np.testing.assert_array_equal(out["parcel_labels_group"], np.asarray([10, 12]))
                np.testing.assert_array_equal(out["parcel_names_group"], np.asarray(["ctx-a", "ctx-c"], dtype=object))
                np.testing.assert_array_equal(out["parcel_keep_indices"], np.asarray([0, 2]))
                np.testing.assert_array_equal(out["parcel_excluded_indices"], np.asarray([1, 3]))
                np.testing.assert_array_equal(out["parcel_excluded_labels"], np.asarray([11, 13]))
                np.testing.assert_array_equal(
                    out["parcel_excluded_names"],
                    np.asarray(["ctx-b", "ctx-d"], dtype=object),
                )

    def test_nbs_work_dir_uses_existing_derivatives_root(self):
        dialog = NBSPrepareDialog.__new__(NBSPrepareDialog)
        dialog._source_path = Path(
            "/tmp/project/derivatives/group/connectivity/mrsi/source_stack.npz"
        )

        self.assertEqual(
            dialog._nbs_work_dir("MRSI"),
            Path("/tmp/project/derivatives/nbs_work/mrsi"),
        )


if __name__ == "__main__":
    unittest.main()
