from pathlib import Path
import tempfile
import unittest

from window.nbs_prepare import (
    NBSPrepareDialog,
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


class NBSPrepareDialogTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
