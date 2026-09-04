#!/usr/bin/env python3
"""Validate python MATLAB-compatible NBS outputs against MATLAB NBS1.2."""

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
from scipy.io import loadmat


SCRIPT_PATH = Path(__file__).resolve()
VIEWER_ROOT = SCRIPT_PATH.parents[1]

try:
    from mrsitoolbox.connectomics.nbs import NBS  # noqa: E402
except ImportError as exc:
    raise ImportError(
        "validate_nbs_parity.py requires the current mrsitoolbox pip package. "
        "Install or upgrade it with: pip install --upgrade mrsitoolbox"
    ) from exc


def _mat_get(obj, field, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(field, default)
    if hasattr(obj, field):
        return getattr(obj, field)
    if isinstance(obj, np.void) and obj.dtype.names and field in obj.dtype.names:
        return obj[field].item()
    return default


def _cell_to_list(cell_obj):
    if cell_obj is None:
        return []
    if isinstance(cell_obj, (list, tuple)):
        return list(cell_obj)
    arr = np.asarray(cell_obj)
    if arr.dtype == object:
        return [item for item in arr.ravel()]
    return [arr]


def _to_dense_bool(mat):
    if mat is None:
        return None
    try:
        from scipy import sparse

        if sparse.issparse(mat):
            mat = mat.toarray()
    except Exception:
        pass
    mat = np.asarray(mat)
    mat = np.squeeze(mat)
    if mat.ndim == 1:
        n = int(np.sqrt(mat.size))
        if n * n == mat.size:
            mat = mat.reshape(n, n)
    if mat.ndim != 2:
        return None
    return mat != 0


def _load_matlab_result(mat_path: Path, alpha: float):
    data = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    nbs = data.get("nbs")
    if nbs is None:
        raise ValueError(f"Missing `nbs` struct in MATLAB result: {mat_path}")
    nbs_struct = _mat_get(nbs, "NBS")
    if nbs_struct is None:
        raise ValueError(f"Missing `NBS` field in MATLAB result: {mat_path}")

    pval = _mat_get(nbs_struct, "pval")
    con_mat = _mat_get(nbs_struct, "con_mat")
    test_stat = _mat_get(nbs_struct, "test_stat")

    pvals_list = []
    if pval is not None and np.size(pval) > 0:
        pvals_list = np.atleast_1d(np.asarray(pval, dtype=float)).tolist()

    comp_pvals = []
    comp_masks = []
    for idx, comp in enumerate(_cell_to_list(con_mat)):
        mat_bool = _to_dense_bool(comp)
        if mat_bool is None:
            continue
        mat_bool = np.asarray(mat_bool, dtype=bool)
        if mat_bool.ndim != 2 or mat_bool.shape[0] == 0 or mat_bool.shape[1] == 0:
            continue
        mat_bool = mat_bool | mat_bool.T
        comp_masks.append(mat_bool)
        comp_pvals.append(float(pvals_list[idx]) if idx < len(pvals_list) else 1.0)

    if test_stat is None or np.size(test_stat) == 0:
        raise ValueError("MATLAB result missing test_stat matrix.")
    t_mat = np.asarray(test_stat, dtype=float)
    t_mat = np.squeeze(t_mat)
    if t_mat.ndim != 2:
        raise ValueError("MATLAB result test_stat is not 2D.")

    sig_mask = np.zeros_like(t_mat, dtype=bool)
    for pval_i, mask in zip(comp_pvals, comp_masks):
        if pval_i <= alpha and mask.shape == sig_mask.shape:
            sig_mask |= mask

    return {
        "t_mat": t_mat,
        "comp_pvals": comp_pvals,
        "comp_masks": comp_masks,
        "sig_mask": sig_mask,
    }


def _load_export_matrices(export_dir: Path):
    matrices_dir = export_dir / "matrices"
    if not matrices_dir.is_dir():
        raise FileNotFoundError(f"Missing matrices folder: {matrices_dir}")
    files = sorted(
        matrices_dir.glob("subject*.txt"),
        key=lambda path: int(re.search(r"subject(\d+)", path.name).group(1))
        if re.search(r"subject(\d+)", path.name)
        else path.name,
    )
    if not files:
        raise FileNotFoundError(f"No subject*.txt files found in: {matrices_dir}")
    mats = [np.loadtxt(str(path), dtype=float) for path in files]
    n = mats[0].shape[0]
    for idx, mat in enumerate(mats):
        if mat.shape != (n, n):
            raise ValueError(f"Matrix shape mismatch at index {idx}: {mat.shape} != {(n, n)}")
    return np.stack(mats, axis=2)


def _parse_contrast(raw: str):
    text = str(raw).strip()
    if not text:
        raise ValueError("--contrast is required.")
    if ";" in text or "\n" in text:
        raise ValueError("Only single-row contrasts are supported.")
    if (text.startswith("[") and text.endswith("]")) or (
        text.startswith("(") and text.endswith(")")
    ):
        text = text[1:-1].strip()
    parts = [token for token in re.split(r"[,\s]+", text) if token]
    if not parts:
        raise ValueError("Could not parse contrast.")
    return np.asarray([float(token) for token in parts], dtype=float)


def _escape_matlab_string(value: str):
    return str(value).replace("'", "''")


def _run_matlab_nbs(
    export_dir: Path,
    matlab_cmd: str,
    matlab_nbs_path: str,
    contrast: str,
    test: str,
    size: str,
    thresh: float,
    alpha: float,
    perms: int,
    nthreads: int,
    output_mat: str,
):
    helper_dir = VIEWER_ROOT / "window"
    helper_script = helper_dir / "nbs_run_cli.m"
    if not helper_script.is_file():
        raise FileNotFoundError(f"Missing nbs_run_cli.m at {helper_script}")

    call = (
        "addpath('{helper}');"
        "nbs_run_cli('export_dir','{export_dir}',"
        "'nbs_path','{nbs_path}',"
        "'contrast','{contrast}',"
        "'test','{test}',"
        "'size','{size}',"
        "'thresh',{thresh},"
        "'alpha',{alpha},"
        "'perms',{perms},"
        "'tail','right',"
        "'nthreads',{nthreads},"
        "'no_precompute',true,"
        "'output_mat','{output_mat}')"
    ).format(
        helper=_escape_matlab_string(str(helper_dir)),
        export_dir=_escape_matlab_string(str(export_dir)),
        nbs_path=_escape_matlab_string(str(matlab_nbs_path)),
        contrast=_escape_matlab_string(str(contrast)),
        test=_escape_matlab_string(str(test)),
        size=_escape_matlab_string(str(size)),
        thresh=float(thresh),
        alpha=float(alpha),
        perms=int(perms),
        nthreads=max(1, int(nthreads)),
        output_mat=_escape_matlab_string(str(output_mat)),
    )
    cmd = [str(matlab_cmd), "-batch", call]
    subprocess.run(cmd, check=True)


def _component_mask_union(comp_masks):
    if not comp_masks:
        return None
    out = np.zeros_like(np.asarray(comp_masks[0], dtype=bool))
    for mask in comp_masks:
        mask_b = np.asarray(mask, dtype=bool)
        if mask_b.shape == out.shape:
            out |= mask_b
    return out


def _match_components(py_masks, mat_masks, min_overlap=0.5):
    """Greedily pair python/MATLAB significant components by edge-mask Jaccard overlap.

    MATLAB and the python port do not necessarily enumerate components in the same
    order, so comparing `comp_pvals[i]` to `comp_pvals[i]` directly is meaningless.
    Matching by mask overlap first is required to compare p-values component-by-component.

    Returns (matches, unmatched_py, unmatched_mat) where matches is a list of
    (py_idx, mat_idx, overlap_jaccard) and the unmatched lists are component indices
    present in only one engine's output.
    """
    n_py = len(py_masks)
    n_mat = len(mat_masks)
    overlap = np.zeros((n_py, n_mat), dtype=float)
    for i in range(n_py):
        pm = np.asarray(py_masks[i], dtype=bool)
        for j in range(n_mat):
            mm = np.asarray(mat_masks[j], dtype=bool)
            if pm.shape != mm.shape:
                continue
            inter = np.logical_and(pm, mm).sum()
            union = np.logical_or(pm, mm).sum()
            overlap[i, j] = float(inter / union) if union else 0.0

    candidates = sorted(
        ((overlap[i, j], i, j) for i in range(n_py) for j in range(n_mat)),
        reverse=True,
    )
    matches = []
    used_py, used_mat = set(), set()
    for ov, i, j in candidates:
        if ov < min_overlap:
            break
        if i in used_py or j in used_mat:
            continue
        matches.append((i, j, ov))
        used_py.add(i)
        used_mat.add(j)
    unmatched_py = [i for i in range(n_py) if i not in used_py]
    unmatched_mat = [j for j in range(n_mat) if j not in used_mat]
    return matches, unmatched_py, unmatched_mat


def main():
    parser = argparse.ArgumentParser(
        description="Compare python MATLAB-compatible NBS output with MATLAB NBS1.2."
    )
    parser.add_argument("--export-dir", required=True, help="Path to MATLAB export folder.")
    parser.add_argument(
        "--contrast",
        required=True,
        help="Single-row contrast string, e.g. '0 0 0 1'.",
    )
    parser.add_argument("--test", default="t", choices=["t", "F"], help="Test type.")
    parser.add_argument(
        "--size",
        default="extent",
        choices=["extent", "intensity"],
        help="Component size measure.",
    )
    parser.add_argument("--thresh", type=float, default=3.5, help="Primary threshold.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Corrected alpha.")
    parser.add_argument("--perms", type=int, default=1000, help="Permutations.")
    parser.add_argument("--nthreads", type=int, default=8, help="CPU threads.")
    parser.add_argument("--seed", type=int, default=None, help="Optional python RNG seed.")
    parser.add_argument(
        "--matlab-cmd",
        default=os.getenv("MRSI_MATLAB_CMD") or os.getenv("MATLAB_CMD") or shutil.which("matlab") or "",
        help="MATLAB executable.",
    )
    parser.add_argument(
        "--matlab-nbs-path",
        default=os.getenv("MRSI_NBS_PATH") or os.getenv("MATLAB_NBS_PATH") or os.getenv("NBS_PATH") or "",
        help="Path containing NBSrun.m.",
    )
    parser.add_argument(
        "--matlab-output-mat",
        default=None,
        help="Optional existing MATLAB result .mat file. If omitted, MATLAB will be executed.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-8,
        help="Maximum absolute difference allowed for observed test-statistics.",
    )
    parser.add_argument(
        "--min-jaccard",
        type=float,
        default=0.95,
        help="Minimum Jaccard overlap required between the python and MATLAB "
             "significant-edge masks.",
    )
    parser.add_argument(
        "--pval-tolerance",
        type=float,
        default=0.02,
        help="Maximum absolute difference allowed between matched components' corrected "
             "p-values (matched by mask overlap, not index). Widen this at low --perms, "
             "since independent permutation draws add Monte-Carlo noise to both engines.",
    )
    parser.add_argument(
        "--min-component-overlap",
        type=float,
        default=0.5,
        help="Minimum edge-mask Jaccard overlap for two components (one per engine) to be "
             "considered the same component when matching for the p-value comparison.",
    )
    args = parser.parse_args()

    export_dir = Path(args.export_dir).expanduser().resolve()
    if not export_dir.is_dir():
        raise FileNotFoundError(f"Export directory not found: {export_dir}")

    design_path = export_dir / "designMatrix.txt"
    if not design_path.is_file():
        raise FileNotFoundError(f"Missing design matrix file: {design_path}")
    design_matrix = np.loadtxt(str(design_path), dtype=float)
    if design_matrix.ndim == 1:
        design_matrix = design_matrix.reshape(-1, 1)

    matrices = _load_export_matrices(export_dir)
    contrast_vec = _parse_contrast(args.contrast)

    nbs = NBS()
    py_res = nbs.bct_glm_matlab_compat(
        matrices,
        design_matrix=design_matrix,
        contrast=contrast_vec,
        test=args.test,
        size=args.size,
        t_thresh=args.thresh,
        n_perms=args.perms,
        nthreads=args.nthreads,
        alpha=args.alpha,
        seed=args.seed,
        return_significant_only=True,
    )

    if args.matlab_output_mat:
        matlab_mat_path = Path(args.matlab_output_mat).expanduser().resolve()
        if not matlab_mat_path.is_file():
            raise FileNotFoundError(f"MATLAB result file not found: {matlab_mat_path}")
    else:
        matlab_cmd = str(args.matlab_cmd or "").strip()
        matlab_nbs_path = str(args.matlab_nbs_path or "").strip()
        if not matlab_cmd:
            raise ValueError("MATLAB executable missing. Set --matlab-cmd or env MRSI_MATLAB_CMD/MATLAB_CMD.")
        if not matlab_nbs_path:
            raise ValueError("NBS path missing. Set --matlab-nbs-path or env MRSI_NBS_PATH/MATLAB_NBS_PATH.")
        matlab_mat_path = export_dir / "nbs_results_validate.mat"
        _run_matlab_nbs(
            export_dir=export_dir,
            matlab_cmd=matlab_cmd,
            matlab_nbs_path=matlab_nbs_path,
            contrast=args.contrast,
            test=args.test,
            size=args.size,
            thresh=args.thresh,
            alpha=args.alpha,
            perms=args.perms,
            nthreads=args.nthreads,
            output_mat=matlab_mat_path.name,
        )
    mat_res = _load_matlab_result(matlab_mat_path, alpha=args.alpha)

    iu = np.triu_indices(py_res["t_mat"].shape[0], 1)
    py_flat = np.asarray(py_res["t_mat"], dtype=float)[iu]
    mat_flat = np.asarray(mat_res["t_mat"], dtype=float)[iu]
    max_abs_t_diff = float(np.max(np.abs(py_flat - mat_flat)))
    mae_t = float(np.mean(np.abs(py_flat - mat_flat)))
    corr_t = float(np.corrcoef(py_flat, mat_flat)[0, 1]) if py_flat.size else float("nan")

    py_sig_mask = _component_mask_union(py_res["comp_masks"])
    mat_sig_mask = _component_mask_union(mat_res["comp_masks"])
    if py_sig_mask is None and mat_sig_mask is None:
        jaccard = 1.0
    elif py_sig_mask is None or mat_sig_mask is None:
        jaccard = 0.0
    else:
        inter = np.logical_and(py_sig_mask, mat_sig_mask).sum()
        union = np.logical_or(py_sig_mask, mat_sig_mask).sum()
        jaccard = float(inter / union) if union else 1.0

    matches, unmatched_py, unmatched_mat = _match_components(
        py_res["comp_masks"], mat_res["comp_masks"], min_overlap=args.min_component_overlap
    )
    pval_diffs = [
        abs(float(py_res["comp_pvals"][i]) - float(mat_res["comp_pvals"][j]))
        for i, j, _ov in matches
    ]
    max_pval_diff = max(pval_diffs) if pval_diffs else 0.0

    print("=== NBS Parity Validation ===")
    print(f"Export dir: {export_dir}")
    print(f"MATLAB result: {matlab_mat_path}")
    print(f"Observed t-stat corr: {corr_t:.8f}")
    print(f"Observed t-stat MAE:  {mae_t:.8g}")
    print(f"Observed t-stat max|Δ|: {max_abs_t_diff:.8g}")
    print(f"Python significant components: {len(py_res['comp_pvals'])}")
    print(f"MATLAB significant components: {len(mat_res['comp_pvals'])}")
    print(f"Significant-edge Jaccard: {jaccard:.6f}")
    print(
        f"Matched components: {len(matches)}  "
        f"(python-only: {len(unmatched_py)}, matlab-only: {len(unmatched_mat)})"
    )
    if pval_diffs:
        print(f"Matched-component p-value max|Δ|: {max_pval_diff:.6g}")
    if py_res["comp_pvals"]:
        print("Python p-values:", ", ".join(f"{float(v):.6g}" for v in py_res["comp_pvals"]))
    else:
        print("Python p-values: none")
    if mat_res["comp_pvals"]:
        print("MATLAB p-values:", ", ".join(f"{float(v):.6g}" for v in mat_res["comp_pvals"]))
    else:
        print("MATLAB p-values: none")

    failures = []
    if max_abs_t_diff > float(args.tolerance):
        failures.append(
            f"observed test-statistics differ by max {max_abs_t_diff:.8g} "
            f"(tolerance {float(args.tolerance):.8g})"
        )
    if jaccard < float(args.min_jaccard):
        failures.append(
            f"significant-edge Jaccard {jaccard:.6f} below --min-jaccard "
            f"{float(args.min_jaccard):.6f}"
        )
    if unmatched_py or unmatched_mat:
        failures.append(
            f"component structure mismatch: {len(unmatched_py)} python-only, "
            f"{len(unmatched_mat)} matlab-only component(s) with no overlap "
            f">= {float(args.min_component_overlap):.2f}"
        )
    if max_pval_diff > float(args.pval_tolerance):
        failures.append(
            f"matched-component p-values differ by max {max_pval_diff:.6g} "
            f"(tolerance {float(args.pval_tolerance):.6g})"
        )

    if failures:
        raise SystemExit("FAIL: " + "; ".join(failures) + ".")
    print("PASS: statistics, component structure and p-values all match within tolerance.")


if __name__ == "__main__":
    main()
