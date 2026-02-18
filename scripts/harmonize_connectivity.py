#!/usr/bin/env python3
"""Connectivity harmonization helpers using neuroCombat.

This module can be used from the GUI or as a lightweight CLI.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from neuroCombat.neuroCombat import neuroCombat
except Exception:
    neuroCombat = None

from matplotlib import pyplot as plt

try:
    from mrsitoolbox.graphplot.simmatrix import SimMatrixPlot
except Exception:
    SimMatrixPlot = None


def _decode_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode()
        except Exception:
            return str(value)
    return value


def _display_text(value: Any) -> str:
    value = _decode_scalar(value)
    if value is None:
        return ""
    return str(value)


def _stack_axis(shape: Sequence[int]) -> Optional[int]:
    if len(shape) != 3:
        return None
    a, b, c = shape
    if a == b != c:
        return 2
    if a == c != b:
        return 1
    if b == c != a:
        return 0
    return None


def _to_subject_first_stack(matrix_3d: np.ndarray, axis: int) -> np.ndarray:
    if axis == 0:
        return np.asarray(matrix_3d, dtype=float)
    if axis == 1:
        return np.asarray(matrix_3d.transpose(1, 0, 2), dtype=float)
    return np.asarray(matrix_3d.transpose(2, 0, 1), dtype=float)


def _slugify(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")
    return text or "na"


def _npz_scalar_text(npz, key: str) -> str:
    if key not in npz:
        return ""
    value = np.asarray(npz[key])
    if value.ndim == 0:
        return _display_text(value.item())
    if value.size == 1:
        return _display_text(value.reshape(-1)[0])
    return ""


def _ensure_pandas() -> None:
    if pd is None:
        raise ImportError("pandas is required for harmonization.")


def _ensure_neurocombat() -> None:
    if neuroCombat is None:
        raise ImportError("neuroCombat is not available in this environment.")


def _resolve_column_name(df, requested: str) -> str:
    wanted = str(requested or "").strip()
    if wanted == "":
        raise ValueError("Column name cannot be empty.")
    if wanted in df.columns:
        return wanted
    col_map = {str(col).strip().lower(): str(col) for col in df.columns}
    resolved = col_map.get(wanted.lower())
    if resolved is None:
        raise ValueError(f"Column '{requested}' not found in covariates.")
    return resolved


def _looks_numeric(values: Iterable[Any]) -> bool:
    has_value = False
    for value in values:
        text = _display_text(value).strip()
        if text == "":
            continue
        has_value = True
        try:
            float(text)
        except Exception:
            return False
    return has_value


def infer_covariate_type(values: Iterable[Any]) -> str:
    return "continuous" if _looks_numeric(values) else "categorical"


def _ordered_unique_indices(indices: np.ndarray) -> np.ndarray:
    seen = set()
    ordered = []
    for value in indices.tolist():
        idx = int(value)
        if idx in seen:
            continue
        seen.add(idx)
        ordered.append(idx)
    return np.asarray(ordered, dtype=int)


def _extract_subject_session(npz, covars_df, selected_indices: np.ndarray, stack_len: int):
    subj_arr = None
    ses_arr = None

    if "subject_id_list" in npz:
        candidate = np.asarray(npz["subject_id_list"])
        if candidate.ndim == 1 and candidate.shape[0] == stack_len:
            subj_arr = np.asarray([_display_text(v) for v in candidate[selected_indices]], dtype=object)

    if "session_id_list" in npz:
        candidate = np.asarray(npz["session_id_list"])
        if candidate.ndim == 1 and candidate.shape[0] == stack_len:
            ses_arr = np.asarray([_display_text(v) for v in candidate[selected_indices]], dtype=object)

    if subj_arr is None:
        for col in ("participant_id", "subject_id", "Code"):
            if col in covars_df.columns:
                subj_arr = np.asarray(
                    [_display_text(v) for v in covars_df.iloc[selected_indices][col].to_numpy()],
                    dtype=object,
                )
                break
    if ses_arr is None:
        for col in ("session_id", "ses"):
            if col in covars_df.columns:
                ses_arr = np.asarray(
                    [_display_text(v) for v in covars_df.iloc[selected_indices][col].to_numpy()],
                    dtype=object,
                )
                break

    if subj_arr is None:
        subj_arr = np.asarray([f"sub-{idx + 1:04d}" for idx in selected_indices], dtype=object)
    if ses_arr is None:
        ses_arr = np.asarray(["ses-NA" for _ in selected_indices], dtype=object)

    return subj_arr, ses_arr


def load_matrix_stack_from_npz(
    source_path: Path | str,
    matrix_key: str,
    selected_indices: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Load a matrix stack and covariates from NPZ, aligned on subject rows."""

    _ensure_pandas()
    src = Path(source_path)
    if not src.is_file():
        raise FileNotFoundError(f"Input NPZ not found: {src}")

    with np.load(src, allow_pickle=True) as npz:
        if matrix_key not in npz:
            raise KeyError(f"Matrix key '{matrix_key}' not found in {src.name}.")
        raw = np.asarray(npz[matrix_key], dtype=float)
        if raw.ndim != 3:
            raise ValueError("Harmonization requires a 3D matrix stack.")

        axis = _stack_axis(raw.shape)
        if axis is None:
            raise ValueError("Selected matrix is not a stack of square matrices.")

        stack_len = int(raw.shape[axis])
        if stack_len <= 1:
            raise ValueError("Harmonization requires at least two matrices.")

        if "covars" not in npz:
            raise ValueError("Source NPZ is missing covars.")
        covars_raw = np.asarray(npz["covars"])
        covars_df = pd.DataFrame.from_records(covars_raw)
        if covars_df.shape[0] != stack_len:
            raise ValueError("Covars length does not match matrix stack length.")

        if selected_indices is None:
            selected = np.arange(stack_len, dtype=int)
        else:
            selected = np.asarray(list(selected_indices), dtype=int).reshape(-1)
            selected = _ordered_unique_indices(selected)

        if selected.size == 0:
            raise ValueError("No rows selected for harmonization.")
        if int(selected.min()) < 0 or int(selected.max()) >= stack_len:
            raise ValueError("Selected row index is outside matrix stack bounds.")

        matrix_stack = _to_subject_first_stack(raw, axis)
        matrix_stack = np.asarray(matrix_stack[selected], dtype=float)
        covars_sel = covars_df.iloc[selected].reset_index(drop=True)

        subject_id_list, session_id_list = _extract_subject_session(npz, covars_df, selected, stack_len)

        if "parcel_labels_group" in npz:
            parcel_labels = np.asarray(npz["parcel_labels_group"])
        elif "parcel_labels_group.npy" in npz:
            parcel_labels = np.asarray(npz["parcel_labels_group.npy"])
        else:
            parcel_labels = None
        if "parcel_names_group" in npz:
            parcel_names = np.asarray(npz["parcel_names_group"])
        elif "parcel_names_group.npy" in npz:
            parcel_names = np.asarray(npz["parcel_names_group.npy"])
        else:
            parcel_names = None
        group = _npz_scalar_text(npz, "group")
        modality = _npz_scalar_text(npz, "modality").lower()
        metabolites = np.asarray(npz["metabolites"]) if "metabolites" in npz else None

        metab_profiles = None
        if "metab_profiles_subj_list" in npz:
            mp = np.asarray(npz["metab_profiles_subj_list"])
            if mp.ndim >= 1 and mp.shape[0] == stack_len:
                metab_profiles = np.asarray(mp[selected])

    return {
        "source_path": src,
        "matrix_key": str(matrix_key),
        "selected_indices": np.asarray(selected, dtype=int),
        "matrix_stack": np.asarray(matrix_stack, dtype=float),
        "covars_df": covars_sel,
        "subject_id_list": np.asarray(subject_id_list, dtype=object),
        "session_id_list": np.asarray(session_id_list, dtype=object),
        "parcel_labels_group": parcel_labels,
        "parcel_names_group": parcel_names,
        "group": str(group or ""),
        "modality": str(modality or ""),
        "metabolites": metabolites,
        "metab_profiles_subj_list": metab_profiles,
    }


def _impute_covariates(covars_df, categorical_cols: Sequence[str], continuous_cols: Sequence[str]):
    covars = covars_df.copy()

    for col in continuous_cols:
        numeric = pd.to_numeric(covars[col], errors="coerce")
        if numeric.isna().all():
            raise ValueError(f"Continuous covariate '{col}' has only missing/non-numeric values.")
        fill_value = float(numeric.median())
        covars[col] = numeric.fillna(fill_value)

    for col in categorical_cols:
        series = covars[col].astype(object)
        text_vals = series.apply(lambda v: _display_text(v).strip())
        missing_mask = text_vals == ""
        if bool(missing_mask.any()):
            non_missing = text_vals.loc[~missing_mask]
            if non_missing.empty:
                fill_value = "Unknown"
            else:
                mode_vals = non_missing.mode(dropna=True)
                fill_value = _display_text(mode_vals.iloc[0]) if not mode_vals.empty else "Unknown"
            text_vals.loc[missing_mask] = fill_value
        covars[col] = text_vals.astype(str)

    return covars


def harmonize_matrix_stack(
    matrix_stack: np.ndarray,
    covars_df,
    batch_col: str,
    categorical_cols: Optional[Sequence[str]] = None,
    continuous_cols: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Run neuroCombat on a subject-first matrix stack (N, P, P)."""

    _ensure_pandas()
    _ensure_neurocombat()

    stack = np.asarray(matrix_stack, dtype=float)
    if stack.ndim != 3:
        raise ValueError("matrix_stack must be 3D (subjects, parcels, parcels).")
    n_subjects, n_nodes, n_nodes2 = stack.shape
    if n_nodes != n_nodes2:
        raise ValueError("matrix_stack must contain square matrices.")
    if n_subjects <= 1:
        raise ValueError("At least two subjects are required.")

    resolved_batch = _resolve_column_name(covars_df, batch_col)
    categorical_cols = list(categorical_cols or [])
    continuous_cols = list(continuous_cols or [])
    resolved_cat = [_resolve_column_name(covars_df, col) for col in categorical_cols]
    resolved_cont = [_resolve_column_name(covars_df, col) for col in continuous_cols]

    overlap = set(resolved_cat) & set(resolved_cont)
    if overlap:
        raise ValueError(f"Covariates cannot be both categorical and continuous: {sorted(overlap)}")

    resolved_cat = [col for col in resolved_cat if col != resolved_batch]
    resolved_cont = [col for col in resolved_cont if col != resolved_batch]

    model_cols = [resolved_batch] + resolved_cat + resolved_cont
    model_cols = list(dict.fromkeys(model_cols))
    covars_model = covars_df.loc[:, model_cols].copy()
    covars_model = _impute_covariates(covars_model, resolved_cat + [resolved_batch], resolved_cont)

    batch_values = covars_model[resolved_batch].astype(str).to_numpy()
    if np.unique(batch_values).size < 2:
        raise ValueError(f"Batch column '{resolved_batch}' must contain at least two distinct values.")

    dat = stack.reshape(n_subjects, n_nodes * n_nodes).T
    combat_result = neuroCombat(
        dat=dat,
        covars=covars_model,
        batch_col=resolved_batch,
        categorical_cols=resolved_cat,
        continuous_cols=resolved_cont,
    )

    harmonized_data = np.asarray(combat_result["data"], dtype=float)
    harmonized_stack = harmonized_data.T.reshape(n_subjects, n_nodes, n_nodes)
    harmonized_stack = np.nan_to_num(harmonized_stack, nan=0.0, posinf=0.0, neginf=0.0)

    # Keep matrices symmetric after harmonization.
    harmonized_stack = 0.5 * (harmonized_stack + np.transpose(harmonized_stack, (0, 2, 1)))
    diag_idx = np.arange(n_nodes)
    harmonized_stack[:, diag_idx, diag_idx] = stack[:, diag_idx, diag_idx]

    summary = {
        "n_subjects": int(n_subjects),
        "n_parcels": int(n_nodes),
        "batch_col": resolved_batch,
        "batch_counts": {
            str(level): int(count)
            for level, count in zip(*np.unique(batch_values, return_counts=True))
        },
        "categorical_cols": list(resolved_cat),
        "continuous_cols": list(resolved_cont),
        "mean_original": float(np.nanmean(stack)),
        "mean_harmonized": float(np.nanmean(harmonized_stack)),
        "std_original": float(np.nanstd(stack)),
        "std_harmonized": float(np.nanstd(harmonized_stack)),
    }

    return {
        "harmonized_stack": harmonized_stack,
        "covars_model": covars_model,
        "batch_col": resolved_batch,
        "batch_values": batch_values,
        "categorical_cols": list(resolved_cat),
        "continuous_cols": list(resolved_cont),
        "summary": summary,
        "combat_result": combat_result,
    }


def build_default_output_path(
    source_path: Path | str,
    matrix_key: str,
    batch_col: str,
    categorical_cols: Optional[Sequence[str]] = None,
    continuous_cols: Optional[Sequence[str]] = None,
    output_dir: Optional[Path | str] = None,
) -> Path:
    src = Path(source_path)
    out_dir = Path(output_dir) if output_dir else src.parent
    out_dir = out_dir.expanduser()
    nuisances = list(categorical_cols or []) + list(continuous_cols or [])
    nuisance_tag = "none" if not nuisances else "-".join(_slugify(col) for col in nuisances)
    name = (
        f"{_slugify(src.stem)}_{_slugify(matrix_key)}_batch-{_slugify(batch_col)}"
        f"_cov-{nuisance_tag}_harmonized_connmat.npz"
    )
    return out_dir / name


def build_harmonized_payload(prepared: Dict[str, Any], harmonized_result: Dict[str, Any]) -> Dict[str, Any]:
    stack = np.asarray(harmonized_result["harmonized_stack"], dtype=float)
    pop_avg = np.asarray(stack.mean(axis=0), dtype=float)

    covars_df = harmonized_result["covars_model"].copy()
    covars_df.insert(0, "participant_id", np.asarray(prepared["subject_id_list"], dtype=object))
    covars_df.insert(1, "session_id", np.asarray(prepared["session_id_list"], dtype=object))

    parcel_labels = prepared.get("parcel_labels_group")
    if parcel_labels is None:
        parcel_labels = np.arange(1, stack.shape[1] + 1, dtype=int)
    parcel_names = prepared.get("parcel_names_group")
    if parcel_names is None:
        parcel_names = np.asarray(
            [f"parcel_{idx + 1}" for idx in range(stack.shape[1])],
            dtype=object,
        )

    payload: Dict[str, Any] = {
        "matrix_subj_list": stack,
        "matrix_pop_avg": pop_avg,
        "covars": covars_df.to_records(index=False),
        "subject_id_list": np.asarray(prepared["subject_id_list"], dtype=object),
        "session_id_list": np.asarray(prepared["session_id_list"], dtype=object),
        "parcel_labels_group": np.asarray(parcel_labels),
        "parcel_names_group": np.asarray(parcel_names),
        "group": np.asarray(str(prepared.get("group", ""))),
        "modality": np.asarray(str(prepared.get("modality", ""))),
        "harmonize_source_file": np.asarray(str(prepared.get("source_path", ""))),
        "harmonize_source_key": np.asarray(str(prepared.get("matrix_key", "matrix_subj_list"))),
        "harmonize_batch_col": np.asarray(str(harmonized_result.get("batch_col", ""))),
        "harmonize_categorical_cols": np.asarray(harmonized_result.get("categorical_cols", []), dtype=object),
        "harmonize_continuous_cols": np.asarray(harmonized_result.get("continuous_cols", []), dtype=object),
        "harmonize_selected_indices": np.asarray(prepared.get("selected_indices", []), dtype=int),
    }

    metabolites = prepared.get("metabolites")
    if metabolites is not None:
        payload["metabolites"] = np.asarray(metabolites)

    metab_profiles = prepared.get("metab_profiles_subj_list")
    if metab_profiles is not None:
        payload["metab_profiles_subj_list"] = np.asarray(metab_profiles)

    matrix_key = str(prepared.get("matrix_key", "matrix_subj_list"))
    if matrix_key and matrix_key != "matrix_subj_list":
        payload[matrix_key] = stack

    return payload


def save_harmonized_payload(output_path: Path | str, payload: Dict[str, Any]) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **payload)
    return out


def create_harmonization_plots(
    original_stack: np.ndarray,
    harmonized_stack: np.ndarray,
    batch_values: Sequence[Any],
    batch_col: str,
    figure=None,
    sample_index: int = 0,
):
    """Create a quick harmonization summary figure (histograms + matrix examples)."""

    original = np.asarray(original_stack, dtype=float)
    harmonized = np.asarray(harmonized_stack, dtype=float)
    if original.ndim != 3 or harmonized.ndim != 3:
        raise ValueError("original_stack and harmonized_stack must be 3D arrays.")
    if original.shape != harmonized.shape:
        raise ValueError("original_stack and harmonized_stack shape mismatch.")

    n_subjects = original.shape[0]
    if n_subjects == 0:
        raise ValueError("No subjects available for plotting.")

    index = max(0, min(int(sample_index), n_subjects - 1))
    batches = np.asarray([_display_text(v) for v in batch_values], dtype=object)
    if batches.shape[0] != n_subjects:
        batches = np.asarray(["all"] * n_subjects, dtype=object)

    if figure is None:
        fig = plt.figure(figsize=(15, 9))
    else:
        fig = figure
        fig.clear()

    axes = fig.subplots(2, 3)
    ax_hist_og, ax_hist_h, ax_scatter = axes[0]
    ax_m_og, ax_m_h, ax_m_diff = axes[1]

    unique_batches = list(dict.fromkeys(batches.tolist()))
    for batch in unique_batches:
        mask = batches == batch
        ax_hist_og.hist(
            original[mask].ravel(),
            bins=40,
            alpha=0.45,
            label=f"{batch}",
        )
        ax_hist_h.hist(
            harmonized[mask].ravel(),
            bins=40,
            alpha=0.45,
            label=f"{batch}",
        )

    ax_hist_og.set_title(f"Original by {batch_col}")
    ax_hist_h.set_title(f"Harmonized by {batch_col}")
    ax_hist_og.set_xlabel("Connectivity")
    ax_hist_h.set_xlabel("Connectivity")
    ax_hist_og.set_ylabel("Count")
    ax_hist_h.set_ylabel("Count")
    if len(unique_batches) <= 8:
        ax_hist_og.legend(fontsize=8)
        ax_hist_h.legend(fontsize=8)

    x = original[index].ravel()
    y = harmonized[index].ravel()
    ax_scatter.scatter(x, y, s=5, alpha=0.3, color="#111111")
    low = float(min(np.nanmin(x), np.nanmin(y)))
    high = float(max(np.nanmax(x), np.nanmax(y)))
    if np.isfinite(low) and np.isfinite(high) and low < high:
        ax_scatter.plot([low, high], [low, high], "r--", linewidth=1)
        ax_scatter.set_xlim(low, high)
        ax_scatter.set_ylim(low, high)
    ax_scatter.set_title(f"Sample {index}: Original vs Harmonized")
    ax_scatter.set_xlabel("Original")
    ax_scatter.set_ylabel("Harmonized")
    ax_scatter.grid(alpha=0.3, linestyle="--")

    matrix_og = original[index]
    matrix_h = harmonized[index]
    matrix_d = matrix_h - matrix_og
    vmin = float(min(np.nanmin(matrix_og), np.nanmin(matrix_h)))
    vmax = float(max(np.nanmax(matrix_og), np.nanmax(matrix_h)))
    max_abs_diff = float(np.nanmax(np.abs(matrix_d)))
    if not np.isfinite(max_abs_diff) or max_abs_diff <= 0:
        max_abs_diff = 1.0

    if SimMatrixPlot is not None:
        SimMatrixPlot.plot_simmatrix(
            matrix_og,
            ax=ax_m_og,
            titles="Original matrix",
            show_colorbar=True,
            colormap="plasma",
            vmin=vmin,
            vmax=vmax,
        )
        SimMatrixPlot.plot_simmatrix(
            matrix_h,
            ax=ax_m_h,
            titles="Harmonized matrix",
            show_colorbar=True,
            colormap="plasma",
            vmin=vmin,
            vmax=vmax,
        )
        SimMatrixPlot.plot_simmatrix(
            matrix_d,
            ax=ax_m_diff,
            titles="Difference (harm - orig)",
            show_colorbar=True,
            colormap="coolwarm",
            vmin=-max_abs_diff,
            vmax=max_abs_diff,
        )
    else:
        im0 = ax_m_og.imshow(matrix_og, cmap="plasma", vmin=vmin, vmax=vmax)
        im1 = ax_m_h.imshow(matrix_h, cmap="plasma", vmin=vmin, vmax=vmax)
        im2 = ax_m_diff.imshow(
            matrix_d,
            cmap="coolwarm",
            vmin=-max_abs_diff,
            vmax=max_abs_diff,
        )
        ax_m_og.set_title("Original matrix")
        ax_m_h.set_title("Harmonized matrix")
        ax_m_diff.set_title("Difference (harm - orig)")
        fig.colorbar(im0, ax=ax_m_og, fraction=0.046)
        fig.colorbar(im1, ax=ax_m_h, fraction=0.046)
        fig.colorbar(im2, ax=ax_m_diff, fraction=0.046)

    fig.tight_layout()
    return fig


def run_harmonization(
    source_path: Path | str,
    matrix_key: str,
    batch_col: str,
    categorical_cols: Optional[Sequence[str]] = None,
    continuous_cols: Optional[Sequence[str]] = None,
    selected_indices: Optional[Sequence[int]] = None,
    output_path: Optional[Path | str] = None,
    show_plots: bool = False,
) -> Dict[str, Any]:
    """High-level API used by the GUI and CLI."""

    prepared = load_matrix_stack_from_npz(
        source_path=source_path,
        matrix_key=matrix_key,
        selected_indices=selected_indices,
    )
    original_stack = np.asarray(prepared["matrix_stack"], dtype=float)

    result = harmonize_matrix_stack(
        matrix_stack=original_stack,
        covars_df=prepared["covars_df"],
        batch_col=batch_col,
        categorical_cols=categorical_cols,
        continuous_cols=continuous_cols,
    )

    if output_path is None:
        output_path = build_default_output_path(
            source_path=prepared["source_path"],
            matrix_key=prepared["matrix_key"],
            batch_col=result["batch_col"],
            categorical_cols=result["categorical_cols"],
            continuous_cols=result["continuous_cols"],
        )

    payload = build_harmonized_payload(prepared, result)
    output_saved = save_harmonized_payload(output_path, payload)

    fig = None
    if show_plots:
        fig = create_harmonization_plots(
            original_stack=original_stack,
            harmonized_stack=result["harmonized_stack"],
            batch_values=result["batch_values"],
            batch_col=result["batch_col"],
        )
        plt.show(block=False)

    return {
        "output_path": Path(output_saved),
        "prepared": prepared,
        "result": result,
        "payload": payload,
        "figure": fig,
    }


def _split_csv_arg(text: str) -> List[str]:
    values = [chunk.strip() for chunk in str(text or "").split(",")]
    return [val for val in values if val]


def _parse_indices_arg(text: str) -> List[int]:
    values = _split_csv_arg(text)
    if not values:
        return []
    return [int(val) for val in values]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Harmonize an NPZ connectivity matrix stack with neuroCombat.",
    )
    parser.add_argument("--input", required=True, help="Input connectivity NPZ path.")
    parser.add_argument("--matrix-key", required=True, help="Matrix stack key in NPZ.")
    parser.add_argument("--batch-col", required=True, help="Batch/scanner covariate column.")
    parser.add_argument(
        "--categorical",
        default="",
        help="Comma-separated categorical nuisance covariates.",
    )
    parser.add_argument(
        "--continuous",
        default="",
        help="Comma-separated continuous nuisance covariates.",
    )
    parser.add_argument(
        "--select-indices",
        default="",
        help="Optional comma-separated selected row indices.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional output NPZ path (default: source folder + generated name).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable result plotting window.",
    )
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    categorical = _split_csv_arg(args.categorical)
    continuous = _split_csv_arg(args.continuous)
    selected = _parse_indices_arg(args.select_indices)
    selected_indices = selected if selected else None
    output_path = Path(args.output).expanduser() if str(args.output).strip() else None

    run = run_harmonization(
        source_path=Path(args.input).expanduser(),
        matrix_key=str(args.matrix_key),
        batch_col=str(args.batch_col),
        categorical_cols=categorical,
        continuous_cols=continuous,
        selected_indices=selected_indices,
        output_path=output_path,
        show_plots=(not bool(args.no_show)),
    )

    print(f"[HARMONIZE] Saved NPZ: {run['output_path']}")
    summary = run["result"]["summary"]
    print(
        "[HARMONIZE] "
        f"N={summary['n_subjects']} | parcels={summary['n_parcels']} | "
        f"batch={summary['batch_col']} | batches={summary['batch_counts']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
