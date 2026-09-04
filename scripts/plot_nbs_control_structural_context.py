#!/usr/bin/env python3

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

VIEWER_ROOT = Path(__file__).resolve().parents[1]


def _configure_matplotlib_cache_dir() -> None:
    configured = str(os.getenv("MPLCONFIGDIR") or "").strip()
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.append(VIEWER_ROOT / ".matplotlib-cache")

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_test"
            probe.touch(exist_ok=True)
            probe.unlink()
        except Exception:
            continue
        os.environ["MPLCONFIGDIR"] = str(candidate)
        return


_configure_matplotlib_cache_dir()

import matplotlib
SHOW_FIGURES_REQUESTED = "--show_figures" in sys.argv
if not SHOW_FIGURES_REQUESTED:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from nilearn import plotting
from scipy import stats


########## LOCAL HELPERS ##########
RICHCLUB_DENSITIES = [0.01, 0.05, 0.10, 0.20]

def _get_degree_per_node(adjacency_matrix: np.ndarray) -> np.ndarray:
    adjacency = np.asarray(adjacency_matrix)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("Adjacency matrix must be square.")
    return np.asarray(np.count_nonzero(adjacency, axis=1), dtype=int)


def _load_netbasedanalysis() -> type:
    try:
        from mrsitoolbox.connectomics.network import NetBasedAnalysis as imported_nba  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Structural context analysis requires the current mrsitoolbox pip package "
            "with mrsitoolbox.connectomics.network.NetBasedAnalysis. "
            "Install or upgrade it with: pip install --upgrade mrsitoolbox"
        ) from exc
    required_methods = ("binarize", "compute_richclub_stats")
    missing_methods = [name for name in required_methods if not hasattr(imported_nba, name)]
    if missing_methods:
        raise ImportError(
            "Structural context analysis requires a newer mrsitoolbox pip package. "
            f"NetBasedAnalysis is missing: {', '.join(missing_methods)}. "
            "Install or upgrade it with: pip install --upgrade mrsitoolbox"
        )
    return imported_nba


NetBasedAnalysis = _load_netbasedanalysis()


@dataclass
class MatrixBundle:
    path: Path
    matrices: np.ndarray
    parcel_labels: np.ndarray
    parcel_names: np.ndarray
    subject_ids: np.ndarray
    session_ids: np.ndarray
    covars_df: pd.DataFrame
    metadata: dict[str, object]


########## PRINT HELPERS ##########
def print_section(name: str) -> None:
    print(f"\n########## {name.upper()} ##########", flush=True)


def info(message: str) -> None:
    print(message, flush=True)


def warn(message: str) -> None:
    print(f"Warning: {message}", flush=True)


########## GENERIC HELPERS ##########
def _scalar(value: object) -> object:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


def _as_str(value: object) -> str:
    if value is None:
        return ""
    return str(_scalar(value)).strip()


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(_scalar(value))
    except Exception:
        return default


def _normalize_token(value: object, prefix: str | None = None) -> str:
    token = _as_str(value)
    lowered = token.lower()
    if prefix and lowered.startswith(prefix.lower()):
        token = token[len(prefix):]
    return token.strip()


def _normalize_subject_session(subject_id: object, session_id: object) -> tuple[str, str]:
    return (
        _normalize_token(subject_id, prefix="sub-"),
        _normalize_token(session_id, prefix="ses-"),
    )


def _resolve_column(df: pd.DataFrame, *candidates: str) -> str | None:
    lower_map = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        hit = lower_map.get(candidate.lower())
        if hit is not None:
            return hit
    return None


def _covars_to_df(raw_covars: object) -> pd.DataFrame:
    if raw_covars is None:
        return pd.DataFrame()
    arr = np.asarray(raw_covars)
    if arr.size == 0:
        return pd.DataFrame()
    try:
        return pd.DataFrame.from_records(arr)
    except Exception:
        return pd.DataFrame(arr)


def _ensure_pair_columns(df: pd.DataFrame, subject_ids: np.ndarray, session_ids: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    subj_col = _resolve_column(df, "participant_id", "subject_id")
    sess_col = _resolve_column(df, "session_id", "session")
    if subj_col is None:
        df.insert(0, "participant_id", np.asarray(subject_ids, dtype=object))
        subj_col = "participant_id"
    if sess_col is None:
        insert_idx = 1 if subj_col == "participant_id" else 0
        df.insert(insert_idx, "session_id", np.asarray(session_ids, dtype=object))
        sess_col = "session_id"
    df["_pair_subject"] = [_normalize_token(val, prefix="sub-") for val in df[subj_col].astype(str)]
    df["_pair_session"] = [_normalize_token(val, prefix="ses-") for val in df[sess_col].astype(str)]
    return df


def _pair_keys(subject_ids: Iterable[object], session_ids: Iterable[object]) -> list[tuple[str, str]]:
    return [_normalize_subject_session(sid, ses) for sid, ses in zip(subject_ids, session_ids)]


def _safe_name_array(raw: object, fallback_labels: np.ndarray) -> np.ndarray:
    arr = np.asarray(raw) if raw is not None else np.array([], dtype=object)
    if arr.size == len(fallback_labels):
        return arr.astype(object)
    return np.asarray([str(label) for label in fallback_labels], dtype=object)


def _json_ready(value: object) -> object:
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    return value


########## INPUT LOADERS ##########
def _candidate_matrix_keys(data: np.lib.npyio.NpzFile) -> list[str]:
    keys = []
    for key in data.files:
        try:
            arr = np.asarray(data[key])
        except Exception:
            continue
        if arr.ndim == 3 and arr.shape[1] == arr.shape[2] and (str(key) == "matrix_subj_list" or arr.shape[0] == 1):
            keys.append(str(key))
        elif arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
            keys.append(str(key))
    keys.sort(key=lambda item: (0 if item == "matrix_subj_list" else 1, item.lower()))
    return keys


def load_matrix_bundle(npz_path: Path, matrix_key: str = "matrix_subj_list") -> MatrixBundle:
    data = np.load(npz_path, allow_pickle=True)
    resolved_key = str(matrix_key or "matrix_subj_list").strip() or "matrix_subj_list"
    if resolved_key not in data.files:
        candidate_keys = _candidate_matrix_keys(data)
        available_text = ", ".join(candidate_keys) if candidate_keys else ", ".join(sorted(data.files))
        raise KeyError(
            f"{npz_path.name} does not contain the requested matrix key '{resolved_key}'. "
            f"Available matrix-like keys: {available_text}"
        )

    raw_matrices = np.asarray(data[resolved_key], dtype=float)
    direct_matrix_mode = resolved_key != "matrix_subj_list"
    if raw_matrices.ndim == 2 and raw_matrices.shape[0] == raw_matrices.shape[1]:
        matrices = raw_matrices[None, :, :]
        direct_matrix_mode = True
    elif raw_matrices.ndim == 3 and raw_matrices.shape[1] == raw_matrices.shape[2]:
        matrices = raw_matrices
        if direct_matrix_mode:
            if raw_matrices.shape[0] != 1:
                raise ValueError(
                    f"Key '{resolved_key}' in {npz_path.name} contains {raw_matrices.shape[0]} matrices. "
                    "Custom structural keys must contain a single square matrix or use 'matrix_subj_list'."
                )
    else:
        raise ValueError(
            f"Key '{resolved_key}' in {npz_path.name} must contain a square matrix or a subject matrix stack; "
            f"got shape {tuple(raw_matrices.shape)}."
        )

    parcel_labels = np.asarray(data["parcel_labels_group"])
    parcel_names = _safe_name_array(
        data["parcel_names_group"] if "parcel_names_group" in data.files else None,
        parcel_labels,
    )
    if direct_matrix_mode:
        subject_ids = np.asarray([f"direct_matrix::{resolved_key}"], dtype=str)
        session_ids = np.asarray(["direct"], dtype=str)
        covars_df = pd.DataFrame(
            {
                "participant_id": subject_ids,
                "session_id": session_ids,
            }
        )
    else:
        subject_ids = np.asarray(data["subject_id_list"]).astype(str)
        session_ids = np.asarray(data["session_id_list"]).astype(str)
        if matrices.shape[0] != len(subject_ids) or matrices.shape[0] != len(session_ids):
            raise ValueError(
                f"Key '{resolved_key}' in {npz_path.name} has {matrices.shape[0]} matrices, but "
                f"subject/session arrays have lengths {len(subject_ids)} and {len(session_ids)}."
            )
        covars_df = _covars_to_df(data["covars"] if "covars" in data.files else None)
        covars_df = _ensure_pair_columns(covars_df, subject_ids, session_ids)
    metadata = {
        "group": _as_str(data["group"]) if "group" in data.files else "",
        "modality": _as_str(data["modality"]) if "modality" in data.files else "",
        "matrix_key": resolved_key,
        "matrix_mode": "direct" if direct_matrix_mode else "subject_stack",
    }
    return MatrixBundle(
        path=npz_path,
        matrices=matrices,
        parcel_labels=parcel_labels,
        parcel_names=parcel_names,
        subject_ids=subject_ids,
        session_ids=session_ids,
        covars_df=covars_df,
        metadata=metadata,
    )


########## NBS HELPERS ##########
def parse_nuisance_from_filename(path: Path) -> list[str]:
    match = re.search(r"_nuis-([^_]+)", path.name)
    if not match:
        return []
    return [token for token in match.group(1).split("-") if token]


def parse_t_threshold_from_filename(path: Path) -> str | None:
    match = re.search(r"_th-([^_]+)", path.name)
    if not match:
        return None
    token = match.group(1).strip()
    return token if token else None


def default_output_subdir(input_nbs: Path, base_name: str) -> Path:
    threshold = parse_t_threshold_from_filename(input_nbs)
    if threshold is None:
        return input_nbs.parent / base_name
    return input_nbs.parent / f"{base_name}_th-{threshold}"


def extract_selected_nbs_mask(nbs_data: np.lib.npyio.NpzFile, component_index: int | None) -> tuple[np.ndarray, dict[str, object]]:
    comp_masks = None
    if "comp_masks" in nbs_data.files:
        comp_masks = np.asarray(nbs_data["comp_masks"]).astype(bool)
        if comp_masks.ndim == 2:
            comp_masks = comp_masks[None, ...]

    sig_indices = np.asarray(nbs_data["sig_indices"] if "sig_indices" in nbs_data.files else [], dtype=int)
    comp_pvals = np.asarray(nbs_data["comp_pvals"] if "comp_pvals" in nbs_data.files else [], dtype=float)

    if component_index is not None:
        if comp_masks is None:
            raise ValueError("Requested a component index but `comp_masks` is missing in the NBS NPZ.")
        if component_index < 0 or component_index >= comp_masks.shape[0]:
            raise ValueError(
                f"Component index {component_index} is out of range for {comp_masks.shape[0]} component(s)."
            )
        return np.asarray(comp_masks[component_index], dtype=bool), {
            "component_mode": "single",
            "component_index": int(component_index),
            "component_pvalue": float(comp_pvals[component_index]) if component_index < len(comp_pvals) else np.nan,
        }

    if "sig_mask" in nbs_data.files:
        sig_mask = np.asarray(nbs_data["sig_mask"], dtype=bool)
        if np.any(sig_mask):
            return sig_mask, {
                "component_mode": "significant_union",
                "component_indices": sig_indices.astype(int).tolist(),
                "component_pvalues": comp_pvals[sig_indices].astype(float).tolist() if sig_indices.size else [],
            }

    if comp_masks is not None and comp_masks.size:
        return np.any(comp_masks, axis=0).astype(bool), {
            "component_mode": "all_union_fallback",
            "component_indices": list(range(comp_masks.shape[0])),
            "component_pvalues": comp_pvals.astype(float).tolist(),
        }

    if "comp_mask" in nbs_data.files:
        component_idx = nbs_data["component_idx"] if "component_idx" in nbs_data.files else -1
        return np.asarray(nbs_data["comp_mask"], dtype=bool), {
            "component_mode": "single_saved_mask",
            "component_index": _as_int(component_idx, default=-1),
        }

    raise ValueError("Could not extract an NBS mask from the provided NPZ.")


def build_nbs_edge_table(mask: np.ndarray, parcel_labels: np.ndarray, parcel_names: np.ndarray, t_matrix: np.ndarray | None) -> pd.DataFrame:
    rows, cols = np.where(np.triu(mask, k=1))
    records = []
    for row_idx, col_idx in zip(rows.tolist(), cols.tolist()):
        record = {
            "node_i": int(row_idx),
            "node_j": int(col_idx),
            "label_i": parcel_labels[row_idx],
            "label_j": parcel_labels[col_idx],
            "name_i": str(parcel_names[row_idx]),
            "name_j": str(parcel_names[col_idx]),
        }
        if t_matrix is not None and t_matrix.shape == mask.shape:
            record["t_value"] = float(t_matrix[row_idx, col_idx])
        records.append(record)
    return pd.DataFrame.from_records(records)


########## PATH RESOLUTION ##########
def infer_project_names(input_nbs: Path, nbs_data: np.lib.npyio.NpzFile) -> list[str]:
    candidates: list[str] = []
    try:
        candidates.append(input_nbs.parents[2].name)
    except Exception:
        pass

    stored_connectivity = _as_str(nbs_data["connectivity_path"]) if "connectivity_path" in nbs_data.files else ""
    if stored_connectivity:
        match = re.search(r"/([^/]+)/derivatives/", stored_connectivity)
        if match:
            candidates.append(match.group(1))
        stem = Path(stored_connectivity).name
        if "_atlas-" in stem:
            candidates.append(stem.split("_atlas-", 1)[0])
        elif "-atlas-" in stem:
            candidates.append(stem.split("-atlas-", 1)[0])

    group_meta = _as_str(nbs_data["group"]) if "group" in nbs_data.files else ""
    if group_meta:
        candidates.append(group_meta)

    seen = set()
    ordered = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            ordered.append(candidate)
            seen.add(candidate)
    return ordered


def discover_bids_root(project_names: list[str], bids_dir: str | None, connectivity_hint: str | None) -> Path | None:
    candidate_roots: list[Path] = []

    if bids_dir:
        root = Path(bids_dir).expanduser().resolve()
        if root.exists():
            if (root / "derivatives").exists():
                candidate_roots.append(root)
            else:
                for project_name in project_names:
                    child = root / project_name
                    if child.exists() and (child / "derivatives").exists():
                        candidate_roots.append(child)

    if connectivity_hint:
        hint_path = Path(connectivity_hint)
        parts = hint_path.parts
        if "derivatives" in parts:
            idx = parts.index("derivatives")
            root = Path(*parts[:idx])
            if root.exists():
                candidate_roots.append(root)

    for search_root in (Path("/media"), Path("/mnt")):
        if not search_root.exists():
            continue
        for project_name in project_names:
            for hit in search_root.glob(f"*/{project_name}"):
                if hit.exists() and (hit / "derivatives").exists():
                    candidate_roots.append(hit)

    seen = set()
    ordered = []
    for root in candidate_roots:
        root = root.resolve()
        if root not in seen:
            ordered.append(root)
            seen.add(root)
    return ordered[0] if ordered else None


def infer_atlas_tokens(input_nbs: Path, nbs_data: np.lib.npyio.NpzFile) -> list[str]:
    tokens = []
    atlas_from_dir = input_nbs.parent.name
    if atlas_from_dir:
        tokens.extend(
            [
                atlas_from_dir,
                atlas_from_dir.replace("-scale", "_scale"),
                atlas_from_dir.replace("_scale", "-scale"),
            ]
        )

    parc_scheme = _as_str(nbs_data["parc_scheme"]) if "parc_scheme" in nbs_data.files else ""
    scale = _as_int(nbs_data["scale"], default=0)
    if parc_scheme and scale:
        tokens.extend(
            [
                f"chimera{parc_scheme}_scale{scale}",
                f"chimera{parc_scheme}-scale{scale}",
                f"{parc_scheme}_scale{scale}",
                f"{parc_scheme}-scale{scale}",
            ]
        )

    seen = set()
    ordered = []
    for token in tokens:
        if token and token not in seen:
            ordered.append(token)
            seen.add(token)
    return ordered


def resolve_structural_path(
    input_nbs: Path,
    nbs_data: np.lib.npyio.NpzFile,
    input_struct: str | None,
    bids_root: Path | None,
) -> Path:
    if input_struct:
        path = Path(input_struct).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Structural NPZ not found: {path}")
        return path

    if bids_root is None:
        raise FileNotFoundError("Could not resolve a BIDS root automatically. Pass --bids_dir or --input_struct.")

    atlas_tokens = infer_atlas_tokens(input_nbs, nbs_data)
    project_names = infer_project_names(input_nbs, nbs_data)
    dwi_dirs = [
        bids_root / "derivatives" / "group" / "connectivity" / "dwi",
        bids_root / "derivatives" / "group" / "connectivity" / "multimodal",
    ]

    exact_candidates = []
    for project_name in project_names:
        for dwi_dir in dwi_dirs:
            for atlas_token in atlas_tokens:
                exact_candidates.append(dwi_dir / f"{project_name}_atlas-{atlas_token}_desc-group_connectivity_dwi.npz")
                exact_candidates.append(
                    dwi_dir / f"{project_name}_atlas-{atlas_token}_desc-group_connectivity_dwi_key-matrix_subj_list_reg-Diag_n-49_nbs_input.npz"
                )
                exact_candidates.append(
                    dwi_dir / f"{project_name}_atlas-{atlas_token}_desc-group_connectivity_dwi_key-matrix_subj_list_reg-Diag_n-47_nbs_input.npz"
                )

    for candidate in exact_candidates:
        if candidate.exists():
            return candidate

    parc_scheme = _as_str(nbs_data["parc_scheme"]) if "parc_scheme" in nbs_data.files else ""
    scale = _as_int(nbs_data["scale"], default=0)
    glob_hits = []
    for dwi_dir in dwi_dirs:
        if not dwi_dir.exists():
            continue
        for atlas_token in atlas_tokens:
            glob_hits.extend(sorted(dwi_dir.glob(f"*atlas*{atlas_token.replace('-', '*').replace('_', '*')}*dwi*.npz")))
        if parc_scheme and scale:
            glob_hits.extend(sorted(dwi_dir.glob(f"*{parc_scheme}*scale{scale}*dwi*.npz")))

    if glob_hits:
        glob_hits = sorted(
            glob_hits,
            key=lambda path: (
                "nbs_input" in path.name.lower(),
                "multimodal" in str(path.parent).lower(),
                len(path.name),
            ),
        )
        return glob_hits[0]

    raise FileNotFoundError("Could not locate a structural connectivity NPZ. Pass --input_struct explicitly.")


########## ALIGNMENT HELPERS ##########
def align_nbs_and_structural_parcels(
    structural_bundle: MatrixBundle,
    nbs_labels: np.ndarray,
    nbs_names: np.ndarray,
    nbs_mask: np.ndarray,
    t_matrix: np.ndarray | None,
    strict: bool,
) -> tuple[MatrixBundle, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, dict[str, object]]:
    label_to_index = {int(label): idx for idx, label in enumerate(structural_bundle.parcel_labels.tolist())}
    keep_nbs_indices = [idx for idx, label in enumerate(nbs_labels.tolist()) if int(label) in label_to_index]
    missing_labels = [int(label) for label in nbs_labels.tolist() if int(label) not in label_to_index]
    if missing_labels and strict:
        raise ValueError(
            f"Structural matrix is missing {len(missing_labels)} NBS parcel label(s). "
            f"First missing labels: {missing_labels[:10]}"
        )

    keep_nbs_indices_arr = np.asarray(keep_nbs_indices, dtype=int)
    keep_struct_indices_arr = np.asarray(
        [label_to_index[int(nbs_labels[idx])] for idx in keep_nbs_indices_arr.tolist()],
        dtype=int,
    )

    aligned_struct = MatrixBundle(
        path=structural_bundle.path,
        matrices=structural_bundle.matrices[:, keep_struct_indices_arr][:, :, keep_struct_indices_arr],
        parcel_labels=structural_bundle.parcel_labels[keep_struct_indices_arr],
        parcel_names=structural_bundle.parcel_names[keep_struct_indices_arr],
        subject_ids=structural_bundle.subject_ids,
        session_ids=structural_bundle.session_ids,
        covars_df=structural_bundle.covars_df,
        metadata=structural_bundle.metadata,
    )
    aligned_nbs_labels = nbs_labels[keep_nbs_indices_arr]
    aligned_nbs_names = nbs_names[keep_nbs_indices_arr]
    aligned_nbs_mask = nbs_mask[np.ix_(keep_nbs_indices_arr, keep_nbs_indices_arr)]
    aligned_t_matrix = (
        t_matrix[np.ix_(keep_nbs_indices_arr, keep_nbs_indices_arr)]
        if t_matrix is not None and t_matrix.shape == nbs_mask.shape
        else t_matrix
    )
    parcel_meta = {
        "n_nbs_nodes_original": int(len(nbs_labels)),
        "n_struct_nodes_original": int(len(structural_bundle.parcel_labels)),
        "n_nodes_intersection": int(len(aligned_nbs_labels)),
        "missing_nbs_labels_in_struct": missing_labels,
        "nbs_edges_original": int(np.triu(nbs_mask, k=1).sum()),
        "nbs_edges_after_intersection": int(np.triu(aligned_nbs_mask, k=1).sum()),
    }
    return aligned_struct, aligned_nbs_labels, aligned_nbs_names, aligned_nbs_mask, aligned_t_matrix, parcel_meta


def align_bundle_to_pairs(
    bundle: MatrixBundle,
    target_pairs: list[tuple[str, str]],
) -> tuple[MatrixBundle, list[int], list[int], list[tuple[str, str]]]:
    pair_to_index = {}
    for idx, pair in enumerate(_pair_keys(bundle.subject_ids, bundle.session_ids)):
        if pair not in pair_to_index:
            pair_to_index[pair] = idx

    indices = []
    target_indices = []
    missing = []
    for target_idx, pair in enumerate(target_pairs):
        if pair in pair_to_index:
            indices.append(pair_to_index[pair])
            target_indices.append(target_idx)
        else:
            missing.append(pair)

    if missing:
        missing_text = ", ".join(f"{sub}-{ses}" for sub, ses in missing[:20])
        suffix = "" if len(missing) <= 20 else ", ..."
        warn(
            f"{bundle.path.name} is missing {len(missing)} subject-session pair(s). "
            f"Skipping: {missing_text}{suffix}"
        )

    if not indices:
        raise ValueError(f"{bundle.path.name} has no overlap with the NBS subject-session pairs.")

    covars_df = bundle.covars_df.iloc[indices].reset_index(drop=True) if not bundle.covars_df.empty else pd.DataFrame()
    aligned = MatrixBundle(
        path=bundle.path,
        matrices=bundle.matrices[indices],
        parcel_labels=bundle.parcel_labels,
        parcel_names=bundle.parcel_names,
        subject_ids=bundle.subject_ids[indices],
        session_ids=bundle.session_ids[indices],
        covars_df=covars_df,
        metadata=bundle.metadata,
    )
    return aligned, indices, target_indices, missing


########## RICH-CLUB HELPERS ##########
def _symmetrize_matrix(matrix: np.ndarray) -> np.ndarray:
    out = np.asarray(matrix, dtype=float)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out = 0.5 * (out + out.T)
    np.fill_diagonal(out, 0.0)
    return out


########## STYLE ##########
FONTSIZE = 20
TICK_FONTSIZE = FONTSIZE - 2
LINEWIDTH = 2.5
SCATTER_POINT_SIZE = 55
VIOLIN_WIDTH = 0.75
BAR_WIDTH = 0.65
HUB_THRESHOLDS = [1, 5, 10, 20]
EDGE_CLASS_THRESHOLDS = [2, 5, 10, 20]
MAIN_HUB_THRESHOLD = 10
CONTROL_VALUE = 0
DEFAULT_RICHCLUB_DENSITY = float(RICHCLUB_DENSITIES[min(2, len(RICHCLUB_DENSITIES) - 1)]) if len(RICHCLUB_DENSITIES) else 0.10
RICHCLUB_OVERLAY_REFERENCE_DENSITIES = [0.02, 0.05, 0.20]
RICHCLUB_NUM_RANDOM = 100
RICHCLUB_ALPHA = 0.05
RICHCLUB_N_JOBS = 16
RICHCLUB_K_TICK_STEP = 10

BRAIN_EDGE_CMAP = "Reds"
BRAIN_NODE_CMAP = "viridis"
BRAIN_NODE_COLOR = "#d95f02"
SCATTER_COLOR = "#2c7fb8"
SCATTER_LINE_COLOR = "#253494"
NULL_BAND_COLOR = "#bdbdbd"
RICHCLUB_DENSITY_COLORS = {
    0.02: "#1b9e77",
    0.05: "#377eb8",
    0.10: "#d95f02",
    0.20: "#7570b3",
}
NBS_GROUP_COLORS = {
    "NBS nodes": "#c44e52",
    "Non-NBS nodes": "#4c72b0",
}
HUB_THRESHOLD_COLORS = {
    1: "#111111",
    5: "#1b9e77",
    10: "#d95f02",
    20: "#7570b3",
}
EDGE_CLASS_COLORS = {
    "hub-hub": "#c44e52",
    "feeder": "#dd8452",
    "local": "#55a868",
}

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    pass
plt.rcParams.update({
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.alpha": 0.35,
})


########## GENERIC HELPERS ##########
def _save_matplotlib_figure(fig: plt.Figure, output_stem: Path) -> dict[str, str]:
    output_paths = {}
    for suffix in (".png", ".pdf"):
        outpath = output_stem.with_suffix(suffix)
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        output_paths[suffix.lstrip(".")] = str(outpath)
    plt.close(fig)
    return output_paths


def _save_nilearn_display(save_callback, output_stem: Path) -> dict[str, str]:
    output_paths = {}
    for suffix in (".png", ".pdf"):
        outpath = output_stem.with_suffix(suffix)
        save_callback(str(outpath))
        output_paths[suffix.lstrip(".")] = str(outpath)
    return output_paths


def _log_figure_outputs(figure_outputs: dict[str, dict[str, str]]) -> None:
    for figure_name, output_map in figure_outputs.items():
        for fmt, path in output_map.items():
            info(f"Saved figure {figure_name} ({fmt}) to {path}")


def _infer_reference_matrix_label(struct_path: Path) -> str:
    match = re.search(r"(?:^|_)desc-([^_]+)", struct_path.name)
    if match:
        return str(match.group(1))
    return "Value"


def _compose_subplot_panel_figure(
    figure_outputs: dict[str, dict[str, str]],
    output_stem: Path,
    show: bool = False,
) -> dict[str, str]:
    panel_specs = [
        ("A", "metabolic_nbs_subnetwork", "Metabolic NBS subnetwork", (0, slice(0, 3))),
        ("B", "control_structural_strength", "Control structural node strength", (0, slice(3, 6))),
        ("", "reference_matrix", "Reference matrix", (1, slice(0, 2))),
        ("D", "strength_violin", "Structural strength in NBS vs non-NBS nodes", (1, slice(2, 4))),
        ("E", "hub_enrichment", "Hub enrichment across thresholds", (1, slice(4, 6))),
        ("F", "edge_class_enrichment", "Structural class of NBS edges", (2, slice(1, 5))),
    ]

    def _draw_panel_figure(fig: plt.Figure) -> None:
        grid = fig.add_gridspec(3, 6, hspace=0.18, wspace=0.18)
        for panel_letter, key, title, (row_idx, col_slice) in panel_specs:
            ax = fig.add_subplot(grid[row_idx, col_slice])
            png_path = figure_outputs.get(key, {}).get("png")
            if png_path is None or not Path(png_path).exists():
                ax.axis("off")
                ax.text(0.5, 0.5, f"Missing panel {panel_letter}", ha="center", va="center", fontsize=FONTSIZE)
                continue
            image = plt.imread(png_path)
            ax.imshow(image)
            title_text = f"{panel_letter}. {title}" if panel_letter else title
            ax.set_title(title_text, fontsize=FONTSIZE, pad=10)
            ax.axis("off")

    fig = plt.figure(figsize=(24, 16))
    _draw_panel_figure(fig)
    output_paths = _save_matplotlib_figure(fig, output_stem)
    if show:
        fig_show = plt.figure(figsize=(24, 16))
        _draw_panel_figure(fig_show)
        plt.show()
        plt.close(fig_show)
    return output_paths


def _scale_values(values: np.ndarray, min_size: float, max_size: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.array([], dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return np.full(values.shape, min_size, dtype=float)
    vmin = np.nanmin(values[finite])
    vmax = np.nanmax(values[finite])
    if np.isclose(vmin, vmax):
        return np.full(values.shape, 0.5 * (min_size + max_size), dtype=float)
    scaled = (values - vmin) / (vmax - vmin)
    return min_size + scaled * (max_size - min_size)


def _coerce_diag_series(df: pd.DataFrame) -> pd.Series:
    diag_col = _resolve_column(df, "Diag", "diag")
    if diag_col is None:
        raise ValueError("Could not find a diag column in the aligned covariates dataframe.")
    return pd.to_numeric(df[diag_col], errors="coerce")


def _compute_node_strength(matrix: np.ndarray) -> np.ndarray:
    matrix = _symmetrize_matrix(matrix)
    return np.asarray(matrix.sum(axis=1), dtype=float)


def _compute_nbs_node_load(nbs_mask: np.ndarray) -> np.ndarray:
    return np.asarray(np.asarray(nbs_mask, dtype=bool).sum(axis=1), dtype=int)


def _sparse_integer_ticks(values: np.ndarray, step: int) -> np.ndarray:
    values = np.asarray(values, dtype=int)
    if values.size == 0:
        return np.array([], dtype=int)
    if step <= 1 or values.size <= 10:
        return values
    tick_values = np.arange(int(values.min()), int(values.max()) + 1, int(step), dtype=int)
    tick_values = np.unique(np.concatenate([tick_values, [int(values.min()), int(values.max())]]))
    return tick_values


def _top_percent_hub_mask(strength: np.ndarray, percent: float) -> np.ndarray:
    strength = np.asarray(strength, dtype=float)
    n_nodes = strength.size
    n_hubs = max(1, int(np.ceil(n_nodes * (float(percent) / 100.0))))
    order = np.argsort(-strength, kind="mergesort")
    hub_mask = np.zeros(n_nodes, dtype=bool)
    hub_mask[order[:n_hubs]] = True
    return hub_mask


def _empirical_pvalue(observed: float, null_values: np.ndarray, tail: str = "greater") -> float:
    null_values = np.asarray(null_values, dtype=float)
    null_values = null_values[np.isfinite(null_values)]
    if null_values.size == 0:
        return np.nan
    if tail == "greater":
        count = np.sum(null_values >= observed)
    elif tail == "less":
        count = np.sum(null_values <= observed)
    else:
        count = np.sum(np.abs(null_values) >= abs(observed))
    return float((count + 1) / (null_values.size + 1))


def _significance_marker(p_value: float) -> str:
    if not np.isfinite(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def _select_control_average_structural(
    aligned_struct: MatrixBundle,
    covars_df: pd.DataFrame,
    control_value: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    diag_series = _coerce_diag_series(covars_df)
    control_mask = diag_series == int(control_value)
    if not np.any(control_mask):
        raise ValueError(f"No control rows found with diag == {control_value}.")
    control_covars = covars_df.loc[control_mask].reset_index(drop=True)
    control_matrices = np.asarray(aligned_struct.matrices[control_mask.to_numpy()], dtype=float)
    if control_matrices.size == 0:
        raise ValueError("No structural matrices remained after selecting controls.")
    average_matrix = _symmetrize_matrix(np.nanmean(control_matrices, axis=0))
    return average_matrix, control_covars


def _spearman_permutation_test(
    x: np.ndarray,
    y: np.ndarray,
    num_permutations: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    rho, p_standard = stats.spearmanr(x, y)
    null_rhos = np.empty(int(num_permutations), dtype=float)
    for idx in range(int(num_permutations)):
        null_rhos[idx] = stats.spearmanr(x, rng.permutation(y)).statistic
    p_empirical = _empirical_pvalue(float(rho), null_rhos, tail="two-sided")
    return float(rho), float(p_standard), float(p_empirical)


def _mannwhitney_with_effect(
    group_a: np.ndarray,
    group_b: np.ndarray,
) -> tuple[float, float, float]:
    group_a = np.asarray(group_a, dtype=float)
    group_b = np.asarray(group_b, dtype=float)
    group_a = group_a[np.isfinite(group_a)]
    group_b = group_b[np.isfinite(group_b)]
    test = stats.mannwhitneyu(group_a, group_b, alternative="two-sided")
    n1 = len(group_a)
    n2 = len(group_b)
    rank_biserial = (2.0 * test.statistic / (n1 * n2)) - 1.0 if n1 > 0 and n2 > 0 else np.nan
    return float(test.statistic), float(test.pvalue), float(rank_biserial)


def _compute_hub_enrichment(
    nbs_node_mask: np.ndarray,
    node_strength: np.ndarray,
    thresholds: list[int],
    num_permutations: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    n_nodes = len(node_strength)
    nbs_nodes = np.where(nbs_node_mask)[0]
    n_nbs_nodes = len(nbs_nodes)
    all_nodes = np.arange(n_nodes, dtype=int)
    records = []

    for threshold in thresholds:
        hub_mask = _top_percent_hub_mask(node_strength, threshold)
        hub_indices = np.where(hub_mask)[0]
        observed_overlap = int(np.sum(nbs_node_mask[hub_indices]))
        null_overlaps = np.empty(int(num_permutations), dtype=float)
        for idx in range(int(num_permutations)):
            sampled = rng.choice(all_nodes, size=n_nbs_nodes, replace=False)
            null_overlaps[idx] = int(np.sum(hub_mask[sampled]))
        mean_null = float(np.mean(null_overlaps))
        lower_null = float(np.percentile(null_overlaps, 2.5))
        upper_null = float(np.percentile(null_overlaps, 97.5))
        enrichment_ratio = float(observed_overlap / mean_null) if mean_null > 0 else np.nan
        p_empirical = _empirical_pvalue(float(observed_overlap), null_overlaps, tail="greater")
        records.append(
            {
                "hub_threshold_percent": int(threshold),
                "n_hubs": int(len(hub_indices)),
                "hub_node_indices": json.dumps(_json_ready(hub_indices.tolist())),
                "observed_overlap": int(observed_overlap),
                "observed_overlap_fraction": float(observed_overlap / n_nbs_nodes) if n_nbs_nodes else np.nan,
                "mean_null_overlap": mean_null,
                "lower_null_overlap": lower_null,
                "upper_null_overlap": upper_null,
                "enrichment_ratio": enrichment_ratio,
                "lower_null_ratio": float(lower_null / mean_null) if mean_null > 0 else np.nan,
                "upper_null_ratio": float(upper_null / mean_null) if mean_null > 0 else np.nan,
                "p_empirical_greater": p_empirical,
            }
        )
    return pd.DataFrame.from_records(records)


def _classify_edge_classes(
    edge_indices: np.ndarray,
    hub_mask: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, int]]:
    class_names = []
    counts = {"hub-hub": 0, "feeder": 0, "local": 0}
    for i_idx, j_idx in edge_indices.tolist():
        if hub_mask[i_idx] and hub_mask[j_idx]:
            class_name = "hub-hub"
        elif hub_mask[i_idx] or hub_mask[j_idx]:
            class_name = "feeder"
        else:
            class_name = "local"
        class_names.append({"node_i": int(i_idx), "node_j": int(j_idx), "edge_class": class_name})
        counts[class_name] += 1
    return pd.DataFrame.from_records(class_names), counts


def _compute_edge_class_enrichment(
    nbs_mask: np.ndarray,
    hub_mask: np.ndarray,
    num_permutations: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_nodes = hub_mask.size
    edge_rows, edge_cols = np.where(np.triu(np.asarray(nbs_mask, dtype=bool), k=1))
    nbs_edges = np.column_stack([edge_rows, edge_cols]).astype(int)
    if nbs_edges.size == 0:
        raise ValueError("NBS mask contains no edges, cannot compute structural edge classes.")

    edge_class_df, observed_counts = _classify_edge_classes(nbs_edges, hub_mask)

    all_edge_rows, all_edge_cols = np.triu_indices(n_nodes, k=1)
    all_edges = np.column_stack([all_edge_rows, all_edge_cols]).astype(int)
    class_codes = np.zeros(all_edges.shape[0], dtype=int)
    both_hubs = hub_mask[all_edges[:, 0]] & hub_mask[all_edges[:, 1]]
    one_hub = hub_mask[all_edges[:, 0]] ^ hub_mask[all_edges[:, 1]]
    class_codes[both_hubs] = 0
    class_codes[one_hub] = 1
    class_codes[~(both_hubs | one_hub)] = 2
    class_order = ["hub-hub", "feeder", "local"]

    null_counts = np.empty((int(num_permutations), 3), dtype=float)
    for idx in range(int(num_permutations)):
        sampled_idx = rng.choice(all_edges.shape[0], size=nbs_edges.shape[0], replace=False)
        counts = np.bincount(class_codes[sampled_idx], minlength=3)
        null_counts[idx, :] = counts.astype(float)

    records = []
    for class_idx, class_name in enumerate(class_order):
        observed = int(observed_counts[class_name])
        mean_null = float(np.mean(null_counts[:, class_idx]))
        lower_null = float(np.percentile(null_counts[:, class_idx], 2.5))
        upper_null = float(np.percentile(null_counts[:, class_idx], 97.5))
        enrichment_ratio = float(observed / mean_null) if mean_null > 0 else np.nan
        p_empirical = _empirical_pvalue(float(observed), null_counts[:, class_idx], tail="greater")
        records.append(
            {
                "edge_class": class_name,
                "observed_count": observed,
                "mean_null_count": mean_null,
                "lower_null_count": lower_null,
                "upper_null_count": upper_null,
                "enrichment_ratio": enrichment_ratio,
                "lower_null_ratio": float(lower_null / mean_null) if mean_null > 0 else np.nan,
                "upper_null_ratio": float(upper_null / mean_null) if mean_null > 0 else np.nan,
                "p_empirical_greater": p_empirical,
            }
        )

    enrichment_df = pd.DataFrame.from_records(records)
    return edge_class_df, enrichment_df


def _compute_richclub_stats_compat(
    nba,
    adj: np.ndarray,
    density: float,
    num_random: int,
    alpha: float,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    method = nba.compute_richclub_stats
    try:
        param_names = set(inspect.signature(method).parameters.keys())
    except Exception:
        param_names = set()

    kwargs = {}
    for name, value in (
        ("num_random", int(num_random)),
        ("alpha", float(alpha)),
        ("null_model", "random"),
        ("edge_density", float(density)),
        ("n_jobs", int(n_jobs)),
    ):
        if (not param_names) or (name in param_names):
            kwargs[name] = value

    result = method(adj, **kwargs)
    if not isinstance(result, tuple):
        raise TypeError(
            "NetBasedAnalysis.compute_richclub_stats returned an unsupported non-tuple result."
        )

    if len(result) == 3:
        degrees, rc_coefficients, rand_params = result
        if not isinstance(rand_params, dict):
            raise TypeError(
                "NetBasedAnalysis.compute_richclub_stats returned three values, but the third is not a dict."
            )
        return (
            np.asarray(degrees, dtype=int),
            np.asarray(rc_coefficients, dtype=float),
            {
                "null_dist": np.asarray(rand_params.get("null_dist", []), dtype=float),
                "median": np.asarray(rand_params.get("median", []), dtype=float),
                "lower": np.asarray(rand_params.get("lower", []), dtype=float),
                "upper": np.asarray(rand_params.get("upper", []), dtype=float),
                "pvalue": np.asarray(rand_params.get("pvalue", []), dtype=float),
            },
        )

    if len(result) == 5:
        degrees, rc_coefficients, mean_random_rc, std_random_rc, p_values = result
        degrees_arr = np.asarray(degrees, dtype=int)
        rc_arr = np.asarray(rc_coefficients, dtype=float)
        mean_arr = np.asarray(mean_random_rc, dtype=float)
        std_arr = np.asarray(std_random_rc, dtype=float)
        pval_arr = np.asarray(p_values, dtype=float)
        return (
            degrees_arr,
            rc_arr,
            {
                "null_dist": np.empty((len(degrees_arr), 0), dtype=float),
                "median": mean_arr,
                "lower": mean_arr - std_arr,
                "upper": mean_arr + std_arr,
                "pvalue": pval_arr,
            },
        )

    raise TypeError(
        "NetBasedAnalysis.compute_richclub_stats returned an unsupported number of values: "
        f"{len(result)}"
    )


def compute_control_richclub_curve(
    control_average_matrix: np.ndarray,
    density: float,
    num_random: int,
    alpha: float,
    n_jobs: int,
) -> pd.DataFrame:
    nba = NetBasedAnalysis()
    matrix = _symmetrize_matrix(control_average_matrix)
    density = float(density)
    weight_threshold = float(nba.threshold_density(matrix, density))
    adj = nba.binarize(
        matrix,
        threshold=density,
        mode="abs",
        threshold_mode="density",
        binarize=True,
    )
    adj = np.asarray(adj > 0, dtype=int)
    adj = np.maximum(adj, adj.T)
    np.fill_diagonal(adj, 0)

    degrees, rc_coefficients, rand_params = _compute_richclub_stats_compat(
        nba=nba,
        adj=adj,
        density=density,
        num_random=int(num_random),
        alpha=float(alpha),
        n_jobs=int(n_jobs),
    )
    node_degree = _get_degree_per_node(adj)
    median_rand_rc = np.asarray(rand_params.get("median", []), dtype=float)
    lower_rand_rc = np.asarray(rand_params.get("lower", []), dtype=float)
    upper_rand_rc = np.asarray(rand_params.get("upper", []), dtype=float)
    p_values = np.asarray(rand_params.get("pvalue", []), dtype=float)

    n_possible_edges = adj.shape[0] * max(adj.shape[0] - 1, 0) / 2.0
    actual_density = float(np.triu(adj, k=1).sum() / n_possible_edges) if n_possible_edges > 0 else np.nan
    n_nodes = int(adj.shape[0])
    records: list[dict[str, object]] = []
    for idx, k in enumerate(degrees.tolist()):
        hub_node_count = int(np.sum(node_degree >= int(k)))
        hub_node_fraction = float(hub_node_count / n_nodes) if n_nodes > 0 else np.nan
        p_value = float(p_values[idx]) if idx < len(p_values) and np.isfinite(p_values[idx]) else np.nan
        records.append(
            {
                "k": int(k),
                "richclub_coefficient": float(rc_coefficients[idx]) if idx < len(rc_coefficients) else np.nan,
                "median_random_rc": float(median_rand_rc[idx]) if idx < len(median_rand_rc) and np.isfinite(median_rand_rc[idx]) else np.nan,
                "lower_random_rc": float(lower_rand_rc[idx]) if idx < len(lower_rand_rc) and np.isfinite(lower_rand_rc[idx]) else np.nan,
                "upper_random_rc": float(upper_rand_rc[idx]) if idx < len(upper_rand_rc) and np.isfinite(upper_rand_rc[idx]) else np.nan,
                "p_value": p_value,
                "is_significant_k": int(np.isfinite(p_value) and p_value <= float(alpha)),
                "n_nodes": n_nodes,
                "hub_node_count": hub_node_count,
                "hub_node_fraction": hub_node_fraction,
                "hub_node_fraction_percent": float(100.0 * hub_node_fraction) if np.isfinite(hub_node_fraction) else np.nan,
                "requested_density": density,
                "requested_density_percent": float(100.0 * density),
                "actual_density": actual_density,
                "actual_density_percent": float(100.0 * actual_density) if np.isfinite(actual_density) else np.nan,
                "binarization_weight_threshold": weight_threshold,
            }
        )

    return pd.DataFrame.from_records(records)


def compute_control_richclub_curves(
    control_average_matrix: np.ndarray,
    densities: list[float],
    num_random: int,
    alpha: float,
    n_jobs: int,
) -> pd.DataFrame:
    density_values = sorted({float(val) for val in densities})
    density_frames = []
    for density in density_values:
        density_frames.append(
            compute_control_richclub_curve(
                control_average_matrix=control_average_matrix,
                density=density,
                num_random=int(num_random),
                alpha=float(alpha),
                n_jobs=int(n_jobs),
            )
        )
    if not density_frames:
        return pd.DataFrame()
    return pd.concat(density_frames, ignore_index=True)


def _format_richclub_curve_note(richclub_curve_df: pd.DataFrame) -> str:
    plot_df = richclub_curve_df.sort_values("k").reset_index(drop=True)
    if plot_df.empty:
        return "No rich-club data"
    density_pct = float(plot_df["requested_density_percent"].iloc[0])
    actual_pct = float(plot_df["actual_density_percent"].iloc[0])
    cutoff = float(plot_df["binarization_weight_threshold"].iloc[0])
    significant_df = plot_df.loc[plot_df["is_significant_k"].astype(int) > 0]
    if significant_df.empty:
        sig_text = "No RC-significant k"
    else:
        sig_text = f"RC-significant k={int(significant_df['k'].min())}-{int(significant_df['k'].max())}"
    return (
        f"density={density_pct:.1f}% (actual {actual_pct:.1f}%)\n"
        f"|w|>{cutoff:.3g}\n"
        f"{sig_text}"
    )


########## PLOTTING ##########
def plot_metabolic_nbs_subnetwork(
    nbs_mask: np.ndarray,
    t_matrix: np.ndarray | None,
    centroids_world: np.ndarray,
    node_load: np.ndarray,
    output_stem: Path,
) -> dict[str, str]:
    active_nodes = np.where(node_load > 0)[0]
    if active_nodes.size == 0:
        raise ValueError("No NBS nodes were found for the metabolic NBS subnetwork plot.")

    edge_weights = np.asarray(nbs_mask, dtype=float)
    if t_matrix is not None and t_matrix.shape == nbs_mask.shape:
        edge_weights = np.abs(np.asarray(t_matrix, dtype=float)) * np.asarray(nbs_mask, dtype=float)

    edge_weights = edge_weights[np.ix_(active_nodes, active_nodes)]
    coords = np.asarray(centroids_world, dtype=float)[active_nodes]
    node_sizes = _scale_values(node_load[active_nodes], min_size=35.0, max_size=160.0)

    def _save(outpath: str) -> None:
        display = plotting.plot_connectome(
            adjacency_matrix=edge_weights,
            node_coords=coords,
            node_color=BRAIN_NODE_COLOR,
            node_size=node_sizes,
            edge_cmap=BRAIN_EDGE_CMAP,
            edge_vmin=float(np.nanmin(edge_weights[edge_weights > 0])) if np.any(edge_weights > 0) else 0.0,
            edge_vmax=float(np.nanmax(edge_weights)) if np.any(edge_weights > 0) else 1.0,
            display_mode="lyrz",
            black_bg=False,
            annotate=False,
            title="Metabolic NBS subnetwork",
            colorbar=True,
            edge_kwargs={"linewidth": LINEWIDTH},
            output_file=outpath,
        )
        if display is not None:
            try:
                display.close()
            except Exception:
                pass

    return _save_nilearn_display(_save, output_stem)


def plot_structural_strength_map(
    node_strength: np.ndarray,
    centroids_world: np.ndarray,
    hub_mask: np.ndarray,
    output_stem: Path,
) -> dict[str, str]:
    coords = np.asarray(centroids_world, dtype=float)
    sizes = _scale_values(node_strength, min_size=30.0, max_size=130.0)
    sizes = np.where(hub_mask, sizes * 1.2, sizes)

    def _save(outpath: str) -> None:
        display = plotting.plot_markers(
            node_values=node_strength,
            node_coords=coords,
            node_size=sizes,
            node_cmap=BRAIN_NODE_CMAP,
            alpha=0.95,
            display_mode="lyrz",
            title="Control-average structural node strength",
            annotate=False,
            black_bg=False,
            colorbar=True,
            node_kwargs={"edgecolors": "black", "linewidths": 0.35},
            output_file=outpath,
        )
        if display is not None:
            try:
                display.close()
            except Exception:
                pass

    return _save_nilearn_display(_save, output_stem)


def plot_control_richclub_curve(
    richclub_curve_df: pd.DataFrame,
    output_stem: Path,
) -> dict[str, str]:
    plot_df = richclub_curve_df.sort_values(["requested_density", "k"]).reset_index(drop=True)
    density_values = sorted(np.unique(plot_df["requested_density"].to_numpy(dtype=float)).tolist())
    fig, axes = plt.subplots(2, 2, figsize=(18.0, 12.5), sharey=True)
    axes = np.asarray(axes).reshape(-1)
    legend_handles = [
        Patch(facecolor=NULL_BAND_COLOR, edgecolor=NULL_BAND_COLOR, alpha=0.25, label="Null 95% interval"),
        Line2D([0], [0], linestyle="--", color="black", linewidth=1.6, label="Null median"),
        Line2D([0], [0], linestyle="-", color="black", linewidth=LINEWIDTH, label="Control network"),
        Line2D([0], [0], marker="o", linestyle="None", color="black", markersize=8, label="RC-significant k"),
    ]

    for ax, density in zip(axes, density_values):
        density_df = plot_df.loc[np.isclose(plot_df["requested_density"].to_numpy(dtype=float), density)].sort_values("k").reset_index(drop=True)
        x = density_df["k"].to_numpy(dtype=float)
        y = density_df["richclub_coefficient"].to_numpy(dtype=float)
        null_low = density_df["lower_random_rc"].to_numpy(dtype=float)
        null_med = density_df["median_random_rc"].to_numpy(dtype=float)
        null_high = density_df["upper_random_rc"].to_numpy(dtype=float)
        significant_mask = density_df["is_significant_k"].to_numpy(dtype=int) > 0
        color = RICHCLUB_DENSITY_COLORS.get(round(float(density), 4), SCATTER_LINE_COLOR)

        ax.fill_between(x, null_low, null_high, color=color, alpha=0.25, zorder=1)
        ax.plot(x, null_med, linestyle="--", color=color, linewidth=1.4, alpha=0.7, zorder=2)
        ax.plot(x, y, color=color, linewidth=LINEWIDTH, alpha=0.95, zorder=3)
        if np.any(significant_mask):
            ax.scatter(x[significant_mask], y[significant_mask], color=color, s=85, zorder=4)

        tick_values = _sparse_integer_ticks(np.rint(x).astype(int), step=RICHCLUB_K_TICK_STEP)
        ax.set_xticks(tick_values)
        if x.size > 0:
            ax.set_xlim(float(np.nanmin(x)), float(np.nanmax(x)))
        ax.set_ylim(bottom=0)
        ax.set_title(f"{100.0 * float(density):.0f}% density", fontsize=FONTSIZE)
        ax.set_xlabel("k", fontsize=FONTSIZE)
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
        ax.text(
            0.02,
            0.98,
            _format_richclub_curve_note(density_df),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=TICK_FONTSIZE - 3,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#d9d9d9", "alpha": 0.9},
        )

    for idx, ax in enumerate(axes):
        if idx >= len(density_values):
            ax.axis("off")
            continue
        if idx % 2 == 0:
            ax.set_ylabel("Rich-club coefficient", fontsize=FONTSIZE)

    fig.suptitle("C. Control rich-club curves across densities", fontsize=FONTSIZE + 2, y=0.98)
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=4,
        fontsize=TICK_FONTSIZE,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return _save_matplotlib_figure(fig, output_stem)


def plot_reference_matrix(
    reference_matrix: np.ndarray,
    colorbar_label: str,
    output_stem: Path,
) -> dict[str, str]:
    matrix = _symmetrize_matrix(reference_matrix)
    fig, ax = plt.subplots(figsize=(6.0, 5.6))
    image = ax.imshow(
        matrix,
        cmap=BRAIN_NODE_CMAP,
        interpolation="nearest",
        origin="lower",
        vmin=0.0,
        vmax=float(np.nanmax(matrix)) if np.any(np.isfinite(matrix)) else 1.0,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Node index", fontsize=FONTSIZE)
    ax.set_ylabel("Node index", fontsize=FONTSIZE)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(str(colorbar_label), fontsize=FONTSIZE)
    colorbar.ax.tick_params(labelsize=TICK_FONTSIZE)
    fig.tight_layout()
    return _save_matplotlib_figure(fig, output_stem)


def plot_strength_violin(
    df_strength_groups: pd.DataFrame,
    p_value: float,
    rank_biserial: float,
    output_stem: Path,
) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(7.5, 6.5))

    preferred_order = ["Non-NBS nodes", "NBS nodes"]
    available_groups = list(pd.unique(df_strength_groups["group"]))
    group_order = [group for group in preferred_order if group in available_groups]
    group_order.extend(group for group in available_groups if group not in group_order)

    plot_groups: list[str] = []
    grouped_values: list[np.ndarray] = []
    for group in group_order:
        values = (
            df_strength_groups.loc[
                df_strength_groups["group"] == group,
                "control_structural_strength",
            ]
            .dropna()
            .to_numpy(dtype=float)
        )
        if values.size == 0:
            continue
        plot_groups.append(group)
        grouped_values.append(values)

    if not grouped_values:
        raise ValueError("No structural strength values were available to plot.")

    positions = np.arange(len(plot_groups), dtype=float)
    violin = ax.violinplot(
        grouped_values,
        positions=positions,
        widths=VIOLIN_WIDTH,
        showmeans=False,
        showextrema=False,
        showmedians=False,
    )
    for body, group in zip(violin["bodies"], plot_groups):
        body.set_facecolor(NBS_GROUP_COLORS.get(group, "#808080"))
        body.set_edgecolor("black")
        body.set_alpha(0.82)
        body.set_linewidth(1.1)

    ax.boxplot(
        grouped_values,
        positions=positions,
        widths=0.25,
        showfliers=False,
        patch_artist=True,
        manage_ticks=False,
        boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 1.2},
        whiskerprops={"color": "black", "linewidth": 1.2},
        capprops={"color": "black", "linewidth": 1.2},
        medianprops={"color": "black", "linewidth": 1.5},
    )

    rng = np.random.default_rng(42)
    for pos, values in zip(positions, grouped_values):
        jitter = rng.uniform(-0.08, 0.08, size=values.size)
        ax.scatter(
            np.full(values.size, pos) + jitter,
            values,
            color="black",
            alpha=0.55,
            s=16,
            linewidths=0,
            zorder=3,
        )

    ax.set_xlim(-0.5, positions[-1] + 0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(plot_groups)
    ax.set_xlabel("")
    ax.set_ylabel("Control-average structural node strength", fontsize=FONTSIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ymax = float(np.nanmax(df_strength_groups["control_structural_strength"].to_numpy(dtype=float)))
    ytext = ymax * 1.02 if ymax > 0 else ymax + 0.1
    ax.text(
        float(np.mean(positions)),
        ytext,
        f"Mann-Whitney P = {p_value:.3g}\nRank-biserial = {rank_biserial:.2f}",
        ha="center",
        va="bottom",
        fontsize=TICK_FONTSIZE,
    )
    fig.tight_layout()
    return _save_matplotlib_figure(fig, output_stem)


def plot_hub_enrichment(
    enrichment_df: pd.DataFrame,
    output_stem: Path,
) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    x = enrichment_df["hub_threshold_percent"].to_numpy(dtype=float)
    y = enrichment_df["enrichment_ratio"].to_numpy(dtype=float)
    low = enrichment_df["lower_null_ratio"].to_numpy(dtype=float)
    high = enrichment_df["upper_null_ratio"].to_numpy(dtype=float)
    ax.fill_between(x, low, high, color=NULL_BAND_COLOR, alpha=0.35, label="Null 95% interval")
    ax.plot(x, y, color="black", linewidth=LINEWIDTH, zorder=2)
    for _, row in enrichment_df.iterrows():
        threshold = int(row["hub_threshold_percent"])
        color = HUB_THRESHOLD_COLORS.get(threshold, "black")
        ax.scatter(
            row["hub_threshold_percent"],
            row["enrichment_ratio"],
            color=color,
            s=100,
            zorder=3,
        )
        marker = _significance_marker(float(row["p_empirical_greater"]))
        if marker:
            ax.text(
                row["hub_threshold_percent"],
                row["enrichment_ratio"] + 0.08,
                marker,
                ha="center",
                va="bottom",
                fontsize=FONTSIZE,
            )
    ax.axhline(1.0, linestyle="--", color="black", linewidth=1.4)
    ax.set_xlabel("Structural hub threshold (%)", fontsize=FONTSIZE)
    ax.set_ylabel("NBS-node enrichment ratio", fontsize=FONTSIZE)
    ax.set_xticks(x)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.legend(fontsize=TICK_FONTSIZE)
    fig.tight_layout()
    return _save_matplotlib_figure(fig, output_stem)


def plot_edge_class_enrichment(
    enrichment_df: pd.DataFrame,
    thresholds: list[int],
    output_stem: Path,
) -> dict[str, str]:
    class_order = ["hub-hub", "feeder", "local"]
    thresholds = [int(val) for val in thresholds]
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    threshold_order = [threshold for threshold in thresholds if threshold in enrichment_df["hub_threshold_percent"].unique()]
    plot_df = enrichment_df.copy()
    plot_df["edge_class"] = pd.Categorical(plot_df["edge_class"], categories=class_order, ordered=True)
    plot_df["hub_threshold_percent"] = pd.Categorical(
        plot_df["hub_threshold_percent"],
        categories=threshold_order,
        ordered=True,
    )
    plot_df = plot_df.sort_values(["edge_class", "hub_threshold_percent"]).reset_index(drop=True)
    max_y = float(
        np.nanmax(
            np.concatenate(
                [
                    plot_df["enrichment_ratio"].to_numpy(dtype=float),
                    plot_df["upper_null_ratio"].to_numpy(dtype=float),
                ]
            )
        )
    )
    threshold_palette = {threshold: HUB_THRESHOLD_COLORS.get(int(threshold), "black") for threshold in threshold_order}
    x_base = np.arange(len(class_order), dtype=float)
    n_thresholds = max(1, len(threshold_order))
    group_width = 0.8
    bar_width = group_width / n_thresholds

    for offset_idx, threshold in enumerate(threshold_order):
        subset = (
            plot_df[plot_df["hub_threshold_percent"] == int(threshold)]
            .set_index("edge_class")
            .reindex(class_order)
            .reset_index()
        )
        x_positions = x_base - 0.5 * group_width + (offset_idx + 0.5) * bar_width
        bars = ax.bar(
            x_positions,
            subset["enrichment_ratio"].to_numpy(dtype=float),
            width=bar_width * 0.92,
            color=threshold_palette[threshold],
            label=f"Top {int(threshold)}%",
            zorder=2,
        )
        for bar, (_, row) in zip(bars, subset.iterrows()):
            x_center = bar.get_x() + 0.5 * bar.get_width()
            ax.vlines(
                x_center,
                row["lower_null_ratio"],
                row["upper_null_ratio"],
                color="black",
                linewidth=1.4,
                alpha=0.85,
                zorder=3,
            )
            marker = _significance_marker(float(row["p_empirical_greater"]))
            if marker:
                ax.text(
                    x_center,
                    row["enrichment_ratio"] + 0.08,
                    marker,
                    ha="center",
                    va="bottom",
                    fontsize=FONTSIZE,
                )

    ax.axhline(1.0, linestyle="--", color="black", linewidth=1.4)
    ax.set_xlabel("")
    ax.set_ylabel("Edge-class enrichment ratio", fontsize=FONTSIZE)
    ax.set_ylim(0, max(1.15, max_y * 1.18))
    ax.set_xticks(x_base)
    ax.set_xticklabels(class_order)
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE, rotation=12)
    ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    ax.legend(title="Hub threshold", fontsize=TICK_FONTSIZE, title_fontsize=TICK_FONTSIZE)
    fig.tight_layout()
    return _save_matplotlib_figure(fig, output_stem)


########## ARGUMENTS ##########
def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot the structural context of an NBS metabolic subnetwork using the control-average structural connectivity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input_nbs", required=True, help="Path to the NBS components_all NPZ.")
    parser.add_argument("--input_struct", default=None, help="Optional explicit path to the structural group NPZ.")
    parser.add_argument("--input_struct_key", default="matrix_subj_list", help="Key inside the structural NPZ used to load the reference matrix.")
    parser.add_argument("--bids_dir", default=None, help="Optional BIDS root used to resolve the structural NPZ.")
    parser.add_argument("--output_dir", default=None, help="Optional output directory. Defaults next to the NBS file.")
    parser.add_argument("--component", type=int, default=None, help="Optional NBS component index. Default uses the union of significant components.")
    parser.add_argument("--strict_parcels", action="store_true", help="Require the structural NPZ to contain every NBS parcel label.")
    parser.add_argument("--control_value", type=int, default=CONTROL_VALUE, help="Value of diag used to define controls.")
    parser.add_argument("--hub_thresholds", type=int, nargs="+", default=HUB_THRESHOLDS, help="Hub thresholds in percent for enrichment analyses.")
    parser.add_argument("--edge_class_thresholds", type=int, nargs="+", default=EDGE_CLASS_THRESHOLDS, help="Structural hub thresholds in percent used in panel F for edge-class enrichment.")
    parser.add_argument("--main_hub_threshold", type=int, default=MAIN_HUB_THRESHOLD, help="Main structural hub threshold in percent used for hub highlighting on the structural strength map.")
    parser.add_argument("--num_permutations", type=int, default=5000, help="Number of random permutations used for node and edge null models.")
    parser.add_argument("--richclub_density", type=float, default=DEFAULT_RICHCLUB_DENSITY, help="Density used to binarize the control structural graph for panel C.")
    parser.add_argument("--richclub_num_random", type=int, default=RICHCLUB_NUM_RANDOM, help="Number of random graphs used per density by compute_richclub_stats in panel C.")
    parser.add_argument("--richclub_alpha", type=float, default=RICHCLUB_ALPHA, help="Alpha used for the panel C null interval and significant-k flagging.")
    parser.add_argument("--richclub_n_jobs", type=int, default=RICHCLUB_N_JOBS, help="Worker processes passed to compute_richclub_stats in panel C.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for permutation-based analyses.")
    parser.add_argument("--show_figures", action="store_true", help="Display the combined subplot figure with matplotlib.pyplot.show() after saving.")
    return parser


########## MAIN ##########
def main() -> None:
    ########## PARSE ARGUMENTS ##########
    args = build_argument_parser().parse_args()
    input_nbs = Path(args.input_nbs).expanduser().resolve()
    if not input_nbs.exists():
        raise FileNotFoundError(f"NBS input not found: {input_nbs}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_output_subdir(input_nbs, "nbs_control_structural_context")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))

    ########## LOAD NBS INPUT ##########
    print_section("LOAD NBS INPUT")
    nbs_data = np.load(input_nbs, allow_pickle=True)
    nbs_mask, nbs_selection_meta = extract_selected_nbs_mask(nbs_data, args.component)
    nbs_mask = np.asarray(nbs_mask, dtype=bool)
    np.fill_diagonal(nbs_mask, False)
    nbs_mask = np.triu(nbs_mask, k=1)
    nbs_mask = nbs_mask | nbs_mask.T

    parcel_labels = np.asarray(nbs_data["parcel_labels"])
    parcel_names = _safe_name_array(nbs_data["parcel_names"] if "parcel_names" in nbs_data.files else None, parcel_labels)
    centroids_world = np.asarray(nbs_data["centroids_world"], dtype=float)
    centroid_by_label = {
        int(label): centroids_world[idx]
        for idx, label in enumerate(parcel_labels.tolist())
    }
    subject_ids = np.asarray(nbs_data["subject_ids"]).astype(str)
    session_ids = np.asarray(nbs_data["session_ids"]).astype(str)
    covars_df = _covars_to_df(nbs_data["covars"] if "covars" in nbs_data.files else None)
    covars_df = _ensure_pair_columns(covars_df, subject_ids, session_ids)
    t_matrix = np.asarray(nbs_data["t_matrix"], dtype=float) if "t_matrix" in nbs_data.files else None
    target_pairs = _pair_keys(subject_ids, session_ids)

    nbs_summary = {
        "input_nbs": str(input_nbs),
        "group_from_metadata": _as_str(nbs_data["group"]) if "group" in nbs_data.files else "",
        "parc_scheme": _as_str(nbs_data["parc_scheme"]) if "parc_scheme" in nbs_data.files else "",
        "scale": _as_int(nbs_data["scale"], default=0),
        "regressor_name": _as_str(nbs_data["regressor_name"]) if "regressor_name" in nbs_data.files else "",
        "nuisance_terms": parse_nuisance_from_filename(input_nbs),
        "n_subject_session_pairs": len(target_pairs),
        "n_nodes": int(nbs_mask.shape[0]),
        "n_nbs_edges": int(np.triu(nbs_mask, k=1).sum()),
        "selection": nbs_selection_meta,
    }
    info(json.dumps(nbs_summary, indent=2))

    ########## RESOLVE STRUCTURAL INPUT ##########
    print_section("RESOLVE STRUCTURAL INPUT")
    stored_connectivity_path = _as_str(nbs_data["connectivity_path"]) if "connectivity_path" in nbs_data.files else ""
    bids_root = discover_bids_root(
        project_names=infer_project_names(input_nbs, nbs_data),
        bids_dir=args.bids_dir,
        connectivity_hint=stored_connectivity_path,
    )
    struct_path = resolve_structural_path(input_nbs, nbs_data, args.input_struct, bids_root)
    info(f"BIDS root: {bids_root if bids_root is not None else 'not found'}")
    info(f"Resolved structural path: {struct_path}")
    info(f"Structural matrix key: {args.input_struct_key}")

    ########## LOAD STRUCTURAL INPUT ##########
    print_section("LOAD STRUCTURAL INPUT")
    structural_bundle = load_matrix_bundle(struct_path, matrix_key=args.input_struct_key)
    info(
        f"Structural matrices: {structural_bundle.matrices.shape[0]} subjects, "
        f"{structural_bundle.matrices.shape[1]} x {structural_bundle.matrices.shape[2]} nodes"
    )
    structural_matrix_mode = str(structural_bundle.metadata.get("matrix_mode") or "subject_stack")
    direct_matrix_mode = structural_matrix_mode == "direct"

    ########## ALIGN STRUCTURAL INPUT TO NBS ##########
    print_section("ALIGN STRUCTURAL INPUT TO NBS")
    (
        structural_bundle,
        parcel_labels,
        parcel_names,
        nbs_mask,
        t_matrix,
        parcel_alignment_meta,
    ) = align_nbs_and_structural_parcels(
        structural_bundle=structural_bundle,
        nbs_labels=parcel_labels,
        nbs_names=parcel_names,
        nbs_mask=nbs_mask,
        t_matrix=t_matrix,
        strict=bool(args.strict_parcels),
    )
    centroids_world = np.asarray([centroid_by_label[int(label)] for label in parcel_labels.tolist()], dtype=float)
    info(json.dumps(parcel_alignment_meta, indent=2))

    if direct_matrix_mode:
        aligned_struct = structural_bundle
        struct_indices = [0]
        kept_target_indices = []
        missing_pairs = []
        control_average_matrix = _symmetrize_matrix(np.asarray(aligned_struct.matrices[0], dtype=float))
        control_covars = pd.DataFrame(
            {
                "participant_id": [f"direct_matrix::{args.input_struct_key}"],
                "session_id": ["direct"],
            }
        )
        n_controls_used = 0
        info("Using the selected structural matrix key directly; skipped subject matching and control averaging.")
    else:
        aligned_struct, struct_indices, kept_target_indices, missing_pairs = align_bundle_to_pairs(
            structural_bundle,
            target_pairs,
        )
        covars_df = covars_df.iloc[kept_target_indices].reset_index(drop=True)
        if missing_pairs:
            info(f"Matched {len(kept_target_indices)} of {len(target_pairs)} NBS subject-session pair(s).")

    ########## BUILD CONTROL-AVERAGE STRUCTURAL CONNECTOME ##########
    print_section("BUILD CONTROL-AVERAGE STRUCTURAL CONNECTOME")
    if direct_matrix_mode:
        info("Selected key provides the structural matrix directly.")
    else:
        control_average_matrix, control_covars = _select_control_average_structural(
            aligned_struct=aligned_struct,
            covars_df=covars_df,
            control_value=int(args.control_value),
        )
        n_controls_used = int(control_covars.shape[0])
    node_strength = _compute_node_strength(control_average_matrix)
    nbs_node_load = _compute_nbs_node_load(nbs_mask)
    nbs_node_mask = nbs_node_load > 0
    main_hub_mask = _top_percent_hub_mask(node_strength, float(args.main_hub_threshold))
    info(f"Controls used in structural average: {n_controls_used}")
    info(f"NBS nodes: {int(nbs_node_mask.sum())}")

    ########## COMPUTE CONTROL RICH-CLUB CURVE ##########
    print_section("COMPUTE CONTROL RICH-CLUB CURVE")
    richclub_overlay_densities = sorted(
        {
            float(args.richclub_density),
            *[float(val) for val in RICHCLUB_OVERLAY_REFERENCE_DENSITIES],
        }
    )
    control_richclub_curve_df = compute_control_richclub_curves(
        control_average_matrix=control_average_matrix,
        densities=richclub_overlay_densities,
        num_random=int(args.richclub_num_random),
        alpha=float(args.richclub_alpha),
        n_jobs=int(args.richclub_n_jobs),
    )

    ########## COMPUTE NODE-LEVEL METRICS ##########
    print_section("COMPUTE NODE-LEVEL METRICS")
    rho, p_standard, p_empirical = _spearman_permutation_test(
        x=node_strength,
        y=nbs_node_load,
        num_permutations=int(args.num_permutations),
        rng=rng,
    )
    strength_nbs = node_strength[nbs_node_mask]
    strength_non_nbs = node_strength[~nbs_node_mask]
    _, violin_p, rank_biserial = _mannwhitney_with_effect(strength_nbs, strength_non_nbs)
    hub_enrichment_df = _compute_hub_enrichment(
        nbs_node_mask=nbs_node_mask,
        node_strength=node_strength,
        thresholds=[int(val) for val in args.hub_thresholds],
        num_permutations=int(args.num_permutations),
        rng=rng,
    )
    edge_class_thresholds = [int(val) for val in args.edge_class_thresholds]
    edge_class_tables = []
    edge_class_enrichment_tables = []
    edge_class_columns: dict[int, pd.Series] = {}
    for threshold in edge_class_thresholds:
        threshold_hub_mask = _top_percent_hub_mask(node_strength, threshold)
        edge_class_df_thr, edge_class_enrichment_df_thr = _compute_edge_class_enrichment(
            nbs_mask=nbs_mask,
            hub_mask=threshold_hub_mask,
            num_permutations=int(args.num_permutations),
            rng=rng,
        )
        edge_class_df_thr["hub_threshold_percent"] = int(threshold)
        edge_class_enrichment_df_thr["hub_threshold_percent"] = int(threshold)
        edge_class_tables.append(edge_class_df_thr)
        edge_class_enrichment_tables.append(edge_class_enrichment_df_thr)
        edge_class_columns[int(threshold)] = edge_class_df_thr["edge_class"].reset_index(drop=True)
    edge_class_df = pd.concat(edge_class_tables, ignore_index=True)
    edge_class_enrichment_df = pd.concat(edge_class_enrichment_tables, ignore_index=True)

    ########## BUILD OUTPUT TABLES ##########
    print_section("BUILD OUTPUT TABLES")
    node_metrics_df = pd.DataFrame(
        {
            "node_index": np.arange(len(parcel_labels), dtype=int),
            "parcel_label": parcel_labels,
            "parcel_name": parcel_names.astype(str),
            "x": centroids_world[:, 0],
            "y": centroids_world[:, 1],
            "z": centroids_world[:, 2],
            "nbs_node_load": nbs_node_load.astype(int),
            "is_nbs_node": nbs_node_mask.astype(int),
            "control_structural_strength": node_strength.astype(float),
            f"is_top_{int(args.main_hub_threshold)}pct_hub": main_hub_mask.astype(int),
        }
    )
    for threshold in [int(val) for val in args.hub_thresholds]:
        node_metrics_df[f"is_top_{threshold}pct_hub"] = _top_percent_hub_mask(node_strength, threshold).astype(int)

    strength_group_df = pd.DataFrame(
        {
            "group": np.where(nbs_node_mask, "NBS nodes", "Non-NBS nodes"),
            "control_structural_strength": node_strength.astype(float),
        }
    )
    nbs_edge_table = build_nbs_edge_table(nbs_mask, parcel_labels, parcel_names, t_matrix)
    if not edge_class_df.empty:
        nbs_edge_table = nbs_edge_table.reset_index(drop=True)
        for threshold in edge_class_thresholds:
            nbs_edge_table[f"edge_class_top_{threshold}pct"] = edge_class_columns[int(threshold)]

    ########## GENERATE FIGURES ##########
    print_section("GENERATE FIGURES")
    figure_outputs = {}
    reference_matrix_label = _infer_reference_matrix_label(struct_path)
    figure_outputs["metabolic_nbs_subnetwork"] = plot_metabolic_nbs_subnetwork(
        nbs_mask=nbs_mask,
        t_matrix=t_matrix,
        centroids_world=centroids_world,
        node_load=nbs_node_load,
        output_stem=output_dir / "figure_1_metabolic_nbs_subnetwork",
    )
    figure_outputs["control_structural_strength"] = plot_structural_strength_map(
        node_strength=node_strength,
        centroids_world=centroids_world,
        hub_mask=main_hub_mask,
        output_stem=output_dir / "figure_2_control_structural_node_strength",
    )
    figure_outputs["control_richclub_curve"] = plot_control_richclub_curve(
        richclub_curve_df=control_richclub_curve_df,
        output_stem=output_dir / "richlcub_curve",
    )
    figure_outputs["reference_matrix"] = plot_reference_matrix(
        reference_matrix=control_average_matrix,
        colorbar_label=reference_matrix_label,
        output_stem=output_dir / "figure_3b_reference_structural_matrix",
    )
    figure_outputs["strength_violin"] = plot_strength_violin(
        df_strength_groups=strength_group_df,
        p_value=violin_p,
        rank_biserial=rank_biserial,
        output_stem=output_dir / "figure_4_structural_strength_nbs_vs_non_nbs",
    )
    figure_outputs["hub_enrichment"] = plot_hub_enrichment(
        enrichment_df=hub_enrichment_df,
        output_stem=output_dir / "figure_5_hub_enrichment_across_thresholds",
    )
    figure_outputs["edge_class_enrichment"] = plot_edge_class_enrichment(
        enrichment_df=edge_class_enrichment_df,
        thresholds=edge_class_thresholds,
        output_stem=output_dir / "figure_6_structural_class_of_nbs_edges",
    )
    figure_outputs["combined_subplot_panel"] = _compose_subplot_panel_figure(
        figure_outputs=figure_outputs,
        output_stem=output_dir / "figure_all_structural_context_panels",
        show=bool(args.show_figures),
    )
    _log_figure_outputs(figure_outputs)
    if bool(args.show_figures):
        info("Requested --show_figures: displayed the combined subplot figure with matplotlib.")

    ########## SAVE RESULTS ##########
    print_section("SAVE RESULTS")
    np.savez_compressed(
        output_dir / "control_average_structural_connectivity.npz",
        matrix_pop_avg=control_average_matrix.astype(np.float32),
        parcel_labels_group=parcel_labels,
        parcel_names_group=parcel_names,
        control_subject_id_list=control_covars[_resolve_column(control_covars, "participant_id", "subject_id")].astype(str).to_numpy(),
        control_session_id_list=control_covars[_resolve_column(control_covars, "session_id", "session")].astype(str).to_numpy(),
        structural_input_path=str(struct_path),
        structural_input_key=str(args.input_struct_key),
        nbs_input_path=str(input_nbs),
    )
    node_metrics_path = output_dir / "node_metrics.tsv"
    node_metrics_df.to_csv(node_metrics_path, sep="\t", index=False)
    hub_enrichment_path = output_dir / "hub_enrichment.tsv"
    hub_enrichment_df.to_csv(hub_enrichment_path, sep="\t", index=False)
    strength_group_path = output_dir / "nbs_vs_non_nbs_structural_strength.tsv"
    strength_group_df.to_csv(strength_group_path, sep="\t", index=False)
    nbs_edge_path = output_dir / "nbs_edges_with_structural_classes.tsv"
    nbs_edge_table.to_csv(nbs_edge_path, sep="\t", index=False)
    edge_class_path = output_dir / "edge_class_enrichment.tsv"
    edge_class_enrichment_df.to_csv(edge_class_path, sep="\t", index=False)
    richclub_curve_path = output_dir / "control_richclub_curve.tsv"
    control_richclub_curve_df.to_csv(richclub_curve_path, sep="\t", index=False)
    control_covars_path = output_dir / "control_covars_used.tsv"
    control_covars.to_csv(control_covars_path, sep="\t", index=False)
    info(f"Saved node metrics to {node_metrics_path}")
    info(f"Saved hub enrichment to {hub_enrichment_path}")
    info(f"Saved NBS edge classes to {nbs_edge_path}")
    info(f"Saved control rich-club curve to {richclub_curve_path}")

    stats_summary = {
        "spearman_rho": float(rho),
        "spearman_p": float(p_standard),
        "spearman_permutation_p": float(p_empirical),
        "mannwhitney_p": float(violin_p),
        "rank_biserial": float(rank_biserial),
        "main_hub_threshold_percent": int(args.main_hub_threshold),
        "edge_class_thresholds_percent": edge_class_thresholds,
        "richclub_density": float(args.richclub_density),
        "richclub_num_random": int(args.richclub_num_random),
        "richclub_alpha": float(args.richclub_alpha),
        "n_controls": int(n_controls_used),
        "n_nbs_nodes": int(nbs_node_mask.sum()),
        "n_non_nbs_nodes": int((~nbs_node_mask).sum()),
        "n_nbs_edges": int(np.triu(nbs_mask, k=1).sum()),
    }
    summary = {
        "nbs_summary": nbs_summary,
        "parcel_alignment": parcel_alignment_meta,
        "pair_alignment": {
            "mode": "direct_matrix_key" if direct_matrix_mode else "matched_subject_stack",
            "n_target_pairs": int(len(target_pairs)),
            "n_matched_pairs": int(len(kept_target_indices)),
            "n_missing_pairs": int(len(missing_pairs)),
            "missing_pairs": [f"{sub}-{ses}" for sub, ses in missing_pairs],
        },
        "structural_input_path": str(struct_path),
        "structural_input_key": str(args.input_struct_key),
        "output_dir": str(output_dir),
        "control_value": int(args.control_value),
        "hub_thresholds": [int(val) for val in args.hub_thresholds],
        "edge_class_thresholds": edge_class_thresholds,
        "main_hub_threshold": int(args.main_hub_threshold),
        "num_permutations": int(args.num_permutations),
        "seed": int(args.seed),
        "stats_summary": stats_summary,
        "tables": {
            "node_metrics_tsv": str(node_metrics_path),
            "hub_enrichment_tsv": str(hub_enrichment_path),
            "strength_groups_tsv": str(strength_group_path),
            "nbs_edges_with_structural_classes_tsv": str(nbs_edge_path),
            "edge_class_enrichment_tsv": str(edge_class_path),
            "control_richclub_curve_tsv": str(richclub_curve_path),
            "control_covars_used_tsv": str(control_covars_path),
        },
        "figures": figure_outputs,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(_json_ready(summary), indent=2), encoding="utf-8")
    info(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
