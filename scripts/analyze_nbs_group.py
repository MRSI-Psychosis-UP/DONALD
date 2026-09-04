#!/usr/bin/env python3

import argparse, json, os
from os.path import isfile, join
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import datasets
import re
from collections import OrderedDict
from glob import glob
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.patches as m_patches
import matplotlib.path as m_path

try:
    from mrsitoolbox.tools.debug import Debug
    from mrsitoolbox.tools.datautils import DataUtils
    from mrsitoolbox.connectomics.nettools import NetTools
    from mrsitoolbox.graphplot.circular import plot_connectivity_circle
    from mrsitoolbox.graphplot.colorbar import ColorBar
except ImportError as exc:
    raise ImportError(
        "analyze_nbs_group.py requires the current mrsitoolbox pip package. "
        "Install or upgrade it with: pip install --upgrade mrsitoolbox"
    ) from exc
import matplotlib.pyplot as plt
from rich.table import Table
from rich.console import Console
from scipy.interpolate import make_interp_spline
from scipy.stats import zscore
from matplotlib.lines import Line2D
from sklearn.mixture import GaussianMixture
import subprocess
import sys
from glob import glob


debug = Debug()
dutils = DataUtils()
nettools = NetTools()
color_loader = ColorBar()
console = Console()
FONTSIZE = 22
MAXLEN_NODE_NAME = 20
SIMILARITY_PATH_RADII = {"left": 10.6, "right": 10.2}


def _combine_signed(values):
    """Combine multiple values by keeping strongest magnitude if signs differ, else average."""
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return 0.0
    has_pos = np.any(vals > 0)
    has_neg = np.any(vals < 0)
    if has_pos and has_neg:
        idx = np.argmax(np.abs(vals))
        return float(vals[idx])
    return float(np.nanmean(vals))

def collapse_parcels(con_matrix, parcel_names, node_values, significant_indices):
    """Collapse parcels ending with _<int> into base parcels."""
    if con_matrix is None or parcel_names is None:
        return con_matrix, parcel_names, node_values, set(parcel_names[i] for i in significant_indices)

    groups = OrderedDict()
    for idx, name in enumerate(parcel_names):
        base = re.sub(r"_(\d+)$", "", name)
        groups.setdefault(base, []).append(idx)

    n_groups = len(groups)
    collapsed_matrix = np.zeros((n_groups, n_groups), dtype=float)
    collapsed_values = np.zeros(n_groups, dtype=float)
    collapsed_names = list(groups.keys())

    for i, (_, idx_i) in enumerate(groups.items()):
        collapsed_values[i] = np.nanmean(node_values[idx_i]) if node_values is not None else 0.0
        for j, (_, idx_j) in enumerate(groups.items()):
            block = con_matrix[np.ix_(idx_i, idx_j)]
            collapsed_matrix[i, j] = _combine_signed(block.ravel())

    sig_labels = set()
    for idx in significant_indices:
        base = re.sub(r"_(\d+)$", "", parcel_names[idx])
        sig_labels.add(base)

    return collapsed_matrix, collapsed_names, collapsed_values, sig_labels


def exclude_parcels_by_substring(
    con_matrix,
    parcel_names,
    node_values,
    node_values_ref,
    substrings,
    parcel_labels=None,
):
    """Remove matching parcels while preserving alignment across plot arrays."""
    normalized = tuple(
        token.strip().lower()
        for value in substrings
        for token in str(value).split(",")
        if token.strip()
    )
    names = [str(name) for name in parcel_names]
    if not normalized:
        return con_matrix, names, node_values, node_values_ref, parcel_labels, []

    keep_indices = [
        idx
        for idx, name in enumerate(names)
        if not any(token in name.lower() for token in normalized)
    ]
    excluded_names = [name for idx, name in enumerate(names) if idx not in keep_indices]
    if not keep_indices:
        raise ValueError(
            "--exclude-parcel-substring removed every parcel from the circular plot."
        )

    keep = np.asarray(keep_indices, dtype=int)
    filtered_matrix = np.asarray(con_matrix)[np.ix_(keep, keep)]
    filtered_values = np.asarray(node_values)[keep]
    filtered_values_ref = (
        np.asarray(node_values_ref)[keep] if node_values_ref is not None else None
    )
    filtered_labels = (
        np.asarray(parcel_labels)[keep] if parcel_labels is not None else None
    )
    filtered_names = [names[idx] for idx in keep_indices]
    return (
        filtered_matrix,
        filtered_names,
        filtered_values,
        filtered_values_ref,
        filtered_labels,
        excluded_names,
    )


def _object_items(value):
    """Normalize object-array/list payloads from saved NPZ dictionaries."""
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            value = value.item()
        else:
            return value.reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _object_dict(value):
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, dict):
        return dict(value)
    try:
        return dict(value)
    except (TypeError, ValueError):
        return {}


def _path_parcel_token(name, collapse_parcels=False):
    token = str(name).strip().lower()
    if collapse_parcels:
        token = re.sub(r"_(\d+)$", "", token)
    return token


def _path_channel_order(value):
    order = []
    for channel in str(value or "").strip().upper():
        if channel in {"R", "G", "B"} and channel not in order:
            order.append(channel)
    return "".join(order) if len(order) == 3 else ""


def load_highlight_path_sequence(
    paths_npz,
    *,
    path_group="any",
    path_index=None,
    random_seed=None,
    collapse_parcels=False,
):
    """Select complete three-endpoint CTX paths and induced GM-adjacency edges."""
    input_path = os.path.abspath(os.path.expanduser(str(paths_npz)))
    if not isfile(input_path):
        raise FileNotFoundError(f"Free-energy paths NPZ not found: {input_path}")

    with np.load(input_path, allow_pickle=True) as path_data:
        if "groups" not in path_data.files:
            raise ValueError(f"Free-energy paths NPZ has no 'groups' payload: {input_path}")
        source_parcel_names = (
            np.asarray(path_data["parcel_names"], dtype=object).reshape(-1)
            if "parcel_names" in path_data.files
            else np.asarray([], dtype=object)
        )
        adjacency_path = (
            str(np.asarray(path_data["adjacency_path"]).reshape(-1)[0]).strip()
            if "adjacency_path" in path_data.files
            else ""
        )
        candidates_by_group = {"lh": [], "rh": []}
        for raw_group in _object_items(path_data["groups"]):
            group_dict = _object_dict(raw_group)
            group_name = str(group_dict.get("group", "")).strip().lower()
            if group_name not in candidates_by_group:
                continue
            path_order = _path_channel_order(
                group_dict.get("path_order", group_dict.get("color_order", ""))
            )
            if not path_order:
                continue
            required_route_segments = [
                path_order[0] + path_order[1],
                path_order[1] + path_order[2],
            ]
            saved_segment_labels = [
                str(label).strip().upper()
                for label in _object_items(group_dict.get("ctx_segment_labels", []))
                if str(label).strip()
            ]
            records = _object_items(group_dict.get("ctx_paths", []))
            if not records and group_dict.get("ctx_optimal_path") is not None:
                records = [group_dict.get("ctx_optimal_path")]
            for record_index, raw_record in enumerate(records, start=1):
                record = _object_dict(raw_record)
                segment_labels = [
                    str(label).strip().upper()
                    for label in _object_items(record.get("segment_labels", []))
                    if str(label).strip()
                ]
                record_order = _path_channel_order(record.get("path_label", ""))
                if record_order and record_order != path_order:
                    continue
                effective_segment_labels = segment_labels or saved_segment_labels
                if not all(
                    label in effective_segment_labels for label in required_route_segments
                ):
                    continue
                node_names = [
                    str(name)
                    for name in _object_items(record.get("node_names", []))
                    if str(name).strip()
                ]
                if not node_names:
                    for raw_node in _object_items(record.get("nodes", [])):
                        try:
                            node_index = int(raw_node)
                        except (TypeError, ValueError):
                            continue
                        if 0 <= node_index < source_parcel_names.size:
                            node_names.append(str(source_parcel_names[node_index]))
                if node_names:
                    candidates_by_group[group_name].append(
                        {
                            "group": group_name or "unknown",
                            "record_index": record_index,
                            "node_names": node_names,
                            "path_order": path_order,
                            "segment_labels": effective_segment_labels,
                        }
                    )

    rng = np.random.default_rng(random_seed)

    def _select_candidate(candidates, selection_label):
        if not candidates:
            raise ValueError(
                f"No saved complete three-endpoint CTX paths matched group "
                f"'{selection_label}' in {input_path}."
            )
        if path_index is None:
            selected_position = int(rng.integers(0, len(candidates)))
        else:
            selected_position = int(path_index) - 1
            if selected_position < 0 or selected_position >= len(candidates):
                raise ValueError(
                    f"--highlight-path-index must be between 1 and {len(candidates)} "
                    f"for group '{selection_label}'."
                )
        selected = dict(candidates[selected_position])
        selected["candidate_index"] = selected_position + 1
        selected["candidate_count"] = len(candidates)
        return selected

    if path_group == "both":
        selections = [
            _select_candidate(candidates_by_group["lh"], "lh"),
            _select_candidate(candidates_by_group["rh"], "rh"),
        ]
    elif path_group in {"lh", "rh"}:
        selections = [_select_candidate(candidates_by_group[path_group], path_group)]
    else:
        all_candidates = candidates_by_group["lh"] + candidates_by_group["rh"]
        selections = [_select_candidate(all_candidates, "any")]

    tokens = {
        _path_parcel_token(name, collapse_parcels=collapse_parcels)
        for selected in selections
        for name in selected["node_names"]
    }
    tokens.discard("")
    for selected in selections:
        selected["unique_parcel_count"] = len(
            {
                _path_parcel_token(name, collapse_parcels=collapse_parcels)
                for name in selected["node_names"]
            }
        )
        selected["input_path"] = input_path

    if adjacency_path and not os.path.isabs(adjacency_path):
        adjacency_path = os.path.join(os.path.dirname(input_path), adjacency_path)
    adjacency_path = os.path.abspath(os.path.expanduser(adjacency_path))
    if not adjacency_path or not isfile(adjacency_path):
        raise FileNotFoundError(
            f"GM adjacency referenced by the free-energy paths NPZ was not found: {adjacency_path}"
        )
    with np.load(adjacency_path, allow_pickle=True) as adjacency_data:
        if "adjacency_mat" not in adjacency_data.files:
            raise ValueError(f"GM adjacency NPZ has no 'adjacency_mat': {adjacency_path}")
        adjacency_matrix = np.asarray(adjacency_data["adjacency_mat"], dtype=float)
        adjacency_names = (
            np.asarray(adjacency_data["parcel_names"], dtype=object).reshape(-1)
            if "parcel_names" in adjacency_data.files
            else source_parcel_names
        )
    if adjacency_matrix.shape != (adjacency_names.size, adjacency_names.size):
        raise ValueError(
            f"GM adjacency shape {adjacency_matrix.shape} does not match "
            f"{adjacency_names.size} parcel names."
        )

    selected_exact_tokens = {
        _path_parcel_token(name, collapse_parcels=False)
        for selected in selections
        for name in selected["node_names"]
    }
    selected_adjacency_indices = [
        idx
        for idx, name in enumerate(adjacency_names)
        if _path_parcel_token(name, collapse_parcels=False) in selected_exact_tokens
    ]
    edge_name_pairs = []
    if selected_adjacency_indices:
        selected_idx = np.asarray(selected_adjacency_indices, dtype=int)
        selected_matrix = adjacency_matrix[np.ix_(selected_idx, selected_idx)]
        edge_rows, edge_cols = np.where(np.triu(selected_matrix != 0, k=1))
        edge_name_pairs = [
            (
                str(adjacency_names[selected_idx[row]]),
                str(adjacency_names[selected_idx[col]]),
            )
            for row, col in zip(edge_rows.tolist(), edge_cols.tolist())
        ]
    if not edge_name_pairs:
        raise ValueError(
            "The selected saved path parcels have no matching edges in the referenced GM adjacency."
        )
    return tokens, selections, edge_name_pairs, adjacency_path


def build_path_gm_adjacency(parcel_names, edge_name_pairs, *, collapse_parcels=False):
    """Build a binary GM-adjacency matrix aligned to the plotted parcel order."""
    name_to_index = {
        _path_parcel_token(name, collapse_parcels=collapse_parcels): idx
        for idx, name in enumerate(parcel_names)
    }
    matrix = np.zeros((len(parcel_names), len(parcel_names)), dtype=float)
    matched_edges = 0
    for source_name, target_name in edge_name_pairs:
        source_idx = name_to_index.get(
            _path_parcel_token(source_name, collapse_parcels=collapse_parcels)
        )
        target_idx = name_to_index.get(
            _path_parcel_token(target_name, collapse_parcels=collapse_parcels)
        )
        if source_idx is None or target_idx is None or source_idx == target_idx:
            continue
        if matrix[source_idx, target_idx] == 0:
            matched_edges += 1
        matrix[source_idx, target_idx] = 1.0
        matrix[target_idx, source_idx] = 1.0
    return matrix, matched_edges


def _draw_similarity_edge(ax, theta_a, theta_b, radius=8.5, color="white", linewidth=3.5):
    """Draw a bezier edge between two angles on the circle."""
    control_radius = max(3.0, radius - 3.0)
    verts = [(theta_a, radius), (theta_a, control_radius), (theta_b, control_radius), (theta_b, radius)]
    codes = [
        m_path.Path.MOVETO,
        m_path.Path.CURVE4,
        m_path.Path.CURVE4,
        m_path.Path.LINETO,
    ]
    patch = m_patches.PathPatch(
        m_path.Path(verts, codes),
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
        alpha=1.0,
        zorder=0.1,
        capstyle="round",
    )
    ax.add_patch(patch)


def _resolve_similarity_label(raw_name, name_lookup, parcel_names):
    token = str(raw_name).strip()
    lowered = token.lower()
    candidates = [lowered, re.sub(r"_(\d+)$", "", lowered)]
    for cand in candidates:
        if cand in name_lookup:
            return name_lookup[cand]
    if lowered.isdigit():
        idx = int(lowered)
        if 0 <= idx < len(parcel_names):
            base = re.sub(r"_(\d+)$", "", parcel_names[idx]).lower()
            return name_lookup.get(base)
    return None


def _overlay_similarity_path(
    ax,
    hub_sequence,
    name_lookup,
    angle_lookup,
    parcel_names,
    path_label,
    radius,
    draw=True,
):
    """Overlay ordered similarity hubs as white edges following the provided sequence."""
    if not hub_sequence:
        return []
    resolved_angles = []
    missing = []
    resolved_names = []
    for raw_name in hub_sequence:
        full_name = _resolve_similarity_label(raw_name, name_lookup, parcel_names)
        if full_name is None:
            missing.append(raw_name)
            continue
        resolved_angles.append(angle_lookup[full_name])
        resolved_names.append(full_name)

    if missing:
        debug.warning(
            f"Similarity hubs ({path_label}) missing from collapsed nodes: {', '.join(missing)}"
        )
    if len(resolved_angles) < 2:
        debug.warning(f"Not enough similarity hubs resolved for {path_label} to draw path.")
        return resolved_names

    if draw:
        for theta_a, theta_b in zip(resolved_angles[:-1], resolved_angles[1:]):
            _draw_similarity_edge(ax, theta_a, theta_b, radius=radius)
    return resolved_names


def compute_cluster_metab_deltas(
    cmp_values: np.ndarray,
    ref_values: np.ndarray,
    significant_indices: np.ndarray,
    group_splits: dict,
    compare_group: float,
    ref_group: float,
    n_clusters: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Cluster compare-group Gradient values and compute metabolic profile deltas per node."""
    cmp_values = np.asarray(cmp_values, dtype=float)
    ref_values = np.asarray(ref_values, dtype=float)
    if cmp_values.size == 0 or significant_indices.size == 0:
        return None, None
    n_clusters = max(1, min(int(n_clusters), cmp_values.size))
    gm = GaussianMixture(n_components=n_clusters, random_state=42)
    cluster_labels = gm.fit_predict(cmp_values.reshape(-1, 1))

    cmp_metab = np.asarray(group_splits[compare_group]["metab_profiles"])
    ref_metab = np.asarray(group_splits[ref_group]["metab_profiles"])
    if cmp_metab.size == 0 or ref_metab.size == 0:
        return None, None

    if cmp_metab.ndim < 3:
        return None, None
    n_metabolites = cmp_metab.shape[-1]
    deltas = np.zeros((cmp_values.size, n_metabolites), dtype=float)
    for cluster_id in range(n_clusters):
        mask = cluster_labels == cluster_id
        if not np.any(mask):
            continue
        node_idx = significant_indices[mask]
        cmp_subset = cmp_metab[:, node_idx, ...]
        ref_subset = ref_metab[:, node_idx, ...]
        reduce_axes = tuple(range(cmp_subset.ndim - 1))
        cmp_profile = np.nanmean(cmp_subset, axis=reduce_axes)
        ref_profile = np.nanmean(ref_subset, axis=reduce_axes)
        deltas[mask] = ref_profile - cmp_profile
    return deltas, cluster_labels


def _load_gradient_rgb_payload(path_value):
    if path_value is None or str(path_value).strip() == "":
        return None
    path = os.path.expanduser(str(path_value))
    if not os.path.isfile(path):
        debug.warning(f"Gradient RGB payload not found: {path}")
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        debug.warning(f"Failed to read Gradient RGB payload {path}: {exc}")
        return None
    if not isinstance(payload, dict) or not bool(payload.get("enabled", False)):
        return None
    model = payload.get("model")
    if not isinstance(model, dict):
        debug.warning("Gradient RGB payload is missing its fitted model.")
        return None
    if "vertices" not in model or "anchor_points" not in model:
        debug.warning("Gradient RGB payload model is missing vertices/anchor points.")
        return None
    payload["payload_path"] = path
    return payload


def _lookup_token(value):
    token = str(value).strip().lower()
    if not token:
        return ""
    token = re.sub(r"\s+", " ", token)
    return token


def _add_lookup_token(lookup, value, index):
    token = _lookup_token(value)
    if token:
        lookup.setdefault(token, int(index))
        lookup.setdefault(re.sub(r"_(\d+)$", "", token), int(index))
    try:
        numeric = int(float(str(value).strip()))
        lookup.setdefault(str(numeric), int(index))
    except Exception:
        pass


def _payload_reference_coords(payload, parcel_labels, parcel_names):
    x_values = np.asarray(payload.get("rgb_x_values", []), dtype=float).reshape(-1)
    y_values = np.asarray(payload.get("rgb_y_values", []), dtype=float).reshape(-1)
    if x_values.size == 0 or x_values.shape != y_values.shape:
        return None, 0

    lookup = {}
    point_ids = list(payload.get("point_ids", []))
    point_labels = list(payload.get("point_labels", []))
    for idx in range(x_values.size):
        if idx < len(point_ids):
            _add_lookup_token(lookup, point_ids[idx], idx)
        if idx < len(point_labels):
            _add_lookup_token(lookup, point_labels[idx], idx)

    parcel_labels = np.asarray(parcel_labels).reshape(-1)
    parcel_names = np.asarray(parcel_names, dtype=object).reshape(-1)
    coords = np.full((parcel_labels.size, 2), np.nan, dtype=float)
    matched = 0
    for out_idx, label in enumerate(parcel_labels.tolist()):
        candidates = [label]
        if out_idx < parcel_names.size:
            candidates.append(parcel_names[out_idx])
        source_idx = None
        for candidate in candidates:
            token = _lookup_token(candidate)
            source_idx = lookup.get(token)
            if source_idx is None:
                source_idx = lookup.get(re.sub(r"_(\d+)$", "", token))
            if source_idx is not None:
                break
        if source_idx is None:
            continue
        coords[out_idx, 0] = x_values[int(source_idx)]
        coords[out_idx, 1] = y_values[int(source_idx)]
        matched += 1
    return coords, matched


def _align_coords_to_reference(coords, reference_coords):
    coords = np.asarray(coords, dtype=float)
    reference_coords = np.asarray(reference_coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2 or reference_coords.shape != coords.shape:
        return coords, 0
    finite_mask = np.all(np.isfinite(coords), axis=1) & np.all(np.isfinite(reference_coords), axis=1)
    if np.count_nonzero(finite_mask) < 3:
        return coords, int(np.count_nonzero(finite_mask))

    source = coords[finite_mask]
    target = reference_coords[finite_mask]
    source_mean = np.nanmean(source, axis=0, keepdims=True)
    target_mean = np.nanmean(target, axis=0, keepdims=True)
    source_centered = source - source_mean
    target_centered = target - target_mean
    denom = float(np.sum(np.square(source_centered)))
    if not np.isfinite(denom) or denom <= 1e-12:
        return coords, int(np.count_nonzero(finite_mask))
    try:
        u_mat, singular_values, vt_mat = np.linalg.svd(source_centered.T @ target_centered)
    except Exception:
        return coords, int(np.count_nonzero(finite_mask))
    rotation = u_mat @ vt_mat
    scale = float(np.sum(singular_values) / denom)
    aligned = (coords - source_mean) @ rotation * scale + target_mean
    return np.asarray(aligned, dtype=float), int(np.count_nonzero(finite_mask))


def _normalized_rgb_order(model, fallback="RBG"):
    raw = str(dict(model or {}).get("order", fallback) or fallback).strip().upper()
    order = []
    for channel in raw:
        if channel in {"R", "G", "B"} and channel not in order:
            order.append(channel)
    for channel in str(fallback or "RBG").strip().upper():
        if channel in {"R", "G", "B"} and channel not in order:
            order.append(channel)
    for channel in ("R", "G", "B"):
        if channel not in order:
            order.append(channel)
    return order[:3]


def _triangle_rgb_weights_from_model(x_values, y_values, model):
    x_valid = np.asarray(x_values, dtype=float).reshape(-1)
    y_valid = np.asarray(y_values, dtype=float).reshape(-1)
    weights_full = np.zeros((x_valid.shape[0], 3), dtype=float)
    finite_mask = np.isfinite(x_valid) & np.isfinite(y_valid)
    if not np.any(finite_mask):
        return weights_full
    vertices = np.asarray(model.get("vertices"), dtype=float)
    if vertices.shape != (3, 2):
        return weights_full
    v0, v1, v2 = vertices
    denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
    if np.isclose(denom, 0.0):
        return weights_full
    points = np.column_stack((x_valid[finite_mask], y_valid[finite_mask]))
    w0 = (
        (v1[1] - v2[1]) * (points[:, 0] - v2[0])
        + (v2[0] - v1[0]) * (points[:, 1] - v2[1])
    ) / denom
    w1 = (
        (v2[1] - v0[1]) * (points[:, 0] - v2[0])
        + (v0[0] - v2[0]) * (points[:, 1] - v2[1])
    ) / denom
    w2 = 1.0 - w0 - w1
    weights = np.column_stack((w0, w1, w2))
    weights = np.clip(weights, 0.0, 1.0)
    weight_sum = weights.sum(axis=1, keepdims=True)
    weight_sum[weight_sum <= 0.0] = 1.0
    weights_full[finite_mask] = weights / weight_sum
    return weights_full


def _square_rgb_weights_from_model(x_values, y_values, model):
    x_valid = np.asarray(x_values, dtype=float).reshape(-1)
    y_valid = np.asarray(y_values, dtype=float).reshape(-1)
    weights_full = np.zeros((x_valid.shape[0], 3), dtype=float)
    finite_mask = np.isfinite(x_valid) & np.isfinite(y_valid)
    if not np.any(finite_mask):
        return weights_full
    anchors = np.asarray(model.get("anchor_points"), dtype=float)
    if anchors.shape != (3, 2):
        return weights_full
    points = np.column_stack((x_valid[finite_mask], y_valid[finite_mask]))
    distances = np.sqrt(np.sum(np.square(points[:, np.newaxis, :] - anchors[np.newaxis, :, :]), axis=2))
    weights = 1.0 / np.maximum(distances, 1e-9)
    close_mask = distances <= 1e-9
    if np.any(close_mask):
        for row_idx in np.flatnonzero(np.any(close_mask, axis=1)).tolist():
            weights[row_idx, :] = close_mask[row_idx, :].astype(float)
    weight_sum = weights.sum(axis=1, keepdims=True)
    weight_sum[weight_sum <= 0.0] = 1.0
    weights_full[finite_mask] = weights / weight_sum
    return weights_full


def _rgb_scalar_from_model(x_values, y_values, model, fallback_order="RBG"):
    fit_mode = str(dict(model or {}).get("fit_mode", "triangle") or "triangle").strip().lower()
    if fit_mode == "square":
        weights = _square_rgb_weights_from_model(x_values, y_values, model)
    else:
        weights = _triangle_rgb_weights_from_model(x_values, y_values, model)
    scalar_map = {"R": 0.0, "B": 0.5, "G": 1.0}
    order = _normalized_rgb_order(model, fallback=fallback_order)
    vertex_scalars = np.asarray(
        [scalar_map.get(channel, float(idx)) for idx, channel in enumerate(order)],
        dtype=float,
    )
    scalar_values = np.full(weights.shape[0], np.nan, dtype=float)
    valid_mask = np.sum(weights, axis=1) > 0.0
    if np.any(valid_mask):
        scalar_values[valid_mask] = weights[valid_mask] @ vertex_scalars
    return scalar_values


def _gradient_pair_from_matrix(matrix):
    matrix_arr = np.asarray(matrix, dtype=float)
    gradient1 = nettools.dimreduce_matrix(
        matrix_arr,
        method="diffusion",
        output_dim=1,
        scale_factor=1.0,
        norm=False,
    )
    gradient2 = nettools.dimreduce_matrix(
        matrix_arr,
        method="diffusion",
        output_dim=2,
        scale_factor=1.0,
        norm=False,
    )
    return np.column_stack((np.asarray(gradient2, dtype=float), np.asarray(gradient1, dtype=float)))


def _gradient_values_from_payload(matrix, payload, parcel_labels, parcel_names):
    coords = _gradient_pair_from_matrix(matrix)
    reference_coords, matched = _payload_reference_coords(payload, parcel_labels, parcel_names)
    if reference_coords is not None and matched >= 3:
        coords, aligned_count = _align_coords_to_reference(coords, reference_coords)
        if aligned_count >= 3:
            debug.info(f"Aligned group Gradient coordinates to triangular RGB payload ({aligned_count} nodes).")
    else:
        debug.warning("Gradient RGB payload did not match enough parcel labels; using unaligned group coordinates.")
    model = dict(payload.get("model") or {})
    fallback_order = str(payload.get("color_order", model.get("order", "RBG")) or "RBG")
    return _rgb_scalar_from_model(coords[:, 0], coords[:, 1], model, fallback_order=fallback_order)


def _gradient_values_from_matrix(matrix, color_order="RBG"):
    try:
        if hasattr(nettools, "metsim_triangle_scalar_values"):
            return np.asarray(
                nettools.metsim_triangle_scalar_values(
                    matrix,
                    color_order=color_order,
                    x_component=2,
                    y_component=1,
                    scale_factor=1.0,
                    return_details=False,
                ),
                dtype=float,
            )
    except Exception as exc:
        debug.warning(f"Triangular RGB scalar helper failed; falling back to local model: {exc}")
    coords = _gradient_pair_from_matrix(matrix)
    model = nettools.triangular_rgb_model(coords[:, 0], coords[:, 1], color_order=color_order)
    return _rgb_scalar_from_model(coords[:, 0], coords[:, 1], model, fallback_order=color_order)


def _normalize_id(value: str) -> str:
    token = str(value).strip()
    lowered = token.lower()
    if lowered.startswith("sub-"):
        token = token[4:]
        lowered = token.lower()
    if lowered.startswith("ses-"):
        token = token[4:]
    return token.strip()


def _resolve_column(df: pd.DataFrame, name: str) -> str | None:
    if name in df.columns:
        return name
    lower_map = {col.lower(): col for col in df.columns}
    return lower_map.get(name.lower())


def _parse_regressor_spec(spec: str) -> tuple[str | None, list[str]]:
    if spec is None:
        return None, []
    parts = [p.strip() for p in str(spec).split(",") if p.strip()]
    if not parts:
        return None, []
    return parts[0], parts[1:]


def _infer_component_side(mask: np.ndarray, t_matrix: np.ndarray) -> str:
    """Infer component sign from mean edge t-value."""
    try:
        mask = np.asarray(mask, dtype=bool)
        t_matrix = np.asarray(t_matrix, dtype=float)
        if mask.shape != t_matrix.shape or mask.ndim != 2:
            return "n/a"
        upper = np.triu(mask, k=1)
        vals = t_matrix[upper]
        if vals.size == 0:
            return "n/a"
        mean_t = float(np.nanmean(vals))
        if mean_t > 0:
            return "+"
        if mean_t < 0:
            return "-"
        return "0"
    except Exception:
        return "n/a"


# def main() -> None:
parser = argparse.ArgumentParser(
    description="Inspect previously saved NBS component results and reload group connectivity data."
)
parser.add_argument(
    "--result",
    required=True,
    help="Path to the component NPZ produced by nbs_groups.py.",
)
parser.add_argument(
    "--output-path",
    "--output",
    default=None,
    help=(
        "Optional circular-plot output path. The filename must end in .pdf, .svg, "
        "or .png; parent directories are created automatically."
    ),
)
parser.add_argument(
    "--similarity-hubs-left",
    nargs="+",
    default=None,
    help="Ordered list of collapsed node labels forming the left-hemisphere similarity subnetwork.",
)
parser.add_argument(
    "--similarity-hubs-right",
    nargs="+",
    default=None,
    help="Ordered list of collapsed node labels forming the right-hemisphere similarity subnetwork.",
)
parser.add_argument(
    "--nclusters",
    type=int,
    default=5,
    help="Number of Gaussian clusters for Gradient metabolic profiling.",
)
parser.add_argument(
    "--metabolite-delta-mode",
    choices=["absolute", "percent"],
    default="absolute",
    help="How to report metabolite deltas: absolute value or normalized percentage of z-score range.",
)

parser.add_argument(
    "--subject-id",
    action="append",
    default=[],
    help="Subject ID for single-subject analysis (use with --session). Can be provided multiple times.",
)
parser.add_argument(
    "--session",
    action="append",
    default=[],
    help="Session ID for single-subject analysis (use with --subject-id). Can be provided multiple times.",
)
parser.add_argument(
    "--batch",
    action="store_true",
    help="Run analysis for all subject-session pairs found in the NPZ subset.",
)
parser.add_argument(
    "--align-compare-gradient",
    action="store_true",
    dest="align_compare_gradient",
    help="Compatibility flag; Gradient alignment is handled by the RGB payload when available.",
)
parser.add_argument(
    "--align-compare-msmode",
    action="store_true",
    dest="align_compare_gradient",
    help=argparse.SUPPRESS,
)
parser.add_argument(
    "--aggregate-deltas",
    action="store_true",
    help="After batch processing, aggregate all *_metab_profile_deltas.tsv and plot median/CI.",
)
parser.add_argument(
    "--no-show",
    action="store_true",
    help="Disable interactive plot display (useful for batch mode).",
)
parser.add_argument(
    "--hide-node-labels",
    "--hide-label-names",
    action="store_true",
    help="Hide parcel names around the circular plot.",
)
parser.add_argument(
    "--hide-colorbar",
    "--hide-colorbars",
    action="store_true",
    help="Hide all circular-plot colorbars and their labels.",
)
parser.add_argument(
    "--theme",
    "--plot-theme",
    choices=["dark", "light"],
    default="dark",
    help="Circular-plot theme. Default: dark; light uses a white background and black path edges.",
)
parser.add_argument(
    "--display_ppath",
    "--display-ppath",
    action="store_true",
    help="Overlay similarity path (ppath) on the connectivity circle.",
)
parser.add_argument(
    "--comp",
    type=int,
    default=-1,
    help="Component index to analyze (default: -1 uses union of all components).",
)
parser.add_argument(
    "--regressor",
    default=None,
    help=(
        "Override regressor for group split using covariates. "
        "Format: name or name,val1,val2 (e.g. diag,1,2)."
    ),
)
parser.add_argument(
    "--regressor-type",
    "--regressor_type",
    choices=["categorical", "continuous"],
    default="categorical",
    help="Interpret regressor as categorical labels or continuous numeric values.",
)
parser.add_argument(
    "--collapse-parcels",
    action="store_true",
    help="Collapse parcel names ending with _<int> into base parcels before plotting.",
)
parser.add_argument(
    "--exclude-parcel-substring",
    "--exclude-parcel-substrings",
    action="append",
    default=[],
    metavar="TEXT",
    help=(
        "Exclude circular-plot parcels whose names contain TEXT (case-insensitive). "
        "Repeat this option or provide comma-separated substrings."
    ),
)
parser.add_argument(
    "--highlight-paths-npz",
    "--highlight-path-npz",
    "--paths-npz",
    default=None,
    help=(
        "Optional *_desc-free_energy_paths.npz archive. One saved CTX path is selected "
        "and only its parcels retain their group colors in the circular plot."
    ),
)
parser.add_argument(
    "--highlight-path-group",
    choices=["any", "lh", "rh", "both"],
    default="any",
    help=(
        "Hemisphere from which to select a full saved CTX path. 'both' selects one "
        "independent path from each hemisphere. Default: any."
    ),
)
parser.add_argument(
    "--highlight-path-index",
    type=int,
    default=None,
    help="Optional 1-based path index within the selected group; default selects randomly.",
)
parser.add_argument(
    "--highlight-path-seed",
    type=int,
    default=None,
    help="Optional random seed used when --highlight-path-index is omitted.",
)
parser.add_argument(
    "--gradient-rgb-payload",
    default=None,
    help=(
        "JSON payload exported by the active Gradient scatter triangular RGB view. "
        "When present, NBS node values are scalarized from that fitted RGB model."
    ),
)

args = parser.parse_args()
if args.output_path:
    args.output_path = os.path.abspath(os.path.expanduser(args.output_path))
    output_extension = os.path.splitext(args.output_path)[1].lower()
    if output_extension not in {".pdf", ".svg", ".png"}:
        parser.error("--output-path must end in .pdf, .svg, or .png.")
    if args.batch:
        parser.error("--output-path cannot be used with --batch because each run needs a unique file.")
if not args.highlight_paths_npz and (
    args.highlight_path_index is not None or args.highlight_path_seed is not None
):
    parser.error(
        "--highlight-path-index/--highlight-path-seed require --highlight-paths-npz."
    )
base_plot_dir = os.path.split(args.result)[0]
target_pairs = []
show_figures = False
gradient_rgb_payload = _load_gradient_rgb_payload(args.gradient_rgb_payload)
if gradient_rgb_payload is not None:
    fit_mode = str(gradient_rgb_payload.get("fit_mode", "triangle") or "triangle")
    color_order = str(gradient_rgb_payload.get("color_order", "RBG") or "RBG")
    debug.info(f"Using active Gradient RGB payload (fit={fit_mode}, order={color_order}).")
highlight_path_tokens = set()
highlight_path_infos = []
highlight_gm_edges = []
highlight_adjacency_path = ""
if args.highlight_paths_npz:
    (
        highlight_path_tokens,
        highlight_path_infos,
        highlight_gm_edges,
        highlight_adjacency_path,
    ) = load_highlight_path_sequence(
        args.highlight_paths_npz,
        path_group=args.highlight_path_group,
        path_index=args.highlight_path_index,
        random_seed=args.highlight_path_seed,
        collapse_parcels=args.collapse_parcels,
    )
    for highlight_path_info in highlight_path_infos:
        debug.info(
            "Selected complete three-endpoint highlight path "
            f"{highlight_path_info['candidate_index']}/{highlight_path_info['candidate_count']} "
            f"({highlight_path_info['group']} record {highlight_path_info['record_index']}, "
            f"order={highlight_path_info['path_order']}, "
            f"segments={','.join(highlight_path_info['segment_labels'])}, "
            f"{highlight_path_info['unique_parcel_count']} unique parcels) from "
            f"{highlight_path_info['input_path']}"
        )
        debug.info("Highlight path sequence: " + " -> ".join(highlight_path_info["node_names"]))
    debug.info(f"Using GM adjacency for highlighted path edges: {highlight_adjacency_path}")

if args.subject_id or args.session:
    if len(args.subject_id) != len(args.session):
        raise ValueError("Provide the same number of --subject-id and --session entries.")
    for sid, ses in zip(args.subject_id, args.session):
        target_pairs.append((sid.strip(), ses.strip()))
target_pair_set = set(target_pairs)
if len(target_pair_set) == 1:
    sub_id, ses_id = next(iter(target_pair_set))
    plot_dir = os.path.join(base_plot_dir, f"sub-{sub_id}_ses-{ses_id}")
elif target_pair_set:
    plot_dir = os.path.join(base_plot_dir, "custom_selection")
else:
    plot_dir = os.path.join(base_plot_dir, "group")
os.makedirs(plot_dir, exist_ok=True)
similarity_hubs = {
    "left": args.similarity_hubs_left if args.similarity_hubs_left else [],
    "right": args.similarity_hubs_right if args.similarity_hubs_right else [],
}
if not similarity_hubs["left"] or not similarity_hubs["right"]:
    glob_pattern = os.path.join(
        "results",
        "controls_vs_patients",
        "path_disruptions",
        "*",
        "controls-population_average",
        "metabolic_ppath_*controls-population average_atlas-chimera-LFMIHIFIS-3_nperm-100_desc-ctx_start-*_stop-*_l-*.csv",
    )
    candidate_files = sorted(glob(glob_pattern))
    hubs_loaded = {"left": False, "right": False}
    for csv_path in candidate_files:
        try:
            df_hubs = pd.read_csv(csv_path)
        except Exception as err:
            debug.warning(f"Failed to load similarity hubs from {csv_path}: {err}")
            continue
        for hemi_key, label in (("left", "LH"), ("right", "RH")):
            if similarity_hubs[hemi_key]:
                hubs_loaded[hemi_key] = True
                continue
            subset = df_hubs[df_hubs["hemisphere"].str.upper() == label]
            if subset.empty:
                continue
            similarity_hubs[hemi_key] = subset["node_label"].astype(str).tolist()
            hubs_loaded[hemi_key] = True
        if hubs_loaded["left"] and hubs_loaded["right"]:
            debug.success(f"Loaded similarity hubs from {csv_path}")
            break
    else:
        debug.warning("Could not auto-load similarity hubs; overlay will be skipped.")

npz_path = args.result
if not isfile(npz_path):
    raise FileNotFoundError(f"Result file not found: {npz_path}")

data = np.load(npz_path, allow_pickle=True)
component_idx = int(data.get("component_idx", -1))
pvalue = float(data.get("pvalue", np.nan))
permtest = data.get("permtest", "")
nperm = int(data.get("nperm", 0))
t_thresh = float(data.get("t_thresh", 0.0))
npert = int(data.get("npert", 0))
preproc = str(data.get("preproc", ""))
debug.info(
    f"Component {component_idx}: p={pvalue:.6f}, permtest={permtest}, "
    f"nperm={nperm}, t_thresh={t_thresh}"
)

metrics = {
    "group": str(data.get("group", "")),
    "parc_scheme": str(data.get("parc_scheme", "")),
    "scale": int(data.get("scale", 0)),
    "diag": str(data.get("diag", "")),
    "lobes": str(data.get("lobes", "")),
    "param_tag": str(data.get("param_tag", "")),
    "npert": npert,
    "preproc": preproc,
}
debug.display_dict(metrics, title="Stored metadata")

comp_masks = data.get("comp_masks")
comp_mask = None
if comp_masks is not None:
    comp_masks = np.asarray(comp_masks)
    if comp_masks.ndim == 2:
        comp_masks = comp_masks[None, ...]
    if args.comp is not None and args.comp >= 0:
        if args.comp >= comp_masks.shape[0]:
            raise ValueError(f"--comp {args.comp} out of range (0..{comp_masks.shape[0]-1}).")
        comp_mask = comp_masks[args.comp]
    else:
        comp_mask = np.any(comp_masks.astype(bool), axis=0)
    comp_mask = np.asarray(comp_mask, dtype=bool)
    debug.info(f"Component mask shape: {comp_mask.shape}, edges: {int(comp_mask.sum() // 2)}")
else:
    comp_mask = data.get("comp_mask")
    if comp_mask is not None:
        comp_mask = np.asarray(comp_mask, dtype=bool)
        debug.info(f"Component mask shape: {comp_mask.shape}, edges: {int(comp_mask.sum() // 2)}")


t_mat = data.get("t_matrix")
if t_mat is not None:
    debug.info(f"T-matrix shape: {np.asarray(t_mat).shape}")

test_type_label = str(data.get("test_type", "n/a")).lower() or "n/a"
test_tail_label = str(data.get("test_tail", "n/a")).lower() or "n/a"
comp_pvals = np.asarray(data.get("comp_pvals", []), dtype=float)
sig_indices = np.asarray(data.get("sig_indices", []), dtype=int)
overall_pvalue = np.nan
if comp_pvals.size > 0:
    overall_pvalue = float(np.nanmin(comp_pvals))
else:
    scalar_p = data.get("pvalue")
    if scalar_p is not None:
        try:
            overall_pvalue = float(np.asarray(scalar_p).squeeze())
        except Exception:
            overall_pvalue = np.nan
component_masks_for_table = []
if comp_masks is not None:
    if isinstance(comp_masks, np.ndarray) and comp_masks.ndim == 3:
        component_masks_for_table = [np.asarray(m, dtype=bool) for m in comp_masks]
    elif isinstance(comp_masks, np.ndarray) and comp_masks.ndim == 2:
        component_masks_for_table = [np.asarray(comp_masks, dtype=bool)]
table_pvals = Table(title="NBS P-Values")
table_pvals.add_column("Item")
table_pvals.add_column("Test")
table_pvals.add_column("Tail/Side")
table_pvals.add_column("p-value", justify="right")
table_pvals.add_column("Sig", justify="right")
overall_sig = "yes" if sig_indices.size > 0 else "no"
overall_p_str = "N/A" if np.isnan(overall_pvalue) else f"{overall_pvalue:.6f}"
table_pvals.add_row("overall", test_type_label.upper(), test_tail_label, overall_p_str, overall_sig)
if comp_pvals.size > 0:
    sig_set = set(int(i) for i in sig_indices.tolist())
    for comp_i, pval_i in enumerate(comp_pvals.tolist()):
        side = test_tail_label
        if test_tail_label == "both" and comp_i < len(component_masks_for_table) and t_mat is not None:
            side = f"both ({_infer_component_side(component_masks_for_table[comp_i], t_mat)})"
        pval_str = "N/A" if np.isnan(pval_i) else f"{float(pval_i):.6f}"
        is_sig = "yes" if comp_i in sig_set else "no"
        table_pvals.add_row(f"comp {comp_i}", test_type_label.upper(), side, pval_str, is_sig)
else:
    table_pvals.add_row("components", test_type_label.upper(), test_tail_label, "N/A", "no")
console.print(table_pvals)



connectivity_path = data.get("connectivity_path")
debug.info(connectivity_path)
if connectivity_path is None or str(connectivity_path).strip() == "":
    raise ValueError("Connectivity file path missing in NPZ; regenerate results with updated nbs_groups.py.")

connectivity_path = str(connectivity_path)
if not isfile(connectivity_path):
    raise FileNotFoundError(f"Connectivity NPZ not found: {connectivity_path}")

group_data = np.load(connectivity_path,allow_pickle=True)
covars_df = pd.DataFrame.from_records(group_data["covars"])
MeSiM_pop_avg = group_data["matrix_pop_avg"]
MeSiM_all = np.asarray(group_data["matrix_subj_list"])
subject_id_all = np.asarray(group_data["subject_id_list"]).astype(str)
session_all = np.asarray(group_data["session_id_list"]).astype(str)
metabolites = group_data["metabolites"]
metab_profiles_all = np.asarray(group_data["metab_profiles_subj_list"])
metab_profiles_all = zscore(metab_profiles_all, axis=1)

parcel_labels = np.asarray(group_data["parcel_labels_group"])
parcel_names = np.asarray(group_data["parcel_names_group"])

parc_scheme = metrics["parc_scheme"]
scale = metrics["scale"]
atlas = f"cubic-{scale}" if "cubic" in parc_scheme else f"chimera-{parc_scheme}-{scale}"
parcel_mni_img_nii = nib.load(join(dutils.DEVDATAPATH, "atlas", atlas, f"{atlas}.nii.gz"))
parcel_mni_img_np = parcel_mni_img_nii.get_fdata().astype(int)
mni_template = datasets.load_mni152_template()
centroids_world = nettools.compute_centroids(parcel_mni_img_nii, parcel_labels, world=True)
parcel_df = pd.DataFrame({
    "label": parcel_labels,
    "name": parcel_names,
    "XCoord(mm)": centroids_world[:, 0],
    "YCoord(mm)": centroids_world[:, 1],
    "ZCoord(mm)": centroids_world[:, 2],
}).reset_index(drop=True)
debug.info(
    f"Loaded full MeSiM array with shape {MeSiM_all.shape} "
    f"(subjects: {MeSiM_all.shape[0]}, parcels: {parcel_labels.shape[0]})"
)

stored_subjects = np.asarray(data.get("subject_ids", []), dtype=str)
stored_sessions = np.asarray(data.get("session_ids", []), dtype=str)
regressor_name = str(data.get("regressor_name", ""))
regressor_values_raw = np.asarray(data.get("regressor_values", []))
if args.regressor_type == "continuous":
    regressor_values = pd.to_numeric(regressor_values_raw, errors="coerce").astype(float)
else:
    regressor_values = np.array(
        [str(v) if not pd.isna(v) else np.nan for v in regressor_values_raw], dtype=object
    )
if stored_subjects.size == 0 or stored_sessions.size == 0:
    raise ValueError("NPZ missing subject/session identifiers for the analyzed subset.")

regressor_filter_values = None
override_regressor, override_values = _parse_regressor_spec(args.regressor)
if override_regressor:
    if covars_df is None or covars_df.empty:
        raise ValueError("--regressor specified but covariates table is missing.")
    reg_col = _resolve_column(covars_df, override_regressor)
    if reg_col is None:
        raise ValueError(
            f"Regressor '{override_regressor}' not found in covariates. "
            f"Available: {list(covars_df.columns)}"
        )
    subj_col = _resolve_column(covars_df, "participant_id") or _resolve_column(
        covars_df, "subject_id"
    )
    ses_col = _resolve_column(covars_df, "session_id") or _resolve_column(
        covars_df, "session"
    )
    if subj_col is None or ses_col is None:
        raise ValueError(
            "Covariates table must contain participant_id and session_id columns to "
            "override regressor."
        )
    covar_lookup = {
        (_normalize_id(sid), _normalize_id(ses)): val
        for sid, ses, val in covars_df[[subj_col, ses_col, reg_col]].itertuples(
            index=False, name=None
        )
    }
    reg_vals = []
    missing = 0
    for sid, ses in zip(stored_subjects, stored_sessions):
        key = (_normalize_id(sid), _normalize_id(ses))
        if key not in covar_lookup:
            reg_vals.append(np.nan)
            missing += 1
        else:
            reg_vals.append(covar_lookup[key])
    if missing:
        debug.warning(
            f"{missing} subject/session pairs missing regressor '{reg_col}' values; "
            "they will be dropped."
        )
    reg_series = covars_df[reg_col]
    if args.regressor_type == "continuous":
        if not pd.api.types.is_numeric_dtype(reg_series):
            debug.warning(
                f"Regressor '{reg_col}' is non-numeric but regressor type is continuous; "
                "coercing to numeric with NaNs for non-convertible values."
            )
        regressor_values = pd.to_numeric(reg_vals, errors="coerce").astype(float)
        if override_values:
            regressor_filter_values = np.array([float(v) for v in override_values], dtype=float)
    else:
        if pd.api.types.is_numeric_dtype(reg_series):
            debug.info(
                f"Regressor '{reg_col}' is numeric but regressor type is categorical; "
                "treating values as labels."
            )
        regressor_values = np.array(
            [str(v) if not pd.isna(v) else np.nan for v in reg_vals], dtype=object
        )
        if override_values:
            regressor_filter_values = np.array([str(v) for v in override_values], dtype=object)
    regressor_name = reg_col
    debug.info(f"Using regressor '{regressor_name}' from covariates.")


if regressor_values.size != stored_subjects.size:
    raise ValueError("Regressor values length does not match stored subjects.")

# batch mode: loop over all pairs via subprocess to reuse logic
if args.batch:
    debug.info(f"Batch mode enabled: will iterate over {len(stored_subjects)} subject-session pairs.")
    for sid, ses in zip(stored_subjects, stored_sessions):
        cmd = [
            sys.executable,
            __file__,
            "--result",
            args.result,
            "--subject-id",
            sid,
            "--session",
            ses,
            "--nclusters",
            str(args.nclusters),
            "--metabolite-delta-mode",
            args.metabolite_delta_mode,
        ]
        if args.regressor:
            cmd.extend(["--regressor", args.regressor])
        if args.collapse_parcels:
            cmd.append("--collapse-parcels")
        for substring in args.exclude_parcel_substring:
            cmd.extend(["--exclude-parcel-substring", substring])
        if args.highlight_paths_npz:
            cmd.extend(["--highlight-paths-npz", args.highlight_paths_npz])
            cmd.extend(["--highlight-path-group", args.highlight_path_group])
            if args.highlight_path_index is not None:
                cmd.extend(["--highlight-path-index", str(args.highlight_path_index)])
            if args.highlight_path_seed is not None:
                cmd.extend(["--highlight-path-seed", str(args.highlight_path_seed)])
        if args.gradient_rgb_payload:
            cmd.extend(["--gradient-rgb-payload", args.gradient_rgb_payload])
        if args.aggregate_deltas:
            cmd.append("--aggregate-deltas")
        if args.hide_node_labels:
            cmd.append("--hide-node-labels")
        if args.hide_colorbar:
            cmd.append("--hide-colorbar")
        cmd.extend(["--theme", args.theme])
        cmd.append("--no-show")
        subprocess.run(cmd, check=False)
    if args.aggregate_deltas:
        base_name = os.path.splitext(os.path.basename(args.result))[0]
        delta_files = glob(os.path.join(base_plot_dir, "sub-*", f"{base_name}_metab_profile_deltas.tsv"))
        delta_files += glob(os.path.join(base_plot_dir, "custom_selection", f"{base_name}_metab_profile_deltas.tsv"))
        if delta_files:
            df_list = [pd.read_csv(f) for f in delta_files]
            df_all = pd.concat(df_list, ignore_index=True)
            metab_cols = [col for col in df_all.columns if col.startswith("delta")]
            metab_cols = metab_cols[:5]
            node_labels = sorted(df_all["node_label"].unique())
            fig, axes = plt.subplots(5, 1, figsize=(12, 20), sharex=True)
            for idx, col in enumerate(metab_cols):
                stats = df_all.groupby("node_label")[col].agg(
                    median=lambda x: np.nanmedian(x),
                    low=lambda x: np.nanpercentile(x, 25),
                    high=lambda x: np.nanpercentile(x, 75),
                )
                stats = stats.reindex(node_labels)
                x = np.arange(len(node_labels))
                axes[idx].plot(x, stats["median"], color="black", label="median")
                axes[idx].fill_between(x, stats["low"], stats["high"], color="gray", alpha=0.3, label="IQR")
                axes[idx].set_title(col)
                axes[idx].set_ylabel("Delta")
            axes[-1].set_xticks(np.arange(len(node_labels)))
            axes[-1].set_xticklabels(node_labels, rotation=90, fontsize=8)
            axes[0].legend()
            fig.tight_layout()
            agg_path = os.path.join(base_plot_dir, f"{base_name}_metab_delta_summary.pdf")
            fig.savefig(agg_path, dpi=300, bbox_inches="tight")
            debug.success(f"Saved aggregated delta summary to {agg_path}")
    sys.exit(0)

index_lookup = {
    (sid, ses): idx for idx, (sid, ses) in enumerate(zip(subject_id_all, session_all))
}

kept_subjects = []
kept_sessions = []
kept_regressors = []
for sid, ses, val in zip(stored_subjects, stored_sessions, regressor_values):
    if (sid, ses) in index_lookup:
        kept_subjects.append(sid)
        kept_sessions.append(ses)
        kept_regressors.append(val)
stored_subjects = np.asarray(kept_subjects, dtype=str)
stored_sessions = np.asarray(kept_sessions, dtype=str)
regressor_values = np.asarray(kept_regressors)
if regressor_filter_values is not None:
    filter_mask = np.isin(regressor_values, regressor_filter_values)
    if not np.any(filter_mask):
        raise ValueError(
            f"No subjects match --regressor values {regressor_filter_values} "
            f"for {regressor_name}."
        )
    stored_subjects = stored_subjects[filter_mask]
    stored_sessions = stored_sessions[filter_mask]
    regressor_values = regressor_values[filter_mask]
    debug.info(
        f"Restricting regressor '{regressor_name}' to values {list(regressor_filter_values)}."
    )

nan_mask = pd.isna(regressor_values)
if np.any(nan_mask):
    stored_subjects = stored_subjects[~nan_mask]
    stored_sessions = stored_sessions[~nan_mask]
    regressor_values = regressor_values[~nan_mask]

if regressor_values.size:
    counts = pd.Series(regressor_values).value_counts(dropna=False)
    count_pairs = ", ".join(f"{idx}: {val}" for idx, val in counts.items())
    debug.info(f"Regressor '{regressor_name}' value counts: {count_pairs}")

selected_indices = np.array(
    [index_lookup[(sid, ses)] for sid, ses in zip(stored_subjects, stored_sessions)],
    dtype=int,
)
if selected_indices.size == 0:
    raise ValueError("No subject-session pairs from NPZ found in connectivity file.")
analysis_mask = np.ones(selected_indices.shape[0], dtype=bool)
mesim_subset = MeSiM_all[selected_indices]
metab_subset = metab_profiles_all[selected_indices]
metab_value_scale = None
if metab_subset.size > 0:
    metab_max_abs = np.nanmax(np.abs(metab_subset), axis=(0, 1, 3))
    metab_value_scale = np.where(metab_max_abs == 0, np.nan, metab_max_abs)
debug.info(
    f"Subset MeSiM shape {mesim_subset.shape}; metab profiles shape {metab_subset.shape}"
)

unique_groups = np.unique(regressor_values)
if regressor_filter_values is not None:
    present = []
    missing = []
    for val in regressor_filter_values:
        if np.any(regressor_values == val):
            present.append(val)
        else:
            missing.append(val)
    if missing:
        debug.warning(
            f"Requested regressor values {missing} not present for '{regressor_name}'."
        )
    unique_groups = np.array(present, dtype=regressor_values.dtype)
if unique_groups.size != 2:
    debug.warning(
        f"Expected binary regressor but found values {unique_groups}. Proceeding anyway."
    )
control_group_value = unique_groups[0] if unique_groups.size else None

target_reg_value = None
target_idx = None
if target_pair_set:
    # pick the first matching target pair
    for idx, (sid, ses) in enumerate(zip(stored_subjects, stored_sessions)):
        if (sid, ses) in target_pair_set:
            target_idx = idx
            target_reg_value = regressor_values[idx]
            break
    if target_idx is None:
        raise ValueError(f"Specified subject-session pairs not found in stored subset.{target_pair_set}")

group_splits = {}
for value in unique_groups:
    mask = regressor_values == value
    group_splits[value] = {
        "mesim": mesim_subset[mask],
        "metab_profiles": metab_subset[mask],
        "subjects": stored_subjects[mask],
        "sessions": stored_sessions[mask],
    }
    if group_splits[value]["mesim"].size == 0:
        debug.warning(f"No subjects in group {value} for Gradient/metab calculations; skipping this group.")
    else:
        debug.info(
            f"Group {value}: MeSiM shape {group_splits[value]['mesim'].shape}, "
            f"metab_profiles shape {group_splits[value]['metab_profiles'].shape}"
        )
if target_idx is not None and target_reg_value in group_splits:
    if target_reg_value == control_group_value:
        debug.info("Target pair belongs to control group; keeping control group averaged.")
    else:
        # override compare group with only the target subject/session for Gradient and metab profiles
        group_mask = regressor_values == target_reg_value
        group_indices = np.where(group_mask)[0]
        if target_idx in group_indices:
            local_idx = int(np.where(group_indices == target_idx)[0][0])
            group_splits[target_reg_value]["mesim"] = group_splits[target_reg_value]["mesim"][local_idx : local_idx + 1]
            group_splits[target_reg_value]["metab_profiles"] = group_splits[target_reg_value]["metab_profiles"][local_idx : local_idx + 1]
            debug.info(
                f"Using subject-session {stored_subjects[target_idx]}-{stored_sessions[target_idx]} "
                f"as the compare representative for regressor value {target_reg_value}."
            )
        else:
            debug.warning(
                f"Target pair {stored_subjects[target_idx]}-{stored_sessions[target_idx]} not found in compare group indices; skipping override."
            )


########################## Gradient comparison between groups ##########################
if comp_mask is None:
    raise ValueError("Component mask missing from NPZ; cannot extract NBS nodes.")

mask_bool = np.asarray(comp_mask, dtype=bool)
significant_nodes = np.where(mask_bool.sum(axis=0) > 0)[0]
if significant_nodes.size == 0:
    debug.warning("No nodes associated with the NBS component; skipping edge plotting.")

gradient_by_group: dict[float, np.ndarray] = {}
gradient_color_order = (
    str(gradient_rgb_payload.get("color_order", "RBG") or "RBG")
    if gradient_rgb_payload is not None
    else "RBG"
)
for value in unique_groups:
    group_mesim = group_splits[value]["mesim"]
    if group_mesim.size == 0:
        debug.warning(f"No subjects in group {value} for Gradient calculation; skipping.")
        continue
    avg_mesim = np.nanmean(group_mesim, axis=0)
    if gradient_rgb_payload is not None:
        gradient_values = _gradient_values_from_payload(
            avg_mesim,
            gradient_rgb_payload,
            parcel_labels,
            parcel_names,
        )
    else:
        gradient_values = _gradient_values_from_matrix(avg_mesim, color_order=gradient_color_order)
    gradient_by_group[value] = np.asarray(gradient_values, dtype=float)
    finite_gradient = gradient_by_group[value][np.isfinite(gradient_by_group[value])]
    if finite_gradient.size:
        debug.info(
            f"Gradient {value}: {finite_gradient.min():.3f} --> {finite_gradient.max():.3f}"
        )
    else:
        debug.warning(f"Gradient {value}: no finite values resolved.")


if unique_groups.size >= 2 and len(gradient_by_group) >= 2 and significant_nodes.size > 0:
    ref = unique_groups[0]
    compare = unique_groups[1]
    if ref not in gradient_by_group or compare not in gradient_by_group:
        ref_values_raw = np.array([])
        cmp_values_raw = np.array([])
        debug.warning("One of the groups lacks Gradient data; skipping Gradient delta computation.")
    else:
        ref_values_raw = gradient_by_group[ref][significant_nodes]
        cmp_values_raw = gradient_by_group[compare][significant_nodes]
else:
    ref_values_raw = np.array([])
    cmp_values_raw = np.array([])


if ref_values_raw.size > 0 and cmp_values_raw.size > 0:
    merged_curves = OrderedDict()
    node_labels = parcel_names[significant_nodes]
    for label, val_ref, val_cmp in zip(node_labels, ref_values_raw, cmp_values_raw):
        base = re.sub(r"_(\d+)$", "", label)
        merged_curves.setdefault(base, {"ref": [], "cmp": []})
        merged_curves[base]["ref"].append(val_ref)
        merged_curves[base]["cmp"].append(val_cmp)
    merged_labels = list(merged_curves.keys())
    ref_values = np.array([_combine_signed(vals["ref"]) for vals in merged_curves.values()])
    cmp_values = np.array([_combine_signed(vals["cmp"]) for vals in merged_curves.values()])
    delta = cmp_values - ref_values
    debug.info(
        f"Delta Gradient ({compare} - {ref}) on {len(merged_labels)} merged nodes: "
        f"min={delta.min():.3f}, max={delta.max():.3f}"
    )
    tick_labels = ["\n".join(lbl.split()) for lbl in merged_labels]
    fig_width = max(8, len(merged_labels) * 0.6)
    compare_label = (
        f"{stored_subjects[target_idx]}-{stored_sessions[target_idx]}"
        if target_idx is not None and target_reg_value != control_group_value
        else "patients"
    )
    fig, ax = plt.subplots(figsize=(fig_width, 8))
    ax.plot(cmp_values, marker="o", linestyle="-", color="tab:red", label=compare_label)
    ax.plot(ref_values, marker="s", linestyle="-", color="tab:green", label="controls")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Gradient", fontsize=FONTSIZE)
    ax.set_xlabel("NBS Nodes", fontsize=FONTSIZE)
    ax.legend(loc="best")
    ax.set_xticks(range(len(merged_labels)))
    ax.set_xticklabels(tick_labels, fontsize=FONTSIZE - 6, rotation=90, va="top")
    ax.tick_params(axis="y", labelsize=FONTSIZE - 6)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(6))
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    table = Table(title=f"Gradient Δ ({compare} - {ref})")
    table.add_column("Node Label")
    table.add_column("Δ Gradient", justify="right")
    for label, value in zip(merged_labels, delta):
        table.add_row(label, f"{value:.3f}")
    # console.print(table)
    if npz_path:
        base_name = os.path.splitext(os.path.basename(npz_path))[0]
        data_path = os.path.join(plot_dir, f"{base_name}_gradient_values.tsv")
        df_values = pd.DataFrame({
            "node_label": merged_labels,
            f"gradient_group_{compare}": cmp_values,
            f"gradient_group_{ref}": ref_values,
            "delta": delta,
        })
        df_values.to_csv(data_path, sep="\t", index=False)
        debug.success("Saved Gradient table to", data_path)
    if npz_path:
        base_name = os.path.splitext(os.path.basename(npz_path))[0]
        plot_path_linear = os.path.join(plot_dir, f"{base_name}_gradient_profile.pdf")
        plt.savefig(plot_path_linear, bbox_inches="tight", dpi=300)
        debug.success("Saved Gradient comparison plot to", plot_path_linear)
    show_figures = True
    try:
        delta_metab, cluster_ids = compute_cluster_metab_deltas(
            cmp_values_raw,
            ref_values_raw,
            significant_nodes,
            group_splits,
            compare,
            ref,
            args.nclusters,
        )
    except Exception as err:
        debug.warning(f"Failed to compute metabolic profile deltas: {err}")
        delta_metab = None
        cluster_ids = None
    if delta_metab is not None and cluster_ids is not None:
        metabolite_names = [str(m) for m in metabolites[: delta_metab.shape[1]]]
        delta_display = delta_metab.copy()
        column_prefix = "delta_"
        value_suffix = ""
        if args.metabolite_delta_mode == "percent" and metab_value_scale is not None:
            denom = metab_value_scale[: delta_metab.shape[1]]
            with np.errstate(divide="ignore", invalid="ignore"):
                delta_display = np.divide(
                    delta_metab,
                    denom.reshape(1, -1),
                    out=np.full_like(delta_metab, np.nan),
                    where=~np.isnan(denom.reshape(1, -1)),
                ) * 100.0
            column_prefix = "delta_pct_"
            value_suffix = " (%)"
        df_cluster = pd.DataFrame(
            delta_display, columns=[f"{column_prefix}{m}" for m in metabolite_names]
        )
        df_cluster.insert(0, "cluster_id", cluster_ids)
        node_labels_full = parcel_names[significant_nodes]
        df_cluster.insert(0, "node_label", node_labels_full)
        delta_lookup = {label: val for label, val in zip(merged_labels, delta)}
        df_cluster["delta_gradient"] = [
            delta_lookup.get(re.sub(r"_(\d+)$", "", name), np.nan)
            for name in node_labels_full
        ]
        ref_lookup = {label: val for label, val in zip(merged_labels, ref_values)}
        cmp_lookup = {label: val for label, val in zip(merged_labels, cmp_values)}
        df_cluster["gradient_ref"] = [
            ref_lookup.get(re.sub(r"_(\d+)$", "", name), np.nan) for name in node_labels_full
        ]
        df_cluster["gradient_cmp"] = [
            cmp_lookup.get(re.sub(r"_(\d+)$", "", name), np.nan) for name in node_labels_full
        ]
        table_clusters = Table(title=f"Metabolic Profile Δ per Node (n_clusters={args.nclusters})")
        table_clusters.add_column("Node Label")
        table_clusters.add_column("Cluster", justify="right")
        table_clusters.add_column("Δ Gradient", justify="right")
        table_clusters.add_column("Gradient (ref)", justify="right")
        table_clusters.add_column("Gradient (cmp)", justify="right")
        for m in metabolite_names:
            table_clusters.add_column(f"Δ {m}{value_suffix}", justify="right")

        is_percent_mode = args.metabolite_delta_mode == "percent" and bool(value_suffix)

        def _format_delta(value: float, as_percent: bool = False) -> str:
            if np.isnan(value):
                return "N/A"
            color = "[green]" if value >= 0 else "[red]"
            formatted = f"{value:.4f}"
            if as_percent:
                formatted = f"{formatted}%"
            return f"{color}{formatted}[/]"

        for _, row in df_cluster.iterrows():
            row_values = [
                row["node_label"],
                str(row["cluster_id"]),
                _format_delta(row["delta_gradient"]),
                f"{row['gradient_ref']:.4f}",
                f"{row['gradient_cmp']:.4f}",
            ] + [
                _format_delta(
                    row[f"{column_prefix}{m}"], as_percent=is_percent_mode
                )
                for m in metabolite_names
            ]
            table_clusters.add_row(*row_values)
        console.print(table_clusters)
        if npz_path:
            base_name = os.path.splitext(os.path.basename(npz_path))[0]
            delta_path = os.path.join(plot_dir, f"{base_name}_metab_profile_deltas.tsv")
            df_cluster.to_csv(delta_path, sep="\t", index=False)
            debug.success("Saved metabolic profile deltas to", delta_path)
else:
    if significant_nodes.size == 0:
        debug.warning("No significant nodes; skipping Gradient deltas.")
    else:
        debug.warning("Less than two groups present; cannot compute Gradient deltas.")

# Plot circular connectivity for the significant component (or empty edges)
if t_mat is not None and comp_mask is not None:
    t_matrix = np.asarray(t_mat)
    mask_bool = np.asarray(comp_mask, dtype=bool)
    has_edges = significant_nodes.size > 0
    if mask_bool.shape != t_matrix.shape:
        debug.warning(
            f"Component mask shape {mask_bool.shape} does not match t-matrix "
            f"{t_matrix.shape}; drawing nodes only."
        )
        has_edges = False
        tmasked = np.zeros_like(t_matrix)
    else:
        tmasked = np.where(mask_bool, t_matrix, 0.0) if has_edges else np.zeros_like(t_matrix)
    vmax = np.nanmax(np.abs(tmasked))
    if vmax == 0:
        vmax = 1.0
    order_group = unique_groups[0]
    plot_group = unique_groups[1] if unique_groups.size > 1 else unique_groups[0]
    node_values_plot = gradient_by_group[plot_group]
    node_values_order = gradient_by_group[order_group]
    if args.collapse_parcels:
        collapse_nodes = (
            significant_nodes if has_edges else np.arange(len(parcel_names))
        )
        collapsed_matrix, collapsed_names, collapsed_values, sig_labels = collapse_parcels(
            tmasked, parcel_names, node_values_plot, collapse_nodes
        )
        collapsed_values_order = None
        if unique_groups.size > 1 or plot_group == order_group:
            _, _, collapsed_values_order, _ = collapse_parcels(
                tmasked, parcel_names, node_values_order, collapse_nodes
            )
        if collapsed_matrix is None or collapsed_names is None:
            collapsed_matrix = tmasked
            collapsed_names = parcel_names
            collapsed_values = node_values_plot
            sig_labels = (
                set(parcel_names[idx] for idx in significant_nodes)
                if has_edges
                else set()
            )
            collapsed_values_order = node_values_order
        if collapsed_values_order is None:
            collapsed_values_order = collapsed_values
    else:
        collapsed_matrix = tmasked
        collapsed_names = parcel_names
        collapsed_values = node_values_plot
        collapsed_values_order = node_values_order
        sig_labels = (
            set(parcel_names[idx] for idx in significant_nodes) if has_edges else set()
        )
    (
        collapsed_matrix,
        collapsed_names,
        collapsed_values,
        collapsed_values_order,
        _,
        excluded_parcel_names,
    ) = exclude_parcels_by_substring(
        collapsed_matrix,
        collapsed_names,
        collapsed_values,
        collapsed_values_order,
        args.exclude_parcel_substring,
    )
    if excluded_parcel_names:
        retained_names = set(collapsed_names)
        sig_labels = {name for name in sig_labels if name in retained_names}
        debug.info(
            f"Excluded {len(excluded_parcel_names)} circular-plot parcels by substring: "
            + ", ".join(excluded_parcel_names)
        )
    def _hemi_side(name: str) -> str:
        lowered = name.lower()
        if "lh" in lowered or "left" in lowered: 
            return "left"
        if "rh" in lowered or "right" in lowered:
            return "right"
        return "unknown"

    left_idx = [i for i, name in enumerate(collapsed_names) if _hemi_side(name) == "left"]
    right_idx = [i for i, name in enumerate(collapsed_names) if _hemi_side(name) == "right"]
    unknown_idx = [i for i, name in enumerate(collapsed_names) if _hemi_side(name) == "unknown"]
    left_order = sorted(left_idx, key=lambda idx: collapsed_values_order[idx])
    right_order = sorted(right_idx, key=lambda idx: collapsed_values_order[idx])
    unknown_order = sorted(unknown_idx, key=lambda idx: collapsed_values_order[idx])
    new_order = left_order + right_order + unknown_order
    if new_order:
        collapsed_matrix = collapsed_matrix[np.ix_(new_order, new_order)]
        collapsed_names = [collapsed_names[i] for i in new_order]
        collapsed_values = collapsed_values[new_order]
        if collapsed_values_order is not None:
            collapsed_values_order = collapsed_values_order[new_order]
    path_edge_mode = bool(highlight_path_tokens)
    if path_edge_mode:
        collapsed_matrix, matched_gm_edges = build_path_gm_adjacency(
            collapsed_names,
            highlight_gm_edges,
            collapse_parcels=args.collapse_parcels,
        )
        if matched_gm_edges == 0:
            raise ValueError(
                "No selected-path GM-adjacency edges matched the circular-plot parcels."
            )
        debug.info(
            f"Replaced the NBS edge network with {matched_gm_edges} GM-adjacency "
            "edges among selected path parcels."
        )
    edge_weight_matrix = np.array(collapsed_matrix, copy=True)
    cmap_nodes = color_loader.load_fsl_cmap(map="spectrum_iso", plotly=False)
    norm_nodes = plt.Normalize(
        vmin=np.nanmin(collapsed_values), vmax=np.nanmax(collapsed_values)
    )
    node_colors = [cmap_nodes(norm_nodes(val)) for val in collapsed_values]
    node_colors_ref = None
    if collapsed_values_order is not None and plot_group != order_group:
        norm_nodes_ref = plt.Normalize(
            vmin=np.nanmin(collapsed_values_order), vmax=np.nanmax(collapsed_values_order)
        )
        node_colors_ref = [cmap_nodes(norm_nodes_ref(val)) for val in collapsed_values_order]
    if highlight_path_tokens:
        highlighted_mask = np.asarray(
            [
                _path_parcel_token(name, collapse_parcels=args.collapse_parcels)
                in highlight_path_tokens
                for name in collapsed_names
            ],
            dtype=bool,
        )
        gray_color = mcolors.to_rgba("#b8b8b8")
        node_colors = [
            color if highlighted else gray_color
            for color, highlighted in zip(node_colors, highlighted_mask)
        ]
        if node_colors_ref is not None:
            node_colors_ref = [
                color if highlighted else gray_color
                for color, highlighted in zip(node_colors_ref, highlighted_mask)
            ]
        debug.info(
            f"Highlighted {int(np.count_nonzero(highlighted_mask))}/{len(collapsed_names)} "
            "circular-plot parcels from the selected saved path."
        )
    node_angles_deg = (
        -np.linspace(0, 360, len(collapsed_names), endpoint=False) + 90
    ) % 360
    node_angles_rad = np.deg2rad(node_angles_deg)
    mat = np.array(collapsed_matrix, dtype=float, copy=True)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(mat, 0.0)

    edge_abs_max = np.nanmax(np.abs(mat)) if mat.size else 0.0
    if not np.isfinite(edge_abs_max) or edge_abs_max <= 0:
        edge_abs_max = 1.0

    collapsed_names_display = [name[:MAXLEN_NODE_NAME] for name in collapsed_names]
    plotted_node_names = (
        [""] * len(collapsed_names_display)
        if args.hide_node_labels
        else collapsed_names_display
    )
    name_lookup = {name.lower(): name for name in collapsed_names}
    for display, full in zip(collapsed_names_display, collapsed_names):
        name_lookup.setdefault(display.lower(), full)
    angle_lookup = {name: angle for name, angle in zip(collapsed_names, node_angles_rad)}
    node_label_fontsize = 16
    theme_facecolor = "black" if args.theme == "dark" else "white"
    theme_textcolor = "white" if args.theme == "dark" else "black"
    path_edge_colormap = "Greys_r" if args.theme == "dark" else "Greys"
    fig_circle, ax_circle = plot_connectivity_circle(
        mat,
        plotted_node_names,
        node_colors=node_colors,
        node_colors_ref=node_colors_ref,
        node_angles=node_angles_deg,
        colormap=path_edge_colormap if path_edge_mode else "PiYG",
        vmin=0.0 if path_edge_mode else -edge_abs_max,
        vmax=1.0 if path_edge_mode else edge_abs_max,
        linewidth=1,
        edge_weights=edge_weight_matrix,
        # title=f"T-matrix masked (component {component_idx})",
        fontsize_title=FONTSIZE - 4,
        fontsize_names=node_label_fontsize,
        facecolor=theme_facecolor,
        textcolor=theme_textcolor,
        colorbar=False,
        show=False,
    )
    fig_circle.set_size_inches(16, 11, forward=True)
    resolved_hubs = {"left": [], "right": []}
    for hemi in ("left", "right"):
        seq = similarity_hubs.get(hemi)
        if not seq:
            continue
        radius = SIMILARITY_PATH_RADII.get(hemi, 10.5)
        resolved_hubs[hemi] = _overlay_similarity_path(
            ax_circle,
            seq,
            name_lookup,
            angle_lookup,
            parcel_names,
            path_label=f"{hemi} similarity hubs",
            radius=radius,
            draw=args.display_ppath,
        )
    similarity_node_set = set(name for values in resolved_hubs.values() for name in values)
    if path_edge_mode:
        debug.info("Similarity Hub vs NBS overlap statistics skipped in saved-path mode.")
    elif similarity_node_set and has_edges:
        overlap_nodes = similarity_node_set & sig_labels
        total_similarity = len(similarity_node_set)
        overlap_pct = len(overlap_nodes) / total_similarity * 100
        rng = np.random.default_rng(42)
        iterations = 5000
        chance_mean_pct = np.nan
        p_value = np.nan
        if total_similarity <= len(collapsed_names):
            collapsed_pool = np.asarray(collapsed_names)
            random_counts = np.zeros(iterations, dtype=int)
            for idx_iter in range(iterations):
                sample = rng.choice(
                    collapsed_pool, size=total_similarity, replace=False
                )
                random_counts[idx_iter] = len(set(sample) & sig_labels)
            chance_mean_pct = (
                random_counts.mean() / total_similarity * 100
            )
            p_value = (np.count_nonzero(random_counts >= len(overlap_nodes)) + 1) / (
                iterations + 1
            )
        table_overlap = Table(title="Similarity Hub vs NBS Overlap")
        table_overlap.add_column("Metric")
        table_overlap.add_column("Value", justify="right")
        table_overlap.add_row("Similarity hubs", str(total_similarity))
        table_overlap.add_row("NBS nodes", str(len(sig_labels)))
        table_overlap.add_row(
            "Observed overlap",
            f"{len(overlap_nodes)} ({overlap_pct:.1f}%)",
        )
        chance_str = (
            f"{chance_mean_pct:.1f}%"
            if not np.isnan(chance_mean_pct)
            else "N/A"
        )
        p_str = f"{p_value:.4f}" if not np.isnan(p_value) else "N/A"
        table_overlap.add_row("Chance overlap (mean)", chance_str)
        table_overlap.add_row("Permutation p-value", p_str)
        table_overlap.add_row(
            "Overlapping nodes",
            ", ".join(sorted(overlap_nodes)) if overlap_nodes else "None",
        )
        console.print(table_overlap)
    elif similarity_node_set and not has_edges:
        debug.warning("No significant nodes; overlap statistics skipped.")
    else:
        debug.warning("No similarity hubs resolved; overlap statistics skipped.")

    bg_luminance = np.mean(mcolors.to_rgb(fig_circle.get_facecolor()))
    label_color = "white" if bg_luminance < 0.5 else "black"
    if not args.hide_colorbar:
        if has_edges and not path_edge_mode:
            sm_edges = plt.cm.ScalarMappable(
                cmap="PiYG", norm=plt.Normalize(vmin=-vmax, vmax=vmax)
            )
            sm_edges.set_array([])
            cb_edges = fig_circle.colorbar(sm_edges, ax=ax_circle, fraction=0.046, pad=0.08)
            cb_edges.set_label("T-Score", fontsize=FONTSIZE, color=label_color)
            cb_edges.ax.yaxis.set_tick_params(labelsize=FONTSIZE - 4, color=label_color)
            plt.setp(plt.getp(cb_edges.ax.axes, "yticklabels"), color=label_color)
        sm_nodes = plt.cm.ScalarMappable(cmap=cmap_nodes, norm=norm_nodes)
        sm_nodes.set_array([])
        cb_nodes = fig_circle.colorbar(sm_nodes, ax=ax_circle, fraction=0.046, pad=0.16)
        cb_nodes.set_label("Gradient", fontsize=FONTSIZE, color=label_color)
        cb_nodes.ax.yaxis.set_tick_params(labelsize=FONTSIZE - 4, color=label_color)
        plt.setp(plt.getp(cb_nodes.ax.axes, "yticklabels"), color=label_color)
    if args.output_path:
        plot_path_circle = args.output_path
    elif npz_path:
        base_name = os.path.splitext(os.path.basename(npz_path))[0]
        circle_suffix = (
            "path_gm_adjacency_circle" if path_edge_mode else "tmask_circle"
        )
        plot_path_circle = os.path.join(plot_dir, f"{base_name}_{circle_suffix}.pdf")
    else:
        plot_path_circle = None
    if plot_path_circle:
        os.makedirs(os.path.dirname(os.path.abspath(plot_path_circle)), exist_ok=True)
        fig_circle.savefig(plot_path_circle, bbox_inches="tight", dpi=300)
        debug.success("Saved circular plot to", plot_path_circle)
    show_figures = True
else:
    debug.warning("Missing t-matrix or component mask; skipping connectivity circle plot.")

if show_figures and not args.no_show:
    plt.show()
