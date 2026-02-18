#!/usr/bin/env python3

import argparse, os
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

from tools.debug import Debug
from tools.datautils import DataUtils
from connectomics.nettools import NetTools
import matplotlib.pyplot as plt
from rich.table import Table
from rich.console import Console
from graphplot.circular import plot_connectivity_circle
from graphplot.colorbar import ColorBar
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
    """Cluster cmp MS-mode values and compute metabolic profile deltas per node."""
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
    required=False,
    help="Path to the component NPZ produced by nbs_groups.py.",
    default="/home/flucchetti/Connectome/Dev/mrsitoolbox/results/controls_vs_patients/nbs/LPN-Project/connectome_plots/group/group-LPN-Project_parc-LFMIHIFIS_scale-3_diag-controls_perm-freedman_nperm-1024_th-3.75_reg-state_nuis-age-sex_lobes-all_comp-2_results.npz",
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
    help="Number of Gaussian clusters for MS-mode metabolic profiling.",
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
    "--align-compare-msmode",
    action="store_true",
    help="Allow flipping the compare-group MS-mode to better align with controls.",
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

args = parser.parse_args()
base_plot_dir = os.path.split(args.result)[0]
target_pairs = []
show_figures = False

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
        if args.aggregate_deltas:
            cmd.append("--aggregate-deltas")
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
        debug.warning(f"No subjects in group {value} for MS-mode/metab calculations; skipping this group.")
    else:
        debug.info(
            f"Group {value}: MeSiM shape {group_splits[value]['mesim'].shape}, "
            f"metab_profiles shape {group_splits[value]['metab_profiles'].shape}"
        )
if target_idx is not None and target_reg_value in group_splits:
    if target_reg_value == control_group_value:
        debug.info("Target pair belongs to control group; keeping control group averaged.")
    else:
        # override compare group with only the target subject/session for MS-mode and metab profiles
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


########################## MS-mode comparison between groups ##########################
if comp_mask is None:
    raise ValueError("Component mask missing from NPZ; cannot extract NBS nodes.")

mask_bool = np.asarray(comp_mask, dtype=bool)
significant_nodes = np.where(mask_bool.sum(axis=0) > 0)[0]
if significant_nodes.size == 0:
    debug.warning("No nodes associated with the NBS component; skipping edge plotting.")

msmode_by_group: dict[float, np.ndarray] = {}
for value in unique_groups:
    group_mesim = group_splits[value]["mesim"]
    if group_mesim.size == 0:
        debug.warning(f"No subjects in group {value} for MS-mode calculation; skipping.")
        continue
    avg_mesim = np.nanmean(group_mesim, axis=0)
    msmode = nettools.dimreduce_matrix(
        avg_mesim,
        method="diffusion",
        output_dim=1 ,
        perplexity=30,
        scale_factor=-1.0,
    )
    msmode_by_group[value] = msmode
    debug.warning("msmode",value,"----",msmode.min(),"-->",msmode.max())

if len(msmode_by_group) >= 2:
    group_order = [g for g in unique_groups if g in msmode_by_group]
    aligned = nettools.align_gradients_procrustes(
        [msmode_by_group[g] for g in group_order],
        reference=msmode_by_group[group_order[0]],
    )
    for g, aligned_vec in zip(group_order, aligned):
        msmode_by_group[g] = aligned_vec
    debug.info("Aligned MS-mode gradients across groups using Procrustes.")



if unique_groups.size >= 2 and len(msmode_by_group) >= 2 and significant_nodes.size > 0:
    ref = unique_groups[0]
    compare = unique_groups[1]
    if ref not in msmode_by_group or compare not in msmode_by_group:
        ref_values_raw = np.array([])
        cmp_values_raw = np.array([])
        debug.warning("One of the groups lacks MS-mode data; skipping MS-mode delta computation.")
    else:
        ref_values_raw = msmode_by_group[ref][significant_nodes]
        cmp_values_raw = msmode_by_group[compare][significant_nodes]
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
        f"Delta MS-mode ({compare} - {ref}) on {len(merged_labels)} merged nodes: "
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
    ax.plot(ref_values, marker="o", linestyle="-", color="tab:green", label="controls")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("MS-mode", fontsize=FONTSIZE)
    ax.set_xlabel("NBS Nodes", fontsize=FONTSIZE)
    ax.legend(loc="best")
    ax.set_xticks(range(len(merged_labels)))
    ax.set_xticklabels(tick_labels, fontsize=FONTSIZE - 6, rotation=90, va="top")
    ax.tick_params(axis="y", labelsize=FONTSIZE - 6)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(6))
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    table = Table(title=f"MS-mode Δ ({compare} - {ref})")
    table.add_column("Node Label")
    table.add_column("Δ MS-mode", justify="right")
    for label, value in zip(merged_labels, delta):
        table.add_row(label, f"{value:.3f}")
    # console.print(table)
    if npz_path:
        base_name = os.path.splitext(os.path.basename(npz_path))[0]
        data_path = os.path.join(plot_dir, f"{base_name}_msmode_values.tsv")
        df_values = pd.DataFrame({
            "node_label": merged_labels,
            f"MSmode_group_{compare}": cmp_values,
            f"MSmode_group_{ref}": ref_values,
            "delta": delta,
        })
        df_values.to_csv(data_path, sep="\t", index=False)
        debug.success("Saved MS-mode table to", data_path)
    if npz_path:
        base_name = os.path.splitext(os.path.basename(npz_path))[0]
        plot_path_linear = os.path.join(plot_dir, f"{base_name}_msmode_profile.pdf")
        plt.savefig(plot_path_linear, bbox_inches="tight", dpi=300)
        debug.success("Saved MS-mode comparison plot to", plot_path_linear)
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
        df_cluster["delta_ms_mode"] = [
            delta_lookup.get(re.sub(r"_(\d+)$", "", name), np.nan)
            for name in node_labels_full
        ]
        ref_lookup = {label: val for label, val in zip(merged_labels, ref_values)}
        cmp_lookup = {label: val for label, val in zip(merged_labels, cmp_values)}
        df_cluster["ms_mode_ref"] = [
            ref_lookup.get(re.sub(r"_(\d+)$", "", name), np.nan) for name in node_labels_full
        ]
        df_cluster["ms_mode_cmp"] = [
            cmp_lookup.get(re.sub(r"_(\d+)$", "", name), np.nan) for name in node_labels_full
        ]
        table_clusters = Table(title=f"Metabolic Profile Δ per Node (n_clusters={args.nclusters})")
        table_clusters.add_column("Node Label")
        table_clusters.add_column("Cluster", justify="right")
        table_clusters.add_column("Δ MS-mode", justify="right")
        table_clusters.add_column("MS-mode (ref)", justify="right")
        table_clusters.add_column("MS-mode (cmp)", justify="right")
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
                _format_delta(row["delta_ms_mode"]),
                f"{row['ms_mode_ref']:.4f}",
                f"{row['ms_mode_cmp']:.4f}",
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
        debug.warning("No significant nodes; skipping MS-mode deltas.")
    else:
        debug.warning("Less than two groups present; cannot compute MS-mode deltas.")

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
    node_values_plot = msmode_by_group[plot_group]
    node_values_order = msmode_by_group[order_group]
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
    name_lookup = {name.lower(): name for name in collapsed_names}
    for display, full in zip(collapsed_names_display, collapsed_names):
        name_lookup.setdefault(display.lower(), full)
    angle_lookup = {name: angle for name, angle in zip(collapsed_names, node_angles_rad)}
    node_label_fontsize = 16
    fig_circle, ax_circle = plot_connectivity_circle(
        mat,
        collapsed_names_display,
        node_colors=node_colors,
        node_colors_ref=node_colors_ref,
        node_angles=node_angles_deg,
        colormap="PiYG",
        vmin=-edge_abs_max,
        vmax=edge_abs_max,
        linewidth=1,
        edge_weights=edge_weight_matrix,
        # title=f"T-matrix masked (component {component_idx})",
        fontsize_title=FONTSIZE - 4,
        fontsize_names=node_label_fontsize,
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
    if similarity_node_set and has_edges:
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
    if has_edges:
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
    cb_nodes.set_label("MS Mode", fontsize=FONTSIZE, color=label_color)
    cb_nodes.ax.yaxis.set_tick_params(labelsize=FONTSIZE - 4, color=label_color)
    plt.setp(plt.getp(cb_nodes.ax.axes, "yticklabels"), color=label_color)
    if npz_path:
        base_name = os.path.splitext(os.path.basename(npz_path))[0]
        plot_path_circle = os.path.join(plot_dir, f"{base_name}_tmask_circle.pdf")
        fig_circle.savefig(plot_path_circle, bbox_inches="tight", dpi=300)
        debug.success("Saved circular plot to", plot_path_circle)
    show_figures = True
else:
    debug.warning("Missing t-matrix or component mask; skipping connectivity circle plot.")

if show_figures and not args.no_show:
    plt.show()
