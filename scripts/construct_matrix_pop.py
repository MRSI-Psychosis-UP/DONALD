import numpy as np
from os.path import join, split, exists
import os
import re
import zipfile
import nibabel as nib
import copy
import pandas as pd
from nilearn.image import resample_to_img
from rich.progress import track
from pathlib import Path



from mrsitoolbox.connectomics.nettools import NetTools
from mrsitoolbox.connectomics.parcellate import Parcellate
from mrsitoolbox.connectomics.mesim import MeSiM
from mrsitoolbox.tools.datautils import DataUtils
from mrsitoolbox.tools.debug import Debug
from mrsitoolbox.tools.filetools import FileTools

dutils    = DataUtils()
debug     = Debug()
parc      = Parcellate()
nettools  = NetTools()
mesim     = MeSiM()




def filter_sparse_matrices(matrix_list,sigma=1):
    n_zeros_arr = list()
    for i,sim in enumerate(matrix_list):
        n_zeros = len(np.where(sim==0)[0])
        n_zeros_arr.append(n_zeros)

    n_zeros_arr = np.array(n_zeros_arr)
    debug.info("0 nodal strength count",n_zeros_arr.mean(),"+-",n_zeros_arr.std())

    if n_zeros_arr.mean()<0.001:
        return matrix_list, [], []
    include_indices = list()
    exclude_indices = list()
    matrix_list_refined = list()

    for i,sim in enumerate(matrix_list):
        n_zeros = len(np.where(sim==0)[0])
        if n_zeros<n_zeros_arr.mean()+sigma*n_zeros_arr.std():
            matrix_list_refined.append(sim)
            include_indices.append(i)
        else:
            exclude_indices.append(i)
    return matrix_list_refined,include_indices,exclude_indices

def extract_group_from_inputs(covar_df, con_path_list):
    if covar_df is not None and hasattr(covar_df, "columns"):
        col_map = {str(col).lower(): col for col in covar_df.columns}
        group_col = col_map.get("group")
        if group_col is not None:
            values = [str(v).strip() for v in covar_df[group_col].dropna().tolist() if str(v).strip()]
            if values:
                return values[0]

    if con_path_list:
        parts = list(Path(con_path_list[0]).parts)
        if "derivatives" in parts:
            idx = parts.index("derivatives")
            if idx > 0:
                return parts[idx - 1]
    return "group"


def extract_npert_from_shape(parcel_concentrations, metabolite_list, n_parcels):
    if parcel_concentrations is None:
        return 1
    arr = np.asarray(parcel_concentrations)
    n_metabolites = len(metabolite_list or [])
    candidate_dims = []
    for dim in arr.shape:
        if dim in (0, 1):
            continue
        if n_parcels and dim == int(n_parcels):
            continue
        if n_metabolites and dim == int(n_metabolites):
            continue
        candidate_dims.append(int(dim))
    return candidate_dims[-1] if candidate_dims else 1


def align_covar_df_to_subject_order(covar_df, subject_id_list, session_id_list):
    if covar_df is None:
        return None
    col_map = {str(col).lower(): col for col in covar_df.columns}
    sub_col = (
        col_map.get("participant_id")
        or col_map.get("subject_id")
        or col_map.get("subject")
        or col_map.get("sub")
        or col_map.get("id")
    )
    ses_col = col_map.get("session_id") or col_map.get("session") or col_map.get("ses")
    if sub_col is None or ses_col is None:
        raise ValueError(
            "covar_df must contain participant_id/subject_id and session_id/session columns."
        )

    def normalize_subject(value):
        token = str(value).strip()
        return token[4:] if token.lower().startswith("sub-") else token

    def normalize_session(value):
        token = str(value).strip()
        if token.lower().startswith("ses-"):
            token = token[4:]
        upper = token.upper()
        if upper.startswith("T") and len(upper) > 1:
            return f"V{upper[1:]}"
        if token.isdigit():
            return f"V{token}"
        return token

    lookup = {}
    for _, row in covar_df.iterrows():
        pair = (normalize_subject(row[sub_col]), normalize_session(row[ses_col]))
        if pair not in lookup:
            lookup[pair] = row.to_dict()

    aligned_rows = []
    missing_pairs = []
    for subject_id, session_id in zip(subject_id_list, session_id_list):
        pair = (normalize_subject(subject_id), normalize_session(session_id))
        row = lookup.get(pair)
        if row is None:
            missing_pairs.append(pair)
            continue
        aligned_rows.append(row)

    if missing_pairs:
        raise ValueError(f"Missing covariate rows for subject/session pairs: {missing_pairs[:5]}")
    return pd.DataFrame(aligned_rows, columns=covar_df.columns)


def _parcel_label_key(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode()
        except Exception:
            value = str(value)
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return ""
        try:
            numeric = float(text)
        except Exception:
            return text
        if np.isfinite(numeric) and numeric.is_integer():
            return int(numeric)
        return numeric
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if np.isfinite(numeric) and numeric.is_integer():
            return int(numeric)
        return numeric
    return value


def _coerce_parcel_names(parcel_names, n_labels):
    if parcel_names is None:
        return np.array([""] * int(n_labels), dtype=object)
    arr = np.asarray(parcel_names).reshape(-1)
    if len(arr) == int(n_labels):
        return arr.astype(object, copy=False)
    out = np.array([""] * int(n_labels), dtype=object)
    limit = min(len(arr), int(n_labels))
    if limit > 0:
        out[:limit] = arr[:limit]
    return out


def _coerce_parcel_labels(parcel_labels, fallback_size=0):
    if parcel_labels is None:
        return np.arange(int(fallback_size))
    arr = np.asarray(parcel_labels)
    if arr.ndim == 0:
        if fallback_size:
            return np.arange(int(fallback_size))
        return np.asarray([arr.item()])
    return arr.reshape(-1)


def _coerce_label_list(labels):
    if labels is None:
        return []
    if isinstance(labels, np.ndarray):
        return labels.reshape(-1).tolist()
    if isinstance(labels, (list, tuple, set)):
        return list(labels)
    return [labels]


def _labels_between_bounds(parcel_labels, start_label, end_label=None, prefix="template"):
    labels_arr = _coerce_parcel_labels(parcel_labels, fallback_size=0)
    if labels_arr.size == 0:
        raise ValueError(f"{prefix} has no parcel labels to define a template range.")
    start_key = _parcel_label_key(start_label)
    end_value = start_label if end_label is None else end_label
    end_key = _parcel_label_key(end_value)
    label_keys = [_parcel_label_key(label) for label in labels_arr]

    start_matches = [idx for idx, key in enumerate(label_keys) if key == start_key]
    end_matches = [idx for idx, key in enumerate(label_keys) if key == end_key]
    if not start_matches:
        raise ValueError(f"{prefix} start parcel label {start_label!r} was not found.")
    if not end_matches:
        raise ValueError(f"{prefix} end parcel label {end_value!r} was not found.")
    if len(start_matches) > 1:
        raise ValueError(f"{prefix} start parcel label {start_label!r} is not unique.")
    if len(end_matches) > 1:
        raise ValueError(f"{prefix} end parcel label {end_value!r} is not unique.")

    start_idx = start_matches[0]
    end_idx = end_matches[0]
    if start_idx > end_idx:
        raise ValueError(
            f"{prefix} invalid parcel range: start label {start_label!r} appears after end label {end_value!r}."
        )
    keep_idx = np.arange(start_idx, end_idx + 1, dtype=int)
    return labels_arr[keep_idx]


def _retain_parcel_labels(matrix, profile, parcel_labels, parcel_names, retain_labels, prefix):
    matrix = np.asarray(matrix)
    labels_arr = _coerce_parcel_labels(parcel_labels, fallback_size=matrix.shape[0] if matrix.ndim >= 1 else 0)
    retain_list = _coerce_label_list(retain_labels)
    if len(retain_list) == 0:
        return matrix, profile, labels_arr, _coerce_parcel_names(parcel_names, len(labels_arr))

    names_arr = _coerce_parcel_names(parcel_names, len(labels_arr))
    retain_keys = [_parcel_label_key(label) for label in retain_list]
    label_to_index = {}
    label_to_name = {}
    for idx, label in enumerate(labels_arr):
        key = _parcel_label_key(label)
        if key in label_to_index:
            continue
        label_to_index[key] = idx
        label_to_name[key] = names_arr[idx] if idx < len(names_arr) else ""

    retained_size = len(retain_list)
    matrix_out = np.zeros((retained_size, retained_size), dtype=matrix.dtype)
    retained_names = np.array([label_to_name.get(key, "") for key in retain_keys], dtype=object)
    mapped_ref = []
    mapped_src = []
    for ref_idx, key in enumerate(retain_keys):
        src_idx = label_to_index.get(key)
        if src_idx is None:
            continue
        mapped_ref.append(ref_idx)
        mapped_src.append(src_idx)

    if mapped_ref:
        ref_pos = np.array(mapped_ref, dtype=int)
        src_pos = np.array(mapped_src, dtype=int)
        matrix_out[np.ix_(ref_pos, ref_pos)] = matrix[np.ix_(src_pos, src_pos)]
    else:
        debug.warning(f"{prefix} none of the requested retained parcel labels were found; zero-filling matrix.")

    profile_out = profile
    if profile is not None:
        profile_arr = np.asarray(profile)
        profile_out = profile_arr
        if profile_arr.ndim >= 1:
            if profile_arr.shape[0] == len(labels_arr):
                profile_out = np.zeros((retained_size,) + tuple(profile_arr.shape[1:]), dtype=profile_arr.dtype)
                if mapped_ref:
                    profile_out[np.array(mapped_ref, dtype=int)] = profile_arr[np.array(mapped_src, dtype=int)]
            elif profile_arr.ndim >= 2 and profile_arr.shape[1] == len(labels_arr):
                profile_out = np.zeros(
                    (profile_arr.shape[0], retained_size) + tuple(profile_arr.shape[2:]),
                    dtype=profile_arr.dtype,
                )
                if mapped_ref:
                    profile_out[:, np.array(mapped_ref, dtype=int), ...] = profile_arr[:, np.array(mapped_src, dtype=int), ...]
            elif profile_arr.ndim >= 3 and profile_arr.shape[2] == len(labels_arr):
                profile_out = np.zeros(
                    tuple(profile_arr.shape[:2]) + (retained_size,) + tuple(profile_arr.shape[3:]),
                    dtype=profile_arr.dtype,
                )
                if mapped_ref:
                    profile_out[:, :, np.array(mapped_ref, dtype=int), ...] = profile_arr[:, :, np.array(mapped_src, dtype=int), ...]
            else:
                debug.warning(
                    f"{prefix} could not retain parcel labels for profile with shape {profile_arr.shape}; "
                    "parcel axis did not match the label vector."
                )

    missing_count = retained_size - len(mapped_ref)
    if missing_count > 0:
        debug.warning(f"{prefix} retained {len(mapped_ref)}/{retained_size} parcel labels; missing {missing_count}.")

    return matrix_out, profile_out, np.asarray(retain_list), retained_names


def _compact_matrix_to_parcel_labels(matrix, profile, parcel_labels, parcel_names, prefix):
    matrix_arr = np.asarray(matrix)
    labels_arr = _coerce_parcel_labels(parcel_labels, fallback_size=matrix_arr.shape[0] if matrix_arr.ndim >= 1 else 0)
    names_arr = _coerce_parcel_names(parcel_names, len(labels_arr))

    if matrix_arr.ndim < 2:
        return matrix_arr, profile, labels_arr, names_arr
    if matrix_arr.shape[0] == len(labels_arr) and matrix_arr.shape[1] == len(labels_arr):
        return matrix_arr, profile, labels_arr, names_arr
    if matrix_arr.shape[0] != matrix_arr.shape[1]:
        raise ValueError(f"{prefix} matrix is not square: shape {matrix_arr.shape}")
    if len(labels_arr) == 0:
        raise ValueError(f"{prefix} has no parcel labels to align with matrix shape {matrix_arr.shape}.")

    label_keys = [_parcel_label_key(label) for label in labels_arr]
    if not all(isinstance(key, (int, np.integer)) for key in label_keys):
        raise ValueError(
            f"{prefix} cannot compact matrix shape {matrix_arr.shape} because parcel labels are not integer-coded."
        )

    label_ids = np.asarray(label_keys, dtype=int)
    if len(np.unique(label_ids)) != len(label_ids):
        raise ValueError(f"{prefix} cannot compact matrix because parcel labels are not unique.")

    dense_size = int(matrix_arr.shape[0])
    indices_one_based = label_ids - 1
    indices_zero_based = label_ids
    valid_one_based = (indices_one_based >= 0) & (indices_one_based < dense_size)
    valid_zero_based = (indices_zero_based >= 0) & (indices_zero_based < dense_size)

    count_one_based = int(np.sum(valid_one_based))
    count_zero_based = int(np.sum(valid_zero_based))
    prefer_one_based = bool(np.min(label_ids) >= 1)
    if count_one_based > count_zero_based or (count_one_based == count_zero_based and prefer_one_based):
        dense_indices = indices_one_based
        valid_mask = valid_one_based
    else:
        dense_indices = indices_zero_based
        valid_mask = valid_zero_based
    mapped_count = int(np.sum(valid_mask))
    if mapped_count <= 0:
        raise ValueError(
            f"{prefix} cannot map {len(label_ids)} parcel labels onto matrix shape {matrix_arr.shape}."
        )

    retained_size = len(label_ids)
    matrix_out = np.zeros((retained_size, retained_size), dtype=matrix_arr.dtype)
    if mapped_count < retained_size:
        debug.warning(
            f"{prefix} mapped {mapped_count}/{retained_size} labels onto dense matrix shape {matrix_arr.shape}; "
            "missing labels were zero-filled."
        )
    ref_indices = np.where(valid_mask)[0].astype(int)
    src_dense_indices = np.asarray(dense_indices[valid_mask], dtype=int)
    matrix_out[np.ix_(ref_indices, ref_indices)] = matrix_arr[np.ix_(src_dense_indices, src_dense_indices)]

    profile_out = profile
    if profile is not None:
        profile_arr = np.asarray(profile)
        if profile_arr.ndim >= 1:
            if profile_arr.shape[0] == matrix_arr.shape[0]:
                profile_out = np.zeros((retained_size,) + tuple(profile_arr.shape[1:]), dtype=profile_arr.dtype)
                profile_out[ref_indices, ...] = profile_arr[src_dense_indices, ...]
            elif profile_arr.ndim >= 2 and profile_arr.shape[1] == matrix_arr.shape[0]:
                profile_out = np.zeros(
                    (profile_arr.shape[0], retained_size) + tuple(profile_arr.shape[2:]),
                    dtype=profile_arr.dtype,
                )
                profile_out[:, ref_indices, ...] = profile_arr[:, src_dense_indices, ...]
            elif profile_arr.ndim >= 3 and profile_arr.shape[2] == matrix_arr.shape[0]:
                profile_out = np.zeros(
                    tuple(profile_arr.shape[:2]) + (retained_size,) + tuple(profile_arr.shape[3:]),
                    dtype=profile_arr.dtype,
                )
                profile_out[:, :, ref_indices, ...] = profile_arr[:, :, src_dense_indices, ...]
            elif profile_arr.shape[0] == len(labels_arr):
                profile_out = profile_arr
            else:
                debug.warning(
                    f"{prefix} could not compact profile with shape {profile_arr.shape}; "
                    "parcel axis did not match either the dense matrix or the compact label vector."
                )
                profile_out = profile_arr

    debug.info(
        f"{prefix} compacted matrix from dense label-id space {matrix_arr.shape} to compact parcel space {matrix_out.shape}."
    )
    return matrix_out, profile_out, labels_arr, names_arr


def _crop_parcel_label_space(
    matrix,
    profile,
    parcel_labels,
    parcel_names,
    include_parcels,
    include_parcels_start,
    include_parcels_end,
    prefix,
):
    matrix_arr = np.asarray(matrix)
    labels_arr = _coerce_parcel_labels(parcel_labels, fallback_size=matrix_arr.shape[0] if matrix_arr.ndim >= 1 else 0)
    names_arr = _coerce_parcel_names(parcel_names, len(labels_arr))
    if not include_parcels:
        return matrix_arr, profile, labels_arr, names_arr

    if matrix_arr.ndim < 2 or matrix_arr.shape[0] != len(labels_arr) or matrix_arr.shape[1] != len(labels_arr):
        raise ValueError(
            f"{prefix} cannot crop parcel labels because matrix shape {matrix_arr.shape} "
            f"does not match {len(labels_arr)} labels."
        )

    start_key = _parcel_label_key(include_parcels_start)
    end_value = include_parcels_start if include_parcels_end is None else include_parcels_end
    end_key = _parcel_label_key(end_value)

    label_keys = [_parcel_label_key(label) for label in labels_arr]
    start_matches = [idx for idx, key in enumerate(label_keys) if key == start_key]
    end_matches = [idx for idx, key in enumerate(label_keys) if key == end_key]
    if not start_matches:
        raise ValueError(f"{prefix} start parcel label {include_parcels_start!r} was not found.")
    if not end_matches:
        raise ValueError(f"{prefix} end parcel label {end_value!r} was not found.")
    if len(start_matches) > 1:
        raise ValueError(f"{prefix} start parcel label {include_parcels_start!r} is not unique.")
    if len(end_matches) > 1:
        raise ValueError(f"{prefix} end parcel label {end_value!r} is not unique.")

    start_idx = start_matches[0]
    end_idx = end_matches[0]
    if start_idx > end_idx:
        raise ValueError(
            f"{prefix} invalid parcel range: start label {include_parcels_start!r} "
            f"appears after end label {end_value!r}."
        )

    keep_idx = np.arange(start_idx, end_idx + 1, dtype=int)
    matrix_out = matrix_arr[np.ix_(keep_idx, keep_idx)]

    profile_out = profile
    if profile is not None:
        profile_arr = np.asarray(profile)
        if profile_arr.ndim >= 1:
            if profile_arr.shape[0] == len(labels_arr):
                profile_out = profile_arr[keep_idx, ...]
            elif profile_arr.ndim >= 2 and profile_arr.shape[1] == len(labels_arr):
                profile_out = profile_arr[:, keep_idx, ...]
            elif profile_arr.ndim >= 3 and profile_arr.shape[2] == len(labels_arr):
                profile_out = profile_arr[:, :, keep_idx, ...]
            else:
                debug.warning(
                    f"{prefix} could not crop profile with shape {profile_arr.shape}; "
                    "parcel axis did not match the label vector."
                )
                profile_out = profile_arr

    return matrix_out, profile_out, labels_arr[keep_idx], names_arr[keep_idx]


def _normalize_profile_parcel_axis_first(profile, n_labels, prefix, modality):
    if profile is None:
        return None
    profile_arr = np.asarray(profile)
    if profile_arr.ndim == 0 or n_labels <= 0:
        return profile_arr

    if profile_arr.shape[0] == n_labels:
        return profile_arr
    if profile_arr.ndim >= 2 and profile_arr.shape[1] == n_labels:
        profile_arr = np.moveaxis(profile_arr, 1, 0)
        debug.info(f"{prefix} moved {modality} profile parcel axis from 1 to 0: {profile_arr.shape}")
        return profile_arr
    if profile_arr.ndim >= 3 and profile_arr.shape[2] == n_labels:
        profile_arr = np.moveaxis(profile_arr, 2, 0)
        debug.info(f"{prefix} moved {modality} profile parcel axis from 2 to 0: {profile_arr.shape}")
        return profile_arr
    debug.warning(
        f"{prefix} could not normalize {modality} profile with shape {profile_arr.shape}; "
        f"no axis matched {n_labels} parcel labels."
    )
    return profile_arr


def _pad_profiles_to_common_tail(profiles, prefix, modality):
    arrays = [np.asarray(profile) for profile in profiles if profile is not None]
    if not arrays:
        return profiles

    max_ndim = max(arr.ndim for arr in arrays)
    if max_ndim < 2:
        return [np.asarray(profile) if profile is not None else profile for profile in profiles]

    max_tail = []
    for axis in range(1, max_ndim):
        axis_max = 1
        for arr in arrays:
            if arr.ndim > axis:
                axis_max = max(axis_max, int(arr.shape[axis]))
        max_tail.append(axis_max)
    max_tail = tuple(max_tail)

    padded = []
    changed = False
    for profile in profiles:
        if profile is None:
            padded.append(None)
            continue
        arr = np.asarray(profile)
        target_shape = (arr.shape[0],) + max_tail
        if tuple(arr.shape[1:]) == max_tail:
            padded.append(arr)
            continue
        out = np.zeros(target_shape, dtype=arr.dtype)
        slices = [slice(None)]
        for axis in range(1, arr.ndim):
            slices.append(slice(0, arr.shape[axis]))
        out[tuple(slices)] = arr
        padded.append(out)
        changed = True
    if changed:
        debug.info(f"{prefix} padded {modality} profiles to common trailing shape {max_tail}.")
    return padded


def _load_connectivity_archive(con_path, allow_pickle=True):
    con_path = str(con_path)
    if not exists(con_path):
        raise ValueError(f"Connectivity archive not found: {con_path}")
    try:
        file_size = os.path.getsize(con_path)
    except OSError as exc:
        raise ValueError(f"Unable to access connectivity archive: {exc}") from exc
    if file_size <= 0:
        raise ValueError("Connectivity archive is empty (0 bytes).")
    if str(con_path).lower().endswith(".npz") and not zipfile.is_zipfile(con_path):
        raise ValueError("Connectivity archive is not a valid .npz zip file.")
    try:
        return np.load(con_path, allow_pickle=allow_pickle)
    except Exception as exc:
        raise ValueError(f"Failed to load connectivity archive: {exc}") from exc


def _default_matrix_key_for_modality(modality):
    modality_token = str(modality or "").strip().lower()
    if modality_token == "dwi":
        return "connectome_density"
    if modality_token == "func":
        return "connectivity"
    return "simmatrix_sp"


def _collapse_square_matrix_entry(matrix, key, prefix):
    arr = np.asarray(matrix)
    if arr.ndim == 2:
        if arr.shape[0] != arr.shape[1]:
            raise ValueError(f"{prefix} matrix key '{key}' has non-square shape {arr.shape}.")
        return arr
    if arr.ndim == 3:
        if arr.shape[1] == arr.shape[2]:
            return np.median(arr, axis=0)
        if arr.shape[0] == arr.shape[1]:
            return np.median(arr, axis=2)
    raise ValueError(
        f"{prefix} matrix key '{key}' must have shape (N,N), (K,N,N), or (N,N,K); got {arr.shape}."
    )

def main(con_path_list,covar_df,modality,matrix_outpath,
         matrix_key=None,
         ignore_parc_list=[],qmask_path=None,mrsi_cov=0.67,
         comp_gradients=True,parcellation_img_path=None,
         parcel_labels_retain=[],include_parcels=False,
         include_parcels_start=0,include_parcels_end=None):
    if not con_path_list:
        return "No connectivity paths were provided."

    group = extract_group_from_inputs(covar_df, con_path_list)
    modality = str(modality or "").strip().lower()
    matrix_key = str(matrix_key or "").strip() or None
    debug.title(f"Compute population Matrix for {group}")

    matrix_subjects_list = []
    node_signals_subjects = []
    subject_id_list_found = []
    session_list_found = []
    parcel_labels_list = []
    parcel_names_list = []
    metabolite_list = []
    npert = 1
    filename = split(matrix_outpath)[1]
    resultssubdir = split(matrix_outpath)[0]
    retained_labels = _coerce_label_list(parcel_labels_retain)
    skipped_archives = []
    selected_matrix_key = matrix_key or _default_matrix_key_for_modality(modality)

    for con_path in track(con_path_list, total=len(con_path_list), description="Extracting matrix..."):
        con_path = str(con_path)
        meta = FileTools.extract_metadata(con_path)
        subject_id = str(meta.get("sub") or "").strip() or Path(con_path).stem
        session = str(meta.get("ses") or "").strip() or "NA"
        prefix = f"sub-{subject_id}_ses-{session}"
        con_data = None
        try:
            if modality == "dwi":
                con_data = _load_connectivity_archive(con_path, allow_pickle=True)
                if selected_matrix_key not in con_data:
                    raise ValueError(f"{prefix} matrix key '{selected_matrix_key}' is not present.")
                matrix = _collapse_square_matrix_entry(con_data[selected_matrix_key], selected_matrix_key, prefix)
                profile = np.zeros((matrix.shape[0], 1, 1))
                parcel_labels = np.asarray(con_data.get("parcel_labels"))
                parcel_names = np.asarray(con_data.get("parcel_names"))
            elif modality == "func":
                con_data = _load_connectivity_archive(con_path, allow_pickle=True)
                if selected_matrix_key not in con_data:
                    raise ValueError(f"{prefix} matrix key '{selected_matrix_key}' is not present.")
                matrix = _collapse_square_matrix_entry(con_data[selected_matrix_key], selected_matrix_key, prefix)
                profile = con_data["timeseries"] if "timeseries" in con_data else np.zeros((matrix.shape[0], 1))
                parcel_labels = np.asarray(con_data.get("labels_indices"))
                parcel_names = np.asarray(con_data.get("labels"))
            else:
                con_data = _load_connectivity_archive(con_path, allow_pickle=True)
                if selected_matrix_key not in con_data:
                    raise ValueError(f"{prefix} matrix key '{selected_matrix_key}' is not present.")
                matrix_entry = np.asarray(con_data[selected_matrix_key])
                if selected_matrix_key == "simmatrix_sp":
                    crlb_matrix = np.var(matrix_entry, axis=0) * 100 if len(matrix_entry.shape) == 3 else np.zeros_like(matrix_entry)
                    matrix = np.median(matrix_entry, axis=0) if len(matrix_entry.shape) == 3 else matrix_entry
                    matrix[crlb_matrix > 20] = 0
                else:
                    matrix = _collapse_square_matrix_entry(matrix_entry, selected_matrix_key, prefix)
                parcel_concentrations = np.asarray(con_data["parcel_concentrations"])
                profile = parcel_concentrations
                if "metabolites_leaveout" in con_data:
                    metabolites_key = "metabolites_leaveout"
                elif "metabolites" in con_data:
                    metabolites_key = "metabolites"
                else:
                    raise ValueError(
                        f"{prefix} has neither 'metabolites_leaveout' nor 'metabolites' key."
                    )
                metabolite_list = list(np.asarray(con_data[metabolites_key]).reshape(-1))
                parcel_labels = np.asarray(con_data.get("labels_indices"))
                parcel_names = np.asarray(
                    con_data["labels"] if "labels" in con_data else con_data.get("parcel_names")
                )
                npert = max(
                    npert,
                    extract_npert_from_shape(
                        parcel_concentrations,
                        metabolite_list,
                        matrix.shape[0],
                    ),
                )

            matrix, profile, parcel_labels, parcel_names = _compact_matrix_to_parcel_labels(
                matrix,
                profile,
                parcel_labels,
                parcel_names,
                prefix,
            )
            if include_parcels and not retained_labels:
                retained_labels = _labels_between_bounds(
                    parcel_labels,
                    include_parcels_start,
                    include_parcels_end,
                    prefix=f"{prefix} template",
                ).tolist()
                debug.info(
                    f"{prefix} using template range [{include_parcels_start}.."
                    f"{include_parcels_end if include_parcels_end is not None else include_parcels_start}] "
                    f"with {len(retained_labels)} labels."
                )
            matrix, profile, parcel_labels, parcel_names = _retain_parcel_labels(
                matrix,
                profile,
                parcel_labels,
                parcel_names,
                retained_labels,
                prefix,
            )
            # Crop matrix, node signal and label space
            matrix, profile, parcel_labels, parcel_names = _crop_parcel_label_space(
                matrix,
                profile,
                parcel_labels,
                parcel_names,
                include_parcels,
                include_parcels_start,
                include_parcels_end,
                prefix,
            )
            if modality == "func":
                profile = _normalize_profile_parcel_axis_first(
                    profile,
                    len(parcel_labels),
                    prefix,
                    modality,
                )
            matrix_subjects_list.append(np.asarray(matrix))
            node_signals_subjects.append(np.asarray(profile))
            parcel_labels_list.append(np.asarray(parcel_labels))
            parcel_names_list.append(np.asarray(parcel_names))
            subject_id_list_found.append(subject_id)
            session_list_found.append(session)
        except ValueError as exc:
            skipped_archives.append((prefix, con_path, str(exc)))
            debug.warning(prefix, exc)
        except Exception:
            import traceback
            debug.error(prefix, traceback.format_exc())
        finally:
            if con_data is not None and hasattr(con_data, "close"):
                con_data.close()

    if len(matrix_subjects_list) == 0:
        message = "No connectivity matrices loaded; aborting."
        debug.error(message)
        return message

    if skipped_archives:
        debug.warning(f"Skipped {len(skipped_archives)} invalid connectivity archive(s).")

    if modality != "mrsi":
        npert = 1


    ##### Match matrix rows/cols along their parcel lists #####
    matrix_subjects_list, parcel_labels_group, parcel_names_group = mesim.create_pop_matrix(
        matrix_subjects_list, parcel_labels_list, parcel_names_list
    )

    if modality == "func":
        node_signals_subjects = _pad_profiles_to_common_tail(
            node_signals_subjects,
            prefix=f"{group}:{modality}",
            modality=modality,
        )

    aligned_node_signals = mesim.create_pop_profiles(
        node_signals_subjects,
        parcel_labels_list,
        parcel_labels_group,
        default_tail=(max(1, npert), max(1, len(metabolite_list))) if modality == "mrsi" else None,
        subject_ids=subject_id_list_found,
        session_ids=session_list_found,
    )

    matrix_subjects_list = np.array(matrix_subjects_list)
    node_signals_subjects = np.array(aligned_node_signals)
    metabolite_list = metabolite_list

    ##### remove BND and WM labels before further processing #####
    parcel_names_group = np.array(parcel_names_group)
    parcel_labels_group = np.array(parcel_labels_group)
    if len(parcel_names_group) != len(parcel_labels_group):
        debug.warning(f"Parcel names/labels length mismatch ({len(parcel_names_group)} vs {len(parcel_labels_group)}); truncating to min length.")
        min_len = min(len(parcel_names_group), len(parcel_labels_group))
        parcel_names_group = parcel_names_group[:min_len]
        parcel_labels_group = parcel_labels_group[:min_len]
    keep_mask = np.array([name != "BND" for name in parcel_names_group], dtype=bool)
    wm_start = None
    for idx, name in enumerate(parcel_names_group):
        if isinstance(name, str) and "wm-" in name:
            wm_start = idx
            break
    if wm_start is not None:
        keep_mask[wm_start:] = False
    parcel_names_group = np.array(parcel_names_group)[keep_mask]
    parcel_labels_group = np.array(parcel_labels_group)[keep_mask]
    matrix_subjects_list = matrix_subjects_list[:, keep_mask, :][:, :, keep_mask]
    node_signals_subjects = node_signals_subjects[:, keep_mask, ...]

    debug.info(f"Collected {matrix_subjects_list.shape[0]} matrices of shape {matrix_subjects_list.shape[1::]}")

    ########## Clean simmilarity matrices ##########
    # Diascard sparse subjects matrix from average 
    debug.title("Exclude sparse within-subject-wise matrices")
    matrix_list_sel,i,e = filter_sparse_matrices(matrix_subjects_list,sigma=3)
    node_signals_subjects_sel = np.delete(node_signals_subjects,e,axis=0)
    session_id_arr_sel      = np.delete(session_list_found,e,axis=0)
    subject_id_arr_sel      = np.delete(subject_id_list_found,e,axis=0)
    matrix_list_sel          = np.array(matrix_list_sel)
    discarded_subjects      = [subject_id_list_found[idx] for idx in np.array(e)]
    discarded_sessions      = [session_list_found[idx] for idx in np.array(e)]
    debug.info(f"Excluded {len(e)} sparse matrices of shape, remaining {matrix_list_sel.shape[0]}")
    for sub,ses in zip(discarded_subjects,discarded_sessions):
        debug.info(sub,ses,"was left out")

    ####################### Load Parcellation Image ########################################
    parcellation_required = bool(comp_gradients) or (modality == "mrsi" and qmask_path is not None)
    parcellation_img = None
    if parcellation_required:
        if not parcellation_img_path:
            message = "A parcellation image path is required for qmask filtering or gradient computation."
            debug.error(message)
            return message
        if exists(parcellation_img_path):
            parcellation_img = nib.load(parcellation_img_path)
        else:
            message = f"Parcellation file not found: {parcellation_img_path}"
            debug.error(message)
            return message

    ################ Ignore Parcels defined by low MRSI coveraged <-> QMASK ####################
    debug.separator()
    if modality == "mrsi" and qmask_path is not None:
        qmask_pop_img      = nib.load(qmask_path)
        if parcellation_img.shape != qmask_pop_img.shape:
            debug.warning(f"Parcel image shape {parcellation_img.shape} does not match Qmask shape {qmask_pop_img.shape}")
            debug.proc("Resampling...")
            parcellation_img = resample_to_img(
                source_img=parcellation_img,
                target_img=qmask_pop_img,
                interpolation="nearest"
            )
        n_voxel_counts_dict = parc.count_voxels_inside_parcel(qmask_pop_img.get_fdata(), 
                                                              parcellation_img.get_fdata().astype(int), 
                                                              parcel_labels_group)
        ignore_parcel_idx = [index for index in n_voxel_counts_dict if n_voxel_counts_dict[index] < mrsi_cov]
        ignore_rows = [np.where(parcel_labels_group == parcel_idx)[0][0] for parcel_idx in ignore_parcel_idx if len(np.where(parcel_labels_group == parcel_idx)[0]) != 0]
        ignore_rows = np.sort(np.array(ignore_rows)) 
        # Delete nodes in MeSiM 
        parcel_labels_group         = np.delete(parcel_labels_group,ignore_rows)
        parcel_names_group          = np.delete(parcel_names_group,ignore_rows)
        _matrix_list_sel             = np.delete(matrix_list_sel,ignore_rows,axis=1)
        matrix_list_sel              = np.delete(_matrix_list_sel,ignore_rows,axis=2)
        node_signals_subjects_sel = np.delete(node_signals_subjects_sel,ignore_rows,axis=1)
        debug.info(f"Matrix shape {matrix_list_sel.shape[1::]} from {matrix_list_sel.shape[0]} subjects")

        __n_voxel_counts_dict = parc.count_voxels_inside_parcel(qmask_pop_img.get_fdata(), 
                                                              parcellation_img.get_fdata().astype(int), 
                                                              parcel_labels_group,norm=False)
    else:
        n_voxel_counts_dict = {}
        __n_voxel_counts_dict = {}
        debug.warning("Skipping qmask filtering for non-MRSI modality")

    ############# Detect empty correlations from pop AVG  #############
    # if modality=="mrsi":
    debug.title("Remove sparse nodes")
    matrix_pop_avg                 = matrix_list_sel.mean(axis=0)
    # Cleanup empty nodes
    mask_parcel_indices = list(np.where(np.diag(matrix_pop_avg) == 0)[0]) if modality=="mrsi" else []
    if modality in ["mrsi","func"]:
        nonzero_frac = np.mean(~np.isclose(matrix_list_sel, 0, atol=1e-10), axis=(0, 2))  # per node
        mask_parcel_indices.extend(np.where(nonzero_frac < 0.05)[0])  # tune threshold
        mask_parcel_indices = np.unique(np.array(mask_parcel_indices))
    # delete rowd/cols of empty correlations 
    if len(mask_parcel_indices)!=0:
        _matrix_pop_avg_clean          = np.delete(matrix_pop_avg, mask_parcel_indices, axis=0)
        matrix_pop_avg_clean           = np.delete(_matrix_pop_avg_clean, mask_parcel_indices, axis=1)
        node_signals_subjects_clean  = np.delete(node_signals_subjects_sel, mask_parcel_indices, axis=1)
        _tmp_matrix                    = np.delete(matrix_list_sel, mask_parcel_indices, axis=1)
        matrix_subjects_list_clean     = np.delete(_tmp_matrix, mask_parcel_indices, axis=2)
        for i in mask_parcel_indices:
            debug.info("Removed sparse connectivty node",parcel_labels_group[i],parcel_names_group[i],)

        debug.separator()
        # same for parcellation data 
        parcel_labels_group  = np.delete(parcel_labels_group, mask_parcel_indices)
        parcel_names_group   = np.delete(parcel_names_group, mask_parcel_indices)
    else:
        matrix_subjects_list_clean = copy.deepcopy(matrix_list_sel)
        node_signals_subjects_clean = copy.deepcopy(node_signals_subjects_sel)
        matrix_pop_avg_clean = copy.deepcopy(matrix_list_sel.mean(axis=0))
        debug.info("No sparse nodes detected, none removed")

    # Optionally drop parcels matching substrings provided via --ignore_parc_list
    ignore_tokens = [token.lower() for token in ignore_parc_list if isinstance(token, str) and token.strip()]
    if ignore_tokens:
        ignore_indices = []
        for idx, name in enumerate(parcel_names_group):
            name_str = str(name).lower()
            if any(token in name_str for token in ignore_tokens):
                ignore_indices.append(idx)
        ignore_indices = sorted(set(ignore_indices))
        if ignore_indices:
            ignored_names = [parcel_names_group[i] for i in ignore_indices]
            debug.info(f"Discarding parcels via --ignore_parc_list {ignore_tokens}: {ignored_names}")
            parcel_labels_group = np.delete(parcel_labels_group, ignore_indices)
            parcel_names_group = np.delete(parcel_names_group, ignore_indices)
            _matrix_pop_avg_clean = np.delete(matrix_pop_avg_clean, ignore_indices, axis=0)
            matrix_pop_avg_clean = np.delete(_matrix_pop_avg_clean, ignore_indices, axis=1)
            _matrix_subjects_list_clean = np.delete(matrix_subjects_list_clean, ignore_indices, axis=1)
            matrix_subjects_list_clean = np.delete(_matrix_subjects_list_clean, ignore_indices, axis=2)
            node_signals_subjects_clean = np.delete(node_signals_subjects_clean, ignore_indices, axis=1)
        else:
            debug.warning(f"--ignore_parc_list patterns {ignore_tokens} matched no parcel names")

    debug.info(f"Final Matrix shape {matrix_pop_avg_clean.shape}")
    debug.separator()
    ############## Save intermediate simmatrices and node signals
   
    covar_df = align_covar_df_to_subject_order(covar_df, subject_id_arr_sel, session_id_arr_sel)

    os.makedirs(split(matrix_outpath)[0],exist_ok=True)
    save_kwargs = dict(
                    node_signals_subj_list   = node_signals_subjects_clean,
                    matrix_subj_list         = matrix_subjects_list_clean,
                    matrix_pop_avg           = matrix_pop_avg_clean,
                    parcel_labels_group      = parcel_labels_group,
                    parcel_names_group       = parcel_names_group,
                    subject_id_list          = subject_id_arr_sel,
                    session_id_list          = session_id_arr_sel,
                    discarded_subjects       = discarded_subjects,
                    discarded_sessions       = discarded_sessions,
                    metabolites              = metabolite_list,
                    group                    = group,
                    modality                 = modality,
                    source_matrix_key        = np.asarray(str(selected_matrix_key)))
    save_kwargs["metab_profiles_subj_list"] = node_signals_subjects_clean
    if covar_df is not None:
        save_kwargs["covars"] = covar_df.to_records(index=False)
    np.savez(matrix_outpath, **save_kwargs)
    # debug.separator()
    # print(covar_df)
    # debug.separator()
    # print(subject_id_list)

    debug.success("Saved cleaned subject matrices and population averaged matrix to \n",matrix_outpath)

    # Save subject-session list alongside the NPZ
    tsv_out = matrix_outpath.replace(".npz", "_subjects.tsv")
    try:
        with open(tsv_out, "w", newline="") as tsv_file:
            import csv
            writer = csv.writer(tsv_file, delimiter="\t")
            writer.writerow(["participant_id", "session_id"])
            for sid, ses in zip(subject_id_arr_sel, session_id_arr_sel):
                writer.writerow([sid, ses])
        debug.success("Wrote subject-session list to", tsv_out)
    except Exception as e:
        debug.warning("Failed to write subject-session TSV:", e)

    # Compute gradients and alignm them
    if comp_gradients:
        debug.separator()
        debug.title("Compute diffusion gradients (msmode)")
        n_components = 10
        n_subjects = matrix_subjects_list_clean.shape[0]
        n_nodes = matrix_subjects_list_clean.shape[1]

        subject_gradients = np.zeros((n_subjects, n_nodes, n_components), dtype=float)
        for idx, mat in enumerate(
            track(matrix_subjects_list_clean, total=n_subjects, description="Diffusion gradients")
        ):
            for comp in range(1, n_components + 1):
                subject_gradients[idx, :, comp - 1] = nettools.dimreduce_matrix(
                    mat, method="diffusion", output_dim=comp, scale_factor=1
                )

        avg_gradients = np.zeros((n_nodes, n_components), dtype=float)
        for comp in range(1, n_components + 1):
            avg_gradients[:, comp - 1] = nettools.dimreduce_matrix(
                matrix_pop_avg_clean, method="diffusion", output_dim=comp, scale_factor=1
            )

        aligned_gradients = np.zeros_like(subject_gradients)
        for comp in range(n_components):
            grads_comp = [subject_gradients[i, :, comp] for i in range(n_subjects)]
            aligned_comp = nettools.align_gradients_procrustes(
                grads_comp, reference=avg_gradients[:, comp]
            )
            for i, vec in enumerate(aligned_comp):
                aligned_gradients[i, :, comp] = vec

        gradients_outpath = matrix_outpath.replace("connectivity", "gradients")
        os.makedirs(split(gradients_outpath)[0], exist_ok=True)
        gradients_kwargs = dict(
            gradients=aligned_gradients.transpose(0, 2, 1),  # (N, 10, K)
            gradients_avg=avg_gradients.T,  # (10, K)
            node_signals_subj_list=node_signals_subjects_clean,
            parcel_labels_group=parcel_labels_group,
            parcel_names_group=parcel_names_group,
            subject_id_list=subject_id_arr_sel,
            session_id_list=session_id_arr_sel,
            discarded_subjects=discarded_subjects,
            discarded_sessions=discarded_sessions,
            metabolites=metabolite_list,
            group=group,
            modality=modality,
            parc_path=parcellation_img_path,
        )
        gradients_kwargs["metab_profiles_subj_list"] = node_signals_subjects_clean
        if covar_df is not None:
            gradients_kwargs["covars"] = covar_df.to_records(index=False)
        np.savez(gradients_outpath, **gradients_kwargs)
        debug.success("Saved diffusion gradients to \n", gradients_outpath)


    # Voxel Count
    if modality=="mrsi":
        csv_filename = filename.replace(f"connectivity_{modality}", "voxelcount_per_parcel").replace(".npz",".csv")
        parcel_labels_all = copy.deepcopy(parcel_labels_group)
        parcel_names_all  = copy.deepcopy(parcel_names_group)
        with open(join(resultssubdir,csv_filename), "w", newline="") as f:
            import csv
            writer = csv.writer(f)
            # write header
            writer.writerow(["MRSI voxel counts", "MRSI parcel coverage", "Parcel label", "Parcel name"])
            # write each row
            for i in range(len(parcel_labels_all)):
                try:
                    voxel_counts = np.floor(__n_voxel_counts_dict[parcel_labels_all[i]] / 5**3)
                    if voxel_counts<2:
                        parcel_coverage = 0
                        voxel_counts = 0
                    else:
                        parcel_coverage = round(n_voxel_counts_dict[parcel_labels_all[i]], 2)
                except:
                        parcel_coverage = "NF"
                        voxel_counts    = "NF"
                label = parcel_labels_all[i]
                name = parcel_names_all[i]
                writer.writerow([voxel_counts, parcel_coverage, label, name])
        debug.success("Wrote voxel count per parcel to file",join(resultssubdir,csv_filename))

    return f"Saved stacked connectivity to {matrix_outpath}"





if __name__ == "__main__":
    raise SystemExit("Import and call main(...) with the required inputs.")
