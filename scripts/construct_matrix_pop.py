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


def _retain_parcel_labels(matrix, profile, parcel_labels, parcel_names, retain_labels, prefix):
    matrix = np.asarray(matrix)
    labels_arr = _coerce_parcel_labels(parcel_labels, fallback_size=matrix.shape[0] if matrix.ndim >= 1 else 0)
    if not retain_labels:
        return matrix, profile, labels_arr, _coerce_parcel_names(parcel_names, len(labels_arr))

    names_arr = _coerce_parcel_names(parcel_names, len(labels_arr))
    retain_list = list(retain_labels)
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

def main(con_path_list,covar_df,modality,matrix_outpath,
         ignore_parc_list=[],qmask_path=None,mrsi_cov=0.67,
         comp_gradients=True,parcellation_img_path=None,
         parcel_labels_retain=[]):
    if not con_path_list:
        return "No connectivity paths were provided."

    group = extract_group_from_inputs(covar_df, con_path_list)
    modality = str(modality or "").strip().lower()
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
    retained_labels = list(parcel_labels_retain or [])
    skipped_archives = []

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
                matrix = np.asarray(con_data["connectome_density"])
                profile = np.zeros((matrix.shape[0], 1, 1))
                parcel_labels = np.asarray(con_data.get("parcel_labels"))
                parcel_names = np.asarray(con_data.get("parcel_names"))
            elif modality == "func":
                con_data = _load_connectivity_archive(con_path, allow_pickle=True)
                matrix = np.asarray(con_data["connectivity"])
                profile = con_data["timeseries"] if "timeseries" in con_data else np.zeros((matrix.shape[0], 1))
                parcel_labels = np.asarray(con_data.get("labels_indices"))
                parcel_names = np.asarray(con_data.get("labels"))
            else:
                con_data = _load_connectivity_archive(con_path, allow_pickle=True)
                matrix = np.asarray(con_data["simmatrix_sp"])
                crlb_matrix = np.var(matrix, axis=0) * 100 if len(matrix.shape) == 3 else np.zeros_like(matrix)
                matrix = np.median(matrix, axis=0) if len(matrix.shape) == 3 else matrix
                matrix[crlb_matrix > 20] = 0
                parcel_concentrations = np.asarray(con_data["parcel_concentrations"])
                profile = parcel_concentrations
                metabolite_list = list(np.asarray(con_data["metabolites_leaveout"]).reshape(-1))
                parcel_labels = np.asarray(con_data.get("labels_indices"))
                parcel_names = np.asarray(con_data.get("labels"))
                npert = max(
                    npert,
                    extract_npert_from_shape(
                        parcel_concentrations,
                        metabolite_list,
                        matrix.shape[0],
                    ),
                )

            matrix, profile, parcel_labels, parcel_names = _retain_parcel_labels(
                matrix,
                profile,
                parcel_labels,
                parcel_names,
                retained_labels,
                prefix,
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
    debug.title("Exclude sparse within-subject-wise MeSiMs")
    matrix_list_sel,i,e = filter_sparse_matrices(matrix_subjects_list,sigma=3)
    node_signals_subjects_sel = np.delete(node_signals_subjects,e,axis=0)
    session_id_arr_sel      = np.delete(session_list_found,e,axis=0)
    subject_id_arr_sel      = np.delete(subject_id_list_found,e,axis=0)
    matrix_list_sel          = np.array(matrix_list_sel)
    discarded_subjects      = [subject_id_list_found[idx] for idx in np.array(e)]
    discarded_sessions      = [session_list_found[idx] for idx in np.array(e)]
    debug.info(f"Excluded {len(e)} sparse MeSiMs of shape, remaining {matrix_list_sel.shape[0]}")
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
                    modality                 = modality)
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
