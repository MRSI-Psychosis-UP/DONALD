#!/usr/bin/env python3
"""Data-access helpers for file-backed matrix workspace entries."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class MatrixDataAccess:
    """Owns path-based caches and entry-to-matrix resolution logic."""

    def __init__(
        self,
        *,
        load_covars_info,
        get_valid_keys,
        load_parcel_metadata,
        load_group_value,
        load_matrix_from_npz,
        stack_axis,
        average_to_square,
        select_stack_slice,
        to_string_list,
    ) -> None:
        self._load_covars_info = load_covars_info
        self._get_valid_keys = get_valid_keys
        self._load_parcel_metadata = load_parcel_metadata
        self._load_group_value = load_group_value
        self._load_matrix_from_npz = load_matrix_from_npz
        self._stack_axis = stack_axis
        self._average_to_square = average_to_square
        self._select_stack_slice = select_stack_slice
        self._to_string_list = to_string_list

        self.covars_cache = {}
        self.valid_keys_cache = {}
        self.parcel_metadata_cache = {}
        self.group_values_cache = {}

    @staticmethod
    def entry_source_path(entry):
        if entry is None:
            return None
        source_path = entry.get("source_path", entry.get("path"))
        if not source_path:
            return None
        return Path(source_path)

    def clear_caches(self) -> None:
        self.covars_cache.clear()
        self.valid_keys_cache.clear()
        self.parcel_metadata_cache.clear()
        self.group_values_cache.clear()

    def invalidate_path(self, path: Path) -> None:
        path = Path(path)
        self.covars_cache.pop(path, None)
        self.valid_keys_cache.pop(path, None)
        for cache_key in list(self.parcel_metadata_cache.keys()):
            cache_path = cache_key[0] if isinstance(cache_key, tuple) else cache_key
            if cache_path == path:
                self.parcel_metadata_cache.pop(cache_key, None)
        self.group_values_cache.pop(path, None)

    def covars_info(self, path: Path):
        path = Path(path)
        cached = self.covars_cache.get(path)
        if cached is None:
            cached = self._load_covars_info(path)
            self.covars_cache[path] = cached
        return cached

    def get_valid_keys(self, path: Path):
        path = Path(path)
        cached = self.valid_keys_cache.get(path)
        if cached is None:
            cached = list(self._get_valid_keys(path))
            self.valid_keys_cache[path] = cached
        return cached

    def load_parcel_metadata_cached(self, path: Path, label_key: str = "", name_key: str = ""):
        path = Path(path)
        cache_key = (
            path,
            str(label_key or "").strip(),
            str(name_key or "").strip(),
        )
        if cache_key not in self.parcel_metadata_cache:
            self.parcel_metadata_cache[cache_key] = self._load_parcel_metadata(
                path,
                label_key=cache_key[1],
                name_key=cache_key[2],
            )
        return self.parcel_metadata_cache[cache_key]

    @staticmethod
    def npz_vector_keys(path: Path, *, expected_len=None):
        path = Path(path)
        options = []
        try:
            with np.load(path, allow_pickle=True) as npz:
                for key in npz.files:
                    try:
                        values = np.asarray(npz[key])
                    except Exception:
                        continue
                    if values.ndim == 0:
                        continue
                    try:
                        flat_len = int(values.reshape(-1).size)
                    except Exception:
                        continue
                    if flat_len <= 0:
                        continue
                    if expected_len is not None and flat_len != int(expected_len):
                        continue
                    options.append(
                        {
                            "key": str(key),
                            "shape": tuple(int(dim) for dim in values.shape),
                            "dtype": str(values.dtype),
                            "size": flat_len,
                        }
                    )
        except Exception:
            return []
        return options

    def load_group_value_cached(self, path: Path, index: int):
        path = Path(path)
        if path not in self.group_values_cache:
            group_data = []
            try:
                with np.load(path, allow_pickle=True) as npz:
                    if "group" in npz:
                        group_data = np.asarray(npz["group"]).reshape(-1)
            except Exception:
                group_data = []
            self.group_values_cache[path] = group_data
        group_data = self.group_values_cache.get(path)
        if group_data is None or len(group_data) == 0:
            return self._load_group_value(path, index)
        if index < 0 or index >= len(group_data):
            return None
        value = group_data[index]
        if isinstance(value, np.generic):
            value = value.item()
        return str(value)

    def ensure_entry_key(self, entry):
        if entry.get("kind") != "file":
            return entry.get("selected_key")
        key = entry.get("selected_key")
        if key:
            return key
        valid_keys = self.get_valid_keys(entry["path"])
        if not valid_keys:
            return None
        entry["selected_key"] = valid_keys[0]
        return entry["selected_key"]

    def matrix_for_entry(self, entry):
        if entry.get("kind") == "derived":
            return entry.get("matrix"), entry.get("selected_key")
        key = self.ensure_entry_key(entry)
        if not key:
            raise KeyError("No valid matrix key selected")
        raw = self._load_matrix_from_npz(entry["path"], key, average=False)
        if raw.ndim == 3:
            axis = self._stack_axis(raw.shape)
            if axis is None:
                matrix = self._average_to_square(raw)
            else:
                entry.setdefault("sample_index", -1)
                entry["stack_axis"] = axis
                entry["stack_len"] = raw.shape[axis]
                sample_index = entry.get("sample_index", -1)
                if sample_index is None or sample_index < 0:
                    matrix = self._average_to_square(raw)
                else:
                    matrix = self._select_stack_slice(raw, axis, sample_index)
        else:
            matrix = raw
        return matrix, key

    def entry_parcel_metadata(self, entry, expected_len=None, label_key: str = "", name_key: str = ""):
        labels = entry.get("parcel_labels_group") if entry is not None else None
        names = entry.get("parcel_names_group") if entry is not None else None
        label_key = str(label_key or "").strip()
        name_key = str(name_key or "").strip()
        if label_key:
            labels = None
        if name_key:
            names = None

        source_path = self.entry_source_path(entry)
        if (labels is None or names is None) and source_path is not None and source_path.exists():
            try:
                source_labels, source_names = self.load_parcel_metadata_cached(
                    source_path,
                    label_key=label_key,
                    name_key=name_key,
                )
                if labels is None:
                    labels = source_labels
                if names is None:
                    names = source_names
            except Exception:
                pass

        labels_array = None
        if labels is not None:
            try:
                labels_array = np.asarray(labels).reshape(-1)
            except Exception:
                labels_array = None

        names_list = self._to_string_list(names)

        if expected_len is not None:
            if labels_array is not None and labels_array.size != int(expected_len):
                labels_array = None
            if names_list is not None and len(names_list) != int(expected_len):
                names_list = None

        return labels_array, names_list
