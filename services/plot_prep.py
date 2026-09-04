#!/usr/bin/env python3
"""Helpers for preparing workspace entries for matrix plotting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .display_state import (
    default_entry_display_settings,
    ensure_entry_display_settings,
    log_scale_error,
    normalize_display_scale,
    parse_display_limits,
)


@dataclass
class PlotEntryResolution:
    matrix: np.ndarray
    key: str | None
    source_path: Path | None
    stack_axis: int | None
    stack_len: int | None

def resolve_entry_plot(
    entry,
    *,
    ensure_entry_key,
    load_matrix_from_npz,
    stack_axis,
    average_to_square,
    select_stack_slice,
):
    source_path_raw = entry.get("source_path", entry.get("path"))
    source_path = Path(source_path_raw) if source_path_raw else None

    if entry.get("kind") == "derived":
        raw = np.asarray(entry.get("matrix"))
        axis = None
        stack_len = None
        if raw.ndim == 3:
            axis = stack_axis(raw.shape)
            if axis is None:
                matrix = average_to_square(raw)
            else:
                stack_len = int(raw.shape[axis])
                sample_index = entry.get("sample_index", -1)
                if sample_index is None or sample_index < 0 or sample_index >= stack_len:
                    matrix = average_to_square(raw)
                else:
                    matrix = select_stack_slice(raw, axis, sample_index)
        else:
            matrix = raw
        return PlotEntryResolution(
            matrix=np.asarray(matrix),
            key=entry.get("selected_key"),
            source_path=source_path,
            stack_axis=axis,
            stack_len=stack_len,
        )

    key = ensure_entry_key(entry)
    if not key:
        raise KeyError("No valid matrix key selected.")

    raw = load_matrix_from_npz(entry["path"], key, average=False)
    axis = None
    stack_len = None
    if raw.ndim == 3:
        axis = stack_axis(raw.shape)
        if axis is None:
            matrix = average_to_square(raw)
        else:
            stack_len = int(raw.shape[axis])
            sample_index = entry.get("sample_index", -1)
            if sample_index is None or sample_index < 0 or sample_index >= stack_len:
                matrix = average_to_square(raw)
            else:
                matrix = select_stack_slice(raw, axis, sample_index)
    else:
        matrix = raw
    return PlotEntryResolution(
        matrix=np.asarray(matrix),
        key=key,
        source_path=source_path,
        stack_axis=axis,
        stack_len=stack_len,
    )


def normalize_matrix_labels(matrix, labels, names, *, to_string_list):
    matrix = np.asarray(matrix)
    labels_list = to_string_list(labels)
    names_list = to_string_list(names)
    expected_len = matrix.shape[0] if matrix.ndim >= 1 else 0
    if labels_list is not None and len(labels_list) != expected_len:
        labels_list = None
    if names_list is not None and len(names_list) != expected_len:
        names_list = None
    return labels_list, names_list
