"""Compute metabolic connectivity matrices from MRSIPrep metabolic-profile files.

MRSIPrep writes ``*_desc-metabolicprofiles_mrsi.npz`` per recording, holding the
per-parcel metabolite profile and, when the run perturbed it, an npert-augmented
feature block. This module turns those into similarity matrices without the GUI,
so the logic stays testable and the dialog is only presentation.

mrsitoolbox is the reference for the maths. ``ENGINE_REFERENCE`` calls
``mrsitoolbox.connectomics.mesim._compute_submatrix`` -- the same worker
``MeSiM._compute_metsim_parallel`` dispatches to -- so tie handling, the
NaN/all-zero parcel skips and the p-values match matrices produced anywhere
else in the toolbox.

``ENGINE_FAST`` is the default and reproduces that worker *exactly* (verified
to 0.0 absolute difference on real data, for both correlations, matrices and
p-values) by vectorising it and applying the same skip rules. The reference
loops over all N^2 parcel pairs in Python: 1.13 s versus 0.003 s for one
91-parcel matrix, a 393x difference that in per-perturbation mode separates
38 s from 0.2 s per recording.

MeSiM's parallel driver is bypassed in both cases. It spawns a
``ProcessPoolExecutor`` and pickles the profile to every worker, which costs
more than the chunk it distributes. Parallelism belongs across *files*, which
is also what gives one progress bar per profile.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: Filename marker MRSIPrep uses for profile files.
PROFILE_MARKER = "metabolicprofiles"

#: Written alongside the profile, replacing the marker.
MATRIX_MARKER = "connectivity"

CORRELATION_MODES = ("spearman", "pearson")

#: Correlate the whole npert-augmented feature vector in one pass, or correlate
#: each perturbation separately and summarise across them.
MODE_AUGMENTED = "augmented"
MODE_PER_PERTURBATION = "per_perturbation"
COMPUTE_MODES = (MODE_AUGMENTED, MODE_PER_PERTURBATION)

#: ``mrsitoolbox`` is the reference implementation; ``fast`` is a vectorised
#: path verified to reproduce it exactly (see tests). Default to fast because
#: the reference loops over every parcel pair in Python -- measured at 1.13 s
#: versus 0.003 s for one 91-parcel matrix, i.e. 393x, which in per-perturbation
#: mode is the difference between 38 s and 0.2 s per recording.
ENGINE_FAST = "fast"
ENGINE_REFERENCE = "mrsitoolbox"
ENGINES = (ENGINE_FAST, ENGINE_REFERENCE)


class ProfileError(RuntimeError):
    """Raised when a file is not a usable metabolic-profile npz."""


@dataclass
class ProfileInfo:
    """What the dialog needs to list and filter a profile without loading it fully."""

    path: Path
    subject: str = ""
    session: str = ""
    atlas: str = ""
    metabolites: tuple = ()
    npert: int = 0
    n_parcels: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def supports_perturbation(self) -> bool:
        return self.npert > 1

    def label(self, root: Path = None) -> str:
        try:
            return str(self.path.relative_to(root)) if root else self.path.name
        except ValueError:
            return self.path.name


@dataclass
class ComputeParams:
    correlation: str = "spearman"
    mode: str = MODE_AUGMENTED
    n_parallel: int = 4
    overwrite: bool = False
    ignore_parcels: tuple = field(default_factory=tuple)
    engine: str = ENGINE_FAST


def _entity(name: str, key: str) -> str:
    match = re.search(rf"{key}-([A-Za-z0-9]+)", name)
    return match.group(1) if match else ""


def find_profile_files(folder: Path):
    """Every metabolic-profile npz under ``folder``, sorted for stable listing."""
    folder = Path(folder)
    hits = []
    for path in folder.rglob("*.npz"):
        if PROFILE_MARKER in path.name.lower():
            hits.append(path)
    return sorted(hits, key=lambda p: str(p).lower())


def read_profile_info(path: Path) -> ProfileInfo:
    """Header-only read: enough to list and filter, without holding features."""
    path = Path(path)
    info = ProfileInfo(
        path=path,
        subject=_entity(path.name, "sub"),
        session=_entity(path.name, "ses"),
        atlas=_entity(path.name, "atlas"),
    )
    try:
        with np.load(path, allow_pickle=True) as data:
            if "features" not in data.files:
                info.error = "no 'features' array"
                return info
            features = data["features"]
            if features.ndim != 2:
                info.error = f"features is {features.ndim}D, expected 2D"
                return info
            info.n_parcels = int(features.shape[0])
            info.metabolites = tuple(str(m) for m in data["metabolites"]) if "metabolites" in data.files else ()
            info.npert = int(data["npert"]) if "npert" in data.files else 0
    except Exception as exc:  # unreadable/corrupt file is a listing row, not a crash
        info.error = str(exc)
    return info


def output_path_for(profile_path: Path) -> Path:
    """Matrix path beside the profile, with the profile marker swapped out.

    Keeps the rest of the filename -- atlas, npert, filtering and PVC tags --
    so the matrix stays traceable to the profile and the run that made it.
    """
    profile_path = Path(profile_path)
    name = profile_path.name
    lowered = name.lower()
    index = lowered.find(PROFILE_MARKER)
    if index < 0:
        return profile_path.with_name(profile_path.stem + f"_desc-{MATRIX_MARKER}.npz")
    return profile_path.with_name(name[:index] + MATRIX_MARKER + name[index + len(PROFILE_MARKER):])


def _reference_matrix(concentrations: dict, correlation: str, ignore=None):
    """mrsitoolbox's own worker, the one ``MeSiM._compute_metsim_parallel`` uses.

    Called directly rather than through that driver so no process pool is
    created: it would pickle the profile to every worker for a job measured in
    milliseconds per chunk.
    """
    from mrsitoolbox.connectomics.mesim import _compute_submatrix

    parcel_ids = list(concentrations.keys())
    _, _, matrix, pvalues = _compute_submatrix(
        0, 0, len(parcel_ids), parcel_ids, concentrations, list(ignore or []), correlation
    )
    return np.nan_to_num(matrix, nan=0.0), pvalues


def _fast_matrix(concentrations: dict, correlation: str, ignore=None):
    """Vectorised equivalent of :func:`_reference_matrix`.

    Reproduces the reference's skip rules rather than just its correlation:
    parcels that are all-zero, contain a NaN, or are explicitly ignored are
    left as zero rows/columns, which is what makes the two agree exactly
    instead of merely closely.
    """
    from scipy import stats

    parcel_ids = list(concentrations.keys())
    data = np.vstack([np.asarray(concentrations[pid], dtype=float).ravel() for pid in parcel_ids])
    n = data.shape[0]

    skip = np.zeros(n, dtype=bool)
    ignore_set = set(int(i) for i in (ignore or []))
    for i, pid in enumerate(parcel_ids):
        row = data[i]
        skip[i] = int(pid) in ignore_set or np.isnan(row.mean()) or row.sum() == 0

    matrix = np.zeros((n, n), dtype=float)
    pvalues = np.zeros((n, n), dtype=float)
    keep = ~skip
    if keep.sum() < 2:
        return matrix, pvalues

    block = data[keep]
    if correlation == "spearman":
        result = stats.spearmanr(block.T)
        corr = np.atleast_2d(result.statistic)
        pval = np.atleast_2d(result.pvalue)
    elif correlation == "pearson":
        corr = np.corrcoef(block)
        # Same two-sided t test scipy's linregress reports, which is what the
        # reference path returns for pearson.
        dof = block.shape[1] - 2
        with np.errstate(divide="ignore", invalid="ignore"):
            t_stat = corr * np.sqrt(dof / np.clip(1.0 - corr**2, 1e-300, None))
        pval = 2.0 * stats.t.sf(np.abs(t_stat), dof)
    else:
        raise ProfileError(f"unknown correlation {correlation!r}; expected one of {CORRELATION_MODES}")

    idx = np.flatnonzero(keep)
    matrix[np.ix_(idx, idx)] = np.nan_to_num(corr, nan=0.0)
    pvalues[np.ix_(idx, idx)] = np.nan_to_num(pval, nan=0.0)
    return matrix, pvalues


def _matrix_from_concentrations(concentrations: dict, correlation: str, ignore=None, engine: str = ENGINE_FAST):
    if engine == ENGINE_REFERENCE:
        return _reference_matrix(concentrations, correlation, ignore)
    return _fast_matrix(concentrations, correlation, ignore)


def _concentrations(block: np.ndarray, parcel_ids) -> dict:
    return {int(pid): np.asarray(block[i], dtype=float) for i, pid in enumerate(parcel_ids)}


def compute_matrix(features, parcel_ids, params: ComputeParams, npert: int = 0, n_metabolites: int = 0, progress=None):
    """Similarity matrix (and, in perturbation mode, its spread).

    ``progress`` is called as ``progress(done, total)`` so a caller can drive a
    bar; perturbation mode reports per perturbation, augmented mode is a single
    step because it is one correlation pass over the full feature vector.

    Returns ``(matrix, pvalues, std)`` with ``std`` None outside perturbation
    mode.
    """
    features = np.asarray(features, dtype=float)

    if params.mode == MODE_PER_PERTURBATION:
        if npert <= 1 or n_metabolites <= 0:
            raise ProfileError("per-perturbation mode needs npert > 1 and a known metabolite count")
        expected = npert * n_metabolites
        if features.shape[1] != expected:
            raise ProfileError(
                f"features has {features.shape[1]} columns, expected npert*metabolites = {expected}"
            )
        stack = []
        pvalue_stack = []
        for k in range(npert):
            block = features[:, k * n_metabolites : (k + 1) * n_metabolites]
            matrix, pvalues = _matrix_from_concentrations(
                _concentrations(block, parcel_ids), params.correlation, params.ignore_parcels, params.engine
            )
            stack.append(matrix)
            pvalue_stack.append(pvalues)
            if progress:
                progress(k + 1, npert)
        stacked = np.stack(stack)
        # Mean across perturbations is the matrix; the spread is kept because it
        # is the only record of how stable each edge was under perturbation, and
        # it cannot be recovered from the mean alone.
        return stacked.mean(axis=0), np.stack(pvalue_stack).mean(axis=0), stacked.std(axis=0)

    matrix, pvalues = _matrix_from_concentrations(
        _concentrations(features, parcel_ids), params.correlation, params.ignore_parcels, params.engine
    )
    if progress:
        progress(1, 1)
    return matrix, pvalues, None


def compute_profile(path: Path, params: ComputeParams, progress=None) -> dict:
    """Load one profile, compute its matrix and write it beside the source.

    Returns a summary dict; raises :class:`ProfileError` for anything the user
    can act on (unreadable file, mode that the file cannot support).
    """
    path = Path(path)
    out_path = output_path_for(path)
    if out_path.exists() and not params.overwrite:
        return {"path": path, "output": out_path, "skipped": True, "reason": "output exists"}

    with np.load(path, allow_pickle=True) as data:
        if "features" not in data.files:
            raise ProfileError(f"{path.name}: no 'features' array")
        features = data["features"]
        metabolites = [str(m) for m in data["metabolites"]] if "metabolites" in data.files else []
        npert = int(data["npert"]) if "npert" in data.files else 0
        parcel_ids = (
            np.asarray(data["labels_indices"]).tolist()
            if "labels_indices" in data.files
            else list(range(features.shape[0]))
        )
        parcel_names = (
            np.asarray(data["parcel_names"])
            if "parcel_names" in data.files
            else np.array([str(p) for p in parcel_ids])
        )
        parcel_concentrations = data["parcel_concentrations"] if "parcel_concentrations" in data.files else None

    matrix, pvalues, std = compute_matrix(
        features, parcel_ids, params, npert=npert, n_metabolites=len(metabolites), progress=progress
    )

    payload = {
        "matrix": matrix,
        "pvalues": pvalues,
        "labels_indices": np.asarray(parcel_ids),
        "parcel_names": parcel_names,
        "metabolites": np.array(metabolites),
        "method": params.correlation,
        "engine": params.engine,
        "compute_mode": params.mode,
        "npert": npert,
        "source_profile": str(path),
    }
    if std is not None:
        payload["matrix_std"] = std
    if parcel_concentrations is not None:
        payload["parcel_concentrations"] = parcel_concentrations

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **payload)
    return {
        "path": path,
        "output": out_path,
        "skipped": False,
        "n_parcels": int(matrix.shape[0]),
        "mode": params.mode,
        "correlation": params.correlation,
    }
