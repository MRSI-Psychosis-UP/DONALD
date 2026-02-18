import os,re,glob
import json
import shutil
import sys
import time
import uuid
import hashlib
from os.path import join, isdir, isfile, dirname, abspath
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import subprocess
import sys
from nilearn import datasets
import nibabel as nib
from scipy.io import loadmat

from mrsitoolbox.tools.datautils import DataUtils
from mrsitoolbox.tools.debug import Debug
from mrsitoolbox.tools.participants import ParticipantSelector
from mrsitoolbox.connectomics.nettools import NetTools
from mrsitoolbox.connectomics.nbs import NBS
from mrsitoolbox.graphplot.brain3d import Brain3D
from rich.table import Table


_SCRIPT_PATH = Path(__file__).resolve()
_VIEWER_ROOT = _SCRIPT_PATH.parents[1]
if not os.getenv("DEVANALYSEPATH"):
    os.environ["DEVANALYSEPATH"] = str(_VIEWER_ROOT)
if not os.getenv("BIDSDATAPATH") or os.getenv("BIDSDATAPATH") == ".":
    os.environ["BIDSDATAPATH"] = str(_VIEWER_ROOT / "data" / "BIDS")

debug    = Debug()
dutils   = DataUtils()
nettools = NetTools()
nbs      = NBS()
brain3d  = Brain3D()



def build_design_matrix(participants_df: pd.DataFrame,
                        subject_ids,
                        session_ids,
                        covariate_names: list[str] | None = None) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return design matrix and mask keeping only subjects with complete covariates."""
    if len(subject_ids) != len(session_ids):
        raise ValueError("subject_ids and session_ids must have the same length.")
    covariate_order: list[str] = []
    seen_covs: set[str] = set()
    for name in covariate_names or []:
        cov = name.strip().lower()
        if cov and cov not in seen_covs:
            covariate_order.append(cov)
            seen_covs.add(cov)
    if not covariate_order:
        raise ValueError("No covariate names provided for design matrix construction.")
    subjects = [ParticipantSelector._normalize_identifier(s) for s in subject_ids]
    sessions = [ParticipantSelector._normalize_identifier(s) for s in session_ids]
    if participants_df is None or participants_df.empty:
        raise ValueError("Participants dataframe is empty; cannot build design matrix.")
    df = participants_df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    required_cols = {"participant_id", "session_id"}
    missing_cols = required_cols.difference(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns {missing_cols} in participants dataframe.")
    df = df.dropna(subset=["participant_id", "session_id"])
    df["participant_id"] = df["participant_id"].astype(str).str.strip()
    df["session_id"] = df["session_id"].astype(str).str.strip()
    available_cols = set(df.columns)
    numeric_covs: list[str] = []
    for cov in covariate_order:
        if cov in {"participant_id", "session_id"}:
            continue
        if cov in available_cols:
            df[cov] = pd.to_numeric(df[cov], errors="coerce")
            numeric_covs.append(cov)
        else:
            raise ValueError(f"Covariate `{cov}` not found in participants dataframe.")
    subset_cols = ["participant_id", "session_id"] + numeric_covs
    df_valid = df.dropna(subset=subset_cols)
    if df_valid.empty:
        raise ValueError("No valid participant rows with requested covariates.")
    cov_lookup: dict[tuple[str, str], dict[str, float | str]] = {}
    for row in df_valid.itertuples(index=False):
        key = (row.participant_id, row.session_id)
        entry = {"participant_id": row.participant_id}
        for cov in covariate_order:
            if cov in {"participant_id", "session_id"}:
                continue
            entry[cov] = getattr(row, cov, np.nan)
        cov_lookup[key] = entry
    cov_values: dict[str, list[float]] = {
        cov: [] for cov in covariate_order if cov not in {"participant_id", "session_id"}
    }
    valid_mask = np.zeros(len(subjects), dtype=bool)
    skipped: list[str] = []
    for idx, (subj, sess) in enumerate(zip(subjects, sessions)):
        if not subj or not sess:
            skipped.append(f"{subj or 'NA'}_{sess or 'NA'}")
            continue
        record = cov_lookup.get((subj, sess))
        if record is None:
            skipped.append(f"{subj}_{sess}")
            continue
        missing = False
        temp_values: dict[str, float] = {}
        for cov in cov_values.keys():
            value = record.get(cov)
            if value is None or pd.isna(value):
                missing = True
                break
            temp_values[cov] = float(value)
        if missing:
            skipped.append(f"{subj}_{sess}")
            continue
        for cov, val in temp_values.items():
            cov_values[cov].append(val)
        valid_mask[idx] = True
    if not any(len(vals) for vals in cov_values.values()):
        raise ValueError("No overlapping subjects between MRSI data and participants covariates.")
    if skipped:
        preview = ", ".join(skipped[:5])
        debug.warning(f"Skipped {len(skipped)} subject/session pairs lacking covariates: {preview}")
    design_dict = {cov: np.asarray(vals, dtype=float) for cov, vals in cov_values.items()}
    return design_dict, valid_mask


def _slugify_fragment(value: str) -> str:
    allowed = []
    for ch in str(value):
        if ch.isalnum() or ch in {"-", "_"}:
            allowed.append(ch)
        else:
            allowed.append("-")
    slug = "".join(allowed).strip("-_")
    return slug or "none"


def _infer_parc_scale_from_path(path: str) -> tuple[str | None, int | None]:
    fname = os.path.basename(str(path))
    match = re.search(r"atlas-chimera(?P<parc>[A-Za-z0-9]+)_scale(?P<scale>\d+)", fname)
    if match:
        return match.group("parc"), int(match.group("scale"))
    match = re.search(r"atlas-cubic_scale(?P<scale>\d+)(?:mm)?", fname)
    if match:
        return "cubic", int(match.group("scale"))
    match = re.search(r"atlas-(?P<parc>[^_]+)_scale(?P<scale>\d+)", fname)
    if match:
        return match.group("parc"), int(match.group("scale"))
    return None, None


def _level_token(level: float) -> str:
    if isinstance(level, (int, np.integer)) or (isinstance(level, float) and level.is_integer()):
        token = str(int(level))
    else:
        token = str(level)
    token = token.replace(".", "p")
    if token.startswith("-"):
        token = "neg" + token[1:]
    token = re.sub(r"[^0-9A-Za-z_-]", "-", token)
    return token


def _format_level(value: float) -> str:
    if isinstance(value, (int, np.integer)) or (isinstance(value, float) and value.is_integer()):
        return str(int(value))
    return f"{value:g}"


def _format_mean_std(values: np.ndarray) -> str:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return "NA"
    mean = float(np.nanmean(values))
    std = float(np.nanstd(values))
    return f"{mean:.3f}±{std:.3f}"


def _display_text(value) -> str:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            value = value.item()
        elif value.size == 1:
            value = value.reshape(-1)[0]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode()
        except Exception:
            value = str(value)
    return "" if value is None else str(value)


def _is_integer_like(values: np.ndarray) -> bool:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return False
    return np.all(np.isclose(values, np.round(values)))


def _print_covariate_distribution(regressor_name: str,
                                  designmat_dict: dict[str, np.ndarray],
                                  nuisance_terms: list[str],
                                  debug: Debug) -> None:
    if regressor_name not in designmat_dict:
        debug.warning(f"Regressor '{regressor_name}' not found in design matrix; skipping covariate table.")
        return
    reg_values = np.asarray(designmat_dict[regressor_name], dtype=float)
    valid_mask = ~np.isnan(reg_values)
    if not np.any(valid_mask):
        debug.warning(f"Regressor '{regressor_name}' contains only NaNs; skipping covariate table.")
        return
    levels = np.unique(reg_values[valid_mask])
    if levels.size == 0:
        debug.warning(f"No valid levels found for regressor '{regressor_name}'.")
        return
    nuisances = [n for n in nuisance_terms if n in designmat_dict and n != regressor_name]
    if not nuisances:
        debug.warning("No nuisance covariates available for distribution table.")
        return

    nuisance_specs = []
    for nuisance in nuisances:
        vals = np.asarray(designmat_dict[nuisance], dtype=float)
        clean = vals[~np.isnan(vals)]
        if clean.size > 0 and _is_integer_like(clean):
            unique_vals = np.unique(clean)
            if unique_vals.size <= 6:
                nuisance_specs.append({
                    "name": nuisance,
                    "type": "categorical",
                    "levels": sorted(unique_vals.tolist()),
                })
                continue
        nuisance_specs.append({"name": nuisance, "type": "continuous"})

    table = Table(title=f"Covariate distribution by {regressor_name}", show_lines=False)
    table.add_column(f"{regressor_name} level", justify="left")
    table.add_column("N", justify="right")
    for spec in nuisance_specs:
        if spec["type"] == "categorical":
            for level in spec["levels"]:
                table.add_column(f"{spec['name']}_{_format_level(level)}", justify="right")
        else:
            table.add_column(spec["name"], justify="right")

    for level in levels:
        level_mask = valid_mask & (reg_values == level)
        n_level = int(level_mask.sum())
        row = [_format_level(level), str(n_level)]
        for spec in nuisance_specs:
            vals = np.asarray(designmat_dict[spec["name"]], dtype=float)[level_mask]
            vals = vals[~np.isnan(vals)]
            if spec["type"] == "categorical":
                for level in spec["levels"]:
                    row.append(str(int(np.sum(np.isclose(vals, level)))))
            else:
                row.append(_format_mean_std(vals))
        table.add_row(*row)

    debug.console.print(table)


def _expand_regressor_for_f(designmat_dict: dict[str, np.ndarray],
                            regressor_name: str,
                            include_reference: bool = False) -> dict[str, object] | None:
    """Expand a categorical regressor (>=3 levels) into dummy columns for MATLAB F-tests."""
    reg = designmat_dict.get(regressor_name)
    if reg is None:
        return None
    values = np.asarray(reg, dtype=float)
    unique_vals = np.unique(values[~np.isnan(values)])
    if unique_vals.size <= 2:
        return None
    levels = sorted(unique_vals.tolist())
    ref_level = None if include_reference else levels[0]
    dummy_cols: list[str] = []
    levels_to_encode = levels if include_reference else levels[1:]
    for level in levels_to_encode:
        token = _level_token(level)
        col_name = f"{regressor_name}_lvl{token}"
        dummy_cols.append(col_name)
        designmat_dict[col_name] = (values == level).astype(float)
    designmat_dict.pop(regressor_name, None)
    regressor_for_bct = dummy_cols[-1] if dummy_cols else regressor_name
    return {
        "levels": levels,
        "ref": ref_level,
        "columns": dummy_cols,
        "regressor": regressor_for_bct,
        "include_reference": include_reference,
    }


def _parse_contrast_vector(raw: str | None) -> np.ndarray | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if ";" in text or "\n" in text:
        return None
    if (text.startswith("[") and text.endswith("]")) or (text.startswith("(") and text.endswith(")")):
        text = text[1:-1].strip()
    parts = [p for p in re.split(r"[,\s]+", text) if p]
    if not parts:
        return None
    try:
        return np.asarray([float(p) for p in parts], dtype=float)
    except ValueError:
        return None


def _parse_contrast_with_placeholder(raw: str | None):
    if raw is None:
        return None, False
    text = str(raw).strip()
    if not text:
        return None, False
    if ";" in text or "\n" in text:
        return None, False
    if (text.startswith("[") and text.endswith("]")) or (text.startswith("(") and text.endswith(")")):
        text = text[1:-1].strip()
    parts = [p for p in re.split(r"[,\s]+", text) if p]
    if not parts:
        return None, False
    values = []
    has_b = False
    for p in parts:
        if p.lower() == "b":
            has_b = True
            values.append("b")
            continue
        try:
            values.append(float(p))
        except ValueError:
            return None, False
    return values, has_b


def _infer_tail_from_contrast(raw: str | None) -> str | None:
    vec = _parse_contrast_vector(raw)
    if vec is None:
        return None
    pos = np.any(vec > 0)
    neg = np.any(vec < 0)
    if pos and neg:
        return "both"
    if pos:
        return "right"
    if neg:
        return "left"
    return None


def _expand_contrast_tokens(tokens, design_columns, matlab_regressor_cols, replace_val=None) -> np.ndarray:
    vec_vals = []
    for t in tokens:
        if isinstance(t, str) and t.lower() == "b":
            if replace_val is None:
                raise ValueError("Contrast placeholder 'b' requires replace_val.")
            vec_vals.append(float(replace_val))
        else:
            vec_vals.append(float(t))
    vec = np.asarray(vec_vals, dtype=float)
    if vec.size == len(design_columns):
        return vec
    if vec.size == len(matlab_regressor_cols):
        full = np.zeros(len(design_columns), dtype=float)
        for val, col in zip(vec.tolist(), matlab_regressor_cols):
            if col not in design_columns:
                raise ValueError(f"Contrast column '{col}' not found in design.")
            full[design_columns.index(col)] = val
        return full
    raise ValueError(
        f"--contrast length {vec.size} does not match design columns {len(design_columns)} "
        f"or regressor columns {len(matlab_regressor_cols)}."
    )


def _mat_get(obj, field: str, default=None):
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


def _load_covars_info(path: Path):
    try:
        with np.load(path, allow_pickle=True) as npz:
            if "covars" not in npz:
                return None
            covars = npz["covars"]
    except Exception:
        return None
    df = None
    if pd is not None:
        try:
            df = pd.DataFrame.from_records(covars)
        except Exception:
            df = None
    columns = list(covars.dtype.names) if getattr(covars.dtype, "names", None) else []
    return {"data": covars, "df": df, "columns": columns}


def _escape_matlab_string(value: str) -> str:
    return str(value).replace("'", "''")


def _default_matlab_session_dir(matlab_cmd: str, nbs_path: str) -> str:
    key_src = f"{str(matlab_cmd)}|{str(nbs_path)}|{os.getenv('USER','user')}"
    key = hashlib.sha1(key_src.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return str(Path.home() / ".cache" / "mrsi_viewer" / "matlab_nbs" / key)


def _pid_is_running(pid: int) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _read_pid_file(pid_path: str) -> int | None:
    try:
        text = Path(pid_path).read_text(encoding="utf-8").strip()
        if text:
            return int(text)
    except Exception:
        return None
    return None


def _write_json_atomic(path: str, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + f".tmp.{uuid.uuid4().hex}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(str(tmp_path), str(target))


def _tail_file_since(path: str, offset: int) -> tuple[int, list[str]]:
    if not isfile(path):
        return offset, []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            f.seek(max(0, int(offset)))
            chunk = f.read()
            new_offset = f.tell()
    except Exception:
        return offset, []
    if not chunk:
        return new_offset, []
    lines = [line.rstrip() for line in chunk.splitlines() if line.strip()]
    return new_offset, lines


def _format_matlab_worker_call(script_dir: str, session_dir: str) -> str:
    return (
        f"addpath('{_escape_matlab_string(script_dir)}');"
        f"nbs_worker_loop('{_escape_matlab_string(session_dir)}')"
    )


def _ensure_matlab_worker_running(matlab_cmd: str, script_dir: str, session_dir: str) -> str:
    session_root = Path(session_dir).expanduser().resolve()
    worker_script = Path(script_dir) / "nbs_worker_loop.m"
    if not worker_script.is_file():
        raise FileNotFoundError(
            f"Persistent MATLAB worker script not found: {worker_script}"
        )
    commands_dir = session_root / "commands"
    responses_dir = session_root / "responses"
    session_root.mkdir(parents=True, exist_ok=True)
    commands_dir.mkdir(parents=True, exist_ok=True)
    responses_dir.mkdir(parents=True, exist_ok=True)
    ready_path = session_root / "ready.json"
    pid_path = session_root / "worker.pid"
    log_path = session_root / "worker.log"

    pid = _read_pid_file(str(pid_path))
    if pid is not None and _pid_is_running(pid) and ready_path.is_file():
        return str(log_path)

    if ready_path.exists() and (pid is None or not _pid_is_running(pid)):
        try:
            ready_path.unlink()
        except Exception:
            pass

    worker_call = _format_matlab_worker_call(str(script_dir), str(session_root))
    debug.info(f"Starting persistent MATLAB worker via: {matlab_cmd} -batch \"{worker_call}\"")
    try:
        with open(log_path, "a", encoding="utf-8") as log_stream:
            proc = subprocess.Popen(
                [matlab_cmd, "-batch", worker_call],
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to launch persistent MATLAB worker with `{matlab_cmd}`. ({exc})"
        ) from exc
    try:
        pid_path.write_text(str(proc.pid), encoding="utf-8")
    except Exception:
        pass

    startup_timeout_s = 180.0
    t0 = time.time()
    while (time.time() - t0) < startup_timeout_s:
        if ready_path.is_file():
            return str(log_path)
        if proc.poll() is not None:
            break
        time.sleep(0.2)

    offset, tail_lines = _tail_file_since(str(log_path), max(0, os.path.getsize(log_path) - 12000) if log_path.exists() else 0)
    _ = offset
    message = "Persistent MATLAB worker did not become ready."
    if tail_lines:
        message += " Last log lines:\n" + "\n".join(tail_lines[-20:])
    raise RuntimeError(message)


def _run_matlab_persistent(
    matlab_cmd: str,
    matlab_call: str,
    *,
    script_dir: str,
    session_dir: str,
    expected_output: str | None = None,
) -> None:
    progress_re = re.compile(r"^\|\s*\d+\s*/\s*\d+\s*\|")
    log_path = _ensure_matlab_worker_running(matlab_cmd, script_dir, session_dir)
    session_root = Path(session_dir).expanduser().resolve()
    commands_dir = session_root / "commands"
    responses_dir = session_root / "responses"
    commands_dir.mkdir(parents=True, exist_ok=True)
    responses_dir.mkdir(parents=True, exist_ok=True)

    job_id = uuid.uuid4().hex
    command_path = commands_dir / f"{job_id}.json"
    response_path = responses_dir / f"{job_id}.json"
    payload = {
        "job_id": job_id,
        "matlab_call": matlab_call,
    }
    log_offset = os.path.getsize(log_path) if isfile(log_path) else 0
    debug.info(f"Persistent MATLAB call: {matlab_call}")
    _write_json_atomic(str(command_path), payload)
    debug.info(f"Submitted MATLAB job {job_id} to persistent session: {session_root}")

    last_progress = False
    wait_timeout_s = 24 * 3600
    t0 = time.time()
    output_lines: list[str] = []
    while True:
        log_offset, lines = _tail_file_since(log_path, log_offset)
        for text in lines:
            output_lines.append(text)
            if progress_re.match(text):
                sys.stdout.write("\r" + text)
                sys.stdout.flush()
                last_progress = True
            elif "error" in text.lower() or "exception" in text.lower():
                if last_progress:
                    sys.stdout.write("\n")
                    last_progress = False
                debug.error(text)
            else:
                if last_progress:
                    sys.stdout.write("\n")
                    last_progress = False
                debug.info(text)

        if response_path.is_file():
            break
        if (time.time() - t0) > wait_timeout_s:
            raise TimeoutError(
                f"Timed out waiting for MATLAB persistent job {job_id} response."
            )
        time.sleep(0.25)

    if last_progress:
        sys.stdout.write("\n")
        sys.stdout.flush()

    try:
        with open(response_path, "r", encoding="utf-8") as f:
            response = json.load(f)
    except Exception as exc:
        raise RuntimeError(
            f"Could not read persistent MATLAB job response {response_path}: {exc}"
        ) from exc
    finally:
        try:
            response_path.unlink()
        except Exception:
            pass

    ok = bool(response.get("ok", False))
    err_message = str(response.get("message", "")).strip()
    if not ok:
        if expected_output and isfile(expected_output):
            debug.warning(
                f"MATLAB persistent job failed, but output exists at {expected_output}. Proceeding."
            )
            return
        if err_message:
            for line in err_message.splitlines()[-20:]:
                debug.error(line)
        elif output_lines:
            debug.error("MATLAB persistent job failed. Last output lines:")
            for line in output_lines[-12:]:
                debug.error(line)
        raise subprocess.CalledProcessError(1, [matlab_cmd, "-batch", matlab_call])


def _run_matlab_batch(matlab_cmd: str, matlab_call: str, expected_output: str | None = None) -> None:
    progress_re = re.compile(r"^\|\s*\d+\s*/\s*\d+\s*\|")
    try:
        process = subprocess.Popen(
            [matlab_cmd, "-batch", matlab_call],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to launch MATLAB command `{matlab_cmd}`. Check --matlab-cmd and PATH. ({exc})"
        ) from exc
    last_progress = False
    header_lines = []
    header_shown = False
    output_lines = []
    if process.stdout:
        for line in process.stdout:
            text = line.rstrip()
            if text:
                output_lines.append(text)
            if progress_re.match(text):
                if header_lines and not header_shown:
                    for h in header_lines:
                        debug.info(h)
                    header_shown = True
                sys.stdout.write("\r" + text)
                sys.stdout.flush()
                last_progress = True
            elif text.startswith("|") and ("Permutation" in text or "Random" in text or "p-value" in text):
                if last_progress:
                    sys.stdout.write("\n")
                    last_progress = False
                header_lines.append(text)
                debug.info(text)
            elif "error" in text.lower() or "exception" in text.lower():
                if last_progress:
                    sys.stdout.write("\n")
                    last_progress = False
                debug.error(text)
            elif text:
                if last_progress:
                    sys.stdout.write("\n")
                    last_progress = False
                debug.info(text)
    rc = process.wait()
    if last_progress:
        sys.stdout.write("\n")
        sys.stdout.flush()
    if rc != 0:
        if expected_output and isfile(expected_output):
            debug.warning(
                f"MATLAB exited with code {rc}, but output exists at {expected_output}. Proceeding."
            )
            return
        if output_lines:
            debug.error("MATLAB NBS failed. Last output lines:")
            for line in output_lines[-12:]:
                debug.error(line)
        raise subprocess.CalledProcessError(rc, [matlab_cmd, "-batch", matlab_call])


def _resolve_matlab_helper_dir() -> str:
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir,
        script_dir.parents[1] / "mrsitoolbox" / "experiments" / "MetSiM_analysis",
    ]
    for candidate in candidates:
        if (candidate / "nbs_run_cli.m").is_file():
            return str(candidate)
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not locate nbs_run_cli.m. Searched: {searched}")


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


def _load_matlab_nbs_results(mat_path: str, alpha: float, expected_n: int | None = None) -> dict:
    data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    nbs = data.get("nbs")
    if nbs is None:
        raise ValueError(f"MATLAB results missing 'nbs' struct: {mat_path}")
    nbs_struct = _mat_get(nbs, "NBS")
    if nbs_struct is None:
        raise ValueError(f"MATLAB results missing NBS field: {mat_path}")
    pval = _mat_get(nbs_struct, "pval")
    con_mat = _mat_get(nbs_struct, "con_mat")
    test_stat = _mat_get(nbs_struct, "test_stat")

    pvals_list = []
    if pval is not None and np.size(pval) > 0:
        pvals_list = np.atleast_1d(np.asarray(pval, dtype=float)).tolist()

    comp_pvals: list[float] = []
    comp_masks: list[np.ndarray] = []
    for idx, comp in enumerate(_cell_to_list(con_mat)):
        mat_bool = _to_dense_bool(comp)
        if mat_bool is None:
            continue
        mat_bool = np.asarray(mat_bool, dtype=bool)
        if mat_bool.ndim != 2 or mat_bool.size == 0 or mat_bool.shape[0] == 0 or mat_bool.shape[1] == 0:
            continue
        mat_bool = mat_bool | mat_bool.T
        comp_masks.append(mat_bool)
        if idx < len(pvals_list):
            comp_pvals.append(float(pvals_list[idx]))
        else:
            comp_pvals.append(1.0)

    t_mat = None
    if test_stat is not None and np.size(test_stat) > 0:
        t_mat = np.asarray(test_stat, dtype=float)
        t_mat = np.squeeze(t_mat)
    if t_mat is None or t_mat.ndim != 2:
        if comp_masks:
            n = comp_masks[0].shape[0]
            t_mat = np.zeros((n, n), dtype=float)
        elif expected_n is not None and int(expected_n) > 0:
            t_mat = np.zeros((int(expected_n), int(expected_n)), dtype=float)
        else:
            raise ValueError("MATLAB results missing test statistic matrix.")

    iu = np.triu_indices(t_mat.shape[0], 1)
    comp_edges = []
    for mask in comp_masks:
        if mask.size == 0:
            continue
        upper = np.triu(mask, 1)
        if upper.shape != t_mat.shape:
            continue
        comp_edges.append(np.where(upper[iu])[0])

    sig_mask = np.zeros_like(t_mat, dtype=bool)
    for p, m in zip(comp_pvals, comp_masks):
        if p < alpha:
            if m.shape != sig_mask.shape:
                continue
            sig_mask |= m

    return {
        "comp_pvals": comp_pvals,
        "comp_edges": comp_edges,
        "comp_masks": comp_masks,
        "sig_mask": sig_mask,
        "t_mat": t_mat,
        "iu": iu,
        "null_max": np.array([], dtype=float),
    }


def _load_gm_adjacency_matrix(group: str,
                              atlas: str,
                              parcel_labels: np.ndarray,
                              lobes: list[str],
                              bids_root: str) -> np.ndarray:
    """Load GM adjacency matrix reordered to `parcel_labels`."""
    netpathdir = join(bids_root, group, "derivatives", "group", "networkpaths", atlas)
    adjacency_dicts = []
    for brain_lobe in lobes:
        for hemi in ("lh", "rh"):
            filename = f"{group}_{atlas}_{hemi}_{brain_lobe}_desc-GMadjacency.npz"
            filepath = join(netpathdir, filename)
            if not isfile(filepath):
                continue
            adj_data = np.load(filepath)
            adjacency_dicts.append(
                nettools.mat_to_graphdict(
                    adj_data["adjacency_mat"],
                    labels=adj_data["parcel_labels"],
                    symmetric=True,
                )
            )
    if not adjacency_dicts:
        raise FileNotFoundError("No GM adjacency files located for the requested atlas.")
    adj_dict = nettools.merge_adjacency_dicts(*adjacency_dicts)
    adj_dict = nettools.reorder_adjacency_dict(adj_dict, parcel_labels)
    return nettools.graphdict_to_mat(adj_dict)



# --------------------------------- CLI -------------------------------------
# def main():
parser = argparse.ArgumentParser(description="Load all path disruption results into compact structures")
parser.add_argument('--parc', type=str, default="LFMIHIFIS",
                    help='Chimera parcellation scheme, valid choices: LFMIHIFIS,LFIIIIFIS, LFMIHIFIF. Default: LFMIHIFIS')
parser.add_argument('--scale', type=int, default=3,
                    help="Cortical parcellation scale (default: 3)")
parser.add_argument('--npert', type=int, default=50,
                    help='Number of perturbations as comma-separated integers (default: 50)')
parser.add_argument('--diag', type=str, default="controls",choices=['group', 'controls','patients'],
                help="Only inlcude controls, patients or all ('group'[default])")
parser.add_argument('--preproc', type=str, default="filtbiharmonic_pvcorr_GM",help="Preprocessing of orig MRSI files (default: filtbiharmonic_pvcorr_GM)")
parser.add_argument(
    '--permtest',
    type=str,
    default="freedman",
    choices=["freedman", "gmadj"],
    help=(
        "Null generation strategy: 'freedman' applies Freedman–Lane permutations across subjects "
        "(or sign-flips if no regressor is provided) while 'gmadj' shuffles node identities only "
        "between gray-matter adjacent parcels."
    ),
)
parser.add_argument('-c', '--contrast', type=str, default=None,
                help="Contrast vector for MATLAB NBS (e.g., '[0 0 1 1 1]'). "
                     "For python runs, the sign is used to infer the tail.")
parser.add_argument('--nperm', type=int, default=1500,
                    help="Number of permutations used by the NBS (default: 1000)")
parser.add_argument('--nthreads', type=int, default=8,
                    help="Number of CPU threads for permutation parallelism (default: 8)")
parser.add_argument('--alpha', type=float, default=0.05,
                    help="NBS t-test pvalue threshold (default: 0.05)")
parser.add_argument('--t_thresh', type=float, default=3.5,
                    help="Primary t-statistic threshold applied before component building (default: 3.5)")
parser.add_argument('--nuisance', type=str, default=None,
                    help="Comma-separated list of covariate names to treat as nuisances (default: none)")
parser.add_argument('--lobes', type=str, default="all",choices=["all", "ctx","subc"],
                    help="Brain lobes where to constrain NBS (default: all) (choices:all, ctx ,subc )")
parser.add_argument('--regress', type=str, default="state",
                    help="Covariate name inside the design matrix to test (default: 'state')")
parser.add_argument(
    "--regressor-type",
    "--regressor_type",
    choices=["categorical", "continuous"],
    default="categorical",
    help="Interpret regressor as categorical labels or continuous numeric values.",
)
parser.add_argument('--balance', action="store_true",
                    help="If set, downsample the regressor groups to equal size before NBS.")
parser.add_argument('--train_split', type=float, default=1.0,
                    help="Proportion of subjects to keep for training/NBS (default: 1.0).")
parser.add_argument('--select', '-s', dest='select', action='append', default=[],
                    help="Filter participants by covariate in connectivity NPZ covars. "
                         "Format: COVARNAME,VALUE or COVARNAME,>VALUE (e.g., --select Diag,0).")
parser.add_argument('--engine', type=str, default="matlab",
                    choices=["python", "matlab"],
                    help="Choose implementation to run: python or matlab (default: python).")
default_matlab_cmd = (
    os.getenv("MRSI_MATLAB_CMD")
    or os.getenv("MATLAB_CMD")
    or os.getenv("MATLAB_EXECUTABLE")
    or shutil.which("matlab")
    or ""
)
default_matlab_nbs_path = (
    os.getenv("MRSI_NBS_PATH")
    or os.getenv("MATLAB_NBS_PATH")
    or os.getenv("NBS_PATH")
    or ""
)
parser.add_argument('--matlab-cmd', type=str, default=default_matlab_cmd,
                    help="MATLAB executable for -batch runs (default: env/PATH lookup, else empty).")
parser.add_argument('--matlab-nbs-path', type=str, default=default_matlab_nbs_path,
                    help="Path to NBS MATLAB folder containing NBSrun.m (default: env lookup, else empty).")
parser.add_argument('--matlab-test', type=str, default="t",
                    choices=["t", "F"], help="MATLAB NBS test type (t or F).")
parser.add_argument('--matlab-size', type=str, default="extent",
                    choices=["extent", "intensity"], help="MATLAB NBS size metric.")
parser.add_argument(
    "--matlab-no-precompute",
    action="store_true",
    help=(
        "Force MATLAB NBS to skip precomputing permutation statistics. "
        "Can reduce peak memory but is often slower."
    ),
)
parser.add_argument(
    "--matlab-persistent",
    action="store_true",
    help=(
        "Use a persistent MATLAB worker session so repeated runs reuse the same "
        "MATLAB process and worker pool."
    ),
)
parser.add_argument(
    "--matlab-session-dir",
    type=str,
    default=None,
    help=(
        "Optional session directory used for persistent MATLAB worker IPC/state. "
        "If omitted, a default cache directory is used."
    ),
)
parser.add_argument('--input', '-i', type=str, default=None,
                    help="Optional full path to the connectivity .npz file. Defaults to group connectivity.")
parser.add_argument('--parcellation-path', type=str, default=None,
                    help="Optional full path to a parcellation NIfTI (.nii/.nii.gz) overriding atlas lookup.")
parser.add_argument('--modality', type=str, default=None,
                    choices=["fmri", "mrsi", "dwi", "anat", "morph", "pet", "other"],
                    help="Optional modality override used for results output path.")


args = parser.parse_args()
parc_scheme = args.parc 
scale       = args.scale
npert       = args.npert
preproc_str = args.preproc.replace("filt","")
diag        = args.diag
regressor_name = (args.regress or "state").strip().lower()
if not regressor_name:
    raise ValueError("Regressor name cannot be empty.")
nuisance_cli = [s.strip().lower() for s in args.nuisance.split(",")] if args.nuisance else []
nuisance_cli = [n for n in nuisance_cli if n]
run_matlab = args.engine == "matlab"
contrast_cli = (args.contrast or "").strip() or None
contrast_tokens, contrast_has_b = _parse_contrast_with_placeholder(contrast_cli) if contrast_cli else (None, False)
tail_value = "both" if contrast_has_b else (_infer_tail_from_contrast(contrast_cli) or "both")
contrast_vec = None
contrast_cli_matlab = None
if contrast_cli:
    if contrast_tokens is None:
        raise ValueError("--contrast must be a single-row numeric vector (e.g., '0 0 1 1 1').")
    if not contrast_has_b:
        contrast_vec = np.asarray(contrast_tokens, dtype=float)
        contrast_cli_matlab = " ".join(f"{v:g}" for v in contrast_vec)

lobes_choice = (args.lobes or "all").strip().lower()
if lobes_choice not in {"all", "ctx", "subc"}:
    raise ValueError("--lobes must be one of: all, ctx, subc.")
lobes_for_adj = ["ctx", "subc"] if lobes_choice == "all" else [lobes_choice]

design_covariates: list[str] = []
if regressor_name:
    design_covariates.append(regressor_name)
for cov in nuisance_cli:
    if cov not in design_covariates:
        design_covariates.append(cov)

########################### Load Group MeSiMs ###########################
if not args.input:
    raise ValueError("--input is required; group/modality are now loaded from the NPZ.")
connectivity_path = abspath(os.path.expanduser(args.input))
# debug.warning("con_file",connectivity_path,os.path.exists(connectivity_path))
# import sys;sys.exit()
if not isfile(connectivity_path):
    raise FileNotFoundError(f"Connectivity file not found: {connectivity_path}")

inferred_parc, inferred_scale = _infer_parc_scale_from_path(connectivity_path)
if inferred_parc is not None and inferred_scale is not None:
    if inferred_parc != parc_scheme or inferred_scale != scale:
        debug.warning(
            f"Overriding --parc/--scale with values inferred from input: "
            f"parc={inferred_parc}, scale={inferred_scale}."
        )
    parc_scheme = inferred_parc
    scale = inferred_scale
else:
    debug.warning("Could not infer parc/scale from input path; using --parc/--scale.")

########################### parcel image: ###########################
atlas              = f"cubic-{scale}" if "cubic" in parc_scheme else f"chimera-{parc_scheme}-{scale}"
gm_mask            = datasets.load_mni152_gm_mask().get_fdata().astype(bool)
if args.parcellation_path:
    parcellation_path = abspath(os.path.expanduser(args.parcellation_path))
    if not isfile(parcellation_path):
        raise FileNotFoundError(f"Parcellation file not found: {parcellation_path}")
    debug.info(f"Using parcellation override: {parcellation_path}")
    parcel_mni_img_nii = nib.load(parcellation_path)
else:
    parcel_mni_img_nii = nib.load(join(dutils.DEVDATAPATH, "atlas", f"{atlas}", f"{atlas}.nii.gz"))
parcel_mni_img_np  = parcel_mni_img_nii.get_fdata().astype(int)

data           = np.load(connectivity_path,allow_pickle=True)
group = _display_text(data["group"]).strip() if "group" in data else ""
modality_npz = _display_text(data["modality"]).strip().lower() if "modality" in data else ""
modality_override = (args.modality or "").strip().lower()
modality = modality_override or modality_npz
if not group:
    raise ValueError("Connectivity NPZ missing group.")
if not modality:
    raise ValueError("Connectivity NPZ missing modality. Provide --modality from {fmri,mrsi,dwi,anat,morph,pet,other}.")
resultdirname = f"{group}-{diag}-population_average"
parc_scale_dir = f"parc-{_slugify_fragment(parc_scheme)}_scale-{scale}"
base_dir      = join(dutils.ANARESULTSPATH, "nbs", group, modality, parc_scale_dir)
MeSiM_pop_avg  = data["matrix_pop_avg"]
MeSiM_list     = np.asarray(data["matrix_subj_list"])
subject_id_all = np.asarray(data["subject_id_list"])
session_all    = np.asarray(data["session_id_list"])
metabolites    = data["metabolites"] if "metabolites" in data else []
metab_profiles_subj_list = data["metab_profiles_subj_list"]
parcel_labels  = np.asarray(data["parcel_labels_group"])
parcel_names   = np.asarray(data["parcel_names_group"])
mni_template   = datasets.load_mni152_template()
covars_info = _load_covars_info(Path(connectivity_path))
participants_df = covars_info["df"] if covars_info else None
if participants_df is None or participants_df.empty:
    raise ValueError("Connectivity NPZ missing covars; cannot proceed without covariates.")
centroids_world = nettools.compute_centroids(parcel_mni_img_nii, parcel_labels, world=True)
parcel_df = pd.DataFrame({
    "label": parcel_labels,
    "name": parcel_names,
    "XCoord(mm)": centroids_world[:, 0],
    "YCoord(mm)": centroids_world[:, 1],
    "ZCoord(mm)": centroids_world[:, 2],
}).reset_index(drop=True)

selection_suffix = ""
selection_value_tag = None
if args.select:
    selector = ParticipantSelector(debug=debug, exit_on_error=True)
    selection_result = selector.apply(
        participants_df,
        args.select,
        subject_ids=subject_id_all,
        session_ids=session_all,
    )
    selection_mask = selection_result.pair_mask
    if selection_mask is None:
        raise ValueError("Selection mask could not be computed.")
    included = int(selection_mask.sum())
    if included == 0:
        raise ValueError("No matching subject/session pairs after applying --select filters.")
    if included != selection_mask.size:
        debug.info(
            f"Restricting analysis to {included} of {selection_mask.size} subjects after --select filtering."
        )
    MeSiM_list = MeSiM_list[selection_mask]
    subject_id_all = subject_id_all[selection_mask]
    session_all = session_all[selection_mask]
    if metab_profiles_subj_list.shape[0] == selection_mask.size:
        metab_profiles_subj_list = metab_profiles_subj_list[selection_mask]
    selection_suffix = selection_result.selection_suffix
    selection_value_tag = selection_result.selection_value_tag
#
########################### Load default boundary conditions ###########################
gm_adj = None
if args.permtest == "gmadj":
    try:
        gm_adj = _load_gm_adjacency_matrix(
            group=group,
            atlas=atlas,
            parcel_labels=parcel_labels,
            lobes=lobes_for_adj,
            bids_root=dutils.BIDSDATAPATH,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "GM adjacency files are required when permtest='gmadj', "
            "but none were found."
        ) from exc
elif args.permtest == "freedman" and lobes_choice in {"ctx", "subc"}:
    keep_mask = np.array([
        ("ctx" in str(name).lower()) if lobes_choice == "ctx" else ("ctx" not in str(name).lower())
        for name in parcel_names
    ])
    if not np.any(keep_mask):
        raise ValueError(f"No parcels matched lobe selection '{lobes_choice}'.")
    parcel_labels = parcel_labels[keep_mask]
    parcel_names = parcel_names[keep_mask]
    centroids_world = centroids_world[keep_mask]
    parcel_df = parcel_df.iloc[np.where(keep_mask)[0]].reset_index(drop=True)
    MeSiM_list = MeSiM_list[:, keep_mask][:, :, keep_mask]
    MeSiM_pop_avg = MeSiM_pop_avg[keep_mask][:, keep_mask]

if gm_adj is None:
    try:
        gm_adj = _load_gm_adjacency_matrix(
            group=group,
            atlas=atlas,
            parcel_labels=parcel_labels,
            lobes=lobes_for_adj,
            bids_root=dutils.BIDSDATAPATH,
        )
    except FileNotFoundError:
        gm_adj = None
        debug.warning("GM adjacency files not found; GM-constrained energy will be skipped.")

###########################################################################
############################ NBS Design matrix ############################
###########################################################################
designmat_dict, valid_mask = build_design_matrix(
    participants_df,
    subject_id_all,
    session_all,
    covariate_names=design_covariates,
)
regressor_values_raw = designmat_dict.get(regressor_name).copy() if regressor_name in designmat_dict else None

if valid_mask.shape[0] != MeSiM_list.shape[0]:
    raise ValueError("Mismatch between MRSI subject count and participants covariates.")


if not np.all(valid_mask):
    kept = int(valid_mask.sum())
    dropped = int(valid_mask.size - kept)
    debug.warning(f"Restricting MeSiM data to {kept} subjects with complete covariates "
                  f"(dropped {dropped}).")
    MeSiM_list = MeSiM_list[valid_mask]
    subject_id_all = subject_id_all[valid_mask]
    session_all = session_all[valid_mask]

balance_extra = None
if args.balance:
    reg_array = designmat_dict.get(regressor_name)
    if reg_array is None:
        debug.warning("--balance requested but regressor not found in design matrix.")
    else:
        groups, counts = np.unique(reg_array, return_counts=True)
        if groups.size != 2:
            debug.warning("--balance requires a binary regressor; skipping balancing.")
        else:
            target = counts.min()
            balance_mask = np.zeros(reg_array.shape[0], dtype=bool)
            for val in groups:
                idxs = np.where(reg_array == val)[0]
                rng = np.random.default_rng(seed=args.nperm if args.nperm else None)
                rng.shuffle(idxs)
                balance_mask[idxs[:target]] = True
    if balance_mask.sum() != reg_array.shape[0]:
        drop_idx = np.where(~balance_mask)[0]
        keep_idx = np.where(balance_mask)[0]
        balance_extra = {
            "mesim": MeSiM_list[drop_idx].copy(),
            "regressor": designmat_dict[regressor_name][drop_idx].copy(),
            "subject_id": subject_id_all[drop_idx].copy(),
            "session_id": session_all[drop_idx].copy(),
        }
        debug.info(f"Balancing dataset to {target} subjects per group "
                   f"(moving {drop_idx.size} subjects to validation).")
        MeSiM_list = MeSiM_list[keep_idx]
        subject_id_all = subject_id_all[keep_idx]
        session_all = session_all[keep_idx]
        if regressor_values_raw is not None:
            regressor_values_raw = regressor_values_raw[keep_idx]
        for key in list(designmat_dict.keys()):
            designmat_dict[key] = designmat_dict[key][keep_idx]
    else:
        debug.info("Dataset already balanced; no subjects removed.")

full_subject_ids = subject_id_all.copy()
full_session_ids = session_all.copy()
full_data = {
    "mesim": MeSiM_list,
    "designmat": {k: v.copy() for k, v in designmat_dict.items()},
}
heldout = np.array([], dtype=int)
train_idx = np.arange(MeSiM_list.shape[0])

train_ratio = min(max(args.train_split, 0.0), 1.0)
if train_ratio < 1.0:
    rng = np.random.default_rng(seed=args.nperm if args.nperm else None)
    perm_idx = rng.permutation(MeSiM_list.shape[0])
    n_train = max(1, int(train_ratio * MeSiM_list.shape[0]))
    train_idx = np.sort(perm_idx[:n_train])
    heldout = np.sort(perm_idx[n_train:])
    if heldout.size > 0:
        debug.info(f"Using {n_train} subjects for training/NBS; holding out {heldout.size} for validation.")
    MeSiM_list = MeSiM_list[train_idx]
    subject_id_all = subject_id_all[train_idx]
    session_all = session_all[train_idx]
    if regressor_values_raw is not None:
        regressor_values_raw = regressor_values_raw[train_idx]
for key in list(designmat_dict.keys()):
    designmat_dict[key] = designmat_dict[key][train_idx]

nbs_regressor_name = regressor_name
reg_array = designmat_dict.get(nbs_regressor_name)
if args.regressor_type == "continuous":
    clean = np.asarray(reg_array, dtype=float)
    clean = clean[~np.isnan(clean)]
    mean_val = float(np.nanmean(reg_array))
    std_val = float(np.nanstd(reg_array))
    if std_val == 0 or np.isnan(std_val):
        debug.warning(
            f"Regressor '{nbs_regressor_name}' has zero/NaN std; skipping z-transform."
        )
    else:
        designmat_dict[nbs_regressor_name] = (reg_array - mean_val) / std_val
        debug.info(
            f"Z-transformed continuous regressor '{nbs_regressor_name}' "
            f"(mean={mean_val:.3f}, std={std_val:.3f})."
        )
else:
    debug.info(
        f"Using categorical regressor '{nbs_regressor_name}'; skipping z-transform."
    )

nuisance_terms_preview = list(nuisance_cli)
_print_covariate_distribution(regressor_name, designmat_dict, nuisance_terms_preview, debug)

matlab_regressor_cols = [regressor_name]
regressor_for_bct = regressor_name
expanded_info = None
add_intercept = True
if run_matlab and str(args.matlab_test).upper() == "F":
    expanded_info = _expand_regressor_for_f(
        designmat_dict, regressor_name, include_reference=True
    )
    if expanded_info:
        regressor_for_bct = str(expanded_info["regressor"])
        matlab_regressor_cols = list(expanded_info["columns"])
        add_intercept = False
        debug.info(
            f"MATLAB F-test: expanded '{regressor_name}' into {matlab_regressor_cols} "
            "and removed intercept."
        )
    else:
        debug.warning(
            "MATLAB F-test requested but regressor has <=2 unique values; using single column."
        )
if nuisance_cli:
    nuisance_terms = list(nuisance_cli)
else:
    nuisance_terms = []
nuisance_terms_display = list(nuisance_terms)
if run_matlab and str(args.matlab_test).upper() == "F" and matlab_regressor_cols:
    for col in matlab_regressor_cols:
        if col != regressor_for_bct and col not in nuisance_terms:
            nuisance_terms.append(col)
    nuisance_terms_display = [n for n in nuisance_terms if n not in matlab_regressor_cols]
nuis_tag = "-".join(_slugify_fragment(n) for n in nuisance_terms) if nuisance_terms else "none"
selection_tag = ""
if selection_suffix:
    if selection_value_tag:
        selection_tag = f"_{selection_suffix}_{selection_value_tag}"
    else:
        selection_tag = f"_{selection_suffix}"
test_type_tag = str(args.matlab_test).lower() if run_matlab else "t"
if run_matlab and str(args.matlab_test).upper() == "F":
    tail_tag = "na"
else:
    tail_tag = "both" if contrast_has_b else tail_value
param_tag  = (
    f"perm-{_slugify_fragment(args.permtest)}_nperm-{args.nperm}_th-{args.t_thresh}"
    f"_reg-{_slugify_fragment(regressor_name)}_nuis-{nuis_tag}"
    f"_lobes-{_slugify_fragment(lobes_choice)}_test-{test_type_tag}_tail-{tail_tag}{selection_tag}"
)
plot_dir = join(base_dir, "connectome_plots")
matlab_export_dir = join(base_dir, "matlab_nbs", param_tag)
os.makedirs(plot_dir, exist_ok=True)
predictor_label = regressor_for_bct if regressor_for_bct != regressor_name else regressor_name
if run_matlab and str(args.matlab_test).upper() == "F" and matlab_regressor_cols:
    debug.info(
        f"Start bct_corr with MATLAB F-test across --{', '.join(matlab_regressor_cols)}-- "
        f"and nuisance params -- {nuisance_terms_display} --"
    )
else:
    debug.info(f"Start bct_corr with predictor --{predictor_label}-- and nuisance params -- {nuisance_terms} --")

### Start NBS ###
debug.info(f"Preparing {MeSiM_list.shape[0]} matrices for matlab NBS")
results_dict = nbs.bct_corr(MeSiM_list,
             designmat_dict,
             nuisance = nuisance_terms,
             regress  = regressor_for_bct,
             permtest = args.permtest,
             gm_adj   = gm_adj if args.permtest == "gmadj" else None,
             t_thresh = args.t_thresh,
             n_perms  = 0 if run_matlab else args.nperm,
             nthreads = args.nthreads,
             tail     = tail_value,
             export_matlab_dir = matlab_export_dir,
             node_coords = centroids_world,
             node_names = parcel_names,
             subject_ids = subject_id_all,
             add_intercept = add_intercept,
             )


if parcel_df.shape[0] != results_dict["t_mat"].shape[0]:
    raise ValueError("Parcel coordinate table does not match the NBS matrix dimension.")

export_paths = results_dict.get("export_paths") or {}
if export_paths:
    debug.success(f"MATLAB NBS inputs saved to {export_paths.get('export_dir')}")
    if run_matlab:
        export_dir = export_paths.get("export_dir")
        if not export_dir or not isdir(export_dir):
            raise FileNotFoundError("MATLAB export directory not found; cannot run MATLAB NBS.")
        design_columns = export_paths.get("design_columns") or []
        if design_columns:
            base_vec = ["0"] * len(design_columns)
        else:
            raise ValueError("MATLAB export missing design columns; cannot build contrast.")
        if contrast_vec is not None:
            if contrast_vec.size != len(design_columns):
                if contrast_vec.size == len(matlab_regressor_cols):
                    vec = np.zeros(len(design_columns), dtype=float)
                    for val, col in zip(contrast_vec.tolist(), matlab_regressor_cols):
                        if col not in design_columns:
                            raise ValueError(f"Contrast column '{col}' not found in design.")
                        vec[design_columns.index(col)] = val
                    contrast_vec = vec
                    contrast_cli_matlab = " ".join(f"{v:g}" for v in contrast_vec.tolist())
                else:
                    cols_msg = ", ".join(design_columns)
                    raise ValueError(
                        f"--contrast length {contrast_vec.size} does not match design columns "
                        f"{len(design_columns)}. Design columns: [{cols_msg}]"
                    )

        is_f_test = str(args.matlab_test).upper() == "F"
        if contrast_has_b and is_f_test:
            raise ValueError("Contrast placeholder 'b' is only valid for t-tests.")

        two_tailed_b = bool(contrast_has_b and not is_f_test)
        if is_f_test:
            contrast_specs = [("F", None, None)]
        else:
            if two_tailed_b:
                contrast_specs = [("pos", "right", 1.0), ("neg", "left", -1.0)]
            elif contrast_cli:
                contrast_specs = [("custom", tail_value, None)]
            elif tail_value == "both":
                contrast_specs = [("pos", "right", 1.0), ("neg", "left", -1.0)]
            elif tail_value == "left":
                contrast_specs = [("neg", "left", -1.0)]
            else:
                contrast_specs = [("pos", "right", 1.0)]

        script_dir = dirname(__file__)
        matlab_helper_dir = _resolve_matlab_helper_dir()
        nbs_path = args.matlab_nbs_path or ""
        matlab_cmd = args.matlab_cmd or ""
        if not matlab_cmd:
            raise ValueError(
                "MATLAB executable is not configured. Set it in GUI Preferences "
                "or pass --matlab-cmd."
            )
        if not nbs_path:
            raise ValueError(
                "NBS path is not configured. Set it in GUI Preferences "
                "or pass --matlab-nbs-path."
            )
        matlab_persistent = bool(args.matlab_persistent)
        matlab_session_dir = (
            str(Path(args.matlab_session_dir).expanduser().resolve())
            if args.matlab_session_dir
            else _default_matlab_session_dir(matlab_cmd, nbs_path)
        )
        if matlab_persistent:
            debug.info(f"MATLAB persistent session enabled: {matlab_session_dir}")
        matlab_outputs = []
        for suffix, tail_for_run, sign in contrast_specs:
            if is_f_test:
                contrast_cols = matlab_regressor_cols or [regressor_for_bct]
                missing_cols = [c for c in contrast_cols if c not in design_columns]
                if missing_cols:
                    raise ValueError(f"MATLAB F-test contrast columns missing in design: {missing_cols}")
                if contrast_cli_matlab:
                    matlab_contrast = contrast_cli_matlab
                else:
                    vec = list(base_vec)
                    for col in contrast_cols:
                        vec[design_columns.index(col)] = "1"
                    matlab_contrast = " ".join(vec)
                output_mat = "nbs_results.mat"
            else:
                if two_tailed_b:
                    vec = _expand_contrast_tokens(contrast_tokens, design_columns, matlab_regressor_cols, replace_val=sign)
                    matlab_contrast = " ".join(f"{v:g}" for v in vec.tolist())
                    output_mat = f"nbs_results_{suffix}.mat"
                elif contrast_cli_matlab:
                    matlab_contrast = contrast_cli_matlab
                    output_mat = "nbs_results.mat"
                else:
                    vec = list(base_vec)
                    vec[-1] = str(int(sign)) if sign is not None else "1"
                    matlab_contrast = " ".join(vec)
                    output_mat = "nbs_results.mat" if len(contrast_specs) == 1 else f"nbs_results_{suffix}.mat"
            helper_dir_esc = _escape_matlab_string(matlab_helper_dir)
            script_dir_esc = _escape_matlab_string(script_dir)
            addpath_helper = "" if matlab_helper_dir == script_dir else f"addpath('{helper_dir_esc}');"
            matlab_call = (
                "{addpath_helper}"
                "addpath('{script_dir}');"
                "nbs_run_cli('export_dir','{export_dir}',"
                "'nbs_path','{nbs_path}',"
                "'contrast','{contrast}',"
                "'test','{test}',"
                "'size','{size}',"
                "'thresh',{thresh},"
                "'alpha',{alpha},"
                "'perms',{perms},"
                "'tail','{tail}',"
                "'nthreads',{nthreads},"
                "'no_precompute',{no_precompute},"
                "'output_mat','{output_mat}')"
            ).format(
                addpath_helper=addpath_helper,
                export_dir=_escape_matlab_string(export_dir),
                script_dir=script_dir_esc,
                nbs_path=_escape_matlab_string(nbs_path),
                contrast=_escape_matlab_string(matlab_contrast),
                test=_escape_matlab_string(args.matlab_test),
                size=_escape_matlab_string(args.matlab_size),
                thresh=args.t_thresh,
                alpha=args.alpha,
                perms=args.nperm,
                tail=_escape_matlab_string(tail_for_run or tail_value),
                nthreads=int(max(1, int(args.nthreads))),
                no_precompute="true" if bool(args.matlab_no_precompute) else "false",
                output_mat=_escape_matlab_string(output_mat),
            )
            output_path = join(export_dir, output_mat)
            if matlab_persistent:
                debug.info(
                    f"Running MATLAB NBS via persistent worker in {matlab_session_dir}"
                )
                _run_matlab_persistent(
                    matlab_cmd,
                    matlab_call,
                    script_dir=script_dir,
                    session_dir=matlab_session_dir,
                    expected_output=output_path,
                )
            else:
                debug.info(f"Running MATLAB NBS via: {matlab_cmd} -batch \"{matlab_call}\"")
                _run_matlab_batch(matlab_cmd, matlab_call, expected_output=output_path)
            matlab_outputs.append((suffix, output_mat))
        matlab_results = []
        for suffix, output_mat in matlab_outputs:
            output_path = join(export_dir, output_mat)
            if not isfile(output_path):
                raise FileNotFoundError(f"MATLAB output not found: {output_path}")
            res = _load_matlab_nbs_results(
                output_path,
                alpha=args.alpha,
                expected_n=MeSiM_list.shape[1] if MeSiM_list.ndim >= 3 else None,
            )
            res["export_paths"] = export_paths
            matlab_results.append((suffix, res))
        if two_tailed_b:
            pos_res = next((r for s, r in matlab_results if s == "pos"), None)
            neg_res = next((r for s, r in matlab_results if s == "neg"), None)
            if pos_res is None or neg_res is None:
                raise ValueError("Two-tailed test requested but could not load both MATLAB results.")
            comp_masks = list(pos_res["comp_masks"]) + list(neg_res["comp_masks"])
            comp_edges = list(pos_res["comp_edges"]) + list(neg_res["comp_edges"])
            comp_pvals = [min(1.0, 2.0 * float(p)) for p in pos_res["comp_pvals"]]
            comp_pvals += [min(1.0, 2.0 * float(p)) for p in neg_res["comp_pvals"]]
            sig_mask = np.zeros_like(pos_res["t_mat"], dtype=bool)
            for p, m in zip(comp_pvals, comp_masks):
                if p < args.alpha:
                    if m.shape != sig_mask.shape:
                        continue
                    sig_mask |= m
            results_dict = {
                "comp_pvals": comp_pvals,
                "comp_edges": comp_edges,
                "comp_masks": comp_masks,
                "sig_mask": sig_mask,
                "t_mat": pos_res["t_mat"],
                "iu": pos_res["iu"],
                "null_max": pos_res.get("null_max", np.array([], dtype=float)),
                "export_paths": export_paths,
            }
        else:
            results_dict = matlab_results[0][1]


ids_sig    = np.where(np.array(results_dict["comp_pvals"]) < args.alpha)[0]
n_sig_comp = len(ids_sig)
pvalue_arr = np.array(results_dict["comp_pvals"])
global_pvalue = float(np.nanmin(pvalue_arr)) if pvalue_arr.size else float("nan")
if n_sig_comp:
    debug.success(f"Found {n_sig_comp} significant component(s). Significant p-values:")
    for i in ids_sig:
        debug.info(f"Component {i}: pvalue = {pvalue_arr[i]}")
else:
    debug.warning("No significant components detected; exporting component data anyway.")
    for i, val in enumerate(pvalue_arr):
        debug.info(f"Component {i}: pvalue = {val}")
debug.info(f"Global p-value (min component p): {global_pvalue if np.isfinite(global_pvalue) else 'NA'}")
summary_payload = {
    "n_significant": int(n_sig_comp),
    "significant_indices": [int(i) for i in ids_sig.tolist()],
    "significant_pvalues": [float(pvalue_arr[i]) for i in ids_sig.tolist()],
    "global_pvalue": (float(global_pvalue) if np.isfinite(global_pvalue) else None),
}
print(f"[NBS_SUMMARY]{json.dumps(summary_payload, separators=(',', ':'))}", flush=True)

regressor_values_out_all = regressor_values_raw
if regressor_values_out_all is None and expanded_info:
    level_cols = expanded_info.get("columns") or []
    levels = expanded_info.get("levels") or []
    if level_cols and levels and all(col in designmat_dict for col in level_cols):
        mat = np.column_stack([designmat_dict[col] for col in level_cols])
        idx = np.argmax(mat, axis=1)
        regressor_values_out_all = np.asarray([levels[i] for i in idx], dtype=float)

components_all_path = join(plot_dir, f"{param_tag}_components_all.npz")
covars_records = participants_df.to_records(index=False) if participants_df is not None else np.array([])
comp_masks_all = np.asarray(results_dict["comp_masks"], dtype=np.uint8)
comp_edges_all = np.array(results_dict.get("comp_edges", []), dtype=object)
np.savez(
    components_all_path,
    comp_masks=comp_masks_all,
    comp_pvals=pvalue_arr.astype(float),
    sig_indices=ids_sig.astype(int),
    sig_mask=results_dict.get("sig_mask"),
    t_matrix=results_dict["t_mat"],
    permtest=args.permtest,
    nperm=np.array(args.nperm, dtype=int),
    t_thresh=np.array(args.t_thresh, dtype=float),
    npert=np.array(npert, dtype=int),
    preproc=args.preproc,
    test_type=test_type_tag,
    test_tail=tail_tag,
    connectivity_path=connectivity_path,
    subject_ids=subject_id_all.astype(str),
    session_ids=session_all.astype(str),
    regressor_name=regressor_name,
    regressor_values=regressor_values_out_all if regressor_values_out_all is not None else np.array([]),
    group=group,
    parc_scheme=parc_scheme,
    scale=np.array(scale, dtype=int),
    diag=diag,
    lobes=lobes_choice,
    param_tag=param_tag,
    parcel_labels=parcel_labels,
    parcel_names=parcel_names,
    centroids_world=centroids_world,
    covars=covars_records,
    comp_edges=comp_edges_all,
)
debug.success("Bundled component results saved to", components_all_path)
print(f"[NBS_RESULT]{components_all_path}", flush=True)

summary_rows: list[dict[str, object]] = []
total_components = len(pvalue_arr)
for comp_idx in range(total_components):
    comp_tag = f"{param_tag}_comp-{comp_idx}"
    summary_rows.append({
        "component_idx": comp_idx,
        "pvalue": float(pvalue_arr[comp_idx]),
        "significant": bool(comp_idx in ids_sig),
        "permtest": args.permtest,
        "nperm": args.nperm,
        "t_thresh": args.t_thresh,
        "npert": npert,
        "preproc": args.preproc,
        "test_type": test_type_tag,
        "test_tail": tail_tag,
        "connectivity_path": connectivity_path,
        "regressor_name": regressor_name,
        "npz_path": "",
        "plot_path": "",
        "nodes_path": "",
        "meta_path": "",
    })

for comp_idx in range(total_components):
    is_sig = comp_idx in ids_sig
    comp_tag = f"{param_tag}_comp-{comp_idx}"
    plot_path = join(plot_dir, f"{comp_tag}_connectome.png")
    list_path = join(plot_dir, f"{comp_tag}_nodes.tsv")
    meta_path = join(plot_dir, f"{comp_tag}_summary.tsv")
    npz_path = join(plot_dir, f"{comp_tag}_results.npz")
    raw_mask = np.asarray(results_dict["comp_masks"][comp_idx], dtype=bool)
    nbs_network_arr = raw_mask if is_sig else np.zeros_like(raw_mask, dtype=bool)
    comp_nodes = np.where(nbs_network_arr.sum(axis=0) > 0)[0]
    component_labels = parcel_labels[comp_nodes]
    component_names = [str(parcel_names[i]) for i in comp_nodes]
    component_coords = centroids_world[comp_nodes]
    comp_results = {
        "comp_masks": [nbs_network_arr],
        "comp_pvals": [results_dict["comp_pvals"][comp_idx]],
        "t_mat": results_dict["t_mat"],
    }
    regressor_values_out = regressor_values_raw
    if regressor_values_out is None and expanded_info:
        level_cols = expanded_info.get("columns") or []
        levels = expanded_info.get("levels") or []
        if level_cols and levels and all(col in designmat_dict for col in level_cols):
            mat = np.column_stack([designmat_dict[col] for col in level_cols])
            idx = np.argmax(mat, axis=1)
            regressor_values_out = np.asarray([levels[i] for i in idx], dtype=float)
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("component_idx\tpvalue\tpermtest\tnperm\tt_thresh\tsignificant\n")
        f.write(f"{comp_idx}\t{results_dict['comp_pvals'][comp_idx]:.6f}\t"
                f"{args.permtest}\t{args.nperm}\t{args.t_thresh}\t{int(is_sig)}\n")
    with open(list_path, "w", encoding="utf-8") as f:
        f.write("label\tname\tx\ty\tz\n")
        for lbl, name, coord in zip(component_labels, component_names, component_coords):
            f.write(f"{int(lbl)}\t{name}\t{coord[0]:.3f}\t{coord[1]:.3f}\t{coord[2]:.3f}\n")
    covars_records = participants_df.to_records(index=False) if participants_df is not None else np.array([])
    np.savez(
        npz_path,
        component_idx=np.array(comp_idx, dtype=int),
        comp_mask=nbs_network_arr.astype(np.uint8),
        t_matrix=results_dict["t_mat"],
        pvalue=np.array(results_dict["comp_pvals"][comp_idx], dtype=float),
        significant=np.array(int(is_sig), dtype=int),
        permtest=args.permtest,
        nperm=np.array(args.nperm, dtype=int),
        t_thresh=np.array(args.t_thresh, dtype=float),
        npert=np.array(npert, dtype=int),
        preproc=args.preproc,
        test_type=test_type_tag,
        test_tail=tail_tag,
        connectivity_path=connectivity_path,
        subject_ids=subject_id_all.astype(str),
        session_ids=session_all.astype(str),
        regressor_name=regressor_name,
        regressor_values=regressor_values_out if regressor_values_out is not None else np.array([]),
        group=group,
        parc_scheme=parc_scheme,
        scale=np.array(scale, dtype=int),
        diag=diag,
        lobes=lobes_choice,
        param_tag=param_tag,
        parcel_labels=parcel_labels,
        parcel_names=parcel_names,
        centroids_world=centroids_world,
        covars=covars_records,
    )
    summary_rows[comp_idx].update({
        "npz_path": npz_path,
        "plot_path": plot_path if is_sig else "",
        "nodes_path": list_path,
        "meta_path": meta_path,
        "connectivity_path": connectivity_path,
    })
    if is_sig:
        brain3d.plot_significant_connectome(
            comp_results,
            parcel_df,
            plot_path,
            alpha=0.8,
            node_size=6,
            node_color="cyan",
            edge_cmap="PiYG",
            black_bg=True,
        )
        debug.success(f"Saved component {comp_idx} outputs to {plot_dir}")
if summary_rows:
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = join(plot_dir, f"{param_tag}_components_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    debug.success("Component summary CSV saved to", summary_csv)
