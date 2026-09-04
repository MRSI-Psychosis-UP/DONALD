#!/usr/bin/env python3
"""Audit stored NBS result files for the subject/design mis-pairing defect.

Background: NBS1.2's own MATLAB reader (`readUI.m`, via MATLAB `dir()`) and the python
`analysis_subject_order` workaround it inspired both load connectivity matrices in
*lexicographic* filename order (subject1, subject10, subject11, ..., subject2, ...)
while the design matrix is built in *numeric* subject order. Unpadded exports
(`subject1.txt` instead of `subject001.txt`) therefore paired almost every subject's
connectome with a different subject's covariates. See mrsitoolbox/connectomics/nbs.py's
`_subject_filename` for the fix and NBS1.2/data/mrsi/... for the reconstruction that
first surfaced this.

This script never modifies, moves, or deletes a result file -- it only reads
`*_components_all.npz` files and writes a markdown report classifying each one.

Classification method (no source data required):
  MATLAB NBS1.2 and its python port (`bct_glm_matlab_compat`) both compute corrected
  p-values as p = b / K (no smoothing). The legacy python port (`bct_corr`) uses
  p = (b + 1) / (K + 1). For a given stored p-value and permutation count K, at most
  one of `p*K` or `p*(K+1)-1` is (numerically) an integer, which identifies which rule
  -- and therefore which pairing -- produced the file:
    - "matlab_rule"  -> produced by NBS.m or the compat port with the pre-fix,
                        unpadded export -> pairing was almost certainly scrambled.
    - "legacy_rule"  -> produced by legacy bct_corr, which indexes matrices
                        numerically -> pairing was correct, unaffected.
    - "clean"        -> the file carries the `subject_order="numeric"` tag this audit
                        script's fix started stamping -> known-good regardless of rule.
    - "ambiguous"     -> too few components/permutations to distinguish the two rules,
                        or the file predates both conventions. Needs a manual check
                        (reconstruct the observed statistic from the source connectivity
                        file under both subject orderings and compare, as done by hand
                        for the cases documented in the NBS parity plan).

Usage:
    python scripts/audit_nbs_results.py
    python scripts/audit_nbs_results.py --roots /path/to/results --out report.md
"""

import argparse
import os
from collections import Counter
from pathlib import Path

import numpy as np

DEFAULT_ROOTS = [
    "~/Connectome/Dev/mrsi_viewer/results",
    "~/Connectome/Dev/mrsitoolbox/results",
    "~/Connectome/Dev/NBS1.2/data",
]

DEFAULT_OUT = "~/Connectome/Dev/nbs_parity/INVALID_RESULTS.md"

_INT_TOL = 1e-6

_VERDICT_LABEL = {
    "matlab_rule": "AFFECTED (scrambled pairing)",
    "legacy_rule": "OK (legacy bct_corr, correct pairing)",
    "clean": "OK (post-fix, tagged subject_order=numeric)",
    "ambiguous": "AMBIGUOUS -- verify manually",
    "unknown": "UNKNOWN -- could not read file",
}


def _is_close_to_int(x: float, tol: float = _INT_TOL) -> bool:
    return abs(x - round(x)) < tol


def classify_pvals(pvals, K: int):
    """Return (verdict, reason) from the corrected-p-value formula each engine uses."""
    if K <= 0 or not pvals:
        return "ambiguous", "no p-values or invalid nperm stored"
    fits_matlab = all(_is_close_to_int(p * K) for p in pvals)
    fits_legacy = all(_is_close_to_int(p * (K + 1) - 1) for p in pvals)
    if fits_matlab and not fits_legacy:
        return "matlab_rule", "p*K is an integer for every component (MATLAB/compat p=b/K rule)"
    if fits_legacy and not fits_matlab:
        return "legacy_rule", "p*(K+1)-1 is an integer for every component (legacy p=(b+1)/(K+1) rule)"
    if fits_matlab and fits_legacy:
        return "ambiguous", "both rules fit exactly (too few components/permutations to distinguish)"
    return "ambiguous", "neither rule fits exactly (rounding, or a p-value was post-processed)"


def find_result_files(roots):
    files = []
    for root in roots:
        root_path = Path(root).expanduser()
        if not root_path.is_dir():
            continue
        files.extend(sorted(root_path.rglob("*_components_all.npz")))
    # De-duplicate in case roots overlap or symlink into each other.
    seen = set()
    unique = []
    for f in files:
        resolved = f.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(f)
    return unique


def load_npz_fields(path: Path):
    try:
        z = np.load(str(path), allow_pickle=True)
    except Exception as exc:  # noqa: BLE001 - report, don't crash the audit
        return None, f"could not load npz ({exc})"
    if "comp_pvals" not in z.files or "nperm" not in z.files:
        return None, "missing comp_pvals or nperm field"
    try:
        pvals = [float(v) for v in np.atleast_1d(z["comp_pvals"])]
        nperm = int(np.asarray(z["nperm"]).item())
    except Exception as exc:  # noqa: BLE001
        return None, f"could not parse comp_pvals/nperm ({exc})"
    engine_tag = str(z["engine"]) if "engine" in z.files else None
    subject_order_tag = str(z["subject_order"]) if "subject_order" in z.files else None
    param_tag = str(z["param_tag"]) if "param_tag" in z.files else ""
    return {
        "pvals": pvals,
        "nperm": nperm,
        "engine_tag": engine_tag,
        "subject_order_tag": subject_order_tag,
        "param_tag": param_tag,
    }, None


def classify_file(path: Path):
    info, err = load_npz_fields(path)
    if info is None:
        return "unknown", err, ""
    if info["subject_order_tag"] == "numeric":
        engine_note = f", engine={info['engine_tag']}" if info["engine_tag"] else ""
        return "clean", f"stamped subject_order=numeric{engine_note}", info["param_tag"]
    verdict, reason = classify_pvals(info["pvals"], info["nperm"])
    return verdict, reason, info["param_tag"]


def write_report(out_path: Path, rows, roots):
    counts = Counter(verdict for _path, verdict, _reason, _tag in rows)
    lines = []
    lines.append("# NBS result audit: subject/design pairing")
    lines.append("")
    lines.append(
        "Generated by `scripts/audit_nbs_results.py`. Read-only report -- no result file "
        "was modified, moved, or deleted by generating it."
    )
    lines.append("")
    lines.append(
        "Background: unpadded `subjectN.txt` exports sort lexicographically, not "
        "numerically, so MATLAB NBS1.2 (and the python `matlab_compat` port before its "
        "fix) paired almost every connectome with the wrong subject's covariates. Legacy "
        "`bct_corr` results are unaffected -- see the NBS parity plan for the "
        "underlying evidence and the fix in `mrsitoolbox/connectomics/nbs.py`."
    )
    lines.append("")
    lines.append(f"Searched roots: {', '.join(str(Path(r).expanduser()) for r in roots)}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    for verdict in ("matlab_rule", "legacy_rule", "clean", "ambiguous", "unknown"):
        if counts.get(verdict):
            lines.append(f"- **{_VERDICT_LABEL[verdict]}**: {counts[verdict]}")
    lines.append(f"- **Total scanned**: {len(rows)}")
    no_sig_count = sum(
        1 for _p, verdict, reason, _t in rows
        if verdict == "ambiguous" and reason == "no p-values or invalid nperm stored"
    )
    if no_sig_count:
        lines.append("")
        lines.append(
            f"Of the ambiguous files, {no_sig_count} simply found zero significant "
            "components -- with no p-values to classify by, the pairing used cannot be "
            "recovered from the file alone, but there is also no reported finding at risk. "
            "The remainder need a manual reconstruction against their source connectivity "
            "file, as done by hand for the cases documented in the NBS parity plan."
        )
    lines.append("")
    lines.append("## Detail")
    lines.append("")
    lines.append("| Verdict | Path | Evidence |")
    lines.append("|---|---|---|")
    # Worst-first ordering so affected files are easy to find.
    order = {"matlab_rule": 0, "ambiguous": 1, "unknown": 2, "legacy_rule": 3, "clean": 4}
    for path, verdict, reason, _tag in sorted(rows, key=lambda r: (order.get(r[1], 9), str(r[0]))):
        lines.append(f"| {_VERDICT_LABEL[verdict]} | `{path}` | {reason} |")
    lines.append("")

    out_path = out_path.expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roots", nargs="+", default=DEFAULT_ROOTS,
                         help="Directories to search recursively for *_components_all.npz.")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Markdown report path.")
    args = parser.parse_args()

    files = find_result_files(args.roots)
    rows = [(f, *classify_file(f)) for f in files]

    out_path = Path(args.out)
    write_report(out_path, rows, args.roots)

    counts = Counter(verdict for _path, verdict, _reason, _tag in rows)
    print(f"Scanned {len(rows)} result file(s) under: "
          f"{', '.join(str(Path(r).expanduser()) for r in args.roots)}")
    for verdict in ("matlab_rule", "legacy_rule", "clean", "ambiguous", "unknown"):
        if counts.get(verdict):
            print(f"  {_VERDICT_LABEL[verdict]}: {counts[verdict]}")
    print(f"Report written to: {out_path.expanduser()}")


if __name__ == "__main__":
    main()
