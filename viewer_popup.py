"""
Helper for launching the legacy PyQt5 viewer (gui_v01.py) in a separate process.

The viewer understands the environment variable MRSI_ENTRY_PATH and will try to
load that NIfTI volume on startup. We set BIDSDATAPATH as well so the viewer can
re-use the same BIDS root if it needs it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


def launch_session_viewer(entry_file: Path, bids_root: Optional[Path] = None) -> None:
    """Start gui_v01.py in a separate process for the given NIfTI path."""
    script_path = Path(__file__).with_name("gui_v01.py")
    if not script_path.exists():
        raise FileNotFoundError(f"Viewer script not found at {script_path}")

    env = os.environ.copy()
    env["MRSI_ENTRY_PATH"] = str(entry_file)
    if bids_root:
        env["BIDSDATAPATH"] = str(bids_root)

    cmd = [sys.executable, str(script_path)]
    subprocess.Popen(cmd, env=env, cwd=str(script_path.parent))


__all__ = ["launch_session_viewer"]
