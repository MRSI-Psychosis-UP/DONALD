# DONALD

![Donald](icons/donald_title.png)

Donald is a Qt GUI to load, inspect, aggregate, harmonize, and analyze connectome matrices stored in `.npz` files.

## Table of Contents
- [Platform Support](#platform-support)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Environment Configuration](#environment-configuration)
- [Quick Usage](#quick-usage)
- [Gradient Free-Energy Paths](#gradient-free-energy-paths)
- [Data Releases](#data-releases)
- [Troubleshooting](#troubleshooting)
- [Notes](#notes)

## Platform Support
- Linux: fully supported (`.desktop` launcher installed)
- macOS: supported (`~/Applications/Donald.app` + `~/Applications/Donald.command` installed)

## Prerequisites
- `conda` (Miniconda/Anaconda), initialized in shell
- `bash`, `python`, `curl`, `tar`, `zstd`
- checksum tool: `sha256sum` (Linux) or `shasum` (macOS)
- optional: `unzip` (if release asset is `.zip`)
- optional for NBS: MATLAB executable + NBS toolbox path (configure later in GUI Preferences)

## Installation

```bash
cd mrsi_viewer
./install_donald.sh
```

Useful options:

```bash
./install_donald.sh --help
```
Common flags:
- `--skip-env`
- `--skip-data`
- `--force-data`
- `--data-repo OWNER/REPO`
- `--data-tag data-vYYYYMMDD`
- `--data-url ... --data-sha256 ...` (manual override)
- `--skip-desktop`
- `--non-interactive`
- `--env-name donald`

After first install, if needed:

```bash
source ~/.bashrc   # Linux
# or on macOS:
source ~/.zprofile && source ~/.zshrc
```

Launch:

```bash
donald
```

## Environment Configuration
Installer writes `${REPO}/.env`:
- `DEVANALYSEPATH=<repo root>`

## Quick Usage
1. Open Donald.
2. Add one or more `.npz` files.
3. Pick matrix key and sample/average.
4. Use side panels:
   - `Gradients > Compute`
   - `Selector > Prepare`
   - `Harmonize > Prepare`
   - `NBS > Prepare`
5. Use `Write to File` to export selected matrix.

## Gradient Free-Energy Paths

In the gradient scatter/path workspace, `Write free energy` exports the current saved endpoints, generated paths, path energies, and free energies to a `*_desc-free_energy_paths.npz` file.

After writing that file, Donald can offer a follow-up dialog to run downstream null computations from the `mrsitoolbox` repository:

- `test_rostrocaudal_gradient.py` for Moran/Burt rostrocaudal null statistics.
- `endpoint_neighbor_free_energy_nulls.py` for endpoint-neighbor uncertainty nulls.

The endpoint-neighbor null option is enabled by default in the dialog. It uses `lh,rh` and 32 workers when available. `Shortest only` is available as an opt-in fallback for subjects with pathological simple-path enumeration. The command preview shows the exact commands that will be launched.

## Data Releases
Code repo does not track heavy `data/` payloads.


Installer auto-detects latest matching release asset (`donald_data_*` or legacy `connectome_viewer_data_*`).
Default release repo: `MRSI-Psychosis-UP/DONALD`.

## Troubleshooting
- NBS blocked: set MATLAB executable and NBS path in `Settings > Preferences`.
- `donald: command not found`: reload shell config or open a new terminal.
