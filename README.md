# DONALD: Distributed Open-source Network AnaLysis Dashboard

Donald is a Qt GUI to load, inspect, aggregate, harmonize, and analyze connectome matrices stored in `.npz` files.

## Table of Contents
- [Platform Support](#platform-support)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Environment Configuration](#environment-configuration)
- [Quick Usage](#quick-usage)
- [Data Releases](#data-releases)
- [Troubleshooting](#troubleshooting)
- [Notes](#notes)

## Platform Support
- Linux: fully supported (`.desktop` launcher installed)
- macOS: supported (`~/Applications/Donald.command` shortcut installed)

## Prerequisites
- `conda` (Miniconda/Anaconda), initialized in shell
- `bash`, `python3`, `curl`, `tar`, `zstd`
- checksum tool: `sha256sum` (Linux) or `shasum` (macOS)
- optional: `unzip` (if release asset is `.zip`)
- optional for NBS: MATLAB executable + NBS toolbox path (configure later in GUI Preferences)

## Installation

```bash
cd mrsi_viewer
./install_donald.sh
```

What the installer does:
1. Create/update conda env from `environment.yaml` (default env name: `donald`, fast solver `libmamba`, fallback `classic`)
2. Download/update `data/` from GitHub Releases (auto-detect latest compatible release)
3. Write `.env` with `DEVANALYSEPATH` and `BIDSDATAPATH`
4. Install CLI launcher `~/.local/bin/donald`
5. Install desktop integration (`.desktop` on Linux, `.command` shortcut on macOS)

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
- `--bids-path /path/to/BIDS`
- `--skip-desktop`
- `--non-interactive`
- `--env-name donald`

Legacy installer name still works:

```bash
./install_connectome_viewer.sh
```

After first install, if needed:

```bash
source ~/.bashrc   # Linux
# or on macOS:
source ~/.zshrc
```

Launch:

```bash
donald
```

## Environment Configuration
Installer writes `${REPO}/.env`:
- `DEVANALYSEPATH=<repo root>`
- `BIDSDATAPATH=<chosen path>` (default `${REPO}/data/BIDS`)

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

## Data Releases
Code repo does not track heavy `data/` payloads.

Create/publish a data release asset:

```bash
./scripts/publish_data_release.sh --tag data-vYYYYMMDD
GITHUB_TOKEN=... ./scripts/publish_data_release.sh --tag data-vYYYYMMDD --upload
```

Default asset name produced by publisher:
- `donald_data_<tag>.tar.zst`

Installer auto-detects latest matching release asset (`donald_data_*` or legacy `connectome_viewer_data_*`).

## Troubleshooting
- `conda: command not found`: install Miniconda/Anaconda and run `conda init`.
- Data extracted but `data/` missing: archive must contain top-level `data/` directory.
- NBS blocked: set MATLAB executable and NBS path in `Settings > Preferences`.
- `donald: command not found`: reload shell config or open a new terminal.

## Notes
- `setup.py` is no longer used by this install flow.
- `environment.yaml` is the dependency source of truth.
