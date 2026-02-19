#!/usr/bin/env bash
set -euo pipefail

OS_NAME="$(uname -s)"
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_PATH="${BASE_DIR}/connectome_viewer.py"
ICON_PATH="${BASE_DIR}/icons/conviewer.png"
ENV_FILE_PATH="${BASE_DIR}/environment.yaml"
ENV_NAME="donald"
ENV_DOTFILE="${BASE_DIR}/.env"
DATA_DIR="${BASE_DIR}/data"

DESKTOP_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
DESKTOP_FILE="${DESKTOP_DIR}/donald.desktop"
BIN_DIR="${HOME}/.local/bin"
BIN_LINK="${BIN_DIR}/donald"
declare -a PATH_RC_FILES
SHELL_RC_OVERRIDE=0
if [[ "${OS_NAME}" == "Darwin" ]]; then
  SHELL_RC="${HOME}/.zshrc"
  PATH_RC_FILES=("${HOME}/.zprofile" "${HOME}/.zshrc")
else
  SHELL_RC="${HOME}/.bashrc"
  PATH_RC_FILES=("${HOME}/.bashrc")
fi

DATA_URL="${DONALD_DATA_URL:-${CONNECTOME_VIEWER_DATA_URL:-}}"
DATA_SHA256="${DONALD_DATA_SHA256:-${CONNECTOME_VIEWER_DATA_SHA256:-}}"
DATA_RELEASE_REPO="${DONALD_DATA_REPO:-${CONNECTOME_VIEWER_DATA_REPO:-MRSI-Psychosis-UP/DONALD}}"
DATA_RELEASE_TAG="${DONALD_DATA_TAG:-${CONNECTOME_VIEWER_DATA_TAG:-}}"
DATA_SHA256_URL=""
RESOLVED_DATA_RELEASE_TAG=""
RESOLVED_DATA_ASSET_NAME=""
DATA_RELEASE_MARKER="${DATA_DIR}/.donald_data_release_tag"

SKIP_ENV=0
SKIP_DATA=0
SKIP_DESKTOP=0
FORCE_DATA=0
NON_INTERACTIVE=0

usage() {
  cat <<'USAGE'
Usage: install_donald.sh [options]

Integrated installer for Donald (Linux + macOS):
1) Create/update conda environment
2) Download/extract data release asset (optional)
3) Configure .env (DEVANALYSEPATH)
4) Install launcher + desktop integration

Options:
  --env-name NAME       Conda environment name (default: donald)
  --skip-env            Skip conda environment creation/update
  --data-url URL        Public archive URL containing top-level data/ folder
  --data-sha256 HASH    Optional SHA256 checksum for data archive
  --data-repo OWNER/REPO  GitHub repo used for auto data release lookup
                          (default: MRSI-Psychosis-UP/DONALD)
  --data-tag TAG        Use a specific data release tag (default: newest release
                        containing a matching data archive asset)
  --skip-data           Do not download/extract data archive
  --force-data          Re-download and overwrite existing data/ folder
  --skip-desktop        Skip desktop integration (Linux .desktop / macOS .app + shortcut)
  --shell-rc FILE       Shell rc file to append PATH update (default auto:
                        ~/.bashrc on Linux, ~/.zprofile + ~/.zshrc on macOS)
  --non-interactive     Do not prompt; use defaults for unanswered values
  -h, --help            Show this help

Environment variables:
  DONALD_DATA_URL
  DONALD_DATA_SHA256
  DONALD_DATA_REPO
  DONALD_DATA_TAG
  CONNECTOME_VIEWER_DATA_URL (deprecated alias)
  CONNECTOME_VIEWER_DATA_SHA256 (deprecated alias)
  CONNECTOME_VIEWER_DATA_REPO (deprecated alias)
  CONNECTOME_VIEWER_DATA_TAG (deprecated alias)
USAGE
}

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

detect_repo_slug() {
  local remote
  remote="$(git -C "${BASE_DIR}" remote get-url origin 2>/dev/null || true)"
  if [[ -z "${remote}" ]]; then
    return 1
  fi
  if [[ "${remote}" =~ ^git@github.com:([^/]+)/([^/]+)(\.git)?$ ]]; then
    echo "${BASH_REMATCH[1]}/${BASH_REMATCH[2]}" | sed 's/\.git$//'
    return 0
  fi
  if [[ "${remote}" =~ ^https://github.com/([^/]+)/([^/]+)(\.git)?$ ]]; then
    echo "${BASH_REMATCH[1]}/${BASH_REMATCH[2]}" | sed 's/\.git$//'
    return 0
  fi
  return 1
}

expand_home_path() {
  local value="$1"
  if [[ "${value}" == ~* ]]; then
    printf '%s' "${HOME}${value:1}"
  else
    printf '%s' "${value}"
  fi
}

extract_archive() {
  local archive_path="$1"
  local target_dir="$2"
  case "${archive_path}" in
    *.tar.gz|*.tgz) tar -xzf "${archive_path}" -C "${target_dir}" ;;
    *.tar.zst|*.tzst)
      if tar --help 2>/dev/null | grep -q -- '--zstd'; then
        tar --zstd -xf "${archive_path}" -C "${target_dir}"
      else
        need_cmd zstd
        zstd -dc "${archive_path}" | tar -xf - -C "${target_dir}"
      fi
      ;;
    *.tar) tar -xf "${archive_path}" -C "${target_dir}" ;;
    *.zip)
      need_cmd unzip
      unzip -q "${archive_path}" -d "${target_dir}"
      ;;
    *)
      echo "Unsupported archive format: ${archive_path}" >&2
      echo "Supported: .tar.gz .tgz .tar.zst .tzst .tar .zip" >&2
      exit 1
      ;;
  esac
}

sha256_of_file() {
  local filepath="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${filepath}" | awk '{print $1}'
    return 0
  fi
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "${filepath}" | awk '{print $1}'
    return 0
  fi
  echo "Missing required command: sha256sum or shasum" >&2
  exit 1
}

verify_sha256() {
  local expected="$1"
  local filepath="$2"
  local actual
  actual="$(sha256_of_file "${filepath}")"
  if [[ "${actual,,}" != "${expected,,}" ]]; then
    echo "SHA256 mismatch for ${filepath}" >&2
    echo "Expected: ${expected}" >&2
    echo "Actual:   ${actual}" >&2
    exit 1
  fi
}

read_env_key() {
  local file="$1"
  local key="$2"
  [[ -f "${file}" ]] || return 0
  local line
  line="$(grep -E "^${key}=" "${file}" | tail -n1 || true)"
  [[ -n "${line}" ]] || return 0
  local value="${line#*=}"
  value="${value%\"}"
  value="${value#\"}"
  value="${value%\'}"
  value="${value#\'}"
  printf '%s' "${value}"
}

upsert_env_key() {
  local file="$1"
  local key="$2"
  local value="$3"
  local escaped="${value//\\/\\\\}"
  escaped="${escaped//\"/\\\"}"

  mkdir -p "$(dirname "${file}")"
  touch "${file}"
  if grep -q -E "^${key}=" "${file}"; then
    local tmp_file
    tmp_file="$(mktemp)"
    awk -v k="${key}" -v v="${escaped}" '
      BEGIN { pat = "^" k "=" }
      $0 ~ pat { print k "=\"" v "\""; next }
      { print }
    ' "${file}" > "${tmp_file}"
    mv "${tmp_file}" "${file}"
  else
    printf '%s="%s"\n' "${key}" "${escaped}" >> "${file}"
  fi
}

ensure_bin_dir_on_path() {
  local files=()
  local rc_file

  if [[ "${SHELL_RC_OVERRIDE}" -eq 1 ]]; then
    files=("${SHELL_RC}")
  else
    files=("${PATH_RC_FILES[@]}")
  fi

  for rc_file in "${files[@]}"; do
    rc_file="$(expand_home_path "${rc_file}")"
    [[ -n "${rc_file}" ]] || continue

    mkdir -p "$(dirname "${rc_file}")"
    touch "${rc_file}"
    if ! grep -Fq "export PATH=\"${BIN_DIR}:\$PATH\"" "${rc_file}" 2>/dev/null; then
      {
        echo ""
        echo "# Added by donald installer"
        echo "export PATH=\"${BIN_DIR}:\$PATH\""
      } >> "${rc_file}"
      echo "Added ${BIN_DIR} to PATH in ${rc_file}."
    fi
  done
}

build_macos_icns() {
  local source_png="$1"
  local output_icns="$2"

  need_cmd sips
  need_cmd iconutil

  local tmp_dir iconset_dir size retina
  tmp_dir="$(mktemp -d)"
  iconset_dir="${tmp_dir}/AppIcon.iconset"
  mkdir -p "${iconset_dir}"

  for size in 16 32 128 256 512; do
    retina=$((size * 2))
    sips -z "${size}" "${size}" "${source_png}" --out "${iconset_dir}/icon_${size}x${size}.png" >/dev/null
    sips -z "${retina}" "${retina}" "${source_png}" --out "${iconset_dir}/icon_${size}x${size}@2x.png" >/dev/null
  done

  if ! iconutil -c icns "${iconset_dir}" -o "${output_icns}"; then
    rm -rf "${tmp_dir}"
    echo "Failed to generate macOS app icon from ${source_png}." >&2
    exit 1
  fi
  rm -rf "${tmp_dir}"
}

install_macos_bundle() {
  local app_dir="${HOME}/Applications"
  local app_bundle="${app_dir}/Donald.app"
  local app_contents="${app_bundle}/Contents"
  local app_macos="${app_contents}/MacOS"
  local app_resources="${app_contents}/Resources"
  local app_exec="${app_macos}/donald"
  local app_plist="${app_contents}/Info.plist"
  local app_icon="${app_resources}/AppIcon.icns"
  local shortcut="${app_dir}/Donald.command"

  mkdir -p "${app_macos}" "${app_resources}"

  cat > "${app_exec}" <<MACAPP
#!/usr/bin/env bash
set -euo pipefail
exec "${BIN_LINK}" "\$@"
MACAPP
  chmod +x "${app_exec}"

  build_macos_icns "${ICON_PATH}" "${app_icon}"

  cat > "${app_plist}" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>
  <string>Donald</string>
  <key>CFBundleDisplayName</key>
  <string>Donald</string>
  <key>CFBundleIdentifier</key>
  <string>org.connectome.donald</string>
  <key>CFBundleVersion</key>
  <string>1.0</string>
  <key>CFBundleShortVersionString</key>
  <string>1.0</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleExecutable</key>
  <string>donald</string>
  <key>CFBundleIconFile</key>
  <string>AppIcon</string>
  <key>NSHighResolutionCapable</key>
  <true/>
</dict>
</plist>
PLIST

  cat > "${shortcut}" <<MACSHORTCUT
#!/usr/bin/env bash
exec "${BIN_LINK}" "\$@"
MACSHORTCUT
  chmod +x "${shortcut}"
  touch "${app_bundle}"

  echo "Installed macOS app bundle at ${app_bundle}"
  echo "Installed macOS launcher shortcut at ${shortcut}"
}

probe_release_asset_for_tag() {
  local tag="$1"
  [[ -n "${tag}" ]] || return 1
  local candidates=(
    "donald_data_${tag}.tar.zst"
    "connectome_viewer_data_${tag}.tar.zst"
    "donald_data_${tag}.tar.gz"
    "connectome_viewer_data_${tag}.tar.gz"
    "donald_data_${tag}.zip"
    "connectome_viewer_data_${tag}.zip"
    "donald_data_${tag}.tar"
    "connectome_viewer_data_${tag}.tar"
  )

  local asset_name asset_url
  for asset_name in "${candidates[@]}"; do
    asset_url="https://github.com/${DATA_RELEASE_REPO}/releases/download/${tag}/${asset_name}"
    if curl -fsIL "${asset_url}" >/dev/null 2>&1; then
      DATA_URL="${asset_url}"
      RESOLVED_DATA_RELEASE_TAG="${tag}"
      RESOLVED_DATA_ASSET_NAME="${asset_name}"
      local sha_url="${asset_url}.sha256"
      if curl -fsIL "${sha_url}" >/dev/null 2>&1; then
        DATA_SHA256_URL="${sha_url}"
      fi
      return 0
    fi
  done
  return 1
}

resolve_latest_data_release_from_repo() {
  [[ -n "${DATA_URL}" ]] && return 0

  need_cmd curl
  need_cmd python3

  if [[ -n "${DATA_RELEASE_TAG}" ]]; then
    if probe_release_asset_for_tag "${DATA_RELEASE_TAG}"; then
      return 0
    fi
  fi

  local api_url
  if [[ -n "${DATA_RELEASE_TAG}" ]]; then
    api_url="https://api.github.com/repos/${DATA_RELEASE_REPO}/releases/tags/${DATA_RELEASE_TAG}"
  else
    api_url="https://api.github.com/repos/${DATA_RELEASE_REPO}/releases?per_page=20"
  fi

  local release_json
  if ! release_json="$(curl -fsSL -H "Accept: application/vnd.github+json" "${api_url}")"; then
    if [[ -z "${DATA_RELEASE_TAG}" ]]; then
      local latest_url latest_tag
      latest_url="$(curl -fsSL -o /dev/null -w '%{url_effective}' "https://github.com/${DATA_RELEASE_REPO}/releases/latest" || true)"
      if [[ -n "${latest_url}" && "${latest_url}" == *"/releases/tag/"* ]]; then
        latest_tag="${latest_url##*/}"
        if probe_release_asset_for_tag "${latest_tag}"; then
          return 0
        fi
      fi
    fi
    echo "Warning: could not query release metadata from ${api_url}" >&2
    return 1
  fi

  local json_file
  json_file="$(mktemp)"
  printf '%s' "${release_json}" > "${json_file}"
  local parsed
  if ! parsed="$(python3 - "${json_file}" <<'PY'
import json
from pathlib import Path
import re
import sys

json_path = sys.argv[1]
text = Path(json_path).read_text(encoding="utf-8")
if not text.strip():
    raise SystemExit(1)

obj = json.loads(text)
releases = obj if isinstance(obj, list) else [obj]
archive_re = re.compile(r"^(donald_data|connectome_viewer_data)_.*\.(tar\.zst|tar\.gz|tgz|zip|tar)$")

def pick_release():
    for rel in releases:
        assets = rel.get("assets") or []
        archives = [a for a in assets if archive_re.match(a.get("name", ""))]
        if not archives:
            continue
        archive = archives[0]
        sha = None
        exact_sha_name = archive.get("name", "") + ".sha256"
        for item in assets:
            name = item.get("name", "")
            if name == exact_sha_name:
                sha = item
                break
        if sha is None:
            for item in assets:
                name = item.get("name", "")
                if name.endswith(".sha256") and archive.get("name", "") in name:
                    sha = item
                    break
        return rel, archive, sha
    return None

picked = pick_release()
if not picked:
    raise SystemExit(2)

rel, archive, sha = picked
print(f"ARCHIVE_URL={archive.get('browser_download_url', '')}")
print(f"ARCHIVE_NAME={archive.get('name', '')}")
print(f"RELEASE_TAG={rel.get('tag_name', '')}")
if sha:
    print(f"SHA_URL={sha.get('browser_download_url', '')}")
PY
)"; then
    rm -f "${json_file}"
    echo "Warning: release metadata was fetched for ${DATA_RELEASE_REPO}, but no compatible data asset was detected." >&2
    return 1
  fi
  rm -f "${json_file}"

  local key value
  while IFS='=' read -r key value; do
    case "${key}" in
      ARCHIVE_URL) DATA_URL="${value}" ;;
      ARCHIVE_NAME) RESOLVED_DATA_ASSET_NAME="${value}" ;;
      RELEASE_TAG) RESOLVED_DATA_RELEASE_TAG="${value}" ;;
      SHA_URL) DATA_SHA256_URL="${value}" ;;
    esac
  done <<< "${parsed}"

  [[ -n "${DATA_URL}" ]]
}

resolve_latest_data_release() {
  local primary_repo="${DATA_RELEASE_REPO}"
  local legacy_repo="fedlucchetti/ConnectomeViewer"

  if resolve_latest_data_release_from_repo; then
    return 0
  fi

  if [[ "${primary_repo}" != "${legacy_repo}" ]]; then
    echo "Warning: no compatible data asset found in ${primary_repo}; trying fallback repo ${legacy_repo}."
    DATA_RELEASE_REPO="${legacy_repo}"
    if resolve_latest_data_release_from_repo; then
      return 0
    fi
  fi

  DATA_RELEASE_REPO="${primary_repo}"
  return 1
}

ensure_conda_env() {
  [[ "${SKIP_ENV}" -eq 0 ]] || return 0
  need_cmd conda
  if [[ ! -f "${ENV_FILE_PATH}" ]]; then
    echo "Missing environment file: ${ENV_FILE_PATH}" >&2
    exit 1
  fi

  if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    echo "Updating conda env '${ENV_NAME}' from ${ENV_FILE_PATH} ..."
    if ! conda env update -n "${ENV_NAME}" -f "${ENV_FILE_PATH}" --prune; then
      echo "Conda failed to update env '${ENV_NAME}' from ${ENV_FILE_PATH}." >&2
      echo "No alternative solver fallback is used by this installer." >&2
      echo "Fix the conda error above and re-run install_donald.sh." >&2
      exit 1
    fi
  else
    echo "Creating conda env '${ENV_NAME}' from ${ENV_FILE_PATH} ..."
    if ! conda env create -n "${ENV_NAME}" -f "${ENV_FILE_PATH}"; then
      echo "Conda failed to create env '${ENV_NAME}' from ${ENV_FILE_PATH}." >&2
      echo "No alternative solver fallback is used by this installer." >&2
      echo "Fix the conda error above and re-run install_donald.sh." >&2
      exit 1
    fi
  fi
}

download_data_if_needed() {
  [[ "${SKIP_DATA}" -eq 0 ]] || return 0
  local data_dir_has_files=0
  local data_has_bids=0
  if [[ -d "${DATA_DIR}" && -n "$(find "${DATA_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    data_dir_has_files=1
  fi
  if [[ -d "${DATA_DIR}/BIDS" ]]; then
    data_has_bids=1
  fi
  if [[ "${data_dir_has_files}" -eq 1 && "${FORCE_DATA}" -eq 0 ]]; then
    if [[ -n "${DATA_URL}" ]]; then
      echo "Data folder already exists: ${DATA_DIR} (skip download). Use --force-data to refresh."
      return 0
    fi
    if resolve_latest_data_release; then
      local installed_tag=""
      if [[ -f "${DATA_RELEASE_MARKER}" ]]; then
        installed_tag="$(tr -d '[:space:]' < "${DATA_RELEASE_MARKER}" || true)"
      fi
      if [[ -n "${installed_tag}" && -n "${RESOLVED_DATA_RELEASE_TAG}" && "${installed_tag}" == "${RESOLVED_DATA_RELEASE_TAG}" ]]; then
        echo "Data folder already at latest known release (${installed_tag})."
        return 0
      fi
      echo "Data update detected: local='${installed_tag:-unknown}' remote='${RESOLVED_DATA_RELEASE_TAG:-unknown}'."
      echo "Refreshing local data folder..."
      FORCE_DATA=1
    else
      echo "Data folder already exists: ${DATA_DIR} (skip download; could not resolve latest release)."
      return 0
    fi
  fi
  if [[ -d "${DATA_DIR}" && "${data_dir_has_files}" -eq 0 ]]; then
    echo "Data folder exists but is empty: ${DATA_DIR}. Downloading release asset..."
  fi
  if [[ -z "${DATA_URL}" ]]; then
    if resolve_latest_data_release; then
      echo "Resolved data release from ${DATA_RELEASE_REPO} (${RESOLVED_DATA_RELEASE_TAG:-latest})."
      echo "Using data asset: ${RESOLVED_DATA_ASSET_NAME:-$(basename "${DATA_URL%%\?*}")}"
    else
      echo "No data URL configured and auto-resolution failed."
      echo "Tried release repo(s): ${DATA_RELEASE_REPO} and fallback fedlucchetti/ConnectomeViewer."
      echo "Set DONALD_DATA_URL, or pass --data-url, or set --data-repo/--data-tag."
      if [[ "${FORCE_DATA}" -eq 1 || "${data_dir_has_files}" -eq 0 || "${data_has_bids}" -eq 0 ]]; then
        echo "Data download was requested but no valid release asset was found. Aborting." >&2
        exit 1
      fi
      echo "Keeping existing local data at ${DATA_DIR}."
      return 0
    fi
  fi
  if [[ -n "${DATA_URL}" && -z "${DATA_SHA256}" && -z "${DATA_SHA256_URL}" ]]; then
    DATA_SHA256_URL="${DATA_URL}.sha256"
  fi

  need_cmd curl
  need_cmd tar
  local tmp_dir
  tmp_dir="$(mktemp -d)"
  local archive_name archive_path
  archive_name="$(basename "${DATA_URL%%\?*}")"
  archive_path="${tmp_dir}/${archive_name}"

  echo "Downloading data archive..."
  if ! curl -L --fail --retry 3 "${DATA_URL}" -o "${archive_path}"; then
    local alt_url=""
    if [[ "${DATA_URL}" == *"connectome_viewer_data_"* ]]; then
      alt_url="${DATA_URL/connectome_viewer_data_/donald_data_}"
    elif [[ "${DATA_URL}" == *"donald_data_"* ]]; then
      alt_url="${DATA_URL/donald_data_/connectome_viewer_data_}"
    fi
    if [[ -n "${alt_url}" && "${alt_url}" != "${DATA_URL}" ]]; then
      echo "Primary data URL failed. Retrying with alternate asset prefix..."
      echo "Retry URL: ${alt_url}"
      if curl -L --fail --retry 3 "${alt_url}" -o "${archive_path}"; then
        DATA_URL="${alt_url}"
      else
        echo "Failed to download data archive from both URLs." >&2
        echo "Check your release tag and asset name on GitHub Releases." >&2
        rm -rf "${tmp_dir}"
        exit 1
      fi
    else
      echo "Failed to download data archive from URL: ${DATA_URL}" >&2
      echo "Check your release tag and asset name on GitHub Releases." >&2
      rm -rf "${tmp_dir}"
      exit 1
    fi
  fi

  if [[ -z "${DATA_SHA256}" && -n "${DATA_SHA256_URL}" ]]; then
    local sha_path parsed_sha
    sha_path="${tmp_dir}/$(basename "${DATA_SHA256_URL%%\?*}")"
    echo "Downloading checksum file..."
    if curl -L --fail --retry 3 "${DATA_SHA256_URL}" -o "${sha_path}"; then
      parsed_sha="$(awk 'NF > 0 { print $1; exit }' "${sha_path}" || true)"
      if [[ "${parsed_sha}" =~ ^[0-9a-fA-F]{64}$ ]]; then
        DATA_SHA256="${parsed_sha}"
      else
        echo "Warning: could not parse SHA256 from ${DATA_SHA256_URL}; skipping checksum verification."
      fi
    else
      echo "Warning: failed to download checksum file ${DATA_SHA256_URL}; skipping checksum verification."
    fi
  fi

  if [[ -n "${DATA_SHA256}" ]]; then
    verify_sha256 "${DATA_SHA256}" "${archive_path}"
  fi

  if [[ "${FORCE_DATA}" -eq 1 && -d "${DATA_DIR}" ]]; then
    rm -rf "${DATA_DIR}"
  fi

  echo "Extracting data archive..."
  extract_archive "${archive_path}" "${BASE_DIR}"
  rm -rf "${tmp_dir}"

  if [[ ! -d "${DATA_DIR}" ]]; then
    echo "Archive extracted, but ${DATA_DIR} not found." >&2
    echo "Make sure your archive contains a top-level 'data/' folder." >&2
    exit 1
  fi
  if [[ -n "${RESOLVED_DATA_RELEASE_TAG}" ]]; then
    printf '%s\n' "${RESOLVED_DATA_RELEASE_TAG}" > "${DATA_RELEASE_MARKER}"
  elif [[ -n "${DATA_RELEASE_TAG}" ]]; then
    printf '%s\n' "${DATA_RELEASE_TAG}" > "${DATA_RELEASE_MARKER}"
  fi
  echo "Data installed at ${DATA_DIR}"
}

configure_env_file() {
  upsert_env_key "${ENV_DOTFILE}" "DEVANALYSEPATH" "${BASE_DIR}"

  echo "Configured ${ENV_DOTFILE}:"
  echo "  DEVANALYSEPATH=${BASE_DIR}"
}

install_launcher() {
  local conda_hint=""
  if command -v conda >/dev/null 2>&1; then
    conda_hint="$(command -v conda)"
  fi

  mkdir -p "${BIN_DIR}"
  cat > "${BIN_LINK}" <<LAUNCHER
#!/usr/bin/env bash
set -euo pipefail
APP_PATH="${APP_PATH}"
ENV_NAME="${ENV_NAME}"
CONDA_HINT="${conda_hint}"

CONDA_BIN=""
if [[ -n "\${CONDA_HINT}" && -x "\${CONDA_HINT}" ]]; then
  CONDA_BIN="\${CONDA_HINT}"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BIN="\$(command -v conda)"
else
  for candidate in \
    "\${HOME}/miniconda3/bin/conda" \
    "\${HOME}/anaconda3/bin/conda" \
    "\${HOME}/mambaforge/bin/conda" \
    "\${HOME}/miniforge3/bin/conda"; do
    if [[ -x "\${candidate}" ]]; then
      CONDA_BIN="\${candidate}"
      break
    fi
  done
fi

if [[ -n "\${CONDA_BIN}" ]]; then
  exec "\${CONDA_BIN}" run --no-capture-output -n "\${ENV_NAME}" python "\${APP_PATH}" "\$@"
fi

for candidate_python in \
  "\${HOME}/miniconda3/envs/\${ENV_NAME}/bin/python" \
  "\${HOME}/anaconda3/envs/\${ENV_NAME}/bin/python" \
  "\${HOME}/mambaforge/envs/\${ENV_NAME}/bin/python" \
  "\${HOME}/miniforge3/envs/\${ENV_NAME}/bin/python"; do
  if [[ -x "\${candidate_python}" ]]; then
    exec "\${candidate_python}" "\${APP_PATH}" "\$@"
  fi
done

if command -v python3 >/dev/null 2>&1; then
  exec python3 "\${APP_PATH}" "\$@"
fi
if command -v python >/dev/null 2>&1; then
  exec python "\${APP_PATH}" "\$@"
fi

echo "Unable to launch Donald: conda and python were not found." >&2
echo "Expected conda env: \${ENV_NAME}" >&2
exit 1
LAUNCHER
  chmod +x "${BIN_LINK}"
  ensure_bin_dir_on_path

  if ! command -v donald >/dev/null 2>&1; then
    echo "donald is installed at ${BIN_LINK}"
    echo "Open a new terminal, or run: source ${SHELL_RC}"
  fi
}

install_desktop_entry() {
  [[ "${SKIP_DESKTOP}" -eq 0 ]] || return 0
  if [[ ! -f "${ICON_PATH}" ]]; then
    echo "Missing icon: ${ICON_PATH}" >&2
    exit 1
  fi

  if [[ "${OS_NAME}" == "Darwin" ]]; then
    install_macos_bundle
    return 0
  fi

  mkdir -p "${DESKTOP_DIR}"
  cat > "${DESKTOP_FILE}" <<DESKTOP
[Desktop Entry]
Type=Application
Name=Donald
Comment=View connectome similarity matrices
Exec=${BIN_LINK}
Icon=${ICON_PATH}
Terminal=false
StartupNotify=true
StartupWMClass=donald
Categories=Science;Utility;
DESKTOP

  echo "Installed desktop entry at ${DESKTOP_FILE}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      [[ $# -ge 2 ]] || { echo "Missing value for --env-name" >&2; exit 1; }
      ENV_NAME="$2"
      shift 2
      ;;
    --skip-env)
      SKIP_ENV=1
      shift
      ;;
    --data-url)
      [[ $# -ge 2 ]] || { echo "Missing value for --data-url" >&2; exit 1; }
      DATA_URL="$2"
      shift 2
      ;;
    --data-sha256)
      [[ $# -ge 2 ]] || { echo "Missing value for --data-sha256" >&2; exit 1; }
      DATA_SHA256="$2"
      shift 2
      ;;
    --data-repo)
      [[ $# -ge 2 ]] || { echo "Missing value for --data-repo" >&2; exit 1; }
      DATA_RELEASE_REPO="$2"
      shift 2
      ;;
    --data-tag)
      [[ $# -ge 2 ]] || { echo "Missing value for --data-tag" >&2; exit 1; }
      DATA_RELEASE_TAG="$2"
      shift 2
      ;;
    --skip-data)
      SKIP_DATA=1
      shift
      ;;
    --force-data)
      FORCE_DATA=1
      shift
      ;;
    --skip-desktop)
      SKIP_DESKTOP=1
      shift
      ;;
    --shell-rc)
      [[ $# -ge 2 ]] || { echo "Missing value for --shell-rc" >&2; exit 1; }
      SHELL_RC="$2"
      SHELL_RC_OVERRIDE=1
      PATH_RC_FILES=("${SHELL_RC}")
      shift 2
      ;;
    --non-interactive)
      NON_INTERACTIVE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "${APP_PATH}" ]]; then
  echo "Missing app: ${APP_PATH}" >&2
  exit 1
fi

ensure_conda_env
download_data_if_needed
configure_env_file
install_launcher
install_desktop_entry

echo "Installation complete."
echo "Launcher: ${BIN_LINK}"
echo "CLI launch: donald"
if [[ "${OS_NAME}" == "Darwin" ]]; then
  echo "macOS app: ${HOME}/Applications/Donald.app"
  echo "macOS shortcut: ${HOME}/Applications/Donald.command"
fi
if [[ "${SKIP_ENV}" -eq 0 ]]; then
  echo "Conda env: ${ENV_NAME}"
fi

# Backward-compat alias for legacy command name.
if [[ -e "${BIN_DIR}/connectome_viewer" || -L "${BIN_DIR}/connectome_viewer" ]]; then
  rm -f "${BIN_DIR}/connectome_viewer"
fi
ln -s "${BIN_LINK}" "${BIN_DIR}/connectome_viewer"
