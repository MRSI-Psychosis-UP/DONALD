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
if [[ "${OS_NAME}" == "Darwin" ]]; then
  SHELL_RC="${HOME}/.zshrc"
else
  SHELL_RC="${HOME}/.bashrc"
fi

DATA_URL="${DONALD_DATA_URL:-${CONNECTOME_VIEWER_DATA_URL:-}}"
DATA_SHA256="${DONALD_DATA_SHA256:-${CONNECTOME_VIEWER_DATA_SHA256:-}}"
DATA_RELEASE_REPO="${DONALD_DATA_REPO:-${CONNECTOME_VIEWER_DATA_REPO:-fedlucchetti/ConnectomeViewer}}"
DATA_RELEASE_TAG="${DONALD_DATA_TAG:-${CONNECTOME_VIEWER_DATA_TAG:-}}"
DATA_SHA256_URL=""
RESOLVED_DATA_RELEASE_TAG=""
RESOLVED_DATA_ASSET_NAME=""
DATA_RELEASE_MARKER="${DATA_DIR}/.donald_data_release_tag"
BIDS_PATH=""

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
3) Configure .env (BIDSDATAPATH + DEVANALYSEPATH)
4) Install launcher + desktop integration

Options:
  --env-name NAME       Conda environment name (default: donald)
  --skip-env            Skip conda environment creation/update
  --data-url URL        Public archive URL containing top-level data/ folder
  --data-sha256 HASH    Optional SHA256 checksum for data archive
  --data-repo OWNER/REPO  GitHub repo used for auto data release lookup
                          (default: fedlucchetti/ConnectomeViewer)
  --data-tag TAG        Use a specific data release tag (default: newest release
                        containing a matching data archive asset)
  --skip-data           Do not download/extract data archive
  --force-data          Re-download and overwrite existing data/ folder
  --bids-path PATH      Set BIDSDATAPATH in .env (default: <repo>/data/BIDS)
  --skip-desktop        Skip desktop integration (Linux .desktop / macOS shortcut)
  --shell-rc FILE       Shell rc file to append PATH update (default auto:
                        ~/.bashrc on Linux, ~/.zshrc on macOS)
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

run_conda_with_solver() {
  local solver="$1"
  shift
  CONDA_CHANNEL_PRIORITY=strict conda "$@" --solver "${solver}"
}

resolve_latest_data_release() {
  [[ -n "${DATA_URL}" ]] && return 0

  need_cmd curl
  need_cmd python3

  local api_url
  if [[ -n "${DATA_RELEASE_TAG}" ]]; then
    api_url="https://api.github.com/repos/${DATA_RELEASE_REPO}/releases/tags/${DATA_RELEASE_TAG}"
  else
    api_url="https://api.github.com/repos/${DATA_RELEASE_REPO}/releases?per_page=20"
  fi

  local release_json
  if ! release_json="$(curl -fsSL -H "Accept: application/vnd.github+json" "${api_url}")"; then
    return 1
  fi

  local parsed
  if ! parsed="$(printf '%s' "${release_json}" | python3 - <<'PY'
import json
import re
import sys

text = sys.stdin.read()
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
    return 1
  fi

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

ensure_conda_env() {
  [[ "${SKIP_ENV}" -eq 0 ]] || return 0
  need_cmd conda
  if [[ ! -f "${ENV_FILE_PATH}" ]]; then
    echo "Missing environment file: ${ENV_FILE_PATH}" >&2
    exit 1
  fi

  if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
    echo "Updating conda env '${ENV_NAME}' from ${ENV_FILE_PATH} ..."
    if ! run_conda_with_solver libmamba env update -n "${ENV_NAME}" -f "${ENV_FILE_PATH}" --prune; then
      echo "libmamba solver failed; falling back to classic solver..."
      run_conda_with_solver classic env update -n "${ENV_NAME}" -f "${ENV_FILE_PATH}" --prune
    fi
  else
    echo "Creating conda env '${ENV_NAME}' from ${ENV_FILE_PATH} ..."
    if ! run_conda_with_solver libmamba env create -n "${ENV_NAME}" -f "${ENV_FILE_PATH}"; then
      echo "libmamba solver failed; falling back to classic solver..."
      run_conda_with_solver classic env create -n "${ENV_NAME}" -f "${ENV_FILE_PATH}"
    fi
  fi
}

download_data_if_needed() {
  [[ "${SKIP_DATA}" -eq 0 ]] || return 0
  if [[ -d "${DATA_DIR}" && "${FORCE_DATA}" -eq 0 ]]; then
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
  if [[ -z "${DATA_URL}" ]]; then
    if resolve_latest_data_release; then
      echo "Resolved data release from ${DATA_RELEASE_REPO} (${RESOLVED_DATA_RELEASE_TAG:-latest})."
      echo "Using data asset: ${RESOLVED_DATA_ASSET_NAME:-$(basename "${DATA_URL%%\?*}")}"
    else
      echo "No data URL configured and auto-resolution failed."
      echo "Set DONALD_DATA_URL, or pass --data-url, or set --data-repo/--data-tag."
      echo "Continuing without downloading data."
      return 0
    fi
  fi

  need_cmd curl
  need_cmd tar
  local tmp_dir
  tmp_dir="$(mktemp -d)"
  local archive_name archive_path
  archive_name="$(basename "${DATA_URL%%\?*}")"
  archive_path="${tmp_dir}/${archive_name}"

  echo "Downloading data archive..."
  curl -L --fail --retry 3 "${DATA_URL}" -o "${archive_path}"

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

resolve_bids_path() {
  if [[ -n "${BIDS_PATH}" ]]; then
    expand_home_path "${BIDS_PATH}"
    return
  fi

  local current
  current="$(read_env_key "${ENV_DOTFILE}" "BIDSDATAPATH")"
  local default_path
  if [[ -n "${current}" ]]; then
    default_path="${current}"
  else
    default_path="${BASE_DIR}/data/BIDS"
  fi

  if [[ "${NON_INTERACTIVE}" -eq 1 || ! -t 0 ]]; then
    printf '%s' "${default_path}"
    return
  fi

  read -rp "BIDSDATAPATH [${default_path}]: " input_path
  if [[ -z "${input_path}" ]]; then
    printf '%s' "${default_path}"
  else
    expand_home_path "${input_path}"
  fi
}

configure_env_file() {
  local bids_value
  bids_value="$(resolve_bids_path)"

  upsert_env_key "${ENV_DOTFILE}" "DEVANALYSEPATH" "${BASE_DIR}"
  upsert_env_key "${ENV_DOTFILE}" "BIDSDATAPATH" "${bids_value}"

  echo "Configured ${ENV_DOTFILE}:"
  echo "  DEVANALYSEPATH=${BASE_DIR}"
  echo "  BIDSDATAPATH=${bids_value}"
}

install_launcher() {
  mkdir -p "${BIN_DIR}"
  cat > "${BIN_LINK}" <<LAUNCHER
#!/usr/bin/env bash
set -euo pipefail
APP_PATH="${APP_PATH}"
ENV_NAME="${ENV_NAME}"

if command -v conda >/dev/null 2>&1; then
  exec conda run --no-capture-output -n "\${ENV_NAME}" python "\${APP_PATH}" "\$@"
fi
exec python "\${APP_PATH}" "\$@"
LAUNCHER
  chmod +x "${BIN_LINK}"

  if ! command -v donald >/dev/null 2>&1; then
    if ! grep -q "${BIN_DIR}" "${SHELL_RC}" 2>/dev/null; then
      {
        echo ""
        echo "# Added by donald installer"
        echo "export PATH=\"${BIN_DIR}:\$PATH\""
      } >> "${SHELL_RC}"
      echo "Added ${BIN_DIR} to PATH in ${SHELL_RC}."
      echo "Run: source ${SHELL_RC}"
    fi
  fi
}

install_desktop_entry() {
  [[ "${SKIP_DESKTOP}" -eq 0 ]] || return 0
  if [[ ! -f "${ICON_PATH}" ]]; then
    echo "Missing icon: ${ICON_PATH}" >&2
    exit 1
  fi

  if [[ "${OS_NAME}" == "Darwin" ]]; then
    local app_dir="${HOME}/Applications"
    local shortcut="${app_dir}/Donald.command"
    mkdir -p "${app_dir}"
    cat > "${shortcut}" <<MACSHORTCUT
#!/usr/bin/env bash
exec "${BIN_LINK}" "\$@"
MACSHORTCUT
    chmod +x "${shortcut}"
    echo "Installed macOS launcher shortcut at ${shortcut}"
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
    --bids-path)
      [[ $# -ge 2 ]] || { echo "Missing value for --bids-path" >&2; exit 1; }
      BIDS_PATH="$2"
      shift 2
      ;;
    --skip-desktop)
      SKIP_DESKTOP=1
      shift
      ;;
    --shell-rc)
      [[ $# -ge 2 ]] || { echo "Missing value for --shell-rc" >&2; exit 1; }
      SHELL_RC="$2"
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
