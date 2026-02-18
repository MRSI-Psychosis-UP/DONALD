#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${BASE_DIR}/data"
OUT_ROOT="${BASE_DIR}/dist/releases"
TAG="data-v$(date +%Y%m%d)"
RELEASE_NAME="Donald Data"
REPO_SLUG="${GITHUB_REPOSITORY:-}"
UPLOAD=0
OVERWRITE=0

usage() {
  cat <<USAGE
Usage: scripts/publish_data_release.sh [options]

Options:
  --tag TAG              Release tag (default: ${TAG})
  --name NAME            Release name (default: "${RELEASE_NAME}")
  --repo OWNER/REPO      GitHub repo slug; auto-detected from origin if omitted
  --upload               Create GitHub Release and upload assets (requires GITHUB_TOKEN)
  --overwrite            Delete release assets with same name if present
  -h, --help             Show help

Environment:
  GITHUB_TOKEN           GitHub token with repo contents/release permissions
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

guess_content_type() {
  local file="$1"
  case "${file}" in
    *.zst) echo "application/zstd" ;;
    *.gz) echo "application/gzip" ;;
    *.tar) echo "application/x-tar" ;;
    *.zip) echo "application/zip" ;;
    *.sha256) echo "text/plain" ;;
    *) echo "application/octet-stream" ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)
      [[ $# -ge 2 ]] || { echo "Missing value for --tag" >&2; exit 1; }
      TAG="$2"
      shift 2
      ;;
    --name)
      [[ $# -ge 2 ]] || { echo "Missing value for --name" >&2; exit 1; }
      RELEASE_NAME="$2"
      shift 2
      ;;
    --repo)
      [[ $# -ge 2 ]] || { echo "Missing value for --repo" >&2; exit 1; }
      REPO_SLUG="$2"
      shift 2
      ;;
    --upload)
      UPLOAD=1
      shift
      ;;
    --overwrite)
      OVERWRITE=1
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

need_cmd tar
need_cmd zstd
need_cmd sha256sum
need_cmd jq
need_cmd curl

if [[ ! -d "${DATA_DIR}" ]]; then
  echo "Missing data directory: ${DATA_DIR}" >&2
  exit 1
fi

if [[ -z "${REPO_SLUG}" ]]; then
  REPO_SLUG="$(detect_repo_slug || true)"
fi

if [[ -z "${REPO_SLUG}" ]]; then
  echo "Could not detect GitHub repo slug. Use --repo OWNER/REPO." >&2
  exit 1
fi

OUT_DIR="${OUT_ROOT}/${TAG}"
mkdir -p "${OUT_DIR}"

ARCHIVE_NAME="donald_data_${TAG}.tar.zst"
ARCHIVE_PATH="${OUT_DIR}/${ARCHIVE_NAME}"
SHA_PATH="${ARCHIVE_PATH}.sha256"

# Create an archive with top-level data/ folder.
# shellcheck disable=SC2086
tar -C "${BASE_DIR}" --use-compress-program=zstd -cf "${ARCHIVE_PATH}" data
(
  cd "${OUT_DIR}"
  sha256sum "${ARCHIVE_NAME}" > "${SHA_PATH}"
)

echo "[INFO] Data archive created: ${ARCHIVE_PATH}"
echo "[INFO] Checksum file:       ${SHA_PATH}"

auth_header=()
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
  auth_header=(-H "Authorization: Bearer ${GITHUB_TOKEN}")
fi

if [[ "${UPLOAD}" -eq 0 ]]; then
  echo "[INFO] Upload skipped. Re-run with --upload to publish GitHub Release assets."
  exit 0
fi

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "GITHUB_TOKEN is required for --upload." >&2
  exit 1
fi

API_BASE="https://api.github.com/repos/${REPO_SLUG}"
CREATE_PAYLOAD="$(jq -n --arg tag "${TAG}" --arg name "${RELEASE_NAME}" '{tag_name:$tag,name:$name,draft:false,prerelease:false,generate_release_notes:true}')"
RELEASE_JSON="${OUT_DIR}/release.json"

create_code="$(curl -sS -o "${RELEASE_JSON}" -w "%{http_code}" \
  "${auth_header[@]}" \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  -X POST "${API_BASE}/releases" \
  -d "${CREATE_PAYLOAD}")"

if [[ "${create_code}" == "201" ]]; then
  echo "[INFO] Created new release for tag ${TAG}."
elif [[ "${create_code}" == "422" ]]; then
  get_code="$(curl -sS -o "${RELEASE_JSON}" -w "%{http_code}" \
    "${auth_header[@]}" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    "${API_BASE}/releases/tags/${TAG}")"
  if [[ "${get_code}" != "200" ]]; then
    echo "Failed to fetch existing release for tag ${TAG}. HTTP ${get_code}" >&2
    cat "${RELEASE_JSON}" >&2 || true
    exit 1
  fi
  echo "[INFO] Using existing release for tag ${TAG}."
else
  echo "Failed to create release. HTTP ${create_code}" >&2
  cat "${RELEASE_JSON}" >&2 || true
  exit 1
fi

UPLOAD_URL="$(jq -r '.upload_url' "${RELEASE_JSON}" | sed 's/{?name,label}$//')"
RELEASE_ID="$(jq -r '.id' "${RELEASE_JSON}")"
HTML_URL="$(jq -r '.html_url' "${RELEASE_JSON}")"
if [[ -z "${UPLOAD_URL}" || "${UPLOAD_URL}" == "null" ]]; then
  echo "Release upload URL missing." >&2
  exit 1
fi

if [[ "${OVERWRITE}" -eq 1 ]]; then
  for file in "${ARCHIVE_PATH}" "${SHA_PATH}"; do
    name="$(basename "${file}")"
    asset_id="$(jq -r --arg n "${name}" '.assets[]? | select(.name==$n) | .id' "${RELEASE_JSON}" | head -n1)"
    if [[ -n "${asset_id}" && "${asset_id}" != "null" ]]; then
      del_code="$(curl -sS -o /dev/null -w "%{http_code}" \
        "${auth_header[@]}" \
        -H "Accept: application/vnd.github+json" \
        -H "X-GitHub-Api-Version: 2022-11-28" \
        -X DELETE "${API_BASE}/releases/assets/${asset_id}")"
      if [[ "${del_code}" != "204" ]]; then
        echo "Failed to delete existing asset ${name} (HTTP ${del_code})." >&2
        exit 1
      fi
    fi
  done
fi

for file in "${ARCHIVE_PATH}" "${SHA_PATH}"; do
  name="$(basename "${file}")"
  type="$(guess_content_type "${file}")"
  upload_code="$(curl -sS -o "${OUT_DIR}/upload_${name}.json" -w "%{http_code}" \
    "${auth_header[@]}" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" \
    -H "Content-Type: ${type}" \
    -X POST "${UPLOAD_URL}?name=${name}" \
    --data-binary @"${file}")"
  if [[ "${upload_code}" != "201" ]]; then
    echo "Failed to upload ${name}. HTTP ${upload_code}" >&2
    cat "${OUT_DIR}/upload_${name}.json" >&2 || true
    exit 1
  fi
  echo "[INFO] Uploaded asset: ${name}"
done

echo "[SUCCESS] Release published: ${HTML_URL}"
echo "[INFO] Use this URL in installer:"
echo "        https://github.com/${REPO_SLUG}/releases/download/${TAG}/${ARCHIVE_NAME}"
