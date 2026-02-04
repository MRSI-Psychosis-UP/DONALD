#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_PATH="${BASE_DIR}/connectome_viewer.py"
ICON_PATH="${BASE_DIR}/icons/conviewer.png"
DESKTOP_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
DESKTOP_FILE="${DESKTOP_DIR}/connectome_viewer.desktop"
BIN_DIR="${HOME}/.local/bin"
BIN_LINK="${BIN_DIR}/connectome_viewer"
SHELL_RC="${HOME}/.bashrc"

if [[ ! -f "${APP_PATH}" ]]; then
  echo "Missing app: ${APP_PATH}" >&2
  exit 1
fi

if [[ ! -f "${ICON_PATH}" ]]; then
  echo "Missing icon: ${ICON_PATH}" >&2
  exit 1
fi

mkdir -p "${DESKTOP_DIR}"
chmod +x "${APP_PATH}"

mkdir -p "${BIN_DIR}"
ln -sf "${APP_PATH}" "${BIN_LINK}"

if ! command -v connectome_viewer >/dev/null 2>&1; then
  if ! grep -q "${BIN_DIR}" "${SHELL_RC}" 2>/dev/null; then
    {
      echo ""
      echo "# Added by connectome_viewer installer"
      echo "export PATH=\"${BIN_DIR}:\$PATH\""
    } >> "${SHELL_RC}"
    echo "Added ${BIN_DIR} to PATH in ${SHELL_RC}."
    echo "Run: source ${SHELL_RC}"
  fi
fi

cat > "${DESKTOP_FILE}" <<EOF
[Desktop Entry]
Type=Application
Name=Connectome Viewer
Comment=View connectome similarity matrices
Exec=${APP_PATH}
Icon=${ICON_PATH}
Terminal=false
Categories=Science;Utility;
EOF

echo "Installed desktop entry at ${DESKTOP_FILE}"
echo "You can now launch it from your app menu and pin it to the taskbar."
echo "CLI launch: connectome_viewer"
