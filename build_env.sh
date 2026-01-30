#!/bin/bash
set -euo pipefail

ENV_NAME="mrsiviewer"
ENV_FILE="environment.yaml"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda command not found. Please install Miniconda/Anaconda first."
    exit 1
fi

create_or_update_env() {
    cd "$SCRIPT_DIR"
    if conda env list | grep -qE "^\s*${ENV_NAME}\s"; then
        echo "Updating '${ENV_NAME}' from ${ENV_FILE}..."
        conda env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune
    else
        echo "Creating '${ENV_NAME}' from ${ENV_FILE}..."
        conda env create -n "${ENV_NAME}" -f "${ENV_FILE}"
    fi
    echo "Done. Use option 2 to activate."
}

activate_env() {
    # shellcheck disable=SC1091
    conda activate "${ENV_NAME}"
}

echo "Select an option:"
echo "1) Build or update the '${ENV_NAME}' environment"
echo "2) Activate the '${ENV_NAME}' environment"
echo "3) Deactivate the current environment"
read -rp "Enter your choice (1/2/3): " choice

case "$choice" in
    1) create_or_update_env ;;
    2) activate_env ;;
    3) conda deactivate ;;
    *) echo "Invalid choice."; exit 1 ;;
esac
