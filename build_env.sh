#!/bin/bash
set -euo pipefail

ENV_NAME="donald"
ENV_FILE="environment.yaml"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda command not found. Please install Miniconda/Anaconda first."
    exit 1
fi

if [[ ! -f "${SCRIPT_DIR}/${ENV_FILE}" ]]; then
    echo "Missing ${ENV_FILE} in ${SCRIPT_DIR}."
    exit 1
fi

run_conda_classic() {
    CONDA_SOLVER=classic CONDA_CHANNEL_PRIORITY=strict conda "$@"
}

create_or_update_env() {
    cd "$SCRIPT_DIR"
    if run_conda_classic env list | grep -qE "^\s*${ENV_NAME}\s"; then
        echo "Updating '${ENV_NAME}' from ${ENV_FILE}..."
        if ! run_conda_classic env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune --solver classic; then
            echo "Conda update failed."
            echo "If you still see libmamba errors, run: conda config --set solver classic"
            exit 1
        fi
    else
        echo "Creating '${ENV_NAME}' from ${ENV_FILE}..."
        if ! run_conda_classic env create -n "${ENV_NAME}" -f "${ENV_FILE}" --solver classic; then
            echo "Conda create failed."
            echo "If you still see libmamba errors, run: conda config --set solver classic"
            exit 1
        fi
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
