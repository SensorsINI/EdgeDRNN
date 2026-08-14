#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

VERILATOR="${VERILATOR:-$(command -v verilator || true)}"
if [[ -z "${VERILATOR}" || ! -x "${VERILATOR}" ]]; then
    echo "Verilator is not available" >&2
    exit 127
fi

"${PYTHON:-python3}" "${SCRIPT_DIR}/validate_stimuli.py" "${STIMULI_DIR}"

"${VERILATOR}" \
    --lint-only \
    --timing \
    --assert \
    -Wall \
    -Wno-SYNCASYNCNET \
    -DSIM_DEBUG \
    -I"${REPO_ROOT}/hdl" \
    -I"${STIMULI_DIR}" \
    --top-module edgedrnn_tb \
    "${RTL_SOURCES[@]}" "${TB_SOURCES[@]}"

echo "Verilator lint passed"
