#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

BUILD_DIR="${BUILD_ROOT}/verilator"
prepare_build_directory "${BUILD_DIR}"

VERILATOR="${VERILATOR:-$(command -v verilator || true)}"
VERILATOR_JOBS="${VERILATOR_JOBS:-4}"
if [[ -z "${VERILATOR}" || ! -x "${VERILATOR}" ]]; then
    echo "Verilator is not available" >&2
    exit 127
fi

pushd "${BUILD_DIR}" >/dev/null
"${VERILATOR}" \
    --binary \
    --timing \
    --assert \
    -Wall \
    -Wno-SYNCASYNCNET \
    -j "${VERILATOR_JOBS}" \
    -DSIM_DEBUG \
    -I"${REPO_ROOT}/hdl" \
    -I"${STIMULI_DIR}" \
    --Mdir obj_dir \
    --top-module edgedrnn_tb \
    "${RTL_SOURCES[@]}" "${TB_SOURCES[@]}"
./obj_dir/Vedgedrnn_tb | tee verilator.log
grep -q "RTL_TEST_PASS" verilator.log
popd >/dev/null

echo "Verilator verification passed"
