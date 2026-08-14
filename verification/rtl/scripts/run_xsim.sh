#!/usr/bin/env bash

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

BUILD_DIR="${BUILD_ROOT}/xsim"
prepare_build_directory "${BUILD_DIR}"

if command -v xvlog >/dev/null 2>&1; then
    XVLOG="$(command -v xvlog)"
    XELAB="$(command -v xelab)"
    XSIM="$(command -v xsim)"
else
    VIVADO_HOME="${VIVADO_HOME:-/opt/Xilinx/2026.1/Vivado}"
    XVLOG="${VIVADO_HOME}/bin/xvlog"
    XELAB="${VIVADO_HOME}/bin/xelab"
    XSIM="${VIVADO_HOME}/bin/xsim"
fi

for tool in "${XVLOG}" "${XELAB}" "${XSIM}"; do
    if [[ ! -x "${tool}" ]]; then
        echo "Required xsim tool is not executable: ${tool}" >&2
        exit 127
    fi
done

pushd "${BUILD_DIR}" >/dev/null
"${XVLOG}" -sv -d SIM_DEBUG \
    -i "${REPO_ROOT}/hdl" \
    -i "${STIMULI_DIR}" \
    "${RTL_SOURCES[@]}" "${TB_SOURCES[@]}"
"${XELAB}" edgedrnn_tb -s edgedrnn_tb_sim --debug typical
"${XSIM}" edgedrnn_tb_sim --runall | tee xsim.log
grep -q "RTL_TEST_PASS" xsim.log
popd >/dev/null

echo "xsim verification passed"
