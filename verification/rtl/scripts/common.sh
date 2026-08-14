#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VERIFY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${VERIFY_ROOT}/../.." && pwd)"
STIMULI_DIR="${STIMULI_DIR:-${VERIFY_ROOT}/stimuli}"
BUILD_ROOT="${BUILD_ROOT:-${VERIFY_ROOT}/build}"

RTL_SOURCES=(
    "${REPO_ROOT}/hdl/edgedrnn_pkg.sv"
    "${REPO_ROOT}/hdl/axis_fifo.sv"
    "${REPO_ROOT}/hdl/bram_sdp.sv"
    "${REPO_ROOT}/hdl/mem_acc.sv"
    "${REPO_ROOT}/hdl/mem_state.sv"
    "${REPO_ROOT}/hdl/sigmoid_lut.sv"
    "${REPO_ROOT}/hdl/tanh_lut.sv"
    "${REPO_ROOT}/hdl/hpe.sv"
    "${REPO_ROOT}/hdl/hpc.sv"
    "${REPO_ROOT}/hdl/ieu.sv"
    "${REPO_ROOT}/hdl/edgedrnn.sv"
)

TB_SOURCES=(
    "${VERIFY_ROOT}/tb/interfaces/axis_stream_if.sv"
    "${VERIFY_ROOT}/tb/models/axis_file_source.sv"
    "${VERIFY_ROOT}/tb/models/axi_datamover_read_model.sv"
    "${VERIFY_ROOT}/tb/checkers/axis_stream_protocol_checker.sv"
    "${VERIFY_ROOT}/tb/edgedrnn_tb.sv"
)

prepare_build_directory() {
    local build_dir="$1"
    case "${build_dir}" in
        "${BUILD_ROOT}"/*) ;;
        *)
            echo "Refusing to clean build directory outside ${BUILD_ROOT}: ${build_dir}" >&2
            return 2
            ;;
    esac

    "${PYTHON:-python3}" "${SCRIPT_DIR}/validate_stimuli.py" "${STIMULI_DIR}"
    mkdir -p "${build_dir}"
    find "${build_dir}" -mindepth 1 -delete
    cp "${STIMULI_DIR}/edgedrnn_tb_creg.txt" "${build_dir}/"
    cp "${STIMULI_DIR}/edgedrnn_tb_gold_rnn.txt" "${build_dir}/"
    cp "${STIMULI_DIR}/edgedrnn_tb_params.txt" "${build_dir}/"
    cp "${STIMULI_DIR}/edgedrnn_tb_stim.txt" "${build_dir}/"
}
