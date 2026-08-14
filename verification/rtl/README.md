# RTL verification

This directory contains simulator-independent testbench code, generated model
stimuli, and thin build scripts. Synthesizable sources remain under `hdl/`.

## Layout

- `tb/interfaces/`: small AXI4-Stream interface and ownership modports.
- `tb/models/`: registered file source and AXI Datamover/DDR read model.
- `tb/checkers/`: portable ready/valid stability assertions.
- `tb/edgedrnn_tb.sv`: reset, backpressure, scoreboard, and timeout control.
- `stimuli/`: exporter-generated vectors, lookup tables, and size manifests.
- `scripts/`: isolated simulator build and run entry points.
- `build/`: ignored local simulator output.

Run both supported simulators from the repository root:

```bash
make rtl
```

Or run one simulator directly:

```bash
make rtl-lint
make rtl-xsim
make rtl-verilator
```

The scripts copy generated memories into an isolated build directory, compile
the same testbench sources, and require the `RTL_TEST_PASS` completion marker.
The scoreboard compares every output bit and TLAST boundary with the exported
PyTorch result. Deterministic weight/output stalls exercise backpressure, and
any protocol violation, mismatch, extra response, or timeout exits nonzero.

See [`../../docs/rtl_design_walkthrough.md`](../../docs/rtl_design_walkthrough.md)
for the paper-to-RTL map and a cycle-level explanation of the verification flow.
