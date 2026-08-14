# RTL verification

This directory contains simulator-independent testbench code, generated model
stimuli, and thin build scripts. Synthesizable sources remain under `hdl/`.

## Layout

- `tb/`: portable SystemVerilog testbench components shared by xsim and Verilator.
- `stimuli/`: exporter-generated vectors, lookup tables, and size manifests.
- `scripts/`: isolated simulator build and run entry points.
- `build/`: ignored local simulator output.

Run both supported simulators from the repository root:

```bash
make rtl
```

Or run one simulator directly:

```bash
make rtl-xsim
make rtl-verilator
```

The scripts copy generated memories into an isolated build directory, compile
the same testbench sources, and require the `RTL_TEST_PASS` completion marker.
Any mismatch or timeout exits nonzero.
