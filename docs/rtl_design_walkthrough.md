# EdgeDRNN RTL design walkthrough

This guide connects the accelerator described in the EdgeDRNN paper to the
implementation under `hdl/`. It is intended as a reading companion for someone
who understands basic synchronous logic but is new to sparse recurrent-neural-
network hardware.

## The idea in one page

A conventional GRU multiplies full input and hidden-state vectors by weight
matrices every timestep. DeltaGRU instead remembers reference vectors
`x_hat` and `h_hat` and forms differences:

```text
delta_x = x - x_hat
delta_h = h - h_hat
```

If the magnitude of an element is below its configured threshold, hardware
emits zero and leaves the reference unchanged. If it reaches the threshold,
hardware emits the delta and updates the reference to the new value. Most
elements therefore cause no memory request and no multiply-accumulate work.

For each emitted delta, the accelerator fetches one whole weight-matrix column
and adds `delta * column` into persistent partial sums. The four partial-sum
regions correspond to the reset gate, update gate, input candidate term, and
hidden candidate term. Once all delta-x and delta-h columns for a layer are
processed, the same arithmetic lanes calculate the GRU activations.

```text
packed input / previous-layer feedback
                  |
                  v
       +---------------------+
       | IEU                 |  threshold, update x_hat/h_hat,
       | input encoding unit |  generate {delta, column index}
       +----------+----------+
                  |
          paired value/index FIFOs
                  |
        +---------+----------+
        |                    |
        v                    v
  Datamover command      non-zero delta
        |                    |
        v                    |
 external INT8 weights -----+
                  |
                  v
       +---------------------+
       | HPC + HPE           |  sparse column MAC, then r/u/c/h
       | + ACC memory        |
       +----------+----------+
                  |
           hidden activation
            /           \
       feedback       final output
```

## Paper-to-code map

| Paper concept | RTL | What to look for |
| --- | --- | --- |
| Top-level architecture and data paths (Fig. 3) | `edgedrnn.sv` | Configuration registers, Datamover command assembly, IEU/HPC wiring, and the feedback/output stream fork |
| Delta unit and on-the-fly index generation (Eq. 2, Fig. 4) | `ieu.sv` | `IEU_BIAS`, `IEU_DX`, and `IEU_DH`; threshold comparison; conditional reference-memory write enables |
| Current and reference state storage | `mem_state.sv` | Separate `current` and `previous` BRAM banks for `h`, `x_hat`, and `h_hat` |
| Sparse matrix-vector control | `hpc.sv` | `HPC_MAC`, valid/last pipelines, accumulation-memory address mapping |
| Time-multiplexed processing elements (Fig. 5) | `hpe.sv` | The three-stage multiply/add pipeline and operand selectors `s0` through `s3` |
| Delta accumulation memories (Eq. 3, Fig. 6) | `mem_acc.sv` | One BRAM bank per PE and four logical gate regions per layer |
| Overlapped GRU activation schedule (Fig. 7) | `hpc.sv`, `hpe.sv` | `HPC_ACT0` through `HPC_ACT4`, sigmoid/tanh enables, and delayed activation operands |
| Quantized nonlinear functions | `sigmoid_lut.sv`, `tanh_lut.sv` | Exhaustive Q7.8 lookup tables and saturation behavior |
| Elastic delta buffering | `axis_fifo.sv` | Registered-output FIFO and preservation of TLAST under stalls |
| Shared register-map/sizing definitions | `edgedrnn_pkg.sv` | Configuration indices, command width, and elaboration helpers |

The three `.v` files are integration compatibility code. `edgedrnn_wrapper.v`
adapts the SystemVerilog top level to older Vivado IP projects;
`s_axi_lite_v1_0*.v` is the generated 32-register AXI-Lite bank. The numerical
datapath is in the `.sv` files above.

## One timestep through the hardware

### 1. Select input or feedback

The IEU handles every network layer with the same circuitry. For the first
layer, `IEU_DX` reads the external packed input stream. For later layers it
reads the previous layer's just-computed hidden state from current-state
memory. `IEU_FB` writes incoming activation feedback into that memory.

Layer sizes come from the configuration bank. An external input size is rounded
up to a `NUM_PE` boundary because one stream beat contains one element per PE
lane. Padding values are generated as zeros by the Python exporter.

### 2. Generate sparse value/index pairs

`IEU_BIAS` first emits the synthetic bias column. `IEU_DX` then compares input
elements with `x_hat`, and `IEU_DH` compares hidden elements with `h_hat`.
The subtract/threshold path is pipelined:

1. Select and register the current value, reference value, and global column
   index.
2. Subtract the reference and compare against `Theta_x` or `Theta_h`.
3. If non-zero, write the new reference value into exactly one BRAM lane and
   enqueue both the delta and its index.

The value and index FIFOs are coupled. A non-zero pair advances only if both
FIFOs can accept it. An all-zero vector still produces a TLAST marker on the
value side; the HPC uses that marker to finish the phase without requesting a
weight column.

### 3. Convert an index into a memory command

`edgedrnn.sv` turns each index into an 80-bit Xilinx AXI Datamover MM2S command.
With INT8 weights, one DeltaGRU column contains three `hidden_size`-byte regions
(reset, update, and candidate), so:

```text
BTT           = 3 * hidden_size
start_address = weight_base + column_index * BTT
```

The command and delta remain synchronized by ready/valid backpressure. The
portable testbench models this transaction without requiring Xilinx IP.

### 4. Accumulate a sparse column

In `HPC_MAC`, the HPC accepts a delta and weight beat together. The HPE
broadcasts the scalar delta to all lanes, sign-extends each lane's weight, and
performs:

```text
new_partial_sum[pe] = old_partial_sum[pe] + delta * weight[pe]
```

Read addresses, valid flags, TLAST flags, and write addresses are delayed to
match the three HPE pipeline registers. `sel_xh_rd` and `sel_xh_wr` distinguish
the delta-x and delta-h memory layouts.

### 5. Calculate the GRU activation

After both sparse phases, `HPC_ACT0` through `HPC_ACT4` repeatedly select the
four accumulation regions and reuse the HPE multiplier, adder, sigmoid LUT, and
tanh LUT. The five controller states represent ten overlapped pipeline phases
from paper Fig. 7. After filling the pipeline, one packed group of `NUM_PE`
hidden-state elements is emitted every schedule iteration.

The fixed-point assignments in `hpe.sv` deliberately preserve historical
two's-complement wrapping and rounding. Some expressions may look more verbose
than floating-point code because their expression width is part of the
software/RTL numerical contract.

### 6. Feed back and optionally publish

Every activation beat returns to the IEU so it can become input to the next
layer or state for the next timestep. Final-layer beats must also reach the
external output. The top level implements an atomic two-consumer fork with one
`done` bit per consumer.

The packet's final/non-final destination is latched at its first beat. This is
important: the IEU may accept the feedback TLAST and change its layer counter
while the external output is still stalled. A live layer comparison would
otherwise withdraw output TVALID before the handshake.

## Memory organization

`mem_state.sv` contains two bank groups per PE lane:

- **current state:** the most recent hidden activation for each layer;
- **previous/reference state:** two address halves used for `x_hat` and
  `h_hat`.

`mem_acc.sv` contains four logical regions per layer:

```text
offset 0 * groups : M_r
offset 1 * groups : M_u
offset 2 * groups : M_xc
offset 3 * groups : M_hc

groups = hidden_size / NUM_PE
```

The layer number forms the high BRAM address bits. The region/group address
forms the low bits. All PE banks receive the same address, so one cycle reads or
writes `NUM_PE` neuron rows.

Simulation builds define `SIM_DEBUG`, which clears inferred memories for a
deterministic test. Synthesis must not define it: a reset loop over every memory
element can prevent Vivado from inferring block RAM.

## Configuration registers

The shared indices are declared in `edgedrnn_pkg.sv` and written by
`python/edgedrnn/exporter.py`.

| Index | Meaning |
| ---: | --- |
| 0 | Byte base address of packed weights in external memory |
| 1 | Quantized input-delta threshold `Theta_x` |
| 2 | Quantized hidden-delta threshold `Theta_h` |
| 3 | Number of recurrent layers |
| 4 | PE-padded network input size |
| 5 onward | Hidden size for each recurrent layer |

In hardware, the AXI-Lite bank supplies these values. Under `SIM_DEBUG`, the
top level loads the exported `edgedrnn_tb_creg.txt` directly.

## Fixed-point formats and limitations

The checked default configuration uses:

- activation and hidden state: signed 16-bit Q7.8;
- weights: signed 8-bit Q0.7;
- accumulator: signed 24-bit product precision;
- sigmoid output: Q0.8 held in a 16-bit signed container;
- tanh output: signed Q7.8.

The LUT case entries are generated for Q7.8, and Datamover byte-address
arithmetic assumes packed INT8 weights. Elaboration assertions fail early if a
parameter set violates these assumptions instead of silently producing an
incorrect numerical format.

## Ready/valid rules used throughout

For any AXI stream:

1. A transfer occurs only on a rising edge with both `tvalid` and `tready` high.
2. A producer controls `tvalid`, `tdata`, and `tlast`; a consumer controls
   `tready`.
3. If `tvalid` is high and `tready` is low, the producer must hold the entire
   beat stable until it is accepted.
4. `tlast` belongs to the data beat and follows the same stability rule.

The testbench assertion checker applies rules 3 and 4 to input, command, weight,
and output streams in both supported simulators.

## Verification structure

```text
verification/rtl/
├── tb/
│   ├── interfaces/   shared AXI stream bundle and modports
│   ├── models/       file source and Datamover/DDR read model
│   ├── checkers/     portable stream protocol assertions
│   └── edgedrnn_tb.sv
├── stimuli/          exporter-owned vectors, manifest, and generated sizes
├── scripts/          xsim and Verilator compile/run entry points
└── build/            ignored, disposable simulator output
```

The scoreboard checks every packed output bit against the quantized PyTorch
golden file and also checks TLAST every
`EDGEDRNN_TB_RESPONSES_PER_TIMESTEP` beats. Deterministic bubbles are inserted
on weight and output streams to exercise stalls reproducibly. Success is
reported only after every expected response matches.

## Suggested reading order

1. `edgedrnn_pkg.sv` for shared names.
2. `edgedrnn.sv` for the dataflow and stream fork.
3. `ieu.sv`, followed by `mem_state.sv`, for temporal sparsity.
4. `hpc.sv`, `mem_acc.sv`, and `hpe.sv`, for sparse MAC and activation flow.
5. The two LUTs and `axis_fifo.sv` for leaf implementation details.
6. `verification/rtl/tb/edgedrnn_tb.sv` to see the complete transaction flow
   exercised and checked.
