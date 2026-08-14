/**
 * EdgeDRNN shared SystemVerilog definitions.
 *
 * This package is deliberately small: it contains only constants and pure
 * sizing helpers that are shared by more than one synthesizable module.  The
 * datapath itself remains parameterized inside each module.  Keeping the
 * register map here avoids the compilation-unit parameters traditionally
 * placed in `hdr_macros.v`, which were sensitive to source ordering and could
 * silently collide with names in an integrating design.
 */
package edgedrnn_pkg;
    timeunit 1ns;
    timeprecision 1ps;

    // Configuration-register map used by software, the testbench, and RTL.
    // See Fig. 3 of the EdgeDRNN paper: these registers form the CFG block.
    localparam int unsigned CFG_IDX_W_BASEADDR = 0;  // DDR byte address of W.
    localparam int unsigned CFG_IDX_THX        = 1;  // Input-delta threshold.
    localparam int unsigned CFG_IDX_THH        = 2;  // Hidden-delta threshold.
    localparam int unsigned CFG_IDX_NUM_LAYER  = 3;  // Number of GRU layers.
    localparam int unsigned CFG_IDX_INPUT_SIZE = 4;  // Padded first-layer input.
    localparam int unsigned CFG_IDX_HIDDEN_BASE = 5; // Hidden sizes start here.

    // Xilinx AXI Datamover command layout used by edgedrnn.sv.
    localparam int unsigned DATAMOVER_CMD_WIDTH = 80;

    /** Return at least one address bit, including for a depth of one. */
    function automatic int unsigned clog2_safe(input int unsigned value);
        return (value <= 1) ? 1 : $clog2(value);
    endfunction

    /** True when value is a non-zero power of two. */
    function automatic bit is_power_of_two(input int unsigned value);
        return (value != 0) && ((value & (value - 1)) == 0);
    endfunction

    /** Round value up to the next integer multiple. */
    function automatic int unsigned round_up_to_multiple(
        input int unsigned value,
        input int unsigned multiple
    );
        return ((value + multiple - 1) / multiple) * multiple;
    endfunction

endpackage
